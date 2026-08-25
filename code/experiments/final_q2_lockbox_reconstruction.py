"""Panel-equivalent estimator reconstruction for the final Q2 lockbox."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import inspect
import json
import os
import platform
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

import joblib
import numpy as np
import pandas as pd

from features.index_sentiment import build_matched_index_features
from experiments.final_q2_lockbox_contract import (
    CandidateSpec,
    LockboxProtocol,
    load_lockbox_protocol,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
FIT_CUTOFF = pd.Timestamp("2025-07-01T00:00:00Z")
REFERENCE_END = pd.Timestamp("2026-04-01T00:00:00Z")
DEFAULT_PROTOCOL_PATH = CODE_ROOT / "configs" / "final_q2_lockbox_protocol.json"
DEFAULT_OUTPUT_ROOT = (
    CODE_ROOT / "experiments" / "cache" / "final_q2_lockbox" / "reconstructed_models"
)
LOOKBACK_DAYS = 180
PROBABILITY_COLUMNS = ("p_short", "p_flat", "p_long")
PROBABILITY_ATOL = 1e-6
NORMALIZATION_ATOL = 1e-7


@dataclass(frozen=True)
class ReconstructionSpec:
    stream: str
    arm: str
    model_name: str
    width_bps: int
    fit_start: pd.Timestamp
    fit_cutoff: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    reference_panel: Path
    reference_sha256: str
    probability_atol: float = PROBABILITY_ATOL
    mean_probability_atol: float = PROBABILITY_ATOL
    torch_threads: int = 16
    decision_taus: tuple[float, ...] = ()

    @property
    def fit_key(self) -> str:
        return f"{self.stream}:{self.arm}:{self.model_name}:w{self.width_bps}"


@dataclass(frozen=True)
class PanelEquivalenceAudit:
    rows: int
    max_probability_error: float
    mean_probability_error: float


@dataclass(frozen=True)
class ReconstructionResult:
    spec: ReconstructionSpec
    estimator_path: Path
    estimator_sha256: str
    rebuilt_panel_path: Path
    rebuilt_panel_sha256: str
    manifest_path: Path
    audit: PanelEquivalenceAudit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _panel_artifact(candidate: CandidateSpec, model_name: str) -> tuple[str, str]:
    role = f"{model_name}_forward_panel"
    matches = [artifact for artifact in candidate.artifacts if artifact.role == role]
    if len(matches) != 1:
        raise ValueError(
            f"{candidate.candidate_id} must bind one {role}; found {len(matches)}"
        )
    return matches[0].path, matches[0].sha256


def reconstruction_specs(
    protocol: LockboxProtocol, *, code_root: str | Path = CODE_ROOT
) -> tuple[ReconstructionSpec, ...]:
    root = Path(code_root)
    unique: dict[str, ReconstructionSpec] = {}
    for candidate in protocol.candidates:
        for member in candidate.members:
            path_text, expected_hash = _panel_artifact(candidate, member.model_name)
            path = root / path_text
            if not path.is_file():
                raise FileNotFoundError(path)
            if _sha256(path) != expected_hash:
                raise ValueError(f"reference panel hash changed: {path_text}")
            fit_key = f"{candidate.stream}:{candidate.arm}:{member.model_name}:w{member.width_bps}"
            tolerance = protocol.reconstruction_exceptions.get(
                fit_key, protocol.default_reconstruction_tolerance
            )
            decision_taus = tuple(
                sorted(
                    {
                        float(member.signal_tau)
                        if member.signal_tau is not None
                        else float(candidate.decision_tau)
                    }
                )
            ) if candidate.combiner != "soft_vote" else ()
            spec = ReconstructionSpec(
                stream=candidate.stream,
                arm=candidate.arm,
                model_name=member.model_name,
                width_bps=member.width_bps,
                fit_start=FIT_CUTOFF - pd.Timedelta(days=LOOKBACK_DAYS),
                fit_cutoff=FIT_CUTOFF,
                test_start=FIT_CUTOFF,
                test_end=REFERENCE_END,
                reference_panel=path,
                reference_sha256=expected_hash,
                probability_atol=tolerance.max_abs_probability_error,
                mean_probability_atol=tolerance.mean_abs_probability_error,
                torch_threads=tolerance.torch_threads,
                decision_taus=decision_taus,
            )
            existing = unique.get(spec.fit_key)
            if existing is not None:
                comparable = replace(existing, decision_taus=spec.decision_taus)
                comparable_new = replace(spec, decision_taus=existing.decision_taus)
                if replace(comparable, decision_taus=()) != replace(comparable_new, decision_taus=()):
                    raise ValueError(f"conflicting reconstruction identity: {spec.fit_key}")
                spec = replace(
                    existing,
                    decision_taus=tuple(
                        sorted(set(existing.decision_taus).union(spec.decision_taus))
                    ),
                )
            unique[spec.fit_key] = spec
    ordered = tuple(unique[key] for key in sorted(unique))
    if len(ordered) != 21:
        raise ValueError(f"frozen candidate set must require 21 unique fits, found {len(ordered)}")
    return ordered


def _normalized_probabilities(frame: pd.DataFrame, *, label: str) -> np.ndarray:
    missing = set(PROBABILITY_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} panel misses probability columns: {sorted(missing)}")
    values = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} probabilities must be finite")
    if (values < 0.0).any() or (values > 1.0).any():
        raise ValueError(f"{label} probabilities must lie in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=NORMALIZATION_ATOL):
        raise ValueError(f"{label} probabilities are not normalized")
    return values


def assert_panel_equivalence(
    reference: pd.DataFrame,
    rebuilt: pd.DataFrame,
    *,
    probability_atol: float = PROBABILITY_ATOL,
    mean_probability_atol: float = PROBABILITY_ATOL,
) -> PanelEquivalenceAudit:
    required = {"timestamp", "pred", *PROBABILITY_COLUMNS}
    for label, frame in (("reference", reference), ("rebuilt", rebuilt)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} panel misses columns: {sorted(missing)}")
    reference_time = pd.DatetimeIndex(pd.to_datetime(reference["timestamp"], utc=True))
    rebuilt_time = pd.DatetimeIndex(pd.to_datetime(rebuilt["timestamp"], utc=True))
    if not reference_time.equals(rebuilt_time):
        raise ValueError("reconstructed timestamp keys/order differ from reference")
    reference_class = reference["pred"].to_numpy(dtype=int)
    rebuilt_class = rebuilt["pred"].to_numpy(dtype=int)
    if not np.array_equal(reference_class, rebuilt_class):
        raise ValueError("reconstructed class labels differ from reference")
    reference_probability = _normalized_probabilities(reference, label="reference")
    rebuilt_probability = _normalized_probabilities(rebuilt, label="rebuilt")
    difference = np.abs(reference_probability - rebuilt_probability)
    maximum = float(difference.max()) if difference.size else 0.0
    mean = float(difference.mean()) if difference.size else 0.0
    if maximum > float(probability_atol):
        raise ValueError(
            f"reconstructed probability error {maximum:.12g} exceeds {probability_atol:.12g}"
        )
    if mean > float(mean_probability_atol):
        raise ValueError(
            f"reconstructed mean probability error {mean:.12g} exceeds "
            f"{mean_probability_atol:.12g}"
        )
    return PanelEquivalenceAudit(
        rows=len(reference),
        max_probability_error=maximum,
        mean_probability_error=mean,
    )


def assert_threshold_decision_equivalence(
    reference: pd.DataFrame,
    rebuilt: pd.DataFrame,
    *,
    tau: float,
) -> None:
    reference_decision = _threshold_decision(reference, tau=tau)
    rebuilt_decision = _threshold_decision(rebuilt, tau=tau)
    if not np.array_equal(reference_decision, rebuilt_decision):
        raise ValueError(f"reconstructed decision vector differs at tau={float(tau):.6g}")


def _threshold_decision(frame: pd.DataFrame, *, tau: float) -> np.ndarray:
    if not {"pred", "confidence"}.issubset(frame.columns):
        raise ValueError("panel misses decision columns")
    signal = frame["pred"].map({0: -1, 1: 0, 2: 1}).to_numpy(int)
    return np.where(frame["confidence"].to_numpy(float) >= float(tau), signal, 0)


def _union_decision(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    values = np.column_stack([first, second])
    active = (values != 0).sum(axis=1)
    net = values.sum(axis=1)
    return np.where(np.abs(net) == active, np.sign(net), 0).astype(int)


def assert_union_decision_equivalence(
    lstm_reference: pd.DataFrame,
    svm_reference: pd.DataFrame,
    lstm_rebuilt: pd.DataFrame,
    svm_rebuilt: pd.DataFrame,
    *,
    lstm_tau: float,
    svm_tau: float,
) -> None:
    frames = (lstm_reference, svm_reference, lstm_rebuilt, svm_rebuilt)
    timestamps = [
        pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
        for frame in frames
    ]
    if any(not timestamps[0].equals(current) for current in timestamps[1:]):
        raise ValueError("Qualified Union timestamp panels differ")
    reference = _union_decision(
        _threshold_decision(lstm_reference, tau=lstm_tau),
        _threshold_decision(svm_reference, tau=svm_tau),
    )
    rebuilt = _union_decision(
        _threshold_decision(lstm_rebuilt, tau=lstm_tau),
        _threshold_decision(svm_rebuilt, tau=svm_tau),
    )
    if not np.array_equal(reference, rebuilt):
        raise ValueError("Qualified Union decision vector differs from the frozen panel")


def _soft_vote_frame(panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    if not panels:
        raise ValueError("soft-vote requires model panels")
    ordered = [panels[key] for key in sorted(panels)]
    timestamps = [
        pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
        for frame in ordered
    ]
    if any(not timestamps[0].equals(current) for current in timestamps[1:]):
        raise ValueError("soft-vote timestamp panels differ")
    probabilities = np.mean(
        [_normalized_probabilities(frame, label="soft-vote member") for frame in ordered],
        axis=0,
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps[0],
            "pred": probabilities.argmax(axis=1).astype(int),
            "confidence": probabilities.max(axis=1),
        }
    )


def assert_soft_vote_decision_equivalence(
    reference_panels: Mapping[str, pd.DataFrame],
    rebuilt_panels: Mapping[str, pd.DataFrame],
    *,
    tau: float,
) -> None:
    if set(reference_panels) != set(rebuilt_panels):
        raise ValueError("soft-vote model membership differs")
    reference = _threshold_decision(_soft_vote_frame(reference_panels), tau=tau)
    rebuilt = _threshold_decision(_soft_vote_frame(rebuilt_panels), tau=tau)
    if not np.array_equal(reference, rebuilt):
        raise ValueError("soft-vote decision vector differs from the frozen panel")


def _aligned_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(getattr(model, "classes_", (0, 1, 2)), dtype=int)
    if raw.shape != (len(X), len(classes)):
        raise ValueError("model probability shape differs from its classes")
    output = np.zeros((len(X), 3), dtype=float)
    for source, label in enumerate(classes):
        if int(label) not in (0, 1, 2):
            raise ValueError("model returned a class outside short/flat/long")
        output[:, int(label)] = raw[:, source]
    row_sum = output.sum(axis=1, keepdims=True)
    return np.divide(
        output,
        row_sum,
        out=np.full_like(output, 1.0 / 3.0),
        where=row_sum > 0.0,
    )


def fit_panel_estimator(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable[[], Any],
    fit_start: pd.Timestamp,
    fit_cutoff: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    label_tail_trim: int,
    sample_weight: np.ndarray | pd.Series | None = None,
) -> tuple[Any, pd.DataFrame]:
    if label_tail_trim < 0:
        raise ValueError("label_tail_trim must be non-negative")
    if not X.index.equals(y.index):
        y = y.reindex(X.index)
    fit_mask = (X.index >= fit_start) & (X.index < fit_cutoff)
    test_mask = (X.index >= test_start) & (X.index < test_end)
    X_train, y_train = X.loc[fit_mask], y.loc[fit_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    if X_train.empty or X_test.empty:
        raise ValueError("reconstruction fit/test span is empty")
    if label_tail_trim:
        if len(X_train) <= label_tail_trim:
            raise ValueError("training span is too small after label-tail trim")
        X_train = X_train.iloc[:-label_tail_trim]
        y_train = y_train.iloc[:-label_tail_trim]
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)[fit_mask][:-label_tail_trim]
    elif sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)[fit_mask]
    if X_train.index.max() >= fit_cutoff:
        raise AssertionError("reconstruction fit crossed its frozen cutoff")
    if y_train.nunique() < 2:
        raise ValueError("reconstruction training span needs at least two classes")
    model = model_factory()
    if sample_weight is None:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, sample_weight=np.asarray(sample_weight, dtype=float))
    probability = _aligned_probability(model, X_test)
    prediction = probability.argmax(axis=1).astype(int)
    frame = pd.DataFrame(
        {
            "timestamp": X_test.index,
            "y_true": y_test.astype(int).to_numpy(),
            "pred": prediction,
            "confidence": probability.max(axis=1),
            "p_short": probability[:, 0],
            "p_flat": probability[:, 1],
            "p_long": probability[:, 2],
        }
    )
    return model, frame


def serialize_estimator(model: Any, path: str | Path) -> tuple[Path, str]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    joblib.dump(model, temporary)
    temporary.replace(output)
    return output, _sha256(output)


def load_serialized_estimator(path: str | Path, *, expected_sha256: str) -> Any:
    source = Path(path)
    if _sha256(source) != str(expected_sha256).lower():
        raise ValueError("serialized estimator hash changed")
    return joblib.load(source)


def _frame_hash(frame: pd.DataFrame) -> str:
    current = frame.copy()
    index_hash = pd.util.hash_pandas_object(current.index.to_series(), index=False)
    value_hash = pd.util.hash_pandas_object(current, index=False)
    digest = hashlib.sha256()
    digest.update(index_hash.to_numpy(dtype="uint64").tobytes())
    digest.update(value_hash.to_numpy(dtype="uint64").tobytes())
    digest.update(json.dumps(list(current.columns)).encode("utf-8"))
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, engine="pyarrow", index=False)
    os.replace(temporary, path)
    return path


def _finalize_reconstruction(
    spec: ReconstructionSpec,
    *,
    X: pd.DataFrame,
    model: Any,
    prediction: pd.DataFrame,
    output_root: str | Path,
    fit_callable: Callable[..., Any] = fit_panel_estimator,
) -> ReconstructionResult:
    import torch

    if _sha256(spec.reference_panel) != spec.reference_sha256:
        raise ValueError("reference panel hash changed before reconstruction")
    reference = pd.read_parquet(spec.reference_panel)
    audit = assert_panel_equivalence(
        reference,
        prediction,
        probability_atol=spec.probability_atol,
        mean_probability_atol=spec.mean_probability_atol,
    )
    for tau in spec.decision_taus:
        assert_threshold_decision_equivalence(reference, prediction, tau=tau)
    root = (
        Path(output_root)
        / spec.stream
        / spec.arm
        / spec.model_name
        / f"w{spec.width_bps}"
    )
    estimator_path, estimator_sha256 = serialize_estimator(
        model, root / "estimator.joblib"
    )
    rebuilt_panel_path = _atomic_parquet(
        root / "rebuilt_reference_panel.parquet", prediction
    )
    rebuilt_panel_sha256 = _sha256(rebuilt_panel_path)
    dependency_lock = CODE_ROOT / "requirements-repro.txt"
    manifest_path = _atomic_json(
        root / "reconstruction_manifest.json",
        {
            "schema_version": "1.0",
            "fit_key": spec.fit_key,
            "stream": spec.stream,
            "arm": spec.arm,
            "model_name": spec.model_name,
            "width_bps": int(spec.width_bps),
            "fit_start": spec.fit_start.isoformat(),
            "fit_cutoff": spec.fit_cutoff.isoformat(),
            "test_start": spec.test_start.isoformat(),
            "test_end": spec.test_end.isoformat(),
            "feature_columns": list(X.columns),
            "feature_order_sha256": hashlib.sha256(
                json.dumps(list(X.columns), separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "reference_feature_matrix_sha256": _frame_hash(
                X.loc[(X.index >= spec.fit_start) & (X.index < spec.test_end)]
            ),
            "reference_panel": spec.reference_panel.as_posix(),
            "reference_panel_sha256": spec.reference_sha256,
            "serialized_estimator": estimator_path.as_posix(),
            "serialized_estimator_sha256": estimator_sha256,
            "rebuilt_panel": rebuilt_panel_path.as_posix(),
            "rebuilt_panel_sha256": rebuilt_panel_sha256,
            "rows": int(audit.rows),
            "max_probability_error": float(audit.max_probability_error),
            "mean_probability_error": float(audit.mean_probability_error),
            "probability_atol": spec.probability_atol,
            "mean_probability_atol": spec.mean_probability_atol,
            "max_probability_atol": spec.probability_atol,
            "normalization_atol": NORMALIZATION_ATOL,
            "decision_taus": list(spec.decision_taus),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "dependency_lock": dependency_lock.as_posix(),
            "dependency_lock_sha256": _sha256(dependency_lock),
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "fit_implementation_sha256": hashlib.sha256(
                inspect.getsource(fit_callable).encode("utf-8")
            ).hexdigest(),
            "dtype": "float64_input_or_model_native",
            "device": "cpu",
            "torch_version": str(torch.__version__),
            "torch_threads": int(spec.torch_threads),
            "torch_runtime_threads": int(torch.get_num_threads()),
            "torch_interop_threads": int(torch.get_num_interop_threads()),
            "cpu_model": os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor(),
            "thread_settings": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
            "q2_decoded": False,
        },
    )
    return ReconstructionResult(
        spec=spec,
        estimator_path=estimator_path,
        estimator_sha256=estimator_sha256,
        rebuilt_panel_path=rebuilt_panel_path,
        rebuilt_panel_sha256=rebuilt_panel_sha256,
        manifest_path=manifest_path,
        audit=audit,
    )


def reconstruct_from_frames(
    spec: ReconstructionSpec,
    *,
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable[[], Any],
    output_root: str | Path,
    label_tail_trim: int,
    sample_weight: np.ndarray | pd.Series | None = None,
    fit_callable: Callable[..., Any] = fit_panel_estimator,
) -> ReconstructionResult:
    model, prediction = fit_panel_estimator(
        X=X,
        y=y,
        model_factory=model_factory,
        fit_start=spec.fit_start,
        fit_cutoff=spec.fit_cutoff,
        test_start=spec.test_start,
        test_end=spec.test_end,
        label_tail_trim=label_tail_trim,
        sample_weight=sample_weight,
    )
    return _finalize_reconstruction(
        spec,
        X=X,
        model=model,
        prediction=prediction,
        output_root=output_root,
        fit_callable=fit_callable,
    )


def reconstruct_btc_spec(
    spec: ReconstructionSpec,
    *,
    prepared: Any,
    model_factory: Callable[[], Any],
    output_root: str | Path,
) -> ReconstructionResult:
    import torch

    from experiments.run_catboost_matched_ablation import (
        REGIMES,
        regime_balanced_training_weights,
    )

    torch.set_num_threads(int(spec.torch_threads))
    X, y = prepared.features[int(spec.width_bps)]
    regimes = prepared.regimes.reindex(X.index)
    known = regimes.isin(REGIMES)
    X, y, regimes = X.loc[known], y.reindex(X.index).loc[known], regimes.loc[known]
    train_mask = (X.index >= spec.fit_start) & (X.index < spec.fit_cutoff)
    test_mask = (X.index >= spec.test_start) & (X.index < spec.test_end)
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    train_regimes = regimes.loc[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    if len(X_train) <= 1 or X_test.empty:
        raise ValueError("BTC reconstruction span is too small")
    X_train, y_train, train_regimes = (
        X_train.iloc[:-1],
        y_train.iloc[:-1],
        train_regimes.iloc[:-1],
    )
    if X_train.index.max() >= spec.fit_cutoff or y_train.nunique() < 2:
        raise ValueError("BTC reconstruction fit boundary/classes changed")
    model = model_factory()
    weights = regime_balanced_training_weights(train_regimes)
    model.fit(X_train, y_train, sample_weight=weights.to_numpy(dtype=float))
    probability = _aligned_probability(model, X_test)
    prediction = pd.DataFrame(
        {
            "timestamp": X_test.index,
            "y_true": y_test.astype(int).to_numpy(),
            "pred": probability.argmax(axis=1).astype(int),
            "confidence": probability.max(axis=1),
            "p_short": probability[:, 0],
            "p_flat": probability[:, 1],
            "p_long": probability[:, 2],
        }
    )
    return _finalize_reconstruction(
        spec,
        X=X,
        model=model,
        prediction=prediction,
        output_root=output_root,
        fit_callable=reconstruct_btc_spec,
    )


def reconstruct_index_spec(
    spec: ReconstructionSpec,
    *,
    runner: Any,
    model_factory: Callable[[], Any],
    output_root: str | Path,
) -> ReconstructionResult:
    import torch

    torch.set_num_threads(int(spec.torch_threads))
    X, y = build_index_reconstruction_dataset(runner, spec)
    return reconstruct_from_frames(
        spec,
        X=X,
        y=y,
        model_factory=model_factory,
        output_root=output_root,
        label_tail_trim=1,
        fit_callable=reconstruct_index_spec,
    )


def build_index_reconstruction_dataset(
    runner: Any, spec: ReconstructionSpec
) -> tuple[pd.DataFrame, pd.Series]:
    if not hasattr(runner, "_price_and_vix_frames"):
        return runner.dataset(spec.arm, int(spec.width_bps))
    if spec.arm not in {"deberta_matched", "deepseek_matched"}:
        raise ValueError("final index reconstruction requires a matched sentiment arm")
    from experiments.index_replication import make_index_label

    _, price_vix = runner._price_and_vix_frames()
    scorer = "classic" if spec.arm == "deberta_matched" else "llm"
    sentiment = build_matched_index_features(
        runner.config.stream, runner.bars.index, scorer=scorer
    )
    X = pd.concat([price_vix, sentiment], axis=1).replace(
        [np.inf, -np.inf], np.nan
    )
    X = X.loc[price_vix.index].dropna().astype(float)
    y = make_index_label(runner.bars, int(spec.width_bps)).reindex(X.index)
    valid = y.ne(-1) & y.notna() & X.notna().all(axis=1)
    return X.loc[valid], y.loc[valid].astype(int)


def reconstruct_all(
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[ReconstructionResult, ...]:
    from experiments.all_model_sentiment_raw import prepare_arm
    from experiments.final_q2_lockbox_state import GLOBAL_SENTINEL_PATH
    from experiments.index_replication import IndexReplicationConfig, IndexReplicationRunner
    from models.zoo import MODELS

    if GLOBAL_SENTINEL_PATH.exists():
        raise PermissionError("panel reconstruction is pre-open work; OPENED.json exists")
    protocol = load_lockbox_protocol(protocol_path)
    specs = reconstruction_specs(protocol, code_root=CODE_ROOT)
    btc_prepared = prepare_arm("none")
    runner_scratch = Path(output_root).parent / "runner_scratch"
    index_runners = {
        stream: IndexReplicationRunner(
            IndexReplicationConfig.for_stream(stream, output_base=runner_scratch)
        )
        for stream in ("usa500", "usatech")
    }
    results: list[ReconstructionResult] = []
    for number, spec in enumerate(specs, start=1):
        print(f"[reconstruct {number:02d}/{len(specs)}] {spec.fit_key}", flush=True)
        factory = lambda name=spec.model_name: MODELS[name]({})
        if spec.stream == "btcusdt":
            result = reconstruct_btc_spec(
                spec,
                prepared=btc_prepared,
                model_factory=factory,
                output_root=output_root,
            )
        else:
            result = reconstruct_index_spec(
                spec,
                runner=index_runners[spec.stream],
                model_factory=factory,
                output_root=output_root,
            )
        results.append(result)
    decision_audit = validate_registered_candidate_decisions(protocol, results)
    summary_path = Path(output_root).parent / "reconstruction_summary.json"
    _atomic_json(
        summary_path,
        {
            "status": "complete",
            "q2_decoded": False,
            "estimators": len(results),
            "candidate_decision_mismatches": decision_audit,
            "fits": [
                {
                    "fit_key": result.spec.fit_key,
                    "estimator": result.estimator_path.as_posix(),
                    "estimator_sha256": result.estimator_sha256,
                    "manifest": result.manifest_path.as_posix(),
                    "manifest_sha256": _sha256(result.manifest_path),
                    "rows": result.audit.rows,
                    "max_probability_error": result.audit.max_probability_error,
                }
                for result in results
            ],
        },
    )
    return tuple(results)


def validate_registered_candidate_decisions(
    protocol: LockboxProtocol,
    results: list[ReconstructionResult] | tuple[ReconstructionResult, ...],
) -> dict[str, int]:
    indexed = {result.spec.fit_key: result for result in results}

    def panels(candidate: CandidateSpec):
        reference: dict[str, pd.DataFrame] = {}
        rebuilt: dict[str, pd.DataFrame] = {}
        for member in candidate.members:
            key = f"{candidate.stream}:{candidate.arm}:{member.model_name}:w{member.width_bps}"
            if key not in indexed:
                raise ValueError(f"candidate decision audit misses fit {key}")
            result = indexed[key]
            reference[member.model_name] = pd.read_parquet(result.spec.reference_panel)
            rebuilt[member.model_name] = pd.read_parquet(result.rebuilt_panel_path)
        return reference, rebuilt

    audit: dict[str, int] = {}
    for candidate in protocol.candidates:
        reference, rebuilt = panels(candidate)
        if candidate.combiner == "opposite_signal_veto":
            members = {member.model_name: member for member in candidate.members}
            assert_union_decision_equivalence(
                reference["lstm"],
                reference["svm_linear"],
                rebuilt["lstm"],
                rebuilt["svm_linear"],
                lstm_tau=float(members["lstm"].signal_tau),
                svm_tau=float(members["svm_linear"].signal_tau),
            )
        elif candidate.combiner == "soft_vote":
            assert_soft_vote_decision_equivalence(
                reference, rebuilt, tau=candidate.decision_tau
            )
        elif candidate.combiner is None:
            model = candidate.members[0].model_name
            assert_threshold_decision_equivalence(
                reference[model], rebuilt[model], tau=candidate.decision_tau
            )
        else:
            raise ValueError(f"unsupported candidate combiner: {candidate.combiner}")
        audit[candidate.candidate_id] = 0
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    results = reconstruct_all(
        protocol_path=args.protocol,
        output_root=args.output_root,
    )
    print(json.dumps({"status": "complete", "estimators": len(results)}, indent=2))
    return 0


__all__ = [
    "FIT_CUTOFF",
    "LOOKBACK_DAYS",
    "NORMALIZATION_ATOL",
    "PROBABILITY_ATOL",
    "PROBABILITY_COLUMNS",
    "PanelEquivalenceAudit",
    "ReconstructionResult",
    "ReconstructionSpec",
    "assert_panel_equivalence",
    "assert_soft_vote_decision_equivalence",
    "assert_threshold_decision_equivalence",
    "assert_union_decision_equivalence",
    "build_index_reconstruction_dataset",
    "fit_panel_estimator",
    "load_serialized_estimator",
    "reconstruction_specs",
    "reconstruct_btc_spec",
    "reconstruct_from_frames",
    "reconstruct_index_spec",
    "validate_registered_candidate_decisions",
    "reconstruct_all",
    "serialize_estimator",
]


if __name__ == "__main__":
    raise SystemExit(main())
