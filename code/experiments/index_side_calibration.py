"""OOF side-specific probability calibration for the two index streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from experiments.index_replication import (
    ARMS,
    STREAM_CONFIG,
    IndexReplicationConfig,
    IndexReplicationRunner,
    _frame_hash,
    build_selection_folds,
    simulate_one_bar,
)
from experiments.index_replication_protocol import (
    CALIBRATION_START,
    CUTOFF,
    FORWARD_END,
    FORWARD_START,
    MODEL_NAMES,
    SELECTION_END,
    SELECTION_START,
    TAUS,
    WIDTHS,
    daily_economics,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CODE_ROOT / "data"
SOURCE_CACHE_BASE = CODE_ROOT / "experiments" / "cache" / "index_replication"
CACHE_BASE = CODE_ROOT / "experiments" / "cache" / "index_side_calibration"
PROTOCOL_VERSION = "index-side-calibration-v1"
SIDE_THRESHOLDS = tuple(round(float(value), 2) for value in np.arange(0.25, 0.81, 0.05))
ARM_ORDER = {arm: rank for rank, arm in enumerate(ARMS)}
EVIDENCE_ROLE = "secondary_reused_forward_posthoc_diagnostic"

_SIDE_COLUMNS = {0: ("SHORT", "p_short"), 2: ("LONG", "p_long")}
_CALIBRATION_CLIP = 1e-6


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _canonical(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return _utc(value).isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list, np.ndarray, pd.Index)):
        return [_canonical(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _payload_hash(value: Any) -> str:
    payload = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_canonical(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, engine="pyarrow", index=False)
    os.replace(temporary, path)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    bounded = np.clip(np.asarray(value, dtype=float), -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-bounded))


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        np.asarray(probability, dtype=float),
        _CALIBRATION_CLIP,
        1.0 - _CALIBRATION_CLIP,
    )
    return np.log(clipped / (1.0 - clipped))


def _validated_prediction_frame(
    predictions: pd.DataFrame,
    *,
    probability_column: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    required = {"timestamp", "y_true", "pred", probability_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"prediction source misses columns: {sorted(missing)}")
    current = predictions.copy()
    current["timestamp"] = pd.to_datetime(current["timestamp"], utc=True)
    if current.empty:
        raise ValueError("prediction source is empty")
    if current["timestamp"].duplicated().any():
        raise ValueError("prediction source contains duplicate timestamps")
    if not current["timestamp"].between(start, end, inclusive="left").all():
        raise ValueError("calibration source must contain only 2024 OOF rows")
    probability = pd.to_numeric(current[probability_column], errors="coerce")
    if not np.isfinite(probability).all() or not probability.between(0.0, 1.0).all():
        raise ValueError("prediction probabilities must be finite values in [0, 1]")
    current[probability_column] = probability.astype(float)
    return current.sort_values("timestamp").reset_index(drop=True)


def fit_side_calibrator(
    predictions: pd.DataFrame,
    *,
    side_class: int,
    start: pd.Timestamp = SELECTION_START,
    end: pd.Timestamp = SELECTION_END,
) -> dict[str, object]:
    """Fit a deterministic one-dimensional sigmoid on blocked OOF rows."""
    if int(side_class) not in _SIDE_COLUMNS:
        raise ValueError("side_class must be 0 (SHORT) or 2 (LONG)")
    side, probability_column = _SIDE_COLUMNS[int(side_class)]
    start_utc, end_utc = _utc(start), _utc(end)
    current = _validated_prediction_frame(
        predictions,
        probability_column=probability_column,
        start=start_utc,
        end=end_utc,
    )
    target = current["y_true"].astype(int).eq(int(side_class)).astype(int)
    if target.nunique() != 2:
        raise ValueError("side calibration requires both binary classes")
    raw_probability = current[probability_column].to_numpy(dtype=float)
    design = _logit(raw_probability).reshape(-1, 1)
    estimator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        random_state=42,
        max_iter=2_000,
    )
    estimator.fit(design, target.to_numpy(dtype=int))
    coefficient = float(estimator.coef_[0, 0])
    intercept = float(estimator.intercept_[0])
    calibrated = _sigmoid(coefficient * design[:, 0] + intercept)
    identity = current[["timestamp", "y_true", "pred", probability_column]].copy()
    return {
        "side": side,
        "side_class": int(side_class),
        "probability_column": probability_column,
        "coefficient": coefficient,
        "intercept": intercept,
        "rows": int(len(current)),
        "positive_rows": int(target.sum()),
        "negative_rows": int(len(target) - target.sum()),
        "raw_brier": float(brier_score_loss(target, raw_probability)),
        "calibrated_brier": float(brier_score_loss(target, calibrated)),
        "selection_start": start_utc.isoformat(),
        "selection_end_exclusive": end_utc.isoformat(),
        "source_sha256": _frame_hash(identity),
    }


def _apply_one_calibrator(
    probability: pd.Series,
    calibrator: Mapping[str, object],
) -> np.ndarray:
    values = pd.to_numeric(probability, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("prediction probabilities must be finite values in [0, 1]")
    coefficient = float(calibrator["coefficient"])
    intercept = float(calibrator["intercept"])
    if not np.isfinite([coefficient, intercept]).all():
        raise ValueError("calibrator parameters must be finite")
    return _sigmoid(coefficient * _logit(values) + intercept)


def apply_side_calibrators(
    predictions: pd.DataFrame,
    short_calibrator: Mapping[str, object],
    long_calibrator: Mapping[str, object],
) -> pd.DataFrame:
    """Add calibrated SHORT and LONG probabilities without changing argmax."""
    required = {"timestamp", "pred", "p_short", "p_long"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"prediction source misses columns: {sorted(missing)}")
    if int(short_calibrator.get("side_class", -1)) != 0:
        raise ValueError("SHORT calibrator identity changed")
    if int(long_calibrator.get("side_class", -1)) != 2:
        raise ValueError("LONG calibrator identity changed")
    current = predictions.copy()
    current["timestamp"] = pd.to_datetime(current["timestamp"], utc=True)
    if current["timestamp"].duplicated().any():
        raise ValueError("prediction source contains duplicate timestamps")
    current["calibrated_short"] = _apply_one_calibrator(
        current["p_short"], short_calibrator
    )
    current["calibrated_long"] = _apply_one_calibrator(
        current["p_long"], long_calibrator
    )
    return current


def gate_side_predictions(
    predictions: pd.DataFrame,
    *,
    tau_short: float,
    tau_long: float,
) -> pd.DataFrame:
    """Preserve the model argmax, replacing directions below their gate by FLAT."""
    required = {"pred", "calibrated_short", "calibrated_long"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"calibrated predictions miss columns: {sorted(missing)}")
    if not (0.0 <= float(tau_short) <= 1.0 and 0.0 <= float(tau_long) <= 1.0):
        raise ValueError("side thresholds must be values in [0, 1]")
    current = predictions.copy()
    original = pd.to_numeric(current["pred"], errors="raise").astype(int)
    if not original.isin((0, 1, 2)).all():
        raise ValueError("pred must use SHORT=0, FLAT=1, LONG=2")
    short_probability = pd.to_numeric(current["calibrated_short"], errors="raise")
    long_probability = pd.to_numeric(current["calibrated_long"], errors="raise")
    admitted_short = original.eq(0) & short_probability.ge(float(tau_short))
    admitted_long = original.eq(2) & long_probability.ge(float(tau_long))
    current["pred"] = np.select(
        [admitted_short, admitted_long], [0, 2], default=1
    ).astype(int)
    current["confidence"] = np.select(
        [admitted_short, admitted_long],
        [short_probability, long_probability],
        default=0.0,
    ).astype(float)
    return current


def _structural_shortfall(frame: pd.DataFrame) -> pd.Series:
    return (
        (50 - frame["trades"].astype(int)).clip(lower=0)
        + (15 - frame["n_long"].astype(int)).clip(lower=0)
        + (15 - frame["n_short"].astype(int)).clip(lower=0)
        + (3 - frame["positive_months"].astype(int)).clip(lower=0)
    ).astype(int)


def _validate_candidate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "arm",
        "model_name",
        "width_bps",
        "tau_short",
        "tau_long",
        "trades",
        "n_long",
        "n_short",
        "positive_months",
        "net_return",
        "daily_sharpe",
        "daily_sortino",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"H1 candidate grid misses columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("H1 candidate grid is empty")
    current = frame.copy()
    numeric = [
        "width_bps",
        "tau_short",
        "tau_long",
        "trades",
        "n_long",
        "n_short",
        "positive_months",
        "net_return",
        "daily_sharpe",
        "daily_sortino",
    ]
    converted = current[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(converted.to_numpy(dtype=float)).all():
        raise ValueError("H1 candidate metrics must be finite")
    current[numeric] = converted
    return current


def select_arm_policy(candidates: pd.DataFrame) -> pd.Series:
    """Freeze one side-threshold pair for a single arm/model using H1 only."""
    current = _validate_candidate_metrics(candidates)
    if current[["arm", "model_name"]].drop_duplicates().shape[0] != 1:
        raise ValueError("arm selection requires exactly one arm/model identity")
    current["structural_shortfall"] = _structural_shortfall(current)
    current["h1_pass"] = (
        current["structural_shortfall"].eq(0)
        & current["net_return"].gt(0.0)
        & current["daily_sharpe"].gt(0.0)
        & current["daily_sortino"].gt(0.0)
    )
    if current["h1_pass"].any():
        ranked = current.loc[current["h1_pass"]].sort_values(
            [
                "trades",
                "daily_sortino",
                "net_return",
                "daily_sharpe",
                "tau_short",
                "tau_long",
            ],
            ascending=[False, False, False, False, True, True],
            kind="mergesort",
        )
        status = "Pass"
        rule = "pass,trades,sortino,net,sharpe,tau_short,tau_long"
    else:
        ranked = current.sort_values(
            [
                "structural_shortfall",
                "daily_sortino",
                "net_return",
                "trades",
                "tau_short",
                "tau_long",
            ],
            ascending=[True, False, False, False, True, True],
            kind="mergesort",
        )
        status = "Below gate"
        rule = "shortfall,sortino,net,trades,tau_short,tau_long"
    winner = ranked.iloc[0].copy()
    winner["h1_status"] = status
    winner["selection_rule"] = rule
    return winner.drop(labels="h1_pass")


def select_model_policy(arm_policies: pd.DataFrame) -> pd.Series:
    """Freeze one feature arm for a model using only its H1 arm winners."""
    required = {
        "arm",
        "model_name",
        "h1_status",
        "trades",
        "daily_sortino",
        "net_return",
        "daily_sharpe",
    }
    missing = required.difference(arm_policies.columns)
    if missing or arm_policies.empty:
        raise ValueError(f"model policy table misses columns: {sorted(missing)}")
    current = arm_policies.copy()
    if current["model_name"].nunique() != 1 or current["arm"].duplicated().any():
        raise ValueError("model selection requires unique arms for one model")
    if not current["arm"].isin(ARM_ORDER).all():
        raise ValueError("model selection contains an unregistered arm")
    if not current["h1_status"].isin(("Pass", "Below gate")).all():
        raise ValueError("model selection contains an invalid H1 status")
    current["__pass"] = current["h1_status"].eq("Pass").astype(int)
    current["__arm_order"] = current["arm"].map(ARM_ORDER).astype(int)
    ranked = current.sort_values(
        [
            "__pass",
            "trades",
            "daily_sortino",
            "net_return",
            "daily_sharpe",
            "__arm_order",
        ],
        ascending=[False, False, False, False, False, True],
        kind="mergesort",
    )
    winner = ranked.iloc[0].copy()
    winner["model_selection_rule"] = "pass,trades,sortino,net,sharpe,arm_order"
    return winner.drop(labels=["__pass", "__arm_order"])


def validate_selected_policies(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and stably order the frozen four-arm by nine-model source grid."""
    required = {
        "arm",
        "model_name",
        "width_bps",
        "tau",
        "eligible",
        "h1_execution_status",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"selected-policy source misses columns: {sorted(missing)}")
    expected = set(product(ARMS, MODEL_NAMES))
    identities = frame[["arm", "model_name"]]
    widths = pd.to_numeric(frame["width_bps"], errors="coerce")
    taus = pd.to_numeric(frame["tau"], errors="coerce")
    exact = (
        len(frame) == len(expected)
        and not identities.duplicated().any()
        and set(map(tuple, identities.to_numpy())) == expected
        and np.isfinite(widths).all()
        and np.isfinite(taus).all()
        and set(widths.astype(int)).issubset(set(WIDTHS))
        and set(taus.astype(float)).issubset({float(value) for value in TAUS})
        and pd.api.types.is_bool_dtype(frame["eligible"])
    )
    if not exact:
        raise ValueError("selected-policy source must be the exact 36-policy grid")
    expected_status = frame["eligible"].map(
        {True: "eligible", False: "diagnostic_only_no_eligible_policy"}
    )
    if not frame["h1_execution_status"].eq(expected_status).all():
        raise ValueError("selected-policy H1 execution status changed")
    current = frame.copy()
    current["__arm_order"] = current["arm"].map(ARM_ORDER)
    current["__model_order"] = current["model_name"].map(
        {name: rank for rank, name in enumerate(MODEL_NAMES)}
    )
    return (
        current.sort_values(["__arm_order", "__model_order"], kind="mergesort")
        .drop(columns=["__arm_order", "__model_order"])
        .reset_index(drop=True)
    )


class IndexPredictionSource:
    """Read the exact frozen replication predictions without fitting models."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.protocol_path = self.root / "protocol_manifest.json"
        self.selected_path = self.root / "h1_selected_policies.parquet"
        self.vix_path = self.root / "vix_admission.json"
        for path in (self.protocol_path, self.selected_path, self.vix_path):
            if not path.exists():
                raise FileNotFoundError(path)
        self.protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        self.vix = json.loads(self.vix_path.read_text(encoding="utf-8"))
        self._selected = validate_selected_policies(pd.read_parquet(self.selected_path))
        self._path_cache: dict[tuple[str, str, str, int], tuple[Path, ...]] = {}
        self._identity_cache: dict[str, object] | None = None
        self._fingerprint_runner: IndexReplicationRunner | None = None
        self._gate_fingerprint_cache: dict[tuple[str, int, int], str] = {}
        self._validate_protocol()

    def _validate_protocol(self) -> None:
        stored_hash = self.protocol.get("protocol_hash")
        body = {key: value for key, value in self.protocol.items() if key != "protocol_hash"}
        if stored_hash != _payload_hash(body):
            raise ValueError("source protocol hash changed")
        expected_ranges = {
            "selection": (SELECTION_START, SELECTION_END),
            "calibration": (CALIBRATION_START, FORWARD_START),
            "forward": (FORWARD_START, FORWARD_END),
        }
        for key, expected in expected_ranges.items():
            actual = self.protocol.get(key)
            if not isinstance(actual, list) or len(actual) != 2:
                raise ValueError(f"source protocol misses {key} chronology")
            if tuple(_utc(value) for value in actual) != expected:
                raise ValueError(f"source protocol changed the {key} chronology")
        if self.vix.get("protocol_hash") != stored_hash:
            raise ValueError("VIX decision is stale for the source protocol")
        if self.vix.get("selected_base") not in {"price", "price_vix"}:
            raise ValueError("VIX decision has no registered selected base")
        for key in ("source_max_index", "vix_source_max_index"):
            if _utc(self.protocol[key]) >= CUTOFF:
                raise PermissionError("source protocol crossed the Q2 boundary")

    def selected_policies(self) -> pd.DataFrame:
        return self._selected.copy()

    @staticmethod
    def _semantic_hash(path: Path) -> str:
        frame = pd.read_parquet(path)
        columns = [
            column
            for column in (
                "timestamp",
                "y_true",
                "pred",
                "confidence",
                "p_short",
                "p_flat",
                "p_long",
            )
            if column in frame
        ]
        return _frame_hash(frame[columns])

    @classmethod
    def _resolve_equivalent_group(
        cls,
        paths: list[Path],
        segment: str,
        *,
        expected_fingerprint: str | None = None,
    ) -> Path:
        if not paths:
            raise FileNotFoundError(f"missing prediction segment {segment}")
        hashes = {cls._semantic_hash(path) for path in paths}
        if len(hashes) == 1:
            return sorted(paths, key=lambda path: path.name)[0]
        if expected_fingerprint is not None:
            matches = []
            for path in paths:
                frame = pd.read_parquet(path, columns=["fingerprint"])
                fingerprints = frame["fingerprint"].astype(str).unique()
                if len(fingerprints) == 1 and fingerprints[0] == expected_fingerprint:
                    matches.append(path)
            if len(matches) == 1:
                return matches[0]
        raise ValueError(f"ambiguous non-equivalent prediction segment {segment}")

    def _gate_fingerprint(
        self, model_name: str, width_bps: int, fold_id: int
    ) -> str:
        key = (model_name, int(width_bps), int(fold_id))
        if key in self._gate_fingerprint_cache:
            return self._gate_fingerprint_cache[key]
        if self._fingerprint_runner is None:
            stream = self.root.name
            config = IndexReplicationConfig.for_stream(
                stream,
                data_dir=DATA_DIR,
                output_base=self.root.parent,
            )
            runner = IndexReplicationRunner.__new__(IndexReplicationRunner)
            runner.config = config
            runner.output_root = self.root
            runner.model_factories = {}
            runner.bars = runner._validate_bars(runner._read_grid(stream))
            runner.vix_bars = runner._validate_bars(runner._read_grid("volidx"))
            runner._feature_cache = {}
            runner._label_cache = {}
            self._fingerprint_runner = runner
        source_arm = str(self.vix["selected_base"])
        X, y = self._fingerprint_runner._selection_data(source_arm, int(width_bps))
        folds = build_selection_folds(X.index)
        fold = folds[int(fold_id)]
        fingerprint = self._fingerprint_runner._prediction_fingerprint(
            X=X,
            y=y,
            fold=fold,
            model_name=model_name,
            arm=source_arm,
            width_bps=int(width_bps),
            stage="gate",
        )
        self._gate_fingerprint_cache[key] = fingerprint
        return fingerprint

    def _prediction_paths(
        self, stage: str, arm: str, model_name: str, width_bps: int
    ) -> tuple[Path, ...]:
        key = (stage, arm, model_name, int(width_bps))
        if key in self._path_cache:
            return self._path_cache[key]
        if stage == "selection":
            source_arm = str(self.vix["selected_base"]) if arm == "selected_base" else arm
            source_stage = "gate" if arm == "selected_base" else "selection"
            expected_segments = tuple(f"fold{number}" for number in range(5))
        elif stage == "calibration":
            source_arm = arm
            source_stage = "calibration"
            expected_segments = tuple(f"2025_{month:02d}" for month in range(1, 7))
        elif stage == "forward":
            source_arm = arm
            source_stage = "forward"
            expected_segments = ("2025_07_to_2026_03",)
        else:
            raise ValueError("prediction stage must be selection, calibration or forward")
        directory = self.root / "predictions" / source_stage / source_arm / model_name
        resolved: list[Path] = []
        for segment in expected_segments:
            matches = list(directory.glob(f"w{int(width_bps)}_{segment}_*.parquet"))
            expected_fingerprint = None
            if stage == "selection" and arm == "selected_base":
                expected_fingerprint = self._gate_fingerprint(
                    model_name, int(width_bps), int(segment.removeprefix("fold"))
                )
            resolved.append(
                self._resolve_equivalent_group(
                    matches,
                    segment,
                    expected_fingerprint=expected_fingerprint,
                )
            )
        self._path_cache[key] = tuple(resolved)
        return self._path_cache[key]

    def identity(self) -> dict[str, object]:
        if self._identity_cache is not None:
            return dict(self._identity_cache)
        catalog: dict[str, str] = {}
        for row in self._selected.itertuples(index=False):
            for stage in ("selection", "calibration", "forward"):
                for path in self._prediction_paths(
                    stage, str(row.arm), str(row.model_name), int(row.width_bps)
                ):
                    catalog[path.relative_to(self.root).as_posix()] = _file_sha256(path)
        identity = {
            "source_protocol_hash": str(self.protocol["protocol_hash"]),
            "source_protocol_file_sha256": _file_sha256(self.protocol_path),
            "selected_policy_sha256": _file_sha256(self.selected_path),
            "vix_admission_sha256": _file_sha256(self.vix_path),
            "prediction_catalog_sha256": _payload_hash(catalog),
            "prediction_file_count": int(len(catalog)),
            "selection": [SELECTION_START, SELECTION_END],
            "calibration": [CALIBRATION_START, FORWARD_START],
            "forward": [FORWARD_START, FORWARD_END],
            "q2_loaded": False,
        }
        self._identity_cache = identity
        return dict(identity)

    def load_predictions(
        self, stage: str, arm: str, model_name: str, width_bps: int
    ) -> pd.DataFrame:
        paths = self._prediction_paths(stage, arm, model_name, int(width_bps))
        current = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        required = {
            "timestamp",
            "y_true",
            "pred",
            "confidence",
            "p_short",
            "p_flat",
            "p_long",
            "model_name",
            "width_bps",
            "fit_id",
        }
        missing = required.difference(current.columns)
        if missing:
            raise ValueError(f"prediction source misses columns: {sorted(missing)}")
        current["timestamp"] = pd.to_datetime(current["timestamp"], utc=True)
        if current.empty or current["timestamp"].duplicated().any():
            raise ValueError("prediction source is empty or has duplicate timestamps")
        ranges = {
            "selection": (SELECTION_START, SELECTION_END),
            "calibration": (CALIBRATION_START, FORWARD_START),
            "forward": (FORWARD_START, FORWARD_END),
        }
        start, end = ranges[stage]
        if not current["timestamp"].between(start, end, inclusive="left").all():
            raise PermissionError(f"{stage} prediction crossed its causal boundary")
        if (
            current["model_name"].astype(str).nunique() != 1
            or str(current["model_name"].iloc[0]) != model_name
            or current["width_bps"].astype(int).nunique() != 1
            or int(current["width_bps"].iloc[0]) != int(width_bps)
        ):
            raise ValueError("prediction identity changed")
        probabilities = current[["p_short", "p_flat", "p_long"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if (
            not np.isfinite(probabilities.to_numpy(dtype=float)).all()
            or not probabilities.ge(0.0).all().all()
            or not probabilities.le(1.0).all().all()
        ):
            raise ValueError("prediction probabilities changed")
        return current.sort_values("timestamp").reset_index(drop=True)


@dataclass(frozen=True)
class IndexSideCalibrationConfig:
    stream: str
    instrument: str
    cost_bps: float
    source_root: Path
    output_root: Path
    data_dir: Path
    end_exclusive: pd.Timestamp = CUTOFF

    def __post_init__(self) -> None:
        if self.stream not in STREAM_CONFIG:
            raise ValueError(f"stream must be one of {tuple(STREAM_CONFIG)}")
        expected_instrument, expected_cost = STREAM_CONFIG[self.stream]
        if self.instrument != expected_instrument or float(self.cost_bps) != expected_cost:
            raise ValueError("instrument and cost must match the frozen stream contract")
        if _utc(self.end_exclusive) != CUTOFF:
            raise PermissionError("side calibration boundary must remain 2026-04-01 UTC")
        if Path(self.source_root).name != self.stream:
            raise ValueError("source_root must end with the instrument stream name")
        if Path(self.output_root).name != self.stream:
            raise ValueError("output_root must end with the instrument stream name")

    @classmethod
    def for_stream(
        cls,
        stream: str,
        *,
        source_base: str | Path = SOURCE_CACHE_BASE,
        output_base: str | Path = CACHE_BASE,
        data_dir: str | Path = DATA_DIR,
        end_exclusive: str | pd.Timestamp = CUTOFF,
    ) -> "IndexSideCalibrationConfig":
        if stream not in STREAM_CONFIG:
            raise ValueError(f"stream must be one of {tuple(STREAM_CONFIG)}")
        instrument, cost = STREAM_CONFIG[stream]
        return cls(
            stream=stream,
            instrument=instrument,
            cost_bps=float(cost),
            source_root=Path(source_base) / stream,
            output_root=Path(output_base) / stream,
            data_dir=Path(data_dir),
            end_exclusive=_utc(end_exclusive),
        )


def _ratio(mean: float, scale: float) -> float:
    return float(mean / scale) if np.isfinite(scale) and scale > 0.0 else 0.0


def _fast_economics(
    ledger: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    stage_bar_count: int,
) -> dict[str, object]:
    """Match daily_economics without rebuilding a full per-bar Series per grid row."""
    start_utc, end_utc = _utc(start), _utc(end)
    calendar = pd.date_range(
        start_utc.normalize(), end_utc.normalize(), freq="1D", inclusive="left", tz="UTC"
    )
    current = ledger.copy()
    if len(current):
        current["entry_time"] = pd.to_datetime(current["entry_time"], utc=True)
        current = current.loc[
            current["entry_time"].between(start_utc, end_utc, inclusive="left")
        ]
        by_entry = current.groupby("entry_time")["net_return"].sum()
        daily = by_entry.resample("1D").sum().reindex(calendar, fill_value=0.0)
    else:
        by_entry = pd.Series(dtype=float)
        daily = pd.Series(0.0, index=calendar, dtype=float)
    mean = float(daily.mean()) if len(daily) else 0.0
    standard = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    downside = daily.loc[daily < 0.0]
    downside_scale = (
        float(np.sqrt(np.mean(np.square(downside.to_numpy(dtype=float)))))
        if len(downside)
        else 0.0
    )
    equity = (1.0 + daily).cumprod()
    drawdown = equity / equity.cummax() - 1.0 if len(equity) else pd.Series(dtype=float)
    trades = int(len(current))
    gross = float(current.get("gross_return", pd.Series(dtype=float)).sum())
    net = float(current.get("net_return", pd.Series(dtype=float)).sum())
    nonzero_entries = int(by_entry.ne(0.0).sum()) if len(by_entry) else 0
    return {
        "calendar_days": int(len(calendar)),
        "trades": trades,
        "n_long": int((current.get("side", pd.Series(dtype=float)) == 1).sum()),
        "n_short": int((current.get("side", pd.Series(dtype=float)) == -1).sum()),
        "trades_per_day": float(trades / len(calendar)) if len(calendar) else 0.0,
        "gross_return": gross,
        "cost_return": gross - net,
        "net_return": net,
        "net_bps_per_trade": float(net * 10_000.0 / trades) if trades else 0.0,
        "exposure": float(nonzero_entries / stage_bar_count) if stage_bar_count else 0.0,
        "daily_sharpe": _ratio(mean, standard) * np.sqrt(365.0),
        "daily_sortino": _ratio(mean, downside_scale) * np.sqrt(365.0),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
    }


class IndexSideCalibrationRunner:
    """Fit OOF side calibrators, freeze H1 gates, and replay diagnostic forward."""

    def __init__(
        self,
        config: IndexSideCalibrationConfig,
        *,
        source: Any | None = None,
        bars: pd.DataFrame | None = None,
    ) -> None:
        self.config = config
        self.output_root = Path(config.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.source = source if source is not None else IndexPredictionSource(config.source_root)
        self.bars = self._validate_bars(bars if bars is not None else self._read_bars())

    def _read_bars(self) -> pd.DataFrame:
        path = Path(self.config.data_dir) / f"{self.config.stream}_15min_2021_2026.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_parquet(
            path,
            filters=[("timestamp", "<", self.config.end_exclusive.to_pydatetime())],
        )

    @staticmethod
    def _validate_bars(frame: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close", "available_at", "complete_bar"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"M15 frame misses columns: {sorted(missing)}")
        current = frame.copy().sort_index()
        if current.index.tz is None:
            raise ValueError("M15 index must be timezone-aware")
        current.index = current.index.tz_convert("UTC")
        current["available_at"] = pd.to_datetime(current["available_at"], utc=True)
        current = current.loc[
            current["complete_bar"].astype(bool)
            & (current.index < CUTOFF)
            & current["available_at"].le(CUTOFF)
        ]
        if current.empty or current.index.max() >= CUTOFF:
            raise PermissionError("M15 source crossed or missed the sealed Q2 boundary")
        if not current.index.is_unique or not current.index.is_monotonic_increasing:
            raise ValueError("M15 index must be unique and sorted")
        return current

    def _protocol_payload(self, source_identity: Mapping[str, object]) -> dict[str, object]:
        body = {
            "protocol_version": PROTOCOL_VERSION,
            "config": asdict(self.config),
            "models": list(MODEL_NAMES),
            "arms": list(ARMS),
            "side_thresholds": list(SIDE_THRESHOLDS),
            "selection": [SELECTION_START, SELECTION_END],
            "calibration": [CALIBRATION_START, FORWARD_START],
            "forward": [FORWARD_START, FORWARD_END],
            "q2_cutoff_exclusive": CUTOFF,
            "q2_loaded": False,
            "cost_bps": float(self.config.cost_bps),
            "execution": "next consecutive M15 open to same-bar close",
            "calibration_method": "separate unweighted logit-sigmoid for SHORT and LONG",
            "h1_selection": "pass,trades,sortino,net,sharpe; structural fallback",
            "evidence_role": EVIDENCE_ROLE,
            "source_identity": dict(source_identity),
            "source_identity_sha256": _payload_hash(source_identity),
            "implementation_sha256": _file_sha256(Path(__file__)),
        }
        return {**body, "protocol_hash": _payload_hash(body)}

    def _write_or_validate_protocol(
        self, source_identity: Mapping[str, object]
    ) -> dict[str, object]:
        payload = self._protocol_payload(source_identity)
        path = self.output_root / "protocol.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != _canonical(payload):
                raise ValueError("stale side-calibration protocol")
        else:
            _atomic_json(payload, path)
        return payload

    def _state(self, stage: str, **detail: object) -> None:
        _atomic_json(
            {
                "stage": stage,
                "stream": self.config.stream,
                "q2_loaded": False,
                **detail,
            },
            self.output_root / "run_state.json",
        )

    def _dense_events(
        self,
        calibrated: pd.DataFrame,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        all_directions = gate_side_predictions(
            calibrated, tau_short=0.0, tau_long=0.0
        )
        ledger, _ = simulate_one_bar(
            self.bars,
            all_directions,
            start=start,
            end=end,
            tau=0.0,
            cost_bps=self.config.cost_bps,
        )
        if ledger.empty:
            ledger["calibrated_probability"] = pd.Series(dtype=float)
            return ledger
        lookup = calibrated.copy()
        lookup["timestamp"] = pd.to_datetime(lookup["timestamp"], utc=True)
        lookup = lookup.set_index("timestamp")
        signal = pd.to_datetime(ledger["signal_bar_open"], utc=True)
        probabilities = np.where(
            ledger["side"].to_numpy(dtype=int) == -1,
            lookup.loc[signal, "calibrated_short"].to_numpy(dtype=float),
            lookup.loc[signal, "calibrated_long"].to_numpy(dtype=float),
        )
        ledger["calibrated_probability"] = probabilities
        entries = pd.to_datetime(ledger["entry_time"], utc=True)
        exits = pd.to_datetime(ledger["exit_time"], utc=True)
        if len(ledger) > 1 and (entries.iloc[1:].to_numpy() < exits.iloc[:-1].to_numpy()).any():
            raise AssertionError("one-bar event optimization encountered overlapping trades")
        return ledger.reset_index(drop=True)

    @staticmethod
    def _filter_events(
        events: pd.DataFrame, *, tau_short: float, tau_long: float
    ) -> pd.DataFrame:
        probability = pd.to_numeric(events["calibrated_probability"], errors="raise")
        keep = (
            events["side"].eq(-1) & probability.ge(float(tau_short))
        ) | (
            events["side"].eq(1) & probability.ge(float(tau_long))
        )
        return events.loc[keep].drop(columns="calibrated_probability").reset_index(drop=True)

    def _per_bar(
        self, ledger: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        index = self.bars.index[
            self.bars.index.to_series().between(start, end, inclusive="left").to_numpy()
        ]
        per_bar = pd.Series(0.0, index=index, name="net_return")
        if len(ledger):
            by_entry = ledger.assign(
                __entry=pd.to_datetime(ledger["entry_time"], utc=True)
            ).groupby("__entry")["net_return"].sum()
            common = per_bar.index.intersection(by_entry.index)
            per_bar.loc[common] = by_entry.reindex(common).to_numpy(dtype=float)
        return per_bar

    def _assert_canonical_parity(
        self,
        calibrated: pd.DataFrame,
        filtered: pd.DataFrame,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tau_short: float,
        tau_long: float,
    ) -> pd.Series:
        gated = gate_side_predictions(
            calibrated, tau_short=tau_short, tau_long=tau_long
        )
        expected_ledger, expected_per_bar = simulate_one_bar(
            self.bars,
            gated,
            start=start,
            end=end,
            tau=0.0,
            cost_bps=self.config.cost_bps,
        )
        if filtered.empty and expected_ledger.empty:
            if list(filtered.columns) != list(expected_ledger.columns):
                raise AssertionError("empty ledger schemas diverged")
        else:
            pd.testing.assert_frame_equal(
                filtered.reset_index(drop=True),
                expected_ledger.reset_index(drop=True),
                check_dtype=True,
                check_exact=True,
            )
        actual_per_bar = self._per_bar(filtered, start=start, end=end)
        pd.testing.assert_series_equal(actual_per_bar, expected_per_bar, check_exact=True)
        return actual_per_bar

    def _h1_grid(
        self,
        calibrated: pd.DataFrame,
        *,
        arm: str,
        model_name: str,
        width_bps: int,
        short_hash: str,
        long_hash: str,
    ) -> pd.DataFrame:
        events = self._dense_events(
            calibrated, start=CALIBRATION_START, end=FORWARD_START
        )
        stage_bar_count = int(
            self.bars.index.to_series()
            .between(CALIBRATION_START, FORWARD_START, inclusive="left")
            .sum()
        )
        month_edges = pd.date_range(CALIBRATION_START, FORWARD_START, freq="MS")
        rows: list[dict[str, object]] = []
        for tau_short, tau_long in product(SIDE_THRESHOLDS, repeat=2):
            ledger = self._filter_events(
                events, tau_short=tau_short, tau_long=tau_long
            )
            economics = _fast_economics(
                ledger,
                start=CALIBRATION_START,
                end=FORWARD_START,
                stage_bar_count=stage_bar_count,
            )
            if len(ledger):
                entry = pd.to_datetime(ledger["entry_time"], utc=True)
                positive_months = sum(
                    float(
                        ledger.loc[entry.between(left, right, inclusive="left"), "net_return"].sum()
                    )
                    > 0.0
                    for left, right in zip(month_edges[:-1], month_edges[1:])
                )
            else:
                positive_months = 0
            rows.append(
                {
                    "stream": self.config.stream,
                    "arm": arm,
                    "model_name": model_name,
                    "width_bps": int(width_bps),
                    "tau_short": float(tau_short),
                    "tau_long": float(tau_long),
                    "positive_months": int(positive_months),
                    "short_calibrator_sha256": short_hash,
                    "long_calibrator_sha256": long_hash,
                    **economics,
                }
            )
        grid = pd.DataFrame(rows)
        winner = select_arm_policy(grid)
        selected = self._filter_events(
            events,
            tau_short=float(winner["tau_short"]),
            tau_long=float(winner["tau_long"]),
        )
        per_bar = self._assert_canonical_parity(
            calibrated,
            selected,
            start=CALIBRATION_START,
            end=FORWARD_START,
            tau_short=float(winner["tau_short"]),
            tau_long=float(winner["tau_long"]),
        )
        canonical = daily_economics(
            selected, per_bar, start=CALIBRATION_START, end=FORWARD_START
        )
        for key, value in canonical.items():
            if not np.isclose(float(winner[key]), float(value), rtol=1e-12, atol=1e-12):
                raise AssertionError(f"fast H1 economics diverged for {key}")
        return grid

    @staticmethod
    def _calibrator_hash(record: Mapping[str, object]) -> str:
        return _payload_hash(record)

    @staticmethod
    def _stem(arm: str, model_name: str) -> str:
        return f"{arm}__{model_name}"

    def _expected_artifacts(self) -> list[Path]:
        checkpoints = [
            self.output_root / "forward" / f"{self._stem(arm, model)}.json"
            for arm, model in product(ARMS, MODEL_NAMES)
        ]
        ledgers = [
            self.output_root
            / "forward_ledgers"
            / f"{self._stem(arm, model)}{suffix}.parquet"
            for arm, model in product(ARMS, MODEL_NAMES)
            for suffix in ("", "_per_bar")
        ]
        return [
            self.output_root / "protocol.json",
            self.output_root / "calibrators.parquet",
            self.output_root / "h1_candidates.parquet",
            self.output_root / "h1_arm_policies.parquet",
            self.output_root / "h1_model_policies.parquet",
            self.output_root / "forward_arm_summary.parquet",
            self.output_root / "forward_summary.parquet",
            *checkpoints,
            *ledgers,
        ]

    def _write_manifest(self, protocol: Mapping[str, object]) -> None:
        expected_checkpoints = {
            path for path in self._expected_artifacts() if path.parent.name == "forward"
        }
        expected_ledgers = {
            path
            for path in self._expected_artifacts()
            if path.parent.name == "forward_ledgers"
        }
        if set((self.output_root / "forward").glob("*.json")) != expected_checkpoints:
            raise ValueError("side-calibration manifest requires exactly 36 checkpoints")
        if set((self.output_root / "forward_ledgers").glob("*.parquet")) != expected_ledgers:
            raise ValueError("side-calibration manifest requires exactly 72 ledger artifacts")
        missing = [path for path in self._expected_artifacts() if not path.exists()]
        if missing:
            raise FileNotFoundError(missing[0])
        artifacts = {
            path.relative_to(self.output_root).as_posix(): _file_sha256(path)
            for path in self._expected_artifacts()
        }
        _atomic_json(
            {
                "protocol_hash": protocol["protocol_hash"],
                "source_identity_sha256": protocol["source_identity_sha256"],
                "q2_loaded": False,
                "artifacts": artifacts,
            },
            self.output_root / "manifest.json",
        )

    def _resume_if_complete(
        self, protocol: Mapping[str, object]
    ) -> dict[str, object] | None:
        manifest_path = self.output_root / "manifest.json"
        if not manifest_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("protocol_hash") != protocol["protocol_hash"]
            or manifest.get("source_identity_sha256") != protocol["source_identity_sha256"]
            or manifest.get("q2_loaded") is not False
        ):
            raise ValueError("stale side-calibration manifest")
        expected = {
            path.relative_to(self.output_root).as_posix(): path
            for path in self._expected_artifacts()
        }
        if set(manifest.get("artifacts", {})) != set(expected):
            raise ValueError("side-calibration manifest artifact set changed")
        for relative, path in expected.items():
            if (
                not path.exists()
                or manifest["artifacts"].get(relative) != _file_sha256(path)
            ):
                raise ValueError(f"artifact hash changed for {relative}")
        expected_rows = {
            "calibrators.parquet": 72,
            "h1_candidates.parquet": 36 * len(SIDE_THRESHOLDS) ** 2,
            "h1_arm_policies.parquet": 36,
            "h1_model_policies.parquet": 9,
            "forward_arm_summary.parquet": 36,
            "forward_summary.parquet": 9,
        }
        for name, rows in expected_rows.items():
            frame = pd.read_parquet(self.output_root / name)
            if len(frame) != rows or frame.isna().any().any():
                raise ValueError(f"resume table changed for {name}")
            numeric = frame.select_dtypes(include="number")
            if not np.isfinite(numeric.to_numpy(dtype=float)).all():
                raise ValueError(f"resume metrics changed for {name}")
        result_path = self.output_root / "result.json"
        if not result_path.exists():
            raise FileNotFoundError(result_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("q2_loaded") is not False or _utc(result["max_prediction_timestamp"]) >= CUTOFF:
            raise PermissionError("resume result crossed the Q2 boundary")
        result["resumed_forward_policies"] = 36
        _atomic_json(result, result_path)
        self._state("complete", **result)
        return result

    def run(self) -> dict[str, object]:
        source_identity = self.source.identity()
        if source_identity.get("q2_loaded") is not False:
            raise PermissionError("source identity reports Q2 as loaded")
        selected = validate_selected_policies(self.source.selected_policies())
        protocol = self._write_or_validate_protocol(source_identity)
        resumed = self._resume_if_complete(protocol)
        if resumed is not None:
            return resumed

        calibrator_rows: list[dict[str, object]] = []
        candidate_frames: list[pd.DataFrame] = []
        arm_policy_rows: list[dict[str, object]] = []
        calibrators: dict[tuple[str, str, int], dict[str, object]] = {}
        source_policies = {
            (str(row.arm), str(row.model_name)): row._asdict()
            for row in selected.itertuples(index=False)
        }
        for number, policy in enumerate(selected.itertuples(index=False), start=1):
            arm = str(policy.arm)
            model_name = str(policy.model_name)
            width_bps = int(policy.width_bps)
            oof = self.source.load_predictions("selection", arm, model_name, width_bps)
            for side_class in (0, 2):
                record = fit_side_calibrator(oof, side_class=side_class)
                record.update(
                    {
                        "stream": self.config.stream,
                        "arm": arm,
                        "model_name": model_name,
                        "width_bps": width_bps,
                    }
                )
                record["calibrator_sha256"] = self._calibrator_hash(record)
                calibrators[(arm, model_name, side_class)] = record
                calibrator_rows.append(record)
            h1 = self.source.load_predictions("calibration", arm, model_name, width_bps)
            short = calibrators[(arm, model_name, 0)]
            long = calibrators[(arm, model_name, 2)]
            calibrated = apply_side_calibrators(h1, short, long)
            grid = self._h1_grid(
                calibrated,
                arm=arm,
                model_name=model_name,
                width_bps=width_bps,
                short_hash=str(short["calibrator_sha256"]),
                long_hash=str(long["calibrator_sha256"]),
            )
            candidate_frames.append(grid)
            arm_policy_rows.append(select_arm_policy(grid).to_dict())
            self._state("h1", completed=number, total=36, arm=arm, model_name=model_name)
            print(
                f"[{self.config.stream}] side H1 {number}/36: {arm}/{model_name}",
                flush=True,
            )

        calibrator_table = pd.DataFrame(calibrator_rows)
        candidate_table = pd.concat(candidate_frames, ignore_index=True)
        arm_policies = pd.DataFrame(arm_policy_rows)
        model_policies = pd.DataFrame(
            [
                select_model_policy(
                    arm_policies.loc[arm_policies["model_name"].eq(model_name)]
                ).to_dict()
                for model_name in MODEL_NAMES
            ]
        )
        _atomic_parquet(calibrator_table, self.output_root / "calibrators.parquet")
        _atomic_parquet(candidate_table, self.output_root / "h1_candidates.parquet")
        _atomic_parquet(arm_policies, self.output_root / "h1_arm_policies.parquet")
        _atomic_parquet(model_policies, self.output_root / "h1_model_policies.parquet")

        forward_rows: list[dict[str, object]] = []
        prediction_maxima: list[pd.Timestamp] = []
        for number, policy in enumerate(arm_policies.itertuples(index=False), start=1):
            arm = str(policy.arm)
            model_name = str(policy.model_name)
            width_bps = int(policy.width_bps)
            tau_short = float(policy.tau_short)
            tau_long = float(policy.tau_long)
            source_policy = source_policies[(arm, model_name)]
            raw = self.source.load_predictions("forward", arm, model_name, width_bps)
            prediction_max = pd.to_datetime(raw["timestamp"], utc=True).max()
            if not (FORWARD_START <= prediction_max < FORWARD_END):
                raise PermissionError("forward prediction crossed the Q2 boundary")
            prediction_maxima.append(prediction_max)
            short = calibrators[(arm, model_name, 0)]
            long = calibrators[(arm, model_name, 2)]
            calibrated = apply_side_calibrators(raw, short, long)
            gated = gate_side_predictions(
                calibrated, tau_short=tau_short, tau_long=tau_long
            )
            ledger, per_bar = simulate_one_bar(
                self.bars,
                gated,
                start=FORWARD_START,
                end=FORWARD_END,
                tau=0.0,
                cost_bps=self.config.cost_bps,
            )
            economics = daily_economics(
                ledger, per_bar, start=FORWARD_START, end=FORWARD_END
            )
            control_ledger, control_per_bar = simulate_one_bar(
                self.bars,
                raw,
                start=FORWARD_START,
                end=FORWARD_END,
                tau=float(source_policy["tau"]),
                cost_bps=self.config.cost_bps,
            )
            control = daily_economics(
                control_ledger,
                control_per_bar,
                start=FORWARD_START,
                end=FORWARD_END,
            )
            row = {
                "stream": self.config.stream,
                "arm": arm,
                "model_name": model_name,
                "width_bps": width_bps,
                "tau_short": tau_short,
                "tau_long": tau_long,
                "source_tau": float(source_policy["tau"]),
                "h1_status": str(policy.h1_status),
                "h1_trades": int(policy.trades),
                "h1_n_long": int(policy.n_long),
                "h1_n_short": int(policy.n_short),
                "h1_net_return": float(policy.net_return),
                "h1_daily_sharpe": float(policy.daily_sharpe),
                "h1_daily_sortino": float(policy.daily_sortino),
                "short_calibrator_sha256": str(short["calibrator_sha256"]),
                "long_calibrator_sha256": str(long["calibrator_sha256"]),
                "evidence_role": EVIDENCE_ROLE,
                "status": "traded" if economics["trades"] else "no_trades",
                **economics,
                "control_trades": int(control["trades"]),
                "control_n_long": int(control["n_long"]),
                "control_n_short": int(control["n_short"]),
                "control_net_return": float(control["net_return"]),
                "control_daily_sharpe": float(control["daily_sharpe"]),
                "control_daily_sortino": float(control["daily_sortino"]),
                "delta_trades": int(economics["trades"] - control["trades"]),
                "delta_long": int(economics["n_long"] - control["n_long"]),
                "delta_short": int(economics["n_short"] - control["n_short"]),
                "delta_net_return": float(economics["net_return"] - control["net_return"]),
                "delta_daily_sharpe": float(
                    economics["daily_sharpe"] - control["daily_sharpe"]
                ),
                "delta_daily_sortino": float(
                    economics["daily_sortino"] - control["daily_sortino"]
                ),
                "promising": bool(
                    economics["n_long"] > 0
                    and economics["n_short"] > 0
                    and economics["trades"] > control["trades"]
                    and economics["net_return"] >= 0.0
                    and economics["daily_sharpe"] >= 0.0
                    and economics["daily_sortino"] >= 0.0
                ),
                "fit_id": str(raw["fit_id"].iloc[0]),
                "prediction_max_timestamp": prediction_max,
            }
            forward_rows.append(row)
            stem = self._stem(arm, model_name)
            ledger_path = self.output_root / "forward_ledgers" / f"{stem}.parquet"
            per_bar_path = self.output_root / "forward_ledgers" / f"{stem}_per_bar.parquet"
            _atomic_parquet(ledger, ledger_path)
            _atomic_parquet(
                per_bar.rename("net_return").rename_axis("timestamp").reset_index(),
                per_bar_path,
            )
            _atomic_json(
                {
                    "protocol_hash": protocol["protocol_hash"],
                    "policy": {
                        "arm": arm,
                        "model_name": model_name,
                        "width_bps": width_bps,
                        "tau_short": tau_short,
                        "tau_long": tau_long,
                    },
                    "summary_sha256": _payload_hash(row),
                    "artifacts": {
                        "ledger": {
                            "path": ledger_path.relative_to(self.output_root).as_posix(),
                            "sha256": _file_sha256(ledger_path),
                        },
                        "per_bar": {
                            "path": per_bar_path.relative_to(self.output_root).as_posix(),
                            "sha256": _file_sha256(per_bar_path),
                        },
                    },
                },
                self.output_root / "forward" / f"{stem}.json",
            )
            self._state("forward", completed=number, total=36, arm=arm, model_name=model_name)

        forward_arm = pd.DataFrame(forward_rows)
        model_forward_rows = []
        for policy in model_policies.itertuples(index=False):
            matched = forward_arm.loc[
                forward_arm["model_name"].eq(str(policy.model_name))
                & forward_arm["arm"].eq(str(policy.arm))
            ]
            if len(matched) != 1:
                raise AssertionError("model-level frozen arm does not map to one forward row")
            model_forward_rows.append(matched.iloc[0].to_dict())
        forward_summary = pd.DataFrame(model_forward_rows)
        _atomic_parquet(forward_arm, self.output_root / "forward_arm_summary.parquet")
        _atomic_parquet(forward_summary, self.output_root / "forward_summary.parquet")
        max_prediction = max(prediction_maxima)
        result = {
            "stream": self.config.stream,
            "evidence_role": EVIDENCE_ROLE,
            "calibrator_rows": int(len(calibrator_table)),
            "h1_candidate_rows": int(len(candidate_table)),
            "h1_arm_policy_rows": int(len(arm_policies)),
            "h1_model_policy_rows": int(len(model_policies)),
            "forward_arm_rows": int(len(forward_arm)),
            "forward_rows": int(len(forward_summary)),
            "resumed_forward_policies": 0,
            "max_prediction_timestamp": max_prediction,
            "q2_loaded": False,
        }
        _atomic_json(result, self.output_root / "result.json")
        self._state("complete", **result)
        self._write_manifest(protocol)
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", choices=tuple(STREAM_CONFIG), required=True)
    args = parser.parse_args(argv)
    result = IndexSideCalibrationRunner(
        IndexSideCalibrationConfig.for_stream(args.stream)
    ).run()
    print(json.dumps(_canonical(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_ORDER",
    "CACHE_BASE",
    "IndexSideCalibrationConfig",
    "IndexSideCalibrationRunner",
    "IndexPredictionSource",
    "MODEL_NAMES",
    "PROTOCOL_VERSION",
    "SIDE_THRESHOLDS",
    "apply_side_calibrators",
    "fit_side_calibrator",
    "gate_side_predictions",
    "select_arm_policy",
    "select_model_policy",
    "validate_selected_policies",
]
