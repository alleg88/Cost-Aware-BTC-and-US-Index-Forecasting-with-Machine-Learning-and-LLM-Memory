"""Causal all-nine-model ensembles for the two frozen index replications."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.index_all_model_forward import (
    _atomic_json,
    _atomic_parquet,
    _file_sha256,
    _payload_hash,
    _stress_economics,
    validate_selected_policies,
)
from experiments.index_replication import (
    ARMS,
    CACHE_BASE as SOURCE_CACHE_BASE,
    STREAM_CONFIG,
    IndexReplicationConfig,
    IndexReplicationRunner,
    _frame_hash,
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
    select_h1_policy,
)


PROBABILITY_COLUMNS = ("p_short", "p_flat", "p_long")
ENSEMBLE_VARIANTS = ("soft_vote", "directional_majority", "stack")
LABEL_AVAILABLE_LAG = pd.Timedelta(minutes=30)
CODE_ROOT = Path(__file__).resolve().parents[1]
CACHE_BASE = CODE_ROOT / "experiments" / "cache" / "index_all_model_ensemble"
PROTOCOL_VERSION = "index-all-model-ensemble-v1"
EVIDENCE_ROLE = "secondary_reused_forward_diagnostic"


@dataclass(frozen=True)
class IndexAllModelEnsembleConfig:
    """Immutable paths and lockbox boundary for one isolated index stream."""

    stream: str
    data_dir: Path
    source_root: Path
    output_root: Path
    end_exclusive: pd.Timestamp = CUTOFF

    def __post_init__(self) -> None:
        if self.stream not in STREAM_CONFIG:
            raise ValueError("stream must be usa500 or usatech")
        if Path(self.source_root).name != self.stream or Path(self.output_root).name != self.stream:
            raise ValueError("source_root and output_root must end with the stream")
        end = pd.Timestamp(self.end_exclusive)
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        if end != CUTOFF:
            raise PermissionError("ensemble boundary must remain 2026-04-01 UTC")

    @property
    def cost_bps(self) -> float:
        return float(STREAM_CONFIG[self.stream][1])

    @classmethod
    def for_stream(
        cls,
        stream: str,
        *,
        data_dir: str | Path = CODE_ROOT / "data",
        source_base: str | Path = SOURCE_CACHE_BASE,
        output_base: str | Path = CACHE_BASE,
    ) -> "IndexAllModelEnsembleConfig":
        return cls(
            stream=stream,
            data_dir=Path(data_dir),
            source_root=Path(source_base) / stream,
            output_root=Path(output_base) / stream,
        )


def validate_source_contract(config: IndexAllModelEnsembleConfig) -> dict[str, Any]:
    """Validate the frozen replication/VIX/control inputs without loading forward."""
    paths = {
        "protocol_manifest.json": Path(config.source_root) / "protocol_manifest.json",
        "vix_admission.json": Path(config.source_root) / "vix_admission.json",
        "vix_gate_paired_2024.parquet": Path(config.source_root) / "vix_gate_paired_2024.parquet",
        "h1_selected_policies.parquet": Path(config.source_root) / "h1_selected_policies.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"ensemble source contract misses files: {missing}")

    protocol = json.loads(paths["protocol_manifest.json"].read_text(encoding="utf-8"))
    protocol_hash = str(protocol.get("protocol_hash", ""))
    protocol_body = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    if not protocol_hash or _payload_hash(protocol_body) != protocol_hash:
        raise ValueError("source protocol hash changed")
    expected = {
        "model_names": list(MODEL_NAMES),
        "widths_bps": list(WIDTHS),
        "taus": list(TAUS),
        "arms": list(ARMS),
        "selection": [SELECTION_START.isoformat(), SELECTION_END.isoformat()],
        "calibration": [CALIBRATION_START.isoformat(), FORWARD_START.isoformat()],
        "forward": [FORWARD_START.isoformat(), FORWARD_END.isoformat()],
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"source protocol {key} changed")

    vix = json.loads(paths["vix_admission.json"].read_text(encoding="utf-8"))
    if (
        vix.get("selected_base") != "price_vix"
        or vix.get("admitted") is not True
        or vix.get("gate_complete") is not True
        or vix.get("frozen_before_sentiment") is not True
    ):
        raise ValueError("VIX must be admitted and frozen before the ensemble")
    if vix.get("stream") != config.stream or vix.get("protocol_hash") != protocol_hash:
        raise ValueError("VIX admission identity changed")
    gate = pd.read_parquet(paths["vix_gate_paired_2024.parquet"])
    if vix.get("paired_table_sha256") != _frame_hash(gate):
        raise ValueError("VIX gate table changed")

    controls = validate_selected_policies(
        pd.read_parquet(paths["h1_selected_policies.parquet"])
    )
    return {
        "stream": config.stream,
        "selected_base": "price_vix",
        "control_rows": int(len(controls)),
        "source_protocol_hash": protocol_hash,
        "source_files": {
            name: _file_sha256(path) for name, path in sorted(paths.items())
        },
        "q2_loaded": False,
    }


@dataclass(frozen=True)
class AlignedPanel:
    """One timestamp/label grid with ordered probabilities for all nine models."""

    timestamp: pd.DatetimeIndex
    y_true: np.ndarray
    probabilities: Mapping[str, np.ndarray]
    fit_ids: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class CausalH1Result:
    """Causal H1 ensemble predictions, fit audit, and final frozen stack."""

    predictions: Mapping[str, pd.DataFrame]
    audit: pd.DataFrame
    final_stack: Any


def align_model_predictions(frames: Mapping[str, pd.DataFrame]) -> AlignedPanel:
    """Validate and align the exact frozen nine-model probability panel."""
    if set(frames) != set(MODEL_NAMES) or len(frames) != len(MODEL_NAMES):
        raise ValueError("ensemble requires the exact nine-model panel")
    required = {"timestamp", "y_true", "fit_id", *PROBABILITY_COLUMNS}
    ordered: dict[str, pd.DataFrame] = {}
    for model in MODEL_NAMES:
        current = frames[model].copy()
        missing = required.difference(current.columns)
        if missing:
            raise ValueError(f"{model} prediction misses columns: {sorted(missing)}")
        current["timestamp"] = pd.to_datetime(current["timestamp"], utc=True)
        current = current.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        if current.empty or current["timestamp"].duplicated().any():
            raise ValueError(f"{model} timestamps must be non-empty and unique")
        probabilities = current.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)
        valid = (
            np.isfinite(probabilities).all()
            and (probabilities >= 0.0).all()
            and (probabilities <= 1.0).all()
            and np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-9)
        )
        if not valid:
            raise ValueError(f"{model} must provide finite normalized probabilities")
        labels = current["y_true"].to_numpy(dtype=int)
        if not np.isin(labels, (0, 1, 2)).all():
            raise ValueError(f"{model} labels must be short/flat/long")
        if current["fit_id"].isna().any() or current["fit_id"].astype(str).str.len().eq(0).any():
            raise ValueError(f"{model} fit identifiers are incomplete")
        ordered[model] = current

    reference = ordered[MODEL_NAMES[0]]
    reference_time = pd.DatetimeIndex(reference["timestamp"])
    reference_y = reference["y_true"].to_numpy(dtype=int)
    for model in MODEL_NAMES[1:]:
        current = ordered[model]
        if not pd.DatetimeIndex(current["timestamp"]).equals(reference_time):
            raise ValueError(f"model timestamps differ for {model}")
        if not np.array_equal(current["y_true"].to_numpy(dtype=int), reference_y):
            raise ValueError(f"model labels differ for {model}")

    return AlignedPanel(
        timestamp=reference_time,
        y_true=reference_y,
        probabilities={
            model: ordered[model].loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)
            for model in MODEL_NAMES
        },
        fit_ids={
            model: ordered[model]["fit_id"].astype(str).to_numpy()
            for model in MODEL_NAMES
        },
    )


def stack_feature_matrix(panel: AlignedPanel) -> np.ndarray:
    """Return P(short)/P(long) in frozen model order (nine times two)."""
    if tuple(panel.probabilities) != MODEL_NAMES:
        raise ValueError("stack requires probabilities in exact nine-model order")
    rows = len(panel.timestamp)
    arrays = []
    for model in MODEL_NAMES:
        values = np.asarray(panel.probabilities[model], dtype=float)
        if values.shape != (rows, 3):
            raise ValueError(f"stack probability shape changed for {model}")
        arrays.append(values[:, [0, 2]])
    return np.column_stack(arrays)


def fit_logistic_stack(X: np.ndarray, y: np.ndarray):
    """Fit the fixed balanced L2 multinomial meta-learner."""
    matrix = np.asarray(X, dtype=float)
    labels = np.asarray(y, dtype=int)
    if matrix.ndim != 2 or matrix.shape[1] != 18 or len(matrix) != len(labels):
        raise ValueError("stack training requires aligned rows with 18 features")
    if set(np.unique(labels)) != {0, 1, 2}:
        raise ValueError("stack training requires all three class labels")
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            l1_ratio=0.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    ).fit(matrix, labels)


def combine_probabilities(
    variant: str,
    panel: AlignedPanel,
    *,
    stack_model=None,
) -> np.ndarray:
    """Combine all nine model probabilities under one frozen variant."""
    if tuple(panel.probabilities) != MODEL_NAMES:
        raise ValueError("ensemble requires probabilities in exact nine-model order")
    arrays = [np.asarray(panel.probabilities[model], dtype=float) for model in MODEL_NAMES]
    if any(values.shape != (len(panel.timestamp), 3) for values in arrays):
        raise ValueError("ensemble probability shapes differ")
    if variant == "soft_vote":
        return np.mean(np.stack(arrays, axis=0), axis=0)
    if variant == "directional_majority":
        votes = np.stack([values.argmax(axis=1) for values in arrays], axis=1)
        shares = np.eye(3, dtype=float)[votes].mean(axis=1)
        directional = (shares[:, 0] >= 5.0 / 9.0) | (shares[:, 2] >= 5.0 / 9.0)
        output = shares.copy()
        output[~directional] = np.array([0.0, 1.0, 0.0])
        return output
    if variant == "stack":
        if stack_model is None:
            raise ValueError("stack variant requires a fitted meta-model")
        classes = np.asarray(getattr(stack_model, "classes_", ()), dtype=int)
        if set(classes.tolist()) != {0, 1, 2} or len(classes) != 3:
            raise ValueError("stack model must contain all three class labels")
        predicted = np.asarray(
            stack_model.predict_proba(stack_feature_matrix(panel)), dtype=float
        )
        if predicted.shape != (len(panel.timestamp), 3):
            raise ValueError("stack probability shape changed")
        output = np.zeros_like(predicted)
        for source, label in enumerate(classes):
            output[:, int(label)] = predicted[:, source]
        if (
            not np.isfinite(output).all()
            or (output < 0.0).any()
            or not np.allclose(output.sum(axis=1), 1.0, rtol=0.0, atol=1e-9)
        ):
            raise ValueError("stack returned invalid probabilities")
        return output
    raise ValueError(f"unknown ensemble variant: {variant}")


def _prediction_frame(
    panel: AlignedPanel, probabilities: np.ndarray, fit_id: str
) -> pd.DataFrame:
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (len(panel.timestamp), 3):
        raise ValueError("ensemble prediction shape changed")
    prediction = values.argmax(axis=1).astype(int)
    return pd.DataFrame(
        {
            "timestamp": panel.timestamp,
            "y_true": panel.y_true,
            "pred": prediction,
            "confidence": values.max(axis=1),
            "p_short": values[:, 0],
            "p_flat": values[:, 1],
            "p_long": values[:, 2],
            "fit_id": fit_id,
        }
    )


def _subset_panel(panel: AlignedPanel, mask: np.ndarray) -> AlignedPanel:
    current = np.asarray(mask, dtype=bool)
    if current.shape != (len(panel.timestamp),):
        raise ValueError("panel mask shape changed")
    return AlignedPanel(
        timestamp=panel.timestamp[current],
        y_true=panel.y_true[current],
        probabilities={
            model: np.asarray(panel.probabilities[model])[current]
            for model in MODEL_NAMES
        },
        fit_ids={
            model: np.asarray(panel.fit_ids[model])[current]
            for model in MODEL_NAMES
        },
    )


def _available_training(
    oof_panel: AlignedPanel,
    h1_panel: AlignedPanel,
    cutoff: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    timestamps: list[pd.DatetimeIndex] = []
    for panel in (oof_panel, h1_panel):
        available = (panel.timestamp + LABEL_AVAILABLE_LAG) <= cutoff
        if available.any():
            subset = _subset_panel(panel, np.asarray(available, dtype=bool))
            matrices.append(stack_feature_matrix(subset))
            labels.append(subset.y_true)
            timestamps.append(subset.timestamp)
    if not matrices:
        raise ValueError("causal stack has no labels available before cutoff")
    return (
        np.vstack(matrices),
        np.concatenate(labels),
        timestamps[0].append(timestamps[1:]),
    )


def build_causal_h1_predictions(
    oof_panel: AlignedPanel,
    h1_panel: AlignedPanel,
    *,
    stack_factory: Callable[[np.ndarray, np.ndarray], Any] = fit_logistic_stack,
) -> CausalH1Result:
    """Build causal monthly H1 variants and freeze the final pre-forward stack."""
    if oof_panel.timestamp.empty or not (oof_panel.timestamp < CALIBRATION_START).all():
        raise ValueError("OOF rows must end before H1")
    if (
        h1_panel.timestamp.empty
        or not (
            (h1_panel.timestamp >= CALIBRATION_START)
            & (h1_panel.timestamp < FORWARD_START)
        ).all()
    ):
        raise ValueError("H1 rows must remain inside H1")
    if not oof_panel.timestamp.is_monotonic_increasing or not h1_panel.timestamp.is_monotonic_increasing:
        raise ValueError("causal panels must be sorted")
    stack_feature_matrix(oof_panel)
    stack_feature_matrix(h1_panel)

    predictions: dict[str, pd.DataFrame] = {}
    for variant in ("soft_vote", "directional_majority"):
        predictions[variant] = _prediction_frame(
            h1_panel,
            combine_probabilities(variant, h1_panel),
            f"all-nine:{variant}:h1",
        )

    periods = pd.Index(h1_panel.timestamp.strftime("%Y-%m")).unique().tolist()
    stack_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for period in periods:
        month_start = pd.Timestamp(f"{period}-01T00:00:00Z")
        month_mask = h1_panel.timestamp.strftime("%Y-%m") == period
        current = _subset_panel(h1_panel, np.asarray(month_mask, dtype=bool))
        train_X, train_y, train_time = _available_training(
            oof_panel, h1_panel, month_start
        )
        stack = stack_factory(train_X, train_y)
        stack_frames.append(
            _prediction_frame(
                current,
                combine_probabilities("stack", current, stack_model=stack),
                f"causal-stack:h1:{period}:rows{len(train_y)}",
            )
        )
        max_available = (train_time + LABEL_AVAILABLE_LAG).max()
        if max_available > month_start:
            raise AssertionError("stack training used a label unavailable at prediction time")
        audit_rows.append(
            {
                "prediction_period": period,
                "training_cutoff": month_start,
                "meta_train_rows": int(len(train_y)),
                "meta_train_start": train_time.min(),
                "meta_train_last_prediction": train_time.max(),
                "max_label_available_at": max_available,
            }
        )

    final_X, final_y, final_time = _available_training(
        oof_panel, h1_panel, FORWARD_START
    )
    final_stack = stack_factory(final_X, final_y)
    final_max_available = (final_time + LABEL_AVAILABLE_LAG).max()
    if final_max_available > FORWARD_START:
        raise AssertionError("final stack crossed the forward boundary")
    audit_rows.append(
        {
            "prediction_period": "forward_freeze",
            "training_cutoff": FORWARD_START,
            "meta_train_rows": int(len(final_y)),
            "meta_train_start": final_time.min(),
            "meta_train_last_prediction": final_time.max(),
            "max_label_available_at": final_max_available,
        }
    )
    predictions["stack"] = (
        pd.concat(stack_frames, ignore_index=True)
        .sort_values("timestamp", kind="mergesort")
        .reset_index(drop=True)
    )
    return CausalH1Result(
        predictions=predictions,
        audit=pd.DataFrame(audit_rows),
        final_stack=final_stack,
    )


def build_h1_grid(
    bars: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    *,
    arm: str,
    width_bps: int,
    cost_bps: float,
) -> pd.DataFrame:
    """Evaluate the fixed H1 threshold grid for prebuilt ensemble predictions."""
    if int(width_bps) not in WIDTHS:
        raise ValueError("ensemble width is outside the frozen grid")
    if not predictions or not set(predictions).issubset(ENSEMBLE_VARIANTS):
        raise ValueError("ensemble prediction variants changed")
    month_edges = pd.date_range(CALIBRATION_START, FORWARD_START, freq="MS")
    rows: list[dict[str, Any]] = []
    for variant, source in predictions.items():
        prediction = source.copy()
        required = {"timestamp", "pred", "confidence"}
        missing = required.difference(prediction.columns)
        if missing:
            raise ValueError(f"{variant} H1 prediction misses columns: {sorted(missing)}")
        prediction["timestamp"] = pd.to_datetime(prediction["timestamp"], utc=True)
        if (
            prediction.empty
            or prediction["timestamp"].duplicated().any()
            or not prediction["timestamp"].between(
                CALIBRATION_START, FORWARD_START, inclusive="left"
            ).all()
            or not np.isfinite(
                prediction[["pred", "confidence"]].to_numpy(dtype=float)
            ).all()
        ):
            raise ValueError(f"{variant} H1 prediction contract changed")
        for tau in TAUS:
            ledger, per_bar = simulate_one_bar(
                bars,
                prediction,
                start=CALIBRATION_START,
                end=FORWARD_START,
                tau=float(tau),
                cost_bps=float(cost_bps),
            )
            economics = daily_economics(
                ledger,
                per_bar,
                start=CALIBRATION_START,
                end=FORWARD_START,
            )
            positive_months = sum(
                float(
                    per_bar.loc[
                        (per_bar.index >= left) & (per_bar.index < right)
                    ].sum()
                )
                > 0.0
                for left, right in zip(month_edges[:-1], month_edges[1:])
            )
            row = {
                "arm": str(arm),
                "variant": str(variant),
                "width_bps": int(width_bps),
                "tau": float(tau),
                "positive_months": int(positive_months),
                **economics,
            }
            numeric = np.asarray(
                [value for value in row.values() if isinstance(value, (int, float))],
                dtype=float,
            )
            if not np.isfinite(numeric).all():
                raise ValueError(f"non-finite H1 economics for {arm}/{variant}")
            rows.append(row)
    return pd.DataFrame(rows)


def select_ensemble_rows(grid: pd.DataFrame) -> pd.DataFrame:
    """Select one width/threshold row for every available arm/variant pair."""
    required = {
        "arm",
        "variant",
        "width_bps",
        "tau",
        "trades",
        "n_long",
        "n_short",
        "positive_months",
        "daily_sortino",
        "net_return",
    }
    missing = required.difference(grid.columns)
    if missing:
        raise ValueError(f"ensemble H1 grid misses columns: {sorted(missing)}")
    if grid.empty:
        raise ValueError("ensemble H1 grid is empty")
    rows = []
    for (arm, variant), scope in grid.groupby(["arm", "variant"], sort=False):
        winner = select_h1_policy(scope).to_dict()
        winner.update(
            {
                "arm": str(arm),
                "variant": str(variant),
                "h1_execution_status": (
                    "eligible"
                    if bool(winner["eligible"])
                    else "diagnostic_only_no_eligible_policy"
                ),
            }
        )
        rows.append(winner)
    return pd.DataFrame(rows).reset_index(drop=True)


def select_overall_ensemble(candidates: pd.DataFrame) -> dict[str, Any]:
    """Freeze one index-level H1 winner across arm/ensemble candidate rows."""
    winner = select_h1_policy(candidates).to_dict()
    winner["selection_scope"] = "h1_only_across_feature_arm_and_ensemble_variant"
    return winner


def promotion_decision(
    winner: Mapping[str, Any], control: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the fixed H1 ensemble-versus-single promotion gate."""
    required_winner = {"eligible", "net_return", "daily_sortino", "trades"}
    required_control = {"net_return", "daily_sortino", "trades"}
    if required_winner.difference(winner) or required_control.difference(control):
        raise ValueError("promotion rows miss required H1 fields")
    control_trades = int(control["trades"])
    retention = (
        float(int(winner["trades"]) / control_trades)
        if control_trades > 0
        else 0.0
    )
    conditions = {
        "h1_eligible": bool(winner["eligible"]),
        "net_above_control": bool(
            float(winner["net_return"]) > float(control["net_return"])
        ),
        "sortino_above_control": bool(
            float(winner["daily_sortino"]) > float(control["daily_sortino"])
        ),
        "trade_retention_80pct": bool(retention >= 0.80),
    }
    return {
        "promoted": all(conditions.values()),
        "conditions": conditions,
        "trade_retention": retention,
    }


class IndexAllModelEnsembleRunner:
    """Build H1-only ensembles and replay their frozen forward candidates."""

    def __init__(
        self,
        config: IndexAllModelEnsembleConfig,
        *,
        source_runner_factory: Callable[..., IndexReplicationRunner] = IndexReplicationRunner,
    ) -> None:
        self.config = config
        self.source_runner_factory = source_runner_factory
        self.output_root = Path(config.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _state(self, stage: str, **detail: Any) -> None:
        _atomic_json(
            {
                "stage": stage,
                "updated_at_utc": pd.Timestamp.now("UTC"),
                **detail,
            },
            self.output_root / "run_state.json",
        )

    def _freeze_protocol(self, source: Mapping[str, Any]) -> str:
        implementation_hash = hashlib.sha256(
            inspect.getsource(IndexAllModelEnsembleRunner).encode("utf-8")
        ).hexdigest()
        body = {
            "protocol_version": PROTOCOL_VERSION,
            "stream": self.config.stream,
            "evidence_role": EVIDENCE_ROLE,
            "models": list(MODEL_NAMES),
            "arms": list(ARMS),
            "variants": list(ENSEMBLE_VARIANTS),
            "widths_bps": list(WIDTHS),
            "taus": list(TAUS),
            "selection": [SELECTION_START, SELECTION_END],
            "calibration": [CALIBRATION_START, FORWARD_START],
            "forward": [FORWARD_START, FORWARD_END],
            "q2_start": CUTOFF,
            "q2_loaded": False,
            "selected_base": "price_vix",
            "cost_bps": self.config.cost_bps,
            "execution": "next_consecutive_m15_open_to_same_bar_close",
            "meta_learner": (
                "StandardScaler + balanced L2 LogisticRegression"
                "(C=0.1, random_state=42)"
            ),
            "meta_features": "P(short) and P(long) from all nine models",
            "h1_selection": "constraint,daily_sortino,net_return,trades,width,tau",
            "promotion": "eligible and better Net/Sortino and 80pct trade retention",
            "source_protocol_hash": source["source_protocol_hash"],
            "source_files": source["source_files"],
            "implementation_sha256": implementation_hash,
        }
        payload = {**body, "protocol_hash": _payload_hash(body)}
        path = self.output_root / "protocol.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != json.loads(json.dumps(payload, default=str)):
                # Timestamp spelling is canonicalised by the atomic writer; the
                # protocol hash is the authoritative equality check.
                if existing.get("protocol_hash") != payload["protocol_hash"]:
                    raise ValueError("existing ensemble protocol changed")
        _atomic_json(payload, path)
        return str(payload["protocol_hash"])

    def _source_runner(self) -> IndexReplicationRunner:
        source_config = IndexReplicationConfig.for_stream(
            self.config.stream,
            data_dir=self.config.data_dir,
            output_base=Path(self.config.source_root).parent,
        )
        return self.source_runner_factory(source_config)

    @staticmethod
    def _oof_panel(
        source: IndexReplicationRunner, arm: str, width_bps: int
    ) -> AlignedPanel:
        frames: dict[str, pd.DataFrame] = {}
        for model in MODEL_NAMES:
            _classification, folds = source._oof_arm(arm, model, width_bps)
            frames[model] = pd.concat(folds, ignore_index=True).sort_values(
                "timestamp", kind="mergesort"
            )
        return align_model_predictions(frames)

    @staticmethod
    def _h1_panel(
        source: IndexReplicationRunner, arm: str, width_bps: int
    ) -> AlignedPanel:
        return align_model_predictions(
            {
                model: source._monthly_predictions(arm, model, width_bps)
                for model in MODEL_NAMES
            }
        )

    @staticmethod
    def _forward_panel(
        source: IndexReplicationRunner, arm: str, width_bps: int
    ) -> AlignedPanel:
        return align_model_predictions(
            {
                model: source._forward_prediction(arm, model, width_bps)
                for model in MODEL_NAMES
            }
        )

    @staticmethod
    def _meta_rows(
        stack: Any, *, arm: str, width_bps: int
    ) -> list[dict[str, Any]]:
        scaler = stack.named_steps["standardscaler"]
        logistic = stack.named_steps["logisticregression"]
        feature_names = [
            f"{model}__{direction}"
            for model in MODEL_NAMES
            for direction in ("p_short", "p_long")
        ]
        rows: list[dict[str, Any]] = []
        for class_position, class_id in enumerate(logistic.classes_):
            for feature_position, feature_name in enumerate(feature_names):
                rows.append(
                    {
                        "arm": arm,
                        "width_bps": int(width_bps),
                        "class_id": int(class_id),
                        "feature": feature_name,
                        "coefficient": float(
                            logistic.coef_[class_position, feature_position]
                        ),
                        "intercept": float(logistic.intercept_[class_position]),
                        "scaler_mean": float(scaler.mean_[feature_position]),
                        "scaler_scale": float(scaler.scale_[feature_position]),
                    }
                )
        return rows

    def _build_h1(
        self,
        source: IndexReplicationRunner,
        *,
        protocol_hash: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, int], Any]]:
        grids: list[pd.DataFrame] = []
        audits: list[pd.DataFrame] = []
        meta_rows: list[dict[str, Any]] = []
        final_stacks: dict[tuple[str, int], Any] = {}
        total = len(ARMS) * len(WIDTHS)
        completed = 0
        for arm in ARMS:
            for width in WIDTHS:
                causal = build_causal_h1_predictions(
                    self._oof_panel(source, arm, int(width)),
                    self._h1_panel(source, arm, int(width)),
                )
                final_stacks[(arm, int(width))] = causal.final_stack
                grid = build_h1_grid(
                    source.bars,
                    causal.predictions,
                    arm=arm,
                    width_bps=int(width),
                    cost_bps=self.config.cost_bps,
                )
                grids.append(grid)
                audits.append(
                    causal.audit.assign(arm=arm, width_bps=int(width))
                )
                meta_rows.extend(
                    self._meta_rows(causal.final_stack, arm=arm, width_bps=int(width))
                )
                completed += 1
                self._state(
                    "h1_build",
                    completed=completed,
                    total=total,
                    arm=arm,
                    width_bps=int(width),
                )
        h1_grid = pd.concat(grids, ignore_index=True)
        expected_grid_rows = len(ARMS) * len(ENSEMBLE_VARIANTS) * len(WIDTHS) * len(TAUS)
        if len(h1_grid) != expected_grid_rows:
            raise AssertionError(
                f"ensemble H1 grid expected {expected_grid_rows} rows, found {len(h1_grid)}"
            )
        candidates = select_ensemble_rows(h1_grid)
        candidates["role"] = "ensemble"
        candidates["model_name"] = "ALL_NINE"
        candidates["candidate_id"] = (
            candidates["arm"].astype(str)
            + "__"
            + candidates["variant"].astype(str)
        )
        if len(candidates) != len(ARMS) * len(ENSEMBLE_VARIANTS):
            raise AssertionError("ensemble H1 selection must contain twelve rows")

        controls = validate_selected_policies(
            pd.read_parquet(Path(self.config.source_root) / "h1_selected_policies.parquet")
        )
        control = select_h1_policy(controls).to_dict()
        control.update(
            {
                "role": "best_single_control",
                "variant": "best_single",
                "candidate_id": (
                    f"best_single__{control['arm']}__{control['model_name']}"
                ),
                "h1_execution_status": (
                    "eligible"
                    if bool(control["eligible"])
                    else "diagnostic_only_no_eligible_policy"
                ),
            }
        )
        control_frame = pd.DataFrame([control])
        winner = select_overall_ensemble(candidates)
        promotion = promotion_decision(winner, control)
        selection = {
            "protocol_hash": protocol_hash,
            "stream": self.config.stream,
            "evidence_role": EVIDENCE_ROLE,
            "candidate_rows": int(len(candidates)),
            "candidate_sha256": _frame_hash(candidates),
            "control_sha256": _frame_hash(control_frame),
            "h1_winner": {
                key: winner[key]
                for key in (
                    "candidate_id",
                    "arm",
                    "variant",
                    "width_bps",
                    "tau",
                    "eligible",
                    "net_return",
                    "daily_sortino",
                    "trades",
                )
            },
            "control": {
                key: control[key]
                for key in (
                    "candidate_id",
                    "arm",
                    "variant",
                    "model_name",
                    "width_bps",
                    "tau",
                    "eligible",
                    "net_return",
                    "daily_sortino",
                    "trades",
                )
            },
            "promotion": promotion,
            "selection_data_end_exclusive": FORWARD_START,
            "forward_loaded": False,
            "q2_loaded": False,
        }
        _atomic_parquet(h1_grid, self.output_root / "h1_policy_grid.parquet")
        _atomic_parquet(candidates, self.output_root / "h1_selected_candidates.parquet")
        _atomic_parquet(controls, self.output_root / "h1_single_controls.parquet")
        _atomic_parquet(control_frame, self.output_root / "h1_selected_control.parquet")
        _atomic_parquet(pd.DataFrame(meta_rows), self.output_root / "meta_coefficients.parquet")
        _atomic_parquet(pd.concat(audits, ignore_index=True), self.output_root / "stack_audit.parquet")
        _atomic_json(selection, self.output_root / "h1_selection.json")
        self._state(
            "selection_frozen_before_forward",
            protocol_hash=protocol_hash,
            candidate_rows=len(candidates),
        )
        return h1_grid, candidates, control_frame, final_stacks

    def _load_frozen_h1(
        self, protocol_hash: str
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        paths = {
            "grid": self.output_root / "h1_policy_grid.parquet",
            "candidates": self.output_root / "h1_selected_candidates.parquet",
            "controls": self.output_root / "h1_single_controls.parquet",
            "control": self.output_root / "h1_selected_control.parquet",
            "meta": self.output_root / "meta_coefficients.parquet",
            "audit": self.output_root / "stack_audit.parquet",
            "selection": self.output_root / "h1_selection.json",
        }
        present = {name: path.exists() for name, path in paths.items()}
        if not any(present.values()):
            return None
        if not all(present.values()):
            raise ValueError("partial frozen H1 artifact set")
        selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
        if (
            selection.get("protocol_hash") != protocol_hash
            or selection.get("forward_loaded") is not False
            or selection.get("q2_loaded") is not False
        ):
            raise ValueError("frozen H1 selection identity changed")
        candidates = pd.read_parquet(paths["candidates"])
        control = pd.read_parquet(paths["control"])
        if (
            len(candidates) != 12
            or len(control) != 1
            or selection.get("candidate_rows") != 12
            or selection.get("candidate_sha256") != _frame_hash(candidates)
            or selection.get("control_sha256") != _frame_hash(control)
        ):
            raise ValueError("frozen H1 selection tables changed")
        grid = pd.read_parquet(paths["grid"])
        if len(grid) != len(ARMS) * len(ENSEMBLE_VARIANTS) * len(WIDTHS) * len(TAUS):
            raise ValueError("frozen H1 grid changed")
        validate_selected_policies(pd.read_parquet(paths["controls"]))
        meta = pd.read_parquet(paths["meta"])
        audit = pd.read_parquet(paths["audit"])
        if meta.empty or audit.empty or not np.isfinite(
            meta.select_dtypes(include="number").to_numpy(dtype=float)
        ).all():
            raise ValueError("frozen stack evidence changed")
        return candidates, control

    def _rebuild_final_stack(
        self, source: IndexReplicationRunner, arm: str, width_bps: int
    ) -> Any:
        return build_causal_h1_predictions(
            self._oof_panel(source, arm, width_bps),
            self._h1_panel(source, arm, width_bps),
        ).final_stack

    @staticmethod
    def _identity(policy: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "candidate_id": str(policy["candidate_id"]),
            "role": str(policy["role"]),
            "arm": str(policy["arm"]),
            "variant": str(policy["variant"]),
            "model_name": str(policy["model_name"]),
            "width_bps": int(policy["width_bps"]),
            "tau": float(policy["tau"]),
            "h1_eligible": bool(policy["eligible"]),
            "h1_execution_status": str(policy["h1_execution_status"]),
        }

    @staticmethod
    def _validate_forward_prediction(
        prediction: pd.DataFrame, candidate_id: str
    ) -> pd.DataFrame:
        required = {"timestamp", "pred", "confidence", "fit_id"}
        missing = required.difference(prediction.columns)
        current = prediction.copy()
        if missing or current.empty:
            raise ValueError(f"forward prediction is incomplete for {candidate_id}")
        current["timestamp"] = pd.to_datetime(current["timestamp"], utc=True)
        if (
            current["timestamp"].duplicated().any()
            or not current["timestamp"].between(
                FORWARD_START, FORWARD_END, inclusive="left"
            ).all()
            or not np.isfinite(
                current[["pred", "confidence"]].to_numpy(dtype=float)
            ).all()
        ):
            raise ValueError(f"forward prediction contract changed for {candidate_id}")
        return current.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    @staticmethod
    def _monthly_rows(
        ledger: pd.DataFrame,
        per_bar: pd.Series,
        identity: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        edges = pd.date_range(FORWARD_START, FORWARD_END, freq="MS")
        for start, end in zip(edges[:-1], edges[1:]):
            economics = daily_economics(
                ledger,
                per_bar,
                start=start,
                end=end,
            )
            rows.append(
                {
                    **identity,
                    "month": start.strftime("%Y-%m"),
                    "period_start": start,
                    "period_end": end,
                    **economics,
                }
            )
        return rows

    def _completed_candidate(
        self,
        identity: Mapping[str, Any],
        *,
        protocol_hash: str,
        candidate_hash: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], pd.Timestamp] | None:
        candidate_id = str(identity["candidate_id"])
        checkpoint = self.output_root / "forward" / f"{candidate_id}.json"
        if not checkpoint.exists():
            return None
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_hash") != protocol_hash
            or payload.get("candidate_set_sha256") != candidate_hash
            or payload.get("policy") != dict(identity)
        ):
            raise ValueError(f"resume identity changed for {candidate_id}")
        expected_paths = {
            "ledger": f"forward_ledgers/{candidate_id}.parquet",
            "per_bar": f"forward_ledgers/{candidate_id}_per_bar.parquet",
        }
        records = payload.get("artifacts", {})
        resolved: dict[str, Path] = {}
        for name, relative in expected_paths.items():
            record = records.get(name)
            path = self.output_root / relative
            if (
                not isinstance(record, dict)
                or record.get("path") != relative
                or not path.exists()
                or record.get("sha256") != _file_sha256(path)
            ):
                raise ValueError(f"resume artifact hash changed for {candidate_id}")
            resolved[name] = path
        ledger = pd.read_parquet(resolved["ledger"])
        per_bar_frame = pd.read_parquet(resolved["per_bar"])
        if not {"timestamp", "net_return"}.issubset(per_bar_frame.columns):
            raise ValueError(f"resume per-bar schema changed for {candidate_id}")
        per_bar_frame["timestamp"] = pd.to_datetime(per_bar_frame["timestamp"], utc=True)
        if len(per_bar_frame) and not per_bar_frame["timestamp"].between(
            FORWARD_START, FORWARD_END, inclusive="left"
        ).all():
            raise ValueError(f"resume timestamp changed for {candidate_id}")
        per_bar = per_bar_frame.set_index("timestamp")["net_return"].astype(float)
        economics = daily_economics(
            ledger, per_bar, start=FORWARD_START, end=FORWARD_END
        )
        stress = _stress_economics(ledger, per_bar)
        summary = dict(payload.get("summary", {}))
        expected = {
            **identity,
            **economics,
            "stress_2x_net_return": stress["net_return"],
            "stress_2x_daily_sharpe": stress["daily_sharpe"],
            "stress_2x_daily_sortino": stress["daily_sortino"],
        }
        for key, value in expected.items():
            actual = summary.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)):
                if actual is None or not np.isclose(
                    float(actual), float(value), rtol=1e-12, atol=1e-12
                ):
                    raise ValueError(f"resume economics changed for {candidate_id}: {key}")
            elif actual != value:
                raise ValueError(f"resume summary changed for {candidate_id}: {key}")
        maximum = pd.Timestamp(payload["prediction_max_timestamp"])
        maximum = maximum.tz_localize("UTC") if maximum.tzinfo is None else maximum.tz_convert("UTC")
        if not (FORWARD_START <= maximum < FORWARD_END):
            raise ValueError(f"resume prediction boundary changed for {candidate_id}")
        return summary, self._monthly_rows(ledger, per_bar, identity), maximum

    def _artifact_paths(self, candidate_ids: list[str]) -> list[Path]:
        fixed = [
            self.output_root / "protocol.json",
            self.output_root / "h1_policy_grid.parquet",
            self.output_root / "h1_selected_candidates.parquet",
            self.output_root / "h1_single_controls.parquet",
            self.output_root / "h1_selected_control.parquet",
            self.output_root / "h1_selection.json",
            self.output_root / "meta_coefficients.parquet",
            self.output_root / "stack_audit.parquet",
            self.output_root / "forward_summary.parquet",
            self.output_root / "forward_monthly.parquet",
            self.output_root / "result.json",
        ]
        return [
            *fixed,
            *(self.output_root / "forward" / f"{value}.json" for value in candidate_ids),
            *(
                self.output_root / "forward_ledgers" / f"{value}{suffix}.parquet"
                for value in candidate_ids
                for suffix in ("", "_per_bar")
            ),
        ]

    def _write_manifest(self, protocol_hash: str, candidate_ids: list[str]) -> None:
        expected_checkpoints = {
            self.output_root / "forward" / f"{value}.json" for value in candidate_ids
        }
        expected_ledgers = {
            self.output_root / "forward_ledgers" / f"{value}{suffix}.parquet"
            for value in candidate_ids
            for suffix in ("", "_per_bar")
        }
        if set((self.output_root / "forward").glob("*.json")) != expected_checkpoints:
            raise ValueError("ensemble manifest found an unexpected checkpoint set")
        if set((self.output_root / "forward_ledgers").glob("*.parquet")) != expected_ledgers:
            raise ValueError("ensemble manifest found an unexpected ledger set")
        paths = self._artifact_paths(candidate_ids)
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"ensemble manifest misses artifacts: {missing}")
        payload = {
            "protocol_hash": protocol_hash,
            "q2_loaded": False,
            "artifacts": {
                path.relative_to(self.output_root).as_posix(): _file_sha256(path)
                for path in paths
            },
        }
        _atomic_json(payload, self.output_root / "manifest.json")

    def _validate_existing_manifest(self, protocol_hash: str) -> None:
        path = self.output_root / "manifest.json"
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_hash") != protocol_hash
            or payload.get("q2_loaded") is not False
            or not isinstance(payload.get("artifacts"), dict)
        ):
            raise ValueError("existing ensemble manifest identity changed")
        for relative, expected_hash in payload["artifacts"].items():
            artifact = self.output_root / str(relative)
            if not artifact.exists() or _file_sha256(artifact) != expected_hash:
                raise ValueError(f"manifest artifact hash changed: {relative}")

    def run(self) -> dict[str, Any]:
        source_contract = validate_source_contract(self.config)
        protocol_hash = self._freeze_protocol(source_contract)
        self._validate_existing_manifest(protocol_hash)
        frozen = self._load_frozen_h1(protocol_hash)
        source_runner: IndexReplicationRunner | None = None
        final_stacks: dict[tuple[str, int], Any] = {}
        if frozen is None:
            source_runner = self._source_runner()
            _grid, candidates, control_frame, final_stacks = self._build_h1(
                source_runner, protocol_hash=protocol_hash
            )
        else:
            candidates, control_frame = frozen

        policies = [
            *candidates.to_dict("records"),
            *control_frame.to_dict("records"),
        ]
        identities = [self._identity(policy) for policy in policies]
        candidate_hash = _payload_hash(identities)
        rows: list[dict[str, Any]] = []
        monthly_rows: list[dict[str, Any]] = []
        maxima: list[pd.Timestamp] = []
        resumed = 0
        for number, (policy, identity) in enumerate(
            zip(policies, identities), start=1
        ):
            completed = self._completed_candidate(
                identity,
                protocol_hash=protocol_hash,
                candidate_hash=candidate_hash,
            )
            if completed is not None:
                summary, months, maximum = completed
                rows.append(summary)
                monthly_rows.extend(months)
                maxima.append(maximum)
                resumed += 1
                continue
            if source_runner is None:
                source_runner = self._source_runner()
            arm = identity["arm"]
            width = identity["width_bps"]
            if identity["role"] == "best_single_control":
                prediction = source_runner._forward_prediction(
                    arm, identity["model_name"], width
                )
            else:
                panel = self._forward_panel(source_runner, arm, width)
                stack = None
                if identity["variant"] == "stack":
                    key = (arm, width)
                    if key not in final_stacks:
                        final_stacks[key] = self._rebuild_final_stack(
                            source_runner, arm, width
                        )
                    stack = final_stacks[key]
                probabilities = combine_probabilities(
                    identity["variant"], panel, stack_model=stack
                )
                prediction = _prediction_frame(
                    panel,
                    probabilities,
                    f"index-all-nine:{arm}:{identity['variant']}:w{width}",
                )
            prediction = self._validate_forward_prediction(
                prediction, identity["candidate_id"]
            )
            maximum = prediction["timestamp"].max()
            maxima.append(maximum)
            ledger, per_bar = simulate_one_bar(
                source_runner.bars,
                prediction,
                start=FORWARD_START,
                end=FORWARD_END,
                tau=identity["tau"],
                cost_bps=self.config.cost_bps,
            )
            economics = daily_economics(
                ledger, per_bar, start=FORWARD_START, end=FORWARD_END
            )
            stress = _stress_economics(ledger, per_bar)
            summary = {
                "stream": self.config.stream,
                **identity,
                "evidence_role": EVIDENCE_ROLE,
                "status": "traded" if economics["trades"] else "no_trades",
                **economics,
                "stress_2x_net_return": stress["net_return"],
                "stress_2x_daily_sharpe": stress["daily_sharpe"],
                "stress_2x_daily_sortino": stress["daily_sortino"],
                "fit_id": str(prediction["fit_id"].iloc[0]),
            }
            if any(value is None for value in summary.values()) or not np.isfinite(
                np.asarray(
                    [
                        value
                        for value in summary.values()
                        if isinstance(value, (int, float, np.integer, np.floating))
                    ],
                    dtype=float,
                )
            ).all():
                raise ValueError(
                    f"non-finite forward summary for {identity['candidate_id']}"
                )
            candidate_id = identity["candidate_id"]
            ledger_path = self.output_root / "forward_ledgers" / f"{candidate_id}.parquet"
            per_bar_path = self.output_root / "forward_ledgers" / f"{candidate_id}_per_bar.parquet"
            _atomic_parquet(ledger, ledger_path)
            _atomic_parquet(
                per_bar.rename("net_return").rename_axis("timestamp").reset_index(),
                per_bar_path,
            )
            _atomic_json(
                {
                    "protocol_hash": protocol_hash,
                    "candidate_set_sha256": candidate_hash,
                    "policy": identity,
                    "prediction_max_timestamp": maximum,
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
                    "summary": summary,
                },
                self.output_root / "forward" / f"{candidate_id}.json",
            )
            rows.append(summary)
            monthly_rows.extend(self._monthly_rows(ledger, per_bar, identity))
            self._state(
                "forward",
                completed=number,
                total=len(policies),
                candidate_id=candidate_id,
            )

        summary_frame = pd.DataFrame(rows).sort_values(
            ["role", "arm", "variant"], kind="mergesort"
        ).reset_index(drop=True)
        monthly_frame = pd.DataFrame(monthly_rows).sort_values(
            ["candidate_id", "period_start"], kind="mergesort"
        ).reset_index(drop=True)
        if len(summary_frame) != 13 or len(monthly_frame) != 13 * 9:
            raise AssertionError("ensemble forward artifacts must contain 13 candidates and 9 months")
        if summary_frame.isna().any().any() or monthly_frame.isna().any().any():
            raise ValueError("ensemble forward artifacts contain missing values")
        _atomic_parquet(summary_frame, self.output_root / "forward_summary.parquet")
        _atomic_parquet(monthly_frame, self.output_root / "forward_monthly.parquet")
        selection = json.loads(
            (self.output_root / "h1_selection.json").read_text(encoding="utf-8")
        )
        result = {
            "stream": self.config.stream,
            "protocol_version": PROTOCOL_VERSION,
            "protocol_hash": protocol_hash,
            "evidence_role": EVIDENCE_ROLE,
            "selected_base": "price_vix",
            "models_per_ensemble": len(MODEL_NAMES),
            "arms": len(ARMS),
            "variants": len(ENSEMBLE_VARIANTS),
            "h1_candidate_rows": len(candidates),
            "forward_rows": len(summary_frame),
            "resumed_forward_candidates": resumed,
            "h1_winner": selection["h1_winner"],
            "control": selection["control"],
            "promotion": selection["promotion"],
            "max_prediction_timestamp": max(maxima),
            "q2_loaded": False,
        }
        _atomic_json(result, self.output_root / "result.json")
        candidate_ids = [identity["candidate_id"] for identity in identities]
        self._write_manifest(protocol_hash, candidate_ids)
        self._state("complete", **result)
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", choices=tuple(STREAM_CONFIG), required=True)
    args = parser.parse_args(argv)
    result = IndexAllModelEnsembleRunner(
        IndexAllModelEnsembleConfig.for_stream(args.stream)
    ).run()
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AlignedPanel",
    "CausalH1Result",
    "IndexAllModelEnsembleConfig",
    "IndexAllModelEnsembleRunner",
    "ENSEMBLE_VARIANTS",
    "LABEL_AVAILABLE_LAG",
    "MODEL_NAMES",
    "PROBABILITY_COLUMNS",
    "align_model_predictions",
    "build_causal_h1_predictions",
    "build_h1_grid",
    "combine_probabilities",
    "fit_logistic_stack",
    "main",
    "promotion_decision",
    "select_ensemble_rows",
    "select_overall_ensemble",
    "stack_feature_matrix",
    "validate_source_contract",
]
