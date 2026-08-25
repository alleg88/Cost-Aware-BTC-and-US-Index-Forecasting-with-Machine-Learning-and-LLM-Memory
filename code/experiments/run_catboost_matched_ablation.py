"""Resumable 2024 matched-candidate CatBoost scoring."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import catboost
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from evaluation.economics import economics_summary
from evaluation.trades_intrabar import simulate_bracket_trades_intrabar
from experiments.catboost_matched_ablation import (
    CALIBRATION_END,
    FORWARD_END,
    GEOMETRIES,
    REGIMES,
    REGIME_LOOKBACK_BARS,
    REGIME_RETURN_THRESHOLD,
    SELECTION_END,
    SELECTION_START,
    TAUS,
    WIDTHS,
    economic_ranking_key,
    f1_ranking_key,
    fixed_splitter,
    load_candidates,
    past_regime_labels,
    policy_choices,
    robust_f1_score,
    robust_score,
    total_constraint_violation,
    validate_fold_regime_counts,
)
from experiments.run_walkforward import build_walkforward_xy
from features.build import make_label
from models.zoo import DEFAULT_CATBOOST, MODELS, _aligned_proba

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = CODE_ROOT / "configs" / "default.yaml"
MINUTE_PATH = CODE_ROOT / "data" / "btcusdt_1m_2024_2026.parquet"
CACHE_ROOT = CODE_ROOT / "experiments" / "cache" / "tuning" / "catboost_matched_ablation"
PROTOCOL_VERSION = "matched-catboost-2024-v1"
LABEL_TAIL_TRIM = 1
BAR_SIZE = pd.Timedelta(minutes=15)
PREDICTION_SCHEMA_VERSION = 2
POLICY_SCHEMA_VERSION = 2
PREDICTION_COLUMNS = (
    "timestamp", "width_bps", "candidate_id", "fold_id", "y_true", "pred",
    "confidence", "p_short", "p_flat", "p_long", "train_start", "train_end",
    "test_start", "test_end", "refit_id",
)
SIMULATOR_VERSION = hashlib.sha256(
    inspect.getsource(simulate_bracket_trades_intrabar).encode("utf-8")
).hexdigest()[:16]


@dataclass(frozen=True)
class PreparedData:
    bars: pd.DataFrame
    minute: pd.DataFrame
    features: Mapping[int, tuple[pd.DataFrame, pd.Series]]
    regimes: pd.Series
    m15_fingerprint: str
    minute_fingerprint: str
    sentiment_mode: str = "both"
    feature_columns: tuple[str, ...] = ()


def _canonical(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        value = (
            value.tz_localize("UTC")
            if value.tzinfo is None
            else value.tz_convert("UTC")
        )
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _content_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _canonical(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def frame_fingerprint(frame: pd.DataFrame | pd.Series) -> str:
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    metadata = (
        {"name": frame.name, "dtype": str(frame.dtype)}
        if isinstance(frame, pd.Series)
        else {
            "columns": list(frame.columns),
            "dtypes": [str(dtype) for dtype in frame.dtypes],
        }
    )
    digest = hashlib.sha256(values.tobytes())
    digest.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def merged_catboost_params(candidate_params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "loss_function": "MultiClass",
        "verbose": 0,
        **DEFAULT_CATBOOST,
        **dict(candidate_params),
    }


def prediction_cache_fingerprint(
    *,
    width_bps: int,
    candidate_params: Mapping[str, Any],
    fold_metadata: Mapping[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    regimes: pd.Series,
    data_fingerprint: str,
    model_name: str | None = None,
) -> str:
    fold = {
        key: fold_metadata[key]
        for key in ("fold_id", "train_start", "train_end", "test_start", "test_end")
    }
    payload = {
            "protocol_version": PROTOCOL_VERSION,
            "catboost_version": catboost.__version__,
            "catboost_params": merged_catboost_params(candidate_params),
            "width_bps": int(width_bps),
            "fold": fold,
            "feature_columns": list(X.columns),
            "feature_fingerprint": frame_fingerprint(X),
            "label_fingerprint": frame_fingerprint(y),
            "data_fingerprint": data_fingerprint,
            "regime_fingerprint": frame_fingerprint(regimes),
            "regime_rules": {
                "names": REGIMES,
                "lookback_bars": REGIME_LOOKBACK_BARS,
                "return_threshold": REGIME_RETURN_THRESHOLD,
                "unknown": "drop",
                "training_weight": "equal aggregate per regime",
                "validation_weight": "none",
            },
            "label_tail_trim": LABEL_TAIL_TRIM,
        }
    if model_name is not None:
        payload.pop("catboost_version")
        payload.pop("catboost_params")
        payload["model_name"] = str(model_name)
        payload["model_params"] = dict(candidate_params)
    return _content_hash(payload)


def policy_cache_fingerprint(
    *,
    width_bps: int,
    candidate_id: int,
    prediction_fingerprints: Sequence[str],
    m15_fingerprint: str,
    minute_fingerprint: str,
    fee_bps: float,
) -> str:
    return _content_hash(
        {
            "protocol_version": PROTOCOL_VERSION,
            "width_bps": width_bps,
            "candidate_id": candidate_id,
            "prediction_fingerprints": list(prediction_fingerprints),
            "m15_fingerprint": m15_fingerprint,
            "minute_fingerprint": minute_fingerprint,
            "fee_bps_per_side": fee_bps,
            "simulator_version": SIMULATOR_VERSION,
            "thresholds": TAUS,
            "geometries": GEOMETRIES,
            "fold_simulation": "independent",
            "safe_signal": "t+15min*(max_hold+1)<=fold_end_exclusive",
        }
    )


def stage_prediction_cache_fingerprint(
    *,
    stage: str,
    width_bps: int,
    candidate_params: Mapping[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    regimes: pd.Series,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    data_fingerprint: str,
    model_name: str | None = None,
    lookback_days: int | None = None,
) -> str:
    train_mask = (X.index < train_end) & regimes.reindex(X.index).isin(REGIMES)
    if lookback_days is not None:
        train_mask &= X.index >= train_end - pd.Timedelta(days=int(lookback_days))
    test_mask = (X.index >= test_start) & (X.index < test_end)
    payload = {
            "protocol_version": PROTOCOL_VERSION,
            "catboost_version": catboost.__version__,
            "stage": stage,
            "width_bps": int(width_bps),
            "catboost_params": merged_catboost_params(candidate_params),
            "train_end_exclusive": train_end,
            "test_start": test_start,
            "test_end_exclusive": test_end,
            "lookback_days": lookback_days,
            "training_features": frame_fingerprint(X.loc[train_mask]),
            "training_labels": frame_fingerprint(y.reindex(X.index).loc[train_mask]),
            "training_regimes": frame_fingerprint(regimes.reindex(X.index).loc[train_mask]),
            "test_features": frame_fingerprint(X.loc[test_mask]),
            "test_labels": frame_fingerprint(y.reindex(X.index).loc[test_mask]),
            "data_fingerprint": data_fingerprint,
            "regime_rules": {
                "names": REGIMES,
                "lookback_bars": REGIME_LOOKBACK_BARS,
                "return_threshold": REGIME_RETURN_THRESHOLD,
                "training_weight": "equal aggregate per regime",
            },
            "label_tail_trim": LABEL_TAIL_TRIM,
        }
    if model_name is not None:
        payload.pop("catboost_version")
        payload.pop("catboost_params")
        payload["model_name"] = str(model_name)
        payload["model_params"] = dict(candidate_params)
    return _content_hash(payload)


def continuous_policy_cache_fingerprint(
    *,
    stage: str,
    width_bps: int,
    candidate_id: int,
    prediction_fingerprint: str,
    m15_fingerprint: str,
    minute_fingerprint: str,
    fee_bps: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> str:
    return _content_hash(
        {
            "protocol_version": PROTOCOL_VERSION,
            "stage": stage,
            "width_bps": int(width_bps),
            "candidate_id": int(candidate_id),
            "prediction_fingerprint": prediction_fingerprint,
            "m15_fingerprint": m15_fingerprint,
            "minute_fingerprint": minute_fingerprint,
            "fee_bps_per_side": float(fee_bps),
            "simulator_version": SIMULATOR_VERSION,
            "thresholds": TAUS,
            "geometries": GEOMETRIES,
            "start": start,
            "end_exclusive": end,
            "simulation": "one continuous span; monthly diagnostics sliced",
            "safe_signal": "t+15min*(max_hold+1)<=end_exclusive",
        }
    )

def _metadata_path(path: Path) -> Path:
    return path.with_suffix(".json")


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    temporary.write_text(
        json.dumps(_canonical(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def write_prediction_cache(
    path: str | Path, frame: pd.DataFrame, fingerprint: str
) -> None:
    target = Path(path)
    _atomic_parquet(frame, target)
    _atomic_json(
        {
            "fingerprint": fingerprint,
            "protocol_version": PROTOCOL_VERSION,
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "columns": list(frame.columns),
            "rows": len(frame),
            "content_fingerprint": frame_fingerprint(frame),
        },
        _metadata_path(target),
    )


def load_prediction_cache(
    path: str | Path, fingerprint: str
) -> pd.DataFrame | None:
    target = Path(path)
    metadata_path = _metadata_path(target)
    if not target.exists() or not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("fingerprint") != fingerprint
            or metadata.get("schema_version") != PREDICTION_SCHEMA_VERSION
        ):
            return None
        frame = pd.read_parquet(target)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if tuple(frame.columns) != PREDICTION_COLUMNS:
        return None
    if len(frame) != int(metadata.get("rows", -1)):
        return None
    if frame_fingerprint(frame) != metadata.get("content_fingerprint"):
        return None
    try:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        for column in ("train_start", "train_end", "test_start", "test_end"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        assert_before_boundary(frame.set_index("timestamp"))
    except (TypeError, ValueError):
        return None
    return frame


def _write_table_cache(path: Path, frame: pd.DataFrame, fingerprint: str) -> None:
    _atomic_parquet(frame, path)
    _atomic_json(
        {
            "fingerprint": fingerprint,
            "protocol_version": PROTOCOL_VERSION,
            "schema_version": POLICY_SCHEMA_VERSION,
            "columns": list(frame.columns),
            "rows": len(frame),
            "content_fingerprint": frame_fingerprint(frame),
        },
        _metadata_path(path),
    )


def _load_policy_cache(path: Path, fingerprint: str) -> pd.DataFrame | None:
    metadata_path = _metadata_path(path)
    if not path.exists() or not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame = pd.read_parquet(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    required = {
        "width_bps", "candidate_id", "policy_id", "tau", "tp_bps", "sl_bps",
        "max_hold", "trades", "positive_segments", "robust_score",
    }
    if (
        metadata.get("fingerprint") != fingerprint
        or metadata.get("schema_version") != POLICY_SCHEMA_VERSION
        or len(frame) != len(policy_choices())
        or not required.issubset(frame.columns)
        or list(frame.columns) != metadata.get("columns")
        or frame_fingerprint(frame) != metadata.get("content_fingerprint")
    ):
        return None
    return frame


def assert_before_boundary(
    frame: pd.DataFrame | pd.Series | pd.DatetimeIndex,
    *,
    end_exclusive: pd.Timestamp = FORWARD_END,
) -> None:
    index = frame if isinstance(frame, pd.DatetimeIndex) else pd.DatetimeIndex(frame.index)
    if index.tz is None:
        raise ValueError("timestamps must be timezone-aware")
    boundary = pd.Timestamp(end_exclusive)
    boundary = (
        boundary.tz_localize("UTC")
        if boundary.tzinfo is None
        else boundary.tz_convert("UTC")
    )
    if bool((index.tz_convert("UTC") >= boundary).any()):
        raise ValueError(f"lockbox boundary {boundary.isoformat()} was reached")


def safe_signal_mask(
    index: pd.DatetimeIndex,
    *,
    end_exclusive: pd.Timestamp,
    max_hold: int,
) -> np.ndarray:
    if max_hold < 1:
        raise ValueError("max_hold must be >= 1")
    timestamps = pd.DatetimeIndex(index)
    if timestamps.tz is None:
        raise ValueError("signal timestamps must be timezone-aware")
    boundary = pd.Timestamp(end_exclusive)
    boundary = (
        boundary.tz_localize(timestamps.tz)
        if boundary.tzinfo is None
        else boundary.tz_convert(timestamps.tz)
    )
    return timestamps + pd.Timedelta(minutes=15 * (max_hold + 1)) <= boundary


def regime_balanced_training_weights(regimes: pd.Series) -> pd.Series:
    if regimes.empty or not regimes.isin(REGIMES).all():
        raise ValueError("training regimes must be known bull/sideways/bear rows")
    counts = regimes.value_counts()
    return regimes.map(len(regimes) / (len(counts) * counts)).astype(float)


def fit_fold_predictions(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    regimes: pd.Series,
    fold_metadata: Mapping[str, Any],
    width_bps: int,
    candidate_id: int,
    candidate_params: Mapping[str, Any],
    model_factory: Callable,
    refit_id: str,
) -> pd.DataFrame:
    train_positions = np.asarray(fold_metadata["train_positions"], dtype=int)
    test_positions = np.asarray(fold_metadata["test_positions"], dtype=int)
    X_train = X.iloc[train_positions]
    y_train = y.reindex(X_train.index)
    train_regimes = regimes.reindex(X_train.index)
    known = train_regimes.isin(REGIMES)
    X_train, y_train, train_regimes = (
        X_train.loc[known],
        y_train.loc[known],
        train_regimes.loc[known],
    )
    if len(X_train) <= LABEL_TAIL_TRIM:
        raise ValueError("training fold is too small after label-tail trim")
    X_train = X_train.iloc[:-LABEL_TAIL_TRIM]
    y_train = y_train.iloc[:-LABEL_TAIL_TRIM]
    train_regimes = train_regimes.iloc[:-LABEL_TAIL_TRIM]
    X_test = X.iloc[test_positions]
    y_test = y.reindex(X_test.index)
    if not regimes.reindex(X_test.index).isin(REGIMES).all():
        raise ValueError("unknown regime remained inside validation fold")
    if y_train.nunique() < 2:
        raise ValueError("training fold needs at least two classes")

    model = model_factory(dict(candidate_params))
    weights = regime_balanced_training_weights(train_regimes)
    model.fit(X_train, y_train, sample_weight=weights.to_numpy(dtype=float))
    probabilities = _aligned_proba(model, X_test)
    predicted = np.asarray((0, 1, 2))[probabilities.argmax(axis=1)]
    frame = pd.DataFrame(
        {
            "timestamp": X_test.index,
            "width_bps": width_bps,
            "candidate_id": candidate_id,
            "fold_id": int(fold_metadata["fold_id"]),
            "y_true": y_test.astype(int).to_numpy(),
            "pred": predicted,
            "confidence": probabilities.max(axis=1),
            "p_short": probabilities[:, 0],
            "p_flat": probabilities[:, 1],
            "p_long": probabilities[:, 2],
            "train_start": X_train.index[0],
            "train_end": X_train.index[-1],
            "test_start": pd.Timestamp(fold_metadata["test_start"]),
            "test_end": pd.Timestamp(fold_metadata["test_end"]),
            "refit_id": refit_id,
        }
    )
    return frame.loc[:, PREDICTION_COLUMNS]

def fit_span_predictions(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    regimes: pd.Series,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    width_bps: int,
    candidate_id: int,
    candidate_params: Mapping[str, Any],
    model_factory: Callable,
    fit_id: str,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """Fit once before a stage boundary and predict its untouched span."""
    train_end = pd.Timestamp(train_end)
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)
    train_mask = (X.index < train_end) & regimes.reindex(X.index).isin(REGIMES)
    if lookback_days is not None:
        if int(lookback_days) <= 0:
            raise ValueError("lookback_days must be positive")
        train_mask &= X.index >= train_end - pd.Timedelta(days=int(lookback_days))
    test_mask = (X.index >= test_start) & (X.index < test_end)
    X_train = X.loc[train_mask]
    y_train = y.reindex(X_train.index)
    train_regimes = regimes.reindex(X_train.index)
    X_test = X.loc[test_mask]
    y_test = y.reindex(X_test.index)
    if len(X_train) <= LABEL_TAIL_TRIM:
        raise ValueError("stage training span is too small after label-tail trim")
    if X_test.empty:
        raise ValueError("stage test span is empty")
    X_train = X_train.iloc[:-LABEL_TAIL_TRIM]
    y_train = y_train.iloc[:-LABEL_TAIL_TRIM]
    train_regimes = train_regimes.iloc[:-LABEL_TAIL_TRIM]
    if y_train.nunique() < 2:
        raise ValueError("stage training span needs at least two classes")
    model = model_factory(dict(candidate_params))
    weights = regime_balanced_training_weights(train_regimes)
    model.fit(X_train, y_train, sample_weight=weights.to_numpy(dtype=float))
    probabilities = _aligned_proba(model, X_test)
    predicted = np.asarray((0, 1, 2))[probabilities.argmax(axis=1)]
    frame = pd.DataFrame(
        {
            "timestamp": X_test.index,
            "width_bps": int(width_bps),
            "candidate_id": int(candidate_id),
            "fold_id": -1,
            "y_true": y_test.astype(int).to_numpy(),
            "pred": predicted,
            "confidence": probabilities.max(axis=1),
            "p_short": probabilities[:, 0],
            "p_flat": probabilities[:, 1],
            "p_long": probabilities[:, 2],
            "train_start": X_train.index[0],
            "train_end": X_train.index[-1],
            "test_start": test_start,
            "test_end": test_end,
            "refit_id": fit_id,
        }
    )
    # Parquet cannot store seconds; normalise before hashing the frozen cache.
    for column in ("timestamp", "train_start", "train_end", "test_start", "test_end"):
        frame[column] = frame[column].dt.as_unit("ns")
    return frame.loc[:, PREDICTION_COLUMNS]


def calibration_month_spans() -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    """Return the six half-open 2025-H1 monthly prediction spans."""
    edges = pd.date_range(SELECTION_END, CALIBRATION_END, freq="MS")
    return tuple((left, right) for left, right in zip(edges[:-1], edges[1:]))


def combine_monthly_predictions(
    frames: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Validate and concatenate one causal prediction frame per H1 month."""
    spans = calibration_month_spans()
    if len(frames) != len(spans):
        raise ValueError(f"expected {len(spans)} monthly prediction frames")
    required = {"timestamp", "train_end", "test_start", "test_end", "refit_id"}
    checked = []
    for frame, (start, end) in zip(frames, spans):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"monthly prediction frame misses columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"monthly prediction frame is empty for {start:%Y-%m}")
        current = frame.copy()
        for column in ("timestamp", "train_end", "test_start", "test_end"):
            current[column] = pd.to_datetime(current[column], utc=True)
        in_month = (current["timestamp"] >= start) & (current["timestamp"] < end)
        if not in_month.all():
            raise ValueError(f"monthly predictions cross the {start:%Y-%m} boundary")
        if not current["test_start"].eq(start).all() or not current["test_end"].eq(end).all():
            raise ValueError(f"monthly prediction metadata changed for {start:%Y-%m}")
        if not current["train_end"].lt(current["test_start"]).all():
            raise ValueError("monthly fit used future or current-month training data")
        if current["refit_id"].nunique() != 1:
            raise ValueError(f"monthly fit identity changed inside {start:%Y-%m}")
        checked.append(current)
    combined = pd.concat(checked, ignore_index=True).sort_values("timestamp")
    if combined["timestamp"].duplicated().any():
        raise ValueError("monthly prediction timestamps overlap")
    return combined.reset_index(drop=True)


def forward_reporting_periods() -> tuple[
    tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...],
    tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...],
]:
    month_edges = pd.date_range(CALIBRATION_END, FORWARD_END, freq="MS")
    months = tuple(
        (start.strftime("%Y-%m"), start, end)
        for start, end in zip(month_edges[:-1], month_edges[1:])
    )
    quarters = (
        (
            "2025-Q3",
            pd.Timestamp("2025-07-01", tz="UTC"),
            pd.Timestamp("2025-10-01", tz="UTC"),
        ),
        (
            "2025-Q4",
            pd.Timestamp("2025-10-01", tz="UTC"),
            pd.Timestamp("2026-01-01", tz="UTC"),
        ),
        (
            "2026-Q1",
            pd.Timestamp("2026-01-01", tz="UTC"),
            FORWARD_END,
        ),
    )
    return months, quarters


def validate_frozen_policy_rows(rows: pd.DataFrame) -> None:
    if rows.empty:
        raise ValueError("forward policy rows are empty")
    policy_columns = (
        "candidate_id", "policy_id", "tau", "tp_bps", "sl_bps",
        "max_hold", "fit_id",
    )
    required = {"objective", "width_bps", *policy_columns}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"forward policy rows miss columns: {sorted(missing)}")
    for _, group in rows.groupby(["objective", "width_bps"], sort=False):
        if any(group[column].nunique(dropna=False) != 1 for column in policy_columns):
            raise ValueError("policy or fitted model changed inside forward evaluation")
def _period_evidence_row(
    *,
    label: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    per_bar: pd.Series,
    ledger: pd.DataFrame,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    returns = per_bar.loc[(per_bar.index >= start) & (per_bar.index < end)]
    entries = pd.to_datetime(ledger.get("entry_time", pd.Series(dtype="datetime64[ns, UTC]")), utc=True)
    period_ledger = ledger.loc[(entries >= start) & (entries < end)]
    summary = economics_summary(returns)
    row = {
        key: policy[key]
        for key in (
            "objective", "width_bps", "candidate_id", "policy_id", "tau",
            "tp_bps", "sl_bps", "max_hold", "fit_id",
        )
    }
    row.update(
        {
            "period": label,
            "period_start": start,
            "period_end": end,
            "gross_return": float(period_ledger.get("gross_return", pd.Series(dtype=float)).sum()),
            "net_return": float(summary["net_return_sum"]),
            "sortino": float(summary["sortino"]),
            "sharpe": float(summary["sharpe"]),
            "max_drawdown": float(summary["max_drawdown"]),
            "trades": int(len(period_ledger)),
            "n_long": int((period_ledger.get("side", pd.Series(dtype=int)) == 1).sum()),
            "n_short": int((period_ledger.get("side", pd.Series(dtype=int)) == -1).sum()),
            "long_net": float(
                period_ledger.loc[
                    period_ledger.get("side", pd.Series(index=period_ledger.index, dtype=int)) == 1,
                    "net_return",
                ].sum()
            ) if "net_return" in period_ledger else 0.0,
            "short_net": float(
                period_ledger.loc[
                    period_ledger.get("side", pd.Series(index=period_ledger.index, dtype=int)) == -1,
                    "net_return",
                ].sum()
            ) if "net_return" in period_ledger else 0.0,
        }
    )
    return row


def summarize_forward_evidence(
    *,
    per_bar: pd.Series,
    ledger: pd.DataFrame,
    regimes: pd.Series,
    policy: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Slice one frozen nine-month simulation into report periods."""
    returns = per_bar.sort_index().astype(float)
    if returns.index.tz is None:
        raise ValueError("forward return timestamps must be timezone-aware")
    if not returns.index.is_unique:
        raise ValueError("forward return timestamps must be unique")
    months, quarters = forward_reporting_periods()
    monthly = pd.DataFrame(
        [
            _period_evidence_row(
                label=label,
                start=start,
                end=end,
                per_bar=returns,
                ledger=ledger,
                policy=policy,
            )
            for label, start, end in months
        ]
    )
    quarterly = pd.DataFrame(
        [
            _period_evidence_row(
                label=label,
                start=start,
                end=end,
                per_bar=returns,
                ledger=ledger,
                policy=policy,
            )
            for label, start, end in quarters
        ]
    )
    full = _period_evidence_row(
        label="2025-07_to_2026-03",
        start=CALIBRATION_END,
        end=FORWARD_END,
        per_bar=returns,
        ledger=ledger,
        policy=policy,
    )
    full["positive_months"] = int((monthly["net_return"] > 0.0).sum())
    full["positive_segments"] = full["positive_months"]
    bar_regimes = regimes.reindex(returns.index)
    regime_sortino = {
        regime: float(
            economics_summary(returns.loc[bar_regimes == regime])["sortino"]
        )
        for regime in REGIMES
    }
    full.update(
        {
            "bull_sortino": regime_sortino["bull"],
            "sideways_sortino": regime_sortino["sideways"],
            "bear_sortino": regime_sortino["bear"],
        }
    )
    full["robust_score"] = robust_score(
        pooled_sortino=full["sortino"],
        pooled_sharpe=full["sharpe"],
        bull_sortino=full["bull_sortino"],
        sideways_sortino=full["sideways_sortino"],
        bear_sortino=full["bear_sortino"],
    )
    full["constraint_violation"] = total_constraint_violation(
        {
            "trades": full["trades"],
            "n_long": full["n_long"],
            "n_short": full["n_short"],
            "positive_segments": full["positive_months"],
        },
        n_segments=9,
    )
    return monthly, quarterly, pd.DataFrame([full])
class RecordingToyModel:
    """Deterministic smoke model; its fit never runs CatBoost."""

    fit_calls = 0

    def __init__(self, _params: Mapping[str, Any] | None = None):
        self.classes_ = np.array((0, 1, 2), dtype=int)

    def fit(self, X, y, sample_weight=None):
        type(self).fit_calls += 1
        self.classes_ = np.array(sorted(pd.Series(y).astype(int).unique()))
        self._offset = float(np.asarray(X.iloc[0], dtype=float).sum())
        return self

    def predict_proba(self, X):
        n_classes = len(self.classes_)
        other = 0.0 if n_classes == 1 else 0.2 / (n_classes - 1)
        out = np.full((len(X), n_classes), other, dtype=float)
        values = np.nan_to_num(X.iloc[:, 0].to_numpy(dtype=float) + self._offset)
        selected = np.abs(np.rint(values).astype(np.int64)) % n_classes
        out[np.arange(len(X)), selected] = 1.0 if n_classes == 1 else 0.8
        return out


def _read_before(path: Path, *, end_exclusive: pd.Timestamp) -> pd.DataFrame:
    boundary = pd.Timestamp(end_exclusive).tz_convert("UTC")
    frame = pd.read_parquet(
        path,
        filters=[("timestamp", "<", boundary.to_pydatetime())],
    )
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.sort_index()
    assert_before_boundary(frame, end_exclusive=boundary)
    return frame


def load_prepared_data(
    *,
    widths: Sequence[int],
    config_path: Path = CONFIG_PATH,
    minute_path: Path = MINUTE_PATH,
    sentiment: str = "both",
) -> PreparedData:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bars_path = CODE_ROOT / cfg["instruments"]["btc"]["working_parquet"]
    bars = _read_before(bars_path, end_exclusive=FORWARD_END)
    minute = _read_before(minute_path, end_exclusive=FORWARD_END)
    regimes = past_regime_labels(bars["close"])
    features: dict[int, tuple[pd.DataFrame, pd.Series]] = {}
    reference_index = None
    for width in widths:
        X, y, _ = build_walkforward_xy(
            "btc",
            cfg,
            horizon=1,
            sentiment=sentiment,
            label_fn=lambda frame, w=int(width): make_label(
                frame, threshold_bps=w, horizon=1
            ),
            orderflow=True,
            positioning=True,
            end_exclusive=FORWARD_END,
        )
        y = y.reindex(X.index)
        assert_before_boundary(X)
        if reference_index is None:
            reference_index = X.index
        elif not X.index.equals(reference_index):
            raise ValueError("all widths must use identical 2024 fold rows")
        features[int(width)] = (X, y)
    return PreparedData(
        bars=bars,
        minute=minute,
        features=features,
        regimes=regimes,
        m15_fingerprint=frame_fingerprint(bars),
        minute_fingerprint=frame_fingerprint(minute),
        sentiment_mode=sentiment,
        feature_columns=tuple(next(iter(features.values()))[0].columns),
    )


def _folds(index: pd.DatetimeIndex) -> tuple[dict[str, Any], ...]:
    rows = []
    for fold_id, (train_positions, test_positions) in enumerate(
        fixed_splitter().split(index)
    ):
        rows.append(
            {
                "fold_id": fold_id,
                "train_positions": tuple(int(value) for value in train_positions),
                "test_positions": tuple(int(value) for value in test_positions),
                "train_start": index[train_positions[0]],
                "train_end": index[train_positions[-1]] + BAR_SIZE,
                "test_start": index[test_positions[0]],
                "test_end": index[test_positions[-1]] + BAR_SIZE,
            }
        )
    if len(rows) != 5:
        raise ValueError("matched scoring requires exactly five folds")
    return tuple(rows)


def _macro_f1(actual: pd.Series, predicted: pd.Series) -> float:
    return float(
        f1_score(
            actual.astype(int),
            predicted.astype(int),
            labels=(0, 1, 2),
            average="macro",
            zero_division=0,
        )
    )


class MatchedAblationRunner:
    def __init__(
        self,
        *,
        output_root: str | Path = CACHE_ROOT,
        widths: Sequence[int] = WIDTHS,
        candidates: Sequence[Mapping[str, Any]] | None = None,
        candidate_ids: Sequence[int] | None = None,
        model_factory: Callable = MODELS["catboost_balanced"],
        prepared: PreparedData | None = None,
        smoke: bool = False,
        fee_bps: float = 5.0,
        stage1_only: bool = False,
        handoff_path: str | Path | None = None,
    ):
        self.handoff_path = Path(handoff_path) if handoff_path is not None else None
        self.upstream_handoff_fingerprint = ""
        self.sentiment_mode = "both"
        self.lookback_days: dict[int, int | None] = {
            int(width): None for width in widths
        }
        self.expected_feature_columns: tuple[str, ...] = ()
        if self.handoff_path is not None:
            from experiments.notebook02_handoff import load_pipeline_handoff

            handoff = load_pipeline_handoff(self.handoff_path)
            handoff_widths = tuple(int(width) for width in handoff["widths"])
            if tuple(int(width) for width in widths) != handoff_widths:
                raise ValueError("runner widths do not match the Notebook 02 handoff")
            self.lookback_days = {
                int(width): int(days)
                for width, days in handoff["training_histories_days"].items()
            }
            self.sentiment_mode = "none"
            self.expected_feature_columns = tuple(handoff["features"]["columns"])
            self.upstream_handoff_fingerprint = str(handoff["handoff_fingerprint"])
        self.output_root = Path(output_root)
        self.widths = tuple(int(width) for width in widths)
        pool = load_candidates() if candidates is None else candidates
        self.candidates = tuple(dict(candidate) for candidate in pool)
        self.candidate_ids = (
            tuple(range(len(self.candidates)))
            if candidate_ids is None
            else tuple(int(candidate_id) for candidate_id in candidate_ids)
        )
        if (
            len(self.candidate_ids) != len(self.candidates)
            or len(set(self.candidate_ids)) != len(self.candidate_ids)
            or any(candidate_id < 0 for candidate_id in self.candidate_ids)
        ):
            raise ValueError("candidate IDs must be unique nonnegative pool IDs")
        self._candidate_by_id = dict(zip(self.candidate_ids, self.candidates))
        self.model_factory = model_factory
        self.prepared = prepared
        self.smoke = bool(smoke)
        self.fee_bps = float(fee_bps)
        self.stage1_only = bool(stage1_only)
        self.fits = 0
        self.cache_hits = 0
        self.policy_cache_hits = 0

    def _prediction_path(
        self, width: int, candidate_id: int, fold_id: int, fingerprint: str
    ) -> Path:
        name = (
            f"w{width}_candidate_{candidate_id:02d}_fold_{fold_id:02d}_"
            f"{fingerprint}.parquet"
        )
        return self.output_root / "predictions" / name

    def _policy_path(
        self, width: int, candidate_id: int, fingerprint: str
    ) -> Path:
        name = f"w{width}_candidate_{candidate_id:02d}_{fingerprint}.parquet"
        return self.output_root / "policies" / name

    def _stage_prediction_path(
        self,
        stage: str,
        width: int,
        candidate_id: int,
        fingerprint: str,
    ) -> Path:
        name = f"w{width}_candidate_{candidate_id:02d}_{fingerprint}.parquet"
        return self.output_root / "stage_predictions" / stage / name

    def _continuous_policy_path(
        self,
        stage: str,
        width: int,
        candidate_id: int,
        fingerprint: str,
    ) -> Path:
        name = f"w{width}_candidate_{candidate_id:02d}_{fingerprint}.parquet"
        return self.output_root / "stage_policies" / stage / name

    def _fit_or_load_stage(
        self,
        *,
        stage: str,
        width: int,
        candidate_id: int,
        X: pd.DataFrame,
        y: pd.Series,
        regimes: pd.Series,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
        data_fingerprint: str,
    ) -> tuple[pd.DataFrame, str]:
        params = self._candidate_by_id[candidate_id]
        fingerprint = stage_prediction_cache_fingerprint(
            stage=stage,
            width_bps=width,
            candidate_params=params,
            X=X,
            y=y,
            regimes=regimes,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            data_fingerprint=data_fingerprint,
            lookback_days=self.lookback_days.get(int(width)),
        )
        path = self._stage_prediction_path(
            stage, width, candidate_id, fingerprint
        )
        frame = load_prediction_cache(path, fingerprint)
        if frame is None:
            fit_id = f"{stage}-w{width}-c{candidate_id:02d}-{fingerprint[:12]}"
            frame = fit_span_predictions(
                X=X,
                y=y,
                regimes=regimes,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                width_bps=width,
                candidate_id=candidate_id,
                candidate_params=params,
                model_factory=self.model_factory,
                fit_id=fit_id,
                lookback_days=self.lookback_days.get(int(width)),
            )
            write_prediction_cache(path, frame, fingerprint)
            self.fits += 1
        else:
            self.cache_hits += 1
        return frame, fingerprint

    def _fit_or_load_monthly_calibration(
        self,
        *,
        width: int,
        candidate_id: int,
        X: pd.DataFrame,
        y: pd.Series,
        regimes: pd.Series,
        data_fingerprint: str,
    ) -> tuple[pd.DataFrame, tuple[str, ...]]:
        """Fit at each H1 month boundary and return causal monthly predictions."""
        frames, fingerprints = [], []
        for start, end in calibration_month_spans():
            frame, fingerprint = self._fit_or_load_stage(
                stage=f"calibration_{start:%Y_%m}",
                width=width,
                candidate_id=candidate_id,
                X=X,
                y=y,
                regimes=regimes,
                train_end=start,
                test_start=start,
                test_end=end,
                data_fingerprint=data_fingerprint,
            )
            frames.append(frame)
            fingerprints.append(fingerprint)
        return combine_monthly_predictions(frames), tuple(fingerprints)

    def _simulate_continuous_policy(
        self,
        *,
        prediction_frame: pd.DataFrame,
        prepared: PreparedData,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tau: float,
        tp_bps: int,
        sl_bps: int,
        max_hold: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        indexed = prediction_frame.set_index("timestamp")
        safe = safe_signal_mask(
            indexed.index,
            end_exclusive=end,
            max_hold=max_hold,
        )
        scope_bars = prepared.bars.loc[
            (prepared.bars.index >= start) & (prepared.bars.index < end)
        ]
        ledger, per_bar = simulate_bracket_trades_intrabar(
            scope_bars,
            prepared.minute,
            indexed.loc[safe, "pred"].astype(int),
            indexed.loc[safe, "confidence"].astype(float),
            tau=float(tau),
            tp_bps=float(tp_bps),
            sl_bps=float(sl_bps),
            max_hold=int(max_hold),
            fee_bps=self.fee_bps,
        )
        if not np.isclose(
            float(per_bar.sum()),
            float(ledger["net_return"].sum()),
            atol=1e-10,
        ):
            raise AssertionError("continuous return series does not reconcile to ledger")
        return ledger, per_bar

    def _score_continuous_policies(
        self,
        *,
        stage: str,
        width: int,
        candidate_id: int,
        prediction_frame: pd.DataFrame,
        prepared: PreparedData,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        segment_edges = pd.date_range(start, end, freq="MS")
        rows = []
        for policy_id, (tau, geometry) in enumerate(policy_choices()):
            tp_bps, sl_bps, max_hold = geometry
            ledger, per_bar = self._simulate_continuous_policy(
                prediction_frame=prediction_frame,
                prepared=prepared,
                start=start,
                end=end,
                tau=float(tau),
                tp_bps=int(tp_bps),
                sl_bps=int(sl_bps),
                max_hold=int(max_hold),
            )
            summary = economics_summary(per_bar)
            bar_regimes = prepared.regimes.reindex(per_bar.index)
            regime_sortino = {
                regime: float(
                    economics_summary(per_bar.loc[bar_regimes == regime])["sortino"]
                )
                for regime in REGIMES
            }
            segment_nets = [
                float(per_bar.loc[(per_bar.index >= left) & (per_bar.index < right)].sum())
                for left, right in zip(segment_edges[:-1], segment_edges[1:])
            ]
            row = {
                "stage": stage,
                "width_bps": int(width),
                "candidate_id": int(candidate_id),
                "policy_id": int(policy_id),
                "tau": float(tau),
                "tp_bps": int(tp_bps),
                "sl_bps": int(sl_bps),
                "max_hold": int(max_hold),
                "trades": int(len(ledger)),
                "pooled_gross": float(ledger["gross_return"].sum()),
                "pooled_net": float(ledger["net_return"].sum()),
                "pooled_sortino": float(summary["sortino"]),
                "pooled_sharpe": float(summary["sharpe"]),
                "positive_segments": int(sum(value > 0.0 for value in segment_nets)),
                "n_long": int((ledger["side"] == 1).sum()),
                "n_short": int((ledger["side"] == -1).sum()),
                "long_net": float(ledger.loc[ledger["side"] == 1, "net_return"].sum()),
                "short_net": float(ledger.loc[ledger["side"] == -1, "net_return"].sum()),
                "bull_sortino": regime_sortino["bull"],
                "sideways_sortino": regime_sortino["sideways"],
                "bear_sortino": regime_sortino["bear"],
            }
            row["robust_score"] = robust_score(
                pooled_sortino=row["pooled_sortino"],
                pooled_sharpe=row["pooled_sharpe"],
                bull_sortino=row["bull_sortino"],
                sideways_sortino=row["sideways_sortino"],
                bear_sortino=row["bear_sortino"],
            )
            rows.append(row)
        return pd.DataFrame(rows)
    def _classification_row(
        self,
        *,
        width: int,
        candidate_id: int,
        fold_frames: Sequence[pd.DataFrame],
        regimes: pd.Series,
    ) -> dict[str, Any]:
        combined = pd.concat(fold_frames, ignore_index=True).sort_values("timestamp")
        combined = combined.set_index("timestamp")
        pooled_regimes = regimes.reindex(combined.index)
        fold_scores, fold_overall = [], []
        for frame in fold_frames:
            indexed = frame.set_index("timestamp")
            fold_regimes = regimes.reindex(indexed.index)
            fold_overall.append(_macro_f1(indexed["y_true"], indexed["pred"]))
            fold_scores.append(
                [
                    _macro_f1(
                        indexed.loc[fold_regimes == regime, "y_true"],
                        indexed.loc[fold_regimes == regime, "pred"],
                    )
                    for regime in REGIMES
                ]
            )
        pooled_scores = {
            regime: _macro_f1(
                combined.loc[pooled_regimes == regime, "y_true"],
                combined.loc[pooled_regimes == regime, "pred"],
            )
            for regime in REGIMES
        }
        return {
            "width_bps": width,
            "candidate_id": candidate_id,
            "overall_f1": float(np.mean(fold_overall)),
            "bull_f1": pooled_scores["bull"],
            "sideways_f1": pooled_scores["sideways"],
            "bear_f1": pooled_scores["bear"],
            "robust_f1": robust_f1_score(fold_scores),
        }

    def _score_policies(
        self,
        *,
        width: int,
        candidate_id: int,
        fold_frames: Sequence[pd.DataFrame],
        folds: Sequence[Mapping[str, Any]],
        prepared: PreparedData,
    ) -> pd.DataFrame:
        rows = []
        for policy_id, (tau, geometry) in enumerate(policy_choices()):
            tp_bps, sl_bps, max_hold = geometry
            ledgers, returns, segment_nets = [], [], []
            for fold, prediction_frame in zip(folds, fold_frames):
                indexed = prediction_frame.set_index("timestamp")
                safe = safe_signal_mask(
                    indexed.index,
                    end_exclusive=pd.Timestamp(fold["test_end"]),
                    max_hold=max_hold,
                )
                scope_bars = prepared.bars.loc[
                    (prepared.bars.index >= pd.Timestamp(fold["test_start"]))
                    & (prepared.bars.index < pd.Timestamp(fold["test_end"]))
                ]
                ledger, per_bar = simulate_bracket_trades_intrabar(
                    scope_bars,
                    prepared.minute,
                    indexed.loc[safe, "pred"].astype(int),
                    indexed.loc[safe, "confidence"].astype(float),
                    tau=float(tau),
                    tp_bps=float(tp_bps),
                    sl_bps=float(sl_bps),
                    max_hold=int(max_hold),
                    fee_bps=self.fee_bps,
                )
                ledgers.append(ledger)
                returns.append(per_bar)
                segment_nets.append(float(ledger["net_return"].sum()))
            ledger = pd.concat(ledgers, ignore_index=True)
            per_bar = pd.concat(returns).sort_index()
            summary = economics_summary(per_bar)
            bar_regimes = prepared.regimes.reindex(per_bar.index)
            regime_sortino = {
                regime: float(
                    economics_summary(per_bar.loc[bar_regimes == regime])["sortino"]
                )
                for regime in REGIMES
            }
            row = {
                "width_bps": width,
                "candidate_id": candidate_id,
                "policy_id": policy_id,
                "tau": float(tau),
                "tp_bps": int(tp_bps),
                "sl_bps": int(sl_bps),
                "max_hold": int(max_hold),
                "trades": int(len(ledger)),
                "pooled_gross": float(ledger["gross_return"].sum()),
                "pooled_net": float(ledger["net_return"].sum()),
                "pooled_sortino": float(summary["sortino"]),
                "pooled_sharpe": float(summary["sharpe"]),
                "positive_segments": int(sum(value > 0.0 for value in segment_nets)),
                "n_long": int((ledger["side"] == 1).sum()),
                "n_short": int((ledger["side"] == -1).sum()),
                "long_net": float(
                    ledger.loc[ledger["side"] == 1, "net_return"].sum()
                ),
                "short_net": float(
                    ledger.loc[ledger["side"] == -1, "net_return"].sum()
                ),
                "bull_sortino": regime_sortino["bull"],
                "sideways_sortino": regime_sortino["sideways"],
                "bear_sortino": regime_sortino["bear"],
            }
            row["robust_score"] = robust_score(
                pooled_sortino=row["pooled_sortino"],
                pooled_sharpe=row["pooled_sharpe"],
                bull_sortino=row["bull_sortino"],
                sideways_sortino=row["sideways_sortino"],
                bear_sortino=row["bear_sortino"],
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def _selected_candidates(
        self, classification: pd.DataFrame, winners: pd.DataFrame
    ) -> pd.DataFrame:
        rows = []
        for width in self.widths:
            class_width = classification.loc[classification["width_bps"] == width]
            if 0 in self.candidate_ids:
                baseline = class_width.loc[class_width["candidate_id"] == 0].iloc[0]
                rows.append(
                    {
                        "objective": "baseline",
                        "selection_rule": "fixed candidate 0",
                        "width_bps": width,
                        "candidate_id": 0,
                        "overall_f1": float(baseline["overall_f1"]),
                        "robust_f1": float(baseline["robust_f1"]),
                        "temporary_policy_id": pd.NA,
                    }
                )
            f1_choice = min(
                class_width.to_dict(orient="records"), key=f1_ranking_key
            )
            rows.append(
                {
                    "objective": "F1",
                    "selection_rule": "highest Robust F1, then Overall macro-F1",
                    "width_bps": width,
                    "candidate_id": int(f1_choice["candidate_id"]),
                    "overall_f1": float(f1_choice["overall_f1"]),
                    "robust_f1": float(f1_choice["robust_f1"]),
                    "temporary_policy_id": pd.NA,
                }
            )
            winner_width = winners.loc[winners["width_bps"] == width]
            economic_choice = min(
                winner_width.to_dict(orient="records"),
                key=lambda row: economic_ranking_key(row, n_segments=5),
            )
            selected_classification = class_width.loc[
                class_width["candidate_id"] == int(economic_choice["candidate_id"])
            ].iloc[0]
            rows.append(
                {
                    "objective": "economic",
                    "selection_rule": "economic ranking over fixed candidate pool",
                    "width_bps": width,
                    "candidate_id": int(economic_choice["candidate_id"]),
                    "overall_f1": float(selected_classification["overall_f1"]),
                    "robust_f1": float(selected_classification["robust_f1"]),
                    "temporary_policy_id": int(economic_choice["policy_id"]),
                }
            )
        return pd.DataFrame(rows)

    def _run_later_stages(
        self,
        *,
        prepared: PreparedData,
        selected_candidates: pd.DataFrame,
    ) -> dict[str, Any]:
        logical_rows = selected_candidates.to_dict(orient="records")
        physical_keys = sorted(
            {
                (int(row["width_bps"]), int(row["candidate_id"]))
                for row in logical_rows
            }
        )
        calibration_predictions: dict[tuple[int, int], pd.DataFrame] = {}
        calibration_fingerprints: dict[tuple[int, int], tuple[str, ...]] = {}
        for width, candidate_id in physical_keys:
            X, y = prepared.features[width]
            prediction, fingerprints = self._fit_or_load_monthly_calibration(
                width=width,
                candidate_id=candidate_id,
                X=X,
                y=y,
                regimes=prepared.regimes.reindex(X.index),
                data_fingerprint=prepared.m15_fingerprint,
            )
            calibration_predictions[(width, candidate_id)] = prediction
            calibration_fingerprints[(width, candidate_id)] = fingerprints

        physical_calibration_grids: dict[tuple[int, int], pd.DataFrame] = {}
        for width, candidate_id in physical_keys:
            fingerprint = continuous_policy_cache_fingerprint(
                stage="calibration",
                width_bps=width,
                candidate_id=candidate_id,
                prediction_fingerprint=_content_hash(
                    {
                        "stage": "monthly_h1_calibration",
                        "prediction_fingerprints": calibration_fingerprints[(width, candidate_id)],
                    }
                ),
                m15_fingerprint=prepared.m15_fingerprint,
                minute_fingerprint=prepared.minute_fingerprint,
                fee_bps=self.fee_bps,
                start=SELECTION_END,
                end=CALIBRATION_END,
            )
            path = self._continuous_policy_path(
                "calibration", width, candidate_id, fingerprint
            )
            grid = _load_policy_cache(path, fingerprint)
            if grid is None:
                grid = self._score_continuous_policies(
                    stage="calibration",
                    width=width,
                    candidate_id=candidate_id,
                    prediction_frame=calibration_predictions[(width, candidate_id)],
                    prepared=prepared,
                    start=SELECTION_END,
                    end=CALIBRATION_END,
                )
                _write_table_cache(path, grid, fingerprint)
            else:
                self.policy_cache_hits += 1
            physical_calibration_grids[(width, candidate_id)] = grid

        calibration_grid_rows, selected_policy_rows = [], []
        for logical in logical_rows:
            objective = str(logical["objective"])
            width = int(logical["width_bps"])
            candidate_id = int(logical["candidate_id"])
            prediction = calibration_predictions[(width, candidate_id)]
            monthly_fit_ids = sorted(prediction["refit_id"].astype(str).unique())
            fit_id = "monthly-h1-" + _content_hash({"fit_ids": monthly_fit_ids})[:12]
            grid = physical_calibration_grids[(width, candidate_id)].copy()
            grid.insert(0, "objective", objective)
            grid["fit_id"] = fit_id
            grid["monthly_fit_count"] = len(monthly_fit_ids)
            grid["train_start"] = pd.to_datetime(prediction["train_start"], utc=True).min()
            grid["train_end"] = pd.to_datetime(prediction["train_end"], utc=True).max()
            calibration_grid_rows.append(grid)
            choice = min(
                grid.to_dict(orient="records"),
                key=lambda row: economic_ranking_key(row, n_segments=6),
            )
            selected_policy_rows.append(choice)

        calibration_grid = pd.concat(calibration_grid_rows, ignore_index=True)
        selected_policies = pd.DataFrame(selected_policy_rows)
        _atomic_parquet(
            calibration_grid,
            self.output_root / "calibration_policy_grid_2025h1.parquet",
        )
        _atomic_parquet(
            selected_policies,
            self.output_root / "selected_policies_2025h1.parquet",
        )

        forward_predictions: dict[tuple[int, int], pd.DataFrame] = {}
        forward_fingerprints: dict[tuple[int, int], str] = {}
        for width, candidate_id in physical_keys:
            X, y = prepared.features[width]
            prediction, fingerprint = self._fit_or_load_stage(
                stage="forward",
                width=width,
                candidate_id=candidate_id,
                X=X,
                y=y,
                regimes=prepared.regimes.reindex(X.index),
                train_end=CALIBRATION_END,
                test_start=CALIBRATION_END,
                test_end=FORWARD_END,
                data_fingerprint=prepared.m15_fingerprint,
            )
            forward_predictions[(width, candidate_id)] = prediction
            forward_fingerprints[(width, candidate_id)] = fingerprint

        monthly_frames, quarterly_frames, summary_frames = [], [], []
        audit_root = self.output_root / "forward_evidence"
        for selected_policy in selected_policies.to_dict(orient="records"):
            objective = str(selected_policy["objective"])
            width = int(selected_policy["width_bps"])
            candidate_id = int(selected_policy["candidate_id"])
            prediction = forward_predictions[(width, candidate_id)]
            forward_fit_id = str(prediction["refit_id"].iloc[0])
            ledger, per_bar = self._simulate_continuous_policy(
                prediction_frame=prediction,
                prepared=prepared,
                start=CALIBRATION_END,
                end=FORWARD_END,
                tau=float(selected_policy["tau"]),
                tp_bps=int(selected_policy["tp_bps"]),
                sl_bps=int(selected_policy["sl_bps"]),
                max_hold=int(selected_policy["max_hold"]),
            )
            slug = f"{objective.lower()}_w{width}"
            _atomic_parquet(ledger, audit_root / f"{slug}_ledger.parquet")
            _atomic_parquet(
                per_bar.rename("net_return").rename_axis("timestamp").reset_index(),
                audit_root / f"{slug}_per_bar.parquet",
            )
            frozen_policy = {
                "objective": objective,
                "width_bps": width,
                "candidate_id": candidate_id,
                "policy_id": int(selected_policy["policy_id"]),
                "tau": float(selected_policy["tau"]),
                "tp_bps": int(selected_policy["tp_bps"]),
                "sl_bps": int(selected_policy["sl_bps"]),
                "max_hold": int(selected_policy["max_hold"]),
                "fit_id": forward_fit_id,
            }
            monthly, quarterly, summary = summarize_forward_evidence(
                per_bar=per_bar,
                ledger=ledger,
                regimes=prepared.regimes,
                policy=frozen_policy,
            )
            monthly_frames.append(monthly)
            quarterly_frames.append(quarterly)
            summary_frames.append(summary)

        forward_monthly = pd.concat(monthly_frames, ignore_index=True)
        forward_quarterly = pd.concat(quarterly_frames, ignore_index=True)
        forward_summary = pd.concat(summary_frames, ignore_index=True)
        validate_frozen_policy_rows(forward_monthly)
        logical_count = len(logical_rows)
        expected_counts = {
            "calibration_policy_rows": logical_count * len(policy_choices()),
            "selected_policy_rows": logical_count,
            "forward_monthly_rows": logical_count * 9,
            "forward_quarterly_rows": logical_count * 3,
            "forward_summary_rows": logical_count,
        }
        actual_counts = {
            "calibration_policy_rows": len(calibration_grid),
            "selected_policy_rows": len(selected_policies),
            "forward_monthly_rows": len(forward_monthly),
            "forward_quarterly_rows": len(forward_quarterly),
            "forward_summary_rows": len(forward_summary),
        }
        if actual_counts != expected_counts:
            raise AssertionError(
                f"later-stage artifact mismatch: expected={expected_counts} "
                f"actual={actual_counts}"
            )
        artifacts = {
            "forward_monthly.parquet": forward_monthly,
            "forward_quarterly.parquet": forward_quarterly,
            "forward_summary.parquet": forward_summary,
        }
        for name, frame in artifacts.items():
            _atomic_parquet(frame, self.output_root / name)

        calibration_fit_ids = sorted(
            {
                str(fit_id)
                for frame in calibration_predictions.values()
                for fit_id in frame["refit_id"].unique()
            }
        )
        forward_fit_ids = sorted(
            {
                str(frame["refit_id"].iloc[0])
                for frame in forward_predictions.values()
            }
        )
        manifest = {
            "protocol_version": PROTOCOL_VERSION,
            "upstream_notebook02_handoff_fingerprint": self.upstream_handoff_fingerprint,
            "training_histories_days": self.lookback_days,
            "sentiment": self.sentiment_mode,
            "feature_columns": list(prepared.feature_columns),
            "selection": {
                "start": SELECTION_START,
                "end_exclusive": SELECTION_END,
                "blocking_folds": 5,
            },
            "calibration": {
                "start": SELECTION_END,
                "end_exclusive": CALIBRATION_END,
                "method": "monthly_walk_forward",
                "month_count": len(calibration_month_spans()),
                "logical_records": logical_count,
                "physical_fit_ids": calibration_fit_ids,
                "prediction_fingerprints": sorted(
                    fingerprint
                    for fingerprints in calibration_fingerprints.values()
                    for fingerprint in fingerprints
                ),
            },
            "forward": {
                "start": CALIBRATION_END,
                "end_exclusive": FORWARD_END,
                "logical_records": logical_count,
                "physical_fit_ids": forward_fit_ids,
                "prediction_fingerprints": sorted(forward_fingerprints.values()),
            },
            "sealed_lockbox": True,
            "lockbox_start": FORWARD_END,
            "m15_max_timestamp": prepared.bars.index.max(),
            "minute_max_timestamp": prepared.minute.index.max(),
            "artifact_rows": actual_counts,
        }
        if (
            prepared.bars.index.max() >= FORWARD_END
            or prepared.minute.index.max() >= FORWARD_END
        ):
            raise AssertionError("sealed lockbox boundary reached before manifest write")
        _atomic_json(manifest, self.output_root / "manifest.json")
        return {**actual_counts, "manifest": manifest}
    def run(self) -> dict[str, Any]:
        self.fits = 0
        self.cache_hits = 0
        self.policy_cache_hits = 0
        prepared = self.prepared or load_prepared_data(
            widths=self.widths,
            sentiment=self.sentiment_mode,
        )
        if self.handoff_path is not None:
            if prepared.sentiment_mode != "none":
                raise ValueError("Notebook 02b prepared data must disable sentiment")
            actual_columns = tuple(next(iter(prepared.features.values()))[0].columns)
            if actual_columns != self.expected_feature_columns:
                raise ValueError("prepared feature columns do not match Notebook 02 handoff")
        assert_before_boundary(prepared.bars)
        assert_before_boundary(prepared.minute)
        classification_rows, policy_frames = [], []

        for width in self.widths:
            X_all, y_all = prepared.features[width]
            assert_before_boundary(X_all)
            regimes_all = prepared.regimes.reindex(X_all.index)
            selection = (
                (X_all.index >= SELECTION_START)
                & (X_all.index < SELECTION_END)
            )
            X, y = X_all.loc[selection], y_all.reindex(X_all.index).loc[selection]
            regimes = regimes_all.loc[selection]
            known = regimes.isin(REGIMES)
            X, y, regimes = (
                X.loc[known],
                y.reindex(X.index).loc[known],
                regimes.loc[known],
            )
            folds = _folds(X.index)
            validate_fold_regime_counts(
                [
                    regimes.iloc[list(fold["test_positions"])]
                    .value_counts()
                    .to_dict()
                    for fold in folds
                ]
            )
            for candidate_id, candidate_params in zip(
                self.candidate_ids, self.candidates
            ):
                fold_frames, fold_fingerprints = [], []
                for fold in folds:
                    fingerprint = prediction_cache_fingerprint(
                        width_bps=width,
                        candidate_params=candidate_params,
                        fold_metadata=fold,
                        X=X,
                        y=y,
                        regimes=regimes,
                        data_fingerprint=prepared.m15_fingerprint,
                    )
                    path = self._prediction_path(
                        width, candidate_id, int(fold["fold_id"]), fingerprint
                    )
                    frame = load_prediction_cache(path, fingerprint)
                    if frame is None:
                        refit_id = (
                            f"w{width}-c{candidate_id:02d}-"
                            f"f{int(fold['fold_id']):02d}-{fingerprint[:12]}"
                        )
                        frame = fit_fold_predictions(
                            X=X,
                            y=y,
                            regimes=regimes,
                            fold_metadata=fold,
                            width_bps=width,
                            candidate_id=candidate_id,
                            candidate_params=candidate_params,
                            model_factory=self.model_factory,
                            refit_id=refit_id,
                        )
                        write_prediction_cache(path, frame, fingerprint)
                        self.fits += 1
                    else:
                        self.cache_hits += 1
                    fold_frames.append(frame)
                    fold_fingerprints.append(fingerprint)

                classification_rows.append(
                    self._classification_row(
                        width=width,
                        candidate_id=candidate_id,
                        fold_frames=fold_frames,
                        regimes=prepared.regimes,
                    )
                )
                policy_fingerprint = policy_cache_fingerprint(
                    width_bps=width,
                    candidate_id=candidate_id,
                    prediction_fingerprints=fold_fingerprints,
                    m15_fingerprint=prepared.m15_fingerprint,
                    minute_fingerprint=prepared.minute_fingerprint,
                    fee_bps=self.fee_bps,
                )
                policy_path = self._policy_path(
                    width, candidate_id, policy_fingerprint
                )
                policy_frame = _load_policy_cache(policy_path, policy_fingerprint)
                if policy_frame is None:
                    policy_frame = self._score_policies(
                        width=width,
                        candidate_id=candidate_id,
                        fold_frames=fold_frames,
                        folds=folds,
                        prepared=prepared,
                    )
                    _write_table_cache(
                        policy_path, policy_frame, policy_fingerprint
                    )
                else:
                    self.policy_cache_hits += 1
                policy_frames.append(policy_frame)

        classification = pd.DataFrame(classification_rows)
        policy_grid = pd.concat(policy_frames, ignore_index=True)
        winner_rows = []
        for width in self.widths:
            for candidate_id in self.candidate_ids:
                candidate_grid = policy_grid.loc[
                    (policy_grid["width_bps"] == width)
                    & (policy_grid["candidate_id"] == candidate_id)
                ]
                winner_rows.append(
                    min(
                        candidate_grid.to_dict(orient="records"),
                        key=lambda row: economic_ranking_key(row, n_segments=5),
                    )
                )
        winners = pd.DataFrame(winner_rows)
        selected = self._selected_candidates(classification, winners)
        artifacts = {
            "classification_2024.parquet": classification,
            "economic_policy_grid_2024.parquet": policy_grid,
            "economic_candidate_winners_2024.parquet": winners,
            "selected_candidates_2024.parquet": selected,
        }
        for name, frame in artifacts.items():
            _atomic_parquet(frame, self.output_root / name)

        expected = {
            "classification_2024.parquet": len(self.widths) * len(self.candidates),
            "economic_policy_grid_2024.parquet": (
                len(self.widths) * len(self.candidates) * len(policy_choices())
            ),
            "economic_candidate_winners_2024.parquet": (
                len(self.widths) * len(self.candidates)
            ),
            "selected_candidates_2024.parquet": (
                len(self.widths) * (3 if 0 in self.candidate_ids else 2)
            ),
        }
        actual = {name: len(frame) for name, frame in artifacts.items()}
        if actual != expected:
            raise AssertionError(
                f"stage-1 artifact row-count mismatch: expected={expected} actual={actual}"
            )
        if (
            not self.smoke
            and len(self.widths) == len(WIDTHS)
            and len(self.candidates) == 15
        ):
            full_expected = {
                "classification_2024.parquet": 45,
                "economic_policy_grid_2024.parquet": 1485,
                "economic_candidate_winners_2024.parquet": 45,
                "selected_candidates_2024.parquet": 9,
            }
            if actual != full_expected:
                raise AssertionError(f"full stage-1 artifact mismatch: {actual}")

        later_result = {}
        if not self.stage1_only:
            later_result = self._run_later_stages(
                prepared=prepared,
                selected_candidates=selected,
            )
        result = {
            "status": (
                "stage_1_complete"
                if self.stage1_only
                else ("smoke_complete" if self.smoke else "complete")
            ),
            "protocol_version": PROTOCOL_VERSION,
            "widths": list(self.widths),
            "candidate_count": len(self.candidates),
            "fold_count": 5,
            "fits": self.fits,
            "cache_hits": self.cache_hits,
            "policy_cache_hits": self.policy_cache_hits,
            "classification_rows": len(classification),
            "policy_rows": len(policy_grid),
            "winner_rows": len(winners),
            "selected_rows": len(selected),
            "lockbox_end_exclusive": FORWARD_END.isoformat(),
            **{key: value for key, value in later_result.items() if key != "manifest"},
        }
        _atomic_json(result, self.output_root / "result.json")
        return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--width", type=int, choices=WIDTHS, default=55)
    parser.add_argument("--candidate-id", type=int, default=0)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--handoff", type=Path)
    args = parser.parse_args(argv)
    all_candidates = load_candidates()
    if args.full:
        widths, candidates = WIDTHS, all_candidates
        candidate_ids = tuple(range(len(all_candidates)))
        model_factory = MODELS["catboost_balanced"]
        output_root = args.output_root or CACHE_ROOT
    else:
        if not 0 <= args.candidate_id < len(all_candidates):
            parser.error("candidate-id is outside tracked candidate pool")
        widths = WIDTHS if args.handoff is not None else (args.width,)
        candidates = (all_candidates[args.candidate_id],)
        candidate_ids = (args.candidate_id,)
        model_factory = RecordingToyModel
        output_root = args.output_root or (CACHE_ROOT / "smoke")
    runner = MatchedAblationRunner(
        output_root=output_root,
        widths=widths,
        candidates=candidates,
        candidate_ids=candidate_ids,
        model_factory=model_factory,
        smoke=args.smoke,
        handoff_path=args.handoff,
    )
    print(json.dumps(_canonical(runner.run()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
