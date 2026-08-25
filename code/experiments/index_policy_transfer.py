"""Frozen BTC-to-index policy controls with explicit source provenance.

The controls answer a narrow transfer question.  They do not tune a BTC
policy on index outcomes: the literal arm keeps every BTC Union parameter,
while the sensitivity arm scales only basis-point geometry by a volatility
ratio estimated from consecutive 2024 M15 returns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from experiments.catboost_execution_scoring import simulate_policy
from experiments.index_replication import (
    CUTOFF,
    FORWARD_END,
    FORWARD_START,
    IndexReplicationConfig,
    _span_fold,
    fit_prediction_frame,
    make_index_label,
)
from experiments.index_replication_protocol import CALIBRATION_START, daily_economics
from features.build import FEATURE_COLS, add_features
from models.zoo import MODELS


PROTOCOL_VERSION = "index-btc-policy-transfer-v2-bounded-hash"
BAR_SIZE = pd.Timedelta(minutes=15)
CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CODE_ROOT / "data"
CACHE_ROOT = CODE_ROOT / "experiments" / "cache"
OUTPUT_ROOT = CACHE_ROOT / "index_policy_transfer"
SOURCE_ROOT = (
    "tuning/all_model_sentiment_policy_180d_fixed15_monthly_h1/none"
)
H1_MIN_TRADES = 50
H1_MIN_SIDE_TRADES = 15
H1_MIN_POSITIVE_SEGMENTS = 4
STAGES = {
    "h1_2025": (CALIBRATION_START, FORWARD_START),
    "forward": (FORWARD_START, FORWARD_END),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_frame_sha256(frame: pd.DataFrame) -> str:
    """Hash a caller-bounded frame without reopening its full source file."""
    current = frame.sort_index().copy()
    values = pd.util.hash_pandas_object(current, index=True).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256(values.tobytes())
    digest.update(
        json.dumps(
            {
                "columns": list(current.columns),
                "dtypes": [str(dtype) for dtype in current.dtypes],
                "rows": len(current),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _single_policy_row(
    frame: pd.DataFrame,
    *,
    model_name: str,
    width_bps: int,
    tau: float,
    tp_bps: int,
    sl_bps: int,
    max_hold: int,
    source_name: str,
) -> pd.Series:
    required = {
        "model_name",
        "width_bps",
        "tau",
        "tp_bps",
        "sl_bps",
        "max_hold",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source_name} misses columns: {sorted(missing)}")
    mask = (
        frame["model_name"].eq(model_name)
        & frame["width_bps"].eq(int(width_bps))
        & np.isclose(frame["tau"].astype(float), float(tau), atol=1e-12)
        & frame["tp_bps"].eq(int(tp_bps))
        & frame["sl_bps"].eq(int(sl_bps))
        & frame["max_hold"].eq(int(max_hold))
    )
    matched = frame.loc[mask]
    if len(matched) != 1:
        raise ValueError(
            f"{source_name} must contain one exact frozen row; found {len(matched)}"
        )
    return matched.iloc[0]


def build_btc_policy_registry(
    cache_root: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read, validate, and hash the exact two BTC Union member policies."""
    root = Path(cache_root)
    protocol_path = root / "qualified_union_v1" / "protocol.json"
    if not protocol_path.exists():
        raise FileNotFoundError(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "qualified-union-v1":
        raise ValueError("unexpected BTC Union protocol version")
    if protocol.get("lockbox_2026_q2_used") is not False:
        raise AssertionError("BTC source must not use the Q2-2026 lockbox")
    execution = protocol.get("execution", {})
    required_execution = {
        "fee_bps_per_side": 5.0,
        "max_hold": 1,
        "sl_bps": 100,
        "tp_bps": 200,
    }
    if execution != required_execution:
        raise ValueError(f"unexpected BTC execution geometry: {execution}")
    members = protocol.get("members")
    expected_members = {
        ("lstm", 55, 0.75),
        ("svm_linear", 75, 0.0),
    }
    actual_members = {
        (str(row["model"]), int(row["width_bps"]), float(row["tau"]))
        for row in (members or [])
    }
    if actual_members != expected_members or len(members or []) != 2:
        raise ValueError(f"unexpected BTC Union members: {members}")

    source_hashes = {str(protocol_path): _sha256(protocol_path)}
    rows: list[dict[str, Any]] = []
    for member in members:
        model_name = str(member["model"])
        model_root = root / Path(SOURCE_ROOT) / model_name
        h1_path = model_root / "selected_policies_2025h1.parquet"
        forward_path = model_root / "forward_summary.parquet"
        for path in (h1_path, forward_path):
            if not path.exists():
                raise FileNotFoundError(path)
            source_hashes[str(path)] = _sha256(path)
        frozen = {
            "model_name": model_name,
            "width_bps": int(member["width_bps"]),
            "tau": float(member["tau"]),
            "tp_bps": int(execution["tp_bps"]),
            "sl_bps": int(execution["sl_bps"]),
            "max_hold": int(execution["max_hold"]),
        }
        h1 = _single_policy_row(
            pd.read_parquet(h1_path), **frozen, source_name=str(h1_path)
        )
        forward = _single_policy_row(
            pd.read_parquet(forward_path), **frozen, source_name=str(forward_path)
        )
        h1_eligible = bool(
            int(h1["trades"]) >= H1_MIN_TRADES
            and int(h1["n_long"]) >= H1_MIN_SIDE_TRADES
            and int(h1["n_short"]) >= H1_MIN_SIDE_TRADES
            and int(h1["positive_segments"]) >= H1_MIN_POSITIVE_SEGMENTS
        )
        rows.append(
            {
                **frozen,
                "hold_bars": int(execution["max_hold"]),
                "source_h1_trades": int(h1["trades"]),
                "source_h1_n_long": int(h1["n_long"]),
                "source_h1_n_short": int(h1["n_short"]),
                "source_h1_positive_segments": int(h1["positive_segments"]),
                "source_h1_net_return": float(h1["pooled_net"])
                if "pooled_net" in h1
                else float("nan"),
                "source_h1_eligible": h1_eligible,
                "source_rank_label": (
                    "eligible_union_member"
                    if h1_eligible
                    else "union_member_below_eligibility_floor"
                ),
                "source_forward_trades": int(forward["trades"]),
                "source_forward_net_return": float(forward["net_return"]),
                "source_forward_sortino": float(forward["sortino"]),
                "source_h1_path": str(h1_path),
                "source_forward_path": str(forward_path),
            }
        )
    registry = pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)
    registry_records = json.loads(registry.to_json(orient="records", date_format="iso"))
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "source_union_protocol": str(protocol_path),
        "source_lockbox_2026_q2_used": False,
        "eligibility_floor": {
            "trades": H1_MIN_TRADES,
            "per_side": H1_MIN_SIDE_TRADES,
            "positive_segments": H1_MIN_POSITIVE_SEGMENTS,
        },
        "source_sha256": source_hashes,
        "registry_sha256": _canonical_json_sha256(registry_records),
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        registry.to_parquet(destination / "btc_policy_registry.parquet", index=False)
        (destination / "btc_policy_registry_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return registry, manifest


def volatility_scaled_registry(
    registry: pd.DataFrame, *, ratio: float
) -> pd.DataFrame:
    """Scale only basis-point fields; tau and holding period stay frozen."""
    if not np.isfinite(ratio) or float(ratio) <= 0:
        raise ValueError("volatility ratio must be finite and positive")
    required = {"width_bps", "tp_bps", "sl_bps", "tau", "hold_bars"}
    missing = required.difference(registry.columns)
    if missing:
        raise ValueError(f"registry misses columns: {sorted(missing)}")
    scaled = registry.copy()
    for column in ("width_bps", "tp_bps", "sl_bps"):
        scaled[f"source_{column}"] = scaled[column].astype(int)
        scaled[column] = np.maximum(
            1, np.floor(scaled[column].astype(float) * float(ratio) + 0.5)
        ).astype(int)
    scaled["transfer_variant"] = "volatility_scaled"
    scaled["volatility_ratio"] = float(ratio)
    return scaled


def _consecutive_abs_returns(frame: pd.DataFrame) -> pd.Series:
    if "close" not in frame.columns:
        raise ValueError("price frame must contain close")
    current = frame.copy().sort_index()
    if not isinstance(current.index, pd.DatetimeIndex):
        raise TypeError("price frame index must be a DatetimeIndex")
    index = pd.to_datetime(current.index, utc=True)
    mask = (index >= pd.Timestamp("2024-01-01", tz="UTC")) & (
        index < pd.Timestamp("2025-01-01", tz="UTC")
    )
    scoped = current.loc[mask, "close"].astype(float)
    gaps = scoped.index.to_series().diff().eq(BAR_SIZE)
    returns = scoped.pct_change().abs()
    return returns.loc[gaps & returns.notna()]


def pre2025_volatility_ratio(
    index_prices: pd.DataFrame, btc_prices: pd.DataFrame
) -> dict[str, Any]:
    """Median absolute consecutive M15 volatility ratio from 2024 only."""
    index_returns = _consecutive_abs_returns(index_prices)
    btc_returns = _consecutive_abs_returns(btc_prices)
    if index_returns.empty or btc_returns.empty:
        raise ValueError("2024 consecutive returns are required for both instruments")
    index_scale = float(index_returns.median())
    btc_scale = float(btc_returns.median())
    if not np.isfinite(btc_scale) or btc_scale <= 0:
        raise ValueError("BTC 2024 volatility scale must be positive")
    return {
        "estimation_start": "2024-01-01T00:00:00+00:00",
        "estimation_end_exclusive": "2025-01-01T00:00:00+00:00",
        "metric": "median_absolute_consecutive_m15_simple_return",
        "index_scale": index_scale,
        "btc_scale": btc_scale,
        "ratio": index_scale / btc_scale,
        "index_observations": int(len(index_returns)),
        "btc_observations": int(len(btc_returns)),
    }


def filter_consecutive_signals(
    predictions: pd.DataFrame, bars_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Drop signals whose following model bar is missing across a session gap."""
    if "timestamp" not in predictions.columns:
        raise ValueError("predictions must contain timestamp")
    ordered = pd.DatetimeIndex(pd.to_datetime(bars_index, utc=True)).sort_values()
    next_by_time = pd.Series(ordered[1:], index=ordered[:-1])
    work = predictions.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    following = work["timestamp"].map(next_by_time)
    keep = following.sub(work["timestamp"]).eq(BAR_SIZE)
    return work.loc[keep].reset_index(drop=True)


def combine_union_predictions(
    predictions: Mapping[str, pd.DataFrame],
    taus: Mapping[str, float],
) -> pd.DataFrame:
    """Apply each member gate, then frozen union-with-opposite-signal-veto."""
    members = ("lstm", "svm_linear")
    if set(predictions) != set(members) or set(taus) != set(members):
        raise ValueError("qualified Union requires exactly LSTM and SVM predictions/taus")
    prepared: list[pd.DataFrame] = []
    for model_name in members:
        frame = predictions[model_name]
        required = {"timestamp", "pred", "confidence"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{model_name} predictions miss columns: {sorted(missing)}")
        current = frame.loc[:, ["timestamp", "pred", "confidence"]].copy()
        current["timestamp"] = pd.to_datetime(current["timestamp"], utc=True)
        if current["timestamp"].duplicated().any():
            raise ValueError(f"{model_name} predictions must have unique timestamps")
        current = current.rename(
            columns={
                "pred": f"pred_{model_name}",
                "confidence": f"confidence_{model_name}",
            }
        )
        prepared.append(current)
    merged = prepared[0].merge(
        prepared[1], on="timestamp", how="outer", validate="one_to_one"
    ).sort_values("timestamp")
    effective: dict[str, np.ndarray] = {}
    for model_name in members:
        pred = merged[f"pred_{model_name}"].fillna(1).astype(int).to_numpy()
        confidence = (
            merged[f"confidence_{model_name}"].fillna(0.0).astype(float).to_numpy()
        )
        directional = np.isin(pred, (0, 2)) & (confidence >= float(taus[model_name]))
        effective[model_name] = np.where(directional, pred, 1).astype(int)
        merged[f"effective_pred_{model_name}"] = effective[model_name]
    first, second = (effective[name] for name in members)
    opposite = np.isin(first, (0, 2)) & np.isin(second, (0, 2)) & (first != second)
    union = np.where(opposite, 1, np.where(first != 1, first, second)).astype(int)
    merged["pred"] = union
    merged["confidence"] = np.where(np.isin(union, (0, 2)), 1.0, 0.0)
    merged["opposite_signal_veto"] = opposite
    return merged


def _atomic_parquet(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=index)
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_parquet_range(
    path: Path,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    import pyarrow.parquet as pq

    names = pq.read_schema(path).names
    index_field = "timestamp" if "timestamp" in names else "__index_level_0__"
    filters: list[tuple[str, str, pd.Timestamp]] = []
    if start is not None:
        filters.append((index_field, ">=", start))
    if end is not None:
        filters.append((index_field, "<", end))
    frame = pd.read_parquet(path, filters=filters or None).sort_index()
    frame.index = pd.to_datetime(frame.index, utc=True)
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{path.name} index must be unique and increasing")
    return frame


def materialize_transfer_mapping(
    stream: str,
    *,
    cache_root: str | Path = CACHE_ROOT,
    data_dir: str | Path = DATA_DIR,
    output_base: str | Path = OUTPUT_ROOT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Freeze strict/scaled mappings from BTC and 2024 volatility before H1."""
    config = IndexReplicationConfig.for_stream(
        stream, data_dir=data_dir, output_base=Path(output_base) / "config_probe"
    )
    output = Path(output_base) / stream
    registry, registry_manifest = build_btc_policy_registry(
        cache_root, output_dir=output
    )
    estimation_start = pd.Timestamp("2024-01-01", tz="UTC")
    estimation_end = pd.Timestamp("2025-01-01", tz="UTC")
    index_path = Path(data_dir) / f"{stream}_15min_2021_2026.parquet"
    btc_path = Path(data_dir) / "btcusdt_15min_2021_2026.parquet"
    index_2024 = _read_parquet_range(
        index_path, start=estimation_start, end=estimation_end
    )
    if "complete_bar" in index_2024:
        index_2024 = index_2024.loc[index_2024["complete_bar"].astype(bool)]
    btc_2024 = _read_parquet_range(
        btc_path, start=estimation_start, end=estimation_end
    )
    volatility = pre2025_volatility_ratio(index_2024, btc_2024)
    strict = registry.copy()
    for column in ("width_bps", "tp_bps", "sl_bps"):
        strict[f"source_{column}"] = strict[column].astype(int)
    strict["transfer_variant"] = "strict"
    strict["volatility_ratio"] = 1.0
    scaled = volatility_scaled_registry(registry, ratio=float(volatility["ratio"]))
    mapping = pd.concat([strict, scaled], ignore_index=True)
    key_columns = [
        "model_name",
        "transfer_variant",
        "width_bps",
        "tau",
        "tp_bps",
        "sl_bps",
        "hold_bars",
        "source_width_bps",
        "source_tp_bps",
        "source_sl_bps",
        "volatility_ratio",
    ]
    mapping_hash = _canonical_json_sha256(mapping.loc[:, key_columns].to_dict("records"))
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stream": stream,
        "symbol": config.instrument,
        "mapping_sha256": mapping_hash,
        "btc_registry_sha256": registry_manifest["registry_sha256"],
        "source_sha256": {
            **registry_manifest["source_sha256"],
            str(index_path): _canonical_frame_sha256(index_2024),
            str(btc_path): _canonical_frame_sha256(btc_2024),
        },
        "market_source_sha256_scope": "canonical_predicate_filtered_2024_rows",
        "volatility_mapping": volatility,
        "mapping_frozen_before_h1_read": True,
        "mapping_max_input_timestamp": max(index_2024.index.max(), btc_2024.index.max()),
        "strict_changes": "none",
        "scaled_changes": "width_bps_tp_bps_sl_bps_only",
        "tau_and_hold_frozen": True,
        "q2_2026_loaded": False,
    }
    _atomic_parquet(mapping, output / "mapped_policy_registry.parquet")
    _atomic_json(manifest, output / "mapping_manifest.json")
    return mapping, manifest


def _prediction_fingerprint(
    *,
    stream: str,
    variant: str,
    model_name: str,
    width_bps: int,
    stage: str,
    segment: str,
    fold: Mapping[str, Any],
    mapping_hash: str,
    data_sha256: str,
) -> str:
    return _canonical_json_sha256(
        {
            "protocol_version": PROTOCOL_VERSION,
            "stream": stream,
            "variant": variant,
            "model_name": model_name,
            "width_bps": int(width_bps),
            "stage": stage,
            "segment": segment,
            "fold": {
                key: fold[key]
                for key in ("train_start", "train_end", "test_start", "test_end")
            },
            "mapping_sha256": mapping_hash,
            "data_sha256": data_sha256,
            "feature_columns": list(FEATURE_COLS),
        }
    )


def _fit_cached_prediction(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    fold: Mapping[str, Any],
    stream: str,
    variant: str,
    model_name: str,
    width_bps: int,
    stage: str,
    segment: str,
    output: Path,
    mapping_hash: str,
    data_sha256: str,
) -> pd.DataFrame:
    fingerprint = _prediction_fingerprint(
        stream=stream,
        variant=variant,
        model_name=model_name,
        width_bps=width_bps,
        stage=stage,
        segment=segment,
        fold=fold,
        mapping_hash=mapping_hash,
        data_sha256=data_sha256,
    )
    path = (
        output
        / "predictions"
        / variant
        / model_name
        / f"{stage}_{segment}_w{width_bps}_{fingerprint[:16]}.parquet"
    )
    if path.exists():
        cached = pd.read_parquet(path)
        if cached["fingerprint"].eq(fingerprint).all():
            return cached
    prediction = fit_prediction_frame(
        X=X,
        y=y,
        model_factory=MODELS[model_name],
        model_name=model_name,
        arm=f"btc_transfer_{variant}",
        width_bps=width_bps,
        fold=fold,
        fit_id=f"{stream}-{variant}-{model_name}-{stage}-{segment}-{fingerprint[:12]}",
    )
    prediction["stream"] = stream
    prediction["transfer_variant"] = variant
    prediction["fingerprint"] = fingerprint
    _atomic_parquet(prediction, path)
    return prediction


def _fit_stage_predictions(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    stream: str,
    variant: str,
    model_name: str,
    width_bps: int,
    stage: str,
    output: Path,
    mapping_hash: str,
    data_sha256: str,
) -> pd.DataFrame:
    start, end = STAGES[stage]
    if stage == "h1_2025":
        starts = list(pd.date_range(start, end, freq="MS", inclusive="left"))
        spans: Sequence[tuple[pd.Timestamp, pd.Timestamp, str]] = [
            (
                month,
                min(month + pd.offsets.MonthBegin(1), end),
                month.strftime("%Y_%m"),
            )
            for month in starts
        ]
    else:
        spans = [(start, end, "2025_07_to_2026_03")]
    frames: list[pd.DataFrame] = []
    for test_start, test_end, segment in spans:
        fold = _span_fold(
            X.index,
            train_end=test_start,
            test_start=test_start,
            test_end=test_end,
        )
        frames.append(
            _fit_cached_prediction(
                X=X,
                y=y,
                fold=fold,
                stream=stream,
                variant=variant,
                model_name=model_name,
                width_bps=width_bps,
                stage=stage,
                segment=segment,
                output=output,
                mapping_hash=mapping_hash,
                data_sha256=data_sha256,
            )
        )
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def _expected_fit_failure(error: Exception) -> bool:
    """Recognise only preregistered sparse-label failures, never code failures."""
    message = str(error).lower()
    return isinstance(error, ValueError) and (
        "less than 3 examples for at least one class" in message
        or "training span needs at least two classes" in message
    )


def _unavailable_economics(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    return {
        "calendar_days": int((pd.Timestamp(end) - pd.Timestamp(start)) / pd.Timedelta(days=1)),
        "trades": 0,
        "n_long": 0,
        "n_short": 0,
        "trades_per_day": 0.0,
        "gross_return": float("nan"),
        "cost_return": float("nan"),
        "net_return": float("nan"),
        "net_bps_per_trade": float("nan"),
        "exposure": float("nan"),
        "daily_sharpe": float("nan"),
        "daily_sortino": float("nan"),
        "max_drawdown": float("nan"),
    }


def _replay(
    *,
    bars: pd.DataFrame,
    minute: pd.DataFrame,
    prediction: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    tau: float,
    tp_bps: int,
    sl_bps: int,
    hold_bars: int,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.Series]:
    safe = filter_consecutive_signals(prediction, bars.index)
    return simulate_policy(
        bars=bars,
        execution=minute,
        prediction_frame=safe,
        start=start,
        end=end,
        resolution="1m",
        tau=float(tau),
        tp_bps=int(tp_bps),
        sl_bps=int(sl_bps),
        max_hold=int(hold_bars),
        fee_bps=float(cost_bps) / 2.0,
    )


def run_index_policy_transfer(
    stream: str,
    *,
    mapping: pd.DataFrame | None = None,
    mapping_manifest: Mapping[str, Any] | None = None,
    cache_root: str | Path = CACHE_ROOT,
    data_dir: str | Path = DATA_DIR,
    output_base: str | Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    """Fit price-only index members and replay strict/scaled H1/forward controls."""
    config = IndexReplicationConfig.for_stream(
        stream, data_dir=data_dir, output_base=Path(output_base) / "config_probe"
    )
    output = Path(output_base) / stream
    if mapping is None or mapping_manifest is None:
        mapping, mapping_manifest = materialize_transfer_mapping(
            stream,
            cache_root=cache_root,
            data_dir=data_dir,
            output_base=output_base,
        )
    mapping_hash = str(mapping_manifest["mapping_sha256"])
    m15_path = Path(data_dir) / f"{stream}_15min_2021_2026.parquet"
    m1_path = Path(data_dir) / f"{stream}_1m_2021_2026.parquet"
    bars = _read_parquet_range(m15_path, end=CUTOFF)
    minute = _read_parquet_range(m1_path, end=CUTOFF)
    if "complete_bar" in bars:
        bars = bars.loc[bars["complete_bar"].astype(bool)]
    if "complete_bar" in minute:
        minute = minute.loc[minute["complete_bar"].astype(bool)]
    if bars.empty or minute.empty or max(bars.index.max(), minute.index.max()) >= CUTOFF:
        raise AssertionError("index transfer source crossed or missed Q2 boundary")
    features = add_features(bars).loc[:, FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    m15_data_sha256 = _canonical_frame_sha256(bars)
    m1_execution_sha256 = _canonical_frame_sha256(minute)
    datasets: dict[int, tuple[pd.DataFrame, pd.Series]] = {}
    predictions: dict[tuple[str, str, str], pd.DataFrame | None] = {}
    fit_failures: list[dict[str, Any]] = []
    total_fits = len(mapping) * len(STAGES)
    completed = 0
    for policy in mapping.to_dict("records"):
        variant = str(policy["transfer_variant"])
        model_name = str(policy["model_name"])
        width = int(policy["width_bps"])
        if width not in datasets:
            label = make_index_label(bars, width).reindex(features.index)
            valid = features.notna().all(axis=1) & label.notna() & label.ne(-1)
            datasets[width] = (features.loc[valid].astype(float), label.loc[valid].astype(int))
        X, y = datasets[width]
        for stage in STAGES:
            key = (variant, model_name, stage)
            try:
                predictions[key] = _fit_stage_predictions(
                    X=X,
                    y=y,
                    stream=stream,
                    variant=variant,
                    model_name=model_name,
                    width_bps=width,
                    stage=stage,
                    output=output,
                    mapping_hash=mapping_hash,
                    data_sha256=m15_data_sha256,
                )
                fit_status = "completed"
            except Exception as error:
                if not _expected_fit_failure(error):
                    raise
                predictions[key] = None
                fit_status = "fit_failed_sparse_class"
                fit_failures.append(
                    {
                        "stream": stream,
                        "transfer_variant": variant,
                        "model_name": model_name,
                        "stage": stage,
                        "width_bps": width,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "fail_closed": True,
                    }
                )
            completed += 1
            _atomic_json(
                {
                    "stage": "model_predictions",
                    "completed": completed,
                    "total": total_fits,
                    "stream": stream,
                    "variant": variant,
                    "model_name": model_name,
                    "evaluation_stage": stage,
                    "fit_status": fit_status,
                    "fit_failures": len(fit_failures),
                    "q2_2026_loaded": False,
                },
                output / "run_state.json",
            )
            print(
                f"[{stream}] BTC transfer predictions {completed}/{total_fits}: "
                f"{variant}/{model_name}/{stage} [{fit_status}]",
                flush=True,
            )

    summary_rows: list[dict[str, Any]] = []
    for variant in ("strict", "volatility_scaled"):
        variant_rows = mapping.loc[mapping["transfer_variant"].eq(variant)]
        if set(variant_rows["model_name"]) != {"lstm", "svm_linear"}:
            raise AssertionError("transfer mapping lost a Union member")
        if variant_rows[["tp_bps", "sl_bps", "hold_bars"]].drop_duplicates().shape[0] != 1:
            raise AssertionError("Union member execution geometry must match")
        geometry = variant_rows.iloc[0]
        taus = variant_rows.set_index("model_name")["tau"].astype(float).to_dict()
        for stage, (start, end) in STAGES.items():
            member_predictions: dict[str, pd.DataFrame | None] = {
                model_name: predictions[(variant, model_name, stage)]
                for model_name in ("lstm", "svm_linear")
            }
            evaluation_rows = [
                (
                    str(policy["model_name"]),
                    member_predictions[str(policy["model_name"])],
                    float(policy["tau"]),
                    bool(policy["source_h1_eligible"]),
                    str(policy["source_rank_label"]),
                    (
                        ""
                        if member_predictions[str(policy["model_name"])] is not None
                        else next(
                            row["error_message"]
                            for row in fit_failures
                            if row["transfer_variant"] == variant
                            and row["model_name"] == str(policy["model_name"])
                            and row["stage"] == stage
                        )
                    ),
                )
                for policy in variant_rows.to_dict("records")
            ]
            available_members = {
                name: frame for name, frame in member_predictions.items() if frame is not None
            }
            union_prediction = (
                combine_union_predictions(available_members, taus)
                if len(available_members) == 2
                else None
            )
            evaluation_rows.append(
                (
                    "qualified_union",
                    union_prediction,
                    0.0,
                    True,
                    "frozen_two_member_union",
                    "" if union_prediction is not None else "one_or_more_union_members_unavailable",
                )
            )
            for (
                model_name,
                prediction,
                tau,
                source_eligible,
                source_label,
                unavailable_reason,
            ) in evaluation_rows:
                common_row = {
                    "stream": stream,
                    "transfer_variant": variant,
                    "stage": stage,
                    "model_name": model_name,
                    "tau": tau,
                    "tp_bps": int(geometry["tp_bps"]),
                    "sl_bps": int(geometry["sl_bps"]),
                    "hold_bars": int(geometry["hold_bars"]),
                    "source_h1_eligible": source_eligible,
                    "source_rank_label": source_label,
                }
                if prediction is None:
                    summary_rows.append(
                        {
                            **common_row,
                            "status": "unavailable_fail_closed",
                            "unavailable_reason": unavailable_reason,
                            **_unavailable_economics(start, end),
                            "stress_2x_net_return": float("nan"),
                            "stress_2x_daily_sortino": float("nan"),
                        }
                    )
                    continue
                ledger, per_bar = _replay(
                    bars=bars,
                    minute=minute,
                    prediction=prediction,
                    start=start,
                    end=end,
                    tau=tau,
                    tp_bps=int(geometry["tp_bps"]),
                    sl_bps=int(geometry["sl_bps"]),
                    hold_bars=int(geometry["hold_bars"]),
                    cost_bps=config.cost_bps,
                )
                stress_ledger, stress_per_bar = _replay(
                    bars=bars,
                    minute=minute,
                    prediction=prediction,
                    start=start,
                    end=end,
                    tau=tau,
                    tp_bps=int(geometry["tp_bps"]),
                    sl_bps=int(geometry["sl_bps"]),
                    hold_bars=int(geometry["hold_bars"]),
                    cost_bps=2.0 * config.cost_bps,
                )
                economics = daily_economics(ledger, per_bar, start=start, end=end)
                stress = daily_economics(
                    stress_ledger, stress_per_bar, start=start, end=end
                )
                summary_rows.append(
                    {
                        **common_row,
                        "status": "completed",
                        "unavailable_reason": "",
                        **economics,
                        "stress_2x_net_return": stress["net_return"],
                        "stress_2x_daily_sortino": stress["daily_sortino"],
                    }
                )
                ledger = ledger.assign(
                    stream=stream,
                    transfer_variant=variant,
                    stage=stage,
                    model_name=model_name,
                )
                _atomic_parquet(
                    ledger,
                    output / "ledgers" / variant / stage / f"{model_name}.parquet",
                )
                _atomic_parquet(
                    per_bar.rename("net_return").rename_axis("timestamp").reset_index(),
                    output
                    / "ledgers"
                    / variant
                    / stage
                    / f"{model_name}_per_bar.parquet",
                )
    summary = pd.DataFrame(summary_rows)
    _atomic_parquet(summary, output / "transfer_summary.parquet")
    _atomic_parquet(pd.DataFrame(fit_failures), output / "fit_failures.parquet")
    completed_predictions = [frame for frame in predictions.values() if frame is not None]
    if not completed_predictions:
        raise RuntimeError("every BTC transfer control was unavailable")
    max_prediction = max(
        pd.to_datetime(frame["timestamp"], utc=True).max() for frame in completed_predictions
    )
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "stream": stream,
        "mapping_sha256": mapping_hash,
        "m15_source_sha256": m15_data_sha256,
        "m1_execution_source_sha256": m1_execution_sha256,
        "market_source_sha256_scope": "canonical_predicate_filtered_pre_q2_rows",
        "mapping_frozen_before_h1_read": bool(
            mapping_manifest["mapping_frozen_before_h1_read"]
        ),
        "variants": ["strict", "volatility_scaled"],
        "models": ["lstm", "svm_linear", "qualified_union"],
        "summary_rows": int(len(summary)),
        "completed_summary_rows": int(summary["status"].eq("completed").sum()),
        "unavailable_summary_rows": int(summary["status"].ne("completed").sum()),
        "fit_failure_rows": int(len(fit_failures)),
        "max_input_timestamp": max(bars.index.max(), minute.index.max()),
        "max_prediction_timestamp": max_prediction,
        "price_only": True,
        "vix_used": False,
        "sentiment_used": False,
        "h1_or_forward_used_for_mapping": False,
        "q2_2026_loaded": False,
    }
    _atomic_json(result, output / "result.json")
    _atomic_json({"stage": "complete", **result}, output / "run_state.json")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--streams", nargs="+", choices=("usa500", "usatech"), default=["usa500", "usatech"]
    )
    parser.add_argument("--mapping-only", action="store_true")
    args = parser.parse_args(argv)
    frozen: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
    for stream in args.streams:
        frozen[stream] = materialize_transfer_mapping(stream)
        print(f"[{stream}] BTC policy mapping frozen before H1", flush=True)
    if args.mapping_only:
        return 0
    for stream in args.streams:
        mapping, manifest = frozen[stream]
        result = run_index_policy_transfer(
            stream, mapping=mapping, mapping_manifest=manifest
        )
        print(json.dumps(_json_safe(result), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
