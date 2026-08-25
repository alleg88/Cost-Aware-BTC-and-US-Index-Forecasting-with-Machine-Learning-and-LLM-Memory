"""Immutable Union-plus-coverage opportunities for Reflection Agent v3."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.qualified_union import (
    FEE_BPS,
    MAX_HOLD,
    MEMBERS,
    SL_BPS,
    TP_BPS,
    load_member_panel,
    prediction_paths,
)
from experiments.run_unified_2021_ensemble import (
    SourcePaths,
    _read_bounded,
    _source_identity,
    load_bounded_sources,
)
from experiments.unified_2021_ensemble_data import build_unified_decisions


CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEVELOPMENT_CACHE = (
    CODE_ROOT / "experiments" / "cache" / "union_v1_episode_reentry"
)
DEFAULT_UNION_CACHE = CODE_ROOT / "experiments" / "cache" / "qualified_union_v1"
Q2_START = pd.Timestamp("2026-04-01", tz="UTC")
BAR_INTERVAL = pd.Timedelta(minutes=15)

_STAGES = {
    "h1": (
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-07-01", tz="UTC"),
    ),
    "forward": (
        pd.Timestamp("2025-07-01", tz="UTC"),
        Q2_START,
    ),
}

OUTPUT_COLUMNS = (
    "opportunity_id",
    "stage",
    "source_role",
    "fold_id",
    "row_key",
    "source_artifact_hash",
    "decision_time",
    "feature_available_time",
    "outcome_available_time",
    "entry_time",
    "exit_time",
    "route",
    "side",
    "confidence_tier",
    "signal_run_bucket",
    "gross_return",
    "net_return",
    "round_trip_cost",
    "exit_reason",
    "vol_regime",
    "trend_regime",
    "funding_regime",
    "oi_regime",
    "path_complete",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_artifacts(
    root: Path, manifest_name: str, names: Iterable[str]
) -> dict[str, str]:
    manifest_path = root / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("artifact_hashes", {})
    verified: dict[str, str] = {}
    for name in names:
        path = (root / name).resolve()
        if path.parent != root.resolve() or not path.is_file():
            raise AssertionError(f"required immutable artifact is missing: {name}")
        actual = _sha256_file(path)
        if expected.get(name) != actual:
            raise AssertionError(f"immutable artifact hash mismatch: {name}")
        verified[name] = actual
    verified[manifest_name] = _sha256_file(manifest_path)
    return verified


def _z_regime(value: object, low: str, middle: str, high: str) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(numeric):
        return "MISSING"
    if float(numeric) <= -0.5:
        return low
    if float(numeric) >= 0.5:
        return high
    return middle


def regime_tags(row: pd.Series) -> dict[str, str]:
    """Map preregistered past-only values to the fixed v3 categories."""

    vol = pd.to_numeric(pd.Series([row.get("vol_z")]), errors="coerce").iloc[0]
    trend = pd.to_numeric(
        pd.Series([row.get("channel_slope_20")]), errors="coerce"
    ).iloc[0]
    if not np.isfinite(vol):
        raise ValueError("vol_z must be finite before fixed regime mapping")
    if not np.isfinite(trend):
        raise ValueError("channel_slope_20 must be finite before fixed regime mapping")
    return {
        "vol_regime": "LOW" if vol <= -0.5 else "HIGH" if vol >= 0.5 else "NORMAL",
        "trend_regime": "DOWN" if trend <= -2.0 else "UP" if trend >= 2.0 else "FLAT",
        "funding_regime": _z_regime(
            row.get("funding_z"), "NEGATIVE", "NEUTRAL", "POSITIVE"
        ),
        "oi_regime": _z_regime(row.get("oi_z"), "FALLING", "FLAT", "RISING"),
    }


def coverage_candidate_mask(frame: pd.DataFrame) -> pd.Series:
    """Return the frozen, outcome-blind Union-flat LSTM candidate predicate."""

    required = {
        "union_signal",
        "pred_lstm",
        "pred_svm_linear",
        "p_short_lstm",
        "p_flat_lstm",
        "p_long_lstm",
        "path_complete",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"candidate frame lacks columns: {missing}")
    side = frame["pred_lstm"].map({0: -1, 1: 0, 2: 1})
    svm_side = frame["pred_svm_linear"].map({0: -1, 1: 0, 2: 1})
    if side.isna().any() or svm_side.isna().any():
        raise ValueError("member predictions must use class labels 0, 1 or 2")
    confidence = frame[
        ["p_short_lstm", "p_flat_lstm", "p_long_lstm"]
    ].max(axis=1)
    return (
        frame["union_signal"].eq(0)
        & side.ne(0)
        & confidence.ge(0.60)
        & confidence.lt(0.75)
        & ~(svm_side.ne(0) & svm_side.ne(side))
        & frame["path_complete"].astype(bool)
    )


def _confidence_tier(confidence: pd.Series) -> pd.Series:
    tier = pd.Series(pd.NA, index=confidence.index, dtype="string")
    tier.loc[confidence.ge(0.70) & confidence.lt(0.75)] = "HIGH_EXTRA"
    tier.loc[confidence.ge(0.65) & confidence.lt(0.70)] = "MID_EXTRA"
    tier.loc[confidence.ge(0.60) & confidence.lt(0.65)] = "LOW_EXTRA"
    return tier


def _signal_run_buckets(frame: pd.DataFrame, candidate_mask: pd.Series) -> pd.Series:
    side = frame["pred_lstm"].map({0: -1, 1: 0, 2: 1}).astype(int)
    decision_time = pd.to_datetime(frame["decision_time"], utc=True)
    output = pd.Series(pd.NA, index=frame.index, dtype="string")
    ordered = frame.assign(
        _candidate=candidate_mask,
        _side=side,
        _decision_time=decision_time,
    ).sort_values(["fold_id", "_decision_time"], kind="stable")
    for _, fold in ordered.groupby("fold_id", sort=True):
        run = 0
        previous_side: int | None = None
        previous_time: pd.Timestamp | None = None
        previous_was_candidate = False
        for index in fold.index:
            if not bool(fold.at[index, "_candidate"]):
                run = 0
                previous_side = None
                previous_time = None
                previous_was_candidate = False
                continue
            current_time = pd.Timestamp(fold.at[index, "_decision_time"])
            current_side = int(fold.at[index, "_side"])
            is_continuation = (
                previous_was_candidate
                and previous_side == current_side
                and previous_time is not None
                and current_time - previous_time == BAR_INTERVAL
            )
            run = run + 1 if is_continuation else 1
            output.at[index] = (
                "FIRST" if run == 1 else "SECOND" if run == 2 else "THIRD_PLUS"
            )
            previous_side = current_side
            previous_time = current_time
            previous_was_candidate = True
    return output


def _normalize_exit_reason(value: object) -> str:
    normalized = str(value).strip().lower()
    mapping = {
        "take_profit": "TAKE_PROFIT",
        "tp": "TAKE_PROFIT",
        "stop_loss": "STOP_LOSS",
        "sl": "STOP_LOSS",
        "timeout": "TIMEOUT",
    }
    if normalized not in mapping:
        raise ValueError(f"unknown exit reason: {value!r}")
    return mapping[normalized]


def _replay_independent_candidate_paths(
    candidates: pd.DataFrame,
    *,
    bars: pd.DataFrame,
    minute: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve every candidate independently with frozen Union execution."""

    if MAX_HOLD != 1:
        raise AssertionError("v3 independent replay is frozen to one M15 hold bar")
    if bars.index.tz is None or minute.index.tz is None:
        raise ValueError("execution sources must be timezone-aware")
    minute_ns = minute.index.as_unit("ns").view("int64")
    minute_high = minute["high"].to_numpy(float)
    minute_low = minute["low"].to_numpy(float)
    minute_close = minute["close"].to_numpy(float)
    bar_open = bars["open"].to_numpy(float)
    bar_positions = pd.Series(
        np.arange(len(bars), dtype=int), index=pd.DatetimeIndex(bars.index)
    )
    one_minute_ns = int(pd.Timedelta(minutes=1).as_unit("ns").value)
    hold_ns = int(BAR_INTERVAL.as_unit("ns").value)
    round_trip_cost = 2.0 * float(FEE_BPS) / 10_000.0
    records: list[dict[str, object]] = []

    for row in candidates.itertuples(index=False):
        decision_time = pd.Timestamp(row.decision_time)
        if decision_time not in bar_positions.index:
            raise AssertionError(f"candidate signal bar is missing: {decision_time}")
        signal_position = int(bar_positions.at[decision_time])
        entry_position = signal_position + 1
        if entry_position >= len(bars):
            raise AssertionError(f"candidate entry bar is missing: {decision_time}")
        entry_time = pd.Timestamp(bars.index[entry_position])
        entry_ns = int(entry_time.as_unit("ns").value)
        end_ns = entry_ns + hold_ns
        lo = int(np.searchsorted(minute_ns, entry_ns, side="left"))
        hi = int(np.searchsorted(minute_ns, end_ns, side="left"))
        observed = minute_ns[lo:hi]
        expected = np.arange(entry_ns, end_ns, one_minute_ns, dtype=np.int64)
        if not np.array_equal(observed, expected):
            raise AssertionError(
                f"candidate lacks a complete frozen M1 path: {decision_time}"
            )

        side = 1 if str(row.side) == "LONG" else -1
        entry_price = float(bar_open[entry_position])
        tp_distance = float(TP_BPS) / 10_000.0
        sl_distance = float(SL_BPS) / 10_000.0
        if side == 1:
            tp_price = entry_price * (1.0 + tp_distance)
            sl_price = entry_price * (1.0 - sl_distance)
        else:
            tp_price = entry_price * (1.0 - tp_distance)
            sl_price = entry_price * (1.0 + sl_distance)

        exit_ns = int(observed[-1])
        exit_price = float(minute_close[hi - 1])
        exit_reason = "timeout"
        for position in range(lo, hi):
            hit_stop = (
                minute_low[position] <= sl_price
                if side == 1
                else minute_high[position] >= sl_price
            )
            hit_profit = (
                minute_high[position] >= tp_price
                if side == 1
                else minute_low[position] <= tp_price
            )
            if hit_stop:
                exit_ns = int(minute_ns[position])
                exit_price = sl_price
                exit_reason = "stop_loss"
                break
            if hit_profit:
                exit_ns = int(minute_ns[position])
                exit_price = tp_price
                exit_reason = "take_profit"
                break

        gross_return = side * (exit_price / entry_price - 1.0)
        records.append(
            {
                "row_key": str(row.row_key),
                "entry_time": entry_time,
                "exit_time": entry_time,
                "outcome_available_time": pd.Timestamp(exit_ns, unit="ns", tz="UTC"),
                "gross_return": float(gross_return),
                "net_return": float(gross_return - round_trip_cost),
                "round_trip_cost": round_trip_cost,
                "exit_reason": _normalize_exit_reason(exit_reason),
                "path_complete": True,
            }
        )
    return pd.DataFrame.from_records(records)


def _opportunity_context(frame: pd.DataFrame) -> pd.DataFrame:
    regimes = frame.apply(regime_tags, axis=1, result_type="expand")
    output = frame.copy()
    for column in regimes.columns:
        output[column] = regimes[column]
    return output


def _normalize_union_rows(
    *,
    stage: str,
    source_role: str,
    signals: pd.DataFrame,
    ledger: pd.DataFrame,
    source_hash: str,
) -> pd.DataFrame:
    current = ledger.copy()
    current["decision_time"] = pd.to_datetime(current["signal_time"], utc=True)
    if "row_key" not in current:
        keys = signals.set_index("decision_time")["row_key"]
        current["row_key"] = current["decision_time"].map(keys)
    current = current.drop(columns=["fold_id"], errors="ignore")
    context = signals.set_index("row_key")
    current = current.join(
        context[
            [
                "fold_id",
                "feature_available_time",
                "vol_z",
                "channel_slope_20",
                "funding_z",
                "oi_z",
            ]
        ],
        on="row_key",
        validate="many_to_one",
    )
    if current["row_key"].isna().any() or current["fold_id"].isna().any():
        raise AssertionError("Union ledger did not join to its causal signal")
    current = _opportunity_context(current)
    output = pd.DataFrame(index=current.index)
    output["stage"] = stage
    output["source_role"] = source_role
    output["fold_id"] = current["fold_id"].astype(int)
    output["row_key"] = current["row_key"].astype(str)
    output["source_artifact_hash"] = source_hash
    output["decision_time"] = current["decision_time"]
    output["feature_available_time"] = pd.to_datetime(
        current["feature_available_time"], utc=True
    )
    output["outcome_available_time"] = pd.to_datetime(
        current["intrabar_exit_time"], utc=True
    )
    output["entry_time"] = pd.to_datetime(current["entry_time"], utc=True)
    output["exit_time"] = pd.to_datetime(current["exit_time"], utc=True)
    output["route"] = "UNION_BASE"
    output["side"] = current["side"].map({-1: "SHORT", 1: "LONG"})
    output["confidence_tier"] = pd.Series(pd.NA, index=output.index, dtype="string")
    output["signal_run_bucket"] = pd.Series(pd.NA, index=output.index, dtype="string")
    output["gross_return"] = current["gross_return"].astype(float)
    output["net_return"] = current["net_return"].astype(float)
    output["round_trip_cost"] = output["gross_return"] - output["net_return"]
    output["exit_reason"] = current["exit_reason"].map(_normalize_exit_reason)
    for column in ("vol_regime", "trend_regime", "funding_regime", "oi_regime"):
        output[column] = current[column]
    output["path_complete"] = True
    output["opportunity_id"] = stage + ":UNION_BASE:" + output["row_key"]
    return output.loc[:, OUTPUT_COLUMNS]


def _normalize_candidate_rows(
    *,
    stage: str,
    source_role: str,
    signals: pd.DataFrame,
    candidate_paths: pd.DataFrame,
    source_hash: str,
) -> pd.DataFrame:
    candidates = signals.loc[signals["is_coverage_candidate"]].copy()
    replay_columns = [column for column in candidate_paths.columns if column != "row_key"]
    candidates = candidates.drop(columns=replay_columns, errors="ignore")
    candidates = candidates.merge(
        candidate_paths,
        on="row_key",
        how="left",
        validate="one_to_one",
    )
    required = [
        "entry_time",
        "exit_time",
        "outcome_available_time",
        "gross_return",
        "net_return",
        "round_trip_cost",
        "exit_reason",
    ]
    if candidates[required].isna().any().any():
        raise AssertionError("candidate execution path is incomplete")
    candidates = _opportunity_context(candidates)
    output = pd.DataFrame(index=candidates.index)
    output["stage"] = stage
    output["source_role"] = source_role
    output["fold_id"] = candidates["fold_id"].astype(int)
    output["row_key"] = candidates["row_key"].astype(str)
    output["source_artifact_hash"] = source_hash
    output["decision_time"] = pd.to_datetime(candidates["decision_time"], utc=True)
    output["feature_available_time"] = pd.to_datetime(
        candidates["feature_available_time"], utc=True
    )
    output["outcome_available_time"] = pd.to_datetime(
        candidates["outcome_available_time"], utc=True
    )
    output["entry_time"] = pd.to_datetime(candidates["entry_time"], utc=True)
    output["exit_time"] = pd.to_datetime(candidates["exit_time"], utc=True)
    output["route"] = "COVERAGE_CANDIDATE"
    output["side"] = candidates["candidate_side"]
    output["confidence_tier"] = candidates["confidence_tier"].astype("string")
    output["signal_run_bucket"] = candidates["signal_run_bucket"].astype("string")
    output["gross_return"] = candidates["gross_return"].astype(float)
    output["net_return"] = candidates["net_return"].astype(float)
    output["round_trip_cost"] = candidates["round_trip_cost"].astype(float)
    output["exit_reason"] = candidates["exit_reason"]
    for column in ("vol_regime", "trend_regime", "funding_regime", "oi_regime"):
        output[column] = candidates[column]
    output["path_complete"] = candidates["path_complete"].astype(bool)
    output["opportunity_id"] = stage + ":COVERAGE_CANDIDATE:" + output["row_key"]
    return output.loc[:, OUTPUT_COLUMNS]


def _finalize(union_rows: pd.DataFrame, candidate_rows: pd.DataFrame) -> pd.DataFrame:
    output = pd.concat([union_rows, candidate_rows], ignore_index=True).sort_values(
        ["decision_time", "route"], kind="stable"
    )
    output = output.reset_index(drop=True)
    if output["opportunity_id"].duplicated().any() or output["row_key"].duplicated().any():
        raise AssertionError("opportunity identifiers must be unique")
    if output["side"].isna().any():
        raise AssertionError("opportunity side is not LONG or SHORT")
    if not output["feature_available_time"].le(output["decision_time"]).all():
        raise AssertionError("future feature entered an opportunity")
    if not output["decision_time"].lt(output["outcome_available_time"]).all():
        raise AssertionError("opportunity outcome is not strictly future")
    if output["outcome_available_time"].ge(Q2_START).any():
        raise AssertionError("opportunity outcome reached the sealed Q2 interval")
    return output.loc[:, OUTPUT_COLUMNS]


def _prepare_candidate_fields(signals: pd.DataFrame) -> pd.DataFrame:
    output = signals.copy()
    output["decision_time"] = pd.to_datetime(output["decision_time"], utc=True)
    output["path_complete"] = output["path_complete"].astype(bool)
    mask = coverage_candidate_mask(output)
    confidence = output[
        ["p_short_lstm", "p_flat_lstm", "p_long_lstm"]
    ].max(axis=1)
    output["is_coverage_candidate"] = mask
    output["candidate_side"] = output["pred_lstm"].map(
        {0: "SHORT", 1: pd.NA, 2: "LONG"}
    )
    output["confidence_tier"] = _confidence_tier(confidence)
    output["signal_run_bucket"] = _signal_run_buckets(output, mask)
    if output.loc[mask, ["candidate_side", "confidence_tier", "signal_run_bucket"]].isna().any().any():
        raise AssertionError("coverage candidate categorization is incomplete")
    return output


def build_development_opportunities(
    cache: str | Path = DEFAULT_DEVELOPMENT_CACHE,
    *,
    source_paths: SourcePaths = SourcePaths(),
) -> pd.DataFrame:
    """Build the 2021-2024 OOF Union and Union-flat coverage stream."""

    root = Path(cache).resolve()
    names = ("development_signals.parquet", "development_control_ledger.parquet")
    verified = _verify_artifacts(root, "development_artifacts.json", names)
    signals = pd.read_parquet(root / names[0])
    control = pd.read_parquet(root / names[1])
    signals = _prepare_candidate_fields(signals)
    candidates = signals.loc[
        signals["is_coverage_candidate"], ["row_key", "decision_time", "candidate_side"]
    ].rename(columns={"candidate_side": "side"})
    start = pd.Timestamp(signals["decision_time"].min())
    end = pd.Timestamp("2025-01-01", tz="UTC")
    bars = _read_bounded(source_paths.m15, start, end)
    minute = _read_bounded(source_paths.minute, start, end)
    identities = {
        "m15_execution": _source_identity("m15_execution", source_paths.m15, bars),
        "minute_execution": _source_identity(
            "minute_execution", source_paths.minute, minute
        ),
    }
    source_hash = _canonical_hash(
        {"frozen_development": verified, "bounded_execution": identities}
    )
    paths = _replay_independent_candidate_paths(candidates, bars=bars, minute=minute)
    union_rows = _normalize_union_rows(
        stage="development",
        source_role="OOF_TEST",
        signals=signals,
        ledger=control,
        source_hash=source_hash,
    )
    candidate_rows = _normalize_candidate_rows(
        stage="development",
        source_role="OOF_TEST",
        signals=signals,
        candidate_paths=paths,
        source_hash=source_hash,
    )
    return _finalize(union_rows, candidate_rows)


def _exact_signal_context(
    stage: str,
    *,
    union_cache: Path,
    source_paths: SourcePaths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    start, end = _STAGES[stage]
    if end > Q2_START:
        raise AssertionError("exact source request reached the sealed Q2 interval")
    frozen = pd.read_parquet(union_cache / f"{stage}_signals.parquet")
    frozen = frozen.rename(columns={"timestamp": "decision_time"})
    frozen["decision_time"] = pd.to_datetime(frozen["decision_time"], utc=True)
    frozen = frozen.set_index("decision_time").sort_index()
    lstm = load_member_panel("lstm", 55, stage)
    svm = load_member_panel("svm_linear", 75, stage)
    if not frozen.index.equals(lstm.index) or not frozen.index.equals(svm.index):
        raise AssertionError("frozen exact member grids are not aligned")
    signals = frozen.copy()
    signals["pred_lstm"] = lstm["pred"].astype(int)
    signals["p_short_lstm"] = lstm["p_short"].astype(float)
    signals["p_flat_lstm"] = lstm["p_flat"].astype(float)
    signals["p_long_lstm"] = lstm["p_long"].astype(float)
    signals["pred_svm_linear"] = svm["pred"].astype(int)
    signals["fold_id"] = pd.factorize(lstm["refit_id"].astype(str), sort=False)[0]
    signals["path_complete"] = True
    signals = signals.reset_index()
    signals["row_key"] = signals["decision_time"].map(
        lambda value: f"exact-{stage}-{pd.Timestamp(value).strftime('%Y%m%dT%H%M%SZ')}"
    )

    context_start = start - pd.Timedelta(days=120)
    bundle = load_bounded_sources(context_start, end, source_paths)
    features = build_unified_decisions(bundle.m15, bundle.positioning)
    feature_columns = [
        "decision_time",
        "feature_available_time",
        "vol_z",
        "channel_slope_20",
        "funding_z",
        "oi_z",
    ]
    signals = signals.merge(
        features.loc[:, feature_columns],
        on="decision_time",
        how="left",
        validate="one_to_one",
    )
    signals = _prepare_candidate_fields(signals)
    selected = signals["is_coverage_candidate"] | signals["union_signal"].ne(0)
    required_context = ["feature_available_time", "vol_z", "channel_slope_20"]
    if signals.loc[selected, required_context].isna().any().any():
        raise AssertionError("exact causal feature context is incomplete")

    bars = _read_bounded(source_paths.m15, start, end)
    minute = bundle.minute.loc[(bundle.minute.index >= start) & (bundle.minute.index < end)]
    identities: dict[str, object] = {
        "feature_sources": bundle.source_identities,
        "m15_execution": _source_identity("m15_execution", source_paths.m15, bars),
        "raw_predictions": {
            f"{member.model}:{path.name}": _sha256_file(path)
            for member in MEMBERS
            for path in prediction_paths(member, stage)
        },
    }
    ledger = pd.read_parquet(union_cache / f"{stage}_ledger.parquet")
    return signals, ledger, bars, minute, identities


def build_exact_opportunities(
    stage: str,
    *,
    union_cache: str | Path = DEFAULT_UNION_CACHE,
    source_paths: SourcePaths = SourcePaths(),
) -> pd.DataFrame:
    """Build frozen H1 or secondary-forward opportunities without Q2 access."""

    normalized_stage = stage.strip().lower()
    if normalized_stage not in _STAGES:
        raise ValueError("stage must be 'h1' or 'forward'")
    root = Path(union_cache).resolve()
    names = (
        f"{normalized_stage}_signals.parquet",
        f"{normalized_stage}_ledger.parquet",
    )
    verified = _verify_artifacts(root, "manifest.json", names)
    signals, ledger, bars, minute, identities = _exact_signal_context(
        normalized_stage, union_cache=root, source_paths=source_paths
    )
    source_hash = _canonical_hash(
        {"frozen_union": verified, "bounded_sources": identities}
    )
    candidates = signals.loc[
        signals["is_coverage_candidate"], ["row_key", "decision_time", "candidate_side"]
    ].rename(columns={"candidate_side": "side"})
    paths = _replay_independent_candidate_paths(candidates, bars=bars, minute=minute)
    union_rows = _normalize_union_rows(
        stage=normalized_stage,
        source_role="FROZEN_EXACT",
        signals=signals,
        ledger=ledger,
        source_hash=source_hash,
    )
    candidate_rows = _normalize_candidate_rows(
        stage=normalized_stage,
        source_role="FROZEN_EXACT",
        signals=signals,
        candidate_paths=paths,
        source_hash=source_hash,
    )
    return _finalize(union_rows, candidate_rows)


def assign_observation_episodes(
    frame: pd.DataFrame,
    *,
    min_candidates: int = 20,
    max_candidates: int = 30,
    min_per_side: int = 6,
) -> pd.DataFrame:
    """Close causal episodes on resolved candidates without crossing folds."""

    required = {
        "opportunity_id",
        "stage",
        "fold_id",
        "route",
        "side",
        "decision_time",
        "outcome_available_time",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"opportunity frame lacks columns: {missing}")
    if not (0 < min_per_side * 2 <= min_candidates <= max_candidates):
        raise ValueError("observation episode thresholds are inconsistent")
    output = frame.copy()
    output["decision_time"] = pd.to_datetime(output["decision_time"], utc=True)
    output["outcome_available_time"] = pd.to_datetime(
        output["outcome_available_time"], utc=True
    )
    output = output.sort_values(
        ["fold_id", "outcome_available_time", "decision_time", "opportunity_id"],
        kind="stable",
    ).reset_index(drop=True)
    assignments: list[dict[str, object]] = []
    stage = str(output["stage"].iloc[0]) if len(output) else "stage"

    for fold_id, fold in output.groupby("fold_id", sort=True):
        pending: list[int] = []
        episode_number = 0

        def close_episode(can_propose: bool) -> None:
            nonlocal episode_number
            if not pending:
                return
            subset = output.loc[pending]
            candidates = subset.loc[subset["route"].eq("COVERAGE_CANDIDATE")]
            long_count = int(candidates["side"].eq("LONG").sum())
            short_count = int(candidates["side"].eq("SHORT").sum())
            cutoff = subset["outcome_available_time"].max()
            episode_id = f"{stage}-fold{int(fold_id)}-episode{episode_number}"
            for index in pending:
                assignments.append(
                    {
                        "index": index,
                        "observation_episode_id": episode_id,
                        "episode_cutoff_utc": cutoff,
                        "episode_status": (
                            "COMPLETE" if can_propose else "INSUFFICIENT_SIDE_SUPPORT"
                        ),
                        "episode_can_propose": can_propose,
                        "episode_opportunity_count": len(subset),
                        "episode_candidate_count": len(candidates),
                        "episode_long_candidate_count": long_count,
                        "episode_short_candidate_count": short_count,
                    }
                )
            pending.clear()
            episode_number += 1

        for index in fold.index:
            pending.append(int(index))
            subset = output.loc[pending]
            candidates = subset.loc[subset["route"].eq("COVERAGE_CANDIDATE")]
            candidate_count = len(candidates)
            long_count = int(candidates["side"].eq("LONG").sum())
            short_count = int(candidates["side"].eq("SHORT").sum())
            if (
                candidate_count >= min_candidates
                and long_count >= min_per_side
                and short_count >= min_per_side
            ):
                close_episode(True)
            elif candidate_count >= max_candidates:
                close_episode(False)
        close_episode(False)

    metadata = pd.DataFrame(assignments).set_index("index")
    output = output.join(metadata, how="left")
    if output["observation_episode_id"].isna().any():
        raise AssertionError("every opportunity must belong to an observation episode")
    if output.groupby("observation_episode_id")["fold_id"].nunique().gt(1).any():
        raise AssertionError("observation episode crossed a fold boundary")
    if not output["outcome_available_time"].le(output["episode_cutoff_utc"]).all():
        raise AssertionError("episode cutoff precedes a supporting outcome")
    return output


__all__ = [
    "OUTPUT_COLUMNS",
    "assign_observation_episodes",
    "build_development_opportunities",
    "build_exact_opportunities",
    "coverage_candidate_mask",
    "regime_tags",
]
