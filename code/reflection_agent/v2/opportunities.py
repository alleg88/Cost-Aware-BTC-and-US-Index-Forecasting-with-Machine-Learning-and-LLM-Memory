"""Immutable causal opportunity and observation-episode ledgers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.run_unified_2021_ensemble import SourcePaths, load_bounded_sources
from experiments.unified_2021_ensemble_data import build_unified_decisions
from experiments.union_v1_episode_reentry_policy import (
    identify_same_side_episodes,
    replay_union_control_and_reentry,
)


CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEVELOPMENT_CACHE = CODE_ROOT / "experiments" / "cache" / "union_v1_episode_reentry"
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

_OUTPUT_COLUMNS = (
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
    "gross_return",
    "net_return",
    "round_trip_cost",
    "exit_reason",
    "member_pattern",
    "signal_episode_id",
    "signal_episode_bar",
    "episode_bar_bucket",
    "previous_exit_reason",
    "previous_exit_reason_available_time",
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


def _verify_artifacts(root: Path, manifest_name: str, names: Iterable[str]) -> dict[str, str]:
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
    """Apply only preregistered, outcome-independent regime boundaries."""

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


def _normalize_exit_reason(value: object) -> str:
    normalized = str(value).strip().lower()
    mapping = {
        "take_profit": "TAKE_PROFIT",
        "stop_loss": "STOP_LOSS",
        "timeout": "TIMEOUT",
    }
    if normalized not in mapping:
        raise ValueError(f"unknown exit reason: {value!r}")
    return mapping[normalized]


def _member_pattern(row: pd.Series) -> str:
    lstm = int(pd.to_numeric(row["lstm_signal"], errors="raise"))
    svm = int(pd.to_numeric(row["svm_linear_signal"], errors="raise"))
    union = int(pd.to_numeric(row["union_signal"], errors="raise"))
    if union == 0:
        raise ValueError("an opportunity cannot come from a flat Union signal")
    if lstm == union and svm == union:
        return "BOTH_AGREE"
    if lstm == union and svm == 0:
        return "LSTM_ONLY"
    if svm == union and lstm == 0:
        return "SVM_ONLY"
    raise ValueError("member signals do not reconcile with frozen Union")


def _previous_exit_reasons(
    control: pd.DataFrame, reentries: pd.DataFrame
) -> dict[str, tuple[str, pd.Timestamp]]:
    controls = control.sort_values("signal_time", kind="stable")
    result: dict[str, tuple[str, pd.Timestamp]] = {}
    for row in reentries.itertuples(index=False):
        preceding = controls.loc[
            controls["episode_id"].eq(row.episode_id)
            & controls["signal_time"].lt(row.signal_time)
        ]
        if preceding.empty:
            raise AssertionError("re-entry lacks a preceding Union trade")
        previous = preceding.iloc[-1]
        available_at = pd.Timestamp(previous["intrabar_exit_time"])
        if available_at >= pd.Timestamp(row.entry_time):
            raise AssertionError("preceding Union exit is not known before re-entry execution")
        result[str(row.row_key)] = (
            _normalize_exit_reason(previous["exit_reason"]),
            available_at,
        )
    return result


def _normalize_opportunities(
    *,
    stage: str,
    source_role: str,
    signals: pd.DataFrame,
    control: pd.DataFrame,
    reentries: pd.DataFrame,
    source_hash: str,
) -> pd.DataFrame:
    signal_frame = signals.copy()
    for name in ("decision_time", "feature_available_time"):
        signal_frame[name] = pd.to_datetime(signal_frame[name], utc=True)
    signal_frame = signal_frame.set_index("row_key")
    if not signal_frame.index.is_unique:
        raise AssertionError("signal row keys must be unique")

    prepared: list[pd.DataFrame] = []
    for route, ledger in (("UNION_BASE", control), ("REENTRY", reentries)):
        current = ledger.copy()
        for name in (
            "signal_time",
            "entry_time",
            "exit_time",
            "intrabar_exit_time",
        ):
            current[name] = pd.to_datetime(current[name], utc=True)
        if "row_key" not in current:
            key_by_time = signals.set_index("decision_time")["row_key"]
            current["row_key"] = current["signal_time"].map(key_by_time)
        if current["row_key"].isna().any():
            raise AssertionError("trade ledger did not join to its causal signal")
        current["route"] = route
        prepared.append(current)

    control_frame, reentry_frame = prepared
    previous_reasons = _previous_exit_reasons(control_frame, reentry_frame)
    ledger = pd.concat(prepared, ignore_index=True).sort_values(
        ["signal_time", "route"], kind="stable"
    )
    joined = ledger.join(
        signal_frame[
            [
                "fold_id",
                "feature_available_time",
                "lstm_signal",
                "svm_linear_signal",
                "union_signal",
                "episode_id",
                "episode_bar",
                "vol_z",
                "channel_slope_20",
                "funding_z",
                "oi_z",
            ]
        ],
        on="row_key",
        rsuffix="_signal",
        validate="many_to_one",
    )
    if joined[
        ["fold_id", "feature_available_time", "episode_id", "episode_bar"]
    ].isna().any().any():
        raise AssertionError("opportunity context is incomplete")

    regimes = joined.apply(regime_tags, axis=1, result_type="expand")
    output = pd.DataFrame(index=joined.index)
    output["stage"] = stage
    output["source_role"] = source_role
    output["fold_id"] = joined["fold_id"].astype(int)
    output["row_key"] = joined["row_key"].astype(str)
    output["source_artifact_hash"] = source_hash
    output["decision_time"] = joined["signal_time"]
    output["feature_available_time"] = joined["feature_available_time"]
    output["outcome_available_time"] = joined["intrabar_exit_time"]
    output["entry_time"] = joined["entry_time"]
    output["exit_time"] = joined["exit_time"]
    output["route"] = joined["route"]
    output["side"] = joined["side"].map({-1: "SHORT", 1: "LONG"})
    output["gross_return"] = joined["gross_return"].astype(float)
    output["net_return"] = joined["net_return"].astype(float)
    output["round_trip_cost"] = output["gross_return"] - output["net_return"]
    output["exit_reason"] = joined["exit_reason"].map(_normalize_exit_reason)
    output["member_pattern"] = joined.apply(_member_pattern, axis=1)
    output["signal_episode_id"] = joined["episode_id"].astype(int)
    output["signal_episode_bar"] = joined["episode_bar"].astype(int)
    is_reentry = output["route"].eq("REENTRY")
    if output.loc[is_reentry, "signal_episode_bar"].lt(2).any():
        raise AssertionError("a re-entry cannot be the first signal in an episode")
    output["episode_bar_bucket"] = None
    output.loc[is_reentry, "episode_bar_bucket"] = np.where(
        output.loc[is_reentry, "signal_episode_bar"].eq(2), "SECOND", "THIRD_PLUS"
    )
    output["previous_exit_reason"] = None
    output["previous_exit_reason_available_time"] = pd.Series(
        pd.NaT, index=output.index, dtype="datetime64[ns, UTC]"
    )
    output.loc[is_reentry, "previous_exit_reason"] = output.loc[
        is_reentry, "row_key"
    ].map(lambda key: previous_reasons[key][0])
    output.loc[is_reentry, "previous_exit_reason_available_time"] = output.loc[
        is_reentry, "row_key"
    ].map(lambda key: previous_reasons[key][1])
    output["previous_exit_reason_available_time"] = pd.to_datetime(
        output["previous_exit_reason_available_time"], utc=True
    )
    for column in regimes.columns:
        output[column] = regimes[column]
    output["path_complete"] = True
    output["opportunity_id"] = (
        output["stage"] + ":" + output["route"] + ":" + output["row_key"]
    )

    if output["side"].isna().any():
        raise AssertionError("opportunity side is not LONG or SHORT")
    if output["opportunity_id"].duplicated().any() or output["row_key"].duplicated().any():
        raise AssertionError("opportunity identifiers must be unique")
    if not output["decision_time"].lt(output["outcome_available_time"]).all():
        raise AssertionError("opportunity outcome is not strictly future to its decision")
    if not output["feature_available_time"].le(output["decision_time"]).all():
        raise AssertionError("future feature entered an opportunity")
    return output.loc[:, _OUTPUT_COLUMNS].sort_values(
        ["decision_time", "route"], kind="stable"
    ).reset_index(drop=True)


def build_development_opportunities(
    cache: str | Path = DEFAULT_DEVELOPMENT_CACHE,
) -> pd.DataFrame:
    """Read and normalize the completed 2021-2024 five-fold OOF universe."""

    root = Path(cache).resolve()
    names = (
        "development_signals.parquet",
        "development_control_ledger.parquet",
        "development_reentry_ledger.parquet",
    )
    verified = _verify_artifacts(root, "development_artifacts.json", names)
    signals = pd.read_parquet(root / names[0])
    control = pd.read_parquet(root / names[1])
    reentries = pd.read_parquet(root / names[2])
    return _normalize_opportunities(
        stage="development",
        source_role="OOF_TEST",
        signals=signals,
        control=control,
        reentries=reentries,
        source_hash=_canonical_hash(verified),
    )


def _exact_signal_context(
    stage: str,
    *,
    union_cache: Path,
    source_paths: SourcePaths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    stage_start, stage_end = _STAGES[stage]
    context_start = stage_start - pd.Timedelta(days=120)
    bundle = load_bounded_sources(context_start, stage_end, source_paths)
    if bundle.max_loaded_timestamp >= Q2_START:
        raise AssertionError("exact source reader reached the sealed Q2 interval")

    frozen = pd.read_parquet(union_cache / f"{stage}_signals.parquet").rename(
        columns={"timestamp": "decision_time"}
    )
    frozen["decision_time"] = pd.to_datetime(frozen["decision_time"], utc=True)
    frozen["row_key"] = frozen["decision_time"].map(
        lambda value: f"exact-{stage}-{pd.Timestamp(value).strftime('%Y%m%dT%H%M%SZ')}"
    )
    frozen = identify_same_side_episodes(frozen)

    features = build_unified_decisions(bundle.m15, bundle.positioning)
    feature_columns = [
        "decision_time",
        "feature_available_time",
        "vol_z",
        "channel_slope_20",
        "funding_z",
        "oi_z",
    ]
    context = features.loc[:, feature_columns].copy()
    signals = frozen.merge(context, on="decision_time", how="left", validate="one_to_one")
    causal_required = ["feature_available_time", "vol_z", "channel_slope_20"]
    if signals[causal_required].isna().any().any():
        missing = signals.loc[signals[causal_required].isna().any(axis=1), "decision_time"]
        raise AssertionError(f"exact causal feature context is missing at {missing.iloc[0]}")
    signals["fold_id"] = (
        (signals["decision_time"].dt.year - stage_start.year) * 12
        + signals["decision_time"].dt.month
        - stage_start.month
    ).astype(int)

    stage_m15 = bundle.m15.loc[
        (bundle.m15.index >= stage_start) & (bundle.m15.index < stage_end)
    ]
    stage_minute = bundle.minute.loc[
        (bundle.minute.index >= stage_start) & (bundle.minute.index < stage_end)
    ]
    replay = replay_union_control_and_reentry(signals, stage_m15, stage_minute)
    control = replay.control_ledger.copy()
    reentries = replay.reentry_ledger.copy()
    key_by_time = signals.set_index("decision_time")["row_key"]
    for ledger in (control, reentries):
        ledger["row_key"] = pd.to_datetime(ledger["signal_time"], utc=True).map(key_by_time)

    frozen_control = pd.read_parquet(union_cache / f"{stage}_ledger.parquet").sort_values(
        "signal_time", kind="stable"
    )
    replay_control = control.sort_values("signal_time", kind="stable")
    if len(frozen_control) != len(replay_control):
        raise AssertionError(f"exact {stage} Union trade count drifted")
    if not pd.DatetimeIndex(frozen_control["signal_time"]).equals(
        pd.DatetimeIndex(replay_control["signal_time"])
    ):
        raise AssertionError(f"exact {stage} Union signal times drifted")
    if not np.allclose(
        frozen_control["net_return"], replay_control["net_return"], rtol=0.0, atol=1e-12
    ):
        raise AssertionError(f"exact {stage} Union economics drifted")
    return signals, control, reentries, bundle.source_identities


def build_exact_opportunities(
    stage: str,
    *,
    union_cache: str | Path = DEFAULT_UNION_CACHE,
    source_paths: SourcePaths = SourcePaths(),
) -> pd.DataFrame:
    """Build exact frozen-Union H1 or forward opportunities without Q2 access."""

    normalized_stage = stage.strip().lower()
    if normalized_stage not in _STAGES:
        raise ValueError("stage must be 'h1' or 'forward'")
    root = Path(union_cache).resolve()
    names = (f"{normalized_stage}_signals.parquet", f"{normalized_stage}_ledger.parquet")
    verified = _verify_artifacts(root, "manifest.json", names)
    signals, control, reentries, source_identities = _exact_signal_context(
        normalized_stage, union_cache=root, source_paths=source_paths
    )
    return _normalize_opportunities(
        stage=normalized_stage,
        source_role="FROZEN_EXACT",
        signals=signals,
        control=control,
        reentries=reentries,
        source_hash=_canonical_hash(
            {"union_artifacts": verified, "bounded_sources": source_identities}
        ),
    )


def assign_observation_episodes(
    frame: pd.DataFrame,
    *,
    min_opportunities: int = 60,
    max_opportunities: int = 90,
    min_reentry_opportunities: int = 10,
    min_reentry_per_side: int = 3,
) -> pd.DataFrame:
    """Close resolution-time episodes causally and never across fold boundaries."""

    required = {
        "opportunity_id",
        "fold_id",
        "route",
        "side",
        "decision_time",
        "outcome_available_time",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"opportunity frame lacks columns: {missing}")
    if not (
        0 < min_opportunities <= max_opportunities
        and 0 < min_reentry_opportunities <= min_opportunities
        and 0 < min_reentry_per_side * 2 <= min_reentry_opportunities
    ):
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
    stage = str(output["stage"].iloc[0]) if len(output) and "stage" in output else "stage"

    for fold_id, fold in output.groupby("fold_id", sort=True):
        pending: list[int] = []
        episode_number = 0

        def close_episode(can_propose: bool) -> None:
            nonlocal episode_number
            if not pending:
                return
            subset = output.loc[pending]
            long_count = int(subset["side"].eq("LONG").sum())
            short_count = int(subset["side"].eq("SHORT").sum())
            reentries = subset.loc[subset["route"].eq("REENTRY")]
            reentry_long_count = int(reentries["side"].eq("LONG").sum())
            reentry_short_count = int(reentries["side"].eq("SHORT").sum())
            cutoff = subset["outcome_available_time"].max()
            episode_id = f"{stage}-fold{int(fold_id)}-episode{episode_number}"
            for index in pending:
                assignments.append(
                    {
                        "index": index,
                        "observation_episode_id": episode_id,
                        "episode_cutoff_utc": cutoff,
                        "episode_status": (
                            "COMPLETE" if can_propose else "INSUFFICIENT_EVIDENCE"
                        ),
                        "episode_can_propose": can_propose,
                        "episode_opportunity_count": len(pending),
                        "episode_long_count": long_count,
                        "episode_short_count": short_count,
                        "episode_reentry_count": len(reentries),
                        "episode_reentry_long_count": reentry_long_count,
                        "episode_reentry_short_count": reentry_short_count,
                    }
                )
            pending.clear()
            episode_number += 1

        for index in fold.index:
            pending.append(int(index))
            subset = output.loc[pending]
            count = len(pending)
            reentries = subset.loc[subset["route"].eq("REENTRY")]
            reentry_long_count = int(reentries["side"].eq("LONG").sum())
            reentry_short_count = int(reentries["side"].eq("SHORT").sum())
            if (
                count >= min_opportunities
                and len(reentries) >= min_reentry_opportunities
                and reentry_long_count >= min_reentry_per_side
                and reentry_short_count >= min_reentry_per_side
            ):
                close_episode(True)
            elif count >= max_opportunities:
                close_episode(False)
        close_episode(False)

    metadata = pd.DataFrame(assignments).set_index("index")
    output = output.join(metadata, how="left")
    if output["observation_episode_id"].isna().any():
        raise AssertionError("every opportunity must belong to one observation episode")
    if output.groupby("observation_episode_id")["fold_id"].nunique().gt(1).any():
        raise AssertionError("observation episode crossed a fold boundary")
    if not output["outcome_available_time"].le(output["episode_cutoff_utc"]).all():
        raise AssertionError("episode cutoff precedes a supporting outcome")
    return output


__all__ = [
    "assign_observation_episodes",
    "build_development_opportunities",
    "build_exact_opportunities",
    "regime_tags",
]
