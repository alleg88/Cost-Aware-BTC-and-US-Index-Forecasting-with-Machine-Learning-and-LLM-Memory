"""Run the resumable dev-only Fast-T2 sequential entry experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from evaluation.channel_window_validation import (
    PurgedFold,
    effective_sample_size,
    expanding_purged_folds,
    interval_uniqueness,
)
from evaluation.fast_t2_action_policy import (
    first_crossing_entries,
    replay_entry_capacity,
    select_inner_quantile,
)
from experiments.fast_t2_action_models import MODEL_NAMES, fit_predict_action_model
from experiments.fast_t2_entry_dataset import (
    ENTRY_FEATURE_COLUMNS,
    SEQUENCE_FEATURE_COLUMNS,
    EntryDecisionConfig,
    build_entry_decisions,
    build_entry_sequences,
)
from experiments.fast_t2_study import DEV_END, DEV_START


CODE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_EVENTS = (
    CODE_ROOT
    / "experiments"
    / "cache"
    / "two_trigger_model_study"
    / "dev"
    / "events_fast_t2_2bps.parquet"
)
MINUTE_SOURCE = CODE_ROOT / "data" / "btcusdt_1m_2021_2026.parquet"
OUT = CODE_ROOT / "experiments" / "cache" / "fast_t2_entry_policy" / "dev"
DEFAULT_CONFIG = EntryDecisionConfig()
CAPACITIES: tuple[tuple[str, int | None], ...] = (
    ("unlimited", None),
    ("capacity_3", 3),
    ("capacity_1", 1),
)

SCORE_COLUMNS = (
    "window_id",
    "decision_id",
    "side",
    "channel_episode_id",
    "t2_time",
    "decision_time",
    "entry_time",
    "label_start",
    "label_end",
    "active_end_time",
    "entry_price",
    "stop_price",
    "target_price",
    "exit_time",
    "exit_price",
    "outcome",
    "r_net",
    "label_net_positive",
    "filled",
    "holding_minutes",
    "minutes_since_t2",
)


def _json_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalise_models(models: Iterable[str]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(models))
    unknown = sorted(set(requested).difference(MODEL_NAMES))
    if unknown:
        raise ValueError(f"unsupported models: {unknown}")
    if not requested:
        raise ValueError("at least one model is required")
    return tuple(name for name in MODEL_NAMES if name in requested)


def build_protocol(
    *,
    output_dir: Path | None = None,
    decision_minutes: int = 15,
    models: Iterable[str] = MODEL_NAMES,
) -> dict[str, object]:
    """Build the path-independent registered protocol and its stable hash."""
    del output_dir  # Paths must never influence or leak into the protocol.
    config = replace(DEFAULT_CONFIG, decision_minutes=decision_minutes)
    normalised_models = _normalise_models(models)
    payload: dict[str, object] = {
        "stage": "development",
        "arm": "fast_t2_2bps",
        "period_start": DEV_START.isoformat(),
        "period_end_exclusive": DEV_END.isoformat(),
        "entry_config": asdict(config),
        "static_features": list(ENTRY_FEATURE_COLUMNS),
        "sequence_features": list(SEQUENCE_FEATURE_COLUMNS),
        "models": list(normalised_models),
        "validation": "seven expanding six-month episode-purged folds",
        "threshold_selection": "inner chronological six-month score quantiles",
        "primary_portfolio": "all signals (unlimited)",
        "capacity_sensitivity": [3, 1],
        "forward_or_lockbox_loaded": False,
    }
    payload["protocol_hash"] = _json_hash(payload)
    return payload


def write_protocol(
    output_dir: Path,
    *,
    decision_minutes: int = 15,
    models: Iterable[str] = MODEL_NAMES,
    overwrite: bool = False,
) -> dict[str, object]:
    """Write a protocol, rejecting an incompatible resume by default."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol(
        output_dir=output_dir,
        decision_minutes=decision_minutes,
        models=models,
    )
    path = output_dir / "protocol.json"
    if path.exists() and not overwrite:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("protocol_hash") != protocol["protocol_hash"]:
            raise ValueError("protocol hash mismatch; use --rebuild intentionally")
        return stored
    path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return protocol


def validate_resume(
    output_dir: Path,
    *,
    decision_minutes: int = 15,
    models: Iterable[str] = MODEL_NAMES,
) -> dict[str, object]:
    """Verify that existing artifacts belong to the requested protocol."""
    output_dir = Path(output_dir)
    path = output_dir / "protocol.json"
    if not path.exists():
        raise FileNotFoundError(path)
    expected = build_protocol(
        output_dir=output_dir,
        decision_minutes=decision_minutes,
        models=models,
    )
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored.get("protocol_hash") != expected["protocol_hash"]:
        raise ValueError("protocol hash mismatch; use --rebuild intentionally")
    return stored


def inner_episode_purged_indices(
    decisions: pd.DataFrame,
    outer_train: np.ndarray,
    *,
    inner_cut: pd.Timestamp,
    outer_valid_start: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    """Split only outer-training rows while purging whole shared episodes."""
    positions = np.asarray(outer_train, dtype=np.int64)
    if positions.ndim != 1:
        raise ValueError("outer_train must be one-dimensional")
    subset = decisions.iloc[positions].copy()
    for column in ("decision_time", "label_end"):
        subset[column] = pd.to_datetime(subset[column], utc=True, errors="raise")
    inner_cut = pd.Timestamp(inner_cut)
    outer_valid_start = pd.Timestamp(outer_valid_start)
    inner_cut = (
        inner_cut.tz_localize("UTC")
        if inner_cut.tzinfo is None
        else inner_cut.tz_convert("UTC")
    )
    outer_valid_start = (
        outer_valid_start.tz_localize("UTC")
        if outer_valid_start.tzinfo is None
        else outer_valid_start.tz_convert("UTC")
    )
    groups = subset.groupby("channel_episode_id", sort=False).agg(
        group_start=("decision_time", "min"),
        group_decision_end=("decision_time", "max"),
        group_label_end=("label_end", "max"),
    )
    fit_episodes = groups.index[groups["group_label_end"].lt(inner_cut)]
    valid_episodes = groups.index[
        groups["group_start"].ge(inner_cut)
        & groups["group_decision_end"].lt(outer_valid_start)
        & groups["group_label_end"].lt(outer_valid_start)
    ]
    fit_mask = subset["channel_episode_id"].isin(fit_episodes)
    valid_mask = subset["channel_episode_id"].isin(valid_episodes)
    fit = positions[np.flatnonzero(fit_mask.to_numpy())]
    valid = positions[np.flatnonzero(valid_mask.to_numpy())]
    overlap = set(decisions.iloc[fit]["channel_episode_id"]).intersection(
        decisions.iloc[valid]["channel_episode_id"]
    )
    if overlap:
        raise AssertionError(f"inner split shares episodes: {sorted(overlap)}")
    return fit, valid


def _load_rich_minutes() -> pd.DataFrame:
    minute = pd.read_parquet(
        MINUTE_SOURCE,
        columns=[
            "open",
            "high",
            "low",
            "close",
            "volume",
            "taker_buy_base",
            "count",
        ],
        filters=[("timestamp", ">=", DEV_START), ("timestamp", "<", DEV_END)],
    ).sort_index(kind="stable")
    minute = minute[(minute.index >= DEV_START) & (minute.index < DEV_END)]
    if minute.index.has_duplicates:
        raise ValueError("native one-minute source has duplicate timestamps")
    if len(minute) and minute.index.max() >= DEV_END:
        raise AssertionError("runner loaded data at or beyond the dev boundary")
    return minute


def _dataset_protocol(config: EntryDecisionConfig) -> dict[str, object]:
    if not SOURCE_EVENTS.exists():
        raise FileNotFoundError(SOURCE_EVENTS)
    if not MINUTE_SOURCE.exists():
        raise FileNotFoundError(MINUTE_SOURCE)
    payload: dict[str, object] = {
        "arm": "fast_t2_2bps",
        "period_start": DEV_START.isoformat(),
        "period_end_exclusive": DEV_END.isoformat(),
        "config": asdict(config),
        "static_features": list(ENTRY_FEATURE_COLUMNS),
        "sequence_features": list(SEQUENCE_FEATURE_COLUMNS),
        "event_source_bytes": SOURCE_EVENTS.stat().st_size,
        "minute_source_bytes": MINUTE_SOURCE.stat().st_size,
    }
    payload["dataset_hash"] = _json_hash(payload)
    return payload


def prepare_dataset(
    *, output_dir: Path = OUT, config: EntryDecisionConfig = DEFAULT_CONFIG
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    """Create or resume the immutable decision and sequence cache."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_path = output_dir / "entry_decisions.parquet"
    sequence_path = output_dir / "entry_sequences.npy"
    manifest_path = output_dir / "dataset_manifest.json"
    expected = _dataset_protocol(config)
    if decision_path.exists() and sequence_path.exists() and manifest_path.exists():
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored.get("dataset_hash") == expected["dataset_hash"]:
            decisions = pd.read_parquet(decision_path)
            sequences = np.load(sequence_path, mmap_mode="r")
            if len(decisions) != len(sequences):
                raise ValueError("decision and sequence caches are misaligned")
            if _json_hash(decisions["decision_id"].astype(str).tolist()) != stored.get(
                "decision_ledger_hash"
            ):
                raise ValueError("decision cache ledger hash mismatch")
            print(f"resume decision dataset: {len(decisions):,} rows", flush=True)
            return decisions, sequences, stored

    events = pd.read_parquet(SOURCE_EVENTS)
    for column in ("decision_time", "label_end"):
        events[column] = pd.to_datetime(events[column], utc=True, errors="raise")
    events = events[
        events["decision_time"].ge(DEV_START)
        & events["decision_time"].lt(DEV_END)
        & events["label_end"].lt(DEV_END)
    ].copy()
    minute = _load_rich_minutes()
    decisions, audit = build_entry_decisions(events, minute, config)
    if decisions.empty:
        raise ValueError("Fast-T2 entry dataset is empty")
    if pd.to_datetime(decisions["label_end"], utc=True).max() >= DEV_END:
        raise AssertionError("entry labels reach the sealed post-dev period")
    sequences = build_entry_sequences(
        decisions, minute, sequence_minutes=config.sequence_minutes
    )
    decisions.to_parquet(decision_path, index=False)
    np.save(sequence_path, sequences)
    manifest = {
        **expected,
        "decision_ledger_hash": _json_hash(
            decisions["decision_id"].astype(str).tolist()
        ),
        "rows": int(len(decisions)),
        "windows": int(decisions["window_id"].nunique()),
        "max_loaded_timestamp": minute.index.max().isoformat(),
        "audit": audit,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"built decision dataset: {len(decisions):,} rows, "
        f"{decisions['window_id'].nunique():,} windows",
        flush=True,
    )
    return decisions, sequences, manifest


def _balanced_weights(labels: np.ndarray, uniqueness: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int8)
    classes, counts = np.unique(labels, return_counts=True)
    if set(classes) != {0, 1}:
        raise ValueError("training fold needs both binary classes")
    factors = {
        int(label): len(labels) / (2.0 * int(count))
        for label, count in zip(classes, counts, strict=True)
    }
    return uniqueness * np.array([factors[int(value)] for value in labels])


def _score_frame(
    decisions: pd.DataFrame, positions: np.ndarray, scores: np.ndarray
) -> pd.DataFrame:
    output = decisions.iloc[positions][list(SCORE_COLUMNS)].copy()
    output["score"] = np.asarray(scores, dtype=float)
    return output.reset_index(drop=True)


def _fit_and_score(
    decisions: pd.DataFrame,
    sequences: np.ndarray,
    *,
    model_name: str,
    train_index: np.ndarray,
    score_index: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    train_index = np.asarray(train_index, dtype=np.int64)
    score_index = np.asarray(score_index, dtype=np.int64)
    if not len(train_index) or not len(score_index):
        raise ValueError("fit and score indices cannot be empty")
    train_episodes = set(decisions.iloc[train_index]["channel_episode_id"])
    train_positions = set(train_index.tolist())
    score_only = np.array(
        [position for position in score_index if position not in train_positions],
        dtype=np.int64,
    )
    score_only_episodes = set(
        decisions.iloc[score_only]["channel_episode_id"] if len(score_only) else []
    )
    if train_episodes.intersection(score_only_episodes):
        raise AssertionError("fit and score indices share channel episodes")
    labels = decisions.iloc[train_index]["label_net_positive"].to_numpy(
        dtype=np.int8
    )
    raw_uniqueness = interval_uniqueness(decisions, train_index, normalize=False)
    uniqueness = raw_uniqueness / raw_uniqueness.mean()
    weights = _balanced_weights(labels, uniqueness)
    recurrent = model_name in {"gru", "lstm"}
    scores = fit_predict_action_model(
        model_name,
        decisions.iloc[train_index][list(ENTRY_FEATURE_COLUMNS)].to_numpy(dtype=float),
        np.asarray(sequences[train_index]) if recurrent else None,
        labels,
        weights,
        decisions.iloc[score_index][list(ENTRY_FEATURE_COLUMNS)].to_numpy(dtype=float),
        np.asarray(sequences[score_index]) if recurrent else None,
    )
    return _score_frame(decisions, score_index, scores), raw_uniqueness


def score_outer_fold(
    decisions: pd.DataFrame,
    sequences: np.ndarray,
    fold: PurgedFold,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Select a threshold on inner history and score one untouched outer fold."""
    inner_cut = fold.valid_start - pd.DateOffset(months=6)
    inner_fit, inner_valid = inner_episode_purged_indices(
        decisions,
        fold.train,
        inner_cut=inner_cut,
        outer_valid_start=fold.valid_start,
    )
    inner_scores, inner_uniqueness = _fit_and_score(
        decisions,
        sequences,
        model_name=model_name,
        train_index=inner_fit,
        score_index=inner_valid,
    )
    chosen_quantile, inner_table = select_inner_quantile(
        inner_scores,
        evaluation_days=(fold.valid_start - inner_cut).days,
    )

    combined = np.concatenate([fold.train, fold.valid])
    combined_scores, outer_uniqueness = _fit_and_score(
        decisions,
        sequences,
        model_name=model_name,
        train_index=fold.train,
        score_index=combined,
    )
    train_scores = combined_scores.iloc[: len(fold.train)]
    outer_scores = combined_scores.iloc[len(fold.train):].reset_index(drop=True)
    threshold = float(train_scores["score"].quantile(chosen_quantile))
    entries, actions = first_crossing_entries(outer_scores, threshold)
    outer_scores.insert(0, "model", model_name)
    outer_scores.insert(1, "fold_id", fold.fold_id)
    for frame in (entries, actions):
        frame.insert(0, "model", model_name)
        frame.insert(1, "fold_id", fold.fold_id)
    inner_table.insert(0, "model", model_name)
    inner_table.insert(1, "fold_id", fold.fold_id)
    inner_table["chosen"] = inner_table["quantile"].eq(chosen_quantile)
    inner_table["outer_threshold"] = threshold

    overlap = set(decisions.iloc[fold.train]["channel_episode_id"]).intersection(
        decisions.iloc[fold.valid]["channel_episode_id"]
    )
    audit: dict[str, object] = {
        "model": model_name,
        "fold_id": fold.fold_id,
        "train_rows": int(len(fold.train)),
        "validation_rows": int(len(fold.valid)),
        "inner_fit_rows": int(len(inner_fit)),
        "inner_validation_rows": int(len(inner_valid)),
        "train_windows": int(decisions.iloc[fold.train]["window_id"].nunique()),
        "validation_windows": int(
            decisions.iloc[fold.valid]["window_id"].nunique()
        ),
        "train_episodes": int(
            decisions.iloc[fold.train]["channel_episode_id"].nunique()
        ),
        "validation_episodes": int(
            decisions.iloc[fold.valid]["channel_episode_id"].nunique()
        ),
        "episode_overlap": int(len(overlap)),
        "train_positive_pct": float(
            100.0 * decisions.iloc[fold.train]["label_net_positive"].mean()
        ),
        "validation_positive_pct": float(
            100.0 * decisions.iloc[fold.valid]["label_net_positive"].mean()
        ),
        "inner_uniqueness_mean": float(inner_uniqueness.mean()),
        "outer_uniqueness_mean": float(outer_uniqueness.mean()),
        "inner_uniqueness_effective_rows": effective_sample_size(inner_uniqueness),
        "outer_uniqueness_effective_rows": effective_sample_size(outer_uniqueness),
        "chosen_quantile": float(chosen_quantile),
        "threshold": threshold,
        "submitted_entries": int(len(entries)),
        "filled_entries": int(entries["filled"].sum()) if len(entries) else 0,
        "validation_start": fold.valid_start.isoformat(),
        "validation_end_exclusive": fold.valid_end.isoformat(),
    }
    if overlap:
        raise AssertionError(f"outer fold shares episodes: {sorted(overlap)}")
    return outer_scores, entries, actions, inner_table, audit


def _artifact_paths(output_dir: Path, fold_id: str, model_name: str) -> dict[str, Path]:
    stem = f"{fold_id}_{model_name}"
    fold_dir = output_dir / "folds"
    return {
        "scores": fold_dir / f"scores_{stem}.parquet",
        "entries": fold_dir / f"entries_{stem}.parquet",
        "actions": fold_dir / f"actions_{stem}.parquet",
        "frontier": fold_dir / f"thresholds_{stem}.csv",
        "audit": fold_dir / f"audit_{stem}.json",
    }


def _artifacts_exist(paths: dict[str, Path]) -> bool:
    return all(path.exists() for path in paths.values())


def _write_state(output_dir: Path, **values: object) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_state.json").write_text(
        json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _evaluation_days(folds: list[PurgedFold]) -> int:
    return int(sum((fold.valid_end - fold.valid_start).days for fold in folds))


def _ledger_hash(values: pd.Series) -> str:
    return _json_hash(sorted(values.astype(str).tolist()))


def _consolidate(
    *,
    output_dir: Path,
    models: tuple[str, ...],
    folds: list[PurgedFold],
    protocol: dict[str, object],
) -> None:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    days = _evaluation_days(folds)
    summaries: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    model_ledgers: dict[str, pd.DataFrame] = {}
    baseline_scores: pd.DataFrame | None = None
    all_audits: list[dict[str, object]] = []
    for model_name in models:
        scores = pd.concat(
            [
                pd.read_parquet(_artifact_paths(output_dir, fold.fold_id, model_name)["scores"])
                for fold in folds
            ],
            ignore_index=True,
        )
        entries = pd.concat(
            [
                pd.read_parquet(_artifact_paths(output_dir, fold.fold_id, model_name)["entries"])
                for fold in folds
            ],
            ignore_index=True,
        )
        actions = pd.concat(
            [
                pd.read_parquet(_artifact_paths(output_dir, fold.fold_id, model_name)["actions"])
                for fold in folds
            ],
            ignore_index=True,
        )
        frontiers = pd.concat(
            [
                pd.read_csv(_artifact_paths(output_dir, fold.fold_id, model_name)["frontier"])
                for fold in folds
            ],
            ignore_index=True,
        )
        for fold in folds:
            all_audits.append(
                json.loads(
                    _artifact_paths(output_dir, fold.fold_id, model_name)["audit"].read_text(
                        encoding="utf-8"
                    )
                )
            )
        scores.to_parquet(output_dir / f"oof_decisions_{model_name}.parquet", index=False)
        entries.to_parquet(output_dir / f"oof_entries_{model_name}.parquet", index=False)
        actions.to_parquet(output_dir / f"oof_actions_{model_name}.parquet", index=False)
        frontiers.to_csv(output_dir / f"threshold_frontier_{model_name}.csv", index=False)
        model_ledgers[model_name] = entries
        if baseline_scores is None:
            baseline_scores = scores[scores["minutes_since_t2"].eq(0)].copy()

        target = scores["label_net_positive"].to_numpy(dtype=np.int8)
        auc = (
            float(roc_auc_score(target, scores["score"]))
            if len(np.unique(target)) == 2
            else np.nan
        )
        brier = float(brier_score_loss(target, scores["score"]))
        counts = actions["action"].value_counts()
        action_rows.append(
            {
                "model": model_name,
                "wait_actions": int(counts.get("WAIT", 0)),
                "enter_actions": int(counts.get("ENTER", 0)),
                "skip_actions": int(counts.get("SKIP", 0)),
            }
        )
        for label, capacity in CAPACITIES:
            replay = replay_entry_capacity(
                entries,
                capacity=capacity,
                evaluation_days=days,
            )
            replay.orders.to_parquet(
                output_dir / f"orders_{model_name}_{label}.parquet", index=False
            )
            summaries.append(
                {
                    "model": model_name,
                    "policy": "nested_first_crossing",
                    "capacity": label,
                    "roc_auc": auc,
                    "brier": brier,
                    **replay.metrics,
                }
            )

    if baseline_scores is None:
        raise AssertionError("full consolidation has no baseline rows")
    baseline = replay_entry_capacity(
        baseline_scores,
        capacity=None,
        evaluation_days=days,
    )
    baseline.orders.to_parquet(output_dir / "orders_baseline_immediate.parquet", index=False)
    baseline_row = {
        "model": "fixed_immediate_entry",
        "policy": "enter_at_t2",
        "capacity": "unlimited",
        "roc_auc": np.nan,
        "brier": np.nan,
        **baseline.metrics,
    }
    summaries.append(baseline_row)
    summary = pd.DataFrame(summaries)
    action_counts = pd.DataFrame(action_rows)
    audit_frame = pd.DataFrame(all_audits)
    summary.to_csv(output_dir / "model_capacity_summary.csv", index=False)
    action_counts.to_csv(output_dir / "action_counts.csv", index=False)
    audit_frame.to_csv(output_dir / "fold_audit.csv", index=False)

    primary = summary[
        summary["model"].isin(models) & summary["capacity"].eq("unlimited")
    ].copy()
    eligible = primary[
        primary["trades_per_day"].ge(1.0)
        & primary["long_trades"].gt(0)
        & primary["short_trades"].gt(0)
    ]
    pool = eligible if not eligible.empty else primary
    winner = (
        pool.assign(
            _mean=pool["mean_net_r"].fillna(-np.inf),
            _total=pool["total_net_r"].fillna(-np.inf),
        )
        .sort_values(["_mean", "_total", "model"], ascending=[False, False, True])
        .iloc[0]
    )
    selected_model = str(winner["model"])
    winner_orders = model_ledgers[selected_model]
    baseline_mean = float(baseline.metrics["mean_net_r"])
    economic_success = bool(
        winner["trades_per_day"] >= 1.0
        and winner["long_trades"] > 0
        and winner["short_trades"] > 0
        and winner["mean_net_r"] >= 0.05
        and winner["bootstrap_low"] > 0.0
        and winner["mean_net_r"] > baseline_mean
    )
    freeze = {
        "protocol_hash": protocol["protocol_hash"],
        "selected_model": selected_model,
        "selected_policy": "nested_first_crossing",
        "selected_capacity": "unlimited",
        "promotion": "promoted" if economic_success else "exploratory",
        "frequency_constraint_met": bool(
            winner["trades_per_day"] >= 1.0
            and winner["long_trades"] > 0
            and winner["short_trades"] > 0
        ),
        "economic_success": economic_success,
        "baseline_mean_net_r": baseline_mean,
        "selected_mean_net_r": float(winner["mean_net_r"]),
        "selected_trades_per_day": float(winner["trades_per_day"]),
        "decision_ledger_hash": _ledger_hash(winner_orders["decision_id"]),
        "filled_ledger_hash": _ledger_hash(
            winner_orders.loc[winner_orders["filled"].astype(bool), "decision_id"]
        ),
        "submitted_orders": int(len(winner_orders)),
        "filled_trades": int(winner_orders["filled"].sum()),
        "forward_or_lockbox_loaded": False,
    }
    (output_dir / "entry_freeze.json").write_text(
        json.dumps(freeze, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run(
    *,
    models: Iterable[str] = MODEL_NAMES,
    limit_folds: int | None = None,
    rebuild: bool = False,
    output_dir: Path = OUT,
) -> None:
    """Run requested fold/model cells and consolidate only the full matrix."""
    model_names = _normalise_models(models)
    if limit_folds is not None and limit_folds < 1:
        raise ValueError("limit_folds must be positive")
    output_dir = Path(output_dir)
    protocol = write_protocol(
        output_dir,
        models=model_names,
        overwrite=rebuild,
    )
    _write_state(
        output_dir,
        status="running",
        active="dataset",
        completed=[],
        protocol_hash=protocol["protocol_hash"],
        consolidated=False,
    )
    decisions, sequences, dataset_manifest = prepare_dataset(output_dir=output_dir)
    decisions = decisions.sort_values(
        ["decision_time", "decision_id"], kind="stable"
    ).reset_index(drop=True)
    if dataset_manifest["decision_ledger_hash"] != _json_hash(
        decisions["decision_id"].astype(str).tolist()
    ):
        raise ValueError("sorted decision ledger does not match its manifest")
    folds = expanding_purged_folds(decisions)
    selected_folds = folds[:limit_folds] if limit_folds is not None else folds
    completed: list[str] = []
    (output_dir / "folds").mkdir(parents=True, exist_ok=True)
    for fold in selected_folds:
        if not len(fold.train) or not len(fold.valid):
            raise ValueError(f"empty registered fold: {fold.fold_id}")
        for model_name in model_names:
            key = f"{fold.fold_id}:{model_name}"
            paths = _artifact_paths(output_dir, fold.fold_id, model_name)
            if _artifacts_exist(paths) and not rebuild:
                completed.append(key)
                print(f"resume {key}", flush=True)
                continue
            _write_state(
                output_dir,
                status="running",
                active=key,
                completed=completed,
                protocol_hash=protocol["protocol_hash"],
                consolidated=False,
            )
            print(f"fit {key}", flush=True)
            scores, entries, actions, frontier, audit = score_outer_fold(
                decisions, sequences, fold, model_name
            )
            scores.to_parquet(paths["scores"], index=False)
            entries.to_parquet(paths["entries"], index=False)
            actions.to_parquet(paths["actions"], index=False)
            frontier.to_csv(paths["frontier"], index=False)
            paths["audit"].write_text(
                json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            completed.append(key)

    full_matrix = (
        limit_folds is None
        and model_names == MODEL_NAMES
        and len(folds) == 7
        and all(
            _artifacts_exist(_artifact_paths(output_dir, fold.fold_id, model_name))
            for fold in folds
            for model_name in MODEL_NAMES
        )
    )
    if full_matrix:
        _consolidate(
            output_dir=output_dir,
            models=model_names,
            folds=folds,
            protocol=protocol,
        )
    _write_state(
        output_dir,
        status="complete",
        active=None,
        completed=completed,
        expected=int(len(selected_folds) * len(model_names)),
        protocol_hash=protocol["protocol_hash"],
        consolidated=full_matrix,
        max_loaded_timestamp=dataset_manifest["max_loaded_timestamp"],
        forward_or_lockbox_loaded=False,
    )
    print(
        f"complete: {len(completed)}/{len(selected_folds) * len(model_names)} "
        f"fold-model cells; consolidated={full_matrix}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES)
    )
    parser.add_argument("--limit-folds", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    run(
        models=tuple(args.models),
        limit_folds=args.limit_folds,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
