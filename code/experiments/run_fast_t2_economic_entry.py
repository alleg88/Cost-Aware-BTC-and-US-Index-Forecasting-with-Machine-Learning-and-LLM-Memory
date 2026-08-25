"""Run the dev-only Fast-T2 economic enter-versus-skip experiment."""
from __future__ import annotations

import argparse
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
from evaluation.fast_t2_action_policy import replay_entry_capacity
from evaluation.fast_t2_economic_policy import (
    SELECTION_NET_R_CLIP,
    economic_first_crossing_entries,
    mark_frequency_eligibility,
    select_primary_economic_arm,
    select_economic_threshold,
    summarise_loss_streaks,
)
from experiments.fast_t2_economic_models import fit_predict_economic_model
from experiments.fast_t2_entry_dataset import ENTRY_FEATURE_COLUMNS
from experiments.run_fast_t2_entry_policy import inner_episode_purged_indices


CODE_ROOT = Path(__file__).resolve().parents[1]
ENTRY_DIR = CODE_ROOT / "experiments" / "cache" / "fast_t2_entry_policy" / "dev"
OUT = CODE_ROOT / "experiments" / "cache" / "fast_t2_economic_entry" / "dev"
ECONOMIC_ARMS = (
    "ridge_pooled",
    "catboost_pooled",
    "catboost_split_side",
)
RR_SENSITIVITY = (None, 1.0, 1.5, 2.0)
CAPACITY = None

SCORE_COLUMNS = (
    "window_id",
    "decision_id",
    "side",
    "side_sign",
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
    "rr_proxy",
    "distance_to_stop_bps",
    "distance_to_target_bps",
)


def _json_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rr_label(min_rr: float | None) -> str:
    return "none" if min_rr is None else f"rr_{min_rr:.1f}".replace(".", "_")


def _arm_spec(arm: str) -> tuple[str, str]:
    mapping = {
        "ridge_pooled": ("ridge", "pooled"),
        "catboost_pooled": ("catboost", "pooled"),
        "catboost_split_side": ("catboost", "split_side"),
        "xgboost_pooled": ("xgboost", "pooled"),
        "xgboost_split_side": ("xgboost", "split_side"),
    }
    try:
        return mapping[arm]
    except KeyError as exc:
        raise ValueError(f"unsupported economic arm: {arm}") from exc


def _normalise_arms(arms: Iterable[str]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(arms))
    unknown = sorted(set(requested).difference(ECONOMIC_ARMS))
    if unknown:
        raise ValueError(f"unsupported economic arms: {unknown}")
    if not requested:
        raise ValueError("at least one economic arm is required")
    return tuple(arm for arm in ECONOMIC_ARMS if arm in requested)


def build_economic_protocol(
    entry_protocol: dict[str, object],
    entry_manifest: dict[str, object],
    *,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """Build a path-independent continuation protocol from frozen E inputs."""
    del output_dir
    if bool(entry_protocol.get("forward_or_lockbox_loaded")):
        raise ValueError("forward and lockbox must remain sealed")
    if entry_protocol.get("period_end_exclusive") != "2025-07-01T00:00:00+00:00":
        raise ValueError("economic continuation must remain dev-only")
    payload: dict[str, object] = {
        "stage": "development",
        "source_entry_protocol_hash": entry_protocol["protocol_hash"],
        "source_dataset_hash": entry_manifest["dataset_hash"],
        "source_decision_ledger_hash": entry_manifest["decision_ledger_hash"],
        "period_start": entry_protocol["period_start"],
        "period_end_exclusive": entry_protocol["period_end_exclusive"],
        "target": "winsorised_net_r_enter_minus_skip_0R",
        "target_winsorisation": "training-only 1st/99th percentiles",
        "selection_net_r_clip": list(SELECTION_NET_R_CLIP),
        "arms": list(ECONOMIC_ARMS),
        "model_parameters": {
            "ridge": {"alpha": 10.0, "scaled": True},
            "catboost": {
                "iterations": 300,
                "depth": 4,
                "learning_rate": 0.03,
                "l2_leaf_reg": 10.0,
                "loss_function": "RMSE",
                "random_seed": 42,
                "thread_count": 1,
            },
        },
        "rr_sensitivity": list(RR_SENSITIVITY),
        "primary_rr": None,
        "minimum_trades_per_day": 1.0,
        "threshold_selection": "inner chronological robust net R under frequency floor",
        "validation": "seven expanding six-month episode-purged folds",
        "uniqueness_weighting": "training labels only; no class balancing",
        "primary_portfolio": "all signals (unlimited)",
        "forward_or_lockbox_loaded": False,
    }
    payload["protocol_hash"] = _json_hash(payload)
    return payload


def write_economic_protocol(
    output_dir: Path,
    entry_protocol: dict[str, object],
    entry_manifest: dict[str, object],
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = build_economic_protocol(entry_protocol, entry_manifest)
    path = output_dir / "protocol.json"
    if path.exists() and not overwrite:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("protocol_hash") != expected["protocol_hash"]:
            raise ValueError("economic protocol hash mismatch; use --rebuild")
        return stored
    path.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return expected


def _load_entry_inputs() -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    required = {
        "run_state.json",
        "protocol.json",
        "dataset_manifest.json",
        "entry_decisions.parquet",
        "entry_freeze.json",
    }
    missing = sorted(name for name in required if not (ENTRY_DIR / name).exists())
    if missing:
        raise FileNotFoundError(f"Notebook E artifacts missing: {missing}")
    state = json.loads((ENTRY_DIR / "run_state.json").read_text(encoding="utf-8"))
    protocol = json.loads((ENTRY_DIR / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (ENTRY_DIR / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if state.get("status") != "complete" or not state.get("consolidated"):
        raise ValueError("economic continuation requires complete Notebook E artifacts")
    if bool(state.get("forward_or_lockbox_loaded")) or bool(
        protocol.get("forward_or_lockbox_loaded")
    ):
        raise ValueError("forward and lockbox must remain sealed")
    decisions = pd.read_parquet(ENTRY_DIR / "entry_decisions.parquet")
    decisions = decisions.sort_values(
        ["decision_time", "decision_id"], kind="stable"
    ).reset_index(drop=True)
    ledger_hash = _json_hash(decisions["decision_id"].astype(str).tolist())
    if ledger_hash != manifest.get("decision_ledger_hash"):
        raise ValueError("Notebook E decision ledger hash mismatch")
    end = pd.Timestamp(protocol["period_end_exclusive"])
    if pd.to_datetime(decisions["label_end"], utc=True).max() >= end:
        raise AssertionError("economic labels reach the sealed post-dev period")
    if pd.Timestamp(manifest["max_loaded_timestamp"]) >= end:
        raise AssertionError("economic source loaded the sealed post-dev period")
    return decisions, protocol, manifest


def _fit_and_score(
    decisions: pd.DataFrame,
    *,
    model_name: str,
    variant: str,
    train_index: np.ndarray,
    score_index: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, float | int]]:
    train_index = np.asarray(train_index, dtype=np.int64)
    score_index = np.asarray(score_index, dtype=np.int64)
    if not len(train_index) or not len(score_index):
        raise ValueError("economic fit and score indices cannot be empty")
    train_positions = set(train_index.tolist())
    score_only = np.array(
        [position for position in score_index if position not in train_positions],
        dtype=np.int64,
    )
    train_episodes = set(decisions.iloc[train_index]["channel_episode_id"])
    score_episodes = set(
        decisions.iloc[score_only]["channel_episode_id"] if len(score_only) else []
    )
    if train_episodes.intersection(score_episodes):
        raise AssertionError("economic fit and score indices share episodes")
    raw_uniqueness = interval_uniqueness(decisions, train_index, normalize=False)
    weights = raw_uniqueness / raw_uniqueness.mean()
    prediction = fit_predict_economic_model(
        model_name,
        variant,
        decisions.iloc[train_index][list(ENTRY_FEATURE_COLUMNS)].to_numpy(float),
        decisions.iloc[train_index]["r_net"].to_numpy(float),
        weights,
        decisions.iloc[train_index]["side_sign"].to_numpy(float),
        decisions.iloc[score_index][list(ENTRY_FEATURE_COLUMNS)].to_numpy(float),
        decisions.iloc[score_index]["side_sign"].to_numpy(float),
    )
    output = decisions.iloc[score_index][list(SCORE_COLUMNS)].copy()
    output["score"] = prediction.scores
    output["target_low"] = prediction.target_low
    output["target_high"] = prediction.target_high
    metadata: dict[str, float | int] = {
        "target_low": prediction.target_low,
        "target_high": prediction.target_high,
        "fitted_models": prediction.fitted_models,
    }
    return output.reset_index(drop=True), raw_uniqueness, metadata


def replay_rr_sensitivities(
    scored: pd.DataFrame,
    *,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply every RR sensitivity to one unchanged economic threshold."""
    entry_frames: list[pd.DataFrame] = []
    action_frames: list[pd.DataFrame] = []
    for min_rr in RR_SENSITIVITY:
        label = _rr_label(min_rr)
        entries, actions = economic_first_crossing_entries(
            scored, threshold=threshold, min_rr=min_rr
        )
        for frame in (entries, actions):
            frame.insert(0, "rr_label", label)
            frame.insert(1, "min_rr", min_rr)
            frame.insert(2, "outer_threshold", threshold)
        entry_frames.append(entries)
        action_frames.append(actions)
    return (
        pd.concat(entry_frames, ignore_index=True),
        pd.concat(action_frames, ignore_index=True),
    )


def score_outer_economic_fold(
    decisions: pd.DataFrame,
    fold: PurgedFold,
    arm: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Fit one economic arm and score one untouched outer validation fold."""
    model_name, variant = _arm_spec(arm)
    inner_cut = fold.valid_start - pd.DateOffset(months=6)
    inner_fit, inner_valid = inner_episode_purged_indices(
        decisions,
        fold.train,
        inner_cut=inner_cut,
        outer_valid_start=fold.valid_start,
    )
    inner_scores, inner_uniqueness, inner_meta = _fit_and_score(
        decisions,
        model_name=model_name,
        variant=variant,
        train_index=inner_fit,
        score_index=inner_valid,
    )
    primary_quantile, _ = select_economic_threshold(
        inner_scores,
        evaluation_days=(fold.valid_start - inner_cut).days,
        min_rr=None,
        min_trades_per_day=1.0,
    )
    frontier_frames: list[pd.DataFrame] = []
    for min_rr in RR_SENSITIVITY:
        label = _rr_label(min_rr)
        _, table = select_economic_threshold(
            inner_scores,
            evaluation_days=(fold.valid_start - inner_cut).days,
            min_rr=min_rr,
            min_trades_per_day=1.0,
        )
        table.insert(0, "arm", arm)
        table.insert(1, "model", model_name)
        table.insert(2, "variant", variant)
        table.insert(3, "fold_id", fold.fold_id)
        table.insert(4, "rr_label", label)
        table["chosen"] = table["quantile"].eq(primary_quantile)
        frontier_frames.append(table)

    combined = np.concatenate([fold.train, fold.valid])
    combined_scores, outer_uniqueness, outer_meta = _fit_and_score(
        decisions,
        model_name=model_name,
        variant=variant,
        train_index=fold.train,
        score_index=combined,
    )
    train_scores = combined_scores.iloc[: len(fold.train)]
    outer_scores = combined_scores.iloc[len(fold.train):].reset_index(drop=True)

    threshold = float(train_scores["score"].quantile(primary_quantile))
    entries, actions = replay_rr_sensitivities(
        outer_scores, threshold=threshold
    )
    for frame in (entries, actions):
        frame.insert(0, "arm", arm)
        frame.insert(1, "model", model_name)
        frame.insert(2, "variant", variant)
        frame.insert(3, "fold_id", fold.fold_id)

    outer_scores.insert(0, "arm", arm)
    outer_scores.insert(1, "model", model_name)
    outer_scores.insert(2, "variant", variant)
    outer_scores.insert(3, "fold_id", fold.fold_id)

    frontier = pd.concat(frontier_frames, ignore_index=True)
    frontier["outer_threshold"] = threshold
    overlap = set(decisions.iloc[fold.train]["channel_episode_id"]).intersection(
        decisions.iloc[fold.valid]["channel_episode_id"]
    )
    validation_target = outer_scores["r_net"].clip(
        outer_meta["target_low"], outer_meta["target_high"]
    )
    audit: dict[str, object] = {
        "arm": arm,
        "model": model_name,
        "variant": variant,
        "fold_id": fold.fold_id,
        "train_rows": int(len(fold.train)),
        "validation_rows": int(len(fold.valid)),
        "inner_fit_rows": int(len(inner_fit)),
        "inner_validation_rows": int(len(inner_valid)),
        "train_episodes": int(decisions.iloc[fold.train]["channel_episode_id"].nunique()),
        "validation_episodes": int(decisions.iloc[fold.valid]["channel_episode_id"].nunique()),
        "episode_overlap": int(len(overlap)),
        "inner_uniqueness_mean": float(inner_uniqueness.mean()),
        "outer_uniqueness_mean": float(outer_uniqueness.mean()),
        "outer_uniqueness_effective_rows": effective_sample_size(outer_uniqueness),
        "inner_target_low": inner_meta["target_low"],
        "inner_target_high": inner_meta["target_high"],
        "outer_target_low": outer_meta["target_low"],
        "outer_target_high": outer_meta["target_high"],
        "outer_fitted_models": outer_meta["fitted_models"],
        "validation_mae_robust_r": float(
            np.mean(np.abs(validation_target - outer_scores["score"]))
        ),
        "validation_rank_ic": float(
            outer_scores["score"].corr(outer_scores["r_net"], method="spearman")
        ),
        "validation_start": fold.valid_start.isoformat(),
        "validation_end_exclusive": fold.valid_end.isoformat(),
    }
    for min_rr in RR_SENSITIVITY:
        label = _rr_label(min_rr)
        audit[f"chosen_quantile_{label}"] = primary_quantile
        audit[f"threshold_{label}"] = threshold
    if overlap:
        raise AssertionError(f"economic outer fold shares episodes: {sorted(overlap)}")
    return (
        outer_scores,
        entries,
        actions,
        frontier,
        audit,
    )


def _artifact_paths(output_dir: Path, fold_id: str, arm: str) -> dict[str, Path]:
    stem = f"{fold_id}_{arm}"
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
    arms: tuple[str, ...],
    folds: list[PurgedFold],
    protocol: dict[str, object],
) -> None:
    days = _evaluation_days(folds)
    summaries: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    primary_ledgers: dict[str, pd.DataFrame] = {}
    decile_frames: list[pd.DataFrame] = []
    for arm in arms:
        model_name, variant = _arm_spec(arm)
        scores = pd.concat(
            [pd.read_parquet(_artifact_paths(output_dir, f.fold_id, arm)["scores"]) for f in folds],
            ignore_index=True,
        )
        entries = pd.concat(
            [pd.read_parquet(_artifact_paths(output_dir, f.fold_id, arm)["entries"]) for f in folds],
            ignore_index=True,
        )
        actions = pd.concat(
            [pd.read_parquet(_artifact_paths(output_dir, f.fold_id, arm)["actions"]) for f in folds],
            ignore_index=True,
        )
        frontiers = pd.concat(
            [pd.read_csv(_artifact_paths(output_dir, f.fold_id, arm)["frontier"]) for f in folds],
            ignore_index=True,
        )
        for fold in folds:
            audits.append(
                json.loads(
                    _artifact_paths(output_dir, fold.fold_id, arm)["audit"].read_text(
                        encoding="utf-8"
                    )
                )
            )
        scores.to_parquet(output_dir / f"oof_scores_{arm}.parquet", index=False)
        entries.to_parquet(output_dir / f"oof_entries_{arm}.parquet", index=False)
        actions.to_parquet(output_dir / f"oof_actions_{arm}.parquet", index=False)
        frontiers.to_csv(output_dir / f"threshold_frontier_{arm}.csv", index=False)

        counts = actions.groupby("rr_label")["action"].value_counts().unstack(fill_value=0)
        for min_rr in RR_SENSITIVITY:
            label = _rr_label(min_rr)
            ledger = entries[entries["rr_label"].eq(label)].copy()
            replay = replay_entry_capacity(
                ledger, capacity=CAPACITY, evaluation_days=days
            )
            replay.orders.to_parquet(
                output_dir / f"orders_{arm}_{label}.parquet", index=False
            )
            filled = replay.orders[replay.orders["filled"].astype(bool)]
            robust_mean = (
                float(filled["r_net"].clip(*SELECTION_NET_R_CLIP).mean())
                if len(filled)
                else np.nan
            )
            summaries.append(
                {
                    "arm": arm,
                    "model": model_name,
                    "variant": variant,
                    "rr_label": label,
                    "min_rr": min_rr,
                    "primary_rr": min_rr is None,
                    "robust_mean_net_r": robust_mean,
                    **replay.metrics,
                }
            )
            action_rows.append(
                {
                    "arm": arm,
                    "model": model_name,
                    "variant": variant,
                    "rr_label": label,
                    "wait_actions": int(counts.loc[label].get("WAIT", 0)),
                    "enter_actions": int(counts.loc[label].get("ENTER", 0)),
                    "skip_actions": int(counts.loc[label].get("SKIP", 0)),
                }
            )
            if min_rr is None:
                primary_ledgers[arm] = replay.orders
                selected = replay.orders[replay.orders["filled"].astype(bool)].copy()
                if len(selected):
                    bins = min(10, len(selected))
                    selected["score_decile"] = pd.qcut(
                        selected["score"].rank(method="first"), bins, labels=False
                    ) + 1
                    deciles = selected.groupby("score_decile", as_index=False).agg(
                        trades=("decision_id", "size"),
                        mean_score=("score", "mean"),
                        mean_net_r=("r_net", "mean"),
                        win_rate=("r_net", lambda values: float((values > 0).mean())),
                    )
                    deciles.insert(0, "arm", arm)
                    decile_frames.append(deciles)

    e_orders = pd.read_parquet(ENTRY_DIR / "orders_logreg_unlimited.parquet")
    e_baseline = replay_entry_capacity(
        e_orders, capacity=None, evaluation_days=days
    )
    e_filled = e_baseline.orders[e_baseline.orders["filled"].astype(bool)]
    summaries.append(
        {
            "arm": "notebook_e_logreg",
            "model": "logreg",
            "variant": "binary_sign_baseline",
            "rr_label": "frozen",
            "min_rr": np.nan,
            "primary_rr": False,
            "robust_mean_net_r": float(
                e_filled["r_net"].clip(*SELECTION_NET_R_CLIP).mean()
            ),
            **e_baseline.metrics,
        }
    )
    summary = pd.DataFrame(summaries)
    summary = mark_frequency_eligibility(summary, minimum=1.0)
    primary = summary[
        summary["arm"].isin(arms) & summary["rr_label"].eq("none")
    ].copy()
    winner = select_primary_economic_arm(primary)
    selected_arm = str(winner["arm"])
    selected_orders = primary_ledgers[selected_arm]
    selected_orders.to_parquet(output_dir / "selected_orders.parquet", index=False)

    loss_rows: list[dict[str, object]] = []
    for label, ledger in [
        ("Notebook E LogReg", e_baseline.orders),
        *[(arm, primary_ledgers[arm]) for arm in arms],
    ]:
        filled = ledger[ledger["filled"].astype(bool)]
        loss_rows.append({"policy": label, **summarise_loss_streaks(filled)})
    pd.DataFrame(loss_rows).to_csv(
        output_dir / "loss_streak_summary.csv", index=False
    )
    if decile_frames:
        pd.concat(decile_frames, ignore_index=True).to_csv(
            output_dir / "score_deciles.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(output_dir / "score_deciles.csv", index=False)
    pd.DataFrame(audits).to_csv(output_dir / "fold_audit.csv", index=False)
    pd.DataFrame(action_rows).to_csv(output_dir / "action_counts.csv", index=False)
    summary.to_csv(output_dir / "economic_policy_summary.csv", index=False)

    baseline_mean = float(e_baseline.metrics["mean_net_r"])
    economic_success = bool(
        winner["frequency_eligible"]
        and winner["mean_net_r"] >= 0.05
        and winner["bootstrap_low"] > 0.0
        and winner["mean_net_r"] > baseline_mean
    )
    result = {
        "protocol_hash": protocol["protocol_hash"],
        "selected_arm": selected_arm,
        "selected_rr_label": "none",
        "selection_uses_rr_sensitivity": False,
        "minimum_trades_per_day": 1.0,
        "frequency_constraint_met": bool(winner["frequency_eligible"]),
        "selected_filled_trades": int(winner["filled_trades"]),
        "selected_trades_per_day": float(winner["trades_per_day"]),
        "selected_mean_net_r": float(winner["mean_net_r"]),
        "selected_robust_mean_net_r": float(winner["robust_mean_net_r"]),
        "selected_bootstrap_low": float(winner["bootstrap_low"]),
        "selected_bootstrap_high": float(winner["bootstrap_high"]),
        "notebook_e_baseline_mean_net_r": baseline_mean,
        "improvement_vs_notebook_e_r": float(winner["mean_net_r"] - baseline_mean),
        "economic_success": economic_success,
        "status": "promoted" if economic_success else "exploratory",
        "selected_ledger_hash": _ledger_hash(selected_orders["decision_id"]),
        "primary_trials": len(arms),
        "rr_rows_are_sensitivity_only": True,
        "removed_from_active_path": [
            "early_exit_model",
            "accuracy_or_win_rate_selection",
            "gru_lstm_entry_models",
            "capacity_threshold_search",
            "split_side_catboost",
            "hard_rr_gate",
        ],
        "forward_or_lockbox_loaded": False,
    }
    (output_dir / "economic_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run(
    *,
    arms: Iterable[str] = ECONOMIC_ARMS,
    limit_folds: int | None = None,
    rebuild: bool = False,
    output_dir: Path = OUT,
) -> None:
    """Run requested fold/arm cells and consolidate only the full matrix."""
    arm_names = _normalise_arms(arms)
    if limit_folds is not None and limit_folds < 1:
        raise ValueError("limit_folds must be positive")
    output_dir = Path(output_dir)
    decisions, entry_protocol, entry_manifest = _load_entry_inputs()
    protocol = write_economic_protocol(
        output_dir,
        entry_protocol,
        entry_manifest,
        overwrite=rebuild,
    )
    folds = expanding_purged_folds(decisions)
    selected_folds = folds[:limit_folds] if limit_folds is not None else folds
    completed: list[str] = []
    (output_dir / "folds").mkdir(parents=True, exist_ok=True)
    _write_state(
        output_dir,
        status="running",
        active="folds",
        completed=completed,
        consolidated=False,
        protocol_hash=protocol["protocol_hash"],
        forward_or_lockbox_loaded=False,
    )
    for fold in selected_folds:
        for arm in arm_names:
            key = f"{fold.fold_id}:{arm}"
            paths = _artifact_paths(output_dir, fold.fold_id, arm)
            if _artifacts_exist(paths) and not rebuild:
                completed.append(key)
                print(f"resume {key}", flush=True)
                continue
            _write_state(
                output_dir,
                status="running",
                active=key,
                completed=completed,
                consolidated=False,
                protocol_hash=protocol["protocol_hash"],
                forward_or_lockbox_loaded=False,
            )
            print(f"fit {key}", flush=True)
            scores, entries, actions, frontier, audit = score_outer_economic_fold(
                decisions, fold, arm
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
        and arm_names == ECONOMIC_ARMS
        and len(folds) == 7
        and all(
            _artifacts_exist(_artifact_paths(output_dir, fold.fold_id, arm))
            for fold in folds
            for arm in ECONOMIC_ARMS
        )
    )
    if full_matrix:
        _consolidate(
            output_dir=output_dir,
            arms=arm_names,
            folds=folds,
            protocol=protocol,
        )
    _write_state(
        output_dir,
        status="complete",
        active=None,
        completed=completed,
        expected=int(len(selected_folds) * len(arm_names)),
        consolidated=full_matrix,
        protocol_hash=protocol["protocol_hash"],
        max_loaded_timestamp=entry_manifest["max_loaded_timestamp"],
        forward_or_lockbox_loaded=False,
    )
    print(
        f"complete: {len(completed)}/{len(selected_folds) * len(arm_names)} "
        f"fold-arm cells; consolidated={full_matrix}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms", nargs="+", choices=ECONOMIC_ARMS, default=list(ECONOMIC_ARMS)
    )
    parser.add_argument("--limit-folds", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    run(arms=tuple(args.arms), limit_folds=args.limit_folds, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
