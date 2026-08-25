"""Run the resumable dev-only Fast-T2 early-exit experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.channel_window_validation import (
    PurgedFold,
    effective_sample_size,
    expanding_purged_folds,
    interval_uniqueness,
)
from evaluation.fast_t2_action_policy import replay_entry_capacity
from experiments.fast_t2_action_models import MODEL_NAMES, fit_predict_action_model
from experiments.fast_t2_exit_policy import (
    EXIT_FEATURE_COLUMNS,
    ExitPolicyConfig,
    build_exit_states,
    replay_exit_policy,
)
from experiments.fast_t2_study import DEV_END, DEV_START
from experiments.run_fast_t2_entry_policy import inner_episode_purged_indices


CODE_ROOT = Path(__file__).resolve().parents[1]
ENTRY_DIR = CODE_ROOT / "experiments" / "cache" / "fast_t2_entry_policy" / "dev"
OUT = CODE_ROOT / "experiments" / "cache" / "fast_t2_exit_policy" / "dev"
MINUTE_SOURCE = CODE_ROOT / "data" / "btcusdt_1m_2021_2026.parquet"
DEFAULT_CONFIG = ExitPolicyConfig()
EXIT_SCORE_QUANTILES = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
CAPACITIES: tuple[tuple[str, int | None], ...] = (
    ("unlimited", None),
    ("capacity_3", 3),
    ("capacity_1", 1),
)

STATE_SCORE_COLUMNS = (
    "state_id",
    "decision_id",
    "window_id",
    "side",
    "channel_episode_id",
    "entry_time",
    "entry_price",
    "stop_price",
    "target_price",
    "decision_time",
    "exit_decision_time",
    "label_start",
    "label_end",
    "active_end_time",
    "exit_now_price",
    "exit_now_r_net",
    "baseline_exit_time",
    "baseline_exit_price",
    "baseline_outcome",
    "baseline_r_net",
    "continue_positive",
    "minutes_held",
)


def _json_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ledger_hash(values: pd.Series) -> str:
    return _json_hash(sorted(values.astype(str).tolist()))


def load_entry_freeze(
    entry_dir: Path = ENTRY_DIR,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Load only a completed, consolidated, dev-only Notebook E handoff."""
    entry_dir = Path(entry_dir)
    state_path = entry_dir / "run_state.json"
    if not state_path.exists():
        raise ValueError("Notebook F requires a complete Notebook E run")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "complete" or not state.get("consolidated", False):
        raise ValueError("Notebook F requires a complete Notebook E run")
    protocol_path = entry_dir / "protocol.json"
    freeze_path = entry_dir / "entry_freeze.json"
    if not protocol_path.exists() or not freeze_path.exists():
        raise ValueError("Notebook F requires a complete Notebook E run")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("protocol_hash") != protocol.get("protocol_hash"):
        raise ValueError("Notebook E freeze does not match its protocol")
    if freeze.get("selected_model") not in MODEL_NAMES:
        raise ValueError("Notebook E selected an unsupported model")
    if any(
        bool(payload.get("forward_or_lockbox_loaded", False))
        for payload in (state, protocol, freeze)
    ):
        raise ValueError("Notebook E handoff is contaminated by later data")
    if pd.Timestamp(protocol["period_end_exclusive"]) != DEV_END:
        raise ValueError("Notebook E handoff is not development-bounded")
    return protocol, freeze, state


def build_exit_protocol(
    entry_protocol: dict[str, object],
    entry_freeze: dict[str, object],
    *,
    output_dir: Path | None = None,
    config: ExitPolicyConfig = DEFAULT_CONFIG,
) -> dict[str, object]:
    """Register the immutable Notebook E handoff and early-exit method."""
    del output_dir
    if entry_freeze.get("protocol_hash") != entry_protocol.get("protocol_hash"):
        raise ValueError("entry freeze protocol hash mismatch")
    if entry_protocol.get("forward_or_lockbox_loaded", False) or entry_freeze.get(
        "forward_or_lockbox_loaded", False
    ):
        raise ValueError("exit protocol cannot inherit later data")
    payload: dict[str, object] = {
        "stage": "development",
        "period_start": entry_protocol["period_start"],
        "period_end_exclusive": entry_protocol["period_end_exclusive"],
        "entry_protocol_hash": entry_protocol["protocol_hash"],
        "entry_decision_ledger_hash": entry_freeze["decision_ledger_hash"],
        "entry_filled_ledger_hash": entry_freeze["filled_ledger_hash"],
        "entry_model": entry_freeze["selected_model"],
        "entry_policy": entry_freeze["selected_policy"],
        "entry_policy_changed": False,
        "exit_config": asdict(config),
        "exit_features": list(EXIT_FEATURE_COLUMNS),
        "training_entry_selection": (
            "one stable label-blind feasible delay per Fast-T2 window"
        ),
        "validation_entries": "exact frozen Notebook E OOF filled ledger",
        "validation": "seven expanding six-month episode-purged folds",
        "threshold_selection": "inner chronological exit-score quantiles",
        "primary_portfolio": "all frozen entry signals (unlimited)",
        "capacity_sensitivity": [3, 1],
        "forward_or_lockbox_loaded": False,
    }
    payload["protocol_hash"] = _json_hash(payload)
    return payload


def write_exit_protocol(
    output_dir: Path,
    entry_protocol: dict[str, object],
    entry_freeze: dict[str, object],
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = build_exit_protocol(
        entry_protocol, entry_freeze, output_dir=output_dir
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


def validate_exit_resume(
    output_dir: Path,
    entry_protocol: dict[str, object],
    entry_freeze: dict[str, object],
) -> dict[str, object]:
    path = Path(output_dir) / "protocol.json"
    if not path.exists():
        raise FileNotFoundError(path)
    expected = build_exit_protocol(
        entry_protocol, entry_freeze, output_dir=output_dir
    )
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored.get("protocol_hash") != expected["protocol_hash"]:
        raise ValueError("protocol hash mismatch; use --rebuild intentionally")
    return stored


def select_training_proxy_entries(decisions: pd.DataFrame) -> pd.DataFrame:
    """Select one stable feasible delay per window without consulting labels."""
    required = {"window_id", "decision_id", "minutes_since_t2", "filled"}
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"entry decisions missing proxy columns: {missing}")
    feasible = decisions[decisions["filled"].astype(bool)].copy()
    chosen: list[int] = []
    for window_id, group in feasible.groupby("window_id", sort=False):
        ordered = group.sort_values(
            ["minutes_since_t2", "decision_id"], kind="stable"
        )
        digest = hashlib.sha256(str(window_id).encode("utf-8")).digest()
        position = int.from_bytes(digest[:8], "big") % len(ordered)
        chosen.append(int(ordered.index[position]))
    output = decisions.loc[chosen].copy()
    return output.sort_values(
        ["entry_time", "decision_id"]
        if "entry_time" in output
        else ["window_id", "decision_id"],
        kind="stable",
    ).reset_index(drop=True)


def _load_minutes() -> pd.DataFrame:
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
    if minute.index.has_duplicates or minute.index.max() >= DEV_END:
        raise ValueError("exit runner minute source violates the dev boundary")
    return minute


def _prepare_entry_ledgers(
    entry_freeze: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decisions = pd.read_parquet(ENTRY_DIR / "entry_decisions.parquet")
    proxies = select_training_proxy_entries(decisions)
    model = str(entry_freeze["selected_model"])
    frozen = pd.read_parquet(ENTRY_DIR / f"orders_{model}_unlimited.parquet")
    if _ledger_hash(frozen["decision_id"]) != entry_freeze["decision_ledger_hash"]:
        raise ValueError("frozen entry decision ledger hash mismatch")
    frozen_filled = frozen[frozen["filled"].astype(bool)].copy()
    if _ledger_hash(frozen_filled["decision_id"]) != entry_freeze["filled_ledger_hash"]:
        raise ValueError("frozen filled entry ledger hash mismatch")
    if len(frozen_filled) != int(entry_freeze["filled_trades"]):
        raise ValueError("frozen filled entry count mismatch")

    context_columns = ["decision_id", "channel_r2", "channel_width_pct"]
    context = decisions[context_columns].drop_duplicates("decision_id")
    missing_context = [
        column for column in ("channel_r2", "channel_width_pct")
        if column not in frozen_filled
    ]
    if missing_context:
        frozen_filled = frozen_filled.merge(
            context[["decision_id", *missing_context]],
            on="decision_id",
            how="left",
            validate="one_to_one",
        )
    if frozen_filled[["channel_r2", "channel_width_pct"]].isna().any().any():
        raise ValueError("frozen entries lack channel context")

    combined = pd.concat([frozen_filled, proxies], ignore_index=True, sort=False)
    combined = combined.drop_duplicates("decision_id", keep="first")
    proxy_ids = set(proxies["decision_id"].astype(str))
    frozen_ids = set(frozen_filled["decision_id"].astype(str))
    combined["is_training_proxy"] = combined["decision_id"].astype(str).isin(proxy_ids)
    combined["is_frozen_entry"] = combined["decision_id"].astype(str).isin(frozen_ids)
    return proxies, frozen_filled.reset_index(drop=True), combined.reset_index(drop=True)


def _dataset_manifest_payload(
    protocol: dict[str, object],
    proxies: pd.DataFrame,
    frozen: pd.DataFrame,
    combined: pd.DataFrame,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "entry_protocol_hash": protocol["entry_protocol_hash"],
        "exit_config": protocol["exit_config"],
        "proxy_ledger_hash": _ledger_hash(proxies["decision_id"]),
        "frozen_ledger_hash": _ledger_hash(frozen["decision_id"]),
        "combined_ledger_hash": _ledger_hash(combined["decision_id"]),
        "training_proxy_entries": int(len(proxies)),
        "frozen_entries": int(len(frozen)),
        "combined_entries": int(len(combined)),
        "period_end_exclusive": DEV_END.isoformat(),
        "minute_source_bytes": MINUTE_SOURCE.stat().st_size,
    }
    payload["dataset_hash"] = _json_hash(payload)
    return payload


def prepare_exit_dataset(
    *,
    protocol: dict[str, object],
    entry_freeze: dict[str, object],
    output_dir: Path = OUT,
    batch_size: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build or resume label-blind proxy states plus exact frozen states."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proxies, frozen, combined = _prepare_entry_ledgers(entry_freeze)
    expected = _dataset_manifest_payload(protocol, proxies, frozen, combined)
    state_path = output_dir / "exit_states.parquet"
    manifest_path = output_dir / "dataset_manifest.json"
    proxies.to_parquet(output_dir / "training_proxy_entries.parquet", index=False)
    frozen.to_parquet(output_dir / "frozen_entries.parquet", index=False)
    if state_path.exists() and manifest_path.exists():
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored.get("dataset_hash") == expected["dataset_hash"]:
            states = pd.read_parquet(state_path)
            if _ledger_hash(states["state_id"]) != stored.get("state_ledger_hash"):
                raise ValueError("exit state cache ledger hash mismatch")
            minute = _load_minutes()
            print(f"resume exit dataset: {len(states):,} states", flush=True)
            return states, proxies, frozen, minute, stored

    minute = _load_minutes()
    batch_dir = output_dir / "state_batches" / str(expected["dataset_hash"])[:12]
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_paths: list[Path] = []
    batches = (len(combined) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(combined), batch_size), start=1):
        path = batch_dir / f"states_{batch_number:03d}.parquet"
        batch_paths.append(path)
        if path.exists():
            print(f"resume state batch {batch_number}/{batches}", flush=True)
            continue
        batch = combined.iloc[start:start + batch_size]
        states = build_exit_states(batch, minute)
        states.to_parquet(path, index=False)
        print(
            f"built state batch {batch_number}/{batches}: {len(states):,} rows",
            flush=True,
        )
    states = pd.concat([pd.read_parquet(path) for path in batch_paths], ignore_index=True)
    roles = combined[["decision_id", "is_training_proxy", "is_frozen_entry"]].copy()
    frozen_folds = frozen[["decision_id", "fold_id"]].rename(
        columns={"fold_id": "frozen_fold_id"}
    )
    roles = roles.merge(frozen_folds, on="decision_id", how="left", validate="one_to_one")
    states = states.merge(roles, on="decision_id", how="left", validate="many_to_one")
    if states[["is_training_proxy", "is_frozen_entry"]].isna().any().any():
        raise ValueError("exit state roles are incomplete")
    states = states.sort_values(["decision_time", "state_id"], kind="stable").reset_index(
        drop=True
    )
    states.to_parquet(state_path, index=False)
    manifest = {
        **expected,
        "state_rows": int(len(states)),
        "state_entries": int(states["decision_id"].nunique()),
        "state_ledger_hash": _ledger_hash(states["state_id"]),
        "max_loaded_timestamp": minute.index.max().isoformat(),
        "batches": batches,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"built exit dataset: {len(states):,} states from "
        f"{states['decision_id'].nunique():,} entries",
        flush=True,
    )
    return states, proxies, frozen, minute, manifest


def _balanced_weights(labels: np.ndarray, uniqueness: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int8)
    classes, counts = np.unique(labels, return_counts=True)
    if set(classes) != {0, 1}:
        raise ValueError("exit training fold needs both binary classes")
    factors = {
        int(label): len(labels) / (2.0 * int(count))
        for label, count in zip(classes, counts, strict=True)
    }
    return uniqueness * np.array([factors[int(value)] for value in labels])


def _fit_exit_scores(
    states: pd.DataFrame,
    *,
    model_name: str,
    train_index: np.ndarray,
    score_index: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    if model_name in {"gru", "lstm"}:
        raise ValueError("the frozen recurrent exit branch needs registered sequences")
    train_index = np.asarray(train_index, dtype=np.int64)
    score_index = np.asarray(score_index, dtype=np.int64)
    if not len(train_index) or not len(score_index):
        raise ValueError("exit fit and score indices cannot be empty")
    train_positions = set(train_index.tolist())
    score_only = np.array(
        [position for position in score_index if position not in train_positions],
        dtype=np.int64,
    )
    train_episodes = set(states.iloc[train_index]["channel_episode_id"])
    score_only_episodes = set(
        states.iloc[score_only]["channel_episode_id"] if len(score_only) else []
    )
    if train_episodes.intersection(score_only_episodes):
        raise AssertionError("exit fit and score rows share channel episodes")
    labels = states.iloc[train_index]["continue_positive"].to_numpy(dtype=np.int8)
    raw_uniqueness = interval_uniqueness(states, train_index, normalize=False)
    uniqueness = raw_uniqueness / raw_uniqueness.mean()
    weights = _balanced_weights(labels, uniqueness)
    score = fit_predict_action_model(
        model_name,
        states.iloc[train_index][list(EXIT_FEATURE_COLUMNS)].to_numpy(dtype=float),
        None,
        labels,
        weights,
        states.iloc[score_index][list(EXIT_FEATURE_COLUMNS)].to_numpy(dtype=float),
        None,
    )
    output = states.iloc[score_index][list(STATE_SCORE_COLUMNS)].copy()
    output["score"] = np.asarray(score, dtype=float)
    return output.reset_index(drop=True), raw_uniqueness


def _exit_threshold_frontier(
    scored_states: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    finite = pd.to_numeric(scored_states["score"], errors="coerce").dropna()
    if finite.empty:
        raise ValueError("exit threshold selection needs finite scores")
    ordered = scored_states.sort_values(
        ["exit_decision_time", "state_id"], kind="stable"
    )
    baseline = ordered.drop_duplicates("decision_id").set_index("decision_id")[
        "baseline_r_net"
    ]
    rows: list[dict[str, object]] = []
    for quantile in EXIT_SCORE_QUANTILES:
        threshold = float(finite.quantile(quantile))
        exits = ordered[ordered["score"].lt(threshold)].drop_duplicates("decision_id")
        realised = baseline.copy()
        if len(exits):
            realised.loc[exits["decision_id"]] = exits.set_index("decision_id")[
                "exit_now_r_net"
            ]
        rows.append(
            {
                "quantile": float(quantile),
                "threshold": threshold,
                "entries": int(len(realised)),
                "model_exit_trades": int(len(exits)),
                "mean_net_r": float(realised.mean()),
                "total_net_r": float(realised.sum()),
                "baseline_mean_net_r": float(baseline.mean()),
                "mean_delta_r": float((realised - baseline).mean()),
            }
        )
    table = pd.DataFrame(rows)
    winner = (
        table.assign(
            _mean=table["mean_net_r"].fillna(-np.inf),
            _delta=table["mean_delta_r"].fillna(-np.inf),
        )
        .sort_values(
            ["_mean", "_delta", "quantile"],
            ascending=[False, False, True],
            kind="stable",
        )
        .iloc[0]
    )
    return float(winner["quantile"]), table


def score_outer_exit_fold(
    states: pd.DataFrame,
    proxies: pd.DataFrame,
    frozen_entries: pd.DataFrame,
    minute: pd.DataFrame,
    fold: PurgedFold,
    *,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    proxy_train = fold.train[
        states.iloc[fold.train]["is_training_proxy"].astype(bool).to_numpy()
    ]
    inner_cut = fold.valid_start - pd.DateOffset(months=6)
    inner_fit, inner_valid = inner_episode_purged_indices(
        states,
        proxy_train,
        inner_cut=inner_cut,
        outer_valid_start=fold.valid_start,
    )
    inner_scores, inner_uniqueness = _fit_exit_scores(
        states,
        model_name=model_name,
        train_index=inner_fit,
        score_index=inner_valid,
    )
    chosen_quantile, frontier = _exit_threshold_frontier(inner_scores)

    fold_entries = frozen_entries[frozen_entries["fold_id"].eq(fold.fold_id)].copy()
    frozen_ids = set(fold_entries["decision_id"].astype(str))
    outer_valid = fold.valid[
        states.iloc[fold.valid]["decision_id"].astype(str).isin(frozen_ids).to_numpy()
    ]
    if not len(fold_entries) or not len(outer_valid):
        raise ValueError(f"fold {fold.fold_id} has no frozen exit validation states")
    combined = np.concatenate([proxy_train, outer_valid])
    combined_scores, outer_uniqueness = _fit_exit_scores(
        states,
        model_name=model_name,
        train_index=proxy_train,
        score_index=combined,
    )
    train_scores = combined_scores.iloc[: len(proxy_train)]
    outer_scores = combined_scores.iloc[len(proxy_train):].reset_index(drop=True)
    threshold = float(train_scores["score"].quantile(chosen_quantile))
    trades, actions = replay_exit_policy(
        fold_entries,
        outer_scores[["decision_id", "exit_decision_time", "score"]],
        hold_threshold=threshold,
        minute_bars=minute,
        return_actions=True,
    )
    if trades["decision_id"].tolist() != fold_entries["decision_id"].astype(str).tolist():
        raise AssertionError("exit replay changed the frozen entry ledger")
    outer_scores.insert(0, "model", model_name)
    outer_scores.insert(1, "fold_id", fold.fold_id)
    trades["model"] = model_name
    actions.insert(0, "model", model_name)
    actions.insert(1, "fold_id", fold.fold_id)
    frontier.insert(0, "model", model_name)
    frontier.insert(1, "fold_id", fold.fold_id)
    frontier["chosen"] = frontier["quantile"].eq(chosen_quantile)
    frontier["outer_threshold"] = threshold

    overlap = set(states.iloc[proxy_train]["channel_episode_id"]).intersection(
        states.iloc[outer_valid]["channel_episode_id"]
    )
    audit: dict[str, object] = {
        "model": model_name,
        "fold_id": fold.fold_id,
        "train_state_rows": int(len(proxy_train)),
        "validation_state_rows": int(len(outer_valid)),
        "inner_fit_state_rows": int(len(inner_fit)),
        "inner_validation_state_rows": int(len(inner_valid)),
        "train_entries": int(states.iloc[proxy_train]["decision_id"].nunique()),
        "validation_entries": int(len(fold_entries)),
        "validation_entries_with_states": int(
            states.iloc[outer_valid]["decision_id"].nunique()
        ),
        "train_episodes": int(states.iloc[proxy_train]["channel_episode_id"].nunique()),
        "validation_episodes": int(
            states.iloc[outer_valid]["channel_episode_id"].nunique()
        ),
        "episode_overlap": int(len(overlap)),
        "train_continue_positive_pct": float(
            100.0 * states.iloc[proxy_train]["continue_positive"].mean()
        ),
        "validation_continue_positive_pct": float(
            100.0 * states.iloc[outer_valid]["continue_positive"].mean()
        ),
        "inner_uniqueness_mean": float(inner_uniqueness.mean()),
        "outer_uniqueness_mean": float(outer_uniqueness.mean()),
        "inner_uniqueness_effective_rows": effective_sample_size(inner_uniqueness),
        "outer_uniqueness_effective_rows": effective_sample_size(outer_uniqueness),
        "chosen_quantile": float(chosen_quantile),
        "threshold": threshold,
        "hold_actions": int(actions["action"].eq("HOLD").sum()),
        "exit_now_actions": int(actions["action"].eq("EXIT NOW").sum()),
        "missing_score_actions": int((~actions["score_available"]).sum()),
        "validation_start": fold.valid_start.isoformat(),
        "validation_end_exclusive": fold.valid_end.isoformat(),
    }
    if overlap:
        raise AssertionError(f"exit fold shares episodes: {sorted(overlap)}")
    return outer_scores, trades, actions, frontier, audit


def _artifact_paths(output_dir: Path, fold_id: str) -> dict[str, Path]:
    fold_dir = output_dir / "folds"
    return {
        "scores": fold_dir / f"scores_{fold_id}.parquet",
        "trades": fold_dir / f"trades_{fold_id}.parquet",
        "actions": fold_dir / f"actions_{fold_id}.parquet",
        "frontier": fold_dir / f"thresholds_{fold_id}.csv",
        "audit": fold_dir / f"audit_{fold_id}.json",
    }


def _artifacts_exist(paths: dict[str, Path]) -> bool:
    return all(path.exists() for path in paths.values())


def _write_state(output_dir: Path, **values: object) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_state.json").write_text(
        json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _paired_episode_bootstrap(
    trades: pd.DataFrame, *, replicates: int = 2_000, seed: int = 42
) -> tuple[float, float]:
    grouped = trades.groupby("channel_episode_id", sort=False)["delta_r"].agg(
        ["sum", "count"]
    )
    if grouped.empty:
        return np.nan, np.nan
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(grouped), size=(replicates, len(grouped)))
    means = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _consolidate(
    *,
    output_dir: Path,
    folds: list[PurgedFold],
    frozen_entries: pd.DataFrame,
    protocol: dict[str, object],
) -> None:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    scores = pd.concat(
        [pd.read_parquet(_artifact_paths(output_dir, fold.fold_id)["scores"])
         for fold in folds],
        ignore_index=True,
    )
    early = pd.concat(
        [pd.read_parquet(_artifact_paths(output_dir, fold.fold_id)["trades"])
         for fold in folds],
        ignore_index=True,
    )
    actions = pd.concat(
        [pd.read_parquet(_artifact_paths(output_dir, fold.fold_id)["actions"])
         for fold in folds],
        ignore_index=True,
    )
    frontiers = pd.concat(
        [pd.read_csv(_artifact_paths(output_dir, fold.fold_id)["frontier"])
         for fold in folds],
        ignore_index=True,
    )
    audits = pd.DataFrame(
        [
            json.loads(
                _artifact_paths(output_dir, fold.fold_id)["audit"].read_text(
                    encoding="utf-8"
                )
            )
            for fold in folds
        ]
    )
    if set(early["decision_id"]) != set(frozen_entries["decision_id"]):
        raise AssertionError("full exit replay changed frozen entry IDs")
    if early["decision_id"].duplicated().any() or len(early) != len(frozen_entries):
        raise AssertionError("full exit replay changed frozen entry count")
    if _ledger_hash(early["decision_id"]) != protocol["entry_filled_ledger_hash"]:
        raise AssertionError("full exit replay changed frozen entry hash")

    scores.to_parquet(output_dir / "oof_exit_scores.parquet", index=False)
    early.to_parquet(output_dir / "oof_early_exit_trades.parquet", index=False)
    actions.to_parquet(output_dir / "oof_exit_actions.parquet", index=False)
    frontiers.to_csv(output_dir / "threshold_frontier.csv", index=False)
    audits.to_csv(output_dir / "fold_audit.csv", index=False)
    frozen_entries.to_parquet(output_dir / "oof_baseline_trades.parquet", index=False)

    days = int(sum((fold.valid_end - fold.valid_start).days for fold in folds))
    summary_rows: list[dict[str, object]] = []
    for arm, ledger in (("baseline", frozen_entries), ("early_exit", early)):
        for label, capacity in CAPACITIES:
            replay = replay_entry_capacity(
                ledger, capacity=capacity, evaluation_days=days
            )
            replay.orders.to_parquet(
                output_dir / f"orders_{arm}_{label}.parquet", index=False
            )
            summary_rows.append(
                {"arm": arm, "capacity": label, **replay.metrics}
            )
    summary = pd.DataFrame(summary_rows)
    unlimited = summary[summary["capacity"].eq("unlimited")].set_index("arm")
    if unlimited.loc["baseline", "filled_trades"] != unlimited.loc[
        "early_exit", "filled_trades"
    ]:
        raise AssertionError("unlimited early exit changed entry count")
    summary.to_csv(output_dir / "matched_capacity_summary.csv", index=False)

    target = scores["continue_positive"].to_numpy(dtype=np.int8)
    model_summary = {
        "model": protocol["entry_model"],
        "oof_state_rows": int(len(scores)),
        "continue_positive_pct": float(100.0 * target.mean()),
        "roc_auc": float(roc_auc_score(target, scores["score"]))
        if len(np.unique(target)) == 2
        else np.nan,
        "brier": float(brier_score_loss(target, scores["score"])),
        "hold_actions": int(actions["action"].eq("HOLD").sum()),
        "exit_now_actions": int(actions["action"].eq("EXIT NOW").sum()),
        "missing_score_actions": int((~actions["score_available"]).sum()),
        "model_exit_trades": int(early["outcome"].eq("model_exit").sum()),
    }
    pd.DataFrame([model_summary]).to_csv(output_dir / "exit_model_summary.csv", index=False)
    early["delta_r"] = early["r_net"] - early["baseline_r_net"]
    delta_low, delta_high = _paired_episode_bootstrap(early)
    baseline_row = unlimited.loc["baseline"]
    early_row = unlimited.loc["early_exit"]
    economic_success = bool(
        early_row["mean_net_r"] >= 0.05
        and early_row["bootstrap_low"] > 0.0
        and early_row["mean_net_r"] > baseline_row["mean_net_r"]
        and delta_low > 0.0
    )
    result = {
        "protocol_hash": protocol["protocol_hash"],
        "entry_protocol_hash": protocol["entry_protocol_hash"],
        "entry_policy_changed": False,
        "entry_model": protocol["entry_model"],
        "status": "promoted" if economic_success else "exploratory",
        "economic_success": economic_success,
        "baseline_filled_trades": int(baseline_row["filled_trades"]),
        "early_exit_filled_trades": int(early_row["filled_trades"]),
        "baseline_mean_net_r": float(baseline_row["mean_net_r"]),
        "early_exit_mean_net_r": float(early_row["mean_net_r"]),
        "mean_delta_r": float(early["delta_r"].mean()),
        "paired_delta_bootstrap_low": delta_low,
        "paired_delta_bootstrap_high": delta_high,
        "entry_ledger_hash": _ledger_hash(early["decision_id"]),
        "forward_or_lockbox_loaded": False,
    }
    (output_dir / "exit_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run(
    *,
    limit_folds: int | None = None,
    rebuild: bool = False,
    output_dir: Path = OUT,
) -> None:
    if limit_folds is not None and limit_folds < 1:
        raise ValueError("limit_folds must be positive")
    entry_protocol, entry_freeze, _ = load_entry_freeze()
    output_dir = Path(output_dir)
    protocol = write_exit_protocol(
        output_dir, entry_protocol, entry_freeze, overwrite=rebuild
    )
    model_name = str(protocol["entry_model"])
    _write_state(
        output_dir,
        status="running",
        active="dataset",
        completed=[],
        protocol_hash=protocol["protocol_hash"],
        consolidated=False,
    )
    states, proxies, frozen, minute, manifest = prepare_exit_dataset(
        protocol=protocol,
        entry_freeze=entry_freeze,
        output_dir=output_dir,
    )
    states = states.sort_values(["decision_time", "state_id"], kind="stable").reset_index(
        drop=True
    )
    folds = expanding_purged_folds(states)
    selected_folds = folds[:limit_folds] if limit_folds is not None else folds
    (output_dir / "folds").mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    for fold in selected_folds:
        paths = _artifact_paths(output_dir, fold.fold_id)
        if _artifacts_exist(paths) and not rebuild:
            completed.append(fold.fold_id)
            print(f"resume {fold.fold_id}", flush=True)
            continue
        _write_state(
            output_dir,
            status="running",
            active=fold.fold_id,
            completed=completed,
            protocol_hash=protocol["protocol_hash"],
            consolidated=False,
        )
        print(f"fit {fold.fold_id}:{model_name}", flush=True)
        scores, trades, actions, frontier, audit = score_outer_exit_fold(
            states,
            proxies,
            frozen,
            minute,
            fold,
            model_name=model_name,
        )
        scores.to_parquet(paths["scores"], index=False)
        trades.to_parquet(paths["trades"], index=False)
        actions.to_parquet(paths["actions"], index=False)
        frontier.to_csv(paths["frontier"], index=False)
        paths["audit"].write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        completed.append(fold.fold_id)

    full_matrix = (
        limit_folds is None
        and len(folds) == 7
        and all(_artifacts_exist(_artifact_paths(output_dir, fold.fold_id)) for fold in folds)
    )
    if full_matrix:
        _consolidate(
            output_dir=output_dir,
            folds=folds,
            frozen_entries=frozen,
            protocol=protocol,
        )
    _write_state(
        output_dir,
        status="complete",
        active=None,
        completed=completed,
        expected=len(selected_folds),
        protocol_hash=protocol["protocol_hash"],
        consolidated=full_matrix,
        max_loaded_timestamp=manifest["max_loaded_timestamp"],
        entry_policy_changed=False,
        forward_or_lockbox_loaded=False,
    )
    print(
        f"complete: {len(completed)}/{len(selected_folds)} exit folds; "
        f"consolidated={full_matrix}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-folds", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    run(limit_folds=args.limit_folds, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
