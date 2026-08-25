"""Development-only Notebook P conditional opportunity study."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from evaluation.event_window_opportunity_policy import (
    causal_crossing_alerts,
    collapse_episode_time,
)
from experiments.event_window_conditional_oof import (
    ConditionalOOFConfig,
    ConditionalOOFResult,
    _decisions,
    run_conditional_fold,
)
from experiments.event_window_cost_aware_oof import _outer_folds, _partitions
from experiments.event_window_large_move_dataset import build_opportunity_dataset
from experiments.event_window_large_move_models import fit_predict_opportunity
from experiments.event_window_large_move_oof import _fit_binary_platt
from experiments.event_window_magnitude_dataset import align_magnitude_dataset
from experiments.event_window_opportunity_oof import (
    OpportunityOOFConfig,
    _opportunity_decisions,
    _sigmoid,
)
from experiments.event_window_tail_oof import _half_open_uniqueness
from experiments.run_event_window_cost_aware_entry import _Store, _sha256, _sha_payload
from experiments.run_event_window_magnitude_timing import (
    READER_ARTIFACTS as O_READER_ARTIFACTS,
    RUN_ROOT as FROZEN_O_ROOT,
    load_frozen_n_artifacts,
)
from experiments.run_event_window_opportunity_head import (
    _align_labels,
    _source_hash as _n_source_hash,
)
from experiments.run_event_window_tail_models import (
    FROZEN_J_ROOT,
    _build_tail_dataset,
    load_frozen_j_artifacts,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = CODE_ROOT / "data"
RUN_ROOT = CODE_ROOT / "experiments" / "cache" / "event_window_conditional_opportunity"
MODELS = ("logreg", "xgboost")
READER_ARTIFACTS = (
    "oof_predictions.parquet",
    "policy_calibration_predictions.parquet",
    "fold_audit.csv",
    "head_calibration_audit.csv",
    "n3_reconstruction_audit.csv",
    "predictive_metrics.csv",
    "conditional_timing_metrics.csv",
    "fold_improvements.csv",
    "paired_bootstrap.csv",
    "policy_metrics.csv",
    "promotion_table.csv",
    "leakage_audit.csv",
    "protocol.json",
    "frozen_protocol.json",
    "summary.json",
)


@dataclass(frozen=True)
class FrozenOArtifacts:
    run_hash: str
    protocol_hash: str
    source_hash: str
    input_hash: str
    labels_sha256: str
    run_dir: Path
    labels: pd.DataFrame
    protocol: dict[str, object]
    summary: dict[str, object]
    frozen: dict[str, object]


@dataclass(frozen=True)
class ConditionalStudyConfig:
    development_start: str = "2021-01-01"
    development_end_exclusive: str = "2025-07-01"
    primary_activation_target_per_day: float = 2.0
    activation_rate_sensitivities_per_day: tuple[float, ...] = (1.0, 3.0)
    cooldown_minutes: int = 60
    threshold_grid_size: int = 31
    bootstrap_draws: int = 500
    bootstrap_seed: int = 42
    oof: ConditionalOOFConfig = field(default_factory=ConditionalOOFConfig)


@dataclass(frozen=True)
class ConditionalRunResult:
    run_dir: Path
    summary: dict[str, object]


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path.name}")
    return value


def load_frozen_o_artifacts(
    run_root: Path = FROZEN_O_ROOT,
    *,
    expected_run_hash: str | None = None,
) -> FrozenOArtifacts:
    """Validate the complete published development-only Notebook O handoff."""
    root = Path(run_root)
    pointer = None if expected_run_hash is not None else _read_json(root / "latest_dev.json")
    run_hash = str(expected_run_hash or pointer.get("run_hash", ""))
    expected_relative = f"{run_hash}/full"
    if not run_hash or (
        pointer is not None and pointer.get("relative_path") != expected_relative
    ):
        raise ValueError("frozen Notebook O pointer is invalid")
    run_dir = (root / expected_relative).resolve()
    if not run_dir.is_relative_to(root.resolve()):
        raise ValueError("frozen Notebook O path escaped its root")
    state = _read_json(run_dir / "run_state.json")
    protocol_hash = str(state.get("protocol_hash", ""))
    if (
        state.get("status") != "complete"
        or state.get("run_hash") != run_hash
        or not protocol_hash
        or (
            pointer is not None
            and protocol_hash != str(pointer.get("protocol_hash", ""))
        )
    ):
        raise ValueError("frozen Notebook O run is incomplete or changed")
    records = state.get("artifacts")
    if not isinstance(records, dict):
        raise ValueError("frozen Notebook O artifact registry is missing")
    for name in O_READER_ARTIFACTS:
        path = run_dir / name
        record = records.get(name)
        if (
            not path.is_file()
            or not isinstance(record, dict)
            or int(record.get("size", -1)) != path.stat().st_size
            or str(record.get("sha256", "")) != _sha256(path)
        ):
            raise ValueError(f"frozen Notebook O artifact changed: {name}")
    protocol = _read_json(run_dir / "protocol.json")
    summary = _read_json(run_dir / "summary.json")
    frozen = _read_json(run_dir / "frozen_protocol.json")
    for field in ("run_hash", "protocol_hash", "source_hash", "input_hash"):
        if protocol.get(field) != state.get(field) or summary.get(field) != state.get(field):
            raise ValueError(f"frozen Notebook O {field} identity changed")
    if state.get("summary") != summary:
        raise ValueError("frozen Notebook O state summary changed")
    if (
        protocol.get("stage") != "dev"
        or bool(protocol.get("smoke", False))
        or summary.get("forward_or_lockbox_loaded") is not False
    ):
        raise ValueError("Notebook P accepts only the bounded full Notebook O run")
    labels_path = run_dir / "magnitude_labels.parquet"
    if frozen.get("magnitude_labels_sha256") != _sha256(labels_path):
        raise ValueError("frozen Notebook O labels changed")
    labels = pd.read_parquet(labels_path)
    if labels.duplicated(["window_id", "step"]).any():
        raise ValueError("frozen Notebook O labels contain duplicate keys")
    return FrozenOArtifacts(
        run_hash=run_hash,
        protocol_hash=protocol_hash,
        source_hash=str(protocol["source_hash"]),
        input_hash=str(protocol["input_hash"]),
        labels_sha256=_sha256(labels_path),
        run_dir=run_dir,
        labels=labels,
        protocol=protocol,
        summary=summary,
        frozen=frozen,
    )


def protocol_dict(
    config: ConditionalStudyConfig = ConditionalStudyConfig(),
    *,
    smoke: bool = False,
) -> dict[str, object]:
    return {
        "notebook": "P_event_window_conditional_opportunity",
        "stage": "dev",
        "smoke": smoke,
        "development_start": config.development_start,
        "development_end_exclusive": config.development_end_exclusive,
        "frozen_incidence_head": "N3_side_neutral_volatility",
        "timing_intervals_minutes": [[1, 15], [16, 30], [31, 60], [61, 120]],
        "diagnostic_five_minute_timing_only": True,
        "severity_thresholds_b": [1.0, 1.5, 2.0],
        "conditional_models": list(MODELS),
        "inner_partitions": {"fit": 0.70, "probability_calibration": 0.15, "policy_calibration": 0.15},
        "weights": "fixed half-open [t,t+120m) uniqueness for all new heads",
        "primary_activation_target_per_day": config.primary_activation_target_per_day,
        "activation_rate_sensitivities_per_day": list(config.activation_rate_sensitivities_per_day),
        "cooldown_minutes": config.cooldown_minutes,
        "alert_rule": "past-calibrated threshold; new upward crossing or new episode; forward-only cooldown",
        "truth_event_primary_gap_minutes": 5,
        "truth_event_sensitivity_gap_minutes": 60,
        "direction_head_trained": False,
        "trading_policy_trained": False,
        "economics_evaluated": False,
        "forward_or_lockbox_loaded": False,
        "oof_config": asdict(config.oof),
    }


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        CODE_ROOT / "experiments" / "event_window_conditional_oof.py",
        CODE_ROOT / "evaluation" / "event_window_opportunity_policy.py",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _latest(root: Path, run_hash: str, protocol_hash: str) -> None:
    payload = {
        "run_hash": run_hash,
        "protocol_hash": protocol_hash,
        "relative_path": f"{run_hash}/full",
    }
    path = root / "latest_dev.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _validated_completed_summary(
    run_dir: Path,
    identity: dict[str, str],
    frozen_o: FrozenOArtifacts,
) -> dict[str, object] | None:
    try:
        state = _read_json(run_dir / "run_state.json")
        if state.get("status") != "complete" or any(
            state.get(name) != value for name, value in identity.items()
        ):
            return None
        records = state.get("artifacts")
        if not isinstance(records, dict):
            return None
        for name in READER_ARTIFACTS:
            path = run_dir / name
            record = records.get(name)
            if (
                not path.is_file()
                or not isinstance(record, dict)
                or int(record.get("size", -1)) != path.stat().st_size
                or str(record.get("sha256", "")) != _sha256(path)
            ):
                return None
        frozen = _read_json(run_dir / "frozen_protocol.json")
        summary = _read_json(run_dir / "summary.json")
        if frozen.get("frozen_o_run_hash") != frozen_o.run_hash:
            return None
        if frozen.get("frozen_o_labels_sha256") != frozen_o.labels_sha256:
            return None
        if state.get("summary") != summary:
            return None
        return summary
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _frozen_outer_probability(frozen_n, decisions: pd.DataFrame, positions: np.ndarray, fold_id: str) -> np.ndarray:
    keys = ["window_id", "step"]
    source = frozen_n.n3_scores.set_index(keys)
    wanted = pd.MultiIndex.from_frame(decisions.iloc[positions][keys])
    selected = source.reindex(wanted)
    if selected["p_hit"].isna().any() or not selected["fold_id"].eq(fold_id).all():
        raise ValueError(f"frozen N3 does not cover conditional outer fold {fold_id}")
    return selected["p_hit"].to_numpy(float)


def _reconstruct_n3_fold(
    fold,
    fixed_decisions: pd.DataFrame,
    n3_dataset,
    frozen_n,
    config: ConditionalOOFConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    _, _, reserved = _partitions(fixed_decisions, fold.train, config.fold)
    outer = np.asarray(fold.valid, dtype=np.int64)
    outer = outer[fixed_decisions.iloc[outer]["model_target_valid"].to_numpy(bool)]
    requested = np.concatenate([reserved, outer])
    n3_decisions = _opportunity_decisions(n3_dataset)
    matching = [candidate for candidate in _outer_folds(n3_decisions) if candidate.fold_id == fold.fold_id]
    if len(matching) != 1:
        raise ValueError(f"cannot reconstruct frozen N3 fold {fold.fold_id}")
    n3_fit, n3_early, _ = _partitions(n3_decisions, matching[0].train, config.fold)
    fit_weights = _half_open_uniqueness(n3_decisions, n3_fit)
    raw = fit_predict_opportunity(
        "xgboost",
        train_x=n3_dataset.tabular[n3_fit],
        labels=n3_decisions.iloc[n3_fit]["opportunity_code"].to_numpy(int),
        sample_weight=fit_weights,
        score_x=n3_dataset.tabular[np.concatenate([n3_early, requested])],
        config=OpportunityOOFConfig().model,
    )
    if raw.opportunity_logit is None:
        raise AssertionError("N3 reconstruction did not expose a binary logit")
    early_weights = _half_open_uniqueness(n3_decisions, n3_early)
    slope, intercept, fallback = _fit_binary_platt(
        raw.opportunity_logit[: len(n3_early)],
        n3_decisions.iloc[n3_early]["opportunity_code"].to_numpy(int),
        early_weights,
    )
    reconstructed = _sigmoid(raw.opportunity_logit[len(n3_early) :], slope, intercept)
    frozen_outer = _frozen_outer_probability(frozen_n, fixed_decisions, outer, fold.fold_id)
    max_abs = float(np.max(np.abs(reconstructed[len(reserved) :] - frozen_outer)))
    if not np.allclose(reconstructed[len(reserved) :], frozen_outer, rtol=0.0, atol=1e-12):
        raise AssertionError(f"reconstructed N3 fold {fold.fold_id} differs from frozen outer scores")
    p_hit = np.full(len(fixed_decisions), np.nan, dtype=float)
    p_hit[reserved] = reconstructed[: len(reserved)]
    p_hit[outer] = frozen_outer
    return p_hit, {
        "fold": fold.fold_id,
        "reconstruction_skipped_smoke": False,
        "fit_rows": len(n3_fit),
        "early_rows": len(n3_early),
        "reserved_rows_scored": len(reserved),
        "outer_rows_compared": len(outer),
        "platt_identity_fallback": fallback,
        "outer_max_abs_difference": max_abs,
    }


def _smoke_n3_probabilities(
    fold,
    fixed_decisions: pd.DataFrame,
    n3_dataset,
    frozen_n,
    config: ConditionalOOFConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    fit, early, reserved = _partitions(fixed_decisions, fold.train, config.fold)
    outer = np.asarray(fold.valid, dtype=np.int64)
    outer = outer[fixed_decisions.iloc[outer]["model_target_valid"].to_numpy(bool)]
    n3_decisions = _opportunity_decisions(n3_dataset)
    past = np.concatenate([fit, early])
    valid_past = past[n3_decisions.iloc[past]["model_target_valid"].to_numpy(bool)]
    weights = _half_open_uniqueness(n3_decisions, valid_past)
    prevalence = float(
        np.average(n3_decisions.iloc[valid_past]["opportunity_code"].to_numpy(int), weights=weights)
    )
    p_hit = np.full(len(fixed_decisions), np.nan, dtype=float)
    p_hit[reserved] = np.clip(prevalence, 1e-6, 1.0 - 1e-6)
    p_hit[outer] = _frozen_outer_probability(frozen_n, fixed_decisions, outer, fold.fold_id)
    return p_hit, {
        "fold": fold.fold_id,
        "reconstruction_skipped_smoke": True,
        "fit_rows": len(valid_past),
        "early_rows": 0,
        "reserved_rows_scored": len(reserved),
        "outer_rows_compared": len(outer),
        "platt_identity_fallback": True,
        "outer_max_abs_difference": 0.0,
    }


def _binary_metrics(target: np.ndarray, probability: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    y = np.asarray(target, dtype=np.int8)
    p = np.clip(np.asarray(probability, dtype=float), 1e-8, 1.0 - 1e-8)
    w = np.asarray(weights, dtype=float)
    prevalence = float(np.average(y, weights=w))
    threshold = float(np.quantile(p, 0.9))
    top = p >= threshold
    top_rate = float(np.average(y[top], weights=w[top]))
    return {
        "rows": len(y),
        "prevalence": prevalence,
        "brier": float(np.average(np.square(p - y), weights=w)),
        "logloss": float(log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1], sample_weight=w)),
        "pr_auc": float(average_precision_score(y, p, sample_weight=w)) if np.unique(y).size == 2 else np.nan,
        "roc_auc": float(roc_auc_score(y, p, sample_weight=w)) if np.unique(y).size == 2 else np.nan,
        "top_decile_rate": top_rate,
        "top_decile_lift": top_rate / prevalence if prevalence > 0.0 else np.nan,
    }


def _conditional_timing_metrics(frame: pd.DataFrame, *, prefix: str) -> dict[str, float]:
    hit = frame["y_hit"].astype(bool).to_numpy()
    y = frame.loc[hit, "timing_interval"].to_numpy(int)
    stem = "q0_t" if prefix == "q0_" else "q_t"
    q = frame.loc[hit, [f"{stem}_{name}" for name in ("15", "30", "60", "120")]].to_numpy(float)
    w = frame.loc[hit, "sample_weight"].to_numpy(float)
    predicted_cdf = np.cumsum(q, axis=1)[:, :-1]
    observed_cdf = y[:, None] <= np.arange(3)[None, :]
    rps = np.square(predicted_cdf - observed_cdf).mean(axis=1)
    return {
        "rows": len(y),
        "logloss": float(log_loss(y, q, labels=[0, 1, 2, 3], sample_weight=w)),
        "ranked_probability_score": float(np.average(rps, weights=w)),
    }


def _metric_tables(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictive = []
    timing = []
    fold_rows = []
    targets = {
        "t_le_15": "y_t_le_15",
        "t_le_30": "y_t_le_30",
        "t_le_60": "y_t_le_60",
        "t_le_120": "y_hit",
        "ge_150": "y_ge_150",
        "ge_200": "y_ge_200",
    }
    for model, group in scores.groupby("model", sort=False):
        for arm, prefix in (("anchored_empirical", "p0_"), ("conditional", "p_")):
            for target_name, target_column in targets.items():
                probability_column = f"{prefix}{target_name}"
                predictive.append(
                    {
                        "model": model,
                        "arm": arm,
                        "target": target_name,
                        **_binary_metrics(
                            group[target_column].to_numpy(int),
                            group[probability_column].to_numpy(float),
                            group["sample_weight"].to_numpy(float),
                        ),
                    }
                )
            timing.append(
                {"model": model, "arm": arm, **_conditional_timing_metrics(group, prefix="q0_" if arm == "anchored_empirical" else "")}
            )
        for fold_id, fold_group in group.groupby("fold_id", sort=False):
            hit = fold_group["y_hit"].astype(bool).to_numpy()
            y = fold_group.loc[hit, "timing_interval"].to_numpy(int)
            w = fold_group.loc[hit, "sample_weight"].to_numpy(float)
            losses = {}
            for arm, prefix in (("baseline", "q0_"), ("candidate", "")):
                stem = "q0_t" if prefix == "q0_" else "q_t"
                q = fold_group.loc[hit, [f"{stem}_{name}" for name in ("15", "30", "60", "120")]].to_numpy(float)
                cdf = np.cumsum(q, axis=1)[:, :-1]
                observed = y[:, None] <= np.arange(3)[None, :]
                losses[arm] = float(np.average(np.square(cdf - observed).mean(axis=1), weights=w))
            fold_rows.append(
                {"model": model, "fold": fold_id, "comparison": "conditional_timing_rps", "point_improvement": losses["baseline"] - losses["candidate"]}
            )
            for comparison, target, candidate, baseline in (
                ("t60_brier", "y_t_le_60", "p_t_le_60", "p0_t_le_60"),
                ("ge150_brier", "y_ge_150", "p_ge_150", "p0_ge_150"),
            ):
                target_value = fold_group[target].to_numpy(float)
                weight = fold_group["sample_weight"].to_numpy(float)
                improvement = np.square(fold_group[baseline] - target_value) - np.square(fold_group[candidate] - target_value)
                fold_rows.append(
                    {"model": model, "fold": fold_id, "comparison": comparison, "point_improvement": float(np.average(improvement, weights=weight))}
                )
    return pd.DataFrame(predictive), pd.DataFrame(timing), pd.DataFrame(fold_rows)


def _paired_bootstrap(scores: pd.DataFrame, *, draws: int, seed: int) -> pd.DataFrame:
    rows = []
    for model_number, (model, group) in enumerate(scores.groupby("model", sort=False)):
        definitions = []
        hit = group["y_hit"].astype(bool).to_numpy()
        q = group[[f"q_t_{name}" for name in ("15", "30", "60", "120")]].to_numpy(float)
        q0 = group[[f"q0_t_{name}" for name in ("15", "30", "60", "120")]].to_numpy(float)
        y_interval = group["timing_interval"].to_numpy(int)
        observed = y_interval[hit, None] <= np.arange(3)[None, :]
        model_rps = np.square(np.cumsum(q[hit], axis=1)[:, :-1] - observed).mean(axis=1)
        base_rps = np.square(np.cumsum(q0[hit], axis=1)[:, :-1] - observed).mean(axis=1)
        definitions.append(("conditional_timing_rps", hit, base_rps - model_rps))
        for comparison, target_column, candidate, baseline in (
            ("t60_brier", "y_t_le_60", "p_t_le_60", "p0_t_le_60"),
            ("ge150_brier", "y_ge_150", "p_ge_150", "p0_ge_150"),
            ("ge200_brier", "y_ge_200", "p_ge_200", "p0_ge_200"),
        ):
            target = group[target_column].to_numpy(float)
            delta = np.square(group[baseline].to_numpy(float) - target) - np.square(group[candidate].to_numpy(float) - target)
            definitions.append((comparison, np.ones(len(group), dtype=bool), delta))
        for comparison, mask, delta in definitions:
            selected = group.loc[mask, ["channel_episode_id", "sample_weight"]].copy()
            selected["weighted_delta"] = selected["sample_weight"].to_numpy(float) * delta
            by_episode = selected.groupby("channel_episode_id", sort=False).agg(
                numerator=("weighted_delta", "sum"), denominator=("sample_weight", "sum")
            )
            numerator = by_episode["numerator"].to_numpy(float)
            denominator = by_episode["denominator"].to_numpy(float)
            point = float(numerator.sum() / denominator.sum())
            rng = np.random.default_rng(seed + model_number * 100 + len(rows))
            values = np.empty(draws, dtype=float)
            for draw in range(draws):
                sampled = rng.integers(0, len(by_episode), len(by_episode))
                values[draw] = numerator[sampled].sum() / denominator[sampled].sum()
            low, high = np.quantile(values, [0.025, 0.975])
            rows.append(
                {"model": model, "comparison": comparison, "point_improvement": point, "ci_low": float(low), "ci_high": float(high), "bootstrap_draws": draws}
            )
    return pd.DataFrame(rows)


def _policy_frame(frame: pd.DataFrame, *, score: str, target: str) -> pd.DataFrame:
    work = frame[["channel_episode_id", "decision_time", score, target, "tth_100_min", "magnitude_ratio"]].copy()
    work["decision_time"] = pd.to_datetime(work["decision_time"], utc=True)
    return (
        work.groupby(["channel_episode_id", "decision_time"], as_index=False, sort=False)
        .agg(score=(score, "max"), target=(target, "max"), tth_100_min=("tth_100_min", "min"), magnitude_ratio=("magnitude_ratio", "max"))
        .sort_values(["decision_time", "channel_episode_id"], kind="stable")
        .reset_index(drop=True)
    )


def _threshold_frontier(frame: pd.DataFrame, *, grid_size: int, cooldown_minutes: int) -> pd.DataFrame:
    work = collapse_episode_time(frame, score_column="score")
    values = work["score"].to_numpy(float)
    candidates = np.unique(np.quantile(values, np.linspace(0.0, 1.0, min(grid_size, len(values)))))
    days = int((work.decision_time.max().normalize() - work.decision_time.min().normalize()).days + 1)
    rows = []
    for threshold in candidates:
        replay = causal_crossing_alerts(work, threshold=float(threshold), cooldown_minutes=cooldown_minutes)
        alerts = int(replay.alert.sum())
        rows.append({"threshold": float(threshold), "calibration_alerts": alerts, "calibration_days": days, "calibration_rate": alerts / days})
    return pd.DataFrame(rows)


def _truth_clusters(frame: pd.DataFrame, *, gap_minutes: int) -> pd.Series:
    selected = frame.loc[frame["target"].astype(bool)].copy()
    if selected.empty:
        return pd.Series(dtype="int64", index=selected.index)
    selected = selected.sort_values(["channel_episode_id", "decision_time"], kind="stable")
    time = pd.to_datetime(selected["decision_time"], utc=True)
    new = selected.channel_episode_id.ne(selected.channel_episode_id.shift()) | time.diff().gt(pd.Timedelta(minutes=gap_minutes))
    clusters = new.cumsum().astype(int)
    clusters.index = selected.index
    return clusters


def _policy_metric(frame: pd.DataFrame, *, threshold: float, cooldown_minutes: int, gap_minutes: int) -> dict[str, float]:
    replay = causal_crossing_alerts(frame, threshold=threshold, cooldown_minutes=cooldown_minutes)
    alerts = replay["alert"].astype(bool)
    days = int((replay.decision_time.max().normalize() - replay.decision_time.min().normalize()).days + 1)
    clusters = _truth_clusters(replay, gap_minutes=gap_minutes)
    replay["truth_cluster"] = pd.NA
    replay.loc[clusters.index, "truth_cluster"] = clusters
    truth_events = int(clusters.nunique())
    recalled = 0
    for _, event in replay.loc[replay.target.astype(bool)].groupby("truth_cluster", sort=False):
        recalled += int(event.alert.any())
    selected = replay.loc[alerts]
    return {
        "outer_calendar_days": days,
        "selected_activations": int(alerts.sum()),
        "actual_activations_per_day": float(alerts.sum() / days),
        "days_with_activation": int(selected.decision_time.dt.normalize().nunique()),
        "day_coverage": float(selected.decision_time.dt.normalize().nunique() / days),
        "activation_precision": float(selected.target.mean()) if len(selected) else np.nan,
        "truth_events": truth_events,
        "recalled_truth_events": recalled,
        "event_recall": recalled / truth_events if truth_events else np.nan,
        "false_activations_per_day": float((~selected.target.astype(bool)).sum() / days) if len(selected) else 0.0,
        "median_tth_minutes": float(selected.loc[selected.target.astype(bool), "tth_100_min"].median()) if len(selected) else np.nan,
        "mean_magnitude_ratio": float(selected.magnitude_ratio.mean()) if len(selected) else np.nan,
    }


def _policy_metrics(
    scores: pd.DataFrame,
    calibration: pd.DataFrame,
    config: ConditionalStudyConfig,
) -> pd.DataFrame:
    rows = []
    rates = (config.primary_activation_target_per_day, *config.activation_rate_sensitivities_per_day)
    comparisons = (
        ("timing_60", "y_t_le_60", "p0_t_le_60", "p_t_le_60"),
        ("severity_150", "y_ge_150", "p0_ge_150", "p_ge_150"),
    )
    for (model, fold_id), outer in scores.groupby(["model", "fold_id"], sort=False):
        reserved = calibration.loc[(calibration.model == model) & (calibration.fold_id == fold_id)]
        for objective, target, baseline, candidate in comparisons:
            for arm, column in (("anchored_empirical", baseline), ("conditional", candidate)):
                calibration_frame = _policy_frame(reserved, score=column, target=target)
                outer_frame = _policy_frame(outer, score=column, target=target)
                frontier = _threshold_frontier(
                    calibration_frame[["channel_episode_id", "decision_time", "score"]],
                    grid_size=config.threshold_grid_size,
                    cooldown_minutes=config.cooldown_minutes,
                )
                for rate in rates:
                    chosen = frontier.assign(distance=(frontier.calibration_rate - rate).abs()).sort_values(
                        ["distance", "threshold"], ascending=[True, False], kind="stable"
                    ).iloc[0]
                    for gap in (5, 60):
                        rows.append(
                            {
                                "model": model,
                                "fold": fold_id,
                                "objective": objective,
                                "arm": arm,
                                "target_activations_per_day": rate,
                                "truth_gap_minutes": gap,
                                "threshold": float(chosen.threshold),
                                "calibration_activations_per_day": float(chosen.calibration_rate),
                                "calibration_rows": len(calibration_frame),
                                "policy_is_causal": True,
                                **_policy_metric(
                                    outer_frame,
                                    threshold=float(chosen.threshold),
                                    cooldown_minutes=config.cooldown_minutes,
                                    gap_minutes=gap,
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def _promotion_table(
    predictive: pd.DataFrame,
    folds: pd.DataFrame,
    bootstrap: pd.DataFrame,
    policy: pd.DataFrame,
    primary_rate: float,
) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        fold_model = folds.loc[folds.model.eq(model)]
        boot = bootstrap.loc[bootstrap.model.eq(model)].set_index("comparison")
        metrics = predictive.loc[predictive.model.eq(model)]
        primary = policy.loc[
            policy.model.eq(model)
            & policy.target_activations_per_day.eq(primary_rate)
            & policy.truth_gap_minutes.eq(5)
        ]
        policy_pivot = primary.pivot_table(
            index="objective", columns="arm", values=["event_recall", "activation_precision", "actual_activations_per_day"]
        )
        timing_folds = int(
            (fold_model.loc[fold_model.comparison.eq("conditional_timing_rps"), "point_improvement"] >= 0.0).sum()
        )
        severity_folds = int(
            (fold_model.loc[fold_model.comparison.eq("ge150_brier"), "point_improvement"] >= 0.0).sum()
        )
        timing_policy = bool(
            "timing_60" in policy_pivot.index
            and policy_pivot.loc["timing_60", ("event_recall", "conditional")] >= policy_pivot.loc["timing_60", ("event_recall", "anchored_empirical")]
            and policy_pivot.loc["timing_60", ("activation_precision", "conditional")] >= policy_pivot.loc["timing_60", ("activation_precision", "anchored_empirical")]
            and policy_pivot.loc["timing_60", ("actual_activations_per_day", "conditional")] >= 1.0
        )
        severity_policy = bool(
            "severity_150" in policy_pivot.index
            and policy_pivot.loc["severity_150", ("activation_precision", "conditional")] >= policy_pivot.loc["severity_150", ("activation_precision", "anchored_empirical")]
            and policy_pivot.loc["severity_150", ("actual_activations_per_day", "conditional")] >= 1.0
        )
        timing_pass = bool(
            boot.at["conditional_timing_rps", "ci_low"] > 0.0
            and timing_folds >= 5
            and timing_policy
        )
        severity_pass = bool(
            boot.at["ge150_brier", "ci_low"] > 0.0
            and severity_folds >= 5
            and severity_policy
        )
        candidate_tail = metrics.loc[(metrics.arm == "conditional") & (metrics.target == "ge_150"), "top_decile_rate"].iloc[0]
        baseline_tail = metrics.loc[(metrics.arm == "anchored_empirical") & (metrics.target == "ge_150"), "top_decile_rate"].iloc[0]
        rows.append(
            {
                "model": model,
                "timing_ci_low": boot.at["conditional_timing_rps", "ci_low"],
                "timing_nonnegative_folds": timing_folds,
                "timing_policy_pass": timing_policy,
                "timing_pass": timing_pass,
                "severity_ci_low": boot.at["ge150_brier", "ci_low"],
                "severity_nonnegative_folds": severity_folds,
                "severity_policy_pass": severity_policy,
                "severity_top_decile_rate": candidate_tail,
                "severity_baseline_top_decile_rate": baseline_tail,
                "severity_pass": severity_pass,
                "promoted": timing_pass and severity_pass and candidate_tail >= baseline_tail,
            }
        )
    return pd.DataFrame(rows)


def run_conditional_study(
    *,
    stage: str = "dev",
    smoke: bool = False,
    data_root: Path = DEFAULT_DATA_ROOT,
    frozen_j_root: Path = FROZEN_J_ROOT,
    frozen_n_root: Path | None = None,
    frozen_o_root: Path = FROZEN_O_ROOT,
    run_root: Path = RUN_ROOT,
    config: ConditionalStudyConfig = ConditionalStudyConfig(),
) -> ConditionalRunResult:
    if stage != "dev":
        raise ValueError("Notebook P permits development only; forward and Q2 are sealed")
    protocol = protocol_dict(config, smoke=smoke)
    protocol_hash = _sha_payload(protocol)
    frozen_o = load_frozen_o_artifacts(Path(frozen_o_root))
    frozen_n = load_frozen_n_artifacts() if frozen_n_root is None else load_frozen_n_artifacts(Path(frozen_n_root))
    if _n_source_hash() != frozen_n.source_hash:
        raise ValueError("current N3 reconstruction source differs from frozen Notebook N")
    frozen_j = load_frozen_j_artifacts(Path(frozen_j_root))
    if frozen_o.frozen.get("frozen_n_run_hash") != frozen_n.run_hash:
        raise ValueError("frozen Notebook O and N identities differ")
    if frozen_n.frozen_j_run_hash != frozen_j.run_hash:
        raise ValueError("frozen Notebook N and J identities differ")
    source_hash = _source_hash()
    input_hash = _sha_payload(
        {
            "frozen_o_run_hash": frozen_o.run_hash,
            "frozen_o_labels_sha256": frozen_o.labels_sha256,
            "frozen_n_run_hash": frozen_n.run_hash,
            "frozen_n_labels_sha256": frozen_n.labels_sha256,
            "frozen_n_oof_sha256": frozen_n.oof_sha256,
            "frozen_j_run_hash": frozen_j.run_hash,
            "frozen_j_manifest_sha256": frozen_j.manifest_sha256,
        }
    )
    run_hash = _sha_payload({"protocol_hash": protocol_hash, "source_hash": source_hash, "input_hash": input_hash})[:20]
    run_dir = Path(run_root) / run_hash / ("smoke" if smoke else "full")
    identity = {"run_hash": run_hash, "protocol_hash": protocol_hash, "source_hash": source_hash, "input_hash": input_hash}
    cached = _validated_completed_summary(run_dir, identity, frozen_o)
    if cached is not None:
        if not smoke:
            _latest(Path(run_root), run_hash, protocol_hash)
        return ConditionalRunResult(run_dir, cached)
    store = _Store(run_dir, identity)
    try:
        store.json("protocol.json", {**protocol, **identity})
        base, _, loaded = _build_tail_dataset(frozen_j, data_root=Path(data_root), smoke=smoke)
        expected_start = pd.Timestamp(config.development_start, tz="UTC")
        expected_end = pd.Timestamp(config.development_end_exclusive, tz="UTC")
        if not smoke and (
            loaded.read_start != expected_start
            or loaded.read_end_exclusive != expected_end
            or loaded.max_loaded_timestamp >= expected_end
        ):
            raise AssertionError("Notebook P bounded development inputs changed")
        n_labels = _align_labels(base, frozen_n.labels)
        n3_dataset = build_opportunity_dataset(base, n_labels, include_volatility=True)
        o_labels = _align_labels(base, frozen_o.labels)
        dataset = align_magnitude_dataset(n3_dataset, o_labels)
        decisions = _decisions(dataset)
        folds = [fold for fold in _outer_folds(decisions) if len(fold.train) and len(fold.valid)]
        if not folds:
            raise RuntimeError("no Notebook P OOF folds are available")
        model_config = replace(config.oof.model, xgb_estimators=12) if smoke else config.oof.model
        oof_config = replace(config.oof, model=model_config)
        results: list[ConditionalOOFResult] = []
        reconstruction_audit = []
        for fold in folds:
            if smoke:
                p_hit, audit = _smoke_n3_probabilities(fold, decisions, n3_dataset, frozen_n, oof_config)
            else:
                p_hit, audit = _reconstruct_n3_fold(fold, decisions, n3_dataset, frozen_n, oof_config)
            reconstruction_audit.append(audit)
            for model in MODELS:
                results.append(
                    run_conditional_fold(model, fold, dataset, p_hit_by_position=p_hit, config=oof_config)
                )
        scores = pd.concat([result.scores for result in results], ignore_index=True)
        calibration_scores = pd.concat([result.calibration_scores for result in results], ignore_index=True)
        fold_audit = pd.concat([result.fold_audit for result in results], ignore_index=True)
        head_audit = pd.concat([result.calibration_audit for result in results], ignore_index=True)
        n3_audit = pd.DataFrame(reconstruction_audit)
        predictive, timing, fold_improvements = _metric_tables(scores)
        draws = 20 if smoke else config.bootstrap_draws
        paired = _paired_bootstrap(scores, draws=draws, seed=config.bootstrap_seed)
        policy = _policy_metrics(scores, calibration_scores, config)
        promotion = _promotion_table(
            predictive, fold_improvements, paired, policy, config.primary_activation_target_per_day
        )
        store.parquet("oof_predictions.parquet", scores)
        store.parquet("policy_calibration_predictions.parquet", calibration_scores)
        store.csv("fold_audit.csv", fold_audit)
        store.csv("head_calibration_audit.csv", head_audit)
        store.csv("n3_reconstruction_audit.csv", n3_audit)
        store.csv("predictive_metrics.csv", predictive)
        store.csv("conditional_timing_metrics.csv", timing)
        store.csv("fold_improvements.csv", fold_improvements)
        store.csv("paired_bootstrap.csv", paired)
        store.csv("policy_metrics.csv", policy)
        store.csv("promotion_table.csv", promotion)
        calibration_before_outer = True
        for result in results:
            calibration_before_outer &= result.calibration_scores.decision_time.max() < result.scores.decision_time.min()
        leakage = pd.DataFrame(
            [
                {"check": "development boundary", "passed": bool(smoke or loaded.max_loaded_timestamp < expected_end), "detail": str(loaded.max_loaded_timestamp)},
                {"check": "forward and Q2 excluded", "passed": True, "detail": "stage rejected before handoff loading"},
                {"check": "frozen O labels", "passed": len(o_labels) == len(dataset.decisions), "detail": frozen_o.labels_sha256},
                {"check": "fixed 120m labels", "passed": decisions.label_end.eq(decisions.decision_time + pd.Timedelta(minutes=120)).all(), "detail": "all rows"},
                {"check": "episode-disjoint OOF", "passed": fold_audit.episode_overlap.eq(0).all(), "detail": "all folds and models"},
                {"check": "purged live labels", "passed": fold_audit.train_label_end_max.le(fold_audit.validation_start).all(), "detail": "all folds and models"},
                {"check": "reserved precedes outer", "passed": calibration_before_outer, "detail": "policy thresholds use past-only rows"},
                {"check": "frozen N3 outer reconstruction", "passed": n3_audit.outer_max_abs_difference.le(1e-12).all(), "detail": "skipped smoke uses literal frozen outer" if smoke else "exact fold replay"},
                {"check": "P120 equals frozen N3", "passed": fold_audit.p120_identity_max_abs.eq(0.0).all(), "detail": "by construction"},
                {"check": "nested timing", "passed": fold_audit.timing_monotonic_violations.eq(0).all(), "detail": "15<=30<=60<=120"},
                {"check": "nested severity", "passed": fold_audit.severity_monotonic_violations.eq(0).all(), "detail": "2B<=1.5B<=1B"},
                {"check": "causal alert policy", "passed": policy.policy_is_causal.all(), "detail": "reserved threshold plus forward-only crossing cooldown"},
            ]
        )
        if not leakage.passed.all():
            failed = leakage.loc[~leakage.passed, "check"].tolist()
            raise AssertionError(f"Notebook P leakage audit failed: {failed}")
        store.csv("leakage_audit.csv", leakage)
        promoted = promotion.loc[promotion.promoted, "model"].tolist()
        decision = (
            f"promote conditional opportunity model: {promoted[0]}" if promoted else "do not promote conditional timing/severity beyond frozen N3"
        )
        store.json(
            "frozen_protocol.json",
            {
                "frozen_o_run_hash": frozen_o.run_hash,
                "frozen_o_protocol_hash": frozen_o.protocol_hash,
                "frozen_o_labels_sha256": frozen_o.labels_sha256,
                "frozen_n_run_hash": frozen_n.run_hash,
                "frozen_n_labels_sha256": frozen_n.labels_sha256,
                "frozen_n_oof_sha256": frozen_n.oof_sha256,
                "frozen_j_run_hash": frozen_j.run_hash,
                "frozen_j_manifest_sha256": frozen_j.manifest_sha256,
                "direction_head_trained": False,
                "economics_evaluated": False,
                "forward_or_lockbox_loaded": False,
            },
        )
        summary = {
            **identity,
            "decision": decision,
            "promoted_models": promoted,
            "decision_rows": len(dataset.decisions),
            "oof_rows_per_model": scores.groupby("model").size().to_dict(),
            "feature_count": len(dataset.tabular_features),
            "folds": len(folds),
            "p120_identity_max_abs": float(fold_audit.p120_identity_max_abs.max()),
            "n3_reconstruction_max_abs": float(n3_audit.outer_max_abs_difference.max()),
            "timing_monotonic_violations": int(fold_audit.timing_monotonic_violations.sum()),
            "severity_monotonic_violations": int(fold_audit.severity_monotonic_violations.sum()),
            "policy_is_causal": bool(policy.policy_is_causal.all()),
            "promotion": promotion.set_index("model").to_dict(orient="index"),
            "max_loaded_timestamp": loaded.max_loaded_timestamp,
            "read_end_exclusive": loaded.read_end_exclusive,
            "direction_head_trained": False,
            "trading_policy_trained": False,
            "economics_evaluated": False,
            "forward_or_lockbox_loaded": False,
        }
        store.json("summary.json", summary)
        missing = [name for name in READER_ARTIFACTS if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"Notebook P artifacts missing: {missing}")
        store.complete(summary)
        if not smoke:
            _latest(Path(run_root), run_hash, protocol_hash)
        return ConditionalRunResult(run_dir, summary)
    except BaseException as error:
        store.fail(error)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="dev")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--frozen-j-root", type=Path, default=FROZEN_J_ROOT)
    parser.add_argument("--frozen-o-root", type=Path, default=FROZEN_O_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_conditional_study(
        stage=args.stage,
        smoke=args.smoke,
        data_root=args.data_root,
        frozen_j_root=args.frozen_j_root,
        frozen_o_root=args.frozen_o_root,
        run_root=args.run_root,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ConditionalRunResult",
    "ConditionalStudyConfig",
    "FROZEN_O_ROOT",
    "READER_ARTIFACTS",
    "RUN_ROOT",
    "load_frozen_o_artifacts",
    "protocol_dict",
    "run_conditional_study",
]
