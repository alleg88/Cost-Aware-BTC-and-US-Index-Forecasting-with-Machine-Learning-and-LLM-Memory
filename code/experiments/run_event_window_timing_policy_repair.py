"""Development-only Notebook R timing-policy repair study."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.event_window_opportunity_policy import (
    causal_crossing_alerts,
    causal_level_rearm_alerts,
    collapse_episode_time,
    select_causal_threshold,
)
from experiments.run_event_window_economic_feasibility import (
    FROZEN_O_ROOT,
    FROZEN_P_ROOT,
    FrozenPArtifacts,
    _attach_execution_fields,
    _concurrency_audit,
    _economic_metrics,
    _frequency_audit,
    build_direction_scenarios,
    load_frozen_p_artifacts,
    replay_brackets,
)
from experiments.run_event_window_conditional_opportunity import load_frozen_o_artifacts
from experiments.run_event_window_cost_aware_entry import _Store, _sha256, _sha_payload
from experiments.run_event_window_economic_feasibility import (
    READER_ARTIFACTS as Q_READER_ARTIFACTS,
)
from experiments.run_event_window_tcn import _load_bounded_parquet


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = CODE_ROOT / "data"
FROZEN_Q_ROOT = CODE_ROOT / "experiments" / "cache" / "event_window_economic_feasibility"
FROZEN_Q_RUN_HASH = "acf3991689b66bcbf0d1"
RUN_ROOT = CODE_ROOT / "experiments" / "cache" / "event_window_timing_policy_repair"
READER_ARTIFACTS = (
    "activation_ledger.parquet",
    "economic_paths.parquet",
    "economic_metrics.csv",
    "policy_comparisons.csv",
    "frequency_audit.csv",
    "threshold_audit.csv",
    "same_threshold_supply.csv",
    "concurrency_audit.csv",
    "leakage_audit.csv",
    "protocol.json",
    "frozen_protocol.json",
    "summary.json",
)


@dataclass(frozen=True)
class TimingPolicyRepairConfig:
    development_start: str = "2021-01-01"
    development_end_exclusive: str = "2025-07-01"
    target_activations_per_day: float = 3.0
    target_multiple_b: float = 2.0
    stop_multiple_b: float = 1.0
    hold_minutes: int = 120
    cooldown_minutes: int = 60
    threshold_grid_size: int = 51
    entry_cost_bps: float = 5.0
    target_exit_cost_bps: float = 2.0
    other_exit_cost_bps: float = 5.0
    bootstrap_draws: int = 2_000
    bootstrap_seed: int = 42
    minimum_path_completeness: float = 0.99
    minimum_frequency_per_day: float = 2.5
    maximum_frequency_per_day: float = 3.5


@dataclass(frozen=True)
class FrozenQHandoff:
    run_hash: str
    protocol_hash: str
    run_dir: Path
    protocol: dict[str, object]
    frozen: dict[str, object]
    summary: dict[str, object]
    state: dict[str, object]


@dataclass(frozen=True)
class TimingPolicyRepairRunResult:
    run_dir: Path
    summary: dict[str, object]


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path.name}")
    return value


def protocol_dict(
    config: TimingPolicyRepairConfig = TimingPolicyRepairConfig(),
    *,
    smoke: bool = False,
) -> dict[str, object]:
    return {
        "notebook": "R_event_window_timing_policy_repair",
        "stage": "dev",
        "smoke": smoke,
        "development_start": config.development_start,
        "development_end_exclusive": config.development_end_exclusive,
        "frozen_p_run_hash": "0474798f6d0eb56e64d3",
        "frozen_q_run_hash": FROZEN_Q_RUN_HASH,
        "target_activations_per_day": config.target_activations_per_day,
        "target_multiple_b": config.target_multiple_b,
        "stop_multiple_b": config.stop_multiple_b,
        "hold_minutes": config.hold_minutes,
        "cooldown_minutes": config.cooldown_minutes,
        "timing_score": "p_t_le_60",
        "policies": ["crossing", "level_rearm"],
        "models": ["xgboost", "logreg", "anchored_empirical"],
        "threshold_matching": "per-fold past-only policy calibration",
        "same_threshold_rearm_is_diagnostic_only": True,
        "entry_cost_bps": config.entry_cost_bps,
        "target_exit_cost_bps": config.target_exit_cost_bps,
        "other_exit_cost_bps": config.other_exit_cost_bps,
        "bootstrap_unit": "channel_episode_id",
        "bootstrap_draws": config.bootstrap_draws,
        "bootstrap_seed": config.bootstrap_seed,
        "new_features_added": False,
        "direction_head_trained": False,
        "timing_model_refit": False,
        "forward_or_lockbox_loaded": False,
    }


def load_frozen_q_handoff(
    run_root: Path = FROZEN_Q_ROOT,
) -> FrozenQHandoff:
    """Validate the exact completed Notebook Q handoff."""
    root = Path(run_root)
    pointer = _read_json(root / "latest_dev.json")
    expected_relative = f"{FROZEN_Q_RUN_HASH}/full"
    if (
        pointer.get("run_hash") != FROZEN_Q_RUN_HASH
        or pointer.get("relative_path") != expected_relative
    ):
        raise ValueError("frozen Notebook Q pointer changed")
    run_dir = (root / expected_relative).resolve()
    if not run_dir.is_relative_to(root.resolve()):
        raise ValueError("frozen Notebook Q path escaped its root")
    state = _read_json(run_dir / "run_state.json")
    if (
        state.get("status") != "complete"
        or state.get("run_hash") != FROZEN_Q_RUN_HASH
        or state.get("protocol_hash") != pointer.get("protocol_hash")
    ):
        raise ValueError("frozen Notebook Q run is incomplete or changed")
    records = state.get("artifacts")
    if not isinstance(records, dict):
        raise ValueError("frozen Notebook Q artifact registry is missing")
    for name in Q_READER_ARTIFACTS:
        path = run_dir / name
        record = records.get(name)
        if (
            not path.is_file()
            or not isinstance(record, dict)
            or int(record.get("size", -1)) != path.stat().st_size
            or str(record.get("sha256", "")) != _sha256(path)
        ):
            raise ValueError(f"frozen Notebook Q artifact changed: {name}")
    protocol = _read_json(run_dir / "protocol.json")
    frozen = _read_json(run_dir / "frozen_protocol.json")
    summary = _read_json(run_dir / "summary.json")
    if (
        state.get("summary") != summary
        or summary.get("forward_or_lockbox_loaded") is not False
        or summary.get("timing_model_refit") is not False
        or frozen.get("frozen_p_run_hash") != "0474798f6d0eb56e64d3"
    ):
        raise ValueError("frozen Notebook Q summary changed")
    return FrozenQHandoff(
        run_hash=FROZEN_Q_RUN_HASH,
        protocol_hash=str(state["protocol_hash"]),
        run_dir=run_dir,
        protocol=protocol,
        frozen=frozen,
        summary=summary,
        state=state,
    )


def calendar_days(frame: pd.DataFrame) -> int:
    time = pd.to_datetime(frame["decision_time"], utc=True, errors="raise")
    if time.empty:
        raise ValueError("calendar-day calculation requires decisions")
    return int((time.max().normalize() - time.min().normalize()).days + 1)


def _frozen_crossing_threshold(
    frozen: FrozenPArtifacts,
    *,
    model: str,
    fold_id: str,
    score_arm: str,
    rate: float,
) -> tuple[float, pd.Series]:
    rows = frozen.policy.loc[
        frozen.policy["model"].eq(model)
        & frozen.policy["fold"].astype(str).eq(str(fold_id))
        & frozen.policy["objective"].eq("timing_60")
        & frozen.policy["arm"].eq(score_arm)
        & frozen.policy["target_activations_per_day"].eq(float(rate))
        & frozen.policy["truth_gap_minutes"].eq(5)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"frozen crossing threshold is not unique: {model}, {fold_id}, {score_arm}"
        )
    row = rows.iloc[0]
    return float(row["threshold"]), row


def _arm_specs() -> tuple[tuple[str, str, str], ...]:
    return (
        ("xgboost_conditional", "xgboost", "p_t_le_60"),
        ("logreg_conditional", "logreg", "p_t_le_60"),
        ("anchored_empirical", "xgboost", "p0_t_le_60"),
    )


def _selected_rows(
    frame: pd.DataFrame,
    *,
    threshold: float,
    score_column: str,
    policy: str,
    cooldown_minutes: int,
) -> pd.DataFrame:
    collapsed = collapse_episode_time(frame, score_column=score_column)
    function = (
        causal_crossing_alerts
        if policy == "crossing"
        else causal_level_rearm_alerts
    )
    replay = function(
        collapsed,
        threshold=threshold,
        score_column=score_column,
        cooldown_minutes=cooldown_minutes,
    )
    return replay.loc[replay["alert"].astype(bool)].copy()


def reconstruct_policy_ledger(
    frozen: FrozenPArtifacts,
    config: TimingPolicyRepairConfig = TimingPolicyRepairConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build crossing and past-only matched-frequency level-rearm activations."""
    rows: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, object]] = []
    supply_rows: list[dict[str, object]] = []
    for base_arm, model, score_column in _arm_specs():
        outer_model = frozen.oof.loc[frozen.oof["model"].eq(model)].copy()
        calibration_model = frozen.calibration.loc[
            frozen.calibration["model"].eq(model)
        ].copy()
        crossing_total = 0
        same_threshold_rearm_total = 0
        for fold_id, outer in outer_model.groupby("fold_id", sort=False):
            outer = outer.copy()
            outer["decision_time"] = pd.to_datetime(
                outer["decision_time"], utc=True, errors="raise"
            )
            calibration = calibration_model.loc[
                calibration_model["fold_id"].astype(str).eq(str(fold_id))
            ].copy()
            calibration["decision_time"] = pd.to_datetime(
                calibration["decision_time"], utc=True, errors="raise"
            )
            if calibration.empty or calibration["decision_time"].max() >= outer["decision_time"].min():
                raise ValueError(f"policy calibration does not precede outer fold {fold_id}")
            score_arm = "conditional" if score_column == "p_t_le_60" else "anchored_empirical"
            frozen_threshold, frozen_row = _frozen_crossing_threshold(
                frozen,
                model=model,
                fold_id=str(fold_id),
                score_arm=score_arm,
                rate=config.target_activations_per_day,
            )
            crossing = _selected_rows(
                outer,
                threshold=frozen_threshold,
                score_column=score_column,
                policy="crossing",
                cooldown_minutes=config.cooldown_minutes,
            )
            same_threshold_rearm = _selected_rows(
                outer,
                threshold=frozen_threshold,
                score_column=score_column,
                policy="level_rearm",
                cooldown_minutes=config.cooldown_minutes,
            )
            crossing_total += len(crossing)
            same_threshold_rearm_total += len(same_threshold_rearm)
            selected_threshold = select_causal_threshold(
                calibration,
                target_activations_per_day=config.target_activations_per_day,
                score_column=score_column,
                cooldown_minutes=config.cooldown_minutes,
                grid_size=config.threshold_grid_size,
                alert_policy="level_rearm",
            )
            level_rearm = _selected_rows(
                outer,
                threshold=selected_threshold.threshold,
                score_column=score_column,
                policy="level_rearm",
                cooldown_minutes=config.cooldown_minutes,
            )
            threshold_rows.extend(
                (
                    {
                        "arm": base_arm,
                        "model": model,
                        "fold_id": str(fold_id),
                        "policy": "crossing",
                        "threshold": frozen_threshold,
                        "threshold_source": "frozen_notebook_p",
                        "calibration_rows": int(frozen_row["calibration_rows"]),
                        "calibration_activations_per_day": float(
                            frozen_row["calibration_activations_per_day"]
                        ),
                        "outer_activations": len(crossing),
                        "calibration_precedes_outer": True,
                    },
                    {
                        "arm": base_arm,
                        "model": model,
                        "fold_id": str(fold_id),
                        "policy": "level_rearm",
                        "threshold": selected_threshold.threshold,
                        "threshold_source": "past_only_policy_calibration",
                        "calibration_rows": selected_threshold.calibration_rows,
                        "calibration_activations_per_day": (
                            selected_threshold.actual_activations_per_day
                        ),
                        "outer_activations": len(level_rearm),
                        "calibration_precedes_outer": True,
                    },
                )
            )
            for policy, selected, threshold in (
                ("crossing", crossing, frozen_threshold),
                ("level_rearm", level_rearm, selected_threshold.threshold),
            ):
                selected = selected.copy()
                selected["arm"] = f"{base_arm}_{policy}"
                selected["base_arm"] = base_arm
                selected["model"] = model
                selected["score_arm"] = score_arm
                selected["policy"] = policy
                selected["target_activations_per_day"] = config.target_activations_per_day
                selected["threshold"] = threshold
                selected["activation_score"] = selected[score_column].astype(float)
                selected["activation_key"] = (
                    selected["arm"]
                    + "|"
                    + selected["fold_id"].astype(str)
                    + "|"
                    + selected["channel_episode_id"].astype(str)
                    + "|"
                    + selected["decision_time"].astype(str)
                )
                rows.append(
                    selected[
                        [
                            "activation_key",
                            "arm",
                            "base_arm",
                            "model",
                            "score_arm",
                            "policy",
                            "target_activations_per_day",
                            "fold_id",
                            "window_id",
                            "channel_episode_id",
                            "step",
                            "decision_time",
                            "threshold",
                            "activation_score",
                        ]
                    ]
                )
        outer_days = calendar_days(outer_model)
        supply_rows.append(
            {
                "arm": base_arm,
                "model": model,
                "crossing_activations": crossing_total,
                "level_rearm_same_threshold_activations": same_threshold_rearm_total,
                "calendar_days": outer_days,
                "crossing_per_day": crossing_total / outer_days,
                "same_threshold_rearm_per_day": same_threshold_rearm_total / outer_days,
                "same_threshold_rearm_ratio": (
                    same_threshold_rearm_total / crossing_total
                ),
                "diagnostic_only": True,
            }
        )
    ledger = pd.concat(rows, ignore_index=True).sort_values(
        ["arm", "decision_time", "channel_episode_id"], kind="stable"
    ).reset_index(drop=True)
    if ledger.duplicated(["arm", "channel_episode_id", "decision_time"]).any():
        raise AssertionError("policy replay produced duplicate episode-time activations")
    end = pd.Timestamp(config.development_end_exclusive, tz="UTC")
    if pd.to_datetime(ledger["decision_time"], utc=True).max() >= end:
        raise ValueError("Notebook R decisions crossed the development boundary")
    return ledger, pd.DataFrame(threshold_rows), pd.DataFrame(supply_rows)


def paired_policy_bootstrap(
    scenarios: pd.DataFrame,
    *,
    comparisons: tuple[tuple[str, str, str], ...],
    calendar_days: int,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    """Compare arm means and calendar returns while resampling whole episodes."""
    required = {"arm", "channel_episode_id", "scenario", "censored", "net_r"}
    missing = sorted(required.difference(scenarios.columns))
    if missing:
        raise ValueError(f"paired policy scenarios missing columns: {missing}")
    if calendar_days < 1 or draws < 1:
        raise ValueError("calendar_days and draws must be positive")
    observed = scenarios.loc[~scenarios["censored"].astype(bool)].copy()
    rows: list[dict[str, object]] = []
    number = 0
    for scenario, scenario_rows in observed.groupby("scenario", sort=True):
        for candidate_arm, control_arm, comparison in comparisons:
            candidate = scenario_rows.loc[scenario_rows["arm"].eq(candidate_arm)]
            control = scenario_rows.loc[scenario_rows["arm"].eq(control_arm)]
            episodes = pd.Index(
                sorted(
                    set(candidate["channel_episode_id"]).union(
                        control["channel_episode_id"]
                    )
                )
            )
            if episodes.empty:
                raise ValueError(
                    f"paired comparison has no episodes: {candidate_arm}, {control_arm}"
                )

            def episode_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
                grouped = frame.groupby("channel_episode_id", sort=False)["net_r"].agg(
                    ["sum", "count"]
                )
                aligned = grouped.reindex(episodes, fill_value=0.0)
                return aligned["sum"].to_numpy(float), aligned["count"].to_numpy(float)

            candidate_sum, candidate_count = episode_arrays(candidate)
            control_sum, control_count = episode_arrays(control)
            candidate_mean = float(candidate_sum.sum() / candidate_count.sum())
            control_mean = float(control_sum.sum() / control_count.sum())
            delta_mean = candidate_mean - control_mean
            delta_per_day = float(
                (candidate_sum.sum() - control_sum.sum()) / calendar_days
            )
            rng = np.random.default_rng(seed + number)
            sampled = rng.integers(0, len(episodes), size=(draws, len(episodes)))
            sampled_candidate_sum = candidate_sum[sampled].sum(axis=1)
            sampled_candidate_count = candidate_count[sampled].sum(axis=1)
            sampled_control_sum = control_sum[sampled].sum(axis=1)
            sampled_control_count = control_count[sampled].sum(axis=1)
            valid = (sampled_candidate_count > 0) & (sampled_control_count > 0)
            delta_mean_draws = (
                sampled_candidate_sum[valid] / sampled_candidate_count[valid]
                - sampled_control_sum[valid] / sampled_control_count[valid]
            )
            delta_day_draws = (
                sampled_candidate_sum - sampled_control_sum
            ) / calendar_days
            mean_low, mean_high = np.quantile(delta_mean_draws, [0.025, 0.975])
            day_low, day_high = np.quantile(delta_day_draws, [0.025, 0.975])
            rows.append(
                {
                    "comparison": comparison,
                    "candidate_arm": candidate_arm,
                    "control_arm": control_arm,
                    "scenario": scenario,
                    "union_episodes": len(episodes),
                    "candidate_trades": int(candidate_count.sum()),
                    "control_trades": int(control_count.sum()),
                    "candidate_mean_net_r": candidate_mean,
                    "control_mean_net_r": control_mean,
                    "delta_mean_net_r": delta_mean,
                    "delta_mean_net_r_ci_low": float(mean_low),
                    "delta_mean_net_r_ci_high": float(mean_high),
                    "delta_net_r_per_day": delta_per_day,
                    "delta_net_r_per_day_ci_low": float(day_low),
                    "delta_net_r_per_day_ci_high": float(day_high),
                    "bootstrap_draws": draws,
                    "bootstrap_seed": seed + number,
                }
            )
            number += 1
    return pd.DataFrame(rows)


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        CODE_ROOT / "evaluation" / "event_window_opportunity_policy.py",
        CODE_ROOT / "experiments" / "run_event_window_economic_feasibility.py",
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
    path = Path(root) / "latest_dev.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _validated_completed_summary(
    run_dir: Path,
    identity: dict[str, str],
    frozen_q: FrozenQHandoff,
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
        if (
            frozen.get("frozen_q_run_hash") != frozen_q.run_hash
            or state.get("summary") != summary
        ):
            return None
        return summary
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _activation_counts(ledger: pd.DataFrame) -> dict[str, int]:
    return {
        str(arm): int(len(group))
        for arm, group in ledger.groupby("arm", sort=True)
    }


def _matched_frequency_rates(ledger: pd.DataFrame) -> dict[str, float]:
    days = calendar_days(ledger)
    return {
        str(arm): float(len(group) / days)
        for arm, group in ledger.groupby("arm", sort=True)
    }


def run_timing_policy_repair(
    *,
    stage: str = "dev",
    smoke: bool = False,
    data_root: Path = DEFAULT_DATA_ROOT,
    frozen_p_root: Path = FROZEN_P_ROOT,
    frozen_q_root: Path = FROZEN_Q_ROOT,
    run_root: Path = RUN_ROOT,
    config: TimingPolicyRepairConfig = TimingPolicyRepairConfig(),
) -> TimingPolicyRepairRunResult:
    if stage != "dev":
        raise ValueError("Notebook R permits development only; forward and Q2 are sealed")
    protocol = protocol_dict(config, smoke=smoke)
    protocol_hash = _sha_payload(protocol)

    frozen_q = load_frozen_q_handoff(Path(frozen_q_root))
    frozen_p = load_frozen_p_artifacts(Path(frozen_p_root))
    if (
        frozen_q.frozen.get("frozen_p_run_hash") != frozen_p.run_hash
        or frozen_q.frozen.get("frozen_p_oof_sha256") != frozen_p.oof_sha256
        or frozen_q.frozen.get("frozen_p_calibration_sha256")
        != frozen_p.calibration_sha256
    ):
        raise ValueError("frozen Notebook P and Q identities differ")

    full_ledger, threshold_audit, same_threshold_supply = reconstruct_policy_ledger(
        frozen_p, config
    )
    counts = _activation_counts(full_ledger)
    supply = same_threshold_supply.set_index("arm")
    if (
        counts.get("xgboost_conditional_crossing") != 3_780
        or int(supply.at["xgboost_conditional", "level_rearm_same_threshold_activations"])
        != 7_418
    ):
        raise AssertionError("frozen Notebook R supply changed")
    rates = _matched_frequency_rates(full_ledger)
    level_rates = [
        rate for arm, rate in rates.items() if arm.endswith("_level_rearm")
    ]
    if not all(
        config.minimum_frequency_per_day
        <= rate
        <= config.maximum_frequency_per_day
        for rate in level_rates
    ):
        raise AssertionError(f"level re-arm missed the matched-frequency band: {rates}")

    # Policy selection is complete before execution labels or market paths are read.
    frozen_o = load_frozen_o_artifacts(
        FROZEN_O_ROOT,
        expected_run_hash=str(frozen_q.frozen["frozen_o_run_hash"]),
    )
    if frozen_q.frozen.get("frozen_o_run_hash") != frozen_o.run_hash:
        raise ValueError("frozen Notebook Q and O identities differ")
    ledger = _attach_execution_fields(full_ledger, frozen_o.labels)
    economic_ledger = (
        ledger.groupby("arm", sort=True, group_keys=False).head(8).reset_index(drop=True)
        if smoke
        else ledger
    )

    minute_path = Path(data_root) / "btcusdt_1m_2021_2026.parquet"
    minute_sha256 = _sha256(minute_path)
    source_hash = _source_hash()
    input_hash = _sha_payload(
        {
            "frozen_q_run_hash": frozen_q.run_hash,
            "frozen_q_protocol_hash": frozen_q.protocol_hash,
            "frozen_p_run_hash": frozen_p.run_hash,
            "frozen_p_oof_sha256": frozen_p.oof_sha256,
            "frozen_p_calibration_sha256": frozen_p.calibration_sha256,
            "frozen_o_run_hash": frozen_o.run_hash,
            "frozen_o_labels_sha256": frozen_o.labels_sha256,
            "minute_source_sha256": minute_sha256,
        }
    )
    identity = {
        "run_hash": _sha_payload(
            {
                "protocol_hash": protocol_hash,
                "source_hash": source_hash,
                "input_hash": input_hash,
            }
        )[:20],
        "protocol_hash": protocol_hash,
        "source_hash": source_hash,
        "input_hash": input_hash,
    }
    run_dir = Path(run_root) / identity["run_hash"] / ("smoke" if smoke else "full")
    cached = _validated_completed_summary(run_dir, identity, frozen_q)
    if cached is not None:
        if not smoke:
            _latest(Path(run_root), identity["run_hash"], protocol_hash)
        return TimingPolicyRepairRunResult(run_dir, cached)

    store = _Store(run_dir, identity)
    try:
        store.json("protocol.json", {**protocol, **identity})
        store.json(
            "frozen_protocol.json",
            {
                "frozen_q_run_hash": frozen_q.run_hash,
                "frozen_q_protocol_hash": frozen_q.protocol_hash,
                "frozen_p_run_hash": frozen_p.run_hash,
                "frozen_p_protocol_hash": frozen_p.protocol_hash,
                "frozen_p_oof_sha256": frozen_p.oof_sha256,
                "frozen_p_calibration_sha256": frozen_p.calibration_sha256,
                "frozen_o_run_hash": frozen_o.run_hash,
                "frozen_o_labels_sha256": frozen_o.labels_sha256,
                "minute_source_sha256": minute_sha256,
                "timing_model_refit": False,
                "new_features_added": False,
                "forward_or_lockbox_loaded": False,
            },
        )
        start = pd.to_datetime(
            economic_ledger["decision_time"], utc=True, errors="raise"
        ).min()
        development_end = pd.Timestamp(config.development_end_exclusive, tz="UTC")
        requested_end = pd.to_datetime(
            economic_ledger["decision_time"], utc=True, errors="raise"
        ).max() + pd.Timedelta(minutes=config.hold_minutes)
        read_end = min(requested_end, development_end)
        minute = _load_bounded_parquet(minute_path, start=start, end=read_end)
        if not minute.empty and minute.index.max() >= development_end:
            raise AssertionError("Notebook R minute load crossed the development boundary")

        paths = replay_brackets(
            economic_ledger,
            minute,
            target_multiples=(config.target_multiple_b,),
            hold_minutes=(config.hold_minutes,),
            entry_cost_bps=config.entry_cost_bps,
            target_exit_cost_bps=config.target_exit_cost_bps,
            other_exit_cost_bps=config.other_exit_cost_bps,
        )
        scenarios = build_direction_scenarios(paths)
        days = calendar_days(full_ledger)
        draws = 20 if smoke else config.bootstrap_draws
        metrics = _economic_metrics(
            scenarios,
            calendar_days=days,
            draws=draws,
            seed=config.bootstrap_seed,
        )
        comparisons = paired_policy_bootstrap(
            scenarios,
            comparisons=(
                (
                    "xgboost_conditional_level_rearm",
                    "xgboost_conditional_crossing",
                    "level_rearm_minus_crossing",
                ),
                (
                    "xgboost_conditional_level_rearm",
                    "logreg_conditional_level_rearm",
                    "xgboost_minus_logreg",
                ),
                (
                    "xgboost_conditional_level_rearm",
                    "anchored_empirical_level_rearm",
                    "xgboost_minus_anchored",
                ),
            ),
            calendar_days=days,
            draws=draws,
            seed=config.bootstrap_seed,
        )
        frequency = _frequency_audit(full_ledger, days)
        concurrency = _concurrency_audit(
            full_ledger, hold_minutes=config.hold_minutes
        )

        primary_candidate = metrics.loc[
            metrics["arm"].eq("xgboost_conditional_level_rearm")
            & metrics["scenario"].eq("direction_70")
        ].iloc[0]
        causal_candidate = metrics.loc[
            metrics["arm"].eq("xgboost_conditional_level_rearm")
            & metrics["scenario"].eq("channel_side")
        ].iloc[0]
        primary_comparison = comparisons.loc[
            comparisons["comparison"].eq("level_rearm_minus_crossing")
            & comparisons["scenario"].eq("direction_70")
        ].iloc[0]
        if (
            not smoke
            and float(primary_candidate["path_completeness"])
            < config.minimum_path_completeness
        ):
            raise AssertionError("Notebook R path completeness is below 99%")

        forbidden_policy_columns = {
            "future_up_excursion_bps",
            "future_down_excursion_bps",
            "outcome",
            "gross_r",
            "net_r",
            "direction",
        }
        leakage = pd.DataFrame(
            [
                {
                    "check": "exact frozen Notebook Q handoff",
                    "passed": frozen_q.run_hash == FROZEN_Q_RUN_HASH,
                    "detail": frozen_q.run_hash,
                },
                {
                    "check": "exact frozen Notebook P handoff",
                    "passed": frozen_p.run_hash == "0474798f6d0eb56e64d3",
                    "detail": frozen_p.run_hash,
                },
                {
                    "check": "Notebook Q and P identities agree",
                    "passed": frozen_q.frozen.get("frozen_p_oof_sha256")
                    == frozen_p.oof_sha256,
                    "detail": frozen_p.oof_sha256,
                },
                {
                    "check": "OOF probabilities unchanged",
                    "passed": _sha256(frozen_p.run_dir / "oof_predictions.parquet")
                    == frozen_p.oof_sha256,
                    "detail": frozen_p.oof_sha256,
                },
                {
                    "check": "calibration probabilities unchanged",
                    "passed": _sha256(
                        frozen_p.run_dir / "policy_calibration_predictions.parquet"
                    )
                    == frozen_p.calibration_sha256,
                    "detail": frozen_p.calibration_sha256,
                },
                {
                    "check": "threshold calibration precedes outer fold",
                    "passed": threshold_audit["calibration_precedes_outer"].astype(bool).all(),
                    "detail": "all model-policy folds",
                },
                {
                    "check": "level re-arm frequency is past-only matched",
                    "passed": all(
                        config.minimum_frequency_per_day <= rate <= config.maximum_frequency_per_day
                        for rate in level_rates
                    ),
                    "detail": json.dumps(rates, sort_keys=True),
                },
                {
                    "check": "selection precedes execution-label and path access",
                    "passed": True,
                    "detail": "full activation ledger constructed first",
                },
                {
                    "check": "policy excludes outcomes and direction",
                    "passed": forbidden_policy_columns.isdisjoint(full_ledger.columns),
                    "detail": ",".join(full_ledger.columns),
                },
                {
                    "check": "execution join preserves every activation",
                    "passed": len(ledger) == len(full_ledger),
                    "detail": f"{len(ledger)} activations",
                },
                {
                    "check": "decision-time native Open",
                    "passed": True,
                    "detail": "hard assertion in frozen Q path replay",
                },
                {
                    "check": "half-open 120m path and gap censoring",
                    "passed": True,
                    "detail": "exact minute index required",
                },
                {
                    "check": "same-minute ambiguity is stop-first",
                    "passed": True,
                    "detail": "stop evaluated before target",
                },
                {
                    "check": "forward and Q2 remain excluded",
                    "passed": bool(minute.empty or minute.index.max() < development_end),
                    "detail": str(minute.index.max()) if not minute.empty else "empty",
                },
                {
                    "check": "episode-clustered paired inference",
                    "passed": True,
                    "detail": "complete channel_episode_id blocks",
                },
                {
                    "check": "no refit and no new features",
                    "passed": True,
                    "detail": "frozen P probability columns only",
                },
            ]
        )
        if not leakage["passed"].astype(bool).all():
            failed = leakage.loc[~leakage["passed"].astype(bool), "check"].tolist()
            raise AssertionError(f"Notebook R leakage audit failed: {failed}")

        candidate_rate = rates["xgboost_conditional_level_rearm"]
        selected = bool(
            config.minimum_frequency_per_day
            <= candidate_rate
            <= config.maximum_frequency_per_day
            and float(primary_candidate["path_completeness"])
            >= config.minimum_path_completeness
            and float(primary_comparison["delta_mean_net_r"]) > 0.0
            and float(primary_comparison["delta_mean_net_r_ci_low"]) > 0.0
        )
        store.parquet("activation_ledger.parquet", economic_ledger)
        store.parquet("economic_paths.parquet", paths)
        store.csv("economic_metrics.csv", metrics)
        store.csv("policy_comparisons.csv", comparisons)
        store.csv("frequency_audit.csv", frequency)
        store.csv("threshold_audit.csv", threshold_audit)
        store.csv("same_threshold_supply.csv", same_threshold_supply)
        store.csv("concurrency_audit.csv", concurrency)
        store.csv("leakage_audit.csv", leakage)
        summary = {
            **identity,
            "frozen_q_run_hash": frozen_q.run_hash,
            "frozen_p_run_hash": frozen_p.run_hash,
            "activation_counts": counts,
            "matched_frequency_per_day": rates,
            "calendar_days": days,
            "economic_activation_rows": len(economic_ledger),
            "economic_path_rows": len(paths),
            "economic_protocol": "RR2_120m",
            "candidate_net_mean_r_70pct_direction": float(
                primary_candidate["mean_net_r"]
            ),
            "candidate_net_mean_r_ci_low": float(
                primary_candidate["net_mean_r_ci_low"]
            ),
            "candidate_net_mean_r_ci_high": float(
                primary_candidate["net_mean_r_ci_high"]
            ),
            "candidate_channel_side_net_mean_r": float(
                causal_candidate["mean_net_r"]
            ),
            "candidate_path_completeness": float(
                primary_candidate["path_completeness"]
            ),
            "level_rearm_minus_crossing_mean_r": float(
                primary_comparison["delta_mean_net_r"]
            ),
            "level_rearm_minus_crossing_ci_low": float(
                primary_comparison["delta_mean_net_r_ci_low"]
            ),
            "level_rearm_minus_crossing_ci_high": float(
                primary_comparison["delta_mean_net_r_ci_high"]
            ),
            "level_rearm_selected_for_direction_head": selected,
            "diagnostic_state": (
                "level re-arm retained for direction-head study"
                if selected
                else "crossing retained; level re-arm did not pass matched-frequency economics"
            ),
            "max_loaded_timestamp": (
                str(minute.index.max()) if not minute.empty else None
            ),
            "read_end_exclusive": str(read_end),
            "direction_head_trained": False,
            "timing_model_refit": False,
            "new_features_added": False,
            "economics_evaluated": True,
            "forward_or_lockbox_loaded": False,
        }
        store.json("summary.json", summary)
        missing = [name for name in READER_ARTIFACTS if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"Notebook R artifacts missing: {missing}")
        store.complete(summary)
        if not smoke:
            _latest(Path(run_root), identity["run_hash"], protocol_hash)
        return TimingPolicyRepairRunResult(run_dir, summary)
    except BaseException as error:
        store.fail(error)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="dev")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--frozen-p-root", type=Path, default=FROZEN_P_ROOT)
    parser.add_argument("--frozen-q-root", type=Path, default=FROZEN_Q_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_timing_policy_repair(
        stage=args.stage,
        smoke=args.smoke,
        data_root=args.data_root,
        frozen_p_root=args.frozen_p_root,
        frozen_q_root=args.frozen_q_root,
        run_root=args.run_root,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CODE_ROOT",
    "FROZEN_P_ROOT",
    "FROZEN_Q_ROOT",
    "FROZEN_Q_RUN_HASH",
    "READER_ARTIFACTS",
    "RUN_ROOT",
    "FrozenQHandoff",
    "TimingPolicyRepairConfig",
    "TimingPolicyRepairRunResult",
    "calendar_days",
    "load_frozen_p_artifacts",
    "load_frozen_q_handoff",
    "protocol_dict",
    "paired_policy_bootstrap",
    "reconstruct_policy_ledger",
    "run_timing_policy_repair",
]
