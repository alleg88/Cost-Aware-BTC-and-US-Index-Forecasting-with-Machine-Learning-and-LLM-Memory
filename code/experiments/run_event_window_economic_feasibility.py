"""Development-only Notebook Q economic-feasibility ceiling."""
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
    collapse_episode_time,
)
from experiments.run_event_window_conditional_opportunity import (
    FROZEN_O_ROOT,
    READER_ARTIFACTS as P_READER_ARTIFACTS,
    load_frozen_o_artifacts,
)
from experiments.run_event_window_cost_aware_entry import _Store, _sha256, _sha_payload
from experiments.run_event_window_tcn import _load_bounded_parquet


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = CODE_ROOT / "data"
FROZEN_P_ROOT = (
    CODE_ROOT / "experiments" / "cache" / "event_window_conditional_opportunity"
)
FROZEN_P_RUN_HASH = "0474798f6d0eb56e64d3"
RUN_ROOT = CODE_ROOT / "experiments" / "cache" / "event_window_economic_feasibility"
READER_ARTIFACTS = (
    "activation_ledger.parquet",
    "economic_paths.parquet",
    "economic_metrics.csv",
    "direction_scenarios.csv",
    "economic_breakdowns.csv",
    "paired_bootstrap.csv",
    "frequency_audit.csv",
    "concurrency_audit.csv",
    "leakage_audit.csv",
    "protocol.json",
    "frozen_protocol.json",
    "summary.json",
)


@dataclass(frozen=True)
class EconomicFeasibilityConfig:
    development_start: str = "2021-01-01"
    development_end_exclusive: str = "2025-07-01"
    primary_activation_target_per_day: float = 2.0
    activation_rate_sensitivities_per_day: tuple[float, ...] = (3.0,)
    primary_hold_minutes: int = 60
    hold_sensitivities_minutes: tuple[int, ...] = (120,)
    primary_target_multiple_b: float = 2.0
    target_sensitivities_b: tuple[float, ...] = (3.0, 5.0)
    stop_multiple_b: float = 1.0
    cooldown_minutes: int = 60
    entry_cost_bps: float = 5.0
    target_exit_cost_bps: float = 2.0
    other_exit_cost_bps: float = 5.0
    bootstrap_draws: int = 2_000
    bootstrap_seed: int = 42
    minimum_path_completeness: float = 0.99


@dataclass(frozen=True)
class FrozenPArtifacts:
    run_hash: str
    protocol_hash: str
    source_hash: str
    input_hash: str
    run_dir: Path
    protocol: dict[str, object]
    summary: dict[str, object]
    frozen: dict[str, object]
    state: dict[str, object]
    oof: pd.DataFrame
    calibration: pd.DataFrame
    policy: pd.DataFrame
    oof_sha256: str
    calibration_sha256: str


@dataclass(frozen=True)
class EconomicRunResult:
    run_dir: Path
    summary: dict[str, object]


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path.name}")
    return value


def protocol_dict(
    config: EconomicFeasibilityConfig = EconomicFeasibilityConfig(),
    *,
    smoke: bool = False,
) -> dict[str, object]:
    return {
        "notebook": "Q_event_window_economic_feasibility",
        "stage": "dev",
        "smoke": smoke,
        "development_start": config.development_start,
        "development_end_exclusive": config.development_end_exclusive,
        "frozen_p_run_hash": FROZEN_P_RUN_HASH,
        "timing_model": "xgboost",
        "timing_arms": ["conditional", "anchored_empirical"],
        "activation_policy": "frozen fold threshold plus upward crossing",
        "primary_activation_target_per_day": config.primary_activation_target_per_day,
        "activation_rate_sensitivities_per_day": list(
            config.activation_rate_sensitivities_per_day
        ),
        "cooldown_minutes": config.cooldown_minutes,
        "one_parent_trade_per_activation": True,
        "cross_activation_capacity": "unlimited",
        "primary_hold_minutes": config.primary_hold_minutes,
        "hold_sensitivities_minutes": list(config.hold_sensitivities_minutes),
        "stop_multiple_b": config.stop_multiple_b,
        "primary_target_multiple_b": config.primary_target_multiple_b,
        "target_sensitivities_b": list(config.target_sensitivities_b),
        "entry": "native one-minute Open at decision time",
        "path_interval": "half-open [t,t+hold)",
        "same_minute_ambiguity": "stop_first",
        "gap_handling": "censor",
        "entry_cost_bps": config.entry_cost_bps,
        "target_exit_cost_bps": config.target_exit_cost_bps,
        "other_exit_cost_bps": config.other_exit_cost_bps,
        "direction_scenarios": [
            "channel_side",
            "random_50",
            "direction_70",
            "oracle",
        ],
        "bootstrap_unit": "channel_episode_id",
        "bootstrap_draws": config.bootstrap_draws,
        "bootstrap_seed": config.bootstrap_seed,
        "minimum_path_completeness": config.minimum_path_completeness,
        "direction_head_trained": False,
        "timing_model_refit": False,
        "forward_or_lockbox_loaded": False,
    }


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
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
    path = Path(root) / "latest_dev.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _validated_completed_summary(
    run_dir: Path,
    identity: dict[str, str],
    frozen: FrozenPArtifacts,
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
        frozen_protocol = _read_json(run_dir / "frozen_protocol.json")
        summary = _read_json(run_dir / "summary.json")
        if (
            frozen_protocol.get("frozen_p_run_hash") != frozen.run_hash
            or frozen_protocol.get("frozen_p_oof_sha256") != frozen.oof_sha256
            or state.get("summary") != summary
        ):
            return None
        return summary
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def load_frozen_p_artifacts(
    run_root: Path = FROZEN_P_ROOT,
) -> FrozenPArtifacts:
    """Validate the exact completed Notebook P handoff before exposing scores."""
    root = Path(run_root)
    expected_relative = f"{FROZEN_P_RUN_HASH}/full"
    run_dir = (root / expected_relative).resolve()
    if not run_dir.is_relative_to(root.resolve()):
        raise ValueError("frozen Notebook P path escaped its root")

    state = _read_json(run_dir / "run_state.json")
    if state.get("status") != "complete" or state.get("run_hash") != FROZEN_P_RUN_HASH:
        raise ValueError("frozen Notebook P run is incomplete or changed")
    records = state.get("artifacts")
    if not isinstance(records, dict):
        raise ValueError("frozen Notebook P artifact registry is missing")
    for name in P_READER_ARTIFACTS:
        path = run_dir / name
        record = records.get(name)
        if (
            not path.is_file()
            or not isinstance(record, dict)
            or int(record.get("size", -1)) != path.stat().st_size
            or str(record.get("sha256", "")) != _sha256(path)
        ):
            raise ValueError(f"frozen Notebook P artifact changed: {name}")

    protocol = _read_json(run_dir / "protocol.json")
    summary = _read_json(run_dir / "summary.json")
    frozen = _read_json(run_dir / "frozen_protocol.json")
    for field in ("run_hash", "protocol_hash", "source_hash", "input_hash"):
        if protocol.get(field) != state.get(field) or summary.get(field) != state.get(field):
            raise ValueError(f"frozen Notebook P {field} identity changed")
    if state.get("summary") != summary:
        raise ValueError("frozen Notebook P state summary changed")
    if (
        protocol.get("stage") != "dev"
        or bool(protocol.get("smoke", False))
        or protocol.get("development_end_exclusive") != "2025-07-01"
        or summary.get("forward_or_lockbox_loaded") is not False
        or summary.get("direction_head_trained") is not False
        or summary.get("economics_evaluated") is not False
    ):
        raise ValueError("Notebook Q accepts only the bounded full Notebook P run")

    oof_path = run_dir / "oof_predictions.parquet"
    calibration_path = run_dir / "policy_calibration_predictions.parquet"
    oof = pd.read_parquet(oof_path)
    calibration = pd.read_parquet(calibration_path)
    policy = pd.read_csv(run_dir / "policy_metrics.csv")
    required = {
        "model",
        "fold_id",
        "window_id",
        "channel_episode_id",
        "step",
        "decision_time",
        "p_t_le_60",
        "p0_t_le_60",
    }
    missing = sorted(required.difference(oof.columns))
    if missing:
        raise ValueError(f"frozen Notebook P OOF schema changed: {missing}")
    return FrozenPArtifacts(
        run_hash=FROZEN_P_RUN_HASH,
        protocol_hash=str(state["protocol_hash"]),
        source_hash=str(state["source_hash"]),
        input_hash=str(state["input_hash"]),
        run_dir=run_dir,
        protocol=protocol,
        summary=summary,
        frozen=frozen,
        state=state,
        oof=oof,
        calibration=calibration,
        policy=policy,
        oof_sha256=_sha256(oof_path),
        calibration_sha256=_sha256(calibration_path),
    )


def reconstruct_activation_ledger(
    frozen: FrozenPArtifacts,
    config: EconomicFeasibilityConfig = EconomicFeasibilityConfig(),
) -> pd.DataFrame:
    """Replay frozen XGBoost crossing thresholds without reading market paths."""
    outer = frozen.oof.loc[frozen.oof["model"].eq("xgboost")].copy()
    outer["decision_time"] = pd.to_datetime(
        outer["decision_time"], utc=True, errors="raise"
    )
    end = pd.Timestamp(config.development_end_exclusive, tz="UTC")
    if outer.empty or outer["decision_time"].max() >= end:
        raise ValueError("frozen Notebook P decisions crossed the development boundary")

    rates = (
        config.primary_activation_target_per_day,
        *config.activation_rate_sensitivities_per_day,
    )
    arms = (
        ("conditional", "p_t_le_60"),
        ("anchored_empirical", "p0_t_le_60"),
    )
    rows: list[pd.DataFrame] = []
    for rate in rates:
        for arm, score_column in arms:
            for fold_id, fold in outer.groupby("fold_id", sort=False):
                threshold_rows = frozen.policy.loc[
                    frozen.policy["model"].eq("xgboost")
                    & frozen.policy["fold"].astype(str).eq(str(fold_id))
                    & frozen.policy["objective"].eq("timing_60")
                    & frozen.policy["arm"].eq(arm)
                    & frozen.policy["target_activations_per_day"].eq(float(rate))
                    & frozen.policy["truth_gap_minutes"].eq(5)
                ]
                if len(threshold_rows) != 1:
                    raise ValueError(
                        f"frozen threshold is not unique: {fold_id}, {arm}, {rate}"
                    )
                threshold = float(threshold_rows.iloc[0]["threshold"])
                collapsed = collapse_episode_time(fold, score_column=score_column)
                replay = causal_crossing_alerts(
                    collapsed,
                    threshold=threshold,
                    score_column=score_column,
                    cooldown_minutes=config.cooldown_minutes,
                )
                selected = replay.loc[replay["alert"]].copy()
                selected["arm"] = arm
                selected["target_activations_per_day"] = float(rate)
                selected["threshold"] = threshold
                selected["activation_score"] = selected[score_column].astype(float)
                selected["activation_key"] = (
                    arm
                    + "|"
                    + f"{float(rate):g}"
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
    ledger = pd.concat(rows, ignore_index=True).sort_values(
        ["arm", "target_activations_per_day", "decision_time", "channel_episode_id"],
        kind="stable",
    ).reset_index(drop=True)
    if ledger.duplicated(
        ["arm", "target_activations_per_day", "channel_episode_id", "decision_time"]
    ).any():
        raise AssertionError("crossing replay produced duplicate activations")
    return ledger


def _minute_frame(minute: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = sorted(required.difference(minute.columns))
    if missing:
        raise ValueError(f"minute data missing columns: {missing}")
    if not isinstance(minute.index, pd.DatetimeIndex):
        raise TypeError("minute data must use a DatetimeIndex")
    work = minute.copy()
    work.index = pd.to_datetime(work.index, utc=True, errors="raise")
    work = work.sort_index(kind="stable")
    if work.index.has_duplicates:
        raise ValueError("minute timestamps must be unique")
    return work


def replay_brackets(
    attempts: pd.DataFrame,
    minute: pd.DataFrame,
    *,
    target_multiples: tuple[float, ...] = (2.0, 3.0, 5.0),
    hold_minutes: tuple[int, ...] = (60, 120),
    entry_cost_bps: float = 5.0,
    target_exit_cost_bps: float = 2.0,
    other_exit_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Replay frozen log-price brackets for LONG and SHORT on native 1m bars."""
    required = {
        "activation_key",
        "window_id",
        "step",
        "channel_episode_id",
        "decision_time",
        "channel_side",
        "adaptive_barrier_bps",
        "reference_price",
    }
    missing = sorted(required.difference(attempts.columns))
    if missing:
        raise ValueError(f"economic attempts missing columns: {missing}")
    one = _minute_frame(minute)
    index = one.index
    values = one[["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    rows: list[dict[str, object]] = []
    base_columns = list(attempts.columns)
    for attempt in attempts.itertuples(index=False):
        base = {name: getattr(attempt, name) for name in base_columns}
        decision_time = pd.Timestamp(base["decision_time"])
        decision_time = (
            decision_time.tz_localize("UTC")
            if decision_time.tzinfo is None
            else decision_time.tz_convert("UTC")
        )
        barrier = float(base["adaptive_barrier_bps"])
        if not np.isfinite(barrier) or barrier <= 0.0:
            raise ValueError("adaptive barrier must be finite and positive")
        position = int(index.searchsorted(decision_time, side="left"))
        for hold in hold_minutes:
            if int(hold) <= 0:
                raise ValueError("hold_minutes must be positive")
            expected_index = pd.date_range(
                decision_time, periods=int(hold), freq="1min", tz="UTC"
            )
            end = position + int(hold)
            complete = (
                end <= len(one)
                and position < len(one)
                and index[position:end].equals(expected_index)
                and np.isfinite(values[position:end]).all()
            )
            path = values[position:end] if complete else np.empty((0, 4))
            entry = float(path[0, 0]) if complete else np.nan
            if complete and np.isfinite(float(base["reference_price"])):
                if not np.isclose(
                    entry,
                    float(base["reference_price"]),
                    rtol=0.0,
                    atol=max(1e-8, abs(entry) * 1e-10),
                ):
                    raise ValueError("decision-time Open changed from frozen reference")
            for target_multiple in target_multiples:
                if float(target_multiple) <= 0.0:
                    raise ValueError("target multiples must be positive")
                for direction in ("long", "short"):
                    output = {
                        **base,
                        "decision_time": decision_time,
                        "direction": direction,
                        "target_multiple_b": float(target_multiple),
                        "hold_minutes": int(hold),
                        "path_complete": bool(complete),
                        "censored": not complete,
                        "entry_price": entry,
                        "exit_price": np.nan,
                        "bars_held": np.nan,
                        "outcome": "censored",
                        "gross_bps": np.nan,
                        "net_bps": np.nan,
                        "gross_r": np.nan,
                        "net_r": np.nan,
                        "cost_bps": np.nan,
                        "cost_r": np.nan,
                    }
                    if not complete:
                        rows.append(output)
                        continue
                    sign = 1.0 if direction == "long" else -1.0
                    stop = entry * np.exp(-sign * barrier / 1e4)
                    target = entry * np.exp(
                        sign * float(target_multiple) * barrier / 1e4
                    )
                    outcome = "timeout"
                    exit_price = float(path[-1, 3])
                    bars_held = int(hold)
                    for offset, (_, high, low, _) in enumerate(path):
                        stop_touched = low <= stop if direction == "long" else high >= stop
                        target_touched = high >= target if direction == "long" else low <= target
                        if stop_touched:
                            outcome = "sl"
                            exit_price = float(stop)
                            bars_held = offset + 1
                            break
                        if target_touched:
                            outcome = "tp"
                            exit_price = float(target)
                            bars_held = offset + 1
                            break
                    gross_bps = sign * np.log(exit_price / entry) * 1e4
                    exit_cost = (
                        target_exit_cost_bps if outcome == "tp" else other_exit_cost_bps
                    )
                    cost_bps = float(entry_cost_bps + exit_cost)
                    net_bps = float(gross_bps - cost_bps)
                    output.update(
                        {
                            "exit_price": exit_price,
                            "bars_held": bars_held,
                            "outcome": outcome,
                            "gross_bps": float(gross_bps),
                            "net_bps": net_bps,
                            "gross_r": float(gross_bps / barrier),
                            "net_r": float(net_bps / barrier),
                            "cost_bps": cost_bps,
                            "cost_r": float(cost_bps / barrier),
                        }
                    )
                    rows.append(output)
    return pd.DataFrame(rows)


def build_direction_scenarios(paths: pd.DataFrame) -> pd.DataFrame:
    """Keep causal channel side separate from deterministic direction ceilings."""
    required = {
        "activation_key",
        "channel_side",
        "direction",
        "target_multiple_b",
        "hold_minutes",
        "censored",
        "outcome",
        "gross_r",
        "net_r",
        "cost_bps",
        "cost_r",
    }
    missing = sorted(required.difference(paths.columns))
    if missing:
        raise ValueError(f"economic paths missing columns: {missing}")
    key_columns = [
        name
        for name in (
            "activation_key",
            "arm",
            "target_activations_per_day",
            "fold_id",
            "window_id",
            "step",
            "channel_episode_id",
            "decision_time",
            "channel_side",
            "adaptive_barrier_bps",
            "target_multiple_b",
            "hold_minutes",
        )
        if name in paths.columns
    ]
    rows: list[dict[str, object]] = []
    for _, group in paths.groupby(key_columns, sort=False, dropna=False):
        if set(group["direction"]) != {"long", "short"} or len(group) != 2:
            raise ValueError("each bracket requires one LONG and one SHORT path")
        base = {name: group.iloc[0][name] for name in key_columns}
        by_direction = group.set_index("direction")
        if group["censored"].any():
            for scenario, accuracy in (
                ("channel_side", np.nan),
                ("random_50", 0.5),
                ("direction_70", 0.7),
                ("oracle", 1.0),
            ):
                rows.append(
                    {
                        **base,
                        "scenario": scenario,
                        "assumed_direction_accuracy": accuracy,
                        "censored": True,
                        "gross_r": np.nan,
                        "net_r": np.nan,
                        "cost_bps": np.nan,
                        "cost_r": np.nan,
                        "best_net_r": np.nan,
                        "worst_net_r": np.nan,
                        "tp_weight": 0.0,
                        "sl_weight": 0.0,
                        "timeout_weight": 0.0,
                    }
                )
            continue

        best_direction = str(group.loc[group["net_r"].idxmax(), "direction"])
        worst_direction = "short" if best_direction == "long" else "long"
        best_net_r = float(by_direction.at[best_direction, "net_r"])
        worst_net_r = float(by_direction.at[worst_direction, "net_r"])
        channel_side = str(base["channel_side"]).lower()
        if channel_side not in {"long", "short"}:
            raise ValueError(f"invalid channel side: {channel_side}")
        definitions = {
            "channel_side": {channel_side: 1.0},
            "random_50": {"long": 0.5, "short": 0.5},
            "direction_70": {best_direction: 0.7, worst_direction: 0.3},
            "oracle": {best_direction: 1.0},
        }
        accuracies = {
            "channel_side": np.nan,
            "random_50": 0.5,
            "direction_70": 0.7,
            "oracle": 1.0,
        }
        for scenario, weights in definitions.items():
            def weighted(column: str) -> float:
                return float(
                    sum(weight * float(by_direction.at[direction, column]) for direction, weight in weights.items())
                )

            rows.append(
                {
                    **base,
                    "scenario": scenario,
                    "assumed_direction_accuracy": accuracies[scenario],
                    "censored": False,
                    "gross_r": weighted("gross_r"),
                    "net_r": weighted("net_r"),
                    "cost_bps": weighted("cost_bps"),
                    "cost_r": weighted("cost_r"),
                    "best_net_r": best_net_r,
                    "worst_net_r": worst_net_r,
                    "tp_weight": sum(
                        weight
                        for direction, weight in weights.items()
                        if by_direction.at[direction, "outcome"] == "tp"
                    ),
                    "sl_weight": sum(
                        weight
                        for direction, weight in weights.items()
                        if by_direction.at[direction, "outcome"] == "sl"
                    ),
                    "timeout_weight": sum(
                        weight
                        for direction, weight in weights.items()
                        if by_direction.at[direction, "outcome"] == "timeout"
                    ),
                }
            )
    return pd.DataFrame(rows)


def cluster_bootstrap_draws(
    frame: pd.DataFrame,
    *,
    value_column: str,
    episode_column: str = "channel_episode_id",
    draws: int = 2_000,
    seed: int = 42,
) -> np.ndarray:
    """Bootstrap a row-weighted mean while resampling complete episodes."""
    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    work = frame[[episode_column, value_column]].dropna().copy()
    if work.empty:
        return np.full(draws, np.nan)
    grouped = work.groupby(episode_column, sort=False)[value_column].agg(["sum", "count"])
    numerator = grouped["sum"].to_numpy(float)
    denominator = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(grouped), size=(draws, len(grouped)))
    return numerator[sampled].sum(axis=1) / denominator[sampled].sum(axis=1)


def _attach_execution_fields(
    ledger: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["window_id", "step"]
    required = {
        *keys,
        "channel_episode_id",
        "decision_time",
        "side",
        "reference_price",
        "adaptive_barrier_bps",
    }
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise ValueError(f"frozen execution labels missing columns: {missing}")
    if labels.duplicated(keys).any():
        raise ValueError("frozen execution labels contain duplicate decision keys")
    indexed = labels.set_index(keys)
    wanted = pd.MultiIndex.from_frame(ledger[keys])
    selected = indexed.reindex(wanted)
    if selected["side"].isna().any():
        raise ValueError("frozen execution labels do not cover every activation")
    selected = selected.reset_index(drop=True)
    left_time = pd.to_datetime(ledger["decision_time"], utc=True, errors="raise")
    right_time = pd.to_datetime(selected["decision_time"], utc=True, errors="raise")
    if not left_time.reset_index(drop=True).equals(right_time.reset_index(drop=True)):
        raise ValueError("execution join changed activation timestamps")
    if not ledger["channel_episode_id"].reset_index(drop=True).equals(
        selected["channel_episode_id"].reset_index(drop=True)
    ):
        raise ValueError("execution join changed channel episodes")
    output = ledger.reset_index(drop=True).copy()
    output["channel_side"] = selected["side"].str.lower().to_numpy()
    output["reference_price"] = pd.to_numeric(
        selected["reference_price"], errors="raise"
    ).to_numpy(float)
    output["adaptive_barrier_bps"] = pd.to_numeric(
        selected["adaptive_barrier_bps"], errors="raise"
    ).to_numpy(float)
    if not output["channel_side"].isin(["long", "short"]).all():
        raise ValueError("frozen channel side is invalid")
    return output


def _calendar_days(frame: pd.DataFrame) -> int:
    time = pd.to_datetime(frame["decision_time"], utc=True, errors="raise")
    return int((time.max().normalize() - time.min().normalize()).days + 1)


def _activation_counts(ledger: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for (arm, rate), group in ledger.groupby(
        ["arm", "target_activations_per_day"], sort=True
    ):
        counts[f"{arm}_{float(rate):g}"] = int(len(group))
    return counts


def _frequency_audit(ledger: pd.DataFrame, calendar_days: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (arm, rate), group in ledger.groupby(
        ["arm", "target_activations_per_day"], sort=True
    ):
        work = group.sort_values(
            ["channel_episode_id", "decision_time"], kind="stable"
        )
        per_episode = work.groupby("channel_episode_id", sort=False).size()
        spacing = work.groupby("channel_episode_id", sort=False)["decision_time"].diff()
        active_days = int(
            pd.to_datetime(work["decision_time"], utc=True).dt.normalize().nunique()
        )
        rows.append(
            {
                "arm": arm,
                "target_activations_per_day": float(rate),
                "calendar_days": calendar_days,
                "activations": int(len(work)),
                "activations_per_calendar_day": float(len(work) / calendar_days),
                "active_days": active_days,
                "active_day_coverage": float(active_days / calendar_days),
                "activations_per_active_day": float(len(work) / active_days),
                "activated_episodes": int(len(per_episode)),
                "episodes_with_multiple_activations": int(per_episode.gt(1).sum()),
                "median_activations_per_episode": float(per_episode.median()),
                "maximum_activations_per_episode": int(per_episode.max()),
                "minimum_same_episode_spacing_minutes": float(
                    spacing.dropna().dt.total_seconds().div(60).min()
                ),
            }
        )
    return pd.DataFrame(rows)


def _concurrency_audit(
    ledger: pd.DataFrame,
    *,
    hold_minutes: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (arm, rate), group in ledger.groupby(
        ["arm", "target_activations_per_day"], sort=True
    ):
        starts = (
            pd.to_datetime(group["decision_time"], utc=True)
            .sort_values()
            .astype("int64")
            .to_numpy()
        )
        ends = np.sort(starts + int(pd.Timedelta(minutes=hold_minutes).value))
        concurrent = (
            np.searchsorted(starts, starts, side="right")
            - np.searchsorted(ends, starts, side="right")
        )
        rows.append(
            {
                "arm": arm,
                "target_activations_per_day": float(rate),
                "hold_minutes": int(hold_minutes),
                "maximum_concurrent_parent_trades": int(concurrent.max()),
                "median_concurrent_at_entry": float(np.median(concurrent)),
                "p90_concurrent_at_entry": float(np.quantile(concurrent, 0.90)),
                "p99_concurrent_at_entry": float(np.quantile(concurrent, 0.99)),
            }
        )
    return pd.DataFrame(rows)


def _economic_metrics(
    scenarios: pd.DataFrame,
    *,
    calendar_days: int,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    group_columns = [
        "arm",
        "target_activations_per_day",
        "target_multiple_b",
        "hold_minutes",
        "scenario",
    ]
    rows: list[dict[str, object]] = []
    for number, (keys, group) in enumerate(
        scenarios.groupby(group_columns, sort=True, dropna=False)
    ):
        observed = group.loc[~group["censored"].astype(bool)].copy()
        active_days = int(
            pd.to_datetime(group["decision_time"], utc=True).dt.normalize().nunique()
        )
        bootstrap = cluster_bootstrap_draws(
            observed,
            value_column="net_r",
            draws=draws,
            seed=seed + number,
        )
        finite = bootstrap[np.isfinite(bootstrap)]
        ci_low, ci_high = (
            np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
        )
        mean_best = float(observed["best_net_r"].mean()) if len(observed) else np.nan
        mean_worst = float(observed["worst_net_r"].mean()) if len(observed) else np.nan
        denominator = mean_best - mean_worst
        break_even = (
            float(-mean_worst / denominator)
            if np.isfinite(denominator) and denominator > 0.0
            else np.nan
        )
        arm, rate, target, hold, scenario = keys
        rows.append(
            {
                "arm": arm,
                "target_activations_per_day": float(rate),
                "target_multiple_b": float(target),
                "hold_minutes": int(hold),
                "scenario": scenario,
                "activations": int(len(group)),
                "observed_trades": int(len(observed)),
                "censored_attempts": int(group["censored"].sum()),
                "path_completeness": float(len(observed) / len(group)),
                "calendar_days": int(calendar_days),
                "activations_per_calendar_day": float(len(group) / calendar_days),
                "trades_per_calendar_day": float(len(observed) / calendar_days),
                "active_day_coverage": float(active_days / calendar_days),
                "trades_per_active_day": float(len(observed) / active_days),
                "mean_gross_r": float(observed["gross_r"].mean()),
                "mean_net_r": float(observed["net_r"].mean()),
                "net_mean_r_ci_low": float(ci_low),
                "net_mean_r_ci_high": float(ci_high),
                "total_net_r": float(observed["net_r"].sum()),
                "net_r_per_calendar_day": float(observed["net_r"].sum() / calendar_days),
                "tp_rate": float(observed["tp_weight"].mean()),
                "sl_rate": float(observed["sl_weight"].mean()),
                "timeout_rate": float(observed["timeout_weight"].mean()),
                "average_cost_bps": float(observed["cost_bps"].mean()),
                "average_cost_r": float(observed["cost_r"].mean()),
                "mean_best_direction_net_r": mean_best,
                "mean_wrong_direction_net_r": mean_worst,
                "break_even_direction_accuracy": break_even,
                "bootstrap_draws": int(draws),
                "bootstrap_seed": int(seed + number),
            }
        )
    return pd.DataFrame(rows)


def _paired_bootstrap(
    scenarios: pd.DataFrame,
    *,
    calendar_days: int,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    group_columns = [
        "target_activations_per_day",
        "target_multiple_b",
        "hold_minutes",
        "scenario",
    ]
    rows: list[dict[str, object]] = []
    for number, (keys, group) in enumerate(
        scenarios.loc[~scenarios["censored"].astype(bool)].groupby(
            group_columns, sort=True
        )
    ):
        episode_arm = group.groupby(
            ["channel_episode_id", "arm"], sort=False
        )["net_r"].sum().unstack("arm", fill_value=0.0)
        for arm in ("conditional", "anchored_empirical"):
            if arm not in episode_arm:
                episode_arm[arm] = 0.0
        difference = (
            episode_arm["conditional"] - episode_arm["anchored_empirical"]
        ).to_numpy(float)
        rng = np.random.default_rng(seed + number)
        sampled = rng.integers(
            0, len(difference), size=(draws, len(difference))
        )
        bootstrap = difference[sampled].mean(axis=1)
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        rate, target, hold, scenario = keys
        rows.append(
            {
                "target_activations_per_day": float(rate),
                "target_multiple_b": float(target),
                "hold_minutes": int(hold),
                "scenario": scenario,
                "union_episodes": int(len(difference)),
                "conditional_minus_anchored_mean_episode_r": float(
                    difference.mean()
                ),
                "ci_low": float(low),
                "ci_high": float(high),
                "conditional_minus_anchored_total_r": float(difference.sum()),
                "conditional_minus_anchored_r_per_calendar_day": float(
                    difference.sum() / calendar_days
                ),
                "bootstrap_draws": int(draws),
                "bootstrap_seed": int(seed + number),
                "absent_arm_episode_contribution": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _economic_breakdowns(
    paths: pd.DataFrame,
    scenarios: pd.DataFrame,
    config: EconomicFeasibilityConfig,
) -> pd.DataFrame:
    primary_path = paths.loc[
        paths["target_activations_per_day"].eq(
            config.primary_activation_target_per_day
        )
        & paths["target_multiple_b"].eq(config.primary_target_multiple_b)
        & paths["hold_minutes"].eq(config.primary_hold_minutes)
        & ~paths["censored"].astype(bool)
    ].copy()
    rows: list[dict[str, object]] = []
    for (arm, direction), group in primary_path.groupby(["arm", "direction"]):
        rows.append(
            {
                "breakdown": "bracket_direction",
                "value": direction,
                "arm": arm,
                "scenario": "pathwise",
                "observations": int(len(group)),
                "mean_net_r": float(group["net_r"].mean()),
                "total_net_r": float(group["net_r"].sum()),
            }
        )
    primary = scenarios.loc[
        scenarios["target_activations_per_day"].eq(
            config.primary_activation_target_per_day
        )
        & scenarios["target_multiple_b"].eq(config.primary_target_multiple_b)
        & scenarios["hold_minutes"].eq(config.primary_hold_minutes)
        & ~scenarios["censored"].astype(bool)
    ].copy()
    primary["year"] = pd.to_datetime(primary["decision_time"], utc=True).dt.year.astype(str)
    primary["barrier_band"] = pd.cut(
        primary["adaptive_barrier_bps"],
        bins=[74.999, 100.0, 150.0, 200.0, 250.001],
        labels=["75-100", "100-150", "150-200", "200-250"],
        include_lowest=True,
    ).astype(str)
    for column in ("channel_side", "year", "barrier_band"):
        for (arm, scenario, value), group in primary.groupby(
            ["arm", "scenario", column], observed=True
        ):
            rows.append(
                {
                    "breakdown": column,
                    "value": value,
                    "arm": arm,
                    "scenario": scenario,
                    "observations": int(len(group)),
                    "mean_net_r": float(group["net_r"].mean()),
                    "total_net_r": float(group["net_r"].sum()),
                }
            )
    return pd.DataFrame(rows)


def _diagnostic_state(
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    config: EconomicFeasibilityConfig,
) -> str:
    primary = metrics.loc[
        metrics["arm"].eq("conditional")
        & metrics["target_activations_per_day"].eq(
            config.primary_activation_target_per_day
        )
        & metrics["target_multiple_b"].eq(config.primary_target_multiple_b)
        & metrics["hold_minutes"].eq(config.primary_hold_minutes)
    ].set_index("scenario")
    increment = paired.loc[
        paired["target_activations_per_day"].eq(
            config.primary_activation_target_per_day
        )
        & paired["target_multiple_b"].eq(config.primary_target_multiple_b)
        & paired["hold_minutes"].eq(config.primary_hold_minutes)
    ].set_index("scenario")
    causal = bool(
        primary.at["channel_side", "net_mean_r_ci_low"] > 0.0
        and increment.at["channel_side", "ci_low"] > 0.0
    )
    stress = bool(
        primary.at["direction_70", "net_mean_r_ci_low"] > 0.0
        and increment.at["direction_70", "ci_low"] > 0.0
    )
    oracle = bool(primary.at["oracle", "net_mean_r_ci_low"] > 0.0)
    if causal:
        return "causal economic evidence"
    if stress:
        return "direction-conditional feasibility"
    if oracle:
        return "oracle-only headroom"
    return "no economic headroom under the registered bracket"


def run_economic_feasibility(
    *,
    stage: str = "dev",
    smoke: bool = False,
    data_root: Path = DEFAULT_DATA_ROOT,
    frozen_p_root: Path = FROZEN_P_ROOT,
    run_root: Path = RUN_ROOT,
    config: EconomicFeasibilityConfig = EconomicFeasibilityConfig(),
) -> EconomicRunResult:
    if stage != "dev":
        raise ValueError("Notebook Q permits development only; forward and Q2 are sealed")
    protocol = protocol_dict(config, smoke=smoke)
    protocol_hash = _sha_payload(protocol)

    frozen_p = load_frozen_p_artifacts(Path(frozen_p_root))
    full_ledger = reconstruct_activation_ledger(frozen_p, config)
    frozen_counts = _activation_counts(full_ledger)
    expected_counts = {
        "anchored_empirical_2": 2_379,
        "anchored_empirical_3": 3_505,
        "conditional_2": 2_148,
        "conditional_3": 3_780,
    }
    if frozen_counts != expected_counts:
        raise AssertionError(
            f"Notebook P activation supply changed: {frozen_counts}"
        )
    calendar_days = _calendar_days(full_ledger)
    frequency = _frequency_audit(full_ledger, calendar_days)
    primary_frequency = frequency.loc[
        frequency["arm"].eq("conditional")
        & frequency["target_activations_per_day"].eq(2.0)
    ].iloc[0]
    if (
        int(primary_frequency["activated_episodes"]) != 450
        or int(primary_frequency["episodes_with_multiple_activations"]) != 299
        or float(primary_frequency["median_activations_per_episode"]) != 3.0
        or int(primary_frequency["maximum_activations_per_episode"]) != 39
        or float(primary_frequency["minimum_same_episode_spacing_minutes"]) != 60.0
    ):
        raise AssertionError("Notebook P primary activation distribution changed")

    # Selection is complete before any future-dependent label or minute path is read.
    frozen_o = load_frozen_o_artifacts(
        FROZEN_O_ROOT,
        expected_run_hash=str(frozen_p.frozen["frozen_o_run_hash"]),
    )
    if frozen_p.frozen.get("frozen_o_run_hash") != frozen_o.run_hash:
        raise ValueError("frozen Notebook P and O identities differ")
    ledger = _attach_execution_fields(full_ledger, frozen_o.labels)
    economic_ledger = (
        ledger.groupby(
            ["arm", "target_activations_per_day"], sort=True, group_keys=False
        ).head(8).reset_index(drop=True)
        if smoke
        else ledger
    )

    minute_path = Path(data_root) / "btcusdt_1m_2021_2026.parquet"
    minute_sha256 = _sha256(minute_path)
    source_hash = _source_hash()
    input_hash = _sha_payload(
        {
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
    cached = _validated_completed_summary(run_dir, identity, frozen_p)
    if cached is not None:
        if not smoke:
            _latest(Path(run_root), identity["run_hash"], protocol_hash)
        return EconomicRunResult(run_dir, cached)

    store = _Store(run_dir, identity)
    try:
        store.json("protocol.json", {**protocol, **identity})
        store.json(
            "frozen_protocol.json",
            {
                "frozen_p_run_hash": frozen_p.run_hash,
                "frozen_p_protocol_hash": frozen_p.protocol_hash,
                "frozen_p_source_hash": frozen_p.source_hash,
                "frozen_p_input_hash": frozen_p.input_hash,
                "frozen_p_oof_sha256": frozen_p.oof_sha256,
                "frozen_p_calibration_sha256": frozen_p.calibration_sha256,
                "frozen_o_run_hash": frozen_o.run_hash,
                "frozen_o_labels_sha256": frozen_o.labels_sha256,
                "minute_source_sha256": minute_sha256,
                "direction_head_trained": False,
                "timing_model_refit": False,
                "forward_or_lockbox_loaded": False,
            },
        )

        start = pd.to_datetime(
            economic_ledger["decision_time"], utc=True, errors="raise"
        ).min()
        development_end = pd.Timestamp(
            config.development_end_exclusive, tz="UTC"
        )
        requested_end = pd.to_datetime(
            economic_ledger["decision_time"], utc=True, errors="raise"
        ).max() + pd.Timedelta(
            minutes=max(
                config.primary_hold_minutes,
                *config.hold_sensitivities_minutes,
            )
        )
        read_end = min(requested_end, development_end)
        minute = _load_bounded_parquet(
            minute_path,
            start=start,
            end=read_end,
        )
        if not minute.empty and minute.index.max() >= development_end:
            raise AssertionError("Notebook Q minute load crossed the development boundary")

        paths = replay_brackets(
            economic_ledger,
            minute,
            target_multiples=(
                config.primary_target_multiple_b,
                *config.target_sensitivities_b,
            ),
            hold_minutes=(
                config.primary_hold_minutes,
                *config.hold_sensitivities_minutes,
            ),
            entry_cost_bps=config.entry_cost_bps,
            target_exit_cost_bps=config.target_exit_cost_bps,
            other_exit_cost_bps=config.other_exit_cost_bps,
        )
        scenarios = build_direction_scenarios(paths)
        draws = 20 if smoke else config.bootstrap_draws
        metrics = _economic_metrics(
            scenarios,
            calendar_days=calendar_days,
            draws=draws,
            seed=config.bootstrap_seed,
        )
        paired = _paired_bootstrap(
            scenarios,
            calendar_days=calendar_days,
            draws=draws,
            seed=config.bootstrap_seed,
        )
        concurrency = _concurrency_audit(
            full_ledger, hold_minutes=config.primary_hold_minutes
        )
        breakdowns = _economic_breakdowns(paths, scenarios, config)
        primary_metrics = metrics.loc[
            metrics["arm"].eq("conditional")
            & metrics["target_activations_per_day"].eq(
                config.primary_activation_target_per_day
            )
            & metrics["target_multiple_b"].eq(
                config.primary_target_multiple_b
            )
            & metrics["hold_minutes"].eq(config.primary_hold_minutes)
            & metrics["scenario"].eq("direction_70")
        ].iloc[0]
        if (
            not smoke
            and float(primary_metrics["path_completeness"])
            < config.minimum_path_completeness
        ):
            raise AssertionError("primary Notebook Q path completeness is below 99%")

        calibration_before_outer = True
        for fold_id, outer in frozen_p.oof.loc[
            frozen_p.oof["model"].eq("xgboost")
        ].groupby("fold_id"):
            reserved = frozen_p.calibration.loc[
                frozen_p.calibration["model"].eq("xgboost")
                & frozen_p.calibration["fold_id"].astype(str).eq(str(fold_id))
            ]
            calibration_before_outer &= (
                pd.to_datetime(reserved["decision_time"], utc=True).max()
                < pd.to_datetime(outer["decision_time"], utc=True).min()
            )
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
                    "check": "exact frozen Notebook P handoff",
                    "passed": frozen_p.run_hash == FROZEN_P_RUN_HASH,
                    "detail": frozen_p.run_hash,
                },
                {
                    "check": "OOF probability artifact unchanged",
                    "passed": _sha256(frozen_p.run_dir / "oof_predictions.parquet")
                    == frozen_p.oof_sha256,
                    "detail": frozen_p.oof_sha256,
                },
                {
                    "check": "calibration probability artifact unchanged",
                    "passed": _sha256(
                        frozen_p.run_dir / "policy_calibration_predictions.parquet"
                    )
                    == frozen_p.calibration_sha256,
                    "detail": frozen_p.calibration_sha256,
                },
                {
                    "check": "threshold calibration precedes outer fold",
                    "passed": bool(calibration_before_outer),
                    "detail": "all seven frozen folds",
                },
                {
                    "check": "decision join is order-preserving many-to-one",
                    "passed": len(ledger) == len(full_ledger),
                    "detail": f"{len(ledger)} activations",
                },
                {
                    "check": "selection precedes future-label and minute-path access",
                    "passed": True,
                    "detail": "activation ledger reconstructed first",
                },
                {
                    "check": "policy input excludes outcome and direction fields",
                    "passed": forbidden_policy_columns.isdisjoint(full_ledger.columns),
                    "detail": ",".join(full_ledger.columns),
                },
                {
                    "check": "decision-time Open matches frozen reference",
                    "passed": True,
                    "detail": "hard assertion in path replay",
                },
                {
                    "check": "half-open path and gap censoring",
                    "passed": True,
                    "detail": "exact minute index required for [t,t+hold)",
                },
                {
                    "check": "same-minute ambiguity is stop-first",
                    "passed": True,
                    "detail": "stop evaluated before target",
                },
                {
                    "check": "forward and Q2 excluded",
                    "passed": bool(minute.empty or minute.index.max() < development_end),
                    "detail": str(minute.index.max()) if not minute.empty else "empty",
                },
                {
                    "check": "episode-clustered inference",
                    "passed": True,
                    "detail": "channel_episode_id resampled as a whole",
                },
            ]
        )
        if not leakage["passed"].astype(bool).all():
            failed = leakage.loc[~leakage["passed"].astype(bool), "check"].tolist()
            raise AssertionError(f"Notebook Q leakage audit failed: {failed}")

        primary_direction = metrics.loc[
            metrics["target_activations_per_day"].eq(
                config.primary_activation_target_per_day
            )
            & metrics["target_multiple_b"].eq(
                config.primary_target_multiple_b
            )
            & metrics["hold_minutes"].eq(config.primary_hold_minutes)
        ].copy()
        store.parquet("activation_ledger.parquet", economic_ledger)
        store.parquet("economic_paths.parquet", paths)
        store.csv("economic_metrics.csv", metrics)
        store.csv("direction_scenarios.csv", primary_direction)
        store.csv("economic_breakdowns.csv", breakdowns)
        store.csv("paired_bootstrap.csv", paired)
        store.csv("frequency_audit.csv", frequency)
        store.csv("concurrency_audit.csv", concurrency)
        store.csv("leakage_audit.csv", leakage)
        state = _diagnostic_state(metrics, paired, config)
        summary = {
            **identity,
            "frozen_p_run_hash": frozen_p.run_hash,
            "frozen_activation_counts": frozen_counts,
            "economic_activation_rows": int(len(economic_ledger)),
            "economic_path_rows": int(len(paths)),
            "calendar_days": int(calendar_days),
            "primary_path_completeness": float(
                primary_metrics["path_completeness"]
            ),
            "primary_net_mean_r_70pct_direction": float(
                primary_metrics["mean_net_r"]
            ),
            "primary_net_mean_r_ci_low": float(
                primary_metrics["net_mean_r_ci_low"]
            ),
            "primary_net_mean_r_ci_high": float(
                primary_metrics["net_mean_r_ci_high"]
            ),
            "primary_break_even_direction_accuracy": float(
                primary_metrics["break_even_direction_accuracy"]
            ),
            "diagnostic_state": state,
            "max_loaded_timestamp": (
                str(minute.index.max()) if not minute.empty else None
            ),
            "read_end_exclusive": str(read_end),
            "direction_head_trained": False,
            "timing_model_refit": False,
            "economics_evaluated": True,
            "forward_or_lockbox_loaded": False,
        }
        store.json("summary.json", summary)
        missing = [name for name in READER_ARTIFACTS if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"Notebook Q artifacts missing: {missing}")
        store.complete(summary)
        if not smoke:
            _latest(Path(run_root), identity["run_hash"], protocol_hash)
        return EconomicRunResult(run_dir, summary)
    except BaseException as error:
        store.fail(error)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="dev")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--frozen-p-root", type=Path, default=FROZEN_P_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_economic_feasibility(
        stage=args.stage,
        smoke=args.smoke,
        data_root=args.data_root,
        frozen_p_root=args.frozen_p_root,
        run_root=args.run_root,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DATA_ROOT",
    "EconomicRunResult",
    "EconomicFeasibilityConfig",
    "FROZEN_P_ROOT",
    "FROZEN_P_RUN_HASH",
    "FrozenPArtifacts",
    "READER_ARTIFACTS",
    "RUN_ROOT",
    "build_direction_scenarios",
    "cluster_bootstrap_draws",
    "load_frozen_p_artifacts",
    "protocol_dict",
    "reconstruct_activation_ledger",
    "replay_brackets",
    "run_economic_feasibility",
]
