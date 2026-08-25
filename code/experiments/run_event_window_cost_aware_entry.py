"""Run the development-only Notebook L cost-aware entry experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.event_window_cost_aware_policy import (
    CostAwarePolicyConfig,
    replay_cost_aware_policy,
)
from experiments.event_window_cost_aware_dataset import (
    MakerExecutionConfig,
    build_cost_aware_dataset,
    label_maker_window_steps,
)
from experiments.event_window_cost_aware_oof import (
    CostAwareOOFConfig,
    assert_identical_cost_aware_keys,
    run_cost_aware_model_oof,
)
from experiments.run_event_window_tail_models import (
    FROZEN_J_ROOT,
    _build_tail_dataset,
    load_frozen_j_artifacts,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = CODE_ROOT / "data"
RUN_ROOT = CODE_ROOT / "experiments" / "cache" / "event_window_cost_aware_entry"
DEV_END = pd.Timestamp("2025-07-01", tz="UTC")
MODELS = ("logreg", "xgboost")
ACTIONS = ("ENTER", "WAIT", "SKIP")
READER_ARTIFACTS = (
    "feature_audit.csv",
    "fold_audit.csv",
    "oof_predictions.parquet",
    "action_audit.csv",
    "policy_results.csv",
    "calibration_audit.csv",
    "selected_entries.parquet",
    "protocol.json",
    "frozen_protocol.json",
    "summary.json",
)
EXTRA_ARTIFACTS = (
    "maker_labels.parquet",
    "label_audit.csv",
    "daily_frequency.csv",
    "paired_comparison.csv",
    "episode_bootstrap.csv",
)


@dataclass(frozen=True)
class CostAwareStudyConfig:
    development_start: str = "2021-01-01"
    development_end_exclusive: str = "2025-07-01"
    confidence_z: float = 1.645
    minimum_filled_trades_per_day: float = 1.0
    bootstrap_draws: int = 2_000
    bootstrap_seed: int = 42
    execution: MakerExecutionConfig = MakerExecutionConfig()


@dataclass(frozen=True)
class CostAwareRunResult:
    run_dir: Path
    summary: dict[str, object]


def _jsonable(value: object) -> object:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_jsonable).encode()


def _sha_payload(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hash() -> str:
    paths = (
        CODE_ROOT / "experiments" / "event_window_cost_aware_dataset.py",
        CODE_ROOT / "experiments" / "event_window_cost_aware_models.py",
        CODE_ROOT / "experiments" / "event_window_cost_aware_oof.py",
        CODE_ROOT / "evaluation" / "event_window_cost_aware_policy.py",
        CODE_ROOT / "evaluation" / "channel_backtest.py",
        CODE_ROOT / "features" / "event_window_inputs.py",
        CODE_ROOT / "features" / "event_windows.py",
        CODE_ROOT / "features" / "linear_channels.py",
        CODE_ROOT / "experiments" / "event_window_dataset.py",
        CODE_ROOT / "experiments" / "event_window_tail_dataset.py",
        CODE_ROOT / "experiments" / "run_event_window_tail_models.py",
        CODE_ROOT / "experiments" / "run_event_window_tcn.py",
        Path(__file__),
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def protocol_dict(config: CostAwareStudyConfig, *, smoke: bool) -> dict[str, object]:
    return {
        "stage": "dev",
        "smoke": bool(smoke),
        "development_start": config.development_start,
        "development_end_exclusive": config.development_end_exclusive,
        "models": list(MODELS),
        "actions": list(ACTIONS),
        "execution": asdict(config.execution),
        "selection_rule": "first conservative net EV >= 0 and enter advantage >= 0",
        "ev_calibration": "fold-local mean residual plus one-sided 90% episode SE",
        "confidence_z": config.confidence_z,
        "minimum_filled_trades_per_day": config.minimum_filled_trades_per_day,
        "max_orders_per_window": 1,
        "cross_window_capacity": "unlimited",
        "fee_sensitivity_grid": False,
        "old_oof_selected_trades_reused": False,
        "fill_assumption": "full fill only after 1 bps trade-through beyond the resting limit",
        "censoring_rule": "exclude incomplete windows before model policy replay",
        "costs_fit_inside_training": True,
        "forward_or_lockbox_loaded": False,
    }


class _Store:
    def __init__(self, run_dir: Path, identity: dict[str, str]) -> None:
        self.run_dir = run_dir
        self.state_path = run_dir / "run_state.json"
        self.state: dict[str, object] = {**identity, "status": "running", "artifacts": {}}
        run_dir.mkdir(parents=True, exist_ok=True)
        self._flush()

    def _flush(self) -> None:
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True, default=_jsonable), encoding="utf-8")
        temporary.replace(self.state_path)

    def _record(self, name: str) -> None:
        path = self.run_dir / name
        self.state["artifacts"][name] = {"sha256": _sha256(path), "size": path.stat().st_size}
        self._flush()

    def json(self, name: str, value: object) -> None:
        path = self.run_dir / name
        path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_jsonable), encoding="utf-8")
        self._record(name)

    def csv(self, name: str, frame: pd.DataFrame) -> None:
        frame.to_csv(self.run_dir / name, index=False)
        self._record(name)

    def parquet(self, name: str, frame: pd.DataFrame) -> None:
        frame.to_parquet(self.run_dir / name, index=False)
        self._record(name)

    def complete(self, summary: dict[str, object]) -> None:
        self.state["status"] = "complete"
        self.state["summary"] = summary
        self._flush()

    def fail(self, error: BaseException) -> None:
        self.state["status"] = "failed"
        self.state["error"] = repr(error)
        self._flush()


def _calendar(scores: pd.DataFrame) -> pd.DatetimeIndex:
    times = pd.to_datetime(scores["decision_time"], utc=True)
    start = times.min().normalize()
    end = times.max().normalize()
    return pd.date_range(start, end, freq="D", tz="UTC")


def _episode_bootstrap(
    selected: pd.DataFrame,
    *,
    episode_universe: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, float, float]:
    universe = pd.Index(np.asarray(episode_universe), dtype=object).drop_duplicates()
    filled = selected.loc[selected["outcome"].isin(["sl", "tp", "timeout"])].copy()
    episode = (
        filled.groupby("channel_episode_id")["realized_net_r"].sum()
        .reindex(universe, fill_value=0.0)
        .to_numpy(float)
    )
    rng = np.random.default_rng(seed)
    values = np.array(
        [episode[rng.integers(0, len(episode), len(episode))].sum() for _ in range(draws)]
    )
    low, high = np.quantile(values, [0.025, 0.975])
    return pd.DataFrame({"draw": np.arange(draws), "total_net_r": values}), float(low), float(high)


def _paired_bootstrap(
    selected: pd.DataFrame,
    *,
    episode_universe: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    ledger = selected.copy()
    ledger["value"] = np.where(
        ledger["outcome"].isin(["sl", "tp", "timeout"]),
        ledger["realized_net_r"],
        0.0,
    )
    universe = pd.Index(np.asarray(episode_universe), dtype=object).drop_duplicates()
    by_model = {
        model: (
            ledger.loc[ledger["model"].eq(model)]
            .groupby("channel_episode_id")["value"]
            .sum()
            .reindex(universe, fill_value=0.0)
            .to_numpy(float)
        )
        for model in MODELS
    }
    by_episode = by_model["xgboost"] - by_model["logreg"]
    delta = float(by_episode.sum()) if len(by_episode) else 0.0
    rng = np.random.default_rng(seed)
    draws_values = (
        np.array([by_episode[rng.integers(0, len(by_episode), len(by_episode))].sum() for _ in range(draws)])
        if len(by_episode)
        else np.zeros(draws)
    )
    low, high = np.quantile(draws_values, [0.025, 0.975])
    return delta, float(low), float(high)


def run_cost_aware_entry_study(
    *,
    stage: str = "dev",
    config: CostAwareStudyConfig = CostAwareStudyConfig(),
    data_root: Path = DEFAULT_DATA_ROOT,
    run_root: Path = RUN_ROOT,
    frozen_j_root: Path = FROZEN_J_ROOT,
    smoke: bool = False,
) -> CostAwareRunResult:
    if stage != "dev":
        raise PermissionError("Notebook L is development-only; forward and Q2 remain sealed")
    if config.development_end_exclusive != "2025-07-01":
        raise ValueError("the development boundary is frozen")
    frozen = load_frozen_j_artifacts(Path(frozen_j_root))
    protocol = protocol_dict(config, smoke=smoke)
    protocol_hash = _sha_payload(protocol)
    source_hash = _source_hash()
    run_hash = _sha_payload(
        {
            "protocol_hash": protocol_hash,
            "source_hash": source_hash,
            "input_hash": frozen.input_hash,
            "manifest": frozen.manifest_sha256,
        }
    )[:20]
    mode = "smoke" if smoke else "full"
    output_root = Path(run_root)
    run_dir = output_root / run_hash / mode
    store = _Store(
        run_dir,
        {
            "run_hash": run_hash,
            "protocol_hash": protocol_hash,
            "input_hash": frozen.input_hash,
            "source_hash": source_hash,
        },
    )
    protocol_artifact = {
        **protocol,
        "run_hash": run_hash,
        "protocol_hash": protocol_hash,
        "source_hash": source_hash,
        "input_hash": frozen.input_hash,
    }
    store.json("protocol.json", protocol_artifact)
    try:
        base, _, loaded = _build_tail_dataset(
            frozen,
            data_root=Path(data_root),
            smoke=smoke,
        )
        if loaded.max_loaded_timestamp >= DEV_END:
            raise AssertionError("Notebook L crossed the development boundary")
        maker_labels = label_maker_window_steps(
            base.sequences,
            loaded.five_minute,
            loaded.minute,
            config.execution,
        )
        dataset = build_cost_aware_dataset(base, maker_labels, config.execution)
        store.parquet("maker_labels.parquet", maker_labels)
        label_audit = (
            maker_labels.groupby(["outcome", "filled", "model_target_valid"], dropna=False)
            .size()
            .rename("rows")
            .reset_index()
        )
        store.csv("label_audit.csv", label_audit)

        original_features = set(base.tabular_features)
        dropped = set(dataset.dropped_features)
        feature_rows = [
            {
                "feature": name,
                "kept": name not in dropped,
                "reason": "algebraic duplicate or constant mask" if name in dropped else "retained causal feature",
            }
            for name in base.tabular_features
        ]
        feature_rows.extend(
            {"feature": name, "kept": True, "reason": "registered execution-cost feature"}
            for name in dataset.tabular_features
            if name not in original_features
        )
        store.csv("feature_audit.csv", pd.DataFrame(feature_rows))

        oof_config = CostAwareOOFConfig(
            execution=config.execution,
            confidence_z=config.confidence_z,
        )
        if smoke:
            oof_config = replace(
                oof_config,
                model=replace(oof_config.model, xgb_estimators=20),
            )
        results = [run_cost_aware_model_oof(model, dataset, oof_config) for model in MODELS]
        assert_identical_cost_aware_keys(results)
        fold_audit = pd.concat([result.fold_audit for result in results], ignore_index=True)
        if not fold_audit["episode_overlap"].eq(0).all():
            raise AssertionError("Notebook L episode leakage audit failed")
        calibration_audit = pd.concat(
            [result.calibration_audit for result in results], ignore_index=True
        )
        store.csv("fold_audit.csv", fold_audit)
        store.csv("calibration_audit.csv", calibration_audit)

        policy_config = CostAwarePolicyConfig(
            threshold=0.0,
            maker_entry_bps=config.execution.maker_entry_bps,
            maker_tp_exit_bps=config.execution.maker_tp_exit_bps,
            taker_sl_exit_bps=config.execution.taker_sl_exit_bps,
            timeout_exit_bps=config.execution.timeout_exit_bps,
        )
        prediction_frames: list[pd.DataFrame] = []
        selected_frames: list[pd.DataFrame] = []
        action_frames: list[pd.DataFrame] = []
        policy_rows: list[dict[str, object]] = []
        bootstrap_frames: list[pd.DataFrame] = []
        daily_frames: list[pd.DataFrame] = []
        common_keys = results[0].scores[["window_id", "step"]]
        common_labels = common_keys.merge(
            dataset.decisions,
            on=["window_id", "step"],
            how="left",
            validate="one_to_one",
        )
        common_labels["incomplete_path"] = (
            common_labels["geometry_valid"].astype(bool)
            & ~common_labels["path_observed"].astype(bool)
        )
        incomplete_window = common_labels.groupby("window_id", sort=False)["incomplete_path"].any()
        evaluation_windows = set(incomplete_window[~incomplete_window].index)
        excluded_censored_windows = int(incomplete_window.sum())
        evaluation_labels = common_labels.loc[
            common_labels["window_id"].isin(evaluation_windows)
        ].copy()
        episode_universe = evaluation_labels["channel_episode_id"].drop_duplicates().to_numpy()
        for result in results:
            evaluation_scores = result.scores.loc[
                result.scores["window_id"].isin(evaluation_windows)
            ].copy()
            replay = replay_cost_aware_policy(evaluation_scores, evaluation_labels, policy_config)
            selected = replay.trades.copy()
            if len(selected) and (~selected["path_observed"].astype(bool)).any():
                raise AssertionError("censoring-safe evaluation selected an incomplete path")
            selected.insert(0, "model", result.model_name)
            actions = replay.actions.copy()
            actions.insert(0, "model", result.model_name)
            predictions = evaluation_scores.merge(
                actions[["window_id", "step", "action"]],
                on=["window_id", "step"],
                how="left",
            )
            predictions["action"] = predictions["action"].fillna("SKIP")
            prediction_frames.append(predictions)
            selected_frames.append(selected)
            action_frames.append(actions)
            calendar = _calendar(evaluation_scores)
            decision_day = pd.to_datetime(selected["decision_time"], utc=True).dt.normalize() if len(selected) else pd.Series(dtype="datetime64[ns, UTC]")
            filled = selected["outcome"].isin(["sl", "tp", "timeout"]) if len(selected) else pd.Series(dtype=bool)
            entry_day = pd.to_datetime(selected.loc[filled, "entry_time"], utc=True).dt.normalize() if len(selected) else pd.Series(dtype="datetime64[ns, UTC]")
            daily = pd.DataFrame({"date": calendar})
            daily["model"] = result.model_name
            daily["orders"] = daily["date"].map(decision_day.value_counts()).fillna(0).astype(int)
            daily["filled_trades"] = daily["date"].map(entry_day.value_counts()).fillna(0).astype(int)
            if len(selected):
                net_by_day = selected.loc[filled].groupby(entry_day)["realized_net_r"].sum()
                daily["net_r"] = daily["date"].map(net_by_day).fillna(0.0)
            else:
                daily["net_r"] = 0.0
            daily_frames.append(daily)
            bootstrap, ci_low, ci_high = _episode_bootstrap(
                selected,
                episode_universe=episode_universe,
                draws=20 if smoke else config.bootstrap_draws,
                seed=config.bootstrap_seed,
            )
            bootstrap.insert(0, "model", result.model_name)
            bootstrap_frames.append(bootstrap)
            filled_rows = selected.loc[filled]
            realised = filled_rows["realized_net_r"].astype(float)
            filled_per_day = len(filled_rows) / len(calendar)
            policy_rows.append(
                {
                    "model": result.model_name,
                    "attempted_orders": len(selected),
                    "observed_orders": replay.observed_trades,
                    "unfilled_orders": int(selected["outcome"].eq("unfilled").sum()) if len(selected) else 0,
                    "filled_trades": len(filled_rows),
                    "filled_trades_per_day": filled_per_day,
                    "zero_trade_days": int(daily["filled_trades"].eq(0).sum()),
                    "mean_net_r": float(realised.mean()) if len(realised) else np.nan,
                    "total_net_r": float(realised.sum()) if len(realised) else 0.0,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "frequency_pass": bool(filled_per_day >= config.minimum_filled_trades_per_day),
                    "economic_pass": bool(len(realised) and realised.sum() > 0 and realised.mean() > 0 and ci_low > 0),
                }
            )
        predictions = pd.concat(prediction_frames, ignore_index=True)
        selected_entries = pd.concat(selected_frames, ignore_index=True)
        actions = pd.concat(action_frames, ignore_index=True)
        policy_results = pd.DataFrame(policy_rows)
        action_audit = actions.groupby(["model", "action"]).size().rename("count").reset_index()
        daily_frequency = pd.concat(daily_frames, ignore_index=True)
        episode_bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
        delta, delta_low, delta_high = _paired_bootstrap(
            selected_entries,
            episode_universe=episode_universe,
            draws=20 if smoke else config.bootstrap_draws,
            seed=config.bootstrap_seed,
        )
        paired = pd.DataFrame(
            [{
                "candidate": "xgboost",
                "baseline": "logreg",
                "delta_total_net_r": delta,
                "delta_ci_low": delta_low,
                "delta_ci_high": delta_high,
            }]
        )
        xgb = policy_results.set_index("model").loc["xgboost"]
        logreg = policy_results.set_index("model").loc["logreg"]
        xgb_pass = bool(xgb["frequency_pass"] and xgb["economic_pass"])
        logreg_pass = bool(logreg["frequency_pass"] and logreg["economic_pass"])
        chosen = (
            "xgboost"
            if xgb_pass and delta_low > 0
            else "logreg"
            if logreg_pass
            else None
        )
        decision = (
            f"promote {chosen} within development"
            if chosen
            else "no model passed both registered economics and frequency"
        )
        store.parquet("oof_predictions.parquet", predictions)
        store.csv("action_audit.csv", action_audit)
        store.csv("policy_results.csv", policy_results)
        store.parquet("selected_entries.parquet", selected_entries)
        store.csv("daily_frequency.csv", daily_frequency)
        store.csv("paired_comparison.csv", paired)
        store.csv("episode_bootstrap.csv", episode_bootstrap)
        frozen_protocol = {
            "source": "Notebook J immutable event-window manifest; Notebook L rebuilt maker labels",
            "manifest_hash": frozen.manifest_sha256,
            "labels_hash": _sha256(run_dir / "maker_labels.parquet"),
            "frozen_j_run_hash": frozen.run_hash,
            "old_oof_selected_trades_reused": False,
        }
        store.json("frozen_protocol.json", frozen_protocol)
        summary: dict[str, object] = {
            "run_hash": run_hash,
            "protocol_hash": protocol_hash,
            "models": list(MODELS),
            "forward_or_lockbox_loaded": False,
            "decision": decision,
            "chosen_model": chosen,
            "maker_label_rows": len(maker_labels),
            "evaluation_windows": len(evaluation_windows),
            "excluded_censored_windows": excluded_censored_windows,
            "oof_rows_per_model": {result.model_name: len(result.scores) for result in results},
            "dropped_features": len(dataset.dropped_features),
            "retained_features": len(dataset.tabular_features),
            "paired_xgboost_minus_logreg": paired.iloc[0].to_dict(),
            "model_results": policy_results.set_index("model").to_dict("index"),
            "read_end_exclusive": loaded.read_end_exclusive,
            "max_loaded_timestamp": loaded.max_loaded_timestamp,
        }
        store.json("summary.json", summary)
        missing = [name for name in (*READER_ARTIFACTS, *EXTRA_ARTIFACTS) if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"Notebook L artifacts missing: {missing}")
        store.complete(summary)
    except BaseException as error:
        store.fail(error)
        raise
    if not smoke:
        pointer = {
            "run_hash": run_hash,
            "relative_path": f"{run_hash}/full",
            "protocol_hash": protocol_hash,
        }
        (output_root / "latest_dev.json").write_text(
            json.dumps(pointer, indent=2, sort_keys=True), encoding="utf-8"
        )
    return CostAwareRunResult(run_dir, summary)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="dev")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_cost_aware_entry_study(stage=args.stage, smoke=args.smoke)
    print(result.run_dir)
    print(json.dumps(result.summary, indent=2, sort_keys=True, default=_jsonable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CostAwareRunResult",
    "CostAwareStudyConfig",
    "READER_ARTIFACTS",
    "main",
    "protocol_dict",
    "run_cost_aware_entry_study",
]
