"""Run Notebook M: adaptive large-move LogReg versus multiclass XGBoost."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.event_window_large_move_policy import (
    LargeMovePolicyConfig,
    policy_summary,
    replay_large_move_policy,
)
from experiments.event_window_large_move_dataset import (
    AdaptiveMoveConfig,
    build_large_move_dataset,
    label_adaptive_large_moves,
)
from experiments.event_window_large_move_models import LargeMoveModelConfig
from experiments.event_window_large_move_oof import (
    LargeMoveOOFConfig,
    assert_identical_large_move_keys,
    run_large_move_model_oof,
)
from experiments.run_event_window_cost_aware_entry import (
    _Store,
    _jsonable,
    _sha256,
    _sha_payload,
)
from experiments.run_event_window_tail_models import (
    FROZEN_J_ROOT,
    _build_tail_dataset,
    load_frozen_j_artifacts,
)
from experiments.event_window_tail_oof import _half_open_uniqueness


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = CODE_ROOT / "data"
RUN_ROOT = CODE_ROOT / "experiments" / "cache" / "event_window_adaptive_large_move"
MODELS = ("logreg", "xgboost")
READER_ARTIFACTS = (
    "adaptive_labels.parquet",
    "feature_audit.csv",
    "label_audit.csv",
    "fold_audit.csv",
    "calibration_audit.csv",
    "threshold_frontier.csv",
    "oof_predictions.parquet",
    "selected_entries.parquet",
    "policy_results.csv",
    "episode_bootstrap.csv",
    "paired_comparison.csv",
    "protocol.json",
    "frozen_protocol.json",
    "summary.json",
)


@dataclass(frozen=True)
class AdaptiveMoveStudyConfig:
    development_start: str = "2021-01-01"
    development_end_exclusive: str = "2025-07-01"
    minimum_trades_per_day: float = 1.0
    bootstrap_draws: int = 2_000
    bootstrap_seed: int = 42
    target: AdaptiveMoveConfig = field(default_factory=AdaptiveMoveConfig)
    policy: LargeMovePolicyConfig = field(default_factory=LargeMovePolicyConfig)
    model: LargeMoveModelConfig = field(default_factory=LargeMoveModelConfig)


@dataclass(frozen=True)
class AdaptiveMoveRunResult:
    run_dir: Path
    summary: dict[str, object]


def _source_hash() -> str:
    paths = (
        CODE_ROOT / "data" / "load.py",
        CODE_ROOT / "features" / "event_window_inputs.py",
        CODE_ROOT / "features" / "event_windows.py",
        CODE_ROOT / "features" / "linear_channels.py",
        CODE_ROOT / "evaluation" / "channel_window_validation.py",
        CODE_ROOT / "experiments" / "event_window_dataset.py",
        CODE_ROOT / "experiments" / "event_window_tail_dataset.py",
        CODE_ROOT / "experiments" / "event_window_tail_oof.py",
        CODE_ROOT / "experiments" / "event_window_cost_aware_oof.py",
        CODE_ROOT / "experiments" / "event_window_large_move_dataset.py",
        CODE_ROOT / "experiments" / "event_window_large_move_models.py",
        CODE_ROOT / "experiments" / "event_window_large_move_oof.py",
        CODE_ROOT / "evaluation" / "event_window_large_move_policy.py",
        CODE_ROOT / "experiments" / "run_event_window_tcn.py",
        CODE_ROOT / "experiments" / "run_event_window_tail_models.py",
        CODE_ROOT / "experiments" / "run_event_window_cost_aware_entry.py",
        Path(__file__),
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(CODE_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def protocol_dict(
    config: AdaptiveMoveStudyConfig = AdaptiveMoveStudyConfig(), *, smoke: bool = False
) -> dict[str, object]:
    return {
        "stage": "dev",
        "smoke": bool(smoke),
        "development_start": config.development_start,
        "development_end_exclusive": config.development_end_exclusive,
        "models": list(MODELS),
        "target_classes": ["NO_BIG_MOVE", "UP_BIG", "DOWN_BIG"],
        "adaptive_barrier": (
            f"min({config.target.maximum_barrier_bps:g}, "
            f"{config.target.minimum_barrier_bps:g} + "
            f"{config.target.volatility_addon_weight:g} * sigma_5m_12 * "
            f"sqrt({config.target.horizon_minutes / 5.0:g}) * 10000) bps"
        ),
        "barrier_reference": "decision-time one-minute Open; volatility uses only completed history",
        "horizon_minutes": config.target.horizon_minutes,
        "decision_cadence": "5min",
        "execution_cadence": "1min",
        "entry": "decision-time one-minute Open; target exit maker, other exits taker",
        "target_cost_bps": config.target.target_cost_bps,
        "other_cost_bps": config.target.other_cost_bps,
        "selection": "first direction passing a positive neutral-timeout EV proxy and the fold-local calibration threshold",
        "ev_proxy": "NO_BIG gross return is assumed zero for selection; realised timeout return is used only for evaluation",
        "natural_policy": "first positive-EV direction without a frequency threshold",
        "desired_trades_per_day": [
            config.policy.desired_trades_per_day_low,
            config.policy.desired_trades_per_day_high,
        ],
        "minimum_trades_per_day": config.minimum_trades_per_day,
        "predictive_gate": "outer episode-bootstrap lower bounds: selected BIG lift > 1 and direction accuracy on BIG > 0.5; exact correct-direction rate >= 70%",
        "exact_direction_target": config.policy.minimum_exact_direction_accuracy,
        "max_trades_per_window": 1,
        "cross_window_capacity": "unlimited",
        "feature_set": "Notebook L base features plus causal barrier and raw directional counterparts",
        "old_oof_selected_trades_reused": False,
        "bootstrap_unit": "channel_episode_id including zero-trade episodes",
        "risk_unit": "adaptive barrier equals 1R; net bps is also reported for equal-notional interpretation",
        "adaptive_constants_status": "pre-registered before the full M run; not tuned after outcomes",
        "volatility_estimator": "sample standard deviation (ddof=1) of configured completed 5m returns",
        "forward_or_lockbox_loaded": False,
        "target_config": asdict(config.target),
        "policy_config": asdict(config.policy),
        "model_config": asdict(config.model),
    }


def _latest(run_root: Path, run_hash: str, protocol_hash: str) -> None:
    value = {
        "run_hash": run_hash,
        "protocol_hash": protocol_hash,
        "relative_path": f"{run_hash}/full",
    }
    path = run_root / "latest_dev.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _validated_completed_summary(
    run_dir: Path,
    identity: dict[str, str],
    *,
    frozen_j_run_hash: str,
    frozen_manifest_hash: str,
    frozen_input_hash: str,
) -> dict[str, object] | None:
    """Return a completed summary only when every published byte is verified."""
    try:
        state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
        if state.get("status") != "complete":
            return None
        if any(state.get(name) != value for name, value in identity.items()):
            return None
        records = state.get("artifacts")
        if not isinstance(records, dict):
            return None
        for name in READER_ARTIFACTS:
            path = run_dir / name
            record = records.get(name)
            if not path.is_file() or not isinstance(record, dict):
                return None
            if int(record.get("size", -1)) != path.stat().st_size:
                return None
            if str(record.get("sha256", "")) != _sha256(path):
                return None

        protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
        frozen_protocol = json.loads(
            (run_dir / "frozen_protocol.json").read_text(encoding="utf-8")
        )
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if any(protocol.get(name) != value for name, value in identity.items()):
            return None
        if any(summary.get(name) != value for name, value in identity.items()):
            return None
        if state.get("summary") != summary:
            return None
        expected_frozen = {
            "frozen_j_run_hash": frozen_j_run_hash,
            "manifest_hash": frozen_manifest_hash,
            "frozen_input_hash": frozen_input_hash,
            "old_oof_selected_trades_reused": False,
        }
        if any(
            frozen_protocol.get(name) != value
            for name, value in expected_frozen.items()
        ):
            return None
        if frozen_protocol.get("labels_hash") != _sha256(
            run_dir / "adaptive_labels.parquet"
        ):
            return None
        return summary
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _feature_audit(dataset) -> pd.DataFrame:
    kept = pd.DataFrame(
        {
            "feature": dataset.tabular_features,
            "kept": True,
            "reason": "retained causal base/adaptive feature",
        }
    )
    dropped = pd.DataFrame(
        {
            "feature": dataset.dropped_features,
            "kept": False,
            "reason": "algebraic duplicate or constant mask",
        }
    )
    return pd.concat([kept, dropped], ignore_index=True)


def _episode_bootstrap(
    selected: pd.DataFrame,
    *,
    episode_universe: pd.Index,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, float, float]:
    if selected.empty:
        values = np.zeros(len(episode_universe), dtype=float)
    else:
        values = (
            selected.groupby("channel_episode_id")["net_r"]
            .sum()
            .reindex(episode_universe, fill_value=0.0)
            .to_numpy(float)
        )
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [values[rng.integers(0, len(values), len(values))].sum() for _ in range(draws)]
    )
    low, high = np.quantile(samples, [0.025, 0.975])
    return samples, float(low), float(high)


def _paired_bootstrap(
    selected: pd.DataFrame,
    *,
    episode_universe: pd.Index,
    draws: int,
    seed: int,
) -> dict[str, float | str]:
    by_model = {}
    for model in MODELS:
        model_rows = selected.loc[selected["model"].eq(model)]
        by_model[model] = (
            np.zeros(len(episode_universe), dtype=float)
            if model_rows.empty
            else model_rows.groupby("channel_episode_id")["net_r"]
            .sum()
            .reindex(episode_universe, fill_value=0.0)
            .to_numpy(float)
        )
    delta = by_model["xgboost"] - by_model["logreg"]
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [delta[rng.integers(0, len(delta), len(delta))].sum() for _ in range(draws)]
    )
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "candidate": "xgboost",
        "baseline": "logreg",
        "delta_total_net_r": float(delta.sum()),
        "delta_ci_low": float(low),
        "delta_ci_high": float(high),
    }


def _discrimination_base(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["window_id", "step"]
    scored = scores.merge(
        labels[keys + ["move_code", "label_start", "label_end"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if scored[["move_code", "label_start", "label_end"]].isna().any().any():
        raise ValueError("adaptive labels do not cover every OOF score")
    weights = _half_open_uniqueness(scored, np.arange(len(scored), dtype=np.int64))
    scored = scored.assign(
        _base_weight=weights,
        _base_big_weight=weights * scored["move_code"].ne(0).to_numpy(float),
    )
    return scored.groupby("channel_episode_id", sort=False).agg(
        base_weight=("_base_weight", "sum"),
        base_big_weight=("_base_big_weight", "sum"),
    )


def _discrimination_bootstrap(
    selected: pd.DataFrame,
    base: pd.DataFrame,
    *,
    episode_universe: pd.Index,
    draws: int,
    seed: int,
) -> dict[str, float]:
    if selected.empty:
        chosen = pd.DataFrame(
            columns=[
                "selected_count",
                "selected_big",
                "selected_correct_big",
                "selected_exact_direction",
            ]
        )
    else:
        chosen_work = selected.assign(
            _selected_count=1.0,
            _selected_big=selected["actual_big"].astype(float),
            _selected_correct_big=(
                selected["actual_big"].astype(bool)
                & selected["direction_correct"].astype(bool)
            ).astype(float),
            _selected_exact_direction=selected["direction_correct"].astype(float),
        )
        chosen = chosen_work.groupby("channel_episode_id", sort=False).agg(
            selected_count=("_selected_count", "sum"),
            selected_big=("_selected_big", "sum"),
            selected_correct_big=("_selected_correct_big", "sum"),
            selected_exact_direction=("_selected_exact_direction", "sum"),
        )
    episode = pd.DataFrame(index=episode_universe).join(base).join(chosen).fillna(0.0)
    values = episode[
        [
            "base_weight",
            "base_big_weight",
            "selected_count",
            "selected_big",
            "selected_correct_big",
            "selected_exact_direction",
        ]
    ].to_numpy(float)

    def metrics(sample: np.ndarray) -> tuple[float, float, float, float]:
        totals = sample.sum(axis=0)
        base_rate = totals[1] / totals[0] if totals[0] > 0 else np.nan
        selected_rate = totals[3] / totals[2] if totals[2] > 0 else np.nan
        lift = selected_rate / base_rate if base_rate > 0 else np.nan
        direction = totals[4] / totals[3] if totals[3] > 0 else np.nan
        exact = totals[5] / totals[2] if totals[2] > 0 else np.nan
        return float(base_rate), float(lift), float(direction), float(exact)

    base_rate, lift, direction, exact = metrics(values)
    rng = np.random.default_rng(seed)
    sampled = np.asarray(
        [metrics(values[rng.integers(0, len(values), len(values))]) for _ in range(draws)]
    )

    def interval(column: int) -> tuple[float, float]:
        finite = sampled[:, column][np.isfinite(sampled[:, column])]
        if not len(finite):
            return np.nan, np.nan
        low, high = np.quantile(finite, [0.025, 0.975])
        return float(low), float(high)

    lift_low, lift_high = interval(1)
    direction_low, direction_high = interval(2)
    exact_low, exact_high = interval(3)
    return {
        "base_big_prevalence": base_rate,
        "selected_big_move_lift": lift,
        "selected_big_move_lift_ci_low": lift_low,
        "selected_big_move_lift_ci_high": lift_high,
        "direction_accuracy_on_big": direction,
        "direction_accuracy_on_big_ci_low": direction_low,
        "direction_accuracy_on_big_ci_high": direction_high,
        "selected_exact_direction_accuracy": exact,
        "selected_exact_direction_accuracy_ci_low": exact_low,
        "selected_exact_direction_accuracy_ci_high": exact_high,
    }


def _policy_rows(results, labels, config, *, smoke: bool):
    selected_frames: list[pd.DataFrame] = []
    policy_rows: list[dict[str, object]] = []
    bootstrap_rows: list[pd.DataFrame] = []
    episode_universe = pd.Index(
        pd.concat([result.scores for result in results])["channel_episode_id"].unique()
    )
    draws = 20 if smoke else config.bootstrap_draws
    for result in results:
        discrimination_base = _discrimination_base(result.scores, labels)
        for policy in ("natural", "calibrated"):
            selected = replay_large_move_policy(
                result.scores,
                labels,
                threshold=0.0 if policy == "natural" else None,
                threshold_column=(
                    None if policy == "natural" else "calibration_threshold"
                ),
                config=config.policy,
                execution=config.target,
            )
            selected["model"] = result.model_name
            selected.insert(0, "policy", policy)
            selected_frames.append(selected)
            summary = policy_summary(selected, result.scores)
            discrimination = _discrimination_bootstrap(
                selected,
                discrimination_base,
                episode_universe=episode_universe,
                draws=draws,
                seed=config.bootstrap_seed + (0 if policy == "natural" else 10),
            )
            samples, low, high = _episode_bootstrap(
                selected,
                episode_universe=episode_universe,
                draws=draws,
                seed=config.bootstrap_seed,
            )
            bootstrap_rows.append(
                pd.DataFrame(
                    {
                        "model": result.model_name,
                        "policy": policy,
                        "draw": np.arange(draws),
                        "total_net_r": samples,
                    }
                )
            )
            policy_rows.append(
                {
                    "model": result.model_name,
                    "policy": policy,
                    **summary,
                    **discrimination,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "frequency_pass": config.minimum_trades_per_day
                    <= summary["trades_per_day"]
                    <= config.policy.desired_trades_per_day_high,
                    "economic_pass": summary["total_net_r"] > 0.0 and low > 0.0,
                    "predictive_pass": (
                        discrimination["selected_big_move_lift_ci_low"] > 1.0
                        and discrimination["direction_accuracy_on_big_ci_low"] > 0.5
                        and discrimination["selected_exact_direction_accuracy"]
                        >= config.policy.minimum_exact_direction_accuracy
                    ),
                }
            )
    selected_all = pd.concat(selected_frames, ignore_index=True)
    policies = pd.DataFrame(policy_rows)
    bootstraps = pd.concat(bootstrap_rows, ignore_index=True)
    registered_selected = selected_all.loc[selected_all["policy"].eq("calibrated")]
    paired = _paired_bootstrap(
        registered_selected,
        episode_universe=episode_universe,
        draws=draws,
        seed=config.bootstrap_seed + 1,
    )
    return selected_all, policies, bootstraps, paired


def _choose_model(
    registered: pd.DataFrame, paired: dict[str, float | str]
) -> str | None:
    passing = registered.index[
        registered["frequency_pass"]
        & registered["economic_pass"]
        & registered["predictive_pass"]
    ].tolist()
    xgboost_pass = "xgboost" in passing and float(paired["delta_ci_low"]) > 0.0
    logreg_pass = "logreg" in passing
    return "xgboost" if xgboost_pass else "logreg" if logreg_pass else None


def run_adaptive_large_move_study(
    *,
    stage: str = "dev",
    smoke: bool = False,
    data_root: Path = DEFAULT_DATA_ROOT,
    frozen_j_root: Path = FROZEN_J_ROOT,
    run_root: Path = RUN_ROOT,
    config: AdaptiveMoveStudyConfig = AdaptiveMoveStudyConfig(),
) -> AdaptiveMoveRunResult:
    if stage != "dev":
        raise ValueError("Notebook M permits development only; forward and Q2 are sealed")
    protocol = protocol_dict(config, smoke=smoke)
    protocol_hash = _sha_payload(protocol)
    frozen = load_frozen_j_artifacts(Path(frozen_j_root))
    source_hash = _source_hash()
    input_hash = _sha_payload(
        {
            "frozen_j_run_hash": frozen.run_hash,
            "frozen_manifest": frozen.manifest_sha256,
            "frozen_input": frozen.input_hash,
        }
    )
    run_hash = _sha_payload(
        {
            "protocol_hash": protocol_hash,
            "source_hash": source_hash,
            "input_hash": input_hash,
        }
    )[:20]
    run_dir = Path(run_root) / run_hash / ("smoke" if smoke else "full")
    identity = {
        "run_hash": run_hash,
        "protocol_hash": protocol_hash,
        "source_hash": source_hash,
        "input_hash": input_hash,
    }
    cached = _validated_completed_summary(
        run_dir,
        identity,
        frozen_j_run_hash=frozen.run_hash,
        frozen_manifest_hash=frozen.manifest_sha256,
        frozen_input_hash=frozen.input_hash,
    )
    if cached is not None:
        if not smoke:
            _latest(Path(run_root), run_hash, protocol_hash)
        return AdaptiveMoveRunResult(run_dir, cached)
    store = _Store(run_dir, identity)
    try:
        store.json("protocol.json", {**protocol, **identity})
        base, _, loaded = _build_tail_dataset(
            frozen, data_root=Path(data_root), smoke=smoke
        )
        expected_start = pd.Timestamp(config.development_start, tz="UTC")
        expected_end = pd.Timestamp(config.development_end_exclusive, tz="UTC")
        if not smoke and (
            loaded.read_start != expected_start
            or loaded.read_end_exclusive != expected_end
            or loaded.max_loaded_timestamp >= expected_end
        ):
            raise AssertionError("Notebook M bounded development inputs changed")
        labels = label_adaptive_large_moves(base.decisions, loaded.minute, config.target)
        dataset = build_large_move_dataset(base, labels, feature_set="directional")
        store.parquet("adaptive_labels.parquet", labels)
        label_audit = (
            labels.groupby(["move_label", "model_target_valid"], dropna=False)
            .size()
            .rename("rows")
            .reset_index()
        )
        store.csv("label_audit.csv", label_audit)
        store.csv("feature_audit.csv", _feature_audit(dataset))
        model_config = (
            replace(config.model, xgb_estimators=12) if smoke else config.model
        )
        oof_config = LargeMoveOOFConfig(
            model=model_config, policy=config.policy, target=config.target
        )
        results = [
            run_large_move_model_oof(model, dataset, oof_config) for model in MODELS
        ]
        assert_identical_large_move_keys(results)
        scores = pd.concat([result.scores for result in results], ignore_index=True)
        fold_audit = pd.concat(
            [result.fold_audit for result in results], ignore_index=True
        )
        calibration_audit = pd.concat(
            [result.calibration_audit for result in results], ignore_index=True
        )
        frontier = pd.concat(
            [result.threshold_frontier for result in results], ignore_index=True
        )
        if not fold_audit["episode_overlap"].eq(0).all():
            raise AssertionError("Notebook M leakage audit failed")
        store.parquet("oof_predictions.parquet", scores)
        store.csv("fold_audit.csv", fold_audit)
        store.csv("calibration_audit.csv", calibration_audit)
        store.csv("threshold_frontier.csv", frontier)
        selected, policies, bootstraps, paired = _policy_rows(
            results, labels, config, smoke=smoke
        )
        store.parquet("selected_entries.parquet", selected)
        store.csv("policy_results.csv", policies)
        store.csv("episode_bootstrap.csv", bootstraps)
        store.csv("paired_comparison.csv", pd.DataFrame([paired]))
        store.json(
            "frozen_protocol.json",
            {
                "source": "Notebook J immutable event-window manifest; Notebook M rebuilt adaptive labels",
                "manifest_hash": frozen.manifest_sha256,
                "frozen_input_hash": frozen.input_hash,
                "labels_hash": _sha256(run_dir / "adaptive_labels.parquet"),
                "frozen_j_run_hash": frozen.run_hash,
                "old_oof_selected_trades_reused": False,
            },
        )
        registered = policies.loc[policies["policy"].eq("calibrated")].set_index("model")
        chosen = _choose_model(registered, paired)
        decision = (
            f"promote {chosen} on development evidence" if chosen else
            "no model passed registered economics, frequency and predictive discrimination"
        )
        summary = {
            **identity,
            "models": list(MODELS),
            "decision": decision,
            "chosen_model": chosen,
            "adaptive_label_rows": len(labels),
            "valid_label_rows": int(labels["model_target_valid"].sum()),
            "oof_rows_per_model": {
                result.model_name: len(result.scores) for result in results
            },
            "retained_features": len(dataset.tabular_features),
            "dropped_features": len(dataset.dropped_features),
            "model_results": registered.to_dict(orient="index"),
            "paired_xgboost_minus_logreg": paired,
            "max_loaded_timestamp": loaded.max_loaded_timestamp,
            "read_end_exclusive": loaded.read_end_exclusive,
            "forward_or_lockbox_loaded": False,
        }
        store.json("summary.json", summary)
        missing = [name for name in READER_ARTIFACTS if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"Notebook M artifacts missing: {missing}")
        store.complete(summary)
        if not smoke:
            _latest(Path(run_root), run_hash, protocol_hash)
        return AdaptiveMoveRunResult(run_dir, summary)
    except BaseException as error:
        store.fail(error)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="dev")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--frozen-j-root", type=Path, default=FROZEN_J_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_adaptive_large_move_study(
        stage=args.stage,
        smoke=args.smoke,
        data_root=args.data_root,
        frozen_j_root=args.frozen_j_root,
        run_root=args.run_root,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True, default=_jsonable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AdaptiveMoveStudyConfig",
    "AdaptiveMoveRunResult",
    "protocol_dict",
    "run_adaptive_large_move_study",
]
