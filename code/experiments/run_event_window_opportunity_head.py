"""Run Notebook N: one direction-free adaptive opportunity head."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.event_window_large_move_dataset import (
    AdaptiveMoveConfig,
    LargeMoveDecisionDataset,
    add_opportunity_target_contract,
    build_large_move_dataset,
    build_opportunity_dataset,
)
from experiments.event_window_large_move_models import LargeMoveModelConfig
from experiments.event_window_opportunity_oof import (
    OpportunityOOFConfig,
    OpportunityOOFResult,
    assert_identical_opportunity_keys,
    opportunity_metrics,
    run_opportunity_model_oof,
)
from experiments.run_event_window_adaptive_large_move import (
    READER_ARTIFACTS as M_READER_ARTIFACTS,
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


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = CODE_ROOT / "data"
FROZEN_M_ROOT = (
    CODE_ROOT / "experiments" / "cache" / "event_window_adaptive_large_move"
)
RUN_ROOT = CODE_ROOT / "experiments" / "cache" / "event_window_opportunity_head"
BASELINE_ARM = "N0_frozen_m"
TRAINED_ARMS = (
    ("N1_binary_legacy", "xgboost", "legacy"),
    ("N2_side_neutral", "xgboost", "side_neutral"),
    ("N3_side_neutral_volatility", "xgboost", "side_neutral_volatility"),
    ("N3_logreg", "logreg", "side_neutral_volatility"),
)
ALL_ARMS = (BASELINE_ARM, *(arm for arm, _, _ in TRAINED_ARMS))
PROMOTABLE_ARMS = (
    "N2_side_neutral",
    "N3_side_neutral_volatility",
    "N3_logreg",
)
READER_ARTIFACTS = (
    "opportunity_labels.parquet",
    "feature_audit.csv",
    "label_audit.csv",
    "fold_audit.csv",
    "calibration_audit.csv",
    "oof_predictions.parquet",
    "metrics.csv",
    "paired_bootstrap.csv",
    "reliability.csv",
    "matched_frequency.csv",
    "matched_decisions.parquet",
    "protocol.json",
    "frozen_protocol.json",
    "summary.json",
)


@dataclass(frozen=True)
class FrozenMArtifacts:
    run_hash: str
    protocol_hash: str
    input_hash: str
    manifest_hash: str
    frozen_j_run_hash: str
    labels_sha256: str
    oof_sha256: str
    labels: pd.DataFrame
    xgboost_scores: pd.DataFrame
    protocol: dict[str, object]
    summary: dict[str, object]


@dataclass(frozen=True)
class OpportunityStudyConfig:
    development_start: str = "2021-01-01"
    development_end_exclusive: str = "2025-07-01"
    matched_decisions_per_day: float = 1.1174628034455756
    bootstrap_draws: int = 500
    bootstrap_seed: int = 42
    target: AdaptiveMoveConfig = field(default_factory=AdaptiveMoveConfig)
    model: LargeMoveModelConfig = field(default_factory=LargeMoveModelConfig)


@dataclass(frozen=True)
class OpportunityRunResult:
    run_dir: Path
    summary: dict[str, object]


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path.name}")
    return value


def load_frozen_m_artifacts(run_root: Path = FROZEN_M_ROOT) -> FrozenMArtifacts:
    """Validate the completed development-only Notebook M handoff."""
    root = Path(run_root)
    pointer = _read_json(root / "latest_dev.json")
    run_hash = str(pointer.get("run_hash", ""))
    protocol_hash = str(pointer.get("protocol_hash", ""))
    expected_relative = f"{run_hash}/full"
    if not run_hash or pointer.get("relative_path") != expected_relative:
        raise ValueError("frozen Notebook M pointer is invalid")
    run_dir = (root / expected_relative).resolve()
    if not run_dir.is_relative_to(root.resolve()):
        raise ValueError("frozen Notebook M path escaped its root")
    state = _read_json(run_dir / "run_state.json")
    if (
        state.get("status") != "complete"
        or state.get("run_hash") != run_hash
        or state.get("protocol_hash") != protocol_hash
    ):
        raise ValueError("frozen Notebook M run is not complete or identity changed")
    records = state.get("artifacts")
    if not isinstance(records, dict):
        raise ValueError("frozen Notebook M artifact registry is missing")
    for name in M_READER_ARTIFACTS:
        path = run_dir / name
        record = records.get(name)
        if not path.is_file() or not isinstance(record, dict):
            raise ValueError(f"frozen Notebook M artifact missing: {name}")
        if int(record.get("size", -1)) != path.stat().st_size:
            raise ValueError(f"frozen Notebook M artifact size changed: {name}")
        if str(record.get("sha256", "")) != _sha256(path):
            raise ValueError(f"frozen Notebook M artifact hash changed: {name}")
    protocol = _read_json(run_dir / "protocol.json")
    summary = _read_json(run_dir / "summary.json")
    frozen_protocol = _read_json(run_dir / "frozen_protocol.json")
    for name in ("run_hash", "protocol_hash", "source_hash", "input_hash"):
        if protocol.get(name) != state.get(name) or summary.get(name) != state.get(name):
            raise ValueError(f"frozen Notebook M published identity changed: {name}")
    if state.get("summary") != summary:
        raise ValueError("frozen Notebook M state summary changed")
    if (
        protocol.get("stage") != "dev"
        or bool(protocol.get("smoke", False))
        or protocol.get("forward_or_lockbox_loaded") is not False
        or summary.get("forward_or_lockbox_loaded") is not False
    ):
        raise ValueError("Notebook N accepts only the full bounded M development run")
    labels_path = run_dir / "adaptive_labels.parquet"
    oof_path = run_dir / "oof_predictions.parquet"
    if frozen_protocol.get("labels_hash") != _sha256(labels_path):
        raise ValueError("frozen Notebook M label handoff hash changed")
    labels = pd.read_parquet(labels_path)
    scores = pd.read_parquet(oof_path)
    xgboost = scores.loc[scores["model"].eq("xgboost")].reset_index(drop=True)
    if labels.duplicated(["window_id", "step"]).any():
        raise ValueError("frozen Notebook M labels contain duplicate keys")
    if xgboost.empty or xgboost.duplicated(["window_id", "step"]).any():
        raise ValueError("frozen Notebook M XGBoost OOF handoff is invalid")
    expected_rows = int(summary.get("oof_rows_per_model", {}).get("xgboost", -1))
    if len(xgboost) != expected_rows:
        raise ValueError("frozen Notebook M XGBoost OOF row count changed")
    return FrozenMArtifacts(
        run_hash=run_hash,
        protocol_hash=protocol_hash,
        input_hash=str(protocol["input_hash"]),
        manifest_hash=str(frozen_protocol["manifest_hash"]),
        frozen_j_run_hash=str(frozen_protocol["frozen_j_run_hash"]),
        labels_sha256=_sha256(labels_path),
        oof_sha256=_sha256(oof_path),
        labels=labels,
        xgboost_scores=xgboost,
        protocol=protocol,
        summary=summary,
    )


def _source_hash() -> str:
    paths = (
        CODE_ROOT / "data" / "load.py",
        CODE_ROOT / "features" / "event_window_inputs.py",
        CODE_ROOT / "features" / "event_windows.py",
        CODE_ROOT / "features" / "linear_channels.py",
        CODE_ROOT / "experiments" / "event_window_dataset.py",
        CODE_ROOT / "experiments" / "event_window_tail_dataset.py",
        CODE_ROOT / "experiments" / "event_window_tail_oof.py",
        CODE_ROOT / "experiments" / "event_window_cost_aware_oof.py",
        CODE_ROOT / "experiments" / "event_window_large_move_dataset.py",
        CODE_ROOT / "experiments" / "event_window_large_move_models.py",
        CODE_ROOT / "experiments" / "event_window_large_move_oof.py",
        CODE_ROOT / "experiments" / "event_window_opportunity_oof.py",
        CODE_ROOT / "experiments" / "run_event_window_tail_models.py",
        CODE_ROOT / "experiments" / "run_event_window_adaptive_large_move.py",
        Path(__file__),
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(CODE_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def protocol_dict(
    config: OpportunityStudyConfig = OpportunityStudyConfig(), *, smoke: bool = False
) -> dict[str, object]:
    return {
        "stage": "dev",
        "smoke": bool(smoke),
        "development_start": config.development_start,
        "development_end_exclusive": config.development_end_exclusive,
        "objective": "one binary direction-free opportunity head",
        "output": "P(adaptive barrier is touched in either direction within 120 minutes)",
        "arms": list(ALL_ARMS),
        "promotable_arms": list(PROMOTABLE_ARMS),
        "baseline": "frozen Notebook M XGBoost with P(hit)=1-P(NO_BIG)",
        "target": "same frozen M adaptive 75-250 bps first-touch label",
        "double_touch": "opportunity positive; direction undefined and unused",
        "censoring": "a censored or gapped decision invalidates its entire window",
        "features": {
            "N1_binary_legacy": "same 380 M inputs; factorisation diagnostic only",
            "N2_side_neutral": "physical causal inputs with all LONG/SHORT semantics removed",
            "N3_side_neutral_volatility": "N2 plus the registered 19-feature causal volatility block",
            "N3_logreg": "simple baseline on the exact N3 matrix",
        },
        "split": "seven expanding episode-disjoint half-year OOF folds; 70/15/15 inner episodes",
        "weights": "half-open label uniqueness on fit, calibration and outer rows",
        "calibration": "Platt fit on early rows only; reserved calibration rows are not used",
        "primary_metrics": ["PR-AUC", "Brier", "log-loss"],
        "promotion": "side-neutral candidate must beat N0 with positive paired episode-bootstrap lower bounds for PR-AUC and Brier improvement; log-loss point estimate must improve",
        "bootstrap_unit": "channel_episode_id",
        "bootstrap_draws": 20 if smoke else config.bootstrap_draws,
        "bootstrap_confidence": 0.95,
        "matched_rate_diagnostic": (
            "post-hoc rank-only comparison at identical decisions/day; not a deployable threshold or trade policy"
        ),
        "matched_decisions_per_day": config.matched_decisions_per_day,
        "direction_head_trained": False,
        "trading_policy_trained": False,
        "economics_evaluated": False,
        "forward_or_lockbox_loaded": False,
        "target_config": asdict(config.target),
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
    frozen_m: FrozenMArtifacts,
) -> dict[str, object] | None:
    try:
        state = _read_json(run_dir / "run_state.json")
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
            if (
                not path.is_file()
                or not isinstance(record, dict)
                or int(record.get("size", -1)) != path.stat().st_size
                or str(record.get("sha256", "")) != _sha256(path)
            ):
                return None
        protocol = _read_json(run_dir / "protocol.json")
        frozen_protocol = _read_json(run_dir / "frozen_protocol.json")
        summary = _read_json(run_dir / "summary.json")
        if any(protocol.get(name) != value for name, value in identity.items()):
            return None
        if any(summary.get(name) != value for name, value in identity.items()):
            return None
        if state.get("summary") != summary:
            return None
        if frozen_protocol.get("frozen_m_run_hash") != frozen_m.run_hash:
            return None
        if frozen_protocol.get("frozen_m_labels_sha256") != frozen_m.labels_sha256:
            return None
        if frozen_protocol.get("frozen_m_oof_sha256") != frozen_m.oof_sha256:
            return None
        if frozen_protocol.get("opportunity_labels_sha256") != _sha256(
            run_dir / "opportunity_labels.parquet"
        ):
            return None
        return summary
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _align_labels(base, labels: pd.DataFrame) -> pd.DataFrame:
    keys = ["window_id", "step"]
    expected = base.decisions[keys].reset_index(drop=True)
    source = labels.set_index(keys)
    if not source.index.is_unique:
        raise ValueError("frozen M labels contain duplicate decision keys")
    wanted = pd.MultiIndex.from_frame(expected)
    missing = wanted.difference(source.index)
    if len(missing):
        raise ValueError("frozen M labels do not cover rebuilt Notebook N decisions")
    aligned = source.reindex(wanted).reset_index()
    if not expected.equals(aligned[keys]):
        raise AssertionError("frozen M labels changed decision order")
    return aligned


def _feature_audit(
    arm: str,
    model: str,
    dataset: LargeMoveDecisionDataset,
) -> pd.DataFrame:
    kept = pd.DataFrame(
        {
            "arm": arm,
            "model": model,
            "feature": dataset.tabular_features,
            "kept": True,
            "reason": f"retained by {dataset.feature_set}",
        }
    )
    dropped = pd.DataFrame(
        {
            "arm": arm,
            "model": model,
            "feature": dataset.dropped_features,
            "kept": False,
            "reason": "mask, duplicate or LONG/SHORT-dependent field",
        }
    )
    return pd.concat([kept, dropped], ignore_index=True)


def _baseline_result(
    frozen_m: FrozenMArtifacts,
    reference: OpportunityOOFResult,
) -> OpportunityOOFResult:
    keys = ["window_id", "step"]
    baseline = reference.scores.drop(columns=["arm", "model", "p_hit"]).merge(
        frozen_m.xgboost_scores[keys + ["fold_id", "decision_time", "p_no_big"]],
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_m"),
    )
    if baseline[["p_no_big", "fold_id_m", "decision_time_m"]].isna().any().any():
        raise ValueError("frozen M baseline does not cover every opportunity OOF key")
    if not baseline["fold_id"].eq(baseline["fold_id_m"]).all():
        raise ValueError("frozen M and Notebook N fold identities differ")
    if not pd.to_datetime(baseline["decision_time"], utc=True).eq(
        pd.to_datetime(baseline["decision_time_m"], utc=True)
    ).all():
        raise ValueError("frozen M and Notebook N decision timestamps differ")
    baseline = baseline.drop(columns=["fold_id_m", "decision_time_m"])
    baseline.insert(0, "model", "frozen_m_xgboost")
    baseline.insert(0, "arm", BASELINE_ARM)
    baseline["p_hit"] = 1.0 - pd.to_numeric(baseline.pop("p_no_big"), errors="raise")
    metric_rows = []
    for fold, group in baseline.groupby("fold_id", sort=False):
        metric_rows.append(
            {
                "arm": BASELINE_ARM,
                "model": "frozen_m_xgboost",
                "fold": fold,
                "feature_set": "frozen M multiclass",
                "features": int(frozen_m.summary["retained_features"]),
                "fit_rows": np.nan,
                "early_rows": np.nan,
                "reserved_calibration_rows": np.nan,
                "validation_rows": len(group),
                "fit_episodes": np.nan,
                "early_episodes": np.nan,
                "reserved_calibration_episodes": np.nan,
                "validation_episodes": group["channel_episode_id"].nunique(),
                "episode_overlap": 0,
                "train_label_end_max": pd.NaT,
                "validation_start": group["decision_time"].min(),
                "fit_model_hash": "frozen Notebook M",
                **opportunity_metrics(
                    group["opportunity_code"].to_numpy(int),
                    group["p_hit"].to_numpy(float),
                    group["sample_weight"].to_numpy(float),
                ),
            }
        )
    return OpportunityOOFResult(
        BASELINE_ARM,
        "frozen_m_xgboost",
        baseline,
        pd.DataFrame(metric_rows),
        pd.DataFrame(
            [
                {
                    "arm": BASELINE_ARM,
                    "model": "frozen_m_xgboost",
                    "fold": "frozen",
                    "platt_slope": np.nan,
                    "platt_intercept": np.nan,
                    "identity_fallback": False,
                    "early_rows": np.nan,
                    "early_episodes": np.nan,
                    "early_effective_weight": np.nan,
                    "reserved_calibration_rows": np.nan,
                    "threshold_status": "frozen M temperature; no N threshold",
                }
            ]
        ),
    )


def _aggregate_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, group in scores.groupby("arm", sort=False):
        metric = opportunity_metrics(
            group["opportunity_code"].to_numpy(int),
            group["p_hit"].to_numpy(float),
            group["sample_weight"].to_numpy(float),
        )
        rows.append(
            {
                "arm": arm,
                "model": group["model"].iloc[0],
                "rows": len(group),
                "episodes": group["channel_episode_id"].nunique(),
                **metric,
            }
        )
    return pd.DataFrame(rows)


def _reliability(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    edges = np.linspace(0.0, 1.0, 11)
    for arm, group in scores.groupby("arm", sort=False):
        probability = group["p_hit"].to_numpy(float)
        target = group["opportunity_code"].to_numpy(float)
        weights = group["sample_weight"].to_numpy(float)
        bins = np.minimum(np.searchsorted(edges, probability, side="right") - 1, 9)
        bins = np.maximum(bins, 0)
        for index in range(10):
            mask = bins == index
            if not mask.any():
                continue
            rows.append(
                {
                    "arm": arm,
                    "bin": index,
                    "bin_left": edges[index],
                    "bin_right": edges[index + 1],
                    "rows": int(mask.sum()),
                    "weight": float(weights[mask].sum()),
                    "mean_probability": float(np.average(probability[mask], weights=weights[mask])),
                    "observed_rate": float(np.average(target[mask], weights=weights[mask])),
                }
            )
    return pd.DataFrame(rows)


def _matched_frequency(
    scores: pd.DataFrame,
    *,
    decisions_per_day: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for arm, arm_rows in scores.groupby("arm", sort=False):
        selected_by_fold = []
        calendar_days = 0
        for _, fold in arm_rows.groupby("fold_id", sort=False):
            times = pd.to_datetime(fold["decision_time"], utc=True)
            days = len(pd.date_range(times.min().normalize(), times.max().normalize(), freq="D"))
            calendar_days += days
            count = max(1, int(round(decisions_per_day * days)))
            best_per_window = (
                fold.sort_values(["p_hit", "decision_time"], ascending=[False, True], kind="stable")
                .drop_duplicates("window_id", keep="first")
                .head(count)
                .copy()
            )
            best_per_window["matched_rank"] = np.arange(1, len(best_per_window) + 1)
            selected_by_fold.append(best_per_window)
        selected = pd.concat(selected_by_fold, ignore_index=True)
        selected_frames.append(selected)
        weights = selected["sample_weight"].to_numpy(float)
        target = selected["opportunity_code"].to_numpy(float)
        all_weights = arm_rows["sample_weight"].to_numpy(float)
        all_target = arm_rows["opportunity_code"].to_numpy(float)
        prevalence = float(np.average(all_target, weights=all_weights))
        precision = float(np.average(target, weights=weights))
        summary_rows.append(
            {
                "arm": arm,
                "matched_decisions": len(selected),
                "calendar_days": calendar_days,
                "matched_decisions_per_day": len(selected) / calendar_days,
                "opportunity_precision": precision,
                "opportunity_recall": float(
                    np.sum(weights * target) / np.sum(all_weights * all_target)
                ),
                "opportunity_lift": precision / prevalence if prevalence > 0.0 else np.nan,
                "diagnostic_only": True,
            }
        )
    return pd.concat(selected_frames, ignore_index=True), pd.DataFrame(summary_rows)


def _metric_bootstrap_draws(
    scores: pd.DataFrame,
    episode_multiplicity: np.ndarray,
    episode_codes: np.ndarray,
    *,
    chunk_size: int = 8,
) -> dict[str, np.ndarray]:
    y = scores["opportunity_code"].to_numpy(float)
    p = np.clip(scores["p_hit"].to_numpy(float), 1e-8, 1.0 - 1e-8)
    base_weight = scores["sample_weight"].to_numpy(float)
    draws = episode_multiplicity.shape[1]
    output = {
        "opportunity_pr_auc": np.empty(draws, dtype=float),
        "opportunity_brier": np.empty(draws, dtype=float),
        "opportunity_logloss": np.empty(draws, dtype=float),
    }
    order = np.argsort(-p, kind="stable")
    sorted_p = p[order]
    group_ends = np.r_[np.flatnonzero(sorted_p[:-1] != sorted_p[1:]), len(p) - 1]
    brier_error = np.square(p - y)
    log_error = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
    for start in range(0, draws, chunk_size):
        stop = min(draws, start + chunk_size)
        multiplicity = episode_multiplicity[episode_codes, start:stop]
        weight = base_weight[:, None] * multiplicity
        denominator = weight.sum(axis=0)
        output["opportunity_brier"][start:stop] = (
            (weight * brier_error[:, None]).sum(axis=0) / denominator
        )
        output["opportunity_logloss"][start:stop] = (
            (weight * log_error[:, None]).sum(axis=0) / denominator
        )
        sorted_weight = weight[order]
        cumulative_total = np.cumsum(sorted_weight, axis=0)
        cumulative_positive = np.cumsum(sorted_weight * y[order, None], axis=0)
        true_positive = cumulative_positive[group_ends]
        predicted_positive = cumulative_total[group_ends]
        increment = np.diff(
            np.vstack([np.zeros((1, stop - start)), true_positive]), axis=0
        )
        precision = np.divide(
            true_positive,
            predicted_positive,
            out=np.zeros_like(true_positive),
            where=predicted_positive > 0.0,
        )
        total_positive = true_positive[-1]
        output["opportunity_pr_auc"][start:stop] = np.divide(
            (precision * increment).sum(axis=0),
            total_positive,
            out=np.full(stop - start, np.nan),
            where=total_positive > 0.0,
        )
    return output


def _paired_bootstrap(
    scores: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    baseline = scores.loc[scores["arm"].eq(BASELINE_ARM)].sort_values(
        ["window_id", "step"], kind="stable"
    ).reset_index(drop=True)
    episodes, episode_codes = np.unique(
        baseline["channel_episode_id"].astype(str).to_numpy(), return_inverse=True
    )
    rng = np.random.default_rng(seed)
    multiplicity = np.empty((len(episodes), draws), dtype=np.int16)
    for draw in range(draws):
        sampled = rng.integers(0, len(episodes), len(episodes))
        multiplicity[:, draw] = np.bincount(sampled, minlength=len(episodes))
    baseline_draws = _metric_bootstrap_draws(
        baseline, multiplicity, episode_codes
    )
    point = metrics.set_index("arm")
    rows: list[dict[str, object]] = []
    for arm in ALL_ARMS[1:]:
        candidate = scores.loc[scores["arm"].eq(arm)].sort_values(
            ["window_id", "step"], kind="stable"
        ).reset_index(drop=True)
        if not candidate[["window_id", "step"]].equals(
            baseline[["window_id", "step"]]
        ):
            raise AssertionError("paired bootstrap requires identical ordered OOF keys")
        if not np.allclose(candidate["sample_weight"], baseline["sample_weight"]):
            raise AssertionError("paired bootstrap requires identical uniqueness weights")
        candidate_draws = _metric_bootstrap_draws(
            candidate, multiplicity, episode_codes
        )
        definitions = (
            (
                "pr_auc_delta",
                "opportunity_pr_auc",
                candidate_draws["opportunity_pr_auc"] - baseline_draws["opportunity_pr_auc"],
                point.at[arm, "opportunity_pr_auc"] - point.at[BASELINE_ARM, "opportunity_pr_auc"],
            ),
            (
                "brier_improvement",
                "opportunity_brier",
                baseline_draws["opportunity_brier"] - candidate_draws["opportunity_brier"],
                point.at[BASELINE_ARM, "opportunity_brier"] - point.at[arm, "opportunity_brier"],
            ),
            (
                "logloss_improvement",
                "opportunity_logloss",
                baseline_draws["opportunity_logloss"] - candidate_draws["opportunity_logloss"],
                point.at[BASELINE_ARM, "opportunity_logloss"] - point.at[arm, "opportunity_logloss"],
            ),
        )
        for comparison, metric, values, point_delta in definitions:
            finite = values[np.isfinite(values)]
            low, high = np.quantile(finite, [0.025, 0.975])
            rows.append(
                {
                    "candidate": arm,
                    "baseline": BASELINE_ARM,
                    "comparison": comparison,
                    "metric": metric,
                    "point_improvement": float(point_delta),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "bootstrap_draws": draws,
                }
            )
    return pd.DataFrame(rows)


def _promotion_decision(
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    result = metrics.copy()
    result["promotable"] = result["arm"].isin(PROMOTABLE_ARMS)
    result["predictive_pass"] = False
    for arm in PROMOTABLE_ARMS:
        comparison = paired.loc[paired["candidate"].eq(arm)].set_index("comparison")
        passes = bool(
            comparison.at["pr_auc_delta", "ci_low"] > 0.0
            and comparison.at["brier_improvement", "ci_low"] > 0.0
            and comparison.at["logloss_improvement", "point_improvement"] > 0.0
        )
        result.loc[result["arm"].eq(arm), "predictive_pass"] = passes
    passing = result.loc[result["predictive_pass"]].sort_values(
        ["opportunity_pr_auc", "opportunity_brier"],
        ascending=[False, True],
        kind="stable",
    )
    chosen = None if passing.empty else str(passing.iloc[0]["arm"])
    return result, chosen


def run_opportunity_study(
    *,
    stage: str = "dev",
    smoke: bool = False,
    data_root: Path = DEFAULT_DATA_ROOT,
    frozen_j_root: Path = FROZEN_J_ROOT,
    frozen_m_root: Path = FROZEN_M_ROOT,
    run_root: Path = RUN_ROOT,
    config: OpportunityStudyConfig = OpportunityStudyConfig(),
) -> OpportunityRunResult:
    if stage != "dev":
        raise ValueError("Notebook N permits development only; forward and Q2 are sealed")
    protocol = protocol_dict(config, smoke=smoke)
    protocol_hash = _sha_payload(protocol)
    frozen_m = load_frozen_m_artifacts(Path(frozen_m_root))
    frozen_j = load_frozen_j_artifacts(Path(frozen_j_root))
    if frozen_m.frozen_j_run_hash != frozen_j.run_hash:
        raise ValueError("frozen Notebook M and J run identities differ")
    if frozen_m.manifest_hash != frozen_j.manifest_sha256:
        raise ValueError("frozen Notebook M and J manifests differ")
    source_hash = _source_hash()
    input_hash = _sha_payload(
        {
            "frozen_m_run_hash": frozen_m.run_hash,
            "frozen_m_protocol_hash": frozen_m.protocol_hash,
            "frozen_m_labels_sha256": frozen_m.labels_sha256,
            "frozen_m_oof_sha256": frozen_m.oof_sha256,
            "frozen_j_run_hash": frozen_j.run_hash,
            "frozen_j_manifest_sha256": frozen_j.manifest_sha256,
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
        run_dir, identity, frozen_m=frozen_m
    )
    if cached is not None:
        if not smoke:
            _latest(Path(run_root), run_hash, protocol_hash)
        return OpportunityRunResult(run_dir, cached)
    store = _Store(run_dir, identity)
    try:
        store.json("protocol.json", {**protocol, **identity})
        base, _, loaded = _build_tail_dataset(
            frozen_j, data_root=Path(data_root), smoke=smoke
        )
        expected_start = pd.Timestamp(config.development_start, tz="UTC")
        expected_end = pd.Timestamp(config.development_end_exclusive, tz="UTC")
        if not smoke and (
            loaded.read_start != expected_start
            or loaded.read_end_exclusive != expected_end
            or loaded.max_loaded_timestamp >= expected_end
        ):
            raise AssertionError("Notebook N bounded development inputs changed")
        legacy_labels = _align_labels(base, frozen_m.labels)
        legacy_valid = legacy_labels["model_target_valid"].astype(bool).to_numpy()
        labels = add_opportunity_target_contract(legacy_labels)
        if not np.array_equal(legacy_valid, labels["model_target_valid"].to_numpy(bool)):
            raise AssertionError("Notebook N changed frozen M three-class validity")
        store.parquet("opportunity_labels.parquet", labels)
        label_audit = (
            labels.groupby(
                ["move_label", "opportunity_code", "opportunity_target_valid"],
                dropna=False,
            )
            .size()
            .rename("rows")
            .reset_index()
        )
        store.csv("label_audit.csv", label_audit)

        legacy = build_large_move_dataset(base, labels, feature_set="directional")
        side_neutral = build_opportunity_dataset(
            base, labels, include_volatility=False
        )
        volatility = build_opportunity_dataset(
            base, labels, include_volatility=True
        )
        datasets = {
            "legacy": legacy,
            "side_neutral": side_neutral,
            "side_neutral_volatility": volatility,
        }
        feature_audit = pd.concat(
            [
                _feature_audit(arm, model, datasets[dataset_name])
                for arm, model, dataset_name in TRAINED_ARMS
            ],
            ignore_index=True,
        )
        store.csv("feature_audit.csv", feature_audit)
        model_config = replace(config.model, xgb_estimators=12) if smoke else config.model
        oof_config = OpportunityOOFConfig(model=model_config)
        trained = [
            run_opportunity_model_oof(
                arm, model, datasets[dataset_name], oof_config
            )
            for arm, model, dataset_name in TRAINED_ARMS
        ]
        assert_identical_opportunity_keys(trained)
        baseline = _baseline_result(frozen_m, trained[0])
        results = [baseline, *trained]
        assert_identical_opportunity_keys(results)
        scores = pd.concat([result.scores for result in results], ignore_index=True)
        fold_audit = pd.concat(
            [result.fold_audit for result in results], ignore_index=True
        )
        calibration_audit = pd.concat(
            [result.calibration_audit for result in results], ignore_index=True
        )
        if not fold_audit["episode_overlap"].fillna(0).eq(0).all():
            raise AssertionError("Notebook N episode leakage audit failed")
        store.parquet("oof_predictions.parquet", scores)
        store.csv("fold_audit.csv", fold_audit)
        store.csv("calibration_audit.csv", calibration_audit)

        metrics = _aggregate_metrics(scores)
        reliability = _reliability(scores)
        matched, matched_summary = _matched_frequency(
            scores, decisions_per_day=config.matched_decisions_per_day
        )
        draws = 20 if smoke else config.bootstrap_draws
        paired = _paired_bootstrap(
            scores, metrics, draws=draws, seed=config.bootstrap_seed
        )
        metrics, chosen = _promotion_decision(metrics, paired)
        store.csv("metrics.csv", metrics)
        store.csv("paired_bootstrap.csv", paired)
        store.csv("reliability.csv", reliability)
        store.csv("matched_frequency.csv", matched_summary)
        store.parquet("matched_decisions.parquet", matched)
        store.json(
            "frozen_protocol.json",
            {
                "source": "validated full development Notebook M and its frozen Notebook J manifest",
                "frozen_m_run_hash": frozen_m.run_hash,
                "frozen_m_protocol_hash": frozen_m.protocol_hash,
                "frozen_m_labels_sha256": frozen_m.labels_sha256,
                "frozen_m_oof_sha256": frozen_m.oof_sha256,
                "frozen_j_run_hash": frozen_j.run_hash,
                "frozen_j_manifest_sha256": frozen_j.manifest_sha256,
                "opportunity_labels_sha256": _sha256(
                    run_dir / "opportunity_labels.parquet"
                ),
                "common_oof_rows_per_arm": int(
                    scores.groupby("arm").size().min()
                ),
                "direction_head_trained": False,
                "forward_or_lockbox_loaded": False,
            },
        )
        decision = (
            f"promote {chosen} as the standalone opportunity head"
            if chosen
            else "no side-neutral opportunity head beat frozen M with registered paired evidence"
        )
        summary = {
            **identity,
            "arms": list(ALL_ARMS),
            "decision": decision,
            "chosen_arm": chosen,
            "adaptive_label_rows": len(labels),
            "valid_opportunity_rows": int(labels["opportunity_target_valid"].sum()),
            "ambiguous_opportunity_rows": int(labels["move_label"].eq("ambiguous").sum()),
            "oof_rows_per_arm": scores.groupby("arm").size().astype(int).to_dict(),
            "feature_counts": {
                arm: len(datasets[dataset_name].tabular_features)
                for arm, _, dataset_name in TRAINED_ARMS
            },
            "metrics": metrics.set_index("arm").to_dict(orient="index"),
            "matched_frequency": matched_summary.set_index("arm").to_dict(orient="index"),
            "max_loaded_timestamp": loaded.max_loaded_timestamp,
            "read_end_exclusive": loaded.read_end_exclusive,
            "direction_head_trained": False,
            "economics_evaluated": False,
            "forward_or_lockbox_loaded": False,
        }
        store.json("summary.json", summary)
        missing = [name for name in READER_ARTIFACTS if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"Notebook N artifacts missing: {missing}")
        store.complete(summary)
        if not smoke:
            _latest(Path(run_root), run_hash, protocol_hash)
        return OpportunityRunResult(run_dir, summary)
    except BaseException as error:
        store.fail(error)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="dev")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--frozen-j-root", type=Path, default=FROZEN_J_ROOT)
    parser.add_argument("--frozen-m-root", type=Path, default=FROZEN_M_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    result = run_opportunity_study(
        stage=arguments.stage,
        smoke=arguments.smoke,
        data_root=arguments.data_root,
        frozen_j_root=arguments.frozen_j_root,
        frozen_m_root=arguments.frozen_m_root,
        run_root=arguments.run_root,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True, default=_jsonable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALL_ARMS",
    "BASELINE_ARM",
    "FROZEN_M_ROOT",
    "OpportunityRunResult",
    "OpportunityStudyConfig",
    "PROMOTABLE_ARMS",
    "READER_ARTIFACTS",
    "load_frozen_m_artifacts",
    "protocol_dict",
    "run_opportunity_study",
]
