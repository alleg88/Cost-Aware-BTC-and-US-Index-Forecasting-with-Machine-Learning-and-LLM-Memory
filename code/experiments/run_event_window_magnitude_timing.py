"""Run Notebook O: full-path magnitude and time-to-hit opportunity study."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.event_window_large_move_dataset import build_opportunity_dataset
from experiments.event_window_large_move_models import LargeMoveModelConfig
from experiments.event_window_magnitude_dataset import (
    MAGNITUDE_BIN_LABELS,
    MAGNITUDE_THRESHOLDS,
    MagnitudeTargetConfig,
    TTH_HORIZONS_MINUTES,
    align_magnitude_dataset,
    assert_magnitude_feature_isolation,
    label_full_path_magnitude,
    time_to_hit_bucket,
)
from experiments.event_window_magnitude_oof import (
    CUMULATIVE_CLASS_STARTS,
    MagnitudeOOFConfig,
    magnitude_metrics,
    run_magnitude_oof,
)
from experiments.event_window_opportunity_oof import opportunity_metrics
from experiments.run_event_window_cost_aware_entry import (
    _Store,
    _sha256,
    _sha_payload,
)
from experiments.run_event_window_opportunity_head import (
    READER_ARTIFACTS as N_READER_ARTIFACTS,
    _align_labels,
    _metric_bootstrap_draws,
    _source_hash as _n_source_hash,
)
from experiments.run_event_window_tail_models import (
    FROZEN_J_ROOT,
    _build_tail_dataset,
    load_frozen_j_artifacts,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = CODE_ROOT / "data"
FROZEN_N_ROOT = CODE_ROOT / "experiments" / "cache" / "event_window_opportunity_head"
RUN_ROOT = CODE_ROOT / "experiments" / "cache" / "event_window_magnitude_timing"
MODEL_NAME = "O1_xgboost_multiclass_magnitude"
BASELINE_NAME = "N3_frozen_binary_opportunity"
READER_ARTIFACTS = (
    "magnitude_labels.parquet",
    "feature_audit.csv",
    "label_audit.csv",
    "tth_audit.csv",
    "target_nesting.csv",
    "fold_audit.csv",
    "calibration_audit.csv",
    "leakage_audit.csv",
    "oof_predictions.parquet",
    "comparison_predictions.parquet",
    "magnitude_metrics.csv",
    "cumulative_metrics.csv",
    "primary_comparison.csv",
    "paired_bootstrap.csv",
    "event_metrics.csv",
    "protocol.json",
    "frozen_protocol.json",
    "summary.json",
)


@dataclass(frozen=True)
class FrozenNArtifacts:
    run_hash: str
    protocol_hash: str
    source_hash: str
    input_hash: str
    labels_sha256: str
    oof_sha256: str
    frozen_j_run_hash: str
    frozen_j_manifest_sha256: str
    labels: pd.DataFrame
    n3_scores: pd.DataFrame
    protocol: dict[str, object]
    summary: dict[str, object]


@dataclass(frozen=True)
class MagnitudeStudyConfig:
    development_start: str = "2021-01-01"
    development_end_exclusive: str = "2025-07-01"
    diagnostic_alerts_per_day: float = 1.0
    bootstrap_draws: int = 500
    bootstrap_seed: int = 42
    target: MagnitudeTargetConfig = field(default_factory=MagnitudeTargetConfig)
    model: LargeMoveModelConfig = field(default_factory=LargeMoveModelConfig)


@dataclass(frozen=True)
class MagnitudeRunResult:
    run_dir: Path
    summary: dict[str, object]


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path.name}")
    return value


def load_frozen_n_artifacts(
    run_root: Path = FROZEN_N_ROOT,
    *,
    expected_run_hash: str | None = None,
) -> FrozenNArtifacts:
    """Validate the complete development-only Notebook N handoff."""
    root = Path(run_root)
    pointer = None if expected_run_hash is not None else _read_json(root / "latest_dev.json")
    run_hash = str(expected_run_hash or pointer.get("run_hash", ""))
    expected_relative = f"{run_hash}/full"
    if not run_hash or (
        pointer is not None and pointer.get("relative_path") != expected_relative
    ):
        raise ValueError("frozen Notebook N pointer is invalid")
    run_dir = (root / expected_relative).resolve()
    if not run_dir.is_relative_to(root.resolve()):
        raise ValueError("frozen Notebook N path escaped its root")
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
        raise ValueError("frozen Notebook N run is incomplete or changed")
    records = state.get("artifacts")
    if not isinstance(records, dict):
        raise ValueError("frozen Notebook N artifact registry is missing")
    for name in N_READER_ARTIFACTS:
        path = run_dir / name
        record = records.get(name)
        if not path.is_file() or not isinstance(record, dict):
            raise ValueError(f"frozen Notebook N artifact missing: {name}")
        if int(record.get("size", -1)) != path.stat().st_size:
            raise ValueError(f"frozen Notebook N artifact size changed: {name}")
        if str(record.get("sha256", "")) != _sha256(path):
            raise ValueError(f"frozen Notebook N artifact hash changed: {name}")
    protocol = _read_json(run_dir / "protocol.json")
    summary = _read_json(run_dir / "summary.json")
    frozen = _read_json(run_dir / "frozen_protocol.json")
    for name in ("run_hash", "protocol_hash", "source_hash", "input_hash"):
        if protocol.get(name) != state.get(name) or summary.get(name) != state.get(name):
            raise ValueError(f"frozen Notebook N published identity changed: {name}")
    if state.get("summary") != summary:
        raise ValueError("frozen Notebook N state summary changed")
    if (
        protocol.get("stage") != "dev"
        or bool(protocol.get("smoke", False))
        or summary.get("chosen_arm") != "N3_side_neutral_volatility"
        or protocol.get("forward_or_lockbox_loaded") is not False
        or summary.get("forward_or_lockbox_loaded") is not False
    ):
        raise ValueError("Notebook O accepts only the complete bounded N3 development run")
    labels_path = run_dir / "opportunity_labels.parquet"
    scores_path = run_dir / "oof_predictions.parquet"
    if frozen.get("opportunity_labels_sha256") != _sha256(labels_path):
        raise ValueError("frozen Notebook N label handoff hash changed")
    labels = pd.read_parquet(labels_path)
    scores = pd.read_parquet(scores_path)
    n3 = scores.loc[scores["arm"].eq("N3_side_neutral_volatility")].reset_index(drop=True)
    if labels.duplicated(["window_id", "step"]).any():
        raise ValueError("frozen Notebook N labels contain duplicate keys")
    if n3.empty or n3.duplicated(["window_id", "step"]).any():
        raise ValueError("frozen Notebook N N3 OOF handoff is invalid")
    expected_rows = int(summary.get("oof_rows_per_arm", {}).get("N3_side_neutral_volatility", -1))
    if len(n3) != expected_rows:
        raise ValueError("frozen Notebook N N3 OOF row count changed")
    return FrozenNArtifacts(
        run_hash=run_hash,
        protocol_hash=protocol_hash,
        source_hash=str(protocol["source_hash"]),
        input_hash=str(protocol["input_hash"]),
        labels_sha256=_sha256(labels_path),
        oof_sha256=_sha256(scores_path),
        frozen_j_run_hash=str(frozen["frozen_j_run_hash"]),
        frozen_j_manifest_sha256=str(frozen["frozen_j_manifest_sha256"]),
        labels=labels,
        n3_scores=n3,
        protocol=protocol,
        summary=summary,
    )


def _source_hash() -> str:
    paths = (
        CODE_ROOT / "experiments" / "event_window_magnitude_dataset.py",
        CODE_ROOT / "experiments" / "event_window_magnitude_models.py",
        CODE_ROOT / "experiments" / "event_window_magnitude_oof.py",
        CODE_ROOT / "experiments" / "event_window_large_move_dataset.py",
        CODE_ROOT / "experiments" / "event_window_large_move_models.py",
        CODE_ROOT / "experiments" / "event_window_cost_aware_oof.py",
        CODE_ROOT / "experiments" / "event_window_tail_oof.py",
        CODE_ROOT / "experiments" / "run_event_window_opportunity_head.py",
        Path(__file__),
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(CODE_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def protocol_dict(
    config: MagnitudeStudyConfig = MagnitudeStudyConfig(),
    *,
    smoke: bool = False,
) -> dict[str, object]:
    return {
        "stage": "dev",
        "smoke": bool(smoke),
        "development_start": config.development_start,
        "development_end_exclusive": config.development_end_exclusive,
        "objective": "one direction-free five-bin full-path magnitude head",
        "decision_calendar": "exact frozen N3 decisions: completed 5m bars inside causal event windows",
        "reference": "1m Open at decision time t",
        "future_path": "complete half-open [t,t+120m); offsets 0..119 correspond to TTH minutes 1..120",
        "target": "M=max(up excursion, down excursion)/frozen adaptive barrier B",
        "magnitude_bins": list(MAGNITUDE_BIN_LABELS),
        "magnitude_thresholds": list(MAGNITUDE_THRESHOLDS),
        "primary_binary_projection": "M>=1.0B",
        "time_to_hit_horizons_minutes": list(TTH_HORIZONS_MINUTES),
        "censoring": "row-level strict complete case; gaps/non-finite bars are censored, never negative",
        "label_interval": "label_start=t and label_end=t+120m for every row, independent of hit time",
        "features": "exact frozen N3 side-neutral causal 248-column matrix",
        "model": MODEL_NAME,
        "baseline": BASELINE_NAME,
        "split": "same seven expanding episode-disjoint half-year OOF folds as Notebook N; chronological 70/15/15 inner episodes",
        "weights": "half-open uniqueness on the single [t,t+120m) label interval",
        "calibration": "one multiclass temperature on early episodes only",
        "primary_metrics": ["multiclass log-loss", "ranked probability score"],
        "secondary_metrics": "cumulative PR-AUC, ROC-AUC, Brier, log-loss and top-decile lift",
        "baseline_comparison": "P(M>=1B) versus frozen N3 P(hit) on identical OOF keys and weights",
        "bootstrap_unit": "channel_episode_id",
        "bootstrap_draws": 20 if smoke else config.bootstrap_draws,
        "event_diagnostic": {
            "alerts_per_day": config.diagnostic_alerts_per_day,
            "truth_event": "contiguous positive 5m decisions within one channel_episode_id",
            "alert_event": "contiguous selected 5m decisions within one channel_episode_id",
            "lead_time": "first selected positive decision to its first future 1B hit",
        },
        "direction_head_trained": False,
        "trading_policy_trained": False,
        "economics_evaluated": False,
        "hawkes_or_evt_used": False,
        "forward_or_lockbox_loaded": False,
        "target_config": {
            "adaptive": asdict(config.target.adaptive),
            "thresholds": list(config.target.thresholds),
            "time_to_hit_horizons": list(config.target.time_to_hit_horizons),
        },
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
    frozen_n: FrozenNArtifacts,
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
        frozen = _read_json(run_dir / "frozen_protocol.json")
        summary = _read_json(run_dir / "summary.json")
        if any(protocol.get(name) != value for name, value in identity.items()):
            return None
        if any(summary.get(name) != value for name, value in identity.items()):
            return None
        if state.get("summary") != summary:
            return None
        if frozen.get("frozen_n_run_hash") != frozen_n.run_hash:
            return None
        if frozen.get("frozen_n_labels_sha256") != frozen_n.labels_sha256:
            return None
        if frozen.get("frozen_n_oof_sha256") != frozen_n.oof_sha256:
            return None
        if frozen.get("magnitude_labels_sha256") != _sha256(
            run_dir / "magnitude_labels.parquet"
        ):
            return None
        return summary
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _aggregate_tables(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    probability_columns = [f"p_bin_{number}" for number in range(5)]
    values = magnitude_metrics(
        scores["magnitude_class"].to_numpy(int),
        scores[probability_columns].to_numpy(float),
        scores["sample_weight"].to_numpy(float),
    )
    magnitude = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "multiclass_logloss": values["multiclass_logloss"],
                "ranked_probability_score": values["ranked_probability_score"],
                "monotonic_violation_rate": values["monotonic_violation_rate"],
                "oof_rows": len(scores),
            }
        ]
    )
    cumulative_rows = []
    for name, start in CUMULATIVE_CLASS_STARTS.items():
        cumulative_rows.append(
            {
                "threshold_b": {"075": 0.75, "100": 1.0, "150": 1.5, "200": 2.0}[name],
                "class_start": start,
                "prevalence": values[f"ge_{name}_prevalence"],
                "pr_auc": values[f"ge_{name}_pr_auc"],
                "roc_auc": values[f"ge_{name}_roc_auc"],
                "brier": values[f"ge_{name}_brier"],
                "logloss": values[f"ge_{name}_logloss"],
                "top_decile_rate": values[f"ge_{name}_top_decile_rate"],
                "top_decile_lift": values[f"ge_{name}_top_decile_lift"],
            }
        )
    return magnitude, pd.DataFrame(cumulative_rows)


def _baseline_comparison(
    scores: pd.DataFrame,
    frozen_n: FrozenNArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["window_id", "step"]
    baseline = frozen_n.n3_scores[
        keys + ["fold_id", "decision_time", "p_hit"]
    ].copy()
    comparison = scores.merge(
        baseline,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_n3"),
    )
    if comparison.empty:
        raise ValueError("frozen N3 and Notebook O have no common OOF decisions")
    if len(comparison) != len(scores):
        raise ValueError("frozen N3 does not cover every Notebook O OOF decision")
    if not comparison["fold_id"].eq(comparison["fold_id_n3"]).all():
        raise ValueError("frozen N3 and Notebook O fold identities differ")
    if not pd.to_datetime(comparison["decision_time"], utc=True).eq(
        pd.to_datetime(comparison["decision_time_n3"], utc=True)
    ).all():
        raise ValueError("frozen N3 and Notebook O decision timestamps differ")
    comparison = comparison.drop(columns=["fold_id_n3", "decision_time_n3"])
    labels = comparison["y_ge_100"].to_numpy(int)
    weights = comparison["sample_weight"].to_numpy(float)
    rows = []
    for model, column in ((MODEL_NAME, "p_ge_100"), (BASELINE_NAME, "p_hit")):
        metrics = opportunity_metrics(
            labels,
            comparison[column].to_numpy(float),
            weights,
        )
        rows.append({"model": model, "common_oof_rows": len(comparison), **metrics})
    return comparison, pd.DataFrame(rows)


def _paired_primary_bootstrap(
    comparison: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    ordered = comparison.sort_values(["window_id", "step"], kind="stable").reset_index(drop=True)
    episodes, episode_codes = np.unique(
        ordered["channel_episode_id"].astype(str).to_numpy(), return_inverse=True
    )
    rng = np.random.default_rng(seed)
    multiplicity = np.empty((len(episodes), draws), dtype=np.int16)
    for draw in range(draws):
        sampled = rng.integers(0, len(episodes), len(episodes))
        multiplicity[:, draw] = np.bincount(sampled, minlength=len(episodes))

    def frame(probability: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "opportunity_code": ordered["y_ge_100"].to_numpy(int),
                "p_hit": ordered[probability].to_numpy(float),
                "sample_weight": ordered["sample_weight"].to_numpy(float),
            }
        )

    candidate_draws = _metric_bootstrap_draws(
        frame("p_ge_100"), multiplicity, episode_codes
    )
    baseline_draws = _metric_bootstrap_draws(
        frame("p_hit"), multiplicity, episode_codes
    )
    point = metrics.set_index("model")
    definitions = (
        (
            "pr_auc_delta",
            "opportunity_pr_auc",
            candidate_draws["opportunity_pr_auc"] - baseline_draws["opportunity_pr_auc"],
            point.at[MODEL_NAME, "opportunity_pr_auc"] - point.at[BASELINE_NAME, "opportunity_pr_auc"],
        ),
        (
            "brier_improvement",
            "opportunity_brier",
            baseline_draws["opportunity_brier"] - candidate_draws["opportunity_brier"],
            point.at[BASELINE_NAME, "opportunity_brier"] - point.at[MODEL_NAME, "opportunity_brier"],
        ),
        (
            "logloss_improvement",
            "opportunity_logloss",
            baseline_draws["opportunity_logloss"] - candidate_draws["opportunity_logloss"],
            point.at[BASELINE_NAME, "opportunity_logloss"] - point.at[MODEL_NAME, "opportunity_logloss"],
        ),
    )
    rows = []
    for comparison_name, metric_name, values, point_value in definitions:
        finite = values[np.isfinite(values)]
        low, high = np.quantile(finite, [0.025, 0.975])
        rows.append(
            {
                "candidate": MODEL_NAME,
                "baseline": BASELINE_NAME,
                "comparison": comparison_name,
                "metric": metric_name,
                "point_improvement": float(point_value),
                "ci_low": float(low),
                "ci_high": float(high),
                "bootstrap_draws": draws,
            }
        )
    return pd.DataFrame(rows)


def _cluster_ids(frame: pd.DataFrame, mask: pd.Series) -> pd.Series:
    selected = frame.loc[mask].copy()
    if selected.empty:
        return pd.Series(dtype="int64", index=selected.index)
    selected = selected.sort_values(
        ["channel_episode_id", "decision_time", "window_id", "step"], kind="stable"
    )
    time = pd.to_datetime(selected["decision_time"], utc=True)
    new_cluster = (
        selected["channel_episode_id"].ne(selected["channel_episode_id"].shift())
        | time.diff().gt(pd.Timedelta(minutes=5))
    )
    cluster = new_cluster.cumsum().astype(int)
    cluster.index = selected.index
    return cluster


def event_alert_metrics(
    scores: pd.DataFrame,
    *,
    alerts_per_day: float = 1.0,
) -> pd.DataFrame:
    """Event-level diagnostic at a fixed alert count, not a trading threshold."""
    required = {
        "window_id", "channel_episode_id", "step", "decision_time",
        "y_ge_100", "p_ge_100", "tth_100_min",
    }
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"event diagnostic missing columns: {missing}")
    work = scores[list(required)].copy().reset_index(drop=True)
    work["decision_time"] = pd.to_datetime(work["decision_time"], utc=True)
    # Overlapping windows can expose the same episode-time decision more than
    # once. Event metrics count the market opportunity once, keeping the most
    # confident alert and the earliest observed hit for that timestamp.
    work = (
        work.groupby(["channel_episode_id", "decision_time"], as_index=False, sort=False)
        .agg(
            window_id=("window_id", "first"),
            step=("step", "min"),
            y_ge_100=("y_ge_100", "max"),
            p_ge_100=("p_ge_100", "max"),
            tth_100_min=("tth_100_min", "min"),
        )
        .reset_index(drop=True)
    )
    calendar_days = int(
        (work["decision_time"].max().normalize() - work["decision_time"].min().normalize()).days
        + 1
    )
    alert_count = min(len(work), max(1, int(round(alerts_per_day * calendar_days))))
    ranked = work.sort_values(
        ["p_ge_100", "decision_time", "window_id", "step"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    alert_mask = pd.Series(False, index=work.index)
    alert_mask.loc[ranked.index[:alert_count]] = True
    truth_mask = work["y_ge_100"].astype(bool)
    truth_clusters = _cluster_ids(work, truth_mask)
    alert_clusters = _cluster_ids(work, alert_mask)
    work["truth_cluster"] = pd.NA
    work.loc[truth_clusters.index, "truth_cluster"] = truth_clusters
    work["alert_cluster"] = pd.NA
    work.loc[alert_clusters.index, "alert_cluster"] = alert_clusters
    truth_events = int(truth_clusters.nunique())
    recalled = 0
    lead_times = []
    for _, event in work.loc[truth_mask].groupby("truth_cluster", sort=False):
        selected = event.loc[alert_mask.loc[event.index]]
        if selected.empty:
            continue
        recalled += 1
        first_alert = selected.sort_values("decision_time", kind="stable").iloc[0]
        lead_times.append(float(first_alert["tth_100_min"]))
    false_clusters = 0
    for _, event in work.loc[alert_mask].groupby("alert_cluster", sort=False):
        if not event["y_ge_100"].astype(bool).any():
            false_clusters += 1
    return pd.DataFrame(
        [
            {
                "fixed_alerts_per_day": alerts_per_day,
                "calendar_days": calendar_days,
                "selected_alerts": alert_count,
                "actual_alerts_per_day": alert_count / calendar_days,
                "truth_events": truth_events,
                "recalled_truth_events": recalled,
                "event_recall": recalled / truth_events if truth_events else np.nan,
                "alert_clusters": int(alert_clusters.nunique()),
                "false_alert_clusters": false_clusters,
                "false_alert_clusters_per_day": false_clusters / calendar_days,
                "median_lead_time_minutes": float(np.median(lead_times)) if lead_times else np.nan,
                "diagnostic_only": True,
            }
        ]
    )


def _label_tables(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_audit = (
        labels.groupby(["magnitude_bin", "magnitude_class", "magnitude_target_valid"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    valid = labels.loc[labels["magnitude_target_valid"]].copy()
    valid["tth_bucket"] = time_to_hit_bucket(valid["tth_100_min"])
    tth = (
        valid.groupby("tth_bucket", observed=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    tth["share"] = tth["rows"] / len(valid)
    rows = []
    for threshold, start in zip(MAGNITUDE_THRESHOLDS, (1, 2, 3, 4), strict=True):
        positive = valid["magnitude_class"].ge(start)
        rows.append(
            {
                "threshold_b": threshold,
                "class_start": start,
                "positive_rows": int(positive.sum()),
                "valid_rows": len(valid),
                "prevalence": float(positive.mean()),
                "nested_with_previous": True,
            }
        )
    return label_audit, tth, pd.DataFrame(rows)


def run_magnitude_study(
    *,
    stage: str = "dev",
    smoke: bool = False,
    data_root: Path = DEFAULT_DATA_ROOT,
    frozen_j_root: Path = FROZEN_J_ROOT,
    frozen_n_root: Path = FROZEN_N_ROOT,
    run_root: Path = RUN_ROOT,
    config: MagnitudeStudyConfig = MagnitudeStudyConfig(),
) -> MagnitudeRunResult:
    if stage != "dev":
        raise ValueError("Notebook O permits development only; forward and Q2 are sealed")
    protocol = protocol_dict(config, smoke=smoke)
    protocol_hash = _sha_payload(protocol)
    frozen_n = load_frozen_n_artifacts(Path(frozen_n_root))
    if _n_source_hash() != frozen_n.source_hash:
        raise ValueError("current N3 reconstruction source differs from frozen Notebook N")
    frozen_j = load_frozen_j_artifacts(Path(frozen_j_root))
    if frozen_n.frozen_j_run_hash != frozen_j.run_hash:
        raise ValueError("frozen Notebook N and J run identities differ")
    if frozen_n.frozen_j_manifest_sha256 != frozen_j.manifest_sha256:
        raise ValueError("frozen Notebook N and J manifests differ")
    source_hash = _source_hash()
    input_hash = _sha_payload(
        {
            "frozen_n_run_hash": frozen_n.run_hash,
            "frozen_n_protocol_hash": frozen_n.protocol_hash,
            "frozen_n_labels_sha256": frozen_n.labels_sha256,
            "frozen_n_oof_sha256": frozen_n.oof_sha256,
            "frozen_j_run_hash": frozen_j.run_hash,
            "frozen_j_manifest_sha256": frozen_j.manifest_sha256,
        }
    )
    run_hash = _sha_payload(
        {"protocol_hash": protocol_hash, "source_hash": source_hash, "input_hash": input_hash}
    )[:20]
    run_dir = Path(run_root) / run_hash / ("smoke" if smoke else "full")
    identity = {
        "run_hash": run_hash,
        "protocol_hash": protocol_hash,
        "source_hash": source_hash,
        "input_hash": input_hash,
    }
    cached = _validated_completed_summary(run_dir, identity, frozen_n)
    if cached is not None:
        if not smoke:
            _latest(Path(run_root), run_hash, protocol_hash)
        return MagnitudeRunResult(run_dir, cached)
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
            raise AssertionError("Notebook O bounded development inputs changed")
        causal_labels = _align_labels(base, frozen_n.labels)
        frozen_n3 = build_opportunity_dataset(
            base, causal_labels, include_volatility=True
        )
        labels = label_full_path_magnitude(
            base.decisions,
            loaded.minute,
            config.target,
            causal_features=causal_labels,
        )
        dataset = align_magnitude_dataset(frozen_n3, labels)
        assert_magnitude_feature_isolation(dataset.tabular_features)
        store.parquet("magnitude_labels.parquet", labels)
        label_audit, tth_audit, target_nesting = _label_tables(labels)
        store.csv("label_audit.csv", label_audit)
        store.csv("tth_audit.csv", tth_audit)
        store.csv("target_nesting.csv", target_nesting)
        feature_audit = pd.DataFrame(
            {
                "model": MODEL_NAME,
                "feature": dataset.tabular_features,
                "kept": True,
                "causal_source": "exact frozen N3 matrix",
                "future_or_target_field": False,
            }
        )
        store.csv("feature_audit.csv", feature_audit)

        model_config = replace(config.model, xgb_estimators=12) if smoke else config.model
        result = run_magnitude_oof(
            dataset, MagnitudeOOFConfig(model=model_config)
        )
        scores = result.scores
        store.parquet("oof_predictions.parquet", scores)
        store.csv("fold_audit.csv", result.fold_audit)
        store.csv("calibration_audit.csv", result.calibration_audit)
        magnitude_table, cumulative_table = _aggregate_tables(scores)
        store.csv("magnitude_metrics.csv", magnitude_table)
        store.csv("cumulative_metrics.csv", cumulative_table)
        comparison, primary = _baseline_comparison(scores, frozen_n)
        store.parquet("comparison_predictions.parquet", comparison)
        store.csv("primary_comparison.csv", primary)
        draws = 20 if smoke else config.bootstrap_draws
        paired = _paired_primary_bootstrap(
            comparison, primary, draws=draws, seed=config.bootstrap_seed
        )
        store.csv("paired_bootstrap.csv", paired)
        event = event_alert_metrics(
            scores, alerts_per_day=config.diagnostic_alerts_per_day
        )
        store.csv("event_metrics.csv", event)

        barrier_equal = np.allclose(
            pd.to_numeric(labels["adaptive_barrier_bps"], errors="coerce"),
            pd.to_numeric(causal_labels["adaptive_barrier_bps"], errors="coerce"),
            equal_nan=True,
            rtol=0.0,
            atol=1e-6,
        )
        fixed_end = pd.to_datetime(labels["label_end"], utc=True).eq(
            pd.to_datetime(labels["decision_time"], utc=True) + pd.Timedelta(minutes=120)
        ).all()
        leakage = pd.DataFrame(
            [
                {"check": "development boundary", "passed": loaded.max_loaded_timestamp < expected_end, "detail": str(loaded.max_loaded_timestamp)},
                {"check": "forward and Q2 excluded", "passed": True, "detail": "stage rejected before input loading"},
                {"check": "frozen N3 decision calendar", "passed": len(labels) == len(frozen_n3.decisions), "detail": f"{len(labels)} rows"},
                {"check": "frozen causal barrier", "passed": barrier_equal, "detail": "target rebuild did not change B"},
                {"check": "strict full 120m path", "passed": bool(labels.loc[labels.magnitude_target_valid, "path_complete_120m"].all()), "detail": "gaps are censored"},
                {"check": "fixed label end", "passed": fixed_end, "detail": "label_end=t+120m for all rows"},
                {"check": "feature target deny-list", "passed": not feature_audit.future_or_target_field.any(), "detail": f"{len(feature_audit)} causal features"},
                {"check": "episode-disjoint OOF", "passed": result.fold_audit.episode_overlap.eq(0).all(), "detail": "all folds"},
                {"check": "purged live labels", "passed": result.fold_audit.train_label_end_max.le(result.fold_audit.validation_start).all(), "detail": "all folds"},
                {"check": "nested cumulative probabilities", "passed": magnitude_table.monotonic_violation_rate.eq(0.0).all(), "detail": "P2<=P1.5<=P1<=P0.75"},
            ]
        )
        if not leakage["passed"].all():
            failed = leakage.loc[~leakage["passed"], "check"].tolist()
            raise AssertionError(f"Notebook O leakage audit failed: {failed}")
        store.csv("leakage_audit.csv", leakage)

        paired_pass = bool(paired["ci_low"].gt(0.0).all())
        decision = (
            "full-path magnitude head improves the frozen N3 1B projection with paired episode evidence"
            if paired_pass
            else "full-path magnitude head is not promoted over frozen N3 on the primary 1B projection"
        )
        store.json(
            "frozen_protocol.json",
            {
                "source": "validated full development Notebook N3 and its frozen Notebook J manifest",
                "frozen_n_run_hash": frozen_n.run_hash,
                "frozen_n_protocol_hash": frozen_n.protocol_hash,
                "frozen_n_labels_sha256": frozen_n.labels_sha256,
                "frozen_n_oof_sha256": frozen_n.oof_sha256,
                "frozen_j_run_hash": frozen_j.run_hash,
                "frozen_j_manifest_sha256": frozen_j.manifest_sha256,
                "magnitude_labels_sha256": _sha256(run_dir / "magnitude_labels.parquet"),
                "direction_head_trained": False,
                "trading_policy_trained": False,
                "forward_or_lockbox_loaded": False,
            },
        )
        first_minute = int(
            labels.loc[labels["magnitude_target_valid"], "tth_100_min"].eq(1).sum()
        )
        valid_rows = int(labels["magnitude_target_valid"].sum())
        summary = {
            **identity,
            "model": MODEL_NAME,
            "baseline": BASELINE_NAME,
            "decision": decision,
            "primary_predictive_pass": paired_pass,
            "decision_rows": len(labels),
            "valid_full_path_rows": valid_rows,
            "censored_rows": int(len(labels) - valid_rows),
            "feature_count": len(dataset.tabular_features),
            "oof_rows": len(scores),
            "common_baseline_oof_rows": len(comparison),
            "first_minute_1b_hits": first_minute,
            "first_minute_1b_hit_share": first_minute / valid_rows if valid_rows else np.nan,
            "magnitude_metrics": magnitude_table.iloc[0].to_dict(),
            "primary_comparison": primary.set_index("model").to_dict(orient="index"),
            "event_metrics": event.iloc[0].to_dict(),
            "max_loaded_timestamp": loaded.max_loaded_timestamp,
            "read_end_exclusive": loaded.read_end_exclusive,
            "direction_head_trained": False,
            "trading_policy_trained": False,
            "economics_evaluated": False,
            "hawkes_or_evt_used": False,
            "forward_or_lockbox_loaded": False,
        }
        store.json("summary.json", summary)
        missing = [name for name in READER_ARTIFACTS if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"Notebook O artifacts missing: {missing}")
        store.complete(summary)
        if not smoke:
            _latest(Path(run_root), run_hash, protocol_hash)
        return MagnitudeRunResult(run_dir, summary)
    except BaseException as error:
        store.fail(error)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="dev")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--frozen-j-root", type=Path, default=FROZEN_J_ROOT)
    parser.add_argument("--frozen-n-root", type=Path, default=FROZEN_N_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_magnitude_study(
        stage=args.stage,
        smoke=args.smoke,
        data_root=args.data_root,
        frozen_j_root=args.frozen_j_root,
        frozen_n_root=args.frozen_n_root,
        run_root=args.run_root,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASELINE_NAME",
    "FROZEN_N_ROOT",
    "MODEL_NAME",
    "MagnitudeRunResult",
    "MagnitudeStudyConfig",
    "READER_ARTIFACTS",
    "RUN_ROOT",
    "event_alert_metrics",
    "load_frozen_n_artifacts",
    "protocol_dict",
    "run_magnitude_study",
]
