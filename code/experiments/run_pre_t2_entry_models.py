"""Registered dev-only runner for Notebook I pre-T2 entry models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.channel_window_validation import (
    effective_sample_size,
    expanding_purged_folds,
    interval_uniqueness,
)
from evaluation.fast_t2_action_policy import replay_entry_capacity
from evaluation.fast_t2_causal_ev_policy import expected_net_r
from evaluation.pre_t2_entry_policy import first_positive_lifecycle_entries
from experiments.fast_t2_causal_ev_models import apply_temperature, select_temperature
from experiments.five_minute_two_trigger_windows import TwoTriggerConfig
from experiments.pre_t2_feature_engineering import (
    fold_correlation_filter,
    profile_features,
)
from experiments.pre_t2_ev_models import ENSEMBLE_NAME, MODEL_NAMES
from experiments.pre_t2_ev_models import (
    PreT2Prediction,
    combine_predictions,
    fit_predict_pre_t2_model,
)
from experiments.pre_t2_lifecycle_dataset import (
    BASE_FEATURE_COLUMNS,
    SEQUENCE_FEATURE_COLUMNS,
    LifecycleDecisionConfig,
    build_lifecycle_decisions,
    detect_t1_lifecycles,
)
from experiments.run_fast_t2_entry_policy import inner_episode_purged_indices


CODE_ROOT = Path(__file__).resolve().parents[1]
OUT = CODE_ROOT / "experiments" / "cache" / "pre_t2_entry_models" / "dev"
CHANNEL_SOURCE = (
    CODE_ROOT / "experiments" / "cache" / "channel_5m_two_trigger"
    / "fast_t2" / "channel_context.parquet"
)
MINUTE_SOURCE = CODE_ROOT / "data" / "btcusdt_1m_2021_2026.parquet"
BASELINE_SOURCE = (
    CODE_ROOT / "experiments" / "cache" / "fast_t2_causal_ev" / "dev"
    / "geometry_ablation.csv"
)
DEV_START = pd.Timestamp("2021-01-01", tz="UTC")
DEV_END = pd.Timestamp("2025-07-01", tz="UTC")
MODELS = MODEL_NAMES
SCORERS = (*MODEL_NAMES, ENSEMBLE_NAME)
GRU_EPOCHS = 8

SCORE_COLUMNS = (
    "arm_id",
    "window_id",
    "decision_id",
    "side",
    "channel_episode_id",
    "lifecycle_status",
    "decision_phase",
    "t1_time",
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
    "outcome_class",
    "r_net",
    "observed_gross_r",
    "planned_cost_r_10bps",
    "planned_tp_gross_r",
    "filled",
    "holding_minutes",
    "stop_invalidated_before_decision",
    "distance_to_stop_bps",
    "distance_to_target_bps",
    "rr_proxy",
)


def _hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_protocol(source: dict[str, object]) -> dict[str, object]:
    """Freeze the Notebook I design before loading any sealed period."""
    if bool(source.get("forward_or_lockbox_loaded")):
        raise ValueError("forward and Q2-2026 lockbox must remain sealed")
    end = pd.Timestamp(source["period_end_exclusive"])
    maximum = pd.Timestamp(source["max_loaded_timestamp"])
    if maximum >= end:
        raise ValueError("source reaches the sealed period")
    payload: dict[str, object] = {
        "stage": "development",
        "period_start": source["period_start"],
        "period_end_exclusive": source["period_end_exclusive"],
        "channel_source_hash": source["channel_source_hash"],
        "minute_source_hash": source["minute_source_hash"],
        "models": list(MODELS),
        "ensemble": ENSEMBLE_NAME,
        "model_hyperparameters": {
            "xgboost_estimators": 200,
            "gru_hidden_units": 16,
            "gru_epochs": GRU_EPOCHS,
            "ensemble_weights": [0.5, 0.5],
        },
        "tabular_features": list(BASE_FEATURE_COLUMNS),
        "tabular_feature_count": len(BASE_FEATURE_COLUMNS),
        "sequence_features": list(SEQUENCE_FEATURE_COLUMNS),
        "sequence_shape": [30, len(SEQUENCE_FEATURE_COLUMNS)],
        "include_expired_t1": True,
        "pre_t2_decisions": "every completed 1m boundary from T1 until confirmation or expiry",
        "post_t2_decision_minutes": 15,
        "primary_phase": "all",
        "phase_diagnostics": ["pre_t2", "post_t2"],
        "primary_min_risk_bps": 25.0,
        "geometry": "frozen structural stop and opposite channel rail",
        "max_hold_minutes": 120,
        "round_trip_cost_bps": 10.0,
        "entry_rule": "first strict predicted EV > 0; no threshold tuning",
        "correlation_filter": "training-fold Spearman >=0.95 drops lower-priority feature",
        "validation": "seven expanding six-month episode-purged folds",
        "uniqueness_weighting": "training labels only",
        "minimum_trades_per_calendar_day": 1.0,
        "economic_success": "mean net R >=0.05, episode-bootstrap low >0, and beats frozen baseline",
        "frozen_baseline": "Notebook H causal LogReg + 25 bps",
        "forward_or_lockbox_loaded": False,
    }
    payload["protocol_hash"] = _hash(payload)
    return payload


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_minutes() -> pd.DataFrame:
    minute = pd.read_parquet(
        MINUTE_SOURCE,
        columns=["open", "high", "low", "close", "volume", "taker_buy_base", "count"],
        filters=[("timestamp", ">=", DEV_START), ("timestamp", "<", DEV_END)],
    ).sort_index(kind="stable")
    minute = minute[(minute.index >= DEV_START) & (minute.index < DEV_END)]
    if not isinstance(minute.index, pd.DatetimeIndex) or minute.index.tz is None:
        raise ValueError("one-minute source must have a timezone-aware index")
    if minute.index.has_duplicates:
        raise ValueError("one-minute source contains duplicate timestamps")
    if minute.index.max() >= DEV_END:
        raise AssertionError("one-minute source reaches the sealed period")
    return minute


def _source_manifest() -> dict[str, object]:
    for path in (CHANNEL_SOURCE, MINUTE_SOURCE, BASELINE_SOURCE):
        if not path.exists():
            raise FileNotFoundError(path)
    minute_tail = pd.read_parquet(
        MINUTE_SOURCE,
        columns=["close"],
        filters=[("timestamp", ">=", DEV_START), ("timestamp", "<", DEV_END)],
    )
    maximum = pd.Timestamp(minute_tail.index.max()).tz_convert("UTC")
    return {
        "period_start": DEV_START.isoformat(),
        "period_end_exclusive": DEV_END.isoformat(),
        "max_loaded_timestamp": maximum.isoformat(),
        "channel_source_hash": _file_hash(CHANNEL_SOURCE),
        "minute_source_hash": _file_hash(MINUTE_SOURCE),
        "forward_or_lockbox_loaded": False,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_protocol(output_dir: Path, source: dict[str, object], *, rebuild: bool) -> dict[str, object]:
    expected = build_protocol(source)
    path = output_dir / "protocol.json"
    if path.exists() and not rebuild:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("protocol_hash") != expected["protocol_hash"]:
            raise ValueError("Notebook I protocol changed; use --rebuild intentionally")
        return stored
    _write_json(path, expected)
    return expected


_FEATURE_DESCRIPTIONS = {
    "side_sign": "Long = +1; short = -1.",
    "channel_slope_bps_5m_side": "Causal channel-midpoint slope, aligned to trade side.",
    "channel_r2": "Frozen T1 channel fit quality.",
    "channel_width_pct": "Frozen channel span divided by its midpoint.",
    "channel_confluence_count": "Number of agreeing 60/90/120 channel windows.",
    "channel_position_side": "Completed close position from the relevant channel edge.",
    "channel_age_minutes": "Minutes in the causal channel episode at T1.",
    "t1_edge_depth": "T1 penetration beyond the relevant frozen rail.",
    "t1_range_bps": "T1 high-low range in basis points.",
    "t1_body_fraction": "Absolute T1 body divided by range.",
    "t1_wick_share_side": "Rejection wick share on the trade side.",
    "t1_close_location_side": "T1 close location aligned to reversal direction.",
    "trigger_margin_bps_side": "Completed close relative to the frozen T2 threshold.",
    "trigger_velocity_1m_bps_side": "Latest completed one-minute return toward confirmation.",
    "trigger_velocity_3m_bps_side": "Three-minute completed return toward confirmation.",
    "t2_confirmed": "One only after T2 is already observed.",
    "minutes_since_t1": "Completed minutes elapsed since T1.",
    "minutes_since_t2": "Completed minutes since T2; -1 while unconfirmed.",
    "consecutive_closes_toward_trigger": "Trailing positive close changes toward T2.",
    "price_from_t1_bps_side": "Completed close move from T1 close, side aligned.",
    "since_t1_mfe_bps": "Past-only favourable excursion since T1.",
    "since_t1_mae_bps": "Past-only adverse excursion since T1.",
    "realized_vol_30m_bps": "Volatility of the previous 30 completed one-minute bars.",
    "distance_to_stop_bps": "Frozen structural-stop distance from completed close.",
    "distance_to_target_bps": "Opposite frozen-rail distance from completed close.",
    "rr_proxy": "Planned target distance divided by stop distance.",
    "atr_15m_bps": "Mean true range over 15 completed one-minute bars.",
    "return_1m_side": "Latest completed return aligned to side.",
    "return_5m_side": "Five-minute completed return aligned to side.",
    "volume_ratio_5_20": "Recent five-minute volume versus prior twenty-minute mean.",
    "trade_count_ratio_5_20": "Recent trade count versus prior twenty-minute mean.",
    "taker_imbalance_5_side": "Five-minute taker imbalance aligned to side.",
    "hour_sin": "Cyclical UTC hour sine.",
    "hour_cos": "Cyclical UTC hour cosine.",
}


def _write_feature_artifacts(decisions: pd.DataFrame, output_dir: Path) -> None:
    dictionary = pd.DataFrame(
        [
            {
                "feature": feature,
                "description": _FEATURE_DESCRIPTIONS[feature],
                "timing": "known at decision boundary from completed bars",
                "model": "XGBoost tabular",
            }
            for feature in BASE_FEATURE_COLUMNS
        ]
    )
    dictionary.to_csv(output_dir / "feature_dictionary.csv", index=False)
    profile_features(decisions, BASE_FEATURE_COLUMNS).to_csv(
        output_dir / "feature_profile.csv", index=False
    )
    decisions.loc[:, BASE_FEATURE_COLUMNS].corr(method="spearman").to_csv(
        output_dir / "feature_correlation_matrix.csv"
    )


def prepare_dataset(
    *, output_dir: Path = OUT, protocol: dict[str, object], rebuild: bool = False
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    """Build or resume the immutable T1-lifecycle decision ledger."""
    decision_path = output_dir / "pre_t2_decisions.parquet"
    sequence_path = output_dir / "pre_t2_sequences.npy"
    lifecycle_path = output_dir / "t1_lifecycles.parquet"
    audit_path = output_dir / "lifecycle_dataset_audit.json"
    if all(path.exists() for path in (decision_path, sequence_path, lifecycle_path, audit_path)) and not rebuild:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("protocol_hash") != protocol["protocol_hash"]:
            raise ValueError("cached lifecycle dataset belongs to another protocol")
        decisions = pd.read_parquet(decision_path)
        sequences = np.load(sequence_path, mmap_mode="r")
        if len(decisions) != len(sequences):
            raise ValueError("cached decisions and sequences are misaligned")
        return decisions, sequences, audit

    channel = pd.read_parquet(CHANNEL_SOURCE)
    channel["decision_time"] = pd.to_datetime(channel["decision_time"], utc=True)
    channel = channel[
        channel["decision_time"].ge(DEV_START)
        & channel["decision_time"].le(DEV_END)
    ].copy()
    minute = _load_minutes()
    lifecycles, detection_audit = detect_t1_lifecycles(
        channel,
        minute,
        config=TwoTriggerConfig(cooldown_minutes=60),
        confirmation_buffer_bps=2.0,
    )
    if detection_audit["armed_t1"] != 15_258 or detection_audit["confirmed_t2"] != 8_668:
        raise AssertionError(f"frozen Fast-T2 lifecycle count drift: {detection_audit}")
    decisions, sequences, decision_audit = build_lifecycle_decisions(
        lifecycles,
        channel,
        minute,
        LifecycleDecisionConfig(),
    )
    if decisions.empty:
        raise ValueError("Notebook I lifecycle dataset is empty")
    if len(decisions) != len(sequences):
        raise AssertionError("decision and sequence rows are misaligned")
    if pd.to_datetime(decisions["label_end"], utc=True).max() >= DEV_END:
        raise AssertionError("Notebook I labels reach the sealed period")
    if not np.isfinite(decisions.loc[:, BASE_FEATURE_COLUMNS].to_numpy(float)).all():
        raise ValueError("Notebook I tabular features contain non-finite values")
    if not np.isfinite(sequences).all():
        raise ValueError("Notebook I sequences contain non-finite values")
    if not lifecycles["status"].eq("expired").any():
        raise AssertionError("expired T1 arms were lost")

    lifecycles.to_parquet(lifecycle_path, index=False)
    decisions.to_parquet(decision_path, index=False)
    np.save(sequence_path, sequences)
    audit: dict[str, object] = {
        "protocol_hash": protocol["protocol_hash"],
        "detection": detection_audit,
        "decisions": decision_audit,
        "rows": int(len(decisions)),
        "arms": int(decisions["arm_id"].nunique()),
        "confirmed_arms_with_rows": int(
            decisions.loc[decisions["lifecycle_status"].eq("confirmed"), "arm_id"].nunique()
        ),
        "expired_arms_with_rows": int(
            decisions.loc[decisions["lifecycle_status"].eq("expired"), "arm_id"].nunique()
        ),
        "pre_t2_rows": int(decisions["decision_phase"].eq("pre_t2").sum()),
        "post_t2_rows": int(decisions["decision_phase"].eq("post_t2").sum()),
        "decision_ledger_hash": _hash(decisions["decision_id"].astype(str).tolist()),
        "sequence_shape": list(sequences.shape),
        "max_label_end": pd.Timestamp(decisions["label_end"].max()).isoformat(),
        "forward_or_lockbox_loaded": False,
    }
    _write_json(audit_path, audit)
    _write_feature_artifacts(decisions, output_dir)
    return decisions, sequences, audit


def _selection(
    decisions: pd.DataFrame, positions: np.ndarray, *, fold_id: str, scope: str
) -> tuple[tuple[str, ...], pd.DataFrame]:
    selection = fold_correlation_filter(
        decisions.iloc[positions], feature_order=BASE_FEATURE_COLUMNS
    )
    audit = selection.audit.copy()
    audit.insert(0, "scope", scope)
    audit.insert(0, "fold_id", fold_id)
    audit["selected_feature_count"] = len(selection.selected_features)
    audit["removed_features"] = ";".join(selection.removed_features)
    if audit.empty:
        audit = pd.DataFrame(
            [
                {
                    "fold_id": fold_id,
                    "scope": scope,
                    "feature_a": "",
                    "feature_b": "",
                    "abs_spearman": np.nan,
                    "action": "none",
                    "selected_feature_count": len(selection.selected_features),
                    "removed_features": "",
                }
            ]
        )
    return selection.selected_features, audit


def _fit_one(
    decisions: pd.DataFrame,
    sequences: np.ndarray,
    *,
    model_name: str,
    train: np.ndarray,
    valid: np.ndarray,
    features: tuple[str, ...],
    epochs: int,
) -> tuple[PreT2Prediction, np.ndarray]:
    weights = interval_uniqueness(decisions, train, normalize=False)
    weights = weights / weights.mean()
    prediction = fit_predict_pre_t2_model(
        model_name,
        decisions.iloc[train].loc[:, features].to_numpy(float),
        np.asarray(sequences[train]),
        decisions.iloc[train]["outcome_class"].to_numpy(np.int64),
        decisions.iloc[train]["observed_gross_r"].to_numpy(float),
        weights,
        decisions.iloc[valid].loc[:, features].to_numpy(float),
        np.asarray(sequences[valid]),
        epochs=epochs,
    )
    return prediction, weights


def _score_prediction(
    decisions: pd.DataFrame,
    valid: np.ndarray,
    prediction: PreT2Prediction,
    *,
    scorer: str,
    fold_id: str,
    temperature: float,
) -> pd.DataFrame:
    scored = decisions.iloc[valid].loc[:, SCORE_COLUMNS].copy()
    scored.insert(0, "scorer", scorer)
    scored.insert(1, "fold_id", fold_id)
    probabilities = prediction.outcome_probabilities
    scored["p_sl"] = probabilities[:, 0]
    scored["p_tp"] = probabilities[:, 1]
    scored["p_timeout"] = probabilities[:, 2]
    scored["timeout_gross_prediction"] = prediction.timeout_gross_r
    scored["temperature"] = float(temperature)
    scored["ev_score"] = expected_net_r(
        probabilities,
        planned_tp_gross_r=scored["planned_tp_gross_r"].to_numpy(float),
        timeout_gross_r=prediction.timeout_gross_r,
        planned_cost_r=scored["planned_cost_r_10bps"].to_numpy(float),
    )
    scored["score"] = scored["ev_score"]
    return scored.reset_index(drop=True)


def _predictive_row(
    scored: pd.DataFrame,
    *,
    train_rows: int,
    train_episodes: int,
    validation_episodes: int,
    effective_rows: float,
    selected_features: int,
) -> dict[str, object]:
    from sklearn.metrics import log_loss, roc_auc_score

    labels = scored["outcome_class"].to_numpy(np.int64)
    probabilities = scored[["p_sl", "p_tp", "p_timeout"]].to_numpy(float)
    one_hot = np.eye(3)[labels]
    try:
        auc = float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro"))
    except ValueError:
        auc = np.nan
    return {
        "scorer": scored["scorer"].iloc[0],
        "fold_id": scored["fold_id"].iloc[0],
        "train_rows": int(train_rows),
        "validation_rows": int(len(scored)),
        "train_episodes": int(train_episodes),
        "validation_episodes": int(validation_episodes),
        "episode_overlap": 0,
        "effective_train_rows": float(effective_rows),
        "selected_tabular_features": int(selected_features),
        "temperature": float(scored["temperature"].iloc[0]),
        "multiclass_log_loss": float(log_loss(labels, probabilities, labels=[0, 1, 2])),
        "multiclass_brier": float(np.mean(np.square(probabilities - one_hot).sum(axis=1))),
        "macro_ovr_auc_diagnostic_only": auc,
        "ev_rank_ic_diagnostic_only": float(scored["ev_score"].corr(scored["r_net"], method="spearman")),
    }


def _fold_paths(output_dir: Path, fold_id: str, scorer: str) -> tuple[Path, Path]:
    safe = scorer.replace("_", "-")
    return (
        output_dir / "folds" / f"scores_{fold_id}_{safe}.parquet",
        output_dir / "folds" / f"audit_{fold_id}_{safe}.json",
    )


def _run_fold(
    decisions: pd.DataFrame,
    sequences: np.ndarray,
    fold: object,
    *,
    output_dir: Path,
    rebuild: bool,
) -> list[pd.DataFrame]:
    correlation_audits: list[pd.DataFrame] = []
    outer_features, outer_audit = _selection(
        decisions, fold.train, fold_id=fold.fold_id, scope="outer_train"
    )
    correlation_audits.append(outer_audit)
    inner_cut = fold.valid_start - pd.DateOffset(months=6)
    inner_fit, inner_valid = inner_episode_purged_indices(
        decisions,
        fold.train,
        inner_cut=inner_cut,
        outer_valid_start=fold.valid_start,
    )
    inner_features, inner_audit = _selection(
        decisions, inner_fit, fold_id=fold.fold_id, scope="inner_fit"
    )
    correlation_audits.append(inner_audit)
    model_predictions: dict[str, PreT2Prediction] = {}
    temperatures: dict[str, float] = {}
    model_scores: dict[str, pd.DataFrame] = {}
    for model_name in MODELS:
        score_path, audit_path = _fold_paths(output_dir, fold.fold_id, model_name)
        if score_path.exists() and audit_path.exists() and not rebuild:
            score = pd.read_parquet(score_path)
            model_scores[model_name] = score
            model_predictions[model_name] = PreT2Prediction(
                score[["p_sl", "p_tp", "p_timeout"]].to_numpy(float),
                score["timeout_gross_prediction"].to_numpy(float),
                float(score["timeout_gross_prediction"].min()),
                float(score["timeout_gross_prediction"].max()),
            )
            temperatures[model_name] = float(score["temperature"].iloc[0])
            continue
        inner_prediction, _ = _fit_one(
            decisions,
            sequences,
            model_name=model_name,
            train=inner_fit,
            valid=inner_valid,
            features=inner_features,
            epochs=GRU_EPOCHS,
        )
        calibration_weights = interval_uniqueness(decisions, inner_valid, normalize=False)
        temperature = select_temperature(
            inner_prediction.outcome_probabilities,
            decisions.iloc[inner_valid]["outcome_class"].to_numpy(np.int64),
            calibration_weights,
        )
        outer_prediction, train_weights = _fit_one(
            decisions,
            sequences,
            model_name=model_name,
            train=fold.train,
            valid=fold.valid,
            features=outer_features,
            epochs=GRU_EPOCHS,
        )
        calibrated = PreT2Prediction(
            apply_temperature(outer_prediction.outcome_probabilities, temperature),
            outer_prediction.timeout_gross_r,
            outer_prediction.timeout_target_low,
            outer_prediction.timeout_target_high,
        )
        score = _score_prediction(
            decisions,
            fold.valid,
            calibrated,
            scorer=model_name,
            fold_id=fold.fold_id,
            temperature=temperature,
        )
        overlap = set(decisions.iloc[fold.train]["channel_episode_id"]).intersection(
            decisions.iloc[fold.valid]["channel_episode_id"]
        )
        if overlap:
            raise AssertionError("outer fold shares channel episodes")
        audit = _predictive_row(
            score,
            train_rows=len(fold.train),
            train_episodes=decisions.iloc[fold.train]["channel_episode_id"].nunique(),
            validation_episodes=decisions.iloc[fold.valid]["channel_episode_id"].nunique(),
            effective_rows=effective_sample_size(train_weights),
            selected_features=len(outer_features),
        )
        score.to_parquet(score_path, index=False)
        _write_json(audit_path, audit)
        model_scores[model_name] = score
        model_predictions[model_name] = calibrated
        temperatures[model_name] = temperature

    ensemble_path, ensemble_audit_path = _fold_paths(output_dir, fold.fold_id, ENSEMBLE_NAME)
    if not (ensemble_path.exists() and ensemble_audit_path.exists() and not rebuild):
        ensemble_prediction = combine_predictions(
            model_predictions["xgboost"], model_predictions["gru"]
        )
        ensemble_score = _score_prediction(
            decisions,
            fold.valid,
            ensemble_prediction,
            scorer=ENSEMBLE_NAME,
            fold_id=fold.fold_id,
            temperature=1.0,
        )
        train_weights = interval_uniqueness(decisions, fold.train, normalize=False)
        audit = _predictive_row(
            ensemble_score,
            train_rows=len(fold.train),
            train_episodes=decisions.iloc[fold.train]["channel_episode_id"].nunique(),
            validation_episodes=decisions.iloc[fold.valid]["channel_episode_id"].nunique(),
            effective_rows=effective_sample_size(train_weights),
            selected_features=len(outer_features),
        )
        ensemble_score.to_parquet(ensemble_path, index=False)
        _write_json(ensemble_audit_path, audit)
    return correlation_audits


def _evaluation_days(folds: list[object]) -> int:
    return int(sum((fold.valid_end - fold.valid_start).days for fold in folds))


def _baseline() -> dict[str, float | int]:
    table = pd.read_csv(BASELINE_SOURCE)
    row = table[table["step"].eq("Causal stop + minimum risk 25 bps")]
    if len(row) != 1:
        raise ValueError("Notebook H causal LogReg +25 baseline is missing")
    return row.iloc[0].to_dict()


_SCORER_LABELS = {
    "xgboost": "XGBoost",
    "gru": "GRU",
    ENSEMBLE_NAME: "XGBoost + GRU 50/50",
}


def _consolidate(
    decisions: pd.DataFrame,
    folds: list[object],
    *,
    output_dir: Path,
    protocol: dict[str, object],
) -> None:
    days = _evaluation_days(folds)
    baseline = _baseline()
    baseline_mean = float(baseline["mean_net_r"])
    summaries: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for scorer in SCORERS:
        scores = pd.concat(
            [pd.read_parquet(_fold_paths(output_dir, fold.fold_id, scorer)[0]) for fold in folds],
            ignore_index=True,
        ).sort_values(["decision_time", "decision_id"], kind="stable")
        scores.to_parquet(output_dir / f"oof_scores_{scorer}.parquet", index=False)
        for fold in folds:
            diagnostics.append(
                json.loads(_fold_paths(output_dir, fold.fold_id, scorer)[1].read_text(encoding="utf-8"))
            )
        for phase in ("all", "pre_t2", "post_t2"):
            entries, actions = first_positive_lifecycle_entries(scores, phase=phase)
            entries.to_parquet(output_dir / f"oof_entries_{scorer}_{phase}.parquet", index=False)
            actions.to_parquet(output_dir / f"oof_actions_{scorer}_{phase}.parquet", index=False)
            replay = replay_entry_capacity(
                entries,
                capacity=None,
                evaluation_days=days,
                bootstrap_reps=2_000,
            )
            replay.orders.to_parquet(output_dir / f"orders_{scorer}_{phase}.parquet", index=False)
            metrics = replay.metrics
            row = {
                "scorer": _SCORER_LABELS[scorer],
                "scorer_key": scorer,
                "phase": phase,
                **metrics,
                "improvement_vs_frozen_baseline_r": float(metrics["mean_net_r"] - baseline_mean),
                "frequency_eligible": bool(metrics["trades_per_day"] >= 1.0),
                "economic_success": bool(
                    metrics["trades_per_day"] >= 1.0
                    and np.isfinite(metrics["mean_net_r"])
                    and metrics["mean_net_r"] >= 0.05
                    and metrics["bootstrap_low"] > 0.0
                    and metrics["mean_net_r"] > baseline_mean
                ),
            }
            phase_rows.append(row)
            if phase == "all":
                summaries.append(row)
    baseline_row = {
        "scorer": "Frozen Notebook H causal LogReg + 25 bps",
        "scorer_key": "notebook_h_logreg_25bps",
        "phase": "all",
        **{name: baseline.get(name, np.nan) for name in (
            "submitted_orders", "filled_trades", "trades_per_day", "median_daily_trades",
            "zero_trade_days", "days_one", "days_two", "days_three_plus", "long_trades",
            "short_trades", "mean_concurrent", "max_concurrent", "capacity_skips",
            "mean_net_r", "total_net_r", "win_rate", "mean_holding_minutes",
            "max_drawdown_r", "bootstrap_low", "bootstrap_high",
        )},
        "improvement_vs_frozen_baseline_r": 0.0,
        "frequency_eligible": bool(float(baseline["trades_per_day"]) >= 1.0),
        "economic_success": False,
    }
    summary = pd.DataFrame([baseline_row, *summaries])
    summary.to_csv(output_dir / "pre_t2_policy_summary.csv", index=False)
    pd.DataFrame(phase_rows).to_csv(output_dir / "phase_ablation.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output_dir / "predictive_diagnostics.csv", index=False)
    primary = pd.DataFrame(summaries).sort_values(
        ["mean_net_r", "bootstrap_low", "scorer_key"],
        ascending=[False, False, True],
        kind="stable",
    )
    leader = primary.iloc[0]
    result = {
        "protocol_hash": protocol["protocol_hash"],
        "development_leader": str(leader["scorer"]),
        "leader_filled_trades": int(leader["filled_trades"]),
        "leader_trades_per_day": float(leader["trades_per_day"]),
        "leader_mean_net_r": float(leader["mean_net_r"]),
        "leader_bootstrap_low": float(leader["bootstrap_low"]),
        "leader_bootstrap_high": float(leader["bootstrap_high"]),
        "frozen_baseline_mean_net_r": baseline_mean,
        "beats_frozen_baseline": bool(leader["mean_net_r"] > baseline_mean),
        "frequency_success": bool(leader["trades_per_day"] >= 1.0),
        "economic_success": bool(leader["economic_success"]),
        "plain_language_conclusion": (
            "The development leader passes every registered gate; this does not open forward or Q2-2026."
            if bool(leader["economic_success"])
            else "No model passes all registered economic, uncertainty, frequency, and baseline gates; forward and Q2-2026 stay sealed."
        ),
        "forward_or_lockbox_loaded": False,
    }
    _write_json(output_dir / "pre_t2_result.json", result)


def run(*, output_dir: Path = OUT, rebuild: bool = False) -> None:
    """Run the complete resumable Notebook I development experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "folds").mkdir(parents=True, exist_ok=True)
    source = _source_manifest()
    protocol = _write_protocol(output_dir, source, rebuild=rebuild)
    _write_json(
        output_dir / "run_state.json",
        {
            "status": "running",
            "stage": "dataset",
            "protocol_hash": protocol["protocol_hash"],
            "forward_or_lockbox_loaded": False,
        },
    )
    decisions, sequences, dataset_audit = prepare_dataset(
        output_dir=output_dir, protocol=protocol, rebuild=rebuild
    )
    folds = expanding_purged_folds(decisions)
    if len(folds) != 7:
        raise AssertionError(f"expected seven folds, found {len(folds)}")
    correlation_audits: list[pd.DataFrame] = []
    for position, fold in enumerate(folds, start=1):
        print(
            f"[{position}/7] {fold.fold_id}: train={len(fold.train):,}, valid={len(fold.valid):,}",
            flush=True,
        )
        _write_json(
            output_dir / "run_state.json",
            {
                "status": "running",
                "stage": "walk_forward",
                "active_fold": fold.fold_id,
                "completed_folds": position - 1,
                "protocol_hash": protocol["protocol_hash"],
                "forward_or_lockbox_loaded": False,
            },
        )
        correlation_audits.extend(
            _run_fold(
                decisions,
                sequences,
                fold,
                output_dir=output_dir,
                rebuild=rebuild,
            )
        )
    pd.concat(correlation_audits, ignore_index=True).to_csv(
        output_dir / "fold_correlation_audit.csv", index=False
    )
    _consolidate(decisions, folds, output_dir=output_dir, protocol=protocol)
    _write_json(
        output_dir / "run_state.json",
        {
            "status": "complete",
            "stage": "consolidated",
            "completed_folds": 7,
            "dataset_rows": int(dataset_audit["rows"]),
            "protocol_hash": protocol["protocol_hash"],
            "max_loaded_timestamp": source["max_loaded_timestamp"],
            "forward_or_lockbox_loaded": False,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    run(output_dir=args.output_dir, rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
