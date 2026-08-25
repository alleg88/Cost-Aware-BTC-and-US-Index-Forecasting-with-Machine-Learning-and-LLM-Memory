"""Event-level model study for Strict-5m and Fast-T2 channel windows."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.channel_window_validation import (
    PurgedFold,
    expanding_purged_folds,
    interval_uniqueness,
)
from evaluation.two_trigger_execution import resolve_fixed_trade


MODEL_NAMES = ("logreg", "catboost", "xgboost", "gru")

FEATURE_COLUMNS = (
    "side_sign",
    "channel_slope_bps_5m_side",
    "channel_r2",
    "channel_width_pct",
    "channel_confluence_count",
    "t1_edge_depth",
    "t1_range_bps",
    "t1_body_fraction",
    "t1_wick_share_side",
    "t1_close_location_side",
    "t1_distance_to_rail_bps",
    "confirmation_lag_minutes",
    "breakout_margin_bps",
    "return_t1_to_t2_side",
    "pre_t2_mfe_bps",
    "pre_t2_mae_bps",
    "realized_vol_15m_bps",
    "return_1m_side",
    "return_5m_side",
    "volume_ratio_5_10",
    "trade_count_ratio_5_10",
    "taker_imbalance_5_side",
    "hour_sin",
    "hour_cos",
    "risk_bps_decision",
    "rr_planned_decision",
)

SEQUENCE_FEATURE_COLUMNS = (
    "return_1m_side",
    "range_bps",
    "body_bps_side",
    "taker_imbalance_side",
    "log_volume",
)


@dataclass(frozen=True)
class EventLabelConfig:
    stop_buffer_bps: float = 5.0
    min_risk_bps: float = 0.0
    max_risk_bps: float | None = None
    min_rr: float = 0.0
    max_hold_minutes: int = 120
    round_trip_cost_bps: float = 10.0
    feature_lookback_minutes: int = 15


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _history(
    minute: pd.DataFrame,
    decision_time: pd.Timestamp,
    length: int,
    *,
    extra: int = 0,
) -> pd.DataFrame | None:
    expected = pd.date_range(
        end=decision_time - pd.Timedelta(minutes=1),
        periods=length + extra,
        freq="1min",
        tz="UTC",
    )
    observed = minute.reindex(expected)
    required = ["open", "high", "low", "close", "volume", "taker_buy_base", "count"]
    if observed[required].isna().any().any():
        return None
    return observed


def _event_features(
    window: object,
    channel: pd.Series,
    previous_channel: pd.Series | None,
    history: pd.DataFrame,
    confirmation_price: float,
    confirmation_lag_minutes: int,
) -> dict[str, float]:
    side_sign = 1.0 if window.side == "long" else -1.0
    lower = float(channel.channel_lower)
    upper = float(channel.channel_upper)
    span = upper - lower
    t1_open, t1_high = float(channel.open), float(window.t1_high)
    t1_low, t1_close = float(window.t1_low), float(channel.close)
    candle_range = t1_high - t1_low
    mid = (lower + upper) / 2.0
    previous_mid = (
        (float(previous_channel.channel_lower) + float(previous_channel.channel_upper)) / 2.0
        if previous_channel is not None else mid
    )
    slope_side = (mid / previous_mid - 1.0) * 10_000.0 * side_sign
    if side_sign > 0:
        edge_depth = (t1_low - lower) / span
        wick_share = (min(t1_open, t1_close) - t1_low) / candle_range
        close_location = (t1_close - t1_low) / candle_range
        rail_distance = (t1_low - lower) / t1_close * 10_000.0
        breakout = (confirmation_price / t1_high - 1.0) * 10_000.0
        stop = t1_low * 0.9995
        target = upper
    else:
        edge_depth = (upper - t1_high) / span
        wick_share = (t1_high - max(t1_open, t1_close)) / candle_range
        close_location = (t1_high - t1_close) / candle_range
        rail_distance = (upper - t1_high) / t1_close * 10_000.0
        breakout = (t1_low / confirmation_price - 1.0) * 10_000.0
        stop = t1_high * 1.0005
        target = lower

    close = history["close"].to_numpy(dtype=float)
    high = history["high"].to_numpy(dtype=float)
    low = history["low"].to_numpy(dtype=float)
    returns = np.diff(np.log(close))
    volume = history["volume"].to_numpy(dtype=float)
    counts = history["count"].to_numpy(dtype=float)
    taker = np.divide(
        2.0 * history["taker_buy_base"].to_numpy(dtype=float),
        volume,
        out=np.ones_like(volume),
        where=volume > 0,
    ) - 1.0
    recent = history[history.index >= _utc(window.t1_time)]
    recent_high = recent["high"].max() if not recent.empty else t1_high
    recent_low = recent["low"].min() if not recent.empty else t1_low
    mfe = ((recent_high / t1_close - 1.0) if side_sign > 0
           else (t1_close / recent_low - 1.0)) * 10_000.0
    mae = ((recent_low / t1_close - 1.0) if side_sign > 0
           else (t1_close / recent_high - 1.0)) * 10_000.0
    risk = (confirmation_price - stop) if side_sign > 0 else (stop - confirmation_price)
    reward = (target - confirmation_price) if side_sign > 0 else (confirmation_price - target)
    hour = _utc(window.t2_time).hour + _utc(window.t2_time).minute / 60.0

    return {
        "side_sign": side_sign,
        "channel_slope_bps_5m_side": slope_side,
        "channel_r2": float(channel.channel_r2),
        "channel_width_pct": span / confirmation_price,
        "channel_confluence_count": float(channel.channel_confluence_count),
        "t1_edge_depth": edge_depth,
        "t1_range_bps": candle_range / t1_close * 10_000.0,
        "t1_body_fraction": abs(t1_close - t1_open) / candle_range,
        "t1_wick_share_side": wick_share,
        "t1_close_location_side": close_location,
        "t1_distance_to_rail_bps": rail_distance,
        "confirmation_lag_minutes": float(confirmation_lag_minutes),
        "breakout_margin_bps": breakout,
        "return_t1_to_t2_side": (confirmation_price / t1_close - 1.0) * 10_000.0 * side_sign,
        "pre_t2_mfe_bps": mfe,
        "pre_t2_mae_bps": mae,
        "realized_vol_15m_bps": float(np.std(returns, ddof=1) * 10_000.0),
        "return_1m_side": float(returns[-1] * 10_000.0 * side_sign),
        "return_5m_side": float(np.log(close[-1] / close[-6]) * 10_000.0 * side_sign),
        "volume_ratio_5_10": float(volume[-5:].mean() / volume[-15:-5].mean()),
        "trade_count_ratio_5_10": float(counts[-5:].mean() / counts[-15:-5].mean()),
        "taker_imbalance_5_side": float(taker[-5:].mean() * side_sign),
        "hour_sin": float(np.sin(2.0 * np.pi * hour / 24.0)),
        "hour_cos": float(np.cos(2.0 * np.pi * hour / 24.0)),
        "risk_bps_decision": risk / confirmation_price * 10_000.0,
        "rr_planned_decision": reward / risk,
    }


def build_event_dataset(
    windows: pd.DataFrame,
    channel_frame: pd.DataFrame,
    minute_bars: pd.DataFrame,
    *,
    arm: str,
    config: EventLabelConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build one causal, independently labelled row per confirmed window."""
    if arm not in {"strict_5m", "fast_t2_2bps"}:
        raise ValueError("arm must be strict_5m or fast_t2_2bps")
    minute = minute_bars.sort_index(kind="stable")
    channel = channel_frame.copy()
    channel["decision_time"] = pd.to_datetime(channel["decision_time"], utc=True)
    channel = channel.set_index("decision_time", drop=False).sort_index()
    audit = {
        "raw_windows": int(len(windows)),
        "missing_context": 0,
        "geometry_rejected": 0,
        "entry_geometry_cancelled": 0,
        "censored": 0,
        "labelled_events": 0,
    }
    rows: list[dict[str, object]] = []
    for window in windows.sort_values("t2_time", kind="stable").itertuples(index=False):
        t1_time, t2_time = _utc(window.t1_time), _utc(window.t2_time)
        entry_time = _utc(window.window_start)
        if t1_time not in channel.index or (arm == "strict_5m" and t2_time not in channel.index):
            audit["missing_context"] += 1
            continue
        t1 = channel.loc[t1_time]
        prior_time = t1_time - pd.Timedelta(minutes=5)
        prior = channel.loc[prior_time] if prior_time in channel.index else None
        history = _history(minute, t2_time, config.feature_lookback_minutes)
        if history is None or entry_time not in minute.index:
            audit["missing_context"] += 1
            continue
        confirmation_price = (
            float(window.confirmation_price)
            if hasattr(window, "confirmation_price")
            else float(channel.loc[t2_time].close)
        )
        lag = (
            int(window.confirmation_lag_minutes)
            if hasattr(window, "confirmation_lag_minutes")
            else int(window.confirmation_lag_bars) * 5
        )
        features = _event_features(window, t1, prior, history, confirmation_price, lag)
        if not np.isfinite(np.array(list(features.values()), dtype=float)).all():
            audit["missing_context"] += 1
            continue
        risk_bps = features["risk_bps_decision"]
        planned_rr = features["rr_planned_decision"]
        risk_ok = risk_bps > 0.0 and risk_bps >= config.min_risk_bps
        if config.max_risk_bps is not None:
            risk_ok = risk_ok and risk_bps <= config.max_risk_bps
        reward_ok = planned_rr > 0.0 and planned_rr >= config.min_rr
        if not (risk_ok and reward_ok):
            audit["geometry_rejected"] += 1
            continue

        side_sign = 1.0 if window.side == "long" else -1.0
        stop = (float(window.t1_low) * (1.0 - config.stop_buffer_bps / 10_000.0)
                if side_sign > 0 else
                float(window.t1_high) * (1.0 + config.stop_buffer_bps / 10_000.0))
        target = float(t1.channel_upper if side_sign > 0 else t1.channel_lower)
        resolution = resolve_fixed_trade(
            minute,
            entry_time=entry_time,
            side=window.side,
            stop=stop,
            target=target,
            max_hold_minutes=config.max_hold_minutes,
            round_trip_cost_bps=config.round_trip_cost_bps,
        )
        if resolution is None:
            audit["censored"] += 1
            continue
        if resolution.outcome == "entry_cancelled":
            audit["entry_geometry_cancelled"] += 1
        entry = resolution.entry_price
        exit_time = resolution.exit_time
        exit_price = resolution.exit_price
        outcome = resolution.outcome
        r_net = resolution.r_net

        row: dict[str, object] = {
            "candidate_id": f"{arm}:{window.side}:{t1_time.isoformat()}",
            "arm": arm,
            "side": window.side,
            "channel_episode_id": int(window.channel_episode_id),
            "t1_time": t1_time,
            "decision_time": t2_time,
            "label_start": entry_time,
            "label_end": exit_time,
            "entry_time": entry_time,
            "entry_price": entry,
            "stop_price": stop,
            "target_price": target,
            "exit_price": exit_price,
            "outcome": outcome,
            "r_net": float(r_net),
            "label_net_positive": int(r_net > 0),
            **features,
        }
        rows.append(row)

    events = pd.DataFrame(rows)
    if not events.empty:
        events = events.sort_values("decision_time", kind="stable").reset_index(drop=True)
    audit["labelled_events"] = int(len(events))
    return events, audit


def build_event_sequences(
    events: pd.DataFrame,
    minute_bars: pd.DataFrame,
    *,
    sequence_length: int = 15,
) -> np.ndarray:
    """Return past-only 1m sequences with static event features repeated."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be >= 2")
    minute = minute_bars.sort_index(kind="stable")
    sequences: list[np.ndarray] = []
    for event in events.itertuples(index=False):
        decision_time = _utc(event.decision_time)
        history = _history(minute, decision_time, sequence_length, extra=1)
        if history is None:
            raise ValueError(f"missing sequence history before {decision_time}")
        side_sign = float(event.side_sign)
        close = history["close"].to_numpy(dtype=float)
        open_ = history["open"].to_numpy(dtype=float)[1:]
        high = history["high"].to_numpy(dtype=float)[1:]
        low = history["low"].to_numpy(dtype=float)[1:]
        volume = history["volume"].to_numpy(dtype=float)[1:]
        taker = history["taker_buy_base"].to_numpy(dtype=float)[1:]
        returns = np.diff(np.log(close)) * 10_000.0 * side_sign
        seq = np.column_stack(
            [
                returns,
                (high - low) / close[1:] * 10_000.0,
                (history["close"].to_numpy(dtype=float)[1:] - open_) / close[1:] * 10_000.0 * side_sign,
                (np.divide(2.0 * taker, volume, out=np.ones_like(volume), where=volume > 0) - 1.0) * side_sign,
                np.log1p(volume),
            ]
        )
        static = np.array([getattr(event, column) for column in FEATURE_COLUMNS], dtype=float)
        repeated = np.repeat(static[None, :], sequence_length, axis=0)
        sequences.append(np.concatenate([seq, repeated], axis=1))
    return np.stack(sequences).astype(np.float32) if sequences else np.empty(
        (0, sequence_length, len(SEQUENCE_FEATURE_COLUMNS) + len(FEATURE_COLUMNS)),
        dtype=np.float32,
    )


def split_statistics(events: pd.DataFrame, folds: list[PurgedFold]) -> pd.DataFrame:
    """Report exact train/validation/purged shares within each fold's history."""
    decision = pd.to_datetime(events["decision_time"], utc=True)
    rows: list[dict[str, object]] = []
    for fold in folds:
        universe = np.flatnonzero((decision < fold.valid_end).to_numpy())
        used = np.union1d(fold.train, fold.valid)
        purged = np.setdiff1d(universe, used)
        total = len(universe)
        rows.append(
            {
                "fold_id": fold.fold_id,
                "train_start": decision.iloc[fold.train].min() if len(fold.train) else pd.NaT,
                "train_end_exclusive": fold.train_end,
                "validation_start": fold.valid_start,
                "validation_end_exclusive": fold.valid_end,
                "total_rows": int(total),
                "train_rows": int(len(fold.train)),
                "validation_rows": int(len(fold.valid)),
                "purged_rows": int(len(purged)),
                "train_pct": 100.0 * len(fold.train) / total if total else np.nan,
                "validation_pct": 100.0 * len(fold.valid) / total if total else np.nan,
                "purged_pct": 100.0 * len(purged) / total if total else np.nan,
                "train_episodes": int(events.iloc[fold.train]["channel_episode_id"].nunique()),
                "validation_episodes": int(events.iloc[fold.valid]["channel_episode_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _balanced_weights(labels: np.ndarray, uniqueness: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int8)
    classes, counts = np.unique(labels, return_counts=True)
    if set(classes) != {0, 1}:
        raise ValueError("training fold needs both binary classes")
    factors = {int(cls): len(labels) / (2.0 * int(count))
               for cls, count in zip(classes, counts, strict=True)}
    return uniqueness * np.array([factors[int(value)] for value in labels], dtype=float)


def _fit_gru(
    train_sequence: np.ndarray,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    validation_sequence: np.ndarray,
    *,
    epochs: int,
) -> np.ndarray:
    import torch
    from torch import nn

    torch.manual_seed(42)
    np.random.seed(42)
    mean = train_sequence.mean(axis=(0, 1), keepdims=True)
    std = train_sequence.std(axis=(0, 1), keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    train_x = ((train_sequence - mean) / std).astype(np.float32)
    valid_x = ((validation_sequence - mean) / std).astype(np.float32)

    class CompactGRU(nn.Module):
        def __init__(self, n_features: int):
            super().__init__()
            self.gru = nn.GRU(n_features, 16, batch_first=True)
            self.head = nn.Linear(16, 1)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            output, _ = self.gru(values)
            return self.head(output[:, -1]).squeeze(1)

    model = CompactGRU(train_x.shape[2])
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x_tensor = torch.from_numpy(train_x)
    y_tensor = torch.from_numpy(train_y.astype(np.float32))
    w_tensor = torch.from_numpy(train_weight.astype(np.float32))
    batch_size = 256
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(x_tensor))
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            optimiser.zero_grad()
            logits = model(x_tensor[index])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y_tensor[index], reduction="none"
            )
            loss = (loss * w_tensor[index]).sum() / w_tensor[index].sum()
            loss.backward()
            optimiser.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(valid_x))).numpy()


def run_oof_models(
    events: pd.DataFrame,
    minute_bars: pd.DataFrame | None,
    *,
    model_names: tuple[str, ...] = MODEL_NAMES,
    valid_blocks: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] | None = None,
    gru_epochs: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run approved models on identical expanding, episode-purged folds."""
    unknown = sorted(set(model_names).difference(MODEL_NAMES))
    if unknown:
        raise ValueError(f"unsupported models: {unknown}")
    required = {
        *FEATURE_COLUMNS, "candidate_id", "side", "decision_time", "label_start",
        "label_end", "channel_episode_id", "label_net_positive", "r_net",
    }
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"events missing model columns: {missing}")
    work = events.sort_values("decision_time", kind="stable").reset_index(drop=True).copy()
    if not work["candidate_id"].is_unique:
        raise ValueError("candidate_id must be unique")
    folds = (
        expanding_purged_folds(work)
        if valid_blocks is None
        else expanding_purged_folds(work, valid_blocks=valid_blocks)
    )
    sequences = None
    if "gru" in model_names:
        if minute_bars is None:
            raise ValueError("GRU requires minute_bars")
        sequences = build_event_sequences(work, minute_bars, sequence_length=15)

    from catboost import CatBoostClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for fold in folds:
        train, valid = fold.train, fold.valid
        if not len(train) or not len(valid):
            continue
        train_y = work.iloc[train]["label_net_positive"].to_numpy(dtype=np.int8)
        if set(np.unique(train_y)) != {0, 1}:
            continue
        uniqueness = interval_uniqueness(work, train)
        weights = _balanced_weights(train_y, uniqueness)
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        train_x = imputer.fit_transform(work.iloc[train][list(FEATURE_COLUMNS)])
        valid_x = imputer.transform(work.iloc[valid][list(FEATURE_COLUMNS)])
        for model_name in model_names:
            if model_name == "logreg":
                scaler = StandardScaler()
                scaled_train = scaler.fit_transform(train_x)
                model = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
                model.fit(scaled_train, train_y, sample_weight=weights)
                score = model.predict_proba(scaler.transform(valid_x))[:, 1]
            elif model_name == "catboost":
                model = CatBoostClassifier(
                    iterations=300,
                    depth=4,
                    learning_rate=0.03,
                    l2_leaf_reg=10.0,
                    loss_function="Logloss",
                    random_seed=42,
                    allow_writing_files=False,
                    verbose=False,
                    thread_count=1,
                )
                model.fit(train_x, train_y, sample_weight=weights)
                score = model.predict_proba(valid_x)[:, 1]
            elif model_name == "xgboost":
                model = XGBClassifier(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.03,
                    min_child_weight=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=10.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=42,
                    n_jobs=1,
                )
                model.fit(train_x, train_y, sample_weight=weights)
                score = model.predict_proba(valid_x)[:, 1]
            else:
                assert sequences is not None
                score = _fit_gru(
                    sequences[train], train_y, weights, sequences[valid], epochs=gru_epochs
                )

            output_columns = [
                "candidate_id", "side", "decision_time", "label_start", "label_end",
                "channel_episode_id", "label_net_positive", "r_net",
            ]
            if "arm" in work:
                output_columns.insert(1, "arm")
            output = work.iloc[valid][output_columns].copy()
            output.insert(0, "model", model_name)
            output.insert(1, "fold_id", fold.fold_id)
            output["score"] = np.asarray(score, dtype=float)
            predictions.append(output)
            overlap = set(work.iloc[train]["channel_episode_id"]).intersection(
                work.iloc[valid]["channel_episode_id"]
            )
            audits.append(
                {
                    "model": model_name,
                    "fold_id": fold.fold_id,
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(valid)),
                    "train_episodes": int(work.iloc[train]["channel_episode_id"].nunique()),
                    "validation_episodes": int(work.iloc[valid]["channel_episode_id"].nunique()),
                    "episode_overlap": int(len(overlap)),
                    "train_positive_pct": float(100.0 * train_y.mean()),
                    "validation_positive_pct": float(
                        100.0 * work.iloc[valid]["label_net_positive"].mean()
                    ),
                    "uniqueness_mean": float(uniqueness.mean()),
                }
            )
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return prediction_frame, pd.DataFrame(audits)


def summarise_oof_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarise discrimination and fixed-rule economics for each model/arm."""
    if predictions.empty:
        return pd.DataFrame()
    from sklearn.metrics import brier_score_loss, roc_auc_score

    work = predictions.copy()
    group_columns = ["model"] + (["arm"] if "arm" in work else [])
    work["selected_05"] = work["score"] >= 0.5
    work["top30"] = work.groupby(group_columns + ["fold_id"])["score"].transform(
        lambda values: values >= values.quantile(0.70)
    )
    rows: list[dict[str, object]] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, group in work.groupby(grouper, sort=False):
        keys = (key,) if not isinstance(key, tuple) else key
        selected = group[group["selected_05"]]
        top = group[group["top30"]]
        target = group["label_net_positive"].to_numpy(dtype=int)
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "oof_rows": int(len(group)),
                "positive_pct": float(100.0 * target.mean()),
                "roc_auc": float(roc_auc_score(target, group["score"]))
                if len(np.unique(target)) == 2 else np.nan,
                "brier": float(brier_score_loss(target, group["score"])),
                "take_all_mean_net_r": float(group["r_net"].mean()),
                "take_all_total_net_r": float(group["r_net"].sum()),
                "selected_rows": int(len(selected)),
                "selected_mean_net_r": float(selected["r_net"].mean()) if len(selected) else np.nan,
                "selected_total_net_r": float(selected["r_net"].sum()),
                "top30_rows": int(len(top)),
                "top30_mean_net_r": float(top["r_net"].mean()) if len(top) else np.nan,
                "top30_total_net_r": float(top["r_net"].sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
