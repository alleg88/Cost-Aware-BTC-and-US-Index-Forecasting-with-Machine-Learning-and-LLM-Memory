"""Side-conditioned first-touch labels resolved on future one-minute bars."""
from __future__ import annotations

import pandas as pd

SL_FIRST = 0
TIMEOUT = 1
TP_FIRST = 2

_RESERVED = {
    "entry_time",
    "outcome_close_time",
    "side",
    "outcome",
    "gross_return",
}


def build_first_touch_candidates(
    minute: pd.DataFrame,
    signals: pd.Series,
    features: pd.DataFrame,
    *,
    tp_bps: float,
    sl_bps: float,
    max_hold: int,
) -> pd.DataFrame:
    """Label every directional M15 signal independently on its future 1m path."""
    if max_hold < 1:
        raise ValueError("max_hold must be >= 1")
    if not isinstance(minute.index, pd.DatetimeIndex) or minute.index.tz is None:
        raise ValueError("minute index must be timezone-aware")
    if not minute.index.is_unique or not minute.index.is_monotonic_increasing:
        raise ValueError("minute index must be unique and sorted")
    collisions = _RESERVED.intersection(features.columns)
    if collisions:
        raise ValueError(f"reserved feature columns: {sorted(collisions)}")

    directional = signals[signals.isin((0, 2))].astype(int)
    missing_features = directional.index.difference(features.index)
    if len(missing_features):
        raise ValueError("features missing for directional signals")

    rows: list[dict] = []
    minutes_needed = max_hold * 15
    one_minute = pd.Timedelta(minutes=1)
    for signal_time, predicted_class in directional.items():
        entry_time = signal_time + pd.Timedelta(minutes=15)
        lo = int(minute.index.searchsorted(entry_time, side="left"))
        hi = lo + minutes_needed
        if lo >= len(minute) or hi > len(minute) or minute.index[lo] != entry_time:
            continue
        path = minute.iloc[lo:hi]
        expected = pd.date_range(entry_time, periods=minutes_needed, freq="1min", tz=minute.index.tz)
        if not path.index.equals(expected):
            continue

        side = 1 if predicted_class == 2 else -1
        entry_price = float(path.iloc[0]["open"])
        tp_price = entry_price * (1.0 + side * float(tp_bps) / 10_000.0)
        sl_price = entry_price * (1.0 - side * float(sl_bps) / 10_000.0)
        outcome = TIMEOUT
        exit_price = float(path.iloc[-1]["close"])
        exit_row = len(path) - 1
        for j, (_, bar) in enumerate(path.iterrows()):
            hit_sl = float(bar["low"]) <= sl_price if side == 1 else float(bar["high"]) >= sl_price
            hit_tp = float(bar["high"]) >= tp_price if side == 1 else float(bar["low"]) <= tp_price
            if hit_sl:
                outcome, exit_price, exit_row = SL_FIRST, sl_price, j
                break
            if hit_tp:
                outcome, exit_price, exit_row = TP_FIRST, tp_price, j
                break

        row = features.loc[signal_time].to_dict()
        row.update(
            {
                "signal_time": signal_time,
                "entry_time": entry_time,
                "outcome_close_time": path.index[exit_row] + one_minute,
                "side": side,
                "outcome": outcome,
                "gross_return": side * (exit_price / entry_price - 1.0),
            }
        )
        rows.append(row)

    if not rows:
        columns = ["entry_time", "outcome_close_time", "side", "outcome", "gross_return", *features.columns]
        return pd.DataFrame(columns=columns).rename_axis("signal_time")
    return pd.DataFrame(rows).set_index("signal_time").sort_index()
