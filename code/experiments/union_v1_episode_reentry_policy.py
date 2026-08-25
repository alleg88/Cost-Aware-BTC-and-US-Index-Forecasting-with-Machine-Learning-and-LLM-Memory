"""Paired Union-v1-style control and one-extra episode re-entry policy."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.trades_intrabar import simulate_bracket_trades_intrabar


BAR_INTERVAL = pd.Timedelta(minutes=15)
TP_BPS = 200.0
SL_BPS = 100.0
MAX_HOLD_BARS = 1
FEE_BPS_PER_SIDE = 5.0


@dataclass
class PairedReplay:
    control_ledger: pd.DataFrame
    reentry_ledger: pd.DataFrame
    candidate_ledger: pd.DataFrame
    control_returns: pd.Series
    reentry_returns: pd.Series
    candidate_returns: pd.Series
    episodes: pd.DataFrame
    selected_reentries: pd.DataFrame


def _class_to_signal(values: pd.Series) -> np.ndarray:
    classes = pd.to_numeric(values, errors="coerce").to_numpy(float)
    mapped = np.zeros(len(classes), dtype=np.int8)
    mapped[classes == 0] = -1
    mapped[classes == 2] = 1
    return mapped


def build_union_v1_style_signals(predictions: pd.DataFrame) -> pd.DataFrame:
    """Apply frozen LSTM tau=.75, class-only SVM, and opposite-member veto."""
    required = {
        "row_key",
        "decision_time",
        "pred_lstm",
        "pred_svm_linear",
        "p_short_lstm",
        "p_flat_lstm",
        "p_long_lstm",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Union signal predictions lack columns: {missing}")
    output = predictions.copy()
    output["decision_time"] = pd.to_datetime(output["decision_time"], utc=True)
    output = output.sort_values("decision_time", kind="stable").reset_index(drop=True)
    if output["row_key"].duplicated().any() or output["decision_time"].duplicated().any():
        raise AssertionError("Union prediction keys and times must be unique")
    probability = output[
        ["p_short_lstm", "p_flat_lstm", "p_long_lstm"]
    ].to_numpy(float)
    if not np.isfinite(probability).all():
        raise ValueError("Union LSTM probabilities must be finite")
    confidence = probability.max(axis=1)
    lstm_signal = _class_to_signal(output["pred_lstm"])
    lstm_signal[confidence < 0.75] = 0
    svm_signal = _class_to_signal(output["pred_svm_linear"])
    conflict = (lstm_signal != 0) & (svm_signal != 0) & (lstm_signal != svm_signal)
    active = (lstm_signal != 0).astype(np.int8) + (svm_signal != 0).astype(np.int8)
    total = lstm_signal + svm_signal
    union = np.sign(total).astype(np.int8)
    union[np.abs(total) != active] = 0
    output["lstm_confidence"] = confidence
    output["lstm_signal"] = lstm_signal
    output["svm_linear_signal"] = svm_signal
    output["active_members"] = active
    output["member_conflict"] = conflict
    output["union_signal"] = union
    return identify_same_side_episodes(output)


def identify_same_side_episodes(signals: pd.DataFrame) -> pd.DataFrame:
    """Label maximal uninterrupted 15-minute runs of one non-zero Union side."""
    required = {"decision_time", "union_signal"}
    missing = sorted(required.difference(signals.columns))
    if missing:
        raise ValueError(f"Episode input lacks columns: {missing}")
    output = signals.copy()
    output["decision_time"] = pd.to_datetime(output["decision_time"], utc=True)
    output = output.sort_values("decision_time", kind="stable").reset_index(drop=True)
    if output["decision_time"].duplicated().any():
        raise AssertionError("Episode decision times must be unique")
    side = pd.to_numeric(output["union_signal"], errors="coerce").fillna(0).astype(int)
    active = side.ne(0)
    new_episode = active & (
        side.ne(side.shift(1))
        | output["decision_time"].diff().ne(BAR_INTERVAL)
        | ~active.shift(1, fill_value=False)
    )
    episode_number = new_episode.cumsum().where(active)
    output["episode_id"] = episode_number.astype("Int64")
    output["episode_bar"] = output.groupby("episode_id", dropna=True).cumcount().add(1)
    output.loc[~active, "episode_bar"] = pd.NA
    output["episode_bar"] = output["episode_bar"].astype("Int64")
    return output


def _bar_intersects(entry: pd.Timestamp, occupied_entries: set[pd.Timestamp]) -> bool:
    return entry in occupied_entries


def select_episode_reentries(
    signals: pd.DataFrame,
    control_ledger: pd.DataFrame,
) -> pd.DataFrame:
    """Select the earliest skipped qualified bar once per episode, without PnL."""
    frame = identify_same_side_episodes(signals)
    required = {"signal_time", "entry_time", "exit_time"}
    missing = sorted(required.difference(control_ledger.columns))
    if missing:
        raise ValueError(f"Control ledger lacks columns: {missing}")
    control = control_ledger.copy()
    for column in ("signal_time", "entry_time", "exit_time"):
        control[column] = pd.to_datetime(control[column], utc=True)
    used_signal_times = set(control["signal_time"])
    occupied_entries = set(control["entry_time"])
    accepted_entries: set[pd.Timestamp] = set()
    selected: list[dict[str, object]] = []
    for episode_id, episode in frame.loc[frame["episode_id"].notna()].groupby(
        "episode_id", sort=True
    ):
        episode_times = set(episode["decision_time"])
        episode_control = control.loc[control["signal_time"].isin(episode_times)]
        if episode_control.empty:
            continue
        for row in episode.itertuples(index=False):
            signal_time = pd.Timestamp(row.decision_time)
            if signal_time in used_signal_times:
                continue
            preceding = episode_control.loc[episode_control["signal_time"].lt(signal_time)]
            if preceding.empty:
                continue
            prior_exit = preceding["exit_time"].max()
            if prior_exit > signal_time:
                continue
            entry_time = signal_time + BAR_INTERVAL
            overlaps_control = _bar_intersects(entry_time, occupied_entries)
            overlaps_extra = _bar_intersects(entry_time, accepted_entries)
            if overlaps_control or overlaps_extra:
                continue
            selected.append(
                {
                    "row_key": getattr(row, "row_key", str(signal_time)),
                    "episode_id": int(episode_id),
                    "signal_time": signal_time,
                    "entry_time": entry_time,
                    "side": int(row.union_signal),
                    "overlap_control": False,
                    "overlap_extra": False,
                }
            )
            accepted_entries.add(entry_time)
            break
    columns = [
        "row_key",
        "episode_id",
        "signal_time",
        "entry_time",
        "side",
        "overlap_control",
        "overlap_extra",
    ]
    return pd.DataFrame(selected, columns=columns)


def _trade_keys(ledger: pd.DataFrame) -> pd.Series:
    return (
        ledger["route"].astype(str)
        + "|"
        + ledger["signal_time"].map(lambda value: pd.Timestamp(value).isoformat())
    )


def _enrich_ledger(
    ledger: pd.DataFrame,
    route: str,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    output = ledger.copy()
    if len(output):
        output.insert(
            0,
            "signal_time",
            pd.to_datetime(output["entry_time"], utc=True) - BAR_INTERVAL,
        )
    else:
        output.insert(0, "signal_time", pd.Series(dtype="datetime64[ns, UTC]"))
    episode_map = episodes.set_index("decision_time")["episode_id"]
    output.insert(1, "episode_id", output["signal_time"].map(episode_map).astype("Int64"))
    output.insert(2, "route", route)
    output.insert(3, "trade_key", _trade_keys(output))
    return output


def _simulate(
    signals: pd.Series,
    m15: pd.DataFrame,
    minute: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    prediction = signals.map({-1: 0, 0: 1, 1: 2}).astype(int)
    ledger, per_bar = simulate_bracket_trades_intrabar(
        m15,
        minute,
        prediction,
        None,
        tau=0.0,
        tp_bps=TP_BPS,
        sl_bps=SL_BPS,
        max_hold=MAX_HOLD_BARS,
        fee_bps=FEE_BPS_PER_SIDE,
        expected_interval=pd.Timedelta(minutes=1),
        include_audit=True,
    )
    return ledger, per_bar.rename("net_return")


def replay_union_control_and_reentry(
    signals: pd.DataFrame,
    m15: pd.DataFrame,
    minute: pd.DataFrame,
) -> PairedReplay:
    """Freeze the native control, add selected extras, and reconcile paired ledgers."""
    episodes = identify_same_side_episodes(signals)
    signal_series = episodes.set_index("decision_time")["union_signal"].astype(int)
    control_raw, control_returns = _simulate(signal_series, m15, minute)
    control = _enrich_ledger(control_raw, "union_control", episodes)
    selected = select_episode_reentries(episodes, control)
    selected_signal = pd.Series(
        selected["side"].to_numpy(dtype=int),
        index=pd.DatetimeIndex(selected["signal_time"]),
        dtype=int,
    )
    reentry_raw, reentry_returns = _simulate(selected_signal, m15, minute)
    reentry = _enrich_ledger(reentry_raw, "episode_reentry", episodes)
    if len(reentry) != len(selected):
        raise AssertionError("Every selected re-entry must produce one complete native-path trade")
    candidate = pd.concat([control, reentry], ignore_index=True).sort_values(
        ["entry_time", "route"], kind="stable"
    ).reset_index(drop=True)
    if candidate["trade_key"].duplicated().any():
        raise AssertionError("Candidate trade keys must be unique")
    if candidate["entry_time"].duplicated().any():
        raise AssertionError("Control and re-entry positions overlap")
    candidate_returns = control_returns.add(reentry_returns, fill_value=0.0).rename(
        "net_return"
    )
    for ledger, returns, name in (
        (control, control_returns, "control"),
        (reentry, reentry_returns, "re-entry"),
        (candidate, candidate_returns, "candidate"),
    ):
        if not np.isclose(
            returns.sum(), ledger["net_return"].sum(), rtol=0.0, atol=1e-10
        ):
            raise AssertionError(f"Union {name} per-bar returns do not reconcile")
    return PairedReplay(
        control_ledger=control,
        reentry_ledger=reentry,
        candidate_ledger=candidate,
        control_returns=control_returns,
        reentry_returns=reentry_returns,
        candidate_returns=candidate_returns,
        episodes=episodes,
        selected_reentries=selected,
    )


def _summary_value(summary: dict[str, object], key: str) -> float:
    if key not in summary:
        raise ValueError(f"Development summary lacks {key}")
    value = float(summary[key])
    if not np.isfinite(value):
        raise ValueError(f"Development summary {key} must be finite")
    return value


def evaluate_reentry_development(
    control: dict[str, object],
    candidate: dict[str, object],
    incremental: dict[str, object],
    fold_metrics: pd.DataFrame,
) -> dict[str, object]:
    """Evaluate the registered development conjunction with no tuning fallback."""
    required_fold = {"fold_id", "control_net_return", "incremental_net_return"}
    missing = sorted(required_fold.difference(fold_metrics.columns))
    if missing:
        raise ValueError(f"Development fold metrics lack columns: {missing}")
    if len(fold_metrics) != 5 or fold_metrics["fold_id"].nunique() != 5:
        raise ValueError("Development admission requires exactly five folds")
    control_trades = int(_summary_value(control, "trades"))
    candidate_trades = int(_summary_value(candidate, "trades"))
    incremental_trades = int(_summary_value(incremental, "trades"))
    required_candidate_trades = math.ceil(115 * control_trades / 100)
    control_total = _summary_value(control, "net_return")
    control_long = _summary_value(control, "long_net_return")
    control_short = _summary_value(control, "short_net_return")
    candidate_total = _summary_value(candidate, "net_return")
    candidate_long = _summary_value(candidate, "long_net_return")
    candidate_short = _summary_value(candidate, "short_net_return")
    incremental_total = _summary_value(incremental, "net_return")
    incremental_long = _summary_value(incremental, "long_net_return")
    incremental_short = _summary_value(incremental, "short_net_return")
    tolerance = 1e-12
    gates: dict[str, object] = {
        "required_candidate_trades": required_candidate_trades,
        "audit_clean": bool(control.get("audit_clean", False))
        and bool(candidate.get("audit_clean", False))
        and bool(incremental.get("audit_clean", False))
        and bool(fold_metrics.get("audit_clean", pd.Series(True, index=fold_metrics.index)).all()),
        "trade_count_reconciles": candidate_trades == control_trades + incremental_trades,
        "control_total_positive": control_total > 0.0,
        "control_long_positive": control_long > 0.0,
        "control_short_positive": control_short > 0.0,
        "control_three_positive_folds": int((fold_metrics["control_net_return"] > 0.0).sum()) >= 3,
        "frequency_gain": candidate_trades >= required_candidate_trades,
        "candidate_total_noninferior": candidate_total + tolerance >= control_total,
        "candidate_long_noninferior": candidate_long + tolerance >= control_long,
        "candidate_short_noninferior": candidate_short + tolerance >= control_short,
        "incremental_total_nonnegative": incremental_total >= -tolerance,
        "incremental_long_nonnegative": incremental_long >= -tolerance,
        "incremental_short_nonnegative": incremental_short >= -tolerance,
        "incremental_three_nonnegative_folds": int(
            (fold_metrics["incremental_net_return"] >= -tolerance).sum()
        )
        >= 3,
        "candidate_minimum_long_trades": int(_summary_value(candidate, "long_trades")) >= 15,
        "candidate_minimum_short_trades": int(_summary_value(candidate, "short_trades")) >= 15,
        "total_return_reconciles": bool(
            np.isclose(candidate_total, control_total + incremental_total, rtol=0.0, atol=1e-10)
        ),
        "long_return_reconciles": bool(
            np.isclose(candidate_long, control_long + incremental_long, rtol=0.0, atol=1e-10)
        ),
        "short_return_reconciles": bool(
            np.isclose(candidate_short, control_short + incremental_short, rtol=0.0, atol=1e-10)
        ),
    }
    terminal = [value for key, value in gates.items() if key != "required_candidate_trades"]
    gates["development_pass"] = bool(all(terminal))
    gates["decision"] = (
        "development_pass_open_h1"
        if gates["development_pass"]
        else "development_fail_keep_union_v1"
    )
    return gates


__all__ = [
    "BAR_INTERVAL",
    "FEE_BPS_PER_SIDE",
    "MAX_HOLD_BARS",
    "PairedReplay",
    "SL_BPS",
    "TP_BPS",
    "build_union_v1_style_signals",
    "evaluate_reentry_development",
    "identify_same_side_episodes",
    "replay_union_control_and_reentry",
    "select_episode_reentries",
]
