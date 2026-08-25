"""Immutable LSTM/SVM qualified-Union baseline shared by later experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.economics import economics_summary
from evaluation.trades_intrabar import simulate_bracket_trades_intrabar


CODE_ROOT = Path(__file__).resolve().parents[1]
TUNING_ROOT = CODE_ROOT / "experiments" / "cache" / "tuning"
H1_ROOT = TUNING_ROOT / "all_model_sentiment_policy_180d_fixed15_monthly_h1" / "none"
FORWARD_ROOT = TUNING_ROOT / "all_model_sentiment_raw_180d_fixed15" / "none"
BARS_PATH = CODE_ROOT / "data" / "btcusdt_m15_2024_2025.parquet"
MINUTE_PATH = CODE_ROOT / "data" / "btcusdt_1m_2025_2026.parquet"

H1_START = pd.Timestamp("2025-01-01", tz="UTC")
FORWARD_START = pd.Timestamp("2025-07-01", tz="UTC")
LOCKBOX_START = pd.Timestamp("2026-04-01", tz="UTC")


@dataclass(frozen=True)
class Member:
    model: str
    width_bps: int
    tau: float


MEMBERS = (
    Member("lstm", 55, 0.75),
    Member("svm_linear", 75, 0.0),
)
TP_BPS = 200
SL_BPS = 100
MAX_HOLD = 1
FEE_BPS = 5.0


def protocol() -> dict[str, object]:
    return {
        "protocol_version": "qualified-union-v1",
        "members": [asdict(member) for member in MEMBERS],
        "combiner": "union_with_opposite_signal_veto",
        "position_size": "equal_one_unit",
        "execution": {
            "tp_bps": TP_BPS,
            "sl_bps": SL_BPS,
            "max_hold": MAX_HOLD,
            "fee_bps_per_side": FEE_BPS,
        },
        "h1_start": H1_START.isoformat(),
        "forward_start": FORWARD_START.isoformat(),
        "lockbox_start": LOCKBOX_START.isoformat(),
        "lockbox_2026_q2_used": False,
    }


def prediction_paths(member: Member, stage: str) -> list[Path]:
    if stage == "h1":
        pattern = f"stage_predictions/calibration_2025_*/w{member.width_bps}_*.parquet"
        paths = sorted((H1_ROOT / member.model).glob(pattern))
        if len(paths) != 6:
            raise ValueError(
                f"{member.model} DZ{member.width_bps}: expected six H1 monthly files, "
                f"found {len(paths)}"
            )
        return paths
    if stage == "forward":
        pattern = f"stage_predictions/raw_forward/w{member.width_bps}_*.parquet"
        paths = sorted((FORWARD_ROOT / member.model).glob(pattern))
        if len(paths) != 1:
            raise ValueError(
                f"{member.model} DZ{member.width_bps}: expected one frozen-forward file, "
                f"found {len(paths)}"
            )
        return paths
    raise ValueError("stage must be 'h1' or 'forward'")


def load_member_panel(model: str, width_bps: int, stage: str) -> pd.DataFrame:
    member = next(
        (
            candidate
            for candidate in MEMBERS
            if candidate.model == model and candidate.width_bps == int(width_bps)
        ),
        None,
    )
    if member is None:
        raise ValueError(f"member is not frozen in Union v1: {model} DZ{width_bps}")
    frames = [pd.read_parquet(path) for path in prediction_paths(member, stage)]
    frame = pd.concat(frames, ignore_index=True)
    required = {
        "timestamp",
        "pred",
        "confidence",
        "p_short",
        "p_flat",
        "p_long",
        "refit_id",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"prediction panel is missing columns: {missing}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame = frame.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    start = H1_START if stage == "h1" else FORWARD_START
    end = FORWARD_START if stage == "h1" else LOCKBOX_START
    if frame.empty or frame.index.min() < start or frame.index.max() >= end:
        raise ValueError(f"{stage} prediction timestamps cross the frozen interval")
    expected_refits = 6 if stage == "h1" else 1
    if frame["refit_id"].astype(str).nunique() != expected_refits:
        raise ValueError(f"{stage} prediction refit identity changed")
    return frame


def member_signal(frame: pd.DataFrame, *, tau: float) -> pd.Series:
    signal = frame["pred"].map({0: -1.0, 1: 0.0, 2: 1.0}).astype(float)
    if float(tau) > 0.0:
        signal = signal.where(frame["confidence"].astype(float) >= float(tau), 0.0)
    return signal


def latent_direction(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(frame["p_long"].astype(float) >= frame["p_short"].astype(float), 1.0, -1.0),
        index=frame.index,
    )


def conditional_direction_confidence(frame: pd.DataFrame) -> pd.Series:
    short = frame["p_short"].astype(float)
    long = frame["p_long"].astype(float)
    total = short + long
    return pd.Series(
        np.divide(
            np.maximum(short, long),
            total,
            out=np.full(len(frame), 0.5, dtype=float),
            where=total.to_numpy() > 0.0,
        ),
        index=frame.index,
    )


def build_member_signals(stage: str) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for member in MEMBERS:
        frame = load_member_panel(member.model, member.width_bps, stage)
        prefix = member.model
        columns[f"{prefix}_signal"] = member_signal(frame, tau=member.tau)
        columns[f"{prefix}_latent_side"] = latent_direction(frame)
        columns[f"{prefix}_direction_confidence"] = conditional_direction_confidence(frame)
    output = pd.DataFrame(columns).sort_index()
    if output.isna().any().any():
        raise ValueError("frozen member prediction grids are not aligned")
    return output


def combine_union(member_signals: pd.DataFrame) -> pd.Series:
    signal_columns = [column for column in member_signals if column.endswith("_signal")]
    if not signal_columns:
        raise ValueError("member_signals contains no *_signal columns")
    signals = member_signals[signal_columns].astype(float)
    if not signals.isin((-1.0, 0.0, 1.0)).all().all():
        raise ValueError("member signals must be -1, 0 or 1")
    net = signals.sum(axis=1)
    active = signals.ne(0.0).sum(axis=1)
    output = np.sign(net)
    output[net.abs() != active] = 0.0
    return pd.Series(output, index=signals.index, name="union_signal", dtype=float)


def build_union_frame(stage: str) -> pd.DataFrame:
    members = build_member_signals(stage)
    members["active_members"] = members.filter(like="_signal").ne(0.0).sum(axis=1)
    members["member_conflict"] = (
        members["lstm_signal"].ne(0.0)
        & members["svm_linear_signal"].ne(0.0)
        & members["lstm_signal"].ne(members["svm_linear_signal"])
    )
    members["union_signal"] = combine_union(members)
    return members


def _read_market(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        filters=[
            ("timestamp", ">=", start.to_pydatetime()),
            ("timestamp", "<", end.to_pydatetime()),
        ],
    )
    frame.index = pd.to_datetime(frame.index, utc=True)
    if frame.empty or frame.index.min() < start or frame.index.max() >= end:
        raise ValueError(f"market read crossed requested interval: {path.name}")
    return frame.sort_index()


def simulate_union(
    signal: pd.Series,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series]:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.tz is None or end.tz is None:
        raise ValueError("simulation boundaries must be timezone-aware")
    if end > LOCKBOX_START or signal.index.max() >= LOCKBOX_START:
        raise ValueError("Union v1 must not read or score the Q2 lockbox")
    bars = _read_market(BARS_PATH, start, end)
    minute = _read_market(MINUTE_PATH, start, end)
    scoped = signal.loc[(signal.index >= start) & (signal.index < end)]
    prediction = scoped.map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
    ledger, per_bar = simulate_bracket_trades_intrabar(
        bars,
        minute,
        prediction,
        None,
        tau=0.0,
        tp_bps=TP_BPS,
        sl_bps=SL_BPS,
        max_hold=MAX_HOLD,
        fee_bps=FEE_BPS,
        expected_interval=pd.Timedelta(minutes=1),
        include_audit=True,
    )
    if len(ledger):
        ledger.insert(0, "signal_time", pd.to_datetime(ledger["entry_time"], utc=True) - pd.Timedelta(minutes=15))
    else:
        ledger.insert(0, "signal_time", pd.Series(dtype="datetime64[ns, UTC]"))
    if not np.isclose(per_bar.sum(), ledger["net_return"].sum(), atol=1e-10):
        raise AssertionError("Union per-bar return does not reconcile to its ledger")
    return ledger, per_bar.rename("net_return")


def summarize(ledger: pd.DataFrame, per_bar: pd.Series, *, phase: str) -> dict[str, object]:
    economics = economics_summary(per_bar)
    side = pd.to_numeric(ledger["side"], errors="coerce")
    gross = ledger["gross_return"].astype(float)
    return {
        "phase": phase,
        "period_start": per_bar.index.min().isoformat(),
        "period_end_exclusive": (per_bar.index.max() + pd.Timedelta(minutes=15)).isoformat(),
        "trades": int(len(ledger)),
        "long_trades": int((side > 0).sum()),
        "short_trades": int((side < 0).sum()),
        "gross_return": float(gross.sum()),
        "gross_bps_per_trade": float(gross.mean() * 10_000.0) if len(gross) else np.nan,
        "net_return": float(economics["net_return_sum"]),
        "sortino": float(economics["sortino"]),
        "sharpe": float(economics["sharpe"]),
        "max_drawdown": float(economics["max_drawdown"]),
    }


def monthly_summary(per_bar: pd.Series, *, phase: str) -> pd.DataFrame:
    monthly = per_bar.groupby(per_bar.index.tz_convert(None).to_period("M")).sum()
    return pd.DataFrame(
        {
            "phase": phase,
            "month": monthly.index.astype(str),
            "net_return": monthly.to_numpy(float),
        }
    )


__all__ = [
    "CODE_ROOT",
    "FEE_BPS",
    "FORWARD_START",
    "H1_START",
    "LOCKBOX_START",
    "MAX_HOLD",
    "MEMBERS",
    "SL_BPS",
    "TP_BPS",
    "build_member_signals",
    "build_union_frame",
    "combine_union",
    "conditional_direction_confidence",
    "latent_direction",
    "load_member_panel",
    "member_signal",
    "monthly_summary",
    "prediction_paths",
    "protocol",
    "simulate_union",
    "summarize",
]
