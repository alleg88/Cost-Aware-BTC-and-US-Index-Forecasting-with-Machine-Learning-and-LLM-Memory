"""Uncalibrated fixed-hold control for the frozen nine-model BTC study."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluation.economics import economics_summary
from evaluation.trades import TRADE_COLUMNS, _empty_ledger

MODEL_NAMES = (
    "logreg",
    "decision_tree",
    "random_forest",
    "svm_linear",
    "xgboost_balanced",
    "catboost_balanced",
    "mlp",
    "lstm",
    "gru",
)
WIDTHS = (55, 65, 75)
HOLD_MINUTES = (15, 30)
SUMMARY_ROWS = len(MODEL_NAMES) * len(WIDTHS) * len(HOLD_MINUTES)
CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_MATCHED_ROOT = (
    CODE_ROOT / "experiments" / "cache" / "tuning" / "matched_model_zoo_1m"
)
DEFAULT_CATBOOST_SELECTION_ROOT = (
    CODE_ROOT
    / "experiments"
    / "cache"
    / "tuning"
    / "catboost_execution_resolution"
    / "one_minute"
)
DEFAULT_CATBOOST_PREDICTION_ROOT = (
    CODE_ROOT
    / "experiments"
    / "cache"
    / "tuning"
    / "catboost_matched_ablation"
    / "stage_predictions"
    / "frozen_post_selection"
)
DEFAULT_OUTPUT = (
    CODE_ROOT
    / "experiments"
    / "cache"
    / "tuning"
    / "matched_model_zoo_raw_hold"
    / "raw_hold_summary.parquet"
)
CONFIG_PATH = CODE_ROOT / "configs" / "default.yaml"
FORWARD_START = pd.Timestamp("2025-07-01", tz="UTC")
FORWARD_END = pd.Timestamp("2026-04-01", tz="UTC")
MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm_linear": "Linear SVM",
    "xgboost_balanced": "XGBoost",
    "catboost_balanced": "CatBoost",
    "mlp": "MLP",
    "lstm": "LSTM",
    "gru": "GRU",
}


def simulate_fixed_hold(
    bars: pd.DataFrame,
    pred: pd.Series,
    *,
    hold_bars: int,
    fee_bps: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Enter at the next M15 open and exit after a fixed number of closes."""
    if int(hold_bars) != hold_bars or hold_bars < 1:
        raise ValueError("hold_bars must be a positive integer")
    if bars.index.tz is None or pred.index.tz is None:
        raise ValueError("bars and predictions must use a timezone-aware index")
    missing = {"open", "close"}.difference(bars.columns)
    if missing:
        raise ValueError(f"bars miss required columns: {sorted(missing)}")

    bars = bars.sort_index()
    signal = pred.reindex(bars.index)
    open_price = bars["open"].to_numpy(dtype=float)
    close_price = bars["close"].to_numpy(dtype=float)
    values = signal.to_numpy(dtype=float)
    per_bar = np.zeros(len(bars), dtype=float)
    cost = float(fee_bps) / 10_000.0
    trades: list[dict] = []

    signal_index = 0
    while signal_index + hold_bars < len(bars):
        predicted = values[signal_index]
        if predicted not in (0.0, 2.0):
            signal_index += 1
            continue
        side = 1 if predicted == 2.0 else -1
        entry_index = signal_index + 1
        exit_index = signal_index + int(hold_bars)
        entry_price = open_price[entry_index]
        exit_price = close_price[exit_index]
        gross_return = side * (exit_price / entry_price - 1.0)
        net_return = gross_return - 2.0 * cost

        marks = np.concatenate(
            ([entry_price], close_price[entry_index : exit_index + 1])
        )
        increments = side * np.diff(marks) / entry_price
        per_bar[entry_index : exit_index + 1] += increments
        per_bar[entry_index] -= cost
        per_bar[exit_index] -= cost
        trades.append(
            {
                "entry_time": bars.index[entry_index],
                "exit_time": bars.index[exit_index],
                "side": side,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "bars_held": int(hold_bars),
                "exit_reason": "fixed_hold",
                "gross_return": float(gross_return),
                "net_return": float(net_return),
            }
        )
        signal_index = exit_index

    ledger = (
        pd.DataFrame(trades, columns=TRADE_COLUMNS)
        if trades
        else _empty_ledger()
    )
    returns = pd.Series(per_bar, index=bars.index, name="net_return")
    if not np.isclose(
        float(returns.sum()),
        float(ledger["net_return"].sum()),
        atol=1e-10,
    ):
        raise AssertionError("fixed-hold returns do not reconcile to the ledger")
    return ledger, returns


def validate_raw_hold_summary(
    frame: pd.DataFrame,
    *,
    model_names: Sequence[str] = MODEL_NAMES,
) -> None:
    """Validate the exact raw-control grid and its sealed time boundaries."""
    required = {
        "model_name",
        "model",
        "width_bps",
        "candidate_id",
        "hold_minutes",
        "fee_bps_per_side",
        "fit_end",
        "period_start",
        "period_end",
        "trades",
        "n_long",
        "n_short",
        "gross_return",
        "net_return",
        "sortino",
        "sharpe",
        "positive_months",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"raw-hold summary misses columns: {sorted(missing)}")
    expected_rows = len(tuple(model_names)) * len(WIDTHS) * len(HOLD_MINUTES)
    if len(frame) != expected_rows:
        raise ValueError(f"raw-hold summary must contain exactly {expected_rows} rows")
    keys = ["model_name", "width_bps", "hold_minutes"]
    if frame.duplicated(keys).any():
        raise ValueError("raw-hold model/DZ/hold rows must be unique")
    expected_grid = {
        (model_name, width, hold)
        for model_name in model_names
        for width in WIDTHS
        for hold in HOLD_MINUTES
    }
    actual_grid = set(
        frame[keys].itertuples(index=False, name=None)
    )
    if actual_grid != expected_grid:
        raise ValueError("raw-hold summary does not match the exact model/DZ/hold grid")
    if not frame["fee_bps_per_side"].astype(float).eq(5.0).all():
        raise ValueError("raw-hold summary must use 5 bps per side")
    fit_end = pd.to_datetime(frame["fit_end"], utc=True)
    if fit_end.gt(pd.Timestamp("2025-01-01", tz="UTC")).any():
        raise ValueError("raw-hold summary must use 2024-only fitted models")
    period_start = pd.to_datetime(frame["period_start"], utc=True)
    period_end = pd.to_datetime(frame["period_end"], utc=True)
    if not period_start.eq(pd.Timestamp("2025-07-01", tz="UTC")).all():
        raise ValueError("raw-hold summary must start on 2025-07-01")
    if period_end.gt(pd.Timestamp("2026-04-01", tz="UTC")).any():
        raise ValueError("raw-hold summary reached the sealed 2026 Q2 lockbox")
    if not (frame["trades"] == frame["n_long"] + frame["n_short"]).all():
        raise ValueError("raw-hold trade sides do not reconcile")
    metric_columns = (
        "gross_return",
        "net_return",
        "sortino",
        "sharpe",
    )
    if not np.isfinite(frame.loc[:, metric_columns].to_numpy(dtype=float)).all():
        raise ValueError("raw-hold economic metrics must be finite")


def _source_roots(
    model_name: str,
    *,
    matched_root: Path,
    catboost_selection_root: Path,
    catboost_prediction_root: Path,
) -> tuple[Path, Path]:
    if model_name == "catboost_balanced":
        return catboost_selection_root, catboost_prediction_root
    model_root = matched_root / model_name
    return (
        model_root,
        model_root
        / "prediction_cache"
        / "stage_predictions"
        / "frozen_post_selection",
    )


def _load_selected_prediction(
    *,
    model_name: str,
    width_bps: int,
    candidate_id: int,
    prediction_root: Path,
) -> pd.DataFrame:
    matches = sorted(
        prediction_root.glob(
            f"w{int(width_bps)}_candidate_{int(candidate_id):02d}_*.parquet"
        )
    )
    if len(matches) != 1:
        raise ValueError(
            f"{model_name} DZ{width_bps} candidate {candidate_id}: "
            f"expected one frozen prediction cache, found {len(matches)}"
        )
    frame = pd.read_parquet(matches[0]).copy()
    required = {
        "timestamp",
        "width_bps",
        "candidate_id",
        "pred",
        "train_end",
        "test_start",
        "test_end",
        "refit_id",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"frozen prediction cache misses columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if not frame["width_bps"].astype(int).eq(int(width_bps)).all():
        raise ValueError(f"{model_name} DZ{width_bps}: prediction width changed")
    if not frame["candidate_id"].astype(int).eq(int(candidate_id)).all():
        raise ValueError(f"{model_name} DZ{width_bps}: prediction candidate changed")
    if pd.to_datetime(frame["train_end"], utc=True).gt(
        pd.Timestamp("2025-01-01", tz="UTC")
    ).any():
        raise ValueError(f"{model_name} DZ{width_bps}: prediction was not fit on 2024 only")
    if frame["refit_id"].nunique() != 1:
        raise ValueError(f"{model_name} DZ{width_bps}: fitted model changed")
    return frame.sort_values("timestamp").reset_index(drop=True)


def _load_m15_bars(config_path: Path) -> pd.DataFrame:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    path = CODE_ROOT / config["instruments"]["btc"]["working_parquet"]
    bars = pd.read_parquet(path)
    bars.index = pd.to_datetime(bars.index, utc=True)
    bars = bars.sort_index()
    return bars.loc[(bars.index >= FORWARD_START) & (bars.index < FORWARD_END)]


def _positive_months(returns: pd.Series) -> int:
    edges = pd.date_range(FORWARD_START, FORWARD_END, freq="MS")
    month_nets = [
        float(returns.loc[(returns.index >= start) & (returns.index < end)].sum())
        for start, end in zip(edges[:-1], edges[1:])
    ]
    return int(sum(value > 0.0 for value in month_nets))


def build_raw_hold_summary(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    matched_root: Path = DEFAULT_MATCHED_ROOT,
    catboost_selection_root: Path = DEFAULT_CATBOOST_SELECTION_ROOT,
    catboost_prediction_root: Path = DEFAULT_CATBOOST_PREDICTION_ROOT,
    config_path: Path = CONFIG_PATH,
) -> pd.DataFrame:
    """Build the 54-row raw fixed-hold comparison from frozen caches."""
    output_path = Path(output_path)
    matched_root = Path(matched_root)
    catboost_selection_root = Path(catboost_selection_root)
    catboost_prediction_root = Path(catboost_prediction_root)
    bars = _load_m15_bars(Path(config_path))
    rows: list[dict] = []
    for model_name in MODEL_NAMES:
        selection_root, prediction_root = _source_roots(
            model_name,
            matched_root=matched_root,
            catboost_selection_root=catboost_selection_root,
            catboost_prediction_root=catboost_prediction_root,
        )
        selected = pd.read_parquet(
            selection_root / "selected_candidates_2024.parquet"
        )
        for width_bps in WIDTHS:
            choice = selected.loc[
                selected["width_bps"].astype(int) == int(width_bps)
            ]
            if len(choice) != 1:
                raise ValueError(
                    f"{model_name} DZ{width_bps}: expected one selected candidate"
                )
            candidate_id = int(choice.iloc[0]["candidate_id"])
            prediction = _load_selected_prediction(
                model_name=model_name,
                width_bps=width_bps,
                candidate_id=candidate_id,
                prediction_root=prediction_root,
            )
            prediction = prediction.loc[
                (prediction["timestamp"] >= FORWARD_START)
                & (prediction["timestamp"] < FORWARD_END)
            ]
            pred = prediction.set_index("timestamp")["pred"].astype(int)
            for hold_minutes in HOLD_MINUTES:
                ledger, per_bar = simulate_fixed_hold(
                    bars,
                    pred,
                    hold_bars=hold_minutes // 15,
                    fee_bps=5.0,
                )
                summary = economics_summary(per_bar)
                rows.append(
                    {
                        "model_name": model_name,
                        "model": MODEL_LABELS[model_name],
                        "width_bps": int(width_bps),
                        "candidate_id": candidate_id,
                        "hold_minutes": int(hold_minutes),
                        "fee_bps_per_side": 5.0,
                        "fit_end": pd.to_datetime(
                            prediction["train_end"].iloc[0], utc=True
                        ),
                        "period_start": FORWARD_START,
                        "period_end": FORWARD_END,
                        "trades": int(len(ledger)),
                        "n_long": int((ledger["side"] == 1).sum()),
                        "n_short": int((ledger["side"] == -1).sum()),
                        "gross_return": float(ledger["gross_return"].sum()),
                        "net_return": float(summary["net_return_sum"]),
                        "sortino": float(summary["sortino"]),
                        "sharpe": float(summary["sharpe"]),
                        "positive_months": _positive_months(per_bar),
                    }
                )
    result = pd.DataFrame(rows).sort_values(
        ["model_name", "width_bps", "hold_minutes"]
    ).reset_index(drop=True)
    validate_raw_hold_summary(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".part")
    result.to_parquet(temporary, index=False)
    temporary.replace(output_path)
    return result


def load_raw_hold_summary(path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Load and validate the persisted raw fixed-hold comparison."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    validate_raw_hold_summary(frame)
    return frame


def main() -> int:
    summary = build_raw_hold_summary()
    print(summary.sort_values("sortino", ascending=False).head(10).to_string(index=False))
    print(f"\nwrote {len(summary)} rows -> {DEFAULT_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
