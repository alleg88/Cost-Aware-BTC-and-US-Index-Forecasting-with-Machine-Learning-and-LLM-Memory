"""Validated combined scoreboards for the Notebook 02c baseline study."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluation.economics import economics_summary
from experiments.baseline_model_zoo_1m import MODEL_NAMES, validate_model_artifacts
from experiments.raw_hold_control import (
    CONFIG_PATH,
    FORWARD_END,
    FORWARD_START,
    _load_m15_bars,
    simulate_fixed_hold,
)
from experiments.run_baseline_model_zoo_1m import DEFAULT_ROOT

TABLE_FILES = {
    "classification": "classification_2024.parquet",
    "hold_grid": "hold_grid_2024.parquet",
    "selected_holds": "selected_holds_2024.parquet",
    "calibration_grid": "calibration_policy_grid_2025h1.parquet",
    "selected_policies": "selected_policies_2025h1.parquet",
    "raw_forward": "raw_forward_summary.parquet",
    "forward": "forward_summary.parquet",
    "forward_monthly": "forward_monthly.parquet",
    "forward_quarterly": "forward_quarterly.parquet",
}


def _load_table(root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for model_name in MODEL_NAMES:
        frame = pd.read_parquet(root / model_name / filename).copy()
        frame["model_name"] = model_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _winners(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for criterion, column in (
        ("Sortino", "sortino"),
        ("Sharpe", "sharpe"),
        ("Net return", "net_return"),
    ):
        row = frame.loc[frame[column].astype(float).idxmax()].to_dict()
        row["criterion"] = criterion
        rows.append(row)
    return pd.DataFrame(rows)


def _top_models(frame: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Rank distinct model families by their best-DZ net result."""
    ranked = frame.sort_values(
        ["net_return", "sortino", "sharpe"],
        ascending=[False, False, False],
    )
    return ranked.drop_duplicates("model_name", keep="first").head(limit).reset_index(
        drop=True
    )


def _forward_prediction(
    root: Path,
    model_name: str,
    width_bps: int,
    lookback_days: int,
    fit_id: str,
) -> pd.DataFrame:
    paths = sorted(
        (root / model_name / "stage_predictions" / "forward").glob(
            f"w{width_bps}_lb{lookback_days}_candidate_00_*.parquet"
        )
    )
    matches = []
    for path in paths:
        frame = pd.read_parquet(path)
        if not frame.empty and frame["refit_id"].astype(str).eq(fit_id).all():
            matches.append(frame)
    if len(matches) != 1:
        raise ValueError(
            f"{model_name} DZ{width_bps} {lookback_days}D: "
            "expected one matching forward prediction"
        )
    frame = matches[0].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp")


def _raw_forward_monthly(root: Path, raw_forward: pd.DataFrame) -> pd.DataFrame:
    """Replay frozen fixed-hold predictions into nine monthly evidence rows."""
    bars = _load_m15_bars(CONFIG_PATH)
    edges = pd.date_range(FORWARD_START, FORWARD_END, freq="MS")
    rows = []
    for summary in raw_forward.to_dict("records"):
        model_name = str(summary["model_name"])
        width_bps = int(summary["width_bps"])
        lookback_days = int(summary["lookback_days"])
        prediction = _forward_prediction(
            root, model_name, width_bps, lookback_days, str(summary["fit_id"])
        )
        pred = prediction.set_index("timestamp")["pred"].astype(int)
        ledger, returns = simulate_fixed_hold(
            bars,
            pred,
            hold_bars=int(summary["max_hold"]),
            fee_bps=5.0,
        )
        entries = pd.to_datetime(ledger["entry_time"], utc=True)
        for start, end in zip(edges[:-1], edges[1:]):
            period_returns = returns.loc[
                (returns.index >= start) & (returns.index < end)
            ]
            period_ledger = ledger.loc[(entries >= start) & (entries < end)]
            metrics = economics_summary(period_returns)
            rows.append(
                {
                    "model_name": model_name,
                    "width_bps": width_bps,
                    "lookback_days": lookback_days,
                    "period": start.strftime("%Y-%m"),
                    "hold_minutes": int(summary["hold_minutes"]),
                    "trades": int(len(period_ledger)),
                    "gross_return": float(period_ledger["gross_return"].sum()),
                    "net_return": float(period_returns.sum()),
                    "sortino": float(metrics["sortino"]),
                    "sharpe": float(metrics["sharpe"]),
                }
            )
    monthly = pd.DataFrame(rows)
    keys = ["model_name", "width_bps", "lookback_days"]
    reconciled = monthly.groupby(keys)["net_return"].sum()
    expected = raw_forward.set_index(keys)["net_return"]
    if float((reconciled - expected).abs().max()) >= 1e-10:
        raise AssertionError("uncalibrated monthly returns do not reconcile")
    return monthly


def build_scoreboards(root: Path = DEFAULT_ROOT) -> dict[str, pd.DataFrame]:
    root = Path(root)
    for model_name in MODEL_NAMES:
        validate_model_artifacts(root / model_name, model_name=model_name)
    tables = {
        name: _load_table(root, filename)
        for name, filename in TABLE_FILES.items()
    }
    tables["raw_winners"] = _winners(tables["raw_forward"])
    tables["calibrated_winners"] = _winners(tables["forward"])
    tables["raw_top_models"] = _top_models(tables["raw_forward"])
    tables["calibrated_top_models"] = _top_models(tables["forward"])
    tables["raw_forward_monthly"] = _raw_forward_monthly(
        root, tables["raw_forward"]
    )
    return tables


def write_scoreboards(root: Path = DEFAULT_ROOT) -> dict[str, pd.DataFrame]:
    root = Path(root)
    tables = build_scoreboards(root)
    output = root / "combined"
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_parquet(output / f"{name}.parquet", index=False)
        frame.to_csv(output / f"{name}.csv", index=False)
    return tables


if __name__ == "__main__":
    tables = write_scoreboards()
    print(tables["raw_winners"].to_string(index=False))
    print(tables["calibrated_winners"].to_string(index=False))
