"""Causal Sharpe-weighted, regime-gated ensemble for Notebook 03b.

The nine frozen model probabilities are combined with non-negative weights based
on net Sharpe from 2024 OOF and completed past months only. H1 selects DZ,
base threshold and volatility sensitivity; July 2025-March 2026 is replayed
once after that selection. The 2026-Q2 lockbox is never accessed.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.economics import economics_summary
from evaluation.trades_intrabar import simulate_bracket_trades_intrabar
from experiments.all_model_sentiment_raw import prepare_arm
from experiments.baseline_model_zoo_1m import (
    CALIBRATION_START,
    FORWARD_END,
    FORWARD_START,
    WIDTHS,
)
from experiments.correlation_ensemble import (
    FEE_BPS,
    PROBABILITY_COLUMNS,
    load_aligned_stage,
    probabilities_to_frame,
)
from experiments.raw_hold_control import MODEL_NAMES
from experiments.run_catboost_matched_ablation import (
    _atomic_json,
    _atomic_parquet,
    safe_signal_mask,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = CODE_ROOT / "experiments" / "cache" / "tuning" / "causal_sharpe_ensemble"
ARM = "none"
H1_STAGES = tuple(f"calibration_2025_{month:02d}" for month in range(1, 7))
TAU_BASE_GRID = (0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
ALPHA_GRID = (0.0, 0.025, 0.05)
WEIGHT_TAU = 0.0
TP_BPS = 200
SL_BPS = 100
MAX_HOLD_GRID = (1, 2)
WEIGHT_MAX_HOLD = 1
BE_TRIGGER_BPS = 25.0
TRAIL_BASE_BPS = 75.0
TRAIL_VOL_BETA = 0.25
TRAIL_MIN_BPS = 25.0
TRAIL_MAX_BPS = 150.0
VOL_Z_LOOKBACK = 672
FUNDING_Z_LIMIT = 2.0


def positive_sharpe_weights(
    history: Mapping[str, pd.Series],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return max(net Sharpe, 0) weights and the underlying Sharpe scores."""
    sharpes = {
        model: float(economics_summary(pd.concat(parts).sort_index())["sharpe"])
        if isinstance(parts, list) and parts
        else float(economics_summary(parts)["sharpe"])
        for model, parts in history.items()
    }
    positive = {model: max(score, 0.0) for model, score in sharpes.items()}
    total = float(sum(positive.values()))
    weights = {
        model: value / total if total > 0.0 else 0.0
        for model, value in positive.items()
    }
    return weights, sharpes


def combine_probabilities(
    probabilities: Mapping[str, np.ndarray], weights: Mapping[str, float]
) -> np.ndarray:
    """Weighted probability average; an all-zero allocation is explicitly flat."""
    missing = set(MODEL_NAMES).difference(probabilities, weights)
    if missing:
        raise ValueError(f"probability/weight panel misses models: {sorted(missing)}")
    n = len(next(iter(probabilities.values())))
    total = float(sum(weights[model] for model in MODEL_NAMES))
    if total <= 0.0:
        out = np.zeros((n, 3), dtype=float)
        out[:, 1] = 1.0
        return out
    return sum(
        float(weights[model]) * probabilities[model]
        for model in MODEL_NAMES
    ) / total


def volatility_z(context: pd.DataFrame) -> pd.Series:
    vol = context["vol_20"].astype(float)
    mean = vol.rolling(VOL_Z_LOOKBACK, min_periods=96).mean()
    std = vol.rolling(VOL_Z_LOOKBACK, min_periods=96).std().replace(0.0, np.nan)
    return ((vol - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def dynamic_thresholds(
    context: pd.DataFrame, *, tau_base: float, alpha: float
) -> pd.Series:
    return (float(tau_base) + float(alpha) * volatility_z(context)).clip(1.0 / 3.0, 0.90)


def volatility_trail_bps(context: pd.DataFrame) -> pd.Series:
    values = TRAIL_BASE_BPS * np.exp(TRAIL_VOL_BETA * volatility_z(context))
    return values.clip(TRAIL_MIN_BPS, TRAIL_MAX_BPS).rename("trail_bps")


def apply_regime_gate(
    prediction: pd.DataFrame,
    context: pd.DataFrame,
    *,
    tau_base: float,
    alpha: float,
    funding_filter: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply causal volatility threshold and the high-funding long veto."""
    frame = prediction.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp").sort_index()
    aligned = context.reindex(frame.index)
    required = {"vol_20", "funding_z"}
    missing = required.difference(aligned.columns)
    if missing:
        raise ValueError(f"regime gate misses context columns: {sorted(missing)}")
    threshold = dynamic_thresholds(aligned, tau_base=tau_base, alpha=alpha)
    raw_pred = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float).argmax(axis=1)
    confidence = frame[list(PROBABILITY_COLUMNS)].max(axis=1)
    pass_confidence = confidence >= threshold
    blocked_funding = (
        (raw_pred == 2) & (aligned["funding_z"].astype(float) > FUNDING_Z_LIMIT)
        if funding_filter
        else np.zeros(len(frame), dtype=bool)
    )
    frame["pred"] = np.where(pass_confidence & ~blocked_funding, raw_pred, 1).astype(int)
    frame["confidence"] = confidence
    diagnostics = pd.DataFrame(
        {
            "timestamp": frame.index,
            "tau_t": threshold.to_numpy(dtype=float),
            "z_vol": volatility_z(aligned).to_numpy(dtype=float),
            "funding_z": aligned["funding_z"].to_numpy(dtype=float),
            "raw_pred": raw_pred,
            "pred": frame["pred"].to_numpy(dtype=int),
            "blocked_by_confidence": ~pass_confidence.to_numpy(dtype=bool),
            "blocked_long_by_funding": np.asarray(blocked_funding, dtype=bool),
        }
    )
    return frame.reset_index(), diagnostics


def _execute(
    prepared: Any,
    prediction: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    trail_bps: pd.Series | None = None,
    be_trigger_bps: float | None = None,
    tau: float = 0.0,
    max_hold: int = WEIGHT_MAX_HOLD,
) -> tuple[pd.DataFrame, pd.Series]:
    indexed = prediction.copy()
    indexed["timestamp"] = pd.to_datetime(indexed["timestamp"], utc=True)
    indexed = indexed.set_index("timestamp").sort_index()
    safe = safe_signal_mask(indexed.index, end_exclusive=end, max_hold=max_hold)
    scope = prepared.bars.loc[(prepared.bars.index >= start) & (prepared.bars.index < end)]
    ledger, per_bar = simulate_bracket_trades_intrabar(
        scope,
        prepared.minute,
        indexed.loc[safe, "pred"].astype(int),
        indexed.loc[safe, "confidence"].astype(float),
        tau=float(tau),
        tp_bps=TP_BPS,
        sl_bps=SL_BPS,
        max_hold=int(max_hold),
        fee_bps=FEE_BPS,
        trail_bps=trail_bps,
        be_trigger_bps=be_trigger_bps,
        expected_interval=pd.Timedelta(minutes=1),
        include_audit=True,
    )
    if not np.isclose(per_bar.sum(), ledger["net_return"].sum(), atol=1e-10):
        raise AssertionError("execution return series does not reconcile to ledger")
    return ledger, per_bar


def _raw_frame(
    timestamp: pd.Series,
    y_true: pd.Series,
    probabilities: np.ndarray,
    refit_id: str,
) -> pd.DataFrame:
    return probabilities_to_frame(
        timestamp=timestamp,
        y_true=y_true,
        probabilities=probabilities,
        refit_id=refit_id,
    )


def _score_model_stage(
    prepared: Any,
    timestamp: pd.Series,
    y_true: pd.Series,
    probabilities: Mapping[str, np.ndarray],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    stage: str,
) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for model in MODEL_NAMES:
        frame = _raw_frame(timestamp, y_true, probabilities[model], f"weight-score:{stage}:{model}")
        _, returns = _execute(
            prepared,
            frame,
            start=start,
            end=end,
            tau=WEIGHT_TAU,
            max_hold=WEIGHT_MAX_HOLD,
        )
        out[model] = returns
    return out


def _snapshot_rows(
    weights: Mapping[str, float],
    sharpes: Mapping[str, float],
    *,
    width_bps: int,
    effective_from: pd.Timestamp,
    phase: str,
) -> list[dict[str, Any]]:
    return [
        {
            "width_bps": int(width_bps),
            "phase": phase,
            "effective_from": effective_from,
            "model_name": model,
            "net_sharpe": float(sharpes[model]),
            "weight": float(weights[model]),
            "enabled": bool(weights[model] > 0.0),
        }
        for model in MODEL_NAMES
    ]


def build_h1_panel(
    prepared: Any,
    width_bps: int,
    *,
    stage_loader: Callable = load_aligned_stage,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[pd.Series]]]:
    timestamp, y_true, probabilities, _ = stage_loader(
        ARM, MODEL_NAMES, "oof_2024", width_bps=width_bps
    )
    oof_start = pd.to_datetime(timestamp, utc=True).min()
    oof_end = pd.to_datetime(timestamp, utc=True).max() + pd.Timedelta(minutes=15)
    scored = _score_model_stage(
        prepared, timestamp, y_true, probabilities,
        start=oof_start, end=oof_end, stage=f"oof_2024_w{width_bps}",
    )
    history: dict[str, list[pd.Series]] = {model: [scored[model]] for model in MODEL_NAMES}
    frames: list[pd.DataFrame] = []
    snapshots: list[dict[str, Any]] = []

    edges = pd.date_range(CALIBRATION_START, FORWARD_START, freq="MS")
    for stage, start, end in zip(H1_STAGES, edges[:-1], edges[1:]):
        timestamp, y_true, probabilities, _ = stage_loader(
            ARM, MODEL_NAMES, stage, width_bps=width_bps
        )
        weights, sharpes = positive_sharpe_weights(history)
        snapshots.extend(
            _snapshot_rows(
                weights, sharpes, width_bps=width_bps,
                effective_from=start, phase="h1",
            )
        )
        combined = combine_probabilities(probabilities, weights)
        frames.append(
            _raw_frame(timestamp, y_true, combined, f"causal-sharpe:{stage}:w{width_bps}")
        )
        scored = _score_model_stage(
            prepared, timestamp, y_true, probabilities,
            start=start, end=end, stage=f"{stage}_w{width_bps}",
        )
        for model in MODEL_NAMES:
            history[model].append(scored[model])
    return (
        pd.concat(frames, ignore_index=True).sort_values("timestamp"),
        pd.DataFrame(snapshots),
        history,
    )


def build_forward_panel(
    prepared: Any,
    width_bps: int,
    history: dict[str, list[pd.Series]],
    *,
    stage_loader: Callable = load_aligned_stage,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamp, y_true, probabilities, _ = stage_loader(
        ARM, MODEL_NAMES, "forward", width_bps=width_bps
    )
    timestamp = pd.to_datetime(timestamp, utc=True)
    frames: list[pd.DataFrame] = []
    snapshots: list[dict[str, Any]] = []
    edges = pd.date_range(FORWARD_START, FORWARD_END, freq="MS")
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (timestamp >= start) & (timestamp < end)
        stage_time = pd.Series(timestamp[mask]).reset_index(drop=True)
        stage_y = y_true.loc[mask].reset_index(drop=True)
        stage_probs = {model: values[np.asarray(mask)] for model, values in probabilities.items()}
        weights, sharpes = positive_sharpe_weights(history)
        snapshots.extend(
            _snapshot_rows(
                weights, sharpes, width_bps=width_bps,
                effective_from=start, phase="forward",
            )
        )
        frames.append(
            _raw_frame(
                stage_time,
                stage_y,
                combine_probabilities(stage_probs, weights),
                f"causal-sharpe:forward:{start:%Y-%m}:w{width_bps}",
            )
        )
        scored = _score_model_stage(
            prepared, stage_time, stage_y, stage_probs,
            start=start, end=end, stage=f"forward_{start:%Y_%m}_w{width_bps}",
        )
        for model in MODEL_NAMES:
            history[model].append(scored[model])
    return pd.concat(frames, ignore_index=True), pd.DataFrame(snapshots)


def _summary(
    ledger: pd.DataFrame,
    per_bar: pd.Series,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    **fields: Any,
) -> dict[str, Any]:
    metrics = economics_summary(per_bar)
    month_edges = pd.date_range(start, end, freq="MS")
    month_nets = [
        float(per_bar.loc[(per_bar.index >= left) & (per_bar.index < right)].sum())
        for left, right in zip(month_edges[:-1], month_edges[1:])
    ]
    return {
        **fields,
        "period_start": start,
        "period_end": end,
        "trades": int(len(ledger)),
        "n_long": int((ledger["side"] == 1).sum()) if len(ledger) else 0,
        "n_short": int((ledger["side"] == -1).sum()) if len(ledger) else 0,
        "positive_months": int(sum(value > 0.0 for value in month_nets)),
        "net_return": float(metrics["net_return_sum"]),
        "sortino": float(metrics["sortino"]),
        "sharpe": float(metrics["sharpe"]),
        "max_drawdown": float(metrics["max_drawdown"]),
    }


def _rank(row: Mapping[str, Any]) -> tuple[float, ...]:
    violation = float(
        max(0, 50 - int(row["trades"]))
        + max(0, 15 - int(row["n_long"]))
        + max(0, 15 - int(row["n_short"]))
        + 10 * max(0, 4 - int(row["positive_months"]))
    )
    robust = min(float(row["sortino"]), float(row["sharpe"]))
    return (
        violation,
        -robust,
        -float(row["sortino"]),
        -float(row["net_return"]),
        -int(row["trades"]),
        int(row["width_bps"]),
        float(row["tau_base"]),
        float(row["alpha"]),
    )


def _run_variant(
    prepared: Any,
    prediction: pd.DataFrame,
    context: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    tau_base: float,
    alpha: float,
    funding_filter: bool,
    dynamic_exits: bool,
    variant: str,
    width_bps: int,
    max_hold: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.Series, pd.DataFrame]:
    gated, diagnostics = apply_regime_gate(
        prediction,
        context,
        tau_base=tau_base,
        alpha=alpha,
        funding_filter=funding_filter,
    )
    trail = volatility_trail_bps(context) if dynamic_exits else None
    ledger, per_bar = _execute(
        prepared,
        gated,
        start=start,
        end=end,
        trail_bps=trail,
        be_trigger_bps=BE_TRIGGER_BPS if dynamic_exits else None,
        max_hold=max_hold,
    )
    summary = _summary(
        ledger,
        per_bar,
        start=start,
        end=end,
        variant=variant,
        width_bps=int(width_bps),
        tau_base=float(tau_base),
        alpha=float(alpha),
        max_hold=int(max_hold),
        funding_filter=bool(funding_filter),
        dynamic_exits=bool(dynamic_exits),
    )
    return summary, ledger, per_bar, diagnostics


def run(
    output_root: Path = DEFAULT_ROOT,
    *,
    stage_loader: Callable = load_aligned_stage,
    protocol: str = "causal-sharpe-regime-ensemble-v1",
) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json({"status": "running"}, output_root / "run_state.json")
    prepared = prepare_arm(ARM)
    required_features = {"ofi", "funding_z", "oi_z", "vol_20"}
    if not required_features.issubset(prepared.feature_columns):
        raise ValueError(
            f"Notebook 03b requires OHLCV/OFI/funding/OI context; missing "
            f"{sorted(required_features.difference(prepared.feature_columns))}"
        )

    h1_rows: list[dict[str, Any]] = []
    h1_panels: dict[int, pd.DataFrame] = {}
    h1_histories: dict[int, dict[str, list[pd.Series]]] = {}
    weight_frames: list[pd.DataFrame] = []
    for width_bps in WIDTHS:
        panel, weights, history = build_h1_panel(
            prepared, width_bps, stage_loader=stage_loader
        )
        h1_panels[width_bps] = panel
        h1_histories[width_bps] = history
        weight_frames.append(weights)
        context = prepared.features[width_bps][0]
        for tau_base in TAU_BASE_GRID:
            for alpha in ALPHA_GRID:
                for max_hold in MAX_HOLD_GRID:
                    summary, _, _, _ = _run_variant(
                        prepared,
                        panel,
                        context,
                        start=CALIBRATION_START,
                        end=FORWARD_START,
                        tau_base=tau_base,
                        alpha=alpha,
                        funding_filter=True,
                        dynamic_exits=True,
                        variant=f"full_hold{15 * max_hold}",
                        width_bps=width_bps,
                        max_hold=max_hold,
                    )
                    h1_rows.append(summary)

    h1_grid = pd.DataFrame(h1_rows)
    selected_by_hold = [
        min(group.to_dict("records"), key=_rank)
        for _, group in h1_grid.groupby("max_hold", sort=True)
    ]
    overall_selected = min(selected_by_hold, key=_rank)
    forward_panels: dict[int, pd.DataFrame] = {}
    for width_bps in sorted({int(row["width_bps"]) for row in selected_by_hold}):
        history = {model: list(parts) for model, parts in h1_histories[width_bps].items()}
        forward_panel, forward_weights = build_forward_panel(
            prepared, width_bps, history, stage_loader=stage_loader
        )
        forward_panels[width_bps] = forward_panel
        weight_frames.append(forward_weights)

    forward_rows: list[dict[str, Any]] = []
    trades: list[pd.DataFrame] = []
    returns = pd.DataFrame(index=prepared.bars.loc[
        (prepared.bars.index >= FORWARD_START) & (prepared.bars.index < FORWARD_END)
    ].index)
    gate_diagnostics: list[pd.DataFrame] = []
    forward_specs = [
        {
            "variant": f"full_hold{15 * int(row['max_hold'])}",
            "selection": row,
            "alpha": float(row["alpha"]),
            "funding_filter": True,
            "dynamic_exits": True,
        }
        for row in selected_by_hold
    ]
    forward_specs.extend(
        [
            {
                "variant": "static_tau",
                "selection": overall_selected,
                "alpha": 0.0,
                "funding_filter": True,
                "dynamic_exits": True,
            },
            {
                "variant": "no_funding_filter",
                "selection": overall_selected,
                "alpha": float(overall_selected["alpha"]),
                "funding_filter": False,
                "dynamic_exits": True,
            },
            {
                "variant": "fixed_exits",
                "selection": overall_selected,
                "alpha": float(overall_selected["alpha"]),
                "funding_filter": True,
                "dynamic_exits": False,
            },
        ]
    )
    for spec in forward_specs:
        selection = spec["selection"]
        width_bps = int(selection["width_bps"])
        variant = str(spec["variant"])
        summary, ledger, per_bar, diagnostics = _run_variant(
            prepared,
            forward_panels[width_bps],
            prepared.features[width_bps][0],
            start=FORWARD_START,
            end=FORWARD_END,
            tau_base=float(selection["tau_base"]),
            alpha=float(spec["alpha"]),
            funding_filter=bool(spec["funding_filter"]),
            dynamic_exits=bool(spec["dynamic_exits"]),
            variant=variant,
            width_bps=width_bps,
            max_hold=int(selection["max_hold"]),
        )
        forward_rows.append(summary)
        ledger = ledger.assign(variant=variant)
        trades.append(ledger)
        returns[variant] = per_bar.reindex(returns.index, fill_value=0.0)
        diagnostics["variant"] = variant
        gate_diagnostics.append(diagnostics)

    artifacts = {
        "h1_grid": h1_grid,
        "h1_selected": pd.DataFrame(selected_by_hold),
        "weight_snapshots": pd.concat(weight_frames, ignore_index=True),
        "forward_summary": pd.DataFrame(forward_rows),
        "forward_trades": pd.concat(trades, ignore_index=True),
        "forward_returns": returns.reset_index(names="timestamp"),
        "forward_gate_diagnostics": pd.concat(gate_diagnostics, ignore_index=True),
        "h1_selected_predictions": pd.concat(
            [
                h1_panels[int(row["width_bps"])].assign(max_hold=int(row["max_hold"]))
                for row in selected_by_hold
            ],
            ignore_index=True,
        ),
        "forward_predictions": pd.concat(
            [panel.assign(width_bps=width) for width, panel in forward_panels.items()],
            ignore_index=True,
        ),
    }
    for name, frame in artifacts.items():
        _atomic_parquet(frame, output_root / f"{name}.parquet")

    manifest = {
        "protocol": str(protocol),
        "arm": ARM,
        "base_models": list(MODEL_NAMES),
        "input_blocks": ["OHLCV", "funding", "open_interest", "OFI"],
        "weight_rule": "monthly max(net Sharpe, 0), normalized; all non-positive => flat",
        "weight_score_policy": {
            "tau": WEIGHT_TAU,
            "tp_bps": TP_BPS,
            "sl_bps": SL_BPS,
            "max_hold": WEIGHT_MAX_HOLD,
            "fee_bps_per_side": FEE_BPS,
        },
        "tau_base_grid": list(TAU_BASE_GRID),
        "alpha_grid": list(ALPHA_GRID),
        "max_hold_grid": list(MAX_HOLD_GRID),
        "funding_long_veto_z": FUNDING_Z_LIMIT,
        "breakeven_trigger_bps": BE_TRIGGER_BPS,
        "trailing_rule": (
            f"clip({TRAIL_BASE_BPS} * exp({TRAIL_VOL_BETA} * z_vol), "
            f"{TRAIL_MIN_BPS}, {TRAIL_MAX_BPS})"
        ),
        "selected_by_hold": selected_by_hold,
        "overall_h1_selected": overall_selected,
        "h1_period": "2025-01-01 to 2025-07-01 exclusive",
        "forward_period": "2025-07-01 to 2026-04-01 exclusive",
        "forward_replay_count": 1,
        "lockbox_2026_q2_used": False,
        "artifact_rows": {name: len(frame) for name, frame in artifacts.items()},
    }
    _atomic_json(manifest, output_root / "manifest.json")
    _atomic_json({"status": "complete", **manifest}, output_root / "run_state.json")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output_root), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
