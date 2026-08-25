"""Run the bounded calibrated XGBoost add-on contest after frozen Union v1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.economics import economics_summary
from evaluation.trades_intrabar import simulate_bracket_trades_intrabar
from experiments.qualified_union import (
    BARS_PATH,
    CODE_ROOT,
    FEE_BPS,
    FORWARD_START,
    H1_START,
    LOCKBOX_START,
    MAX_HOLD,
    MINUTE_PATH,
    SL_BPS,
    TP_BPS,
    summarize,
)
from experiments.xgb_strong_move_admission import (
    DIRECTION_THRESHOLDS,
    MOVE_THRESHOLDS,
    OUTPUT_ROOT,
    apply_calibrators,
    binary_metrics,
    build_addon_signal,
    combine_with_addon,
    decompose_xgb_probabilities,
    fit_calibrators,
    load_xgb_panel,
    reliability_table,
)


UNION_CACHE = CODE_ROOT / "experiments" / "cache" / "qualified_union_v1"
SVM_CACHE = CODE_ROOT / "experiments" / "cache" / "svm_temperature_calibration"
SELECTION_END = pd.Timestamp("2025-04-01", tz="UTC")
CONFIRMATION_END = FORWARD_START
FORWARD_FILES = (
    "forward_addon_ledger.parquet",
    "forward_combined_ledger.parquet",
    "forward_addon_per_bar.parquet",
    "forward_combined_per_bar.parquet",
    "forward_signals.parquet",
    "forward_summary.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_union(stage: str) -> pd.DataFrame:
    path = UNION_CACHE / f"{stage}_signals.parquet"
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame = frame.set_index("timestamp").sort_index()
    end = FORWARD_START if stage == "h1" else LOCKBOX_START
    if frame.index.max() >= end:
        raise ValueError(f"{stage} Union artifact crosses its frozen boundary")
    return frame


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
        raise ValueError(f"bounded market read failed: {path.name}")
    return frame.sort_index()


def _market_slice(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    if end > LOCKBOX_START:
        raise ValueError("market slice cannot cross Q2-2026")
    return _read_market(BARS_PATH, start, end), _read_market(MINUTE_PATH, start, end)


def _simulate(
    signal: pd.Series,
    bars: pd.DataFrame,
    minute: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    scoped = signal.reindex(bars.index).fillna(0.0)
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
        ledger.insert(
            0,
            "signal_time",
            pd.to_datetime(ledger["entry_time"], utc=True) - pd.Timedelta(minutes=15),
        )
    else:
        ledger.insert(0, "signal_time", pd.Series(dtype="datetime64[ns, UTC]"))
    if not np.isclose(per_bar.sum(), ledger["net_return"].sum(), atol=1e-10):
        raise AssertionError("admission replay does not reconcile")
    return ledger, per_bar.rename("net_return")


def _prefix_summary(
    prefix: str,
    ledger: pd.DataFrame,
    per_bar: pd.Series,
    *,
    phase: str,
) -> dict[str, object]:
    row = summarize(ledger, per_bar, phase=phase)
    keep = {
        "trades": "trades",
        "long_trades": "long_trades",
        "short_trades": "short_trades",
        "gross_bps_per_trade": "gross_bps_per_trade",
        "net_return": "net",
        "sortino": "sortino",
        "sharpe": "sharpe",
        "max_drawdown": "max_drawdown",
    }
    return {f"{prefix}_{target}": row[source] for source, target in keep.items()}


def _eligible(row: dict[str, object]) -> bool:
    gross = float(row["addon_gross_bps_per_trade"])
    return bool(
        int(row["addon_trades"]) >= 10
        and int(row["addon_long_trades"]) >= 2
        and int(row["addon_short_trades"]) >= 2
        and np.isfinite(gross)
        and gross > 10.0
        and float(row["addon_net"]) > 0.0
        and float(row["combined_net"]) > float(row["baseline_net"])
        and float(row["combined_sortino"]) >= float(row["baseline_sortino"])
    )


def _evaluate_policy(
    union_frame: pd.DataFrame,
    xgb_frame: pd.DataFrame,
    *,
    move_threshold: float,
    direction_threshold: float,
    bars: pd.DataFrame,
    minute: pd.DataFrame,
    baseline_metrics: dict[str, object],
    phase: str,
) -> tuple[dict[str, object], dict[str, object]]:
    addon = build_addon_signal(
        union_frame,
        xgb_frame,
        move_threshold=move_threshold,
        direction_threshold=direction_threshold,
    )
    combined = combine_with_addon(union_frame["union_signal"], addon)
    addon_ledger, addon_per_bar = _simulate(addon, bars, minute)
    combined_ledger, combined_per_bar = _simulate(combined, bars, minute)
    row: dict[str, object] = {
        "phase": phase,
        "move_threshold": float(move_threshold),
        "direction_threshold": float(direction_threshold),
        "addon_signal_bars": int(addon.reindex(bars.index).fillna(0.0).ne(0.0).sum()),
        **baseline_metrics,
        **_prefix_summary("addon", addon_ledger, addon_per_bar, phase=phase),
        **_prefix_summary("combined", combined_ledger, combined_per_bar, phase=phase),
    }
    row["eligible"] = _eligible(row)
    evidence = {
        "addon_signal": addon,
        "combined_signal": combined,
        "addon_ledger": addon_ledger,
        "addon_per_bar": addon_per_bar,
        "combined_ledger": combined_ledger,
        "combined_per_bar": combined_per_bar,
    }
    return row, evidence


def _baseline_metrics(
    union_frame: pd.DataFrame,
    bars: pd.DataFrame,
    minute: pd.DataFrame,
    *,
    phase: str,
) -> dict[str, object]:
    ledger, per_bar = _simulate(union_frame["union_signal"], bars, minute)
    return _prefix_summary("baseline", ledger, per_bar, phase=phase)


def _calibration_artifacts(
    frame: pd.DataFrame,
    *,
    period: str,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    decomposed = decompose_xgb_probabilities(frame)
    labels = frame["y_true"].astype(int)
    move_target = labels.ne(1).astype(int)
    directional = move_target.eq(1)
    rows: list[dict[str, object]] = []
    reliability: list[pd.DataFrame] = []
    definitions = (
        (
            "move",
            move_target,
            decomposed["p_move_raw"],
            frame["p_move_cal"],
        ),
        (
            "direction_given_move",
            labels.loc[directional].eq(2).astype(int),
            decomposed.loc[directional, "p_long_given_move_raw"],
            frame.loc[directional, "p_long_given_move_cal"],
        ),
    )
    for target_name, target, raw, calibrated in definitions:
        for metric in binary_metrics(target, raw, calibrated):
            rows.append({"period": period, "target": target_name, **metric})
        reliability.extend(
            [
                reliability_table(
                    target,
                    raw,
                    period=period,
                    target_name=target_name,
                    arm="raw",
                ),
                reliability_table(
                    target,
                    calibrated,
                    period=period,
                    target_name=target_name,
                    arm="calibrated",
                ),
            ]
        )
    return rows, reliability


def _remove_stale_forward() -> None:
    root = OUTPUT_ROOT.resolve()
    for filename in FORWARD_FILES:
        path = (OUTPUT_ROOT / filename).resolve()
        if path.parent != root:
            raise ValueError("forward cleanup escaped the admission cache")
        if path.exists():
            path.unlink()


def _store_evidence(prefix: str, evidence: dict[str, object], bars: pd.DataFrame) -> None:
    signal = pd.DataFrame(
        {
            "timestamp": bars.index,
            "addon_signal": pd.Series(evidence["addon_signal"]).reindex(bars.index).fillna(0.0).to_numpy(),
            "combined_signal": pd.Series(evidence["combined_signal"]).reindex(bars.index).fillna(0.0).to_numpy(),
        }
    )
    signal.to_parquet(OUTPUT_ROOT / f"{prefix}_signals.parquet", index=False)
    for name in ("addon", "combined"):
        pd.DataFrame(evidence[f"{name}_ledger"]).to_parquet(
            OUTPUT_ROOT / f"{prefix}_{name}_ledger.parquet", index=False
        )
        pd.Series(evidence[f"{name}_per_bar"]).rename_axis("timestamp").reset_index().to_parquet(
            OUTPUT_ROOT / f"{prefix}_{name}_per_bar.parquet", index=False
        )


def run() -> dict[str, object]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _remove_stale_forward()

    svm_summary = json.loads((SVM_CACHE / "summary.json").read_text(encoding="utf-8"))
    if svm_summary["selected_temperature"] != 1.0 or not svm_summary["h1_confirmation_pass"]:
        raise ValueError("04b requires the completed class-preserving 04a handoff")
    union_manifest = json.loads((UNION_CACHE / "manifest.json").read_text(encoding="utf-8"))
    if union_manifest["decision"] != "freeze_union_v1":
        raise ValueError("04b requires frozen Union v1")

    oof, oof_paths = load_xgb_panel("2024_oof")
    calibrators = fit_calibrators(oof)
    oof_calibrated = apply_calibrators(oof, calibrators)
    h1, h1_paths = load_xgb_panel("h1")
    h1_calibrated = apply_calibrators(h1, calibrators)
    union_h1 = _load_union("h1")
    if not h1_calibrated.index.equals(union_h1.index):
        raise ValueError("XGBoost H1 and frozen Union grids differ")

    calibration_rows: list[dict[str, object]] = []
    reliability_frames: list[pd.DataFrame] = []
    for period, frame in (("2024_oof", oof_calibrated), ("2025_h1", h1_calibrated)):
        rows, reliability = _calibration_artifacts(frame, period=period)
        calibration_rows.extend(rows)
        reliability_frames.extend(reliability)
    pd.DataFrame(calibration_rows).to_csv(
        OUTPUT_ROOT / "calibration_metrics.csv", index=False
    )
    pd.concat(reliability_frames, ignore_index=True).to_csv(
        OUTPUT_ROOT / "reliability_bins.csv", index=False
    )
    _write_json(
        OUTPUT_ROOT / "calibrators.json",
        {name: calibrator.to_dict() for name, calibrator in calibrators.items()},
    )
    h1_export = h1_calibrated.reset_index()[
        [
            "timestamp",
            "y_true",
            "pred",
            "p_move_raw",
            "p_move_cal",
            "p_long_given_move_raw",
            "p_long_given_move_cal",
            "xgb_latent_side",
            "xgb_direction_confidence",
        ]
    ]
    h1_export.to_parquet(OUTPUT_ROOT / "calibrated_h1.parquet", index=False)

    selection_bars, selection_minute = _market_slice(H1_START, SELECTION_END)
    selection_baseline = _baseline_metrics(
        union_h1,
        selection_bars,
        selection_minute,
        phase="2025_q1_selection",
    )
    grid_rows = []
    for move_threshold in MOVE_THRESHOLDS:
        for direction_threshold in DIRECTION_THRESHOLDS:
            row, _ = _evaluate_policy(
                union_h1,
                h1_calibrated,
                move_threshold=move_threshold,
                direction_threshold=direction_threshold,
                bars=selection_bars,
                minute=selection_minute,
                baseline_metrics=selection_baseline,
                phase="2025_q1_selection",
            )
            grid_rows.append(row)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(OUTPUT_ROOT / "h1_selection_grid.csv", index=False)
    eligible = grid.loc[grid["eligible"]].sort_values(
        ["combined_trades", "combined_net", "move_threshold", "direction_threshold"],
        ascending=[False, False, False, False],
    )
    diagnostic = grid.sort_values(
        ["combined_net", "combined_trades"], ascending=[False, False]
    ).iloc[0]

    selected_policy: dict[str, float] | None = None
    confirmation_row: dict[str, object] | None = None
    full_h1_row: dict[str, object] | None = None
    full_h1_evidence: dict[str, object] | None = None
    h1_pass = False
    if len(eligible):
        chosen = eligible.iloc[0]
        selected_policy = {
            "move_threshold": float(chosen["move_threshold"]),
            "direction_threshold": float(chosen["direction_threshold"]),
        }
        confirmation_bars, confirmation_minute = _market_slice(
            SELECTION_END, CONFIRMATION_END
        )
        confirmation_baseline = _baseline_metrics(
            union_h1,
            confirmation_bars,
            confirmation_minute,
            phase="2025_q2_confirmation",
        )
        confirmation_row, _ = _evaluate_policy(
            union_h1,
            h1_calibrated,
            **selected_policy,
            bars=confirmation_bars,
            minute=confirmation_minute,
            baseline_metrics=confirmation_baseline,
            phase="2025_q2_confirmation",
        )
        full_bars, full_minute = _market_slice(H1_START, CONFIRMATION_END)
        full_baseline = _baseline_metrics(
            union_h1, full_bars, full_minute, phase="2025_h1_full"
        )
        full_h1_row, full_h1_evidence = _evaluate_policy(
            union_h1,
            h1_calibrated,
            **selected_policy,
            bars=full_bars,
            minute=full_minute,
            baseline_metrics=full_baseline,
            phase="2025_h1_full",
        )
        full_growth_pass = bool(
            int(full_h1_row["combined_trades"]) >= 132
            and float(full_h1_row["addon_net"]) > 0.0
            and float(full_h1_row["addon_gross_bps_per_trade"]) > 10.0
        )
        h1_pass = bool(
            chosen["eligible"]
            and confirmation_row["eligible"]
            and full_growth_pass
        )
        _store_evidence("h1", full_h1_evidence, full_bars)

    pd.DataFrame([confirmation_row] if confirmation_row else [], columns=grid.columns).to_csv(
        OUTPUT_ROOT / "h1_confirmation.csv", index=False
    )
    pd.DataFrame([full_h1_row] if full_h1_row else [], columns=grid.columns).to_csv(
        OUTPUT_ROOT / "h1_full.csv", index=False
    )

    forward_loaded = False
    forward_promoted = False
    forward_row: dict[str, object] | None = None
    source_paths = [*oof_paths, *h1_paths]
    if h1_pass and selected_policy is not None:
        xgb_forward, forward_paths = load_xgb_panel("forward")
        source_paths.extend(forward_paths)
        xgb_forward = apply_calibrators(xgb_forward, calibrators)
        union_forward = _load_union("forward")
        if not xgb_forward.index.equals(union_forward.index):
            raise ValueError("XGBoost forward and frozen Union grids differ")
        forward_bars, forward_minute = _market_slice(FORWARD_START, LOCKBOX_START)
        forward_baseline = _baseline_metrics(
            union_forward,
            forward_bars,
            forward_minute,
            phase="development_forward",
        )
        forward_row, forward_evidence = _evaluate_policy(
            union_forward,
            xgb_forward,
            **selected_policy,
            bars=forward_bars,
            minute=forward_minute,
            baseline_metrics=forward_baseline,
            phase="development_forward",
        )
        forward_loaded = True
        forward_promoted = bool(
            int(forward_row["combined_trades"]) >= 111
            and float(forward_row["combined_net"]) > float(forward_row["baseline_net"])
            and float(forward_row["combined_sortino"]) >= float(forward_row["baseline_sortino"])
            and float(forward_row["combined_max_drawdown"])
            <= float(forward_row["baseline_max_drawdown"])
            and float(forward_row["addon_net"]) > 0.0
            and float(forward_row["addon_gross_bps_per_trade"]) > 10.0
            and int(forward_row["addon_long_trades"]) > 0
            and int(forward_row["addon_short_trades"]) > 0
        )
        _store_evidence("forward", forward_evidence, forward_bars)
        pd.DataFrame([forward_row]).to_csv(OUTPUT_ROOT / "forward_summary.csv", index=False)

    max_loaded = h1_calibrated.index.max()
    if forward_loaded:
        max_loaded = pd.Timestamp(forward_row and LOCKBOX_START - pd.Timedelta(minutes=15))
    summary = {
        "study": "calibrated_xgboost_strong_move_admission",
        "xgboost_model": "xgboost_balanced_dz65",
        "calibration_fit_period": "2024_oof",
        "policy_selection_period": "2025_q1",
        "policy_confirmation_period": "2025_q2_calendar",
        "move_thresholds": list(MOVE_THRESHOLDS),
        "direction_thresholds": list(DIRECTION_THRESHOLDS),
        "calibrators": {name: value.to_dict() for name, value in calibrators.items()},
        "selected_policy": selected_policy,
        "eligible_selection_policies": int(len(eligible)),
        "diagnostic_leader": {
            "move_threshold": float(diagnostic["move_threshold"]),
            "direction_threshold": float(diagnostic["direction_threshold"]),
            "combined_trades": int(diagnostic["combined_trades"]),
            "combined_net": float(diagnostic["combined_net"]),
            "addon_trades": int(diagnostic["addon_trades"]),
            "addon_net": float(diagnostic["addon_net"]),
            "eligible": bool(diagnostic["eligible"]),
        },
        "selection_result": None if selected_policy is None else {key: chosen[key].item() if hasattr(chosen[key], "item") else chosen[key] for key in grid.columns},
        "confirmation_result": confirmation_row,
        "full_h1_result": full_h1_row,
        "h1_pass": h1_pass,
        "forward_loaded": forward_loaded,
        "forward_result": forward_row,
        "forward_promoted": forward_promoted,
        "decision": "promote_union_v2" if forward_promoted else "reject_xgboost_keep_union_v1",
        "final_ensemble": "union_v2_with_xgboost" if forward_promoted else "qualified_union_v1",
        "lockbox_2026_q2_used": False,
        "max_loaded_timestamp": str(max_loaded),
    }
    _write_json(OUTPUT_ROOT / "summary.json", summary)

    artifact_paths = sorted(path for path in OUTPUT_ROOT.iterdir() if path.name != "manifest.json")
    dependency_paths = [
        UNION_CACHE / "manifest.json",
        SVM_CACHE / "summary.json",
        *source_paths,
    ]
    manifest = {
        "dependency_hashes": {
            str(path.relative_to(CODE_ROOT)).replace("\\", "/"): _sha256(path)
            for path in dependency_paths
        },
        "artifact_hashes": {path.name: _sha256(path) for path in artifact_paths},
        "forward_loaded": forward_loaded,
        "lockbox_2026_q2_used": False,
        "decision": summary["decision"],
    }
    _write_json(OUTPUT_ROOT / "manifest.json", manifest)
    return summary


def main() -> int:
    summary = run()
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
