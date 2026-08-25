"""Run the staged Notebook 04c channel-blind opportunity experiment."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from typing import TypeVar

import numpy as np
import pandas as pd

from experiments.channel_blind_union_addon import (
    AddonConfig,
    align_union_asof,
    build_portfolio_series,
    causal_crossings,
    concurrency_audit,
    h1_access_gate,
    qualify_union_side,
    replay_addons,
    select_frequency_threshold,
)
from experiments.qualified_union import summarize


CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = CODE_ROOT / "data"
W_RUN_HASH = "1321c7ed5547c321259e"
W_ROOT = (
    CODE_ROOT
    / "experiments"
    / "cache"
    / "channel_vs_volatility_ablation"
    / W_RUN_HASH
    / "full"
)
W_OOF_PATH = W_ROOT / "oof_predictions.parquet"
W_OOF_SHA256 = (
    "3643e8939b302d4e96da53d2513522c6685102e45620aa10ad9ea5941db11e72"
)
UNION_CACHE = CODE_ROOT / "experiments" / "cache" / "qualified_union_v1"
OUTPUT_ROOT = CODE_ROOT / "experiments" / "cache" / "channel_blind_union_addon"
LOCKBOX_START = pd.Timestamp("2026-04-01", tz="UTC")
H1_START = pd.Timestamp("2025-01-01", tz="UTC")
H1_END = pd.Timestamp("2025-07-01", tz="UTC")
FORWARD_START = H1_END
FORWARD_FILENAMES = (
    "forward_threshold_calibration.csv",
    "forward_candidate_funnel.csv",
    "forward_addon_ledger.parquet",
    "forward_addon_per_bar.parquet",
    "forward_union_reference.parquet",
    "forward_combined_per_bar.parquet",
    "forward_summary.csv",
    "forward_monthly.csv",
    "forward_scores.parquet",
)

T = TypeVar("T")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hashed_w_oof() -> pd.DataFrame:
    """Load only the pinned XGBoost calibration and H1 OOF rows."""
    actual = sha256(W_OOF_PATH)
    if actual != W_OOF_SHA256:
        raise ValueError(f"pinned W OOF hash changed: {actual}")
    frame = pd.read_parquet(W_OOF_PATH)
    frame["decision_time"] = pd.to_datetime(
        frame["decision_time"], utc=True, errors="raise"
    )
    selected = frame.loc[
        frame["model"].eq("xgboost")
        & frame["fold_id"].isin(("2024H2", "2025H1"))
    ].copy()
    if selected.duplicated(["model", "decision_time"]).any():
        raise ValueError("pinned W OOF rows contain duplicate decision times")
    expected = {"2024H2": 52_992, "2025H1": 52_128}
    counts = selected.groupby("fold_id").size().to_dict()
    if counts != expected:
        raise ValueError(f"pinned W fold counts changed: {counts}")
    return selected.sort_values(["fold_id", "decision_time"]).reset_index(drop=True)


def maybe_run_forward(
    h1_summary: Mapping[str, object],
    loader: Callable[[], T],
) -> T | None:
    """Keep the forward loader unreachable until the registered H1 gate passes."""
    if not h1_access_gate(h1_summary):
        return None
    return loader()


def _read_bounded(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        filters=[
            ("timestamp", ">=", start.to_pydatetime()),
            ("timestamp", "<", end.to_pydatetime()),
        ],
    )
    frame.index = pd.to_datetime(frame.index, utc=True, errors="raise")
    frame = frame.sort_index(kind="stable")
    if (
        frame.empty
        or frame.index.min() < start
        or frame.index.max() >= end
        or frame.index.has_duplicates
    ):
        raise ValueError(f"bounded source read failed: {path.name}")
    return frame


def load_bounded_sources(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    data_root: Path = DATA_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read minute, five-minute and positioning rows without touching Q2."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("bounded source timestamps must be timezone-aware")
    start = start.tz_convert("UTC")
    end = end.tz_convert("UTC")
    if start >= end:
        raise ValueError("bounded source interval must be positive")
    if end > LOCKBOX_START:
        raise ValueError("bounded source read cannot cross Q2-2026")
    root = Path(data_root)
    minute = _read_bounded(
        root / "btcusdt_1m_2021_2026.parquet", start, end
    )
    five_minute = _read_bounded(
        root / "btcusdt_5min_2021_2026.parquet", start, end
    )
    positioning = _read_bounded(
        root / "btcusdt_positioning_15min_2021_2026.parquet", start, end
    )
    return minute, five_minute, positioning


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    Path(path).write_text(
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def load_frozen_union(
    stage: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Verify the entire frozen manifest, then load one immutable stage."""
    if stage not in {"h1", "forward"}:
        raise ValueError("Union stage must be h1 or forward")
    manifest = _read_json(UNION_CACHE / "manifest.json")
    if manifest.get("decision") != "freeze_union_v1":
        raise ValueError("Notebook 04c requires frozen Qualified Union v1")
    for filename, expected in dict(manifest["artifact_hashes"]).items():
        path = UNION_CACHE / filename
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"frozen Union artifact changed: {filename}")

    signals = pd.read_parquet(UNION_CACHE / f"{stage}_signals.parquet")
    ledger = pd.read_parquet(UNION_CACHE / f"{stage}_ledger.parquet")
    per_bar_frame = pd.read_parquet(UNION_CACHE / f"{stage}_per_bar.parquet")
    signals["timestamp"] = pd.to_datetime(
        signals["timestamp"], utc=True, errors="raise"
    )
    for column in (
        "signal_time",
        "entry_time",
        "exit_time",
        "intrabar_exit_time",
    ):
        if column in ledger:
            ledger[column] = pd.to_datetime(
                ledger[column], utc=True, errors="raise"
            )
    per_bar_frame["timestamp"] = pd.to_datetime(
        per_bar_frame["timestamp"], utc=True, errors="raise"
    )
    per_bar = per_bar_frame.set_index("timestamp")["net_return"].astype(float)
    per_bar.name = "net_return"
    boundary = H1_END if stage == "h1" else LOCKBOX_START
    if (
        signals["timestamp"].max() >= boundary
        or per_bar.index.max() >= boundary
        or (
            len(ledger)
            and pd.to_datetime(
                ledger["intrabar_exit_time"], utc=True, errors="raise"
            ).max()
            >= boundary
        )
    ):
        raise ValueError(f"frozen {stage} Union artifact crosses its boundary")
    if not np.isclose(
        per_bar.sum(),
        pd.to_numeric(ledger["net_return"], errors="raise").sum(),
        atol=1e-12,
    ):
        raise AssertionError(f"frozen {stage} Union ledger does not reconcile")
    return signals, ledger, per_bar


def _prefixed_summary(
    prefix: str,
    ledger: pd.DataFrame,
    per_bar: pd.Series,
    *,
    phase: str,
) -> dict[str, object]:
    row = summarize(ledger, per_bar, phase=phase)
    return {
        f"{prefix}_trades": int(row["trades"]),
        f"{prefix}_long_trades": int(row["long_trades"]),
        f"{prefix}_short_trades": int(row["short_trades"]),
        f"{prefix}_gross_bps_per_trade": float(row["gross_bps_per_trade"]),
        f"{prefix}_net_return": float(row["net_return"]),
        f"{prefix}_sortino": float(row["sortino"]),
        f"{prefix}_sharpe": float(row["sharpe"]),
        f"{prefix}_max_drawdown": float(row["max_drawdown"]),
    }


def _monthly_table(
    phase: str,
    series_by_arm: Mapping[str, pd.Series],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for arm, values in series_by_arm.items():
        monthly = values.groupby(
            values.index.tz_convert(None).to_period("M")
        ).sum()
        rows.extend(
            {
                "phase": phase,
                "arm": arm,
                "month": str(month),
                "net_return": float(value),
            }
            for month, value in monthly.items()
        )
    return pd.DataFrame(rows)


def evaluate_stage(
    activations: pd.DataFrame,
    union_signals: pd.DataFrame,
    union_ledger: pd.DataFrame,
    union_per_bar: pd.Series,
    minute: pd.DataFrame,
    config: AddonConfig,
    phase: str,
) -> dict[str, object]:
    """Qualify, replay, account, and summarise one fixed activation stream."""
    raw = activations.sort_values("decision_time").reset_index(drop=True)
    finite_path = (
        raw["path_complete"].astype(bool)
        & np.isfinite(pd.to_numeric(raw["reference_price"], errors="coerce"))
        & np.isfinite(
            pd.to_numeric(raw["adaptive_barrier_bps"], errors="coerce")
        )
    )
    replayable = raw.loc[finite_path].copy()
    aligned = align_union_asof(replayable, union_signals)
    qualified = qualify_union_side(aligned, union_ledger)
    addon_ledger = replay_addons(qualified, minute, config)
    addon_per_bar, combined_per_bar = build_portfolio_series(
        union_per_bar, addon_ledger
    )
    combined_ledger = pd.concat(
        [union_ledger, addon_ledger],
        ignore_index=True,
        sort=False,
    )

    summary = {
        "phase": phase,
        "raw_activations": int(len(raw)),
        "path_complete_activations": int(finite_path.sum()),
        **_prefixed_summary(
            "union", union_ledger, union_per_bar, phase=phase
        ),
        **_prefixed_summary(
            "addon", addon_ledger, addon_per_bar, phase=phase
        ),
        **_prefixed_summary(
            "combined", combined_ledger, combined_per_bar, phase=phase
        ),
    }
    april = pd.Timestamp("2025-04-01", tz="UTC")
    summary["apr_jun_addon_net_return"] = float(
        addon_per_bar.loc[addon_per_bar.index >= april].sum()
        if phase == "h1"
        else np.nan
    )
    summary["opportunity_rate"] = float(
        pd.to_numeric(raw.loc[finite_path, "opportunity"], errors="coerce").mean()
    )
    resolved = (
        addon_ledger["direction_valid"].astype(bool)
        if len(addon_ledger) and "direction_valid" in addon_ledger
        else pd.Series(dtype=bool)
    )
    if resolved.any():
        expected_side = np.where(
            addon_ledger.loc[resolved, "direction_up"].astype(int).eq(1),
            1,
            -1,
        )
        summary["side_accuracy_resolved"] = float(
            np.mean(
                addon_ledger.loc[resolved, "side"].astype(int).to_numpy()
                == expected_side
            )
        )
    else:
        summary["side_accuracy_resolved"] = np.nan
    summary["mean_net_r"] = float(
        pd.to_numeric(addon_ledger["net_r"], errors="coerce").mean()
        if len(addon_ledger)
        else np.nan
    )
    summary.update(concurrency_audit(union_ledger, addon_ledger))

    funnel_rows = [
        {"phase": phase, "decision": "raw_activation", "count": len(raw)},
        {
            "phase": phase,
            "decision": "reject_incomplete_path",
            "count": int((~finite_path).sum()),
        },
    ]
    funnel_rows.extend(
        {
            "phase": phase,
            "decision": str(decision),
            "count": int(count),
        }
        for decision, count in qualified["decision"].value_counts().items()
    )
    funnel_rows.extend(
        [
            {
                "phase": phase,
                "decision": "reject_addon_overlap",
                "count": int(
                    qualified["decision"].eq("accept").sum()
                    - len(addon_ledger)
                ),
            },
            {
                "phase": phase,
                "decision": "executed_addon",
                "count": int(len(addon_ledger)),
            },
        ]
    )
    return {
        "summary": summary,
        "funnel": pd.DataFrame(funnel_rows),
        "qualified": qualified,
        "addon_ledger": addon_ledger,
        "addon_per_bar": addon_per_bar,
        "combined_per_bar": combined_per_bar,
        "monthly": _monthly_table(
            phase,
            {
                "union_v1": union_per_bar,
                "addon": addon_per_bar,
                "combined": combined_per_bar,
            },
        ),
    }


def _remove_stale_forward() -> None:
    root = OUTPUT_ROOT.resolve()
    for filename in FORWARD_FILENAMES:
        path = (OUTPUT_ROOT / filename).resolve()
        if path.parent != root:
            raise ValueError("forward cleanup escaped the 04c cache")
        if path.exists():
            path.unlink()


def _write_series(path: Path, values: pd.Series) -> None:
    values.rename("net_return").rename_axis("timestamp").reset_index().to_parquet(
        path, index=False
    )


def _write_manifest(
    summary: Mapping[str, object],
    dependency_paths: list[Path],
) -> dict[str, object]:
    artifact_paths = sorted(
        path
        for path in OUTPUT_ROOT.iterdir()
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "dependency_hashes": {
            str(path.relative_to(CODE_ROOT)).replace("\\", "/"): sha256(path)
            for path in dependency_paths
        },
        "artifact_hashes": {
            path.name: sha256(path) for path in artifact_paths
        },
        "forward_loaded": bool(summary["forward_loaded"]),
        "lockbox_2026_q2_used": False,
        "max_loaded_timestamp": summary["max_loaded_timestamp"],
        "decision": summary["decision"],
    }
    _write_json(OUTPUT_ROOT / "manifest.json", manifest)
    return manifest


def run_h1(
    config: AddonConfig = AddonConfig(),
) -> dict[str, object]:
    """Materialise Stage A without exposing any forward input."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _remove_stale_forward()
    predictions = load_hashed_w_oof()
    calibration = predictions.loc[predictions["fold_id"].eq("2024H2")]
    calibration_scores = calibration.set_index("decision_time")[
        "opportunity_score"
    ]
    threshold, frontier = select_frequency_threshold(calibration_scores, config)
    h1 = predictions.loc[predictions["fold_id"].eq("2025H1")].copy()
    h1_scores = h1.set_index("decision_time")["opportunity_score"]
    activation_times = causal_crossings(
        h1_scores,
        threshold=threshold,
        refractory=pd.Timedelta(minutes=config.refractory_minutes),
    )
    allowed_columns = [
        "decision_time",
        "reference_price",
        "adaptive_barrier_bps",
        "opportunity_score",
        "opportunity",
        "direction_valid",
        "direction_up",
        "path_complete",
    ]
    activations = h1.loc[
        h1["decision_time"].isin(activation_times), allowed_columns
    ].copy()

    union_signals, union_ledger, union_per_bar = load_frozen_union("h1")
    minute, five_minute, positioning = load_bounded_sources(H1_START, H1_END)
    evidence = evaluate_stage(
        activations,
        union_signals,
        union_ledger,
        union_per_bar,
        minute,
        config,
        "h1",
    )
    h1_summary = dict(evidence["summary"])
    h1_pass = h1_access_gate(h1_summary)
    selected_row = frontier.loc[frontier["selected"]].iloc[0]
    max_loaded = max(
        minute.index.max(),
        five_minute.index.max(),
        positioning.index.max(),
        h1["decision_time"].max(),
        union_signals["timestamp"].max(),
    )
    summary = {
        "study": "channel_blind_union_addon",
        "w_run_hash": W_RUN_HASH,
        "w_oof_sha256": W_OOF_SHA256,
        "threshold_calibration_period": "2024H2",
        "h1_evaluation_period": "2025H1",
        "h1_threshold": float(threshold),
        "h1_calibration_raw_activations": int(selected_row["activations"]),
        "h1_calibration_activations_per_day": float(
            selected_row["activations_per_day"]
        ),
        "h1_result": h1_summary,
        "h1_pass": bool(h1_pass),
        "forward_loaded": False,
        "forward_promoted": False,
        "lockbox_2026_q2_used": False,
        "max_loaded_timestamp": max_loaded.isoformat(),
        "decision": (
            "h1_pass_open_forward"
            if h1_pass
            else "h1_fail_keep_union_v1"
        ),
        "final_ensemble": "qualified_union_v1",
    }
    protocol = {
        "study": "notebook_04c_channel_blind_union_addon",
        **asdict(config),
        "w_run_hash": W_RUN_HASH,
        "w_oof_sha256": W_OOF_SHA256,
        "calibration_fold": "2024H2",
        "h1_fold": "2025H1",
        "union_side_availability_lag_minutes": 15,
        "score_is_calibrated_probability": False,
        "offline_topk_used": False,
        "direction_model_used": False,
        "entry": "native one-minute Open at decision_time",
        "same_minute_ambiguity": "stop_first",
        "round_trip_cost_bps": config.entry_cost_bps + config.exit_cost_bps,
        "lockbox_start": LOCKBOX_START.isoformat(),
    }

    _write_json(OUTPUT_ROOT / "protocol.json", protocol)
    frontier.to_csv(
        OUTPUT_ROOT / "h1_threshold_calibration.csv", index=False
    )
    pd.DataFrame(evidence["funnel"]).to_csv(
        OUTPUT_ROOT / "h1_candidate_funnel.csv", index=False
    )
    pd.DataFrame(evidence["addon_ledger"]).to_parquet(
        OUTPUT_ROOT / "h1_addon_ledger.parquet", index=False
    )
    _write_series(
        OUTPUT_ROOT / "h1_addon_per_bar.parquet",
        pd.Series(evidence["addon_per_bar"]),
    )
    shutil.copyfile(
        UNION_CACHE / "h1_per_bar.parquet",
        OUTPUT_ROOT / "h1_union_reference.parquet",
    )
    _write_series(
        OUTPUT_ROOT / "h1_combined_per_bar.parquet",
        pd.Series(evidence["combined_per_bar"]),
    )
    pd.DataFrame([h1_summary]).to_csv(
        OUTPUT_ROOT / "h1_summary.csv", index=False
    )
    pd.DataFrame(evidence["monthly"]).to_csv(
        OUTPUT_ROOT / "h1_monthly.csv", index=False
    )
    _write_json(OUTPUT_ROOT / "summary.json", summary)
    _write_manifest(
        summary,
        [
            W_OOF_PATH,
            UNION_CACHE / "manifest.json",
            UNION_CACHE / "h1_signals.parquet",
            UNION_CACHE / "h1_ledger.parquet",
            UNION_CACHE / "h1_per_bar.parquet",
        ],
    )
    return summary


def run() -> dict[str, object]:
    """Run the fixed experiment and enforce its observed pre-forward stop."""
    summary = run_h1()

    def unexpected_forward_open() -> dict[str, object]:
        raise AssertionError(
            "pinned H1 unexpectedly passed; do not open forward without a "
            "new registered experiment"
        )

    forward = maybe_run_forward(
        dict(summary["h1_result"]),
        unexpected_forward_open,
    )
    if forward is not None:
        raise AssertionError("fixed failed-H1 experiment returned forward evidence")
    return summary


def main() -> int:
    print(json.dumps(run(), indent=2, sort_keys=True, default=str))
    return 0


__all__ = [
    "LOCKBOX_START",
    "OUTPUT_ROOT",
    "UNION_CACHE",
    "W_OOF_SHA256",
    "W_RUN_HASH",
    "load_bounded_sources",
    "load_frozen_union",
    "load_hashed_w_oof",
    "maybe_run_forward",
    "run",
    "run_h1",
    "sha256",
]


if __name__ == "__main__":
    raise SystemExit(main())
