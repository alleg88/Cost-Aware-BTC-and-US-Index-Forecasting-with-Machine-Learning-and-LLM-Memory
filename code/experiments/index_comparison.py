"""Matched, artifact-only comparison of the two index replications."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from experiments.index_replication import ARMS
from experiments.index_replication_protocol import (
    CUTOFF,
    FORWARD_END,
    FORWARD_START,
    MODEL_NAMES,
    holm_adjust,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = CODE_ROOT / "experiments" / "cache" / "index_replication"
DEFAULT_OUTPUT = CODE_ROOT / "experiments" / "cache" / "index_comparison"
SENTIMENT_ARMS = tuple(arm for arm in ARMS if arm != "selected_base")
PROTOCOL_VERSION = "index-comparison-v3-estimability-calendar-hac"


def _read_ledger(root: Path, arm: str, model: str) -> pd.DataFrame:
    path = root / "forward_ledgers" / arm / f"{model}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    required = {"entry_bar_open", "side", "net_return"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} misses columns: {sorted(missing)}")
    frame = frame.loc[:, list(required)].copy()
    frame["entry_bar_open"] = pd.to_datetime(frame["entry_bar_open"], utc=True)
    if frame.duplicated(["entry_bar_open", "side"]).any():
        raise ValueError(f"{path} has duplicate completed-trade keys")
    if len(frame) and frame["entry_bar_open"].ge(CUTOFF).any():
        raise AssertionError(f"{path} crossed Q2-2026")
    return frame


def _hac_mean(values: pd.Series, *, max_lag: int = 7) -> dict[str, float]:
    """Conservative Newey-West inference for one UTC-daily aggregate series."""
    array = values.astype(float).to_numpy()
    count = len(array)
    mean = float(np.mean(array)) if count else 0.0
    if count < 2 or np.allclose(array, array[0] if count else 0.0):
        return {
            "hac_mean_daily_delta": mean,
            "hac_standard_error": 0.0,
            "hac_ci_low": mean,
            "hac_ci_high": mean,
            "raw_p_value": 1.0,
        }
    centered = array - mean
    lag = min(int(max_lag), count - 1)
    long_run_variance = float(np.dot(centered, centered) / count)
    for offset in range(1, lag + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / count)
        long_run_variance += 2.0 * (1.0 - offset / (lag + 1.0)) * covariance
    standard_error = float(np.sqrt(max(long_run_variance, 0.0) / count))
    if standard_error <= np.finfo(float).eps:
        p_value = 1.0
    else:
        p_value = float(2.0 * norm.sf(abs(mean / standard_error)))
    return {
        "hac_mean_daily_delta": mean,
        "hac_standard_error": standard_error,
        "hac_ci_low": mean - 1.96 * standard_error,
        "hac_ci_high": mean + 1.96 * standard_error,
        "raw_p_value": p_value,
    }


def _calendar_returns(series: pd.Series) -> pd.Series:
    calendar = pd.date_range(
        FORWARD_START.normalize(),
        (FORWARD_END - pd.Timedelta(days=1)).normalize(),
        freq="D",
    )
    current = series.loc[(series.index >= FORWARD_START) & (series.index < FORWARD_END)]
    daily = current.groupby(current.index.normalize()).sum()
    return daily.reindex(calendar, fill_value=0.0).astype(float)


def _eligible_models(root: Path, arm: str, models: Sequence[str]) -> tuple[str, ...]:
    """Use all requested models in fixtures, but eligible intersections in sealed runs."""
    path = root / "forward_summary.parquet"
    if not path.exists():
        return tuple(models)
    summary = pd.read_parquet(path)
    required = {"arm", "model_name", "h1_eligible"}
    if not required.issubset(summary.columns):
        raise ValueError(f"{path} misses eligibility columns")
    base = set(
        summary.loc[
            summary["arm"].eq("selected_base") & summary["h1_eligible"].astype(bool),
            "model_name",
        ]
    )
    candidate = set(
        summary.loc[
            summary["arm"].eq(arm) & summary["h1_eligible"].astype(bool),
            "model_name",
        ]
    )
    return tuple(model for model in models if model in base.intersection(candidate))


def calendar_arm_hypotheses(
    root: str | Path,
    *,
    arms: Sequence[str] = SENTIMENT_ARMS,
    models: Sequence[str] = MODEL_NAMES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare independently selected policies on complete UTC-day PnL calendars."""
    root = Path(root)
    rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for arm in arms:
        eligible_models = _eligible_models(root, arm, models)
        daily_deltas: dict[str, pd.Series] = {}
        for model in eligible_models:
            base = _read_ledger(root, "selected_base", model)
            candidate = _read_ledger(root, arm, model)
            base_daily = _calendar_returns(_read_per_bar(root, "selected_base", model))
            arm_daily = _calendar_returns(_read_per_bar(root, arm, model))
            delta = arm_daily - base_daily
            daily_deltas[model] = delta
            inference = _hac_mean(delta)
            model_rows.append(
                {
                    "arm": arm,
                    "model_name": model,
                    "calendar_days": int(len(delta)),
                    "base_net_return": float(base_daily.sum()),
                    "arm_net_return": float(arm_daily.sum()),
                    "net_delta": float(delta.sum()),
                    "base_trades": int(len(base)),
                    "arm_trades": int(len(candidate)),
                    "trade_ratio": (
                        float(len(candidate) / len(base)) if len(base) else np.nan
                    ),
                    **inference,
                }
            )
        scoped = [row for row in model_rows if row["arm"] == arm]
        if daily_deltas:
            daily_frame = pd.DataFrame(daily_deltas)
            aggregate_daily = daily_frame.median(axis=1)
            inference = _hac_mean(aggregate_daily)
            net_deltas = pd.Series([row["net_delta"] for row in scoped], dtype=float)
            base_nets = pd.Series([row["base_net_return"] for row in scoped], dtype=float)
            arm_nets = pd.Series([row["arm_net_return"] for row in scoped], dtype=float)
            ratios = pd.Series([row["trade_ratio"] for row in scoped], dtype=float)
        else:
            aggregate_daily = _calendar_returns(pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC")))
            inference = {
                "hac_mean_daily_delta": np.nan,
                "hac_standard_error": np.nan,
                "hac_ci_low": np.nan,
                "hac_ci_high": np.nan,
                "raw_p_value": np.nan,
            }
            net_deltas = base_nets = arm_nets = ratios = pd.Series(dtype=float)
        estimable = bool(daily_deltas)
        rows.append(
            {
                "arm": arm,
                "eligible_model_count": int(len(eligible_models)),
                "estimable": estimable,
                "estimability_reason": (
                    "estimated"
                    if estimable
                    else "no_jointly_h1_eligible_same_model_pair"
                ),
                "calendar_days": int(len(aggregate_daily)),
                "base_model_median_net_return": float(base_nets.median()) if len(base_nets) else np.nan,
                "arm_model_median_net_return": float(arm_nets.median()) if len(arm_nets) else np.nan,
                "model_median_net_delta": float(net_deltas.median()) if len(net_deltas) else np.nan,
                "model_net_delta_q25": float(net_deltas.quantile(0.25)) if len(net_deltas) else np.nan,
                "model_net_delta_q75": float(net_deltas.quantile(0.75)) if len(net_deltas) else np.nan,
                "positive_models": int(net_deltas.gt(0.0).sum()),
                "median_model_trade_ratio": float(ratios.median()) if len(ratios) else np.nan,
                "calendar_missing_fill": "zero",
                "aggregation": "daily_median_across_eligible_models",
                "policy_parameters_shared": False,
                **inference,
            }
        )
    result = pd.DataFrame(rows)
    if len(result):
        finite = result["raw_p_value"].notna()
        result["holm_p_value"] = np.nan
        significant = pd.Series(pd.NA, index=result.index, dtype="boolean")
        if finite.any():
            adjusted = holm_adjust(result.loc[finite, "raw_p_value"])
            result.loc[finite, "holm_p_value"] = adjusted
            significant.loc[finite] = adjusted.lt(0.05).to_numpy()
        result["holm_significant_5pct"] = significant
    return result, pd.DataFrame(model_rows)


def _read_per_bar(root: Path, arm: str, model: str) -> pd.Series:
    path = root / "forward_ledgers" / arm / f"{model}_per_bar.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    if not {"timestamp", "net_return"}.issubset(frame.columns):
        raise ValueError(f"{path} misses timestamp/net_return")
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    if timestamp.duplicated().any() or (len(timestamp) and timestamp.ge(CUTOFF).any()):
        raise AssertionError(f"{path} has duplicate/Q2 timestamps")
    return pd.Series(frame["net_return"].astype(float).to_numpy(), index=timestamp)


def common_timestamp_summary(
    roots: Mapping[str, str | Path],
    *,
    arms: Sequence[str] = ARMS,
    models: Sequence[str] = MODEL_NAMES,
) -> pd.DataFrame:
    """Report full and identical-USA500/USATECH-timestamp net returns."""
    if set(roots) != {"usa500", "usatech"}:
        raise ValueError("common comparison requires separate USA500 and USATECH roots")
    rows: list[dict[str, Any]] = []
    for arm in arms:
        for model in models:
            series = {
                stream: _read_per_bar(Path(root), arm, model)
                for stream, root in roots.items()
            }
            common = series["usa500"].index.intersection(series["usatech"].index)
            for stream, values in series.items():
                rows.append(
                    {
                        "stream": stream,
                        "arm": arm,
                        "model_name": model,
                        "full_timestamp_rows": int(len(values)),
                        "common_timestamp_rows": int(len(common)),
                        "full_net_return": float(values.sum()),
                        "common_net_return": float(values.reindex(common).sum()),
                    }
                )
    return pd.DataFrame(rows)


def transferability_table(
    calendar_tests: pd.DataFrame, *, trade_retention_floor: float = 0.80
) -> pd.DataFrame:
    """Descriptively require non-negative model medians and retained coverage."""
    required = {
        "stream",
        "arm",
        "estimable",
        "model_median_net_delta",
        "median_model_trade_ratio",
        "holm_p_value",
    }
    missing = required.difference(calendar_tests.columns)
    if missing:
        raise ValueError(f"calendar tests miss columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for arm, frame in calendar_tests.groupby("arm", sort=True):
        indexed = frame.set_index("stream")
        complete = set(indexed.index) == {"usa500", "usatech"}
        estimable = bool(
            complete and indexed["estimable"].fillna(False).astype(bool).all()
        )
        nonnegative = bool(
            estimable and indexed["model_median_net_delta"].ge(0.0).all()
        )
        retained = bool(
            estimable
            and indexed["median_model_trade_ratio"].ge(float(trade_retention_floor)).all()
        )
        reason = (
            "estimated"
            if estimable
            else (
                "both_markets_not_present"
                if not complete
                else "no_jointly_h1_eligible_same_model_pair"
            )
        )
        rows.append(
            {
                "arm": arm,
                "both_markets_present": complete,
                "estimable_both": estimable,
                "assessment_status": "estimated" if estimable else "not_estimable",
                "estimability_reason": reason,
                "nonnegative_both": nonnegative if estimable else pd.NA,
                "trade_retention_both": retained if estimable else pd.NA,
                "descriptive_transfer_both": (
                    bool(nonnegative and retained) if estimable else pd.NA
                ),
                "holm_significant_both": (
                    bool(indexed["holm_p_value"].lt(0.05).all())
                    if estimable
                    else pd.NA
                ),
                "trade_retention_floor": float(trade_retention_floor),
                "claim_scope": (
                    "descriptive_not_noninferiority"
                    if estimable
                    else "not_estimable_no_joint_pairs"
                ),
            }
        )
    result = pd.DataFrame(rows)
    for column in (
        "estimable_both",
        "nonnegative_both",
        "trade_retention_both",
        "descriptive_transfer_both",
        "holm_significant_both",
    ):
        if column in result:
            result[column] = result[column].astype("boolean")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def build_comparison_artifacts(
    *,
    root: str | Path = DEFAULT_ROOT,
    output: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    root, output = Path(root), Path(output)
    roots = {stream: root / stream for stream in ("usa500", "usatech")}
    calendar_frames = []
    model_frames = []
    for stream, stream_root in roots.items():
        calendar, model = calendar_arm_hypotheses(stream_root)
        calendar.insert(0, "stream", stream)
        model.insert(0, "stream", stream)
        calendar_frames.append(calendar)
        model_frames.append(model)
    calendar = pd.concat(calendar_frames, ignore_index=True)
    model = pd.concat(model_frames, ignore_index=True)
    common = common_timestamp_summary(roots)
    transferability = transferability_table(calendar)
    artifacts = {
        "calendar_arm_hypotheses.parquet": calendar,
        "model_calendar_hypotheses.parquet": model,
        "common_timestamp_economics.parquet": common,
        "transferability.parquet": transferability,
    }
    for name, frame in artifacts.items():
        _atomic_parquet(frame, output / name)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "source_roots": {key: str(value) for key, value in roots.items()},
        "artifacts": {name: _sha256(output / name) for name in artifacts},
        "inference_unit": "UTC calendar day after median across eligible models",
        "calendar_missing_fill": "zero",
        "policy_parameters_shared": False,
        "p_value_method": "Newey-West HAC lag 7",
        "p_value_adjustment": "Holm within three declared sentiment arms per instrument",
        "common_timestamp_only": True,
        "q2_2026_loaded": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(build_comparison_artifacts(root=args.root, output=args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
