"""Deterministic adapters needed by the canonical Bitcoin rebuild graph."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from evaluation.economics import economics_summary, strategy_returns
from evaluation.splits import BlockingTimeSeriesSplit
from experiments.frozen_evidence import stage_frozen_evidence
from experiments.notebook02_handoff import (
    NOTEBOOK01_WIDTHS,
    PIPELINE_HANDOFF,
    WIDTHS,
    write_notebook01_handoff,
    write_pipeline_handoff,
)
from features.build import build_dataset
from models.zoo import MODELS


CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
WORKING = CODE_ROOT / "data" / "btcusdt_m15_2024_2025.parquet"
POSITIONING = CODE_ROOT / "data" / "btcusdt_positioning_m15_2024_2026.parquet"
RAW_V3 = (
    CODE_ROOT
    / "experiments"
    / "cache"
    / "tuning"
    / "all_model_sentiment_raw_180d_fixed15_v3"
    / "none"
)
POLICY_V3 = (
    CODE_ROOT
    / "experiments"
    / "cache"
    / "tuning"
    / "all_model_sentiment_policy_180d_fixed15_monthly_h1_v3"
    / "none"
)
RAW_LEGACY = RAW_V3.parents[1] / "all_model_sentiment_raw_180d_fixed15" / "none"
POLICY_LEGACY = (
    POLICY_V3.parents[1]
    / "all_model_sentiment_policy_180d_fixed15_monthly_h1"
    / "none"
)


def calculate_baseline_handoff(
    working_path: Path = WORKING,
    positioning_path: Path = POSITIONING,
    config_path: Path = CONFIG,
) -> pd.DataFrame:
    """Reproduce only the three frozen Notebook 01 positioning handoff rows."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    frame = pd.read_parquet(working_path).loc["2024-01-01":"2025-06-30"]
    positioning = pd.read_parquet(positioning_path)
    joined = frame.join(positioning.reindex(frame.index))
    split = cfg["split"]
    splitter = BlockingTimeSeriesSplit(
        n_splits=split["n_splits"],
        train_frac=split["train_frac"],
        embargo=split["embargo_bars"],
    )
    fee_bps = float(cfg["taker_fee_bps"])
    forward = frame["close"].pct_change().shift(-1)
    rows: list[dict[str, float | int]] = []
    for width in sorted(WIDTHS, reverse=True):
        base_x, _ = build_dataset(joined, threshold_bps=width, horizon=1)
        positioned_x, positioned_y = build_dataset(
            joined,
            threshold_bps=width,
            horizon=1,
            positioning=True,
        )
        common = base_x.index.intersection(positioned_x.index)
        positioned_x = positioned_x.loc[common]
        positioned_y = positioned_y.loc[common]
        predictions = pd.Series(np.nan, index=common)
        for train, test in splitter.split(positioned_x):
            model = MODELS["catboost_balanced"](None)
            model.fit(positioned_x.iloc[train], positioned_y.iloc[train])
            predictions.iloc[test] = (
                np.asarray(model.predict(positioned_x.iloc[test])).ravel().astype(int)
            )
        predictions = predictions.dropna().astype(int)
        returns = strategy_returns(
            predictions,
            forward.reindex(predictions.index),
            fee_bps,
        )
        summary = economics_summary(returns, predictions)
        rows.append(
            {
                "width_bps": int(width),
                "sortino": float(summary["sortino"]),
                "sharpe": float(summary["sharpe"]),
                "net_return": float(summary["net_return_sum"]),
                "trades": int(summary["trade_count"]),
            }
        )
    return pd.DataFrame(rows)


def write_baseline_handoff(
    metrics: pd.DataFrame,
    *,
    widths_path: Path = NOTEBOOK01_WIDTHS,
    pipeline_path: Path = PIPELINE_HANDOFF,
) -> tuple[Path, Path]:
    """Write both validated handoffs consumed by the next Bitcoin stage."""
    widths = write_notebook01_handoff(metrics, widths_path)
    pipeline = write_pipeline_handoff(upstream=metrics, path=pipeline_path)
    return widths, pipeline


def _replace_tree(source: Path, destination: Path) -> None:
    source = Path(source).resolve(strict=True)
    destination = Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("source and compatibility target must be separate trees")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    backup = destination.with_name(f".{destination.name}.{uuid4().hex}.bak")
    shutil.copytree(source, temporary)
    try:
        if destination.exists():
            os.replace(destination, backup)
        os.replace(temporary, destination)
    except Exception:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    if backup.exists():
        shutil.rmtree(backup)


def project_legacy_layout(
    raw_source: Path = RAW_V3,
    policy_source: Path = POLICY_V3,
    raw_target: Path = RAW_LEGACY,
    policy_target: Path = POLICY_LEGACY,
) -> tuple[Path, Path]:
    """Copy the v3 no-sentiment artifacts into the frozen legacy reader paths."""
    _replace_tree(raw_source, raw_target)
    _replace_tree(policy_source, policy_target)
    return Path(raw_target), Path(policy_target)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def stage_agent_evidence(
    evidence_root: Path = CODE_ROOT / ".source_evidence",
    manifest_path: Path = CODE_ROOT / "source_evidence_manifest.json",
    receipt_path: Path = CODE_ROOT / ".rebuild" / "btc_agent_calls_staged.json",
) -> Path:
    report = stage_frozen_evidence(
        evidence_root,
        CODE_ROOT,
        manifest_path,
        "agent_calls",
    )
    _atomic_json(
        receipt_path,
        {
            "status": "READY",
            "kind": report.kind,
            "files": report.files,
            "bytes": report.bytes,
            "paths": list(report.paths),
        },
    )
    return receipt_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("baseline_handoff", "project_legacy", "stage_agent"),
    )
    args = parser.parse_args(argv)
    if args.action == "baseline_handoff":
        outputs = write_baseline_handoff(calculate_baseline_handoff())
    elif args.action == "project_legacy":
        outputs = project_legacy_layout()
    else:
        outputs = (stage_agent_evidence(),)
    print(json.dumps([str(path) for path in outputs], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "calculate_baseline_handoff",
    "project_legacy_layout",
    "stage_agent_evidence",
    "write_baseline_handoff",
]
