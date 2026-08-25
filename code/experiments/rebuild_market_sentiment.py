"""Concrete source-to-market and source-to-sentiment rebuild recipes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


CUTOFF = pd.Timestamp("2026-04-01T00:00:00Z")
WORKING_END = pd.Timestamp("2026-01-01T00:00:00Z")


def month_range(start: str, end_exclusive: str) -> tuple[str, ...]:
    start_period = pd.Period(start, freq="M")
    end_period = pd.Period(end_exclusive, freq="M")
    if end_period <= start_period:
        raise ValueError("month range must be non-empty and half-open")
    return tuple(str(period) for period in pd.period_range(start_period, end_period - 1, freq="M"))


M15_MONTHS = month_range("2024-01", "2026-04")
M1_MONTHS = month_range("2021-01", "2026-04")
POSITIONING_MONTHS = month_range("2024-01", "2026-04")


def assert_before_cutoff(
    frame: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    label: str,
) -> None:
    """Reject a derived frame containing any timestamp at/after its exclusive bound."""
    if frame.empty:
        raise ValueError(f"{label} is empty")
    if isinstance(frame.index, pd.DatetimeIndex):
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    else:
        columns = [column for column in ("available_at", "seendate", "timestamp") if column in frame]
        if not columns:
            raise ValueError(f"{label} has no timestamp index or column")
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame[columns[0]], utc=True, errors="raise"))
    bound = pd.Timestamp(cutoff)
    bound = bound.tz_localize("UTC") if bound.tzinfo is None else bound.tz_convert("UTC")
    if timestamps.max() >= bound:
        raise ValueError(f"{label} reaches or crosses exclusive cutoff {bound.isoformat()}")


def _write_receipt(path: Path, payload: dict[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def verify_sources(evidence_root: Path, manifest: Path, receipt: Path) -> Path:
    from experiments.source_evidence import verify_manifest

    report = verify_manifest(evidence_root, manifest)
    if report["status"] != "READY":
        raise RuntimeError(f"source-only bundle verification failed: {report}")
    return _write_receipt(receipt, report)


def stage_snapshots(
    evidence_root: Path,
    code_root: Path,
    manifest: Path,
    receipt: Path,
) -> Path:
    from experiments.frozen_evidence import stage_frozen_evidence
    from sentiment.direct_events import validate_canonical_direct_coverage

    report = stage_frozen_evidence(
        evidence_root,
        code_root,
        manifest,
        kind="snapshot_raw",
    )
    coverage = validate_canonical_direct_coverage(
        code_root / "sentiment" / "raw" / "direct_events.parquet",
        code_root / "sentiment" / "raw" / "trump_truth_posts.csv",
    )
    return _write_receipt(
        receipt,
        {
            "status": "READY",
            "files": report.files,
            "bytes": report.bytes,
            "coverage": coverage,
        },
    )


def _require_downloads(statuses: Iterable[str], label: str) -> None:
    failed = [status for status in statuses if not status.startswith(("ok", "skip"))]
    if failed:
        raise RuntimeError(f"{label} acquisition incomplete: {failed[:5]}")


def download_binance_dataset(dataset: str, output_dir: Path, receipt: Path) -> Path:
    """Download one disjoint public Binance source family with exchange checksums."""
    from data import download_binance as source

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses: list[str] = []
    if dataset in {"m15", "m1"}:
        interval = "15m" if dataset == "m15" else "1m"
        months = M15_MONTHS if dataset == "m15" else M1_MONTHS
        source.RAW_DIR = output_dir
        statuses = [source.fetch_month(month, interval) for month in months]
    elif dataset == "positioning":
        source.FUT_DIR = output_dir
        for month in POSITIONING_MONTHS:
            statuses.extend(source.fetch_metrics_day(day) for day in source._month_days(month))
            statuses.append(source.fetch_funding_month(month))
    else:
        raise ValueError(f"unknown Binance rebuild dataset: {dataset}")
    _require_downloads(statuses, dataset)
    return _write_receipt(
        receipt,
        {
            "status": "READY",
            "dataset": dataset,
            "months": list(M15_MONTHS if dataset == "m15" else M1_MONTHS if dataset == "m1" else POSITIONING_MONTHS),
            "objects": len(statuses),
        },
    )


def build_btc_m15(raw_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    from data.load import load_bars

    frame = load_bars(
        raw_dir,
        base_interval="15m",
        target_interval="15min",
        start="2024-01-01",
        end="2026-03-31",
    )
    assert_before_cutoff(frame, CUTOFF, "BTC M15")
    working = frame.loc[frame.index < WORKING_END]
    lockbox = frame.loc[(frame.index >= WORKING_END) & (frame.index < CUTOFF)]
    if working.empty or lockbox.empty:
        raise ValueError("BTC M15 working or Q1 lockbox slice is empty")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    working_path = output_dir / "btcusdt_m15_2024_2025.parquet"
    lockbox_path = output_dir / "btcusdt_m15_lockbox_2026Q1.parquet"
    working.to_parquet(working_path)
    lockbox.to_parquet(lockbox_path)
    return working_path, lockbox_path


def build_btc_m1(
    raw_dir: Path, output_dir: Path
) -> tuple[Path, Path, Path, Path, Path, Path]:
    from data.build_grids import build_grid
    from data.load import load_bars

    frame = load_bars(
        raw_dir,
        base_interval="1m",
        target_interval="1min",
        start="2021-01-01",
        end="2026-03-31",
    )
    assert_before_cutoff(frame, CUTOFF, "BTC M1")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "btcusdt_1m_2021_2026.parquet"
    study_path = output_dir / "btcusdt_1m_2024_2026.parquet"
    recent_path = output_dir / "btcusdt_1m_2025_2026.parquet"
    grid_paths = {
        interval: output_dir / f"btcusdt_{interval}_2021_2026.parquet"
        for interval in ("5min", "15min", "1h")
    }
    frame.to_parquet(full_path)
    frame.loc[frame.index >= pd.Timestamp("2024-01-01T00:00:00Z")].to_parquet(study_path)
    frame.loc[frame.index >= pd.Timestamp("2025-01-01T00:00:00Z")].to_parquet(recent_path)
    for interval, path in grid_paths.items():
        grid = build_grid(frame, interval)
        assert_before_cutoff(grid, CUTOFF, f"BTC extended {interval}")
        grid.to_parquet(path)
    return (
        full_path,
        study_path,
        recent_path,
        grid_paths["5min"],
        grid_paths["15min"],
        grid_paths["1h"],
    )


def build_btc_positioning(raw_dir: Path, data_dir: Path) -> tuple[Path, Path]:
    from data import build_positioning as positioning

    raw_dir = Path(raw_dir)
    data_dir = Path(data_dir)
    positioning.MET_DIR = raw_dir / "metrics"
    positioning.FUND_DIR = raw_dir / "fundingRate"
    working = pd.read_parquet(data_dir / "btcusdt_m15_2024_2025.parquet")
    lockbox = pd.read_parquet(data_dir / "btcusdt_m15_lockbox_2026Q1.parquet")
    extended = pd.read_parquet(data_dir / "btcusdt_15min_2021_2026.parquet")
    index = pd.DatetimeIndex(working.index).union(pd.DatetimeIndex(lockbox.index)).sort_values()
    current = positioning.build(index, end_exclusive=CUTOFF)
    extended_frame = positioning.build(pd.DatetimeIndex(extended.index), end_exclusive=CUTOFF)
    assert_before_cutoff(current, CUTOFF, "BTC positioning")
    assert_before_cutoff(extended_frame, CUTOFF, "BTC extended positioning")
    current_path = data_dir / "btcusdt_positioning_m15_2024_2026.parquet"
    extended_path = data_dir / "btcusdt_positioning_15min_2021_2026.parquet"
    current.to_parquet(current_path)
    extended_frame.to_parquet(extended_path)
    return current_path, extended_path


def build_index_market(raw_dir: Path, output_dir: Path) -> tuple[Path, ...]:
    from data.index_market import INSTRUMENTS, write_index_grids

    outputs: list[Path] = []
    for instrument in INSTRUMENTS:
        current = write_index_grids(
            raw_dir,
            output_dir,
            instrument,
            end_exclusive=CUTOFF,
        )
        for name, path in current.items():
            if name != "audit":
                assert_before_cutoff(pd.read_parquet(path), CUTOFF, f"{instrument} {name}")
            outputs.append(path)
    return tuple(outputs)


def normalise_gdelt(input_dir: Path, output_dir: Path) -> tuple[Path, ...]:
    from sentiment import gdelt_bigquery as gdelt

    gdelt.IN_DIR = Path(input_dir)
    gdelt.OUT_DIR = Path(output_dir)
    gdelt.OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = gdelt.load_all()
    if raw is None:
        raise FileNotFoundError(f"no GDELT snapshots in {input_dir}")
    btc = raw.loc[raw["stream"].eq("btc")]
    market = raw.loc[~raw["stream"].eq("btc")]
    tech = market.loc[gdelt._is_tech(market)]
    for frame, stream in ((btc, "btc"), (market, "usa500"), (tech, "usatech")):
        gdelt._write(frame, stream)
    paths = tuple(gdelt.OUT_DIR / f"gdelt_{stream}.parquet" for stream in ("btc", "usa500", "usatech"))
    for path in paths:
        assert_before_cutoff(pd.read_parquet(path), CUTOFF, path.name)
    return paths


def split_direct_events(master: Path, truth_posts: Path, output_dir: Path) -> tuple[Path, ...]:
    from sentiment.direct_events import (
        rebuild_direct_streams,
        validate_canonical_direct_coverage,
    )

    validate_canonical_direct_coverage(master, truth_posts)
    outputs = rebuild_direct_streams(master, output_dir)
    for path in outputs.values():
        assert_before_cutoff(pd.read_parquet(path), CUTOFF, path.name)
    return tuple(outputs.values())


def score_deberta(raw_dir: Path, source_prefix: str) -> tuple[Path, ...]:
    from sentiment import score

    score.RAW_DIR = Path(raw_dir)
    paths = tuple(
        score.score_stream(stream, source_prefix=source_prefix)
        for stream in ("btc", "usa500", "usatech")
    )
    for path in paths:
        assert_before_cutoff(pd.read_parquet(path), CUTOFF, path.name)
    return paths


def stage_llm_scores(evidence_root: Path, code_root: Path, manifest: Path) -> tuple[Path, ...]:
    from experiments.frozen_evidence import stage_frozen_evidence

    report = stage_frozen_evidence(evidence_root, code_root, manifest, kind="llm_scores")
    return tuple(Path(code_root) / path for path in report.paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "verify_sources",
            "stage_snapshots",
            "download_m15",
            "download_m1",
            "download_positioning",
            "build_m15",
            "build_m1",
            "build_positioning",
            "build_indices",
            "normalise_gdelt",
            "split_direct",
            "score_gdelt",
            "score_direct",
            "stage_llm",
        ),
    )
    parser.add_argument("--evidence-root", type=Path, default=Path(".source_evidence"))
    parser.add_argument("--manifest", type=Path, default=Path("source_evidence_manifest.json"))
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    code_root = Path.cwd()

    if args.action == "verify_sources":
        verify_sources(args.evidence_root, args.manifest, args.receipt or Path(".rebuild/source_verified.json"))
    elif args.action == "stage_snapshots":
        stage_snapshots(
            args.evidence_root,
            code_root,
            args.manifest,
            args.receipt or Path(".rebuild/snapshots_staged.json"),
        )
    elif args.action.startswith("download_"):
        dataset = args.action.removeprefix("download_")
        if args.output_dir is None:
            raise ValueError("download actions require --output-dir")
        download_binance_dataset(
            dataset,
            args.output_dir,
            args.receipt or Path(f".rebuild/{args.action}.json"),
        )
    elif args.action == "build_m15":
        build_btc_m15(args.raw_dir or Path(".rebuild/raw/binance/m15"), args.output_dir or Path("data"))
    elif args.action == "build_m1":
        build_btc_m1(args.raw_dir or Path(".rebuild/raw/binance/m1"), args.output_dir or Path("data"))
    elif args.action == "build_positioning":
        build_btc_positioning(
            args.raw_dir or Path(".rebuild/raw/binance/positioning"),
            args.output_dir or Path("data"),
        )
    elif args.action == "build_indices":
        build_index_market(args.raw_dir or Path("data/raw"), args.output_dir or Path("data"))
    elif args.action == "normalise_gdelt":
        normalise_gdelt(
            args.raw_dir or Path("sentiment/raw/gdelt_bq"),
            args.output_dir or Path("sentiment/raw"),
        )
    elif args.action == "split_direct":
        raw = args.raw_dir or Path("sentiment/raw")
        split_direct_events(
            raw / "direct_events.parquet",
            raw / "trump_truth_posts.csv",
            args.output_dir or raw,
        )
    elif args.action in {"score_gdelt", "score_direct"}:
        score_deberta(
            args.raw_dir or Path("sentiment/raw"),
            "gdelt" if args.action == "score_gdelt" else "direct_events",
        )
    else:
        stage_llm_scores(args.evidence_root, code_root, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
