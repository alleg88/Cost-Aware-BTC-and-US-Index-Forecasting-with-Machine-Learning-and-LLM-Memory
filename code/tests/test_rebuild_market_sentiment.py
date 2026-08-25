from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from experiments.rebuild_graph import load_graph
from experiments.rebuild_market_sentiment import (
    CUTOFF,
    M1_MONTHS,
    M15_MONTHS,
    POSITIONING_MONTHS,
    assert_before_cutoff,
    build_btc_m1,
    build_btc_m15,
    month_range,
)


REBUILD_TASKS = Path(__file__).parents[1] / "configs" / "rebuild_tasks.json"


def test_market_sentiment_recipes_cover_every_canonical_feature_input():
    graph = load_graph(REBUILD_TASKS)
    required = {
        "data/btcusdt_m15_2024_2025.parquet",
        "data/btcusdt_1m_2024_2026.parquet",
        "data/btcusdt_1m_2025_2026.parquet",
        "data/btcusdt_5min_2021_2026.parquet",
        "data/btcusdt_15min_2021_2026.parquet",
        "data/btcusdt_1h_2021_2026.parquet",
        "data/btcusdt_positioning_m15_2024_2026.parquet",
        "data/usa500_15min_2021_2026.parquet",
        "data/usatech_15min_2021_2026.parquet",
        "data/volidx_15min_2021_2026.parquet",
        "sentiment/raw/gdelt_btc.parquet",
        "sentiment/raw/gdelt_usa500.parquet",
        "sentiment/raw/gdelt_usatech.parquet",
        "sentiment/raw/direct_events_btc.parquet",
        "sentiment/raw/scores_btc.parquet",
        "sentiment/raw/scores_llm_btc.parquet",
    }
    assert required.issubset(graph.registered_outputs())


def test_market_sentiment_graph_contains_the_approved_recipe_order():
    graph = load_graph(REBUILD_TASKS)
    required_ids = {
        "source.verify",
        "source.stage_snapshots",
        "market.binance.download_m15",
        "market.binance.download_m1",
        "market.binance.download_positioning",
        "market.binance.build_m15",
        "market.binance.build_m1",
        "market.binance.build_positioning",
        "market.indices.build",
        "news.gdelt.normalise",
        "news.direct_events.split",
        "news.deberta.score_gdelt",
        "news.deberta.score_direct",
        "news.llm.stage_frozen",
    }
    order = graph.topological_order()

    assert required_ids.issubset(order)
    assert order.index("source.verify") < order.index("source.stage_snapshots")
    assert order.index("market.binance.download_m1") < order.index("market.binance.build_m1")
    assert order.index("news.gdelt.normalise") < order.index("news.deberta.score_gdelt")


def test_registered_download_ranges_are_exact_and_half_open():
    assert month_range("2024-01", "2026-04") == M15_MONTHS
    assert month_range("2021-01", "2026-04") == M1_MONTHS
    assert month_range("2024-01", "2026-04") == POSITIONING_MONTHS
    assert M15_MONTHS[0] == "2024-01" and M15_MONTHS[-1] == "2026-03"
    assert M1_MONTHS[0] == "2021-01" and M1_MONTHS[-1] == "2026-03"


def test_causal_boundary_guard_rejects_any_q2_timestamp():
    safe = pd.DataFrame(
        {"value": [1.0]},
        index=pd.DatetimeIndex([CUTOFF - pd.Timedelta(minutes=1)]),
    )
    assert_before_cutoff(safe, CUTOFF, "safe")
    contaminated = pd.DataFrame({"value": [1.0]}, index=pd.DatetimeIndex([CUTOFF]))

    with pytest.raises(ValueError, match="exclusive cutoff"):
        assert_before_cutoff(contaminated, CUTOFF, "contaminated")


def _write_binance_csv(raw_dir: Path, interval: str, timestamps: list[str]) -> None:
    rows = []
    for number, timestamp in enumerate(timestamps, start=1):
        open_time = int(pd.Timestamp(timestamp).timestamp() * 1000)
        price = 100.0 + number
        rows.append(
            [
                open_time,
                price,
                price + 1,
                price - 1,
                price + 0.5,
                10.0,
                open_time + 59_999,
                1_000.0,
                5,
                4.0,
                400.0,
                0,
            ]
        )
    raw_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(
        raw_dir / f"BTCUSDT-{interval}-fixture.csv",
        index=False,
        header=False,
    )


def test_btc_market_recipes_build_declared_outputs_from_fixture_csvs(tmp_path):
    raw_m15 = tmp_path / "m15"
    raw_m1 = tmp_path / "m1"
    output = tmp_path / "data"
    _write_binance_csv(
        raw_m15,
        "15m",
        ["2024-01-01T00:00:00Z", "2025-12-31T23:45:00Z", "2026-01-01T00:00:00Z", "2026-03-31T23:45:00Z"],
    )
    _write_binance_csv(
        raw_m1,
        "1m",
        ["2021-01-01T00:00:00Z", "2025-01-01T00:00:00Z", "2026-03-31T23:59:00Z"],
    )

    m15_outputs = build_btc_m15(raw_m15, output)
    m1_outputs = build_btc_m1(raw_m1, output)

    assert all(path.is_file() for path in (*m15_outputs, *m1_outputs))
    assert pd.read_parquet(output / "btcusdt_m15_2024_2025.parquet").index.max() < pd.Timestamp("2026-01-01T00:00:00Z")
    assert pd.read_parquet(output / "btcusdt_m15_lockbox_2026Q1.parquet").index.max() < CUTOFF
    assert pd.read_parquet(output / "btcusdt_1m_2025_2026.parquet").index.min() >= pd.Timestamp("2025-01-01T00:00:00Z")
    assert pd.read_parquet(output / "btcusdt_1m_2024_2026.parquet").index.min() >= pd.Timestamp("2024-01-01T00:00:00Z")
    assert pd.read_parquet(output / "btcusdt_5min_2021_2026.parquet").index.max() < CUTOFF
    assert pd.read_parquet(output / "btcusdt_1h_2021_2026.parquet").index.max() < CUTOFF
