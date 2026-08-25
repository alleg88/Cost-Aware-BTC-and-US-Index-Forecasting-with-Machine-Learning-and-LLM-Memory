from __future__ import annotations

from pathlib import Path

import pandas as pd

import data.build_positioning as positioning


CUTOFF = pd.Timestamp("2026-04-01T00:00:00Z")


def test_positioning_cutoff_excludes_q2_files_before_csv_read(
    tmp_path: Path, monkeypatch
) -> None:
    metrics_dir = tmp_path / "metrics"
    funding_dir = tmp_path / "funding"
    metrics_dir.mkdir()
    funding_dir.mkdir()
    columns = {
        "create_time": ["2026-03-31T23:55:00Z"],
        "sum_open_interest": [100.0],
        "sum_toptrader_long_short_ratio": [1.1],
        "sum_taker_long_short_vol_ratio": [0.9],
    }
    pd.DataFrame(columns).to_csv(
        metrics_dir / "BTCUSDT-metrics-2026-03-31.csv", index=False
    )
    pd.DataFrame({**columns, "create_time": ["2026-04-01T00:00:00Z"]}).to_csv(
        metrics_dir / "BTCUSDT-metrics-2026-04-01.csv", index=False
    )
    pd.DataFrame(
        {"calc_time": [int(pd.Timestamp("2026-03-31T16:00:00Z").timestamp() * 1000)], "last_funding_rate": [0.0001]}
    ).to_csv(funding_dir / "BTCUSDT-fundingRate-2026-03.csv", index=False)
    pd.DataFrame(
        {"calc_time": [int(pd.Timestamp("2026-04-01T00:00:00Z").timestamp() * 1000)], "last_funding_rate": [0.0002]}
    ).to_csv(funding_dir / "BTCUSDT-fundingRate-2026-04.csv", index=False)
    monkeypatch.setattr(positioning, "MET_DIR", metrics_dir)
    monkeypatch.setattr(positioning, "FUND_DIR", funding_dir)

    metrics = positioning.load_metrics(end_exclusive=CUTOFF)
    funding = positioning.load_funding(end_exclusive=CUTOFF)

    assert metrics["create_time"].max() < CUTOFF
    assert funding["time"].max() < CUTOFF
    assert metrics["sum_open_interest"].tolist() == [100.0]
    assert funding["funding_rate"].tolist() == [0.0001]
