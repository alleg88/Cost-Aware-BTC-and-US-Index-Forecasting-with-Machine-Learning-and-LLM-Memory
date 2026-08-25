from __future__ import annotations

import importlib.util
import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest


def test_sensitivity_json_writer_is_lf_stable(tmp_path) -> None:
    from experiments.run_q2_sentiment_sensitivity import _write_json_lf

    output = _write_json_lf(tmp_path / "audit.json", {"alpha": 1, "beta": 2})

    assert b"\r\n" not in output.read_bytes()


def test_normalise_q2_news_builds_market_and_overlapping_tech_streams() -> None:
    assert importlib.util.find_spec("experiments.run_q2_sentiment_sensitivity") is not None
    from experiments.run_q2_sentiment_sensitivity import normalise_q2_index_news

    export = pd.DataFrame(
        [
            {
                "DATE": 20260401000000,
                "url": "https://example.com/sp500",
                "domain": "example.com",
                "V2Tone": "-2.5,1,3",
                "title": "S&P 500 outlook",
                "stream": "usa500",
            },
            {
                "DATE": 20260401001500,
                "url": "https://example.com/nvidia",
                "domain": "example.com",
                "V2Tone": "4.0,2,1",
                "title": "Nvidia and semiconductor stocks rise",
                "stream": "usatech",
            },
            {
                "DATE": 20260401003000,
                "url": "https://example.com/bitcoin",
                "domain": "example.com",
                "V2Tone": "1.0,2,1",
                "title": "Bitcoin update",
                "stream": "btc",
            },
        ]
    )

    streams = normalise_q2_index_news(export)

    assert set(streams) == {"usa500", "usatech"}
    assert streams["usa500"]["url"].tolist() == [
        "https://example.com/sp500",
        "https://example.com/nvidia",
    ]
    assert streams["usatech"]["url"].tolist() == ["https://example.com/nvidia"]
    assert streams["usa500"]["tone"].tolist() == [-2.5, 4.0]
    assert str(streams["usa500"]["seendate"].dt.tz) == "UTC"
    assert streams["usa500"]["seendate"].iloc[0] == pd.Timestamp(
        "2026-04-01", tz="UTC"
    )
    assert list(streams["usa500"].columns) == [
        "seendate",
        "url",
        "domain",
        "tone",
        "title",
        "themes",
    ]


def test_overlay_replaces_only_q2_sentiment_columns() -> None:
    from experiments.run_q2_sentiment_sensitivity import overlay_q2_sentiment
    from features.index_sentiment import MATCHED_FEATURES

    index = pd.DatetimeIndex(
        ["2026-03-31T23:45:00Z", "2026-04-01T00:00:00Z"]
    )
    baseline = pd.DataFrame({"price_feature": [1.0, 2.0]}, index=index)
    for column in MATCHED_FEATURES:
        baseline[column] = 0.0
    fresh = pd.DataFrame(
        [[0.1, 2.0, -0.3, 0.4, 1.0]],
        index=index[1:],
        columns=MATCHED_FEATURES,
    )

    overlaid = overlay_q2_sentiment(
        baseline,
        fresh,
        start=pd.Timestamp("2026-04-01", tz="UTC"),
        end=pd.Timestamp("2026-07-01", tz="UTC"),
    )

    pd.testing.assert_series_equal(overlaid["price_feature"], baseline["price_feature"])
    assert overlaid.loc[index[0], list(MATCHED_FEATURES)].eq(0.0).all()
    assert overlaid.loc[index[1], list(MATCHED_FEATURES)].tolist() == fresh.iloc[0].tolist()


def test_sensitivity_manifest_loader_binds_manifest_protocol_and_q2_registry(tmp_path) -> None:
    from experiments.run_q2_sentiment_sensitivity import (
        load_frozen_manifest_for_sensitivity,
    )

    path = tmp_path / "manifest.json"
    payload = {
        "protocol": {"protocol_hash": "protocol-1"},
        "q2_sources": {"prices": {"sha256": "price-hash"}},
        "reconstructed_estimators": [],
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    identity = SimpleNamespace(
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        protocol_hash="protocol-1",
        q2_source_hashes={"prices": "price-hash"},
    )

    loaded = load_frozen_manifest_for_sensitivity(path, identity)
    assert loaded == payload

    path.write_text(json.dumps({**payload, "changed": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash"):
        load_frozen_manifest_for_sensitivity(path, identity)


def test_build_comparison_uses_trade_side_columns_and_computes_deltas() -> None:
    from experiments.run_q2_sentiment_sensitivity import build_comparison

    common = {
        "stream": "usa500",
        "candidate_id": "policy",
        "role": "primary",
        "arm": "deberta_matched",
        "daily_sharpe": 1.0,
        "daily_sortino": 2.0,
        "max_drawdown": -0.01,
    }
    original = pd.DataFrame(
        [{**common, "trades": 3, "long_trades": 3, "short_trades": 0, "net_return": 0.01}]
    )
    fresh = pd.DataFrame(
        [{**common, "trades": 4, "long_trades": 3, "short_trades": 1, "net_return": 0.012}]
    )

    comparison = build_comparison(original, fresh)

    row = comparison.iloc[0]
    assert row["fresh_long_trades"] == 3
    assert row["fresh_short_trades"] == 1
    assert row["delta_trades"] == 1
    assert row["delta_net_return"] == pytest.approx(0.002)
