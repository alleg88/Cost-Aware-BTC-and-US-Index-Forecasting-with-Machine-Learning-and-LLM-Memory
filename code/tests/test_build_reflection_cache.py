from pathlib import Path

import pandas as pd

from experiments.build_reflection_cache import (
    BAR_PATH,
    CONFIG_PATH,
    build_market_context,
    build_news_events,
    build_probability_panel,
)
from reflection_agent.config import load_config


def test_real_frozen_panel_has_nine_models_dz55_and_no_q2():
    panel, sources = build_probability_panel()
    assert len(sources) == 9
    assert len(panel) == 26_303
    assert panel.index.min() == pd.Timestamp("2025-07-01", tz="UTC")
    assert panel.index.max() < pd.Timestamp("2026-04-01", tz="UTC")
    assert sum(column.endswith("_p_short") for column in panel) == 9
    assert "y_true" not in panel.columns
    for source in sources:
        frame = pd.read_parquet(source)
        assert pd.to_datetime(frame["train_end"], utc=True).lt(
            pd.to_datetime(frame["test_start"], utc=True)
        ).all()
        assert frame["refit_id"].nunique() == 1


def test_real_news_cache_contains_all_balanced_source_families_and_political_posts():
    events, _ = build_news_events()
    assert set(events["source_family"]) == {"gdelt_news", "direct_policy_event", "fred_macro", "fear_greed"}
    assert (events["publisher_category"] == "political_official").sum() > 0
    assert "is_trump" not in events.columns
    assert events["summary"].str.len().max() <= 500


def test_market_context_is_causal_aligned_and_compiler_ready():
    config = load_config(CONFIG_PATH)
    panel, _ = build_probability_panel()
    events, _ = build_news_events()
    bars = pd.read_parquet(BAR_PATH)
    context = build_market_context(panel, bars, events, config)
    assert context.index.equals(panel.index)
    assert not context.isna().any().any()
    assert set(context["vol_regime"]) <= {"low", "normal", "high"}
    assert set(context["trend_regime"]) <= {"down", "flat", "up"}
    assert context.index.max() < pd.Timestamp(config.sealed_start_utc)
