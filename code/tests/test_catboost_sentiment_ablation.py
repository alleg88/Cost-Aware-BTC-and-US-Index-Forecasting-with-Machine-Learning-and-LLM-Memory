from pathlib import Path

import pandas as pd


def test_each_arm_gets_exactly_its_declared_schema(monkeypatch):
    """Every arm joins the base features to the block ARM_SPECS declares for it."""
    import experiments.catboost_sentiment_ablation as study
    from experiments.run_catboost_matched_ablation import PreparedData

    index = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")
    base_columns = [f"base_{i}" for i in range(24)]
    X = pd.DataFrame(1.0, index=index, columns=base_columns)
    y = pd.Series([0, 1, 2, 1], index=index)
    prepared = PreparedData(
        bars=pd.DataFrame({"close": [1, 2, 3, 4]}, index=index),
        minute=pd.DataFrame({"close": [1, 2, 3, 4]}, index=index),
        features={width: (X.copy(), y.copy()) for width in study.WIDTHS},
        regimes=pd.Series("sideways", index=index),
        m15_fingerprint="m15",
        minute_fingerprint="m1",
        sentiment_mode="none",
        feature_columns=tuple(base_columns),
    )

    def frame(columns):
        return pd.DataFrame({column: 0.1 for column in columns}, index=index)

    monkeypatch.setattr(study, "load_prepared_data", lambda **kwargs: prepared)
    monkeypatch.setattr(
        study, "build_matched_sentiment_features",
        lambda stream, bar_index, scorer, weighting: frame(
            study.MATCHED_SENTIMENT_FEATURES_BTC).reindex(bar_index))
    monkeypatch.setattr(
        study, "build_direct_event_block",
        lambda stream, bar_index, scorer, weighting="own": frame(
            study.DIRECT_EVENT_FEATURES).reindex(bar_index))
    monkeypatch.setattr(
        study, "build_llm_full_features",
        lambda stream, bar_index: frame(study.LLM_FULL_FEATURES_BTC).reindex(bar_index))

    for arm, spec in study.ARM_SPECS.items():
        got = study.prepare_arm_data(arm, expected_base_columns=base_columns)
        assert got.feature_columns == tuple([*base_columns, *spec["schema"]])
        assert got.sentiment_mode == f"matched_{arm}"

    # every arm sees the same sources; llm_full folds the pulse in inside its own
    # builder, the other two through prepare_arm_data
    assert study.ARM_SPECS["classic"]["direct"] is True
    assert study.ARM_SPECS["llm"]["direct"] is True
    assert study.ARM_SPECS["llm_full"]["direct"] is False

    per_width = study.prepare_arm_data("classic", expected_base_columns=base_columns)
    expected = tuple([*base_columns, *study.ARM_SPECS["classic"]["schema"]])
    for width in study.WIDTHS:
        assert tuple(per_width.features[width][0].columns) == expected


def test_sentiment_study_freezes_notebook02_lookbacks_and_candidate_zero():
    import inspect
    import experiments.catboost_sentiment_ablation as study

    assert study.LOOKBACK_DAYS == {55: 180, 65: 180, 75: 180}
    source = inspect.getsource(study.run_arm)
    assert "candidate_ids=(0,)" in source
    assert "load_candidates()[0]" in source
    assert "runner.lookback_days = LOOKBACK_DAYS.copy()" in source


def test_sentiment_output_roots_are_isolated_by_scorer(tmp_path):
    import experiments.catboost_sentiment_ablation as study

    assert study.arm_output_root("classic", tmp_path) == tmp_path / "classic"
    assert study.arm_output_root("llm", tmp_path) == tmp_path / "llm"
