import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _economic_row(**overrides):
    row = {
        "candidate_id": 0,
        "policy_id": 0,
        "trades": 50,
        "n_long": 15,
        "n_short": 15,
        "positive_segments": 4,
        "robust_score": 0.4,
        "pooled_sortino": 0.8,
        "pooled_net": 0.03,
    }
    row.update(overrides)
    return row


def _default_candidate_payload():
    from experiments.catboost_matched_ablation import DEFAULT_CANDIDATE_PATH

    return json.loads(DEFAULT_CANDIDATE_PATH.read_text(encoding="utf-8"))


def _write_candidate_payload(tmp_path, payload):
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path

def test_protocol_spans_are_utc_half_open_and_keep_lockbox_sealed():
    from experiments.catboost_matched_ablation import (
        CALIBRATION_END,
        FORWARD_END,
        LOCKBOX_START,
        SELECTION_END,
        SELECTION_START,
        span_mask,
    )

    assert SELECTION_START == pd.Timestamp("2024-01-01", tz="UTC")
    assert SELECTION_END == pd.Timestamp("2025-01-01", tz="UTC")
    assert CALIBRATION_END == pd.Timestamp("2025-07-01", tz="UTC")
    assert FORWARD_END == pd.Timestamp("2026-04-01", tz="UTC")
    assert LOCKBOX_START == FORWARD_END

    index = pd.DatetimeIndex(
        [
            "2023-12-31 23:45Z",
            "2024-01-01 00:00Z",
            "2024-12-31 23:45Z",
            "2025-01-01 00:00Z",
            "2025-06-30 23:45Z",
            "2025-07-01 00:00Z",
            "2026-03-31 23:45Z",
            "2026-04-01 00:00Z",
        ]
    )
    assert span_mask(index, SELECTION_START, SELECTION_END).tolist() == [
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert span_mask(index, SELECTION_END, CALIBRATION_END).tolist() == [
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
    ]
    assert span_mask(index, CALIBRATION_END, FORWARD_END).tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
    ]


def test_fixed_candidate_and_policy_grid_contract():
    from experiments.catboost_matched_ablation import (
        CANDIDATE_COUNT,
        GEOMETRIES,
        TAUS,
        WIDTHS,
        policy_choices,
    )

    assert WIDTHS == (55, 65, 75)
    assert CANDIDATE_COUNT == 15
    assert TAUS == (0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
    assert GEOMETRIES == ((150, 75, 1), (150, 100, 1), (200, 100, 1))
    assert len(policy_choices()) == 33
    assert len(set(policy_choices())) == 33


def test_selected_candidates_include_fixed_baseline_f1_and_economic():
    from experiments.run_catboost_matched_ablation import MatchedAblationRunner

    runner = MatchedAblationRunner(
        widths=(55,),
        candidates=({"depth": 6}, {"depth": 7}),
        candidate_ids=(0, 1),
        smoke=True,
        stage1_only=True,
    )
    classification = pd.DataFrame(
        [
            {"width_bps": 55, "candidate_id": 0, "overall_f1": 0.40, "robust_f1": 0.35},
            {"width_bps": 55, "candidate_id": 1, "overall_f1": 0.45, "robust_f1": 0.42},
        ]
    )
    winners = pd.DataFrame(
        [
            {**_economic_row(candidate_id=0, policy_id=2), "width_bps": 55},
            {**_economic_row(candidate_id=1, policy_id=3, pooled_sortino=1.0), "width_bps": 55},
        ]
    )

    selected = runner._selected_candidates(classification, winners)

    assert selected["objective"].tolist() == ["baseline", "F1", "economic"]
    assert selected.set_index("objective")["candidate_id"].to_dict() == {
        "baseline": 0,
        "F1": 1,
        "economic": 1,
    }
    assert selected.loc[selected["objective"] == "baseline", "selection_rule"].iloc[0] == "fixed candidate 0"


def test_protocol_is_self_contained_and_default_pool_is_tracked():
    import experiments.catboost_matched_ablation as protocol
    from experiments.catboost_matched_ablation import (
        CANDIDATE_POOL_FINGERPRINT,
        DEFAULT_CANDIDATE_PATH,
        load_candidates,
    )

    source = Path(protocol.__file__).read_text(encoding="utf-8")
    assert "catboost_economic_optuna" not in source
    assert DEFAULT_CANDIDATE_PATH.name == "catboost_candidates_15.json"
    assert CANDIDATE_POOL_FINGERPRINT == (
        "17a1b1999630d824f8bd257aacf8541f67058c31d05d572d36f60c85516288d7"
    )
    candidates = load_candidates()
    assert len(candidates) == 15
    assert candidates[0] == {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.1,
        "l2_leaf_reg": 3.0,
    }
    assert candidates[-1]["iterations"] == 700
    assert candidates[-1]["depth"] == 7


def test_candidate_validation_rejects_bad_provenance(tmp_path):
    from experiments.catboost_matched_ablation import load_candidates

    payload = _default_candidate_payload()
    payload["generated_before_economics"] = False
    with pytest.raises(ValueError, match="outcome-independent"):
        load_candidates(_write_candidate_payload(tmp_path, payload))


def test_candidate_validation_rejects_changed_widths(tmp_path):
    from experiments.catboost_matched_ablation import load_candidates

    payload = _default_candidate_payload()
    payload["widths"] = [55, 65, 70]
    with pytest.raises(ValueError, match="widths"):
        load_candidates(_write_candidate_payload(tmp_path, payload))


def test_candidate_validation_rejects_changed_count(tmp_path):
    from experiments.catboost_matched_ablation import load_candidates

    payload = _default_candidate_payload()
    payload["candidates"].pop()
    with pytest.raises(ValueError, match="exactly 15"):
        load_candidates(_write_candidate_payload(tmp_path, payload))


def test_candidate_validation_rejects_empty_parameters(tmp_path):
    from experiments.catboost_matched_ablation import load_candidates

    payload = _default_candidate_payload()
    payload["candidates"][4] = {}
    with pytest.raises(ValueError, match="non-empty"):
        load_candidates(_write_candidate_payload(tmp_path, payload))


def test_candidate_validation_rejects_duplicate_parameters(tmp_path):
    from experiments.catboost_matched_ablation import load_candidates

    payload = _default_candidate_payload()
    payload["candidates"][4] = copy.deepcopy(payload["candidates"][3])
    with pytest.raises(ValueError, match="unique"):
        load_candidates(_write_candidate_payload(tmp_path, payload))


def test_candidate_validation_rejects_fingerprint_mismatch(tmp_path):
    from experiments.catboost_matched_ablation import load_candidates

    payload = _default_candidate_payload()
    payload["candidates"][4]["iterations"] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        load_candidates(_write_candidate_payload(tmp_path, payload))

def test_fixed_blocking_splitter_is_ordered_embargoed_and_non_overlapping():
    from experiments.catboost_matched_ablation import fixed_splitter

    splitter = fixed_splitter()
    assert splitter.n_splits == 5
    assert splitter.train_frac == 0.8
    assert splitter.embargo == 4

    folds = list(splitter.split(range(1_000)))
    assert len(folds) == 5
    seen = set()
    for train, validation in folds:
        assert train[-1] < validation[0]
        assert validation[0] - train[-1] - 1 == 4
        current = set(train) | set(validation)
        assert seen.isdisjoint(current)
        seen.update(current)
    assert all(folds[i][1][-1] < folds[i + 1][0][0] for i in range(4))


def test_causal_regime_labels_use_672_trailing_bars_and_two_percent_thresholds():
    from experiments.catboost_matched_ablation import (
        MIN_REGIME_ROWS,
        REGIME_LOOKBACK_BARS,
        REGIME_RETURN_THRESHOLD,
        past_regime_labels,
    )

    assert REGIME_LOOKBACK_BARS == 672
    assert REGIME_RETURN_THRESHOLD == 0.02
    assert MIN_REGIME_ROWS == 50
    index = pd.date_range("2024-01-01", periods=675, freq="15min", tz="UTC")
    close = pd.Series(1.0, index=index)
    close.iloc[672] = 1.03
    close.iloc[673] = 0.97
    close.iloc[674] = 1.01

    regimes = past_regime_labels(close)

    assert regimes.iloc[:672].eq("unknown").all()
    assert regimes.iloc[672:].tolist() == ["bull", "bear", "sideways"]
    changed_future = close.copy()
    changed_future.iloc[674] = 100.0
    pd.testing.assert_series_equal(
        past_regime_labels(changed_future).iloc[:674], regimes.iloc[:674]
    )


def test_each_of_five_validation_folds_needs_every_regime():
    from experiments.catboost_matched_ablation import validate_fold_regime_counts

    adequate = [
        {"bull": 50, "sideways": 51, "bear": 52} for _ in range(5)
    ]
    validate_fold_regime_counts(adequate)
    with pytest.raises(ValueError, match="exactly five"):
        validate_fold_regime_counts(adequate[:4])
    inadequate = copy.deepcopy(adequate)
    inadequate[3]["bear"] = 49
    with pytest.raises(ValueError, match="fold 3.*at least 50"):
        validate_fold_regime_counts(inadequate)


def test_robust_f1_is_mean_of_exactly_five_fold_minima():
    from experiments.catboost_matched_ablation import robust_f1_score

    fold_scores = [
        (0.8, 0.7, 0.6),
        (0.5, 0.6, 0.7),
        (0.4, 0.9, 0.8),
        (0.3, 0.5, 0.7),
        (0.2, 0.4, 0.9),
    ]
    assert robust_f1_score(fold_scores) == pytest.approx(0.4)
    with pytest.raises(ValueError, match="exactly five"):
        robust_f1_score(fold_scores[:4])


def test_robust_economic_score_is_self_contained_and_nonfinite_is_worst():
    from experiments.catboost_matched_ablation import WORST_RANK_VALUE, robust_score

    metrics = {
        "pooled_sortino": 1.1,
        "pooled_sharpe": 0.8,
        "bull_sortino": 0.7,
        "sideways_sortino": 0.9,
        "bear_sortino": -0.2,
    }
    assert robust_score(**metrics) == -0.2
    assert robust_score(**{**metrics, "bear_sortino": float("nan")}) == WORST_RANK_VALUE


def test_f1_ranking_ties_use_nonnegative_integer_candidate_ids():
    from experiments.catboost_matched_ablation import f1_ranking_key

    tied = [
        {"candidate_id": 10, "robust_f1": 0.55, "overall_f1": 0.60},
        {"candidate_id": 2, "robust_f1": 0.55, "overall_f1": 0.60},
    ]
    assert min(tied, key=f1_ranking_key)["candidate_id"] == 2
    for invalid_id in ("2", "candidate_2", -1, True):
        with pytest.raises(ValueError, match="candidate_id"):
            f1_ranking_key({**tied[0], "candidate_id": invalid_id})
    with pytest.raises(ValueError, match="finite"):
        f1_ranking_key({**tied[0], "robust_f1": float("nan")})

def test_economic_ranking_is_adequacy_first_and_breaks_ties_stably():
    from experiments.catboost_matched_ablation import (
        economic_ranking_key,
        total_constraint_violation,
    )

    assert math.ceil(2 * 6 / 3) == 4
    adequate = _economic_row(candidate_id=9, policy_id=7)
    sparse = _economic_row(
        candidate_id=1,
        policy_id=1,
        trades=49,
        n_long=14,
        n_short=13,
        positive_segments=3,
        robust_score=99.0,
    )
    assert total_constraint_violation(adequate, n_segments=6) == 0.0
    assert total_constraint_violation(sparse, n_segments=6) == 5.0
    assert min([sparse, adequate], key=lambda row: economic_ranking_key(row, n_segments=6)) is adequate

    tied = [
        _economic_row(candidate_id=10, policy_id=8),
        _economic_row(candidate_id=2, policy_id=8),
        _economic_row(candidate_id=2, policy_id=3),
    ]
    winner = min(tied, key=lambda row: economic_ranking_key(row, n_segments=6))
    assert (winner["candidate_id"], winner["policy_id"]) == (2, 3)


def test_economic_ranking_rejects_ambiguous_ids_and_nonfinite_metrics():
    from experiments.catboost_matched_ablation import economic_ranking_key

    base = _economic_row()
    for field in ("candidate_id", "policy_id"):
        for invalid_id in ("2", "candidate_2", -1, True):
            with pytest.raises(ValueError, match=field):
                economic_ranking_key(
                    {**base, field: invalid_id}, n_segments=6
                )
    for field in ("robust_score", "pooled_sortino", "pooled_net"):
        with pytest.raises(ValueError, match="finite"):
            economic_ranking_key(
                {**base, field: float("nan")}, n_segments=6
            )
    for field in ("trades", "n_long", "n_short", "positive_segments"):
        for invalid_count in (float("nan"), "50", True):
            with pytest.raises(ValueError, match="integer"):
                economic_ranking_key(
                    {**base, field: invalid_count}, n_segments=6
                )

def test_expected_artifact_row_counts_are_locked_and_assertable():
    from experiments.catboost_matched_ablation import (
        EXPECTED_ARTIFACT_ROWS,
        assert_artifact_row_counts,
    )

    assert tuple(EXPECTED_ARTIFACT_ROWS.values()) == (
        45,
        1485,
        45,
        9,
        297,
        9,
        81,
        27,
        9,
    )
    assert_artifact_row_counts(EXPECTED_ARTIFACT_ROWS)
    with pytest.raises(AssertionError, match="forward_monthly"):
        assert_artifact_row_counts({**EXPECTED_ARTIFACT_ROWS, "forward_monthly": 53})


def _runner_fold_fixture():
    index = pd.date_range("2024-01-01", periods=12, freq="15min", tz="UTC")
    X = pd.DataFrame({"feature": np.arange(12, dtype=float)}, index=index)
    y = pd.Series([0, 1, 2] * 4, index=index, name="label")
    regimes = pd.Series(
        ["bull", "sideways", "bear"] * 4, index=index, name="regime"
    )
    fold = {
        "fold_id": 2,
        "train_positions": tuple(range(8)),
        "test_positions": tuple(range(9, 12)),
        "train_start": index[0],
        "train_end": index[8],
        "test_start": index[9],
        "test_end": index[-1] + pd.Timedelta(minutes=15),
    }
    return X, y, regimes, fold


def test_prediction_cache_fingerprint_tracks_every_fit_input():
    from experiments.run_catboost_matched_ablation import (
        prediction_cache_fingerprint,
    )

    X, y, regimes, fold = _runner_fold_fixture()
    kwargs = {
        "width_bps": 55,
        "candidate_params": {"depth": 6},
        "fold_metadata": fold,
        "X": X,
        "y": y,
        "regimes": regimes,
        "data_fingerprint": "m15-a",
    }
    baseline = prediction_cache_fingerprint(**kwargs)
    assert baseline == prediction_cache_fingerprint(**kwargs)
    assert baseline != prediction_cache_fingerprint(
        **{**kwargs, "width_bps": 65}
    )
    assert baseline != prediction_cache_fingerprint(
        **{**kwargs, "candidate_params": {"depth": 7}}
    )
    changed_X = X.copy()
    changed_X.iloc[0, 0] += 1.0
    assert baseline != prediction_cache_fingerprint(**{**kwargs, "X": changed_X})
    changed_regimes = regimes.copy()
    changed_regimes.iloc[0] = "bear"
    assert baseline != prediction_cache_fingerprint(
        **{**kwargs, "regimes": changed_regimes}
    )
    changed_fold = {
        **fold,
        "test_end": fold["test_end"] + pd.Timedelta(minutes=15),
    }
    assert baseline != prediction_cache_fingerprint(
        **{**kwargs, "fold_metadata": changed_fold}
    )


def test_prediction_cache_resumes_only_matching_valid_schema(tmp_path):
    from experiments.run_catboost_matched_ablation import (
        PREDICTION_COLUMNS,
        load_prediction_cache,
        write_prediction_cache,
    )

    path = tmp_path / "fold.parquet"
    frame = pd.DataFrame([{column: 0 for column in PREDICTION_COLUMNS}])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    write_prediction_cache(path, frame, "fingerprint-a")
    loaded = load_prediction_cache(path, "fingerprint-a")
    assert loaded is not None
    assert list(loaded.columns) == list(PREDICTION_COLUMNS)
    assert load_prediction_cache(path, "fingerprint-b") is None

    corrupted = pd.read_parquet(path)
    corrupted.loc[0, "p_short"] = 1
    corrupted.to_parquet(path, index=False)
    assert load_prediction_cache(path, "fingerprint-a") is None

    write_prediction_cache(path, frame.drop(columns="p_flat"), "fingerprint-a")
    assert load_prediction_cache(path, "fingerprint-a") is None


def test_classification_overall_f1_is_mean_of_five_fold_scores(tmp_path):
    from sklearn.metrics import f1_score

    from experiments.run_catboost_matched_ablation import MatchedAblationRunner

    runner = MatchedAblationRunner(
        output_root=tmp_path,
        widths=(55,),
        candidates=({"depth": 4},),
        candidate_ids=(0,),
        smoke=True,
    )
    fold_frames = []
    regime_values = []
    for fold_id in range(5):
        index = pd.date_range(
            "2024-01-01", periods=3, freq="15min", tz="UTC"
        ) + pd.Timedelta(days=fold_id)
        actual = np.array([0, 1, 2])
        predicted = actual if fold_id == 0 else np.array([1, 1, 1])
        fold_frames.append(
            pd.DataFrame(
                {"timestamp": index, "y_true": actual, "pred": predicted}
            )
        )
        regime_values.extend(zip(index, ("bull", "sideways", "bear")))
    regimes = pd.Series(
        {timestamp: regime for timestamp, regime in regime_values}, name="regime"
    )
    row = runner._classification_row(
        width=55,
        candidate_id=0,
        fold_frames=fold_frames,
        regimes=regimes,
    )
    expected = np.mean(
        [
            f1_score(
                frame["y_true"],
                frame["pred"],
                labels=(0, 1, 2),
                average="macro",
                zero_division=0,
            )
            for frame in fold_frames
        ]
    )
    assert row["overall_f1"] == pytest.approx(expected)

def test_fold_fit_trims_tail_weights_training_only_and_maps_model_classes():
    from experiments.run_catboost_matched_ablation import fit_fold_predictions

    X, y, regimes, fold = _runner_fold_fixture()
    fitted = {}

    class RecordingModel:
        classes_ = np.array([2, 0])

        def fit(self, train_X, train_y, sample_weight=None):
            fitted["index"] = train_X.index
            fitted["y"] = train_y.copy()
            fitted["weights"] = pd.Series(sample_weight, index=train_X.index)
            return self

        def predict_proba(self, test_X):
            return np.tile([0.8, 0.2], (len(test_X), 1))

    predictions = fit_fold_predictions(
        X=X,
        y=y,
        regimes=regimes,
        fold_metadata=fold,
        width_bps=55,
        candidate_id=4,
        candidate_params={"depth": 6},
        model_factory=lambda _params: RecordingModel(),
        refit_id="fit-abc",
    )

    assert fitted["index"].equals(X.index[:7])
    assert X.index[7] not in fitted["index"]
    totals = fitted["weights"].groupby(regimes.reindex(fitted["index"])).sum()
    assert totals.nunique() == 1
    assert len(fitted["weights"]) == len(fitted["y"])
    assert list(predictions.columns) == [
        "timestamp", "width_bps", "candidate_id", "fold_id", "y_true",
        "pred", "confidence", "p_short", "p_flat", "p_long",
        "train_start", "train_end", "test_start", "test_end", "refit_id",
    ]
    assert predictions["p_short"].eq(0.2).all()
    assert predictions["p_flat"].eq(0.0).all()
    assert predictions["p_long"].eq(0.8).all()
    assert predictions["pred"].eq(2).all()
    assert predictions["confidence"].eq(0.8).all()
    assert predictions["train_end"].eq(X.index[6]).all()
    assert predictions["test_end"].eq(fold["test_end"]).all()
    assert predictions["refit_id"].eq("fit-abc").all()


def test_build_walkforward_filters_m15_and_positioning_immediately(monkeypatch):
    import experiments.run_walkforward as walkforward

    boundary = pd.Timestamp("2026-04-01", tz="UTC")
    index = pd.DatetimeIndex(["2026-03-31 23:45Z", "2026-04-01 00:00Z"])
    bars = pd.DataFrame({"close": [1.0, 2.0]}, index=index)
    positioning = pd.DataFrame({"position_source": [3.0, 4.0]}, index=index)
    reads = iter((bars, positioning))
    seen = {}

    observed_filters = []

    def read_before(_path, *, filters=None):
        observed_filters.append(filters)
        frame = next(reads).copy()
        for column, operator, value in filters or ():
            assert column == "timestamp" and operator == "<"
            frame = frame.loc[frame.index < pd.Timestamp(value)]
        return frame

    monkeypatch.setattr(pd, "read_parquet", read_before)
    monkeypatch.setattr(walkforward, "FEATURE_COLS", ("feature",))
    monkeypatch.setattr(
        walkforward, "POSITIONING_FEATURE_COLS", ("position_feature",)
    )
    monkeypatch.setattr(walkforward, "ORDERFLOW_FEATURE_COLS", ())

    def fake_add_features(frame):
        seen["feature_input"] = frame.copy()
        out = frame.copy()
        out["feature"] = 1.0
        out["position_feature"] = out["position_source"]
        out["vol_60"] = 1.0
        out["hour"] = 0
        out["dayofweek"] = 0
        return out

    monkeypatch.setattr(walkforward, "add_features", fake_add_features)
    monkeypatch.setattr(
        walkforward,
        "make_label",
        lambda frame, threshold_bps, horizon: pd.Series(1, index=frame.index),
    )
    cfg = {
        "instruments": {
            "btc": {
                "working_parquet": "bars.parquet",
                "threshold_bps": 55,
            }
        },
        "dates": {
            "train": ["2024-01-01"],
            "walkforward": ["2025-01-01"],
        },
    }
    X, y, _ = walkforward.build_walkforward_xy(
        "btc",
        cfg,
        sentiment="none",
        orderflow=False,
        positioning=True,
        end_exclusive=boundary,
    )

    assert seen["feature_input"].index.tolist() == [index[0]]
    assert seen["feature_input"]["position_source"].tolist() == [3.0]
    assert observed_filters == [
        [("timestamp", "<", boundary.to_pydatetime())],
        [("timestamp", "<", boundary.to_pydatetime())],
    ]
    assert X.index.max() < boundary
    assert y.index.max() < boundary


def test_signal_safety_is_inclusive_at_the_full_path_boundary():
    from experiments.run_catboost_matched_ablation import (
        assert_before_boundary,
        safe_signal_mask,
    )

    end = pd.Timestamp("2024-02-01 00:00Z")
    signals = pd.DatetimeIndex(
        [end - pd.Timedelta(minutes=30), end - pd.Timedelta(minutes=15)]
    )
    assert safe_signal_mask(
        signals, end_exclusive=end, max_hold=1
    ).tolist() == [True, False]
    with pytest.raises(ValueError, match="lockbox boundary"):
        assert_before_boundary(
            pd.DataFrame(index=signals.append(pd.DatetimeIndex([end]))),
            end_exclusive=end,
        )


def test_one_width_one_candidate_smoke_resumes_without_refit(tmp_path):
    from experiments.run_catboost_matched_ablation import (
        MatchedAblationRunner,
        PreparedData,
        RecordingToyModel,
    )

    index = pd.date_range("2024-01-01", periods=4_000, freq="15min", tz="UTC")
    minute_index = pd.date_range(
        index.min(),
        index.max() + pd.Timedelta(minutes=14),
        freq="1min",
        tz="UTC",
    )
    prices = 100.0 + np.arange(len(index)) * 0.001
    bars = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.1,
            "low": prices - 0.1,
            "close": prices,
        },
        index=index,
    )
    minute_prices = 100.0 + np.arange(len(minute_index)) * 0.00001
    minute = pd.DataFrame(
        {
            "open": minute_prices,
            "high": minute_prices + 0.01,
            "low": minute_prices - 0.01,
            "close": minute_prices,
        },
        index=minute_index,
    )
    X = pd.DataFrame(
        {"feature": np.arange(len(index), dtype=float)}, index=index
    )
    y = pd.Series(np.arange(len(index)) % 3, index=index, name="label")
    regimes = pd.Series(
        np.resize(np.array(["bull", "sideways", "bear"]), len(index)),
        index=index,
        name="regime",
    )
    prepared = PreparedData(
        bars=bars,
        minute=minute,
        features={55: (X, y)},
        regimes=regimes,
        m15_fingerprint="m15-smoke",
        minute_fingerprint="m1-smoke",
    )
    RecordingToyModel.fit_calls = 0
    runner = MatchedAblationRunner(
        output_root=tmp_path,
        widths=(55,),
        candidates=({"depth": 1},),
        candidate_ids=(7,),
        model_factory=RecordingToyModel,
        prepared=prepared,
        smoke=True,
        stage1_only=True,
    )
    first = runner.run()
    assert first["fits"] == 5
    assert RecordingToyModel.fit_calls == 5
    assert first["classification_rows"] == 1
    assert first["policy_rows"] == 33
    classification = pd.read_parquet(tmp_path / "classification_2024.parquet")
    assert classification["candidate_id"].tolist() == [7]

    second = runner.run()
    assert second["fits"] == 0
    assert RecordingToyModel.fit_calls == 5
    assert second["cache_hits"] == 5


def test_span_fit_freezes_boundaries_and_trims_one_training_row():
    from experiments.run_catboost_matched_ablation import fit_span_predictions

    index = pd.date_range("2024-12-31 22:00", periods=16, freq="15min", tz="UTC")
    X = pd.DataFrame({"feature": np.arange(len(index), dtype=float)}, index=index)
    y = pd.Series(np.arange(len(index)) % 3, index=index, name="label")
    regimes = pd.Series(
        np.resize(np.array(["bull", "sideways", "bear"]), len(index)),
        index=index,
        name="regime",
    )
    fitted = {}

    class RecordingModel:
        classes_ = np.array([0, 1, 2])

        def fit(self, train_X, train_y, sample_weight=None):
            fitted["index"] = train_X.index
            return self

        def predict_proba(self, test_X):
            return np.tile([0.2, 0.6, 0.2], (len(test_X), 1))

    prediction = fit_span_predictions(
        X=X,
        y=y,
        regimes=regimes,
        train_end=pd.Timestamp("2025-01-01", tz="UTC"),
        test_start=pd.Timestamp("2025-01-01", tz="UTC"),
        test_end=pd.Timestamp("2025-01-01 02:00", tz="UTC"),
        width_bps=55,
        candidate_id=3,
        candidate_params={"depth": 5},
        model_factory=lambda _params: RecordingModel(),
        fit_id="calibration-w55-c03",
    )

    assert fitted["index"].max() == pd.Timestamp("2024-12-31 23:30", tz="UTC")
    assert prediction["timestamp"].min() == pd.Timestamp("2025-01-01", tz="UTC")
    assert prediction["timestamp"].max() < pd.Timestamp("2025-01-01 02:00", tz="UTC")
    assert prediction["refit_id"].unique().tolist() == ["calibration-w55-c03"]
    assert prediction["train_end"].unique().tolist() == [pd.Timestamp("2024-12-31 23:30", tz="UTC")]


def test_span_fit_applies_selected_lookback_before_tail_trim():
    from experiments.run_catboost_matched_ablation import fit_span_predictions

    index = pd.date_range("2024-12-20", "2025-01-02", freq="15min", inclusive="left", tz="UTC")
    X = pd.DataFrame({"feature": np.arange(len(index), dtype=float)}, index=index)
    y = pd.Series(np.arange(len(index)) % 3, index=index, name="label")
    regimes = pd.Series(
        np.resize(np.array(["bull", "sideways", "bear"]), len(index)),
        index=index,
    )
    fitted = {}

    class RecordingModel:
        classes_ = np.array([0, 1, 2])

        def fit(self, train_X, train_y, sample_weight=None):
            fitted["index"] = train_X.index
            return self

        def predict_proba(self, test_X):
            return np.tile([0.2, 0.6, 0.2], (len(test_X), 1))

    fit_span_predictions(
        X=X,
        y=y,
        regimes=regimes,
        train_end=pd.Timestamp("2025-01-01", tz="UTC"),
        test_start=pd.Timestamp("2025-01-01", tz="UTC"),
        test_end=pd.Timestamp("2025-01-02", tz="UTC"),
        width_bps=55,
        candidate_id=0,
        candidate_params={"depth": 5},
        model_factory=lambda _params: RecordingModel(),
        fit_id="lookback-test",
        lookback_days=2,
    )

    assert fitted["index"].min() == pd.Timestamp("2024-12-30", tz="UTC")
    assert fitted["index"].max() == pd.Timestamp("2024-12-31 23:30", tz="UTC")


def test_runner_consumes_notebook02_handoff(monkeypatch, tmp_path):
    from experiments.catboost_matched_ablation import CANDIDATE_POOL_FINGERPRINT
    from experiments.notebook02_handoff import write_pipeline_handoff
    from experiments.run_catboost_matched_ablation import MatchedAblationRunner

    upstream = pd.DataFrame(
        {
            "width_bps": [55, 65, 75],
            "sortino": [-5.7, -4.4, -2.5],
            "sharpe": [-4.0, -3.2, -1.7],
            "net_return": [-0.5, -0.3, -0.2],
            "trades": [500, 400, 300],
        }
    )
    handoff = tmp_path / "handoff.json"
    write_pipeline_handoff(
        path=handoff,
        upstream=upstream,
    )
    runner = MatchedAblationRunner(
        output_root=tmp_path / "matched",
        handoff_path=handoff,
        candidates=({"depth": 1},),
        candidate_ids=(0,),
        smoke=True,
    )

    assert runner.widths == (55, 65, 75)
    assert runner.lookback_days == {55: 180, 65: 180, 75: 180}
    assert runner.sentiment_mode == "none"
    assert len(runner.upstream_handoff_fingerprint) == 64


def test_forward_reporting_periods_and_policy_freeze_are_exact():
    from experiments.run_catboost_matched_ablation import (
        forward_reporting_periods,
        validate_frozen_policy_rows,
    )

    months, quarters = forward_reporting_periods()
    assert [label for label, _, _ in months] == [
        "2025-07", "2025-08", "2025-09", "2025-10", "2025-11",
        "2025-12", "2026-01", "2026-02", "2026-03",
    ]
    assert [label for label, _, _ in quarters] == ["2025-Q3", "2025-Q4", "2026-Q1"]
    frozen = pd.DataFrame(
        {
            "objective": ["F1"] * 9,
            "width_bps": [55] * 9,
            "candidate_id": [2] * 9,
            "policy_id": [7] * 9,
            "tau": [0.55] * 9,
            "tp_bps": [200] * 9,
            "sl_bps": [100] * 9,
            "max_hold": [1] * 9,
            "fit_id": ["forward-w55-c02"] * 9,
        }
    )
    validate_frozen_policy_rows(frozen)
    changed = frozen.copy()
    changed.loc[8, "tau"] = 0.60
    with pytest.raises(ValueError, match="changed inside forward evaluation"):
        validate_frozen_policy_rows(changed)


def test_h1_monthly_walkforward_spans_are_causal_and_complete():
    from experiments.run_catboost_matched_ablation import calibration_month_spans

    spans = calibration_month_spans()

    assert len(spans) == 6
    assert [start.strftime("%Y-%m") for start, _ in spans] == [
        "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
    ]
    assert spans[0][0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert spans[-1][1] == pd.Timestamp("2025-07-01", tz="UTC")
    assert all(left_end == right_start for (_, left_end), (right_start, _) in zip(spans[:-1], spans[1:]))


def test_monthly_prediction_frames_reject_leakage_and_combine_six_fits():
    from experiments.run_catboost_matched_ablation import (
        calibration_month_spans,
        combine_monthly_predictions,
    )

    frames = []
    for month_id, (start, end) in enumerate(calibration_month_spans()):
        timestamp = pd.date_range(start, periods=2, freq="15min", tz="UTC")
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamp,
                    "train_end": [start - pd.Timedelta(minutes=30)] * 2,
                    "test_start": [start] * 2,
                    "test_end": [end] * 2,
                    "refit_id": [f"monthly-{month_id}"] * 2,
                }
            )
        )

    combined = combine_monthly_predictions(frames)
    assert len(combined) == 12
    assert combined["refit_id"].nunique() == 6
    assert combined["timestamp"].is_monotonic_increasing
    assert not combined["timestamp"].duplicated().any()

    leaked = [frame.copy() for frame in frames]
    leaked[2]["train_end"] = leaked[2]["test_start"]
    with pytest.raises(ValueError, match="future or current-month training data"):
        combine_monthly_predictions(leaked)

def test_forward_evidence_is_sliced_from_one_series_with_zero_periods_retained():
    from experiments.run_catboost_matched_ablation import (
        summarize_forward_evidence,
        validate_frozen_policy_rows,
    )

    index = pd.DatetimeIndex(
        [pd.Timestamp(f"{month}-01", tz="UTC") for month in (
            "2025-07", "2025-08", "2025-09", "2025-10", "2025-11",
            "2025-12", "2026-01", "2026-02", "2026-03",
        )]
    )
    per_bar = pd.Series(
        [0.01, 0.0, -0.01, 0.02, 0.0, 0.01, -0.005, 0.0, 0.015],
        index=index,
    )
    ledger = pd.DataFrame(
        {
            "entry_time": index[[0, 3]],
            "side": [1, -1],
            "gross_return": [0.012, 0.022],
            "net_return": [0.01, 0.02],
        }
    )
    regimes = pd.Series(
        ["bull", "bull", "sideways", "sideways", "sideways", "bear", "bear", "bear", "bull"],
        index=index,
    )
    policy = {
        "objective": "F1", "width_bps": 55, "candidate_id": 2,
        "policy_id": 7, "tau": 0.55, "tp_bps": 200, "sl_bps": 100,
        "max_hold": 1, "fit_id": "forward-w55-c02",
    }

    monthly, quarterly, summary = summarize_forward_evidence(
        per_bar=per_bar,
        ledger=ledger,
        regimes=regimes,
        policy=policy,
    )

    assert len(monthly) == 9
    assert len(quarterly) == 3
    assert len(summary) == 1
    assert monthly["trades"].sum() == 2
    assert monthly["net_return"].sum() == pytest.approx(summary.iloc[0]["net_return"])
    assert quarterly["net_return"].sum() == pytest.approx(summary.iloc[0]["net_return"])
    assert monthly.loc[monthly["period"] == "2025-08", "trades"].iloc[0] == 0
    validate_frozen_policy_rows(monthly)
