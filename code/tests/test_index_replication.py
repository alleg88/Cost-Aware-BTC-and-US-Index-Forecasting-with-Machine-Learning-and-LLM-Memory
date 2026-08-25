from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

import experiments.index_replication as index_replication
from experiments.index_replication import (
    CUTOFF,
    PROTOCOL_VERSION,
    IndexReplicationConfig,
    IndexReplicationRunner,
    build_selection_folds,
    fit_prediction_frame,
    make_index_label,
    simulate_one_bar,
    _frame_hash,
)


def _write_valid_vix_admission(runner: IndexReplicationRunner, selected_base: str = "price"):
    gate = pd.DataFrame({"gate": [1]})
    gate.to_parquet(runner.output_root / "vix_gate_paired_2024.parquet", index=False)
    protocol = json.loads(
        (runner.output_root / "protocol_manifest.json").read_text(encoding="utf-8")
    )
    (runner.output_root / "vix_admission.json").write_text(
        json.dumps(
            {
                "selected_base": selected_base,
                "protocol_version": PROTOCOL_VERSION,
                "protocol_hash": protocol["protocol_hash"],
                "gate_complete": True,
                "paired_table_sha256": _frame_hash(gate),
                "index_source_sha256": _frame_hash(runner.bars),
                "vix_source_sha256": _frame_hash(runner.vix_bars),
            }
        ),
        encoding="utf-8",
    )


def test_configs_are_instrument_isolated_and_q2_sealed(tmp_path: Path):
    usa = IndexReplicationConfig.for_stream(
        "usa500", data_dir=tmp_path, output_base=tmp_path / "cache"
    )
    tech = IndexReplicationConfig.for_stream(
        "usatech", data_dir=tmp_path, output_base=tmp_path / "cache"
    )

    assert usa.instrument == "USA500IDXUSD"
    assert tech.instrument == "USATECHIDXUSD"
    assert usa.output_root != tech.output_root
    assert usa.output_root.name == "usa500"
    assert tech.output_root.name == "usatech"
    assert usa.end_exclusive == tech.end_exclusive == CUTOFF
    assert usa.cost_bps == 2.0
    assert tech.cost_bps == 3.0


def test_runner_accepts_timezone_aware_datetime_index(tmp_path: Path):
    config = IndexReplicationConfig.for_stream(
        "usa500", data_dir=tmp_path, output_base=tmp_path / "cache"
    )
    index = pd.date_range("2024-01-01", periods=30, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": np.linspace(100.0, 101.0, len(index)),
            "volume": 100.0 + np.arange(len(index)) % 17,
            "available_at": index + pd.Timedelta(minutes=15),
            "complete_bar": True,
        },
        index=index,
    )

    runner = IndexReplicationRunner(config, bars=frame, vix_bars=frame)

    assert runner.bars.index.equals(index)


def test_runner_pushes_sealed_cutoff_into_parquet_read(tmp_path: Path, monkeypatch):
    config = IndexReplicationConfig.for_stream(
        "usa500", data_dir=tmp_path, output_base=tmp_path / "cache"
    )
    index = pd.to_datetime(["2026-03-31 23:30", "2026-04-01 00:00"], utc=True)
    frame = pd.DataFrame(
        {
            "open": [100.0, 999.0],
            "high": [101.0, 999.0],
            "low": [99.0, 999.0],
            "close": [100.0, 999.0],
            "volume": [1.0, 1.0],
            "available_at": index + pd.Timedelta(minutes=15),
            "complete_bar": True,
        },
        index=index.rename("timestamp"),
    )
    source = tmp_path / "usa500_15min_2021_2026.parquet"
    frame.to_parquet(source)
    observed = {}
    original = pd.read_parquet

    def capture(path, *args, **kwargs):
        observed["filters"] = kwargs.get("filters")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(index_replication.pd, "read_parquet", capture)
    runner = IndexReplicationRunner(config, vix_bars=frame.iloc[:1])

    assert observed["filters"]
    assert any(term[0] == "timestamp" and term[1] == "<" for term in observed["filters"])
    assert runner.bars.index.max() < CUTOFF


def test_selection_folds_are_five_ordered_nonoverlapping_and_embargoed():
    index = pd.date_range("2024-01-01", periods=1_000, freq="15min", tz="UTC")
    folds = build_selection_folds(index)

    assert len(folds) == 5
    seen: set[int] = set()
    for fold in folds:
        train = set(fold["train_positions"])
        test = set(fold["test_positions"])
        assert max(train) + 4 < min(test)
        assert not seen.intersection(train | test)
        assert fold["train_end"] <= fold["test_start"]
        seen.update(train | test)


def test_index_label_rejects_nonconsecutive_session_gap():
    index = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 00:15", "2024-01-02 00:00"], utc=True
    )
    bars = pd.DataFrame({"close": [100.0, 101.0, 110.0]}, index=index)

    label = make_index_label(bars, threshold_bps=5)

    assert label.iloc[0] == 2
    assert label.iloc[1] == -1
    assert label.iloc[2] == -1


def test_macro_feature_reader_filters_after_last_available_bar_before_aggregation(
    tmp_path: Path, monkeypatch
):
    from features import sentiment as sentiment_features

    pd.DataFrame(
        {
            "release_time": pd.to_datetime(
                ["2026-03-31 12:00", "2026-04-01 12:00"], utc=True
            )
        }
    ).to_parquet(tmp_path / "fred_calendar.parquet")
    observed: dict[str, pd.Series] = {}

    def capture(times, values, bar_index, prefix, **kwargs):
        observed["times"] = pd.to_datetime(times, utc=True)
        return pd.DataFrame(
            {"macro_decay": 0.0, "macro_cnt_24h": 0.0}, index=bar_index
        )

    monkeypatch.setattr(sentiment_features, "RAW_DIR", tmp_path)
    monkeypatch.setattr(sentiment_features, "_window_features", capture)
    bar_index = pd.date_range("2026-03-31 23:00", periods=4, freq="15min", tz="UTC")

    sentiment_features.build_macro_features(bar_index)

    assert observed["times"].max() < pd.Timestamp("2026-04-01", tz="UTC")


def test_tone_feature_reader_pushes_last_available_bar_into_parquet_filter(
    tmp_path: Path, monkeypatch
):
    from features import sentiment as sentiment_features

    pd.DataFrame(
        {
            "seendate": pd.to_datetime(
                ["2026-03-31 12:00", "2026-04-01 12:00"], utc=True
            ),
            "title": ["pre", "q2"],
            "tone": [1.0, -9.0],
        }
    ).to_parquet(tmp_path / "gdelt_usa500.parquet")
    observed = {}
    original = pd.read_parquet

    def capture(path, *args, **kwargs):
        observed["filters"] = kwargs.get("filters")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(sentiment_features, "RAW_DIR", tmp_path)
    monkeypatch.setattr(sentiment_features.pd, "read_parquet", capture)
    monkeypatch.setattr(
        "sentiment.dedup.first_seen_only", lambda frame: frame
    )
    bar_index = pd.date_range("2026-03-31 23:00", periods=4, freq="15min", tz="UTC")

    sentiment_features.build_tone_features("usa500", bar_index)

    assert observed["filters"]
    assert any(term[0] == "seendate" and term[1] == "<" for term in observed["filters"])


def test_matched_sentiment_arms_include_the_same_direct_event_pulse(
    tmp_path: Path, monkeypatch
):
    from features import sentiment as sentiment_features

    config = IndexReplicationConfig.for_stream(
        "usa500", data_dir=tmp_path, output_base=tmp_path / "cache"
    )
    index = pd.date_range("2024-01-01", periods=30, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": np.linspace(100.0, 101.0, len(index)),
            "volume": 100.0 + np.arange(len(index)) % 7,
            "available_at": index + pd.Timedelta(minutes=15),
            "complete_bar": True,
        },
        index=index,
    )
    runner = IndexReplicationRunner(config, bars=frame, vix_bars=frame)
    runner._feature_cache["price"] = pd.DataFrame({"price": 1.0}, index=index)
    runner._feature_cache["price_vix"] = runner._feature_cache["price"]
    _write_valid_vix_admission(runner)
    calls = []

    def capture(stream, bar_index, *, scorer):
        calls.append(scorer)
        return pd.DataFrame({"sent_direct_decay": 0.0}, index=bar_index)

    from features import index_sentiment

    monkeypatch.setattr(index_sentiment, "build_matched_index_features", capture)
    monkeypatch.setattr(
        index_sentiment,
        "build_deepseek_full_features",
        lambda stream, bar_index: pd.DataFrame(
            {"sent_direct_decay": 0.0, "sent_llm_relevance_decay": 0.0},
            index=bar_index,
        ),
    )
    runner._feature_frame("deberta_matched")
    runner._feature_frame("deepseek_matched")
    full = runner._feature_frame("deepseek_full")

    assert calls == ["classic", "llm"]
    assert "sent_direct_decay" in full.columns
    assert "sent_direct_pulse" not in full.columns


def test_runner_rejects_stale_vix_admission_protocol(tmp_path: Path):
    config = IndexReplicationConfig.for_stream(
        "usa500", data_dir=tmp_path, output_base=tmp_path / "cache"
    )
    index = pd.date_range("2024-01-01", periods=30, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
            "available_at": index + pd.Timedelta(minutes=15),
            "complete_bar": True,
        },
        index=index,
    )
    runner = IndexReplicationRunner(config, bars=frame, vix_bars=frame)
    (config.output_root / "vix_admission.json").write_text(
        json.dumps({"selected_base": "price", "protocol_version": "stale-v1"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="protocol"):
        runner._load_vix_decision()


def test_one_bar_replay_enters_next_open_keeps_both_sides_and_skips_gap():
    index = pd.to_datetime(
        [
            "2025-01-01 00:00",
            "2025-01-01 00:15",
            "2025-01-01 00:30",
            "2025-01-02 00:00",
        ],
        utc=True,
    )
    bars = pd.DataFrame(
        {
            "open": [99.0, 100.5, 101.5, 109.0],
            "close": [100.0, 101.0, 100.0, 110.0],
            "available_at": index + pd.Timedelta(minutes=15),
        },
        index=index,
    )
    predictions = pd.DataFrame(
        {
            "timestamp": index[:3],
            "pred": [2, 0, 2],
            "confidence": [0.9, 0.9, 0.9],
        }
    )

    ledger, per_bar = simulate_one_bar(
        bars,
        predictions,
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-01-03", tz="UTC"),
        tau=0.5,
        cost_bps=2.0,
    )

    assert ledger["side"].tolist() == [1, -1]
    assert ledger["cost_return"].eq(0.0002).all()
    assert ledger["signal_bar_open"].tolist() == index[:2].tolist()
    assert ledger["entry_bar_open"].tolist() == index[1:3].tolist()
    assert ledger.iloc[0]["entry_price"] == 100.5
    assert ledger.iloc[0]["net_return"] == pytest.approx(101 / 100.5 - 1 - 0.0002)
    assert ledger.iloc[1]["net_return"] == pytest.approx(-(100 / 101.5 - 1) - 0.0002)
    assert per_bar.loc[index[0]] == 0.0
    assert per_bar.loc[index[1]] == pytest.approx(ledger.iloc[0]["net_return"])


class _SpyModel:
    def __init__(self):
        self.fit_rows = 0
        self.fit_last = None
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X, y):
        self.fit_rows = len(X)
        self.fit_last = X.index[-1]
        return self

    def predict_proba(self, X):
        return np.tile(np.array([[0.2, 0.3, 0.5]]), (len(X), 1))


def test_fit_prediction_trims_train_tail_and_records_causal_boundaries():
    index = pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC")
    X = pd.DataFrame({"x": np.arange(100, dtype=float)}, index=index)
    y = pd.Series(np.arange(100) % 3, index=index)
    fold = {
        "fold_id": 0,
        "train_positions": tuple(range(70)),
        "test_positions": tuple(range(74, 100)),
        "train_start": index[0],
        "train_end": index[70],
        "test_start": index[74],
        "test_end": index[-1] + pd.Timedelta(minutes=15),
    }
    model = _SpyModel()

    prediction = fit_prediction_frame(
        X=X,
        y=y,
        model_factory=lambda params=None: model,
        model_name="spy",
        arm="price",
        width_bps=5,
        fold=fold,
        fit_id="spy-fit",
    )

    assert model.fit_rows == 69
    assert model.fit_last == index[68]
    assert prediction["timestamp"].min() == fold["test_start"]
    assert prediction["train_end_exclusive"].lt(prediction["test_start"]).all()
    assert prediction["pred"].eq(2).all()


def test_single_arm_runner_completes_h1_and_frozen_forward(tmp_path: Path):
    config = IndexReplicationConfig.for_stream(
        "usa500", data_dir=tmp_path, output_base=tmp_path / "cache"
    )
    index = pd.date_range(
        pd.Timestamp("2024-01-01", tz="UTC"),
        CUTOFF,
        freq="15min",
        inclusive="left",
    )
    wave = np.sin(np.arange(len(index)) / 3.0) * 0.002
    close = 100.0 * np.exp(np.cumsum(wave))
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 100.0 + np.arange(len(index)) % 17,
            "available_at": index + pd.Timedelta(minutes=15),
            "complete_bar": True,
        },
        index=index,
    )
    vix = bars.assign(close=np.linspace(15.0, 20.0, len(index)))
    runner = IndexReplicationRunner(
        config,
        model_factories={"spy": lambda params=None: _SpyModel()},
        bars=bars,
        vix_bars=vix,
    )
    _write_valid_vix_admission(runner)

    result = runner.run_models(
        model_names=("spy",), arms=("selected_base",), widths=(5,)
    )

    assert result["classification_rows"] == 1
    assert result["h1_grid_rows"] == len(
        (0.00, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
    )
    assert result["selected_policy_rows"] == 1
    assert result["forward_rows"] == 1
    forward = pd.read_parquet(config.output_root / "forward_summary.parquet")
    assert forward["h1_eligible"].iloc[0] == False
    assert forward["status"].iloc[0] == "ineligible_flat"
    assert forward["trades"].iloc[0] == 0
    assert result["max_prediction_timestamp"] is None
