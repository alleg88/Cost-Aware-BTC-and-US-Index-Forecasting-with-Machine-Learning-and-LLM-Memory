from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.index_policy_transfer import (
    _canonical_frame_sha256,
    _expected_fit_failure,
    build_btc_policy_registry,
    combine_union_predictions,
    filter_consecutive_signals,
    pre2025_volatility_ratio,
    volatility_scaled_registry,
)


def test_canonical_frame_hash_is_bounded_to_already_filtered_rows():
    index = pd.to_datetime(["2025-01-01", "2026-04-02"], utc=True)
    full = pd.DataFrame({"close": [100.0, 999.0]}, index=index)
    bounded = full.loc[full.index < pd.Timestamp("2026-04-01", tz="UTC")]
    first = _canonical_frame_sha256(bounded)
    full.loc[index[1], "close"] = 12345.0

    assert _canonical_frame_sha256(
        full.loc[full.index < pd.Timestamp("2026-04-01", tz="UTC")]
    ) == first


def _write_btc_sources(root: Path) -> None:
    union = root / "qualified_union_v1"
    union.mkdir(parents=True)
    (union / "protocol.json").write_text(
        json.dumps(
            {
                "protocol_version": "qualified-union-v1",
                "lockbox_2026_q2_used": False,
                "execution": {
                    "fee_bps_per_side": 5.0,
                    "max_hold": 1,
                    "sl_bps": 100,
                    "tp_bps": 200,
                },
                "members": [
                    {"model": "lstm", "tau": 0.75, "width_bps": 55},
                    {"model": "svm_linear", "tau": 0.0, "width_bps": 75},
                ],
            }
        ),
        encoding="utf-8",
    )
    base = (
        root
        / "tuning"
        / "all_model_sentiment_policy_180d_fixed15_monthly_h1"
        / "none"
    )
    for model, width, tau, h1_trades, n_long, n_short, positive, fwd_trades in (
        ("lstm", 55, 0.75, 79, 50, 29, 4, 54),
        ("svm_linear", 75, 0.0, 17, 11, 6, 2, 30),
    ):
        path = base / model
        path.mkdir(parents=True)
        pd.DataFrame(
            {
                "model_name": [model],
                "width_bps": [width],
                "tau": [tau],
                "tp_bps": [200],
                "sl_bps": [100],
                "max_hold": [1],
                "trades": [h1_trades],
                "n_long": [n_long],
                "n_short": [n_short],
                "positive_segments": [positive],
            }
        ).to_parquet(path / "selected_policies_2025h1.parquet")
        pd.DataFrame(
            {
                "model_name": [model],
                "width_bps": [width],
                "tau": [tau],
                "tp_bps": [200],
                "sl_bps": [100],
                "max_hold": [1],
                "trades": [fwd_trades],
                "net_return": [0.03 if model == "lstm" else 0.05],
                "sortino": [1.5 if model == "lstm" else 2.4],
            }
        ).to_parquet(path / "forward_summary.parquet")


def test_registry_contains_exact_union_members_source_hashes_and_eligibility(tmp_path: Path):
    _write_btc_sources(tmp_path)

    registry, manifest = build_btc_policy_registry(tmp_path)

    assert set(registry["model_name"]) == {"lstm", "svm_linear"}
    rows = registry.set_index("model_name")
    assert rows.loc["lstm", "width_bps"] == 55
    assert rows.loc["lstm", "tau"] == pytest.approx(0.75)
    assert rows.loc["svm_linear", "width_bps"] == 75
    assert rows.loc["svm_linear", "tau"] == pytest.approx(0.0)
    assert rows.loc["lstm", "source_h1_eligible"]
    assert not rows.loc["svm_linear", "source_h1_eligible"]
    assert rows.loc["svm_linear", "source_rank_label"] == "union_member_below_eligibility_floor"
    assert len(manifest["source_sha256"]) == 5
    assert len(manifest["registry_sha256"]) == 64


def test_scaled_registry_changes_only_bps_fields_and_keeps_source_values():
    registry = pd.DataFrame(
        {
            "model_name": ["lstm"],
            "width_bps": [55],
            "tau": [0.75],
            "tp_bps": [200],
            "sl_bps": [100],
            "hold_bars": [1],
        }
    )

    scaled = volatility_scaled_registry(registry, ratio=0.25)

    assert scaled.loc[0, "width_bps"] == 14
    assert scaled.loc[0, "tp_bps"] == 50
    assert scaled.loc[0, "sl_bps"] == 25
    assert scaled.loc[0, "source_width_bps"] == 55
    assert scaled.loc[0, "source_tp_bps"] == 200
    assert scaled.loc[0, "tau"] == pytest.approx(0.75)
    assert scaled.loc[0, "hold_bars"] == 1


def test_pre2025_ratio_uses_only_consecutive_2024_returns():
    btc_index = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 00:15", "2025-01-01 00:00"], utc=True
    )
    idx_index = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 00:15", "2025-01-01 00:00"], utc=True
    )
    btc = pd.DataFrame({"close": [100.0, 101.0, 500.0]}, index=btc_index)
    index = pd.DataFrame({"close": [100.0, 100.5, 500.0]}, index=idx_index)

    result = pre2025_volatility_ratio(index, btc)

    expected = abs(float(pd.Series([100.0, 100.5]).pct_change().iloc[-1])) / abs(
        float(pd.Series([100.0, 101.0]).pct_change().iloc[-1])
    )
    assert result["ratio"] == pytest.approx(expected)
    assert result["index_observations"] == 1
    assert result["btc_observations"] == 1


def test_transfer_signals_reject_session_gap_before_native_execution():
    bars_index = pd.to_datetime(
        ["2025-07-01 00:00", "2025-07-01 00:15", "2025-07-02 00:00"], utc=True
    )
    predictions = pd.DataFrame(
        {"timestamp": bars_index[:2], "pred": [2, 0], "confidence": [0.9, 0.9]}
    )

    filtered = filter_consecutive_signals(predictions, bars_index)

    assert filtered["timestamp"].tolist() == [bars_index[0]]


def test_union_combiner_applies_member_tau_and_vetoes_opposite_directions():
    timestamp = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")
    lstm = pd.DataFrame(
        {
            "timestamp": timestamp,
            "pred": [2, 2, 0, 2],
            "confidence": [0.80, 0.70, 0.90, 0.90],
        }
    )
    svm = pd.DataFrame(
        {
            "timestamp": timestamp,
            "pred": [1, 0, 0, 2],
            "confidence": [0.50, 0.90, 0.90, 0.90],
        }
    )

    combined = combine_union_predictions(
        {"lstm": lstm, "svm_linear": svm},
        {"lstm": 0.75, "svm_linear": 0.0},
    )

    # LSTM alone; LSTM abstains then SVM short; both short; both long.
    assert combined["pred"].tolist() == [2, 0, 0, 2]
    assert combined["confidence"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert combined["opposite_signal_veto"].tolist() == [False, False, False, False]

    svm.loc[0, "pred"] = 0
    vetoed = combine_union_predictions(
        {"lstm": lstm, "svm_linear": svm},
        {"lstm": 0.75, "svm_linear": 0.0},
    )
    assert vetoed.loc[0, "pred"] == 1
    assert bool(vetoed.loc[0, "opposite_signal_veto"])


def test_only_known_sparse_class_fit_failures_are_fail_closed_controls():
    assert _expected_fit_failure(
        ValueError("Requesting 3-fold cross-validation but provided less than 3 examples for at least one class.")
    )
    assert _expected_fit_failure(ValueError("training span needs at least two classes"))
    assert not _expected_fit_failure(ValueError("unexpected shape mismatch"))
