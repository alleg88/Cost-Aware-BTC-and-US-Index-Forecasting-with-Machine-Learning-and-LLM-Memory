from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_causal_crossings_start_unarmed_and_require_rearm():
    from experiments.channel_blind_union_addon import causal_crossings

    index = pd.date_range("2024-07-01", periods=18, freq="5min", tz="UTC")
    scores = pd.Series(
        [
            0.9,
            0.1,
            0.8,
            0.9,
            0.1,
            0.8,
            0.9,
            0.1,
            0.9,
            0.1,
            0.2,
            0.3,
            0.1,
            0.8,
            0.9,
            0.1,
            0.8,
            0.1,
        ],
        index=index,
    )

    actual = causal_crossings(
        scores,
        threshold=0.5,
        refractory=pd.Timedelta(minutes=60),
    )

    assert list(actual) == [index[2], index[16]]


def test_frequency_threshold_is_deterministic_and_maximises_rate_below_target():
    from experiments.channel_blind_union_addon import (
        AddonConfig,
        select_frequency_threshold,
    )

    index = pd.date_range("2024-07-01", periods=288 * 3, freq="5min", tz="UTC")
    values = np.full(len(index), 0.1)
    values[np.arange(12, len(index), 30)] = 0.5
    for day_start in range(0, len(index), 288):
        values[[day_start + 60, day_start + 150, day_start + 240]] = 0.9
    scores = pd.Series(values, index=index)

    threshold_a, frontier_a = select_frequency_threshold(scores, AddonConfig())
    threshold_b, frontier_b = select_frequency_threshold(scores, AddonConfig())

    assert threshold_a == threshold_b == pytest.approx(0.9)
    pd.testing.assert_frame_equal(frontier_a, frontier_b)
    selected = frontier_a.loc[frontier_a["selected"]].iloc[0]
    assert selected["activations"] == 9
    assert selected["activations_per_day"] == pytest.approx(3.0)
    assert selected["activations"] == frontier_a.loc[
        frontier_a["activations_per_day"].le(3.0), "activations"
    ].max()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("addon_net_return", 0.0),
        ("addon_trades", 19),
        ("addon_long_trades", 0),
        ("addon_short_trades", 0),
        ("apr_jun_addon_net_return", -0.0001),
    ],
)
def test_h1_access_gate_requires_every_registered_condition(field, value):
    from experiments.channel_blind_union_addon import h1_access_gate

    summary = {
        "addon_net_return": 0.01,
        "addon_trades": 20,
        "addon_long_trades": 10,
        "addon_short_trades": 10,
        "apr_jun_addon_net_return": 0.0,
    }
    assert h1_access_gate(summary)

    summary[field] = value

    assert not h1_access_gate(summary)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("addon_net_return", 0.0),
        ("addon_long_trades", 0),
        ("addon_short_trades", 0),
        ("combined_trades", 74),
        ("combined_net_return", 0.06315),
    ],
)
def test_forward_promotion_requires_positive_increment_and_more_trades(field, value):
    from experiments.channel_blind_union_addon import forward_promotion_gate

    summary = {
        "addon_net_return": 0.01,
        "addon_long_trades": 3,
        "addon_short_trades": 4,
        "combined_trades": 81,
        "combined_net_return": 0.07,
        "union_net_return": 0.06315,
    }
    assert forward_promotion_gate(summary)

    summary[field] = value

    assert not forward_promotion_gate(summary)


def test_union_side_is_asof_available_and_never_future():
    from experiments.channel_blind_union_addon import align_union_asof

    union_time = pd.date_range("2025-01-01", periods=3, freq="15min", tz="UTC")
    union = pd.DataFrame(
        {
            "timestamp": union_time,
            "union_signal": [0.0, 0.0, 0.0],
            "member_conflict": [False, False, False],
            "lstm_latent_side": [1.0, -1.0, 1.0],
            "svm_linear_latent_side": [1.0, -1.0, 1.0],
        }
    )
    activations = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(
                ["2025-01-01 00:20Z", "2025-01-01 00:35Z"]
            )
        }
    )

    actual = align_union_asof(activations, union)

    assert list(actual["union_timestamp"]) == [union_time[0], union_time[1]]
    assert (
        actual["union_timestamp"] + pd.Timedelta(minutes=15)
        <= actual["decision_time"]
    ).all()


def test_existing_union_position_and_side_disagreement_are_rejected():
    from experiments.channel_blind_union_addon import qualify_union_side

    aligned = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(
                ["2025-01-01 00:25Z", "2025-01-01 00:40Z", "2025-01-01 00:55Z"]
            ),
            "union_signal": [0.0, 0.0, 0.0],
            "member_conflict": [False, False, False],
            "lstm_latent_side": [1.0, 1.0, -1.0],
            "svm_linear_latent_side": [1.0, -1.0, -1.0],
        }
    )
    ledger = pd.DataFrame(
        {
            "entry_time": [pd.Timestamp("2025-01-01 00:20Z")],
            "intrabar_exit_time": [pd.Timestamp("2025-01-01 00:34Z")],
        }
    )

    actual = qualify_union_side(aligned, ledger)

    assert list(actual["decision"]) == [
        "reject_union_open",
        "reject_side_disagreement",
        "accept",
    ]
    assert actual.loc[actual["decision"].eq("accept"), "side"].item() == -1


def test_non_overlapping_selector_uses_actual_exit_minute():
    from experiments.channel_blind_union_addon import select_non_overlapping_addons

    start = pd.Timestamp("2025-01-01", tz="UTC")
    paths = pd.DataFrame(
        {
            "decision_time": [
                start,
                start + pd.Timedelta(minutes=30),
                start + pd.Timedelta(minutes=65),
            ],
            "bars_held": [60, 10, 10],
            "path_complete": [True, True, True],
            "net_return": [0.01, 0.02, 0.03],
        }
    )

    actual = select_non_overlapping_addons(paths)

    assert list(actual["decision_time"]) == [
        start,
        start + pd.Timedelta(minutes=65),
    ]
    assert list(actual["actual_exit_time"]) == [
        start + pd.Timedelta(minutes=59),
        start + pd.Timedelta(minutes=74),
    ]


def test_replay_addons_applies_uniform_cost_in_portfolio_return_units():
    from experiments.channel_blind_union_addon import AddonConfig, replay_addons

    start = pd.Timestamp("2025-01-01", tz="UTC")
    index = pd.date_range(start, periods=120, freq="1min", tz="UTC")
    minute = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        index=index,
    )
    qualified = pd.DataFrame(
        {
            "decision_time": [start],
            "opportunity_score": [0.9],
            "reference_price": [100.0],
            "adaptive_barrier_bps": [100.0],
            "side": [1],
        }
    )

    actual = replay_addons(qualified, minute, AddonConfig())

    assert len(actual) == 1
    assert actual.iloc[0]["net_bps"] == pytest.approx(-10.0)
    assert actual.iloc[0]["net_r"] == pytest.approx(-0.1)
    assert actual.iloc[0]["net_return"] == pytest.approx(-0.001)
    assert actual.iloc[0]["actual_exit_time"] == index[-1]


def test_portfolio_series_preserves_union_and_reconciles_addons():
    from experiments.channel_blind_union_addon import build_portfolio_series

    index = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    union = pd.Series(
        [0.01, 0.0, -0.004, 0.0, 0.0, 0.0, 0.0, 0.0],
        index=index,
        name="net_return",
    )
    ledger = pd.DataFrame(
        {
            "decision_time": [index[1] + pd.Timedelta(minutes=5)],
            "net_return": [0.003],
        }
    )

    addon, combined = build_portfolio_series(union, ledger)

    pd.testing.assert_series_equal(combined - addon, union, check_names=False)
    assert np.isclose(addon.sum(), ledger["net_return"].sum())


def test_concurrency_audit_reports_later_union_overlap_without_blocking():
    from experiments.channel_blind_union_addon import concurrency_audit

    union = pd.DataFrame(
        {
            "entry_time": [pd.Timestamp("2025-01-01 00:30Z")],
            "intrabar_exit_time": [pd.Timestamp("2025-01-01 00:44Z")],
        }
    )
    addon = pd.DataFrame(
        {
            "decision_time": [pd.Timestamp("2025-01-01 00:00Z")],
            "actual_exit_time": [pd.Timestamp("2025-01-01 00:59Z")],
        }
    )

    actual = concurrency_audit(union, addon)

    assert actual["max_gross_exposure"] == 2
    assert actual["union_addon_overlap_minutes"] == 15


def test_pinned_w_source_and_channel_blind_h1_rows():
    from experiments.run_channel_vs_volatility_ablation import OPPORTUNITY_FEATURES
    from experiments.run_channel_blind_union_addon import load_hashed_w_oof

    frame = load_hashed_w_oof()

    assert set(frame["fold_id"]) == {"2024H2", "2025H1"}
    assert set(frame["model"]) == {"xgboost"}
    assert len(frame.loc[frame["fold_id"].eq("2025H1")]) == 52_128
    assert (
        frame.loc[frame["fold_id"].eq("2025H1"), "decision_time"]
        .sort_values()
        .diff()
        .dropna()
        .min()
        == pd.Timedelta(minutes=5)
    )
    assert not any("channel" in name for name in OPPORTUNITY_FEATURES)


def test_forward_gate_is_the_only_route_to_forward_loader():
    from experiments.run_channel_blind_union_addon import maybe_run_forward

    calls: list[str] = []
    failed = {
        "addon_net_return": -0.001,
        "addon_trades": 30,
        "addon_long_trades": 10,
        "addon_short_trades": 20,
        "apr_jun_addon_net_return": 0.001,
    }

    actual = maybe_run_forward(failed, lambda: calls.append("opened"))

    assert actual is None
    assert calls == []


def test_bounded_source_loader_rejects_q2():
    from experiments.run_channel_blind_union_addon import load_bounded_sources

    with pytest.raises(ValueError, match="Q2-2026"):
        load_bounded_sources(
            pd.Timestamp("2025-07-01", tz="UTC"),
            pd.Timestamp("2026-04-02", tz="UTC"),
        )


def test_frozen_h1_union_loader_verifies_and_preserves_sources():
    from experiments.run_channel_blind_union_addon import load_frozen_union

    signals, ledger, per_bar = load_frozen_union("h1")

    assert len(signals) == 17_376
    assert len(ledger) == 88
    assert len(per_bar) == 17_376
    assert signals["timestamp"].max() < pd.Timestamp("2025-07-01", tz="UTC")
    assert np.isclose(per_bar.sum(), ledger["net_return"].sum(), atol=1e-12)


def test_h1_runner_writes_registered_artifacts_without_forward():
    import json
    from pathlib import Path

    from experiments.run_channel_blind_union_addon import (
        OUTPUT_ROOT,
        UNION_CACHE,
        run_h1,
        sha256,
    )

    summary = run_h1()

    required = {
        "protocol.json",
        "manifest.json",
        "h1_threshold_calibration.csv",
        "h1_candidate_funnel.csv",
        "h1_addon_ledger.parquet",
        "h1_addon_per_bar.parquet",
        "h1_union_reference.parquet",
        "h1_combined_per_bar.parquet",
        "h1_summary.csv",
        "h1_monthly.csv",
        "summary.json",
    }
    assert required.issubset({path.name for path in Path(OUTPUT_ROOT).iterdir()})
    manifest = json.loads((OUTPUT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["artifact_hashes"].items():
        assert sha256(OUTPUT_ROOT / filename) == expected
    source = pd.read_parquet(UNION_CACHE / "h1_per_bar.parquet")
    reference = pd.read_parquet(OUTPUT_ROOT / "h1_union_reference.parquet")
    pd.testing.assert_frame_equal(reference, source)
    assert summary["forward_loaded"] is False
    assert summary["lockbox_2026_q2_used"] is False
    assert not any(path.name.startswith("forward_") for path in OUTPUT_ROOT.iterdir())


def test_completed_h1_result_rejects_addon_before_forward():
    from experiments.run_channel_blind_union_addon import OUTPUT_ROOT, run

    summary = run()
    result = summary["h1_result"]
    frontier = pd.read_csv(OUTPUT_ROOT / "h1_threshold_calibration.csv")
    funnel = pd.read_csv(OUTPUT_ROOT / "h1_candidate_funnel.csv")

    assert list(frontier.columns) == [
        "threshold",
        "activations",
        "observed_utc_days",
        "activations_per_day",
        "selected",
    ]
    selected = frontier.loc[frontier["selected"]].iloc[0]
    assert selected["activations"] == 552
    assert selected["activations_per_day"] == pytest.approx(3.0)
    assert result["raw_activations"] == 349
    assert result["addon_trades"] == 182
    assert result["addon_net_return"] == pytest.approx(-0.30433228187292594)
    assert result["addon_gross_bps_per_trade"] == pytest.approx(-6.721553949061856)
    assert result["side_accuracy_resolved"] == pytest.approx(0.44776119402985076)
    assert result["opportunity_rate"] == pytest.approx(0.38968481375358166)
    assert funnel.loc[funnel["decision"].eq("executed_addon"), "count"].item() == 182
    assert summary["h1_pass"] is False
    assert summary["forward_loaded"] is False
    assert summary["decision"] == "h1_fail_keep_union_v1"
    assert summary["final_ensemble"] == "qualified_union_v1"
    assert pd.Timestamp(summary["max_loaded_timestamp"]) < pd.Timestamp(
        "2025-07-01", tz="UTC"
    )
