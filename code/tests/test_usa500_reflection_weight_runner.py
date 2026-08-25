from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.index_all_model_ensemble import AlignedPanel
from experiments.run_usa500_reflection_weight_agent import (
    AGENT_VARIANTS,
    CONTROL_VARIANTS,
    finalize_experiment,
    main,
    prepare_common_artifacts,
    run_preflight,
    run_variant,
)
from reflection_agent.index_v1.config import MODEL_NAMES
from reflection_agent.index_v1.contracts import DirectBatchDecision, WeeklyWeightDecision
from reflection_agent.index_v1.engine import (
    add_control_payoffs,
    build_causal_week_states,
    build_registered_opportunities,
    build_weekly_cards,
    causal_hedge_schedule,
    constrained_grid_weights,
    eligible_memory,
    equal_weights,
    memory_statistics,
    opportunity_keys,
    replay_registered_sides,
    shuffled_memory,
    tune_static_h1_weights,
    weighted_sides,
)
from reflection_agent.v2.transport import SchemaCallResult


CODE_ROOT = Path(__file__).parents[1]
CONFIG = CODE_ROOT / "configs" / "usa500_reflection_weight_agent_v1.yaml"
MODEL_DIGEST = "d3f1c87447216481a8001f48c517a51e13bfb141853a8df5e52f81bf765dabc3"


def _bars(start: str = "2025-07-01T00:00:00Z", periods: int = 5) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="15min", tz="UTC")
    opens = [100.0, 100.0, 102.0, 101.0, 103.0]
    closes = [100.5, 102.0, 101.0, 103.0, 102.5]
    for index_value in range(5, periods):
        current = 100.0 + 0.2 * index_value
        opens.append(current)
        closes.append(current * (1.002 if index_value % 2 == 0 else 0.999))
    return pd.DataFrame(
        {
            "open": opens[:periods],
            "high": [max(open_, close) + 1.0 for open_, close in zip(opens, closes)][:periods],
            "low": [min(open_, close) - 1.0 for open_, close in zip(opens, closes)][:periods],
            "close": closes[:periods],
            "volume": [10.0] * periods,
            "available_at": index + pd.Timedelta(minutes=15),
            "complete_bar": [True] * periods,
        },
        index=index,
    )


def _panel(start: str = "2025-07-01T00:00:00Z", periods: int = 4) -> AlignedPanel:
    timestamps = pd.date_range(start, periods=periods, freq="15min", tz="UTC")
    probabilities: dict[str, np.ndarray] = {}
    for model_index, model in enumerate(MODEL_NAMES):
        pattern = [
            [0.02, 0.03, 0.95],
            [0.95, 0.03, 0.02] if model_index < 8 else [0.02, 0.03, 0.95],
            [0.02, 0.03, 0.95] if model_index < 5 else [0.95, 0.03, 0.02],
            [0.02, 0.03, 0.95],
        ]
        probabilities[model] = np.asarray(
            [pattern[index_value % 4] for index_value in range(periods)], dtype=float
        )
    return AlignedPanel(
        timestamp=timestamps,
        y_true=np.asarray([[2, 0, 2, 0][index_value % 4] for index_value in range(periods)], dtype=int),
        probabilities=probabilities,
        fit_ids={model: np.asarray([f"{model}-fit"] * periods) for model in MODEL_NAMES},
    )


def _state_frame(start: str = "2025-07-01T00:00:00Z", periods: int = 5) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "available_at": index + pd.Timedelta(minutes=15),
            "state_vix_regime": np.linspace(-0.5, 0.5, periods),
            "state_trailing_vol": np.linspace(0.01, 0.02, periods),
            "state_trailing_trend": np.linspace(-0.02, 0.02, periods),
        },
        index=index,
    )


def test_registered_opportunities_apply_frozen_vote_tau_and_nonoverlap():
    opportunities, original_ledger = build_registered_opportunities(
        _panel(),
        _bars(),
        start="2025-07-01T00:00:00Z",
        end="2025-07-01T01:15:00Z",
        tau=0.80,
        cost_bps=2.0,
    )

    assert len(opportunities) == len(original_ledger) == 3
    assert opportunities["opportunity_id"].tolist() == ["FWD-000000", "FWD-000001", "FWD-000002"]
    assert opportunities["original_side"].tolist() == [1, -1, 1]
    assert opportunities["signal_bar_open"].tolist() == list(_panel().timestamp[[0, 1, 3]])
    assert (opportunities["entry_time"].iloc[1:].reset_index(drop=True) >= opportunities["exit_time"].iloc[:-1].reset_index(drop=True)).all()
    probability_columns = [column for column in opportunities if column.startswith("m") and "_p_" in column]
    assert len(probability_columns) == 27
    assert opportunities[probability_columns].notna().all().all()


def test_replay_recomputes_both_sides_without_changing_opportunities():
    opportunities, _ = build_registered_opportunities(
        _panel(),
        _bars(),
        start="2025-07-01T00:00:00Z",
        end="2025-07-01T01:15:00Z",
        tau=0.80,
        cost_bps=2.0,
    )
    long, _ = replay_registered_sides(opportunities, [1, 1, 1], cost_bps=2.0)
    short, _ = replay_registered_sides(opportunities, [-1, -1, -1], cost_bps=2.0)

    assert opportunity_keys(long).equals(opportunity_keys(short))
    entry = 100.0
    exit_ = 102.0
    assert long.loc[0, "gross_return"] == pytest.approx(exit_ / entry - 1.0)
    assert short.loc[0, "gross_return"] == pytest.approx(1.0 - exit_ / entry)
    assert long.loc[0, "cost_return"] == pytest.approx(0.0002)
    assert short.loc[0, "net_return"] != pytest.approx(-long.loc[0, "net_return"])


def test_replay_requires_exact_binary_side_count_and_immutable_keys():
    opportunities, _ = build_registered_opportunities(
        _panel(),
        _bars(),
        start="2025-07-01T00:00:00Z",
        end="2025-07-01T01:15:00Z",
        tau=0.80,
        cost_bps=2.0,
    )

    with pytest.raises(ValueError, match="one side"):
        replay_registered_sides(opportunities, [1, -1], cost_bps=2.0)
    with pytest.raises(ValueError, match="LONG or SHORT"):
        replay_registered_sides(opportunities, [1, 0, -1], cost_bps=2.0)
    duplicated = pd.concat([opportunities, opportunities.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        replay_registered_sides(duplicated, [1, -1, 1, -1], cost_bps=2.0)


def test_opportunity_builder_rejects_q2_input():
    with pytest.raises(PermissionError, match="Q2"):
        build_registered_opportunities(
            _panel("2026-04-01T00:00:00Z"),
            _bars("2026-04-01T00:00:00Z"),
            start="2026-04-01T00:00:00Z",
            end="2026-04-01T01:00:00Z",
            tau=0.80,
            cost_bps=2.0,
        )


def test_prepare_common_artifacts_reconciles_and_hashes_frozen_inputs(tmp_path: Path):
    h1_panel = _panel("2025-01-06T00:00:00Z")
    forward_panel = _panel()
    bars = pd.concat([_bars("2025-01-06T00:00:00Z"), _bars()]).sort_index()
    _, expected = build_registered_opportunities(
        forward_panel,
        bars,
        start="2025-07-01T00:00:00Z",
        end="2026-04-01T00:00:00Z",
        tau=0.80,
        cost_bps=2.0,
    )

    manifest = prepare_common_artifacts(
        output_root=tmp_path / "usa500_reflection_weight_agent",
        bars=bars,
        h1_panel=h1_panel,
        forward_panel=forward_panel,
        expected_forward_ledger=expected,
        source_identity={"synthetic_fixture_sha256": "a" * 64},
        state_frame=pd.concat(
            [_state_frame("2025-01-06T00:00:00Z"), _state_frame()]
        ).sort_index(),
    )

    assert manifest["q2_loaded"] is False
    assert manifest["stage_counts"] == {"h1": 3, "forward": 3}
    assert manifest["reconciliation"]["exact_rows"] is True
    assert manifest["reconciliation"]["exact_keys"] is True
    assert len(manifest["protocol_hash"]) == 64
    assert len(manifest["implementation_hash"]) == 64
    assert set(manifest["prompt_hashes"]) == {
        "direct_system",
        "direct_task",
        "weekly_system",
        "weekly_task",
    }
    assert set(manifest["schema_hashes"]) == {"direct_batch", "weekly_weights"}
    assert all(len(value) == 64 for value in manifest["prompt_hashes"].values())
    assert all(len(value) == 64 for value in manifest["schema_hashes"].values())
    for relative, expected_hash in manifest["artifact_hashes"].items():
        path = tmp_path / "usa500_reflection_weight_agent" / "common" / relative
        assert path.is_file()
        assert len(expected_hash) == 64


def test_prepare_common_artifacts_fails_on_parent_ledger_mismatch(tmp_path: Path):
    forward_panel = _panel()
    bars = pd.concat([_bars("2025-01-06T00:00:00Z"), _bars()]).sort_index()
    _, expected = build_registered_opportunities(
        forward_panel,
        bars,
        start="2025-07-01T00:00:00Z",
        end="2026-04-01T00:00:00Z",
        tau=0.80,
        cost_bps=2.0,
    )
    expected.loc[0, "entry_price"] += 1.0

    with pytest.raises(ValueError, match="reconcile"):
        prepare_common_artifacts(
            output_root=tmp_path / "usa500_reflection_weight_agent",
            bars=bars,
            h1_panel=_panel("2025-01-06T00:00:00Z"),
            forward_panel=forward_panel,
            expected_forward_ledger=expected,
            source_identity={"synthetic_fixture_sha256": "a" * 64},
            state_frame=pd.concat(
                [_state_frame("2025-01-06T00:00:00Z"), _state_frame()]
            ).sort_index(),
        )


def _two_week_opportunities() -> pd.DataFrame:
    frames = []
    for week_number, start in enumerate(
        ("2025-01-06T00:00:00Z", "2025-01-13T00:00:00Z")
    ):
        frame, _ = build_registered_opportunities(
            _panel(start),
            _bars(start),
            start=start,
            end=pd.Timestamp(start) + pd.Timedelta(hours=1, minutes=15),
            tau=0.80,
            cost_bps=2.0,
        )
        frame["opportunity_id"] = [
            f"H1-W{week_number}-{index:03d}" for index in range(len(frame))
        ]
        frame["state_vix_regime"] = 0.5 + week_number
        frame["state_trailing_vol"] = 0.01 + 0.01 * week_number
        frame["state_trailing_trend"] = -0.02 + 0.03 * week_number
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_weighted_direction_is_binary_and_uses_original_side_on_exact_tie():
    opportunities = _two_week_opportunities().iloc[:3].copy()
    first_model = [1.0] + [0.0] * 8
    assert weighted_sides(opportunities, first_model).tolist() == [1, -1, 1]

    for model_index in range(9):
        opportunities[f"m{model_index:02d}_p_short"] = 0.5
        opportunities[f"m{model_index:02d}_p_flat"] = 0.0
        opportunities[f"m{model_index:02d}_p_long"] = 0.5
    assert weighted_sides(opportunities, equal_weights()).tolist() == opportunities[
        "original_side"
    ].tolist()


def test_deterministic_control_weights_are_exact_grid_and_material():
    weights = constrained_grid_weights([8.0, 3.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0])

    assert len(weights) == 9
    assert sum(weights) == pytest.approx(1.0, abs=1e-12)
    assert all(round(value * 100) == pytest.approx(value * 100) for value in weights)
    assert max(weights) <= 0.80
    assert sum(value >= 0.05 for value in weights) >= 3
    assert weights[0] > weights[-1]


def test_weekly_memory_is_resolved_strictly_before_the_next_week():
    opportunities = _two_week_opportunities()
    cards = build_weekly_cards(opportunities, cost_bps=2.0)

    assert len(cards) == 2
    first_start = pd.Timestamp(cards[0]["week_start"])
    second_start = pd.Timestamp(cards[1]["week_start"])
    assert eligible_memory(cards, first_start) == []
    visible = eligible_memory(cards, second_start)
    assert [card["week_id"] for card in visible] == [cards[0]["week_id"]]
    assert pd.Timestamp(visible[0]["available_at"]) < second_start
    with pytest.raises(PermissionError, match="Q2"):
        eligible_memory(cards, pd.Timestamp("2026-04-01T00:00:00Z"))


def test_weekly_cards_contain_model_economics_quality_state_and_controls():
    cards = build_weekly_cards(_two_week_opportunities(), cost_bps=2.0)
    card = cards[0]

    assert len(card["model_statistics"]) == 9
    assert {row["model_index"] for row in card["model_statistics"]} == set(range(9))
    assert set(card["controls"]) == {"original", "equal_weight"}
    assert set(card["market_state"]) == {
        "vix_regime",
        "trailing_volatility",
        "trailing_trend",
    }
    for row in card["model_statistics"]:
        assert row["sample_count"] == 3
        for key in (
            "directional_accuracy",
            "net_return",
            "long_net_return",
            "short_net_return",
            "brier_score",
            "mean_confidence",
            "marginal_net_vs_equal",
        ):
            assert np.isfinite(row[key])
    assert np.isfinite(card["agreement"])
    assert np.isfinite(card["probability_dispersion"])


def test_shuffled_memory_changes_attribution_but_not_timing_or_card_count():
    cards = build_weekly_cards(_two_week_opportunities(), cost_bps=2.0)
    shuffled = shuffled_memory(cards, seed=42)

    assert len(shuffled) == len(cards)
    assert [item["week_id"] for item in shuffled] == [item["week_id"] for item in cards]
    assert [item["available_at"] for item in shuffled] == [item["available_at"] for item in cards]
    original_nets = [row["net_return"] for row in cards[0]["model_statistics"]]
    shuffled_nets = [row["net_return"] for row in shuffled[0]["model_statistics"]]
    assert shuffled_nets != original_nets
    assert sorted(shuffled_nets) == sorted(original_nets)


def test_memory_statistics_report_literal_one_four_twelve_week_windows():
    cards = build_weekly_cards(_two_week_opportunities(), cost_bps=2.0)
    rows = memory_statistics(cards)

    assert len(rows) == 9
    assert {row["model_index"] for row in rows} == set(range(9))
    for row in rows:
        assert row["rolling_1"]["weeks"] == 1
        assert row["rolling_4"]["weeks"] == 2
        assert row["rolling_12"]["weeks"] == 2


def test_static_h1_and_hedge_controls_are_causal_and_constrained():
    opportunities = _two_week_opportunities()
    static = tune_static_h1_weights(opportunities, cost_bps=2.0)
    cards = build_weekly_cards(opportunities, cost_bps=2.0)
    schedule = causal_hedge_schedule(cards, eta=0.5)

    assert len(static) == 9 and sum(static) == pytest.approx(1.0, abs=1e-12)
    assert set(schedule) == {card["week_id"] for card in cards}
    first = schedule[cards[0]["week_id"]]
    second = schedule[cards[1]["week_id"]]
    assert first == constrained_grid_weights([0.0] * 9)
    assert second != first
    for weights in (static, first, second):
        assert sum(weights) == pytest.approx(1.0, abs=1e-12)
        assert all(value >= 0.05 for value in weights)


def test_state_join_and_week_snapshot_use_only_strictly_available_rows():
    state = _state_frame()
    opportunities, _ = build_registered_opportunities(
        _panel(),
        _bars(),
        start="2025-07-01T00:00:00Z",
        end="2025-07-01T01:15:00Z",
        tau=0.80,
        cost_bps=2.0,
        state_frame=state,
    )
    assert opportunities["state_vix_regime"].tolist() == pytest.approx(
        state.loc[opportunities["signal_bar_open"], "state_vix_regime"].tolist()
    )
    week_start = pd.Timestamp("2025-07-08T00:00:00Z")
    snapshots = build_causal_week_states(state, [week_start])
    assert snapshots.loc[0, "state_available_at"] < week_start
    assert snapshots.loc[0, "state_vix_regime"] == pytest.approx(0.5)

    leaking = state.copy()
    leaking.loc[leaking.index[0], "available_at"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="available"):
        build_registered_opportunities(
            _panel(),
            _bars(),
            start="2025-07-01T00:00:00Z",
            end="2025-07-01T01:15:00Z",
            tau=0.80,
            cost_bps=2.0,
            state_frame=leaking,
        )


def test_resolved_cards_include_static_and_hedge_counterfactual_controls():
    opportunities = _two_week_opportunities()
    cards = build_weekly_cards(opportunities, cost_bps=2.0)
    static = tune_static_h1_weights(opportunities, cost_bps=2.0)
    hedge = causal_hedge_schedule(cards, eta=0.5)
    augmented = add_control_payoffs(
        cards,
        opportunities,
        control_weights={
            "static_h1": {card["week_id"]: static for card in cards},
            "hedge_weekly": hedge,
        },
        cost_bps=2.0,
    )

    for card in augmented:
        assert set(card["controls"]) == {
            "original",
            "equal_weight",
            "static_h1",
            "hedge_weekly",
        }
        for control in card["controls"].values():
            assert np.isfinite(control["net_return"])
            assert control["long_trades"] + control["short_trades"] == card["opportunities"]


def _prepared_variant_cache(tmp_path: Path, *, forward_periods: int = 16) -> Path:
    root = tmp_path / "usa500_reflection_weight_agent"
    h1_panel = _panel("2025-01-06T00:00:00Z")
    forward_panel = _panel(periods=forward_periods)
    bars = pd.concat(
        [
            _bars("2025-01-06T00:00:00Z"),
            _bars(periods=forward_periods + 1),
        ]
    ).sort_index()
    state = pd.concat(
        [
            _state_frame("2025-01-06T00:00:00Z"),
            _state_frame(periods=forward_periods + 1),
        ]
    ).sort_index()
    _, expected = build_registered_opportunities(
        forward_panel,
        bars,
        start="2025-07-01T00:00:00Z",
        end="2026-04-01T00:00:00Z",
        tau=0.80,
        cost_bps=2.0,
    )
    prepare_common_artifacts(
        output_root=root,
        bars=bars,
        h1_panel=h1_panel,
        forward_panel=forward_panel,
        expected_forward_ledger=expected,
        source_identity={"synthetic_fixture_sha256": "a" * 64},
        state_frame=state,
    )
    return root


class _DecisionCaller:
    def __init__(self, *, fail: bool = False):
        from reflection_agent.index_v1.config import load_index_agent_config

        self.config = load_index_agent_config(CONFIG)
        self.fail = fail
        self.calls: list[dict] = []

    def call(self, *, role, messages, response_model, allowed_ids):
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "response_model": response_model,
                "allowed_ids": allowed_ids,
            }
        )
        if self.fail:
            value = None
            status = "schema_failure"
        elif response_model is DirectBatchDecision:
            value = DirectBatchDecision.model_validate(
                {
                    "schema_version": "1.0",
                    "decisions": [
                        {
                            "opportunity_index": index,
                            "side": "SHORT" if index % 2 else "LONG",
                            "evidence_indices": [],
                            "memory_indices": [],
                        }
                        for index in allowed_ids["opportunity_indices"]
                    ],
                }
            )
            status = "success"
        elif response_model is WeeklyWeightDecision:
            value = WeeklyWeightDecision.model_validate(
                {
                    "schema_version": "1.0",
                    "weights": [0.12, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11],
                    "evidence_indices": [],
                    "memory_indices": [],
                }
            )
            status = "success"
        else:
            raise AssertionError(response_model)
        raw = "" if value is None else value.model_dump_json()
        call_number = len(self.calls)
        return SchemaCallResult(
            status=status,
            value=value,
            raw_content=raw,
            request_hash=f"{call_number:064x}",
            response_hash=f"{call_number + 100:064x}",
            schema_hash=f"{call_number + 200:064x}",
            attempts=1,
            latency_seconds=0.01,
            metadata={"model": self.config.model},
            errors=() if value is not None else ("synthetic failure",),
        )


def _preflight(root: Path) -> None:
    caller = _DecisionCaller()
    result = run_preflight(
        output_root=root,
        caller=caller,
        model_record={
            "model": caller.config.model,
            "digest": MODEL_DIGEST,
            "capabilities": ["thinking"],
            "ollama_version": "0.0-test",
        },
    )
    assert result["passed"] is True


def test_registered_controls_and_agents_keep_exact_opportunities_and_call_contract(tmp_path: Path):
    root = _prepared_variant_cache(tmp_path)
    expected = len(pd.read_parquet(root / "common" / "forward_opportunities.parquet"))
    assert expected > 10
    _preflight(root)

    for variant in ("original_frozen", "hedge_weekly", "static_h1"):
        summary = run_variant(variant, output_root=root)
        assert summary["trades"] == expected
        assert summary["transport_calls"] == 0

    direct_caller = _DecisionCaller()
    direct = run_variant("direct_real_memory", output_root=root, caller=direct_caller)
    assert direct["trades"] == expected
    assert direct["transport_calls"] == 2
    direct_calls = [call for call in direct_caller.calls if call["role"] == "direct"]
    assert [len(call["allowed_ids"]["opportunity_indices"]) for call in direct_calls] == [10, 2]

    weekly_caller = _DecisionCaller()
    weekly = run_variant("weekly_real_memory", output_root=root, caller=weekly_caller)
    assert weekly["trades"] == expected
    assert weekly["transport_calls"] == 1
    assert [call["role"] for call in weekly_caller.calls] == ["weekly_weights"]

    original_keys = opportunity_keys(
        pd.read_parquet(root / "original_frozen" / "ledger.parquet")
    )
    for variant in (
        "hedge_weekly",
        "static_h1",
        "direct_real_memory",
        "weekly_real_memory",
    ):
        assert opportunity_keys(pd.read_parquet(root / variant / "ledger.parquet")).equals(
            original_keys
        )


def test_direct_resume_continues_after_checkpoint_without_repeating_batch(tmp_path: Path):
    root = _prepared_variant_cache(tmp_path)
    _preflight(root)
    caller = _DecisionCaller()

    partial = run_variant(
        "direct_no_memory",
        output_root=root,
        caller=caller,
        max_batches=1,
    )
    assert partial["status"] == "partial"
    assert len(caller.calls) == 1
    completed = run_variant("direct_no_memory", output_root=root, caller=caller)

    assert completed["status"] == "complete"
    assert len(caller.calls) == 2
    decisions = pd.read_parquet(root / "direct_no_memory" / "decisions.parquet")
    assert len(decisions) == completed["trades"]
    assert decisions["opportunity_id"].is_unique


def test_invalid_agent_output_uses_registered_safe_fallbacks(tmp_path: Path):
    root = _prepared_variant_cache(tmp_path, forward_periods=4)
    _preflight(root)
    opportunities = pd.read_parquet(root / "common" / "forward_opportunities.parquet")

    direct = run_variant(
        "direct_shuffled_memory",
        output_root=root,
        caller=_DecisionCaller(fail=True),
    )
    direct_decisions = pd.read_parquet(
        root / "direct_shuffled_memory" / "decisions.parquet"
    )
    assert direct["transport_failures"] == 1
    assert direct_decisions["side"].tolist() == opportunities["original_side"].tolist()

    weekly = run_variant(
        "weekly_shuffled_memory",
        output_root=root,
        caller=_DecisionCaller(fail=True),
    )
    weights = pd.read_parquet(root / "weekly_shuffled_memory" / "weekly_weights.parquet")
    assert weekly["transport_failures"] == 1
    for index in range(9):
        assert weights.loc[0, f"w{index:02d}"] == pytest.approx(1 / 9)


def test_cli_runs_all_registered_controls_without_llm(tmp_path: Path, capsys):
    root = _prepared_variant_cache(tmp_path, forward_periods=4)

    assert main(["--run-controls", "--cache-dir", str(root)]) == 0
    capsys.readouterr()
    for variant in ("original_frozen", "hedge_weekly", "static_h1"):
        summary = json.loads((root / variant / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "complete"
        assert summary["transport_calls"] == 0


def _run_all_synthetic_variants(root: Path) -> None:
    _preflight(root)
    for variant in CONTROL_VARIANTS:
        run_variant(variant, output_root=root)
    for variant in AGENT_VARIANTS:
        run_variant(variant, output_root=root, caller=_DecisionCaller())


def test_finalize_writes_finite_ranked_paired_tables_and_candidate_gate(tmp_path: Path):
    root = _prepared_variant_cache(tmp_path, forward_periods=8)
    _run_all_synthetic_variants(root)

    summary = finalize_experiment(output_root=root)

    assert summary["status"] == "complete"
    assert summary["registered_arms"] == 9
    assert summary["q2_loaded"] is False
    expected_artifacts = {
        "results_table.parquet",
        "side_results.parquet",
        "weekly_results.parquet",
        "memory_ablation.parquet",
        "paired_bootstrap.parquet",
        "leakage_audit.parquet",
        "candidate_gate.parquet",
        "summary.json",
        "manifest.json",
    }
    assert expected_artifacts.issubset({path.name for path in root.iterdir()})
    results = pd.read_parquet(root / "results_table.parquet")
    assert len(results) == 9
    assert results["net_return"].is_monotonic_decreasing
    assert results["trades"].nunique() == 1
    for column in (
        "net_return",
        "daily_sharpe",
        "daily_sortino",
        "max_drawdown",
        "win_rate",
        "trades_per_day",
    ):
        assert np.isfinite(results[column].to_numpy(float)).all()
    paired = pd.read_parquet(root / "paired_bootstrap.parquet")
    original = paired.loc[paired["variant"].eq("original_frozen")].iloc[0]
    assert original["net_delta"] == pytest.approx(0.0)
    assert original["ci_low"] == pytest.approx(0.0)
    assert original["ci_high"] == pytest.approx(0.0)
    gates = pd.read_parquet(root / "candidate_gate.parquet")
    assert set(gates["variant"]) == {"direct_real_memory", "weekly_real_memory"}
    assert gates["minimum_side_trades_pass"].eq(False).all()


def test_finalize_proves_memory_timing_nonmemory_parity_and_q2_seal(tmp_path: Path):
    root = _prepared_variant_cache(tmp_path, forward_periods=8)
    _run_all_synthetic_variants(root)
    finalize_experiment(output_root=root)

    audit = pd.read_parquet(root / "leakage_audit.parquet")
    assert not audit.empty
    assert audit["passed"].all()
    assert {
        "q2_sealed",
        "exact_opportunity_keys",
        "binary_decisions",
        "same_week_outcomes_absent",
        "weekly_weights_precede_opportunities",
        "memory_ablation_nonmemory_parity",
        "shuffled_memory_timing_parity",
        "execution_reconciled",
    }.issubset(set(audit["check_id"]))
    for variant in ("direct_real_memory", "direct_no_memory", "direct_shuffled_memory"):
        decisions = pd.read_parquet(root / variant / "decisions.parquet")
        assert "nonmemory_hash" in decisions
        assert "memory_max_available_at" in decisions
    hashes = {
        variant: pd.read_parquet(root / variant / "decisions.parquet")["nonmemory_hash"].tolist()
        for variant in ("direct_real_memory", "direct_no_memory", "direct_shuffled_memory")
    }
    assert hashes["direct_real_memory"] == hashes["direct_no_memory"] == hashes[
        "direct_shuffled_memory"
    ]


def test_variant_resume_rejects_protocol_identity_drift(tmp_path: Path):
    root = _prepared_variant_cache(tmp_path, forward_periods=4)
    run_variant("original_frozen", output_root=root)
    checkpoint_path = root / "original_frozen" / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["protocol_hash"] = "0" * 64
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(ValueError, match="resume identity"):
        run_variant("original_frozen", output_root=root)


def test_cli_finalizes_only_after_all_registered_arms_exist(tmp_path: Path, capsys):
    root = _prepared_variant_cache(tmp_path, forward_periods=4)
    with pytest.raises(FileNotFoundError, match="registered arm"):
        main(["--finalize", "--cache-dir", str(root)])
    _run_all_synthetic_variants(root)

    assert main(["--finalize", "--cache-dir", str(root)]) == 0
    capsys.readouterr()
    assert (root / "manifest.json").is_file()
