"""Explicit reader contracts between canonical notebooks and rebuild outputs."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.notebook_hygiene import NOTEBOOK_SEQUENCES
from experiments.rebuild_graph import RebuildGraph


_WALK_FORWARD = tuple(
    f"experiments/cache/walkforward/btc_of-{arm}_dz{width}_devwf.parquet"
    for width in (55, 60, 65, 75)
    for arm in ("base", "pos", "placebo")
)

NOTEBOOK_INPUTS: dict[str, dict[str, tuple[str, ...]]] = {
    "Bitcoin": {
        "01_data_labels_and_baseline.ipynb": (
            "data/btcusdt_m15_2024_2025.parquet",
            "data/btcusdt_positioning_m15_2024_2026.parquet",
        ),
        "01b_positioning_ablation.ipynb": (
            "data/btcusdt_m15_2024_2025.parquet",
            "data/btcusdt_positioning_m15_2024_2026.parquet",
            *_WALK_FORWARD,
        ),
        "02b_catboost_economic_optuna.ipynb": (
            "experiments/cache/tuning/notebook01_handoff/selected_widths.parquet",
            "experiments/cache/tuning/notebook02_no_sentiment/handoff.json",
            "experiments/cache/tuning/notebook02b_handoff/handoff.json",
            "experiments/cache/tuning/notebook02_no_sentiment/matched_catboost_monthly_h1",
        ),
        "02c_sentiment_data_and_methodology.ipynb": (
            "data/btcusdt_m15_2024_2025.parquet",
            "sentiment/raw/scores_btc.parquet",
            "sentiment/raw/scores_direct_events_btc.parquet",
            "sentiment/raw/scores_llm_btc.parquet",
            "sentiment/raw/scores_llm_direct_events_btc.parquet",
        ),
        "02d_all_model_sentiment.ipynb": (
            "experiments/cache/tuning/notebook02b_handoff/handoff.json",
            "experiments/cache/tuning/all_model_sentiment_raw_180d_fixed15_v3",
        ),
        "02e_all_model_sentiment_policy.ipynb": (
            "experiments/cache/tuning/all_model_sentiment_policy_180d_fixed15_monthly_h1_v3",
        ),
        "03_all_model_stacking.ipynb": (
            "experiments/cache/tuning/all_model_stacking",
        ),
        "03a_stacking_forward.ipynb": (
            "experiments/cache/tuning/all_model_stacking",
        ),
        "03c_qualified_union_ensemble.ipynb": (
            "experiments/cache/qualified_union_v1",
        ),
        "04a_svm_temperature_calibration.ipynb": (
            "experiments/cache/svm_temperature_calibration",
        ),
        "04b_xgboost_strong_move_admission.ipynb": (
            "experiments/cache/xgb_strong_move_admission",
        ),
        "04d_unified_2021_ensemble.ipynb": (
            "experiments/cache/qualified_union_v1",
            "experiments/cache/unified_2021_ensemble",
        ),
        "04g_lstm_gmadl_shadow.ipynb": (
            "experiments/cache/qualified_union_v1",
            "experiments/cache/unified_expected_net_ensemble",
            "experiments/cache/lstm_gmadl_shadow",
        ),
        "04h_union_v1_episode_reentry.ipynb": (
            "experiments/cache/qualified_union_v1",
            "experiments/cache/union_v1_episode_reentry",
        ),
        "05c_causal_policy_router_agent.ipynb": (
            "experiments/cache/reflection_policy_router_v4/preflight.json",
            "experiments/cache/reflection_policy_router_v4/final_report.json",
            "experiments/cache/reflection_policy_router_v4/final_report_manifest.json",
            "experiments/cache/reflection_policy_router_v4/results_table.parquet",
            "experiments/cache/reflection_policy_router_v4/coverage_gates.parquet",
            "experiments/cache/reflection_policy_router_v4/paired_comparisons.parquet",
        ),
    },
    "Indices": {
        "06a_index_nine_models.ipynb": tuple(
            path
            for stream in ("usa500", "usatech")
            for path in (
                f"experiments/cache/index_replication/{stream}",
                f"experiments/cache/index_all_model_forward/{stream}",
            )
        ),
        "06b_index_vix.ipynb": tuple(
            f"experiments/cache/index_replication/{stream}/{name}"
            for stream in ("usa500", "usatech")
            for name in (
                "vix_admission.json",
                "vix_gate_paired_2024.parquet",
                "vix_gate_classification_2024.parquet",
            )
        ),
        "06c_index_deberta.ipynb": tuple(
            f"experiments/cache/index_replication/{stream}"
            for stream in ("usa500", "usatech")
        ),
        "06d_index_llm.ipynb": tuple(
            f"experiments/cache/index_replication/{stream}"
            for stream in ("usa500", "usatech")
        ),
        "06g_index_all_model_ensemble.ipynb": tuple(
            f"experiments/cache/index_all_model_ensemble/{stream}"
            for stream in ("usa500", "usatech")
        ),
        "06i_index_comparison.ipynb": tuple(
            f"experiments/cache/{family}/{stream}"
            for stream in ("usa500", "usatech")
            for family in (
                "index_all_model_forward",
                "index_trade_coverage",
                "index_side_calibration",
                "index_all_model_ensemble",
                "index_channel_replication",
            )
        ),
    },
    "Channels": {
        "A_channel_strategy.ipynb": (
            "experiments/cache/channel_study/BTCUSDT_dev_1hw60_5min_macro-hard_fa48d9681c/events.parquet",
            "experiments/cache/channel_study/BTCUSDT_dev_1hw60_5min_macro-hard_fa48d9681c/trades.parquet",
            "experiments/cache/channel_study/BTCUSDT_dev_1hw60_5min_macro-hard_fa48d9681c/ranking_e8_table.csv",
            "experiments/cache/channel_study/BTCUSDT_dev_1hw60_5min_macro-hard_fa48d9681c/ranking_e8_sides.csv",
        ),
        "U_volatility_timing_feature_consolidation.ipynb": (
            "experiments/cache/event_window_feature_consolidation",
        ),
        "V_economic_direction_head.ipynb": (
            "experiments/cache/event_window_direction_head",
        ),
        "W_channel_vs_volatility_ablation.ipynb": (
            "experiments/cache/channel_vs_volatility_ablation",
        ),
    },
    "Final confirmation": {
        "07_final_q2_lockbox.ipynb": (
            ".rebuild/final_q2_verified.json",
            "experiments/cache/final_q2_lockbox/8185dee25cc468ed9be8b79aad59eb33eeda874f4cb245c31883ef127f205312",
        ),
        "07a_q2_sentiment_sensitivity.ipynb": (
            ".rebuild/q2_sentiment_sensitivity_verified.json",
            "experiments/cache/q2_sentiment_sensitivity",
        ),
    },
}


@dataclass(frozen=True)
class DependencyAudit:
    sequence: str
    unregistered: tuple[str, ...]
    duplicate_producers: tuple[str, ...]


def audit_notebook_dependencies(sequence: str, graph: RebuildGraph) -> DependencyAudit:
    if sequence not in NOTEBOOK_SEQUENCES:
        raise ValueError(f"unknown notebook sequence: {sequence}")
    contracts = NOTEBOOK_INPUTS.get(sequence, {})
    missing_contracts = [
        f"notebook:{name}"
        for name in NOTEBOOK_SEQUENCES[sequence]
        if name not in contracts
    ]
    producers: dict[str, list[str]] = {}
    for task in graph.tasks:
        for output in task.outputs:
            producers.setdefault(output, []).append(task.id)
    required = sorted({path for paths in contracts.values() for path in paths})
    registered_sources = set(graph.sources)
    unregistered = missing_contracts + [
        path
        for path in required
        if not producers.get(path) and path not in registered_sources
    ]
    duplicate = [path for path in required if len(producers.get(path, ())) > 1]
    return DependencyAudit(
        sequence=sequence,
        unregistered=tuple(unregistered),
        duplicate_producers=tuple(duplicate),
    )


__all__ = ["DependencyAudit", "NOTEBOOK_INPUTS", "audit_notebook_dependencies"]
