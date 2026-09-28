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
        "01_RQ1_A_BTC_data_labels_baseline.ipynb": (
            "data/btcusdt_m15_2024_2025.parquet",
            "data/btcusdt_positioning_m15_2024_2026.parquet",
        ),
        "02_RQ1_B_BTC_positioning_ablation.ipynb": (
            "data/btcusdt_m15_2024_2025.parquet",
            "data/btcusdt_positioning_m15_2024_2026.parquet",
            *_WALK_FORWARD,
        ),
        "03_RQ1_C_BTC_CatBoost_economic_objectives.ipynb": (
            "experiments/cache/tuning/notebook01_handoff/selected_widths.parquet",
            "experiments/cache/tuning/notebook02_no_sentiment/handoff.json",
            "experiments/cache/tuning/notebook02b_handoff/handoff.json",
            "experiments/cache/tuning/notebook02_no_sentiment/matched_catboost_monthly_h1",
        ),
        "12_RQ3_A_BTC_sentiment_data_methodology.ipynb": (
            "data/btcusdt_m15_2024_2025.parquet",
            "sentiment/raw/scores_btc.parquet",
            "sentiment/raw/scores_direct_events_btc.parquet",
            "sentiment/raw/scores_llm_btc.parquet",
            "sentiment/raw/scores_llm_direct_events_btc.parquet",
        ),
        "13_RQ3_B_BTC_nine_model_sentiment_ablation.ipynb": (
            "experiments/cache/tuning/notebook02b_handoff/handoff.json",
            "experiments/cache/tuning/all_model_sentiment_raw_180d_fixed15_v3",
        ),
        "14_RQ3_C_BTC_sentiment_policy_ablation.ipynb": (
            "experiments/cache/tuning/all_model_sentiment_policy_180d_fixed15_monthly_h1_v3",
        ),
        "06_RQ2_A_BTC_all_model_stacking.ipynb": (
            "experiments/cache/tuning/all_model_stacking",
        ),
        "07_RQ2_B_BTC_stacking_forward_validation.ipynb": (
            "experiments/cache/tuning/all_model_stacking",
        ),
        "08_RQ2_C_BTC_qualified_union_ensemble.ipynb": (
            "experiments/cache/qualified_union_v1",
        ),
        "09_RQ2_F_BTC_LSTM_GMADL_shadow.ipynb": (
            "experiments/cache/qualified_union_v1",
            "experiments/cache/unified_expected_net_ensemble",
            "experiments/cache/lstm_gmadl_shadow",
        ),
        "18_RQ4_A_BTC_LLM_policy_router.ipynb": (
            "experiments/cache/reflection_ensemble_v5",
            ".rebuild/btc_agent_weights_verified.json",
        ),
    },
    "Indices": {
        "04_RQ1_E_indices_nine_model_benchmark.ipynb": tuple(
            path
            for stream in ("usa500", "usatech")
            for path in (
                f"experiments/cache/index_replication/{stream}",
                f"experiments/cache/index_all_model_forward/{stream}",
            )
        ),
        "05_RQ1_F_indices_VIX_ablation.ipynb": tuple(
            f"experiments/cache/index_replication/{stream}/{name}"
            for stream in ("usa500", "usatech")
            for name in (
                "vix_admission.json",
                "vix_gate_paired_2024.parquet",
                "vix_gate_classification_2024.parquet",
            )
        ),
        "15_RQ3_D_indices_DeBERTa_sentiment.ipynb": tuple(
            f"experiments/cache/index_replication/{stream}"
            for stream in ("usa500", "usatech")
        ),
        "16_RQ3_E_indices_LLM_sentiment.ipynb": tuple(
            f"experiments/cache/index_replication/{stream}"
            for stream in ("usa500", "usatech")
        ),
        "10_RQ2_H_indices_all_model_ensemble.ipynb": tuple(
            f"experiments/cache/index_all_model_ensemble/{stream}"
            for stream in ("usa500", "usatech")
        ),
        "11_RQ2_I_indices_policy_comparison.ipynb": tuple(
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
        "19_RQ5_B_BTC_volatility_feature_consolidation.ipynb": (
            "experiments/cache/event_window_feature_consolidation",
        ),
        "20_RQ5_C_BTC_economic_direction_head.ipynb": (
            "experiments/cache/event_window_direction_head",
        ),
        "21_RQ5_D_BTC_channel_vs_volatility_ablation.ipynb": (
            "experiments/cache/channel_vs_volatility_ablation",
        ),
    },
    "Final confirmation": {
        "22_Lockbox_Q2_2026.ipynb": (
            ".rebuild/final_q2_verified.json",
            "experiments/cache/final_q2_lockbox/8185dee25cc468ed9be8b79aad59eb33eeda874f4cb245c31883ef127f205312",
        ),
        "17_RQ3_F_indices_Q2_sentiment_sensitivity.ipynb": (
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
