import json
import pandas as pd

from experiments.build_reflection_cache import DEFAULT_OUTPUT
from experiments.export_reflection_results import export_results


def test_reflection_exports_are_manifest_scoped_and_q2_free():
    summary = export_results(DEFAULT_OUTPUT)
    manifest = json.loads((DEFAULT_OUTPUT / "protocol_manifest.json").read_text(encoding="utf-8"))
    assert summary["protocol_hash"] == manifest["protocol_hash"]
    assert summary["sealed_start_utc"].startswith("2026-04-01")
    assert summary["promoted_policies"] == 0
    assert f"opened {summary['shadows_opened']} shadows" in summary["claim_reason"]
    assert "no promoted policy exists" in summary["claim_reason"]
    export_root = DEFAULT_OUTPUT / "exports"
    for name in (
        "controls.parquet", "benchmark_returns.parquet", "llm_reliability.parquet", "historical_evaluations.parquet",
        "candidate_funnel.parquet", "source_coverage.parquet", "ablation_registry.parquet",
        "evaluation_summary.json",
    ):
        assert (export_root / name).exists()
    controls = pd.read_parquet(export_root / "controls.parquet").set_index("control_id")
    assert controls.loc["svm_linear_dz75", "benchmark_role"] == "secondary_reporting"
    assert int(controls.loc["svm_linear_dz75", "trades"]) == 30
    assert abs(float(controls.loc["svm_linear_dz75", "net_return"]) - 0.05454678421263185) < 1e-12
    benchmark_returns = pd.read_parquet(export_root / "benchmark_returns.parquet")
    assert list(benchmark_returns.columns) == ["timestamp", "lstm", "svm_linear_dz75"]
    assert benchmark_returns[["lstm", "svm_linear_dz75"]].notna().all().all()
    ablations = pd.read_parquet(export_root / "ablation_registry.parquet")
    assert set(ablations["variant"]) == {
        "reflection_no_memory", "reflection_real_memory", "reflection_shuffled_memory",
        "news_none", "news_aggregate",
    }
    assert set(ablations["news_mode"]) == {"none", "aggregate", "bounded_text"}
