"""Direct Run All must compute the selected notebook, not launch all readers."""
from pathlib import Path
import json

import pytest

from experiments.rebuild_graph import load_graph


CODE = Path(__file__).parents[1]


def _runtime():
    from importlib import import_module

    assert (CODE / "experiments/notebook_rebuild.py").is_file(), "Direct notebook rebuild is missing"
    return import_module("experiments.notebook_rebuild")


def test_selected_second_notebook_maps_to_positioning_not_the_full_project():
    runtime = _runtime()
    graph = load_graph(CODE / "configs/rebuild_tasks.json")
    assert set(runtime.notebook_tasks(graph, "02_RQ1_B_BTC_positioning_ablation.ipynb")) == {
        "market.binance.build_m15", "market.binance.build_positioning", "btc.positioning_ablation",
    }


@pytest.mark.parametrize("name", ["15_RQ3_D_indices_DeBERTa_sentiment.ipynb", "16_RQ3_E_indices_LLM_sentiment.ipynb"])
def test_index_sentiment_includes_forward_and_annotation_metadata(name):
    runtime = _runtime()
    graph = load_graph(CODE / "configs/rebuild_tasks.json")
    ids = set(runtime.notebook_tasks(graph, name))
    outputs = {output for task in graph.tasks if task.id in ids for output in task.outputs}
    for stream in ("usa500", "usatech"):
        assert f"experiments/cache/index_all_model_forward/{stream}" in outputs
        llm = "llm_" if name == "16_RQ3_E_indices_LLM_sentiment.ipynb" else ""
        assert f"sentiment/raw/scores_{llm}{stream}.manifest.json" in outputs


def test_first_notebook_uses_supplied_normalized_inputs_without_a_graph_run(tmp_path, monkeypatch, capsys):
    runtime = _runtime()
    for name in ("btcusdt_m15_2024_2025.parquet", "btcusdt_positioning_m15_2024_2026.parquet"):
        path = tmp_path / "data" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"test normalized input")
    monkeypatch.setattr(runtime, "execute_graph", lambda *a, **kw: pytest.fail("First notebook downloaded raw history"))
    report = runtime.rebuild_notebook_inputs("01_RQ1_A_BTC_data_labels_baseline.ipynb", tmp_path)
    assert report.executed == ()
    assert "normalized inputs" in capsys.readouterr().out


def test_every_notebook_has_a_registered_rebuild_plan():
    runtime = _runtime()
    from experiments.notebook_hygiene import execution_order

    graph = load_graph(CODE / "configs/rebuild_tasks.json")
    for name in execution_order():
        assert runtime.notebook_tasks(graph, name), name


def test_run_all_executes_selected_producer_with_ancestors_but_no_other_notebook(tmp_path):
    runtime = _runtime()
    (tmp_path / "configs").mkdir()
    (tmp_path / "raw.txt").write_text("source")
    outputs = [
        "experiments/cache/tuning/notebook01_handoff/selected_widths.parquet",
        "experiments/cache/tuning/notebook02_no_sentiment/handoff.json",
        "experiments/cache/tuning/notebook02b_handoff/handoff.json",
        "experiments/cache/tuning/notebook02_no_sentiment/matched_catboost_monthly_h1",
    ]
    tasks = []
    for name, inputs, results, depends_on in (
        ("upstream", ["raw.txt"], ["features.txt"], []),
        ("selected", ["features.txt"], outputs, ["upstream"]),
        ("unrelated", [], ["must-not-exist.txt"], []),
    ):
        tasks.append(dict(id=name, module=name, args=[], inputs=inputs, outputs=results,
                          depends_on=depends_on, profile="canonical", comparison="exact"))
        (tmp_path / f"{name}.py").write_text(
            "from pathlib import Path\n"
            + "".join(f"p = Path({result!r}); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('new')\n"
                      for result in results)
        )
    (tmp_path / "configs/rebuild_tasks.json").write_text(json.dumps(dict(schema_version=1, sources=["raw.txt"], tasks=tasks)))
    first = runtime.rebuild_notebook_inputs("03_RQ1_C_BTC_CatBoost_economic_objectives.ipynb", tmp_path)
    second = runtime.rebuild_notebook_inputs("03_RQ1_C_BTC_CatBoost_economic_objectives.ipynb", tmp_path)
    assert first.executed == ("upstream", "selected")
    assert second.executed == ("selected",) and second.skipped == ("upstream",)
    assert not (tmp_path / "must-not-exist.txt").exists()
    report = json.loads((tmp_path / ".rebuild/notebooks/03_RQ1_C_BTC_CatBoost_economic_objectives/preparation.json").read_text())
    assert report["status"] == "INPUTS_READY"
    assert report["full_project_executed"] is False
    assert report["numerical_equivalence_checked"] is False


def test_compact_recipe_mode_is_available_to_following_notebook_cells(tmp_path, monkeypatch):
    from experiments.channel_rebuild_contract import MODE_ENV, recomputed_handoffs
    runtime = _runtime()
    monkeypatch.delenv(MODE_ENV, raising=False)
    for name in ("btcusdt_m15_2024_2025.parquet", "btcusdt_positioning_m15_2024_2026.parquet"):
        path = tmp_path / "data" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"normalized source")
    (tmp_path / "configs").mkdir()
    graph = tmp_path / "configs/rebuild_tasks.json"
    graph.write_text(json.dumps({"schema_version": 1, "result_comparison": "not_required", "sources": [], "tasks": []}))
    runtime.rebuild_notebook_inputs("01_RQ1_A_BTC_data_labels_baseline.ipynb", tmp_path)
    assert recomputed_handoffs()
    report = json.loads((tmp_path / ".rebuild/notebooks/01_RQ1_A_BTC_data_labels_baseline/preparation.json").read_text())
    assert report["channel_handoffs"] == "recomputed"
    graph.write_text(json.dumps({"schema_version": 1, "sources": [], "tasks": []}))
    runtime.rebuild_notebook_inputs("01_RQ1_A_BTC_data_labels_baseline.ipynb", tmp_path)
    assert not recomputed_handoffs()
