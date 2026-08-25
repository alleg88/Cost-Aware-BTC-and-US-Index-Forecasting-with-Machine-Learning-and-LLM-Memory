from pathlib import Path

import nbformat


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "01_data_labels_and_baseline.ipynb"
NOTEBOOKS_DIR = NOTEBOOK.parent


def test_notebook_01_writes_direct_fixed_180d_handoff_without_lookback_study():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in nb.cells)

    assert "write_notebook01_handoff" in source
    assert "write_pipeline_handoff" in source
    assert "PIPELINE_HANDOFF" in source
    assert "Notebook 02a" not in source
    assert "## Why 90 days - DZ40 later-pipeline reference" not in source
    assert "btc_balanced_dz40_lookback_eval.parquet" not in source
    assert 'sentiment="both"' not in source


def test_notebook_01_places_frozen_width_handoff_immediately_after_positioning():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    sources = [cell.source for cell in nb.cells]
    positioning = next(i for i, source in enumerate(sources) if "pos = pd.read_parquet" in source)
    selection = next(i for i, source in enumerate(sources) if "selected_widths =" in source)
    leakage = next(i for i, source in enumerate(sources) if "y_shuffled =" in source)
    selection_block = "\n".join(sources[selection:leakage])

    assert positioning < selection < leakage
    assert "selected_widths = posr.loc[sorted(WIDTHS, reverse=True)].copy()" in selection_block
    assert 'posr.sort_values("sortino_pos", ascending=False).head(3)' not in selection_block
    assert "All three remain net-negative" not in selection_block

def test_notebook_01_positioning_table_reports_sortino_and_sharpe_for_all_widths():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in nb.cells)

    assert '"sortino_base": nb["sortino"]' in source
    assert '"sortino_pos": np_["sortino"]' in source
    assert '"sharpe_base": nb["sharpe"]' in source
    assert '"sharpe_pos": np_["sharpe"]' in source
    assert '"Sortino (+pos)"' in source
    assert '"Sharpe (+pos)"' in source


def test_notebook_01_labels_catboost_dead_zones_and_omits_redundant_diagnostics():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in nb.cells)

    assert "THRESHOLD_BPS = 40" in source
    assert "THRS = [25, 30, 35, 40, 45, 50, 55, 60, 65, 75]" in source
    assert "Confusion matrix" not in source
    assert "ax.bar(CLASS_NAMES" not in source
    assert "Class distribution @" not in source


def test_notebook_01_positions_paired_ablation_before_leakage_and_explains_sample_change():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    sources = [cell.source for cell in nb.cells]
    positioning = next(i for i, source in enumerate(sources) if "pos = pd.read_parquet" in source)
    leakage = next(i for i, source in enumerate(sources) if "y_shuffled =" in source)
    joined = "\n".join(sources)

    assert positioning < leakage
    assert "BlockingTimeSeriesSplit" in joined


def test_active_notebook_series_excludes_one_second_execution_audit():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in nb.cells).lower()
    readme = (NOTEBOOKS_DIR / "README.md").read_text(encoding="utf-8").lower()

    assert not (NOTEBOOKS_DIR / "02c_catboost_execution_resolution.ipynb").exists()
    assert "02c_catboost_execution_resolution" not in readme
    assert "one-second" not in source
    assert "1-second" not in source
    assert "btcusdt_1s" not in source


def test_active_sentiment_chain_is_sequential_and_old_notebook_04_is_removed():
    readme = (NOTEBOOKS_DIR / "README.md").read_text(encoding="utf-8")
    chain = [
        "02b_catboost_economic_optuna.ipynb",
        "02c_sentiment_data_and_methodology.ipynb",
        "02d_all_model_sentiment.ipynb",
        "02e_all_model_sentiment_policy.ipynb",
        "03_all_model_stacking.ipynb",
        "03a_stacking_forward.ipynb",
    ]
    positions = [readme.index(name) for name in chain]
    assert positions == sorted(positions)
    assert all((NOTEBOOKS_DIR / name).exists() for name in chain)
    assert not (NOTEBOOKS_DIR / "02c_all_model_sentiment.ipynb").exists()
    assert not (NOTEBOOKS_DIR / "04_sentiment_ablation.ipynb").exists()
