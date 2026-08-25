import json
from pathlib import Path
import sys

import nbformat
from jupyter_client.kernelspec import KernelSpecManager

from experiments.notebook_hygiene import (
    BASE_COLAB_DEPENDENCIES,
    NOTEBOOK_SEQUENCES,
    canonical_colab_setup,
    current_python_kernel,
    execute_notebook,
    execution_order,
    execute_all,
    normalize_notebook,
)


CODE_ROOT = Path(__file__).parents[1]
NOTEBOOK_ROOT = CODE_ROOT / "notebooks"

EXPECTED_SEQUENCES = {
    "Bitcoin": (
        "01_data_labels_and_baseline.ipynb",
        "01b_positioning_ablation.ipynb",
        "02b_catboost_economic_optuna.ipynb",
        "02c_sentiment_data_and_methodology.ipynb",
        "02d_all_model_sentiment.ipynb",
        "02e_all_model_sentiment_policy.ipynb",
        "03_all_model_stacking.ipynb",
        "03a_stacking_forward.ipynb",
        "03c_qualified_union_ensemble.ipynb",
        "04a_svm_temperature_calibration.ipynb",
        "04b_xgboost_strong_move_admission.ipynb",
        "04d_unified_2021_ensemble.ipynb",
        "04g_lstm_gmadl_shadow.ipynb",
        "04h_union_v1_episode_reentry.ipynb",
        "05c_causal_policy_router_agent.ipynb",
    ),
    "Indices": (
        "06a_index_nine_models.ipynb",
        "06b_index_vix.ipynb",
        "06c_index_deberta.ipynb",
        "06d_index_llm.ipynb",
        "06g_index_all_model_ensemble.ipynb",
        "06i_index_comparison.ipynb",
    ),
    "Channels": (
        "A_channel_strategy.ipynb",
        "U_volatility_timing_feature_consolidation.ipynb",
        "V_economic_direction_head.ipynb",
        "W_channel_vs_volatility_ablation.ipynb",
    ),
    "Final confirmation": (
        "07_final_q2_lockbox.ipynb",
        "07a_q2_sentiment_sensitivity.ipynb",
    ),
}


def test_canonical_sequences_are_explicit_disjoint_and_exhaustive():
    assert NOTEBOOK_SEQUENCES == EXPECTED_SEQUENCES
    registered = [name for sequence in NOTEBOOK_SEQUENCES.values() for name in sequence]
    assert len(registered) == len(set(registered)) == 27
    actual = {
        path.name
        for path in NOTEBOOK_ROOT.glob("*.ipynb")
        if path.name != "00_run_in_colab.ipynb"
    }
    assert actual == set(registered)


def test_reader_ledger_names_every_registered_notebook():
    readme = (NOTEBOOK_ROOT / "README.md").read_text(encoding="utf-8")
    missing = [
        name
        for sequence in NOTEBOOK_SEQUENCES.values()
        for name in sequence
        if f"`{name}`" not in readme
    ]
    assert missing == []


def test_reader_ledger_presents_bitcoin_then_indices_then_channels():
    readme = (NOTEBOOK_ROOT / "README.md").read_text(encoding="utf-8")
    headings = [readme.index(f"## {name}") for name in EXPECTED_SEQUENCES]
    assert headings == sorted(headings)
    for index, (sequence_name, notebooks) in enumerate(EXPECTED_SEQUENCES.items()):
        start = headings[index]
        end = headings[index + 1] if index + 1 < len(headings) else len(readme)
        positions = [readme.index(f"`{name}`") for name in notebooks]
        assert positions == sorted(positions)
        assert all(start < position < end for position in positions)


def test_normalizer_replaces_one_old_setup_and_preserves_reader_cells(tmp_path):
    path = tmp_path / "reader.ipynb"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Reader title"),
            nbformat.v4.new_code_cell(
                "# Portable setup and completed-run reader\n"
                "from matplotlib import pyplot as plt\n"
                "CODE_ROOT = 'reader-owned'\n"
                "reader_state = 42\n"
            ),
            nbformat.v4.new_code_cell(
                "# >>> Set this to your code/ folder path\nCODE_ROOT = 'old-bootstrap'\n"
            ),
            nbformat.v4.new_markdown_cell("## Method"),
            nbformat.v4.new_code_cell("answer = 42"),
        ]
    )
    nbformat.write(notebook, path)

    assert normalize_notebook(path, extra_dependencies=("xgboost==3.2.0",)) is True
    normalized = nbformat.read(path, as_version=4)
    assert normalized.cells[0].cell_type == "code"
    assert normalized.cells[0].source == canonical_colab_setup(("xgboost==3.2.0",))
    assert normalized.cells[1].source == "# Reader title"
    assert "reader_state = 42" in normalized.cells[2].source
    assert [cell.source for cell in normalized.cells].count("## Method") == 1
    assert [cell.source for cell in normalized.cells].count("answer = 42") == 1
    assert "old-bootstrap" not in "\n".join(cell.source for cell in normalized.cells)
    for dependency in (*BASE_COLAB_DEPENDENCIES, "xgboost==3.2.0"):
        assert dependency in normalized.cells[0].source

    first_bytes = path.read_bytes()
    assert normalize_notebook(path, extra_dependencies=("xgboost==3.2.0",)) is False
    assert path.read_bytes() == first_bytes


def test_notebook_json_is_valid_after_normalization(tmp_path):
    path = tmp_path / "reader.ipynb"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Reader title"),
            nbformat.v4.new_code_cell("value = 1"),
        ]
    )
    nbformat.write(notebook, path)
    normalize_notebook(path)
    json.loads(path.read_text(encoding="utf-8"))
    nbformat.validate(nbformat.read(path, as_version=4))


def test_execution_order_flattens_all_reader_sequences():
    expected = tuple(
        name for sequence in EXPECTED_SEQUENCES.values() for name in sequence
    )
    assert execution_order() == expected
    assert execution_order(("Indices", "Channels")) == (
        *EXPECTED_SEQUENCES["Indices"],
        *EXPECTED_SEQUENCES["Channels"],
    )


def test_execute_notebook_runs_top_to_bottom_and_persists_outputs(tmp_path):
    path = tmp_path / "reader.ipynb"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("value = 40"),
            nbformat.v4.new_code_cell("print(value + 2)"),
        ],
        metadata={"kernelspec": {"display_name": "msc-code", "language": "python", "name": "msc-code"}},
    )
    nbformat.write(notebook, path)

    execute_notebook(path, working_directory=CODE_ROOT, timeout=60)

    executed = nbformat.read(path, as_version=4)
    assert all(cell.execution_count is not None for cell in executed.cells)
    assert executed.cells[1].outputs[0].text.strip() == "42"


def test_current_python_kernel_is_temporary_and_uses_active_interpreter():
    with current_python_kernel() as kernel_name:
        spec = KernelSpecManager().get_kernel_spec(kernel_name)
        assert Path(spec.argv[0]).resolve() == Path(sys.executable).resolve()

    assert kernel_name not in KernelSpecManager().find_kernel_specs()


def test_background_notebook_ignores_a_host_specific_kernel_class(tmp_path, monkeypatch):
    profile = tmp_path / "ipython" / "profile_default"
    profile.mkdir(parents=True)
    (profile / "ipython_kernel_config.py").write_text(
        "c = get_config()\nc.IPKernelApp.kernel_class = 'missing_vendor_kernel.Kernel'\n"
        "c.InteractiveShellApp.extensions = ['missing_vendor_extension']\n"
        "c.InteractiveShellApp.reraise_ipython_extension_failures = True\n"
    )
    monkeypatch.setenv("IPYTHONDIR", str(profile.parent))
    path = tmp_path / "reader.ipynb"
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell("import sys\nprint(sys.executable)"),
    ])
    nbformat.write(notebook, path)
    execute_notebook(path, working_directory=CODE_ROOT, timeout=30)
    result = nbformat.read(path, as_version=4)
    assert Path(result.cells[0].outputs[0].text.strip()).resolve() == Path(sys.executable).resolve()


def test_execute_all_rejects_a_resume_point_outside_the_selected_sequence(tmp_path):
    try:
        execute_all(
            ("Indices",),
            notebook_root=NOTEBOOK_ROOT,
            start_at="A_channel_strategy.ipynb",
        )
    except ValueError as error:
        assert "outside the selected sequence" in str(error)
    else:
        raise AssertionError("an unrelated resume point must be rejected")
