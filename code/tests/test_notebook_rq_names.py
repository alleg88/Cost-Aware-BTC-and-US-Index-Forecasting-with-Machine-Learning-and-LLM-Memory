"""Public notebook names identify their primary research question."""
import json
from pathlib import Path
import re

from experiments.notebook_hygiene import NOTEBOOK_SEQUENCES
from experiments.notebook_runtime import notebook_inputs


NOTEBOOK_ROOT = Path(__file__).parents[1] / "notebooks"
EXPECTED_IDS = {
    f"RQ{question}_{chr(65 + index)}"
    for question, count in ((1, 6), (2, 9), (3, 6), (4, 1), (5, 4))
    for index in range(count)
} - {"RQ1_D", "RQ2_D", "RQ2_E", "RQ2_G", "RQ5_A"}


def test_all_22_public_notebooks_have_unique_rq_names_titles_and_setup_bindings():
    paths = sorted(NOTEBOOK_ROOT.glob("*.ipynb"))
    assert len(paths) == 22
    identities = set()
    for path in paths:
        if path.name == "22_Lockbox_Q2_2026.ipynb":
            identity = "Lockbox"
        else:
            match = re.fullmatch(r"\d{2}_(RQ[1-5]_[A-Z])_[A-Za-z0-9_]+\.ipynb", path.name)
            assert match is not None, path.name
            identity = match.group(1)
        assert identity not in identities
        identities.add(identity)
        notebook = json.loads(path.read_text(encoding="utf-8"))
        first_markdown = next(cell for cell in notebook["cells"] if cell["cell_type"] == "markdown")
        assert "".join(first_markdown["source"]).startswith(f"# {path.name[:2]} - {identity}: ")
        first_code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
        setup = "".join(first_code["source"])
        if identity == "RQ4_A":
            assert "loader.prepare(" in setup
        else:
            assert path.name in setup
    assert identities == EXPECTED_IDS | {"Lockbox"}
    assert {name for sequence in NOTEBOOK_SEQUENCES.values() for name in sequence} == {
        path.name for path in paths
    }


def test_renamed_index_llm_reader_retains_llm_specific_input_contract():
    paths = notebook_inputs("16_RQ3_E_indices_LLM_sentiment.ipynb")
    assert "sentiment/raw/scores_llm_usa500.manifest.json" in paths
    assert "sentiment/raw/scores_llm_direct_events_usatech.manifest.json" in paths
    assert "sentiment/raw/index_*_identity.json" in paths
    assert "sentiment/raw/scores_usa500.manifest.json" not in paths


def test_renamed_index_deberta_reader_retains_non_llm_input_contract():
    paths = notebook_inputs("15_RQ3_D_indices_DeBERTa_sentiment.ipynb")
    assert "sentiment/raw/scores_usa500.manifest.json" in paths
    assert "sentiment/raw/scores_direct_events_usatech.manifest.json" in paths
    assert not any("scores_llm_" in path or "index_*_identity" in path for path in paths)


def test_notebook_instructions_link_to_every_numbered_file_in_order():
    readme = (NOTEBOOK_ROOT / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\(([^()]+\.ipynb)\)", readme)
    assert links == sorted(path.name for path in NOTEBOOK_ROOT.glob("*.ipynb"))
