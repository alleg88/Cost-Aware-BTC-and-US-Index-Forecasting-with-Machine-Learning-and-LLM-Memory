"""Public numbering, launch bindings and the default run order stay aligned."""
import json
from pathlib import Path
import re

from experiments.notebook_hygiene import execution_order
from experiments.notebook_runtime import notebook_inputs


NOTEBOOKS = Path(__file__).parents[1] / "notebooks"


def test_numbered_catalog_is_contiguous_and_preserves_research_questions():
    names = sorted(path.name for path in NOTEBOOKS.glob("*.ipynb"))
    assert [name[:3] for name in names] == [f"{number:02d}_" for number in range(1, 23)]
    assert [name.split("_", 2)[1] for name in names] == (
        ["RQ1"] * 5 + ["RQ2"] * 6 + ["RQ3"] * 6 + ["RQ4"] + ["RQ5"] * 3 + ["Lockbox"]
    )


def test_default_run_order_matches_the_visible_numbered_catalog():
    assert execution_order() == tuple(sorted(path.name for path in NOTEBOOKS.glob("*.ipynb")))


def test_input_instructions_use_the_same_numbered_order():
    instructions = (NOTEBOOKS / "DATA.md").read_text(encoding="utf-8")
    names = re.findall(r"^\*\*([^*\n]+\.ipynb)\*\*$", instructions, flags=re.MULTILINE)
    assert names == sorted(path.name for path in NOTEBOOKS.glob("*.ipynb"))


def test_numbered_setup_bindings_resolve_their_input_contracts():
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        first_code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
        setup = "".join(first_code["source"])
        if path.name == "18_RQ4_A_BTC_LLM_policy_router.ipynb":
            publication = json.loads(
                (NOTEBOOKS.parent / "configs/rq4_colab_publication.json").read_text(encoding="utf-8")
            )
            assert "loader.prepare(" in setup
            assert all(value in setup for value in publication.values())
            continue
        assert path.name in setup
        assert notebook_inputs(path.name)
