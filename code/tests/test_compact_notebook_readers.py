"""Reader contracts follow recomputed upstream identities, not old row counts."""
import ast
import json
from pathlib import Path

import pytest

from experiments.channel_rebuild_contract import MODE_ENV, handoff_run_hash, recomputed_handoffs


NOTEBOOKS = Path(__file__).parents[1] / "notebooks"


def _assignment(notebook_name, target, namespace):
    notebook = json.loads((NOTEBOOKS / notebook_name).read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][2]["source"])
    tree = ast.parse(source)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and any(isinstance(name, ast.Name) and name.id == target for name in node.targets)]
    assert len(matches) == 1
    exec(compile(ast.Module(body=matches, type_ignores=[]), notebook_name, "exec"), namespace)
    return namespace[target]


@pytest.mark.parametrize("compact", (False, True))
def test_u_reader_selects_correct_p_r_identities(tmp_path, monkeypatch, compact):
    monkeypatch.setenv(MODE_ENV, "recomputed" if compact else "")
    for target, folder, reference in (
        ("FROZEN_P_RUN_HASH", "event_window_conditional_opportunity", "0474798f6d0eb56e64d3"),
        ("FROZEN_R_RUN_HASH", "event_window_timing_policy_repair", "f8de389583c836f48140"),
    ):
        root = tmp_path / "experiments/cache" / folder
        root.mkdir(parents=True)
        (root / "latest_dev.json").write_text(json.dumps({"run_hash": "a" * 20, "relative_path": f"{'a' * 20}/full"}))
        actual = _assignment("19_RQ5_B_BTC_volatility_feature_consolidation.ipynb", target,
            {"CODE_ROOT": tmp_path, "handoff_run_hash": handoff_run_hash})
        assert actual == ("a" * 20 if compact else reference)


@pytest.mark.parametrize("compact", (False, True))
def test_v_reader_matches_current_source_and_scored_counts(monkeypatch, compact):
    monkeypatch.setenv(MODE_ENV, "recomputed" if compact else "")
    actual = _assignment("20_RQ5_C_BTC_economic_direction_head.ipynb", "required_summary",
        {"protocol": {"expected_source_activations": 7, "expected_scored_activations": 6},
         "recomputed_handoffs": recomputed_handoffs})
    assert actual["source_activations"] == (7 if compact else 3431)
    assert actual["scored_activations"] == (6 if compact else 2939)
    assert actual["warmup_activations"] == (1 if compact else 492)
    assert actual["round_trip_cost_bps"] == 10.0 and actual["forced_direction"] is True


@pytest.mark.parametrize("compact", (False, True))
def test_w_reader_uses_current_matched_fold_total(monkeypatch, compact):
    monkeypatch.setenv(MODE_ENV, "recomputed" if compact else "")
    actual = _assignment("21_RQ5_D_BTC_channel_vs_volatility_ablation.ipynb", "required",
        {"protocol": {"matched_fold_counts": {"2022H2": 2, "2023H1": 3}},
         "recomputed_handoffs": recomputed_handoffs})
    assert actual["matched_total_activations"] == (5 if compact else 2939)
    assert actual["hold_minutes"] == 120 and actual["round_trip_cost_bps"] == 10.0
    assert actual["forward_or_lockbox_loaded"] is False
