import json
from pathlib import Path

import pytest
import yaml

from reflection_agent.config import ProtocolConfig, load_config
from reflection_agent.manifest import build_manifest, write_manifest

CODE_ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_config_is_valid_and_q2_is_sealed():
    config = load_config(CODE_ROOT / "configs" / "reflection_agent.yaml")
    assert config.model == "glm-5.2:cloud"
    assert config.development_end_utc <= config.sealed_start_utc
    assert config.candidate_budget == 6


def test_config_rejects_model_or_grid_drift():
    payload = yaml.safe_load((CODE_ROOT / "configs" / "reflection_agent.yaml").read_text(encoding="utf-8"))
    payload["model"] = "glm-5:cloud"
    with pytest.raises(ValueError):
        ProtocolConfig.model_validate(payload)
    payload["model"] = "glm-5.2:cloud"
    payload["confidence_grid"] = [0.73]
    with pytest.raises(ValueError, match="grid changed"):
        ProtocolConfig.model_validate(payload)


def test_manifest_is_stable_and_refuses_different_overwrite(tmp_path):
    config = load_config(CODE_ROOT / "configs" / "reflection_agent.yaml")
    source = tmp_path / "input.txt"
    source.write_text("frozen", encoding="utf-8")
    first = build_manifest(config, code_root=CODE_ROOT, input_paths=[source], output_mode="schema")
    second = build_manifest(config, code_root=CODE_ROOT, input_paths=[source], output_mode="schema")
    assert first == second
    path = write_manifest(tmp_path / "manifest.json", first)
    assert json.loads(path.read_text(encoding="utf-8"))["protocol_hash"] == first["protocol_hash"]
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_manifest(path, first | {"protocol_hash": "changed"})

