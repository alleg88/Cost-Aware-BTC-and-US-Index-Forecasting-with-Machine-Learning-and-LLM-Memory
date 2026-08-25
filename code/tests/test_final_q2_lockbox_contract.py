from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from experiments.final_q2_lockbox_contract import (
    CostSpec,
    canonical_hash,
    load_lockbox_protocol,
)
from experiments.final_q2_lockbox_preflight import (
    REPOSITORY_ROOT,
    critical_source_registry,
    q2_source_registry,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = CODE_ROOT / "configs" / "final_q2_lockbox_protocol.json"
MANIFEST_PATH = CODE_ROOT / "configs" / "final_q2_lockbox_manifest.json"

EXPECTED_CANDIDATES = [
    "btc_qualified_union_v1",
    "btc_lstm_dz55",
    "usa500_best_single_deberta_svm",
    "usa500_deberta_soft_vote",
    "usatech_best_single_deepseek_lstm",
    "usatech_deepseek_soft_vote",
]


def test_protocol_contains_only_six_frozen_candidates() -> None:
    protocol = load_lockbox_protocol(PROTOCOL_PATH)

    assert [item.candidate_id for item in protocol.candidates] == EXPECTED_CANDIDATES
    assert protocol.primary_estimand == "btc_union_minus_lstm_total_net"
    assert not any("agent" in item.candidate_id for item in protocol.candidates)
    assert [item.role for item in protocol.candidates] == [
        "primary",
        "control",
        "primary",
        "secondary",
        "primary",
        "secondary",
    ]


def test_costs_and_half_open_interval_are_frozen() -> None:
    protocol = load_lockbox_protocol(PROTOCOL_PATH)

    assert protocol.start_utc.isoformat() == "2026-04-01T00:00:00+00:00"
    assert protocol.end_utc.isoformat() == "2026-07-01T00:00:00+00:00"
    assert protocol.costs["btcusdt"] == CostSpec(5.0, 10.0)
    assert protocol.costs["usa500"] == CostSpec(1.0, 2.0)
    assert protocol.costs["usatech"] == CostSpec(1.5, 3.0)
    assert protocol.bootstrap_replicates == 5_000
    assert protocol.bootstrap_seed == 42


def test_btc_lstm_has_the_only_preregistered_reconstruction_exception() -> None:
    protocol = load_lockbox_protocol(PROTOCOL_PATH)

    assert protocol.default_reconstruction_tolerance.max_abs_probability_error == 1e-6
    assert set(protocol.reconstruction_exceptions) == {"btcusdt:none:lstm:w55"}
    exception = protocol.reconstruction_exceptions["btcusdt:none:lstm:w55"]
    assert exception.max_abs_probability_error == 2e-5
    assert exception.mean_abs_probability_error == 5e-7
    assert exception.torch_threads == 16
    assert exception.require_exact_classes is True
    assert exception.require_exact_decisions is True


def test_prelockbox_manifest_binds_every_estimator_input_and_source() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=CODE_ROOT.parent, text=True
    ).strip()
    try:
        parent = subprocess.check_output(
            ["git", "rev-parse", "HEAD^"],
            cwd=CODE_ROOT.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        parent = None
    critical_paths = [
        path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
        for path in critical_source_registry().values()
    ]
    critical_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *critical_paths],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).strip()
    if manifest["implementation_commit"] not in {head, parent} or critical_status:
        pytest.skip("tracked pre-lockbox manifest awaits regeneration at current HEAD")

    assert manifest["q2_decoded"] is False
    assert len(manifest["reconstructed_estimators"]) == 21
    assert len({row["fit_key"] for row in manifest["reconstructed_estimators"]}) == 21
    assert set(manifest["q2_sources"]) == set(q2_source_registry(CODE_ROOT))
    assert len(manifest["warmups"]) == 29
    assert [row["candidate_id"] for row in manifest["candidates"]] == list(
        EXPECTED_CANDIDATES
    )
    assert manifest["model_identities"]["deepseek"]["batch_size"] == 10
    assert manifest["model_identities"]["deepseek"]["live_verified"] is True
    assert manifest["model_identities"]["deberta"]["revision"] == (
        "9e10915c245a80a89b18d1ac51350e093c7bb35a"
    )
    provenance = json.loads(
        Path(manifest["warmups"]["warmup_provenance"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert provenance["q2_decoded"] is False
    assert set(provenance["artifacts"]) == set(manifest["warmups"]).difference(
        {"warmup_provenance"}
    )
    assert all(entry["sources"] for entry in provenance["artifacts"].values())
    assert all(
        entry["destination"]["sha256"]
        == manifest["warmups"][key]["sha256"]
        for key, entry in provenance["artifacts"].items()
    )
    assert manifest["protocol"]["protocol_hash"] == canonical_hash(
        json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    )


def test_union_has_two_distinct_frozen_members() -> None:
    union = load_lockbox_protocol(PROTOCOL_PATH).candidates[0]

    assert union.combiner == "opposite_signal_veto"
    assert union.decision_tau == 0.0
    assert [(member.model_name, member.width_bps, member.signal_tau) for member in union.members] == [
        ("lstm", 55, 0.75),
        ("svm_linear", 75, 0.0),
    ]


def test_index_candidate_widths_match_the_frozen_h1_selection() -> None:
    candidates = {
        item.candidate_id: item for item in load_lockbox_protocol(PROTOCOL_PATH).candidates
    }

    usa_single = candidates["usa500_best_single_deberta_svm"]
    assert [(member.model_name, member.width_bps) for member in usa_single.members] == [
        ("svm_linear", 15)
    ]
    assert usa_single.decision_tau == 0.4
    assert len(candidates["usa500_deberta_soft_vote"].members) == 9
    assert {member.width_bps for member in candidates["usa500_deberta_soft_vote"].members} == {15}
    assert candidates["usa500_deberta_soft_vote"].decision_tau == 0.55
    tech_single = candidates["usatech_best_single_deepseek_lstm"]
    assert [(member.model_name, member.width_bps) for member in tech_single.members] == [
        ("lstm", 15)
    ]
    assert tech_single.decision_tau == 0.65
    assert len(candidates["usatech_deepseek_soft_vote"].members) == 9
    assert {member.width_bps for member in candidates["usatech_deepseek_soft_vote"].members} == {5}
    assert candidates["usatech_deepseek_soft_vote"].decision_tau == 0.55


def test_every_candidate_binds_all_source_artifacts() -> None:
    protocol = load_lockbox_protocol(PROTOCOL_PATH)

    assert all(candidate.artifacts for candidate in protocol.candidates)
    for candidate in protocol.candidates:
        assert len({artifact.path for artifact in candidate.artifacts}) == len(
            candidate.artifacts
        )
        assert all(len(artifact.sha256) == 64 for artifact in candidate.artifacts)
        assert all(artifact.role for artifact in candidate.artifacts)


def test_protocol_rejects_an_unregistered_candidate(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["candidates"].append(
        {
            "stream": "btcusdt",
            "candidate_id": "reflection_agent",
            "role": "primary",
            "arm": "none",
            "decision_tau": 0.75,
            "members": [{"model_name": "lstm", "width_bps": 55, "signal_tau": None}],
            "combiner": None,
            "artifacts": payload["candidates"][1]["artifacts"],
        }
    )
    path = tmp_path / "bad_protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="candidate"):
        load_lockbox_protocol(path)


def test_protocol_dataclasses_are_json_safe() -> None:
    protocol = load_lockbox_protocol(PROTOCOL_PATH)
    payload = asdict(protocol)

    assert payload["protocol_version"] == "final-q2-lockbox-v1"
    assert len(payload["candidates"]) == 6
