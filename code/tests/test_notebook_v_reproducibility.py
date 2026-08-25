from __future__ import annotations

from pathlib import Path

from experiments import run_event_window_direction_head as runner


def _audit_module():
    from experiments import audit_notebook_v_reproducibility as audit

    return audit


def test_notebook_v_transitive_local_imports_are_source_hashed():
    audit = _audit_module()
    closure = audit.local_import_closure(
        (Path(runner.__file__).resolve(),), code_root=runner.CODE_ROOT
    )
    registered = {path.resolve() for path in runner._SOURCE_DEPENDENCIES}

    assert closure.issubset(registered)


def test_notebook_v_artifact_reproduction_contract_is_git_tracked():
    audit = _audit_module()
    report = audit.audit_repository(code_root=runner.CODE_ROOT, verify_inputs=False)

    assert report["artifact_status"] == "REPRODUCIBLE_ARTIFACT"
    assert report["missing_paths"] == []
    assert report["untracked_paths"] == []
    assert report["published_run_hash"] == "e000a2e332da35621928"
    assert report["source_hash_matches"] is False
    assert report["status"] == "REPRODUCIBLE_ARTIFACT_SOURCE_DRIFT"
    assert report["exact_rerun_required"] is True
    assert report["artifact_hashes_valid"] is True
    assert report["frozen_u_valid"] is True
    assert report["frozen_j_valid"] is True


def test_notebook_v_published_artifact_stays_verifiable_after_source_evolves():
    audit = _audit_module()
    report = audit.audit_repository(code_root=runner.CODE_ROOT, verify_inputs=False)

    assert report["artifact_status"] == "REPRODUCIBLE_ARTIFACT"
    if report["source_hash_matches"]:
        assert report["status"] == "REPRODUCIBLE_ARTIFACT"
    else:
        assert report["status"] == "REPRODUCIBLE_ARTIFACT_SOURCE_DRIFT"
        assert report["exact_rerun_required"] is True


def test_notebook_v_environment_lock_is_exact():
    audit = _audit_module()
    locked = audit.read_environment_lock(runner.CODE_ROOT / "requirements-v-repro.txt")

    assert locked == {
        "catboost": "1.2.10",
        "ipykernel": "7.3.0",
        "matplotlib": "3.11.0",
        "nbclient": "0.11.0",
        "nbformat": "5.10.4",
        "numpy": "2.4.6",
        "pandas": "3.0.3",
        "pyarrow": "24.0.0",
        "pyyaml": "6.0.3",
        "scikit-learn": "1.9.0",
        "scipy": "1.17.1",
        "torch": "2.13.0",
        "xgboost": "3.2.0",
    }
