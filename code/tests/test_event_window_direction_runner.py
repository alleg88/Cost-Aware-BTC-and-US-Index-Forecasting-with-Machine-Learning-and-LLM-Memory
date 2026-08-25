from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments import run_event_window_direction_head as runner


def test_protocol_freezes_uniform_ten_bps_and_forced_choice():
    protocol = runner.protocol_dict(runner.DirectionHeadConfig())

    assert protocol["entry_cost_bps"] == 5.0
    assert protocol["target_exit_cost_bps"] == 5.0
    assert protocol["other_exit_cost_bps"] == 5.0
    assert protocol["round_trip_cost_bps"] == 10.0
    assert protocol["target_multiple_b"] == 2.0
    assert protocol["hold_minutes"] == 120
    assert protocol["forced_direction"] is True
    assert protocol["trade_gate_included"] is False
    assert protocol["timing_model_refit"] is False
    raw_identity = protocol["raw_input_identity_contract"]
    assert raw_identity == {
        "method": "pyarrow_filtered_canonical_ipc_sha256_v1",
        "development_start": "2021-01-01T00:00:00+00:00",
        "development_end_exclusive": "2025-07-01T00:00:00+00:00",
        "source_roles": ["minute", "five_minute", "hourly", "positioning"],
        "pyarrow_filter_before_pandas": True,
        "whole_file_bytes_hashed": False,
        "file_metadata_hashed": False,
        "absolute_paths_hashed": False,
        "applies_to_full_run": True,
    }


def test_non_development_stage_is_rejected_before_frozen_inputs_are_loaded(monkeypatch):
    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("frozen inputs were loaded before stage validation")

    monkeypatch.setattr(runner, "load_frozen_u_artifacts", fail_if_loaded)

    with pytest.raises(ValueError, match="development only"):
        runner.run_direction_head(stage="forward", smoke=True)


@pytest.mark.parametrize(
    "config",
    [
        replace(runner.DirectionHeadConfig(), bootstrap_draws=1_999),
        replace(runner.DirectionHeadConfig(), bootstrap_seed=43),
        replace(runner.DirectionHeadConfig(), minimum_path_completeness=0.991),
        *[
            replace(
                runner.DirectionHeadConfig(),
                model=replace(runner.DirectionHeadConfig().model, **{name: value}),
            )
            for name, value in (
                ("xgb_estimators", 299),
                ("xgb_depth", 4),
                ("xgb_learning_rate", 0.04),
                ("xgb_min_child_weight", 19.0),
                ("xgb_reg_lambda", 9.0),
                ("logreg_c", 2.0),
                ("logreg_max_iter", 1_999),
                ("random_seed", 43),
                ("n_jobs", 2),
            )
        ],
    ],
)
def test_complete_registered_config_is_frozen_before_loading(config):
    with pytest.raises(ValueError, match="frozen protocol changed"):
        runner._validate_config(config)


def _partial_frozen_u_root(tmp_path: Path) -> tuple[Path, Path]:
    source_root = runner.FROZEN_U_ROOT
    source_run = source_root / runner.FROZEN_U_RUN_HASH / "full"
    root = tmp_path / "frozen-u"
    run_dir = root / runner.FROZEN_U_RUN_HASH / "full"
    run_dir.mkdir(parents=True)
    shutil.copy2(source_root / "latest_dev.json", root / "latest_dev.json")
    shutil.copy2(source_run / "run_state.json", run_dir / "run_state.json")
    shutil.copy2(source_run / "protocol.json", run_dir / "protocol.json")
    return root, run_dir


def test_frozen_u_rejects_a_self_consistently_replaced_manifest(tmp_path):
    root, run_dir = _partial_frozen_u_root(tmp_path)
    state_path = run_dir / "run_state.json"
    state = json.loads(state_path.read_text("utf-8"))
    state["artifacts"]["activation_ledger.parquet"]["sha256"] = "0" * 64
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), "utf-8")

    with pytest.raises(ValueError, match="manifest digest changed"):
        runner.load_frozen_u_artifacts(root)


def test_frozen_u_recomputes_protocol_identity_before_loading_artifacts(tmp_path):
    root, run_dir = _partial_frozen_u_root(tmp_path)
    protocol_path = run_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text("utf-8"))
    protocol["target_exit_cost_bps"] = 99.0
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True), "utf-8")

    with pytest.raises(ValueError, match="protocol hash changed"):
        runner.load_frozen_u_artifacts(root)


def test_frozen_u_recomputes_run_identity_from_protocol_source_and_input_hashes():
    run_dir = runner.FROZEN_U_ROOT / runner.FROZEN_U_RUN_HASH / "full"
    pointer = json.loads(
        (runner.FROZEN_U_ROOT / "latest_dev.json").read_text("utf-8")
    )
    state = json.loads((run_dir / "run_state.json").read_text("utf-8"))
    protocol = json.loads((run_dir / "protocol.json").read_text("utf-8"))
    state["source_hash"] = "1" * 64
    protocol["source_hash"] = "1" * 64

    with pytest.raises(ValueError, match="run identity changed"):
        runner._validate_frozen_u_identity(
            pointer=pointer,
            state=state,
            protocol=protocol,
            manifest_sha256=runner.FROZEN_U_MANIFEST_SHA256,
        )


@pytest.mark.parametrize("target", runner._SOURCE_DEPENDENCIES)
def test_source_identity_changes_when_an_executed_dependency_changes(
    monkeypatch, target: Path
):
    target = target.resolve()
    original_read_text = Path.read_text
    expected = runner._source_hash()

    def changed_read_text(path: Path, *args, **kwargs) -> str:
        payload = original_read_text(path, *args, **kwargs)
        return payload + "reviewed-dependency-change" if path.resolve() == target else payload

    monkeypatch.setattr(Path, "read_text", changed_read_text)

    assert runner._source_hash() != expected


def test_source_identity_is_stable_across_git_line_endings(monkeypatch):
    original_read_text = Path.read_text
    expected = runner._source_hash()

    def crlf_read_text(path: Path, *args, **kwargs) -> str:
        payload = original_read_text(path, *args, **kwargs)
        return payload.replace("\n", "\r\n")

    monkeypatch.setattr(Path, "read_text", crlf_read_text)

    assert runner._source_hash() == expected


def test_threshold_csv_comparison_rejects_a_tolerated_last_digit_change():
    ledger = pd.DataFrame(
        {"fold_id": ["2022H1"], "threshold": [0.2571410508114367]}
    )
    exact = pd.DataFrame(
        {"fold_id": ["2022H1"], "threshold": ["0.2571410508114367"]}
    )
    changed = exact.assign(threshold="0.2571410508114368")

    assert runner._thresholds_match_canonical_csv(ledger, exact) is True
    assert runner._thresholds_match_canonical_csv(ledger, changed) is False


_RAW_FILES = {
    "minute": "btcusdt_1m_2021_2026.parquet",
    "five_minute": "btcusdt_5min_2021_2026.parquet",
    "hourly": "btcusdt_1h_2021_2026.parquet",
    "positioning": "btcusdt_positioning_15min_2021_2026.parquet",
}


def _write_raw_sources(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    index = pd.DatetimeIndex(
        ["2024-01-01T00:00:00Z", "2025-06-30T23:00:00Z"],
        name="timestamp",
    )
    paths: dict[str, Path] = {}
    for position, (role, filename) in enumerate(_RAW_FILES.items(), start=1):
        path = root / filename
        pd.DataFrame(
            {
                "value": np.array([position, position + 0.25], dtype=np.float64),
                "role_code": np.array([position, position], dtype=np.int64),
            },
            index=index,
        ).to_parquet(path)
        paths[role] = path
    return paths


def _append_future_row(path: Path) -> None:
    frame = pd.read_parquet(path)
    future = pd.DataFrame(
        {
            "value": np.array([99_999.0], dtype=np.float64),
            "role_code": np.array([int(frame["role_code"].iloc[0])], dtype=np.int64),
        },
        index=pd.DatetimeIndex(["2025-07-01T00:00:00Z"], name="timestamp"),
    )
    pd.concat([frame, future]).to_parquet(path)


def _change_development_row(path: Path) -> None:
    frame = pd.read_parquet(path)
    frame.iloc[0, frame.columns.get_loc("value")] += 1_000.0
    frame.to_parquet(path)


def _install_identity_only_full_run(monkeypatch, captured_payloads: list[dict]) -> None:
    from experiments import run_event_window_tail_models as tail_runner

    frozen_u = SimpleNamespace(
        run_hash="u" * 20,
        manifest_sha256="1" * 64,
        activation_ledger_sha256="2" * 64,
        oof_sha256="3" * 64,
        economic_paths_sha256="4" * 64,
    )
    frozen_j = SimpleNamespace(
        run_hash="j" * 20,
        manifest_sha256="5" * 64,
        input_hash="6" * 64,
    )
    monkeypatch.setattr(runner, "load_frozen_u_artifacts", lambda *args, **kwargs: frozen_u)
    monkeypatch.setattr(tail_runner, "load_frozen_j_artifacts", lambda *args, **kwargs: frozen_j)
    monkeypatch.setattr(runner, "_source_hash", lambda: "7" * 64)
    monkeypatch.setattr(
        runner,
        "_validated_completed_summary",
        lambda *args, **kwargs: {"identity_only": True},
    )
    monkeypatch.setattr(runner, "_latest", lambda *args, **kwargs: None)

    real_sha256 = runner._sha256

    def reject_whole_raw_sha(path: Path) -> str:
        if Path(path).name in set(_RAW_FILES.values()):
            raise AssertionError(f"whole raw file was SHA-256 hashed: {Path(path).name}")
        return real_sha256(path)

    monkeypatch.setattr(runner, "_sha256", reject_whole_raw_sha)
    real_sha_payload = runner._sha_payload

    def capture_payload(payload: object) -> str:
        if isinstance(payload, dict):
            captured_payloads.append(json.loads(json.dumps(payload, default=str)))
        return real_sha_payload(payload)

    monkeypatch.setattr(runner, "_sha_payload", capture_payload)


@pytest.mark.parametrize("changed_role", list(_RAW_FILES))
def test_full_input_identity_ignores_future_rows_but_tracks_development_rows(
    tmp_path, monkeypatch, changed_role: str
):
    data_root = tmp_path / "raw"
    paths = _write_raw_sources(data_root)
    captured_payloads: list[dict] = []
    _install_identity_only_full_run(monkeypatch, captured_payloads)

    baseline = runner.run_direction_head(
        data_root=data_root,
        run_root=tmp_path / "runs",
        smoke=False,
    )
    _append_future_row(paths[changed_role])
    future_changed = runner.run_direction_head(
        data_root=data_root,
        run_root=tmp_path / "runs",
        smoke=False,
    )
    _change_development_row(paths[changed_role])
    development_changed = runner.run_direction_head(
        data_root=data_root,
        run_root=tmp_path / "runs",
        smoke=False,
    )

    assert future_changed.run_dir == baseline.run_dir
    assert development_changed.run_dir != baseline.run_dir
    full_payloads = [
        payload
        for payload in captured_payloads
        if payload.get("frozen_j_run_hash") == "j" * 20
    ]
    assert len(full_payloads) == 3
    assert (
        full_payloads[1]["bounded_development_raw_identity"]
        == full_payloads[0]["bounded_development_raw_identity"]
    )
    assert (
        full_payloads[2]["bounded_development_raw_identity"]
        != full_payloads[0]["bounded_development_raw_identity"]
    )
    for payload in full_payloads:
        assert "minute_source_sha256" not in payload
        assert "five_minute_source_sha256" not in payload
        bounded = payload["bounded_development_raw_identity"]
        assert set(bounded["source_fingerprints"]) == set(_RAW_FILES)


def test_bounded_development_fingerprint_excludes_path_and_file_metadata(tmp_path):
    first_root = tmp_path / "first" / "raw"
    second_root = tmp_path / "second" / "raw"
    _write_raw_sources(first_root)
    second_paths = _write_raw_sources(second_root)
    for path in second_paths.values():
        path.touch()

    first = runner._bounded_development_raw_identity(
        first_root, runner.DirectionHeadConfig()
    )
    second = runner._bounded_development_raw_identity(
        second_root, runner.DirectionHeadConfig()
    )

    assert first == second
    serialized = json.dumps(first, sort_keys=True)
    assert str(first_root) not in serialized
    assert str(second_root) not in serialized


def test_economic_viability_requires_every_registered_gate():
    passing = {
        "path_completeness": 0.99,
        "scored_activations": 2939,
        "mean_net_r_ci_low": 0.001,
        "versus_channel_ci_low": 0.001,
        "leakage_passed": True,
    }
    assert runner.economic_viability(**passing) is True
    for name, value in (
        ("path_completeness", 0.989),
        ("scored_activations", 2938),
        ("mean_net_r_ci_low", 0.0),
        ("versus_channel_ci_low", 0.0),
        ("leakage_passed", False),
    ):
        assert runner.economic_viability(**{**passing, name: value}) is False


def test_smoke_run_writes_complete_causal_artifacts(tmp_path):
    result = runner.run_direction_head(run_root=tmp_path, smoke=True)

    assert result.summary["forward_or_lockbox_loaded"] is False
    assert result.summary["direction_feature_count"] == 28
    assert result.summary["source_activations"] == 3431
    assert result.summary["scored_activations"] == 2939
    assert result.summary["research_claim"] == "smoke_only_no_claim"
    assert result.summary["effective_xgb_estimators"] == 12
    assert result.summary["effective_bootstrap_draws"] == 20
    bounded_identity = result.summary["bounded_development_input_identity"]
    assert bounded_identity["applicable"] is False
    assert bounded_identity["passed"] is True
    assert bounded_identity["aggregate_sha256"] is None
    published = {path.name for path in result.run_dir.iterdir()}
    assert set(runner.READER_ARTIFACTS).issubset(published)
    leakage = pd.read_csv(result.run_dir / "leakage_audit.csv")
    assert leakage["passed"].all()
    leakage = leakage.set_index("check")
    assert bool(leakage.loc["bounded development raw-content identity", "passed"])
    assert bool(leakage.loc["forward and Q2 remain sealed", "passed"])
    protocol = json.loads((result.run_dir / "protocol.json").read_text("utf-8"))
    assert protocol["registered_model_config"]["xgb_estimators"] == 300
    assert protocol["registered_bootstrap_draws"] == 2_000
    assert protocol["effective_model_config"]["xgb_estimators"] == 12
    assert protocol["effective_bootstrap_draws"] == 20
    assert protocol["raw_input_identity_contract"]["whole_file_bytes_hashed"] is False
    assert protocol["bounded_development_input_identity"] == bounded_identity
    frozen_protocol = json.loads(
        (result.run_dir / "frozen_protocol.json").read_text("utf-8")
    )
    assert frozen_protocol["bounded_development_input_identity"] == bounded_identity

    combined = pd.read_parquet(result.run_dir / "combined_policy_ledger.parquet")
    assert set(combined["model"]) == {"logreg", "xgboost"}
    assert set(combined["chosen_direction"]) == {"long", "short"}
    timing_columns = [
        "activation_key",
        "fold_id",
        "window_id",
        "channel_episode_id",
        "step",
        "decision_time",
        "threshold",
        "activation_score",
    ]
    frozen = pd.read_parquet(
        runner.FROZEN_U_ROOT
        / runner.FROZEN_U_RUN_HASH
        / "full"
        / "activation_ledger.parquet"
    )
    frozen = frozen.loc[
        frozen["arm"].eq(runner.FROZEN_TIMING_ARM)
        & frozen["fold_id"].isin(runner.SCORED_FOLDS),
        timing_columns,
    ].sort_values("activation_key", kind="stable").reset_index(drop=True)
    for model in runner.MODELS:
        actual = combined.loc[combined["model"].eq(model), timing_columns]
        actual = actual.sort_values("activation_key", kind="stable").reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, frozen, check_exact=True)

    predictions = pd.read_parquet(
        result.run_dir / "oof_direction_predictions.parquet"
    )
    assert predictions.groupby("model")["activation_key"].nunique().eq(2939).all()
    paths = pd.read_parquet(result.run_dir / "economic_paths.parquet")
    primary = paths.loc[paths["geometry"].eq("adaptive_primary")]
    assert set(primary["cost_bps"]) == {10.0}
    assert set(primary["target_multiple_b"]) == {2.0}
    assert set(primary["hold_minutes"]) == {120}
    policy = pd.read_parquet(result.run_dir / "policy_paths.parquet")
    expected_scenarios = {
        "logreg",
        "xgboost",
        "channel_side",
        "always_long",
        "always_short",
        "random_50",
        "oracle",
    }
    assert set(policy["scenario"]) == expected_scenarios
    assert policy.groupby("scenario")["activation_key"].nunique().eq(2939).all()

    state = json.loads((result.run_dir / "run_state.json").read_text("utf-8"))
    assert state["status"] == "complete"
    assert state["summary"] == result.summary
    for name in runner.READER_ARTIFACTS:
        payload = (result.run_dir / name).read_bytes()
        assert state["artifacts"][name]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_runner_preserves_task2_boundary_directions_verbatim(tmp_path, monkeypatch):
    run_direction_oof = runner.run_direction_oof
    expected: dict[tuple[str, str], str] = {}

    def boundary_predictions(dataset, *, config):
        result = run_direction_oof(dataset, config=config)
        predictions = result.predictions.copy()
        logreg_index = predictions.index[
            predictions["model"].eq("logreg")
            & predictions["channel_side"].eq("short")
        ][0]
        predictions.loc[
            logreg_index,
            ["direction_score", "p_long", "chosen_direction"],
        ] = [0.0, 0.5, "long"]
        xgb_index = predictions.index[predictions["model"].eq("xgboost")][0]
        xgb_fallback = str(predictions.loc[xgb_index, "channel_side"])
        predictions.loc[
            xgb_index,
            ["direction_score", "predicted_delta_r", "chosen_direction"],
        ] = [0.0, 0.0, xgb_fallback]
        expected[("logreg", str(predictions.loc[logreg_index, "activation_key"]))] = "long"
        expected[("xgboost", str(predictions.loc[xgb_index, "activation_key"]))] = xgb_fallback
        return replace(result, predictions=predictions)

    monkeypatch.setattr(runner, "run_direction_oof", boundary_predictions)
    result = runner.run_direction_head(run_root=tmp_path, smoke=True)
    published = pd.read_parquet(
        result.run_dir / "oof_direction_predictions.parquet"
    ).set_index(["model", "activation_key"])
    combined = pd.read_parquet(
        result.run_dir / "combined_policy_ledger.parquet"
    ).set_index(["model", "activation_key"])

    for key, chosen_direction in expected.items():
        assert published.loc[key, "chosen_direction"] == chosen_direction
        assert combined.loc[key, "chosen_direction"] == chosen_direction


def test_default_runner_paths_stay_inside_code_tree():
    assert runner.CODE_ROOT in Path(runner.RUN_ROOT).parents
    assert runner.CODE_ROOT in Path(runner.FROZEN_U_ROOT).parents
