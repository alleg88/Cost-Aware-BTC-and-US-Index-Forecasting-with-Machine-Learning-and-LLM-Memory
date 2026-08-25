"""Development-only Notebook V economic direction-head runner."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.event_window_direction_dataset import (
    DIRECTION_FEATURES,
    DirectionDataset,
    build_direction_dataset,
    pair_direction_paths,
)
from experiments.event_window_direction_models import DirectionModelConfig
from experiments.event_window_direction_oof import (
    SCORED_FOLDS,
    run_direction_oof,
)
from experiments.event_window_large_move_dataset import LargeMoveDecisionDataset
from experiments.run_event_window_cost_aware_entry import _Store, _sha256, _sha_payload
from experiments.run_event_window_economic_feasibility import replay_brackets
CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = CODE_ROOT / "data"
FROZEN_U_ROOT = (
    CODE_ROOT / "experiments" / "cache" / "event_window_feature_consolidation"
)
FROZEN_U_RUN_HASH = "ad36396ae54e17571e81"
FROZEN_U_PROTOCOL_HASH = (
    "9791d63148e6249045897865323b480c2770fbe4d049b8cd3c07d3b874f29f1d"
)
FROZEN_U_MANIFEST_SHA256 = (
    "137ca48502422a3ac30ff07c54d429517364a39fb872622bfeca3d0104979897"
)
FROZEN_TIMING_ARM = "xgboost_base"
FROZEN_U_ARTIFACTS = (
    "protocol.json",
    "summary.json",
    "activation_ledger.parquet",
    "oof_predictions.parquet",
    "threshold_audit.csv",
    "economic_paths.parquet",
)
RUN_ROOT = CODE_ROOT / "experiments" / "cache" / "event_window_direction_head"
MODELS = ("logreg", "xgboost")
BASELINES = ("channel_side", "always_long", "always_short", "random_50", "oracle")
EXPECTED_SOURCE_ACTIVATIONS = 3_431
EXPECTED_SCORED_ACTIVATIONS = 2_939
READER_ARTIFACTS = (
    "direction_dataset.parquet",
    "economic_paths.parquet",
    "geometry_audit.csv",
    "feature_audit.csv",
    "correlation_audit.csv",
    "fold_audit.csv",
    "oof_direction_predictions.parquet",
    "predictive_metrics.csv",
    "policy_paths.parquet",
    "combined_policy_ledger.parquet",
    "economic_metrics.csv",
    "paired_bootstrap.csv",
    "frequency_audit.csv",
    "concurrency_audit.csv",
    "leakage_audit.csv",
    "protocol.json",
    "frozen_protocol.json",
    "summary.json",
)
_TIMING_COLUMNS = (
    "activation_key",
    "arm",
    "fold_id",
    "window_id",
    "channel_episode_id",
    "step",
    "decision_time",
    "threshold",
    "activation_score",
    "channel_side",
    "reference_price",
    "adaptive_barrier_bps",
)
_FUTURE_FEATURE_TOKENS = (
    "future",
    "outcome",
    "move_code",
    "bars_held",
    "terminal",
    "net_r",
    "gross_r",
    "delta_r",
    "best_side",
)
_SOURCE_DEPENDENCIES = (
    Path(__file__),
    CODE_ROOT / ".python-version",
    CODE_ROOT / "requirements-v-repro.txt",
    CODE_ROOT / "evaluation" / "channel_backtest.py",
    CODE_ROOT / "evaluation" / "channel_window_validation.py",
    CODE_ROOT / "evaluation" / "event_window_cost_aware_policy.py",
    CODE_ROOT / "evaluation" / "event_window_economics.py",
    CODE_ROOT / "evaluation" / "event_window_large_move_policy.py",
    CODE_ROOT / "evaluation" / "event_window_opportunity_policy.py",
    CODE_ROOT / "evaluation" / "event_window_tail_policy.py",
    CODE_ROOT / "experiments" / "event_window_conditional_oof.py",
    CODE_ROOT / "experiments" / "event_window_cost_aware_dataset.py",
    CODE_ROOT / "experiments" / "event_window_cost_aware_models.py",
    CODE_ROOT / "experiments" / "event_window_cost_aware_oof.py",
    CODE_ROOT / "experiments" / "event_window_dataset.py",
    CODE_ROOT / "experiments" / "event_window_direction_dataset.py",
    CODE_ROOT / "experiments" / "event_window_direction_models.py",
    CODE_ROOT / "experiments" / "event_window_direction_oof.py",
    CODE_ROOT / "experiments" / "event_window_large_move_dataset.py",
    CODE_ROOT / "experiments" / "event_window_large_move_models.py",
    CODE_ROOT / "experiments" / "event_window_large_move_oof.py",
    CODE_ROOT / "experiments" / "event_window_magnitude_dataset.py",
    CODE_ROOT / "experiments" / "event_window_magnitude_models.py",
    CODE_ROOT / "experiments" / "event_window_magnitude_oof.py",
    CODE_ROOT / "experiments" / "event_window_opportunity_oof.py",
    CODE_ROOT / "experiments" / "event_window_tail_calibration.py",
    CODE_ROOT / "experiments" / "event_window_tail_dataset.py",
    CODE_ROOT / "experiments" / "event_window_tail_neural.py",
    CODE_ROOT / "experiments" / "event_window_tail_oof.py",
    CODE_ROOT / "experiments" / "event_window_tail_tabular.py",
    CODE_ROOT / "experiments" / "event_window_tcn.py",
    CODE_ROOT / "experiments" / "run_event_window_adaptive_large_move.py",
    CODE_ROOT / "experiments" / "run_event_window_conditional_opportunity.py",
    CODE_ROOT / "experiments" / "run_event_window_cost_aware_entry.py",
    CODE_ROOT / "experiments" / "run_event_window_economic_feasibility.py",
    CODE_ROOT / "experiments" / "run_event_window_magnitude_timing.py",
    CODE_ROOT / "experiments" / "run_event_window_opportunity_head.py",
    CODE_ROOT / "experiments" / "run_event_window_tail_models.py",
    CODE_ROOT / "experiments" / "run_event_window_tcn.py",
    CODE_ROOT / "features" / "event_window_inputs.py",
    CODE_ROOT / "features" / "event_windows.py",
    CODE_ROOT / "features" / "linear_channels.py",
)
_U_IDENTITY_FIELDS = ("run_hash", "protocol_hash", "source_hash", "input_hash")
_RAW_SOURCE_ROLES = ("minute", "five_minute", "hourly", "positioning")
_RAW_FINGERPRINT_METHOD = "pyarrow_filtered_canonical_ipc_sha256_v1"
_FINGERPRINT_CHUNK_ROWS = 65_536


@dataclass(frozen=True)
class DirectionHeadConfig:
    development_start: str = "2021-01-01"
    development_end_exclusive: str = "2025-07-01"
    scored_start: str = "2022-07-01"
    target_multiple_b: float = 2.0
    stop_multiple_b: float = 1.0
    hold_minutes: int = 120
    diagnostic_barriers_bps: tuple[float, ...] = (75.0, 120.0)
    entry_cost_bps: float = 5.0
    target_exit_cost_bps: float = 5.0
    other_exit_cost_bps: float = 5.0
    bootstrap_draws: int = 2_000
    bootstrap_seed: int = 42
    minimum_path_completeness: float = 0.99
    expected_source_activations: int = EXPECTED_SOURCE_ACTIVATIONS
    expected_scored_activations: int = EXPECTED_SCORED_ACTIVATIONS
    model: DirectionModelConfig = field(default_factory=DirectionModelConfig)


@dataclass(frozen=True)
class FrozenUArtifacts:
    run_hash: str
    protocol_hash: str
    source_hash: str
    input_hash: str
    run_dir: Path
    protocol: dict[str, object]
    summary: dict[str, object]
    state: dict[str, object]
    ledger: pd.DataFrame
    manifest_sha256: str
    activation_ledger_sha256: str
    oof_sha256: str
    economic_paths_sha256: str


@dataclass(frozen=True)
class DirectionHeadRunResult:
    run_dir: Path
    summary: dict[str, object]


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path.name}")
    return value


def protocol_dict(
    config: DirectionHeadConfig = DirectionHeadConfig(), *, smoke: bool = False
) -> dict[str, object]:
    effective_model, effective_draws = _effective_execution(config, smoke=smoke)
    return {
        "study": "notebook_v_economic_direction_value_head",
        "stage": "dev",
        "smoke": bool(smoke),
        **asdict(config),
        "model": asdict(effective_model),
        "bootstrap_draws": effective_draws,
        "registered_model_config": asdict(config.model),
        "effective_model_config": asdict(effective_model),
        "registered_bootstrap_draws": config.bootstrap_draws,
        "effective_bootstrap_draws": effective_draws,
        "frozen_u_run_hash": FROZEN_U_RUN_HASH,
        "frozen_timing_arm": FROZEN_TIMING_ARM,
        "models": list(MODELS),
        "direction_features": list(DIRECTION_FEATURES),
        "direction_feature_count": len(DIRECTION_FEATURES),
        "round_trip_cost_bps": (
            config.entry_cost_bps + config.other_exit_cost_bps
        ),
        "entry": "native one-minute Open at frozen activation time",
        "path_interval": "half-open [t,t+120m)",
        "same_minute_ambiguity": "stop_first",
        "risk_geometry": "adaptive 1B stop, 2B target, 120 minute hold",
        "diagnostic_only_barriers_bps": list(config.diagnostic_barriers_bps),
        "forced_direction": True,
        "trade_gate_included": False,
        "direction_confidence_gate": False,
        "threshold_search": False,
        "timing_model_refit": False,
        "activation_frequency_optimised": False,
        "bootstrap_unit": "channel_episode_id",
        "raw_input_identity_contract": _raw_input_identity_contract(config),
        "forward_or_lockbox_loaded": False,
    }


def _effective_execution(
    config: DirectionHeadConfig, *, smoke: bool
) -> tuple[DirectionModelConfig, int]:
    model = replace(config.model, xgb_estimators=12) if smoke else config.model
    draws = 20 if smoke else config.bootstrap_draws
    return model, draws


def _development_bounds(
    config: DirectionHeadConfig,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    return (
        pd.Timestamp(config.development_start, tz="UTC"),
        pd.Timestamp(config.development_end_exclusive, tz="UTC"),
    )


def _raw_input_identity_contract(config: DirectionHeadConfig) -> dict[str, object]:
    start, end = _development_bounds(config)
    return {
        "method": _RAW_FINGERPRINT_METHOD,
        "development_start": start.isoformat(),
        "development_end_exclusive": end.isoformat(),
        "source_roles": list(_RAW_SOURCE_ROLES),
        "pyarrow_filter_before_pandas": True,
        "whole_file_bytes_hashed": False,
        "file_metadata_hashed": False,
        "absolute_paths_hashed": False,
        "applies_to_full_run": True,
    }


def _canonical_bounded_frame_sha256(
    frame: pd.DataFrame,
    *,
    source_role: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> str:
    """Hash one bounded frame without path or whole-file metadata."""
    import pyarrow as pa

    if source_role not in _RAW_SOURCE_ROLES:
        raise ValueError(f"unknown raw source role: {source_role}")
    index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True, errors="raise"))
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f"{source_role} bounded timestamps are not unique and sorted")
    if len(index) and (index.min() < start or index.max() >= end):
        raise AssertionError(f"{source_role} fingerprint crossed development bounds")
    if frame.columns.duplicated().any() or not all(
        isinstance(column, str) for column in frame.columns
    ):
        raise ValueError(f"{source_role} bounded columns must be unique strings")
    ordered_columns = sorted(frame.columns)
    reserved_index = "__bounded_index_utc_ns__"
    if reserved_index in ordered_columns:
        raise ValueError(f"{source_role} uses the reserved fingerprint index column")
    metadata = {
        "method": _RAW_FINGERPRINT_METHOD,
        "source_role": source_role,
        "development_start": start.isoformat(),
        "development_end_exclusive": end.isoformat(),
        "index_name": frame.index.name,
        "index_dtype": "datetime64[ns, UTC]",
        "columns": [
            {"name": name, "dtype": str(frame[name].dtype)}
            for name in ordered_columns
        ],
        "rows": len(frame),
        "chunk_rows": _FINGERPRINT_CHUNK_ROWS,
    }
    digest = hashlib.sha256()
    encoded_metadata = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    digest.update(len(encoded_metadata).to_bytes(8, "big"))
    digest.update(encoded_metadata)
    for left in range(0, len(frame), _FINGERPRINT_CHUNK_ROWS):
        right = min(left + _FINGERPRINT_CHUNK_ROWS, len(frame))
        chunk = frame.iloc[left:right].loc[:, ordered_columns].copy()
        chunk.insert(0, reserved_index, index[left:right].asi8)
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        table = table.replace_schema_metadata(None)
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        encoded_chunk = sink.getvalue().to_pybytes()
        digest.update(len(encoded_chunk).to_bytes(8, "big"))
        digest.update(encoded_chunk)
    return digest.hexdigest()


def _bounded_development_raw_identity(
    data_root: Path,
    config: DirectionHeadConfig,
) -> dict[str, object]:
    """Fingerprint only DEVELOPMENT rows from every raw V input role."""
    from experiments.run_event_window_tcn import (
        EventWindowStudyConfig,
        _input_paths,
        _load_bounded_parquet,
    )

    start, end = _development_bounds(config)
    paths = _input_paths(EventWindowStudyConfig(), Path(data_root))
    if set(paths) != set(_RAW_SOURCE_ROLES):
        raise AssertionError("Notebook V raw source roles changed")
    fingerprints: dict[str, dict[str, object]] = {}
    for role in _RAW_SOURCE_ROLES:
        frame = _load_bounded_parquet(paths[role], start=start, end=end)
        maximum = frame.index.max() if len(frame) else None
        fingerprints[role] = {
            "content_sha256": _canonical_bounded_frame_sha256(
                frame,
                source_role=role,
                start=start,
                end=end,
            ),
            "rows": len(frame),
            "max_timestamp": maximum.isoformat() if maximum is not None else None,
        }
    payload = {
        "method": _RAW_FINGERPRINT_METHOD,
        "development_start": start.isoformat(),
        "development_end_exclusive": end.isoformat(),
        "source_fingerprints": fingerprints,
    }
    return {**payload, "aggregate_sha256": _sha_payload(payload)}


def _bounded_development_identity_passes(
    identity: dict[str, object] | None,
    config: DirectionHeadConfig,
) -> bool:
    if not isinstance(identity, dict):
        return False
    start, end = _development_bounds(config)
    fingerprints = identity.get("source_fingerprints")
    if (
        identity.get("method") != _RAW_FINGERPRINT_METHOD
        or identity.get("development_start") != start.isoformat()
        or identity.get("development_end_exclusive") != end.isoformat()
        or not isinstance(fingerprints, dict)
        or set(fingerprints) != set(_RAW_SOURCE_ROLES)
    ):
        return False
    for record in fingerprints.values():
        if not isinstance(record, dict):
            return False
        sha256 = record.get("content_sha256")
        rows = record.get("rows")
        maximum = record.get("max_timestamp")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or not isinstance(rows, int)
            or rows < 0
            or (rows == 0) != (maximum is None)
        ):
            return False
        if maximum is not None and pd.Timestamp(maximum) >= end:
            return False
    payload = {
        name: value
        for name, value in identity.items()
        if name != "aggregate_sha256"
    }
    return identity.get("aggregate_sha256") == _sha_payload(payload)


def _bounded_identity_evidence(
    *,
    smoke: bool,
    identity: dict[str, object] | None,
    config: DirectionHeadConfig,
) -> dict[str, object]:
    contract = _raw_input_identity_contract(config)
    if smoke:
        return {
            **contract,
            "applicable": False,
            "passed": True,
            "aggregate_sha256": None,
            "source_fingerprints": {},
            "detail": "smoke consumes only the pinned bounded Notebook U handoff",
        }
    passed = _bounded_development_identity_passes(identity, config)
    return {
        **contract,
        "applicable": True,
        "passed": passed,
        "aggregate_sha256": (
            identity.get("aggregate_sha256") if isinstance(identity, dict) else None
        ),
        "source_fingerprints": (
            identity.get("source_fingerprints", {})
            if isinstance(identity, dict)
            else {}
        ),
        "detail": "four raw roles filtered to DEVELOPMENT before pandas",
    }


def economic_viability(
    *,
    path_completeness: float,
    scored_activations: int,
    mean_net_r_ci_low: float,
    versus_channel_ci_low: float,
    leakage_passed: bool,
) -> bool:
    return bool(
        path_completeness >= 0.99
        and scored_activations == EXPECTED_SCORED_ACTIVATIONS
        and mean_net_r_ci_low > 0.0
        and versus_channel_ci_low > 0.0
        and leakage_passed
    )


def _reject_sealed_paths(*paths: Path) -> None:
    forbidden = {"forward", "lockbox", "q2"}
    for path in paths:
        lowered = str(Path(path)).replace("\\", "/").lower()
        components = {
            token
            for component in lowered.split("/")
            for token in component.replace("-", "_").split("_")
        }
        if forbidden.intersection(components):
            raise ValueError(f"forward/Q2/lockbox path is sealed: {path}")


def _validate_config(config: DirectionHeadConfig) -> None:
    expected = DirectionHeadConfig()
    if config != expected:
        raise ValueError("Notebook V frozen protocol changed from its complete config")


def _validate_frozen_u_identity(
    *,
    pointer: dict[str, object],
    state: dict[str, object],
    protocol: dict[str, object],
    manifest_sha256: str,
) -> None:
    """Validate U's pinned bytes and reproduce its protocol/run identities."""
    if manifest_sha256 != FROZEN_U_MANIFEST_SHA256:
        raise ValueError("frozen Notebook U manifest digest changed")
    expected_relative = f"{FROZEN_U_RUN_HASH}/full"
    if (
        pointer.get("run_hash") != FROZEN_U_RUN_HASH
        or pointer.get("relative_path") != expected_relative
    ):
        raise ValueError("frozen Notebook U pointer identity changed")
    if state.get("status") != "complete" or state.get("run_hash") != FROZEN_U_RUN_HASH:
        raise ValueError("frozen Notebook U run is incomplete or changed")
    if (
        state.get("protocol_hash") != FROZEN_U_PROTOCOL_HASH
        or pointer.get("protocol_hash") != FROZEN_U_PROTOCOL_HASH
    ):
        raise ValueError("frozen Notebook U protocol identity changed")
    if any(protocol.get(name) != state.get(name) for name in _U_IDENTITY_FIELDS):
        raise ValueError("frozen Notebook U protocol metadata identity changed")
    protocol_payload = {
        name: value
        for name, value in protocol.items()
        if name not in _U_IDENTITY_FIELDS
    }
    if _sha_payload(protocol_payload) != FROZEN_U_PROTOCOL_HASH:
        raise ValueError("frozen Notebook U protocol hash changed")
    run_hash = _sha_payload(
        {
            "protocol_hash": state.get("protocol_hash"),
            "source_hash": state.get("source_hash"),
            "input_hash": state.get("input_hash"),
        }
    )[:20]
    if run_hash != state.get("run_hash"):
        raise ValueError("frozen Notebook U run identity changed")


def _canonical_float64_text(value: object) -> str:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("canonical threshold values must be finite")
    return repr(number)


def _thresholds_match_canonical_csv(
    ledger: pd.DataFrame, threshold_audit: pd.DataFrame
) -> bool:
    required = {"fold_id", "threshold"}
    if not required.issubset(ledger.columns) or not required.issubset(
        threshold_audit.columns
    ):
        return False
    left = ledger[["fold_id", "threshold"]].copy()
    right = threshold_audit[["fold_id", "threshold"]].copy()
    if right["fold_id"].isna().any() or right["fold_id"].duplicated().any():
        return False
    try:
        left["canonical"] = left["threshold"].map(_canonical_float64_text)
        right["canonical"] = right["threshold"].astype("string").str.strip()
    except (TypeError, ValueError):
        return False
    if left.groupby("fold_id")["canonical"].nunique().ne(1).any():
        return False
    registered = right.set_index("fold_id")["canonical"].to_dict()
    actual = left.groupby("fold_id", sort=False)["canonical"].first().to_dict()
    return actual == registered


def load_frozen_u_artifacts(
    run_root: Path = FROZEN_U_ROOT,
    *,
    config: DirectionHeadConfig = DirectionHeadConfig(),
) -> FrozenUArtifacts:
    """Validate the exact completed U handoff before loading its ledger."""
    root = Path(run_root)
    _reject_sealed_paths(root)
    pointer = _read_json(root / "latest_dev.json")
    expected_relative = f"{FROZEN_U_RUN_HASH}/full"
    run_dir = (root / expected_relative).resolve()
    if not run_dir.is_relative_to(root.resolve()):
        raise ValueError("frozen Notebook U path escaped its root")

    state_path = run_dir / "run_state.json"
    state = _read_json(state_path)
    protocol = _read_json(run_dir / "protocol.json")
    manifest_sha256 = _sha256(state_path)
    _validate_frozen_u_identity(
        pointer=pointer,
        state=state,
        protocol=protocol,
        manifest_sha256=manifest_sha256,
    )
    records = state.get("artifacts")
    if not isinstance(records, dict):
        raise ValueError("frozen Notebook U artifact registry is missing")
    for name in FROZEN_U_ARTIFACTS:
        path = run_dir / name
        record = records.get(name)
        if (
            not path.is_file()
            or not isinstance(record, dict)
            or int(record.get("size", -1)) != path.stat().st_size
            or str(record.get("sha256", "")) != _sha256(path)
        ):
            raise ValueError(f"frozen Notebook U artifact changed: {name}")

    summary = _read_json(run_dir / "summary.json")
    if state.get("summary") != summary:
        raise ValueError("frozen Notebook U state summary changed")
    if (
        protocol.get("stage") != "dev"
        or bool(protocol.get("smoke", False))
        or protocol.get("development_end_exclusive")
        != config.development_end_exclusive
        or summary.get("forward_or_lockbox_loaded") is not False
        or summary.get("activation_counts", {}).get(FROZEN_TIMING_ARM)
        != config.expected_source_activations
    ):
        raise ValueError("Notebook V accepts only the bounded full Notebook U run")

    ledger_path = run_dir / "activation_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    ledger = ledger.loc[ledger["arm"].eq(FROZEN_TIMING_ARM)].copy().reset_index(drop=True)
    missing = sorted(set(_TIMING_COLUMNS).difference(ledger.columns))
    if missing:
        raise ValueError(f"frozen Notebook U ledger schema changed: {missing}")
    ledger["decision_time"] = pd.to_datetime(
        ledger["decision_time"], utc=True, errors="raise"
    )
    if len(ledger) != config.expected_source_activations:
        raise ValueError("frozen Notebook U activation count changed")
    if ledger["activation_key"].isna().any() or ledger["activation_key"].duplicated().any():
        raise ValueError("frozen Notebook U activation keys are not unique")
    if set(ledger["fold_id"]) != {"2022H1", *SCORED_FOLDS}:
        raise ValueError("frozen Notebook U fold population changed")
    scored = ledger["fold_id"].isin(SCORED_FOLDS)
    if int(scored.sum()) != config.expected_scored_activations:
        raise ValueError("frozen Notebook U scored activation count changed")
    development_end = pd.Timestamp(config.development_end_exclusive, tz="UTC")
    if ledger["decision_time"].max() >= development_end:
        raise ValueError("frozen Notebook U ledger crossed the development boundary")
    numeric = ledger[
        ["threshold", "activation_score", "reference_price", "adaptive_barrier_bps"]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("frozen Notebook U timing fields are not finite")

    oof_path = run_dir / "oof_predictions.parquet"
    oof_columns = [
        "arm",
        "fold_id",
        "window_id",
        "channel_episode_id",
        "step",
        "decision_time",
        "p_t_le_60",
    ]
    oof = pd.read_parquet(oof_path, columns=oof_columns)
    oof = oof.loc[oof["arm"].eq(FROZEN_TIMING_ARM)].drop(columns="arm")
    oof["decision_time"] = pd.to_datetime(oof["decision_time"], utc=True, errors="raise")
    join_keys = [
        "fold_id",
        "window_id",
        "channel_episode_id",
        "step",
        "decision_time",
    ]
    if oof.duplicated(join_keys).any():
        raise ValueError("frozen Notebook U OOF timing keys changed")
    score_check = ledger.merge(oof, on=join_keys, how="left", validate="one_to_one")
    if score_check["p_t_le_60"].isna().any() or not np.array_equal(
        score_check["activation_score"].to_numpy(dtype=float),
        score_check["p_t_le_60"].to_numpy(dtype=float),
    ):
        raise ValueError("frozen Notebook U activation scores changed")

    threshold = pd.read_csv(
        run_dir / "threshold_audit.csv", dtype={"threshold": "string"}
    )
    threshold = threshold.loc[threshold["arm"].eq(FROZEN_TIMING_ARM), ["fold_id", "threshold"]]
    if not _thresholds_match_canonical_csv(ledger, threshold):
        raise ValueError("frozen Notebook U activation thresholds changed")

    return FrozenUArtifacts(
        run_hash=FROZEN_U_RUN_HASH,
        protocol_hash=str(state["protocol_hash"]),
        source_hash=str(state["source_hash"]),
        input_hash=str(state["input_hash"]),
        run_dir=run_dir,
        protocol=protocol,
        summary=summary,
        state=state,
        ledger=ledger,
        manifest_sha256=manifest_sha256,
        activation_ledger_sha256=_sha256(ledger_path),
        oof_sha256=_sha256(oof_path),
        economic_paths_sha256=_sha256(run_dir / "economic_paths.parquet"),
    )


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in _SOURCE_DEPENDENCIES:
        digest.update(str(path.relative_to(CODE_ROOT)).encode("utf-8"))
        canonical_text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        canonical_text = canonical_text.replace("\r", "\n")
        digest.update(canonical_text.encode("utf-8"))
    return digest.hexdigest()


def _latest(run_root: Path, run_hash: str, protocol_hash: str) -> None:
    path = Path(run_root) / "latest_dev.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "run_hash": run_hash,
                "protocol_hash": protocol_hash,
                "relative_path": f"{run_hash}/full",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _validated_completed_summary(
    run_dir: Path,
    identity: dict[str, str],
    *,
    frozen_u: FrozenUArtifacts,
) -> dict[str, object] | None:
    try:
        state = _read_json(run_dir / "run_state.json")
        if state.get("status") != "complete" or any(
            state.get(name) != value for name, value in identity.items()
        ):
            return None
        records = state.get("artifacts")
        if not isinstance(records, dict):
            return None
        for name in READER_ARTIFACTS:
            path = run_dir / name
            record = records.get(name)
            if (
                not path.is_file()
                or not isinstance(record, dict)
                or int(record.get("size", -1)) != path.stat().st_size
                or str(record.get("sha256", "")) != _sha256(path)
            ):
                return None
        frozen = _read_json(run_dir / "frozen_protocol.json")
        summary = _read_json(run_dir / "summary.json")
        if (
            frozen.get("frozen_u_run_hash") != frozen_u.run_hash
            or frozen.get("frozen_u_manifest_sha256") != frozen_u.manifest_sha256
            or state.get("summary") != summary
        ):
            return None
        return summary
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _smoke_primary_paths(frozen_u: FrozenUArtifacts) -> pd.DataFrame:
    """Reprice U's registered native stop-first paths without using U net returns."""
    paths = pd.read_parquet(frozen_u.run_dir / "economic_paths.parquet")
    paths = paths.loc[
        paths["arm"].eq(FROZEN_TIMING_ARM)
        & paths["target_multiple_b"].eq(2.0)
        & paths["hold_minutes"].eq(120)
    ].copy()
    if len(paths) != 2 * EXPECTED_SOURCE_ACTIVATIONS or paths.duplicated(
        ["activation_key", "direction"]
    ).any():
        raise ValueError("frozen Notebook U native paths changed")
    if set(paths["activation_key"]) != set(frozen_u.ledger["activation_key"]):
        raise ValueError("frozen Notebook U native path keys changed")
    if not paths["path_complete"].astype(bool).all():
        raise ValueError("Notebook V smoke requires complete registered native paths")
    barrier = paths["adaptive_barrier_bps"].to_numpy(dtype=float)
    gross_bps = paths["gross_bps"].to_numpy(dtype=float)
    paths["entry_cost_bps"] = 5.0
    paths["exit_cost_bps"] = 5.0
    paths["cost_bps"] = 10.0
    paths["cost_r"] = 10.0 / barrier
    paths["net_bps"] = gross_bps - 10.0
    paths["net_r"] = paths["net_bps"].to_numpy(dtype=float) / barrier
    paths["geometry"] = "adaptive_primary"
    return paths.reset_index(drop=True)


def _smoke_directional(ledger: pd.DataFrame) -> LargeMoveDecisionDataset:
    """Create causal-only structural features for a fast all-key smoke run."""
    rows = np.arange(len(ledger), dtype=float)
    times = pd.to_datetime(ledger["decision_time"], utc=True)
    episode_code = ledger["channel_episode_id"].astype("category").cat.codes.to_numpy(float)
    side = np.where(ledger["channel_side"].eq("long"), 1.0, -1.0)
    sources = np.column_stack(
        [
            ledger["activation_score"].to_numpy(float),
            ledger["threshold"].to_numpy(float),
            ledger["activation_score"].to_numpy(float)
            - ledger["threshold"].to_numpy(float),
            ledger["adaptive_barrier_bps"].to_numpy(float) / 250.0,
            np.log(ledger["reference_price"].to_numpy(float)),
            ledger["step"].to_numpy(float) / 100.0,
            times.dt.hour.to_numpy(float) / 24.0,
            times.dt.dayofweek.to_numpy(float) / 7.0,
            episode_code / max(1.0, episode_code.max()),
            side,
            rows / max(1.0, rows.max()),
        ]
    )
    columns = []
    for position, _ in enumerate(DIRECTION_FEATURES[:-1]):
        source = sources[:, position % sources.shape[1]]
        columns.append(
            source * (1.0 + position / 100.0)
            + 0.001 * np.sin((rows + 1.0) / (position + 2.0))
        )
    decisions = ledger[
        ["window_id", "channel_episode_id", "step", "decision_time"]
    ].copy()
    return LargeMoveDecisionDataset(
        decisions=decisions,
        tabular=np.column_stack(columns).astype(np.float32),
        tabular_features=DIRECTION_FEATURES[:-1],
        dropped_features=(),
        feature_set="directional_smoke_causal",
    )


def _full_inputs(
    frozen_u: FrozenUArtifacts,
    *,
    frozen_j,
    data_root: Path,
    config: DirectionHeadConfig,
) -> tuple[LargeMoveDecisionDataset, pd.DataFrame, pd.Timestamp]:
    """Build the causal directional matrix and replay native V paths."""
    from experiments.event_window_large_move_dataset import (
        AdaptiveMoveConfig,
        build_large_move_dataset,
        label_adaptive_large_moves,
    )
    from experiments.run_event_window_tail_models import (
        _build_tail_dataset,
    )
    from experiments.run_event_window_tcn import _load_bounded_parquet

    base, _, loaded = _build_tail_dataset(
        frozen_j,
        data_root=Path(data_root),
        smoke=False,
        validate_legacy_input_fingerprint=False,
    )
    development_start = pd.Timestamp(config.development_start, tz="UTC")
    development_end = pd.Timestamp(config.development_end_exclusive, tz="UTC")
    if (
        loaded.read_start != development_start
        or loaded.read_end_exclusive != development_end
        or loaded.max_loaded_timestamp >= development_end
    ):
        raise AssertionError("Notebook V bounded causal inputs changed")
    target = AdaptiveMoveConfig(
        horizon_minutes=config.hold_minutes,
        taker_entry_bps=config.entry_cost_bps,
        maker_target_exit_bps=config.target_exit_cost_bps,
        taker_other_exit_bps=config.other_exit_cost_bps,
    )
    labels = label_adaptive_large_moves(base.decisions, loaded.minute, target)
    directional = build_large_move_dataset(base, labels, feature_set="directional")

    start = frozen_u.ledger["decision_time"].min()
    read_end = min(
        frozen_u.ledger["decision_time"].max()
        + pd.Timedelta(minutes=config.hold_minutes),
        development_end,
    )
    minute = _load_bounded_parquet(
        Path(data_root) / "btcusdt_1m_2021_2026.parquet",
        start=start,
        end=read_end,
    )
    if not minute.empty and minute.index.max() >= development_end:
        raise AssertionError("Notebook V native replay crossed development")
    primary = replay_brackets(
        frozen_u.ledger,
        minute,
        target_multiples=(config.target_multiple_b,),
        hold_minutes=(config.hold_minutes,),
        entry_cost_bps=config.entry_cost_bps,
        target_exit_cost_bps=config.target_exit_cost_bps,
        other_exit_cost_bps=config.other_exit_cost_bps,
    )
    primary["entry_cost_bps"] = config.entry_cost_bps
    primary["exit_cost_bps"] = config.other_exit_cost_bps
    primary["geometry"] = "adaptive_primary"
    geometries = [primary]
    for barrier in config.diagnostic_barriers_bps:
        attempts = frozen_u.ledger.copy()
        attempts["adaptive_barrier_bps"] = float(barrier)
        diagnostic = replay_brackets(
            attempts,
            minute,
            target_multiples=(config.target_multiple_b,),
            hold_minutes=(config.hold_minutes,),
            entry_cost_bps=config.entry_cost_bps,
            target_exit_cost_bps=config.target_exit_cost_bps,
            other_exit_cost_bps=config.other_exit_cost_bps,
        )
        diagnostic["entry_cost_bps"] = config.entry_cost_bps
        diagnostic["exit_cost_bps"] = config.other_exit_cost_bps
        diagnostic["geometry"] = f"fixed_{int(barrier)}bps_diagnostic"
        geometries.append(diagnostic)
    return directional, pd.concat(geometries, ignore_index=True), loaded.max_loaded_timestamp


def _direction_frame(dataset: DirectionDataset) -> pd.DataFrame:
    frame = dataset.decisions.copy().reset_index(drop=True)
    for position, name in enumerate(dataset.tabular_features):
        frame[name] = dataset.tabular[:, position]
    return frame


def _feature_audit(dataset: DirectionDataset) -> pd.DataFrame:
    rows = []
    for position, name in enumerate(dataset.tabular_features):
        values = np.asarray(dataset.tabular[:, position], dtype=float)
        rows.append(
            {
                "feature": name,
                "position": position,
                "included": True,
                "known_at_decision_time": True,
                "future_column": any(token in name.lower() for token in _FUTURE_FEATURE_TOKENS),
                "finite_rows": int(np.isfinite(values).sum()),
                "rows": len(values),
                "finite_fraction": float(np.isfinite(values).mean()),
            }
        )
    return pd.DataFrame(rows)


def _correlation_audit(dataset: DirectionDataset) -> pd.DataFrame:
    frame = pd.DataFrame(dataset.tabular, columns=dataset.tabular_features)
    correlation = frame.corr(method="spearman", min_periods=max(2, len(frame) // 20))
    rows = []
    for left in range(len(dataset.tabular_features)):
        for right in range(left + 1, len(dataset.tabular_features)):
            rows.append(
                {
                    "feature_left": dataset.tabular_features[left],
                    "feature_right": dataset.tabular_features[right],
                    "spearman": float(correlation.iloc[left, right]),
                    "abs_spearman": float(abs(correlation.iloc[left, right])),
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _combined_policy_ledger(
    ledger: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    scored = ledger.loc[ledger["fold_id"].isin(SCORED_FOLDS), list(_TIMING_COLUMNS)].copy()
    scored["activation_margin"] = scored["activation_score"] - scored["threshold"]
    rows = []
    for model in MODELS:
        model_predictions = predictions.loc[
            predictions["model"].eq(model),
            [
                "activation_key",
                "direction_score",
                "p_long",
                "predicted_delta_r",
                "chosen_direction",
            ],
        ]
        combined = scored.merge(
            model_predictions,
            on="activation_key",
            how="left",
            sort=False,
            validate="one_to_one",
        )
        if len(combined) != EXPECTED_SCORED_ACTIVATIONS:
            raise AssertionError(f"{model} changed the frozen activation count")
        if combined["direction_score"].isna().any() or combined[
            "chosen_direction"
        ].isna().any():
            raise AssertionError(f"{model} did not score every frozen activation")
        if not combined["chosen_direction"].isin(("long", "short")).all():
            raise AssertionError(f"{model} emitted a non-direction action")
        pd.testing.assert_frame_equal(
            combined[list(_TIMING_COLUMNS)].reset_index(drop=True),
            scored[list(_TIMING_COLUMNS)].reset_index(drop=True),
            check_exact=True,
        )
        combined.insert(0, "model", model)
        rows.append(combined)
    output = pd.concat(rows, ignore_index=True)
    if output.duplicated(["model", "activation_key"]).any():
        raise AssertionError("combined policy duplicated frozen activation keys")
    return output


def _predictive_metrics(
    predictions: pd.DataFrame, dataset: DirectionDataset
) -> pd.DataFrame:
    truth = dataset.decisions.loc[
        dataset.decisions["fold_id"].isin(SCORED_FOLDS),
        ["activation_key", "delta_r", "economic_value", "best_side"],
    ]
    rows = []
    for model in MODELS:
        work = predictions.loc[predictions["model"].eq(model)].merge(
            truth, on="activation_key", how="left", validate="one_to_one"
        )
        non_tie = work["best_side"].ne("tie")
        correct = work.loc[non_tie, "chosen_direction"].eq(
            work.loc[non_tie, "best_side"]
        ).to_numpy(dtype=float)
        weights = work.loc[non_tie, "economic_value"].to_numpy(dtype=float)
        weighted_accuracy = (
            float(np.average(correct, weights=weights)) if weights.sum() > 0.0 else np.nan
        )
        row = {
            "model": model,
            "scored_rows": len(work),
            "tie_rows": int((~non_tie).sum()),
            "raw_sign_accuracy": float(correct.mean()) if len(correct) else np.nan,
            "value_weighted_sign_accuracy": weighted_accuracy,
            "mae_delta_r": np.nan,
            "rmse_delta_r": np.nan,
            "value_weighted_mae_delta_r": np.nan,
            "value_weighted_rmse_delta_r": np.nan,
        }
        if model == "xgboost":
            errors = (
                work["predicted_delta_r"].to_numpy(dtype=float)
                - work["delta_r"].to_numpy(dtype=float)
            )
            value = work["economic_value"].to_numpy(dtype=float)
            row.update(
                {
                    "mae_delta_r": float(np.mean(np.abs(errors))),
                    "rmse_delta_r": float(np.sqrt(np.mean(errors**2))),
                    "value_weighted_mae_delta_r": float(
                        np.average(np.abs(errors), weights=value)
                    ),
                    "value_weighted_rmse_delta_r": float(
                        np.sqrt(np.average(errors**2, weights=value))
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _policy_paths(
    combined: pd.DataFrame,
    primary_paths: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    scored_timing = combined.loc[
        combined["model"].eq(MODELS[0]), list(_TIMING_COLUMNS)
    ].reset_index(drop=True)
    path_columns = [
        "activation_key",
        "direction",
        "path_complete",
        "censored",
        "entry_price",
        "exit_price",
        "bars_held",
        "outcome",
        "gross_bps",
        "net_bps",
        "gross_r",
        "net_r",
        "cost_bps",
        "cost_r",
        "target_multiple_b",
        "hold_minutes",
    ]
    lookup = primary_paths.loc[
        primary_paths["activation_key"].isin(scored_timing["activation_key"]),
        path_columns,
    ]
    if lookup.duplicated(["activation_key", "direction"]).any():
        raise AssertionError("primary path lookup contains duplicate directions")
    paired_scored = scored_timing[["activation_key", "channel_side"]].merge(
        paired, on="activation_key", how="left", validate="one_to_one"
    )
    oracle = np.where(
        paired_scored["net_r_long"] > paired_scored["net_r_short"],
        "long",
        np.where(
            paired_scored["net_r_long"] < paired_scored["net_r_short"],
            "short",
            paired_scored["channel_side"],
        ),
    )
    rng = np.random.default_rng(seed)
    selections: list[tuple[str, pd.DataFrame]] = []
    for model in MODELS:
        selections.append(
            (
                model,
                combined.loc[
                    combined["model"].eq(model),
                    ["activation_key", "chosen_direction"],
                ],
            )
        )
    selections.extend(
        [
            (
                "channel_side",
                scored_timing[["activation_key", "channel_side"]].rename(
                    columns={"channel_side": "chosen_direction"}
                ),
            ),
            (
                "always_long",
                scored_timing[["activation_key"]].assign(chosen_direction="long"),
            ),
            (
                "always_short",
                scored_timing[["activation_key"]].assign(chosen_direction="short"),
            ),
            (
                "random_50",
                scored_timing[["activation_key"]].assign(
                    chosen_direction=np.where(
                        rng.integers(0, 2, len(scored_timing)) == 1, "long", "short"
                    )
                ),
            ),
            (
                "oracle",
                paired_scored[["activation_key"]].assign(chosen_direction=oracle),
            ),
        ]
    )
    rows = []
    for scenario, chosen in selections:
        selected = scored_timing.merge(
            chosen, on="activation_key", how="left", sort=False, validate="one_to_one"
        ).merge(
            lookup,
            left_on=["activation_key", "chosen_direction"],
            right_on=["activation_key", "direction"],
            how="left",
            sort=False,
            validate="one_to_one",
        )
        if len(selected) != EXPECTED_SCORED_ACTIVATIONS or selected["net_r"].isna().any():
            raise AssertionError(f"{scenario} did not preserve every scored path")
        if not selected["decision_time"].reset_index(drop=True).equals(
            scored_timing["decision_time"].reset_index(drop=True)
        ):
            raise AssertionError(f"{scenario} changed activation timing")
        selected.insert(0, "scenario", scenario)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def _cluster_interval(
    frame: pd.DataFrame,
    *,
    value_column: str,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    usable = frame.loc[np.isfinite(frame[value_column])]
    grouped = usable.groupby("channel_episode_id")[value_column].agg(["sum", "count"])
    if grouped.empty:
        return np.nan, np.nan, np.nan
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    point = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=float)
    for draw in range(draws):
        sample = rng.integers(0, len(grouped), len(grouped))
        values[draw] = sums[sample].sum() / counts[sample].sum()
    low, high = np.quantile(values, [0.025, 0.975])
    return point, float(low), float(high)


def _economic_tables(
    policy_paths: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    oracle = policy_paths.loc[
        policy_paths["scenario"].eq("oracle"), ["activation_key", "net_r"]
    ].rename(columns={"net_r": "oracle_net_r"})
    rows = []
    bootstrap_rows = []
    for position, (scenario, group) in enumerate(
        policy_paths.groupby("scenario", sort=False)
    ):
        complete = group.loc[group["path_complete"].astype(bool)].copy()
        point, low, high = _cluster_interval(
            complete,
            value_column="net_r",
            draws=draws,
            seed=seed + position,
        )
        with_oracle = complete.merge(
            oracle, on="activation_key", how="left", validate="one_to_one"
        )
        regret = with_oracle["oracle_net_r"] - with_oracle["net_r"]
        oracle_mean = float(with_oracle["oracle_net_r"].mean())
        rows.append(
            {
                "scenario": scenario,
                "activations": len(group),
                "complete_paths": len(complete),
                "path_completeness": float(len(complete) / len(group)),
                "mean_gross_r": float(complete["gross_r"].mean()),
                "total_gross_r": float(complete["gross_r"].sum()),
                "mean_net_r": point,
                "total_net_r": float(complete["net_r"].sum()),
                "mean_gross_bps": float(complete["gross_bps"].mean()),
                "total_gross_bps": float(complete["gross_bps"].sum()),
                "mean_net_bps": float(complete["net_bps"].mean()),
                "total_net_bps": float(complete["net_bps"].sum()),
                "mean_net_r_ci_low": low,
                "mean_net_r_ci_high": high,
                "tp_fraction": float(complete["outcome"].eq("tp").mean()),
                "sl_fraction": float(complete["outcome"].eq("sl").mean()),
                "timeout_fraction": float(complete["outcome"].eq("timeout").mean()),
                "mean_oracle_regret_r": float(regret.mean()),
                "oracle_value_capture": (
                    float(point / oracle_mean) if not np.isclose(oracle_mean, 0.0) else np.nan
                ),
            }
        )
        bootstrap_rows.append(
            {
                "comparison": f"absolute::{scenario}",
                "candidate": scenario,
                "baseline": "",
                "point_mean_net_r": point,
                "ci_low": low,
                "ci_high": high,
                "draws": draws,
                "bootstrap_unit": "channel_episode_id",
            }
        )

    comparisons = [(scenario, "channel_side") for scenario in MODELS]
    comparisons.append(("xgboost", "logreg"))
    for position, (candidate, baseline) in enumerate(comparisons, start=20):
        left = policy_paths.loc[
            policy_paths["scenario"].eq(candidate),
            ["activation_key", "channel_episode_id", "net_r"],
        ].rename(columns={"net_r": "candidate_net_r"})
        right = policy_paths.loc[
            policy_paths["scenario"].eq(baseline), ["activation_key", "net_r"]
        ].rename(columns={"net_r": "baseline_net_r"})
        paired = left.merge(right, on="activation_key", how="inner", validate="one_to_one")
        if len(paired) != EXPECTED_SCORED_ACTIVATIONS:
            raise AssertionError(f"paired bootstrap keys changed: {candidate}, {baseline}")
        paired["delta_net_r"] = paired["candidate_net_r"] - paired["baseline_net_r"]
        point, low, high = _cluster_interval(
            paired,
            value_column="delta_net_r",
            draws=draws,
            seed=seed + position,
        )
        bootstrap_rows.append(
            {
                "comparison": f"{candidate}_minus_{baseline}",
                "candidate": candidate,
                "baseline": baseline,
                "point_mean_net_r": point,
                "ci_low": low,
                "ci_high": high,
                "draws": draws,
                "bootstrap_unit": "channel_episode_id",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def _frequency_audit(policy_paths: pd.DataFrame, config: DirectionHeadConfig) -> pd.DataFrame:
    days = len(
        pd.date_range(
            pd.Timestamp(config.scored_start, tz="UTC"),
            pd.Timestamp(config.development_end_exclusive, tz="UTC")
            - pd.Timedelta(days=1),
            freq="D",
        )
    )
    reference = set(
        policy_paths.loc[policy_paths["scenario"].eq("channel_side"), "activation_key"]
    )
    rows = []
    for scenario, group in policy_paths.groupby("scenario", sort=False):
        rows.append(
            {
                "scenario": scenario,
                "activations": len(group),
                "unique_activation_keys": group["activation_key"].nunique(),
                "calendar_days": days,
                "activations_per_day": float(len(group) / days),
                "keys_equal_frozen": set(group["activation_key"]) == reference,
                "timing_owned_by_frozen_u": True,
            }
        )
    return pd.DataFrame(rows)


def _concurrency_audit(ledger: pd.DataFrame, hold_minutes: int) -> pd.DataFrame:
    times = pd.DatetimeIndex(pd.to_datetime(ledger["decision_time"], utc=True)).sort_values()
    active: list[pd.Timestamp] = []
    concurrent = []
    for time in times:
        active = [end for end in active if end > time]
        active.append(time + pd.Timedelta(minutes=hold_minutes))
        concurrent.append(len(active))
    values = np.asarray(concurrent, dtype=float)
    return pd.DataFrame(
        [
            {
                "source": FROZEN_TIMING_ARM,
                "activations": len(times),
                "hold_minutes": hold_minutes,
                "mean_active_at_entry": float(values.mean()),
                "maximum_active_at_entry": int(values.max()),
                "overlap_fraction": float((values > 1.0).mean()),
                "capacity_suppression": False,
            }
        ]
    )


def _geometry_audit(
    paths: pd.DataFrame,
    *,
    smoke: bool,
    config: DirectionHeadConfig,
) -> pd.DataFrame:
    rows = []
    specifications = [
        ("adaptive_primary", True, 75.0, 250.0),
        ("fixed_75bps_diagnostic", False, 75.0, 75.0),
        ("fixed_120bps_diagnostic", False, 120.0, 120.0),
    ]
    for geometry, primary, minimum, maximum in specifications:
        selected = paths.loc[paths["geometry"].eq(geometry)]
        rows.append(
            {
                "geometry": geometry,
                "primary": primary,
                "diagnostic_only": not primary,
                "stop_bps_minimum": minimum,
                "stop_bps_maximum": maximum,
                "target_multiple_b": config.target_multiple_b,
                "hold_minutes": config.hold_minutes,
                "entry_cost_bps": config.entry_cost_bps,
                "exit_cost_bps": config.other_exit_cost_bps,
                "round_trip_cost_bps": 10.0,
                "native_one_minute": True,
                "stop_first": True,
                "executed": bool(len(selected)),
                "path_rows": len(selected),
                "path_completeness": (
                    float(selected["path_complete"].astype(bool).mean())
                    if len(selected)
                    else np.nan
                ),
                "smoke_deferred": bool(smoke and not primary),
            }
        )
    return pd.DataFrame(rows)


def _artifact_schemas() -> dict[str, set[str]]:
    return {
        "direction_dataset.parquet": {"activation_key", *DIRECTION_FEATURES},
        "economic_paths.parquet": {
            "activation_key",
            "direction",
            "geometry",
            "net_r",
            "cost_bps",
        },
        "geometry_audit.csv": {"geometry", "primary", "round_trip_cost_bps"},
        "feature_audit.csv": {"feature", "position", "included"},
        "correlation_audit.csv": {"feature_left", "feature_right", "abs_spearman"},
        "fold_audit.csv": {"fold_id", "episode_overlap", "validation_rows"},
        "oof_direction_predictions.parquet": {
            "model",
            "activation_key",
            "direction_score",
            "chosen_direction",
        },
        "predictive_metrics.csv": {"model", "raw_sign_accuracy"},
        "policy_paths.parquet": {"scenario", "activation_key", "net_r"},
        "combined_policy_ledger.parquet": {
            "model",
            "activation_key",
            "threshold",
            "activation_score",
            "chosen_direction",
        },
        "economic_metrics.csv": {"scenario", "mean_net_r", "mean_net_r_ci_low"},
        "paired_bootstrap.csv": {"comparison", "ci_low", "ci_high"},
        "frequency_audit.csv": {"scenario", "activations", "keys_equal_frozen"},
        "concurrency_audit.csv": {"source", "maximum_active_at_entry"},
        "leakage_audit.csv": {"check", "passed", "detail"},
    }


def _validate_written_artifacts(store: _Store) -> None:
    records = store.state.get("artifacts")
    if not isinstance(records, dict):
        raise AssertionError("artifact registry is missing")
    missing = [name for name in READER_ARTIFACTS if not (store.run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Notebook V artifacts missing: {missing}")
    for name in READER_ARTIFACTS:
        record = records.get(name)
        path = store.run_dir / name
        if (
            not isinstance(record, dict)
            or int(record.get("size", -1)) != path.stat().st_size
            or str(record.get("sha256", "")) != _sha256(path)
        ):
            raise AssertionError(f"Notebook V artifact hash changed before completion: {name}")
    for name, required in _artifact_schemas().items():
        path = store.run_dir / name
        frame = pd.read_parquet(path) if name.endswith(".parquet") else pd.read_csv(path)
        missing_columns = sorted(required.difference(frame.columns))
        if missing_columns:
            raise AssertionError(f"Notebook V artifact schema changed: {name}: {missing_columns}")


def run_direction_head(
    *,
    stage: str = "dev",
    smoke: bool = False,
    data_root: Path = DEFAULT_DATA_ROOT,
    frozen_u_root: Path = FROZEN_U_ROOT,
    run_root: Path = RUN_ROOT,
    config: DirectionHeadConfig = DirectionHeadConfig(),
) -> DirectionHeadRunResult:
    if stage != "dev":
        raise ValueError("Notebook V permits development only; forward and Q2 are sealed")
    _validate_config(config)
    _reject_sealed_paths(Path(data_root), Path(frozen_u_root), Path(run_root))
    frozen_u = load_frozen_u_artifacts(Path(frozen_u_root), config=config)
    protocol = protocol_dict(config, smoke=smoke)
    source_hash = _source_hash()
    input_payload = {
        "frozen_u_run_hash": frozen_u.run_hash,
        "frozen_u_manifest_sha256": frozen_u.manifest_sha256,
        "frozen_u_activation_ledger_sha256": frozen_u.activation_ledger_sha256,
        "frozen_u_oof_sha256": frozen_u.oof_sha256,
        "frozen_u_economic_paths_sha256": frozen_u.economic_paths_sha256,
    }
    frozen_j = None
    bounded_raw_identity = None
    if not smoke:
        from experiments.run_event_window_tail_models import (
            FROZEN_J_ROOT,
            load_frozen_j_artifacts,
        )

        frozen_j = load_frozen_j_artifacts(FROZEN_J_ROOT)
        bounded_raw_identity = _bounded_development_raw_identity(
            Path(data_root), config
        )
        input_payload["frozen_j_run_hash"] = frozen_j.run_hash
        input_payload["frozen_j_manifest_sha256"] = frozen_j.manifest_sha256
        input_payload["frozen_j_input_hash"] = frozen_j.input_hash
        input_payload["bounded_development_raw_identity"] = bounded_raw_identity
    bounded_identity_evidence = _bounded_identity_evidence(
        smoke=smoke,
        identity=bounded_raw_identity,
        config=config,
    )
    if not bool(bounded_identity_evidence["passed"]):
        raise AssertionError("Notebook V bounded raw input identity failed")
    protocol["bounded_development_input_identity"] = bounded_identity_evidence
    protocol_hash = _sha_payload(protocol)
    input_hash = _sha_payload(input_payload)
    identity = {
        "run_hash": _sha_payload(
            {
                "protocol_hash": protocol_hash,
                "source_hash": source_hash,
                "input_hash": input_hash,
            }
        )[:20],
        "protocol_hash": protocol_hash,
        "source_hash": source_hash,
        "input_hash": input_hash,
    }
    run_dir = Path(run_root) / identity["run_hash"] / ("smoke" if smoke else "full")
    cached = _validated_completed_summary(run_dir, identity, frozen_u=frozen_u)
    if cached is not None:
        if not smoke:
            _latest(Path(run_root), identity["run_hash"], protocol_hash)
        return DirectionHeadRunResult(run_dir, cached)

    store = _Store(run_dir, identity)
    try:
        store.json("protocol.json", {**protocol, **identity})
        if smoke:
            paths = _smoke_primary_paths(frozen_u)
            directional = _smoke_directional(frozen_u.ledger)
            max_loaded_timestamp = pd.Timestamp(
                frozen_u.summary["max_loaded_timestamp"]
            )
            path_source = "registered_u_native_1m_stop_first_repriced_to_5_plus_5"
        else:
            if frozen_j is None:
                raise AssertionError("full Notebook V run is missing frozen J inputs")
            directional, paths, max_loaded_timestamp = _full_inputs(
                frozen_u,
                frozen_j=frozen_j,
                data_root=Path(data_root),
                config=config,
            )
            path_source = "native_1m_replay_brackets"
        development_end = pd.Timestamp(config.development_end_exclusive, tz="UTC")
        if max_loaded_timestamp.tzinfo is None:
            max_loaded_timestamp = max_loaded_timestamp.tz_localize("UTC")
        else:
            max_loaded_timestamp = max_loaded_timestamp.tz_convert("UTC")
        if max_loaded_timestamp >= development_end:
            raise AssertionError("Notebook V loaded a sealed later timestamp")

        primary = paths.loc[paths["geometry"].eq("adaptive_primary")].copy()
        paired = pair_direction_paths(primary)
        dataset_ledger = frozen_u.ledger.drop(
            columns=[
                name
                for name in DIRECTION_FEATURES[:-1]
                if name in frozen_u.ledger.columns
            ]
        )
        dataset = build_direction_dataset(dataset_ledger, directional, paired)
        if dataset.tabular_features != DIRECTION_FEATURES:
            raise AssertionError("Notebook V direction feature contract changed")
        if len(dataset.decisions) != EXPECTED_SOURCE_ACTIVATIONS:
            raise AssertionError("Notebook V direction dataset changed frozen frequency")
        feature_audit = _feature_audit(dataset)
        correlation_audit = _correlation_audit(dataset)
        model_config, draws = _effective_execution(config, smoke=smoke)
        oof = run_direction_oof(dataset, config=model_config)
        predictions = oof.predictions.copy()
        combined = _combined_policy_ledger(frozen_u.ledger, predictions)
        policy_paths = _policy_paths(
            combined,
            primary,
            paired,
            seed=config.bootstrap_seed,
        )
        predictive = _predictive_metrics(predictions, dataset)
        economics, bootstraps = _economic_tables(
            policy_paths,
            draws=draws,
            seed=config.bootstrap_seed,
        )
        frequency = _frequency_audit(policy_paths, config)
        scored_ledger = frozen_u.ledger.loc[
            frozen_u.ledger["fold_id"].isin(SCORED_FOLDS)
        ]
        concurrency = _concurrency_audit(scored_ledger, config.hold_minutes)
        geometry = _geometry_audit(paths, smoke=smoke, config=config)

        timing_preserved = True
        for model in MODELS:
            model_timing = combined.loc[
                combined["model"].eq(model), list(_TIMING_COLUMNS)
            ].reset_index(drop=True)
            frozen_timing = scored_ledger[list(_TIMING_COLUMNS)].reset_index(drop=True)
            timing_preserved &= model_timing.equals(frozen_timing)
        path_completeness = float(primary["path_complete"].astype(bool).mean())
        feature_allow_list = dataset.tabular_features == DIRECTION_FEATURES
        future_fields_absent = not any(
            token in feature.lower()
            for feature in dataset.tabular_features
            for token in _FUTURE_FEATURE_TOKENS
        )
        forward_seal_passed = bool(bounded_identity_evidence["passed"]) and (
            frozen_u.ledger["decision_time"].max() < development_end
            and max_loaded_timestamp < development_end
        )
        leakage = pd.DataFrame(
            [
                {"check": "exact frozen Notebook U run hash", "passed": frozen_u.run_hash == FROZEN_U_RUN_HASH, "detail": frozen_u.run_hash},
                {"check": "valid frozen U SHA-256 manifest", "passed": True, "detail": frozen_u.manifest_sha256},
                {"check": "frozen U timing arm exact", "passed": set(frozen_u.ledger["arm"]) == {FROZEN_TIMING_ARM}, "detail": FROZEN_TIMING_ARM},
                {"check": "3,431 frozen activation keys exact", "passed": len(frozen_u.ledger) == EXPECTED_SOURCE_ACTIVATIONS and frozen_u.ledger["activation_key"].nunique() == EXPECTED_SOURCE_ACTIVATIONS, "detail": str(len(frozen_u.ledger))},
                {"check": "2,939 scored keys per model", "passed": combined.groupby("model")["activation_key"].nunique().eq(EXPECTED_SCORED_ACTIVATIONS).all(), "detail": str(combined.groupby("model")["activation_key"].nunique().to_dict())},
                {"check": "combined ledger preserves U timing bytes and keys", "passed": timing_preserved, "detail": "activation key/time/score/threshold/episode fields"},
                {"check": "uniform 5+5 costs", "passed": set(primary["cost_bps"]) == {10.0} and set(primary["entry_cost_bps"]) == {5.0} and set(primary["exit_cost_bps"]) == {5.0}, "detail": "10 bps every exit path"},
                {"check": "native RR2/120m stop-first paths", "passed": set(primary["target_multiple_b"]) == {2.0} and set(primary["hold_minutes"]) == {120}, "detail": path_source},
                {"check": "one forced side per model activation", "passed": not combined.duplicated(["model", "activation_key"]).any() and combined["chosen_direction"].isin(("long", "short")).all(), "detail": "no WAIT or confidence gate"},
                {"check": "direction feature allow-list exact", "passed": feature_allow_list and len(dataset.tabular_features) == 28, "detail": ",".join(dataset.tabular_features)},
                {"check": "future columns denied", "passed": future_fields_absent and not feature_audit["future_column"].astype(bool).any(), "detail": ",".join(_FUTURE_FEATURE_TOKENS)},
                {"check": "fold-local imputation and weights", "passed": np.isfinite(oof.fold_audit[["uniqueness_mean", "uniqueness_min", "uniqueness_max"]].to_numpy(float)).all() and (oof.fold_audit["uniqueness_min"] > 0.0).all(), "detail": "Task 2 fold-local model interface"},
                {"check": "120-minute training purge", "passed": (pd.to_datetime(oof.fold_audit["train_label_end_max"], utc=True) < pd.to_datetime(oof.fold_audit["validation_start"], utc=True)).all(), "detail": "strictly before validation"},
                {"check": "training and validation episodes disjoint", "passed": oof.fold_audit["episode_overlap"].eq(0).all(), "detail": f"{len(oof.fold_audit)} scored folds"},
                {"check": "activation timestamps bounded", "passed": frozen_u.ledger["decision_time"].max() < development_end and max_loaded_timestamp < development_end, "detail": str(max_loaded_timestamp)},
                {"check": "bounded development raw-content identity", "passed": bounded_identity_evidence["passed"], "detail": bounded_identity_evidence["detail"]},
                {"check": "forward and Q2 remain sealed", "passed": forward_seal_passed, "detail": "bounded identity and max-timestamp guards passed"},
                {"check": "timing head and threshold not refit", "passed": True, "detail": "frozen U xgboost_base owns WHEN and frequency"},
                {"check": "episode-cluster bootstrap", "passed": set(bootstraps["bootstrap_unit"]) == {"channel_episode_id"}, "detail": f"{draws} smoke draws" if smoke else f"{draws} registered draws"},
            ]
        )
        if not leakage["passed"].astype(bool).all():
            failed = leakage.loc[~leakage["passed"].astype(bool), "check"].tolist()
            raise AssertionError(f"Notebook V leakage audit failed: {failed}")

        model_viability: dict[str, bool] = {}
        for model in MODELS:
            economic = economics.loc[economics["scenario"].eq(model)].iloc[0]
            versus = bootstraps.loc[
                bootstraps["comparison"].eq(f"{model}_minus_channel_side")
            ].iloc[0]
            model_viability[model] = economic_viability(
                path_completeness=float(economic["path_completeness"]),
                scored_activations=int(economic["activations"]),
                mean_net_r_ci_low=float(economic["mean_net_r_ci_low"]),
                versus_channel_ci_low=float(versus["ci_low"]),
                leakage_passed=bool(leakage["passed"].astype(bool).all()),
            )
        selected_model: str | None = None
        if not smoke:
            if model_viability["logreg"] and model_viability["xgboost"]:
                xgb_minus_logreg = bootstraps.loc[
                    bootstraps["comparison"].eq("xgboost_minus_logreg"), "ci_low"
                ].iloc[0]
                selected_model = "xgboost" if float(xgb_minus_logreg) > 0.0 else "logreg"
            elif model_viability["logreg"]:
                selected_model = "logreg"
            elif model_viability["xgboost"]:
                selected_model = "xgboost"
        decision = (
            "smoke only; no economic claim"
            if smoke
            else (
                f"retain {selected_model} forced direction head"
                if selected_model
                else "negative direction result; add no trade gate"
            )
        )

        store.parquet("direction_dataset.parquet", _direction_frame(dataset))
        store.parquet("economic_paths.parquet", paths)
        store.csv("geometry_audit.csv", geometry)
        store.csv("feature_audit.csv", feature_audit)
        store.csv("correlation_audit.csv", correlation_audit)
        store.csv("fold_audit.csv", oof.fold_audit)
        store.parquet("oof_direction_predictions.parquet", predictions)
        store.csv("predictive_metrics.csv", predictive)
        store.parquet("policy_paths.parquet", policy_paths)
        store.parquet("combined_policy_ledger.parquet", combined)
        store.csv("economic_metrics.csv", economics)
        store.csv("paired_bootstrap.csv", bootstraps)
        store.csv("frequency_audit.csv", frequency)
        store.csv("concurrency_audit.csv", concurrency)
        store.csv("leakage_audit.csv", leakage)
        store.json(
            "frozen_protocol.json",
            {
                "frozen_u_run_hash": frozen_u.run_hash,
                "frozen_u_protocol_hash": frozen_u.protocol_hash,
                "frozen_u_source_hash": frozen_u.source_hash,
                "frozen_u_input_hash": frozen_u.input_hash,
                "frozen_u_manifest_sha256": frozen_u.manifest_sha256,
                "frozen_u_activation_ledger_sha256": frozen_u.activation_ledger_sha256,
                "frozen_u_oof_sha256": frozen_u.oof_sha256,
                "frozen_u_economic_paths_sha256": frozen_u.economic_paths_sha256,
                "frozen_j_run_hash": input_payload.get("frozen_j_run_hash"),
                "frozen_j_manifest_sha256": input_payload.get(
                    "frozen_j_manifest_sha256"
                ),
                "frozen_timing_arm": FROZEN_TIMING_ARM,
                "timing_model_refit": False,
                "timing_frequency_changed": False,
                "path_source": path_source,
                "registered_xgb_estimators": config.model.xgb_estimators,
                "effective_xgb_estimators": model_config.xgb_estimators,
                "registered_bootstrap_draws": config.bootstrap_draws,
                "effective_bootstrap_draws": draws,
                "bounded_development_input_identity": bounded_identity_evidence,
                "forward_or_lockbox_loaded": not forward_seal_passed,
            },
        )
        summary = {
            **identity,
            "frozen_u_run_hash": frozen_u.run_hash,
            "frozen_timing_arm": FROZEN_TIMING_ARM,
            "source_activations": len(frozen_u.ledger),
            "scored_activations": int(scored_ledger["activation_key"].nunique()),
            "warmup_activations": int(
                frozen_u.ledger["fold_id"].eq("2022H1").sum()
            ),
            "scored_folds": list(SCORED_FOLDS),
            "direction_feature_count": len(DIRECTION_FEATURES),
            "oof_rows_per_model": predictions.groupby("model").size().astype(int).to_dict(),
            "combined_rows_per_model": combined.groupby("model").size().astype(int).to_dict(),
            "path_completeness": path_completeness,
            "entry_cost_bps": config.entry_cost_bps,
            "exit_cost_bps": config.other_exit_cost_bps,
            "round_trip_cost_bps": 10.0,
            "registered_xgb_estimators": config.model.xgb_estimators,
            "effective_xgb_estimators": model_config.xgb_estimators,
            "registered_bootstrap_draws": config.bootstrap_draws,
            "effective_bootstrap_draws": draws,
            "model_viability": model_viability,
            "selected_model": selected_model,
            "decision": decision,
            "research_claim": "smoke_only_no_claim" if smoke else "development_result",
            "max_loaded_timestamp": str(max_loaded_timestamp),
            "bounded_development_input_identity": bounded_identity_evidence,
            "timing_model_refit": False,
            "timing_frequency_changed": False,
            "trade_gate_included": False,
            "forced_direction": True,
            "forward_or_lockbox_loaded": not forward_seal_passed,
        }
        store.json("summary.json", summary)
        _validate_written_artifacts(store)
        store.complete(summary)
        if not smoke:
            _latest(Path(run_root), identity["run_hash"], protocol_hash)
        return DirectionHeadRunResult(run_dir, summary)
    except BaseException as error:
        store.fail(error)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("dev",), default="dev")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--frozen-u-root", type=Path, default=FROZEN_U_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_direction_head(
        stage=args.stage,
        smoke=args.smoke,
        data_root=args.data_root,
        frozen_u_root=args.frozen_u_root,
        run_root=args.run_root,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True, default=str))
    print(result.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASELINES",
    "CODE_ROOT",
    "DirectionHeadConfig",
    "DirectionHeadRunResult",
    "FROZEN_TIMING_ARM",
    "FROZEN_U_ARTIFACTS",
    "FROZEN_U_ROOT",
    "FROZEN_U_RUN_HASH",
    "FrozenUArtifacts",
    "MODELS",
    "READER_ARTIFACTS",
    "RUN_ROOT",
    "SCORED_FOLDS",
    "economic_viability",
    "load_frozen_u_artifacts",
    "protocol_dict",
    "run_direction_head",
]
