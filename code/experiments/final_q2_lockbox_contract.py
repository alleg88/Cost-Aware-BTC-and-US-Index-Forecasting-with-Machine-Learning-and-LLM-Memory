"""Immutable candidate and cost contract for the final Q2-2026 lockbox."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


PROTOCOL_VERSION = "final-q2-lockbox-v1"
Q2_START = pd.Timestamp("2026-04-01T00:00:00Z")
Q2_END = pd.Timestamp("2026-07-01T00:00:00Z")
PRIMARY_ESTIMAND = "btc_union_minus_lstm_total_net"
MODEL_NAMES = (
    "logreg",
    "decision_tree",
    "random_forest",
    "svm_linear",
    "xgboost_balanced",
    "catboost_balanced",
    "mlp",
    "lstm",
    "gru",
)
EXPECTED_CANDIDATE_IDS = (
    "btc_qualified_union_v1",
    "btc_lstm_dz55",
    "usa500_best_single_deberta_svm",
    "usa500_deberta_soft_vote",
    "usatech_best_single_deepseek_lstm",
    "usatech_deepseek_soft_vote",
)


@dataclass(frozen=True)
class ArtifactRef:
    role: str
    path: str
    sha256: str


@dataclass(frozen=True)
class MemberSpec:
    model_name: str
    width_bps: int
    signal_tau: float | None


@dataclass(frozen=True)
class CandidateSpec:
    stream: str
    candidate_id: str
    role: str
    arm: str
    decision_tau: float
    members: tuple[MemberSpec, ...]
    combiner: str | None
    artifacts: tuple[ArtifactRef, ...]


@dataclass(frozen=True)
class CostSpec:
    per_side_bps: float
    round_trip_bps: float


@dataclass(frozen=True)
class ReconstructionTolerance:
    max_abs_probability_error: float
    mean_abs_probability_error: float
    require_exact_classes: bool
    require_exact_decisions: bool
    torch_threads: int


@dataclass(frozen=True)
class LockboxProtocol:
    protocol_version: str
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp
    candidates: tuple[CandidateSpec, ...]
    costs: Mapping[str, CostSpec]
    primary_estimand: str
    bootstrap_replicates: int
    bootstrap_seed: int
    default_reconstruction_tolerance: ReconstructionTolerance
    reconstruction_exceptions: Mapping[str, ReconstructionTolerance]


def _canonical(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        stamp = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return stamp.isoformat()
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: _canonical(getattr(value, key))
            for key in value.__dataclass_fields__
        }
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expect_keys(payload: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{label} keys changed: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _timestamp(value: str, *, label: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return stamp.tz_convert("UTC")


def _artifact(payload: Mapping[str, Any]) -> ArtifactRef:
    _expect_keys(payload, {"role", "path", "sha256"}, label="artifact")
    artifact = ArtifactRef(
        role=str(payload["role"]),
        path=str(payload["path"]).replace("\\", "/"),
        sha256=str(payload["sha256"]).lower(),
    )
    if not artifact.role or not artifact.path:
        raise ValueError("artifact role/path must be non-empty")
    if len(artifact.sha256) != 64 or any(
        character not in "0123456789abcdef" for character in artifact.sha256
    ):
        raise ValueError("artifact sha256 must be 64 lowercase hexadecimal characters")
    if Path(artifact.path).is_absolute() or ".." in Path(artifact.path).parts:
        raise ValueError("artifact path must be repository-relative")
    return artifact


def _member(payload: Mapping[str, Any]) -> MemberSpec:
    _expect_keys(payload, {"model_name", "width_bps", "signal_tau"}, label="member")
    signal_tau = payload["signal_tau"]
    member = MemberSpec(
        model_name=str(payload["model_name"]),
        width_bps=int(payload["width_bps"]),
        signal_tau=None if signal_tau is None else float(signal_tau),
    )
    if member.model_name not in MODEL_NAMES or member.width_bps <= 0:
        raise ValueError("member model/width is outside the frozen registry")
    if member.signal_tau is not None and not 0.0 <= member.signal_tau <= 1.0:
        raise ValueError("member signal_tau must lie in [0, 1]")
    return member


def _candidate(payload: Mapping[str, Any]) -> CandidateSpec:
    _expect_keys(
        payload,
        {
            "stream",
            "candidate_id",
            "role",
            "arm",
            "decision_tau",
            "members",
            "combiner",
            "artifacts",
        },
        label="candidate",
    )
    candidate = CandidateSpec(
        stream=str(payload["stream"]),
        candidate_id=str(payload["candidate_id"]),
        role=str(payload["role"]),
        arm=str(payload["arm"]),
        decision_tau=float(payload["decision_tau"]),
        members=tuple(_member(item) for item in payload["members"]),
        combiner=None if payload["combiner"] is None else str(payload["combiner"]),
        artifacts=tuple(_artifact(item) for item in payload["artifacts"]),
    )
    if candidate.stream not in {"btcusdt", "usa500", "usatech"}:
        raise ValueError("candidate stream changed")
    if candidate.role not in {"primary", "control", "secondary"}:
        raise ValueError("candidate role changed")
    if not 0.0 <= candidate.decision_tau <= 1.0:
        raise ValueError("candidate decision_tau must lie in [0, 1]")
    if not candidate.members or not candidate.artifacts:
        raise ValueError("candidate must bind members and artifacts")
    if len({artifact.path for artifact in candidate.artifacts}) != len(candidate.artifacts):
        raise ValueError("candidate artifact paths must be unique")
    return candidate


def _cost(payload: Mapping[str, Any]) -> CostSpec:
    _expect_keys(payload, {"per_side_bps", "round_trip_bps"}, label="cost")
    cost = CostSpec(
        per_side_bps=float(payload["per_side_bps"]),
        round_trip_bps=float(payload["round_trip_bps"]),
    )
    if cost.per_side_bps < 0 or cost.round_trip_bps < 0:
        raise ValueError("costs must be non-negative")
    if abs(2.0 * cost.per_side_bps - cost.round_trip_bps) > 1e-12:
        raise ValueError("round-trip cost must equal twice per-side cost")
    return cost


def _tolerance(payload: Mapping[str, Any]) -> ReconstructionTolerance:
    _expect_keys(
        payload,
        {
            "max_abs_probability_error",
            "mean_abs_probability_error",
            "require_exact_classes",
            "require_exact_decisions",
            "torch_threads",
        },
        label="reconstruction tolerance",
    )
    tolerance = ReconstructionTolerance(
        max_abs_probability_error=float(payload["max_abs_probability_error"]),
        mean_abs_probability_error=float(payload["mean_abs_probability_error"]),
        require_exact_classes=bool(payload["require_exact_classes"]),
        require_exact_decisions=bool(payload["require_exact_decisions"]),
        torch_threads=int(payload["torch_threads"]),
    )
    if (
        tolerance.max_abs_probability_error <= 0.0
        or tolerance.mean_abs_probability_error <= 0.0
        or tolerance.mean_abs_probability_error > tolerance.max_abs_probability_error
        or not tolerance.require_exact_classes
        or not tolerance.require_exact_decisions
        or tolerance.torch_threads != 16
    ):
        raise ValueError("reconstruction tolerance differs from the approved rule")
    return tolerance


def validate_registered_candidates(protocol: LockboxProtocol) -> None:
    ids = tuple(candidate.candidate_id for candidate in protocol.candidates)
    if ids != EXPECTED_CANDIDATE_IDS:
        raise ValueError("candidate registry differs from the approved six identities")
    if tuple(candidate.role for candidate in protocol.candidates) != (
        "primary",
        "control",
        "primary",
        "secondary",
        "primary",
        "secondary",
    ):
        raise ValueError("candidate roles changed")
    union = protocol.candidates[0]
    if (
        union.combiner != "opposite_signal_veto"
        or union.decision_tau != 0.0
        or tuple(
            (member.model_name, member.width_bps, member.signal_tau)
            for member in union.members
        )
        != (("lstm", 55, 0.75), ("svm_linear", 75, 0.0))
    ):
        raise ValueError("Qualified Union member contract changed")
    usa_single, usa_vote, tech_single, tech_vote = protocol.candidates[2:]
    if (
        usa_single.arm != "deberta_matched"
        or usa_single.members != (MemberSpec("svm_linear", 15, None),)
        or usa_single.decision_tau != 0.4
    ):
        raise ValueError("USA500 single-model contract changed")
    if (
        usa_vote.arm != "deberta_matched"
        or usa_vote.combiner != "soft_vote"
        or tuple(member.model_name for member in usa_vote.members) != MODEL_NAMES
        or {member.width_bps for member in usa_vote.members} != {15}
        or usa_vote.decision_tau != 0.55
    ):
        raise ValueError("USA500 soft-vote contract changed")
    if (
        tech_single.arm != "deepseek_matched"
        or tech_single.members != (MemberSpec("lstm", 15, None),)
        or tech_single.decision_tau != 0.65
    ):
        raise ValueError("USATECH single-model contract changed")
    if (
        tech_vote.arm != "deepseek_matched"
        or tech_vote.combiner != "soft_vote"
        or tuple(member.model_name for member in tech_vote.members) != MODEL_NAMES
        or {member.width_bps for member in tech_vote.members} != {5}
        or tech_vote.decision_tau != 0.55
    ):
        raise ValueError("USATECH soft-vote contract changed")


def load_lockbox_protocol(path: str | Path) -> LockboxProtocol:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _expect_keys(
        payload,
        {
            "protocol_version",
            "start_utc",
            "end_utc",
            "primary_estimand",
            "bootstrap_replicates",
            "bootstrap_seed",
            "reconstruction_tolerance",
            "costs",
            "candidates",
        },
        label="protocol",
    )
    costs_payload = payload["costs"]
    if set(costs_payload) != {"btcusdt", "usa500", "usatech"}:
        raise ValueError("cost stream registry changed")
    tolerance_payload = payload["reconstruction_tolerance"]
    _expect_keys(
        tolerance_payload,
        {"default", "exceptions"},
        label="reconstruction tolerance registry",
    )
    exceptions_payload = tolerance_payload["exceptions"]
    if set(exceptions_payload) != {"btcusdt:none:lstm:w55"}:
        raise ValueError("only BTC LSTM DZ55 may have a reconstruction exception")
    protocol = LockboxProtocol(
        protocol_version=str(payload["protocol_version"]),
        start_utc=_timestamp(payload["start_utc"], label="start_utc"),
        end_utc=_timestamp(payload["end_utc"], label="end_utc"),
        candidates=tuple(_candidate(item) for item in payload["candidates"]),
        costs={name: _cost(costs_payload[name]) for name in sorted(costs_payload)},
        primary_estimand=str(payload["primary_estimand"]),
        bootstrap_replicates=int(payload["bootstrap_replicates"]),
        bootstrap_seed=int(payload["bootstrap_seed"]),
        default_reconstruction_tolerance=_tolerance(tolerance_payload["default"]),
        reconstruction_exceptions={
            key: _tolerance(value) for key, value in exceptions_payload.items()
        },
    )
    if protocol.protocol_version != PROTOCOL_VERSION:
        raise ValueError("protocol version changed")
    if protocol.start_utc != Q2_START or protocol.end_utc != Q2_END:
        raise ValueError("Q2 interval changed")
    if protocol.primary_estimand != PRIMARY_ESTIMAND:
        raise ValueError("primary estimand changed")
    if protocol.bootstrap_replicates != 5_000 or protocol.bootstrap_seed != 42:
        raise ValueError("bootstrap contract changed")
    if protocol.costs != {
        "btcusdt": CostSpec(5.0, 10.0),
        "usa500": CostSpec(1.0, 2.0),
        "usatech": CostSpec(1.5, 3.0),
    }:
        raise ValueError("cost contract changed")
    if protocol.default_reconstruction_tolerance != ReconstructionTolerance(
        1e-6, 1e-6, True, True, 16
    ) or protocol.reconstruction_exceptions != {
        "btcusdt:none:lstm:w55": ReconstructionTolerance(
            2e-5, 5e-7, True, True, 16
        )
    }:
        raise ValueError("reconstruction tolerance registry changed")
    validate_registered_candidates(protocol)
    return protocol


__all__ = [
    "ArtifactRef",
    "CandidateSpec",
    "CostSpec",
    "EXPECTED_CANDIDATE_IDS",
    "LockboxProtocol",
    "MemberSpec",
    "MODEL_NAMES",
    "PRIMARY_ESTIMAND",
    "PROTOCOL_VERSION",
    "Q2_END",
    "Q2_START",
    "ReconstructionTolerance",
    "canonical_hash",
    "load_lockbox_protocol",
    "validate_registered_candidates",
]
