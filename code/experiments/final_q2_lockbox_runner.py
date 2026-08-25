"""Frozen scoring, replay and one-shot orchestration for the final Q2 lockbox."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from evaluation.trades_intrabar import simulate_bracket_trades_intrabar
from experiments.final_q2_lockbox_contract import (
    CandidateSpec,
    LockboxProtocol,
)
from experiments.final_q2_lockbox_state import (
    GLOBAL_STATE_ROOT,
    OpeningIdentity,
    assert_exact_resume,
    begin_global_open,
    mark_complete,
    mark_failed_after_open,
    require_global_opening,
)
from models.zoo import _aligned_proba


OPEN_AUTHORIZATION = "OPEN-Q2-2026-VARIANT-A"
PRELOCKBOX_TAG = "pre-lockbox-q2-2026"
CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent
DEFAULT_PROTOCOL_PATH = CODE_ROOT / "configs" / "final_q2_lockbox_protocol.json"
DEFAULT_MANIFEST_PATH = CODE_ROOT / "configs" / "final_q2_lockbox_manifest.json"
BAR_SIZE = pd.Timedelta(minutes=15)
BTC_TP_BPS = 200.0
BTC_SL_BPS = 100.0
BTC_MAX_HOLD = 1
PROBABILITY_COLUMNS = ("p_short", "p_flat", "p_long")


@dataclass(frozen=True)
class ReplayAudit:
    candidate_decisions: int
    executed_decisions: int
    terminal_censored_count: int
    session_gap_censored_count: int
    nonoverlap_censored_count: int = 0
    started_flat: bool = True


@dataclass(frozen=True)
class ReplayResult:
    candidate_id: str
    ledger: pd.DataFrame
    per_bar: pd.Series
    audit: ReplayAudit


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode(
            "utf-8"
        )
    )
    os.replace(temporary, path)
    return path


def _atomic_parquet(path: Path, frame: pd.DataFrame, *, index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, engine="pyarrow", index=index)
    os.replace(temporary, path)
    return path


def _write_parquet_once(
    path: Path, frame: pd.DataFrame, *, index: bool = False
) -> Path:
    if not path.exists():
        return _atomic_parquet(path, frame, index=index)
    existing = pd.read_parquet(path)
    expected = frame if not index else frame.reset_index()
    try:
        pd.testing.assert_frame_equal(
            existing.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_freq=False,
        )
    except AssertionError as exc:
        raise ValueError(f"immutable parquet artifact differs: {path}") from exc
    return path


def _write_json_once(path: Path, payload: Mapping[str, object]) -> Path:
    if not path.exists():
        return _atomic_json(path, payload)
    existing = json.loads(path.read_text(encoding="utf-8"))
    normalized = json.loads(json.dumps(payload, default=str))
    if existing != normalized:
        raise ValueError(f"immutable JSON artifact differs: {path}")
    return path


def _manifest_path(entry: Mapping[str, object]) -> Path:
    return Path(str(entry["path"]))


def _candidate_checkpoint_paths(
    output_root: Path, candidate_id: str
) -> dict[str, Path]:
    return {
        "prediction": output_root / "predictions" / f"{candidate_id}.parquet",
        "ledger": output_root / "ledgers" / f"{candidate_id}.parquet",
        "audit": output_root / "candidate_audits" / f"{candidate_id}.json",
        "checkpoint": output_root / "checkpoints" / f"{candidate_id}.json",
    }


def _load_candidate_checkpoint(
    output_root: str | Path, candidate_id: str
) -> tuple[pd.DataFrame, ReplayResult]:
    root = Path(output_root)
    paths = _candidate_checkpoint_paths(root, candidate_id)
    if not paths["checkpoint"].is_file():
        raise FileNotFoundError(paths["checkpoint"])
    checkpoint = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
    if checkpoint.get("candidate_id") != candidate_id:
        raise ValueError("candidate checkpoint identity changed")
    for key in ("prediction", "ledger", "audit"):
        if not paths[key].is_file() or _sha256(paths[key]) != checkpoint.get(
            f"{key}_sha256"
        ):
            raise ValueError(f"candidate checkpoint {key} hash changed")
    prediction = pd.read_parquet(paths["prediction"])
    prediction["timestamp"] = pd.to_datetime(prediction["timestamp"], utc=True)
    prediction = prediction.set_index("timestamp")
    prediction.index.name = checkpoint.get("prediction_index_name")
    ledger = pd.read_parquet(paths["ledger"])
    audit_payload = json.loads(paths["audit"].read_text(encoding="utf-8"))
    audit = ReplayAudit(**audit_payload)
    return prediction, ReplayResult(
        candidate_id,
        ledger,
        pd.Series(dtype=float, name="net_return"),
        audit,
    )


def _persist_candidate_checkpoint(
    output_root: str | Path,
    candidate_id: str,
    prediction: pd.DataFrame,
    result: ReplayResult,
) -> tuple[pd.DataFrame, ReplayResult]:
    root = Path(output_root)
    paths = _candidate_checkpoint_paths(root, candidate_id)
    if paths["checkpoint"].exists():
        existing_prediction, existing_result = _load_candidate_checkpoint(
            root, candidate_id
        )
        try:
            pd.testing.assert_frame_equal(
                existing_prediction, prediction, check_dtype=False, check_freq=False
            )
            pd.testing.assert_frame_equal(
                existing_result.ledger,
                result.ledger,
                check_dtype=False,
                check_freq=False,
            )
        except AssertionError as exc:
            raise ValueError("candidate checkpoint is immutable") from exc
        if existing_result.audit != result.audit:
            raise ValueError("candidate checkpoint audit is immutable")
        return existing_prediction, existing_result
    if any(path.exists() for key, path in paths.items() if key != "checkpoint"):
        raise ValueError("partial candidate checkpoint exists without its hash record")
    prediction_frame = prediction.copy().rename_axis("timestamp").reset_index()
    _atomic_parquet(paths["prediction"], prediction_frame, index=False)
    _atomic_parquet(paths["ledger"], result.ledger, index=False)
    _atomic_json(paths["audit"], asdict(result.audit))
    _atomic_json(
        paths["checkpoint"],
        {
            "candidate_id": candidate_id,
            "prediction_index_name": prediction.index.name,
            "prediction_sha256": _sha256(paths["prediction"]),
            "ledger_sha256": _sha256(paths["ledger"]),
            "audit_sha256": _sha256(paths["audit"]),
        },
    )
    return _load_candidate_checkpoint(root, candidate_id)


def _persist_stream_audit(
    output_root: Path, stream: str, audit: Mapping[str, object]
) -> Path:
    return _write_json_once(
        output_root / "stream_audits" / f"{stream}.json", audit
    )


def _load_stream_audit(output_root: Path, stream: str) -> dict[str, object]:
    path = output_root / "stream_audits" / f"{stream}.json"
    if not path.is_file():
        raise ValueError(f"completed {stream} checkpoints miss their immutable stream audit")
    return json.loads(path.read_text(encoding="utf-8"))


def _sentiment_files(score_root: Path) -> dict[str, Path]:
    return {
        f"sentiment__{path.name}": path
        for path in sorted(score_root.glob("*"))
        if path.is_file() and path.name != "sentiment_checkpoint.json"
    }


def _load_sentiment_checkpoint(score_root: Path) -> dict[str, Path] | None:
    checkpoint_path = score_root / "sentiment_checkpoint.json"
    if not checkpoint_path.exists():
        return None
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    files = _sentiment_files(score_root)
    expected = dict(checkpoint.get("artifact_hashes", {}))
    actual = {key: _sha256(path) for key, path in files.items()}
    if actual != expected:
        raise ValueError("completed sentiment checkpoint hash registry changed")
    return {**files, "sentiment__checkpoint": checkpoint_path}


def _persist_sentiment_checkpoint(score_root: Path) -> dict[str, Path]:
    existing = _load_sentiment_checkpoint(score_root)
    if existing is not None:
        return existing
    files = _sentiment_files(score_root)
    if not files:
        raise ValueError("sentiment checkpoint cannot be empty")
    checkpoint_path = _atomic_json(
        score_root / "sentiment_checkpoint.json",
        {"artifact_hashes": {key: _sha256(path) for key, path in files.items()}},
    )
    return {**files, "sentiment__checkpoint": checkpoint_path}


def _completed_result_hashes(
    output_root: Path, identity: OpeningIdentity
) -> dict[str, str] | None:
    manifest_path = output_root / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("opening_identity") != identity.to_dict():
        raise ValueError("existing final manifest uses a different opening identity")
    hashes: dict[str, str] = {}
    resolved_root = output_root.resolve(strict=True)
    for key, entry in dict(manifest.get("artifact_hashes", {})).items():
        registered = Path(str(entry["path"]))
        if registered.is_absolute():
            raise ValueError(f"completed result artifact path is not portable: {key}")
        path = (resolved_root / registered).resolve(strict=True)
        try:
            path.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(
                f"completed result artifact escapes the result root: {key}"
            ) from error
        actual = _sha256(path)
        if actual != entry.get("sha256"):
            raise ValueError(f"completed result artifact hash changed: {key}")
        hashes[str(key)] = actual
    hashes["manifest"] = _sha256(manifest_path)
    return hashes


def _utc(value: str | pd.Timestamp, *, label: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return stamp.tz_convert("UTC")


def _stream_candidates(
    protocol: LockboxProtocol, stream: str
) -> tuple[CandidateSpec, ...]:
    candidates = tuple(
        candidate for candidate in protocol.candidates if candidate.stream == stream
    )
    if len(candidates) != 2:
        raise ValueError(f"{stream} must have exactly two registered candidates")
    return candidates


def _validate_prediction_registry(
    predictions: Mapping[str, pd.DataFrame], candidates: Sequence[CandidateSpec]
) -> None:
    registered = {candidate.candidate_id for candidate in candidates}
    unknown = set(predictions).difference(registered)
    if unknown:
        raise ValueError(f"prediction contains an unregistered candidate: {sorted(unknown)}")
    if not predictions:
        raise ValueError("prediction registry is empty")


def _validate_exact_candidate_ids(
    protocol: LockboxProtocol, candidate_ids: Sequence[str] | None
) -> tuple[str, ...]:
    registered = tuple(candidate.candidate_id for candidate in protocol.candidates)
    selected = registered if candidate_ids is None else tuple(candidate_ids)
    if selected != registered:
        raise ValueError("candidate IDs differ from the exact registered candidate order")
    return selected


def _validate_features(features: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(features.index, pd.DatetimeIndex) or features.index.tz is None:
        raise ValueError("feature index must be timezone-aware")
    if features.empty or features.index.has_duplicates or not features.index.is_monotonic_increasing:
        raise ValueError("features must be non-empty, unique and sorted")
    current = features.astype(float)
    if not np.isfinite(current.to_numpy()).all():
        raise ValueError("features must be finite")
    return current


def _probability_panel(model, features: pd.DataFrame) -> pd.DataFrame:
    probability = _aligned_proba(model, features)
    if probability.shape != (len(features), 3) or not np.isfinite(probability).all():
        raise ValueError("model returned invalid three-class probabilities")
    if not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-7):
        raise ValueError("model probabilities are not normalized")
    predicted = probability.argmax(axis=1).astype(int)
    return pd.DataFrame(
        {
            "pred": predicted,
            "confidence": probability.max(axis=1),
            "p_short": probability[:, 0],
            "p_flat": probability[:, 1],
            "p_long": probability[:, 2],
        },
        index=features.index,
    )


def _signal(panel: pd.DataFrame, tau: float) -> pd.Series:
    output = panel["pred"].map({0: -1.0, 1: 0.0, 2: 1.0}).astype(float)
    if float(tau) > 0.0:
        output = output.where(panel["confidence"].ge(float(tau)), 0.0)
    return output.rename("signal")


def _fit_key(candidate: CandidateSpec, model_name: str, width_bps: int) -> str:
    return f"{candidate.stream}:{candidate.arm}:{model_name}:w{int(width_bps)}"


def _score_members(
    candidates: Sequence[CandidateSpec],
    features: pd.DataFrame,
    estimators: Mapping[str, object],
) -> dict[str, pd.DataFrame]:
    required = {
        _fit_key(candidate, member.model_name, member.width_bps)
        for candidate in candidates
        for member in candidate.members
    }
    missing = required.difference(estimators)
    extra = set(estimators).difference(required)
    if missing or extra:
        raise ValueError(
            f"estimator registry differs from frozen members; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return {key: _probability_panel(estimators[key], features) for key in sorted(required)}


def _scope_panel(
    panel: pd.DataFrame, *, start: str | pd.Timestamp, end: str | pd.Timestamp
) -> pd.DataFrame:
    start_utc, end_utc = _utc(start, label="start"), _utc(end, label="end")
    if end_utc <= start_utc:
        raise ValueError("end must be after start")
    scoped = panel.loc[(panel.index >= start_utc) & (panel.index < end_utc)].copy()
    if scoped.empty:
        raise ValueError("candidate scoring produced no rows in the interval")
    return scoped


def score_btc_candidates(
    features: pd.DataFrame,
    estimators: Mapping[str, object],
    protocol: LockboxProtocol,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Score the frozen BTC LSTM control and opposite-signal-veto Union."""
    current = _validate_features(features)
    candidates = _stream_candidates(protocol, "btcusdt")
    member_panels = _score_members(candidates, current, estimators)
    output: dict[str, pd.DataFrame] = {}
    for candidate in candidates:
        if candidate.combiner is None:
            member = candidate.members[0]
            panel = member_panels[_fit_key(candidate, member.model_name, member.width_bps)].copy()
            panel["signal"] = _signal(panel, candidate.decision_tau)
        elif candidate.combiner == "opposite_signal_veto":
            signals = []
            panel = pd.DataFrame(index=current.index)
            for member in candidate.members:
                member_panel = member_panels[
                    _fit_key(candidate, member.model_name, member.width_bps)
                ]
                signals.append(
                    _signal(member_panel, float(member.signal_tau)).rename(
                        f"{member.model_name}_signal"
                    )
                )
            votes = pd.concat(signals, axis=1)
            net = votes.sum(axis=1)
            active = votes.ne(0.0).sum(axis=1)
            combined = np.sign(net)
            combined[net.abs().ne(active)] = 0.0
            panel = panel.join(votes)
            panel["signal"] = combined.astype(float)
        else:
            raise ValueError(f"unsupported BTC combiner: {candidate.combiner}")
        output[candidate.candidate_id] = _scope_panel(panel, start=start, end=end)
    return output


def score_index_candidates(
    stream: str,
    features: pd.DataFrame,
    estimators: Mapping[str, object],
    protocol: LockboxProtocol,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Score one index's registered best single and arithmetic all-nine vote."""
    if stream not in {"usa500", "usatech"}:
        raise ValueError("index stream must be usa500 or usatech")
    current = _validate_features(features)
    candidates = _stream_candidates(protocol, stream)
    member_panels = _score_members(candidates, current, estimators)
    output: dict[str, pd.DataFrame] = {}
    for candidate in candidates:
        if candidate.combiner is None:
            member = candidate.members[0]
            panel = member_panels[_fit_key(candidate, member.model_name, member.width_bps)].copy()
        elif candidate.combiner == "soft_vote":
            probabilities = np.stack(
                [
                    member_panels[_fit_key(candidate, member.model_name, member.width_bps)][
                        list(PROBABILITY_COLUMNS)
                    ].to_numpy(float)
                    for member in candidate.members
                ],
                axis=0,
            ).mean(axis=0)
            panel = pd.DataFrame(
                probabilities, index=current.index, columns=PROBABILITY_COLUMNS
            )
            panel["pred"] = probabilities.argmax(axis=1).astype(int)
            panel["confidence"] = probabilities.max(axis=1)
            panel = panel[["pred", "confidence", *PROBABILITY_COLUMNS]]
        else:
            raise ValueError(f"unsupported index combiner: {candidate.combiner}")
        panel["signal"] = _signal(panel, candidate.decision_tau)
        output[candidate.candidate_id] = _scope_panel(panel, start=start, end=end)
    return output


def _validate_signal_frame(
    frame: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    if "signal" not in frame:
        raise ValueError("candidate prediction requires signal")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("candidate signal index must be timezone-aware")
    index = frame.index.tz_convert("UTC")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("candidate signals must be unique and sorted")
    if (index < start).any() or (index >= end).any():
        raise ValueError("candidate signals cross the replay interval; carry-in is forbidden")
    signal = pd.Series(
        pd.to_numeric(frame["signal"], errors="raise").to_numpy(float),
        index=index,
        name="signal",
    )
    if not signal.isin((-1.0, 0.0, 1.0)).all():
        raise ValueError("candidate signals must be -1, 0 or 1")
    return signal


def _empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "entry_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "exit_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "side": pd.Series(dtype="int64"),
            "entry_price": pd.Series(dtype="float64"),
            "exit_price": pd.Series(dtype="float64"),
            "gross_return": pd.Series(dtype="float64"),
            "cost_return": pd.Series(dtype="float64"),
            "net_return": pd.Series(dtype="float64"),
        }
    )


def _terminal_mask(signal: pd.Series, end: pd.Timestamp) -> pd.Series:
    return signal.ne(0.0) & (signal.index + 2 * BAR_SIZE >= end)


def replay_index_candidates(
    stream: str,
    bars: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    protocol: LockboxProtocol,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> dict[str, ReplayResult]:
    """Replay next-consecutive-M15 open-to-close trades for one index."""
    candidates = _stream_candidates(protocol, stream)
    _validate_prediction_registry(predictions, candidates)
    start_utc, end_utc = _utc(start, label="start"), _utc(end, label="end")
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError("index bars must have a timezone-aware index")
    market = bars.copy().sort_index()
    market.index = market.index.tz_convert("UTC")
    if market.index.has_duplicates or (market.index < start_utc).any() or (market.index >= end_utc).any():
        raise ValueError("index bars cross the exact replay interval")
    required = {"open", "close", "complete_bar", "available_at"}
    if not required.issubset(market):
        raise ValueError(f"index bars miss columns: {sorted(required - set(market))}")
    market = market.loc[market["complete_bar"].astype(bool)]
    cost = protocol.costs[stream].round_trip_bps / 10_000.0
    output: dict[str, ReplayResult] = {}
    for candidate in candidates:
        if candidate.candidate_id not in predictions:
            continue
        signal = _validate_signal_frame(
            predictions[candidate.candidate_id], start=start_utc, end=end_utc
        )
        terminal = _terminal_mask(signal, end_utc)
        records: list[dict[str, object]] = []
        gap_censored = 0
        for signal_time, side in signal.loc[signal.ne(0.0) & ~terminal].items():
            entry_time = signal_time + BAR_SIZE
            exit_time = entry_time + BAR_SIZE
            if entry_time not in market.index:
                gap_censored += 1
                continue
            entry_bar = market.loc[entry_time]
            entry_price = float(entry_bar["open"])
            exit_price = float(entry_bar["close"])
            gross = float(side) * (exit_price / entry_price - 1.0)
            records.append(
                {
                    "signal_time": signal_time,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "side": int(side),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_return": gross,
                    "cost_return": float(cost),
                    "net_return": gross - float(cost),
                }
            )
        ledger = pd.DataFrame(records) if records else _empty_ledger()
        per_bar = pd.Series(0.0, index=market.index, name="net_return")
        if len(ledger):
            per_bar.loc[pd.to_datetime(ledger["entry_time"], utc=True)] = ledger[
                "net_return"
            ].to_numpy(float)
        audit = ReplayAudit(
            candidate_decisions=int(signal.ne(0.0).sum()),
            executed_decisions=int(len(ledger)),
            terminal_censored_count=int(terminal.sum()),
            session_gap_censored_count=int(gap_censored),
            nonoverlap_censored_count=0,
        )
        output[candidate.candidate_id] = ReplayResult(
            candidate.candidate_id, ledger, per_bar, audit
        )
    return output


def replay_btc_candidates(
    bars: pd.DataFrame,
    minute_bars: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    protocol: LockboxProtocol,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> dict[str, ReplayResult]:
    """Replay BTC TP200/SL100/one-M15 trades from a flat Q2 opening state."""
    candidates = _stream_candidates(protocol, "btcusdt")
    _validate_prediction_registry(predictions, candidates)
    start_utc, end_utc = _utc(start, label="start"), _utc(end, label="end")
    market = bars.copy().sort_index()
    market.index = pd.to_datetime(market.index, utc=True)
    minute = minute_bars.copy().sort_index()
    minute.index = pd.to_datetime(minute.index, utc=True)
    if market.index.has_duplicates or (market.index < start_utc).any() or (market.index >= end_utc).any():
        raise ValueError("BTC bars cross the exact replay interval")
    if minute.index.has_duplicates or (minute.index < start_utc).any() or (minute.index >= end_utc).any():
        raise ValueError("BTC minute bars cross the exact replay interval")
    fee_per_side = protocol.costs["btcusdt"].per_side_bps
    output: dict[str, ReplayResult] = {}
    for candidate in candidates:
        if candidate.candidate_id not in predictions:
            continue
        signal = _validate_signal_frame(
            predictions[candidate.candidate_id], start=start_utc, end=end_utc
        )
        terminal = _terminal_mask(signal, end_utc)
        executable = signal.mask(terminal, 0.0)
        prediction = executable.map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
        ledger, per_bar = simulate_bracket_trades_intrabar(
            market,
            minute,
            prediction,
            None,
            tau=0.0,
            tp_bps=BTC_TP_BPS,
            sl_bps=BTC_SL_BPS,
            max_hold=BTC_MAX_HOLD,
            fee_bps=fee_per_side,
            expected_interval=pd.Timedelta(minutes=1),
            include_audit=True,
        )
        if len(ledger):
            ledger.insert(
                0,
                "signal_time",
                pd.to_datetime(ledger["entry_time"], utc=True) - BAR_SIZE,
            )
            ledger["cost_return"] = (
                ledger["gross_return"].astype(float) - ledger["net_return"].astype(float)
            )
        else:
            ledger = _empty_ledger()
        eligible = int((signal.ne(0.0) & ~terminal).sum())
        nonoverlap_censored = eligible - len(ledger)
        audit = ReplayAudit(
            candidate_decisions=int(signal.ne(0.0).sum()),
            executed_decisions=int(len(ledger)),
            terminal_censored_count=int(terminal.sum()),
            session_gap_censored_count=0,
            nonoverlap_censored_count=int(nonoverlap_censored),
        )
        output[candidate.candidate_id] = ReplayResult(
            candidate.candidate_id, ledger, per_bar.rename("net_return"), audit
        )
    return output


def _verify_identity_entry(label: str, entry: Mapping[str, object]) -> None:
    path = _manifest_path(entry)
    if not path.is_file():
        raise FileNotFoundError(path)
    if int(entry.get("size", -1)) != path.stat().st_size:
        raise ValueError(f"{label} size differs from the pre-lockbox manifest")
    if str(entry.get("sha256", "")) != _sha256(path):
        raise ValueError(f"{label} hash differs from the pre-lockbox manifest")


def validate_prelockbox_manifest_files(
    manifest_path: str | Path,
) -> dict[str, object]:
    """Re-hash every bound file without decoding a Q2 payload."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("q2_decoded") is not False:
        raise ValueError("pre-lockbox manifest must state q2_decoded=false")
    protocol = manifest.get("protocol", {})
    _verify_identity_entry(
        "protocol",
        {
            "path": protocol.get("path"),
            "size": Path(str(protocol.get("path"))).stat().st_size,
            "sha256": protocol.get("sha256"),
        },
    )
    for section in ("q2_sources", "warmups", "critical_sources"):
        entries = manifest.get(section)
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"pre-lockbox manifest section is empty: {section}")
        for label, entry in entries.items():
            _verify_identity_entry(f"{section}.{label}", entry)
    reconstructions = manifest.get("reconstructed_estimators")
    if not isinstance(reconstructions, list) or len(reconstructions) != 21:
        raise ValueError("pre-lockbox manifest must bind 21 reconstructed estimators")
    for entry in reconstructions:
        for label, path_key, hash_key in (
            ("manifest", "manifest", "manifest_sha256"),
            ("estimator", "serialized_estimator", "serialized_estimator_sha256"),
            ("rebuilt panel", "rebuilt_panel", "rebuilt_panel_sha256"),
        ):
            target = Path(str(entry[path_key]))
            if not target.is_file() or _sha256(target) != str(entry[hash_key]):
                raise ValueError(f"reconstruction {label} changed: {entry.get('fit_key')}")
    identities = manifest.get("model_identities", {})
    deberta_files = dict(identities.get("deberta", {}).get("files", {}))
    if len(deberta_files) != 7:
        raise ValueError("DeBERTa identity must bind seven snapshot files")
    for name, entry in deberta_files.items():
        _verify_identity_entry(f"deberta.{name}", entry)
    deepseek_file = identities.get("deepseek", {}).get("identity_file")
    if not isinstance(deepseek_file, dict):
        raise ValueError("DeepSeek identity file is not bound")
    _verify_identity_entry("deepseek.identity_file", deepseek_file)
    return manifest


def _git_text(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY_ROOT, text=True, encoding="utf-8"
    ).strip()


def _assert_annotated_tag(tag_name: str) -> None:
    if _git_text("cat-file", "-t", f"refs/tags/{tag_name}") != "tag":
        raise ValueError("pre-lockbox tag must be an annotated tag object")


def opening_identity_from_manifest(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    tag_name: str = PRELOCKBOX_TAG,
) -> OpeningIdentity:
    """Validate the tagged manifest-only commit and return its immutable identity."""
    from experiments.final_q2_lockbox_preflight import validate_manifest_commit_binding

    path = Path(manifest_path)
    manifest = validate_prelockbox_manifest_files(path)
    _assert_annotated_tag(tag_name)
    manifest_commit = _git_text("rev-parse", f"{tag_name}^{{commit}}")
    parent_commit = _git_text("rev-parse", f"{manifest_commit}^")
    relative = path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    committed_bytes = subprocess.check_output(
        ["git", "show", f"{manifest_commit}:{relative}"], cwd=REPOSITORY_ROOT
    )
    binding = validate_manifest_commit_binding(
        path,
        manifest_commit=manifest_commit,
        parent_commit=parent_commit,
        committed_manifest_bytes=committed_bytes,
    )
    if _git_text("rev-parse", "HEAD") != binding.manifest_commit:
        raise ValueError("HEAD must equal the tagged manifest commit before Q2 opens")
    source_hashes = {
        str(key): str(entry["sha256"])
        for key, entry in dict(manifest["q2_sources"]).items()
    }
    return OpeningIdentity(
        implementation_commit=binding.implementation_commit,
        manifest_commit=binding.manifest_commit,
        protocol_hash=str(manifest["protocol"]["protocol_hash"]),
        manifest_sha256=binding.manifest_sha256,
        q2_source_hashes=source_hashes,
    ).validate()


def _bound_path(manifest: Mapping[str, object], section: str, key: str) -> Path:
    entries = dict(manifest[section])
    if key not in entries:
        raise ValueError(f"pre-lockbox manifest misses {section}.{key}")
    return Path(str(entries[key]["path"]))


def load_reconstructed_estimators(
    manifest: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Load all 21 hash-bound estimators and their frozen feature orders."""
    import torch

    from experiments.final_q2_lockbox_reconstruction import load_serialized_estimator

    torch.set_num_threads(16)
    estimators: dict[str, object] = {}
    metadata: dict[str, dict[str, object]] = {}
    for entry in manifest["reconstructed_estimators"]:
        fit_key = str(entry["fit_key"])
        manifest_path = Path(str(entry["manifest"]))
        reconstruction = json.loads(manifest_path.read_text(encoding="utf-8"))
        estimator = load_serialized_estimator(
            entry["serialized_estimator"],
            expected_sha256=str(entry["serialized_estimator_sha256"]),
        )
        if fit_key in estimators:
            raise ValueError(f"duplicate reconstructed fit: {fit_key}")
        estimators[fit_key] = estimator
        metadata[fit_key] = reconstruction
    if len(estimators) != 21:
        raise ValueError("lockbox execution requires exactly 21 loaded estimators")
    return estimators, metadata


def _feature_order_for_stream(
    metadata: Mapping[str, Mapping[str, object]], stream: str
) -> tuple[str, ...]:
    orders = {
        tuple(value["feature_columns"])
        for key, value in metadata.items()
        if key.startswith(f"{stream}:")
    }
    if len(orders) != 1:
        raise ValueError(f"{stream} reconstructed feature orders are inconsistent")
    return next(iter(orders))


def _combine_context(warmup: pd.DataFrame, q2: pd.DataFrame, *, label: str) -> pd.DataFrame:
    left = warmup.copy()
    right = q2.copy()
    left.index = pd.to_datetime(left.index, utc=True)
    right.index = pd.to_datetime(right.index, utc=True)
    if left.empty or right.empty or left.index.max() >= right.index.min():
        raise ValueError(f"{label} warm-up and Q2 partitions overlap or are empty")
    combined = pd.concat([left, right], axis=0, sort=False).sort_index()
    if combined.index.has_duplicates or not combined.index.is_monotonic_increasing:
        raise ValueError(f"{label} context timestamps are not unique and sorted")
    return combined


def _build_btc_execution_inputs(
    identity: OpeningIdentity,
    manifest: Mapping[str, object],
    metadata: Mapping[str, Mapping[str, object]],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    from experiments.final_q2_lockbox_inputs import (
        load_exact_q2_parquet,
        load_q2_parquet_partition,
    )
    from features.build import add_features

    q2_bars = load_exact_q2_parquet(
        _bound_path(manifest, "q2_sources", "btc_m15_q2"),
        identity=identity,
        start=start,
        end=end,
    )
    q2_minute = load_q2_parquet_partition(
        _bound_path(manifest, "q2_sources", "btc_m1_q2_source"),
        identity=identity,
        start=start,
        end=end,
    )
    q2_positioning = load_q2_parquet_partition(
        _bound_path(manifest, "q2_sources", "btc_positioning_q2_source"),
        identity=identity,
        start=start,
        end=end,
    )
    warm_bars = pd.read_parquet(_bound_path(manifest, "warmups", "btc_m15"))
    warm_positioning = pd.read_parquet(
        _bound_path(manifest, "warmups", "btc_positioning")
    )
    reference_features = pd.read_parquet(
        _bound_path(manifest, "warmups", "btc_features")
    )
    bars = _combine_context(warm_bars, q2_bars, label="BTC M15")
    positioning = _combine_context(
        warm_positioning, q2_positioning, label="BTC positioning"
    )
    joined = bars.join(positioning.reindex(bars.index))
    columns = _feature_order_for_stream(metadata, "btcusdt")
    features = add_features(joined).loc[:, list(columns)].dropna()
    reference_features.index = pd.to_datetime(reference_features.index, utc=True)
    rebuilt_reference = features.reindex(reference_features.index)
    if (
        tuple(rebuilt_reference.columns) != tuple(reference_features.columns)
        or rebuilt_reference.isna().any().any()
        or not np.allclose(
            rebuilt_reference.to_numpy(float),
            reference_features.to_numpy(float),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError("BTC pre-Q2 feature warm-up no longer reproduces")
    if features.loc[(features.index >= start) & (features.index < end)].empty:
        raise ValueError("BTC Q2 feature partition is empty")
    audit = {
        "m15_rows": int(len(q2_bars)),
        "m1_rows": int(len(q2_minute)),
        "positioning_rows": int(len(q2_positioning)),
        "feature_rows": int(((features.index >= start) & (features.index < end)).sum()),
        "feature_columns": list(columns),
        "warmup_max": reference_features.index.max().isoformat(),
        "q2_min": q2_bars.index.min().isoformat(),
        "q2_max": q2_bars.index.max().isoformat(),
        "positioning_stale_share": float(
            q2_positioning.get("positioning_stale", pd.Series(False, index=q2_positioning.index))
            .astype(bool)
            .mean()
        ),
    }
    return q2_bars, q2_minute, features, audit


def _score_q2_sentiment(
    identity: OpeningIdentity,
    score_root: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, str]:
    from sentiment.index_scoring import (
        DEBERTA_REVISION,
        load_saved_deepseek_identity,
        score_deberta_index,
        score_deepseek_index,
    )
    from sentiment.score import _build_scorer

    raw = CODE_ROOT / "sentiment" / "raw"
    score_root.mkdir(parents=True, exist_ok=True)
    deberta_scorer = _build_scorer(
        64, revision=DEBERTA_REVISION, local_files_only=True
    )
    deepseek_identity = load_saved_deepseek_identity()
    outputs: dict[str, str] = {}
    for stream in ("usa500", "usatech"):
        for prefix in ("gdelt", "direct_events"):
            label = f"{stream}_{prefix}"
            print(f"[sentiment] DeBERTa {label}", flush=True)
            classic = score_deberta_index(
                stream,
                source_prefix=prefix,
                raw_dir=raw,
                output_dir=score_root,
                start_inclusive=start,
                end_exclusive=end,
                opening_identity=identity,
                scorer=deberta_scorer,
            )
            print(f"[sentiment] LLM {label}", flush=True)
            llm = score_deepseek_index(
                stream,
                identity=deepseek_identity,
                source_prefix=prefix,
                raw_dir=raw,
                output_dir=score_root,
                start_inclusive=start,
                end_exclusive=end,
                opening_identity=identity,
                workers=2,
            )
            outputs[f"deberta_{label}"] = classic.as_posix()
            outputs[f"llm_{label}"] = llm.as_posix()
    return outputs


def _build_index_execution_inputs(
    stream: str,
    identity: OpeningIdentity,
    manifest: Mapping[str, object],
    metadata: Mapping[str, Mapping[str, object]],
    score_root: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    from data.index_market import resample_index_minutes
    from experiments.final_q2_lockbox_inputs import load_q2_index_minutes
    from experiments.index_replication_protocol import (
        VIX_FEATURE_COLS,
        build_vix_block,
        join_completed_vix,
    )
    from features.build import FEATURE_COLS, add_features
    from features.index_sentiment import build_matched_index_features

    instruments = {"usa500": "USA500IDXUSD", "usatech": "USATECHIDXUSD"}
    minute = load_q2_index_minutes(
        _bound_path(manifest, "q2_sources", f"{stream}_bid_q2_source"),
        _bound_path(manifest, "q2_sources", f"{stream}_ask_q2_source"),
        identity=identity,
        start=start,
        end=end,
        instrument=instruments[stream],
    )
    vix_minute = load_q2_index_minutes(
        _bound_path(manifest, "q2_sources", "volidx_bid_q2_source"),
        _bound_path(manifest, "q2_sources", "volidx_ask_q2_source"),
        identity=identity,
        start=start,
        end=end,
        instrument="VOLIDXUSD",
    )
    q2_bars = resample_index_minutes(minute, "15min")
    q2_bars = q2_bars.loc[q2_bars["complete_bar"].astype(bool)]
    q2_vix = resample_index_minutes(vix_minute, "15min")
    q2_vix = q2_vix.loc[q2_vix["complete_bar"].astype(bool)]
    warm_bars = pd.read_parquet(
        _bound_path(manifest, "warmups", f"{stream}_15min")
    )
    warm_vix = pd.read_parquet(
        _bound_path(manifest, "warmups", "volidx_15min")
    )
    full_bars = _combine_context(warm_bars, q2_bars, label=f"{stream} M15")
    full_vix = _combine_context(warm_vix, q2_vix, label="VIX M15")
    engineered = add_features(full_bars)
    price = engineered.loc[:, FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    decisions = full_bars[["close"]].copy()
    decisions["decision_time"] = pd.to_datetime(full_bars["available_at"], utc=True)
    vix = join_completed_vix(decisions, build_vix_block(full_vix))
    price_vix = pd.concat(
        [price, vix.loc[:, list(VIX_FEATURE_COLS)]], axis=1, sort=False
    )
    q2_index = q2_bars.index.intersection(price_vix.dropna().index)
    scorer = "classic" if stream == "usa500" else "llm"
    sentiment = build_matched_index_features(
        stream,
        q2_index,
        scorer=scorer,
        score_root=score_root,
        warmup_score_root=_bound_path(
            manifest, "warmups", f"scores_{stream}"
        ).parent,
        source_root=CODE_ROOT / "sentiment" / "raw",
        available_start=start,
        available_end=end,
        opening_identity=identity,
        continuous_context=True,
    )
    q2_features = pd.concat(
        [price_vix.reindex(q2_index), sentiment], axis=1, sort=False
    ).dropna()
    columns = _feature_order_for_stream(metadata, stream)
    if tuple(q2_features.columns) != columns:
        raise ValueError(f"{stream} Q2 feature order differs from reconstruction")
    warm_features = pd.read_parquet(
        _bound_path(manifest, "warmups", f"{stream}_features")
    )
    full_features = _combine_context(
        warm_features, q2_features, label=f"{stream} model features"
    )
    audit = {
        "m1_rows": int(len(minute)),
        "m15_complete_rows": int(len(q2_bars)),
        "vix_m1_rows": int(len(vix_minute)),
        "vix_m15_complete_rows": int(len(q2_vix)),
        "feature_rows": int(len(q2_features)),
        "feature_columns": list(columns),
        "warmup_max": pd.to_datetime(warm_features.index, utc=True).max().isoformat(),
        "q2_min": q2_bars.index.min().isoformat(),
        "q2_max": q2_bars.index.max().isoformat(),
    }
    return q2_bars, full_features, audit


def _stream_estimators(
    stream: str,
    protocol: LockboxProtocol,
    estimators: Mapping[str, object],
) -> dict[str, object]:
    keys = {
        _fit_key(candidate, member.model_name, member.width_bps)
        for candidate in _stream_candidates(protocol, stream)
        for member in candidate.members
    }
    return {key: estimators[key] for key in sorted(keys)}


def _write_result_artifacts(
    output_root: Path,
    protocol: LockboxProtocol,
    predictions: Mapping[str, pd.DataFrame],
    results: Mapping[str, ReplayResult],
    input_audit: Mapping[str, object],
    identity: OpeningIdentity,
    prelockbox_manifest_path: Path,
    intermediate_artifacts: Mapping[str, Path] | None = None,
) -> dict[str, str]:
    from experiments.final_q2_lockbox_metrics import (
        PRIMARY_ESTIMAND,
        daily_net_series,
        paired_weekly_bootstrap,
        summarise_candidate,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    result_paths: dict[str, Path] = {}
    summaries: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    daily_by_candidate: dict[str, pd.Series] = {}
    execution_audit: dict[str, object] = {}
    for candidate in protocol.candidates:
        candidate_id = candidate.candidate_id
        _, checkpoint_result = _persist_candidate_checkpoint(
            output_root,
            candidate_id,
            predictions[candidate_id],
            results[candidate_id],
        )
        paths = _candidate_checkpoint_paths(output_root, candidate_id)
        result_paths[f"prediction__{candidate_id}"] = paths["prediction"]
        result_paths[f"ledger__{candidate_id}"] = paths["ledger"]
        result_paths[f"candidate_audit__{candidate_id}"] = paths["audit"]
        result_paths[f"checkpoint__{candidate_id}"] = paths["checkpoint"]
        result = checkpoint_result
        daily = daily_net_series(
            result.ledger, start=protocol.start_utc, end=protocol.end_utc
        )
        daily_by_candidate[candidate_id] = daily
        daily_path = _write_parquet_once(
            output_root / "daily" / f"{candidate_id}.parquet",
            daily.rename_axis("date").reset_index(),
            index=False,
        )
        result_paths[f"daily__{candidate_id}"] = daily_path
        summary = summarise_candidate(
            result.ledger,
            candidate_id=candidate_id,
            stream=candidate.stream,
            start=protocol.start_utc,
            end=protocol.end_utc,
        )
        summary.update({"role": candidate.role, "arm": candidate.arm})
        summaries.append(summary)
        for month, value in daily.resample("MS").sum().items():
            monthly_rows.append(
                {
                    "stream": candidate.stream,
                    "candidate_id": candidate_id,
                    "month": month.strftime("%Y-%m"),
                    "net_return": float(value),
                }
            )
        execution_audit[candidate_id] = asdict(result.audit)

    summary_frame = pd.DataFrame(summaries).sort_values(
        ["stream", "net_return"], ascending=[True, False]
    )
    summary_path = _write_parquet_once(
        output_root / "summaries.parquet", summary_frame, index=False
    )
    monthly_path = _write_parquet_once(
        output_root / "monthly.parquet", pd.DataFrame(monthly_rows), index=False
    )
    result_paths.update({"summaries": summary_path, "monthly": monthly_path})

    contrast_rows: list[dict[str, object]] = []
    for stream in ("btcusdt", "usa500", "usatech"):
        candidates = _stream_candidates(protocol, stream)
        policy = next(candidate for candidate in candidates if candidate.role == "primary")
        comparator = next(candidate for candidate in candidates if candidate.role != "primary")
        estimand = (
            PRIMARY_ESTIMAND
            if stream == "btcusdt"
            else f"{stream}_primary_minus_{comparator.role}_total_net"
        )
        bootstrap = paired_weekly_bootstrap(
            daily_by_candidate[policy.candidate_id],
            daily_by_candidate[comparator.candidate_id],
            reps=protocol.bootstrap_replicates,
            seed=protocol.bootstrap_seed,
            estimand=estimand,
            start=protocol.start_utc,
            end=protocol.end_utc,
        )
        contrast_rows.append(
            {
                "stream": stream,
                "policy_id": policy.candidate_id,
                "comparator_id": comparator.candidate_id,
                **asdict(bootstrap),
            }
        )
    contrasts = pd.DataFrame(contrast_rows)
    contrast_path = _write_parquet_once(
        output_root / "paired_contrasts.parquet", contrasts, index=False
    )
    result_paths["paired_contrasts"] = contrast_path

    input_path = _write_json_once(output_root / "input_audit.json", input_audit)
    execution_path = _write_json_once(
        output_root / "execution_audit.json", execution_audit
    )
    leakage = {
        "no_q2_fitting": True,
        "no_q2_calibration": True,
        "no_q2_threshold_selection": True,
        "started_flat": all(audit["started_flat"] for audit in execution_audit.values()),
        "warmup_strictly_pre_q2": all(
            pd.Timestamp(details["warmup_max"]) < protocol.start_utc
            for details in input_audit["streams"].values()
        ),
        "prediction_bounds_valid": all(
            frame.index.min() >= protocol.start_utc
            and frame.index.max() < protocol.end_utc
            for frame in predictions.values()
        ),
        "registered_candidates_only": set(predictions)
        == {candidate.candidate_id for candidate in protocol.candidates},
    }
    if not all(leakage.values()):
        raise AssertionError(f"post-open leakage audit failed: {leakage}")
    leakage_path = _write_json_once(output_root / "leakage_audit.json", leakage)
    result_paths.update(
        {
            "input_audit": input_path,
            "execution_audit": execution_path,
            "leakage_audit": leakage_path,
        }
    )

    btc = contrasts.loc[contrasts["stream"].eq("btcusdt")].iloc[0]
    verdict = {
        "primary_estimand": PRIMARY_ESTIMAND,
        "point_estimate": float(btc["point_estimate"]),
        "bootstrap_95_lower": float(btc["lower_95"]),
        "bootstrap_95_upper": float(btc["upper_95"]),
        "primary_confirmatory_support": bool(btc["primary_confirmatory_support"]),
        "interpretation": (
            "BTC Qualified Union received confirmatory support over its frozen LSTM control."
            if bool(btc["primary_confirmatory_support"])
            else "BTC Qualified Union did not receive confirmatory support over its frozen LSTM control."
        ),
        "index_status": "descriptive transport evidence only; no Q2 promotion is permitted",
        "final_no_retuning": True,
    }
    verdict_path = _write_json_once(output_root / "verdict.json", verdict)
    result_paths["verdict"] = verdict_path
    result_paths.update(dict(intermediate_artifacts or {}))
    resolved_output_root = output_root.resolve(strict=True)
    artifact_hashes = {}
    for key, path in sorted(result_paths.items()):
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(resolved_output_root)
        artifact_hashes[key] = {
            "path": relative.as_posix(),
            "sha256": _sha256(resolved),
        }
    final_manifest = {
        "schema_version": "final-q2-lockbox-result-v1",
        "state": "COMPLETE",
        "opening_identity": identity.to_dict(),
        "prelockbox_manifest": {
            "path": prelockbox_manifest_path.as_posix(),
            "sha256": _sha256(prelockbox_manifest_path),
        },
        "interval": [protocol.start_utc.isoformat(), protocol.end_utc.isoformat()],
        "candidate_ids": [candidate.candidate_id for candidate in protocol.candidates],
        "artifact_hashes": artifact_hashes,
        "primary_confirmatory_support": verdict["primary_confirmatory_support"],
        "no_retuning": True,
    }
    final_manifest_path = _write_json_once(
        output_root / "manifest.json", final_manifest
    )
    return {
        **{key: entry["sha256"] for key, entry in artifact_hashes.items()},
        "manifest": _sha256(final_manifest_path),
    }


def run_registered_q2(
    identity: OpeningIdentity,
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_root: str | Path | None = None,
) -> Mapping[str, str]:
    """Decode, score and replay the six frozen candidates without fitting."""
    from experiments.final_q2_lockbox_contract import load_lockbox_protocol

    require_global_opening(identity)
    prelockbox_path = Path(manifest_path)
    manifest = validate_prelockbox_manifest_files(prelockbox_path)
    if _sha256(prelockbox_path) != identity.manifest_sha256:
        raise ValueError("pre-lockbox manifest hash differs from OPENED identity")
    if {
        key: entry["sha256"] for key, entry in manifest["q2_sources"].items()
    } != dict(identity.q2_source_hashes):
        raise ValueError("Q2 source registry differs from OPENED identity")
    protocol = load_lockbox_protocol(protocol_path)
    if manifest["protocol"]["protocol_hash"] != identity.protocol_hash:
        raise ValueError("protocol hash differs from OPENED identity")
    destination = (
        GLOBAL_STATE_ROOT / identity.protocol_hash
        if output_root is None
        else Path(output_root)
    )
    completed = _completed_result_hashes(destination, identity)
    if completed is not None:
        return completed
    start, end = protocol.start_utc, protocol.end_utc
    predictions: dict[str, pd.DataFrame] = {}
    results: dict[str, ReplayResult] = {}
    registered_ids = {candidate.candidate_id for candidate in protocol.candidates}
    for candidate_id in sorted(registered_ids):
        checkpoint = _candidate_checkpoint_paths(destination, candidate_id)["checkpoint"]
        if checkpoint.exists():
            prediction, result = _load_candidate_checkpoint(destination, candidate_id)
            predictions[candidate_id] = prediction
            results[candidate_id] = result
    missing = registered_ids.difference(predictions)
    estimators: dict[str, object] = {}
    metadata: dict[str, dict[str, object]] = {}
    if missing:
        estimators, metadata = load_reconstructed_estimators(manifest)

    stream_audits: dict[str, object] = {}
    btc_ids = {
        candidate.candidate_id
        for candidate in _stream_candidates(protocol, "btcusdt")
    }
    if missing.intersection(btc_ids):
        print("[Q2 1/4] decoding BTC and building frozen features", flush=True)
        btc_bars, btc_minute, btc_features, btc_audit = _build_btc_execution_inputs(
            identity, manifest, metadata, start=start, end=end
        )
        _persist_stream_audit(destination, "btcusdt", btc_audit)
        scored = score_btc_candidates(
            btc_features,
            _stream_estimators("btcusdt", protocol, estimators),
            protocol,
            start=start,
            end=end,
        )
        replayed = replay_btc_candidates(
            btc_bars, btc_minute, scored, protocol, start=start, end=end
        )
        for candidate_id in sorted(missing.intersection(btc_ids)):
            prediction, result = _persist_candidate_checkpoint(
                destination, candidate_id, scored[candidate_id], replayed[candidate_id]
            )
            predictions[candidate_id] = prediction
            results[candidate_id] = result
        stream_audits["btcusdt"] = dict(btc_audit)
    else:
        stream_audits["btcusdt"] = _load_stream_audit(destination, "btcusdt")

    score_root = destination / "sentiment_scores"
    index_ids = {
        candidate.candidate_id
        for stream in ("usa500", "usatech")
        for candidate in _stream_candidates(protocol, stream)
    }
    sentiment_artifacts = _load_sentiment_checkpoint(score_root)
    if missing.intersection(index_ids) and sentiment_artifacts is None:
        print("[Q2 2/4] scoring frozen DeBERTa and LLM sentiment", flush=True)
        _score_q2_sentiment(identity, score_root, start=start, end=end)
        sentiment_artifacts = _persist_sentiment_checkpoint(score_root)
    if sentiment_artifacts is None:
        raise ValueError("index candidate checkpoints miss their sentiment checkpoint")

    for number, stream in enumerate(("usa500", "usatech"), start=3):
        stream_ids = {
            candidate.candidate_id for candidate in _stream_candidates(protocol, stream)
        }
        if missing.intersection(stream_ids):
            print(f"[Q2 {number}/4] decoding and replaying {stream}", flush=True)
            bars, features, audit = _build_index_execution_inputs(
                stream,
                identity,
                manifest,
                metadata,
                score_root,
                start=start,
                end=end,
            )
            _persist_stream_audit(destination, stream, audit)
            scored = score_index_candidates(
                stream,
                features,
                _stream_estimators(stream, protocol, estimators),
                protocol,
                start=start,
                end=end,
            )
            replayed = replay_index_candidates(
                stream, bars, scored, protocol, start=start, end=end
            )
            for candidate_id in sorted(missing.intersection(stream_ids)):
                prediction, result = _persist_candidate_checkpoint(
                    destination,
                    candidate_id,
                    scored[candidate_id],
                    replayed[candidate_id],
                )
                predictions[candidate_id] = prediction
                results[candidate_id] = result
            stream_audits[stream] = dict(audit)
        else:
            stream_audits[stream] = _load_stream_audit(destination, stream)

    if set(predictions) != registered_ids:
        raise AssertionError("Q2 execution did not produce all six registered candidates")
    intermediate_artifacts = {
        **sentiment_artifacts,
        **{
            f"stream_audit__{stream}": destination
            / "stream_audits"
            / f"{stream}.json"
            for stream in ("btcusdt", "usa500", "usatech")
        },
    }
    input_audit = {
        "interval": [start.isoformat(), end.isoformat()],
        "streams": stream_audits,
        "sentiment_outputs": {
            key: path.resolve(strict=True)
            .relative_to(destination.resolve(strict=True))
            .as_posix()
            for key, path in sentiment_artifacts.items()
        },
        "source_hashes_reverified": True,
    }
    return _write_result_artifacts(
        destination,
        protocol,
        predictions,
        results,
        input_audit,
        identity,
        prelockbox_path,
        intermediate_artifacts=intermediate_artifacts,
    )


def open_q2(
    identity: OpeningIdentity,
    protocol: LockboxProtocol,
    *,
    authorization: str,
    run: Callable[[OpeningIdentity], Mapping[str, str]],
    root: str | Path | None = None,
    candidate_ids: Sequence[str] | None = None,
) -> Mapping[str, str]:
    """Create the irreversible sentinel and execute all registered candidates once."""
    if authorization != OPEN_AUTHORIZATION:
        raise PermissionError("literal Q2 opening authorization is required")
    _validate_exact_candidate_ids(protocol, candidate_ids)
    begin_global_open(identity, root=root)
    try:
        result_hashes = dict(run(identity))
        mark_complete(identity, result_hashes, root=root)
        return result_hashes
    except Exception as exc:
        mark_failed_after_open(identity, str(exc), root=root)
        raise


def resume_q2(
    identity: OpeningIdentity,
    protocol: LockboxProtocol,
    *,
    run: Callable[[OpeningIdentity], Mapping[str, str]],
    root: str | Path | None = None,
    candidate_ids: Sequence[str] | None = None,
) -> Mapping[str, str]:
    """Resume only the same immutable opening identity and candidate registry."""
    _validate_exact_candidate_ids(protocol, candidate_ids)
    assert_exact_resume(identity, root=root)
    try:
        result_hashes = dict(run(identity))
        mark_complete(identity, result_hashes, root=root)
        return result_hashes
    except Exception as exc:
        mark_failed_after_open(identity, str(exc), root=root)
        raise


def preflight(
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
    write_manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> Mapping[str, object]:
    from experiments.final_q2_lockbox_preflight import preflight as run_preflight

    return run_preflight(
        protocol_path=protocol_path,
        write_manifest_path=write_manifest_path,
        verify_deepseek_live=True,
        materialize_warmups=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    from experiments.final_q2_lockbox_contract import load_lockbox_protocol

    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    sealed = subcommands.add_parser("preflight")
    sealed.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    sealed.add_argument("--write-manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    opening = subcommands.add_parser("open-q2")
    opening.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    opening.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    opening.add_argument("--authorization", required=True)
    continuing = subcommands.add_parser("resume")
    continuing.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    continuing.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args(argv)

    if args.command == "preflight":
        manifest = preflight(
            protocol_path=args.protocol,
            write_manifest_path=args.write_manifest,
        )
        print(
            json.dumps(
                {
                    "state": "SEALED",
                    "q2_decoded": manifest["q2_decoded"],
                    "estimators": len(manifest["reconstructed_estimators"]),
                    "q2_sources": len(manifest["q2_sources"]),
                    "warmups": len(manifest["warmups"]),
                    "manifest": Path(args.write_manifest).as_posix(),
                },
                indent=2,
            )
        )
        return 0

    identity = opening_identity_from_manifest(args.manifest)
    protocol = load_lockbox_protocol(args.protocol)
    execute = lambda opening_identity: run_registered_q2(
        opening_identity,
        protocol_path=args.protocol,
        manifest_path=args.manifest,
    )
    if args.command == "open-q2":
        hashes = open_q2(
            identity,
            protocol,
            authorization=args.authorization,
            run=execute,
        )
    else:
        hashes = resume_q2(identity, protocol, run=execute)
    print(json.dumps({"state": "COMPLETE", "artifacts": len(hashes)}, indent=2))
    return 0


__all__ = [
    "OPEN_AUTHORIZATION",
    "PRELOCKBOX_TAG",
    "ReplayAudit",
    "ReplayResult",
    "load_reconstructed_estimators",
    "main",
    "open_q2",
    "opening_identity_from_manifest",
    "preflight",
    "replay_btc_candidates",
    "replay_index_candidates",
    "resume_q2",
    "run_registered_q2",
    "score_btc_candidates",
    "score_index_candidates",
    "validate_prelockbox_manifest_files",
]


if __name__ == "__main__":
    raise SystemExit(main())
