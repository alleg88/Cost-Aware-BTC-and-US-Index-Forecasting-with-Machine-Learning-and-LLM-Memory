"""Cutoff-safe, identity-pinned sentiment scoring for the index experiments."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd
import requests

from experiments.final_q2_lockbox_contract import Q2_END
from experiments.final_q2_lockbox_state import OpeningIdentity, require_global_opening
from sentiment.score import MODEL as DEBERTA_MODEL
from sentiment.score import REVISION as DEBERTA_REVISION
from sentiment.score import VERSION as DEBERTA_VERSION
from sentiment.score import _build_scorer, _norm
from sentiment.score_llm import SYSTEM, _ITEM, _batch_schema, _parse, _parse_text


CODE_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = CODE_ROOT / "sentiment" / "raw"
CUTOFF = pd.Timestamp("2026-04-01T00:00:00Z")
STREAMS = ("usa500", "usatech")
SOURCE_PREFIXES = ("gdelt", "direct_events")
DEEPSEEK_TAG = "deepseek-v4-flash:0731-cloud"
OFFICIAL_SHORT_DIGEST = "031ce2a95446"
DEEPSEEK_VERSION = "index-0731-v1"
DEEPSEEK_BATCH = 10
DEEPSEEK_BATCH_PROTOCOL = "indexed-json-object-v1"
DEEPSEEK_NUM_CTX = 8192
DEEPSEEK_THINK = "low"
DEEPSEEK_TEMPERATURE = 0.0
STATE_IDENTITY_COLUMNS = {
    "instrument",
    "title_norm",
    "cache_key",
    "llm_sent",
    "llm_relevance",
    "llm_impact",
    "llm_asset",
    "model",
    "model_digest",
    "remote_model",
    "remote_host",
    "registry_digest_prefix",
    "prompt_hash",
    "schema_hash",
    "think",
    "temperature",
    "num_ctx",
    "batch_size",
    "batch_protocol",
    "scorer_implementation_hash",
}


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, engine="pyarrow", index=False)
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prepare_scoring_rows(
    source: pd.DataFrame | str | Path,
    *,
    start_inclusive: str | pd.Timestamp | None = None,
    end_exclusive: str | pd.Timestamp = CUTOFF,
    opening_identity: OpeningIdentity | None = None,
) -> pd.DataFrame:
    """Apply the sealed cutoff before normalization, deduplication or todo work."""
    cutoff = _utc(end_exclusive)
    start = None if start_inclusive is None else _utc(start_inclusive)
    q2_access = cutoff > CUTOFF or (start is not None and start >= CUTOFF)
    if q2_access:
        if start != CUTOFF or cutoff != Q2_END:
            raise PermissionError("sentiment Q2 access must equal the registered lockbox interval")
        if opening_identity is None:
            raise PermissionError("sentiment Q2 access requires the global OPENED sentinel")
        require_global_opening(opening_identity)
    elif cutoff > CUTOFF:
        raise PermissionError("sentiment scoring cannot cross the sealed cutoff")
    if start is not None and start >= cutoff:
        raise ValueError("start_inclusive must precede end_exclusive")
    from_frame = isinstance(source, pd.DataFrame)
    filters = [("seendate", "<", cutoff.to_pydatetime())]
    if start is not None:
        filters.insert(0, ("seendate", ">=", start.to_pydatetime()))
    raw = (
        source.copy()
        if from_frame
        else pd.read_parquet(
            source,
            filters=filters,
        )
    )
    required = {"seendate", "url", "title"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"sentiment source misses columns: {sorted(missing)}")
    seen = pd.to_datetime(raw["seendate"], utc=True, errors="raise")
    if not from_frame and seen.ge(cutoff).any():
        raise AssertionError("sentiment predicate pushdown crossed the sealed cutoff")
    visible = seen < cutoff
    if start is not None:
        visible &= seen >= start
    raw = raw.loc[visible].copy()
    raw["seendate"] = seen.loc[visible]
    rows = raw[["seendate", "url", "title"]].copy()
    rows["score_text"] = raw["score_text"] if "score_text" in raw else raw["title"]
    rows["title_norm"] = _norm(rows["score_text"])
    rows = rows.loc[rows["title_norm"].ne("")].reset_index(drop=True)
    if not rows.empty:
        if rows["seendate"].max() >= cutoff:
            raise AssertionError("sentiment rows crossed the requested cutoff")
        if start is not None and rows["seendate"].min() < start:
            raise AssertionError("sentiment rows crossed the requested lower boundary")
    return rows


def canonical_scoring_source_hash(rows: pd.DataFrame) -> str:
    """Hash only the canonical pre-cutoff rows visible to either scorer."""
    columns = ["seendate", "url", "title", "score_text", "title_norm"]
    missing = set(columns).difference(rows.columns)
    if missing:
        raise ValueError(f"scoring rows miss hash columns: {sorted(missing)}")
    current = rows[columns].copy()
    current["seendate"] = pd.to_datetime(current["seendate"], utc=True).map(
        lambda value: value.isoformat()
    )
    for column in columns[1:]:
        current[column] = current[column].fillna("").astype(str)
    current = current.sort_values(columns, kind="mergesort").reset_index(drop=True)
    payload = json.dumps(
        {"columns": columns, "rows": current.values.tolist()},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScorerIdentity:
    tag: str
    digest: str
    remote_model: str
    remote_host: str
    registry_digest_prefix: str
    ollama_version: str
    pulled_at_utc: str
    prompt_hash: str
    schema_hash: str
    temperature: float = DEEPSEEK_TEMPERATURE
    think: str = DEEPSEEK_THINK
    num_ctx: int = DEEPSEEK_NUM_CTX
    batch_size: int = DEEPSEEK_BATCH
    batch_protocol: str = DEEPSEEK_BATCH_PROTOCOL
    scorer_implementation_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_deepseek_identity(
    tags_payload: dict[str, Any],
    *,
    ollama_version: str,
    pulled_at_utc: str | None = None,
) -> ScorerIdentity:
    """Resolve the explicit 0731 Cloud wrapper and its official remote model."""
    records = list(tags_payload.get("models", []))
    matching = [
        record
        for record in records
        if record.get("name") == DEEPSEEK_TAG or record.get("model") == DEEPSEEK_TAG
    ]
    if len(matching) != 1:
        raise RuntimeError(f"exact {DEEPSEEK_TAG} model is not uniquely installed")
    record = matching[0]
    digest = str(record.get("digest", ""))
    remote_model = str(record.get("remote_model", ""))
    remote_host = str(record.get("remote_host", "")).rstrip("/")
    if len(digest) != 64:
        raise RuntimeError(f"{DEEPSEEK_TAG} must have a full local wrapper digest")
    if remote_model != "deepseek-v4-flash:0731" or remote_host != "https://ollama.com":
        raise RuntimeError(
            f"{DEEPSEEK_TAG} must resolve to official remote deepseek-v4-flash:0731"
        )
    return ScorerIdentity(
        tag=DEEPSEEK_TAG,
        digest=digest,
        remote_model=remote_model,
        remote_host=remote_host,
        registry_digest_prefix=OFFICIAL_SHORT_DIGEST,
        ollama_version=str(ollama_version).strip(),
        pulled_at_utc=pulled_at_utc or datetime.now(UTC).isoformat(),
        prompt_hash=_canonical_hash(
            {
                "system": SYSTEM,
                "user_line_template": "n{index}: {title}",
                "batch_protocol": DEEPSEEK_BATCH_PROTOCOL,
            }
        ),
        schema_hash=_canonical_hash(
            {"item": _ITEM, "envelope": "object keyed n0..n{batch_size-1}"}
        ),
        scorer_implementation_hash=_canonical_hash(
            {
                "score_batch": inspect.getsource(_score_deepseek_batch),
                "batch_schema": inspect.getsource(_batch_schema),
                "parse": inspect.getsource(_parse),
                "parse_text": inspect.getsource(_parse_text),
            }
        ),
    )


def cache_key(instrument: str, title_norm: str, identity: ScorerIdentity) -> str:
    """Identity-complete key; mutable aliases and cross-instrument reuse are impossible."""
    return _canonical_hash(
        {
            "instrument": str(instrument),
            "title_norm": str(title_norm),
            "tag": identity.tag,
            "digest": identity.digest,
            "remote_model": identity.remote_model,
            "remote_host": identity.remote_host,
            "registry_digest_prefix": identity.registry_digest_prefix,
            "prompt_hash": identity.prompt_hash,
            "schema_hash": identity.schema_hash,
            "temperature": identity.temperature,
            "think": identity.think,
            "num_ctx": identity.num_ctx,
            "batch_size": identity.batch_size,
            "batch_protocol": identity.batch_protocol,
            "scorer_implementation_hash": identity.scorer_implementation_hash,
        }
    )


def _live_tags(host: str = "http://localhost:11434") -> dict[str, Any]:
    response = requests.get(host.rstrip("/") + "/api/tags", timeout=30)
    response.raise_for_status()
    return response.json()


def _ollama_version() -> str:
    result = subprocess.run(
        ["ollama", "--version"], check=True, text=True, capture_output=True
    )
    return (result.stdout or result.stderr).strip()


def verify_live_identity(
    identity: ScorerIdentity, *, host: str = "http://localhost:11434"
) -> None:
    current = resolve_deepseek_identity(
        _live_tags(host),
        ollama_version=identity.ollama_version,
        pulled_at_utc=identity.pulled_at_utc,
    )
    stable = (
        "tag",
        "digest",
        "remote_model",
        "remote_host",
        "registry_digest_prefix",
        "prompt_hash",
        "schema_hash",
        "temperature",
        "think",
        "num_ctx",
        "batch_size",
        "batch_protocol",
        "scorer_implementation_hash",
    )
    if any(getattr(current, field) != getattr(identity, field) for field in stable):
        raise RuntimeError("DeepSeek identity or scorer protocol changed during the run")


def _response_content(response: Any) -> str:
    try:
        return str(response["message"]["content"])
    except (KeyError, TypeError):
        return str(response.message.content or "")


def _score_deepseek_batch(
    titles: Sequence[str], identity: ScorerIdentity
) -> list[dict[str, Any] | None]:
    import ollama

    n = len(titles)
    lines = "\n".join(f"n{i}: {title}" for i, title in enumerate(titles))
    for _ in range(2):
        try:
            response = ollama.Client(timeout=300.0).chat(
                model=identity.tag,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": lines},
                ],
                stream=False,
                think=identity.think,
                format=_batch_schema(n),
                options={"temperature": identity.temperature, "num_ctx": identity.num_ctx},
            )
            content = _response_content(response)
            try:
                decoded = json.loads(content)
                parsed = [
                    _parse(decoded[f"n{i}"]) if f"n{i}" in decoded else None
                    for i in range(n)
                ]
            except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                parsed = _parse_text(content, n)
            if all(value is not None for value in parsed):
                return parsed
        except Exception:
            continue
    return [None] * n


def run_deepseek_preflight(
    *,
    raw_dir: str | Path = RAW_DIR,
    host: str = "http://localhost:11434",
    pull: bool = True,
    batch_scorer: Callable[[Sequence[str], ScorerIdentity], list[dict[str, Any] | None]] = _score_deepseek_batch,
) -> ScorerIdentity:
    """Pull, resolve, schema-probe and persist one identity for both streams."""
    if pull:
        subprocess.run(["ollama", "pull", DEEPSEEK_TAG], check=True)
    identity = resolve_deepseek_identity(
        _live_tags(host), ollama_version=_ollama_version()
    )
    show = requests.post(
        host.rstrip("/") + "/api/show",
        json={"model": DEEPSEEK_TAG},
        timeout=30,
    )
    show.raise_for_status()
    capabilities = set(show.json().get("capabilities", []))
    if "thinking" not in capabilities:
        raise RuntimeError("pinned DeepSeek model does not advertise thinking support")
    probe = batch_scorer(
        ["S&P 500 closes unchanged after a quiet session."], identity
    )
    if len(probe) != 1 or probe[0] is None:
        raise RuntimeError("DeepSeek structured schema probe failed")
    verify_live_identity(identity, host=host)
    _atomic_json(
        {
            "identity": identity.to_dict(),
            "capabilities": sorted(capabilities),
            "probe": probe[0],
            "verified_at_utc": datetime.now(UTC).isoformat(),
        },
        Path(raw_dir) / "index_deepseek_identity.json",
    )
    return identity


def load_saved_deepseek_identity(
    raw_dir: str | Path = RAW_DIR,
    *,
    verify_identity: Callable[[ScorerIdentity], None] = verify_live_identity,
) -> ScorerIdentity:
    """Load and live-verify the frozen identity without changing its state path."""
    path = Path(raw_dir) / "index_deepseek_identity.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity_payload = payload.get("identity")
    if not isinstance(identity_payload, dict):
        raise ValueError("saved DeepSeek identity is missing")
    try:
        identity = ScorerIdentity(**identity_payload)
    except TypeError as exc:
        raise ValueError("saved DeepSeek identity is incomplete") from exc
    verify_identity(identity)
    return identity


def _source_and_output(
    raw_dir: Path,
    stream: str,
    source_prefix: str,
    *,
    deepseek: bool,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    source = raw_dir / f"{source_prefix}_{stream}.parquet"
    if deepseek:
        name = (
            f"scores_llm_{stream}.parquet"
            if source_prefix == "gdelt"
            else f"scores_llm_{source_prefix}_{stream}.parquet"
        )
    else:
        name = (
            f"scores_{stream}.parquet"
            if source_prefix == "gdelt"
            else f"scores_{source_prefix}_{stream}.parquet"
        )
    return source, (raw_dir if output_dir is None else output_dir) / name


def score_deberta_index(
    stream: str,
    *,
    source_prefix: str = "gdelt",
    raw_dir: str | Path = RAW_DIR,
    output_dir: str | Path | None = None,
    start_inclusive: str | pd.Timestamp | None = None,
    end_exclusive: str | pd.Timestamp = CUTOFF,
    opening_identity: OpeningIdentity | None = None,
    batch_size: int = 64,
    scorer: Callable[[list[str]], list[float]] | None = None,
) -> Path:
    """Score one pre-Q2 index stream with the existing fixed DeBERTa model."""
    if stream not in STREAMS:
        raise ValueError(f"stream must be one of {STREAMS}")
    raw_dir = Path(raw_dir)
    output_root = raw_dir if output_dir is None else Path(output_dir)
    source, output = _source_and_output(
        raw_dir, stream, source_prefix, deepseek=False, output_dir=output_root
    )
    if not source.exists():
        raise FileNotFoundError(source)
    rows = prepare_scoring_rows(
        source,
        start_inclusive=start_inclusive,
        end_exclusive=end_exclusive,
        opening_identity=opening_identity,
    )
    unique = rows[["score_text", "title_norm"]].drop_duplicates("title_norm")
    unique["cache_key"] = unique["title_norm"].map(
        lambda title: _canonical_hash(
            {
                "instrument": stream,
                "title_norm": title,
                "model": DEBERTA_MODEL,
                "revision": DEBERTA_REVISION,
                "version": DEBERTA_VERSION,
            }
        )
    )
    cached = pd.DataFrame(columns=["cache_key", "sent"])
    if output.exists():
        previous = pd.read_parquet(output)
        required = {"cache_key", "sent", "model", "revision", "version"}
        if required.issubset(previous.columns):
            cached = previous.loc[
                previous["model"].eq(DEBERTA_MODEL)
                & previous["revision"].eq(DEBERTA_REVISION)
                & previous["version"].astype(str).eq(str(DEBERTA_VERSION))
                & previous["sent"].notna(),
                ["cache_key", "sent"],
            ].drop_duplicates("cache_key")
    todo = unique.loc[~unique["cache_key"].isin(cached["cache_key"])]
    q2_interval = (
        start_inclusive is not None
        and _utc(start_inclusive) == CUTOFF
        and _utc(end_exclusive) == Q2_END
    )
    score_fn = scorer
    if score_fn is None and len(todo):
        score_fn = (
            _build_scorer(
                batch_size,
                revision=DEBERTA_REVISION,
                local_files_only=True,
            )
            if q2_interval
            else _build_scorer(batch_size)
        )
    fresh = pd.DataFrame(columns=["cache_key", "sent"])
    if len(todo):
        assert score_fn is not None
        fresh = pd.DataFrame(
            {
                "cache_key": todo["cache_key"].to_numpy(),
                "sent": score_fn(todo["score_text"].tolist()),
            }
        )
    score_map = pd.concat([cached, fresh], ignore_index=True).drop_duplicates(
        "cache_key", keep="last"
    )
    article = rows.merge(unique[["title_norm", "cache_key"]], on="title_norm", how="left")
    article = article.merge(score_map, on="cache_key", how="left", validate="many_to_one")
    if article["sent"].isna().any():
        raise RuntimeError(f"DeBERTa left {int(article['sent'].isna().sum())} rows unscored")
    article["model"] = DEBERTA_MODEL
    article["revision"] = DEBERTA_REVISION
    article["version"] = str(DEBERTA_VERSION)
    article["scored_at"] = pd.Timestamp.now("UTC")
    _atomic_parquet(article, output)
    _atomic_json(
        {
            "stream": stream,
            "source_prefix": source_prefix,
            "start_inclusive": (
                None if start_inclusive is None else _utc(start_inclusive).isoformat()
            ),
            "cutoff_exclusive": _utc(end_exclusive).isoformat(),
            "source_sha256": canonical_scoring_source_hash(rows),
            "source_sha256_scope": (
                "canonical_pre_cutoff_scoring_rows"
                if start_inclusive is None and _utc(end_exclusive) == CUTOFF
                else "canonical_interval_scoring_rows"
            ),
            "complete": True,
            "model": DEBERTA_MODEL,
            "revision": DEBERTA_REVISION,
            "local_files_only": bool(q2_interval),
            "version": str(DEBERTA_VERSION),
            "rows_after_cutoff": len(article),
            "unique_titles": int(unique["title_norm"].nunique()),
        },
        output.with_suffix(".manifest.json"),
    )
    return output


def _state_path(
    raw_dir: Path,
    stream: str,
    source_prefix: str,
    identity: ScorerIdentity,
) -> Path:
    identity_key = _canonical_hash(identity.to_dict())[:16]
    return raw_dir / f"index_deepseek_state_{source_prefix}_{stream}_{identity_key}.parquet"


def _validate_state_cache_keys(
    state: pd.DataFrame, identity: ScorerIdentity, stream: str
) -> None:
    if state.empty:
        return
    expected = state["title_norm"].astype(str).map(
        lambda title: cache_key(stream, title, identity)
    )
    if not expected.eq(state["cache_key"].astype(str)).all():
        raise ValueError("DeepSeek state contains an invalid cache key")


def _identity_columns(identity: ScorerIdentity, stream: str) -> dict[str, Any]:
    return {
        "instrument": stream,
        "model": identity.tag,
        "model_digest": identity.digest,
        "remote_model": identity.remote_model,
        "remote_host": identity.remote_host,
        "registry_digest_prefix": identity.registry_digest_prefix,
        "prompt_hash": identity.prompt_hash,
        "schema_hash": identity.schema_hash,
        "think": identity.think,
        "temperature": identity.temperature,
        "num_ctx": identity.num_ctx,
        "batch_size": identity.batch_size,
        "batch_protocol": identity.batch_protocol,
        "scorer_implementation_hash": identity.scorer_implementation_hash,
    }


def _load_deepseek_state(
    path: Path, identity: ScorerIdentity, stream: str
) -> pd.DataFrame:
    fields = [
        "instrument",
        "title_norm",
        "cache_key",
        "llm_sent",
        "llm_relevance",
        "llm_impact",
        "llm_asset",
        "model",
        "model_digest",
        "remote_model",
        "remote_host",
        "registry_digest_prefix",
        "prompt_hash",
        "schema_hash",
        "think",
        "temperature",
        "num_ctx",
        "batch_size",
        "batch_protocol",
        "scorer_implementation_hash",
    ]
    if not path.exists():
        return pd.DataFrame(columns=fields)
    state = pd.read_parquet(path)
    missing = STATE_IDENTITY_COLUMNS.difference(state.columns)
    if missing:
        raise ValueError(f"DeepSeek state misses identity columns: {sorted(missing)}")
    _validate_state_cache_keys(state, identity, stream)
    matching = (
        state["instrument"].eq(stream)
        & state["model"].eq(identity.tag)
        & state["model_digest"].eq(identity.digest)
        & state["remote_model"].eq(identity.remote_model)
        & state["remote_host"].astype(str).str.rstrip("/").eq(identity.remote_host.rstrip("/"))
        & state["registry_digest_prefix"].eq(identity.registry_digest_prefix)
        & state["prompt_hash"].eq(identity.prompt_hash)
        & state["schema_hash"].eq(identity.schema_hash)
        & state["think"].eq(identity.think)
        & state["temperature"].astype(float).eq(identity.temperature)
        & state["num_ctx"].astype(int).eq(identity.num_ctx)
        & state["batch_size"].astype(int).eq(identity.batch_size)
        & state["batch_protocol"].eq(identity.batch_protocol)
        & state["scorer_implementation_hash"].eq(identity.scorer_implementation_hash)
    )
    return state.loc[matching, fields].drop_duplicates("cache_key", keep="last")


def score_deepseek_index(
    stream: str,
    *,
    identity: ScorerIdentity,
    source_prefix: str = "gdelt",
    raw_dir: str | Path = RAW_DIR,
    output_dir: str | Path | None = None,
    start_inclusive: str | pd.Timestamp | None = None,
    end_exclusive: str | pd.Timestamp = CUTOFF,
    opening_identity: OpeningIdentity | None = None,
    workers: int = 2,
    limit: int | None = None,
    checkpoint_batches: int = 50,
    batch_scorer: Callable[[Sequence[str], ScorerIdentity], list[dict[str, Any] | None]] = _score_deepseek_batch,
    verify_identity: Callable[[ScorerIdentity], None] = verify_live_identity,
) -> Path:
    """Score one index stream with resumable identity-complete DeepSeek state."""
    if stream not in STREAMS:
        raise ValueError(f"stream must be one of {STREAMS}")
    if (
        identity.tag != DEEPSEEK_TAG
        or identity.remote_model != "deepseek-v4-flash:0731"
        or identity.remote_host.rstrip("/") != "https://ollama.com"
        or identity.registry_digest_prefix != OFFICIAL_SHORT_DIGEST
        or len(identity.digest) != 64
    ):
        raise ValueError("DeepSeek identity is not the frozen 0731 model")
    raw_dir = Path(raw_dir)
    output_root = raw_dir if output_dir is None else Path(output_dir)
    source, output = _source_and_output(
        raw_dir, stream, source_prefix, deepseek=True, output_dir=output_root
    )
    if not source.exists():
        raise FileNotFoundError(source)
    rows = prepare_scoring_rows(
        source,
        start_inclusive=start_inclusive,
        end_exclusive=end_exclusive,
        opening_identity=opening_identity,
    )
    unique = rows[["score_text", "title_norm"]].drop_duplicates("title_norm")
    unique["cache_key"] = unique["title_norm"].map(
        lambda title: cache_key(stream, title, identity)
    )
    state_path = _state_path(output_root, stream, source_prefix, identity)
    state = _load_deepseek_state(state_path, identity, stream)
    todo = unique.loc[~unique["cache_key"].isin(state["cache_key"])]
    if limit is not None:
        todo = todo.head(int(limit))
    verify_identity(identity)

    identity_columns = _identity_columns(identity, stream)
    batches = [
        todo.iloc[start : start + identity.batch_size]
        for start in range(0, len(todo), identity.batch_size)
    ]
    fresh_rows: list[dict[str, Any]] = []

    def call(batch: pd.DataFrame) -> list[dict[str, Any] | None]:
        return batch_scorer(batch["score_text"].tolist(), identity)

    if batches:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            for batch_number, (batch, results) in enumerate(
                zip(batches, executor.map(call, batches)), start=1
            ):
                if len(results) != len(batch):
                    raise RuntimeError("DeepSeek batch result length changed")
                for (_, row), result in zip(batch.iterrows(), results):
                    if result is not None:
                        fresh_rows.append(
                            {
                                "title_norm": row["title_norm"],
                                "cache_key": row["cache_key"],
                                **result,
                                **identity_columns,
                            }
                        )
                if batch_number % max(1, checkpoint_batches) == 0:
                    combined = pd.concat(
                        [state, pd.DataFrame(fresh_rows)], ignore_index=True
                    ).drop_duplicates("cache_key", keep="last")
                    _atomic_parquet(combined, state_path)
                    verify_identity(identity)
        state = pd.concat([state, pd.DataFrame(fresh_rows)], ignore_index=True).drop_duplicates(
            "cache_key", keep="last"
        )
        _atomic_parquet(state, state_path)
    verify_identity(identity)

    requested = unique if limit is None else unique.loc[
        unique["cache_key"].isin(state["cache_key"])
    ]
    missing_keys = requested.loc[~requested["cache_key"].isin(state["cache_key"])]
    if len(missing_keys):
        raise RuntimeError(
            f"DeepSeek left {len(missing_keys)} requested unique headlines unscored; resume required"
        )
    if limit is not None:
        return state_path
    article = rows.merge(unique[["title_norm", "cache_key"]], on="title_norm", how="left")
    score_columns = [
        "cache_key",
        "llm_sent",
        "llm_relevance",
        "llm_impact",
        "llm_asset",
        *identity_columns.keys(),
    ]
    article = article.merge(
        state[score_columns], on="cache_key", how="left", validate="many_to_one"
    )
    if limit is None and article["llm_sent"].isna().any():
        raise RuntimeError(
            f"DeepSeek article cache has {int(article['llm_sent'].isna().sum())} missing rows"
        )
    article["version"] = DEEPSEEK_VERSION
    article["scored_at"] = pd.Timestamp.now("UTC")
    _atomic_parquet(article, output)
    _atomic_json(
        {
            "stream": stream,
            "source_prefix": source_prefix,
            "start_inclusive": (
                None if start_inclusive is None else _utc(start_inclusive).isoformat()
            ),
            "cutoff_exclusive": _utc(end_exclusive).isoformat(),
            "source_sha256": canonical_scoring_source_hash(rows),
            "source_sha256_scope": (
                "canonical_pre_cutoff_scoring_rows"
                if start_inclusive is None and _utc(end_exclusive) == CUTOFF
                else "canonical_interval_scoring_rows"
            ),
            "complete": True,
            "identity": identity.to_dict(),
            "rows_after_cutoff": len(article),
            "unique_titles": int(unique["title_norm"].nunique()),
            "scored_unique_titles": int(state["cache_key"].nunique()),
            "state_path": state_path.name,
        },
        output.with_suffix(".manifest.json"),
    )
    return output


def materialize_matched_coverage(
    stream: str,
    *,
    source_prefix: str = "gdelt",
    raw_dir: str | Path = RAW_DIR,
) -> Path:
    """Persist the exact successfully scored article intersection for both scorers."""
    raw_dir = Path(raw_dir)
    _, classic_path = _source_and_output(raw_dir, stream, source_prefix, deepseek=False)
    _, deepseek_path = _source_and_output(raw_dir, stream, source_prefix, deepseek=True)
    classic = pd.read_parquet(classic_path)
    deepseek = pd.read_parquet(deepseek_path)
    keys = ["seendate", "url", "title", "title_norm"]
    matched = classic[[*keys, "sent"]].dropna(subset=["sent"]).merge(
        deepseek[
            [
                *keys,
                "llm_sent",
                "llm_relevance",
                "llm_impact",
                "llm_asset",
                "model_digest",
                "remote_model",
                "remote_host",
                "registry_digest_prefix",
                "prompt_hash",
                "schema_hash",
                "num_ctx",
                "batch_size",
                "batch_protocol",
                "scorer_implementation_hash",
            ]
        ].dropna(subset=["llm_sent"]),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if pd.to_datetime(matched["seendate"], utc=True).ge(CUTOFF).any():
        raise AssertionError("matched scorer set crossed Q2")
    output = raw_dir / f"matched_{source_prefix}_{stream}.parquet"
    _atomic_parquet(matched, output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    preflight.add_argument("--no-pull", action="store_true")
    scoring = sub.add_parser("score")
    scoring.add_argument("--streams", nargs="+", choices=STREAMS, default=list(STREAMS))
    scoring.add_argument(
        "--scorers", nargs="+", choices=("deberta", "deepseek"), default=["deberta", "deepseek"]
    )
    scoring.add_argument("--source-prefixes", nargs="+", default=list(SOURCE_PREFIXES))
    scoring.add_argument("--workers", type=int, default=2)
    scoring.add_argument("--limit", type=int)
    scoring.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    scoring.add_argument("--no-pull", action="store_true")
    scoring.add_argument("--resume-saved-identity", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "preflight":
        identity = run_deepseek_preflight(raw_dir=args.raw_dir, pull=not args.no_pull)
        print(json.dumps(identity.to_dict(), indent=2))
        return 0

    identity = None
    if "deepseek" in args.scorers:
        identity = (
            load_saved_deepseek_identity(args.raw_dir)
            if args.resume_saved_identity
            else run_deepseek_preflight(raw_dir=args.raw_dir, pull=not args.no_pull)
        )
    for source_prefix in args.source_prefixes:
        for stream in args.streams:
            if "deberta" in args.scorers:
                path = score_deberta_index(
                    stream, source_prefix=source_prefix, raw_dir=args.raw_dir
                )
                print(f"DeBERTa {stream}/{source_prefix}: {path}")
            if "deepseek" in args.scorers:
                assert identity is not None
                path = score_deepseek_index(
                    stream,
                    identity=identity,
                    source_prefix=source_prefix,
                    raw_dir=args.raw_dir,
                    workers=args.workers,
                    limit=args.limit,
                )
                print(f"DeepSeek {stream}/{source_prefix}: {path}")
            if set(args.scorers) == {"deberta", "deepseek"} and args.limit is None:
                matched = materialize_matched_coverage(
                    stream, source_prefix=source_prefix, raw_dir=args.raw_dir
                )
                print(f"matched {stream}/{source_prefix}: {matched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CUTOFF",
    "DEEPSEEK_TAG",
    "ScorerIdentity",
    "cache_key",
    "load_saved_deepseek_identity",
    "materialize_matched_coverage",
    "prepare_scoring_rows",
    "canonical_scoring_source_hash",
    "resolve_deepseek_identity",
    "run_deepseek_preflight",
    "score_deberta_index",
    "score_deepseek_index",
    "verify_live_identity",
]
