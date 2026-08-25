"""Supplementary frozen-policy sensitivity to newly completed Q2 sentiment feeds."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from features.index_sentiment import MATCHED_FEATURES
from sentiment.gdelt_bigquery import _is_tech, _tone


CODE_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = CODE_ROOT / "experiments" / "cache" / "q2_sentiment_sensitivity"
GDELT_STAGE = (
    CODE_ROOT
    / "sentiment"
    / "raw"
    / "gdelt_bq"
    / "q2_staging"
    / "gdelt_q2_2026_english_whitelist.csv"
)
DIRECT_STAGE = CODE_ROOT / "sentiment" / "raw" / "direct_events_q2_staging"
FROZEN_RAW = CODE_ROOT / "sentiment" / "raw"
Q2_START = pd.Timestamp("2026-04-01T00:00:00Z")
Q2_END = pd.Timestamp("2026-07-01T00:00:00Z")


def normalise_q2_index_news(export: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Convert the combined BigQuery export into market and overlapping tech streams."""
    required = {"DATE", "url", "domain", "V2Tone", "title", "stream"}
    missing = required - set(export.columns)
    if missing:
        raise ValueError(f"Q2 GDELT export missing columns: {sorted(missing)}")

    frame = export.copy()
    frame["seendate"] = pd.to_datetime(
        frame["DATE"].astype("int64").astype(str),
        format="%Y%m%d%H%M%S",
        utc=True,
    )
    frame["tone"] = _tone(frame["V2Tone"])
    frame["title"] = frame["title"].map(
        lambda value: html.unescape(value) if isinstance(value, str) else value
    )
    frame["themes"] = frame["V2Themes"] if "V2Themes" in frame else pd.NA
    market = frame.loc[frame["stream"].ne("btc")].copy()
    tech = market.loc[_is_tech(market)].copy()

    columns = ["seendate", "url", "domain", "tone", "title", "themes"]

    def clean(current: pd.DataFrame) -> pd.DataFrame:
        return (
            current.loc[:, columns]
            .dropna(subset=["url", "title"])
            .drop_duplicates("url")
            .sort_values("seendate")
            .reset_index(drop=True)
        )

    return {"usa500": clean(market), "usatech": clean(tech)}


def overlay_q2_sentiment(
    baseline_features: pd.DataFrame,
    fresh_sentiment: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Replace only the five Q2 sentiment channels in a frozen feature matrix."""
    missing = set(MATCHED_FEATURES) - set(baseline_features.columns)
    if missing:
        raise ValueError(f"baseline features miss sentiment columns: {sorted(missing)}")
    if tuple(fresh_sentiment.columns) != tuple(MATCHED_FEATURES):
        raise ValueError("fresh sentiment schema differs from the frozen matched schema")
    q2_index = baseline_features.index[
        (baseline_features.index >= start) & (baseline_features.index < end)
    ]
    if not fresh_sentiment.index.equals(q2_index):
        raise ValueError("fresh sentiment index differs from frozen Q2 feature rows")

    output = baseline_features.copy()
    output.loc[q2_index, list(MATCHED_FEATURES)] = fresh_sentiment.to_numpy(float)
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_lf(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode(
            "utf-8"
        )
    )
    return path


def load_frozen_manifest_for_sensitivity(path: Path, identity) -> dict:
    """Bind immutable pre-lockbox inputs without validating later reader code."""
    if _sha256(path) != identity.manifest_sha256:
        raise ValueError("pre-lockbox manifest hash differs from OPENED identity")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["protocol"]["protocol_hash"] != identity.protocol_hash:
        raise ValueError("pre-lockbox protocol hash differs from OPENED identity")
    q2_hashes = {
        key: entry["sha256"] for key, entry in manifest["q2_sources"].items()
    }
    if q2_hashes != dict(identity.q2_source_hashes):
        raise ValueError("pre-lockbox Q2 source registry differs from OPENED identity")
    return manifest


def prepare_supplementary_sources(
    output_root: Path = CACHE_ROOT,
) -> tuple[Path, Path, dict[str, object]]:
    """Create isolated Q2 scorer inputs and continuous tone/macro context."""
    if not GDELT_STAGE.is_file():
        raise FileNotFoundError(GDELT_STAGE)
    direct_manifest_path = DIRECT_STAGE / "manifest.json"
    if not direct_manifest_path.is_file():
        raise FileNotFoundError(direct_manifest_path)
    direct_manifest = json.loads(direct_manifest_path.read_text(encoding="utf-8"))
    for name, expected in direct_manifest["artifact_sha256"].items():
        path = DIRECT_STAGE / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"staged direct-event artifact changed: {name}")

    raw_q2 = output_root / "raw_q2"
    source_context = output_root / "source_context"
    raw_q2.mkdir(parents=True, exist_ok=True)
    source_context.mkdir(parents=True, exist_ok=True)

    streams = normalise_q2_index_news(pd.read_csv(GDELT_STAGE))
    audit: dict[str, object] = {"gdelt_rows": {}, "direct_rows": {}}
    for stream, q2_news in streams.items():
        q2_news.to_parquet(raw_q2 / f"gdelt_{stream}.parquet", index=False)
        pre_q2 = pd.read_parquet(FROZEN_RAW / f"gdelt_{stream}.parquet")
        context = (
            pd.concat([pre_q2, q2_news], ignore_index=True, sort=False)
            .drop_duplicates("url", keep="last")
            .sort_values("seendate")
            .reset_index(drop=True)
        )
        context.to_parquet(source_context / f"gdelt_{stream}.parquet", index=False)
        direct_source = DIRECT_STAGE / f"direct_events_{stream}.parquet"
        shutil.copy2(direct_source, raw_q2 / direct_source.name)
        audit["gdelt_rows"][stream] = int(len(q2_news))
        audit["direct_rows"][stream] = int(len(pd.read_parquet(direct_source)))
    shutil.copy2(FROZEN_RAW / "fred_calendar.parquet", source_context / "fred_calendar.parquet")
    audit.update(
        {
            "q2_start": Q2_START.isoformat(),
            "q2_end_exclusive": Q2_END.isoformat(),
            "gdelt_stage_sha256": _sha256(GDELT_STAGE),
            "direct_manifest_sha256": _sha256(direct_manifest_path),
        }
    )
    return raw_q2, source_context, audit


def score_supplementary_sentiment(
    identity,
    raw_q2: Path,
    score_root: Path,
) -> dict[str, Path]:
    """Score the exact staged rows with the frozen DeBERTa and 0731 LLM identities."""
    from sentiment.index_scoring import (
        DEBERTA_REVISION,
        load_saved_deepseek_identity,
        score_deberta_index,
        score_deepseek_index,
    )
    from sentiment.score import _build_scorer

    score_root.mkdir(parents=True, exist_ok=True)
    deberta = _build_scorer(64, revision=DEBERTA_REVISION, local_files_only=True)
    llm_identity = load_saved_deepseek_identity()
    outputs: dict[str, Path] = {}
    for stream in ("usa500", "usatech"):
        for prefix in ("gdelt", "direct_events"):
            label = f"{stream}_{prefix}"
            print(f"[supplementary] DeBERTa {label}", flush=True)
            outputs[f"deberta_{label}"] = score_deberta_index(
                stream,
                source_prefix=prefix,
                raw_dir=raw_q2,
                output_dir=score_root,
                start_inclusive=Q2_START,
                end_exclusive=Q2_END,
                opening_identity=identity,
                scorer=deberta,
            )
            print(f"[supplementary] LLM {label}", flush=True)
            outputs[f"llm_{label}"] = score_deepseek_index(
                stream,
                identity=llm_identity,
                source_prefix=prefix,
                raw_dir=raw_q2,
                output_dir=score_root,
                start_inclusive=Q2_START,
                end_exclusive=Q2_END,
                opening_identity=identity,
                workers=2,
            )
    return outputs


def _assert_baseline_reproduction(
    rebuilt: dict[str, pd.DataFrame],
    final_root: Path,
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for candidate_id, frame in rebuilt.items():
        reference = pd.read_parquet(final_root / "predictions" / f"{candidate_id}.parquet")
        reference = reference.set_index(pd.to_datetime(reference.pop("timestamp"), utc=True))
        if not frame.index.equals(reference.index):
            raise ValueError(f"baseline timestamps changed: {candidate_id}")
        columns = ["confidence", "p_short", "p_flat", "p_long"]
        maximum = float(
            np.max(np.abs(frame[columns].to_numpy(float) - reference[columns].to_numpy(float)))
        )
        if maximum > 1e-10 or not frame["signal"].eq(reference["signal"]).all():
            raise ValueError(f"baseline prediction no longer reproduces: {candidate_id}")
        errors[candidate_id] = maximum
    return errors


def build_comparison(original: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Align original and fresh-feed economics for the same frozen candidates."""
    metrics = [
        "trades",
        "long_trades",
        "short_trades",
        "net_return",
        "daily_sharpe",
        "daily_sortino",
        "max_drawdown",
    ]
    left = original[["stream", "candidate_id", "role", "arm", *metrics]].rename(
        columns={name: f"original_{name}" for name in metrics}
    )
    right = fresh[["stream", "candidate_id", *metrics]].rename(
        columns={name: f"fresh_{name}" for name in metrics}
    )
    comparison = left.merge(right, on=["stream", "candidate_id"], validate="one_to_one")
    comparison["delta_trades"] = comparison["fresh_trades"] - comparison["original_trades"]
    comparison["delta_net_return"] = (
        comparison["fresh_net_return"] - comparison["original_net_return"]
    )
    comparison["delta_sortino"] = (
        comparison["fresh_daily_sortino"] - comparison["original_daily_sortino"]
    )
    return comparison.sort_values("fresh_net_return", ascending=False).reset_index(drop=True)


def run_sensitivity(
    identity,
    raw_q2: Path,
    source_context: Path,
    score_root: Path,
    output_root: Path,
) -> pd.DataFrame:
    """Replay only the two registered frozen index candidates per market."""
    from experiments.final_q2_lockbox_contract import load_lockbox_protocol
    from experiments.final_q2_lockbox_metrics import summarise_candidate
    from experiments.final_q2_lockbox_runner import (
        DEFAULT_MANIFEST_PATH,
        DEFAULT_PROTOCOL_PATH,
        _bound_path,
        _build_index_execution_inputs,
        _stream_candidates,
        _stream_estimators,
        load_reconstructed_estimators,
        replay_index_candidates,
        score_index_candidates,
    )
    from features.index_sentiment import build_matched_index_features

    del raw_q2  # inputs are already bound through score/source manifests
    manifest = load_frozen_manifest_for_sensitivity(DEFAULT_MANIFEST_PATH, identity)
    protocol = load_lockbox_protocol(DEFAULT_PROTOCOL_PATH)
    estimators, metadata = load_reconstructed_estimators(manifest)
    final_root = (
        CODE_ROOT / "experiments" / "cache" / "final_q2_lockbox" / identity.protocol_hash
    )
    original = pd.read_parquet(final_root / "summaries.parquet")
    original = original.loc[original["stream"].isin(["usa500", "usatech"])].copy()

    output_root.mkdir(parents=True, exist_ok=True)
    fresh_summaries: list[dict[str, object]] = []
    audits: dict[str, object] = {}
    for stream in ("usa500", "usatech"):
        bars, baseline_features, input_audit = _build_index_execution_inputs(
            stream,
            identity,
            manifest,
            metadata,
            final_root / "sentiment_scores",
            start=Q2_START,
            end=Q2_END,
        )
        stream_estimators = _stream_estimators(stream, protocol, estimators)
        baseline_predictions = score_index_candidates(
            stream,
            baseline_features,
            stream_estimators,
            protocol,
            start=Q2_START,
            end=Q2_END,
        )
        reproduction = _assert_baseline_reproduction(baseline_predictions, final_root)
        q2_index = baseline_features.index[
            (baseline_features.index >= Q2_START) & (baseline_features.index < Q2_END)
        ]
        scorer = "classic" if stream == "usa500" else "llm"
        fresh_sentiment = build_matched_index_features(
            stream,
            q2_index,
            scorer=scorer,
            score_root=score_root,
            warmup_score_root=_bound_path(
                manifest, "warmups", f"scores_{stream}"
            ).parent,
            source_root=source_context,
            available_start=Q2_START,
            available_end=Q2_END,
            opening_identity=identity,
            continuous_context=True,
        )
        fresh_features = overlay_q2_sentiment(
            baseline_features,
            fresh_sentiment,
            start=Q2_START,
            end=Q2_END,
        )
        changed_feature_rows = int(
            fresh_features.loc[q2_index, list(MATCHED_FEATURES)]
            .ne(baseline_features.loc[q2_index, list(MATCHED_FEATURES)])
            .any(axis=1)
            .sum()
        )
        fresh_predictions = score_index_candidates(
            stream,
            fresh_features,
            stream_estimators,
            protocol,
            start=Q2_START,
            end=Q2_END,
        )
        replayed = replay_index_candidates(
            stream,
            bars,
            fresh_predictions,
            protocol,
            start=Q2_START,
            end=Q2_END,
        )
        signal_changes: dict[str, int] = {}
        for candidate in _stream_candidates(protocol, stream):
            candidate_id = candidate.candidate_id
            prediction = fresh_predictions[candidate_id].copy()
            prediction.rename_axis("timestamp").reset_index().to_parquet(
                output_root / f"prediction__{candidate_id}.parquet", index=False
            )
            result = replayed[candidate_id]
            result.ledger.to_parquet(
                output_root / f"ledger__{candidate_id}.parquet", index=False
            )
            row = summarise_candidate(
                result.ledger,
                candidate_id=candidate_id,
                stream=stream,
                start=Q2_START,
                end=Q2_END,
            )
            row.update({"role": candidate.role, "arm": candidate.arm})
            fresh_summaries.append(row)
            signal_changes[candidate_id] = int(
                fresh_predictions[candidate_id]["signal"]
                .ne(baseline_predictions[candidate_id]["signal"])
                .sum()
            )
        audits[stream] = {
            "baseline_max_probability_error": reproduction,
            "changed_feature_rows": changed_feature_rows,
            "signal_changes": signal_changes,
            "input_audit": input_audit,
        }

    fresh = pd.DataFrame(fresh_summaries)
    fresh.to_parquet(output_root / "fresh_summaries.parquet", index=False)
    comparison = build_comparison(original, fresh)
    comparison.to_parquet(output_root / "comparison.parquet", index=False)
    _write_json_lf(output_root / "audit.json", audits)
    artifacts = sorted(output_root.glob("*.parquet")) + [output_root / "audit.json"]
    _write_json_lf(
        output_root / "manifest.json",
        {
            "schema_version": "1.0",
            "analysis": "fresh_q2_sentiment_availability_sensitivity",
            "opening_identity": identity.to_dict(),
            "no_fitting": True,
            "no_threshold_selection": True,
            "no_candidate_selection": True,
            "artifact_sha256": {path.name: _sha256(path) for path in artifacts},
        },
    )
    return comparison


def main() -> int:
    from experiments.final_q2_lockbox_state import load_opening_identity

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=CACHE_ROOT)
    args = parser.parse_args()
    identity = load_opening_identity()
    raw_q2, source_context, source_audit = prepare_supplementary_sources(args.output_root)
    score_root = args.output_root / "sentiment_scores"
    outputs = score_supplementary_sentiment(identity, raw_q2, score_root)
    result_root = args.output_root / "results"
    comparison = run_sensitivity(
        identity,
        raw_q2,
        source_context,
        score_root,
        result_root,
    )
    resolved_output_root = args.output_root.resolve(strict=True)
    _write_json_lf(
        args.output_root / "source_audit.json",
        {
            **source_audit,
            "score_outputs": {
                key: path.resolve(strict=True)
                .relative_to(resolved_output_root)
                .as_posix()
                for key, path in outputs.items()
            },
        },
    )
    print(comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
