"""Opaque preflight and tracked identity manifest for the final Q2 lockbox."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd

from experiments.final_q2_lockbox_inputs import (
    copy_audited_pre_q2,
    opaque_file_identity,
    opaque_sha256,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent
MANIFEST_SCHEMA = "final-q2-lockbox-manifest-v1"
Q2_START = pd.Timestamp("2026-04-01T00:00:00Z")
Q1_START = pd.Timestamp("2026-01-01T00:00:00Z")
RECONSTRUCTION_ROOT = (
    CODE_ROOT / "experiments" / "cache" / "final_q2_lockbox" / "reconstructed_models"
)
WARMUP_DATA_ROOT = CODE_ROOT / "data" / "final_q2_lockbox"
WARMUP_SENTIMENT_ROOT = CODE_ROOT / "sentiment" / "final_q2_lockbox" / "pre_q2_scores"

DEBERTA_REVISION = "9e10915c245a80a89b18d1ac51350e093c7bb35a"
DEBERTA_FILE_HASHES = {
    "model.safetensors": "96a27b12ec06202bb959c66b774e7ead6e99ce227d150c0d777ecf974524c139",
    "config.json": "d7b9b097c0ac5da894de6906ab00662c7df7ec313bfef830d4fa9cf71b0d2a9c",
    "tokenizer.json": "05402ffae6dd382a8491b1d29bfc139bec5d332662e86a026f433ce54c25c202",
    "tokenizer_config.json": "557b3d33d3f41b81ad769244e506549e98a1857d41dd58160aacd4d98d710b5a",
    "spm.model": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
    "special_tokens_map.json": "9463f61e1b109a8eb4688b829260d7c6b1e6dff04c98ff7269bb89e2b92369b9",
    "added_tokens.json": "dc046d04c9b0ada7ae6f1dc89c465801799acdf0c9a6aab8c15a1b2d5ca4e91f",
}
DEEPSEEK_EXPECTED = {
    "tag": "deepseek-v4-flash:0731-cloud",
    "digest": "d3f1c87447216481a8001f48c517a51e13bfb141853a8df5e52f81bf765dabc3",
    "remote_model": "deepseek-v4-flash:0731",
    "remote_host": "https://ollama.com",
    "registry_digest_prefix": "031ce2a95446",
    "prompt_hash": "94bea4d8071a2e8eb1114b39815d0e08f669b8c484d78100e1dbffc5895eda40",
    "schema_hash": "57ac5a4ea650657dcca3e7f9327670500344243899f7cf879580be90d7db6f63",
    "scorer_implementation_hash": "6edb573f6a77dee78d72f9b89dc0001827dff814a4d77448c9062749a520094c",
    "batch_size": 10,
    "batch_protocol": "indexed-json-object-v1",
    "num_ctx": 8192,
    "temperature": 0.0,
    "think": "low",
}


@dataclass(frozen=True)
class ManifestCommitBinding:
    implementation_commit: str
    manifest_commit: str
    manifest_sha256: str


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value.lower())


def _identity(path: str | Path) -> dict[str, Any]:
    return opaque_file_identity(path).to_dict()


def q2_source_registry(code_root: str | Path = CODE_ROOT) -> dict[str, Path]:
    """Return the complete deterministic set of opaque Q2 sources and checksums."""
    root = Path(code_root)
    raw = root / "data" / "raw"
    binance = raw / "binance"
    sentiment = root / "sentiment" / "raw"
    registry: dict[str, Path] = {
        "btc_m15_q2": root / "data" / "btcusdt_m15_lockbox_2026Q2.parquet",
        "btc_m1_q2_source": root / "data" / "btcusdt_1m_2025_2026.parquet",
        "btc_positioning_q2_source": root / "data" / "btcusdt_positioning_m15_2024_2026.parquet",
        "usa500_bid_q2_source": raw / "USA500IDXUSD_1 Min_Bid_2021.01.01_2026.08.01.csv",
        "usa500_ask_q2_source": raw / "USA500IDXUSD_1 Min_Ask_2021.01.01_2026.08.01.csv",
        "usatech_bid_q2_source": raw / "USATECHIDXUSD_1 Min_Bid_2021.01.01_2026.08.01.csv",
        "usatech_ask_q2_source": raw / "USATECHIDXUSD_1 Min_Ask_2021.01.01_2026.08.01.csv",
        "volidx_bid_q2_source": raw / "VOLIDXUSD_1 Min_Bid_2022.10.05_2026.08.01.csv",
        "volidx_ask_q2_source": raw / "VOLIDXUSD_1 Min_Ask_2022.10.05_2026.08.01.csv",
        "gdelt_usa500_q2_source": sentiment / "gdelt_usa500.parquet",
        "direct_events_usa500_q2_source": sentiment / "direct_events_usa500.parquet",
        "gdelt_usatech_q2_source": sentiment / "gdelt_usatech.parquet",
        "direct_events_usatech_q2_source": sentiment / "direct_events_usatech.parquet",
        "fred_q2_source": sentiment / "fred_calendar.parquet",
    }
    for month in ("2026-04", "2026-05", "2026-06"):
        month_id = month.replace("-", "_")
        for interval in ("15m", "1m"):
            name = f"BTCUSDT-{interval}-{month}.zip"
            prefix = f"btc_{interval}_{month_id}"
            registry[f"{prefix}_zip"] = binance / name
            registry[f"{prefix}_checksum"] = binance / f"{name}.CHECKSUM"
        name = f"BTCUSDT-fundingRate-{month}.zip"
        prefix = f"btc_funding_{month_id}"
        registry[f"{prefix}_zip"] = binance / "futures" / "fundingRate" / name
        registry[f"{prefix}_checksum"] = (
            binance / "futures" / "fundingRate" / f"{name}.CHECKSUM"
        )
    for day in pd.date_range("2026-04-01", "2026-06-30", freq="D"):
        label = day.strftime("%Y-%m-%d")
        source_id = label.replace("-", "_")
        name = f"BTCUSDT-metrics-{label}.zip"
        registry[f"btc_metrics_{source_id}_zip"] = (
            binance / "futures" / "metrics" / name
        )
        registry[f"btc_metrics_{source_id}_checksum"] = (
            binance / "futures" / "metrics" / f"{name}.CHECKSUM"
        )
    return registry


def verify_provider_checksums(registry: Mapping[str, Path]) -> dict[str, str]:
    """Verify each registered provider zip against its adjacent checksum file."""
    verified: dict[str, str] = {}
    checksum_ids = {key for key in registry if key.endswith("_checksum")}
    zip_ids = {key for key in registry if key.endswith("_zip")}
    if {key.removesuffix("_checksum") for key in checksum_ids} != {
        key.removesuffix("_zip") for key in zip_ids
    }:
        raise ValueError("provider zip/checksum registry is incomplete")
    for zip_id in sorted(zip_ids):
        checksum_id = zip_id.removesuffix("_zip") + "_checksum"
        archive = Path(registry[zip_id])
        checksum = Path(registry[checksum_id])
        if not archive.is_file() or not checksum.is_file():
            raise FileNotFoundError(archive if not archive.is_file() else checksum)
        tokens = checksum.read_text(encoding="utf-8").split()
        if not tokens or not _is_hex(tokens[0], 64):
            raise ValueError(f"provider checksum is malformed: {checksum}")
        actual = opaque_sha256(archive)
        if actual != tokens[0].lower():
            raise ValueError(f"provider checksum mismatch: {archive.name}")
        verified[zip_id] = actual
    return verified


def reconstruction_manifest_paths(
    reconstruction_root: str | Path = RECONSTRUCTION_ROOT,
) -> tuple[Path, ...]:
    paths = tuple(sorted(Path(reconstruction_root).rglob("reconstruction_manifest.json")))
    if len(paths) != 21:
        raise ValueError(f"final lockbox requires exactly 21 reconstruction manifests, found {len(paths)}")
    return paths


def critical_source_registry(code_root: str | Path = CODE_ROOT) -> dict[str, Path]:
    root = Path(code_root)
    relative = (
        "configs/final_q2_lockbox_protocol.json",
        "configs/default.yaml",
        "requirements-repro.txt",
        "experiments/final_q2_lockbox_contract.py",
        "experiments/final_q2_lockbox_state.py",
        "experiments/final_q2_lockbox_inputs.py",
        "experiments/final_q2_lockbox_reconstruction.py",
        "experiments/final_q2_lockbox_metrics.py",
        "experiments/final_q2_lockbox_runner.py",
        "experiments/final_q2_lockbox_preflight.py",
        "experiments/all_model_sentiment_raw.py",
        "experiments/baseline_model_zoo_1m.py",
        "experiments/catboost_execution_resolution.py",
        "experiments/catboost_execution_scoring.py",
        "experiments/catboost_matched_ablation.py",
        "experiments/catboost_sentiment_ablation.py",
        "experiments/notebook02_handoff.py",
        "experiments/raw_hold_control.py",
        "experiments/run_catboost_matched_ablation.py",
        "experiments/run_walkforward.py",
        "experiments/spans.py",
        "experiments/walkforward.py",
        "experiments/index_replication.py",
        "experiments/index_replication_protocol.py",
        "evaluation/trades_intrabar.py",
        "evaluation/trades.py",
        "evaluation/economics.py",
        "evaluation/splits.py",
        "features/build.py",
        "features/index_sentiment.py",
        "features/sentiment.py",
        "models/deep.py",
        "models/zoo.py",
        "sentiment/index_scoring.py",
        "sentiment/dedup.py",
        "sentiment/score.py",
        "sentiment/score_llm.py",
        "data/build_positioning.py",
        "data/index_market.py",
        "data/load.py",
    )
    optional = (
        "experiments/build_notebook_07.py",
        "tests/test_notebook_07_final_q2_lockbox.py",
    )
    paths = [*relative, *(name for name in optional if (root / name).is_file())]
    return {name.replace("/", "__"): root / name for name in paths}


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, engine="pyarrow")
    os.replace(temporary, path)
    return path


def _utc_index(frame: pd.DataFrame) -> pd.DataFrame:
    current = frame.copy()
    if not isinstance(current.index, pd.DatetimeIndex):
        if "timestamp" not in current:
            raise ValueError("warm-up frame misses a timestamp index")
        current = current.set_index("timestamp")
    current.index = pd.to_datetime(current.index, utc=True)
    current.index.name = "timestamp"
    current = current.sort_index()
    if current.empty or current.index.has_duplicates:
        raise ValueError("warm-up frame must be non-empty with unique timestamps")
    return current


def _build_btc_warmups(code_root: Path, output_root: Path) -> dict[str, Path]:
    from data.build_positioning import build as build_positioning
    from data.load import _read_one, clean
    from features.build import add_features

    source_m15 = code_root / "data" / "btcusdt_m15_lockbox_2026Q1.parquet"
    m15 = _utc_index(pd.read_parquet(source_m15))
    if m15.index.min() < Q1_START or m15.index.max() >= Q2_START:
        raise ValueError("BTC Q1 M15 warm-up source crosses its fixed interval")
    m15_path = output_root / "btc_m15_warmup_pre_q2.parquet"
    copy_audited_pre_q2(
        source_m15,
        m15_path,
        {
            "cutoff_exclusive": Q2_START.isoformat(),
            "end_utc": m15.index.max().isoformat(),
        },
        expected_source_sha256=opaque_sha256(source_m15),
    )

    raw = code_root / "data" / "raw" / "binance"
    minute_frames: list[pd.DataFrame] = []
    for month in (1, 2, 3):
        csv_path = raw / f"BTCUSDT-1m-2026-{month:02d}.csv"
        if csv_path.is_file():
            minute_frames.append(_read_one(csv_path))
            continue
        zip_path = csv_path.with_suffix(".zip")
        if not zip_path.is_file():
            raise FileNotFoundError(zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(members) != 1:
                raise ValueError(f"expected one CSV in {zip_path.name}")
            with archive.open(members[0]) as handle:
                minute_frames.append(_read_one(handle))
    minute = clean(pd.concat(minute_frames), "1m")
    minute = minute.loc[(minute.index >= Q1_START) & (minute.index < Q2_START)]
    if minute.empty or minute.index.max() >= Q2_START:
        raise ValueError("BTC Q1 M1 warm-up crossed Q2")
    minute_path = _atomic_parquet(
        output_root / "btc_m1_warmup_pre_q2.parquet", minute
    )

    positioning = build_positioning(m15.index, end_exclusive=Q2_START)
    positioning = _utc_index(positioning)
    if positioning.index.max() >= Q2_START:
        raise ValueError("BTC positioning warm-up crossed Q2")
    positioning_path = _atomic_parquet(
        output_root / "btc_positioning_warmup_pre_q2.parquet", positioning
    )

    feature_columns = None
    for fit_key in ("btcusdt/none/lstm/w55", "btcusdt/none/svm_linear/w75"):
        manifest = json.loads(
            (RECONSTRUCTION_ROOT / fit_key / "reconstruction_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        columns = tuple(manifest["feature_columns"])
        if feature_columns is None:
            feature_columns = columns
        elif columns != feature_columns:
            raise ValueError("BTC reconstructed members use different feature order")
    joined = m15.join(positioning.reindex(m15.index))
    features = add_features(joined).loc[:, list(feature_columns)].dropna()
    features = features.loc[(features.index >= Q1_START) & (features.index < Q2_START)]
    feature_path = _atomic_parquet(
        output_root / "btc_feature_warmup_pre_q2.parquet", features
    )
    return {
        "btc_m15": m15_path,
        "btc_m1": minute_path,
        "btc_positioning": positioning_path,
        "btc_features": feature_path,
    }


def _copy_index_market_warmups(code_root: Path, output_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for stream in ("usa500", "usatech", "volidx"):
        audit_path = code_root / "data" / f"{stream}_market_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        for grid in ("1m", "15min"):
            source = code_root / "data" / f"{stream}_{grid}_2021_2026.parquet"
            destination = output_root / f"{stream}_{grid}_warmup_pre_q2.parquet"
            copy_audited_pre_q2(
                source,
                destination,
                audit,
                expected_source_sha256=str(audit["grids"][grid]["output_sha256"]),
                audit_path=audit_path,
            )
            paths[f"{stream}_{grid}"] = destination
    return paths


def _copy_sentiment_warmups(code_root: Path, output_root: Path) -> dict[str, Path]:
    raw = code_root / "sentiment" / "raw"
    paths: dict[str, Path] = {}
    for stream in ("usa500", "usatech"):
        for name in (
            f"scores_{stream}.parquet",
            f"scores_llm_{stream}.parquet",
            f"scores_direct_events_{stream}.parquet",
            f"scores_llm_direct_events_{stream}.parquet",
        ):
            source = raw / name
            manifest = source.with_suffix(".manifest.json")
            audit = json.loads(manifest.read_text(encoding="utf-8"))
            destination = output_root / name
            copy_audited_pre_q2(
                source,
                destination,
                audit,
                expected_source_sha256=opaque_sha256(source),
                audit_path=manifest,
            )
            manifest_destination = destination.with_suffix(".manifest.json")
            copy_audited_pre_q2(
                manifest,
                manifest_destination,
                audit,
                expected_source_sha256=opaque_sha256(manifest),
                audit_path=manifest,
            )
            key = name.removesuffix(".parquet")
            paths[key] = destination
            paths[f"{key}_manifest"] = manifest_destination
    return paths


def _build_index_feature_warmups(code_root: Path, output_root: Path) -> dict[str, Path]:
    from experiments.index_replication import IndexReplicationConfig, IndexReplicationRunner
    from experiments.index_replication_protocol import VIX_FEATURE_COLS
    from features.build import FEATURE_COLS, add_features
    from features.index_sentiment import build_matched_index_features

    scratch = code_root / "experiments" / "cache" / "final_q2_lockbox" / "preflight_runner_scratch"
    paths: dict[str, Path] = {}
    for stream, arm, scorer in (
        ("usa500", "deberta_matched", "classic"),
        ("usatech", "deepseek_matched", "llm"),
    ):
        runner = IndexReplicationRunner(
            IndexReplicationConfig.for_stream(stream, output_base=scratch)
        )
        _, price_vix = runner._price_and_vix_frames()
        sentiment = build_matched_index_features(
            stream, runner.bars.index, scorer=scorer
        )
        features = pd.concat([price_vix, sentiment], axis=1, sort=False).replace(
            [np.inf, -np.inf], np.nan
        )
        features = features.loc[(features.index >= Q1_START) & (features.index < Q2_START)].dropna()
        manifest_paths = sorted(
            (RECONSTRUCTION_ROOT / stream / arm).rglob("reconstruction_manifest.json")
        )
        expected_orders = {
            tuple(json.loads(path.read_text(encoding="utf-8"))["feature_columns"])
            for path in manifest_paths
        }
        if expected_orders != {tuple(features.columns)}:
            raise ValueError(f"{stream} Q1 feature warm-up order differs from reconstruction")
        if tuple(features.columns[: len(FEATURE_COLS)]) != tuple(FEATURE_COLS):
            raise ValueError(f"{stream} price feature order changed")
        if not set(VIX_FEATURE_COLS).issubset(features.columns):
            raise ValueError(f"{stream} Q1 feature warm-up misses admitted VIX")
        destination = _atomic_parquet(
            output_root / f"{stream}_feature_warmup_pre_q2.parquet", features
        )
        paths[f"{stream}_features"] = destination
    return paths


def _write_warmup_provenance(
    code_root: Path, warmups: Mapping[str, Path], destination: Path
) -> Path:
    from data.build_positioning import FUND_DIR, MET_DIR, _files_before

    def identities(paths: Sequence[Path]) -> list[dict[str, Any]]:
        return [opaque_file_identity(path).to_dict() for path in paths]

    def record(
        key: str,
        *,
        method: str,
        sources: Sequence[Path],
        audits: Sequence[Path] = (),
    ) -> dict[str, Any]:
        target = warmups[key]
        entry: dict[str, Any] = {
            "method": method,
            "cutoff_exclusive": Q2_START.isoformat(),
            "destination": opaque_file_identity(target).to_dict(),
            "sources": identities(tuple(sources)),
            "source_audits": identities(tuple(audits)),
        }
        if target.suffix == ".parquet":
            frame = pd.read_parquet(target)
            index = pd.to_datetime(frame.index, utc=True)
            if index.empty or index.max() >= Q2_START:
                raise ValueError(f"warm-up provenance crossed Q2: {key}")
            entry["maximum_timestamp"] = index.max().isoformat()
            entry["rows"] = int(len(frame))
        return entry

    raw = code_root / "data" / "raw" / "binance"
    provenance: dict[str, Any] = {
        "btc_m15": record(
            "btc_m15",
            method="validated exact-Q1 source plus byte-identical copy",
            sources=[code_root / "data" / "btcusdt_m15_lockbox_2026Q1.parquet"],
        ),
        "btc_m1": record(
            "btc_m1",
            method="decoded only checksum-verified 2026-Q1 monthly archives",
            sources=[
                path
                for month in (1, 2, 3)
                for path in (
                    raw / f"BTCUSDT-1m-2026-{month:02d}.zip",
                    raw / f"BTCUSDT-1m-2026-{month:02d}.zip.CHECKSUM",
                )
            ],
        ),
        "btc_positioning": record(
            "btc_positioning",
            method="filename-filtered pre-Q2 metrics and funding build",
            sources=[
                *_files_before(
                    sorted(MET_DIR.glob("BTCUSDT-metrics-*.csv")), Q2_START
                ),
                *_files_before(
                    sorted(FUND_DIR.glob("BTCUSDT-fundingRate-*.csv")), Q2_START
                ),
                code_root / "data" / "build_positioning.py",
            ],
        ),
        "btc_features": record(
            "btc_features",
            method="causal feature build from bound M15 and positioning warm-ups",
            sources=[
                warmups["btc_m15"],
                warmups["btc_positioning"],
                code_root / "features" / "build.py",
            ],
        ),
    }
    for stream in ("usa500", "usatech", "volidx"):
        audit = code_root / "data" / f"{stream}_market_audit.json"
        for grid in ("1m", "15min"):
            key = f"{stream}_{grid}"
            provenance[key] = record(
                key,
                method="audit-hash-bound byte-identical pre-Q2 grid copy",
                sources=[code_root / "data" / f"{stream}_{grid}_2021_2026.parquet"],
                audits=[audit],
            )
    score_names: dict[str, list[str]] = {"usa500": [], "usatech": []}
    for stream in score_names:
        for name in (
            f"scores_{stream}",
            f"scores_llm_{stream}",
            f"scores_direct_events_{stream}",
            f"scores_llm_direct_events_{stream}",
        ):
            manifest_key = f"{name}_manifest"
            source = code_root / "sentiment" / "raw" / f"{name}.parquet"
            audit = source.with_suffix(".manifest.json")
            provenance[name] = record(
                name,
                method="score-manifest-bound byte-identical pre-Q2 copy",
                sources=[source],
                audits=[audit],
            )
            provenance[manifest_key] = record(
                manifest_key,
                method="byte-identical source score-manifest copy",
                sources=[audit],
                audits=[audit],
            )
            score_names[stream].extend([name, manifest_key])
    for stream in ("usa500", "usatech"):
        key = f"{stream}_features"
        provenance[key] = record(
            key,
            method="causal frozen-arm feature snapshot from bound market/VIX/scores",
            sources=[
                warmups[f"{stream}_15min"],
                warmups["volidx_15min"],
                *(warmups[name] for name in score_names[stream]),
                code_root / "features" / "build.py",
                code_root / "features" / "index_sentiment.py",
                code_root / "experiments" / "index_replication_protocol.py",
            ],
        )
    if set(provenance) != set(warmups):
        raise AssertionError("warm-up provenance registry does not cover every artifact")
    return write_manifest(
        destination,
        {
            "schema_version": "final-q2-warmup-provenance-v1",
            "q2_decoded": False,
            "artifacts": provenance,
        },
    )


def materialize_pre_q2_warmups(
    code_root: str | Path = CODE_ROOT,
    *,
    data_root: str | Path = WARMUP_DATA_ROOT,
    sentiment_root: str | Path = WARMUP_SENTIMENT_ROOT,
) -> dict[str, Path]:
    """Create/copy only cutoff-proven pre-Q2 context required by frozen models."""
    root = Path(code_root)
    data_output = Path(data_root)
    sentiment_output = Path(sentiment_root)
    warmups = {
        **_build_btc_warmups(root, data_output),
        **_copy_index_market_warmups(root, data_output),
        **_copy_sentiment_warmups(root, sentiment_output),
        **_build_index_feature_warmups(root, data_output),
    }
    provenance = _write_warmup_provenance(
        root, warmups, data_output / "warmup_provenance.json"
    )
    return {**warmups, "warmup_provenance": provenance}


def verify_model_identities(*, verify_deepseek_live: bool = True) -> dict[str, dict[str, Any]]:
    from huggingface_hub.constants import HF_HUB_CACHE
    from sentiment.index_scoring import load_saved_deepseek_identity

    snapshot = (
        Path(HF_HUB_CACHE)
        / "models--mrm8488--deberta-v3-ft-financial-news-sentiment-analysis"
        / "snapshots"
        / DEBERTA_REVISION
    )
    deberta_files: dict[str, dict[str, Any]] = {}
    for name, expected in DEBERTA_FILE_HASHES.items():
        path = snapshot / name
        actual = opaque_sha256(path)
        if actual != expected:
            raise ValueError(f"DeBERTa snapshot hash changed: {name}")
        deberta_files[name] = _identity(path)

    verify = None if verify_deepseek_live else (lambda identity: None)
    deepseek = load_saved_deepseek_identity(verify_identity=verify) if verify is not None else load_saved_deepseek_identity()
    deepseek_payload = asdict(deepseek)
    for key, expected in DEEPSEEK_EXPECTED.items():
        if deepseek_payload.get(key) != expected:
            raise ValueError(f"DeepSeek identity field changed: {key}")
    identity_path = CODE_ROOT / "sentiment" / "raw" / "index_deepseek_identity.json"
    return {
        "deberta": {
            "model": "mrm8488/deberta-v3-ft-financial-news-sentiment-analysis",
            "revision": DEBERTA_REVISION,
            "files": deberta_files,
        },
        "deepseek": {
            **deepseek_payload,
            "identity_file": _identity(identity_path),
            "live_verified": bool(verify_deepseek_live),
        },
    }


def _git(*arguments: str, cwd: Path = REPOSITORY_ROOT) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=cwd, text=True, encoding="utf-8"
    ).strip()


def _assert_critical_tree_clean(paths: Mapping[str, Path]) -> None:
    relative = [path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix() for path in paths.values()]
    status = _git("status", "--porcelain", "--", *relative)
    if status:
        raise ValueError(f"lockbox-critical working tree is not clean:\n{status}")


def _verify_candidate_artifacts(protocol, code_root: Path) -> None:
    checked: dict[Path, str] = {}
    for candidate in protocol.candidates:
        for artifact in candidate.artifacts:
            path = code_root / artifact.path
            previous = checked.get(path)
            if previous is not None and previous != artifact.sha256:
                raise ValueError(f"registered artifact has conflicting hashes: {path}")
            if opaque_sha256(path) != artifact.sha256:
                raise ValueError(f"registered candidate artifact changed: {path}")
            checked[path] = artifact.sha256


def preflight(
    *,
    protocol_path: str | Path,
    write_manifest_path: str | Path,
    code_root: str | Path = CODE_ROOT,
    verify_deepseek_live: bool = True,
    materialize_warmups: bool = True,
) -> dict[str, Any]:
    """Run the complete pre-open identity check and write no decoded Q2 value."""
    from experiments.final_q2_lockbox_contract import canonical_hash, load_lockbox_protocol
    from experiments.final_q2_lockbox_state import GLOBAL_SENTINEL_PATH

    root = Path(code_root)
    if GLOBAL_SENTINEL_PATH.exists():
        raise PermissionError("Q2 is already opened; sealed preflight is unavailable")
    protocol_file = Path(protocol_path)
    protocol = load_lockbox_protocol(protocol_file)
    sources = q2_source_registry(root)
    if any(not path.is_file() for path in sources.values()):
        missing = [path for path in sources.values() if not path.is_file()]
        raise FileNotFoundError(missing[0])
    verify_provider_checksums(sources)
    warmups = (
        materialize_pre_q2_warmups(root)
        if materialize_warmups
        else {
            path.stem: path
            for path in sorted(WARMUP_DATA_ROOT.glob("*.parquet"))
        }
    )
    if not warmups or any(not path.is_file() for path in warmups.values()):
        raise FileNotFoundError("pre-Q2 warm-up registry is incomplete")
    _verify_candidate_artifacts(protocol, root)
    reconstructions = reconstruction_manifest_paths(
        root / "experiments" / "cache" / "final_q2_lockbox" / "reconstructed_models"
    )
    critical = critical_source_registry(root)
    _assert_critical_tree_clean(critical)
    model_identities = verify_model_identities(
        verify_deepseek_live=verify_deepseek_live
    )
    protocol_payload = json.loads(protocol_file.read_text(encoding="utf-8"))
    manifest = build_prelockbox_manifest(
        implementation_commit=_git("rev-parse", "HEAD"),
        protocol_path=protocol_file,
        protocol_hash=canonical_hash(protocol_payload),
        candidates=[asdict(candidate) for candidate in protocol.candidates],
        costs={stream: asdict(cost) for stream, cost in protocol.costs.items()},
        q2_sources=sources,
        warmups=warmups,
        reconstruction_manifests=reconstructions,
        critical_sources=critical,
        model_identities=model_identities,
    )
    if len(manifest["reconstructed_estimators"]) != 21:
        raise AssertionError("preflight manifest did not bind 21 estimators")
    write_manifest(write_manifest_path, manifest)
    return manifest


def _reconstruction_entry(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("q2_decoded") is not False or not payload.get("fit_key"):
        raise ValueError(f"invalid pre-Q2 reconstruction manifest: {manifest_path}")
    estimator = Path(payload["serialized_estimator"])
    rebuilt = Path(payload["rebuilt_panel"])
    if opaque_sha256(estimator) != payload.get("serialized_estimator_sha256"):
        raise ValueError(f"reconstructed estimator hash changed: {payload['fit_key']}")
    if opaque_sha256(rebuilt) != payload.get("rebuilt_panel_sha256"):
        raise ValueError(f"rebuilt panel hash changed: {payload['fit_key']}")
    return {
        "fit_key": str(payload["fit_key"]),
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": opaque_sha256(manifest_path),
        "serialized_estimator": estimator.as_posix(),
        "serialized_estimator_sha256": str(payload["serialized_estimator_sha256"]),
        "rebuilt_panel": rebuilt.as_posix(),
        "rebuilt_panel_sha256": str(payload["rebuilt_panel_sha256"]),
        "feature_order_sha256": payload.get("feature_order_sha256"),
        "fit_cutoff": payload.get("fit_cutoff"),
    }


def build_prelockbox_manifest(
    *,
    implementation_commit: str,
    protocol_path: str | Path,
    protocol_hash: str,
    candidates: Sequence[Mapping[str, Any]],
    costs: Mapping[str, Mapping[str, float]],
    q2_sources: Mapping[str, str | Path],
    warmups: Mapping[str, str | Path],
    reconstruction_manifests: Sequence[str | Path],
    critical_sources: Mapping[str, str | Path],
    model_identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a complete manifest using hashes/bytes only for every Q2 source."""
    if not _is_hex(implementation_commit, 40):
        raise ValueError("implementation_commit must be a 40-character commit hash")
    if not _is_hex(protocol_hash, 64):
        raise ValueError("protocol_hash must be a 64-character sha256")
    reconstructed = sorted(
        (_reconstruction_entry(path) for path in reconstruction_manifests),
        key=lambda row: row["fit_key"],
    )
    if len({row["fit_key"] for row in reconstructed}) != len(reconstructed):
        raise ValueError("reconstruction fit keys must be unique")
    return {
        "schema_version": MANIFEST_SCHEMA,
        "q2_decoded": False,
        "implementation_commit": implementation_commit.lower(),
        "protocol": {
            "path": Path(protocol_path).as_posix(),
            "sha256": opaque_sha256(protocol_path),
            "protocol_hash": protocol_hash.lower(),
        },
        "candidates": [dict(candidate) for candidate in candidates],
        "costs": {
            str(stream): {str(key): float(value) for key, value in values.items()}
            for stream, values in sorted(costs.items())
        },
        "q2_sources": {
            key: _identity(path) for key, path in sorted(q2_sources.items())
        },
        "warmups": {
            key: _identity(path) for key, path in sorted(warmups.items())
        },
        "reconstructed_estimators": reconstructed,
        "critical_sources": {
            key: _identity(path) for key, path in sorted(critical_sources.items())
        },
        "model_identities": {
            str(key): dict(value) for key, value in sorted(model_identities.items())
        },
        "metrics": {
            "calendar": "91 zero-filled UTC days",
            "annualisation": 365,
            "bootstrap_blocks": "Monday-Sunday UTC with zero-padded boundaries",
            "bootstrap_replicates": 5_000,
            "bootstrap_seed": 42,
            "primary_estimand": "btc_union_minus_lstm_total_net",
        },
    }


def write_manifest(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    os.replace(temporary, destination)
    return destination


def validate_manifest_commit_binding(
    manifest_path: str | Path,
    *,
    manifest_commit: str,
    parent_commit: str,
    committed_manifest_bytes: bytes,
) -> ManifestCommitBinding:
    """Bind a manifest-only child commit to its implementation parent and bytes."""
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    implementation = str(payload.get("implementation_commit", "")).lower()
    if implementation != str(parent_commit).lower():
        raise ValueError("manifest commit parent differs from implementation_commit")
    if not _is_hex(manifest_commit, 40) or not _is_hex(parent_commit, 40):
        raise ValueError("manifest/parent commits must be 40-character hashes")
    current = path.read_bytes()
    if current != committed_manifest_bytes:
        raise ValueError("working manifest bytes differ from the manifest commit")
    return ManifestCommitBinding(
        implementation_commit=implementation,
        manifest_commit=str(manifest_commit).lower(),
        manifest_sha256=hashlib.sha256(current).hexdigest(),
    )


__all__ = [
    "DEBERTA_FILE_HASHES",
    "DEBERTA_REVISION",
    "DEEPSEEK_EXPECTED",
    "MANIFEST_SCHEMA",
    "ManifestCommitBinding",
    "build_prelockbox_manifest",
    "critical_source_registry",
    "materialize_pre_q2_warmups",
    "preflight",
    "q2_source_registry",
    "reconstruction_manifest_paths",
    "validate_manifest_commit_binding",
    "verify_model_identities",
    "verify_provider_checksums",
    "write_manifest",
]
