from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import experiments.final_q2_lockbox_state as state_module
from experiments.final_q2_lockbox_contract import Q2_END, Q2_START
from experiments.final_q2_lockbox_inputs import (
    collect_q2_archive_identities,
    load_q2_index_minutes,
    load_q2_parquet_partition,
    load_exact_q2_csv_pair,
    copy_audited_pre_q2,
    load_exact_q2_parquet,
    opaque_sha256,
)
from experiments.final_q2_lockbox_state import OpeningIdentity, begin_global_open


def _identity(**changes) -> OpeningIdentity:
    base = OpeningIdentity(
        implementation_commit="a" * 40,
        manifest_commit="b" * 40,
        protocol_hash="c" * 64,
        manifest_sha256="d" * 64,
        q2_source_hashes={"fixture": "e" * 64},
    )
    return replace(base, **changes)


def test_preopen_sealer_never_calls_parquet_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.parquet"
    source.write_bytes(b"opaque parquet bytes")
    audit = {
        "cutoff_exclusive": "2026-04-01T00:00:00+00:00",
        "end_utc": "2026-03-31T23:59:00+00:00",
    }
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("decoded")),
    )

    identity = copy_audited_pre_q2(
        source,
        tmp_path / "warmup_pre_q2.parquet",
        audit,
        expected_source_sha256=opaque_sha256(source),
    )

    assert identity.source_sha256 == identity.destination_sha256
    assert identity.source_size == identity.destination_size == len(source.read_bytes())


def test_pre_q2_copy_rejects_a_missing_or_late_cutoff(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    source.write_bytes(b"source")

    with pytest.raises(ValueError, match="cutoff"):
        copy_audited_pre_q2(source, tmp_path / "copy.parquet", {})
    with pytest.raises(ValueError, match="pre-Q2"):
        copy_audited_pre_q2(
            source,
            tmp_path / "copy.parquet",
            {
                "cutoff_exclusive": "2026-04-01T00:00:00+00:00",
                "end_utc": "2026-04-01T00:00:00+00:00",
            },
        )


def test_opaque_archive_identity_uses_raw_bytes_only(tmp_path: Path) -> None:
    first = tmp_path / "a.zip"
    second = tmp_path / "b.zip"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    identities = collect_q2_archive_identities([second, first])

    assert [Path(item.path).name for item in identities] == ["a.zip", "b.zip"]
    assert identities[0].sha256 == hashlib.sha256(b"first").hexdigest()
    assert opaque_sha256(second) == hashlib.sha256(b"second").hexdigest()


def test_postopen_loader_requires_the_physical_repository_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "q2.parquet"
    pd.DataFrame(
        {"timestamp": [Q2_START], "value": [1.0]}
    ).to_parquet(path, index=False)
    monkeypatch.setattr(state_module, "GLOBAL_SENTINEL_PATH", tmp_path / "OPENED.json")

    with pytest.raises(PermissionError, match="OPENED"):
        load_exact_q2_parquet(path, identity=_identity(), start=Q2_START, end=Q2_END)


def test_postopen_loader_is_half_open_and_unique(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity()
    monkeypatch.setattr(state_module, "GLOBAL_SENTINEL_PATH", tmp_path / "OPENED.json")
    begin_global_open(identity, root=tmp_path)
    path = tmp_path / "q2.parquet"
    expected = pd.DataFrame(
        {
            "timestamp": [Q2_START, Q2_END - pd.Timedelta(minutes=15)],
            "value": [1.0, 2.0],
        }
    )
    expected.to_parquet(path, index=False)

    loaded = load_exact_q2_parquet(
        path, identity=identity, start=Q2_START, end=Q2_END
    )

    assert loaded.index.tolist() == expected["timestamp"].tolist()
    assert loaded["value"].tolist() == [1.0, 2.0]


def test_postopen_loader_rejects_end_boundary_and_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity()
    monkeypatch.setattr(state_module, "GLOBAL_SENTINEL_PATH", tmp_path / "OPENED.json")
    begin_global_open(identity, root=tmp_path)
    path = tmp_path / "bad.parquet"
    pd.DataFrame(
        {
            "timestamp": [Q2_START, Q2_START, Q2_END],
            "value": [1.0, 2.0, 3.0],
        }
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="interval|unique"):
        load_exact_q2_parquet(path, identity=identity, start=Q2_START, end=Q2_END)


def test_postopen_partition_loader_pushes_down_the_exact_q2_predicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity()
    monkeypatch.setattr(state_module, "GLOBAL_SENTINEL_PATH", tmp_path / "OPENED.json")
    begin_global_open(identity, root=tmp_path)
    path = tmp_path / "broad.parquet"
    index = pd.DatetimeIndex(
        [Q2_START - pd.Timedelta(minutes=1), Q2_START, Q2_END - pd.Timedelta(minutes=1), Q2_END],
        name="timestamp",
    )
    pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0]}, index=index).to_parquet(path)

    loaded = load_q2_parquet_partition(
        path, identity=identity, start=Q2_START, end=Q2_END
    )

    assert loaded.index.tolist() == [Q2_START, Q2_END - pd.Timedelta(minutes=1)]
    assert loaded["value"].tolist() == [1.0, 2.0]


def test_postopen_jforex_loader_filters_before_numeric_decode_and_builds_midpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity()
    monkeypatch.setattr(state_module, "GLOBAL_SENTINEL_PATH", tmp_path / "OPENED.json")
    begin_global_open(identity, root=tmp_path)
    bid_path, ask_path = tmp_path / "bid.csv", tmp_path / "ask.csv"
    rows = ["2026.04.01 02:59:00", "2026.04.01 03:00:00", "2026.07.01 03:00:00"]
    base = pd.DataFrame(
        {
            "Time (EET)": rows,
            "Open": [99.0, 100.0, 999.0],
            "High": [99.5, 101.0, 999.0],
            "Low": [98.5, 99.0, 999.0],
            "Close": [99.0, 100.5, 999.0],
            "Volume": [1.0, 2.0, 3.0],
        }
    )
    base.to_csv(bid_path, index=False)
    base.assign(Open=[99.2, 100.2, 999.2], High=[99.7, 101.2, 999.2], Low=[98.7, 99.2, 999.2], Close=[99.2, 100.7, 999.2]).to_csv(ask_path, index=False)

    minute = load_q2_index_minutes(
        bid_path,
        ask_path,
        identity=identity,
        start=Q2_START,
        end=Q2_END,
        instrument="USA500IDXUSD",
    )

    assert minute.index.tolist() == [Q2_START]
    assert minute.loc[Q2_START, "open"] == pytest.approx(100.1)
    assert minute.loc[Q2_START, "close"] == pytest.approx(100.6)
    assert minute.loc[Q2_START, "available_at"] == Q2_START + pd.Timedelta(minutes=1)


def test_postopen_csv_pair_requires_identical_timestamp_grids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity()
    monkeypatch.setattr(state_module, "GLOBAL_SENTINEL_PATH", tmp_path / "OPENED.json")
    begin_global_open(identity, root=tmp_path)
    bid_path = tmp_path / "bid.csv"
    ask_path = tmp_path / "ask.csv"
    pd.DataFrame({"timestamp": [Q2_START], "close": [100.0]}).to_csv(
        bid_path, index=False
    )
    pd.DataFrame({"timestamp": [Q2_START], "close": [100.2]}).to_csv(
        ask_path, index=False
    )

    paired = load_exact_q2_csv_pair(
        bid_path,
        ask_path,
        identity=identity,
        start=Q2_START,
        end=Q2_END,
        timestamp_column="timestamp",
    )
    assert paired.loc[Q2_START, "close_bid"] == 100.0
    assert paired.loc[Q2_START, "close_ask"] == 100.2

    pd.DataFrame(
        {"timestamp": [Q2_START + pd.Timedelta(minutes=1)], "close": [100.2]}
    ).to_csv(ask_path, index=False)
    with pytest.raises(ValueError, match="timestamp grids"):
        load_exact_q2_csv_pair(
            bid_path,
            ask_path,
            identity=identity,
            start=Q2_START,
            end=Q2_END,
            timestamp_column="timestamp",
        )


def test_copy_identity_is_json_serializable(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"abc")
    identity = copy_audited_pre_q2(
        source,
        tmp_path / "warmup_pre_q2.bin",
        {
            "cutoff_exclusive": "2026-04-01T00:00:00+00:00",
            "end_utc": "2026-03-31T23:59:00+00:00",
        },
        expected_source_sha256=opaque_sha256(source),
    )

    assert json.dumps(identity.to_dict(), sort_keys=True)
