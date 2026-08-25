from __future__ import annotations

from pathlib import Path

import pytest

from experiments.source_registry import (
    PolicyError,
    SourcePolicy,
    build_source_records,
)


def _rule(
    rule_id: str,
    evidence_class: str,
    glob: str,
    *,
    stage: str,
) -> dict[str, object]:
    return {
        "id": rule_id,
        "class": evidence_class,
        "glob": glob,
        "provider": rule_id,
        "stage": stage,
    }


def test_policy_classifies_each_selected_source_once_and_ignores_derived_rows() -> None:
    policy = SourcePolicy.from_dict(
        {
            "schema_version": 1,
            "rules": [
                _rule(
                    "binance",
                    "public_raw",
                    "data/raw/binance/**/*.zip",
                    stage="data/raw/binance",
                ),
                _rule(
                    "truth",
                    "snapshot_raw",
                    "sentiment/raw/trump_truth_posts.csv",
                    stage="sentiment/raw",
                ),
            ],
        }
    )
    full_manifest = {
        "files": [
            {
                "path": "data/raw/binance/BTCUSDT-15m-2024-01.zip",
                "bytes": 3,
                "sha256": "a" * 64,
            },
            {
                "path": "sentiment/raw/trump_truth_posts.csv",
                "bytes": 4,
                "sha256": "b" * 64,
            },
            {
                "path": "experiments/cache/tuning/prediction.parquet",
                "bytes": 5,
                "sha256": "c" * 64,
            },
        ]
    }

    records = build_source_records(policy, full_manifest)

    assert [(record.path, record.evidence_class) for record in records] == [
        ("data/raw/binance/BTCUSDT-15m-2024-01.zip", "public_raw"),
        ("sentiment/raw/trump_truth_posts.csv", "snapshot_raw"),
    ]
    assert [record.stage_path for record in records] == [
        "data/raw/binance/BTCUSDT-15m-2024-01.zip",
        "sentiment/raw/trump_truth_posts.csv",
    ]


def test_policy_rejects_a_source_that_matches_more_than_one_rule() -> None:
    policy = SourcePolicy.from_dict(
        {
            "schema_version": 1,
            "rules": [
                _rule(
                    "all_csv",
                    "snapshot_raw",
                    "sentiment/raw/*.csv",
                    stage="sentiment/raw",
                ),
                _rule(
                    "truth",
                    "snapshot_raw",
                    "sentiment/raw/trump*.csv",
                    stage="sentiment/raw",
                ),
            ],
        }
    )

    with pytest.raises(PolicyError, match="matched more than one rule"):
        policy.classify("sentiment/raw/trump_truth_posts.csv")


@pytest.mark.parametrize(
    ("field", "value"),
    (("glob", "../*.csv"), ("stage", "../outside")),
)
def test_policy_rejects_paths_that_escape_the_code_root(field: str, value: str) -> None:
    rule = _rule(
        "bad",
        "snapshot_raw",
        "sentiment/raw/*.csv",
        stage="sentiment/raw",
    )
    rule[field] = value

    with pytest.raises(PolicyError, match="escapes"):
        SourcePolicy.from_dict({"schema_version": 1, "rules": [rule]})


def test_policy_expands_brace_alternatives_without_broadening_the_rule() -> None:
    policy = SourcePolicy.from_dict(
        {
            "schema_version": 1,
            "rules": [
                _rule(
                    "q2_direct",
                    "snapshot_raw",
                    "sentiment/raw/q2/{direct_events.parquet,manifest.json}",
                    stage="sentiment/raw/q2",
                )
            ],
        }
    )

    assert policy.classify("sentiment/raw/q2/direct_events.parquet").id == "q2_direct"
    assert policy.classify("sentiment/raw/q2/manifest.json").id == "q2_direct"
    assert policy.classify("sentiment/raw/q2/unregistered.parquet") is None


def test_manifest_row_validation_rejects_invalid_hashes_and_sizes() -> None:
    policy = SourcePolicy.from_dict(
        {
            "schema_version": 1,
            "rules": [
                _rule(
                    "truth",
                    "snapshot_raw",
                    "sentiment/raw/trump_truth_posts.csv",
                    stage="sentiment/raw",
                )
            ],
        }
    )

    with pytest.raises(PolicyError, match="SHA-256"):
        build_source_records(
            policy,
            {
                "files": [
                    {
                        "path": "sentiment/raw/trump_truth_posts.csv",
                        "bytes": 1,
                        "sha256": "not-a-hash",
                    }
                ]
            },
        )
    with pytest.raises(PolicyError, match="byte count"):
        build_source_records(
            policy,
            {
                "files": [
                    {
                        "path": "sentiment/raw/trump_truth_posts.csv",
                        "bytes": -1,
                        "sha256": "d" * 64,
                    }
                ]
            },
        )


@pytest.mark.parametrize(
    "filename",
    (
        "USA500IDXUSD_1 Min_Ask_2021.01.01_2026.08.01.csv",
        "USA500IDXUSD_1 Min_Bid_2021.01.01_2026.08.01.csv",
        "USATECHIDXUSD_1 Min_Ask_2021.01.01_2026.08.01.csv",
        "USATECHIDXUSD_1 Min_Bid_2021.01.01_2026.08.01.csv",
    ),
)
def test_real_policy_registers_extended_index_minute_sources(filename: str) -> None:
    policy = SourcePolicy.load(
        Path(__file__).parents[1] / "configs" / "source_evidence_policy.json"
    )

    rule = policy.classify(f"data/raw/{filename}")

    assert rule is not None
    assert rule.id == "jforex_index_minutes"
    assert rule.evidence_class == "snapshot_raw"
