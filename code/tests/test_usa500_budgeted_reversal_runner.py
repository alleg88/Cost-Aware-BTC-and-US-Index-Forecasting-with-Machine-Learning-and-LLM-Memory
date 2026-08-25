from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.run_usa500_budgeted_reversal_agent import (
    prepare_common_artifacts,
    finalize_results,
    freeze_h1_policies,
    run_controls,
    run_preflight,
    run_score_variant,
)
from experiments.run_usa500_reflection_weight_agent import (
    _implementation_hash as source_implementation_hash,
)
from reflection_agent.index_v2.contracts import ReversalScoreBatch


CODE_ROOT = Path(__file__).parents[1]
CONFIG = CODE_ROOT / "configs" / "usa500_budgeted_reversal_agent_v1.yaml"
MODEL = "deepseek-v4-flash:0731-cloud"
DIGEST = "d3f1c87447216481a8001f48c517a51e13bfb141853a8df5e52f81bf765dabc3"
AGENT_VARIANTS = (
    "budgeted_real_memory",
    "budgeted_no_memory",
    "budgeted_shuffled_memory",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _opportunities(
    prefix: str,
    first_week: str,
    *,
    first_week_rows: int,
    second_week_rows: int,
) -> pd.DataFrame:
    total = first_week_rows + second_week_rows
    week_one = pd.Timestamp(first_week)
    week_two = week_one + pd.Timedelta(days=7)
    rows = []
    for index in range(total):
        week = week_one if index < first_week_rows else week_two
        within = index if index < first_week_rows else index - first_week_rows
        signal = week + pd.Timedelta(hours=14, minutes=15 * within)
        entry = signal + pd.Timedelta(minutes=15)
        exit_ = entry + pd.Timedelta(minutes=15)
        side = 1 if index % 2 == 0 else -1
        entry_price = 100.0 + index
        exit_price = entry_price * (1.0 + side * 0.001)
        gross = side * (exit_price / entry_price - 1.0)
        row = {
            "opportunity_id": f"{prefix}-{index:03d}",
            "signal_bar_open": signal,
            "decision_time": entry,
            "entry_bar_open": entry,
            "exit_bar_open": entry,
            "entry_time": entry,
            "exit_time": exit_,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "confidence": 0.8,
            "gross_return": gross,
            "cost_return": 0.0002,
            "net_return": gross - 0.0002,
            "original_side": side,
            "outcome_available_at": exit_,
            "week_start": week,
            "y_true": 2 if side == 1 else 0,
            "state_vix_regime": 0.1 * index,
            "state_trailing_vol": 0.01,
            "state_trailing_trend": -0.001 * index,
        }
        for model_index in range(9):
            long = 0.65 if (index + model_index) % 3 else 0.35
            short = 0.35 if long == 0.65 else 0.65
            row[f"m{model_index:02d}_p_short"] = short
            row[f"m{model_index:02d}_p_flat"] = 0.0
            row[f"m{model_index:02d}_p_long"] = long
        rows.append(row)
    return pd.DataFrame(rows)


def _memory_cards(frame: pd.DataFrame) -> list[dict]:
    cards = []
    for number, (week_start, current) in enumerate(frame.groupby("week_start", sort=True)):
        cards.append(
            {
                "week_id": f"W-{number}",
                "week_start": pd.Timestamp(week_start).isoformat(),
                "available_at": pd.to_datetime(
                    current["outcome_available_at"], utc=True
                ).max().isoformat(),
                "opportunities": int(len(current)),
                "model_statistics": [
                    {
                        "model_index": model_index,
                        "sample_count": int(len(current)),
                        "directional_accuracy": 0.40 + model_index / 100.0,
                        "net_return": 0.001 * model_index,
                        "long_net_return": 0.0005 * model_index,
                        "short_net_return": 0.0005 * model_index,
                        "brier_score": 0.50,
                        "mean_confidence": 0.60,
                        "marginal_net_vs_equal": 0.0,
                    }
                    for model_index in range(9)
                ],
                "controls": {"original": {"net_return": 0.0}},
                "market_state": {
                    "vix_regime": 0.1,
                    "trailing_volatility": 0.01,
                    "trailing_trend": 0.0,
                },
                "agreement": 0.8,
                "probability_dispersion": 0.1,
            }
        )
    return cards


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source_root(
    tmp_path: Path,
    *,
    q2: bool = False,
    h1_counts: tuple[int, int] = (12, 1),
    forward_counts: tuple[int, int] = (10, 1),
) -> Path:
    root = tmp_path / "source"
    common = root / "common"
    common.mkdir(parents=True)
    h1 = _opportunities(
        "H1",
        "2025-01-06T00:00:00Z",
        first_week_rows=h1_counts[0],
        second_week_rows=h1_counts[1],
    )
    forward = _opportunities(
        "FWD",
        "2025-07-07T00:00:00Z",
        first_week_rows=forward_counts[0],
        second_week_rows=forward_counts[1],
    )
    if q2:
        forward.loc[0, "signal_bar_open"] = pd.Timestamp("2026-04-01T00:00:00Z")
    h1.to_parquet(common / "h1_opportunities.parquet", index=False)
    forward.to_parquet(common / "forward_opportunities.parquet", index=False)
    _write_jsonl(common / "h1_memory_cards.jsonl", _memory_cards(h1))
    _write_jsonl(common / "forward_base_memory_cards.jsonl", _memory_cards(forward))
    pd.DataFrame(
        {
            "week_start": pd.to_datetime(forward["week_start"], utc=True).unique(),
            "state_available_at": [
                pd.Timestamp(item) - pd.Timedelta(minutes=1)
                for item in pd.to_datetime(forward["week_start"], utc=True).unique()
            ],
            "state_vix_regime": [0.1, 0.2],
            "state_trailing_vol": [0.01, 0.02],
            "state_trailing_trend": [0.0, -0.01],
        }
    ).to_parquet(common / "forward_weekly_state.parquet", index=False)
    protocol_hash = "a" * 64
    protocol = {"protocol_hash": protocol_hash, "q2_loaded": False}
    (common / "protocol.json").write_text(
        json.dumps(protocol, sort_keys=True), encoding="utf-8"
    )
    artifact_names = (
        "protocol.json",
        "h1_opportunities.parquet",
        "forward_opportunities.parquet",
        "h1_memory_cards.jsonl",
        "forward_base_memory_cards.jsonl",
        "forward_weekly_state.parquet",
    )
    manifest = {
        "status": "complete",
        "protocol_hash": protocol_hash,
        "implementation_hash": source_implementation_hash(),
        "artifact_hashes": {
            name: _sha256(common / name) for name in artifact_names
        },
        "stage_counts": {"h1": len(h1), "forward": len(forward)},
        "q2_loaded": False,
    }
    (common / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return root


class RecordingCaller:
    def __init__(self, *, malformed: bool = False) -> None:
        self.calls: list[dict] = []
        self.malformed = malformed

    def call(self, *, role, messages, response_model, allowed_ids):
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "allowed_ids": allowed_ids,
            }
        )
        count = len(allowed_ids["opportunity_indices"])
        payload = json.loads(messages[-1]["content"].split("INPUT_JSON=", 1)[1])
        priors = [
            int(row["uncertainty_prior"])
            for row in payload["opportunities"]
        ]
        value = None
        if not self.malformed:
            value = ReversalScoreBatch.model_validate(
                {
                    "schema_version": "1.0",
                    "decisions": [
                        {
                            "opportunity_index": index,
                            "reversal_score": priors[index],
                            "evidence_indices": [],
                            "memory_indices": [],
                        }
                        for index in range(count)
                    ],
                }
            )
        return SimpleNamespace(
            status="success" if value is not None else "schema_failure",
            value=value,
            raw_content="{}",
            request_hash=f"{len(self.calls):064x}",
            response_hash=f"{len(self.calls) + 1:064x}",
            schema_hash="c" * 64,
            attempts=1,
            latency_seconds=0.01,
            metadata={},
            errors=(),
        )


def _prepare(tmp_path: Path) -> tuple[Path, Path]:
    source = _source_root(tmp_path)
    root = tmp_path / "target"
    prepare_common_artifacts(
        source_root=source, output_root=root, config_path=CONFIG
    )
    caller = RecordingCaller()
    run_preflight(
        output_root=root,
        config_path=CONFIG,
        caller=caller,
        model_record={
            "model": MODEL,
            "digest": DIGEST,
            "capabilities": ["thinking"],
            "ollama_version": "test",
        },
    )
    return source, root


def test_prepare_mirrors_exact_h1_forward_keys_and_seals_q2(tmp_path: Path):
    source = _source_root(tmp_path)
    root = tmp_path / "target"

    manifest = prepare_common_artifacts(
        source_root=source, output_root=root, config_path=CONFIG
    )

    assert manifest["stage_counts"] == {"h1": 13, "forward": 11}
    assert manifest["q2_loaded"] is False
    assert (root / "common/source_manifest.json").is_file()
    assert pd.read_parquet(root / "common/forward_opportunities.parquet")[
        "opportunity_id"
    ].tolist() == [f"FWD-{index:03d}" for index in range(11)]


def test_prepare_rejects_source_hash_drift_or_q2_timestamp(tmp_path: Path):
    source = _source_root(tmp_path / "hash")
    with (source / "common/h1_memory_cards.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="hash"):
        prepare_common_artifacts(
            source_root=source,
            output_root=tmp_path / "hash-target",
            config_path=CONFIG,
        )

    q2_source = _source_root(tmp_path / "q2", q2=True)
    with pytest.raises(PermissionError, match="Q2"):
        prepare_common_artifacts(
            source_root=q2_source,
            output_root=tmp_path / "q2-target",
            config_path=CONFIG,
        )


def test_preflight_binds_exact_model_digest_and_score_schema(tmp_path: Path):
    source = _source_root(tmp_path)
    root = tmp_path / "target"
    prepare_common_artifacts(source_root=source, output_root=root, config_path=CONFIG)
    caller = RecordingCaller()

    result = run_preflight(
        output_root=root,
        config_path=CONFIG,
        caller=caller,
        model_record={
            "model": MODEL,
            "digest": DIGEST,
            "capabilities": ["thinking"],
            "ollama_version": "test",
        },
    )

    assert result["passed"] is True
    assert result["model_digest"] == DIGEST
    assert result["score_status"] == "success"
    assert len(caller.calls) == 1


def test_h1_scoring_is_week_causal_bounded_and_pointwise_complete(tmp_path: Path):
    _source, root = _prepare(tmp_path)
    caller = RecordingCaller()

    result = run_score_variant(
        "budgeted_real_memory",
        output_root=root,
        config_path=CONFIG,
        caller=caller,
    )

    assert result["status"] == "complete"
    h1 = pd.read_parquet(root / "budgeted_real_memory/h1_scores.parquet")
    forward = pd.read_parquet(root / "budgeted_real_memory/forward_scores.parquet")
    assert len(h1) == 13 and len(forward) == 11
    assert h1["opportunity_id"].tolist() == [f"H1-{index:03d}" for index in range(13)]
    assert h1["batch_size"].max() <= 10
    first_week = h1.loc[h1["week_start"].eq(h1["week_start"].min())]
    second_week = h1.loc[h1["week_start"].eq(h1["week_start"].max())]
    assert first_week["memory_cards_visible"].eq(0).all()
    assert second_week["memory_cards_visible"].eq(1).all()
    assert pd.Timestamp(second_week["memory_max_available_at"].iloc[0]) < pd.Timestamp(
        second_week["week_start"].iloc[0]
    )
    assert all(
        len(call["allowed_ids"]["opportunity_indices"]) <= 10
        for call in caller.calls
    )
    assert all(
        '"uncertainty_prior":' in call["messages"][1]["content"]
        for call in caller.calls
    )


def test_real_no_and_shuffled_memory_keep_identical_nonmemory_inputs(tmp_path: Path):
    _source, root = _prepare(tmp_path)
    for variant in AGENT_VARIANTS:
        run_score_variant(
            variant,
            output_root=root,
            config_path=CONFIG,
            caller=RecordingCaller(),
        )

    for split in ("h1", "forward"):
        frames = [
            pd.read_parquet(root / variant / f"{split}_scores.parquet")
            for variant in AGENT_VARIANTS
        ]
        hashes = [frame["nonmemory_hash"].tolist() for frame in frames]
        assert hashes[0] == hashes[1] == hashes[2]
        assert frames[1]["memory_cards_visible"].eq(0).all()
        assert frames[0]["memory_cards_visible"].tolist() == frames[2][
            "memory_cards_visible"
        ].tolist()


def test_resume_continues_after_checkpoint_without_repeating_batch(tmp_path: Path):
    _source, root = _prepare(tmp_path)
    first = RecordingCaller()

    partial = run_score_variant(
        "budgeted_real_memory",
        output_root=root,
        config_path=CONFIG,
        caller=first,
        max_batches=1,
    )
    assert partial["status"] == "partial"
    assert len(first.calls) == 1

    second = RecordingCaller()
    complete = run_score_variant(
        "budgeted_real_memory",
        output_root=root,
        config_path=CONFIG,
        caller=second,
    )
    assert complete["status"] == "complete"
    assert len(second.calls) >= 1
    scores = pd.read_parquet(root / "budgeted_real_memory/h1_scores.parquet")
    assert scores["opportunity_id"].is_unique
    assert len(scores) == 13


def test_resume_rejects_checkpoint_identity_drift(tmp_path: Path):
    _source, root = _prepare(tmp_path)
    run_score_variant(
        "budgeted_real_memory",
        output_root=root,
        config_path=CONFIG,
        caller=RecordingCaller(),
        max_batches=1,
    )
    path = root / "budgeted_real_memory/checkpoint.json"
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    checkpoint["protocol_hash"] = "0" * 64
    path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(ValueError, match="identity"):
        run_score_variant(
            "budgeted_real_memory",
            output_root=root,
            config_path=CONFIG,
            caller=RecordingCaller(),
        )


def test_invalid_output_uses_current_uncertainty_score_and_records_fallback(tmp_path: Path):
    _source, root = _prepare(tmp_path)

    result = run_score_variant(
        "budgeted_real_memory",
        output_root=root,
        config_path=CONFIG,
        caller=RecordingCaller(malformed=True),
    )

    assert result["status"] == "complete"
    h1 = pd.read_parquet(root / "budgeted_real_memory/h1_scores.parquet")
    assert h1["fallback_used"].all()
    assert h1["call_status"].eq("schema_failure").all()
    assert h1["reversal_score"].between(0, 1000).all()


def test_h1_freezes_one_rate_and_controls_share_it_before_forward_actions(tmp_path: Path):
    _source, root = _prepare(tmp_path)
    for variant in AGENT_VARIANTS:
        run_score_variant(
            variant,
            output_root=root,
            config_path=CONFIG,
            caller=RecordingCaller(),
        )

    frozen = freeze_h1_policies(output_root=root, config_path=CONFIG)
    result = run_controls(output_root=root, config_path=CONFIG)

    assert frozen["selected_target_rate"] in {0.15, 0.20, 0.25}
    assert set(frozen["policies"]) == set(AGENT_VARIANTS) | {
        "uncertainty_control",
        "seeded_hash_control",
    }
    assert {
        round(policy["target_rate"], 2)
        for policy in frozen["policies"].values()
    } == {round(frozen["selected_target_rate"], 2)}
    assert result["status"] == "complete"
    for variant in (
        "frozen_parent",
        *AGENT_VARIANTS,
        "uncertainty_control",
        "seeded_hash_control",
    ):
        decisions = pd.read_parquet(root / variant / "decisions.parquet")
        assert len(decisions) == 11
        assert decisions["opportunity_id"].tolist() == [
            f"FWD-{index:03d}" for index in range(11)
        ]
        assert decisions["side"].isin((-1, 1)).all()


def _completed_diverse_root(tmp_path: Path) -> Path:
    source = _source_root(
        tmp_path,
        h1_counts=(30, 1),
        forward_counts=(24, 1),
    )
    root = tmp_path / "target"
    prepare_common_artifacts(source_root=source, output_root=root, config_path=CONFIG)
    run_preflight(
        output_root=root,
        config_path=CONFIG,
        caller=RecordingCaller(),
        model_record={
            "model": MODEL,
            "digest": DIGEST,
            "capabilities": ["thinking"],
            "ollama_version": "test",
        },
    )
    for variant in AGENT_VARIANTS:
        run_score_variant(
            variant,
            output_root=root,
            config_path=CONFIG,
            caller=RecordingCaller(),
        )
        h1_path = root / variant / "h1_scores.parquet"
        h1_scores = pd.read_parquet(h1_path)
        h1_scores["reversal_score"] = np.rint(
            np.linspace(0, 1000, len(h1_scores))
        ).astype(int)
        h1_scores.to_parquet(h1_path, index=False)
        forward_path = root / variant / "forward_scores.parquet"
        forward_scores = pd.read_parquet(forward_path)
        forward_scores["reversal_score"] = np.asarray(
            [1000] * 5 + [0] * (len(forward_scores) - 5), dtype=int
        )
        forward_scores.to_parquet(forward_path, index=False)
    freeze_h1_policies(output_root=root, config_path=CONFIG)
    run_controls(output_root=root, config_path=CONFIG)
    return root


def test_finalize_separates_influence_from_economic_promotion(tmp_path: Path):
    root = _completed_diverse_root(tmp_path)

    summary = finalize_results(output_root=root, config_path=CONFIG)

    primary = summary["primary_gate"]
    assert primary["all_registered_opportunities_pass"] is True
    assert primary["minimum_direction_changes_pass"] is True
    assert primary["influence_gate_pass"] is True
    assert primary["economic_gate_pass"] is False
    assert primary["candidate_pass"] is False
    assert summary["q2_loaded"] is False


def test_finalize_outputs_finite_net_ranked_reconciled_artifacts(tmp_path: Path):
    root = _completed_diverse_root(tmp_path)

    finalize_results(output_root=root, config_path=CONFIG)
    results = pd.read_parquet(root / "results_table.parquet")
    quality = pd.read_parquet(root / "change_quality.parquet")
    bootstrap = pd.read_parquet(root / "paired_bootstrap.parquet")
    audit = pd.read_parquet(root / "leakage_audit.parquet")

    assert results["variant"].nunique() == 6
    assert results["trades"].eq(25).all()
    assert results["net_return"].is_monotonic_decreasing
    assert {"daily_sharpe", "daily_sortino"}.issubset(results.columns)
    assert np.isfinite(results.select_dtypes(include=[np.number])).all().all()
    assert (
        quality["beneficial_changes"]
        + quality["harmful_changes"]
        + quality["neutral_changes"]
    ).equals(quality["direction_changes"])
    assert bootstrap["variant"].nunique() == 6
    assert audit["passed"].all()
    assert (root / "manifest.json").is_file()


def test_finalize_rejects_hash_valid_but_economically_corrupted_replay(tmp_path: Path):
    root = _completed_diverse_root(tmp_path)
    ledger_path = root / "budgeted_real_memory" / "ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    ledger.loc[0, "net_return"] += 0.5
    ledger.to_parquet(ledger_path, index=False)
    manifest_path = root / "budgeted_real_memory" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["ledger.parquet"] = _sha256(ledger_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="replay"):
        finalize_results(output_root=root, config_path=CONFIG)
