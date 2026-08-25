from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.index_replication import ARMS as REGISTERED_ARMS, _frame_hash
from experiments.index_replication_protocol import MODEL_NAMES, TAUS, WIDTHS
from experiments.index_replication_protocol import CUTOFF
from experiments.index_trade_coverage import (
    COVERAGE_MIN_POSITIVE_MONTHS,
    COVERAGE_PROTOCOL_VERSION,
    IndexTradeCoverageConfig,
    IndexTradeCoverageRunner,
    coverage_eligibility,
    select_coverage_policies,
    select_coverage_policy,
    validate_h1_grid,
)
from experiments.index_replication_protocol import FORWARD_END, FORWARD_START, daily_economics


ARMS = ("selected_base", "deberta_matched", "deepseek_matched", "deepseek_full")


def _row(**overrides) -> dict:
    row = {
        "arm": "selected_base",
        "model_name": "logreg",
        "width_bps": 10,
        "tau": 0.5,
        "trades": 100,
        "n_long": 50,
        "n_short": 50,
        "positive_months": 3,
        "net_return": 0.02,
        "daily_sharpe": 0.4,
        "daily_sortino": 0.5,
    }
    row.update(overrides)
    return row


def test_coverage_policy_requires_volume_both_sides_stability_and_positive_quality():
    grid = pd.DataFrame(
        [
            _row(model_name="mlp", trades=700, net_return=-0.01),
            _row(model_name="gru", trades=650, n_short=14),
            _row(model_name="svm_linear", trades=620, positive_months=2),
            _row(model_name="lstm", trades=600, n_long=510, n_short=90),
        ]
    )

    eligible = coverage_eligibility(grid)
    selected = select_coverage_policy(grid)

    assert eligible.tolist() == [False, False, False, True]
    assert COVERAGE_MIN_POSITIVE_MONTHS == 3
    assert selected["model_name"] == "lstm"
    assert bool(selected["coverage_eligible"])
    assert selected["selection_rule"].startswith("coverage_eligible,trades")


def test_coverage_policy_ranks_trades_before_quality_and_uses_quality_as_tie_break():
    grid = pd.DataFrame(
        [
            _row(model_name="logreg", trades=100, daily_sortino=8.0),
            _row(model_name="lstm", trades=120, daily_sortino=0.1),
            _row(model_name="gru", trades=120, daily_sortino=0.2),
        ]
    )

    selected = select_coverage_policy(grid)

    assert selected["model_name"] == "gru"
    assert int(selected["trades"]) == 120


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trades", 49),
        ("n_long", 14),
        ("n_short", 14),
        ("positive_months", 2),
        ("net_return", 0.0),
        ("daily_sharpe", 0.0),
        ("daily_sortino", 0.0),
    ],
)
def test_each_coverage_gate_fails_closed(field: str, value: float):
    frame = pd.DataFrame([_row(**{field: value})])

    assert not bool(coverage_eligibility(frame).iloc[0])


def test_coverage_policy_fails_closed_when_no_candidate_is_eligible():
    grid = pd.DataFrame([_row(net_return=-0.01), _row(daily_sortino=-0.1)])

    with pytest.raises(ValueError, match="no coverage-eligible"):
        select_coverage_policy(grid)


def test_selects_exactly_one_policy_per_arm_after_screening_all_models():
    rows = []
    for arm in ARMS:
        for rank, model in enumerate(MODEL_NAMES):
            rows.append(_row(arm=arm, model_name=model, trades=100 + rank))
    selected = select_coverage_policies(pd.DataFrame(rows))

    assert selected["arm"].tolist() == list(ARMS)
    assert selected["model_name"].eq("gru").all()
    assert selected["models_screened"].eq(9).all()
    assert selected["eligible_models"].eq(9).all()


def _complete_grid() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(arm=arm, model_name=model, width_bps=width, tau=tau)
            for arm, model, width, tau in product(ARMS, MODEL_NAMES, WIDTHS, TAUS)
        ]
    )


def test_h1_grid_validation_requires_exact_registered_cartesian_product():
    complete = _complete_grid()

    validate_h1_grid(complete)
    with pytest.raises(ValueError, match="exact registered grid"):
        validate_h1_grid(complete.iloc[:-1])


@pytest.mark.parametrize(
    ("column", "value"),
    [("width_bps", 5.5), ("tau", 0.400000001)],
)
def test_h1_grid_validation_rejects_lossy_near_registered_values(column: str, value: float):
    mutated = _complete_grid()
    mutated[column] = mutated[column].astype(float)
    target = mutated.index[
        mutated["width_bps"].eq(5) & mutated["tau"].eq(0.4)
    ][0]
    mutated.loc[target, column] = value

    with pytest.raises(ValueError, match="exact registered grid"):
        validate_h1_grid(mutated)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_interrupted_resume_is_bound_to_policy_ledgers_and_actual_prediction_time(tmp_path):
    output = tmp_path / "coverage" / "usa500"
    config = IndexTradeCoverageConfig(
        stream="usa500",
        data_dir=tmp_path / "data",
        source_root=tmp_path / "source" / "usa500",
        output_root=output,
    )
    runner = IndexTradeCoverageRunner(config)
    policy = {
        "arm": "selected_base",
        "model_name": "logreg",
        "width_bps": 5,
        "tau": 0.4,
    }
    entry = FORWARD_START + pd.Timedelta(minutes=15)
    ledger = pd.DataFrame(
        {
            "entry_time": [entry],
            "exit_time": [entry + pd.Timedelta(minutes=15)],
            "side": [1],
            "gross_return": [0.011],
            "net_return": [0.010],
        }
    )
    per_bar = pd.DataFrame({"timestamp": [entry], "net_return": [0.010]})
    ledger_path = output / "forward_ledgers" / "selected_base.parquet"
    per_bar_path = output / "forward_ledgers" / "selected_base_per_bar.parquet"
    ledger_path.parent.mkdir(parents=True)
    ledger.to_parquet(ledger_path, index=False)
    per_bar.to_parquet(per_bar_path, index=False)
    economics = daily_economics(
        ledger,
        per_bar.set_index("timestamp")["net_return"],
        start=FORWARD_START,
        end=FORWARD_END,
    )
    prediction_max = FORWARD_START + pd.Timedelta(days=1)
    checkpoint = {
        "protocol_hash": "protocol",
        "selected_policy_sha256": "selection",
        "policy": policy,
        "prediction_max_timestamp": prediction_max.isoformat(),
        "artifacts": {
            "ledger": {
                "path": "forward_ledgers/selected_base.parquet",
                "sha256": _sha256(ledger_path),
            },
            "per_bar": {
                "path": "forward_ledgers/selected_base_per_bar.parquet",
                "sha256": _sha256(per_bar_path),
            },
        },
        "summary": {
            "stream": "usa500",
            "evidence_role": "secondary_reused_forward",
            **policy,
            **economics,
        },
    }
    checkpoint_path = output / "forward" / "selected_base.json"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    summary, resumed_max = runner._completed_arm(
        policy, "protocol", "selection"
    )
    assert summary["trades"] == 1
    assert resumed_max == prediction_max
    assert runner._completed_arm(
        {**policy, "arm": "deberta_matched"}, "protocol", "selection"
    ) is None
    with pytest.raises(ValueError, match="policy tuple"):
        runner._completed_arm({**policy, "tau": 0.5}, "protocol", "selection")

    ledger.assign(net_return=0.02).to_parquet(ledger_path, index=False)
    with pytest.raises(ValueError, match="artifact hash"):
        runner._completed_arm(policy, "protocol", "selection")


@pytest.mark.parametrize("stream", ("usa500", "usatech"))
def test_completed_coverage_artifacts_reconcile_and_remain_pre_q2(stream: str):
    code_root = Path(__file__).parents[1]
    source_root = code_root / "experiments" / "cache" / "index_replication" / stream
    root = code_root / "experiments" / "cache" / "index_trade_coverage" / stream
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    source_protocol = json.loads(
        (source_root / "protocol_manifest.json").read_text(encoding="utf-8")
    )

    assert protocol["protocol_version"] == COVERAGE_PROTOCOL_VERSION
    assert result["protocol_version"] == COVERAGE_PROTOCOL_VERSION
    assert result["resumed_forward_arms"] == 4
    assert len(list((root / "forward").glob("*.json"))) == 4
    assert len(list((root / "forward_ledgers").glob("*.parquet"))) == 8
    for name, expected in manifest["artifacts"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    source_grid = pd.read_parquet(source_root / "h1_policy_grid.parquet")
    selected = pd.read_parquet(root / "h1_selected_policies.parquet")
    forward = pd.read_parquet(root / "forward_summary.parquet")

    expected_selected = select_coverage_policies(source_grid)
    pd.testing.assert_frame_equal(selected, expected_selected)
    assert protocol["source_protocol_hash"] == source_protocol["protocol_hash"]
    assert protocol["source_h1_grid_sha256"] == _frame_hash(source_grid)
    assert protocol["selected_policy_sha256"] == _frame_hash(selected)
    assert protocol["q2_loaded"] is False and result["q2_loaded"] is False
    assert result["evidence_role"] == "secondary_reused_forward"
    assert pd.Timestamp(result["max_prediction_timestamp"]) < CUTOFF
    assert selected["arm"].tolist() == list(REGISTERED_ARMS)
    assert selected["models_screened"].eq(9).all()
    assert coverage_eligibility(selected).all()
    assert len(forward) == 4
    assert np.isfinite(forward[["daily_sharpe", "daily_sortino"]]).all().all()
    for path in (root / "forward_ledgers").glob("*.parquet"):
        frame = pd.read_parquet(path)
        time_column = "timestamp" if path.stem.endswith("_per_bar") else "exit_time"
        if len(frame):
            assert pd.to_datetime(frame[time_column], utc=True).max() < CUTOFF
