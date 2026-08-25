import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.index_all_model_forward as all_model_forward
from experiments.index_replication import ARMS
from experiments.index_replication_protocol import (
    CUTOFF,
    FORWARD_END,
    FORWARD_START,
    MODEL_NAMES,
    TAUS,
    WIDTHS,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
ALL_MODEL_CACHE = CODE_ROOT / "experiments" / "cache" / "index_all_model_forward"
SOURCE_CACHE = CODE_ROOT / "experiments" / "cache" / "index_replication"


def test_all_model_forward_module_exists() -> None:
    assert (CODE_ROOT / "experiments" / "index_all_model_forward.py").is_file()


def test_selected_policy_validator_api_exists() -> None:
    assert callable(getattr(all_model_forward, "validate_selected_policies", None))


def _selected_policies() -> pd.DataFrame:
    rows = []
    for number, (arm, model_name) in enumerate(product(ARMS, MODEL_NAMES)):
        eligible = number % 5 == 0
        rows.append(
            {
                "arm": arm,
                "model_name": model_name,
                "width_bps": WIDTHS[number % len(WIDTHS)],
                "tau": TAUS[number % len(TAUS)],
                "eligible": eligible,
                "h1_execution_status": (
                    "eligible" if eligible else "diagnostic_only_no_eligible_policy"
                ),
            }
        )
    return pd.DataFrame(rows)


def test_selected_policy_validator_accepts_only_stable_exact_36_grid() -> None:
    shuffled = _selected_policies().sample(frac=1.0, random_state=17)

    actual = all_model_forward.validate_selected_policies(shuffled)

    assert len(actual) == len(ARMS) * len(MODEL_NAMES) == 36
    assert list(zip(actual["arm"], actual["model_name"])) == list(
        product(ARMS, MODEL_NAMES)
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda frame: frame.iloc[:-1].copy(),
        lambda frame: pd.concat([frame.iloc[:-1], frame.iloc[[0]]], ignore_index=True),
        lambda frame: frame.assign(
            arm=frame["arm"].mask(frame.index == 0, "unregistered_arm")
        ),
        lambda frame: frame.assign(
            model_name=frame["model_name"].mask(frame.index == 0, "unregistered_model")
        ),
        lambda frame: frame.assign(
            width_bps=frame["width_bps"].mask(frame.index == 0, 7)
        ),
        lambda frame: frame.assign(tau=frame["tau"].mask(frame.index == 0, 0.4000001)),
    ),
)
def test_selected_policy_validator_rejects_changed_grid(mutation) -> None:
    with pytest.raises(ValueError, match="exact 36-policy grid"):
        all_model_forward.validate_selected_policies(mutation(_selected_policies()))


def test_selected_policy_validator_rejects_missing_contract_column() -> None:
    with pytest.raises(ValueError, match="misses columns"):
        all_model_forward.validate_selected_policies(
            _selected_policies().drop(columns="eligible")
        )


def test_all_model_forward_runner_api_exists() -> None:
    assert callable(getattr(all_model_forward, "IndexAllModelForwardConfig", None))
    assert callable(getattr(all_model_forward, "IndexAllModelForwardRunner", None))


def _source_protocol() -> dict:
    body = {
        "protocol_version": "test-source-v1",
        "calibration": ["2025-01-01T00:00:00+00:00", FORWARD_START.isoformat()],
        "forward": [FORWARD_START.isoformat(), FORWARD_END.isoformat()],
        "q2_loaded": False,
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**body, "protocol_hash": hashlib.sha256(encoded).hexdigest()}


def _write_source_contract(root: Path) -> None:
    root.mkdir(parents=True)
    _selected_policies().to_parquet(root / "h1_selected_policies.parquet", index=False)
    (root / "protocol_manifest.json").write_text(
        json.dumps(_source_protocol(), indent=2) + "\n",
        encoding="utf-8",
    )


class _FakeSourceRunner:
    def __init__(self, config, calls: list[tuple[str, str, int]]) -> None:
        self.config = config
        self.calls = calls
        index = pd.date_range(FORWARD_START, periods=3, freq="15min")
        self.bars = pd.DataFrame(
            {
                "open": [100.0, 100.0, 101.0],
                "high": [100.0, 101.0, 102.0],
                "low": [100.0, 100.0, 101.0],
                "close": [100.0, 101.0, 102.0],
                "volume": [1.0, 1.0, 1.0],
                "available_at": index + pd.Timedelta(minutes=15),
                "complete_bar": [True, True, True],
            },
            index=index,
        )

    def _forward_prediction(
        self, arm: str, model_name: str, width_bps: int
    ) -> pd.DataFrame:
        self.calls.append((arm, model_name, width_bps))
        return pd.DataFrame(
            {
                "timestamp": [FORWARD_START],
                "pred": [1 if model_name == "logreg" else 2],
                "confidence": [1.0],
                "fit_id": [f"test::{arm}::{model_name}::{width_bps}"],
            }
        )


def _runner_fixture(tmp_path: Path):
    source_root = tmp_path / "source" / "usa500"
    output_root = tmp_path / "output" / "usa500"
    _write_source_contract(source_root)
    config = all_model_forward.IndexAllModelForwardConfig(
        stream="usa500",
        data_dir=tmp_path / "data",
        source_root=source_root,
        output_root=output_root,
    )
    calls: list[tuple[str, str, int]] = []
    runner = all_model_forward.IndexAllModelForwardRunner(
        config,
        source_runner_factory=lambda source_config: _FakeSourceRunner(
            source_config, calls
        ),
    )
    return runner, config, calls


def test_runner_materialises_all_36_real_rows_and_resumes(tmp_path: Path) -> None:
    runner, config, calls = _runner_fixture(tmp_path)

    first = runner.run()

    expected_calls = [
        (row.arm, row.model_name, int(row.width_bps))
        for row in all_model_forward.validate_selected_policies(
            _selected_policies()
        ).itertuples(index=False)
    ]
    assert calls == expected_calls
    assert first["forward_rows"] == 36
    assert first["resumed_forward_policies"] == 0
    assert first["evidence_role"] == "secondary_reused_forward_diagnostic"
    assert first["q2_loaded"] is False
    assert pd.Timestamp(first["max_prediction_timestamp"]) < CUTOFF

    summary = pd.read_parquet(config.output_root / "forward_summary.parquet")
    numeric = [
        "width_bps",
        "tau",
        "trades",
        "n_long",
        "n_short",
        "trades_per_day",
        "net_return",
        "daily_sharpe",
        "daily_sortino",
        "max_drawdown",
        "stress_2x_net_return",
        "stress_2x_daily_sharpe",
        "stress_2x_daily_sortino",
    ]
    assert len(summary) == 36
    assert summary.groupby("arm")["model_name"].nunique().eq(9).all()
    assert np.isfinite(summary[numeric].to_numpy(dtype=float)).all()
    assert set(summary["status"]) == {"traded", "no_trades"}
    assert summary.loc[summary["status"].eq("no_trades"), "trades"].eq(0).all()
    assert len(list((config.output_root / "forward").glob("*.json"))) == 36
    assert len(list((config.output_root / "forward_ledgers").glob("*.parquet"))) == 72

    resumed = all_model_forward.IndexAllModelForwardRunner(
        config,
        source_runner_factory=lambda _config: pytest.fail(
            "valid resume must not construct the source runner"
        ),
    ).run()
    assert resumed["resumed_forward_policies"] == 36


def test_runner_fails_closed_when_a_resume_ledger_changes(tmp_path: Path) -> None:
    runner, config, _calls = _runner_fixture(tmp_path)
    runner.run()
    ledger_path = next((config.output_root / "forward_ledgers").glob("*.parquet"))
    ledger = pd.read_parquet(ledger_path)
    ledger.assign(net_return=ledger["net_return"] + 0.01).to_parquet(
        ledger_path, index=False
    )

    with pytest.raises(ValueError, match="artifact hash changed"):
        runner.run()


@pytest.mark.parametrize("stream", ("usa500", "usatech"))
def test_completed_all_model_forward_artifacts_reconcile(stream: str) -> None:
    root = ALL_MODEL_CACHE / stream
    source = SOURCE_CACHE / stream
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    summary = pd.read_parquet(root / "forward_summary.parquet")

    assert result["q2_loaded"] is False
    assert result["forward_rows"] == 36
    assert result["resumed_forward_policies"] == 36
    assert result["evidence_role"] == "secondary_reused_forward_diagnostic"
    assert pd.Timestamp(result["max_prediction_timestamp"]) < CUTOFF
    assert len(summary) == 36
    assert not summary[["arm", "model_name"]].duplicated().any()
    assert summary.groupby("arm")["model_name"].nunique().eq(9).all()
    assert not summary.isna().any().any()
    assert np.isfinite(summary.select_dtypes(include="number").to_numpy()).all()

    assert protocol["q2_loaded"] is False
    assert protocol["source_selected_file_sha256"] == hashlib.sha256(
        (source / "h1_selected_policies.parquet").read_bytes()
    ).hexdigest()
    assert protocol["source_protocol_file_sha256"] == hashlib.sha256(
        (source / "protocol_manifest.json").read_bytes()
    ).hexdigest()
    assert manifest["protocol_hash"] == protocol["protocol_hash"]
    assert manifest["q2_loaded"] is False
    assert len(manifest["artifacts"]) == 112
    for relative, expected_hash in manifest["artifacts"].items():
        path = root / relative
        assert path.exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash

    checkpoints = list((root / "forward").glob("*.json"))
    ledgers = list((root / "forward_ledgers").glob("*.parquet"))
    assert len(checkpoints) == 36
    assert len(ledgers) == 72
    for path in ledgers:
        frame = pd.read_parquet(path)
        if "timestamp" in frame:
            assert pd.to_datetime(frame["timestamp"], utc=True).lt(CUTOFF).all()
        if "entry_time" in frame and len(frame):
            assert pd.to_datetime(frame["entry_time"], utc=True).lt(CUTOFF).all()
            assert pd.to_datetime(frame["exit_time"], utc=True).le(CUTOFF).all()
