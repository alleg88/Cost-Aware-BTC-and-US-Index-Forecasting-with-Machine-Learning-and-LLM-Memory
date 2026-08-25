from __future__ import annotations

from copy import deepcopy
import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from experiments.index_all_model_ensemble import AlignedPanel


CODE_ROOT = Path(__file__).parents[1]
DIRECT_CONFIG = CODE_ROOT / "configs" / "usatech_reflection_weight_agent_v1.yaml"
REVERSAL_CONFIG = CODE_ROOT / "configs" / "usatech_budgeted_reversal_agent_v1.yaml"
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


def test_usatech_modules_are_registered():
    assert importlib.util.find_spec("reflection_agent.usatech_v1.config") is not None
    assert importlib.util.find_spec("reflection_agent.usatech_v1.engine") is not None
    assert importlib.util.find_spec("reflection_agent.usatech_v2.config") is not None


def _config_loaders():
    direct = importlib.import_module("reflection_agent.usatech_v1.config")
    reversal = importlib.import_module("reflection_agent.usatech_v2.config")
    return direct.load_usatech_agent_config, reversal.load_usatech_reversal_config


def test_usatech_configs_freeze_parent_cost_runtime_and_q2():
    load_direct, load_reversal = _config_loaders()
    direct = load_direct(DIRECT_CONFIG)
    reversal = load_reversal(REVERSAL_CONFIG)

    assert direct.stream_name == reversal.stream_name == "usatech"
    assert direct.source_candidate_id == reversal.source_candidate_id == "deepseek_full__soft_vote"
    assert direct.source_arm == reversal.source_arm == "deepseek_full"
    assert direct.source_variant == reversal.source_variant == "soft_vote"
    assert direct.source_width_bps == reversal.source_width_bps == 10
    assert direct.source_tau == reversal.source_tau == 0.55
    assert direct.round_trip_cost_bps == reversal.round_trip_cost_bps == 3.0
    assert direct.model_names == reversal.model_names == MODEL_NAMES
    assert direct.direct_batch_size == reversal.score_batch_size == 10
    assert direct.forward_end_utc == reversal.forward_end_utc == direct.q2_start_utc
    assert direct.model == reversal.model == "deepseek-v4-flash:0731-cloud"
    assert direct.required_model_digest == reversal.required_model_digest
    assert len(direct.required_model_digest) == 64


@pytest.mark.parametrize(
    ("path", "loader_index", "field", "value"),
    [
        (DIRECT_CONFIG, 0, "stream_name", "usa500"),
        (DIRECT_CONFIG, 0, "round_trip_cost_bps", 2.0),
        (DIRECT_CONFIG, 0, "source_tau", 0.50),
        (REVERSAL_CONFIG, 1, "target_change_rates", [0.10, 0.20, 0.25]),
        (REVERSAL_CONFIG, 1, "q2_start_utc", "2026-04-02T00:00:00Z"),
    ],
)
def test_usatech_configs_reject_identity_or_protocol_drift(
    tmp_path: Path,
    path: Path,
    loader_index: int,
    field: str,
    value,
):
    loaders = _config_loaders()
    payload = deepcopy(yaml.safe_load(path.read_text(encoding="utf-8")))
    payload[field] = value
    drift = tmp_path / "drift.yaml"
    drift.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises((ValidationError, ValueError)):
        loaders[loader_index](drift)


def _soft_vote_panel(start: str = "2025-07-01T00:00:00Z") -> AlignedPanel:
    timestamp = pd.date_range(start, periods=3, freq="15min", tz="UTC")
    probabilities: dict[str, np.ndarray] = {}
    for model_index, model in enumerate(MODEL_NAMES):
        if model_index < 4:
            first = [0.05, 0.05, 0.90]
            second = [0.90, 0.05, 0.05]
        else:
            first = [0.51, 0.00, 0.49]
            second = [0.49, 0.00, 0.51]
        probabilities[model] = np.asarray(
            [first, second, [0.20, 0.60, 0.20]], dtype=float
        )
    return AlignedPanel(
        timestamp=timestamp,
        y_true=np.asarray([2, 0, 1], dtype=int),
        probabilities=probabilities,
        fit_ids={model: np.asarray([f"{model}-fit"] * 3) for model in MODEL_NAMES},
    )


def _bars(start: str = "2025-07-01T00:00:00Z") -> pd.DataFrame:
    index = pd.date_range(start, periods=5, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 100.0, 102.0, 101.0],
            "high": [101.0, 102.0, 102.0, 103.0, 102.0],
            "low": [99.0, 99.5, 99.0, 100.0, 100.0],
            "close": [100.5, 100.0, 102.0, 101.0, 101.5],
            "volume": [10.0] * 5,
            "available_at": index + pd.Timedelta(minutes=15),
            "complete_bar": [True] * 5,
        },
        index=index,
    )


def _state(start: str = "2025-07-01T00:00:00Z") -> pd.DataFrame:
    index = pd.date_range(start, periods=5, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "available_at": index + pd.Timedelta(minutes=15),
            "state_vix_regime": np.linspace(-0.2, 0.2, 5),
            "state_trailing_vol": np.linspace(0.01, 0.02, 5),
            "state_trailing_trend": np.linspace(-0.01, 0.01, 5),
        },
        index=index,
    )


def test_soft_vote_builder_uses_probability_average_not_five_of_nine():
    engine = importlib.import_module("reflection_agent.usatech_v1.engine")
    opportunities, ledger = engine.build_soft_vote_opportunities(
        _soft_vote_panel(),
        _bars(),
        start="2025-07-01T00:00:00Z",
        end="2025-07-01T01:00:00Z",
        tau=0.55,
        cost_bps=3.0,
        state_frame=_state(),
    )

    assert opportunities["original_side"].tolist() == [1, -1]
    assert opportunities["opportunity_id"].tolist() == ["FWD-000000", "FWD-000001"]
    assert ledger["side"].tolist() == [1, -1]
    assert ledger["cost_return"].eq(0.0003).all()
    probability_columns = [
        column
        for column in opportunities
        if column.startswith("m") and "_p_" in column
    ]
    assert len(probability_columns) == 27
    assert opportunities[probability_columns].notna().all().all()


def test_soft_vote_builder_rejects_q2_or_future_state():
    engine = importlib.import_module("reflection_agent.usatech_v1.engine")
    with pytest.raises(PermissionError, match="Q2"):
        engine.build_soft_vote_opportunities(
            _soft_vote_panel("2026-04-01T00:00:00Z"),
            _bars("2026-04-01T00:00:00Z"),
            start="2026-04-01T00:00:00Z",
            end="2026-04-01T01:00:00Z",
            tau=0.55,
            cost_bps=3.0,
            state_frame=_state("2026-04-01T00:00:00Z"),
        )

    state = _state()
    state["available_at"] = state.index + pd.Timedelta(hours=2)
    with pytest.raises(ValueError, match="available"):
        engine.build_soft_vote_opportunities(
            _soft_vote_panel(),
            _bars(),
            start="2025-07-01T00:00:00Z",
            end="2025-07-01T01:00:00Z",
            tau=0.55,
            cost_bps=3.0,
            state_frame=state,
        )
