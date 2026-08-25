from __future__ import annotations

import pytest

from experiments.run_event_window_cost_aware_entry import (
    CostAwareStudyConfig,
    _episode_bootstrap,
    _paired_bootstrap,
    protocol_dict,
    run_cost_aware_entry_study,
)
import numpy as np
import pandas as pd


def test_protocol_registers_one_fee_schedule_and_fresh_oof():
    protocol = protocol_dict(CostAwareStudyConfig(), smoke=False)
    assert protocol["models"] == ["logreg", "xgboost"]
    assert protocol["actions"] == ["ENTER", "WAIT", "SKIP"]
    assert protocol["fee_sensitivity_grid"] is False
    assert protocol["old_oof_selected_trades_reused"] is False
    assert protocol["execution"]["maker_entry_bps"] == 2.0
    assert protocol["execution"]["taker_sl_exit_bps"] == 5.0


def test_runner_rejects_forward_before_input_discovery():
    with pytest.raises(PermissionError, match="sealed"):
        run_cost_aware_entry_study(stage="forward")


def test_bootstrap_uses_complete_zero_filled_episode_universe():
    selected = pd.DataFrame(
        {
            "model": ["logreg", "xgboost"],
            "channel_episode_id": ["e1", "e1"],
            "outcome": ["tp", "tp"],
            "realized_net_r": [1.0, 2.0],
        }
    )
    universe = np.array(["e1", "e2", "e3"], dtype=object)
    draws, _, _ = _episode_bootstrap(
        selected.loc[selected.model.eq("logreg")],
        episode_universe=universe,
        draws=100,
        seed=42,
    )
    assert (draws.total_net_r == 0.0).any()
    delta, _, _ = _paired_bootstrap(
        selected,
        episode_universe=universe,
        draws=100,
        seed=42,
    )
    assert delta == 1.0
