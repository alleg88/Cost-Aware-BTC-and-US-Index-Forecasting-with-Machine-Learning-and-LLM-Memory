from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.run_portfolio import config_economics, select_members
from experiments.spans import CALIBRATION_END


def _frame() -> pd.DataFrame:
    """Six months either side of the config calibration split; profitable only
    above conf 0.7 (so tau=0.7 wins on the calibration half)."""
    split = pd.Timestamp(CALIBRATION_END, tz="UTC")
    idx = pd.date_range(split - pd.DateOffset(months=6),
                        split + pd.DateOffset(months=6), freq="15min", tz="UTC")
    rng = np.random.default_rng(3)
    conf = rng.uniform(0.4, 1.0, len(idx))
    pred = rng.choice([0, 2], size=len(idx))
    fwd = np.where(conf >= 0.7,
                   np.where(pred == 2, 1.0, -1.0) * 0.002,   # correct when confident
                   np.where(pred == 2, -1.0, 1.0) * 0.002)   # wrong when not
    return pd.DataFrame({"m_pred": pred, "m_conf": conf, "forward_return": fwd}, index=idx)


def test_config_economics_calibrates_on_q1_and_evaluates_after():
    summary, returns = config_economics(_frame(), "m", fee_bps=5.0)
    assert summary["tau"] >= 0.7                # calibration grid finds the profitable gate
    assert "calibration_sortino" in summary
    assert returns.index.min() >= pd.Timestamp(CALIBRATION_END, tz="UTC")
    assert summary["sortino"] > 0                      # regime persists into evaluation


def test_select_members_uses_only_the_calibration_score():
    rows = [
        {"config": "a", "calibration_sortino": 2.0, "sortino": -9.0},   # bad eval, good Q1
        {"config": "b", "calibration_sortino": -1.0, "sortino": 9.0},   # good eval, bad Q1
        {"config": "c", "calibration_sortino": 0.0, "sortino": 0.0},
    ]
    picked = select_members(rows, cal_col="calibration_sortino", min_cal_score=0.0)
    assert [r["config"] for r in picked] == ["a", "c"]   # eval column never consulted
