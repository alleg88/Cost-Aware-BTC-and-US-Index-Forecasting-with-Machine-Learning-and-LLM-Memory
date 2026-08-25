import numpy as np
import pandas as pd
import pytest

import experiments.run_event_window_opportunity_head as runner


def _scores() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2024-01-01", tz="UTC")
    for arm in runner.ALL_ARMS:
        for episode in range(80):
            for step, target in enumerate((0, 1)):
                decision = start + pd.Timedelta(days=episode, minutes=5 * step)
                baseline = arm == runner.BASELINE_ARM
                rows.append(
                    {
                        "arm": arm,
                        "model": "baseline" if baseline else "candidate",
                        "fold_id": "test",
                        "window_id": f"w{episode}_{step}",
                        "channel_episode_id": f"e{episode}",
                        "step": step,
                        "decision_time": decision,
                        "label_start": decision,
                        "label_end": decision + pd.Timedelta(minutes=120),
                        "opportunity_code": target,
                        "p_hit": 0.5 if baseline else (0.9 if target else 0.1),
                        "sample_weight": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_protocol_is_one_head_only_and_seals_later_periods():
    protocol = runner.protocol_dict()
    assert protocol["objective"] == "one binary direction-free opportunity head"
    assert protocol["output"].startswith("P(adaptive barrier")
    assert protocol["direction_head_trained"] is False
    assert protocol["trading_policy_trained"] is False
    assert protocol["economics_evaluated"] is False
    assert protocol["forward_or_lockbox_loaded"] is False
    assert "direction" not in " ".join(protocol["primary_metrics"]).lower()


def test_runner_rejects_forward_before_loading_any_frozen_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner,
        "load_frozen_m_artifacts",
        lambda *_: (_ for _ in ()).throw(AssertionError("loader must not run")),
    )
    with pytest.raises(ValueError, match="development only"):
        runner.run_opportunity_study(stage="forward", run_root=tmp_path)


def test_frozen_m_loader_accepts_only_registered_complete_dev_handoff():
    frozen = runner.load_frozen_m_artifacts()
    assert frozen.run_hash == "af5ad771ae4c000d030c"
    assert len(frozen.xgboost_scores) == 128_271
    assert frozen.protocol["forward_or_lockbox_loaded"] is False
    assert frozen.summary["forward_or_lockbox_loaded"] is False


def test_paired_episode_bootstrap_recognises_better_opportunity_probabilities():
    scores = _scores()
    metrics = runner._aggregate_metrics(scores)
    paired = runner._paired_bootstrap(scores, metrics, draws=80, seed=7)
    for arm in runner.ALL_ARMS[1:]:
        comparison = paired.loc[paired["candidate"].eq(arm)].set_index("comparison")
        assert comparison.at["pr_auc_delta", "ci_low"] > 0.0
        assert comparison.at["brier_improvement", "ci_low"] > 0.0
        assert comparison.at["logloss_improvement", "point_improvement"] > 0.0
    promoted_metrics, chosen = runner._promotion_decision(metrics, paired)
    assert chosen in runner.PROMOTABLE_ARMS
    assert not promoted_metrics.loc[
        promoted_metrics["arm"].eq("N1_binary_legacy"), "promotable"
    ].item()


def test_matched_frequency_is_equal_count_rank_diagnostic_not_trade_policy():
    scores = _scores()
    selected, summary = runner._matched_frequency(scores, decisions_per_day=1.0)
    counts = summary.set_index("arm")["matched_decisions"]
    assert counts.nunique() == 1
    assert summary["diagnostic_only"].all()
    assert selected.groupby("arm")["window_id"].nunique().eq(counts).all()
