from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.index_comparison import (
    calendar_arm_hypotheses,
    common_timestamp_summary,
    transferability_table,
)


def _write_forward(
    root: Path,
    stream: str,
    arm: str,
    model: str,
    values: list[float],
    *,
    start: str = "2025-07-01 00:00",
):
    timestamps = pd.date_range(start, periods=len(values), freq="15min", tz="UTC")
    ledger = pd.DataFrame(
        {
            "entry_bar_open": timestamps,
            "side": [1, -1][: len(values)],
            "net_return": values,
        }
    )
    per_bar = pd.DataFrame({"timestamp": timestamps, "net_return": values})
    path = root / stream / "forward_ledgers" / arm
    path.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(path / f"{model}.parquet", index=False)
    per_bar.to_parquet(path / f"{model}_per_bar.parquet", index=False)


def test_calendar_hypotheses_keep_arm_only_and_base_only_returns_with_zero_fill(
    tmp_path: Path,
):
    _write_forward(tmp_path, "usa500", "selected_base", "logreg", [0.01])
    _write_forward(
        tmp_path,
        "usa500",
        "deberta_matched",
        "logreg",
        [0.02],
        start="2025-07-01 00:15",
    )

    summary, models = calendar_arm_hypotheses(
        tmp_path / "usa500", arms=("deberta_matched",), models=("logreg",)
    )

    assert models.loc[0, "base_net_return"] == 0.01
    assert models.loc[0, "arm_net_return"] == 0.02
    assert models.loc[0, "net_delta"] == 0.01
    assert summary.loc[0, "model_median_net_delta"] == 0.01
    assert summary.loc[0, "calendar_missing_fill"] == "zero"
    assert summary.loc[0, "aggregation"] == "daily_median_across_eligible_models"
    assert 0 <= summary.loc[0, "holm_p_value"] <= 1


def test_calendar_hypotheses_mark_zero_jointly_eligible_pairs_not_estimable(
    tmp_path: Path,
):
    root = tmp_path / "usa500"
    _write_forward(tmp_path, "usa500", "selected_base", "logreg", [0.01])
    _write_forward(tmp_path, "usa500", "deberta_matched", "logreg", [0.02])
    _write_forward(tmp_path, "usa500", "deepseek_matched", "logreg", [0.03])
    pd.DataFrame(
        [
            {"arm": "selected_base", "model_name": "logreg", "h1_eligible": True},
            {"arm": "deberta_matched", "model_name": "logreg", "h1_eligible": True},
            {"arm": "deepseek_matched", "model_name": "logreg", "h1_eligible": False},
        ]
    ).to_parquet(root / "forward_summary.parquet", index=False)

    summary, models = calendar_arm_hypotheses(
        root,
        arms=("deberta_matched", "deepseek_matched"),
        models=("logreg",),
    )
    estimated = summary.set_index("arm").loc["deberta_matched"]
    unavailable = summary.set_index("arm").loc["deepseek_matched"]

    assert bool(estimated["estimable"])
    assert pd.notna(estimated["raw_p_value"])
    assert pd.notna(estimated["holm_p_value"])
    assert not bool(unavailable["estimable"])
    assert unavailable["estimability_reason"] == "no_jointly_h1_eligible_same_model_pair"
    for column in (
        "hac_mean_daily_delta",
        "hac_standard_error",
        "hac_ci_low",
        "hac_ci_high",
        "raw_p_value",
        "holm_p_value",
        "holm_significant_5pct",
    ):
        assert pd.isna(unavailable[column])
    assert set(models["arm"]) == {"deberta_matched"}


def test_common_timestamp_summary_excludes_market_specific_rows(tmp_path: Path):
    _write_forward(tmp_path, "usa500", "selected_base", "logreg", [0.01, 0.02])
    _write_forward(tmp_path, "usatech", "selected_base", "logreg", [0.03])

    result = common_timestamp_summary(
        {"usa500": tmp_path / "usa500", "usatech": tmp_path / "usatech"},
        arms=("selected_base",),
        models=("logreg",),
    ).set_index("stream")

    assert result.loc["usa500", "full_net_return"] == 0.03
    assert result.loc["usa500", "common_net_return"] == 0.01
    assert result.loc["usatech", "common_timestamp_rows"] == 1


def test_transferability_requires_both_markets_nonnegative_and_trade_retention():
    tests = pd.DataFrame(
        {
            "stream": ["usa500", "usatech"],
            "arm": ["deberta_matched", "deberta_matched"],
            "model_median_net_delta": [0.01, -0.001],
            "median_model_trade_ratio": [1.0, 0.9],
            "holm_p_value": [0.04, 0.20],
            "estimable": [True, True],
        }
    )

    result = transferability_table(tests)

    assert not bool(result.loc[0, "descriptive_transfer_both"])
    assert result.loc[0, "claim_scope"] == "descriptive_not_noninferiority"


def test_transferability_is_unavailable_when_paired_effect_is_not_estimable():
    tests = pd.DataFrame(
        {
            "stream": ["usa500", "usatech"],
            "arm": ["deepseek_full", "deepseek_full"],
            "estimable": [False, False],
            "model_median_net_delta": [float("nan"), float("nan")],
            "median_model_trade_ratio": [float("nan"), float("nan")],
            "holm_p_value": [float("nan"), float("nan")],
        }
    )

    result = transferability_table(tests)

    assert not bool(result.loc[0, "estimable_both"])
    assert result.loc[0, "assessment_status"] == "not_estimable"
    assert result.loc[0, "estimability_reason"] == "no_jointly_h1_eligible_same_model_pair"
    assert pd.isna(result.loc[0, "descriptive_transfer_both"])
