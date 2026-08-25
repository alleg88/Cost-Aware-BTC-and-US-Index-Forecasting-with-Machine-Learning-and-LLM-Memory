from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.unified_2021_ensemble_models import MODEL_NAMES


def _row(
    key: str,
    timestamp: pd.Timestamp,
    *,
    xgb=(0.0, 0.0),
    lstm=(0.0, 0.0),
    svm=(0.0, 0.0),
    fold_id: int = 0,
) -> dict[str, object]:
    record: dict[str, object] = {
        "row_key": key,
        "decision_time": timestamp,
        "fold_id": fold_id,
    }
    for model, values in zip(MODEL_NAMES, (xgb, lstm, svm)):
        record[f"pred_long_{model}"] = values[0]
        record[f"pred_short_{model}"] = values[1]
    return record


def _prediction_fixture() -> pd.DataFrame:
    time = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")
    return pd.DataFrame(
        [
            _row(
                "two-long",
                time[0],
                xgb=(4.0, -2.0),
                lstm=(6.0, -3.0),
                svm=(-1.0, 3.0),
            ),
            _row(
                "split",
                time[1],
                xgb=(4.0, -1.0),
                lstm=(-2.0, 5.0),
                svm=(-1.0, -2.0),
            ),
            _row(
                "nonpositive",
                time[2],
                xgb=(-1.0, -2.0),
                lstm=(-3.0, -1.0),
                svm=(0.0, 0.0),
            ),
            _row(
                "xgb-only",
                time[3],
                xgb=(12.0, -8.0),
                lstm=(-1.0, -2.0),
                svm=(-2.0, -1.0),
            ),
        ]
    )


def test_fixed_route_requires_two_positive_same_side_model_votes():
    from experiments.unified_expected_net_policy import (
        score_fixed_expected_net_routes,
    )

    scored = score_fixed_expected_net_routes(_prediction_fixture()).set_index(
        "row_key"
    )

    assert scored.loc["two-long", "candidate_side"] == "long"
    assert scored.loc["two-long", "agreeing_votes"] == 2
    assert scored.loc["two-long", "predicted_net_bps"] == 5.0
    assert scored.loc["split", "candidate_side"] == "wait"
    assert scored.loc["nonpositive", "candidate_side"] == "wait"
    assert scored.loc["xgb-only", "candidate_side"] == "wait"


def test_xgboost_is_never_a_solo_route_even_at_large_expected_net():
    from experiments.unified_expected_net_policy import (
        score_fixed_expected_net_routes,
    )

    scored = score_fixed_expected_net_routes(_prediction_fixture())

    assert not scored["route"].eq("xgboost_solo").any()
    assert not scored["xgboost_solo"].any()


def _economic_paths(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in predictions.to_dict(orient="records"):
        decision_time = pd.Timestamp(record["decision_time"])
        for side in ("long", "short"):
            rows.append(
                {
                    "row_key": record["row_key"],
                    "decision_time": decision_time,
                    "direction": side,
                    "path_complete": True,
                    "entry_time": decision_time + pd.Timedelta(minutes=1),
                    "actual_exit_time": decision_time + pd.Timedelta(minutes=5),
                    "entry_price": 100.0,
                    "exit_price": 101.0 if side == "long" else 99.0,
                    "gross_return": 0.01,
                    "net_return": 0.009,
                    "entry_cost_bps": 5.0,
                    "exit_cost_bps": 5.0,
                }
            )
    return pd.DataFrame(rows)


def test_policy_uses_cold_crossing_rearm_and_global_sixty_minute_refractory():
    from experiments.unified_expected_net_policy import (
        apply_fixed_expected_net_policy,
    )

    time = pd.date_range("2024-01-01", periods=9, freq="15min", tz="UTC")
    rows = []
    for index, timestamp in enumerate(time):
        active = index in {1, 2, 4, 8}
        rows.append(
            _row(
                f"row-{index}",
                timestamp,
                xgb=(5.0, -2.0) if active else (-1.0, -2.0),
                lstm=(6.0, -2.0) if active else (-2.0, -1.0),
                svm=(-1.0, 2.0),
            )
        )
    predictions = pd.DataFrame(rows)

    ledger = apply_fixed_expected_net_policy(
        predictions, _economic_paths(predictions)
    )

    assert ledger["row_key"].tolist() == ["row-1", "row-8"]
    assert ledger["selected_side"].tolist() == ["long", "long"]
    assert (pd.to_datetime(ledger["decision_time"], utc=True).diff().dropna() >= pd.Timedelta(minutes=60)).all()
    assert not ledger["route"].eq("xgboost_solo").any()


def _qualifying_ledger() -> pd.DataFrame:
    rows = []
    for fold_id in range(5):
        for side in ("long", "short"):
            for trade in range(15):
                rows.append(
                    {
                        "row_key": f"{fold_id}-{side}-{trade}",
                        "fold_id": fold_id,
                        "direction": side,
                        "selected_side": side,
                        "route": "unanimous",
                        "net_return": 0.001,
                    }
                )
    return pd.DataFrame(rows)


def _clean_audit() -> dict[str, object]:
    return {
        "leakage_clean": True,
        "reconciliation_clean": True,
        "path_contract_clean": True,
        "cost_contract_clean": True,
    }


def test_development_gate_requires_frequency_dynamic_side_floor_and_fold_economics():
    from experiments.unified_expected_net_policy import (
        evaluate_expected_net_development,
    )

    result = evaluate_expected_net_development(_qualifying_ledger(), _clean_audit())

    assert result["trades"] == 150
    assert result["required_side_trades"] == 30
    assert result["long_trades"] == 75
    assert result["short_trades"] == 75
    assert result["total_positive_folds"] == 5
    assert result["long_positive_folds"] == 5
    assert result["short_positive_folds"] == 5
    assert result["xgboost_solo_trades"] == 0
    assert result["development_pass"]


def test_development_gate_rejects_side_imbalance_and_dirty_reconciliation():
    from experiments.unified_expected_net_policy import (
        evaluate_expected_net_development,
    )

    ledger = _qualifying_ledger()
    ledger.loc[ledger.index[:115], ["direction", "selected_side"]] = "long"
    audit = _clean_audit()
    audit["reconciliation_clean"] = False

    result = evaluate_expected_net_development(ledger, audit)

    assert result["required_side_trades"] == 30
    assert result["short_trades"] < result["required_side_trades"]
    assert not result["reconciliation_clean"]
    assert not result["development_pass"]
