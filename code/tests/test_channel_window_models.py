"""OOF isolation, architecture and continuation tests for Notebook B models."""

import numpy as np
import pandas as pd

from experiments.channel_window_dataset import CHANNEL_WINDOW_FEATURES
from experiments.channel_window_models import (
    OOFResult,
    score_frozen_model,
    logreg_continuation,
    run_catboost_oof,
    run_logreg_oof,
)


def _events() -> pd.DataFrame:
    block_starts = [pd.Timestamp("2021-01-10", tz="UTC")]
    block_starts.extend(
        pd.Timestamp(start, tz="UTC")
        for start in (
            "2022-01-10", "2022-07-10", "2023-01-10", "2023-07-10",
            "2024-01-10", "2024-07-10", "2025-01-10",
        )
    )
    rows = []
    episode = 0
    rng = np.random.default_rng(42)
    for block_no, start in enumerate(block_starts):
        for pair in range(24):
            episode += 1
            for within in range(2):
                row_no = pair * 2 + within
                decision = start + pd.Timedelta(days=pair, minutes=within * 5)
                side = "long" if row_no % 2 == 0 else "short"
                latent = (1.0 if pair % 2 == 0 else -1.0) + rng.normal(0, 0.15)
                values = {
                    feature: latent * (1.0 + feature_no / 100.0)
                    + rng.normal(0, 0.03)
                    for feature_no, feature in enumerate(CHANNEL_WINDOW_FEATURES)
                }
                values["positioning_stale"] = int(pair % 7 == 0)
                r_net = 0.8 if latent > 0 else -0.6
                rows.append(
                    {
                        "candidate_id": f"c-{block_no}-{row_no}",
                        "decision_time": decision,
                        "label_start": decision,
                        "label_end": decision + pd.Timedelta("20min"),
                        "active_end_time": decision + pd.Timedelta("20min"),
                        "channel_episode_id": episode,
                        "window_id": f"w-{episode}",
                        "side": side,
                        "cadence": "5min",
                        "filled": True,
                        "r_net": r_net,
                        "label_net_positive": int(r_net > 0),
                        **values,
                    }
                )
    return pd.DataFrame(rows)


def _flat_oof_result() -> OOFResult:
    predictions = pd.DataFrame(
        {
            "row_id": np.arange(20),
            "fold_id": "2025H1",
            "side": np.where(np.arange(20) % 2, "long", "short"),
            "score": np.linspace(0.0, 1.0, 20),
            "r_net": 0.0,
            "filled": True,
            "channel_episode_id": np.arange(20),
        }
    )
    return OOFResult(
        predictions=predictions,
        audit=pd.DataFrame(),
        feature_columns=CHANNEL_WINDOW_FEATURES,
        architecture="pooled",
        model_kind="logreg",
    )


def _passing_oof_result() -> OOFResult:
    score = np.linspace(0.0, 1.0, 100)
    selected = score >= np.quantile(score, 0.70)
    predictions = pd.DataFrame(
        {
            "row_id": np.arange(100),
            "fold_id": "all",
            "side": np.where(np.arange(100) % 2, "long", "short"),
            "score": score,
            "r_net": np.where(selected, 0.8, -0.2),
            "filled": True,
            "channel_episode_id": np.arange(100),
        }
    )
    return OOFResult(
        predictions=predictions,
        audit=pd.DataFrame(),
        feature_columns=CHANNEL_WINDOW_FEATURES,
        architecture="pooled",
        model_kind="logreg",
    )


def test_every_score_is_produced_without_its_episode_in_training():
    result = run_logreg_oof(_events(), architecture="pooled")

    assert result.predictions["score"].notna().any()
    assert result.audit["train_valid_episode_overlap"].eq(0).all()
    assert result.audit["train_ess"].gt(0).all()


def test_separate_architecture_fits_long_and_short_independently():
    result = run_logreg_oof(_events(), architecture="separate")

    assert set(result.audit["model_side"].dropna()) == {"long", "short"}
    assert set(result.predictions["side"]) == {"long", "short"}


def test_catboost_does_not_receive_automatic_permission_when_logreg_fails():
    assert not logreg_continuation(_flat_oof_result())
    assert logreg_continuation(_passing_oof_result())


def test_fixed_catboost_regression_produces_deterministic_oof_scores():
    events = _events()
    first = run_catboost_oof(events, architecture="pooled")
    second = run_catboost_oof(events, architecture="pooled")

    assert first.model_kind == "catboost_regressor"
    np.testing.assert_allclose(first.predictions["score"], second.predictions["score"])
    pd.testing.assert_frame_equal(first.audit, second.audit)


def test_frozen_full_dev_refit_scores_new_rows_without_changing_their_order():
    events = _events()
    train = events[events["decision_time"] < pd.Timestamp("2025-01-01", tz="UTC")]
    score = events[events["decision_time"] >= pd.Timestamp("2025-01-01", tz="UTC")]

    predictions = score_frozen_model(
        train, score, architecture="pooled", model_kind="catboost_regressor",
    )

    assert predictions["candidate_id"].tolist() == score["candidate_id"].tolist()
    assert predictions["score"].notna().all()
