from __future__ import annotations

import importlib
import hashlib
import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.index_replication import _frame_hash


EXPECTED_MODELS = (
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
CODE_ROOT = Path(__file__).resolve().parents[1]
REAL_ENSEMBLE_CACHE = CODE_ROOT / "experiments" / "cache" / "index_all_model_ensemble"


def test_index_ensemble_exposes_the_exact_nine_model_contract() -> None:
    try:
        module = importlib.import_module("experiments.index_all_model_ensemble")
    except ModuleNotFoundError:
        pytest.fail("index all-model ensemble module is missing")

    assert module.MODEL_NAMES == EXPECTED_MODELS


def _prediction_frames() -> dict[str, pd.DataFrame]:
    timestamps = pd.DatetimeIndex(
        ["2024-03-01T10:00:00Z", "2024-03-01T10:15:00Z"]
    )
    return {
        model: pd.DataFrame(
            {
                "timestamp": timestamps,
                "y_true": [1, 2],
                "p_short": [0.2, 0.1],
                "p_flat": [0.5, 0.2],
                "p_long": [0.3, 0.7],
                "fit_id": [f"{model}-fold0", f"{model}-fold0"],
            }
        )
        for model in EXPECTED_MODELS
    }


def test_alignment_returns_one_exact_ordered_nine_model_panel() -> None:
    module = importlib.import_module("experiments.index_all_model_ensemble")

    panel = module.align_model_predictions(_prediction_frames())

    assert panel.timestamp.tolist() == [
        pd.Timestamp("2024-03-01T10:00:00Z"),
        pd.Timestamp("2024-03-01T10:15:00Z"),
    ]
    assert panel.y_true.tolist() == [1, 2]
    assert tuple(panel.probabilities) == EXPECTED_MODELS
    np.testing.assert_allclose(
        panel.probabilities["logreg"],
        np.array([[0.2, 0.5, 0.3], [0.1, 0.2, 0.7]]),
    )


def test_alignment_rejects_a_missing_ninth_model() -> None:
    module = importlib.import_module("experiments.index_all_model_ensemble")
    frames = _prediction_frames()
    frames.pop("gru")

    with pytest.raises(ValueError, match="exact nine-model panel"):
        module.align_model_predictions(frames)


def test_alignment_rejects_shifted_timestamps_and_changed_labels() -> None:
    module = importlib.import_module("experiments.index_all_model_ensemble")
    shifted = _prediction_frames()
    shifted["gru"] = shifted["gru"].assign(
        timestamp=pd.DatetimeIndex(
            ["2024-03-01T10:00:00Z", "2024-03-01T10:30:00Z"]
        )
    )
    with pytest.raises(ValueError, match="timestamps differ"):
        module.align_model_predictions(shifted)

    relabelled = _prediction_frames()
    relabelled["gru"] = relabelled["gru"].assign(y_true=[1, 0])
    with pytest.raises(ValueError, match="labels differ"):
        module.align_model_predictions(relabelled)


@pytest.mark.parametrize(
    "probabilities",
    (
        (np.nan, 0.5, 0.5),
        (-0.1, 0.5, 0.6),
        (0.2, 0.2, 0.2),
    ),
)
def test_alignment_rejects_invalid_probability_rows(probabilities) -> None:
    module = importlib.import_module("experiments.index_all_model_ensemble")
    frames = _prediction_frames()
    frames["gru"].loc[0, ["p_short", "p_flat", "p_long"]] = probabilities

    with pytest.raises(ValueError, match="finite normalized probabilities"):
        module.align_model_predictions(frames)


def _panel_for_votes(votes: tuple[int, ...]):
    module = importlib.import_module("experiments.index_all_model_ensemble")
    assert len(votes) == 9
    return module.AlignedPanel(
        timestamp=pd.DatetimeIndex(["2024-03-01T10:00:00Z"]),
        y_true=np.array([1]),
        probabilities={
            model: np.eye(3, dtype=float)[[vote]]
            for model, vote in zip(EXPECTED_MODELS, votes)
        },
        fit_ids={model: np.array([f"{model}-fit"]) for model in EXPECTED_MODELS},
    )


def test_soft_vote_is_the_equal_average_of_all_nine_models() -> None:
    module = importlib.import_module("experiments.index_all_model_ensemble")
    panel = _panel_for_votes((0, 1, 1, 1, 1, 1, 1, 1, 1))

    actual = module.combine_probabilities("soft_vote", panel)

    np.testing.assert_allclose(actual, np.array([[1 / 9, 8 / 9, 0.0]]))


def test_directional_majority_requires_five_of_nine_directional_votes() -> None:
    module = importlib.import_module("experiments.index_all_model_ensemble")
    four_long = _panel_for_votes((2, 2, 2, 2, 1, 1, 0, 0, 0))
    five_long = _panel_for_votes((2, 2, 2, 2, 2, 1, 0, 0, 0))
    five_short = _panel_for_votes((0, 0, 0, 0, 0, 1, 2, 2, 2))

    np.testing.assert_allclose(
        module.combine_probabilities("directional_majority", four_long),
        np.array([[0.0, 1.0, 0.0]]),
    )
    np.testing.assert_allclose(
        module.combine_probabilities("directional_majority", five_long),
        np.array([[3 / 9, 1 / 9, 5 / 9]]),
    )
    np.testing.assert_allclose(
        module.combine_probabilities("directional_majority", five_short),
        np.array([[5 / 9, 1 / 9, 3 / 9]]),
    )


def test_stack_matrix_has_eighteen_ordered_directional_features() -> None:
    module = importlib.import_module("experiments.index_all_model_ensemble")
    frames = _prediction_frames()
    for number, model in enumerate(EXPECTED_MODELS, start=1):
        frames[model]["p_short"] = number / 100.0
        frames[model]["p_long"] = number / 50.0
        frames[model]["p_flat"] = 1.0 - frames[model]["p_short"] - frames[model]["p_long"]
    panel = module.align_model_predictions(frames)

    matrix = module.stack_feature_matrix(panel)

    assert matrix.shape == (2, 18)
    np.testing.assert_allclose(matrix[0, :2], [0.01, 0.02])
    np.testing.assert_allclose(matrix[0, -2:], [0.09, 0.18])


def test_fixed_logistic_stack_is_balanced_l2_and_three_class() -> None:
    module = importlib.import_module("experiments.index_all_model_ensemble")
    rng = np.random.default_rng(42)
    X = rng.normal(size=(90, 18))
    y = np.tile(np.array([0, 1, 2]), 30)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        stack = module.fit_logistic_stack(X, y)

    logistic = stack.named_steps["logisticregression"]
    assert logistic.C == 0.1
    assert logistic.class_weight == "balanced"
    assert logistic.l1_ratio == 0.0
    assert logistic.random_state == 42
    assert logistic.classes_.tolist() == [0, 1, 2]


class _ReorderedStack:
    classes_ = np.array([2, 0, 1])

    def predict_proba(self, _matrix):
        return np.array([[0.7, 0.2, 0.1]])


class _MissingClassStack:
    classes_ = np.array([0, 2])

    def predict_proba(self, _matrix):
        return np.array([[0.4, 0.6]])


def test_stack_probabilities_are_mapped_by_class_label() -> None:
    module = importlib.import_module("experiments.index_all_model_ensemble")
    panel = _panel_for_votes((0, 1, 2, 0, 1, 2, 0, 1, 2))

    actual = module.combine_probabilities("stack", panel, stack_model=_ReorderedStack())

    np.testing.assert_allclose(actual, np.array([[0.2, 0.1, 0.7]]))
    with pytest.raises(ValueError, match="all three class labels"):
        module.combine_probabilities("stack", panel, stack_model=_MissingClassStack())


def _dated_panel(timestamps: list[str], labels: list[int]):
    module = importlib.import_module("experiments.index_all_model_ensemble")
    index = pd.DatetimeIndex(timestamps)
    rows = len(index)
    return module.AlignedPanel(
        timestamp=index,
        y_true=np.asarray(labels, dtype=int),
        probabilities={
            model: np.tile(np.array([[0.25, 0.45, 0.30]]), (rows, 1))
            for model in EXPECTED_MODELS
        },
        fit_ids={
            model: np.asarray([f"{model}-fit"] * rows)
            for model in EXPECTED_MODELS
        },
    )


class _ConstantThreeClassStack:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, matrix):
        return np.tile(np.array([[0.2, 0.5, 0.3]]), (len(matrix), 1))


def test_causal_h1_stack_uses_only_labels_available_before_each_month() -> None:
    module = importlib.import_module("experiments.index_all_model_ensemble")
    oof = _dated_panel(
        [
            "2024-10-01T10:00:00Z",
            "2024-10-01T10:15:00Z",
            "2024-10-01T10:30:00Z",
            "2024-11-01T10:00:00Z",
            "2024-11-01T10:15:00Z",
            "2024-11-01T10:30:00Z",
            "2024-12-01T10:00:00Z",
            "2024-12-01T10:15:00Z",
            "2024-12-01T10:30:00Z",
        ],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
    )
    h1 = _dated_panel(
        [
            "2025-01-15T10:00:00Z",
            "2025-01-31T23:45:00Z",
            "2025-02-15T10:00:00Z",
            "2025-02-28T10:00:00Z",
        ],
        [0, 1, 2, 0],
    )
    fit_sizes: list[int] = []

    def recording_factory(X, y):
        assert len(X) == len(y)
        fit_sizes.append(len(y))
        return _ConstantThreeClassStack()

    result = module.build_causal_h1_predictions(
        oof, h1, stack_factory=recording_factory
    )

    assert fit_sizes == [9, 10, 13]
    assert tuple(result.predictions) == (
        "soft_vote",
        "directional_majority",
        "stack",
    )
    assert all(len(frame) == 4 for frame in result.predictions.values())
    audit = result.audit.set_index("prediction_period")
    assert int(audit.loc["2025-01", "meta_train_rows"]) == 9
    assert int(audit.loc["2025-02", "meta_train_rows"]) == 10
    assert int(audit.loc["forward_freeze", "meta_train_rows"]) == 13
    assert audit.loc["2025-02", "max_label_available_at"] <= pd.Timestamp(
        "2025-02-01T00:00:00Z"
    )
    assert audit.loc["forward_freeze", "training_cutoff"] == pd.Timestamp(
        "2025-07-01T00:00:00Z"
    )


def test_causal_h1_builder_rejects_oof_or_h1_rows_outside_their_windows() -> None:
    module = importlib.import_module("experiments.index_all_model_ensemble")
    valid_oof = _dated_panel(
        ["2024-10-01T10:00:00Z", "2024-10-01T10:15:00Z", "2024-10-01T10:30:00Z"],
        [0, 1, 2],
    )
    leaked_oof = _dated_panel(
        ["2025-01-01T00:00:00Z", "2025-01-01T00:15:00Z", "2025-01-01T00:30:00Z"],
        [0, 1, 2],
    )
    valid_h1 = _dated_panel(
        ["2025-01-15T10:00:00Z"],
        [0],
    )
    leaked_h1 = _dated_panel(
        ["2025-07-01T00:00:00Z"],
        [0],
    )

    with pytest.raises(ValueError, match="OOF rows must end before H1"):
        module.build_causal_h1_predictions(leaked_oof, valid_h1)
    with pytest.raises(ValueError, match="H1 rows must remain inside H1"):
        module.build_causal_h1_predictions(valid_oof, leaked_h1)


def _h1_row(**overrides) -> dict:
    row = {
        "arm": "selected_base",
        "variant": "soft_vote",
        "width_bps": 10,
        "tau": 0.5,
        "positive_months": 4,
        "trades": 60,
        "n_long": 30,
        "n_short": 30,
        "daily_sortino": 1.0,
        "net_return": 0.02,
    }
    row.update(overrides)
    return row


def test_h1_selection_prefers_an_eligible_candidate_over_higher_ineligible_net() -> None:
    module = importlib.import_module("experiments.index_all_model_ensemble")
    grid = pd.DataFrame(
        [
            _h1_row(),
            _h1_row(
                width_bps=5,
                tau=0.35,
                positive_months=6,
                trades=100,
                n_long=100,
                n_short=0,
                daily_sortino=10.0,
                net_return=1.0,
            ),
            _h1_row(
                variant="stack",
                width_bps=15,
                daily_sortino=0.7,
                net_return=0.015,
            ),
        ]
    )

    selected = module.select_ensemble_rows(grid)
    winner = module.select_overall_ensemble(selected)

    assert len(selected) == 2
    soft = selected.loc[selected["variant"].eq("soft_vote")].iloc[0]
    assert bool(soft["eligible"])
    assert int(soft["n_short"]) == 30
    assert winner["variant"] == "soft_vote"
    assert bool(winner["eligible"])


@pytest.mark.parametrize(
    ("mutation", "failed_condition"),
    (
        ({"eligible": False}, "h1_eligible"),
        ({"net_return": 0.02}, "net_above_control"),
        ({"daily_sortino": 1.1}, "sortino_above_control"),
        ({"trades": 79}, "trade_retention_80pct"),
    ),
)
def test_promotion_requires_every_h1_quality_and_retention_condition(
    mutation, failed_condition
) -> None:
    module = importlib.import_module("experiments.index_all_model_ensemble")
    winner = {
        "eligible": True,
        "net_return": 0.03,
        "daily_sortino": 1.2,
        "trades": 80,
    }
    control = {
        "net_return": 0.02,
        "daily_sortino": 1.1,
        "trades": 100,
    }
    passing = module.promotion_decision(winner, control)
    assert passing["promoted"] is True
    assert passing["trade_retention"] == pytest.approx(0.8)

    rejected = module.promotion_decision({**winner, **mutation}, control)
    assert rejected["promoted"] is False
    assert rejected["conditions"][failed_condition] is False


def test_h1_grid_uses_the_frozen_one_bar_execution_and_finite_daily_metrics() -> None:
    module = importlib.import_module("experiments.index_all_model_ensemble")
    timestamps = pd.date_range("2025-01-02T10:00:00Z", periods=3, freq="15min")
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0],
            "high": [100.0, 101.0, 102.0],
            "low": [100.0, 100.0, 101.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [1.0, 1.0, 1.0],
            "available_at": timestamps + pd.Timedelta(minutes=15),
            "complete_bar": [True, True, True],
        },
        index=timestamps,
    )
    prediction = pd.DataFrame(
        {
            "timestamp": [timestamps[0]],
            "y_true": [2],
            "pred": [2],
            "confidence": [1.0],
            "p_short": [0.0],
            "p_flat": [0.0],
            "p_long": [1.0],
            "fit_id": ["literal-h1"],
        }
    )

    grid = module.build_h1_grid(
        bars,
        {"soft_vote": prediction, "directional_majority": prediction},
        arm="selected_base",
        width_bps=10,
        cost_bps=2.0,
    )

    assert len(grid) == 22
    assert set(grid["variant"]) == {"soft_vote", "directional_majority"}
    assert grid["trades"].eq(1).all()
    assert grid["n_long"].eq(1).all()
    assert grid["n_short"].eq(0).all()
    numeric = grid.select_dtypes(include="number").to_numpy(dtype=float)
    assert np.isfinite(numeric).all()


def _source_selected_policies() -> pd.DataFrame:
    rows = []
    for number, (arm, model) in enumerate(
        product(
            (
                "selected_base",
                "deberta_matched",
                "deepseek_matched",
                "deepseek_full",
            ),
            EXPECTED_MODELS,
        )
    ):
        rows.append(
            _h1_row(
                arm=arm,
                model_name=model,
                variant=None,
                width_bps=(5, 10, 15)[number % 3],
                tau=(0.35, 0.5, 0.65)[number % 3],
                positive_months=4 if number == 0 else 3,
                trades=80 + number,
                n_long=40,
                n_short=40 + number,
                daily_sortino=1.5 if number == 0 else 0.5,
                net_return=0.04 if number == 0 else 0.01,
                eligible=number == 0,
                h1_execution_status=(
                    "eligible" if number == 0 else "diagnostic_only_no_eligible_policy"
                ),
            )
        )
    return pd.DataFrame(rows).drop(columns="variant")


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_source_contract(root: Path, *, selected_base: str = "price_vix") -> None:
    root.mkdir(parents=True)
    protocol_body = {
        "protocol_version": "index-replication-v3-vix-majority-next-open",
        "model_names": list(EXPECTED_MODELS),
        "widths_bps": [5, 10, 15],
        "taus": [0.0, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
        "arms": [
            "selected_base",
            "deberta_matched",
            "deepseek_matched",
            "deepseek_full",
        ],
        "selection": [
            "2024-01-01T00:00:00+00:00",
            "2025-01-01T00:00:00+00:00",
        ],
        "calibration": [
            "2025-01-01T00:00:00+00:00",
            "2025-07-01T00:00:00+00:00",
        ],
        "forward": [
            "2025-07-01T00:00:00+00:00",
            "2026-04-01T00:00:00+00:00",
        ],
    }
    protocol = {**protocol_body, "protocol_hash": _canonical_hash(protocol_body)}
    (root / "protocol_manifest.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    gate = pd.DataFrame({"pair": [1], "net_delta": [0.01]})
    gate.to_parquet(root / "vix_gate_paired_2024.parquet", index=False)
    vix = {
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": protocol["protocol_hash"],
        "stream": root.name,
        "selected_base": selected_base,
        "admitted": selected_base == "price_vix",
        "gate_complete": True,
        "frozen_before_sentiment": True,
        "paired_table_sha256": _frame_hash(gate),
    }
    (root / "vix_admission.json").write_text(
        json.dumps(vix, indent=2) + "\n", encoding="utf-8"
    )
    _source_selected_policies().to_parquet(
        root / "h1_selected_policies.parquet", index=False
    )


def _ensemble_config(tmp_path: Path, *, selected_base: str = "price_vix"):
    module = importlib.import_module("experiments.index_all_model_ensemble")
    source = tmp_path / "source" / "usa500"
    output = tmp_path / "output" / "usa500"
    _write_source_contract(source, selected_base=selected_base)
    return module.IndexAllModelEnsembleConfig(
        stream="usa500",
        data_dir=tmp_path / "data",
        source_root=source,
        output_root=output,
    )


def test_source_contract_requires_the_admitted_vix_base_and_exact_36_controls(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("experiments.index_all_model_ensemble")
    config = _ensemble_config(tmp_path)

    contract = module.validate_source_contract(config)

    assert contract["selected_base"] == "price_vix"
    assert contract["control_rows"] == 36
    assert contract["q2_loaded"] is False
    assert len(contract["source_files"]) == 4

    rejected_config = _ensemble_config(
        tmp_path / "rejected", selected_base="price"
    )
    with pytest.raises(ValueError, match="VIX must be admitted"):
        module.validate_source_contract(rejected_config)


def test_source_contract_rejects_changed_vix_gate_or_control_grid(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("experiments.index_all_model_ensemble")
    config = _ensemble_config(tmp_path)
    gate_path = config.source_root / "vix_gate_paired_2024.parquet"
    pd.DataFrame({"pair": [1], "net_delta": [-1.0]}).to_parquet(
        gate_path, index=False
    )
    with pytest.raises(ValueError, match="VIX gate table changed"):
        module.validate_source_contract(config)

    config = _ensemble_config(tmp_path / "grid")
    controls = pd.read_parquet(config.source_root / "h1_selected_policies.parquet")
    controls.iloc[:-1].to_parquet(
        config.source_root / "h1_selected_policies.parquet", index=False
    )
    with pytest.raises(ValueError, match="exact 36-policy grid"):
        module.validate_source_contract(config)


def test_ensemble_config_rejects_wrong_stream_paths_and_lockbox_boundary(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("experiments.index_all_model_ensemble")

    with pytest.raises(ValueError, match="must end with the stream"):
        module.IndexAllModelEnsembleConfig(
            stream="usa500",
            data_dir=tmp_path,
            source_root=tmp_path / "usatech",
            output_root=tmp_path / "usa500",
        )
    with pytest.raises(PermissionError, match="2026-04-01"):
        module.IndexAllModelEnsembleConfig(
            stream="usa500",
            data_dir=tmp_path,
            source_root=tmp_path / "usa500",
            output_root=tmp_path / "usa500",
            end_exclusive=pd.Timestamp("2026-07-01T00:00:00Z"),
        )


def _stage_timestamps(stage: str) -> pd.DatetimeIndex:
    if stage == "oof":
        months = pd.date_range("2024-10-01", "2024-12-01", freq="MS", tz="UTC")
    elif stage == "h1":
        months = pd.date_range("2025-01-01", "2025-06-01", freq="MS", tz="UTC")
    elif stage == "forward":
        months = pd.date_range("2025-07-01", "2026-03-01", freq="MS", tz="UTC")
    else:
        raise ValueError(stage)
    return pd.DatetimeIndex(
        [
            month + pd.Timedelta(days=1, hours=10, minutes=minute)
            for month in months
            for minute in (0, 15, 30)
        ]
    )


def _stage_prediction(
    *, stage: str, arm: str, model: str, width_bps: int
) -> pd.DataFrame:
    timestamps = _stage_timestamps(stage)
    labels = np.tile(np.array([0, 1, 2]), len(timestamps) // 3)
    probabilities = np.full((len(labels), 3), 0.1, dtype=float)
    probabilities[np.arange(len(labels)), labels] = 0.8
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "y_true": labels,
            "pred": probabilities.argmax(axis=1),
            "confidence": probabilities.max(axis=1),
            "p_short": probabilities[:, 0],
            "p_flat": probabilities[:, 1],
            "p_long": probabilities[:, 2],
            "fit_id": f"fake::{stage}::{arm}::{model}::w{width_bps}",
        }
    )


def _execution_bars() -> pd.DataFrame:
    months = pd.date_range("2025-01-01", "2026-03-01", freq="MS", tz="UTC")
    timestamps = pd.DatetimeIndex(
        [
            month + pd.Timedelta(days=1, hours=10, minutes=minute)
            for month in months
            for minute in (0, 15, 30, 45)
        ]
    )
    pattern = np.tile(np.array([100.0, 100.0, 101.0, 100.0]), len(months))
    closes = np.tile(np.array([100.0, 101.0, 100.0, 101.0]), len(months))
    return pd.DataFrame(
        {
            "open": pattern,
            "high": np.maximum(pattern, closes) + 0.1,
            "low": np.minimum(pattern, closes) - 0.1,
            "close": closes,
            "volume": 1.0,
            "available_at": timestamps + pd.Timedelta(minutes=15),
            "complete_bar": True,
        },
        index=timestamps,
    )


class _FakeEnsembleSourceRunner:
    def __init__(self, config, *, selection_path: Path, calls: list[tuple]) -> None:
        self.config = config
        self.selection_path = selection_path
        self.calls = calls
        self.bars = _execution_bars()

    def _oof_arm(self, arm: str, model_name: str, width_bps: int):
        self.calls.append(("oof", arm, model_name, width_bps))
        frame = _stage_prediction(
            stage="oof", arm=arm, model=model_name, width_bps=width_bps
        )
        return {}, [frame]

    def _monthly_predictions(
        self, arm: str, model_name: str, width_bps: int
    ) -> pd.DataFrame:
        self.calls.append(("h1", arm, model_name, width_bps))
        return _stage_prediction(
            stage="h1", arm=arm, model=model_name, width_bps=width_bps
        )

    def _forward_prediction(
        self, arm: str, model_name: str, width_bps: int
    ) -> pd.DataFrame:
        assert self.selection_path.exists()
        selection = json.loads(self.selection_path.read_text(encoding="utf-8"))
        assert selection["forward_loaded"] is False
        self.calls.append(("forward", arm, model_name, width_bps))
        return _stage_prediction(
            stage="forward", arm=arm, model=model_name, width_bps=width_bps
        )


def _runner_fixture(tmp_path: Path):
    module = importlib.import_module("experiments.index_all_model_ensemble")
    config = _ensemble_config(tmp_path)
    calls: list[tuple] = []
    runner = module.IndexAllModelEnsembleRunner(
        config,
        source_runner_factory=lambda source_config: _FakeEnsembleSourceRunner(
            source_config,
            selection_path=config.output_root / "h1_selection.json",
            calls=calls,
        ),
    )
    return module, runner, config, calls


def test_runner_freezes_h1_before_forward_and_materialises_thirteen_rows(
    tmp_path: Path,
) -> None:
    module, runner, config, calls = _runner_fixture(tmp_path)

    result = runner.run()

    assert result["h1_candidate_rows"] == 12
    assert result["forward_rows"] == 13
    assert result["q2_loaded"] is False
    assert pd.Timestamp(result["max_prediction_timestamp"]) < pd.Timestamp(
        "2026-04-01T00:00:00Z"
    )
    first_forward = next(i for i, call in enumerate(calls) if call[0] == "forward")
    assert all(call[0] in {"oof", "h1"} for call in calls[:first_forward])
    selection = json.loads(
        (config.output_root / "h1_selection.json").read_text(encoding="utf-8")
    )
    assert selection["forward_loaded"] is False
    assert selection["candidate_rows"] == 12
    summary = pd.read_parquet(config.output_root / "forward_summary.parquet")
    assert len(summary) == 13
    assert summary["role"].value_counts().to_dict() == {
        "ensemble": 12,
        "best_single_control": 1,
    }
    assert summary.loc[summary["role"].eq("ensemble"), "arm"].nunique() == 4
    assert summary.loc[summary["role"].eq("ensemble"), "variant"].nunique() == 3
    assert not summary.isna().any().any()
    assert np.isfinite(summary.select_dtypes(include="number").to_numpy()).all()
    assert len(list((config.output_root / "forward").glob("*.json"))) == 13
    assert len(list((config.output_root / "forward_ledgers").glob("*.parquet"))) == 26
    protocol = json.loads(
        (config.output_root / "protocol.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (config.output_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert protocol["selected_base"] == "price_vix"
    assert protocol["q2_loaded"] is False
    assert manifest["q2_loaded"] is False
    assert manifest["protocol_hash"] == protocol["protocol_hash"]
    assert len(manifest["artifacts"]) == 50
    for relative, expected_hash in manifest["artifacts"].items():
        path = config.output_root / relative
        assert path.exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash

    resumed = module.IndexAllModelEnsembleRunner(
        config,
        source_runner_factory=lambda _config: pytest.fail(
            "complete resume must not construct the source runner"
        ),
    ).run()
    assert resumed["resumed_forward_candidates"] == 13


def test_runner_rejects_a_changed_resume_ledger(tmp_path: Path) -> None:
    _module, runner, config, _calls = _runner_fixture(tmp_path)
    runner.run()
    ledger_path = next((config.output_root / "forward_ledgers").glob("*.parquet"))
    ledger = pd.read_parquet(ledger_path)
    ledger.assign(net_return=ledger["net_return"] + 0.01).to_parquet(
        ledger_path, index=False
    )

    with pytest.raises(ValueError, match="artifact hash changed"):
        runner.run()


def test_runner_never_loads_forward_when_h1_construction_fails(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("experiments.index_all_model_ensemble")
    config = _ensemble_config(tmp_path)
    calls: list[tuple] = []

    class FailingH1Source(_FakeEnsembleSourceRunner):
        def _monthly_predictions(self, arm, model_name, width_bps):
            self.calls.append(("h1_failure", arm, model_name, width_bps))
            raise RuntimeError("literal H1 failure")

    runner = module.IndexAllModelEnsembleRunner(
        config,
        source_runner_factory=lambda source_config: FailingH1Source(
            source_config,
            selection_path=config.output_root / "h1_selection.json",
            calls=calls,
        ),
    )

    with pytest.raises(RuntimeError, match="literal H1 failure"):
        runner.run()
    assert not any(call[0] == "forward" for call in calls)
    assert not (config.output_root / "h1_selection.json").exists()


def test_manifest_tampering_fails_before_source_runner_construction(
    tmp_path: Path,
) -> None:
    module, runner, config, _calls = _runner_fixture(tmp_path)
    runner.run()
    selection_path = config.output_root / "h1_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["forward_loaded"] = True
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest artifact hash changed"):
        module.IndexAllModelEnsembleRunner(
            config,
            source_runner_factory=lambda _config: pytest.fail(
                "tampered complete run must fail before constructing source"
            ),
        ).run()


@pytest.mark.parametrize("stream", ("usa500", "usatech"))
def test_completed_real_index_ensemble_artifacts_reconcile(stream: str) -> None:
    module = importlib.import_module("experiments.index_all_model_ensemble")
    root = REAL_ENSEMBLE_CACHE / stream
    required = (
        "protocol.json",
        "h1_policy_grid.parquet",
        "h1_selected_candidates.parquet",
        "h1_selection.json",
        "forward_summary.parquet",
        "forward_monthly.parquet",
        "manifest.json",
        "result.json",
    )
    missing = [name for name in required if not (root / name).exists()]
    assert not missing, f"completed {stream} ensemble artifacts are missing: {missing}"

    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    selection = json.loads((root / "h1_selection.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    h1 = pd.read_parquet(root / "h1_selected_candidates.parquet")
    forward = pd.read_parquet(root / "forward_summary.parquet")
    monthly = pd.read_parquet(root / "forward_monthly.parquet")

    assert result["q2_loaded"] is False
    assert result["selected_base"] == "price_vix"
    assert result["models_per_ensemble"] == 9
    assert result["h1_candidate_rows"] == 12
    assert result["forward_rows"] == 13
    assert pd.Timestamp(result["max_prediction_timestamp"]) < pd.Timestamp(
        "2026-04-01T00:00:00Z"
    )
    assert protocol["q2_loaded"] is False
    assert protocol["selected_base"] == "price_vix"
    assert protocol["source_files"] == module.validate_source_contract(
        module.IndexAllModelEnsembleConfig.for_stream(stream)
    )["source_files"]
    assert selection["forward_loaded"] is False
    assert selection["q2_loaded"] is False
    assert pd.Timestamp(selection["selection_data_end_exclusive"]) == pd.Timestamp(
        "2025-07-01T00:00:00Z"
    )
    assert len(h1) == 12
    assert set(h1["arm"]) == {
        "selected_base",
        "deberta_matched",
        "deepseek_matched",
        "deepseek_full",
    }
    assert set(h1["variant"]) == {
        "soft_vote",
        "directional_majority",
        "stack",
    }
    assert len(forward) == 13 and len(monthly) == 117
    assert forward["role"].value_counts().to_dict() == {
        "ensemble": 12,
        "best_single_control": 1,
    }
    assert not forward.isna().any().any()
    assert not monthly.isna().any().any()
    assert np.isfinite(forward.select_dtypes(include="number").to_numpy()).all()
    assert np.isfinite(monthly.select_dtypes(include="number").to_numpy()).all()
    assert manifest["protocol_hash"] == protocol["protocol_hash"]
    assert manifest["q2_loaded"] is False
    assert len(manifest["artifacts"]) == 50
    for relative, expected_hash in manifest["artifacts"].items():
        artifact = root / relative
        assert artifact.exists()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected_hash
    for ledger_path in (root / "forward_ledgers").glob("*.parquet"):
        ledger = pd.read_parquet(ledger_path)
        if "timestamp" in ledger:
            assert pd.to_datetime(ledger["timestamp"], utc=True).lt(
                "2026-04-01T00:00:00Z"
            ).all()
        if "entry_time" in ledger and len(ledger):
            assert pd.to_datetime(ledger["entry_time"], utc=True).lt(
                "2026-04-01T00:00:00Z"
            ).all()
            assert pd.to_datetime(ledger["exit_time"], utc=True).le(
                "2026-04-01T00:00:00Z"
            ).all()
