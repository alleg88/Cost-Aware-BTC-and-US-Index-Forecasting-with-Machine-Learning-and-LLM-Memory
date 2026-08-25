"""Load and validate the frozen nine-model one-minute comparison."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.matched_model_zoo_1m import (
    NEW_MODELS,
    WIDTHS,
    validate_model_artifacts,
)

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = CODE_ROOT / "experiments" / "cache" / "tuning" / "matched_model_zoo_1m"
DEFAULT_CATBOOST_ROOT = (
    CODE_ROOT
    / "experiments"
    / "cache"
    / "tuning"
    / "catboost_execution_resolution"
    / "one_minute"
)

MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm_linear": "Linear SVM",
    "xgboost_balanced": "XGBoost",
    "catboost_balanced": "CatBoost",
    "mlp": "MLP",
    "lstm": "LSTM",
    "gru": "GRU",
}


def _named(frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "model_name", model_name)
    out.insert(1, "model", MODEL_LABELS[model_name])
    return out


def _validate_three_widths(frame: pd.DataFrame, *, label: str) -> None:
    if len(frame) != 3 or set(frame["width_bps"].astype(int)) != set(WIDTHS):
        raise ValueError(f"{label} must contain exactly DZ55, DZ65, and DZ75")


def load_comparison(
    root: Path = DEFAULT_ROOT,
    catboost_root: Path = DEFAULT_CATBOOST_ROOT,
) -> pd.DataFrame:
    """Return the validated 27-row frozen-forward comparison."""
    root, catboost_root = Path(root), Path(catboost_root)
    frames = []
    for model_name in NEW_MODELS:
        validate_model_artifacts(root / model_name, model_name=model_name)
        frame = pd.read_parquet(root / model_name / "forward_summary.parquet")
        _validate_three_widths(frame, label=model_name)
        frames.append(_named(frame, model_name))
    catboost = pd.read_parquet(catboost_root / "forward_summary.parquet")
    _validate_three_widths(catboost, label="catboost_balanced")
    frames.append(_named(catboost, "catboost_balanced"))
    comparison = pd.concat(frames, ignore_index=True)
    if comparison.duplicated(["model_name", "width_bps"]).any():
        raise ValueError("duplicate model/dead-zone rows in comparison")
    if not comparison["resolution"].eq("1m").all():
        raise ValueError("comparison must use one-minute execution only")
    if pd.to_datetime(comparison["period_end"], utc=True).gt("2026-04-01").any():
        raise ValueError("sealed 2026 Q2 boundary reached by comparison")
    return comparison.sort_values(["model_name", "width_bps"]).reset_index(drop=True)


def best_by_metric(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return separately named winners for the three primary economic metrics."""
    rows = []
    for criterion, metric in (
        ("Highest Sortino", "sortino"),
        ("Highest Sharpe", "sharpe"),
        ("Highest net return", "net_return"),
    ):
        row = comparison.loc[comparison[metric].idxmax()].to_dict()
        row = {"criterion": criterion, **row}
        rows.append(row)
    return pd.DataFrame(rows)


def load_selection_tables(
    root: Path = DEFAULT_ROOT,
    catboost_root: Path = DEFAULT_CATBOOST_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return 27 selected 2024 candidates and 27 H1 frozen policies."""
    root, catboost_root = Path(root), Path(catboost_root)
    candidates, policies = [], []
    for model_name in NEW_MODELS:
        candidate = pd.read_parquet(root / model_name / "selected_candidates_2024.parquet")
        policy = pd.read_parquet(root / model_name / "selected_policies_2025h1.parquet")
        _validate_three_widths(candidate, label=f"{model_name} candidates")
        _validate_three_widths(policy, label=f"{model_name} policies")
        candidates.append(_named(candidate, model_name))
        policies.append(_named(policy, model_name))
    cat_candidate = pd.read_parquet(catboost_root / "selected_candidates_2024.parquet")
    cat_policy = pd.read_parquet(catboost_root / "selected_policies_2025h1.parquet")
    _validate_three_widths(cat_candidate, label="CatBoost candidates")
    _validate_three_widths(cat_policy, label="CatBoost policies")
    candidates.append(_named(cat_candidate, "catboost_balanced"))
    policies.append(_named(cat_policy, "catboost_balanced"))
    return (
        pd.concat(candidates, ignore_index=True),
        pd.concat(policies, ignore_index=True),
    )
