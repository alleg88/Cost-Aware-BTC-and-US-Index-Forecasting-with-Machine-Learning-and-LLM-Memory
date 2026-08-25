"""Model registry, baselines, and fold runner.

The registry covers the RQ1 model-zoo families (classical / boosting / deep) plus the
dummy baselines. Every model handles class imbalance inside its own training fold
(class weights or balanced sample weights); test folds keep the observed class mix.
Deep models (models/deep.py) train on GPU automatically when CUDA is available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from evaluation.metrics import classification_scores

DEFAULT_CATBOOST = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.1,
    "random_seed": 42,
    "auto_class_weights": "Balanced",
}
DEFAULT_CATBOOST_REGRESSOR = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.1,
    "random_seed": 42,
}
# Depth/child-weight/subsampling chosen on the blocking CV: XGBoost's stock defaults
# (depth 6, min_child_weight 1) overfit M15 noise and collapse predictions to the flat
# prior (macro-F1 0.402, 86% flat predictions); these settings restore directional
# recall (0.419, balanced accuracy 0.401 -> 0.450). CatBoost needs no such correction —
# ordered boosting + symmetric trees regularize by construction.
DEFAULT_XGBOOST = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.1,
                   "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8,
                   "reg_lambda": 10, "random_state": 42, "tree_method": "hist",
                   "n_jobs": -1}


def make_catboost(params: dict | None = None) -> CatBoostClassifier:
    """Return the class-balanced CatBoost classifier."""
    p = {**DEFAULT_CATBOOST, **(params or {})}
    return CatBoostClassifier(
        loss_function="MultiClass",
        verbose=0,
        **p,
    )


def make_catboost_regressor(
    params: dict | None = None,
) -> CatBoostRegressor:
    """Return a deterministic CatBoost continuous-target regressor."""
    p = {**DEFAULT_CATBOOST_REGRESSOR, **(params or {})}
    return CatBoostRegressor(loss_function="RMSE", verbose=0, **p)


class XGBBalanced:
    """XGBoost with per-fit balanced sample weights (XGB has no class_weight arg)."""

    def __init__(self, params: dict | None = None):
        self.model = XGBClassifier(
            objective="multi:softprob", num_class=3,
            **{**DEFAULT_XGBOOST, **(params or {})},
        )

    def fit(self, X, y, sample_weight=None):
        weight = compute_sample_weight("balanced", y)
        if sample_weight is not None:
            weight = weight * np.asarray(sample_weight, dtype=float)
        self.model.fit(X, y, sample_weight=weight)
        self.classes_ = self.model.classes_
        return self
    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class LogisticRegressionBalanced:
    """Scaled balanced logistic regression with a uniform weighted-fit API."""

    def __init__(self, params: dict | None = None):
        p = {
            "max_iter": 2000,
            "class_weight": "balanced",
            "C": 1.0,
            **(params or {}),
        }
        self.model = make_pipeline(StandardScaler(), LogisticRegression(**p))

    def fit(self, X, y, sample_weight=None):
        fit_params = {}
        if sample_weight is not None:
            fit_params["logisticregression__sample_weight"] = sample_weight
        self.model.fit(X, y, **fit_params)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def make_logreg(params: dict | None = None) -> LogisticRegressionBalanced:
    return LogisticRegressionBalanced(params)

def make_decision_tree(params: dict | None = None):
    p = {"max_depth": 8, "min_samples_leaf": 200, "class_weight": "balanced",
         "random_state": 42, **(params or {})}
    return DecisionTreeClassifier(**p)


def make_random_forest(params: dict | None = None):
    p = {"n_estimators": 300, "min_samples_leaf": 50,
         "class_weight": "balanced_subsample", "random_state": 42, "n_jobs": -1,
         **(params or {})}
    return RandomForestClassifier(**p)


class SVMLinear:
    """Class-weighted linear SVM with weighted probability calibration."""

    def __init__(self, params: dict | None = None):
        p = {
            "C": 0.1,
            "class_weight": "balanced",
            "max_iter": 50_000,
            "dual": False,
            **(params or {}),
        }
        self._scaler = StandardScaler()
        self._svm = LinearSVC(**p)
        self._calibrated = CalibratedClassifierCV(LinearSVC(**p), cv=3)

    def fit(self, X, y, sample_weight=None):
        scaled = self._scaler.fit_transform(X)
        self._svm.fit(scaled, y, sample_weight=sample_weight)
        self._calibrated.fit(scaled, y, sample_weight=sample_weight)
        self.classes_ = self._svm.classes_
        return self

    def predict(self, X):
        return self._svm.predict(self._scaler.transform(X))

    def predict_proba(self, X):
        return self._calibrated.predict_proba(self._scaler.transform(X))

def make_svm(params: dict | None = None) -> SVMLinear:
    return SVMLinear(params)


def make_xgboost(params: dict | None = None) -> XGBBalanced:
    return XGBBalanced(params)


def make_mlp(params: dict | None = None):
    from models.deep import TorchMLPClassifier

    return TorchMLPClassifier(**(params or {}))


def make_lstm(params: dict | None = None):
    from models.deep import TorchSequenceClassifier

    return TorchSequenceClassifier(cell="lstm", **(params or {}))


def make_gru(params: dict | None = None):
    from models.deep import TorchSequenceClassifier

    return TorchSequenceClassifier(cell="gru", **(params or {}))


def make_stack_lstm_catboost(params: dict | None = None):
    """Reference stacking pair: LSTM + CatBoost bases, CatBoost meta-learner."""
    from ensemble.stacking import StackingEnsemble

    return StackingEnsemble(
        base_factories={"catboost_balanced": make_catboost, "lstm": make_lstm},
        base_params=params,
    )


def make_stack_all(params: dict | None = None):
    """Stack of one strong model per family (classical / boosting / deep)."""
    from ensemble.stacking import StackingEnsemble

    return StackingEnsemble(
        base_factories={
            "logreg": make_logreg,
            "random_forest": make_random_forest,
            "xgboost_balanced": make_xgboost,
            "catboost_balanced": make_catboost,
            "lstm": make_lstm,
        },
        base_params=params,
    )


# Baselines keep the observed class distribution.
MODELS = {
    "always_flat": lambda params=None: DummyClassifier(strategy="most_frequent"),
    "stratified_random": lambda params=None: DummyClassifier(
        strategy="stratified", random_state=42
    ),
    "logreg": make_logreg,
    "decision_tree": make_decision_tree,
    "random_forest": make_random_forest,
    "svm_linear": make_svm,
    "xgboost_balanced": make_xgboost,
    "catboost_balanced": make_catboost,
    "mlp": make_mlp,
    "lstm": make_lstm,
    "gru": make_gru,
    "stack_lstm_catboost": make_stack_lstm_catboost,
    "stack_all": make_stack_all,
}


def _aligned_proba(model, X, classes=(0, 1, 2)) -> np.ndarray:
    """3-class probabilities aligned to (0,1,2), row-normalised."""
    raw = np.asarray(model.predict_proba(X), dtype=float)
    src = list(getattr(model, "classes_", classes))
    out = np.zeros((len(X), len(classes)), dtype=float)
    for j, c in enumerate(src):
        if int(c) in classes:
            out[:, classes.index(int(c))] = raw[:, j]
    row = out.sum(axis=1, keepdims=True)
    return np.divide(out, row, out=np.full_like(out, 1 / len(classes)), where=row > 0)


def run_cv(model_factory, X: pd.DataFrame, y: pd.Series, splitter, params: dict | None = None):
    """Fit on each train fold, predict the held-out test fold, collect metrics.

    Returns a dict with per-fold macro-F1 / balanced-accuracy, their mean/std, the
    aggregate confusion matrix, and (where the model exposes predict_proba) a
    class-weighted multinomial log-loss — a proper scoring rule that penalises
    over-confident wrong predictions, for probability-quality tuning objectives.
    """
    from sklearn.metrics import log_loss

    fold_f1, fold_bal_acc, fold_logloss, confusions = [], [], [], []
    for train_idx, test_idx in splitter.split(X):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

        model = model_factory(params)
        model.fit(X_tr, y_tr)
        y_pred = np.asarray(model.predict(X_te)).ravel().astype(int)

        s = classification_scores(y_te, y_pred)
        fold_f1.append(s["macro_f1"])
        fold_bal_acc.append(s["balanced_accuracy"])
        confusions.append(s["confusion_matrix"])

        if hasattr(model, "predict_proba"):
            proba = _aligned_proba(model, X_te)
            sw = compute_sample_weight("balanced", y_te)   # counter flat dominance
            fold_logloss.append(
                log_loss(y_te, proba, labels=[0, 1, 2], sample_weight=sw))

    return {
        "fold_macro_f1": fold_f1,
        "fold_balanced_accuracy": fold_bal_acc,
        "mean_macro_f1": float(np.mean(fold_f1)),
        "std_macro_f1": float(np.std(fold_f1)),
        "mean_balanced_accuracy": float(np.mean(fold_bal_acc)),
        "mean_log_loss": float(np.mean(fold_logloss)) if fold_logloss else float("nan"),
        "confusion_matrix": np.sum(confusions, axis=0),
    }
