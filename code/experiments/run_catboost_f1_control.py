"""Build a like-for-like F1-tuned control for Notebook 02b.

The project F1 candidate pool is reselected inside the exact same 2024 and
rolling inner windows used by economic Optuna.  After F1 selects CatBoost
hyperparameters, the same 33 execution policies are ranked on that inner
window and frozen for the same untouched outer month.  Existing compatible
prediction caches are reused; no model is retrained here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from experiments.catboost_economic_optuna import (
    BASELINE_PARAMS,
    DEVELOPMENT_END,
    WIDTHS,
    parameter_fingerprint,
    prediction_cache_path as economic_prediction_cache_path,
    primary_evaluation_fold_ids,
)
from experiments.catboost_f1_control import (
    combine_objective_tables,
    control_scopes,
    select_f1_candidate,
)
from experiments.frozen_model_study import validate_prediction_frame
from experiments.run_catboost_economic_optuna import (
    MODEL,
    OUTPUT_ROOT as ECONOMIC_ROOT,
    EconomicOptunaRunner,
)
from experiments.run_tune_antibull_widths import (
    CANDIDATES_PATH,
    PRED_DIR as LEGACY_PREDICTION_ROOT,
    prediction_cache_path as legacy_prediction_cache_path,
)
from experiments.run_tune_dz75_regime import REGIMES

CODE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ECONOMIC_ROOT / "f1_control"
CLASSIFICATION_PATH = OUTPUT_ROOT / "classification_grid.parquet"
SELECTED_MODELS_PATH = OUTPUT_ROOT / "selected_models.parquet"
POLICY_GRID_PATH = OUTPUT_ROOT / "policy_grid.parquet"
SELECTED_POLICIES_PATH = OUTPUT_ROOT / "selected_policies.parquet"
PRIMARY_MONTHLY_PATH = OUTPUT_ROOT / "primary_monthly.parquet"
ROLLING_MONTHLY_PATH = OUTPUT_ROOT / "rolling_monthly.parquet"
COMPARISON_PATH = OUTPUT_ROOT / "comparison.parquet"
OBJECTIVE_COMPARISON_PATH = OUTPUT_ROOT / "objective_comparison.parquet"
RESULT_PATH = OUTPUT_ROOT / "result.json"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _macro_f1(actual: pd.Series, predicted: pd.Series) -> float:
    return float(
        f1_score(
            actual.astype(int),
            predicted.astype(int),
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    )


def _load_candidates() -> list[dict]:
    payload = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    candidates = payload["candidates"]
    if len(candidates) != 15:
        raise ValueError(f"expected 15 F1 candidates, found {len(candidates)}")
    return candidates


def _read_prediction(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing required prediction cache: {path}")
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return validate_prediction_frame(
        frame.sort_index(), model=MODEL, development_end=DEVELOPMENT_END
    )


class F1ControlRunner:
    def __init__(self) -> None:
        self.output_root = OUTPUT_ROOT
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.widths = WIDTHS
        self.scopes = control_scopes()
        self.candidates = _load_candidates()
        self.runner = EconomicOptunaRunner(
            output_root=ECONOMIC_ROOT,
            widths=WIDTHS,
            target_trials=15,
        )
        self.selected_records: dict[tuple[int, str], dict] = {}

    def _legacy_path(self, width: int, candidate: int, fold_id: int) -> Path:
        fold = self.runner.folds[fold_id]
        month = f"{fold.validation_start:%Y-%m}"
        return legacy_prediction_cache_path(
            width,
            candidate,
            self.candidates[candidate],
            fold_id,
            month,
        )

    def verify_baseline_compatibility(self) -> int:
        """Prove the legacy and economic runners generated identical predictions."""
        checked = 0
        compare_columns = [
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
            "y_true",
            f"{MODEL}_pred",
            f"{MODEL}_conf",
            f"{MODEL}_p0",
            f"{MODEL}_p1",
            f"{MODEL}_p2",
        ]
        for width in self.widths:
            # Economic Optuna evaluates trial 0 only through fold 13; fold 14
            # is generated solely for each scope's selected outer model.
            for fold_id, fold in enumerate(self.runner.folds[:14]):
                month = f"{fold.validation_start:%Y-%m}"
                old = _read_prediction(self._legacy_path(width, 0, fold_id))
                new_path = economic_prediction_cache_path(
                    ECONOMIC_ROOT / "predictions",
                    width=width,
                    params=BASELINE_PARAMS,
                    fold_id=fold_id,
                    month=month,
                )
                new = _read_prediction(new_path)
                pd.testing.assert_index_equal(old.index, new.index)
                for column in compare_columns:
                    if pd.api.types.is_numeric_dtype(old[column]):
                        np.testing.assert_allclose(
                            old[column].to_numpy(),
                            new[column].to_numpy(),
                            rtol=0.0,
                            atol=0.0,
                        )
                    else:
                        pd.testing.assert_series_equal(
                            old[column], new[column], check_names=False
                        )
                checked += 1
        return checked

    def preload_predictions(self) -> int:
        loaded = 0
        for width in self.widths:
            for candidate, params in enumerate(self.candidates):
                fingerprint = parameter_fingerprint(params)
                for fold_id in range(len(self.runner.folds)):
                    frame = _read_prediction(
                        self._legacy_path(width, candidate, fold_id)
                    )
                    self.runner.predictions[(width, fingerprint, fold_id)] = frame
                    loaded += 1
        return loaded

    def classification_grid(self) -> pd.DataFrame:
        rows = []
        for scope in self.scopes:
            for width in self.widths:
                for candidate, params in enumerate(self.candidates):
                    frames = [
                        self.runner.prediction(width, params, fold_id)
                        for fold_id in scope.inner_fold_ids
                    ]
                    frame = pd.concat(frames).sort_index()
                    actual = frame["y_true"].astype(int)
                    predicted = frame[f"{MODEL}_pred"].astype(int)
                    regimes = self.runner.bar_regimes.reindex(frame.index)
                    regime_scores = {}
                    for regime in REGIMES:
                        mask = regimes.eq(regime)
                        if not mask.any():
                            raise ValueError(
                                f"{scope.name}/dz{width} has no {regime} rows"
                            )
                        regime_scores[regime] = _macro_f1(
                            actual.loc[mask], predicted.loc[mask]
                        )
                    rows.append(
                        {
                            "scope": scope.name,
                            "width": width,
                            "candidate": candidate,
                            "params_fingerprint": parameter_fingerprint(params),
                            "overall_f1": _macro_f1(actual, predicted),
                            "bull_f1": regime_scores["bull"],
                            "sideways_f1": regime_scores["sideways"],
                            "bear_f1": regime_scores["bear"],
                            "robust_f1": min(regime_scores.values()),
                        }
                    )
        return pd.DataFrame(rows)

    def select_models_and_policies(
        self, classification: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        selected_models = []
        policy_grids = []
        selected_policies = []
        for scope in self.scopes:
            for width in self.widths:
                candidates = classification[
                    classification["scope"].eq(scope.name)
                    & classification["width"].eq(width)
                ]
                choice = select_f1_candidate(candidates).copy()
                candidate = int(choice["candidate"])
                params = self.candidates[candidate]
                choice["selected_params_json"] = json.dumps(
                    params, sort_keys=True
                )
                selected_models.append(choice.to_dict())

                grid = self.runner.policy_grid(
                    width=width,
                    scope=scope,
                    trial_number=candidate,
                    params=params,
                )
                grid["model_objective"] = "F1-tuned"
                policy_grids.append(grid)
                policy = grid.loc[grid["trial_selected"]].iloc[0].copy()
                for metric in (
                    "overall_f1",
                    "bull_f1",
                    "sideways_f1",
                    "bear_f1",
                    "robust_f1",
                ):
                    policy[metric] = choice[metric]
                policy["selected_params_json"] = json.dumps(
                    params, sort_keys=True
                )
                policy["model_objective"] = "F1-tuned"
                selected_policies.append(policy.to_dict())
                self.selected_records[(width, scope.name)] = {
                    "params": params,
                    "policy": policy.to_dict(),
                }
        return (
            pd.DataFrame(selected_models),
            pd.concat(policy_grids, ignore_index=True),
            pd.DataFrame(selected_policies),
        )

    def evaluate(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        primary_rows = []
        rolling_rows = []
        primary_components = {}
        rolling_components = {width: [] for width in self.widths}
        for width in self.widths:
            primary = self.selected_records[(width, "primary_2024")]
            monthly, components = self.runner.evaluate_policy(
                width=width,
                scope_name="primary_2024",
                params=primary["params"],
                policy=primary["policy"],
                fold_ids=primary_evaluation_fold_ids(),
                mode="frozen",
            )
            monthly["model_objective"] = "F1-tuned"
            primary_rows.append(monthly)
            primary_components[width] = components

            for scope in self.scopes:
                if scope.outer_fold_id is None:
                    continue
                record = self.selected_records[(width, scope.name)]
                outer, components = self.runner.evaluate_policy(
                    width=width,
                    scope_name=scope.name,
                    params=record["params"],
                    policy=record["policy"],
                    fold_ids=(scope.outer_fold_id,),
                    mode="rolling",
                )
                outer["model_objective"] = "F1-tuned"
                rolling_rows.append(outer)
                rolling_components[width].extend(components)

        primary_monthly = pd.concat(primary_rows, ignore_index=True)
        rolling_monthly = pd.concat(rolling_rows, ignore_index=True)
        comparisons = []
        for width in self.widths:
            comparisons.append(
                self.runner._aggregate_components(
                    width=width,
                    mode="frozen_2025_h1",
                    components=primary_components[width],
                )
            )
            comparisons.append(
                self.runner._aggregate_components(
                    width=width,
                    mode="frozen_apr_jun",
                    components=primary_components[width][3:6],
                )
            )
            comparisons.append(
                self.runner._aggregate_components(
                    width=width,
                    mode="rolling_apr_jun",
                    components=rolling_components[width],
                )
            )
        comparison = pd.DataFrame(comparisons)
        comparison["model_objective"] = "F1-tuned"
        return primary_monthly, rolling_monthly, comparison

    def run(self) -> dict:
        started = time.time()
        compatibility_checks = self.verify_baseline_compatibility()
        prediction_cells = self.preload_predictions()
        classification = self.classification_grid()
        selected_models, policy_grid, selected_policies = (
            self.select_models_and_policies(classification)
        )
        primary_monthly, rolling_monthly, comparison = self.evaluate()
        economic = pd.read_parquet(ECONOMIC_ROOT / "comparison.parquet")
        objective_comparison = combine_objective_tables(economic, comparison)

        expected = {
            "classification": (len(classification), 180),
            "selected_models": (len(selected_models), 12),
            "policy_grid": (len(policy_grid), 396),
            "selected_policies": (len(selected_policies), 12),
            "primary_monthly": (len(primary_monthly), 18),
            "rolling_monthly": (len(rolling_monthly), 9),
            "comparison": (len(comparison), 9),
            "objective_comparison": (len(objective_comparison), 18),
        }
        mismatches = {key: value for key, value in expected.items() if value[0] != value[1]}
        if mismatches:
            raise RuntimeError(f"unexpected artifact row counts: {mismatches}")

        classification.to_parquet(CLASSIFICATION_PATH, index=False)
        selected_models.to_parquet(SELECTED_MODELS_PATH, index=False)
        policy_grid.to_parquet(POLICY_GRID_PATH, index=False)
        selected_policies.to_parquet(SELECTED_POLICIES_PATH, index=False)
        primary_monthly.to_parquet(PRIMARY_MONTHLY_PATH, index=False)
        rolling_monthly.to_parquet(ROLLING_MONTHLY_PATH, index=False)
        comparison.to_parquet(COMPARISON_PATH, index=False)
        objective_comparison.to_parquet(OBJECTIVE_COMPARISON_PATH, index=False)

        result = {
            "control": "project 15-candidate robust-F1 tuning",
            "same_inner_windows": True,
            "same_execution_policy_grid": True,
            "baseline_compatibility_checks": compatibility_checks,
            "prediction_cells_reused": prediction_cells,
            "classification_rows": len(classification),
            "selected_model_rows": len(selected_models),
            "policy_rows": len(policy_grid),
            "selected_policy_rows": len(selected_policies),
            "primary_monthly_rows": len(primary_monthly),
            "rolling_monthly_rows": len(rolling_monthly),
            "comparison_rows": len(comparison),
            "objective_comparison_rows": len(objective_comparison),
            "manual_selection": True,
            "elapsed_s": round(time.time() - started, 1),
        }
        _write_json(RESULT_PATH, result)
        return result


def main() -> int:
    result = F1ControlRunner().run()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())