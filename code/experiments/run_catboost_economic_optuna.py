"""Run true economic Optuna for CatBoost on dz55/dz65/dz75.

The primary study tunes on April-December 2024 and freezes the selected model
hyperparameters, confidence threshold, and bracket for January-June 2025.
Three independent rolling studies retune on nine earlier months and audit the
next untouched month (April, May, June 2025).  All trials and failed diagnostic
rows remain visible; this runner never emits an automatic no-trade decision.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml

from evaluation.economics import economics_summary
from evaluation.trades_intrabar import simulate_bracket_trades_intrabar
from experiments.catboost_economic_optuna import (
    BASELINE_PARAMS,
    DEVELOPMENT_END,
    GEOMETRIES,
    MIN_SIDE_TRADES,
    MIN_TRADES,
    SEED,
    TAUS,
    TRIALS,
    WIDTHS,
    StudyScope,
    completed_trial_count,
    constraint_values,
    constraints_from_trial,
    monthly_folds,
    parameter_fingerprint,
    policy_choices,
    prediction_cache_path,
    primary_evaluation_fold_ids,
    protocol_fingerprint,
    rank_policy,
    robust_score,
    sample_catboost_params,
    study_scopes,
)
from experiments.frozen_model_study import validate_prediction_frame
from experiments.run_tune_dz75_regime import (
    REGIMES,
    past_regime_labels,
    regime_balanced_weights,
)
from experiments.run_walkforward import build_walkforward_xy
from experiments.walkforward import run_walkforward_predictions
from features.build import make_label
from models.zoo import MODELS

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
MINUTE_PATH = CODE_ROOT / "data" / "btcusdt_1m_2024_2026.parquet"
OUTPUT_ROOT = (
    CODE_ROOT / "experiments" / "cache" / "tuning" / "catboost_economic_optuna"
)
MODEL = "catboost_balanced"


def _json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _diagnostics(row: pd.Series | dict, *, n_folds: int) -> tuple[str, ...]:
    required_positive = math.ceil(2 * n_folds / 3)
    checks = (
        (int(row["trades"]) >= MIN_TRADES, f"trades<{MIN_TRADES}"),
        (int(row["n_long"]) >= MIN_SIDE_TRADES, f"long<{MIN_SIDE_TRADES}"),
        (int(row["n_short"]) >= MIN_SIDE_TRADES, f"short<{MIN_SIDE_TRADES}"),
        (
            int(row["positive_folds"]) >= required_positive,
            f"positive_folds<{required_positive}",
        ),
    )
    return tuple(label for passed, label in checks if not passed)


class EconomicOptunaRunner:
    def __init__(
        self,
        *,
        output_root: Path = OUTPUT_ROOT,
        widths: tuple[int, ...] = WIDTHS,
        target_trials: int = TRIALS,
        scopes: tuple[str, ...] | None = None,
        smoke: bool = False,
    ) -> None:
        self.output_root = Path(output_root)
        self.widths = widths
        self.target_trials = int(target_trials)
        self.smoke = bool(smoke)
        if any(width not in WIDTHS for width in widths):
            raise ValueError(f"unsupported widths: {widths}")
        if self.target_trials < 1:
            raise ValueError("target_trials must be positive")

        available = study_scopes()
        if smoke:
            available = [StudyScope("smoke_primary", (0,), None)]
        if scopes is not None:
            requested = set(scopes)
            available = [scope for scope in available if scope.name in requested]
            if len(available) != len(requested):
                found = {scope.name for scope in available}
                raise ValueError(f"unknown scopes: {sorted(requested - found)}")
        self.scopes = available
        self.folds = monthly_folds()
        self.policy_space = (
            ((TAUS[0], GEOMETRIES[0]), (TAUS[1], GEOMETRIES[0]))
            if smoke
            else policy_choices()
        )

        self.cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        self.fee_bps = float(self.cfg["instruments"]["btc"]["taker_fee_bps"])
        self.prediction_root = self.output_root / "predictions"
        self.study_root = self.output_root / "studies"
        self.prediction_root.mkdir(parents=True, exist_ok=True)
        self.study_root.mkdir(parents=True, exist_ok=True)

        bars = pd.read_parquet(
            CODE_ROOT / self.cfg["instruments"]["btc"]["working_parquet"]
        )
        bars.index = pd.to_datetime(bars.index, utc=True)
        self.bars = bars.sort_index().loc[lambda frame: frame.index < DEVELOPMENT_END]
        minute = pd.read_parquet(MINUTE_PATH)
        minute.index = pd.to_datetime(minute.index, utc=True)
        self.minute = minute.sort_index().loc[
            lambda frame: frame.index < DEVELOPMENT_END
        ]
        self.bar_regimes = past_regime_labels(self.bars["close"])
        self.features: dict[int, tuple[pd.DataFrame, pd.Series]] = {}
        self.predictions: dict[tuple[int, str, int], pd.DataFrame] = {}
        self.simulations: dict[
            tuple[int, str, int, float, int, int, int],
            tuple[pd.DataFrame, pd.Series],
        ] = {}
        self.selected_records: dict[tuple[int, str], dict] = {}

    def _xy(self, width: int) -> tuple[pd.DataFrame, pd.Series]:
        if width in self.features:
            return self.features[width]
        X, y, _ = build_walkforward_xy(
            "btc",
            self.cfg,
            horizon=1,
            sentiment="both",
            label_fn=lambda feat: make_label(feat, threshold_bps=width, horizon=1),
            orderflow=True,
            positioning=True,
        )
        X = X[X.index < DEVELOPMENT_END]
        y = y.reindex(X.index)
        known = self.bar_regimes.reindex(X.index).isin(REGIMES)
        self.features[width] = (X.loc[known], y.loc[known])
        return self.features[width]

    def prediction(self, width: int, params: dict, fold_id: int) -> pd.DataFrame:
        fingerprint = parameter_fingerprint(params)
        key = (width, fingerprint, fold_id)
        if key in self.predictions:
            return self.predictions[key]
        fold = self.folds[fold_id]
        month = f"{fold.validation_start:%Y-%m}"
        path = prediction_cache_path(
            self.prediction_root,
            width=width,
            params=params,
            fold_id=fold_id,
            month=month,
        )
        if path.exists():
            frame = pd.read_parquet(path)
            frame.index = pd.to_datetime(frame.index, utc=True)
            frame = validate_prediction_frame(
                frame.sort_index(), model=MODEL, development_end=DEVELOPMENT_END
            )
            self.predictions[key] = frame
            return frame

        X, y = self._xy(width)

        def training_weights(
            X_train: pd.DataFrame, _y_train: pd.Series
        ) -> pd.Series:
            regimes = self.bar_regimes.reindex(X_train.index)
            if not regimes.isin(REGIMES).all():
                raise ValueError("unknown regime remained inside a training fold")
            return regime_balanced_weights(regimes)

        frame = run_walkforward_predictions(
            X,
            y,
            windows=[fold],
            model_factory=MODELS[MODEL],
            model_name=MODEL,
            params=params,
            cache_path=path,
            min_train_rows=500,
            min_validation_rows=50,
            train_tail_trim=1,
            sample_weight_fn=training_weights,
        )
        if frame.empty:
            raise RuntimeError(
                f"empty predictions: dz{width} {fingerprint} {fold.name}"
            )
        frame = validate_prediction_frame(
            frame.sort_index(), model=MODEL, development_end=DEVELOPMENT_END
        )
        self.predictions[key] = frame
        return frame

    def simulate(
        self,
        width: int,
        params: dict,
        fold_id: int,
        *,
        tau: float,
        geometry: tuple[int, int, int],
    ) -> tuple[pd.DataFrame, pd.Series]:
        fingerprint = parameter_fingerprint(params)
        tp_bps, sl_bps, max_hold = geometry
        key = (width, fingerprint, fold_id, tau, tp_bps, sl_bps, max_hold)
        if key in self.simulations:
            return self.simulations[key]
        fold = self.folds[fold_id]
        frame = self.prediction(width, params, fold_id)
        prediction = frame[f"{MODEL}_pred"].astype(int)
        confidence = frame[f"{MODEL}_conf"].astype(float)
        boundary = fold.validation_end + pd.Timedelta(minutes=15)
        path_safe = (
            prediction.index + pd.Timedelta(minutes=15 * (max_hold + 1))
            <= boundary
        )
        scope_bars = self.bars[
            (self.bars.index >= fold.validation_start)
            & (self.bars.index < boundary)
        ]
        ledger, per_bar = simulate_bracket_trades_intrabar(
            scope_bars,
            self.minute,
            prediction.loc[path_safe],
            confidence.loc[path_safe],
            tau=float(tau),
            tp_bps=float(tp_bps),
            sl_bps=float(sl_bps),
            max_hold=int(max_hold),
            fee_bps=self.fee_bps,
        )
        self.simulations[key] = (ledger, per_bar)
        return ledger, per_bar

    def _summary(
        self,
        ledgers: list[pd.DataFrame],
        returns: list[pd.Series],
        fold_nets: list[float],
    ) -> dict:
        ledger = pd.concat(ledgers, ignore_index=True)
        per_bar = pd.concat(returns).sort_index()
        summary = economics_summary(per_bar)
        regimes = self.bar_regimes.reindex(per_bar.index)
        regime_sortino = {
            regime: economics_summary(per_bar[regimes == regime])["sortino"]
            for regime in REGIMES
        }
        result = {
            "trades": int(len(ledger)),
            "pooled_gross": float(ledger["gross_return"].sum()),
            "pooled_net": float(ledger["net_return"].sum()),
            "pooled_sortino": float(summary["sortino"]),
            "pooled_sharpe": float(summary["sharpe"]),
            "positive_folds": int(sum(value > 0.0 for value in fold_nets)),
            "n_long": int((ledger["side"] == 1).sum()),
            "n_short": int((ledger["side"] == -1).sum()),
            "long_net": float(
                ledger.loc[ledger["side"] == 1, "net_return"].sum()
            ),
            "short_net": float(
                ledger.loc[ledger["side"] == -1, "net_return"].sum()
            ),
            "bull_sortino": float(regime_sortino["bull"]),
            "sideways_sortino": float(regime_sortino["sideways"]),
            "bear_sortino": float(regime_sortino["bear"]),
        }
        result["robust_score"] = robust_score(
            pooled_sortino=result["pooled_sortino"],
            pooled_sharpe=result["pooled_sharpe"],
            bull_sortino=result["bull_sortino"],
            sideways_sortino=result["sideways_sortino"],
            bear_sortino=result["bear_sortino"],
        )
        return result

    def policy_grid(
        self,
        *,
        width: int,
        scope: StudyScope,
        trial_number: int,
        params: dict,
    ) -> pd.DataFrame:
        rows = []
        for tau, geometry in self.policy_space:
            ledgers: list[pd.DataFrame] = []
            returns: list[pd.Series] = []
            fold_nets: list[float] = []
            for fold_id in scope.inner_fold_ids:
                ledger, per_bar = self.simulate(
                    width, params, fold_id, tau=tau, geometry=geometry
                )
                ledgers.append(ledger)
                returns.append(per_bar)
                fold_nets.append(float(ledger["net_return"].sum()))
            row = self._summary(ledgers, returns, fold_nets)
            row.update(
                {
                    "scope": scope.name,
                    "width": width,
                    "trial": trial_number,
                    "params_fingerprint": parameter_fingerprint(params),
                    "tau": float(tau),
                    "tp_bps": int(geometry[0]),
                    "sl_bps": int(geometry[1]),
                    "max_hold": int(geometry[2]),
                }
            )
            constraints = constraint_values(row, n_folds=len(scope.inner_fold_ids))
            row["constraint_violation"] = float(
                sum(max(0.0, value) for value in constraints)
            )
            row["diagnostics"] = ", ".join(
                _diagnostics(row, n_folds=len(scope.inner_fold_ids))
            )
            rows.append(row)
        grid = pd.DataFrame(rows)
        selected = rank_policy(grid, n_folds=len(scope.inner_fold_ids))
        grid["trial_selected"] = False
        grid.loc[selected.name, "trial_selected"] = True
        return grid

    def _scope_dir(self, width: int, scope: StudyScope) -> Path:
        return self.study_root / f"dz{width}" / scope.name

    def _study(self, width: int, scope: StudyScope) -> optuna.Study:
        directory = self._scope_dir(width, scope)
        directory.mkdir(parents=True, exist_ok=True)
        db = directory / "optuna.db"
        sampler = optuna.samplers.TPESampler(
            seed=SEED,
            n_startup_trials=min(5, self.target_trials),
            constraints_func=constraints_from_trial,
        )
        study = optuna.create_study(
            study_name=f"catboost_economic_dz{width}_{scope.name}",
            storage=f"sqlite:///{db.as_posix()}",
            direction="maximize",
            sampler=sampler,
            pruner=optuna.pruners.NopPruner(),
            load_if_exists=True,
        )
        fingerprint = protocol_fingerprint()
        saved = study.user_attrs.get("protocol_fingerprint")
        if saved is not None and saved != fingerprint:
            raise RuntimeError(
                f"study fingerprint mismatch: saved={saved} current={fingerprint}"
            )
        study.set_user_attr("protocol_fingerprint", fingerprint)
        if not study.trials:
            study.enqueue_trial(BASELINE_PARAMS, user_attrs={"control": "project baseline"})
        return study

    def optimize_scope(self, width: int, scope: StudyScope) -> tuple[pd.DataFrame, pd.Series]:
        study = self._study(width, scope)
        directory = self._scope_dir(width, scope)

        def objective(trial: optuna.Trial) -> float:
            params = sample_catboost_params(trial)
            grid = self.policy_grid(
                width=width,
                scope=scope,
                trial_number=trial.number,
                params=params,
            )
            selected = grid.loc[grid["trial_selected"]].iloc[0]
            constraints = constraint_values(
                selected, n_folds=len(scope.inner_fold_ids)
            )
            trial.set_user_attr("constraints", list(constraints))
            for key in (
                "tau",
                "tp_bps",
                "sl_bps",
                "max_hold",
                "trades",
                "n_long",
                "n_short",
                "positive_folds",
                "pooled_gross",
                "pooled_net",
                "pooled_sortino",
                "pooled_sharpe",
                "bull_sortino",
                "sideways_sortino",
                "bear_sortino",
                "robust_score",
                "constraint_violation",
                "diagnostics",
            ):
                trial.set_user_attr(key, _json_value(selected[key]))
            grid.to_parquet(directory / f"trial_{trial.number:03d}_policy_grid.parquet", index=False)
            return float(selected["robust_score"])

        remaining = self.target_trials - completed_trial_count(study)
        if remaining > 0:
            study.optimize(objective, n_trials=remaining, gc_after_trial=True)
        if completed_trial_count(study) != self.target_trials:
            raise RuntimeError(
                f"{scope.name}/dz{width}: expected {self.target_trials} complete trials, "
                f"found {completed_trial_count(study)}"
            )

        complete = {
            trial.number: trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        }
        grids = []
        trial_rows = []
        winner_rows = []
        for trial_number, trial in complete.items():
            path = directory / f"trial_{trial_number:03d}_policy_grid.parquet"
            if not path.exists():
                raise FileNotFoundError(f"missing completed-trial grid: {path}")
            grid = pd.read_parquet(path)
            grids.append(grid)
            winner = grid.loc[grid["trial_selected"]].iloc[0].to_dict()
            winner_rows.append(winner)
            trial_rows.append(
                {
                    "scope": scope.name,
                    "width": width,
                    "trial": trial_number,
                    "objective_value": float(trial.value),
                    "feasible": all(value <= 0 for value in constraints_from_trial(trial)),
                    "params_fingerprint": parameter_fingerprint(trial.params),
                    **{f"param_{key}": value for key, value in trial.params.items()},
                    **{f"selected_{key}": value for key, value in trial.user_attrs.items() if key != "constraints"},
                }
            )
        winners = pd.DataFrame(winner_rows)
        selected = rank_policy(winners, n_folds=len(scope.inner_fold_ids)).copy()
        selected_trial = complete[int(selected["trial"])]
        selected["selected_params_json"] = json.dumps(
            selected_trial.params, sort_keys=True
        )
        selected["feasible"] = not bool(selected["diagnostics"])
        selected["protocol_fingerprint"] = protocol_fingerprint()
        self.selected_records[(width, scope.name)] = {
            "row": selected.to_dict(),
            "params": dict(selected_trial.params),
        }
        return pd.DataFrame(trial_rows), pd.concat(grids, ignore_index=True)

    def evaluate_policy(
        self,
        *,
        width: int,
        scope_name: str,
        params: dict,
        policy: dict,
        fold_ids: tuple[int, ...],
        mode: str,
    ) -> tuple[pd.DataFrame, list[tuple[pd.DataFrame, pd.Series]]]:
        geometry = (
            int(policy["tp_bps"]),
            int(policy["sl_bps"]),
            int(policy["max_hold"]),
        )
        rows = []
        components = []
        for fold_id in fold_ids:
            ledger, per_bar = self.simulate(
                width,
                params,
                fold_id,
                tau=float(policy["tau"]),
                geometry=geometry,
            )
            metrics = self._summary(
                [ledger], [per_bar], [float(ledger["net_return"].sum())]
            )
            metrics.update(
                {
                    "mode": mode,
                    "scope": scope_name,
                    "width": width,
                    "month": f"{self.folds[fold_id].validation_start:%Y-%m}",
                    "fold_id": fold_id,
                    "tau": float(policy["tau"]),
                    "tp_bps": geometry[0],
                    "sl_bps": geometry[1],
                    "max_hold": geometry[2],
                    "params_fingerprint": parameter_fingerprint(params),
                    "selected_params_json": json.dumps(params, sort_keys=True),
                }
            )
            rows.append(metrics)
            components.append((ledger, per_bar))
        return pd.DataFrame(rows), components

    def _aggregate_components(
        self,
        *,
        width: int,
        mode: str,
        components: list[tuple[pd.DataFrame, pd.Series]],
    ) -> dict:
        ledgers = [ledger for ledger, _ in components]
        returns = [per_bar for _, per_bar in components]
        fold_nets = [float(ledger["net_return"].sum()) for ledger in ledgers]
        result = self._summary(ledgers, returns, fold_nets)
        result.update({"width": width, "mode": mode, "months": len(components)})
        result["diagnostics"] = ", ".join(
            _diagnostics(result, n_folds=len(components))
        )
        return result

    def run(self) -> dict:
        started = time.time()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        _write_json(
            self.output_root / "manifest.json",
            {
                "protocol_fingerprint": protocol_fingerprint(),
                "widths": list(self.widths),
                "target_trials": self.target_trials,
                "study_scopes": [scope.name for scope in self.scopes],
                "smoke": self.smoke,
                "objective": "maximize weakest pooled/regime Sortino-Sharpe metric",
                "constraints": [
                    "trades>=50",
                    "long>=15",
                    "short>=15",
                    "positive_folds>=2/3",
                ],
                "manual_selection": True,
            },
        )

        all_trials = []
        all_grids = []
        selected_rows = []
        completed_studies = []
        for width in self.widths:
            for scope in self.scopes:
                trial_table, grid = self.optimize_scope(width, scope)
                all_trials.append(trial_table)
                all_grids.append(grid)
                selected = self.selected_records[(width, scope.name)]["row"]
                selected_rows.append(selected)
                completed_studies.append(f"dz{width}/{scope.name}")
                _write_json(
                    self.output_root / "run_state.json",
                    {
                        "protocol_fingerprint": protocol_fingerprint(),
                        "completed_studies": completed_studies,
                        "current": None,
                    },
                )

        trials = pd.concat(all_trials, ignore_index=True)
        grids = pd.concat(all_grids, ignore_index=True)
        selected = pd.DataFrame(selected_rows)
        trials.to_parquet(self.output_root / "trials.parquet", index=False)
        grids.to_parquet(self.output_root / "policy_grid.parquet", index=False)
        selected.to_parquet(
            self.output_root / "selected_policies.parquet", index=False
        )

        if self.smoke:
            result = {
                "status": "smoke_complete",
                "protocol_fingerprint": protocol_fingerprint(),
                "study_count": len(completed_studies),
                "trial_rows": len(trials),
                "policy_rows": len(grids),
                "elapsed_s": round(time.time() - started, 1),
            }
            _write_json(self.output_root / "result.json", result)
            return result

        primary_rows = []
        primary_components: dict[int, list[tuple[pd.DataFrame, pd.Series]]] = {}
        rolling_rows = []
        rolling_components: dict[int, list[tuple[pd.DataFrame, pd.Series]]] = {
            width: [] for width in self.widths
        }
        for width in self.widths:
            primary_record = self.selected_records[(width, "primary_2024")]
            frame, components = self.evaluate_policy(
                width=width,
                scope_name="primary_2024",
                params=primary_record["params"],
                policy=primary_record["row"],
                fold_ids=primary_evaluation_fold_ids(),
                mode="frozen",
            )
            primary_rows.append(frame)
            primary_components[width] = components

            for scope in self.scopes:
                if scope.outer_fold_id is None:
                    continue
                record = self.selected_records[(width, scope.name)]
                outer, outer_components = self.evaluate_policy(
                    width=width,
                    scope_name=scope.name,
                    params=record["params"],
                    policy=record["row"],
                    fold_ids=(scope.outer_fold_id,),
                    mode="rolling",
                )
                rolling_rows.append(outer)
                rolling_components[width].extend(outer_components)

        primary_monthly = pd.concat(primary_rows, ignore_index=True)
        rolling_monthly = pd.concat(rolling_rows, ignore_index=True)
        comparison_rows = []
        for width in self.widths:
            comparison_rows.append(
                self._aggregate_components(
                    width=width,
                    mode="frozen_2025_h1",
                    components=primary_components[width],
                )
            )
            comparison_rows.append(
                self._aggregate_components(
                    width=width,
                    mode="frozen_apr_jun",
                    components=primary_components[width][3:6],
                )
            )
            comparison_rows.append(
                self._aggregate_components(
                    width=width,
                    mode="rolling_apr_jun",
                    components=rolling_components[width],
                )
            )
        comparison = pd.DataFrame(comparison_rows)
        primary_monthly.to_parquet(
            self.output_root / "primary_monthly.parquet", index=False
        )
        rolling_monthly.to_parquet(
            self.output_root / "rolling_monthly.parquet", index=False
        )
        comparison.to_parquet(self.output_root / "comparison.parquet", index=False)

        result = {
            "protocol_fingerprint": protocol_fingerprint(),
            "study_count": len(completed_studies),
            "trial_rows": len(trials),
            "policy_rows": len(grids),
            "selected_policy_rows": len(selected),
            "primary_monthly_rows": len(primary_monthly),
            "rolling_monthly_rows": len(rolling_monthly),
            "comparison_rows": len(comparison),
            "manual_selection": True,
            "elapsed_s": round(time.time() - started, 1),
        }
        _write_json(self.output_root / "result.json", result)
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--widths", type=int, nargs="+", default=list(WIDTHS))
    parser.add_argument("--scopes", nargs="+")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    root = OUTPUT_ROOT / "smoke" if args.smoke else OUTPUT_ROOT
    runner = EconomicOptunaRunner(
        output_root=root,
        widths=tuple(args.widths),
        target_trials=args.trials,
        scopes=None if args.scopes is None else tuple(args.scopes),
        smoke=args.smoke,
    )
    result = runner.run()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
