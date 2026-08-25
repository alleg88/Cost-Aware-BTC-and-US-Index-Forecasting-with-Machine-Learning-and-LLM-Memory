"""Frozen baseline-model protocol used by Notebook 02e."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.catboost_matched_ablation import (
    REGIMES,
    SELECTION_END,
    SELECTION_START,
    GEOMETRIES,
    MIN_SIDE_TRADES,
    MIN_TRADES,
    TAUS,
    economic_ranking_key,
    robust_f1_score,
    robust_score,
    validate_fold_regime_counts,
)
from evaluation.economics import economics_summary
from experiments.catboost_execution_scoring import simulate_policy
from experiments.raw_hold_control import MODEL_NAMES
from experiments.raw_hold_control import simulate_fixed_hold
from experiments.run_catboost_matched_ablation import (
    PreparedData,
    _atomic_json,
    _atomic_parquet,
    _folds,
    _macro_f1,
    combine_monthly_predictions,
    fit_fold_predictions,
    fit_span_predictions,
    load_prediction_cache,
    prediction_cache_fingerprint,
    stage_prediction_cache_fingerprint,
    summarize_forward_evidence,
    validate_frozen_policy_rows,
    write_prediction_cache,
)

WIDTHS = (55, 65, 75)
LOOKBACK_DAYS = (180,)
BASELINE_CANDIDATE_ID = 0
HOLD_BARS = (1,)
FOLD_COUNT = 5
POSITIVE_FOLDS_REQUIRED = 4
CALIBRATION_START = pd.Timestamp("2025-01-01", tz="UTC")
FORWARD_START = pd.Timestamp("2025-07-01", tz="UTC")
FORWARD_END = pd.Timestamp("2026-04-01", tz="UTC")


def policy_choices_for_hold(
    hold_bars: int,
) -> tuple[tuple[float, tuple[int, int, int]], ...]:
    """Return the 33 threshold/TP/SL policies for one 2024-frozen hold."""
    if hold_bars not in HOLD_BARS:
        raise ValueError(f"hold_bars must be one of {HOLD_BARS}")
    return tuple(
        (float(tau), (int(tp_bps), int(sl_bps), int(hold_bars)))
        for tau in TAUS
        for tp_bps, sl_bps, _ in GEOMETRIES
    )


def fit_plan(
    width_bps: int, lookback_days: int
) -> tuple[dict[str, object], ...]:
    """Return six causal H1 refits followed by the frozen forward fit."""
    if width_bps not in WIDTHS:
        raise ValueError(f"width_bps must be one of {WIDTHS}")
    if lookback_days not in LOOKBACK_DAYS:
        raise ValueError(f"lookback_days must be one of {LOOKBACK_DAYS}")
    edges = pd.date_range(CALIBRATION_START, FORWARD_START, freq="MS")
    rows = [
        {
            "stage": "calibration",
            "train_end": start,
            "test_start": start,
            "test_end": end,
            "lookback_days": int(lookback_days),
        }
        for start, end in zip(edges[:-1], edges[1:])
    ]
    rows.append(
        {
            "stage": "forward",
            "train_end": FORWARD_START,
            "test_start": FORWARD_START,
            "test_end": FORWARD_END,
            "lookback_days": int(lookback_days),
        }
    )
    return tuple(rows)


def hold_constraint_violation(row: Mapping[str, object]) -> float:
    """Measure failure of the pre-declared five-fold adequacy guards."""
    return float(
        max(0, MIN_TRADES - int(row["trades"]))
        + max(0, MIN_SIDE_TRADES - int(row["n_long"]))
        + max(0, MIN_SIDE_TRADES - int(row["n_short"]))
        + max(0, POSITIVE_FOLDS_REQUIRED - int(row["positive_segments"]))
    )


def select_hold_rows(grid: pd.DataFrame) -> pd.DataFrame:
    """Validate and retain the pre-declared 15-minute hold per Model/DZ."""
    required = {
        "model_name",
        "width_bps",
        "candidate_id",
        "hold_bars",
        "trades",
        "n_long",
        "n_short",
        "positive_segments",
        "pooled_net",
        "pooled_sortino",
        "pooled_sharpe",
    }
    missing = required.difference(grid.columns)
    if missing:
        raise ValueError(f"hold grid misses columns: {sorted(missing)}")
    rows = []
    keys = ["model_name", "width_bps", "candidate_id"]
    for _, group in grid.groupby(keys, sort=False):
        if len(group) != 1 or set(group["hold_bars"].astype(int)) != {1}:
            raise ValueError("each Model/DZ must contain only the fixed 15-minute hold")
        winner = group.iloc[0].to_dict()
        winner["constraint_violation"] = hold_constraint_violation(winner)
        winner["selection_rule"] = "pre-declared fixed 15-minute hold"
        rows.append(winner)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def select_h1_policy_rows(grid: pd.DataFrame) -> pd.DataFrame:
    """Select one H1-frozen policy for every Model/DZ at fixed 180 days."""
    required = {
        "model_name",
        "width_bps",
        "candidate_id",
        "lookback_days",
        "policy_id",
    }
    missing = required.difference(grid.columns)
    if missing:
        raise ValueError(f"H1 grid misses columns: {sorted(missing)}")
    rows = []
    keys = ["model_name", "width_bps", "candidate_id"]
    for _, group in grid.groupby(keys, sort=False):
        if set(group["lookback_days"].astype(int)) != set(LOOKBACK_DAYS):
            raise ValueError("each Model/DZ must contain only the 180-day history")
        expected = len(LOOKBACK_DAYS) * len(policy_choices_for_hold(1))
        if len(group) != expected:
            raise ValueError(f"each Model/DZ must contain exactly {expected} H1 rows")
        winner = min(
            group.to_dict("records"),
            key=lambda row: (
                *economic_ranking_key(row, n_segments=6),
            ),
        )
        winner["selection_rule"] = (
            "adequacy, robust score, Sortino, net return, trades, policy"
        )
        rows.append(winner)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def expected_model_counts() -> dict[str, int]:
    return {
        "classification_2024.parquet": 3,
        "hold_grid_2024.parquet": 3,
        "selected_holds_2024.parquet": 3,
        "calibration_policy_grid_2025h1.parquet": 99,
        "selected_policies_2025h1.parquet": 3,
        "raw_forward_summary.parquet": 3,
        "forward_monthly.parquet": 27,
        "forward_quarterly.parquet": 9,
        "forward_summary.parquet": 3,
    }


def validate_model_artifacts(root: Path, *, model_name: str) -> dict[str, int]:
    """Validate the exact completed artifact contract for one model family."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unknown model_name: {model_name}")
    actual: dict[str, int] = {}
    for filename, expected in expected_model_counts().items():
        path = Path(root) / filename
        if not path.exists():
            raise FileNotFoundError(f"missing {filename}")
        rows = len(pd.read_parquet(path))
        if rows != expected:
            raise ValueError(f"{filename} must contain exactly {expected} rows")
        actual[filename] = rows
    return actual


def _classification_row(
    *,
    model_name: str,
    width_bps: int,
    fold_frames: Sequence[pd.DataFrame],
    regimes: pd.Series,
) -> dict[str, object]:
    combined = pd.concat(fold_frames, ignore_index=True).set_index("timestamp")
    combined.index = pd.to_datetime(combined.index, utc=True)
    fold_scores, fold_overall = [], []
    for frame in fold_frames:
        indexed = frame.copy().set_index("timestamp")
        indexed.index = pd.to_datetime(indexed.index, utc=True)
        fold_regimes = regimes.reindex(indexed.index)
        fold_overall.append(_macro_f1(indexed["y_true"], indexed["pred"]))
        fold_scores.append(
            [
                _macro_f1(
                    indexed.loc[fold_regimes == regime, "y_true"],
                    indexed.loc[fold_regimes == regime, "pred"],
                )
                for regime in REGIMES
            ]
        )
    pooled_regimes = regimes.reindex(combined.index)
    pooled = {
        regime: _macro_f1(
            combined.loc[pooled_regimes == regime, "y_true"],
            combined.loc[pooled_regimes == regime, "pred"],
        )
        for regime in REGIMES
    }
    return {
        "model_name": model_name,
        "objective": "baseline",
        "width_bps": int(width_bps),
        "candidate_id": BASELINE_CANDIDATE_ID,
        "overall_f1": float(np.mean(fold_overall)),
        "robust_f1": robust_f1_score(fold_scores),
        "bull_f1": pooled["bull"],
        "sideways_f1": pooled["sideways"],
        "bear_f1": pooled["bear"],
    }


def _economic_row(
    *,
    model_name: str,
    width_bps: int,
    candidate_id: int,
    policy_id: int,
    tau: float,
    tp_bps: int,
    sl_bps: int,
    hold_bars: int,
    ledgers: Sequence[pd.DataFrame],
    returns: Sequence[pd.Series],
    segment_nets: Sequence[float],
    regimes: pd.Series,
) -> dict[str, object]:
    ledger = pd.concat(ledgers, ignore_index=True)
    per_bar = pd.concat(returns).sort_index()
    summary = economics_summary(per_bar)
    bar_regimes = regimes.reindex(per_bar.index)
    regime_sortino = {
        regime: float(economics_summary(per_bar.loc[bar_regimes == regime])["sortino"])
        for regime in REGIMES
    }
    row: dict[str, object] = {
        "model_name": model_name,
        "objective": "baseline",
        "width_bps": int(width_bps),
        "candidate_id": int(candidate_id),
        "policy_id": int(policy_id),
        "tau": float(tau),
        "tp_bps": int(tp_bps),
        "sl_bps": int(sl_bps),
        "max_hold": int(hold_bars),
        "hold_bars": int(hold_bars),
        "hold_minutes": int(15 * hold_bars),
        "trades": int(len(ledger)),
        "pooled_gross": float(ledger["gross_return"].sum()),
        "pooled_net": float(ledger["net_return"].sum()),
        "pooled_sortino": float(summary["sortino"]),
        "pooled_sharpe": float(summary["sharpe"]),
        "positive_segments": int(sum(value > 0 for value in segment_nets)),
        "n_long": int((ledger["side"] == 1).sum()),
        "n_short": int((ledger["side"] == -1).sum()),
        "long_net": float(ledger.loc[ledger["side"] == 1, "net_return"].sum()),
        "short_net": float(ledger.loc[ledger["side"] == -1, "net_return"].sum()),
        "bull_sortino": regime_sortino["bull"],
        "sideways_sortino": regime_sortino["sideways"],
        "bear_sortino": regime_sortino["bear"],
    }
    row["robust_score"] = robust_score(
        pooled_sortino=float(row["pooled_sortino"]),
        pooled_sharpe=float(row["pooled_sharpe"]),
        bull_sortino=float(row["bull_sortino"]),
        sideways_sortino=float(row["sideways_sortino"]),
        bear_sortino=float(row["bear_sortino"]),
    )
    return row


class BaselineModelRunner:
    """Run one fixed model-family baseline through the matched policy protocol."""

    def __init__(
        self,
        *,
        output_root: Path,
        prepared: PreparedData,
        model_name: str,
        model_factory: Callable,
        widths: Sequence[int] = WIDTHS,
        candidate_id: int = BASELINE_CANDIDATE_ID,
        candidate_params: Mapping[str, Any] | None = None,
        sentiment_mode: str = "none",
        fee_bps: float = 5.0,
        smoke: bool = False,
    ):
        if model_name not in MODEL_NAMES:
            raise ValueError(f"unknown model_name: {model_name}")
        if candidate_id != BASELINE_CANDIDATE_ID:
            raise ValueError("matched baseline protocol permits only candidate 0")
        self.output_root = Path(output_root)
        self.prepared = prepared
        self.model_name = model_name
        self.model_factory = model_factory
        self.widths = tuple(int(width) for width in widths)
        self.candidate_id = int(candidate_id)
        self.candidate_params = dict(candidate_params or {})
        self.sentiment_mode = str(sentiment_mode)
        self.fee_bps = float(fee_bps)
        self.smoke = bool(smoke)
        self.fits = 0
        self.cache_hits = 0

    def _prediction_path(self, width: int, fold_id: int, fingerprint: str) -> Path:
        return self.output_root / "predictions" / (
            f"w{width}_candidate_00_fold_{fold_id:02d}_{fingerprint}.parquet"
        )

    def _stage_path(
        self, stage: str, width: int, lookback_days: int, fingerprint: str
    ) -> Path:
        return self.output_root / "stage_predictions" / stage / (
            f"w{width}_lb{lookback_days}_candidate_00_{fingerprint}.parquet"
        )

    def _fit_fold(
        self,
        *,
        width: int,
        X: pd.DataFrame,
        y: pd.Series,
        regimes: pd.Series,
        fold: Mapping[str, Any],
    ) -> pd.DataFrame:
        fingerprint = prediction_cache_fingerprint(
            width_bps=width,
            candidate_params=self.candidate_params,
            fold_metadata=fold,
            X=X,
            y=y,
            regimes=regimes,
            data_fingerprint=self.prepared.m15_fingerprint,
            model_name=self.model_name,
        )
        path = self._prediction_path(width, int(fold["fold_id"]), fingerprint)
        frame = load_prediction_cache(path, fingerprint)
        if frame is not None:
            self.cache_hits += 1
            return frame
        frame = fit_fold_predictions(
            X=X,
            y=y,
            regimes=regimes,
            fold_metadata=fold,
            width_bps=width,
            candidate_id=self.candidate_id,
            candidate_params=self.candidate_params,
            model_factory=self.model_factory,
            refit_id=(
                f"{self.model_name}-w{width}-f{int(fold['fold_id']):02d}-"
                f"{fingerprint[:12]}"
            ),
        )
        write_prediction_cache(path, frame, fingerprint)
        self.fits += 1
        return frame

    def _fit_stage(
        self,
        *,
        width: int,
        stage: str,
        lookback_days: int,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
    ) -> pd.DataFrame:
        X, y = self.prepared.features[width]
        regimes = self.prepared.regimes.reindex(X.index)
        fingerprint = stage_prediction_cache_fingerprint(
            stage=stage,
            width_bps=width,
            candidate_params=self.candidate_params,
            X=X,
            y=y,
            regimes=regimes,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            data_fingerprint=self.prepared.m15_fingerprint,
            model_name=self.model_name,
            lookback_days=lookback_days,
        )
        path = self._stage_path(stage, width, lookback_days, fingerprint)
        frame = load_prediction_cache(path, fingerprint)
        if frame is not None:
            self.cache_hits += 1
            return frame
        frame = fit_span_predictions(
            X=X,
            y=y,
            regimes=regimes,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            width_bps=width,
            candidate_id=self.candidate_id,
            candidate_params=self.candidate_params,
            model_factory=self.model_factory,
            fit_id=(
                f"{self.model_name}-{stage}-w{width}-lb{lookback_days}-"
                f"{fingerprint[:12]}"
            ),
            lookback_days=lookback_days,
        )
        write_prediction_cache(path, frame, fingerprint)
        self.fits += 1
        return frame

    def _score_holds(
        self,
        *,
        width: int,
        fold_frames: Sequence[pd.DataFrame],
        folds: Sequence[Mapping[str, Any]],
    ) -> pd.DataFrame:
        rows = []
        for hold_bars in HOLD_BARS:
            ledgers, returns, nets = [], [], []
            for frame, fold in zip(fold_frames, folds):
                start, end = pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"])
                bars = self.prepared.bars.loc[
                    (self.prepared.bars.index >= start)
                    & (self.prepared.bars.index < end)
                ]
                prediction = frame.set_index("timestamp")["pred"].astype(int)
                ledger, per_bar = simulate_fixed_hold(
                    bars, prediction, hold_bars=hold_bars, fee_bps=self.fee_bps
                )
                ledgers.append(ledger)
                returns.append(per_bar)
                nets.append(float(ledger["net_return"].sum()))
            rows.append(
                _economic_row(
                    model_name=self.model_name,
                    width_bps=width,
                    candidate_id=self.candidate_id,
                    policy_id=hold_bars - 1,
                    tau=0.0,
                    tp_bps=0,
                    sl_bps=0,
                    hold_bars=hold_bars,
                    ledgers=ledgers,
                    returns=returns,
                    segment_nets=nets,
                    regimes=self.prepared.regimes,
                )
            )
        return pd.DataFrame(rows)

    def _score_h1_policies(
        self,
        *,
        width: int,
        hold_bars: int,
        prediction: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = []
        month_edges = pd.date_range(CALIBRATION_START, FORWARD_START, freq="MS")
        for policy_id, (tau, geometry) in enumerate(policy_choices_for_hold(hold_bars)):
            tp_bps, sl_bps, max_hold = geometry
            ledger, per_bar = simulate_policy(
                bars=self.prepared.bars,
                execution=self.prepared.minute,
                prediction_frame=prediction,
                start=CALIBRATION_START,
                end=FORWARD_START,
                resolution="1m",
                tau=tau,
                tp_bps=tp_bps,
                sl_bps=sl_bps,
                max_hold=max_hold,
                fee_bps=self.fee_bps,
            )
            nets = [
                float(per_bar.loc[(per_bar.index >= left) & (per_bar.index < right)].sum())
                for left, right in zip(month_edges[:-1], month_edges[1:])
            ]
            rows.append(
                _economic_row(
                    model_name=self.model_name,
                    width_bps=width,
                    candidate_id=self.candidate_id,
                    policy_id=policy_id,
                    tau=tau,
                    tp_bps=tp_bps,
                    sl_bps=sl_bps,
                    hold_bars=max_hold,
                    ledgers=(ledger,),
                    returns=(per_bar,),
                    segment_nets=nets,
                    regimes=self.prepared.regimes,
                )
            )
        grid = pd.DataFrame(rows)
        fit_ids = sorted(prediction["refit_id"].astype(str).unique())
        grid["fit_id"] = "monthly-h1:" + "|".join(fit_ids)
        grid["monthly_fit_count"] = len(fit_ids)
        grid["train_start"] = pd.to_datetime(prediction["train_start"], utc=True).min()
        grid["train_end"] = pd.to_datetime(prediction["train_end"], utc=True).max()
        return grid

    def run(self) -> dict[str, object]:
        classification_rows, hold_frames = [], []
        fold_count = FOLD_COUNT
        for width in self.widths:
            X_all, y_all = self.prepared.features[width]
            selection = (X_all.index >= SELECTION_START) & (X_all.index < SELECTION_END)
            X = X_all.loc[selection]
            y = y_all.reindex(X.index)
            regimes = self.prepared.regimes.reindex(X.index)
            known = regimes.isin(REGIMES)
            X, y, regimes = X.loc[known], y.loc[known], regimes.loc[known]
            folds = _folds(X.index)[:fold_count]
            validate_fold_regime_counts(
                [
                    regimes.iloc[list(fold["test_positions"])].value_counts().to_dict()
                    for fold in folds
                ]
            )
            fold_frames = [
                self._fit_fold(width=width, X=X, y=y, regimes=regimes, fold=fold)
                for fold in folds
            ]
            classification_rows.append(
                _classification_row(
                    model_name=self.model_name,
                    width_bps=width,
                    fold_frames=fold_frames,
                    regimes=self.prepared.regimes,
                )
            )
            hold_frames.append(
                self._score_holds(width=width, fold_frames=fold_frames, folds=folds)
            )
        classification = pd.DataFrame(classification_rows)
        hold_grid = pd.concat(hold_frames, ignore_index=True)
        selected_holds = select_hold_rows(hold_grid)
        _atomic_parquet(classification, self.output_root / "classification_2024.parquet")
        _atomic_parquet(hold_grid, self.output_root / "hold_grid_2024.parquet")
        _atomic_parquet(selected_holds, self.output_root / "selected_holds_2024.parquet")
        if self.smoke:
            result = {
                "status": "smoke_complete",
                "model_name": self.model_name,
                "fits": self.fits,
                "cache_hits": self.cache_hits,
            }
            _atomic_json(result, self.output_root / "result.json")
            return result

        policy_grids = []
        for width in self.widths:
            hold = int(
                selected_holds.loc[selected_holds["width_bps"] == width, "hold_bars"].iloc[0]
            )
            for lookback_days in LOOKBACK_DAYS:
                monthly = [
                    self._fit_stage(
                        width=width,
                        stage=f"calibration_{row['test_start']:%Y_%m}",
                        lookback_days=lookback_days,
                        train_end=pd.Timestamp(row["train_end"]),
                        test_start=pd.Timestamp(row["test_start"]),
                        test_end=pd.Timestamp(row["test_end"]),
                    )
                    for row in fit_plan(width, lookback_days)[:6]
                ]
                calibration = combine_monthly_predictions(monthly)
                grid = self._score_h1_policies(
                    width=width, hold_bars=hold, prediction=calibration
                )
                grid["lookback_days"] = int(lookback_days)
                policy_grids.append(grid)
        calibration_grid = pd.concat(policy_grids, ignore_index=True)
        selected_policies = select_h1_policy_rows(calibration_grid)
        _atomic_parquet(
            calibration_grid,
            self.output_root / "calibration_policy_grid_2025h1.parquet",
        )
        _atomic_parquet(
            selected_policies, self.output_root / "selected_policies_2025h1.parquet"
        )

        forward_predictions: dict[tuple[int, int], pd.DataFrame] = {}
        for policy in selected_policies.to_dict("records"):
            width = int(policy["width_bps"])
            lookback_days = int(policy["lookback_days"])
            forward_row = fit_plan(width, lookback_days)[-1]
            forward_predictions[(width, lookback_days)] = self._fit_stage(
                width=width,
                stage="forward",
                lookback_days=lookback_days,
                train_end=pd.Timestamp(forward_row["train_end"]),
                test_start=pd.Timestamp(forward_row["test_start"]),
                test_end=pd.Timestamp(forward_row["test_end"]),
            )

        raw_rows, monthly_frames, quarterly_frames, summary_frames = [], [], [], []
        evidence_root = self.output_root / "forward_evidence"
        for policy in selected_policies.to_dict("records"):
            width = int(policy["width_bps"])
            lookback_days = int(policy["lookback_days"])
            prediction = forward_predictions[(width, lookback_days)]
            fit_id = str(prediction["refit_id"].iloc[0])
            hold = int(policy["max_hold"])

            raw_pred = prediction.set_index("timestamp")["pred"].astype(int)
            forward_bars = self.prepared.bars.loc[
                (self.prepared.bars.index >= FORWARD_START)
                & (self.prepared.bars.index < FORWARD_END)
            ]
            raw_ledger, raw_returns = simulate_fixed_hold(
                forward_bars, raw_pred, hold_bars=hold, fee_bps=self.fee_bps
            )
            raw_policy = {
                "objective": "raw_fixed_hold",
                "width_bps": width,
                "candidate_id": self.candidate_id,
                "policy_id": -1,
                "tau": 0.0,
                "tp_bps": 0,
                "sl_bps": 0,
                "max_hold": hold,
                "fit_id": fit_id,
            }
            _, _, raw_summary = summarize_forward_evidence(
                per_bar=raw_returns,
                ledger=raw_ledger,
                regimes=self.prepared.regimes,
                policy=raw_policy,
            )
            raw_summary.insert(0, "model_name", self.model_name)
            raw_summary["lookback_days"] = lookback_days
            raw_summary["hold_minutes"] = hold * 15
            raw_rows.append(raw_summary)

            ledger, per_bar = simulate_policy(
                bars=self.prepared.bars,
                execution=self.prepared.minute,
                prediction_frame=prediction,
                start=FORWARD_START,
                end=FORWARD_END,
                resolution="1m",
                tau=float(policy["tau"]),
                tp_bps=int(policy["tp_bps"]),
                sl_bps=int(policy["sl_bps"]),
                max_hold=hold,
                fee_bps=self.fee_bps,
            )
            frozen = {
                "objective": "baseline",
                "width_bps": width,
                "candidate_id": self.candidate_id,
                "policy_id": int(policy["policy_id"]),
                "tau": float(policy["tau"]),
                "tp_bps": int(policy["tp_bps"]),
                "sl_bps": int(policy["sl_bps"]),
                "max_hold": hold,
                "fit_id": fit_id,
            }
            monthly, quarterly, summary = summarize_forward_evidence(
                per_bar=per_bar,
                ledger=ledger,
                regimes=self.prepared.regimes,
                policy=frozen,
            )
            for frame in (monthly, quarterly, summary):
                frame.insert(0, "model_name", self.model_name)
                frame["lookback_days"] = lookback_days
            monthly_frames.append(monthly)
            quarterly_frames.append(quarterly)
            summary_frames.append(summary)
            _atomic_parquet(
                ledger, evidence_root / f"w{width}_lb{lookback_days}_ledger.parquet"
            )
            _atomic_parquet(
                per_bar.rename("net_return").rename_axis("timestamp").reset_index(),
                evidence_root / f"w{width}_lb{lookback_days}_per_bar.parquet",
            )

        artifacts = {
            "raw_forward_summary.parquet": pd.concat(raw_rows, ignore_index=True),
            "forward_monthly.parquet": pd.concat(monthly_frames, ignore_index=True),
            "forward_quarterly.parquet": pd.concat(quarterly_frames, ignore_index=True),
            "forward_summary.parquet": pd.concat(summary_frames, ignore_index=True),
        }
        validate_frozen_policy_rows(artifacts["forward_monthly.parquet"])
        for filename, frame in artifacts.items():
            _atomic_parquet(frame, self.output_root / filename)
        counts = validate_model_artifacts(self.output_root, model_name=self.model_name)
        manifest = {
            "protocol_version": "baseline-model-zoo-1m-monthly-h1-180d-fixed15-v4",
            "model_name": self.model_name,
            "candidate_id": self.candidate_id,
            "candidate_params": self.candidate_params,
            "sentiment": self.sentiment_mode,
            "training_history_days": 180,
            "hold_selection": "pre-declared fixed 15-minute hold; 2024 five-fold OOF control only",
            "h1_calibration": (
                "monthly walk-forward; fixed 180-day history and 33 threshold/TP/SL policies"
            ),
            "forward_fit": "2025-07-01 with fixed 180-day history; frozen to 2026-04-01",
            "lockbox_start": FORWARD_END.isoformat(),
            "sealed_lockbox": True,
            "feature_columns": list(self.prepared.feature_columns),
            "artifact_rows": counts,
        }
        _atomic_json(manifest, self.output_root / "manifest.json")
        result = {
            "status": "complete",
            "model_name": self.model_name,
            "sentiment": self.sentiment_mode,
            "fits": self.fits,
            "cache_hits": self.cache_hits,
            "artifact_rows": counts,
        }
        _atomic_json(result, self.output_root / "result.json")
        return result
