"""Resumable economic runner for one CatBoost execution-resolution arm."""
from __future__ import annotations

import json
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.economics import economics_summary
from experiments.catboost_execution_resolution import (
    EXPECTED_ARM_ROWS,
    PartitionedIntrabarStore,
    assert_arm_artifact_counts,
    execution_policy_fingerprint,
    policy_choices,
    select_economic_candidates,
    write_run_state,
)
from experiments.catboost_matched_ablation import (
    CALIBRATION_END,
    FORWARD_END,
    REGIMES,
    SELECTION_END,
    SELECTION_START,
    economic_ranking_key,
    robust_f1_score,
    robust_score,
    validate_fold_regime_counts,
)
from experiments.run_catboost_matched_ablation import (
    CACHE_ROOT as MATCHED_CACHE_ROOT,
    PreparedData,
    _atomic_json,
    _atomic_parquet,
    _folds,
    _macro_f1,
    fit_fold_predictions,
    fit_span_predictions,
    frame_fingerprint,
    load_prediction_cache,
    prediction_cache_fingerprint,
    stage_prediction_cache_fingerprint,
    summarize_forward_evidence,
    validate_frozen_policy_rows,
    write_prediction_cache,
)
from experiments.catboost_execution_scoring import (
    compare_paired_ledgers,
    score_continuous_policy_grid,
    simulate_policy,
)
from models.zoo import MODELS


def _policy_metadata_path(path: Path) -> Path:
    return path.with_suffix(".json")


def _write_policy_grid(path: Path, frame: pd.DataFrame, fingerprint: str) -> None:
    _atomic_parquet(frame, path)
    _atomic_json(
        {
            "fingerprint": fingerprint,
            "rows": len(frame),
            "columns": list(frame.columns),
            "content_fingerprint": frame_fingerprint(frame),
        },
        _policy_metadata_path(path),
    )


def _load_policy_grid(path: Path, fingerprint: str) -> pd.DataFrame | None:
    metadata_path = _policy_metadata_path(path)
    if not path.exists() or not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame = pd.read_parquet(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        metadata.get("fingerprint") != fingerprint
        or len(frame) != len(policy_choices())
        or list(frame.columns) != metadata.get("columns")
        or frame_fingerprint(frame) != metadata.get("content_fingerprint")
    ):
        return None
    return frame


def _fold_policy_grid(
    *,
    width_bps: int,
    candidate_id: int,
    fold_frames: Sequence[pd.DataFrame],
    folds: Sequence[Mapping[str, Any]],
    execution_frames: Sequence[pd.DataFrame],
    prepared: PreparedData,
    resolution: str,
    fee_bps: float,
) -> pd.DataFrame:
    rows = []
    for policy_id, (tau, geometry) in enumerate(policy_choices()):
        tp_bps, sl_bps, max_hold = geometry
        ledgers, returns, segment_nets = [], [], []
        for fold, prediction_frame, execution in zip(
            folds, fold_frames, execution_frames
        ):
            start = pd.Timestamp(fold["test_start"])
            end = pd.Timestamp(fold["test_end"])
            ledger, per_bar = simulate_policy(
                bars=prepared.bars,
                execution=execution,
                prediction_frame=prediction_frame,
                start=start,
                end=end,
                resolution=resolution,
                tau=tau,
                tp_bps=tp_bps,
                sl_bps=sl_bps,
                max_hold=max_hold,
                fee_bps=fee_bps,
            )
            ledgers.append(ledger)
            returns.append(per_bar)
            segment_nets.append(float(ledger["net_return"].sum()))
        ledger = pd.concat(ledgers, ignore_index=True)
        per_bar = pd.concat(returns).sort_index()
        summary = economics_summary(per_bar)
        bar_regimes = prepared.regimes.reindex(per_bar.index)
        regime_sortino = {
            regime: float(
                economics_summary(per_bar.loc[bar_regimes == regime])["sortino"]
            )
            for regime in REGIMES
        }
        ambiguous = int(ledger["ambiguous_touch"].sum())
        row = {
            "stage": "selection",
            "resolution": resolution,
            "width_bps": int(width_bps),
            "candidate_id": int(candidate_id),
            "policy_id": int(policy_id),
            "tau": float(tau),
            "tp_bps": int(tp_bps),
            "sl_bps": int(sl_bps),
            "max_hold": int(max_hold),
            "trades": int(len(ledger)),
            "pooled_gross": float(ledger["gross_return"].sum()),
            "pooled_net": float(ledger["net_return"].sum()),
            "pooled_sortino": float(summary["sortino"]),
            "pooled_sharpe": float(summary["sharpe"]),
            "positive_segments": int(sum(value > 0.0 for value in segment_nets)),
            "n_long": int((ledger["side"] == 1).sum()),
            "n_short": int((ledger["side"] == -1).sum()),
            "long_net": float(ledger.loc[ledger["side"] == 1, "net_return"].sum()),
            "short_net": float(ledger.loc[ledger["side"] == -1, "net_return"].sum()),
            "bull_sortino": regime_sortino["bull"],
            "sideways_sortino": regime_sortino["sideways"],
            "bear_sortino": regime_sortino["bear"],
            "ambiguous_exits": ambiguous,
            "ambiguous_share": float(ambiguous / len(ledger)) if len(ledger) else 0.0,
        }
        row["robust_score"] = robust_score(
            pooled_sortino=row["pooled_sortino"],
            pooled_sharpe=row["pooled_sharpe"],
            bull_sortino=row["bull_sortino"],
            sideways_sortino=row["sideways_sortino"],
            bear_sortino=row["bear_sortino"],
        )
        rows.append(row)
    return pd.DataFrame(rows)


class ExecutionResolutionRunner:
    """Run selection for one execution arm while sharing CatBoost predictions."""

    def __init__(
        self,
        *,
        output_root: Path,
        store: PartitionedIntrabarStore,
        prepared: PreparedData,
        prediction_root: Path = MATCHED_CACHE_ROOT,
        widths: Sequence[int] = (55, 65, 75),
        candidates: Sequence[Mapping[str, Any]],
        candidate_ids: Sequence[int] | None = None,
        model_factory: Callable = MODELS["catboost_balanced"],
        model_name: str | None = None,
        fee_bps: float = 5.0,
        smoke: bool = False,
        fold_limit: int = 5,
        stage1_only: bool = False,
    ) -> None:
        self.output_root = Path(output_root)
        self.prediction_root = Path(prediction_root)
        self.store = store
        self.prepared = prepared
        self.widths = tuple(int(width) for width in widths)
        self.candidates = tuple(dict(candidate) for candidate in candidates)
        self.candidate_ids = (
            tuple(range(len(self.candidates)))
            if candidate_ids is None
            else tuple(int(candidate) for candidate in candidate_ids)
        )
        if len(self.candidate_ids) != len(self.candidates):
            raise ValueError("candidate IDs and candidate parameters must align")
        if not 1 <= fold_limit <= 5:
            raise ValueError("fold_limit must be between 1 and 5")
        self._candidate_by_id = dict(zip(self.candidate_ids, self.candidates))
        self.model_factory = model_factory
        self.model_name = None if model_name is None else str(model_name)
        self.fee_bps = float(fee_bps)
        self.smoke = bool(smoke)
        self.fold_limit = int(fold_limit)
        self.stage1_only = bool(stage1_only)
        self.fits = 0
        self.cache_hits = 0
        self.policy_cache_hits = 0

    def _prediction_path(
        self, width: int, candidate_id: int, fold_id: int, fingerprint: str
    ) -> Path:
        return self.prediction_root / "predictions" / (
            f"w{width}_candidate_{candidate_id:02d}_fold_{fold_id:02d}_"
            f"{fingerprint}.parquet"
        )

    def _policy_path(
        self, width: int, candidate_id: int, fingerprint: str
    ) -> Path:
        return self.output_root / "policies" / "selection" / (
            f"w{width}_candidate_{candidate_id:02d}_{fingerprint}.parquet"
        )
    def _stage_prediction_path(
        self, width: int, candidate_id: int, fingerprint: str
    ) -> Path:
        return self.prediction_root / "stage_predictions" / "frozen_post_selection" / (
            f"w{width}_candidate_{candidate_id:02d}_{fingerprint}.parquet"
        )

    def _fit_or_load_frozen_predictions(
        self,
        width: int,
        candidate_id: int,
        X: pd.DataFrame,
        y: pd.Series,
        regimes: pd.Series,
    ) -> tuple[pd.DataFrame, str]:
        params = self._candidate_by_id[candidate_id]
        fingerprint = stage_prediction_cache_fingerprint(
            stage="frozen_post_selection",
            width_bps=width,
            candidate_params=params,
            X=X,
            y=y,
            regimes=regimes,
            train_end=SELECTION_END,
            test_start=SELECTION_END,
            test_end=FORWARD_END,
            data_fingerprint=self.prepared.m15_fingerprint,
            model_name=self.model_name,
        )
        path = self._stage_prediction_path(width, candidate_id, fingerprint)
        frame = load_prediction_cache(path, fingerprint)
        if frame is None:
            frame = fit_span_predictions(
                X=X,
                y=y,
                regimes=regimes,
                train_end=SELECTION_END,
                test_start=SELECTION_END,
                test_end=FORWARD_END,
                width_bps=width,
                candidate_id=candidate_id,
                candidate_params=params,
                model_factory=self.model_factory,
                fit_id=f"frozen-w{width}-c{candidate_id:02d}-{fingerprint[:12]}",
            )
            write_prediction_cache(path, frame, fingerprint)
            self.fits += 1
        else:
            self.cache_hits += 1
        return frame, fingerprint

    @staticmethod
    def _classification_row(
        *,
        width: int,
        candidate_id: int,
        fold_frames: Sequence[pd.DataFrame],
        regimes: pd.Series,
    ) -> dict[str, Any]:
        fold_regime_scores, fold_overall = [], []
        pooled = pd.concat(fold_frames, ignore_index=True).set_index("timestamp")
        for frame in fold_frames:
            indexed = frame.set_index("timestamp")
            fold_regimes = regimes.reindex(indexed.index)
            fold_overall.append(_macro_f1(indexed["y_true"], indexed["pred"]))
            fold_regime_scores.append(
                [
                    _macro_f1(
                        indexed.loc[fold_regimes == regime, "y_true"],
                        indexed.loc[fold_regimes == regime, "pred"],
                    )
                    for regime in REGIMES
                ]
            )
        pooled_regimes = regimes.reindex(pooled.index)
        row = {
            "width_bps": int(width),
            "candidate_id": int(candidate_id),
            "overall_f1": float(np.mean(fold_overall)),
        }
        for regime in REGIMES:
            row[f"{regime}_f1"] = _macro_f1(
                pooled.loc[pooled_regimes == regime, "y_true"],
                pooled.loc[pooled_regimes == regime, "pred"],
            )
        row["robust_f1"] = (
            robust_f1_score(fold_regime_scores)
            if len(fold_regime_scores) == 5
            else float(np.mean([min(scores) for scores in fold_regime_scores]))
        )
        return row

    def _run_stage1(self) -> dict[str, Any]:
        classification_rows, policy_frames = [], []
        for width in self.widths:
            X_all, y_all = self.prepared.features[width]
            selection = (
                (X_all.index >= SELECTION_START) & (X_all.index < SELECTION_END)
            )
            X = X_all.loc[selection]
            y = y_all.reindex(X.index)
            regimes = self.prepared.regimes.reindex(X.index)
            known = regimes.isin(REGIMES)
            X, y, regimes = X.loc[known], y.loc[known], regimes.loc[known]
            folds = _folds(X.index)[: self.fold_limit]
            if not self.smoke:
                validate_fold_regime_counts(
                    [
                        regimes.iloc[list(fold["test_positions"])]
                        .value_counts()
                        .to_dict()
                        for fold in folds
                    ]
                )
            execution_frames = [
                self.store.load_span(
                    pd.Timestamp(fold["test_start"]),
                    pd.Timestamp(fold["test_end"]),
                )
                for fold in folds
            ]
            for candidate_id, candidate_params in zip(
                self.candidate_ids, self.candidates
            ):
                fold_frames, prediction_fingerprints = [], []
                for fold in folds:
                    fingerprint = prediction_cache_fingerprint(
                        width_bps=width,
                        candidate_params=candidate_params,
                        fold_metadata=fold,
                        X=X,
                        y=y,
                        regimes=regimes,
                        data_fingerprint=self.prepared.m15_fingerprint,
                        model_name=self.model_name,
                    )
                    path = self._prediction_path(
                        width, candidate_id, int(fold["fold_id"]), fingerprint
                    )
                    frame = load_prediction_cache(path, fingerprint)
                    if frame is None:
                        frame = fit_fold_predictions(
                            X=X,
                            y=y,
                            regimes=regimes,
                            fold_metadata=fold,
                            width_bps=width,
                            candidate_id=candidate_id,
                            candidate_params=candidate_params,
                            model_factory=self.model_factory,
                            refit_id=(
                                f"w{width}-c{candidate_id:02d}-"
                                f"f{int(fold['fold_id']):02d}-{fingerprint[:12]}"
                            ),
                        )
                        write_prediction_cache(path, frame, fingerprint)
                        self.fits += 1
                    else:
                        self.cache_hits += 1
                    fold_frames.append(frame)
                    prediction_fingerprints.append(fingerprint)
                classification_rows.append(
                    self._classification_row(
                        width=width,
                        candidate_id=candidate_id,
                        fold_frames=fold_frames,
                        regimes=self.prepared.regimes,
                    )
                )
                policy_fingerprint = execution_policy_fingerprint(
                    stage="selection",
                    width_bps=width,
                    candidate_id=candidate_id,
                    prediction_fingerprints=prediction_fingerprints,
                    m15_fingerprint=self.prepared.m15_fingerprint,
                    resolution=self.store.resolution,
                    execution_data_fingerprint=self.store.data_fingerprint,
                    fee_bps=self.fee_bps,
                    model_name=self.model_name,
                )
                policy_path = self._policy_path(
                    width, candidate_id, policy_fingerprint
                )
                policy_grid = _load_policy_grid(policy_path, policy_fingerprint)
                if policy_grid is None:
                    policy_grid = _fold_policy_grid(
                        width_bps=width,
                        candidate_id=candidate_id,
                        fold_frames=fold_frames,
                        folds=folds,
                        execution_frames=execution_frames,
                        prepared=self.prepared,
                        resolution=self.store.resolution,
                        fee_bps=self.fee_bps,
                    )
                    _write_policy_grid(
                        policy_path, policy_grid, policy_fingerprint
                    )
                else:
                    self.policy_cache_hits += 1
                policy_frames.append(policy_grid)

        classification = pd.DataFrame(classification_rows)
        policy_grid = pd.concat(policy_frames, ignore_index=True)
        winner_rows = []
        for width in self.widths:
            for candidate_id in self.candidate_ids:
                choices = policy_grid.loc[
                    (policy_grid["width_bps"] == width)
                    & (policy_grid["candidate_id"] == candidate_id)
                ]
                winner_rows.append(
                    min(
                        choices.to_dict(orient="records"),
                        key=lambda row: economic_ranking_key(
                            row, n_segments=self.fold_limit
                        ),
                    )
                )
        winners = pd.DataFrame(winner_rows)
        selected = select_economic_candidates(winners, widths=self.widths)
        artifacts = {
            "classification_2024.parquet": classification,
            "economic_policy_grid_2024.parquet": policy_grid,
            "economic_candidate_winners_2024.parquet": winners,
            "selected_candidates_2024.parquet": selected,
        }
        for name, frame in artifacts.items():
            _atomic_parquet(frame, self.output_root / name)
        return {
            "classification_rows": len(classification),
            "policy_rows": len(policy_grid),
            "winner_rows": len(winners),
            "selected_rows": len(selected),
        }

    def _continuous_policy_path(
        self, stage: str, width: int, candidate_id: int, fingerprint: str
    ) -> Path:
        return self.output_root / "policies" / stage / (
            f"w{width}_candidate_{candidate_id:02d}_{fingerprint}.parquet"
        )

    def _run_later_stages(self, selected_candidates: pd.DataFrame) -> dict[str, Any]:
        selected_rows = selected_candidates.to_dict(orient="records")
        frozen_predictions: dict[tuple[int, int], pd.DataFrame] = {}
        frozen_fingerprints: dict[tuple[int, int], str] = {}
        for selected in selected_rows:
            width = int(selected["width_bps"])
            candidate_id = int(selected["candidate_id"])
            X, y = self.prepared.features[width]
            prediction, fingerprint = self._fit_or_load_frozen_predictions(
                width,
                candidate_id,
                X,
                y.reindex(X.index),
                self.prepared.regimes.reindex(X.index),
            )
            frozen_predictions[(width, candidate_id)] = prediction
            frozen_fingerprints[(width, candidate_id)] = fingerprint

        calibration_execution = self.store.load_span(SELECTION_END, CALIBRATION_END)
        calibration_grids, selected_policy_rows = [], []
        for selected in selected_rows:
            width = int(selected["width_bps"])
            candidate_id = int(selected["candidate_id"])
            prediction = frozen_predictions[(width, candidate_id)]
            fingerprint = execution_policy_fingerprint(
                stage="calibration",
                width_bps=width,
                candidate_id=candidate_id,
                prediction_fingerprints=[frozen_fingerprints[(width, candidate_id)]],
                m15_fingerprint=self.prepared.m15_fingerprint,
                resolution=self.store.resolution,
                execution_data_fingerprint=self.store.data_fingerprint,
                fee_bps=self.fee_bps,
                model_name=self.model_name,
            )
            path = self._continuous_policy_path(
                "calibration", width, candidate_id, fingerprint
            )
            grid = _load_policy_grid(path, fingerprint)
            if grid is None:
                grid = score_continuous_policy_grid(
                    stage="calibration",
                    width_bps=width,
                    candidate_id=candidate_id,
                    prediction_frame=prediction,
                    bars=self.prepared.bars,
                    execution=calibration_execution,
                    regimes=self.prepared.regimes,
                    start=SELECTION_END,
                    end=CALIBRATION_END,
                    resolution=self.store.resolution,
                    fee_bps=self.fee_bps,
                )
                grid.insert(0, "objective", "economic")
                grid["fit_id"] = str(prediction["refit_id"].iloc[0])
                _write_policy_grid(path, grid, fingerprint)
            else:
                self.policy_cache_hits += 1
            calibration_grids.append(grid)
            selected_policy_rows.append(
                min(
                    grid.to_dict(orient="records"),
                    key=lambda row: economic_ranking_key(row, n_segments=6),
                )
            )
        calibration_grid = pd.concat(calibration_grids, ignore_index=True)
        selected_policies = pd.DataFrame(selected_policy_rows)
        _atomic_parquet(
            calibration_grid,
            self.output_root / "calibration_policy_grid_2025h1.parquet",
        )
        _atomic_parquet(
            selected_policies,
            self.output_root / "selected_policies_2025h1.parquet",
        )
        del calibration_execution

        forward_execution = self.store.load_span(CALIBRATION_END, FORWARD_END)
        monthly_frames, quarterly_frames, summary_frames = [], [], []
        evidence_root = self.output_root / "forward_evidence"
        for policy in selected_policies.to_dict(orient="records"):
            width = int(policy["width_bps"])
            candidate_id = int(policy["candidate_id"])
            prediction = frozen_predictions[(width, candidate_id)]
            ledger, per_bar = simulate_policy(
                bars=self.prepared.bars,
                execution=forward_execution,
                prediction_frame=prediction,
                start=CALIBRATION_END,
                end=FORWARD_END,
                resolution=self.store.resolution,
                tau=float(policy["tau"]),
                tp_bps=int(policy["tp_bps"]),
                sl_bps=int(policy["sl_bps"]),
                max_hold=int(policy["max_hold"]),
                fee_bps=self.fee_bps,
            )
            slug = f"economic_w{width}"
            _atomic_parquet(ledger, evidence_root / f"{slug}_ledger.parquet")
            _atomic_parquet(
                per_bar.rename("net_return").rename_axis("timestamp").reset_index(),
                evidence_root / f"{slug}_per_bar.parquet",
            )
            frozen_policy = {
                "objective": "economic",
                "width_bps": width,
                "candidate_id": candidate_id,
                "policy_id": int(policy["policy_id"]),
                "tau": float(policy["tau"]),
                "tp_bps": int(policy["tp_bps"]),
                "sl_bps": int(policy["sl_bps"]),
                "max_hold": int(policy["max_hold"]),
                "fit_id": str(prediction["refit_id"].iloc[0]),
            }
            monthly, quarterly, summary = summarize_forward_evidence(
                per_bar=per_bar,
                ledger=ledger,
                regimes=self.prepared.regimes,
                policy=frozen_policy,
            )
            for frame in (monthly, quarterly, summary):
                frame.insert(1, "resolution", self.store.resolution)
            ambiguous = int(ledger["ambiguous_touch"].sum())
            summary["ambiguous_exits"] = ambiguous
            summary["ambiguous_share"] = (
                float(ambiguous / len(ledger)) if len(ledger) else 0.0
            )
            monthly_frames.append(monthly)
            quarterly_frames.append(quarterly)
            summary_frames.append(summary)

        forward_monthly = pd.concat(monthly_frames, ignore_index=True)
        forward_quarterly = pd.concat(quarterly_frames, ignore_index=True)
        forward_summary = pd.concat(summary_frames, ignore_index=True)
        validate_frozen_policy_rows(forward_monthly)
        for name, frame in {
            "forward_monthly.parquet": forward_monthly,
            "forward_quarterly.parquet": forward_quarterly,
            "forward_summary.parquet": forward_summary,
        }.items():
            _atomic_parquet(frame, self.output_root / name)
        manifest = {
            "protocol_version": "catboost-execution-resolution-v1",
            "resolution": self.store.resolution,
            "execution_data_fingerprint": self.store.data_fingerprint,
            "selection_start": SELECTION_START,
            "selection_end_exclusive": SELECTION_END,
            "calibration_end_exclusive": CALIBRATION_END,
            "forward_end_exclusive": FORWARD_END,
            "model_training_end_exclusive": SELECTION_END,
            "frozen_prediction_fingerprints": sorted(frozen_fingerprints.values()),
            "frozen_fit_ids": sorted(
                {str(frame["refit_id"].iloc[0]) for frame in frozen_predictions.values()}
            ),
            "sealed_lockbox": True,
            "forward_execution_max_timestamp": forward_execution.index.max(),
        }
        if len(forward_execution) and forward_execution.index.max() >= FORWARD_END:
            raise AssertionError("sealed boundary reached by forward execution data")
        _atomic_json(manifest, self.output_root / "manifest.json")
        return {
            "calibration_policy_grid_2025h1": len(calibration_grid),
            "selected_policies_2025h1": len(selected_policies),
            "forward_monthly": len(forward_monthly),
            "forward_quarterly": len(forward_quarterly),
            "forward_summary": len(forward_summary),
        }
    def run(self) -> dict[str, Any]:
        state_path = self.output_root / "run_state.json"
        write_run_state(
            state_path,
            status="running",
            detail={"resolution": self.store.resolution, "stage": "selection"},
        )
        try:
            stage1 = self._run_stage1()
            later: dict[str, Any] = {}
            if not self.stage1_only:
                write_run_state(
                    state_path,
                    status="running",
                    detail={"resolution": self.store.resolution, "stage": "calibration_forward"},
                )
                selected = pd.read_parquet(
                    self.output_root / "selected_candidates_2024.parquet"
                )
                later = self._run_later_stages(selected)
            if (
                not self.smoke
                and self.fold_limit == 5
                and tuple(self.widths) == (55, 65, 75)
                and len(self.candidates) == 15
            ):
                assert_arm_artifact_counts(
                    {
                        "economic_policy_grid_2024": stage1["policy_rows"],
                        "economic_candidate_winners_2024": stage1["winner_rows"],
                        "selected_candidates_2024": stage1["selected_rows"],
                        **later,
                    }
                )
            result = {
                "status": "stage_1_complete" if self.stage1_only else "complete",
                "resolution": self.store.resolution,
                "model_name": self.model_name,
                "fold_count": self.fold_limit,
                "candidate_count": len(self.candidates),
                "fits": self.fits,
                "prediction_cache_hits": self.cache_hits,
                "policy_cache_hits": self.policy_cache_hits,
                **stage1,
                **later,
            }
            _atomic_json(result, self.output_root / "result.json")
            write_run_state(state_path, status="complete", detail=result)
            return result
        except Exception:
            write_run_state(
                state_path,
                status="failed",
                detail={"traceback": traceback.format_exc()},
            )
            raise


def run_paired_replay(
    runners: Mapping[str, ExecutionResolutionRunner],
    *,
    output_root: Path,
) -> dict[str, int]:
    """Replay each arm's selected policy on both resolutions with fixed signals."""
    if set(runners) != {"1m", "1s"}:
        raise ValueError("paired replay requires exactly the 1m and 1s runners")
    if len({runner.prepared.m15_fingerprint for runner in runners.values()}) != 1:
        raise ValueError("paired replay runners must share identical M15 data")
    output_root = Path(output_root)
    execution = {
        resolution: runner.store.load_span(CALIBRATION_END, FORWARD_END)
        for resolution, runner in runners.items()
    }
    metrics_rows, paired_rows, outcome_frames, transition_frames = [], [], [], []
    evidence_root = output_root / "paired_evidence"
    for source_resolution, source_runner in runners.items():
        selected = pd.read_parquet(
            source_runner.output_root / "selected_policies_2025h1.parquet"
        ).sort_values("width_bps")
        for policy in selected.to_dict(orient="records"):
            width = int(policy["width_bps"])
            candidate_id = int(policy["candidate_id"])
            X, y = source_runner.prepared.features[width]
            prediction, _ = source_runner._fit_or_load_frozen_predictions(
                width,
                candidate_id,
                X,
                y.reindex(X.index),
                source_runner.prepared.regimes.reindex(X.index),
            )
            ledgers: dict[str, pd.DataFrame] = {}
            metric_by_target: dict[str, dict[str, Any]] = {}
            for target_resolution in ("1m", "1s"):
                ledger, per_bar = simulate_policy(
                    bars=source_runner.prepared.bars,
                    execution=execution[target_resolution],
                    prediction_frame=prediction,
                    start=CALIBRATION_END,
                    end=FORWARD_END,
                    resolution=target_resolution,
                    tau=float(policy["tau"]),
                    tp_bps=int(policy["tp_bps"]),
                    sl_bps=int(policy["sl_bps"]),
                    max_hold=int(policy["max_hold"]),
                    fee_bps=source_runner.fee_bps,
                )
                ledgers[target_resolution] = ledger
                frozen = {
                    "objective": "economic",
                    "width_bps": width,
                    "candidate_id": candidate_id,
                    "policy_id": int(policy["policy_id"]),
                    "tau": float(policy["tau"]),
                    "tp_bps": int(policy["tp_bps"]),
                    "sl_bps": int(policy["sl_bps"]),
                    "max_hold": int(policy["max_hold"]),
                    "fit_id": str(prediction["refit_id"].iloc[0]),
                }
                _, _, summary = summarize_forward_evidence(
                    per_bar=per_bar,
                    ledger=ledger,
                    regimes=source_runner.prepared.regimes,
                    policy=frozen,
                )
                row = summary.iloc[0].to_dict()
                row.update(
                    {
                        "source_resolution": source_resolution,
                        "target_resolution": target_resolution,
                        "ambiguous_exits": int(ledger["ambiguous_touch"].sum()),
                    }
                )
                metric_by_target[target_resolution] = row
                metrics_rows.append(row)
                _atomic_parquet(
                    ledger,
                    evidence_root
                    / f"source_{source_resolution}_w{width}_target_{target_resolution}_ledger.parquet",
                )


            comparison, outcomes, transitions = compare_paired_ledgers(
                ledgers["1m"], ledgers["1s"]
            )
            common = {
                "source_resolution": source_resolution,
                "width_bps": width,
                "candidate_id": candidate_id,
                "policy_id": int(policy["policy_id"]),
                "tau": float(policy["tau"]),
                "tp_bps": int(policy["tp_bps"]),
                "sl_bps": int(policy["sl_bps"]),
                "max_hold": int(policy["max_hold"]),
            }
            one_minute = metric_by_target["1m"]
            one_second = metric_by_target["1s"]
            paired_rows.append(
                {
                    **common,
                    **comparison,
                    "net_return_1m": float(one_minute["net_return"]),
                    "net_return_1s": float(one_second["net_return"]),
                    "net_delta_1s_minus_1m": float(
                        one_second["net_return"] - one_minute["net_return"]
                    ),
                    "sortino_1m": float(one_minute["sortino"]),
                    "sortino_1s": float(one_second["sortino"]),
                    "sharpe_1m": float(one_minute["sharpe"]),
                    "sharpe_1s": float(one_second["sharpe"]),
                    "trades_1m": int(one_minute["trades"]),
                    "trades_1s": int(one_second["trades"]),
                    "ambiguous_exits_1m": int(one_minute["ambiguous_exits"]),
                    "ambiguous_exits_1s": int(one_second["ambiguous_exits"]),
                }
            )
            outcomes = outcomes.assign(**common)
            transitions = transitions.assign(**common)
            outcome_frames.append(outcomes)
            transition_frames.append(transitions)

    replay_metrics = pd.DataFrame(metrics_rows)
    paired_summary = pd.DataFrame(paired_rows)
    paired_outcomes = pd.concat(outcome_frames, ignore_index=True)
    paired_transitions = pd.concat(transition_frames, ignore_index=True)
    for name, frame in {
        "paired_replay_metrics.parquet": replay_metrics,
        "paired_replay_summary.parquet": paired_summary,
        "paired_replay_outcomes.parquet": paired_outcomes,
        "paired_exit_transitions.parquet": paired_transitions,
    }.items():
        _atomic_parquet(frame, output_root / name)
    return {
        "paired_metric_rows": len(replay_metrics),
        "paired_summary_rows": len(paired_summary),
        "paired_outcome_rows": len(paired_outcomes),
        "paired_transition_rows": len(paired_transitions),
    }
