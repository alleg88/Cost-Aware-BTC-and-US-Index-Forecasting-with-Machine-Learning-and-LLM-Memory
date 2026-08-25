"""Raw all-model sentiment ablation used by Notebook 02d.

The experiment crosses nine fixed model-family baselines with no sentiment,
matched DeBERTa features and matched LLM features at DZ55/DZ65/DZ75.  It uses
one fixed 15-minute hold and deliberately excludes confidence and TP/SL policy
calibration.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.baseline_model_zoo_1m import (
    BASELINE_CANDIDATE_ID,
    FOLD_COUNT,
    FORWARD_END,
    FORWARD_START,
    LOOKBACK_DAYS,
    WIDTHS,
    BaselineModelRunner,
    _classification_row,
    fit_plan,
)
from experiments.catboost_execution_resolution import write_run_state
from experiments.catboost_matched_ablation import (
    REGIMES,
    SELECTION_END,
    SELECTION_START,
    validate_fold_regime_counts,
)
from experiments.catboost_sentiment_ablation import ARM_SPECS, prepare_arm_data
from experiments.raw_hold_control import MODEL_NAMES, simulate_fixed_hold
from experiments.run_catboost_matched_ablation import (
    PreparedData,
    _atomic_json,
    _atomic_parquet,
    _folds,
    load_prepared_data,
    summarize_forward_evidence,
)
from models.zoo import MODELS


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    CODE_ROOT
    / "experiments"
    / "cache"
    / "tuning"
    / "all_model_sentiment_raw_180d_fixed15_v3"
)
ARMS = ("none", "classic", "llm", "llm_full")
POLICY_COLUMNS = {"policy_id", "tau", "tp_bps", "sl_bps"}


def expected_combination_count() -> int:
    return len(ARMS) * len(MODEL_NAMES) * len(WIDTHS)


def expected_raw_model_counts() -> dict[str, int]:
    return {
        "classification_2024.parquet": len(WIDTHS),
        "raw_forward_summary.parquet": len(WIDTHS),
    }


def protocol_manifest(
    model_name: str,
    arm: str,
    *,
    widths: Sequence[int] = WIDTHS,
    smoke: bool = False,
) -> dict[str, Any]:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unknown model_name: {model_name}")
    if arm not in ARMS:
        raise ValueError(f"unknown sentiment arm: {arm}")
    return {
        "protocol_version": "all-model-sentiment-raw-180d-fixed15-v3",
        "model_name": model_name,
        "sentiment_arm": arm,
        "candidate_id": BASELINE_CANDIDATE_ID,
        "candidate_params": {},
        "widths": [int(width) for width in widths],
        "lookback_days": list(LOOKBACK_DAYS),
        "hold_bars": [1],
        "confidence_threshold": None,
        "tp_bps": None,
        "sl_bps": None,
        "policy_calibration_used": False,
        "one_minute_execution_used": False,
        "smoke": bool(smoke),
    }


def validate_raw_forward(frame: pd.DataFrame, *, model_name: str) -> None:
    forbidden = POLICY_COLUMNS.intersection(frame.columns)
    if forbidden:
        raise ValueError(f"raw forward contains policy columns: {sorted(forbidden)}")
    required = {
        "model_name",
        "width_bps",
        "lookback_days",
        "hold_minutes",
        "period_start",
        "period_end",
        "trades",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"raw forward misses columns: {sorted(missing)}")
    if len(frame) != len(WIDTHS):
        raise ValueError(f"raw forward must contain exactly {len(WIDTHS)} rows")
    if set(frame["model_name"].astype(str)) != {model_name}:
        raise ValueError("raw forward model changed")
    if set(frame["width_bps"].astype(int)) != set(WIDTHS):
        raise ValueError("raw forward dead zones changed")
    if set(frame["lookback_days"].astype(int)) != {180}:
        raise ValueError("raw forward history must be 180 days")
    if set(frame["hold_minutes"].astype(int)) != {15}:
        raise ValueError("raw forward hold must be 15 minutes")
    if not pd.to_datetime(frame["period_start"], utc=True).eq(FORWARD_START).all():
        raise ValueError("raw forward start changed")
    if not pd.to_datetime(frame["period_end"], utc=True).eq(FORWARD_END).all():
        raise ValueError("raw forward end changed")
    if {"n_long", "n_short"}.issubset(frame.columns):
        if not frame["trades"].eq(frame["n_long"] + frame["n_short"]).all():
            raise ValueError("raw forward trade directions do not reconcile")


def validate_raw_model_artifacts(root: Path, *, model_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for filename, expected in expected_raw_model_counts().items():
        path = Path(root) / filename
        if not path.exists():
            raise FileNotFoundError(f"missing {filename}")
        frame = pd.read_parquet(path)
        if len(frame) != expected:
            raise ValueError(f"{filename} must contain exactly {expected} rows")
        counts[filename] = len(frame)
        if filename == "raw_forward_summary.parquet":
            validate_raw_forward(frame, model_name=model_name)
    return counts


class RawSentimentModelRunner(BaselineModelRunner):
    """Run classification diagnostics and one raw frozen-forward replay."""

    def __init__(self, *, sentiment_arm: str, **kwargs: Any):
        if sentiment_arm not in ARMS:
            raise ValueError(f"unknown sentiment arm: {sentiment_arm}")
        super().__init__(**kwargs)
        self.sentiment_arm = sentiment_arm

    def run(self) -> dict[str, Any]:
        classification_rows = []
        for width in self.widths:
            X_all, y_all = self.prepared.features[width]
            selection = (X_all.index >= SELECTION_START) & (X_all.index < SELECTION_END)
            X = X_all.loc[selection]
            y = y_all.reindex(X.index)
            regimes = self.prepared.regimes.reindex(X.index)
            known = regimes.isin(REGIMES)
            X, y, regimes = X.loc[known], y.loc[known], regimes.loc[known]
            folds = _folds(X.index)[:FOLD_COUNT]
            validate_fold_regime_counts(
                [
                    regimes.iloc[list(fold["test_positions"])]
                    .value_counts()
                    .to_dict()
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
        classification = pd.DataFrame(classification_rows)
        classification.insert(1, "sentiment_arm", self.sentiment_arm)
        _atomic_parquet(classification, self.output_root / "classification_2024.parquet")

        if self.smoke:
            result = {
                "status": "smoke_complete",
                "model_name": self.model_name,
                "sentiment_arm": self.sentiment_arm,
                "fits": self.fits,
                "cache_hits": self.cache_hits,
            }
            _atomic_json(result, self.output_root / "result.json")
            return result

        summaries = []
        for width in self.widths:
            plan = fit_plan(width, 180)[-1]
            prediction = self._fit_stage(
                width=width,
                stage="raw_forward",
                lookback_days=180,
                train_end=pd.Timestamp(plan["train_end"]),
                test_start=pd.Timestamp(plan["test_start"]),
                test_end=pd.Timestamp(plan["test_end"]),
            )
            pred = prediction.set_index("timestamp")["pred"].astype(int)
            bars = self.prepared.bars.loc[
                (self.prepared.bars.index >= FORWARD_START)
                & (self.prepared.bars.index < FORWARD_END)
            ]
            ledger, per_bar = simulate_fixed_hold(
                bars, pred, hold_bars=1, fee_bps=self.fee_bps
            )
            placeholder = {
                "objective": "raw_fixed_hold",
                "width_bps": int(width),
                "candidate_id": self.candidate_id,
                "policy_id": -1,
                "tau": 0.0,
                "tp_bps": 0,
                "sl_bps": 0,
                "max_hold": 1,
                "fit_id": str(prediction["refit_id"].iloc[0]),
            }
            _, _, summary = summarize_forward_evidence(
                per_bar=per_bar,
                ledger=ledger,
                regimes=self.prepared.regimes,
                policy=placeholder,
            )
            summary = summary.drop(columns=sorted(POLICY_COLUMNS))
            summary.insert(0, "model_name", self.model_name)
            summary.insert(1, "sentiment_arm", self.sentiment_arm)
            summary["lookback_days"] = 180
            summary["hold_minutes"] = 15
            summaries.append(summary)

        raw_forward = pd.concat(summaries, ignore_index=True)
        validate_raw_forward(raw_forward, model_name=self.model_name)
        _atomic_parquet(raw_forward, self.output_root / "raw_forward_summary.parquet")
        counts = validate_raw_model_artifacts(
            self.output_root, model_name=self.model_name
        )
        manifest = {
            **protocol_manifest(self.model_name, self.sentiment_arm),
            "training_history_days": 180,
            "forward_fit": "2025-07-01; preceding 180 days; no later refit",
            "forward_end_exclusive": FORWARD_END.isoformat(),
            "fee_bps_per_side": self.fee_bps,
            "sealed_lockbox": True,
            "feature_columns": list(self.prepared.feature_columns),
            "artifact_rows": counts,
        }
        _atomic_json(manifest, self.output_root / "manifest.json")
        result = {
            "status": "complete",
            "model_name": self.model_name,
            "sentiment_arm": self.sentiment_arm,
            "fits": self.fits,
            "cache_hits": self.cache_hits,
            "artifact_rows": counts,
        }
        _atomic_json(result, self.output_root / "result.json")
        return result


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _matching_complete(
    root: Path, *, model_name: str, arm: str
) -> dict[str, Any] | None:
    if _load_json(root / "protocol.json") != protocol_manifest(model_name, arm):
        return None
    result = _load_json(root / "result.json")
    if not result or result.get("status") != "complete":
        return None
    try:
        validate_raw_model_artifacts(root, model_name=model_name)
    except (FileNotFoundError, ValueError):
        return None
    return {**result, "resumed": True}


def run_one(
    arm: str,
    model_name: str,
    *,
    output_root: Path,
    prepared: PreparedData,
    smoke: bool,
) -> dict[str, Any]:
    widths = (WIDTHS[0],) if smoke else WIDTHS
    root = Path(output_root) / arm / model_name
    if not smoke:
        complete = _matching_complete(root, model_name=model_name, arm=arm)
        if complete is not None:
            return complete
    _atomic_json(
        protocol_manifest(model_name, arm, widths=widths, smoke=smoke),
        root / "protocol.json",
    )
    runner = RawSentimentModelRunner(
        output_root=root,
        prepared=prepared,
        model_name=model_name,
        model_factory=MODELS[model_name],
        widths=widths,
        candidate_id=BASELINE_CANDIDATE_ID,
        candidate_params={},
        smoke=smoke,
        sentiment_arm=arm,
    )
    return runner.run()


def _write_state(path: Path, *, status: str, detail: dict[str, Any]) -> None:
    for attempt in range(5):
        try:
            write_run_state(path, status=status, detail=detail)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def run_sequence(
    *,
    arms: Sequence[str],
    model_names: Sequence[str],
    output_root: Path,
    smoke: bool,
    prepared_by_arm: Mapping[str, PreparedData] | None = None,
    prepare_arm_fn: Callable[[str], PreparedData] | None = None,
) -> dict[str, Any]:
    if prepared_by_arm is None and prepare_arm_fn is None:
        raise ValueError("prepared_by_arm or prepare_arm_fn is required")
    output_root = Path(output_root)
    state_path = output_root / "run_state.json"
    completed: list[str] = []
    _write_state(state_path, status="running", detail={"completed": completed})
    active = ""
    try:
        for arm in arms:
            if arm not in ARMS:
                raise ValueError(f"unknown sentiment arm: {arm}")
            prepared = (
                prepared_by_arm[arm]
                if prepared_by_arm is not None
                else prepare_arm_fn(arm)  # type: ignore[misc]
            )
            for model_name in model_names:
                active = f"{arm}/{model_name}"
                _write_state(
                    state_path,
                    status="running",
                    detail={"active": active, "completed": completed},
                )
                run_one(
                    arm,
                    model_name,
                    output_root=output_root,
                    prepared=prepared,
                    smoke=smoke,
                )
                completed.append(active)
        result = {"status": "complete", "completed": completed}
        _write_state(state_path, status="complete", detail=result)
        return result
    except Exception:
        _write_state(
            state_path,
            status="failed",
            detail={
                "active": active,
                "completed": completed,
                "traceback": traceback.format_exc(),
            },
        )
        raise


def prepare_arm(arm: str) -> PreparedData:
    if arm == "none":
        return load_prepared_data(widths=WIDTHS, sentiment="none")
    if arm in ARM_SPECS:
        return prepare_arm_data(arm)
    raise ValueError(f"unknown sentiment arm: {arm}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--arm", choices=("all", *ARMS), default="all")
    parser.add_argument("--model", choices=("all", *MODEL_NAMES), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)

    output_root = args.output_root
    if args.smoke and output_root == DEFAULT_ROOT:
        output_root = DEFAULT_ROOT.with_name(DEFAULT_ROOT.name + "_smoke")
    arms = ARMS if args.arm == "all" else (args.arm,)
    models = MODEL_NAMES if args.model == "all" else (args.model,)
    result = run_sequence(
        arms=arms,
        model_names=models,
        output_root=output_root,
        smoke=args.smoke,
        prepare_arm_fn=prepare_arm,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
