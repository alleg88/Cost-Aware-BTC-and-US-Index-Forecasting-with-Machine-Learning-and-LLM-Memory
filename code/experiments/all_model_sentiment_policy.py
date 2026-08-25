"""Policy-only all-model sentiment calibration for Notebook 02e.

Notebook 02d already owns 2024 diagnostics and the raw July-fit predictions.
This runner adds only six causal H1 prediction blocks, selects one of 33
threshold/TP/SL policies per Model/Arm/DZ, and replays that policy on the exact
Notebook 02d frozen-forward prediction.
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

from experiments.all_model_sentiment_raw import (
    ARMS,
    DEFAULT_ROOT as RAW_ROOT,
    prepare_arm,
    validate_raw_model_artifacts,
)
from experiments.baseline_model_zoo_1m import (
    BASELINE_CANDIDATE_ID,
    FORWARD_END,
    FORWARD_START,
    LOOKBACK_DAYS,
    MODEL_NAMES,
    WIDTHS,
    BaselineModelRunner,
    fit_plan,
    select_h1_policy_rows,
)
from experiments.catboost_execution_resolution import write_run_state
from experiments.catboost_execution_scoring import simulate_policy
from experiments.run_catboost_matched_ablation import (
    PreparedData,
    _atomic_json,
    _atomic_parquet,
    combine_monthly_predictions,
    load_prediction_cache,
    summarize_forward_evidence,
    validate_frozen_policy_rows,
)
from models.zoo import MODELS


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    CODE_ROOT
    / "experiments"
    / "cache"
    / "tuning"
    / "all_model_sentiment_policy_180d_fixed15_monthly_h1_v3"
)
POLICY_FIELDS = ("policy_id", "tau", "tp_bps", "sl_bps", "max_hold")


def expected_combination_count() -> int:
    return len(ARMS) * len(MODEL_NAMES) * len(WIDTHS)


def expected_policy_model_counts(widths: Sequence[int] = WIDTHS) -> dict[str, int]:
    n = len(tuple(widths))
    return {
        "calibration_policy_grid_2025h1.parquet": 33 * n,
        "selected_policies_2025h1.parquet": n,
        "forward_monthly.parquet": 9 * n,
        "forward_quarterly.parquet": 3 * n,
        "forward_summary.parquet": n,
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
        "protocol_version": "all-model-sentiment-policy-only-180d-fixed15-v4",
        "model_name": model_name,
        "sentiment_arm": arm,
        "candidate_id": BASELINE_CANDIDATE_ID,
        "candidate_params": {},
        "widths": [int(width) for width in widths],
        "lookback_days": list(LOOKBACK_DAYS),
        "hold_bars": [1],
        "policy_count_per_model_arm_width": 33,
        "calibration_months": 6,
        "policy_calibration_used": True,
        "one_minute_execution_used": True,
        "reuse_notebook02c_raw_forward": True,
        "rerun_2024_folds": False,
        "smoke": bool(smoke),
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_raw_forward_prediction(
    raw_model_root: Path,
    *,
    width_bps: int,
    model_name: str,
) -> tuple[pd.DataFrame, str]:
    """Load the one validated Notebook 02d July-fit prediction for a DZ."""
    paths = sorted(
        (Path(raw_model_root) / "stage_predictions" / "raw_forward").glob(
            f"w{int(width_bps)}_lb180_candidate_00_*.parquet"
        )
    )
    valid: list[tuple[pd.DataFrame, str]] = []
    for path in paths:
        fingerprint = path.stem.rsplit("_", 1)[-1]
        frame = load_prediction_cache(path, fingerprint)
        if frame is not None:
            valid.append((frame, fingerprint))
    if len(valid) != 1:
        raise ValueError(
            f"{model_name} DZ{width_bps}: expected one current Notebook 02d prediction"
        )
    frame, fingerprint = valid[0]
    if not frame["width_bps"].astype(int).eq(int(width_bps)).all():
        raise ValueError("raw forward prediction dead zone changed")
    if not frame["candidate_id"].astype(int).eq(BASELINE_CANDIDATE_ID).all():
        raise ValueError("raw forward prediction candidate changed")
    if frame["refit_id"].astype(str).nunique() != 1:
        raise ValueError("raw forward model was refitted")
    if not pd.to_datetime(frame["test_start"], utc=True).eq(FORWARD_START).all():
        raise ValueError("raw forward prediction start changed")
    if not pd.to_datetime(frame["test_end"], utc=True).eq(FORWARD_END).all():
        raise ValueError("raw forward prediction end changed")
    if pd.to_datetime(frame["train_end"], utc=True).max() >= FORWARD_START:
        raise ValueError("raw forward fit crossed the forward boundary")
    return frame.sort_values("timestamp"), fingerprint


def validate_policy_model_artifacts(
    root: Path,
    *,
    model_name: str,
    widths: Sequence[int] = WIDTHS,
) -> dict[str, int]:
    expected = expected_policy_model_counts(widths)
    frames: dict[str, pd.DataFrame] = {}
    counts: dict[str, int] = {}
    for filename, rows in expected.items():
        path = Path(root) / filename
        if not path.exists():
            raise FileNotFoundError(f"missing {filename}")
        frame = pd.read_parquet(path)
        if len(frame) != rows:
            raise ValueError(f"{filename} must contain exactly {rows} rows")
        frames[filename] = frame
        counts[filename] = len(frame)
    selected = frames["selected_policies_2025h1.parquet"]
    forward = frames["forward_summary.parquet"]
    expected_widths = {int(width) for width in widths}
    if set(selected["width_bps"].astype(int)) != expected_widths:
        raise ValueError("selected policy dead zones changed")
    if set(forward["width_bps"].astype(int)) != expected_widths:
        raise ValueError("forward policy dead zones changed")
    if not selected["monthly_fit_count"].astype(int).eq(6).all():
        raise ValueError("H1 policy selection must use six monthly fits")
    for field in POLICY_FIELDS:
        left = selected.set_index("width_bps")[field].sort_index()
        right = forward.set_index("width_bps")[field].sort_index()
        if not left.equals(right):
            raise ValueError(f"frozen forward {field} differs from H1 selection")
    if set(forward["model_name"].astype(str)) != {model_name}:
        raise ValueError("forward model name changed")
    if not pd.to_datetime(forward["period_start"], utc=True).eq(FORWARD_START).all():
        raise ValueError("frozen forward start changed")
    if not pd.to_datetime(forward["period_end"], utc=True).eq(FORWARD_END).all():
        raise ValueError("frozen forward end changed")
    validate_frozen_policy_rows(frames["forward_monthly.parquet"])
    return counts


class PolicyOnlyModelRunner:
    """Calibrate H1 policies and reuse Notebook 02d forward predictions."""

    def __init__(
        self,
        *,
        output_root: Path,
        raw_model_root: Path,
        prepared: PreparedData,
        model_name: str,
        model_factory: Callable,
        widths: Sequence[int],
        sentiment_mode: str,
        fee_bps: float = 5.0,
        smoke: bool = False,
    ):
        self.output_root = Path(output_root)
        self.raw_model_root = Path(raw_model_root)
        self.prepared = prepared
        self.model_name = model_name
        self.widths = tuple(int(width) for width in widths)
        self.sentiment_mode = sentiment_mode
        self.fee_bps = float(fee_bps)
        self.smoke = bool(smoke)
        self.core = BaselineModelRunner(
            output_root=self.output_root,
            prepared=prepared,
            model_name=model_name,
            model_factory=model_factory,
            widths=self.widths,
            candidate_id=BASELINE_CANDIDATE_ID,
            candidate_params={},
            sentiment_mode=sentiment_mode,
            fee_bps=fee_bps,
            smoke=False,
        )

    def _validate_raw_source(self) -> None:
        validate_raw_model_artifacts(
            self.raw_model_root, model_name=self.model_name
        )
        manifest = _load_json(self.raw_model_root / "manifest.json")
        if not manifest or manifest.get("sentiment_arm") != self.sentiment_mode:
            raise ValueError("Notebook 02d sentiment identity changed")
        if manifest.get("feature_columns") != list(self.prepared.feature_columns):
            raise ValueError("Notebook 02d feature schema changed")

    def run(self) -> dict[str, Any]:
        self._validate_raw_source()
        grids = []
        for width in self.widths:
            monthly = [
                self.core._fit_stage(
                    width=width,
                    stage=f"calibration_{row['test_start']:%Y_%m}",
                    lookback_days=180,
                    train_end=pd.Timestamp(row["train_end"]),
                    test_start=pd.Timestamp(row["test_start"]),
                    test_end=pd.Timestamp(row["test_end"]),
                )
                for row in fit_plan(width, 180)[:6]
            ]
            calibration = combine_monthly_predictions(monthly)
            grid = self.core._score_h1_policies(
                width=width, hold_bars=1, prediction=calibration
            )
            grid["lookback_days"] = 180
            grids.append(grid)
        calibration_grid = pd.concat(grids, ignore_index=True)
        selected = select_h1_policy_rows(calibration_grid)
        _atomic_parquet(
            calibration_grid,
            self.output_root / "calibration_policy_grid_2025h1.parquet",
        )
        _atomic_parquet(
            selected, self.output_root / "selected_policies_2025h1.parquet"
        )

        monthly_frames, quarterly_frames, summary_frames = [], [], []
        raw_fingerprints: dict[str, str] = {}
        evidence_root = self.output_root / "forward_evidence"
        for policy in selected.to_dict("records"):
            width = int(policy["width_bps"])
            prediction, raw_fingerprint = load_raw_forward_prediction(
                self.raw_model_root,
                width_bps=width,
                model_name=self.model_name,
            )
            raw_fingerprints[str(width)] = raw_fingerprint
            fit_id = str(prediction["refit_id"].iloc[0])
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
                max_hold=int(policy["max_hold"]),
                fee_bps=self.fee_bps,
            )
            frozen = {
                "objective": "baseline",
                "width_bps": width,
                "candidate_id": BASELINE_CANDIDATE_ID,
                "policy_id": int(policy["policy_id"]),
                "tau": float(policy["tau"]),
                "tp_bps": int(policy["tp_bps"]),
                "sl_bps": int(policy["sl_bps"]),
                "max_hold": int(policy["max_hold"]),
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
                frame["lookback_days"] = 180
            monthly_frames.append(monthly)
            quarterly_frames.append(quarterly)
            summary_frames.append(summary)
            _atomic_parquet(
                ledger, evidence_root / f"w{width}_lb180_ledger.parquet"
            )
            _atomic_parquet(
                per_bar.rename("net_return").rename_axis("timestamp").reset_index(),
                evidence_root / f"w{width}_lb180_per_bar.parquet",
            )

        artifacts = {
            "forward_monthly.parquet": pd.concat(monthly_frames, ignore_index=True),
            "forward_quarterly.parquet": pd.concat(quarterly_frames, ignore_index=True),
            "forward_summary.parquet": pd.concat(summary_frames, ignore_index=True),
        }
        for filename, frame in artifacts.items():
            _atomic_parquet(frame, self.output_root / filename)
        counts = validate_policy_model_artifacts(
            self.output_root,
            model_name=self.model_name,
            widths=self.widths,
        )
        manifest = {
            **protocol_manifest(
                self.model_name,
                self.sentiment_mode,
                widths=self.widths,
                smoke=self.smoke,
            ),
            "fee_bps_per_side": self.fee_bps,
            "calibration_start": "2025-01-01T00:00:00+00:00",
            "calibration_end_exclusive": FORWARD_START.isoformat(),
            "forward_start": FORWARD_START.isoformat(),
            "forward_end_exclusive": FORWARD_END.isoformat(),
            "raw_model_root": str(self.raw_model_root),
            "raw_forward_prediction_fingerprints": raw_fingerprints,
            "feature_columns": list(self.prepared.feature_columns),
            "sealed_lockbox": True,
            "artifact_rows": counts,
        }
        _atomic_json(manifest, self.output_root / "manifest.json")
        result = {
            "status": "smoke_complete" if self.smoke else "complete",
            "model_name": self.model_name,
            "sentiment_arm": self.sentiment_mode,
            "fits": self.core.fits,
            "cache_hits": self.core.cache_hits,
            "artifact_rows": counts,
        }
        _atomic_json(result, self.output_root / "result.json")
        return result


def _matching_complete(
    root: Path, *, model_name: str, arm: str
) -> dict[str, Any] | None:
    if _load_json(root / "protocol.json") != protocol_manifest(model_name, arm):
        return None
    result = _load_json(root / "result.json")
    manifest = _load_json(root / "manifest.json")
    if not result or result.get("status") != "complete":
        return None
    if not manifest or manifest.get("sentiment_arm") != arm:
        return None
    try:
        validate_policy_model_artifacts(root, model_name=model_name)
    except (FileNotFoundError, ValueError):
        return None
    return {**result, "resumed": True}


def run_one(
    arm: str,
    model_name: str,
    *,
    output_root: Path,
    raw_root: Path = RAW_ROOT,
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
    runner = PolicyOnlyModelRunner(
        output_root=root,
        raw_model_root=Path(raw_root) / arm / model_name,
        prepared=prepared,
        model_name=model_name,
        model_factory=MODELS[model_name],
        widths=widths,
        sentiment_mode=arm,
        smoke=smoke,
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
    raw_root: Path = RAW_ROOT,
    smoke: bool,
    prepared_by_arm: Mapping[str, PreparedData] | None = None,
    prepare_arm_fn: Callable[[str], PreparedData] | None = None,
) -> dict[str, Any]:
    if prepared_by_arm is None and prepare_arm_fn is None:
        raise ValueError("prepared_by_arm or prepare_arm_fn is required")
    output_root = Path(output_root)
    state_path = output_root / "run_state.json"
    completed: list[str] = []
    active = ""
    _write_state(state_path, status="running", detail={"completed": completed})
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
                    raw_root=raw_root,
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--arm", choices=("all", *ARMS), default="all")
    parser.add_argument("--model", choices=("all", *MODEL_NAMES), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
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
        raw_root=args.raw_root,
        smoke=args.smoke,
        prepare_arm_fn=prepare_arm,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
