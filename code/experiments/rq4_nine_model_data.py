"""Prepare continuous nine-model forecasts for RQ4 from January 2024.

The 2024 forecasts use one fit on the preceding 180 days of 2023 per model.
The six 2025 calibration months and frozen forward forecasts are reused verbatim.
The 2024 period remains development evidence under the later frozen policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from experiments.catboost_matched_ablation import REGIMES, past_regime_labels
from experiments.raw_hold_control import MODEL_NAMES
from experiments.run_catboost_matched_ablation import (
    _atomic_json,
    _atomic_parquet,
    regime_balanced_training_weights,
)
from features.build import (
    FEATURE_COLS,
    ORDERFLOW_FEATURE_COLS,
    POSITIONING_FEATURE_COLS,
    POSITIONING_SOURCE_COLS,
    add_features,
    make_label,
)
from models.zoo import MODELS, _aligned_proba
from experiments.qualified_union import (
    MEMBERS, build_union_frame, combine_union, member_signal, prediction_paths,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
CACHE = CODE_ROOT / "experiments" / "cache" / "reflection_ensemble_v5"
RAW = CODE_ROOT / "experiments/cache/tuning/all_model_sentiment_raw_180d_fixed15_v3/none"
POLICY = CODE_ROOT / "experiments/cache/tuning/all_model_sentiment_policy_180d_fixed15_monthly_h1_v3/none"
FEATURES = tuple(FEATURE_COLS + ORDERFLOW_FEATURE_COLS + POSITIONING_FEATURE_COLS)
WIDTH_BPS = 65
BASELINE_WIDTH_BPS = 55
UNION_SVM_WIDTH_BPS = 75
PROBABILITY_COLUMNS = ("p_short", "p_flat", "p_long")
BAR = pd.Timedelta(minutes=15)
DEVELOPMENT_START = pd.Timestamp("2024-01-01", tz="UTC")
DEVELOPMENT_END = pd.Timestamp("2025-01-01", tz="UTC")
FIT_START = DEVELOPMENT_START - pd.Timedelta(days=180)
FORWARD_START = pd.Timestamp("2025-07-01", tz="UTC")
FORWARD_END = pd.Timestamp("2026-04-01", tz="UTC")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _bounded(path: Path, end: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(path, filters=[
        ("timestamp", ">=", FIT_START - pd.Timedelta(days=8)), ("timestamp", "<", end)
    ])
    frame.index = pd.to_datetime(frame.index, utc=True)
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"source timestamps are not unique and ordered: {path.name}")
    if frame.empty or frame.index.max() >= end:
        raise ValueError(f"source escaped its boundary: {path.name}")
    return frame


def load_development_features() -> tuple[pd.DataFrame, dict[int, pd.Series], pd.Series]:
    history = _bounded(CODE_ROOT / "data/btcusdt_15min_2021_2026.parquet", DEVELOPMENT_START)
    current = _bounded(CODE_ROOT / "data/btcusdt_m15_2024_2025.parquet", DEVELOPMENT_END)
    bars = pd.concat([history, current.loc[current.index >= DEVELOPMENT_START]])
    history_positioning = _bounded(
        CODE_ROOT / "data/btcusdt_positioning_15min_2021_2026.parquet", DEVELOPMENT_START
    )
    current_positioning = _bounded(
        CODE_ROOT / "data/btcusdt_positioning_m15_2024_2026.parquet", DEVELOPMENT_END
    )
    positioning = pd.concat([
        history_positioning.loc[:, POSITIONING_SOURCE_COLS],
        current_positioning.loc[current_positioning.index >= DEVELOPMENT_START, POSITIONING_SOURCE_COLS],
    ])
    if bars.index.has_duplicates or positioning.index.has_duplicates:
        raise ValueError("historical and current source intervals overlap")
    features = add_features(
        bars.join(positioning.loc[:, POSITIONING_SOURCE_COLS].reindex(bars.index))
    )
    X = features.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan)
    next_time = pd.Series(bars.index, index=bars.index).shift(-1)
    exact_next_bar = next_time.eq(pd.Series(bars.index + BAR, index=bars.index))
    labels = {
        width: make_label(bars, threshold_bps=width, horizon=1).where(exact_next_bar, -1)
        for width in (BASELINE_WIDTH_BPS, WIDTH_BPS, UNION_SVM_WIDTH_BPS)
    }
    return X, labels, past_regime_labels(bars["close"])


def training_parameters(model_name: str) -> dict[str, Any]:
    # Bound parallelism without changing the registered statistical settings.
    if model_name in {"random_forest", "xgboost_balanced"}:
        return {"n_jobs": 4}
    if model_name == "catboost_balanced":
        return {"thread_count": 4}
    return {}


def validate_prediction(frame: pd.DataFrame, width: int) -> None:
    required = {
        "timestamp", "y_true", "pred", "confidence", "train_start", "train_end",
        "width_bps", *PROBABILITY_COLUMNS,
    }
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("prediction frame is incomplete")
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    if timestamp.duplicated().any() or not timestamp.is_monotonic_increasing:
        raise ValueError("prediction timestamps must be unique and ordered")
    if timestamp.max() >= FORWARD_END:
        raise ValueError("prediction frame crossed the Q2 boundary")
    if not pd.to_datetime(frame["train_end"], utc=True).lt(timestamp).all():
        raise ValueError("prediction frame contains a noncausal fit")
    if not frame["width_bps"].eq(width).all():
        raise ValueError("prediction label width changed")
    values = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("prediction probabilities are invalid")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("prediction probabilities do not sum to one")


def fit_development_model(
    X: pd.DataFrame,
    y: pd.Series,
    regimes: pd.Series,
    model_name: str,
    width: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = X.index[(X.index >= FIT_START) & (X.index < DEVELOPMENT_START)]
    test = X.index[(X.index >= DEVELOPMENT_START) & (X.index < DEVELOPMENT_END)]
    train = train[y.reindex(train).ge(0) & regimes.reindex(train).isin(REGIMES)]
    test = test[y.reindex(test).ge(0)]
    # Match the one-label tail trim in the existing 180-day stage fits.
    train = train[:-1]
    if len(train) < 100 or len(test) < 100:
        raise ValueError("insufficient rows for the 2024 development fit")
    if train.max() + 2 * BAR >= test.min() + BAR:
        raise ValueError("a training label is unavailable at the first test decision")

    raw_train, raw_test = X.loc[train], X.loc[test]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    fitted = pd.DataFrame(imputer.fit_transform(raw_train), index=train, columns=FEATURES)
    scored = pd.DataFrame(imputer.transform(raw_test), index=test, columns=FEATURES)
    model = MODELS[model_name](training_parameters(model_name))
    sample_weight = regime_balanced_training_weights(regimes.reindex(train))
    with threadpool_limits(limits=4):
        model.fit(fitted, y.reindex(train).astype(int), sample_weight=sample_weight.to_numpy())
        probability = _aligned_proba(model, scored)
    prediction = probability.argmax(axis=1)
    frame = pd.DataFrame({
        "timestamp": test,
        "width_bps": width,
        "candidate_id": 0,
        "fold_id": 0,
        "model_name": model_name,
        "y_true": y.reindex(test).astype(int).to_numpy(),
        "pred": prediction,
        "confidence": probability.max(axis=1),
        **{name: probability[:, i] for i, name in enumerate(PROBABILITY_COLUMNS)},
        "train_start": train.min(),
        "train_end": train.max(),
        "test_start": test.min(),
        "test_end": test.max() + BAR,
        "refit_id": f"rq4-v5-frozen-2024-{model_name}-w{width}",
    })
    validate_prediction(frame, width)
    audit = {
        "model_name": model_name, "width_bps": width, "fold_id": 0,
        "train_rows": len(train), "test_rows": len(test),
        "train_start": train.min(), "train_end": train.max(),
        "latest_training_label_available": train.max() + 2 * BAR,
        "first_test_decision": test.min() + BAR,
        "test_end": test.max() + BAR,
        "missing_train_values": int(raw_train.isna().sum().sum()),
        "missing_test_values": int(raw_test.isna().sum().sum()),
        "empty_training_features": "|".join(raw_train.columns[raw_train.isna().all()]),
        "imputation": "training-fold median; all-missing training feature becomes zero",
    }
    return frame, audit


def source_stage_paths(model_name: str, width: int, stage: str) -> list[Path]:
    if stage == "h1":
        directories = [
            POLICY / model_name / "stage_predictions" / f"calibration_2025_{month:02d}"
            for month in range(1, 7)
        ]
    elif stage == "forward":
        directories = [RAW / model_name / "stage_predictions" / "raw_forward"]
    else:
        raise ValueError(stage)
    paths = []
    for directory in directories:
        matches = sorted(directory.glob(f"w{width}_*.parquet"))
        if len(matches) != 1:
            raise ValueError(f"expected one source prediction file in {directory}")
        paths.extend(matches)
    return paths


def prepare(output_root: Path = CACHE) -> dict[str, Any]:
    import torch

    torch.set_num_threads(4)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source_paths = [
        CODE_ROOT / "data/btcusdt_15min_2021_2026.parquet",
        CODE_ROOT / "data/btcusdt_positioning_15min_2021_2026.parquet",
        CODE_ROOT / "data/btcusdt_m15_2024_2025.parquet",
        CODE_ROOT / "data/btcusdt_positioning_m15_2024_2026.parquet",
        Path(__file__), CODE_ROOT / "features/build.py",
        CODE_ROOT / "models/zoo.py", CODE_ROOT / "models/deep.py",
    ]
    source_hashes = {str(path.relative_to(CODE_ROOT)): file_hash(path) for path in source_paths}
    signature = payload_hash({
        "sources": source_hashes, "features": FEATURES, "threads": 4,
        "fit_start": FIT_START, "fit_end": DEVELOPMENT_START, "test_end": DEVELOPMENT_END,
    })
    jobs = [(model, WIDTH_BPS) for model in MODEL_NAMES] + [
        ("lstm", BASELINE_WIDTH_BPS), ("svm_linear", UNION_SVM_WIDTH_BPS)
    ]
    predictions: dict[str, dict[str, list[str]]] = {stage: {} for stage in ("development", "h1", "forward")}
    source_forecasts: dict[str, str] = {}
    audits = []
    started = time.monotonic()
    completed = 0
    total = len(jobs)
    X, labels, regimes = load_development_features()
    oof_checks = []
    for model in MODEL_NAMES:
        files = sorted((RAW / model / "predictions").glob(f"w{WIDTH_BPS}_candidate_00_fold_*.parquet"))
        if len(files) != 5:
            raise ValueError(f"expected five saved 2024 OOF files for {model}")
        for path in files:
            oof = pd.read_parquet(path, columns=["timestamp", "y_true", "fold_id"])
            times = pd.DatetimeIndex(pd.to_datetime(oof["timestamp"], utc=True))
            actual = labels[WIDTH_BPS].reindex(times)
            if actual.isna().any() or not np.array_equal(actual.to_numpy(), oof["y_true"].to_numpy()):
                raise ValueError(f"saved OOF label contract differs: {path.name}")
            oof_checks.append({"model_name": model, "fold_id": int(oof["fold_id"].iloc[0]),
                               "rows": len(oof), "labels_match": True, "sha256": file_hash(path)})
    _atomic_parquet(pd.DataFrame(oof_checks), output_root / "saved_oof_checks.parquet")
    _atomic_json({"status": "fitting", "completed": 0, "total": total}, output_root / "data_run_state.json")

    for model, width in jobs:
        identity = payload_hash({"data": signature, "model": model, "width": width})
        relative = Path("predictions/development") / f"{model}_w{width}_{identity[:12]}.parquet"
        path = output_root / relative
        audit_path = path.with_suffix(".json")
        state = {
            "status": "fitting", "completed": completed, "total": total,
            "model": model, "width_bps": width,
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }
        _atomic_json(state, output_root / "data_run_state.json")
        print(f"RQ4 forecasts {completed + 1}/{total}: frozen 2024, {model}, DZ{width}", flush=True)
        if path.is_file() and audit_path.is_file():
            frame = pd.read_parquet(path)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if audit.get("input_signature") != identity or audit.get("sha256") != file_hash(path):
                raise ValueError(f"forecast checkpoint changed: {path.name}")
            validate_prediction(frame, width)
        else:
            frame, audit = fit_development_model(X, labels[width], regimes, model, width)
            _atomic_parquet(frame, path)
            audit.update({"input_signature": identity, "sha256": file_hash(path)})
            _atomic_json(audit, audit_path)
        key = f"{model}:w{width}"
        predictions["development"].setdefault(key, []).append(relative.as_posix())
        audits.append(audit)
        completed += 1

    for stage in ("h1", "forward"):
        for model, width in jobs:
            paths = source_stage_paths(model, width, stage)
            for path in paths:
                source_forecasts[str(path.relative_to(CODE_ROOT))] = file_hash(path)
            frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True).sort_values("timestamp")
            validate_prediction(frame, width)
            relative = Path("predictions") / stage / f"{model}_w{width}.parquet"
            _atomic_parquet(frame, output_root / relative)
            predictions[stage][f"{model}:w{width}"] = [relative.as_posix()]

    union_signals = {}
    for stage in predictions:
        if stage == "development":
            members = {}
            for member in MEMBERS:
                name = predictions[stage][f"{member.model}:w{member.width_bps}"][0]
                frame = pd.read_parquet(output_root / name).set_index("timestamp")
                members[f"{member.model}_signal"] = member_signal(frame, tau=member.tau)
            signal = combine_union(pd.DataFrame(members))
        else:
            # Preserve the exact older qualified-Union predictions for fallback.
            signal = build_union_frame(stage)["union_signal"]
            for member in MEMBERS:
                for path in prediction_paths(member, stage):
                    source_forecasts[str(path.relative_to(CODE_ROOT))] = file_hash(path)
        relative = f"predictions/{stage}/union_signals.parquet"
        _atomic_parquet(signal.rename_axis("timestamp").reset_index(), output_root / relative)
        union_signals[stage] = relative

    output = {
        "protocol": "rq4-nine-model-forecasts-v5", "data_signature": signature,
        "model_names": list(MODEL_NAMES), "width_bps": WIDTH_BPS,
        "baseline_model": "lstm", "baseline_width_bps": BASELINE_WIDTH_BPS,
        "features": list(FEATURES), "source_hashes": source_hashes,
        "reused_forecast_hashes": source_forecasts, "predictions": predictions,
        "union_signals": union_signals,
        "development_fit_count": completed,
        "development_role": "continuous 2024 diagnostic; execution policy was selected in 2025 H1",
        "development_fit_start": FIT_START, "development_fit_end_exclusive": DEVELOPMENT_START,
        "development_model_refits": 1,
        "saved_oof_label_checks": len(oof_checks),
        "early_missing_features": "training-fold median imputation; all-missing training columns become zero",
        "q2_accessed": False,
    }
    output["artifact_hashes"] = {
        name: file_hash(output_root / name)
        for stage in predictions.values() for names in stage.values() for name in names
    }
    output["artifact_hashes"].update({name: file_hash(output_root / name) for name in union_signals.values()})
    _atomic_parquet(pd.DataFrame(audits), output_root / "training_audit.parquet")
    _atomic_json(output, output_root / "data_manifest.json")
    _atomic_json({
        "status": "complete", "completed": completed, "total": total,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }, output_root / "data_run_state.json")
    return output


def load_panel(output_root: Path, stage: str, *, width: int = WIDTH_BPS,
               model_names: tuple[str, ...] = MODEL_NAMES) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    root = Path(output_root)
    manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
    reference: pd.DataFrame | None = None
    probabilities = {}
    for model in model_names:
        names = manifest["predictions"][stage][f"{model}:w{width}"]
        frames = []
        for name in names:
            if file_hash(root / name) != manifest["artifact_hashes"][name]:
                raise ValueError(f"forecast artifact hash changed: {name}")
            frames.append(pd.read_parquet(root / name))
        frame = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        validate_prediction(frame, width)
        if reference is None:
            reference = frame.loc[:, ["timestamp", "y_true", "fold_id"]].copy()
        elif not reference[["timestamp", "y_true"]].equals(frame[["timestamp", "y_true"]]):
            raise ValueError(f"all-nine prediction rows or labels do not align: {stage}/{model}")
        probabilities[model] = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    if reference is None:
        raise ValueError("model list is empty")
    reference["decision_time"] = pd.to_datetime(reference["timestamp"], utc=True) + BAR
    return reference, probabilities


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=CACHE)
    args = parser.parse_args()
    result = prepare(args.output_root)
    print(json.dumps({"status": "complete", "new_fits": result["development_fit_count"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
