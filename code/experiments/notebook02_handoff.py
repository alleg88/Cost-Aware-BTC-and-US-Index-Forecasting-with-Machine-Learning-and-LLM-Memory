"""Strict file contracts for the direct Notebook 01 -> 02b pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from experiments.catboost_matched_ablation import CANDIDATE_POOL_FINGERPRINT


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK01_ROOT = CODE_ROOT / "experiments" / "cache" / "tuning" / "notebook01_handoff"
NOTEBOOK01_WIDTHS = NOTEBOOK01_ROOT / "selected_widths.parquet"
PIPELINE_ROOT = CODE_ROOT / "experiments" / "cache" / "tuning" / "notebook02_no_sentiment"
PIPELINE_HANDOFF = PIPELINE_ROOT / "handoff.json"
MATCHED_ROOT = PIPELINE_ROOT / "matched_catboost_monthly_h1"

WIDTHS = (55, 65, 75)
TRAINING_HISTORY_DAYS = 180
WIDTH_COLUMNS = ("width_bps", "sortino", "sharpe", "net_return", "trades")
BASE_FEATURE_COLUMNS = (
    "r1", "r5", "r20", "vol_10", "vol_20", "vol_60", "hl_range", "co_range",
    "rsi_14", "volume", "vol_z", "hour", "dayofweek", "ofi", "ofi_z20",
    "ofi_mom5", "trade_intensity_z", "funding_rate", "funding_z", "oi_chg_1h",
    "oi_chg_4h", "oi_z", "toptrader_ls_z", "taker_ls_z",
)
SENTIMENT_MARKERS = (
    "sentiment",
    "news_",
    "fear_greed",
    "macro_",
    "llm_",
    "finbert",
)


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    canonical = frame.reset_index(drop=True)
    values = pd.util.hash_pandas_object(canonical, index=True).to_numpy()
    digest = hashlib.sha256(values.tobytes())
    digest.update("|".join(map(str, canonical.columns)).encode("utf-8"))
    return digest.hexdigest()


def _validate_width_set(values: Sequence[int]) -> None:
    if set(map(int, values)) != set(WIDTHS) or len(values) != len(WIDTHS):
        raise ValueError("handoff must contain exactly DZ55, DZ65 and DZ75")


def write_notebook01_handoff(frame: pd.DataFrame, path: str | Path = NOTEBOOK01_WIDTHS) -> Path:
    missing = set(WIDTH_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Notebook 01 handoff is missing columns: {sorted(missing)}")
    out = frame.loc[:, WIDTH_COLUMNS].copy()
    _validate_width_set(out["width_bps"].tolist())
    out["width_bps"] = out["width_bps"].astype(int)
    out["trades"] = out["trades"].astype(int)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(target, index=False)
    return target


def load_notebook01_handoff(path: str | Path = NOTEBOOK01_WIDTHS) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Notebook 01 handoff is missing: {source}")
    frame = pd.read_parquet(source)
    missing = set(WIDTH_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Notebook 01 handoff is missing columns: {sorted(missing)}")
    frame = frame.loc[:, WIDTH_COLUMNS].copy()
    _validate_width_set(frame["width_bps"].tolist())
    return frame


def _validate_feature_columns(columns: Sequence[str]) -> list[str]:
    names = [str(column) for column in columns]
    if not names or len(names) != len(set(names)):
        raise ValueError("feature columns must be a non-empty unique list")
    forbidden = [
        column
        for column in names
        if any(marker in column.lower() for marker in SENTIMENT_MARKERS)
    ]
    if forbidden:
        raise ValueError(f"sentiment features are forbidden in Notebook 02: {forbidden}")
    return names


def write_pipeline_handoff(
    *,
    upstream: pd.DataFrame,
    path: str | Path = PIPELINE_HANDOFF,
) -> Path:
    validated_upstream = upstream.loc[:, WIDTH_COLUMNS].copy()
    _validate_width_set(validated_upstream["width_bps"].tolist())
    columns = _validate_feature_columns(BASE_FEATURE_COLUMNS)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "upstream_notebook": "01_data_labels_and_baseline.ipynb",
        "widths": list(WIDTHS),
        "training_histories_days": {
            str(width): TRAINING_HISTORY_DAYS for width in WIDTHS
        },
        "features": {
            "price": True,
            "order_flow": True,
            "positioning": True,
            "sentiment": False,
            "columns": columns,
        },
        "sentiment": "none",
        "candidate_pool_fingerprint": CANDIDATE_POOL_FINGERPRINT,
        "upstream_fingerprint": notebook01_fingerprint(validated_upstream),
    }
    payload["handoff_fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def load_pipeline_handoff(path: str | Path = PIPELINE_HANDOFF) -> Mapping[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Notebook 01 handoff is missing: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("sentiment") != "none" or payload.get("features", {}).get("sentiment") is not False:
        raise ValueError("Notebook 01 handoff must disable sentiment")
    _validate_width_set(payload.get("widths", []))
    histories = payload.get("training_histories_days", {})
    if set(histories) != {str(width) for width in WIDTHS} or any(
        int(value) != TRAINING_HISTORY_DAYS for value in histories.values()
    ):
        raise ValueError("Notebook 01 handoff must use 180 days for every dead zone")
    _validate_feature_columns(payload.get("features", {}).get("columns", []))
    if payload.get("candidate_pool_fingerprint") != CANDIDATE_POOL_FINGERPRINT:
        raise ValueError("candidate pool fingerprint does not match the frozen pool")
    fingerprint = payload.pop("handoff_fingerprint", None)
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload["handoff_fingerprint"] = fingerprint
    if fingerprint != expected:
        raise ValueError("Notebook 01 handoff fingerprint mismatch")
    return payload


def notebook01_fingerprint(frame: pd.DataFrame) -> str:
    return _frame_fingerprint(frame.loc[:, WIDTH_COLUMNS])
