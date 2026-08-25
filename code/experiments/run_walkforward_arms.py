"""Frozen CatBoost walk-forward arms on the positioning substrate (new protocol).

The legacy, F1, and economic arms differ only in hyperparameters; the
SqrtBalanced control reuses the economic dz40 parameters and changes only automatic
class weighting. All use identical price, order-flow, and positioning features, so
the forward months adjudicate the tuning-objective question.
honestly — including the econ-tuned arm, whose dev-window numbers were in-sample:

  legacy   btc_both_m15_catboost_balanced.json          (2024-only, tuned @25 bps)
  f1       btc_bothofpos_dz{40,55}_catboost_balanced.json   (dev window, macro-F1)
  econ     btc_bothofpos_dz{40,55}_econ_catboost_balanced.json (min gated So/Sh)
  sqrtbalanced  econ dz40 parameters with SqrtBalanced class weighting only

Widths 40 / 55 / 75 bps (75 borrows the dz55 parameters — nearest tuned width).
Weekly retraining over the config walkforward span; caches are resumable.
Economics (gate calibration on 2025-H1, frozen; per-bar + 1-minute bracket
engines; DM tests) live in run_arms_economics.py.

Run:  python -m experiments.run_walkforward_arms
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from experiments.run_walkforward import build_walkforward_xy
from experiments.spans import CACHE_SUFFIX
from experiments.walkforward import run_walkforward_predictions, weekly_walkforward_windows
from features.build import make_label
from models.zoo import MODELS

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
TUNING = CODE_ROOT / "experiments" / "cache" / "tuning"
WF_DIR = CODE_ROOT / "experiments" / "cache" / "walkforward"

WIDTHS = (40, 55, 75)
MODEL = "catboost_balanced"
ARMS = ("legacy", "f1", "econ", "sqrtbalanced")
SQRTBALANCED_WIDTHS = (40,)


def arms_for_width(width: int) -> tuple[str, ...]:
    """Return the pre-registered comparison arms available at one label width."""
    if width not in WIDTHS:
        raise ValueError(f"unsupported label width: {width}")
    return tuple(
        arm for arm in ARMS
        if arm != "sqrtbalanced" or width in SQRTBALANCED_WIDTHS
    )


def arm_params(arm: str, width: int) -> dict:
    w = min(width, 55)                      # 75 borrows dz55 (nearest tuned width)
    name = {
        "legacy": "btc_both_m15_catboost_balanced.json",
        "f1": f"btc_bothofpos_dz{w}_catboost_balanced.json",
        "econ": f"btc_bothofpos_dz{w}_econ_catboost_balanced.json",
        "sqrtbalanced": f"btc_bothofpos_dz{w}_econ_catboost_balanced.json",
    }[arm]
    params = json.loads((TUNING / name).read_text(encoding="utf-8"))["best_params"]
    if arm == "sqrtbalanced":
        return {**params, "auto_class_weights": "SqrtBalanced"}
    return params


def cache_path(arm: str, width: int) -> Path:
    return WF_DIR / f"btc_bothofpos_cb-{arm}_dz{width}_{CACHE_SUFFIX}.parquet"


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for width in WIDTHS:
        X, y, aux = build_walkforward_xy(
            "btc", cfg, horizon=1, sentiment="both",
            label_fn=lambda feat, w=width: make_label(feat, threshold_bps=w, horizon=1),
            orderflow=True, positioning=True)
        walk_start, walk_end = cfg["dates"]["walkforward"]
        windows = weekly_walkforward_windows(
            X.index, walk_start=walk_start, walk_end=walk_end, train_lookback="180D")

        for arm in arms_for_width(width):
            out = cache_path(arm, width)
            if out.exists():
                print(f"[cached] {out.name}")
                continue
            preds = run_walkforward_predictions(
                X, y, windows=windows, model_factory=MODELS[MODEL],
                model_name=MODEL, params=arm_params(arm, width),
                min_train_rows=500, min_validation_rows=50,
                include_features=False, progress_label=f"dz{width}:{arm}",
                train_tail_trim=1)
            if preds.empty:
                raise RuntimeError(f"no predictions for dz{width}:{arm}")
            preds = preds.join(aux[["forward_return", "vol_regime"]], how="left")
            out.parent.mkdir(parents=True, exist_ok=True)
            preds.to_parquet(out)
            print(f"wrote {len(preds):,} rows -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
