"""Create 2025 weekly walk-forward prediction caches for any registry model.

Optuna-tuned parameters (experiments/cache/tuning/) are applied automatically when
present; for stack_all every tuned base picks up its own tuned parameters.

Examples:
    python -m experiments.run_walkforward --instrument btc --sentiment both --limit-windows 2
    python -m experiments.run_walkforward --instrument btc --sentiment both --model gru
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from experiments.walkforward import run_walkforward_predictions, weekly_walkforward_windows
from features.build import (FEATURE_COLS, ORDERFLOW_FEATURE_COLS,
                            POSITIONING_FEATURE_COLS, POSITIONING_SOURCE_COLS,
                            add_features, make_label)
from features.sentiment import assemble_sentiment_features, build_llm_features
from models.zoo import MODELS

STACK_ALL_BASES = ("logreg", "random_forest", "xgboost_balanced", "catboost_balanced", "lstm")

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
OUT_DIR = CODE_ROOT / "experiments" / "cache" / "walkforward"


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    return out.sort_index()


def _read_parquet_before(
    path: str | Path, *, end_exclusive: str | pd.Timestamp
) -> pd.DataFrame:
    boundary = pd.Timestamp(end_exclusive)
    boundary = (
        boundary.tz_localize("UTC")
        if boundary.tzinfo is None
        else boundary.tz_convert("UTC")
    )
    frame = pd.read_parquet(
        Path(path), filters=[("timestamp", "<", boundary.to_pydatetime())]
    )
    output = _ensure_utc_index(frame)
    if len(output) and output.index.max() >= boundary:
        raise AssertionError("predicate-pushed parquet read crossed its cutoff")
    return output


def build_walkforward_xy(
    instrument: str,
    cfg: dict,
    *,
    horizon: int = 1,
    sentiment: str = "both",
    label_fn=None,
    orderflow: bool = True,
    positioning: bool = False,
    end_exclusive: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build features, labels, and report columns for one instrument.

    label_fn (feature frame -> label Series) overrides the default dead-zone
    label — used by the threshold sweep and the triple-barrier retraining.
    forward_return in the report columns stays the next-`horizon`-bar return
    either way (it feeds the per-bar economics, not the label).

    orderflow (default on) adds the Binance aggressor-side microstructure block
    to the feature set where the source provides it; pass False for the
    price+sentiment-only ablation.
    """
    inst = cfg["instruments"][instrument]
    bars_path = CODE_ROOT / inst["working_parquet"]
    if end_exclusive is not None:
        bars = _read_parquet_before(bars_path, end_exclusive=end_exclusive)
        boundary = pd.Timestamp(end_exclusive)
        boundary = boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")
    else:
        bars = _ensure_utc_index(pd.read_parquet(bars_path))
    if positioning:
        # perp funding/OI block (BTC only; see data/build_positioning.py)
        positioning_path = CODE_ROOT / "data" / "btcusdt_positioning_m15_2024_2026.parquet"
        if end_exclusive is not None:
            pos = _read_parquet_before(
                positioning_path, end_exclusive=end_exclusive
            )
        else:
            pos = _ensure_utc_index(pd.read_parquet(positioning_path))
        bars = bars.join(pos.reindex(bars.index))
    feat = add_features(bars)
    if label_fn is None:
        y = make_label(feat, threshold_bps=inst["threshold_bps"], horizon=horizon)
    else:
        y = label_fn(feat)

    base_cols = list(FEATURE_COLS)
    if orderflow:
        missing = [c for c in ORDERFLOW_FEATURE_COLS if c not in feat.columns]
        if missing:
            raise ValueError(
                f"order-flow features unavailable for {instrument} "
                f"(re-snapshot with the taker columns): {missing}")
        base_cols += ORDERFLOW_FEATURE_COLS
    if positioning:
        missing = [c for c in POSITIONING_FEATURE_COLS if c not in feat.columns]
        if missing:
            raise ValueError(f"positioning features unavailable: {missing}")
        base_cols += POSITIONING_FEATURE_COLS
    X = feat[base_cols].copy()
    if sentiment in {"classic", "both"}:
        X = X.join(assemble_sentiment_features(instrument, bars.index))
    if sentiment in {"llm", "both"}:
        X = X.join(build_llm_features(instrument, bars.index))

    train_start = pd.Timestamp(cfg["dates"]["train"][0], tz="UTC")
    walk_start = pd.Timestamp(cfg["dates"]["walkforward"][0], tz="UTC")
    train_vol = feat.loc[(feat.index >= train_start) & (feat.index < walk_start), "vol_60"]
    vol_cut = train_vol.dropna().median()
    aux = pd.DataFrame(index=bars.index)
    aux["forward_return"] = bars["close"].shift(-horizon) / bars["close"] - 1.0
    aux["vol_regime"] = (feat["vol_60"] >= vol_cut).map({True: "high", False: "low"})
    aux["hour"] = feat["hour"]
    aux["dayofweek"] = feat["dayofweek"]

    valid = X.notna().all(axis=1) & (y != -1)
    return X.loc[valid], y.loc[valid], aux.loc[valid]


def default_output_path(instrument: str, sentiment: str, horizon: int,
                        model: str = "catboost") -> Path:
    from experiments.horizons import horizon_label
    from experiments.spans import CACHE_SUFFIX

    return OUT_DIR / f"{instrument}_{sentiment}_{model}_{horizon_label(horizon)}_{CACHE_SUFFIX}.parquet"


def resolve_params(model: str, instrument: str, sentiment: str, horizon: int,
                   cfg: dict) -> dict | None:
    """Tuned params where they exist; stack_all gets per-base tuned params."""
    from experiments.run_tuning import load_tuned_params

    if model == "stack_all":
        per_base = {
            base: load_tuned_params(instrument, sentiment, horizon, base)
            for base in STACK_ALL_BASES
        }
        per_base = {base: params for base, params in per_base.items() if params}
        return per_base or None
    params = load_tuned_params(instrument, sentiment, horizon, model)
    if params is None and model == "catboost_balanced":
        params = cfg["model"]["catboost"]
    return params


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", default="btc", choices=["btc", "usa500", "usatech"])
    ap.add_argument("--sentiment", default="both", choices=["none", "classic", "llm", "both"])
    from experiments.horizons import DEFAULT_LABEL, parse_horizon
    ap.add_argument("--horizon", type=parse_horizon, default=DEFAULT_LABEL,
                    help="m15 (default) / h1 / h4, or a bar count")
    ap.add_argument("--model", default="catboost_balanced", choices=sorted(MODELS),
                    help="registry model to walk forward (tuned params applied if cached)")
    ap.add_argument("--train-lookback", default="180D")
    ap.add_argument("--limit-windows", type=int, default=None)
    ap.add_argument("--price-only", action="store_true",
                    help="drop the order-flow block (price+sentiment ablation)")
    ap.add_argument("--out", default=None, help="override output parquet path")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    orderflow = not args.price_only         # order-flow is the default feature set
    X, y, aux = build_walkforward_xy(
        args.instrument,
        cfg,
        horizon=args.horizon,
        sentiment=args.sentiment,
        orderflow=orderflow,
    )
    # order-flow (default) gets an "of" sentiment tag so its cache is distinct
    # from the price+sentiment ablation (btc_bothof_* vs btc_both_*).
    sentiment_tag = args.sentiment + ("" if args.price_only else "of")
    walk_start, walk_end = cfg["dates"]["walkforward"]
    windows = weekly_walkforward_windows(
        X.index,
        walk_start=walk_start,
        walk_end=walk_end,
        train_lookback=args.train_lookback,
    )
    if args.limit_windows is not None:
        windows = windows[: args.limit_windows]

    # full registry name in the filename; the legacy btc_*_catboost_* cache (untuned,
    # config params) stays untouched because the RQ3 reflection artifacts depend on it
    model_tag = args.model
    out_path = Path(args.out) if args.out else default_output_path(
        args.instrument, sentiment_tag, args.horizon, model_tag)
    params = resolve_params(args.model, args.instrument, args.sentiment, args.horizon, cfg)
    preds = run_walkforward_predictions(
        X,
        y,
        windows=windows,
        model_factory=MODELS[args.model],
        model_name=model_tag,
        params=params,
        cache_path=None,
        min_train_rows=500,
        min_validation_rows=50,
        include_features=True,
        progress_label=f"{args.instrument}:{args.sentiment}:{model_tag}",
        train_tail_trim=args.horizon,
    )
    if preds.empty:
        raise RuntimeError("no walk-forward predictions produced")
    preds = preds.join(aux[["forward_return", "vol_regime"]], how="left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out_path)

    print(
        f"[{args.instrument}] wrote {len(preds):,} validation predictions "
        f"across {preds['window'].nunique()} window(s) -> {out_path}"
    )
    print(f"  first={preds.index.min()} last={preds.index.max()} features={X.shape[1]}")
    print(f"  label_dist={dict(y.loc[preds.index].value_counts().sort_index())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
