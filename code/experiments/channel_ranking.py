"""E8 — can a ranking model select a profitable subset from an unprofitable pool?

The mechanical rule does not clear its costs. This asks the only question that
follows: whether the candidates it produces differ from each other in a way a model
can read at decision time.

Three choices make the answer trustworthy rather than flattering:

  forward-chaining folds over channel episodes, never random ones. Candidates inside
  one channel share a regime, and a random split would put a trade in the test fold
  whose own channel trained the model.

  the score for every candidate comes from a fold that did not see it, so the
  threshold is swept over out-of-fold predictions and the reported result is not the
  one the threshold was chosen on.

  the metric is net R per trade, not accuracy or AUC. A model can rank well and still
  select trades that do not pay their costs, and only the economic figure notices.

The control is take-every-candidate. A model that cannot beat it is not a weak model;
it is evidence that the pool is homogeneous, which is itself a result.

Run:  python -m experiments.channel_ranking --events <events.parquet>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CODE_ROOT = Path(__file__).resolve().parents[1]
CACHE = CODE_ROOT / "experiments" / "cache" / "channel_study"

NON_FEATURES = {
    "signal_time", "decision_time", "side", "channel_episode_id",
    "order_status", "filled", "r_net", "label_net_positive",
}
# Known only after the fact. Named explicitly so a future column cannot drift in.
FORBIDDEN = {"bars_held", "outcome", "exit_time", "exit_price", "r_gross",
             "net_return", "rr_realised"}


def feature_columns(events: pd.DataFrame) -> list[str]:
    cols = [c for c in events.columns if c not in NON_FEATURES]
    leaked = sorted(set(cols) & FORBIDDEN)
    if leaked:
        raise ValueError(f"outcome-derived columns in the feature set: {leaked}")
    return cols


def episode_folds(episodes: pd.Series, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward-chaining folds whose boundaries fall between channel episodes.

    Each fold trains on everything before a cut and tests on the block after it, so
    no test candidate is ever contemporaneous with its training data and no episode
    is split across the boundary.
    """
    order = episodes.drop_duplicates().sort_values().to_numpy()
    if len(order) < n_splits + 1:
        raise ValueError(f"{len(order)} episodes cannot make {n_splits} folds")
    bounds = np.array_split(order, n_splits + 1)
    folds = []
    for k in range(1, n_splits + 1):
        train_eps = np.concatenate(bounds[:k])
        test_eps = bounds[k]
        folds.append((episodes.isin(train_eps).to_numpy(),
                      episodes.isin(test_eps).to_numpy()))
    return folds


def out_of_fold_scores(X: pd.DataFrame, y: np.ndarray, episodes: pd.Series,
                       n_splits: int = 5, C: float = 1.0) -> np.ndarray:
    """Score every candidate with a model that never saw it. Rows before the first
    test block keep NaN: nothing legitimate can score them."""
    scores = np.full(len(X), np.nan)
    for train, test in episode_folds(episodes, n_splits):
        if train.sum() < 50 or test.sum() < 10 or len(np.unique(y[train])) < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=2000, class_weight="balanced"),
        )
        model.fit(X[train], y[train])
        scores[test] = model.predict_proba(X[test])[:, 1]
    return scores


def bootstrap_by_episode(values: pd.Series, episodes: pd.Series,
                         n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Resample whole channels, not individual trades. Trades inside a channel are
    not independent draws, and resampling them separately narrows the interval to a
    width the data does not support."""
    rng = np.random.default_rng(seed)
    groups = [v.to_numpy() for _, v in values.groupby(episodes)]
    if not groups:
        return (np.nan, np.nan)
    means = np.empty(n_boot)
    idx = np.arange(len(groups))
    for b in range(n_boot):
        pick = rng.choice(idx, size=len(groups), replace=True)
        means[b] = np.concatenate([groups[i] for i in pick]).mean()
    return tuple(np.quantile(means, [0.025, 0.975]))


def evaluate(events: pd.DataFrame, *, n_splits: int = 5, C: float = 1.0,
             quantiles=(0.10, 0.20, 0.30, 0.50)) -> dict:
    filled = events[events["filled"].astype(bool)].copy()
    if filled.empty:
        raise ValueError("no filled events to evaluate")
    cols = feature_columns(filled)
    X = filled[cols].astype(float).fillna(0.0)
    y = filled["label_net_positive"].astype(int).to_numpy()
    eps = filled["channel_episode_id"]

    filled["score"] = out_of_fold_scores(X, y, eps, n_splits=n_splits, C=C)
    scored = filled[filled["score"].notna()].copy()

    lo, hi = bootstrap_by_episode(scored["r_net"], scored["channel_episode_id"])
    control = {"selection": "take all", "n": int(len(scored)),
               "share": 1.0, "mean_net_r": float(scored["r_net"].mean()),
               "ci_low": float(lo), "ci_high": float(hi),
               "episodes": int(scored["channel_episode_id"].nunique())}

    rows = [control]
    for q in quantiles:
        cut = scored["score"].quantile(1 - q)
        sel = scored[scored["score"] >= cut]
        if len(sel) < 30:
            continue
        lo, hi = bootstrap_by_episode(sel["r_net"], sel["channel_episode_id"])
        rows.append({"selection": f"top {q:.0%} by model", "n": int(len(sel)),
                     "share": round(len(sel) / len(scored), 3),
                     "mean_net_r": float(sel["r_net"].mean()),
                     "ci_low": float(lo), "ci_high": float(hi),
                     "episodes": int(sel["channel_episode_id"].nunique())})

    table = pd.DataFrame(rows)
    table["beats_control"] = table["ci_low"] > control["mean_net_r"]
    return {"table": table, "features": cols, "scored": scored,
            "n_features": len(cols), "n_splits": n_splits, "C": C}


def side_comparison(events: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Pooled against one model per side, and whether a pooled score ranks INSIDE
    each side. If a pooled model merely preferred the better-performing side, its
    slices would be skewed towards it and it would not rank within either; the
    within-side rows are what separate real ranking from side arbitrage."""
    rows = []
    pooled = evaluate(events, n_splits=n_splits)
    s = pooled["scored"]
    for q in (0.10, 0.30):
        sel = s[s["score"] >= s["score"].quantile(1 - q)]
        rows.append({"model": "pooled", "scope": "both sides", "slice": f"top {q:.0%}",
                     "n": len(sel), "short share": round((sel["side"] == "short").mean(), 3),
                     "mean_net_r": round(sel["r_net"].mean(), 4)})
    rows.append({"model": "pooled", "scope": "both sides", "slice": "all",
                 "n": len(s), "short share": round((s["side"] == "short").mean(), 3),
                 "mean_net_r": round(s["r_net"].mean(), 4)})

    for side in ("long", "short"):
        sub = s[s["side"] == side]
        cut = sub["score"].quantile(0.70)
        rows.append({"model": "pooled", "scope": f"within {side}", "slice": "top 30%",
                     "n": int((sub["score"] >= cut).sum()), "short share": np.nan,
                     "mean_net_r": round(sub.loc[sub["score"] >= cut, "r_net"].mean(), 4)})
        rows.append({"model": "pooled", "scope": f"within {side}", "slice": "all",
                     "n": len(sub), "short share": np.nan,
                     "mean_net_r": round(sub["r_net"].mean(), 4)})
        try:
            own = evaluate(events[events["side"] == side], n_splits=max(3, n_splits - 1))
            t = own["table"]
            for _, r in t[t["selection"].isin(["take all", "top 30% by model"])].iterrows():
                rows.append({"model": f"{side}-only", "scope": side,
                             "slice": r["selection"], "n": int(r["n"]),
                             "short share": np.nan,
                             "mean_net_r": round(float(r["mean_net_r"]), 4)})
        except Exception:
            pass
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", type=Path, default=None,
                    help="events.parquet from a channel_study run")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--trials-so-far", type=int, default=160,
                    help="configurations evaluated during design, carried into the "
                         "multiple-testing correction rather than recalled later")
    args = ap.parse_args()

    path = args.events
    if path is None:
        cands = [d / "events.parquet" for d in CACHE.glob("*/")
                 if (d / "events.parquet").exists()]
        if not cands:
            raise SystemExit("no events.parquet found; run experiments.channel_study first")
        path = max(cands, key=lambda p: p.stat().st_mtime)

    events = pd.read_parquet(path)
    res = evaluate(events, n_splits=args.splits, C=args.C)
    table = res["table"]

    print(f"events: {path.parent.name}")
    print(f"features ({res['n_features']}): {', '.join(res['features'])}")
    print(f"folds: {args.splits} forward-chaining over channel episodes\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    control = table.iloc[0]["mean_net_r"]
    best = table.iloc[1:]["mean_net_r"].max() if len(table) > 1 else np.nan
    print(f"\ncontrol (take all): {control:+.4f} net R per trade")
    if len(table) > 1:
        print(f"best model slice:   {best:+.4f}")
        print("model beats control with a 95% interval clear of it: "
              f"{bool(table.iloc[1:]['beats_control'].any())}")

    sides = side_comparison(events, n_splits=args.splits)
    print()
    print("pooled versus one model per side")
    print(sides.to_string(index=False))
    sides.to_csv(path.parent / "ranking_e8_sides.csv", index=False)
    table.to_csv(path.parent / "ranking_e8_table.csv", index=False)

    out = path.parent / "ranking_e8.json"
    out.write_text(json.dumps(
        {"events": str(path), "features": res["features"], "n_splits": args.splits,
         "C": args.C, "trials_so_far": args.trials_so_far,
         "table": table.to_dict(orient="records")}, indent=2), encoding="utf-8")
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
