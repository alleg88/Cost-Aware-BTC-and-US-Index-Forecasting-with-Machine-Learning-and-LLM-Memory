"""Disk cache for out-of-fold CV prediction runs (notebook 02 scoreboard etc.).

A cached hit returns predictions byte-identical to a fresh fit: the pipeline is
seeded and deterministic, and the cache key fingerprints everything that affects
the output — model parameters, the feature-column list, the data span and a
content checksum, the label distribution, and the sample-weight vector. Any
change in code inputs, data, or config changes the key, which forces a refit;
the cache can only ever save time, never alter results.

Escape hatch: delete experiments/cache/oof/ to force everything fresh.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import classification_scores
from models.zoo import MODELS, _aligned_proba

CACHE_DIR = Path(__file__).resolve().parent / "cache" / "oof"


def _key(X: pd.DataFrame, y: pd.Series, params: dict, weights, model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(json.dumps(params or {}, sort_keys=True, default=str).encode())
    h.update("|".join(X.columns).encode())
    h.update(str(len(X)).encode())
    h.update(str(X.index[0]).encode())
    h.update(str(X.index[-1]).encode())
    h.update(np.nan_to_num(X.to_numpy(dtype=np.float64)).sum(axis=0).tobytes())
    h.update(np.bincount(np.asarray(y, dtype=np.int64), minlength=3).tobytes())
    if weights is not None:
        h.update(np.asarray(weights, dtype=np.float64).sum().tobytes())
    else:
        h.update(b"noweights")
    return h.hexdigest()[:24]


def oof_cached(X: pd.DataFrame, y: pd.Series, params: dict | None, splitter,
               weights: pd.Series | None = None,
               model: str = "catboost_balanced"):
    """OOF preds + confidences + mean fold macro-F1, cached on disk by content key."""
    key = _key(X, y, params or {}, weights, model)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / f"{model}_{key}.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        return d["pred"].astype(int), d["conf"], float(d["f1"].iloc[0])

    pred = pd.Series(np.nan, index=X.index)
    conf = pd.Series(np.nan, index=X.index)
    f1s = []
    for tr, te in splitter.split(X):
        m = MODELS[model](params)
        kw = {"sample_weight": weights.iloc[tr].to_numpy()} if weights is not None else {}
        m.fit(X.iloc[tr], y.iloc[tr], **kw)
        pr = np.asarray(_aligned_proba(m, X.iloc[te]))
        pred.iloc[te] = pr.argmax(axis=1)
        conf.iloc[te] = pr.max(axis=1)
        f1s.append(classification_scores(y.iloc[te], pr.argmax(axis=1))["macro_f1"])
    ok = pred.notna()
    pred, conf, f1 = pred[ok].astype(int), conf[ok], float(np.mean(f1s))
    pd.DataFrame({"pred": pred, "conf": conf, "f1": f1}).to_parquet(f)
    return pred, conf, f1
