"""Benchmark LLM scorers on speed, coverage, and agreement.

Each candidate model scores the same headline sample with batched, temperature-0,
schema-constrained settings. Agreement metrics compare each model with the reference scorer.

Run:  python -m experiments.benchmark_scorers --n 40
      python -m experiments.benchmark_scorers --models gemma4:12b gemma4:4b deepseek-v4-flash:cloud
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from sentiment.score import _norm
from sentiment.score_llm import BATCH, RAW_DIR, _score_batch

REFERENCE = "gemma4:12b"
DEFAULT_MODELS = ["gemma4:12b", "gemma4:4b", "deepseek-v4-flash:cloud"]


def sample_headlines(n: int, seed: int = 42) -> list[str]:
    df = pd.read_parquet(RAW_DIR / "gdelt_btc.parquet")[["title"]].copy()
    df["title_norm"] = _norm(df["title"])
    uniq = df[df["title_norm"] != ""].drop_duplicates("title_norm")
    return uniq.sample(n, random_state=seed)["title"].tolist()


def run_model(model: str, titles: list[str]) -> tuple[pd.DataFrame, float]:
    batches = [titles[i:i + BATCH] for i in range(0, len(titles), BATCH)]
    t0 = time.time()
    results = []
    for b in batches:
        results.extend(_score_batch(b, model=model))
    dt = time.time() - t0
    rows = [(r if r is not None else
             {"llm_sent": np.nan, "llm_relevance": np.nan, "llm_impact": np.nan, "llm_asset": None})
            for r in results]
    return pd.DataFrame(rows), dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=40, help="headlines to score per model")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    titles = sample_headlines(args.n, args.seed)
    print(f"benchmarking {len(args.models)} models on {len(titles)} headlines "
          f"(batch={BATCH}, reference={REFERENCE})\n")

    outputs: dict[str, pd.DataFrame] = {}
    for m in args.models:
        out, dt = run_model(m, titles)
        outputs[m] = out
        print(f"  {m:28s} {dt:6.1f}s total = {dt/len(titles):5.2f}s/headline | "
              f"valid {out['llm_sent'].notna().mean():4.0%}")

    if REFERENCE in outputs:
        ref = outputs[REFERENCE]
        print(f"\nagreement vs {REFERENCE}:")
        print(f"  {'model':28s} {'sent MAE':>9} {'sent corr':>10} {'impact=':>8} {'asset=':>7}")
        for m, out in outputs.items():
            if m == REFERENCE:
                continue
            both = ref["llm_sent"].notna() & out["llm_sent"].notna()
            mae = (ref["llm_sent"][both] - out["llm_sent"][both]).abs().mean()
            corr = ref["llm_sent"][both].corr(out["llm_sent"][both])
            imp = (ref["llm_impact"][both] == out["llm_impact"][both]).mean()
            ast = (ref["llm_asset"][both] == out["llm_asset"][both]).mean()
            print(f"  {m:28s} {mae:9.3f} {corr:10.3f} {imp:8.0%} {ast:7.0%}")

    # side-by-side examples for eyeballing
    print("\nexamples (sentiment / impact / asset):")
    show = min(6, len(titles))
    for i in range(show):
        print(f"  {titles[i][:72]}")
        for m, out in outputs.items():
            r = out.iloc[i]
            print(f"    {m:28s} {r['llm_sent']!s:>5} / {r['llm_impact']!s:>3} / {r['llm_asset']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
