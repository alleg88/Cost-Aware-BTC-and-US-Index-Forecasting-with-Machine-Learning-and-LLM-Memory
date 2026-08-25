"""Score news headlines with an LLM (Gemma via Ollama) into structured fields.

Path B of the sentiment layer (papers/wiki/design/news-scoring-howto.md): where the DeBERTa
scorer (score.py) gives tone only, the LLM returns sentiment + relevance + impact + asset in
one call — so downstream features can *filter* market-moving news from noise instead of
averaging every headline's tone.

Each UNIQUE headline is scored once via a forced JSON schema (deterministic, temperature 0),
mapped back to article level, and cached to scores_llm_<stream>.parquet. Caching is
incremental and resumable: a re-run only scores headlines not already cached for this
model+version, so a long run can be stopped and continued.

GPU: Ollama offloads the model to the GPU automatically (check with `ollama ps`). gemma4:12b
(~8 GB) fits a 16 GB card; use `--model gemma4:31b-cloud` to run on Ollama Cloud instead.

Run:  python -m sentiment.score_llm --stream btc --workers 4
      python -m sentiment.score_llm --stream btc --limit 200      # quick sample
      python -m sentiment.score_llm --stream btc --source-prefix direct_events --workers 4
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = CODE_ROOT / "sentiment" / "raw"

MODEL = "gemma4:12b"
VERSION = "4"                        # bump when the prompt/schema/model changes
STREAMS = ("btc", "usa500", "usatech")
BATCH = 10                           # headlines per LLM call (fixed-key schema forces completeness)
NUM_CTX = 8192

# Per-headline fields. Batching wraps N of these in a fixed-key object (n0..n{N-1}) whose
# keys are all "required", so grammar-constrained decoding MUST emit every headline — a
# variable-length array lets the model stop early (it truncated to ~2 items in testing).
_ITEM = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "number"},                                  # -1 bearish .. +1 bullish
        "relevance": {"type": "number"},                                  # 0 .. 1 to the asset's price
        "impact":    {"type": "string", "enum": ["low", "medium", "high"]},
        "asset":     {"type": "string", "enum": ["BTC", "US500", "USTECH", "macro", "other"]},
    },
    "required": ["sentiment", "relevance", "impact", "asset"],
}
SYSTEM = (
    "You are a financial-market news analyst. Score EACH numbered news item (n0, n1, …) for the "
    "market: sentiment (-1 very bearish to +1 very bullish), relevance (0 to 1: how much it "
    "could move the asset's price), impact (low/medium/high), asset (BTC/US500/USTECH/macro/"
    "other). Judge only from the headline; be conservative when unsure. "
    'Reply ONLY with a JSON object of the form {"n0": {"sentiment": 0.0, "relevance": 0.0, '
    '"impact": "low", "asset": "other"}, "n1": {...}, ...} — exactly one entry per headline, '
    "impact and asset only from the allowed values, no other text."
)
_IMPACT = {"low": 0, "medium": 1, "high": 2}
_ASSETS = {"BTC", "US500", "USTECH", "macro", "other"}


def _batch_schema(n: int) -> dict:
    keys = [f"n{i}" for i in range(n)]
    return {"type": "object", "properties": {k: _ITEM for k in keys}, "required": keys}


def _parse(o: dict) -> dict:
    asset = str(o["asset"])
    if asset not in _ASSETS:
        asset = "other"
    return {
        "llm_sent": max(-1.0, min(1.0, float(o["sentiment"]))),
        "llm_relevance": max(0.0, min(1.0, float(o["relevance"]))),
        "llm_impact": _IMPACT.get(str(o["impact"]).lower(), 0),
        "llm_asset": asset,
    }


# fallback for models that ignore the JSON-schema constraint (some cloud models) and reply
# in plain text like:  n0: sentiment=0.5, relevance=0.8, impact=medium, asset=BTC
_TEXT_LINE = re.compile(
    r"n(\d+)\s*[:.)]?\s*sentiment[\s=:]*([-+]?[\d.]+)[,;\s]+relevance[\s=:]*([\d.]+)"
    r"[,;\s]+impact[\s=:]*(low|medium|high)[,;\s]+asset[\s=:]*([A-Za-z0-9]+)",
    re.IGNORECASE,
)


def _parse_text(content: str, n: int) -> list[dict | None]:
    out: list[dict | None] = [None] * n
    for m in _TEXT_LINE.finditer(content):
        i = int(m.group(1))
        if 0 <= i < n:
            asset = m.group(5)
            if asset not in _ASSETS:
                asset = "other"
            out[i] = {
                "llm_sent": max(-1.0, min(1.0, float(m.group(2)))),
                "llm_relevance": max(0.0, min(1.0, float(m.group(3)))),
                "llm_impact": _IMPACT.get(m.group(4).lower(), 0),
                "llm_asset": asset,
            }
    return out


def _score_batch(titles: list[str], model: str = MODEL) -> list[dict | None]:
    """Score up to BATCH headlines in one call; returns aligned list (None where it failed)."""
    n = len(titles)
    lines = "\n".join(f"n{i}: {t}" for i, t in enumerate(titles))
    for _ in range(2):                                   # one retry on a bad/parse-failed call
        try:
            import ollama

            r = ollama.chat(
                model=model, format=_batch_schema(n),
                options={"temperature": 0, "num_ctx": NUM_CTX},
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": lines}],
            )
            content = r["message"]["content"]
            try:
                d = json.loads(content)
                out = [(_parse(d[f"n{i}"]) if f"n{i}" in d else None) for i in range(n)]
            except (json.JSONDecodeError, TypeError):
                out = _parse_text(content, n)            # model ignored the schema constraint
            if any(o is not None for o in out):
                return out
        except Exception:
            continue
    return [None] * n


def _paths(stream: str, source_prefix: str) -> tuple[Path, Path]:
    src = RAW_DIR / f"{source_prefix}_{stream}.parquet"
    out = RAW_DIR / (
        f"scores_llm_{stream}.parquet"
        if source_prefix == "gdelt"
        else f"scores_llm_{source_prefix}_{stream}.parquet"
    )
    return src, out


def score_stream(stream: str, workers: int = 2, limit: int | None = None,
                 model: str = MODEL, source_prefix: str = "gdelt") -> Path:
    """Score a stream's unique texts with the LLM and write/update its cache."""
    from sentiment.dedup import add_cluster_keys
    from sentiment.score import _norm

    src, out_path = _paths(stream, source_prefix)
    if not src.exists():
        raise FileNotFoundError(f"{src} not found - run the collector for source_prefix={source_prefix}")
    raw = pd.read_parquet(src)
    df = raw[["seendate", "url", "title"]].copy()
    df["score_text"] = raw["score_text"] if "score_text" in raw.columns else raw["title"]
    df["title_norm"] = _norm(df["score_text"])
    df = df[df["title_norm"] != ""]

    fields = ["llm_sent", "llm_relevance", "llm_impact", "llm_asset"]
    cached = pd.DataFrame(columns=["title_norm", *fields])
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        same = prev[(prev["model"] == model) & (prev["version"] == VERSION)
                    & prev["llm_sent"].notna()]
        cached = same[["title_norm", *fields]].drop_duplicates("title_norm")

    unique = add_cluster_keys(
        df[["title", "score_text", "title_norm"]].drop_duplicates("title_norm")
    )
    cmap = unique[["title_norm", "cluster_key"]]
    cached_cl = (cmap.merge(cached, on="title_norm")
                     .drop_duplicates("cluster_key")[["cluster_key", *fields]])

    todo = (unique[~unique["cluster_key"].isin(cached_cl["cluster_key"])]
            .assign(_len=lambda d: d["score_text"].str.len())
            .sort_values("_len")
            .drop_duplicates("cluster_key"))
    if limit:
        todo = todo.head(limit)
    print(f"[{source_prefix}:{stream}] {len(df):,} articles | {len(unique):,} unique | "
          f"{unique['cluster_key'].nunique():,} clusters | {len(todo):,} to score "
          f"({len(cached_cl):,} cached) | model={model} batch={BATCH} workers={workers}")

    fresh_rows = []
    if len(todo):
        from tqdm import tqdm

        texts = todo["score_text"].tolist()
        keys = todo["cluster_key"].tolist()
        batches = [texts[i:i + BATCH] for i in range(0, len(texts), BATCH)]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            batch_out = list(tqdm(ex.map(lambda b: _score_batch(b, model=model), batches),
                                  total=len(batches), desc=f"{source_prefix}:{stream}", unit="batch"))
        results = [r for batch in batch_out for r in batch]
        for key, res in zip(keys, results):
            if res is not None:
                fresh_rows.append({"cluster_key": key, **res})

    scored_cl = pd.concat([cached_cl, pd.DataFrame(fresh_rows)], ignore_index=True) \
                  .drop_duplicates("cluster_key") if fresh_rows else cached_cl
    art = df.merge(cmap, on="title_norm", how="left") \
            .merge(scored_cl, on="cluster_key", how="left") \
            .drop(columns=["cluster_key"])
    art["model"] = model
    art["version"] = VERSION
    art["scored_at"] = pd.Timestamp.now("UTC")
    art.to_parquet(out_path)
    ok = art["llm_sent"].notna().mean()
    print(f"[{source_prefix}:{stream}] wrote {len(art):,} articles ({ok:.0%} scored) -> {out_path}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stream", choices=STREAMS, help="score one stream (default: all)")
    ap.add_argument("--workers", type=int, default=2, help="concurrent LLM requests")
    ap.add_argument("--limit", type=int, help="only score the first N unseen headlines (testing)")
    ap.add_argument("--model", default=MODEL,
                    help="Ollama model tag (e.g. gemma4:12b, deepseek-v4-flash:cloud)")
    ap.add_argument("--source-prefix", default="gdelt", help="input prefix, e.g. gdelt or direct_events")
    args = ap.parse_args()
    for s in ([args.stream] if args.stream else STREAMS):
        score_stream(s, args.workers, args.limit, args.model, args.source_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
