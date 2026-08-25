"""Score collected news headlines with a financial-sentiment model and cache the result.

Step 2c/2d of the sentiment layer (see papers/wiki/design/news-scoring-howto.md). Reads a
source stream parquet (``gdelt_<stream>.parquet`` or ``direct_events_<stream>.parquet``), scores each
UNIQUE headline once, and writes an article-level cache ``scores_<stream>.parquet``.

The model is a DeBERTa-v3 fine-tuned on financial news (labels negative/neutral/positive).
The single signed feature is ``sent = P(positive) - P(negative)`` in [-1, 1]. Headlines are
deduplicated (normalised text) before scoring so syndicated copies are scored once; the
score is then mapped back to every article that shares the headline.

Caching is incremental and reproducible: results are stamped with model + version, and a
re-run only scores headlines not already in the cache.

Run:  python sentiment/score.py --stream btc     (default: all streams)
      python sentiment/score.py --stream btc --source-prefix direct_events
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = CODE_ROOT / "sentiment" / "raw"

MODEL = "mrm8488/deberta-v3-ft-financial-news-sentiment-analysis"
REVISION = "9e10915c245a80a89b18d1ac51350e093c7bb35a"
VERSION = "1"                       # bump when the model or scoring logic changes
STREAMS = ("btc", "usa500", "usatech")
BATCH_SIZE = 64


def _norm(title: pd.Series) -> pd.Series:
    """Normalise a headline for dedup: lowercase, collapse whitespace, strip."""
    return (
        title.fillna("").astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _build_scorer(
    batch_size: int,
    *,
    revision: str = REVISION,
    local_files_only: bool = False,
):
    """Return a callable list[str] -> list[signed sentiment] using DeBERTa-finance."""
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    print(f"  scoring on {'GPU (cuda:0)' if device == 0 else 'CPU'}")
    clf = pipeline(
        "text-classification", model=MODEL, top_k=None,   # return all class scores
        device=device, batch_size=batch_size, truncation=True, max_length=128,
        revision=revision, local_files_only=bool(local_files_only),
    )

    def score(texts: list[str]) -> list[float]:
        out = []
        for row in clf(texts):
            probs = {d["label"].lower(): d["score"] for d in row}
            out.append(float(probs.get("positive", 0.0) - probs.get("negative", 0.0)))
        return out

    return score


def _paths(stream: str, source_prefix: str) -> tuple[Path, Path]:
    src = RAW_DIR / f"{source_prefix}_{stream}.parquet"
    out = RAW_DIR / (
        f"scores_{stream}.parquet"
        if source_prefix == "gdelt"
        else f"scores_{source_prefix}_{stream}.parquet"
    )
    return src, out


def score_stream(stream: str, batch_size: int = BATCH_SIZE, source_prefix: str = "gdelt") -> Path:
    """Score one stream's unique texts and write/update its article-level cache."""
    src, out_path = _paths(stream, source_prefix)
    if not src.exists():
        raise FileNotFoundError(f"{src} not found - run the collector for source_prefix={source_prefix}")
    raw = pd.read_parquet(src)
    df = raw[["seendate", "url", "title"]].copy()
    df["score_text"] = raw["score_text"] if "score_text" in raw.columns else raw["title"]
    df["title_norm"] = _norm(df["score_text"])
    df = df[df["title_norm"] != ""]

    cached = pd.DataFrame(columns=["title_norm", "sent"])
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        required = {"title_norm", "sent", "model", "revision", "version"}
        if required.issubset(prev.columns):
            same = prev[
                (prev["model"] == MODEL)
                & (prev["revision"] == REVISION)
                & (prev["version"] == VERSION)
                & prev["sent"].notna()
            ]
            cached = same[["title_norm", "sent"]].drop_duplicates("title_norm")

    unique = df[["score_text", "title_norm"]].drop_duplicates("title_norm")
    todo = unique[~unique["title_norm"].isin(cached["title_norm"])]
    print(f"[{source_prefix}:{stream}] {len(df):,} articles | {len(unique):,} unique texts | "
          f"{len(todo):,} to score ({len(cached):,} cached)")

    scored = cached
    if len(todo):
        scorer = _build_scorer(batch_size)
        sents = scorer(todo["score_text"].tolist())
        fresh = pd.DataFrame({"title_norm": todo["title_norm"].to_numpy(), "sent": sents})
        scored = pd.concat([cached, fresh], ignore_index=True).drop_duplicates("title_norm")

    art = df.merge(scored, on="title_norm", how="left")
    art["model"] = MODEL
    art["revision"] = REVISION
    art["version"] = VERSION
    art["scored_at"] = pd.Timestamp.now("UTC")
    art = art[["seendate", "url", "title", "score_text", "title_norm", "sent", "model", "revision", "version", "scored_at"]]
    art.to_parquet(out_path)
    print(f"[{source_prefix}:{stream}] wrote {len(art):,} scored articles "
          f"(sent mean {art['sent'].mean():+.3f}) -> {out_path}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stream", choices=STREAMS, help="score one stream (default: all)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--source-prefix", default="gdelt", help="input prefix, e.g. gdelt or direct_events")
    args = ap.parse_args()
    for s in ([args.stream] if args.stream else STREAMS):
        score_stream(s, args.batch_size, args.source_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
