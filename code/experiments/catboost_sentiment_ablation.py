"""Matched CatBoost sentiment study for Notebook 02c.

Classic (financial DeBERTa) and LLM arms reuse the exact Notebook 02b baseline
CatBoost candidate, DZ55/DZ65/DZ75, 180-day histories, 2024 blocking
folds, H1 monthly policy calibration and frozen-forward execution. Both arms
receive the same six sentiment columns; only two scalar sentiment values differ.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from experiments.catboost_matched_ablation import WIDTHS, load_candidates
from experiments.notebook02_handoff import PIPELINE_HANDOFF, load_pipeline_handoff
from experiments.run_catboost_matched_ablation import (
    MatchedAblationRunner,
    PreparedData,
    load_prepared_data,
)
from features.sentiment import (
    DIRECT_EVENT_FEATURES,
    LLM_FULL_FEATURES_BTC,
    MATCHED_SENTIMENT_FEATURES_BTC,
    build_direct_event_block,
    build_llm_full_features,
    build_matched_sentiment_features,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    CODE_ROOT / "experiments" / "cache" / "tuning" / "catboost_sentiment_matched_180d"
)
LOOKBACK_DAYS = {55: 180, 65: 180, 75: 180}

# Each sentiment arm and the exact feature block it is allowed to build:
#   classic   DeBERTa scalar, filtered and weighted by DeBERTa's own outputs
#   llm       LLM scalar, everything unweighted — the like-for-like control
#   llm_full  the LLM's structured output kept as its own channels
ARM_SPECS = {
    "classic": {"scorer": "classic", "weighting": "own", "direct": True,
                "schema": (*MATCHED_SENTIMENT_FEATURES_BTC, *DIRECT_EVENT_FEATURES)},
    "llm": {"scorer": "llm", "weighting": "plain", "direct": True,
            "schema": (*MATCHED_SENTIMENT_FEATURES_BTC, *DIRECT_EVENT_FEATURES)},
    "llm_full": {"scorer": "llm", "weighting": "own", "direct": False,
                 "schema": tuple(LLM_FULL_FEATURES_BTC)},   # builder adds them itself
}
SCORERS = tuple(ARM_SPECS)


def arm_output_root(scorer: str, output_root: Path = DEFAULT_ROOT) -> Path:
    if scorer not in SCORERS:
        raise ValueError(f"unknown scorer: {scorer}")
    return Path(output_root) / scorer


def prepare_arm_data(
    scorer: str,
    *,
    expected_base_columns: Sequence[str] | None = None,
) -> PreparedData:
    """Join one matched sentiment arm to the validated no-sentiment features."""
    if scorer not in ARM_SPECS:
        raise ValueError(f"unknown scorer: {scorer}")
    spec = ARM_SPECS[scorer]
    base = load_prepared_data(widths=WIDTHS, sentiment="none")
    actual_base = tuple(next(iter(base.features.values()))[0].columns)
    if expected_base_columns is not None and actual_base != tuple(expected_base_columns):
        raise ValueError("base feature schema does not match Notebook 02 handoff")
    if scorer == "llm_full":
        sentiment = build_llm_full_features("btc", base.bars.index)
    else:
        sentiment = build_matched_sentiment_features(
            "btc", base.bars.index, scorer=spec["scorer"], weighting=spec["weighting"]
        )
        if spec["direct"]:
            sentiment = pd.concat([sentiment, build_direct_event_block(
                "btc", base.bars.index, scorer=spec["scorer"],
                weighting=spec["weighting"])], axis=1)
    if tuple(sentiment.columns) != spec["schema"]:
        raise ValueError(f"sentiment schema for arm {scorer!r} changed")

    features = {}
    for width, (X, y) in base.features.items():
        joined = X.join(sentiment.reindex(X.index), validate="one_to_one")
        if joined[list(spec["schema"])].isna().any().any():
            raise ValueError("matched sentiment join introduced missing values")
        features[int(width)] = (joined, y.reindex(joined.index))
    columns = tuple(next(iter(features.values()))[0].columns)
    return PreparedData(
        bars=base.bars,
        minute=base.minute,
        features=features,
        regimes=base.regimes,
        m15_fingerprint=base.m15_fingerprint,
        minute_fingerprint=base.minute_fingerprint,
        sentiment_mode=f"matched_{scorer}",
        feature_columns=columns,
    )


def run_arm(
    scorer: str,
    *,
    output_root: Path = DEFAULT_ROOT,
    smoke: bool = False,
    stage1_only: bool = False,
) -> dict:
    """Run one scorer with candidate 0 and the frozen Notebook 02 histories."""
    handoff = load_pipeline_handoff(PIPELINE_HANDOFF)
    expected_widths = tuple(int(width) for width in handoff["widths"])
    if expected_widths != tuple(WIDTHS):
        raise ValueError("Notebook 01 widths do not match the sentiment study")
    if {int(k): int(v) for k, v in handoff["training_histories_days"].items()} != LOOKBACK_DAYS:
        raise ValueError("Notebook 01 training histories must all be 180 days")
    prepared = prepare_arm_data(
        scorer,
        expected_base_columns=handoff["features"]["columns"],
    )
    runner = MatchedAblationRunner(
        output_root=arm_output_root(scorer, output_root),
        widths=WIDTHS,
        candidates=(load_candidates()[0],),
        candidate_ids=(0,),
        prepared=prepared,
        smoke=smoke,
        stage1_only=stage1_only,
    )
    runner.lookback_days = LOOKBACK_DAYS.copy()
    runner.sentiment_mode = f"matched_{scorer}"
    runner.upstream_handoff_fingerprint = str(handoff["handoff_fingerprint"])
    return runner.run()


def _write_state(output_root: Path, payload: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_state.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", choices=(*SCORERS, "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage1-only", action="store_true")
    args = parser.parse_args()
    scorers = SCORERS if args.scorer == "all" else (args.scorer,)
    state = {"status": "running", "completed": [], "active": None}
    _write_state(args.output_root, state)
    try:
        for scorer in scorers:
            state["active"] = scorer
            _write_state(args.output_root, state)
            run_arm(
                scorer,
                output_root=args.output_root,
                smoke=args.smoke,
                stage1_only=args.stage1_only,
            )
            state["completed"].append(scorer)
        state.update(status="complete", active=None)
        _write_state(args.output_root, state)
    except Exception as exc:
        state.update(status="failed", error=repr(exc))
        _write_state(args.output_root, state)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
