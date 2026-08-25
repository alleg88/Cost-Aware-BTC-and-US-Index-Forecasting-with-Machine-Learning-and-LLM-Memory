"""Build frozen nine-model, causal market, and balanced-news agent inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from reflection_agent.config import ProtocolConfig, load_config
from reflection_agent.contracts import MODEL_IDS
from reflection_agent.manifest import build_manifest, sha256_payload, write_manifest
from reflection_agent.news import normalize_events

CODE_ROOT = Path(__file__).resolve().parents[1]
RAW_PREDICTION_ROOT = CODE_ROOT / "experiments" / "cache" / "tuning" / "all_model_sentiment_raw_180d_fixed15"
SENTIMENT_ROOT = CODE_ROOT / "sentiment" / "raw"
BAR_PATH = CODE_ROOT / "data" / "btcusdt_m15_2024_2025.parquet"
MINUTE_PATH = CODE_ROOT / "data" / "btcusdt_1m_2025_2026.parquet"
CONFIG_PATH = CODE_ROOT / "configs" / "reflection_agent.yaml"
DEFAULT_OUTPUT = CODE_ROOT / "experiments" / "cache" / "reflection_agent"
PROBABILITY_COLUMNS = ("p_short", "p_flat", "p_long")


def prediction_path(root: Path, model_id: str) -> Path:
    candidates = sorted((root / "none" / model_id / "stage_predictions" / "raw_forward").glob("w55*.parquet"))
    if len(candidates) != 1:
        raise ValueError(f"expected one DZ55 raw-forward prediction for {model_id}, found {len(candidates)}")
    return candidates[0]


def build_probability_panel(root: Path = RAW_PREDICTION_ROOT) -> tuple[pd.DataFrame, list[Path]]:
    panel = None
    sources = []
    reference_y = None
    for model_id in MODEL_IDS:
        path = prediction_path(root, model_id)
        sources.append(path)
        frame = pd.read_parquet(path).set_index("timestamp").sort_index()
        if frame.index.tz is None:
            raise ValueError(f"{model_id} timestamps must be timezone-aware")
        required_metadata = {"train_end", "test_start", "test_end", "refit_id"}
        missing_metadata = required_metadata.difference(frame.columns)
        if missing_metadata:
            raise ValueError(f"{model_id} prediction metadata missing: {sorted(missing_metadata)}")
        train_end = pd.to_datetime(frame["train_end"], utc=True)
        test_start = pd.to_datetime(frame["test_start"], utc=True)
        test_end = pd.to_datetime(frame["test_end"], utc=True)
        if not (
            (train_end < test_start).all()
            and (frame.index >= pd.DatetimeIndex(test_start)).all()
            and (frame.index < pd.DatetimeIndex(test_end)).all()
            and frame["refit_id"].nunique() == 1
        ):
            raise ValueError(f"{model_id} raw-forward predictions are not one frozen causal fit")
        if set(frame["width_bps"].astype(int)) != {55}:
            raise ValueError(f"{model_id} dead zone changed")
        probabilities = frame[list(PROBABILITY_COLUMNS)].astype(float)
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError(f"{model_id} probabilities do not sum to one")
        if panel is None:
            panel = pd.DataFrame(index=frame.index)
            panel["y_true"] = frame["y_true"].astype(int)
            reference_y = panel["y_true"]
        elif not frame.index.equals(panel.index) or not frame["y_true"].astype(int).equals(reference_y):
            raise ValueError(f"{model_id} prediction index or labels differ")
        for column in PROBABILITY_COLUMNS:
            panel[f"{model_id}_{column}"] = probabilities[column]
    if panel is None:
        raise ValueError("empty probability panel")
    return panel.drop(columns=["y_true"]), sources


def _publisher_categories(source: pd.Series) -> pd.Series:
    return source.astype(str).map({"truth_social": "political_official", "fed": "central_bank_official"}).fillna("official")


def build_news_events(root: Path = SENTIMENT_ROOT) -> tuple[pd.DataFrame, list[Path]]:
    paths = [
        root / "scores_llm_btc.parquet",
        root / "direct_events_btc.parquet",
        root / "scores_llm_direct_events_btc.parquet",
        root / "fred_calendar.parquet",
        root / "crypto_fear_greed.parquet",
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing news inputs: {missing}")

    scored_news = pd.read_parquet(paths[0])
    gdelt = normalize_events(
        scored_news,
        source_family="gdelt_news",
        available_column="seendate",
        summary_column="title",
        publisher_category="news_publisher",
        impact_column="llm_impact",
        sentiment_column="llm_sent",
    )

    direct = pd.read_parquet(paths[1])
    direct_scores = pd.read_parquet(paths[2])[["seendate", "url", "llm_sent", "llm_impact"]]
    direct = direct.merge(direct_scores, on=["seendate", "url"], how="left", validate="one_to_one")
    normalized_direct = []
    for category, group in direct.assign(publisher_category=_publisher_categories(direct["source"])).groupby("publisher_category"):
        normalized_direct.append(normalize_events(
            group,
            source_family="direct_policy_event",
            available_column="available_at",
            summary_column="score_text",
            publisher_category=str(category),
            impact_column="llm_impact",
            sentiment_column="llm_sent",
        ))

    fred = pd.read_parquet(paths[3]).copy()
    fred["summary"] = fred.apply(lambda row: f"{row['label']}: {row['value']}", axis=1)
    fred["impact"] = 1.0
    fred_events = normalize_events(
        fred,
        source_family="fred_macro",
        available_column="release_time",
        summary_column="summary",
        publisher_category="macro_release",
        impact_column="impact",
    )

    fear = pd.read_parquet(paths[4]).reset_index().rename(columns={"timestamp": "available_at"})
    fear["summary"] = fear.apply(lambda row: f"Fear and Greed {row['classification']}: {row['value']}", axis=1)
    fear["sentiment"] = ((fear["value"].astype(float) - 50.0) / 50.0).clip(-1.0, 1.0)
    fear["impact"] = fear["value"].astype(float).diff().abs().fillna(0.0).div(25.0).clip(0.0, 3.0)
    fear_events = normalize_events(
        fear,
        source_family="fear_greed",
        available_column="available_at",
        summary_column="summary",
        publisher_category="market_index",
        impact_column="impact",
        sentiment_column="sentiment",
    )
    events = pd.concat([gdelt, *normalized_direct, fred_events, fear_events], ignore_index=True)
    events = events.sort_values(["available_at_utc", "event_id"], kind="mergesort").drop_duplicates("event_id")
    return events.reset_index(drop=True), paths


def _hour_block(hour: int) -> str:
    if hour < 8:
        return "asia"
    if hour < 16:
        return "europe"
    return "us"


def build_market_context(
    panel: pd.DataFrame,
    bars: pd.DataFrame,
    events: pd.DataFrame,
    config: ProtocolConfig,
) -> pd.DataFrame:
    bars = bars.reindex(panel.index)
    if bars[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("market bars do not cover the frozen panel")
    returns = bars["close"].pct_change()
    recent = bars["close"].pct_change(config.market.recent_return_bars).fillna(0.0)
    realized = returns.rolling(config.market.realized_volatility_bars, min_periods=96).std().mul(np.sqrt(35_040))
    low = realized.rolling(config.market.volatility_reference_bars, min_periods=672).quantile(1 / 3).shift(1)
    high = realized.rolling(config.market.volatility_reference_bars, min_periods=672).quantile(2 / 3).shift(1)
    vol_regime = pd.Series("normal", index=panel.index)
    vol_regime.loc[realized < low] = "low"
    vol_regime.loc[realized > high] = "high"
    trend = pd.Series("flat", index=panel.index)
    trend.loc[recent > config.market.trend_threshold] = "up"
    trend.loc[recent < -config.market.trend_threshold] = "down"

    model_predictions = pd.DataFrame({
        model_id: panel[[f"{model_id}_{column}" for column in PROBABILITY_COLUMNS]].to_numpy().argmax(axis=1)
        for model_id in MODEL_IDS
    }, index=panel.index)
    mean_probabilities = np.mean(np.stack([
        panel[[f"{model_id}_{column}" for column in PROBABILITY_COLUMNS]].to_numpy()
        for model_id in MODEL_IDS
    ]), axis=0)
    ensemble_prediction = mean_probabilities.argmax(axis=1)
    ensemble_confidence = mean_probabilities.max(axis=1)
    agreement = (model_predictions.to_numpy() == ensemble_prediction[:, None]).mean(axis=1)

    event_time = pd.to_datetime(events["available_at_utc"], utc=True).dt.ceil("15min")
    event_bins = events.assign(bin=event_time).groupby("bin").agg(
        impact=("impact", "max"), sentiment=("sentiment", "mean")
    ).reindex(panel.index)
    event_bins = event_bins.fillna(0.0)
    news_impact = event_bins["impact"].rolling(config.market.news_lookback_bars, min_periods=1).max()
    news_dispersion = event_bins["sentiment"].rolling(config.market.news_lookback_bars, min_periods=1).std(ddof=0)

    return pd.DataFrame({
        "vol_regime": vol_regime,
        "trend_regime": trend,
        "realized_volatility": realized.fillna(0.0),
        "recent_return": recent,
        "model_disagreement": 1.0 - agreement,
        "ensemble_confidence": ensemble_confidence,
        "news_impact": news_impact,
        "news_dispersion": news_dispersion,
        "data_quality_state": "ok",
        "hour_block": [_hour_block(timestamp.hour) for timestamp in panel.index],
        "day_of_week": [timestamp.day_name()[:3] for timestamp in panel.index],
    }, index=panel.index)


def validate_boundaries(frame: pd.DataFrame, config: ProtocolConfig) -> None:
    index = frame.index if isinstance(frame.index, pd.DatetimeIndex) else pd.DatetimeIndex(
        pd.to_datetime(frame["available_at_utc"], utc=True)
    )
    if index.tz is None:
        raise ValueError("artifact timestamps must be timezone-aware")
    if len(index) and index.max() >= pd.Timestamp(config.sealed_start_utc):
        raise ValueError("artifact enters the 2026-Q2 lockbox")


def build_cache(
    output_root: Path = DEFAULT_OUTPUT,
    *,
    config_path: Path = CONFIG_PATH,
    allow_replace_manifest: bool = False,
) -> dict[str, object]:
    config = load_config(config_path)
    panel, prediction_sources = build_probability_panel()
    start = pd.Timestamp(config.development_start_utc)
    end = pd.Timestamp(config.development_end_utc)
    panel = panel.loc[(panel.index >= start) & (panel.index < end)]
    events, news_sources = build_news_events()
    events = events.loc[pd.to_datetime(events["available_at_utc"], utc=True) < end].reset_index(drop=True)
    bars = pd.read_parquet(BAR_PATH).sort_index().loc[start:end - pd.Timedelta(minutes=15)]
    context = build_market_context(panel, bars, events, config)
    validate_boundaries(panel, config)
    validate_boundaries(context, config)
    validate_boundaries(events, config)

    output_root.mkdir(parents=True, exist_ok=True)
    panel.rename_axis("timestamp").reset_index().to_parquet(output_root / "frozen_probability_panel.parquet", index=False)
    context.rename_axis("timestamp").reset_index().to_parquet(output_root / "market_context.parquet", index=False)
    events.to_parquet(output_root / "news_events.parquet", index=False)
    manifest = build_manifest(
        config,
        code_root=CODE_ROOT,
        input_paths=[config_path, BAR_PATH, MINUTE_PATH, *prediction_sources, *news_sources],
        output_mode="schema",
    )
    manifest.pop("protocol_hash")
    manifest["artifacts"] = {
        "probability_rows": len(panel),
        "context_rows": len(context),
        "news_rows": len(events),
        "start_utc": panel.index.min().isoformat(),
        "end_utc_exclusive": end.isoformat(),
        "direct_policy_events": int((events["source_family"] == "direct_policy_event").sum()),
        "political_official_events": int((events["publisher_category"] == "political_official").sum()),
    }
    manifest["protocol_hash"] = sha256_payload(manifest)
    write_manifest(
        output_root / "protocol_manifest.json", manifest, allow_replace=allow_replace_manifest
    )
    return manifest


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force-manifest", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_cache(
        args.output_root,
        config_path=args.config,
        allow_replace_manifest=args.force_manifest,
    )
    print(json.dumps({"protocol_hash": manifest["protocol_hash"], **manifest["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
