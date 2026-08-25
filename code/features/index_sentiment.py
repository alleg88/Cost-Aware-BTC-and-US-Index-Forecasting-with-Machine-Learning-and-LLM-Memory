"""Causal matched sentiment features for USA500 and USATECH replications.

Both matched arms consume the same successfully scored rows, aggregation and
five-column schema. Each scorer uses only its own outputs. The full DeepSeek
arm adds three structured channels to the exact DeepSeek matched base.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.final_q2_lockbox_contract import Q2_END, Q2_START
from experiments.final_q2_lockbox_state import OpeningIdentity, require_global_opening


CODE_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = CODE_ROOT / "sentiment" / "raw"
FREQUENCY = "15min"
BARS_PER_HOUR = 4
MATCHED_FEATURES = (
    "sent_news_decay",
    "sent_news_count_24h",
    "sent_direct_decay",
    "sent_tone_decay",
    "sent_macro_decay",
)
DEEPSEEK_STRUCTURED_FEATURES = (
    "sent_llm_relevance_decay",
    "sent_llm_hi_impact_decay",
    "sent_llm_topic_share_24h",
)
DEEPSEEK_FULL_FEATURES = (*MATCHED_FEATURES, *DEEPSEEK_STRUCTURED_FEATURES)
_ASSET = {"usa500": "US500", "usatech": "USTECH"}
_IMPACT_WEIGHT = {0: 1.0, 1: 2.0, 2: 3.0}
_IDENTITY_FIELDS = (
    "tag", "digest", "remote_model", "remote_host", "registry_digest_prefix",
    "prompt_hash", "schema_hash", "temperature", "think", "num_ctx",
    "batch_size", "batch_protocol", "scorer_implementation_hash",
)


def _grid_aggregate(times, values, bar_index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    events = pd.DataFrame(
        {
            "time": pd.to_datetime(pd.Series(times), utc=True),
            "value": np.asarray(values, dtype=float),
        }
    ).dropna(subset=["time"])
    if events.empty:
        grid = pd.date_range(bar_index.min(), bar_index.max(), freq=FREQUENCY, tz="UTC")
        return pd.DataFrame({"sum": 0.0, "count": 0.0}, index=grid), grid
    events["bin"] = events["time"].dt.floor(FREQUENCY)
    aggregate = events.groupby("bin")["value"].agg(**{"sum": "sum", "count": "count"})
    start = min(aggregate.index.min(), bar_index.min())
    end = max(aggregate.index.max(), bar_index.max())
    grid = pd.date_range(start, end, freq=FREQUENCY, tz="UTC")
    return aggregate.reindex(grid, fill_value=0.0), grid


def _window_features(
    times,
    values,
    bar_index: pd.DatetimeIndex,
    prefix: str,
    *,
    windows_h: tuple[int, ...] = (24,),
    halflife_h: float = 6.0,
    include_mean: bool = True,
    include_count: bool = True,
) -> pd.DataFrame:
    aggregate, grid = _grid_aggregate(times, values, bar_index)
    output = {}
    for hours in windows_h:
        window = hours * BARS_PER_HOUR
        count = aggregate["count"].rolling(window, min_periods=1).sum()
        total = aggregate["sum"].rolling(window, min_periods=1).sum()
        if include_count:
            output[f"{prefix}_count_{hours}h"] = count
        if include_mean:
            output[f"{prefix}_mean_{hours}h"] = np.where(
                count > 0, total / count.replace(0, np.nan), 0.0
            )
    output[f"{prefix}_decay"] = aggregate["sum"].ewm(
        halflife=halflife_h * BARS_PER_HOUR, adjust=False
    ).mean()
    return pd.DataFrame(output, index=grid).reindex(bar_index).fillna(0.0)


def _decayed_weighted_mean(
    times,
    values,
    weights,
    bar_index: pd.DatetimeIndex,
    *,
    halflife_h: float,
) -> pd.Series:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    numerator, grid = _grid_aggregate(times, values * weights, bar_index)
    denominator, _ = _grid_aggregate(times, weights, bar_index)
    span = halflife_h * BARS_PER_HOUR
    numerator_decay = numerator["sum"].ewm(halflife=span, adjust=False).mean()
    denominator_decay = denominator["sum"].ewm(halflife=span, adjust=False).mean()
    output = np.where(
        denominator_decay.to_numpy() > 1e-12,
        numerator_decay.to_numpy() / denominator_decay.replace(0.0, np.nan).to_numpy(),
        0.0,
    )
    return pd.Series(output, index=grid).reindex(bar_index).fillna(0.0)


def _first_seen_with_echoes(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep causal primary stories; never weight them by later echoes."""
    from sentiment.dedup import first_seen_only

    return first_seen_only(frame).assign(echo_count=1)


def _score_paths(
    stream: str, *, direct: bool, score_root: str | Path | None = None
) -> tuple[Path, Path]:
    if stream not in _ASSET:
        raise ValueError(f"unsupported index stream: {stream}")
    tag = "_direct_events" if direct else ""
    root = RAW_DIR if score_root is None else Path(score_root)
    return (
        root / f"scores{tag}_{stream}.parquet",
        root / f"scores_llm{tag}_{stream}.parquet",
    )


def _read_json(path: Path) -> dict:
    import json

    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_score_manifests(
    stream: str,
    *,
    direct: bool,
    score_root: str | Path | None = None,
    available_start: pd.Timestamp | None = None,
    available_end: pd.Timestamp,
    opening_identity: OpeningIdentity | None = None,
) -> None:
    if available_end > Q2_START or (
        available_start is not None and available_start >= Q2_START
    ):
        if available_start != Q2_START or available_end != Q2_END:
            raise PermissionError("sentiment feature interval differs from Q2")
        if opening_identity is None:
            raise PermissionError("Q2 sentiment features require OPENED identity")
        require_global_opening(opening_identity)
    classic_path, llm_path = _score_paths(
        stream, direct=direct, score_root=score_root
    )
    classic = _read_json(classic_path.with_suffix(".manifest.json"))
    llm = _read_json(llm_path.with_suffix(".manifest.json"))
    identity = _read_json(RAW_DIR / "index_deepseek_identity.json").get("identity", {})
    prefix = "direct_events" if direct else "gdelt"
    q2_interval = available_start == Q2_START and available_end == Q2_END
    manifest_end = Q2_END if q2_interval else Q2_START
    for name, manifest in (("DeBERTa", classic), ("DeepSeek", llm)):
        if manifest.get("complete") is not True:
            raise ValueError(f"{name} {prefix} manifest is not complete")
        if manifest.get("stream") != stream or manifest.get("source_prefix") != prefix:
            raise ValueError(f"{name} manifest stream/source changed")
        if manifest.get("cutoff_exclusive") != manifest_end.isoformat():
            raise ValueError(f"{name} manifest cutoff changed")
        expected_start = (
            None if available_start is None else available_start.isoformat()
        )
        if manifest.get("start_inclusive") not in {None, expected_start}:
            raise ValueError(f"{name} manifest lower boundary changed")
        expected_scope = (
            "canonical_pre_cutoff_scoring_rows"
            if not q2_interval
            else "canonical_interval_scoring_rows"
        )
        if manifest.get("source_sha256_scope") != expected_scope:
            raise ValueError(f"{name} manifest source scope changed")
        if q2_interval and name == "DeBERTa":
            from sentiment.index_scoring import DEBERTA_REVISION

            if (
                manifest.get("revision") != DEBERTA_REVISION
                or manifest.get("local_files_only") is not True
            ):
                raise ValueError("DeBERTa Q2 score was not pinned to the local revision")
    if classic.get("source_sha256") != llm.get("source_sha256"):
        raise ValueError("DeBERTa and DeepSeek used different raw scoring snapshots")
    frozen = llm.get("identity", {})
    if any(frozen.get(field) != identity.get(field) for field in _IDENTITY_FIELDS):
        raise ValueError("DeepSeek score manifest identity differs from frozen preflight")


def _read_score(
    path: Path,
    columns: list[str],
    *,
    available_start: pd.Timestamp | None,
    available_end: pd.Timestamp,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    filters = [("seendate", "<", available_end.to_pydatetime())]
    if available_start is not None:
        filters.insert(0, ("seendate", ">=", available_start.to_pydatetime()))
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=filters,
    )
    frame["seendate"] = pd.to_datetime(frame["seendate"], utc=True)
    if len(frame) and frame["seendate"].ge(available_end).any():
        raise AssertionError(f"{path.name} crossed the feature boundary")
    if (
        available_start is not None
        and len(frame)
        and frame["seendate"].lt(available_start).any()
    ):
        raise AssertionError(f"{path.name} crossed the feature lower boundary")
    return frame


def _load_matched_scores(
    stream: str,
    scorer: str,
    *,
    direct: bool,
    score_root: str | Path | None = None,
    available_start: pd.Timestamp | None = None,
    available_end: pd.Timestamp,
    opening_identity: OpeningIdentity | None = None,
) -> pd.DataFrame:
    if scorer not in {"classic", "llm"}:
        raise ValueError("scorer must be classic or llm")
    _validate_score_manifests(
        stream,
        direct=direct,
        score_root=score_root,
        available_start=available_start,
        available_end=available_end,
        opening_identity=opening_identity,
    )
    keys = ["seendate", "url", "title"]
    classic_path, llm_path = _score_paths(
        stream, direct=direct, score_root=score_root
    )
    classic = _read_score(
        classic_path,
        [*keys, "sent"],
        available_start=available_start,
        available_end=available_end,
    ).dropna(subset=["sent"])
    llm_fields = ["llm_relevance", "llm_impact", "llm_asset"]
    llm = _read_score(
        llm_path,
        [*keys, "llm_sent", *llm_fields],
        available_start=available_start,
        available_end=available_end,
    ).dropna(subset=["llm_sent"])
    matched = _first_seen_with_echoes(
        classic.merge(llm, on=keys, how="inner", validate="one_to_one")
    )
    score_column = "llm_sent" if scorer == "llm" else "sent"
    return matched[[*keys, score_column, "echo_count", *llm_fields]].rename(
        columns={score_column: "matched_sentiment"}
    )


def _own_weights(
    frame: pd.DataFrame, *, scorer: str, stream: str
) -> tuple[np.ndarray, np.ndarray]:
    if scorer == "classic":
        keep = np.ones(len(frame), dtype=bool)
        confidence = frame["matched_sentiment"].abs().clip(0.0, 1.0)
        weight = confidence * (1.0 + np.log1p(frame["echo_count"].astype(float)))
    else:
        keep = frame["llm_asset"].isin([_ASSET[stream], "macro"]).to_numpy()
        relevance = frame["llm_relevance"].astype(float).clip(0.0, 1.0)
        impact = frame["llm_impact"].map(_IMPACT_WEIGHT).fillna(1.0)
        weight = relevance * impact
    return keep, weight.astype(float).fillna(0.0).to_numpy()


def _matched_decay(
    stream: str,
    bar_index: pd.DatetimeIndex,
    *,
    scorer: str,
    direct: bool,
    halflife_h: float,
    score_root: str | Path | None = None,
    warmup_score_root: str | Path | None = None,
    available_start: pd.Timestamp | None = None,
    available_end: pd.Timestamp | None = None,
    opening_identity: OpeningIdentity | None = None,
) -> tuple[pd.Series, pd.Series | None]:
    feature_end = (
        bar_index.max() + pd.Timedelta(minutes=15)
        if available_end is None
        else pd.Timestamp(available_end)
    )
    scored = _load_matched_scores(
        stream,
        scorer,
        direct=direct,
        score_root=score_root,
        available_start=available_start,
        available_end=feature_end,
        opening_identity=opening_identity,
    )
    if warmup_score_root is not None:
        if available_start != Q2_START or feature_end != Q2_END:
            raise PermissionError("sentiment warm-up context is only valid for exact Q2")
        warmup = _load_matched_scores(
            stream,
            scorer,
            direct=direct,
            score_root=warmup_score_root,
            available_start=None,
            available_end=Q2_START,
            opening_identity=None,
        )
        scored = _first_seen_with_echoes(
            pd.concat([warmup, scored], ignore_index=True, sort=False).sort_values(
                "seendate"
            )
        )
    keep, weight = _own_weights(scored, scorer=scorer, stream=stream)
    retained = scored.loc[keep]
    decay = _decayed_weighted_mean(
        retained["seendate"],
        retained["matched_sentiment"],
        weight[keep],
        bar_index,
        halflife_h=halflife_h,
    )
    if direct:
        return decay, None
    count = _window_features(
        retained["seendate"],
        retained["matched_sentiment"],
        bar_index,
        "sent_news",
        windows_h=(24,),
        halflife_h=halflife_h,
        include_mean=False,
        include_count=True,
    )["sent_news_count_24h"]
    return decay, count


def _tone_and_macro(
    stream: str,
    bar_index: pd.DatetimeIndex,
    *,
    source_root: str | Path | None = None,
    available_start: pd.Timestamp | None = None,
    available_end: pd.Timestamp | None = None,
    opening_identity: OpeningIdentity | None = None,
    continuous_context: bool = False,
) -> tuple[pd.Series, pd.Series]:
    feature_end = (
        bar_index.max() + pd.Timedelta(minutes=15)
        if available_end is None
        else pd.Timestamp(available_end)
    )
    if feature_end > Q2_START or (
        available_start is not None and available_start >= Q2_START
    ):
        if available_start != Q2_START or feature_end != Q2_END:
            raise PermissionError("tone/macro feature interval differs from Q2")
        if opening_identity is None:
            raise PermissionError("Q2 tone/macro features require OPENED identity")
        require_global_opening(opening_identity)
    root = RAW_DIR if source_root is None else Path(source_root)
    tone_path = root / f"gdelt_{stream}.parquet"
    context_start = None if continuous_context else available_start
    tone_filters = [("seendate", "<", feature_end.to_pydatetime())]
    if context_start is not None:
        tone_filters.insert(0, ("seendate", ">=", context_start.to_pydatetime()))
    tone = pd.read_parquet(
        tone_path,
        columns=["seendate", "title", "tone"],
        filters=tone_filters,
    ).dropna(subset=["tone"])
    tone["seendate"] = pd.to_datetime(tone["seendate"], utc=True)
    if len(tone) and tone["seendate"].ge(feature_end).any():
        raise AssertionError("GDELT tone crossed the feature boundary")
    if context_start is not None and len(tone) and tone["seendate"].lt(context_start).any():
        raise AssertionError("GDELT tone crossed the lower feature boundary")
    from sentiment.dedup import first_seen_only

    tone = first_seen_only(tone)
    tone_decay = _window_features(
        tone["seendate"],
        tone["tone"],
        bar_index,
        "tone",
        windows_h=(24,),
        include_mean=True,
        include_count=False,
    )["tone_decay"]
    macro_filters = [("release_time", "<", feature_end.to_pydatetime())]
    if context_start is not None:
        macro_filters.insert(0, ("release_time", ">=", context_start.to_pydatetime()))
    macro = pd.read_parquet(
        root / "fred_calendar.parquet",
        columns=["release_time"],
        filters=macro_filters,
    ).dropna(subset=["release_time"])
    macro_time = pd.to_datetime(macro["release_time"], utc=True)
    if len(macro_time) and macro_time.ge(feature_end).any():
        raise AssertionError("FRED release crossed the feature boundary")
    if context_start is not None and len(macro_time) and macro_time.lt(context_start).any():
        raise AssertionError("FRED release crossed the lower feature boundary")
    macro_decay = _window_features(
        macro_time,
        np.ones(len(macro)),
        bar_index,
        "macro",
        windows_h=(24,),
        include_mean=False,
        include_count=True,
    )["macro_decay"]
    return tone_decay, macro_decay


def build_matched_index_features(
    stream: str,
    bar_index: pd.DatetimeIndex,
    *,
    scorer: str,
    halflife_h: float = 6.0,
    score_root: str | Path | None = None,
    warmup_score_root: str | Path | None = None,
    source_root: str | Path | None = None,
    available_start: pd.Timestamp | None = None,
    available_end: pd.Timestamp | None = None,
    opening_identity: OpeningIdentity | None = None,
    continuous_context: bool = False,
) -> pd.DataFrame:
    """Return the identical five-column schema for either scorer."""
    if bar_index.tz is None:
        raise ValueError("bar_index must be timezone-aware")
    if len(bar_index) == 0:
        return pd.DataFrame(index=bar_index, columns=MATCHED_FEATURES, dtype=float)
    if continuous_context and warmup_score_root is None:
        raise ValueError("continuous sentiment context requires a warm-up score root")
    news_decay, news_count = _matched_decay(
        stream,
        bar_index,
        scorer=scorer,
        direct=False,
        halflife_h=halflife_h,
        score_root=score_root,
        warmup_score_root=warmup_score_root,
        available_start=available_start,
        available_end=available_end,
        opening_identity=opening_identity,
    )
    direct_decay, _ = _matched_decay(
        stream,
        bar_index,
        scorer=scorer,
        direct=True,
        halflife_h=halflife_h,
        score_root=score_root,
        warmup_score_root=warmup_score_root,
        available_start=available_start,
        available_end=available_end,
        opening_identity=opening_identity,
    )
    tone_decay, macro_decay = _tone_and_macro(
        stream,
        bar_index,
        source_root=source_root,
        available_start=available_start,
        available_end=available_end,
        opening_identity=opening_identity,
        continuous_context=continuous_context,
    )
    return pd.DataFrame(
        {
            "sent_news_decay": news_decay,
            "sent_news_count_24h": news_count,
            "sent_direct_decay": direct_decay,
            "sent_tone_decay": tone_decay,
            "sent_macro_decay": macro_decay,
        },
        index=bar_index,
    ).reindex(columns=MATCHED_FEATURES).fillna(0.0)


def build_deepseek_full_features(
    stream: str,
    bar_index: pd.DatetimeIndex,
    *,
    halflife_h: float = 6.0,
) -> pd.DataFrame:
    """Add structured 0731 channels to the exact DeepSeek matched base."""
    base = build_matched_index_features(
        stream, bar_index, scorer="llm", halflife_h=halflife_h
    )
    if len(bar_index) == 0:
        return base.reindex(columns=DEEPSEEK_FULL_FEATURES)
    available_end = bar_index.max() + pd.Timedelta(minutes=15)
    scored = _load_matched_scores(
        stream, "llm", direct=False, available_end=available_end
    )
    keep, weight = _own_weights(scored, scorer="llm", stream=stream)
    retained = scored.loc[keep]
    relevance = scored["llm_relevance"].astype(float).clip(0.0, 1.0).to_numpy()
    impact = scored["llm_impact"].map(_IMPACT_WEIGHT).fillna(1.0).to_numpy()
    ones = np.ones(len(scored))
    relevance_decay = _decayed_weighted_mean(
        retained["seendate"], relevance[keep], ones[keep], bar_index, halflife_h=halflife_h
    )
    high = keep & (impact >= _IMPACT_WEIGHT[2])
    high_rows = scored.loc[high]
    high_decay = (
        _decayed_weighted_mean(
            high_rows["seendate"],
            high_rows["matched_sentiment"],
            weight[high],
            bar_index,
            halflife_h=halflife_h,
        )
        if high.any()
        else pd.Series(0.0, index=bar_index)
    )
    on_count = _window_features(
        retained["seendate"],
        ones[keep],
        bar_index,
        "on",
        windows_h=(24,),
        include_mean=False,
        include_count=True,
    )["on_count_24h"]
    all_count = _window_features(
        scored["seendate"],
        ones,
        bar_index,
        "all",
        windows_h=(24,),
        include_mean=False,
        include_count=True,
    )["all_count_24h"]
    structured = pd.DataFrame(
        {
            "sent_llm_relevance_decay": relevance_decay,
            "sent_llm_hi_impact_decay": high_decay,
            "sent_llm_topic_share_24h": (on_count / all_count.where(all_count > 0)).fillna(0.0),
        },
        index=bar_index,
    )
    return pd.concat([base, structured], axis=1).reindex(
        columns=DEEPSEEK_FULL_FEATURES
    ).fillna(0.0)


__all__ = [
    "DEEPSEEK_FULL_FEATURES",
    "MATCHED_FEATURES",
    "build_deepseek_full_features",
    "build_matched_index_features",
]
