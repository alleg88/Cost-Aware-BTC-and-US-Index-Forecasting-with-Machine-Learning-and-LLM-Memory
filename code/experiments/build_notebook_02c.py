"""Build Notebook 02c as the sentiment data and methodology handoff."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = CODE_ROOT / "notebooks" / "02c_sentiment_data_and_methodology.ipynb"


def build_notebook(path: Path = NOTEBOOK) -> Path:
    cells = [
        nbf.v4.new_markdown_cell(
            """# 02c — Sentiment data and methodology

This notebook continues Notebook 02b and specifies the sentiment layer before any
cross-model comparison. It owns four things and claims nothing else: **where the data comes
from, how it is scored, what a model may see without look-ahead, and how the per-bar values
are aggregated.** No model, policy or economic winner is selected here — that is Notebook
02d onwards. **2026 Q2 remains sealed.**

Two design decisions are argued from measurement rather than taste, and §3–§5 are the
evidence for them:

1. **Each arm is improved only by its own means.** The DeBERTa arm filters and weights using
   the headline text and DeBERTa's own outputs; the LLM arm uses its own `relevance`,
   `impact` and `asset`. No LLM field ever reaches the DeBERTa arm — otherwise that arm
   would be irreproducible from a DeBERTa-only pipeline, and the comparison would stop
   testing *layer against layer*.
2. **Direct events are one column, not several.** There are ~65 Fed and Trump events in the
   whole span, so a family of channels would claim resolution the sample cannot support. A
   single signed decayed pulse states how much event mass is live, what it said, and how
   heavily its own arm rated it. §5 is that measurement."""
        ),
        nbf.v4.new_markdown_cell(
            """## 1. Sources, scorers and causal alignment

| Source | Information | Timestamp | Downstream role |
|---|---|---|---|
| GDELT GKG via Google BigQuery | English financial headlines and V2Tone | crawler-visible `seendate` | headline score, attention count, tone |
| Federal Reserve releases and Donald Trump Truth Social posts | direct policy / market events | event publication time | §5 case study (not a feature) |
| FRED API | scheduled US macro releases | exact release time | macro-event decay |
| Alternative.me Fear & Greed | BTC market mood | daily publication | seven-day change |

Two scorers process the same eligible headline ledger:

- **DeBERTa** — `mrm8488/deberta-v3-ft-financial-news-sentiment-analysis`, reduced to a
  scalar from −1 (bearish) to +1 (bullish). It returns nothing else.
- **LLM** — `deepseek-v4-flash:cloud`, prompt/cache version 4, temperature `0`, batches of
  at most ten headlines. It returns sentiment `[-1, 1]`, relevance `[0, 1]`, impact
  (`low`/`medium`/`high`) and asset (`BTC`/`US500`/`USTECH`/`macro`/`other`):

> Score EACH numbered news item for the market: sentiment, relevance, impact and asset.
> Judge only from the headline; be conservative when unsure. Reply ONLY with a JSON object,
> exactly one entry per headline.

```text
n0: Bitcoin drops 6%, giving back all of its new year gains as traders stay on ETF watch
{"n0":{"sentiment":-0.7,"relevance":1.0,"impact":"high","asset":"BTC"}}
```

**Causal alignment.** Every feature uses only information visible by the close of its M15
bar. `seendate` is when the crawler *saw* the story, not when it was published — the
conservative choice. Headlines pass chronologically through a **72-hour causal story
ledger**: canonical-URL and exact-title echoes are dropped, and cross-domain titles need
≥95% similarity with matching numbers, prices, percentages and dates. **Only earlier
stories may suppress a later echo**, so appending future data cannot change a past feature.
GDELT tone, FRED releases and Fear & Greed are joined identically in both arms."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from IPython.display import display

CODE_ROOT = Path.cwd().resolve()
if CODE_ROOT.name == "notebooks":
    CODE_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from features.sentiment import (
    DIRECT_EVENT_HALFLIFE_H,
    MATCHED_SENTIMENT_FEATURES_BTC,
    build_direct_event_block,
    build_llm_full_features,
    build_matched_sentiment_features,
)
from sentiment.dedup import first_seen_only

ARTIFACT_DIR = CODE_ROOT / "notebooks" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
RAW = CODE_ROOT / "sentiment" / "raw"
LOCKBOX = pd.Timestamp("2026-04-01", tz="UTC")
plt.rcParams.update({"figure.figsize": (10, 4), "axes.grid": True, "grid.alpha": 0.2})
pd.set_option("display.max_colwidth", 86)

cfg = yaml.safe_load((CODE_ROOT / "configs" / "default.yaml").read_text())
price = pd.read_parquet(CODE_ROOT / cfg["instruments"]["btc"]["working_parquet"],
                        columns=["close"])
price.index = pd.to_datetime(price.index, utc=True)
price = price.loc[price.index < LOCKBOX]
bars = price.index

keys = ["seendate", "url", "title"]
classic_raw = pd.read_parquet(RAW / "scores_btc.parquet")
llm_raw = pd.read_parquet(RAW / "scores_llm_btc.parquet")
matched = classic_raw[[*keys, "sent"]].merge(
    llm_raw[[*keys, "llm_sent", "llm_relevance", "llm_impact", "llm_asset"]],
    on=keys, how="inner", validate="one_to_one").dropna(subset=["sent", "llm_sent"])
matched["seendate"] = pd.to_datetime(matched["seendate"], utc=True)
matched = first_seen_only(matched).sort_values("seendate")
matched = matched.loc[matched["seendate"] < LOCKBOX]

print(f"bars {len(bars):,} ({bars[0]:%Y-%m-%d} -> {bars[-1]:%Y-%m-%d}) | "
      f"matched unique stories {len(matched):,} | lockbox {LOCKBOX.date()} excluded")"""
        ),
        nbf.v4.new_markdown_cell(
            """## 2. Coverage and where the scorers disagree

Both scorers are intersected on timestamp, URL and title and then passed through the same
causal de-duplication the feature pipeline uses, so a coverage difference can never be
mistaken for a scorer difference.

Overall agreement is moderate, and the largest disagreements are not random — they share a
structure worth naming, because it is exactly what the LLM's `asset` field exists to
resolve."""
        ),
        nbf.v4.new_code_cell(
            """coverage = pd.DataFrame([{
    "Stream": "GDELT headlines", "DeBERTa rows": len(classic_raw),
    "LLM rows": len(llm_raw), "Matched unique stories": len(matched),
    "First": matched["seendate"].min(), "Last": matched["seendate"].max(),
}])
display(coverage)

r_all = matched["sent"].corr(matched["llm_sent"])
sign_all = float((np.sign(matched["sent"]) == np.sign(matched["llm_sent"])).mean())
print(f"Scorer agreement on matched headlines: Pearson r = {r_all:.3f}, "
      f"sign agreement = {sign_all:.1%}")

examples = (matched.assign(gap=(matched["sent"] - matched["llm_sent"]).abs())
            .sort_values(["gap", "llm_relevance"], ascending=False).head(6)
            [["seendate", "title", "sent", "llm_sent", "llm_relevance", "llm_impact",
              "llm_asset"]]
            .rename(columns={"sent": "DeBERTa", "llm_sent": "LLM",
                             "llm_relevance": "Relevance", "llm_impact": "Impact",
                             "llm_asset": "Asset"}))
print("\\nLargest disagreements (descriptive examples, not selected trading events):")
display(examples.round({"DeBERTa": 2, "LLM": 2, "Relevance": 2}))"""
        ),
        nbf.v4.new_markdown_cell(
            """## 3. Story quality, topicality and scorer bias

Three properties of the matched story set determine how a per-bar value has to be built.
Each is measured here, and each motivates one element of the aggregation specified in §4.

**(a) Relevance tracks signal quality, and it is externally validated.** Agreement between
two *independent* scorers rises monotonically with the LLM's relevance rating. That the
rating predicts how much a completely different model agrees is evidence it measures
something real, rather than a stylistic preference of the LLM.

**(b) Nearly half the feed is off-topic.** A large share of scored headlines is tagged
`other` — stories with no plausible channel to Bitcoin. Averaging them into a "BTC
sentiment" column mixes an unrelated signal into the measurement.

**(c) DeBERTa carries a systematic, asset-specific bias.** It is consistently more bearish
than the LLM *on BTC stories specifically*, and neutral elsewhere. The §2 examples show the
mechanism: a finance-tuned encoder reacts to negative valence ("crisis", "destroyed",
"selling") without tracking that the negativity attaches to the dollar while the prediction
is about Bitcoin. It has no notion of which asset a headline is *about*; the LLM does."""
        ),
        nbf.v4.new_code_cell(
            """band = pd.cut(matched["llm_relevance"], [0, 0.3, 0.5, 0.7, 0.9, 1.0])
ladder = matched.groupby(band, observed=True).apply(
    lambda g: pd.Series({"stories": len(g),
                         "scorer agreement r": g["sent"].corr(g["llm_sent"])}))
print("(a) Scorer agreement by LLM relevance band:")
display(ladder.round(3))

mix = (matched["llm_asset"].value_counts(normalize=True) * 100).round(1).rename("share %")
bias = (matched.assign(gap=matched["sent"] - matched["llm_sent"])
        .groupby("llm_asset")["gap"].agg(["mean", "count"]).round(3))
print("\\n(b) Asset mix and (c) mean (DeBERTa - LLM) by asset "
      "[negative = DeBERTa more bearish]:")
display(mix.to_frame().join(bias).rename(columns={"mean": "DeBERTa - LLM", "count": "n"}))

fig, ax = plt.subplots(figsize=(9, 3.2))
ax.hist(matched["sent"], bins=60, alpha=0.55, label="DeBERTa", color="tab:blue")
ax.hist(matched["llm_sent"], bins=60, alpha=0.55, label="LLM", color="tab:orange")
ax.set(title="Score distributions on the matched headline set",
       xlabel="sentiment", ylabel="stories")
ax.legend()
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "02c_score_distributions.png", dpi=160, bbox_inches="tight")
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## 4. How the per-bar value is built

Each arm is now filtered and weighted **only by quantities its own pipeline produces**:

| | on-topic test | weight |
|---|---|---|
| **DeBERTa arm** | keyword match on the headline text (`bitcoin`, `btc`, `crypto`, …) | the model's own confidence `abs(sent)`, times `log(1 + outlets carrying the story)` |
| **LLM arm** | the scorer's own `asset` tag in {BTC, macro} | `relevance` times `impact` |

The outlet count comes from the same causal 72-hour ledger, so it is available to a
DeBERTa-only pipeline and cannot see the future.

All three arms see the same *sources* — headlines, the shared controls and the direct-event
pulse — so a difference between them can only come from the scorer and from how each arm
weights what it reads, never from one arm having been given more data than another.

**Why not simply weight both arms by LLM relevance?** It would be the cleaner *matched*
design — the arms would then differ in exactly one number. But the DeBERTa arm would no
longer be DeBERTa: nobody deploying that encoder alone could reproduce it, an improvement
owed to the LLM would be credited to the baseline, and the comparison would answer "which
scalar is better *given* LLM metadata" instead of the question this project actually asks —
**does an LLM sentiment layer beat an encoder sentiment layer, end to end?** A behavioural
test in `tests/test_matched_sentiment_features.py` enforces the separation: perturbing
`llm_relevance`, `llm_impact` and `llm_asset` must leave the DeBERTa arm bit-for-bit
identical while the LLM arm moves.

The per-bar value is a decayed **weighted mean**, not a weighted sum. A sum would rise
with story volume as well as with story direction — ten mildly bullish stories would score
like two strongly bullish ones — and volume already has its own column. Normalising by the
decayed weight mass keeps this column purely directional and lets it use its full range.

The columns each arm hands downstream. Both arms carry the headline block, the shared
controls and the direct-event pulse of §5; `llm_full` adds the three structured channels
only the LLM can produce:

| column | meaning | scorer-dependent |
|---|---|---|
| `sent_news_decay` | decayed weighted mean headline sentiment | **yes** |
| `sent_news_count_24h` | unique matched stories in the trailing 24 h — attention, not direction | no |
| `sent_tone_decay` | decayed GDELT V2Tone | no |
| `sent_macro_decay` | decayed FRED release pulse | no |
| `sent_fng_change_7d` | BTC Fear & Greed versus seven days earlier | no |
| `sent_direct_pulse` | Fed / Trump signed decayed pulse (§5) | **yes** |"""
        ),
        nbf.v4.new_code_cell(
            """arms = {}
for scorer, label in (("classic", "DeBERTa"), ("llm", "LLM")):
    for weighting in ("plain", "own"):
        arms[f"{label}-{weighting}"] = build_matched_sentiment_features(
            "btc", bars, scorer=scorer, weighting=weighting)

summary = pd.concat(
    {name: frame[MATCHED_SENTIMENT_FEATURES_BTC].agg(["mean", "std", "min", "max"]).T
     for name, frame in arms.items()},
    names=["Arm", "Feature"])
print("Exact per-bar values handed to Notebook 02d "
      f"({len(MATCHED_SENTIMENT_FEATURES_BTC)} columns; `plain` counts every story "
      "equally):")
display(summary.round(4))

headline = pd.DataFrame({
    name: {"std": f["sent_news_decay"].std(), "min": f["sent_news_decay"].min(),
           "max": f["sent_news_decay"].max(),
           "stories per 24h": f["sent_news_count_24h"].mean()}
    for name, f in arms.items()}).T
print("\\nThe one column that differs between arms, under both weightings:")
display(headline.round(3))

# The three arms exactly as Notebook 02d receives them.
deployed = {
    "DeBERTa": pd.concat([arms["DeBERTa-own"],
                          build_direct_event_block("btc", bars, scorer="classic")],
                         axis=1),
    "LLM-matched": pd.concat([arms["LLM-plain"], build_direct_event_block(
        "btc", bars, scorer="llm", weighting="plain")], axis=1),
    "LLM-full": build_llm_full_features("btc", bars),
}
print("\\nColumn budget per arm — deliberately unequal, since each arm carries "
      "only what it can produce:")
display(pd.DataFrame({
    "sentiment columns": {k: v.shape[1] for k, v in deployed.items()},
    "direct-event channels": {k: sum(c.startswith("sent_direct") for c in v.columns)
                              for k, v in deployed.items()},
    "LLM structured channels": {k: sum(c.startswith("sent_llm") for c in v.columns)
                                for k, v in deployed.items()},
}))"""
        ),
        nbf.v4.new_markdown_cell(
            """## 5. Direct events: Fed releases and Trump posts

37 Federal Reserve releases and 28 Trump Truth Social posts, each scored by both arms and
carried as two channels weighted the same way headlines are — the LLM arm by `relevance ×
impact` with its own `asset` tag, the DeBERTa arm by its own confidence. The LLM reads this
stream more confidently than it reads headlines: **46% high impact against 12%**, mean
relevance **0.69 against 0.50**, and it attributes the events (38 `macro`, 21 `BTC`).

Three properties shape the design. First, **one column**: with ~65 events in 27 months,
several channels would claim resolution the sample cannot support. `sent_direct_pulse` is a
signed decayed **sum** — magnitude says how much event mass is still live, sign says what
the recent events said, and zero says none are recent. A mean would instead hold the last
event's direction indefinitely.

Second, the scorers agree on **direction** at chance level and correlate negatively: a
finance-tuned encoder scoring "Fed holds rates steady" cannot express which asset is
affected, whereas the LLM tags it. So the magnitude is the reliable part of this column and
the sign is the fragile part — keeping them together lets a model lean on the former without
being forced to trust the latter.

Third, policy events are digested over days, so the block uses a **48-hour half-life**
rather than the six hours used for headlines; at a headline half-life it would be zero on
almost every bar."""
        ),
        nbf.v4.new_code_cell(
            """direct_classic = pd.read_parquet(RAW / "scores_direct_events_btc.parquet")
direct_llm = pd.read_parquet(RAW / "scores_llm_direct_events_btc.parquet")
direct = direct_classic[[*keys, "sent"]].merge(
    direct_llm[[*keys, "llm_sent", "llm_relevance", "llm_impact", "llm_asset"]],
    on=keys, how="inner", validate="one_to_one").dropna(subset=["sent", "llm_sent"])
direct["seendate"] = pd.to_datetime(direct["seendate"], utc=True)
direct = first_seen_only(direct)
direct = direct.loc[direct["seendate"] < LOCKBOX]
direct["source"] = np.where(
    direct["url"].str.contains("federalreserve", case=False, na=False), "Fed", "Trump")

agree = float((np.sign(direct["sent"]) == np.sign(direct["llm_sent"])).mean())
print(f"Direct events: {len(direct)} matched "
      f"({direct['source'].value_counts().to_dict()}), "
      f"{direct['seendate'].min():%Y-%m} to {direct['seendate'].max():%Y-%m}")
print(f"  scorer agreement on direction: r = {direct['sent'].corr(direct['llm_sent']):+.3f}, "
      f"sign agreement {agree:.0%} (chance 50%) — hence the direction-free pulse")
print(f"  LLM structure here vs on headlines: high impact "
      f"{(direct['llm_impact'] >= 2).mean():.0%} vs {(matched['llm_impact'] >= 2).mean():.0%}, "
      f"mean relevance {direct['llm_relevance'].mean():.2f} vs "
      f"{matched['llm_relevance'].mean():.2f}")

block = {label: build_direct_event_block("btc", bars, scorer=scorer)
         for scorer, label in (("classic", "DeBERTa"), ("llm", "LLM"))}
shape = pd.DataFrame({
    "pulse std": {k: v["sent_direct_pulse"].std() for k, v in block.items()},
    "bars with a live pulse (|pulse| > 0.05)": {
        k: float((v["sent_direct_pulse"].abs() > 0.05).mean()) for k, v in block.items()},
    "share of live bars signed positive": {
        k: float((v["sent_direct_pulse"] > 0).mean()) for k, v in block.items()},
    "std of sent_news_decay": {
        "DeBERTa": arms["DeBERTa-own"]["sent_news_decay"].std(),
        "LLM": arms["LLM-own"]["sent_news_decay"].std()},
})
print(f"\\nThe direct-event pulse at a {DIRECT_EVENT_HALFLIFE_H:g}h half-life:")
display(shape.round(4))"""
        ),
        nbf.v4.new_markdown_cell(
            """## 6. Model-free diagnostics, and the ceiling they imply

Two descriptive checks. Neither is a forecast test and neither selects anything.

**Left — linear association with the next M15 return**, for each arm over the exact columns
it hands to Notebook 02d, the direct-event pair included. Reported with the sample size and
a ±2 standard-error band, because across tens of thousands of bars a correlation can sit
several standard errors from zero and still explain a rounding error's worth of variance.

The arms carry different numbers of columns by construction, and that is itself a variable:
Notebook 01b showed that adding columns changes a boosted model's behaviour even when the
columns hold no information. It is the reason the direct-event block is a **single**
column rather than a family of them — with 65 events, more columns would buy resolution the
sample cannot support.

**Right — event study over ±24 hours.** Strong matched headlines (|score| ≥ 0.9 in either
scorer) are aligned on the bar where the crawler first saw them, and mean absolute M15
movement is plotted around that point. The window is deliberately wide: elevated movement
*precedes* the headline, and a narrow window would hide how far back it begins.

This panel is the ceiling on the entire sentiment layer. If the market has already moved
before a headline becomes visible, then a headline-derived feature at M15 documents news
rather than anticipating it, and no amount of scorer quality recovers what is already
priced."""
        ),
        nbf.v4.new_code_cell(
            """forward_return = price["close"].shift(-1).div(price["close"]).sub(1.0).rename("next")
corr_rows = []
for name, frame in deployed.items():
    joined = frame.join(forward_return)
    for feature in frame.columns:
        corr_rows.append({"Arm": name, "Feature": feature.replace("sent_", ""),
                          "r": joined[feature].corr(joined["next"])})
corr = pd.DataFrame(corr_rows)
n_obs = int(forward_return.notna().sum())
se = 1.0 / np.sqrt(n_obs)

WINDOW = 96                                    # 24 hours of M15 bars
strong = matched.loc[matched[["sent", "llm_sent"]].abs().max(axis=1) >= 0.9]
log_return = np.log(price["close"]).diff()
positions = price.index.searchsorted(
    pd.DatetimeIndex(strong["seendate"]).to_numpy(), side="right") - 1
positions = positions[(positions >= WINDOW) & (positions < len(price) - WINDOW - 1)]
matrix = np.stack([log_return.to_numpy()[p - WINDOW:p + WINDOW + 1] for p in positions])
offsets = np.arange(-WINDOW, WINDOW + 1)
move = np.nanmean(np.abs(matrix), axis=0) * 1e4
baseline = float(np.nanmean(np.abs(log_return)) * 1e4)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
corr.pivot(index="Feature", columns="Arm", values="r").plot(kind="barh", ax=axes[0],
                                                            width=0.8)
axes[0].axvline(0, color="black", lw=0.7)
for edge in (-2 * se, 2 * se):
    axes[0].axvline(edge, color="grey", ls=":", lw=1)
axes[0].set(title=f"Correlation with next M15 return (n={n_obs:,}, dotted = ±2 SE)",
            xlabel="Pearson r")
axes[0].legend(fontsize=7)

axes[1].plot(offsets / 4, move, lw=1.4, color="tab:blue")
axes[1].axhline(baseline, color="grey", ls="--", lw=1,
                label=f"all-bar baseline ({baseline:.1f} bps)")
axes[1].axvline(0, color="tab:red", lw=1, label="headline visible")
axes[1].set(title=f"Event study, ±24h: {len(positions):,} strong headlines",
            xlabel="hours around crawler-visible arrival",
            ylabel="mean |M15 return| (bps)")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "02c_sentiment_diagnostics.png", dpi=160, bbox_inches="tight")
plt.show()

excess = move - baseline
peak = float(excess[WINDOW])
before = excess[:WINDOW]
onset = (offsets[np.argmax(before > 0.5 * peak)] / 4
         if bool((before > 0.5 * peak).any()) else float("nan"))
print(f"Event study | baseline {baseline:.2f} bps | -24h {move[0]:.2f} | "
      f"-4h {move[WINDOW - 16]:.2f} | headline bar {move[WINDOW]:.2f} | "
      f"+4h {move[WINDOW + 16]:.2f} | +24h {move[-1]:.2f}")
print(f"  excess movement first exceeds half its peak {abs(onset):.1f} h BEFORE the "
      f"headline becomes visible")
print(f"  share of the peak excess already present one hour before: "
      f"{excess[WINDOW - 4] / peak:.0%}")
display(corr.pivot(index="Feature", columns="Arm", values="r").round(4))"""
        ),
        nbf.v4.new_markdown_cell(
            """## 7. Handoff to Notebook 02d

This notebook establishes provenance, causal alignment, coverage, the aggregation rule and
the measured ceiling. It infers **no** trading value from descriptive correlation.

Notebook 02d continues with four strictly matched arms over the unchanged 24 price,
order-flow and positioning features:

| arm | sentiment columns | built from |
|---|---|---|
| **No sentiment** | none | — (unchanged control from 02b) |
| **DeBERTa** | 6 — headline block, shared controls, direct-event pulse | DeBERTa scalar + headline text |
| **LLM-matched** | 6 — the same blocks, everything unweighted | LLM scalar only: the control that isolates weighting and structure |
| **LLM-full** | 9 — the above plus `relevance`, `impact`, `asset` channels and the direct-event pulse | the complete LLM output |

**DeBERTa against LLM-full** is the comparison the dissertation's claim rests on: each layer
improved by what it can produce alone. **LLM-matched** stays as the control that isolates how
much of any difference is the scalar rather than the metadata.

DZ55/DZ65/DZ75, the fixed 180-day history, raw 15-minute execution and 5 bps per side are
unchanged from 02b, and Notebook 02e calibrates execution policy on exactly these
predictions. No later result revises anything specified here. **2026 Q2 remains sealed.**"""
        ),
    ]

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "msc-code",
        "language": "python",
        "name": "msc-code",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)
    return path


if __name__ == "__main__":
    print(build_notebook())
