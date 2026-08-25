"""Build Notebook A — the linear-regression channel branch.

A short standalone narrative for a line of work that does not sit in the numbered
M15 sequence: it trades inside detected channels on a 5-minute grid rather than
predicting next-bar direction on M15. Letter-named to keep the two apart.

Deliberately thin. Every table is either computed here from the cached grids or
read from a run directory under experiments/cache/channel_study/, so each number
traces to an artefact rather than to prose.

Run:  python -m experiments.build_notebook_a
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

from experiments.notebook_hygiene import canonical_colab_setup

CODE_ROOT = Path(__file__).resolve().parents[1]
OUT = CODE_ROOT / "notebooks" / "A_channel_strategy.ipynb"

SETUP = '''\
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(CODE_ROOT))
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

from features.linear_channels import compute_linear_regression_channels, label_channel_regime

DATA = CODE_ROOT / "data"
CACHE = CODE_ROOT / "experiments" / "cache" / "channel_study"
DEV_END = pd.Timestamp("2025-07-01", tz="UTC")

SPLITS = pd.DataFrame(
    [("dev", "2021-01-01", "2025-07-01", "rules, geometry, thresholds"),
     ("tune", "2025-07-01", "2025-10-01", "ranking threshold only"),
     ("forward", "2025-10-01", "2026-04-01", "evaluated once, after freezing"),
     ("lockbox", "2026-04-01", "2026-07-01", "sealed")],
    columns=["stage", "from", "to", "used for"])
display(SPLITS)

h1 = pd.read_parquet(DATA / "btcusdt_1h_2021_2026.parquet")
m5 = pd.read_parquet(DATA / "btcusdt_5min_2021_2026.parquet")
print(f"1h bars {len(h1):,}   5m bars {len(m5):,}   "
      f"{h1.index.min().date()} .. {h1.index.max().date()}")
print(f"incomplete 1h bars: {int((h1['minute_count'] < 60).sum())}  "
      f"(windows containing one are dropped, not silently fitted)")
'''

CENSUS = '''\
# How many channels exist. Trades and episodes have been reported elsewhere; this
# counts the structures themselves, which is what the two datasets are built around.
hi = h1[h1.index < DEV_END]
rows = []
for window in (60, 90, 120):
    ch = compute_linear_regression_channels(hi, window=window, log_price=True,
                                            method="quantile", quantile=0.10)
    ch["channel_slope"] = ch["channel_slope"] * 1e4          # bps per 1h bar
    reg = label_channel_regime(ch, min_slope=5.0, min_r2=0.40)
    run = (reg != reg.shift()).cumsum()
    entry = {"window": window}
    for kind in ("up", "down"):
        sizes = run[reg == kind].value_counts()
        entry[f"{kind} channels"] = len(sizes)
        entry[f"{kind} mean hours"] = round(sizes.mean(), 1) if len(sizes) else np.nan
    entry["total"] = entry["up channels"] + entry["down channels"]
    entry["share of time"] = round(float((reg != "none").mean()), 3)
    rows.append(entry)

census = pd.DataFrame(rows)
days = (DEV_END - pd.Timestamp("2021-01-01", tz="UTC")).days
census["per month"] = (census["total"] / (days / 30.4)).round(1)
display(census)
print(f"dev spans {days} days ({days/365:.1f} years)")
'''

RUNS = '''\
# Every completed run, read from its own artefact. The columns that differ between
# rows are the design decisions under test; everything else is held fixed.
rows = []
for d in sorted(CACHE.glob("*/")):
    cfg_p, sum_p = d / "config.json", d / "summary.json"
    if not (cfg_p.exists() and sum_p.exists()):
        continue
    c, s = json.loads(cfg_p.read_text()), json.loads(sum_p.read_text())
    rows.append({
        "target": c.get("target_mode"),
        "capacity": c.get("max_concurrent") or "unlimited",
        "min R2": c.get("min_r2"),
        "risk bps": f'{c.get("min_risk_bps"):g}-{c.get("max_risk_bps"):g}',
        "min R:R": c.get("min_rr"),
        "hold": c.get("max_hold_bars"),
        "trades": s.get("num_trades"),
        "per day": round(s.get("trades_per_day", np.nan), 2),
        "TP-first": s.get("tp_first_rate"),
        "gross R": s.get("mean_r_gross"),
        "net R": s.get("mean_r_net"),
    })
runs = pd.DataFrame(rows).sort_values("trades").reset_index(drop=True)
for col in ("TP-first", "gross R", "net R"):
    runs[col] = runs[col].astype(float).round(4)
display(runs)

best = runs.loc[runs["net R"].idxmax()]
print(f"least negative configuration: {int(best['trades']):,} trades at "
      f"{best['per day']:.2f}/day, gross {best['gross R']:+.4f}, net {best['net R']:+.4f}")
'''

FUNNEL = '''\
# The deployed configuration, from signal to filled trade.
# The deployed configuration, not the largest run: sections 2 and 3 must describe
# the same thing, and a superseded variant can easily hold more rows.
def _deployed(d):
    c = json.loads((d / "config.json").read_text())
    s = json.loads((d / "summary.json").read_text())
    return (c.get("target_mode") == "measured", c.get("max_concurrent") is None,
            s.get("mean_r_net", -9))

run_dir = max((d for d in CACHE.glob("*/")
               if (d / "summary.json").exists() and (d / "config.json").exists()),
              key=_deployed)
s = json.loads((run_dir / "summary.json").read_text())
sk = s["skipped"]
signals = s["num_trades"] + sum(sk.values())

funnel = pd.DataFrame(
    [("reversal signals", signals, 1.0)]
    + [(f"lost to {k}", -v, -v / signals) for k, v in sk.items() if v]
    + [("filled trades", s["num_trades"], s["num_trades"] / signals)],
    columns=["stage", "count", "share of signals"])
funnel["share of signals"] = funnel["share of signals"].round(3)
display(funnel)
cfg = json.loads((run_dir / "config.json").read_text())
print(f"artefact: {run_dir.name}")
print(f"target={cfg['target_mode']}  capacity={cfg['max_concurrent'] or 'unlimited'}  "
      f"entry={cfg['entry_mode']} offset {cfg['limit_offset_bps']:g}bps")

ev = pd.read_parquet(run_dir / "events.parquet")
print(f"\\nE7 event dataset: {len(ev):,} rows, "
      f"{ev['channel_episode_id'].nunique()} channel episodes, "
      f"{int(ev['label_net_positive'].sum())} positive labels "
      f"({ev['label_net_positive'].mean():.1%})")
print("long/short split:", ev["side"].value_counts().to_dict())
print("decision-time features:", [c for c in ev.columns
      if c not in ("signal_time","decision_time","side","channel_episode_id",
                   "order_status","filled","r_net","label_net_positive")])
'''

SEPARABILITY = '''\
# Whether the pool a ranking model would receive is separable at all. A filter can
# only extract what varies; if every stratum returns the same edge, more candidates
# is only more commission.
filled = ev[ev["filled"].astype(bool)].copy()
out = []
for col, bins, labels in [
    ("channel_confluence", [-0.1, 0.5, 1.1], ["one window", "two or more"]),
    ("episode_trade_number", [-0.1, 0.5, 2.5, 99], ["first", "2nd-3rd", "4th+"]),
    ("risk_bps_decision", None, None),
    ("rr_planned_decision", None, None),
]:
    if col not in filled:
        continue
    if bins is None:
        q = filled[col].quantile([0, 1/3, 2/3, 1]).to_numpy()
        grp = pd.cut(filled[col], bins=np.unique(q), include_lowest=True)
    else:
        grp = pd.cut(filled[col], bins=bins, labels=labels)
    g = filled.groupby(grp, observed=True)["r_net"]
    for name, sub in g:
        if len(sub) < 30:
            continue
        out.append({"feature": col, "stratum": str(name), "n": len(sub),
                    "mean net R": round(sub.mean(), 3),
                    "se": round(sub.std(ddof=1) / np.sqrt(len(sub)), 3)})
display(pd.DataFrame(out))
'''

LEVERS = '''# Every lever that could add trades, each measured through the same runner. Count and
# edge are shown together, because while the strategy is loss-making a larger count is
# a larger loss, not a better result: the metric that matters here is net R per trade.
csv = CACHE / "frequency_levers_dev.csv"
levers = pd.read_csv(csv).sort_values("net R", ascending=False).reset_index(drop=True)
display(levers)

ref = levers[levers["variant"].str.contains("reference")].iloc[0]
best = levers.iloc[0]
print(f"reference: {int(ref['trades']):,} trades at {ref['per day']:.2f}/day, "
      f"net {ref['net R']:+.4f}")
print(f"best per trade: {best['variant']} -> {int(best['trades']):,} trades at "
      f"{best['per day']:.2f}/day, net {best['net R']:+.4f}")
print(f"largest pool: {int(levers['trades'].max()):,} trades at "
      f"{levers.loc[levers['trades'].idxmax(), 'per day']:.2f}/day, "
      f"net {levers.loc[levers['trades'].idxmax(), 'net R']:+.4f}")
'''

E8 = """# The ranking test. Scores come from forward-chaining folds cut between channel
# episodes, so no candidate is scored by a model that saw its own channel, and the
# threshold is swept over out-of-fold predictions rather than fitted ones.
run = max((d for d in CACHE.glob("*/") if (d / "ranking_e8_table.csv").exists()),
          key=lambda d: (d / "ranking_e8_table.csv").stat().st_mtime)
table = pd.read_csv(run / "ranking_e8_table.csv")
display(table)

control = table.loc[table["selection"] == "take all", "mean_net_r"].iloc[0]
best = table.loc[table["selection"] != "take all", "mean_net_r"].max()
print(f"control {control:+.4f}   best model slice {best:+.4f}   lift {best - control:+.4f} R")
print(f"any slice with a 95% interval clear of the control: "
      f"{bool(table['beats_control'].any())}")
"""

SIDES = """# Whether one model over both sides beats one model per side, and whether a pooled
# score ranks inside each side or merely prefers the better-performing one.
sides = pd.read_csv(run / "ranking_e8_sides.csv")
display(sides)

def lift(model, scope):
    sub = sides[(sides["model"] == model) & (sides["scope"] == scope)]
    if len(sub) < 2:
        return np.nan
    top = sub[sub["slice"].str.contains("top 30")]["mean_net_r"].iloc[0]
    allr = sub[sub["slice"].str.contains("all")]["mean_net_r"].iloc[0]
    return round(top - allr, 4)

comp = pd.DataFrame([
    {"side": "long", "pooled model": lift("pooled", "within long"),
     "own model": lift("long-only", "long")},
    {"side": "short", "pooled model": lift("pooled", "within short"),
     "own model": lift("short-only", "short")},
])
print()
print("lift of the top 30% over take-all, in R per trade:")
display(comp)

sl = sides[(sides["model"] == "pooled") & (sides["scope"] == "both sides")]
print("short share of each pooled slice:", 
      dict(zip(sl["slice"], sl["short share"])))
"""

CELLS = [
    ("code", canonical_colab_setup()),
    ("md", """# A — Trading inside linear-regression channels

A branch, not a continuation. The numbered notebooks predict next-bar direction on
M15 and evaluate the prediction economically. This one asks a different question:
whether a trend channel, detected causally on an hourly grid, marks a region where
a five-minute mean-reversion entry is worth taking.

Nothing here feeds the M15 chain and nothing there is assumed. The two share only
the price data, the cost assumption and the sealed quarter.

**Design.** Three layers, each answering one question and each verifiable alone:

| layer | question | how |
|---|---|---|
| macro | which side may be opened | daily close against its own 200-day average |
| channel | is there a structure, and where are its edges | rolling OLS on 1h log price, window 60, bands at the 10th/90th residual quantile |
| entry | when to act inside it | reversal on the 5m grid, near the channel edge |

Risk is set by rule, not by model: the stop sits beyond the recent swing extreme and
the target is the size of the completed leg, both fixed at entry. Entries rest as
limit orders, so an unfilled order is recorded rather than assumed away.

Two datasets are kept apart — rising channels with long entries, falling channels
with short entries — because the two are trained independently and a shared frame
invites a feature computed across both."""),
    ("code", SETUP),
    ("md", """## 1. How much structure the market offers

The channel is the unit the datasets are built around, so its count sets the scale
of everything downstream. Bars inside one channel share a regime and are not
independent draws; the channel count, not the bar count, bounds what can be claimed."""),
    ("code", CENSUS),
    ("md", """## 2. Design decisions, each measured

Every completed run writes its configuration next to its result. The rows below
differ only in the decisions under test — the target rule, the position capacity,
and the geometry thresholds — so the comparison is like for like.

Two readings matter more than the ranking. A target placed on the far channel edge
is reached in a small minority of trades, because at a deep entry it sits five or
more risk units away; a target sized to the completed leg is nearer and is reached
far more often. And a capacity limit is a portfolio choice rather than a signal one:
holding the signal fixed, it removes trades without improving the ones that remain."""),
    ("code", RUNS),
    ("md", """## 3. From signal to filled trade

The funnel shows where candidates are lost. Losses split into two kinds that must
not be conflated: geometry rejections are a statement about the trade, while
capacity and fill failures are statements about execution."""),
    ("code", FUNNEL),
    ("md", """## 4. How far the frequency can be pushed

Every constraint that could be loosened to admit more trades is swept below: the two
geometry floors, the resting limit's price offset, and how long that limit is left
in the book. The signal itself is untouched throughout, so each row differs from the
reference only in what it is willing to accept or wait for.

A single relaxation improves both count and per-trade result; the rest buy volume by
admitting worse candidates. That asymmetry is the finding, and it places a ceiling on
frequency for any configuration that retains an edge."""),
    ("code", LEVERS),
    ("md", """## 5. Is the candidate pool separable?

A ranking model can only extract structure that already varies across candidates.
Each stratum below is defined by a quantity known at the decision bar."""),
    ("code", SEPARABILITY),
    ("md", """## 6. Can a model rank what the rule produces?

The rule does not clear its costs, so the question becomes whether its candidates
differ from one another in a way that is readable before the trade is placed. The
control is take-every-candidate: a model that cannot beat it has not found a weak
signal, it has found a homogeneous pool."""),
    ("code", E8),
    ("md", """### 6.1 One model, or one per side?

The design keeps rising-channel longs and falling-channel shorts apart. That is the
cleaner arrangement, and it is affordable only if each side carries enough channels
to train on.

Two readings decide it. If a pooled model merely preferred whichever side performs
better, its selected slices would be skewed towards that side and it would not rank
within either. The within-side rows below test exactly that."""),
    ("code", SIDES),
    ("md", """## 7. What this branch has established

The mechanical rule, in every configuration tried, does not clear its costs. The
least negative configuration reaches a gross figure close to zero and a negative net,
which places the entire question on whether a ranking model can select a profitable
subset from a pool whose average is not profitable.

**Register of limitations**, carried forward rather than resolved:

- around four hundred channels over four and a half years bound the effective sample;
  confidence intervals belong at the channel level, not the trade level
- one bear market in the sample, so any statement conditioned on falling regimes rests
  on a single observation of that regime
- roughly 160 configurations have been evaluated during design; a final claim requires
  a multiple-testing correction with that count stated
- the entry is a resting limit order, so its fill rate is a modelled quantity and an
  unfilled order carries no return rather than a missing one

**Next.** A causal logistic baseline over the event dataset, grouped by channel
episode, against the take-every-candidate control. Only if that separates does a
larger model follow. The forward window and the sealed quarter stay untouched until
the configuration is frozen."""),
]


def build() -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md"
                else nbf.v4.new_code_cell(src) for kind, src in CELLS]
    nb.metadata = {"kernelspec": {"display_name": "MSC Project (Python 3.12)",
                                  "language": "python", "name": "msc-code"},
                   "language_info": {"name": "python"}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    return OUT


def main() -> int:
    path = build()
    print(f"wrote {path}")
    result = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
         "--execute", "--inplace", "--ExecutePreprocessor.kernel_name=msc-code",
         "--ExecutePreprocessor.timeout=900", str(path)],
        cwd=str(CODE_ROOT / "notebooks"), capture_output=True, text=True)
    sys.stderr.write(result.stderr[-3000:])
    print(f"NBEXIT={result.returncode}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
