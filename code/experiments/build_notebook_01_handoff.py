"""Build the direct Notebook 01 -> 02b fixed-history handoff."""
from __future__ import annotations

from pathlib import Path

import nbformat


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = CODE_ROOT / "notebooks" / "01_RQ1_A_BTC_data_labels_baseline.ipynb"


def main() -> int:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    imports = "\n".join(
        line
        for line in notebook.cells[2].source.splitlines()
        if not line.startswith("from experiments.notebook02_handoff import")
    )
    handoff_import = "from experiments.notebook02_handoff import NOTEBOOK01_WIDTHS, PIPELINE_HANDOFF, WIDTHS, write_notebook01_handoff, write_pipeline_handoff"
    notebook.cells[2].source = imports + "\n" + handoff_import

    selection_heading = next(
        cell
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        and "Dead zones carried forward" in cell.source
    )
    selection_heading.source = """### 6.1 Dead zones carried forward to Notebook 02b

The original paired development run froze DZ75, DZ65 and DZ55, the 24-column no-sentiment schema and a fixed 180-day training history in the downstream handoff. Reruns report the current source values for those widths but do not reselect them, because provider revisions must not change a frozen downstream policy. Notebook 02b reads that artifact rather than retyping the selection; DZ40 is not carried forward."""

    heading_index = notebook.cells.index(selection_heading)
    positioning_reading = next(
        cell
        for cell in reversed(notebook.cells[:heading_index])
        if cell.cell_type == "markdown"
    )
    positioning_reading.source = """**Reading:** compare only within each row because both arms use identical rows and folds. The table shows how positioning changes direction metrics, selectivity and after-cost economics for this input snapshot. Historical futures archives can be revised, so the signs are snapshot-specific; promotion is decided only by the later frozen gated walk-forward."""

    selection_code = next(
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "handoff_rows = pd.DataFrame" in cell.source
    )
    selection_code.source = """# Report the widths frozen by the original development run; never reselect on a rerun.
selected_widths = posr.loc[sorted(WIDTHS, reverse=True)].copy()
handoff_rows = pd.DataFrame({
    "width_bps": selected_widths.index.astype(int),
    "sortino": selected_widths["sortino_pos"].astype(float).to_numpy(),
    "sharpe": selected_widths["sharpe_pos"].astype(float).to_numpy(),
    "net_return": selected_widths["net_pos"].astype(float).to_numpy(),
    "trades": selected_widths["trades_pos"].astype(int).to_numpy(),
})
write_notebook01_handoff(handoff_rows, NOTEBOOK01_WIDTHS)
write_pipeline_handoff(upstream=handoff_rows, path=PIPELINE_HANDOFF)
width_table = pd.DataFrame({
    "dead zone": handoff_rows["width_bps"].map(lambda value: f"DZ{value}"),
    "Sortino": handoff_rows["sortino"].map("{:+.2f}".format),
    "Sharpe": handoff_rows["sharpe"].map("{:+.2f}".format),
    "net after costs": handoff_rows["net_return"].map("{:+.1%}".format),
    "events": handoff_rows["trades"].map("{:,}".format),
})
print(f"Notebook 01 handoff written: {NOTEBOOK01_WIDTHS}")
print(f"Notebook 02b pipeline handoff written: {PIPELINE_HANDOFF}")
display(width_table.set_index("dead zone"))"""

    selection_index = notebook.cells.index(selection_code)
    selection_reading = next(
        cell
        for cell in notebook.cells[selection_index + 1 :]
        if cell.cell_type == "markdown"
        and cell.source.startswith("**Reading:**")
    )
    selection_reading.source = """**Reading:** The original development run froze DZ75, DZ65 and DZ55. This table reports their values for the current input snapshot without reranking after provider revisions; it is not a profitability or promotion claim."""

    notebook.cells = [
        cell
        for cell in notebook.cells
        if not (
            cell.source.startswith("## Why 90 days - DZ40 later-pipeline reference")
            or "btc_balanced_dz40_lookback_eval.parquet" in cell.source
            or cell.source.startswith("**Why 90 days:**")
        )
    ]

    conclusions = next(
        cell
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        and cell.source.startswith("## Conclusions and next steps")
    )
    conclusions.source = """## Conclusions and next steps

**What this notebook establishes**
- The no-sentiment BTC baseline uses price, order-flow and positioning features with five 2024 `BlockingTimeSeriesSplit` folds, balanced training weights and 5 bps per side.
- Positioning reduces the ungated loss mainly by suppressing weak trades; it does not establish a profitable strategy by itself.
- The original paired development run froze **DZ55, DZ65 and DZ75**; reruns report those candidates without changing the downstream policy after provider revisions. Their snapshot-specific development economics are not a profitability or promotion claim.

**Where the pipeline continues**
1. **Notebook 02b** reads this handoff, compares fixed-baseline, F1-selected and economically selected CatBoost candidates, then calibrates execution policy causally on 2025 H1.
2. **Notebook 02c** documents the sentiment sources, scorer values, causal alignment and exact matched features.
3. **Notebook 02d** compares no sentiment, DeBERTa and LLM features across all nine model families and DZ55/DZ65/DZ75 using raw one-bar execution.
4. **Notebook 02e** calibrates confidence and TP/SL policies for the same nine models and sentiment arms.
5. **Notebook 03** constructs the all-nine stack from the candidates frozen by Notebook 02e.

Index replication (USA500 and USATECH) follows the same forward-only design. **2026 Q2 remains sealed** for the final lockbox check."""

    appendix = next(
        cell
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        and cell.source.startswith("## Appendix - Data sources and how each is used")
    )
    appendix.source = """## Appendix - Data sources and how each is used

All sources are free and publicly downloadable. The table shows the exact role of each source; 2026 Q2 remains sealed as the final lockbox.

| source (direct data link) | what | granularity / span | used for |
|---|---|---|---|
| [Binance spot 15m klines](https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCUSDT/15m/) | BTCUSDT OHLCV, taker-buy volume and trade count | **M15**, 2024-01 to 2026-06 | model features and labels in the BTC notebooks |
| [Binance spot 1m klines](https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCUSDT/1m/) | BTCUSDT OHLCV | **1-minute**, 2024-01 to 2026-06 | intrabar TP/SL ordering and fixed-hold execution scoring; not a model feature |
| [Binance futures metrics](https://data.binance.vision/?prefix=data/futures/um/daily/metrics/BTCUSDT/) | open interest, top-trader long/short ratio and taker buy/sell ratio | **5-minute** snapshots, 2024-01 to 2026-06 | positioning feature block |
| [Binance futures funding rate](https://data.binance.vision/?prefix=data/futures/um/monthly/fundingRate/BTCUSDT/) | perpetual-futures funding events | one row per **8 hours**, 2024-01 to 2026-06 | `funding_rate` and `funding_z` positioning features |
| [GDELT 2.0 event/GKG feed](http://data.gdeltproject.org/gdeltv2/masterfilelist.txt) - [project page](https://www.gdeltproject.org/data.html) | timestamped global news headlines | continuous **15-minute** updates, 2024 to 2026 | future sentiment layer; excluded from Notebooks 01 and 02b |
| [Dukascopy historical data feed](https://www.dukascopy.com/swiss/english/marketwatch/historical/) | USA500 and USATECH index CFD ticks | aggregated to M15, 2024 to 2026-Q1 | future index replication; the same 1-minute execution approach will be used where suitable data are available |

Reproduction scripts are `data/download_binance.py`, `data/load.py`, `data/build_1m.py`, `data/build_positioning.py` and `data/dukascopy.py`. Source values are joined backward as-of, so every M15 bar can only use information timestamped at or before its close. BTC costs are 5 bps per side; spot klines contain no bid/ask spread, so no separate per-bar spread is modelled."""

    nbformat.write(notebook, NOTEBOOK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
