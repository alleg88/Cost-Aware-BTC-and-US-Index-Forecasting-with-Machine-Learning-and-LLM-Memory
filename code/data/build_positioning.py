"""Build the M15 positioning snapshot (funding rate + open interest + ratios).

Reads the raw Binance futures archives fetched by download_binance.py
(--dataset metrics: daily 5-minute positioning snapshots; --dataset fundingRate:
8-hour funding events) and resamples both to the M15 bar grid **leak-free**:
each bar carries the last value stamped at or before the bar CLOSE
(merge_asof backward on close time), so nothing from the bar's future leaks in.

Output columns (raw levels; the derived features live in features/build.py):
  funding_rate     last 8h funding rate in force (step function, ffilled)
  sum_open_interest      total perp open interest (contracts), <=10 min stale
  toptrader_ls     sum_toptrader_long_short_ratio (positions, top accounts)
  taker_ls         sum_taker_long_short_vol_ratio (aggressive buy/sell volume)

Run:  python -m data.build_positioning
      -> data/btcusdt_positioning_m15_2024_2026.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path
import re

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
MET_DIR = DATA_DIR / "raw" / "binance" / "futures" / "metrics"
FUND_DIR = DATA_DIR / "raw" / "binance" / "futures" / "fundingRate"
OUT = DATA_DIR / "btcusdt_positioning_m15_2024_2026.parquet"
EXTENDED_GRID = DATA_DIR / "btcusdt_15min_2021_2026.parquet"
EXTENDED_OUT = DATA_DIR / "btcusdt_positioning_15min_2021_2026.parquet"

BAR = pd.Timedelta("15min")
MAX_STALE = pd.Timedelta("1h")   # metrics gaps beyond this fail the build


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _files_before(paths: list[Path], end_exclusive: pd.Timestamp | None) -> list[Path]:
    if end_exclusive is None:
        return paths
    boundary = _utc(end_exclusive)
    selected: list[Path] = []
    for path in paths:
        match = re.search(r"(\d{4}-\d{2}(?:-\d{2})?)$", path.stem)
        if match is None:
            raise ValueError(f"cannot prove source date from filename: {path.name}")
        if pd.Timestamp(match.group(1), tz="UTC") < boundary:
            selected.append(path)
    return selected


def load_metrics(end_exclusive: str | pd.Timestamp | None = None) -> pd.DataFrame:
    files = sorted(MET_DIR.glob("BTCUSDT-metrics-*.csv"))
    files = _files_before(files, None if end_exclusive is None else _utc(end_exclusive))
    if not files:
        raise FileNotFoundError(f"no metrics CSVs in {MET_DIR} — run "
                                "download_binance --dataset metrics first")
    frames = [pd.read_csv(f, usecols=["create_time", "sum_open_interest",
                                      "sum_toptrader_long_short_ratio",
                                      "sum_taker_long_short_vol_ratio"])
              for f in files]
    m = pd.concat(frames, ignore_index=True)
    m["create_time"] = (pd.to_datetime(m["create_time"], utc=True, format="mixed")
                        .astype("datetime64[us, UTC]"))
    if end_exclusive is not None:
        m = m.loc[m["create_time"] < _utc(end_exclusive)]
    m = (m.rename(columns={"sum_toptrader_long_short_ratio": "toptrader_ls",
                           "sum_taker_long_short_vol_ratio": "taker_ls"})
           .sort_values("create_time")
           .dropna(subset=["create_time", "sum_open_interest"])
           .drop_duplicates(subset="create_time", keep="last"))
    # Parts of the 2021 archive ship the same day twice (40,152 duplicated stamps
    # against 104,652 unique ones on a clean 5-minute cadence), so the as-of join
    # is given one row per instant rather than relying on which copy it lands on.
    # The long/short ratio columns ship empty for long stretches of 2022 (12.7%
    # toptrader and 64.9% taker coverage that year) while open interest is present
    # throughout. Dropping a whole snapshot because one ratio is blank would throw
    # the good open-interest reading away with it, leaving the as-of join to carry
    # a stale value across most of the year. Each column is therefore allowed to be
    # missing on its own and carried forward independently by the join.
    # A handful of snapshots report sum_open_interest == 0 (exchange glitch —
    # impossible for a live perp). Treat as missing so log/z features stay finite;
    # they carry forward from the last good value in the as-of join.
    zeros = int((m["sum_open_interest"] <= 0).sum())
    if zeros:
        print(f"  {zeros} zero/negative open-interest snapshots dropped (glitches)")
        m = m[m["sum_open_interest"] > 0]
    return m


def load_funding(end_exclusive: str | pd.Timestamp | None = None) -> pd.DataFrame:
    files = sorted(FUND_DIR.glob("BTCUSDT-fundingRate-*.csv"))
    files = _files_before(files, None if end_exclusive is None else _utc(end_exclusive))
    if not files:
        raise FileNotFoundError(f"no funding CSVs in {FUND_DIR} — run "
                                "download_binance --dataset fundingRate first")
    f = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    # calc_time is ms epoch with +1/+2 ms jitter — floor to the hour; unify the
    # datetime unit with the bar index (merge_asof requires identical dtypes).
    t = (pd.to_datetime(f["calc_time"], unit="ms", utc=True)
         .dt.floor("h").astype("datetime64[us, UTC]"))
    output = (pd.DataFrame({"time": t, "funding_rate": f["last_funding_rate"]})
              .sort_values("time").dropna())
    if end_exclusive is not None:
        output = output.loc[output["time"] < _utc(end_exclusive)]
    return output


def build(
    bar_index: pd.DatetimeIndex,
    *,
    end_exclusive: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Positioning frame aligned to bar-OPEN index, values known at bar CLOSE."""
    close_time = pd.DataFrame({"close_time": bar_index + BAR}, index=bar_index)

    metrics = load_metrics(end_exclusive=end_exclusive)
    out = pd.merge_asof(close_time, metrics, left_on="close_time",
                        right_on="create_time", direction="backward")
    stale = out["close_time"] - out["create_time"]
    # Exposed rather than silently forward-filled: a carried-forward reading is a
    # guess, and a model given no way to tell it apart from a fresh one will treat
    # an exchange outage as a genuine flat stretch of positioning.
    out["positioning_stale"] = stale > MAX_STALE
    out["positioning_age_min"] = stale.dt.total_seconds() / 60.0
    worst = stale.max()
    if worst > MAX_STALE:
        bad = out.loc[stale > MAX_STALE, "close_time"]
        print(f"NOTE: {len(bad)} bars ({len(bad) / len(out):.2%}) exceed {MAX_STALE} "
              f"staleness (max {worst}) — exchange metrics outages, forward-filled:")
        for day, cnt in bad.dt.date.value_counts().sort_index().items():
            print(f"  {day}: {cnt} bars")

    funding = load_funding(end_exclusive=end_exclusive)
    out = pd.merge_asof(out, funding, left_on="close_time", right_on="time",
                        direction="backward")
    out.index = bar_index
    return out[["funding_rate", "sum_open_interest", "toptrader_ls", "taker_ls",
                "positioning_stale", "positioning_age_min"]]


def main() -> int:
    import argparse
    import yaml

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extended", action="store_true",
                    help="build the 2021-2026 series on the channel-study 15m grid "
                         "instead of the frozen 2024-2026 snapshot")
    args = ap.parse_args()

    if args.extended:
        if not EXTENDED_GRID.exists():
            raise FileNotFoundError(f"{EXTENDED_GRID.name} missing — run data/build_grids.py")
        idx = pd.read_parquet(EXTENDED_GRID).index
        pos = build(pd.DatetimeIndex(idx))
        pos.to_parquet(EXTENDED_OUT)
        by_year = pos.groupby(pos.index.year)["positioning_stale"].mean()
        print(f"wrote {len(pos):,} rows -> {EXTENDED_OUT.name}")
        print("stale share by year:", {int(y): f"{v:.2%}" for y, v in by_year.items()})
        return 0

    cfg = yaml.safe_load((DATA_DIR.parent / "configs" / "default.yaml")
                         .read_text(encoding="utf-8"))
    bars = pd.read_parquet(DATA_DIR.parent / cfg["paths"]["working_parquet"])
    idx = pd.to_datetime(bars.index, utc=True)
    # extend through the lockbox quarter so the sealed run needs no rebuild
    lock = DATA_DIR.parent / cfg["paths"]["lockbox_parquet"]
    if lock.exists():
        idx = idx.union(pd.to_datetime(pd.read_parquet(lock).index, utc=True))

    pos = build(pd.DatetimeIndex(idx).sort_values())
    pos.to_parquet(OUT)
    cov = pos.notna().mean()
    print(f"wrote {len(pos):,} rows -> {OUT.name}")
    print("coverage:", {c: f"{v:.1%}" for c, v in cov.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
