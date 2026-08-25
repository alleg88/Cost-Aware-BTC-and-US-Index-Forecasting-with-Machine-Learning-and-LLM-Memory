"""Collect low-lag direct text/event sources for sentiment scoring.

Two working sources (both leak-safe: a post/statement is public at its own timestamp):
  * Federal Reserve FOMC statements + minutes (official FOMC calendar).
  * Donald Trump Truth Social posts, market-filtered, from
    sentiment/raw/trump_truth_posts.csv (prepared by prepare_trump_posts.py).
An optional manual CSV (sentiment/raw/manual_direct_events.csv) can add curated events.

Outputs (kept separate from GDELT — these are primary-source events, not article echoes):
  * sentiment/raw/direct_events.parquet
  * sentiment/raw/direct_events_<stream>.parquet for btc, usa500, usatech

Run:  python -m sentiment.direct_events

(SEC/Treasury/RSS crawlers were removed: SEC rate-limits this machine and Treasury/RSS
produced no usable historical rows. Add such events via manual_direct_events.csv instead.)
"""
from __future__ import annotations

import html
import re
from pathlib import Path
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import pandas as pd
import requests

CODE_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = CODE_ROOT / "sentiment" / "raw"
OUT = RAW_DIR / "direct_events.parquet"
STREAMS = ("btc", "usa500", "usatech")

START = pd.Timestamp("2024-01-01", tz="UTC")
END = pd.Timestamp("2026-04-01", tz="UTC")
USER_AGENT = "msc-dissertation-research/0.1"

FED_BASE = "https://www.federalreserve.gov"
FED_FOMC = f"{FED_BASE}/monetarypolicy/fomccalendars.htm"
TRUMP_CSV = RAW_DIR / "trump_truth_posts.csv"
TRUMP_TEMPLATE = RAW_DIR / "trump_truth_posts_template.csv"
MANUAL_CSV = RAW_DIR / "manual_direct_events.csv"
MANUAL_TEMPLATE = RAW_DIR / "manual_direct_events_template.csv"

CRYPTO_RE = re.compile(
    r"\b(bitcoin|btc|crypto|cryptocurrency|digital asset|digital assets|stablecoin|"
    r"ethereum|ether|etf|coinbase|binance|ripple|xrp|token|tokens)\b",
    re.I,
)
TRUMP_MARKET_RE = re.compile(
    r"\b(tariff|tariffs|china|powell|federal reserve|fed|rates?|inflation|jobs|"
    r"crypto|bitcoin|btc|dollar|treasury|oil|tax|taxes)\b",
    re.I,
)


def _get(url: str) -> str:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=45)
    resp.raise_for_status()
    return resp.text


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _title_from_html(page: str) -> str:
    for pat in [
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)',
        r"<title[^>]*>(.*?)</title>",
        r"<h1[^>]*>(.*?)</h1>",
    ]:
        m = re.search(pat, page, flags=re.I | re.S)
        if m:
            return _strip_html(m.group(1)).replace(" | Federal Reserve", "")
    return ""


def _cut_text(text: str, starts: tuple[str, ...], ends: tuple[str, ...]) -> str:
    start_positions = [text.find(marker) for marker in starts if text.find(marker) >= 0]
    if start_positions:
        text = text[min(start_positions):]
    end_positions = [text.find(marker) for marker in ends if text.find(marker) > 0]
    if end_positions:
        text = text[:min(end_positions)]
    return text.strip()


def _page_text(page: str, event_type: str = "") -> str:
    for tag in ("main", "article"):
        m = re.search(rf"(?is)<{tag}[^>]*>(.*?)</{tag}>", page)
        if m:
            text = _strip_html(m.group(1))
            break
    else:
        text = _strip_html(page)

    if event_type == "fed_fomc_statement":
        return _cut_text(
            text,
            starts=(
                "Available indicators",
                "Recent indicators",
                "The Committee seeks",
            ),
            ends=(
                "For media inquiries",
                "Implementation Note issued",
                "Last Update:",
            ),
        )
    if event_type == "fed_fomc_minutes":
        return _cut_text(
            text,
            starts=(
                "Developments in Financial Markets and Open Market Operations",
                "Staff Review of the Economic Situation",
                "Staff Review of the Financial Situation",
                "Participants' Views on Current Conditions and the Economic Outlook",
                "Participants’ Views on Current Conditions and the Economic Outlook",
                "Minutes of the Federal Open Market Committee",
            ),
            ends=("Back to Top", "Last Update:", "Board of Governors of the Federal Reserve System"),
        )
    return text


def _et_to_utc(date_yyyymmdd: str, hour: int = 14, minute: int = 0) -> pd.Timestamp:
    local = pd.Timestamp(
        f"{date_yyyymmdd[:4]}-{date_yyyymmdd[4:6]}-{date_yyyymmdd[6:]} {hour:02d}:{minute:02d}",
        tz=ZoneInfo("America/New_York"),
    )
    return local.tz_convert("UTC")


def _in_window(
    ts: pd.Timestamp,
    *,
    start: pd.Timestamp = START,
    end: pd.Timestamp = END,
) -> bool:
    return start <= ts < end


def _streams_for_text(text: str, default: tuple[str, ...] = ("usa500", "usatech")) -> tuple[str, ...]:
    streams = set(default)
    if CRYPTO_RE.search(text):
        streams.add("btc")
    return tuple(s for s in STREAMS if s in streams)


def _make_rows(
    *,
    event_time: pd.Timestamp,
    source: str,
    event_type: str,
    streams: tuple[str, ...],
    title: str,
    text: str,
    url: str,
    start: pd.Timestamp = START,
    end: pd.Timestamp = END,
) -> list[dict]:
    if not _in_window(event_time, start=start, end=end):
        return []
    score_text = re.sub(r"\s+", " ", f"{title}. {text[:1500]}").strip()
    domain = re.sub(r"^https?://([^/]+)/.*$", r"\1", url)
    return [
        {
            "seendate": event_time,
            "event_time": event_time,
            "available_at": event_time,
            "source": source,
            "event_type": event_type,
            "stream": stream,
            "url": url,
            "domain": domain,
            "title": title,
            "text": text,
            "score_text": score_text,
        }
        for stream in streams
    ]


def collect_fed_fomc(
    *,
    start: pd.Timestamp = START,
    end: pd.Timestamp = END,
) -> list[dict]:
    """FOMC statements + minutes from the official Fed calendar (applies to all streams)."""
    page = _get(FED_FOMC)
    links: set[tuple[str, str, str]] = set()
    for date, suffix in re.findall(r"/newsevents/pressreleases/monetary(\d{8})(a)\.htm", page):
        links.add((date, "fed_fomc_statement", f"/newsevents/pressreleases/monetary{date}{suffix}.htm"))
    minutes_pattern = re.compile(
        r'href=["\']/monetarypolicy/fomcminutes(\d{8})\.htm["\']'
        r'.*?\(Released\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})\)',
        flags=re.I | re.S,
    )
    for meeting_date, released in minutes_pattern.findall(page):
        release_date = pd.Timestamp(released).strftime("%Y%m%d")
        links.add(
            (
                release_date,
                "fed_fomc_minutes",
                f"/monetarypolicy/fomcminutes{meeting_date}.htm",
            )
        )

    rows: list[dict] = []
    for date, event_type, href in sorted(links):
        event_time = _et_to_utc(date)
        if not _in_window(event_time, start=start, end=end):
            continue
        url = urljoin(FED_BASE, href)
        try:
            detail = _get(url)
        except requests.RequestException:
            continue
        rows.extend(
            _make_rows(
                event_time=event_time,
                source="fed",
                event_type=event_type,
                streams=STREAMS,
                title=_title_from_html(detail) or event_type.replace("_", " ").title(),
                text=_page_text(detail, event_type=event_type),
                url=url,
                start=start,
                end=end,
            )
        )
    return rows


def collect_trump_csv() -> list[dict]:
    """Market-filtered Trump Truth Social posts (prepared by prepare_trump_posts.py)."""
    if not TRUMP_CSV.exists():
        if not TRUMP_TEMPLATE.exists():
            TRUMP_TEMPLATE.write_text("created_at,url,content\n", encoding="utf-8")
        print(f"  no Trump Truth import found; template -> {TRUMP_TEMPLATE}")
        return []
    df = pd.read_csv(TRUMP_CSV)
    if "created_at" not in df.columns or "content" not in df.columns:
        raise ValueError(f"{TRUMP_CSV} must contain created_at and content columns")
    rows: list[dict] = []
    for i, row in df.iterrows():
        content = str(row.get("content", "")).strip()
        if not TRUMP_MARKET_RE.search(content):
            continue
        ts = pd.to_datetime(row["created_at"], utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        url = str(row.get("url", "")).strip() or f"manual://truthsocial/realDonaldTrump/{i}"
        rows.extend(
            _make_rows(
                event_time=ts,
                source="truth_social",
                event_type="trump_truth_post",
                # posts are already market-filtered upstream (tariffs/Fed/inflation/…), so
                # they are index/macro signal by default; also tag BTC when crypto-related.
                streams=_streams_for_text(content),
                title=content[:180],
                text=content,
                url=url,
            )
        )
    return rows


def collect_manual_direct_events() -> list[dict]:
    """Optional curated events from a CSV (for anything official scraping can't reach).

    Columns: event_time, source, event_type, stream, url, title, text
    (`stream` may be one of btc/usa500/usatech or a pipe/comma-separated list).
    """
    if not MANUAL_CSV.exists():
        if not MANUAL_TEMPLATE.exists():
            MANUAL_TEMPLATE.write_text(
                "event_time,source,event_type,stream,url,title,text\n", encoding="utf-8"
            )
        print(f"  no manual direct-event import found; template -> {MANUAL_TEMPLATE}")
        return []
    df = pd.read_csv(MANUAL_CSV)
    required = {"event_time", "source", "event_type", "stream", "url", "title", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{MANUAL_CSV} missing columns: {sorted(missing)}")
    rows: list[dict] = []
    for _, row in df.iterrows():
        ts = pd.to_datetime(row["event_time"], utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        streams = tuple(
            s.strip() for s in re.split(r"[|,]", str(row["stream"])) if s.strip() in STREAMS
        )
        if not streams:
            continue
        rows.extend(
            _make_rows(
                event_time=ts,
                source=str(row["source"]).strip(),
                event_type=str(row["event_type"]).strip(),
                streams=streams,
                title=str(row["title"]).strip(),
                text=str(row["text"]).strip(),
                url=str(row["url"]).strip(),
            )
        )
    return rows


COLLECTORS = (
    ("fed_fomc", collect_fed_fomc),
    ("trump_csv", collect_trump_csv),
    ("manual_csv", collect_manual_direct_events),
)


def write_outputs(
    rows: list[dict],
    *,
    output_dir: Path = RAW_DIR,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = ["seendate", "event_time", "available_at", "source", "event_type", "stream",
            "url", "domain", "title", "text", "score_text"]
    if rows:
        df = pd.DataFrame(rows)
        for c in ("seendate", "event_time", "available_at"):
            df[c] = pd.to_datetime(df[c], utc=True)
        df = (df.drop_duplicates(["stream", "url", "event_type"])
                .sort_values(["available_at", "stream", "url"], kind="mergesort")
                .reset_index(drop=True))
    else:
        df = pd.DataFrame(columns=cols)
    out = output_dir / "direct_events.parquet"
    df.to_parquet(out)
    print(f"wrote {len(df):,} direct-event stream rows -> {out}")
    for stream in STREAMS:
        part = df[df["stream"] == stream].copy()
        part.to_parquet(output_dir / f"direct_events_{stream}.parquet")
        print(f"  [{stream}] {len(part):,} rows")
    return df


def validate_canonical_direct_coverage(
    master_path: Path,
    truth_posts_path: Path,
) -> dict[str, int]:
    """Fail closed unless the frozen pre-Q2 direct-event coverage is exact."""
    master = pd.read_parquet(master_path, columns=["source", "url"])
    truth = pd.read_csv(truth_posts_path)
    required = {"created_at", "url", "content"}
    if not required.issubset(truth.columns):
        raise ValueError(f"Truth Social snapshot missing columns: {sorted(required - set(truth.columns))}")
    created = pd.to_datetime(truth["created_at"], utc=True, errors="raise")
    if not ((created >= START) & (created < END)).all():
        raise ValueError("Truth Social snapshot contains rows outside the pre-Q2 window")
    report = {
        "fed_stream_rows": int(master["source"].eq("fed").sum()),
        "truth_social_posts": int(truth["url"].nunique()),
        "truth_social_stream_rows": int(master["source"].eq("truth_social").sum()),
    }
    expected = {
        "fed_stream_rows": 111,
        "truth_social_posts": 1660,
        "truth_social_stream_rows": 3162,
    }
    if report != expected:
        raise ValueError(f"canonical direct-event coverage mismatch: {report} != {expected}")
    return report


def rebuild_direct_streams(master_path: Path, output_root: Path) -> dict[str, Path]:
    """Deterministically split a verified direct-event master into market streams."""
    frame = pd.read_parquet(master_path)
    required = {
        "seendate",
        "event_time",
        "available_at",
        "source",
        "event_type",
        "stream",
        "url",
        "domain",
        "title",
        "text",
        "score_text",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"direct-event master missing columns: {sorted(required - set(frame.columns))}")
    if not isinstance(frame["available_at"].dtype, pd.DatetimeTZDtype):
        raise ValueError("direct-event available_at must be timezone-aware UTC")
    for column in ("seendate", "event_time", "available_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if not frame["stream"].isin(STREAMS).all():
        raise ValueError("direct-event master contains an unknown stream")
    frame = (
        frame.drop_duplicates(["stream", "url", "event_type"])
        .sort_values(["available_at", "stream", "url"], kind="mergesort")
        .reset_index(drop=True)
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {stream: output_root / f"direct_events_{stream}.parquet" for stream in STREAMS}
    for stream, path in outputs.items():
        frame.loc[frame["stream"].eq(stream)].to_parquet(path, index=False)
    return outputs


def main() -> int:
    rows: list[dict] = []
    for name, fn in COLLECTORS:
        got = fn()
        rows.extend(got)
        print(f"  {name}: {len(got):,} stream rows")
    write_outputs(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
