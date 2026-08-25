"""Collect and stage Q2 2026 direct events without changing frozen inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from sentiment.direct_events import _make_rows, _streams_for_text, write_outputs
from sentiment.prepare_trump_posts import matches_financial_market


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truth_rows_from_archive(
    posts: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[dict]:
    """Convert archived Trump posts into causal, market-filtered event rows."""
    required = {"id", "created_at", "content", "url"}
    missing = required - set(posts.columns)
    if missing:
        raise ValueError(f"Truth archive missing columns: {sorted(missing)}")

    frame = posts.copy()
    frame["event_time"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
    frame = frame[
        frame["event_time"].notna()
        & frame["event_time"].ge(start)
        & frame["event_time"].lt(end)
    ].drop_duplicates("id")

    rows: list[dict] = []
    for row in frame.itertuples(index=False):
        content = str(row.content or "").strip()
        if not content or not matches_financial_market(content):
            continue
        rows.extend(
            _make_rows(
                event_time=row.event_time,
                source="truth_social",
                event_type="trump_truth_post",
                streams=_streams_for_text(content),
                title=content[:180],
                text=content,
                url=str(row.url),
                start=start,
                end=end,
            )
        )
    return rows


def stage_q2_direct_events(
    posts: pd.DataFrame,
    *,
    fed_rows: list[dict],
    output_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Write an isolated Q2 snapshot and combined causal direct-event parquets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    truth_snapshot = posts.copy()
    truth_snapshot["event_time"] = pd.to_datetime(
        truth_snapshot["created_at"], utc=True, errors="coerce"
    )
    truth_snapshot = truth_snapshot[
        truth_snapshot["event_time"].notna()
        & truth_snapshot["event_time"].ge(start)
        & truth_snapshot["event_time"].lt(end)
    ].drop_duplicates("id")
    truth_snapshot.to_parquet(output_dir / "truth_social_q2_all.parquet", index=False)

    bounded_fed_rows = [
        row
        for row in fed_rows
        if start <= pd.Timestamp(row["available_at"]) < end
    ]
    truth_rows = truth_rows_from_archive(posts, start=start, end=end)
    return write_outputs(
        bounded_fed_rows + truth_rows,
        output_dir=output_dir,
    )


def collect_q2_to_directory(
    *,
    truth_payload: bytes,
    fed_rows: list[dict],
    output_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    truth_source_url: str,
) -> tuple[pd.DataFrame, dict]:
    """Parse downloaded source bytes, stage Q2 files and bind a manifest."""
    records = json.loads(truth_payload.decode("utf-8"))
    if not isinstance(records, list):
        raise ValueError("Truth archive JSON must contain a list of posts")
    posts = pd.DataFrame(records)
    staged = stage_q2_direct_events(
        posts,
        fed_rows=fed_rows,
        output_dir=output_dir,
        start=start,
        end=end,
    )

    artifacts = [
        output_dir / "direct_events.parquet",
        *(output_dir / f"direct_events_{stream}.parquet" for stream in ("btc", "usa500", "usatech")),
        output_dir / "truth_social_q2_all.parquet",
    ]
    truth_snapshot = pd.read_parquet(output_dir / "truth_social_q2_all.parquet")
    manifest = {
        "window_start_utc": start.isoformat(),
        "window_end_utc_exclusive": end.isoformat(),
        "truth_source_url": truth_source_url,
        "truth_payload_sha256": hashlib.sha256(truth_payload).hexdigest(),
        "truth_archive_q2_rows": int(len(truth_snapshot)),
        "truth_relevant_posts": int(
            staged.loc[staged["source"].eq("truth_social"), "url"].nunique()
        ),
        "fed_documents": int(staged.loc[staged["source"].eq("fed"), "url"].nunique()),
        "stream_rows": {
            stream: int(staged["stream"].eq(stream).sum())
            for stream in ("btc", "usa500", "usatech")
        },
        "duplicate_stream_events": int(
            staged.duplicated(["stream", "url", "event_type"]).sum()
        ),
        "artifact_sha256": {path.name: _sha256(path) for path in artifacts},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return staged, manifest
