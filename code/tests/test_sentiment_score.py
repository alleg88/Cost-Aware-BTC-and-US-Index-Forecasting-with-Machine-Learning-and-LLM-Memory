from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd

from sentiment import score


def test_default_scorer_pins_registered_revision(monkeypatch):
    captured: dict[str, object] = {}

    def pipeline(task, **kwargs):
        captured.update({"task": task, **kwargs})
        return lambda texts: [
            [
                {"label": "positive", "score": 0.7},
                {"label": "neutral", "score": 0.2},
                {"label": "negative", "score": 0.1},
            ]
            for _ in texts
        ]

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=pipeline))

    scorer = score._build_scorer(10)

    assert captured["model"] == score.MODEL
    assert captured["revision"] == score.REVISION
    assert captured["local_files_only"] is False
    assert scorer(["headline"]) == [0.6]


def test_score_stream_stamps_revision_and_reuses_only_matching_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "RAW_DIR", tmp_path)
    source = pd.DataFrame(
        {
            "seendate": pd.to_datetime(["2024-01-01"], utc=True),
            "url": ["https://example.test/a"],
            "title": ["Market rises"],
        }
    )
    source.to_parquet(tmp_path / "gdelt_btc.parquet", index=False)
    calls: list[list[str]] = []

    def builder(_batch_size):
        def scorer(texts):
            calls.append(list(texts))
            return [0.25] * len(texts)

        return scorer

    monkeypatch.setattr(score, "_build_scorer", builder)
    output = score.score_stream("btc")
    first = pd.read_parquet(output)

    assert calls == [["Market rises"]]
    assert first["revision"].tolist() == [score.REVISION]

    monkeypatch.setattr(
        score,
        "_build_scorer",
        lambda *_: (_ for _ in ()).throw(AssertionError("matching cache was not reused")),
    )
    score.score_stream("btc")
