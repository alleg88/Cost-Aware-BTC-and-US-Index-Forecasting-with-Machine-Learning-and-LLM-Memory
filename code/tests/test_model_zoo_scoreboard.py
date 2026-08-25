import json

import pandas as pd
import pytest

from experiments.model_zoo_protocol import BASE_MODELS, protocol_fingerprint
from experiments.model_zoo_scoreboard import build_scoreboard, write_scoreboard


def _write_result(root, model, *, decision, fingerprint=None, **metrics):
    directory = root / model
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "protocol_fingerprint": fingerprint or protocol_fingerprint(),
        "decision": decision,
        "selected_width": 75 if decision == "trade" else None,
        "selected_candidate": 3 if decision == "trade" else None,
        "selected_tau": 0.7 if decision == "trade" else None,
        "selected_development_metrics": metrics or None,
        "outer_audit": [
            {"outer_month": "2025-04", "outer_net": metrics.get("outer_net", 0.0)}
        ],
    }
    (directory / "result.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_scoreboard_never_ranks_failed_guards_as_eligible(tmp_path):
    _write_result(
        tmp_path,
        "gru",
        decision="no_trade",
        pooled_net=0.20,
        robust_score=9.0,
    )
    _write_result(
        tmp_path,
        "logreg",
        decision="trade",
        robust_score=0.4,
        pooled_sortino=0.8,
        pooled_net=0.03,
        trades=60,
    )

    board = build_scoreboard(tmp_path)

    assert len(board) == len(BASE_MODELS)
    indexed = board.set_index("model")
    assert not bool(indexed.loc["gru", "eligible"])
    assert bool(indexed.loc["logreg", "eligible"])
    assert board.iloc[0]["model"] == "logreg"
    assert indexed.loc["decision_tree", "status"] == "incomplete"


def test_scoreboard_uses_frozen_tie_break_order(tmp_path):
    common = {"robust_score": 0.3, "pooled_sortino": 0.7, "trades": 60}
    _write_result(
        tmp_path, "logreg", decision="trade", pooled_net=0.02, **common
    )
    _write_result(
        tmp_path, "gru", decision="trade", pooled_net=0.05, **common
    )

    board = build_scoreboard(tmp_path)

    assert list(board.loc[:1, "model"]) == ["gru", "logreg"]


def test_scoreboard_rejects_mixed_protocols(tmp_path):
    _write_result(
        tmp_path,
        "gru",
        decision="no_trade",
        fingerprint="different",
    )
    with pytest.raises(ValueError, match="fingerprint"):
        build_scoreboard(tmp_path)


def test_write_scoreboard_creates_three_readable_formats(tmp_path):
    _write_result(tmp_path, "gru", decision="no_trade")
    board = write_scoreboard(tmp_path)

    parquet = pd.read_parquet(tmp_path / "primary_scoreboard.parquet")
    csv = pd.read_csv(tmp_path / "primary_scoreboard.csv")
    payload = json.loads(
        (tmp_path / "primary_scoreboard.json").read_text(encoding="utf-8")
    )
    assert len(board) == len(parquet) == len(csv) == len(payload) == len(BASE_MODELS)
