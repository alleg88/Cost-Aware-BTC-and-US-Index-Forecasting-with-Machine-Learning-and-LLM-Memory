"""Read-only scoreboard for completed frozen base-model studies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.frozen_model_study import STUDY_ROOT
from experiments.model_zoo_protocol import BASE_MODELS, protocol_fingerprint

METRIC_COLUMNS = (
    "trades",
    "pooled_gross",
    "pooled_net",
    "pooled_sortino",
    "pooled_sharpe",
    "positive_folds",
    "n_long",
    "n_short",
    "long_net",
    "short_net",
    "bull_sortino",
    "sideways_sortino",
    "bear_sortino",
    "robust_score",
)


def _classification_metrics(directory: Path, result: dict) -> dict:
    path = directory / "classification_grid.parquet"
    if not path.exists():
        return {"robust_f1": np.nan, "overall_f1": np.nan}
    grid = pd.read_parquet(path)
    width = result.get("selected_width")
    candidate = result.get("selected_candidate")
    selected = grid
    if width is not None and candidate is not None:
        selected = grid[
            grid["width"].eq(width) & grid["candidate"].eq(candidate)
        ]
    if selected.empty:
        selected = grid
    row = selected.sort_values(
        ["robust_f1", "overall_f1"], ascending=[False, False]
    ).iloc[0]
    return {
        "robust_f1": float(row["robust_f1"]),
        "overall_f1": float(row["overall_f1"]),
    }


def _outer_net(result: dict) -> float:
    return float(
        sum(float(row.get("outer_net", 0.0)) for row in result.get("outer_audit", []))
    )


def build_scoreboard(root: Path = STUDY_ROOT) -> pd.DataFrame:
    """Return all nine models, keeping missing and no-trade outcomes visible."""
    fingerprint = protocol_fingerprint()
    rows = []
    for order, model in enumerate(BASE_MODELS):
        directory = Path(root) / model
        result_path = directory / "result.json"
        if not result_path.exists():
            rows.append(
                {
                    "model": model,
                    "model_order": order,
                    "status": "incomplete",
                    "eligible": False,
                    "decision": None,
                    "protocol_fingerprint": fingerprint,
                }
            )
            continue

        result = json.loads(result_path.read_text(encoding="utf-8"))
        actual_fingerprint = result.get("protocol_fingerprint")
        if actual_fingerprint != fingerprint:
            raise ValueError(
                f"protocol fingerprint mismatch for {model}: {actual_fingerprint}"
            )
        decision = result.get("decision", "no_trade")
        metrics = result.get("selected_development_metrics") or {}
        row = {
            "model": model,
            "model_order": order,
            "status": "complete",
            "eligible": decision == "trade",
            "decision": decision,
            "protocol_fingerprint": actual_fingerprint,
            "selected_width": result.get("selected_width"),
            "selected_candidate": result.get("selected_candidate"),
            "selected_tau": result.get("selected_tau"),
            "outer_audit_net": _outer_net(result),
            **_classification_metrics(directory, result),
        }
        row.update({column: metrics.get(column, np.nan) for column in METRIC_COLUMNS})
        rows.append(row)

    board = pd.DataFrame(rows)
    for column in (
        "robust_score",
        "pooled_sortino",
        "pooled_net",
        "trades",
    ):
        if column not in board:
            board[column] = np.nan
    board = board.sort_values(
        [
            "eligible",
            "robust_score",
            "pooled_sortino",
            "pooled_net",
            "trades",
            "model_order",
        ],
        ascending=[False, False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return board.drop(columns="model_order")


def write_scoreboard(root: Path = STUDY_ROOT) -> pd.DataFrame:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    board = build_scoreboard(root)
    board.to_parquet(root / "primary_scoreboard.parquet", index=False)
    board.to_csv(root / "primary_scoreboard.csv", index=False)
    records = board.astype(object).where(pd.notna(board), None).to_dict("records")
    (root / "primary_scoreboard.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    return board


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=STUDY_ROOT)
    args = parser.parse_args(argv)
    board = write_scoreboard(args.root)
    print(board.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
