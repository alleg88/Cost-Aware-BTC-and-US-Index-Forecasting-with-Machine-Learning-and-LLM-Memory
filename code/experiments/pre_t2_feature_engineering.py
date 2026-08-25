"""Training-fold-only feature quality checks for Notebook I."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorrelationSelection:
    """Selected columns and the auditable pairwise decisions behind them."""

    selected_features: tuple[str, ...]
    removed_features: tuple[str, ...]
    audit: pd.DataFrame


def profile_features(
    frame: pd.DataFrame, features: Iterable[str]
) -> pd.DataFrame:
    """Report simple quality statistics without altering the data."""
    rows: list[dict[str, float | int | str]] = []
    for feature in features:
        if feature not in frame:
            raise ValueError(f"missing feature: {feature}")
        numeric = pd.to_numeric(frame[feature], errors="coerce")
        values = numeric.to_numpy(dtype=float)
        rows.append(
            {
                "feature": feature,
                "rows": int(len(values)),
                "missing_pct": float(np.isnan(values).mean() * 100.0),
                "non_finite_pct": float(np.isinf(values).mean() * 100.0),
                "unique_values": int(numeric.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def fold_correlation_filter(
    frame: pd.DataFrame,
    *,
    feature_order: Iterable[str],
    drop_threshold: float = 0.95,
    report_threshold: float = 0.80,
) -> CorrelationSelection:
    """Drop only near-duplicates, respecting a registered priority order.

    This function is intended to be called independently inside every outer
    training fold. Validation rows must never be passed to it.
    """
    order = tuple(feature_order)
    missing = sorted(set(order).difference(frame.columns))
    if missing:
        raise ValueError(f"training fold missing features: {missing}")
    if not 0.0 <= report_threshold < drop_threshold <= 1.0:
        raise ValueError("correlation thresholds must satisfy 0 <= report < drop <= 1")

    numeric = frame.loc[:, order].apply(pd.to_numeric, errors="coerce")
    correlation = numeric.corr(method="spearman").abs()
    selected: list[str] = []
    removed: list[str] = []
    rows: list[dict[str, object]] = []
    for feature in order:
        blockers: list[tuple[str, float]] = []
        for kept in selected:
            value = correlation.loc[kept, feature]
            if np.isfinite(value) and float(value) >= drop_threshold:
                blockers.append((kept, float(value)))
        if blockers:
            kept, value = max(blockers, key=lambda item: item[1])
            removed.append(feature)
            rows.append(
                {
                    "feature_a": kept,
                    "feature_b": feature,
                    "abs_spearman": value,
                    "action": "drop_second",
                }
            )
            continue
        for kept in selected:
            value = correlation.loc[kept, feature]
            if np.isfinite(value) and float(value) >= report_threshold:
                rows.append(
                    {
                        "feature_a": kept,
                        "feature_b": feature,
                        "abs_spearman": float(value),
                        "action": "report_only",
                    }
                )
        selected.append(feature)

    audit = pd.DataFrame(
        rows, columns=["feature_a", "feature_b", "abs_spearman", "action"]
    )
    return CorrelationSelection(tuple(selected), tuple(removed), audit)
