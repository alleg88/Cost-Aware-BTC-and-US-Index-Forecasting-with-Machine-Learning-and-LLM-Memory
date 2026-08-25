"""Horizon naming shared by all experiment scripts.

Internally a horizon is a bar count on the M15 grid; everywhere a user sees it
(CLI flags, cache filenames, tables) it is a duration label so `h1` can never be
misread as "1 bar": m15 = next 15-minute bar (the project default), h1 = 1 hour
(4 bars), h4 = 4 hours (16 bars).
"""
from __future__ import annotations

BARS_PER_LABEL = {"m15": 1, "m30": 2, "m45": 3, "h1": 4, "h2": 8, "h4": 16, "d1": 96}
LABEL_PER_BARS = {bars: label for label, bars in BARS_PER_LABEL.items()}

DEFAULT_LABEL = "m15"


def parse_horizon(value: str | int) -> int:
    """Accept a duration label ('m15', 'h1', ...) or a legacy bar count."""
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text in BARS_PER_LABEL:
        return BARS_PER_LABEL[text]
    if text.isdigit():
        return int(text)
    raise ValueError(f"unknown horizon {value!r}; use one of {sorted(BARS_PER_LABEL)} or a bar count")


def horizon_label(bars: int) -> str:
    """Duration label for a bar count (falls back to total minutes, e.g. m90)."""
    return LABEL_PER_BARS.get(int(bars), f"m{int(bars) * 15}")
