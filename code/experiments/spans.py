"""Shared walk-forward span / split constants, read from the config.

Centralises the three values that the forward-2026 extension made config-driven so
the pipeline runners don't hardcode them:

  CACHE_SUFFIX     cache-name span tag (prior 2025-only runs used the literal "2025";
                   the extended run uses "to2026" so both cache sets coexist).
  CALIBRATION_END  the freeze point — tau/brackets calibrated on data strictly before
                   this (Q1-2025), then frozen (unchanged by the extension).
  LOCKBOX_START    first bar of the sealed lockbox; the evaluation period is
                   [CALIBRATION_END, LOCKBOX_START) so lockbox rows never leak into
                   calibration or reported economics.
"""
from __future__ import annotations

from pathlib import Path

import yaml

_CFG = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "configs" / "default.yaml").read_text(encoding="utf-8"))
_DATES = _CFG["dates"]

CACHE_SUFFIX = str(_DATES.get("cache_suffix", "2025"))
CALIBRATION_END = str(_DATES.get("calibration_end", "2025-04-01"))
LOCKBOX_START = str(_DATES.get("lockbox", ["2026-04-01", "2026-06-30"])[0])
