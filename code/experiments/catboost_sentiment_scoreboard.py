"""Reader-facing tables for the raw 02c and policy-calibrated 02d sentiment studies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from experiments.catboost_sentiment_ablation import DEFAULT_ROOT
from experiments.notebook02_handoff import MATCHED_ROOT
from experiments.notebook02b_handoff import HANDOFF_PATH, load_notebook02b_handoff
from experiments.raw_hold_control import CONFIG_PATH, _load_m15_bars, simulate_fixed_hold
from evaluation.economics import economics_summary
from sentiment.dedup import add_story_ids


CODE_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = CODE_ROOT / "sentiment" / "raw"
ARM_LABELS = {"baseline": "No sentiment", "classic": "Classic (DeBERTa)", "llm": "LLM"}
WIDTHS = (55, 65, 75)
FORWARD_START = pd.Timestamp("2025-07-01", tz="UTC")
FORWARD_END = pd.Timestamp("2026-04-01", tz="UTC")
RAW_OUTPUT_ROOT = CODE_ROOT / "experiments" / "cache" / "tuning" / "catboost_sentiment_raw_180d"
RAW_CLASSIFICATION = RAW_OUTPUT_ROOT / "classification_2024.parquet"
RAW_ECONOMICS = RAW_OUTPUT_ROOT / "raw_forward_economics.parquet"
RAW_MANIFEST = RAW_OUTPUT_ROOT / "manifest.json"
CALIBRATED_OUTPUT_ROOT = CODE_ROOT / "experiments" / "cache" / "tuning" / "catboost_sentiment_calibrated_180d"
CALIBRATED_POLICIES = CALIBRATED_OUTPUT_ROOT / "selected_policies_2025h1.parquet"
CALIBRATED_ECONOMICS = CALIBRATED_OUTPUT_ROOT / "calibrated_forward_economics.parquet"
CALIBRATED_MANIFEST = CALIBRATED_OUTPUT_ROOT / "manifest.json"
BASE_FEATURES = {
    "Price": [
        ("r1", "one-bar return"), ("r5", "five-bar return"),
        ("r20", "twenty-bar return"), ("vol_10", "10-bar return volatility"),
        ("vol_20", "20-bar return volatility"), ("vol_60", "60-bar return volatility"),
        ("hl_range", "high-low range"), ("co_range", "close-open range"),
        ("rsi_14", "14-bar RSI"), ("volume", "bar volume"),
        ("vol_z", "standardised volume"), ("hour", "UTC hour"),
        ("dayofweek", "UTC day of week"),
    ],
    "Order flow": [
        ("ofi", "signed taker imbalance"), ("ofi_z20", "20-bar OFI z-score"),
        ("ofi_mom5", "five-bar OFI change"),
        ("trade_intensity_z", "standardised trade intensity"),
    ],
    "Positioning": [
        ("funding_rate", "latest published funding rate"),
        ("funding_z", "seven-day funding z-score"),
        ("oi_chg_1h", "one-hour open-interest change"),
        ("oi_chg_4h", "four-hour open-interest change"),
        ("oi_z", "open-interest z-score"),
        ("toptrader_ls_z", "top-trader long/short z-score"),
        ("taker_ls_z", "taker long/short z-score"),
    ],
    "Matched sentiment": [
        ("sent_news_decay", "six-hour half-life decay of headline sentiment"),
        ("sent_news_count_24h", "unique news stories in the trailing 24 hours"),
        ("sent_direct_decay", "six-hour half-life decay of Fed/Trump sentiment"),
        ("sent_tone_decay", "six-hour half-life decay of GDELT V2Tone"),
        ("sent_macro_decay", "six-hour half-life decay of FRED release events"),
        ("sent_fng_change_7d", "change in Crypto Fear & Greed over seven days"),
    ],
}
GDELT_BTC_DOMAINS = (
    "cointelegraph.com, coindesk.com, forbes.com, investing.com, bitcoinmagazine.com, "
    "cnbc.com, businessinsider.com, fortune.com, bloomberg.com, marketwatch.com, "
    "barrons.com, apnews.com, wsj.com, kelownacapnews.com, wnewsj.com, decrypt.co, "
    "skift.com, microsoft.com, theblock.co, vendingmarketwatch.com, forbes.com.au, "
    "thomsonreuters.com"
)


def feature_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Block": block, "Feature": feature, "Definition": definition}
            for block, entries in BASE_FEATURES.items()
            for feature, definition in entries
        ]
    )


def source_table() -> pd.DataFrame:
    gdelt = pd.read_parquet(RAW_DIR / "gdelt_btc.parquet")
    direct = pd.read_parquet(RAW_DIR / "direct_events_btc.parquet")
    fred = pd.read_parquet(RAW_DIR / "fred_calendar.parquet")
    fear = pd.read_parquet(RAW_DIR / "crypto_fear_greed.parquet")
    classic = pd.read_parquet(RAW_DIR / "scores_btc.parquet")
    llm = pd.read_parquet(RAW_DIR / "scores_llm_btc.parquet")
    return pd.DataFrame([
        {"Input": "BTCUSDT M15 + one-minute execution", "Provider": "Binance public market data", "Rows": "see Notebook 01", "Use": "model inputs and TP/SL replay"},
        {"Input": "Funding, open interest and long/short ratios", "Provider": "Binance USD-M Futures", "Rows": "see Notebook 01", "Use": "positioning features"},
        {"Input": "English financial/crypto news", "Provider": "GDELT GKG via Google BigQuery", "Rows": len(gdelt), "Use": "headline, V2Tone and publication time"},
        {"Input": "Financial headline score", "Provider": str(classic["model"].iloc[0]), "Rows": int(classic["sent"].notna().sum()), "Use": "Classic scalar sentiment"},
        {"Input": "LLM headline score", "Provider": f'{llm["model"].iloc[0]} / prompt v{llm["version"].iloc[0]}', "Rows": int(llm["llm_sent"].notna().sum()), "Use": "LLM scalar sentiment"},
        {"Input": "FOMC statements/minutes", "Provider": "Federal Reserve", "Rows": int((direct["source"] == "fed").sum()), "Use": "direct-event sentiment"},
        {"Input": "Market-relevant posts", "Provider": "Truth Social via chrissoria/trump-truth-social", "Rows": int((direct["source"] == "truth_social").sum()), "Use": "direct-event sentiment"},
        {"Input": "Ten US macro series", "Provider": "FRED API", "Rows": len(fred), "Use": "release-time pulse"},
        {"Input": "Crypto Fear & Greed", "Provider": "Alternative.me API", "Rows": len(fear), "Use": "seven-day change"},
    ])


def dedup_audit() -> pd.DataFrame:
    keys = ["seendate", "url", "title"]
    classic = pd.read_parquet(RAW_DIR / "scores_btc.parquet")[[*keys, "sent"]].dropna()
    llm = pd.read_parquet(RAW_DIR / "scores_llm_btc.parquet")[[*keys, "llm_sent"]].dropna()
    common = classic.merge(llm, on=keys, how="inner", validate="one_to_one")
    audit = add_story_ids(common)
    rows = (
        audit.groupby(["dedup_reason", "keep_story"], dropna=False)
        .size().rename("Rows").reset_index()
        .rename(columns={"dedup_reason": "Decision", "keep_story": "Kept"})
    )
    return rows.sort_values(["Kept", "Decision"], ascending=[False, True]).reset_index(drop=True)


def _arm_root(arm: str, root: Path) -> Path:
    return MATCHED_ROOT if arm == "baseline" else Path(root) / arm


def _read_manifest(root: Path, upstream_fingerprint: str) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("training_histories_days") != {"55": 180, "65": 180, "75": 180}:
        raise ValueError(f"{root.name}: training histories are not fixed at 180 days")
    if manifest.get("upstream_notebook02_handoff_fingerprint") != upstream_fingerprint:
        raise ValueError(f"{root.name}: upstream feature handoff changed")
    if not manifest.get("sealed_lockbox"):
        raise ValueError(f"{root.name}: lockbox is not sealed")
    if pd.Timestamp(manifest.get("lockbox_start")) != FORWARD_END:
        raise ValueError(f"{root.name}: lockbox boundary changed")
    return manifest


def _load_current_prediction(root: Path, manifest: dict, width: int) -> tuple[pd.DataFrame, str]:
    allowed = set(manifest["forward"]["prediction_fingerprints"])
    candidates = [
        path
        for path in (root / "stage_predictions" / "forward").glob(
            f"w{width}_candidate_00_*.parquet"
        )
        if path.stem.rsplit("_", 1)[-1] in allowed
    ]
    if len(candidates) != 1:
        raise ValueError(f"{root.name} DZ{width}: expected one current candidate-0 prediction")
    fingerprint = candidates[0].stem.rsplit("_", 1)[-1]
    frame = pd.read_parquet(candidates[0]).copy()
    required = {
        "timestamp", "width_bps", "candidate_id", "pred", "train_end",
        "test_start", "test_end", "refit_id",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{root.name} DZ{width}: missing prediction columns {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if not frame["width_bps"].astype(int).eq(width).all():
        raise ValueError(f"{root.name} DZ{width}: dead zone changed")
    if not frame["candidate_id"].astype(int).eq(0).all():
        raise ValueError(f"{root.name} DZ{width}: baseline candidate changed")
    if frame["refit_id"].nunique() != 1:
        raise ValueError(f"{root.name} DZ{width}: forward model was refitted")
    if frame["refit_id"].iloc[0] not in manifest["forward"]["physical_fit_ids"]:
        raise ValueError(f"{root.name} DZ{width}: refit id is not in the sealed manifest")
    if not pd.to_datetime(frame["test_start"], utc=True).eq(FORWARD_START).all():
        raise ValueError(f"{root.name} DZ{width}: forward start changed")
    if not pd.to_datetime(frame["test_end"], utc=True).eq(FORWARD_END).all():
        raise ValueError(f"{root.name} DZ{width}: forward end changed")
    if pd.to_datetime(frame["train_end"], utc=True).max() >= FORWARD_START:
        raise ValueError(f"{root.name} DZ{width}: fit crossed the forward boundary")
    return frame.sort_values("timestamp"), fingerprint


def _write_artifacts(classification: pd.DataFrame, economics: pd.DataFrame, manifest: dict) -> None:
    RAW_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    classification.to_parquet(RAW_CLASSIFICATION, index=False)
    economics.to_parquet(RAW_ECONOMICS, index=False)
    payload = dict(manifest)
    payload["manifest_fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    RAW_MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_scoreboards(root: Path = DEFAULT_ROOT) -> dict[str, pd.DataFrame]:
    """Build the raw candidate-0 sentiment comparison without policy artifacts."""
    handoff = load_notebook02b_handoff(HANDOFF_PATH)
    bars = _load_m15_bars(CONFIG_PATH)
    classifications: list[pd.DataFrame] = []
    economic_rows: list[dict] = []
    source_fingerprints: dict[str, dict[str, str]] = {}

    for arm, label in ARM_LABELS.items():
        arm_root = _arm_root(arm, root)
        manifest = _read_manifest(arm_root, handoff["upstream_handoff_fingerprint"])
        classification = pd.read_parquet(arm_root / "classification_2024.parquet")
        classification = classification.loc[
            classification["candidate_id"].eq(0)
            & classification["width_bps"].isin(WIDTHS)
        ].copy()
        if len(classification) != 3:
            raise ValueError(f"{arm}: expected candidate 0 for all three dead zones")
        classification.insert(0, "Arm", label)
        classifications.append(classification)
        source_fingerprints[arm] = {}

        for width in WIDTHS:
            prediction, fingerprint = _load_current_prediction(arm_root, manifest, width)
            source_fingerprints[arm][str(width)] = fingerprint
            pred = prediction.set_index("timestamp")["pred"].astype(int)
            ledger, per_bar = simulate_fixed_hold(
                bars, pred, hold_bars=1, fee_bps=5.0
            )
            summary = economics_summary(per_bar)
            economic_rows.append({
                "Arm": label,
                "width_bps": width,
                "candidate_id": 0,
                "training_history_days": 180,
                "hold_minutes": 15,
                "fee_bps_per_side": 5.0,
                "fit_end": pd.to_datetime(prediction["train_end"].iloc[0], utc=True),
                "period_start": FORWARD_START,
                "period_end": FORWARD_END,
                "trades": int(len(ledger)),
                "n_long": int((ledger["side"] == 1).sum()),
                "n_short": int((ledger["side"] == -1).sum()),
                "gross_return": float(ledger["gross_return"].sum()),
                "net_return": float(summary["net_return_sum"]),
                "sortino": float(summary["sortino"]),
                "sharpe": float(summary["sharpe"]),
            })

    classification = pd.concat(classifications, ignore_index=True).sort_values(
        ["width_bps", "Arm"]
    ).reset_index(drop=True)
    economics = pd.DataFrame(economic_rows).sort_values(
        ["width_bps", "Arm"]
    ).reset_index(drop=True)
    if len(classification) != 9 or len(economics) != 9:
        raise ValueError("raw sentiment scoreboards must contain three arms x three DZ rows")
    if not economics["trades"].eq(economics["n_long"] + economics["n_short"]).all():
        raise ValueError("raw trade directions do not reconcile")
    if economics["period_end"].max() > FORWARD_END:
        raise ValueError("raw sentiment replay reached the 2026 Q2 lockbox")

    raw_manifest = {
        "protocol": "notebook02c_raw_sentiment_v1",
        "notebook02b_handoff_fingerprint": handoff["handoff_fingerprint"],
        "candidate_id": 0,
        "widths": list(WIDTHS),
        "training_history_days": 180,
        "hold_minutes": 15,
        "fee_bps_per_side": 5.0,
        "confidence_threshold": None,
        "tp_bps": None,
        "sl_bps": None,
        "policy_calibration_used": False,
        "one_minute_execution_used": False,
        "period_start": FORWARD_START.isoformat(),
        "period_end_exclusive": FORWARD_END.isoformat(),
        "sealed_lockbox": True,
        "source_prediction_fingerprints": source_fingerprints,
        "artifact_rows": {"classification_2024.parquet": 9, "raw_forward_economics.parquet": 9},
    }
    _write_artifacts(classification, economics, raw_manifest)
    return {"classification": classification, "economics": economics}


def build_calibrated_scoreboards(root: Path = DEFAULT_ROOT) -> dict[str, pd.DataFrame]:
    """Build the matched candidate-0 H1-policy and calibrated forward tables."""
    handoff = load_notebook02b_handoff(HANDOFF_PATH)
    policy_frames: list[pd.DataFrame] = []
    forward_frames: list[pd.DataFrame] = []
    arm_manifests: dict[str, dict] = {}
    for arm, label in ARM_LABELS.items():
        arm_root = _arm_root(arm, root)
        manifest = _read_manifest(arm_root, handoff["upstream_handoff_fingerprint"])
        arm_manifests[arm] = {
            "calibration_fit_ids": manifest["calibration"]["physical_fit_ids"],
            "forward_fit_ids": manifest["forward"]["physical_fit_ids"],
        }
        policies = pd.read_parquet(arm_root / "selected_policies_2025h1.parquet")
        policies = policies.loc[
            policies["objective"].eq("baseline")
            & policies["candidate_id"].eq(0)
            & policies["width_bps"].isin(WIDTHS)
        ].copy()
        forward = pd.read_parquet(arm_root / "forward_summary.parquet")
        forward = forward.loc[
            forward["objective"].eq("baseline")
            & forward["candidate_id"].eq(0)
            & forward["width_bps"].isin(WIDTHS)
        ].copy()
        if len(policies) != 3 or len(forward) != 3:
            raise ValueError(f"{arm}: expected three calibrated candidate-0 rows")
        if not policies["monthly_fit_count"].astype(int).eq(6).all():
            raise ValueError(f"{arm}: H1 calibration must contain six causal monthly fits")
        if not pd.to_datetime(forward["period_start"], utc=True).eq(FORWARD_START).all():
            raise ValueError(f"{arm}: calibrated forward start changed")
        if not pd.to_datetime(forward["period_end"], utc=True).eq(FORWARD_END).all():
            raise ValueError(f"{arm}: calibrated forward end changed")
        for field in ("policy_id", "tau", "tp_bps", "sl_bps", "max_hold"):
            left = policies.set_index("width_bps")[field].sort_index()
            right = forward.set_index("width_bps")[field].sort_index()
            if not left.equals(right):
                raise ValueError(f"{arm}: forward {field} differs from frozen H1 policy")
        policies.insert(0, "Arm", label)
        forward.insert(0, "Arm", label)
        policy_frames.append(policies)
        forward_frames.append(forward)

    policies = pd.concat(policy_frames, ignore_index=True).sort_values(
        ["width_bps", "Arm"]
    ).reset_index(drop=True)
    economics = pd.concat(forward_frames, ignore_index=True).sort_values(
        ["width_bps", "Arm"]
    ).reset_index(drop=True)
    if len(policies) != 9 or len(economics) != 9:
        raise ValueError("calibrated sentiment tables must contain three arms x three DZ rows")
    if not economics["trades"].eq(economics["n_long"] + economics["n_short"]).all():
        raise ValueError("calibrated trade directions do not reconcile")

    CALIBRATED_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    policies.to_parquet(CALIBRATED_POLICIES, index=False)
    economics.to_parquet(CALIBRATED_ECONOMICS, index=False)
    manifest = {
        "protocol": "notebook02d_catboost_sentiment_policy_v1",
        "notebook02b_handoff_fingerprint": handoff["handoff_fingerprint"],
        "candidate_id": 0,
        "widths": list(WIDTHS),
        "training_history_days": 180,
        "calibration_start": "2025-01-01T00:00:00+00:00",
        "calibration_end_exclusive": FORWARD_START.isoformat(),
        "calibration_months": 6,
        "policy_count_per_arm_width": 33,
        "policy_calibration_used": True,
        "one_minute_execution_used": True,
        "forward_start": FORWARD_START.isoformat(),
        "forward_end_exclusive": FORWARD_END.isoformat(),
        "sealed_lockbox": True,
        "arm_fit_manifests": arm_manifests,
        "artifact_rows": {
            "selected_policies_2025h1.parquet": 9,
            "calibrated_forward_economics.parquet": 9,
        },
    }
    manifest["manifest_fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    CALIBRATED_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"policies": policies, "economics": economics}
