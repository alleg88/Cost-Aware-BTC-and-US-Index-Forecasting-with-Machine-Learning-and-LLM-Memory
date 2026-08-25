"""Run the dev-only four-model contest for Strict-5m and Fast-T2 +2 bps."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from evaluation.channel_window_validation import expanding_purged_folds
from experiments.fast_t2_study import DEV_END, DEV_START
from experiments.two_trigger_model_study import (
    MODEL_NAMES,
    EventLabelConfig,
    build_event_dataset,
    run_oof_models,
    split_statistics,
    summarise_oof_predictions,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = CODE_ROOT / "experiments" / "cache" / "channel_5m_two_trigger" / "fast_t2"
OUT = CODE_ROOT / "experiments" / "cache" / "two_trigger_model_study" / "dev"
ARMS = {
    "strict_5m": SOURCE / "strict_5m" / "window_manifest.parquet",
    "fast_t2_2bps": SOURCE / "fast_2bps" / "window_manifest.parquet",
}


def _write_state(**values: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run_state.json").write_text(
        json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _load_rich_minutes() -> pd.DataFrame:
    path = CODE_ROOT / "data" / "btcusdt_1m_2021_2026.parquet"
    return pd.read_parquet(
        path,
        columns=[
            "open", "high", "low", "close", "volume", "taker_buy_base", "count"
        ],
        filters=[("timestamp", ">=", DEV_START), ("timestamp", "<", DEV_END)],
    ).sort_index()


def prepare_datasets(*, rebuild: bool = False) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    OUT.mkdir(parents=True, exist_ok=True)
    channel = pd.read_parquet(SOURCE / "channel_context.parquet")
    minute = _load_rich_minutes()
    config = EventLabelConfig()
    audits: dict[str, object] = {
        "period_start": DEV_START.isoformat(),
        "period_end_exclusive": DEV_END.isoformat(),
        "label_config": asdict(config),
        "arms": {},
    }
    events_by_arm: dict[str, pd.DataFrame] = {}
    for arm, manifest_path in ARMS.items():
        event_path = OUT / f"events_{arm}.parquet"
        if event_path.exists() and not rebuild:
            events = pd.read_parquet(event_path)
            audit = {"resumed": True, "labelled_events": int(len(events))}
        else:
            windows = pd.read_parquet(manifest_path)
            events, audit = build_event_dataset(
                windows, channel, minute, arm=arm, config=config
            )
            events.to_parquet(event_path, index=False)
        folds = expanding_purged_folds(events)
        splits = split_statistics(events, folds)
        splits.insert(0, "arm", arm)
        splits.to_csv(OUT / f"splits_{arm}.csv", index=False)
        events_by_arm[arm] = events
        audits["arms"][arm] = audit
        print(f"{arm}: {len(events):,} labelled events", flush=True)
    (OUT / "dataset_audit.json").write_text(
        json.dumps(audits, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return events_by_arm, minute


def run(*, models: tuple[str, ...], rebuild: bool = False) -> None:
    unknown = sorted(set(models).difference(MODEL_NAMES))
    if unknown:
        raise ValueError(f"unsupported models: {unknown}")
    _write_state(status="running", active="dataset", completed=[])
    events_by_arm, minute = prepare_datasets(rebuild=rebuild)
    completed: list[str] = []
    for arm, events in events_by_arm.items():
        for model in models:
            key = f"{arm}:{model}"
            prediction_path = OUT / f"oof_{arm}_{model}.parquet"
            audit_path = OUT / f"audit_{arm}_{model}.csv"
            if prediction_path.exists() and audit_path.exists() and not rebuild:
                completed.append(key)
                print(f"resume {key}", flush=True)
                continue
            _write_state(status="running", active=key, completed=completed)
            print(f"fit {key}", flush=True)
            predictions, audit = run_oof_models(
                events,
                minute if model == "gru" else None,
                model_names=(model,),
            )
            predictions.to_parquet(prediction_path, index=False)
            audit.to_csv(audit_path, index=False)
            completed.append(key)

    predictions = []
    audits = []
    for arm in ARMS:
        for model in models:
            predictions.append(pd.read_parquet(OUT / f"oof_{arm}_{model}.parquet"))
            audit = pd.read_csv(OUT / f"audit_{arm}_{model}.csv")
            audit.insert(0, "arm", arm)
            audits.append(audit)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    audit_frame = pd.concat(audits, ignore_index=True)
    summary = summarise_oof_predictions(prediction_frame)
    prediction_frame.to_parquet(OUT / "oof_all.parquet", index=False)
    audit_frame.to_csv(OUT / "model_fold_audit.csv", index=False)
    summary.to_csv(OUT / "model_summary.csv", index=False)
    split_frame = pd.concat(
        [pd.read_csv(OUT / f"splits_{arm}.csv") for arm in ARMS], ignore_index=True
    )
    split_frame.to_csv(OUT / "split_statistics.csv", index=False)
    protocol = {
        "stage": "dev",
        "period_start": DEV_START.isoformat(),
        "period_end_exclusive": DEV_END.isoformat(),
        "models": list(models),
        "arms": list(ARMS),
        "model_architecture": "pooled long/short with side_sign",
        "validation": "seven expanding six-month episode-purged folds",
        "selection_metrics": [
            "ROC-AUC", "Brier", "p>=0.5 net R", "within-fold top-30% net R"
        ],
        "forward_or_lockbox_loaded": False,
    }
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_state(status="complete", active=None, completed=completed)
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", *MODEL_NAMES), default="all")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--dataset-only", action="store_true")
    args = parser.parse_args()
    if args.dataset_only:
        prepare_datasets(rebuild=args.rebuild)
        return
    models = MODEL_NAMES if args.model == "all" else (args.model,)
    run(models=models, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
