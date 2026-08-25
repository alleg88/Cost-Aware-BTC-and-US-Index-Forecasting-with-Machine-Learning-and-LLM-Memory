"""Add matched XGBoost rows to Notebook G without refitting its baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.channel_window_validation import expanding_purged_folds
from experiments.run_fast_t2_economic_entry import (
    ECONOMIC_ARMS as BASE_ARMS,
    OUT as BASE_OUT,
    RR_SENSITIVITY,
    _artifact_paths,
    _artifacts_exist,
    _consolidate,
    _json_hash,
    _load_entry_inputs,
    _write_state,
    score_outer_economic_fold,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
OUT = CODE_ROOT / "experiments" / "cache" / "fast_t2_economic_xgboost" / "dev"
XGBOOST_ARMS = ("xgboost_pooled", "xgboost_split_side")


def build_xgboost_extension_protocol(
    entry_protocol: dict[str, object],
    entry_manifest: dict[str, object],
    *,
    base_protocol_hash: str,
) -> dict[str, object]:
    """Register a matched, dev-only model extension to the completed G run."""
    if bool(entry_protocol.get("forward_or_lockbox_loaded")):
        raise ValueError("forward and lockbox must remain sealed")
    if entry_protocol.get("period_end_exclusive") != "2025-07-01T00:00:00+00:00":
        raise ValueError("XGBoost continuation must remain dev-only")
    if not base_protocol_hash:
        raise ValueError("base economic protocol hash is required")
    payload: dict[str, object] = {
        "stage": "development",
        "role": "post-hoc matched model extension",
        "post_hoc_model_extension": True,
        "base_protocol_hash": base_protocol_hash,
        "base_arms_reused": list(BASE_ARMS),
        "catboost_retrained": False,
        "logreg_retrained": False,
        "source_entry_protocol_hash": entry_protocol["protocol_hash"],
        "source_dataset_hash": entry_manifest["dataset_hash"],
        "source_decision_ledger_hash": entry_manifest["decision_ledger_hash"],
        "period_start": entry_protocol["period_start"],
        "period_end_exclusive": entry_protocol["period_end_exclusive"],
        "target": "winsorised_net_r_enter_minus_skip_0R",
        "target_winsorisation": "training-only 1st/99th percentiles",
        "arms": list(XGBOOST_ARMS),
        "model_parameters": {
            "xgboost": {
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.03,
                "reg_lambda": 10.0,
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": 1,
            }
        },
        "rr_sensitivity": list(RR_SENSITIVITY),
        "primary_rr": None,
        "minimum_trades_per_day": 1.0,
        "threshold_selection": (
            "inner chronological robust net R under frequency floor"
        ),
        "validation": "seven expanding six-month episode-purged folds",
        "uniqueness_weighting": "training labels only; no class balancing",
        "primary_portfolio": "all signals (unlimited)",
        "forward_or_lockbox_loaded": False,
    }
    payload["protocol_hash"] = _json_hash(payload)
    return payload


def _write_protocol(
    output_dir: Path,
    expected: dict[str, object],
    *,
    rebuild: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "protocol.json"
    if path.exists() and not rebuild:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("protocol_hash") != expected["protocol_hash"]:
            raise ValueError("XGBoost protocol hash mismatch; use --rebuild")
        return stored
    path.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return expected


def _load_completed_base() -> tuple[dict[str, object], dict[str, object]]:
    required = ("protocol.json", "run_state.json", "economic_policy_summary.csv")
    missing = [name for name in required if not (BASE_OUT / name).exists()]
    if missing:
        raise FileNotFoundError(f"completed Notebook G artifacts missing: {missing}")
    protocol = json.loads((BASE_OUT / "protocol.json").read_text(encoding="utf-8"))
    state = json.loads((BASE_OUT / "run_state.json").read_text(encoding="utf-8"))
    if state.get("status") != "complete" or not state.get("consolidated"):
        raise ValueError("XGBoost extension requires a completed Notebook G run")
    if bool(protocol.get("forward_or_lockbox_loaded")) or bool(
        state.get("forward_or_lockbox_loaded")
    ):
        raise ValueError("forward and lockbox must remain sealed")
    return protocol, state


def _mark_extension_result(output_dir: Path) -> None:
    path = output_dir / "economic_result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result.update(
        {
            "post_hoc_model_extension": True,
            "catboost_retrained": False,
            "logreg_retrained": False,
            "base_artifacts_mutated": False,
            "comparison_not_promotion_test": True,
            "removed_from_active_path": [
                "early_exit_model",
                "accuracy_or_win_rate_selection",
                "gru_lstm_entry_models",
                "capacity_threshold_search",
                "hard_rr_gate",
            ],
            "forward_or_lockbox_loaded": False,
        }
    )
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run(*, rebuild: bool = False, output_dir: Path = OUT) -> None:
    """Fit only XGBoost pooled and split-side arms on the frozen G protocol."""
    output_dir = Path(output_dir)
    if output_dir.resolve() == BASE_OUT.resolve():
        raise ValueError("XGBoost extension must not overwrite base Notebook G artifacts")
    decisions, entry_protocol, entry_manifest = _load_entry_inputs()
    base_protocol, _ = _load_completed_base()
    expected = build_xgboost_extension_protocol(
        entry_protocol,
        entry_manifest,
        base_protocol_hash=str(base_protocol["protocol_hash"]),
    )
    protocol = _write_protocol(output_dir, expected, rebuild=rebuild)
    folds = expanding_purged_folds(decisions)
    if len(folds) != 7:
        raise AssertionError(f"expected seven development folds, found {len(folds)}")

    completed: list[str] = []
    (output_dir / "folds").mkdir(parents=True, exist_ok=True)
    _write_state(
        output_dir,
        status="running",
        active="folds",
        completed=completed,
        consolidated=False,
        protocol_hash=protocol["protocol_hash"],
        base_protocol_hash=base_protocol["protocol_hash"],
        catboost_retrained=False,
        logreg_retrained=False,
        forward_or_lockbox_loaded=False,
    )
    for fold in folds:
        for arm in XGBOOST_ARMS:
            key = f"{fold.fold_id}:{arm}"
            paths = _artifact_paths(output_dir, fold.fold_id, arm)
            if _artifacts_exist(paths) and not rebuild:
                completed.append(key)
                continue
            _write_state(
                output_dir,
                status="running",
                active=key,
                completed=completed,
                consolidated=False,
                protocol_hash=protocol["protocol_hash"],
                base_protocol_hash=base_protocol["protocol_hash"],
                catboost_retrained=False,
                logreg_retrained=False,
                forward_or_lockbox_loaded=False,
            )
            scores, entries, actions, frontier, audit = score_outer_economic_fold(
                decisions, fold, arm
            )
            scores.to_parquet(paths["scores"], index=False)
            entries.to_parquet(paths["entries"], index=False)
            actions.to_parquet(paths["actions"], index=False)
            frontier.to_csv(paths["frontier"], index=False)
            paths["audit"].write_text(
                json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            completed.append(key)
        print(f"complete XGBoost fold {fold.fold_id}", flush=True)

    full_matrix = all(
        _artifacts_exist(_artifact_paths(output_dir, fold.fold_id, arm))
        for fold in folds
        for arm in XGBOOST_ARMS
    )
    if not full_matrix:
        raise AssertionError("XGBoost extension matrix is incomplete")
    _consolidate(
        output_dir=output_dir,
        arms=XGBOOST_ARMS,
        folds=folds,
        protocol=protocol,
    )
    _mark_extension_result(output_dir)
    _write_state(
        output_dir,
        status="complete",
        active=None,
        completed=completed,
        expected=len(folds) * len(XGBOOST_ARMS),
        consolidated=True,
        protocol_hash=protocol["protocol_hash"],
        base_protocol_hash=base_protocol["protocol_hash"],
        max_loaded_timestamp=entry_manifest["max_loaded_timestamp"],
        catboost_retrained=False,
        logreg_retrained=False,
        forward_or_lockbox_loaded=False,
    )
    print(f"complete: {len(completed)}/{len(folds) * len(XGBOOST_ARMS)} XGBoost cells")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    run(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
