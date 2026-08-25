"""Run the frozen CatBoost 1m-versus-1s economic execution study."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from experiments.catboost_execution_scoring import (
    compare_paired_ledgers,
    score_continuous_policy_grid,
    simulate_policy as _simulate_policy,
)
from experiments.catboost_execution_runner import (
    ExecutionResolutionRunner,
    run_paired_replay,
)

def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json
    import traceback

    from experiments.catboost_execution_resolution import (
        PartitionedIntrabarStore,
        write_run_state,
    )
    from experiments.catboost_matched_ablation import WIDTHS, load_candidates
    from experiments.run_catboost_matched_ablation import (
        CACHE_ROOT as MATCHED_CACHE_ROOT,
        MINUTE_PATH,
        RecordingToyModel,
        _atomic_json,
        _atomic_parquet,
        load_prepared_data,
    )
    from models.zoo import MODELS

    code_root = Path(__file__).resolve().parents[1]
    default_root = (
        code_root / "experiments" / "cache" / "tuning" / "catboost_execution_resolution"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--arm", choices=("all", "1m", "1s"), default="all")
    parser.add_argument("--width", type=int, choices=WIDTHS, default=55)
    parser.add_argument("--candidate-id", type=int, default=0)
    parser.add_argument("--stage1-only", action="store_true")
    parser.add_argument("--output-root", type=Path, default=default_root)
    parser.add_argument("--prediction-root", type=Path, default=MATCHED_CACHE_ROOT)
    parser.add_argument("--minute-path", type=Path, default=MINUTE_PATH)
    parser.add_argument(
        "--one-second-root",
        type=Path,
        default=code_root / "data" / "btcusdt_1s_2024_2026",
    )
    args = parser.parse_args(argv)
    candidates = load_candidates()
    if args.full:
        widths = WIDTHS
        selected_candidates = candidates
        candidate_ids = tuple(range(len(candidates)))
        model_factory = MODELS["catboost_balanced"]
        fold_limit = 5
        stage1_only = args.stage1_only
    else:
        if not 0 <= args.candidate_id < len(candidates):
            parser.error("candidate-id is outside the tracked pool")
        widths = (args.width,)
        selected_candidates = (candidates[args.candidate_id],)
        candidate_ids = (args.candidate_id,)
        model_factory = RecordingToyModel
        fold_limit = 1
        stage1_only = True

    state_path = args.output_root / "run_state.json"
    write_run_state(state_path, status="running", detail={"stage": "prepare"})
    try:
        prepared = load_prepared_data(widths=widths, minute_path=args.minute_path)
        requested = ("1m", "1s") if args.arm == "all" else (args.arm,)
        stores = {}
        if "1m" in requested:
            stores["1m"] = PartitionedIntrabarStore.one_minute(args.minute_path)
        if "1s" in requested:
            stores["1s"] = PartitionedIntrabarStore.one_second(args.one_second_root)
        if args.smoke:
            import numpy as np

            smoke_start = pd.Timestamp("2024-01-15", tz="UTC")
            smoke_end = pd.Timestamp("2024-01-16", tz="UTC")
            smoke_bars = prepared.bars.loc[
                (prepared.bars.index >= smoke_start)
                & (prepared.bars.index < smoke_end)
            ]
            if smoke_bars.empty:
                raise ValueError("real-data smoke span has no M15 bars")
            sequence = np.arange(len(smoke_bars))
            predictions = pd.DataFrame(
                {
                    "timestamp": smoke_bars.index,
                    "pred": np.where(sequence % 3 == 0, 2, np.where(sequence % 3 == 1, 0, 1)),
                    "confidence": 0.9,
                }
            )
            smoke_rows = {}
            for resolution in requested:
                grid = score_continuous_policy_grid(
                    stage="real_data_smoke",
                    width_bps=args.width,
                    candidate_id=args.candidate_id,
                    prediction_frame=predictions,
                    bars=prepared.bars,
                    execution=stores[resolution].load_span(smoke_start, smoke_end),
                    regimes=prepared.regimes,
                    start=smoke_start,
                    end=smoke_end,
                    resolution=resolution,
                    fee_bps=5.0,
                )
                if len(grid) != 66:
                    raise AssertionError("real-data smoke did not score all 66 policies")
                _atomic_parquet(
                    grid,
                    args.output_root / "smoke" / f"{resolution}_policy_grid.parquet",
                )
                smoke_rows[resolution] = len(grid)
            result = {
                "status": "smoke_complete",
                "span_start": smoke_start.isoformat(),
                "span_end_exclusive": smoke_end.isoformat(),
                "policy_rows": smoke_rows,
            }
            write_run_state(state_path, status="complete", detail=result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        runners = {}
        results = {}
        arm_names = {"1m": "one_minute", "1s": "one_second"}
        for resolution in requested:
            write_run_state(
                state_path,
                status="running",
                detail={"stage": "arm", "resolution": resolution},
            )
            runner = ExecutionResolutionRunner(
                output_root=args.output_root / arm_names[resolution],
                prediction_root=args.prediction_root,
                store=stores[resolution],
                prepared=prepared,
                widths=widths,
                candidates=selected_candidates,
                candidate_ids=candidate_ids,
                model_factory=model_factory,
                smoke=args.smoke,
                fold_limit=fold_limit,
                stage1_only=stage1_only,
            )
            runners[resolution] = runner
            results[resolution] = runner.run()

        paired = {}
        if set(runners) == {"1m", "1s"} and not stage1_only:
            primary = pd.concat(
                [
                    pd.read_parquet(runner.output_root / "forward_summary.parquet")
                    for runner in runners.values()
                ],
                ignore_index=True,
            )
            _atomic_parquet(primary, args.output_root / "primary_comparison.parquet")
            write_run_state(state_path, status="running", detail={"stage": "paired_replay"})
            paired = run_paired_replay(runners, output_root=args.output_root)
        result = {"status": "complete", "arms": results, **paired}
        _atomic_json(result, args.output_root / "result.json")

        write_run_state(state_path, status="complete", detail=result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception:
        write_run_state(
            state_path,
            status="failed",
            detail={"traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())