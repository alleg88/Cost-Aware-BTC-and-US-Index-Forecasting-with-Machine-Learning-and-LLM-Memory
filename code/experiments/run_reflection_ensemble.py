"""Run the exploratory nine-model RQ4 memory comparison from January 2024."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import inspect
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from evaluation.economics import economics_summary
from experiments.catboost_execution_scoring import simulate_policy
from experiments.qualified_union import _read_market
from experiments.rq4_nine_model_data import (
    BAR, CACHE, CODE_ROOT, MODEL_NAMES, file_hash, load_panel, payload_hash,
)
from experiments.run_catboost_matched_ablation import _atomic_json, _atomic_parquet
from reflection_agent.ensemble import (
    WeightChoice, array_weights, candidate_outcomes, hedge_weights, memory_cards,
    messages_for, replay_signals, shuffle_cards, weight_signal,
)
from reflection_agent.v2.transport import DeepSeekSchemaCaller

STAGES = {
    "development": (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")),
    "h1": (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-07-01", tz="UTC")),
    "forward": (pd.Timestamp("2025-07-01", tz="UTC"), pd.Timestamp("2026-04-01", tz="UTC")),
}
LLM_ARMS = ("NoMemory", "RealMemory", "ShuffledMemory")
ARMS = (*LLM_ARMS, "Hedge", "LSTM_fixed")
MODEL_DIGEST = "5166728b9358990e5f6c34f87cbe48716be2f2cd2d3b98527dff27ea755bf3ba"


@dataclass(frozen=True)
class TransportConfig:
    model: str = "deepseek-v4-flash:cloud"
    think: str = "low"
    stream: bool = False
    temperature: float = 0.0
    num_predict: int = 16384
    timeout_seconds: float = 300.0
    repair_attempts: int = 1


def stage_blocks(stage: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start, end = STAGES[stage]
    edges = list(pd.date_range(start, end, freq="7D"))
    if edges[-1] != end:
        edges.append(end)
    return list(zip(edges[:-1], edges[1:]))


def current_state(bars: pd.DataFrame, times: pd.DatetimeIndex,
                  probabilities: np.ndarray, decision: pd.Timestamp) -> dict:
    known = bars.loc[bars.index + BAR <= decision].tail(7 * 96 + 1)
    if len(known) < 7 * 96 + 1:
        raise ValueError("current-state history is incomplete")
    close = known["close"]
    volume = known["volume"]
    last = int(times.searchsorted(decision, side="right")) - 1
    return {
        "return_1d_bps": round(float(close.iloc[-1] / close.iloc[-97] - 1) * 10000, 4),
        "return_7d_bps": round(float(close.iloc[-1] / close.iloc[0] - 1) * 10000, 4),
        "m15_volatility_7d_bps": round(float(close.pct_change().std()) * 10000, 4),
        "volume_1d_over_7d_daily_mean": round(float(volume.tail(96).sum() / (volume.tail(7 * 96).sum() / 7)), 4),
        "current_probabilities": probabilities[last].round(6).tolist() if last >= 0 else None,
        "prediction_age_minutes": float((decision - times[last]).total_seconds() / 60) if last >= 0 else None,
    }


def _checked_parquet(root: Path, name: str, hashes: dict[str, str]) -> pd.DataFrame:
    if file_hash(root / name) != hashes[name]:
        raise ValueError(f"artifact changed: {name}")
    return pd.read_parquet(root / name)


def prepare_inputs(root: Path) -> tuple[dict, dict, pd.DataFrame, dict, pd.DataFrame]:
    manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
    paths = [CODE_ROOT / "data" / name for name in (
        "btcusdt_m15_2024_2025.parquet", "btcusdt_1m_2024_2026.parquet",
        "btcusdt_1m_2025_2026.parquet", "btcusdt_15min_2021_2026.parquet",
    )]
    market_hashes = {str(p.relative_to(CODE_ROOT)): file_hash(p) for p in paths}
    start, end = STAGES["development"][0], STAGES["forward"][1]
    bars = _read_market(paths[0], start, end)
    history = _read_market(paths[3], start - pd.Timedelta(days=8), start)
    state_bars = pd.concat([history, bars])
    _atomic_parquet(state_bars.rename_axis("timestamp").reset_index(), root / "market_state_bars.parquet")
    path_signature = payload_hash({
        "market": market_hashes,
        "engine": file_hash(CODE_ROOT / "evaluation/trades_intrabar.py"),
        "wrapper": file_hash(CODE_ROOT / "experiments/catboost_execution_scoring.py"),
        "candidates": inspect.getsource(candidate_outcomes),
    })
    panels, outcomes, baselines = {}, {}, []
    equal_audit = {}
    expert_parts: dict[str, list[pd.DataFrame]] = {name: [] for name in MODEL_NAMES}
    for stage, (left, right) in STAGES.items():
        print(f"RQ4 prepare: {stage}", flush=True)
        reference, probabilities = load_panel(root, stage)
        cube = np.stack([probabilities[name] for name in MODEL_NAMES], axis=1)
        grid = bars.index[(bars.index >= left) & (bars.index < right)]
        union_name = manifest["union_signals"][stage]
        union = _checked_parquet(root, union_name, manifest["artifact_hashes"]).set_index("timestamp")["union_signal"]
        if not reference.timestamp.equals(pd.Series(union.index, name="timestamp")):
            raise ValueError("fallback and ensemble forecast grids differ")
        path = root / "execution_paths" / f"{stage}_{path_signature[:16]}.parquet"
        path_audit = path.with_suffix(".json")
        if path.exists() and path_audit.exists():
            audit = json.loads(path_audit.read_text(encoding="utf-8"))
            if audit["sha256"] != file_hash(path):
                raise ValueError("cached execution paths changed")
            outcome = pd.read_parquet(path)
        else:
            minute = _read_market(paths[1] if stage == "development" else paths[2], left, right)
            outcome = candidate_outcomes(bars, minute, left, right)
            _atomic_parquet(outcome, path)
            _atomic_json({"signature": path_signature, "sha256": file_hash(path)}, path_audit)
        outcomes[stage] = outcome
        panels[stage] = {"reference": reference, "cube": cube, "grid": grid, "union": union}
        for i, name in enumerate(MODEL_NAMES):
            probability = probabilities[name]
            signals = np.where(probability.max(axis=1) >= 0.55, probability.argmax(axis=1) - 1, 0)
            ledger, _ = replay_signals(grid, pd.Series(signals, index=reference.timestamp), outcome)
            ledger["stage"] = stage
            expert_parts[name].append(ledger)
            _atomic_parquet(ledger, root / "expert_ledgers" / f"{stage}_{name}.parquet")
        baseline_ref, baseline_probs = load_panel(root, stage, width=55, model_names=("lstm",))
        p = baseline_probs["lstm"]
        signal = pd.Series(np.where(p.max(axis=1) >= 0.75, p.argmax(axis=1) - 1, 0), index=baseline_ref.timestamp)
        ledger, returns = replay_signals(grid, signal, outcome, 200)
        _atomic_parquet(ledger, root / "ledgers" / f"{stage}_LSTM_fixed.parquet")
        _atomic_parquet(returns.rename_axis("timestamp").reset_index(), root / "returns" / f"{stage}_LSTM_fixed.parquet")
        metrics = summarise(stage, "LSTM_fixed", ledger, returns)
        baselines.append(metrics)
        if stage == "forward":
            legacy = pd.read_parquet(CODE_ROOT / "experiments/cache/tuning/all_model_stacking/forward_summary.parquet")
            gold = legacy.loc[legacy.sentiment_arm.eq("none") & legacy.objective.eq("best_single")].iloc[0]
            for metric in ("net_return", "sortino", "sharpe", "max_drawdown", "trades"):
                if not np.isclose(metrics[metric], gold[metric], atol=1e-10, rtol=0):
                    raise AssertionError(f"old LSTM benchmark changed: {metric}")
            # Also verify the equal-weight probability contract against notebook07.
            equal = weight_signal(cube, np.full((len(cube), 9), 1 / 9))
            equal_ledger, equal_returns = replay_signals(grid, pd.Series(equal, index=reference.timestamp), outcome)
            gold_equal = legacy.loc[legacy.sentiment_arm.eq("none") & legacy.objective.eq("soft_vote")].iloc[0]
            # Older stacking used a different saved fit generation. Preserve v3
            # inputs for all four new arms and record any threshold crossings.
            saved_equal = pd.read_parquet(CODE_ROOT / "experiments/cache/tuning/all_model_stacking/predictions/none/w65_soft_vote_forward.parquet")
            old_signal = np.where(saved_equal.confidence >= 0.55, saved_equal.pred - 1, 0)
            equal_audit = {"legacy_trades": int(gold_equal.trades), "v3_trades": len(equal_ledger),
                           "legacy_net": float(gold_equal.net_return), "v3_net": float(equal_returns.sum()),
                           "changed_raw_signals": int(np.sum(equal != old_signal)),
                           "reason": "older stacking forecast generation versus current v3 caches"}
            minute = _read_market(paths[2], left, right)
            direct_equal, direct_equal_returns = simulate_policy(
                bars=bars, execution=minute,
                prediction_frame=pd.DataFrame({"timestamp": reference.timestamp, "pred": equal + 1, "confidence": 1.0}),
                start=left, end=right, resolution="1m", tau=0, tp_bps=150,
                sl_bps=100, max_hold=1, fee_bps=5,
            )
            pd.testing.assert_frame_equal(equal_ledger[direct_equal.columns], direct_equal, check_dtype=False)
            np.testing.assert_allclose(equal_returns, direct_equal_returns, atol=1e-12)
        # Real-data equivalence for the unchanged fallback including occupancy.
        union_ledger, union_returns = replay_signals(grid, union, outcome, 200)
        minute = _read_market(paths[1] if stage == "development" else paths[2], left, right)
        direct, direct_returns = simulate_policy(
            bars=bars, execution=minute,
            prediction_frame=pd.DataFrame({"timestamp": union.index, "pred": union.to_numpy() + 1, "confidence": 1.0}),
            start=left, end=right, resolution="1m", tau=0, tp_bps=200,
            sl_bps=100, max_hold=1, fee_bps=5,
        )
        pd.testing.assert_frame_equal(union_ledger[direct.columns], direct, check_dtype=False)
        np.testing.assert_allclose(union_returns, direct_returns, atol=1e-12)
    experts = {name: pd.concat(parts, ignore_index=True).sort_values("available_time") for name, parts in expert_parts.items()}
    _atomic_json({"market_hashes": market_hashes, "path_signature": path_signature,
                  "old_forward_lstm_exact": True, "v3_soft_vote_replay_exact": True,
                  "legacy_equal_weight_comparison": equal_audit,
                  "union_replay_exact_all_stages": True, "q2_rows_read": 0}, root / "execution_audit.json")
    return panels, outcomes, state_bars, experts, pd.DataFrame(baselines)


def summarise(stage: str, arm: str, ledger: pd.DataFrame, returns: pd.Series,
              fallback_weeks: int = 0) -> dict:
    score = economics_summary(returns)
    return {"stage": stage, "arm": arm, "net_return": score["net_return_sum"],
            "sortino": score["sortino"], "sharpe": score["sharpe"],
            "max_drawdown": score["max_drawdown"], "trades": len(ledger),
            "n_long": int(ledger.side.eq(1).sum()), "n_short": int(ledger.side.eq(-1).sum()),
            "fallback_weeks": fallback_weeks}


def apply_choices(panel: dict, choices: list[dict], outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    reference, cube, grid = panel["reference"], panel["cube"], panel["grid"]
    allocation = np.full((len(reference), 9), 1 / 9)
    covered = np.zeros(len(reference), dtype=bool)
    fallback = np.zeros(len(reference), dtype=bool)
    for choice in choices:
        selected = reference.decision_time.ge(choice["decision_time"]) & reference.decision_time.lt(choice["end_time"])
        allocation[selected] = choice["weights"]
        covered[selected] = True
        fallback[selected] = choice["fallback"]
    tradable = reference.decision_time.lt(grid[-1] + BAR).to_numpy()
    if not covered[tradable].all():
        raise ValueError("weekly choices do not cover all predictions")
    signal = weight_signal(cube, allocation)
    signal[fallback] = panel["union"].reindex(pd.DatetimeIndex(reference.timestamp)).to_numpy(int)[fallback]
    tp = pd.Series(np.where(fallback, 200, 150), index=reference.timestamp).reindex(grid, fill_value=150)
    return replay_signals(grid, pd.Series(signal, index=reference.timestamp), outcomes, tp)


def hedge_choices(stage: str, cards: dict, eta: float) -> list[dict]:
    return [{"decision_time": left, "end_time": right, "weights": hedge_weights(cards[left], eta),
             "fallback": False} for left, right in stage_blocks(stage)]


def choose_eta(root: Path, panel: dict, outcomes: pd.DataFrame, cards: dict) -> float:
    rows = []
    for eta in (0.25, 0.5, 1.0, 2.0):
        ledger, returns = apply_choices(panel, hedge_choices("h1", cards, eta), outcomes)
        rows.append({"eta": eta, **summarise("h1", "Hedge", ledger, returns)})
    table = pd.DataFrame(rows)
    _atomic_parquet(table, root / "hedge_calibration.parquet")
    selected = float(table.sort_values(["net_return", "eta"], ascending=[False, True]).iloc[0].eta)
    print(f"RQ4 Hedge eta frozen on H1 only: {selected}", flush=True)
    return selected


def cached_call(root: Path, messages: list[dict], data_signature: str) -> dict:
    config = TransportConfig()
    request = {"messages": messages, "schema": WeightChoice.model_json_schema(),
               "transport": asdict(config), "model_digest": MODEL_DIGEST,
               "data_signature": data_signature}
    identity = payload_hash(request)
    path = root / "llm_calls" / f"{identity}.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["cache_key"] != identity or payload_hash(record["request"]) != identity:
            raise ValueError("cached LLM request changed")
        if record["validated_content"] is not None:
            WeightChoice.model_validate(record["validated_content"])
        return record
    result = DeepSeekSchemaCaller(config).call(
        role="weekly_nine_model_weights", messages=messages, response_model=WeightChoice,
        allowed_ids={"models": list(MODEL_NAMES)},
    )
    record = {"cache_key": identity, "request": request,
              "status": result.status, "validated_content": result.value.model_dump(mode="json") if result.value else None,
              "raw_content": result.raw_content, "request_hash": result.request_hash,
              "response_hash": result.response_hash, "schema_hash": result.schema_hash,
              "attempts": result.attempts, "latency_seconds": result.latency_seconds,
              "metadata": result.metadata, "errors": list(result.errors)}
    _atomic_json(record, path)
    return record


def paired_intervals(weekly: pd.DataFrame) -> pd.DataFrame:
    table = weekly.loc[weekly.stage.eq("forward")].pivot(index="decision_time", columns="arm", values="net_return")
    rng = np.random.default_rng(42)
    n, repetitions, length = len(table), 10000, 4
    starts = rng.integers(n, size=(repetitions, int(np.ceil(n / length))))
    indices = ((starts[:, :, None] + np.arange(length)) % n).reshape(repetitions, -1)[:, :n]
    rows = []
    for arm in ("NoMemory", "ShuffledMemory", "Hedge"):
        difference = (table.RealMemory - table[arm]).to_numpy(float)
        sampled = difference[indices].sum(axis=1)
        low, high = np.quantile(sampled, [0.025, 0.975])
        family_low, family_high = np.quantile(sampled, [0.05 / 6, 1 - 0.05 / 6])
        rows.append({"contrast": f"RealMemory - {arm}", "net_difference": float(difference.sum()),
                     "ci95_low": low, "ci95_high": high,
                     "familywise95_low": family_low, "familywise95_high": family_high,
                     "weeks": n, "bootstrap_block_weeks": length, "bootstrap_repetitions": repetitions})
    return pd.DataFrame(rows)


def run(root: Path = CACHE, *, prepare_only: bool = False, max_decisions: int | None = None,
        workers: int = 3) -> dict:
    started = time.monotonic()
    root = Path(root)
    manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
    panels, outcomes, bars, experts, baselines = prepare_inputs(root)
    cards = {left: memory_cards(experts, left, STAGES["development"][0])
             for stage in STAGES for left, _ in stage_blocks(stage)}
    eta = choose_eta(root, panels["h1"], outcomes["h1"], cards)
    protocol = {
        "version": "rq4-nine-model-weights-v5", "model_names": list(MODEL_NAMES),
        "transport": asdict(TransportConfig()), "model_digest": MODEL_DIGEST,
        "memory_completed_weeks": 4, "memory_metrics": ["net_return_bps", "trades", "maximum_additive_drawdown_bps"],
        "memory_feedback": "all-nine individual expert outcomes under the same DZ65 policy; actual minute-close availability < decision",
        "no_memory": "fresh independent request; empty performance cards; no previous weights or responses",
        "shuffle": "fixed-seed derangement of expert identities within each entire numeric card row",
        "hedge": "rolling four-week exponential weights; reward tanh(weekly_net_return / 0.01)",
        "hedge_eta": eta, "hedge_eta_grid": [0.25, 0.5, 1.0, 2.0], "eta_selection": "maximum H1 net; ties favour smaller eta",
        "common_policy": {"width_bps": 65, "tau": 0.55, "tp_bps": 150, "sl_bps": 100, "max_hold": 1, "fee_bps_per_side": 5},
        "baseline": "old LSTM DZ55/tau0.75/TP200/SL100/hold1; exact old forward score verified",
        "fallback": "exact Qualified Union signals and TP200 only after failed LLM output; ordinary Flat stays out",
        "seed": 42, "stages": {key: [str(x) for x in value] for key, value in STAGES.items()},
        "primary_period": "2025-07-01 through 2026-03-31; reused exploratory evidence",
        "secondary_periods": "2024 development and H1-2025 calibration; common trading policy selected using H1-2025",
        "llm_calendar_dates_and_price_levels_sent": False,
        "limitation": "LLM training data may include historical market knowledge; calendar masking cannot prove absence of pretraining contamination",
        "source_hashes": {"data_manifest.json": file_hash(root / "data_manifest.json"),
                          "engine": file_hash(CODE_ROOT / "reflection_agent/ensemble.py"),
                          "runner": file_hash(Path(__file__))}, "q2_rows_read": 0,
    }
    _atomic_json(protocol, root / "protocol.json")
    if prepare_only:
        return {"status": "prepared", "hedge_eta": eta}
    import ollama
    available = ollama.Client(timeout=10).list().models
    target = [model for model in available if model.model == TransportConfig().model]
    if len(target) != 1 or target[0].digest != MODEL_DIGEST:
        raise RuntimeError("required exact DeepSeek model tag/digest is unavailable")
    times = pd.DatetimeIndex(pd.concat([v["reference"].decision_time for v in panels.values()], ignore_index=True))
    all_probabilities = np.concatenate([v["cube"] for v in panels.values()])
    choices: dict[tuple[str, str], list[dict]] = {(stage, arm): [] for stage in STAGES for arm in LLM_ARMS}
    audit_rows = []
    total = sum(len(stage_blocks(stage)) for stage in STAGES)
    done = 0
    for stage in STAGES:
        for left, right in stage_blocks(stage):
            state = current_state(bars, times, all_probabilities, left)
            memories = {"NoMemory": [], "RealMemory": cards[left],
                        "ShuffledMemory": shuffle_cards(cards[left], 42 + int(left.value // 10**9))}
            messages = {arm: messages_for(state, memories[arm]) for arm in LLM_ARMS}
            # Empty-memory or otherwise identical payloads share one actual call.
            unique = {payload_hash(value): value for value in messages.values()}
            with ThreadPoolExecutor(max_workers=min(workers, len(unique))) as pool:
                records = dict(zip(unique, pool.map(lambda value: cached_call(root, value, manifest["data_signature"]), unique.values())))
            if done == 0 and all(record["validated_content"] is None for record in records.values()):
                raise RuntimeError("LLM preflight failed; no ensemble comparison reported")
            for arm in LLM_ARMS:
                record = records[payload_hash(messages[arm])]
                content = record["validated_content"]
                fallback = content is None
                weights = np.full(9, 1 / 9) if fallback else array_weights(WeightChoice.model_validate(content))
                choice = {"decision_time": left, "end_time": right, "weights": weights, "fallback": fallback}
                choices[(stage, arm)].append(choice)
                audit_rows.append({"stage": stage, "arm": arm, "decision_time": left, "end_time": right,
                                   "status": record["status"], "fallback": fallback,
                                   "memory_cards": len(memories[arm]), "request_hash": record["request_hash"],
                                   "cache_key": record["cache_key"], "attempts": record["attempts"],
                                   "reason": content["reason"] if content else "Qualified Union fallback",
                                   **{f"w_{name}": float(weights[i]) for i, name in enumerate(MODEL_NAMES)}})
            done += 1
            _atomic_parquet(pd.DataFrame(audit_rows), root / "weight_decisions.parquet")
            _atomic_json({"status": "running", "completed_weeks": done, "total_weeks": total,
                          "logical_decisions": len(audit_rows), "fallback_weeks": int(sum(row["fallback"] for row in audit_rows)),
                          "stage": stage, "decision_time": left, "elapsed_seconds": round(time.monotonic() - started, 1)}, root / "run_state.json")
            print(f"RQ4 LLM weeks {done}/{total}: {stage} {left.date()}, " + ", ".join(f"{arm}={audit_rows[-3+i]['status']}" for i, arm in enumerate(LLM_ARMS)), flush=True)
            if sum(row["fallback"] for row in audit_rows) > max(3, 0.05 * len(audit_rows)):
                raise RuntimeError("LLM failures exceed the fixed five-percent completion gate")
            if max_decisions is not None and done >= max_decisions:
                return {"status": "partial", "completed_weeks": done, "total_weeks": total}
    summaries = baselines.to_dict("records")
    weekly_rows = []
    for stage in STAGES:
        hedge = hedge_choices(stage, cards, eta)
        for choice in hedge:
            audit_rows.append({"stage": stage, "arm": "Hedge", "decision_time": choice["decision_time"],
                               "end_time": choice["end_time"], "status": "algorithm", "fallback": False,
                               "memory_cards": len(cards[choice["decision_time"]]),
                               **{f"w_{name}": float(choice["weights"][i]) for i, name in enumerate(MODEL_NAMES)}})
        for arm in ARMS:
            if arm == "LSTM_fixed":
                returns = pd.read_parquet(root / "returns" / f"{stage}_{arm}.parquet").set_index("timestamp")["bracket_return"]
            else:
                selected = hedge if arm == "Hedge" else choices[(stage, arm)]
                ledger, returns = apply_choices(panels[stage], selected, outcomes[stage])
                summaries.append(summarise(stage, arm, ledger, returns, sum(c["fallback"] for c in selected)))
                _atomic_parquet(ledger, root / "ledgers" / f"{stage}_{arm}.parquet")
                _atomic_parquet(returns.rename_axis("timestamp").reset_index(), root / "returns" / f"{stage}_{arm}.parquet")
            for left, right in stage_blocks(stage):
                weekly_rows.append({"stage": stage, "arm": arm, "decision_time": left, "end_time": right,
                                    "net_return": float(returns.loc[(returns.index >= left) & (returns.index < right)].sum())})
    summary, weekly = pd.DataFrame(summaries), pd.DataFrame(weekly_rows)
    _atomic_parquet(summary, root / "summary.parquet")
    _atomic_parquet(weekly, root / "weekly_returns.parquet")
    _atomic_parquet(paired_intervals(weekly), root / "paired_comparisons.parquet")
    _atomic_parquet(pd.DataFrame(audit_rows), root / "weight_decisions.parquet")
    call_keys = sorted({row["cache_key"] for row in audit_rows if "cache_key" in row})
    call_table = pd.DataFrame({"cache_key": call_keys,
                              "record_json": [(root / "llm_calls" / f"{key}.json").read_text(encoding="utf-8") for key in call_keys]})
    _atomic_parquet(call_table, root / "llm_call_audit.parquet")
    result = {"status": "complete", "completed_weeks": done, "total_weeks": total,
              "logical_llm_decisions": done * 3, "unique_llm_calls": len({row["cache_key"] for row in audit_rows if "cache_key" in row}),
              "fallback_weeks": int(sum(row["fallback"] for row in audit_rows)),
              "elapsed_seconds": round(time.monotonic() - started, 1)}
    _atomic_json(result, root / "run_state.json")
    result["artifact_hashes"] = {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()
                                 and p.suffix in {".parquet", ".json"} and p.name != "run_manifest.json"}
    _atomic_json(result, root / "run_manifest.json")
    print(summary.loc[summary.stage.eq("forward")].to_string(index=False), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=CACHE)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-decisions", type=int)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    result = run(args.output_root, prepare_only=args.prepare_only, max_decisions=args.max_decisions, workers=args.workers)
    print(json.dumps({k: v for k, v in result.items() if k != "artifact_hashes"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
