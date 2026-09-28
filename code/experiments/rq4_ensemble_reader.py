"""Lightweight independent audit and reader inputs for the RQ4 weight experiment."""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BAR = pd.Timedelta(minutes=15)
PROBS = ["p_short", "p_flat", "p_long"]


def digest(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def reader_files(root: Path) -> list[str]:
    data = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
    execution = json.loads((root / "execution_audit.json").read_text(encoding="utf-8"))
    files = [
        "data_manifest.json", "protocol.json", "execution_audit.json", "run_state.json",
        "summary.parquet", "weight_decisions.parquet", "weekly_returns.parquet",
        "paired_comparisons.parquet", "hedge_calibration.parquet", "training_audit.parquet",
        "saved_oof_checks.parquet", "market_state_bars.parquet", "llm_call_audit.parquet",
    ]
    files.extend(data["artifact_hashes"])
    files.extend(f"execution_paths/{stage}_{execution['path_signature'][:16]}.parquet" for stage in data["predictions"])
    files.extend(str(path.relative_to(root)).replace("\\", "/")
                 for folder in ("expert_ledgers", "ledgers", "returns")
                 for path in (root / folder).glob("*.parquet"))
    files.extend(name for name in ("acquisition_run_state.json", "validation_boundary_audit.json",
                                  "offline_replay_audit.json") if (root / name).is_file())
    return sorted(set(files))


def seal_reader(root: Path) -> dict:
    state = json.loads((root / "run_state.json").read_text(encoding="utf-8"))
    if state["status"] != "complete":
        raise ValueError("cannot seal an incomplete experiment")
    manifest = {"version": "rq4-nine-model-weights-reader-v5",
                "reader_code_sha256": digest(Path(__file__)),
                "files": {name: digest(root / name) for name in reader_files(root)}}
    (root / "reader_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _cards(experts: dict, names: list[str], decision: pd.Timestamp,
           observed_start: pd.Timestamp) -> list[dict]:
    result = []
    for age in range(4, 0, -1):
        left = decision - pd.Timedelta(days=7 * age)
        right = left + pd.Timedelta(days=7)
        if left < observed_start:
            continue
        values = []
        for name in names:
            source = experts[name]
            realised = source.loc[source.available_time.ge(left) & source.available_time.lt(right)]
            returns = realised.sort_values("available_time").net_return.to_numpy(float)
            curve = np.r_[0.0, returns.cumsum()]
            values.append([round(10000 * float(returns.sum()), 6), len(realised),
                           round(10000 * float((np.maximum.accumulate(curve) - curve).max()), 6)])
        result.append({"age_weeks": age, "values": values})
    return result


def _shuffled(cards: list[dict], decision: pd.Timestamp, count: int) -> list[dict]:
    result = []
    for card in cards:
        rng = np.random.default_rng(42 + int(decision.value // 10**9) + card["age_weeks"])
        permutation = rng.permutation(count)
        while np.any(permutation == np.arange(count)):
            permutation = rng.permutation(count)
        result.append({"age_weeks": card["age_weeks"],
                       "values": [card["values"][int(i)] for i in permutation]})
    return result


def _expected_state(bars: pd.DataFrame, time: pd.DatetimeIndex, cube: np.ndarray,
                    decision: pd.Timestamp) -> dict:
    past = bars.loc[bars.index <= decision - BAR].iloc[-673:]
    prices, volume = past.close.to_numpy(float), past.volume.to_numpy(float)
    index = time.searchsorted(decision, side="right") - 1
    return {"return_1d_bps": round(float(prices[-1] / prices[-97] - 1) * 10000, 4),
            "return_7d_bps": round(float(prices[-1] / prices[0] - 1) * 10000, 4),
            "m15_volatility_7d_bps": round(float(np.std(prices[1:] / prices[:-1] - 1, ddof=1)) * 10000, 4),
            "volume_1d_over_7d_daily_mean": round(float(volume[-96:].sum() / (volume[-672:].sum() / 7)), 4),
            "current_probabilities": cube[index].round(6).tolist() if index >= 0 else None,
            "prediction_age_minutes": float((decision - time[index]).total_seconds() / 60) if index >= 0 else None}


def _metrics(returns: pd.Series) -> dict:
    values = returns.to_numpy(float)
    deviation = np.std(values, ddof=1)
    downside = np.sqrt(np.mean(np.minimum(values, 0) ** 2))
    curve = np.cumsum(values)
    return {"net_return": float(values.sum()),
            "sharpe": float(values.mean() / deviation * np.sqrt(35040)) if deviation > 0 else 0.0,
            "sortino": float(values.mean() / downside * np.sqrt(35040)) if downside > 0 else 0.0,
            "max_drawdown": float((np.maximum.accumulate(curve) - curve).max())}


def _replay(grid: pd.DatetimeIndex, signal: pd.Series, tp: pd.Series,
            paths: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    sides = signal.reindex(grid, fill_value=0).to_numpy(int)
    potential = np.flatnonzero(sides[:-1])
    selected, next_free = [], 0
    for index in potential:
        if index >= next_free:
            selected.append(index)
            next_free = index + 2
    keys = [(grid[i], int(tp.loc[grid[i]]), int(sides[i])) for i in selected]
    indexed = paths.set_index(["signal_time", "policy_tp_bps", "side"], drop=False)
    ledger = indexed.loc[keys].reset_index(drop=True) if keys else paths.iloc[:0].copy()
    returns = pd.Series(0.0, index=grid, name="bracket_return")
    returns.loc[pd.DatetimeIndex(ledger.entry_time)] = ledger.net_return.to_numpy(float)
    return ledger, returns


def verify_reader(root: Path) -> dict:
    root = Path(root)
    manifest = json.loads((root / "reader_manifest.json").read_text(encoding="utf-8"))
    if digest(Path(__file__)) != manifest["reader_code_sha256"]:
        raise ValueError("reader implementation changed")
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f"reader input changed: {name}")
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    data = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
    execution = json.loads((root / "execution_audit.json").read_text(encoding="utf-8"))
    state = json.loads((root / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete" and protocol["q2_rows_read"] == execution["q2_rows_read"] == 0
    names = protocol["model_names"]
    stages = dict(sorted(protocol["stages"].items(), key=lambda item: pd.Timestamp(item[1][0])))
    assert len(names) == 9 and execution["old_forward_lstm_exact"]
    summary = pd.read_parquet(root / "summary.parquet")
    calibration = pd.read_parquet(root / "hedge_calibration.parquet")
    assert calibration.stage.eq("h1").all()
    assert float(calibration.sort_values(["net_return", "eta"], ascending=[False, True]).iloc[0].eta) == protocol["hedge_eta"]
    decisions = pd.read_parquet(root / "weight_decisions.parquet")
    weekly = pd.read_parquet(root / "weekly_returns.parquet")
    call_table = pd.read_parquet(root / "llm_call_audit.parquet")
    calls = {row.cache_key: json.loads(row.record_json) for row in call_table.itertuples()}
    corrected_calls = []
    for key, call in calls.items():
        assert canonical_hash(call["request"]) == key == call["cache_key"]
        assert hashlib.sha256(call["raw_content"].encode()).hexdigest() == call["response_hash"]
        assert call["request"]["transport"] == protocol["transport"]
        assert call["request"]["model_digest"] == protocol["model_digest"]
        assert call["request"]["data_signature"] == data["data_signature"]
        if correction := call.get("validation_correction"):
            original_text = correction["original_record_json"]
            assert hashlib.sha256(original_text.encode()).hexdigest() == correction["original_record_sha256"]
            original = json.loads(original_text)
            assert original["status"] == "schema_failure" and original["validated_content"] is None
            assert correction["weights_changed"] is False
            for field in ("request", "raw_content", "response_hash", "errors", "metadata", "attempts"):
                assert original[field] == call[field]
            assert json.loads(call["raw_content"]) == call["validated_content"]
            values = list(call["validated_content"]["weights"].values())
            assert not np.isclose(sum(values), 1.0, atol=1e-6, rtol=0)
            assert abs(sum(Decimal(str(value)) for value in values) - 1) <= Decimal("0.000001")
            corrected_calls.append(key)
    assert len(summary) == 15 and summary.groupby("stage").arm.nunique().eq(5).all()
    assert not decisions.duplicated(["stage", "arm", "decision_time"]).any()
    panels, experts = {}, {}
    for stage in stages:
        arrays = []
        reference = None
        for name in names:
            paths = data["predictions"][stage][f"{name}:w65"]
            frame = pd.concat([pd.read_parquet(root / path) for path in paths], ignore_index=True).sort_values("timestamp")
            assert frame.width_bps.eq(65).all() and pd.to_datetime(frame.train_end, utc=True).lt(frame.timestamp).all()
            if reference is not None:
                pd.testing.assert_frame_equal(frame[["timestamp", "y_true"]].reset_index(drop=True), reference)
            reference = frame[["timestamp", "y_true"]].reset_index(drop=True)
            values = frame[PROBS].to_numpy(float)
            assert np.isfinite(values).all() and (values >= 0).all() and np.allclose(values.sum(1), 1, atol=1e-5)
            arrays.append(values)
        panels[stage] = (reference, np.stack(arrays, axis=1))
    for name in names:
        experts[name] = pd.concat([pd.read_parquet(root / "expert_ledgers" / f"{stage}_{name}.parquet") for stage in stages])
        ledger = experts[name]
        assert ledger.available_time.eq(ledger.intrabar_exit_time + pd.Timedelta(minutes=1)).all()
    times = pd.DatetimeIndex(pd.concat([x[0].timestamp for x in panels.values()], ignore_index=True)) + BAR
    assert times.is_monotonic_increasing and times.is_unique
    cube = np.concatenate([x[1] for x in panels.values()])
    market = pd.read_parquet(root / "market_state_bars.parquet").set_index("timestamp")
    observed_start = pd.Timestamp(stages["development"][0])
    counts = {}
    for decision, group in decisions.groupby("decision_time", sort=True):
        cards = _cards(experts, names, decision, observed_start)
        expected_state = _expected_state(market, times, cube, decision)
        for row in group.itertuples():
            weights = np.array([getattr(row, "w_" + name) for name in names])
            assert np.isfinite(weights).all() and (weights >= 0).all() and (weights <= 1).all()
            assert abs(sum(Decimal(str(value)) for value in weights) - 1) <= Decimal("0.000001")
            if row.arm == "Hedge":
                score = sum((np.tanh(np.array(card["values"])[:, 0] / 100) for card in cards), start=np.zeros(9))
                expected = np.exp(protocol["hedge_eta"] * score - np.max(protocol["hedge_eta"] * score))
                np.testing.assert_allclose(weights, expected / expected.sum(), atol=1e-12, rtol=0)
                continue
            call = calls[row.cache_key]
            messages = call["request"]["messages"]
            assert len(messages) == 2 and [m["role"] for m in messages] == ["system", "user"]
            payload = json.loads(messages[1]["content"])
            assert set(payload) == {"model_order", "current_state", "memory"}
            assert payload["model_order"] == names and payload["current_state"] == expected_state
            expected_cards = [] if row.arm == "NoMemory" else _shuffled(cards, decision, 9) if row.arm == "ShuffledMemory" else cards
            assert payload["memory"] == expected_cards and row.memory_cards == len(expected_cards)
            assert bool(row.fallback) == (call["validated_content"] is None)
            if not row.fallback:
                assert json.loads(call["raw_content"]) == call["validated_content"]
                expected = [call["validated_content"]["weights"][name] for name in names]
                np.testing.assert_allclose(weights, expected, atol=1e-12, rtol=0)
    for stage, (start, end) in stages.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        grid = pd.date_range(start, end, freq="15min", inclusive="left")
        reference, cube = panels[stage]
        paths = pd.read_parquet(root / "execution_paths" / f"{stage}_{execution['path_signature'][:16]}.parquet")
        for i, name in enumerate(names):
            probability = cube[:, i, :]
            expert_signal = pd.Series(np.where(probability.max(1) >= .55, probability.argmax(1) - 1, 0), index=reference.timestamp)
            rebuilt_expert, _ = _replay(grid, expert_signal, pd.Series(150, index=grid), paths)
            saved_expert = experts[name].loc[experts[name].stage.eq(stage)].drop(columns="stage").reset_index(drop=True)
            pd.testing.assert_frame_equal(rebuilt_expert[saved_expert.columns], saved_expert, check_dtype=False, atol=1e-12, rtol=0)
        union = pd.read_parquet(root / data["union_signals"][stage]).set_index("timestamp").union_signal
        counts[stage] = int(decisions.loc[decisions.stage.eq(stage) & decisions.arm.eq("RealMemory")].shape[0])
        for arm in summary.loc[summary.stage.eq(stage), "arm"]:
            if arm == "LSTM_fixed":
                baseline = pd.concat([pd.read_parquet(root / p) for p in data["predictions"][stage]["lstm:w55"]])
                p = baseline[PROBS].to_numpy(float)
                signal = pd.Series(np.where(p.max(1) >= .75, p.argmax(1) - 1, 0), index=baseline.timestamp)
                tp = pd.Series(200, index=grid)
            else:
                rows = decisions.loc[decisions.stage.eq(stage) & decisions.arm.eq(arm)]
                allocation = np.full((len(reference), 9), 1 / 9)
                fallback = np.zeros(len(reference), dtype=bool)
                covered = np.zeros(len(reference), dtype=bool)
                for row in rows.itertuples():
                    included = reference.timestamp.add(BAR).ge(row.decision_time) & reference.timestamp.add(BAR).lt(row.end_time)
                    assert not covered[included].any()
                    allocation[included] = [getattr(row, "w_" + name) for name in names]
                    fallback[included] = row.fallback
                    covered[included] = True
                assert covered[reference.timestamp < end - BAR].all()
                mixed = (cube * allocation[:, :, None]).sum(axis=1)
                signal = pd.Series(np.where(mixed.max(1) >= .55, mixed.argmax(1) - 1, 0), index=reference.timestamp)
                signal.iloc[np.flatnonzero(fallback)] = union.reindex(signal.index).iloc[np.flatnonzero(fallback)].to_numpy()
                tp = pd.Series(np.where(fallback, 200, 150), index=signal.index).reindex(grid, fill_value=150)
            ledger, returns = _replay(grid, signal, tp, paths)
            saved = pd.read_parquet(root / "ledgers" / f"{stage}_{arm}.parquet")
            pd.testing.assert_frame_equal(ledger[saved.columns], saved, check_dtype=False, atol=1e-12, rtol=0)
            saved_returns = pd.read_parquet(root / "returns" / f"{stage}_{arm}.parquet").set_index("timestamp").bracket_return
            np.testing.assert_allclose(returns, saved_returns, atol=1e-12, rtol=0)
            metrics = summary.loc[summary.stage.eq(stage) & summary.arm.eq(arm)].iloc[0]
            assert len(ledger) == metrics.trades
            for key, value in _metrics(returns).items():
                assert np.isclose(value, metrics[key], atol=1e-10, rtol=0), (stage, arm, key)
            for row in weekly.loc[weekly.stage.eq(stage) & weekly.arm.eq(arm)].itertuples():
                actual = returns.loc[(returns.index >= row.decision_time) & (returns.index < row.end_time)].sum()
                assert np.isclose(actual, row.net_return, atol=1e-12, rtol=0)
    primary = weekly.loc[weekly.stage.eq("forward")].pivot(index="decision_time", columns="arm", values="net_return")
    intervals = pd.read_parquet(root / "paired_comparisons.parquet")
    rng = np.random.default_rng(42)
    n = len(primary)
    starts = rng.integers(n, size=(10000, int(np.ceil(n / 4))))
    bootstrap = ((starts[:, :, None] + np.arange(4)) % n).reshape(10000, -1)[:, :n]
    for row in intervals.itertuples():
        comparator = row.contrast.removeprefix("RealMemory - ")
        difference = (primary.RealMemory - primary[comparator]).to_numpy()
        sampled = difference[bootstrap].sum(1)
        quantiles = np.quantile(sampled, [.025, .975, .05 / 6, 1 - .05 / 6])
        np.testing.assert_allclose(quantiles, [row.ci95_low, row.ci95_high, row.familywise95_low, row.familywise95_high], atol=1e-12, rtol=0)
        assert np.isclose(difference.sum(), row.net_difference, atol=1e-12, rtol=0)
    assert sum(counts.values()) * 3 == state["logical_llm_decisions"] == len(decisions.loc[decisions.arm.ne("Hedge")])
    assert decisions.fallback.sum() == state["fallback_weeks"]
    assert state["fallback_weeks"] <= .05 * state["logical_llm_decisions"]
    if corrected_calls:
        correction_audit = json.loads((root / "validation_boundary_audit.json").read_text(encoding="utf-8"))
        assert correction_audit["corrected_unique_calls"] == len(corrected_calls)
        assert sorted(row["cache_key"] for row in correction_audit["corrections"]) == sorted(corrected_calls)
        assert correction_audit["corrected_logical_decisions"] == decisions.cache_key.isin(corrected_calls).sum()
        assert correction_audit["responses_and_weights_changed"] is False
    replay_path = root / "offline_replay_audit.json"
    if replay_path.is_file():
        replay_audit = json.loads(replay_path.read_text(encoding="utf-8"))
        assert replay_audit["status"] == "complete" and replay_audit["fresh_llm_calls"] == 0
    return {"verified": True, "input_files": len(manifest["files"]), "models": len(names),
            "stage_weeks": counts, "logical_llm_decisions": state["logical_llm_decisions"],
            "unique_llm_calls": len(calls), "fallback_weeks": state["fallback_weeks"],
            "memory_and_current_state_audits": state["logical_llm_decisions"],
            "replayed_stage_arms": len(summary), "paired_intervals_reproduced": len(intervals),
            "decimal_boundary_revalidations": len(corrected_calls),
            "old_forward_lstm_exact": True, "q2_rows_read": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "experiments/cache/reflection_ensemble_v5")
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--report", type=Path, help="Save the verified replay report for the rebuild graph.")
    args = parser.parse_args()
    if args.seal:
        seal_reader(args.root)
    report = json.dumps(verify_reader(args.root), indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
