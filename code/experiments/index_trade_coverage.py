"""H1-only coverage-first follow-up for the two frozen index replications.

The experiment screens the already materialised H1 grid, freezes one policy
per feature arm, and only then loads the reused forward span.  It never reads
or writes the sealed Q2-2026 lockbox.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from experiments.index_replication import (
    ARMS,
    CACHE_BASE as SOURCE_CACHE_BASE,
    IndexReplicationConfig,
    IndexReplicationRunner,
    _frame_hash,
    simulate_one_bar,
)
from experiments.index_replication_protocol import (
    CUTOFF,
    FORWARD_END,
    FORWARD_START,
    MODEL_NAMES,
    TAUS,
    WIDTHS,
    daily_economics,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
CACHE_BASE = CODE_ROOT / "experiments" / "cache" / "index_trade_coverage"
COVERAGE_PROTOCOL_VERSION = "index-trade-coverage-v2"
COVERAGE_MIN_TRADES = 50
COVERAGE_MIN_SIDE_TRADES = 15
COVERAGE_MIN_POSITIVE_MONTHS = 3
COVERAGE_EVIDENCE_ROLE = "secondary_reused_forward"

_REQUIRED_H1_COLUMNS = {
    "arm",
    "model_name",
    "width_bps",
    "tau",
    "trades",
    "n_long",
    "n_short",
    "positive_months",
    "net_return",
    "daily_sharpe",
    "daily_sortino",
}


def _canonical(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        stamp = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return stamp.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list, np.ndarray, pd.Index)):
        return [_canonical(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _payload_hash(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_canonical(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, engine="pyarrow", index=index)
    os.replace(temporary, path)


def _require_columns(frame: pd.DataFrame) -> None:
    missing = _REQUIRED_H1_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"H1 coverage grid misses columns: {sorted(missing)}")


def coverage_eligibility(frame: pd.DataFrame) -> pd.Series:
    """Return the exact seven-condition H1 coverage gate."""
    _require_columns(frame)
    numeric_columns = [
        "trades",
        "n_long",
        "n_short",
        "positive_months",
        "net_return",
        "daily_sharpe",
        "daily_sortino",
    ]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    finite = pd.Series(
        np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1), index=frame.index
    )
    return (
        finite
        & numeric["trades"].ge(COVERAGE_MIN_TRADES)
        & numeric["n_long"].ge(COVERAGE_MIN_SIDE_TRADES)
        & numeric["n_short"].ge(COVERAGE_MIN_SIDE_TRADES)
        & numeric["positive_months"].ge(COVERAGE_MIN_POSITIVE_MONTHS)
        & numeric["net_return"].gt(0.0)
        & numeric["daily_sharpe"].gt(0.0)
        & numeric["daily_sortino"].gt(0.0)
    ).astype(bool)


def select_coverage_policy(grid: pd.DataFrame) -> pd.Series:
    """Select the highest-volume eligible H1 row with deterministic tie-breaks."""
    if grid.empty:
        raise ValueError("no coverage-eligible H1 policy")
    work = grid.copy()
    work["coverage_eligible"] = coverage_eligibility(work)
    eligible = work.loc[work["coverage_eligible"]].copy()
    if eligible.empty:
        raise ValueError("no coverage-eligible H1 policy")
    model_order = {name: rank for rank, name in enumerate(MODEL_NAMES)}
    eligible["__model_order"] = eligible["model_name"].map(model_order)
    if eligible["__model_order"].isna().any():
        raise ValueError("coverage grid contains an unregistered model")
    ranked = eligible.sort_values(
        [
            "trades",
            "daily_sortino",
            "net_return",
            "daily_sharpe",
            "__model_order",
            "width_bps",
            "tau",
        ],
        ascending=[False, False, False, False, True, True, True],
        kind="mergesort",
    )
    winner = ranked.iloc[0].drop(labels="__model_order").copy()
    winner["selection_rule"] = (
        "coverage_eligible,trades,daily_sortino,net_return,daily_sharpe,"
        "model_order,width,tau"
    )
    return winner


def select_coverage_policies(grid: pd.DataFrame) -> pd.DataFrame:
    """Freeze one coverage-first H1 policy for every registered feature arm."""
    _require_columns(grid)
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_grid = grid.loc[grid["arm"].eq(arm)].copy()
        winner = select_coverage_policy(arm_grid).to_dict()
        eligible = arm_grid.loc[coverage_eligibility(arm_grid)]
        winner.update(
            {
                "models_screened": int(arm_grid["model_name"].nunique()),
                "eligible_models": int(eligible["model_name"].nunique()),
                "eligible_policies": int(len(eligible)),
            }
        )
        rows.append(winner)
    selected = pd.DataFrame(rows)
    if selected["arm"].tolist() != list(ARMS):
        raise AssertionError("coverage policy order changed")
    return selected


def validate_h1_grid(grid: pd.DataFrame) -> None:
    """Require the exact four-arm x nine-model x three-width x tau grid."""
    _require_columns(grid)
    keys = ["arm", "model_name", "width_bps", "tau"]
    if not pd.api.types.is_numeric_dtype(grid["width_bps"]) or not pd.api.types.is_numeric_dtype(
        grid["tau"]
    ):
        raise ValueError("H1 coverage source must be the exact registered grid")
    widths = pd.to_numeric(grid["width_bps"], errors="coerce")
    taus = pd.to_numeric(grid["tau"], errors="coerce")
    if (
        not np.isfinite(widths.to_numpy(dtype=float)).all()
        or not np.isfinite(taus.to_numpy(dtype=float)).all()
        or not widths.isin(WIDTHS).all()
        or not taus.isin(TAUS).all()
    ):
        raise ValueError("H1 coverage source must be the exact registered grid")
    actual = {
        (str(arm), str(model), float(width), float(tau))
        for arm, model, width, tau in grid[keys].itertuples(index=False, name=None)
    }
    expected = {
        (arm, model, float(width), float(tau))
        for arm, model, width, tau in product(ARMS, MODEL_NAMES, WIDTHS, TAUS)
    }
    if len(grid) != len(expected) or grid.duplicated(keys).any() or actual != expected:
        raise ValueError("H1 coverage source must be the exact registered grid")


@dataclass(frozen=True)
class IndexTradeCoverageConfig:
    stream: str
    data_dir: Path
    source_root: Path
    output_root: Path

    @classmethod
    def for_stream(
        cls,
        stream: str,
        *,
        data_dir: str | Path = CODE_ROOT / "data",
        source_base: str | Path = SOURCE_CACHE_BASE,
        output_base: str | Path = CACHE_BASE,
    ) -> "IndexTradeCoverageConfig":
        if stream not in {"usa500", "usatech"}:
            raise ValueError("stream must be usa500 or usatech")
        return cls(
            stream=stream,
            data_dir=Path(data_dir),
            source_root=Path(source_base) / stream,
            output_root=Path(output_base) / stream,
        )


class IndexTradeCoverageRunner:
    """Freeze H1 choices first, then run exactly those choices on reused forward."""

    def __init__(
        self,
        config: IndexTradeCoverageConfig,
        *,
        source_runner_factory: Callable[..., IndexReplicationRunner] = IndexReplicationRunner,
    ) -> None:
        self.config = config
        self.source_runner_factory = source_runner_factory
        self.output_root = Path(config.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _state(self, stage: str, **details: Any) -> None:
        _atomic_json(
            {"stage": stage, "updated_at_utc": pd.Timestamp.now("UTC"), **details},
            self.output_root / "run_state.json",
        )

    def _source_contract(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        grid_path = self.config.source_root / "h1_policy_grid.parquet"
        protocol_path = self.config.source_root / "protocol_manifest.json"
        if not grid_path.exists() or not protocol_path.exists():
            raise FileNotFoundError("completed source H1 grid and protocol are required")
        grid = pd.read_parquet(grid_path)
        validate_h1_grid(grid)
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        expected_calibration = [
            pd.Timestamp("2025-01-01T00:00:00Z").isoformat(),
            FORWARD_START.isoformat(),
        ]
        if protocol.get("calibration") != expected_calibration:
            raise ValueError("source H1 calibration boundary changed")
        if protocol.get("forward") != [FORWARD_START.isoformat(), FORWARD_END.isoformat()]:
            raise ValueError("source forward boundary changed")
        return grid, protocol

    def _freeze_selection(
        self,
        grid: pd.DataFrame,
        source_protocol: Mapping[str, Any],
    ) -> tuple[pd.DataFrame, str, str]:
        candidates = grid.copy()
        candidates["coverage_eligible"] = coverage_eligibility(candidates)
        selected = select_coverage_policies(candidates)
        if not selected["models_screened"].eq(len(MODEL_NAMES)).all():
            raise AssertionError("all nine models were not screened")
        grid_hash = _frame_hash(grid)
        selected_hash = _frame_hash(selected)
        implementation_hash = hashlib.sha256(
            (
                inspect.getsource(coverage_eligibility)
                + inspect.getsource(select_coverage_policy)
                + inspect.getsource(select_coverage_policies)
                + inspect.getsource(validate_h1_grid)
                + inspect.getsource(IndexTradeCoverageRunner)
            ).encode("utf-8")
        ).hexdigest()
        body = {
            "protocol_version": COVERAGE_PROTOCOL_VERSION,
            "stream": self.config.stream,
            "evidence_role": COVERAGE_EVIDENCE_ROLE,
            "source_protocol_hash": source_protocol.get("protocol_hash"),
            "source_h1_grid_sha256": grid_hash,
            "selected_policy_sha256": selected_hash,
            "arms": list(ARMS),
            "models": list(MODEL_NAMES),
            "widths_bps": list(WIDTHS),
            "taus": list(TAUS),
            "h1_gate": {
                "min_trades": COVERAGE_MIN_TRADES,
                "min_long": COVERAGE_MIN_SIDE_TRADES,
                "min_short": COVERAGE_MIN_SIDE_TRADES,
                "min_positive_months_of_6": COVERAGE_MIN_POSITIVE_MONTHS,
                "net_return_strictly_positive": True,
                "daily_sharpe_strictly_positive": True,
                "daily_sortino_strictly_positive": True,
            },
            "selection_rule": (
                "max_trades_then_sortino_net_sharpe_model_order_width_tau"
            ),
            "forward": [FORWARD_START, FORWARD_END],
            "q2_start": CUTOFF,
            "q2_loaded": False,
            "implementation_sha256": implementation_hash,
        }
        protocol_hash = _payload_hash(body)
        protocol_payload = {**body, "protocol_hash": protocol_hash}
        protocol_path = self.output_root / "protocol.json"
        if protocol_path.exists():
            existing = json.loads(protocol_path.read_text(encoding="utf-8"))
            if existing != _canonical(protocol_payload):
                raise ValueError("existing coverage protocol does not match current selection")
        _atomic_json(protocol_payload, protocol_path)
        _atomic_parquet(candidates, self.output_root / "h1_candidates.parquet")
        _atomic_parquet(selected, self.output_root / "h1_selected_policies.parquet")
        self._state(
            "selection_frozen_before_forward",
            protocol_hash=protocol_hash,
            selected_policy_sha256=selected_hash,
        )
        return selected, protocol_hash, selected_hash

    @staticmethod
    def _policy_tuple(policy: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "arm": str(policy["arm"]),
            "model_name": str(policy["model_name"]),
            "width_bps": int(policy["width_bps"]),
            "tau": float(policy["tau"]),
        }

    def _completed_arm(
        self,
        policy: Mapping[str, Any],
        protocol_hash: str,
        selected_hash: str,
    ) -> tuple[dict[str, Any], pd.Timestamp] | None:
        expected_policy = self._policy_tuple(policy)
        arm = expected_policy["arm"]
        path = self.output_root / "forward" / f"{arm}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_hash") != protocol_hash
            or payload.get("selected_policy_sha256") != selected_hash
        ):
            raise ValueError(f"stale coverage forward artifact for {arm}")
        if payload.get("policy") != expected_policy:
            raise ValueError(f"coverage resume policy tuple changed for {arm}")
        prediction_max = pd.Timestamp(payload.get("prediction_max_timestamp"))
        if prediction_max.tzinfo is None:
            prediction_max = prediction_max.tz_localize("UTC")
        else:
            prediction_max = prediction_max.tz_convert("UTC")
        if not (FORWARD_START <= prediction_max < FORWARD_END):
            raise ValueError(f"coverage resume prediction timestamp is invalid for {arm}")

        expected_paths = {
            "ledger": f"forward_ledgers/{arm}.parquet",
            "per_bar": f"forward_ledgers/{arm}_per_bar.parquet",
        }
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != set(expected_paths):
            raise ValueError(f"coverage resume artifact set changed for {arm}")
        resolved: dict[str, Path] = {}
        for key, relative in expected_paths.items():
            record = artifacts.get(key)
            if not isinstance(record, dict) or record.get("path") != relative:
                raise ValueError(f"coverage resume artifact path changed for {arm}")
            artifact_path = self.output_root / relative
            if not artifact_path.exists() or record.get("sha256") != _file_sha256(
                artifact_path
            ):
                raise ValueError(f"coverage resume artifact hash changed for {arm}")
            resolved[key] = artifact_path

        ledger = pd.read_parquet(resolved["ledger"])
        per_bar_frame = pd.read_parquet(resolved["per_bar"])
        if not {"timestamp", "net_return"}.issubset(per_bar_frame.columns):
            raise ValueError(f"coverage resume per-bar schema changed for {arm}")
        per_bar_frame["timestamp"] = pd.to_datetime(per_bar_frame["timestamp"], utc=True)
        if len(per_bar_frame) and not per_bar_frame["timestamp"].between(
            FORWARD_START, FORWARD_END, inclusive="left"
        ).all():
            raise ValueError(f"coverage resume per-bar timestamp changed for {arm}")
        for column in ("entry_time", "exit_time"):
            if column in ledger and len(ledger):
                stamps = pd.to_datetime(ledger[column], utc=True)
                upper_inclusive = column == "exit_time"
                valid = stamps.ge(FORWARD_START) & (
                    stamps.le(FORWARD_END) if upper_inclusive else stamps.lt(FORWARD_END)
                )
                if not valid.all():
                    raise ValueError(f"coverage resume ledger timestamp changed for {arm}")
        per_bar = per_bar_frame.set_index("timestamp")["net_return"].astype(float)
        recalculated = daily_economics(
            ledger, per_bar, start=FORWARD_START, end=FORWARD_END
        )
        summary = dict(payload.get("summary", {}))
        if (
            summary.get("stream") != self.config.stream
            or summary.get("evidence_role") != COVERAGE_EVIDENCE_ROLE
            or any(summary.get(key) != value for key, value in expected_policy.items())
        ):
            raise ValueError(f"coverage resume summary policy tuple changed for {arm}")
        for key, expected in recalculated.items():
            actual = summary.get(key)
            if actual is None or not np.isclose(float(actual), float(expected), rtol=1e-12, atol=1e-12):
                raise ValueError(f"coverage resume summary economics changed for {arm}: {key}")
        return summary, prediction_max

    def run(self) -> dict[str, Any]:
        grid, source_protocol = self._source_contract()
        selected, protocol_hash, selected_hash = self._freeze_selection(
            grid, source_protocol
        )

        source_config = IndexReplicationConfig.for_stream(
            self.config.stream,
            data_dir=self.config.data_dir,
            output_base=self.config.source_root.parent,
        )
        source_runner: IndexReplicationRunner | None = None
        forward_rows: list[dict[str, Any]] = []
        prediction_maxima: list[pd.Timestamp] = []
        reused_arms = 0
        for number, policy in enumerate(selected.to_dict("records"), start=1):
            arm = str(policy["arm"])
            completed = self._completed_arm(policy, protocol_hash, selected_hash)
            if completed is not None:
                completed_summary, completed_maximum = completed
                forward_rows.append(completed_summary)
                prediction_maxima.append(completed_maximum)
                reused_arms += 1
                continue
            if source_runner is None:
                source_runner = self.source_runner_factory(source_config)
            model_name = str(policy["model_name"])
            width = int(policy["width_bps"])
            tau = float(policy["tau"])
            prediction = source_runner._forward_prediction(arm, model_name, width)
            maximum = pd.to_datetime(prediction["timestamp"], utc=True).max()
            if maximum >= CUTOFF:
                raise AssertionError("coverage prediction crossed the Q2 boundary")
            prediction_maxima.append(maximum)
            ledger, per_bar = simulate_one_bar(
                source_runner.bars,
                prediction,
                start=FORWARD_START,
                end=FORWARD_END,
                tau=tau,
                cost_bps=source_config.cost_bps,
            )
            stress_ledger, stress_per_bar = simulate_one_bar(
                source_runner.bars,
                prediction,
                start=FORWARD_START,
                end=FORWARD_END,
                tau=tau,
                cost_bps=2.0 * source_config.cost_bps,
            )
            economics = daily_economics(
                ledger, per_bar, start=FORWARD_START, end=FORWARD_END
            )
            stress = daily_economics(
                stress_ledger,
                stress_per_bar,
                start=FORWARD_START,
                end=FORWARD_END,
            )
            summary = {
                "stream": self.config.stream,
                "arm": arm,
                "model_name": model_name,
                "width_bps": width,
                "tau": tau,
                "evidence_role": COVERAGE_EVIDENCE_ROLE,
                **economics,
                "stress_2x_net_return": stress["net_return"],
                "stress_2x_daily_sharpe": stress["daily_sharpe"],
                "stress_2x_daily_sortino": stress["daily_sortino"],
                "fit_id": str(prediction["fit_id"].iloc[0]),
            }
            ledger_path = self.output_root / "forward_ledgers" / f"{arm}.parquet"
            per_bar_path = self.output_root / "forward_ledgers" / f"{arm}_per_bar.parquet"
            _atomic_parquet(ledger, ledger_path)
            _atomic_parquet(
                per_bar.rename("net_return").rename_axis("timestamp").reset_index(),
                per_bar_path,
            )
            policy_tuple = self._policy_tuple(policy)
            _atomic_json(
                {
                    "protocol_hash": protocol_hash,
                    "selected_policy_sha256": selected_hash,
                    "policy": policy_tuple,
                    "prediction_max_timestamp": maximum,
                    "artifacts": {
                        "ledger": {
                            "path": ledger_path.relative_to(self.output_root).as_posix(),
                            "sha256": _file_sha256(ledger_path),
                        },
                        "per_bar": {
                            "path": per_bar_path.relative_to(self.output_root).as_posix(),
                            "sha256": _file_sha256(per_bar_path),
                        },
                    },
                    "summary": summary,
                },
                self.output_root / "forward" / f"{arm}.json",
            )
            forward_rows.append(summary)
            self._state(
                "forward",
                completed=number,
                total=len(selected),
                arm=arm,
                model=model_name,
            )

        forward = pd.DataFrame(forward_rows)
        forward["__arm_order"] = forward["arm"].map(
            {arm: rank for rank, arm in enumerate(ARMS)}
        )
        forward = forward.sort_values("__arm_order").drop(columns="__arm_order")
        _atomic_parquet(forward, self.output_root / "forward_summary.parquet")
        result = {
            "stream": self.config.stream,
            "protocol_version": COVERAGE_PROTOCOL_VERSION,
            "protocol_hash": protocol_hash,
            "evidence_role": COVERAGE_EVIDENCE_ROLE,
            "models_screened_per_arm": len(MODEL_NAMES),
            "selected_policy_rows": len(selected),
            "forward_rows": len(forward),
            "forward_replay_count": 1,
            "resumed_forward_arms": reused_arms,
            "max_prediction_timestamp": max(prediction_maxima) if prediction_maxima else None,
            "q2_loaded": False,
        }
        _atomic_json(result, self.output_root / "result.json")
        checkpoint_paths = {
            self.output_root / "forward" / f"{arm}.json" for arm in ARMS
        }
        ledger_paths = {
            self.output_root / "forward_ledgers" / f"{arm}{suffix}.parquet"
            for arm in ARMS
            for suffix in ("", "_per_bar")
        }
        if set((self.output_root / "forward").glob("*.json")) != checkpoint_paths:
            raise ValueError("coverage manifest requires exactly four arm checkpoints")
        if set((self.output_root / "forward_ledgers").glob("*.parquet")) != ledger_paths:
            raise ValueError("coverage manifest requires exactly eight arm ledger artifacts")
        artifact_paths = [
            self.output_root / "protocol.json",
            self.output_root / "h1_candidates.parquet",
            self.output_root / "h1_selected_policies.parquet",
            self.output_root / "forward_summary.parquet",
            self.output_root / "result.json",
            *sorted(checkpoint_paths),
            *sorted(ledger_paths),
        ]
        manifest = {
            "protocol_hash": protocol_hash,
            "q2_loaded": False,
            "artifacts": {
                path.relative_to(self.output_root).as_posix(): _file_sha256(path)
                for path in artifact_paths
            },
        }
        _atomic_json(manifest, self.output_root / "manifest.json")
        self._state("complete", **result)
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", choices=("usa500", "usatech"), required=True)
    args = parser.parse_args(argv)
    result = IndexTradeCoverageRunner(
        IndexTradeCoverageConfig.for_stream(args.stream)
    ).run()
    print(json.dumps(_canonical(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CACHE_BASE",
    "COVERAGE_EVIDENCE_ROLE",
    "COVERAGE_MIN_POSITIVE_MONTHS",
    "COVERAGE_PROTOCOL_VERSION",
    "IndexTradeCoverageConfig",
    "IndexTradeCoverageRunner",
    "coverage_eligibility",
    "select_coverage_policies",
    "select_coverage_policy",
    "validate_h1_grid",
]
