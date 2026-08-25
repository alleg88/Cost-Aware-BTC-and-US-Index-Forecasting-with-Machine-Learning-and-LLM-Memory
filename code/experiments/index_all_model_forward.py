"""Secondary all-model forward diagnostics for the two index streams."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from dataclasses import dataclass
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
CACHE_BASE = CODE_ROOT / "experiments" / "cache" / "index_all_model_forward"
PROTOCOL_VERSION = "index-all-model-forward-v1"
EVIDENCE_ROLE = "secondary_reused_forward_diagnostic"

_SELECTED_POLICY_COLUMNS = {
    "arm",
    "model_name",
    "width_bps",
    "tau",
    "eligible",
    "h1_execution_status",
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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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


def _stress_economics(
    ledger: pd.DataFrame,
    per_bar: pd.Series,
) -> dict[str, Any]:
    stressed_ledger = ledger.copy()
    stressed_per_bar = pd.Series(0.0, index=per_bar.index, name="net_return")
    if len(stressed_ledger):
        stressed_ledger["cost_return"] = (
            pd.to_numeric(stressed_ledger["cost_return"], errors="raise") * 2.0
        )
        stressed_ledger["net_return"] = (
            pd.to_numeric(stressed_ledger["gross_return"], errors="raise")
            - stressed_ledger["cost_return"]
        )
        entry_times = pd.to_datetime(stressed_ledger["entry_time"], utc=True)
        stressed_by_bar = stressed_ledger.assign(__entry=entry_times).groupby(
            "__entry"
        )["net_return"].sum()
        stressed_per_bar.loc[
            stressed_per_bar.index.intersection(stressed_by_bar.index)
        ] = stressed_by_bar.reindex(
            stressed_per_bar.index.intersection(stressed_by_bar.index)
        ).to_numpy()
    return daily_economics(
        stressed_ledger,
        stressed_per_bar,
        start=FORWARD_START,
        end=FORWARD_END,
    )


def validate_selected_policies(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and stably order the frozen four-arm by nine-model grid."""
    missing = _SELECTED_POLICY_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"selected-policy source misses columns: {sorted(missing)}")

    expected = set(product(ARMS, MODEL_NAMES))
    identities = frame[["arm", "model_name"]]
    actual = set(map(tuple, identities.to_numpy()))
    widths = pd.to_numeric(frame["width_bps"], errors="coerce")
    taus = pd.to_numeric(frame["tau"], errors="coerce")
    exact_grid = (
        len(frame) == len(expected)
        and not identities.duplicated().any()
        and actual == expected
        and np.isfinite(widths).all()
        and np.isfinite(taus).all()
        and set(widths.astype(float)).issubset({float(value) for value in WIDTHS})
        and set(taus.astype(float)).issubset({float(value) for value in TAUS})
        and pd.api.types.is_bool_dtype(frame["eligible"])
    )
    if not exact_grid:
        raise ValueError("selected-policy source must be the exact 36-policy grid")

    expected_status = frame["eligible"].map(
        {
            True: "eligible",
            False: "diagnostic_only_no_eligible_policy",
        }
    )
    if not frame["h1_execution_status"].eq(expected_status).all():
        raise ValueError("selected-policy H1 execution status changed")

    ordered = frame.copy()
    ordered["__arm_order"] = ordered["arm"].map(
        {arm: rank for rank, arm in enumerate(ARMS)}
    )
    ordered["__model_order"] = ordered["model_name"].map(
        {model: rank for rank, model in enumerate(MODEL_NAMES)}
    )
    return (
        ordered.sort_values(["__arm_order", "__model_order"], kind="mergesort")
        .drop(columns=["__arm_order", "__model_order"])
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class IndexAllModelForwardConfig:
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
    ) -> "IndexAllModelForwardConfig":
        if stream not in {"usa500", "usatech"}:
            raise ValueError("stream must be usa500 or usatech")
        return cls(
            stream=stream,
            data_dir=Path(data_dir),
            source_root=Path(source_base) / stream,
            output_root=Path(output_base) / stream,
        )


class IndexAllModelForwardRunner:
    """Evaluate all 36 frozen H1 choices on reused forward data."""

    def __init__(
        self,
        config: IndexAllModelForwardConfig,
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

    def _source_contract(self) -> tuple[pd.DataFrame, dict[str, Any], str, str]:
        selected_path = self.config.source_root / "h1_selected_policies.parquet"
        protocol_path = self.config.source_root / "protocol_manifest.json"
        if not selected_path.exists() or not protocol_path.exists():
            raise FileNotFoundError(
                "completed source selected policies and protocol are required"
            )
        selected = validate_selected_policies(pd.read_parquet(selected_path))
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol_hash = protocol.get("protocol_hash")
        protocol_body = {
            key: value for key, value in protocol.items() if key != "protocol_hash"
        }
        if not protocol_hash or protocol_hash != _payload_hash(protocol_body):
            raise ValueError("source protocol hash is missing or changed")
        if protocol.get("calibration") != [
            pd.Timestamp("2025-01-01T00:00:00Z").isoformat(),
            FORWARD_START.isoformat(),
        ]:
            raise ValueError("source H1 calibration boundary changed")
        if protocol.get("forward") != [
            FORWARD_START.isoformat(),
            FORWARD_END.isoformat(),
        ]:
            raise ValueError("source forward boundary changed")
        if protocol.get("q2_loaded") not in (None, False):
            raise PermissionError("source protocol reports Q2 as loaded")
        source_config = protocol.get("config")
        if isinstance(source_config, dict):
            if source_config.get("stream") != self.config.stream:
                raise ValueError("source protocol stream changed")
            if pd.Timestamp(source_config.get("end_exclusive")) != CUTOFF:
                raise PermissionError("source protocol crossed the Q2 boundary")
        return (
            selected,
            protocol,
            _file_sha256(selected_path),
            _file_sha256(protocol_path),
        )

    def _freeze(
        self,
        selected: pd.DataFrame,
        source_protocol: Mapping[str, Any],
        source_selected_file_hash: str,
        source_protocol_file_hash: str,
    ) -> tuple[str, str]:
        selected_hash = _frame_hash(selected)
        body = {
            "protocol_version": PROTOCOL_VERSION,
            "stream": self.config.stream,
            "evidence_role": EVIDENCE_ROLE,
            "source_protocol_hash": source_protocol["protocol_hash"],
            "source_protocol_file_sha256": source_protocol_file_hash,
            "source_selected_file_sha256": source_selected_file_hash,
            "selected_policy_sha256": selected_hash,
            "arms": list(ARMS),
            "models": list(MODEL_NAMES),
            "widths_bps": list(WIDTHS),
            "taus": list(TAUS),
            "selection": "reuse_each_model_arm_h1_width_tau_without_reselection",
            "forward": [FORWARD_START, FORWARD_END],
            "q2_start": CUTOFF,
            "q2_loaded": False,
            "implementation_sha256": _file_sha256(Path(inspect.getfile(type(self)))),
        }
        protocol_hash = _payload_hash(body)
        payload = {**body, "protocol_hash": protocol_hash}
        protocol_path = self.output_root / "protocol.json"
        if protocol_path.exists():
            existing = json.loads(protocol_path.read_text(encoding="utf-8"))
            if existing != _canonical(payload):
                raise ValueError("existing all-model forward protocol changed")
        _atomic_json(payload, protocol_path)
        _atomic_parquet(selected, self.output_root / "h1_selected_policies.parquet")
        self._state(
            "selection_frozen_before_forward",
            protocol_hash=protocol_hash,
            selected_policy_sha256=selected_hash,
        )
        return protocol_hash, selected_hash

    @staticmethod
    def _policy_tuple(policy: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "arm": str(policy["arm"]),
            "model_name": str(policy["model_name"]),
            "width_bps": int(policy["width_bps"]),
            "tau": float(policy["tau"]),
            "h1_eligible": bool(policy["eligible"]),
            "h1_execution_status": str(policy["h1_execution_status"]),
        }

    @staticmethod
    def _stem(policy: Mapping[str, Any]) -> str:
        return f"{policy['arm']}__{policy['model_name']}"

    def _completed_policy(
        self,
        policy: Mapping[str, Any],
        protocol_hash: str,
        selected_hash: str,
    ) -> tuple[dict[str, Any], pd.Timestamp] | None:
        expected_policy = self._policy_tuple(policy)
        stem = self._stem(expected_policy)
        checkpoint_path = self.output_root / "forward" / f"{stem}.json"
        if not checkpoint_path.exists():
            return None
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_hash") != protocol_hash
            or payload.get("selected_policy_sha256") != selected_hash
            or payload.get("policy") != expected_policy
        ):
            raise ValueError(f"stale all-model forward checkpoint for {stem}")
        prediction_max = pd.Timestamp(payload.get("prediction_max_timestamp"))
        prediction_max = (
            prediction_max.tz_localize("UTC")
            if prediction_max.tzinfo is None
            else prediction_max.tz_convert("UTC")
        )
        if not (FORWARD_START <= prediction_max < FORWARD_END):
            raise ValueError(f"resume prediction timestamp changed for {stem}")

        expected_paths = {
            "ledger": f"forward_ledgers/{stem}.parquet",
            "per_bar": f"forward_ledgers/{stem}_per_bar.parquet",
        }
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != set(expected_paths):
            raise ValueError(f"resume artifact set changed for {stem}")
        resolved: dict[str, Path] = {}
        for key, relative in expected_paths.items():
            record = artifacts.get(key)
            artifact_path = self.output_root / relative
            if (
                not isinstance(record, dict)
                or record.get("path") != relative
                or not artifact_path.exists()
                or record.get("sha256") != _file_sha256(artifact_path)
            ):
                raise ValueError(f"resume artifact hash changed for {stem}")
            resolved[key] = artifact_path

        ledger = pd.read_parquet(resolved["ledger"])
        per_bar_frame = pd.read_parquet(resolved["per_bar"])
        if not {"timestamp", "net_return"}.issubset(per_bar_frame.columns):
            raise ValueError(f"resume per-bar schema changed for {stem}")
        per_bar_frame["timestamp"] = pd.to_datetime(
            per_bar_frame["timestamp"], utc=True
        )
        if len(per_bar_frame) and not per_bar_frame["timestamp"].between(
            FORWARD_START, FORWARD_END, inclusive="left"
        ).all():
            raise ValueError(f"resume per-bar timestamp changed for {stem}")
        for column in ("entry_time", "exit_time"):
            if column in ledger and len(ledger):
                stamps = pd.to_datetime(ledger[column], utc=True)
                valid = stamps.ge(FORWARD_START) & (
                    stamps.le(FORWARD_END)
                    if column == "exit_time"
                    else stamps.lt(FORWARD_END)
                )
                if not valid.all():
                    raise ValueError(f"resume ledger timestamp changed for {stem}")
        per_bar = per_bar_frame.set_index("timestamp")["net_return"].astype(float)
        economics = daily_economics(
            ledger, per_bar, start=FORWARD_START, end=FORWARD_END
        )
        stress = _stress_economics(ledger, per_bar)
        expected_metrics = {
            **economics,
            "stress_2x_net_return": stress["net_return"],
            "stress_2x_daily_sharpe": stress["daily_sharpe"],
            "stress_2x_daily_sortino": stress["daily_sortino"],
        }
        summary = dict(payload.get("summary", {}))
        if (
            summary.get("stream") != self.config.stream
            or summary.get("evidence_role") != EVIDENCE_ROLE
            or any(summary.get(key) != value for key, value in expected_policy.items())
        ):
            raise ValueError(f"resume summary identity changed for {stem}")
        expected_status = "traded" if economics["trades"] else "no_trades"
        if summary.get("status") != expected_status:
            raise ValueError(f"resume trade status changed for {stem}")
        for key, expected in expected_metrics.items():
            actual = summary.get(key)
            if actual is None or not np.isclose(
                float(actual), float(expected), rtol=1e-12, atol=1e-12
            ):
                raise ValueError(f"resume economics changed for {stem}: {key}")
        return summary, prediction_max

    @staticmethod
    def _validate_prediction(prediction: pd.DataFrame, stem: str) -> pd.DataFrame:
        required = {"timestamp", "pred", "confidence", "fit_id"}
        missing = required.difference(prediction.columns)
        if missing or prediction.empty:
            raise ValueError(f"forward prediction is empty or misses columns for {stem}")
        current = prediction.copy()
        current["timestamp"] = pd.to_datetime(current["timestamp"], utc=True)
        if (
            current["timestamp"].duplicated().any()
            or not current["timestamp"].between(
                FORWARD_START, FORWARD_END, inclusive="left"
            ).all()
            or current["fit_id"].astype(str).nunique() != 1
        ):
            raise ValueError(f"forward prediction contract changed for {stem}")
        return current.sort_values("timestamp").reset_index(drop=True)

    def _artifact_paths(self) -> list[Path]:
        checkpoint_paths = [
            self.output_root / "forward" / f"{arm}__{model}.json"
            for arm, model in product(ARMS, MODEL_NAMES)
        ]
        ledger_paths = [
            self.output_root
            / "forward_ledgers"
            / f"{arm}__{model}{suffix}.parquet"
            for arm, model in product(ARMS, MODEL_NAMES)
            for suffix in ("", "_per_bar")
        ]
        return [
            self.output_root / "protocol.json",
            self.output_root / "h1_selected_policies.parquet",
            self.output_root / "forward_summary.parquet",
            self.output_root / "result.json",
            *checkpoint_paths,
            *ledger_paths,
        ]

    def _write_and_validate_manifest(self, protocol_hash: str) -> None:
        expected_checkpoints = {
            path for path in self._artifact_paths() if path.parent.name == "forward"
        }
        expected_ledgers = {
            path
            for path in self._artifact_paths()
            if path.parent.name == "forward_ledgers"
        }
        if set((self.output_root / "forward").glob("*.json")) != expected_checkpoints:
            raise ValueError("all-model manifest requires exactly 36 checkpoints")
        if set((self.output_root / "forward_ledgers").glob("*.parquet")) != expected_ledgers:
            raise ValueError("all-model manifest requires exactly 72 ledger artifacts")
        paths = self._artifact_paths()
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"all-model manifest misses artifacts: {missing}")
        manifest = {
            "protocol_hash": protocol_hash,
            "q2_loaded": False,
            "artifacts": {
                path.relative_to(self.output_root).as_posix(): _file_sha256(path)
                for path in paths
            },
        }
        _atomic_json(manifest, self.output_root / "manifest.json")
        saved = json.loads((self.output_root / "manifest.json").read_text(encoding="utf-8"))
        for relative, expected_hash in saved["artifacts"].items():
            if _file_sha256(self.output_root / relative) != expected_hash:
                raise ValueError(f"manifest artifact hash changed: {relative}")

    def run(self) -> dict[str, Any]:
        (
            selected,
            source_protocol,
            source_selected_file_hash,
            source_protocol_file_hash,
        ) = self._source_contract()
        protocol_hash, selected_hash = self._freeze(
            selected,
            source_protocol,
            source_selected_file_hash,
            source_protocol_file_hash,
        )
        source_config = IndexReplicationConfig.for_stream(
            self.config.stream,
            data_dir=self.config.data_dir,
            output_base=self.config.source_root.parent,
        )
        source_runner: IndexReplicationRunner | None = None
        rows: list[dict[str, Any]] = []
        maxima: list[pd.Timestamp] = []
        resumed = 0
        for number, policy in enumerate(selected.to_dict("records"), start=1):
            completed = self._completed_policy(policy, protocol_hash, selected_hash)
            if completed is not None:
                summary, maximum = completed
                rows.append(summary)
                maxima.append(maximum)
                resumed += 1
                continue
            if source_runner is None:
                source_runner = self.source_runner_factory(source_config)
            identity = self._policy_tuple(policy)
            stem = self._stem(identity)
            prediction = self._validate_prediction(
                source_runner._forward_prediction(
                    identity["arm"],
                    identity["model_name"],
                    identity["width_bps"],
                ),
                stem,
            )
            maximum = prediction["timestamp"].max()
            if maximum >= CUTOFF:
                raise AssertionError("all-model prediction crossed the Q2 boundary")
            maxima.append(maximum)
            ledger, per_bar = simulate_one_bar(
                source_runner.bars,
                prediction,
                start=FORWARD_START,
                end=FORWARD_END,
                tau=identity["tau"],
                cost_bps=source_config.cost_bps,
            )
            economics = daily_economics(
                ledger, per_bar, start=FORWARD_START, end=FORWARD_END
            )
            stress = _stress_economics(ledger, per_bar)
            summary = {
                "stream": self.config.stream,
                **identity,
                "evidence_role": EVIDENCE_ROLE,
                "status": "traded" if economics["trades"] else "no_trades",
                **economics,
                "stress_2x_net_return": stress["net_return"],
                "stress_2x_daily_sharpe": stress["daily_sharpe"],
                "stress_2x_daily_sortino": stress["daily_sortino"],
                "fit_id": str(prediction["fit_id"].iloc[0]),
            }
            numeric_values = [
                value
                for key, value in summary.items()
                if key
                in {
                    "width_bps",
                    "tau",
                    *economics.keys(),
                    "stress_2x_net_return",
                    "stress_2x_daily_sharpe",
                    "stress_2x_daily_sortino",
                }
            ]
            if not np.isfinite(np.asarray(numeric_values, dtype=float)).all():
                raise ValueError(f"non-finite all-model economics for {stem}")
            ledger_path = self.output_root / "forward_ledgers" / f"{stem}.parquet"
            per_bar_path = (
                self.output_root / "forward_ledgers" / f"{stem}_per_bar.parquet"
            )
            _atomic_parquet(ledger, ledger_path)
            _atomic_parquet(
                per_bar.rename("net_return").rename_axis("timestamp").reset_index(),
                per_bar_path,
            )
            _atomic_json(
                {
                    "protocol_hash": protocol_hash,
                    "selected_policy_sha256": selected_hash,
                    "policy": identity,
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
                self.output_root / "forward" / f"{stem}.json",
            )
            rows.append(summary)
            self._state(
                "forward",
                completed=number,
                total=len(selected),
                arm=identity["arm"],
                model=identity["model_name"],
            )
            print(
                f"[{self.config.stream}] all-model forward {number}/{len(selected)}: {stem}",
                flush=True,
            )

        summary_frame = pd.DataFrame(rows)
        summary_frame["__arm_order"] = summary_frame["arm"].map(
            {arm: rank for rank, arm in enumerate(ARMS)}
        )
        summary_frame["__model_order"] = summary_frame["model_name"].map(
            {model: rank for rank, model in enumerate(MODEL_NAMES)}
        )
        summary_frame = (
            summary_frame.sort_values(
                ["__arm_order", "__model_order"], kind="mergesort"
            )
            .drop(columns=["__arm_order", "__model_order"])
            .reset_index(drop=True)
        )
        if len(summary_frame) != len(ARMS) * len(MODEL_NAMES):
            raise AssertionError("all-model forward summary must contain 36 rows")
        _atomic_parquet(summary_frame, self.output_root / "forward_summary.parquet")
        result = {
            "stream": self.config.stream,
            "protocol_version": PROTOCOL_VERSION,
            "protocol_hash": protocol_hash,
            "evidence_role": EVIDENCE_ROLE,
            "models_per_arm": len(MODEL_NAMES),
            "selected_policy_rows": len(selected),
            "forward_rows": len(summary_frame),
            "resumed_forward_policies": resumed,
            "max_prediction_timestamp": max(maxima) if maxima else None,
            "q2_loaded": False,
        }
        _atomic_json(result, self.output_root / "result.json")
        if (
            _file_sha256(self.config.source_root / "h1_selected_policies.parquet")
            != source_selected_file_hash
            or _file_sha256(self.config.source_root / "protocol_manifest.json")
            != source_protocol_file_hash
        ):
            raise ValueError("source contract changed during all-model execution")
        self._write_and_validate_manifest(protocol_hash)
        self._state("complete", **result)
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", choices=("usa500", "usatech"), required=True)
    args = parser.parse_args(argv)
    result = IndexAllModelForwardRunner(
        IndexAllModelForwardConfig.for_stream(args.stream)
    ).run()
    print(json.dumps(_canonical(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CACHE_BASE",
    "EVIDENCE_ROLE",
    "PROTOCOL_VERSION",
    "IndexAllModelForwardConfig",
    "IndexAllModelForwardRunner",
    "validate_selected_policies",
]
