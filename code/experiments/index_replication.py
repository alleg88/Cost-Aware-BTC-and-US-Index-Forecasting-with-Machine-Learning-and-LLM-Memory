"""Separate causal nine-model replications for USA500 and USATECH.

The runner is deliberately staged. ``run_gate`` touches only price and VIX
features and freezes one instrument-level base choice from 2024 OOF evidence.
Only after that artifact exists may ``run_models`` load sentiment caches, fit
the four conditional-base arms, calibrate DZ/tau on H1 2025, and replay the
frozen July-2025 to March-2026 forward span.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

from evaluation.splits import BlockingTimeSeriesSplit
from experiments.index_replication_protocol import (
    CALIBRATION_START,
    CUTOFF,
    FORWARD_END,
    FORWARD_START,
    MODEL_NAMES,
    SELECTION_END,
    SELECTION_START,
    TAUS,
    VIX_FEATURE_COLS,
    WIDTHS,
    build_vix_block,
    daily_economics,
    decide_vix_admission,
    join_completed_vix,
    select_h1_policy,
)
from features.build import FEATURE_COLS, add_features
from features import index_sentiment as index_sentiment_features
from models.zoo import MODELS, _aligned_proba


CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CODE_ROOT / "data"
CACHE_BASE = CODE_ROOT / "experiments" / "cache" / "index_replication"
PROTOCOL_VERSION = "index-replication-v3-vix-majority-next-open"
INDEX_SENTIMENT_IMPLEMENTATION_SHA256 = hashlib.sha256(
    inspect.getsource(index_sentiment_features).encode("utf-8")
).hexdigest()
BAR_SIZE = pd.Timedelta(minutes=15)
LOOKBACK_DAYS = 180
ARMS = ("selected_base", "deberta_matched", "deepseek_matched", "deepseek_full")
STREAM_CONFIG = {
    "usa500": ("USA500IDXUSD", 2.0),
    "usatech": ("USATECHIDXUSD", 3.0),
}


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _canonical(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return _utc(value).isoformat()
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


def _frame_hash(frame: pd.DataFrame | pd.Series) -> str:
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256(values.tobytes())
    if isinstance(frame, pd.DataFrame):
        digest.update(json.dumps(list(frame.columns)).encode("utf-8"))
    else:
        digest.update(str(frame.name).encode("utf-8"))
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, engine="pyarrow", index=index)
    os.replace(temporary, path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_canonical(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class IndexReplicationConfig:
    stream: str
    instrument: str
    cost_bps: float
    data_dir: Path
    output_root: Path
    end_exclusive: pd.Timestamp = CUTOFF

    def __post_init__(self) -> None:
        if self.stream not in STREAM_CONFIG:
            raise ValueError(f"stream must be one of {tuple(STREAM_CONFIG)}")
        expected_instrument, expected_cost = STREAM_CONFIG[self.stream]
        if self.instrument != expected_instrument or float(self.cost_bps) != expected_cost:
            raise ValueError("instrument and cost must match the frozen stream contract")
        if _utc(self.end_exclusive) != CUTOFF:
            raise PermissionError("index model boundary must remain 2026-04-01 UTC")
        if Path(self.output_root).name != self.stream:
            raise ValueError("output_root must end with the instrument stream name")

    @classmethod
    def for_stream(
        cls,
        stream: str,
        *,
        data_dir: str | Path = DATA_DIR,
        output_base: str | Path = CACHE_BASE,
    ) -> "IndexReplicationConfig":
        if stream not in STREAM_CONFIG:
            raise ValueError(f"stream must be one of {tuple(STREAM_CONFIG)}")
        instrument, cost = STREAM_CONFIG[stream]
        return cls(
            stream=stream,
            instrument=instrument,
            cost_bps=cost,
            data_dir=Path(data_dir),
            output_root=Path(output_base) / stream,
        )


def build_selection_folds(index: pd.DatetimeIndex) -> tuple[dict[str, Any], ...]:
    """Return the frozen five non-overlapping 80/20 blocks with four-bar embargo."""
    rows: list[dict[str, Any]] = []
    splitter = BlockingTimeSeriesSplit(n_splits=5, train_frac=0.8, embargo=4)
    for fold_id, (train, test) in enumerate(splitter.split(index)):
        rows.append(
            {
                "fold_id": int(fold_id),
                "train_positions": tuple(int(value) for value in train),
                "test_positions": tuple(int(value) for value in test),
                "train_start": index[train[0]],
                "train_end": index[train[-1]] + BAR_SIZE,
                "test_start": index[test[0]],
                "test_end": index[test[-1]] + BAR_SIZE,
            }
        )
    if len(rows) != 5:
        raise ValueError("index selection requires exactly five folds")
    return tuple(rows)


def make_index_label(bars: pd.DataFrame, threshold_bps: float) -> pd.Series:
    """Next completed M15 direction; a session gap is never called a 15m target."""
    close = bars["close"].astype(float)
    forward = close.shift(-1) / close - 1.0
    current = pd.Series(bars.index, index=bars.index)
    following = current.shift(-1)
    consecutive = following.sub(current).eq(BAR_SIZE)
    threshold = float(threshold_bps) / 10_000.0
    label = pd.Series(1, index=bars.index, dtype="int64", name="label")
    label.loc[forward.ge(threshold) & consecutive] = 2
    label.loc[forward.le(-threshold) & consecutive] = 0
    label.loc[~consecutive | forward.isna()] = -1
    return label


def fit_prediction_frame(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable,
    model_name: str,
    arm: str,
    width_bps: int,
    fold: Mapping[str, Any],
    fit_id: str,
) -> pd.DataFrame:
    """Fit one causal fold/span and return aligned three-class probabilities."""
    train_positions = np.asarray(fold["train_positions"], dtype=int)
    test_positions = np.asarray(fold["test_positions"], dtype=int)
    X_train = X.iloc[train_positions]
    y_train = y.reindex(X_train.index)
    X_test = X.iloc[test_positions]
    y_test = y.reindex(X_test.index)
    if len(X_train) <= 1 or X_test.empty:
        raise ValueError("fit span is too small")
    X_train = X_train.iloc[:-1]
    y_train = y_train.iloc[:-1]
    if y_train.nunique() < 2:
        raise ValueError("training span needs at least two classes")
    model = model_factory({})
    model.fit(X_train, y_train)
    probabilities = _aligned_proba(model, X_test)
    prediction = np.asarray((0, 1, 2))[probabilities.argmax(axis=1)]
    train_end_exclusive = X_train.index[-1] + BAR_SIZE
    test_start = X_test.index[0]
    if train_end_exclusive >= test_start:
        raise AssertionError("training label tail was not separated from test")
    return pd.DataFrame(
        {
            "timestamp": X_test.index,
            "model_name": model_name,
            "arm": arm,
            "width_bps": int(width_bps),
            "fold_id": int(fold["fold_id"]),
            "y_true": y_test.astype(int).to_numpy(),
            "pred": prediction,
            "confidence": probabilities.max(axis=1),
            "p_short": probabilities[:, 0],
            "p_flat": probabilities[:, 1],
            "p_long": probabilities[:, 2],
            "train_start": X_train.index[0],
            "train_end_exclusive": train_end_exclusive,
            "test_start": test_start,
            "test_end": X_test.index[-1] + BAR_SIZE,
            "fit_id": fit_id,
        }
    )


def _span_fold(
    index: pd.DatetimeIndex,
    *,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    lookback_days: int = LOOKBACK_DAYS,
) -> dict[str, Any]:
    train_start = train_end - pd.Timedelta(days=int(lookback_days))
    train_positions = np.flatnonzero((index >= train_start) & (index < train_end))
    test_positions = np.flatnonzero((index >= test_start) & (index < test_end))
    if len(train_positions) <= 1 or not len(test_positions):
        raise ValueError("monthly/frozen span has insufficient rows")
    return {
        "fold_id": -1,
        "train_positions": tuple(int(value) for value in train_positions),
        "test_positions": tuple(int(value) for value in test_positions),
        "train_start": index[train_positions[0]],
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
    }


def simulate_one_bar(
    bars: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    tau: float,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Trade the next complete M15 bar after a close-time prediction."""
    start_utc, end_utc = _utc(start), _utc(end)
    stage = bars.loc[(bars.index >= start_utc) & (bars.index < end_utc)].sort_index()
    per_bar = pd.Series(0.0, index=stage.index, name="net_return")
    prediction = predictions.copy()
    prediction["timestamp"] = pd.to_datetime(prediction["timestamp"], utc=True)
    prediction = prediction.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    position = {timestamp: offset for offset, timestamp in enumerate(bars.index)}
    active_until = pd.Timestamp.min.tz_localize("UTC")
    rows: list[dict[str, Any]] = []
    for row in prediction.itertuples(index=False):
        timestamp = pd.Timestamp(row.timestamp)
        if timestamp < start_utc or timestamp >= end_utc or timestamp not in position:
            continue
        if int(row.pred) == 1 or float(row.confidence) < float(tau):
            continue
        offset = position[timestamp]
        if offset + 1 >= len(bars):
            continue
        next_timestamp = bars.index[offset + 1]
        if next_timestamp - timestamp != BAR_SIZE or next_timestamp >= end_utc:
            continue
        decision_time = pd.Timestamp(bars.iloc[offset]["available_at"])
        entry_time = next_timestamp
        exit_time = pd.Timestamp(bars.iloc[offset + 1]["available_at"])
        if decision_time != entry_time or exit_time != next_timestamp + BAR_SIZE:
            continue
        if exit_time > end_utc:
            continue
        if entry_time < active_until:
            continue
        side = 1 if int(row.pred) == 2 else -1
        entry = float(bars.iloc[offset + 1]["open"])
        exit_ = float(bars.iloc[offset + 1]["close"])
        gross = side * (exit_ / entry - 1.0)
        cost = float(cost_bps) / 10_000.0
        net = gross - cost
        rows.append(
            {
                "signal_bar_open": timestamp,
                "decision_time": decision_time,
                "entry_bar_open": next_timestamp,
                "exit_bar_open": next_timestamp,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "side": side,
                "entry_price": entry,
                "exit_price": exit_,
                "confidence": float(row.confidence),
                "gross_return": gross,
                "cost_return": cost,
                "net_return": net,
            }
        )
        if next_timestamp in per_bar.index:
            per_bar.loc[next_timestamp] += net
        active_until = exit_time
    ledger_columns = [
        "signal_bar_open",
        "decision_time",
        "entry_bar_open",
        "exit_bar_open",
        "entry_time",
        "exit_time",
        "side",
        "entry_price",
        "exit_price",
        "confidence",
        "gross_return",
        "cost_return",
        "net_return",
    ]
    return pd.DataFrame(rows, columns=ledger_columns), per_bar


class IndexReplicationRunner:
    """Resumable, instrument-isolated gate and four-arm experiment runner."""

    def __init__(
        self,
        config: IndexReplicationConfig,
        *,
        model_factories: Mapping[str, Callable] | None = None,
        bars: pd.DataFrame | None = None,
        vix_bars: pd.DataFrame | None = None,
    ):
        self.config = config
        self.model_factories = dict(model_factories or {name: MODELS[name] for name in MODEL_NAMES})
        self.output_root = Path(config.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.bars = self._validate_bars(bars if bars is not None else self._read_grid(config.stream))
        self.vix_bars = self._validate_bars(
            vix_bars if vix_bars is not None else self._read_grid("volidx")
        )
        self._feature_cache: dict[str, pd.DataFrame] = {}
        self._label_cache: dict[int, pd.Series] = {}
        self._write_protocol_manifest()

    def _read_grid(self, stem: str) -> pd.DataFrame:
        path = Path(self.config.data_dir) / f"{stem}_15min_2021_2026.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_parquet(
            path,
            filters=[("timestamp", "<", self.config.end_exclusive.to_pydatetime())],
        )

    def _validate_bars(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close", "volume", "available_at", "complete_bar"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"M15 frame misses columns: {sorted(missing)}")
        current = frame.copy().sort_index()
        if current.index.tz is None:
            raise ValueError("M15 index must be timezone-aware")
        current["available_at"] = pd.to_datetime(current["available_at"], utc=True)
        current = current.loc[
            current["complete_bar"].astype(bool)
            & (current.index < self.config.end_exclusive)
            & current["available_at"].le(self.config.end_exclusive)
        ]
        if current.empty or current.index.max() >= CUTOFF:
            raise AssertionError("M15 source crossed or missed the sealed boundary")
        if not current.index.is_unique or not current.index.is_monotonic_increasing:
            raise ValueError("M15 index must be unique and sorted")
        return current

    def _write_protocol_manifest(self) -> None:
        body = {
            "protocol_version": PROTOCOL_VERSION,
            "config": asdict(self.config),
            "model_names": list(MODEL_NAMES),
            "widths_bps": list(WIDTHS),
            "taus": list(TAUS),
            "arms": list(ARMS),
            "selection": [SELECTION_START, SELECTION_END],
            "calibration": [CALIBRATION_START, FORWARD_START],
            "forward": [FORWARD_START, FORWARD_END],
            "label": "next consecutive completed M15 close, short/flat/long",
            "execution": "decision at signal-bar close; enter next consecutive M15 open",
            "hold": "exit that entry bar's close, non-overlapping",
            "source_max_index": self.bars.index.max(),
            "vix_source_max_index": self.vix_bars.index.max(),
            "implementation_sha256": hashlib.sha256(
                inspect.getsource(IndexReplicationRunner).encode("utf-8")
            ).hexdigest(),
            "sentiment_feature_implementation_sha256": INDEX_SENTIMENT_IMPLEMENTATION_SHA256,
        }
        _atomic_json({**body, "protocol_hash": _payload_hash(body)}, self.output_root / "protocol_manifest.json")

    def _state(self, stage: str, **detail: Any) -> None:
        _atomic_json(
            {"stage": stage, "updated_at_utc": pd.Timestamp.now("UTC"), **detail},
            self.output_root / "run_state.json",
        )

    def _price_and_vix_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if "price" in self._feature_cache and "price_vix" in self._feature_cache:
            return self._feature_cache["price"], self._feature_cache["price_vix"]
        engineered = add_features(self.bars)
        price = engineered.loc[:, FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
        decisions = self.bars[["close"]].copy()
        decisions["decision_time"] = self.bars["available_at"]
        vix = join_completed_vix(decisions, build_vix_block(self.vix_bars))
        price_vix = pd.concat([price, vix.loc[:, VIX_FEATURE_COLS]], axis=1)
        common = price.dropna().index.intersection(price_vix.dropna().index)
        self._feature_cache["price"] = price.loc[common].astype(float)
        self._feature_cache["price_vix"] = price_vix.loc[common].astype(float)
        return self._feature_cache["price"], self._feature_cache["price_vix"]

    def _load_vix_decision(self) -> dict[str, Any]:
        path = self.output_root / "vix_admission.json"
        if not path.exists():
            raise FileNotFoundError("run and freeze the per-index VIX gate first")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("stale or unknown VIX admission protocol")
        manifest = json.loads(
            (self.output_root / "protocol_manifest.json").read_text(encoding="utf-8")
        )
        if payload.get("protocol_hash") != manifest.get("protocol_hash"):
            raise ValueError("VIX admission does not match the active protocol hash")
        if payload.get("gate_complete") is not True:
            raise ValueError("VIX admission is not atomically complete")
        if payload.get("index_source_sha256") != _frame_hash(self.bars):
            raise ValueError("VIX admission index data fingerprint changed")
        if payload.get("vix_source_sha256") != _frame_hash(self.vix_bars):
            raise ValueError("VIX admission VIX data fingerprint changed")
        gate_path = self.output_root / "vix_gate_paired_2024.parquet"
        if not gate_path.exists() or payload.get("paired_table_sha256") != _frame_hash(
            pd.read_parquet(gate_path)
        ):
            raise ValueError("VIX admission gate table is missing or changed")
        if payload.get("selected_base") not in {"price", "price_vix"}:
            raise ValueError("invalid frozen VIX decision")
        return payload

    def _feature_frame(self, arm: str) -> pd.DataFrame:
        if arm in {"price", "price_vix"}:
            return self._price_and_vix_frames()[0 if arm == "price" else 1]
        if arm in self._feature_cache:
            return self._feature_cache[arm]
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")

        decision = self._load_vix_decision()
        base = self._feature_frame(str(decision["selected_base"]))
        if arm == "selected_base":
            self._feature_cache[arm] = base
            return base
        bar_index = self.bars.index
        if arm == "deberta_matched":
            sentiment = index_sentiment_features.build_matched_index_features(
                self.config.stream, bar_index, scorer="classic"
            )
        elif arm == "deepseek_matched":
            sentiment = index_sentiment_features.build_matched_index_features(
                self.config.stream, bar_index, scorer="llm"
            )
        else:
            sentiment = index_sentiment_features.build_deepseek_full_features(
                self.config.stream, bar_index
            )
        combined = pd.concat([base, sentiment], axis=1).replace([np.inf, -np.inf], np.nan)
        combined = combined.loc[base.index].dropna()
        self._feature_cache[arm] = combined.astype(float)
        return self._feature_cache[arm]

    def dataset(self, arm: str, width_bps: int) -> tuple[pd.DataFrame, pd.Series]:
        if int(width_bps) <= 0:
            raise ValueError("width must be a positive integer number of basis points")
        X = self._feature_frame(arm)
        if int(width_bps) not in self._label_cache:
            self._label_cache[int(width_bps)] = make_index_label(self.bars, width_bps)
        y = self._label_cache[int(width_bps)].reindex(X.index)
        valid = y.ne(-1) & y.notna() & X.notna().all(axis=1)
        return X.loc[valid], y.loc[valid].astype(int)

    def _prediction_fingerprint(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        fold: Mapping[str, Any],
        model_name: str,
        arm: str,
        width_bps: int,
        stage: str,
    ) -> str:
        train = list(fold["train_positions"])
        test = list(fold["test_positions"])
        fingerprint_payload = {
                "protocol": PROTOCOL_VERSION,
                "stream": self.config.stream,
                "cost_bps": self.config.cost_bps,
                "stage": stage,
                "model": model_name,
                "arm": arm,
                "width_bps": int(width_bps),
                "fold": {key: fold[key] for key in ("fold_id", "train_start", "train_end", "test_start", "test_end")},
                "train_X": _frame_hash(X.iloc[train]),
                "train_y": _frame_hash(y.iloc[train]),
                "test_X": _frame_hash(X.iloc[test]),
                "feature_columns": list(X.columns),
            }
        if arm not in {"price", "price_vix"}:
            fingerprint_payload["sentiment_feature_implementation"] = (
                INDEX_SENTIMENT_IMPLEMENTATION_SHA256
            )
        return _payload_hash(fingerprint_payload)

    def _fit_cached(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        fold: Mapping[str, Any],
        model_name: str,
        arm: str,
        width_bps: int,
        stage: str,
        segment: str,
    ) -> pd.DataFrame:
        fingerprint = self._prediction_fingerprint(
            X=X,
            y=y,
            fold=fold,
            model_name=model_name,
            arm=arm,
            width_bps=width_bps,
            stage=stage,
        )
        path = (
            self.output_root
            / "predictions"
            / stage
            / arm
            / model_name
            / f"w{width_bps}_{segment}_{fingerprint[:16]}.parquet"
        )
        if path.exists():
            cached = pd.read_parquet(path)
            if cached["fingerprint"].eq(fingerprint).all() and cached["stream"].eq(self.config.stream).all():
                return cached
        fit_id = f"{self.config.stream}-{stage}-{arm}-{model_name}-w{width_bps}-{segment}-{fingerprint[:12]}"
        prediction = fit_prediction_frame(
            X=X,
            y=y,
            model_factory=self.model_factories[model_name],
            model_name=model_name,
            arm=arm,
            width_bps=width_bps,
            fold=fold,
            fit_id=fit_id,
        )
        prediction["stream"] = self.config.stream
        prediction["fingerprint"] = fingerprint
        _atomic_parquet(prediction, path)
        return prediction

    def _selection_data(self, arm: str, width_bps: int) -> tuple[pd.DataFrame, pd.Series]:
        X, y = self.dataset(arm, width_bps)
        mask = (X.index >= SELECTION_START) & (X.index < SELECTION_END)
        return X.loc[mask], y.loc[mask]

    @staticmethod
    def _classification_row(
        *, arm: str, model_name: str, width_bps: int, frames: Sequence[pd.DataFrame]
    ) -> dict[str, Any]:
        combined = pd.concat(frames, ignore_index=True)
        actual = combined["y_true"].astype(int)
        predicted = combined["pred"].astype(int)
        recalls = recall_score(
            actual, predicted, labels=[0, 1, 2], average=None, zero_division=0
        )
        return {
            "arm": arm,
            "model_name": model_name,
            "width_bps": int(width_bps),
            "rows": len(combined),
            "macro_f1": float(f1_score(actual, predicted, average="macro", labels=[0, 1, 2], zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
            "short_recall": float(recalls[0]),
            "flat_recall": float(recalls[1]),
            "long_recall": float(recalls[2]),
        }

    def run_gate(
        self,
        *,
        model_names: Sequence[str] = MODEL_NAMES,
        widths: Sequence[int] = WIDTHS,
        fold_limit: int = 5,
        freeze: bool = True,
    ) -> dict[str, Any]:
        """Fit price versus price+VIX OOF pairs without touching sentiment files."""
        rows: list[dict[str, Any]] = []
        classification: list[dict[str, Any]] = []
        total = len(model_names) * len(widths)
        completed = 0
        for width in widths:
            price_X, price_y = self._selection_data("price", int(width))
            vix_X, vix_y = self._selection_data("price_vix", int(width))
            common = price_X.index.intersection(vix_X.index)
            price_X, price_y = price_X.loc[common], price_y.loc[common]
            vix_X, vix_y = vix_X.loc[common], vix_y.loc[common]
            if not price_y.equals(vix_y):
                raise AssertionError("price and VIX labels/timestamps are not paired")
            folds = build_selection_folds(common)[: int(fold_limit)]
            for model_name in model_names:
                arm_frames: dict[str, list[pd.DataFrame]] = {"price": [], "price_vix": []}
                for fold in folds:
                    predictions: dict[str, pd.DataFrame] = {}
                    for arm, X in (("price", price_X), ("price_vix", vix_X)):
                        predictions[arm] = self._fit_cached(
                            X=X,
                            y=price_y,
                            fold=fold,
                            model_name=model_name,
                            arm=arm,
                            width_bps=int(width),
                            stage="gate",
                            segment=f"fold{int(fold['fold_id'])}",
                        )
                        arm_frames[arm].append(predictions[arm])
                    ledgers: dict[str, pd.DataFrame] = {}
                    for arm in ("price", "price_vix"):
                        ledgers[arm], _ = simulate_one_bar(
                            self.bars,
                            predictions[arm],
                            start=fold["test_start"],
                            end=fold["test_end"],
                            tau=0.0,
                            cost_bps=self.config.cost_bps,
                        )
                    common_hash = _payload_hash([stamp.isoformat() for stamp in predictions["price"]["timestamp"]])
                    rows.append(
                        {
                            "model_name": model_name,
                            "width_bps": int(width),
                            "fold_id": int(fold["fold_id"]),
                            "price_net": float(ledgers["price"]["net_return"].sum()),
                            "vix_net": float(ledgers["price_vix"]["net_return"].sum()),
                            "price_trades": int(len(ledgers["price"])),
                            "vix_trades": int(len(ledgers["price_vix"])),
                            "common_timestamp_hash": common_hash,
                        }
                    )
                for arm in ("price", "price_vix"):
                    classification.append(
                        self._classification_row(
                            arm=arm,
                            model_name=model_name,
                            width_bps=int(width),
                            frames=arm_frames[arm],
                        )
                    )
                completed += 1
                self._state("vix_gate", completed=completed, total=total, model=model_name, width_bps=int(width))
                print(f"[{self.config.stream}] gate {completed}/{total}: {model_name} DZ{width}", flush=True)
        paired = pd.DataFrame(rows)
        _atomic_parquet(paired, self.output_root / "vix_gate_paired_2024.parquet")
        _atomic_parquet(pd.DataFrame(classification), self.output_root / "vix_gate_classification_2024.parquet")
        if not freeze:
            return {"status": "smoke_complete", "rows": len(paired)}
        if tuple(model_names) != MODEL_NAMES or tuple(int(value) for value in widths) != WIDTHS or int(fold_limit) != 5:
            raise ValueError("only the complete frozen gate can write an admission decision")
        decision = decide_vix_admission(paired)
        protocol = json.loads(
            (self.output_root / "protocol_manifest.json").read_text(encoding="utf-8")
        )
        payload = {
            **decision.to_dict(),
            "protocol_version": PROTOCOL_VERSION,
            "protocol_hash": protocol["protocol_hash"],
            "gate_complete": True,
            "execution": "next_consecutive_m15_open_to_same_bar_close",
            "stream": self.config.stream,
            "selection_start": SELECTION_START,
            "selection_end_exclusive": SELECTION_END,
            "paired_table_sha256": _frame_hash(paired),
            "index_source_sha256": _frame_hash(self.bars),
            "vix_source_sha256": _frame_hash(self.vix_bars),
            "frozen_before_sentiment": True,
        }
        _atomic_json(payload, self.output_root / "vix_admission.json")
        self._state("vix_gate_complete", selected_base=decision.selected_base, admitted=decision.admitted)
        return payload

    def _oof_arm(
        self, arm: str, model_name: str, width_bps: int
    ) -> tuple[dict[str, Any], list[pd.DataFrame]]:
        feature_key = arm
        cache_arm = arm
        stage = "selection"
        if arm == "selected_base":
            feature_key = str(self._load_vix_decision()["selected_base"])
            cache_arm = feature_key
            stage = "gate"
        X, y = self._selection_data(feature_key, width_bps)
        folds = build_selection_folds(X.index)
        frames = [
            self._fit_cached(
                X=X,
                y=y,
                fold=fold,
                model_name=model_name,
                arm=cache_arm,
                width_bps=width_bps,
                stage=stage,
                segment=f"fold{int(fold['fold_id'])}",
            )
            for fold in folds
        ]
        return self._classification_row(
            arm=arm, model_name=model_name, width_bps=width_bps, frames=frames
        ), frames

    def _monthly_predictions(
        self, arm: str, model_name: str, width_bps: int
    ) -> pd.DataFrame:
        X, y = self.dataset(arm, width_bps)
        edges = pd.date_range(CALIBRATION_START, FORWARD_START, freq="MS")
        frames = []
        for start, end in zip(edges[:-1], edges[1:]):
            fold = _span_fold(
                X.index,
                train_end=start,
                test_start=start,
                test_end=end,
            )
            frames.append(
                self._fit_cached(
                    X=X,
                    y=y,
                    fold=fold,
                    model_name=model_name,
                    arm=arm,
                    width_bps=width_bps,
                    stage="calibration",
                    segment=f"{start:%Y_%m}",
                )
            )
        combined = pd.concat(frames, ignore_index=True).sort_values("timestamp")
        if pd.to_datetime(combined["timestamp"], utc=True).duplicated().any():
            raise AssertionError("monthly predictions overlap")
        return combined

    def _h1_grid(
        self, arm: str, model_name: str, width_bps: int, prediction: pd.DataFrame
    ) -> pd.DataFrame:
        rows = []
        month_edges = pd.date_range(CALIBRATION_START, FORWARD_START, freq="MS")
        for tau in TAUS:
            ledger, per_bar = simulate_one_bar(
                self.bars,
                prediction,
                start=CALIBRATION_START,
                end=FORWARD_START,
                tau=tau,
                cost_bps=self.config.cost_bps,
            )
            economics = daily_economics(
                ledger, per_bar, start=CALIBRATION_START, end=FORWARD_START
            )
            positive_months = sum(
                float(per_bar.loc[(per_bar.index >= left) & (per_bar.index < right)].sum()) > 0
                for left, right in zip(month_edges[:-1], month_edges[1:])
            )
            rows.append(
                {
                    "arm": arm,
                    "model_name": model_name,
                    "width_bps": int(width_bps),
                    "tau": float(tau),
                    "positive_months": int(positive_months),
                    **economics,
                }
            )
        return pd.DataFrame(rows)

    def _forward_prediction(
        self, arm: str, model_name: str, width_bps: int
    ) -> pd.DataFrame:
        X, y = self.dataset(arm, width_bps)
        fold = _span_fold(
            X.index,
            train_end=FORWARD_START,
            test_start=FORWARD_START,
            test_end=FORWARD_END,
        )
        return self._fit_cached(
            X=X,
            y=y,
            fold=fold,
            model_name=model_name,
            arm=arm,
            width_bps=width_bps,
            stage="forward",
            segment="2025_07_to_2026_03",
        )

    def run_models(
        self,
        *,
        model_names: Sequence[str] = MODEL_NAMES,
        arms: Sequence[str] = ARMS,
        widths: Sequence[int] = WIDTHS,
    ) -> dict[str, Any]:
        """Run all four conditional-base arms after the VIX decision is frozen."""
        decision = self._load_vix_decision()
        classification_rows: list[dict[str, Any]] = []
        h1_frames: list[pd.DataFrame] = []
        selected_rows: list[dict[str, Any]] = []
        total = len(arms) * len(model_names)
        completed = 0
        for arm in arms:
            self._feature_frame(arm)
            for model_name in model_names:
                model_h1 = []
                for width in widths:
                    classification, _ = self._oof_arm(arm, model_name, int(width))
                    classification_rows.append(classification)
                    monthly = self._monthly_predictions(arm, model_name, int(width))
                    model_h1.append(self._h1_grid(arm, model_name, int(width), monthly))
                grid = pd.concat(model_h1, ignore_index=True)
                h1_frames.append(grid)
                winner = select_h1_policy(grid).to_dict()
                winner.update(
                    {
                        "arm": arm,
                        "model_name": model_name,
                        "h1_execution_status": (
                            "eligible" if bool(winner["eligible"])
                            else "diagnostic_only_no_eligible_policy"
                        ),
                    }
                )
                selected_rows.append(winner)
                completed += 1
                self._state("models_h1", completed=completed, total=total, arm=arm, model=model_name)
                print(f"[{self.config.stream}] models {completed}/{total}: {arm}/{model_name}", flush=True)

        classification_table = pd.DataFrame(classification_rows)
        h1_grid = pd.concat(h1_frames, ignore_index=True)
        selected = pd.DataFrame(selected_rows)
        _atomic_parquet(classification_table, self.output_root / "classification_2024.parquet")
        _atomic_parquet(h1_grid, self.output_root / "h1_policy_grid.parquet")
        _atomic_parquet(selected, self.output_root / "h1_selected_policies.parquet")

        forward_rows: list[dict[str, Any]] = []
        forward_prediction_maxima: list[pd.Timestamp] = []
        ledger_root = self.output_root / "forward_ledgers"
        for number, policy in enumerate(selected.to_dict("records"), start=1):
            arm = str(policy["arm"])
            model_name = str(policy["model_name"])
            width = int(policy["width_bps"])
            tau = float(policy["tau"])
            eligible = bool(policy["eligible"])
            if eligible:
                prediction = self._forward_prediction(arm, model_name, width)
                forward_prediction_maxima.append(
                    pd.to_datetime(prediction["timestamp"], utc=True).max()
                )
                fit_id = str(prediction["fit_id"].iloc[0])
                status = "completed"
            else:
                prediction = pd.DataFrame(
                    {
                        "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                        "pred": pd.Series(dtype=int),
                        "confidence": pd.Series(dtype=float),
                    }
                )
                fit_id = "INELIGIBLE_H1_NO_FORWARD_FIT"
                status = "ineligible_flat"
            ledger, per_bar = simulate_one_bar(
                self.bars,
                prediction,
                start=FORWARD_START,
                end=FORWARD_END,
                tau=tau,
                cost_bps=self.config.cost_bps,
            )
            stress_ledger, stress_per_bar = simulate_one_bar(
                self.bars,
                prediction,
                start=FORWARD_START,
                end=FORWARD_END,
                tau=tau,
                cost_bps=2.0 * self.config.cost_bps,
            )
            economics = daily_economics(
                ledger, per_bar, start=FORWARD_START, end=FORWARD_END
            )
            stress = daily_economics(
                stress_ledger, stress_per_bar, start=FORWARD_START, end=FORWARD_END
            )
            forward_rows.append(
                {
                    "arm": arm,
                    "model_name": model_name,
                    "width_bps": width,
                    "tau": tau,
                    "h1_eligible": eligible,
                    "status": status,
                    **economics,
                    "stress_2x_net_return": stress["net_return"],
                    "stress_2x_daily_sortino": stress["daily_sortino"],
                    "fit_id": fit_id,
                }
            )
            ledger_path = ledger_root / arm / f"{model_name}.parquet"
            _atomic_parquet(ledger, ledger_path)
            _atomic_parquet(
                per_bar.rename("net_return").rename_axis("timestamp").reset_index(),
                ledger_path.with_name(f"{model_name}_per_bar.parquet"),
            )
            self._state("forward", completed=number, total=len(selected), arm=arm, model=model_name)
        forward = pd.DataFrame(forward_rows)
        _atomic_parquet(forward, self.output_root / "forward_summary.parquet")
        base = forward.loc[forward["arm"].eq("selected_base"), ["model_name", "net_return", "trades"]].rename(
            columns={"net_return": "base_net_return", "trades": "base_trades"}
        )
        deltas = forward.merge(base, on="model_name", how="left", validate="many_to_one")
        eligible_base = forward.loc[
            forward["arm"].eq("selected_base"), ["model_name", "h1_eligible"]
        ].rename(columns={"h1_eligible": "base_h1_eligible"})
        deltas = deltas.merge(eligible_base, on="model_name", how="left", validate="many_to_one")
        comparable = deltas["h1_eligible"] & deltas["base_h1_eligible"]
        deltas["paired_net_delta"] = np.where(
            comparable, deltas["net_return"] - deltas["base_net_return"], np.nan
        )
        deltas["trade_ratio"] = np.where(
            comparable & deltas["base_trades"].gt(0),
            deltas["trades"] / deltas["base_trades"],
            np.nan,
        )
        _atomic_parquet(deltas, self.output_root / "forward_arm_deltas.parquet")
        result = {
            "stream": self.config.stream,
            "selected_base": decision["selected_base"],
            "classification_rows": len(classification_table),
            "h1_grid_rows": len(h1_grid),
            "selected_policy_rows": len(selected),
            "forward_rows": len(forward),
            "max_prediction_timestamp": (
                max(forward_prediction_maxima) if forward_prediction_maxima else None
            ),
            "q2_loaded": False,
        }
        _atomic_json(result, self.output_root / "result.json")
        self._state("complete", **result)
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", choices=tuple(STREAM_CONFIG), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gate", action="store_true")
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    config = IndexReplicationConfig.for_stream(args.stream)
    if args.smoke:
        smoke_config = IndexReplicationConfig.for_stream(
            args.stream, output_base=CACHE_BASE / "smoke"
        )
        runner = IndexReplicationRunner(smoke_config)
        result = runner.run_gate(
            model_names=("logreg",), widths=(5,), fold_limit=1, freeze=False
        )
    else:
        runner = IndexReplicationRunner(config)
        result = runner.run_gate() if args.gate else runner.run_models()
    print(json.dumps(_canonical(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARMS",
    "CUTOFF",
    "IndexReplicationConfig",
    "IndexReplicationRunner",
    "build_selection_folds",
    "fit_prediction_frame",
    "make_index_label",
    "simulate_one_bar",
]
