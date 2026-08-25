"""Torch deep classifiers (MLP / LSTM / GRU) with a scikit-learn-style API.

All three train with class-weighted cross-entropy (the flat class dominates M15
labels), standardize features using train-fold statistics only, and run on GPU
automatically when CUDA is available. The sequence models build their input
windows from PAST rows only: the window for row t ends at row t, with the head
of a fold padded by edge replication rather than by reaching into earlier data.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

CLASSES = (0, 1, 2)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_float32(X) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("X must be 2-D (rows x features)")
    return arr


def _class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.array([(y == c).sum() for c in CLASSES], dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    weights = len(y) / (len(CLASSES) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weight: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """Class- and sample-weighted mean matching CrossEntropyLoss semantics."""
    per_row = nn.functional.cross_entropy(
        logits, targets, weight=class_weight, reduction="none"
    )
    denominator = (class_weight[targets] * sample_weight).sum().clamp_min(
        torch.finfo(per_row.dtype).eps
    )
    return (per_row * sample_weight).sum() / denominator


def make_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Return (n, seq_len, features) windows where window t ends at row t.

    The first seq_len-1 rows lack full history, so the head is padded by
    repeating row 0 — no row ever sees data from after itself.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    pad = np.repeat(X[:1], seq_len - 1, axis=0)
    padded = np.concatenate([pad, X], axis=0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (seq_len, X.shape[1]))
    return np.ascontiguousarray(windows[:, 0])


class _MLPNet(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)]
            prev = width
        layers.append(nn.Linear(prev, len(CLASSES)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _SeqNet(nn.Module):
    def __init__(self, n_features: int, cell: str, hidden_size: int, num_layers: int,
                 dropout: float):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, len(CLASSES))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])       # hidden state at the window's final bar


class _TorchClassifierBase:
    """Shared fit/predict machinery; subclasses provide _build_net and _prepare."""

    def __init__(self, *, epochs: int, batch_size: int, lr: float, seed: int):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.classes_ = np.array(CLASSES)
        self._net: nn.Module | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def _standardize(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self._mean = X.mean(axis=0)
            self._std = np.where(X.std(axis=0) > 1e-12, X.std(axis=0), 1.0)
        return (X - self._mean) / self._std

    def _build_net(self, n_features: int) -> nn.Module:
        raise NotImplementedError

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        """Map standardized 2-D features to the network's input layout."""
        return X

    def fit(self, X, y, sample_weight=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = self._standardize(_to_float32(X), fit=True)
        y = np.asarray(y, dtype=np.int64)
        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=np.float32)
        sample_weight = np.asarray(sample_weight, dtype=np.float32)
        if sample_weight.shape != (len(y),):
            raise ValueError("sample_weight must have one value per training row")
        if not np.isfinite(sample_weight).all() or (sample_weight < 0).any():
            raise ValueError("sample_weight must be finite and non-negative")
        device = _device()
        inputs = torch.tensor(self._prepare(X), device=device)
        targets = torch.tensor(y, device=device)
        weights = torch.tensor(sample_weight, device=device)
        class_weight = _class_weights(y).to(device)
        net = self._build_net(X.shape[1]).to(device)
        optim = torch.optim.Adam(net.parameters(), lr=self.lr)

        net.train()
        n = len(inputs)
        for _ in range(self.epochs):
            order = torch.randperm(n, device=device)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                optim.zero_grad()
                loss = _weighted_cross_entropy(
                    net(inputs[idx]), targets[idx], class_weight, weights[idx]
                )
                loss.backward()
                optim.step()
        self._net = net.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("fit must be called before predict")
        X = self._standardize(_to_float32(X), fit=False)
        device = _device()
        inputs = torch.tensor(self._prepare(X), device=device)
        probs = []
        for start in range(0, len(inputs), self.batch_size):
            logits = self._net(inputs[start:start + self.batch_size])
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class TorchMLPClassifier(_TorchClassifierBase):
    def __init__(self, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.1,
                 epochs: int = 20, batch_size: int = 1024, lr: float = 1e-3, seed: int = 42):
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
        self.hidden = tuple(hidden)
        self.dropout = dropout

    def _build_net(self, n_features: int) -> nn.Module:
        return _MLPNet(n_features, self.hidden, self.dropout)


class TorchSequenceClassifier(_TorchClassifierBase):
    def __init__(self, cell: str = "lstm", seq_len: int = 32, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.0, epochs: int = 10,
                 batch_size: int = 512, lr: float = 1e-3, seed: int = 42):
        if cell not in ("lstm", "gru"):
            raise ValueError("cell must be 'lstm' or 'gru'")
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
        self.cell = cell
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

    def _build_net(self, n_features: int) -> nn.Module:
        return _SeqNet(n_features, self.cell, self.hidden_size, self.num_layers, self.dropout)

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        return make_sequences(X, self.seq_len)
