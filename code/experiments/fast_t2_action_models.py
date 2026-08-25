"""Matched tabular and recurrent scorers for Fast-T2 action rows."""
from __future__ import annotations

import numpy as np


MODEL_NAMES = ("logreg", "catboost", "xgboost", "gru", "lstm")


def _validate_inputs(
    model_name: str,
    train_static: np.ndarray,
    train_sequence: np.ndarray | None,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    score_static: np.ndarray,
    score_sequence: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unsupported action model: {model_name}")
    train_x = np.asarray(train_static, dtype=float)
    score_x = np.asarray(score_static, dtype=float)
    labels = np.asarray(train_y, dtype=np.int8)
    weights = np.asarray(train_weight, dtype=float)
    if train_x.ndim != 2 or score_x.ndim != 2:
        raise ValueError("static inputs must be two-dimensional")
    if train_x.shape[1] != score_x.shape[1]:
        raise ValueError("static feature counts must match")
    if len(train_x) != len(labels) or len(labels) != len(weights):
        raise ValueError("training arrays must have equal row counts")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("training labels must contain both binary classes")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("training weights must be positive and finite")
    if model_name in {"gru", "lstm"}:
        if train_sequence is None or score_sequence is None:
            raise ValueError("recurrent models require sequence inputs")
        train_seq = np.asarray(train_sequence, dtype=np.float32)
        score_seq = np.asarray(score_sequence, dtype=np.float32)
        if train_seq.ndim != 3 or len(train_seq) != len(train_x):
            raise ValueError("training sequence must align with static rows")
        if score_seq.ndim != 3 or len(score_seq) != len(score_x):
            raise ValueError("scoring sequence must align with static rows")
        if train_seq.shape[1:] != score_seq.shape[1:]:
            raise ValueError("training and scoring sequence shapes must match")
        if not np.isfinite(train_seq).all() or not np.isfinite(score_seq).all():
            raise ValueError("sequence inputs must be finite")
    return train_x, score_x, labels, weights


def _fit_recurrent(
    kind: str,
    train_static: np.ndarray,
    train_sequence: np.ndarray,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    score_static: np.ndarray,
    score_sequence: np.ndarray,
    epochs: int,
) -> np.ndarray:
    import torch
    from torch import nn

    torch.manual_seed(42)
    np.random.seed(42)
    static_mean = train_static.mean(axis=0, keepdims=True)
    static_std = train_static.std(axis=0, keepdims=True)
    static_std = np.where(static_std > 1e-8, static_std, 1.0)
    sequence_mean = train_sequence.mean(axis=(0, 1), keepdims=True)
    sequence_std = train_sequence.std(axis=(0, 1), keepdims=True)
    sequence_std = np.where(sequence_std > 1e-8, sequence_std, 1.0)
    train_static_z = ((train_static - static_mean) / static_std).astype(np.float32)
    score_static_z = ((score_static - static_mean) / static_std).astype(np.float32)
    train_sequence_z = (
        (train_sequence - sequence_mean) / sequence_std
    ).astype(np.float32)
    score_sequence_z = (
        (score_sequence - sequence_mean) / sequence_std
    ).astype(np.float32)

    class CompactActionRNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            recurrent = nn.GRU if kind == "gru" else nn.LSTM
            self.rnn = recurrent(
                train_sequence_z.shape[2], 16, num_layers=1, batch_first=True
            )
            self.static = nn.Linear(train_static_z.shape[1], 16)
            self.head = nn.Linear(32, 1)

        def forward(
            self, sequence: torch.Tensor, static: torch.Tensor
        ) -> torch.Tensor:
            recurrent_output, _ = self.rnn(sequence)
            final_state = recurrent_output[:, -1, :]
            static_state = torch.relu(self.static(static))
            joined = torch.cat([final_state, static_state], dim=1)
            return self.head(joined).squeeze(1)

    model = CompactActionRNN()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sequence_tensor = torch.from_numpy(train_sequence_z)
    static_tensor = torch.from_numpy(train_static_z)
    label_tensor = torch.from_numpy(train_y.astype(np.float32))
    weight_tensor = torch.from_numpy(train_weight.astype(np.float32))
    batch_size = 256
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(label_tensor))
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            optimiser.zero_grad()
            logits = model(sequence_tensor[index], static_tensor[index])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, label_tensor[index], reduction="none"
            )
            loss = (losses * weight_tensor[index]).sum() / weight_tensor[index].sum()
            loss.backward()
            optimiser.step()
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(score_sequence_z), torch.from_numpy(score_static_z)
        )
        return torch.sigmoid(logits).cpu().numpy()


def fit_predict_action_model(
    model_name: str,
    train_static: np.ndarray,
    train_sequence: np.ndarray | None,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    score_static: np.ndarray,
    score_sequence: np.ndarray | None,
    *,
    epochs: int = 12,
) -> np.ndarray:
    """Fit one registered model and return positive-net-R probabilities."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    train_x, score_x, labels, weights = _validate_inputs(
        model_name,
        train_static,
        train_sequence,
        train_y,
        train_weight,
        score_static,
        score_sequence,
    )

    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_x = imputer.fit_transform(train_x)
    score_x = imputer.transform(score_x)
    if model_name == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        model = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
        model.fit(scaler.fit_transform(train_x), labels, sample_weight=weights)
        return model.predict_proba(scaler.transform(score_x))[:, 1]
    if model_name == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=300,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=10.0,
            loss_function="Logloss",
            random_seed=42,
            allow_writing_files=False,
            verbose=False,
            thread_count=1,
        )
        model.fit(train_x, labels, sample_weight=weights)
        return model.predict_proba(score_x)[:, 1]
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            min_child_weight=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=1,
        )
        model.fit(train_x, labels, sample_weight=weights)
        return model.predict_proba(score_x)[:, 1]

    assert train_sequence is not None and score_sequence is not None
    return _fit_recurrent(
        model_name,
        train_x,
        np.asarray(train_sequence, dtype=np.float32),
        labels,
        weights,
        score_x,
        np.asarray(score_sequence, dtype=np.float32),
        epochs,
    )
