"""Validation and non-IID partitioning for precomputed face embeddings."""
from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_embeddings(x_path: str, y_path: str) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    X = np.load(x_path, allow_pickle=False); y_raw = np.load(y_path, allow_pickle=False)
    if X.ndim != 2: raise ValueError("X must be 2-D (samples, embedding_dim)")
    if y_raw.ndim != 1: raise ValueError("y must be 1-D")
    if len(X) != len(y_raw): raise ValueError("X and y length mismatch")
    if len(X) < 4: raise ValueError("at least four samples are required")
    if not np.isfinite(X).all(): raise ValueError("X contains NaN or infinity")
    encoder = LabelEncoder(); y = encoder.fit_transform(y_raw.astype(str))
    if len(encoder.classes_) < 2: raise ValueError("at least two identity classes are required")
    return X.astype(np.float32), y.astype(np.int64), encoder

def split_embeddings(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split once before clients are formed to avoid train/test index overlap."""
    if not 0.05 <= test_size <= 0.5: raise ValueError("test_size must be between 0.05 and 0.5")
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(X) < 4: raise ValueError("invalid X/y shapes or sample count")
    _, counts = np.unique(y, return_counts=True); n_test = int(np.ceil(len(y) * test_size))
    stratify_ok = counts.min() >= 2 and n_test >= len(counts) and len(y) - n_test >= len(counts)
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y if stratify_ok else None)

def dirichlet_partition_indices(labels: Sequence[int] | np.ndarray, num_clients: int = 3, alpha: float = 0.5, seed: int = 42) -> list[np.ndarray]:
    """Class-wise Dirichlet partition of training-row indices; each row appears once."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) < num_clients: raise ValueError("labels must be 1-D and have enough rows")
    if num_clients < 2: raise ValueError("num_clients must be >= 2")
    if alpha <= 0: raise ValueError("alpha must be positive")
    rng = np.random.default_rng(seed); parts: list[list[int]] = [[] for _ in range(num_clients)]
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label); rng.shuffle(idx)
        proportions = rng.dirichlet(np.full(num_clients, alpha)); cuts = (np.cumsum(proportions)[:-1] * len(idx)).astype(int)
        for client, chunk in enumerate(np.split(idx, cuts)): parts[client].extend(chunk.tolist())
    for client in range(num_clients):
        if not parts[client]:
            donor = max(range(num_clients), key=lambda i: len(parts[i]))
            if len(parts[donor]) < 2: raise ValueError("not enough samples to populate every client")
            parts[client].append(parts[donor].pop())
    return [np.asarray(sorted(part), dtype=np.int64) for part in parts]
