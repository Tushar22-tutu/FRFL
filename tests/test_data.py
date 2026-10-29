import numpy as np
import pytest
from frfl.data import dirichlet_partition_indices, split_embeddings

def test_split_does_not_overlap_rows():
    X = np.arange(80, dtype=np.float32).reshape(20, 4); y = np.repeat(np.arange(4), 5)
    Xtr, Xte, ytr, yte = split_embeddings(X, y, test_size=0.2, seed=7)
    assert {tuple(row) for row in Xtr}.isdisjoint({tuple(row) for row in Xte})
    assert len(Xtr) + len(Xte) == len(X)
    assert len(ytr) == len(Xtr) and len(yte) == len(Xte)

def test_partition_assigns_each_row_once():
    labels = np.repeat(np.arange(4), 20); parts = dirichlet_partition_indices(labels, num_clients=3, alpha=0.3, seed=42)
    assert sorted(np.concatenate(parts).tolist()) == list(range(len(labels)))

def test_partition_is_reproducible():
    labels = np.repeat(np.arange(3), 15)
    a = dirichlet_partition_indices(labels, num_clients=3, alpha=0.5, seed=9)
    b = dirichlet_partition_indices(labels, num_clients=3, alpha=0.5, seed=9)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))

def test_invalid_alpha_raises():
    with pytest.raises(ValueError): dirichlet_partition_indices(np.arange(6), num_clients=3, alpha=0)
