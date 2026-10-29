import numpy as np
import pytest
import torch
from frfl.metrics import classification_metrics
from frfl.model import EmbeddingClassifier
from frfl.trainer import train_local_model

def test_perfect_predictions_have_perfect_metrics():
    m = classification_metrics(np.array([0, 1, 1]), np.array([0, 1, 1]))
    assert m["accuracy"] == pytest.approx(1) and m["f1_macro"] == pytest.approx(1)

def test_metrics_reject_empty_arrays():
    with pytest.raises(ValueError): classification_metrics(np.array([]), np.array([]))

def test_fedprox_client_training_does_not_mutate_global_model():
    torch.manual_seed(2); model = EmbeddingClassifier(input_dim=4, num_classes=2, hidden_dim=8)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    X = np.array([[1., 0., 0., 1.], [0., 1., 1., 0.], [1., 1., 0., 0.], [0., 0., 1., 1.]], dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    state, count, loss = train_local_model(model, X, y, epochs=1, batch_size=2, mu=0.01, seed=3)
    assert count == len(X) and loss >= 0 and set(state) == set(before)
    assert all(torch.equal(model.state_dict()[key], value) for key, value in before.items())
