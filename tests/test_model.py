import pytest
import torch
from frfl.model import EmbeddingClassifier

def test_model_output_shape():
    model = EmbeddingClassifier(input_dim=16, num_classes=4, hidden_dim=12)
    assert model(torch.randn(5, 16)).shape == (5, 4)

def test_model_rejects_non_2d_features():
    model = EmbeddingClassifier(input_dim=16, num_classes=4)
    with pytest.raises(ValueError): model(torch.randn(16))

def test_model_rejects_single_class():
    with pytest.raises(ValueError): EmbeddingClassifier(input_dim=8, num_classes=1)
