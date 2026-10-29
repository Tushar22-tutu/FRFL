"""MLP classifier for fixed face embeddings."""
from __future__ import annotations
import torch
from torch import nn

class EmbeddingClassifier(nn.Module):
    """Classify precomputed ArcFace-style embeddings."""
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        if input_dim <= 0: raise ValueError("input_dim must be positive")
        if num_classes < 2: raise ValueError("num_classes must be at least 2")
        if hidden_dim <= 0: raise ValueError("hidden_dim must be positive")
        middle = max(hidden_dim // 2, 8)
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, middle), nn.ReLU(), nn.Linear(middle, num_classes))
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2: raise ValueError("features must be 2-D: (batch, embedding_dim)")
        return self.network(features)
