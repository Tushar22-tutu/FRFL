"""Local client update with FedProx regularization."""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def train_local_model(global_model: nn.Module, X: np.ndarray, y: np.ndarray, *, epochs: int = 2, batch_size: int = 32, learning_rate: float = 1e-3, mu: float = 0.01, seed: int = 42) -> tuple[dict[str, torch.Tensor], int, float]:
    """Train a local copy; return state, row count and mean minibatch loss."""
    if len(X) == 0 or len(X) != len(y): raise ValueError("X and y must have equal non-zero lengths")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or mu < 0: raise ValueError("invalid local training hyperparameters")
    torch.manual_seed(seed); local = copy.deepcopy(global_model); local.train()
    global_state = {k: v.detach().clone() for k, v in global_model.state_dict().items()}
    reference = {n: p.detach().clone() for n, p in global_model.named_parameters()}
    data = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))
    loader = DataLoader(data, batch_size=min(batch_size, len(data)), shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(local.parameters(), lr=learning_rate); criterion = nn.CrossEntropyLoss(); loss_total = 0.0; batches = 0
    for _ in range(epochs):
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True); loss = criterion(local(features), labels)
            if mu:
                penalty = torch.zeros((), dtype=loss.dtype)
                for name, param in local.named_parameters(): penalty = penalty + torch.sum((param - reference[name]) ** 2)
                loss = loss + (mu / 2.0) * penalty
            loss.backward(); optimizer.step(); loss_total += float(loss.detach().item()); batches += 1
    state = {k: v.detach().clone() for k, v in local.state_dict().items()}
    if any(not torch.equal(value, global_state[key]) for key, value in global_model.state_dict().items()): raise RuntimeError("local training mutated global model")
    return state, len(data), loss_total / max(1, batches)
