"""FedAvg baseline and L2 parameter-drift aggregation."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import math
import torch
State = Mapping[str, torch.Tensor]

def _validate(states: Sequence[State]) -> list[str]:
    if not states: raise ValueError("at least one client state is required")
    keys = list(states[0])
    for i, state in enumerate(states):
        if set(state) != set(keys): raise ValueError(f"client state {i} has different keys")
        for key in keys:
            if state[key].shape != states[0][key].shape: raise ValueError(f"shape mismatch for {key}")
    return keys

def l2_parameter_drift(reference: State, client_state: State) -> float:
    """Euclidean distance between all matching state tensors."""
    if set(reference) != set(client_state): raise ValueError("reference and client keys differ")
    total = 0.0
    for key in reference:
        if reference[key].shape != client_state[key].shape: raise ValueError(f"shape mismatch for {key}")
        delta = reference[key].detach().double() - client_state[key].detach().double()
        total += float(torch.sum(delta * delta).item())
    return math.sqrt(total)

def sample_weighted_average(client_states: Sequence[State], sample_counts: Sequence[int]) -> dict[str, torch.Tensor]:
    """FedAvg aggregation weighted by number of local samples."""
    keys = _validate(client_states)
    if len(client_states) != len(sample_counts): raise ValueError("state/count length mismatch")
    if any(n <= 0 for n in sample_counts): raise ValueError("sample counts must be positive")
    total = float(sum(sample_counts)); result = {}
    for key in keys:
        acc = torch.zeros_like(client_states[0][key])
        for state, count in zip(client_states, sample_counts): acc = acc + state[key].to(acc.dtype) * (count / total)
        result[key] = acc
    return result

def drift_weights(reference: State, client_states: Sequence[State], temperature: float = 1.0) -> list[float]:
    """Softmax of negative normalized L2 drift; heuristic, not accuracy-aware."""
    _validate(client_states)
    if temperature < 0: raise ValueError("temperature must be non-negative")
    drifts = [l2_parameter_drift(reference, state) for state in client_states]
    if len(drifts) == 1: return [1.0]
    scale = max(drifts) or 1.0
    logits = [-temperature * d / scale for d in drifts]
    peak = max(logits); exp_values = [math.exp(v - peak) for v in logits]; denom = sum(exp_values)
    return [v / denom for v in exp_values]

def drift_weighted_average(reference: State, client_states: Sequence[State], temperature: float = 1.0) -> tuple[dict[str, torch.Tensor], list[float]]:
    """Aggregate states with normalized coefficients derived from parameter drift."""
    keys = _validate(client_states); weights = drift_weights(reference, client_states, temperature); result = {}
    for key in keys:
        acc = torch.zeros_like(client_states[0][key])
        for state, weight in zip(client_states, weights): acc = acc + state[key].to(acc.dtype) * weight
        result[key] = acc
    return result, weights
