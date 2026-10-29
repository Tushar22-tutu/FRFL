import pytest
import torch
from frfl.aggregation import drift_weighted_average, drift_weights, l2_parameter_drift, sample_weighted_average

def state(v): return {"weight": torch.tensor([float(v), float(v)])}

def test_same_state_has_zero_drift(): assert l2_parameter_drift(state(1), state(1)) == pytest.approx(0)

def test_fedavg_sample_weighting():
    result = sample_weighted_average([state(0), state(10)], [1, 3])
    assert result["weight"].tolist() == pytest.approx([7.5, 7.5])

def test_drift_weights_sum_to_one(): assert sum(drift_weights(state(0), [state(1), state(2), state(3)])) == pytest.approx(1)

def test_lower_drift_gets_higher_weight():
    weights = drift_weights(state(0), [state(0.1), state(3.0)])
    assert weights[0] > weights[1]

def test_drift_aggregate_returns_weights():
    aggregate, weights = drift_weighted_average(state(0), [state(1), state(2)])
    assert aggregate["weight"].shape == (2,)
    assert sum(weights) == pytest.approx(1)

def test_invalid_fedavg_counts_raise():
    with pytest.raises(ValueError): sample_weighted_average([state(1)], [0])

def test_empty_client_list_raises():
    with pytest.raises(ValueError): sample_weighted_average([], [])
