# tests/test_portfolio_optimizer.py
import pytest
import numpy as np
from src.portfolio_optimizer import optimize_portfolio

def test_portfolio_optimization():
    expected_returns = np.array([0.1, 0.15])
    covariance = np.eye(2)  # Uncorrelated assets
    weights = optimize_portfolio(expected_returns, covariance)
    assert np.isclose(weights.sum(), 1.0), "Portfolio must be fully invested."