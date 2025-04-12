import numpy as np
import pandas as pd
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from simulator import Simulator


# to test, enter "pytest test_simulator.py" into the terminal

def generate_mock_data(num_assets=5, num_timesteps=2000, seed=42):
    np.random.seed(seed)
    times = pd.date_range(start='2022-01-01', periods=num_timesteps, freq='T')
    data_dict = {}
    for i in range(num_assets):
        price_series = 100 + np.cumsum(np.random.normal(0, 0.05, size=num_timesteps))
        df = pd.DataFrame({'midprice': price_series}, index=times)
        data_dict[f"ASSET_{i}"] = df
    return data_dict

def test_standardized_returns_shape():
    data_dict = generate_mock_data()
    sim = Simulator(data_dict)
    standardized = sim.compute_rolling_standardized_returns(sim.log_return_df, window=60)
    assert standardized.shape[0] > 0
    assert standardized.shape[1] == len(sim.asset_names)

def test_q_matrix_shape_and_validity():
    data_dict = generate_mock_data()
    sim = Simulator(data_dict)
    st_ret = sim.compute_rolling_standardized_returns(sim.log_return_df, 60)
    Y = st_ret.T.values
    t = 100
    Q = sim.compute_Q_matrix(Y, window=60, t=t, k=3)
    assert Q.shape == (len(sim.asset_names), 3)
    assert not np.any(np.isnan(Q))
    assert not np.any(np.isinf(Q))

def test_ou_parameter_estimation():
    # Simulate an AR(1) process
    np.random.seed(0)
    T = 100
    true_kappa = 0.2
    m = 0.0
    sigma = 1.0
    dt = 1
    X = [0]
    for _ in range(T-1):
        X.append(X[-1] + true_kappa * (m - X[-1]) * dt + sigma * np.random.normal())
    X = np.array(X)

    kappa, m_hat, sigma_hat, sigma_eq = Simulator.fit_ou_parameters(X)
    assert 0 < kappa < 1
    assert sigma_hat > 0
    assert sigma_eq > 0

def test_simulator_runs():
    data_dict = generate_mock_data(num_assets=4, num_timesteps=1600)
    sim = Simulator(data_dict, ret_window=60, pca_window=60, ou_window=60, k=2)
    sim.run_simulation()
    assert len(sim.pnl_series) > 0
    assert len(sim.equity_curve) == len(sim.pnl_series)
    assert sim.equity_curve[-1] > 0  # Equity should stay positive
