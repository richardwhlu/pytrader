"""Metrics are the optimization target they're reported as. Lock in the math."""
from __future__ import annotations

import numpy as np
import pandas as pd

import pytest

from pytrader import metrics as M


def test_total_return_flat_is_zero():
    eq = pd.Series([1.0] * 10)
    assert M.total_return(eq) == 0.0


def test_total_return_compounds():
    eq = pd.Series([1.0, 1.05, 1.1025])
    assert M.total_return(eq) == pytest.approx(0.1025, rel=1e-9)


def test_sharpe_zero_vol_is_zero():
    assert M.sharpe(np.zeros(10)) == 0.0


def test_sharpe_positive_with_positive_returns():
    r = np.full(100, 0.001)
    assert M.sharpe(r) > 0


def test_max_drawdown_of_monotonic_is_zero():
    eq = np.cumprod(np.concatenate([[1.0], np.ones(50) * 1.001]))
    assert M.max_drawdown(eq) == 0.0


def test_max_drawdown_known_value():
    eq = np.array([1.0, 1.2, 0.8, 1.0])
    # peak 1.2, trough 0.8 → dd = 0.8/1.2 - 1
    assert abs(M.max_drawdown(eq) - (0.8 / 1.2 - 1.0)) < 1e-12


def test_hit_rate_all_positive():
    assert M.hit_rate(np.array([0.01, 0.02])) == 1.0
    assert M.hit_rate(np.array([-0.01, 0.02])) == 0.5


def test_information_ratio_identical_is_zero():
    r = np.array([0.01, -0.02, 0.01, 0.03])
    assert M.information_ratio(r, r) == 0.0


def test_beta_alpha_with_identity():
    s = np.array([0.01, -0.02, 0.01, 0.03])
    b, a = M.beta_alpha(s, s)
    assert abs(b - 1.0) < 1e-9
    assert abs(a) < 1e-9


def test_summarize_shape_and_keys():
    eq = pd.Series(np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, 100)),
                   index=pd.bdate_range("2024-01-01", periods=100))
    bench = pd.Series(np.cumprod(1 + np.random.default_rng(1).normal(0, 0.01, 100)),
                      index=eq.index)
    out = M.summarize(eq, eq.index, bench)
    assert "metrics" in out and "benchmark" in out
    for k in ("sharpe", "cagr", "max_drawdown", "total_return", "ann_vol",
              "sortino", "hit_rate", "n_days"):
        assert k in out["metrics"]
    for k in ("total_return", "cagr", "information_ratio", "beta", "alpha"):
        assert k in out["benchmark"]


def test_cagr_monotonic_around_total_return():
    # 100 days = ~0.4 years; 10% total return → cagr ~ 25%
    eq = np.linspace(1.0, 1.10, 100)
    cagr = M.cagr(eq)
    assert 0.18 < cagr < 0.32
