"""Custom metric registry — developers can attach new metrics to every run."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pytrader.metrics as M


@pytest.fixture(autouse=True)
def _clear():
    M.clear_custom_metrics()
    yield
    M.clear_custom_metrics()


def test_register_direct_is_reported():
    M.register_metric("max_loss", lambda equity, dates, periods:
                      min((p.net_return for p in periods), default=0.0))
    class P:
        net_return = -0.03
    eq = pd.Series([1.0, 1.1, 1.05])
    out = M.summarize(eq, eq.index, None, periods=[P(), P()])
    assert out["metrics"]["max_loss"] == pytest.approx(-0.03)


def test_register_as_decorator_is_reported():
    @M.register_metric("n_winners")
    def winner(equity, dates, periods):
        return float(sum(p.net_return > 0 for p in periods))
    class P:
        net_return = 0.05
    out = M.summarize(pd.Series([1.0, 1.1]), None, None, periods=[P()])
    assert out["metrics"]["n_winners"] == 1.0


def test_custom_overrides_and_extra():
    # caller-supplied custom values merge in and override registered ones
    M.register_metric("x", lambda *a: 1.0)
    out = M.summarize(pd.Series([1.0, 2.0]), None, None, custom={"x": 99.0, "y": 7.0})
    assert out["metrics"]["x"] == 99.0
    assert out["metrics"]["y"] == 7.0


def test_broken_custom_metric_does_not_break_run():
    M.register_metric("boom", lambda *a: 1 / 0)
    out = M.summarize(pd.Series([1.0, 1.0, 1.0]))
    assert "sharpe" in out["metrics"]
    assert "boom" not in out["metrics"]


def test_clear_custom_metrics():
    M.register_metric("tmp", lambda *a: 1.0)
    assert "tmp" in M.summarize(pd.Series([1.0, 1.0]))["metrics"]
    M.clear_custom_metrics()
    assert "tmp" not in M.summarize(pd.Series([1.0, 1.0]))["metrics"]