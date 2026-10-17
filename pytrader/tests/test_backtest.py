"""Backtest simulator: the heart of the platform. Lock in correctness of
costing, turnover, benchmark alignment, and the decision-window rule."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pytrader.backtest import (
    BacktestConfig,
    _topn_weights,
    _turnover,
    predictions_from_panel,
    simulate,
)
from pytrader.data import DailyPrice


def _mkclose(n=60, start="2024-01-01", drift=0.0) -> tuple[pd.DatetimeIndex, pd.Series]:
    idx = pd.bdate_range(start, periods=n)
    vals = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(drift, 0.005, n)))
    return idx, pd.Series(vals, index=idx, dtype="float64")


def _series(idx, close, ticker) -> DailyPrice:
    return DailyPrice(ticker, idx, close)


def test_topn_weights_equal_weight():
    pred = np.array([0.5, -0.2, 1.0, 0.0])
    w = _topn_weights(pred, top_n=2, col_idx={})
    # picks indices 2 (1.0) and 0 (0.5), each 1/2
    assert abs(w[0] - 0.5) < 1e-12
    assert abs(w[2] - 0.5) < 1e-12
    assert w[1] == 0
    assert abs(w.sum() - 1.0) < 1e-12


def test_topn_weights_handles_all_nan():
    pred = np.full(3, np.nan)
    w = _topn_weights(pred, top_n=2, col_idx={})
    assert abs(w.sum() - 1.0) < 1e-12  # uniform fallback


def test_turnover_first_period_is_full():
    assert _turnover(None, np.array([0.5, 0.5, 0.0])) == 1.0


def test_turnover_no_change_is_zero():
    w = np.array([0.5, 0.5, 0.0])
    assert _turnover(w, w) == 0.0


def test_turnover_full_swap_is_one_sided():
    # Platform convention: turnover = fraction of portfolio replaced
    # (one-sided). A full swap replaces the whole book => 1.0.
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert _turnover(a, b) == pytest.approx(1.0)


def test_turnover_partial_swap():
    a = np.array([0.5, 0.5, 0.0])
    b = np.array([0.5, 0.0, 0.5])
    # keep 0.5, replace 0.5
    assert _turnover(a, b) == pytest.approx(0.5)


def test_simulate_empty_predictions_raises(price_map):
    preds = pd.DataFrame()
    with pytest.raises(ValueError, match="empty"):
        simulate(preds, list(price_map.values()), BacktestConfig())


def test_simulate_unknown_ticker_raises(price_map):
    idx = pd.bdate_range("2024-01-01", periods=30)
    preds = pd.DataFrame({"FAKE": np.zeros(30)}, index=idx)
    with pytest.raises(ValueError, match="unknown tickers"):
        simulate(preds, list(price_map.values()), BacktestConfig())


def test_simulate_equity_starts_at_one(price_map):
    """The reported curve must be normalized to start exactly at 1.0."""
    # constant-score predictions: every date picks the same top-3
    idx = pd.bdate_range("2024-06-01", periods=120)
    tickers = [t for t in price_map if t != "SPY"]
    scores = {t: float(0.1 * i) for i, t in enumerate(sorted(tickers))}
    preds = pd.DataFrame([{t: scores[t] for t in tickers}] * len(idx), index=idx)
    sim = simulate(preds, list(price_map.values()),
                   BacktestConfig(horizon=5, top_n=3), benchmark="SPY")
    assert abs(sim.equity.iloc[0] - 1.0) < 1e-9
    assert len(sim.equity) == len(sim.dates)
    # equity monotone structure: no negative values
    assert (sim.equity > 0).all()


def test_simulate_benchmark_aligned_to_equity(price_map):
    idx = pd.bdate_range("2024-06-01", periods=120)
    tickers = [t for t in price_map if t != "SPY"]
    preds = pd.DataFrame(
        [{t: float(0.05 * i) for i, t in enumerate(sorted(tickers))}] * len(idx),
        index=idx,
    )
    sim = simulate(preds, list(price_map.values()),
                   BacktestConfig(horizon=5, top_n=3), benchmark="SPY")
    assert sim.benchmark_equity is not None
    # benchmark index must be a subset of the equity index (no missing marks)
    for d in sim.benchmark_equity.index:
        assert d in sim.equity.index
    assert sim.benchmark_equity.notna().all()
    assert abs(sim.benchmark_equity.iloc[0] - 1.0) < 1e-6


def test_simulate_costs_reduce_return(price_map):
    """Higher costs must produce a lower-or-equal terminal equity."""
    idx = pd.bdate_range("2024-06-01", periods=150)
    tickers = [t for t in price_map if t != "SPY"]
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.02, size=(len(idx), len(tickers)))
    preds = pd.DataFrame(base, index=idx, columns=tickers)
    cfg_low = BacktestConfig(horizon=5, top_n=3, commission_bps=0, slippage_bps=0)
    cfg_high = BacktestConfig(horizon=5, top_n=3, commission_bps=20, slippage_bps=20)
    low = simulate(preds, list(price_map.values()), cfg_low, benchmark="SPY")
    high = simulate(preds, list(price_map.values()), cfg_high, benchmark="SPY")
    assert high.equity.iloc[-1] <= low.equity.iloc[-1] + 1e-9


def test_simulate_config_validation():
    with pytest.raises(ValueError):
        BacktestConfig(horizon=0)
    with pytest.raises(ValueError):
        BacktestConfig(top_n=0)
    with pytest.raises(ValueError):
        BacktestConfig(commission_bps=-1)


def test_predictions_from_panel_pivots_wide(price_map):
    from pytrader.features import FeatureSpec, compute_panel
    s = list(price_map.values())
    X, y = compute_panel(s, FeatureSpec(), horizon=5, benchmark="SPY")
    mask = y.notna()
    Xf = X[mask]
    scores = np.arange(len(Xf), dtype="float64")
    preds = predictions_from_panel(Xf, scores)
    # one row per date, one col per ticker
    assert preds.shape[0] == Xf["date"].nunique()
    assert set(preds.columns) == set(Xf["ticker"])
