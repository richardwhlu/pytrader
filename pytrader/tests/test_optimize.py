"""Hermetic tests for the mean-variance basket optimizer.

All data is deterministic synthetic closes — no DB, no network, no yfinance —
so these run offline and exercise the real solver (scipy SLSQP) over controlled
windows. The properties under test are the ones the user cares about:

  * λ = 0            → 100% in the single best-mover (pure gain).
  * λ > 0            → spreads across names → lower volatility, more positions.
  * weights          → always non-negative, sum to 1, ≤ 1 each.
  * efficient_frontier → a smooth gain↔vol "distribution" from high-return to
                         min-variance.
  * degenerate windows → raise a clear error instead of crashing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pytrader import optimize as opt


def _window(seed: int = 1, n: int = 60) -> pd.DataFrame:
    """Deterministic, well-behaved price panel with ordered drifts.

    A is the clear best-mover (high drift); B is medium drift but volatile;
    C is a low-drift, low-vol "ballast". A/B/C share the full window (no NaNs).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    a = 100.0 * np.exp(np.cumsum(rng.normal(0.0035, 0.012, n)))
    b = 100.0 * np.exp(np.cumsum(rng.normal(0.0012, 0.035, n)))
    c = 100.0 * np.exp(np.cumsum(rng.normal(0.0008, 0.006, n)))
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)


# ---------------------------------------------------------------------------
# daily_returns
# ---------------------------------------------------------------------------

def test_daily_returns_inner_joins_and_drops_constant():
    idx = pd.bdate_range("2024-01-02", periods=10)
    raw = pd.DataFrame({
        "A": np.linspace(100, 110, 10),
        "B": np.full(10, 50.0),  # constant → dropped
        "C": np.linspace(10, 12, 10),
    }, index=idx)
    rets = opt.daily_returns(raw)
    assert list(rets.columns) == ["A", "C"]  # B dropped as flat
    assert rets.shape == (9, 2)


def test_daily_returns_inner_join_on_missing_dates():
    idx = pd.bdate_range("2024-01-02", periods=8)
    raw = pd.DataFrame({"A": np.linspace(10, 11, 8),
                        "B": np.linspace(5, 6, 8)}, index=idx)
    # B has a mid-window gap that A does not → that date is dropped for all.
    raw.loc[idx[3], "B"] = np.nan
    rets = opt.daily_returns(raw)
    assert idx[3] not in rets.index
    assert rets.notna().all().all()


@pytest.mark.parametrize("n", [1, 2])
def test_daily_returns_short_window_raises(n):
    idx = pd.bdate_range("2024-01-02", periods=n)
    raw = pd.DataFrame({"A": [1.0 + i for i in range(n)]}, index=idx)
    with pytest.raises(ValueError):
        opt.daily_returns(raw)


def test_daily_returns_all_constant_raises():
    idx = pd.bdate_range("2024-01-02", periods=6)
    raw = pd.DataFrame({"A": np.full(6, 42.0)}, index=idx)
    with pytest.raises(ValueError, match="flat"):
        opt.daily_returns(raw)


# ---------------------------------------------------------------------------
# optimize_window
# ---------------------------------------------------------------------------

def test_pure_gain_concentrates_on_top_mover():
    closes = _window()
    res = opt.optimize_window(closes, risk_aversion=0.0)
    # The "pure gain" extreme: the entire basket sits in one name.
    assert res.n_positions == 1
    winner = max(res.weights, key=res.weights.get)
    assert res.weights[winner] == pytest.approx(1.0, abs=1e-4)
    # And that name is the best total-return mover over the shared window.
    tot = (closes.iloc[-1] / closes.iloc[0] - 1.0).to_dict()
    assert winner == max(tot, key=tot.get)


def test_weights_are_feasible():
    closes = _window(seed=2)
    for lam in (0.0, 0.1, 1.0, 1e3):
        res = opt.optimize_window(closes, risk_aversion=lam)
        w = res.weights
        assert len(w) == 3
        assert all(v >= -1e-9 for v in w.values())
        assert all(v <= 1.0 + 1e-9 for v in w.values())
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-5)


def test_higher_lam_smooths_and_spreads():
    """More risk aversion → lower vol (smoothed) and more spread-out weights."""
    closes = _window(seed=3)
    lo = opt.optimize_window(closes, risk_aversion=0.0)
    hi = opt.optimize_window(closes, risk_aversion=1e3)
    # A fully concentrated basket (λ=0) is the high-vol corner; the min-var
    # basket (large λ) is smoother.
    assert hi.ann_vol < lo.ann_vol
    assert hi.n_positions >= lo.n_positions


def test_single_ticker_is_fully_weighted():
    idx = pd.bdate_range("2024-01-02", periods=20)
    closes = pd.DataFrame({"A": np.linspace(100, 130, 20)}, index=idx)
    res = opt.optimize_window(closes, risk_aversion=1.0)
    assert res.weights["A"] == pytest.approx(1.0)
    assert res.n_positions == 1


def test_result_stats_are_sane():
    closes = _window(seed=4)
    res = opt.optimize_window(closes, risk_aversion=10.0)
    assert res.n_days > 0
    assert res.ann_vol >= 0
    assert res.max_drawdown <= 0          # a drawdown is never positive
    d = res.to_dict()
    assert set(d) >= {"risk_aversion", "weights", "stats"}
    assert d["stats"]["n_days"] == res.n_days


def test_too_short_window_raises():
    # 2 closes = 1 return row — below the minimum needed to compute a
    # covariance.
    idx = pd.bdate_range("2024-01-02", periods=2)
    closes = pd.DataFrame({"A": [100, 101], "B": [50, 51]}, index=idx)
    with pytest.raises(ValueError):
        opt.optimize_window(closes, risk_aversion=1.0)


def test_negative_risk_aversion_rejected():
    closes = _window()
    with pytest.raises(ValueError, match="risk_aversion"):
        opt.optimize_window(closes, risk_aversion=-1.0)


# ---------------------------------------------------------------------------
# efficient_frontier (the "distribution")
# ---------------------------------------------------------------------------

def test_frontier_spans_gain_to_variance():
    closes = _window(seed=5)
    frontier = opt.efficient_frontier(closes, steps=8)
    points = frontier["points"]
    assert len(points) == 8
    # First point is λ=0 → the pure-gain corner: fully concentrated.
    assert points[0]["risk_aversion"] == 0.0
    assert points[0]["stats"]["n_positions"] == 1
    # Last point is the min-var corner → more spread out and lower vol.
    assert points[-1]["stats"]["ann_vol"] < points[0]["stats"]["ann_vol"]
    assert points[-1]["stats"]["n_positions"] >= 2


def test_frontier_points_are_feasible():
    closes = _window(seed=6)
    for p in opt.efficient_frontier(closes, steps=6)["points"]:
        assert sum(p["weights"].values()) == pytest.approx(1.0, abs=1e-5)
        assert all(v >= -1e-9 for v in p["weights"].values())


def test_frontier_steps_minimum():
    # steps < 2 is clamped to exactly 2 (the two corners): 0 and λ_max.
    assert len(opt.efficient_frontier(_window(seed=7), steps=1)["points"]) == 2
    assert len(opt.efficient_frontier(_window(seed=7), steps=0)["points"]) == 2
