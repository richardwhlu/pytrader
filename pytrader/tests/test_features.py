"""Features: factor correctness + panel alignment. These guard against
lookahead (labels must use future prices only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pytrader import features as F
from pytrader.data import DailyPrice


def _close(idx, vals) -> DailyPrice:
    return DailyPrice("T", idx, pd.Series(vals, index=idx, dtype="float64"))


@pytest.fixture
def close100() -> pd.Series:
    idx = pd.bdate_range("2024-01-01", periods=100)
    vals = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 100)))
    return pd.Series(vals, index=idx, dtype="float64")


def test_feature_columns_are_deterministic():
    spec = F.FeatureSpec()
    assert spec.column_names() == [
        "momentum_21", "momentum_63", "volatility_21", "rsi_14", "trend_21"
    ]


def test_momentum_is_trailing_log_return(close100):
    m = F.momentum(close100, window=5)
    expected = np.log(close100 / close100.shift(5))
    pd.testing.assert_series_equal(m.iloc[5:], expected.iloc[5:], check_names=False)


def test_momentum_first_window_is_nan(close100):
    assert close100.iloc[:5].index.equals(F.momentum(close100, 5).iloc[:5].index)
    assert F.momentum(close100, 5).iloc[:5].isna().all()
    assert F.momentum(close100, 5).iloc[5:].notna().all()


def test_volatility_nonnegative(close100):
    v = F.volatility(close100, 10).dropna()
    assert (v >= 0).all()


def test_rsi_bounded_0_100(close100):
    r = F.rsi(close100, 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_future_return_is_forward(close100):
    """Label for day t must equal log(P_{t+h}/P_t): uses FUTURE prices only."""
    h = 5
    label = F.future_return(close100, h)
    at = 10
    expected = np.log(close100.iloc[at + h] / close100.iloc[at])
    assert abs(label.iloc[at] - expected) < 1e-9
    # last h rows must be NaN (no future to predict)
    assert label.iloc[-h:].isna().all()


def test_panel_drops_labels_and_keeps_features(price_map):
    s = list(price_map.values())
    spec = F.FeatureSpec()
    X, y = F.compute_panel(s, spec, horizon=5, benchmark="SPY")
    for c in spec.column_names():
        assert c in X.columns
    assert "ticker" in X.columns
    # y aligned to X index
    assert len(y) == len(X)
    # no NaNs in features
    assert X[spec.column_names()].notna().all().all()
    # labels exist for most rows
    assert y.notna().sum() > 0


def test_panel_excludes_benchmark_rows(price_map):
    s = list(price_map.values())
    spec = F.FeatureSpec()
    X, _ = F.compute_panel(s, spec, horizon=5, benchmark="SPY")
    assert "SPY" not in set(X["ticker"])


def test_panel_empty_raises(price_map):
    # too-short series
    series = [DailyPrice("T", pd.bdate_range("2024-01-01", periods=3),
                         pd.Series([1, 2, 3], dtype="float64"))]
    with pytest.raises(Exception):
        F.compute_panel(series, F.FeatureSpec(), horizon=5)
