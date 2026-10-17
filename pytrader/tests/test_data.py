"""Data-layer contract: DailyPrice canonicalization + the PriceStore ABC."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pytrader.data import DataUnavailable, DailyPrice
from pytrader.testing import InMemoryStore


def test_naive_daily_stamps_pass_through():
    idx = pd.bdate_range("2024-01-01", periods=5)
    s = DailyPrice("T", idx, pd.Series([1, 2, 3, 4, 5], index=idx))
    assert s.index.dtype.kind == "M"
    assert s.close.index.equals(s.index)
    assert not s.index.has_duplicates


def test_tz_aware_daily_stamps_become_naive():
    """yfinance returns exchange-local tz-aware stamps; we must normalize."""
    idx = pd.bdate_range("2024-01-01", periods=7, tz="America/New_York")
    assert idx.tz is not None
    vals = np.arange(1, len(idx) + 1, dtype="float64")
    s = DailyPrice("T", idx, pd.Series(vals, index=idx))
    assert s.index.dtype.kind == "M"
    assert getattr(s.index, "tz", None) is None
    assert s.close.index.equals(s.index)
    assert len(s) == len(idx)


def test_intraday_stamps_snap_to_trading_day():
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02 16:00"), pd.Timestamp("2024-01-03 16:00")]
    )
    s = DailyPrice("T", idx, pd.Series([10.0, 11.0], index=idx))
    assert s.index[0] == pd.Timestamp("2024-01-02 00:00")
    assert s.index[1] == pd.Timestamp("2024-01-03 00:00")


def test_duplicate_days_collapsed_keeps_first():
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02 09:00"),
         pd.Timestamp("2024-01-02 16:00"),
         pd.Timestamp("2024-01-03 16:00")]
    )
    s = DailyPrice("T", idx, pd.Series([1, 2, 3], index=idx))
    assert len(s) == 2
    assert s.close.iloc[0] == 1  # first of the duplicate day kept
    assert len(s.index) == len(s.close)


def test_close_and_index_stay_aligned():
    idx = pd.bdate_range("2024-01-01", periods=10)
    vals = np.random.default_rng(0).normal(100, 1, 10)
    s = DailyPrice("T", idx, pd.Series(vals, index=idx))
    assert list(s.close.index) == list(s.index)
    assert np.allclose(s.close.to_numpy(), vals)


def test_daily_returns_length():
    idx = pd.bdate_range("2024-01-01", periods=10)
    s = DailyPrice("T", idx, pd.Series(np.linspace(100, 110, 10), index=idx))
    r = s.daily_returns()
    assert len(r) == 9  # one fewer than observations


def test_store_fetch_missing_ticker_raises():
    idx = pd.bdate_range("2024-01-01", periods=5)
    store = InMemoryStore({"AAA": DailyPrice("AAA", idx, pd.Series([1, 2, 3, 4, 5], index=idx))})
    with pytest.raises(DataUnavailable):
        store.fetch(["ZZZ"], pd.Timestamp("2024-01-01").date())
