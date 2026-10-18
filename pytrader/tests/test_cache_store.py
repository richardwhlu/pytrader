"""CachedPriceStore: serve from cache, hit upstream only for misses.

`refresh_after_days=None` is the default — once a ticker is in the cache
we never hit upstream again. This is the rate-limit-safe behavior.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import pytest

from pytrader.data import CachedPriceStore, DailyPrice, DataUnavailable, PriceStore
from pytrader.storage import DuckDBStore


class _Fake(PriceStore):
    """Counts fetch calls; returns a fixed series."""

    def __init__(self, series: dict[str, pd.Series]) -> None:
        self.calls = 0
        self.series = {t.upper(): s for t, s in series.items()}

    def fetch(self, tickers: list[str], start: datetime.date) -> list[DailyPrice]:
        self.calls += 1
        out = []
        for t in tickers:
            s = self.series.get(t.upper())
            if s is None:
                raise DataUnavailable(t)
            mask = s.index >= pd.Timestamp(start)
            out.append(DailyPrice(t.upper(), s.index[mask], s[mask]))
        return out

    def metadata(self, tickers: list[str]) -> dict[str, dict[str, str]]:
        return {t.upper(): {"name": t, "sector": "Fake"} for t in tickers}


@pytest.fixture
def db(tmp_path: Path) -> DuckDBStore:
    return DuckDBStore(tmp_path / "cache.duckdb")


def _five_days() -> pd.Series:
    idx = pd.bdate_range("2024-01-01", periods=5)
    return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx, dtype="float64")


def test_first_call_hits_upstream_and_caches(db):
    upstream = _Fake({"AAA": _five_days()})
    store = CachedPriceStore(upstream, cache=db)
    out1 = store.fetch(["AAA"], start=datetime.date(2024, 1, 1))
    assert upstream.calls == 1, "first call must hit upstream exactly once"
    assert out1[0].close.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert db.end_date("AAA") == pd.Timestamp("2024-01-05").normalize()


def test_second_call_is_served_from_cache_no_upstream(db):
    upstream = _Fake({"AAA": _five_days()})
    store = CachedPriceStore(upstream, cache=db)
    store.fetch(["AAA"], start=datetime.date(2024, 1, 1))
    calls_after_first = upstream.calls
    # Second call: cache hit — upstream should NOT be hit.
    out2 = store.fetch(["AAA"], start=datetime.date(2024, 1, 1))
    assert upstream.calls == calls_after_first, "second call must not hit upstream"
    assert out2[0].close.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_start_date_slices_cached(db):
    upstream = _Fake({"AAA": _five_days()})
    store = CachedPriceStore(upstream, cache=db)
    store.fetch(["AAA"], start=datetime.date(2024, 1, 1))
    # Requesting a later start date must slice the cached series.
    out = store.fetch(["AAA"], start=datetime.date(2024, 1, 3))
    assert out[0].close.tolist() == [3.0, 4.0, 5.0]
    assert upstream.calls == 1


def test_multiple_tickers_share_cache(db):
    upstream = _Fake({"AAA": _five_days(), "BBB": _five_days() * 2})
    store = CachedPriceStore(upstream, cache=db)
    store.fetch(["AAA", "BBB"], start=datetime.date(2024, 1, 1))
    assert db.end_date("AAA") is not None
    assert db.end_date("BBB") is not None
    # Now both are cached: second call must not hit upstream for either.
    calls_after_first = upstream.calls
    store.fetch(["AAA", "BBB"], start=datetime.date(2024, 1, 1))
    assert upstream.calls == calls_after_first


def test_metadata_is_cached_after_first_call(db):
    class _MetaCounter(_Fake):
        def __init__(self):
            super().__init__({"AAA": _five_days()})
            self.meta_calls = 0
        def metadata(self, tickers):
            self.meta_calls += 1
            return super().metadata(tickers)

    upstream = _MetaCounter()
    store = CachedPriceStore(upstream, cache=db)
    m1 = store.metadata(["AAA"])
    assert m1["AAA"]["name"] == "AAA"
    assert upstream.meta_calls == 1
    m2 = store.metadata(["AAA"])
    assert m2["AAA"]["name"] == "AAA"
    assert upstream.meta_calls == 1, "second metadata call must be a cache hit"


def test_refresh_after_days_triggers_upstream_when_stale(db):
    upstream = _Fake({"AAA": _five_days()})
    store = CachedPriceStore(upstream, cache=db, refresh_after_days=30)
    base_calls = upstream.calls
    # First call: misses → hits upstream.
    store.fetch(["AAA"], start=datetime.date(2024, 1, 1))
    # Our fake data ends 2024-01-05; today (2026-10-17) is way more than
    # 30 days out, so the stale branch fires on the next call.
    store.fetch(["AAA"], start=datetime.date(2024, 1, 1))
    assert upstream.calls > base_calls, "stale refresh must hit upstream"
