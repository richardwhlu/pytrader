"""batch_download: chunked, rate-limited, idempotent universe load."""
from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import pytest

from pytrader.data import DailyPrice, DataUnavailable, PriceStore, batch_download
from pytrader.storage import DuckDBStore


class _Fake(PriceStore):
    def __init__(self, known: set[str]):
        self.known = {t.upper() for t in known}
        self.calls = 0

    def fetch(self, tickers, start):
        self.calls += 1
        out = []
        for t in tickers:
            if t.upper() not in self.known:
                raise DataUnavailable(t)
            idx = pd.bdate_range("2024-01-01", periods=5)
            out.append(DailyPrice(t.upper(), idx, pd.Series(range(5), index=idx, dtype="float64")))
        return out

    def metadata(self, tickers):
        return {t.upper(): {"name": t, "sector": "F"} for t in tickers}


@pytest.fixture
def db(tmp_path: Path):
    return DuckDBStore(tmp_path / "batch.duckdb")


def test_first_run_fetches_and_caches(db):
    up = _Fake(["A", "B", "C"])
    res = batch_download(up, db, ["A", "B", "C"],
                         datetime.date(2024, 1, 1),
                         batch_size=2, pause_seconds=0)
    assert res["n_fetched"] == 3
    assert res["n_failed"] == 0
    assert res["n_cached"] == 0
    assert {db.load("A").iloc[-1], db.load("B").iloc[-1], db.load("C").iloc[-1]} == {4.0}


def test_second_run_is_all_cache_hits(db):
    up = _Fake(["A", "B", "C"])
    batch_download(up, db, ["A", "B", "C"], datetime.date(2024, 1, 1),
                   batch_size=25, pause_seconds=0)
    calls_after_first = up.calls
    up2 = _Fake(["A", "B", "C"])
    res2 = batch_download(up2, db, ["A", "B", "C"], datetime.date(2024, 1, 1),
                          batch_size=25, pause_seconds=0)
    assert res2["n_cached"] == 3
    assert res2["n_fetched"] == 0
    assert up2.calls == 0, "second run must not hit upstream for cached tickers"


def test_overwrite_forces_refetch(db):
    up = _Fake(["A"])
    batch_download(up, db, ["A"], datetime.date(2024, 1, 1), pause_seconds=0)
    up2 = _Fake(["A"])
    _ = up2.calls
    res = batch_download(up2, db, ["A"], datetime.date(2024, 1, 1),
                         pause_seconds=0, overwrite=True)
    assert res["n_fetched"] == 1
    assert res["n_cached"] == 0


def test_failed_ticker_is_reported_not_raised(db):
    up = _Fake(["GOOD"])
    res = batch_download(up, db, ["GOOD", "BAD", "MISSING"],
                         datetime.date(2024, 1, 1), pause_seconds=0)
    assert res["n_fetched"] == 1
    assert set(res["failed"]) == {"BAD", "MISSING"}
    assert db.end_date("GOOD") is not None
    assert db.end_date("BAD") is None


def test_all_tickers_cached_regardless_of_batch_size(db):
    up = _Fake(["A", "B", "C", "D", "E"])
    res = batch_download(up, db, ["A", "B", "C", "D", "E"], datetime.date(2024, 1, 1),
                         batch_size=2, pause_seconds=0)
    assert res["n_fetched"] == 5
    assert res["n_failed"] == 0
    for t in ["A", "B", "C", "D", "E"]:
        assert db.end_date(t) is not None
