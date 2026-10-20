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


class _FailingUpstream(PriceStore):
    """Raises a *wrapped* exception: our wrapper from an inner KeyError.

    This is the shape of a real yfinance failure (e.g. rate-limit 429 or a
    missing symbol surfacing as a nested exception), and it is the case the
    user said they could not debug before — the failure list had the ticker
    but nothing to look at.
    """

    def __init__(self) -> None:
        self.calls = 0

    def fetch(self, tickers, start):
        self.calls += 1
        for t in tickers:
            raise DataUnavailable(f"no data for {t}: underlying boom") \
                from KeyError(f"symbol for {t.upper()}")

    def metadata(self, tickers):
        return {t.upper(): {"name": t, "sector": ""} for t in tickers}


def test_failures_carry_cause_chain_and_traceback(db):
    """A failed ticker must surface a debuggable ErrorDetail, not just a name.

    This is the exact scenario the fix is for: the previous CLI output was
    ``failed: SPY, NVDA`` with no stack to debug against. Now the summary
    carries a structured detail with the wrapped library exception — its
    type, message, cause chain, and full traceback.
    """
    from pytrader.diagnostics import ErrorDetail  # noqa: F401

    up = _FailingUpstream()
    res = batch_download(up, db, ["A"], datetime.date(2024, 1, 1), pause_seconds=0)

    # Backward-compatible shape still present:
    assert res["n_failed"] == 1
    assert res["failed"] == ["A"]

    # New: structured failures, one per failed ticker.
    assert len(res["failures"]) == 1, f"summary: {res}"
    d = res["failures"][0]
    assert isinstance(d, ErrorDetail)
    assert d.where == "A"
    assert d.type == "DataUnavailable"

    # The *wrapped* library exception is preserved in cause_chain:
    assert "KeyError" in d.cause_chain, f"expected KeyError in {d.cause_chain!r}"

    # And the traceback is real — it must show BOTH the inner library frame
    # (KeyError) and our wrapper (DataUnavailable), linked by the standard
    # "was the direct cause of" bridge. That is the "full stacktrace" the
    # user was missing before.
    assert d.traceback
    assert "KeyError" in d.traceback
    assert "DataUnavailable" in d.traceback
    assert "no data for A" in d.message
    assert "direct cause" in d.traceback or "Traceback" in d.traceback

    # `format()` renders something a human can read, including the cause.
    rendered = d.format()
    assert "KeyError" in rendered
    assert "DataUnavailable" in rendered


class _BoundedFake(PriceStore):
    """Upstream that only returns closes since its own earliest date.

    Simulates yfinance's real behavior: it returns whatever is in the
    provider's history. We don't care about the exact numbers, only that
    the code path that "fills a front gap" runs and lands more rows.
    """

    def __init__(self, series: dict[str, pd.Series]):
        self.series = series
        self.calls = 0
        self.starts: list[pd.Timestamp] = []

    def fetch(self, tickers, start):
        self.calls += 1
        out = []
        s_start = pd.Timestamp(start)
        for t in tickers:
            s = self.series.get(t.upper())
            if s is None:
                raise DataUnavailable(t)
            self.starts.append(s_start)
            out.append(DailyPrice(t.upper(), s.index, s))
        return out

    def metadata(self, tickers):
        return {t.upper(): {"name": t, "sector": ""} for t in tickers}


def test_front_gap_gets_filled(db):
    """If we have 1yr cached and the user asks for 5yr,
    batch_download must refetch the wider window and land more history."""
    full = pd.Series(
        range(500),  # 500 trading days
        index=pd.bdate_range("2022-01-01", periods=500),
        dtype="float64",
    )
    # Simulate "we have 1yr" — save only the last 250 rows.
    tail = full.iloc[-250:]
    db.save("AAA", tail)
    assert db.start_date("AAA") > full.index[0], \
        "precondition: cached data must start AFTER the full data's first day"

    # --- Case 1: user asks for a start LATER than what's cached → skip ---
    later_than_cached = tail.index[10].date()  # safely inside cached range
    up = _BoundedFake({"AAA": full})
    res = batch_download(up, db, ["AAA"], later_than_cached,
                         batch_size=10, pause_seconds=0)
    assert "AAA" in res["cached_skip"]
    assert res["n_fetched"] == 0
    assert up.calls == 0  # no upstream call at all

    # --- Case 2: user asks for a start EARLIER than what's cached → gap-fill ---
    earlier_start = (full.index[0] - pd.Timedelta(days=5)).date()
    up2 = _BoundedFake({"AAA": full})
    res2 = batch_download(up2, db, ["AAA"], earlier_start,
                          batch_size=10, pause_seconds=0)
    assert res2["n_fetched"] == 1, f"expected gap-fill; got {res2}"
    assert res2["n_cached"] == 0
    assert "AAA" in res2["fetched"]
    assert up2.calls >= 1
    # And our DB now actually has more history than before:
    assert db.start_date("AAA") <= full.index[0], \
        f"expected start_date <= {full.index[0]}, got {db.start_date('AAA')}"
