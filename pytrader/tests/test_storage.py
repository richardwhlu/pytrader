"""Storage: duckdb round-trip, ordering, missing-run handling."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pytrader.storage import DuckDBStore, RunRecord


def _equity(n=20) -> pd.DataFrame:
    idx = np.arange(1_700_000_000, 1_700_000_000 + n * 86400, 86400)
    eq = np.cumprod(1 + np.random.default_rng(0).normal(0, 0.005, n))
    return pd.DataFrame({
        "equity": eq,
        "benchmark_equity": np.cumprod(1 + np.random.default_rng(1).normal(0, 0.005, n)),
    }).set_index(pd.DatetimeIndex(pd.to_datetime(idx, unit="s")))


def test_save_and_reload(tmp_db):
    rec = RunRecord("r1", "first", "2024-01-01T00:00:00Z",
                    {"k": 1}, {"m": 2})
    tmp_db.save_run(rec, _equity())
    got = tmp_db.get_run("r1")
    assert got is not None
    assert got.name == "first"
    assert got.config == {"k": 1}
    assert got.metrics == {"m": 2}
    eq = tmp_db.get_equity("r1")
    assert eq is not None
    assert set(eq.columns) >= {"equity", "benchmark_equity"}


def test_list_runs_order(tmp_db):
    for i in range(3):
        rec = RunRecord(f"r{i}", f"n{i}", "2024-01-01T00:00:00Z", {}, {})
        tmp_db.save_run(rec, _equity(n=5 + i))
    runs = tmp_db.list_runs()
    assert [r.run_id for r in runs] == ["r0", "r1", "r2"]


def test_missing_run_returns_none(tmp_db):
    assert tmp_db.get_run("nope") is None
    assert tmp_db.get_equity("nope") is None


def test_upsert_replaces(tmp_db, tmp_path):
    tmp_db.save_run(RunRecord("x", "v1", "t", {}, {"sharpe": 1})
                    , _equity(n=10))
    tmp_db.save_run(RunRecord("x", "v2", "t", {}, {"sharpe": 2})
                    , _equity(n=12))
    got = tmp_db.get_run("x")
    assert got.metrics == {"sharpe": 2}
    assert tmp_db.list_runs() and len(tmp_db.list_runs()) == 1


def test_equity_index_roundtrips_across_resolutions(tmp_db):
    """Save frames at both ns and µs resolution; read-back must match."""
    # ns resolution (classic pandas default)
    ns_idx = pd.bdate_range("2025-01-02", periods=10).as_unit("ns")
    assert str(ns_idx.dtype).endswith("[ns]")
    eq_ns = pd.Series(np.linspace(1, 2, 10), index=ns_idx)
    bench_ns = pd.Series(np.linspace(1, 1.5, 10), index=ns_idx)
    df_ns = pd.DataFrame({
        "equity": eq_ns.to_numpy(),
        "benchmark_equity": bench_ns.to_numpy(),
    }, index=ns_idx)
    tmp_db.save_run(RunRecord("ns", "ns", "t", {}, {}), df_ns)
    o1 = tmp_db.get_equity("ns")
    assert o1.index[0] == ns_idx[0]
    assert o1.index[-1] == ns_idx[-1]
    assert o1["equity"].tolist() == pytest.approx(
        np.linspace(1, 2, 10).tolist(), rel=1e-9
    )

    # microsecond resolution (new pandas default in 3.x)
    us_idx = pd.bdate_range("2025-01-02", periods=6).as_unit("us")
    assert str(us_idx.dtype).endswith("[us]")
    df_us = pd.DataFrame({
        "equity": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        "benchmark_equity": None,
    }, index=us_idx)
    tmp_db.save_run(RunRecord("us", "us", "t", {}, {}), df_us)
    o2 = tmp_db.get_equity("us")
    assert o2.index[0] == us_idx[0]
    assert o2.index[-1] == us_idx[-1]
    assert o2["equity"].tolist() == [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]


def test_equity_index_sort_after_roundtrip(tmp_db):
    """If the incoming frame is unordered, the read-back must be ordered."""
    idx = pd.bdate_range("2025-01-02", periods=6)
    eq = pd.DataFrame({"equity": [1, 2, 3, 4, 5, 6], "benchmark_equity": None})
    eq.index = idx
    # reverse the row order before saving
    eq_rev = eq.iloc[::-1]
    tmp_db.save_run(RunRecord("r", "r", "t", {}, {}), eq_rev)
    o = tmp_db.get_equity("r")
    assert o.index.is_monotonic_increasing
