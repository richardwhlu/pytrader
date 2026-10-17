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
