"""Shared pytest fixtures.

Keeps tests hermetic: synthetic data only, in-memory store, temp duckdb.
No network calls unless a test is explicitly marked `network`.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

# xgboost on macOS needs the LLVM OpenMP runtime; brew installs it keg-only.
if "DYLD_LIBRARY_PATH" not in os.environ and Path(
    "/opt/homebrew/opt/libomp/lib"
).exists():
    os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/libomp/lib"
os.environ.setdefault("XGBOOST_VERBOSITY", "0")

from pytrader.testing import InMemoryStore, synthetic_series  # noqa: E402


@pytest.fixture
def price_map() -> dict:
    """8 synthetic tickers + SPY benchmark, ~700 days."""
    return synthetic_series(n=700, n_tickers=8, horizon=5, seed=7)


@pytest.fixture
def store(price_map) -> InMemoryStore:
    return InMemoryStore(price_map)


@pytest.fixture
def tradable(price_map) -> list[str]:
    return [t for t in price_map if t != "SPY"]


@pytest.fixture
def tmp_db(tmp_path) -> "object":
    from pytrader.storage import DuckDBStore
    return DuckDBStore(tmp_path / "test.duckdb")
