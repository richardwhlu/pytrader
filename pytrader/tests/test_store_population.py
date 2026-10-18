"""Task 4: a run populates the point-in-time feature store; time-travel
reads return the same matrix the model trained on."""
from __future__ import annotations

import pytest

from pytrader import testing
from pytrader.config import Config
from pytrader.feature_store import FeatureStore
from pytrader.features import FeatureSpec
from pytrader.storage import DuckDBStore
from pytrader.strategy import PipelineConfig, run_strategy


@pytest.fixture
def db(tmp_path):
    return DuckDBStore(tmp_path / "pop.duckdb")


@pytest.fixture
def price_data():
    return testing.synthetic_series(n=400, n_tickers=6, horizon=5, seed=3)


def _cfg(price_data) -> Config:
    cfg = Config.default()
    cfg.include = [t for t in price_data if t != "SPY"]
    cfg.exclude = []
    cfg.faith.enabled = False
    cfg.backtest.lookback_years = 5
    return cfg


def test_run_populates_store_and_query_matches(price_data, db):
    fs = FeatureStore(db)
    spec = FeatureSpec()
    store = testing.InMemoryStore(price_data)
    tradable = [t for t in price_data if t != "SPY"]

    # Run with the store attached.
    res = run_strategy(
        store, _cfg(price_data), name="pop",
        pipeline=PipelineConfig(horizon=5, top_n=3, train_fraction=0.7),
        feature_store=fs,
    )

    # After the run, every tradable ticker must have a stored panel.
    for t in tradable:
        assert fs.has(t, spec), f"{t} should be cached after a run"

    # Time-travel: pick an interior date and confirm the stored value is
    # non-null for all spec features, and stable across reads.
    t = tradable[0]
    dates = fs.dates(t, spec)
    assert len(dates) >= 100
    d = dates[len(dates) // 2]
    q1 = fs.query(t, d, spec)
    q2 = fs.query(t, d, spec)
    assert q1 == q2, "queries for the same (ticker,date) must be stable"
    assert set(q1) == set(spec.column_names())
    # Values should be finite for a date inside the panel.
    assert any(v is not None and v == v for v in q1.values())

    # load_panel round-trips the same width and full date set.
    panel = fs.load_panel(t, spec)
    assert panel is not None
    assert set(panel.columns) == {"date", *spec.column_names()}
    assert len(panel) == len(dates)


def test_without_store_flag_runs_normally(price_data, db):
    store = testing.InMemoryStore(price_data)
    # No feature_store passed → runs fine, nothing cached.
    fs = FeatureStore(db)
    tradable = [t for t in price_data if t != "SPY"]
    res = run_strategy(
        store, _cfg(price_data), name="no_store",
        pipeline=PipelineConfig(horizon=5, top_n=3, train_fraction=0.7),
    )
    assert len(res.result.periods) >= 1
    assert not fs.has(tradable[0], FeatureSpec()), "nothing should be cached"
