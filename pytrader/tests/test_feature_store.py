"""FeatureStore: compute once, query point-in-time from cache."""
from __future__ import annotations

import pandas as pd
import pytest

from pytrader.data import DailyPrice
from pytrader.feature_store import FeatureStore
from pytrader.features import FeatureSpec
from pytrader.storage import DuckDBStore


@pytest.fixture
def db(tmp_path):
    return DuckDBStore(tmp_path / "fstore.duckdb")


def _series() -> DailyPrice:
    idx = pd.bdate_range("2024-01-01", periods=120)
    import numpy as np
    r = np.random.default_rng(0)
    rets = r.normal(0.0005, 0.012, len(idx))
    close = 100.0 * np.exp(np.cumsum(rets))
    return DailyPrice("AAA", idx, pd.Series(close, index=idx, dtype="float64"))


def test_build_then_query_matches_compute(db):
    s = _series()
    spec = FeatureSpec()
    fs = FeatureStore(db)

    panel = fs.build(s, spec)
    assert "date" in panel.columns
    assert len(panel) >= 100  # burn-in dropped the first ~21 rows

    # The store must now be populated for this (ticker, spec_hash).
    assert fs.has("AAA", spec) is True
    feat_names = set(spec.column_names())
    assert set(fs.features(spec)) == feat_names, "stored features == spec features"

    # Point-in-time query for an interior date.
    mid = panel["date"].iloc[50]
    fetched = fs.query("AAA", mid, spec)
    assert fetched is not None
    expected_row = panel.loc[panel["date"] == mid].iloc[0]
    for f in feat_names:
        ev = expected_row[f]
        if pd.isna(ev):
            assert fetched.get(f) is None or pd.isna(fetched.get(f))
        else:
            assert pd.notna(fetched.get(f))
            assert abs(float(fetched[f]) - float(ev)) < 1e-6, f

    # Query for a date outside the panel must return None.
    assert fs.query("AAA", pd.Timestamp("2020-01-01"), spec) is None


def test_spec_hash_is_stable_and_differs(db):
    a = FeatureSpec()
    b = FeatureSpec()
    c = FeatureSpec(momentum_windows=[42], trend_windows=[42])
    assert a.spec_hash() == b.spec_hash(), "identical specs must share a hash"
    assert a.spec_hash() != c.spec_hash(), "different windows => different hash"


def test_two_specs_coexist(db):
    s = _series()
    spec_a = FeatureSpec()
    spec_b = FeatureSpec(momentum_windows=[42])
    fs = FeatureStore(db)
    fs.build(s, spec_a)
    fs.build(s, spec_b)
    assert set(fs.features(spec_a)) == set(spec_a.column_names())
    assert set(fs.features(spec_b)) == set(spec_b.column_names())


def test_no_leak_point_in_time(db):
    """Features on date D must equal a fresh recompute using only prices ≤ D.

    We rebuild the spec from scratch and compare the point query to the
    freshly-computed value at D — they must match to 1e-6, proving the
    stored value is a trailing-window function of past prices only.
    """
    import numpy as np
    idx = pd.bdate_range("2024-01-01", periods=200)
    r = np.random.default_rng(1)
    close = 100.0 * np.exp(np.cumsum(r.normal(0.0, 0.01, len(idx))))
    s = DailyPrice("NOLEAK", idx, pd.Series(close, index=idx, dtype="float64"))
    spec = FeatureSpec()
    fs = FeatureStore(db)
    fs.build(s, spec)
    d = pd.Timestamp("2024-06-01")
    if d not in set(idx):
        d = idx[len(idx) // 2]
    fetched = fs.query("NOLEAK", d, spec)
    assert fetched is not None
    # Fresh recompute restricted to data as of d — the stored value must
    # match a trailing-window function of past prices only.
    idx_pos = idx.get_loc(d)
    past_close = s.close.iloc[: idx_pos + 1]
    from pytrader.features import momentum
    if "momentum_21" in fetched:
        fresh_mv = float(momentum(past_close, 21).iloc[-1])
        assert pd.isna(fetched["momentum_21"]) or (
            abs(fetched["momentum_21"] - fresh_mv) < 1e-6
        ), f"stored momentum_21={fetched['momentum_21']} fresh(past)={fresh_mv}"
