"""End-to-end integration: full run_strategy on synthetic data, offline.

These tests double as the regression suite: the pipeline, the persisted run,
and the metrics block must come back coherent.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from pytrader.config import Config
from pytrader.strategy import PipelineConfig, run_strategy
from pytrader.testing import InMemoryStore, synthetic_series


@pytest.fixture
def price_data():
    return synthetic_series(n=700, n_tickers=8, horizon=5, seed=7)


@pytest.fixture
def cfg(price_data) -> Config:
    cfg = Config.default()
    cfg.include = [t for t in price_data if t != "SPY"]
    cfg.exclude = []
    cfg.faith.enabled = False
    cfg.backtest.lookback_years = 5
    return cfg


def _run(store, cfg, db=None, name="it", p=None):
    p = p or PipelineConfig(horizon=5, top_n=3, train_fraction=0.7,
                            random_state=1)
    return run_strategy(store, cfg, db=db, name=name, pipeline=p)


def test_run_produces_coherent_metrics(price_data, cfg, tmp_db):
    store = InMemoryStore(price_data)
    res = _run(store, cfg, db=tmp_db)
    m = res.metrics["metrics"]
    b = res.metrics["benchmark"]
    for k in ("sharpe", "cagr", "max_drawdown", "total_return", "ann_vol",
              "hit_rate", "n_days", "train_ic", "valid_ic", "avg_turnover"):
        assert k in m, f"missing metric: {k}"
    assert m["n_days"] >= 5
    # finite
    for v in m.values():
        if isinstance(v, float):
            assert np.isfinite(v)
    # benchmark block present (SPY was in the store)
    assert "total_return" in b
    assert "information_ratio" in b


def test_run_persists_and_reloads(price_data, cfg, tmp_db):
    store = InMemoryStore(price_data)
    res = _run(store, cfg, db=tmp_db)
    rec = tmp_db.get_run(res.run_id)
    assert rec is not None
    assert rec.name == "it"
    eq = tmp_db.get_equity(res.run_id)
    assert eq is not None
    # first saved equity value ≈ 1.0 (normalized)
    assert abs(float(eq["equity"].iloc[0]) - 1.0) < 1e-6
    # length matches the simulation
    assert len(eq) == len(res.result.dates)


def test_run_deterministic_given_seed(price_data, cfg, tmp_db):
    store = InMemoryStore(price_data)
    r1 = _run(store, cfg, name="a",
              p=PipelineConfig(horizon=5, top_n=3, train_fraction=0.7,
                               random_state=123))
    r2 = _run(store, cfg, name="b",
              p=PipelineConfig(horizon=5, top_n=3, train_fraction=0.7,
                               random_state=123))
    a = r1.metrics["metrics"]
    b = r2.metrics["metrics"]
    for k in ("sharpe", "total_return", "max_drawdown", "hit_rate"):
        assert a[k] == b[k]


def test_run_topn_constraint(price_data, cfg, tmp_db):
    store = InMemoryStore(price_data)
    p = PipelineConfig(horizon=5, top_n=6, train_fraction=0.7, random_state=1)
    res = _run(store, cfg, db=tmp_db, p=p)
    # every period holds exactly top_n tickers (enough candidates exist)
    assert all(len(pr.holdings) == 6 for pr in res.result.periods)


def test_run_insufficient_tickers_raises(price_data, cfg, tmp_db):
    cfg2 = cfg
    cfg2.include = ["TKR00", "TKR01", "TKR02"]  # only 3 tickers
    cfg2.faith.enabled = False
    store = InMemoryStore(price_data)
    with pytest.raises(ValueError):
        _run(store, cfg2, p=PipelineConfig(top_n=5))


def test_run_universe_reasons_visible_in_config(price_data, cfg, tmp_db):
    # add a faith rule that blocks one of the synthetic tickers
    cfg.faith.enabled = True
    cfg.faith.blocked_tickers = ["TKR01"]
    store = InMemoryStore(price_data)
    res = _run(store, cfg, db=tmp_db)
    rej = res.config["universe_rejected"]
    r1 = [r for r in rej if r["ticker"] == "TKR01"][0]
    assert r1["reasons"][0].startswith("faith:ticker")


def test_run_does_not_leak_benchmark_into_tradables(price_data, cfg, tmp_db):
    store = InMemoryStore(price_data)
    res = _run(store, cfg, db=tmp_db)
    for pr in res.result.periods:
        assert "SPY" not in pr.holdings


def test_run_config_records_ticker_list(price_data, cfg, tmp_db):
    store = InMemoryStore(price_data)
    res = _run(store, cfg, db=tmp_db)
    assert set(res.config["tickers"]) == set(cfg.include)
    assert res.config["n_tickers"] == len(cfg.include)


def test_pipeline_config_roundtrip():
    cfg = Config.default()
    p = PipelineConfig.from_config(cfg)
    d = p.to_dict()
    assert d["horizon"] == cfg.backtest.horizon
    assert d["top_n"] == cfg.backtest.top_n
    assert d["commission_bps"] == cfg.backtest.commission_bps
    p2 = PipelineConfig(**d)
    assert p2.top_n == p.top_n


def test_strategy_runs_under_two_minutes(price_data, cfg, tmp_db):
    store = InMemoryStore(price_data)
    t0 = time.perf_counter()
    res = _run(store, cfg, db=tmp_db)
    dt = time.perf_counter() - t0
    assert dt < 120
    assert res.run_id  # sanity
