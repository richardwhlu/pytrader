"""End-to-end smoke: run_strategy(..., walk_forward=True) returns a PipelineResult.

Uses a tiny XGBoostModel (n_estimators=10) to keep it fast. This is
deliberately a thin smoke test — the heavy algorithmic guarantees are in
test_walk_forward.py against the pure ``walk_forward_predict`` function.
"""
from __future__ import annotations

import pytest

from pytrader import testing
from pytrader.config import Config
from pytrader.model import XGBoostModel
from pytrader.strategy import PipelineConfig, run_strategy


@pytest.fixture
def price_data():
    return testing.synthetic_series(n=260, n_tickers=6, horizon=5, seed=3)


def test_run_strategy_walk_forward_smoke(price_data):
    cfg = Config.default()
    cfg.include = [t for t in price_data if t != "SPY"]
    cfg.exclude = []
    cfg.faith.enabled = False
    cfg.backtest.lookback_years = 5

    res = run_strategy(
        testing.InMemoryStore(price_data),
        cfg,
        name="wf_smoke",
        pipeline=PipelineConfig(horizon=5, top_n=3, train_fraction=0.4),
        model=XGBoostModel(n_estimators=10, max_depth=2,
                           early_stopping_rounds=None, random_state=0),
        walk_forward=True,
        retrain_every=5,
    )
    # Result is a PipelineResult with plausible shape
    assert res.run_id
    assert res.metrics["metrics"]["train_ic"] is not None
    assert "sharpe" in res.metrics["metrics"]
    assert res.result is not None
    # Config dict carries fold metadata
    assert res.config.get("walk_forward") is True
    assert isinstance(res.config.get("n_folds"), int) and res.config["n_folds"] >= 1
    assert res.config.get("retrain_every") == 5
