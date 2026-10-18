"""Binary task ("up in horizon") — end-to-end over run_strategy."""
from __future__ import annotations

import pytest

from pytrader import testing
from pytrader.config import Config
from pytrader.strategy import PipelineConfig, run_strategy


@pytest.fixture
def price_data():
    return testing.synthetic_series(n=300, n_tickers=6, horizon=21, seed=11)


def _cfg(price_data) -> Config:
    cfg = Config.default()
    cfg.include = [t for t in price_data if t != "SPY"]
    cfg.exclude = []
    cfg.faith.enabled = False
    cfg.backtest.lookback_years = 5
    return cfg


def test_binary_task_runs_and_reports_discrimination(price_data):
    cfg = _cfg(price_data)
    res = run_strategy(
        testing.InMemoryStore(price_data),
        cfg,
        name="binary_smoke",
        pipeline=PipelineConfig(horizon=21, top_n=3, train_fraction=0.7),
        task="binary",
    )
    assert res.config.get("task") == "binary"
    m = res.metrics["metrics"]
    # Binary task must report discrimination metrics, not IC.
    assert "val_auc" in m
    assert 0.0 <= m["val_auc"] <= 1.0
    assert "val_base_rate" in m
    # IC is zero for the binary task (uninformative on a 0/1 target).
    assert m["train_ic"] == 0.0
    assert m["valid_ic"] == 0.0
    # The backtest still runs to completion on the (probability) scores.
    assert len(res.result.periods) >= 1


def test_regression_task_is_unchanged_default(price_data):
    cfg = _cfg(price_data)
    res = run_strategy(
        testing.InMemoryStore(price_data),
        cfg,
        name="reg_default",
        pipeline=PipelineConfig(horizon=21, top_n=3, train_fraction=0.7),
    )
    # Default path unchanged: reports IC, not binary stats.
    m = res.metrics["metrics"]
    assert "val_auc" not in m
    assert "train_ic" in m
    assert "valid_ic" in m
