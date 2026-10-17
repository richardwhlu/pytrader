"""Flexibility contract: developers bring their own features, learner, metrics.

These verify the three pluggable seams are actually open, end-to-end, through
run_strategy — requirement #2 from the owner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pytrader.metrics as M
from pytrader.config import Config
from pytrader.features import FeatureSpec
from pytrader.strategy import PipelineConfig, run_strategy


@pytest.fixture
def price_data():
    from pytrader.testing import synthetic_series
    return synthetic_series(n=600, n_tickers=6, horizon=5, seed=3)


@pytest.fixture
def cfg(price_data) -> Config:
    cfg = Config.default()
    cfg.include = [t for t in price_data if t != "SPY"]
    cfg.exclude = []
    cfg.faith.enabled = False
    config = cfg
    return config


@pytest.fixture(autouse=True)
def _reset():
    M.clear_custom_metrics()
    yield
    M.clear_custom_metrics()


def test_feature_spec_custom_factor_is_used(price_data, cfg):
    spec = FeatureSpec(custom={"my_dd": lambda c: c / c.rolling(30).max() - 1.0})
    assert "my_dd" in spec.column_names()
    store = InMemoryStore(price_data)
    res = run_strategy(store, cfg, name="ff", pipeline=PipelineConfig(random_state=1),
                       features=spec)
    names = [k for k, _ in res.feature_importance]
    assert "my_dd" in names


def test_custom_learner_runs_end_to_end(price_data, cfg):
    class Linear:
        def __init__(self):
            self._w = None
        def fit(self, X, y):
            Xn = X.to_numpy(float)
            # ridge-ish least squares
            self._w = np.linalg.lstsq(Xn + 1e-3, y, rcond=None)[0]
            self._feats = list(X.columns)
            return self
        def predict(self, X):
            return X.to_numpy(float) @ self._w
        def feature_names(self):
            return self._feats
        def save(self, *a, **k):
            return None
    store = InMemoryStore(price_data)
    res = run_strategy(store, cfg, name="lin", pipeline=PipelineConfig(random_state=1),
                       model=Linear())
    assert isinstance(res.metrics["metrics"]["sharpe"], float)
    # custom learner has no feature_importance → empty list, no crash
    assert res.feature_importance == []


def test_registered_custom_metric_in_run_metrics(price_data, cfg):
    @M.register_metric("worst_period")
    def worst(equity, dates, periods):
        return float(min((p.net_return for p in periods), default=0.0))

    store = InMemoryStore(price_data)
    res = run_strategy(store, cfg, name="mm", pipeline=PipelineConfig(random_state=1))
    assert "worst_period" in res.metrics["metrics"]
    # value must equal the min over the periods it was given
    expected = min(p.net_return for p in res.result.periods)
    assert res.metrics["metrics"]["worst_period"] == pytest.approx(expected, abs=1e-6)


def test_bad_custom_metric_does_not_break_run(price_data, cfg):
    M.register_metric("boom", lambda *a: (_ for _ in ()).throw(ValueError("x")))
    M.register_metric("nan", lambda *a: float("nan"))
    store = InMemoryStore(price_data)
    res = run_strategy(store, cfg, name="nb", pipeline=PipelineConfig(random_state=1))
    assert res.metrics["metrics"]["sharpe"]
    assert "boom" not in res.metrics["metrics"]
    assert "nan" not in res.metrics["metrics"]


def test_model_rejects_nonconforming_learner(price_data, cfg):
    class NoFit:
        def predict(self, X):
            return np.zeros(len(X))
    with pytest.raises(TypeError):
        run_strategy(InMemoryStore(price_data), cfg, name="bad",
                     pipeline=PipelineConfig(random_state=1), model=NoFit())


def test_default_model_still_has_feature_importance(price_data, cfg):
    store = InMemoryStore(price_data)
    res = run_strategy(store, cfg, name="def", pipeline=PipelineConfig(random_state=1))
    # XGBoostModel exposes feature importance
    assert len(res.feature_importance) >= 3


def test_feature_spec_roundtrip_columns(cfg):
    spec = FeatureSpec(custom={"a": lambda c: c.diff(), "b": lambda c: c.pct_change()})
    X = spec.compute(pd.Series(np.linspace(100, 110, 200),
                                index=pd.bdate_range("2024-01-01", periods=200)))
    assert set(spec.column_names()) <= set(X.columns)
    assert {"a", "b"} <= set(X.columns)


def InMemoryStore(data):
    from pytrader.testing import InMemoryStore as S
    return S(data)


# import at module level so tests using the name above resolve
from pytrader.testing import InMemoryStore  # noqa: E402
