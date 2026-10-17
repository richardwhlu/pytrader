"""Model wrapper: fit/predict, persistence round-trip, feature importance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pytrader.model import XGBoostModel


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "noise": rng.normal(0, 1, n),
    })
    # y depends only on 'a' so feature importance is predictable
    y = 2.0 * X["a"] + 0.1 * rng.normal(0, 1, n)
    return X, y


def test_fit_predict_shape(toy):
    X, y = toy
    m = XGBoostModel(n_estimators=20, early_stopping_rounds=None)
    m.fit(X, y.to_numpy())
    p = m.predict(X)
    assert p.shape == (len(X),)
    # should learn the signal: correlation with y is strongly positive
    assert np.corrcoef(p, y.to_numpy())[0, 1] > 0.7


def test_raises_before_fit():
    m = XGBoostModel()
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(RuntimeError):
        m.predict(X)


def test_feature_importance_favors_a(toy):
    X, y = toy
    m = XGBoostModel(n_estimators=40, early_stopping_rounds=None)
    m.fit(X, y.to_numpy())
    imp = dict(m.feature_importance())
    assert imp["a"] > imp["noise"]


def test_save_load_roundtrip(toy, tmp_path):
    X, y = toy
    m = XGBoostModel(n_estimators=25, early_stopping_rounds=None)
    m.fit(X, y.to_numpy())
    before = m.predict(X)
    m.save(tmp_path, meta={"k": "v"})
    loaded = XGBoostModel.load(tmp_path)
    after = loaded.predict(X)
    assert np.allclose(before, after)
    assert loaded.feature_names() == ["a", "b", "noise"]
    assert loaded.meta.get("k") == "v"


def test_early_stopping_still_converges(toy):
    X, y = toy
    m = XGBoostModel(n_estimators=200, early_stopping_rounds=20)
    m.fit(X, y.to_numpy())
    p = m.predict(X)
    assert np.corrcoef(p, y.to_numpy())[0, 1] > 0.7
