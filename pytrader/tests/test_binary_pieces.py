"""Binary label + classifier contract (pure pieces, no pipeline)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pytrader.features import binary_updown, future_return
from pytrader.metrics import roc_auc, precision_at_k, classification_report
from pytrader.model import XGBoostBinaryModel


def test_binary_updown_is_01_and_nan_at_end():
    close = pd.Series([100, 101, 102, 99, 103, 102.0])
    h = 2
    out = binary_updown(close, h).to_numpy()
    # Hand-computed expectation for the leading rows.
    assert out[0] == 1.0   # close[2]=102 > close[0]=100 → up
    assert out[1] == 0.0   # close[3]=99  < close[1]=101 → down
    # Full leading slice vs recomputed expectation (exhaustive).
    exp = np.array([
        1.0 if close.iloc[i + h] > close.iloc[i] else 0.0
        for i in range(len(close) - h)
    ])
    assert np.array_equal(out[:len(exp)], exp), (out, exp)
    # Tailing h rows must be NaN.
    assert np.isnan(out[len(close) - h:]).all()


def test_binary_updown_differs_from_reg_label_where_direction_matches():
    close = pd.Series([100.0, 110.0, 120.0], index=pd.bdate_range("2024-01-01", periods=3))
    assert future_return(close, 1).iloc[0] > 0
    assert binary_updown(close, 1).iloc[0] == 1.0
    # A down move:
    close2 = pd.Series([120.0, 110.0, 100.0], index=pd.bdate_range("2024-01-01", periods=3))
    assert future_return(close2, 1).iloc[0] < 0
    assert binary_updown(close2, 1).iloc[0] == 0.0


def test_classifier_outputs_probability_scores():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120)})
    X["target"] = 0.5 * X["a"] + 0.3 * X["b"] + 0.2 * rng.normal(size=120)
    y = (X["target"] > np.median(X["target"])).astype(int)
    m = XGBoostBinaryModel(n_estimators=30, random_state=0)
    m.fit(X[["a", "b"]], y)
    p = m.predict(X[["a", "b"]])
    assert (p >= 0.0).all() and (p <= 1.0).all()
    # The model should learn that high 'target' → high P(up).
    assert p[y == 1].mean() > p[y == 0].mean()


def test_classifier_rejects_single_class():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=20)})
    with pytest.raises(ValueError, match="both classes"):
        XGBoostBinaryModel(n_estimators=5, random_state=0).fit(X, np.zeros(20))


def test_roc_auc_and_precision_sanity():
    # Perfect, random, inverse.
    assert roc_auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert roc_auc([1, 1, 0, 0], [0.5, 0.5, 0.5, 0.5]) == 0.5
    # precision@2 on a 4-row panel where the top-2 picked are both positive.
    assert precision_at_k([0, 1, 1, 0], [0.1, 0.9, 0.95, 0.3], k=2) == 1.0
    # Degenerate (all one class) → base rate / 0.5 by definition.
    assert precision_at_k([1, 1, 1], [0.9, 0.5, 0.2], k=2) == 1.0
    r = classification_report([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert set(r) == {"auc", "precision_at_5", "n_pos", "base_rate"}
