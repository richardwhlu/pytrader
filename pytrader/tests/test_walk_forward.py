"""walk_forward_predict — pure anchored walk-forward over a panel.

Uses a small deterministic stand-in for ``Model`` so we don't need xgboost
in the test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pytrader.strategy import _pearson, walk_forward_predict


class _LinearFit:
    """Tiny deterministic model: fit is least-squares; predict is linear."""

    def __init__(self) -> None:
        self.coef: np.ndarray | None = None
        self.basis: int | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "_LinearFit":
        n = len(X)
        if n < 2:
            # degenerate; return zeros
            self.coef = np.zeros(1)
            self.basis = 0
            return self
        # Fit a single-coefficient model: y = coef * X[f0]
        x0 = X[X.columns[0]].to_numpy(dtype="float64")
        # Least-squares on a single feature
        A = np.stack([x0, np.full_like(x0, 1.0)], axis=1)
        self.coef = np.linalg.lstsq(A, y, rcond=None)[0]
        self.basis = 1
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef is None or self.basis != 1:
            raise RuntimeError("not fit")
        x0 = X[X.columns[0]].to_numpy(dtype="float64")
        return self.coef[0] * x0 + self.coef[1]

    def feature_names(self) -> list[str]:
        return ["f"]

    def feature_importance(self) -> list[tuple[str, float]]:
        return [("f", 1.0)]


def _make_panel(n_dates: int, n_tickers: int, seed: int = 0) -> pd.DataFrame:
    """Deterministic panel: y = +1 * feature value (so the linear model learns it)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    rows = []
    for i, dt in enumerate(dates):
        for t in range(n_tickers):
            f = float(i * 0.01 + t * 0.5 + rng.standard_normal() * 0.1)
            rows.append({"date": dt, "ticker": f"T{t}", "f": f,
                         "y": f + rng.standard_normal() * 0.05})
    return pd.DataFrame(rows)


def test_fold_count_and_no_lookahead():
    n_tickers = 5
    X = _make_panel(n_dates=30, n_tickers=n_tickers)
    fit_row_counts: list[int] = []

    class _Spy(_LinearFit):
        def fit(self, X: pd.DataFrame, y: np.ndarray) -> "_Spy":
            fit_row_counts.append(len(X))
            return super().fit(X, y)

    factory = lambda: _Spy()
    preds, folds = walk_forward_predict(X, ["f"], factory,
                                        min_train_rows=10, retrain_every=1)
    assert len(folds) >= 5
    dates = [dt for dt, _ in folds]
    assert dates == sorted(dates), "folds must be in date order"
    assert len(fit_row_counts) == len(folds), "one fit per fold"

    # Rows-per-date is constant (n_tickers). So for a fold at date index k,
    # the training window = (k dates * n_tickers) rows. If the fold's own
    # rows leaked in, the count would be 5 higher on the first fold (where
    # the window is smallest relative to the leak).
    # The first fit is the tightest: train window = 2 dates = 10 rows (min).
    # A leak would give 15.
    # Compute expected first-fold train size independently:
    per_date = X.groupby(pd.to_datetime(X["date"])).size().sort_index()
    cum = per_date.cumsum()
    first_idx = int(np.searchsorted(cum.to_numpy(), 10, side="left")) + 1
    expected_first_train = int(cum.iloc[first_idx - 1])
    assert fit_row_counts[0] == expected_first_train, (
        f"fit0={fit_row_counts[0]} expected={expected_first_train}"
    )
    # Train windows strictly grow as folds advance.
    assert fit_row_counts == sorted(fit_row_counts)
    assert len(set(fit_row_counts)) == len(fit_row_counts), "strict growth"


def test_retrain_every_thins_folds():
    X = _make_panel(n_dates=30, n_tickers=5)
    _, dense = walk_forward_predict(X, ["f"], lambda: _LinearFit(),
                                    min_train_rows=10, retrain_every=1)
    _, sparse = walk_forward_predict(X, ["f"], lambda: _LinearFit(),
                                     min_train_rows=10, retrain_every=3)
    assert len(sparse) < len(dense)
    # Same starting date as dense (min_train_rows unchanged).
    assert sparse[0][0] == dense[0][0]


def test_output_shape_matches_predictions_from_panel():
    X = _make_panel(n_dates=30, n_tickers=5)
    preds, folds = walk_forward_predict(X, ["f"], lambda: _LinearFit(),
                                        min_train_rows=10, retrain_every=1)
    # One row per fold, one column per ticker, index is DatetimeIndex.
    assert isinstance(preds.index, pd.DatetimeIndex)
    assert set(preds.columns) == {"T0", "T1", "T2", "T3", "T4"}
    assert len(preds) == len(folds)
    # Scores are finite on the rows of the fold
    for dt, _ic in folds:
        row = preds.loc[dt]
        assert np.all(np.isfinite(row.to_numpy()))


def test_raises_when_too_few_rows():
    X = _make_panel(n_dates=2, n_tickers=3)
    with pytest.raises(ValueError, match="need"):
        walk_forward_predict(X, ["f"], lambda: _LinearFit(),
                             min_train_rows=1000, retrain_every=1)


def test_pearson_handles_singular_inputs():
    assert _pearson(np.array([1.0, 1.0]), np.array([2.0, 2.0])) == 0.0
    assert _pearson(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == pytest.approx(1.0)
