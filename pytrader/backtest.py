"""Backtest engine: model predictions → top-N cost-adjusted equity curve.

Design: sequential fixed-length holding periods of `horizon` trading days,
starting at model decision dates. Turnover cost = fraction of portfolio
swapped at the boundary × total cost bps. Long-only, equal-weight, no
shorting, no leverage.

`simulate(...)` / `SimulationResult` is the stable interface; this module
can be swapped for vectorbt (or a custom engine) later without touching the
model, feature, or UI layers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from pytrader.data import DailyPrice


@dataclass
class BacktestConfig:
    horizon: int = 5
    top_n: int = 5
    commission_bps: float = 1.0
    slippage_bps: float = 5.0

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1 trading day")
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs must be >= 0")

    @property
    def cost_bps(self) -> float:
        return float(self.commission_bps + self.slippage_bps)


@dataclass
class Period:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    gross_return: float
    turnover: float
    net_return: float
    holdings: list[str]


@dataclass
class SimulationResult:
    periods: list[Period]
    dates: pd.DatetimeIndex          # daily mark dates (start of first period → end of last)
    equity: pd.Series                # normalized (starts at 1.0), aligned to dates
    benchmark_equity: pd.Series | None

    @property
    def first_date(self) -> pd.Timestamp:
        return self.dates[0]

    @property
    def last_date(self) -> pd.Timestamp:
        return self.dates[-1]

    @property
    def n_days(self) -> int:
        return len(self.dates)


def _price_table(
    series: list[DailyPrice], benchmark: str | None
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Return (tradable price frame, benchmark close) both on a shared date index.

    The benchmark is dropped from the tradable frame and returned separately.
    """
    frames = {s.ticker.upper(): s.close for s in series}
    bench_close: pd.Series | None = None
    for t in list(frames):
        if benchmark is not None and t == benchmark.upper():
            bench_close = frames.pop(t)
    frame = pd.DataFrame(frames).sort_index()
    frame = frame.dropna(how="all")
    frame = frame.ffill().dropna(how="any")  # common trading calendar
    if bench_close is not None:
        b = bench_close.reindex(frame.index)
        if not b.notna().all():
            bench_close = None
    return frame, bench_close


def _turnover(prev: np.ndarray | None, new: np.ndarray) -> float:
    if prev is None:
        return float(np.sum(new))
    return float(np.sum(new) - float(np.minimum(prev, new).sum()))


def _validate_predictions(
    predictions: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    dec = predictions
    missing = [c for c in dec.columns if c not in prices.columns]
    if missing:
        raise ValueError(f"predictions for unknown tickers: {missing[:5]}")
    # reindex COLUMNS (not index!) so all tradable tickers are present.
    dec = dec.reindex(columns=prices.columns)
    return dec


def simulate(
    predictions: pd.DataFrame,
    series: list[DailyPrice],
    config: BacktestConfig,
    benchmark: str | None = "SPY",
) -> SimulationResult:
    """Run the top-N backtest.

    predictions: DataFrame indexed by decision dates (rows), with one column
    per tradable ticker and predicted next-horizon log returns. Decisions
    may be daily; the engine takes every `horizon`-th eligible decision date.

    Eligibility: a decision date d is usable iff prices exist for d and for
    horizon full trading days after d (exclusive start at d). Decisions that
    fall too close to the end are dropped, never padded.
    """
    if predictions is None or len(predictions) == 0:
        raise ValueError("empty predictions")
    prices, bench = _price_table(series, benchmark)
    if len(prices) == 0:
        raise ValueError("no common price calendar")
    dec = _validate_predictions(predictions, prices)
    cost = config.cost_bps / 10_000.0
    n = len(prices)
    price_arr = prices.to_numpy(dtype="float64")
    cols = list(prices.columns)
    col_idx = {c: i for i, c in enumerate(cols)}

    usable = [d for d in dec.index if d in prices.index]
    usable_dates: list[tuple[int, pd.Timestamp]] = []
    for d in usable:
        i = prices.index.get_loc(d)
        if i + config.horizon < n:
            usable_dates.append((i, d))
    if not usable_dates:
        raise ValueError("no decision date has a full horizon of forward prices")
    # keep decisions greedily so periods are adjacent (no gaps)
    picked: list[int] = []
    last_end = -1
    for i, _d in usable_dates:
        if i > last_end:
            picked.append(i)
            last_end = i + config.horizon

    periods: list[Period] = []
    prev_w: np.ndarray | None = None
    for i in picked:
        window = price_arr[i : i + config.horizon]  # days i..i+horizon-1
        fwd = window[-1] / window[0]
        pred_vec = _row_at(dec, prices.index[i], col_idx, cols)
        w = _topn_weights(pred_vec, config.top_n, col_idx)
        gross = float(np.sum(w * fwd)) - 1.0
        turnover = _turnover(prev_w, w)
        net = gross - turnover * cost
        holdings = [
            cols[k] for k in np.where(w > 0)[0]
        ]
        periods.append(
            Period(
                start_date=prices.index[i],
                end_date=prices.index[i + config.horizon - 1],
                gross_return=gross,
                turnover=turnover,
                net_return=net,
                holdings=holdings,
            )
        )
        prev_w = w

    if not periods:
        raise ValueError("no complete rebalance periods available in window")

    # Build daily equity curve: within each period the (net) return is
    # interpolated log-linearly over the holding days; at period boundaries
    # the exact compounded value is pinned.
    daily: dict[pd.Timestamp, float] = {}
    loc = {d: i for i, d in enumerate(prices.index)}
    eq_at_start = 1.0
    for p in periods:
        i0 = loc[p.start_date]
        i1 = loc[p.end_date]
        if i0 == i1:
            day_range = [prices.index[i0]]
            marks = [eq_at_start * (1.0 + p.net_return)]
        else:
            day_range = list(prices.index[i0 : i1 + 1])
            ln0, ln1 = np.log(eq_at_start), np.log(eq_at_start * (1.0 + p.net_return))
            t = np.linspace(0.0, 1.0, len(day_range))
            marks = np.exp(ln0 + t * (ln1 - ln0)).tolist()
            marks[-1] = eq_at_start * (1.0 + p.net_return)  # pin boundary
        for d, m in zip(day_range[:-1], marks[:-1]):
            daily[d] = float(m)
        daily[day_range[-1]] = float(marks[-1])
        eq_at_start *= 1.0 + p.net_return

    dates = pd.DatetimeIndex(sorted(daily.keys()))
    equity = pd.Series([daily[d] for d in dates], index=dates, name="equity")
    equity = equity / equity.iloc[0]

    bench_curve = None
    if bench is not None and len(bench) > 1:
        b = bench.reindex(dates)
        if b.notna().all():
            bench_curve = (b / b.iloc[0]).rename("benchmark_equity")

    return SimulationResult(
        periods=periods,
        dates=dates,
        equity=equity,
        benchmark_equity=bench_curve,
    )


def _row_at(
    dec: pd.DataFrame, d: pd.Timestamp, col_idx: dict[str, int], cols: list[str]
) -> np.ndarray:
    try:
        row = dec.loc[d]
    except KeyError:
        raise ValueError(f"decision date {d} missing from predictions")
    vec = np.full(len(cols), np.nan)
    for c in cols:
        v = row[c]
        if pd.notna(v):
            vec[col_idx[c]] = float(v)
    return vec


def _topn_weights(
    pred_vec: np.ndarray, top_n: int, col_idx: dict[str, int]
) -> np.ndarray:
    w = np.zeros(len(pred_vec))
    valid = np.where(~np.isnan(pred_vec))[0]
    if len(valid) == 0:
        return np.full(len(pred_vec), 1.0 / len(w))
    order = valid[np.argsort(pred_vec[valid])[::-1]]
    k = min(top_n, len(order))
    for j in order[:k]:
        w[j] = 1.0 / k
    return w


def predictions_from_panel(
    X: pd.DataFrame, scores: np.ndarray, decision_every: int = 1
) -> pd.DataFrame:
    """Reshape long-form (date, ticker) scores into wide decision rows.

    X must have columns `date`, `ticker`, the feature columns; scores share
    X's index order. One row per date with `decision_every` skipped rows.
    """
    if len(X) != len(scores):
        raise ValueError("score length mismatch")
    df = pd.DataFrame({"date": X["date"].to_numpy(), "ticker": X["ticker"].to_numpy(),
                       "score": np.asarray(scores, dtype="float64")})
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(index="date", columns="ticker", values="score")
    wide.index = pd.DatetimeIndex(wide.index).sort_values()
    return wide.iloc[::decision_every]
