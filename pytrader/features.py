"""Feature engineering: compute cross-sectional factors from prices.

Each feature is a column in the model matrix indexed by (date, ticker).
`compute_panel` aligns all tickers onto a common trading calendar (the
intersection of all series), drops leading NaNs, and returns model-ready
arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pytrader.data import DailyPrice

# --- individual factor functions (each independent, easily extended) --------


def momentum(close: pd.Series, window: int) -> pd.Series:
    """Trailing-window log return."""
    return np.log(close / close.shift(window))


def volatility(close: pd.Series, window: int) -> pd.Series:
    """Annualized realized vol of log returns."""
    r = np.log(close).diff()
    return r.rolling(window).std() * np.sqrt(252)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    r = close.diff()
    up = r.clip(lower=0.0)
    dn = -r.clip(upper=0.0)
    su = up.ewm(alpha=1 / window, adjust=False).mean()
    sd = dn.ewm(alpha=1 / window, adjust=False).mean()
    rs = su / sd.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(50.0)


def trend(close: pd.Series, window: int) -> pd.Series:
    """Close relative to moving average (z-scored by rolling vol)."""
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    return ((close - ma) / sd.replace(0.0, np.nan)).fillna(0.0)


@dataclass
class FeatureSpec:
    """Which factors to compute; extend freely.

    ``custom`` maps a factor name to a callable ``(close: pd.Series) ->
    pd.Series`` — register your own features here without touching this
    module, e.g. ``FeatureSpec(custom={"bb_width": lambda c: ...})``.
    """

    momentum_windows: list[int] = None  # type: ignore[assignment]
    vol_windows: list[int] = None  # type: ignore[assignment]
    rsi: int = 14
    trend_windows: list[int] = None  # type: ignore[assignment]
    custom: dict | None = None

    def __post_init__(self) -> None:
        if self.momentum_windows is None:
            self.momentum_windows = [21, 63]
        if self.vol_windows is None:
            self.vol_windows = [21]
        if self.trend_windows is None:
            self.trend_windows = [21]

    def column_names(self) -> list[str]:
        cols = []
        for w in self.momentum_windows:
            cols.append(f"momentum_{w}")
        for w in self.vol_windows:
            cols.append(f"volatility_{w}")
        cols.append(f"rsi_{self.rsi}")
        for w in self.trend_windows:
            cols.append(f"trend_{w}")
        if self.custom:
            cols.extend(self.custom.keys())
        return cols

    def compute(self, close: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame(index=close.index)
        for w in self.momentum_windows:
            df[f"momentum_{w}"] = momentum(close, w)
        for w in self.vol_windows:
            df[f"volatility_{w}"] = volatility(close, w)
        df[f"rsi_{self.rsi}"] = rsi(close, self.rsi)
        for w in self.trend_windows:
            df[f"trend_{w}"] = trend(close, w)
        if self.custom:
            for name, fn in self.custom.items():
                df[name] = fn(close)
        return df


def future_return(close: pd.Series, horizon: int) -> pd.Series:
    """Label: log return `horizon` days ahead (NaN at the end)."""
    return np.log(close.shift(-horizon) / close)


def compute_panel(
    series: list[DailyPrice],
    spec: FeatureSpec,
    horizon: int,
    benchmark: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Stack all series into a long panel.

    Returns (X_panel, y_panel) where the index is a MultiIndex
    (date, ticker) and X_panel columns are spec.column_names().
    Rows without a valid future label are kept in X but dropped from y;
    the model caller aligns on non-NaN y.
    """
    frames = []
    labels = {}
    bench_close = None
    for s in series:
        if benchmark and s.ticker == benchmark.upper() and bench_close is None:
            bench_close = s.close
            continue
        f = spec.compute(s.close).dropna()
        if f.empty:
            continue
        f = f.reset_index()
        f = f.rename(columns={f.columns[0]: "date"})
        f["ticker"] = s.ticker.upper()
        y = future_return(s.close, horizon)
        y = y.reset_index()
        y.columns = ["date", "y"]
        f = f.merge(y, on="date", how="left")
        frames.append(f)
        labels[s.ticker.upper()] = s.close
    if not frames:
        raise ValueError("no usable series in panel (too short?)")
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    panel["date"] = pd.to_datetime(panel["date"])
    if bench_close is not None:
        labels["BENCH"] = bench_close
    y = panel["y"]
    return panel.drop(columns=["y"]), y
