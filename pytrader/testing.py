"""Testing helpers: deterministic synthetic prices + in-memory price store.

Synthetic data has a real (seeded) cross-sectional signal so feature, model,
and backtest tests exercise actual logic without touching yfinance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pytrader.data import DailyPrice, PriceStore


class InMemoryStore(PriceStore):
    def __init__(
        self,
        series: dict[str, DailyPrice],
        meta: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.series = {t.upper(): s for t, s in series.items()}
        self.meta = meta or {
            t: {"name": t, "sector": "Test"} for t in self.series
        }

    def fetch(self, tickers: list[str], start) -> list[DailyPrice]:
        out = []
        for t in tickers:
            s = self.series.get(t.upper())
            if s is None:
                from pytrader.data import DataUnavailable
                raise DataUnavailable(f"no series for {t}")
            mask = s.index >= pd.Timestamp(start)
            idx = s.index[mask]
            close = s.close.iloc[mask]
            out.append(DailyPrice(s.ticker, idx, close))
        return out

    def metadata(self, tickers: list[str]) -> dict[str, dict[str, str]]:
        return {t.upper(): self.meta.get(t.upper(), {"name": t, "sector": ""})
                for t in tickers}


def synthetic_series(
    n: int = 900,
    n_tickers: int = 12,
    horizon: int = 5,
    seed: int = 7,
    start: str = "2023-01-02",
    tickers: list[str] | None = None,
    benchmark: str = "SPY",
) -> dict[str, DailyPrice]:
    """Generate correlated daily closes with a momentum signal.

    Each ticker's alpha is a slowly rotating random walk; a few days' of
    past alpha (within horizon) feeds forward returns, creating exploitable
    momentum without leaking future information.
    """
    rng = np.random.default_rng(seed)
    tickers = tickers or [f"TKR{i:02d}" for i in range(n_tickers)]
    idx = pd.bdate_range(start, periods=n)
    out: dict[str, DailyPrice] = {}
    market = np.cumsum(rng.normal(0.0002, 0.010, n))
    for i, t in enumerate(tickers):
        skill = rng.normal(1.0, 0.5)  # ticker-level skill multiplier
        # alpha is a lightly decaying AR(1); the forward-return coupling is
        # calibrated so a perfect model gets ~15-40% return over ~100 days,
        # which is a realistic top-decile alpha.
        alpha = np.zeros(n)
        a = 0.0
        for d in range(1, n):
            a += rng.normal(0.0, 0.012)
            a *= 0.99
            alpha[d] = a
        lag = np.roll(alpha, horizon)
        lag[:horizon] = 0.0
        rets = 0.8 * (market[1:] - market[:-1]) + 0.2 * skill * lag[:-1] \
            + rng.normal(0.0, 0.006, n - 1)
        closes = 100.0 * np.exp(np.cumsum(rets))
        closes = np.concatenate([[100.0], closes])
        out[t] = DailyPrice(t, idx, pd.Series(closes, index=idx, dtype="float64"))
    # benchmark = market
    bclose = 100.0 * np.exp(market)
    out[benchmark] = DailyPrice(benchmark, idx, pd.Series(bclose, index=idx, dtype="float64"))
    return out
