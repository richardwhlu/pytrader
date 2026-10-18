"""Price data layer.

`PriceStore` is the interface everything else uses. Today the implementation
is yfinance; tomorrow it can be a different vendor with no downstream changes.
All prices are *adjusted* close prices (dividend+split adjusted) on the
exchange's trading calendar, one row per date ascending.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pytrader.storage import PriceStorage


@dataclass
class DailyPrice:
    """A single ticker's adjusted daily close series, date-ascending."""

    ticker: str
    index: pd.DatetimeIndex
    close: pd.Series  # float64, aligned to index

    def __post_init__(self) -> None:
        # Canonicalize to a tz-naive, date-only, ascending DatetimeIndex so
        # downstream code (features, backtest, storage) never has to care
        # whether a source handed us exchange-local tz-aware daily stamps.
        idx = pd.DatetimeIndex(self.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)          # keep local wall-clock day
        norm = idx.normalize()                   # snap to that day's date

        # Align close to the original index, then collapse any timestamps that
        # land on the same calendar day, keeping the first close value.
        close = pd.Series(self.close.to_numpy(dtype="float64"), index=norm)
        close = close[~close.index.duplicated(keep="first")]
        close = close[close.notna()].sort_index()
        self.index = close.index
        self.close = close

    @property
    def start(self) -> pd.Timestamp:
        return self.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.index[-1]

    def __len__(self) -> int:
        return len(self.index)

    def daily_returns(self) -> pd.Series:
        adj = self.close
        return adj.pct_change().dropna()


class PriceStore(abc.ABC):
    @abc.abstractmethod
    def fetch(self, tickers: list[str], start: date) -> list[DailyPrice]:
        """Fetch adjusted daily closes for each ticker since `start`.

        Implementations must raise `DataUnavailable` for tickers they cannot
        serve so callers can report them explicitly instead of silently
        dropping rows.
        """

    @abc.abstractmethod
    def metadata(self, tickers: list[str]) -> dict[str, dict[str, str]]:
        """Return {TICKER: {"name": ..., "sector": ...}} for names used in UI."""


class DataUnavailable(RuntimeError):
    pass


class CachedPriceStore(PriceStore):
    """Decorator: serve prices from a `PriceStorage` cache, hitting
    `upstream` only when we don't already have the ticker.

    The stale-refresh heuristic is opt-in (`refresh_after_days`). The
    default (today) is "cache is authoritative" — once a ticker is in the
    cache we never fetch it again until the process restarts. That is the
    rate-limit-safe behavior we want: the first run pays the network cost,
    every subsequent run is a pure cache read.

    Tests wire this over `InMemoryStore` so cache hits/misses are
    observable with zero network calls.
    """

    def __init__(
        self,
        upstream: PriceStore,
        cache: "PriceStorage",
        refresh_after_days: int | None = None,
    ) -> None:
        self._upstream = upstream
        self._cache = cache
        self._refresh_after_days = refresh_after_days

    def fetch(self, tickers: list[str], start: date) -> list[DailyPrice]:
        out: list[DailyPrice] = []
        for t in tickers:
            end = self._cache.end_date(t)
            if end is None:
                # Never cached — fetch full window from upstream and persist.
                for series in self._upstream.fetch([t], start):
                    self._cache.save(series.ticker, series.close)
                    out.append(series)
                continue
            cached = self._cache.load(t)
            needs_refresh = (
                self._refresh_after_days is not None
                and (pd.Timestamp(date.today()).normalize() - end).days
                    >= self._refresh_after_days
            )
            if needs_refresh:
                try:
                    for series in self._upstream.fetch(
                        [t], pd.Timestamp(end).to_pydatetime()
                    ):
                        self._cache.save(series.ticker, series.close)
                    cached = self._cache.load(t)
                except DataUnavailable:
                    # Upstream refused; serve what we have rather than fail.
                    pass
            start_ts = pd.Timestamp(start)
            if start_ts > cached.index[0]:
                idx = cached.index[cached.index >= start_ts]
                cached = cached.loc[idx]
            out.append(DailyPrice(t.upper(), cached.index, cached))
        return out

    def metadata(self, tickers: list[str]) -> dict[str, dict[str, str]]:
        out: dict[str, dict[str, str]] = {}
        for t in tickers:
            cached = self._cache.load_meta(t.upper())
            if cached is not None:
                out[t.upper()] = cached
                continue
            try:
                upstream = self._upstream.metadata([t])[t.upper()]
            except Exception:  # noqa: BLE001 — metadata is best-effort
                upstream = {"name": t, "sector": ""}
            self._cache.save_meta(t.upper(), upstream)
            out[t.upper()] = upstream
        return out




class YFinanceStore(PriceStore):
    """yfinance-backed PriceStore with a small retry on transient failures."""

    def __init__(self, max_retries: int = 2) -> None:
        self._max_retries = max_retries
        self._yf = None

    def _lazy(self):
        if self._yf is None:
            import yfinance as yf  # heavy import, keep out of module load
            self._yf = yf
        return self._yf

    def fetch(self, tickers: list[str], start: date) -> list[DailyPrice]:
        yf = self._lazy()
        out: list[DailyPrice] = []
        for t in tickers:
            df: pd.DataFrame | None = None
            last_err: Exception | None = None
            for _ in range(self._max_retries + 1):
                try:
                    df = yf.download(
                        t, start=start.isoformat(), progress=False,
                        auto_adjust=False, threads=False,
                    )
                    break
                except Exception as e:  # noqa: BLE001 – yfinance raises broadly
                    last_err = e
            if df is None or df.empty:
                raise DataUnavailable(f"no data for {t}: {last_err}")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_levelValues(0)
            close = df["Close"].dropna()
            if close.empty or close.isna().all():
                raise DataUnavailable(f"no closes for {t}")
            close = close.astype("float64")
            # yfinance's auto_adjust=False → use Adj Close if present
            if "Adj Close" in df.columns:
                adj = df["Adj Close"].dropna().astype("float64")
                if not adj.empty:
                    close = adj
            idx = pd.DatetimeIndex(close.index)
            out.append(DailyPrice(t.upper(), idx, close.reset_index(drop=True)))
        return out

    def metadata(self, tickers: list[str]) -> dict[str, dict[str, str]]:
        yf = self._lazy()
        out: dict[str, dict[str, str]] = {}
        for t in tickers:
            try:
                info = yf.Ticker(t).info  # type: ignore[no-untyped-call]
                out[t.upper()] = {
                    "name": str(info.get("longName") or info.get("shortName") or t),
                    "sector": str(info.get("sector") or "").strip(),
                }
            except Exception:  # noqa: BLE001
                out[t.upper()] = {"name": t, "sector": ""}
        return out


def make_store(
    kind: str = "yfinance",
    cache: object | None = None,
    **kw: object,
) -> PriceStore:
    """Factory. When `cache` is a `PriceStorage`, wraps the raw store in a
    `CachedPriceStore` so repeated runs hit the cache instead of the network."""
    if kind == "yfinance":
        raw: PriceStore = YFinanceStore(**kw)  # type: ignore[arg-type]
    else:
        raise ValueError(f"unknown price store kind: {kind!r}")
    if cache is not None:
        return CachedPriceStore(raw, cache)  # type: ignore[arg-type]
    return raw
