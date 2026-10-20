"""Price data layer.

`PriceStore` is the interface everything else uses. Today the implementation
is yfinance; tomorrow it can be a different vendor with no downstream changes.
All prices are *adjusted* close prices (dividend+split adjusted) on the
exchange's trading calendar, one row per date ascending.
"""
from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from pytrader.diagnostics import ErrorDetail, capture

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
                # Chain the underlying yfinance/pandas exception so the full
                # library traceback is reachable via `__cause__` (and visible
                # in `ErrorDetail.cause_chain` + `traceback`).
                raise DataUnavailable(
                    f"no data for {t}"
                    + (f": {last_err}" if last_err else "")
                ) from last_err
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close = df["Close"].dropna()
            if close.empty or close.isna().all():
                raise DataUnavailable(
                    f"no closes for {t} (got {len(df)} rows, columns={list(df.columns)})"
                )
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
            except Exception as e:  # noqa: BLE001 – metadata is best-effort
                # Surface the real error (rate limit, network, schema change)
                # so a failed metadata pull is not silently swallowed.
                logging.getLogger("pytrader.data").warning(
                    "metadata failed for %s: %s: %s",
                    t, type(e).__name__, e, exc_info=e,
                )
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


def batch_download(
    upstream: PriceStore,
    cache: "PriceStorage",
    tickers: list[str],
    start: date,
    *,
    batch_size: int = 25,
    pause_seconds: float = 1.0,
    overwrite: bool = False,
) -> dict:
    """Bulk-load a universe into the price cache with rate limiting.

    Fetches in `batch_size` chunks, pausing `pause_seconds` between
    chunks, so a large universe (hundreds of tickers) is collected without
    tripping Yahoo's per-IP limit.

    Gap intelligence:
      * not in cache at all          -> fetch the full window.
      * in cache, but its earliest
        date is LATER than `start`  -> the user asked for more history than
                                      we have; refetch from `start` and merge
                                      (upsert) so the earlier rows are added.
      * in cache and already covers
        `start`                     -> skip ("cached"), unless `overwrite`.
    So `--lookback-years 5` after `--lookback-years 1` fills the missing
    4 years instead of reporting "already present".

    Returns a summary dict: {fetched, cached_skip, gap_fill, failed,
    n_fetched, n_cached, n_gap, n_failed, failures}.
    """
    if batch_size < 1:
        batch_size = 1
    start_ts = pd.Timestamp(start)
    fetched: list[str] = []
    cached_skip: list[str] = []
    gap_fill: list[str] = []
    failed: list[str] = []
    failures: list[ErrorDetail] = []
    import time as _time

    # Classify each ticker:
    #   todo    -> needs/wants a fetch (never cached, gap at the front, or overwrite)
    #   cached_skip -> already covers `start`; do nothing
    # A ticker in `todo` *that already has data starting after `start`* is a
    # gap-fill; we record it so the summary can report it separately.
    todo: list[str] = []
    for t in tickers:
        tu = t.upper()
        end = cache.end_date(tu)
        if overwrite and end is not None:
            todo.append(tu)
            continue
        if end is None:
            todo.append(tu)
            continue
        # Has data: only refetch if it starts later than what we asked for.
        first = cache.start_date(tu)
        if first is not None and first > start_ts:
            todo.append(tu)
            gap_fill.append(tu)
        else:
            cached_skip.append(tu)

    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        if i > 0 and pause_seconds > 0:
            _time.sleep(pause_seconds)
        for t in chunk:
            try:
                series = upstream.fetch([t], start)
                for s in series:
                    cache.save(s.ticker, s.close)
                meta = upstream.metadata([t]).get(t, {"name": t, "sector": ""})
                cache.save_meta(t, meta)
                fetched.append(t)
            except Exception as e:  # noqa: BLE001 — one bad ticker must not kill the batch
                # Capture the full traceback so the CLI can show it. We keep
                # the ticker in the `failed` list (for the existing shape) and
                # add the structured detail to `failures` (new, opt-in).
                failed.append(t)
                failures.append(capture(e, where=t, context={"start": start.isoformat()}))
    return {
        "fetched": fetched,
        "cached_skip": cached_skip,
        "gap_fill": gap_fill,
        "failed": failed,
        "n_fetched": len(fetched),
        "n_cached": len(cached_skip),
        "n_gap": len(gap_fill),
        "n_failed": len(failed),
        "failures": failures,
    }
