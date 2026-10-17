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

import pandas as pd


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


def make_store(kind: str = "yfinance", **kw: object) -> PriceStore:
    """Factory so the backend can be swapped from config without imports elsewhere."""
    if kind == "yfinance":
        return YFinanceStore(**kw)  # type: ignore[arg-type]
    raise ValueError(f"unknown price store kind: {kind!r}")
