"""Persistence layer.

`DB` is the interface for run records. `DuckDBStore` is the local
implementation; a PostgresStore can implement the same interface later.

Tables:

  runs(run_id PK, name, created_at, config_json, metrics_json)
  run_equity(run_id, ts, equity, benchmark_equity)
  models(model_id PK, run_id, artifact_path, feature_names_json, created_at)
  market_prices(ticker, as_of, close)      -- raw price cache
  price_meta(ticker PK, name, sector)      -- cached ticker metadata

`PriceStorage` is the abstract interface the price cache implements:
  load_prices(tickers, start) -> list[(ticker, pd.Series)]
  save_prices(ticker, pd.Series) -> None
  latest_date(ticker) -> pd.Timestamp | None
  load_meta(ticker) -> dict | None
  save_meta(ticker, dict) -> None
"""
from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from pytrader import paths


@dataclass
class RunRecord:
    run_id: str
    name: str
    created_at: str
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


class DB(abc.ABC):
    @abc.abstractmethod
    def save_run(self, rec: RunRecord, equity: pd.DataFrame) -> None:
        """equity: index of unix seconds with columns equity, benchmark_equity."""

    @abc.abstractmethod
    def list_runs(self) -> list[RunRecord]:
        ...

    @abc.abstractmethod
    def get_run(self, run_id: str) -> RunRecord | None:
        ...

    @abc.abstractmethod
    def get_equity(self, run_id: str) -> pd.DataFrame | None:
        ...

    @abc.abstractmethod
    def close(self) -> None: ...


class PriceStorage(abc.ABC):
    """Raw price cache — separate from the run-record DB on purpose.

    Methods:
      end_date(ticker)          -> pd.Timestamp | None
                                   Date of the last cached close, or None.
                                   `CachedPriceStore` uses this to decide
                                   whether to hit upstream.
      start_date(ticker)        -> pd.Timestamp | None
                                   Date of the earliest cached close, or
                                   None. Used for gap-fill (see
                                   CachedPriceStore / batch_download).
      load(ticker)              -> pd.Series
                                   Cached close (full history since first
                                   save). Caller slices to `start` if needed.
      save(ticker, close)       -> None
                                   Upsert (INSERT OR REPLACE on
                                   (ticker, date)); callers can refresh
                                   stale series by re-saving.
      load_meta(ticker)         -> dict[str, str] | None
      save_meta(ticker, meta)   -> None
      tickers()                 -> list[str]
                                   All tickers in the cache (for pickers).
    """

    @abc.abstractmethod
    def end_date(self, ticker: str) -> pd.Timestamp | None: ...

    @abc.abstractmethod
    def start_date(self, ticker: str) -> pd.Timestamp | None: ...

    @abc.abstractmethod
    def load(self, ticker: str) -> pd.Series: ...

    @abc.abstractmethod
    def save(self, ticker: str, close: pd.Series) -> None: ...

    @abc.abstractmethod
    def load_meta(self, ticker: str) -> dict[str, str] | None: ...

    @abc.abstractmethod
    def save_meta(self, ticker: str, meta: dict[str, str]) -> None: ...

    @abc.abstractmethod
    def tickers(self) -> list[str]: ...


class FeatureStorage(abc.ABC):
    """Point-in-time feature cache.

    Long form: one row per (ticker, as_of, feature_name). A `spec_hash`
    scopes rows by the `FeatureSpec` that produced them, so different
    specs coexist in one table. Time-travel is a point query:
    `query(ticker, as_of, spec_hash)` returns that ticker's feature values
    on that date.

    Point-in-time guarantee: a row written for (ticker, D) was computed
    from prices ≤ D — features are trailing/rolling windows, so the store
    never contains a label that used future prices.
    """

    @abc.abstractmethod
    def upsert_panel(self, ticker: str, spec_hash: str,
                     panel: pd.DataFrame) -> int:
        """Bulk-write one ticker's feature panel (columns: date + features).

        Replaces all existing rows for (ticker, spec_hash). Returns the
        number of rows written.
        """

    @abc.abstractmethod
    def query(self, ticker: str, as_of: pd.Timestamp,
              spec_hash: str) -> dict[str, float] | None:
        """Feature values for a single (ticker, date). None if absent."""

    @abc.abstractmethod
    def dates(self, ticker: str, spec_hash: str) -> list[pd.Timestamp]:
        """All dated feature rows for a ticker under a spec, ascending."""

    @abc.abstractmethod
    def spec_features(self, spec_hash: str) -> list[str]:
        """Distinct feature names present under a spec hash (sorted)."""


class DuckDBStore(DB, PriceStorage, FeatureStorage):
    def __init__(self, path: Path | str | None = None) -> None:
        # Default to the canonical, CWD-independent location so every entry
        # point (download CLI, dashboard, backtest) shares one database.
        self.path = str(path) if path else str(paths.db_path())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        import duckdb
        self._con = duckdb.connect(self.path)
        self._init_schema()

    def _init_schema(self) -> None:
        c = self._con
        c.execute("""
            CREATE TABLE IF NOT EXISTS runs(
              run_id VARCHAR PRIMARY KEY,
              name VARCHAR NOT NULL,
              created_at VARCHAR NOT NULL,
              config_json VARCHAR NOT NULL,
              metrics_json VARCHAR NOT NULL
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS run_equity(
              run_id VARCHAR NOT NULL,
              ts BIGINT NOT NULL,
              equity DOUBLE NOT NULL,
              benchmark_equity DOUBLE,
              PRIMARY KEY(run_id, ts)
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS models(
              model_id VARCHAR PRIMARY KEY,
              run_id VARCHAR,
              artifact_path VARCHAR NOT NULL,
              feature_names_json VARCHAR,
              created_at VARCHAR
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS market_prices(
              ticker VARCHAR NOT NULL,
              as_of  DATE NOT NULL,
              close  DOUBLE NOT NULL,
              PRIMARY KEY(ticker, as_of)
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS price_meta(
              ticker VARCHAR PRIMARY KEY,
              name   VARCHAR,
              sector VARCHAR
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS feature_store(
              ticker    VARCHAR NOT NULL,
              as_of     DATE NOT NULL,
              spec_hash VARCHAR NOT NULL,
              feature   VARCHAR NOT NULL,
              value     DOUBLE,
              PRIMARY KEY(ticker, as_of, spec_hash, feature)
            )""")

    def save_run(self, rec: RunRecord, equity: pd.DataFrame) -> None:
        self._con.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)",
            [rec.run_id, rec.name, rec.created_at,
             json.dumps(rec.config), json.dumps(rec.metrics)],
        )
        rows = [
            (rec.run_id, int(ts), float(eq), float(bench) if pd.notna(bench) else None)
            for ts, eq, bench in zip(
                equity.index.as_unit("ns").asi8,
                equity["equity"].astype("float64"),
                equity["benchmark_equity"].astype("float64"),
            )
        ]
        self._con.executemany(
            "INSERT INTO run_equity(run_id, ts, equity, benchmark_equity) "
            "VALUES (?,?,?,?) ON CONFLICT(run_id, ts) DO NOTHING",
            rows,
        )

    def list_runs(self) -> list[RunRecord]:
        r = self._con.execute(
            "SELECT run_id, name, created_at, config_json, metrics_json "
            "FROM runs ORDER BY created_at DESC"
        ).fetchall()
        return [
            RunRecord(rid, name, ca, json.loads(cj), json.loads(mj))
            for rid, name, ca, cj, mj in r
        ]

    def get_run(self, run_id: str) -> RunRecord | None:
        r = self._con.execute(
            "SELECT run_id, name, created_at, config_json, metrics_json "
            "FROM runs WHERE run_id=?", [run_id]
        ).fetchone()
        if r is None:
            return None
        return RunRecord(r[0], r[1], r[2], json.loads(r[3]), json.loads(r[4]))

    def get_equity(self, run_id: str) -> pd.DataFrame | None:
        rows = self._con.execute(
            "SELECT ts, equity, benchmark_equity FROM run_equity "
            "WHERE run_id=? ORDER BY ts", [run_id]
        ).fetchall()
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["ts", "equity", "benchmark_equity"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ns")
        return df.set_index("ts").sort_index()

    def close(self) -> None:
        try:
            self._con.close()
        except Exception:  # noqa: BLE001 - already closed in tests/cleanup
            pass

    # ----------------------------------------------------------------
    # PriceStorage
    # ----------------------------------------------------------------

    def end_date(self, ticker: str) -> pd.Timestamp | None:
        row = self._con.execute(
            "SELECT MAX(as_of) FROM market_prices WHERE ticker=?",
            [ticker],
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return pd.Timestamp(row[0]).normalize()

    def start_date(self, ticker: str) -> pd.Timestamp | None:
        row = self._con.execute(
            "SELECT MIN(as_of) FROM market_prices WHERE ticker=?",
            [ticker],
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return pd.Timestamp(row[0]).normalize()

    def load(self, ticker: str) -> pd.Series:
        rows = self._con.execute(
            "SELECT as_of, close FROM market_prices WHERE ticker=? ORDER BY as_of",
            [ticker],
        ).fetchall()
        if not rows:
            return pd.Series(dtype="float64")
        s = pd.Series([r[1] for r in rows],
                      index=pd.DatetimeIndex([r[0] for r in rows]),
                      dtype="float64")
        s.index.name = None
        return s

    def save(self, ticker: str, close: pd.Series) -> None:
        if len(close) == 0:
            return
        idx = pd.DatetimeIndex(close.index)
        rows = [
            (ticker, d.date(), float(v))
            for d, v in zip(idx, close.to_numpy(dtype="float64"))
            if not (v != v)  # NaN guard
        ]
        if not rows:
            return
        self._con.executemany(
            "INSERT INTO market_prices(ticker, as_of, close) "
            "VALUES (?,?,?) "
            "ON CONFLICT(ticker, as_of) DO UPDATE SET close = EXCLUDED.close",
            rows,
        )

    def load_meta(self, ticker: str) -> dict[str, str] | None:
        row = self._con.execute(
            "SELECT name, sector FROM price_meta WHERE ticker=?", [ticker]
        ).fetchone()
        if row is None:
            return None
        return {"name": row[0] or ticker, "sector": row[1] or ""}

    def tickers(self) -> list[str]:
        """All tickers currently in the price cache, sorted alphabetically.

        Powers the UI's ticker pickers (historical visualizer, universe).
        O(distinct) — one tiny aggregation.
        """
        rows = self._con.execute(
            "SELECT DISTINCT ticker FROM market_prices ORDER BY ticker"
        ).fetchall()
        return [r[0] for r in rows]

    def save_meta(self, ticker: str, meta: dict[str, str]) -> None:
        self._con.execute(
            "INSERT INTO price_meta(ticker, name, sector) VALUES (?,?,?) "
            "ON CONFLICT(ticker) DO UPDATE SET name = EXCLUDED.name, sector = EXCLUDED.sector",
            [ticker, str(meta.get("name", ticker)), str(meta.get("sector", ""))],
        )

    # ----------------------------------------------------------------
    # FeatureStorage
    # ----------------------------------------------------------------

    def upsert_panel(self, ticker: str, spec_hash: str,
                     panel: pd.DataFrame) -> int:
        if panel is None or len(panel) == 0:
            return 0
        feat_cols = [c for c in panel.columns if c not in ("date", "ticker")]
        if not feat_cols or "date" not in panel.columns:
            return 0
        # Isolate the rows we'll persist (dropping fully-missing feature rows).
        wide = panel.loc[:, ["date", *feat_cols]].dropna(how="all").copy()
        if len(wide) == 0:
            return 0
        # Wide → long: one row per (date, feature). Stack with `date` as index
        # yields columns [date, feature, value].
        long = (
            wide.set_index("date")
            .stack()
            .reset_index(name="value")
            .rename(columns={"level_1": "feature"})
        )
        long["value"] = pd.to_numeric(long["value"], errors="coerce").astype("float64")
        long.insert(0, "ticker", ticker.upper())
        long.insert(1, "h", spec_hash)
        long["as_of"] = pd.to_datetime(long["date"]).dt.date
        df = long[["ticker", "as_of", "h", "feature", "value"]]
        # One bulk statement — far faster than a per-cell executemany.
        self._con.execute(
            "INSERT INTO feature_store(ticker, as_of, spec_hash, feature, value) "
            "SELECT ticker, as_of, h, feature, value FROM df "
            "ON CONFLICT(ticker, as_of, spec_hash, feature) "
            "DO UPDATE SET value = EXCLUDED.value",
        )
        return int(len(df))

    def query(self, ticker: str, as_of: pd.Timestamp,
              spec_hash: str) -> dict[str, float] | None:
        d = pd.Timestamp(as_of).normalize().date()
        rows = self._con.execute(
            "SELECT feature, value FROM feature_store "
            "WHERE ticker=? AND as_of=? AND spec_hash=?",
            [ticker, d, spec_hash],
        ).fetchall()
        if not rows:
            return None
        out: dict[str, float] = {}
        for name, v in rows:
            out[name] = None if v is None else float(v)
        return out

    def dates(self, ticker: str, spec_hash: str) -> list[pd.Timestamp]:
        rows = self._con.execute(
            "SELECT DISTINCT as_of FROM feature_store "
            "WHERE ticker=? AND spec_hash=? ORDER BY as_of",
            [ticker, spec_hash],
        ).fetchall()
        return [pd.Timestamp(r[0]) for r in rows]

    def spec_features(self, spec_hash: str) -> list[str]:
        rows = self._con.execute(
            "SELECT DISTINCT feature FROM feature_store WHERE spec_hash=? "
            "ORDER BY feature", [spec_hash]
        ).fetchall()
        return [r[0] for r in rows]
