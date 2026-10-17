"""Persistence layer.

`DB` is the interface. `DuckDBStore` is the local implementation; a
PostgresStore can implement the same interface later. Tables:

  runs(run_id PK, name, created_at, config_json, metrics_json)
  run_equity(run_id, ts, equity, benchmark_equity)
  models(model_id PK, run_id, artifact_path, feature_names_json, created_at)
"""
from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


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


class DuckDBStore(DB):
    def __init__(self, path: Path | str = ".pytrader/pytrader.duckdb") -> None:
        self.path = str(path)
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
