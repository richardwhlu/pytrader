"""Doctor / migration: find stray pytrader databases, merge them into the
canonical one.

Why this exists: pytrader used to resolve its data dir as a *relative*
path (`.pytrader/pytrader.duckdb`), so running the download CLI from the
package directory and the dashboard from the repo root produced two
different databases — the CLI wrote to `pytrader/.pytrader/` while the
server read from `.pytrader/`. The doctor finds any of these orphans and
can merge their tables (upsert-safe) into the canonical database under
the current data home.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import pandas as pd

from pytrader import paths


# Known places the pre-fix code could have left a DB, in priority order.
# Order matters: we scan both possible "stray" locations explicitly.
_LEGACY_LOCATIONS: list[Path] = [
    Path(".pytrader") / "pytrader.duckdb",
    Path("pytrader") / ".pytrader" / "pytrader.duckdb",
]

# Tables that hold user data and should be merged. We copy them in this
# order purely for readability; there are no cross-table FK constraints
# in DuckDB to worry about.
_MERGE_TABLES = ["price_meta", "market_prices", "feature_store", "models",
                 "runs", "run_equity"]


def _table_names(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()
    return {r[0] for r in rows}


def scan_orphans() -> list[Path]:
    """Return legacy DB paths (that exist but are NOT the canonical one)."""
    canonical = paths.db_path().resolve()
    out: list[Path] = []
    for p in _LEGACY_LOCATIONS:
        if not p.exists():
            continue
        abs_p = p.resolve()
        if abs_p == canonical:
            continue
        out.append(abs_p)
    return out


def migrate(orphan: Path, *, to: Path | None = None, dry_run: bool = False) -> dict:
    """Merge one orphan DB into the canonical (or `to`) database.

    Idempotent: inserts use ON CONFLICT DO NOTHING, so running twice doesn't
    double data. Only tables that exist in both and have rows are touched.

    Returns: {source, destination, tables: {table: rows_in_source}, dry_run}
    """
    dest_path = (to or paths.db_path()).resolve()
    src = duckdb.connect(str(orphan.resolve()), read_only=True)
    # Dry runs only read from the destination (to enumerate tables / counts),
    # so open it read-only — safe even while the running server holds a
    # write lock on the same file. A real merge needs a write handle.
    dst = duckdb.connect(str(dest_path), read_only=dry_run)
    moved: dict[str, int] = {}
    try:
        src_tabs = _table_names(src)
        dst_tabs = _table_names(dst)
        common = [t for t in _MERGE_TABLES if t in src_tabs and t in dst_tabs]

        for tbl in common:
            n = int(src.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0])
            if n == 0:
                continue
            if dry_run:
                moved[tbl] = n
                continue
            df: pd.DataFrame = src.execute(f'SELECT * FROM "{tbl}"').df()
            if df.empty:
                continue
            # Bind the df as a virtual table on dst, then upsert.
            # INSERT ... SELECT * FROM df respects column position/order;
            # our table schemas are ours and stable, so this is safe.
            dst.execute(
                f'INSERT INTO "{tbl}" SELECT * FROM df ON CONFLICT DO NOTHING'
            )
            del df  # let pandas release it before the next table
            moved[tbl] = n
        return {
            "source": str(orphan),
            "destination": str(dest_path),
            "tables": moved,
            "dry_run": dry_run,
        }
    finally:
        src.close()
        dst.close()


def doctor(*, dry_run: bool = False) -> dict:
    """Diagnose the local data layout and (if there's exactly one orphan)
    produce a migration preview without writing.

    Returns: {canonical, orphans: [path...], merge_preview: {...} | None}
    """
    canonical = paths.db_path().resolve()
    orphans = scan_orphans()
    preview = None
    if len(orphans) == 1:
        preview = migrate(orphans[0], dry_run=True)
    return {
        "canonical": str(canonical),
        "orphans": [str(o) for o in orphans],
        "merge_preview": preview,
    }


def archive(paths_to_move: list[Path], archive_dir: Path) -> list[Path]:
    """After a successful merge, move originals into `archive` so the user
    can verify the dashboard works before deleting. Not deleted.
    """
    archive_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for p in paths_to_move:
        if not p.exists():
            continue
        target = archive_dir / p.name
        i = 1
        while target.exists():
            target = archive_dir / f"{p.stem}-{i}{p.suffix}"
            i += 1
        shutil.move(str(p), str(target))
        out.append(target)
    return out
