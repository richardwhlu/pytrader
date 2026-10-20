"""Canonical, CWD-independent location for all persistent pytrader state.

Why this exists: pytrader used to hardcode the *relative* path
``.pytrader/pytrader.duckdb`` in several modules. That resolves against the
current working directory, so running the download CLI from the ``pytrader/``
package dir wrote to ``pytrader/.pytrader/`` while the dashboard (run from the
repo root) read ``.pytrader/`` — two different databases. Everything that
needs the data dir now calls :func:`data_home` (or :func:`db_path`) so every
entry point agrees on one file no matter where it's launched from.

Resolution order for :func:`data_home`:
  1. ``$PYTRADER_HOME`` if set (lets tests and the user point it anywhere),
  2. otherwise ``<repo root>/.pytrader`` (the documented default).
"""
from __future__ import annotations

import os
from pathlib import Path

# pytrader/<thisfile>.py -> parent is the package dir -> parent is the project.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def data_home() -> Path:
    """Absolute path to pytrader's data directory (created on demand)."""
    env = os.environ.get("PYTRADER_HOME")
    home = Path(env) if env else _PROJECT_ROOT / ".pytrader"
    return home


def db_path() -> Path:
    """Absolute path to the DuckDB file every entry point shares."""
    return data_home() / "pytrader.duckdb"


def models_dir(run_id: str) -> Path:
    """Directory for a run's saved model artifacts."""
    return data_home() / "models" / run_id
