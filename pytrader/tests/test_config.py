"""Config round-trip + defaults sanity."""
from __future__ import annotations

from pathlib import Path

import pytest

from pytrader.config import Config


def test_default_is_nonempty_universe():
    cfg = Config.default()
    assert len(cfg.include) >= 5
    assert "SMCI" in cfg.exclude  # example exclusion ships by default


def test_yaml_roundtrip(tmp_path: Path):
    cfg = Config.default()
    cfg.include = ["AAPL", "MSFT"]
    cfg.faith.blocked_tickers = ["NVDA"]
    cfg.backtest.top_n = 7
    p = cfg.save(tmp_path)
    assert p.exists()
    loaded = Config.load(tmp_path)
    assert loaded.include == cfg.include
    assert loaded.faith.blocked_tickers == cfg.faith.blocked_tickers
    assert loaded.backtest.top_n == 7


def test_case_preserved_but_resolution_case_insensitive(tmp_path: Path):
    cfg = Config.from_dict({"include": ["aapl"], "exclude": ["smci"]})
    cfg.save(tmp_path)
    loaded = Config.load(tmp_path)
    assert loaded.include == ["aapl"]
    assert loaded.exclude == ["smci"]
    # resolution is case-insensitive
    from pytrader.universe import TickerInfo, resolve_universe
    cand = {t: TickerInfo(t, t, "Tech") for t in ["AAPL", "SMCI"]}
    sel, rej = resolve_universe(cand, loaded)
    assert [d.info.ticker for d in sel] == ["AAPL"]
    assert [d.info.ticker for d in rej] == ["SMCI"]


def test_unknown_key_is_ignored(tmp_path: Path):
    import yaml
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({"weird": 1, "benchmark": "QQQ"}))
    cfg = Config.load(tmp_path)
    assert cfg.benchmark == "QQQ"


def test_save_returns_path(tmp_path: Path):
    p = Config.default().save(tmp_path)
    assert p.name == "config.yaml"
