"""Universe resolution: include/exclude precedence + faith screening."""
from __future__ import annotations

from pytrader.config import Config, FaithScreen
from pytrader.universe import TickerInfo, resolve_universe, screen


def _cfg(**kw) -> Config:
    d = {"include": ["AAA", "BBB", "CCC"], "exclude": []}
    d.update(kw)
    return Config.from_dict(d)


def test_explicit_exclude_wins_over_include():
    cfg = _cfg(exclude=["BBB"])
    cand = {t: TickerInfo(t, t, "Tech") for t in ["AAA", "BBB", "CCC"]}
    sel, rej = resolve_universe(cand, cfg)
    assert [d.info.ticker for d in sel] == ["AAA", "CCC"]
    assert [d.info.ticker for d in rej] == ["BBB"]
    assert rej[0].reasons == ["user-exclude"]


def test_explicit_exclude_wins_over_faith():
    cfg = _cfg(
        exclude=["BBB"],
        faith={"blocked_tickers": ["BBB"]},
    )
    cand = {t: TickerInfo(t, t, "Tech") for t in ["AAA", "BBB", "CCC"]}
    sel, rej = resolve_universe(cand, cfg)
    # user-exclude is reported before faith
    assert rej[0].reasons == ["user-exclude"]


def test_faith_blocked_sector():
    cfg = _cfg(faith={"blocked_sectors": ["GAMBLING"]})
    cand = {
        "AAA": TickerInfo("AAA", "A", "Tech"),
        "BBB": TickerInfo("BBB", "B", "Gambling"),
        "CCC": TickerInfo("CCC", "C", "Tech"),
    }
    sel, rej = resolve_universe(cand, cfg)
    assert [d.info.ticker for d in sel] == ["AAA", "CCC"]
    assert rej[0].reasons == ["faith:sector (Gambling)"]


def test_faith_blocked_ticker():
    cfg = _cfg(faith={"blocked_tickers": ["CCC"]})
    cand = {t: TickerInfo(t, t, "Tech") for t in ["AAA", "BBB", "CCC"]}
    sel, rej = resolve_universe(cand, cfg)
    assert [d.info.ticker for d in rej] == ["CCC"]
    assert rej[0].reasons == ["faith:ticker (user: CCC)"]


def test_faith_disabled_passes_all():
    cfg = _cfg(faith={"enabled": False, "blocked_tickers": ["CCC"]})
    cand = {t: TickerInfo(t, t, "Tech") for t in ["AAA", "BBB", "CCC"]}
    sel, rej = resolve_universe(cand, cfg)
    assert [d.info.ticker for d in sel] == ["AAA", "BBB", "CCC"]
    assert rej == []


def test_not_in_include_list_is_rejected():
    cfg = _cfg(include=["AAA"])
    cand = {"AAA": TickerInfo("AAA", "A", "Tech"),
            "ZZZ": TickerInfo("ZZZ", "Z", "Tech")}
    sel, rej = resolve_universe(cand, cfg)
    assert [d.info.ticker for d in sel] == ["AAA"]
    assert rej[0].info.ticker == "ZZZ"
    assert rej[0].reasons == ["not-in-include-list"]


def test_order_follows_include_list():
    cfg = _cfg(include=["BBB", "AAA", "CCC"])
    cand = {t: TickerInfo(t, t, "Tech") for t in ["CCC", "AAA", "BBB"]}
    sel, _ = resolve_universe(cand, cfg)
    assert [d.info.ticker for d in sel] == ["BBB", "AAA", "CCC"]


def test_screen_is_pure():
    fi = TickerInfo("X", "X", "Tech")
    assert screen(fi, _cfg(faith={"blocked_sectors": ["BET"]})) == []
    assert screen(fi, _cfg(faith={"blocked_sectors": ["Tech"]})) != []
