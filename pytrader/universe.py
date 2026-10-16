"""Universe management: explicit include/exclude lists + faith screening.

The output of `resolve_universe` is the single source of truth for which
tickers a model may be trained on. Rejections are always explained with a
reason string (`user-exclude`, `faith:sector`, `faith:ticker`, `no-data`),
so the user can audit exactly why a name was dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pytrader.config import Config


@dataclass
class TickerInfo:
    ticker: str
    name: str = ""
    sector: str = ""


@dataclass
class UniverseDecision:
    info: TickerInfo
    selected: bool
    reasons: list[str] = field(default_factory=list)


def screen(info: TickerInfo, cfg: Config) -> list[str]:
    """Return list of rejection reasons for a candidate. Empty = pass."""
    if not cfg.faith.enabled:
        return []
    reasons: list[str] = []
    if cfg.faith.blocked_sectors and (info.sector or "").strip().lower() in [
        s.lower() for s in cfg.faith.blocked_sectors
    ]:
        reasons.append(f"faith:sector ({info.sector})")
    if info.ticker.upper() in [t.upper() for t in cfg.faith.blocked_tickers]:
        reasons.append(f"faith:ticker (user: {info.ticker})")
    return reasons


def resolve_universe(
    candidates: dict[str, TickerInfo], cfg: Config
) -> tuple[list[UniverseDecision], list[UniverseDecision]]:
    """Split candidates into (selected, rejected) with reasons.

    Explicit exclude always wins, then faith screening, then no-data.
    Include list is a *preference* ordering — everything not excluded and
    passing screen is a candidate.
    """
    included = [t.upper() for t in cfg.include]
    excluded = {t.upper() for t in cfg.exclude}
    order = {t: i for i, t in enumerate(included)}

    selected: list[UniverseDecision] = []
    rejected: list[UniverseDecision] = []
    for ticker, info in candidates.items():
        t = ticker.upper()
        if t in excluded:
            rejected.append(UniverseDecision(info, False, ["user-exclude"]))
            continue
        if cfg.include and t not in order:
            rejected.append(UniverseDecision(info, False, ["not-in-include-list"]))
            continue
        reasons = screen(info, cfg)
        if reasons:
            rejected.append(UniverseDecision(info, False, reasons))
            continue
        selected.append(UniverseDecision(info, True, []))

    selected.sort(key=lambda d: order.get(d.info.ticker, 10_000))
    rejected.sort(key=lambda d: d.info.ticker)
    return selected, rejected
