"""User-managed configuration: universe, faith screening, costs, lookback.

Stored as YAML in the project's data dir (default: <repo>/.pytrader/config.yaml).
This is THE file to hand-edit to add/remove tickers or screening rules.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_HOME = Path(".pytrader")


@dataclass
class FaithScreen:
    """Conservative faith-based screening.

    Rules are deliberately explicit lists of tickers + sectors. They are a
    *screen*, not an endorsement; the investor makes final calls per ticker.
    - blocked_sectors: sectors with heavy exposure to excluded themes
      (gambling, contraception, adult entertainment, etc. per common
      shareholder-guidelines language).
    - blocked_tickers: explicit ticker blocklist (e.g. companies publicly
      opposed to biblical teaching).
    """

    enabled: bool = True
    blocked_sectors: list[str] = field(
        default_factory=lambda: [
            "Gambling",
        ]
    )
    blocked_tickers: list[str] = field(default_factory=list)
    notes: str = (
        "Screener only. Review each candidate against the guidelines you "
        "follow (e.g. Faith & Freedom shareholder guidelines) before investing."
    )


@dataclass
class BacktestDefaults:
    horizon: int = 5            # trading days per rebalance
    top_n: int = 5             # number of holdings
    commission_bps: float = 1.0
    slippage_bps: float = 5.0
    lookback_years: int = 3
    train_fraction: float = 0.7


@dataclass
class Config:
    benchmark: str = "SPY"
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    faith: FaithScreen = field(default_factory=FaithScreen)
    backtest: BacktestDefaults = field(default_factory=BacktestDefaults)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        faith = FaithScreen(**{**FaithScreen().__dict__, **d.get("faith", {})})
        bt = BacktestDefaults(**{**BacktestDefaults().__dict__, **d.get("backtest", {})})
        return cls(
            benchmark=d.get("benchmark", "SPY"),
            include=list(d.get("include", [])),
            exclude=list(d.get("exclude", [])),
            faith=faith,
            backtest=bt,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def default(cls) -> "Config":
        c = cls()
        c.include = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AVGO", "NFLX"]
        c.include += ["CRM", "ADBE", "ORCL", "CSCO", "AMD", "QCOM", "TXN", "MU"]
        c.include += ["XOM", "CVX", "LLY", "UNH", "JNJ", "MRK", "ABBV", "PFE"]
        c.include += ["JPM", "BAC", "V", "MA", "C", "HD", "COST", "WMT"]
        c.include += ["UPS", "CAT", "GE", "HON", "LOW", "TGT", "MCD", "SBUX"]
        c.exclude = ["SMCI", "PLUG", "RIVN", "LCID", "BYND"]  # example exclusions
        return c

    def save(self, home: Path | str = DEFAULT_HOME) -> Path:
        home = Path(home)
        home.mkdir(parents=True, exist_ok=True)
        p = home / "config.yaml"
        p.write_text(self.to_yaml())
        return p

    @classmethod
    def load(cls, home: Path | str = DEFAULT_HOME) -> "Config":
        p = Path(home) / "config.yaml"
        if not p.exists():
            c = cls.default()
            c.save(home)
            return c
        return cls.from_dict(yaml.safe_load(p.read_text()))
