"""Performance metrics from an equity curve (returns already net of costs)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(equity: pd.Series | np.ndarray) -> float:
    eq = np.asarray(equity, dtype="float64")
    return float(eq[-1] / eq[0] - 1.0) if len(eq) > 1 else 0.0


def cagr(equity: pd.Series | np.ndarray) -> float:
    eq = np.asarray(equity, dtype="float64")
    n = len(eq) - 1
    if n <= 0 or eq[-1] <= 0:
        return 0.0
    return float(np.exp(np.log(eq[-1] / eq[0]) / n * TRADING_DAYS) - 1.0)


def ann_vol(returns: pd.Series | np.ndarray) -> float:
    r = np.asarray(returns, dtype="float64")
    if len(r) < 2:
        return 0.0
    return float(np.std(r, ddof=1) * math.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series | np.ndarray) -> float:
    eq = np.asarray(equity, dtype="float64")
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return float(dd.min())


def sharpe(returns: pd.Series | np.ndarray, rf: float = 0.0) -> float:
    r = np.asarray(returns, dtype="float64") - rf / TRADING_DAYS
    if len(r) < 2:
        return 0.0
    sd = np.std(r, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(r) / sd * math.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series | np.ndarray) -> float:
    r = np.asarray(returns, dtype="float64")
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    if len(downside) == 0:
        return 0.0
    sd = np.std(downside, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(r) / sd * math.sqrt(TRADING_DAYS))


def hit_rate(returns: pd.Series | np.ndarray) -> float:
    r = np.asarray(returns, dtype="float64")
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def information_ratio(strategy_ret: pd.Series, bench_ret: pd.Series) -> float:
    active = np.asarray(strategy_ret, dtype="float64") - np.asarray(
        bench_ret, dtype="float64"
    )
    active = active[~np.isnan(active)]
    sd = np.std(active, ddof=1)
    if len(active) < 2 or sd == 0:
        return 0.0
    return float(np.mean(active) / sd * math.sqrt(TRADING_DAYS))


def beta_alpha(strategy_ret: pd.Series, bench_ret: pd.Series) -> tuple[float, float]:
    s = np.asarray(strategy_ret, dtype="float64")
    b = np.asarray(bench_ret, dtype="float64")
    mask = ~(np.isnan(s) | np.isnan(b))
    s, b = s[mask], b[mask]
    if len(s) < 2:
        return (0.0, 0.0)
    cov = np.cov(s, b)
    var = b.var(ddof=1)
    if var == 0:
        return (0.0, 0.0)
    beta = cov[0, 1] / var
    alpha = float(np.mean(s - (beta * b))) * TRADING_DAYS
    return (float(beta), float(alpha))


def summarize(
    equity: pd.Series,
    dates: pd.DatetimeIndex | None = None,
    benchmark_equity: pd.Series | None = None,
) -> dict[str, dict[str, float]]:
    """One-stop metric block for a run."""
    eq = np.asarray(equity, dtype="float64")
    rets = np.diff(eq) / eq[:-1]
    out: dict[str, float] = {
        "total_return": total_return(eq),
        "cagr": cagr(eq),
        "ann_vol": ann_vol(rets),
        "sharpe": sharpe(rets),
        "sortino": sortino(rets),
        "max_drawdown": max_drawdown(eq),
        "hit_rate": hit_rate(rets),
        "n_days": int(len(eq) - 1),
    }
    bench: dict[str, float] = {}
    if benchmark_equity is not None and len(benchmark_equity) > 1:
        beq = np.asarray(benchmark_equity, dtype="float64")
        bret = np.diff(beq) / beq[:-1]
        bench = {
            "total_return": total_return(beq),
            "cagr": cagr(beq),
            "ann_vol": ann_vol(bret),
            "sharpe": sharpe(bret),
            "max_drawdown": max_drawdown(beq),
        }
        if len(bret) >= 2 and len(rets) >= 2:
            bench["information_ratio"] = information_ratio(rets, bret)
            b, a = beta_alpha(rets, bret)
            bench["beta"], bench["alpha"] = b, a
    return {"metrics": {k: _r(v) for k, v in out.items()},
            "benchmark": {k: _r(v) for k, v in bench.items()}}


def _r(v: float) -> float:
    """Round floats for stable JSON output while preserving meaning."""
    if isinstance(v, float):
        return round(v, 6)
    return v
