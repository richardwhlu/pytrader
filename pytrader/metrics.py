"""Performance metrics from an equity curve (returns already net of costs).

Built-in metrics are pure functions below. To optimize against your own
metric, either:

  * register it once via ``register_metric`` and it will be computed and
    stored with every run, or
  * compute it yourself from ``SimulationResult.periods`` / ``equity`` and
    pass it through ``summarize(..., custom={name: value})``.

This keeps the metric set open-ended without touching the engine.
"""
from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# A custom metric is given the equity curve, its dates, and (optionally) the
# list of holding periods, and returns a single finite float.
MetricFn = Callable[..., float]
_CUSTOM: dict[str, MetricFn] = {}


def register_metric(name: str, fn: MetricFn | None = None) -> MetricFn:
    """Register a named custom metric so every run reports it.

    ``fn(equity, dates, periods) -> float``. Supports both styles::

        M.register_metric("max_loss", my_fn)
        @M.register_metric("max_loss")
        def my_fn(equity, dates, periods): ...

    Re-registering the same name overwrites it.
    """
    if fn is not None:
        _CUSTOM[name] = fn
        return fn

    def deco(f: MetricFn) -> MetricFn:
        _CUSTOM[name] = f
        return f

    return deco  # type: ignore[return-value]


def clear_custom_metrics() -> None:
    _CUSTOM.clear()


def custom_metrics(equity, dates, periods) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, fn in _CUSTOM.items():
        try:
            v = float(fn(equity, dates, periods))
            if math.isfinite(v):
                out[name] = v
        except Exception:  # noqa: BLE001 — a bad custom metric must not kill a run
            continue
    return out


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


def classification_report(
    y_true: Sequence[float] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    top_k: int = 5,
) -> dict[str, float]:
    """Discrimination block for the binary task.

    AUC (ranking quality) and precision_at_k ("of the top-k picks, how
    often are they actually up?"). These are the right summary stats for a
    0/1 "up in 6 months" target — in-sample IC on a binary target is
    uninformative.
    """
    return {
        "auc": roc_auc(y_true, y_score),
        f"precision_at_{int(top_k)}": precision_at_k(y_true, y_score, k=top_k),
        "n_pos": int(np.asarray(y_true).sum()),
        "base_rate": float(np.asarray(y_true).mean()),
    }


def roc_auc(y_true: Sequence[float] | np.ndarray,
            y_score: Sequence[float] | np.ndarray) -> float:
    """Area under the ROC curve (no external deps — computed by ranking).

    `y_score` is the model's continuous confidence (e.g. P(up)); `y_true`
    is the 0/1 label. Returns 1.0 for perfect ranking, 0.5 for random.
    Degenerate inputs (no positive or no negative) return 0.5 (undefined).
    """
    yt = np.asarray(y_true, dtype="float64")
    ys = np.asarray(y_score, dtype="float64")
    m = ~(np.isnan(yt) | np.isnan(ys))
    yt, ys = yt[m], ys[m]
    pos = (yt == 1)
    if pos.sum() == 0 or (~pos).sum() == 0:
        return 0.5
    # Rank-based AUC (Mann–Whitney U): P(score(pos) > score(neg)), with
    # 0.5 credit for ties.
    order = np.argsort(ys, kind="mergesort")
    ranks = np.empty(len(ys), dtype="float64")
    ranks[order] = np.arange(1, len(ys) + 1)
    # Ties get the mean rank.
    sorted_scores = ys[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i:j + 1]].mean())
            ranks[order[i:j + 1]] = avg
        i = j + 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    rank_sum_pos = ranks[pos].sum()
    # Mann–Whitney U statistic
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    auc = u / (n_pos * n_neg)
    return float(auc)


def precision_at_k(y_true: Sequence[float] | np.ndarray,
                   y_score: Sequence[float] | np.ndarray,
                   k: int = 5) -> float:
    """Fraction of the top-k scored names that are actually positive.

    The most decision-relevant metric for this product — "when we pick the
    top 5, how often are they up?"  Falls back to the base rate when there
    are fewer than k scored rows.
    """
    yt = np.asarray(y_true, dtype="float64")
    ys = np.asarray(y_score, dtype="float64")
    m = ~(np.isnan(yt) | np.isnan(ys))
    yt, ys = yt[m], ys[m]
    if len(ys) == 0 or (yt == 1).sum() == 0:
        return 0.0
    k = max(1, min(int(k), len(ys)))
    top_k = np.argsort(ys)[::-1][:k]
    return float((yt[top_k] == 1).mean())


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
    periods: "Sequence[object] | None" = None,
    custom: Mapping[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """One-stop metric block for a run.

    Registered custom metrics (see ``register_metric``) are always computed
    from ``(equity, dates, periods)`` and folded into ``metrics``. ``custom``
    lets a caller supply pre-computed values, overriding same-named entries.
    """
    merged_custom = dict(custom_metrics(equity, dates, periods))
    if custom:
        merged_custom.update({k: float(v) for k, v in custom.items() if math.isfinite(float(v))})

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
    out.update(merged_custom)
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
