"""Mean-variance basket optimizer for a historical window.

Given daily prices for N tickers over a start→end window, find non-negative
weights that sum to 1 and trade off **gain** against **volatility**. This is the
classical Markowitz / "reward minus risk" problem:

    maximize over w:    return(w)  -  λ · variance(w)
    subject to:         0 ≤ w_i ≤ 1,   Σ w_i = 1

  * ``λ = 0``   → pure gain. The optimizer puts 100% into the single ticker that
                 went up the most over the window. This is the "unsmoothed"
                 extreme the user asked about.
  * ``λ > 0``   → each unit of variance is penalized, so the basket spreads
                 across names to "smooth" the equity curve while still chasing
                 return. Larger λ = smoother, less concentrated portfolio.

Because the objective depends only on the mean daily returns and the daily
covariance matrix, everything here is a pure function of prices: no DB, no
network, deterministic given the input. That makes it trivially testable and
usable from the dashboard or a one-shot CLI call.

Public surface
--------------
- :func:`optimize_window`     optimize one basket for a chosen risk aversion
- :func:`efficient_frontier`  the gain↔volatility trade-off curve ("distribution")
- :func:`daily_returns`       closes → cleaned daily-return matrix
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def daily_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Convert a closes DataFrame into a clean daily-returns matrix.

    ``closes`` has a DatetimeIndex (rows) and one column per ticker. We:

      1. keep only dates present for *every* ticker (inner join on the index,
         dropping NaN rows), so each weight always applies to a full basket;
      2. drop tickers that are constant (zero-vol, e.g. a flat price) — they
         can't diversify and would make the covariance matrix singular;
      3. compute ``pct_change()``.

    Raises :class:`ValueError` when the shared window is too short to optimize.
    """
    if closes is None or closes.shape[0] < 2 or closes.shape[1] < 1:
        raise ValueError("need at least 2 shared dates and 1 ticker to optimize")

    clean = closes.dropna()
    if len(clean) < 2:
        raise ValueError(
            "the selected tickers share fewer than 2 common trading days in "
            "this window — widen the date range or drop tickers with gaps"
        )

    # Drop tickers with near-zero variance (flat/constant prices). A constant
    # column makes the covariance singular and can't contribute to diversification.
    # Use a matrix std so single-ticker input (scalar .std()) is handled too.
    stds = np.asarray(clean.std(ddof=1), dtype="float64").ravel()
    keep = [c for c, s in zip(clean.columns, stds) if np.isfinite(s) and s > 0]
    if not keep:
        raise ValueError("every ticker is flat in this window — nothing to optimize")
    clean = clean[keep]

    rets = clean.pct_change().dropna()
    if len(rets) < 2:
        raise ValueError("window is too short after dropping constant tickers")
    return rets


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def _solve(mu: np.ndarray, cov: np.ndarray, lam: float) -> np.ndarray:
    """Return weights for ``max w·mu - λ·wᵀΣw`` under box + simplex constraints."""
    n = mu.shape[0]
    if n == 1:
        return np.array([1.0])

    cov = 0.5 * (cov + cov.T)  # enforce exact symmetry for the solver
    bnd = [(0.0, 1.0)] * n

    def neg_obj(w: np.ndarray) -> float:
        return float(-(w @ mu - lam * (w @ cov @ w)))

    # Two feasible starting points: greedy max-return and equal weight. Both
    # are feasible, so SLSQP has a valid seed regardless of λ.
    starts = [_onehot(int(np.argmax(mu)), n), np.full(n, 1.0 / n)]

    best_w, best_val = None, np.inf
    for w0 in starts:
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        res = minimize(
            neg_obj, w0, method="SLSQP", bounds=bnd, constraints=cons,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if res.success or (res.x is not None and np.all(np.isfinite(res.x))):
            w = np.clip(np.asarray(res.x), 0.0, 1.0)
            s = w.sum()
            w = w / s if s > 0 else np.full(n, 1.0 / n)
            v = neg_obj(w)
            if v < best_val:
                best_val, best_w = v, w
    if best_w is None:
        best_w = np.full(n, 1.0 / n)  # safe feasible fallback
    best_w = np.clip(best_w, 0.0, 1.0)
    best_w = best_w / best_w.sum()
    return best_w


def _onehot(i: int, n: int) -> np.ndarray:
    """Unit vector of length n with a 1 at position i (0-argmax seed)."""
    w = np.zeros(n)
    w[i] = 1.0
    return w


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class PortfolioResult:
    """A single optimized basket plus its realized (daily-rebalanced) stats."""
    risk_aversion: float                 # the λ used (0 = pure gain)
    tickers: list[str]                   # the names in the basket (kept columns)
    weights: dict[str, float]            # normalized weights, sum ≈ 1
    total_return: float                  # equity[-1]/equity[0] - 1 over the window
    ann_return: float                    # annualized return
    ann_vol: float                       # annualized volatility of daily basket returns
    sharpe: float                        # annualized Sharpe (rf = 0)
    max_drawdown: float                  # deepest peak-to-trough on the basket curve
    n_days: int                          # shared trading days used
    n_positions: int                     # number of tickers with weight > 1e-9

    def to_dict(self) -> dict:
        return {
            "risk_aversion": self.risk_aversion,
            "tickers": self.tickers,
            "weights": self.weights,
            "stats": {
                "total_return": _r(self.total_return),
                "ann_return": _r(self.ann_return),
                "ann_vol": _r(self.ann_vol),
                "sharpe": _r(self.sharpe),
                "max_drawdown": _r(self.max_drawdown),
                "n_days": self.n_days,
                "n_positions": self.n_positions,
            },
        }


def optimize_window(
    closes: pd.DataFrame,
    risk_aversion: float = 0.0,
) -> PortfolioResult:
    """Optimize the weighted basket for a chosen gain↔volatility trade-off.

    Parameters
    ----------
    closes:
        DataFrame of daily closes: rows = dates, one column per ticker.
        Use :func:`daily_returns`-friendly input (the shared window is
        computed inside here, so tickers with gaps are just dropped for the
        common dates).
    risk_aversion:
        ``λ`` — penalty on portfolio variance. ``0`` = pure gain (100% the
        top mover); larger = smoother, more diversified.
    """
    if risk_aversion is None or risk_aversion < 0:
        raise ValueError("risk_aversion must be >= 0 (0 = pure gain)")

    rets = daily_returns(closes)
    tickers = [str(c) for c in rets.columns]
    mu = rets.mean().to_numpy(dtype="float64")
    # ddof=1 sample covariance; guard the n=1 degenerate case.
    cov = rets.cov(ddof=1).to_numpy(dtype="float64")
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])

    w = _solve(mu, cov, float(risk_aversion))

    # Realized stats from the daily-rebalanced constant-weight equity curve.
    port = rets.to_numpy(dtype="float64") @ w
    equity = np.cumprod(1.0 + port)
    n = len(equity)
    total = float(equity[-1] / equity[0] - 1.0) if n > 1 else 0.0
    ann_ret = float(equity[-1] ** (TRADING_DAYS / max(1, n - 1)) - 1.0) if n > 1 and equity[-1] > 0 else 0.0
    sd = float(np.std(port, ddof=1)) if n >= 2 else 0.0
    ann_vol = sd * math.sqrt(TRADING_DAYS)
    sharpe = (float(np.mean(port) / sd) * math.sqrt(TRADING_DAYS)) if sd > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    mdd = float((equity / peak - 1.0).min()) if n else 0.0

    weights = {t: float(wi) for t, wi in zip(tickers, w)}
    return PortfolioResult(
        risk_aversion=float(risk_aversion),
        tickers=tickers,
        weights=weights,
        total_return=total,
        ann_return=ann_ret,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=mdd,
        n_days=max(0, n - 1),
        n_positions=int((w > 1e-9).sum()),
    )


def _min_variance(closes: pd.DataFrame) -> float:
    """Minimum achievable daily variance (used to scale the 0–100 slider)."""
    try:
        rets = daily_returns(closes)
        mu = rets.mean().to_numpy()
        cov = rets.cov(ddof=1).to_numpy()
        if cov.ndim == 0:
            return float(cov)
        lam = 1e12  # effectively pure min-variance
        w = _solve(mu, cov, lam)
        return float(w @ cov @ w)
    except Exception:  # noqa: BLE001
        return 0.0


def efficient_frontier(
    closes: pd.DataFrame,
    steps: int = 12,
) -> dict:
    """The gain↔volatility trade-off curve across a range of risk aversions.

    This is the "distribution" the UI should plot: one point per λ, moving
    smoothly from **max return / high vol** (λ=0, the top single gainer) at
    one end to **min variance / lower return** at the other. Each point carries
    its recommended weights so the caller can show *what* the basket holds.

    ``steps`` = number of points to sample (≥ 2). Risk aversion is swept on a
    geometric scale from ~0 up to the value that fully dominates the return
    term (≈ min-variance), so the whole realistic range is covered.
    """
    steps = max(2, int(steps))
    daily_returns(closes)  # validate once up front; propagate clean error below

    min_var = _min_variance(closes)
    # λ_max ≈ 1/(2·min_daily_var): at this point the variance penalty (≈½)
    # swamps a typical daily return (≈1e-3), so we are at the min-var end.
    lam_max = (1.0 / (2.0 * min_var)) if min_var > 1e-18 else 1e6
    lo = max(lam_max * 1e-4, 1e-9)
    hi = max(lam_max, 1e-9)
    if steps == 2:
        lams = [0.0, float(hi)]
    else:
        # λ=0 corner first, then (steps-1) log-spaced points up to λ_max.
        grid = np.logspace(math.log10(lo), math.log10(hi), steps - 1)
        lams = [0.0] + [float(x) for x in grid]

    points: list[PortfolioResult] = [
        optimize_window(closes, risk_aversion=float(lam)) for lam in lams
    ]
    return {
        "objective": "return − λ·variance (daily rebalanced)",
        "n": len(points),
        "points": [p.to_dict() for p in points],
    }


def _r(v: float) -> float:
    if isinstance(v, float) and math.isfinite(v):
        return round(v, 6)
    return v
