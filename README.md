# PyTrader

A personal, extensible platform for developing, backtesting, and tracking
XGBoost-based equity strategies — built around three first-class abstractions:

1. **Universe** — the tickers you research and trust (include/exclude lists,
   plus an opt-in faith-based screen for sectors and companies whose business
   or stated goals conflict with your worldview).
2. **Strategies** — a pluggable pipeline you own end-to-end: data → feature
   engineering → model → backtest → metrics → persisted run. Any layer can be
   swapped (data source, backtester, database, model, UI) without touching the
   rest, because each is a small module behind an explicit interface.
3. **Performance** — every run is persisted with a full metric block
   (Sharpe, CAGR, max drawdown, Sortino, hit rate, turnover, benchmark IR,
   beta/alpha, train/validation IC) and viewable side-by-side in a dashboard.

Not an investment product. No live order execution. Everything runs locally.

---

## Quick start

> **Using the build as-is?** See [`USAGE.md`](USAGE.md) for a task-by-task
> walkthrough: collect data, run a regression or binary up/down backtest,
> walk-forward, and query the point-in-time feature store. Everything below
> explains *why it's designed this way*.

```bash
# environment (uv-managed, Python 3.12)
uv sync --extra dev

# macOS: xgboost's native lib needs the LLVM OpenMP runtime (brew, keg-only)
brew install libomp   # or: hermit install openmp  (if you add a package for it)
# the venv's activate script already points DYLD_LIBRARY_PATH at it.

# run the test suite (hermetic — synthetic data, no network)
uv run pytest

# start the dashboard (uses yfinance for prices; .pytrader/config.yaml for the universe)
uv run pytrader-serve --port 8000        # → http://127.0.0.1:8000

# one-shot backtest from the CLI
uv run pytrader-backtest --name demo
```

The config lives at `.pytrader/config.yaml` (created on first use):

```yaml
benchmark: SPY
include: [AAPL, MSFT, ...]      # your watchlist — the research universe
exclude: [SMCI]                 # tickers you've decided to avoid
faith:
  enabled: false
  blocked_sectors: [GAMBLING, ADULT]
  blocked_tickers: []           # e.g. companies whose stated goals you reject
backtest:
  horizon: 5                    # holding period in trading days
  top_n: 5
  commission_bps: 1.0
  slippage_bps: 5.0
  lookback_years: 3
  train_fraction: 0.7           # time-ordered split for the validation metric
```

---

## Architecture

Small, single-purpose modules. Each is either a pure function, a thin
wrapper, or one implementation of an interface — easy to replace.

```
pytrader/
├── config.py     user settings (YAML)
├── data.py       PriceStore ABC · YFinanceStore   ← swap for a paid feed later
├── universe.py   include/exclude + faith screening with human-readable reasons
├── features.py   factors (momentum, vol, RSI, trend) + label generation
├── model.py      XGBoost wrapper, artifact save/load  ← swap for a different model
├── backtest.py   top-N cost-adjusted simulator        ← swap for vectorbt later
├── metrics.py    sharpe/cagr/mdd/sortino/IR/beta/alpha (pure functions)
├── strategy.py   wires the pipeline together (run_strategy)
├── storage.py    DB ABC · DuckDBStore                ← swap for Postgres later
├── server.py     Flask API + static dashboard        ← swap for Django later
├── cli.py        pytrader-serve / pytrader-backtest
├── testing.py    InMemoryStore + synthetic data (used by all tests)
├── static/       dashboard (index.html, app.js, style.css — zero dependencies)
└── tests/        78 hermetic tests (synthetic data; no live API calls)
```

### The pipeline (`run_strategy`)

```
config ─→ universe (include/exclude/faith) ─→ data store.fetch
   ─→ features.compute_panel (X, y, aligned long panel)
   ─→ time-ordered split at train_fraction ─→ XGBoost fit (+ train/valid IC)
   ─→ predict post-split dates ─→ backtest.simulate (top-N, costs)
   ─→ metrics.summarize (+ benchmark) ─→ storage.save_run (duckdb)
```

### Design decisions

- **No lookahead:** labels use only *future* prices (`compute_panel`), and the
  backtest only spends money with prices *after* the decision date. A
  regression test pins the label definition.
- **Costs:** commission + slippage in bps, applied to the fraction of the
  portfolio replaced at each rebalance (one-sided turnover — cheapest
  reasonable for a long-only book).
- **Replaceable interfaces:** `PriceStore`, `DB`, and the model wrapper are
  the three seams where you would later swap in a professional data vendor,
  Postgres, or vectorbt. Nothing outside the seam needs to change.
- **Tests are hermetic:** everything runs on synthetic data through the same
  `run_strategy` code path the API uses. No test hits a live API.

---

## Roadmap (open questions for the owner)

See `DEV_LOG.md` for the running development log and decision record.
Notable undecided items:

- *Walk-forward retraining* — retrain the model on rolling windows instead of
  once at the start. Higher realism for long backtests; modest code cost.
- *Hyperparameter search* with a proper objective (e.g. validation IC) and
  persisted "what we tried" history.
- *Per-position attribution* — which tickers did what, when. Needed before
  we trust the backtest on real data.
- *Faith-screen metadata source* — sectors are currently inferred from
  yfinance's free metadata, which is coarse. A curated list (owned by you)
  is the right long-term tool.
- *More features for the dashboard*: drawdown chart, per-period holdings,
  cost-sensitivity sweep, config-vs-metric scatter.
