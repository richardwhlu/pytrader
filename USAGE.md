# PyTrader — Usage Walkthrough (current build)

This is the "how do I actually use it right now" guide. For the *design*
rationale and roadmap see `README.md`; for the decision history see
`DEV_LOG.md`. Everything here runs **locally** and **offline after the first
data pull** — nothing hits a live order book, and tests never touch the network.

---

## What's available now

| Capability | Entry point |
|------------|-------------|
| Collect a universe of prices (rate-limited, cached) | `pytrader-download` |
| Run a backtest (regression *or* binary up/down) | `pytrader-backtest`, `POST /api/backtest`, or `run_strategy(...)` |
| Walk-forward retraining (anchored, no look-ahead) | `run_strategy(walk_forward=True)` |
| Point-in-time feature store ("time-travel") | `FeatureStore.query(ticker, date, spec)` |
| Swap in your own features / model / metrics | `FeatureSpec(custom=...)`, `run_strategy(model=...)`, `register_metric(...)` |
| Dashboard + JSON API | `pytrader-serve` → `http://127.0.0.1:8000` |

The three data artifacts live under `.pytrader/`:
- `config.yaml` — your universe + screen + backtest defaults (hand-edit this).
- `pytrader.duckdb` — one file holding runs, equity curves, model artifacts,
  **price cache**, and the **feature store**.

---

## 0. Setup (once)

```bash
uv sync --extra dev            # create venv + deps (Python 3.12)
uv run pytest                  # 127 hermetic tests, all offline — should be green

# macOS only: xgboost's native lib needs the OpenMP runtime.
brew install libomp            # the venv's activate script already sets DYLD_LIBRARY_PATH
```

If `pytest` is green you have a working, fully-offline install.

---

## 1. Your universe (the config you edit by hand)

`.pytrader/config.yaml` is created on first use with a reasonable default.
Edit it to make it yours:

```yaml
benchmark: SPY                 # performance yardstick
include: [AAPL, MSFT, NVDA, ...]   # the research universe the model may pick from
exclude: [SMCI, PLUG]         # hard block — always wins
faith:                         # opt-in conservative screen (off by default below)
  enabled: false
  blocked_sectors: [Gambling]
  blocked_tickers: []          # companies you've decided to avoid on principle
backtest:
  horizon: 5                   # holding period, in trading days
  top_n: 5                     # number of holdings
  commission_bps: 1.0
  slippage_bps: 5.0
  lookback_years: 3
  train_fraction: 0.7          # time-ordered split used for the validation metric
```

Every rejection (exclude / faith / no-data) is recorded with a human-readable
reason, visible in the dashboard's **Universe** pane.

---

## 2. Collect data (`pytrader-download`)

This pulls the universe from yfinance **once**, into the local DuckDB cache,
chunked with a pause so a big universe doesn't trip Yahoo's per-IP limit.
Re-running is idempotent: cached tickers are skipped, failures are reported
(not raised), so a partial day can be re-issued cheaply.

```bash
# Pull the config's include list, 5 years back, 25 at a time, 1s between chunks.
uv run pytrader-download --lookback-years 5 --batch-size 25 --pause-seconds 1

# Or an explicit list (benchmark always pulled too):
uv run pytrader-download \
    --tickers NVDA,MSFT,GOOGL,AAPL,AMD,INTC \
    --benchmark SPY --lookback-years 5 \
    --batch-size 25 --pause-seconds 1

# First day only? Collect a thin slice to sanity-check the feed:
uv run pytrader-download --tickers NVDA,MSFT,GOOGL --lookback-years 1

# Force a re-fetch of what's already cached (e.g. after a bad day):
uv run pytrader-download --tickers NVDA,MSFT --overwrite
```

You'll get a summary like:

```
fetched :    6   NVDA, MSFT, GOOGL, AAPL, AMD, INTC
cached  :    0  already present
failed  :    0
```

> **Budget note.** Yahoo's free limit is a *per-IP rate limit* (thousands of
> calls/day before a 429), not a hard 200/batch cap. The cache is what makes
> this sustainable: the first day pays the network cost, and every run after
> that is a pure cache read.

---

## 3. Run a backtest

### 3a. One-shot CLI (regression, the default)

```bash
uv run pytrader-backtest --name demo
```

Prints `run_id`, the full metric block, top-5 features, and the data span.
The run is persisted to DuckDB, so it's immediately selectable in the dashboard.

### 3b. In the dashboard

```bash
uv run pytrader-serve --port 8000    # → http://127.0.0.1:8000
```

- **New Backtest** form → `POST /api/backtest` with `{horizon, top_n, commission_bps, slippage_bps, name}`.
- **Performance** pane → equity vs benchmark, metric block, side-by-side compare (up to 10 runs).
- **Universe** pane → selected/rejected tickers and *why*.

### 3c. From Python (where the new features live)

`run_strategy` is the single entry point. `store`, `cfg`, `db` are the wiring;
the flags on the right turn on each new feature. `task`, `walk_forward`, and
`retrain_every` are keyword-only.

```python
from pathlib import Path
from pytrader.config import Config
from pytrader.data import make_store
from pytrader.feature_store import FeatureStore
from pytrader.storage import DuckDBStore
from pytrader.strategy import run_strategy

cfg  = Config.load()                                   # .pytrader/config.yaml
db   = DuckDBStore(Path(".pytrader") / "pytrader.duckdb")
store = make_store("yfinance", cache=db)               # cache-over-network
fs   = FeatureStore(db)                                 # point-in-time store

# (1) Regression: predict the horizon log return. Reports IC.
res = run_strategy(store, cfg, db=db, name="reg", feature_store=fs)

# (2) Binary: predict "close higher `horizon` ahead". Reports AUC + top-K precision.
res = run_strategy(store, cfg, db=db, name="bin", feature_store=fs, task="binary")

# (3) Walk-forward: retrain on an anchored window at each step (no look-ahead).
res = run_strategy(store, cfg, db=db, name="wf",
                   task="regression", walk_forward=True, retrain_every=1)
```

Each call returns a `PipelineResult` with `.run_id`, `.name`, `.metrics`,
`.config`, `.feature_importance`, and `.result` (the equity/holdings record).

### 3d. Your own features

```python
from pytrader.features import FeatureSpec

# Add a Bollinger-band-width factor alongside the defaults:
def bb_width(close, n=20):
    mid = close.rolling(n).mean(); sd = close.rolling(n).std()
    return (close - mid) / sd.replace(0, float("nan"))

spec = FeatureSpec(custom={"bb_width": bb_width})
res = run_strategy(store, cfg, db=db, name="custom", features=spec)
```

### 3e. Your own model / metrics

```python
import pytrader.metrics as M

# Custom metrics receive (equity, dates, periods) → float, and land in every run's
# metric block under metrics["metrics"]:
def excess_up(equity, dates, periods):
    ...
    return 0.62
M.register_metric("excess_up", excess_up)

res = run_strategy(store, cfg, db=db, name="mymodel", model=MyModel())  # implement fit/predict
```

> **`spec_hash` caveat.** The feature store is partitioned by the *shape* of
> the spec (windows, RSI, custom factor **names**). Two specs whose custom
> factors share a name but differ in behavior will collide — the name *is*
> the feature's identity in the matrix. Keep custom factor names unique.

---

## 4. Point-in-time "time-travel"

This is the feature store's reason to exist: once a run has persisted a
ticker's features, any historical `(ticker, date)` is a cheap DB read — no
recompute, no network, and it's the *exact* matrix the model trained on.

```python
import pandas as pd
from pytrader.features import FeatureSpec

spec = FeatureSpec()                 # must match the spec the run used
as_of = pd.Timestamp("2024-06-03")
vec = fs.query("NVDA", as_of, spec)   # dict[str, float] | None
# → {'momentum_21': 0.031, 'volatility_21': 0.34, 'rsi_14': 57.2, 'trend_21': 1.1}

fs.has("NVDA", spec)                  # is anything cached for this ticker/spec?
fs.dates("NVDA", spec)                # every cached date for it
fs.load_panel("NVDA", spec)           # full wide DataFrame back
```

Because the cache and the model come from the *same* `save_panel` call inside
`run_strategy`, the value you query for a past date is guaranteed to be what
the model saw in the backtest — that's the no-drift property.

---

## 5. Metric reference

Every run records a `metrics` block. The two tasks emphasize different
signals on the same backtest engine:

| Metric | Regression (`task` default) | Binary (`task="binary"`) |
|--------|:---:|:---:|
| Sharpe, Sortino, CAGR, Max DD, Ann vol | ✅ | ✅ |
| Total return, Hit rate, Avg turnover, Benchmark IR | ✅ | ✅ |
| Beta / Alpha | ✅ | ✅ |
| **Train IC / Valid IC** | ✅ (rank correlation to log return) | `0.0` (uninformative on a 0/1 target) |
| **Val AUC** | — | ✅ (discrimination of up vs down) |
| **Val top-K precision** | — | ✅ (how often the top-N actually went up) |
| Val base rate | — | ✅ (P(up) in the validation window — the baseline to beat) |

Reading a binary run: compare `val_top{N}_precision` against `val_base_rate`.
If the model's top picks go up *less* often than the base rate, the AUC may
still be >0.5 (rank order is informative) but the *selection* isn't beating
do-nothing. That's the signal worth chasing.

---

## 6. API surface (for scripting the dashboard)

```
GET  /api/health                  → liveness
GET  /api/universe                → selected/rejected tickers + reasons
GET  /api/runs                    → all saved runs (most recent first) + key metrics
GET  /api/runs/<run_id>           → full run: metrics, config, equity + benchmark series
GET  /api/compare?run_ids=a,b     → up to 10 runs, equity + metrics side-by-side
POST /api/backtest                → {horizon, top_n, commission_bps, slippage_bps, name}
```

Example:

```bash
curl -s localhost:8000/api/health
curl -s -X POST localhost:8000/api/backtest \
     -H 'content-type: application/json' \
     -d '{"horizon": 21, "top_n": 5, "commission_bps": 1, "slippage_bps": 5, "name": "monthly"}'
```

---

## 7. Typical day-to-day loop

```bash
# once a day (or per new universe)
uv run pytrader-download --lookback-years 5          # fill/refresh the cache

# explore
uv run pytrader-backtest --name daily                 # regression baseline
uv run pytrader-serve --port 8000                     # open dashboard, compare runs,
                                                      # flip horizon/top_n in the form
# deeper (Python)
uv run python -c "from pytrader.strategy import run_strategy; ..."
```

---

## Known limitations / gotchas

- **No live trading.** This is a research/backtest tool; outputs are not orders.
- **Binary IC is 0 by design** — use AUC / top-K precision instead.
- **Walk-forward `valid_ic` = `train_ic` today** (no per-fold holdout yet); the
  per-fold IC series is the next improvement.
- **`spec_hash` collision** on same-named-but-different custom factors (see §3d).
- **Faith screen metadata** is coarse (yfinance sectors); the owner-curated
  `faith_blocklist.yaml` is the intended long-term source.
- **Rate limits** are per-IP and soft; the cache is what keeps re-runs free.

---

## Open (owner to decide)

- Faith blocklist file contents (`faith_blocklist.yaml`).
- Custom-metric namespacing (`"sharpe"` override vs forced `custom_` prefix).
- Walk-forward proper holdout / per-fold IC series vs today's `valid_ic = train_ic`.
- A named "6-month" binary preset vs passing `horizon` by hand.
