# DEV_LOG — pytrader

Running development log. Newest entries at the bottom.
Open decisions, assumptions, and things "I" (the assistant) was unsure about
are marked **[?]** — review these first.

---

## 2026-10-16

### Environment
- uv-managed venv, Python 3.12.13. Project deps: xgboost, pandas, numpy,
  duckdb, flask, yfinance, scikit-learn, pyyaml.
- **macOS quirk:** the xgboost wheel needs `libomp.dylib`. Brew's `libomp`
  is keg-only, so I added its lib dir to `.venv/bin/activate`
  (`DYLD_LIBRARY_PATH`). If you use a different shell setup, that export
  will need to happen somewhere visible to your shell.
  **[?]** you asked for hermit too; I initialized it in the repo and installed
  node via it for JS syntax checking, but hermit has no `libomp` package, so
  brew remains the source. If you want hermit as the canonical tool source,
  we can add a custom hermit manifest pointing at the LLVM tarball.

### Bugs found & fixed (each now has a regression test)
1. `testing.py` — synthetic series built a `pd.Series` with a RangeIndex,
   silently dropping the dates entirely (features then trained on int labels).
   Caught by the first end-to-end run.
2. `backtest.py::_validate_predictions` — `dec.reindex(prices.columns)`
   reindexed **rows**, collapsing the date frame to 8 ticker rows. Fixed to
   `reindex(columns=...)`. Locked in by `test_simulate_equity_starts_at_one`.
3. `backtest.py::_price_table` — after popping the benchmark I unconditionally
   reset it to `None`, so the benchmark curve never survived. Locked in by
   `test_simulate_benchmark_aligned_to_equity`.
4. `strategy.py` — equity was persisted with the index pre-scaled to
   `ns // 1e9` ("seconds"), so DuckDB read-back corrupted all dates
   (`2025-03-12` came back as `1970-01-21`). The dashboard would render wrong
   x-axes if not for the API tests. Fixed: store int64 ns on the way in,
   `pd.to_datetime(..., unit="ns")` on the way out; `strategy` now passes a
   real `DatetimeIndex`. Regression: `test_equity_index_roundtrips_across_resolutions`.
5. `model.py` — used a non-existent `verbose=` kwarg (xgboost API changed).
   Also added a `feature_names()` accessor the API depends on.
6. `strategy.py` — `Timestamp.utcnow()` is deprecated in this pandas version.
   Switched to `Timestamp.now(tz="UTC")`.

### Design decisions (recorded for you)
- **Top-N long-only, equal-weight** backtest. No shorting, no leverage, no
  partial trades — this is the simplest book where "cost = fraction of book
  replaced × bps" is defensible. A more realistic engine (per-position
  PnL, intraday marks) is vectorbt territory and is the planned replacement
  at the `backtest.py` seam.
- **Time-ordered split, not k-fold.** For time series, k-fold leaks future
  info. We split at `train_fraction` of dates; the model sees only the past.
  Reported as `train_ic` / `valid_ic` for transparency.
- **One model per run, trained once.** Live-quality backtests should retrain
  every few weeks (walk-forward). Not implemented yet; see roadmap.
  **[?]** do you want walk-forward now, or is it OK to defer while we prove
  the rest of the stack with synthetic data?
- **Synthetic data signal strength:** I calibrated the test data so a
  perfect model earns ~80% over the test window (top-decile territory). Too
  weak and bugs hide in the noise; too strong and we can't tell bugs from
  alpha. **[?]** flag me if you want a different calibration.
- **`benchmark_ir`, `beta`, `alpha`** are computed against the chosen
  benchmark (SPY by default). Long-only top-N will look low-beta by design
  — that's the point of the screen.

### What's working end-to-end now
- `pytrader-backtest --name demo` → trains XGBoost on the config universe,
  simulates ~20+ rebalance periods, saves to `.pytrader/pytrader.duckdb`.
- `pytrader-serve` → single-page dashboard: runs table with per-run
  Sharpe/CAGR/MDD/IR, checkbox compare mode that renders a shared canvas
  chart + a metric-diff table, detail view, and the selected/rejected
  universe pane with human-readable rejection reasons (e.g.
  `faith:sector (Gambling)`, `user-exclude`, `not-in-include-list`).
- 78 test suite, all green, no network, ~5 s.

### What I was unsure about (open questions)
- **[?] Faith screen granularity:** I use yfinance's free `sector`/`industry`
  strings, which are coarse (e.g. "Consumer Cyclical" hides a lot). A curated
  `faith_blocklist.yaml` you own is the right shape; I'm happy to draft it
  from a starter list you review — I don't want to guess at what you reject
  on principle.
- **[?] Universe size:** the `include` list ships with ~40 large caps.
  YFinance's free tier works fine in batches, but if you go past ~200
  tickers per batch we'll hit rate limits. The seam for a real data vendor
  is `pytrader/data.py::PriceStore`; nothing else moves.
- **[?] DuckDB file location:** I put it at `.pytrader/pytrader.duckdb`
  (project-local, git-ignored). If you later want a shared Postgres copy,
  `storage.py::DB` is the seam.
- **[?] `random_state` default** is currently 42 everywhere — fine for
  reproducible dev runs, but I'd suggest a CLI `--seed` override before we
  trust any single number out of it.

## 2026-10-17

### Customizability seams — landed
Three independent extensions, each with its own regression, committed as
two logical slices:

1. **`features.py` — custom callable factors.** `FeatureSpec(custom={name: fn})`
   lets callers register their own `(close: pd.Series) -> pd.Series` factor
   without touching the built-in set. `column_names()` and `compute()` both
   extend for it; `strategy.run_strategy(features=spec)` passes it straight
   through. Committed as `139ff95`.
2. **`metrics.py` — custom metric registry.** `register_metric(name, fn)` or
   the decorator form. `summarize(...)` now takes `periods` and `custom`,
   folds registered metrics through `custom_metrics(equity, dates, periods)`
   and caller-supplied `custom` overrides a same-named entry by name. Broken
   metrics are swallowed (a metric must not kill a run). Committed with slice 2.
3. **`strategy.py` — swappable model + features param.** `run_strategy`
   accepts `model=` (anything with `fit/predict/feature_names`; must raise
   `TypeError` if it doesn't) and `features=`. `feature_importance` is
   guarded by `hasattr` so non-XGBoost models without it don't crash the
   pipeline. Committed as `595b4e3`.

- 97 tests green.
- **Why the stream was noisy before:** I was landing all three in one
  pass and the tool output was interleaving in ways you saw as errors.
  Split into two slices; each one is independently verifiable.

### What's next (ordered, each one small)
- `[done]` `walk-forward` flag on `run_strategy`: `run_strategy(...,
  walk_forward=True, retrain_every=5)`. The pure function
  `pytrader.strategy.walk_forward_predict` is independently testable
  (no-leak, fold growth, NaN-y exclusion). `min_train_rows` is derived
  from `train_fraction`, no new config surface. Committed `77f1299`.
- `[done]` Data + feature store + binary task — see the later
  **2026-10-17** section (tasks 1–5): price cache, point-in-time feature
  store, `task="binary"` predictor, run wiring, and `pytrader-download`.
- `[next]` Faith blocklist: still the `faith_blocklist.yaml` you own.
- **[?] `summarize(custom=...)` and `register_metric` both write into the
  same `out` dict. If you register "sharpe" as a custom name it overrides
  a built-in. Do we want a namespacing rule (custom names prefixed
  `custom_`) or is override-intended the right default?**
- **[? Walk-forward `valid_ic`:** currently `valid_ic = train_ic` because
  there is no out-of-fold split by design. Add a proper `valid_ic`
  (fold's *next-step* IC as a sanity check) or a `fold_ic` series in the
  result? **]
- **[?] Binary 6-month horizon:** the binary task uses `pipeline.horizon`
  as the "ahead" window. To mean "6 months" set `horizon=126` (≈6 trading
  months) when calling. Do we want a named preset
  (`task="binary_6m"`) or is raw `horizon=126` clearer? **]
## 2026-10-17 — Data + feature store + binary task (tasks 1–5)

The owner asked to: collect real data, generate features, build a
lightweight **time-travel feature store**, start training, and add a
coarse **"will this be up in 6 months"** binary prediction (smooths macro
shocks, more robust than raw RMSE on noisy daily labels). Began on the
question of whether the yfinance *free budget* (per-IP rate limit, not a
hard 200/batch cap) needed orchestrating. All five tasks landed, each
committed and verified independently. **127 tests green (up from 97).**

| # | Task | What shipped | Commit |
|---|------|--------------|--------|
| 1 | Price cache | `PriceStorage` ABC; `DuckDBStore` implements it; `CachedPriceStore` decorator serves from cache, hits upstream only on miss (or stale if `refresh_after_days` set). 2nd+ run = pure cache read → rate-limit safe. | `312b62e` |
| 2 | Feature store | `feature_store` table (long: ticker, as_of, spec_hash, feature, value); `FeatureSpec.spec_hash()` scopes partitions; `FeatureStore.save/load/query` → point-in-time reads. | `35bb612` |
| 3 | Binary task | `binary_updown` label; `XGBoostBinaryModel` (P(up) output); `roc_auc` + `precision_at_k` (rank-based, no deps); `run_strategy(task="binary")` reports AUC + top-K precision instead of IC. | `8365ba2` |
| 4 | Wiring | `run_strategy(feature_store=...)` persists the *exact* trained matrix (no drift); `upsert_panel` rewritten as one bulk `INSERT…SELECT` (was per-cell → 17ms now). Server + CLI populate the store. | `fcf682b` |
| 5 | Batch download | `batch_download()` chunked rate-limited loader; `pytrader-download` CLI. Idempotent: cached tickers skipped, failures reported not raised. | `b9547b6` |

### How the pieces compose (the workflow you asked for)
```
pytrader-download --tickers NVDA,MSFT,... --lookback-years 5   # task 5: cache prices
    → runs populate price cache (task 1)
run_strategy(..., feature_store=FeatureStore(db))               # task 4: fill feature store
    → point-in-time features (task 2), retrievable for any historical date
run_strategy(..., task="binary")                                 # task 3: up/down predictor
    → reports AUC + P@k, backtest unchanged (scores still "higher = more likely up")
```

### Design notes / decisions
- **Binary `predict()` returns P(up), not a 0/1 class.** The top-N backtest
  picks "highest score" → "most likely to rise" still holds, so the
  regression and binary tasks share one backtest engine. IC is reported as
  0.0 on the binary task (uninformative on a 0/1 target) — the meaningful
  stats (AUC, top-K precision) are reported separately.
- **`spec_hash` collision caveat:** two specs whose custom factors *share a
  name but differ in behavior* collide in the store (we can't hash a
  callable). Documented on `FeatureSpec.spec_hash()`.
- **NaN feature rows are kept as NULL in the store**, but `run_strategy`
  drops NaN rows before `save_panel`, so the cached matrix is the one the
  model actually trained on.

### Open (unchanged from earlier)
- `[next]` Faith blocklist — `faith_blocklist.yaml` owner-owned.
- **[?] Custom metric namespacing** (`"sharpe"` override vs `custom_` prefix).
- **[?] Walk-forward `valid_ic`** = `train_ic` today; proper holdout or a
  per-fold IC series?
