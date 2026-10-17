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
