"""Flask API + web dashboard.

Routes:
  GET  /                        → serves the static web dashboard
  GET  /api/health              → basic liveness
  GET  /api/universe            → selected/rejected tickers + reasons
  GET  /api/runs                → list of saved runs (most recent first)
  GET  /api/runs/<run_id>       → run detail + config + metrics
  GET  /api/compare?run_ids=a,b → equity + metrics side-by-side
  GET  /api/tickers             → cached tickers (name, sector, end_date)
  GET  /api/series              → re-indexed pct-gain series + composite
  GET  /api/optimize            → mean-variance basket weights for a window
  POST /api/backtest            → run a new backtest with the given config
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from pytrader import paths
from pytrader.config import Config
from pytrader import optimize as opt
from pytrader.data import make_store, PriceStore
from pytrader.feature_store import FeatureStore
from pytrader.storage import DB, DuckDBStore
from pytrader.strategy import PipelineConfig, run_strategy


def create_app(
    cfg: Config | None = None,
    store: PriceStore | None = None,
    db: DB | None = None,
    db_path: str | Path | None = None,
) -> Flask:
    """Create the Flask app.

    **Concurrency model**
    DuckDB is single-process: any connection (read-only or read-write)
    held for the life of the process blocks every other process that
    tries to open the same DB — including the ``pytrader-download`` and
    ``pytrader-backtest`` CLIs. To let the dashboard and the CLIs coexist,
    this app opens a *fresh* ``DuckDBStore`` per request and closes it
    before returning. Between requests no lock is held on the DB, so a
    CLI can always run.

    Injection points for tests:
      * ``cfg``     — user config (defaults to ``Config.load()``)
      * ``store``   — price data source (defaults to yfinance)
      * ``db``      — read-only price/run repository. Inject a live
                      ``DuckDBStore`` to bypass the per-request opening
                      (hermetic tests do this).
      * ``db_path`` — alternate DB location
    """
    static = Path(__file__).parent / "static"
    app = Flask(
        __name__,
        static_folder=str(static),
        static_url_path="/static",
    )
    cfg = cfg or Config.load()
    from contextlib import contextmanager

    @contextmanager
    def _store_ctx(read_only: bool = True):
        """Yield ``(store, db)`` for one request.

        DuckDB is single-process: an **open** connection (read-only or
        read-write) blocks any other process from opening the same
        database — including our own ``pytrader-download`` and
        ``pytrader-backtest`` CLIs. So the app opens a **fresh**
        ``DuckDBStore`` per request and closes it before the response
        is sent. Between requests no lock is held, so the CLI can
        always proceed.

        If the caller injected ``db`` (hermetic tests), we honour it
        and don't open a separate handle.
        """
        if db is not None:
            _s = store if store is not None else make_store("yfinance")
            yield (_s, db)
            return
        p = Path(db_path) if db_path else None
        d = DuckDBStore(p, read_only=read_only)
        s = make_store("yfinance", cache=d) if store is None else store
        try:
            yield (s, d)
        finally:
            d.close()

    @app.get("/")
    def index():  # type: ignore[return]
        return app.send_static_file("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "benchmark": cfg.benchmark})

    @app.get("/api/universe")
    def universe():
        from pytrader.universe import TickerInfo
        with _store_ctx(read_only=True) as (st, _):
            meta = st.metadata(cfg.include)
            cand = {
                t: TickerInfo(t, meta.get(t, {}).get("name", t),
                              meta.get(t, {}).get("sector", ""))
                for t in cfg.include
            }
            sel, rej = _resolve(cand, cfg)
        return jsonify({
            "benchmark": cfg.benchmark,
            "selected": [
                {"ticker": d.info.ticker, "name": d.info.name,
                 "sector": d.info.sector, "reasons": d.reasons}
                for d in sel
            ],
            "rejected": [
                {"ticker": d.info.ticker, "name": d.info.name,
                 "sector": d.info.sector, "reasons": d.reasons}
                for d in rej
            ],
        })

    @app.get("/api/tickers")
    def tickers():
        """List every ticker we have cached prices for.

        Powers the UI's ticker pickers (historical visualizer, universe).
        Returns sorted tickers + their cached name/sector/end_date so the
        picker can render readable rows without an extra round-trip per row.
        """
        with _store_ctx(read_only=True) as (_, db_):
            out = []
            for t in db_.tickers():
                meta = db_.load_meta(t) or {}
                end = db_.end_date(t)
                out.append({
                    "ticker": t,
                    "name": meta.get("name", t),
                    "sector": meta.get("sector", ""),
                    "end_date": end.date().isoformat() if end is not None else None,
                })
        return jsonify({"tickers": out})

    @app.get("/api/series")
    def series():
        """Percent-gain series from `start_date` (0% on the base day).

        Query params:
          tickers    (required, comma-separated) e.g. "NVDA,MSFT,SPY"
          start_date (optional, YYYY-MM-DD; defaults to earliest cached
                      close among the selected tickers)
          horizon    (optional, days) caps the chart to start + horizon.
                     Use 0 / omit to go to the latest cached date.

        Response shape:
          {
            "start": "YYYY-MM-DD",           # earliest base date across selected
            "end":   "YYYY-MM-DD",           # end of the returned window
            "series": {T: {"name","sector","base_date","base_close","end_close",
                           "end_pct", "points": [["YYYY-MM-DD", pct_gain], ...]}},
            "missing": [T, ...]              # requested but not cached
          }

        Values are **percent gain from the base day**: the first trading day
        at/after `start_date` for each ticker → 0%, then whatever % it's
        up/down by the end. So "NVDA at 0 on 2020-01-01, +200% two years
        later" reads straight off the y-axis (base=0, end≈+200).
        """
        raw_tickers = (request.args.get("tickers") or "").split(",")
        tickers = [t.strip().upper() for t in raw_tickers if t.strip()]
        if not tickers:
            return jsonify({"error": "tickers is required"}), 400

        start_raw = request.args.get("start_date")
        try:
            horizon = int(request.args.get("horizon") or 0)
        except ValueError:
            return jsonify({"error": "horizon must be an integer (trading days)"}), 400
        if horizon < 0:
            return jsonify({"error": "horizon must be >= 0"}), 400

        start_ts: pd.Timestamp | None = None
        if start_raw:
            try:
                start_ts = pd.Timestamp(start_raw).normalize()
            except (ValueError, TypeError):
                return jsonify({"error": "start_date must be YYYY-MM-DD"}), 400

        out_series: dict[str, dict] = {}
        missing: list[str] = []
        # Track the earliest start-date we can actually offer per ticker, and
        # the global earliest/latest close so the UI has the bounds.
        global_earliest: pd.Timestamp | None = None
        global_latest: pd.Timestamp | None = None
        with _store_ctx(read_only=True) as (_, db):
            for t in tickers:
                s: pd.Series | None = db.load(t)
                if s is None or len(s) == 0:
                    missing.append(t)
                    continue
                if start_ts is not None:
                    s = s.loc[s.index >= start_ts]
                    if len(s) == 0:
                        # Nothing after start_date — treat as missing.
                        missing.append(t)
                        continue
                    base = s.iloc[0]
                else:
                    base = s.iloc[0]
                if base <= 0:
                    # Can't reindex against a non-positive price.
                    missing.append(t)
                    continue
                # Percent gain from the base day: 0% on the base day itself.
                gain = (s / base - 1.0) * 100.0
                # Cap the window at `horizon` trading days from the base.
                if horizon > 0:
                    gain = gain.iloc[:horizon]
                end_close = float(s.loc[gain.index[-1]])
                # Update global bounds.
                e, l = gain.index[0], gain.index[-1]
                global_earliest = e if global_earliest is None else min(global_earliest, e)
                global_latest = l if global_latest is None else max(global_latest, l)
                meta = db.load_meta(t) or {}
                out_series[t] = {
                    "name": meta.get("name", t),
                    "sector": meta.get("sector", ""),
                    "base_date": gain.index[0].date().isoformat(),
                    "base_close": float(base),
                    "end_close": end_close,
                    "end_pct": float(gain.iloc[-1]),   # % change base→end
                    "points": [[d.date().isoformat(), float(v)]
                               for d, v in list(gain.items())],
                }
        return jsonify({
            "start": global_earliest.date().isoformat() if global_earliest else None,
            "end": global_latest.date().isoformat() if global_latest else None,
            "series": out_series,
            "missing": missing,
        })

    @app.get("/api/optimize")
    def optimize():
        """Mean-variance basket optimizer for a historical window.

        Given tickers + an optional date range, returns a single optimized
        portfolio or the full efficient-frontier "distribution" across
        risk-aversion values. The underlying solver is the classical
        Markowitz reward-minus-risk problem:

            maximize  w·μ  −  λ·wᵀΣw    s.t.  0 ≤ wᵢ ≤ 1,  Σwᵢ = 1

        Query params:
          tickers        (required, comma-separated)  e.g. "A,B,C"
          start_date     (optional YYYY-MM-DD, default earliest cached)
          end_date       (optional YYYY-MM-DD, default latest cached)
          risk_aversion  (optional float ≥ 0; 0 = pure gain / 100 % top mover)
          frontier       (optional "true" → return the full trade-off curve)
          steps          (optional int ≥ 2, number of frontier points)

        Response (always the same shape):
          {
            "objective": "return − λ·variance (daily rebalanced)",
            "tickers": ["A", "B"],
            "start":  "2019-11-15",
            "end":    "2020-03-15",
            "n":        1,                  # = len(points)
            "points": [ {
              "risk_aversion":  0.0,
              "weights":        {"A": 1.0, "B": 0.0},
              "stats": { "total_return", "ann_vol", "sharpe",
                         "max_drawdown", "n_days", "n_positions" }
            }, … ]
          }
        """
        raw = (request.args.get("tickers") or "").split(",")
        tickers = [t.strip().upper() for t in raw if t.strip()]
        if not tickers:
            return jsonify({"error": "tickers is required (comma-separated)"}), 400

        start_raw = request.args.get("start_date")
        end_raw = request.args.get("end_date")
        try:
            start_ts = pd.Timestamp(start_raw).normalize() if start_raw else None
        except (ValueError, TypeError):
            return jsonify({"error": "start_date must be YYYY-MM-DD"}), 400
        try:
            end_ts = pd.Timestamp(end_raw).normalize() if end_raw else None
        except (ValueError, TypeError):
            return jsonify({"error": "end_date must be YYYY-MM-DD"}), 400

        # Parse optional risk aversion / frontier params.
        lam_raw = request.args.get("risk_aversion")
        lam: float | None = None
        if lam_raw is not None:
            try:
                lam = float(lam_raw)
            except ValueError:
                return jsonify({"error": "risk_aversion must be a number"}), 400
            if lam < 0:
                return jsonify({"error": "risk_aversion must be ≥ 0"}), 400

        want_frontier = (request.args.get("frontier") or "").lower() in ("1", "true", "yes")
        # steps only applies to frontier mode; single-point mode ignores it,
        # so it's only validated when we actually need it.
        steps: int | None = None
        if want_frontier:
            steps_raw = request.args.get("steps")
            if steps_raw is not None:
                try:
                    steps = int(steps_raw)
                except ValueError:
                    return jsonify({"error": "steps must be an integer ≥ 2"}), 400
                if steps < 2:
                    return jsonify({"error": "steps must be ≥ 2"}), 400

        # Build the closes DataFrame from the cache.
        series_map: dict[str, pd.Series] = {}
        missing: list[str] = []
        with _store_ctx(read_only=True) as (_, d):
            for t in tickers:
                s = d.load(t)
                if s is None or len(s) == 0:
                    missing.append(t)
                    continue
                if start_ts is not None:
                    s = s.loc[s.index >= start_ts]
                if end_ts is not None:
                    s = s.loc[s.index <= end_ts]
                if len(s) < 2:
                    missing.append(t)
                    continue
                series_map[t] = s

        kept = list(series_map.keys())
        if len(kept) < 1:
            return jsonify({"error": "no tickers have cached data in this window",
                           "missing": missing}), 400

        closes = pd.DataFrame(series_map)
        all_dates = closes.dropna().index
        if len(all_dates) < 2:
            return jsonify({
                "error": "the selected tickers share fewer than 2 common trading "
                         "days in this window — widen the range",
                "missing": missing,
                "kept": kept,
            }), 400

        try:
            if want_frontier:
                result = opt.efficient_frontier(closes, steps=steps or 12)
                if lam is not None:
                    # Also compute the exact point at the requested λ so the
                    # UI can show a "recommended basket" even when the slider
                    # lands between two samples of the frontier.
                    single = opt.optimize_window(closes, risk_aversion=lam)
                    result["recommended"] = single.to_dict()
            else:
                single = opt.optimize_window(closes, risk_aversion=lam or 0.0)
                result = {
                    "objective": "return − λ·variance (daily rebalanced)",
                    "n": 1,
                    "points": [single.to_dict()],
                    "recommended": single.to_dict(),
                }
        except ValueError as ve:
            return jsonify({"error": str(ve), "missing": missing, "kept": kept}), 400
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"optimizer failed: {exc}",
                           "missing": missing, "kept": kept}), 500

        return jsonify({
            **result,
            "tickers": kept,
            "missing": missing,
            "start": all_dates[0].date().isoformat(),
            "end":   all_dates[-1].date().isoformat(),
        })

    @app.get("/api/runs")
    def runs():
        with _store_ctx(read_only=True) as (_, db_):
            recs = db_.list_runs()
            out = []
            for r in recs:
                m = r.metrics.get("metrics", r.metrics)
                out.append({
                    "run_id": r.run_id,
                    "name": r.name,
                    "created_at": r.created_at,
                    "metrics": {
                        "sharpe": m.get("sharpe", 0),
                        "cagr": m.get("cagr", 0),
                        "max_drawdown": m.get("max_drawdown", 0),
                        "total_return": m.get("total_return", 0),
                        "ann_vol": m.get("ann_vol", 0),
                        "hit_rate": m.get("hit_rate", 0),
                        "sortino": m.get("sortino", 0),
                        "avg_turnover": m.get("avg_turnover", 0),
                        "train_ic": m.get("train_ic", 0),
                        "valid_ic": m.get("valid_ic", 0),
                        "benchmark_ir": m.get("benchmark_ir", 0),
                        "n_days": m.get("n_days", 0),
                        "n_tickers": r.config.get("n_tickers", 0),
                    },
                })
        return jsonify({"runs": out})

    @app.get("/api/runs/<run_id>")
    def run_detail(run_id: str):
        with _store_ctx(read_only=True) as (_, db_):
            r = db_.get_run(run_id)
            if r is None:
                return jsonify({"error": "run not found"}), 404
            eq = db_.get_equity(run_id)
            if eq is None:
                return jsonify({"error": "run found but no equity data"}), 500
            eq_list = [
                [int(ts.timestamp() * 1000),
                 _f("equity", eq, ts),
                 _f("benchmark_equity", eq, ts)]
                for ts in eq.index
            ]
            return jsonify({
                "run_id": r.run_id,
                "name": r.name,
                "created_at": r.created_at,
                "metrics": r.metrics,
                "config": r.config,
                "equity": eq_list,
            })

    @app.get("/api/compare")
    def compare():
        ids = (request.args.get("run_ids") or "").split(",")
        ids = [s.strip() for s in ids if s.strip()]
        if not ids:
            return jsonify({"error": "run_ids required"}), 400
        out = []
        with _store_ctx(read_only=True) as (_, db_):
            for rid in ids[:10]:
                r = db_.get_run(rid)
                if r is None:
                    continue
                eq = db_.get_equity(rid)
                if eq is None:
                    continue
                equity = [
                    [int(ts.timestamp() * 1000), _f("equity", eq, ts)]
                    for ts in eq.index
                ]
                bench = [
                    [int(ts.timestamp() * 1000), _f("benchmark_equity", eq, ts)]
                    for ts in eq.index if _f("benchmark_equity", eq, ts) is not None
                ]
                m = r.metrics.get("metrics", r.metrics)
                out.append({
                    "run_id": rid, "name": r.name, "created_at": r.created_at,
                    "metrics": m,
                    "equity": equity,
                    "benchmark_equity": bench,
                    "n_days": m.get("n_days", 0),
                    "n_tickers": r.config.get("n_tickers", 0),
                })
        return jsonify({"runs": out})

    @app.post("/api/backtest")
    def backtest():
        body = request.get_json(silent=True) or {}
        base = PipelineConfig.from_config(cfg)
        # explicit validation with a helpful error message
        try:
            horizon = int(body.get("horizon", base.horizon))
            top_n = int(body.get("top_n", base.top_n))
            commission_bps = float(body.get("commission_bps", base.commission_bps))
            slippage_bps = float(body.get("slippage_bps", base.slippage_bps))
        except (TypeError, ValueError) as e:
            return jsonify({"error": f"invalid numeric parameter: {e}"}), 400
        if horizon < 1:
            return jsonify({"error": "horizon must be >= 1"}), 400
        if top_n < 1:
            return jsonify({"error": "top_n must be >= 1"}), 400
        if commission_bps < 0 or slippage_bps < 0:
            return jsonify({"error": "costs must be non-negative"}), 400
        p = PipelineConfig(
            horizon=horizon,
            top_n=top_n,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            train_fraction=base.train_fraction,
            random_state=base.random_state,
        )
        name = str(body.get("name", "")).strip() or f"run-{p.run_id if hasattr(p, 'run_id') else 'inline'}"
        try:
            from pytrader.storage import FeatureStorage as _FS
            # Backtests must **write** (run + equity + feature cache), so
            # use a writable connection — even though the app's default is
            # read-only, since a read-only connection would break the
            # write path and leave no lock held on any subsequent request.
            with _store_ctx(read_only=False) as (st, db_):
                fs = FeatureStore(db_) if isinstance(db_, _FS) else None
                res = run_strategy(st, cfg, db=db_, name=name, pipeline=p,
                                   feature_store=fs)
        except Exception as e:  # noqa: BLE001
            import traceback as _tb
            # Surface the full stack in the JSON so a browser-side error has a
            # debuggable payload (mirrors the CLI behavior). 500 for genuine
            # failures — the caller was not at fault.
            logging.getLogger("pytrader.server").error("backtest failed", exc_info=e)
            return jsonify({
                "error": str(e),
                "type": type(e).__name__,
                "traceback": _tb.format_exc(),
            }), 500
        return jsonify({
            "run_id": res.run_id,
            "name": res.name,
            "created_at": res.created_at,
            "metrics": res.metrics,
            "config": res.config,
        }), 201

    return app


def _resolve(cand: dict, cfg: Config):
    from pytrader.universe import resolve_universe
    return resolve_universe(cand, cfg)


def _f(col: str, eq: pd.DataFrame, ts):
    v = eq.at[ts, col]
    if v is None or pd.isna(v):
        return None
    return float(v)


def run_cli(host: str = "127.0.0.1", port: int = 8000, cfg: Config | None = None) -> None:
    app = create_app(cfg)
    app.run(host=host, port=port, debug=False, use_reloader=False)


# Keep the name the pyproject console-script expects.
main = run_cli


def run_backtest_cli(name: str = "cli-run") -> None:
    """One-shot CLI backtest using default config + data (yfinance)."""
    cfg = Config.load()
    db = DuckDBStore(paths.db_path())
    store = make_store("yfinance", cache=db)
    p = PipelineConfig.from_config(cfg)
    res = run_strategy(store, cfg, db=db, name=name, pipeline=p,
                       feature_store=FeatureStore(db))
    print(f"run_id     : {res.run_id}")
    print(f"name       : {res.name}")
    print(f"created_at : {res.created_at}")
    print(json.dumps(res.metrics, indent=2))
    print(f"top features: {res.feature_importance[:5]}")
    print(f"data       : {len(res.result.dates)} days, "
          f"{len(res.result.periods)} rebalances, "
          f"start={res.result.first_date:%Y-%m-%d}  end={res.result.last_date:%Y-%m-%d}")
