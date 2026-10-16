"""Flask API + web dashboard.

Routes:
  GET  /                        → serves the static web dashboard
  GET  /api/health              → basic liveness
  GET  /api/universe            → selected/rejected tickers + reasons
  GET  /api/runs                → list of saved runs (most recent first)
  GET  /api/runs/<run_id>       → run detail + config + metrics
  GET  /api/compare?run_ids=a,b → equity + metrics side-by-side
  POST /api/backtest            → run a new backtest with the given config
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from pytrader.config import Config
from pytrader.data import make_store
from pytrader.storage import DuckDBStore
from pytrader.strategy import PipelineConfig, run_strategy


def create_app(cfg: Config | None = None) -> Flask:
    static = Path(__file__).parent / "static"
    app = Flask(
        __name__,
        static_folder=str(static),
        static_url_path="/static",
    )
    cfg = cfg or Config.load()
    store = make_store("yfinance")
    db = DuckDBStore(Path(".pytrader") / "pytrader.duckdb")

    @app.get("/")
    def index():  # type: ignore[return]
        return app.send_static_file("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "benchmark": cfg.benchmark})

    @app.get("/api/universe")
    def universe():
        from pytrader.universe import TickerInfo
        meta = store.metadata(cfg.include)
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

    @app.get("/api/runs")
    def runs():
        recs = db.list_runs()
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
        r = db.get_run(run_id)
        if r is None:
            return jsonify({"error": "run not found"}), 404
        eq = db.get_equity(run_id)
        if eq is None:
            return jsonify({"error": "run found but no equity data"}), 500
        eq_list = [
            [int(ts), float(eq.loc[(ts,), "equity"][0]),
             None if pd.isna(eq.loc[(ts,), "benchmark_equity"][0])
             else float(eq.loc[(ts,), "benchmark_equity"][0])]
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
        for rid in ids[:10]:
            r = db.get_run(rid)
            if r is None:
                continue
            eq = db.get_equity(rid)
            if eq is None:
                continue
            equity = [[int(ts), _f("equity", eq, ts)] for ts in eq.index]
            bench = [[int(ts), _f("benchmark_equity", eq, ts)]
                     for ts in eq.index if _f("benchmark_equity", eq, ts) is not None]
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
        p = PipelineConfig(
            horizon=int(body.get("horizon", base.horizon)),
            top_n=int(body.get("top_n", base.top_n)),
            commission_bps=float(body.get("commission_bps", base.commission_bps)),
            slippage_bps=float(body.get("slippage_bps", base.slippage_bps)),
            train_fraction=base.train_fraction,
            random_state=base.random_state,
        )
        name = str(body.get("name", "")).strip() or f"run-{p.run_id if hasattr(p, 'run_id') else 'inline'}"
        try:
            res = run_strategy(store, cfg, db=db, name=name, pipeline=p)
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": str(e)}), 400
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
    v = eq.loc[(ts,), col].iloc[0]
    if pd.isna(v):
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
    store = make_store("yfinance")
    db = DuckDBStore(Path(".pytrader") / "pytrader.duckdb")
    p = PipelineConfig.from_config(cfg)
    res = run_strategy(store, cfg, db=db, name=name, pipeline=p)
    print(f"run_id     : {res.run_id}")
    print(f"name       : {res.name}")
    print(f"created_at : {res.created_at}")
    print(json.dumps(res.metrics, indent=2))
    print(f"top features: {res.feature_importance[:5]}")
    print(f"data       : {len(res.result.dates)} days, "
          f"{len(res.result.periods)} rebalances, "
          f"start={res.result.first_date:%Y-%m-%d}  end={res.result.last_date:%Y-%m-%d}")
