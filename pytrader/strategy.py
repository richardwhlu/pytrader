"""End-to-end strategy pipeline: data → features → model → backtest → save.

This is *the* function tests and the API call. Every concern stays in its
own module; this file only wires them together.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from pytrader.backtest import (
    BacktestConfig,
    SimulationResult,
    predictions_from_panel,
    simulate,
)
from pytrader.config import Config
from pytrader.data import DailyPrice, PriceStore
from pytrader.features import FeatureSpec, compute_panel
from pytrader.metrics import summarize
from pytrader.model import XGBoostModel
from pytrader.storage import DB, RunRecord
from pytrader.universe import resolve_universe


@dataclass
class PipelineConfig:
    horizon: int = 5
    top_n: int = 5
    commission_bps: float = 1.0
    slippage_bps: float = 5.0
    train_fraction: float = 0.7
    random_state: int = 42

    @classmethod
    def from_config(cls, cfg: Config) -> "PipelineConfig":
        b = cfg.backtest
        return cls(
            horizon=b.horizon,
            top_n=b.top_n,
            commission_bps=b.commission_bps,
            slippage_bps=b.slippage_bps,
            train_fraction=b.train_fraction,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PipelineResult:
    run_id: str
    name: str
    created_at: str
    config: dict
    metrics: dict
    result: SimulationResult
    feature_importance: list[tuple[str, float]] = field(default_factory=list)


def run_strategy(
    store: PriceStore,
    cfg: Config,
    db: DB | None = None,
    name: str = "run",
    pipeline: PipelineConfig | None = None,
    model_dir: Path | None = None,
    universe: dict[str, dict] | None = None,
) -> PipelineResult:
    """Fetch, screen, train, backtest, and (optionally) persist a run."""
    pipeline = pipeline or PipelineConfig.from_config(cfg)

    candidates = list(cfg.include) if cfg.include else (list(universe.keys()) if universe else [])
    if not candidates:
        raise ValueError("config.include is empty and no universe provided")
    if universe:
        # universe: {ticker: {name, sector}} → build TickerInfos + resolve
        from pytrader.universe import TickerInfo
        cand = {t: TickerInfo(t, v.get("name", t), v.get("sector", ""))
                for t, v in universe.items()}
    else:
        from pytrader.universe import TickerInfo
        meta = store.metadata(candidates)
        cand = {t: TickerInfo(t, meta.get(t, {}).get("name", t),
                              meta.get(t, {}).get("sector", "")) for t in candidates}

    selected, rejected = resolve_universe(cand, cfg)
    if len(selected) < pipeline.top_n + 1:
        raise ValueError(
            f"only {len(selected)} tickers selected, need >= {pipeline.top_n + 1}"
        )
    tickers = [d.info.ticker for d in selected]

    start = date.today() - timedelta(days=int(cfg.backtest.lookback_years * 365.25))
    series = store.fetch([cfg.benchmark] + tickers, start)
    if len(series) < len(tickers) + 1:
        raise ValueError(
            f"expected {len(tickers)+1} series, got {len(series)} from store"
        )

    spec = FeatureSpec()
    X, y = compute_panel(series, spec, horizon=pipeline.horizon,
                         benchmark=cfg.benchmark)
    feat_cols = spec.column_names()

    dates = X["date"].sort_values().unique()
    cut = dates[max(1, int(round(len(dates) * pipeline.train_fraction)))]
    tr = (X["date"] <= cut) & y.notna()
    va = (X["date"] > cut) & y.notna()
    if tr.sum() < 50:
        raise ValueError(f"train set too small: {int(tr.sum())} rows")
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]

    model = XGBoostModel(random_state=pipeline.random_state)
    model.fit(Xtr[feat_cols], ytr.to_numpy(dtype="float64"))
    scores_train = model.predict(Xtr[feat_cols])
    scores_valid = model.predict(Xva[feat_cols])
    train_ic = _pearson(scores_train, ytr.to_numpy())
    valid_ic = _pearson(scores_valid, yva.to_numpy())
    allmask = y.notna()
    Xall = X[allmask]
    yall = y[allmask]
    scores_all = model.predict(Xall[feat_cols])
    preds = predictions_from_panel(Xall, scores_all)
    preds = preds[(preds.index > pd.Timestamp(cut))]  # decision dates post-split

    bt = BacktestConfig(
        horizon=pipeline.horizon,
        top_n=pipeline.top_n,
        commission_bps=pipeline.commission_bps,
        slippage_bps=pipeline.slippage_bps,
    )
    sim = simulate(preds, series, bt, benchmark=cfg.benchmark)

    # average turnover from periods
    avg_turn = float(np.mean([p.turnover for p in sim.periods])) if sim.periods else 0.0
    metrics = summarize(
        sim.equity, sim.dates, sim.benchmark_equity
    )
    metrics["metrics"]["avg_turnover"] = round(avg_turn, 6)
    metrics["metrics"]["train_ic"] = round(train_ic, 6)
    metrics["metrics"]["valid_ic"] = round(valid_ic, 6)

    if sim.benchmark_equity is not None:
        metrics["metrics"].setdefault("benchmark_ir", metrics["benchmark"].get(
            "information_ratio", 0.0))

    run_id = _make_run_id()
    created_at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    out_cfg = dict(pipeline.to_dict())
    out_cfg.update({
        "benchmark": cfg.benchmark,
        "n_tickers": len(tickers),
        "tickers": tickers,
        "universe_selected": len(selected),
        "universe_rejected": [
            {"ticker": d.info.ticker, "reasons": d.reasons} for d in rejected
        ][:20],
        "feature_set": feat_cols + ["horizon", "top_n"],
    })

    model_dir = model_dir or (Path(".pytrader") / "models" / run_id)
    try:
        model.save(model_dir, meta={"train_ic": train_ic, "valid_ic": valid_ic,
                                   "run_id": run_id})
    except Exception:  # noqa: BLE001 — persistence is best-effort for POC
        pass

    res = PipelineResult(
        run_id=run_id,
        name=name,
        created_at=created_at,
        config=out_cfg,
        metrics=metrics,
        result=sim,
        feature_importance=model.feature_importance(),
    )

    if db is not None:
        eq = pd.DataFrame({
            "equity": sim.equity.values.astype("float64"),
            "benchmark_equity": (
                sim.benchmark_equity.reindex(sim.dates).to_numpy(dtype="float64")
                if sim.benchmark_equity is not None
                else np.full(len(sim.equity), np.nan)
            ),
        })
        eq = eq.set_index(pd.DatetimeIndex(sim.dates).asi8 // 10**9)
        import json as _json
        rec = RunRecord(
            run_id=run_id, name=name, created_at=created_at,
            config=_json.loads(_json.dumps(out_cfg, default=str)),
            metrics=metrics,
        )
        db.save_run(rec, eq)
    return res


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _make_run_id() -> str:
    ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"
