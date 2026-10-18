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
    features: FeatureSpec | None = None,  # your own factor set
    model: Model | None = None,           # your own learner
    *,
    task: str = "regression",             # "regression" | "binary"
    walk_forward: bool = False,           # anchored walk-forward (off → one-shot)
    retrain_every: int = 1,               # one fold per calendar step
) -> PipelineResult:
    """Fetch, screen, train, backtest, and (optionally) persist a run.

    ``task``:
      * ``"regression"`` (default) — predict the horizon log return.
      * ``"binary"`` — predict "close higher ``horizon`` ahead" (0/1).
        The model outputs P(up); top-N selection still means "most likely
        to rise". This is the coarse, macro-robust signal requested above.
    """
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

    spec = features or FeatureSpec()
    label_kind = "binary" if task == "binary" else "regression"
    X, y = compute_panel(series, spec, horizon=pipeline.horizon,
                         benchmark=cfg.benchmark, label_kind=label_kind)
    feat_cols = spec.column_names()

    # Panel with `y` column visible — walk_forward_predict needs it there.
    X_full = X.copy()
    X_full["y"] = y.to_numpy()

    # Model factory. The task picks the learner; `model=` overrides.
    if model is None:
        def _model_factory() -> Model:
            if task == "binary":
                from pytrader.model import XGBoostBinaryModel
                return XGBoostBinaryModel(random_state=pipeline.random_state)
            return XGBoostModel(random_state=pipeline.random_state)
    else:
        # User-supplied model: same object returned each fold; caller can
        # inspect it after `run_strategy` returns to get the final state.
        _model_factory = lambda: model  # type: ignore[return-value]

    model_obj: Model | None = None   # bound below in whichever path runs
    wf_meta: dict = {}               # populated only when walk_forward is on
    train_ic: float = 0.0            # IC only meaningful for the regression task
    valid_ic: float = 0.0
    binary_stats: dict = {}          # populated for the binary task

    if walk_forward:
        min_train_rows = max(10, int(round(len(X_full) * pipeline.train_fraction)))
        preds, wf_folds, wf_models = walk_forward_predict(
            X_full, feat_cols, _model_factory,
            min_train_rows=min_train_rows,
            retrain_every=retrain_every,
        )
        train_ic = float(np.mean([ic for _dt, ic in wf_folds])) if wf_folds else 0.0
        valid_ic = float(train_ic)  # walk-forward has no out-of-fold split
        preds = preds[(preds.index > wf_folds[0][0])] if wf_folds else preds
        model_obj = wf_models[-1] if wf_models else None
        wf_meta = {"walk_forward": True, "n_folds": len(wf_models),
                   "retrain_every": int(retrain_every),
                   "min_train_rows": int(min_train_rows)}
    else:
        dates = X_full["date"].sort_values().unique()
        cut = dates[max(1, int(round(len(dates) * pipeline.train_fraction)))]
        tr = (X_full["date"] <= cut) & X_full["y"].notna()
        va = (X_full["date"] > cut) & X_full["y"].notna()
        if tr.sum() < 50:
            raise ValueError(f"train set too small: {int(tr.sum())} rows")
        Xtr, ytr = X_full[tr], X_full["y"][tr]
        Xva, yva = X_full[va], X_full["y"][va]

        model = _model_factory()
        if not hasattr(model, "feature_names"):
            raise TypeError("supplied model must implement fit/predict/feature_names")
        model.fit(Xtr[feat_cols], ytr.to_numpy(dtype="float64"))
        model_obj = model  # for later save / feature_importance
        scores_train = model.predict(Xtr[feat_cols])
        scores_valid = model.predict(Xva[feat_cols])

        if task == "binary":
            # Discrimination on the holdout split (IC is uninformative on
            # a 0/1 target). These are the numbers that matter.
            from pytrader.metrics import roc_auc, precision_at_k
            train_ic = 0.0
            valid_ic = 0.0
            binary_stats = {
                "val_auc": roc_auc(yva.to_numpy(dtype="float64"), scores_valid),
                f"val_top{pipeline.top_n}_precision": precision_at_k(
                    yva.to_numpy(dtype="float64"), scores_valid, k=pipeline.top_n),
                "val_base_rate": float(np.mean(yva.to_numpy())),
            }
        else:
            train_ic = _pearson(scores_train, ytr.to_numpy())
            valid_ic = _pearson(scores_valid, yva.to_numpy())
            binary_stats = {}

        Xall = X_full[X_full["y"].notna()]
        scores_all = model.predict(Xall[feat_cols])
        preds = predictions_from_panel(Xall, scores_all)
        preds = preds[(preds.index > pd.Timestamp(cut))]

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
        sim.equity, sim.dates, sim.benchmark_equity, periods=sim.periods
    )
    metrics["metrics"]["avg_turnover"] = round(avg_turn, 6)
    metrics["metrics"]["train_ic"] = round(train_ic, 6)
    metrics["metrics"]["valid_ic"] = round(valid_ic, 6)
    if binary_stats:
        metrics["metrics"].update({k: round(float(v), 6) for k, v in binary_stats.items()})

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
        "task": task,
    })
    if wf_meta:
        out_cfg.update(wf_meta)

    model_dir = model_dir or (Path(".pytrader") / "models" / run_id)
    try:
        if model_obj is not None and hasattr(model_obj, "save"):
            model_obj.save(model_dir, meta={"train_ic": train_ic, "valid_ic": valid_ic,
                                            "run_id": run_id})
    except Exception:  # noqa: BLE001 — persistence is best-effort for POC
        pass

    _fi = (
        model_obj.feature_importance()
        if model_obj is not None and hasattr(model_obj, "feature_importance") else []
    )
    res = PipelineResult(
        run_id=run_id,
        name=name,
        created_at=created_at,
        config=out_cfg,
        metrics=metrics,
        result=sim,
        feature_importance=_fi,
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
        eq = eq.set_index(pd.DatetimeIndex(sim.dates))
        import json as _json
        rec = RunRecord(
            run_id=run_id, name=name, created_at=created_at,
            config=_json.loads(_json.dumps(out_cfg, default=str)),
            metrics=metrics,
        )
        db.save_run(rec, eq)
    return res


def walk_forward_predict(
    X: pd.DataFrame,
    feat_cols: list[str],
    model_factory: "Callable[[], Model]",
    min_train_rows: int,
    retrain_every: int = 1,
) -> tuple[pd.DataFrame, list[tuple[pd.Timestamp, float]], list[Model]]:
    """Anchored walk-forward over a ``(date, ticker, ...features..., y)`` panel.

    Pure — no I/O, no XGBoost import required. Caller supplies the model
    factory (so any ``Model`` learner works) and the feature column list.
    Walks the date axis in ``retrain_every`` steps starting at the first
    date *after* ``min_train_rows`` of training data have closed. Each fold
    fits a fresh model on the expanding window strictly before the fold
    date (no look-ahead) and predicts on that fold's own rows.

    Returns ``(preds, folds, models)``. ``preds`` is one wide row per fold
    date in the same shape ``predictions_from_panel`` produces. ``folds``
    is ``[(fold_date, in_sample_ic), ...]`` in date order. ``models`` is
    the list of trained models (one per fold, in date order) so the
    caller can persist the final one.
    """
    d = pd.Index(pd.to_datetime(X["date"]).unique()).sort_values()
    if d.size == 0:
        raise ValueError("walk-forward: panel has no dates")
    valid = X["y"].notna()
    # Only labelled rows count toward the min-train budget (trailing rows
    # in a panel have y=NaN because there is no forward window).
    per_date = (
        X[valid]
        .groupby(pd.to_datetime(X[valid]["date"])).size()
        .reindex(d).fillna(0).astype("int64")
    )
    running = per_date.to_numpy().cumsum()
    if int(running[-1]) < min_train_rows:
        raise ValueError(
            f"panel has only {int(running[-1])} labelled rows, need >= {min_train_rows}"
        )
    # First fold is one date after the first whose closed window holds
    # at least min_train_rows labelled rows (so that fold's training set
    # closes at that boundary and excludes its own rows).
    start = int(np.searchsorted(running, min_train_rows, side="left")) + 1
    if start >= d.size:
        raise ValueError("not enough labelled dated rows to open a single fold")

    step = max(1, retrain_every)
    wide: list[pd.DataFrame] = []
    fold_stats: list[tuple[pd.Timestamp, float]] = []
    trained: list[Model] = []
    for k in range(start, d.size, step):
        fold_dt = pd.Timestamp(d[k])
        tr = X[(X["date"] < fold_dt) & valid]
        va = X[(X["date"] == fold_dt) & valid]
        if len(tr) < min_train_rows or len(va) == 0:
            continue
        model = model_factory()
        y_tr = tr["y"].to_numpy(dtype="float64")
        model.fit(tr[feat_cols], y_tr)
        ic_train = _pearson(model.predict(tr[feat_cols]), y_tr)
        scores = np.asarray(model.predict(va[feat_cols]), dtype="float64")
        long = pd.DataFrame({"date": va["date"].to_numpy(),
                             "ticker": va["ticker"].to_numpy()})
        wide.append(predictions_from_panel(long, scores))
        fold_stats.append((fold_dt, float(ic_train)))
        trained.append(model)
    if not wide:
        raise ValueError("walk-forward produced no folds; lower min_train_rows")
    out = pd.concat(wide, ignore_index=False)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out, fold_stats, trained


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
