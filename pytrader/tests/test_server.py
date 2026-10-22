"""Flask API smoke tests — hermetic: InMemoryStore + temp DuckDB, no network."""
from __future__ import annotations

import pytest
from pytrader.config import Config
from pytrader.server import create_app
from pytrader.storage import DuckDBStore
from pytrader.testing import InMemoryStore, synthetic_series


@pytest.fixture
def store():
    return InMemoryStore(synthetic_series(n=700, n_tickers=8, horizon=5, seed=7))


@pytest.fixture
def cfg(store):
    cfg = Config.default()
    cfg.include = [t for t in store.series if t != "SPY"]
    cfg.exclude = []
    cfg.faith.enabled = False
    cfg.backtest.lookback_years = 5
    return cfg


@pytest.fixture
def db(tmp_path):
    return DuckDBStore(tmp_path / "api.duckdb")


@pytest.fixture
def client(store, cfg, db):
    app = create_app(cfg=cfg, store=store, db=db)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["benchmark"] == "SPY"


def test_runs_empty(client):
    r = client.get("/api/runs")
    assert r.status_code == 200
    assert r.get_json() == {"runs": []}


def test_universe_returns_selections(client):
    r = client.get("/api/universe")
    assert r.status_code == 200
    body = r.get_json()
    assert body["benchmark"] == "SPY"
    assert isinstance(body["selected"], list)
    assert isinstance(body["rejected"], list)
    # all 8 tickers selected, zero rejected (faith off, no excludes)
    assert len(body["selected"]) == 8
    assert body["rejected"] == []


def test_backtest_creates_run(client, db):
    r = client.post(
        "/api/backtest",
        json={"name": "api-run", "horizon": 5, "top_n": 3,
              "commission_bps": 1, "slippage_bps": 5},
    )
    assert r.status_code == 201, r.get_json()
    body = r.get_json()
    assert body["name"] == "api-run"
    assert "run_id" in body
    assert body["metrics"]["metrics"]["sharpe"]
    # run is persisted
    runs = db.list_runs()
    assert any(ru.run_id == body["run_id"] for ru in runs)


def test_run_detail(client, db):
    client.post(
        "/api/backtest",
        json={"name": "d", "horizon": 5, "top_n": 3},
    )
    rid = db.list_runs()[-1].run_id
    r = client.get(f"/api/runs/{rid}")
    assert r.status_code == 200
    body = r.get_json()
    assert body["run_id"] == rid
    assert len(body["equity"]) > 5
    # [ts, equity, bench_or_none]
    assert len(body["equity"][0]) == 3


def test_run_detail_404(client):
    r = client.get("/api/runs/nope")
    assert r.status_code == 404


def test_compare_missing_ids(client):
    r = client.get("/api/compare")
    assert r.status_code == 400


def test_compare_multiple_runs(client, db):
    client.post("/api/backtest", json={"name": "a", "horizon": 5, "top_n": 3})
    client.post("/api/backtest", json={"name": "b", "horizon": 5, "top_n": 4})
    ids = ",".join(ru.run_id for ru in db.list_runs())
    r = client.get(f"/api/compare?run_ids={ids}")
    assert r.status_code == 200
    runs = r.get_json()["runs"]
    assert len(runs) == 2
    for run in runs:
        assert "equity" in run and len(run["equity"]) > 5


def test_index_serves_dashboard(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "app.js" in html
    assert "PyTrader" in html


def test_static_served(client):
    r = client.get("/static/app.js")
    assert r.status_code == 200
    assert b"create_app" not in r.get_data()  # not python
    js = r.get_data(as_text=True)
    assert "api/universe" in js


def test_backtest_validation(client):
    # invalid parameter values are rejected before hitting the pipeline
    assert client.post("/api/backtest", json={"horizon": 0}).status_code == 400
    assert client.post("/api/backtest", json={"top_n": 0}).status_code == 400
    assert client.post(
        "/api/backtest", json={"commission_bps": -1}
    ).status_code == 400
    # non-numeric types are rejected too
    assert client.post("/api/backtest", json={"horizon": "lots"}).status_code == 400


# ---------------------------------------------------------------------
# /api/tickers + /api/series  (Historical re-indexed visualizer)
# ---------------------------------------------------------------------
import pandas as pd
import numpy as np
from datetime import date

@pytest.fixture
def price_db(tmp_path):
    """Two tickers with known closes, saved into a temp DuckDBStore."""
    d = DuckDBStore(tmp_path / "hist.duckdb")
    idx = pd.bdate_range("2019-11-15", "2020-03-15")  # ~90 business days
    # A: flat 100, then +30%.
    closes_a = list(np.repeat(100.0, 30)) + list(np.linspace(100, 130, len(idx) - 30))
    # B: flat 100, then -15%.
    closes_b = list(np.repeat(100.0, 30)) + list(np.linspace(100, 85, len(idx) - 30))
    d.save("A", pd.Series(closes_a, index=idx, name="A"))
    d.save("B", pd.Series(closes_b, index=idx, name="B"))
    return d


def test_api_tickers_lists_cached(price_db, cfg):
    app = create_app(cfg=cfg, db=price_db)
    app.config["TESTING"] = True
    with app.test_client() as c:
        r = c.get("/api/tickers")
        assert r.status_code == 200
        body = r.get_json()
        assert {t["ticker"] for t in body["tickers"]} == {"A", "B"}
        # Each row carries name/sector/end_date for the UI to render.
        for row in body["tickers"]:
            assert row["ticker"] in {"A", "B"}
            assert "name" in row and "sector" in row and "end_date" in row


def test_api_series_base_is_zero_and_end_matches(price_db, cfg):
    d = price_db
    first_a = d.load("A").iloc[0]
    last_a = d.load("A").iloc[-1]
    expected_end_a = (last_a / first_a - 1) * 100.0  # ~ +30%

    last_b = d.load("B").iloc[-1]
    expected_end_b = (last_b / d.load("B").iloc[0] - 1) * 100.0  # ~ -15%

    app = create_app(cfg=cfg, db=d)
    app.config["TESTING"] = True
    with app.test_client() as c:
        r = c.get("/api/series?tickers=A,B")
        assert r.status_code == 200
        body = r.get_json()
        assert set(body["series"].keys()) == {"A", "B"}
        a = body["series"]["A"]
        b = body["series"]["B"]
        # First point of each series must be 0 (the base day = 0).
        assert a["points"][0][1] == 0.0, f"A should start at 0, got {a['points'][0]}"
        assert b["points"][0][1] == 0.0
        # End-of-window % gain matches the computed value.
        assert abs(a["end_pct"] - expected_end_a) < 1e-9
        assert abs(b["end_pct"] - expected_end_b) < 1e-9
        # Per-ticker P&L metadata is present.
        assert a["base_close"] > 0


def test_api_series_slices_to_start_date(price_db, cfg):
    d = price_db
    # Start on a day ~15 into the series so the base is the 15th close, not
    # the first close. The base day must still be 0.
    start = d.load("A").index[15].date().isoformat()
    base = d.load("A").iloc[15]  # this becomes the new "100 / 0%"
    app = create_app(cfg=cfg, db=d)
    app.config["TESTING"] = True
    with app.test_client() as c:
        r = c.get(f"/api/series?tickers=A&start_date={start}")
        body = r.get_json()
        a = body["series"]["A"]
        assert a["base_date"] == start
        assert a["points"][0][1] == 0.0
        assert a["base_close"] == base
        # No points before start_date.
        for dstr, _ in a["points"]:
            assert dstr >= start


def test_api_series_horizon_caps_the_window(price_db, cfg):
    d = price_db
    n_days = len(d.load("A"))
    app = create_app(cfg=cfg, db=d)
    app.config["TESTING"] = True
    with app.test_client() as c:
        r = c.get(f"/api/series?tickers=A&horizon=10")
        body = r.get_json()
        assert len(body["series"]["A"]["points"]) == 10


def test_api_series_missing_ticker_reported(price_db, cfg):
    app = create_app(cfg=cfg, db=price_db)
    app.config["TESTING"] = True
    with app.test_client() as c:
        r = c.get("/api/series?tickers=A,ZZZ")
        assert r.status_code == 200
        body = r.get_json()
        assert "A" in body["series"]
        assert "ZZZ" in body["missing"]


# ---------------------------------------------------------------------
# /api/optimize  (mean-variance basket optimizer over a date window)
# ---------------------------------------------------------------------

def test_optimize_requires_tickers(price_db, cfg):
    app = create_app(cfg=cfg, db=price_db)
    app.config["TESTING"] = True
    with app.test_client() as c:
        assert c.get("/api/optimize").status_code == 400
        body = c.get("/api/optimize").get_json()
        assert "error" in body


def test_optimize_pure_gain_puts_on_top_mover(price_db, cfg):
    """λ=0 → 100% in A (up 30%), 0% in B (down 15%)."""
    app = create_app(cfg=cfg, db=price_db)
    app.config["TESTING"] = True
    with app.test_client() as c:
        r = c.get("/api/optimize?tickers=A,B&risk_aversion=0")
        assert r.status_code == 200
        body = r.get_json()
        pts = body["points"]
        assert len(pts) == 1
        w = pts[0]["weights"]
        assert w["A"] == pytest.approx(1.0, abs=1e-4)
        assert w["B"] == pytest.approx(0.0, abs=1e-4)
        # The basket return then equals A's window return.
        a = price_db.load("A")
        expected_a_ret = float(a.iloc[-1]) / float(a.iloc[0]) - 1.0
        assert body["points"][0]["stats"]["total_return"] == pytest.approx(
            expected_a_ret, abs=1e-6,
        )


def test_optimize_feasible_weights(price_db, cfg):
    app = create_app(cfg=cfg, db=price_db)
    app.config["TESTING"] = True
    with app.test_client() as c:
        for lam in ("0", "0.5", "10", "1000"):
            body = c.get(f"/api/optimize?tickers=A,B&risk_aversion={lam}").get_json()
            w = body["points"][0]["weights"]
            assert sum(w.values()) == pytest.approx(1.0, abs=1e-5)
            assert all(v >= -1e-9 for v in w.values())


def test_optimize_frontier_spans_tradeoff(price_db, cfg):
    """With frontier=true we get ≥ 2 points, from pure-gain → lower-vol corner."""
    app = create_app(cfg=cfg, db=price_db)
    app.config["TESTING"] = True
    with app.test_client() as c:
        r = c.get("/api/optimize?tickers=A,B&frontier=true&steps=6")
        assert r.status_code == 200
        body = r.get_json()
        assert len(body["points"]) == 6
        first, last = body["points"][0], body["points"][-1]
        assert first["risk_aversion"] == 0.0
        assert last["stats"]["ann_vol"] <= first["stats"]["ann_vol"] + 1e-9
        # Each point independently carries a feasible weight vector.
        for p in body["points"]:
            assert sum(p["weights"].values()) == pytest.approx(1.0, abs=1e-5)


def test_optimize_slices_to_window(price_db, cfg):
    d = price_db
    start = d.load("A").index[10].date().isoformat()
    end = d.load("A").index[-5].date().isoformat()
    app = create_app(cfg=cfg, db=d)
    app.config["TESTING"] = True
    with app.test_client() as c:
        r = c.get(f"/api/optimize?tickers=A,B&start_date={start}&end_date={end}")
        body = r.get_json()
        assert body["start"] >= start
        assert body["end"] <= end
        # n_days should be a subset of the full window's days.
        assert 0 < body["points"][0]["stats"]["n_days"] < len(d.load("A"))


def test_optimize_unavailable_window(price_db, cfg):
    # Force an end_date at/before the first close → no shared window.
    d = price_db
    first = d.load("A").index[0].date().isoformat()
    app = create_app(cfg=cfg, db=d)
    app.config["TESTING"] = True
    with app.test_client() as c:
        r = c.get(f"/api/optimize?tickers=A,B&start_date={first}&end_date={first}")
        # 400 with a readable error (or 400 with all tickers in missing)
        assert r.status_code == 400
        assert "error" in r.get_json()


def test_optimize_recommended_at_requested_lambda(price_db, cfg):
    """When we ask for the frontier at a specific λ, `recommended` reports
    the exact basket at that λ — even if it falls between frontier samples."""
    app = create_app(cfg=cfg, db=price_db)
    app.config["TESTING"] = True
    with app.test_client() as c:
        for lam in ("0", "0.5", "50", "5000"):
            r = c.get(f"/api/optimize?tickers=A,B&frontier=true"
                      f"&steps=4&risk_aversion={lam}")
            assert r.status_code == 200, r.get_json()
            body = r.get_json()
            assert "recommended" in body
            assert body["recommended"]["risk_aversion"] == pytest.approx(
                float(lam))
            # recommended weights are always feasible.
            assert sum(body["recommended"]["weights"].values()) == pytest.approx(
                1.0, abs=1e-5)


def test_optimize_single_point_with_steps_param_ok(price_db, cfg):
    """The front-end always sends a `steps` param; in single-point mode the
    server must ignore it (not reject it). A regression for a 400 that used
    to fire when steps=1 was passed alongside a scalar risk_aversion."""
    app = create_app(cfg=cfg, db=price_db)
    app.config["TESTING"] = True
    with app.test_client() as c:
        for steps in ("1", "2", "10"):
            r = c.get(f"/api/optimize?tickers=A,B&risk_aversion=1&steps={steps}")
            assert r.status_code == 200, steps
            assert len(r.get_json()["points"]) == 1
