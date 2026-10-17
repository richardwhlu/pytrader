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
