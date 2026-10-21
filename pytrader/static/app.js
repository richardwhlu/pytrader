// PyTrader dashboard — dependency-free vanilla JS.
// API: GET /api/universe | GET /api/runs | GET /api/runs/<id>
//      GET /api/compare?run_ids=a,b | POST /api/backtest | GET /api/health

(() => {
  "use strict";

  /* ---------- helpers ---------- */
  const $ = (sel) => document.querySelector(sel);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };
  const fmtPct = (v, digits = 2) =>
    v == null || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(digits)}%`;
  const fmtNum = (v, digits = 2) =>
    v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(digits);
  const fmtDate = (s) => {
    const d = new Date(s);
    return Number.isNaN(d.getTime()) ? s : d.toLocaleDateString();
  };
  const esc = (s) => el("span", null, String(s ?? ""));

  async function api(path, opts = {}) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...opts,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  }

  function setBusy(node, busy) {
    if (!node) return;
    const had = node.querySelector(".spinner");
    if (busy) {
      if (had) return;
      const s = document.createElement("span");
      s.className = "spinner";
      s.setAttribute("aria-hidden", "true");
      node.insertBefore(s, node.firstChild);
      node.classList.add("busy");
    } else {
      if (had) had.remove();
      node.classList.remove("busy");
    }
  }

  function setStatus(node, msg, isErr = false, busy = false) {
    if (!node) return;
    node.textContent = msg || "";
    node.classList.toggle("err", !!isErr);
    setBusy(node, busy); // after textContent so the spinner survives
  }

  /* ---------- canvas line chart ---------- */
  const PALETTE = ["#4f8cff", "#f59e0b", "#22c55e", "#ec4899",
                   "#8b5cf6", "#14b8a6", "#ef4444", "#eab308"];

  function drawChart(canvas, seriesList) {
    // seriesList: [{label, color, points:[{t, v}]}...]
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 800;
    const cssH = canvas.clientHeight || 320;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const pad = { l: 56, r: 16, t: 16, b: 28 };

    // flatten points to find domains
    let tMin = Infinity, tMax = -Infinity, vMin = Infinity, vMax = -Infinity;
    const allPoints = [];
    for (const s of seriesList) {
      for (const p of s.points) {
        if (!Number.isFinite(p.v)) continue;
        allPoints.push(p);
        tMin = Math.min(tMin, p.t); tMax = Math.max(tMax, p.t);
        vMin = Math.min(vMin, p.v); vMax = Math.max(vMax, v.v);
      }
    }
    ctx.clearRect(0, 0, cssW, cssH);
    if (!allPoints.length) return;
    if (vMax - vMin < 1e-9) { vMin -= 0.05; vMax += 0.05; }
    const vSpan = vMax - vMin;
    vMin -= vSpan * 0.05; vMax += vSpan * 0.05;
    if (tMax - tMin < 1) tMax = tMin + 1;

    const plotW = cssW - pad.l - pad.r;
    const plotH = cssH - pad.t - pad.b;
    const xOf = (t) => pad.l + ((t - tMin) / (tMax - tMin)) * plotW;
    const yOf = (v) => pad.t + (1 - (v - vMin) / (vMax - vMin)) * plotH;

    // style
    const gridColor = "rgba(255,255,255,0.07)";
    const axisColor = "rgba(255,255,255,0.45)";
    ctx.font = "11px -apple-system, system-ui, sans-serif";

    // horizontal gridlines (5)
    const nGrid = 4;
    for (let i = 0; i <= nGrid; i++) {
      const v = vMin + (i / nGrid) * (vMax - vMin);
      const y = yOf(v);
      ctx.strokeStyle = gridColor;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(cssW - pad.r, y);
      ctx.stroke();
      ctx.fillStyle = axisColor;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(v.toFixed(2), pad.l - 6, y);
    }
    // x ticks (first, mid, last as dates)
    const dates = [tMin, (tMin + tMax) / 2, tMax];
    for (const t of dates) {
      const x = xOf(t);
      ctx.fillStyle = axisColor;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const d = new Date(t);
      ctx.fillText(d.toLocaleDateString(), x, cssH - pad.b + 6);
    }

    // lines
    for (const s of seriesList) {
      const pts = s.points
        .filter((p) => Number.isFinite(p.v))
        .sort((a, b) => a.t - b.t);
      if (pts.length < 2) continue;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.moveTo(xOf(pts[0].t), yOf(pts[0].v));
      for (let i = 1; i < pts.length; i++) {
        ctx.lineTo(xOf(pts[i].t), yOf(pts[i].v));
      }
      ctx.stroke();
    }
  }

  /* ---------- chart "computing" overlay ---------- */
  let _overlayEls = new WeakMap(); // box -> {ov, msg}
  function chartBusy(box, on, label) {
    if (!box) return;
    let ref = _overlayEls.get(box);
    if (on && !ref) {
      const ov = el("div", "chart-busy hidden");
      const spin = el("div", "spinner big", null);
      const msg = el("div", "chart-busy-msg", label || "Computing…");
      ov.append(spin, msg);
      box.append(ov);
      ref = { ov, msg };
      _overlayEls.set(box, ref);
    }
    if (!ref) return;
    if (label) ref.msg.textContent = label;
    ref.ov.classList.toggle("hidden", !on);
  }

  /* ---------- state ---------- */
  const state = {
    runs: [],
    selected: new Set(),   // run_ids to plot
  };

  /* ---------- runs table ---------- */
  const fmt = (m) => ({
    Days: m.n_days,
    Tickers: m.n_tickers,
    Sharpe: fmtNum(m.sharpe, 2),
    CAGR: fmtPct(m.cagr),
    MaxDD: fmtPct(m.max_drawdown),
    Return: fmtPct(m.total_return),
    AnnVol: fmtPct(m.ann_vol),
    IR: fmtNum(m.benchmark_ir, 2),
  });

  const METRIC_ROWS = [
    ["Sharpe ratio", (r) => fmtNum(r.metrics.sharpe, 3)],
    ["CAGR", (r) => fmtPct(r.metrics.cagr)],
    ["Total return", (r) => fmtPct(r.metrics.total_return)],
    ["Annual vol", (r) => fmtPct(r.metrics.ann_vol)],
    ["Max drawdown", (r) => fmtPct(r.metrics.max_drawdown)],
    ["Sortino", (r) => fmtNum(r.metrics.sortino, 2)],
    ["Hit rate", (r) => fmtPct(r.metrics.hit_rate)],
    ["Avg turnover", (r) => fmtPct(r.metrics.avg_turnover)],
    ["Benchmark IR", (r) => fmtNum(r.metrics.benchmark_ir, 2)],
    ["Train IC", (r) => fmtNum(r.metrics.train_ic, 3)],
    ["Valid IC", (r) => fmtNum(r.metrics.valid_ic, 3)],
    ["Days", (r) => String(r.metrics.n_days)],
  ];

  function renderRuns() {
    const tbody = $("#runs-tbody");
    tbody.innerHTML = "";
    const has = state.runs.length > 0;
    $("#runs-empty").classList.toggle("hidden", has);
    $("#runs-table-wrap").setAttribute("aria-hidden", String(!has));
    $("#runs-table").querySelector(
      'th[colspan="1"]'
    )?.setAttribute("colspan", "11");

    state.runs.forEach((r, i) => {
      const tr = el("tr");
      // checkbox
      const tdBox = el("td", "col-view");
      const cb = el("input");
      cb.type = "checkbox";
      cb.checked = state.selected.has(r.run_id);
      cb.addEventListener("change", () => {
        if (cb.checked) state.selected.add(r.run_id);
        else state.selected.delete(r.run_id);
        refreshChart();
      });
      tdBox.appendChild(cb);
      tr.appendChild(tdBox);

      tdBox.appendChild(
        el("span", "sr-only", `Select ${r.name} for chart`)
      );
      tr.appendChild(el("td", null, r.name || r.run_id));
      tr.appendChild(el("td", null, fmtDate(r.created_at)));
      const m = r.metrics || {};
      const cells = [
        [m.n_days, "int"], [m.n_tickers, "int"],
        [m.sharpe, "num"], [m.cagr, "pct"], [m.max_drawdown, "pct"],
        [m.total_return, "pct"], [m.ann_vol, "pct"], [m.benchmark_ir, "num"],
      ];
      for (const [v, kind] of cells) {
        const text = kind === "pct" ? fmtPct(v)
                   : kind === "int" ? (v == null ? "–" : String(v))
                   : fmtNum(v, 2);
        tr.appendChild(el("td", "num", text));
      }
      tr.addEventListener("click", (e) => {
        if (e.target !== cb) showDetail(r.run_id);
      });
      tr.tabIndex = 0;
      tr.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && e.target !== cb) showDetail(r.run_id);
      });
      tbody.appendChild(tr);
    });
  }

  /* ---------- chart refresh ---------- */
  async function refreshChart() {
    const status = $("#chart-status");
    const empty = $("#chart-empty");
    const legend = $("#chart-legend");
    legend.innerHTML = "";
    const ids = [...state.selected];
    if (!ids.length) {
      empty.classList.remove("hidden");
      setStatus(status, "");
      drawChart($("#equity-chart"), []);
      return;
    }
    empty.classList.add("hidden");
    setStatus(status, `Loading ${ids.length} run(s)…`, false, true);
    chartBusy($("#chart-box"), true, `Fetching ${ids.length} run(s)…`);
    try {
      const data = await api("/api/compare?run_ids=" + ids.join(","));
      const series = [];
      data.runs.forEach((r) => {
        const color = PALETTE[series.length % PALETTE.length];
        series.push({
          label: r.name || r.run_id,
          color,
          points: (r.equity || []).map(([t, v]) => ({ t, v })),
        });
        const item = el("div", "legend-item");
        item.appendChild(el("span", "swatch", null)).firstChild
          .style.backgroundColor = color;
        item.appendChild(el("span", null, r.name || r.run_id));
        legend.appendChild(item);
      });
      drawChart($("#equity-chart"), series);
      setStatus(status, "");
      chartBusy($("#chart-box"), false);
      renderCompare(data.runs);
    } catch (e) {
      setStatus(status, `Chart error: ${e.message}`, true);
      chartBusy($("#chart-box"), false);
    }
  }

  function renderCompare(runs) {
    const block = $("#compare-block");
    const table = $("#compare-table");
    if (!runs.length) { block.classList.add("hidden"); return; }
    block.classList.remove("hidden");
    const head = ["Metric"].concat(runs.map((r) => r.name || r.run_id));
    table.innerHTML = "";
    const trh = el("tr");
    head.forEach((h) => trh.appendChild(el("th", null, h)));
    table.appendChild(trh);
    for (const [label, fn] of METRIC_ROWS) {
      const tr = el("tr");
      tr.appendChild(el("td", null, label));
      runs.forEach((r) => tr.appendChild(el("td", "num", fn(r))));
      table.appendChild(tr);
    }
  }

  /* ---------- detail ---------- */
  async function showDetail(runId) {
    const status = $("#detail-status");
    const grid = $("#detail-grid");
    setStatus(status, "Loading…", false, true);
    try {
      const r = await api(`/api/runs/${runId}`);
      $("#detail-name").textContent = r.name || runId;
      grid.classList.remove("hidden");
      grid.innerHTML = "";
      const m = (r.metrics && r.metrics.metrics) || r.metrics || {};
      for (const [label, fn] of METRIC_ROWS) {
        const cell = el("div", "detail-cell");
        cell.appendChild(el("div", "detail-label", label));
        cell.appendChild(el("div", "detail-value", fn(r)));
        grid.appendChild(cell);
      }
      setStatus(status, "");
    } catch (e) {
      setStatus(status, e.message, true);
    }
  }

  /* ---------- universe ---------- */
  async function loadUniverse() {
    const status = $("#universe-status");
    const err = $("#universe-error");
    setStatus(status, "Loading…", false, true);
    try {
      const u = await api("/api/universe");
      const sel = u.selected || [];
      const rej = u.rejected || [];
      $("#sel-count").textContent = String(sel.length);
      $("#sel-count").classList.remove("hidden");
      $("#rej-count").textContent = String(rej.length);
      $("#rej-count").classList.toggle("hidden", rej.length === 0);

      renderRows($("#selected-tbody"), sel, [
        (t) => t.ticker, (t) => t.name || "—", (t) => t.sector || "—",
      ]);
      renderRows($("#rejected-tbody"), rej, [
        (t) => t.ticker,
        (t) => (t.reasons && t.reasons.length) ? t.reasons.join("; ") : "—",
      ]);
      setStatus(status, "");
      err.classList.add("hidden");
    } catch (e) {
      setStatus(status, "");
      err.textContent = `Could not load universe: ${e.message}`;
      err.classList.remove("hidden");
    }
  }

  function renderRows(tbody, rows, cells) {
    tbody.innerHTML = "";
    $("#selected-empty").classList.toggle("hidden", tbody === $("#selected-tbody") && rows.length > 0);
    $("#rejected-empty").classList.toggle("hidden", tbody === $("#rejected-tbody") && rows.length > 0);
    for (const r of rows) {
      const tr = el("tr");
      for (const fn of cells) tr.appendChild(el("td", null, fn(r)));
      tbody.appendChild(tr);
    }
  }

  /* ---------- backtest form ---------- */
  function bindForm() {
    const form = $("#backtest-form");
    const status = $("#backtest-status");
    const btn = $("#backtest-submit");
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const name = $("#f-name").value.trim();
      const body = {
        name,
        horizon: parseInt($("#f-horizon").value, 10),
        top_n: parseInt($("#f-topn").value, 10),
        commission_bps: parseFloat($("#f-comm").value),
        slippage_bps: parseFloat($("#f-slip").value),
      };
      btn.disabled = true;
      setStatus(status,
        "Running backtest — training XGBoost, then simulating. " +
        "This can take a minute…", false, true);
      try {
        const r = await api("/api/backtest", {
          method: "POST",
          body: JSON.stringify(body),
        });
        await loadRuns();
        // auto-select the new run so the user sees it in the chart
        state.selected.add(r.run_id);
        refreshChart();
        showDetail(r.run_id);
        setStatus(status, `Done — run "${r.name || r.run_id}" saved.`);
      } catch (err) {
        setStatus(status, `Backtest failed: ${err.message}`, true);
      } finally {
        btn.disabled = false;
      }
    });
  }

  /* ---------- load runs ---------- */
  async function loadRuns() {
    const status = $("#runs-status");
    setStatus(status, "Loading…", false, true);
    try {
      const data = await api("/api/runs");
      state.runs = data.runs || [];
      status && setStatus(status, "");
      renderRuns();
    } catch (e) {
      setStatus(status, `Could not load runs: ${e.message}`, true);
      renderRuns();
    }
  }

  /* ---------- historical re-indexed visualizer ---------- */
  let histTickers = [];            // [{ticker,name,sector,end_date}]
  let histResult = null;           // last /api/series response

  const MSZ = (d) => Date.parse(d + "T00:00:00Z");

  function histSelected() {
    return [...document.querySelectorAll("#hist-ticker-list input[type=checkbox]")
      ].filter((b) => b.checked).map((b) => b.value);
  }

  function parseWeights() {
    const map = new Map();
    for (const line of $("#hist-weights").value.split(/\r?\n/)) {
      const m = line.trim().match(/^([A-Za-z.0-9^\-=]+)\s*=\s*([-+]?[0-9.eE]+)$/);
      if (m) map.set(m[1].toUpperCase(), parseFloat(m[2]));
    }
    return map;
  }

  // Weights rule (documented in the UI): textarea empty -> equal weights.
  // Otherwise a selected ticker's weight is its explicit value if listed,
  // else 0; the whole set is then normalized to sum to 1. The *effective*
  // (normalized) weight is shown in the P&L table.
  function effectiveWeights(selected) {
    const wmap = parseWeights();
    const explicit = new Map();
    for (const t of selected) if (wmap.has(t)) explicit.set(t, wmap.get(t));
    if (!explicit.size) {
      const eq = 1 / (selected.length || 1);
      return selected.map((t) => ({ ticker: t, weight: eq }));
    }
    const weights = selected.map((t) => ({ ticker: t, weight: explicit.get(t) || 0 }));
    const total = weights.reduce((a, b) => a + b.weight, 0) || 1;
    return weights.map((r) => ({ ticker: r.ticker, weight: r.weight / total }));
  }

  // Composite = inner-join of selected series, sum of (weight * pctgain) at
  // each common date. Because every series is ~0 on its base day, this is a
  // clean daily-rebalanced portfolio line.
  function computeComposite(selected, result, normWeights) {
    const w = new Map(normWeights.map((r) => [r.ticker, r.weight]));
    const avail = selected.filter((t) => result.series && result.series[t]);
    if (!avail.length) return null;
    const series = avail.map((t) => new Map(result.series[t].points));
    const days = [...series[0].keys()].sort();
    const out = [];
    for (const d of days) {
      if (!series.every((m) => m.has(d))) continue;
      let v = 0;
      for (let i = 0; i < avail.length; i++) {
        v += (w.get(avail[i]) || 0) * (series[i].get(d) || 0);
      }
      out.push([d, v]);
    }
    return out.length ? out : null;
  }

  function fmtPctSigned(v, digits = 1) {
    if (v == null || Number.isNaN(v)) return "–";
    return `${v >= 0 ? "+" : ""}${v.toFixed(digits)}%`;
  }

  async function loadHistTickers() {
    const list = $("#hist-ticker-list");
    const noCache = $("#hist-no-cache");
    list.innerHTML = "";
    try {
      const data = await api("/api/tickers");
      histTickers = (data.tickers || []).slice().sort((a, b) =>
        a.ticker < b.ticker ? -1 : 1);
      if (!histTickers.length) { noCache.classList.remove("hidden"); return; }
      noCache.classList.add("hidden");
      for (const t of histTickers) {
        const row = el("label", "hist-row");
        const cb = document.createElement("input");
        cb.type = "checkbox"; cb.value = t.ticker;
        row.append(cb, el("span", "tk", t.ticker), el("span", "nm", t.name),
                   el("span", "end", t.end_date || ""));
        list.append(row);
      }
      // Sensible defaults: check up to the first 4 tickers.
      const boxes = [...list.querySelectorAll("input[type=checkbox]")];
      boxes.slice(0, Math.min(4, boxes.length)).forEach((b) => (b.checked = true));
    } catch (e) {
      noCache.textContent = `Could not load cached tickers: ${e.message}`;
    }
  }

  async function generateHist() {
    const status = $("#hist-status");
    const btn = $("#hist-go");
    const selected = histSelected();
    if (!selected.length) { setStatus(status, "Pick at least one ticker.", true); return; }
    const start = $("#hist-start").value;
    const horizon = parseInt($("#hist-horizon").value, 10) || 0;
    btn.disabled = true;
    const histBox = $("#hist-chart").parentElement;
    setStatus(status, `Computing ${selected.length} series…`, false, true);
    chartBusy(histBox, true, `Re-indexing ${selected.length} series…`);
    let result;
    try {
      const qs = new URLSearchParams({ tickers: selected.join(",") });
      if (start) qs.set("start_date", start);
      if (horizon > 0) qs.set("horizon", String(horizon));
      result = await api(`/api/series?${qs.toString()}`);
    } catch (e) {
      setStatus(status, e.message, true);
      chartBusy(histBox, false);
      btn.disabled = false;
      return;
    }
    histResult = result;
    const norm = effectiveWeights(selected);
    const composite = computeComposite(selected, result, norm);

    // ----- chart -----
    const avail = selected.filter((t) => result.series[t]);
    const seriesList = avail.map((t, i) => ({
      label: t,
      color: PALETTE[i % PALETTE.length],
      points: (result.series[t].points || [])
        .map(([d, v]) => ({ t: MSZ(d), v })),
    }));
    if (composite) {
      seriesList.push({
        label: "COMPOSITE",
        color: "#f5f7fa",
        points: composite.map(([d, v]) => ({ t: MSZ(d), v })),
      });
    }
    $("#hist-chart-empty").classList.add("hidden");
    drawChart($("#hist-chart"), seriesList);

    // ----- legend -----
    const legend = $("#hist-legend");
    legend.innerHTML = "";
    for (const s of seriesList) {
      const li = document.createElement("li");
      const sw = el("span", "swatch", "");
      sw.style.background = s.color === "#f5f7fa" ? "" : s.color;
      if (s.color === "#f5f7fa") sw.style.background =
        "repeating-linear-gradient(90deg,#f5f7fa 0 4px,transparent 4px 7px)";
      const nm = el("span", null, s.label === "COMPOSITE"
        ? "Composite (weighted)" : s.label);
      li.append(sw, nm);
      legend.append(li);
    }

    // ----- P&L table -----
    const wmap = new Map(norm.map((r) => [r.ticker, r.weight]));
    const tbody = $("#hist-table-body");
    tbody.innerHTML = "";
    const missing = result.missing || [];
    for (const t of selected) {
      const s = result.series[t];
      const tr = document.createElement("tr");
      if (!s) {
        tr.append(el("td", "ticker", t),
                  el("td", "muted", "not in cache"),
                  el("td", ""), el("td", ""), el("td", ""), el("td", ""));
      } else {
        const ret = s.end_pct;
        const contrib = (wmap.get(t) || 0) * ret;
        tr.append(
          el("td", "ticker", t),
          el("td", null, s.base_date),
          el("td", null, s.points.length ? s.points[s.points.length - 1][0] : ""),
          el("td", "num", (wmap.get(t) || 0).toFixed(3)),
          el("td", "num " + (ret >= 0 ? "up" : "down"), fmtPctSigned(ret)),
          el("td", "num " + (contrib >= 0 ? "up" : "down"), fmtPctSigned(contrib)),
        );
      }
      tbody.append(tr);
    }
    $("#hist-table-empty").classList.add("hidden");

    // ----- composite total -----
    const tot = $("#hist-composite-val");
    if (composite && composite.length) {
      const lastV = composite[composite.length - 1][1];
      tot.textContent = fmtPctSigned(lastV);
      tot.className = "big " + (lastV >= 0 ? "up" : "down");
    } else {
      tot.textContent = "–";
      tot.className = "big";
    }

    setStatus(status,
      `${avail.length}/${selected.length} plotted` +
      (missing.length ? ` · not cached: ${missing.join(", ")}` : "") +
      (composite ? "" : " · no shared dates for composite"),
      !!missing.length);
    chartBusy(histBox, false);
    btn.disabled = false;
  }

  function bindHistorical() {
    const list = $("#hist-ticker-list");
    const go = $("#hist-go");
    const update = () => { go.disabled = histSelected().length === 0; };
    list.addEventListener("change", update);
    go.addEventListener("click", generateHist);
    $("#hist-all").addEventListener("click", () => {
      list.querySelectorAll("input[type=checkbox]").forEach((b) => (b.checked = true));
      update();
    });
    $("#hist-none").addEventListener("click", () => {
      list.querySelectorAll("input[type=checkbox]").forEach((b) => (b.checked = false));
      update();
    });
    update();
  }

  /* ---------- boot ---------- */
  document.addEventListener("DOMContentLoaded", async () => {
    bindForm();
    loadUniverse();
    bindHistorical();
    loadHistTickers();
    await loadRuns();
    window.addEventListener(
      "resize",
      () => requestAnimationFrame(() => refreshChart()),
      { passive: true }
    );
  });
})();
