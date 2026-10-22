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

  function drawChart(canvas, seriesList, opts = {}) {
    // opts.zeroAnchor: force the y-domain to include 0 and pad it with 10%
    // of the data span on each side, so a 0% baseline reads as the
    // "starts here" reference for re-indexed (pct-gain) charts. Drawn with
    // a distinct dashed accent-colored baseline.
    const zeroAnchor = !!opts.zeroAnchor;
    // seriesList: [{label, color, points:[{t, v}]}...]
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 800;
    const cssH = canvas.clientHeight || 320;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    // Extra bottom padding when we need rotated date labels (xRot is always
    // true in practice for now; keep the knob in case we add a mode later).
    const xRot = true;
    const pad = { l: 56, r: 16, t: 16, b: xRot ? 48 : 28 };

    // flatten points to find domains
    let tMin = Infinity, tMax = -Infinity, vMin = Infinity, vMax = -Infinity;
    const allPoints = [];
    for (const s of seriesList) {
      for (const p of s.points) {
        if (!Number.isFinite(p.v)) continue;
        allPoints.push(p);
        tMin = Math.min(tMin, p.t); tMax = Math.max(tMax, p.t);
        vMin = Math.min(vMin, p.v); vMax = Math.max(vMax, p.v);
      }
    }
    ctx.clearRect(0, 0, cssW, cssH);
    if (!allPoints.length) return;

    let yLo = vMin, yHi = vMax;
    if (zeroAnchor) {
      // Re-indexed (pct-gain) chart. Per your request: make 0% obvious and
      // anchor it roughly in the middle of the plot. Center 0% symmetrically —
      // the side that actually has data fills its half, the empty half is
      // headroom. 10% padding so no line sits glued to the top/bottom edge.
      const M = Math.max(Math.abs(vMin), Math.abs(vMax), 1e-9);
      yHi = M * 1.10;
      yLo = -M * 1.10;
    } else {
      if (yHi - yLo < 1e-9) { yLo -= 0.05; yHi += 0.05; }
      const span = yHi - yLo;
      yLo -= span * 0.05; yHi += span * 0.05;
    }
    if (tMax - tMin < 1) tMax = tMin + 1;

    const plotW = cssW - pad.l - pad.r;
    const plotH = cssH - pad.t - pad.b;
    const xOf = (t) => pad.l + ((t - tMin) / (tMax - tMin)) * plotW;
    const yOf = (v) => pad.t + (1 - (v - yLo) / (yHi - yLo)) * plotH;

    // style constants
    const gridColor = "rgba(255,255,255,0.07)";
    const axisColor = "rgba(255,255,255,0.55)";
    const accent = "#4f8cff";

    // horizontal gridlines at "nice" round numbers. For a zero-anchored
    // chart the domain is symmetric about 0, so 0 is always one of the
    // ticks and gets drawn as the distinct dashed accent baseline.
    function niceStep(span, target) {
      const raw = span / Math.max(1, target);
      const mag = Math.pow(10, Math.floor(Math.log10(raw)));
      const norm = raw / mag;
      let step;
      if (norm < 1.5) step = 1; else if (norm < 3) step = 2;
      else if (norm < 7) step = 5; else step = 10;
      return step * mag;
    }
    const step = niceStep(yHi - yLo, 5);
    const first = Math.ceil(yLo / step) * step;
    const ticksY = [];
    for (let v = first; v <= yHi + step * 1e-6; v += step) ticksY.push(v);

    ctx.font = "11px -apple-system, system-ui, sans-serif";
    for (const v of ticksY) {
      const y = yOf(v);
      const isZero = zeroAnchor && Math.abs(v) < step * 1e-4;
      ctx.strokeStyle = isZero ? accent : gridColor;
      ctx.lineWidth = isZero ? 1.5 : 1;
      if (isZero) ctx.setLineDash([4, 3]); else ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(cssW - pad.r, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = isZero ? accent : axisColor;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      const label = zeroAnchor
        ? (Math.round(v * 100) / 100).toString()
        : v.toFixed(2);
      ctx.fillText(label, pad.l - 6, y);
    }

    // x ticks — rotated -35°, roughly 5–6 labels across the plot width
    const tSpan = tMax - tMin;
    const targetTicks = 6;
    const tickStep = tSpan / Math.max(1, targetTicks - 1);
    const ticks = [];
    for (let i = 0; i < targetTicks; i++) ticks.push(tMin + i * tickStep);
    const ang = -Math.PI / 6;   // -30°, close to 45° but readable
    for (const t of ticks) {
      const x = xOf(t);
      const d = new Date(t);
      const label = d.toLocaleDateString();
      ctx.save();
      ctx.translate(x, cssH - pad.b + 8);
      ctx.rotate(ang);
      ctx.fillStyle = axisColor;
      ctx.textAlign = "right";             // anchor: text right end at the tick
      ctx.textBaseline = "middle";
      ctx.fillText(label, 0, 0);
      ctx.restore();
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
  let histWindow = null;           // last rendered {start, end} window

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
    historyChipEls.clear();
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
        // Lowercased haystack for the filter (ticker + name).
        row.dataset.search = (t.ticker + " " + (t.name || "")).toLowerCase();
        row.append(cb, el("span", "tk", t.ticker), el("span", "nm", t.name),
                   el("span", "end", t.end_date || ""));
        list.append(row);
        // Chip registry: ticker -> chip element, for O(1) show/hide.
        const chip = el("span", "chip");
        chip.dataset.ticker = t.ticker;
        const rm = el("button", null, "×");
        rm.type = "button"; rm.setAttribute("aria-label", "Remove " + t.ticker);
        rm.addEventListener("click", (e) => {
          e.stopPropagation();
          const box = list.querySelector(
            `input[value="${t.ticker}"]`);
          if (box) box.checked = false;
        });
        appendChip(t.ticker, chip, rm);
      }
      // Sensible defaults: check up to the first 4 tickers.
      const boxes = [...list.querySelectorAll("input[type=checkbox]")];
      boxes.slice(0, Math.min(4, boxes.length)).forEach((b) => (b.checked = true));
      applyFilter();
      refreshSelectionUI();
    } catch (e) {
      noCache.textContent = `Could not load cached tickers: ${e.message}`;
    }
  }

  // ---- selection UI: chips + count + filtered state ----
  // ticker -> chip wrapper element. Kept in a Map (not in the DOM) so
  // we can order/show-hide chips in O(selected) without scanning.
  const historyChipEls = new Map();

  function appendChip(ticker, chip, rmBtn) {
    chip.append(document.createTextNode(ticker), rmBtn);
    $("#hist-chips").append(chip);
    historyChipEls.set(ticker, chip);
    // Start hidden; loadHistTickers runs refreshSelectionUI() at the end
    // and will reveal chips for the default-checked tickers only.
    chip.classList.add("hidden-chip");
  }

  function refreshSelectionUI() {
    const chips = $("#hist-chips");
    const selected = histSelected();
    const selSet = new Set(selected);

    // Chips stay in the DOM always (one per available ticker); we just
    // toggle visibility. That's O(tickers) regardless of selection size
    // and avoids the DOM churn of append/remove per change. The visual
    // order matches selection order (insertion order is stable per
    // loadHistTickers, which sorts tickers alphabetically — that's the
    // same order the checkboxes appear in, so it reads naturally).
    for (const [t, chip] of historyChipEls) {
      chip.classList.toggle("hidden-chip", !selSet.has(t));
    }

    // Count line.
    $("#hist-count").textContent =
      `${selected.length} selected` +
      (selected.length ? ` (${histTickers.length} available)` : "");
    return selected.length;
  }

  function applyFilter() {
    const q = ($("#hist-filter")?.value || "").trim().toLowerCase();
    const list = $("#hist-ticker-list");
    const rows = [...list.querySelectorAll(".hist-row")];
    let visible = 0;
    for (const row of rows) {
      const match = !q || (row.dataset.search || "").includes(q);
      row.classList.toggle("hidden-row", !match);
      if (match) visible++;
    }
    $("#hist-empty-filter").hidden = !(q && visible === 0);
    return visible;
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
    btn.disabled = false; // re-enable early; the render below is synchronous & fast
    try {
      histResult = result;
      await renderHist(result);
    } catch (e) {
      // A render-stage throw (e.g. a chart bug) used to escape all the way up,
      // leaving the spinner up and the button disabled — which looked like an
      // endless "Computing…". Surface it and clear the overlay instead.
      console.error("historical render failed", e);
      setStatus(status, `Render error: ${e.message || e}`, true);
      chartBusy($("#hist-chart").parentElement, false);
      btn.disabled = false;
    }
  }

  // Synchronous client-side render of a /api/series result. Split from
  // generateHist() so any exception is caught and reported instead of
  // leaving the UI frozen.
  async function renderHist(result) {
    const status = $("#hist-status");
    const btn = $("#hist-go");
    const selected = histSelected();
    const histBox = $("#hist-chart").parentElement;
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
    drawChart($("#hist-chart"), seriesList, { zeroAnchor: true });

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

    /* ---- post-render: wire up the optimizer for this window ----
       The optimizer re-solves the same window that just rendered, so the
       UI has a stable "what happened in this chart" to optimize against.
       We capture the *exact* window used (start, end) from the payload —
       not the form values — so if the UI later re-runs with a different
       window, the optimizer follows along. */
    histWindow = { start: result.start, end: result.end };
    const rec = $("#opt-go");
    if (rec) rec.disabled = false;
    $("#opt-empty").classList.add("hidden");
  }

  /* ---------- optimizer (mean-variance over the current window) ----------
     Solve max w·μ − λ·wᵀΣw for the tickers selected above, on the window
     that just rendered. The slider maps 0–100 to a *log-scaled* λ so a
     single sweep covers "pure gain" (left) through "fully diversified"
     (right) without leaving huge dead zones at any end.

     When the "distribution" checkbox is on we also ask for the frontier —
     a small table of sample points so the user can see the tradeoff curve
     they're picking on. */
  const _OPT = {
    // Slider → λ on a log scale. 0 (leftmost) is the literal 0.0 so the
    // "pure gain" endpoint is exactly reproducible. Values 1–100 span
    // roughly 3 orders of magnitude (≈ 0.01 to ≈ 1000), which covers
    // realistic daily-var scale for equity baskets over a month.
    slideToLam: (v) => (v <= 0 ? 0.0 : Math.pow(10, (v / 100) * 3 - 2)),
    lamToSlide: (lam) => {
      if (lam === 0) return 0;
      const raw = (Math.log10(lam) + 2) / 3 * 100;
      return Math.max(0, Math.min(100, Math.round(raw)));
    },
  };

  async function optimizeBasket() {
    const out = $("#opt-out");
    const empty = $("#opt-empty");
    const btn = $("#opt-go");
    if (!histWindow) {
      empty.classList.remove("hidden");
      out.classList.add("hidden");
      return;
    }
    const selected = histSelected();
    if (!selected.length) {
      empty.classList.remove("hidden");
      out.classList.add("hidden");
      return;
    }
    btn.disabled = true;
    const wantFrontier = $("#opt-frontier").checked;
    const lam = _OPT.slideToLam(Number($("#opt-risk").value));
    const qs = new URLSearchParams({
      tickers: selected.join(","),
      start_date: histWindow.start || "",
      end_date: histWindow.end || "",
      risk_aversion: lam.toExponential(3),
      frontier: wantFrontier ? "true" : "false",
      steps: wantFrontier ? "10" : "1",
    });
    let body;
    try {
      body = await api(`/api/optimize?${qs.toString()}`);
    } catch (e) {
      empty.classList.remove("hidden");
      empty.textContent = `Optimizer error: ${e.message || e}`;
      out.classList.add("hidden");
      btn.disabled = false;
      return;
    }
    renderOptimize(body, lam);
    btn.disabled = false;
  }

  function renderOptimize(body, requestedLam) {
    const out = $("#opt-out");
    const empty = $("#opt-empty");
    if (!body.points || !body.points.length) return;
    const recommended = body.recommended || body.points[0];

    // Frontier canvas (only when we asked for it).
    const frontierWrap = $("#opt-frontier-wrap");
    const frontierCv = $("#opt-frontier");
    if (frontierWrap && body.points.length > 1) {
      frontierWrap.classList.remove("hidden");
      drawFrontier(frontierCv, body.points);
      // Frontier sample list: λ, ret, vol, top 2 by weight.
      const ul = $("#opt-frontier-list");
      ul.innerHTML = "";
      for (const p of body.points) {
        const top = Object.entries(p.weights)
          .sort((a, b) => b[1] - a[1]).slice(0, 2);
        const li = document.createElement("li");
        li.append(
          el("span", "opt-f-lam", `λ=${p.risk_aversion < 1 ? p.risk_aversion.toFixed(3) : p.risk_aversion.toFixed(1)}`),
          el("span", "opt-f-vol", `${(p.stats.ann_vol * 100).toFixed(1)}% vol`),
          el("span", "opt-f-ret", `${fmtPctSigned(p.stats.total_return * 100)}`),
          el("span", "opt-f-top", top.map(([t, w]) => `${t} ${(w*100).toFixed(0)}%`).join(", ")),
        );
        ul.append(li);
      }
    } else if (frontierWrap) {
      frontierWrap.classList.add("hidden");
    }

    // Recommended basket table.
    const tbody = $("#opt-table-body");
    tbody.innerHTML = "";
    const weights = Object.entries(recommended.weights)
      .sort((a, b) => b[1] - a[1]);
    const total = recommended.stats.total_return;
    for (const [t, w] of weights) {
      const contrib = w * total;
      const tr = document.createElement("tr");
      tr.append(
        el("td", "ticker", t),
        el("td", "num", w.toFixed(3)),
        el("td", "num " + (contrib >= 0 ? "up" : "down"),
           fmtPctSigned(contrib * 100)),
      );
      tbody.append(tr);
    }

    // Stat block.
    const stats = $("#opt-stats");
    const s = recommended.stats;
    stats.innerHTML = "";
    const pairs = [
      ["Return", fmtPctSigned(s.total_return * 100)],
      ["Ann vol",    (s.ann_vol * 100).toFixed(1) + "%"],
      ["Sharpe",     (Number.isFinite(s.sharpe) ? s.sharpe.toFixed(2) : "–")],
      ["Max DD",     (s.max_drawdown * 100).toFixed(1) + "%"],
      ["Positions",  String(s.n_positions)],
      ["Days",       String(s.n_days)],
    ];
    for (const [k, v] of pairs) {
      const d = el("div", "opt-stat");
      d.append(el("div", "opt-stat-k", k), el("div", "opt-stat-v", v));
      stats.append(d);
    }

    out.classList.remove("hidden");
    empty.classList.add("hidden");
  }

  // Draw the gain↔vol trade-off scatter. x = annualised vol (lower is
  // "smoother"), y = total return over the window. One dot per frontier
  // sample, coloured by slider proximity to the requested λ. The pure-gain
  // corner (λ=0) is highlighted with a ring. This is the "distribution"
  // the user asked for — see how much gain you're trading to buy smoothness.
  function drawFrontier(canvas, points) {
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 420;
    const cssH = canvas.clientHeight || 220;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const pad = { l: 56, r: 16, t: 16, b: 40 };
    const xs = points.map((p) => p.stats.ann_vol);
    const ys = points.map((p) => p.stats.total_return);
    let xMin = Math.min(...xs), xMax = Math.max(...xs);
    let yMin = Math.min(...ys), yMax = Math.max(...ys);
    if (xMax - xMin < 1e-9) { xMin -= 0.01; xMax += 0.01; }
    if (yMax - yMin < 1e-9) { yMin -= 0.01; yMax += 0.01; }
    const px = (x) => pad.l + (x - xMin) / (xMax - xMin) * (cssW - pad.l - pad.r);
    const py = (y) => pad.t + (1 - (y - yMin) / (yMax - yMin)) * (cssH - pad.t - pad.b);

    ctx.clearRect(0, 0, cssW, cssH);
    // Axis frames.
    ctx.strokeStyle = "rgba(255,255,255,0.10)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, cssH - pad.b);
    ctx.lineTo(cssW - pad.r, cssH - pad.b);
    ctx.stroke();

    // A few ticks.
    ctx.fillStyle = "rgba(255,255,255,0.55)";
    ctx.font = "10px -apple-system, system-ui, sans-serif";
    const xTicks = 4;
    for (let i = 0; i <= xTicks; i++) {
      const v = xMin + (xMax - xMin) * (i / xTicks);
      ctx.fillStyle = "rgba(255,255,255,0.35)";
      ctx.fillText(`${(v * 100).toFixed(0)}%`, px(v) - 8, cssH - pad.b + 14);
    }
    const yTicks = 4;
    for (let i = 0; i <= yTicks; i++) {
      const v = yMin + (yMax - yMin) * (i / yTicks);
      ctx.fillStyle = "rgba(255,255,255,0.35)";
      ctx.textAlign = "right";
      ctx.fillText(`${(v * 100).toFixed(0)}%`, pad.l - 6, py(v) + 3);
      ctx.textAlign = "left";
    }

    // Curve.
    ctx.strokeStyle = "rgba(86,189,232,0.55)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < points.length; i++) {
      const x = px(points[i].stats.ann_vol);
      const y = py(points[i].stats.total_return);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Dots.
    for (const p of points) {
      const x = px(p.stats.ann_vol);
      const y = py(p.stats.total_return);
      ctx.fillStyle = p.risk_aversion === 0 ? "#f59e0b" : "#4f8cff";
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
      if (p.risk_aversion === 0) {
        ctx.strokeStyle = "#f59e0b";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }

  function bindOptimizer() {
    const slider = $("#opt-risk");
    const readout = $("#opt-risk-val");
    const lamReadout = () => {
      if (!readout) return;
      const lam = _OPT.slideToLam(Number(slider.value));
      readout.textContent = lam === 0 ? "0 (pure gain)"
        : lam < 1 ? lam.toFixed(3)
        : lam < 100 ? lam.toFixed(1)
        : String(lam);
    };
    slider.addEventListener("input", lamReadout);
    lamReadout();
    $("#opt-go").addEventListener("click", optimizeBasket);
    $("#opt-frontier").addEventListener("change", () => {
      // Toggle frontier-only mode live (no refetch yet).
    });
  }

  function bindHistorical() {
    const list = $("#hist-ticker-list");
    const go = $("#hist-go");
    const filter = $("#hist-filter");
    let tId = null;                // debounce timer

    const schedule = (fn, ms = 120) => {
      if (tId) clearTimeout(tId);
      tId = setTimeout(() => { tId = null; fn(); }, ms);
    };

    const revalidate = () => {
      refreshSelectionUI();
      go.disabled = histSelected().length === 0;
    };

    // Checkbox change -> refresh chips + count + go-button.
    list.addEventListener("change", revalidate);

    // Filter input -> hide non-matching rows (debounced so we don't
    // re-do the work for each keystroke on a large list).
    if (filter) {
      filter.addEventListener("input", schedule(() => {
        applyFilter();
        // Chips are unchanged (they're independent of the filter), but the
        // "matches nothing" empty-state needs to be up to date.
      }, 80));
    }

    go.addEventListener("click", generateHist);

    // "Select all" = set every *visible* row's checkbox. If a filter is
    // active, that's only the matching rows; otherwise, all of them.
    $("#hist-all").addEventListener("click", () => {
      [...list.querySelectorAll(".hist-row")]
        .filter((r) => !r.classList.contains("hidden-row"))
        .forEach((r) => { r.querySelector("input").checked = true; });
      revalidate();
    });
    // "Clear" = uncheck every checkbox (selected or not). Chips and the
    // filter follow along in revalidate().
    $("#hist-none").addEventListener("click", () => {
      list.querySelectorAll("input[type=checkbox]")
            .forEach((b) => (b.checked = false));
      revalidate();
    });

    // If the user typed in the filter but no row matched, still let them
    // see the "no matches" hint even if chips are still set.
    revalidate();
  }

  /* ---------- boot ---------- */
  document.addEventListener("DOMContentLoaded", async () => {
    bindForm();
    loadUniverse();
    bindHistorical();
    bindOptimizer();
    loadHistTickers();
    await loadRuns();
    window.addEventListener(
      "resize",
      () => requestAnimationFrame(() => refreshChart()),
      { passive: true }
    );
  });
})();
