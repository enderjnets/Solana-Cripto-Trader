#!/usr/bin/env python3
"""
Solana Cripto Trader - Professional Trading Dashboard
Flask backend serving real-time data from bot data files.
Port: 8080
"""

import json
import os
import math
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# ── Data paths ───────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
DATA = BASE / "agents" / "data"
WATCHDOG_LOG = Path("/tmp/solana_watchdog.log")

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

def load_portfolio():
    raw = load_json(DATA / "portfolio.json")
    if not raw:
        return {"positions": [], "capital_usd": 500, "initial_capital": 500,
                "wins": 0, "losses": 0, "total_trades": 0}
    return raw

def load_trade_history():
    """Load all trades from trade_history.json (authoritative full history).
    
    BUG FIX (2026-03-23): Previously used portfolio.json positions as primary source,
    which only contained the last ~10 trades. trade_history.json has 2200+ trades.
    portfolio.json positions are used only for open positions (api/positions).
    """
    # Primary: trade_history.json (full history)
    path = DATA / "trade_history.json"
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            return data
    except Exception:
        pass
    # Fallback: portfolio.json positions
    port = load_json(DATA / "portfolio.json")
    return port.get("positions", [])

def safe_float(v, default=0.0):
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default

# ── HTML template (single-file SPA) ──────────────────────────────────────────
DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Solana Cripto Trader — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js"></script>
<style>
  :root {
    --bg: #0d1117;
    --bg2: #161b22;
    --bg3: #1c2128;
    --border: #30363d;
    --text: #e6edf3;
    --text2: #8b949e;
    --green: #3fb950;
    --red: #f85149;
    --yellow: #d29922;
    --blue: #58a6ff;
    --purple: #bc8cff;
    --orange: #ff7b72;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; }
  a { color: var(--blue); text-decoration: none; }

  /* ── Header ── */
  .header {
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky; top: 0; z-index: 100;
  }
  .header-left { display: flex; align-items: center; gap: 16px; }
  .logo { font-size: 18px; font-weight: 700; color: var(--blue); letter-spacing: -0.5px; }
  .logo span { color: var(--purple); }
  .status-badge {
    display: flex; align-items: center; gap: 6px;
    background: var(--bg3); border: 1px solid var(--border);
    padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600;
  }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; }
  .dot-green { background: var(--green); box-shadow: 0 0 6px var(--green); animation: pulse 2s infinite; }
  .dot-red { background: var(--red); }
  .dot-yellow { background: var(--yellow); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }

  .header-right { display: flex; align-items: center; gap: 12px; }
  .capital-header { font-size: 13px; color: var(--text2); }
  .capital-header .val { color: var(--text); font-weight: 600; font-size: 15px; }
  .capital-header .chg { font-size: 12px; margin-left: 4px; }
  .last-update { font-size: 11px; color: var(--text2); }
  .btn-refresh {
    background: var(--bg3); border: 1px solid var(--border);
    color: var(--blue); padding: 6px 14px; border-radius: 6px;
    cursor: pointer; font-size: 12px; transition: all .2s;
  }
  .btn-refresh:hover { background: var(--blue); color: var(--bg); }
  .mode-badge {
    font-size: 11px; padding: 2px 8px; border-radius: 10px;
    background: rgba(188,140,255,0.15); color: var(--purple); border: 1px solid var(--purple);
  }

  /* ── Layout ── */
  .container { max-width: 1600px; margin: 0 auto; padding: 20px 24px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
  .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
  .grid-7 { display: grid; grid-template-columns: repeat(7, 1fr); gap: 12px; }
  .card {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px;
  }
  .card-title {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1px; color: var(--text2); margin-bottom: 12px;
    display: flex; align-items: center; gap: 6px;
  }
  .card-title .icon { font-size: 14px; }
  section { margin-bottom: 20px; }
  .section-title {
    font-size: 13px; font-weight: 600; color: var(--text2);
    text-transform: uppercase; letter-spacing: 0.5px;
    margin-bottom: 12px; padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 8px;
  }

  /* ── KPI Cards ── */
  .kpi-card {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px;
    display: flex; flex-direction: column; gap: 4px;
  }
  .kpi-label { font-size: 11px; color: var(--text2); font-weight: 500; letter-spacing: 0.3px; }
  .kpi-value { font-size: 22px; font-weight: 700; line-height: 1.2; }
  .kpi-sub { font-size: 11px; color: var(--text2); }
  .kpi-icon { font-size: 20px; align-self: flex-end; opacity: 0.6; margin-top: -8px; }
  .kpi-card.positive { border-left: 3px solid var(--green); }
  .kpi-card.negative { border-left: 3px solid var(--red); }
  .kpi-card.neutral { border-left: 3px solid var(--blue); }
  .kpi-card.warning { border-left: 3px solid var(--yellow); }

  /* Colors */
  .pos { color: var(--green); }
  .neg { color: var(--red); }
  .neu { color: var(--blue); }
  .warn { color: var(--yellow); }

  /* Real-time update animations */
  @keyframes flashGreen {
    0% { background-color: rgba(63, 185, 80, 0.3); }
    100% { background-color: transparent; }
  }
  @keyframes flashRed {
    0% { background-color: rgba(248, 81, 73, 0.3); }
    100% { background-color: transparent; }
  }
  .flash-update { animation: flashGreen 0.5s ease-out; }
  .price-cell { 
    font-family: 'Monaco', 'Consolas', monospace; 
    font-weight: 600;
    transition: color 0.3s ease;
  }
  .price-up { color: var(--green) !important; }
  .price-down { color: var(--red) !important; }

  /* ── Charts ── */
  .chart-container { position: relative; width: 100%; }
  .chart-sm { height: 200px; }
  .chart-md { height: 260px; }
  .chart-lg { height: 340px; }
  canvas { max-width: 100%; }

  /* ── Tables ── */
  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  thead th {
    text-align: left; padding: 8px 10px; font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.5px; color: var(--text2);
    border-bottom: 1px solid var(--border); white-space: nowrap;
  }
  tbody tr { border-bottom: 1px solid rgba(48,54,61,0.5); transition: background .15s; }
  tbody tr:hover { background: rgba(88,166,255,0.04); }
  tbody td { padding: 8px 10px; white-space: nowrap; }
  .badge {
    display: inline-block; padding: 2px 7px; border-radius: 4px;
    font-size: 10px; font-weight: 600; text-transform: uppercase;
  }
  .badge-long { background: rgba(63,185,80,0.15); color: var(--green); }
  .badge-short { background: rgba(248,81,73,0.15); color: var(--red); }
  .badge-win { background: rgba(63,185,80,0.12); color: var(--green); }
  .badge-loss { background: rgba(248,81,73,0.12); color: var(--red); }
  .badge-hold { background: rgba(88,166,255,0.12); color: var(--blue); }

  /* ── Pagination ── */
  .pagination { display: flex; align-items: center; gap: 8px; margin-top: 12px; justify-content: flex-end; }
  .page-btn {
    background: var(--bg3); border: 1px solid var(--border);
    color: var(--text2); padding: 4px 10px; border-radius: 4px;
    cursor: pointer; font-size: 11px;
  }
  .page-btn:hover, .page-btn.active { background: var(--blue); color: var(--bg); border-color: var(--blue); }
  .page-info { font-size: 11px; color: var(--text2); }

  /* ── Filters ── */
  .filters { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
  select, input {
    background: var(--bg3); border: 1px solid var(--border);
    color: var(--text); padding: 5px 10px; border-radius: 6px;
    font-size: 12px; outline: none;
  }
  select:focus, input:focus { border-color: var(--blue); }

  /* ── Log ── */
  .log-container {
    background: #0a0e14; border: 1px solid var(--border);
    border-radius: 8px; padding: 12px; font-family: 'Consolas', monospace;
    font-size: 11px; max-height: 280px; overflow-y: auto;
  }
  .log-line { padding: 2px 0; line-height: 1.6; }
  .log-ok { color: var(--green); }
  .log-err { color: var(--red); }
  .log-warn { color: var(--yellow); }
  .log-info { color: var(--text2); }
  .log-auto { font-size: 10px; color: var(--text2); margin-top: 8px; }

  /* ── Advanced Metrics ── */
  .metric-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(48,54,61,0.5); }
  .metric-row:last-child { border-bottom: none; }
  .metric-label { color: var(--text2); font-size: 12px; }
  .metric-value { font-weight: 600; font-size: 13px; }

  /* ── Equity toolbar ── */
  .chart-toolbar { display: flex; gap: 6px; margin-bottom: 8px; align-items: center; }
  .tb-btn {
    background: var(--bg3); border: 1px solid var(--border);
    color: var(--text2); padding: 3px 9px; border-radius: 4px;
    cursor: pointer; font-size: 11px;
  }
  .tb-btn:hover, .tb-btn.active { background: var(--blue); color: var(--bg); border-color: var(--blue); }
  .tb-sep { flex: 1; }

  /* ── Discrepancy Warning ── */
  .discrepancy-banner {
    background: rgba(210,153,34,0.08); border: 1px solid rgba(210,153,34,0.4);
    border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;
  }
  .discrepancy-banner .disc-title {
    font-size: 12px; font-weight: 700; color: var(--yellow); margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px;
  }
  .discrepancy-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
  }
  .disc-item { display: flex; flex-direction: column; gap: 2px; }
  .disc-label { font-size: 10px; color: var(--text2); }
  .disc-value { font-size: 14px; font-weight: 600; }
  .disc-note { font-size: 10px; color: var(--text2); margin-top: 8px; line-height: 1.5; }
  @media (max-width: 600px) {
    .discrepancy-grid { grid-template-columns: repeat(2, 1fr); }
  }

  /* ── Empty state ── */
  .empty { text-align: center; padding: 30px; color: var(--text2); font-size: 13px; }

  /* ── Responsive ── */
  @media (max-width: 1200px) {
    .grid-7 { grid-template-columns: repeat(4, 1fr); }
  }
  @media (max-width: 900px) {
    .grid-3, .grid-4 { grid-template-columns: 1fr 1fr; }
    .grid-7 { grid-template-columns: repeat(3, 1fr); }
    .header { flex-wrap: wrap; gap: 8px; }
  }
  @media (max-width: 600px) {
    .grid-2 { grid-template-columns: 1fr; }
    .grid-3, .grid-4, .grid-7 { grid-template-columns: 1fr 1fr; }
    .container { padding: 12px; }
    .header-right { flex-wrap: wrap; }
    .chart-lg { height: 220px; }
    .chart-md { height: 180px; }
    .chart-sm { height: 160px; }
    table { font-size: 11px; }
    thead th, tbody td { padding: 5px 6px; }
  }
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="header-left">
    <div class="logo">◎ Solana <span>Cripto</span> Trader</div>
    <div class="status-badge" id="statusBadge">
      <div class="status-dot dot-green" id="statusDot"></div>
      <span id="statusText">ACTIVE</span>
    </div>
    <span class="mode-badge" id="modeBadge">PAPER</span>
  </div>
  <div class="header-right">
    <div class="capital-header">
      Capital: <span class="val" id="headerCapital">$0.00</span>
      <span class="chg" id="headerChg">+0.00%</span>
    </div>
    <div class="last-update" id="lastUpdate">Actualizado: —</div>
    <button class="btn-refresh" onclick="refreshAll()">⟳ Refresh</button>
  </div>
</div>

<div class="container">

  <!-- KPI ROW -->
  <section>
    <div class="grid-7" id="kpiRow">
      <div class="kpi-card neutral">
        <div class="kpi-label">Capital</div>
        <div class="kpi-value neu" id="kpiCapital">$—</div>
        <div class="kpi-sub" id="kpiCapitalSub">—</div>
      </div>
      <div class="kpi-card" id="kpiPnlCard">
        <div class="kpi-label">P&amp;L Total</div>
        <div class="kpi-value" id="kpiPnl">$—</div>
        <div class="kpi-sub" id="kpiPnlSub">—</div>
      </div>
      <div class="kpi-card neutral">
        <div class="kpi-label">Win Rate <span style="font-size:9px;color:var(--text2);">(excl. flat)</span></div>
        <div class="kpi-value" id="kpiWinRate">—%</div>
        <div class="kpi-sub" id="kpiWinRateSub">W/L: —/—</div>
        <div class="kpi-sub" id="kpiWinRateExcl" style="margin-top:2px;font-size:10px;color:var(--text2);">incl. flat: —%</div>
      </div>
      <div class="kpi-card neutral">
        <div class="kpi-label">Profit Factor</div>
        <div class="kpi-value" id="kpiPF">—</div>
        <div class="kpi-sub">ratio ganancia/pérdida</div>
      </div>
      <div class="kpi-card neutral">
        <div class="kpi-label">Total Trades</div>
        <div class="kpi-value" id="kpiTrades">—</div>
        <div class="kpi-sub" id="kpiTradesSub">—</div>
        <div class="kpi-sub" id="kpiFlatSub" style="margin-top:2px;font-size:10px;color:var(--yellow);">flat: —</div>
      </div>
      <div class="kpi-card" id="kpiDdCard">
        <div class="kpi-label">Drawdown</div>
        <div class="kpi-value" id="kpiDD">—%</div>
        <div class="kpi-sub">desde el pico</div>
      </div>
      <div class="kpi-card neutral">
        <div class="kpi-label">Posiciones</div>
        <div class="kpi-value" id="kpiPos">—</div>
        <div class="kpi-sub" id="kpiPosSub">abiertas</div>
      </div>
    </div>
  </section>

  <!-- POST-RESET section removed: main KPIs now show post-reset data -->

  <!-- ACCOUNTING DISCREPANCY BANNER -->
  <div class="discrepancy-banner" id="discBanner" style="display:none;">
    <div class="disc-title">⚠️ Discrepancia Contable Detectada</div>
    <div class="discrepancy-grid">
      <div class="disc-item">
        <span class="disc-label">Cambio Real de Capital</span>
        <span class="disc-value warn" id="discRealChange">—</span>
      </div>
      <div class="disc-item">
        <span class="disc-label">PnL Registrado en Trades</span>
        <span class="disc-value" id="discRecordedPnl">—</span>
      </div>
      <div class="disc-item">
        <span class="disc-label">Gap No Contabilizado</span>
        <span class="disc-value neg" id="discGap">—</span>
      </div>
      <div class="disc-item">
        <span class="disc-label">EMERGENCY_CLOSEs (pnl=0)</span>
        <span class="disc-value warn" id="discEmergency">—</span>
      </div>
    </div>
    <div class="disc-note" id="discNote">
      ℹ️ Diferencia entre capital total y suma de PnL registrados. Puede incluir fees acumuladas no contabilizadas en ciclos anteriores. En EMERGENCY_CLOSE el margen se devuelve correctamente; la discrepancia proviene de fees de salida históricas. Capital actual ($<span id="discCapNote">—</span>) refleja contabilidad post-fix. Gap: <span id="discGapNote">—</span>
    </div>
  </div>

  <!-- EQUITY + DISTRIBUTIONS -->
  <section>
    <div class="grid-2">
      <!-- Equity Curve -->
      <div class="card" style="grid-column: span 1;">
        <div class="card-title"><span class="icon">📈</span> Equity Curve</div>
        <div class="chart-toolbar">
          <button class="tb-btn active" onclick="setEquityRange('all', this)">All</button>
          <button class="tb-btn" onclick="setEquityRange(50, this)">50T</button>
          <button class="tb-btn" onclick="setEquityRange(100, this)">100T</button>
          <button class="tb-btn" onclick="setEquityRange(200, this)">200T</button>
          <div class="tb-sep"></div>
          <button class="tb-btn" onclick="resetZoom()">↺ Reset</button>
        </div>
        <div class="chart-container chart-lg">
          <canvas id="equityChart"></canvas>
        </div>
      </div>

      <!-- Right column: pie + bar -->
      <div style="display:flex; flex-direction:column; gap:16px;">
        <div class="card">
          <div class="card-title"><span class="icon">🥧</span> Wins vs Losses</div>
          <div class="chart-container chart-sm">
            <canvas id="pieChart"></canvas>
          </div>
        </div>
        <div class="card">
          <div class="card-title"><span class="icon">📊</span> P&amp;L por Símbolo</div>
          <div class="chart-container chart-sm">
            <canvas id="symbolChart"></canvas>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- P&L HISTOGRAM -->
  <section>
    <div class="card">
      <div class="card-title"><span class="icon">📉</span> Distribución de P&amp;L por Trade</div>
      <div class="chart-container chart-md">
        <canvas id="histChart"></canvas>
      </div>
    </div>
  </section>

  <!-- OPEN POSITIONS -->
  <section>
    <div class="section-title">⚡ Posiciones Abiertas</div>
    <div class="card">
      <div class="table-wrap" id="positionsTable">
        <div class="empty">Cargando posiciones...</div>
      </div>
    </div>
  </section>

  <!-- TRADE HISTORY -->
  <section>
    <div class="section-title">📋 Historial de Trades</div>
    <div class="card">
      <div class="filters">
        <select id="filterSymbol" onchange="filterTrades()">
          <option value="">Todos los símbolos</option>
        </select>
        <select id="filterResult" onchange="filterTrades()">
          <option value="">Todos los resultados</option>
          <option value="win">Ganadores</option>
          <option value="loss">Perdedores</option>
          <option value="flat">Sin cambio (FLAT)</option>
          <option value="nonflat">Solo W/L (excl. FLAT)</option>
        </select>
        <select id="filterDir" onchange="filterTrades()">
          <option value="">Long &amp; Short</option>
          <option value="long">Long</option>
          <option value="short">Short</option>
        </select>
      </div>
      <div class="table-wrap" id="tradesTable">
        <div class="empty">Cargando trades...</div>
      </div>
      <div class="pagination" id="tradesPagination"></div>
    </div>
  </section>

  <!-- BOTTOM ROW: LOG + ADVANCED METRICS -->
  <section>
    <div class="grid-2">
      <!-- Watchdog Log -->
      <div class="card">
        <div class="card-title"><span class="icon">🔍</span> Watchdog Log</div>
        <div class="log-container" id="logContainer">
          <div class="log-info">Cargando log...</div>
        </div>
        <div class="log-auto">🔄 Auto-refresh cada 30s</div>
      </div>

      <!-- Advanced Metrics -->
      <div class="card">
        <div class="card-title"><span class="icon">🧮</span> Métricas Avanzadas</div>
        <div id="advancedMetrics">
          <div class="metric-row">
            <span class="metric-label">Sharpe Ratio</span>
            <span class="metric-value" id="mSharpe">—</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Avg Win</span>
            <span class="metric-value pos" id="mAvgWin">—</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Avg Loss</span>
            <span class="metric-value neg" id="mAvgLoss">—</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Mejor Trade</span>
            <span class="metric-value pos" id="mBestTrade">—</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Peor Trade</span>
            <span class="metric-value neg" id="mWorstTrade">—</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Max Racha Ganadora</span>
            <span class="metric-value pos" id="mMaxWinStreak">—</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Max Racha Perdedora</span>
            <span class="metric-value neg" id="mMaxLossStreak">—</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Profit Factor</span>
            <span class="metric-value" id="mPF2">—</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Win Rate (excl. flat)</span>
            <span class="metric-value" id="mWRExcl">—</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Trades FLAT/EMERGENCY</span>
            <span class="metric-value warn" id="mFlatTrades">—</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Capital Inicial</span>
            <span class="metric-value" id="mInitCap">—</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Retorno Total</span>
            <span class="metric-value" id="mReturn">—</span>
          </div>
        </div>
      </div>
    </div>
  </section>

</div><!-- /container -->

<script>
// ── State ──────────────────────────────────────────────────────────────────
let allTrades = [];
let filteredTrades = [];
let currentPage = 1;
const PAGE_SIZE = 50;
let equityChart = null, pieChart = null, symbolChart = null, histChart = null;
let equityAllData = { labels: [], datasets: [] };

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  refreshAll();
  // Refresh completo cada 30 segundos
  setInterval(refreshAll, 30000);
  // Refresh de posiciones cada 3 segundos (tiempo real)
  setInterval(loadPositionsRealtime, 3000);
});

async function refreshAll() {
  try {
    await Promise.all([
      loadStats(),
      loadEquity(),
      loadPositions(),
      loadTrades(),
      loadLog()
    ]);
    document.getElementById('lastUpdate').textContent =
      'Actualizado: ' + new Date().toLocaleTimeString('es-MX', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch(e) {
    console.error('Refresh error:', e);
  }
}

// ── API Calls ──────────────────────────────────────────────────────────────
async function loadStats() {
  const r = await fetch('/api/stats');
  const d = await r.json();

  // Header
  document.getElementById('headerCapital').textContent = fmt$(d.capital_usd);
  const chgEl = document.getElementById('headerChg');
  chgEl.textContent = fmtPct(d.return_pct, true);
  chgEl.className = 'chg ' + (d.return_pct >= 0 ? 'pos' : 'neg');
  document.getElementById('modeBadge').textContent = (d.mode || 'PAPER').toUpperCase().replace('_', ' ');

  // Status
  const status = d.portfolio_status || 'ACTIVE';
  const dot = document.getElementById('statusDot');
  document.getElementById('statusText').textContent = status;
  dot.className = 'status-dot ' + (status === 'ACTIVE' ? 'dot-green' : status === 'PAUSED' ? 'dot-yellow' : 'dot-red');

  // KPIs
  document.getElementById('kpiCapital').textContent = fmt$(d.capital_usd);
  document.getElementById('kpiCapitalSub').textContent = 'Inicial: ' + fmt$(d.initial_capital);

  const pnl = d.total_pnl;
  const pnlEl = document.getElementById('kpiPnl');
  const pnlSubEl = document.getElementById('kpiPnlSub');
  const pnlCard = document.getElementById('kpiPnlCard');
  pnlEl.textContent = (pnl >= 0 ? '+' : '') + fmt$(pnl);
  pnlEl.className = 'kpi-value ' + (pnl >= 0 ? 'pos' : 'neg');
  pnlSubEl.textContent = fmtPct(d.return_pct, true);
  pnlSubEl.className = 'kpi-sub ' + (pnl >= 0 ? 'pos' : 'neg');
  pnlCard.className = 'kpi-card ' + (pnl >= 0 ? 'positive' : 'negative');

  // Primary: excl. flat win rate (signal quality), secondary: gross incl. flat
  const wr = d.win_rate_excl_flat || 0;  // excl. flat is the meaningful metric
  const wrGross = d.win_rate || 0;        // gross incl. flat shown as secondary
  const wrEl = document.getElementById('kpiWinRate');
  wrEl.textContent = fmtNum(wr, 1) + '%';
  wrEl.className = 'kpi-value ' + (wr >= 50 ? 'pos' : wr >= 40 ? 'warn' : 'neg');
  document.getElementById('kpiWinRateSub').textContent = `W/L: ${d.wins}/${d.losses}`;
  document.getElementById('kpiWinRateExcl').textContent = `incl. flat: ${fmtNum(wrGross, 1)}%`;

  const pf = d.profit_factor;
  const pfEl = document.getElementById('kpiPF');
  pfEl.textContent = fmtNum(pf, 2);
  pfEl.className = 'kpi-value ' + (pf >= 1.5 ? 'pos' : pf >= 1 ? 'warn' : 'neg');

  document.getElementById('kpiTrades').textContent = d.total_trades;
  document.getElementById('kpiTradesSub').textContent =
    `${d.wins} wins · ${d.losses} losses`;
  const flatCount = d.flat_trades || 0;
  const flatEl = document.getElementById('kpiFlatSub');
  flatEl.textContent = `flat: ${flatCount} (${fmtNum(flatCount / d.total_trades * 100, 0)}%)`;
  flatEl.style.color = flatCount > 100 ? 'var(--yellow)' : 'var(--text2)';

  const dd = d.drawdown_pct;
  const ddEl = document.getElementById('kpiDD');
  ddEl.textContent = fmtNum(dd, 2) + '%';
  ddEl.className = 'kpi-value ' + (dd <= 2 ? 'pos' : dd <= 5 ? 'warn' : 'neg');
  document.getElementById('kpiDdCard').className = 'kpi-card ' + (dd <= 2 ? 'positive' : dd <= 5 ? 'warning' : 'negative');

  const op = d.open_positions;
  document.getElementById('kpiPos').textContent = op;
  document.getElementById('kpiPosSub').textContent = op === 0 ? 'sin exposición' : `$${fmtNum(d.total_exposure_usd, 0)} exp.`;

  // Advanced metrics
  document.getElementById('mSharpe').textContent = fmtNum(d.sharpe_ratio, 3);
  document.getElementById('mSharpe').className = 'metric-value ' + (d.sharpe_ratio >= 1 ? 'pos' : d.sharpe_ratio >= 0 ? 'warn' : 'neg');
  document.getElementById('mAvgWin').textContent = '+' + fmt$(d.avg_win_usd);
  document.getElementById('mAvgLoss').textContent = '-' + fmt$(d.avg_loss_usd);
  document.getElementById('mBestTrade').textContent = '+' + fmt$(d.best_trade);
  document.getElementById('mWorstTrade').textContent = '-' + fmt$(Math.abs(d.worst_trade));
  document.getElementById('mMaxWinStreak').textContent = d.max_win_streak + ' trades';
  document.getElementById('mMaxLossStreak').textContent = d.max_loss_streak + ' trades';
  document.getElementById('mPF2').textContent = fmtNum(d.profit_factor, 2);
  document.getElementById('mPF2').className = 'metric-value ' + (pf >= 1.5 ? 'pos' : pf >= 1 ? 'warn' : 'neg');
  const wrExclEl = document.getElementById('mWRExcl');
  const wre = d.win_rate_excl_flat || 0;
  wrExclEl.textContent = fmtNum(wre, 1) + '%';
  wrExclEl.className = 'metric-value ' + (wre >= 50 ? 'pos' : wre >= 40 ? 'warn' : 'neg');
  document.getElementById('mFlatTrades').textContent =
    `${d.flat_trades || 0} (${d.emergency_zero_pnl || 0} EMRG)`;
  document.getElementById('mInitCap').textContent = fmt$(d.initial_capital);
  const retEl = document.getElementById('mReturn');
  retEl.textContent = fmtPct(d.return_pct, true);
  retEl.className = 'metric-value ' + (d.return_pct >= 0 ? 'pos' : 'neg');

  // Accounting discrepancy banner
  const gap = d.accounting_gap || 0;
  const realChange = d.real_capital_change || 0;
  const recPnl = d.total_pnl || 0;
  const emergencyZero = d.emergency_zero_pnl || 0;
  const banner = document.getElementById('discBanner');
  // Show banner only if gap > 5% of capital OR > $25 (whichever is larger)
  const capitalForThreshold = d.capital_usd || 500;
  const thresholdPct = capitalForThreshold * 0.05;
  const threshold = Math.max(thresholdPct, 25);
  if (Math.abs(gap) > threshold) {  // show if gap > 5% of capital or $25 abs
    banner.style.display = '';
    const rcEl = document.getElementById('discRealChange');
    rcEl.textContent = (realChange >= 0 ? '+' : '') + fmt$(realChange);
    rcEl.className = 'disc-value ' + (realChange >= 0 ? 'pos' : 'neg');
    const rpEl = document.getElementById('discRecordedPnl');
    rpEl.textContent = (recPnl >= 0 ? '+' : '') + fmt$(recPnl);
    rpEl.className = 'disc-value ' + (recPnl >= 0 ? 'pos' : 'neg');
    const gapEl = document.getElementById('discGap');
    gapEl.textContent = (gap >= 0 ? '+' : '') + fmt$(gap);
    gapEl.className = 'disc-value ' + (gap >= 0 ? 'pos' : 'neg');
    document.getElementById('discEmergency').textContent = emergencyZero + ' trades';
    document.getElementById('discCapNote').textContent = fmt$(d.capital_usd);
    const gapNoteEl = document.getElementById('discGapNote');
    if (gapNoteEl) {
      gapNoteEl.textContent = (gap >= 0 ? '+' : '') + fmt$(gap) + ' (' + fmtNum(Math.abs(gap) / capitalForThreshold * 100, 1) + '% del capital)';
    }
  } else {
    banner.style.display = 'none';
  }
}



async function loadEquity() {
  const r = await fetch('/api/equity');
  const d = await r.json();

  equityAllData = {
    labels: d.labels,
    trade_pnls: d.trade_pnls,
    equity: d.equity,
    winners: d.winners,
    losers: d.losers
  };
  updateEquityChart('all');

  // Pie chart
  const wins = d.total_wins || 0, losses = d.total_losses || 0;
  pieChart.data.datasets[0].data = [wins, losses];
  pieChart.update();

  // Symbol P&L bar
  const syms = Object.keys(d.symbol_pnl || {}).sort((a,b) => d.symbol_pnl[b] - d.symbol_pnl[a]);
  const symVals = syms.map(s => d.symbol_pnl[s]);
  symbolChart.data.labels = syms;
  symbolChart.data.datasets[0].data = symVals;
  symbolChart.data.datasets[0].backgroundColor = symVals.map(v => v >= 0 ? 'rgba(63,185,80,0.7)' : 'rgba(248,81,73,0.7)');
  symbolChart.data.datasets[0].borderColor = symVals.map(v => v >= 0 ? '#3fb950' : '#f85149');
  symbolChart.update();

  // Histogram
  const pnls = d.trade_pnls || [];
  if (pnls.length > 0) {
    const min = Math.min(...pnls), max = Math.max(...pnls);
    const bins = 20;
    const step = (max - min) / bins || 1;
    const counts = new Array(bins).fill(0);
    const binLabels = [];
    for (let i = 0; i < bins; i++) binLabels.push(fmtNum(min + i * step, 2));
    pnls.forEach(p => {
      const idx = Math.min(Math.floor((p - min) / step), bins - 1);
      counts[Math.max(0, idx)]++;
    });
    histChart.data.labels = binLabels;
    histChart.data.datasets[0].data = counts;
    histChart.data.datasets[0].backgroundColor = binLabels.map(l => parseFloat(l) >= 0 ? 'rgba(63,185,80,0.6)' : 'rgba(248,81,73,0.6)');
    histChart.update();
  }
}

async function loadPositions() {
  const r = await fetch('/api/positions');
  const d = await r.json();
  renderPositions(d);
}

// Función de actualización en tiempo real (cada 3 seg)
async function loadPositionsRealtime() {
  try {
    const r = await fetch('/api/positions');
    const d = await r.json();
    renderPositions(d);
    // Actualizar timestamp
    document.getElementById('lastUpdate').textContent =
      'Actualizado: ' + new Date().toLocaleTimeString('es-MX', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch(e) {
    // Silencioso en errores de realtime
  }
}

function renderPositions(d) {
  const wrap = document.getElementById('positionsTable');
  if (!d.positions || d.positions.length === 0) {
    wrap.innerHTML = '<div class="empty">⚡ Sin posiciones abiertas en este momento</div>';
    return;
  }
  let html = `<table><thead><tr>
    <th>Símbolo</th><th>Dirección</th><th>Entrada</th><th>Precio Act.</th>
    <th>P&L $</th><th>P&L %</th><th>Margin</th><th>Tamaño</th>
    <th>SL</th><th>TP</th><th>Tiempo</th><th>Estrategia</th>
  </tr></thead><tbody>`;
  for (const p of d.positions) {
    const pnlCls = p.pnl_usd >= 0 ? 'pos' : 'neg';
    // Animación de flash cuando cambia el precio
    const flashCls = p.price_changed ? 'flash-update' : '';
    html += `<tr class="${flashCls}">
      <td><strong>${p.symbol}</strong></td>
      <td><span class="badge badge-${p.direction}">${p.direction.toUpperCase()}</span></td>
      <td>${fmtPrice(p.entry_price)}</td>
      <td class="price-cell">${fmtPrice(p.current_price)}</td>
      <td class="${pnlCls}">${p.pnl_usd >= 0 ? '+' : ''}${fmt$(p.pnl_usd)}</td>
      <td class="${pnlCls}">${fmtPct(p.pnl_pct, true)}</td>
      <td>${fmt$(p.margin_usd)}</td>
      <td>${fmt$(p.size_usd)}</td>
      <td class="neg">${fmtPrice(p.sl_price)}</td>
      <td class="pos">${fmtPrice(p.tp_price)}</td>
      <td>${p.time_open || '—'}</td>
      <td>${p.strategy || '—'}</td>
    </tr>`;
  }
  html += '</tbody></table>';
  wrap.innerHTML = html;
}

async function loadTrades() {
  const r = await fetch('/api/trades?limit=500');
  const d = await r.json();
  allTrades = d.trades || [];
  populateFilters();
  filterTrades();
}

async function loadLog() {
  const r = await fetch('/api/log');
  const d = await r.json();
  const el = document.getElementById('logContainer');
  if (!d.lines || d.lines.length === 0) {
    el.innerHTML = '<div class="log-info">Sin log disponible</div>';
    return;
  }
  el.innerHTML = d.lines.map(line => {
    let cls = 'log-info';
    if (/✅|exitosamente|código: 0/i.test(line)) cls = 'log-ok';
    else if (/⚠️|error|fail|código: 1|exception/i.test(line)) cls = 'log-err';
    else if (/⚙️|🔄|ejecutando/i.test(line)) cls = 'log-warn';
    return `<div class="log-line ${cls}">${escHtml(line)}</div>`;
  }).join('');
  el.scrollTop = el.scrollHeight;
}

// ── Filtering / Pagination ─────────────────────────────────────────────────
function populateFilters() {
  const symbols = [...new Set(allTrades.map(t => t.symbol))].sort();
  const sel = document.getElementById('filterSymbol');
  const current = sel.value;
  sel.innerHTML = '<option value="">Todos los símbolos</option>';
  symbols.forEach(s => sel.innerHTML += `<option value="${s}" ${s === current ? 'selected' : ''}>${s}</option>`);
}

function filterTrades() {
  const sym = document.getElementById('filterSymbol').value;
  const res = document.getElementById('filterResult').value;
  const dir = document.getElementById('filterDir').value;
  filteredTrades = allTrades.filter(t => {
    if (sym && t.symbol !== sym) return false;
    if (dir && t.direction !== dir) return false;
    if (res) {
      if (res === 'win' && t.pnl_usd <= 0) return false;
      if (res === 'loss' && t.pnl_usd >= 0) return false;
      if (res === 'flat' && t.pnl_usd !== 0) return false;
      if (res === 'nonflat' && t.pnl_usd === 0) return false;
    }
    return true;
  });
  currentPage = 1;
  renderTradesPage();
}

function renderTradesPage() {
  const start = (currentPage - 1) * PAGE_SIZE;
  const page = filteredTrades.slice(start, start + PAGE_SIZE);
  const wrap = document.getElementById('tradesTable');

  if (page.length === 0) {
    wrap.innerHTML = '<div class="empty">Sin trades que mostrar</div>';
    document.getElementById('tradesPagination').innerHTML = '';
    return;
  }

  let html = `<table><thead><tr>
    <th>#</th><th>Fecha</th><th>Símbolo</th><th>Dirección</th>
    <th>Estrategia</th><th>Entrada</th><th>Salida</th>
    <th>P&L $</th><th>P&L %</th><th>Resultado</th><th>Razón cierre</th>
  </tr></thead><tbody>`;

  page.forEach((t, i) => {
    const pnlCls = t.pnl_usd > 0 ? 'pos' : t.pnl_usd < 0 ? 'neg' : '';
    const resultBadge = t.pnl_usd > 0 ? 'badge-win' : t.pnl_usd < 0 ? 'badge-loss' : 'badge-hold';
    const resultTxt = t.pnl_usd > 0 ? 'WIN' : t.pnl_usd < 0 ? 'LOSS' : 'FLAT';
    html += `<tr>
      <td class="text2">${start + i + 1}</td>
      <td>${t.open_time ? t.open_time.replace('T',' ').slice(0,16) : '—'}</td>
      <td><strong>${t.symbol}</strong></td>
      <td><span class="badge badge-${t.direction}">${t.direction.toUpperCase()}</span></td>
      <td>${t.strategy || '—'}</td>
      <td>${fmtPrice(t.entry_price)}</td>
      <td>${fmtPrice(t.close_price)}</td>
      <td class="${pnlCls}">${t.pnl_usd >= 0 ? '+' : ''}${fmt$(t.pnl_usd)}</td>
      <td class="${pnlCls}">${fmtPct(t.pnl_pct, true)}</td>
      <td><span class="badge ${resultBadge}">${resultTxt}</span></td>
      <td>${t.close_reason || '—'}</td>
    </tr>`;
  });
  html += '</tbody></table>';
  wrap.innerHTML = html;

  // Pagination
  const totalPages = Math.ceil(filteredTrades.length / PAGE_SIZE);
  let pg = `<span class="page-info">${filteredTrades.length} trades · Pág ${currentPage}/${totalPages}</span>`;
  if (currentPage > 1) pg += `<button class="page-btn" onclick="goPage(${currentPage-1})">← Anterior</button>`;
  // page numbers
  const from = Math.max(1, currentPage - 2), to = Math.min(totalPages, currentPage + 2);
  for (let p = from; p <= to; p++) {
    pg += `<button class="page-btn ${p===currentPage?'active':''}" onclick="goPage(${p})">${p}</button>`;
  }
  if (currentPage < totalPages) pg += `<button class="page-btn" onclick="goPage(${currentPage+1})">Siguiente →</button>`;
  document.getElementById('tradesPagination').innerHTML = pg;
}

function goPage(p) { currentPage = p; renderTradesPage(); }

// ── Charts Init ────────────────────────────────────────────────────────────
function initCharts() {
  Chart.defaults.color = '#8b949e';
  Chart.defaults.borderColor = '#30363d';

  // Equity
  const equityCtx = document.getElementById('equityChart').getContext('2d');
  equityChart = new Chart(equityCtx, {
    type: 'line',
    data: { labels: [], datasets: [{
      label: 'Equity',
      data: [], fill: true,
      borderColor: '#58a6ff', borderWidth: 2,
      backgroundColor: 'rgba(88,166,255,0.08)',
      tension: 0.3, pointRadius: 0, pointHoverRadius: 4
    },{
      label: 'Ganadores',
      data: [], type: 'scatter',
      backgroundColor: 'rgba(63,185,80,0.8)',
      pointStyle: 'triangle', pointRadius: 5, showLine: false
    },{
      label: 'Perdedores',
      data: [], type: 'scatter',
      backgroundColor: 'rgba(248,81,73,0.8)',
      pointStyle: 'rectRot', pointRadius: 4, showLine: false
    }]},
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { intersect: false, mode: 'index' },
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 10, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label: ctx => {
              if (ctx.datasetIndex === 0) return ' Equity: $' + ctx.parsed.y.toFixed(2);
              return ' P&L: $' + (ctx.raw.pnl || 0).toFixed(2);
            }
          }
        },
        zoom: {
          zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' },
          pan: { enabled: true, mode: 'x' }
        }
      },
      scales: {
        x: { grid: { color: 'rgba(48,54,61,0.5)' }, ticks: { maxTicksLimit: 8, font: { size: 10 } } },
        y: {
          grid: { color: 'rgba(48,54,61,0.5)' },
          ticks: { callback: v => '$' + v.toFixed(0), font: { size: 10 } }
        }
      }
    }
  });

  // Pie
  const pieCtx = document.getElementById('pieChart').getContext('2d');
  pieChart = new Chart(pieCtx, {
    type: 'doughnut',
    data: {
      labels: ['Ganadores', 'Perdedores'],
      datasets: [{ data: [0,0], backgroundColor: ['rgba(63,185,80,0.8)','rgba(248,81,73,0.8)'], borderWidth: 2, borderColor: '#161b22' }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: {
        legend: { position: 'right', labels: { font: { size: 11 }, boxWidth: 10 } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.parsed}` } }
      }
    }
  });

  // Symbol bar
  const symCtx = document.getElementById('symbolChart').getContext('2d');
  symbolChart = new Chart(symCtx, {
    type: 'bar',
    data: { labels: [], datasets: [{ data: [], borderWidth: 1, borderRadius: 3 }] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => ' P&L: $' + ctx.parsed.y.toFixed(2) } } },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 10 } } },
        y: { grid: { color: 'rgba(48,54,61,0.5)' }, ticks: { callback: v => '$' + v.toFixed(1), font: { size: 10 } } }
      }
    }
  });

  // Histogram
  const histCtx = document.getElementById('histChart').getContext('2d');
  histChart = new Chart(histCtx, {
    type: 'bar',
    data: { labels: [], datasets: [{ data: [], borderWidth: 1, borderRadius: 2 }] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => ` Trades: ${ctx.parsed.y}` } } },
      scales: {
        x: { grid: { display: false }, title: { display: true, text: 'P&L por Trade ($)', font: { size: 10 } }, ticks: { maxTicksLimit: 10, font: { size: 10 } } },
        y: { grid: { color: 'rgba(48,54,61,0.5)' }, title: { display: true, text: 'Frecuencia', font: { size: 10 } }, ticks: { font: { size: 10 } } }
      }
    }
  });
}

function updateEquityChart(range) {
  if (!equityAllData.equity) return;
  let eq = equityAllData.equity;
  let lb = equityAllData.labels;
  let tr = equityAllData.trade_pnls;
  let wi = equityAllData.winners;
  let lo = equityAllData.losers;

  if (range !== 'all' && typeof range === 'number') {
    const n = eq.length;
    const from = Math.max(0, n - range);
    eq = eq.slice(from);
    lb = lb.slice(from);
    tr = tr.slice(from);
    // filter scatter points within range
    wi = (wi || []).filter(p => p.x >= from).map(p => ({...p, x: p.x - from}));
    lo = (lo || []).filter(p => p.x >= from).map(p => ({...p, x: p.x - from}));
  }

  equityChart.data.labels = lb;
  equityChart.data.datasets[0].data = eq;
  equityChart.data.datasets[1].data = (wi||[]).map(p => ({ x: p.x, y: p.y, pnl: p.pnl }));
  equityChart.data.datasets[2].data = (lo||[]).map(p => ({ x: p.x, y: p.y, pnl: p.pnl }));
  equityChart.update();
}

function setEquityRange(range, btn) {
  document.querySelectorAll('.chart-toolbar .tb-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  updateEquityChart(range);
}

function resetZoom() { equityChart.resetZoom(); }

// ── Helpers ────────────────────────────────────────────────────────────────
function fmt$(v) { return '$' + (v || 0).toFixed(2); }
function fmtNum(v, d=2) { return (v || 0).toFixed(d); }
function fmtPct(v, sign=false) {
  const n = (v || 0).toFixed(2);
  return (sign && v > 0 ? '+' : '') + n + '%';
}
function fmtPrice(v) {
  if (!v || v === 0) return '—';
  if (v < 0.0001) return v.toExponential(4);
  if (v < 1) return '$' + v.toFixed(6);
  if (v < 10) return '$' + v.toFixed(4);
  if (v < 1000) return '$' + v.toFixed(2);
  return '$' + v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
</body>
</html>
"""

# ── API Routes ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/stats')
def api_stats():
    """
    Compute stats from POST-RESET trades only.
    Capital = free_cash + invested_margin + unrealized_pnl (true equity).
    Pre-reset corrupted data (959 EMERGENCY_CLOSEs etc.) is excluded.
    """
    from datetime import datetime as dt

    risk = load_json(DATA / "risk_report.json")
    port = load_portfolio()
    comp = load_json(DATA / "compound_state.json")

    # --- Determine reset baseline ---
    reset_log = load_json(DATA / "reset_log_20260326.json")
    has_reset = bool(reset_log and reset_log.get("reset_date"))
    reset_dt = None
    if has_reset:
        try:
            reset_dt = dt.fromisoformat(reset_log["reset_date"])
        except (ValueError, TypeError):
            has_reset = False

    reset_capital = safe_float(reset_log.get("capital_after", 500.0)) if has_reset else 500.0
    initial_capital = reset_capital if has_reset else safe_float(
        port.get("initial_capital", comp.get("initial_capital", 500)))

    # --- Load trades, filter to post-reset if applicable ---
    trades_raw = load_trade_history()
    closed_all = [t for t in trades_raw if t.get("status") in ("closed", None, "")]

    if has_reset and reset_dt:
        closed = []
        for t in closed_all:
            close_time = t.get("close_time", "")
            if not close_time:
                continue
            try:
                ct = dt.fromisoformat(close_time.replace("Z", "+00:00"))
                if ct >= reset_dt:
                    closed.append(t)
            except (ValueError, TypeError):
                continue
    else:
        closed = closed_all

    # Open positions: from portfolio.json (more reliable for current state)
    port_positions = port.get("positions", [])
    open_pos = [t for t in port_positions if t.get("status") == "open"]
    if not open_pos:
        open_pos = [t for t in trades_raw if t.get("status") == "open"]
    total_exp = sum(safe_float(t.get("size_usd", 0)) for t in open_pos)

    # --- Compute real metrics from closed trades ---
    win_trades  = [t for t in closed if safe_float(t.get("pnl_usd", 0)) > 0]
    loss_trades = [t for t in closed if safe_float(t.get("pnl_usd", 0)) < 0]
    flat_trades = [t for t in closed if safe_float(t.get("pnl_usd", 0)) == 0]
    wins   = len(win_trades)
    losses = len(loss_trades)
    flats  = len(flat_trades)
    total  = len(closed)

    # Win rate including flat trades — industry standard
    win_rate    = (wins / total * 100) if total > 0 else 0.0
    # Win rate excluding flat — shows signal quality without noise
    active_total = wins + losses
    win_rate_excl_flat = (wins / active_total * 100) if active_total > 0 else 0.0

    gross_win   = sum(safe_float(t.get("pnl_usd", 0)) for t in win_trades)
    gross_loss  = abs(sum(safe_float(t.get("pnl_usd", 0)) for t in loss_trades))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else 0.0
    avg_win_usd   = (gross_win  / wins)   if wins   > 0 else 0.0
    avg_loss_usd  = (gross_loss / losses) if losses > 0 else 0.0
    total_pnl     = gross_win - gross_loss

    # Accounting discrepancy: EMERGENCY_CLOSEs with pnl=0 (post-reset only)
    emergency_zero = [t for t in flat_trades if "EMERGENCY" in str(t.get("close_reason", ""))]
    emergency_zero_count = len(emergency_zero)

    # Streaks + best/worst
    max_win_streak = max_loss_streak = cur_win = cur_loss = 0
    best = worst = 0.0
    closed_sorted = sorted(closed, key=lambda t: t.get("open_time", ""))
    for t in closed_sorted:
        p = safe_float(t.get("pnl_usd", 0))
        best = max(best, p)
        worst = min(worst, p)
        if p > 0:
            cur_win += 1; cur_loss = 0
        elif p < 0:
            cur_loss += 1; cur_win = 0
        else:
            cur_win = cur_loss = 0
        max_win_streak  = max(max_win_streak, cur_win)
        max_loss_streak = max(max_loss_streak, cur_loss)

    # Capital: TRUE EQUITY = free_cash + invested_margin + unrealized_pnl
    # portfolio.json capital_usd is FREE CASH (margin deducted when positions open)
    free_cash = safe_float(port.get("capital_usd", initial_capital))
    invested = sum(safe_float(p.get("margin_usd", p.get("size_usd", 0))) for p in open_pos)
    unrealized = sum(safe_float(p.get("pnl_usd", 0)) for p in open_pos)
    capital_usd = free_cash + invested + unrealized

    return_pct = ((capital_usd - initial_capital) / initial_capital * 100) \
                 if initial_capital > 0 else 0.0

    # Drawdown: compute from post-reset closed trade equity curve
    cap  = initial_capital
    peak = initial_capital
    max_dd = 0.0
    for t in closed_sorted:
        cap += safe_float(t.get("pnl_usd", 0))
        if cap > peak:
            peak = cap
        dd = ((peak - cap) / peak * 100) if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # Sharpe: daily returns approximation
    pnls = [safe_float(t.get("pnl_usd", 0)) for t in closed_sorted if safe_float(t.get("pnl_usd", 0)) != 0]
    if len(pnls) >= 2:
        import statistics
        mean_r = statistics.mean(pnls)
        std_r  = statistics.stdev(pnls)
        sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    # Accounting discrepancy: real capital change vs recorded PnL (post-reset only)
    real_capital_change = capital_usd - initial_capital
    accounting_gap = real_capital_change - total_pnl

    return jsonify({
        "capital_usd":        round(capital_usd, 2),
        "initial_capital":    round(initial_capital, 2),
        "total_pnl":          round(total_pnl, 2),
        "return_pct":         round(return_pct, 2),
        "total_trades":       total,
        "wins":               wins,
        "losses":             losses,
        "flat_trades":        flats,
        "emergency_zero_pnl": emergency_zero_count,
        "win_rate":           round(win_rate, 2),
        "win_rate_excl_flat": round(win_rate_excl_flat, 2),
        "avg_win_usd":        round(avg_win_usd, 2),
        "avg_loss_usd":       round(avg_loss_usd, 2),
        "profit_factor":      round(profit_factor, 3),
        "sharpe_ratio":       round(safe_float(sharpe), 3),
        "drawdown_pct":       round(max_dd, 2),
        "open_positions":     len(open_pos),
        "total_exposure_usd": round(total_exp, 2),
        "best_trade":         round(best, 4),
        "worst_trade":        round(worst, 4),
        "max_win_streak":     max_win_streak,
        "max_loss_streak":    max_loss_streak,
        "portfolio_status":   risk.get("portfolio_status", "ACTIVE"),
        "mode":               port.get("mode", "paper"),
        "accounting_gap":     round(accounting_gap, 2),
        "real_capital_change":round(real_capital_change, 2),
    })


@app.route('/api/stats_post_reset')
def api_stats_post_reset():
    """Metrics computed only from trades AFTER the last capital reset.
    Used for visualization only — auto_learner always uses full history."""
    from datetime import datetime as dt

    # Load reset log
    reset_log_path = DATA / "reset_log_20260326.json"
    reset_log = load_json(reset_log_path)
    if not reset_log or not reset_log.get("reset_date"):
        return jsonify({"has_reset": False})

    reset_date_str = reset_log["reset_date"]
    try:
        reset_dt = dt.fromisoformat(reset_date_str)
    except (ValueError, TypeError):
        return jsonify({"has_reset": False})

    reset_capital = safe_float(reset_log.get("capital_after", 500.0))

    # Filter closed trades after reset
    trades_raw = load_trade_history()
    post_reset = []
    for t in trades_raw:
        if t.get("status") not in ("closed", None, ""):
            continue
        close_time = t.get("close_time", "")
        if not close_time:
            continue
        try:
            ct = dt.fromisoformat(close_time.replace("Z", "+00:00"))
            if ct >= reset_dt:
                post_reset.append(t)
        except (ValueError, TypeError):
            continue

    post_reset.sort(key=lambda t: t.get("close_time", ""))

    total = len(post_reset)
    wins = sum(1 for t in post_reset if safe_float(t.get("pnl_usd", 0)) > 0)
    losses = sum(1 for t in post_reset if safe_float(t.get("pnl_usd", 0)) < 0)
    win_rate = (wins / total * 100) if total > 0 else 0.0
    total_pnl = sum(safe_float(t.get("pnl_usd", 0)) for t in post_reset)

    # Current capital: total equity (free cash + margin + unrealized P&L)
    port = load_portfolio()
    free_cash = safe_float(port.get("capital_usd", reset_capital))
    open_pos = [p for p in port.get("positions", []) if p.get("status") == "open"]
    invested = sum(safe_float(p.get("margin_usd", p.get("size_usd", 0))) for p in open_pos)
    unrealized = sum(safe_float(p.get("pnl_usd", 0)) for p in open_pos)
    current_capital = free_cash + invested + unrealized

    # Drawdown since reset
    cap = reset_capital
    peak = reset_capital
    max_dd = 0.0
    for t in post_reset:
        cap += safe_float(t.get("pnl_usd", 0))
        if cap > peak:
            peak = cap
        dd = ((peak - cap) / peak * 100) if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    return_pct = ((current_capital - reset_capital) / reset_capital * 100) if reset_capital > 0 else 0.0

    return jsonify({
        "has_reset": True,
        "reset_date": reset_date_str,
        "reset_capital": round(reset_capital, 2),
        "post_reset_capital": round(current_capital, 2),
        "post_reset_pnl": round(total_pnl, 2),
        "post_reset_trades": total,
        "post_reset_wins": wins,
        "post_reset_losses": losses,
        "post_reset_win_rate": round(win_rate, 1),
        "post_reset_drawdown": round(max_dd, 2),
        "post_reset_return_pct": round(return_pct, 2),
    })


@app.route('/api/equity')
def api_equity():
    from datetime import datetime as dt

    port = load_portfolio()
    trades_raw = load_trade_history()
    closed_all = [t for t in trades_raw if t.get("status") in ("closed", None, "")]

    # Filter to post-reset trades if reset exists
    reset_log = load_json(DATA / "reset_log_20260326.json")
    has_reset = bool(reset_log and reset_log.get("reset_date"))
    reset_dt = None
    if has_reset:
        try:
            reset_dt = dt.fromisoformat(reset_log["reset_date"])
        except (ValueError, TypeError):
            has_reset = False

    if has_reset and reset_dt:
        closed = []
        for t in closed_all:
            close_time = t.get("close_time", "")
            if not close_time:
                continue
            try:
                ct = dt.fromisoformat(close_time.replace("Z", "+00:00"))
                if ct >= reset_dt:
                    closed.append(t)
            except (ValueError, TypeError):
                continue
    else:
        closed = closed_all

    closed.sort(key=lambda t: t.get("close_time", ""))

    # Build equity curve from reset baseline
    init_cap = safe_float(reset_log.get("capital_after", 500.0)) if has_reset else \
               safe_float(port.get("initial_capital", 500))
    capital = init_cap
    equity_full = [init_cap]
    labels_full = ["Start"]
    trade_pnls = []
    winners = []
    losers  = []
    symbol_pnl: dict = {}
    wins = losses = 0

    for i, t in enumerate(closed):
        p = safe_float(t.get("pnl_usd", 0))
        capital += p
        equity_full.append(round(capital, 4))
        ts = t.get("close_time", "")
        labels_full.append(ts[:16].replace("T", " ") if ts else str(i+1))
        trade_pnls.append(p)

        sym = t.get("symbol", "UNK")
        symbol_pnl[sym] = symbol_pnl.get(sym, 0) + p

        idx = i + 1  # offset by 1 (Start point is index 0)
        if p > 0:
            wins += 1
            winners.append({"x": idx, "y": round(capital, 2), "pnl": p})
        elif p < 0:
            losses += 1
            losers.append({"x": idx, "y": round(capital, 2), "pnl": p})

    # Downsample equity curve for chart performance (max 500 points)
    MAX_CHART_PTS = 500
    n = len(equity_full)
    if n > MAX_CHART_PTS:
        step = n / MAX_CHART_PTS
        indices = [0] + [int(i * step) for i in range(1, MAX_CHART_PTS)] + [n - 1]
        indices = sorted(set(indices))
        equity = [equity_full[i] for i in indices]
        labels = [labels_full[i] for i in indices]
        # Re-map winners/losers x to downsampled index
        idx_map = {orig: new for new, orig in enumerate(indices)}
        def nearest(orig_x):
            best = min(indices, key=lambda v: abs(v - orig_x))
            return idx_map[best]
        winners = [{"x": nearest(w["x"]), "y": w["y"], "pnl": w["pnl"]} for w in winners]
        losers  = [{"x": nearest(l["x"]), "y": l["y"], "pnl": l["pnl"]} for l in losers]
    else:
        equity = equity_full
        labels = labels_full

    return jsonify({
        "equity": equity,
        "labels": labels,
        "trade_pnls": trade_pnls,
        "winners": winners,
        "losers":  losers,
        "total_wins": wins,
        "total_losses": losses,
        "symbol_pnl": {k: round(v, 4) for k, v in symbol_pnl.items()},
    })


@app.route('/api/trades')
def api_trades():
    limit = int(request.args.get("limit", 50))
    trades_raw = load_trade_history()
    closed = [t for t in trades_raw if t.get("status") in ("closed", None, "")]
    # sort newest first
    closed.sort(key=lambda t: t.get("close_time", ""), reverse=True)
    result = []
    for t in closed[:limit]:
        result.append({
            "id":          t.get("id", ""),
            "symbol":      t.get("symbol", ""),
            "direction":   t.get("direction", "long"),
            "strategy":    t.get("strategy", ""),
            "entry_price": safe_float(t.get("entry_price", 0)),
            "close_price": safe_float(t.get("close_price", 0)),
            "pnl_usd":     round(safe_float(t.get("pnl_usd", 0)), 4),
            "pnl_pct":     round(safe_float(t.get("pnl_pct", 0)), 4),
            "open_time":   t.get("open_time", ""),
            "close_time":  t.get("close_time", ""),
            "close_reason":t.get("close_reason", ""),
            "margin_usd":  safe_float(t.get("margin_usd", 0)),
            "leverage":    t.get("leverage", 1),
            "confidence":  safe_float(t.get("confidence", 0)),
        })
    return jsonify({"trades": result, "total": len(closed)})


# Cache de precios para no spamear la API
_price_cache = {"prices": {}, "last_fetch": 0}
PRICE_CACHE_TTL = 2  # segundos


def _jupiter_fetch(mints: dict) -> dict:
    """Fetch live prices from Jupiter API v3 for real-time dashboard updates.
    v3 returns data directly at root level (not wrapped in 'data' key).
    """
    import requests as req
    try:
        mint_ids = ",".join(mints.values())
        r = req.get(f"https://lite-api.jup.ag/price/v3?ids={mint_ids}", timeout=5)
        if r.status_code == 200:
            raw = r.json()
            prices = {}
            for symbol, mint in mints.items():
                price_data = raw.get(mint, {})
                if price_data:
                    prices[symbol] = float(price_data.get("usdPrice", 0))
            return prices
    except Exception:
        pass
    return {}


def get_live_prices(symbols: list) -> dict:
    """Get live prices with 2-second cache."""
    import time as _time
    now = _time.time()
    
    if now - _price_cache["last_fetch"] < PRICE_CACHE_TTL and _price_cache["prices"]:
        return _price_cache["prices"]
    
    # Cargar mints de market_data
    market = load_json(DATA / "market_latest.json")
    tokens = market.get("tokens", {})
    mints = {s: tokens[s]["mint"] for s in symbols if s in tokens and "mint" in tokens[s]}
    
    if mints:
        live = _jupiter_fetch(mints)
        if live:
            _price_cache["prices"] = live
            _price_cache["last_fetch"] = now
            return live
    
    return {}


@app.route('/api/positions')
def api_positions():
    port = load_portfolio()
    port_positions = port.get("positions", [])
    # Check portfolio positions first for "open" status
    open_pos = [t for t in port_positions if t.get("status") == "open"]
    # Fallback: trade_history
    if not open_pos:
        trades_raw = load_trade_history()
        open_pos = [t for t in trades_raw if t.get("status") == "open"]

    # Fetch live prices for all open position symbols
    symbols = [t.get("symbol", "") for t in open_pos]
    live_prices = get_live_prices(symbols) if symbols else {}

    result = []
    for t in open_pos:
        ot = t.get("open_time", "")
        symbol = t.get("symbol", "")
        direction = t.get("direction", "long")
        entry_price = safe_float(t.get("entry_price", 0))
        
        # Use live price if available, fallback to stored price
        current_price = live_prices.get(symbol, safe_float(t.get("current_price", 0)))
        
        # Recalculate P&L with live price
        margin = safe_float(t.get("margin_usd", 0))
        leverage = safe_float(t.get("leverage", 3))
        size_usd = margin * leverage
        
        if entry_price > 0 and current_price > 0:
            if direction == "long":
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # short
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            pnl_usd = (pnl_pct / 100) * size_usd
        else:
            pnl_pct = safe_float(t.get("pnl_pct", 0))
            pnl_usd = safe_float(t.get("pnl_usd", 0))
        
        # Compute time open
        time_open_str = "—"
        if ot:
            try:
                opened = datetime.fromisoformat(ot.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                diff = now - opened
                h = int(diff.total_seconds() // 3600)
                m = int((diff.total_seconds() % 3600) // 60)
                time_open_str = f"{h}h {m}m"
            except Exception:
                pass
        result.append({
            "symbol":        symbol,
            "direction":     direction,
            "entry_price":   entry_price,
            "current_price": round(current_price, 8),
            "pnl_usd":       round(pnl_usd, 4),
            "pnl_pct":       round(pnl_pct, 4),
            "margin_usd":    margin,
            "size_usd":      round(size_usd, 2),
            "sl_price":      safe_float(t.get("sl_price", 0)),
            "tp_price":      safe_float(t.get("tp_price", 0)),
            "strategy":      t.get("strategy", ""),
            "time_open":     time_open_str,
        })

    return jsonify({"positions": result})


@app.route('/api/log')
def api_log():
    lines = []
    try:
        if WATCHDOG_LOG.exists():
            with open(WATCHDOG_LOG) as f:
                all_lines = f.readlines()
            lines = [l.rstrip() for l in all_lines[-20:] if l.strip()]
    except Exception as e:
        lines = [f"Error leyendo log: {e}"]
    return jsonify({"lines": lines})


if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
    print(f"🚀 Solana Cripto Trader Dashboard")
    print(f"   URL: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
