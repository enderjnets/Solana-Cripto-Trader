import logging
log = logging.getLogger("dashboard")
#!/usr/bin/env python3
"""
Solana Cripto Trader - Professional Trading Dashboard
Flask backend serving real-time data from bot data files.
Port: 8001
"""

import json
import os
import math
import threading
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, render_template_string, request, Response

app = Flask(__name__)

# ── Data paths ───────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
DATA = BASE / "agents" / "data"
WATCHDOG_LOG = Path("/home/enderj/.config/solana-jupiter-bot/modular.log")

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        log.error(f'JSON decode error in {path}: {e}')
        return {}
    except Exception as e:
        log.error(f'Error reading {path}: {e}')
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
        # Handle dict format: {"trades": [...], "last_updated": "..."}
        if isinstance(data, dict):
            return data.get("trades", [])
    except Exception:
        pass
    # Fallback: portfolio.json positions
    port = load_json(DATA / "portfolio.json")
    return port.get("positions", [])

# ── Agent Notes (Chat) ────────────────────────────────────────────────────────
AGENT_NOTES_FILE = DATA / "agent_notes.json"

def load_notes():
    try:
        with open(AGENT_NOTES_FILE) as f:
            return json.load(f)
    except Exception:
        return {"messages": [], "last_updated": None}

def save_notes(data):
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(AGENT_NOTES_FILE, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def safe_float(v, default=0.0):
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default

def safe_parse_dt(s):
    """Parse ISO datetime string to naive datetime (strips timezone).
    Handles both timezone-aware and naive strings consistently."""
    from datetime import datetime as _dt
    if not s:
        return None
    try:
        d = _dt.fromisoformat(s.replace("Z", "+00:00"))
        if d.tzinfo is not None:
            d = d.replace(tzinfo=None)
        return d
    except (ValueError, TypeError):
        return None

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
  .btn-reset {
    background: var(--bg3); border: 1px solid var(--orange);
    color: var(--orange); padding: 6px 14px; border-radius: 6px;
    cursor: pointer; font-size: 12px; transition: all .2s; margin-left: 8px;
  }
  .btn-reset:hover { background: var(--orange); color: var(--bg); }
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
  /* ── Agent Chat ── */
  .chat-section{margin-top:20px}
  .chat-card{background:var(--bg2);border:1px solid var(--border);border-radius:10px;overflow:hidden;border-left:3px solid var(--purple)}
  .chat-header{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:var(--bg3);border-bottom:1px solid var(--border);font-size:12px;color:var(--text2)}
  .chat-badge{background:rgba(63,185,80,.15);color:var(--green);padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600}
  .chat-messages{max-height:400px;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px}
  .cmsg{display:flex;flex-direction:column;gap:2px}
  .cmsg-agent{align-items:flex-start}
  .cmsg-user{align-items:flex-end}
  .cmsg-meta{font-size:10px;margin-bottom:2px}
  .cmsg-agent .cmsg-meta{color:var(--purple)}
  .cmsg-user .cmsg-meta{color:var(--green)}
  .cmsg-bubble{padding:8px 12px;font-size:12px;line-height:1.5;max-width:85%;border-radius:12px;white-space:pre-wrap;word-wrap:break-word}
  .cmsg-agent .cmsg-bubble{background:#1f1f3d;border:1px solid #3d3d6e;border-radius:12px 12px 12px 2px}
  .cmsg-user .cmsg-bubble{background:#1a3d1a;border:1px solid #2d6e2d;border-radius:12px 12px 2px 12px}
  .chat-input-row{display:flex;gap:8px;padding:10px 12px;border-top:1px solid var(--border);background:var(--bg3)}
  .chat-input-row input{flex:1;background:var(--bg);border:1px solid var(--border);color:var(--text);padding:8px 12px;border-radius:8px;font-size:12px}
  .chat-input-row input:focus{border-color:var(--purple);outline:none}
  .chat-input-row button{background:var(--purple);color:#fff;border:none;padding:8px 16px;border-radius:8px;cursor:pointer;font-size:12px;font-weight:600}
  .chat-input-row button:hover{opacity:.85}
  .chat-input-row button:disabled{opacity:.4;cursor:not-allowed}
  .chat-hint{font-size:10px;color:var(--text2);padding:4px 12px 8px;text-align:center}
  .typing-wrap{display:none;padding:4px 14px}
  .typing-wrap.active{display:flex;align-items:flex-start;flex-direction:column}
  .typing-label{color:var(--purple);font-size:10px;margin-bottom:3px}
  .typing-row{background:#1f1f3d;border:1px solid #3d3d6e;border-radius:12px 12px 12px 2px;padding:8px 14px;display:flex;align-items:center;gap:8px}
  .typing-dots{display:flex;gap:4px}
  .typing-dots .dot{width:7px;height:7px;border-radius:50%;background:var(--purple);animation:dotBounce 1.4s infinite ease-in-out both}
  .typing-dots .dot:nth-child(2){animation-delay:.2s}
  .typing-dots .dot:nth-child(3){animation-delay:.4s}
  @keyframes dotBounce{0%,80%,100%{transform:scale(.4);opacity:.4}40%{transform:scale(1);opacity:1}}
  .typing-text{color:var(--text2);font-size:11px;animation:tPulse 2s infinite}
  @keyframes tPulse{0%,100%{opacity:.5}50%{opacity:1}}
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
    <button class="btn-reset" onclick="resetBot()">🔄 Reset</button>
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

  <!-- RESET HISTORY -->
  <section>
    <div class="section-title">📊 Historial de Resets</div>
    <div class="card">
      <div class="table-wrap" id="resetHistoryTable">
        <div class="empty">Sin resets registrados</div>
      </div>
    </div>
  </section>

  <!-- AGENT CHAT -->
  <section class="chat-section">
    <div class="section-title">💬 Agente de Trading — Chat en Vivo</div>
    <div class="chat-card">
      <div class="chat-header">
        <span>🤖 <strong>Solana Trading Agent</strong></span>
        <span class="chat-badge" id="chatBadge">0 notas</span>
      </div>
      <div class="chat-messages" id="chatMessages">
        <div class="empty">Conectando con el agente...</div>
      </div>
      <div class="typing-wrap" id="chatTyping">
        <div class="typing-label">🤖 Agente</div>
        <div class="typing-row">
          <div class="typing-dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
          <span class="typing-text" id="chatTypingText">Pensando...</span>
        </div>
      </div>
      <div class="chat-input-row">
        <input type="text" id="chatInput" placeholder="Escribe una nota o pregunta al agente..." maxlength="500"
               onkeydown="if(event.key==='Enter')sendChatNote()">
        <button onclick="sendChatNote()" id="chatSendBtn">Enviar ▸</button>
      </div>
      <div class="chat-hint">🚀 El agente responde automáticamente. Presiona Enter para enviar.</div>
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
      loadLog(),
      loadResetHistory()
    ]);
    document.getElementById('lastUpdate').textContent =
      'Actualizado: ' + new Date().toLocaleTimeString('es-MX', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch(e) {
    console.error('Refresh error:', e);
  }
}

async function resetBot() {
  const capital = prompt('¿Capital inicial para resetear? (USD)', '500');
  if (!capital) return;
  
  const capitalNum = parseFloat(capital);
  if (isNaN(capitalNum) || capitalNum < 100) {
    alert('Capital inválido (mínimo $100)');
    return;
  }
  
  if (!confirm(`⚠️ ¿Resetear el bot a $${capitalNum}?\n\nEsto borrará TODO el historial de trades.`)) {
    return;
  }
  
  try {
    const r = await fetch('/api/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ capital: capitalNum })
    });
    const d = await r.json();
    
    if (d.success) {
      alert(`✅ Bot reseteado a $${capitalNum}\n\nLa página se recargará.`);
      location.reload();
    } else {
      alert('❌ Error: ' + (d.error || 'Unknown'));
    }
  } catch(e) {
    alert('❌ Error de conexión: ' + e.message);
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

async function loadResetHistory() {
  const r = await fetch('/api/reset-history');
  const d = await r.json();
  const el = document.getElementById('resetHistoryTable');
  
  if (!d.history || d.history.length === 0) {
    el.innerHTML = '<div class="empty">Sin resets registrados — el historial aparecerá después del primer reset</div>';
    return;
  }
  
  // Ordenar por fecha (más reciente primero)
  const history = d.history.sort((a, b) => new Date(b.reset_date) - new Date(a.reset_date));
  
  let html = `<table>
    <thead>
      <tr>
        <th>#</th>
        <th>Fecha Reset</th>
        <th>Capital Inicial</th>
        <th>Capital Final</th>
        <th>Retorno</th>
        <th>Trades</th>
        <th>W/L/F</th>
        <th>Win Rate</th>
        <th>Mejor</th>
        <th>Peor</th>
      </tr>
    </thead>
    <tbody>`;
  
  history.forEach((h, i) => {
    const num = history.length - i;
    const date = new Date(h.reset_date).toLocaleDateString('es-MX', { day: '2-digit', month: 'short', year: '2-digit', hour: '2-digit', minute: '2-digit' });
    const ret = h.return_pct || 0;
    const returnCls = ret >= 0 ? 'positive' : 'negative';
    const returnSign = ret >= 0 ? '+' : '';
    const wr = h.win_rate || 0;
    const best = h.best_trade_usd || 0;
    const worst = h.worst_trade_usd || 0;
    const flat = h.flat || 0;

    html += `<tr>
      <td><strong>${num}</strong></td>
      <td>${date}</td>
      <td>$${(h.initial_capital||0).toFixed(2)}</td>
      <td>$${(h.final_capital||0).toFixed(2)}</td>
      <td class="${returnCls}">${returnSign}${ret.toFixed(2)}%</td>
      <td>${h.total_trades||0}</td>
      <td><span class="positive">${h.wins||0}</span>/<span class="negative">${h.losses||0}</span>/${flat}</td>
      <td>${wr.toFixed(1)}%</td>
      <td class="positive">$${best.toFixed(2)}</td>
      <td class="negative">$${worst.toFixed(2)}</td>
    </tr>`;
  });
  
  html += '</tbody></table>';
  
  // Agregar resumen de tendencia
  if (history.length >= 2) {
    const avgReturn = history.reduce((sum, h) => sum + (h.return_pct || 0), 0) / history.length;
    const avgWinRate = history.reduce((sum, h) => sum + (h.win_rate || 0), 0) / history.length;
    const improving = (history[0].return_pct || 0) > (history[1].return_pct || 0);
    const trend = improving ? '📈 Mejorando' : '📉 Empeorando';
    const trendCls = improving ? 'positive' : 'negative';
    
    html += `<div style="margin-top: 12px; padding: 10px; background: var(--bg3); border-radius: 6px; font-size: 12px;">
      <strong>Tendencia:</strong> <span class="${trendCls}">${trend}</span> |
      <strong>Retorno Promedio:</strong> ${avgReturn >= 0 ? '+' : ''}${avgReturn.toFixed(2)}% |
      <strong>Win Rate Promedio:</strong> ${avgWinRate.toFixed(1)}%
    </div>`;
  }
  
  el.innerHTML = html;
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

// ── Agent Chat (SSE real-time) ────────────────────────────────────────────
let chatSSE = null;
let chatLastCount = 0;
let chatTypingInterval = null;

function renderChat(notes){
  const msgs = notes.messages || [];
  const count = msgs.length;
  document.getElementById('chatBadge').textContent = count + ' ' + (count===1?'nota':'notas');
  if(count===0){
    document.getElementById('chatMessages').innerHTML='<div class="empty">Sin notas aún. El agente dejará actualizaciones aquí.</div>';
    return;
  }
  let h='';
  for(const m of msgs){
    const cls = m.sender==='agent' ? 'cmsg-agent' : 'cmsg-user';
    const label = m.sender==='agent' ? '🤖 Agente' : '👤 Ender';
    const time = new Date(m.ts).toLocaleString('es-MX',{timeZone:'America/Denver',month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
    h += '<div class="cmsg '+cls+'"><div class="cmsg-meta">'+label+' · '+time+'</div><div class="cmsg-bubble">'+escHtml(m.text)+'</div></div>';
  }
  document.getElementById('chatMessages').innerHTML = h;
  const box = document.getElementById('chatMessages');
  box.scrollTop = box.scrollHeight;
  // New agent message arrived → hide typing
  if(count > chatLastCount && chatLastCount > 0){
    const latest = msgs[msgs.length-1];
    if(latest && latest.sender==='agent'){
      showChatTyping(false);
      if(window._chatTimeout){clearTimeout(window._chatTimeout);window._chatTimeout=null;}
    }
  }
  chatLastCount = count;
}

function showChatTyping(show){
  const el = document.getElementById('chatTyping');
  if(show){
    el.className = 'typing-wrap active';
    const box = document.getElementById('chatMessages');
    box.scrollTop = box.scrollHeight;
    if(!chatTypingInterval){
      const statuses=['Pensando...','Analizando posiciones...','Revisando mercado...','Preparando respuesta...','Escribiendo...'];
      let idx=0;
      chatTypingInterval = setInterval(()=>{
        idx=(idx+1)%statuses.length;
        document.getElementById('chatTypingText').textContent=statuses[idx];
      },3000);
    }
  } else {
    el.className = 'typing-wrap';
    if(chatTypingInterval){clearInterval(chatTypingInterval);chatTypingInterval=null;}
  }
}

async function sendChatNote(){
  const inp = document.getElementById('chatInput');
  const text = inp.value.trim();
  if(!text) return;
  inp.value = '';
  inp.disabled = true;
  document.getElementById('chatSendBtn').disabled = true;
  showChatTyping(true);
  // Typing se oculta cuando llega mensaje del agente via SSE (no polling)
  // Safety timeout: ocultar despues de 60s si no llega respuesta
  if(window._chatTimeout) clearTimeout(window._chatTimeout);
  window._chatTimeout = setTimeout(() => { showChatTyping(false); }, 60000);
  try{
    await fetch('/api/agent-notes',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:text,sender:'user'})
    });
  }catch(e){console.error('Chat send error:',e);showChatTyping(false);}
  inp.disabled = false;
  document.getElementById('chatSendBtn').disabled = false;
  inp.focus();
}

let chatPollInterval = null;
async function pollChatStatus(){
  if(chatPollInterval) return; // ya está corriendo
  chatPollInterval = setInterval(async () => {
    try {
      const r = await fetch('/api/chat-status');
      const d = await r.json();
      if (!d.thinking) {
        // Don't hide typing here - wait for actual agent message via SSE
        // Only stop polling
        clearInterval(chatPollInterval);
        chatPollInterval = null;
        // Give SSE 2 seconds to deliver the message before hiding
        setTimeout(() => {
          const el = document.getElementById('chatTyping');
          if(el.className.includes('active')) showChatTyping(false);
        }, 2000);
      }
    } catch(e) {
      clearInterval(chatPollInterval);
      chatPollInterval = null;
    }
  }, 2000);
}

function connectChatSSE(){
  if(chatSSE) chatSSE.close();
  chatSSE = new EventSource('/api/agent-notes/stream');
  chatSSE.addEventListener('init', function(e){
    try{renderChat(JSON.parse(e.data).notes);}catch(x){}
  });
  chatSSE.addEventListener('update', function(e){
    try{renderChat(JSON.parse(e.data).notes);}catch(x){}
  });
  chatSSE.onmessage = function(e){
    try{const d=JSON.parse(e.data);if(d.notes)renderChat(d.notes);}catch(x){}
  };
  chatSSE.onerror = function(){
    chatSSE.close();
    setTimeout(connectChatSSE, 3000);
  };
}

// Load chat on page load + connect SSE
(async function(){
  try{
    const d = await(await fetch('/api/agent-notes')).json();
    renderChat(d);
  }catch(e){}
  connectChatSSE();
})();
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
            reset_dt = safe_parse_dt(reset_log["reset_date"])
        except (ValueError, TypeError):
            has_reset = False

    reset_capital = safe_float(reset_log.get("capital_after", 500.0)) if has_reset else 500.0
    # Read initial_capital directly from portfolio (set correctly by reset endpoint)
    initial_capital = safe_float(port.get("initial_capital", 500.0))

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
                ct = safe_parse_dt(close_time)
                if ct is not None and reset_dt is not None and ct >= reset_dt:
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

    # Capital: paper_capital from master_state IS the equity
    # The executor DOES deduct margin_usd from capital when opening positions.
    # So capital_usd = free capital only. True equity = capital + locked margin + unrealized PnL.
    capital_usd = safe_float(port.get("capital_usd", initial_capital))
    unrealized = sum(safe_float(p.get("pnl_usd", 0)) for p in open_pos)
    margin_locked = sum(safe_float(p.get("margin_usd", 0)) for p in open_pos)
    invested = sum(safe_float(p.get("margin_usd", p.get("size_usd", 0))) for p in open_pos)
    # True equity includes free capital + margin locked in positions + unrealized PnL
    equity = capital_usd + margin_locked + unrealized

    return_pct = ((equity - initial_capital) / initial_capital * 100) \
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

    # Accounting discrepancy: use equity (not free capital) vs recorded PnL
    # Suppress warning if gap ≈ initial_capital - equity_with_no_trades (manual reset scenario)
    real_capital_change = equity - initial_capital
    accounting_gap = real_capital_change - total_pnl
    # If no closed trades and gap ≈ initial_capital, it's a manual reset — suppress
    if abs(total_pnl) < 0.01 and abs(accounting_gap - initial_capital) < 1.0:
        accounting_gap = 0
        real_capital_change = 0

    return jsonify({
        "capital_usd":        round(equity, 2),
        "initial_capital":    round(initial_capital, 2),
        "total_pnl":          round(total_pnl + unrealized, 2),
        "realized_pnl":       round(total_pnl, 2),
        "unrealized_pnl":     round(unrealized, 2),
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

    # Load reset log — try latest from reset_history first, then fallback
    reset_log = {}
    reset_history = load_json(DATA / "reset_history.json")
    if isinstance(reset_history, list) and reset_history:
        reset_log = reset_history[-1]  # latest reset
    if not reset_log or not reset_log.get("reset_date"):
        reset_log = load_json(DATA / "reset_log_20260326.json")
    if not reset_log or not reset_log.get("reset_date"):
        return jsonify({"has_reset": False})

    reset_date_str = reset_log["reset_date"]
    try:
        reset_dt = safe_parse_dt(reset_date_str)
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
            ct = safe_parse_dt(close_time)
            if ct is not None and reset_dt is not None and ct >= reset_dt:
                post_reset.append(t)
        except (ValueError, TypeError):
            continue

    post_reset.sort(key=lambda t: t.get("close_time", ""))

    total = len(post_reset)
    wins = sum(1 for t in post_reset if safe_float(t.get("pnl_usd", 0)) > 0)
    losses = sum(1 for t in post_reset if safe_float(t.get("pnl_usd", 0)) < 0)
    win_rate = (wins / total * 100) if total > 0 else 0.0
    total_pnl = sum(safe_float(t.get("pnl_usd", 0)) for t in post_reset)

    # Current capital: paper_capital from master_state IS the equity
    # The orchestrator does NOT deduct margin when opening positions
    # So capital_usd already represents total equity
    port = load_portfolio()
    current_capital = safe_float(port.get("capital_usd", reset_capital))
    open_pos = [p for p in port.get("positions", []) if p.get("status") == "open"]

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
            reset_dt = safe_parse_dt(reset_log["reset_date"])
        except (ValueError, TypeError):
            has_reset = False

    if has_reset and reset_dt:
        closed = []
        for t in closed_all:
            close_time = t.get("close_time", "")
            if not close_time:
                continue
            try:
                ct = safe_parse_dt(close_time)
                if ct is not None and reset_dt is not None and ct >= reset_dt:
                    closed.append(t)
            except (ValueError, TypeError):
                continue
    else:
        closed = closed_all

    closed.sort(key=lambda t: t.get("close_time", ""))

    # Build equity curve from reset baseline
    init_cap = safe_float(reset_log.get("capital_after", 500.0)) if has_reset else \
               safe_float(port.get("initial_capital", 1000))
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


@app.route('/api/reset-history')
def api_reset_history():
    """Get reset history for tracking progress."""
    reset_history_file = DATA / "reset_history.json"
    if reset_history_file.exists():
        try:
            with open(reset_history_file) as f:
                history = json.load(f)
            return jsonify({"history": history})
        except:
            pass
    return jsonify({"history": []})


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


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset all bot state to initial values."""
    try:
        data = request.get_json() or {}
        capital = float(data.get("capital", 500.0))
        
        now = datetime.now(timezone.utc).isoformat()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # === GUARDAR SNAPSHOT ANTES DEL RESET ===
        old_portfolio = load_portfolio()
        old_history = load_trade_history()
        
        # Calcular estadísticas del período que termina
        total_trades = len([t for t in old_history if t.get("status") == "closed"])
        wins = len([t for t in old_history if t.get("status") == "closed" and safe_float(t.get("pnl_usd", 0)) > 0])
        losses = len([t for t in old_history if t.get("status") == "closed" and safe_float(t.get("pnl_usd", 0)) < 0])
        flat = total_trades - wins - losses
        
        total_pnl = sum(safe_float(t.get("pnl_usd", 0)) for t in old_history if t.get("status") == "closed")
        
        old_capital = safe_float(old_portfolio.get("capital_usd", 500))
        old_initial = safe_float(old_portfolio.get("initial_capital", 1000))
        return_pct = ((old_capital - old_initial) / old_initial * 100) if old_initial > 0 else 0
        
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        # Mejor y peor trade
        pnls = [safe_float(t.get("pnl_usd", 0)) for t in old_history if t.get("status") == "closed"]
        best_trade = max(pnls) if pnls else 0
        worst_trade = min(pnls) if pnls else 0
        
        # Crear entrada de historial
        reset_entry = {
            "reset_date": now,
            "period_start": old_portfolio.get("created_at", "unknown"),
            "period_end": now,
            "initial_capital": old_initial,
            "final_capital": old_capital,
            "return_pct": round(return_pct, 2),
            "total_pnl_usd": round(total_pnl, 2),
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "flat": flat,
            "win_rate": round(win_rate, 2),
            "best_trade_usd": round(best_trade, 2),
            "worst_trade_usd": round(worst_trade, 2),
            "new_capital": capital,
            "reason": data.get("reason", "Manual reset")
        }
        
        # Cargar historial existente y agregar
        reset_history_file = DATA / "reset_history.json"
        reset_history = []
        if reset_history_file.exists():
            try:
                with open(reset_history_file) as f:
                    reset_history = json.load(f)
            except:
                reset_history = []
        
        reset_history.append(reset_entry)
        
        with open(reset_history_file, "w") as f:
            json.dump(reset_history, f, indent=2)
        
        # === CONTINUAR CON EL RESET NORMAL ===
        
        # 1. Portfolio
        portfolio = {
            "capital_usd": capital,
            "initial_capital": capital,
            "positions": [],
            "status": "ACTIVE",
            "mode": "paper_drift",
            "created_at": now,
            "last_updated": now,
            "total_trades": 0,
            "wins": 0,
            "losses": 0
        }
        with open(DATA / "portfolio.json", "w") as f:
            json.dump(portfolio, f, indent=2)
        
        # 2. Trade History
        with open(DATA / "trade_history.json", "w") as f:
            json.dump([], f)
        
        # 3. Daily Target State
        daily_state = {
            "date": today,
            "starting_capital": capital,
            "target_reached": False,
            "closed_at": None,
            "enabled": True,
            "current_pnl_pct": 0.0,
            "target_pct": 0.05
        }
        with open(DATA / "daily_target_state.json", "w") as f:
            json.dump(daily_state, f, indent=2)
        
        # 4. Daily Report
        daily_report = {
            "timestamp": now,
            "capital_usd": capital,
            "equity": capital,
            "pnl_total_usd": 0.0,
            "pnl_total_pct": 0.0,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "drawdown_pct": 0.0,
            "positions_open": 0
        }
        with open(DATA / "daily_report.json", "w") as f:
            json.dump(daily_report, f, indent=2)
        
        # 5. Compound State
        compound = {
            "enabled": True,
            "last_compound": None,
            "compound_count": 0,
            "total_compounded": 0.0
        }
        with open(DATA / "compound_state.json", "w") as f:
            json.dump(compound, f, indent=2)
        
        # 6. Auto Learner State — PRESERVAR conocimiento pero RECONCILIAR indices
        learner_file = DATA / "auto_learner_state.json"
        if learner_file.exists():
            with open(learner_file) as f:
                existing_learner = json.load(f)
            existing_learner["last_updated"] = now
            existing_learner["notes"] = f"Reset capital ${capital} - conocimiento preservado"
            # FIX: Reconciliar indices de trades para evitar new_trades negativo
            existing_learner["last_trade_count"] = 0
            existing_learner["total_trades_learned"] = 0
            with open(learner_file, "w") as f:
                json.dump(existing_learner, f, indent=2)
        else:
            learner = {
                "params": {
                    "sl_pct": 0.025,
                    "tp_pct": 0.05,
                    "leverage_tier": 2,
                    "risk_per_trade": 0.02,
                    "max_positions": 5
                },
                "total_trades_learned": 0,
                "last_adaptation": None,
                "adaptation_count": 0,
                "last_updated": now,
                "notes": f"Fresh start - capital ${capital}"
            }
            with open(learner_file, "w") as f:
                json.dump(learner, f, indent=2)
        
        # 7. Alerts State
        alerts = {
            "last_daily_report": "",
            "alerted_trades": [],
            "reset_at": now
        }
        with open(DATA / "alerts_state.json", "w") as f:
            json.dump(alerts, f, indent=2)
        
        return jsonify({
            "success": True,
            "capital": capital,
            "timestamp": now,
            "message": f"Bot reseteado a ${capital}"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Agent Notes API ──────────────────────────────────────────────────────────

@app.route('/api/agent-notes', methods=['GET', 'POST'])
def api_agent_notes():
    if request.method == 'POST':
        data = request.get_json(force=True)
        msg_text = data.get("message", "").strip()
        sender = data.get("sender", "user")
        if not msg_text:
            return jsonify({"error": "empty message"}), 400
        notes = load_notes()
        notes.setdefault("messages", []).append({
            "sender": sender,
            "text": msg_text,
            "ts": datetime.now().isoformat()
        })
        save_notes(notes)
        return jsonify({"ok": True})
    return jsonify(load_notes())

@app.route('/api/agent-notes/last')
def api_agent_last_note():
    notes = load_notes()
    agent_msgs = [m for m in notes.get("messages", []) if m.get("sender") == "agent"]
    if agent_msgs:
        return jsonify(agent_msgs[-1])
    return jsonify({"text": None})

@app.route('/api/agent-notes/stream')
def api_agent_notes_stream():
    def generate():
        notes = load_notes()
        yield f"event: init\ndata: {json.dumps({'type': 'init', 'notes': notes}, ensure_ascii=False)}\n\n"
        last_mtime = AGENT_NOTES_FILE.stat().st_mtime if AGENT_NOTES_FILE.exists() else 0
        last_count = len(notes.get("messages", []))
        try:
            while True:
                _time.sleep(1)
                cur_mtime = AGENT_NOTES_FILE.stat().st_mtime if AGENT_NOTES_FILE.exists() else 0
                if cur_mtime != last_mtime:
                    last_mtime = cur_mtime
                    notes = load_notes()
                    msgs = notes.get("messages", [])
                    if len(msgs) != last_count:
                        last_count = len(msgs)
                        yield f"event: update\ndata: {json.dumps({'type': 'update', 'notes': notes}, ensure_ascii=False)}\n\n"
                yield ": keepalive\n\n"
        except GeneratorExit:
            pass
    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no', 'Connection': 'keep-alive'})


CHAT_STATE_FILE = DATA / "chat_agent_state.json"

@app.route('/api/chat-status')
def api_chat_status():
    """Retorna estado del agente de chat (pensando sí/no)."""
    try:
        if CHAT_STATE_FILE.exists():
            state = json.loads(CHAT_STATE_FILE.read_text())
            return jsonify({"thinking": state.get("thinking", False)})
    except:
        pass
    return jsonify({"thinking": False})


if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
    print(f"🚀 Solana Cripto Trader Dashboard")
    print(f"   URL: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
