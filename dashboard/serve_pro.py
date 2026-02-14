#!/usr/bin/env python3
"""Professional Dashboard - Port 8502"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

STATE_FILE = Path('unified_brain_state.json')
DB_FILE = Path('db/unified_trading.db')

HTML = '''<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><meta http-equiv="refresh" content="15">
<title>Eko Trading System - Professional Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',-apple-system,sans-serif;background:#0a0a0a;color:#fff;min-height:100vh;padding:20px}
.header{background:linear-gradient(135deg,#1a1a2e,#16213e);padding:25px;border-radius:15px;margin-bottom:20px;display:flex;justify-content:space-between;align-items:center;border:1px solid #2a2a4a}
.logo{font-size:24px;font-weight:bold;background:linear-gradient(90deg,#00d4ff,#7b2cbf);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.status-badge{background:#00cc96;color:#000;padding:5px 15px;border-radius:20px;font-size:12px;font-weight:bold}
.metrics-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:15px;margin-bottom:20px}
.metric-card{background:linear-gradient(135deg,#1e1e1e,#2d2d2d);border-radius:12px;padding:20px;text-align:center;border:1px solid #3a3a3a}
.metric-value{font-size:32px;font-weight:bold;margin-bottom:5px}
.metric-label{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1px}
.positive{color:#00cc96}.negative{color:#ef553b}.neutral{color:#fff}
.progress-section{background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:15px;padding:25px;margin-bottom:20px;border:1px solid #2a2a4a}
.progress-header{display:flex;justify-content:space-between;margin-bottom:15px}
.progress-bar{height:12px;background:rgba(255,255,255,0.1);border-radius:6px;overflow:hidden}
.progress-fill{height:100%;background:linear-gradient(90deg,#00cc96,#00d4ff);border-radius:6px;transition:width 0.5s}
.section{background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:15px;padding:20px;margin-bottom:20px;border:1px solid #2a2a4a}
.section h3{margin-bottom:15px;padding-bottom:10px;border-bottom:1px solid #2a2a4a;font-size:16px}
.modules-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
.module{background:rgba(255,255,255,0.03);padding:12px;border-radius:8px;text-align:center;font-size:12px;border-left:3px solid #00cc96}
table{width:100%;border-collapse:collapse}
th{text-align:left;padding:10px;color:#888;font-size:11px;text-transform:uppercase;border-bottom:1px solid #2a2a4a}
td{padding:12px 10px;border-bottom:1px solid rgba(255,255,255,0.05)}
tr:hover{background:rgba(255,255,255,0.02)}
.direction{padding:3px 10px;border-radius:4px;font-size:11px;font-weight:bold}
.direction.BUY{background:#00cc96;color:#000}
.direction.SELL{background:#ef553b;color:#fff}
.charts-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;margin-bottom:20px}
.chart-box{background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:15px;padding:20px;border:1px solid #2a2a4a}
.chart-area{height:120px;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.2);border-radius:8px;margin-top:10px}
.bar{display:flex;align-items:flex-end;gap:3px;height:100%;padding:10px}
.bar-segment{width:20px;border-radius:3px 3px 0 0;transition:height 0.3s}
.footer{text-align:center;padding:20px;color:#666;font-size:12px}
.stats-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:15px;margin-top:15px}
.stat-item{background:rgba(255,255,255,0.03);padding:15px;border-radius:8px;text-align:center}
.stat-value{font-size:24px;font-weight:bold}
.stat-label{font-size:10px;color:#888;text-transform:uppercase;margin-top:5px}
</style></head>
<body>

<div class="header"><div class="logo">eko Trading System</div><div class="status-badge"> ONLINE</div></div>

METRICS_SECTION

<div class="progress-section">
<div class="progress-header"><span>Progress - Daily Target: +5%</span><span>PROGRESS%</span></div>
<div class="progress-bar"><div class="progress-fill" style="width:PROGRESS%"></div></div>
</div>

<div class="charts-row">
<div class="chart-box">
<h3>Win/Loss Distribution</h3>
<div class="chart-area"><div class="bar">WIN_LOSS_BARS</div></div>
<div class="stats-row">
<div class="stat-item"><div class="stat-value positive">WINS</div><div class="stat-label">Wins</div></div>
<div class="stat-item"><div class="stat-value negative">LOSSES</div><div class="stat-label">Losses</div></div>
<div class="stat-item"><div class="stat-value neutral">WIN_RATE%</div><div class="stat-label">Win Rate</div></div>
</div>
</div>
<div class="chart-box">
<h3>P&L by Symbol</h3>
<div class="chart-area"><div class="bar">PNL_BARS</div></div>
</div>
</div>

<div class="section"><h3>System Modules</h3><div class="modules-grid">MODULES_HTML</div></div>

<div class="section"><h3>Trade History</h3>
<table><tr><th>Symbol</th><th>Direction</th><th>Entry Price</th><th>Current</th><th>P&L %</th><th>Status</th><th>Time</th></tr>TRADES_HTML</table>
</div>

<div class="footer"><p>Last Updated: TIMESTAMP | Eko Trading System v3.0 | Professional Dashboard</p></div>
</body></html>'''

def get_state():
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}

def get_trades():
    if DB_FILE.exists():
        return json.loads(DB_FILE.read_text()).get('trades', [])
    return []

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        state = get_state()
        trades = get_trades()
        stats = state.get('stats', {})
        modules = state.get('modules', {})
        
        daily_pnl = stats.get('daily_pnl_pct', 0)
        progress = min(abs(daily_pnl) / 5 * 100, 100) if daily_pnl != 0 else 0
        
        # Calculate metrics
        closed = [t for t in trades if t.get('status') == 'closed']
        wins = sum(1 for t in closed if t.get('pnl', 0) > 0)
        losses = len(closed) - wins
        win_rate = wins / len(closed) * 100 if closed else 0
        total_pnl = sum(t.get('pnl', 0) for t in closed)
        total_pnl_pct = sum(t.get('pnl_pct', 0) for t in closed)
        
        # P&L by symbol
        pnl_by_symbol = defaultdict(float)
        for t in trades:
            sym = t.get('symbol', 'UNKNOWN')
            pnl = t.get('pnl_pct') or 0
            pnl_by_symbol[sym] += pnl
        
        # Metrics section
        metrics = f'''
<div class="metrics-grid">
<div class="metric-card"><div class="metric-value neutral">{len(trades)}</div><div class="metric-label">Total Trades</div></div>
<div class="metric-card"><div class="metric-value {'positive' if daily_pnl >= 0 else 'negative'}">{daily_pnl:+.2f}%</div><div class="metric-label">Daily P&L</div></div>
<div class="metric-card"><div class="metric-value neutral">${total_pnl:+.2f}</div><div class="metric-label">Total P&L</div></div>
<div class="metric-card"><div class="metric-value {'positive' if win_rate >= 50 else 'negative'}">{win_rate:.1f}%</div><div class="metric-label">Win Rate</div></div>
<div class="metric-card"><div class="metric-value neutral">{stats.get('cycles', 0)}</div><div class="metric-label">Cycles</div></div>
<div class="metric-card"><div class="metric-value neutral">{stats.get('trades_today', 0)}</div><div class="metric-label">Trades Today</div></div>
</div>'''
        
        # Win/Loss bars
        max_wl = max(wins, losses, 1)
        win_height = (wins / max_wl) * 80 if max_wl > 0 else 0
        loss_height = (losses / max_wl) * 80 if max_wl > 0 else 0
        win_loss_bars = f'''
<div class="bar-segment" style="height:{win_height}%;background:#00cc96" title="Wins: {wins}"></div>
<div class="bar-segment" style="height:{loss_height}%;background:#ef553b" title="Losses: {losses}"></div>'''
        
        # P&L bars
        pnl_bars = ""
        for sym, pnl in sorted(pnl_by_symbol.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
            height = min(abs(pnl) * 10, 80)
            color = "#00cc96" if pnl >= 0 else "#ef553b"
            pnl_bars += f'<div class="bar-segment" style="height:{height}%;background:{color}" title="{sym}: {pnl:+.2f}%"></div>'
        
        # Modules
        modules_html = ""
        for name, active in modules.items():
            icon = "✓" if active else "✗"
            modules_html += f'<div class="module" style="opacity:{"1" if active else "0.5"}">{icon} {name.replace("_", " ").title()}</div>'
        
        # Trades
        trades_html = ""
        for t in reversed(trades[-20:]):
            s = t.get('symbol', '')
            d = t.get('direction', '')
            entry = t.get('entry_price', 0)
            current = t.get('current_price', 0)
            p = t.get('pnl_pct') or 0
            st = t.get('status', '')
            tm = t.get('timestamp', '')[:19] if t.get('timestamp') else ''
            pnl_class = 'positive' if p >= 0 else 'negative'
            trades_html += f'''<tr>
<td>{s}</td>
<td><span class="direction {d}">{d}</span></td>
<td>${entry:.4f}</td>
<td>${current:.4f}</td>
<td class="{pnl_class}">{p:+.2f}%</td>
<td>{st}</td>
<td style="color:#888;font-size:11px">{tm}</td>
</tr>'''
        
        if not trades_html:
            trades_html = '<tr><td colspan="7" style="text-align:center;color:#666">No trades yet</td></tr>'
        
        html = HTML
        html = html.replace('METRICS_SECTION', metrics)
        html = html.replace('PROGRESS', f'{progress:.1f}')
        html = html.replace('WIN_LOSS_BARS', win_loss_bars)
        html = html.replace('WINS', str(wins))
        html = html.replace('LOSSES', str(losses))
        html = html.replace('WIN_RATE', f'{win_rate:.1f}')
        html = html.replace('PNL_BARS', pnl_bars or '<div style="color:#666">No data</div>')
        html = html.replace('MODULES_HTML', modules_html)
        html = html.replace('TRADES_HTML', trades_html)
        html = html.replace('TIMESTAMP', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def log_message(self, *args): pass

print(f"\n Professional Dashboard: http://localhost:8502")
HTTPServer(('', 8502), Handler).serve_forever()
