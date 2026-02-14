#!/usr/bin/env python3
"""
Serve professional HTML dashboard without Streamlit dependencies.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from datetime import datetime
from pathlib import Path

HTML_TEMPLATE = Path('dashboard/professional_dashboard.html').read_text()
STATE_FILE = Path('unified_brain_state.json')
DB_FILE = Path('db/unified_trading.db')

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
        win_rate = 0
        closed = [t for t in trades if t.get('status') == 'closed']
        if closed:
            wins = sum(1 for t in closed if t.get('pnl', 0) > 0)
            win_rate = wins / len(closed) * 100
        
        total_pnl = sum(t.get('pnl', 0) for t in closed)
        
        # Metrics cards
        metrics = f'''
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value neutral">{len(trades)}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if daily_pnl >= 0 else 'negative'}">{daily_pnl:+.2f}%</div>
                <div class="metric-label">Daily P&L</div>
            </div>
            <div class="metric-card">
                <div class="metric-value neutral">{win_rate:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value neutral">${total_pnl:+.2f}</div>
                <div class="metric-label">Total P&L</div>
            </div>
            <div class="metric-card">
                <div class="metric-value neutral">{stats.get('cycles', 0)}</div>
                <div class="metric-label">Cycles</div>
            </div>
            <div class="metric-card">
                <div class="metric-value neutral">{stats.get('trades_today', 0)}</div>
                <div class="metric-label">Trades Today</div>
            </div>
        </div>'''
        
        # Modules
        modules_html = ''
        for name, active in modules.items():
            status = 'active' if active else 'inactive'
            icon = '✓' if active else '✗'
            modules_html += f'<div class="module {status}">{icon} {name.replace("_", " ").title()}</div>'
        
        # Trades
        trades_html = ''
        for t in reversed(trades[-15:]):
            s = t.get('symbol', '')
            d = t.get('direction', '')
            p = t.get('pnl_pct') or 0
            st = t.get('status', '')
            tm = t.get('timestamp', '')[:19] if t.get('timestamp') else ''
            trades_html += f'<tr><td>{s}</td><td><span class="direction {d}">{d}</span></td><td class="{'positive' if p >= 0 else 'negative'}">{p:+.2f}%</td><td>{st}</td><td style="color:#888">{tm}</td></tr>'
        
        if not trades_html:
            trades_html = '<tr><td colspan="5" style="text-align:center;color:#666">No trades yet</td></tr>'
        
        html = HTML_TEMPLATE
        html = html.replace('METRICS_PLACEHOLDER', metrics)
        html = html.replace('PROGRESS_PCT', f'{progress:.1f}')
        html = html.replace('TOTAL_PNL', f'${total_pnl:+.2f}')
        html = html.replace('WIN_RATE', f'{win_rate:.1f}')
        html = html.replace('MODULES_PLACEHOLDER', modules_html)
        html = html.replace('TRADES_PLACEHOLDER', trades_html)
        html = html.replace('TIMESTAMP', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def log_message(self, *args): pass

PORT = 8504
print(f"\n🌐 Professional Dashboard: http://localhost:{PORT}")
print("   Press Ctrl+C to stop\n")
HTTPServer(('', PORT), Handler).serve_forever()
