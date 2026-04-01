#!/usr/bin/env python3
"""
Solana Bot Dashboard — Real-time monitoring
"""
import json, os, sys
from pathlib import Path
from datetime import datetime, timezone

# ── Flask or http.server fallback ──────────────────────────────
try:
    from flask import Flask, jsonify, render_template_string
    FLASK = True
except ImportError:
    import http.server, socketserver, urllib.parse
    FLASK = False

WORKSPACE = Path("/home/enderj/.openclaw/workspace/Solana-Cripto-Trader/agents")
DATA_DIR = WORKSPACE / "data"
PORT = 8085

# ── Data helpers ────────────────────────────────────────────────
def load_json(fname, default=None):
    p = DATA_DIR / fname
    if not p.exists():
        return default or {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return default or {}

def get_fg_label(val):
    if val is None: return "N/A"
    if val <= 20: return "😨 Extreme Fear"
    if val <= 35: return "😰 Fear"
    if val <= 65: return "😐 Neutral"
    if val <= 80: return "😄 Greed"
    return "🤑 Extreme Greed"

def fmt_money(v):
    return f"${v:,.2f}" if v is not None else "$—"

def fmt_pct(v):
    return f"{v*100:.2f}%" if v is not None else "—"

def ago(ts_str):
    """Returns human-readable time since timestamp."""
    if not ts_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        diff = datetime.now(timezone.utc) - dt
        mins = int(diff.total_seconds() / 60)
        if mins < 1: return "just now"
        if mins < 60: return f"{mins}m ago"
        hrs = mins // 60
        if hrs < 24: return f"{hrs}h ago"
        return f"{(hrs//24)}d ago"
    except Exception:
        return ts_str

# ── HTML Template ─────────────────────────────────────────────────
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Solana Bot Dashboard</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0f; color: #e0e0e0; min-height: 100vh; }
.header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px 30px; border-bottom: 1px solid #2a2a3a; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }
.header h1 { font-size: 1.4em; color: #00d4ff; font-weight: 700; letter-spacing: -0.5px; }
.header h1 span { color: #ffd700; }
.header .meta { font-size: 0.8em; color: #888; }
.status-bar { display: flex; gap: 20px; padding: 15px 30px; background: #12121a; border-bottom: 1px solid #1e1e2e; flex-wrap: wrap; }
.stat { display: flex; flex-direction: column; gap: 2px; }
.stat .label { font-size: 0.7em; text-transform: uppercase; color: #666; letter-spacing: 0.5px; }
.stat .value { font-size: 1.1em; font-weight: 600; color: #fff; }
.stat .value.positive { color: #00c850; }
.stat .value.negative { color: #ff4444; }
.stat .value.neutral { color: #ffd700; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; padding: 20px; }
.card { background: #12121a; border: 1px solid #1e1e2e; border-radius: 12px; overflow: hidden; }
.card-header { background: #1a1a2a; padding: 12px 16px; font-size: 0.85em; font-weight: 600; color: #aaa; border-bottom: 1px solid #1e1e2e; display: flex; justify-content: space-between; align-items: center; }
.card-header .refresh { font-size: 0.7em; color: #555; font-weight: 400; }
.card-body { padding: 16px; }
.position-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #1a1a25; }
.position-row:last-child { border-bottom: none; }
.position-row .sym { font-weight: 700; font-size: 1em; }
.position-row .dir { font-size: 0.75em; padding: 2px 6px; border-radius: 4px; margin-left: 6px; }
.dir.short { background: #ff444420; color: #ff6666; }
.dir.long { background: #00c85020; color: #00ff66; }
.position-row .pnl { font-weight: 600; font-size: 0.95em; }
.pnl.pos { color: #00c850; }
.pnl.neg { color: #ff4444; }
.pnl.neu { color: #888; }
.insight-row { display: flex; gap: 10px; padding: 8px 0; border-bottom: 1px solid #1a1a25; font-size: 0.85em; }
.insight-row:last-child { border-bottom: none; }
.insight-row .emoji { font-size: 1.1em; flex-shrink: 0; }
.insight-row .text { color: #ccc; line-height: 1.4; }
.insight-row .conf { font-size: 0.75em; color: #666; white-space: nowrap; }
.rec-row { display: flex; gap: 10px; padding: 10px 0; border-bottom: 1px solid #1a1a25; font-size: 0.85em; }
.rec-row:last-child { border-bottom: none; }
.rec-row .badge { padding: 2px 8px; border-radius: 10px; font-size: 0.7em; font-weight: 700; flex-shrink: 0; }
.badge.auto { background: #00c85030; color: #00c850; }
.badge.manual { background: #ffd70030; color: #ffd700; }
.badge.critical { background: #ff444430; color: #ff4444; }
.anom-row { display: flex; gap: 10px; padding: 8px 0; font-size: 0.85em; color: #ff6666; }
.no-data { text-align: center; color: #555; padding: 30px; font-size: 0.9em; }
.section-title { font-size: 0.9em; font-weight: 700; color: #00d4ff; margin-bottom: 12px; }
.mini-chart { display: flex; gap: 3px; align-items: flex-end; height: 30px; }
.mini-bar { width: 8px; background: #2a2a3a; border-radius: 2px; }
.mini-bar.win { background: #00c850; }
.mini-bar.loss { background: #ff4444; }
.wr-display { font-size: 2em; font-weight: 800; }
.wr-display.win { color: #00c850; }
.wr-display.loss { color: #ff4444; }
.refresh-btn { background: #1e1e2e; color: #00d4ff; border: 1px solid #2a2a3a; padding: 6px 14px; border-radius: 8px; cursor: pointer; font-size: 0.8em; text-decoration: none; }
.refresh-btn:hover { background: #2a2a3a; }
.tabs { display: flex; gap: 2px; padding: 10px 20px; background: #0f0f18; border-bottom: 1px solid #1e1e2e; }
.tab { padding: 8px 16px; border-radius: 8px 8px 0 0; font-size: 0.8em; cursor: pointer; color: #555; }
.tab.active { background: #12121a; color: #00d4ff; }
.tab:hover:not(.active) { color: #888; }
.footer { text-align: center; padding: 20px; color: #444; font-size: 0.75em; }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>🤖 Solana Bot <span>Dream Engine</span></h1>
    <div class="meta">{{ now_str }} · Port {{port}}</div>
  </div>
  <div style="display:flex;gap:10px;align-items:center">
    <a href="/" class="refresh-btn">↻ Refresh</a>
  </div>
</div>

<div class="status-bar">
  <div class="stat">
    <span class="label">Equity</span>
    <span class="value {{ 'positive' if equity_return >= 0 else 'negative' }}">{{ equity }}</span>
  </div>
  <div class="stat">
    <span class="label">Return</span>
    <span class="value {{ 'positive' if equity_return >= 0 else 'negative' }}">{{ equity_return_pct }}</span>
  </div>
  <div class="stat">
    <span class="label">Win Rate</span>
    <span class="value">{{ win_rate }}%</span>
  </div>
  <div class="stat">
    <span class="label">Open Positions</span>
    <span class="value">{{ positions|length }} / 5</span>
  </div>
  <div class="stat">
    <span class="label">Risk Total</span>
    <span class="value {{ 'negative' if risk_total > 10 else 'positive' }}">${{ "%.2f"|format(risk_total) }}</span>
  </div>
  <div class="stat">
    <span class="label">Fear & Greed</span>
    <span class="value">{{ fg_disp }}</span>
  </div>
  <div class="stat">
    <span class="label">Dream Engine</span>
    <span class="value">{{ dream_last }}</span>
  </div>
</div>

<div class="grid">
  <!-- Portfolio -->
  <div class="card">
    <div class="card-header">📊 Portfolio</div>
    <div class="card-body">
      <div class="section-title">Open Positions ({{ positions|length }})</div>
      {% if positions %}
        {% for pos in positions %}
        <div class="position-row">
          <div>
            <span class="sym">{{ pos.symbol }}</span>
            <span class="dir {{ pos.direction }}">{{ pos.direction|upper }}</span>
            <div style="font-size:0.75em;color:#555;margin-top:2px">
              M=${{ "%.2f"|format(pos.margin) }} · N=${{ "%.2f"|format(pos.notional) }} · {{ pos.leverage }}x
            </div>
          </div>
          <div style="text-align:right">
            <div class="pnl {{ 'pos' if pos.pnl >= 0 else 'neg' }}">${{ "%.2f"|format(pos.pnl) }}</div>
            <div style="font-size:0.75em;color:#555">{{ "%.2f"|format(pos.pnl_pct) }}%</div>
          </div>
        </div>
        {% endfor %}
        <div style="margin-top:10px;padding-top:10px;border-top:1px solid #1e1e2e;display:flex;justify-content:space-between;font-size:0.85em;color:#888">
          <span>Total Unrealized P&L</span>
          <span class="{{ 'positive' if total_pnl >= 0 else 'negative' }}" style="font-weight:700">${{ "%.2f"|format(total_pnl) }}</span>
        </div>
      {% else %}
        <div class="no-data">No open positions</div>
      {% endif %}
    </div>
  </div>

  <!-- Win Rate -->
  <div class="card">
    <div class="card-header">📈 Trade History</div>
    <div class="card-body">
      <div style="text-align:center;padding:10px 0">
        <div class="wr-display {{ 'win' if wr_pct >= 50 else 'loss' }}">{{ wr_pct }}%</div>
        <div style="color:#555;font-size:0.85em">Win Rate ({{ total_trades }} trades)</div>
      </div>
      <div style="display:flex;gap:8px;margin-top:12px;font-size:0.85em">
        <div style="flex:1;background:#12121a;border-radius:8px;padding:10px;text-align:center">
          <div style="color:#00c850;font-size:1.4em;font-weight:700">{{ wins }}</div>
          <div style="color:#555;font-size:0.75em">Wins</div>
        </div>
        <div style="flex:1;background:#12121a;border-radius:8px;padding:10px;text-align:center">
          <div style="color:#ff4444;font-size:1.4em;font-weight:700">{{ losses }}</div>
          <div style="color:#555;font-size:0.75em">Losses</div>
        </div>
        <div style="flex:1;background:#12121a;border-radius:8px;padding:10px;text-align:center">
          <div style="color:#ffd700;font-size:1.4em;font-weight:700">${{ avg_win|default(0)|round(2) }}</div>
          <div style="color:#555;font-size:0.75em">Avg Win</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Dream Insights -->
  <div class="card">
    <div class="card-header">💡 Dream Engine Insights <span class="refresh">{{ dream_last }}</span></div>
    <div class="card-body">
      {% if ai_insight %}
      <div style="background:#0d2030;border:1px solid #00d4ff;border-radius:8px;padding:10px;margin-bottom:10px">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
          <span style="font-size:14px">🤖</span>
          <span style="color:#00d4ff;font-weight:700">AI Insight</span>
          <span style="background:#00d4ff20;color:#00d4ff;padding:2px 8px;border-radius:10px;font-size:11px">{{ (ai_insight.confidence*100)|int }}%</span>
        </div>
        <div style="color:#c9d1d9;font-size:13px;line-height:1.4">{{ ai_insight.key_insight }}</div>
        {% if ai_insight.tokens_to_avoid %}
        <div style="margin-top:6px;font-size:12px;color:#ff6666">🔴 Evitar: {{ ai_insight.tokens_to_avoid|join(', ') }}</div>
        {% endif %}
        {% if ai_insight.tokens_to_prefer %}
        <div style="margin-top:4px;font-size:12px;color:#00c850">🟢 Preferir: {{ ai_insight.tokens_to_prefer|join(', ') }}</div>
        {% endif %}
      </div>
      {% endif %}
      {% if insights %}
        {% for ins in insights %}
        <div class="insight-row">
          <span class="emoji">{{ '🔥' if ins.conf >= 0.9 else '💡' if ins.conf >= 0.75 else '💭' }}</span>
          <div style="flex:1">
            <div class="text">{{ ins.text }}</div>
            <div class="conf">{{ (ins.conf*100)|int }}% confidence</div>
          </div>
        </div>
        {% endfor %}
      {% else %}
        <div class="no-data">Sin insights cuantitativos aún.</div>
      {% endif %}
    </div>
  </div>

  <!-- Recommendations -->
  <div class="card">
    <div class="card-header">🎯 Recommendations</div>
    <div class="card-body">
      {% if recommendations %}
        {% for rec in recommendations %}
        <div class="rec-row">
          <span class="badge {{ 'auto' if rec.auto else 'manual' }}">{{ 'AUTO' if rec.auto else 'MANUAL' }}</span>
          <div style="flex:1">
            <div style="color:#ccc">{{ rec.text }}</div>
            <div style="color:#555;font-size:0.75em;margin-top:2px">{{ rec.action }}</div>
          </div>
        </div>
        {% endfor %}
      {% else %}
        <div class="no-data">Sin recomendaciones pendientes</div>
      {% endif %}
    </div>
  </div>

  <!-- Anomalies -->
  <div class="card">
    <div class="card-header">⚠️ Anomalías</div>
    <div class="card-body">
      {% if anomalies %}
        {% for a in anomalies %}
        <div class="anom-row">
          <span>🔴</span>
          <span>{{ a.text }}</span>
        </div>
        {% endfor %}
      {% else %}
        <div class="no-data">Sin anomalías detectadas ✅</div>
      {% endif %}
    </div>
  </div>

  <!-- Bot Status -->
  <div class="card">
    <div class="card-header">🔄 Bot Status</div>
    <div class="card-body">
      <div style="display:flex;flex-direction:column;gap:8px;font-size:0.85em">
        <div style="display:flex;justify-content:space-between"><span style="color:#555">Proceso</span><span class="{{ 'positive' if bot_running else 'negative' }}">{{ '🟢 Corriendo' if bot_running else '🔴 Detenido' }}</span></div>
        <div style="display:flex;justify-content:space-between"><span style="color:#555">Fear & Greed</span><span>{{ fg_disp }}</span></div>
        <div style="display:flex;justify-content:space-between"><span style="color:#555">Margen usado</span><span>${{ margin_used|default(0)|round(2) }} / $500</span></div>
        <div style="display:flex;justify-content:space-between"><span style="color:#555">Capital libre</span><span>${{ capital_free|default(0)|round(2) }}</span></div>
        <div style="display:flex;justify-content:space-between"><span style="color:#555">Crons activos</span><span>~{{ crons_count }}</span></div>
        <div style="display:flex;justify-content:space-between"><span style="color:#555">Portfolio TP</span><span style="color:#ffd700">$2 target</span></div>
      </div>
    </div>
  </div>

</div>

<div class="footer">
  Solana Bot Dream Engine · Auto-refresh cada 30s · {{ now_str }}
</div>

<script>
// Auto-refresh cada 30s
setTimeout(() => location.reload(), 30000);
</script>
</body>
</html>
"""

# ── Data gathering ──────────────────────────────────────────────
def get_dashboard_data():
    portfolio = load_json("portfolio.json", {})
    market = load_json("market_latest.json", {})
    insights_data = load_json("dream_insights.json", {})
    signals = load_json("signals_latest.json", {})
    history = load_json("trade_history.json", {"trades": []})

    # Portfolio
    positions = portfolio.get("positions", [])
    open_pos = [p for p in positions if p.get("status") == "open"]
    capital = portfolio.get("capital_usd", 0)
    initial = portfolio.get("initial_capital", 500)
    total_margin = sum(p.get("margin_usd", 0) for p in open_pos)
    total_pnl = sum(p.get("pnl_usd", 0) for p in open_pos)
    equity = capital + total_margin + total_pnl
    equity_return = equity - initial
    equity_return_pct = (equity_return / initial) * 100 if initial > 0 else 0

    # Trade history
    trades = history.get("trades", []) if isinstance(history, dict) else history
    wins = len([t for t in trades if t.get("pnl_usd", 0) > 0])
    losses = len([t for t in trades if t.get("pnl_usd", 0) <= 0])
    total_trades = wins + losses
    wr_pct = int(wins / total_trades * 100) if total_trades > 0 else 0
    win_list = [t["pnl_usd"] for t in trades if t.get("pnl_usd", 0) > 0]
    avg_win = sum(win_list) / len(win_list) if win_list else 0

    # Risk
    risk_total = 0
    for p in open_pos:
        e = p.get("entry_price", 0)
        n = p.get("notional_value", 0)
        sl = p.get("sl_price", 0)
        if e > 0 and n > 0:
            sl_dist = abs(sl - e) / e
            risk_total += n * sl_dist + n * 0.002

    # F&G
    fg_data = market.get("fear_greed", {})
    fg_val = fg_data.get("value", 50) if isinstance(fg_data, dict) else 50
    fg_disp = f"{fg_val}/100 — {get_fg_label(fg_val)}"

    # Dream insights
    insights = []
    for ins in insights_data.get("insights", [])[:6]:
        insights.append({
            "text": ins.get("finding", str(ins)),
            "conf": ins.get("confidence", 0.5),
        })

    recs = []
    for rec in insights_data.get("recommendations", [])[:4]:
        recs.append({
            "text": rec.get("finding", str(rec)),
            "action": rec.get("action", ""),
            "auto": rec.get("auto_apply", False),
            "is_ai": rec.get("is_ai", False),
        })
    
    ai_insight = insights_data.get("ai_insight", None)

    anomalies = []
    for a in insights_data.get("anomalies", [])[:4]:
        anomalies.append({
            "text": a.get("finding", str(a)),
        })

    # Dream last run
    dream_ts = insights_data.get("timestamp", "")
    dream_last = ago(dream_ts)

    # Format positions
    formatted_pos = []
    for p in open_pos:
        pnl = p.get("pnl_usd", 0)
        pnl_pct = p.get("pnl_pct", 0)
        formatted_pos.append({
            "symbol": p.get("symbol", "?"),
            "direction": p.get("direction", "?"),
            "margin": p.get("margin_usd", 0),
            "notional": p.get("notional_value", 0),
            "leverage": p.get("leverage", 3),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct * 100, 2) if pnl_pct else 0,
        })

    return {
        "equity": f"${equity:,.2f}",
        "equity_return": equity_return,
        "equity_return_pct": f"{equity_return_pct:+.2f}%",
        "win_rate": wr_pct,
        "wr_pct": wr_pct,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "avg_win": round(avg_win, 2),
        "positions": formatted_pos,
        "total_pnl": round(total_pnl, 2),
        "risk_total": round(risk_total, 2),
        "fg_disp": fg_disp,
        "fg_val": fg_val,
        "dream_last": dream_last,
        "insights": insights,
        "recommendations": recs,
        "ai_insight": ai_insight,
        "anomalies": anomalies,
        "bot_running": True,  # checked externally
        "capital_free": round(capital, 2),
        "margin_used": round(total_margin, 2),
        "crons_count": "~24",
        "now_str": datetime.now().strftime("%H:%M"),
        "port": PORT,
    }

# ── Flask app ────────────────────────────────────────────────────
if FLASK:
    app = Flask(__name__)

    @app.route("/")
    def index():
        data = get_dashboard_data()
        return render_template_string(TEMPLATE, **data)

    @app.route("/api/status")
    def api_status():
        data = get_dashboard_data()
        return jsonify(data)

    @app.route("/api/dream")
    def api_dream():
        return jsonify(load_json("dream_insights.json", {}))

    @app.route("/api/portfolio")
    def api_portfolio():
        return jsonify(load_json("portfolio.json", {}))

if __name__ == "__main__":
    if FLASK and app:
        app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
    else:
        # Fallback: simple http.server
        class Handler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/" or self.path == "/index.html":
                    data = get_dashboard_data()
                    html = render_template_string(TEMPLATE, **data)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(html.encode())
                elif self.path == "/api/status":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(get_dashboard_data()).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"🚀 Solana Bot Dashboard on http://localhost:{PORT}")
            httpd.serve_forever()
