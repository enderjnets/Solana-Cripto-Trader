#!/usr/bin/env python3
"""
BitTrader Production Dashboard — YouTube Stats + Video Production Monitor
Flask backend serving real-time data. Port: 8000
"""

import json
import os
import math
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# ── Data paths ────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/enderj/.openclaw/workspace")
DATA_DIR = WORKSPACE / "bittrader/agents/data"
MEMORY_DIR = WORKSPACE / "memory"
SOLANA_DATA = Path("/home/enderj/.openclaw/workspace/Solana-Cripto-Trader/agents/data")

def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default or {}

def load_youtube_stats():
    """Load YouTube stats from latest file."""
    stats_file = DATA_DIR / "youtube_stats_latest.json"
    data = load_json(stats_file)
    if "error" in data:
        # Try to get from memory
        history = sorted(MEMORY_DIR.glob("youtube_stats_*.json"))
        for h in reversed(history):
            d = load_json(h)
            if "error" not in d:
                return d
        return {"error": str(data.get("error", "Unknown"))}
    return data

def load_queue():
    """Load upload queue."""
    q_file = DATA_DIR / "upload_queue.json"
    return load_json(q_file, [])

def load_scripts():
    """Load scripts from guiones_latest.json."""
    g_file = DATA_DIR / "guiones_latest.json"
    return load_json(g_file, {"scripts": []})

def load_production_history():
    """Load production history."""
    history = sorted(DATA_DIR.glob("production_2026-*.json"))
    records = []
    for h in reversed(history[-7:]):  # last 7 days
        d = load_json(h)
        records.append(d)
    return records

def yt_bar(v, max_v, width=200):
    """Simple ASCII bar."""
    if max_v == 0: return ""
    pct = min(1.0, v / max_v)
    filled = int(pct * width)
    return "█" * filled + "░" * (width - filled)

MONTHS_ES = ["ene","feb","mar","abr","may","jun","jul","ago","sep","oct","nov","dic"]

HTML = '''<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>📊 BitTrader Production Dashboard</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:#0d1117; color:#e6edf3; min-height:100vh; }
.container { max-width:1400px; margin:0 auto; padding:20px; }

/* Header */
header { display:flex; justify-content:space-between; align-items:center; padding:15px 0; border-bottom:1px solid #21262d; margin-bottom:20px; }
header h1 { font-size:22px; color:#58a6ff; }
.refresh { background:#238636; color:#fff; border:none; padding:8px 16px; border-radius:6px; cursor:pointer; font-size:13px; }
.refresh:hover { background:#2ea043; }
.ts { font-size:12px; color:#484f58; }

/* Sections */
.section { margin-bottom:30px; }
.section-title { font-size:14px; color:#8b949e; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px; border-left:3px solid #58a6ff; padding-left:10px; }

/* Cards grid */
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }
.card { background:#161b22; border:1px solid #21262d; border-radius:8px; padding:16px; }
.card .label { font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px; }
.card .value { font-size:28px; font-weight:700; color:#e6edf3; }
.card .sub { font-size:11px; color:#8b949e; margin-top:4px; }
.card .positive { color:#3fb950; }
.card .negative { color:#f85149; }
.card .warn { color:#d29922; }

/* Progress bar */
.prog-wrap { background:#21262d; border-radius:4px; height:8px; margin-top:8px; overflow:hidden; }
.prog-fill { height:100%; border-radius:4px; transition:width 0.5s; }
.prog-fill.green { background:linear-gradient(90deg,#238636,#3fb950); }
.prog-fill.blue { background:linear-gradient(90deg,#1f6feb,#58a6ff); }
.prog-fill.yellow { background:linear-gradient(90deg,#9e6a03,#d29922); }
.prog-fill.red { background:linear-gradient(90deg,#b62324,#f85149); }

/* Tables */
table { width:100%; border-collapse:collapse; }
th { text-align:left; font-size:11px; color:#8b949e; padding:8px 10px; border-bottom:1px solid #21262d; }
td { padding:10px; font-size:13px; border-bottom:1px solid #161b22; }
tr:hover td { background:#161b22; }
.status-badge { padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600; }
.status-ready { background:#238636; color:#fff; }
.status-blocked { background:#b62324; color:#fff; }
.status-uploaded { background:#1f6feb; color:#fff; }
.status-pending { background:#9e6a03; color:#fff; }
.status-produced { background:#8b949e; color:#fff; }

/* Two column layout */
.two-col { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
@media(max-width:900px){ .two-col { grid-template-columns:1fr; } }

/* Charts area */
.chart-card { background:#161b22; border:1px solid #21262d; border-radius:8px; padding:16px; margin-bottom:12px; }
.chart-title { font-size:13px; color:#8b949e; margin-bottom:12px; }

/* Script list */
.script-item { background:#161b22; border:1px solid #21262d; border-radius:6px; padding:12px; margin-bottom:8px; }
.script-title { font-size:13px; font-weight:600; color:#e6edf3; margin-bottom:4px; }
.script-meta { font-size:11px; color:#8b949e; }
.script-badge { display:inline-block; padding:1px 6px; border-radius:4px; font-size:10px; margin-right:4px; }
.badge-long { background:#1f6feb33; color:#58a6ff; border:1px solid #1f6feb; }
.badge-short { background:#23863633; color:#3fb950; border:1px solid #238636; }

/* Bar chart */
.bar-row { display:flex; align-items:center; margin-bottom:6px; font-size:12px; }
.bar-label { width:140px; color:#8b949e; flex-shrink:0; }
.bar-val { width:60px; text-align:right; color:#e6edf3; flex-shrink:0; }
.bar-line { flex:1; height:6px; background:#21262d; border-radius:3px; margin:0 8px; overflow:hidden; }
.bar-fill { height:100%; border-radius:3px; }
.bar-fill.blue { background:#1f6feb; }
.bar-fill.green { background:#3fb950; }
.bar-fill.red { background:#f85149; }
.bar-fill.yellow { background:#d29922; }

/* Pipeline tracker */
.pipeline { display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }
.p-step { background:#161b22; border:1px solid #21262d; border-radius:6px; padding:8px 12px; font-size:12px; flex:1; min-width:120px; }
.p-step.active { border-color:#58a6ff; background:#1f6feb22; }
.p-step.done { border-color:#3fb950; background:#23863622; }
.p-step .step-label { font-size:10px; color:#8b949e; }
.p-step .step-val { font-size:13px; font-weight:600; color:#e6edf3; margin-top:2px; }

/* Queue section */
.queue-item { display:flex; justify-content:space-between; align-items:center; padding:10px 0; border-bottom:1px solid #21262d; }
.queue-left { flex:1; }
.queue-title { font-size:13px; color:#e6edf3; margin-bottom:2px; }
.queue-meta { font-size:11px; color:#8b949e; }
.queue-right { text-align:right; }

/* Flex utilities */
.flex { display:flex; gap:16px; align-items:center; }
.gap-sm { gap:8px; }
.gap-md { gap:16px; }
.mt { margin-top:8px; }
.mb { margin-bottom:8px; }
.text-sm { font-size:12px; }
.text-xs { font-size:11px; }
.text-muted { color:#8b949e; }
.text-green { color:#3fb950; }
.text-red { color:#f85149; }
.text-blue { color:#58a6ff; }
.text-yellow { color:#d29922; }
</style>
</head>
<body>
<div class="container">
<header>
  <h1>📊 BitTrader Production Dashboard</h1>
  <div>
    <span class="ts" id="ts">Actualizando...</span>
    <button class="refresh" onclick="location.reload()">🔄 Refresh</button>
  </div>
</header>

<!-- YouTube Channel Stats -->
<div class="section">
  <div class="section-title">📺 YouTube Channel — BitTrader (@bittrader9259)</div>
  <!-- Mac Pool + Live Production -->
  <div class="section-title">💻 Mac Pool + Generación en Vivo</div>
  <div class="grid" style="margin-bottom:20px;">
    <div class="card" style="flex:1" id="mac-mbp">
      <div class="label">🍎 MacBook Pro M3 Pro</div>
      <div class="value text-green" id="mbp-status">...</div>
      <div class="sub" id="mbp-sub">Verificando...</div>
      <div class="prog-wrap mt">
        <div class="prog-fill blue" id="mbp-bar" style="width:0%"></div>
      </div>
      <div class="text-xs text-muted mt" id="mbp-detail"></div>
    </div>
    <div class="card" style="flex:1" id="mac-mba">
      <div class="label">🍏 MacBook Air M4</div>
      <div class="value text-green" id="mba-status">...</div>
      <div class="sub" id="mba-sub">Verificando...</div>
      <div class="prog-wrap mt">
        <div class="prog-fill blue" id="mba-bar" style="width:0%"></div>
      </div>
      <div class="text-xs text-muted mt" id="mba-detail"></div>
    </div>
    <div class="card" style="flex:1" id="gen-live">
      <div class="label">🎬 Producer en Vivo</div>
      <div class="value text-yellow" id="prod-live-status">...</div>
      <div class="sub" id="prod-live-sub">Verificando...</div>
      <div class="prog-wrap mt">
        <div class="prog-fill green" id="prod-bar" style="width:0%"></div>
      </div>
      <div class="text-xs text-muted mt" id="prod-live-detail"></div>
    </div>
    <div class="card" style="flex:1" id="minimax-card">
      <div class="label">🤖 MiniMax API</div>
      <div class="value text-green" id="minimax-status">...</div>
      <div class="sub" id="minimax-sub">Verificando...</div>
      <div class="prog-wrap mt">
        <div class="prog-fill yellow" id="minimax-bar" style="width:100%"></div>
      </div>
    </div>
  </div>

  <!-- Production Pipeline -->
  <div class="section-title">🎬 Pipeline de Producción</div>
      <div class="sub">de 1,000 para YPP</div>
      <div class="prog-wrap mt">
        <div class="prog-fill green" id="subs-bar" style="width:100%"></div>
      </div>
    </div>
    <div class="card">
      <div class="label">Watch Time</div>
      <div class="value" id="wt">—</div>
      <div class="sub">de 4,000h para YPP</div>
      <div class="prog-wrap mt">
        <div class="prog-fill blue" id="wt-bar" style="width:0%"></div>
      </div>
    </div>
    <div class="card">
      <div class="label">Vistas Totales</div>
      <div class="value" id="views">—</div>
      <div class="sub" id="views-48">—</div>
    </div>
    <div class="card">
      <div class="label">Videos Publicados</div>
      <div class="value" id="videos">—</div>
      <div class="sub" id="period">—</div>
    </div>
    <div class="card">
      <div class="label">Velocidad WH/semana</div>
      <div class="value text-yellow" id="wt-speed">—</div>
      <div class="sub" id="wt-target">Meta: 57.7h/sem</div>
    </div>
    <div class="card">
      <div class="label">Meta Monetización</div>
      <div class="value text-yellow" id="meta-pct">—</div>
      <div class="sub" id="meta-date">Estimado</div>
    </div>
  </div>
</div>

<!-- Alert -->
<div id="yt-alert" style="display:none; background:#b6232422; border:1px solid #b62324; border-radius:8px; padding:12px 16px; margin-bottom:20px;" class="text-red text-sm">
  <strong>⚠️ Alerta:</strong> <span id="yt-alert-text"></span>
</div>

<!-- Two column: Pipeline + Queue -->
<div class="two-col">
  <!-- Production Pipeline -->
  <div>
    <div class="section-title">🎬 Pipeline de Producción</div>
    
    <div class="grid">
      <div class="card">
        <div class="label">Scripts Pendientes</div>
        <div class="value text-yellow" id="scripts-pending">—</div>
      </div>
      <div class="card">
        <div class="label">Producidos (hoy)</div>
        <div class="value text-green" id="scripts-produced">—</div>
      </div>
      <div class="card">
        <div class="label">En Cola Upload</div>
        <div class="value text-blue" id="queue-total">—</div>
      </div>
      <div class="card">
        <div class="label">Listos para Subir</div>
        <div class="value text-green" id="queue-ready">—</div>
      </div>
    </div>

    <!-- Current production -->
    <div class="chart-card mt">
      <div class="chart-title">🔄 Producción Activa</div>
      <div id="prod-status" class="text-muted text-sm">Cargando...</div>
    </div>

    <!-- Scripts pendientes -->
    <div class="chart-card">
      <div class="chart-title">📋 Próximos Scripts a Producir</div>
      <div id="scripts-list">Cargando...</div>
    </div>
  </div>

  <!-- Upload Queue -->
  <div>
    <div class="section-title">📤 Cola de Upload</div>
    
    <div class="grid mb">
      <div class="card">
        <div class="label">Blockeados</div>
        <div class="value text-red" id="queue-blocked">—</div>
      </div>
      <div class="card">
        <div class="label">Subidos</div>
        <div class="value" id="queue-uploaded">—</div>
      </div>
    </div>

    <div id="queue-list">Cargando...</div>
  </div>
</div>

<!-- Recent Videos -->
<div class="section">
  <div class="section-title">🎥 Videos Recientes (últimos 7 días)</div>
  <div id="recent-videos">Cargando...</div>
</div>

<!-- Stats History -->
<div class="section">
  <div class="section-title">📈 Evolución de Suscriptores</div>
  <div class="chart-card">
    <div id="sub-chart">Cargando...</div>
  </div>
</div>

</div>

<script>
function fmt(n) {
  if (n >= 1000000) return (n/1000000).toFixed(1)+'M';
  if (n >= 1000) return (n/1000).toFixed(1)+'K';
  return String(n);
}

async function loadMacs() {
  try {
    const r = await fetch('/api/mac-status');
    const macs = await r.json();
    for (const m of macs) {
      const id = m.id; // mbp or mba
      const statusEl = document.getElementById(id + '-status');
      const subEl = document.getElementById(id + '-sub');
      const barEl = document.getElementById(id + '-bar');
      const detailEl = document.getElementById(id + '-detail');
      if (!m.online) {
        statusEl.textContent = '⚠️ Offline';
        statusEl.style.color = '#f85149';
        subEl.textContent = 'ComfyUI no responde';
        barEl.style.width = '0%';
        detailEl.textContent = '';
      } else {
        const busy = m.queue_running + m.queue_pending;
        statusEl.textContent = busy > 0 ? '🔥 Generando' : '✅ Idle';
        statusEl.style.color = busy > 0 ? '#d29922' : '#3fb950';
        subEl.textContent = `ComfyUI ${m.comfyui_version} | ${m.os}`;
        barEl.style.width = (busy > 0 ? 70 : 100) + '%';
        barEl.style.background = busy > 0 
          ? 'linear-gradient(90deg,#9e6a03,#d29922)' 
          : 'linear-gradient(90deg,#238636,#3fb950)';
        const dev = m.devices && m.devices[0];
        detailEl.textContent = dev 
          ? `${dev.device_name || 'MPS'}` 
          : 'Sin info GPU';
      }
    }
  } catch(e) { console.error('Mac status error:', e); }
}

async function loadProducerLive() {
  try {
    const r = await fetch('/api/producer-live');
    const p = await r.json();
    const statusEl = document.getElementById('prod-live-status');
    const subEl = document.getElementById('prod-live-sub');
    const barEl = document.getElementById('prod-bar');
    const detailEl = document.getElementById('prod-live-detail');
    if (!p.running) {
      statusEl.textContent = '⏸️ Idle';
      statusEl.style.color = '#8b949e';
      subEl.textContent = 'Producer en espera';
      barEl.style.width = '0%';
      detailEl.textContent = '';
    } else {
      const stage = p.stage || '';
      statusEl.textContent = stage === 'hybrid' ? '🎬 Hybrid v2' 
        : stage === 'minimax' ? '🎨 MiniMax'
        : stage === 'ltx' ? '🎥 LTX-Video'
        : '⚙️ Procesando';
      statusEl.style.color = '#58a6ff';
      subEl.textContent = p.current_script || 'Script activo';
      // Calculate progress
      let done = 0, total = 0;
      if (p.ltx_total > 0) { done += p.ltx_done; total += p.ltx_total; }
      if (p.mini_total > 0) { done += p.mini_done; total += p.mini_total; }
      const pct = total > 0 ? Math.round((done/total)*100) : 0;
      barEl.style.width = pct + '%';
      barEl.style.background = pct > 80 ? 'linear-gradient(90deg,#238636,#3fb950)' : 'linear-gradient(90deg,#1f6feb,#58a6ff)';
      detailEl.textContent = p.ltx_total > 0 
        ? `LTX: ${p.ltx_done}/${p.ltx_total} | MiniMax: ${p.mini_done}/${p.mini_total}`
        : '';
    }
  } catch(e) { console.error('Producer error:', e); }
}

async function loadMiniMax() {
  try {
    const r = await fetch('/api/minimax-status');
    const m = await r.json();
    const statusEl = document.getElementById('minimax-status');
    const subEl = document.getElementById('minimax-sub');
    const barEl = document.getElementById('minimax-bar');
    if (!m.available) {
      statusEl.textContent = '⚠️ Error';
      statusEl.style.color = '#f85149';
      subEl.textContent = m.error || 'API no responde';
      barEl.style.width = '0%';
    } else {
      statusEl.textContent = '✅ OK';
      statusEl.style.color = '#3fb950';
      subEl.textContent = 'API MiniMax activa';
      barEl.style.width = '100%';
      barEl.style.background = 'linear-gradient(90deg,#238636,#3fb950)';
    }
  } catch(e) { 
    document.getElementById('minimax-status').textContent = '⚠️ Error';
    document.getElementById('minimax-sub').textContent = 'No se pudo verificar';
  }
}

async function load() {
  const [yt, queue, scripts] = await Promise.all([
    fetch('/api/youtube').then(r=>r.json()),
    fetch('/api/queue').then(r=>r.json()),
    fetch('/api/scripts').then(r=>r.json())
  ]);

  // Also load live status
  loadMacs();
  loadMiniMax();
  loadProducerLive();

  document.getElementById('ts').textContent = new Date().toLocaleString('es-MX',{timeZone:'America/Denver'});

  // YouTube stats
  if (!yt.error) {
    const subs = yt.channel?.subscribers || 0;
    const wt = yt.channel?.watch_time_hours || 0;
    const views = yt.channel?.total_views || 0;
    const videos = yt.channel?.total_videos || 0;
    
    document.getElementById('subs').textContent = fmt(subs);
    document.getElementById('views').textContent = fmt(views);
    document.getElementById('videos').textContent = videos;
    document.getElementById('views-48').textContent = yt.recent_48h ? `+${fmt(yt.recent_48h)} en 48h` : '';
    
    // Watch time
    const wt_pct = Math.min(100, (wt/4000)*100);
    document.getElementById('wt').textContent = wt.toFixed(1)+'h';
    document.getElementById('wt-bar').style.width = wt_pct+'%';
    
    // Speed
    const speed = yt.weekly_watch_time || 0;
    document.getElementById('wt-speed').textContent = speed.toFixed(1)+'h/sem';
    
    // Meta
    const wt_needed = Math.max(0, 4000 - wt);
    const weeks_needed = speed > 0 ? wt_needed / speed : 999;
    const meta_pct = Math.min(100, (wt/4000)*100).toFixed(1)+'%';
    document.getElementById('meta-pct').textContent = meta_pct;
    if (weeks_needed < 52) {
      document.getElementById('meta-date').textContent = `~${Math.ceil(weeks_needed)} sem para meta`;
    } else {
      document.getElementById('meta-date').textContent = `~${(weeks_needed/52).toFixed(1)} años para meta`;
    }
    
    // Alert
    if (yt.alert) {
      document.getElementById('yt-alert').style.display='block';
      document.getElementById('yt-alert-text').textContent = yt.alert;
    }
    
    // Subscribers bar (100% since we have 2920 of 1000)
    document.getElementById('subs-bar').style.width='100%';
  }

  // Scripts
  const pending = scripts.filter(s=>s.status==='pending');
  const produced = scripts.filter(s=>s.status==='produced');
  document.getElementById('scripts-pending').textContent = pending.length;
  document.getElementById('scripts-produced').textContent = produced.length;

  // Scripts list
  let scriptsHtml = '';
  for (const s of pending.slice(0,5)) {
    const badge = s.type==='long' ? 'LONG':'SHORT';
    const cls = s.type==='long' ? 'badge-long':'badge-short';
    scriptsHtml += `<div class="script-item">
      <div class="script-title">${s.title||'Sin título'}</div>
      <div class="script-meta">
        <span class="script-badge ${cls}">${badge}</span>
        ${s.tags ? s.tags.slice(0,3).map(t=>`<span class="script-badge" style="background:#21262d;color:#8b949e">${t}</span>`).join('') : ''}
      </div>
    </div>`;
  }
  if (!scriptsHtml) scriptsHtml = '<div class="text-muted text-sm">No hay scripts pendientes</div>';
  document.getElementById('scripts-list').innerHTML = scriptsHtml;

  // Production status
  document.getElementById('prod-status').textContent = pending.length > 0 
    ? `🎬 Producer activo — ${pending.length} scripts en cola. Próximo: ${pending[0].title||'?'}`
    : '⏸️ Sin scripts pendientes. Producer en espera.';

  // Queue
  const total = queue.length;
  const ready = queue.filter(v=>v.status==='ready').length;
  const blocked = queue.filter(v=>v.status==='blocked').length;
  const uploaded = queue.filter(v=>v.status==='uploaded').length;
  document.getElementById('queue-total').textContent = total;
  document.getElementById('queue-ready').textContent = ready;
  document.getElementById('queue-blocked').textContent = blocked;
  document.getElementById('queue-uploaded').textContent = uploaded;

  let queueHtml = '';
  for (const v of queue.slice(0,10)) {
    const status = v.status || 'unknown';
    const sb = status==='ready'?'status-ready':status==='blocked'?'status-blocked':status==='uploaded'?'status-uploaded':'status-pending';
    const scheduled = v.scheduled_date ? new Date(v.scheduled_date).toLocaleString('es-MX',{timeZone:'America/Denver',month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}) : '—';
    queueHtml += `<div class="queue-item">
      <div class="queue-left">
        <div class="queue-title">${v.title||'?'}</div>
        <div class="queue-meta">${v.type||'?'} | ${v.duration||0}s | ${v.size_mb||0}MB | ⏱${scheduled}</div>
      </div>
      <div class="queue-right">
        <span class="status-badge ${sb}">${status}</span>
      </div>
    </div>`;
  }
  document.getElementById('queue-list').innerHTML = queueHtml || '<div class="text-muted text-sm">Cola vacía</div>';
}

load();
setInterval(load, 60000); // refresh every minute
setInterval(loadMacs, 5000); // poll Mac status every 5s
setInterval(loadProducerLive, 10000); // poll producer every 10s
</script>
</body>
</html>'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/youtube')
def api_youtube():
    stats = load_youtube_stats()
    if "error" in stats:
        return jsonify(stats)
    
    channel = stats.get("channel", {})
    recent = stats.get("recent_videos", [])
    
    # Calculate 48h views
    now = datetime.now(timezone.utc)
    recent_48h = 0
    for v in recent:
        try:
            pub = datetime.fromisoformat(v.get("published_at","2000").replace("Z","+00:00"))
            if (now - pub).total_seconds() < 172800:  # 48h
                recent_48h += v.get("views", 0)
        except: pass
    
    # Estimate weekly watch time from views (avg 2min per view for shorts, 8min for longs)
    total_views = channel.get("total_views", 0)
    n_videos = channel.get("total_videos", 1)
    avg_watch = 180  # seconds per video average
    estimated_wt = (total_views * avg_watch) / 3600
    
    return jsonify({
        "channel": {
            "subscribers": channel.get("subscribers", 0),
            "total_views": total_views,
            "total_videos": n_videos,
            "watch_time_hours": round(estimated_wt, 1),
        },
        "recent_48h": recent_48h,
        "weekly_watch_time": round(estimated_wt / 4, 1),  # rough weekly estimate
        "recent_videos": recent[:10],
        "alert": stats.get("alerta", "")
    })

@app.route('/api/queue')
def api_queue():
    return jsonify(load_queue())

@app.route('/api/scripts')
def api_scripts():
    data = load_scripts()
    return jsonify(data.get("scripts", []))

@app.route('/api/production')
def api_production():
    return jsonify(load_production_history())

# ── Mac Pool Status ──────────────────────────────────────────────────────────

def _check_mac_comfyui(url: str, timeout: float = 3.0) -> dict:
    """Check ComfyUI status on a Mac."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{url}/system_stats")
        resp = urllib.request.urlopen(req, timeout=timeout)
        stats = json.loads(resp.read())
        queue_req = urllib.request.Request(f"{url}/queue")
        queue_resp = urllib.request.urlopen(queue_req, timeout=timeout)
        queue = json.loads(queue_resp.read())
        return {
            "online": True,
            "os": stats.get("system", {}).get("os", "?"),
            "comfyui_version": stats.get("system", {}).get("comfyui_version", "?"),
            "python_version": stats.get("system", {}).get("python_version", "?"),
            "devices": stats.get("devices", []),
            "queue_running": len(queue.get("queue_running", [])),
            "queue_pending": len(queue.get("queue_pending", [])),
        }
    except Exception as e:
        return {"online": False, "error": str(e)}

@app.route('/api/mac-status')
def api_mac_status():
    """Status of both Macs running ComfyUI + LTX-Video."""
    macs = [
        {"id": "mbp", "name": "MacBook Pro M3 Pro", "host": "enderj@enders-macbook-pro.local",
         "url": "http://enders-macbook-pro.local:8188", "priority": 1},
        {"id": "mba", "name": "MacBook Air M4", "host": "enderj@100.118.215.73",
         "url": "http://100.118.215.73:8188", "priority": 2},
    ]
    result = []
    for mac in macs:
        status = _check_mac_comfyui(mac["url"])
        result.append({**mac, **status})
    return jsonify(result)

# ── Producer Live Status ────────────────────────────────────────────────────

def _read_producer_log() -> dict:
    """Read the current producer watch log for live progress."""
    try:
        log_path = Path("/tmp/producer_watch.log")
        if not log_path.exists():
            return {"running": False}
        content = log_path.read_text()
        lines = content.strip().split("\n")
        
        # Find last non-empty lines
        active_lines = [l for l in lines if l.strip()][-30:]
        
        # Check if producer is running
        import subprocess
        proc = subprocess.run(["pgrep", "-a", "producer.py"], capture_output=True, text=True)
        running = "producer.py" in proc.stdout
        
        # Find current script being produced
        current_script = ""
        clips_done = 0
        clips_total = 0
        ltx_done = 0
        ltx_total = 0
        mini_done = 0
        mini_total = 0
        stage = "idle"
        last_ltx = ""
        
        for line in reversed(active_lines):
            if "📝" in line and "[" in line:
                current_script = line.split("]")[-1].strip() if "]" in line else current_script
                break
        
        for line in active_lines:
            if "LTX clip:" in line and "✅" in line:
                ltx_done += 1
                last_ltx = line.strip()
            elif "MINIMAX:" in line:
                stage = "minimax"
                parts = line.split("MINIMAX:")[-1].strip()
                if "/" in parts:
                    mini_done = int(parts.split("/")[0].strip().split()[-1]) if parts.split("/")[0].strip() else 0
                    mini_total = int(parts.split("/")[-1].strip().split()[0]) if "/" in parts else 0
            elif "✅ Audio:" in line:
                stage = "audio"
            elif "Hybrid v2:" in line:
                stage = "hybrid"
                # Parse clip counts from "HYBRID v2: X LTX + Y MiniMax"
                import re
                m = re.search(r"(\d+)\s+LTX", line)
                if m: ltx_total = int(m.group(1))
                m = re.search(r"(\d+)\s+MiniMax", line)
                if m: mini_total = int(m.group(1))
        
        return {
            "running": running,
            "stage": stage,
            "current_script": current_script[:80],
            "ltx_done": ltx_done, "ltx_total": ltx_total,
            "mini_done": mini_done, "mini_total": mini_total,
            "last_ltx": last_ltx[:100],
            "recent_lines": active_lines[-10:],
        }
    except Exception as e:
        return {"running": False, "error": str(e)}

@app.route('/api/producer-live')
def api_producer_live():
    return jsonify(_read_producer_log())

# ── MiniMax API Status ────────────────────────────────────────────────────────

def _check_minimax_status() -> dict:
    """Check MiniMax API quota usage."""
    try:
        import urllib.request
        api_key_path = Path("/home/enderj/.openclaw/workspace/bittrader/keys/minimax.json")
        if api_key_path.exists():
            key_data = json.loads(api_key_path.read_text())
            token_key = key_data.get("minimax_token_plan_key", "")
        else:
            token_key = ""
        
        if not token_key:
            return {"available": False, "reason": "No token plan key"}
        
        # Check account info
        req = urllib.request.Request(
            "https://api.minimaxi.chat/v1/account",
            headers={"Authorization": f"Bearer {token_key}"}
        )
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read())
        
        return {
            "available": True,
            "data": data
        }
    except Exception as e:
        return {"available": False, "error": str(e)}

@app.route('/api/minimax-status')
def api_minimax_status():
    return jsonify(_check_minimax_status())

if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"🚀 BitTrader Production Dashboard")
    print(f"   URL: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
