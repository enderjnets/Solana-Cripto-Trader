#!/usr/bin/env python3
"""
BitTrader Production Dashboard — YouTube Stats + Video Production Monitor
Flask backend serving real-time data. Port: 8000
"""

import json
import os
import subprocess
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

# ── Data paths ────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/enderj/.openclaw/workspace")
DATA_DIR = WORKSPACE / "bittrader/agents/data"
MEMORY_DIR = WORKSPACE / "memory"
OUTPUT_DIR = WORKSPACE / "bittrader/agents/output"

def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}

# ── API endpoints ────────────────────────────────────────────────────────────

@app.route('/api/youtube')
def api_youtube():
    stats = load_json(DATA_DIR / "youtube_stats_latest.json")
    ch = stats.get("channel", {})
    subs = ch.get("subscribers", 0)
    views = ch.get("total_views", 0)
    videos = ch.get("total_videos", 0)

    # Estimate watch time: ~80% shorts (avg 15s watch), ~20% longs (avg 3min watch)
    # Weighted avg ≈ 30s per view (conservative)
    avg_view_sec = 30
    est_wt_hours = round((views * avg_view_sec) / 3600, 1)

    # Weekly watch time estimate (channel age ~6 months ≈ 26 weeks)
    channel_weeks = max(1, 26)
    weekly_wt = round(est_wt_hours / channel_weeks, 1)

    # Recent 48h views
    recent = stats.get("recent_videos", [])
    now = datetime.now(timezone.utc)
    views_48h = 0
    for v in recent:
        try:
            pub = datetime.fromisoformat(v.get("published_at", "2000").replace("Z", "+00:00"))
            if (now - pub).total_seconds() < 172800:
                views_48h += v.get("views", 0)
        except:
            pass

    # Recent videos with performance
    recent_list = []
    for v in recent[:15]:
        recent_list.append({
            "id": v.get("id", ""),
            "title": v.get("title", "?"),
            "views": v.get("views", 0),
            "likes": v.get("likes", 0),
            "comments": v.get("comments", 0),
            "duration": v.get("duration_seconds", 0),
            "published": v.get("published_at", ""),
        })

    return jsonify({
        "subs": subs,
        "views": views,
        "videos": videos,
        "watch_time_hours": est_wt_hours,
        "weekly_wt": weekly_wt,
        "views_48h": views_48h,
        "recent": recent_list,
    })

@app.route('/api/queue')
def api_queue():
    q = load_json(DATA_DIR / "upload_queue.json", [])
    total = len(q)
    ready = sum(1 for v in q if v.get("status") == "ready")
    blocked = sum(1 for v in q if v.get("status") == "blocked")
    uploaded = sum(1 for v in q if v.get("status") == "uploaded")
    qa_failed = sum(1 for v in q if v.get("status") == "qa_failed")
    items = []
    for v in q:
        if v.get("status") != "uploaded":
            items.append({
                "title": v.get("title", "?")[:60],
                "type": v.get("type", "?"),
                "status": v.get("status", "?"),
                "size_mb": v.get("size_mb", 0),
            })
    return jsonify({
        "total": total, "ready": ready, "blocked": blocked,
        "uploaded": uploaded, "qa_failed": qa_failed,
        "items": items[:20],
    })

@app.route('/api/scripts')
def api_scripts():
    data = load_json(DATA_DIR / "guiones_latest.json", {"scripts": []})
    scripts = data.get("scripts", [])
    pending = [s for s in scripts if s.get("status") == "pending"]
    produced = [s for s in scripts if s.get("status") == "produced"]
    items = []
    for s in pending[:8]:
        items.append({
            "title": s.get("title", "?")[:70],
            "type": s.get("type", "short"),
            "tags": s.get("tags", [])[:3],
        })
    return jsonify({
        "pending": len(pending),
        "produced": len(produced),
        "total": len(scripts),
        "items": items,
    })

@app.route('/api/mac-status')
def api_mac_status():
    macs = [
        {"id": "mbp", "name": "MacBook Pro M3 Pro",
         "url": "http://enders-macbook-pro.local:8188"},
        {"id": "mba", "name": "MacBook Air M4",
         "url": "http://100.118.215.73:8188"},
    ]
    result = []
    for mac in macs:
        info = _check_comfyui(mac["url"])
        result.append({**mac, **info})
    return jsonify(result)

def _check_comfyui(url, timeout=3):
    try:
        import urllib.request
        req = urllib.request.Request(f"{url}/system_stats")
        resp = urllib.request.urlopen(req, timeout=timeout)
        stats = json.loads(resp.read())
        qreq = urllib.request.Request(f"{url}/queue")
        qresp = urllib.request.urlopen(qreq, timeout=timeout)
        queue = json.loads(qresp.read())
        return {
            "online": True,
            "os": stats.get("system", {}).get("os", "?"),
            "version": stats.get("system", {}).get("comfyui_version", "?"),
            "running": len(queue.get("queue_running", [])),
            "queued": len(queue.get("queue_pending", [])),
            "gpu": (stats.get("devices", [{}])[0].get("name", "MPS")
                    if stats.get("devices") else "MPS"),
        }
    except Exception as e:
        return {"online": False, "error": str(e)[:80]}

@app.route('/api/producer-live')
def api_producer_live():
    try:
        proc = subprocess.run(["pgrep", "-af", "producer.py"],
                              capture_output=True, text=True, timeout=3)
        running = "producer.py" in proc.stdout
    except:
        running = False

    if not running:
        return jsonify({"running": False})

    log_path = Path("/tmp/producer_watch.log")
    if not log_path.exists():
        return jsonify({"running": True, "stage": "unknown"})

    try:
        lines = log_path.read_text().strip().split("\n")[-40:]
    except:
        lines = []

    current_script = ""
    stage = "processing"
    ltx_done = ltx_total = mini_done = mini_total = 0

    for line in reversed(lines):
        if "📝" in line and not current_script:
            current_script = line.split("]")[-1].strip() if "]" in line else line[-80:]
    for line in lines:
        if "LTX clip:" in line and "✅" in line:
            ltx_done += 1
        m = re.search(r"(\d+)\s+LTX", line)
        if m and "Hybrid" in line:
            ltx_total = int(m.group(1))
        m = re.search(r"(\d+)\s+MiniMax", line)
        if m and "Hybrid" in line:
            mini_total = int(m.group(1))
        if "MINIMAX:" in line:
            stage = "minimax"

    return jsonify({
        "running": True,
        "stage": stage,
        "script": current_script[:80],
        "ltx_done": ltx_done, "ltx_total": ltx_total,
        "mini_done": mini_done, "mini_total": mini_total,
    })

@app.route('/api/minimax-status')
def api_minimax_status():
    try:
        import urllib.request
        keys = load_json(WORKSPACE / "bittrader/keys/minimax.json")
        token = keys.get("minimax_token_plan_key", "")
        if not token:
            return jsonify({"ok": False, "error": "No API key"})
        # Simple health check — try the image generation endpoint with minimal data
        req = urllib.request.Request(
            "https://api.minimax.io/v1/image_generation",
            data=json.dumps({"model": "image-01", "prompt": "test", "aspect_ratio": "1:1"}).encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        resp = urllib.request.urlopen(req, timeout=8)
        data = json.loads(resp.read())
        code = data.get("base_resp", {}).get("status_code", -1)
        if code == 0:
            return jsonify({"ok": True, "msg": "API activa (image-01)"})
        else:
            msg = data.get("base_resp", {}).get("status_msg", "unknown")
            return jsonify({"ok": True, "msg": f"API responde: {msg}"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:100]})

@app.route('/api/produced-today')
def api_produced_today():
    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = OUTPUT_DIR / today
    if not out_dir.exists():
        return jsonify({"count": 0, "videos": []})
    videos = []
    for d in sorted(out_dir.iterdir()):
        if not d.is_dir():
            continue
        mp4 = d / f"{d.name}.mp4"
        if mp4.exists():
            sz = round(mp4.stat().st_size / (1024 * 1024), 1)
            videos.append({"id": d.name, "size_mb": sz})
    return jsonify({"count": len(videos), "videos": videos[:20]})


# ── HTML Template ────────────────────────────────────────────────────────────

HTML = r'''<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>📊 BitTrader Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0d1117;color:#e6edf3;min-height:100vh}
.c{max-width:1400px;margin:0 auto;padding:16px}
header{display:flex;justify-content:space-between;align-items:center;padding:12px 0;border-bottom:1px solid #21262d;margin-bottom:16px}
header h1{font-size:20px;color:#58a6ff}
.btn{background:#238636;color:#fff;border:none;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px}
.btn:hover{background:#2ea043}
.ts{font-size:11px;color:#484f58}

.sec{margin-bottom:24px}
.sec-t{font-size:13px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;border-left:3px solid #58a6ff;padding-left:8px}
.sec-t.green{border-color:#3fb950}
.sec-t.yellow{border-color:#d29922}
.sec-t.red{border-color:#f85149}

.g{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}
.g3{grid-template-columns:repeat(auto-fit,minmax(160px,1fr))}
.cd{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px}
.cd .lb{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}
.cd .vl{font-size:26px;font-weight:700}
.cd .sb{font-size:11px;color:#8b949e;margin-top:3px}

.pb{background:#21262d;border-radius:4px;height:8px;margin-top:6px;overflow:hidden}
.pf{height:100%;border-radius:4px;transition:width .5s}

.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:900px){.two{grid-template-columns:1fr}}

.crd{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px;margin-bottom:10px}
.crd-t{font-size:12px;color:#8b949e;margin-bottom:8px}

.si{background:#161b22;border:1px solid #21262d;border-radius:6px;padding:10px;margin-bottom:6px}
.si-t{font-size:12px;font-weight:600}
.si-m{font-size:10px;color:#8b949e;margin-top:2px}
.badge{display:inline-block;padding:1px 6px;border-radius:4px;font-size:10px;margin-right:4px}
.b-long{background:#1f6feb33;color:#58a6ff;border:1px solid #1f6feb}
.b-short{background:#23863633;color:#3fb950;border:1px solid #238636}

.qi{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid #21262d}
.qi:last-child{border:none}
.qi-l{flex:1}
.qi-t{font-size:12px}
.qi-m{font-size:10px;color:#8b949e}
.sb-tag{padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600}
.s-ready{background:#238636;color:#fff}
.s-blocked{background:#b62324;color:#fff}
.s-uploaded{background:#1f6feb;color:#fff}
.s-pending{background:#9e6a03;color:#fff}
.s-qa{background:#6e40c9;color:#fff}

.vi{display:flex;gap:10px;padding:8px 0;border-bottom:1px solid #21262d;font-size:12px}
.vi:last-child{border:none}
.vi-t{flex:1;color:#58a6ff}
.vi-s{color:#8b949e;width:60px;text-align:right}

.green{color:#3fb950}.red{color:#f85149}.blue{color:#58a6ff}.yellow{color:#d29922}.muted{color:#8b949e}
.mt{margin-top:6px}.mb{margin-bottom:6px}

/* Monetization highlight */
.mono{background:linear-gradient(135deg,#161b22,#1a2332);border:2px solid #1f6feb;border-radius:12px;padding:20px;margin-bottom:16px}
.mono h2{font-size:18px;color:#58a6ff;margin-bottom:12px}
.mono .row{display:flex;gap:20px;flex-wrap:wrap}
.mono .col{flex:1;min-width:180px}
.mono .big{font-size:36px;font-weight:800}
.mono .bar{height:12px;background:#21262d;border-radius:6px;margin-top:6px;overflow:hidden}
.mono .fill{height:100%;border-radius:6px}
</style>
</head>
<body>
<div class="c">
<header>
  <h1>📊 BitTrader Production Dashboard</h1>
  <div>
    <span class="ts" id="ts">—</span>
    <button class="btn" onclick="loadAll()">🔄 Refresh</button>
  </div>
</header>

<!-- ═══ MONETIZATION GOAL ═══ -->
<div class="mono" id="mono-box">
  <h2>🎯 Meta: Monetización YouTube (YPP)</h2>
  <div class="row">
    <div class="col">
      <div class="lb" style="font-size:11px;color:#8b949e">SUSCRIPTORES</div>
      <div class="big green" id="m-subs">—</div>
      <div class="sb" style="font-size:12px;color:#8b949e">de 1,000 necesarios</div>
      <div class="bar"><div class="fill" id="m-subs-bar" style="width:0%;background:linear-gradient(90deg,#238636,#3fb950)"></div></div>
      <div class="mt" style="font-size:11px;color:#3fb950" id="m-subs-status">—</div>
    </div>
    <div class="col">
      <div class="lb" style="font-size:11px;color:#8b949e">WATCH TIME</div>
      <div class="big yellow" id="m-wt">—</div>
      <div class="sb" style="font-size:12px;color:#8b949e">de 4,000h necesarias</div>
      <div class="bar"><div class="fill" id="m-wt-bar" style="width:0%;background:linear-gradient(90deg,#9e6a03,#d29922)"></div></div>
      <div class="mt" style="font-size:11px;color:#d29922" id="m-wt-status">—</div>
    </div>
    <div class="col">
      <div class="lb" style="font-size:11px;color:#8b949e">PROGRESO TOTAL</div>
      <div class="big blue" id="m-pct">—</div>
      <div class="sb" style="font-size:12px;color:#8b949e" id="m-eta">Estimando...</div>
      <div class="bar"><div class="fill" id="m-pct-bar" style="width:0%;background:linear-gradient(90deg,#1f6feb,#58a6ff)"></div></div>
      <div class="mt" style="font-size:11px;color:#58a6ff" id="m-speed">—</div>
    </div>
  </div>
</div>

<!-- ═══ YOUTUBE STATS ═══ -->
<div class="sec">
  <div class="sec-t">📺 YouTube — BitTrader (@bittrader9259)</div>
  <div class="g">
    <div class="cd"><div class="lb">Suscriptores</div><div class="vl green" id="yt-subs">—</div><div class="sb" id="yt-subs-sub">meta 1,000 ✅</div></div>
    <div class="cd"><div class="lb">Vistas Totales</div><div class="vl" id="yt-views">—</div><div class="sb" id="yt-views-48">—</div></div>
    <div class="cd"><div class="lb">Videos Publicados</div><div class="vl" id="yt-videos">—</div><div class="sb">en el canal</div></div>
    <div class="cd"><div class="lb">Watch Time (est.)</div><div class="vl yellow" id="yt-wt">—</div><div class="sb">de 4,000h para YPP</div>
      <div class="pb"><div class="pf" id="yt-wt-bar" style="width:0%;background:linear-gradient(90deg,#9e6a03,#d29922)"></div></div>
    </div>
    <div class="cd"><div class="lb">WH/Semana</div><div class="vl yellow" id="yt-speed">—</div><div class="sb">velocidad actual</div></div>
    <div class="cd"><div class="lb">Meta Monetización</div><div class="vl blue" id="yt-meta">—</div><div class="sb" id="yt-meta-sub">estimado</div></div>
  </div>
</div>

<!-- ═══ MAC POOL + LIVE ═══ -->
<div class="sec">
  <div class="sec-t green">💻 Mac Pool + Producción en Vivo</div>
  <div class="g">
    <div class="cd" id="mac-mbp"><div class="lb">🍎 MacBook Pro M3 Pro</div><div class="vl" id="mbp-st">...</div><div class="sb" id="mbp-sb">verificando...</div><div class="pb"><div class="pf" id="mbp-bar" style="width:0%"></div></div><div class="mt muted" style="font-size:10px" id="mbp-dt"></div></div>
    <div class="cd" id="mac-mba"><div class="lb">🍏 MacBook Air M4</div><div class="vl" id="mba-st">...</div><div class="sb" id="mba-sb">verificando...</div><div class="pb"><div class="pf" id="mba-bar" style="width:0%"></div></div><div class="mt muted" style="font-size:10px" id="mba-dt"></div></div>
    <div class="cd"><div class="lb">🎬 Producer</div><div class="vl" id="prod-st">...</div><div class="sb" id="prod-sb">verificando...</div><div class="pb"><div class="pf" id="prod-bar" style="width:0%;background:linear-gradient(90deg,#1f6feb,#58a6ff)"></div></div><div class="mt muted" style="font-size:10px" id="prod-dt"></div></div>
    <div class="cd"><div class="lb">🤖 MiniMax API</div><div class="vl" id="mm-st">...</div><div class="sb" id="mm-sb">verificando...</div><div class="pb"><div class="pf" id="mm-bar" style="width:0%"></div></div></div>
  </div>
</div>

<!-- ═══ PIPELINE + QUEUE ═══ -->
<div class="two">
  <div>
    <div class="sec-t yellow">🎬 Pipeline de Producción</div>
    <div class="g g3">
      <div class="cd"><div class="lb">Scripts Pendientes</div><div class="vl yellow" id="sc-pend">—</div></div>
      <div class="cd"><div class="lb">Producidos (hoy)</div><div class="vl green" id="sc-prod">—</div></div>
      <div class="cd"><div class="lb">En Cola Upload</div><div class="vl blue" id="q-total">—</div></div>
      <div class="cd"><div class="lb">Listos para Subir</div><div class="vl green" id="q-ready">—</div></div>
    </div>
    <div class="crd mt"><div class="crd-t">🔄 Producción Activa</div><div id="prod-info" class="muted" style="font-size:12px">Cargando...</div></div>
    <div class="crd"><div class="crd-t">📋 Próximos Scripts</div><div id="sc-list">Cargando...</div></div>
  </div>
  <div>
    <div class="sec-t red">📤 Cola de Upload</div>
    <div class="g g3 mb">
      <div class="cd"><div class="lb">Blockeados</div><div class="vl red" id="q-blocked">—</div></div>
      <div class="cd"><div class="lb">QA Failed</div><div class="vl" style="color:#6e40c9" id="q-qa">—</div></div>
      <div class="cd"><div class="lb">Subidos</div><div class="vl blue" id="q-uploaded">—</div></div>
    </div>
    <div class="crd"><div class="crd-t">Items pendientes</div><div id="q-list" class="muted" style="font-size:12px">Cargando...</div></div>
  </div>
</div>

<!-- ═══ RECENT VIDEOS ═══ -->
<div class="sec">
  <div class="sec-t">🎥 Videos Recientes</div>
  <div class="crd"><div id="rv-list">Cargando...</div></div>
</div>

</div>

<script>
function fmt(n){if(n>=1e6)return(n/1e6).toFixed(1)+'M';if(n>=1e3)return(n/1e3).toFixed(1)+'K';return String(n)}
function fmtH(h){return h>=1000?fmt(h):h.toFixed(0)+'h'}

async function loadYT(){
  try{
    const d=await(await fetch('/api/youtube')).json();
    const s=d.subs,v=d.views,vid=d.videos,wt=d.watch_time_hours,wk=d.weekly_wt;

    // YouTube cards
    document.getElementById('yt-subs').textContent=fmt(s);
    document.getElementById('yt-views').textContent=fmt(v);
    document.getElementById('yt-views-48').textContent=d.views_48h?`+${fmt(d.views_48h)} en 48h`:'—';
    document.getElementById('yt-videos').textContent=vid;
    document.getElementById('yt-wt').textContent=fmtH(wt);
    const wtPct=Math.min(100,(wt/4000)*100);
    document.getElementById('yt-wt-bar').style.width=wtPct+'%';
    document.getElementById('yt-speed').textContent=wk.toFixed(1)+'h/sem';
    
    const needed=Math.max(0,4000-wt);
    const weeks=wk>0?needed/wk:999;
    const totalPct=((s>=1000?50:s/1000*50)+(wt>=4000?50:wt/4000*50)).toFixed(1);
    document.getElementById('yt-meta').textContent=totalPct+'%';
    document.getElementById('yt-meta-sub').textContent=weeks<52?`~${Math.ceil(weeks)} sem para meta`:`~${(weeks/52).toFixed(1)} años`;

    // Monetization box
    document.getElementById('m-subs').textContent=fmt(s);
    const subsPct=Math.min(100,s/1000*100);
    document.getElementById('m-subs-bar').style.width=subsPct+'%';
    document.getElementById('m-subs-status').textContent=s>=1000?'✅ Meta alcanzada':'Faltan '+(1000-s);

    document.getElementById('m-wt').textContent=fmtH(wt);
    document.getElementById('m-wt-bar').style.width=wtPct+'%';
    document.getElementById('m-wt-status').textContent=wt>=4000?'✅ Meta alcanzada':`Faltan ${fmtH(needed)}`;

    document.getElementById('m-pct').textContent=totalPct+'%';
    document.getElementById('m-pct-bar').style.width=totalPct+'%';
    document.getElementById('m-eta').textContent=weeks<52?`~${Math.ceil(weeks)} semanas restantes`:`~${(weeks/52).toFixed(1)} años restantes`;
    document.getElementById('m-speed').textContent=`Velocidad: ${wk.toFixed(1)}h/sem`;

    // Recent videos
    let rvH='';
    for(const r of d.recent){
      const dur=r.duration<120?(r.duration+'s'):(Math.round(r.duration/60)+'min');
      rvH+=`<div class="vi"><div class="vi-t">${r.title}</div><div class="vi-s">${fmt(r.views)} 👁</div><div class="vi-s">${dur}</div></div>`;
    }
    document.getElementById('rv-list').innerHTML=rvH||'<span class="muted">Sin datos</span>';
  }catch(e){console.error('YT error:',e)}
}

async function loadQueue(){
  try{
    const d=await(await fetch('/api/queue')).json();
    document.getElementById('q-total').textContent=d.total;
    document.getElementById('q-ready').textContent=d.ready;
    document.getElementById('q-blocked').textContent=d.blocked;
    document.getElementById('q-qa').textContent=d.qa_failed;
    document.getElementById('q-uploaded').textContent=d.uploaded;
    let h='';
    for(const i of d.items){
      const cls=i.status==='ready'?'s-ready':i.status==='blocked'?'s-blocked':i.status==='qa_failed'?'s-qa':'s-pending';
      h+=`<div class="qi"><div class="qi-l"><div class="qi-t">${i.title}</div><div class="qi-m">${i.type} | ${i.size_mb||0}MB</div></div><span class="sb-tag ${cls}">${i.status}</span></div>`;
    }
    document.getElementById('q-list').innerHTML=h||'<span class="muted">Cola vacía — todos subidos ✅</span>';
  }catch(e){console.error('Queue error:',e)}
}

async function loadScripts(){
  try{
    const d=await(await fetch('/api/scripts')).json();
    document.getElementById('sc-pend').textContent=d.pending;
    document.getElementById('sc-prod').textContent=d.produced;
    let h='';
    for(const s of d.items){
      const bc=s.type==='long'?'b-long':'b-short';
      const tags=s.tags.map(t=>`<span class="badge" style="background:#21262d;color:#8b949e">${t}</span>`).join('');
      h+=`<div class="si"><div class="si-t">${s.title}</div><div class="si-m"><span class="badge ${bc}">${s.type.toUpperCase()}</span>${tags}</div></div>`;
    }
    document.getElementById('sc-list').innerHTML=h||'<span class="muted">Sin scripts pendientes</span>';
  }catch(e){console.error('Scripts error:',e)}
}

async function loadProduced(){
  try{
    const d=await(await fetch('/api/produced-today')).json();
    document.getElementById('sc-prod').textContent=d.count;
  }catch(e){}
}

async function loadMacs(){
  try{
    const ms=await(await fetch('/api/mac-status')).json();
    for(const m of ms){
      const pre=m.id==='mbp'?'mbp':'mba';
      const st=document.getElementById(pre+'-st');
      const sb=document.getElementById(pre+'-sb');
      const bar=document.getElementById(pre+'-bar');
      const dt=document.getElementById(pre+'-dt');
      if(!m.online){
        st.textContent='⚠️ Offline';st.style.color='#f85149';
        sb.textContent='ComfyUI no responde';bar.style.width='0%';dt.textContent='';
      }else{
        const busy=m.running+m.queued;
        st.textContent=busy>0?'🔥 Generando':'✅ Idle';
        st.style.color=busy>0?'#d29922':'#3fb950';
        sb.textContent=`ComfyUI ${m.version} | ${m.os}`;
        bar.style.width=busy>0?'70%':'100%';
        bar.style.background=busy>0?'linear-gradient(90deg,#9e6a03,#d29922)':'linear-gradient(90deg,#238636,#3fb950)';
        dt.textContent=m.gpu||'MPS';
      }
    }
  }catch(e){console.error('Mac error:',e)}
}

async function loadProducer(){
  try{
    const d=await(await fetch('/api/producer-live')).json();
    const st=document.getElementById('prod-st');
    const sb=document.getElementById('prod-sb');
    const bar=document.getElementById('prod-bar');
    const dt=document.getElementById('prod-dt');
    if(!d.running){
      st.textContent='⏸️ Idle';st.style.color='#8b949e';
      sb.textContent='Producer en espera';bar.style.width='0%';dt.textContent='';
      document.getElementById('prod-info').textContent='⏸️ Producer no está corriendo';
    }else{
      st.textContent='⚙️ Activo';st.style.color='#58a6ff';
      sb.textContent=d.script||'Procesando...';
      const tot=d.ltx_total+d.mini_total;
      const done=d.ltx_done+d.mini_done;
      const pct=tot>0?Math.round(done/tot*100):50;
      bar.style.width=pct+'%';
      dt.textContent=d.ltx_total>0?`LTX:${d.ltx_done}/${d.ltx_total} MiniMax:${d.mini_done}/${d.mini_total}`:'';
      document.getElementById('prod-info').innerHTML=`<strong style="color:#58a6ff">⚙️ Produciendo:</strong> ${d.script||'?'}`;
    }
  }catch(e){console.error('Producer error:',e)}
}

async function loadMiniMax(){
  try{
    const d=await(await fetch('/api/minimax-status')).json();
    const st=document.getElementById('mm-st');
    const sb=document.getElementById('mm-sb');
    const bar=document.getElementById('mm-bar');
    if(d.ok){
      st.textContent='✅ OK';st.style.color='#3fb950';
      sb.textContent=d.msg||'API activa';
      bar.style.width='100%';bar.style.background='linear-gradient(90deg,#238636,#3fb950)';
    }else{
      st.textContent='⚠️ Error';st.style.color='#f85149';
      sb.textContent=d.error||'API no responde';
      bar.style.width='0%';
    }
  }catch(e){
    document.getElementById('mm-st').textContent='⚠️ Error';
    document.getElementById('mm-st').style.color='#f85149';
    document.getElementById('mm-sb').textContent='No se pudo verificar';
  }
}

async function loadAll(){
  document.getElementById('ts').textContent=new Date().toLocaleString('es-MX',{timeZone:'America/Denver'});
  await Promise.all([loadYT(),loadQueue(),loadScripts(),loadProduced(),loadMacs(),loadProducer(),loadMiniMax()]);
}

loadAll();
setInterval(loadAll,60000);
setInterval(loadMacs,10000);
setInterval(loadProducer,15000);
</script>
</body>
</html>'''

@app.route('/')
def index():
    return render_template_string(HTML)

if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"🚀 BitTrader Production Dashboard on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
