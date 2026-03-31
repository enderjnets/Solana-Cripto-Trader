#!/usr/bin/env python3
"""
BitTrader Production Dashboard — YouTube Stats + Video Production Monitor
Flask backend serving real-time data. Port: 8000
"""

import json
import os
import subprocess
import re
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from flask import Flask, jsonify, render_template_string, request, Response

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

# ── Agent notes ──────────────────────────────────────────────────────────────
AGENT_NOTES_FILE = DATA_DIR / "agent_notes.json"

def load_notes():
    try:
        with open(AGENT_NOTES_FILE) as f:
            return json.load(f)
    except:
        return {"messages": [], "last_updated": None}

def save_notes(data):
    data["last_updated"] = datetime.now().isoformat()
    with open(AGENT_NOTES_FILE, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    # Notify SSE clients of new data
    _notify_sse()

# ── SSE streaming ────────────────────────────────────────────────────────────
sse_clients = set()
_sse_lock = threading.Lock()

def _notify_sse():
    """Push current notes state to all connected SSE clients."""
    global sse_clients
    try:
        notes = load_notes()
        data = json.dumps({"type": "update", "notes": notes}, ensure_ascii=False)
        dead = set()
        for client in sse_clients:
            try:
                client.put(data)
            except:
                dead.add(client)
        if dead:
            with _sse_lock:
                sse_clients -= dead
    except Exception as e:
        print(f"SSE notify error: {e}")

class SSEClient:
    """Simple queue-based SSE client."""
    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()
    def put(self, data):
        with self.lock:
            self.queue.append(data)
    def get(self, timeout=50):
        while True:
            time.sleep(1)
            with self.lock:
                if self.queue:
                    return self.queue.pop(0)

def _notes_watcher():
    """Background thread: monitors notes file and triggers agent when Ender writes.
    
    Anti-loop protections:
    1. Only checks the LAST user message (not all messages)
    2. Tracks triggered message ts to avoid re-triggering
    3. Ignores file changes caused by agent writes (cooldown after trigger)
    4. Rate limits: max 1 trigger per 60 seconds
    """
    last_mtime = AGENT_NOTES_FILE.stat().st_mtime if AGENT_NOTES_FILE.exists() else 0
    last_size = AGENT_NOTES_FILE.stat().st_size if AGENT_NOTES_FILE.exists() else 0
    last_triggered_ts = None  # Track last triggered message timestamp
    last_trigger_time = 0     # Track when we last triggered (for cooldown)
    COOLDOWN_SECONDS = 60     # Min seconds between triggers
    
    while True:
        try:
            if AGENT_NOTES_FILE.exists():
                mtime = AGENT_NOTES_FILE.stat().st_mtime
                size = AGENT_NOTES_FILE.stat().st_size
                if mtime != last_mtime or size != last_size:
                    last_mtime = mtime
                    last_size = size
                    
                    # Cooldown check — ignore changes within cooldown period
                    now = time.time()
                    if now - last_trigger_time < COOLDOWN_SECONDS:
                        continue
                    
                    # Only look at the LAST message in the file
                    notes = load_notes()
                    msgs = notes.get("messages", [])
                    if not msgs:
                        continue
                    
                    last_msg = msgs[-1]
                    
                    # Only trigger for USER messages (ignore agent writes)
                    if last_msg.get("sender") != "user":
                        continue
                    
                    ts = last_msg.get("ts", "")
                    
                    # Skip if already triggered for this exact message
                    if ts == last_triggered_ts:
                        continue
                    
                    # Only trigger if recent (< 5 min old)
                    try:
                        msg_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        if msg_time.tzinfo is None:
                            from zoneinfo import ZoneInfo
                            msg_time = msg_time.replace(tzinfo=ZoneInfo("America/Denver"))
                        age = (datetime.now(timezone.utc) - msg_time).total_seconds()
                        if age < 300:
                            last_triggered_ts = ts
                            last_trigger_time = now
                            print(f"NEW MESSAGE from Ender detected: {last_msg['text'][:60]}")
                            _trigger_agent_now(last_msg)
                    except Exception as e:
                        print(f"Watcher time parse error: {e}")
        except Exception as e:
            print(f"Watcher error: {e}")
        time.sleep(3)


def _trigger_agent_now(user_msg):
    """Trigger real-time response via local LLM.
    
    Does NOT write to agent_notes.json (avoids re-triggering the watcher).
    The responder itself handles writing and has its own lockfile.
    """
    try:
        msg_text = user_msg.get("text", "")[:500]

        # Spawn real-time responder (has PID lockfile + rate limit built in)
        import subprocess
        subprocess.Popen(
            ["python3", str(Path(__file__).parent / "realtime_responder.py")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"Agent triggered (real-time): {msg_text[:80]}")
    except Exception as e:
        print(f"Trigger error: {e}")

# Start the background watcher thread
_watcher_thread = threading.Thread(target=_notes_watcher, daemon=True)
_watcher_thread.start()

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
        sched = v.get("scheduled_date", "sin fecha")
        try:
            from datetime import datetime, timezone, timedelta
            dt = datetime.fromisoformat(sched.replace("Z","+00:00"))
            sched = dt.astimezone(timezone(timedelta(hours=-6))).strftime("%d/%m %I:%M %p MT")
        except:
            pass
        items.append({
            "title": v.get("title", "?")[:70],
            "type": v.get("type", "?"),
            "status": v.get("status", "?"),
            "size_mb": v.get("size_mb", 0),
            "scheduled": sched,
            "script_id": v.get("script_id", ""),
            "privacy": v.get("privacy", "public"),
        })
    return jsonify({
        "total": total, "ready": ready, "blocked": blocked,
        "uploaded": uploaded, "qa_failed": qa_failed,
        "items": items,
    })

@app.route('/api/scripts')
def api_scripts():
    data = load_json(DATA_DIR / "guiones_latest.json", {"scripts": []})
    scripts = data.get("scripts", [])
    pending = [s for s in scripts if s.get("status") == "pending"]
    produced = [s for s in scripts if s.get("status") == "produced"]
    items = []
    for s in pending[:20]:
        items.append({
            "title": s.get("title", "?")[:70],
            "type": s.get("type", "short"),
            "tags": s.get("tags", [])[:5],
            "script_id": s.get("script_id", ""),
            "created_at": s.get("created_at", ""),
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
        proc = subprocess.run(["pgrep", "-af", "producer|orchestrator.*produce"],
                              capture_output=True, text=True, timeout=3)
        running = "producer.py" in proc.stdout or "produce" in proc.stdout
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

    # Load guiones for titles/types (has today's produced scripts)
    guiones = load_json(DATA_DIR / "guiones_latest.json", {})
    guiones_by_id = {s.get("id", ""): s for s in guiones.get("scripts", [])}

    # Load queue for scheduled_date
    queue = load_json(DATA_DIR / "upload_queue.json", [])
    # Normalize queue paths: strip leading "output/" or "agents/data/output/"
    queue_by_id = {}
    for v in queue:
        script_id = v.get("script_id", v.get("id", ""))
        if script_id:
            queue_by_id[script_id] = v

    for d in sorted(out_dir.iterdir()):
        if not d.is_dir():
            continue
        mp4 = d / f"{d.name}.mp4"
        if mp4.exists():
            sz = round(mp4.stat().st_size / (1024 * 1024), 1)

            # Get duration
            dur = None
            try:
                r = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-print_format", "json",
                     "-show_streams", str(mp4)],
                    capture_output=True, text=True, timeout=5
                )
                streams = json.loads(r.stdout).get("streams", [])
                for s in streams:
                    if s.get("codec_type") == "video":
                        dur = float(s.get("duration", 0))
                        break
            except Exception:
                pass

            dur_str = f"{int(dur//60)}m {int(dur%60)}s" if dur else "?"

            # Look up by script_id (d.name = script_id from producer)
            guion = guiones_by_id.get(d.name, {})
            q_entry = queue_by_id.get(d.name, {})
            videos.append({
                "id": d.name,
                "size_mb": sz,
                "duration": dur_str,
                "title": guion.get("title", "?")[:70] if guion else "?",
                "type": guion.get("type", "?") if guion else "?",
                "scheduled": q_entry.get("scheduled_date", ""),
            })

    return jsonify({"count": len(videos), "videos": videos})

@app.route('/api/qa-status')
def api_qa_status():
    """QA Agent status: recent checks, alerts, and pending escalations."""
    qa_log = load_json(DATA_DIR / "qa_log.json", [])
    qa_alerts = load_json(DATA_DIR / "qa_alerts_for_ceo.json", [])

    # Last 10 QA checks
    recent = qa_log[-10:] if isinstance(qa_log, list) else []
    passed = sum(1 for r in recent if r.get("passed"))
    failed = sum(1 for r in recent if not r.get("passed"))

    # Pending CEO escalations
    pending_alerts = [a for a in qa_alerts if a.get("status") == "pending"] if isinstance(qa_alerts, list) else []

    return jsonify({
        "recent_checks": len(recent),
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{(passed/(passed+failed)*100):.0f}%" if (passed+failed) > 0 else "N/A",
        "pending_escalations": len(pending_alerts),
        "alerts": [
            {
                "title": a.get("title", "?")[:60],
                "issues": a.get("issues", []),
                "timestamp": a.get("timestamp", ""),
                "action": a.get("action_required", ""),
            }
            for a in pending_alerts[-5:]
        ],
        "recent": [
            {
                "title": r.get("title", "?")[:50],
                "passed": r.get("passed", False),
                "issues": r.get("issues", []),
                "timestamp": r.get("timestamp", ""),
            }
            for r in recent[-5:]
        ],
    })


@app.route('/api/pipeline-status')
def api_pipeline_status():
    """Current pipeline activity: what's running, what's pending."""
    # Check running processes
    try:
        proc = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=3)
        ps_out = proc.stdout
    except:
        ps_out = ""

    producer_running = "producer.py" in ps_out or ("orchestrator" in ps_out and "produce" in ps_out)
    creator_running = "creator.py" in ps_out
    publisher_running = "publisher.py" in ps_out or ("orchestrator" in ps_out and "publish" in ps_out)
    scout_running = "scout" in ps_out and "orchestrator" in ps_out

    # Guiones status
    guiones = load_json(DATA_DIR / "guiones_latest.json", {})
    scripts = guiones.get("scripts", []) if isinstance(guiones, dict) else guiones
    pending = [s for s in scripts if s.get("status") == "pending"]
    produced = [s for s in scripts if s.get("status") in ("produced", "success")]

    # Queue status
    queue = load_json(DATA_DIR / "upload_queue.json", [])
    ready = [v for v in queue if v.get("status") == "ready"]
    qa_failed = [v for v in queue if v.get("status") == "qa_failed"]
    thumb_failed = [v for v in queue if v.get("status") == "thumbnail_failed"]

    # Today's output
    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = OUTPUT_DIR / today
    today_count = len(list(out_dir.iterdir())) if out_dir.exists() else 0

    return jsonify({
        "processes": {
            "producer": producer_running,
            "creator": creator_running,
            "publisher": publisher_running,
            "scout": scout_running,
        },
        "scripts": {
            "pending": len(pending),
            "produced": len(produced),
            "total": len(scripts),
        },
        "queue": {
            "ready_to_upload": len(ready),
            "qa_failed": len(qa_failed),
            "thumbnail_failed": len(thumb_failed),
        },
        "today_produced": today_count,
    })


@app.route('/api/agent-notes', methods=['GET', 'POST'])
def api_agent_notes():
    """GET: return all notes. POST: add a message (sender=user|agent)."""
    if request.method == "POST":
        data = request.get_json() or {}
        msg = data.get("message", "").strip()
        sender = data.get("sender", "user")
        if msg:
            notes = load_notes()
            notes["messages"].append({
                "sender": sender,
                "text": msg,
                "ts": datetime.now().isoformat(),
            })
            save_notes(notes)
        return jsonify({"ok": True})
    return jsonify(load_notes())

@app.route('/api/agent-notes/last')
def api_agent_last_note():
    """GET: return only the last note from the agent (for polling)."""
    notes = load_notes()
    agent_msgs = [m for m in notes.get("messages", []) if m.get("sender") == "agent"]
    if agent_msgs:
        return jsonify(agent_msgs[-1])
    return jsonify({"text": None})

@app.route('/api/agent-notes/stream')
def api_agent_notes_stream():
    """
    SSE stream — pushes note updates to the browser in real time.
    Browser connects once, receives instant pushes on every new note.
    Also immediately triggers CEO agent when Ender sends a message.
    """
    def generate():
        client = SSEClient()
        with _sse_lock:
            sse_clients.add(client)
        # Send initial state immediately
        notes = load_notes()
        yield f"event: init\ndata: {json.dumps({'type': 'init', 'notes': notes}, ensure_ascii=False)}\n\n"
        last_count = len(notes.get("messages", []))
        last_mtime = AGENT_NOTES_FILE.stat().st_mtime if AGENT_NOTES_FILE.exists() else 0
        try:
            while True:
                time.sleep(1)
                # Detect file changes by mtime + message count
                cur_mtime = AGENT_NOTES_FILE.stat().st_mtime if AGENT_NOTES_FILE.exists() else 0
                if cur_mtime != last_mtime:
                    last_mtime = cur_mtime
                    notes = load_notes()
                    msgs = notes.get("messages", [])
                    if len(msgs) != last_count:
                        last_count = len(msgs)
                        yield f"event: update\ndata: {json.dumps({'type': 'update', 'notes': notes}, ensure_ascii=False)}\n\n"
                # Send keepalive every 15s to prevent browser timeout
                yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            with _sse_lock:
                sse_clients.discard(client)
    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no',
                             'Connection': 'keep-alive'})


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

/* ── Agent Notes Chat ── */
.chat-box{background:#161b22;border:1px solid #21262d;border-radius:10px;overflow:hidden}
.chat-agent-header{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:#1c2128;border-bottom:1px solid #21262d;font-size:12px;color:#8b949e}
.chat-badge{background:#238636;color:#fff;padding:1px 8px;border-radius:10px;font-size:10px}
.an-empty{color:#484f58;font-size:12px;text-align:center;padding:20px}
.chat-messages{max-height:320px;overflow-y:auto;padding:10px}
.msg{margin-bottom:10px;display:flex;flex-direction:column}
.msg-agent .msg-bubble{background:#1f2d3d;border:1px solid #2d4a6e;color:#c9d1d9;align-self:flex-start;max-width:85%;border-radius:12px 12px 12px 2px}
.msg-user .msg-bubble{background:#1a3d1a;border:1px solid #2d6e2d;color:#c9d1d9;align-self:flex-end;max-width:85%;border-radius:12px 12px 2px 12px;text-align:right}
.msg-agent .msg-meta{color:#58a6ff;font-size:10px;margin-bottom:3px}
.msg-user .msg-meta{color:#3fb950;font-size:10px;margin-bottom:3px;text-align:right}
.msg-bubble{padding:8px 12px;font-size:12px;line-height:1.5}
.typing-indicator{display:flex;align-items:center;gap:4px;padding:10px 14px}
.typing-indicator .dot{width:8px;height:8px;border-radius:50%;background:#58a6ff;animation:typingBounce 1.4s infinite ease-in-out both}
.typing-indicator .dot:nth-child(1){animation-delay:0s}
.typing-indicator .dot:nth-child(2){animation-delay:.2s}
.typing-indicator .dot:nth-child(3){animation-delay:.4s}
@keyframes typingBounce{0%,80%,100%{transform:scale(0.4);opacity:.4}40%{transform:scale(1);opacity:1}}
.thinking-wrap{display:none;padding:6px 14px}
.thinking-wrap.active{display:flex;align-items:flex-start;flex-direction:column}
.thinking-label{color:#58a6ff;font-size:10px;margin-bottom:4px}
.thinking-bubble{background:#1f2d3d;border:1px solid #2d4a6e;border-radius:12px 12px 12px 2px;padding:10px 16px;display:flex;align-items:center;gap:8px}
.thinking-text{color:#8b949e;font-size:12px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:.5}50%{opacity:1}}
.chat-input-row{display:flex;gap:8px;padding:10px;border-top:1px solid #21262d;background:#1c2128}
.chat-input-row input{flex:1;background:#0d1117;border:1px solid #30363d;color:#e6edf3;padding:8px 12px;border-radius:6px;font-size:12px;outline:none}
.chat-input-row input:focus{border-color:#58a6ff}
.chat-tip{font-size:10px;color:#484f58;padding:4px 14px 8px;background:#1c2128}
.exp-list{margin-top:8px;border-radius:8px;overflow:hidden;background:#161b22;border:1px solid #21262d}
.exp-list .vl-item{padding:6px 10px;border-bottom:1px solid #21262d;font-size:11px;display:flex;justify-content:space-between;align-items:center;gap:8px}
.exp-list .vl-item:last-child{border-bottom:none}
.exp-list .vl-item:hover{background:#1c2128}
.exp-list .vl-title{flex:1;color:#e6edf3;font-weight:500;word-break:break-word;min-width:0}
.exp-list .vl-meta{display:flex;gap:6px;flex-shrink:0;color:#8b949e;font-size:10px;flex-wrap:wrap;justify-content:flex-end}
.exp-list .vl-tag{padding:1px 6px;border-radius:4px;font-size:10px}
.exp-list .vl-tag.short{background:#2f4a1e;color:#85d485}
.exp-list .vl-tag.long{background:#1e3a5f;color:#79c0ff}
.exp-list .vl-tag.ready{background:#1a3a1a;color:#56d364}
.exp-list .vl-tag.blocked{background:#3a1a1a;color:#f85149}
.exp-list .vl-tag.qa{background:#2a1a3a;color:#a371f7}
.exp-list .vl-tag.uploaded{background:#161b22;color:#8b949e}
.exp-list .vl-sched{color:#bc8cff;font-size:10px}
.exp-list .vl-dur{color:#58a6ff;font-size:10px}
.hidden{display:none}
.clickable{cursor:pointer;transition:background 0.15s}
.clickable:hover{background:#1c2128}

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
      <div class="cd clickable" onclick="toggleSection('sc-pend-list')"><div class="lb">Scripts Pendientes</div><div class="vl yellow" id="sc-pend">—</div><div class="lb" style="font-size:9px;color:#555;margin-top:2px">▼ click para ver</div></div>
      <div class="cd clickable" onclick="toggleSection('sc-prod-list')"><div class="lb">Producidos (hoy)</div><div class="vl green" id="sc-prod">—</div><div class="lb" style="font-size:9px;color:#555;margin-top:2px">▼ click para ver</div></div>
      <div class="cd clickable" onclick="toggleSection('q-all-list')"><div class="lb">En Cola Upload</div><div class="vl blue" id="q-total">—</div><div class="lb" style="font-size:9px;color:#555;margin-top:2px">▼ click para ver</div></div>
      <div class="cd clickable" onclick="toggleSection('q-ready-list')"><div class="lb">Listos para Subir</div><div class="vl green" id="q-ready">—</div><div class="lb" style="font-size:9px;color:#555;margin-top:2px">▼ click para ver</div></div>
    </div>
    <!-- Expandable lists -->
    <div id="sc-pend-list" class="exp-list hidden"></div>
    <div id="sc-prod-list" class="exp-list hidden"></div>
    <div id="q-all-list" class="exp-list hidden"></div>
    <div id="q-ready-list" class="exp-list hidden"></div>
    <div class="crd mt"><div class="crd-t">🔄 Producción Activa</div><div id="prod-info" class="muted" style="font-size:12px">Cargando...</div></div>
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

<!-- ═══ PIPELINE & QA STATUS ═══ -->
<div class="two">
  <div class="sec">
    <div class="sec-t yellow">⚙️ Pipeline Status</div>
    <div class="crd">
      <div id="pipeline-processes" style="margin-bottom:10px"></div>
      <div class="g3">
        <div class="cd"><div class="lb">Guiones Pendientes</div><div class="vl blue" id="pl-pending">-</div></div>
        <div class="cd"><div class="lb">Producidos Hoy</div><div class="vl green" id="pl-today">-</div></div>
        <div class="cd"><div class="lb">Listos p/ Upload</div><div class="vl yellow" id="pl-ready">-</div></div>
        <div class="cd"><div class="lb">QA Failed</div><div class="vl red" id="pl-qa-fail">-</div></div>
      </div>
    </div>
  </div>
  <div class="sec">
    <div class="sec-t" style="border-color:#6e40c9">🔬 QA Agent</div>
    <div class="crd">
      <div style="display:flex;gap:12px;margin-bottom:10px">
        <div class="cd" style="flex:1"><div class="lb">Pass Rate</div><div class="vl green" id="qa-rate">-</div></div>
        <div class="cd" style="flex:1"><div class="lb">Alertas CEO</div><div class="vl yellow" id="qa-alerts">-</div></div>
      </div>
      <div class="crd-t">Últimos checks</div>
      <div id="qa-recent" class="muted" style="font-size:12px">Cargando...</div>
    </div>
  </div>
</div>

<!-- ═══ AGENT NOTES / CHAT ═══ -->
<div class="sec">
  <div class="sec-t green">💬 Notas del Agente — BitTrader Team</div>
  <div class="chat-box">
    <div class="chat-agent-header">
      <span>🤖 CEO Agent / Scout</span>
      <span class="chat-badge" id="an-count">0 notas</span>
    </div>
    <div class="chat-messages" id="an-messages">
      <div class="an-empty">Cargando notas del equipo...</div>
    </div>
    <div class="chat-input-row">
      <input type="text" id="an-input" placeholder="Escribe una nota o mensaje para el agente..." maxlength="500" onkeydown="if(event.key==='Enter'&&!event.shiftKey)sendAnNote()">
      <button class="btn" onclick="sendAnNote()">📤</button>
    </div>
    <div class="chat-tip">📌 Las notas del agente aparecen automáticamente. Presiona Enter para enviar.</div>
  </div>
</div>

</div>

<script>
function fmt(n){if(n>=1e6)return(n/1e6).toFixed(1)+'M';if(n>=1e3)return(n/1e3).toFixed(1)+'K';return String(n)}
function fmtH(h){return h>=1000?fmt(h):h.toFixed(0)+'h'}

function toggleSection(id){
  const el=document.getElementById(id);
  if(!el)return;
  const wasHidden=el.classList.contains('hidden');
  // Close all sections first
  document.querySelectorAll('.exp-list').forEach(e=>e.classList.add('hidden'));
  if(wasHidden){
    el.classList.remove('hidden');
    el.scrollIntoView({behavior:'smooth',block:'nearest'});
  }
}

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
    // All queue items (expandable)
    let allH='', readyH='';
    for(const i of d.items){
      const cls=i.status==='ready'?'ready':i.status==='blocked'?'blocked':i.status==='qa_failed'?'qa':'pending';
      const item=`<div class="vl-item">
        <div class="vl-title">${i.title}</div>
        <div class="vl-meta">
          <span class="vl-tag ${i.type}">${(i.type||'short').toUpperCase()}</span>
          <span class="vl-tag ${cls}">${i.status}</span>
          <span class="vl-sched">📅 ${i.scheduled}</span>
        </div>
      </div>`;
      allH+=item;
      if(i.status==='ready') readyH+=item;
    }
    document.getElementById('q-all-list').innerHTML=allH||'<div class="vl-item"><span class="muted">Cola vacía ✅</span></div>';
    document.getElementById('q-ready-list').innerHTML=readyH||'<div class="vl-item"><span class="muted">No hay items listos para subir</span></div>';
    // Legacy compact list
    let h='';
    for(const i of d.items.slice(0,10)){
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
    // Expandable pending scripts list
    let pendH='';
    for(const s of d.items){
      const tags=s.tags.slice(0,4).map(t=>`<span class="vl-tag ${s.type||'short'}">${t}</span>`).join('');
      pendH+=`<div class="vl-item">
        <div class="vl-title">${s.title}</div>
        <div class="vl-meta">
          <span class="vl-tag ${s.type||'short'}">${(s.type||'short').toUpperCase()}</span>
          ${tags}
        </div>
      </div>`;
    }
    document.getElementById('sc-pend-list').innerHTML=pendH||'<div class="vl-item"><span class="muted">Sin scripts pendientes ✅</span></div>';
    // Legacy compact list
    let h='';
    for(const s of d.items.slice(0,5)){
      const bc=s.type==='long'?'b-long':'b-short';
      const tags=s.tags.map(t=>`<span class="badge" style="background:#21262d;color:#8b949e">${t}</span>`).join('');
      h+=`<div class="si"><div class="si-t">${s.title}</div><div class="si-m"><span class="badge ${bc}">${s.type.toUpperCase()}</span>${tags}</div></div>`;
    }
    document.getElementById('sc-list').innerHTML=h||'<span class="muted">Sin scripts pendientes ✅</span>';
  }catch(e){console.error('Scripts error:',e)}
}

async function loadProduced(){
  try{
    const d=await(await fetch('/api/produced-today')).json();
    document.getElementById('sc-prod').textContent=d.count;
    // Populate expandable list
    let h='';
    for(const v of d.videos){
      h+=`<div class="vl-item">
        <div class="vl-title">${v.title||v.id}</div>
        <div class="vl-meta">
          <span class="vl-tag ${v.type||'short'}">${(v.type||'short').toUpperCase()}</span>
          <span class="vl-dur">⏱ ${v.duration||'?'}</span>
          <span class="vl-sched">📅 ${v.scheduled||'sin fecha'}</span>
        </div>
      </div>`;
    }
    document.getElementById('sc-prod-list').innerHTML=h||'<div class="vl-item"><span class="muted">Sin videos producidos hoy</span></div>';
  }catch(e){console.error('Produced error:',e)}
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

async function loadPipeline(){
  try{
    const d=await(await fetch('/api/pipeline-status')).json();
    const procs=d.processes||{};
    const dots=Object.entries(procs).map(([k,v])=>`<span style="color:${v?'#3fb950':'#484f58'};margin-right:10px">● ${k}</span>`).join('');
    document.getElementById('pipeline-processes').innerHTML=dots;
    document.getElementById('pl-pending').textContent=d.scripts?.pending??'-';
    document.getElementById('pl-today').textContent=d.today_produced??'-';
    document.getElementById('pl-ready').textContent=d.queue?.ready_to_upload??'-';
    document.getElementById('pl-qa-fail').textContent=d.queue?.qa_failed??'-';
  }catch(e){}
}

async function loadQA(){
  try{
    const d=await(await fetch('/api/qa-status')).json();
    document.getElementById('qa-rate').textContent=d.pass_rate||'N/A';
    document.getElementById('qa-alerts').textContent=d.pending_escalations??0;
    const recent=d.recent||[];
    if(recent.length===0){
      document.getElementById('qa-recent').innerHTML='Sin checks recientes';
      return;
    }
    document.getElementById('qa-recent').innerHTML=recent.map(r=>{
      const icon=r.passed?'✅':'❌';
      const issues=r.issues?.length?` — ${r.issues.join(', ')}`:'';
      return `<div class="qi"><span class="qi-t">${icon} ${r.title}</span><span class="qi-m">${issues}</span></div>`;
    }).join('');
  }catch(e){}
}

async function loadAll(){
  document.getElementById('ts').textContent=new Date().toLocaleString('es-MX',{timeZone:'America/Denver'});
  await Promise.all([loadYT(),loadQueue(),loadScripts(),loadProduced(),loadMacs(),loadProducer(),loadMiniMax(),loadPipeline(),loadQA()]);
}

// ── Agent Notes — Real-time SSE ──
var evtSource = null;
var anLastCount = 0;
var thinkingEl = null;

function renderNotes(notes){
  const msgs=notes.messages||[];
  const count=msgs.length;
  document.getElementById('an-count').textContent=count+' '+(count===1?'nota':'notas');
  if(count===0){
    document.getElementById('an-messages').innerHTML='<div class="an-empty">Sin notas aún. El agente dejará actualizaciones aquí.</div>';
    return;
  }
  let h='';
  for(const m of msgs){
    const cls=m.sender==='agent'?'msg-agent':'msg-user';
    const label=m.sender==='agent'?'🤖 Agente':'👤 Ender';
    const time=new Date(m.ts).toLocaleString('es-MX',{timeZone:'America/Denver',month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
    h+=`<div class="msg ${cls}"><div class="msg-meta">${label} · ${time}</div><div class="msg-bubble">${escapeHtml(m.text)}</div></div>`;
  }
  document.getElementById('an-messages').innerHTML=h;
  const box=document.getElementById('an-messages');
  box.scrollTop=box.scrollHeight;

  // Control typing indicator based on server-side agent_typing state
  if(notes.agent_typing === true){
    showThinking(true);
  } else {
    showThinking(false);
  }

  if(count>anLastCount && anLastCount>0){
    const hdr=document.querySelector('.chat-agent-header');
    const latest=msgs[msgs.length-1];
    if(latest && latest.sender==='agent'){
      hdr.style.background='#1f3d1f';
      setTimeout(()=>hdr.style.background='#1c2128',2000);
    } else {
      hdr.style.background='#1f3d1f';
      setTimeout(()=>hdr.style.background='#1c2128',1000);
    }
  }
  anLastCount=count;
}

function showThinking(show){
  if(!thinkingEl) thinkingEl=document.getElementById('an-thinking');
  if(!thinkingEl){
    const box=document.getElementById('an-messages');
    const el=document.createElement('div');
    el.id='an-thinking';
    el.className='thinking-wrap';
    el.innerHTML=`
      <div class="thinking-label">🤖 Agente</div>
      <div class="thinking-bubble">
        <div class="typing-indicator">
          <div class="dot"></div><div class="dot"></div><div class="dot"></div>
        </div>
        <span class="thinking-text">Pensando...</span>
      </div>`;
    box.appendChild(el);
    thinkingEl=el;
    box.scrollTop=box.scrollHeight;
  }
  if(show){
    thinkingEl.className='thinking-wrap active';
    const box=document.getElementById('an-messages');
    box.scrollTop=box.scrollHeight;
    // Start cycling status messages
    if(!thinkingEl._interval){
      const statuses=['Pensando...','Analizando datos...','Preparando respuesta...','Consultando al equipo...','Escribiendo...'];
      let idx=0;
      thinkingEl._interval=setInterval(()=>{
        idx=(idx+1)%statuses.length;
        const txt=thinkingEl.querySelector('.thinking-text');
        if(txt)txt.textContent=statuses[idx];
      },3000);
    }
  }else{
    thinkingEl.className='thinking-wrap';
    if(thinkingEl._interval){clearInterval(thinkingEl._interval);thinkingEl._interval=null;}
  }
}

function escapeHtml(t){
  return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

async function sendAnNote(){
  const inp=document.getElementById('an-input');
  const text=inp.value.trim();
  if(!text)return;
  inp.value='';
  inp.disabled=true;
  const btn=inp.nextElementSibling;
  if(btn)btn.disabled=true;
  showThinking(true);
  try{
    await fetch('/api/agent-notes',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:text,sender:'user'})
    });
  }catch(e){console.error('Send error:',e);showThinking(false);}
  inp.disabled=false;
  if(btn)btn.disabled=false;
  inp.focus();
}

function connectSSE(){
  if(evtSource)evtSource.close();
  evtSource=new EventSource('/api/agent-notes/stream');
  // Handle named events
  evtSource.addEventListener('init',function(e){
    try{const d=JSON.parse(e.data);renderNotes(d.notes);}catch(x){console.error('init parse:',x);}
  });
  evtSource.addEventListener('update',function(e){
    try{const d=JSON.parse(e.data);renderNotes(d.notes);}catch(x){console.error('update parse:',x);}
  });
  // Fallback: handle unnamed events too
  evtSource.onmessage=function(e){
    try{
      const d=JSON.parse(e.data);
      if(d.notes)renderNotes(d.notes);
    }catch(x){}
  };
  evtSource.onerror=function(e){
    console.log('SSE reconnecting...');
    evtSource.close();
    setTimeout(connectSSE,3000);
  };
}

// Also load notes via API on page load as fallback
async function loadAgentNotes(){
  try{
    const d=await(await fetch('/api/agent-notes')).json();
    renderNotes(d);
  }catch(e){}
}
loadAgentNotes();
connectSSE();

loadAll();
setInterval(loadAll,60000);
setInterval(loadMacs,10000);
setInterval(loadProducer,15000);
setInterval(loadPipeline,15000);
setInterval(loadQA,30000);
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
