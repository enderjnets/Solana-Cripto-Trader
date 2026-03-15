#!/usr/bin/env python3
"""
🧠 Autonomous Brain — BitTrader Self-Healing System
Ingeniero — 14 marzo 2026

MISIÓN: El sistema debe operar SIN intervención humana.
Cada agente sabe qué hacer, cuándo actuar y cómo colaborar.

ARQUITECTURA DE AUTONOMÍA:
  ┌─────────────────────────────────────────────────────────┐
  │                    CEO AGENT                            │
  │  - Lee estado de TODOS los agentes cada ciclo          │
  │  - Detecta anomalías y toma decisiones autónomas        │
  │  - Delega acciones específicas a cada agente            │
  │  - Verifica resultado y re-intenta si falla             │
  └────────────────────┬────────────────────────────────────┘
                       │ ordena
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   [CREATOR]     [PRODUCER]     [PUBLISHER]
   genera         produce        sube
   guiones        videos         a YT
        ↓              ↓              ↓
   [VALIDATOR]   [GUARDIAN]    [QC POST-UPLOAD]
   verifica       gate check    verifica upload
   contenido      antes queue   completo

REGLAS DE AUTONOMÍA DEL CEO:
  1. Si hay < 3 videos en queue → ordenar producción nueva
  2. Si un video tiene status='invalid_content' → re-generar automáticamente
  3. Si un video fue eliminado de YT → regenerar y re-subir
  4. Si thumbnail falta → generar y subir
  5. Si upload falló → re-intentar hasta 3 veces
  6. Si canal no tiene video hoy → producción urgente
  7. Si LLM falla → usar fallback automático (ya implementado)

ACCIONES AUTÓNOMAS DISPONIBLES:
  - regenerate_video(script_id)   → vuelve a generar guión + video
  - fix_thumbnail(script_id)      → genera y sube thumbnail faltante
  - retry_upload(script_id)       → re-intenta subida fallida
  - run_creator()                 → genera nuevos scripts
  - run_producer()                → produce videos pendientes
  - delete_video(yt_id, reason)   → elimina video malo de YT
  - alert_only(message)           → solo alerta, no actúa
"""

import json
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

BASE     = Path(__file__).parent
DATA_DIR = BASE / "data"
WORKSPACE= BASE.parent.parent

try:
    from llm_config import call_llm as _bittrader_llm
except ImportError:
    def _bittrader_llm(p, system="", max_tokens=500): return None

try:
    from creator import call_llm
except ImportError:
    call_llm = _bittrader_llm

QUEUE_FILE      = DATA_DIR / "upload_queue.json"
PRODUCTION_FILE = DATA_DIR / "production_latest.json"
GUIONES_FILE    = DATA_DIR / "guiones_latest.json"
BRAIN_STATE_FILE= DATA_DIR / "autonomous_brain_state.json"
ACTION_LOG_FILE = DATA_DIR / "autonomous_actions.json"


# ── State Management ──────────────────────────────────────────────────────────

def load_brain_state() -> dict:
    if BRAIN_STATE_FILE.exists():
        return json.loads(BRAIN_STATE_FILE.read_text())
    return {
        "last_run": None,
        "actions_today": [],
        "retry_counts": {},
        "last_production_run": None,
        "last_upload_check": None,
    }

def save_brain_state(state: dict):
    BRAIN_STATE_FILE.write_text(json.dumps(state, indent=2))

def log_action(action: str, details: dict, result: str):
    log = []
    if ACTION_LOG_FILE.exists():
        log = json.loads(ACTION_LOG_FILE.read_text())
    log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action":    action,
        "details":   details,
        "result":    result,
    })
    ACTION_LOG_FILE.write_text(json.dumps(log[-100:], indent=2))  # Keep last 100


# ── System Status Reader ──────────────────────────────────────────────────────

def read_full_system_status() -> dict:
    """Reads status from ALL agents and produces a unified view."""
    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "queue":     {},
        "production":{},
        "guiones":   {},
        "youtube":   {},
        "thumbnails":{},
        "issues":    [],
        "opportunities": [],
    }

    # Queue status
    if QUEUE_FILE.exists():
        queue = json.loads(QUEUE_FILE.read_text())
        by_status = {}
        for v in queue:
            s = v.get("status","?")
            by_status[s] = by_status.get(s, 0) + 1
        status["queue"] = {
            "total":      len(queue),
            "by_status":  by_status,
            "uploaded":   [v for v in queue if v.get("status") == "uploaded"],
            "ready":      [v for v in queue if v.get("status") == "ready"],
            "failed":     [v for v in queue if v.get("status") == "failed"],
            "invalid":    [v for v in queue if v.get("status") == "invalid_content"],
            "deleted":    [v for v in queue if v.get("status") == "deleted_contaminated"],
            "blocked":    [v for v in queue if v.get("status") == "blocked"],
        }

    # Production status
    if PRODUCTION_FILE.exists():
        prod = json.loads(PRODUCTION_FILE.read_text())
        status["production"] = {
            "total":   prod.get("stats", {}).get("total", 0),
            "success": prod.get("stats", {}).get("success", 0),
            "errors":  prod.get("stats", {}).get("errors", 0),
            "videos":  prod.get("videos", []),
        }

    # Guiones status
    if GUIONES_FILE.exists():
        guiones = json.loads(GUIONES_FILE.read_text())
        scripts = guiones.get("scripts", [])
        pending   = [s for s in scripts if s.get("status") == "pending"]
        invalid   = [s for s in scripts if s.get("status") == "invalid_content"]
        status["guiones"] = {
            "total":   len(scripts),
            "pending": len(pending),
            "invalid": len(invalid),
            "pending_scripts": pending,
            "invalid_scripts": invalid,
        }

    # YouTube IDs (published videos)
    yt_ids_file = DATA_DIR / "youtube_v3_ids.json"
    if yt_ids_file.exists():
        yt_ids = json.loads(yt_ids_file.read_text())
        today = datetime.now().strftime("%Y-%m-%d")
        today_videos = [v for v in yt_ids if v.get("published", "").startswith(today[:7])]
        status["youtube"] = {
            "total_published": len(yt_ids),
            "recent": today_videos[-5:],
        }

    # Thumbnail coverage
    thumb_dir = DATA_DIR / "thumbnails"
    status["thumbnails"] = {
        "count": len(list(thumb_dir.glob("*.*"))) if thumb_dir.exists() else 0
    }

    # ── Detect Issues ────────────────────────────────────────────────────
    q = status["queue"]

    # Issue: deleted contaminated videos need regeneration
    if q.get("deleted"):
        for v in q["deleted"]:
            status["issues"].append({
                "type":     "DELETED_NEEDS_REGENERATION",
                "priority": "HIGH",
                "script_id": v.get("script_id",""),
                "title":    v.get("title","?"),
                "reason":   v.get("delete_reason",""),
                "action":   "regenerate_video",
            })

    # Issue: invalid content scripts
    if q.get("invalid"):
        for v in q["invalid"]:
            status["issues"].append({
                "type":     "INVALID_CONTENT_NEEDS_REGEN",
                "priority": "HIGH",
                "script_id": v.get("script_id",""),
                "title":    v.get("title","?"),
                "action":   "regenerate_video",
            })

    # Issue: failed uploads
    if q.get("failed"):
        for v in q["failed"]:
            status["issues"].append({
                "type":     "FAILED_UPLOAD",
                "priority": "MEDIUM",
                "script_id": v.get("script_id",""),
                "title":    v.get("title","?"),
                "action":   "retry_upload",
            })

    # Issue: ready videos missing thumbnails
    for v in q.get("ready", []):
        if not v.get("thumbnail_ready"):
            status["issues"].append({
                "type":     "MISSING_THUMBNAIL",
                "priority": "MEDIUM",
                "script_id": v.get("script_id",""),
                "title":    v.get("title","?"),
                "action":   "fix_thumbnail",
            })

    # Issue: no content uploaded today
    # Count videos uploaded OR that were uploaded then deleted by QA (still produced today)
    today = datetime.now().strftime("%Y-%m-%d")
    uploaded_today = [v for v in q.get("uploaded",[])
                      if today in str(v.get("uploaded_at","") or v.get("created_at",""))]
    # Also count needs_regeneration (was uploaded today but QA removed — will regen tomorrow)
    produced_today = uploaded_today + [
        v for v in q.get("ready",[]) + [
            x for x in (json.loads(QUEUE_FILE.read_text()) if QUEUE_FILE.exists() else [])
            if x.get("status") in ("needs_regeneration","needs_regeneration_confirmed")
        ]
        if today in str(v.get("created_at",""))
    ]
    if len(produced_today) < 1 and len(uploaded_today) < 1:
        status["issues"].append({
            "type":     "NO_UPLOADS_TODAY",
            "priority": "HIGH",
            "count":    len(uploaded_today),
            "action":   "run_full_pipeline",
        })

    # Issue: no videos in queue to upload
    if len(q.get("ready",[])) == 0 and len(q.get("uploaded",[])) < 3:
        status["issues"].append({
            "type":     "QUEUE_EMPTY",
            "priority": "HIGH",
            "action":   "run_creator_and_producer",
        })

    return status


# ── Autonomous Actions ────────────────────────────────────────────────────────

def action_regenerate_video(script_id: str, original_topic: str) -> str:
    """Regenerate a video that had contaminated/invalid content."""
    print(f"  🔄 Regenerating: {script_id} ({original_topic[:40]})")

    # Step 1: Remove from queue (so it can be re-added)
    queue = json.loads(QUEUE_FILE.read_text())
    queue = [v for v in queue if v.get("script_id") != script_id]
    QUEUE_FILE.write_text(json.dumps(queue, indent=2))

    # Step 2: Inject topic into regen_queue.json and run full pipeline
    try:
        import time as _time
        regen_file = DATA_DIR / "regen_queue.json"
        regen_data = json.loads(regen_file.read_text()) if regen_file.exists() else {"scripts": []}

        # Add this topic to regen queue if not already there
        existing_topics = [s.get("original_topic","") for s in regen_data.get("scripts",[])]
        if original_topic not in existing_topics:
            regen_data["scripts"].append({
                "id":             f"regen_{int(_time.time())}",
                "type":           "short" if "short" in script_id else "long",
                "theme":          "noticia",
                "priority":       "alta",
                "title":          original_topic,
                "original_topic": original_topic,
                "status":         "needs_regen",
                "script":         "",
                "created_at":     datetime.now(timezone.utc).isoformat(),
            })
            regen_file.write_text(json.dumps(regen_data, indent=2))

        # Run creator → producer pipeline
        for cmd, timeout in [
            ([sys.executable, str(BASE / "creator.py")], 300),
            ([sys.executable, str(BASE / "producer.py")], 600),
            ([sys.executable, str(BASE / "pipeline_guardian.py"), "--audit"], 60),
            ([sys.executable, str(BASE / "queue_processor.py")], 300),
        ]:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(BASE))
            if r.returncode != 0:
                step = Path(cmd[1]).stem
                log_action("regenerate_video", {"script_id": script_id}, f"{step.upper()}_FAILED: {r.stderr[:100]}")
                return f"{step.upper()}_FAILED"

        log_action("regenerate_video", {"script_id": script_id}, "SUCCESS")
        return "SUCCESS"
    except Exception as e:
        log_action("regenerate_video", {"script_id": script_id}, f"ERROR: {e}")
        return f"ERROR: {e}"


def action_fix_thumbnail(script_id: str, video_path: str = "") -> str:
    """Generate and upload missing thumbnail."""
    print(f"  🖼️  Fixing thumbnail: {script_id}")
    try:
        # Try thumbnail agent
        result = subprocess.run(
            [sys.executable, str(BASE / "thumbnail_agent_huggingface.py"),
             "--script-id", script_id],
            capture_output=True, text=True, timeout=120,
            cwd=str(BASE)
        )
        log_action("fix_thumbnail", {"script_id": script_id},
                   "SUCCESS" if result.returncode == 0 else f"FAILED: {result.stderr[:100]}")
        return "SUCCESS" if result.returncode == 0 else "FAILED"
    except Exception as e:
        log_action("fix_thumbnail", {"script_id": script_id}, f"ERROR: {e}")
        return f"ERROR: {e}"


def action_retry_upload(script_id: str) -> str:
    """Retry a failed upload."""
    print(f"  🔁 Retrying upload: {script_id}")

    # Reset status to ready and run queue processor
    queue = json.loads(QUEUE_FILE.read_text())
    for v in queue:
        if v.get("script_id") == script_id and v.get("status") in ("failed",):
            v["status"] = "ready"
            v["scheduled_date"] = datetime.now(timezone.utc).isoformat()
            v["retry_count"] = v.get("retry_count", 0) + 1
            break
    QUEUE_FILE.write_text(json.dumps(queue, indent=2))

    result = subprocess.run(
        [sys.executable, str(BASE / "queue_processor.py")],
        capture_output=True, text=True, timeout=300,
        cwd=str(BASE)
    )
    log_action("retry_upload", {"script_id": script_id},
               "SUCCESS" if result.returncode == 0 else f"FAILED: {result.stderr[:100]}")
    return "SUCCESS" if result.returncode == 0 else "FAILED"


def action_run_full_pipeline() -> str:
    """Run full content pipeline: Creator → Producer → Thumbnail → Guardian → Queue."""
    print("  🚀 Running full pipeline...")
    steps = [
        [sys.executable, str(BASE / "creator.py")],
        [sys.executable, str(BASE / "producer.py")],
        [sys.executable, str(BASE / "pipeline_guardian.py"), "--audit"],
        [sys.executable, str(BASE / "queue_processor.py")],
    ]
    for cmd in steps:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(BASE))
        if r.returncode != 0:
            step = Path(cmd[1]).stem
            log_action("run_full_pipeline", {"step": step}, f"FAILED: {r.stderr[:100]}")
            return f"FAILED_AT_{step.upper()}"
    log_action("run_full_pipeline", {}, "SUCCESS")
    return "SUCCESS"


# ── CEO Decision Engine ───────────────────────────────────────────────────────

def ceo_decide_and_act(status: dict, dry_run: bool = False) -> list:
    """
    CEO reads system status and autonomously decides what actions to take.
    No human approval needed — acts immediately on HIGH priority issues.
    Returns list of actions taken.
    """
    issues   = status.get("issues", [])
    actions_taken = []

    if not issues:
        print("  ✅ Sistema saludable — sin acciones requeridas")
        return actions_taken

    # Sort by priority: HIGH first
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    issues.sort(key=lambda x: priority_order.get(x.get("priority","LOW"), 2))

    state = load_brain_state()
    today = datetime.now().strftime("%Y-%m-%d")

    for issue in issues:
        itype    = issue.get("type","")
        priority = issue.get("priority","LOW")
        action   = issue.get("action","")
        script_id= issue.get("script_id","")
        title    = issue.get("title","?")

        print(f"\n  [{priority}] {itype} → {action}")

        # Check retry limits
        retry_key = f"{script_id}_{action}"
        retry_count = state.get("retry_counts", {}).get(retry_key, 0)
        if retry_count >= 3:
            print(f"    ⏭️  Max retries reached for {retry_key} — skipping")
            continue

        if dry_run:
            print(f"    [DRY RUN] Would execute: {action}")
            actions_taken.append({"action": action, "script_id": script_id, "dry_run": True})
            continue

        result = "SKIPPED"

        if action == "regenerate_video":
            topic = issue.get("title", script_id)
            result = action_regenerate_video(script_id, topic)

        elif action == "fix_thumbnail":
            result = action_fix_thumbnail(script_id)

        elif action == "retry_upload":
            result = action_retry_upload(script_id)

        elif action in ("run_full_pipeline", "run_creator_and_producer"):
            # Only run once per day
            last_run = state.get("last_production_run") or ""
            if today in last_run:
                print(f"    ⏭️  Pipeline ya corrió hoy — skipping")
                result = "ALREADY_RAN_TODAY"
            else:
                result = action_run_full_pipeline()
                state["last_production_run"] = datetime.now(timezone.utc).isoformat()

        # Update retry count
        if result not in ("SUCCESS", "ALREADY_RAN_TODAY"):
            state.setdefault("retry_counts", {})[retry_key] = retry_count + 1
        else:
            state.setdefault("retry_counts", {})[retry_key] = 0

        actions_taken.append({
            "action":    action,
            "script_id": script_id,
            "priority":  priority,
            "result":    result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        print(f"    → {result}")

    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state["actions_today"] = actions_taken
    save_brain_state(state)

    return actions_taken


# ── CEO Integration Hook ──────────────────────────────────────────────────────

def run_autonomous_cycle(dry_run: bool = False, verbose: bool = True) -> dict:
    """
    Main entry point. Called by CEO Agent on every cycle.
    Reads full system status → decides → acts → reports.
    """
    if verbose:
        print("\n" + "="*60)
        print("🧠 AUTONOMOUS BRAIN — Ciclo de auto-diagnóstico")
        print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M')} (local)")
        print("="*60)

    status  = read_full_system_status()
    issues  = status["issues"]

    if verbose:
        print(f"\n📊 Estado del sistema:")
        print(f"   Cola: {status['queue'].get('by_status', {})}")
        print(f"   Guiones: {status['guiones'].get('total',0)} total, {status['guiones'].get('pending',0)} pendientes")
        print(f"   Issues: {len(issues)}")
        for iss in issues:
            print(f"   [{iss['priority']}] {iss['type']}")

    actions = ceo_decide_and_act(status, dry_run=dry_run)

    report = {
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "issues_found":  len(issues),
        "actions_taken": len(actions),
        "actions":       actions,
        "system_status": {
            "queue_ready":    len(status["queue"].get("ready",[])),
            "queue_uploaded": len(status["queue"].get("uploaded",[])),
            "queue_deleted":  len(status["queue"].get("deleted",[])),
            "guiones_pending": status["guiones"].get("pending",0),
        }
    }

    if verbose:
        print(f"\n✅ Ciclo completado: {len(actions)} acciones tomadas")
        for a in actions:
            icon = "✅" if a.get("result") == "SUCCESS" else "⚠️"
            print(f"   {icon} {a['action']} → {a.get('result','?')}")

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet",   action="store_true")
    args = parser.parse_args()

    report = run_autonomous_cycle(dry_run=args.dry_run, verbose=not args.quiet)
    print(f"\n📄 Acciones: {report['actions_taken']} | Issues: {report['issues_found']}")
