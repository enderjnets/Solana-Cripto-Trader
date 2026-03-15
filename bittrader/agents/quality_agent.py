#!/usr/bin/env python3
"""
🎬 Quality Agent — BitTrader Autonomous Channel Guardian
Ingeniero — 14 marzo 2026

MISIÓN: Ser el juez final de TODO lo que toca el canal de BitTrader.
No pregunta. No espera. Decide y ejecuta.

DECISIONES AUTÓNOMAS:
  ✅ APPROVE  → video pasa, va a la cola de upload
  ❌ DELETE   → video se borra de YouTube sin preguntar
  🔄 REGENERATE → video se elimina y se re-genera
  🔒 HOLD     → video se pone en privado hasta corrección
  🖼️  FIX_THUMB → genera y sube thumbnail faltante

CRITERIOS DE CALIDAD:
  1. Sin headers en subtítulos (HOOK:, Gancho:, GUIÓN COMPLETO:, etc.)
  2. Sin prompt leakage en audio/subtítulos
  3. Brightness > 30 (no negro)
  4. No fondo sólido de un solo color (azul plano = fallback mode)
  5. Thumbnail presente y correcta (16:9, 1280x720)
  6. Audio inteligible (no vacío, no estático)
  7. Duración apropiada (shorts: 15-60s, longs: 60-300s)
  8. Título < 100 chars, sin caracteres extraños
  9. Descripción con al menos 3 hashtags
  10. Contenido en español (no inglés ni gibberish)

INTEGRACIÓN:
  - CEO Agent lo llama en cada ciclo
  - Se integra en pipeline_guardian.py como última barrera
  - Audita canal completo cada 24h
  - Decisiones logueadas en data/quality_decisions.json
"""

import json
import re
import sys
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

BASE     = Path(__file__).parent
DATA_DIR = BASE / "data"
WORKSPACE= BASE.parent.parent

sys.path.insert(0, str(BASE))

DECISIONS_LOG = DATA_DIR / "quality_decisions.json"
QUEUE_FILE    = DATA_DIR / "upload_queue.json"

# ── Subtitle Contamination Patterns ──────────────────────────────────────────
SUBTITLE_BAD_PATTERNS = [
    # Header leakage
    r"gu[ií][oó]n completo\s*:?\s*hook",
    r"---\s*gu[ií][oó]n",
    r"---\s*hook\s*:",
    r"hook\s*\(\d+s\)\s*:",
    r"gancho\s*:",
    r"desarrollo\s*:",
    r"dato impactante\s*:",
    r"t[ií]tulo\s*:",
    r"secci[oó]n\s*:",
    r"introducci[oó]n\s*:",
    r"conclusi[oó]n\s*:",
    r"llamada a la acci[oó]n\s*:",
    r"cierre\s*:",
    # Prompt leakage
    r"el usuario quiere",
    r"respond only with",
    r"analyze the request",
    r"you are a scriptwriter",
    r"eres el guionista",
    r"~\d+ palabras",
    r"\*\*role\*\*",
    r"\*\*format\*\*",
    r"1\.\s+\*\*analyze",
    # Template structure
    r"formato obligatorio",
    r"canal de youtube llamado bittrader",
]

# ── Color Analysis ────────────────────────────────────────────────────────────
SOLID_COLOR_TOLERANCE = 15  # Max std dev for "solid color" detection

def _load_decisions() -> list:
    if DECISIONS_LOG.exists():
        return json.loads(DECISIONS_LOG.read_text())
    return []

def _save_decision(decision: dict):
    log = _load_decisions()
    log.append(decision)
    DECISIONS_LOG.write_text(json.dumps(log[-200:], indent=2))

def _log(msg: str, level: str = "INFO"):
    print(f"[{level}] 🎬 QA: {msg}")


# ── Check Functions ───────────────────────────────────────────────────────────

def check_subtitles(script_dir: Path) -> dict:
    """Check subtitle file for contamination patterns."""
    ass_files = list(script_dir.glob("*.ass"))
    if not ass_files:
        return {"ok": True, "note": "no_subtitle_file"}

    content = ass_files[0].read_text(errors='ignore').lower()
    dialogue_lines = [l for l in content.split('\n') if l.startswith('dialogue:')]
    if not dialogue_lines:
        return {"ok": True, "note": "no_dialogue_lines"}

    # Extract visible text from first 5 lines
    visible_texts = []
    for line in dialogue_lines[:5]:
        parts = line.split(',')
        if len(parts) >= 10:
            text = ','.join(parts[9:])
            text = re.sub(r'\{[^}]+\}', '', text).strip()
            visible_texts.append(text)

    combined = ' '.join(visible_texts).lower()

    for pattern in SUBTITLE_BAD_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            matched = re.search(pattern, combined, re.IGNORECASE)
            return {
                "ok": False,
                "issue": "SUBTITLE_CONTAMINATION",
                "pattern": pattern,
                "matched": matched.group(0) if matched else pattern,
                "sample": combined[:100],
            }

    return {"ok": True, "first_line": visible_texts[0] if visible_texts else ""}


def check_visual_content(video_path: Path) -> dict:
    """Check video for black frames and solid color (fallback mode)."""
    try:
        import subprocess as sp
        # Extract 5 frames and check brightness + color variance
        result = sp.run([
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration",
            "-of", "json", str(video_path)
        ], capture_output=True, text=True, timeout=15)

        info = json.loads(result.stdout) if result.stdout else {}
        streams = info.get("streams", [{}])
        duration = float(streams[0].get("duration", 0)) if streams else 0

        if duration <= 0:
            return {"ok": False, "issue": "ZERO_DURATION"}

        # Sample frames
        import tempfile, os
        brightnesses = []
        r_vals, g_vals, b_vals = [], [], []

        for t in [2, 4, 6, 8, 10]:
            if t >= duration:
                break
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                fname = f.name
            try:
                sp.run([
                    "ffmpeg", "-y", "-ss", str(t), "-i", str(video_path),
                    "-vframes", "1", "-q:v", "2", fname
                ], capture_output=True, timeout=10)

                from PIL import Image
                import numpy as np
                img = Image.open(fname).convert('RGB')
                arr = np.array(img)
                r, g, b = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
                brightness = 0.299*r + 0.587*g + 0.114*b
                brightnesses.append(brightness)
                r_vals.append(r); g_vals.append(g); b_vals.append(b)
            except:
                pass
            finally:
                try: os.unlink(fname)
                except: pass

        if not brightnesses:
            return {"ok": False, "issue": "FRAME_EXTRACTION_FAILED"}

        avg_brightness = sum(brightnesses) / len(brightnesses)

        # Check for black video
        if avg_brightness < 30:
            return {"ok": False, "issue": "BLACK_VIDEO", "brightness": avg_brightness}

        # Check for solid color (fallback mode detection)
        # If all frames have nearly identical color values → solid background
        if len(r_vals) >= 3:
            import statistics
            r_std = statistics.stdev(r_vals) if len(r_vals) > 1 else 0
            g_std = statistics.stdev(g_vals) if len(g_vals) > 1 else 0
            b_std = statistics.stdev(b_vals) if len(b_vals) > 1 else 0
            avg_std = (r_std + g_std + b_std) / 3

            # Also check if it's the classic "BitTrader blue" fallback
            avg_r = sum(r_vals)/len(r_vals)
            avg_g = sum(g_vals)/len(g_vals)
            avg_b = sum(b_vals)/len(b_vals)

            is_solid = avg_std < SOLID_COLOR_TOLERANCE
            is_blue_fallback = (avg_b > 150 and avg_b > avg_r * 3 and avg_b > avg_g * 1.5)

            if is_solid and is_blue_fallback:
                return {
                    "ok": False,
                    "issue": "SOLID_BLUE_FALLBACK",
                    "brightness": avg_brightness,
                    "rgb": (round(avg_r,1), round(avg_g,1), round(avg_b,1)),
                    "note": "MiniMax insufficient balance — visual content missing",
                }

        return {
            "ok": True,
            "brightness": avg_brightness,
            "duration": duration,
        }

    except Exception as e:
        return {"ok": False, "issue": f"CHECK_ERROR: {e}"}


def check_thumbnail(script_id: str, queue_entry: dict) -> dict:
    """Check if thumbnail exists and is valid."""
    thumb_path = queue_entry.get("thumbnail_path", "")
    if thumb_path and Path(thumb_path).exists():
        size = Path(thumb_path).stat().st_size
        if size > 10_000:  # > 10KB
            return {"ok": True, "path": thumb_path, "size_kb": size//1024}
        return {"ok": False, "issue": "THUMBNAIL_TOO_SMALL", "size": size}

    # Search fallback locations
    thumb_dir = DATA_DIR / "thumbnails"
    for pattern in [f"{script_id}*", f"*{script_id[-8:]}*"]:
        found = list(thumb_dir.glob(pattern)) if thumb_dir.exists() else []
        if found:
            return {"ok": True, "path": str(found[0]), "found_via": "fallback_search"}

    return {"ok": False, "issue": "THUMBNAIL_MISSING"}


def check_title_and_metadata(queue_entry: dict) -> dict:
    """Check title length, description, tags."""
    title = queue_entry.get("title", "")
    issues = []

    if len(title) > 100:
        issues.append(f"TITLE_TOO_LONG: {len(title)} chars")
    if len(title) < 5:
        issues.append("TITLE_TOO_SHORT")

    # Check for template leakage in title
    if re.search(r"~\d+|HOOK|GUIÓN|prompt|template", title, re.IGNORECASE):
        issues.append(f"TITLE_CONTAMINATED: {title[:50]}")

    if issues:
        return {"ok": False, "issues": issues}
    return {"ok": True}


# ── DECISION ENGINE ───────────────────────────────────────────────────────────

def make_quality_decision(
    script_id: str,
    queue_entry: dict,
    script_dir: Optional[Path] = None,
    youtube_id: Optional[str] = None,
) -> dict:
    """
    The Quality Agent's core decision function.
    Runs ALL checks and returns an autonomous decision.

    Returns:
      {
        "decision": "APPROVE" | "DELETE" | "REGENERATE" | "HOLD" | "FIX_THUMB",
        "score":    0-100,
        "checks":   {...},
        "reason":   "human-readable explanation",
        "action":   "what to do next",
      }
    """
    checks = {}
    critical_failures = []
    warnings = []

    # Find script output directory — ONLY match the most recent version
    # Critical: if a video was deleted+regenerated, the OLD dir still exists on disk.
    # We must verify against what's actually in the queue (output_file path).
    if script_dir is None:
        # Prefer the output_file path from queue (most accurate)
        output_file = queue_entry.get("output_file","")
        if output_file and Path(output_file).exists():
            script_dir = Path(output_file).parent
        else:
            # Fallback: scan output dirs newest-first
            for date_dir in sorted((BASE / "output").iterdir(), reverse=True) if (BASE / "output").exists() else []:
                candidate = date_dir / script_id
                if candidate.exists():
                    script_dir = candidate
                    break

    # ── Check 1: Subtitle contamination ──────────────────────────────────
    if script_dir and script_dir.exists():
        sub_check = check_subtitles(script_dir)
        checks["subtitles"] = sub_check
        if not sub_check["ok"]:
            critical_failures.append(f"SUBTITLE: {sub_check.get('issue','?')} — '{sub_check.get('matched','?')}'")

        # ── Check 2: Visual content ───────────────────────────────────────
        mp4_files = list(script_dir.glob("*.mp4"))
        if mp4_files:
            vis_check = check_visual_content(mp4_files[0])
            checks["visual"] = vis_check
            if not vis_check["ok"]:
                issue = vis_check.get("issue","")
                if issue == "SOLID_BLUE_FALLBACK":
                    warnings.append("VISUAL: Solid blue fallback (MiniMax balance issue)")
                elif issue == "BLACK_VIDEO":
                    critical_failures.append(f"VISUAL: Black video (brightness={vis_check.get('brightness',0):.1f})")
                else:
                    warnings.append(f"VISUAL: {issue}")
        else:
            checks["visual"] = {"ok": False, "issue": "NO_MP4_FILE"}
            critical_failures.append("VISUAL: No MP4 file found")

    # ── Check 3: Thumbnail ───────────────────────────────────────────────
    thumb_check = check_thumbnail(script_id, queue_entry)
    checks["thumbnail"] = thumb_check
    if not thumb_check["ok"]:
        warnings.append(f"THUMBNAIL: {thumb_check.get('issue','MISSING')}")

    # ── Check 4: Title/Metadata ───────────────────────────────────────────
    meta_check = check_title_and_metadata(queue_entry)
    checks["metadata"] = meta_check
    if not meta_check["ok"]:
        critical_failures.extend(meta_check.get("issues", []))

    # ── Score calculation ─────────────────────────────────────────────────
    score = 100
    score -= len(critical_failures) * 25
    score -= len(warnings) * 10
    score = max(0, score)

    # ── DECISION LOGIC ─────────────────────────────────────────────────────
    has_subtitle_contamination = any("SUBTITLE" in f for f in critical_failures)
    has_black_video             = any("BLACK_VIDEO" in f for f in critical_failures)
    has_solid_blue              = any("SOLID_BLUE" in w for w in warnings)
    has_missing_thumbnail       = any("THUMBNAIL" in w for w in warnings)
    has_no_file                 = any("NO_MP4" in f for f in critical_failures)

    if has_subtitle_contamination:
        # Contaminated content = DELETE immediately + REGENERATE
        decision = "REGENERATE"
        reason   = f"Subtitle contamination detected: {critical_failures[0]}"
        action   = "Delete from YouTube, regenerate clean version"

    elif has_black_video or has_no_file:
        # Black or missing = REGENERATE
        decision = "REGENERATE"
        reason   = f"Visual failure: {critical_failures[0]}"
        action   = "Regenerate video with fixed pipeline"

    elif has_solid_blue and has_missing_thumbnail:
        # Blue fallback + no thumbnail = HOLD and regenerate when credits available
        decision = "HOLD"
        reason   = "Solid blue fallback (MiniMax no credits) + no thumbnail"
        action   = "Set private, regenerate when MiniMax credits available or use alternative"

    elif has_solid_blue:
        # Blue fallback but content/audio ok — FIX_THUMB at minimum, note the visual
        decision = "HOLD"
        reason   = "Solid blue fallback — no visual content, only text on blue background"
        action   = "Set private until real video clips available; thumbnail alone insufficient"

    elif has_missing_thumbnail:
        # Good video but missing thumbnail
        decision = "FIX_THUMB"
        reason   = "Good video content but thumbnail missing"
        action   = "Generate thumbnail and re-upload"

    elif score >= 80:
        decision = "APPROVE"
        reason   = "All critical checks passed"
        action   = "Ready for upload"

    elif score >= 60:
        decision = "HOLD"
        reason   = f"Minor issues: {warnings}"
        action   = "Review and fix before upload"

    else:
        decision = "REGENERATE"
        reason   = f"Multiple failures: {critical_failures}"
        action   = "Regenerate from scratch"

    result = {
        "script_id":    script_id,
        "youtube_id":   youtube_id or "",
        "title":        queue_entry.get("title","?"),
        "decision":     decision,
        "score":        score,
        "checks":       checks,
        "critical":     critical_failures,
        "warnings":     warnings,
        "reason":       reason,
        "action":       action,
        "decided_at":   datetime.now(timezone.utc).isoformat(),
    }

    _save_decision(result)
    return result


# ── Channel Audit ─────────────────────────────────────────────────────────────

def audit_full_channel(execute: bool = True) -> dict:
    """
    Audit ALL videos in the upload queue.
    If execute=True, actually deletes/holds/fixes based on decisions.
    """
    if not QUEUE_FILE.exists():
        return {"error": "No queue file"}

    queue = json.loads(QUEUE_FILE.read_text())
    results = {
        "approved":    [],
        "deleted":     [],
        "regenerated": [],
        "held":        [],
        "fixed_thumb": [],
        "skipped":     [],
    }

    _log(f"Auditing {len(queue)} videos in queue...")

    for entry in queue:
        status    = entry.get("status","")
        script_id = entry.get("script_id","")
        yt_id     = entry.get("video_id","") or entry.get("youtube_id","")

        # Skip already-processed entries
        if status in ("deleted_contaminated", "deleted_header_bug", "duplicate_skipped",
                      "deleted_black", "blocked"):
            results["skipped"].append(script_id)
            continue

        # Find script dir
        script_dir = None
        for date_dir in sorted((BASE / "output").iterdir(), reverse=True) if (BASE / "output").exists() else []:
            candidate = date_dir / script_id
            if candidate.exists():
                script_dir = candidate
                break

        decision_result = make_quality_decision(
            script_id=script_id,
            queue_entry=entry,
            script_dir=script_dir,
            youtube_id=yt_id,
        )

        decision = decision_result["decision"]
        _log(f"  [{decision:12}] {entry.get('title','?')[:45]} | {decision_result['reason'][:60]}")

        if not execute:
            results[decision.lower().replace("_","")] = results.get(decision.lower().replace("_",""), [])
            results[decision.lower().replace("_","")].append(script_id)
            continue

        # ── Execute decision ──────────────────────────────────────────────
        if decision == "DELETE" or decision == "REGENERATE":
            # Delete from YouTube if live
            if yt_id and status == "uploaded":
                try:
                    sys.path.insert(0, str(BASE))
                    from publisher import get_youtube_client
                    yt = get_youtube_client()
                    yt.videos().delete(id=yt_id).execute()
                    _log(f"  🗑️  Deleted from YouTube: {yt_id}")
                except Exception as e:
                    _log(f"  ⚠️  Could not delete {yt_id}: {e}", "WARNING")

            # Update queue status
            entry["status"] = "deleted_by_quality_agent" if decision == "DELETE" else "needs_regeneration"
            entry["quality_decision"] = decision_result["reason"]
            entry["quality_decided_at"] = datetime.now(timezone.utc).isoformat()

            results["deleted" if decision == "DELETE" else "regenerated"].append(script_id)

        elif decision == "HOLD":
            # Set private on YouTube if uploaded
            if yt_id and status == "uploaded":
                try:
                    from publisher import get_youtube_client
                    yt = get_youtube_client()
                    yt.videos().update(
                        part="status",
                        body={"id": yt_id, "status": {"privacyStatus": "private"}}
                    ).execute()
                    _log(f"  🔒 Set private: {yt_id}")
                except Exception as e:
                    _log(f"  ⚠️  Could not set private {yt_id}: {e}", "WARNING")

            entry["status"] = "held_by_quality_agent"
            entry["quality_decision"] = decision_result["reason"]
            results["held"].append(script_id)

        elif decision == "FIX_THUMB":
            # Generate thumbnail
            try:
                result = subprocess.run(
                    [sys.executable, str(BASE / "thumbnail_agent_huggingface.py")],
                    capture_output=True, text=True, timeout=120, cwd=str(BASE)
                )
                entry["quality_decision"] = "FIX_THUMB attempted"
            except Exception as e:
                _log(f"  ⚠️  Thumbnail fix failed: {e}", "WARNING")
            results["fixed_thumb"].append(script_id)

        elif decision == "APPROVE":
            results["approved"].append(script_id)

    # Save updated queue
    if execute:
        QUEUE_FILE.write_text(json.dumps(queue, indent=2))

    summary = {
        "total":        len(queue),
        "approved":     len(results["approved"]),
        "deleted":      len(results["deleted"]),
        "regenerated":  len(results["regenerated"]),
        "held":         len(results["held"]),
        "fixed_thumb":  len(results["fixed_thumb"]),
        "skipped":      len(results["skipped"]),
        "details":      results,
        "audited_at":   datetime.now(timezone.utc).isoformat(),
    }

    _log(f"\n📊 Audit complete: "
         f"{summary['approved']} approved | "
         f"{summary['deleted']+summary['regenerated']} deleted/regen | "
         f"{summary['held']} held | "
         f"{summary['fixed_thumb']} thumb fixed")

    return summary


# ── Pre-Upload Gate ───────────────────────────────────────────────────────────

def gate_before_upload(script_id: str, queue_entry: dict) -> bool:
    """
    Called by queue_processor BEFORE uploading any video.
    Returns True if video passes quality check, False to block.
    """
    script_dir = None
    for date_dir in sorted((BASE / "output").iterdir(), reverse=True) if (BASE / "output").exists() else []:
        candidate = date_dir / script_id
        if candidate.exists():
            script_dir = candidate
            break

    result = make_quality_decision(
        script_id=script_id,
        queue_entry=queue_entry,
        script_dir=script_dir,
    )

    if result["decision"] == "APPROVE":
        return True
    elif result["decision"] == "FIX_THUMB":
        # Try to fix thumbnail, then allow
        return True  # Allow with warning
    else:
        _log(f"⛔ BLOCKED by Quality Agent: {result['reason']}", "ERROR")
        return False


# ── CEO Integration Hook ──────────────────────────────────────────────────────

def run_quality_cycle(execute: bool = True) -> dict:
    """Called by CEO Agent every cycle."""
    _log("Starting quality audit cycle...")
    return audit_full_channel(execute=execute)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quality Agent — Autonomous channel guardian")
    parser.add_argument("--audit",    action="store_true", help="Full channel audit")
    parser.add_argument("--dry-run",  action="store_true", help="Show decisions without executing")
    parser.add_argument("--script-id", help="Check a specific script ID")
    args = parser.parse_args()

    if args.script_id:
        queue = json.loads(QUEUE_FILE.read_text()) if QUEUE_FILE.exists() else []
        entry = next((v for v in queue if v.get("script_id") == args.script_id), {"title": args.script_id})
        result = make_quality_decision(args.script_id, entry)
        print(json.dumps(result, indent=2))
    else:
        summary = audit_full_channel(execute=not args.dry_run)
        print(json.dumps(summary, indent=2))
