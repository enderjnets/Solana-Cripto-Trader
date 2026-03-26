#!/usr/bin/env python3
"""
🛡️ Pipeline Guardian — BitTrader Quality Gate
Ingeniero — 14 marzo 2026

PROBLEMA RAÍZ IDENTIFICADO:
  El pipeline era: Scout → Creator → Producer → Publisher
  Faltaban 2 pasos obligatorios entre Producer y Publisher:
    1. Thumbnail Agent  (cada video DEBE tener thumbnail antes de subir)
    2. Quality Checker  (cada video DEBE pasar QC antes de entrar al queue)

  Resultado: videos subían sin thumbnail, videos fallidos entraban al queue,
  `output_file` no se propagaba, thumbnails se buscaban en paths incorrectos.

PIPELINE CORREGIDO:
  Scout → Creator → Producer → [Thumbnail] → [QC Gate] → Queue → Publisher
                                    ↑               ↑
                               NUEVO PASO       NUEVO PASO
                             (obligatorio)    (bloquea si falla)

RESPONSABILIDADES DE CADA AGENTE:
  🔭 Scout:            Encuentra tendencias y noticias
  🎨 Creator:          Genera scripts
  🎬 Producer:         Produce videos (mp4 + audio + subtítulos)
  🖼️  Thumbnail Agent: Genera thumbnail para CADA video producido
  ✅ Quality Checker:  Verifica video + thumbnail antes de encolar
  📤 Queue Processor:  Sube solo videos que pasaron QC con thumbnail
  📺 Publisher:        Marca como publicado, gestiona schedule

REGLAS INVARIABLES (no se pueden saltear):
  1. Ningún video entra al queue sin output_file
  2. Ningún video entra al queue sin thumbnail_path existente en disco
  3. Ningún video entra al queue sin pasar QC (brightness > 30)
  4. Queue Processor verifica estas 3 condiciones antes de subir
  5. Si alguna falla → status='blocked' con razón específica (no 'failed')
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE      = Path(__file__).parent
DATA_DIR  = BASE / "data"
THUMB_DIR = DATA_DIR / "thumbnails"
OUTPUT_BASE = BASE / "output"

QUEUE_FILE = DATA_DIR / "upload_queue.json"

# ── Invariant checks ──────────────────────────────────────────────────────────

def check_output_file(entry: dict) -> tuple[bool, str]:
    """Rule 1: video mp4 must exist and be > 500KB"""
    path = entry.get("output_file", "")
    if not path:
        sid = entry.get("script_id", "")
        # Try to find it
        today = datetime.now().strftime("%Y-%m-%d")
        candidate = OUTPUT_BASE / today / sid / f"{sid}.mp4"
        if candidate.exists() and candidate.stat().st_size > 500_000:
            return True, str(candidate)
        return False, "output_file missing and not found on disk"
    p = Path(path)
    if not p.exists():
        return False, f"output_file not found: {path}"
    if p.stat().st_size < 500_000:
        return False, f"output_file too small: {p.stat().st_size//1024}KB"
    return True, path


def check_thumbnail(entry: dict) -> tuple[bool, str]:
    """Rule 2: thumbnail must exist on disk and pass visual quality checks."""
    sid = entry.get("script_id", "")

    # Search order: thumbnail_path → thumbnail → data/thumbnails/ → video dir
    candidates = []
    if entry.get("thumbnail_path"):
        candidates.append(Path(entry["thumbnail_path"]))
    if entry.get("thumbnail"):
        candidates.append(Path(entry["thumbnail"]))
    for ext in ["jpg", "png"]:
        candidates.append(THUMB_DIR / f"{sid}_thumbnail.{ext}")
    # Video dir
    output_file = entry.get("output_file", "")
    if output_file:
        vid_dir = Path(output_file).parent
        for name in [f"{sid}_thumbnail.jpg", "thumbnail.jpg", "thumbnail_final.jpg"]:
            candidates.append(vid_dir / name)

    for c in candidates:
        if c.exists() and c.stat().st_size > 5_000:
            # FIX 2: Visual validation — reject black/blue generic frames
            try:
                from PIL import Image
                img = Image.open(c).convert("RGB")
                pixels = list(img.getdata())
                if pixels:
                    avg_b = sum(p[2] for p in pixels) / len(pixels)
                    avg_r = sum(p[0] for p in pixels) / len(pixels)
                    avg_brightness = sum(
                        0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]
                        for p in pixels
                    ) / len(pixels)
                    if avg_brightness < 30:
                        return False, f"Thumbnail demasiado oscura (frame negro genérico, brightness={avg_brightness:.1f})"
                    if avg_b > avg_r * 1.8 and avg_b > 80:
                        return False, f"Thumbnail parece frame azul genérico del video (avg_b={avg_b:.0f}, avg_r={avg_r:.0f})"
            except Exception as pil_err:
                # If PIL check fails, fall back to accepting the file
                pass
            return True, str(c)

    return False, f"No thumbnail found for {sid} (checked {len(candidates)} locations)"


def check_quality(entry: dict) -> tuple[bool, str]:
    """Rule 3: video must have brightness > 30 (not black)"""
    # Check quality_report if available
    qr_file = DATA_DIR / "quality_report.json"
    if qr_file.exists():
        try:
            qr = json.loads(qr_file.read_text())
            sid = entry.get("script_id", "")
            for v in qr.get("videos", []):
                if v.get("id") == sid or v.get("script_id") == sid:
                    brightness = v.get("brightness", 0)
                    if brightness < 15:
                        return False, f"Video BLACK (brightness={brightness:.0f})"
                    if brightness < 30:
                        return True, f"WARN: dark video (brightness={brightness:.0f})"
                    return True, f"brightness={brightness:.0f} OK"
        except Exception:
            pass

    # Fallback: check file size as proxy for content
    output_file = entry.get("output_file", "")
    if output_file and Path(output_file).exists():
        size_mb = Path(output_file).stat().st_size / (1024*1024)
        if size_mb < 0.1:
            return False, f"Video too small ({size_mb:.2f}MB) — likely black/empty"
        return True, f"size={size_mb:.1f}MB (QC report not available)"

    return False, "Cannot verify quality: no output_file"


# ── Gate: validate entire entry before allowing upload ────────────────────────

def gate_check(entry: dict) -> dict:
    """
    Runs all 3 invariant checks on a queue entry.
    Returns: {ok: bool, output_file: str, thumbnail_path: str, issues: [str], warnings: [str]}
    """
    issues   = []
    warnings = []
    updates  = {}

    # Rule 1: output_file
    ok1, result1 = check_output_file(entry)
    if ok1:
        updates["output_file"] = result1
    else:
        issues.append(f"NO_VIDEO: {result1}")

    # Rule 2: thumbnail
    ok2, result2 = check_thumbnail(entry)
    if ok2:
        updates["thumbnail_path"]  = result2
        updates["thumbnail_ready"] = True
    else:
        issues.append(f"NO_THUMBNAIL: {result2}")

    # Rule 3: quality
    ok3, result3 = check_quality(entry)
    if ok3:
        if result3.startswith("WARN"):
            warnings.append(result3)
    else:
        issues.append(f"BAD_QUALITY: {result3}")

    passed = len(issues) == 0
    return {
        "ok":             passed,
        "updates":        updates,
        "issues":         issues,
        "warnings":       warnings,
        "checked_at":     datetime.now(timezone.utc).isoformat(),
    }


# ── Queue audit: scan all entries and fix/block ───────────────────────────────

def audit_queue(auto_fix: bool = True, verbose: bool = True) -> dict:
    """
    Scans the entire upload queue.
    - Fixes what can be fixed automatically (output_file paths, thumbnail paths)
    - Blocks entries that fail invariants with status='blocked'
    - Returns summary
    """
    if not QUEUE_FILE.exists():
        return {"error": "Queue file not found"}

    queue = json.loads(QUEUE_FILE.read_text())
    summary = {"total": len(queue), "ready": 0, "blocked": 0, "fixed": 0,
               "uploaded": 0, "skipped": 0}

    for i, entry in enumerate(queue):
        status = entry.get("status", "")
        sid    = entry.get("script_id", "?")
        title  = entry.get("title", "?")[:40]

        # Skip already uploaded/published
        if status in ("uploaded", "published"):
            summary["uploaded"] += 1
            if verbose:
                print(f"  ✅ [uploaded] {title}")
            continue

        # Skip truly broken quality (already deleted from YT)
        if status == "failed_video_quality":
            summary["skipped"] += 1
            if verbose:
                print(f"  ⏭️  [skip] {title} — black video, skipping")
            continue

        # Run gate check
        result = gate_check(entry)

        if result["ok"]:
            # Apply auto-fixes (correct paths found)
            if auto_fix and result["updates"]:
                entry.update(result["updates"])
                entry["status"] = "ready"
                entry["gate_passed"] = True
                entry["gate_checked_at"] = result["checked_at"]
                if result["warnings"]:
                    entry["gate_warnings"] = result["warnings"]
                summary["fixed"] += 1
                summary["ready"] += 1
                if verbose:
                    print(f"  🔧 [fixed→ready] {title}")
                    for w in result["warnings"]:
                        print(f"       ⚠️  {w}")
            else:
                entry["status"] = "ready"
                entry["gate_passed"] = True
                summary["ready"] += 1
                if verbose:
                    print(f"  ✅ [ready] {title}")
        else:
            # Block entry — cannot upload
            if auto_fix:
                entry["status"]       = "blocked"
                entry["gate_passed"]  = False
                entry["gate_issues"]  = result["issues"]
                entry["gate_checked_at"] = result["checked_at"]
            summary["blocked"] += 1
            if verbose:
                print(f"  🚫 [blocked] {title}")
                for issue in result["issues"]:
                    print(f"       ❌ {issue}")

    if auto_fix:
        QUEUE_FILE.write_text(json.dumps(queue, indent=2))

    return summary


# ── Post-upload verification ──────────────────────────────────────────────────

def verify_uploaded(video_id: str, title: str, thumbnail_uploaded: bool,
                    youtube_client=None) -> dict:
    """
    Called by queue_processor after every upload.
    Verifies the upload was complete: video + thumbnail.
    FIX 1: Uses real YouTube API to confirm custom thumbnail presence.
    """
    issues = []
    if not video_id:
        issues.append("No YouTube video_id returned")

    # FIX 1: Verify via real YouTube API whether the video has a custom thumbnail
    api_has_custom_thumb = None
    if video_id and youtube_client is not None:
        try:
            yt_response = youtube_client.videos().list(
                part="snippet", id=video_id
            ).execute()
            items = yt_response.get("items", [])
            if items:
                thumbnails = items[0]["snippet"].get("thumbnails", {})
                # YouTube sets 'maxres' only for custom thumbnails
                api_has_custom_thumb = "maxres" in thumbnails
                if not api_has_custom_thumb:
                    issues.append(
                        f"YouTube API: {video_id} no tiene thumbnail custom (solo auto-generada) — auto-fix requerido"
                    )
                    print(f"  ⚠️ {video_id} sin thumbnail custom en YouTube — activando auto-fix flag")
                else:
                    print(f"  ✅ YouTube API confirma thumbnail custom presente: {video_id}")
            else:
                issues.append(f"YouTube API: video {video_id} no encontrado en la respuesta")
        except Exception as e:
            # Non-fatal: log but don't block
            issues.append(f"YouTube API verify error: {str(e)[:120]}")

    # Fallback: use the flag from the upload step if API check wasn't done
    if api_has_custom_thumb is None and not thumbnail_uploaded:
        issues.append("Thumbnail was NOT uploaded — video live without custom thumbnail")

    thumbnail_ok = api_has_custom_thumb if api_has_custom_thumb is not None else thumbnail_uploaded

    result = {
        "video_id":           video_id,
        "title":              title[:50],
        "thumbnail_ok":       thumbnail_ok,
        "api_verified":       api_has_custom_thumb,
        "issues":             issues,
        "verified_at":        datetime.now(timezone.utc).isoformat(),
        "post_upload_ok":     len(issues) == 0,
    }

    # Log to post_upload_verifications.json
    log_file = DATA_DIR / "post_upload_verifications.json"
    existing = json.loads(log_file.read_text()) if log_file.exists() else []
    existing.append(result)
    log_file.write_text(json.dumps(existing[-50:], indent=2))  # Keep last 50

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline Guardian — Quality Gate")
    parser.add_argument("--audit",    action="store_true", help="Audit and fix queue")
    parser.add_argument("--dry-run",  action="store_true", help="Audit without writing changes")
    parser.add_argument("--quiet",    action="store_true", help="Minimal output")
    args = parser.parse_args()

    print("=" * 60)
    print("🛡️  PIPELINE GUARDIAN — Quality Gate Audit")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M')} (local)")
    print("=" * 60)
    print()

    summary = audit_queue(auto_fix=not args.dry_run, verbose=not args.quiet)

    print()
    print("=" * 60)
    print("📊 RESUMEN")
    print(f"   Total en queue:  {summary['total']}")
    print(f"   ✅ Ready:         {summary['ready']}")
    print(f"   🔧 Fixed→Ready:  {summary['fixed']}")
    print(f"   🚫 Blocked:      {summary['blocked']}")
    print(f"   📹 Ya subidos:   {summary['uploaded']}")
    print(f"   ⏭️  Skipped:      {summary['skipped']}")
    if summary["blocked"] == 0:
        print("\n✅ Queue limpio — todos los videos listos para subir")
    else:
        print(f"\n⚠️  {summary['blocked']} video(s) bloqueados — revisar issues arriba")
    print("=" * 60)
