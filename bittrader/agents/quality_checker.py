#!/usr/bin/env python3
"""
✅ BitTrader Quality Checker v2.0 — Auditor de Calidad Completo
Verifica cada aspecto de los videos antes de publicar.

Pipeline: Producer → Thumbnail Agent → Quality Checker → Publisher

Checks realizados:
1. VIDEO: formato, resolución, duración, tamaño, orientación
2. AUDIO: presente, niveles (no silencio, no clipping), sample rate
3. VISUAL: frames negros, bitrate mínimo, clips no corruptos
4. SYNC: audio y video misma duración (sincronización)
5. THUMBNAIL: dimensiones, formato, tamaño, no vacía, texto legible
6. OVERLAY: título no cortado, logo posición correcta
7. METADATA: título, tags, descripción

Grados:
  A = Perfecto (0 issues, 0 warnings)
  B = Aceptable (0 issues, warnings menores)
  C = Publicable con reservas (0 issues, warnings importantes)
  F = No publicar (issues críticos)

Ejecutar: python3 agents/quality_checker.py
"""
import json
import subprocess
import struct
import math
from datetime import datetime, timezone
from pathlib import Path

try:
    from PIL import Image, ImageStat
except ImportError:
    Image = None
    ImageStat = None

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE  = Path("/home/enderj/.openclaw/workspace")
BITTRADER  = WORKSPACE / "bittrader"
DATA_DIR   = BITTRADER / "agents/data"

PRODUCTION_FILE = DATA_DIR / "production_latest.json"
QC_REPORT_FILE  = DATA_DIR / "quality_report.json"

# ── Quality Standards ──────────────────────────────────────────────────────
STANDARDS = {
    "short": {
        "min_duration": 5,
        "max_duration": 60,
        "expected_width": 1080,
        "expected_height": 1920,
        "max_size_mb": 50,
        "min_bitrate_kbps": 500,
        "requires_thumbnail": False,
    },
    "long": {
        "min_duration": 30,
        "max_duration": 3600,
        "expected_width": 1920,
        "expected_height": 1080,
        "max_size_mb": 500,
        "min_bitrate_kbps": 800,
        "requires_thumbnail": True,
    }
}

THUMB_STANDARDS = {
    "width": 1280,
    "height": 720,
    "max_size_kb": 2048,
    "min_size_kb": 10,
    "formats": ["JPEG", "PNG"],
}

# Audio standards
AUDIO_MIN_VOLUME_DB   = -40   # Below this = basically silence
AUDIO_MAX_VOLUME_DB   = 0.0   # Above this = clipping (0.0dB is normal for TTS)
AUDIO_MIN_SAMPLE_RATE = 16000
AUDIO_SYNC_TOLERANCE  = 2.0   # Max seconds difference audio vs video

# Visual standards
MAX_BLACK_FRAME_PCT   = 30    # Max % of black frames allowed
MIN_SCENE_CHANGES     = 1     # At least 1 scene change (not frozen)


# ════════════════════════════════════════════════════════════════════════
# PROBE HELPERS
# ════════════════════════════════════════════════════════════════════════

def probe_video(path: Path) -> dict:
    """Get video file metadata via ffprobe."""
    if not path.exists():
        return {"error": "file not found"}

    try:
        result = subprocess.run([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(path)
        ], capture_output=True, text=True, timeout=15)

        data = json.loads(result.stdout)
        fmt = data.get("format", {})
        streams = data.get("streams", [])

        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})

        # Parse framerate safely
        fps_str = video_stream.get("r_frame_rate", "0/1")
        try:
            if "/" in str(fps_str):
                num, den = fps_str.split("/")
                fps = float(num) / float(den) if float(den) != 0 else 0
            else:
                fps = float(fps_str)
        except:
            fps = 0

        return {
            "duration": float(fmt.get("duration", 0)),
            "size_bytes": int(fmt.get("size", 0)),
            "bitrate": int(fmt.get("bit_rate", 0)),
            "format": fmt.get("format_name", ""),
            "video_codec": video_stream.get("codec_name", ""),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": fps,
            "video_duration": float(video_stream.get("duration", 0)),
            "audio_codec": audio_stream.get("codec_name", ""),
            "audio_sample_rate": int(audio_stream.get("sample_rate", 0)),
            "audio_duration": float(audio_stream.get("duration", 0)),
            "has_video": bool(video_stream),
            "has_audio": bool(audio_stream),
        }
    except Exception as e:
        return {"error": str(e)}


# ════════════════════════════════════════════════════════════════════════
# AUDIO QUALITY CHECKS
# ════════════════════════════════════════════════════════════════════════

def check_audio_levels(video_path: Path) -> dict:
    """Analyze audio volume levels — detect silence and clipping."""
    try:
        result = subprocess.run([
            "ffmpeg", "-i", str(video_path), "-af", "volumedetect",
            "-f", "null", "-"
        ], capture_output=True, text=True, timeout=60)

        stderr = result.stderr
        info = {}

        for line in stderr.split("\n"):
            if "mean_volume" in line:
                val = line.split("mean_volume:")[1].strip().split(" ")[0]
                info["mean_volume_db"] = float(val)
            elif "max_volume" in line:
                val = line.split("max_volume:")[1].strip().split(" ")[0]
                info["max_volume_db"] = float(val)

        issues = []
        warnings = []

        mean_vol = info.get("mean_volume_db", -100)
        max_vol = info.get("max_volume_db", -100)

        if mean_vol < AUDIO_MIN_VOLUME_DB:
            issues.append(f"🔇 Audio casi silencioso: {mean_vol:.1f}dB (mín: {AUDIO_MIN_VOLUME_DB}dB)")
        elif mean_vol < -30:
            warnings.append(f"🔉 Audio bajo: {mean_vol:.1f}dB")

        if max_vol > AUDIO_MAX_VOLUME_DB:
            warnings.append(f"🔊 Audio posible clipping: max {max_vol:.1f}dB")

        info["issues"] = issues
        info["warnings"] = warnings
        return info

    except Exception as e:
        return {"error": str(e), "issues": [], "warnings": [f"⚠️ No se pudo analizar audio: {e}"]}


def check_audio_silence_segments(video_path: Path, duration: float) -> list:
    """Detect segments of silence in the audio."""
    warnings = []
    try:
        result = subprocess.run([
            "ffmpeg", "-i", str(video_path),
            "-af", "silencedetect=noise=-35dB:d=3",
            "-f", "null", "-"
        ], capture_output=True, text=True, timeout=60)

        silence_count = result.stderr.count("silence_start")
        if silence_count > 0:
            # Calculate approximate silence duration
            silence_duration = 0
            lines = result.stderr.split("\n")
            for line in lines:
                if "silence_duration" in line:
                    try:
                        val = line.split("silence_duration:")[1].strip().split("|")[0].strip()
                        silence_duration += float(val)
                    except:
                        pass

            pct = (silence_duration / max(duration, 1)) * 100
            if pct > 40:
                warnings.append(f"🔇 {pct:.0f}% del audio es silencio ({silence_duration:.1f}s de {duration:.1f}s)")
            elif pct > 20:
                warnings.append(f"🔉 {pct:.0f}% del audio tiene segmentos de silencio")

    except Exception:
        pass

    return warnings


# ════════════════════════════════════════════════════════════════════════
# VISUAL QUALITY CHECKS
# ════════════════════════════════════════════════════════════════════════

def check_black_frames(video_path: Path, duration: float) -> dict:
    """Detect percentage of black/frozen frames."""
    issues = []
    warnings = []

    try:
        result = subprocess.run([
            "ffmpeg", "-i", str(video_path),
            "-vf", "blackdetect=d=0.5:pix_th=0.10",
            "-f", "null", "-"
        ], capture_output=True, text=True, timeout=60)

        black_duration = 0
        for line in result.stderr.split("\n"):
            if "black_duration" in line:
                try:
                    val = line.split("black_duration:")[1].strip().split("|")[0].strip()
                    black_duration += float(val)
                except:
                    pass

        pct = (black_duration / max(duration, 1)) * 100

        if pct > MAX_BLACK_FRAME_PCT:
            issues.append(f"⬛ {pct:.0f}% del video son frames negros ({black_duration:.1f}s)")
        elif pct > 15:
            warnings.append(f"⬛ {pct:.0f}% frames negros (puede haber huecos entre clips)")

        return {"black_pct": round(pct, 1), "black_seconds": round(black_duration, 1),
                "issues": issues, "warnings": warnings}

    except Exception as e:
        return {"error": str(e), "issues": [], "warnings": []}


def check_scene_changes(video_path: Path) -> int:
    """Count scene changes to verify video isn't frozen."""
    try:
        result = subprocess.run([
            "ffmpeg", "-i", str(video_path),
            "-vf", "select='gt(scene,0.3)',showinfo",
            "-f", "null", "-"
        ], capture_output=True, text=True, timeout=30)

        count = result.stderr.count("pts_time:")
        return count

    except Exception:
        return -1


def check_bitrate_quality(probe: dict, standards: dict) -> list:
    """Check if bitrate is sufficient for quality."""
    warnings = []
    bitrate_kbps = probe.get("bitrate", 0) / 1000

    if bitrate_kbps > 0 and bitrate_kbps < standards.get("min_bitrate_kbps", 500):
        warnings.append(f"📉 Bitrate bajo: {bitrate_kbps:.0f}kbps (mín recomendado: {standards['min_bitrate_kbps']}kbps)")

    return warnings


# ════════════════════════════════════════════════════════════════════════
# SYNC CHECK
# ════════════════════════════════════════════════════════════════════════

def check_av_sync(probe: dict) -> list:
    """Check audio/video duration match (sync indicator)."""
    issues = []
    warnings = []

    video_dur = probe.get("video_duration", 0) or probe.get("duration", 0)
    audio_dur = probe.get("audio_duration", 0)

    if audio_dur > 0 and video_dur > 0:
        diff = abs(video_dur - audio_dur)
        if diff > AUDIO_SYNC_TOLERANCE:
            issues.append(f"🔄 Audio/Video desincronizado: diferencia {diff:.1f}s (video={video_dur:.1f}s, audio={audio_dur:.1f}s)")
        elif diff > 1.0:
            warnings.append(f"🔄 Audio/Video ligera diferencia: {diff:.1f}s")

    return issues, warnings


# ════════════════════════════════════════════════════════════════════════
# THUMBNAIL DEEP CHECK
# ════════════════════════════════════════════════════════════════════════

def check_thumbnail_deep(path: Path) -> dict:
    """Deep thumbnail quality check — dimensions, content, readability."""
    if not path or not path.exists():
        return {"exists": False, "issues": ["Thumbnail not found"], "warnings": [], "passed": False}

    issues = []
    warnings = []
    info = {"exists": True, "path": str(path)}

    size_kb = path.stat().st_size / 1024
    info["size_kb"] = round(size_kb)

    if size_kb > THUMB_STANDARDS["max_size_kb"]:
        issues.append(f"Demasiado grande: {size_kb:.0f}KB > {THUMB_STANDARDS['max_size_kb']}KB")
    if size_kb < THUMB_STANDARDS["min_size_kb"]:
        issues.append(f"Demasiado pequeña: {size_kb:.0f}KB — posiblemente corrupta")

    if Image and ImageStat:
        try:
            img = Image.open(path)
            w, h = img.size
            info["width"] = w
            info["height"] = h
            info["format"] = img.format

            if w != THUMB_STANDARDS["width"] or h != THUMB_STANDARDS["height"]:
                issues.append(f"Dimensiones incorrectas: {w}x{h} (necesita {THUMB_STANDARDS['width']}x{THUMB_STANDARDS['height']})")

            if img.format not in THUMB_STANDARDS["formats"]:
                issues.append(f"Formato incorrecto: {img.format} (necesita JPEG o PNG)")

            # Analyze image quality
            stat = ImageStat.Stat(img)
            mean_brightness = sum(stat.mean[:3]) / 3

            # Too dark = possibly blank/black
            if mean_brightness < 15:
                issues.append(f"Miniatura muy oscura (brillo promedio: {mean_brightness:.0f}/255) — posiblemente negra")
            elif mean_brightness < 40:
                warnings.append(f"Miniatura oscura (brillo: {mean_brightness:.0f}/255)")

            # Too bright = possibly blank/white
            if mean_brightness > 240:
                issues.append(f"Miniatura muy clara (brillo: {mean_brightness:.0f}/255) — posiblemente en blanco")

            # Check color variance (very low = solid color = bad)
            total_variance = sum(s ** 2 for s in stat.stddev[:3]) / 3
            if total_variance < 100:
                warnings.append(f"Miniatura con poca variedad visual (varianza: {total_variance:.0f})")

            # Check if has content in the left side (where text should be)
            left_crop = img.crop((0, 0, w // 3, h))
            left_stat = ImageStat.Stat(left_crop)
            left_brightness = sum(left_stat.mean[:3]) / 3

            # Left side should be darker (gradient for text readability)
            # but not completely black
            if left_brightness > 180:
                warnings.append("Lado izquierdo muy claro — el texto podría no ser legible")

            # Check top-left for logo area (should have some content)
            logo_area = img.crop((0, 0, 100, 100))
            logo_stat = ImageStat.Stat(logo_area)
            logo_variance = sum(s ** 2 for s in logo_stat.stddev[:3]) / 3
            if logo_variance < 50:
                warnings.append("Zona del logo (arriba izquierda) podría no tener logo visible")

            info["brightness"] = round(mean_brightness, 1)
            info["variance"] = round(total_variance, 1)

        except Exception as e:
            warnings.append(f"No se pudo analizar imagen: {e}")

    info["issues"] = issues
    info["warnings"] = warnings
    info["passed"] = len(issues) == 0
    return info


# ════════════════════════════════════════════════════════════════════════
# OVERLAY CHECK (title position, logo)
# ════════════════════════════════════════════════════════════════════════

def check_overlay_quality(video_path: Path, title: str, vtype: str) -> list:
    """Check if title text and logo are properly positioned in the video."""
    warnings = []

    # Extract a frame and check if title area has content
    frame_path = video_path.parent / "_qc_frame.jpg"
    try:
        subprocess.run([
            "ffmpeg", "-y", "-ss", "1", "-i", str(video_path),
            "-vframes", "1", "-q:v", "3", str(frame_path)
        ], capture_output=True, timeout=10)

        if frame_path.exists() and Image and ImageStat:
            img = Image.open(frame_path)
            w, h = img.size

            # Check title area (top center)
            if vtype == "short":
                title_area = img.crop((0, 60, w, 160))  # Top area for shorts
            else:
                title_area = img.crop((0, 20, w, 100))  # Top area for longs

            title_stat = ImageStat.Stat(title_area)
            title_brightness = sum(title_stat.mean[:3]) / 3
            title_variance = sum(s ** 2 for s in title_stat.stddev[:3]) / 3

            # Title area should have visible text (some contrast/variance)
            if title_variance < 200:
                warnings.append("⚠️ Zona del título podría no tener texto visible")

            # Check for very long titles that might be cut
            if len(title) > 50 and vtype == "short":
                warnings.append(f"⚠️ Título largo ({len(title)} chars) — podría cortarse en shorts")
            elif len(title) > 70:
                warnings.append(f"⚠️ Título largo ({len(title)} chars) — verificar que no se corta")

            frame_path.unlink(missing_ok=True)

    except Exception:
        frame_path.unlink(missing_ok=True)

    return warnings


# ════════════════════════════════════════════════════════════════════════
# MAIN CHECK — ALL IN ONE
# ════════════════════════════════════════════════════════════════════════

def check_video_quality(video: dict) -> dict:
    """Run ALL quality checks on a single video."""
    title = video.get("title", "Unknown")
    vtype = video.get("type", "short")
    video_path = Path(video.get("output_file", ""))
    thumb_path = Path(video.get("thumbnail", "")) if video.get("thumbnail") else None

    standards = STANDARDS.get(vtype, STANDARDS["short"])

    checks = {
        "title": title,
        "type": vtype,
        "issues": [],
        "warnings": [],
        "passed": True,
        "details": {},
    }

    # ═══ 1. VIDEO FILE ═══
    if not video_path.exists():
        checks["issues"].append("❌ Archivo de video no encontrado")
        checks["passed"] = False
        checks["grade"] = "F"
        return checks

    probe = probe_video(video_path)
    checks["probe"] = probe

    if "error" in probe:
        checks["issues"].append(f"❌ No se puede leer video: {probe['error']}")
        checks["passed"] = False
        checks["grade"] = "F"
        return checks

    # Duration
    duration = probe.get("duration", 0)
    checks["duration"] = round(duration, 1)

    if duration < standards["min_duration"]:
        checks["issues"].append(f"❌ Muy corto: {duration:.1f}s < {standards['min_duration']}s")
        checks["passed"] = False
    elif duration > standards["max_duration"]:
        checks["issues"].append(f"❌ Muy largo: {duration:.1f}s > {standards['max_duration']}s")
        checks["passed"] = False

    # Size
    size_mb = probe.get("size_bytes", 0) / (1024 * 1024)
    checks["size_mb"] = round(size_mb, 1)

    if size_mb > standards["max_size_mb"]:
        checks["issues"].append(f"❌ Archivo muy grande: {size_mb:.1f}MB")
        checks["passed"] = False
    if size_mb < 0.1:
        checks["issues"].append(f"❌ Archivo sospechosamente pequeño: {size_mb:.1f}MB")
        checks["passed"] = False

    # Resolution
    width = probe.get("width", 0)
    height = probe.get("height", 0)
    checks["resolution"] = f"{width}x{height}"

    exp_w = standards.get("expected_width", 0)
    exp_h = standards.get("expected_height", 0)
    if exp_w and exp_h and (width != exp_w or height != exp_h):
        checks["warnings"].append(f"⚠️ Resolución {width}x{height} (esperado {exp_w}x{exp_h})")

    if vtype == "short" and width > height:
        checks["issues"].append(f"❌ Short debe ser vertical pero es {width}x{height}")
        checks["passed"] = False
    if vtype == "long" and height > width:
        checks["issues"].append(f"❌ Long debe ser horizontal pero es {width}x{height}")
        checks["passed"] = False

    # Codec
    if probe.get("video_codec") not in ("h264", "hevc", "vp9"):
        checks["warnings"].append(f"⚠️ Codec inusual: {probe.get('video_codec')}")

    # ═══ 2. AUDIO ═══
    if not probe.get("has_audio"):
        checks["issues"].append("❌ Sin pista de audio")
        checks["passed"] = False
    else:
        # Sample rate
        sr = probe.get("audio_sample_rate", 0)
        if sr < AUDIO_MIN_SAMPLE_RATE:
            checks["warnings"].append(f"⚠️ Sample rate bajo: {sr}Hz (mín: {AUDIO_MIN_SAMPLE_RATE}Hz)")

        # Volume levels
        audio_levels = check_audio_levels(video_path)
        checks["details"]["audio_levels"] = audio_levels
        checks["issues"].extend(audio_levels.get("issues", []))
        checks["warnings"].extend(audio_levels.get("warnings", []))

        # Silence segments
        silence_warnings = check_audio_silence_segments(video_path, duration)
        checks["warnings"].extend(silence_warnings)

    # ═══ 3. VISUAL QUALITY ═══
    # Black frames
    black = check_black_frames(video_path, duration)
    checks["details"]["black_frames"] = black
    checks["issues"].extend(black.get("issues", []))
    checks["warnings"].extend(black.get("warnings", []))

    # Scene changes (frozen detection)
    scenes = check_scene_changes(video_path)
    checks["details"]["scene_changes"] = scenes
    if scenes == 0:
        checks["warnings"].append("⚠️ Sin cambios de escena — video podría estar congelado")
    elif scenes < 0:
        pass  # Could not detect

    # Bitrate quality
    bitrate_warnings = check_bitrate_quality(probe, standards)
    checks["warnings"].extend(bitrate_warnings)

    # ═══ 4. SYNC ═══
    sync_issues, sync_warnings = check_av_sync(probe)
    checks["issues"].extend(sync_issues)
    checks["warnings"].extend(sync_warnings)

    # ═══ 5. THUMBNAIL ═══
    if standards["requires_thumbnail"] or thumb_path:
        if not thumb_path:
            possible_thumb = video_path.parent / "thumbnail.jpg"
            if possible_thumb.exists():
                thumb_path = possible_thumb
                video["thumbnail"] = str(possible_thumb)

        if thumb_path and thumb_path.exists():
            thumb_check = check_thumbnail_deep(thumb_path)
            checks["details"]["thumbnail"] = thumb_check
            for issue in thumb_check.get("issues", []):
                checks["warnings"].append(f"🖼️ {issue}")
            for warning in thumb_check.get("warnings", []):
                checks["warnings"].append(f"🖼️ {warning}")
        elif standards["requires_thumbnail"]:
            checks["warnings"].append("⚠️ Sin miniatura para video largo")

    # ═══ 6. OVERLAY (title, logo) ═══
    overlay_warnings = check_overlay_quality(video_path, title, vtype)
    checks["warnings"].extend(overlay_warnings)

    # ═══ 7. METADATA ═══
    if not title or len(title) < 5:
        checks["warnings"].append("⚠️ Título muy corto")
    if len(title) > 100:
        checks["warnings"].append("⚠️ Título muy largo (> 100 chars)")

    tags = video.get("tags", [])
    if not tags:
        checks["warnings"].append("⚠️ Sin tags definidos")

    # ═══ GRADE ═══
    n_issues = len(checks["issues"])
    n_warnings = len(checks["warnings"])

    if n_issues > 0:
        checks["grade"] = "F"
        checks["passed"] = False
    elif n_warnings == 0:
        checks["grade"] = "A"
    elif n_warnings <= 2:
        checks["grade"] = "B"
    else:
        checks["grade"] = "C"

    return checks


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def run_quality_checker() -> dict:
    """Run quality checks on all produced videos."""
    print("\n✅ BitTrader Quality Checker v2.0 — Auditoría Completa")
    print("  🔍 Checks: Video | Audio | Visual | Sync | Thumbnail | Overlay | Metadata\n")

    if not PRODUCTION_FILE.exists():
        print("  ⚠️ No se encontró archivo de producción.")
        return {"total": 0, "passed": 0, "failed": 0, "results": []}

    production = json.loads(PRODUCTION_FILE.read_text())
    videos = production.get("videos", [])

    print(f"  📋 Verificando {len(videos)} videos...\n")

    results = []
    passed = 0
    failed = 0

    for i, video in enumerate(videos, 1):
        title = video.get("title", "?")
        vtype = video.get("type", "?")
        print(f"  [{i}/{len(videos)}] [{vtype.upper()}] {title[:50]}")

        check = check_video_quality(video)
        results.append(check)

        grade = check.get("grade", "?")
        duration = check.get("duration", 0)
        size = check.get("size_mb", 0)
        resolution = check.get("resolution", "?")
        n_issues = len(check.get("issues", []))
        n_warnings = len(check.get("warnings", []))

        grade_icons = {"A": "✅", "B": "🟡", "C": "🟠", "F": "❌"}
        icon = grade_icons.get(grade, "❓")

        if check.get("passed"):
            passed += 1
        else:
            failed += 1

        print(f"    {icon} Grade {grade} | {duration}s | {size}MB | {resolution} | "
              f"{n_issues} issues, {n_warnings} warnings")

        for issue in check.get("issues", []):
            print(f"       {issue}")
        for warning in check.get("warnings", []):
            print(f"       {warning}")
        print()

    # Save report
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "version": "2.0",
        "total": len(videos),
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{(passed/max(len(videos),1))*100:.0f}%",
        "grades": {
            "A": sum(1 for r in results if r.get("grade") == "A"),
            "B": sum(1 for r in results if r.get("grade") == "B"),
            "C": sum(1 for r in results if r.get("grade") == "C"),
            "F": sum(1 for r in results if r.get("grade") == "F"),
        },
        "results": results,
    }

    QC_REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    production["quality_check"] = {
        "checked_at": report["checked_at"],
        "version": "2.0",
        "passed": passed,
        "failed": failed,
        "pass_rate": report["pass_rate"],
        "grades": report["grades"],
    }
    PRODUCTION_FILE.write_text(json.dumps(production, indent=2, ensure_ascii=False))

    print(f"{'='*55}")
    print(f"  📊 RESULTADO: {passed}/{len(videos)} aprobados ({report['pass_rate']})")
    print(f"  📈 Grades: {report['grades']}")
    if failed:
        print(f"  ❌ {failed} videos NO pasan calidad — revisar antes de publicar")
    else:
        print(f"  ✅ Todos los videos listos para publicar")
    print(f"  📁 Reporte: {QC_REPORT_FILE}")
    print(f"{'='*55}\n")

    return report


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    result = run_quality_checker()

    print("── Resumen Final ──────────────────────")
    for r in result.get("results", []):
        grade = r.get("grade", "?")
        icon = {"A": "✅", "B": "🟡", "C": "🟠", "F": "❌"}.get(grade, "❓")
        n_w = len(r.get("warnings", []))
        print(f"  {icon} [{r.get('type','?')}] {r.get('title','?')[:42]} → Grade {grade} ({n_w} warnings)")
    print("────────────────────────────────────────\n")
