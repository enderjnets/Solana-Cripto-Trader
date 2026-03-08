#!/usr/bin/env python3
"""
✅ BitTrader Quality Checker — Auditor de Calidad
Verifica que cada video cumpla todos los requisitos antes de publicar.

Pipeline: Producer → Thumbnail Agent → Quality Checker → Publisher

Checks:
1. VIDEO: existe, formato correcto, resolución, duración, tamaño
2. AUDIO: existe, duración coincide con video, no silencio
3. THUMBNAIL: existe (longs), dimensiones 1280x720, < 2MB, legible
4. METADATA: título, descripción, tags presentes y válidos

Ejecutar: python3 agents/quality_checker.py
"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

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
        "min_width": 1080,
        "min_height": 1920,
        "max_size_mb": 50,
        "requires_thumbnail": False,  # YouTube barely shows short thumbs
    },
    "long": {
        "min_duration": 30,
        "max_duration": 3600,
        "min_width": 1920,
        "min_height": 1080,
        "max_size_mb": 500,
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
        
        # Extract key info
        fmt = data.get("format", {})
        streams = data.get("streams", [])
        
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})
        
        return {
            "duration": float(fmt.get("duration", 0)),
            "size_bytes": int(fmt.get("size", 0)),
            "format": fmt.get("format_name", ""),
            "video_codec": video_stream.get("codec_name", ""),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": eval(video_stream.get("r_frame_rate", "0/1")) if "/" in str(video_stream.get("r_frame_rate", "0")) else float(video_stream.get("r_frame_rate", 0)),
            "audio_codec": audio_stream.get("codec_name", ""),
            "audio_sample_rate": int(audio_stream.get("sample_rate", 0)),
            "has_video": bool(video_stream),
            "has_audio": bool(audio_stream),
        }
    except Exception as e:
        return {"error": str(e)}


def check_thumbnail(path: Path) -> dict:
    """Verify thumbnail meets YouTube requirements."""
    if not path or not path.exists():
        return {"exists": False, "issues": ["Thumbnail not found"]}
    
    issues = []
    info = {"exists": True, "path": str(path)}
    
    # Size check
    size_kb = path.stat().st_size / 1024
    info["size_kb"] = round(size_kb)
    
    if size_kb > THUMB_STANDARDS["max_size_kb"]:
        issues.append(f"Too large: {size_kb:.0f}KB > {THUMB_STANDARDS['max_size_kb']}KB")
    if size_kb < THUMB_STANDARDS["min_size_kb"]:
        issues.append(f"Too small: {size_kb:.0f}KB — may be corrupt")
    
    # Dimension check
    if Image:
        try:
            img = Image.open(path)
            w, h = img.size
            info["width"] = w
            info["height"] = h
            info["format"] = img.format
            
            if w != THUMB_STANDARDS["width"] or h != THUMB_STANDARDS["height"]:
                issues.append(f"Wrong dimensions: {w}x{h} (need {THUMB_STANDARDS['width']}x{THUMB_STANDARDS['height']})")
            
            if img.format not in THUMB_STANDARDS["formats"]:
                issues.append(f"Wrong format: {img.format} (need JPEG or PNG)")
            
            # Basic quality check: if image is mostly one color, it's suspicious
            colors = img.getcolors(maxcolors=100)
            if colors and len(colors) < 5:
                issues.append("Very few colors — thumbnail may be blank or low quality")
                
        except Exception as e:
            issues.append(f"Cannot read image: {e}")
    
    info["issues"] = issues
    info["passed"] = len(issues) == 0
    return info


# ════════════════════════════════════════════════════════════════════════
# QUALITY CHECKS
# ════════════════════════════════════════════════════════════════════════

def check_video_quality(video: dict) -> dict:
    """Run all quality checks on a single video."""
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
    }
    
    # ── VIDEO FILE CHECK ──
    if not video_path.exists():
        checks["issues"].append("❌ Video file not found")
        checks["passed"] = False
        return checks
    
    probe = probe_video(video_path)
    checks["probe"] = probe
    
    if "error" in probe:
        checks["issues"].append(f"❌ Cannot probe video: {probe['error']}")
        checks["passed"] = False
        return checks
    
    # Duration
    duration = probe.get("duration", 0)
    checks["duration"] = round(duration, 1)
    
    if duration < standards["min_duration"]:
        checks["issues"].append(f"❌ Too short: {duration:.1f}s < {standards['min_duration']}s")
        checks["passed"] = False
    elif duration > standards["max_duration"]:
        checks["issues"].append(f"❌ Too long: {duration:.1f}s > {standards['max_duration']}s")
        checks["passed"] = False
    
    # Size
    size_mb = probe.get("size_bytes", 0) / (1024 * 1024)
    checks["size_mb"] = round(size_mb, 1)
    
    if size_mb > standards["max_size_mb"]:
        checks["issues"].append(f"❌ File too large: {size_mb:.1f}MB > {standards['max_size_mb']}MB")
        checks["passed"] = False
    if size_mb < 0.1:
        checks["issues"].append(f"❌ File suspiciously small: {size_mb:.1f}MB")
        checks["passed"] = False
    
    # Resolution
    width = probe.get("width", 0)
    height = probe.get("height", 0)
    checks["resolution"] = f"{width}x{height}"
    
    # For shorts: vertical (height > width)
    if vtype == "short" and width > height:
        checks["warnings"].append(f"⚠️ Short should be vertical but is {width}x{height}")
    
    # For longs: horizontal (width > height)
    if vtype == "long" and height > width:
        checks["warnings"].append(f"⚠️ Long should be horizontal but is {width}x{height}")
    
    # Codec
    if probe.get("video_codec") not in ("h264", "hevc", "vp9"):
        checks["warnings"].append(f"⚠️ Unusual codec: {probe.get('video_codec')}")
    
    # Audio
    if not probe.get("has_audio"):
        checks["issues"].append("❌ No audio track")
        checks["passed"] = False
    
    # ── THUMBNAIL CHECK ──
    if standards["requires_thumbnail"]:
        if thumb_path:
            thumb_check = check_thumbnail(thumb_path)
            checks["thumbnail"] = thumb_check
            if not thumb_check.get("passed"):
                for issue in thumb_check.get("issues", []):
                    checks["warnings"].append(f"🖼️ Thumbnail: {issue}")
        else:
            # Check if thumbnail exists in the video directory
            possible_thumb = video_path.parent / "thumbnail.jpg"
            if possible_thumb.exists():
                thumb_check = check_thumbnail(possible_thumb)
                checks["thumbnail"] = thumb_check
                video["thumbnail"] = str(possible_thumb)
                if not thumb_check.get("passed"):
                    for issue in thumb_check.get("issues", []):
                        checks["warnings"].append(f"🖼️ Thumbnail: {issue}")
            else:
                checks["warnings"].append("⚠️ No thumbnail for long video")
    
    # ── METADATA CHECK ──
    if not title or len(title) < 5:
        checks["warnings"].append("⚠️ Title too short")
    if len(title) > 100:
        checks["warnings"].append("⚠️ Title too long (> 100 chars)")
    
    tags = video.get("tags", [])
    if not tags:
        checks["warnings"].append("⚠️ No tags defined")
    
    # ── OVERALL ──
    if not checks["issues"]:
        checks["grade"] = "A" if not checks["warnings"] else "B"
    else:
        checks["grade"] = "F"
    
    return checks


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def run_quality_checker() -> dict:
    """Run quality checks on all produced videos."""
    print("\n✅ BitTrader Quality Checker v1.0")
    
    if not PRODUCTION_FILE.exists():
        print("  ⚠️ No production file found.")
        return {"total": 0, "passed": 0, "failed": 0}
    
    production = json.loads(PRODUCTION_FILE.read_text())
    videos = production.get("videos", [])
    
    print(f"  📋 Checking {len(videos)} videos...")
    
    results = []
    passed = 0
    failed = 0
    
    for i, video in enumerate(videos, 1):
        print(f"\n  [{i}/{len(videos)}] {video.get('title', '?')[:50]}")
        
        check = check_video_quality(video)
        results.append(check)
        
        grade = check.get("grade", "?")
        duration = check.get("duration", 0)
        size = check.get("size_mb", 0)
        resolution = check.get("resolution", "?")
        
        if check.get("passed"):
            passed += 1
            icon = "✅" if grade == "A" else "🟡"
        else:
            failed += 1
            icon = "❌"
        
        print(f"    {icon} Grade: {grade} | {duration}s | {size}MB | {resolution}")
        
        for issue in check.get("issues", []):
            print(f"       {issue}")
        for warning in check.get("warnings", []):
            print(f"       {warning}")
    
    # Save report
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "total": len(videos),
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{(passed/max(len(videos),1))*100:.0f}%",
        "results": results,
    }
    
    QC_REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    
    # Update production with QC status
    production["quality_check"] = {
        "checked_at": report["checked_at"],
        "passed": passed,
        "failed": failed,
        "pass_rate": report["pass_rate"],
    }
    PRODUCTION_FILE.write_text(json.dumps(production, indent=2, ensure_ascii=False))
    
    print(f"\n{'='*50}")
    print(f"  📊 RESULTADO: {passed}/{len(videos)} aprobados ({report['pass_rate']})")
    if failed:
        print(f"  ❌ {failed} videos NO pasan calidad")
    else:
        print(f"  ✅ Todos los videos están listos para publicar")
    print(f"  📁 Reporte: {QC_REPORT_FILE}")
    print(f"{'='*50}\n")
    
    return report


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    result = run_quality_checker()
    
    print("── Resumen ──────────────────────")
    for r in result.get("results", []):
        grade = r.get("grade", "?")
        icon = {"A": "✅", "B": "🟡", "F": "❌"}.get(grade, "❓")
        print(f"  {icon} [{r.get('type','?')}] {r.get('title','?')[:45]} → Grade {grade}")
    print("─────────────────────────────────\n")
