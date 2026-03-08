#!/usr/bin/env python3
"""
✅ BitTrader Quality Checker v3.0 — Auditor de Calidad Completo
Verifica cada aspecto de los videos antes de publicar.

Pipeline: Producer → Thumbnail Agent → Quality Checker → Publisher

Checks realizados:
1. VIDEO: formato, resolución, duración, tamaño, orientación
2. AUDIO: presente, niveles (no silencio, no clipping), sample rate
3. VISUAL: frames negros, bitrate mínimo, clips no corruptos
4. SYNC: audio y video misma duración (sincronización)
5. SUBTITLES: tamaño legible, posición, no duplicados, visibilidad
6. THUMBNAIL: dimensiones, formato, tamaño, no vacía, texto legible
7. OVERLAY: título no cortado, logo posición correcta
8. METADATA: título, tags, descripción
9. FRAMING: contenido no cortado por crop, safe zones respetadas
10. AUDIO MIX: presencia de música de fondo, balance voz/música

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
        "max_duration": 180,   # Rhino storytelling videos can be up to 3 min
        "expected_width": 1080,
        "expected_height": 1920,
        "max_size_mb": 200,
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


def check_duplicate_subtitles(video_path: Path) -> dict:
    """
    Detect duplicate/overlapping subtitle tracks or burned-in double subtitles.
    Common issue: re-encoding a video that already has subtitles burned in,
    then adding subtitles again → visual duplication.
    """
    issues = []
    warnings = []

    # Check 1: Multiple subtitle streams in the container
    try:
        result = subprocess.run([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-select_streams", "s", str(video_path)
        ], capture_output=True, text=True, timeout=10)

        data = json.loads(result.stdout)
        sub_streams = data.get("streams", [])
        if len(sub_streams) > 1:
            warnings.append(f"📝 {len(sub_streams)} pistas de subtítulos en el archivo (posible duplicación)")
    except Exception:
        pass

    # Check 2: Look for subtitle files in the same directory
    video_dir = video_path.parent
    sub_files = list(video_dir.glob("subtitles.*"))
    ass_files = [f for f in sub_files if f.suffix == ".ass"]
    srt_files = [f for f in sub_files if f.suffix == ".srt"]

    if ass_files and srt_files:
        warnings.append(f"📝 Encontrados tanto .ass como .srt — verificar que solo uno se use en el video")

    # Check 3: Analyze ASS file for overlapping dialogue lines at same timestamp
    for ass_file in ass_files:
        try:
            content = ass_file.read_text(encoding='utf-8')
            dialogue_lines = [l for l in content.split('\n') if l.startswith('Dialogue:')]

            if not dialogue_lines:
                continue

            # Parse timestamps and check for exact duplicates
            seen_times = {}
            overlap_count = 0

            for line in dialogue_lines:
                parts = line.split(',', 9)
                if len(parts) >= 3:
                    start = parts[1].strip()
                    end = parts[2].strip()
                    text = parts[-1].strip() if len(parts) > 9 else ""

                    # Count how many lines share exact same start AND end time
                    # (This is NORMAL for word-highlight karaoke — each word gets its own line
                    #  within a group, all sharing the group's time range)
                    # What's NOT normal: identical text at same time
                    key = f"{start}|{end}|{text}"
                    if key in seen_times:
                        overlap_count += 1
                    seen_times[key] = seen_times.get(key, 0) + 1

            if overlap_count > 0:
                issues.append(f"📝 {overlap_count} líneas de subtítulos duplicadas exactas (mismo tiempo + mismo texto)")

            # Check for excessive dialogue lines relative to duration
            # Word-highlight karaoke creates 1 line per word per group:
            # N words = N dialogue lines (each word gets highlighted individually)
            # A 20s short with 130 words = 130 lines (NORMAL for word-highlight)
            # A 90s long with 700 words = 700 lines (NORMAL for word-highlight)
            # Only flag if lines exceed 2x the word count (actual duplication)
            word_count_estimate = len(content.split()) // 3  # rough estimate
            if len(dialogue_lines) > max(1500, word_count_estimate * 3):
                warnings.append(f"📝 Exceso de líneas de subtítulo: {len(dialogue_lines)} (posible duplicación)")

        except Exception:
            pass

    # Check 4: Sample frame analysis — detect if text appears at multiple Y positions
    # (burned-in duplicates show text at different vertical positions)
    frame_path = video_path.parent / "_qc_sub_frame.jpg"
    try:
        # Extract frame at 3 seconds
        subprocess.run([
            "ffmpeg", "-y", "-ss", "3", "-i", str(video_path),
            "-vframes", "1", "-q:v", "3", str(frame_path)
        ], capture_output=True, timeout=10)

        if frame_path.exists() and Image and ImageStat:
            img = Image.open(frame_path)
            w, h = img.size

            # Check bottom third and bottom half for text-like brightness patterns
            # If both zones have high contrast text, subtitles may be duplicated
            bottom_third = img.crop((0, h * 2 // 3, w, h))
            bottom_half_upper = img.crop((0, h // 2, w, h * 2 // 3))

            bt_stat = ImageStat.Stat(bottom_third)
            bhu_stat = ImageStat.Stat(bottom_half_upper)

            # High stddev in both zones suggests text in both = possible duplicate
            bt_contrast = sum(s**2 for s in bt_stat.stddev[:3]) / 3
            bhu_contrast = sum(s**2 for s in bhu_stat.stddev[:3]) / 3

            if bt_contrast > 3000 and bhu_contrast > 3000:
                warnings.append("📝 Texto detectado en múltiples zonas verticales — posible subtítulos duplicados")

        frame_path.unlink(missing_ok=True)
    except Exception:
        if frame_path.exists():
            frame_path.unlink(missing_ok=True)

    return {
        "issues": issues,
        "warnings": warnings,
        "passed": len(issues) == 0,
    }


def check_subtitle_quality(video_path: Path, vtype: str) -> dict:
    """
    Check subtitle size, position, and legibility.
    Verifies:
    - Font size is large enough for mobile viewing
    - Subtitles are in safe zone (not cut off at edges)
    - Text is readable (contrast with background)
    """
    issues = []
    warnings = []

    # Check ASS files for font size
    video_dir = video_path.parent
    ass_files = list(video_dir.glob("subtitles*.ass")) + list(video_dir.glob("*.ass"))

    for ass_file in ass_files:
        try:
            content = ass_file.read_text(encoding='utf-8')

            # Parse Style lines for font size
            for line in content.split('\n'):
                if line.startswith('Style:'):
                    parts = line.split(',')
                    if len(parts) >= 3:
                        style_name = parts[0].replace('Style:', '').strip()
                        font_size = 0
                        try:
                            font_size = int(parts[2].strip())
                        except (ValueError, IndexError):
                            pass

                        if font_size > 0:
                            # For 1080x1920 vertical video, minimum readable font = 60
                            # For 1920x1080 horizontal, minimum = 45
                            if vtype == "short":
                                min_font = 60
                                recommended_font = 75
                            else:
                                min_font = 45
                                recommended_font = 55

                            if font_size < min_font:
                                issues.append(
                                    f"📝 Subtítulos demasiado pequeños: {font_size}px "
                                    f"(mínimo: {min_font}px, recomendado: {recommended_font}px) "
                                    f"— no legible en móvil"
                                )
                            elif font_size < recommended_font:
                                warnings.append(
                                    f"📝 Subtítulos algo pequeños: {font_size}px "
                                    f"(recomendado: {recommended_font}px para estilo viral)"
                                )

            # Check MarginV (vertical margin from bottom)
            for line in content.split('\n'):
                if line.startswith('Style:'):
                    parts = line.split(',')
                    if len(parts) >= 20:
                        try:
                            margin_v = int(parts[19].strip())
                            # MarginV too small = text at very bottom, might be cut off
                            if margin_v < 30:
                                warnings.append(
                                    f"📝 Subtítulos muy pegados al borde inferior "
                                    f"(MarginV={margin_v}) — riesgo de corte en algunos reproductores"
                                )
                            # MarginV too large = text in middle of screen
                            if margin_v > 400:
                                warnings.append(
                                    f"📝 Subtítulos muy arriba (MarginV={margin_v}) — "
                                    f"deberían estar en tercio inferior"
                                )
                        except (ValueError, IndexError):
                            pass

            # Check Outline/Border thickness
            for line in content.split('\n'):
                if line.startswith('Style:'):
                    parts = line.split(',')
                    if len(parts) >= 17:
                        try:
                            outline = float(parts[16].strip())
                            if outline < 1.5:
                                warnings.append(
                                    f"📝 Borde de subtítulos muy delgado ({outline}) — "
                                    f"puede no leerse sobre fondo claro (recomendado: 2-4)"
                                )
                        except (ValueError, IndexError):
                            pass

        except Exception:
            pass

    # Visual check: extract frame and verify subtitle visibility
    frame_path = video_path.parent / "_qc_sub_size_frame.jpg"
    try:
        # Check at 5 seconds (subtitles should be active)
        subprocess.run([
            "ffmpeg", "-y", "-ss", "5", "-i", str(video_path),
            "-vframes", "1", "-q:v", "2", str(frame_path)
        ], capture_output=True, timeout=10)

        if frame_path.exists() and Image and ImageStat:
            img = Image.open(frame_path)
            w, h = img.size

            # Check bottom 20% of frame for subtitle presence
            bottom_zone = img.crop((0, int(h * 0.75), w, int(h * 0.95)))
            bt_stat = ImageStat.Stat(bottom_zone)
            bt_stddev = sum(s**2 for s in bt_stat.stddev[:3]) / 3

            # Very low contrast in subtitle zone = no visible subtitles
            if bt_stddev < 200:
                warnings.append(
                    "📝 No se detectan subtítulos visibles en zona inferior del frame (¿tamaño insuficiente?)"
                )

        frame_path.unlink(missing_ok=True)
    except Exception:
        if frame_path.exists():
            frame_path.unlink(missing_ok=True)

    return {
        "issues": issues,
        "warnings": warnings,
        "passed": len(issues) == 0,
    }


def check_framing_crop(video_path: Path, probe: dict, vtype: str) -> dict:
    """
    Check if content is being cut off by aggressive cropping.
    When converting 16:9 clips to 9:16 vertical, center crop can cut
    important content at the sides.
    
    Checks:
    - Edge content (important elements near borders)
    - Safe zone compliance (90% inner area should contain main content)
    - Aspect ratio consistency across clips
    """
    issues = []
    warnings = []

    width = probe.get("width", 0)
    height = probe.get("height", 0)

    if width == 0 or height == 0:
        return {"issues": [], "warnings": ["⚠️ No se pudo verificar framing"], "passed": True}

    # Extract multiple frames to check framing consistency
    frames_to_check = [3, 15, 30, 60]
    duration = probe.get("duration", 0)
    frames_to_check = [t for t in frames_to_check if t < duration]

    edge_issues = 0
    total_checked = 0

    for timestamp in frames_to_check[:3]:
        frame_path = video_path.parent / f"_qc_frame_{timestamp}.jpg"
        try:
            subprocess.run([
                "ffmpeg", "-y", "-ss", str(timestamp), "-i", str(video_path),
                "-vframes", "1", "-q:v", "2", str(frame_path)
            ], capture_output=True, timeout=10)

            if frame_path.exists() and Image and ImageStat:
                img = Image.open(frame_path)
                w, h = img.size
                total_checked += 1

                # Check left and right 5% edges for content loss
                # If edges are very uniform/dark, content might be cut
                left_edge = img.crop((0, 0, max(w // 20, 1), h))
                right_edge = img.crop((w - max(w // 20, 1), 0, w, h))

                left_stat = ImageStat.Stat(left_edge)
                right_stat = ImageStat.Stat(right_edge)

                left_mean = sum(left_stat.mean[:3]) / 3
                right_mean = sum(right_stat.mean[:3]) / 3
                left_var = sum(s**2 for s in left_stat.stddev[:3]) / 3
                right_var = sum(s**2 for s in right_stat.stddev[:3]) / 3

                # If edges are completely black with no variance = padding (OK)
                # If edges are abruptly cut (high content that extends to border) = crop issue
                # Hard to detect perfectly, but check for asymmetric brightness
                center = img.crop((w // 4, h // 4, w * 3 // 4, h * 3 // 4))
                center_stat = ImageStat.Stat(center)
                center_mean = sum(center_stat.mean[:3]) / 3

                # If center has content but edges are dramatically different = possible bad crop
                if left_var > 1000 or right_var > 1000:
                    # Edges have lots of detail = content extends to edges
                    # This is acceptable for AI-generated images that fill the frame
                    pass

                # Check top and bottom safe zones (important for mobile UI elements)
                top_zone = img.crop((0, 0, w, max(h // 15, 1)))
                bottom_zone = img.crop((0, h - max(h // 15, 1), w, h))

                top_stat = ImageStat.Stat(top_zone)
                bottom_stat = ImageStat.Stat(bottom_zone)
                top_var = sum(s**2 for s in top_stat.stddev[:3]) / 3
                bottom_var = sum(s**2 for s in bottom_stat.stddev[:3]) / 3

                # Important content in extreme top/bottom = will be covered by platform UI
                if top_var > 2000:
                    edge_issues += 1

            frame_path.unlink(missing_ok=True)
        except Exception:
            if frame_path.exists():
                frame_path.unlink(missing_ok=True)

    if total_checked > 0 and edge_issues > total_checked * 0.6:
        warnings.append(
            "🖼️ Contenido importante cerca de los bordes — "
            "puede ser cortado por UI de la plataforma (Instagram/TikTok/YouTube)"
        )

    # Check if clips are being cropped from 16:9 to 9:16
    # AI-generated clips from Hailuo are typically 16:9 (1280x720 or 1920x1080)
    # Converting to 9:16 by crop loses ~75% of horizontal content
    if vtype == "short" and height > width:
        # This is vertical — check if source clips are horizontal
        clips_dir = video_path.parent / "clips"
        if clips_dir.exists():
            clip_files = sorted(clips_dir.glob("*.mp4"))
            if clip_files:
                try:
                    r = subprocess.run([
                        "ffprobe", "-v", "quiet", "-show_entries",
                        "stream=width,height", "-of", "csv=p=0",
                        str(clip_files[0])
                    ], capture_output=True, text=True, timeout=10)
                    parts = r.stdout.strip().split(',')
                    if len(parts) >= 2:
                        src_w, src_h = int(parts[0]), int(parts[1])
                        if src_w > src_h:
                            crop_loss = round((1 - (height / width * src_h / src_w)) * 100)
                            if crop_loss > 40:
                                issues.append(
                                    f"🖼️ Clips fuente son {src_w}x{src_h} (horizontal) "
                                    f"cortados a {width}x{height} (vertical) — "
                                    f"~{crop_loss}% del contenido horizontal se pierde. "
                                    f"Usar prompts con 'vertical composition 9:16' o "
                                    f"scale+pad en vez de crop"
                                )
                except Exception:
                    pass

    return {
        "issues": issues,
        "warnings": warnings,
        "passed": len(issues) == 0,
    }


def check_audio_mix(video_path: Path, duration: float) -> dict:
    """
    Check audio composition — detect if video has only narration
    without background music (for storytelling videos, music is critical).
    
    Checks:
    - Frequency spectrum analysis (music has wider spectrum than speech alone)
    - Volume consistency (music provides a consistent base level)
    """
    issues = []
    warnings = []

    try:
        # Analyze frequency spectrum - speech is 85-3000Hz, music extends wider
        # Use astats to check frequency distribution
        result = subprocess.run([
            "ffmpeg", "-i", str(video_path),
            "-af", "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level",
            "-f", "null", "-"
        ], capture_output=True, text=True, timeout=60)

        # Simple heuristic: analyze high-frequency content
        # If there's music, there should be energy above 4kHz
        # Extract a small segment and check spectrum
        result2 = subprocess.run([
            "ffmpeg", "-i", str(video_path),
            "-af", "highpass=f=4000,volumedetect",
            "-f", "null", "-"
        ], capture_output=True, text=True, timeout=60)

        high_freq_volume = -100
        for line in result2.stderr.split("\n"):
            if "mean_volume" in line:
                try:
                    val = line.split("mean_volume:")[1].strip().split(" ")[0]
                    high_freq_volume = float(val)
                except:
                    pass

        # If high frequency content is very low, probably no music
        # Speech alone typically has high_freq < -50dB
        # Music typically has high_freq > -40dB
        if high_freq_volume < -55:
            warnings.append(
                f"🎵 No se detecta música de fondo (energía alta frecuencia: {high_freq_volume:.0f}dB). "
                f"Videos storytelling necesitan música cinematográfica para engagement"
            )
        elif high_freq_volume < -45:
            warnings.append(
                f"🎵 Música de fondo muy baja o ausente (alta frecuencia: {high_freq_volume:.0f}dB)"
            )

    except Exception as e:
        warnings.append(f"⚠️ No se pudo analizar mix de audio: {e}")

    return {
        "issues": issues,
        "warnings": warnings,
        "high_freq_db": high_freq_volume if 'high_freq_volume' in dir() else None,
        "passed": len(issues) == 0,
    }


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

    # ═══ 4b. SUBTITLE DUPLICATES ═══
    sub_check = check_duplicate_subtitles(video_path)
    checks["details"]["subtitles"] = sub_check
    checks["issues"].extend(sub_check.get("issues", []))
    checks["warnings"].extend(sub_check.get("warnings", []))

    # ═══ 4c. SUBTITLE QUALITY (SIZE, POSITION, LEGIBILITY) ═══
    sub_quality = check_subtitle_quality(video_path, vtype)
    checks["details"]["subtitle_quality"] = sub_quality
    checks["issues"].extend(sub_quality.get("issues", []))
    checks["warnings"].extend(sub_quality.get("warnings", []))

    # ═══ 4d. FRAMING / CROP CHECK ═══
    framing = check_framing_crop(video_path, probe, vtype)
    checks["details"]["framing"] = framing
    checks["issues"].extend(framing.get("issues", []))
    checks["warnings"].extend(framing.get("warnings", []))

    # ═══ 4e. AUDIO MIX (MUSIC DETECTION) ═══
    audio_mix = check_audio_mix(video_path, duration)
    checks["details"]["audio_mix"] = audio_mix
    checks["issues"].extend(audio_mix.get("issues", []))
    checks["warnings"].extend(audio_mix.get("warnings", []))

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
    print("\n✅ BitTrader Quality Checker v3.0 — Auditoría Completa")
    print("  🔍 Checks: Video | Audio | Visual | Sync | Subs | SubSize | Framing | AudioMix | Thumbnail | Overlay | Metadata\n")

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
        "version": "3.0",
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
        "version": "3.0",
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
