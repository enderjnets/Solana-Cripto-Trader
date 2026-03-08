#!/usr/bin/env python3
"""
🎬 BitTrader Producer — Productor Ejecutivo
Convierte guiones en videos completos:
1. Genera audio con edge-tts (gratuito, Microsoft Neural voices)
2. Genera video con ffmpeg (texto animado + fondo)
3. Ensambla video final (mp4 listo para YouTube)
"""
import sys
import json
import os
import re
import asyncio
import subprocess
import requests
import time
import random
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE  = Path("/home/enderj/.openclaw/workspace")
BITTRADER  = WORKSPACE / "bittrader"
DATA_DIR   = BITTRADER / "agents/data"
OUTPUT_DIR = BITTRADER / "agents/output"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GUIONES_FILE  = DATA_DIR / "guiones_latest.json"
PRODUCTION_FILE = DATA_DIR / "production_latest.json"

# ── TTS Config ─────────────────────────────────────────────────────────────
TTS_VOICE = "es-MX-JorgeNeural"  # Neutral Latin American male
TTS_RATE = "+10%"  # Slightly faster for YouTube energy

# ── Video Config ───────────────────────────────────────────────────────────
# Background colors for variety
BG_COLORS = [
    "#0a0a0a",  # Near black
    "#0d1117",  # GitHub dark
    "#1a1a2e",  # Dark navy
    "#16213e",  # Deep blue
    "#0f3460",  # Medium blue
    "#1b1b2f",  # Dark purple
]

# Accent colors for text highlights
ACCENT_COLORS = [
    "#FFD700",  # Gold
    "#00FF88",  # Green
    "#FF6B35",  # Orange
    "#00D4FF",  # Cyan
    "#FF3366",  # Pink
    "#A855F7",  # Purple
]

# Logo path
LOGO_PATH = WORKSPACE / "videos/BIBLIOTECA/bittrader_logo.png"

# ── Claude / LLM (for script shortening) ───────────────────────────────────
CLAUDE_BASE_URL = "http://127.0.0.1:8443/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-6"

ZAI_CODING_KEY = "863f222d3de340df8b6ff7e1e36ab216.DFOH1veovvWzaoFT"
ZAI_CODING_BASE_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
ZAI_CODING_MODEL = "glm-4.7"


def call_claude(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    data = {"model": CLAUDE_MODEL, "max_tokens": max_tokens, "messages": []}
    if system:
        data["messages"].append({"role": "user", "content": system})
    data["messages"].append({"role": "user", "content": prompt})
    try:
        r = requests.post(CLAUDE_BASE_URL, headers=headers, json=data, timeout=120)
        r.raise_for_status()
        return r.json().get("content", [{}])[0].get("text", "")
    except Exception as e:
        print(f"      ⚠️ Claude error: {e}")
        return None


def call_glm_4_7(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ZAI_CODING_KEY}"}
    messages = []
    if system:
        messages.append({"role": "user", "content": system})
    messages.append({"role": "user", "content": prompt})
    data = {"model": ZAI_CODING_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.7}
    try:
        r = requests.post(ZAI_CODING_BASE_URL, headers=headers, json=data, timeout=120)
        r.raise_for_status()
        resp = r.json()
        if "choices" in resp and len(resp["choices"]) > 0:
            msg = resp["choices"][0]["message"]
            return msg.get("content", "") or msg.get("reasoning_content", "")
        return str(resp)
    except Exception as e:
        print(f"      ⚠️ GLM-4.7 error: {e}")
        return None


def call_llm(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    result = call_claude(prompt, system, max_tokens)
    if result:
        return result
    print("      ⚠️ Claude falló, intentando GLM-4.7...")
    result = call_glm_4_7(prompt, system, max_tokens)
    if result:
        return result
    print("      ❌ Todos los LLM fallaron")
    return None


def shorten_script_with_llm(script: str, target_seconds: int = 55) -> str:
    target_words = int(2.2 * target_seconds)  # ~2.2 words/sec in Spanish
    system = "Eres editor de guiones para YouTube Shorts en español latino."
    prompt = (f"Acorta este guión a máximo {target_words} palabras (~{target_seconds}s).\n"
              f"Conserva: gancho inicial fuerte, datos clave, CTA final.\n"
              f"RESPONDE SOLO EL GUIÓN, sin comentarios:\n\n{script}")
    return call_llm(prompt, system, max_tokens=400)


# ════════════════════════════════════════════════════════════════════════
# TTS — edge-tts (gratuito, Microsoft Neural voices)
# ════════════════════════════════════════════════════════════════════════

async def generate_audio_async(text: str, output_path: Path, voice: str = TTS_VOICE) -> float:
    """Genera audio MP3 con edge-tts. Retorna duración en segundos."""
    import edge_tts

    communicate = edge_tts.Communicate(text, voice, rate=TTS_RATE)
    await communicate.save(str(output_path))

    # Get duration with ffprobe
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(output_path)],
        capture_output=True, text=True
    )
    duration = float(result.stdout.strip()) if result.stdout.strip() else 0
    return duration


def generate_audio(text: str, output_path: Path, voice: str = TTS_VOICE) -> float:
    """Sync wrapper for async TTS."""
    return asyncio.run(generate_audio_async(text, output_path, voice))


# ════════════════════════════════════════════════════════════════════════
# VIDEO GENERATION — ffmpeg (text animation + background)
# ════════════════════════════════════════════════════════════════════════

def clean_script_for_display(script: str) -> list:
    """Extract key phrases from script for on-screen display."""
    # Remove stage directions, emojis, etc
    lines = script.strip().split('\n')
    display_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('[') or line.startswith('('):
            continue
        # Remove emojis and clean
        line = re.sub(r'[\U00010000-\U0010ffff]', '', line)
        line = line.strip()
        if len(line) > 10:
            display_lines.append(line)
    return display_lines


def generate_short_video(script: dict, audio_path: Path, output_path: Path, duration: float) -> bool:
    """Generate a YouTube Short (9:16 vertical) with text animation."""
    title = script.get("title", "BitTrader")
    text_lines = clean_script_for_display(script.get("script", ""))

    bg_color = random.choice(BG_COLORS)
    accent = random.choice(ACCENT_COLORS)

    # Create subtitle file for text display
    srt_path = output_path.parent / "subtitles.srt"
    create_srt_from_lines(text_lines, duration, srt_path)

    # Build ffmpeg command for Short (1080x1920)
    filter_complex = []

    # Background
    filter_complex.append(f"color=c={bg_color}:s=1080x1920:d={duration}:r=30[bg]")

    # Title text at top
    safe_title = title.replace("'", "'\\''").replace('"', '\\"').replace(':', '\\:')
    filter_complex.append(
        f"[bg]drawtext=text='{safe_title}':fontcolor={accent}:fontsize=48:"
        f"x=(w-text_w)/2:y=120:font='DejaVu Sans':borderw=3:bordercolor=black[titled]"
    )

    # BitTrader watermark
    filter_complex.append(
        f"[titled]drawtext=text='@BitTrader':fontcolor=white@0.5:fontsize=28:"
        f"x=(w-text_w)/2:y=h-80:font='DejaVu Sans'[watermarked]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={bg_color}:s=1080x1920:d={duration}:r=30",
        "-i", str(audio_path),
    ]

    # Add subtitles
    if srt_path.exists():
        subtitle_filter = (
            f"drawtext=text='{safe_title}':fontcolor={accent}:fontsize=44:"
            f"x=(w-text_w)/2:y=120:font=DejaVu Sans Bold:borderw=3:bordercolor=black,"
            f"subtitles={srt_path}:force_style="
            f"'FontName=DejaVu Sans,FontSize=24,PrimaryColour=&H00FFFFFF,"
            f"OutlineColour=&H00000000,Outline=2,Shadow=1,Alignment=2,MarginV=200'"
        )
        cmd.extend([
            "-vf", subtitle_filter,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path)
        ])
    else:
        cmd.extend([
            "-vf", f"drawtext=text='{safe_title}':fontcolor={accent}:fontsize=44:"
                   f"x=(w-text_w)/2:y=120:font=DejaVu Sans Bold:borderw=3:bordercolor=black",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path)
        ])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.returncode == 0


def generate_long_video(script: dict, audio_path: Path, output_path: Path, duration: float) -> bool:
    """Generate a YouTube long video (16:9 horizontal) with text overlay."""
    title = script.get("title", "BitTrader")
    text_lines = clean_script_for_display(script.get("script", ""))

    bg_color = random.choice(BG_COLORS)
    accent = random.choice(ACCENT_COLORS)

    # Create SRT subtitles
    srt_path = output_path.parent / "subtitles.srt"
    create_srt_from_lines(text_lines, duration, srt_path)

    safe_title = title.replace("'", "'\\''").replace('"', '\\"').replace(':', '\\:')

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={bg_color}:s=1920x1080:d={duration}:r=30",
        "-i", str(audio_path),
    ]

    if srt_path.exists():
        subtitle_filter = (
            f"drawtext=text='{safe_title}':fontcolor={accent}:fontsize=52:"
            f"x=(w-text_w)/2:y=60:font=DejaVu Sans Bold:borderw=3:bordercolor=black,"
            f"subtitles={srt_path}:force_style="
            f"'FontName=DejaVu Sans,FontSize=22,PrimaryColour=&H00FFFFFF,"
            f"OutlineColour=&H00000000,Outline=2,Shadow=1,Alignment=2,MarginV=80'"
        )
        cmd.extend([
            "-vf", subtitle_filter,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path)
        ])
    else:
        cmd.extend([
            "-vf", f"drawtext=text='{safe_title}':fontcolor={accent}:fontsize=52:"
                   f"x=(w-text_w)/2:y=60:font=DejaVu Sans Bold:borderw=3:bordercolor=black",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path)
        ])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return result.returncode == 0


def create_srt_from_lines(lines: list, total_duration: float, srt_path: Path):
    """Create SRT subtitle file from text lines evenly distributed."""
    if not lines:
        return

    # Split long lines
    split_lines = []
    for line in lines:
        words = line.split()
        # Max ~8 words per subtitle
        while len(words) > 8:
            split_lines.append(' '.join(words[:8]))
            words = words[8:]
        if words:
            split_lines.append(' '.join(words))

    if not split_lines:
        return

    time_per_line = total_duration / len(split_lines)
    srt_content = []

    for i, line in enumerate(split_lines):
        start = i * time_per_line
        end = min((i + 1) * time_per_line, total_duration - 0.1)

        start_h = int(start // 3600)
        start_m = int((start % 3600) // 60)
        start_s = int(start % 60)
        start_ms = int((start % 1) * 1000)

        end_h = int(end // 3600)
        end_m = int((end % 3600) // 60)
        end_s = int(end % 60)
        end_ms = int((end % 1) * 1000)

        srt_content.append(f"{i+1}")
        srt_content.append(f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms:03d} --> "
                          f"{end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms:03d}")
        srt_content.append(line)
        srt_content.append("")

    srt_path.write_text('\n'.join(srt_content), encoding='utf-8')


# ════════════════════════════════════════════════════════════════════════
# MAIN PRODUCTION WORKFLOW
# ════════════════════════════════════════════════════════════════════════

def load_scripts() -> dict:
    if not GUIONES_FILE.exists():
        print("  ⚠️ No hay guiones. Ejecuta creator.py primero.")
        return {}
    return json.loads(GUIONES_FILE.read_text())


def produce_single(script: dict, output_dir: Path) -> dict:
    """Produce un video completo a partir de un guión."""
    script_id = script.get("id", f"video_{int(time.time())}")
    title = script.get("title", "BitTrader")
    script_text = script.get("script", "")
    video_type = script.get("type", "short")

    print(f"    🔊 Generando audio ({video_type})...")

    # 1. Generate audio
    audio_path = output_dir / f"{script_id}.mp3"
    try:
        duration = generate_audio(script_text, audio_path)
        print(f"    ✅ Audio: {duration:.1f}s → {audio_path.name}")
    except Exception as e:
        print(f"    ❌ Audio error: {e}")
        return {"status": "error", "error": f"TTS failed: {e}"}

    # 2. For shorts, check duration and shorten if needed
    if video_type == "short" and duration > 58:
        print(f"    ⏱️ Short demasiado largo ({duration:.1f}s), acortando...")
        shortened = shorten_script_with_llm(script_text, 55)
        if shortened:
            script_text = shortened
            script["script"] = shortened
            duration = generate_audio(script_text, audio_path)
            print(f"    ✅ Acortado: {duration:.1f}s")

    # 3. Generate video
    video_path = output_dir / f"{script_id}.mp4"
    print(f"    🎬 Generando video...")

    try:
        if video_type == "short":
            success = generate_short_video(script, audio_path, video_path, duration)
        else:
            success = generate_long_video(script, audio_path, video_path, duration)

        if success and video_path.exists():
            size_mb = video_path.stat().st_size / (1024 * 1024)
            print(f"    ✅ Video: {size_mb:.1f}MB → {video_path.name}")
        else:
            print(f"    ❌ Video generation failed")
            return {"status": "error", "error": "ffmpeg failed"}
    except Exception as e:
        print(f"    ❌ Video error: {e}")
        return {"status": "error", "error": f"Video gen failed: {e}"}

    return {
        "script_id": script_id,
        "title": title,
        "type": video_type,
        "status": "success",
        "audio_file": str(audio_path),
        "output_file": str(video_path),
        "duration": duration,
        "size_mb": round(video_path.stat().st_size / (1024 * 1024), 1),
        "produced_at": datetime.now(timezone.utc).isoformat(),
    }


def run_producer(limit: int = 5) -> dict:
    print("\n🎬 BitTrader Producer v2.0 — edge-tts + ffmpeg")
    print(f"  🔊 Voz: {TTS_VOICE}")

    guiones = load_scripts()
    scripts = guiones.get("scripts", [])

    pending = [s for s in scripts if s.get("status") == "pending"]
    print(f"  📋 {len(pending)} guiones pendientes de {len(scripts)} total")

    if not pending:
        print("  ✅ No hay guiones pendientes")
        return {"processed": 0, "errors": 0, "videos": []}

    to_process = pending[:limit]
    print(f"  🎬 Procesando {len(to_process)} guiones...")

    # Create date-based output directory
    date_str = datetime.now().strftime("%Y-%m-%d")
    day_dir = OUTPUT_DIR / date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    videos = []
    errors = 0

    for i, script in enumerate(to_process, 1):
        script_id = script.get("id", f"v_{int(time.time())}")
        print(f"\n  [{i}/{len(to_process)}] 📝 {script.get('title', '?')[:55]}")

        # Create script-specific directory
        script_dir = day_dir / script_id
        script_dir.mkdir(parents=True, exist_ok=True)

        result = produce_single(script, script_dir)
        videos.append(result)

        if result.get("status") == "success":
            # Update script status
            script["status"] = "produced"
            script["produced_at"] = result.get("produced_at")
            script["output_file"] = result.get("output_file")
            script["audio_file"] = result.get("audio_file")
            script["video_duration"] = result.get("duration")
        else:
            script["status"] = "error"
            script["error"] = result.get("error", "unknown")
            errors += 1

        time.sleep(1)

    # Save updated guiones
    guiones["scripts"] = scripts
    GUIONES_FILE.write_text(json.dumps(guiones, indent=2, ensure_ascii=False))

    # Save production result
    production = {
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "date": date_str,
        "videos": videos,
        "stats": {
            "total": len(videos),
            "success": sum(1 for v in videos if v.get("status") == "success"),
            "errors": errors,
        }
    }
    PRODUCTION_FILE.write_text(json.dumps(production, indent=2, ensure_ascii=False))

    # Also save date-specific
    (DATA_DIR / f"production_{date_str}.json").write_text(
        json.dumps(production, indent=2, ensure_ascii=False))

    print(f"\n✅ Producer completado")
    print(f"   🎬 {production['stats']['success']} videos generados")
    if errors:
        print(f"   ⚠️  {errors} errores")
    print(f"   📁 Output: {day_dir}")

    return production


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BitTrader Producer v2.0")
    parser.add_argument("--limit", type=int, default=5, help="Max scripts to process")
    args = parser.parse_args()

    result = run_producer(limit=args.limit)

    print("\n── Producción ──────────────────────")
    for v in result.get("videos", []):
        status = "✅" if v.get("status") == "success" else "❌"
        print(f"  {status} [{v.get('type','?')}] {v.get('title','?')[:50]} | {v.get('duration',0):.1f}s | {v.get('size_mb',0)}MB")
    print("─────────────────────────────────────────\n")
