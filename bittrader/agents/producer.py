#!/usr/bin/env python3
"""
🎬 BitTrader Producer v3.0 — MiniMax TTS + MiniMax Hailuo Video + ffmpeg
Convierte guiones en videos completos:
1. Genera audio con MiniMax TTS (speech-02-hd)
2. Genera clips de video con MiniMax Hailuo 2.3
3. Ensambla video final con ffmpeg
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
import base64
from datetime import datetime, timezone
from pathlib import Path

# Import karaoke subtitles generator
try:
    from karaoke_subs import generate_karaoke_subs
    HAS_KARAOKE = True
except ImportError:
    HAS_KARAOKE = False

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE  = Path("/home/enderj/.openclaw/workspace")
BITTRADER  = WORKSPACE / "bittrader"
DATA_DIR   = BITTRADER / "agents/data"
OUTPUT_DIR = BITTRADER / "agents/output"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GUIONES_FILE    = DATA_DIR / "guiones_latest.json"
PRODUCTION_FILE = DATA_DIR / "production_latest.json"

# ── MiniMax Config ─────────────────────────────────────────────────────────
MINIMAX_KEYS = json.loads((BITTRADER / "keys/minimax.json").read_text())
MINIMAX_API_KEY = MINIMAX_KEYS["minimax_api_key"]
MINIMAX_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_HEADERS = {
    "Authorization": f"Bearer {MINIMAX_API_KEY}",
    "Content-Type": "application/json"
}

# TTS voice: presenter_male is clear and professional
TTS_VOICE = "presenter_male"
TTS_SPEED = 1.0
TTS_MODEL = "speech-02-hd"

# Video generation
VIDEO_MODEL = "MiniMax-Hailuo-2.3"
VIDEO_DURATION = 6  # seconds per clip
VIDEO_POLL_INTERVAL = 10  # seconds between polls
VIDEO_MAX_WAIT = 600  # 10 min max per clip

# ── Fallback TTS (edge-tts, free) ─────────────────────────────────────────
EDGE_TTS_VOICE = "es-MX-JorgeNeural"
EDGE_TTS_RATE = "+10%"

# ── LLM for script operations ─────────────────────────────────────────────
CLAUDE_BASE_URL = "http://127.0.0.1:8443/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-6"

ZAI_CODING_KEY = "863f222d3de340df8b6ff7e1e36ab216.DFOH1veovvWzaoFT"
ZAI_CODING_BASE_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
ZAI_CODING_MODEL = "glm-4.7"


# ════════════════════════════════════════════════════════════════════════
# LLM FUNCTIONS
# ════════════════════════════════════════════════════════════════════════

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
    target_words = int(2.2 * target_seconds)
    system = "Eres editor de guiones para YouTube Shorts en español latino."
    prompt = (f"Acorta este guión a máximo {target_words} palabras (~{target_seconds}s).\n"
              f"Conserva: gancho inicial fuerte, datos clave, CTA final.\n"
              f"RESPONDE SOLO EL GUIÓN, sin comentarios:\n\n{script}")
    return call_llm(prompt, system, max_tokens=400)


def generate_video_prompts(script: dict) -> list:
    """Use LLM to generate video clip prompts from the script."""
    title = script.get("title", "")
    text = script.get("script", "")
    video_type = script.get("type", "short")
    
    num_clips = 3 if video_type == "short" else 5
    aspect = "9:16 vertical" if video_type == "short" else "16:9 horizontal"
    
    system = "Eres director de video para un canal de YouTube de trading y criptomonedas."
    prompt = f"""Genera {num_clips} prompts de video para este guión. Cada clip dura 6 segundos.
Los clips serán generados con IA (MiniMax Hailuo), así que describe escenas visuales cinematográficas.

Título: {title}
Guión: {text[:500]}

Formato de respuesta (JSON array, SIN comentarios):
[
  "prompt del clip 1: descripción visual detallada en inglés, cinematic, professional",
  "prompt del clip 2: ...",
  ...
]

REGLAS:
- Prompts en INGLÉS (la API funciona mejor en inglés)
- Escenas relacionadas con trading, finanzas, criptomonedas, tecnología
- Incluir: lighting, mood, camera angle, details
- NO incluir texto en los clips (los subtítulos van por separado)
"""
    
    result = call_llm(prompt, system, max_tokens=1000)
    if not result:
        return get_fallback_prompts(video_type)
    
    try:
        # Try to parse JSON from result
        json_match = re.search(r'\[.*?\]', result, re.DOTALL)
        if json_match:
            prompts = json.loads(json_match.group())
            if isinstance(prompts, list) and len(prompts) >= 2:
                return prompts[:num_clips]
    except:
        pass
    
    return get_fallback_prompts(video_type)


def get_fallback_prompts(video_type: str) -> list:
    """Fallback video prompts if LLM fails."""
    short_prompts = [
        "Professional trader analyzing cryptocurrency charts on multiple monitors in dark office, green and red candlestick patterns, dramatic blue lighting, close-up of screen data, cinematic depth of field",
        "Bitcoin and cryptocurrency coins floating and rotating in digital space, golden light particles, futuristic blockchain aesthetic, dark background with subtle grid, 4K cinematic quality",
        "Person confidently typing on keyboard, trading platform showing upward green chart trend, modern minimalist office, warm golden hour lighting through window, success atmosphere",
    ]
    long_prompts = [
        "Wide shot of modern trading floor with multiple monitors showing live market data, blue ambient lighting, professional atmosphere, cinematic camera slowly panning",
        "Close-up of hands on keyboard with trading platform on screen showing detailed candlestick charts, risk management panel visible, professional setup, dramatic lighting",
        "Person studying financial charts on tablet, coffee on desk, morning light through window, educational atmosphere, modern workspace with plants",
        "Time-lapse of cryptocurrency price chart going up with green candles, digital numbers changing rapidly, futuristic data visualization, dark background",
        "Person walking confidently in modern city at golden hour, phone showing trading profits notification, success and financial freedom concept, cinematic slow motion",
    ]
    return short_prompts if video_type == "short" else long_prompts


# ════════════════════════════════════════════════════════════════════════
# MINIMAX TTS
# ════════════════════════════════════════════════════════════════════════

def minimax_tts(text: str, output_path: Path) -> float:
    """Generate audio with MiniMax TTS. Returns duration in seconds."""
    payload = {
        "text": text,
        "voice_setting": {
            "voice_id": TTS_VOICE,
            "speed": TTS_SPEED
        },
        "model": TTS_MODEL
    }
    
    try:
        r = requests.post(f"{MINIMAX_BASE_URL}/t2a_v2", 
                         headers=MINIMAX_HEADERS, json=payload, timeout=60)
        data = r.json()
        
        status = data.get("base_resp", {}).get("status_code", -1)
        if status != 0:
            raise Exception(f"MiniMax TTS error: {data.get('base_resp', {}).get('status_msg', 'unknown')}")
        
        audio_b64 = data.get("data", {}).get("audio", "")
        if not audio_b64:
            raise Exception("No audio data in response")
        
        # Decode base64
        padding = 4 - (len(audio_b64) % 4)
        if padding != 4:
            audio_b64 += "=" * padding
        audio_bytes = base64.b64decode(audio_b64)
        
        # Save as raw PCM then convert to MP3
        raw_path = output_path.with_suffix('.raw')
        with open(raw_path, 'wb') as f:
            f.write(audio_bytes)
        
        # Convert raw PCM to MP3 (MiniMax outputs 32000Hz 16-bit mono PCM)
        result = subprocess.run([
            "ffmpeg", "-y", "-f", "s16le", "-ar", "32000", "-ac", "1",
            "-i", str(raw_path), "-b:a", "192k", str(output_path)
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            raise Exception(f"ffmpeg conversion failed: {result.stderr[:200]}")
        
        # Clean up raw
        raw_path.unlink(missing_ok=True)
        
        # Get duration
        dur = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(output_path)],
            capture_output=True, text=True
        )
        duration = float(dur.stdout.strip()) if dur.stdout.strip() else 0
        return duration
        
    except Exception as e:
        print(f"      ⚠️ MiniMax TTS error: {e}")
        return -1


def edge_tts_fallback(text: str, output_path: Path) -> float:
    """Fallback TTS using edge-tts (free Microsoft voices)."""
    import edge_tts
    
    async def _generate():
        communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE, rate=EDGE_TTS_RATE)
        await communicate.save(str(output_path))
    
    asyncio.run(_generate())
    
    dur = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(output_path)],
        capture_output=True, text=True
    )
    return float(dur.stdout.strip()) if dur.stdout.strip() else 0


def generate_audio(text: str, output_path: Path) -> float:
    """Generate audio: MiniMax primary, edge-tts fallback."""
    # For long texts, MiniMax TTS may have limits — split if needed
    duration = minimax_tts(text, output_path)
    if duration > 0:
        return duration
    
    print("      ⚠️ MiniMax TTS falló, usando edge-tts fallback...")
    return edge_tts_fallback(text, output_path)


# ════════════════════════════════════════════════════════════════════════
# MINIMAX VIDEO GENERATION (Hailuo 2.3)
# ════════════════════════════════════════════════════════════════════════

def create_video_task(prompt: str, aspect_ratio: str = "9:16") -> str:
    """Submit video generation task. Returns task_id."""
    payload = {
        "prompt": prompt,
        "model": VIDEO_MODEL,
        "duration": VIDEO_DURATION,
        "resolution": "1080P"
    }
    
    r = requests.post(f"{MINIMAX_BASE_URL}/video_generation",
                     headers=MINIMAX_HEADERS, json=payload, timeout=30)
    data = r.json()
    
    status = data.get("base_resp", {}).get("status_code", -1)
    if status != 0:
        raise Exception(f"Video gen error: {data.get('base_resp', {}).get('status_msg', 'unknown')}")
    
    return data["task_id"]


def poll_video_task(task_id: str) -> str:
    """Poll until video is ready. Returns file_id."""
    elapsed = 0
    while elapsed < VIDEO_MAX_WAIT:
        time.sleep(VIDEO_POLL_INTERVAL)
        elapsed += VIDEO_POLL_INTERVAL
        
        params = {"task_id": task_id}
        r = requests.get(f"{MINIMAX_BASE_URL}/query/video_generation",
                        headers=MINIMAX_HEADERS, params=params, timeout=30)
        data = r.json()
        status = data.get("status", "")
        
        if status == "Success":
            return data.get("file_id", "")
        elif status == "Fail":
            raise Exception(f"Video failed: {data.get('error_message', 'unknown')}")
        
        # Still processing
        mins = elapsed // 60
        secs = elapsed % 60
        print(f"        ⏳ {mins}m{secs}s...")
    
    raise Exception(f"Video timeout after {VIDEO_MAX_WAIT}s")


def download_video_file(file_id: str, output_path: Path) -> bool:
    """Download video from MiniMax file_id."""
    params = {"file_id": file_id}
    r = requests.get(f"{MINIMAX_BASE_URL}/files/retrieve",
                    headers=MINIMAX_HEADERS, params=params, timeout=30)
    data = r.json()
    
    download_url = data.get("file", {}).get("download_url", "")
    if not download_url:
        raise Exception("No download URL in response")
    
    video = requests.get(download_url, timeout=120)
    video.raise_for_status()
    
    with open(output_path, "wb") as f:
        f.write(video.content)
    
    return True


def generate_video_clip(prompt: str, output_path: Path, clip_num: int) -> bool:
    """Generate a single video clip with MiniMax Hailuo."""
    try:
        task_id = create_video_task(prompt)
        print(f"        📡 Clip {clip_num} task: {task_id}")
        
        file_id = poll_video_task(task_id)
        print(f"        ✅ Clip {clip_num} ready")
        
        download_video_file(file_id, output_path)
        size = output_path.stat().st_size / (1024 * 1024)
        print(f"        💾 Clip {clip_num}: {size:.1f}MB")
        return True
        
    except Exception as e:
        print(f"        ❌ Clip {clip_num} error: {e}")
        return False


# ════════════════════════════════════════════════════════════════════════
# VIDEO ASSEMBLY (ffmpeg)
# ════════════════════════════════════════════════════════════════════════

def create_srt_from_text(text: str, duration: float, srt_path: Path):
    """Create SRT subtitle file from text."""
    # Clean text
    lines = text.strip().split('\n')
    display_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('[') or line.startswith('('):
            continue
        line = re.sub(r'[\U00010000-\U0010ffff]', '', line).strip()
        if len(line) > 5:
            display_lines.append(line)
    
    # Split long lines into subtitle chunks (~8 words each)
    chunks = []
    for line in display_lines:
        words = line.split()
        while len(words) > 8:
            chunks.append(' '.join(words[:8]))
            words = words[8:]
        if words:
            chunks.append(' '.join(words))
    
    if not chunks:
        return
    
    time_per = duration / len(chunks)
    srt_lines = []
    
    for i, chunk in enumerate(chunks):
        start = i * time_per
        end = min((i + 1) * time_per, duration - 0.05)
        
        def fmt(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ms = int((t % 1) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        
        srt_lines.extend([str(i+1), f"{fmt(start)} --> {fmt(end)}", chunk, ""])
    
    srt_path.write_text('\n'.join(srt_lines), encoding='utf-8')


def assemble_video(clips: list, audio_path: Path, output_path: Path, 
                   title: str, duration: float, video_type: str,
                   script_text: str, script_dir: Path) -> bool:
    """Assemble clips + audio + karaoke subtitles into final video."""
    
    # Generate karaoke subtitles (ASS) or fallback to SRT
    sub_path = None
    if HAS_KARAOKE:
        try:
            sub_path = generate_karaoke_subs(audio_path, script_text, script_dir, video_type)
        except Exception as e:
            print(f"      ⚠️ Karaoke subs failed: {e}")
    
    if not sub_path or not sub_path.exists():
        # Fallback to basic SRT
        sub_path = script_dir / "subtitles.srt"
        create_srt_from_text(script_text, duration, sub_path)
    
    # Create concat file for clips
    concat_path = script_dir / "concat.txt"
    valid_clips = [c for c in clips if c.exists()]
    
    if not valid_clips:
        print("      ❌ No valid clips to assemble")
        return False
    
    # Calculate how many times to loop clips to fill audio duration
    clip_duration = VIDEO_DURATION  # 6s per clip
    total_clip_time = len(valid_clips) * clip_duration
    loops_needed = max(1, int(duration / total_clip_time) + 1)
    
    concat_lines = []
    for _ in range(loops_needed):
        for clip in valid_clips:
            concat_lines.append(f"file '{clip}'")
    
    concat_path.write_text('\n'.join(concat_lines))
    
    # Resolution based on type
    scale = "1080:1920" if video_type == "short" else "1920:1080"
    
    # Build ffmpeg command
    safe_title = title.replace("'", "'\\''").replace('"', '\\"').replace(':', ' -')
    
    # Step 1: Concat clips and scale
    intermediate = script_dir / "concat_scaled.mp4"
    cmd1 = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_path),
        "-vf", f"scale={scale}:force_original_aspect_ratio=decrease,pad={scale}:(ow-iw)/2:(oh-ih)/2:black",
        "-t", str(duration + 1),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an",
        str(intermediate)
    ]
    
    r1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=300)
    if r1.returncode != 0:
        print(f"      ❌ Concat failed: {r1.stderr[:200]}")
        return False
    
    # Step 2: Add audio + subtitles + title
    subtitle_filter = ""
    if sub_path.exists():
        sub_escaped = str(sub_path).replace("'", "'\\''").replace(":", "\\:")
        if sub_path.suffix == ".ass":
            # ASS karaoke — use native styling (don't force_style)
            subtitle_filter = f",ass='{sub_escaped}'"
        else:
            # SRT fallback — force BitTrader style
            if video_type == "short":
                subtitle_filter = (
                    f",subtitles='{sub_escaped}':force_style="
                    f"'FontName=DejaVu Sans,FontSize=22,PrimaryColour=&H0000FFFF,"
                    f"OutlineColour=&H00000000,Outline=3,Shadow=0,Alignment=2,MarginV=250'"
                )
            else:
                subtitle_filter = (
                    f",subtitles='{sub_escaped}':force_style="
                    f"'FontName=DejaVu Sans,FontSize=20,PrimaryColour=&H0000FFFF,"
                    f"OutlineColour=&H00000000,Outline=3,Shadow=0,Alignment=2,MarginV=60'"
                )
    
    # Title overlay
    title_y = "100" if video_type == "short" else "40"
    title_size = "42" if video_type == "short" else "48"
    
    vf = (f"drawtext=text='{safe_title}':fontcolor=gold:fontsize={title_size}:"
          f"x=(w-text_w)/2:y={title_y}:font=DejaVu Sans Bold:borderw=3:bordercolor=black"
          f"{subtitle_filter}")
    
    cmd2 = [
        "ffmpeg", "-y",
        "-i", str(intermediate),
        "-i", str(audio_path),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path)
    ]
    
    r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=600)
    
    # Cleanup intermediate
    intermediate.unlink(missing_ok=True)
    concat_path.unlink(missing_ok=True)
    
    if r2.returncode != 0:
        print(f"      ❌ Assembly failed: {r2.stderr[:200]}")
        return False
    
    return output_path.exists()


def make_fallback_video(audio_path: Path, output_path: Path, title: str,
                        duration: float, video_type: str, script_text: str,
                        script_dir: Path) -> bool:
    """Fallback: generate video with colored background + text (no AI clips)."""
    
    BG_COLORS = ["#0a0a0a", "#0d1117", "#1a1a2e", "#16213e", "#0f3460"]
    ACCENT_COLORS = ["#FFD700", "#00FF88", "#FF6B35", "#00D4FF", "#FF3366"]
    
    bg = random.choice(BG_COLORS)
    accent = random.choice(ACCENT_COLORS)
    
    # Generate karaoke subtitles or fallback to SRT
    sub_path = None
    if HAS_KARAOKE:
        try:
            sub_path = generate_karaoke_subs(audio_path, script_text, script_dir, video_type)
        except Exception:
            pass
    
    if not sub_path or not sub_path.exists():
        sub_path = script_dir / "subtitles.srt"
        create_srt_from_text(script_text, duration, sub_path)
    
    size = "1080x1920" if video_type == "short" else "1920x1080"
    safe_title = title.replace("'", "'\\''").replace('"', '\\"').replace(':', ' -')
    title_y = "120" if video_type == "short" else "60"
    title_size = "44" if video_type == "short" else "52"
    
    subtitle_filter = ""
    if sub_path.exists():
        sub_escaped = str(sub_path).replace("'", "'\\''").replace(":", "\\:")
        if sub_path.suffix == ".ass":
            subtitle_filter = f",ass='{sub_escaped}'"
        else:
            margin_v = "200" if video_type == "short" else "80"
            subtitle_filter = (
                f",subtitles='{sub_escaped}':force_style="
                f"'FontName=DejaVu Sans,FontSize=24,PrimaryColour=&H0000FFFF,"
                f"OutlineColour=&H00000000,Outline=3,Shadow=0,Alignment=2,MarginV={margin_v}'"
            )
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={bg}:s={size}:d={duration}:r=30",
        "-i", str(audio_path),
        "-vf", (f"drawtext=text='{safe_title}':fontcolor={accent}:fontsize={title_size}:"
                f"x=(w-text_w)/2:y={title_y}:font=DejaVu Sans Bold:borderw=3:bordercolor=black"
                f"{subtitle_filter}"),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.returncode == 0 and output_path.exists()


# ════════════════════════════════════════════════════════════════════════
# MAIN PRODUCTION WORKFLOW
# ════════════════════════════════════════════════════════════════════════

def load_scripts() -> dict:
    if not GUIONES_FILE.exists():
        print("  ⚠️ No hay guiones. Ejecuta creator.py primero.")
        return {}
    return json.loads(GUIONES_FILE.read_text())


def produce_single(script: dict, output_dir: Path, use_ai_video: bool = True) -> dict:
    """Produce a complete video from a script."""
    script_id = script.get("id", f"video_{int(time.time())}")
    title = script.get("title", "BitTrader")
    script_text = script.get("script", "")
    video_type = script.get("type", "short")
    
    # ── STEP 1: Generate Audio ──
    print(f"    🔊 Generando audio ({video_type})...")
    audio_path = output_dir / f"{script_id}.mp3"
    
    try:
        duration = generate_audio(script_text, audio_path)
        if duration <= 0:
            return {"status": "error", "error": "TTS failed — no duration"}
        print(f"    ✅ Audio: {duration:.1f}s")
    except Exception as e:
        print(f"    ❌ Audio error: {e}")
        return {"status": "error", "error": f"TTS: {e}"}
    
    # For shorts > 58s, shorten
    if video_type == "short" and duration > 58:
        print(f"    ⏱️ Short demasiado largo ({duration:.1f}s), acortando...")
        shortened = shorten_script_with_llm(script_text, 55)
        if shortened:
            script_text = shortened
            script["script"] = shortened
            duration = generate_audio(script_text, audio_path)
            print(f"    ✅ Acortado: {duration:.1f}s")
    
    # ── STEP 2: Generate Video ──
    video_path = output_dir / f"{script_id}.mp4"
    
    if use_ai_video:
        print(f"    🎬 Generando clips con MiniMax Hailuo...")
        
        # Generate video prompts with LLM
        prompts = generate_video_prompts(script)
        
        # Generate clips
        clips = []
        for i, prompt in enumerate(prompts, 1):
            clip_path = output_dir / f"clip_{i}.mp4"
            print(f"      🎥 Clip {i}/{len(prompts)}: {prompt[:60]}...")
            success = generate_video_clip(prompt, clip_path, i)
            if success:
                clips.append(clip_path)
            time.sleep(2)  # Rate limit
        
        if clips:
            print(f"    🔧 Ensamblando {len(clips)} clips + audio...")
            success = assemble_video(
                clips, audio_path, video_path, title, duration,
                video_type, script_text, output_dir
            )
            if success:
                size_mb = video_path.stat().st_size / (1024 * 1024)
                print(f"    ✅ Video final: {size_mb:.1f}MB")
            else:
                print(f"    ⚠️ Ensamble falló, usando fallback...")
                success = make_fallback_video(
                    audio_path, video_path, title, duration,
                    video_type, script_text, output_dir
                )
        else:
            print(f"    ⚠️ No se generaron clips, usando fallback...")
            success = make_fallback_video(
                audio_path, video_path, title, duration,
                video_type, script_text, output_dir
            )
    else:
        # Text-only fallback
        success = make_fallback_video(
            audio_path, video_path, title, duration,
            video_type, script_text, output_dir
        )
    
    if not success or not video_path.exists():
        return {"status": "error", "error": "Video generation failed completely"}
    
    size_mb = video_path.stat().st_size / (1024 * 1024)
    
    return {
        "script_id": script_id,
        "title": title,
        "type": video_type,
        "status": "success",
        "audio_file": str(audio_path),
        "output_file": str(video_path),
        "duration": round(duration, 1),
        "size_mb": round(size_mb, 1),
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "ai_video": use_ai_video,
    }


def run_producer(limit: int = 5, use_ai_video: bool = True) -> dict:
    print(f"\n🎬 BitTrader Producer v3.0 — MiniMax TTS + {'Hailuo Video' if use_ai_video else 'Text Fallback'}")
    print(f"  🔊 Voz: {TTS_VOICE} (MiniMax) | fallback: {EDGE_TTS_VOICE}")
    
    guiones = load_scripts()
    scripts = guiones.get("scripts", [])
    
    pending = [s for s in scripts if s.get("status") == "pending"]
    print(f"  📋 {len(pending)} guiones pendientes de {len(scripts)} total")
    
    if not pending:
        print("  ✅ No hay guiones pendientes")
        return {"processed": 0, "errors": 0, "videos": []}
    
    to_process = pending[:limit]
    print(f"  🎬 Procesando {len(to_process)} guiones...")
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    day_dir = OUTPUT_DIR / date_str
    day_dir.mkdir(parents=True, exist_ok=True)
    
    videos = []
    errors = 0
    
    for i, script in enumerate(to_process, 1):
        script_id = script.get("id", f"v_{int(time.time())}")
        print(f"\n  [{i}/{len(to_process)}] 📝 {script.get('title', '?')[:55]}")
        
        script_dir = day_dir / script_id
        script_dir.mkdir(parents=True, exist_ok=True)
        
        result = produce_single(script, script_dir, use_ai_video)
        videos.append(result)
        
        if result.get("status") == "success":
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
    
    # Save
    guiones["scripts"] = scripts
    GUIONES_FILE.write_text(json.dumps(guiones, indent=2, ensure_ascii=False))
    
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
    parser = argparse.ArgumentParser(description="BitTrader Producer v3.0")
    parser.add_argument("--limit", type=int, default=5, help="Max scripts to process")
    parser.add_argument("--no-ai-video", action="store_true", help="Skip AI video gen (text fallback only)")
    args = parser.parse_args()
    
    result = run_producer(limit=args.limit, use_ai_video=not args.no_ai_video)
    
    print("\n── Producción ──────────────────────")
    for v in result.get("videos", []):
        status = "✅" if v.get("status") == "success" else "❌"
        print(f"  {status} [{v.get('type','?')}] {v.get('title','?')[:50]} | "
              f"{v.get('duration',0):.1f}s | {v.get('size_mb',0)}MB")
    print("─────────────────────────────────────────\n")
