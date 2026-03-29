#!/usr/bin/env python3
"""
🎬 BitTrader Producer v4.0 — Edge-TTS + MiniMax Hailuo Video + ffmpeg
Convierte guiones en videos completos:
1. Genera audio con Edge-TTS Jorge (es-MX-JorgeNeural) — MiniMax TTS deshabilitado (genera estática)
2. Genera clips de video con MiniMax Hailuo 2.3
3. Ensambla video final con ffmpeg

Mejoras v4.0:
- TTS: Edge-TTS Jorge como primario (MiniMax genera estática para español)
- Shorts: blur-fill en vez de barras negras (Hailuo no genera vertical nativo)
- Logo BitTrader: overlay top-right, 240px (shorts) / 180px (longs)
- Subtítulos: 100px (shorts) / 65px (longs) — aumentado para estilo viral
- Video: aspect_ratio 9:16 pasado a Hailuo (sin efecto pero documentado)
- Audio: 44100Hz AAC
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

# ── ComfyUI Config (Mac M3 - primary for video/image gen) ─────────────────
MAC_M3_URL = "http://10.0.0.232:8188"  # MacBook Air con ComfyUI
COMFYUI_TIMEOUT = 300  # 5 min max per video clip
USE_COMFYUI_FIRST = True  # Try ComfyUI before MiniMax Hailuo

# ── MiniMax Config (fallback) ──────────────────────────────────────────────
MINIMAX_KEYS = json.loads((BITTRADER / "keys/minimax.json").read_text())
MINIMAX_API_KEY = MINIMAX_KEYS["minimax_api_key"]
MINIMAX_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_HEADERS = {
    "Authorization": f"Bearer {MINIMAX_API_KEY}",
    "Content-Type": "application/json"
}

# TTS voice: Edge-TTS Jorge is the primary (MiniMax presenter_male generates static noise)
# MiniMax TTS is DISABLED — generates corrupted audio for Spanish text
TTS_VOICE = "presenter_male"  # kept for reference, NOT used
TTS_PRIMARY = "edge"  # Use edge-tts as primary
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

# ── Coin logos ─────────────────────────────────────────────────────────────
COIN_LOGOS_DIR = DATA_DIR / "coin_logos"
COIN_LOGOS_DIR.mkdir(parents=True, exist_ok=True)

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


# ── BitTrader Rhino Character — consistent visual identity ────────────────
RHINO_BASE_SHORT = (
    "anthropomorphic rhinoceros character, hyper-realistic 3D render, "
    "muscular but elegant, wearing modern casual trading clothes, "
    "expressive face, dramatic cinematic lighting, "
    "ultra HD, photorealistic textures, 9:16 vertical aspect ratio, dark moody background"
)
RHINO_BASE_LONG = (
    "anthropomorphic rhinoceros character, hyper-realistic 3D render, "
    "muscular but elegant, wearing modern casual trading clothes, "
    "expressive face, dramatic cinematic lighting, "
    "ultra HD, photorealistic textures, 16:9 widescreen, dark moody background"
)


def inject_rhino(prompt: str, video_type: str) -> str:
    """Ensure every video prompt features the BitTrader rhino mascot."""
    base = RHINO_BASE_SHORT if video_type == "short" else RHINO_BASE_LONG
    p = prompt.strip()
    if "rhinoceros" not in p.lower() and "rhino" not in p.lower():
        p = f"Anthropomorphic rhinoceros — {p}, {base}"
    elif "hyper-realistic" not in p.lower() and "3d render" not in p.lower():
        p = f"{p}, {base}"
    return p


def generate_video_prompts(script: dict) -> list:
    """Generate per-segment video prompts with narrator-sync and zero repetition.

    FIX v4.1:
    - Splits long scripts into N timed segments (one prompt per segment)
    - Each prompt is generated from the ACTUAL NARRATION TEXT for that segment
    - Ensures semantic sync: visual matches what narrator is saying at that moment
    - All prompts are unique — no repetition tracking needed (different text → different scene)
    - For shorts: 3 segment-synced prompts
    - For longs: 1 prompt per ~60s segment (8-12 prompts for a 10-min video)
    """
    title = script.get("title", "")
    text = script.get("script", "")
    video_type = script.get("type", "short")
    existing_prompts = script.get("video_prompts", [])

    # Use pre-generated prompts from creator.py if available (already have rhino injected)
    if existing_prompts:
        # Deduplicate existing prompts before using them
        seen = set()
        unique_prompts = []
        for p in existing_prompts:
            key = p.strip().lower()[:80]
            if key not in seen:
                seen.add(key)
                unique_prompts.append(p)
        return [inject_rhino(p, video_type) for p in unique_prompts]

    aspect = "9:16 vertical" if video_type == "short" else "16:9 horizontal"

    # ── Split script into timed segments for narrator sync ────────────────
    # For shorts: 3 segments; for longs: 1 per ~60s (estimate ~130 wpm)
    words = text.split()
    word_count = len(words)
    words_per_min = 130
    total_min = word_count / words_per_min

    if video_type == "short":
        num_segments = 3
    else:
        # 1 prompt per 60s, min 4, max 16
        num_segments = max(4, min(16, int(total_min) + 1))

    # Split text evenly into segments
    seg_size = max(1, len(words) // num_segments)
    segments = []
    for i in range(num_segments):
        start = i * seg_size
        end = start + seg_size if i < num_segments - 1 else len(words)
        seg_text = " ".join(words[start:end])
        segments.append(seg_text)

    system = (
        "You are a video director for BitTrader YouTube channel. "
        "The channel mascot is an anthropomorphic rhinoceros — a confident, modern trader. "
        "EVERY scene must feature this rhino character as the main subject. "
        "CRITICAL: Each prompt MUST visually represent what the narrator is SAYING in that segment. "
        "Never repeat the same scene description — each clip must be visually distinct."
    )

    segments_json = json.dumps(
        [{"segment": i+1, "narrator_text": seg[:200]} for i, seg in enumerate(segments)],
        ensure_ascii=False
    )

    prompt = f"""Generate {num_segments} video clip prompts for this YouTube video.
Each clip is 6 seconds. CRITICAL: each prompt must visually match what the narrator says in that segment.

Title: {title}
Video type: {video_type} ({aspect})

Narrator segments (each becomes one clip):
{segments_json}

Response format (JSON array of {num_segments} strings, NO comments):
[
  "Anthropomorphic rhinoceros [action matching segment 1 narration], [setting], [mood], hyper-realistic 3D render, {aspect}, dramatic cinematic lighting, dark background",
  ...
]

SYNC RULES (mandatory):
- If narrator says "Bitcoin subiendo" → show rhino at trading desk with green charts rising
- If narrator says "empezar con $100" → show rhino holding dollar bills or mobile wallet
- If narrator says "riesgo/pérdida" → show rhino looking at red chart, concerned expression
- If narrator says "estrategia DCA" → show rhino marking a calendar, systematic approach
- If narrator says "error/mistake" → show rhino face-palming or looking at phone in shock
- If narrator says "exchange/Coinbase" → show rhino using phone/laptop with exchange UI

ANTI-REPETITION RULES (mandatory):
- NO two prompts can have the same setting (trading desk, city, phone, etc.)
- NO two prompts can have the same rhino action
- NO two prompts can have the same lighting/mood
- Vary: indoor/outdoor, day/night, close-up/wide shot, action/contemplation

OUTPUT: JSON array only, no other text.
"""

    result = call_llm(prompt, system, max_tokens=num_segments * 120 + 200)
    if result:
        try:
            json_match = re.search(r'\[.*?\]', result, re.DOTALL)
            if json_match:
                prompts = json.loads(json_match.group())
                if isinstance(prompts, list) and len(prompts) >= 2:
                    # Final dedup pass — ensure no two prompts are too similar
                    unique = _dedup_prompts(prompts[:num_segments])
                    return [inject_rhino(p, video_type) for p in unique]
        except Exception:
            pass

    # Fallback: generate segment-specific prompts without LLM
    return _segment_fallback_prompts(segments, video_type)


def _dedup_prompts(prompts: list) -> list:
    """Remove near-duplicate prompts (same first 60 chars after normalization)."""
    seen = set()
    result = []
    for p in prompts:
        key = re.sub(r'\s+', ' ', p.strip().lower())[:60]
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result if result else prompts


def _segment_fallback_prompts(segments: list, video_type: str) -> list:
    """Generate fallback prompts from segment keywords without LLM."""
    base = RHINO_BASE_SHORT if video_type == "short" else RHINO_BASE_LONG
    aspect = "9:16" if video_type == "short" else "16:9"

    # Keyword → scene mapping for semantic sync
    KEYWORD_SCENES = [
        (["bitcoin", "btc"], "holding a golden Bitcoin coin with dramatic orange glow, city skyline at night behind"),
        (["ethereum", "eth"], "examining a glowing Ethereum crystal on a holographic display, tech lab"),
        (["exchange", "coinbase", "binance", "comprar", "buy"], "using smartphone to navigate a crypto exchange app, green purchase confirmed screen"),
        (["$100", "100 dolar", "empezar", "principiante", "beginner"], "holding a $100 bill transforming into digital coins, warm golden light"),
        (["dca", "dollar cost", "semana", "week", "calendar"], "marking dates on a calendar with a pen, planning crypto purchases systematically"),
        (["riesgo", "risk", "pérdida", "loss", "caer", "red"], "staring at a red crashing chart on a large monitor, concerned expression, red ambient light"),
        (["subir", "ganancia", "profit", "green", "alcista", "bull"], "celebrating in front of multiple green profit charts, triumphant pose, golden lighting"),
        (["error", "mistake", "fomo", "panic", "liquidar"], "face-palming at a phone showing a bad trade, dramatic tense lighting"),
        (["wallet", "hardware", "ledger", "security", "seguridad"], "carefully placing crypto hardware wallet in a secure box, focused expression"),
        (["telegram", "scam", "estafa", "señales", "signal"], "pointing at a suspicious phone message with a warning look, dark moody setting"),
        (["leverage", "apalancamiento"], "standing next to a giant multiplier dial set to 100x, cautionary expression, warning lights"),
        (["estrategia", "strategy", "plan"], "drawing a strategic plan on a whiteboard with crypto symbols, confident pose"),
        (["exchange", "registro", "verificar", "kyc"], "completing identity verification on a laptop, official-looking process, clean office"),
        (["altcoin", "solana", "bnb", "top 10"], "browsing a holographic list of top crypto coins, analytical expression, data streams"),
        (["wallet", "cold storage", "guardar"], "securing digital assets in a vault with blockchain locks, dramatic lighting"),
    ]

    fallback_settings = [
        "trading desk with multiple monitors showing crypto charts",
        "modern city rooftop at sunset with holographic data displays",
        "futuristic server room with blue digital streams",
        "minimalist home office with warm morning light",
        "dark command center with wall-sized trading screens",
        "crypto conference stage under dramatic spotlights",
        "high-tech mobile workstation in a coffee shop",
        "financial district street with digital billboard displays",
        "underground crypto bunker with glowing hardware",
        "penthouse office overlooking a digital city skyline",
        "outdoor plaza with giant holographic crypto charts",
        "studio workspace with portfolio data projections",
        "open-plan office with green trading terminals",
        "garage setup with multiple screens and hardware wallets",
        "train or transit with mobile crypto setup",
        "zen garden with floating digital asset symbols",
    ]

    used_settings = set()
    prompts = []

    for i, seg in enumerate(segments):
        seg_lower = seg.lower()

        # Find best matching scene from keyword map
        scene_desc = None
        for keywords, scene in KEYWORD_SCENES:
            if any(kw in seg_lower for kw in keywords):
                scene_desc = scene
                break

        # Pick a setting that hasn't been used
        setting = None
        for s in fallback_settings:
            if s not in used_settings:
                setting = s
                used_settings.add(s)
                break
        if not setting:
            setting = f"unique location {i+1}, dramatic cinematic environment"

        if scene_desc:
            prompt = f"Anthropomorphic rhinoceros {scene_desc}, {setting}, {base}"
        else:
            # Generic but unique
            actions = [
                "analyzing holographic candlestick charts",
                "pointing confidently at rising price data",
                "checking portfolio on dual monitors",
                "reviewing blockchain transaction data",
                "calculating trading strategy on a tablet",
                "celebrating a successful trade execution",
                "studying market patterns with focused expression",
                "typing trading commands on a glowing keyboard",
            ]
            action = actions[i % len(actions)]
            prompt = f"Anthropomorphic rhinoceros {action}, {setting}, {aspect} aspect ratio, {base}"

        prompts.append(prompt)

    return prompts


def get_fallback_prompts(video_type: str) -> list:
    """Fallback rhino video prompts if LLM fails."""
    short_prompts = [
        f"Anthropomorphic rhinoceros sitting at a trading desk with multiple monitors showing green crypto charts, excited expression, neon-lit dark room, {RHINO_BASE_SHORT}",
        f"Anthropomorphic rhinoceros holding a glowing smartphone showing Bitcoin price spike, standing in a modern city at night, confident pose, {RHINO_BASE_SHORT}",
        f"Anthropomorphic rhinoceros pointing at a large holographic candlestick chart rising upward, dark background with blue data streams, triumphant expression, {RHINO_BASE_SHORT}",
    ]
    long_prompts = [
        f"Anthropomorphic rhinoceros standing in front of a massive wall of trading screens in a dark room, arms crossed, confident and commanding, {RHINO_BASE_LONG}",
        f"Anthropomorphic rhinoceros typing rapidly on a keyboard, laptop screen showing AI code and trading algorithms, focused expression, {RHINO_BASE_LONG}",
        f"Anthropomorphic rhinoceros analyzing a holographic portfolio dashboard with green profit metrics, modern office, impressed expression, {RHINO_BASE_LONG}",
        f"Anthropomorphic rhinoceros in casual clothes checking trading app on phone with a coffee, morning light, relaxed successful atmosphere, {RHINO_BASE_LONG}",
        f"Anthropomorphic rhinoceros looking at camera with a knowing smile, crypto chart rising behind them, dramatic lighting, motivational mood, {RHINO_BASE_LONG}",
        f"Anthropomorphic rhinoceros walking through a futuristic digital city with blockchain holograms floating around, confident stride, cinematic, {RHINO_BASE_LONG}",
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


def clean_script_for_tts(text: str) -> str:
    """Strip ALL structural markers — produce pure narration for TTS.

    ELIMINATES (en todas sus formas):
      - Section headers standalone: HOOK (30s):, PROBLEMA:, DESARROLLO:, etc.
      - Inline prefixes: "Gancho:", "--- HOOK :", "TÍTULO:", "TÍTULO: NEIRO:", etc.
      - Dashed prefixes: "--- HOOK :", "--- SECCIÓN :", "--- INTRO :"
      - Markdown bold headers: **HOOK**, **Desarrollo**
      - VIDEO_PROMPT_N: lines
      - GUIÓN COMPLETO: header
      - Numbered markdown steps: "1. **Analyze...**", "2. **Write...**"
    """
    import re

    # 1. Remove GUION/GUIÓN COMPLETO: header
    text = re.sub(r'^GU[IÍ](?:O|Ó)N(?:\s+COMPLETO)?\s*:\s*\n?', '', text,
                  flags=re.MULTILINE | re.IGNORECASE)

    # 2. Remove VIDEO_PROMPT lines
    text = re.sub(r'^VIDEO_PROMPT[_\d]*:.*$', '', text,
                  flags=re.MULTILINE | re.IGNORECASE)

    # 3. Remove dashed section markers: "--- HOOK :", "--- INTRO :", etc.
    text = re.sub(r'^[ \t]*-{2,}[ \t]*[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑa-z ]+[ \t]*:[ \t]*', '',
                  text, flags=re.MULTILINE)

    # 4. Remove section header lines (standalone or inline prefix):
    #    Matches: "HOOK (30s):", "Gancho:", "Desarrollo:", "CTA:", "TÍTULO:", etc.
    SECTION_WORDS = (
        r'HOOK|GANCHO|INTRO(?:DUCCION|DUCTION)?|PROBLEMA|PROBLEM|'
        r'EXPLICACION|EXPLANATION|DESARROLLO|DEVELOPMENT|'
        r'EJEMPLOS?|EXAMPLES?|CTA|CONCLUSION|OUTRO|CUERPO|BODY|'
        r'SOLUCION|SOLUTION|T[IÍ]TULO?|SECTION|DATO\s+IMPACTANTE|'
        r'REFLEXION|LLAMADA\s+A\s+LA\s+ACCION|CIERRE'
    )
    # Standalone full-line header (with optional timing): "HOOK (30s):"
    text = re.sub(
        rf'^[ \t]*\*{{0,2}}(?:{SECTION_WORDS})\*{{0,2}}\s*(?:\([^)]*\))?\s*:[ \t]*$',
        '', text, flags=re.MULTILINE | re.IGNORECASE
    )
    # Inline prefix at start of line: "Gancho: \"Trump lanzó..." → just the quote stays
    text = re.sub(
        rf'^[ \t]*\*{{0,2}}(?:{SECTION_WORDS})\*{{0,2}}\s*(?:\([^)]*\))?\s*:[ \t]+',
        '', text, flags=re.MULTILINE | re.IGNORECASE
    )

    # 5. Remove standalone **bold** header lines
    text = re.sub(r'^\*\*[^*\n]+\*\*\s*$', '', text, flags=re.MULTILINE)

    # 6. Remove markdown bold/italic inline
    text = re.sub(r'\*{1,3}([^*\n]+)\*{1,3}', r'\1', text)

    # 7. Remove numbered LLM meta-steps: "1. **Analyze the Request:**"
    text = re.sub(r'^\d+\.\s+\*?\*?[A-Z][^\n]{0,60}\*?\*?:\s*$', '', text,
                  flags=re.MULTILINE)

    # 8. Remove bullet sub-items that are LLM metadata (start with * **Role:**)
    text = re.sub(r'^\s*\*\s+\*\*(?:Role|Format|Tone|Style|Output)[^:]*\*\*.*$', '',
                  text, flags=re.MULTILINE | re.IGNORECASE)

    # 9. Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def generate_audio(text: str, output_path: Path) -> float:
    """Generate audio: edge-tts primary (MiniMax TTS disabled — generates static for Spanish)."""
    # Always sanitize script before TTS — strip section headers, video prompts, etc.
    text = clean_script_for_tts(text)
    # Edge-TTS Jorge is reliable and free — use as primary
    print("      🔊 Usando Edge-TTS (es-MX-JorgeNeural)...")
    duration = edge_tts_fallback(text, output_path)
    if duration > 0:
        return duration
    
    # Only try MiniMax as last resort
    print("      ⚠️ Edge-TTS falló, intentando MiniMax TTS...")
    return minimax_tts(text, output_path)


# ════════════════════════════════════════════════════════════════════════
# MINIMAX VIDEO GENERATION (Hailuo 2.3)
# ════════════════════════════════════════════════════════════════════════

def create_video_task(prompt: str, aspect_ratio: str = "9:16") -> str:
    """Submit video generation task. Returns task_id."""
    payload = {
        "prompt": prompt,
        "model": VIDEO_MODEL,
        "duration": VIDEO_DURATION,
        "resolution": "1080P",
        "aspect_ratio": aspect_ratio,
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


def is_comfyui_available(url: str = MAC_M3_URL) -> bool:
    """Check if ComfyUI server is reachable."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"{url}/system_stats", timeout=3):
            return True
    except Exception:
        return False


def generate_video_clip_comfyui(prompt: str, output_path: Path, clip_num: int,
                                 duration: int = 6, server_url: str = MAC_M3_URL) -> bool:
    """Generate video clip via ComfyUI on Mac (using LTX-Video or similar)."""
    import urllib.request
    try:
        prefix = f"btclip_{abs(hash(prompt)) % 99999:05d}_{clip_num}"
        
        # LTX-Video workflow for short clips
        workflow = {
            "1": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": "ltx-video-2b-v0.9.1.safetensors"}},
            "2": {"class_type": "CLIPTextEncode", 
                  "inputs": {"text": prompt, "clip": ["1", 1]}},
            "3": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": "blurry, low quality, distorted, watermark, text, amateur, static",
                             "clip": ["1", 1]}},
            "4": {"class_type": "EmptyLTXVLatentVideo",
                  "inputs": {"width": 768, "height": 512, "length": duration * 8, "batch_size": 1}},
            "5": {"class_type": "KSampler",
                  "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                             "latent_image": ["4", 0], "seed": int(time.time()) % 999999,
                             "steps": 20, "cfg": 3.0, "sampler_name": "euler",
                             "scheduler": "simple", "denoise": 1.0}},
            "6": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
            "7": {"class_type": "VHS_VideoCombine",
                  "inputs": {"images": ["6", 0], "frame_rate": 8, "loop_count": 0,
                             "filename_prefix": prefix, "format": "video/h264-mp4",
                             "save_output": True}}
        }
        
        data = json.dumps({"prompt": workflow}).encode()
        req = urllib.request.Request(
            f"{server_url}/prompt", data=data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"        🍎 Clip {clip_num} → ComfyUI Mac...")
        with urllib.request.urlopen(req, timeout=30) as r:
            prompt_id = json.loads(r.read()).get("prompt_id")
        
        if not prompt_id:
            return False
        
        # Poll for completion
        deadline = time.time() + COMFYUI_TIMEOUT
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{server_url}/history/{prompt_id}", timeout=10) as r:
                    history = json.loads(r.read())
                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get("status", {})
                    if status.get("status_str") == "error":
                        print(f"        ❌ ComfyUI error")
                        return False
                    outputs = entry.get("outputs", {})
                    for node_id, node_out in outputs.items():
                        for vid_info in node_out.get("gifs", []) + node_out.get("videos", []):
                            fname = vid_info.get("filename", "")
                            if fname.endswith(".mp4"):
                                # Download from ComfyUI
                                vid_url = f"{server_url}/view?filename={fname}&type=output"
                                vid_data = urllib.request.urlopen(vid_url, timeout=60).read()
                                output_path.parent.mkdir(parents=True, exist_ok=True)
                                with open(output_path, "wb") as f:
                                    f.write(vid_data)
                                size = output_path.stat().st_size / (1024 * 1024)
                                print(f"        ✅ Clip {clip_num} ComfyUI: {size:.1f}MB")
                                return True
            except Exception:
                pass
            time.sleep(5)
        
        print(f"        ⏰ Clip {clip_num} ComfyUI timeout")
        return False
        
    except Exception as e:
        print(f"        ⚠️ ComfyUI error: {e}")
        return False


def _is_green_screen_clip(video_path: Path) -> bool:
    """
    Detecta si un clip de video es green screen (pantalla verde pura).
    Orden de Ender 2026-03-28: green screen = inaceptable, rechazar automáticamente.
    Retorna True si el clip ES green screen (debe ser rechazado).
    """
    try:
        import subprocess as _sp, tempfile as _tmp, os as _os
        import numpy as _np
        from PIL import Image as _PILI
        
        with _tmp.TemporaryDirectory() as _td:
            # Extraer 3 frames del clip
            _probe = _sp.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
                capture_output=True, text=True, timeout=10
            )
            _dur = float(_probe.stdout.strip() or "6")
            _green_count = 0
            for _i, _t in enumerate([_dur * 0.2, _dur * 0.5, _dur * 0.8]):
                _fp = _os.path.join(_td, f"f{_i}.jpg")
                _sp.run(
                    ["ffmpeg", "-ss", str(_t), "-i", str(video_path),
                     "-frames:v", "1", "-q:v", "2", _fp, "-y"],
                    capture_output=True, timeout=15
                )
                if not _os.path.exists(_fp):
                    continue
                _img = _PILI.open(_fp).convert("RGB")
                _arr = _np.array(_img, dtype=_np.float32)
                _avg_r = _arr[:,:,0].mean()
                _avg_g = _arr[:,:,1].mean()
                _avg_b = _arr[:,:,2].mean()
                if _avg_g > _avg_r * 1.8 and _avg_g > _avg_b * 1.8 and _avg_g > 100:
                    _green_count += 1
            
            if _green_count >= 2:  # ≥2 de 3 frames son green screen
                print(f"        🟢 GREEN SCREEN detectado en clip ({_green_count}/3 frames) — RECHAZADO")
                return True
    except Exception as _e:
        print(f"        ⚠️ No se pudo verificar green screen: {_e}")
    return False


def generate_video_clip(prompt: str, output_path: Path, clip_num: int) -> bool:
    """Generate a single video clip - tries ComfyUI first, then MiniMax Hailuo.
    Valida que el clip NO sea green screen antes de aceptarlo (orden Ender 2026-03-28).
    """
    
    # Try ComfyUI on Mac first (free, local)
    if USE_COMFYUI_FIRST and is_comfyui_available():
        if generate_video_clip_comfyui(prompt, output_path, clip_num):
            if _is_green_screen_clip(output_path):
                output_path.unlink(missing_ok=True)
                print(f"        ⚠️ ComfyUI generó green screen, intentando Hailuo...")
            else:
                return True
        else:
            print(f"        ⚠️ ComfyUI failed, trying MiniMax Hailuo...")
    
    # Fallback to MiniMax Hailuo
    try:
        task_id = create_video_task(prompt)
        print(f"        📡 Clip {clip_num} task: {task_id}")
        
        file_id = poll_video_task(task_id)
        print(f"        ✅ Clip {clip_num} ready")
        
        download_video_file(file_id, output_path)
        size = output_path.stat().st_size / (1024 * 1024)
        print(f"        💾 Clip {clip_num}: {size:.1f}MB")
        
        # Validar que NO sea green screen
        if _is_green_screen_clip(output_path):
            output_path.unlink(missing_ok=True)
            print(f"        ❌ Clip {clip_num} rechazado: GREEN_SCREEN — clip descartado")
            return False
        
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


# ════════════════════════════════════════════════════════════════════════
# COIN LOGO FETCHER
# ════════════════════════════════════════════════════════════════════════

def extract_coin_symbol(script: dict) -> str | None:
    """Extract the main coin ticker/symbol from script tags or title."""
    tags = script.get("tags", [])
    title = script.get("title", "")

    # Common tickers to detect — check tags first (most reliable)
    known = [
        "BTC","ETH","SOL","BNB","AKT","PENGU","PI","ZEC","JUP","BONK","WIF",
        "RAY","DOGE","SHIB","AVAX","DOT","LINK","ADA","MATIC","OP","ARB","SUI",
        "APT","TRX","LTC","XRP","ATOM","FTM","NEAR","INJ","TIA","PYTH","JTO",
        "MEME","PEPE","FLOKI","BRETT","MOG","POPCAT","WEN","NEIRO","GOAT"
    ]
    for tag in tags:
        tag_up = tag.upper()
        for coin in known:
            if coin == tag_up or tag_up.startswith(coin):
                return coin

    # Fallback: scan title for known tickers
    title_up = title.upper()
    for coin in known:
        if coin in title_up:
            return coin

    return None


def fetch_coin_logo(symbol: str) -> Path | None:
    """Download coin logo from CoinGecko. Returns path or None if failed."""
    cached = COIN_LOGOS_DIR / f"{symbol}.png"
    if cached.exists() and cached.stat().st_size > 1000:
        return cached

    # CoinGecko ID map for common coins
    id_map = {
        "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "BNB": "binancecoin",
        "AKT": "akash-network", "PENGU": "pudgy-penguins", "PI": "pi-network",
        "ZEC": "zcash", "JUP": "jupiter-exchange-solana", "BONK": "bonk",
        "WIF": "dogwifcoin", "RAY": "raydium", "DOGE": "dogecoin", "SHIB": "shiba-inu",
        "AVAX": "avalanche-2", "DOT": "polkadot", "LINK": "chainlink",
        "ADA": "cardano", "MATIC": "matic-network", "OP": "optimism",
        "ARB": "arbitrum", "SUI": "sui", "APT": "aptos", "TRX": "tron",
        "LTC": "litecoin", "XRP": "ripple", "ATOM": "cosmos", "FTM": "fantom",
        "NEAR": "near", "INJ": "injective-protocol", "TIA": "celestia",
        "PYTH": "pyth-network", "JTO": "jito-governance-token",
        "PEPE": "pepe", "FLOKI": "floki", "MEME": "memecoin",
    }

    coin_id = id_map.get(symbol.upper())
    if not coin_id:
        # Try searching CoinGecko
        try:
            r = requests.get(
                f"https://api.coingecko.com/api/v3/search?query={symbol}",
                timeout=10
            )
            results = r.json().get("coins", [])
            if results:
                coin_id = results[0]["id"]
        except Exception:
            return None

    try:
        r = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false"
            f"&tickers=false&market_data=false&community_data=false&developer_data=false",
            timeout=15
        )
        img_url = r.json().get("image", {}).get("large", "")
        if not img_url:
            return None

        img_r = requests.get(img_url, timeout=15)
        cached.write_bytes(img_r.content)

        # Verify it's actually a PNG/image (not HTML error page)
        if cached.stat().st_size < 1000:
            cached.unlink()
            return None

        print(f"      🖼️  Coin logo downloaded: {symbol} ({cached.stat().st_size//1024}KB)")
        return cached

    except Exception as e:
        print(f"      ⚠️  Could not fetch logo for {symbol}: {e}")
        return None


def validate_video_quality(video_path: Path) -> bool:
    """Validate that video has visible content (not black)."""
    try:
        from PIL import Image, ImageStat
        
        print(f"      🔍 Validando calidad visual...")
        
        # Get video dimensions and duration
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,duration",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=30
        )
        
        dims = probe.stdout.strip().split(",")
        if len(dims) >= 3:
            w, h, dur = int(dims[0]), int(dims[1]), float(dims[2])
            print(f"      📐 Video: {w}x{h}, {dur:.1f}s")
        
        # Extract 5 evenly spaced frames
        frame_dir = video_path.parent / "quality_check"
        frame_dir.mkdir(exist_ok=True)
        
        total_brightness = 0
        frame_count = 0
        
        for i in range(1, 6):
            frame_path = frame_dir / f"check_{i}.jpg"
            timestamp = i * 2  # Check at 2, 4, 6, 8, 10 seconds
            
            subprocess.run([
                "ffmpeg", "-y", "-ss", str(timestamp), "-i", str(video_path),
                "-vframes", "1", "-q:v", "2", str(frame_path)
            ], capture_output=True, timeout=30)
            
            if frame_path.exists():
                img = Image.open(frame_path)
                stat = ImageStat.Stat(img)
                r, g, b = stat.mean
                brightness = (r + g + b) / 3
                total_brightness += brightness
                frame_count += 1
                
                print(f"      🎞️  Frame {i}: brightness={brightness:.1f} (RGB: {r:.1f},{g:.1f},{b:.1f})")
                
                # If any frame is too dark (<30), video is invalid
                if brightness < 30:
                    print(f"      ❌ Frame {i} too dark: brightness={brightness:.1f} < 30")
                    return False
            else:
                print(f"      ⚠️  Frame {i} extraction failed")
                return False
        
        # Average brightness
        avg_brightness = total_brightness / frame_count if frame_count > 0 else 0
        print(f"      ✅ Quality check passed - avg brightness: {avg_brightness:.1f}")
        
        # Clean up
        import shutil
        shutil.rmtree(frame_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"      ⚠️  Quality validation error: {e}")
        # If validation fails, assume bad quality
        return False


def assemble_video(clips: list, audio_path: Path, output_path: Path, 
                   title: str, duration: float, video_type: str,
                   script_text: str, script_dir: Path,
                   coin_logo_path: Path | None = None) -> bool:
    """Assemble clips + audio + karaoke subtitles into final video."""
    
    # Generate karaoke subtitles (ASS) or fallback to SRT
    # Always use Whisper for real word-level timestamps (edge-tts audio is clean enough)
    sub_path = None
    if HAS_KARAOKE:
        try:
            sub_path = generate_karaoke_subs(
                audio_path, script_text, script_dir, video_type,
                style="word_highlight", use_whisper=True
            )
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

    # Normalize clips to target resolution — fix clips with wrong dimensions (e.g. 1080x2050)
    target_w, target_h = (1080, 1920) if video_type == "short" else (1920, 1080)
    norm_dir = script_dir / "clips_norm"
    norm_dir.mkdir(exist_ok=True)
    normalized_clips = []
    for clip in valid_clips:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0", str(clip)],
            capture_output=True, text=True
        )
        dims = probe.stdout.strip().split(",")
        w, h = (int(dims[0]), int(dims[1])) if len(dims) == 2 else (0, 0)
        norm_clip = norm_dir / clip.name
        if w == target_w and h == target_h:
            norm_clip = clip  # already correct, skip re-encode
        else:
            print(f"      🔧 Normalizing {clip.name}: {w}x{h} → {target_w}x{target_h}")
            if video_type == "short":
                vf = (f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
                      f"crop={target_w}:{target_h},avgblur=30[bg];"
                      f"[bg]scale={target_w}:{target_h}[bg2];"
                      f"movie='{clip}',scale={target_w}:-2:force_original_aspect_ratio=decrease[fg];"
                      f"[bg2][fg]overlay=(W-w)/2:(H-h)/2")
            else:
                vf = (f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                      f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black")
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(clip), "-vf", vf,
                 "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-an", str(norm_clip)],
                capture_output=True, timeout=60
            )
        normalized_clips.append(norm_clip)

    # FIX v4.1: Anti-repetition — never loop the same clip pool.
    # Instead, shuffle and extend WITHOUT exact repeats until duration is covered.
    # Strategy: cycle through clips in shuffled order, never the same clip twice in a row.
    clip_duration = VIDEO_DURATION  # 6s per clip
    total_clip_time = len(normalized_clips) * clip_duration
    clips_needed = max(len(normalized_clips), int(duration / clip_duration) + 2)

    concat_lines = []
    extended_clips = list(normalized_clips)  # start with original set
    last_clip = None

    if len(normalized_clips) >= 2:
        # Extend with shuffled rounds until we have enough clips
        while len(extended_clips) < clips_needed:
            round_clips = list(normalized_clips)
            random.shuffle(round_clips)
            # Avoid same clip appearing consecutively at round boundary
            if last_clip and round_clips and round_clips[0] == last_clip:
                round_clips[0], round_clips[-1] = round_clips[-1], round_clips[0]
            extended_clips.extend(round_clips)

        for clip in extended_clips[:clips_needed]:
            concat_lines.append(f"file '{clip}'")
            last_clip = clip
    else:
        # Only 1 clip — unavoidable repeat, but note it
        print(f"      ⚠️ Only 1 clip available — repetition unavoidable")
        for _ in range(clips_needed):
            for clip in normalized_clips:
                concat_lines.append(f"file '{clip}'")
    
    concat_path.write_text('\n'.join(concat_lines))
    
    # Resolution based on type
    scale = "1080:1920" if video_type == "short" else "1920:1080"
    
    # Build ffmpeg command
    safe_title = title.replace("'", "'\\''").replace('"', '\\"').replace(':', ' -')
    
    # Step 1: Concat clips and scale
    # For SHORT (vertical 9:16): use blur-fill technique instead of black bars
    # Hailuo generates horizontal clips (1366x768) — blur-fill looks professional
    # For LONG (horizontal): simple scale + pad is fine
    intermediate = script_dir / "concat_scaled.mp4"
    if video_type == "short":
        # Blur-fill: blurred background fills black bars, sharp image centered
        vf_scale = (
            "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,avgblur=30[bg];"
            "[0:v]scale=1080:-2:force_original_aspect_ratio=decrease[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2"
        )
        cmd1 = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_path),
            "-filter_complex", vf_scale,
            "-t", str(duration + 1),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-an",
            str(intermediate)
        ]
    else:
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
                    f"'FontName=DejaVu Sans,FontSize=100,PrimaryColour=&H0000FFFF,"
                    f"OutlineColour=&H00000000,Outline=3,Shadow=0,Alignment=2,MarginV=400'"
                )
            else:
                subtitle_filter = (
                    f",subtitles='{sub_escaped}':force_style="
                    f"'FontName=DejaVu Sans,FontSize=65,PrimaryColour=&H0000FFFF,"
                    f"OutlineColour=&H00000000,Outline=3,Shadow=0,Alignment=2,MarginV=60'"
                )
    
    # BitTrader logo overlay (top-right corner, 240px, semi-transparent)
    # Buscar logo en múltiples rutas (orden de preferencia)
    _logo_candidates = [
        WORKSPACE / "bittrader/assets/bittrader_logo_transparent.png",
        WORKSPACE / "bittrader/assets/bittrader_logo.png",
        WORKSPACE / "videos/BIBLIOTECA/bittrader_logo.png",
        BITTRADER / "assets/bittrader_logo.png",
    ]
    LOGO_PATH = next((p for p in _logo_candidates if p.exists()), None) or (WORKSPACE / "videos/BIBLIOTECA/bittrader_logo.png")
    has_logo = LOGO_PATH.exists()
    if not has_logo:
        print(f"      ⚠️  LOGO NO ENCONTRADO — video saldrá sin logo BitTrader. Rutas buscadas: {[str(p) for p in _logo_candidates]}")

    # Logo size: 240px for shorts (vertical), 180px for longs
    logo_size = 240 if video_type == "short" else 180
    # Coin logo size: 100px bottom-left
    coin_logo_size = 100 if video_type == "short" else 80

    if has_logo:
        # Use filter_complex for logo + subtitles + optional coin logo
        sub_filter = ""
        if sub_path.exists():
            sub_escaped = str(sub_path).replace("'", "'\\''").replace(":", "\\:")
            if sub_path.suffix == ".ass":
                sub_filter = f",ass='{sub_escaped}'"
            else:
                if video_type == "short":
                    sub_filter = (
                        f",subtitles='{sub_escaped}':force_style="
                        f"'FontName=DejaVu Sans,FontSize=100,PrimaryColour=&H0000FFFF,"
                        f"OutlineColour=&H00000000,Outline=3,Shadow=0,Alignment=2,MarginV=400'"
                    )
                else:
                    sub_filter = (
                        f",subtitles='{sub_escaped}':force_style="
                        f"'FontName=DejaVu Sans,FontSize=65,PrimaryColour=&H0000FFFF,"
                        f"OutlineColour=&H00000000,Outline=3,Shadow=0,Alignment=2,MarginV=60'"
                    )

        if coin_logo_path and coin_logo_path.exists():
            # 3 inputs: video, audio, BT logo, coin logo
            filter_complex = (
                f"[2:v]scale={logo_size}:-1,format=rgba,colorchannelmixer=aa=0.85[btlogo];"
                f"[3:v]scale={coin_logo_size}:-1,format=rgba,colorchannelmixer=aa=0.90[coin];"
                f"[0:v][btlogo]overlay=W-w-30:30:format=auto[v1];"
                f"[v1][coin]overlay=25:H-h-25:format=auto{sub_filter}"
            )
            cmd2 = [
                "ffmpeg", "-y",
                "-i", str(intermediate),
                "-i", str(audio_path),
                "-i", str(LOGO_PATH),
                "-i", str(coin_logo_path),
                "-filter_complex", filter_complex,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
                "-shortest",
                "-movflags", "+faststart",
                str(output_path)
            ]
            print(f"      🪙 Coin logo overlay: {coin_logo_path.name}")
        else:
            filter_complex = (
                f"[2:v]scale={logo_size}:-1[logo];"
                f"[0:v][logo]overlay=W-w-30:30:format=auto{sub_filter}"
            )
            cmd2 = [
                "ffmpeg", "-y",
                "-i", str(intermediate),
                "-i", str(audio_path),
                "-i", str(LOGO_PATH),
                "-filter_complex", filter_complex,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
                "-shortest",
                "-movflags", "+faststart",
                str(output_path)
            ]
    else:
        # No logo — use simple -vf
        subtitle_filter = ""
        if sub_path.exists():
            sub_escaped = str(sub_path).replace("'", "'\\''").replace(":", "\\:")
            if sub_path.suffix == ".ass":
                subtitle_filter = f",ass='{sub_escaped}'"
            else:
                if video_type == "short":
                    subtitle_filter = (
                        f",subtitles='{sub_escaped}':force_style="
                        f"'FontName=DejaVu Sans,FontSize=100,PrimaryColour=&H0000FFFF,"
                        f"OutlineColour=&H00000000,Outline=3,Shadow=0,Alignment=2,MarginV=400'"
                    )
                else:
                    subtitle_filter = (
                        f",subtitles='{sub_escaped}':force_style="
                        f"'FontName=DejaVu Sans,FontSize=65,PrimaryColour=&H0000FFFF,"
                        f"OutlineColour=&H00000000,Outline=3,Shadow=0,Alignment=2,MarginV=60'"
                    )

        vf = f"[0:v]{subtitle_filter}" if subtitle_filter else None

        cmd2 = [
            "ffmpeg", "-y",
            "-i", str(intermediate),
            "-i", str(audio_path),
        ]
        if vf:
            cmd2 += ["-vf", subtitle_filter.lstrip(",")]
        cmd2 += [
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
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
    
    # NEW: Validate video quality (check for black video)
    if not validate_video_quality(output_path):
        print(f"      ❌ Video quality check failed - video too dark")
        return False
    
    return output_path.exists()


def make_local_clips_video(audio_path: Path, output_path: Path, title: str,
                            duration: float, video_type: str, script_text: str,
                            script_dir: Path,
                            coin_logo_path: Path | None = None) -> bool:
    """
    Fallback using REAL local video clips from bg_videos_cache_local/.
    Uses Ken Burns + concat + subs — no solid color backgrounds.
    This replaces make_fallback_video() as the primary fallback when AI clips fail.
    """
    import random as _random

    # ── Pick the right clip library ──────────────────────────────────────
    target_w, target_h = (1080, 1920) if video_type == "short" else (1920, 1080)

    if video_type == "short":
        clips_dir = DATA_DIR / "bg_videos_cache_local"
        pattern = "short_*.mp4"
    else:
        clips_dir = DATA_DIR / "bg_videos_cache_local"
        pattern = "*.mp4"  # use any clip, will be scaled

    all_clips = sorted(clips_dir.glob(pattern))
    if not all_clips:
        # Fall back to local images with Ken Burns
        return _make_kenburns_fallback(audio_path, output_path, title, duration,
                                       video_type, script_text, script_dir, coin_logo_path)

    # ── Select enough clips to fill the duration ──────────────────────────
    # Each clip is ~5s, shuffle for variety
    selected = _random.sample(all_clips, min(len(all_clips), max(5, int(duration / 5) + 2)))

    # ── Normalize / stretch each clip to target resolution ────────────────
    norm_dir = script_dir / "clips_local_norm"
    norm_dir.mkdir(exist_ok=True)
    normed = []

    for i, clip in enumerate(selected):
        out_clip = norm_dir / f"norm_{i:02d}.mp4"
        # Scale to fill target (blur-fill for shorts to avoid black bars)
        if video_type == "short":
            vf = (
                f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{target_h}[scaled];"
                f"[scaled]split[a][b];"
                f"[b]scale={target_w}:{target_h},avgblur=30[bg];"
                f"[bg][a]overlay=(W-w)/2:(H-h)/2:format=auto[v]"
            )
            cmd = ["ffmpeg", "-y", "-i", str(clip),
                   "-filter_complex", vf, "-map", "[v]",
                   "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                   "-an", str(out_clip)]
        else:
            cmd = ["ffmpeg", "-y", "-i", str(clip),
                   "-vf", f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h}",
                   "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                   "-an", str(out_clip)]

        r = subprocess.run(cmd, capture_output=True, timeout=60)
        if r.returncode == 0 and out_clip.exists():
            normed.append(out_clip)

    if not normed:
        return _make_kenburns_fallback(audio_path, output_path, title, duration,
                                       video_type, script_text, script_dir, coin_logo_path)

    # ── Loop clips to fill full audio duration ────────────────────────────
    # Repeat the list until we have enough total duration
    loop_clips = []
    total_dur = 0.0
    while total_dur < duration + 2:
        for c in normed:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(c)],
                capture_output=True, text=True, timeout=10)
            d = float(probe.stdout.strip() or "5")
            loop_clips.append(c)
            total_dur += d
            if total_dur >= duration + 2:
                break

    # ── Concat clips ──────────────────────────────────────────────────────
    concat_path = script_dir / "concat_local.txt"
    concat_path.write_text("\n".join(f"file '{c}'" for c in loop_clips))

    intermediate = script_dir / "local_clips_concat.mp4"
    r = subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-an",
        str(intermediate)
    ], capture_output=True, timeout=300)

    if r.returncode != 0 or not intermediate.exists():
        return _make_kenburns_fallback(audio_path, output_path, title, duration,
                                       video_type, script_text, script_dir, coin_logo_path)

    # ── Generate subtitles ────────────────────────────────────────────────
    sub_path = None
    if HAS_KARAOKE:
        try:
            sub_path = generate_karaoke_subs(
                audio_path, script_text, script_dir, video_type,
                style="word_highlight", use_whisper=True
            )
        except Exception:
            pass
    if not sub_path or not sub_path.exists():
        sub_path = script_dir / "subtitles_local.srt"
        create_srt_from_text(script_text, duration, sub_path)

    # ── Build filter_complex for overlays + subs ──────────────────────────
    BT_LOGO = WORKSPACE / "videos/BIBLIOTECA/bittrader_logo.png"
    inputs = ["-i", str(intermediate), "-i", str(audio_path)]
    filter_parts = []
    map_video = "[0:v]"
    input_idx = 2

    if BT_LOGO.exists():
        logo_size = 240 if video_type == "short" else 180
        inputs += ["-i", str(BT_LOGO)]
        filter_parts.append(
            f"[{input_idx}:v]scale={logo_size}:-1,format=rgba,colorchannelmixer=aa=0.85[btlogo]"
        )
        filter_parts.append(f"{map_video}[btlogo]overlay=W-w-25:25:format=auto[vbt]")
        map_video = "[vbt]"
        input_idx += 1

    if coin_logo_path and coin_logo_path.exists():
        coin_size = 100 if video_type == "short" else 80
        inputs += ["-i", str(coin_logo_path)]
        filter_parts.append(
            f"[{input_idx}:v]scale={coin_size}:-1,format=rgba,colorchannelmixer=aa=0.90[coin]"
        )
        filter_parts.append(f"{map_video}[coin]overlay=25:H-h-25:format=auto[vcoin]")
        map_video = "[vcoin]"
        input_idx += 1

    if sub_path and sub_path.exists():
        sub_esc = str(sub_path).replace("'", "'\\''").replace(":", "\\:")
        if sub_path.suffix == ".ass":
            filter_parts.append(f"{map_video}ass='{sub_esc}'[vfinal]")
            map_video = "[vfinal]"
        else:
            margin_v = "400" if video_type == "short" else "80"
            font_size = "100" if video_type == "short" else "65"
            subs_filter = (
                f"subtitles='{sub_esc}':force_style="
                f"'FontName=DejaVu Sans,FontSize={font_size},PrimaryColour=&H0000FFFF,"
                f"OutlineColour=&H00000000,Outline=3,Shadow=0,Alignment=2,MarginV={margin_v}'"
            )
            filter_parts.append(f"{map_video}{subs_filter}[vfinal]")
            map_video = "[vfinal]"

    filter_complex = ";".join(filter_parts) if filter_parts else None

    cmd = ["ffmpeg", "-y"] + inputs
    if filter_complex:
        cmd += ["-filter_complex", filter_complex, "-map", map_video, "-map", "1:a"]
    else:
        cmd += ["-map", "0:v", "-map", "1:a"]
    cmd += [
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        "-shortest", "-movflags", "+faststart",
        str(output_path)
    ]

    r = subprocess.run(cmd, capture_output=True, timeout=600)
    intermediate.unlink(missing_ok=True)
    concat_path.unlink(missing_ok=True)

    if r.returncode != 0 or not output_path.exists():
        return _make_kenburns_fallback(audio_path, output_path, title, duration,
                                       video_type, script_text, script_dir, coin_logo_path)

    print(f"      ✅ Local clips video assembled: {output_path.stat().st_size/1024/1024:.1f}MB")
    return True


def _make_kenburns_fallback(audio_path: Path, output_path: Path, title: str,
                              duration: float, video_type: str, script_text: str,
                              script_dir: Path,
                              coin_logo_path: Path | None = None) -> bool:
    """
    Ken Burns fallback: use local images with zoom/pan animation.
    Better than solid color but simpler than AI video.
    """
    import random as _random

    target_w, target_h = (1080, 1920) if video_type == "short" else (1920, 1080)
    imgs_dir = DATA_DIR / "bg_images_cache_local"
    all_imgs = sorted(imgs_dir.glob("*.png")) + sorted(imgs_dir.glob("*.jpg"))

    if not all_imgs:
        # True last resort: solid color
        return make_fallback_video(audio_path, output_path, title, duration,
                                   video_type, script_text, script_dir)

    # Pick enough images
    n_needed = max(3, int(duration / 5) + 2)
    selected = (_random.sample(all_imgs, min(len(all_imgs), n_needed))
                if len(all_imgs) >= n_needed else all_imgs * (n_needed // len(all_imgs) + 1))
    selected = selected[:n_needed]

    KB_STYLES = [
        f"zoompan=z='min(zoom+0.0008,1.3)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=150:s={target_w}x{target_h}:fps=30",
        f"zoompan=z='if(lte(zoom,1.0),1.3,zoom-0.0008)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=150:s={target_w}x{target_h}:fps=30",
        f"zoompan=z='1.25':x='if(lte(on,1),0,min(x+2,iw/5))':y='ih/2-(ih/zoom/2)':d=150:s={target_w}x{target_h}:fps=30",
        f"zoompan=z='1.25':x='if(lte(on,1),iw/5,max(x-2,0))':y='ih/2-(ih/zoom/2)':d=150:s={target_w}x{target_h}:fps=30",
    ]

    kb_dir = script_dir / "kb_clips_fallback"
    kb_dir.mkdir(exist_ok=True)
    kb_clips = []

    for i, img in enumerate(selected):
        kbout = kb_dir / f"kb_{i:02d}.mp4"
        scale_vf = (
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
            f"crop={target_w}:{target_h},"
            f"{KB_STYLES[i % len(KB_STYLES)]}"
        )
        r = subprocess.run([
            "ffmpeg", "-y", "-loop", "1", "-i", str(img),
            "-vf", scale_vf,
            "-t", "5",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-an", str(kbout)
        ], capture_output=True, timeout=60)
        if r.returncode == 0 and kbout.exists():
            kb_clips.append(kbout)

    if not kb_clips:
        return make_fallback_video(audio_path, output_path, title, duration,
                                   video_type, script_text, script_dir)

    # Loop until enough duration
    loop_clips = []
    total_dur = 0.0
    while total_dur < duration + 2:
        for c in kb_clips:
            loop_clips.append(c)
            total_dur += 5.0
            if total_dur >= duration + 2:
                break

    concat_path = script_dir / "concat_kb_fallback.txt"
    concat_path.write_text("\n".join(f"file '{c}'" for c in loop_clips))

    intermediate = script_dir / "kb_concat_fallback.mp4"
    r = subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-an",
        str(intermediate)
    ], capture_output=True, timeout=300)

    if r.returncode != 0 or not intermediate.exists():
        return make_fallback_video(audio_path, output_path, title, duration,
                                   video_type, script_text, script_dir)

    # Subtitles
    sub_path = None
    if HAS_KARAOKE:
        try:
            sub_path = generate_karaoke_subs(audio_path, script_text, script_dir, video_type,
                                              style="word_highlight", use_whisper=True)
        except Exception:
            pass
    if not sub_path or not sub_path.exists():
        sub_path = script_dir / "subtitles_kb.srt"
        create_srt_from_text(script_text, duration, sub_path)

    BT_LOGO = WORKSPACE / "videos/BIBLIOTECA/bittrader_logo.png"
    inputs = ["-i", str(intermediate), "-i", str(audio_path)]
    filter_parts = []
    map_video = "[0:v]"
    input_idx = 2

    if BT_LOGO.exists():
        logo_size = 240 if video_type == "short" else 180
        inputs += ["-i", str(BT_LOGO)]
        filter_parts.append(f"[{input_idx}:v]scale={logo_size}:-1,format=rgba,colorchannelmixer=aa=0.85[btlogo]")
        filter_parts.append(f"{map_video}[btlogo]overlay=W-w-25:25:format=auto[vbt]")
        map_video = "[vbt]"
        input_idx += 1

    if sub_path and sub_path.exists():
        sub_esc = str(sub_path).replace("'", "'\\''").replace(":", "\\:")
        if sub_path.suffix == ".ass":
            filter_parts.append(f"{map_video}ass='{sub_esc}'[vfinal]")
            map_video = "[vfinal]"
        else:
            margin_v = "400" if video_type == "short" else "80"
            font_size = "100" if video_type == "short" else "65"
            subs_filter = (
                f"subtitles='{sub_esc}':force_style="
                f"'FontName=DejaVu Sans,FontSize={font_size},PrimaryColour=&H0000FFFF,"
                f"OutlineColour=&H00000000,Outline=3,Shadow=0,Alignment=2,MarginV={margin_v}'"
            )
            filter_parts.append(f"{map_video}{subs_filter}[vfinal]")
            map_video = "[vfinal]"

    filter_complex = ";".join(filter_parts) if filter_parts else None
    cmd = ["ffmpeg", "-y"] + inputs
    if filter_complex:
        cmd += ["-filter_complex", filter_complex, "-map", map_video, "-map", "1:a"]
    else:
        cmd += ["-map", "0:v", "-map", "1:a"]
    cmd += [
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        "-shortest", "-movflags", "+faststart",
        str(output_path)
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    intermediate.unlink(missing_ok=True)
    concat_path.unlink(missing_ok=True)
    return r.returncode == 0 and output_path.exists()


def make_fallback_video(audio_path: Path, output_path: Path, title: str,
                        duration: float, video_type: str, script_text: str,
                        script_dir: Path) -> bool:
    """Fallback: generate video with colored background + text (no AI clips)."""
    
    # FIXED: Use lighter colors to avoid black video detection
    BG_COLORS = ["#1e3a5f", "#0d47a1", "#1565c0", "#1976d2", "#2196f3"]  # Brighter blues
    ACCENT_COLORS = ["#FFD700", "#00FF88", "#FF6B35", "#00D4FF", "#FF3366"]
    
    bg = random.choice(BG_COLORS)
    accent = random.choice(ACCENT_COLORS)
    
    # Generate karaoke subtitles or fallback to SRT
    sub_path = None
    if HAS_KARAOKE:
        try:
            sub_path = generate_karaoke_subs(
                audio_path, script_text, script_dir, video_type,
                style="word_highlight", use_whisper=True
            )
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
                f"'FontName=DejaVu Sans,FontSize=100,PrimaryColour=&H0000FFFF,"
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
    
    # NEW: Validate video quality for fallback too
    if result.returncode == 0 and output_path.exists():
        if validate_video_quality(output_path):
            return True
        else:
            print(f"      ❌ Fallback video quality check failed - regenerating with brighter background...")
            # Try again with even brighter background
            return _retry_fallback_brighter(audio_path, output_path, title, duration, video_type, script_text, script_dir)
    
    return False


def _retry_fallback_brighter(audio_path: Path, output_path: Path, title: str,
                              duration: float, video_type: str, script_text: str,
                              script_dir: Path) -> bool:
    """Retry fallback with very bright background."""
    BG_COLORS = ["#FFD700", "#FFA500", "#FF6347", "#4CAF50", "#00BCD4"]  # Very bright colors
    ACCENT_COLORS = ["#000000", "#333333", "#555555", "#777777", "#000000"]  # Dark text
    
    bg = random.choice(BG_COLORS)
    accent = random.choice(ACCENT_COLORS)
    
    sub_path = None
    if HAS_KARAOKE:
        try:
            sub_path = generate_karaoke_subs(
                audio_path, script_text, script_dir, video_type,
                style="word_highlight", use_whisper=True
            )
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
                f"'FontName=DejaVu Sans,FontSize=100,PrimaryColour=&H00000000,"
                f"OutlineColour=&H00FFFFFF,Outline=3,Shadow=0,Alignment=2,MarginV={margin_v}'"
            )
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={bg}:s={size}:d={duration}:r=30",
        "-i", str(audio_path),
        "-vf", (f"drawtext=text='{safe_title}':fontcolor={accent}:fontsize={title_size}:"
                f"x=(w-text_w)/2:y={title_y}:font=DejaVu Sans Bold:borderw=3:bordercolor=white"
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
    print(f"    🐛 DEBUG: produce_single() STARTED for script: {script.get('id')}")
    script_id = script.get("id", f"video_{int(time.time())}")
    title = script.get("title", "BitTrader")
    script_text = script.get("script", "")
    video_type = script.get("type", "short")

    # ── PRODUCER CONTENT GUARD: reject contaminated scripts ───────────────
    # These signals indicate the script contains LLM prompt/template text
    # instead of actual video narration content.
    CONTAMINATION_SIGNALS = [
        # LLM meta-instructions leaking into script
        "el usuario quiere",
        "el usuario necesita",
        "el usuario pide",
        "RESPOND ONLY WITH",
        "respond only with",
        "Analyze the Request",
        "~130 palabras",
        "~60 palabras",
        "~150 palabras",
        "~200 palabras",
        "especifica: TITULO",
        "especifico: TITULO",
        "TITULO, DESCRIPCION, TAGS",
        "VIDEO_PROMPT_",
        "VIDEO_PROMPT:",
        "You are a scriptwriter",
        "You are an expert",
        "Eres el guionista",
        "Eres un experto",
        "formato obligatorio",
        "canal de YouTube llamado BitTrader",
        "**Role:**",
        "**Format:**",
        "**Output:**",
        "**Tone:**",
        # Prompt template placeholders literally in script
        "específico: TITULO",
        "TITULO:",
        "DESCRIPCION:",
        "TAGS:",
        "GUION:",
        "GUIÓN:",
        # System prompt leak patterns
        "necesito seguir todas las reglas",
        "guión completo para un video",
        "siguiendo el formato",
        "sin texto adicional",
    ]
    contaminated = next((s for s in CONTAMINATION_SIGNALS if s.lower() in script_text.lower()), None)
    if contaminated:
        print(f"    🚫 PRODUCER GUARD: script contaminado detectado ('{contaminated[:40]}')")
        print(f"       Script preview: {script_text[:200]}")
        return {"status": "error", "error": f"CONTAMINATED_SCRIPT: '{contaminated[:40]}' found in script text"}

    if len(script_text.strip()) < 50:
        return {"status": "error", "error": f"EMPTY_SCRIPT: script too short ({len(script_text)} chars)"}

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
    
    # ── STEP 1.5: Fetch coin logo ──
    coin_symbol = extract_coin_symbol(script)
    coin_logo_path = None
    if coin_symbol:
        print(f"    🪙 Buscando logo de {coin_symbol}...")
        coin_logo_path = fetch_coin_logo(coin_symbol)

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
            print(f"    🐛 DEBUG: Entering assemble_video path (clips={len(clips)})")
            print(f"    🔧 Ensamblando {len(clips)} clips + audio...")
            success = assemble_video(
                clips, audio_path, video_path, title, duration,
                video_type, script_text, output_dir,
                coin_logo_path=coin_logo_path
            )
            if success:
                size_mb = video_path.stat().st_size / (1024 * 1024)
                print(f"    ✅ Video final: {size_mb:.1f}MB")
            else:
                print(f"    ⚠️ Ensamble falló, usando local clips fallback...")
                success = make_local_clips_video(
                    audio_path, video_path, title, duration,
                    video_type, script_text, output_dir,
                    coin_logo_path=coin_logo_path
                )
        else:
            print(f"    🐛 DEBUG: Entering local clips fallback (no AI clips generated)")
            print(f"    ⚠️ No se generaron clips con Hailuo — usando clips locales reales...")
            success = make_local_clips_video(
                audio_path, video_path, title, duration,
                video_type, script_text, output_dir,
                coin_logo_path=coin_logo_path
            )
    else:
        # Text-only mode — still use local clips, not solid color
        print(f"    🐛 DEBUG: use_ai_video=False — usando clips locales reales...")
        success = make_local_clips_video(
            audio_path, video_path, title, duration,
            video_type, script_text, output_dir,
            coin_logo_path=coin_logo_path
        )
    
    if not success or not video_path.exists():
        return {"status": "error", "error": "Video generation failed completely"}
    
    # NEW: CRITICAL - Validate video quality at the END (common to ALL paths)
    print(f"    🔍 Final quality check (validate_video_quality at produce_single end)...")
    if not validate_video_quality(video_path):
        print(f"    ❌ Video quality check FAILED - retrying with local clips...")
        # Re-try with local clips (NOT solid color)
        success = make_local_clips_video(
            audio_path, video_path, title, duration,
            video_type, script_text, output_dir,
            coin_logo_path=coin_logo_path
        )
        if not success or not video_path.exists():
            return {"status": "error", "error": "Video quality check failed and local clips retry failed"}
    
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
