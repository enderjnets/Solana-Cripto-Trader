"""
clip_strategy.py — Estrategia híbrida 50/50 Pexels + SDXL para BitTrader Producer
Orden de Ender 2026-03-28:
  - 50% clips reales de Pexels (stock footage landscape, no portrait)
  - 50% imágenes generadas con SDXL via HuggingFace → animadas con Ken Burns (zoompan)
  - Si HuggingFace falla → 100% Pexels como fallback automático
  - Encoding estándar: profile:v baseline, level 3.1, yuv420p, movflags +faststart
"""

import json
import os
import re
import math
import random
import requests
import subprocess
import tempfile
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).parent
_KEYS_DIR  = _THIS_DIR.parent / "keys"
_MEM_DIR   = _THIS_DIR / "memory"

# ── Pexels config ──────────────────────────────────────────────────────────────
try:
    PEXELS_KEY = json.loads((_KEYS_DIR / "pexels.json").read_text())["api_key"]
except Exception:
    PEXELS_KEY = os.environ.get("PEXELS_API_KEY", "")

PEXELS_RATIO  = 0.5   # 50% of clips from Pexels
CLIP_DURATION = 6     # seconds per clip in the final video
_PEXELS_HEADERS = {"Authorization": PEXELS_KEY}

# ── HuggingFace config ─────────────────────────────────────────────────────────
try:
    _HF_CFG = json.loads((_MEM_DIR / "huggingface_config.json").read_text())
    HF_API_KEY = _HF_CFG.get("api_key", "")
except Exception:
    HF_API_KEY = os.environ.get("HF_TOKEN", "")

# SDXL Inference endpoint (free tier)
HF_SDXL_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"

# ── Video encoding standard (from producer.py) ─────────────────────────────────
VIDEO_ENCODE_FLAGS = [
    "-c:v",    "libx264",
    "-preset", "fast",
    "-crf",    "23",
    "-profile:v", "baseline",
    "-level",  "3.1",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
]

# ── Trading keyword sets for Pexels (same style as test_pexels_v2) ─────────────
_PEXELS_KEYWORD_SETS = [
    "stock market trading", "bitcoin chart", "cryptocurrency trading",
    "financial charts", "trading terminal", "market analysis",
    "candlestick chart", "trading computer screens", "forex trading",
    "risk management business", "stop loss trading", "market crash",
    "red stock chart", "financial loss", "trading risk",
    "successful trader", "financial success", "profit growth",
    "green stock chart", "bitcoin profit", "financial freedom",
    "investment portfolio", "diversification finance", "money management",
    "trading strategy", "financial planning", "wealth management",
    "trading office", "financial district", "bank money",
    "laptop finance", "phone trading app", "crypto wallet",
    "blockchain technology", "digital currency", "fintech",
    "bitcoin gold coin", "ethereum logo", "cryptocurrency exchange",
    "bull market rally", "bear market decline", "market volatility",
    "trader stress", "success money", "passive income investing",
]


# ══════════════════════════════════════════════════════════════════════════════
# PEXELS UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _search_pexels(keyword: str, per_page: int = 3) -> list:
    """Search Pexels videos. Returns list of {url, width, height, duration, keyword, id}."""
    try:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers=_PEXELS_HEADERS,
            params={
                "query":        keyword,
                "per_page":     per_page,
                "orientation":  "landscape",
                "size":         "medium",
                "min_duration": 4,
                "max_duration": 20,
            },
            timeout=15,
        )
        if r.status_code != 200:
            return []
        results = []
        for v in r.json().get("videos", []):
            for vf in v.get("video_files", []):
                w = vf.get("width",  0)
                h = vf.get("height", 0)
                # Only LANDSCAPE HD clips (width > height and w >= 1280)
                if w >= 1280 and w > h and vf.get("file_type") == "video/mp4":
                    results.append({
                        "url":      vf["link"],
                        "width":    w,
                        "height":   h,
                        "duration": v.get("duration", 0),
                        "keyword":  keyword,
                        "id":       v["id"],
                    })
                    break  # best file per video
        return results
    except Exception as e:
        print(f"      ⚠️  Pexels '{keyword}': {e}")
        return []


def _download_clip(url: str, out_path: Path) -> bool:
    """Download a video clip from URL."""
    try:
        r = requests.get(url, stream=True, timeout=60)
        if r.status_code == 200:
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return True
    except Exception:
        pass
    return False


def _is_green_screen(video_path: Path) -> bool:
    """Returns True if clip is a pure green screen (reject it)."""
    try:
        import numpy as np
        from PIL import Image

        r = subprocess.run(
            ["ffmpeg", "-ss", "0.5", "-i", str(video_path),
             "-frames:v", "1", "-q:v", "2", "/tmp/_gs_check_cs.jpg", "-y"],
            capture_output=True, timeout=10,
        )
        check_file = Path("/tmp/_gs_check_cs.jpg")
        if not check_file.exists():
            return False
        img  = Image.open(check_file).convert("RGB")
        arr  = __import__("numpy").array(img, dtype=float)
        r_ch = arr[:, :, 0].mean()
        g_ch = arr[:, :, 1].mean()
        b_ch = arr[:, :, 2].mean()
        return g_ch > r_ch * 1.8 and g_ch > b_ch * 1.8 and g_ch > 100
    except Exception:
        return False


def _trim_and_scale_pexels(src: Path, dst: Path, video_type: str = "long") -> bool:
    """
    Trim clip to CLIP_DURATION seconds and scale to target resolution without black bars.
    long  → 1920×1080
    short → 1080×1920
    Uses scale+crop (lanczos) + setsar + fps=30.
    """
    if video_type == "short":
        vf = (
            "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
            "crop=1080:1920:(iw-1080)/2:(ih-1920)/2,"
            "setsar=1,fps=30"
        )
    else:
        vf = (
            "scale=1920:1080:force_original_aspect_ratio=increase:flags=lanczos,"
            "crop=1920:1080:(iw-1920)/2:(ih-1080)/2,"
            "setsar=1,fps=30"
        )

    r = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(src),
            "-t", str(CLIP_DURATION),
            "-vf", vf,
            *VIDEO_ENCODE_FLAGS,
            "-an",
            str(dst),
        ],
        capture_output=True, timeout=60,
    )
    return r.returncode == 0 and dst.exists()


def _get_pexels_clips(
    segments: list,
    n_needed: int,
    clips_dir: Path,
    video_type: str = "long",
) -> list:
    """
    Download and trim n_needed Pexels clips.
    Uses keywords extracted from segment texts + fallback keyword pool.
    Returns list of Path objects to trimmed .mp4 files.
    """
    # Build keyword list from segments first, then fall back to global pool
    seg_keywords = []
    for seg in segments:
        words = re.findall(r'\b[a-záéíóúñA-ZÁÉÍÓÚÑ]{4,}\b', seg.get("text", ""))
        seg_keywords += [w.lower() for w in words[:3]]

    # Combine segment keywords with trading pool (deduplicated)
    combined_kws = list(dict.fromkeys(seg_keywords + _PEXELS_KEYWORD_SETS))
    random.shuffle(combined_kws)

    all_clips_meta = []
    seen_ids: set = set()

    print(f"      🔍 Pexels: buscando {n_needed} clips (video_type={video_type})...")
    for kw in combined_kws:
        if len(all_clips_meta) >= n_needed:
            break
        results = _search_pexels(kw, per_page=3)
        for rc in results:
            if rc["id"] not in seen_ids:
                seen_ids.add(rc["id"])
                all_clips_meta.append(rc)

    clips_dir.mkdir(parents=True, exist_ok=True)
    ready_clips: list = []

    for i, meta in enumerate(all_clips_meta[:n_needed]):
        raw_path  = clips_dir / f"pexels_raw_{i:03d}.mp4"
        trim_path = clips_dir / f"pexels_{i:03d}.mp4"

        # Use cache if already trimmed
        if trim_path.exists() and trim_path.stat().st_size > 10_000:
            ready_clips.append(trim_path)
            continue

        if not _download_clip(meta["url"], raw_path):
            print(f"      ⚠️  Pexels clip {i}: descarga fallida")
            continue

        # Reject green screen
        if _is_green_screen(raw_path):
            print(f"      🟢 Pexels clip {i}: green screen rechazado")
            raw_path.unlink(missing_ok=True)
            continue

        # Trim + scale
        ok = _trim_and_scale_pexels(raw_path, trim_path, video_type)
        raw_path.unlink(missing_ok=True)

        if ok:
            size_kb = trim_path.stat().st_size // 1024
            print(f"      💾 Pexels {i}: {meta['keyword'][:30]} ({size_kb}KB)")
            ready_clips.append(trim_path)
        else:
            print(f"      ⚠️  Pexels clip {i}: scale/trim falló")

    print(f"      ✅ Pexels: {len(ready_clips)} clips listos")
    return ready_clips


# ══════════════════════════════════════════════════════════════════════════════
# SDXL / HUGGINGFACE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _build_sdxl_prompt(segment_text: str) -> str:
    """Build an SDXL prompt from segment text about crypto/trading topics."""
    text_lower = segment_text.lower()

    # Keyword → scene mapping for semantic sync
    KEYWORD_SCENES = {
        "bitcoin":   "golden Bitcoin coin glowing on a dark desk, cinematic dramatic lighting, ultra HD photography",
        "ethereum":  "Ethereum crystal hologram floating above a futuristic tech surface, blue glow, hyper-realistic",
        "trading":   "multiple monitor trading setup with crypto charts, dark trading desk, neon reflections",
        "profit":    "green profit chart rising sharply on screen, money coins scattered, success concept",
        "loss":      "red declining chart on a dark monitor, financial risk concept, moody lighting",
        "exchange":  "cryptocurrency exchange interface on laptop screen, digital finance, clean setup",
        "wallet":    "hardware cryptocurrency wallet on wooden desk, security concept, product photography",
        "estrategia":"chess pieces on a trading desk next to laptop with charts, strategy concept, dramatic light",
        "riesgo":    "warning red glow over financial documents and chart, risk concept, dark moody",
        "dca":       "calendar and coins arranged systematically, dollar cost averaging concept, clean bright",
        "blockchain":"abstract blockchain network visualization, blue neon nodes and connections, tech background",
    }

    for kw, scene in KEYWORD_SCENES.items():
        if kw in text_lower:
            return scene

    # Generic fallback — still finance/crypto themed
    _GENERIC = [
        "cryptocurrency coins and trading charts on dark background, professional finance photography, dramatic lighting",
        "bitcoin and ethereum logos on dark desktop, digital assets concept, macro photography, high contrast",
        "financial technology concept, abstract data streams and coins, blue and gold colors, professional",
        "trading terminal with green candles chart, dark room, monitor glow, finance photography",
        "crypto wallet app on phone, blurred city lights background, mobile finance, sharp product shot",
        "stock market candlestick pattern on dark background, trading concept, vivid neon colors",
    ]
    return random.choice(_GENERIC)


def _generate_sdxl_image(prompt: str, out_path: Path) -> bool:
    """
    Generate an image via Hugging Face SDXL Inference API.
    Returns True on success.
    """
    if not HF_API_KEY:
        print("      ⚠️  HuggingFace API key no configurada")
        return False

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs":     prompt,
        "parameters": {
            "num_inference_steps": 25,
            "guidance_scale":      7.5,
            "width":               1344,  # 16:9 landscape (1344x768 es nativo SDXL horizontal)
            "height":              768,
        },
    }

    try:
        print(f"      🎨 SDXL: generando imagen...")
        r = requests.post(HF_SDXL_URL, headers=headers, json=payload, timeout=120)

        if r.status_code == 503:
            # Model loading — wait and retry once
            print("      ⏳ SDXL cargando modelo, reintentando en 20s...")
            time.sleep(20)
            r = requests.post(HF_SDXL_URL, headers=headers, json=payload, timeout=120)

        if r.status_code != 200:
            print(f"      ⚠️  SDXL HTTP {r.status_code}: {r.text[:150]}")
            return False

        content_type = r.headers.get("content-type", "")
        if "image" not in content_type:
            print(f"      ⚠️  SDXL respuesta inesperada ({content_type}): {r.text[:100]}")
            return False

        out_path.write_bytes(r.content)
        size_kb = out_path.stat().st_size // 1024
        print(f"      🖼️  SDXL imagen guardada: {size_kb}KB")
        return True

    except Exception as e:
        print(f"      ⚠️  SDXL error: {e}")
        return False


def _image_to_kenburns_clip(img_path: Path, out_path: Path, video_type: str = "long",
                              clip_num: int = 0) -> bool:
    """
    Convert a static image to a 6-second video clip with Ken Burns effect (zoom + pan).
    Uses zoompan ffmpeg filter as specified in the task.
    long  → 1920×1080
    short → 1080×1920
    """
    if video_type == "short":
        target_w, target_h = 1080, 1920
    else:
        target_w, target_h = 1920, 1080

    # Vary KB direction per clip for visual interest
    KB_STYLES = [
        # Slow zoom in, centered
        f"zoompan=z='min(zoom+0.0015,1.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=150:s={target_w}x{target_h}",
        # Slow zoom out from center
        f"zoompan=z='if(lte(zoom,1.0),1.5,zoom-0.0015)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=150:s={target_w}x{target_h}",
        # Zoom in + pan right
        f"zoompan=z='min(zoom+0.0015,1.4)':x='if(lte(on,1),0,min(x+1,iw/4))':y='ih/2-(ih/zoom/2)':d=150:s={target_w}x{target_h}",
        # Zoom in + pan left
        f"zoompan=z='min(zoom+0.0015,1.4)':x='if(lte(on,1),iw/4,max(x-1,0))':y='ih/2-(ih/zoom/2)':d=150:s={target_w}x{target_h}",
    ]
    kb_expr = KB_STYLES[clip_num % len(KB_STYLES)]

    # Primero crop a aspect ratio correcto (16:9 para longs, 9:16 para shorts)
    # Esto garantiza que imágenes cuadradas (1024x1024) o verticales no tengan barras negras
    if video_type == "short":
        # Para shorts: crop a 9:16 (portrait) — iw*9/16 de alto o usar todo el alto
        crop_filter = f"crop=min(iw\\,ih*9/16):min(ih\\,iw*16/9):(iw-min(iw\\,ih*9/16))/2:(ih-min(ih\\,iw*16/9))/2,"
    else:
        # Para longs: crop a 16:9 (landscape) — asegura que no haya barras negras
        crop_filter = f"crop=min(iw\\,ih*16/9):min(ih\\,iw*9/16):(iw-min(iw\\,ih*16/9))/2:(ih-min(ih\\,iw*9/16))/2,"

    # scale to large canvas first (zoompan needs it), then encode
    vf = (
        f"{crop_filter}"
        f"scale=8000:-1:flags=lanczos,"
        f"{kb_expr},"
        f"setsar=1,fps=30"
    )

    r = subprocess.run(
        [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i",    str(img_path),
            "-vf",   vf,
            "-t",    str(CLIP_DURATION),
            *VIDEO_ENCODE_FLAGS,
            "-an",
            str(out_path),
        ],
        capture_output=True, timeout=120,
    )

    if r.returncode != 0:
        # Fallback: simpler scale without zoompan (faster, less memory)
        print(f"      ⚠️  zoompan falló (clip {clip_num}), usando scale simple...")
        vf_simple = (
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={target_w}:{target_h},"
            f"setsar=1,fps=30"
        )
        r2 = subprocess.run(
            [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i",    str(img_path),
                "-vf",   vf_simple,
                "-t",    str(CLIP_DURATION),
                *VIDEO_ENCODE_FLAGS,
                "-an",
                str(out_path),
            ],
            capture_output=True, timeout=60,
        )
        return r2.returncode == 0 and out_path.exists()

    return out_path.exists()


def _get_sdxl_clips(
    segments: list,
    n_needed: int,
    clips_dir: Path,
    video_type: str = "long",
) -> list:
    """
    Generate n_needed SDXL clips (image → Ken Burns video).
    Returns list of Path objects to .mp4 files.
    Falls back to empty list if HuggingFace is unavailable.
    """
    clips_dir.mkdir(parents=True, exist_ok=True)
    ready_clips: list = []

    # Distribute segments evenly — cycle if n_needed > len(segments)
    for i in range(n_needed):
        seg  = segments[i % len(segments)] if segments else {}
        text = seg.get("text", "") if isinstance(seg, dict) else str(seg)

        img_path  = clips_dir / f"sdxl_img_{i:03d}.jpg"
        clip_path = clips_dir / f"sdxl_{i:03d}.mp4"

        # Use cache
        if clip_path.exists() and clip_path.stat().st_size > 10_000:
            ready_clips.append(clip_path)
            continue

        prompt = _build_sdxl_prompt(text)
        ok_img = _generate_sdxl_image(prompt, img_path)

        if not ok_img:
            print(f"      ⚠️  SDXL clip {i}: imagen no generada — usando Pexels fallback")
            break  # signal caller to switch to all-Pexels

        ok_clip = _image_to_kenburns_clip(img_path, clip_path, video_type, clip_num=i)
        img_path.unlink(missing_ok=True)  # clean up image

        if ok_clip:
            size_kb = clip_path.stat().st_size // 1024
            print(f"      🎞️  SDXL clip {i}: {prompt[:40]}... ({size_kb}KB)")
            ready_clips.append(clip_path)
        else:
            print(f"      ⚠️  SDXL clip {i}: conversión a video fallida")

    print(f"      ✅ SDXL: {len(ready_clips)} clips generados")
    return ready_clips


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def get_clips_hybrid(
    segments,
    total_clips_needed: int,
    clips_dir,
    video_type: str = "long",
) -> list:
    """
    Obtiene clips usando estrategia híbrida 50/50:
      - La mitad de Pexels  (stock footage real)
      - La mitad generados con SDXL (imagen estática animada con zoom/pan Ken Burns)

    Si HuggingFace falla → 100% Pexels automáticamente.

    Parámetros:
        segments           : lista de segmentos del guion. Cada elemento puede ser
                             un dict con clave "text" o simplemente un string.
        total_clips_needed : número total de clips .mp4 que se necesitan.
        clips_dir          : Path (o str) donde guardar los clips descargados/generados.
        video_type         : "long" (1920×1080)  |  "short" (1080×1920)

    Retorna: lista de Path a clips .mp4 listos para usar en ffmpeg concat.
    """
    clips_dir = Path(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Normalise segments → list of dicts with "text" key
    norm_segs = []
    for s in (segments or []):
        if isinstance(s, dict):
            norm_segs.append(s)
        else:
            norm_segs.append({"text": str(s)})

    n_pexels = math.ceil(total_clips_needed * PEXELS_RATIO)
    n_sdxl   = total_clips_needed - n_pexels

    print(f"\n   🎬 clip_strategy: {total_clips_needed} clips total "
          f"({n_pexels} Pexels + {n_sdxl} SDXL | video_type={video_type})")

    # ── 1. Pexels clips ──────────────────────────────────────────────────
    pexels_dir = clips_dir / "pexels"
    pexels_clips = _get_pexels_clips(norm_segs, n_pexels, pexels_dir, video_type)

    # ── 2. SDXL clips ────────────────────────────────────────────────────
    sdxl_dir = clips_dir / "sdxl"
    sdxl_clips: list = []

    if n_sdxl > 0:
        sdxl_clips = _get_sdxl_clips(norm_segs, n_sdxl, sdxl_dir, video_type)

    # ── 3. Fallback: if SDXL failed, supplement with more Pexels ─────────
    if len(sdxl_clips) < n_sdxl:
        missing = n_sdxl - len(sdxl_clips)
        print(f"      ⚠️  SDXL entregó {len(sdxl_clips)}/{n_sdxl} — "
              f"completando con {missing} clips de Pexels")
        extra_pexels = _get_pexels_clips(
            norm_segs, n_pexels + missing, pexels_dir, video_type
        )
        # Extend only with NEW clips beyond what we already have
        new_extras = [c for c in extra_pexels if c not in pexels_clips]
        sdxl_clips.extend(new_extras[:missing])

    # ── 4. Merge, shuffle, return ─────────────────────────────────────────
    all_clips = pexels_clips + sdxl_clips
    random.shuffle(all_clips)

    # Trim to exactly what was requested (may have more than needed)
    result = all_clips[:total_clips_needed]

    # Pad if still short (repeat existing clips)
    if len(result) < total_clips_needed and result:
        while len(result) < total_clips_needed:
            result += result
        result = result[:total_clips_needed]

    print(f"   ✅ clip_strategy: {len(result)} clips listos "
          f"({sum(1 for c in result if 'pexels' in c.name)} Pexels, "
          f"{sum(1 for c in result if 'sdxl' in c.name)} SDXL)\n")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MiniMax image-01 — Pure AI clips (orden Ender 2026-03-29)
# 100% generated with MiniMax image-01 + Ken Burns animation

def _build_visual_prompt(seg_text: str, video_type: str = "long") -> str:
    """
    Convert segment text into a vivid visual prompt for MiniMax image generation.
    Handles trading/financial content with specific visual descriptors.
    """
    text = seg_text.lower().strip()

    # Trading/financial keywords → specific visuals
    trading_keywords = {
        "trading": "professional trader at computer with multiple screens showing stock charts and candlestick patterns, dramatic lighting",
        "bitcoin": "golden Bitcoin coin on dark background with green arrow pointing up, dramatic cinematic lighting, 4K quality",
        "ethereum": "ethereum cryptocurrency logo glowing on dark blue background with geometric patterns, dramatic lighting",
        "crypto": "cryptocurrency concept with multiple coins and charts on dark background, green and gold colors, dramatic lighting",
        "bolsa": "stock market trading floor with screens showing stock prices rising, professional photography",
        "bolsas": "stock market trading floor with screens showing stock prices, professional photography",
        "invertir": "investor analyzing financial charts on large monitor, professional office, dramatic lighting",
        "inversion": "investment growth chart with green arrow pointing up, gold coins, dark background, dramatic lighting",
        "dinero": "stack of gold coins and dollar bills on dark background, dramatic lighting, cinematic",
        "ganar dinero": "golden coins growing from laptop computer, dark background, dramatic lighting",
        "perder dinero": "red stock chart declining with worried investor silhouette, dark dramatic background",
        "trader": "professional trader at desk with multiple monitors showing trading charts, dramatic lighting",
        " Wall Street": "Wall Street sign with New York stock exchange building, dramatic urban photography",
        "nasdaq": "NASDAQ sign glowing blue at night, New York city skyline, dramatic photography",
        "media movil": "stock chart with moving average line highlighted in bright yellow, green upward trend, professional trading platform",
        "media movil": "candlestick stock chart with moving average line in orange, professional trading software, dark background",
        "cruza": "stock chart showing moving average crossover point highlighted with bright circle, green arrow, professional trading platform",
        " compra ": "green buy button on trading platform with stock chart rising, professional trading software",
        "venta": "red sell button on trading platform with stock chart, professional trading software",
        "ethereum": "ethereum logo glowing purple on dark background with geometric patterns",
        "solana": "solana blockchain logo with geometric sun rays on dark blue gradient background, dramatic lighting",
        "altcoin": "variety of altcoin logos including Ethereum, Solana, Cardano on dark background with green glow",
        "tokens": "cryptocurrency tokens floating in space on dark background with geometric patterns and glow effects",
        "exchange": "cryptocurrency exchange trading platform on multiple screens, professional setup, dark room",
        " Whale ": "large whale swimming in ocean at sunset, dramatic photography",  # crypto whale
        " Whale": "large whale in ocean at sunset, dramatic nature photography",
        "robot": "futuristic robot working on laptop computer, dark background, neon lighting, sci-fi style",
        "ia artificial": "artificial intelligence visualization with glowing neural network and brain circuits on dark blue background",
        "inteligencia artificial": "artificial intelligence visualization with glowing circuits and data streams on dark background",
        "agentes ia": "humanoid robot talking to human at computer, futuristic office, dramatic lighting",
        "trading bot": "AI trading bot interface on screen with charts and automated trading signals, dark theme",
        "bot de trading": "AI trading bot interface showing automated buy/sell signals on stock charts, dark professional theme",
        "chart": "detailed stock chart with candlesticks and technical indicators on dark background, professional trading",
        "tradingview": "trading chart on TradingView platform with indicators, dark theme, professional look",
        "portafolio": "investment portfolio showing diversified holdings with pie charts and growth graphs, professional",
        "balance": "financial balance scale with gold coins on one side, dark background, dramatic lighting",
        "balance": "trading account balance showing positive PnL with green numbers on dark background",
        "broker": "professional forex broker trading at multiple screens, modern office, dramatic lighting",
        "broker": "online broker trading platform on laptop with charts and financial data, modern minimal style",
        "alza": "stock market surge with arrow pointing sharply upward, green light rays, dramatic",
        "baja": "stock market decline with red arrow pointing down, red light, dramatic dark background",
        "mercado": "global financial market concept with interconnected nodes and charts on dark blue background",
        "sube": "stock chart with sharp green line pointing upward, dramatic lighting, bullish momentum",
        "cae": "stock chart with red line pointing downward, bearish trend, dramatic dark red background",
        "grafico": "detailed financial chart with candlesticks on dark background, professional trading platform",
        "precio": "cryptocurrency price chart with glowing numbers and trend arrows on dark background",
        "analisis": "financial analyst studying stock charts on multiple screens, professional office, dramatic lighting",
        "estrategia": "chess board with stock charts and trading plan documents, strategic planning concept",
        "riesgo": "risk management concept with stop-loss levels on trading chart, professional dark theme",
        "apalancamiento": "leverage trading concept with magnifying glass over stock charts, dramatic lighting",
        "futuros": "futures trading platform showing contract prices, professional dark theme",
        "forex": "forex trading platform showing currency pairs with trend lines, professional dark theme",
        "opciones": "options trading chain with strike prices on professional trading platform, dark theme",
        "metaverso": "futuristic metaverse world with digital avatars and VR headsets, sci-fi dark environment",
        "web3": "Web3 concept with blockchain network visualization on dark background with neon blue lighting",
        "defi": "DeFi decentralized finance concept with floating coins and blockchain network on dark background",
        "nft": "NFT digital art concept with glowing digital artwork frame on dark background, colorful",
        " staking ": "cryptocurrency staking concept with locked coins in glowing lock on blockchain, dark background",
    }

    # Check for keywords and build enhanced prompt
    for keyword, visual in trading_keywords.items():
        if keyword in text:
            # Add subject context
            if any(w in text for w in ["por que", "porque", "razón", "verdad", "error", "fracaso"]):
                return f"{visual}, emotional narrative photography, powerful and dramatic composition"
            elif any(w in text for w in ["como", "manera", "forma", "estrategia", "secret"]):
                return f"{visual}, tutorial style photography, clear and professional, dramatic lighting"
            elif any(w in text for w in ["qué es", "que es", "introduccion", "Guía"]):
                return f"{visual}, educational clear illustration, professional quality, dramatic lighting"
            else:
                return f"{visual}, cinematic photography, dramatic lighting, 4K quality"

    # Generic fallback — but still make it visually interesting for trading/tech
    if video_type == "short":
        return f"futuristic trading concept with abstract financial data streams on dark background, green and gold neon lights, cinematic dramatic lighting, 4K"
    else:
        return f"professional financial technology concept with abstract charts and data visualization on dark blue background, dramatic cinematic lighting, 4K quality"


# ══════════════════════════════════════════════════════════════════════════════

def _get_minimax_key() -> str:
    """Load MiniMax Token Plan API key."""
    try:
        import json
        keys_file = _THIS_DIR.parent / "keys" / "minimax.json"
        return json.loads(keys_file.read_text())["minimax_token_plan_key"]
    except Exception:
        return ""


def _generate_minimax_image(prompt: str, out_path: Path,
                             resolution: str = "1024x1024",
                             aspect_ratio: str = "16:9") -> bool:
    """Generate image with MiniMax image-01 API."""
    api_key = _get_minimax_key()
    if not api_key:
        print(f"      ⚠️ MiniMax: no API key found")
        return False

    try:
        payload = {
            "model": "image-01",
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "prompt_optimizer": True,
        }
        r = requests.post(
            "https://api.minimax.io/v1/image_generation",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload, timeout=120
        )
        data = r.json()
        if r.status_code != 200 or data.get("base_resp", {}).get("status_code") != 0:
            print(f"      ⚠️ MiniMax HTTP {r.status_code}: {str(data)[:100]}")
            return False

        urls = data.get("data", {}).get("image_urls", [])
        if not urls:
            return False

        img_resp = requests.get(urls[0], timeout=120)
        img_resp.raise_for_status()
        out_path.write_bytes(img_resp.content)
        return True
    except Exception as e:
        print(f"      ⚠️ MiniMax error: {e}")
        return False


def get_clips_minimax(
    segments,
    total_clips_needed: int,
    clips_dir,
    video_type: str = "long",
) -> list:
    """
    Genera clips 100% con MiniMax image-01 + Ken Burns.
    Sin Pexels, sin HuggingFace SDXL.

    Parámetros:
        segments: lista de segmentos del guión
        total_clips_needed: número de clips necesarios
        clips_dir: Path donde guardar clips
        video_type: "long" (1920x1080) | "short" (1080x1920)

    Retorna: lista de Path a clips .mp4
    """
    clips_dir = Path(clips_dir)
    minimax_dir = clips_dir / "minimax"
    minimax_dir.mkdir(parents=True, exist_ok=True)

    # Normalise segments
    norm_segs = []
    for s in (segments or []):
        if isinstance(s, dict):
            norm_segs.append(s)
        else:
            norm_segs.append({"text": str(s)})

    # Calculate clips per segment
    clips_per_seg = max(1, total_clips_needed // max(1, len(norm_segs)))
    clip_dur = CLIP_DURATION  # 6 seconds default

    print(f"\n   🎨 clip_strategy: MiniMax image-01 ({total_clips_needed} clips | video_type={video_type})")

    clips = []
    for seg_idx, seg in enumerate(norm_segs):
        seg_text = seg.get("text", "")
        # Generate 1 image per segment, use as 1+ clips with Ken Burns variation
        img_path = minimax_dir / f"seg_{seg_idx:03d}.png"

        # Retry up to 2 times
        ok = False
        for attempt in range(3):
            aspect = "9:16" if video_type == "short" else "16:9"
            res = "1080x1920" if video_type == "short" else "1920x1080"
            visual_prompt = _build_visual_prompt(seg_text, video_type)
            if _generate_minimax_image(visual_prompt, img_path, resolution=res, aspect_ratio=aspect):
                ok = True
                break
            if attempt < 2:
                print(f"      🔄 Reintentando clip {seg_idx}...")
                time.sleep(2)

        if not ok:
            # Fallback: use a black frame
            print(f"      ⚠️ MiniMax falló para seg {seg_idx} — usando fallback")
            img_path = minimax_dir / f"seg_{seg_idx:03d}.png"
            # Create black image as fallback
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=black:s=1920x1080:d=1",
                     "-frames:v", "1", str(img_path)],
                    capture_output=True, timeout=10
                )
            except Exception:
                pass

        # Create 1-2 Ken Burns variants per image
        n_variants = min(2, max(1, clips_per_seg))
        for var in range(n_variants):
            if len(clips) >= total_clips_needed:
                break
            mp4_path = minimax_dir / f"clip_{seg_idx:03d}_{var}.mp4"
            kb_ok = _image_to_kenburns(img_path, mp4_path, seg_idx + var, clip_dur, video_type)
            if kb_ok:
                clips.append(mp4_path)
                print(f"      🎞️ MiniMax clip {len(clips)}: OK ({mp4_path.stat().st_size//1024}KB)")

    # Pad if short
    while len(clips) < total_clips_needed and clips:
        clips.append(clips[len(clips) % len(clips)])

    result = clips[:total_clips_needed]
    print(f"   ✅ clip_strategy: {len(result)} clips MiniMax listos\n")
    return result



# ══════════════════════════════════════════════════════════════════════════════
# MiniMax image-01 — Pure AI clips (orden Ender 2026-03-29)
# 100% generated with MiniMax image-01 + Ken Burns animation
# ══════════════════════════════════════════════════════════════════════════════

def _get_minimax_key() -> str:
    """Load MiniMax Token Plan API key."""
    try:
        import json
        keys_file = _THIS_DIR.parent / "keys" / "minimax.json"
        return json.loads(keys_file.read_text())["minimax_token_plan_key"]
    except Exception:
        return ""


def _generate_minimax_image(prompt: str, out_path: Path,
                             resolution: str = "1024x1024",
                             aspect_ratio: str = "16:9") -> bool:
    """Generate image with MiniMax image-01 API."""
    api_key = _get_minimax_key()
    if not api_key:
        print(f"      WARNING: MiniMax no API key found")
        return False
    try:
        payload = {
            "model": "image-01",
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "prompt_optimizer": True,
        }
        r = requests.post(
            "https://api.minimax.io/v1/image_generation",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload, timeout=120
        )
        data = r.json()
        if r.status_code != 200 or data.get("base_resp", {}).get("status_code") != 0:
            print(f"      WARNING: MiniMax HTTP {r.status_code}")
            return False
        urls = data.get("data", {}).get("image_urls", [])
        if not urls:
            return False
        img_resp = requests.get(urls[0], timeout=120)
        img_resp.raise_for_status()
        out_path.write_bytes(img_resp.content)
        return True
    except Exception as e:
        print(f"      WARNING: MiniMax error: {e}")
        return False


# Ken Burns styles per video type — YouTube correct resolutions
KB_STYLES_SHORT = [   # 1080×1920 (9:16 YouTube Shorts)
    "zoompan=z='min(zoom+0.0015,1.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=150:s=1080x1920",
    "zoompan=z='if(lte(zoom,1.0),1.5,zoom-0.0015)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=150:s=1080x1920",
]
KB_STYLES_LONG = [    # 1920×1080 (16:9 YouTube HD)
    "zoompan=z='min(zoom+0.0015,1.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=150:s=1920x1080",
    "zoompan=z='if(lte(zoom,1.0),1.5,zoom-0.0015)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=150:s=1920x1080",
]


def _image_to_kenburns(img_path: Path, out_path: Path,
                       clip_num: int, duration: float,
                       video_type: str = "long") -> bool:
    """Convert image to Ken Burns MP4 clip at correct YouTube resolution."""
    kb_styles = KB_STYLES_SHORT if video_type == "short" else KB_STYLES_LONG
    kb = kb_styles[clip_num % len(kb_styles)]
    vf = f"scale=8000:-1:flags=lanczos,{kb},setsar=1,fps=30"
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-loop", "1", "-i", str(img_path),
             "-vf", vf, "-t", str(duration),
             "-c:v", "libx264", "-preset", "fast", "-crf", "20",
             "-profile:v", "high", "-level", "4.1",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
             str(out_path)],
            capture_output=True, timeout=120,
        )
        if r.returncode == 0 and out_path.exists():
            return True
        print(f"      WARNING: KenBurns ffmpeg error")
        return False
    except Exception as e:
        print(f"      WARNING: KenBurns error: {e}")
        return False


def get_clips_minimax(
    segments,
    total_clips_needed: int,
    clips_dir,
    video_type: str = "long",
) -> list:
    """
    Genera clips 100% con MiniMax image-01 + Ken Burns.
    Sin Pexels, sin HuggingFace SDXL.
    """
    clips_dir = Path(clips_dir)
    minimax_dir = clips_dir / "minimax"
    minimax_dir.mkdir(parents=True, exist_ok=True)

    norm_segs = []
    for s in (segments or []):
        if isinstance(s, dict):
            norm_segs.append(s)
        else:
            norm_segs.append({"text": str(s)})

    clips_per_seg = max(1, total_clips_needed // max(1, len(norm_segs)))
    clip_dur = CLIP_DURATION

    print(f"   MINIMAX: {total_clips_needed} clips (video_type={video_type})")

    clips = []
    for seg_idx, seg in enumerate(norm_segs):
        seg_text = seg.get("text", "")
        img_path = minimax_dir / f"seg_{seg_idx:03d}.png"

        ok = False
        for attempt in range(3):
            aspect = "9:16" if video_type == "short" else "16:9"
            res = "1080x1920" if video_type == "short" else "1920x1080"
            visual_prompt = _build_visual_prompt(seg_text, video_type)
            if _generate_minimax_image(visual_prompt, img_path, resolution=res, aspect_ratio=aspect):
                ok = True
                break
            if attempt < 2:
                time.sleep(2)

        if not ok:
            print(f"      WARNING: MiniMax fall seg {seg_idx}")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=black:s=1920x1080:d=1",
                     "-frames:v", "1", str(img_path)],
                    capture_output=True, timeout=10
                )
            except Exception:
                pass

        n_variants = min(2, max(1, clips_per_seg))
        for var in range(n_variants):
            if len(clips) >= total_clips_needed:
                break
            mp4_path = minimax_dir / f"clip_{seg_idx:03d}_{var}.mp4"
            kb_ok = _image_to_kenburns(img_path, mp4_path, seg_idx + var, clip_dur, video_type)
            if kb_ok:
                clips.append(mp4_path)

    while len(clips) < total_clips_needed and clips:
        clips.append(clips[len(clips) % len(clips)])

    result = clips[:total_clips_needed]
    print(f"   DONE: {len(result)} clips MiniMax ready")
    return result


# ── Hybrid Mode: MiniMax + LTX-Video ─────────────────────────────────────────

def get_clips_hybrid_v2(
    segments,
    total_clips_needed: int,
    clips_dir,
    video_type: str = "long",
    ltx_ratio: float = 0.3,
) -> list:
    """
    Pipeline híbrido v2: MiniMax image-01 + Ken Burns (default) + LTX-Video (dynamic scenes).
    
    - ltx_ratio: fracción de clips generados con LTX-Video (0.0 a 1.0)
    - Escenas con keywords dinámicas → LTX-Video
    - El resto → MiniMax image-01 + Ken Burns
    - Si Mac no está disponible → 100% MiniMax fallback
    """
    from pathlib import Path
    
    clips_dir = Path(clips_dir)
    
    # Check if LTX-Video is available
    ltx_available = False
    try:
        from ltx_client import is_available
        ltx_available = is_available()
    except Exception:
        pass
    
    if not ltx_available:
        print("   ⚠️ LTX-Video not available, using 100% MiniMax")
        return get_clips_minimax(segments, total_clips_needed, clips_dir, video_type)
    
    # Classify segments: dynamic (LTX) vs static (MiniMax)
    dynamic_keywords = [
        "spinning", "rotating", "moving", "walking", "running", "trading",
        "explosion", "celebration", "action", "typing", "coding", "scrolling",
        "gira", "mueve", "corre", "explota", "celebra", "escribe", "teclea",
    ]
    
    norm_segs = []
    for s in (segments or []):
        if isinstance(s, dict):
            norm_segs.append(s)
        else:
            norm_segs.append({"text": str(s)})
    
    # Determine which segments are dynamic
    ltx_indices = []
    for i, seg in enumerate(norm_segs):
        text_lower = seg.get("text", "").lower()
        if any(kw in text_lower for kw in dynamic_keywords):
            ltx_indices.append(i)
    
    # Cap at ltx_ratio
    max_ltx = max(1, int(total_clips_needed * ltx_ratio))
    ltx_indices = ltx_indices[:max_ltx]
    
    print(f"\n   🎬 clip_strategy HYBRID v2: {len(ltx_indices)} LTX + {total_clips_needed - len(ltx_indices)} MiniMax")
    
    # Generate LTX clips
    ltx_clips = {}
    if ltx_indices:
        from ltx_client import generate_clip
        import random
        ltx_dir = clips_dir / "ltx"
        ltx_dir.mkdir(parents=True, exist_ok=True)
        
        for idx in ltx_indices:
            seg_text = norm_segs[idx].get("text", "trading scene")
            # Create English prompt for LTX (it works better in English)
            prompt = f"{seg_text}, cinematic lighting, high quality, 4K"
            mp4_path = ltx_dir / f"ltx_seg_{idx:03d}.mp4"
            
            ok = generate_clip(
                prompt=prompt,
                output_path=mp4_path,
                # YouTube correct resolutions: LONG 1920x1080, SHORT 1080x1920
                # LTX-Video 2B works best at 16:9 or 9:16 ratios
                width=1280 if video_type == "long" else 540,
                height=720 if video_type == "long" else 960,
                num_frames=97,
                steps=12,
                seed=random.randint(0, 2**32),
            )
            if ok:
                ltx_clips[idx] = mp4_path
    
    # Generate MiniMax clips for the rest
    minimax_indices = [i for i in range(len(norm_segs)) if i not in ltx_clips]
    minimax_segs = [norm_segs[i] for i in minimax_indices]
    minimax_needed = total_clips_needed - len(ltx_clips)
    
    minimax_clips_list = get_clips_minimax(
        minimax_segs, minimax_needed, clips_dir, video_type
    )
    
    # Merge clips in order
    all_clips = []
    mm_idx = 0
    clips_per_seg = max(1, total_clips_needed // max(1, len(norm_segs)))
    
    for i in range(len(norm_segs)):
        if i in ltx_clips:
            all_clips.append(ltx_clips[i])
        else:
            for _ in range(clips_per_seg):
                if mm_idx < len(minimax_clips_list):
                    all_clips.append(minimax_clips_list[mm_idx])
                    mm_idx += 1
    
    # Pad if needed
    while len(all_clips) < total_clips_needed and all_clips:
        all_clips.append(all_clips[len(all_clips) % len(all_clips)])
    
    result = all_clips[:total_clips_needed]
    print(f"   DONE: {len(result)} hybrid clips ({len(ltx_clips)} LTX + {len(result) - len(ltx_clips)} MiniMax)")
    return result
