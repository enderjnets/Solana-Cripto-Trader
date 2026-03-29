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
_KEYS_DIR  = _THIS_DIR / "keys"
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
    long  → 1280×720
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
            "scale=1280:720:force_original_aspect_ratio=increase:flags=lanczos,"
            "crop=1280:720:(iw-1280)/2:(ih-720)/2,"
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
            "width":               1024,
            "height":              576,  # ~16:9 for longs (will be scaled by ffmpeg)
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
    long  → 1280×720
    short → 1080×1920
    """
    if video_type == "short":
        target_w, target_h = 1080, 1920
    else:
        target_w, target_h = 1280, 720

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

    # scale to large canvas first (zoompan needs it), then encode
    vf = (
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
        video_type         : "long" (1280×720)  |  "short" (1080×1920)

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
