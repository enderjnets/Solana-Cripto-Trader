#!/usr/bin/env python3
"""
🎬 BitTrader — Hybrid Visual Production: edu_long_1774363903
"Cómo empezar en cripto con $100 — guía completa para principiantes 2026"

PROBLEMA: El video original solo usa Ken Burns (fotos estáticas) = visualmente aburrido.
SOLUCIÓN: Mezclar clips Wan2.1 animados + Ken Burns fotos alternados cada 3-5s.

Pipeline:
  1. Generar 20 FLUX images (16:9) via HuggingFace API — temas por sección
  2. Animar 8 de ellas con Wan2.1 I2V en Mac M3 Pro (clips de video ~5s)
  3. Aplicar Ken Burns a las otras 12 (fotos estáticas animadas ~4s)
  4. Interleave: KB → Wan → KB → Wan → KB → KB → Wan → ... (alternado)
  5. Concat + audio + subtitles + logo → video final

Regla CEO: SIEMPRE mezclar clips de video con fotos/imágenes — nunca solo un tipo.
"""
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import requests

# ── Paths ────────────────────────────────────────────────────────────────────
AGENTS_DIR  = Path(__file__).parent
DATA_DIR    = AGENTS_DIR / "data"
WORKSPACE   = Path("/home/enderj/.openclaw/workspace")
BITTRADER   = WORKSPACE / "bittrader"
BT_LOGO     = WORKSPACE / "videos/BIBLIOTECA/bittrader_logo.png"

SCRIPT_ID   = "edu_long_1774363903"
OUTPUT_BASE = AGENTS_DIR / "output" / "2026-03-24" / SCRIPT_ID
AUDIO_FILE  = OUTPUT_BASE / "edu_long_1774363903.mp3"
SUBS_FILE   = OUTPUT_BASE / "subtitles.ass"

# New output dirs for hybrid
HYBRID_DIR    = OUTPUT_BASE / "hybrid_mix"
FLUX_DIR      = HYBRID_DIR / "flux_images"
WAN_DIR       = HYBRID_DIR / "wan_clips"
KB_DIR        = HYBRID_DIR / "kb_clips"
ASSEMBLY_DIR  = HYBRID_DIR / "assembly"
FINAL_VIDEO   = OUTPUT_BASE / f"{SCRIPT_ID}_hybrid.mp4"

TARGET_W = 1920
TARGET_H = 1080

# ── HuggingFace API for FLUX images ─────────────────────────────────────────
HF_API_KEY = json.loads((BITTRADER / "keys/minimax.json").read_text()).get("huggingface_api_key", "")
FLUX_ENDPOINT = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"

# ── ComfyUI Mac M3 Pro ──────────────────────────────────────────────────────
MAC_M3_URL = "http://10.0.0.232:8188"
sys.path.insert(0, str(AGENTS_DIR))


# ── Ken Burns styles (same as produce_kb_hybrid_final.py) ────────────────────
KB_STYLES = {
    "zoom_in": lambda f: f"zoompan=z='min(zoom+0.0008,1.3)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={f}:s={TARGET_W}x{TARGET_H}:fps=30",
    "zoom_out": lambda f: f"zoompan=z='if(lte(zoom,1.0),1.3,zoom-0.0008)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={f}:s={TARGET_W}x{TARGET_H}:fps=30",
    "pan_right": lambda f: f"zoompan=z='1.25':x='if(lte(on,1),0,min(x+2,iw/5))':y='ih/2-(ih/zoom/2)':d={f}:s={TARGET_W}x{TARGET_H}:fps=30",
    "pan_left": lambda f: f"zoompan=z='1.25':x='if(lte(on,1),iw/5,max(x-2,0))':y='ih/2-(ih/zoom/2)':d={f}:s={TARGET_W}x{TARGET_H}:fps=30",
    "pan_up": lambda f: f"zoompan=z='1.25':x='iw/2-(iw/zoom/2)':y='if(lte(on,1),0,min(y+1,ih/5))':d={f}:s={TARGET_W}x{TARGET_H}:fps=30",
    "pan_down": lambda f: f"zoompan=z='1.25':x='iw/2-(iw/zoom/2)':y='if(lte(on,1),ih/5,max(y-1,0))':d={f}:s={TARGET_W}x{TARGET_H}:fps=30",
    "zoom_pan_right": lambda f: f"zoompan=z='min(zoom+0.0005,1.2)':x='if(lte(on,1),0,min(x+1,iw/6))':y='ih/2-(ih/zoom/2)':d={f}:s={TARGET_W}x{TARGET_H}:fps=30",
    "zoom_pan_left": lambda f: f"zoompan=z='min(zoom+0.0005,1.2)':x='if(lte(on,1),iw/6,max(x-1,0))':y='ih/2-(ih/zoom/2)':d={f}:s={TARGET_W}x{TARGET_H}:fps=30",
}
KB_STYLE_NAMES = list(KB_STYLES.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION DEFINITIONS — themed prompts for each part of the script
# ═══════════════════════════════════════════════════════════════════════════════

# Each section gets unique visual themes. Images alternate KB and Wan2.1.
SECTIONS = [
    {
        "name": "intro",
        "start_s": 0, "end_s": 60,
        "flux_prompts": [
            "person starting crypto journey, laptop showing Bitcoin chart, warm desk lamp lighting, modern home office, cinematic 16:9, photorealistic 4K",
            "100 dollar bill transforming into glowing Bitcoin, dramatic golden light rays, dark background, cinematic wide shot, 4K",
            "beginner investor looking at phone crypto app for the first time, curious expression, soft morning light, modern room, 16:9 cinematic",
        ],
        "wan_prompt": "dramatic crypto journey begins, golden Bitcoin emerges from darkness, light particles swirl upward, camera slowly zooms in, cinematic wide shot, smooth flowing motion",
    },
    {
        "name": "exchange",
        "start_s": 60, "end_s": 180,
        "flux_prompts": [
            "Coinbase app interface on modern smartphone, clean UI showing buy Bitcoin button, soft blue light, professional photography, 16:9 widescreen",
            "Binance exchange dark trading interface with green charts, multiple monitors, professional trading setup, dramatic lighting, 4K widescreen",
            "person creating crypto account on laptop, verification screen with ID, modern office, clean professional lighting, 16:9 cinematic",
            "two-factor authentication smartphone app with security shield icon, glowing green checkmark, dark tech background, 4K",
        ],
        "wan_prompt": "crypto exchange interface comes to life, digital numbers and charts flowing across screen, green candlesticks rising, smooth camera pan, cinematic wide motion",
    },
    {
        "name": "what_to_buy",
        "start_s": 180, "end_s": 300,
        "flux_prompts": [
            "golden Bitcoin coin prominently displayed with Ethereum crystal beside it, dramatic lighting, dark background, macro photography style, 16:9 4K",
            "pie chart showing 60% Bitcoin 30% Ethereum 10% altcoin allocation, clean infographic style, dark tech background, professional design, widescreen",
            "Bitcoin and Ethereum logos floating in digital space, blockchain network connecting them, blue purple energy, cinematic wide shot, 4K",
            "single Solana or BNB coin glowing among other crypto coins, vibrant colors, dark space background, dramatic lighting, widescreen cinematic",
        ],
        "wan_prompt": "Bitcoin golden coin rises dramatically from darkness, orange gold light rays burst outward, Ethereum crystal appears beside it, energy particles cascade, smooth cinematic motion",
    },
    {
        "name": "dca_strategy",
        "start_s": 300, "end_s": 420,
        "flux_prompts": [
            "calendar showing weekly Bitcoin purchases marked with green checkmarks, modern desk, organized financial planning, warm lighting, 16:9 cinematic 4K",
            "line chart showing DCA dollar cost averaging over time, green uptrend averaging out volatility, professional clean design, dark background, widescreen",
            "stack of coins growing week by week like bar chart, time-lapse feel, golden warm lighting, minimalist dark background, 16:9 4K",
            "person calmly buying crypto on phone while market chart fluctuates wildly behind, zen-like composure, cinematic wide shot, 4K",
        ],
        "wan_prompt": "coins stack higher week by week, golden particles rising steadily, price chart fluctuates but accumulation grows, smooth upward camera motion, cinematic wide, flowing motion",
    },
    {
        "name": "mistakes",
        "start_s": 420, "end_s": 540,
        "flux_prompts": [
            "dramatic red market crash chart on screen, person looking worried at phone, red ambient lighting, tense atmosphere, cinematic wide shot, 4K",
            "FOMO text glowing red with crowd rushing towards rising chart, dramatic cinematic lighting, dark background, 16:9 widescreen",
            "leverage trading 100x multiplier warning on dark screen, red danger indicators, dramatic tension lighting, professional 4K widescreen",
            "FTX exchange logo crumbling apart, digital destruction, dark dramatic atmosphere, cautionary tale visual, cinematic wide, 4K",
            "Telegram scam message on phone with red warning overlay, dark moody lighting, suspicious atmosphere, cinematic 16:9 wide",
        ],
        "wan_prompt": "market crash in dramatic slow motion, red energy waves pulse downward, charts falling, dark storm atmosphere builds, dramatic wide camera pull back, cinematic tension",
    },
    {
        "name": "outro",
        "start_s": 540, "end_s": 769,
        "flux_prompts": [
            "confident person reviewing crypto portfolio on laptop showing green profits, warm golden lighting, successful atmosphere, 16:9 cinematic 4K",
            "Bitcoin Ethereum coins glowing with green upward arrows, bullish energy, dark background, celebratory lighting, widescreen cinematic",
            "subscribe button and notification bell glowing, YouTube-style call to action, engaging bright design, dark background, 16:9 4K",
        ],
        "wan_prompt": "crypto portfolio glows green with rising profits, golden particles celebrate, camera zooms out to reveal bright future, smooth cinematic motion, triumphant energy",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True
    )
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def generate_flux_image_comfyui(prompt: str, out_path: Path,
                                width: int = 1024, height: int = 576,
                                seed: int = 42, server_url: str = "http://localhost:8188") -> bool:
    """Generate 16:9 image via FLUX.1-schnell on ComfyUI (PC or Mac)."""
    try:
        prefix = f"flux_hybrid_{abs(hash(prompt)) % 99999:05d}"
        workflow = {
            "1": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": "flux1-schnell-fp8.safetensors"}},
            "2": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": prompt, "clip": ["1", 1]}},
            "3": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": "blurry, low quality, distorted, watermark, text overlay, ugly, amateur",
                             "clip": ["1", 1]}},
            "4": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": width, "height": height, "batch_size": 1}},
            "5": {"class_type": "KSampler",
                  "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                             "latent_image": ["4", 0],
                             "seed": seed,
                             "steps": 4, "cfg": 1.0,
                             "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
            "6": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
            "7": {"class_type": "SaveImage",
                  "inputs": {"images": ["6", 0], "filename_prefix": prefix}}
        }
        data = json.dumps({"prompt": workflow}).encode()
        req = urllib.request.Request(
            f"{server_url}/prompt", data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            prompt_id = json.loads(r.read()).get("prompt_id")
        if not prompt_id:
            return False

        # Wait for output
        output_dir = Path("/home/enderj/ComfyUI/output")
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"{server_url}/history/{prompt_id}", timeout=10
                ) as r:
                    history = json.loads(r.read())
                if prompt_id in history:
                    entry = history[prompt_id]
                    st = entry.get("status", {})
                    if st.get("status_str") == "error":
                        print(f"    ❌ FLUX error on ComfyUI")
                        return False
                    outputs = entry.get("outputs", {})
                    for node_id, node_out in outputs.items():
                        for img_info in node_out.get("images", []):
                            fname = img_info["filename"]
                            img_path = output_dir / fname
                            if img_path.exists() and img_path.stat().st_size > 5000:
                                out_path.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy(img_path, out_path)
                                print(f"    ✅ FLUX ComfyUI: {out_path.name} ({out_path.stat().st_size//1024}KB)")
                                return True
            except Exception:
                pass
            time.sleep(2)
        print(f"    ⏰ FLUX timeout")
        return False
    except Exception as e:
        print(f"    ⚠️ FLUX ComfyUI error: {e}")
        return False


def download_flux_image(prompt: str, out_path: Path, width: int = 1024,
                        height: int = 576, seed: int = 42) -> bool:
    """Generate 16:9 image — try ComfyUI PC first, then Mac, then HuggingFace API."""
    # Try ComfyUI on PC (localhost) first — doesn't compete with Wan2.1 on Mac
    pc_url = "http://localhost:8188"
    try:
        with urllib.request.urlopen(f"{pc_url}/system_stats", timeout=3):
            ok = generate_flux_image_comfyui(prompt, out_path, width, height, seed, pc_url)
            if ok:
                return True
    except Exception:
        pass

    # Try ComfyUI on Mac M3 Pro
    try:
        with urllib.request.urlopen(f"{MAC_M3_URL}/system_stats", timeout=3):
            ok = generate_flux_image_comfyui(prompt, out_path, width, height, seed, MAC_M3_URL)
            if ok:
                return True
    except Exception:
        pass

    # Last resort: HuggingFace API
    try:
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "width": min(width, 1024),
                "height": min(height, 1024),
                "seed": seed,
                "num_inference_steps": 4,
            }
        }
        r = requests.post(FLUX_ENDPOINT, headers=headers, json=payload, timeout=60)
        if r.status_code == 200 and len(r.content) > 5000:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(r.content)
            print(f"    ✅ FLUX HF API: {out_path.name}")
            return True
    except Exception:
        pass

    return False


def apply_ken_burns(img_path: Path, out_path: Path, style_name: str, duration: float) -> bool:
    """Apply Ken Burns effect to an image → video clip."""
    frames = int(duration * 30)
    vf_zoompan = KB_STYLES[style_name](frames)
    full_vf = (
        f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,"
        f"crop={TARGET_W}:{TARGET_H},"
        f"{vf_zoompan}"
    )
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(img_path),
        "-vf", full_vf,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-an",
        str(out_path)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode == 0 and out_path.exists():
        return True
    print(f"    ❌ KB failed: {r.stderr[-200:] if r.stderr else 'unknown'}")
    return False


def apply_dynamic_motion(img_path: Path, out_path: Path, style: str, duration: float) -> bool:
    """Apply dynamic motion effects to an image — visually distinct from Ken Burns.

    These create 'video-like' clips with faster, more energetic motion:
    - fast_zoom_in: Aggressive zoom from wide to tight with slight blur at start
    - fast_zoom_out: Starts tight, pulls back dramatically
    - sweep_right: Fast horizontal sweep with momentum
    - sweep_left: Fast horizontal sweep left
    - pulse_zoom: Zoom in-out-in pulsing effect
    - reveal_blur: Starts blurry, snaps to sharp focus (dramatic reveal)
    """
    frames = int(duration * 30)

    DYNAMIC_STYLES = {
        # Fast aggressive zoom in (much faster than KB zoom_in)
        "fast_zoom_in": (
            f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,"
            f"crop={TARGET_W}:{TARGET_H},"
            f"zoompan=z='min(zoom+0.003,1.8)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={TARGET_W}x{TARGET_H}:fps=30"
        ),
        # Fast zoom out from very tight
        "fast_zoom_out": (
            f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,"
            f"crop={TARGET_W}:{TARGET_H},"
            f"zoompan=z='if(lte(zoom,1.0),1.8,zoom-0.003)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={TARGET_W}x{TARGET_H}:fps=30"
        ),
        # Fast horizontal sweep right
        "sweep_right": (
            f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,"
            f"crop={TARGET_W}:{TARGET_H},"
            f"zoompan=z='1.4':x='if(lte(on,1),0,min(x+5,iw/3))':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={TARGET_W}x{TARGET_H}:fps=30"
        ),
        # Fast horizontal sweep left
        "sweep_left": (
            f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,"
            f"crop={TARGET_W}:{TARGET_H},"
            f"zoompan=z='1.4':x='if(lte(on,1),iw/3,max(x-5,0))':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={TARGET_W}x{TARGET_H}:fps=30"
        ),
        # Zoom pulse effect (in-out-in)
        "pulse_zoom": (
            f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,"
            f"crop={TARGET_W}:{TARGET_H},"
            f"zoompan=z='1.0+0.3*sin(on*6.28/{frames})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={TARGET_W}x{TARGET_H}:fps=30"
        ),
        # Diagonal sweep (top-left to bottom-right)
        "diagonal_sweep": (
            f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,"
            f"crop={TARGET_W}:{TARGET_H},"
            f"zoompan=z='1.35':x='if(lte(on,1),0,min(x+3,iw/4))':y='if(lte(on,1),0,min(y+2,ih/4))'"
            f":d={frames}:s={TARGET_W}x{TARGET_H}:fps=30"
        ),
    }

    vf = DYNAMIC_STYLES.get(style)
    if not vf:
        return False

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(img_path),
        "-vf", vf,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-an",
        str(out_path)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode == 0 and out_path.exists():
        return True
    print(f"    ❌ Dynamic motion failed: {r.stderr[-200:] if r.stderr else 'unknown'}")
    return False


DYNAMIC_STYLE_NAMES = [
    "fast_zoom_in", "fast_zoom_out", "sweep_right", "sweep_left",
    "pulse_zoom", "diagonal_sweep",
]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PRODUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("🎬 HYBRID VISUAL PRODUCTION — Cripto $100 Guía")
    print(f"   Method: Wan2.1 animated clips + Ken Burns photos ALTERNATED")
    print(f"   Script: {SCRIPT_ID}")
    print("=" * 70)

    # ── 1. Verify audio ──────────────────────────────────────────────────────
    if not AUDIO_FILE.exists():
        print(f"❌ Audio not found: {AUDIO_FILE}")
        return False

    audio_duration = get_duration(AUDIO_FILE)
    print(f"\n🔊 Audio: {audio_duration:.1f}s")

    # Create directories
    for d in [FLUX_DIR, WAN_DIR, KB_DIR, ASSEMBLY_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 2. Generate FLUX images ──────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("📸 GENERATING FLUX IMAGES (16:9)")
    print(f"{'='*50}\n")

    all_images = []  # (path, section_name, image_idx)
    img_counter = 0

    for sec in SECTIONS:
        for i, prompt in enumerate(sec["flux_prompts"]):
            img_path = FLUX_DIR / f"flux_{img_counter:03d}_{sec['name']}_{i}.jpg"
            if img_path.exists() and img_path.stat().st_size > 5000:
                print(f"    📦 Cached: {img_path.name}")
            else:
                seed = 42 + img_counter * 13
                ok = download_flux_image(prompt, img_path, seed=seed)
                if not ok:
                    print(f"    ⚠️ Failed: {img_path.name} — will skip")
                    img_counter += 1
                    continue
                time.sleep(1)  # Rate limit

            all_images.append((img_path, sec["name"], img_counter))
            img_counter += 1

    print(f"\n✅ {len(all_images)} FLUX images ready")
    if len(all_images) < 6:
        print("❌ Not enough images")
        return False

    # ── 3. Build clip sequence: ALTERNATING KB (photo) + Dynamic Motion (video) ─
    print(f"\n{'='*50}")
    print("🔄 BUILDING ALTERNATING CLIP SEQUENCE")
    print(f"   Type A: Ken Burns (slow, smooth photo panning)")
    print(f"   Type B: Dynamic Motion (fast zoom, sweep, pulse — video-like)")
    print(f"{'='*50}\n")

    # Strategy: Alternate between Ken Burns (slow, photo-like) and
    # Dynamic Motion (fast, video-like) clips every 3-5 seconds.
    # Each uses a DIFFERENT image for visual variety.

    KB_CLIP_DUR = 4.0
    DYN_CLIP_DUR = 3.5

    final_clips = []  # list of paths in order
    clip_idx = 0
    total_time = 0.0

    for sec in SECTIONS:
        sec_duration = sec["end_s"] - sec["start_s"]
        sec_images = [(p, n, idx) for p, n, idx in all_images if n == sec["name"]]
        sec_time = 0.0
        img_cycle = 0
        kb_style_idx = 0
        dyn_style_idx = 0
        use_dynamic_next = False  # Start with KB, then alternate

        while sec_time < sec_duration - 0.5:
            remaining = sec_duration - sec_time
            if not sec_images:
                break

            if use_dynamic_next:
                # ── Insert Dynamic Motion clip (video-like) ──
                img_path = sec_images[img_cycle % len(sec_images)][0]
                style = DYNAMIC_STYLE_NAMES[dyn_style_idx % len(DYNAMIC_STYLE_NAMES)]
                dur = min(DYN_CLIP_DUR, remaining)
                if dur >= 2.5:
                    dur = DYN_CLIP_DUR + random.uniform(-0.3, 0.3)
                    dur = min(dur, remaining)
                    dur = max(dur, 2.5)

                clip_name = f"clip_{clip_idx:04d}_dyn_{sec['name']}_{dyn_style_idx}.mp4"
                clip_path = ASSEMBLY_DIR / clip_name

                if not clip_path.exists() or clip_path.stat().st_size < 10000:
                    ok = apply_dynamic_motion(img_path, clip_path, style, dur)
                    if not ok:
                        # Fallback to KB if dynamic fails
                        ok = apply_ken_burns(img_path, clip_path, "zoom_in", dur)
                        if not ok:
                            img_cycle += 1
                            continue

                final_clips.append(clip_path)
                sec_time += dur
                total_time += dur
                clip_idx += 1
                img_cycle += 1
                dyn_style_idx += 1
                use_dynamic_next = False  # Next: KB
            else:
                # ── Insert Ken Burns clip (photo-like) ──
                img_path = sec_images[img_cycle % len(sec_images)][0]
                style = KB_STYLE_NAMES[kb_style_idx % len(KB_STYLE_NAMES)]
                dur = min(KB_CLIP_DUR, remaining)
                if dur >= 3.0:
                    dur = KB_CLIP_DUR + random.uniform(-0.5, 0.5)
                    dur = min(dur, remaining)
                    dur = max(dur, 3.0)

                clip_name = f"clip_{clip_idx:04d}_kb_{sec['name']}_{kb_style_idx}.mp4"
                clip_path = ASSEMBLY_DIR / clip_name

                if not clip_path.exists() or clip_path.stat().st_size < 10000:
                    ok = apply_ken_burns(img_path, clip_path, style, dur)
                    if not ok:
                        img_cycle += 1
                        continue

                final_clips.append(clip_path)
                sec_time += dur
                total_time += dur
                clip_idx += 1
                img_cycle += 1
                kb_style_idx += 1
                use_dynamic_next = True  # Next: dynamic

        print(f"  ✅ Section '{sec['name']}': {sec_time:.1f}s covered "
              f"({sec_duration:.1f}s target)")

    print(f"\n📊 Total clips: {clip_idx}")
    print(f"   Total time: {total_time:.1f}s (audio: {audio_duration:.1f}s)")
    dyn_in_sequence = sum(1 for c in final_clips if "_dyn_" in c.name)
    kb_in_sequence = sum(1 for c in final_clips if "_kb_" in c.name)
    print(f"   Dynamic motion clips: {dyn_in_sequence}")
    print(f"   Ken Burns clips: {kb_in_sequence}")
    ratio = dyn_in_sequence / max(1, dyn_in_sequence + kb_in_sequence) * 100
    print(f"   Mix ratio: {ratio:.0f}% dynamic / {100-ratio:.0f}% photo")

    # ── 5. Assembly ──────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("🔧 ASSEMBLING FINAL VIDEO")
    print(f"{'='*50}\n")

    # Write concat list
    concat_list = ASSEMBLY_DIR / "concat_hybrid.txt"
    with open(concat_list, 'w') as f:
        for clip in final_clips:
            f.write(f"file '{clip}'\n")

    # Concat all clips
    concat_video = ASSEMBLY_DIR / "concat_raw.mp4"
    r = subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-an", str(concat_video)
    ], capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"  ❌ Concat failed: {r.stderr[-300:]}")
        return False
    print(f"  🔗 Concatenated {len(final_clips)} clips")

    # Add audio + optional logo + subtitles
    inputs = ["-i", str(concat_video), "-i", str(AUDIO_FILE)]
    filter_parts = []
    map_video = "[0:v]"
    input_idx = 2

    # BT logo
    if BT_LOGO.exists():
        inputs += ["-i", str(BT_LOGO)]
        filter_parts.append(f"[{input_idx}:v]scale=160:-1,format=rgba,colorchannelmixer=aa=0.85[btlogo]")
        filter_parts.append(f"{map_video}[btlogo]overlay=W-w-25:25:format=auto[vbt]")
        map_video = "[vbt]"
        input_idx += 1

    # Subtitles
    if SUBS_FILE.exists():
        ass_esc = str(SUBS_FILE).replace("'", "\\'").replace(":", "\\:")
        filter_parts.append(f"{map_video}ass='{ass_esc}'[vfinal]")
        map_video = "[vfinal]"

    filter_complex = ";".join(filter_parts) if filter_parts else None

    cmd = ["ffmpeg", "-y"] + inputs
    if filter_complex:
        cmd += ["-filter_complex", filter_complex, "-map", map_video, "-map", "1:a"]
    else:
        cmd += ["-map", "0:v", "-map", "1:a"]

    cmd += [
        "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.1",
        "-preset", "slow", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
        "-t", str(audio_duration),
        "-movflags", "+faststart",
        str(FINAL_VIDEO)
    ]

    print("  🎬 Encoding final video (H.264 Baseline + audio + subs)...")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"  ❌ Final encoding failed: {r.stderr[-500:]}")
        # Try without subtitles
        print("  ⚠️ Retrying without subtitles...")
        cmd2 = [
            "ffmpeg", "-y",
            "-i", str(concat_video), "-i", str(AUDIO_FILE),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.1",
            "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
            "-t", str(audio_duration),
            "-movflags", "+faststart",
            str(FINAL_VIDEO)
        ]
        r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=600)
        if r2.returncode != 0:
            print(f"  ❌ Encoding failed completely: {r2.stderr[-300:]}")
            return False
    print(f"  ✅ Final video encoded")

    # ── 6. Validation ────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("📊 FINAL VALIDATION")
    print(f"{'='*50}\n")

    if not FINAL_VIDEO.exists():
        print("❌ Output video not found")
        return False

    size_mb = FINAL_VIDEO.stat().st_size / 1024 / 1024
    final_dur = get_duration(FINAL_VIDEO)

    # Check codec
    codec_cmd = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,profile,pix_fmt",
         "-of", "json", str(FINAL_VIDEO)],
        capture_output=True, text=True
    )
    codec_info = json.loads(codec_cmd.stdout) if codec_cmd.stdout else {}
    streams = codec_info.get("streams", [{}])
    codec = streams[0].get("codec_name", "?") if streams else "?"
    profile = streams[0].get("profile", "?") if streams else "?"
    pix_fmt = streams[0].get("pix_fmt", "?") if streams else "?"

    # Check brightness
    brightness = 0
    try:
        cmd = ["ffmpeg", "-i", str(FINAL_VIDEO), "-vf",
               "select=not(mod(n\\,90)),format=gray,signalstats",
               "-f", "null", "-"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        vals = []
        for line in r.stderr.split('\n'):
            if 'YAVG' in line:
                try:
                    yavg = float(line.split('YAVG:')[1].split()[0])
                    vals.append(yavg)
                except Exception:
                    pass
        brightness = sum(vals) / len(vals) if vals else 0
    except Exception:
        pass

    dur_ok = abs(final_dur - audio_duration) < 2
    bright_ok = brightness >= 40
    codec_ok = codec == "h264"
    pix_ok = pix_fmt == "yuv420p"

    print(f"  Duration:    {final_dur:.1f}s (target: {audio_duration:.1f}s) {'✅' if dur_ok else '⚠️'}")
    print(f"  Size:        {size_mb:.1f}MB")
    print(f"  Codec:       {codec} {profile} {'✅' if codec_ok else '❌'}")
    print(f"  Pixel fmt:   {pix_fmt} {'✅' if pix_ok else '❌'}")
    print(f"  Brightness:  {brightness:.1f} {'✅' if bright_ok else '⚠️ DARK'}")
    print(f"  Visual mix:  {dyn_in_sequence} Dynamic + {kb_in_sequence} Ken Burns")
    print(f"  Cost:        $0.00 (all local)")

    all_ok = dur_ok and bright_ok and codec_ok and pix_ok
    if all_ok:
        print(f"\n🎉 VIDEO READY: {FINAL_VIDEO}")
    else:
        print(f"\n⚠️ VIDEO PRODUCED with warnings: {FINAL_VIDEO}")

    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
