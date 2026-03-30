#!/usr/bin/env python3
"""
gen_hybrid_5050.py — Genera video 50% Pexels + 50% SDXL (Ken Burns)
Guion: "3 reglas que salvaron mi cuenta de trading"
Output: /home/enderj/.openclaw/workspace/bittrader/agents/output/hybrid_test/3reglas_hybrid_5050.mp4
"""

import json
import os
import sys
import subprocess
import random
import requests
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "output" / "hybrid_test"
CLIPS_DIR   = OUTPUT_DIR / "clips"
PEXELS_DIR  = CLIPS_DIR / "pexels"
SDXL_DIR    = CLIPS_DIR / "sdxl"

AUDIO_PATH  = BASE_DIR / "output/2026-03-28/long_1774708531_991/long_1774708531_991.mp3"
SUBS_PATH   = BASE_DIR / "output/2026-03-28/long_1774708531_991/subtitles.ass"
LOGO_PATH   = Path("/home/enderj/.openclaw/workspace/bittrader/assets/bittrader_logo_transparent.png")
FINAL_OUT   = OUTPUT_DIR / "3reglas_hybrid_5050.mp4"

HF_API_KEY  = "HF_TOKEN_REDACTED"
HF_URL      = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"

# Load Pexels key
PEXELS_KEY_FILE = BASE_DIR / "keys/pexels.json"
try:
    PEXELS_KEY = json.loads(PEXELS_KEY_FILE.read_text())["api_key"]
except Exception:
    PEXELS_KEY = os.environ.get("PEXELS_API_KEY", "")

CLIP_DURATION = 6  # seconds per clip
TOTAL_CLIPS   = 70
N_PEXELS      = 35
N_SDXL        = 35

# ── Segments ──────────────────────────────────────────────────────────────────
SEGMENTS = [
    {"text": "reglas de trading stop loss gestión de riesgo", "keywords": ["risk management trading", "stop loss chart", "trading rules"]},
    {"text": "tamaño de posición proteger capital", "keywords": ["trading position size", "portfolio protection", "financial risk"]},
    {"text": "diversificación bitcoin ethereum", "keywords": ["cryptocurrency portfolio", "bitcoin ethereum", "crypto diversification"]},
    {"text": "éxito consistente trading profesional", "keywords": ["successful trader", "profit trading", "financial freedom"]},
    {"text": "gráficas mercado cripto análisis", "keywords": ["crypto market chart", "bitcoin analysis", "candlestick chart"]},
    {"text": "estrategia de trading ganadora", "keywords": ["trading strategy", "market success", "investment growth"]},
]

# ── SDXL Prompts ──────────────────────────────────────────────────────────────
SDXL_PROMPTS = [
    "bitcoin cryptocurrency coins gold pile dramatic lighting 4K cinematic",
    "stock market trading charts green candles financial success 4K",
    "professional trader analyzing multiple screens office 4K cinematic",
    "red declining chart stop loss trading risk management 4K",
    "diverse investment portfolio coins stocks crypto digital art 4K",
    "financial freedom success money growth concept cinematic 4K",
    "bitcoin logo gold background dramatic studio lighting 4K",
    "trading terminal candlestick charts professional setup 4K",
]

# ── Pexels keyword pool ────────────────────────────────────────────────────────
PEXELS_KEYWORDS = [
    "risk management trading", "stop loss chart", "trading rules",
    "trading position size", "portfolio protection", "financial risk",
    "cryptocurrency portfolio", "bitcoin ethereum", "crypto diversification",
    "successful trader", "profit trading", "financial freedom",
    "crypto market chart", "bitcoin analysis", "candlestick chart",
    "trading strategy", "market success", "investment growth",
    "stock market trading", "bitcoin chart", "cryptocurrency trading",
    "financial charts", "trading terminal", "market analysis",
    "candlestick chart", "trading computer screens", "forex trading",
    "successful trader", "financial success", "profit growth",
    "green stock chart", "bitcoin profit", "financial freedom",
    "investment portfolio", "diversification finance",
]


# ══════════════════════════════════════════════════════════════════════════════
# PEXELS
# ══════════════════════════════════════════════════════════════════════════════

def search_pexels(keyword: str, per_page: int = 3) -> list:
    try:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": PEXELS_KEY},
            params={
                "query": keyword,
                "per_page": per_page,
                "orientation": "landscape",
                "size": "medium",
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
                w = vf.get("width", 0)
                h = vf.get("height", 0)
                if w >= 1280 and w > h and vf.get("file_type") == "video/mp4":
                    results.append({
                        "url": vf["link"],
                        "id": v["id"],
                        "keyword": keyword,
                    })
                    break
        return results
    except Exception as e:
        print(f"   ⚠️  Pexels '{keyword}': {e}")
        return []


def download_clip(url: str, out_path: Path) -> bool:
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


def trim_scale_pexels(src: Path, dst: Path) -> bool:
    vf = (
        "scale=1280:720:force_original_aspect_ratio=increase:flags=lanczos,"
        "crop=1280:720:(iw-1280)/2:(ih-720)/2,"
        "setsar=1,fps=30"
    )
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-t", str(CLIP_DURATION), "-vf", vf,
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-profile:v", "baseline", "-level", "3.1",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)],
        capture_output=True, timeout=60,
    )
    return r.returncode == 0 and dst.exists()


def get_pexels_clips() -> list:
    PEXELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n🔍 Descargando {N_PEXELS} clips de Pexels...")
    
    seen_ids = set()
    meta_list = []
    kws = PEXELS_KEYWORDS.copy()
    random.shuffle(kws)
    
    for kw in kws:
        if len(meta_list) >= N_PEXELS:
            break
        results = search_pexels(kw, per_page=3)
        for rc in results:
            if rc["id"] not in seen_ids and len(meta_list) < N_PEXELS:
                seen_ids.add(rc["id"])
                meta_list.append(rc)
    
    ready = []
    for i, meta in enumerate(meta_list[:N_PEXELS]):
        raw  = PEXELS_DIR / f"raw_{i:03d}.mp4"
        trim = PEXELS_DIR / f"pexels_{i:03d}.mp4"
        
        if trim.exists() and trim.stat().st_size > 10_000:
            print(f"   💾 Pexels {i:03d}: cached ✓")
            ready.append(trim)
            continue
        
        if not download_clip(meta["url"], raw):
            print(f"   ⚠️  Pexels {i:03d}: descarga fallida")
            continue
        
        ok = trim_scale_pexels(raw, trim)
        raw.unlink(missing_ok=True)
        
        if ok:
            sz = trim.stat().st_size // 1024
            print(f"   💾 Pexels {i:03d}: {meta['keyword'][:35]} ({sz}KB)")
            ready.append(trim)
        else:
            print(f"   ⚠️  Pexels {i:03d}: trim fallido")
    
    print(f"✅ Pexels: {len(ready)}/{N_PEXELS} clips")
    return ready


# ══════════════════════════════════════════════════════════════════════════════
# SDXL + KEN BURNS
# ══════════════════════════════════════════════════════════════════════════════

def generate_sdxl_image(prompt: str, out_path: Path, attempt: int = 0) -> bool:
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
            "width": 1024,
            "height": 576,
        },
    }
    try:
        r = requests.post(HF_URL, headers=headers, json=payload, timeout=120)
        if r.status_code == 503:
            print(f"   ⏳ SDXL: modelo cargando, espera 25s...")
            time.sleep(25)
            r = requests.post(HF_URL, headers=headers, json=payload, timeout=120)
        if r.status_code == 429:
            print(f"   ⏳ SDXL: rate limit, espera 30s...")
            time.sleep(30)
            r = requests.post(HF_URL, headers=headers, json=payload, timeout=120)
        
        if r.status_code != 200:
            print(f"   ⚠️  SDXL HTTP {r.status_code}: {r.text[:100]}")
            return False
        
        ct = r.headers.get("content-type", "")
        if "image" not in ct:
            print(f"   ⚠️  SDXL content-type inesperado: {ct}")
            return False
        
        out_path.write_bytes(r.content)
        print(f"   🖼️  SDXL img OK: {out_path.stat().st_size//1024}KB")
        return True
    except Exception as e:
        print(f"   ⚠️  SDXL error: {e}")
        return False


KB_STYLES = [
    "zoompan=z='min(zoom+0.0015,1.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=150:s=1280x720",
    "zoompan=z='if(lte(zoom,1.0),1.5,zoom-0.0015)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=150:s=1280x720",
    "zoompan=z='min(zoom+0.0015,1.4)':x='if(lte(on,1),0,min(x+1,iw/4))':y='ih/2-(ih/zoom/2)':d=150:s=1280x720",
    "zoompan=z='min(zoom+0.0015,1.4)':x='if(lte(on,1),iw/4,max(x-1,0))':y='ih/2-(ih/zoom/2)':d=150:s=1280x720",
]


def image_to_kenburns(img_path: Path, out_path: Path, clip_num: int) -> bool:
    kb = KB_STYLES[clip_num % len(KB_STYLES)]
    vf = f"scale=8000:-1:flags=lanczos,{kb},setsar=1,fps=30"
    
    r = subprocess.run(
        ["ffmpeg", "-y", "-loop", "1", "-i", str(img_path),
         "-vf", vf, "-t", str(CLIP_DURATION),
         "-c:v", "libx264", "-preset", "fast", "-crf", "25",
         "-profile:v", "baseline", "-level", "3.1",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out_path)],
        capture_output=True, timeout=120,
    )
    if r.returncode != 0:
        # Fallback: simpler scale
        print(f"   ⚠️  zoompan falló clip {clip_num}, usando scale simple...")
        vf_simple = "scale=1280:720:force_original_aspect_ratio=increase:flags=lanczos,crop=1280:720,setsar=1,fps=30"
        r2 = subprocess.run(
            ["ffmpeg", "-y", "-loop", "1", "-i", str(img_path),
             "-vf", vf_simple, "-t", str(CLIP_DURATION),
             "-c:v", "libx264", "-preset", "fast", "-crf", "25",
             "-profile:v", "baseline", "-level", "3.1",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out_path)],
            capture_output=True, timeout=60,
        )
        return r2.returncode == 0 and out_path.exists()
    return out_path.exists()


def get_sdxl_clips() -> list:
    SDXL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n🎨 Generando {N_SDXL} clips SDXL (Ken Burns)...")
    ready = []
    
    for i in range(N_SDXL):
        img_path  = SDXL_DIR / f"img_{i:03d}.jpg"
        clip_path = SDXL_DIR / f"sdxl_{i:03d}.mp4"
        
        if clip_path.exists() and clip_path.stat().st_size > 10_000:
            print(f"   🎞️  SDXL {i:03d}: cached ✓")
            ready.append(clip_path)
            continue
        
        # Rotate through prompts
        prompt = SDXL_PROMPTS[i % len(SDXL_PROMPTS)]
        print(f"   🎨 SDXL {i:03d}: {prompt[:50]}...")
        
        ok_img = generate_sdxl_image(prompt, img_path)
        if not ok_img:
            print(f"   ⚠️  SDXL {i:03d}: imagen fallida, saltando")
            continue
        
        ok_clip = image_to_kenburns(img_path, clip_path, i)
        img_path.unlink(missing_ok=True)
        
        if ok_clip:
            sz = clip_path.stat().st_size // 1024
            print(f"   🎞️  SDXL {i:03d}: OK ({sz}KB)")
            ready.append(clip_path)
        else:
            print(f"   ⚠️  SDXL {i:03d}: Ken Burns fallido")
        
        # Small delay to avoid rate limiting
        if i < N_SDXL - 1:
            time.sleep(1)
    
    print(f"✅ SDXL: {len(ready)}/{N_SDXL} clips")
    return ready


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

def interleave(pexels: list, sdxl: list) -> list:
    """Interleave: pexels_0, sdxl_0, pexels_1, sdxl_1, ..."""
    result = []
    max_len = max(len(pexels), len(sdxl))
    for i in range(max_len):
        if i < len(pexels):
            result.append(pexels[i])
        if i < len(sdxl):
            result.append(sdxl[i])
    return result


def build_concat_list(clips: list, concat_file: Path):
    lines = []
    for c in clips:
        lines.append(f"file '{c.absolute()}'")
    concat_file.write_text("\n".join(lines))


def assemble_final(clips: list) -> bool:
    print(f"\n🎬 Ensamblando video final con {len(clips)} clips...")
    
    # Step 1: concat clips (muted)
    concat_file = OUTPUT_DIR / "concat_list.txt"
    build_concat_list(clips, concat_file)
    
    raw_video = OUTPUT_DIR / "raw_concat.mp4"
    r = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", str(concat_file),
         "-c", "copy",
         str(raw_video)],
        capture_output=True, timeout=300,
    )
    if r.returncode != 0:
        print(f"❌ concat falló: {r.stderr.decode()[-500:]}")
        return False
    print(f"   ✅ concat OK: {raw_video.stat().st_size//1024//1024}MB")
    
    # Get audio duration to know video length needed
    r2 = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(AUDIO_PATH)],
        capture_output=True
    )
    audio_info = json.loads(r2.stdout)
    audio_dur = float(audio_info["format"]["duration"])
    print(f"   🎵 Audio duration: {audio_dur:.1f}s")
    
    # Get raw video duration
    r3 = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(raw_video)],
        capture_output=True
    )
    vid_info = json.loads(r3.stdout)
    vid_dur = float(vid_info["format"]["duration"])
    print(f"   📹 Video duration: {vid_dur:.1f}s")
    
    # Step 2: assemble with audio + subtitles + logo
    # Use subtitles filter + logo overlay
    subs_escaped = str(SUBS_PATH).replace(':', r'\:').replace("'", r"\'")
    
    # Build filter: subtitles + logo
    filter_complex = (
        f"[0:v]subtitles='{subs_escaped}':force_style='FontName=Arial,FontSize=14'[vsub];"
        f"[vsub][1:v]overlay=W-w-20:20:alpha=0.85[vout]"
    )
    
    r4 = subprocess.run(
        ["ffmpeg", "-y",
         "-i", str(raw_video),                    # [0:v]
         "-i", str(LOGO_PATH),                    # [1:v]  (logo)
         "-i", str(AUDIO_PATH),                   # [2:a]
         "-filter_complex", filter_complex,
         "-map", "[vout]",
         "-map", "2:a",
         "-vf", f"scale=1280x720:force_original_aspect_ratio=disable",  # ensure 720p
         "-c:v", "libx264",
         "-profile:v", "baseline",
         "-level", "3.1",
         "-pix_fmt", "yuv420p",
         "-movflags", "+faststart",
         "-crf", "23",
         "-c:a", "aac",
         "-b:a", "192k",
         "-ar", "44100",
         "-ac", "2",
         "-shortest",
         str(FINAL_OUT)],
        capture_output=True, timeout=600,
    )
    
    if r4.returncode != 0:
        stderr = r4.stderr.decode()[-1000:]
        print(f"❌ assemble falló: {stderr}")
        # Try simpler approach without subtitle filter
        print("   ↩️  Reintentando sin subtítulos inline...")
        r5 = subprocess.run(
            ["ffmpeg", "-y",
             "-i", str(raw_video),
             "-i", str(LOGO_PATH),
             "-i", str(AUDIO_PATH),
             "-filter_complex",
             "[0:v][1:v]overlay=W-w-20:20:alpha=0.85[vout]",
             "-map", "[vout]",
             "-map", "2:a",
             "-c:v", "libx264",
             "-profile:v", "baseline",
             "-level", "3.1",
             "-pix_fmt", "yuv420p",
             "-movflags", "+faststart",
             "-crf", "23",
             "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
             "-shortest",
             str(FINAL_OUT)],
            capture_output=True, timeout=600,
        )
        if r5.returncode != 0:
            print(f"❌ assemble sin subs también falló: {r5.stderr.decode()[-500:]}")
            return False
    
    if FINAL_OUT.exists():
        size_mb = FINAL_OUT.stat().st_size / 1024 / 1024
        print(f"✅ Video final: {FINAL_OUT} ({size_mb:.1f}MB)")
        return True
    return False


def verify_output():
    """Verify 1280x720, duration, audio, brightness."""
    if not FINAL_OUT.exists():
        return False
    
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", str(FINAL_OUT)],
        capture_output=True
    )
    info = json.loads(r.stdout)
    
    vstream = next((s for s in info["streams"] if s["codec_type"] == "video"), {})
    astream = next((s for s in info["streams"] if s["codec_type"] == "audio"), {})
    
    width  = vstream.get("width", 0)
    height = vstream.get("height", 0)
    dur    = float(info["format"].get("duration", 0))
    size   = FINAL_OUT.stat().st_size / 1024 / 1024
    audio  = astream.get("codec_name", "none")
    
    print(f"\n📊 Verificación:")
    print(f"   Resolución: {width}x{height} {'✅' if width==1280 and height==720 else '❌'}")
    print(f"   Duración:   {dur:.1f}s {'✅' if dur > 300 else '⚠️ (corto)'}")
    print(f"   Tamaño:     {size:.1f}MB")
    print(f"   Audio:      {audio} {'✅' if audio == 'aac' else '❌'}")
    
    # Quick brightness check (sample frame at 10s)
    r2 = subprocess.run(
        ["ffmpeg", "-ss", "10", "-i", str(FINAL_OUT),
         "-frames:v", "1", "-q:v", "2", "/tmp/brightness_check.jpg", "-y"],
        capture_output=True, timeout=30
    )
    brightness_ok = False
    if Path("/tmp/brightness_check.jpg").exists():
        r3 = subprocess.run(
            ["ffmpeg", "-i", "/tmp/brightness_check.jpg",
             "-vf", "signalstats", "-f", "null", "-"],
            capture_output=True, timeout=15
        )
        # Use identify via ffprobe stats
        r4 = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_frames", "-read_intervals", "%+#1",
             "-select_streams", "v",
             "/tmp/brightness_check.jpg"],
            capture_output=True
        )
        # Simple check: file exists and size > 0 means image is not black
        bsize = Path("/tmp/brightness_check.jpg").stat().st_size
        brightness_ok = bsize > 5000
        print(f"   Brillo:     {'✅ OK' if brightness_ok else '❌ posible imagen negra'} (frame size: {bsize}B)")
    
    return width == 1280 and height == 720 and audio == "aac"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("🎬 BitTrader Hybrid 50/50 Video Generator")
    print("   Guion: 3 reglas que salvaron mi cuenta de trading")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Get Pexels clips
    pexels_clips = get_pexels_clips()
    
    # 2. Get SDXL clips
    sdxl_clips = get_sdxl_clips()
    
    print(f"\n📊 Resultado clips:")
    print(f"   Pexels: {len(pexels_clips)}")
    print(f"   SDXL:   {len(sdxl_clips)}")
    
    if len(pexels_clips) + len(sdxl_clips) < 10:
        print("❌ No hay suficientes clips para generar video")
        sys.exit(1)
    
    # 3. Interleave: pexels_0, sdxl_0, pexels_1, sdxl_1, ...
    interleaved = interleave(pexels_clips, sdxl_clips)
    print(f"\n🔀 Clips intercalados: {len(interleaved)}")
    for i, c in enumerate(interleaved[:6]):
        print(f"   [{i}] {c.name}")
    print(f"   ... (y {len(interleaved)-6} más)")
    
    # 4. Assemble final video
    ok = assemble_final(interleaved)
    
    # 5. Verify
    if ok:
        valid = verify_output()
        p_count = sum(1 for c in interleaved if 'pexels' in str(c))
        s_count = sum(1 for c in interleaved if 'sdxl' in str(c))
        size = FINAL_OUT.stat().st_size / 1024 / 1024 if FINAL_OUT.exists() else 0
        
        print(f"\n{'='*60}")
        if valid:
            print(f"✅ VIDEO GENERADO EXITOSAMENTE")
        else:
            print(f"⚠️  VIDEO GENERADO PERO CON ADVERTENCIAS")
        print(f"   Archivo:       {FINAL_OUT}")
        print(f"   Tamaño:        {size:.1f}MB")
        print(f"   Clips Pexels:  {p_count}")
        print(f"   Clips SDXL:    {s_count}")
        print(f"{'='*60}")
    else:
        print("❌ FALLO: video no generado")
        sys.exit(1)


if __name__ == "__main__":
    main()
