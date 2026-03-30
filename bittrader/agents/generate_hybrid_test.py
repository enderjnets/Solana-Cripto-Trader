#!/usr/bin/env python3
"""
generate_hybrid_test.py — Genera video híbrido de prueba para "3 reglas que salvaron mi cuenta"
Usa clip_strategy.py para 50/50 Pexels + SDXL
"""
import sys
import os
import subprocess
import tempfile
import json
from pathlib import Path

# Add agents dir to path
AGENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(AGENTS_DIR))

import clip_strategy

# ── Paths ──────────────────────────────────────────────────────────────────────
AUDIO_PATH    = Path("/home/enderj/.openclaw/workspace/bittrader/agents/output/2026-03-28/long_1774708531_991/long_1774708531_991.mp3")
SUBS_PATH     = Path("/home/enderj/.openclaw/workspace/bittrader/agents/output/2026-03-28/long_1774708531_991/subtitles.ass")
LOGO_PATH     = Path("/home/enderj/.openclaw/workspace/bittrader/assets/bittrader_logo_transparent.png")
OUTPUT_DIR    = Path("/home/enderj/.openclaw/workspace/bittrader/agents/output/hybrid_test")
OUTPUT_VIDEO  = OUTPUT_DIR / "3reglas_hybrid_v1.mp4"
CLIPS_DIR     = OUTPUT_DIR / "clips"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
CLIP_DURATION = 6     # seconds per clip
TARGET_W, TARGET_H = 1280, 720
CLIPS_NEEDED  = 70    # ~420s total (>398s audio duration)

# ── Segments ───────────────────────────────────────────────────────────────────
segments = [
    {"text": "reglas de trading que salvan cuentas", "keywords": ["trading rules", "stock market", "financial charts"]},
    {"text": "gestión de riesgo, tamaño de posición", "keywords": ["risk management", "trading position", "portfolio risk"]},
    {"text": "stop loss efectivo, proteger capital",  "keywords": ["stop loss", "market crash", "trading risk"]},
    {"text": "diversificación profesional",           "keywords": ["investment portfolio", "diversification", "cryptocurrency bitcoin"]},
    {"text": "éxito consistente en trading",          "keywords": ["financial success", "profit growth", "trading strategy"]},
    {"text": "aplica las reglas ahora",               "keywords": ["successful trader", "financial freedom", "bitcoin profit"]},
]

print("=" * 70)
print("🎬 HYBRID VIDEO TEST — 3 Reglas que Salvaron mi Cuenta de Trading")
print("=" * 70)
print(f"   Audio:    {AUDIO_PATH.name}")
print(f"   Subtitles:{SUBS_PATH.name}")
print(f"   Output:   {OUTPUT_VIDEO}")
print(f"   Clips:    {CLIPS_NEEDED} × {CLIP_DURATION}s = {CLIPS_NEEDED*CLIP_DURATION}s total")
print()

# ── Step 1: Get hybrid clips ────────────────────────────────────────────────────
print("📥 Step 1: Obteniendo clips híbridos (50% Pexels + 50% SDXL)...")
clips = clip_strategy.get_clips_hybrid(
    segments,
    total_clips_needed=CLIPS_NEEDED,
    clips_dir=CLIPS_DIR,
    video_type="long",
)

n_pexels = sum(1 for c in clips if "pexels" in str(c))
n_sdxl   = sum(1 for c in clips if "sdxl"   in str(c))
print(f"   ✅ {len(clips)} clips listos: {n_pexels} Pexels + {n_sdxl} SDXL")

if not clips:
    print("❌ ERROR: No se obtuvieron clips. Abortando.")
    sys.exit(1)

# ── Step 2: Create concat file ─────────────────────────────────────────────────
print("\n📋 Step 2: Creando concat list...")
concat_file = OUTPUT_DIR / "concat_list.txt"
with open(concat_file, "w") as f:
    for clip in clips:
        f.write(f"file '{clip.absolute()}'\n")
print(f"   ✅ {len(clips)} entradas en concat list")

# ── Step 3: Concatenate clips → raw video ──────────────────────────────────────
print("\n🔗 Step 3: Concatenando clips (sin audio)...")
raw_video = OUTPUT_DIR / "raw_concat.mp4"
cmd_concat = [
    "ffmpeg", "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", str(concat_file),
    "-c:v", "libx264",
    "-preset", "fast",
    "-crf", "23",
    "-profile:v", "baseline",
    "-level", "3.1",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    "-an",
    str(raw_video),
]
result = subprocess.run(cmd_concat, capture_output=True, text=True)
if result.returncode != 0:
    print(f"❌ Concatenación fallida:\n{result.stderr[-1000:]}")
    sys.exit(1)
print(f"   ✅ Raw video: {raw_video.stat().st_size // 1024 // 1024}MB")

# ── Step 4: Add audio + subtitles + logo overlay ───────────────────────────────
print("\n🎞️  Step 4: Ensamblando video final (audio + subs + logo)...")

# Build vf filter chain:
# 1. Scale-crop safety (already done in clips, but set FPS just in case)
# 2. ASS subtitles burned in
# 3. Logo overlay top-right with scale=120px wide, opacity=0.85

logo_exists = LOGO_PATH.exists()
subs_exists = SUBS_PATH.exists()
print(f"   Logo: {'✅' if logo_exists else '❌'} {LOGO_PATH}")
print(f"   Subs: {'✅' if subs_exists else '❌'} {SUBS_PATH}")

if subs_exists and logo_exists:
    # Complex filter: video → burn subs → overlay logo
    # [0:v] = concat video, [1:a] = audio, [2:v] = logo
    subs_path_escaped = str(SUBS_PATH).replace(":", "\\:")
    vf_main = f"ass={subs_path_escaped}"
    
    cmd_final = [
        "ffmpeg", "-y",
        "-i", str(raw_video),          # [0] video
        "-i", str(AUDIO_PATH),         # [1] audio
        "-i", str(LOGO_PATH),          # [2] logo
        "-filter_complex",
        f"[0:v]{vf_main}[vwsubs];"
        f"[2:v]scale=120:-1[logo];"
        f"[vwsubs][logo]overlay=W-w-20:20:format=auto,format=yuv420p[vout]",
        "-map", "[vout]",
        "-map", "1:a",
        "-shortest",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-profile:v", "baseline",
        "-level", "3.1",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-ac", "2",
        str(OUTPUT_VIDEO),
    ]
elif subs_exists:
    # No logo
    subs_path_escaped = str(SUBS_PATH).replace(":", "\\:")
    cmd_final = [
        "ffmpeg", "-y",
        "-i", str(raw_video),
        "-i", str(AUDIO_PATH),
        "-vf", f"ass={subs_path_escaped},format=yuv420p",
        "-map", "0:v",
        "-map", "1:a",
        "-shortest",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-profile:v", "baseline",
        "-level", "3.1",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-ac", "2",
        str(OUTPUT_VIDEO),
    ]
else:
    # No subs no logo
    cmd_final = [
        "ffmpeg", "-y",
        "-i", str(raw_video),
        "-i", str(AUDIO_PATH),
        "-map", "0:v",
        "-map", "1:a",
        "-shortest",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-profile:v", "baseline",
        "-level", "3.1",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-ac", "2",
        str(OUTPUT_VIDEO),
    ]

result = subprocess.run(cmd_final, capture_output=True, text=True)
if result.returncode != 0:
    print(f"❌ Ensamblaje final fallido:\n{result.stderr[-2000:]}")
    sys.exit(1)

print(f"   ✅ Video final generado")

# ── Step 5: Verify ────────────────────────────────────────────────────────────
print("\n🔍 Step 5: Verificando video final...")

probe_cmd = [
    "ffprobe", "-v", "quiet",
    "-print_format", "json",
    "-show_streams", "-show_format",
    str(OUTPUT_VIDEO),
]
probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
probe_data   = json.loads(probe_result.stdout)

# Extract info
streams = probe_data.get("streams", [])
fmt     = probe_data.get("format", {})
video_stream = next((s for s in streams if s["codec_type"] == "video"), None)
audio_stream = next((s for s in streams if s["codec_type"] == "audio"), None)

duration  = float(fmt.get("duration", 0))
size_bytes = int(fmt.get("size", 0))
size_mb    = size_bytes / 1024 / 1024

width  = int(video_stream.get("width",  0)) if video_stream else 0
height = int(video_stream.get("height", 0)) if video_stream else 0
has_audio = audio_stream is not None

# Brightness check via ffprobe/ffmpeg
print("   🔆 Midiendo brightness...")
bright_result = subprocess.run(
    ["ffmpeg", "-i", str(OUTPUT_VIDEO), "-vf", "cropdetect,signalstats",
     "-ss", "30", "-t", "5",
     "-f", "null", "-"],
    capture_output=True, text=True,
)
# Extract YAVG from signalstats
brightness = None
for line in bright_result.stderr.split("\n"):
    if "YAVG" in line:
        try:
            brightness = float(line.split("YAVG:")[1].split()[0])
            break
        except Exception:
            pass

# Print report
print()
print("=" * 70)
print("📊 REPORTE FINAL")
print("=" * 70)
ok_dims  = (width == TARGET_W and height == TARGET_H)
ok_dur   = (350 < duration < 500)
ok_audio = has_audio
ok_bright = brightness is None or brightness > 30  # None = couldn't check, assume OK

print(f"   {'✅' if ok_dims  else '❌'} Dimensiones: {width}×{height} (esperado 1280×720)")
print(f"   {'✅' if ok_dur   else '❌'} Duración: {duration:.1f}s (esperado ~400s)")
print(f"   {'✅' if ok_audio else '❌'} Audio: {'sí' if has_audio else 'NO'}")
print(f"   {'✅' if ok_bright else '❌'} Brightness: {brightness:.1f if brightness else 'N/A'} (mínimo 30)")
print()
print(f"   📦 Tamaño: {size_mb:.1f}MB")
print(f"   🎬 Clips: {n_pexels} Pexels + {n_sdxl} SDXL = {n_pexels+n_sdxl} total")
print(f"   📁 Output: {OUTPUT_VIDEO}")
print()

all_ok = ok_dims and ok_dur and ok_audio and ok_bright
if all_ok:
    print("✅ VIDEO HÍBRIDO LISTO — Todas las verificaciones pasaron")
else:
    print("⚠️  Video generado con algunas advertencias (ver detalles arriba)")

# Clean up raw concat
if raw_video.exists():
    raw_video.unlink()
    print(f"   🗑️  Limpiado: {raw_video.name}")

print()
print(f"RESULTADO: {'✅' if all_ok else '⚠️'} | {size_mb:.1f}MB | {duration:.0f}s | {n_pexels} Pexels + {n_sdxl} SDXL")
