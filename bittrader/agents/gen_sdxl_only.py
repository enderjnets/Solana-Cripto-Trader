#!/usr/bin/env python3
"""
gen_sdxl_only.py — Genera video 100% SDXL (Ken Burns), sincronizado con narrador.
Sin Pexels. La duración de cada clip se ajusta al audio del narrador.
Output: /home/enderj/.openclaw/workspace/bittrader/agents/output/hybrid_test/3reglas_sdxl_only.mp4
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
OUTPUT_DIR  = BASE_DIR / "output" / "test_sdxl"
CLIPS_DIR   = OUTPUT_DIR / "clips" / "sdxl"

AUDIO_PATH  = Path("/tmp/test_narrator.mp3")
SUBS_PATH   = Path("/tmp/test_subs.ass")
LOGO_PATH   = Path("/home/enderj/.openclaw/workspace/bittrader/assets/bittrader_logo_transparent.png")
FINAL_OUT   = OUTPUT_DIR / "test_sdxl_video.mp4"

MINIMAX_API_KEY = "sk-cp-8tBIgoE2Vs8QE0AIoMjq4MTh8kiHtem3KWlOnNlAJZgKwAlYh_nt6oCq382Y0cmBi2buvch3nJJbMg7uqr_hIV6Z0ZqY3Q_qZ6AStHCUpKKT_IT-e0vEl4A"
HF_API_KEY  = "HF_TOKEN_REDACTED"  # Backup only

N_CLIPS_DEFAULT = 20      # clips por defecto (si no se puede calcular)
MIN_CLIP_DUR    = 4.0     # segundos mínimo por clip
MAX_CLIP_DUR    = 10.0    # segundos máximo por clip

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
    "ethereum crypto token futuristic holographic 4K cinematic",
    "bull market green arrow rising profits celebration 4K",
    "bear market red chart falling economy dramatic 4K cinematic",
    "golden bitcoin on circuit board technology macro 4K",
    "modern trading desk multiple monitors night city view 4K",
    "cryptocurrency wallet security digital lock concept 4K",
    "financial growth bar chart ascending stairs success 4K cinematic",
    "world map connected crypto nodes global network 4K",
]

KB_STYLES = [
    "zoompan=z='min(zoom+0.0015,1.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1280x720",
    "zoompan=z='if(lte(zoom,1.0),1.5,zoom-0.0015)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1280x720",
    "zoompan=z='min(zoom+0.0015,1.4)':x='if(lte(on,1),0,min(x+1,iw/4))':y='ih/2-(ih/zoom/2)':d={frames}:s=1280x720",
    "zoompan=z='min(zoom+0.0015,1.4)':x='if(lte(on,1),iw/4,max(x-1,0))':y='ih/2-(ih/zoom/2)':d={frames}:s=1280x720",
]


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO DURATION & CLIP SYNC
# ══════════════════════════════════════════════════════════════════════════════

def get_audio_duration(audio_path: Path) -> float:
    """Use ffprobe to get audio duration in seconds."""
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(audio_path)],
        capture_output=True,
    )
    if r.returncode != 0:
        print(f"❌ ffprobe falló para {audio_path}")
        sys.exit(1)
    info = json.loads(r.stdout)
    dur = float(info["format"]["duration"])
    return dur


def calculate_clip_durations(audio_dur: float, n_clips: int,
                             min_dur: float = MIN_CLIP_DUR,
                             max_dur: float = MAX_CLIP_DUR) -> list[float]:
    """
    Divide audio duration into n_clips segments.
    Clamp each between min_dur and max_dur.
    If clamped to min_dur, reduce n_clips so total fits.
    Last clip adjusted so sum == audio_dur exactly.
    """
    raw_dur = audio_dur / n_clips

    # If each clip would be shorter than min_dur, reduce clip count
    if raw_dur < min_dur:
        n_clips = max(1, int(audio_dur / min_dur))
        raw_dur = audio_dur / n_clips

    # If each clip would be longer than max_dur, increase clip count
    if raw_dur > max_dur:
        n_clips = max(1, int(audio_dur / max_dur) + 1)
        raw_dur = audio_dur / n_clips

    # Clamp
    clamped = max(min_dur, min(max_dur, raw_dur))
    durations = [clamped] * n_clips

    # Adjust last clip so total == audio_dur
    current_sum = sum(durations[:-1])
    durations[-1] = audio_dur - current_sum

    # Safety: if last clip is negative or too small, redistribute
    if durations[-1] < 1.0:
        n_clips = max(1, n_clips - 1)
        return calculate_clip_durations(audio_dur, n_clips, min_dur, max_dur)

    return durations


# ══════════════════════════════════════════════════════════════════════════════
# SDXL + KEN BURNS
# ══════════════════════════════════════════════════════════════════════════════

def generate_sdxl_image(prompt: str, out_path: Path, attempt: int = 0) -> bool:
    """Genera imagen con MiniMax image-01 (o HF SDXL como fallback)."""
    # Intentar MiniMax image-01 primero
    try:
        headers = {
            "Authorization": f"Bearer {MINIMAX_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "image-01",
            "prompt": prompt,
            "aspect_ratio": "16:9",
            "resolution": "1024x1024",
            "prompt_optimizer": True,
        }
        r = requests.post(
            "https://api.minimax.io/v1/image_generation",
            headers=headers, json=payload, timeout=120
        )
        data = r.json()
        if r.status_code == 200 and data.get("base_resp", {}).get("status_code") == 0:
            image_urls = data.get("data", {}).get("image_urls", [])
            if image_urls:
                img_resp = requests.get(image_urls[0], timeout=120)
                img_resp.raise_for_status()
                # Guardar como PNG
                out_path.write_bytes(img_resp.content)
                print(f"   🖼️  MiniMax img OK: {out_path.stat().st_size//1024}KB")
                return True

        # Si falla MiniMax, intentar HF SDXL
        print(f"   ⚠️  MiniMax falló ({r.status_code}), intentando SDXL...")
        raise Exception("MiniMax unavailable")

    except Exception as e:
        # Fallback: HuggingFace SDXL
        try:
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
            r = requests.post(
                "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0",
                headers=headers, json=payload, timeout=120
            )
            if r.status_code == 200 and "image" in r.headers.get("content-type", ""):
                out_path.write_bytes(r.content)
                print(f"   🖼️  SDXL fallback OK: {out_path.stat().st_size//1024}KB")
                return True
            print(f"   ⚠️  SDXL fallback HTTP {r.status_code}: {r.text[:100]}")
            return False
        except Exception as e2:
            print(f"   ⚠️  SDXL fallback error: {e2}")
            return False


def image_to_kenburns(img_path: Path, out_path: Path, clip_num: int,
                      duration: float) -> bool:
    """Convert SDXL image to Ken Burns video clip with given duration."""
    frames = int(duration * 30)  # 30 fps
    kb_template = KB_STYLES[clip_num % len(KB_STYLES)]
    kb = kb_template.format(frames=frames)
    vf = f"scale=8000:-1:flags=lanczos,{kb},setsar=1,fps=30"

    r = subprocess.run(
        ["ffmpeg", "-y", "-loop", "1", "-i", str(img_path),
         "-vf", vf, "-t", str(duration),
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
             "-vf", vf_simple, "-t", str(duration),
             "-c:v", "libx264", "-preset", "fast", "-crf", "25",
             "-profile:v", "baseline", "-level", "3.1",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out_path)],
            capture_output=True, timeout=60,
        )
        return r2.returncode == 0 and out_path.exists()
    return out_path.exists()


def get_sdxl_clips_synced(audio_dur: float) -> list:
    """Generate SDXL clips synchronized to narrator audio duration."""
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # Calculate number of clips: ~6s each, minimum 10
    n_clips = max(10, int(audio_dur / 6))
    durations = calculate_clip_durations(audio_dur, n_clips)
    n_clips = len(durations)  # may have been adjusted by calculate_clip_durations

    print(f"\n🎨 Generando {n_clips} clips SDXL sincronizados con audio ({audio_dur:.1f}s)...")
    print(f"   Duración por clip: {durations[0]:.1f}s" +
          (f" (último: {durations[-1]:.1f}s)" if abs(durations[-1] - durations[0]) > 0.1 else ""))

    ready = []
    ready_durations = []

    for i in range(n_clips):
        img_path  = CLIPS_DIR / f"img_{i:03d}.jpg"
        clip_path = CLIPS_DIR / f"sdxl_{i:03d}.mp4"
        clip_dur  = durations[i]

        if clip_path.exists() and clip_path.stat().st_size > 10_000:
            print(f"   🎞️  SDXL {i:03d}: cached ✓ (regenerando para sync)")
            # Remove cached clip — duration may differ
            clip_path.unlink()

        # Rotate through prompts
        prompt = SDXL_PROMPTS[i % len(SDXL_PROMPTS)]
        print(f"   🎨 SDXL {i:03d}: {prompt[:50]}... ({clip_dur:.1f}s)")

        # Try to generate image (use cached image if available)
        if not (img_path.exists() and img_path.stat().st_size > 1000):
            ok_img = generate_sdxl_image(prompt, img_path)
            if not ok_img:
                print(f"   ⚠️  SDXL {i:03d}: imagen fallida, saltando")
                continue

        ok_clip = image_to_kenburns(img_path, clip_path, i, clip_dur)
        img_path.unlink(missing_ok=True)

        if ok_clip:
            sz = clip_path.stat().st_size // 1024
            print(f"   🎞️  SDXL {i:03d}: OK ({sz}KB, {clip_dur:.1f}s)")
            ready.append(clip_path)
            ready_durations.append(clip_dur)
        else:
            print(f"   ⚠️  SDXL {i:03d}: Ken Burns fallido")

        # Small delay to avoid rate limiting
        if i < n_clips - 1:
            time.sleep(1)

    total_dur = sum(ready_durations)
    print(f"✅ SDXL: {len(ready)}/{n_clips} clips ({total_dur:.1f}s / {audio_dur:.1f}s audio)")

    # If total video duration < audio, generate additional clips to cover
    if total_dur < audio_dur:
        deficit = audio_dur - total_dur
        print(f"\n⚠️  Déficit de {deficit:.1f}s — generando clips adicionales...")
        extra_idx = n_clips
        while total_dur < audio_dur:
            extra_dur = min(MAX_CLIP_DUR, audio_dur - total_dur)
            if extra_dur < 2.0:
                break
            img_path  = CLIPS_DIR / f"img_{extra_idx:03d}.jpg"
            clip_path = CLIPS_DIR / f"sdxl_{extra_idx:03d}.mp4"
            prompt = SDXL_PROMPTS[extra_idx % len(SDXL_PROMPTS)]

            print(f"   🎨 SDXL extra {extra_idx:03d}: {prompt[:50]}... ({extra_dur:.1f}s)")
            ok_img = generate_sdxl_image(prompt, img_path)
            if ok_img:
                ok_clip = image_to_kenburns(img_path, clip_path, extra_idx, extra_dur)
                img_path.unlink(missing_ok=True)
                if ok_clip:
                    ready.append(clip_path)
                    total_dur += extra_dur
                    print(f"   🎞️  SDXL extra {extra_idx:03d}: OK ({extra_dur:.1f}s, total: {total_dur:.1f}s)")
            extra_idx += 1
            time.sleep(1)

        print(f"✅ Total clips: {len(ready)} ({total_dur:.1f}s / {audio_dur:.1f}s audio)")

    return ready


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

def build_concat_list(clips: list, concat_file: Path):
    lines = []
    for c in clips:
        lines.append(f"file '{c.absolute()}'")
    concat_file.write_text("\n".join(lines))


def assemble_final(clips: list, audio_dur: float) -> bool:
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

    # Get raw video duration
    r3 = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(raw_video)],
        capture_output=True
    )
    vid_info = json.loads(r3.stdout)
    vid_dur = float(vid_info["format"]["duration"])
    print(f"   📹 Video duration: {vid_dur:.1f}s")
    print(f"   🎵 Audio duration: {audio_dur:.1f}s")

    # Logo scaled to 80px height (from 500px)
    # Subtitles + scale in single filter_complex (no -vf conflict)
    subs_escaped = str(SUBS_PATH).replace(':', r'\:').replace("'", r"\'")

    filter_complex = (
        # 1. Scale video to 1280x720
        f"[0:v]scale=1280x720:force_original_aspect_ratio=disable[vsc];"
        # 2. Add subtitles
        f"[vsc]subtitles='{subs_escaped}':force_style='FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2'[vsub];"
        # 3. Scale logo separately, then overlay
        f"[1:v]scale=80:-1[logoscaled];"
        f"[vsub][logoscaled]overlay=W-w-25:H-h-25[vout]"
    )

    r4 = subprocess.run(
        ["ffmpeg", "-y",
         "-i", str(raw_video),                    # [0:v]
         "-i", str(LOGO_PATH),                    # [1:v]  (logo, will be scaled)
         "-i", str(AUDIO_PATH),                   # [2:a]
         "-filter_complex", filter_complex,
         "-map", "[vout]",
         "-map", "2:a",
         "-c:v", "libx264",
         "-preset", "fast",
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
        # Fallback: without subtitles, logo scaled
        filter_fb = (
            "[0:v]scale=1280x720:force_original_aspect_ratio=disable[vsc];"
            "[1:v]scale=80:-1[logoscaled];"
            "[vsc][logoscaled]overlay=W-w-25:H-h-25[vout]"
        )
        r5 = subprocess.run(
            ["ffmpeg", "-y",
             "-i", str(raw_video),
             "-i", str(LOGO_PATH),
             "-i", str(AUDIO_PATH),
             "-filter_complex", filter_fb,
             "-map", "[vout]",
             "-map", "2:a",
             "-c:v", "libx264",
             "-preset", "fast",
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
            print(f"❌ assemble fallback también falló: {r5.stderr.decode()[-500:]}")
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
        r4 = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_frames", "-read_intervals", "%+#1",
             "-select_streams", "v",
             "/tmp/brightness_check.jpg"],
            capture_output=True
        )
        bsize = Path("/tmp/brightness_check.jpg").stat().st_size
        brightness_ok = bsize > 5000
        print(f"   Brillo:     {'✅ OK' if brightness_ok else '❌ posible imagen negra'} (frame size: {bsize}B)")

    return width == 1280 and height == 720 and audio == "aac"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("🎬 BitTrader SDXL-Only Video Generator")
    print("   Clips 100% SDXL + Ken Burns, sincronizados con narrador")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Obtener duración del audio primero
    audio_dur = get_audio_duration(AUDIO_PATH)
    print(f"\n🎵 Audio del narrador: {audio_dur:.1f}s")

    # 2. Generar clips SDXL sincronizados con el audio
    clips = get_sdxl_clips_synced(audio_dur)

    print(f"\n📊 Resultado clips:")
    print(f"   SDXL: {len(clips)}")

    if len(clips) < 5:
        print("❌ No hay suficientes clips para generar video")
        sys.exit(1)

    for i, c in enumerate(clips[:6]):
        print(f"   [{i}] {c.name}")
    if len(clips) > 6:
        print(f"   ... (y {len(clips)-6} más)")

    # 3. Ensamblar
    ok = assemble_final(clips, audio_dur)

    # 4. Verificar
    if ok:
        valid = verify_output()
        size = FINAL_OUT.stat().st_size / 1024 / 1024 if FINAL_OUT.exists() else 0

        print(f"\n{'='*60}")
        if valid:
            print(f"✅ VIDEO GENERADO EXITOSAMENTE")
        else:
            print(f"⚠️  VIDEO GENERADO PERO CON ADVERTENCIAS")
        print(f"   Archivo:       {FINAL_OUT}")
        print(f"   Tamaño:        {size:.1f}MB")
        print(f"   Clips SDXL:    {len(clips)}")
        print(f"   Audio:         {audio_dur:.1f}s")
        print(f"{'='*60}")
    else:
        print("❌ FALLO: video no generado")
        sys.exit(1)


if __name__ == "__main__":
    main()
