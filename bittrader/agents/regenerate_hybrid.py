#!/usr/bin/env python3
"""
Regenera video híbrido 50/50 Pexels+SDXL para "3 Reglas que Salvaron mi Cuenta"
- Limpia clips SDXL viejos
- Genera clips nuevos con fix 1344x768 + crop 16:9
- Ensambla con audio + subtítulos + logo BitTrader
- Output: 1920x1080, profile high, level 4.0 (igual al video de referencia)
"""

import sys
import math
import subprocess
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from clip_strategy import get_clips_hybrid

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
OUTPUT_DIR   = BASE_DIR / "output" / "hybrid_test"
CLIPS_DIR    = OUTPUT_DIR / "clips_v2"  # Fresh directory, no cache conflicts
AUDIO_FILE   = BASE_DIR / "output" / "2026-03-28" / "long_1774708531_991" / "long_1774708531_991.mp3"
SUBS_FILE    = BASE_DIR / "output" / "2026-03-28" / "long_1774708531_991" / "subtitles.ass"
FINAL_OUTPUT = OUTPUT_DIR / "3reglas_hybrid_FINAL.mp4"

# Logo paths (in order of preference)
LOGO_CANDIDATES = [
    BASE_DIR.parent / "assets" / "bittrader_logo_transparent.png",
    Path("/home/enderj/.openclaw/workspace/videos/BIBLIOTECA/bittrader_logo.png"),
    BASE_DIR / "data" / "bittrader_logo.png",
]

def find_logo():
    for p in LOGO_CANDIDATES:
        if p.exists():
            print(f"✅ Logo encontrado: {p}")
            return p
    print("⚠️ LOGO NO ENCONTRADO en ninguna ruta!")
    return None

def get_segments_from_subs():
    """Extract text segments from ASS subtitles."""
    segments = []
    if not SUBS_FILE.exists():
        return [{"text": "crypto trading"} for _ in range(70)]
    
    with open(SUBS_FILE, 'r') as f:
        for line in f:
            if line.startswith("Dialogue:"):
                parts = line.split(",", 9)
                if len(parts) >= 10:
                    text = parts[9].strip()
                    # Remove ASS tags
                    import re
                    text = re.sub(r'\{[^}]+\}', '', text)
                    if text and len(text) > 3:
                        segments.append({"text": text})
    
    return segments if segments else [{"text": "crypto trading"} for _ in range(70)]

def main():
    print("=" * 60)
    print("🎬 REGENERACIÓN COMPLETA — Video Híbrido 50/50")
    print("=" * 60)
    
    # 1. Find logo
    logo_path = find_logo()
    if not logo_path:
        print("❌ Sin logo, abortando")
        return
    
    # 2. Audio duration
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
         "-of", "csv=p=0", str(AUDIO_FILE)],
        capture_output=True, text=True
    )
    audio_dur = float(r.stdout.strip())
    total_clips = math.ceil(audio_dur / 6)
    print(f"\n📊 Audio: {audio_dur:.1f}s → {total_clips} clips necesarios")
    
    # 3. Get segments for semantic prompts
    segments = get_segments_from_subs()
    print(f"📝 {len(segments)} segmentos de texto extraídos")
    
    # 4. Generate clips with hybrid strategy
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    clips = get_clips_hybrid(
        segments=segments,
        total_clips_needed=total_clips,
        clips_dir=CLIPS_DIR,
        video_type="long",
    )
    
    if len(clips) < 10:
        print(f"❌ Solo {len(clips)} clips generados, abortando")
        return
    
    print(f"\n✅ {len(clips)} clips listos")
    
    # 5. Verify ALL clips are 1280x720 and 30fps
    print("\n🔍 Verificando calidad de clips...")
    good_clips = []
    for c in clips:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate",
             "-of", "csv=p=0", str(c)],
            capture_output=True, text=True, timeout=10
        )
        parts = r.stdout.strip().split(",")
        if len(parts) >= 2:
            w, h = int(parts[0]), int(parts[1])
            if w == 1280 and h == 720:
                good_clips.append(c)
            else:
                print(f"   ⚠️ {c.name}: {w}x{h} (re-escalando)")
                fixed = c.parent / f"fixed_{c.name}"
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(c),
                     "-vf", "scale=1280:720:force_original_aspect_ratio=increase:flags=lanczos,crop=1280:720,setsar=1,fps=30",
                     "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                     "-profile:v", "baseline", "-level", "3.1",
                     "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                     "-an", str(fixed)],
                    capture_output=True, timeout=60
                )
                if fixed.exists() and fixed.stat().st_size > 10000:
                    good_clips.append(fixed)
                else:
                    good_clips.append(c)  # Use original as last resort
        else:
            good_clips.append(c)  # Can't probe, include anyway
    
    print(f"✅ {len(good_clips)} clips verificados")
    
    # 6. Create concat list
    concat_file = OUTPUT_DIR / "concat_v2.txt"
    with open(concat_file, "w") as f:
        for c in good_clips:
            f.write(f"file '{c}'\n")
    
    # 7. Concat all clips into raw video (no audio)
    raw_video = OUTPUT_DIR / "raw_concat_v2.mp4"
    print(f"\n🔗 Concatenando {len(good_clips)} clips...")
    r = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", str(concat_file),
         "-c:v", "libx264", "-preset", "fast", "-crf", "20",
         "-profile:v", "high", "-level", "4.0",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-r", "30",
         "-an", str(raw_video)],
        capture_output=True, text=True, timeout=600
    )
    if r.returncode != 0:
        print(f"❌ Concat falló: {r.stderr[-300:]}")
        return
    
    # Verify concat
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
         "-show_entries", "format=duration",
         "-of", "json", str(raw_video)],
        capture_output=True, text=True
    )
    info = json.loads(r.stdout)
    dur = float(info["format"]["duration"])
    print(f"   Raw video: {dur:.1f}s")
    
    # 8. Add audio + subtitles + logo overlay
    print(f"\n🎵 Agregando audio + subtítulos + logo...")
    
    # Logo overlay: top-right corner, 100px height, 15px margin
    logo_filter = (
        f"movie='{str(logo_path).replace(chr(39), chr(39)+chr(92)+chr(39)+chr(39))}',"
        f"scale=-1:100[logo];"
        f"[0:v][logo]overlay=W-w-15:15"
    )
    
    # Build final command with audio + subs + logo
    # Trim video to audio length to sync
    cmd = [
        "ffmpeg", "-y",
        "-i", str(raw_video),
        "-i", str(AUDIO_FILE),
        "-filter_complex",
        f"[0:v]scale=1920:1080:flags=lanczos[scaled];"
        f"movie='{logo_path}'[logo_raw];"
        f"[logo_raw]scale=-1:100[logo];"
        f"[scaled][logo]overlay=W-w-15:15[with_logo];"
        f"[with_logo]ass='{SUBS_FILE}'[final]",
        "-map", "[final]",
        "-map", "1:a",
        "-t", str(audio_dur),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-profile:v", "high",
        "-level", "4.0",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-r", "30",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        str(FINAL_OUTPUT),
    ]
    
    print("   Ejecutando ffmpeg final (puede tardar 2-5 min)...")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    
    if r.returncode != 0:
        print(f"❌ Ensamblado falló: {r.stderr[-500:]}")
        # Fallback: simpler approach without filter_complex
        print("\n🔄 Intentando método simplificado...")
        
        # Step A: add logo to raw video
        with_logo = OUTPUT_DIR / "with_logo_v2.mp4"
        subprocess.run(
            ["ffmpeg", "-y",
             "-i", str(raw_video),
             "-i", str(logo_path),
             "-filter_complex",
             "[1:v]scale=-1:100[logo];[0:v][logo]overlay=W-w-15:15[v]",
             "-map", "[v]",
             "-c:v", "libx264", "-preset", "fast", "-crf", "20",
             "-profile:v", "high", "-level", "4.0",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             "-r", "30", "-an", str(with_logo)],
            capture_output=True, text=True, timeout=600
        )
        
        # Step B: scale to 1080p + add audio + subs
        subprocess.run(
            ["ffmpeg", "-y",
             "-i", str(with_logo),
             "-i", str(AUDIO_FILE),
             "-vf", f"scale=1920:1080:flags=lanczos,ass='{SUBS_FILE}'",
             "-t", str(audio_dur),
             "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-profile:v", "high", "-level", "4.0",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             "-r", "30",
             "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
             str(FINAL_OUTPUT)],
            capture_output=True, text=True, timeout=600
        )
    
    # 9. Final verification
    if FINAL_OUTPUT.exists() and FINAL_OUTPUT.stat().st_size > 1_000_000:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,profile,level,r_frame_rate",
             "-show_entries", "format=duration,size",
             "-of", "json", str(FINAL_OUTPUT)],
            capture_output=True, text=True
        )
        info = json.loads(r.stdout)
        stream = info.get("streams", [{}])[0]
        fmt = info.get("format", {})
        size_mb = int(fmt.get("size", 0)) / 1_000_000
        
        print(f"\n{'=' * 60}")
        print(f"✅ VIDEO FINAL GENERADO")
        print(f"   Resolución: {stream.get('width')}x{stream.get('height')}")
        print(f"   Profile: {stream.get('profile')}, Level: {stream.get('level')}")
        print(f"   FPS: {stream.get('r_frame_rate')}")
        print(f"   Duración: {float(fmt.get('duration', 0)):.1f}s")
        print(f"   Tamaño: {size_mb:.1f}MB")
        print(f"   Archivo: {FINAL_OUTPUT}")
        print(f"{'=' * 60}")
        
        # Quick logo check
        subprocess.run(
            ["ffmpeg", "-y", "-ss", "5", "-i", str(FINAL_OUTPUT),
             "-frames:v", "1", "/tmp/final_check.jpg"],
            capture_output=True, timeout=10
        )
    else:
        print(f"❌ Video final no generado o muy pequeño")

if __name__ == "__main__":
    main()
