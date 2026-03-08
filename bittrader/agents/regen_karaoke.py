#!/usr/bin/env python3
"""
Regenerate a single video with karaoke subtitles.
Usage: python3 regen_karaoke.py <video_index>
       video_index: 0-7 (from production_latest.json)
"""
import sys
import json
import subprocess
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from karaoke_subs import generate_karaoke_subs

DATA_DIR = Path("/home/enderj/.openclaw/workspace/bittrader/agents/data")
PRODUCTION_FILE = DATA_DIR / "production_latest.json"
GUIONES_FILE = DATA_DIR / "guiones_latest.json"


def get_audio_duration(audio_path: Path) -> float:
    try:
        r = subprocess.run([
            "ffprobe", "-v", "quiet", "-show_format", "-print_format", "json",
            str(audio_path)
        ], capture_output=True, text=True, timeout=10)
        return float(json.loads(r.stdout).get("format", {}).get("duration", 30))
    except:
        return 30


def regen_video(index: int):
    production = json.loads(PRODUCTION_FILE.read_text())
    guiones = json.loads(GUIONES_FILE.read_text())
    
    videos = production.get("videos", [])
    scripts = guiones.get("scripts", [])
    
    if index >= len(videos):
        print(f"❌ Index {index} out of range (max {len(videos)-1})")
        return False
    
    video = videos[index]
    title = video.get("title", "Unknown")
    vtype = video.get("type", "short")
    output_file = Path(video.get("output_file", ""))
    
    print(f"\n🎬 [{index}] [{vtype.upper()}] {title[:50]}")
    
    if not output_file.exists():
        print(f"  ❌ Video no encontrado: {output_file}")
        return False
    
    script_dir = output_file.parent
    
    # Find audio file
    audio_files = list(script_dir.glob("*.mp3"))
    if not audio_files:
        print(f"  ❌ Audio no encontrado en {script_dir}")
        return False
    audio_path = audio_files[0]
    
    # Find script text
    script_text = ""
    for s in scripts:
        if s.get("title") == title:
            script_text = s.get("script", "")
            break
    
    if not script_text:
        # Try matching by directory name
        for s in scripts:
            sid = s.get("id", "")
            if sid and sid in str(script_dir):
                script_text = s.get("script", "")
                break
    
    if not script_text:
        print(f"  ⚠️ Script text no encontrado, usando título como fallback")
        script_text = title
    
    duration = get_audio_duration(audio_path)
    print(f"  📝 Script: {len(script_text)} chars, Audio: {duration:.1f}s")
    
    # Clean old subtitle files
    for f in script_dir.glob("subtitles.*"):
        f.unlink(missing_ok=True)
    
    # Generate karaoke subtitles
    sub_path = generate_karaoke_subs(audio_path, script_text, script_dir, vtype)
    
    if not sub_path or not sub_path.exists():
        print(f"  ❌ Subtítulos no generados")
        return False
    
    # Find video clips or intermediate
    clips_dir = script_dir
    clip_files = sorted(clips_dir.glob("clip_*.mp4"))
    
    # Check if we have the concat_scaled intermediate
    intermediate = script_dir / "concat_scaled.mp4"
    
    # We need to re-assemble: take existing video, strip old subs, add new ASS
    # Safest approach: re-encode from intermediate (no subs) + audio + new ASS
    
    # If no intermediate, extract video stream without subs from final
    if not intermediate.exists():
        print(f"  🔧 Extrayendo video sin subtítulos...")
        intermediate = script_dir / "_nosubs.mp4"
        r = subprocess.run([
            "ffmpeg", "-y", "-i", str(output_file),
            "-c:v", "copy", "-an", "-sn",
            str(intermediate)
        ], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            print(f"  ❌ No se pudo extraer video: {r.stderr[:200]}")
            return False
    
    # Build new video with ASS karaoke
    safe_title = title.replace("'", "'\\''").replace('"', '\\"').replace(':', ' -')
    
    sub_escaped = str(sub_path).replace("'", "'\\''").replace(":", "\\:")
    
    if sub_path.suffix == ".ass":
        subtitle_filter = f"ass='{sub_escaped}'"
    else:
        margin_v = "250" if vtype == "short" else "60"
        subtitle_filter = (
            f"subtitles='{sub_escaped}':force_style="
            f"'FontName=DejaVu Sans,FontSize=22,PrimaryColour=&H0000FFFF,"
            f"OutlineColour=&H00000000,Outline=3,Shadow=0,Alignment=2,MarginV={margin_v}'"
        )
    
    # Title overlay
    title_y = "100" if vtype == "short" else "40"
    title_size = "42" if vtype == "short" else "48"
    
    vf = (f"drawtext=text='{safe_title}':fontcolor=gold:fontsize={title_size}:"
          f"x=(w-text_w)/2:y={title_y}:font=DejaVu Sans Bold:borderw=3:bordercolor=black"
          f",{subtitle_filter}")
    
    new_output = script_dir / f"{output_file.stem}_karaoke.mp4"
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(intermediate),
        "-i", str(audio_path),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(new_output)
    ]
    
    print(f"  🎬 Re-ensamblando con karaoke...")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    # Clean intermediate if we created it
    nosubs = script_dir / "_nosubs.mp4"
    if nosubs.exists():
        nosubs.unlink(missing_ok=True)
    
    if r.returncode != 0:
        print(f"  ❌ ffmpeg error: {r.stderr[:300]}")
        return False
    
    if not new_output.exists():
        print(f"  ❌ Output no generado")
        return False
    
    # Replace old video with new one
    output_file.unlink(missing_ok=True)
    new_output.rename(output_file)
    
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  ✅ Video regenerado: {size_mb:.1f}MB → {output_file.name}")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 regen_karaoke.py <index>")
        print("       index: 0-7")
        sys.exit(1)
    
    idx = int(sys.argv[1])
    success = regen_video(idx)
    sys.exit(0 if success else 1)
