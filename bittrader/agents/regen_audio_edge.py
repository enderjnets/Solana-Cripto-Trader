#!/usr/bin/env python3
"""
🔊 Regenerate all 8 published videos with Edge-TTS Jorge audio
Replaces corrupted MiniMax TTS audio, remounts video with new audio + logo + subs
"""
import asyncio
import json
import subprocess
from pathlib import Path

WORKSPACE = Path("/home/enderj/.openclaw/workspace")
BITTRADER = WORKSPACE / "bittrader"
DATA_DIR = BITTRADER / "agents/data"
LOGO_PATH = WORKSPACE / "videos/BIBLIOTECA/bittrader_logo.png"

EDGE_TTS_VOICE = "es-MX-JorgeNeural"
EDGE_TTS_RATE = "+10%"

async def edge_tts(text: str, output_path: Path) -> float:
    import edge_tts
    comm = edge_tts.Communicate(text, EDGE_TTS_VOICE, rate=EDGE_TTS_RATE)
    await comm.save(str(output_path))
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(output_path)],
        capture_output=True, text=True
    )
    return float(r.stdout.strip()) if r.stdout.strip() else 0.0

def remount_video(script_dir: Path, mp3_new: Path, title: str, vtype: str) -> Path:
    """Remount video: existing clips + new audio + logo + subs"""
    # Find concat.txt (list of AI clips used)
    concat_path = script_dir / "concat.txt"
    sub_path = script_dir / "subtitles.ass"
    thumb_path = script_dir / "thumbnail.jpg"
    
    # Find existing final video to get clip info
    final_videos = list(script_dir.glob("*_final.mp4"))
    if not final_videos:
        print(f"  ❌ No final video found in {script_dir}")
        return None
    
    existing = final_videos[0]
    
    # Get duration of new audio
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(mp3_new)],
        capture_output=True, text=True
    )
    duration = float(r.stdout.strip()) if r.stdout.strip() else 60.0
    
    # Remount: existing video (no audio) + new audio + logo + subs
    # Use the existing video file but replace audio
    output_path = script_dir / existing.name.replace("_final.mp4", "_v2_final.mp4")
    
    # Check if concat.txt exists for clip re-assembly
    if concat_path.exists():
        # Re-assemble from clips
        if vtype == "short":
            vf_scale = (
                "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
                "crop=1080:1920,avgblur=30[bg];"
                "[0:v]scale=1080:-2:force_original_aspect_ratio=decrease[fg];"
                "[bg][fg]overlay=(W-w)/2:(H-h)/2"
            )
            intermediate = script_dir / "_concat_v2.mp4"
            r1 = subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_path),
                "-filter_complex", vf_scale,
                "-t", str(duration + 1),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-an",
                str(intermediate)
            ], capture_output=True, text=True, timeout=120)
            
            if r1.returncode != 0:
                print(f"  ⚠️ Blur-fill failed, using existing video: {r1.stderr[:100]}")
                intermediate = existing
        else:
            # Long: scale to 1920x1080 with black bars
            intermediate = script_dir / "_concat_v2.mp4"
            r1 = subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_path),
                "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black",
                "-t", str(duration + 1),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-an",
                str(intermediate)
            ], capture_output=True, text=True, timeout=120)
            
            if r1.returncode != 0:
                print(f"  ⚠️ Scale failed, using existing video: {r1.stderr[:100]}")
                intermediate = existing
    else:
        # No concat.txt — extract video track from existing
        intermediate = script_dir / "_video_only.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(existing), "-an", "-c:v", "copy",
            str(intermediate)
        ], capture_output=True, timeout=30)

    # Assembly: video + new audio + logo + subs
    logo_size = 240 if vtype == "short" else 180
    
    if LOGO_PATH.exists() and sub_path.exists():
        sub_esc = str(sub_path).replace(":", "\\:")
        filter_complex = (
            f"[2:v]scale={logo_size}:-1[logo];"
            f"[0:v][logo]overlay=W-w-30:30:format=auto,ass='{sub_esc}'"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(intermediate),
            "-i", str(mp3_new),
            "-i", str(LOGO_PATH),
            "-filter_complex", filter_complex,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
            "-shortest", "-movflags", "+faststart",
            str(output_path)
        ]
    elif LOGO_PATH.exists():
        cmd = [
            "ffmpeg", "-y",
            "-i", str(intermediate),
            "-i", str(mp3_new),
            "-i", str(LOGO_PATH),
            "-filter_complex", f"[2:v]scale={logo_size}:-1[logo];[0:v][logo]overlay=W-w-30:30:format=auto",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
            "-shortest", "-movflags", "+faststart",
            str(output_path)
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(intermediate),
            "-i", str(mp3_new),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
            "-shortest", "-movflags", "+faststart",
            str(output_path)
        ]
    
    r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    # Cleanup temp
    for tmp in ["_concat_v2.mp4", "_video_only.mp4"]:
        (script_dir / tmp).unlink(missing_ok=True)
    
    if r2.returncode != 0:
        print(f"  ❌ Assembly failed: {r2.stderr[:200]}")
        return None
    
    return output_path if output_path.exists() else None


async def main():
    guiones = json.loads((DATA_DIR / "guiones_latest.json").read_text())
    scripts = guiones["scripts"]
    prod = json.loads((DATA_DIR / "production_latest.json").read_text())
    
    print(f"\n🔊 Regenerating audio for {len(scripts)} videos with Edge-TTS Jorge\n")
    
    results = []
    
    for i, (script, video) in enumerate(zip(scripts, prod["videos"]), 1):
        title = video["title"]
        vtype = video.get("type", "short")
        output_file = Path(video["output_file"])
        script_dir = output_file.parent
        script_text = script.get("script", "")
        
        print(f"[{i}/8] [{vtype.upper()}] {title[:45]}")
        
        if not script_text:
            print(f"  ❌ No script text found")
            continue
        
        # 1. Generate new audio
        mp3_new = script_dir / f"{output_file.stem}_edge.mp3"
        print(f"  🔊 Generating Edge-TTS...")
        
        try:
            duration = await edge_tts(script_text, mp3_new)
            print(f"  ✅ Audio: {duration:.1f}s | {mp3_new.stat().st_size/1024:.0f}KB")
        except Exception as e:
            print(f"  ❌ TTS failed: {e}")
            continue
        
        # 2. Regenerate karaoke subtitles with new timing
        from karaoke_subs import generate_karaoke_subs
        print(f"  📝 Regenerating karaoke subs...")
        try:
            sub_path = generate_karaoke_subs(mp3_new, script_text, script_dir, vtype)
            print(f"  ✅ Subs: {sub_path}")
        except Exception as e:
            print(f"  ⚠️ Subs failed: {e}")
        
        # 3. Remount video
        print(f"  🎬 Remounting video...")
        output_v2 = remount_video(script_dir, mp3_new, title, vtype)
        
        if output_v2:
            size_mb = output_v2.stat().st_size / 1024 / 1024
            print(f"  ✅ Video: {output_v2.name} ({size_mb:.1f}MB)")
            video["output_file_v2"] = str(output_v2)
            results.append({"title": title, "type": vtype, "path": str(output_v2), "youtube_id": video.get("youtube_id","")})
        else:
            print(f"  ❌ Remount failed")
    
    # Save updated production file
    prod["v2_videos"] = results
    (DATA_DIR / "production_v2.json").write_text(json.dumps(prod, indent=2))
    
    print(f"\n{'='*50}")
    print(f"✅ {len(results)}/8 videos regenerated")
    print(f"📁 Saved to production_v2.json")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    asyncio.run(main())
