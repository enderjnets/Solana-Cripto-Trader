#!/usr/bin/env python3
"""
Hailuo (MiniMax) video generation pipeline for BitTrader
Generates 6 clips, checks for green screen, assembles final video
"""

import requests
import json
import time
import os
import subprocess
import sys
from datetime import datetime

API_KEY = "sk-api-dPHO8UbieX0zTe92NRVWSQyN6FfCJQnY0qpbu6RcO12MPhGixFPd--c5pQrkSFNfSft5d6hpvz4w59SZR9GeXyfPN_7pHQPfdqu4_9vFhalNnz0at8r3UV4"
BASE_URL = "https://api.minimaxi.chat/v1"
CLIPS_DIR = "/home/enderj/.openclaw/workspace/bittrader/agents/output/hailuo_test/clips"
OUTPUT_DIR = "/home/enderj/.openclaw/workspace/bittrader/agents/output/hailuo_test"
AUDIO_PATH = "/home/enderj/.openclaw/workspace/bittrader/agents/output/2026-03-28/long_1774708531_991/long_1774708531_991.mp3"
LOGO_PATH = "/home/enderj/.openclaw/workspace/bittrader/assets/bittrader_logo_transparent.png"
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "3reglas_hailuo_test.mp4")
AUDIO_DURATION = 398.9
CLIP_TIMEOUT = 600  # 10 minutes per clip

PROMPTS = [
    "bitcoin trader analyzing charts on multiple screens, professional office, dramatic lighting, cinematic 16:9",
    "risk management concept, trader calculating position size, trading terminal, financial data, 16:9",
    "stop loss order triggered on trading chart, red candles dropping, trader watching screen, 16:9",
    "diversified investment portfolio, multiple asset charts bitcoin ethereum stocks, professional trader, 16:9",
    "successful trader celebrating profit, green candles going up, financial freedom concept, 16:9",
    "call to action trading strategy, trader pointing at camera, charts background, motivational, 16:9",
]

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def submit_video_job(prompt, idx):
    """Submit a video generation job to Hailuo API"""
    payload = {
        "model": "T2V-01",
        "prompt": prompt,
    }
    log(f"Clip {idx+1}/6: Submitting → {prompt[:60]}...")
    try:
        resp = requests.post(f"{BASE_URL}/video_generation", headers=HEADERS, json=payload, timeout=30)
        log(f"Clip {idx+1}: HTTP {resp.status_code}")
        if resp.status_code != 200:
            log(f"Clip {idx+1}: ERROR response: {resp.text[:500]}")
            return None
        data = resp.json()
        log(f"Clip {idx+1}: Response: {json.dumps(data, indent=2)[:300]}")
        # Extract task_id from response
        task_id = data.get("task_id") or data.get("data", {}).get("task_id")
        if not task_id:
            log(f"Clip {idx+1}: No task_id in response: {data}")
            return None
        log(f"Clip {idx+1}: task_id={task_id}")
        return task_id
    except Exception as e:
        log(f"Clip {idx+1}: Submit exception: {e}")
        return None


def poll_job(task_id, idx, timeout=CLIP_TIMEOUT):
    """Poll until job completes or times out"""
    start = time.time()
    poll_interval = 10  # seconds
    log(f"Clip {idx+1}: Polling task {task_id}...")
    
    while time.time() - start < timeout:
        try:
            resp = requests.get(
                f"{BASE_URL}/query/video_generation",
                params={"task_id": task_id},
                headers=HEADERS,
                timeout=30,
            )
            if resp.status_code != 200:
                log(f"Clip {idx+1}: Poll HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(poll_interval)
                continue
            
            data = resp.json()
            # Navigate to status
            status_data = data.get("data", data)
            status = status_data.get("status", "")
            
            elapsed = int(time.time() - start)
            log(f"Clip {idx+1}: [{elapsed}s] status={status}")
            
            if status in ("Success", "Finished", "success", "finished", "completed", "Completed"):
                # Extract file_id
                file_id = (
                    status_data.get("file_id")
                    or status_data.get("video", {}).get("file_id")
                    or status_data.get("output", {}).get("file_id")
                )
                if not file_id:
                    # Try to find file_id anywhere in response
                    log(f"Clip {idx+1}: Full response: {json.dumps(data)[:500]}")
                    file_id = find_file_id(data)
                log(f"Clip {idx+1}: DONE! file_id={file_id}")
                return file_id
            elif status in ("Failed", "failed", "error", "Error"):
                log(f"Clip {idx+1}: FAILED. Response: {json.dumps(data)[:500]}")
                return None
            
            # Still pending
            time.sleep(poll_interval)
        except Exception as e:
            log(f"Clip {idx+1}: Poll exception: {e}")
            time.sleep(poll_interval)
    
    log(f"Clip {idx+1}: TIMEOUT after {timeout}s")
    return None


def find_file_id(data, depth=0):
    """Recursively search for file_id in nested dicts"""
    if depth > 5:
        return None
    if isinstance(data, dict):
        if "file_id" in data:
            return data["file_id"]
        for v in data.values():
            result = find_file_id(v, depth+1)
            if result:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_file_id(item, depth+1)
            if result:
                return result
    return None


def download_clip(file_id, idx):
    """Download video clip from file_id"""
    output_path = os.path.join(CLIPS_DIR, f"clip_{idx+1:02d}.mp4")
    log(f"Clip {idx+1}: Downloading file_id={file_id}")
    
    try:
        # First, get the download URL
        resp = requests.get(
            f"{BASE_URL}/files/retrieve",
            params={"file_id": file_id},
            headers=HEADERS,
            timeout=30,
            allow_redirects=False,
        )
        log(f"Clip {idx+1}: Retrieve HTTP {resp.status_code}")
        
        download_url = None
        if resp.status_code in (301, 302, 303, 307, 308):
            download_url = resp.headers.get("Location")
            log(f"Clip {idx+1}: Redirect to {download_url[:80] if download_url else 'None'}")
        elif resp.status_code == 200:
            data = resp.json()
            log(f"Clip {idx+1}: Retrieve response: {json.dumps(data)[:300]}")
            download_url = (
                data.get("file", {}).get("download_url")
                or data.get("download_url")
                or find_download_url(data)
            )
        
        if not download_url:
            log(f"Clip {idx+1}: No download URL found. Full response: {resp.text[:500]}")
            return None
        
        # Download the actual video
        log(f"Clip {idx+1}: Downloading from URL...")
        video_resp = requests.get(download_url, timeout=120, stream=True)
        if video_resp.status_code != 200:
            log(f"Clip {idx+1}: Download HTTP {video_resp.status_code}")
            return None
        
        with open(output_path, "wb") as f:
            for chunk in video_resp.iter_content(chunk_size=8192):
                f.write(chunk)
        
        size = os.path.getsize(output_path)
        log(f"Clip {idx+1}: Saved {size/1024/1024:.1f}MB → {output_path}")
        return output_path
    except Exception as e:
        log(f"Clip {idx+1}: Download exception: {e}")
        return None


def find_download_url(data, depth=0):
    """Recursively search for download URL in nested dicts"""
    if depth > 5:
        return None
    if isinstance(data, dict):
        for key in ("download_url", "url", "video_url", "link"):
            if key in data and isinstance(data[key], str) and data[key].startswith("http"):
                return data[key]
        for v in data.values():
            result = find_download_url(v, depth+1)
            if result:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_download_url(item, depth+1)
            if result:
                return result
    return None


def check_green_screen(clip_path):
    """Check if clip is a green screen using ffprobe/ffmpeg"""
    try:
        # Sample 3 frames and check average colors
        cmd = [
            "ffprobe", "-v", "error", "-show_streams",
            "-select_streams", "v:0", "-of", "json", clip_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Use ffmpeg to extract a frame and analyze colors
        cmd2 = [
            "ffmpeg", "-i", clip_path, "-vf",
            "thumbnail=300,scale=100:56,extractplanes=r+g+b",
            "-frames:v", "3", "-f", "rawvideo", "-",
        ]
        # Instead use a simpler approach with signalstats
        cmd3 = [
            "ffmpeg", "-i", clip_path, "-vf",
            "select='not(mod(n,30))',scale=100:56,signalstats",
            "-frames:v", "3", "-f", "null", "-",
        ]
        
        # Use histogram approach via ffmpeg
        cmd4 = [
            "ffmpeg", "-i", clip_path,
            "-vf", "select='eq(n,0)+eq(n,30)+eq(n,60)',scale=50:28",
            "-frames:v", "3",
            f"{clip_path}_frame%d.png",
            "-y"
        ]
        subprocess.run(cmd4, capture_output=True, timeout=30)
        
        # Now check the frames
        total_r, total_g, total_b = 0, 0, 0
        frame_count = 0
        for i in range(1, 4):
            frame_path = f"{clip_path}_frame{i}.png"
            if os.path.exists(frame_path):
                # Use ffprobe to get pixel stats
                cmd5 = [
                    "ffmpeg", "-i", frame_path,
                    "-vf", "format=rgb24,signalstats",
                    "-frames:v", "1", "-f", "null", "-"
                ]
                res = subprocess.run(cmd5, capture_output=True, text=True, timeout=15)
                # Parse stats from stderr
                stderr = res.stderr
                # Look for YAVG, UAVG, VAVG or similar
                # Use a simpler python+PIL approach if available
                try:
                    from PIL import Image
                    import numpy as np
                    img = Image.open(frame_path).convert("RGB")
                    arr = list(img.getdata())
                    r = sum(p[0] for p in arr) / len(arr)
                    g = sum(p[1] for p in arr) / len(arr)
                    b = sum(p[2] for p in arr) / len(arr)
                    total_r += r
                    total_g += g
                    total_b += b
                    frame_count += 1
                except ImportError:
                    # Use ffmpeg histogram
                    pass
                os.remove(frame_path)
        
        if frame_count > 0:
            avg_r = total_r / frame_count
            avg_g = total_g / frame_count
            avg_b = total_b / frame_count
            log(f"  Color analysis: R={avg_r:.1f} G={avg_g:.1f} B={avg_b:.1f}")
            
            is_green = (avg_g > avg_r * 1.8 and avg_g > avg_b * 1.8)
            if is_green:
                log(f"  ⚠️  GREEN SCREEN DETECTED - REJECTING")
            else:
                log(f"  ✅ Color OK")
            return is_green
        else:
            log(f"  ⚠️  Could not analyze colors (PIL not available), accepting clip")
            return False
    except Exception as e:
        log(f"  Color check error: {e}, accepting clip")
        return False


def check_green_screen_v2(clip_path):
    """Alternative green screen check using ffmpeg histogram"""
    try:
        # Extract middle frame
        frame_path = clip_path + "_check.png"
        subprocess.run([
            "ffmpeg", "-i", clip_path, "-vf", "select='eq(n,15)',scale=100:56",
            "-frames:v", "1", frame_path, "-y"
        ], capture_output=True, timeout=30)
        
        if not os.path.exists(frame_path):
            log(f"  Could not extract frame for analysis")
            return False
        
        # Try PIL first
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(frame_path).convert("RGB")
            arr = list(img.getdata())
            avg_r = sum(p[0] for p in arr) / len(arr)
            avg_g = sum(p[1] for p in arr) / len(arr)
            avg_b = sum(p[2] for p in arr) / len(arr)
            os.remove(frame_path)
            log(f"  Color analysis: R={avg_r:.1f} G={avg_g:.1f} B={avg_b:.1f}")
            is_green = (avg_g > avg_r * 1.8 and avg_g > avg_b * 1.8)
            if is_green:
                log(f"  ⚠️  GREEN SCREEN DETECTED - REJECTING")
            else:
                log(f"  ✅ Color OK")
            return is_green
        except ImportError:
            pass
        
        # Fallback: use ffmpeg to compute mean pixel values via metadata filter
        res = subprocess.run([
            "ffmpeg", "-i", frame_path,
            "-vf", "format=rgb24,metadata=print:key=lavfi.signalstats.YAVG",
            "-frames:v", "1", "-f", "null", "-"
        ], capture_output=True, text=True, timeout=15)
        
        # Try using identify (ImageMagick) if available
        res2 = subprocess.run(
            ["identify", "-verbose", frame_path],
            capture_output=True, text=True, timeout=15
        )
        
        if res2.returncode == 0:
            # Parse mean values from ImageMagick output
            lines = res2.stdout
            means = {}
            in_channel = None
            for line in lines.split('\n'):
                if 'Red:' in line:
                    in_channel = 'R'
                elif 'Green:' in line:
                    in_channel = 'G'
                elif 'Blue:' in line:
                    in_channel = 'B'
                elif 'mean:' in line.lower() and in_channel:
                    try:
                        val = float(line.split()[-1].strip('()'))
                        if val <= 1.0:
                            val *= 255
                        means[in_channel] = val
                        in_channel = None
                    except:
                        pass
            
            if len(means) == 3:
                avg_r = means.get('R', 128)
                avg_g = means.get('G', 128)
                avg_b = means.get('B', 128)
                log(f"  Color (ImageMagick): R={avg_r:.1f} G={avg_g:.1f} B={avg_b:.1f}")
                is_green = (avg_g > avg_r * 1.8 and avg_g > avg_b * 1.8)
                if is_green:
                    log(f"  ⚠️  GREEN SCREEN DETECTED - REJECTING")
                else:
                    log(f"  ✅ Color OK")
                if os.path.exists(frame_path):
                    os.remove(frame_path)
                return is_green
        
        if os.path.exists(frame_path):
            os.remove(frame_path)
        log(f"  Could not analyze colors, accepting clip")
        return False
    except Exception as e:
        log(f"  Green screen check error: {e}")
        return False


def get_clip_duration(clip_path):
    """Get video clip duration in seconds"""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", clip_path
        ], capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except:
        return 6.0  # Default Hailuo clip duration


def assemble_video(good_clips):
    """Assemble final video from good clips"""
    if not good_clips:
        log("ERROR: No good clips to assemble!")
        return False
    
    log(f"\n=== ASSEMBLING VIDEO ===")
    log(f"Using {len(good_clips)} clips, target duration {AUDIO_DURATION}s")
    
    # Calculate total duration of good clips
    total_clip_duration = sum(get_clip_duration(c) for c in good_clips)
    log(f"Total clip duration: {total_clip_duration:.1f}s")
    
    # Create concat list with looping
    concat_file = os.path.join(OUTPUT_DIR, "concat_list.txt")
    
    # We need to loop the clips to fill AUDIO_DURATION
    loops_needed = max(1, int(AUDIO_DURATION / total_clip_duration) + 2)
    log(f"Looping clips {loops_needed} times to cover {AUDIO_DURATION}s")
    
    with open(concat_file, "w") as f:
        for _ in range(loops_needed):
            for clip in good_clips:
                f.write(f"file '{clip}'\n")
    
    # Intermediate: concat and trim to AUDIO_DURATION, scale to 1280x720
    temp_video = os.path.join(OUTPUT_DIR, "temp_concat.mp4")
    log("Step 1: Concatenating and scaling clips...")
    
    cmd1 = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_file,
        "-t", str(AUDIO_DURATION),
        "-vf", "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an",
        temp_video
    ]
    result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=600)
    if result1.returncode != 0:
        log(f"Concat error: {result1.stderr[-500:]}")
        return False
    log(f"Step 1 done: {os.path.getsize(temp_video)/1024/1024:.1f}MB")
    
    # Step 2: Add logo overlay
    temp_with_logo = os.path.join(OUTPUT_DIR, "temp_logo.mp4")
    log("Step 2: Adding logo overlay...")
    
    cmd2 = [
        "ffmpeg", "-y",
        "-i", temp_video,
        "-i", LOGO_PATH,
        "-filter_complex",
        "[1:v]scale=180:-1[logo];[0:v][logo]overlay=W-w-20:20",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an",
        temp_with_logo
    ]
    result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=600)
    if result2.returncode != 0:
        log(f"Logo error: {result2.stderr[-500:]}")
        # Continue without logo
        temp_with_logo = temp_video
        log("Continuing without logo...")
    else:
        log(f"Step 2 done: {os.path.getsize(temp_with_logo)/1024/1024:.1f}MB")
    
    # Step 3: Add audio
    log("Step 3: Adding audio...")
    cmd3 = [
        "ffmpeg", "-y",
        "-i", temp_with_logo,
        "-i", AUDIO_PATH,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        OUTPUT_VIDEO
    ]
    result3 = subprocess.run(cmd3, capture_output=True, text=True, timeout=600)
    if result3.returncode != 0:
        log(f"Audio error: {result3.stderr[-500:]}")
        return False
    
    # Cleanup temp files
    for f in [temp_video, temp_with_logo, concat_file]:
        if os.path.exists(f) and f != OUTPUT_VIDEO:
            os.remove(f)
    
    final_size = os.path.getsize(OUTPUT_VIDEO) / 1024 / 1024
    log(f"✅ Final video: {OUTPUT_VIDEO}")
    log(f"   Size: {final_size:.1f}MB")
    return True


def main():
    start_time = time.time()
    log("=" * 60)
    log("HAILUO VIDEO PIPELINE - BitTrader")
    log("=" * 60)
    
    # Verify inputs exist
    if not os.path.exists(AUDIO_PATH):
        log(f"ERROR: Audio not found: {AUDIO_PATH}")
        sys.exit(1)
    if not os.path.exists(LOGO_PATH):
        log(f"WARNING: Logo not found: {LOGO_PATH}")
    
    # Phase 1: Submit all jobs first (parallel submission)
    log("\n=== PHASE 1: SUBMITTING ALL JOBS ===")
    task_ids = []
    for idx, prompt in enumerate(PROMPTS):
        task_id = submit_video_job(prompt, idx)
        task_ids.append(task_id)
        if task_id:
            time.sleep(2)  # Small delay between submissions
        else:
            log(f"Clip {idx+1}: Failed to submit, will skip")
    
    log(f"\nSubmitted {sum(1 for t in task_ids if t)} / {len(PROMPTS)} jobs")
    
    # Phase 2: Poll and download
    log("\n=== PHASE 2: POLLING & DOWNLOADING ===")
    good_clips = []
    rejected_clips = 0
    failed_clips = 0
    
    for idx, (task_id, prompt) in enumerate(zip(task_ids, PROMPTS)):
        if not task_id:
            log(f"Clip {idx+1}: Skipped (failed to submit)")
            failed_clips += 1
            continue
        
        # Poll for completion
        file_id = poll_job(task_id, idx)
        if not file_id:
            log(f"Clip {idx+1}: Failed (no file_id)")
            failed_clips += 1
            continue
        
        # Download clip
        clip_path = download_clip(file_id, idx)
        if not clip_path:
            log(f"Clip {idx+1}: Failed to download")
            failed_clips += 1
            continue
        
        # Check for green screen
        log(f"Clip {idx+1}: Checking for green screen...")
        is_green = check_green_screen_v2(clip_path)
        if is_green:
            rejected_clips += 1
            log(f"Clip {idx+1}: REJECTED (green screen)")
        else:
            good_clips.append(clip_path)
            log(f"Clip {idx+1}: ACCEPTED ✅")
    
    log(f"\n=== CLIP SUMMARY ===")
    log(f"  Good clips: {len(good_clips)}")
    log(f"  Rejected (green screen): {rejected_clips}")
    log(f"  Failed: {failed_clips}")
    
    # Phase 3: Assemble video
    if good_clips:
        log("\n=== PHASE 3: ASSEMBLING FINAL VIDEO ===")
        success = assemble_video(good_clips)
    else:
        log("\nERROR: No good clips available for assembly!")
        success = False
    
    # Final report
    elapsed = time.time() - start_time
    log("\n" + "=" * 60)
    log("FINAL REPORT")
    log("=" * 60)
    log(f"  Total time: {elapsed/60:.1f} minutes ({elapsed:.0f}s)")
    log(f"  Clips generated: {len(good_clips) + rejected_clips} / {len(PROMPTS)}")
    log(f"  Clips OK: {len(good_clips)}")
    log(f"  Clips rejected (green screen): {rejected_clips}")
    log(f"  Clips failed: {failed_clips}")
    
    if success and os.path.exists(OUTPUT_VIDEO):
        final_size = os.path.getsize(OUTPUT_VIDEO) / 1024 / 1024
        log(f"  Final video size: {final_size:.1f}MB")
        log(f"  Output: {OUTPUT_VIDEO}")
        log("  STATUS: ✅ SUCCESS")
    else:
        log("  STATUS: ❌ FAILED")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
