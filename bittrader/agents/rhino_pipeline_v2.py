#!/usr/bin/env python3
"""
🦏 BitTrader Rhino Pipeline v2 — Batch Submit + Parallel Poll
Strategy:
  Phase 1: Submit ALL clips across all videos (with 35s rate limit between each)
  Phase 2: Poll all pending tasks until done
  Phase 3: Download + assemble + upload one video at a time

This avoids the issue of long serial waits killing the session.
Run with: python3 rhino_pipeline_v2.py [phase1|phase2|phase3|all]
"""

import json
import sys
import time
import subprocess
import requests
import shutil
from pathlib import Path
from datetime import datetime, timezone

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE   = Path("/home/enderj/.openclaw/workspace")
BITTRADER   = WORKSPACE / "bittrader"
AGENTS      = BITTRADER / "agents"
DATA_DIR    = AGENTS / "data"
OUTPUT_BASE = AGENTS / "output" / "2026-03-07-v3"
RHINO_OUT   = AGENTS / "output" / "rhino_v1"
LOGO        = WORKSPACE / "videos/BIBLIOTECA/bittrader_logo.png"
YT_CREDS    = WORKSPACE / "memory/youtube_credentials.json"

sys.path.insert(0, str(WORKSPACE / "youtube_env/lib/python3.13/site-packages"))
sys.path.insert(0, str(AGENTS))

DATA_DIR.mkdir(parents=True, exist_ok=True)
RHINO_OUT.mkdir(parents=True, exist_ok=True)

# State file
STATE_FILE  = DATA_DIR / "rhino_pipeline_state.json"
DONE_FILE   = DATA_DIR / "rhino_rebuild_done.json"

# ── Minimax API ────────────────────────────────────────────────────────────
MINIMAX_KEY = json.loads((BITTRADER / "keys/minimax.json").read_text())["minimax_api_key"]
SUBMIT_URL  = "https://api.minimax.io/v1/video_generation"
POLL_URL    = "https://api.minimax.io/v1/query/video_generation"
FILES_URL   = "https://api.minimax.io/v1/files/retrieve"
HEADERS = {"Authorization": f"Bearer {MINIMAX_KEY}", "Content-Type": "application/json"}

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ══════════════════════════════════════════════════════════════════════════
# VIDEO DEFINITIONS — ordered by publish date
# ══════════════════════════════════════════════════════════════════════════

VIDEOS = [
    {
        "id": "pengu",
        "title": "Por qué PENGU está subiendo mientras BTC cae",
        "type": "short",
        "audio_source": str(OUTPUT_BASE / "short_1772925869_724/short_1772925869_724_v4_final.mp4"),
        "audio_is_video": True,
        "audio_duration": 47.112,
        "yt_id_old": "gm4_LB4eFj0",
        "publish_at": "2026-03-10T14:00:00.000Z",
        "prompts": [
            "Anthropomorphic rhinoceros looking puzzled at a small cute penguin standing on a rising green price chart, confused but intrigued expression, dark crypto trading room, hyper-realistic 3D render, cinematic lighting, wearing modern casual trading clothes, 9:16 vertical",
            "Anthropomorphic rhinoceros in NFT gallery surrounded by digital penguin art pieces glowing with value tags, impressed expression, futuristic space, hyper-realistic 3D render, photorealistic textures, wearing modern casual trading clothes, 9:16 vertical",
            "Anthropomorphic rhinoceros scrolling on phone showing social media memecoin hype with laughing face emoji, skeptical but curious expression, neon background, hyper-realistic 3D render, wearing modern casual trading clothes, 9:16 vertical",
        ],
    },
    {
        "id": "akt",
        "title": "AKT explota mientras BTC cae un 1.4%",
        "type": "short",
        "audio_source": str(OUTPUT_BASE / "short_1772925855_875/short_1772925855_875_final_edge.mp3"),
        "audio_is_video": False,
        "audio_duration": 48.576,
        "yt_id_old": "DGOvGLZburk",
        "publish_at": "2026-03-10T16:00:00.000Z",
        "prompts": [
            "Anthropomorphic rhinoceros pointing at a glowing AKT coin logo rising against falling BTC chart, excited expression, dark trading room with red and green lights, hyper-realistic 3D render, wearing modern casual trading clothes, 9:16 vertical",
            "Anthropomorphic rhinoceros at server rack wall representing decentralized GPU compute, amazed expression, futuristic data center neon blue glow, hyper-realistic 3D render, wearing modern casual trading clothes, 9:16 vertical",
            "Anthropomorphic rhinoceros holding a smartphone showing crypto price pumping with green candles, confident smirk, dark moody background, hyper-realistic 3D render, wearing modern casual trading clothes, 9:16 vertical",
        ],
    },
    {
        "id": "traders90",
        "title": "El 90% de traders pierde por esto",
        "type": "short",
        "audio_source": str(OUTPUT_BASE / "short_1772925884_383/short_1772925884_383_v4_final.mp4"),
        "audio_is_video": True,
        "audio_duration": 46.632,
        "yt_id_old": "XazgVqCkGCc",
        "publish_at": "2026-03-10T19:00:00.000Z",
        "prompts": [
            "Anthropomorphic rhinoceros with overconfident smirk counting money after a win, surrounded by warning signs being ignored, bright lights, hyper-realistic 3D render, wearing modern casual trading clothes, 9:16 vertical",
            "Anthropomorphic rhinoceros sitting devastated at trading desk with red portfolio screen, head in hands, dramatic dark lighting, hyper-realistic 3D render, wearing modern casual trading clothes, 9:16 vertical",
            "Anthropomorphic rhinoceros calmly studying a trading journal with a pen, disciplined focused expression, warm desk lamp, hyper-realistic 3D render, wearing modern casual trading clothes, 9:16 vertical",
        ],
    },
    {
        "id": "pi",
        "title": "PI coin +13%: ¿trampa o despegue real?",
        "type": "short",
        "audio_source": str(OUTPUT_BASE / "short_1772925672_305/short_1772925672_305_v3_final.mp4"),
        "audio_is_video": True,
        "audio_duration": 48.144,
        "yt_id_old": "7bGQROO-Y9o",
        "publish_at": "2026-03-11T01:00:00.000Z",
        "prompts": [
            "Anthropomorphic rhinoceros holding a cracked golden coin looking at it suspiciously, magnifying glass, dark background with question marks floating, hyper-realistic 3D render, wearing modern casual trading clothes, 9:16 vertical",
            "Anthropomorphic rhinoceros looking at phone showing crypto price plus 13 percent, eyebrow raised skeptically, red and green chart behind, hyper-realistic 3D render, wearing modern casual trading clothes, 9:16 vertical",
            "Anthropomorphic rhinoceros presenting a bear trap disguised as green chart, warning gesture, dramatic cinematic lighting, hyper-realistic 3D render, wearing modern casual trading clothes, 9:16 vertical",
        ],
    },
    {
        "id": "zec",
        "title": "ZEC cae 7%: ¿el fin de las privacy coins?",
        "type": "short",
        "audio_source": str(OUTPUT_BASE / "short_1772925742_739/short_1772925742_739_v3_final.mp4"),
        "audio_is_video": True,
        "audio_duration": 47.016,
        "yt_id_old": "rXZNuFtV8AE",
        "publish_at": "2026-03-12T14:00:00.000Z",
        "prompts": [
            "Anthropomorphic rhinoceros watching a cracked Zcash coin falling through the floor, concerned expression, dark trading room, hyper-realistic 3D render, wearing modern casual trading clothes, 9:16 vertical",
            "Anthropomorphic rhinoceros in a dark hood representing privacy, looking over shoulder nervously at government surveillance screens, hyper-realistic 3D render, wearing modern casual trading clothes, 9:16 vertical",
            "Anthropomorphic rhinoceros gesturing at a graveyard of crypto coins with one highlighted, philosophical expression, cinematic lighting, hyper-realistic 3D render, wearing modern casual trading clothes, 9:16 vertical",
        ],
    },
    {
        "id": "claude_trades",
        "title": "Le di mis trades a Claude y los resultados me sorprendieron",
        "type": "long",
        "audio_source": str(OUTPUT_BASE / "long_1772925728_400/long_1772925728_400_edge_clean.mp3"),
        "audio_is_video": False,
        "audio_duration": 250.320,
        "yt_id_old": "RhZo-Lhs90A",
        "publish_at": "2026-03-12T16:00:00.000Z",
        "prompts": [
            "Anthropomorphic rhinoceros frustrated at trading desk with losing red charts, 3 months of losses visible, dramatic dark office, hyper-realistic 3D render, wearing modern casual trading clothes, 16:9 horizontal",
            "Anthropomorphic rhinoceros typing trading history into an AI chat interface, hopeful expression, laptop glow illuminating face, dark room, hyper-realistic 3D render, wearing modern casual trading clothes, 16:9 horizontal",
            "Anthropomorphic rhinoceros shocked reading AI analysis report showing patterns in losses, jaw dropped, papers flying, dramatic light, hyper-realistic 3D render, wearing modern casual trading clothes, 16:9 horizontal",
            "Anthropomorphic rhinoceros making disciplined trading changes, writing new rules in journal, focused determined expression, clean modern desk, hyper-realistic 3D render, wearing modern casual trading clothes, 16:9 horizontal",
            "Anthropomorphic rhinoceros celebrating first green month, portfolio showing profit, triumphant fist pump, golden light, hyper-realistic 3D render, wearing modern casual trading clothes, 16:9 horizontal",
        ],
    },
    {
        "id": "fondeada",
        "title": "De $0 a cuenta fondeada en 30 días",
        "type": "long",
        "audio_source": str(OUTPUT_BASE / "long_1772925781_782/long_1772925781_782_v3_final.mp4"),
        "audio_is_video": True,
        "audio_duration": 254.928,
        "yt_id_old": "I7dK9aGl9mk",
        "publish_at": "2026-03-13T16:00:00.000Z",
        "prompts": [
            "Anthropomorphic rhinoceros at starting line with zero dollar counter, determined expression, dramatic stadium lighting, challenge banner in background, hyper-realistic 3D render, wearing modern casual trading clothes, 16:9 horizontal",
            "Anthropomorphic rhinoceros following strict trading rules checklist day by day, 30-day calendar on wall, disciplined focused expression, hyper-realistic 3D render, wearing modern casual trading clothes, 16:9 horizontal",
            "Anthropomorphic rhinoceros celebrating passing a trading challenge, funded account certificate, confetti, triumphant pose, hyper-realistic 3D render, wearing modern casual trading clothes, 16:9 horizontal",
            "Anthropomorphic rhinoceros managing funded trading account professionally, multiple monitors, calm confident expression, professional trading room, hyper-realistic 3D render, wearing modern casual trading clothes, 16:9 horizontal",
            "Anthropomorphic rhinoceros looking at camera motivationally, holding a trading success sign, inspiring expression, studio lighting, hyper-realistic 3D render, wearing modern casual trading clothes, 16:9 horizontal",
        ],
    },
    {
        "id": "bot_noticias",
        "title": "El bot que lee noticias y ejecuta tus trades solo",
        "type": "long",
        "audio_source": str(OUTPUT_BASE / "long_1772925821_433/long_1772925821_433_v3_final.mp4"),
        "audio_is_video": True,
        "audio_duration": 272.712,
        "yt_id_old": "zSQgH3aRxxY",
        "publish_at": "2026-03-13T14:00:00.000Z",
        "prompts": [
            "Anthropomorphic rhinoceros sleeping peacefully while robot arms trade on screens behind them, autonomous AI bots working, dark room, hyper-realistic 3D render, wearing modern casual trading clothes, 16:9 horizontal",
            "Anthropomorphic rhinoceros watching news headlines transform into buy sell orders automatically, AI processing visual effect, impressed expression, hyper-realistic 3D render, wearing modern casual trading clothes, 16:9 horizontal",
            "Anthropomorphic rhinoceros on left slowly reading news manually, robotic rhino on right executing trades instantly in split screen, speed comparison, hyper-realistic 3D render, wearing modern casual trading clothes, 16:9 horizontal",
            "Anthropomorphic rhinoceros presenting a flowchart News to AI Analysis to Trade Execution, professor mode, clean background, hyper-realistic 3D render, wearing modern casual trading clothes, 16:9 horizontal",
            "Anthropomorphic rhinoceros checking phone at beach showing autonomous bot profits, relaxed successful lifestyle, golden light, hyper-realistic 3D render, wearing modern casual trading clothes, 16:9 horizontal",
        ],
    },
]

# ══════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"clips": {}, "videos": {}}

def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))

# ══════════════════════════════════════════════════════════════════════════
# PHASE 1: SUBMIT ALL CLIPS
# ══════════════════════════════════════════════════════════════════════════

def phase1_submit_clips():
    """Submit all clips to Hailuo. Saves task IDs to state."""
    log("🚀 PHASE 1: Submitting all clips to Hailuo...")
    state = load_state()
    clips_state = state.get("clips", {})
    
    total_submitted = 0
    total_already = 0
    
    for video in VIDEOS:
        vid_id = video["id"]
        prompts = video["prompts"]
        
        for i, prompt in enumerate(prompts):
            clip_key = f"{vid_id}_clip{i+1}"
            
            # Skip if already submitted or done
            existing = clips_state.get(clip_key, {})
            if existing.get("status") in ("submitted", "done", "downloaded"):
                log(f"  ⏭️  {clip_key}: already {existing['status']}")
                total_already += 1
                continue
            
            # Check if already downloaded
            clip_path = RHINO_OUT / vid_id / "clips" / f"rhino_clip_{i+1}.mp4"
            if clip_path.exists() and clip_path.stat().st_size > 100_000:
                log(f"  ✅ {clip_key}: already downloaded at {clip_path}")
                clips_state[clip_key] = {
                    "status": "downloaded",
                    "path": str(clip_path),
                    "vid_id": vid_id,
                    "clip_idx": i,
                }
                total_already += 1
                state["clips"] = clips_state
                save_state(state)
                continue
            
            # Submit
            log(f"  📤 Submitting {clip_key}: {prompt[:70]}...")
            try:
                resp = requests.post(
                    SUBMIT_URL, headers=HEADERS,
                    json={"model": "T2V-01", "prompt": prompt, "prompt_optimizer": True},
                    timeout=30
                )
                data = resp.json()
                task_id = data.get("task_id")
                if task_id:
                    clips_state[clip_key] = {
                        "status": "submitted",
                        "task_id": task_id,
                        "vid_id": vid_id,
                        "clip_idx": i,
                        "submitted_at": datetime.now().isoformat(),
                    }
                    log(f"  ✅ {clip_key}: task_id={task_id}")
                    total_submitted += 1
                else:
                    log(f"  ❌ {clip_key}: submit failed: {data}")
                    clips_state[clip_key] = {"status": "failed", "error": str(data)}
            except Exception as e:
                log(f"  ❌ {clip_key}: exception: {e}")
                clips_state[clip_key] = {"status": "failed", "error": str(e)}
            
            state["clips"] = clips_state
            save_state(state)
            
            # Rate limit between submissions
            if total_submitted > 0:
                log(f"  ⏳ Rate limit: waiting 35s...")
                time.sleep(35)
    
    log(f"\n✅ Phase 1 complete: {total_submitted} submitted, {total_already} skipped")
    return state

# ══════════════════════════════════════════════════════════════════════════
# PHASE 2: POLL + DOWNLOAD ALL CLIPS
# ══════════════════════════════════════════════════════════════════════════

def phase2_poll_and_download():
    """Poll all submitted tasks until done, then download."""
    log("🔄 PHASE 2: Polling and downloading all clips...")
    state = load_state()
    clips_state = state.get("clips", {})
    
    max_wait = 700  # seconds
    start = time.time()
    
    while True:
        pending = []
        done_count = 0
        fail_count = 0
        
        for clip_key, clip_info in clips_state.items():
            status = clip_info.get("status", "")
            if status == "submitted":
                pending.append((clip_key, clip_info))
            elif status in ("downloaded", "done"):
                done_count += 1
            elif status == "failed":
                fail_count += 1
        
        if not pending:
            log(f"  ✅ All done! {done_count} downloaded, {fail_count} failed")
            break
        
        elapsed = time.time() - start
        if elapsed > max_wait:
            log(f"  ⚠️  Timeout after {max_wait}s. {len(pending)} still pending.")
            break
        
        log(f"  📊 Pending: {len(pending)}, Done: {done_count}, Failed: {fail_count} [{int(elapsed)}s elapsed]")
        
        for clip_key, clip_info in pending:
            task_id = clip_info.get("task_id")
            if not task_id:
                continue
            
            try:
                resp = requests.get(
                    POLL_URL, headers=HEADERS,
                    params={"task_id": task_id},
                    timeout=15
                )
                data = resp.json()
                poll_status = data.get("status", "")
                
                if poll_status == "Success":
                    file_id = data.get("file_id")
                    log(f"  ✅ {clip_key}: Success! file_id={file_id}")
                    
                    # Download
                    vid_id = clip_info["vid_id"]
                    clip_idx = clip_info["clip_idx"]
                    clips_dir = RHINO_OUT / vid_id / "clips"
                    clips_dir.mkdir(parents=True, exist_ok=True)
                    clip_path = clips_dir / f"rhino_clip_{clip_idx+1}.mp4"
                    
                    dl_ok = download_clip(file_id, clip_path)
                    if dl_ok:
                        clips_state[clip_key]["status"] = "downloaded"
                        clips_state[clip_key]["path"] = str(clip_path)
                    else:
                        clips_state[clip_key]["status"] = "failed"
                        clips_state[clip_key]["error"] = "download_failed"
                    
                elif poll_status in ("Fail", "Failed"):
                    log(f"  ❌ {clip_key}: Failed: {data}")
                    clips_state[clip_key]["status"] = "failed"
                else:
                    log(f"  ⏳ {clip_key}: {poll_status}")
                    
            except Exception as e:
                log(f"  ⚠️  {clip_key}: poll exception: {e}")
        
        state["clips"] = clips_state
        save_state(state)
        
        # Wait before next poll round
        time.sleep(20)
    
    state["clips"] = clips_state
    save_state(state)
    log("✅ Phase 2 complete")
    return state


def download_clip(file_id: str, out_path: Path) -> bool:
    """Download clip by file_id."""
    try:
        resp = requests.get(
            FILES_URL, headers=HEADERS,
            params={"file_id": file_id},
            timeout=15
        )
        data = resp.json()
        file_info = data.get("file", {})
        download_url = file_info.get("download_url") or data.get("download_url")
        
        if not download_url:
            log(f"  No download_url in: {data}")
            return False
        
        dl = requests.get(download_url, timeout=120, stream=True)
        with open(out_path, "wb") as f:
            for chunk in dl.iter_content(chunk_size=8192):
                f.write(chunk)
        
        size_kb = out_path.stat().st_size // 1024
        log(f"  Downloaded: {out_path.name} ({size_kb}KB)")
        return size_kb > 100
    except Exception as e:
        log(f"  Download exception: {e}")
        return False

# ══════════════════════════════════════════════════════════════════════════
# PHASE 3: ASSEMBLE + UPLOAD
# ══════════════════════════════════════════════════════════════════════════

def extract_frame(clip_path: Path, out_path: Path, ts: float = 0.5) -> bool:
    cmd = ["ffmpeg", "-y", "-ss", str(ts), "-i", str(clip_path), 
           "-vframes", "1", "-q:v", "2", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return out_path.exists() and out_path.stat().st_size > 0


def apply_ken_burns(frame_path: Path, out_path: Path, duration: float, 
                    video_type: str, direction: int = 0) -> bool:
    dur_frames = max(30, int(duration * 30))
    
    if video_type == "short":
        w, h = 1080, 1920
        zoom_speed = "min(zoom+0.0008,1.4)"
    else:
        w, h = 1920, 1080
        zoom_speed = "min(zoom+0.0005,1.3)"
    
    # Safe pan expressions — using 'if' to clamp at edges
    pan_exprs = [
        ("iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),  # zoom-center
        ("iw/2-(iw/zoom/2)", "ih/4-(ih/zoom/4)"),  # zoom top-center
        ("iw/4-(iw/zoom/4)", "ih/2-(ih/zoom/2)"),  # zoom left-center
    ]
    x_expr, y_expr = pan_exprs[direction % 3]
    
    # Use 1.5x intermediate size for KB (less memory/CPU than 2x)
    scale_mult = 1.5
    iw = int(w * scale_mult)
    ih = int(h * scale_mult)
    # Make even
    iw = iw if iw % 2 == 0 else iw + 1
    ih = ih if ih % 2 == 0 else ih + 1
    
    vf = (
        f"scale={iw}:{ih}:force_original_aspect_ratio=increase,"
        f"crop={iw}:{ih},"
        f"zoompan=z='{zoom_speed}':d={dur_frames}"
        f":x='{x_expr}':y='{y_expr}':s={w}x{h},"
        f"setsar=1"
    )
    
    # Longer timeout for longs, shorter for shorts
    kb_timeout = 600 if video_type == "long" else 300
    
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(frame_path),
        "-vf", vf,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-r", "30", "-an",
        str(out_path)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=kb_timeout)
    ok = out_path.exists() and out_path.stat().st_size > 0
    if not ok:
        log(f"  Ken Burns error: {r.stderr[-300:]}")
        # Try simpler fallback: just scale to output size without zoompan
        vf_simple = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1"
        cmd2 = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(frame_path),
            "-vf", vf_simple,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-r", "30", "-an",
            str(out_path)
        ]
        r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=600)
        ok = out_path.exists() and out_path.stat().st_size > 0
        if not ok:
            log(f"  Simple fallback also failed: {r2.stderr[-200:]}")
    return ok


def extract_audio(source: str, out_path: Path) -> bool:
    src = Path(source)
    if src.suffix == ".mp3":
        shutil.copy2(str(src), str(out_path))
        return True
    cmd = ["ffmpeg", "-y", "-i", str(src),
           "-vn", "-ar", "44100", "-ac", "1", "-b:a", "128k", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return out_path.exists() and out_path.stat().st_size > 0


def assemble_one_video(video: dict, state: dict) -> Path | None:
    """Assemble a single video from clips."""
    vid_id = video["id"]
    video_type = video["type"]
    audio_dur = video["audio_duration"]
    num_prompts = len(video["prompts"])
    work_dir = RHINO_OUT / vid_id
    work_dir.mkdir(parents=True, exist_ok=True)
    
    log(f"  📦 Assembling {vid_id}...")
    
    # Gather clips
    clips_state = state.get("clips", {})
    clip_paths = []
    for i in range(num_prompts):
        clip_key = f"{vid_id}_clip{i+1}"
        info = clips_state.get(clip_key, {})
        if info.get("status") == "downloaded":
            p = Path(info["path"])
            if p.exists():
                clip_paths.append(p)
            else:
                clip_paths.append(None)
        else:
            clip_paths.append(None)
    
    valid_clips = [p for p in clip_paths if p]
    
    # If no valid clips, try to find any downloaded clip from clips dir
    if not valid_clips:
        clips_dir = RHINO_OUT / vid_id / "clips"
        if clips_dir.exists():
            existing = sorted(clips_dir.glob("*.mp4"))
            valid_clips = [p for p in existing if p.stat().st_size > 100_000]
    
    if not valid_clips:
        log(f"  ❌ No valid clips for {vid_id}")
        return None
    
    log(f"  Clips: {len(valid_clips)}/{num_prompts} valid")
    
    # Fill missing clips with last valid (cycle through available)
    filled_clips = []
    for i in range(num_prompts):
        if clip_paths[i]:
            filled_clips.append(clip_paths[i])
        else:
            # Use modulo to cycle through available clips
            filled_clips.append(valid_clips[i % len(valid_clips)])
    
    # Filter out Nones
    filled_clips = [p for p in filled_clips if p]
    actual_count = len(filled_clips)
    dur_per_clip = audio_dur / actual_count
    
    # Step 1: Extract audio
    audio_path = work_dir / "audio.mp3"
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        if not extract_audio(video["audio_source"], audio_path):
            log(f"  ❌ Audio extraction failed")
            return None
        log(f"  Audio extracted: {audio_path}")
    
    # Step 2: For each clip → extract frame → Ken Burns
    kb_clips = []
    for i, clip_path in enumerate(filled_clips):
        kb_path = work_dir / f"kb_{i+1}.mp4"
        if kb_path.exists() and kb_path.stat().st_size > 0:
            log(f"  KB clip {i+1}: already exists")
            kb_clips.append(kb_path)
            continue
        
        frame_path = work_dir / f"frame_{i+1}.png"
        if not extract_frame(clip_path, frame_path, 0.5):
            extract_frame(clip_path, frame_path, 1.0)
        
        if not frame_path.exists() or frame_path.stat().st_size == 0:
            log(f"  ⚠️  Frame {i+1} failed, trying middle of clip")
            extract_frame(clip_path, frame_path, 3.0)
        
        if not frame_path.exists() or frame_path.stat().st_size == 0:
            log(f"  ❌ Frame {i+1} completely failed")
            continue
        
        ok = apply_ken_burns(frame_path, kb_path, dur_per_clip, video_type, i % 3)
        if ok:
            log(f"  KB clip {i+1}: ✅ ({kb_path.stat().st_size//1024}KB)")
            kb_clips.append(kb_path)
        else:
            log(f"  KB clip {i+1}: ❌ failed")
    
    if not kb_clips:
        log(f"  ❌ No Ken Burns clips for {vid_id}")
        return None
    
    # Step 3: Concatenate
    concat_list = work_dir / "concat.txt"
    with open(concat_list, "w") as f:
        for kb in kb_clips:
            f.write(f"file '{kb}'\n")
    
    concat_path = work_dir / "concat.mp4"
    if not concat_path.exists() or concat_path.stat().st_size == 0:
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
               "-c:v", "libx264", "-preset", "fast", "-crf", "23", str(concat_path)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if not concat_path.exists():
            log(f"  ❌ Concat failed: {r.stderr[-200:]}")
            return None
        log(f"  Concatenated: {concat_path}")
    
    # Step 4: Merge video + audio
    merged_path = work_dir / "merged.mp4"
    if not merged_path.exists() or merged_path.stat().st_size == 0:
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", str(concat_path),
            "-i", str(audio_path),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            str(merged_path)
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if not merged_path.exists():
            log(f"  ❌ Merge failed: {r.stderr[-200:]}")
            return None
        log(f"  Merged: {merged_path} ({merged_path.stat().st_size//(1024*1024)}MB)")
    
    # Step 5: Karaoke subs (only for shorts — long videos take too long with whisper)
    subs_path = None
    if video_type == "short":
        try:
            from karaoke_subs import generate_karaoke_subs
            subs_path = generate_karaoke_subs(
                audio_path=audio_path,
                script_text="",
                output_dir=work_dir,
                video_type=video_type,
                style="word_highlight",
                use_whisper=True
            )
            log(f"  Subs: {subs_path}")
        except Exception as e:
            log(f"  Subs skipped: {e}")
    else:
        log(f"  Subs: skipped for long video (whisper too slow)")
    
    # Step 6: Overlay subs
    pre_logo = merged_path
    if subs_path and Path(str(subs_path)).exists():
        subs_video = work_dir / "with_subs.mp4"
        if not subs_video.exists() or subs_video.stat().st_size == 0:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(merged_path),
                "-vf", f"ass={subs_path}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                str(subs_video)
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if subs_video.exists() and subs_video.stat().st_size > 0:
                pre_logo = subs_video
                log(f"  Subs added: {subs_video}")
            else:
                log(f"  Subs overlay failed: {r.stderr[-150:]}")
    
    # Step 7: Add logo
    final_path = work_dir / f"{vid_id}_rhino_final.mp4"
    if not final_path.exists() or final_path.stat().st_size == 0:
        if LOGO.exists():
            logo_scale = "120:120" if video_type == "short" else "150:150"
            overlay_pos = "W-w-20:20"
            cmd = [
                "ffmpeg", "-y",
                "-i", str(pre_logo),
                "-i", str(LOGO),
                "-filter_complex",
                f"[1:v]scale={logo_scale}[logo];[0:v][logo]overlay={overlay_pos}[v]",
                "-map", "[v]", "-map", "0:a",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                str(final_path)
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if not final_path.exists() or final_path.stat().st_size == 0:
                log(f"  Logo failed, using pre_logo: {r.stderr[-150:]}")
                shutil.copy2(str(pre_logo), str(final_path))
        else:
            shutil.copy2(str(pre_logo), str(final_path))
    
    if final_path.exists() and final_path.stat().st_size > 0:
        log(f"  🎬 Final: {final_path} ({final_path.stat().st_size//(1024*1024):.1f}MB)")
        return final_path
    return None


def get_youtube_client():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    d = json.loads(YT_CREDS.read_text())
    creds = Credentials(
        token=d["access_token"], refresh_token=d["refresh_token"],
        token_uri=d["token_uri"], client_id=d["client_id"],
        client_secret=d["client_secret"], scopes=d.get("scope","").split()
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        d["access_token"] = creds.token
        YT_CREDS.write_text(json.dumps(d, indent=2))
    return build("youtube", "v3", credentials=creds)


def upload_to_youtube(yt, video: dict, final_path: Path) -> str | None:
    from googleapiclient.http import MediaFileUpload
    
    title = video["title"]
    if video["type"] == "short" and "#Shorts" not in title:
        title += " #Shorts"
    
    tags = ["crypto", "trading", "bitcoin", "finanzas", "criptomonedas", "bittrader"]
    if video["type"] == "short":
        tags += ["shorts", "cryptoshorts"]
        cat_id = "28"
    else:
        tags += ["educacion financiera", "NAS100"]
        cat_id = "27"
    
    body = {
        "snippet": {
            "title": title,
            "description": "🦏 BitTrader — Análisis de criptomonedas y trading\n\n#crypto #trading #bitcoin #bittrader",
            "tags": tags,
            "categoryId": cat_id,
        },
        "status": {
            "privacyStatus": "private",
            "publishAt": video["publish_at"],
            "selfDeclaredMadeForKids": False,
        }
    }
    
    media = MediaFileUpload(str(final_path), mimetype="video/mp4", resumable=True, chunksize=5*1024*1024)
    
    try:
        req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
        response = None
        while response is None:
            status, response = req.next_chunk()
            if status:
                log(f"    Upload: {int(status.progress()*100)}%")
        return response.get("id")
    except Exception as e:
        log(f"  Upload error: {e}")
        return None


def phase3_assemble_and_upload():
    """Assemble + upload all videos."""
    log("🎬 PHASE 3: Assembling and uploading videos...")
    state = load_state()
    vid_results = state.get("videos", {})
    new_ids = {}
    
    try:
        yt = get_youtube_client()
        log("✅ YouTube connected")
    except Exception as e:
        log(f"❌ YouTube failed: {e}")
        yt = None
    
    for video in VIDEOS:
        vid_id = video["id"]
        
        # Skip if already done
        current_status = vid_results.get(vid_id, {}).get("status", "")
        if current_status == "done":
            log(f"⏭️  {vid_id}: already done")
            new_ids[video["title"]] = vid_results[vid_id].get("new_yt_id")
            continue
        if current_status == "upload_failed":
            log(f"⚠️  {vid_id}: previous upload failed, retrying...")
        
        log(f"\n{'='*50}")
        log(f"📹 {vid_id}: {video['title']}")
        log(f"{'='*50}")
        
        # Check if final video already exists
        final_path = RHINO_OUT / vid_id / f"{vid_id}_rhino_final.mp4"
        
        if not final_path.exists() or final_path.stat().st_size == 0:
            final_path = assemble_one_video(video, state)
        else:
            log(f"  Final video exists: {final_path}")
        
        if not final_path:
            log(f"  ❌ Assembly failed for {vid_id}")
            vid_results[vid_id] = {"status": "assembly_failed"}
            state["videos"] = vid_results
            save_state(state)
            continue
        
        vid_results[vid_id] = {"status": "assembled", "path": str(final_path)}
        state["videos"] = vid_results
        save_state(state)
        
        if yt is None:
            log(f"  ⚠️  No YT client, skipping upload")
            continue
        
        # Delete old
        old_id = video["yt_id_old"]
        log(f"  🗑️  Deleting old: {old_id}")
        try:
            yt.videos().delete(id=old_id).execute()
            log(f"  ✅ Deleted {old_id}")
        except Exception as e:
            log(f"  ⚠️  Delete failed (may not exist): {e}")
        
        time.sleep(3)
        
        # Upload new
        log(f"  📤 Uploading {vid_id}...")
        new_id = upload_to_youtube(yt, video, final_path)
        
        if new_id:
            log(f"  ✅ Uploaded! New ID: {new_id}")
            new_ids[video["title"]] = new_id
            vid_results[vid_id] = {
                "status": "done",
                "new_yt_id": new_id,
                "old_yt_id": old_id,
                "publish_at": video["publish_at"],
            }
        else:
            vid_results[vid_id]["status"] = "upload_failed"
        
        state["videos"] = vid_results
        save_state(state)
    
    # Save new IDs
    ids_path = DATA_DIR / "youtube_v3_ids.json"
    existing = json.loads(ids_path.read_text()) if ids_path.exists() else {}
    for vid in VIDEOS:
        if vid["title"] in new_ids and new_ids[vid["title"]]:
            existing[vid["title"]] = new_ids[vid["title"]]
    ids_path.write_text(json.dumps(existing, indent=2))
    
    # Summary
    done = [k for k,v in vid_results.items() if v.get("status") == "done"]
    failed = [k for k,v in vid_results.items() if "failed" in v.get("status","")]
    log(f"\n✅ Phase 3 complete: {len(done)}/{len(VIDEOS)} done, {len(failed)} failed")
    
    return new_ids, vid_results

# ══════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════

def save_final_report(new_ids: dict, results: dict):
    done_data = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "new_ids": new_ids,
        "results": results,
    }
    DONE_FILE.write_text(json.dumps(done_data, indent=2, default=str))
    log(f"\n🏁 Done signal: {DONE_FILE}")

# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    log(f"🦏 Rhino Pipeline v2 — Phase: {phase}")
    
    if phase in ("phase1", "1", "all"):
        phase1_submit_clips()
    
    if phase in ("phase2", "2", "all"):
        phase2_poll_and_download()
    
    if phase in ("phase3", "3", "all"):
        new_ids, results = phase3_assemble_and_upload()
        save_final_report(new_ids, results)
    
    log("🦏 Pipeline done.")

if __name__ == "__main__":
    main()
