#!/usr/bin/env python3
"""
🦏 BitTrader Rhino Rebuild Pipeline
Genera 8 videos con estilo rinoceronte antropomórfico 3D + Ken Burns.

Orden de prioridad (por fecha publicación):
1. PENGU (Mar 10, 14:00 UTC)
2. AKT (Mar 10, 16:00 UTC)
3. 90% traders (Mar 10, 19:00 UTC)
4. PI coin (Mar 11, 01:00 UTC)
5. ZEC (Mar 12, 14:00 UTC)
6. Claude trades (Mar 12, 16:00 UTC)
7. $0 fondeada (Mar 13, 16:00 UTC)
8. Bot noticias (Mar 13, 14:00 UTC)
"""

import json
import sys
import time
import subprocess
import requests
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta

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

# ── Minimax API ────────────────────────────────────────────────────────────
MINIMAX_KEY = json.loads((BITTRADER / "keys/minimax.json").read_text())["minimax_api_key"]
SUBMIT_URL  = "https://api.minimax.io/v1/video_generation"
POLL_URL    = "https://api.minimax.io/v1/query/video_generation"
FILES_URL   = "https://api.minimax.io/v1/files/retrieve"

HEADERS = {
    "Authorization": f"Bearer {MINIMAX_KEY}",
    "Content-Type": "application/json"
}

# ── Log helpers ────────────────────────────────────────────────────────────
def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ══════════════════════════════════════════════════════════════════════════
# VIDEO DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════

VIDEOS = [
    # ─── SHORTS ───────────────────────────────────────────────────────────
    {
        "id": "pengu",
        "title": "Por qué PENGU está subiendo mientras BTC cae",
        "type": "short",
        "audio_source": OUTPUT_BASE / "short_1772925869_724/short_1772925869_724_v4_final.mp4",
        "audio_is_video": True,
        "audio_duration": 47.112,
        "yt_id_old": "gm4_LB4eFj0",
        "publish_at": "2026-03-10T14:00:00.000Z",
        "prompts": [
            "Anthropomorphic rhinoceros looking puzzled at a small cute penguin standing on a rising green price chart, confused but intrigued expression, dark crypto trading room, hyper-realistic 3D render, 9:16 vertical",
            "Anthropomorphic rhinoceros in NFT gallery surrounded by digital penguin art pieces glowing with value tags, impressed expression, futuristic space, hyper-realistic 3D render, 9:16 vertical",
            "Anthropomorphic rhinoceros scrolling on phone showing social media memecoin hype with laughing face emoji, skeptical but curious, neon background, hyper-realistic 3D render, 9:16 vertical",
        ],
    },
    {
        "id": "akt",
        "title": "AKT explota mientras BTC cae un 1.4%",
        "type": "short",
        "audio_source": OUTPUT_BASE / "short_1772925855_875/short_1772925855_875_final_edge.mp3",
        "audio_is_video": False,
        "audio_duration": 48.576,
        "yt_id_old": "DGOvGLZburk",
        "publish_at": "2026-03-10T16:00:00.000Z",
        "prompts": [
            "Anthropomorphic rhinoceros pointing at a glowing AKT coin logo rising against falling BTC chart, excited expression, dark trading room with red and green lights, hyper-realistic 3D render, 9:16 vertical",
            "Anthropomorphic rhinoceros at server rack wall representing decentralized GPU compute, amazed expression, futuristic data center neon blue, hyper-realistic 3D render, 9:16 vertical",
            "Anthropomorphic rhinoceros holding a smartphone showing AKT price pumping with green candles, confident smirk, dark moody background, hyper-realistic 3D render, 9:16 vertical",
        ],
    },
    {
        "id": "traders90",
        "title": "El 90% de traders pierde por esto",
        "type": "short",
        "audio_source": OUTPUT_BASE / "short_1772925884_383/short_1772925884_383_v4_final.mp4",
        "audio_is_video": True,
        "audio_duration": 46.632,
        "yt_id_old": "XazgVqCkGCc",
        "publish_at": "2026-03-10T19:00:00.000Z",
        "prompts": [
            "Anthropomorphic rhinoceros with overconfident smirk counting money after a win, surrounded by warning signs being ignored, bright lights, hyper-realistic 3D render, 9:16 vertical",
            "Anthropomorphic rhinoceros sitting devastated at trading desk with red portfolio screen, head in hands, dramatic dark lighting, hyper-realistic 3D render, 9:16 vertical",
            "Anthropomorphic rhinoceros calmly studying a trading journal with a pen, disciplined focused expression, warm desk lamp, hyper-realistic 3D render, 9:16 vertical",
        ],
    },
    {
        "id": "pi",
        "title": "PI coin +13%: ¿trampa o despegue real?",
        "type": "short",
        "audio_source": OUTPUT_BASE / "short_1772925672_305/short_1772925672_305_v3_final.mp4",
        "audio_is_video": True,
        "audio_duration": 48.144,
        "yt_id_old": "7bGQROO-Y9o",
        "publish_at": "2026-03-11T01:00:00.000Z",
        "prompts": [
            "Anthropomorphic rhinoceros holding a cracked golden PI coin looking at it suspiciously, magnifying glass, dark background with question marks floating, hyper-realistic 3D render, 9:16 vertical",
            "Anthropomorphic rhinoceros looking at phone showing PI Network price +13%, eyebrow raised skeptically, red and green chart behind, hyper-realistic 3D render, 9:16 vertical",
            "Anthropomorphic rhinoceros presenting a bear trap disguised as green chart, warning gesture, dramatic cinematic lighting, hyper-realistic 3D render, 9:16 vertical",
        ],
    },
    {
        "id": "zec",
        "title": "ZEC cae 7%: ¿el fin de las privacy coins?",
        "type": "short",
        "audio_source": OUTPUT_BASE / "short_1772925742_739/short_1772925742_739_v3_final.mp4",
        "audio_is_video": True,
        "audio_duration": 47.016,
        "yt_id_old": "rXZNuFtV8AE",
        "publish_at": "2026-03-12T14:00:00.000Z",
        "prompts": [
            "Anthropomorphic rhinoceros watching a cracked Zcash ZEC coin falling through the floor, concerned expression, dark trading room, hyper-realistic 3D render, 9:16 vertical",
            "Anthropomorphic rhinoceros in a dark hood representing privacy, looking over shoulder nervously at government surveillance screens, hyper-realistic 3D render, 9:16 vertical",
            "Anthropomorphic rhinoceros gesturing at a graveyard of crypto coins with ZEC highlighted, philosophical expression, cinematic lighting, hyper-realistic 3D render, 9:16 vertical",
        ],
    },
    # ─── LONGS ────────────────────────────────────────────────────────────
    {
        "id": "claude_trades",
        "title": "Le di mis trades a Claude y los resultados me sorprendieron",
        "type": "long",
        "audio_source": OUTPUT_BASE / "long_1772925728_400/long_1772925728_400_edge_clean.mp3",
        "audio_is_video": False,
        "audio_duration": 250.320,
        "yt_id_old": "RhZo-Lhs90A",
        "publish_at": "2026-03-12T16:00:00.000Z",
        "prompts": [
            "Anthropomorphic rhinoceros frustrated at trading desk with losing red charts, 3 months of losses visible, dramatic dark office, hyper-realistic 3D render, 16:9",
            "Anthropomorphic rhinoceros typing trading history into an AI chat interface (Claude), hopeful expression, laptop glow illuminating face, dark room, hyper-realistic 3D render, 16:9",
            "Anthropomorphic rhinoceros shocked reading AI analysis report showing patterns in losses, jaw dropped, papers flying, dramatic light, hyper-realistic 3D render, 16:9",
            "Anthropomorphic rhinoceros making disciplined trading changes, writing new rules in journal, focused determined expression, clean modern desk, hyper-realistic 3D render, 16:9",
            "Anthropomorphic rhinoceros celebrating first green month, portfolio showing profit, triumphant fist pump, golden light, hyper-realistic 3D render, 16:9",
        ],
    },
    {
        "id": "fondeada",
        "title": "De $0 a cuenta fondeada en 30 días",
        "type": "long",
        "audio_source": OUTPUT_BASE / "long_1772925781_782/long_1772925781_782_v3_final.mp4",
        "audio_is_video": True,
        "audio_duration": 254.928,
        "yt_id_old": "I7dK9aGl9mk",
        "publish_at": "2026-03-13T16:00:00.000Z",
        "prompts": [
            "Anthropomorphic rhinoceros at starting line with $0 counter, determined expression, dramatic stadium lighting, challenge banner in background, hyper-realistic 3D render, 16:9",
            "Anthropomorphic rhinoceros following strict trading rules checklist day by day, 30-day calendar on wall, disciplined focused expression, hyper-realistic 3D render, 16:9",
            "Anthropomorphic rhinoceros celebrating passing FTMO challenge, funded account certificate, confetti, triumphant pose, hyper-realistic 3D render, 16:9",
            "Anthropomorphic rhinoceros managing funded account professionally, multiple monitors, calm confident expression, professional trading room, hyper-realistic 3D render, 16:9",
            "Anthropomorphic rhinoceros looking at camera motivationally, holding FTMO logo sign, inspiring expression, studio lighting, hyper-realistic 3D render, 16:9",
        ],
    },
    {
        "id": "bot_noticias",
        "title": "El bot que lee noticias y ejecuta tus trades solo",
        "type": "long",
        "audio_source": OUTPUT_BASE / "long_1772925821_433/long_1772925821_433_v3_final.mp4",
        "audio_is_video": True,
        "audio_duration": 272.712,
        "yt_id_old": "zSQgH3aRxxY",
        "publish_at": "2026-03-13T14:00:00.000Z",
        "prompts": [
            "Anthropomorphic rhinoceros sleeping peacefully while robot arms trade on screens behind them, autonomous AI bots working, dark room, hyper-realistic 3D render, 16:9",
            "Anthropomorphic rhinoceros watching news headlines transform into buy sell orders automatically, AI processing visual effect, impressed expression, hyper-realistic 3D render, 16:9",
            "Split screen: anthropomorphic rhinoceros on left slowly reading news manually, robotic rhino on right executing 100 trades instantly, speed comparison, hyper-realistic 3D render, 16:9",
            "Anthropomorphic rhinoceros presenting a flowchart News to AI Analysis to Trade Execution, professor mode, clean background, hyper-realistic 3D render, 16:9",
            "Anthropomorphic rhinoceros checking phone at beach showing autonomous bot profits, relaxed successful lifestyle, golden light, hyper-realistic 3D render, 16:9",
        ],
    },
]

# ══════════════════════════════════════════════════════════════════════════
# MINIMAX API FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def submit_clip(prompt: str, model: str = "T2V-01", ratio: str = "9:16") -> str | None:
    """Submit a video generation request. Returns task_id."""
    payload = {
        "model": model,
        "prompt": prompt,
        "prompt_optimizer": True,
    }
    # Note: ratio support depends on model - we use prompt to guide orientation
    try:
        resp = requests.post(SUBMIT_URL, headers=HEADERS, json=payload, timeout=30)
        data = resp.json()
        log(f"  Submit response: {data}")
        if data.get("task_id"):
            return data["task_id"]
        elif data.get("base_resp", {}).get("status_code") == 0:
            return data.get("task_id")
        else:
            log(f"  ERROR submit: {data}")
            return None
    except Exception as e:
        log(f"  Exception submit: {e}")
        return None


def poll_clip(task_id: str, max_secs: int = 600) -> str | None:
    """Poll until done. Returns file_id."""
    elapsed = 0
    while elapsed < max_secs:
        try:
            resp = requests.get(
                POLL_URL,
                headers=HEADERS,
                params={"task_id": task_id},
                timeout=15
            )
            data = resp.json()
            status = data.get("status", "")
            log(f"  Poll [{elapsed}s]: status={status}")
            if status == "Success":
                file_id = data.get("file_id")
                return file_id
            elif status in ("Fail", "Failed"):
                log(f"  FAILED: {data}")
                return None
            time.sleep(15)
            elapsed += 15
        except Exception as e:
            log(f"  Poll exception: {e}")
            time.sleep(15)
            elapsed += 15
    log(f"  TIMEOUT after {max_secs}s")
    return None


def download_clip(file_id: str, out_path: Path) -> bool:
    """Download generated clip by file_id."""
    try:
        resp = requests.get(
            FILES_URL,
            headers=HEADERS,
            params={"file_id": file_id},
            timeout=15
        )
        data = resp.json()
        log(f"  Files response keys: {list(data.keys())}")
        
        # Get download URL
        file_info = data.get("file", {})
        download_url = file_info.get("download_url") or data.get("download_url")
        
        if not download_url:
            log(f"  No download_url in: {data}")
            return False
        
        # Download the actual file
        dl = requests.get(download_url, timeout=120, stream=True)
        with open(out_path, "wb") as f:
            for chunk in dl.iter_content(chunk_size=8192):
                f.write(chunk)
        log(f"  Downloaded: {out_path} ({out_path.stat().st_size // 1024}KB)")
        return True
    except Exception as e:
        log(f"  Download exception: {e}")
        return False


def generate_clips(video: dict, clips_dir: Path) -> list[Path]:
    """Generate all rhino clips for a video. Returns list of downloaded clip paths."""
    prompts = video["prompts"]
    clips = []
    
    for i, prompt in enumerate(prompts):
        clip_path = clips_dir / f"rhino_clip_{i+1}.mp4"
        
        if clip_path.exists() and clip_path.stat().st_size > 100_000:
            log(f"  Clip {i+1} already exists, skipping: {clip_path}")
            clips.append(clip_path)
            continue
        
        log(f"  Submitting clip {i+1}/{len(prompts)}: {prompt[:60]}...")
        task_id = submit_clip(prompt)
        
        if not task_id:
            log(f"  RETRY clip {i+1}...")
            time.sleep(10)
            task_id = submit_clip(prompt)
        
        if not task_id:
            log(f"  FAILED clip {i+1}, will use fallback")
            clips.append(None)
            # Still wait rate limit
            if i < len(prompts) - 1:
                log(f"  Rate limit wait 35s...")
                time.sleep(35)
            continue
        
        log(f"  Task {task_id} submitted, polling...")
        file_id = poll_clip(task_id)
        
        if not file_id:
            log(f"  Poll failed for clip {i+1}")
            clips.append(None)
        else:
            ok = download_clip(file_id, clip_path)
            if ok:
                clips.append(clip_path)
            else:
                clips.append(None)
        
        # Rate limit between submissions
        if i < len(prompts) - 1:
            log(f"  Rate limit: waiting 35s before next submission...")
            time.sleep(35)
    
    return clips


# ══════════════════════════════════════════════════════════════════════════
# EXTRACT FRAME FROM CLIP
# ══════════════════════════════════════════════════════════════════════════

def extract_frame(clip_path: Path, out_path: Path, timestamp: float = 0.5) -> bool:
    """Extract best frame from clip."""
    cmd = [
        "ffmpeg", "-y", "-ss", str(timestamp), "-i", str(clip_path),
        "-vframes", "1", "-q:v", "2", str(out_path)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        log(f"  Frame extracted: {out_path}")
        return True
    log(f"  Frame extract failed: {r.stderr[-200:]}")
    return False


# ══════════════════════════════════════════════════════════════════════════
# KEN BURNS EFFECT
# ══════════════════════════════════════════════════════════════════════════

def apply_ken_burns(frame_path: Path, out_path: Path, duration: float, 
                    video_type: str, direction: int = 0) -> bool:
    """
    Apply Ken Burns zoom+pan effect to a frame.
    direction: 0=zoom-center, 1=pan-left-right, 2=zoom-top
    """
    dur_frames = int(duration * 30)
    
    if video_type == "short":
        w, h = 1080, 1920
        zoom_speed = "min(zoom+0.0008,1.4)"
        if direction == 0:
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"
        elif direction == 1:
            x_expr = "iw/2-(iw/zoom/2)+t*5"
            y_expr = "ih/2-(ih/zoom/2)"
        else:
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "0"
    else:
        w, h = 1920, 1080
        zoom_speed = "min(zoom+0.0005,1.3)"
        if direction == 0:
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"
        elif direction == 1:
            x_expr = "iw/2-(iw/zoom/2)+t*3"
            y_expr = "ih/2-(ih/zoom/2)"
        else:
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "0"
    
    vf = (
        f"scale={w*2}:{h*2},"
        f"zoompan=z='{zoom_speed}':d={dur_frames}"
        f":x='{x_expr}':y='{y_expr}':s={w}x{h},"
        f"setsar=1"
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(frame_path),
        "-vf", vf,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-r", "30", "-an",
        str(out_path)
    ]
    
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if out_path.exists() and out_path.stat().st_size > 0:
        log(f"  Ken Burns clip: {out_path} ({out_path.stat().st_size//1024}KB)")
        return True
    log(f"  Ken Burns failed: {r.stderr[-300:]}")
    return False


# ══════════════════════════════════════════════════════════════════════════
# AUDIO EXTRACTION
# ══════════════════════════════════════════════════════════════════════════

def extract_audio(source_path: Path, out_path: Path) -> bool:
    """Extract audio from video or copy mp3."""
    if str(source_path).endswith(".mp3"):
        shutil.copy2(str(source_path), str(out_path))
        return True
    
    cmd = [
        "ffmpeg", "-y", "-i", str(source_path),
        "-vn", "-ar", "44100", "-ac", "1", "-b:a", "128k",
        str(out_path)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if out_path.exists() and out_path.stat().st_size > 0:
        log(f"  Audio extracted: {out_path}")
        return True
    log(f"  Audio extract failed: {r.stderr[-300:]}")
    return False


# ══════════════════════════════════════════════════════════════════════════
# ADD LOGO OVERLAY
# ══════════════════════════════════════════════════════════════════════════

def add_logo(video_path: Path, out_path: Path, video_type: str) -> bool:
    """Add BitTrader logo overlay to video."""
    if not LOGO.exists():
        log(f"  Logo not found: {LOGO}")
        shutil.copy2(str(video_path), str(out_path))
        return True
    
    if video_type == "short":
        logo_scale = "120:120"
        overlay_pos = "W-w-20:20"
    else:
        logo_scale = "150:150"
        overlay_pos = "W-w-30:30"
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(LOGO),
        "-filter_complex",
        f"[1:v]scale={logo_scale}[logo];[0:v][logo]overlay={overlay_pos}[v]",
        "-map", "[v]", "-map", "0:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        str(out_path)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if out_path.exists() and out_path.stat().st_size > 0:
        log(f"  Logo added: {out_path}")
        return True
    log(f"  Logo failed: {r.stderr[-300:]}")
    # Fallback: just copy
    shutil.copy2(str(video_path), str(out_path))
    return True


# ══════════════════════════════════════════════════════════════════════════
# ASSEMBLE VIDEO
# ══════════════════════════════════════════════════════════════════════════

def assemble_video(video: dict, clips: list[Path], work_dir: Path) -> Path | None:
    """
    Full assembly: Ken Burns on frames → concat → add audio → add subs → add logo.
    Returns final video path.
    """
    num_clips = len(video["prompts"])
    audio_dur = video["audio_duration"]
    dur_per_clip = audio_dur / num_clips
    video_type = video["type"]
    
    log(f"  Assembly: {num_clips} clips, {dur_per_clip:.1f}s each, total={audio_dur}s")
    
    # ── 1. Extract audio ──────────────────────────────────────────────────
    audio_path = work_dir / "audio.mp3"
    if not extract_audio(Path(str(video["audio_source"])), audio_path):
        log("  ERROR: could not extract audio")
        return None
    
    # ── 2. For each clip: extract frame + apply Ken Burns ─────────────────
    kb_clips = []
    prev_valid_clip = None
    
    for i, clip_path in enumerate(clips):
        # Fallback if clip failed
        if clip_path is None:
            if prev_valid_clip:
                log(f"  Clip {i+1} failed, reusing clip {i}")
                clip_path = prev_valid_clip
            else:
                log(f"  Clip {i+1} failed and no fallback available, skipping")
                continue
        
        if not clip_path.exists():
            log(f"  Clip path doesn't exist: {clip_path}")
            continue
        
        prev_valid_clip = clip_path
        
        # Extract frame
        frame_path = work_dir / f"frame_{i+1}.png"
        if not extract_frame(clip_path, frame_path, 0.5):
            # Try different timestamp
            extract_frame(clip_path, frame_path, 1.0)
        
        if not frame_path.exists() or frame_path.stat().st_size == 0:
            log(f"  ERROR: frame {i+1} could not be extracted")
            continue
        
        # Apply Ken Burns - alternating directions
        direction = i % 3
        kb_path = work_dir / f"kb_{i+1}.mp4"
        ok = apply_ken_burns(frame_path, kb_path, dur_per_clip, video_type, direction)
        if ok:
            kb_clips.append(kb_path)
        else:
            log(f"  Ken Burns failed for clip {i+1}")
    
    if not kb_clips:
        log("  ERROR: no Ken Burns clips generated")
        return None
    
    # ── 3. Repeat clips to fill full audio duration ────────────────────────
    # Ken Burns clips might not perfectly fill audio — we loop them
    total_kb_dur = len(kb_clips) * dur_per_clip
    
    # If we have fewer clips than prompts (some failed), we need to fill
    # by looping. For now, what we have should be close enough.
    
    # ── 4. Concatenate Ken Burns clips ─────────────────────────────────────
    concat_list = work_dir / "concat.txt"
    with open(concat_list, "w") as f:
        for kb in kb_clips:
            f.write(f"file '{kb}'\n")
    
    concat_path = work_dir / "concat.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        str(concat_path)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if not concat_path.exists():
        log(f"  Concat failed: {r.stderr[-300:]}")
        return None
    
    # ── 5. Merge video + audio (trim/pad to match audio duration) ─────────
    merged_path = work_dir / "merged.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", str(concat_path),  # loop video if shorter
        "-i", str(audio_path),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",  # trim to shorter of the two (audio)
        str(merged_path)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if not merged_path.exists():
        log(f"  Merge failed: {r.stderr[-300:]}")
        return None
    log(f"  Merged: {merged_path}")
    
    # ── 6. Generate karaoke subtitles ──────────────────────────────────────
    subs_path = None
    try:
        from karaoke_subs import generate_karaoke_subs
        subs_path = generate_karaoke_subs(
            audio_path=audio_path,
            script_text="",  # empty = use whisper only
            output_dir=work_dir,
            video_type=video_type,
            style="word_highlight",
            use_whisper=True
        )
        log(f"  Subs generated: {subs_path}")
    except Exception as e:
        log(f"  Subs failed (will skip): {e}")
        subs_path = None
    
    # ── 7. Add subtitles ──────────────────────────────────────────────────
    if subs_path and Path(subs_path).exists():
        with_subs_path = work_dir / "with_subs.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(merged_path),
            "-vf", f"ass={subs_path}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            str(with_subs_path)
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if with_subs_path.exists() and with_subs_path.stat().st_size > 0:
            merged_path = with_subs_path
            log(f"  Subs added: {with_subs_path}")
        else:
            log(f"  Subs overlay failed, continuing without: {r.stderr[-200:]}")
    
    # ── 8. Add BitTrader logo ──────────────────────────────────────────────
    final_path = work_dir / f"{video['id']}_rhino_final.mp4"
    add_logo(merged_path, final_path, video_type)
    
    if final_path.exists() and final_path.stat().st_size > 0:
        log(f"  FINAL VIDEO: {final_path} ({final_path.stat().st_size // (1024*1024):.1f}MB)")
        return final_path
    
    log("  Logo step failed, using merged")
    shutil.copy2(str(merged_path), str(final_path))
    return final_path if final_path.exists() else None


# ══════════════════════════════════════════════════════════════════════════
# YOUTUBE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def get_youtube_client():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    d = json.loads(YT_CREDS.read_text())
    creds = Credentials(
        token=d["access_token"],
        refresh_token=d["refresh_token"],
        token_uri=d["token_uri"],
        client_id=d["client_id"],
        client_secret=d["client_secret"],
        scopes=d.get("scope", "").split()
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        d["access_token"] = creds.token
        YT_CREDS.write_text(json.dumps(d, indent=2))
    return build("youtube", "v3", credentials=creds)


def delete_youtube_video(yt, video_id: str) -> bool:
    """Delete a YouTube video."""
    try:
        yt.videos().delete(id=video_id).execute()
        log(f"  Deleted YT video: {video_id}")
        return True
    except Exception as e:
        log(f"  Delete failed for {video_id}: {e}")
        return False


def upload_youtube_video(yt, video: dict, final_path: Path) -> str | None:
    """Upload video to YouTube. Returns new video_id."""
    from googleapiclient.http import MediaFileUpload
    
    is_short = video["type"] == "short"
    
    # Build title — add #Shorts for short videos
    title = video["title"]
    if is_short and "#Shorts" not in title:
        title = title + " #Shorts"
    
    description = (
        f"🦏 BitTrader — Análisis de criptomonedas y trading\n\n"
        f"📊 Seguime para más contenido de trading y crypto.\n"
        f"💬 Comentá qué coin querés que analice.\n\n"
        f"#crypto #trading #bitcoin #criptomonedas #bittrader"
    )
    
    tags = [
        "crypto", "trading", "bitcoin", "finanzas", "criptomonedas",
        "bittrader", "inversion", "blockchain"
    ]
    if is_short:
        tags += ["shorts", "cryptoshorts"]
        category_id = "28"
    else:
        tags += ["educacion financiera", "NAS100", "futuros"]
        category_id = "27"
    
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": "private",
            "publishAt": video["publish_at"],
            "selfDeclaredMadeForKids": False,
        }
    }
    
    media = MediaFileUpload(
        str(final_path),
        mimetype="video/mp4",
        resumable=True,
        chunksize=5 * 1024 * 1024  # 5MB chunks
    )
    
    try:
        request = yt.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media
        )
        
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                log(f"  Upload progress: {pct}%")
        
        new_id = response.get("id")
        log(f"  Uploaded! New YT ID: {new_id}")
        return new_id
    except Exception as e:
        log(f"  Upload error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def main():
    log("🦏 BitTrader Rhino Rebuild Starting")
    log(f"   Videos to process: {len(VIDEOS)}")
    
    # Load or init results tracker
    results_path = DATA_DIR / "rhino_rebuild_results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
    else:
        results = {}
    
    new_ids = {}
    
    try:
        yt = get_youtube_client()
        log("✅ YouTube client connected")
    except Exception as e:
        log(f"❌ YouTube connection failed: {e}")
        yt = None
    
    for video in VIDEOS:
        vid_id = video["id"]
        title = video["title"]
        
        # Check if already done
        if vid_id in results and results[vid_id].get("status") == "done":
            log(f"\n⏭️  [{vid_id}] Already done, skipping")
            new_ids[title] = results[vid_id].get("new_yt_id")
            continue
        
        log(f"\n{'='*60}")
        log(f"🎬 Processing: [{vid_id}] {title}")
        log(f"   Type: {video['type']} | Duration: {video['audio_duration']}s")
        log(f"   Clips to generate: {len(video['prompts'])}")
        log(f"{'='*60}")
        
        results[vid_id] = {"status": "in_progress", "title": title}
        results_path.write_text(json.dumps(results, indent=2, default=str))
        
        # ── Work directory ─────────────────────────────────────────────────
        work_dir = RHINO_OUT / vid_id
        work_dir.mkdir(parents=True, exist_ok=True)
        clips_dir = work_dir / "clips"
        clips_dir.mkdir(exist_ok=True)
        
        # ── Step 1: Generate Hailuo clips ─────────────────────────────────
        log(f"\n  [1/5] Generating {len(video['prompts'])} rhino clips via Hailuo...")
        clips = generate_clips(video, clips_dir)
        
        valid_clips = [c for c in clips if c is not None]
        log(f"  Got {len(valid_clips)}/{len(video['prompts'])} valid clips")
        
        if not valid_clips:
            log(f"  ❌ All clips failed for {vid_id}")
            results[vid_id] = {"status": "failed", "reason": "all_clips_failed", "title": title}
            results_path.write_text(json.dumps(results, indent=2, default=str))
            continue
        
        # ── Step 2: Assemble video ─────────────────────────────────────────
        log(f"\n  [2/5] Assembling video...")
        final_path = assemble_video(video, clips, work_dir)
        
        if not final_path:
            log(f"  ❌ Assembly failed for {vid_id}")
            results[vid_id] = {"status": "failed", "reason": "assembly_failed", "title": title}
            results_path.write_text(json.dumps(results, indent=2, default=str))
            continue
        
        results[vid_id]["final_path"] = str(final_path)
        results_path.write_text(json.dumps(results, indent=2, default=str))
        
        if yt is None:
            log(f"  ⚠️  No YouTube client, skipping upload")
            results[vid_id]["status"] = "video_ready_no_upload"
            results_path.write_text(json.dumps(results, indent=2, default=str))
            continue
        
        # ── Step 3: Delete old video ────────────────────────────────────────
        log(f"\n  [3/5] Deleting old YT video: {video['yt_id_old']}")
        delete_ok = delete_youtube_video(yt, video["yt_id_old"])
        results[vid_id]["deleted_old"] = delete_ok
        results_path.write_text(json.dumps(results, indent=2, default=str))
        
        # Wait a bit after deletion
        time.sleep(5)
        
        # ── Step 4: Upload new video ────────────────────────────────────────
        log(f"\n  [4/5] Uploading new video...")
        new_yt_id = upload_youtube_video(yt, video, final_path)
        
        if new_yt_id:
            new_ids[title] = new_yt_id
            results[vid_id].update({
                "status": "done",
                "new_yt_id": new_yt_id,
                "old_yt_id": video["yt_id_old"],
                "publish_at": video["publish_at"],
                "clips_generated": len(valid_clips),
                "clips_total": len(video["prompts"]),
            })
            log(f"\n  ✅ {vid_id} DONE! New ID: {new_yt_id}")
        else:
            results[vid_id].update({
                "status": "upload_failed",
                "title": title,
            })
            log(f"\n  ❌ Upload failed for {vid_id}")
        
        results_path.write_text(json.dumps(results, indent=2, default=str))
        
        # ── Step 5: Brief pause between videos ────────────────────────────
        log(f"\n  [5/5] Pausing before next video...")
        time.sleep(10)
    
    # ══ Save final IDs ══════════════════════════════════════════════════════
    log(f"\n{'='*60}")
    log(f"🦏 Rhino Rebuild Complete!")
    log(f"{'='*60}")
    
    # Update youtube_v3_ids.json
    ids_path = DATA_DIR / "youtube_v3_ids.json"
    if ids_path.exists():
        existing = json.loads(ids_path.read_text())
    else:
        existing = {}
    
    # Update with new IDs (keep old titles for ones we didn't process)
    for vid in VIDEOS:
        title = vid["title"]
        if title in new_ids and new_ids[title]:
            existing[title] = new_ids[title]
    
    ids_path.write_text(json.dumps(existing, indent=2))
    log(f"\nNew IDs saved to: {ids_path}")
    
    # Summary
    done = [k for k,v in results.items() if v.get("status") == "done"]
    failed = [k for k,v in results.items() if "failed" in v.get("status","")]
    
    log(f"\n📊 SUMMARY:")
    log(f"  ✅ Done: {len(done)}/{len(VIDEOS)}")
    log(f"  ❌ Failed: {len(failed)}/{len(VIDEOS)}")
    
    for vid_id, res in results.items():
        status = res.get("status","?")
        new_id = res.get("new_yt_id","—")
        clips_ok = res.get("clips_generated","?")
        clips_tot = res.get("clips_total","?")
        log(f"  [{status}] {vid_id}: new_id={new_id} clips={clips_ok}/{clips_tot}")
    
    # Write completion signal
    done_signal = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "videos_done": len(done),
        "videos_failed": len(failed),
        "new_ids": new_ids,
        "results": results,
    }
    done_path = DATA_DIR / "rhino_rebuild_done.json"
    done_path.write_text(json.dumps(done_signal, indent=2))
    log(f"\n🏁 Done signal written: {done_path}")
    
    return results


if __name__ == "__main__":
    main()
