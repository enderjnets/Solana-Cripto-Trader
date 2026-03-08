#!/usr/bin/env python3
"""
🦏 Ken Burns Producer — BitTrader
Genera videos con efecto Ken Burns usando imágenes del rinoceronte BitTrader.
Pipeline: Prompts → Imágenes (Flux.1-schnell via HuggingFace Pro) → Ken Burns (ffmpeg) → Concat → Subs → Final
"""
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

import requests

WORKSPACE   = Path("/home/enderj/.openclaw/workspace")
BITTRADER   = WORKSPACE / "bittrader"
AGENTS_DIR  = BITTRADER / "agents"
DATA_DIR    = AGENTS_DIR / "data"
OUTPUT_BASE = AGENTS_DIR / "output" / "2026-03-07-v3"
BT_LOGO     = WORKSPACE / "videos/BIBLIOTECA/bittrader_logo.png"
COIN_LOGOS  = DATA_DIR / "coin_logos"
PROGRESS    = DATA_DIR / "kenburns_progress.json"

sys.path.insert(0, str(AGENTS_DIR))
from karaoke_subs import generate_karaoke_subs

# ── Rhino character base ───────────────────────────────────────────────────
RHINO_BASE_SHORT = (
    "anthropomorphic rhinoceros character, hyper-realistic 3D render, muscular but elegant, "
    "wearing modern casual trading clothes, sitting at trading desk with multiple crypto monitors "
    "showing green candlestick charts, expressive confident face, dramatic cinematic lighting, "
    "ultra HD, photorealistic textures, dark moody background, depth of field bokeh, professional photography"
)
RHINO_BASE_LONG = (
    "anthropomorphic rhinoceros character, hyper-realistic 3D render, muscular but elegant, "
    "wearing modern casual trading clothes, sitting at trading desk with multiple crypto monitors "
    "showing green candlestick charts, expressive confident face, dramatic cinematic lighting, "
    "ultra HD, photorealistic textures, dark moody background, depth of field bokeh, widescreen cinematic"
)

# ── Ken Burns variants ─────────────────────────────────────────────────────
KB_STYLES = [
    # zoom_in
    lambda W, H: f"zoompan=z='min(zoom+0.0008,1.3)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=150:s={W}x{H}:fps=30",
    # zoom_out
    lambda W, H: f"zoompan=z='if(lte(zoom,1.0),1.3,zoom-0.0008)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=150:s={W}x{H}:fps=30",
    # pan_right
    lambda W, H: f"zoompan=z='1.25':x='if(lte(on,1),0,min(x+2,iw/5))':y='ih/2-(ih/zoom/2)':d=150:s={W}x{H}:fps=30",
    # pan_left
    lambda W, H: f"zoompan=z='1.25':x='if(lte(on,1),iw/5,max(x-2,0))':y='ih/2-(ih/zoom/2)':d=150:s={W}x{H}:fps=30",
    # pan_up
    lambda W, H: f"zoompan=z='1.25':x='iw/2-(iw/zoom/2)':y='if(lte(on,1),0,min(y+2,ih/5))':d=150:s={W}x{H}:fps=30",
    # pan_down
    lambda W, H: f"zoompan=z='1.25':x='iw/2-(iw/zoom/2)':y='if(lte(on,1),ih/5,max(y-2,0))':d=150:s={W}x{H}:fps=30",
]

# ── Video manifest ─────────────────────────────────────────────────────────
VIDEOS = [
    # SHORTS
    {
        "id": "short_1772925855_875",
        "type": "short",
        "title": "AKT explota mientras BTC cae un 1.4%",
        "topic": "AKT Akash Network GPU decentralized crypto altcoin bullish",
        "coin": "AKT",
        "audio": OUTPUT_BASE / "short_1772925855_875/short_1772925855_875_final_edge.mp3",
    },
    {
        "id": "short_1772925869_724",
        "type": "short",
        "title": "Por qué PENGU está subiendo mientras BTC cae",
        "topic": "PENGU Pudgy Penguins NFT meme coin bullish trend",
        "coin": "PENGU",
        "audio": OUTPUT_BASE / "short_1772925869_724/short_1772925869_724_final_edge.mp3",
    },
    {
        "id": "short_1772925884_383",
        "type": "short",
        "title": "El 90% de traders pierde por esto",
        "topic": "trading psychology losses discipline mistakes crypto",
        "coin": None,
        "audio": OUTPUT_BASE / "short_1772925884_383/short_1772925884_383_final_edge.mp3",
    },
    {
        "id": "short_1772925672_305",
        "type": "short",
        "title": "PI coin +13%: ¿trampa o despegue real?",
        "topic": "PI coin mainnet pump crypto altcoin analysis",
        "coin": "PI",
        "audio": OUTPUT_BASE / "short_1772925672_305/short_1772925672_305_final_edge.mp3",
    },
    {
        "id": "short_1772925742_739",
        "type": "short",
        "title": "ZEC cae 7%: ¿el fin de las privacy coins?",
        "topic": "ZEC Zcash privacy coin drop bearish regulation",
        "coin": "ZEC",
        "audio": OUTPUT_BASE / "short_1772925742_739/short_1772925742_739_edge.mp3",
    },
    # LONGS
    {
        "id": "long_1772925728_400",
        "type": "long",
        "title": "Le di mis trades a Claude y los resultados me sorprendieron",
        "topic": "Claude AI trading analysis LLM bot results improvement",
        "coin": None,
        "audio": OUTPUT_BASE / "long_1772925728_400/long_1772925728_400_edge_clean.mp3",
    },
    {
        "id": "long_1772925781_782",
        "type": "long",
        "title": "De $0 a cuenta fondeada en 30 días (plan real paso a paso)",
        "topic": "funded trading account FTMO TopStep prop firm strategy",
        "coin": None,
        "audio": OUTPUT_BASE / "long_1772925781_782/long_1772925781_782_edge.mp3",
    },
    {
        "id": "long_1772925821_433",
        "type": "long",
        "title": "El bot que lee noticias y ejecuta tus trades solo (sin errores)",
        "topic": "trading bot AI news reader auto execute crypto futures",
        "coin": None,
        "audio": OUTPUT_BASE / "long_1772925821_433/long_1772925821_433_edge.mp3",
    },
]


# ════════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════════

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


def load_progress() -> dict:
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text())
    return {}


def save_progress(prog: dict):
    PROGRESS.write_text(json.dumps(prog, indent=2, ensure_ascii=False))


def load_script_text(script_id: str) -> str:
    """Load narration script from guiones_latest.json."""
    f = DATA_DIR / "guiones_latest.json"
    if not f.exists():
        return ""
    d = json.loads(f.read_text())
    items = d.get("scripts", d.get("videos", []))
    for v in items:
        if v.get("id") == script_id:
            return v.get("script", "")
    return ""


# ════════════════════════════════════════════════════════════════════════
# IMAGE GENERATION (Flux.1-schnell via HuggingFace Pro)
# ════════════════════════════════════════════════════════════════════════

# HuggingFace API key
HF_API_KEY = json.loads((BITTRADER / "keys/minimax.json").read_text()).get("huggingface_api_key", "")
FLUX_ENDPOINT = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"

SCENE_CONTEXTS = {
    "short": [
        "sitting at trading desk watching multiple monitors green charts excited expression neon dark room",
        "holding smartphone showing crypto price spike standing city night confident",
        "pointing at holographic candlestick chart rising upward dark blue data streams triumphant",
        "looking at camera knowing smile rising chart behind dramatic lighting motivational",
        "typing on laptop showing crypto analytics focused determined modern office",
        "celebrating fist pump watching profits screen euphoric trading room",
        "studying charts on tablet coffee morning light professional focused",
        "walking through digital blockchain city holograms floating confident stride",
        "sitting in dark room single monitor red chart thoughtful analyzing serious",
        "jumping with joy phone notification showing gains outdoor city background",
    ],
    "long": [
        "standing in front of massive wall of trading screens dark room arms crossed commanding",
        "typing on keyboard laptop showing AI code trading algorithms focused expression",
        "analyzing holographic portfolio dashboard green profit metrics modern office impressed",
        "in casual clothes checking trading app on phone coffee morning light relaxed successful",
        "looking at camera knowing smile motivational dramatic lighting crypto chart rising behind",
        "in front of whiteboard presenting trading strategy professional educator confident",
        "at dual monitor setup comparing charts morning coffee professional workspace",
        "walking through futuristic digital city blockchain holograms floating around",
        "sitting across from another rhino interviewer style discussing trading strategies",
        "in dark trading room surrounded by screens analyzing late night focused intense",
    ],
}


def build_image_prompt(video: dict, scene_idx: int, n_scenes: int) -> str:
    vtype = video["type"]
    topic = video["topic"]
    base = RHINO_BASE_SHORT if vtype == "short" else RHINO_BASE_LONG
    contexts = SCENE_CONTEXTS[vtype]
    ctx = contexts[scene_idx % len(contexts)]

    # First and last scenes: hook and CTA — more impactful
    if scene_idx == 0:
        ctx = "dramatic entrance pose confident powerful trading desk background cinematic lighting"
    elif scene_idx == n_scenes - 1:
        ctx = "looking directly at camera pointing finger viewer call to action confident smile"

    prompt = f"{ctx} {topic} {base}"
    return prompt


def download_image(prompt: str, out_path: Path, width: int, height: int, seed: int) -> bool:
    """Generate image with Flux.1-schnell via HuggingFace Pro."""
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
            out_path.write_bytes(r.content)
            return True
        print(f"      ⚠️  Flux {r.status_code}: {r.text[:100]}")
        # Fallback a Pollinations
        encoded = urllib.parse.quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true&seed={seed}"
        r2 = requests.get(url, timeout=50)
        if r2.status_code == 200 and len(r2.content) > 5000:
            out_path.write_bytes(r2.content)
            return True
        return False
    except Exception as e:
        print(f"      ⚠️  Download error: {e}")
        return False


def extract_frame_fallback(video: dict, out_path: Path) -> bool:
    """Extract a frame from existing clips_v3 as fallback."""
    clips_dir = OUTPUT_BASE / video["id"] / "clips_v3"
    if not clips_dir.exists():
        clips_dir = OUTPUT_BASE / video["id"] / "clips"
    if not clips_dir.exists():
        return False
    clips = sorted(clips_dir.glob("*.mp4"))
    if not clips:
        return False
    clip = clips[0]
    r = subprocess.run(
        ["ffmpeg", "-y", "-ss", "2", "-i", str(clip),
         "-vframes", "1", "-q:v", "2", str(out_path)],
        capture_output=True
    )
    return out_path.exists()


# ════════════════════════════════════════════════════════════════════════
# KEN BURNS
# ════════════════════════════════════════════════════════════════════════

def apply_ken_burns(img_path: Path, out_path: Path, style_idx: int,
                    width: int, height: int, duration: float = 5.0) -> bool:
    """Apply Ken Burns effect to a single image."""
    frames = int(duration * 30)
    vf = KB_STYLES[style_idx % len(KB_STYLES)](width, height)
    # Pad/scale image to target resolution first
    scale_vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"{vf}"
    )
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(img_path),
        "-vf", scale_vf,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-an",
        str(out_path)
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=60)
    return r.returncode == 0 and out_path.exists()


# ════════════════════════════════════════════════════════════════════════
# COIN LOGO
# ════════════════════════════════════════════════════════════════════════

COINGECKO_ID_MAP = {
    "AKT": "akash-network", "PENGU": "pudgy-penguins", "PI": "pi-network-iou",
    "ZEC": "zcash", "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
    "JUP": "jupiter-exchange-solana", "BONK": "bonk", "WIF": "dogwifcoin",
}


def fetch_coin_logo(symbol: str) -> Path | None:
    if not symbol:
        return None
    cached = COIN_LOGOS / f"{symbol}.png"
    if cached.exists() and cached.stat().st_size > 1000:
        return cached
    coin_id = COINGECKO_ID_MAP.get(symbol.upper())
    if not coin_id:
        return None
    try:
        r = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            "?localization=false&tickers=false&market_data=false"
            "&community_data=false&developer_data=false",
            timeout=15
        )
        img_url = r.json().get("image", {}).get("large", "")
        if not img_url:
            return None
        img_r = requests.get(img_url, timeout=15)
        cached.write_bytes(img_r.content)
        if cached.stat().st_size > 1000:
            return cached
        cached.unlink(missing_ok=True)
    except Exception:
        pass
    return None


# ════════════════════════════════════════════════════════════════════════
# FINAL ASSEMBLY
# ════════════════════════════════════════════════════════════════════════

def assemble_final(
    kb_clips: list[Path],
    audio_path: Path,
    audio_dur: float,
    output_path: Path,
    sub_path: Path | None,
    coin_logo: Path | None,
    video_type: str,
) -> bool:
    """Concat Ken Burns clips + audio + overlays + subs → final video."""
    work_dir = output_path.parent
    concat_file = work_dir / "concat_kb.txt"
    concat_file.write_text("\n".join(f"file '{c}'" for c in kb_clips))

    intermediate = work_dir / "kb_concat.mp4"

    # Step 1: concat all KB clips
    r = subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-an",
        str(intermediate)
    ], capture_output=True, timeout=300)
    if r.returncode != 0:
        print(f"      ❌ Concat failed: {r.stderr[-300:]}")
        return False

    # Step 2: overlay + subs + audio
    inputs = ["-i", str(intermediate), "-i", str(audio_path)]
    filter_parts = []
    map_video = "[0:v]"

    input_idx = 2  # next input index

    # BT logo
    if BT_LOGO.exists():
        logo_size = 200 if video_type == "short" else 160
        inputs += ["-i", str(BT_LOGO)]
        filter_parts.append(f"[{input_idx}:v]scale={logo_size}:-1,format=rgba,colorchannelmixer=aa=0.85[btlogo]")
        filter_parts.append(f"{map_video}[btlogo]overlay=W-w-25:25:format=auto[vbt]")
        map_video = "[vbt]"
        input_idx += 1

    # Coin logo
    if coin_logo and coin_logo.exists():
        coin_size = 100 if video_type == "short" else 80
        inputs += ["-i", str(coin_logo)]
        filter_parts.append(f"[{input_idx}:v]scale={coin_size}:-1,format=rgba,colorchannelmixer=aa=0.90[coin]")
        filter_parts.append(f"{map_video}[coin]overlay=25:H-h-25:format=auto[vcoin]")
        map_video = "[vcoin]"
        input_idx += 1

    # Subtitles
    if sub_path and sub_path.exists() and sub_path.suffix == ".ass":
        ass_esc = str(sub_path).replace("'", "\\'").replace(":", "\\:")
        filter_parts.append(f"{map_video}ass='{ass_esc}'[vfinal]")
        map_video = "[vfinal]"

    filter_complex = ";".join(filter_parts) if filter_parts else None

    cmd = ["ffmpeg", "-y"] + inputs
    if filter_complex:
        cmd += ["-filter_complex", filter_complex, "-map", map_video, "-map", "1:a"]
    else:
        cmd += ["-map", "0:v", "-map", "1:a"]

    cmd += [
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
        "-t", str(audio_dur),
        "-movflags", "+faststart",
        str(output_path)
    ]

    r = subprocess.run(cmd, capture_output=True, timeout=600)
    if r.returncode != 0:
        print(f"      ❌ Assembly failed: {r.stderr[-500:]}")
        return False
    return output_path.exists()


# ════════════════════════════════════════════════════════════════════════
# PRODUCE ONE VIDEO
# ════════════════════════════════════════════════════════════════════════

def produce_video(video: dict, prog: dict) -> bool:
    vid_id = video["id"]
    vtype  = video["type"]
    title  = video["title"]
    audio  = video["audio"]

    print(f"\n{'='*60}")
    print(f"🦏 {title}")
    print(f"{'='*60}")

    if prog.get(vid_id) == "done":
        print("  ✅ Already done, skipping")
        return True

    out_dir = OUTPUT_BASE / vid_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check audio
    if not audio.exists():
        print(f"  ❌ Audio not found: {audio}")
        prog[vid_id] = "error_no_audio"
        save_progress(prog)
        return False

    audio_dur = get_duration(audio)
    print(f"  📻 Audio: {audio_dur:.1f}s")

    W, H = (1080, 1920) if vtype == "short" else (1920, 1080)
    n_images = math.ceil(audio_dur / 5.0)
    print(f"  🖼️  Images needed: {n_images} ({W}x{H})")

    # ── 1. Download rhino images ────────────────────────────────────────
    images_dir = out_dir / "kenburns_images"
    images_dir.mkdir(exist_ok=True)
    images = []

    for i in range(n_images):
        img_path = images_dir / f"rhino_{i:03d}.jpg"
        if img_path.exists() and img_path.stat().st_size > 5000:
            print(f"    🖼️  img {i+1}/{n_images} — cached ✅")
            images.append(img_path)
            continue

        prompt = build_image_prompt(video, i, n_images)
        print(f"    🖼️  img {i+1}/{n_images} — downloading...")
        ok = download_image(prompt, img_path, W, H, seed=42 + i * 7)
        if not ok:
            # Fallback: extract frame
            print(f"    ⚠️  Pollinations failed, extracting frame fallback...")
            ok = extract_frame_fallback(video, img_path)
            if not ok:
                # Use previous image if any
                if images:
                    print(f"    ⚠️  Using previous image as fallback")
                    import shutil
                    shutil.copy(images[-1], img_path)
                    ok = True

        if ok and img_path.exists() and img_path.stat().st_size > 1000:
            images.append(img_path)
        else:
            print(f"    ❌ Could not get image {i+1}")
            prog[vid_id] = f"error_image_{i}"
            save_progress(prog)
            return False

    # ── 2. Apply Ken Burns ──────────────────────────────────────────────
    kb_dir = out_dir / "kenburns_clips"
    kb_dir.mkdir(exist_ok=True)
    kb_clips = []

    # Distribute clip durations to cover audio_dur exactly
    base_clip_dur = audio_dur / n_images
    total_assigned = 0.0

    for i, img in enumerate(images):
        clip_path = kb_dir / f"kb_{i:03d}.mp4"
        if clip_path.exists():
            kb_clips.append(clip_path)
            total_assigned += get_duration(clip_path)
            continue

        # Last clip gets remaining time
        if i == n_images - 1:
            dur = round(audio_dur - total_assigned + 0.1, 3)
        else:
            dur = round(base_clip_dur, 3)

        print(f"    🎬 Ken Burns {i+1}/{n_images} (style={i % len(KB_STYLES)}, {dur:.1f}s)...")
        ok = apply_ken_burns(img, clip_path, i, W, H, duration=dur)
        if not ok:
            print(f"    ❌ Ken Burns failed for image {i+1}")
            prog[vid_id] = f"error_kenburns_{i}"
            save_progress(prog)
            return False

        total_assigned += dur
        kb_clips.append(clip_path)

    # ── 3. Karaoke subtitles ────────────────────────────────────────────
    script_text = load_script_text(vid_id)
    sub_path = None
    if script_text:
        try:
            print("    🎤 Generating subtitles (Whisper)...")
            sub_path = generate_karaoke_subs(
                audio, script_text, out_dir, vtype,
                style="word_highlight", use_whisper=True
            )
        except Exception as e:
            print(f"    ⚠️ Subs failed: {e}")

    # ── 4. Coin logo ────────────────────────────────────────────────────
    coin_logo = None
    if video.get("coin"):
        coin_logo = fetch_coin_logo(video["coin"])
        if coin_logo:
            print(f"    🪙 Coin logo: {video['coin']} ✅")

    # ── 5. Final assembly ───────────────────────────────────────────────
    output = out_dir / f"{vid_id}_kenburns_final.mp4"
    print(f"    🔧 Assembling final video...")
    ok = assemble_final(kb_clips, audio, audio_dur, output, sub_path, coin_logo, vtype)
    if not ok:
        prog[vid_id] = "error_assembly"
        save_progress(prog)
        return False

    size_mb = output.stat().st_size / (1024 * 1024)
    dur_out = get_duration(output)
    print(f"    ✅ Done: {output.name} — {size_mb:.1f}MB, {dur_out:.1f}s")

    prog[vid_id] = "done"
    prog[f"{vid_id}_path"] = str(output)
    save_progress(prog)
    return True


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    print("🦏 BitTrader Ken Burns Producer")
    print(f"   Videos to produce: {len(VIDEOS)}")

    prog = load_progress()
    success = 0
    failed = []

    for video in VIDEOS:
        try:
            ok = produce_video(video, prog)
            if ok:
                success += 1
            else:
                failed.append(video["id"])
        except Exception as e:
            print(f"  💥 Exception for {video['id']}: {e}")
            prog[video["id"]] = f"exception: {e}"
            save_progress(prog)
            failed.append(video["id"])

    print(f"\n{'='*60}")
    print(f"🏁 Done: {success}/{len(VIDEOS)} videos produced")
    if failed:
        print(f"   Failed: {failed}")

    # Notify
    os.system('openclaw system event --text "Ken Burns pipeline completo: videos con rinoceronte listos para revision." --mode now')


if __name__ == "__main__":
    main()
