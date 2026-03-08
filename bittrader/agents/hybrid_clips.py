#!/usr/bin/env python3
"""
🎬 Hybrid Clip Generator v2: Pexels first, Hailuo fallback
- Per-scene varied queries (not same clip for all scenes)
- Pexels stock video when relevant (score >= 0.6)
- Hailuo 2.3 AI generation as fallback
- Assembles final v3 video with Edge-TTS audio + logo + karaoke subs
"""
import json
import time
import re
import subprocess
import requests
from pathlib import Path

WORKSPACE = Path("/home/enderj/.openclaw/workspace")
KEYS_FILE = WORKSPACE / "bittrader/keys/minimax.json"
PEXELS_KEY_FILE = WORKSPACE / "bittrader/keys/pexels.json"

mm_keys = json.loads(KEYS_FILE.read_text())
MINIMAX_API_KEY = mm_keys["minimax_api_key"]

HAILUO_SUBMIT = "https://api.minimax.io/v1/video_generation"
HAILUO_QUERY  = "https://api.minimax.io/v1/query/video_generation"
HAILUO_FILES  = "https://api.minimax.io/v1/files/retrieve"
PEXELS_BASE   = "https://api.pexels.com/videos"

PEXELS_MIN_SCORE   = 0.6
CLIP_DURATION_TARGET = 6  # seconds

# Varied fallback queries by scene index (when clips_needed > prompts)
SCENE_FALLBACKS = [
    "stock market trading",
    "cryptocurrency bitcoin",
    "trading charts candlestick",
    "financial technology fintech",
    "trader computer screen",
    "market data analysis",
    "crypto coins bitcoin",
    "professional trader desk",
    "stock exchange market",
    "digital finance technology",
    "investment portfolio",
    "blockchain network",
]

# Keyword->query mapping for scene prompts
KEYWORD_MAP = [
    (["server room", "gpu", "mining"],               "crypto mining server"),
    (["penguin", "pengu"],                            "penguin cartoon"),
    (["trader", "eyes", "screen", "monitor"],         "trader monitor screen"),
    (["coin", "cracked", "golden", "crypto"],         "crypto coin"),
    (["dashboard", "trading", "chart", "green","red"],"trading charts"),
    (["calendar", "30 day", "profit"],                "business calendar progress"),
    (["keyboard", "typing", "hands", "laptop"],       "trader typing keyboard"),
    (["news", "monitor", "dark room","multiple screen"],"trading room monitors"),
    (["futuristic", "ai", "interface", "robot"],      "futuristic technology screen"),
    (["privacy", "anonymous", "zec", "zcash"],        "anonymous privacy technology"),
    (["split screen", "human", "bot", "comparison"],  "human vs robot technology"),
    (["desk", "laptop", "upward", "chart"],           "laptop charts success"),
    (["confident", "professional", "two monitors"],   "professional trader desk"),
    (["bitcoin", "btc"],                              "bitcoin cryptocurrency"),
    (["candlestick", "chart", "trading view"],        "stock chart candlestick"),
    (["akt", "akash", "decentralized", "nodes"],      "decentralized network technology"),
    (["pi coin", "pi network"],                       "mobile phone cryptocurrency"),
    (["35%", "loss", "fail", "losing", "red"],        "trading loss failure"),
    (["profit", "green", "success", "winning"],       "trading profit success"),
    (["phone", "mobile", "app", "notification"],      "trading mobile app"),
    (["city", "skyline", "financial district"],       "financial district city"),
    (["clock", "time", "countdown"],                  "time countdown deadline"),
    (["broker", "fund", "capital", "funded"],         "funded trading account"),
    (["spreadsheet", "data", "left", "right"],        "data analysis spreadsheet"),
    (["split", "slow", "fast", "comparison"],         "speed comparison technology"),
]

def get_pexels_key():
    if PEXELS_KEY_FILE.exists():
        return json.loads(PEXELS_KEY_FILE.read_text()).get("api_key", "")
    return ""

def extract_search_query(prompt: str, scene_idx: int = 0, used_queries: list = None) -> str:
    """Extract varied search query from scene prompt. Avoids repeating same query."""
    prompt_lower = prompt.lower()
    used_queries = used_queries or []

    for keywords, query in KEYWORD_MAP:
        if any(kw in prompt_lower for kw in keywords) and query not in used_queries:
            return query

    # Fallback: extract meaningful words from prompt
    words = re.sub(r'[^\w\s]', '', prompt[:120]).split()
    skip = {'cinematic','vertical','close','shot','with','showing','that','this',
            'from','into','over','wide','dark','light','glowing','short','long',
            'opening','scene','view','split','screen','animation'}
    meaningful = [w for w in words if len(w) > 3 and w.lower() not in skip]
    query = ' '.join(meaningful[:3])

    if query and query not in used_queries:
        return query

    # Last resort: varied pool
    for fb in SCENE_FALLBACKS:
        if fb not in used_queries:
            return fb

    return SCENE_FALLBACKS[scene_idx % len(SCENE_FALLBACKS)]

def search_pexels(query: str, orientation: str = "portrait") -> list:
    key = get_pexels_key()
    if not key:
        return []
    try:
        resp = requests.get(
            f"{PEXELS_BASE}/search",
            params={"query": query, "per_page": 5, "orientation": orientation, "size": "medium"},
            headers={"Authorization": key},
            timeout=10
        )
        if resp.status_code != 200:
            return []
        results = []
        for video in resp.json().get("videos", []):
            duration = video.get("duration", 0)
            w, h = video.get("width", 0), video.get("height", 0)
            dur_score = 1.0 if 4 <= duration <= 12 else (0.7 if 3 <= duration <= 20 else 0.3)
            orient_score = 1.0 if h > w else 0.7
            score = dur_score * 0.6 + orient_score * 0.4

            best_file, best_res = None, 0
            for vf in video.get("video_files", []):
                res = vf.get("width", 0) * vf.get("height", 0)
                if res > best_res and vf.get("width", 0) <= 1920:
                    best_res = res
                    best_file = vf.get("link", "")

            if best_file:
                results.append((score, best_file, duration, w, h))
        results.sort(reverse=True)
        return results
    except Exception as e:
        print(f"    Pexels error: {e}")
        return []

def download_clip(url: str, output_path: Path, target_duration: float) -> bool:
    tmp = output_path.parent / f"_tmp_{output_path.stem}.mp4"
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        with open(tmp, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp), "-t", str(target_duration),
             "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-an",
             str(output_path)],
            capture_output=True, timeout=30
        )
        tmp.unlink(missing_ok=True)
        return output_path.exists()
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"    Download error: {e}")
        return False

def submit_hailuo(prompt: str) -> str:
    if "9:16" not in prompt and "vertical" not in prompt.lower():
        prompt = f"Cinematic vertical 9:16 short clip. {prompt}"
    resp = requests.post(
        HAILUO_SUBMIT,
        headers={"Authorization": f"Bearer {MINIMAX_API_KEY}", "Content-Type": "application/json"},
        json={"model": "MiniMax-Hailuo-2.3", "prompt": prompt[:1000],
              "duration": 6, "aspect_ratio": "9:16"},
        timeout=30
    )
    data = resp.json()
    task_id = data.get("task_id", "")
    if not task_id:
        raise ValueError(f"Hailuo submit failed: {data}")
    return task_id

def poll_hailuo(task_id: str, output_path: Path, timeout: int = 600) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(HAILUO_QUERY, params={"task_id": task_id},
                           headers={"Authorization": f"Bearer {MINIMAX_API_KEY}"}, timeout=15)
        data = resp.json()
        status = data.get("status", "")
        if status == "Success":
            file_id = data.get("file_id", "")
            if not file_id:
                return False
            file_resp = requests.get(f"{HAILUO_FILES}?file_id={file_id}",
                                    headers={"Authorization": f"Bearer {MINIMAX_API_KEY}"}, timeout=15)
            download_url = file_resp.json().get("file", {}).get("download_url", "")
            if not download_url:
                return False
            vid_resp = requests.get(download_url, stream=True, timeout=60)
            with open(output_path, 'wb') as f:
                for chunk in vid_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return output_path.exists()
        elif status in ("Failed", "Cancelled"):
            print(f"    Hailuo failed: {data.get('err_msg','')}")
            return False
        print(f"    Hailuo: {status}...", end="\r")
        time.sleep(15)
    print(f"    Hailuo timeout")
    return False

def get_clip(prompt: str, output_path: Path, scene_idx: int,
             vtype: str, used_queries: list) -> tuple:
    orientation = "portrait" if vtype == "short" else "landscape"
    query = extract_search_query(prompt, scene_idx, used_queries)
    used_queries.append(query)

    print(f"    🔍 Pexels: '{query}'")
    results = search_pexels(query, orientation)

    if results:
        score, url, duration, w, h = results[0]
        print(f"    📹 score={score:.2f}, {w}x{h}, {duration}s")
        if score >= PEXELS_MIN_SCORE:
            ok = download_clip(url, output_path, CLIP_DURATION_TARGET)
            if ok:
                print(f"    ✅ Pexels OK")
                return "pexels", True
            print(f"    ⚠️ Download failed → Hailuo")
        else:
            print(f"    ⚠️ Low score → Hailuo")
    else:
        print(f"    ⚠️ No results → Hailuo")

    print(f"    🎨 Hailuo generating...")
    try:
        task_id = submit_hailuo(prompt)
        ok = poll_hailuo(task_id, output_path)
        return "hailuo", ok
    except Exception as e:
        print(f"    ❌ Hailuo: {e}")
        return "hailuo", False

def generate_clips(script: dict, out_dir: Path) -> list:
    clips_dir = out_dir / "clips_v3"
    clips_dir.mkdir(exist_ok=True)

    prompts = script.get("video_prompts", [])
    vtype   = script.get("type", "short")
    name    = out_dir.name

    edge_candidates = list(out_dir.glob("*edge*.mp3")) + list(out_dir.glob("*_edge.mp3"))
    audio_path = edge_candidates[0] if edge_candidates else out_dir / f"{name}_edge.mp3"
    r = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(audio_path)],
                       capture_output=True, text=True)
    audio_dur = float(r.stdout.strip()) if r.stdout.strip() else 50.0

    clips_needed = max(len(prompts), int(audio_dur / CLIP_DURATION_TARGET) + 1)
    print(f"  Audio: {audio_dur:.1f}s | Prompts: {len(prompts)} | Need: {clips_needed} clips")

    clip_paths = []
    used_queries = []

    for i in range(clips_needed):
        clip_path = clips_dir / f"scene_{i+1:02d}.mp4"

        if clip_path.exists():
            dur = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                                  "-of", "csv=p=0", str(clip_path)],
                                 capture_output=True, text=True).stdout.strip()
            if float(dur or 0) > 2:
                print(f"  Scene {i+1}: SKIP")
                clip_paths.append(clip_path)
                continue

        # Pick prompt: use available prompts, then fallback queries for extra scenes
        if i < len(prompts):
            prompt = prompts[i]
        else:
            fallback_q = SCENE_FALLBACKS[(i - len(prompts)) % len(SCENE_FALLBACKS)]
            prompt = f"Cinematic trading finance scene. {fallback_q}"

        print(f"  Scene {i+1}/{clips_needed}:")
        source, ok = get_clip(prompt, clip_path, i, vtype, used_queries)

        if ok:
            print(f"  ✅ Scene {i+1} ({source})")
            clip_paths.append(clip_path)
        else:
            print(f"  ❌ Scene {i+1} failed")

        if source == "hailuo" and i < clips_needed - 1:
            print(f"    ⏳ Rate limit 35s...")
            time.sleep(35)
        else:
            time.sleep(0.5)

    return clip_paths

def assemble(script: dict, out_dir: Path, clip_paths: list) -> Path:
    LOGO = WORKSPACE / "videos/BIBLIOTECA/bittrader_logo.png"
    vtype  = script.get("type", "short")
    name   = out_dir.name

    # Find edge audio file (may have different naming)
    edge_candidates = list(out_dir.glob("*edge*.mp3")) + list(out_dir.glob("*_edge.mp3"))
    if edge_candidates:
        audio_path = edge_candidates[0]
    else:
        audio_path = out_dir / f"{name}_edge.mp3"
    sub_path    = out_dir / "subtitles.ass"
    output_path = out_dir / f"{name}_v3_final.mp4"

    r = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(audio_path)],
                       capture_output=True, text=True)
    audio_dur = float(r.stdout.strip()) if r.stdout.strip() else 50.0

    # Write concat.txt
    concat_path = out_dir / "concat_v3.txt"
    lines = [f"file '{p.resolve()}'" for p in clip_paths if p.exists()]
    concat_path.write_text("\n".join(lines))

    # Step 1: scale/blur to target resolution
    inter = out_dir / "_inter_v3.mp4"
    if vtype == "short":
        vf = ("[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
              "crop=1080:1920,avgblur=30[bg];"
              "[0:v]scale=1080:-2:force_original_aspect_ratio=decrease[fg];"
              "[bg][fg]overlay=(W-w)/2:(H-h)/2")
        r1 = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_path),
             "-filter_complex", vf, "-t", str(audio_dur + 0.5),
             "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-an", str(inter)],
            capture_output=True, text=True, timeout=120)
    else:
        r1 = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_path),
             "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,"
                    "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black",
             "-t", str(audio_dur + 0.5),
             "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-an", str(inter)],
            capture_output=True, text=True, timeout=120)

    if not inter.exists():
        print(f"  ❌ Intermediate failed: {r1.stderr[-150:]}")
        return None

    # Step 2: audio + logo (filter_complex) then subs (separate -vf pass)
    logo_size = 240 if vtype == "short" else 180
    logo_out  = out_dir / "_logo.mp4"

    r2 = subprocess.run(
        ["ffmpeg", "-y",
         "-i", str(inter), "-i", str(audio_path), "-i", str(LOGO),
         "-filter_complex", f"[2:v]scale={logo_size}:-1[logo];[0:v][logo]overlay=W-w-30:30:format=auto[v]",
         "-map", "[v]", "-map", "1:a",
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
         "-shortest", "-movflags", "+faststart", str(logo_out)],
        capture_output=True, text=True, timeout=180)

    inter.unlink(missing_ok=True)

    if not logo_out.exists():
        print(f"  ❌ Logo overlay failed: {r2.stderr[-150:]}")
        return None

    # Step 3: burn ASS subtitles (separate pass avoids filter_complex escaping issues)
    sub_esc = str(sub_path.resolve())
    r3 = subprocess.run(
        ["ffmpeg", "-y", "-i", str(logo_out),
         "-vf", f"ass={sub_esc}",
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-c:a", "copy", "-movflags", "+faststart", str(output_path)],
        capture_output=True, text=True, timeout=180)

    logo_out.unlink(missing_ok=True)

    if output_path.exists():
        size = output_path.stat().st_size / 1024 / 1024
        dur  = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                               "-of", "csv=p=0", str(output_path)],
                              capture_output=True, text=True).stdout.strip()
        print(f"  ✅ {output_path.name}: {size:.1f}MB, {dur}s")
        return output_path
    else:
        print(f"  ❌ Sub burn failed: {r3.stderr[-150:]}")
        return None


if __name__ == "__main__":
    data_dir = Path("data")
    guiones  = json.loads((data_dir / "guiones_latest.json").read_text())
    scripts  = guiones["scripts"]

    print(f"🎬 Hybrid Clip Generator v2 — {len(scripts)} videos\n")
    if not get_pexels_key():
        print("⚠️  No Pexels key — only Hailuo will be used\n")

    results = {}

    for i, script in enumerate(scripts, 1):
        title    = script["title"]
        out_file = Path(script.get("output_file", ""))
        out_dir  = out_file.parent
        vtype    = script.get("type", "short")

        print(f"[{i}/8] [{vtype.upper()}] {title[:45]}")

        if not out_dir.exists():
            print(f"  ❌ Dir not found\n"); continue

        # Check if v3 already done
        v3_path = out_dir / f"{out_dir.name}_v3_final.mp4"
        if v3_path.exists():
            print(f"  ✅ SKIP (v3 exists: {v3_path.stat().st_size/1024/1024:.1f}MB)\n")
            results[title] = str(v3_path)
            continue

        clip_paths = generate_clips(script, out_dir)
        if not clip_paths:
            print(f"  ❌ No clips\n"); continue

        print(f"  📹 {len(clip_paths)} clips ready")
        output = assemble(script, out_dir, clip_paths)
        if output:
            results[title] = str(output)
        print()

    print(f"✅ {len(results)}/{len(scripts)} videos built")
    (data_dir / "production_v3_paths.json").write_text(json.dumps(results, indent=2))
