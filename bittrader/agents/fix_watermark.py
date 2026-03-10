#!/usr/bin/env python3
"""
🦏 BitTrader Watermark Fixer
Re-applies the BitTrader logo watermark to videos missing it.

Usage: python3 agents/fix_watermark.py [--dry-run]
"""
import json
import subprocess
import sys
from pathlib import Path
from PIL import Image

WORKSPACE  = Path("/home/enderj/.openclaw/workspace")
BITTRADER  = WORKSPACE / "bittrader"
DATA_DIR   = BITTRADER / "agents/data"
PRODUCTION = DATA_DIR / "production_latest.json"
LOGO_PATH  = WORKSPACE / "videos/BIBLIOTECA/bittrader_logo.png"

DRY_RUN = "--dry-run" in sys.argv


def count_orange_pixels(img_path: Path, x1, y1, x2, y2) -> int:
    """Count orange pixels in a region of an image."""
    img = Image.open(img_path).convert("RGB")
    region = img.crop((x1, y1, x2, y2))
    pixels = list(region.getdata())
    return sum(1 for p in pixels if p[0] > 180 and p[1] > 120 and p[1] < 190 and p[2] < 80)


def get_logo_orange_reference() -> int:
    """Get reference orange pixel count from the actual logo."""
    logo = Image.open(LOGO_PATH).convert("RGBA")
    scale = 240 / max(logo.size)
    logo_small = logo.resize((int(logo.width * scale), int(logo.height * scale)), Image.LANCZOS)
    pixels = list(logo_small.getdata())
    return sum(1 for p in pixels if p[3] > 128 and p[0] > 180 and p[1] > 120 and p[1] < 190 and p[2] < 80)


def has_watermark(video_path: Path, vtype: str) -> bool:
    """Check if video already has the BitTrader watermark."""
    frame = video_path.parent / "_wm_check.jpg"
    try:
        subprocess.run([
            "ffmpeg", "-y", "-ss", "2", "-i", str(video_path),
            "-vframes", "1", "-q:v", "2", str(frame)
        ], capture_output=True, timeout=10)

        if not frame.exists():
            return False

        # Get video dimensions
        r = subprocess.run([
            "ffprobe", "-v", "quiet", "-show_entries", "stream=width,height",
            "-of", "csv=p=0", str(video_path)
        ], capture_output=True, text=True, timeout=10)
        parts = r.stdout.strip().split('\n')[0].split(',')
        w, h = int(parts[0]), int(parts[1])

        logo_size = 240 if vtype == "short" else 180
        margin = 30
        x1 = max(0, w - logo_size - margin)
        y1 = margin
        x2 = w - margin
        y2 = logo_size + margin

        orange = count_orange_pixels(frame, x1, y1, x2, y2)
        ref = get_logo_orange_reference()
        threshold = max(ref * 0.25, 200)

        return orange >= threshold
    finally:
        if frame.exists():
            frame.unlink(missing_ok=True)


def apply_watermark(video_path: Path, vtype: str) -> bool:
    """Apply BitTrader logo watermark to video."""
    logo_size = 240 if vtype == "short" else 180
    output = video_path.parent / f"{video_path.stem}_wm{video_path.suffix}"

    # Check if video has subtitles burned in (ASS file)
    sub_file = video_path.parent / "subtitles_fixed.ass"
    if not sub_file.exists():
        sub_file = video_path.parent / "subtitles.ass"

    # Build ffmpeg command
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(LOGO_PATH),
    ]

    filter_complex = (
        f"[1:v]scale={logo_size}:-1,format=rgba,colorchannelmixer=aa=0.85[logo];"
        f"[0:v][logo]overlay=W-w-30:30:format=auto"
    )

    cmd.extend([
        "-filter_complex", filter_complex,
        "-c:a", "copy",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-movflags", "+faststart",
        str(output)
    ])

    print(f"    Applying watermark: {video_path.name} → {output.name}")
    if DRY_RUN:
        print(f"    [DRY RUN] Would run: ffmpeg ... {output.name}")
        return True

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"    ❌ ffmpeg failed: {r.stderr[-200:]}")
        return False

    # Replace original with watermarked version
    import shutil
    backup = video_path.parent / f"{video_path.stem}_nowm{video_path.suffix}"
    shutil.move(str(video_path), str(backup))
    shutil.move(str(output), str(video_path))
    print(f"    ✅ Watermark applied. Original backed up as {backup.name}")
    return True


def main():
    print("\n🦏 BitTrader Watermark Fixer")
    print(f"  Logo: {LOGO_PATH}")
    if DRY_RUN:
        print("  ⚠️  DRY RUN — no changes will be made\n")

    if not LOGO_PATH.exists():
        print("  ❌ Logo not found!")
        return

    if not PRODUCTION.exists():
        print("  ❌ Production file not found!")
        return

    prod = json.loads(PRODUCTION.read_text())
    videos = prod.get("videos", [])
    print(f"  📋 Checking {len(videos)} videos...\n")

    missing = []
    ok = []

    for v in videos:
        title = v.get("title", "?")[:50]
        vtype = v.get("type", "short")
        path = Path(v.get("output_file", ""))
        if not path.exists():
            path = BITTRADER / "agents" / v.get("output_file", "")

        if not path.exists():
            print(f"  ⚠️  NOT FOUND: {title}")
            continue

        wm = has_watermark(path, vtype)
        if wm:
            print(f"  ✅ {title} — logo OK")
            ok.append(v)
        else:
            print(f"  ❌ {title} — NO LOGO")
            missing.append((v, path, vtype))

    print(f"\n  📊 Results: {len(ok)} OK, {len(missing)} missing watermark\n")

    if not missing:
        print("  🎉 All videos have watermark!")
        return

    # Fix missing watermarks
    fixed = 0
    for v, path, vtype in missing:
        title = v.get("title", "?")[:50]
        print(f"  🔧 Fixing: {title}")
        if apply_watermark(path, vtype):
            fixed += 1

    print(f"\n  ✅ Fixed {fixed}/{len(missing)} videos")
    if DRY_RUN:
        print("  ⚠️  DRY RUN — no actual changes were made")


if __name__ == "__main__":
    main()
