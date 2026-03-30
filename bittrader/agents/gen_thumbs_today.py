#!/usr/bin/env python3
"""Generate VERTICAL (1080x1920) thumbnails for today's 2 shorts — QA-ready."""
import sys
import json
import shutil
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("❌ Pillow no instalado")
    sys.exit(1)

OUTPUT_DIR = Path("/home/enderj/.openclaw/workspace/bittrader/agents/output/2026-03-28")
THUMB_DIR  = Path("/home/enderj/.openclaw/workspace/bittrader/agents/data/thumbnails")
LOGO_PATH  = Path("/home/enderj/.openclaw/workspace/bittrader/assets/bittrader_logo.png")
THUMB_DIR.mkdir(parents=True, exist_ok=True)

# Short thumbnail: 1080x1920 (vertical 9:16)
THUMB_W = 1080
THUMB_H = 1920

GOLD  = (245, 166, 35)
WHITE = (255, 255, 255)
BLACK = (15, 15, 15)
RED   = (220, 40, 40)
GREEN = (40, 200, 80)
CYAN  = (0, 200, 255)

TITLE_COLORS = [GOLD, GREEN, CYAN, GOLD, GOLD]


def get_font(size: int) -> ImageFont.FreeTypeFont:
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    ]
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def wrap_text(text: str, max_chars: int = 14) -> list:
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current = (current + " " + word).strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:5]


def create_gradient_bg() -> Image.Image:
    bg = Image.new("RGB", (THUMB_W, THUMB_H), (10, 10, 20))
    draw = ImageDraw.Draw(bg)
    for y in range(THUMB_H):
        ratio = y / THUMB_H
        r = int(10 + 15 * ratio)
        g = int(10 + 5 * ratio)
        b = int(20 + 30 * ratio)
        draw.line([(0, y), (THUMB_W, y)], fill=(r, g, b))
    return bg


def create_vertical_thumbnail(
    title: str,
    output_path: Path,
    title_color: tuple = GOLD,
    bg_color_top=(20, 0, 40),
    bg_color_bot=(0, 10, 30),
) -> bool:
    """Create 1080x1920 vertical thumbnail with BitTrader branding."""
    try:
        # Background gradient
        bg = Image.new("RGB", (THUMB_W, THUMB_H))
        draw = ImageDraw.Draw(bg)
        for y in range(THUMB_H):
            ratio = y / THUMB_H
            r = int(bg_color_top[0] + (bg_color_bot[0] - bg_color_top[0]) * ratio)
            g = int(bg_color_top[1] + (bg_color_bot[1] - bg_color_top[1]) * ratio)
            b = int(bg_color_top[2] + (bg_color_bot[2] - bg_color_top[2]) * ratio)
            draw.line([(0, y), (THUMB_W, y)], fill=(r, g, b))

        # Accent lines
        for i in range(3):
            y_pos = 200 + i * 8
            draw.line([(60, y_pos), (THUMB_W - 60, y_pos)], fill=(*GOLD, ), width=3 - i)

        # Logo top-left
        if LOGO_PATH.exists():
            try:
                logo = Image.open(LOGO_PATH).convert("RGBA")
                logo_h = 100
                logo_w = int(logo.width * (logo_h / logo.height))
                logo = logo.resize((logo_w, logo_h), Image.Resampling.LANCZOS)
                bg.paste(logo, (40, 50), logo)
            except Exception as e:
                print(f"  ⚠️ Logo error: {e}")

        # Main title (large, centered vertically around middle)
        lines = wrap_text(title.upper(), max_chars=14)
        font_size = 130 if len(lines) <= 3 else 105
        font_big = get_font(font_size)

        # Center text block vertically
        total_height = len(lines) * (font_size + 15)
        y_start = (THUMB_H - total_height) // 2 - 100

        for line in lines:
            # Center each line horizontally
            try:
                bbox = font_big.getbbox(line)
                text_w = bbox[2] - bbox[0]
            except Exception:
                text_w = len(line) * font_size // 2
            x = (THUMB_W - text_w) // 2
            draw.text((x, y_start), line, font=font_big, fill=title_color,
                      stroke_width=5, stroke_fill=BLACK)
            y_start += font_size + 15

        # Gold separator line below title
        sep_y = y_start + 30
        draw.rectangle([60, sep_y, THUMB_W - 60, sep_y + 6], fill=GOLD)

        # @bittrader9259 handle at bottom
        font_handle = get_font(55)
        handle_text = "@bittrader9259"
        try:
            bbox = font_handle.getbbox(handle_text)
            handle_w = bbox[2] - bbox[0]
        except Exception:
            handle_w = len(handle_text) * 30
        hx = (THUMB_W - handle_w) // 2
        draw.text((hx, THUMB_H - 160), handle_text, font=font_handle,
                  fill=WHITE, stroke_width=3, stroke_fill=BLACK)

        # WATCH NOW button
        btn_w, btn_h = 380, 90
        btn_x = (THUMB_W - btn_w) // 2
        btn_y = THUMB_H - 310
        draw.rectangle([btn_x, btn_y, btn_x + btn_w, btn_y + btn_h], fill=GOLD)
        draw.rectangle([btn_x, btn_y, btn_x + 80, btn_y + btn_h], fill=RED)
        font_btn = get_font(52)
        draw.text((btn_x + 15, btn_y + 18), "▶", font=font_btn, fill=WHITE)
        draw.text((btn_x + 100, btn_y + 18), "VER AHORA", font=font_btn, fill=BLACK)

        bg.save(str(output_path), "JPEG", quality=92)
        return True
    except Exception as e:
        print(f"  ❌ Error generando thumbnail: {e}")
        import traceback; traceback.print_exc()
        return False


videos = [
    {
        "script_id": "short_1774708157_574",
        "title": "Cathie Wood VENDE Meta, Nvidia y BTC",
        "color": GREEN,
        "bg_top": (0, 20, 10),
        "bg_bot": (10, 40, 5),
        "out_dir": OUTPUT_DIR / "short_1774708157_574",
    },
    {
        "script_id": "short_1774708280_155",
        "title": "SOL rompe $150: Pump inminente?",
        "color": CYAN,
        "bg_top": (0, 10, 30),
        "bg_bot": (5, 25, 50),
        "out_dir": OUTPUT_DIR / "short_1774708280_155",
    }
]

results = []
for i, v in enumerate(videos, 1):
    title     = v["title"]
    sid       = v["script_id"]
    out_dir   = v["out_dir"]

    thumb_path      = out_dir / "thumbnail.jpg"
    thumb_data_path = THUMB_DIR / f"{sid}_thumbnail.jpg"

    print(f"\n[{i}/2] {title}")
    ok = create_vertical_thumbnail(
        title=title,
        output_path=thumb_path,
        title_color=v["color"],
        bg_color_top=v["bg_top"],
        bg_color_bot=v["bg_bot"],
    )
    if ok:
        shutil.copy2(thumb_path, thumb_data_path)
        size_kb = thumb_path.stat().st_size / 1024
        w, h = Image.open(thumb_path).size
        print(f"  ✅ {w}x{h} — {size_kb:.0f}KB")
        print(f"     → {thumb_path}")
        results.append({"script_id": sid, "thumb": str(thumb_path), "data_thumb": str(thumb_data_path), "ok": True})
    else:
        results.append({"script_id": sid, "ok": False})

print("\n=== THUMBNAILS ===")
for r in results:
    status = "✅" if r["ok"] else "❌"
    print(f"  {status} {r['script_id']}")
