#!/usr/bin/env python3
"""Generate thumbnails for 3 edu_ videos (2026-03-24)"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap

LOGO_PATH = Path("/home/enderj/.openclaw/workspace/videos/BIBLIOTECA/bittrader_logo.png")
OUTPUT_DIR = Path("/home/enderj/.openclaw/workspace/bittrader/agents/data/thumbnails")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

videos = [
    {
        "id": "edu_short1_1774356226",
        "title": "El error del 90%\nde nuevos traders",
        "type": "short",
        "bg_color": "#B71C1C",   # Deep red - urgency
        "text_color": "#FFFFFF",
        "accent": "#FFD700",
    },
    {
        "id": "edu_short2_1774359590",
        "title": "STOP LOSS\nTe salva de perder TODO",
        "type": "short",
        "bg_color": "#1565C0",   # Blue - safety/trust
        "text_color": "#FFFFFF",
        "accent": "#00E676",
    },
    {
        "id": "edu_long_1774363903",
        "title": "Cómo empezar en\ncripto con $100",
        "type": "long",
        "bg_color": "#1B5E20",   # Green - money/growth
        "text_color": "#FFFFFF",
        "accent": "#FFD700",
    },
]


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def generate_thumbnail(video: dict) -> Image.Image:
    is_short = video["type"] == "short"
    # Shorts: 1080x1920, Longs: 1280x720
    w, h = (1080, 1920) if is_short else (1280, 720)

    bg_rgb = hex_to_rgb(video["bg_color"])
    text_rgb = hex_to_rgb(video["text_color"])
    accent_rgb = hex_to_rgb(video["accent"])

    img = Image.new('RGB', (w, h), color=bg_rgb)
    draw = ImageDraw.Draw(img)

    # Gradient background (darker at bottom)
    for y in range(0, h, 2):
        shade = 1.0 - (y / h * 0.5)
        r = int(bg_rgb[0] * shade)
        g = int(bg_rgb[1] * shade)
        b = int(bg_rgb[2] * shade)
        draw.rectangle([(0, y), (w, y + 2)], fill=(r, g, b))

    # Decorative accent bar at top
    draw.rectangle([(0, 0), (w, 8)], fill=accent_rgb)

    # Logo
    if LOGO_PATH.exists():
        logo = Image.open(LOGO_PATH).convert("RGBA")
        logo_size = 120 if is_short else 80
        logo = logo.resize((logo_size, logo_size), Image.Resampling.LANCZOS)
        logo_x = (w - logo_size) // 2 if is_short else 20
        logo_y = 40 if is_short else 20
        img.paste(logo, (logo_x, logo_y), logo)

    # Handle text
    try:
        handle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28 if is_short else 22)
    except:
        handle_font = ImageFont.load_default()

    handle_text = "@bittrader9259"
    hbox = draw.textbbox((0, 0), handle_text, font=handle_font)
    hx = (w - (hbox[2] - hbox[0])) // 2 if is_short else (w - (hbox[2] - hbox[0]) - 20)
    hy = (logo_y + logo_size + 10) if is_short else 25
    draw.text((hx, hy), handle_text, fill=accent_rgb, font=handle_font, stroke_width=1, stroke_fill="black")

    # Title text - big and bold
    title_size = 90 if is_short else 60
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", title_size)
    except:
        title_font = ImageFont.load_default()

    lines = video["title"].split("\n")

    # Calculate total text height
    line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        line_heights.append(bbox[3] - bbox[1])
    total_text_h = sum(line_heights) + (len(lines) - 1) * 20

    # Center text vertically
    text_y = (h - total_text_h) // 2

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = (w - tw) // 2

        # Black outline
        for ox in [-3, -2, -1, 1, 2, 3]:
            for oy in [-3, -2, -1, 1, 2, 3]:
                draw.text((tx + ox, text_y + oy), line, fill="black", font=title_font)
        # Main text
        draw.text((tx, text_y), line, fill=text_rgb, font=title_font)
        text_y += th + 20

    # Bottom accent bar
    draw.rectangle([(0, h - 8), (w, h)], fill=accent_rgb)

    # "PRINCIPIANTES" badge
    try:
        badge_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32 if is_short else 24)
    except:
        badge_font = ImageFont.load_default()

    badge_text = "PRINCIPIANTES"
    bbox = draw.textbbox((0, 0), badge_text, font=badge_font)
    bw = bbox[2] - bbox[0] + 30
    bh = bbox[3] - bbox[1] + 20
    bx = (w - bw) // 2
    by = h - 100 if is_short else h - 70

    draw.rounded_rectangle([(bx, by), (bx + bw, by + bh)], radius=10, fill=accent_rgb)
    draw.text((bx + 15, by + 8), badge_text, fill="black", font=badge_font)

    return img


print("Generating edu_ thumbnails...")
for video in videos:
    thumb_path = OUTPUT_DIR / f"{video['id']}_thumbnail.jpg"
    print(f"  {video['id']}: {video['title'].replace(chr(10), ' | ')}")

    thumb = generate_thumbnail(video)
    thumb.save(str(thumb_path), "JPEG", quality=95)
    print(f"    Saved: {thumb_path} ({thumb.size[0]}x{thumb.size[1]})")

print("\nDone - 3 edu_ thumbnails generated")
