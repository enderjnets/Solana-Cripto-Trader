#!/usr/bin/env python3
"""Generate missing thumbnails using local fallback (no API)"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

LOGO_PATH = Path("/home/enderj/.openclaw/workspace/videos/BIBLIOTECA/bittrader_logo.png")
OUTPUT_DIR = Path("/home/enderj/.openclaw/workspace/bittrader/agents/data/thumbnails")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Videos needing thumbnails
videos = [
    {
        "id": "long_1773470276_498",
        "title": "Cómo pasar FTMO",
        "bg_color": "#2E7D32",  # Green
        "text_color": "#FFD700"   # Gold
    },
    {
        "id": "short_1773470119_255",
        "title": "Bitcoin Análisis",
        "bg_color": "#E67E22",  # Orange
        "text_color": "#FFFFFF"   # White
    },
    {
        "id": "short_1773470200_200",
        "title": "TRUMP Coin",
        "bg_color": "#1565C0",  # Blue
        "text_color": "#FFFFFF"   # White
    }
]

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def generate_thumbnail(video: dict) -> Image.Image:
    """Generate thumbnail with solid background"""
    bg_rgb = hex_to_rgb(video["bg_color"])
    text_rgb = hex_to_rgb(video["text_color"])

    # Create image with gradient background
    img = Image.new('RGB', (1024, 576), color=bg_rgb)
    draw = ImageDraw.Draw(img)

    # Add gradient effect (darker at bottom)
    for y in range(0, img.height, 4):
        shade = 1.0 - (y / img.height * 0.4)
        r = int(bg_rgb[0] * shade)
        g = int(bg_rgb[1] * shade)
        b = int(bg_rgb[2] * shade)
        draw.rectangle([(0, y), (img.width, y + 4)], fill=(r, g, b))

    # Logo
    if LOGO_PATH.exists():
        logo = Image.open(LOGO_PATH).convert("RGBA")
        logo = logo.resize((80, 80), Image.Resampling.LANCZOS)
        img.paste(logo, (20, 20), logo)

    # Handle
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    draw.text((img.width - 180, 20), "@bittrader9259", fill="white", font=font, stroke_width=2, stroke_fill="black")

    # Title at bottom (MrBeast style)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except:
        title_font = ImageFont.load_default()

    # Draw dark overlay at bottom
    for y in range(img.height - 180, img.height):
        alpha = int((y - (img.height - 180)) / 180 * 220)
        draw.rectangle([(0, y), (img.width, y)], fill=(0, 0, 0, alpha))

    # Title text
    text_bbox = draw.textbbox((0, 0), video["title"], font=title_font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (img.width - text_width) // 2
    text_y = img.height - 120

    # Draw black outline
    for offset in [-2, -1, 1, 2]:
        draw.text((text_x + offset, text_y), video["title"], fill="black", font=title_font)
        draw.text((text_x, text_y + offset), video["title"], fill="black", font=title_font)
    draw.text((text_x, text_y), video["title"], fill=text_rgb, font=title_font)

    return img

# Generate thumbnails
print("🎨 Generating thumbnails (local fallback)...")
for video in videos:
    print(f"   {video['id']}: {video['title']}")

    thumb_path = OUTPUT_DIR / f"{video['id']}_thumbnail.jpg"

    if thumb_path.exists():
        print(f"      ⏭️  Already exists")
        continue

    thumb = generate_thumbnail(video)
    thumb.save(thumb_path, "JPEG", quality=95)
    print(f"      ✅ Saved: {thumb_path}")

print("\n✅ Done - All thumbnails generated")
