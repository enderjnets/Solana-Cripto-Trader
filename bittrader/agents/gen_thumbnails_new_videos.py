#!/usr/bin/env python3
"""
🎨 Generate thumbnails for 5 new videos + regenerate SOL thumbnail
Uses local PIL only — no API required
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import sys

LOGO_PATH = Path("/home/enderj/.openclaw/workspace/bittrader/assets/bittrader_logo.png")
THUMBS_DIR = Path("/home/enderj/.openclaw/workspace/bittrader/agents/data/thumbnails")
THUMBS_DIR.mkdir(parents=True, exist_ok=True)

FONT_BOLD   = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_NORMAL = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Colors
GOLD    = (245, 166, 35)
WHITE   = (255, 255, 255)
BLACK   = (0,   0,   0)
RED     = (220, 40,  40)
GREEN   = (50,  220, 80)
CYAN    = (0,   200, 255)
PURPLE  = (160, 50,  220)
ORANGE  = (255, 120, 0)

def get_font(size: int, bold: bool = True):
    try:
        path = FONT_BOLD if bold else FONT_NORMAL
        return ImageFont.truetype(path, size)
    except:
        return ImageFont.load_default()

def draw_text_with_outline(draw, pos, text, font, fill, outline_color=BLACK, outline_width=8):
    x, y = pos
    # Draw outline
    for ox in range(-outline_width, outline_width + 1, 2):
        for oy in range(-outline_width, outline_width + 1, 2):
            if ox != 0 or oy != 0:
                draw.text((x + ox, y + oy), text, font=font, fill=outline_color)
    # Draw main text
    draw.text((x, y), text, font=font, fill=fill)

def create_gradient_bg(width, height, color1, color2, vertical=True):
    """Create gradient background"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    for i in range(height if vertical else width):
        t = i / (height if vertical else width)
        r = int(color1[0] * (1 - t) + color2[0] * t)
        g = int(color1[1] * (1 - t) + color2[1] * t)
        b = int(color1[2] * (1 - t) + color2[2] * t)
        if vertical:
            draw.line([(0, i), (width, i)], fill=(r, g, b))
        else:
            draw.line([(i, 0), (i, height)], fill=(r, g, b))
    return img

def add_dark_overlay_bottom(img):
    """Add dark gradient at bottom for text readability"""
    w, h = img.size
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for y in range(h // 2, h):
        alpha = int(200 * (y - h // 2) / (h // 2))
        draw.line([(0, y), (w, y)], fill=(0, 0, 0, alpha))
    img_rgba = img.convert('RGBA')
    img_rgba = Image.alpha_composite(img_rgba, overlay)
    return img_rgba.convert('RGB')

def add_logo_and_handle(draw, img, logo_size=100):
    """Add BitTrader logo and handle"""
    if LOGO_PATH.exists():
        logo = Image.open(LOGO_PATH).convert('RGBA')
        logo = logo.resize((logo_size, logo_size), Image.LANCZOS)
        img.paste(logo, (20, 15), logo)

    font_handle = get_font(24)
    w, h = img.size
    handle = "@bittrader9259"
    draw_text_with_outline(draw, (w - 240, 20), handle, font_handle, WHITE, BLACK, 4)

def add_decorative_bars(draw, img, accent_color):
    """Add visual accent bars"""
    w, h = img.size
    # Left accent bar
    draw.rectangle([(0, 0), (8, h)], fill=accent_color)
    # Bottom accent line
    draw.rectangle([(0, h - 6), (w, h)], fill=accent_color)

def generate_thumbnail(
    script_id: str,
    title: str,
    headline_line1: str,
    headline_line2: str = "",
    emoji: str = "",
    bg_color1=(10, 15, 40),
    bg_color2=(5, 8, 20),
    accent_color=GOLD,
    subtitle: str = "",
    video_type: str = "short",
) -> Path:
    """Generate a single thumbnail with BitTrader branding"""
    # Shorts need 1080x1920 (vertical), longs need 1280x720 (horizontal)
    if video_type == "short":
        W, H = 1080, 1920
    else:
        W, H = 1280, 720
    
    # Background gradient
    img = create_gradient_bg(W, H, bg_color1, bg_color2)
    
    # Add dark overlay at bottom
    img = add_dark_overlay_bottom(img)
    
    # Add accent bars
    draw = ImageDraw.Draw(img)
    add_decorative_bars(draw, img, accent_color)
    
    # Add chart-like decorative lines in background
    import random
    random.seed(hash(script_id) % 10000)
    # Draw subtle chart lines (scale to image size)
    chart_y_center = H // 2
    chart_range = H // 8
    prev_y = chart_y_center
    step = max(20, W // 40)
    for x in range(100, W - 100, step):
        delta = random.randint(-chart_range // 3, chart_range // 3)
        new_y = max(H // 4, min(H * 3 // 4, prev_y + delta))
        opacity_color = tuple(int(c * 0.15) for c in accent_color)
        draw.line([(x - step, prev_y), (x, new_y)], fill=opacity_color, width=2)
        prev_y = new_y
    
    # Logo & handle
    add_logo_and_handle(draw, img, logo_size=min(100, W // 11))
    
    # Scale fonts for image size
    font_scale = H // 720  # 1 for 720p, 2.67 for 1920p
    font_big_size = max(70, 65 * font_scale)
    font_sub_size = max(28, 24 * font_scale)
    line_height = max(90, 85 * font_scale)
    
    font_big = get_font(font_big_size, bold=True)
    font_sub = get_font(font_sub_size, bold=False)
    
    # Position text (bottom of image)
    margin_bottom = max(160, 130 * font_scale)
    y_start = H - margin_bottom - (line_height if headline_line2 else 0)
    
    # Emoji badge (vertical center area for shorts, upper area for longs)
    if emoji:
        emoji_size = max(60, 55 * font_scale)
        font_emoji = get_font(emoji_size, bold=True)
        emoji_y = H // 3 if video_type == "short" else H // 4
        draw.text((W // 2 - emoji_size // 2, emoji_y), emoji, font=font_emoji, fill=WHITE)
    
    # Line 1 headline
    draw_text_with_outline(draw, (30, y_start), headline_line1, font_big, accent_color, BLACK, 10)
    
    # Line 2 headline
    if headline_line2:
        draw_text_with_outline(draw, (30, y_start + line_height), headline_line2, font_big, WHITE, BLACK, 10)
    
    # Subtitle
    if subtitle:
        y_sub = y_start + (line_height * 2 + 10 if headline_line2 else line_height + 10)
        draw_text_with_outline(draw, (30, y_sub), subtitle, font_sub, WHITE, BLACK, 4)
    
    # Save
    out_path = THUMBS_DIR / f"{script_id}_thumbnail.jpg"
    img.save(out_path, quality=95)
    print(f"  ✅ {script_id}: {out_path.name} ({out_path.stat().st_size // 1024}KB)")
    
    # Also copy to the output folder
    output_dir = Path(f"/home/enderj/.openclaw/workspace/bittrader/agents/output/2026-03-28/{script_id}")
    if output_dir.exists():
        import shutil
        dest = output_dir / "thumbnail.jpg"
        shutil.copy2(out_path, dest)
        print(f"     → copied to {dest}")
    
    return out_path


# ── Video definitions ───────────────────────────────────────────────────────

VIDEOS = [
    # 1. TAO - gold/yellow AI vibes  (SHORT → 1080x1920)
    dict(
        script_id="short_1774708351_945",
        title="TAO: La crypto IA que explota HOY",
        headline_line1="TAO EXPLOTA",
        headline_line2="HOY 🔥",
        emoji="🤖",
        bg_color1=(60, 40, 100),   # medium purple (bright enough)
        bg_color2=(30, 15, 70),
        accent_color=GOLD,
        subtitle="Bittensor: IA descentralizada",
        video_type="short",
    ),
    # 2. SIREN y RAIN - purple/cyan degen vibes  (SHORT → 1080x1920)
    dict(
        script_id="short_1774708455_613",
        title="SIREN y RAIN: ¿Próximos gainers?",
        headline_line1="SIREN y RAIN",
        headline_line2="¿+300%?",
        emoji="📈",
        bg_color1=(30, 60, 120),   # medium blue
        bg_color2=(15, 30, 90),
        accent_color=CYAN,
        subtitle="Señales clave que debes ver",
        video_type="short",
    ),
    # 3. SIREN - green/gold tech  (SHORT → 1080x1920)
    # Avoid blue dominance (BLUE_THUMB check: avg_b > avg_r*1.7 && avg_b>100)
    dict(
        script_id="short_1774708563_481",
        title="SIREN: Qué es y por qué sube HOY",
        headline_line1="SIREN +300%",
        headline_line2="¿POR QUÉ?",
        emoji="🔮",
        bg_color1=(30, 100, 50),   # green dominant — avoids blue check
        bg_color2=(15, 60, 25),
        accent_color=GREEN,
        subtitle="IA + Trading en BSC",
        video_type="short",
    ),
    # 4. Claude IA trading - professional blue  (LONG → 1280x720)
    # Need brightness > 30 → use brighter mid-tone background
    dict(
        script_id="long_1774708225_575",
        title="Claude IA: Automatiza tu Trading HOY",
        headline_line1="AUTOMATIZA",
        headline_line2="con CLAUDE IA",
        emoji="💻",
        bg_color1=(30, 70, 140),   # medium-bright blue
        bg_color2=(15, 45, 100),
        accent_color=GOLD,
        subtitle="Análisis de trading en segundos",
        video_type="long",
    ),
    # 5. 3 reglas de trading - red/gold  (LONG → 1280x720)
    dict(
        script_id="long_1774708531_991",
        title="3 reglas que salvaron mi cuenta de trading",
        headline_line1="3 REGLAS",
        headline_line2="QUE TE SALVAN",
        emoji="🛡️",
        bg_color1=(140, 40, 20),   # medium-bright red
        bg_color2=(80, 20, 10),
        accent_color=GOLD,
        subtitle="Gestión de riesgo real",
        video_type="long",
    ),
    # 6. SOL - green breakout (regenerated)  (SHORT → 1080x1920)
    dict(
        script_id="short_1774708280_155",
        title="SOL rompe $150: Pump inminente?",
        headline_line1="SOL ROMPE",
        headline_line2="$150 🚀",
        emoji="⚡",
        bg_color1=(20, 120, 40),   # medium green
        bg_color2=(10, 70, 20),
        accent_color=GREEN,
        subtitle="¿Pump inminente? Análisis clave",
        video_type="short",
    ),
    # 7. RAIN - purple/rain vibes  (SHORT → 1080x1920)
    dict(
        script_id="short_1774708608_208",
        title="RAIN: Que es y por que sube AHORA",
        headline_line1="RAIN +300%",
        headline_line2="¿QUÉ ES?",
        emoji="🌧️",
        bg_color1=(80, 30, 100),   # purple dominant (not too blue)
        bg_color2=(50, 15, 70),
        accent_color=PURPLE,
        subtitle="Recompensas automáticas DeFi",
        video_type="short",
    ),
    # 8. 90% traders fail - dark red/gold warning  (SHORT → 1080x1920)
    dict(
        script_id="short_1774708646_182",
        title="90% de traders fracasa: La verdad",
        headline_line1="90% FRACASA",
        headline_line2="¿Y TÚ?",
        emoji="⚠️",
        bg_color1=(130, 30, 20),   # dark red
        bg_color2=(80, 15, 10),
        accent_color=RED,
        subtitle="La verdad que nadie dice",
        video_type="short",
    ),
]


def main():
    print("=" * 60)
    print("🎨 BitTrader Thumbnail Generator — Local PIL")
    print("=" * 60)
    
    targets = sys.argv[1:] if len(sys.argv) > 1 else None
    
    for v in VIDEOS:
        sid = v["script_id"]
        if targets and sid not in targets:
            continue
        print(f"\n📹 {v['title']}")
        try:
            path = generate_thumbnail(**v)  # video_type is passed via **v
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ Thumbnails generados")
    print("=" * 60)


if __name__ == "__main__":
    main()
