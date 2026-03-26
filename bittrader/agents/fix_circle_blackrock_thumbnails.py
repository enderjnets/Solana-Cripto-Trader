#!/usr/bin/env python3
"""
🔧 Fix Circle/BlackRock Thumbnails — Genera y sube thumbnails a los 4 videos afectados
Generadas con HF SDXL + persona real. Sin frames del video de narración.

Videos afectados:
  - Circle vence a BlackRock en $11B  → -eAvB_OwWyc
  - Circle Supera a BlackRock con $11B → qfolzWqCX78
  - Circle le ganó a BlackRock en $11B → l15HmH7LWGQ
  - Circle tiene $11B en bonos y BlackRock PIERDE → gX9s88XCAlg
"""
import json
import sys
import os
import requests
import tempfile
import time
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("❌ Pillow no instalado. Ejecuta: pip3 install Pillow")
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE  = Path("/home/enderj/.openclaw/workspace")
BITTRADER  = WORKSPACE / "bittrader"
DATA_DIR   = BITTRADER / "agents/data"
THUMB_DIR  = DATA_DIR / "thumbnails"
THUMB_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(WORKSPACE / "youtube_env/lib/python3.13/site-packages"))

YT_CREDS  = WORKSPACE / "memory/youtube_credentials.json"
LOGO_PATH = WORKSPACE / "videos/BIBLIOTECA/bittrader_logo.png"

# ── HuggingFace Config ──────────────────────────────────────────────────────
HF_API_KEY   = "HF_TOKEN_REMOVED"
HF_ENDPOINT  = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"

# ── Videos to fix ──────────────────────────────────────────────────────────
VIDEOS = [
    {
        "video_id": "-eAvB_OwWyc",
        "title": "Circle vence a BlackRock en $11B",
        "headline": "CIRCLE\nVENCE A\nBLACKROCK",
        "subtitle": "$11 Billones en bonos del Tesoro",
    },
    {
        "video_id": "qfolzWqCX78",
        "title": "Circle Supera a BlackRock con $11B",
        "headline": "CIRCLE\nSUPERA A\nBLACKROCK",
        "subtitle": "El gigante cripto tiene $11B en bonos",
    },
    {
        "video_id": "l15HmH7LWGQ",
        "title": "Circle le ganó a BlackRock en $11B",
        "headline": "CIRCLE\nLE GANÓ A\nBLACKROCK",
        "subtitle": "¿Quién realmente controla los bonos?",
    },
    {
        "video_id": "gX9s88XCAlg",
        "title": "Circle tiene $11B en bonos y BlackRock PIERDE",
        "headline": "BLACKROCK\nPIERDE\nCONTRA CIRCLE",
        "subtitle": "$11B en bonos del Tesoro USA",
    },
]

# ── Brand Colors ────────────────────────────────────────────────────────────
GOLD  = (245, 166, 35)
WHITE = (255, 255, 255)
BLACK = (15, 15, 15)
RED   = (220, 40, 40)
GREEN = (40, 200, 80)

THUMB_W, THUMB_H = 1280, 720


# ══════════════════════════════════════════════════════════════════════════
# FONT HELPERS
# ══════════════════════════════════════════════════════════════════════════

def get_font(size: int) -> ImageFont.FreeTypeFont:
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# ══════════════════════════════════════════════════════════════════════════
# GRADIENT BACKGROUND
# ══════════════════════════════════════════════════════════════════════════

def create_gradient_background() -> Image.Image:
    """Dark navy gradient — no video frames ever."""
    bg = Image.new("RGB", (THUMB_W, THUMB_H))
    draw = ImageDraw.Draw(bg)
    for y in range(THUMB_H):
        ratio = y / THUMB_H
        r = int(5 + 15 * (1 - ratio))
        g = int(5 + 20 * (1 - ratio))
        b = int(20 + 60 * (1 - ratio))
        draw.line([(0, y), (THUMB_W, y)], fill=(r, g, b))
    return bg


# ══════════════════════════════════════════════════════════════════════════
# HF SDXL PERSON IMAGE
# ══════════════════════════════════════════════════════════════════════════

def generate_person_image(output_path: Path, prompt: str = None) -> bool:
    """Generate person image via HuggingFace SDXL."""
    if prompt is None:
        prompt = (
            "shocked surprised businessman looking at phone screen, dramatic expression, "
            "stock market charts green going up, gold and green colors, professional studio "
            "lighting, 4K cinematic, high contrast, sharp focus"
        )
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {"width": 512, "height": 512, "num_inference_steps": 30},
    }
    print(f"  🤖 Generando persona con HF SDXL...")
    try:
        resp = requests.post(HF_ENDPOINT, headers=headers, json=payload, timeout=120)
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image"):
            output_path.write_bytes(resp.content)
            kb = output_path.stat().st_size / 1024
            print(f"  ✅ Imagen HF: {kb:.0f}KB → {output_path.name}")
            return True
        else:
            print(f"  ⚠️ HF error {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  ⚠️ HF SDXL exception: {e}")
    return False


# ══════════════════════════════════════════════════════════════════════════
# THUMBNAIL COMPOSER
# ══════════════════════════════════════════════════════════════════════════

def compose_thumbnail(
    headline: str,
    subtitle: str,
    person_path: Path,
    output_path: Path,
    title_color: tuple = GOLD,
) -> bool:
    """Compose a 1280×720 BitTrader thumbnail with person on right side."""

    # Gradient background
    bg = create_gradient_background()

    # Paste person image on right side
    if person_path and person_path.exists():
        try:
            person = Image.open(person_path).convert("RGBA")
            # Scale to full height
            person_h = THUMB_H
            person_w = int(person.width * (person_h / person.height))
            person = person.resize((person_w, person_h), Image.LANCZOS)
            # Position: right half
            paste_x = max(THUMB_W - person_w, THUMB_W // 2 - 60)
            bg_rgba = bg.convert("RGBA")
            bg_rgba.paste(person, (paste_x, 0), person)
            bg = bg_rgba.convert("RGB")
        except Exception as e:
            print(f"  ⚠️ Error pegando persona: {e}")

    draw = ImageDraw.Draw(bg)

    # Darken left 55% for text legibility
    overlay = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    for x in range(THUMB_W * 55 // 100):
        alpha = int(190 * (1 - x / (THUMB_W * 0.55)))
        ov_draw.line([(x, 0), (x, THUMB_H)], fill=(0, 0, 0, alpha))
    bg = Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(bg)

    # Gold diagonal accent
    overlay2 = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    ov2 = ImageDraw.Draw(overlay2)
    ov2.polygon(
        [(THUMB_W * 0.52, THUMB_H), (THUMB_W, THUMB_H * 0.20), (THUMB_W, THUMB_H)],
        fill=(*GOLD, 35),
    )
    bg = Image.alpha_composite(bg.convert("RGBA"), overlay2).convert("RGB")
    draw = ImageDraw.Draw(bg)

    # Logo top-left
    if LOGO_PATH.exists():
        try:
            logo = Image.open(LOGO_PATH).convert("RGBA")
            logo_h = 70
            logo_w = int(logo.width * (logo_h / logo.height))
            logo = logo.resize((logo_w, logo_h), Image.LANCZOS)
            bg.paste(logo, (20, 15), logo)
            draw = ImageDraw.Draw(bg)
        except Exception:
            pass

    # @bittrader9259 top-right
    font_handle = get_font(26)
    handle_text = "@bittrader9259"
    try:
        bbox = draw.textbbox((0, 0), handle_text, font=font_handle)
        handle_w = bbox[2] - bbox[0]
    except Exception:
        handle_w = 180
    draw.text(
        (THUMB_W - handle_w - 20, 20),
        handle_text,
        font=font_handle,
        fill=WHITE,
        stroke_width=2,
        stroke_fill=BLACK,
    )

    # Main headline (multi-line, left side)
    lines = [l.strip() for l in headline.strip().split("\n") if l.strip()]
    font_big = get_font(95 if len(lines) <= 3 else 80)
    y = 95
    line_h = (95 if len(lines) <= 3 else 80) + 12
    for line in lines[:4]:
        draw.text(
            (35, y),
            line,
            font=font_big,
            fill=title_color,
            stroke_width=4,
            stroke_fill=BLACK,
        )
        y += line_h

    # Subtitle
    if subtitle:
        font_sub = get_font(32)
        draw.text(
            (35, y + 15),
            subtitle,
            font=font_sub,
            fill=WHITE,
            stroke_width=2,
            stroke_fill=BLACK,
        )

    # Handle bottom-left
    draw.text(
        (35, THUMB_H - 100),
        "@bittrader9259",
        font=get_font(26),
        fill=WHITE,
        stroke_width=2,
        stroke_fill=BLACK,
    )

    # WATCH NOW button
    btn_y = THUMB_H - 70
    draw.rectangle([35, btn_y, 260, btn_y + 45], fill=GOLD)
    draw.rectangle([35, btn_y, 75, btn_y + 45], fill=RED)
    font_btn = get_font(28)
    draw.text((42, btn_y + 8), "▶", font=font_btn, fill=WHITE)
    draw.text((82, btn_y + 10), "WATCH NOW", font=font_btn, fill=BLACK)

    # Save
    bg.save(str(output_path), "JPEG", quality=95)
    size_kb = output_path.stat().st_size / 1024
    if size_kb > 2048:
        bg.save(str(output_path), "JPEG", quality=80)
        size_kb = output_path.stat().st_size / 1024

    # Validate brightness
    img_check = Image.open(output_path).convert("RGB")
    pixels = list(img_check.getdata())
    avg_brightness = sum(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2] for p in pixels) / len(pixels)
    if avg_brightness < 30:
        print(f"  ❌ Thumbnail demasiado oscura (brightness={avg_brightness:.1f}) — rechazada")
        return False

    print(f"  ✅ Thumbnail: {size_kb:.0f}KB, brightness={avg_brightness:.1f} → {output_path.name}")
    return True


# ══════════════════════════════════════════════════════════════════════════
# YOUTUBE CLIENT
# ══════════════════════════════════════════════════════════════════════════

def get_youtube_client():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    d = json.loads(YT_CREDS.read_text())
    scopes_value = d.get("scopes") or d.get("scope", "")
    if isinstance(scopes_value, str):
        scopes_value = scopes_value.split()

    creds = Credentials(
        token=d.get("token") or d.get("access_token"),
        refresh_token=d["refresh_token"],
        token_uri=d["token_uri"],
        client_id=d["client_id"],
        client_secret=d["client_secret"],
        scopes=scopes_value,
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        d["access_token"] = creds.token
        YT_CREDS.write_text(json.dumps(d, indent=2))
    return build("youtube", "v3", credentials=creds)


def upload_thumbnail(yt, video_id: str, thumb_path: Path) -> bool:
    """Upload thumbnail with 2 retries."""
    from googleapiclient.http import MediaFileUpload

    if not thumb_path.exists():
        print(f"  ❌ Thumbnail no existe: {thumb_path}")
        return False

    for attempt in range(1, 4):
        try:
            media = MediaFileUpload(str(thumb_path), mimetype="image/jpeg")
            yt.thumbnails().set(videoId=video_id, media_body=media).execute()
            print(f"  ✅ Thumbnail subida a YouTube: {video_id} (intento {attempt})")
            return True
        except Exception as e:
            print(f"  ⚠️ Intento {attempt}/3 falló: {str(e)[:120]}")
            if attempt < 3:
                time.sleep(5 * attempt)
    print(f"  ❌ Thumbnail FALLÓ para {video_id} tras 3 intentos")
    return False


def verify_thumbnail_on_youtube(yt, video_id: str) -> bool:
    """Verify via YouTube API that the video has a custom thumbnail (maxres)."""
    try:
        resp = yt.videos().list(part="snippet", id=video_id).execute()
        items = resp.get("items", [])
        if items:
            thumbnails = items[0]["snippet"].get("thumbnails", {})
            has_custom = "maxres" in thumbnails
            print(f"  {'✅' if has_custom else '⚠️'} YouTube API: {'thumbnail custom confirmada' if has_custom else 'SIN thumbnail custom'} → {video_id}")
            return has_custom
    except Exception as e:
        print(f"  ⚠️ Error verificando via API: {e}")
    return False


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("\n🔧 Fix Circle/BlackRock Thumbnails — BitTrader")
    print("=" * 60)

    # Connect YouTube
    print("\n🔗 Conectando a YouTube API...")
    try:
        yt = get_youtube_client()
        print("  ✅ Conectado")
    except Exception as e:
        print(f"  ❌ YouTube auth error: {e}")
        sys.exit(1)

    # Generate shared person image (reuse for all 4 videos)
    print("\n🤖 Generando imagen de persona con HF SDXL...")
    person_path = THUMB_DIR / "circle_blackrock_person.jpg"
    if person_path.exists() and person_path.stat().st_size > 5000:
        print(f"  ⏭️ Ya existe: {person_path.name}")
    else:
        ok = generate_person_image(person_path)
        if not ok:
            print("  ⚠️ HF SDXL falló — continuando sin imagen de persona")

    results = []
    title_colors = [GOLD, GREEN, GOLD, (0, 200, 255)]

    for i, video in enumerate(VIDEOS):
        video_id = video["video_id"]
        title    = video["title"]
        headline = video["headline"]
        subtitle = video["subtitle"]
        color    = title_colors[i % len(title_colors)]

        print(f"\n[{i+1}/{len(VIDEOS)}] {title}")
        print(f"        ID: {video_id}")

        # Generate thumbnail
        thumb_path = THUMB_DIR / f"{video_id}_thumbnail.jpg"
        print(f"  🎨 Generando thumbnail...")
        ok = compose_thumbnail(
            headline=headline,
            subtitle=subtitle,
            person_path=person_path if person_path.exists() else None,
            output_path=thumb_path,
            title_color=color,
        )

        if not ok:
            print(f"  ❌ Thumbnail no pasó validación — saltando")
            results.append({"video_id": video_id, "title": title, "status": "thumb_failed"})
            continue

        # Upload to YouTube (with retries)
        print(f"  📤 Subiendo thumbnail a YouTube...")
        uploaded = upload_thumbnail(yt, video_id, thumb_path)

        # Wait a moment then verify via API
        if uploaded:
            time.sleep(5)
            verified = verify_thumbnail_on_youtube(yt, video_id)
        else:
            verified = False

        results.append({
            "video_id":  video_id,
            "title":     title,
            "thumb_path": str(thumb_path),
            "uploaded":  uploaded,
            "verified":  verified,
            "status":    "success" if uploaded and verified else (
                         "uploaded_unverified" if uploaded else "failed"),
        })

        # Rate limit between videos
        if i < len(VIDEOS) - 1:
            time.sleep(3)

    # Summary
    print("\n" + "=" * 60)
    print("📊 RESUMEN")
    print("=" * 60)
    for r in results:
        icon = "✅" if r.get("status") == "success" else (
               "⚠️" if r.get("status") == "uploaded_unverified" else "❌")
        print(f"  {icon} [{r['video_id']}] {r['title'][:50]}")
        print(f"       status={r['status']}")
        if r.get("thumb_path"):
            print(f"       thumb={Path(r['thumb_path']).name}")

    # Save results
    result_file = DATA_DIR / "circle_blackrock_thumb_fix.json"
    result_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n💾 Resultados guardados → {result_file.name}")
    return results


if __name__ == "__main__":
    main()
