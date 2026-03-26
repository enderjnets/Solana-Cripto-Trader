#!/usr/bin/env python3
"""
BitTrader Thumbnail Generator & Uploader
Generates SDXL-powered thumbnails in official BitTrader style and uploads to YouTube.
"""

import os
import sys
import json
import time
import requests
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

# ─── Config ───────────────────────────────────────────────────────────────────
HF_API_KEY = "HF_TOKEN_REMOVED"
HF_ENDPOINTS = [
    "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0",
    "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell",
    "https://router.huggingface.co/hf-inference/models/runwayml/stable-diffusion-v1-5",
]

LOGO_PATH = "/home/enderj/.openclaw/workspace/bittrader/assets/bittrader_logo.png"
LOGO_FALLBACK = "/home/enderj/.openclaw/workspace/bittrader/assets/bittrader_logo_transparent.png"
CREDS_PATH = "/home/enderj/.openclaw/workspace/memory/youtube_credentials.json"
OUTPUT_DIR = "/home/enderj/.openclaw/workspace/bittrader/thumbnails/output"
FONT_BOLD = "/usr/share/fonts/truetype/space-grotesk/SpaceGrotesk-Bold.ttf"
FONT_REGULAR = "/usr/share/fonts/truetype/space-grotesk/SpaceGrotesk-Medium.ttf"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Videos ──────────────────────────────────────────────────────────────────
VIDEOS = [
    {
        "id": "ebIXqmJcI-c",
        "title": "Stop Loss: la herramienta que te salva de perder todo",
        "badge_color": (220, 30, 30),
        "badge_text": "STOP LOSS",
        "line1": "SALVA TU",
        "line1_color": "#F5A623",
        "line2": "CAPITAL",
        "line2_color": "#FFFFFF",
        "line3": "con Stop Loss",
        "line4": None,
        "line4_color": None,
        "sdxl_prompt": "worried young latin man looking at phone with red trading chart, dramatic expression, financial loss background, red and gold colors, professional studio lighting, 4K cinematic, high contrast, sharp focus, person on right side of frame",
    },
    {
        "id": "SPqnE13NLDA",
        "title": "TRUMP Coin: ¿Perdí $10,000?",
        "badge_color": (220, 30, 30),
        "badge_text": "TRUMP COIN",
        "line1": "PERDÍ",
        "line1_color": "#F5A623",
        "line2": "$10,000?",
        "line2_color": "#FFFFFF",
        "line3": "lo que pasó",
        "line4": None,
        "line4_color": None,
        "sdxl_prompt": "shocked young latin man holding money bills, dramatic expression looking at camera, red and gold colors, crypto charts background, professional studio lighting, 4K cinematic, person on right side of frame",
    },
    {
        "id": "ALd7fyYZj8s",
        "title": "El secreto del trading rentable que nadie te dice",
        "badge_color": (245, 166, 35),
        "badge_text": "SECRETO",
        "line1": "TRADING",
        "line1_color": "#F5A623",
        "line2": "RENTABLE",
        "line2_color": "#FFFFFF",
        "line3": "el secreto",
        "line4": None,
        "line4_color": None,
        "sdxl_prompt": "confident young latin trader man smiling at camera, green charts going up background, gold and green colors, professional studio lighting, 4K cinematic, high contrast, person on right side of frame",
    },
    {
        "id": "acGZBn1WNv8",
        "title": "Por qué SIEMPRE pierdes en crypto (la verdad)",
        "badge_color": (220, 30, 30),
        "badge_text": "LA VERDAD",
        "line1": "¿POR QUÉ",
        "line1_color": "#F5A623",
        "line2": "SIEMPRE PIERDES?",
        "line2_color": "#FFFFFF",
        "line3": "en crypto",
        "line4": None,
        "line4_color": None,
        "sdxl_prompt": "frustrated young latin man with hands on head, red crypto charts going down, dramatic expression, dark background, red and gold colors, 4K cinematic, high contrast, person on right side of frame",
    },
    {
        "id": "ynJgdQR_he8",
        "title": "Por qué SIEMPRE pierdes en crypto (la verdad) v2",
        "badge_color": (220, 30, 30),
        "badge_text": "LA VERDAD",
        "line1": "¿POR QUÉ",
        "line1_color": "#F5A623",
        "line2": "SIEMPRE PIERDES?",
        "line2_color": "#FFFFFF",
        "line3": "en crypto",
        "line4": None,
        "line4_color": None,
        "sdxl_prompt": "shocked young latin man pointing at falling crypto chart on screen, dramatic expression, red background, professional studio lighting, 4K cinematic, person on right side of frame",
    },
    {
        "id": "lBGs1YG33EA",
        "title": "El MIEDO te hace perder dinero en crypto",
        "badge_color": (230, 120, 20),
        "badge_text": "PSICOLOGÍA",
        "line1": "EL MIEDO",
        "line1_color": "#F5A623",
        "line2": "TE ARRUINA",
        "line2_color": "#FFFFFF",
        "line3": "en crypto",
        "line4": None,
        "line4_color": None,
        "sdxl_prompt": "scared nervous young latin man covering face, dramatic expression, red and dark background, crypto fear greed index, professional studio lighting, 4K cinematic, person on right side of frame",
    },
    {
        "id": "n7Tv2u42Kz8",
        "title": "Por qué el 90% de Traders Pierde Todo",
        "badge_color": (220, 30, 30),
        "badge_text": "90% PIERDEN",
        "line1": "90%",
        "line1_color": "#F5A623",
        "line2": "PIERDE TODO",
        "line2_color": "#FFFFFF",
        "line3": "¿y tú?",
        "line4": None,
        "line4_color": None,
        "sdxl_prompt": "shocked young latin man looking at camera with disbelief expression, trading charts red going down background, dark colors, professional studio lighting, 4K cinematic, high contrast, person on right side of frame",
    },
    {
        "id": "StuJJhBtP7Q",
        "title": "Por qué Bitcoin llegará a $1,000,000 — y cuándo",
        "badge_color": (30, 160, 60),
        "badge_text": "BITCOIN",
        "line1": "$1,000,000",
        "line1_color": "#F5A623",
        "line2": "BITCOIN",
        "line2_color": "#FFFFFF",
        "line3": "¿cuándo?",
        "line4": None,
        "line4_color": None,
        "sdxl_prompt": "excited young latin man pointing up with big smile, bitcoin gold coins raining, green and gold colors, professional studio lighting, 4K cinematic, high contrast, sharp focus, person on right side of frame",
    },
]

# ─── Font helpers ─────────────────────────────────────────────────────────────
def get_font(size, bold=True):
    font_path = FONT_BOLD if bold else FONT_REGULAR
    fallbacks = [
        font_path,
        "/usr/share/fonts/truetype/space-grotesk/SpaceGrotesk-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for f in fallbacks:
        try:
            return ImageFont.truetype(f, size)
        except:
            continue
    return ImageFont.load_default()


def hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def draw_text_with_stroke(draw, pos, text, font, fill_color, stroke_width=12, stroke_color=(0, 0, 0)):
    x, y = pos
    # Draw stroke
    for dx in range(-stroke_width, stroke_width + 1, 2):
        for dy in range(-stroke_width, stroke_width + 1, 2):
            if dx*dx + dy*dy <= stroke_width*stroke_width:
                draw.text((x + dx, y + dy), text, font=font, fill=stroke_color)
    # Draw fill
    draw.text((x, y), text, font=font, fill=fill_color)


# ─── SDXL generation ─────────────────────────────────────────────────────────
def generate_person_image(prompt, width=512, height=768):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    for endpoint in HF_ENDPOINTS:
        for attempt in range(2):
            try:
                print(f"  [SDXL] Trying {endpoint.split('/')[-1]} (attempt {attempt+1})...")
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "width": width,
                        "height": height,
                        "num_inference_steps": 20,
                        "guidance_scale": 7.5,
                    }
                }
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=120)
                
                if resp.status_code == 200:
                    img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
                    brightness = get_brightness(img)
                    print(f"  [SDXL] Generated OK, brightness={brightness:.1f}")
                    return img
                elif resp.status_code == 503:
                    print(f"  [SDXL] Model loading (503), waiting 20s...")
                    time.sleep(20)
                else:
                    print(f"  [SDXL] Error {resp.status_code}: {resp.text[:200]}")
                    time.sleep(5)
            except Exception as e:
                print(f"  [SDXL] Exception: {e}")
                time.sleep(5)
    
    print("  [SDXL] All endpoints failed, using gradient placeholder")
    return None


def get_brightness(img):
    """Return average brightness 0-255."""
    gray = img.convert("L")
    arr = np.array(gray, dtype=float)
    return arr.mean()


# ─── Thumbnail builder ───────────────────────────────────────────────────────
def build_thumbnail(video, person_img):
    W, H = 1280, 720
    
    # 1. Dark background gradient (dark navy → black)
    bg = Image.new("RGB", (W, H), (5, 8, 20))
    draw_bg = ImageDraw.Draw(bg)
    
    # Subtle blue-to-black diagonal gradient
    for y in range(H):
        ratio = y / H
        r = int(5 + ratio * 3)
        g = int(8 + ratio * 2)
        b = int(20 + (1 - ratio) * 15)
        draw_bg.rectangle([(0, y), (W, y+1)], fill=(r, g, b))
    
    # 2. Paste person on RIGHT side (40-50% width)
    if person_img:
        person_w = int(W * 0.50)
        person_h = H
        person_resized = person_img.resize((person_w, person_h), Image.LANCZOS)
        
        # Place on right side
        bg.paste(person_resized.convert("RGB"), (W - person_w, 0))
        
        # Blend the left edge of person with gradient for smooth transition
        blend_width = int(W * 0.22)
        for x in range(blend_width):
            alpha = x / blend_width
            px_x = W - person_w + x
            for y in range(H):
                try:
                    px = bg.getpixel((px_x, y))
                    orig_px = person_resized.getpixel((x, y))
                    if isinstance(orig_px, tuple) and len(orig_px) == 4:
                        orig_px = orig_px[:3]
                    blended = tuple(int(orig_px[c] * alpha + px[c] * (1-alpha)) for c in range(3))
                    bg.putpixel((px_x, y), blended)
                except:
                    pass
    
    # 3. Left overlay gradient (dark → transparent) for text readability
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    left_zone = int(W * 0.62)
    for x in range(left_zone):
        ratio = 1.0 - (x / left_zone)
        alpha = int(200 * (ratio ** 0.7))
        draw_ov.rectangle([(x, 0), (x+1, H)], fill=(0, 0, 0, alpha))
    
    bg_rgba = bg.convert("RGBA")
    bg_rgba.alpha_composite(overlay)
    img = bg_rgba.convert("RGB")
    
    draw = ImageDraw.Draw(img)
    
    # 4. Badge (top-left corner)
    badge_color = video["badge_color"]
    badge_text = video["badge_text"]
    badge_font = get_font(28, bold=True)
    
    # Measure badge text
    try:
        bbox = badge_font.getbbox(badge_text)
        bw = bbox[2] - bbox[0] + 30
        bh = bbox[3] - bbox[1] + 14
    except:
        bw, bh = len(badge_text) * 18 + 30, 46
    
    bx, by = 30, 30
    # Badge with rounded-ish corners (just filled rect for simplicity)
    draw.rectangle([(bx, by), (bx + bw, by + bh)], fill=badge_color)
    draw.text((bx + 15, by + 7), badge_text, font=badge_font, fill=(0, 0, 0))
    
    # 5. Text hierarchy (left side)
    text_x = 42
    text_y_start = 120
    
    # Line 1 — large hook word, GOLDEN
    font1 = get_font(110, bold=True)
    line1 = video["line1"]
    draw_text_with_stroke(draw, (text_x, text_y_start), line1, font1,
                          fill_color=hex_to_rgb(video["line1_color"]), stroke_width=12)
    
    try:
        bbox1 = font1.getbbox(line1)
        y_after1 = text_y_start + (bbox1[3] - bbox1[1]) + 8
    except:
        y_after1 = text_y_start + 115
    
    # Line 2 — complement, WHITE
    font2 = get_font(88, bold=True)
    line2 = video["line2"]
    # If line2 is too long, reduce font
    try:
        bbox2_test = font2.getbbox(line2)
        if (bbox2_test[2] - bbox2_test[0]) > 560:
            font2 = get_font(68, bold=True)
    except:
        pass
    draw_text_with_stroke(draw, (text_x, y_after1), line2, font2,
                          fill_color=hex_to_rgb(video["line2_color"]), stroke_width=11)
    
    try:
        bbox2 = font2.getbbox(line2)
        y_after2 = y_after1 + (bbox2[3] - bbox2[1]) + 10
    except:
        y_after2 = y_after1 + 90
    
    # Line 3 — short phrase, white, smaller
    if video.get("line3"):
        font3 = get_font(50, bold=False)
        draw_text_with_stroke(draw, (text_x, y_after2), video["line3"], font3,
                              fill_color=(255, 255, 255), stroke_width=9)
        try:
            bbox3 = font3.getbbox(video["line3"])
            y_after3 = y_after2 + (bbox3[3] - bbox3[1]) + 8
        except:
            y_after3 = y_after2 + 55
    else:
        y_after3 = y_after2
    
    # Line 4 — subtexto (optional)
    if video.get("line4") and video.get("line4_color"):
        font4 = get_font(40, bold=False)
        draw_text_with_stroke(draw, (text_x, y_after3), video["line4"], font4,
                              fill_color=hex_to_rgb(video["line4_color"]), stroke_width=8)
    
    # 6. @bittrader9259 top-right
    handle_font = get_font(32, bold=True)
    handle = "@bittrader9259"
    try:
        hbbox = handle_font.getbbox(handle)
        hw = hbbox[2] - hbbox[0]
    except:
        hw = len(handle) * 20
    hx = W - hw - 40
    draw_text_with_stroke(draw, (hx, 30), handle, handle_font,
                          fill_color=(255, 255, 255), stroke_width=8)
    
    # 7. BitTrader logo (bottom-left)
    logo_path = LOGO_PATH if os.path.exists(LOGO_PATH) else LOGO_FALLBACK
    if os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path).convert("RGBA")
            logo_h = 60
            ratio = logo_h / logo.height
            logo_w = int(logo.width * ratio)
            logo_resized = logo.resize((logo_w, logo_h), Image.LANCZOS)
            lx, ly = 30, H - logo_h - 25
            img.paste(logo_resized.convert("RGB"), (lx, ly), logo_resized.split()[3])
        except Exception as e:
            print(f"  [Logo] Could not paste logo: {e}")
    
    # 8. Gold border
    border_color = (245, 166, 35)
    border_w = 7
    draw_final = ImageDraw.Draw(img)
    for i in range(border_w):
        draw_final.rectangle([(i, i), (W-1-i, H-1-i)], outline=border_color)
    
    return img


# ─── YouTube upload ───────────────────────────────────────────────────────────
def get_youtube_client():
    with open(CREDS_PATH) as f:
        data = json.load(f)
    
    creds = Credentials(
        token=data.get("token") or data.get("access_token"),
        refresh_token=data.get("refresh_token"),
        token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=data.get("client_id"),
        client_secret=data.get("client_secret"),
        scopes=data.get("scopes", ["https://www.googleapis.com/auth/youtube.upload"]),
    )
    return build("youtube", "v3", credentials=creds)


def upload_thumbnail(youtube, video_id, thumb_path):
    media = MediaFileUpload(thumb_path, mimetype="image/jpeg")
    youtube.thumbnails().set(videoId=video_id, media_body=media).execute()


# ─── Main loop ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("BitTrader Thumbnail Generator v2.0")
    print("=" * 60)
    
    youtube = get_youtube_client()
    print("✅ YouTube client ready\n")
    
    results = []
    
    for i, video in enumerate(VIDEOS):
        vid_id = video["id"]
        title = video["title"]
        print(f"\n[{i+1}/8] {vid_id} — {title[:50]}")
        
        attempts = 0
        success = False
        brightness = 0
        
        while attempts < 3 and not success:
            attempts += 1
            try:
                # Step 1: Generate person
                print(f"  Generating person image (attempt {attempts})...")
                person_img = generate_person_image(video["sdxl_prompt"])
                
                # Step 2: Build thumbnail
                print(f"  Building thumbnail...")
                thumb = build_thumbnail(video, person_img)
                
                # Step 3: Check brightness
                brightness = get_brightness(thumb)
                print(f"  Brightness: {brightness:.1f}")
                
                if brightness < 30:
                    print(f"  ⚠️  Brightness {brightness:.1f} < 30, brightening...")
                    # Boost brightness
                    from PIL import ImageEnhance
                    enhancer = ImageEnhance.Brightness(thumb)
                    thumb = enhancer.enhance(1.5)
                    brightness = get_brightness(thumb)
                    print(f"  Brightness after boost: {brightness:.1f}")
                
                # Save
                thumb_path = os.path.join(OUTPUT_DIR, f"{vid_id}.jpg")
                thumb.save(thumb_path, "JPEG", quality=95)
                print(f"  Saved: {thumb_path}")
                
                # Step 4: Upload
                print(f"  Uploading to YouTube...")
                upload_thumbnail(youtube, vid_id, thumb_path)
                print(f"  ✅ Uploaded!")
                success = True
                
            except HttpError as e:
                err_str = str(e)
                print(f"  ❌ YouTube API error: {err_str[:200]}")
                if "thumbnailsNotAllowed" in err_str or "forbidden" in err_str.lower():
                    print(f"  ⚠️  Channel not eligible for custom thumbnails on this video")
                    results.append({
                        "id": vid_id,
                        "title": title[:40],
                        "brightness": round(brightness, 1),
                        "status": "FAIL (YT restriction)"
                    })
                    break
                time.sleep(5)
            except Exception as e:
                print(f"  ❌ Error (attempt {attempts}): {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
        
        if success:
            results.append({
                "id": vid_id,
                "title": title[:40],
                "brightness": round(brightness, 1),
                "status": "✅ UPLOADED"
            })
        elif not any(r["id"] == vid_id for r in results):
            results.append({
                "id": vid_id,
                "title": title[:40],
                "brightness": round(brightness, 1),
                "status": f"❌ FAILED ({attempts} attempts)"
            })
    
    # Final report
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(f"{'Video ID':<16} {'Title':<42} {'Brightness':>10}  Status")
    print("-" * 85)
    for r in results:
        print(f"{r['id']:<16} {r['title']:<42} {str(r['brightness']):>10}  {r['status']}")
    
    uploaded = sum(1 for r in results if "UPLOADED" in r["status"])
    print(f"\n{uploaded}/{len(results)} thumbnails uploaded successfully.")
    
    return results


if __name__ == "__main__":
    main()
