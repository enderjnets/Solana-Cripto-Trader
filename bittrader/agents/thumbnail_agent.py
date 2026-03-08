#!/usr/bin/env python3
"""
🎨 BitTrader Thumbnail Agent — Generador de Miniaturas
Genera thumbnails profesionales estilo BitTrader para cada video producido.

Pipeline: Producer → Thumbnail Agent → Quality Checker → Publisher

Estilo BitTrader:
- Fondo: frame extraído del clip IA (oscurecido a la izquierda)
- Texto: título grande dorado/verde con borde negro
- Subtítulo descriptivo blanco
- Logo BitTrader arriba izquierda
- Handle @bittrader9259 abajo izquierda
- Botón "WATCH NOW" dorado
- Diagonal dorada decorativa

Ejecutar: python3 agents/thumbnail_agent.py [--limit N]
"""
import json
import os
import subprocess
import random
import requests
from datetime import datetime, timezone
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("❌ Pillow no instalado. Ejecuta: pip3 install Pillow")
    exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE  = Path("/home/enderj/.openclaw/workspace")
BITTRADER  = WORKSPACE / "bittrader"
DATA_DIR   = BITTRADER / "agents/data"
OUTPUT_DIR = BITTRADER / "agents/output"

PRODUCTION_FILE = DATA_DIR / "production_latest.json"
LOGO_PATH       = WORKSPACE / "videos/BIBLIOTECA/bittrader_logo.png"

# ── Thumbnail Specs ────────────────────────────────────────────────────────
THUMB_W, THUMB_H = 1280, 720  # YouTube standard
THUMB_MAX_KB     = 2048        # YouTube max 2MB

# ── Colors (BitTrader brand) ──────────────────────────────────────────────
GOLD      = (245, 166, 35)
WHITE     = (255, 255, 255)
BLACK     = (15, 15, 15)
RED       = (220, 40, 40)
GREEN     = (40, 200, 80)
CYAN      = (0, 200, 255)

TITLE_COLORS = [GOLD, GREEN, CYAN, GOLD, GOLD]  # Rotate for variety

# ── LLM Config (for generating catchy thumbnail text) ─────────────────────
CLAUDE_BASE_URL = "http://127.0.0.1:8443/v1/messages"
CLAUDE_MODEL    = "claude-sonnet-4-6"


# ════════════════════════════════════════════════════════════════════════
# FONT HELPERS
# ════════════════════════════════════════════════════════════════════════

def get_font(size: int) -> ImageFont.FreeTypeFont:
    """Get best available bold font."""
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def wrap_text(text: str, max_chars: int = 16) -> list:
    """Split text into lines that fit the thumbnail."""
    words = text.upper().split()
    lines = []
    current = ""
    for word in words:
        if len(current + " " + word) <= max_chars:
            current = (current + " " + word).strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:4]  # Max 4 lines


# ════════════════════════════════════════════════════════════════════════
# LLM — Generate catchy thumbnail text
# ════════════════════════════════════════════════════════════════════════

def generate_thumb_text(title: str, description: str = "") -> dict:
    """Ask LLM for catchy thumbnail headline + subtitle."""
    try:
        headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        prompt = (
            f"Para un video de YouTube sobre trading/crypto con título: \"{title}\"\n"
            f"Genera un texto CORTO para la miniatura (thumbnail) del video.\n\n"
            f"Responde en JSON exacto:\n"
            f'{{"headline": "TEXTO GRANDE (max 3 palabras por línea, max 3 líneas)", '
            f'"subtitle": "frase corta descriptiva (max 8 palabras)"}}\n\n'
            f"REGLAS:\n"
            f"- Headline: impactante, MAYÚSCULAS, 2-3 líneas cortas\n"
            f"- Subtitle: complementa el headline, genera curiosidad\n"
            f"- Estilo YouTube clickbait pero informativo\n"
            f"- En español"
        )
        data = {
            "model": CLAUDE_MODEL, "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}]
        }
        r = requests.post(CLAUDE_BASE_URL, headers=headers, json=data, timeout=30)
        if r.status_code == 200:
            text = r.json().get("content", [{}])[0].get("text", "")
            import re
            json_match = re.search(r'\{.*?\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
    except Exception as e:
        print(f"      ⚠️ LLM thumbnail text error: {e}")
    
    # Fallback: extract from title
    return {
        "headline": title[:40].upper(),
        "subtitle": ""
    }


# ════════════════════════════════════════════════════════════════════════
# THUMBNAIL GENERATION
# ════════════════════════════════════════════════════════════════════════

def extract_frame(video_path: Path, output_path: Path, time_sec: float = 2.0) -> bool:
    """Extract a frame from video at given timestamp."""
    result = subprocess.run([
        "ffmpeg", "-y", "-ss", str(time_sec), "-i", str(video_path),
        "-vframes", "1", "-q:v", "2", str(output_path)
    ], capture_output=True, text=True, timeout=10)
    return result.returncode == 0 and output_path.exists()


def create_thumbnail(
    title: str,
    subtitle: str,
    bg_image_path: Path,
    output_path: Path,
    title_color: tuple = GOLD,
    video_type: str = "long"
) -> bool:
    """Create a professional BitTrader thumbnail."""
    
    # ── Background ──
    if bg_image_path and bg_image_path.exists():
        bg = Image.open(bg_image_path).convert("RGB")
        bg = bg.resize((THUMB_W, THUMB_H), Image.LANCZOS)
        
        # Darken left side for text readability (gradient)
        overlay = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
        ov_draw = ImageDraw.Draw(overlay)
        gradient_width = THUMB_W * 2 // 3
        for x in range(gradient_width):
            alpha = int(210 * (1 - x / gradient_width))
            ov_draw.line([(x, 0), (x, THUMB_H)], fill=(0, 0, 0, alpha))
        bg = Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB")
    else:
        bg = Image.new("RGB", (THUMB_W, THUMB_H), BLACK)
    
    draw = ImageDraw.Draw(bg)
    
    # ── Diagonal gold accent ──
    overlay2 = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    ov2 = ImageDraw.Draw(overlay2)
    ov2.polygon(
        [(THUMB_W * 0.55, THUMB_H), (THUMB_W, THUMB_H * 0.25), (THUMB_W, THUMB_H)],
        fill=(*GOLD, 40)
    )
    bg = Image.alpha_composite(bg.convert("RGBA"), overlay2).convert("RGB")
    draw = ImageDraw.Draw(bg)
    
    # ── Logo (top left) ──
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
    
    # ── Main Title (big text, left side) ──
    lines = wrap_text(title)
    font_size = 95 if len(lines) <= 3 else 80
    font_big = get_font(font_size)
    
    y = 100
    line_height = font_size + 10
    for line in lines:
        draw.text((35, y), line, font=font_big, fill=title_color,
                  stroke_width=4, stroke_fill=BLACK)
        y += line_height
    
    # ── Subtitle ──
    if subtitle:
        font_sub = get_font(32)
        draw.text((35, y + 15), subtitle, font=font_sub, fill=WHITE,
                  stroke_width=2, stroke_fill=BLACK)
    
    # ── Handle @bittrader9259 ──
    font_handle = get_font(26)
    draw.text((35, THUMB_H - 100), "@bittrader9259", font=font_handle,
              fill=WHITE, stroke_width=2, stroke_fill=BLACK)
    
    # ── WATCH NOW button ──
    btn_y = THUMB_H - 70
    draw.rectangle([35, btn_y, 255, btn_y + 45], fill=GOLD)
    draw.rectangle([35, btn_y, 75, btn_y + 45], fill=RED)
    font_btn = get_font(28)
    draw.text((42, btn_y + 8), "▶", font=font_btn, fill=WHITE)
    draw.text((82, btn_y + 10), "WATCH NOW", font=font_btn, fill=BLACK)
    
    # ── Save ──
    bg.save(str(output_path), "JPEG", quality=95)
    
    # Check size (YouTube max 2MB)
    size_kb = output_path.stat().st_size / 1024
    if size_kb > THUMB_MAX_KB:
        # Re-save with lower quality
        bg.save(str(output_path), "JPEG", quality=80)
    
    return output_path.exists()


# ════════════════════════════════════════════════════════════════════════
# MAIN WORKFLOW
# ════════════════════════════════════════════════════════════════════════

def run_thumbnail_agent(limit: int = 10, types: str = "all") -> dict:
    """Generate thumbnails for all produced videos."""
    print("\n🎨 BitTrader Thumbnail Agent v1.0")
    
    if not PRODUCTION_FILE.exists():
        print("  ⚠️ No production file found. Run producer first.")
        return {"generated": 0, "errors": 0}
    
    production = json.loads(PRODUCTION_FILE.read_text())
    videos = production.get("videos", [])
    
    # Filter by type if specified
    if types == "long":
        videos = [v for v in videos if v.get("type") == "long"]
    elif types == "short":
        videos = [v for v in videos if v.get("type") == "short"]
    
    print(f"  📋 {len(videos)} videos to process")
    
    generated = 0
    errors = 0
    results = []
    
    for i, video in enumerate(videos[:limit], 1):
        title = video.get("title", "BitTrader")
        vtype = video.get("type", "short")
        video_path = Path(video.get("output_file", ""))
        script_dir = video_path.parent if video_path.exists() else None
        
        thumb_path = script_dir / "thumbnail.jpg" if script_dir else None
        
        print(f"\n  [{i}/{len(videos)}] [{vtype}] {title[:50]}")
        
        # Skip if thumbnail already exists
        if thumb_path and thumb_path.exists():
            size_kb = thumb_path.stat().st_size / 1024
            print(f"    ⏭️ Ya existe ({size_kb:.0f}KB)")
            results.append({
                "title": title, "type": vtype,
                "thumbnail": str(thumb_path), "status": "exists"
            })
            continue
        
        if not script_dir:
            print(f"    ❌ No output directory")
            errors += 1
            continue
        
        # ── Step 1: Get catchy text from LLM ──
        print(f"    📝 Generando texto...")
        thumb_data = generate_thumb_text(title, video.get("description", ""))
        headline = thumb_data.get("headline", title[:40].upper())
        subtitle = thumb_data.get("subtitle", "")
        print(f"    📝 \"{headline}\" | \"{subtitle}\"")
        
        # ── Step 2: Extract background frame from video ──
        bg_frame = script_dir / "thumb_bg.jpg"
        if video_path.exists():
            # Extract frame at 3 seconds (after intro)
            extract_frame(video_path, bg_frame, time_sec=3.0)
        
        # ── Step 3: Generate thumbnail ──
        print(f"    🎨 Generando miniatura...")
        color_idx = i % len(TITLE_COLORS)
        success = create_thumbnail(
            title=headline,
            subtitle=subtitle,
            bg_image_path=bg_frame,
            output_path=thumb_path,
            title_color=TITLE_COLORS[color_idx],
            video_type=vtype
        )
        
        # Clean up bg frame
        if bg_frame.exists():
            bg_frame.unlink(missing_ok=True)
        
        if success:
            size_kb = thumb_path.stat().st_size / 1024
            print(f"    ✅ Thumbnail: {size_kb:.0f}KB → {thumb_path.name}")
            generated += 1
            
            # Update video entry
            video["thumbnail"] = str(thumb_path)
            
            results.append({
                "title": title, "type": vtype,
                "thumbnail": str(thumb_path),
                "size_kb": round(size_kb),
                "headline": headline,
                "subtitle": subtitle,
                "status": "generated"
            })
        else:
            print(f"    ❌ Falló generación")
            errors += 1
            results.append({
                "title": title, "type": vtype,
                "status": "error"
            })
    
    # Save updated production file with thumbnail paths
    production["videos"] = [v for v in production["videos"]]  # Keep all
    production["thumbnails"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": generated,
        "errors": errors,
    }
    PRODUCTION_FILE.write_text(json.dumps(production, indent=2, ensure_ascii=False))
    
    print(f"\n✅ Thumbnail Agent completado")
    print(f"   🎨 {generated} thumbnails generados")
    if errors:
        print(f"   ⚠️  {errors} errores")
    
    return {
        "generated": generated,
        "errors": errors,
        "results": results
    }


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BitTrader Thumbnail Agent v1.0")
    parser.add_argument("--limit", type=int, default=10, help="Max thumbnails to generate")
    parser.add_argument("--types", choices=["all", "long", "short"], default="all",
                       help="Generate for which video types")
    args = parser.parse_args()
    
    result = run_thumbnail_agent(limit=args.limit, types=args.types)
    
    print("\n── Thumbnails ──────────────────────")
    for r in result.get("results", []):
        status = "✅" if r["status"] in ("generated", "exists") else "❌"
        print(f"  {status} [{r.get('type','?')}] {r.get('title','?')[:45]} | "
              f"{r.get('size_kb', '?')}KB")
    print("─────────────────────────────────────────\n")
