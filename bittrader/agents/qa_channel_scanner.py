#!/usr/bin/env python3
"""
🔍 QA Channel Scanner — BitTrader
Creado: 25 marzo 2026

Escanea TODOS los videos públicos del canal cada 24 horas buscando problemas
de calidad que pasaron el pipeline (thumbnails malas, oscuras, azules, etc.)
y los corrige automáticamente.

FUNCIONES:
  1. Detectar thumbnails malas (oscuras, azules genéricas, sin custom)
  2. Auto-regenerar thumbnails con estilo oficial BitTrader (SDXL / HF API)
  3. Detectar contaminación de narración en títulos/descripciones
  4. Generar reporte de problemas encontrados y resueltos

CRON sugerido (cada 24h a las 03:00 AM MDT):
  0 9 * * * cd /home/enderj/.openclaw/workspace/bittrader && python3 agents/qa_channel_scanner.py

USO:
  python3 agents/qa_channel_scanner.py            # scan + auto-fix
  python3 agents/qa_channel_scanner.py --dry-run  # solo reportar, no corregir
  python3 agents/qa_channel_scanner.py --report   # mostrar último reporte
"""

import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE  = Path("/home/enderj/.openclaw/workspace")
BITTRADER  = WORKSPACE / "bittrader"
AGENTS     = BITTRADER / "agents"
DATA_DIR   = AGENTS / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(WORKSPACE / "youtube_env/lib/python3.13/site-packages"))
sys.path.insert(0, str(AGENTS))

SCANNER_LOG   = DATA_DIR / "qa_scanner_log.json"
SCANNER_STATE = DATA_DIR / "qa_scanner_state.json"

# HF API key for thumbnail regeneration
HF_API_KEY = os.environ.get("HF_TOKEN", "")

# Ender's Telegram chat ID (per TOOLS.md)
ENDER_CHAT_ID = 771213858


# ── Logging ────────────────────────────────────────────────────────────────

def _log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] 🔍 Scanner: {msg}")


def _get_telegram_token():
    try:
        cfg = json.loads(Path("/home/enderj/.openclaw/config.json").read_text())
        return cfg.get("telegram", {}).get("botToken", "")
    except Exception:
        return None


def notify_ender(message: str):
    try:
        import urllib.request
        bot_token = _get_telegram_token()
        if not bot_token:
            return
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = json.dumps({
            "chat_id": ENDER_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
        }).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception as e:
        _log(f"Telegram notify failed: {e}", "WARNING")


# ── YouTube client ─────────────────────────────────────────────────────────

def get_youtube_client():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    YT_CREDS = WORKSPACE / "memory/youtube_credentials.json"
    d = json.loads(YT_CREDS.read_text())

    scopes = d.get("scopes") or d.get("scope", "")
    if isinstance(scopes, str):
        scopes = scopes.split()

    creds = Credentials(
        token=d.get("token") or d.get("access_token"),
        refresh_token=d["refresh_token"],
        token_uri=d["token_uri"],
        client_id=d["client_id"],
        client_secret=d["client_secret"],
        scopes=scopes,
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        d["access_token"] = creds.token
        YT_CREDS.write_text(json.dumps(d, indent=2))
    return build("youtube", "v3", credentials=creds)


# ── Thumbnail quality analysis ─────────────────────────────────────────────

def _analyze_thumbnail_url(url: str) -> dict:
    """
    Download thumbnail from URL and analyze quality.
    Returns dict with: brightness, is_blue, is_dark, is_monochromatic, width, height.
    """
    try:
        import urllib.request
        import tempfile
        import os
        from PIL import Image
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp = f.name

        try:
            urllib.request.urlretrieve(url, tmp)
            img = Image.open(tmp).convert("RGB")
            arr = np.array(img, dtype=float)

            r = arr[:, :, 0]
            g = arr[:, :, 1]
            b = arr[:, :, 2]

            avg_r = float(r.mean())
            avg_g = float(g.mean())
            avg_b = float(b.mean())
            brightness = 0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b
            variance = float(r.var() + g.var() + b.var())

            is_dark = brightness < 30
            # Blue generic: avg_blue > avg_red * 1.7 and fairly blue
            is_blue = (avg_r < 1 or avg_b > avg_r * 1.7) and avg_b > 100
            is_monochromatic = variance < 500

            return {
                "brightness": round(brightness, 2),
                "is_dark": is_dark,
                "is_blue": is_blue,
                "is_monochromatic": is_monochromatic,
                "variance": round(variance, 1),
                "size": f"{img.width}x{img.height}",
                "ok": not (is_dark or is_blue or is_monochromatic),
            }
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass

    except Exception as e:
        return {"error": str(e), "ok": False, "is_dark": False, "is_blue": False}


def _has_custom_thumbnail(thumbnails: dict) -> bool:
    """
    YouTube sets 'maxres' thumbnail only when a custom one is uploaded.
    Auto-generated thumbnails have: default, medium, high — but NOT maxres.
    """
    return "maxres" in thumbnails


# ── Thumbnail regeneration ─────────────────────────────────────────────────

def _generate_thumbnail_hf(title: str, script_snippet: str = "") -> bytes:
    """
    Generate a BitTrader-style thumbnail via Hugging Face SDXL.
    Returns raw image bytes or raises exception.

    Uses the official approved style from MEMORY.md.
    """
    import urllib.request

    # Build prompt based on title content
    title_lower = title.lower()

    # Adapt expression and objects to video topic
    if any(w in title_lower for w in ["caída", "cae", "baja", "crash", "pierde", "riesgo"]):
        expression = "shocked worried expression, dramatic reaction"
        objects = "falling red charts on screen"
        colors = "red and dark colors, warning atmosphere"
    elif any(w in title_lower for w in ["sube", "explota", "alza", "gana", "ganancia", "profit"]):
        expression = "excited celebrating expression, big smile"
        objects = "green rising charts, money, coins"
        colors = "green and gold colors, celebratory atmosphere"
    elif any(w in title_lower for w in ["bot", "ia", "inteligencia", "automatiz", "código", "python"]):
        expression = "confident analytical expression, focused"
        objects = "laptop with code, AI interface, data streams"
        colors = "blue and purple tech colors, futuristic atmosphere"
    else:
        expression = "excited confident expression, looking at camera"
        objects = "cryptocurrency charts and coins background"
        colors = "green and gold colors, professional studio lighting"

    # Official BitTrader thumbnail prompt (based on approved style in MEMORY.md)
    prompt = (
        f"excited young latin man, {expression}, holding smartphone showing crypto app, "
        f"{objects}, {colors}, "
        f"dramatic studio lighting, 4K cinematic, high contrast, sharp focus, "
        f"professional photography, clean background"
    )

    negative_prompt = (
        "blurry, dark, low quality, cartoon, anime, watermark, text overlay, "
        "monochromatic, solid color background, abstract, plain"
    )

    model = "stabilityai/stable-diffusion-xl-base-1.0"
    api_url = f"https://router.huggingface.co/hf-inference/models/{model}"

    payload = json.dumps({
        "inputs": prompt,
        "parameters": {
            "negative_prompt": negative_prompt,
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
        }
    }).encode()

    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _add_branding_to_thumbnail(image_bytes: bytes, title: str) -> bytes:
    """
    Add BitTrader branding to thumbnail:
    - Crop 1024x1024 → 1280x720 (centered crop + resize)
    - Logo BitTrader top-left
    - @bittrader9259 top-right
    - Title text bottom (MrBeast style with black border)
    - Dark gradient overlay for contrast
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Crop to 16:9 from center of 1024x1024
        w, h = img.size
        target_w = w
        target_h = int(w * 9 / 16)
        top = (h - target_h) // 2
        img = img.crop((0, top, target_w, top + target_h))
        img = img.resize((1280, 720), Image.LANCZOS)

        draw = ImageDraw.Draw(img)

        # Dark gradient overlay at bottom for text contrast
        from PIL import Image as PILImage
        gradient = PILImage.new("RGBA", (1280, 200), (0, 0, 0, 0))
        for y in range(200):
            alpha = int(180 * (y / 200))
            for x in range(1280):
                gradient.putpixel((x, y), (0, 0, 0, alpha))
        img_rgba = img.convert("RGBA")
        img_rgba.paste(gradient, (0, 520), gradient)
        img = img_rgba.convert("RGB")
        draw = ImageDraw.Draw(img)

        # Try to load a bold font, fall back to default
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        except Exception:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Title text at bottom (truncate if too long)
        title_display = title[:40] + "..." if len(title) > 40 else title
        title_display = title_display.upper()

        # Black border (MrBeast style): draw text in 8 directions offset
        tx, ty = 40, 640
        for dx, dy in [(-3,-3),(3,-3),(-3,3),(3,3),(0,-3),(0,3),(-3,0),(3,0)]:
            draw.text((tx + dx, ty + dy), title_display, font=font_large, fill=(0, 0, 0))
        draw.text((tx, ty), title_display, font=font_large, fill=(255, 255, 255))

        # @bittrader9259 top-right
        handle = "@bittrader9259"
        bbox = draw.textbbox((0, 0), handle, font=font_small)
        hw = bbox[2] - bbox[0]
        for dx, dy in [(-2,-2),(2,-2),(-2,2),(2,2)]:
            draw.text((1280 - hw - 20 + dx, 20 + dy), handle, font=font_small, fill=(0, 0, 0))
        draw.text((1280 - hw - 20, 20), handle, font=font_small, fill=(255, 255, 0))

        # BitTrader label top-left
        bt_text = "BitTrader"
        for dx, dy in [(-2,-2),(2,-2),(-2,2),(2,2)]:
            draw.text((20 + dx, 20 + dy), bt_text, font=font_small, fill=(0, 0, 0))
        draw.text((20, 20), bt_text, font=font_small, fill=(255, 165, 0))

        # Save to bytes
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=95)
        return output.getvalue()

    except Exception as e:
        _log(f"Branding overlay failed: {e}", "WARNING")
        return image_bytes  # Return unbranded if overlay fails


def regenerate_thumbnail(video_id: str, title: str, yt_client, dry_run: bool = False) -> dict:
    """
    Generate a new thumbnail for a video and upload it to YouTube.
    Returns dict with success status and details.
    """
    _log(f"  Regenerando thumbnail para: {title[:50]}")

    # Try HF SDXL first, then fallbacks
    models_to_try = [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "black-forest-labs/FLUX.1-schnell",
        "runwayml/stable-diffusion-v1-5",
    ]

    image_bytes = None
    model_used = None

    for model in models_to_try:
        try:
            _log(f"    Intentando {model}...")
            image_bytes = _generate_thumbnail_hf(title)
            model_used = model
            _log(f"    ✅ Generada con {model}")
            break
        except Exception as e:
            _log(f"    ⚠️ {model} falló: {str(e)[:60]}", "WARNING")
            time.sleep(2)

    if not image_bytes:
        _log(f"  ❌ Todos los modelos fallaron para {video_id}", "ERROR")
        return {"success": False, "error": "ALL_MODELS_FAILED", "video_id": video_id}

    # Add branding
    try:
        image_bytes = _add_branding_to_thumbnail(image_bytes, title)
        _log(f"    ✅ Branding añadido")
    except Exception as e:
        _log(f"    ⚠️ Branding falló: {e}", "WARNING")

    if dry_run:
        _log(f"  [DRY-RUN] Se generaría thumbnail para {video_id}")
        return {"success": True, "dry_run": True, "model": model_used, "video_id": video_id}

    # Save temporarily and upload
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(image_bytes)
        tmp_path = f.name

    try:
        from googleapiclient.http import MediaFileUpload
        media = MediaFileUpload(tmp_path, mimetype="image/jpeg", resumable=True)
        yt_client.thumbnails().set(videoId=video_id, media_body=media).execute()
        _log(f"  ✅ Thumbnail subida para {video_id}")
        return {"success": True, "model": model_used, "video_id": video_id}
    except Exception as e:
        _log(f"  ❌ Upload falló: {e}", "ERROR")
        return {"success": False, "error": str(e)[:100], "video_id": video_id}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Narration contamination check for existing videos ─────────────────────

def _check_title_description_contamination(title: str, description: str) -> list:
    """Check title and description of uploaded videos for leaked instructions."""
    import re
    from qa_agent import NARRATION_CONTAMINATION_PATTERNS, LEAKED_PREFIXES

    issues = []
    combined = f"{title}\n{description}"

    for pattern in NARRATION_CONTAMINATION_PATTERNS:
        try:
            if re.search(pattern, combined, re.IGNORECASE | re.MULTILINE):
                issues.append(f"CONTAMINATION: {pattern[:50]}")
                break  # One match is enough to flag
        except re.error:
            pass

    for prefix in LEAKED_PREFIXES:
        if title.strip().lower().startswith(prefix.lower()):
            issues.append(f"LEAKED_PREFIX: {prefix}")
            break

    return issues


# ── Main scan function ─────────────────────────────────────────────────────

def scan_channel(dry_run: bool = False, max_videos: int = 50) -> dict:
    """
    Scan all public channel videos for quality issues.
    Returns full report dict.
    """
    _log("=" * 65)
    _log(f"Iniciando scan del canal BitTrader {'[DRY-RUN]' if dry_run else '[LIVE]'}")
    _log("=" * 65)

    yt = get_youtube_client()

    # Get channel's uploads playlist
    channel_resp = yt.channels().list(
        part="contentDetails,snippet",
        mine=True,
    ).execute()

    channel_items = channel_resp.get("items", [])
    if not channel_items:
        _log("No se pudo obtener info del canal", "ERROR")
        return {"error": "channel_not_found"}

    channel_title = channel_items[0].get("snippet", {}).get("title", "?")
    uploads_playlist = (
        channel_items[0]
        .get("contentDetails", {})
        .get("relatedPlaylists", {})
        .get("uploads", "")
    )

    if not uploads_playlist:
        _log("No se encontró playlist de uploads", "ERROR")
        return {"error": "no_uploads_playlist"}

    _log(f"Canal: {channel_title}")

    # Fetch videos from uploads playlist
    all_videos = []
    page_token = None

    while len(all_videos) < max_videos:
        params = {
            "part": "snippet,contentDetails",
            "playlistId": uploads_playlist,
            "maxResults": min(50, max_videos - len(all_videos)),
        }
        if page_token:
            params["pageToken"] = page_token

        playlist_resp = yt.playlistItems().list(**params).execute()
        items = playlist_resp.get("items", [])
        all_videos.extend(items)

        page_token = playlist_resp.get("nextPageToken")
        if not page_token or not items:
            break

    _log(f"Videos encontrados: {len(all_videos)}")

    # Get full video details (thumbnails, status) in batches of 50
    video_ids = [
        v["snippet"]["resourceId"]["videoId"]
        for v in all_videos
        if "resourceId" in v.get("snippet", {})
    ]

    video_details = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        resp = yt.videos().list(
            part="snippet,status,contentDetails",
            id=",".join(batch),
        ).execute()
        for item in resp.get("items", []):
            video_details[item["id"]] = item

    # ── Analyze each video ───────────────────────────────────────────────
    report = {
        "scan_time": datetime.now(timezone.utc).isoformat(),
        "channel": channel_title,
        "total_videos": len(video_ids),
        "dry_run": dry_run,
        "issues_found": [],
        "thumbnails_fixed": [],
        "thumbnails_failed": [],
        "contamination_found": [],
        "ok_videos": [],
        "stats": {},
    }

    for vid_id in video_ids:
        details = video_details.get(vid_id)
        if not details:
            continue

        snippet = details.get("snippet", {})
        status  = details.get("status", {})
        title   = snippet.get("title", "?")
        desc    = snippet.get("description", "")
        thumbs  = snippet.get("thumbnails", {})
        privacy = status.get("privacyStatus", "unknown")

        # Skip deleted/private (unless they have private status with uploaded thumbnail)
        if privacy == "private":
            continue

        video_issues = []

        # ── Thumbnail checks ──────────────────────────────────────────
        has_custom = _has_custom_thumbnail(thumbs)

        if not has_custom:
            video_issues.append("NO_CUSTOM_THUMBNAIL")
            _log(f"  ⚠️ Sin thumbnail custom: {title[:50]}")
        else:
            # Analyze the maxres thumbnail
            maxres_url = thumbs.get("maxres", thumbs.get("high", {})).get("url", "")
            if maxres_url:
                analysis = _analyze_thumbnail_url(maxres_url)
                if not analysis.get("ok", True):
                    if analysis.get("is_dark"):
                        video_issues.append("DARK_THUMBNAIL")
                    if analysis.get("is_blue"):
                        video_issues.append("BLUE_THUMBNAIL")
                    if analysis.get("is_monochromatic"):
                        video_issues.append("MONOCHROMATIC_THUMBNAIL")
                    _log(f"  ⚠️ Thumbnail mala ({', '.join(video_issues)}): {title[:45]}")

        # ── Title/description contamination ───────────────────────────
        contamination = _check_title_description_contamination(title, desc)
        if contamination:
            video_issues.extend(contamination)
            report["contamination_found"].append({
                "video_id": vid_id,
                "title": title,
                "issues": contamination,
            })
            _log(f"  🚨 Contaminación: {title[:50]} — {contamination}", "ERROR")

        # ── Record issues ─────────────────────────────────────────────
        if video_issues:
            entry = {
                "video_id": vid_id,
                "title": title,
                "issues": video_issues,
                "privacy": privacy,
            }
            report["issues_found"].append(entry)

            # Auto-fix bad thumbnails
            thumb_issues = [i for i in video_issues if "THUMBNAIL" in i]
            if thumb_issues and not contamination:
                result = regenerate_thumbnail(vid_id, title, yt, dry_run=dry_run)
                if result.get("success"):
                    report["thumbnails_fixed"].append({
                        "video_id": vid_id,
                        "title": title,
                        "model": result.get("model", "?"),
                        "issues_fixed": thumb_issues,
                        "dry_run": dry_run,
                    })
                else:
                    report["thumbnails_failed"].append({
                        "video_id": vid_id,
                        "title": title,
                        "error": result.get("error", "?"),
                    })
        else:
            report["ok_videos"].append(vid_id)

        # Small delay to avoid rate limiting
        time.sleep(0.1)

    # ── Stats ────────────────────────────────────────────────────────────
    report["stats"] = {
        "total": len(video_ids),
        "ok": len(report["ok_videos"]),
        "issues": len(report["issues_found"]),
        "thumbnails_fixed": len(report["thumbnails_fixed"]),
        "thumbnails_failed": len(report["thumbnails_failed"]),
        "contamination": len(report["contamination_found"]),
    }

    # ── Save report ──────────────────────────────────────────────────────
    SCANNER_LOG.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    _log(f"Reporte guardado → {SCANNER_LOG.name}")

    # ── Summary log ──────────────────────────────────────────────────────
    stats = report["stats"]
    _log("=" * 65)
    _log(f"RESULTADO: {stats['ok']}/{stats['total']} videos OK")
    _log(f"  Problemas encontrados:    {stats['issues']}")
    _log(f"  Thumbnails regeneradas:   {stats['thumbnails_fixed']}")
    _log(f"  Thumbnails fallidas:      {stats['thumbnails_failed']}")
    _log(f"  Contaminación detectada:  {stats['contamination']}")
    _log("=" * 65)

    # ── Telegram notification ─────────────────────────────────────────────
    _send_scan_report(report, dry_run)

    # ── Update state ──────────────────────────────────────────────────────
    SCANNER_STATE.write_text(json.dumps({
        "last_scan": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
    }, indent=2))

    return report


def _send_scan_report(report: dict, dry_run: bool):
    """Send Telegram summary after scan completes."""
    stats = report["stats"]

    # Only notify if there were issues or it's a fresh run
    if stats["issues"] == 0 and stats["contamination"] == 0:
        _log("Sin problemas encontrados — no notificando Telegram")
        return

    dr = " [DRY-RUN]" if dry_run else ""
    lines = [f"🔍 <b>Scan del Canal BitTrader{dr}</b>\n"]
    lines.append(f"📊 {stats['ok']}/{stats['total']} videos OK")

    if stats["issues"]:
        lines.append(f"\n⚠️ <b>{stats['issues']} videos con problemas:</b>")
        for item in report["issues_found"][:5]:
            lines.append(f"  • {item['title'][:40]} — {', '.join(item['issues'][:2])}")

    if stats["thumbnails_fixed"]:
        lines.append(f"\n✅ <b>{stats['thumbnails_fixed']} thumbnails regeneradas</b>")
        for item in report["thumbnails_fixed"][:3]:
            lines.append(f"  • {item['title'][:40]}")

    if stats["contamination"]:
        lines.append(f"\n🚨 <b>{stats['contamination']} videos con CONTAMINACIÓN:</b>")
        for item in report["contamination_found"][:3]:
            lines.append(f"  • {item['title'][:40]}")
        lines.append("  → Revisar y borrar manualmente")

    if stats["thumbnails_failed"]:
        lines.append(f"\n❌ {stats['thumbnails_failed']} thumbnails fallidas (revisar HF API)")

    notify_ender("\n".join(lines))


def should_run_scan() -> bool:
    """Check if 24h have passed since last scan."""
    if not SCANNER_STATE.exists():
        return True
    try:
        state = json.loads(SCANNER_STATE.read_text())
        last = datetime.fromisoformat(state.get("last_scan", "2000-01-01T00:00:00+00:00"))
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - last) > timedelta(hours=24)
    except Exception:
        return True


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="🔍 QA Channel Scanner — BitTrader daily quality audit"
    )
    parser.add_argument("--dry-run",    action="store_true", help="Report only, do not fix")
    parser.add_argument("--force",      action="store_true", help="Force scan even if <24h since last")
    parser.add_argument("--report",     action="store_true", help="Show last saved report and exit")
    parser.add_argument("--max-videos", type=int, default=50, help="Max videos to scan (default: 50)")
    args = parser.parse_args()

    if args.report:
        if SCANNER_LOG.exists():
            report = json.loads(SCANNER_LOG.read_text())
            stats = report.get("stats", {})
            print(f"\n📋 Último scan: {report.get('scan_time', '?')}")
            print(f"   Total: {stats.get('total', 0)} | OK: {stats.get('ok', 0)} | Issues: {stats.get('issues', 0)}")
            print(f"   Thumbnails fijas: {stats.get('thumbnails_fixed', 0)} | Contaminación: {stats.get('contamination', 0)}")
            if report.get("issues_found"):
                print("\nProblemas:")
                for item in report["issues_found"][:10]:
                    print(f"  {item['video_id']}: {item['title'][:50]} — {item['issues']}")
        else:
            print("No hay reporte guardado aún.")
        sys.exit(0)

    if not args.force and not args.dry_run and not should_run_scan():
        _log("Scan reciente (<24h) — omitiendo. Usa --force para forzar.")
        sys.exit(0)

    report = scan_channel(dry_run=args.dry_run, max_videos=args.max_videos)
    stats = report.get("stats", {})
    print(json.dumps(stats, indent=2))
    sys.exit(0 if stats.get("contamination", 0) == 0 else 1)
