#!/usr/bin/env python3
"""
📊 YouTube Stats Collector — Reporte Automático de Estadísticas
Ejecuta manual o via cron cada X horas
"""
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import Counter

# ── Paths ──────────────────────────────────────────────────────────
WORKSPACE = Path("/home/enderj/.openclaw/workspace")
DATA_DIR = WORKSPACE / "bittrader/agents/data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# YouTube env
sys.path.insert(0, str(WORKSPACE / "youtube_env/lib/python3.13/site-packages"))

# ── Config ─────────────────────────────────────────────────────────
YT_CREDS = WORKSPACE / "memory/youtube_credentials.json"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"


def get_youtube_client():
    """Obtener cliente de YouTube OAuth2"""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    d = json.loads(YT_CREDS.read_text())
    creds = Credentials(
        token=d["access_token"],
        refresh_token=d["refresh_token"],
        token_uri=d["token_uri"],
        client_id=d["client_id"],
        client_secret=d["client_secret"],
        scopes=d.get("scope", "").split()
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        d["access_token"] = creds.token
        YT_CREDS.write_text(json.dumps(d, indent=2))
    return build("youtube", "v3", credentials=creds)


def parse_iso_duration(d: str) -> int:
    """Convertir ISO 8601 duration a segundos"""
    import re
    m = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', d or "")
    if m:
        return int(m.group(1) or 0)*3600 + int(m.group(2) or 0)*60 + int(m.group(3) or 0)
    return 0


def collect_youtube_stats() -> dict:
    """Recolectar estadísticas completas del canal"""
    print("  📊 Recolectando estadísticas de YouTube...")

    try:
        yt = get_youtube_client()

        # Channel info
        ch_resp = yt.channels().list(mine=True, part="statistics,contentDetails,id").execute()
        ch = ch_resp["items"][0]
        stats = ch["statistics"]

        channel_stats = {
            "subscribers": int(stats.get("subscriberCount", 0)),
            "total_views": int(stats.get("viewCount", 0)),
            "total_videos": int(stats.get("videoCount", 0)),
            "total_comments": int(stats.get("commentCount", 0)),
        }

        # Get upload playlist
        playlist_id = ch["contentDetails"]["relatedPlaylists"]["uploads"]

        # Fetch recent videos (últimos 50)
        all_videos = []
        next_page = None
        page_count = 0

        while True and page_count < 2:  # Solo las primeras 2 páginas (~100 videos)
            pl = yt.playlistItems().list(
                playlistId=playlist_id, part="snippet,contentDetails",
                maxResults=50, pageToken=next_page
            ).execute()

            video_ids = [item["contentDetails"]["videoId"] for item in pl["items"]]

            if not video_ids:
                break

            # Batch fetch details
            details = yt.videos().list(
                id=",".join(video_ids),
                part="statistics,contentDetails,snippet"
            ).execute()

            for item in details["items"]:
                v_stats = item["statistics"]
                v_snippet = item["snippet"]
                c_details = item["contentDetails"]

                vid = {
                    "id": item["id"],
                    "title": v_snippet["title"],
                    "published_at": v_snippet["publishedAt"],
                    "duration_seconds": parse_iso_duration(c_details["duration"]),
                    "views": int(v_stats.get("viewCount", 0)),
                    "likes": int(v_stats.get("likeCount", 0)),
                    "comments": int(v_stats.get("commentCount", 0)),
                }
                all_videos.append(vid)

            next_page = pl.get("nextPageToken")
            page_count += 1

        # Analizar videos recientes
        recent_videos = sorted(all_videos[:30], key=lambda v: v["published_at"], reverse=True)

        # Cálculos
        if recent_videos:
            avg_views = sum(v["views"] for v in recent_videos) / len(recent_videos)
            top_video = max(recent_videos, key=lambda v: v["views"])
            bottom_video = min(recent_videos, key=lambda v: v["views"])

            # Distribución por duración
            durations = [v["duration_seconds"] for v in recent_videos]
            avg_duration = sum(durations) / len(durations)

            # Views per day (últimos 7 días)
            now = datetime.now(timezone.utc)
            seven_days_ago = now - timedelta(days=7)
            recent_7d = [v for v in recent_videos if datetime.fromisoformat(v["published_at"].replace("Z", "+00:00")) >= seven_days_ago]
            views_7d = sum(v["views"] for v in recent_7d)
            views_per_day_7d = views_7d / 7 if recent_7d else 0
        else:
            avg_views = 0
            top_video = None
            bottom_video = None
            avg_duration = 0
            views_per_day_7d = 0

        return {
            "channel": channel_stats,
            "recent_videos": recent_videos,
            "summary": {
                "avg_views_last_30": round(avg_views, 0),
                "top_video": {
                    "title": top_video["title"],
                    "views": top_video["views"],
                    "url": f"https://youtube.com/watch?v={top_video['id']}"
                } if top_video else None,
                "bottom_video": {
                    "title": bottom_video["title"],
                    "views": bottom_video["views"],
                    "url": f"https://youtube.com/watch?v={bottom_video['id']}"
                } if bottom_video else None,
                "avg_duration_seconds": round(avg_duration, 0),
                "avg_duration_formatted": f"{int(avg_duration//60)}:{int(avg_duration%60):02d}",
                "views_per_day_7d": round(views_per_day_7d, 0),
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


def build_report(stats: dict) -> str:
    """Construir reporte formateado"""
    report = []
    report.append("📊 YOUTUBE STATS REPORT")
    report.append("=" * 50)
    report.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Mountain Time)")

    # Channel stats
    ch = stats.get("channel", {})
    if ch:
        report.append("\n📱 CANAL BITTRADER")
        report.append(f"  Suscriptores: {ch.get('subscribers', 0):,}")
        report.append(f"  Views totales: {ch.get('total_views', 0):,}")
        report.append(f"  Videos totales: {ch.get('total_videos', 0):,}")
        report.append(f"  Comentarios totales: {ch.get('total_comments', 0):,}")
        report.append(f"  Views por sub: {round(ch.get('total_views', 0) / max(1, ch.get('subscribers', 1)), 1)}")

    # Summary
    summ = stats.get("summary", {})
    if summ and not stats.get("error"):
        report.append("\n📈 ÚLTIMOS 30 VIDEOS")
        report.append(f"  Promedio views: {summ.get('avg_views_last_30', 0):,}")
        report.append(f"  Duración promedio: {summ.get('avg_duration_formatted', '0:00')}")

        top = summ.get("top_video")
        if top:
            report.append(f"\n  🥇 TOP VIDEO")
            report.append(f"    {top['title']}")
            report.append(f"    👁 {top['views']:,} views")
            report.append(f"    🔗 {top['url']}")

        bottom = summ.get("bottom_video")
        if bottom:
            report.append(f"\n  📉 LOWEST VIDEO")
            report.append(f"    {bottom['title']}")
            report.append(f"    👁 {bottom['views']:,} views")

        report.append(f"\n📊 Views/día (últimos 7 días): {summ.get('views_per_day_7d', 0):,}")

    # Recent videos
    recents = stats.get("recent_videos", [])[:10]
    if recents:
        report.append("\n🎬 VIDEOS RECIENTES (últimos 10)")
        for i, v in enumerate(recents, 1):
            published = datetime.fromisoformat(v["published_at"].replace("Z", "+00:00")).strftime("%m/%d %H:%M")
            mins = v["duration_seconds"] // 60
            secs = v["duration_seconds"] % 60
            duration = f"{mins}:{secs:02d}"
            report.append(f"  {i:2d}. {v['title'][:40]}")
            report.append(f"      📅 {published} | ⏱️ {duration} | 👁 {v['views']:,} | 👍 {v['likes']:,} | 💬 {v['comments']:,}")

    # Error
    if stats.get("error"):
        report.append(f"\n❌ ERROR: {stats['error']}")

    report.append("\n" + "=" * 50)
    return "\n".join(report)


def save_report(stats: dict):
    """Guardar reporte y enviarlo"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = build_report(stats)

    # Guardar en archivo
    report_file = DATA_DIR / f"youtube_stats_{timestamp}.txt"
    report_file.write_text(report)

    # Guardar latest
    latest_file = DATA_DIR / "youtube_stats_latest.json"
    latest_file.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print(f"\n✅ Reporte guardado: {report_file.name}")

    return report


def main():
    print("\n" + "=" * 50)
    print("📊 YouTube Stats Collector")
    print("=" * 50)

    stats = collect_youtube_stats()
    report = save_report(stats)

    # Enviar por Telegram
    try:
        import requests
        tg_config = WORKSPACE / "memory/telegram_config.json"
        if tg_config.exists():
            cfg = json.loads(tg_config.read_text())
            bot_tok = cfg.get("bot_token", "")
            chat_id = cfg.get("chat_id", "")
            if bot_tok and chat_id:
                url = f"https://api.telegram.org/bot{bot_tok}/sendMessage"
                requests.post(url, json={
                    "chat_id": chat_id,
                    "text": report,
                    "parse_mode": "HTML"
                }, timeout=10)
                print("  📱 Enviado por Telegram")
    except Exception as e:
        print(f"  ⚠️ Telegram error: {e}")

    print("\n✅ Completado")


if __name__ == "__main__":
    main()
