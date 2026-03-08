#!/usr/bin/env python3
"""
📅 BitTrader Schedule Optimizer — Horarios Óptimos de Publicación
Analiza datos de YouTube Analytics para determinar las mejores horas
para publicar videos según la audiencia del canal.

Datos que analiza:
1. Hora de publicación vs views (correlación)
2. Día de la semana vs rendimiento
3. Retención por tipo de contenido (shorts vs longs)
4. Horarios de mayor actividad de la audiencia
5. País/región de la audiencia → zona horaria dominante

Ejecutar: python3 agents/schedule_optimizer.py
"""
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import Counter, defaultdict

# ── Paths ──────────────────────────────────────────────────────────
WORKSPACE = Path("/home/enderj/.openclaw/workspace")
DATA_DIR = WORKSPACE / "bittrader/agents/data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(WORKSPACE / "youtube_env/lib/python3.13/site-packages"))

YT_CREDS = WORKSPACE / "memory/youtube_credentials.json"
SCHEDULE_FILE = DATA_DIR / "optimal_schedule.json"

# Days in Spanish
DAYS_ES = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves",
           4: "Viernes", 5: "Sábado", 6: "Domingo"}


def get_youtube_client():
    """Get authenticated YouTube client."""
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


def get_youtube_analytics_client():
    """Get authenticated YouTube Analytics client."""
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

    try:
        return build("youtubeAnalytics", "v2", credentials=creds)
    except Exception:
        return None


def parse_iso_duration(d: str) -> int:
    """ISO 8601 duration to seconds."""
    import re
    m = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', d or "")
    if m:
        return int(m.group(1) or 0)*3600 + int(m.group(2) or 0)*60 + int(m.group(3) or 0)
    return 0


# ════════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ════════════════════════════════════════════════════════════════════════

def collect_video_data(yt, max_videos: int = 200) -> list:
    """Collect all video data with publish times and metrics."""
    print("  📥 Recopilando datos de videos...")

    ch_resp = yt.channels().list(mine=True, part="contentDetails").execute()
    playlist_id = ch_resp["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    all_videos = []
    next_page = None
    pages = 0

    while pages < (max_videos // 50 + 1):
        pl = yt.playlistItems().list(
            playlistId=playlist_id, part="snippet,contentDetails",
            maxResults=50, pageToken=next_page
        ).execute()

        video_ids = [item["contentDetails"]["videoId"] for item in pl["items"]]
        if not video_ids:
            break

        details = yt.videos().list(
            id=",".join(video_ids),
            part="statistics,contentDetails,snippet"
        ).execute()

        for item in details["items"]:
            v_stats = item["statistics"]
            v_snippet = item["snippet"]
            c_details = item["contentDetails"]
            duration = parse_iso_duration(c_details["duration"])

            published = datetime.fromisoformat(
                v_snippet["publishedAt"].replace("Z", "+00:00"))

            vid = {
                "id": item["id"],
                "title": v_snippet["title"],
                "published_at": v_snippet["publishedAt"],
                "published_hour": published.hour,
                "published_weekday": published.weekday(),
                "published_day_name": DAYS_ES.get(published.weekday(), "?"),
                "duration_seconds": duration,
                "is_short": duration <= 60,
                "views": int(v_stats.get("viewCount", 0)),
                "likes": int(v_stats.get("likeCount", 0)),
                "comments": int(v_stats.get("commentCount", 0)),
                "days_since_publish": (datetime.now(timezone.utc) - published).days,
            }

            # Calculate views per day (normalized metric)
            days = max(1, vid["days_since_publish"])
            vid["views_per_day"] = round(vid["views"] / days, 2)

            all_videos.append(vid)

        next_page = pl.get("nextPageToken")
        pages += 1
        if not next_page:
            break

    print(f"  📊 {len(all_videos)} videos recopilados")
    return all_videos


def collect_analytics_data(analytics) -> dict:
    """Collect audience activity data from YouTube Analytics API."""
    if not analytics:
        return {}

    print("  📈 Recopilando datos de Analytics...")

    try:
        # Get viewer demographics and activity
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

        # Country data
        country_data = analytics.reports().query(
            ids="channel==MINE",
            startDate=start_date,
            endDate=end_date,
            metrics="views,estimatedMinutesWatched",
            dimensions="country",
            sort="-views",
            maxResults=10
        ).execute()

        countries = []
        for row in country_data.get("rows", []):
            countries.append({
                "country": row[0],
                "views": row[1],
                "watch_minutes": row[2],
            })

        # Day of week data
        day_data = analytics.reports().query(
            ids="channel==MINE",
            startDate=start_date,
            endDate=end_date,
            metrics="views,estimatedMinutesWatched,averageViewDuration",
            dimensions="day",
            sort="day"
        ).execute()

        daily = []
        for row in day_data.get("rows", []):
            date = datetime.strptime(row[0], "%Y-%m-%d")
            daily.append({
                "date": row[0],
                "weekday": date.weekday(),
                "day_name": DAYS_ES.get(date.weekday(), "?"),
                "views": row[1],
                "watch_minutes": row[2],
                "avg_view_duration": row[3],
            })

        return {
            "countries": countries,
            "daily": daily,
        }

    except Exception as e:
        print(f"  ⚠️ Analytics error: {e}")
        return {"error": str(e)}


# ════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ════════════════════════════════════════════════════════════════════════

def analyze_best_hours(videos: list) -> dict:
    """Analyze which publication hours get the most views."""
    print("  🕐 Analizando mejores horas de publicación...")

    hour_stats = defaultdict(lambda: {"count": 0, "total_views": 0, "total_vpd": 0})

    for v in videos:
        h = v["published_hour"]
        hour_stats[h]["count"] += 1
        hour_stats[h]["total_views"] += v["views"]
        hour_stats[h]["total_vpd"] += v["views_per_day"]

    # Calculate averages
    hour_analysis = []
    for hour in range(24):
        s = hour_stats[hour]
        if s["count"] > 0:
            hour_analysis.append({
                "hour": hour,
                "hour_label": f"{hour:02d}:00",
                "videos_published": s["count"],
                "avg_views": round(s["total_views"] / s["count"]),
                "avg_views_per_day": round(s["total_vpd"] / s["count"], 2),
            })

    # Sort by avg_views_per_day (normalized for time since publish)
    hour_analysis.sort(key=lambda x: x["avg_views_per_day"], reverse=True)

    return {
        "all_hours": hour_analysis,
        "best_hours": hour_analysis[:5],
        "worst_hours": hour_analysis[-3:] if len(hour_analysis) > 3 else [],
    }


def analyze_best_days(videos: list) -> dict:
    """Analyze which days of the week perform best."""
    print("  📅 Analizando mejores días de la semana...")

    day_stats = defaultdict(lambda: {"count": 0, "total_views": 0, "total_vpd": 0})

    for v in videos:
        d = v["published_weekday"]
        day_stats[d]["count"] += 1
        day_stats[d]["total_views"] += v["views"]
        day_stats[d]["total_vpd"] += v["views_per_day"]

    day_analysis = []
    for day in range(7):
        s = day_stats[day]
        if s["count"] > 0:
            day_analysis.append({
                "day": day,
                "day_name": DAYS_ES[day],
                "videos_published": s["count"],
                "avg_views": round(s["total_views"] / s["count"]),
                "avg_views_per_day": round(s["total_vpd"] / s["count"], 2),
            })

    day_analysis.sort(key=lambda x: x["avg_views_per_day"], reverse=True)

    return {
        "all_days": day_analysis,
        "best_days": day_analysis[:3],
    }


def analyze_shorts_vs_longs(videos: list) -> dict:
    """Compare shorts vs long video performance."""
    print("  📊 Comparando Shorts vs Longs...")

    shorts = [v for v in videos if v["is_short"]]
    longs = [v for v in videos if not v["is_short"]]

    def stats(vids):
        if not vids:
            return {"count": 0, "avg_views": 0, "avg_vpd": 0}
        return {
            "count": len(vids),
            "avg_views": round(sum(v["views"] for v in vids) / len(vids)),
            "avg_views_per_day": round(sum(v["views_per_day"] for v in vids) / len(vids), 2),
            "avg_likes": round(sum(v["likes"] for v in vids) / len(vids)),
            "avg_comments": round(sum(v["comments"] for v in vids) / len(vids), 1),
        }

    return {
        "shorts": stats(shorts),
        "longs": stats(longs),
        "recommendation": "shorts" if (shorts and longs and
            stats(shorts)["avg_views_per_day"] > stats(longs)["avg_views_per_day"])
            else "longs" if longs else "shorts",
    }


def analyze_audience_timezone(analytics_data: dict) -> dict:
    """Determine primary audience timezone from country data."""
    countries = analytics_data.get("countries", [])
    if not countries:
        return {"primary_tz": "America/Denver", "note": "Sin datos — usando zona del creador"}

    # Country → approximate UTC offset mapping
    tz_map = {
        "US": -6, "MX": -6, "CO": -5, "AR": -3, "ES": 1, "VE": -4,
        "CL": -3, "PE": -5, "EC": -5, "BR": -3, "DO": -4, "GT": -6,
        "PA": -5, "CR": -6, "UY": -3, "PY": -4, "BO": -4, "HN": -6,
        "SV": -6, "NI": -6, "CU": -5, "PR": -4,
    }

    # Weight by views
    total_views = sum(c["views"] for c in countries)
    weighted_offset = 0
    for c in countries:
        offset = tz_map.get(c["country"], -5)
        weight = c["views"] / max(1, total_views)
        weighted_offset += offset * weight

    return {
        "top_countries": countries[:5],
        "weighted_utc_offset": round(weighted_offset, 1),
        "primary_tz": "America/Denver" if -7 <= weighted_offset <= -5 else
                      "America/New_York" if -5 < weighted_offset <= -4 else
                      "America/Argentina/Buenos_Aires" if -4 < weighted_offset <= -2 else
                      "Europe/Madrid" if weighted_offset > 0 else "America/Denver",
    }


def analyze_daily_patterns(analytics_data: dict) -> dict:
    """Analyze daily view patterns from Analytics data."""
    daily = analytics_data.get("daily", [])
    if not daily:
        return {}

    # Group by day of week
    day_views = defaultdict(list)
    for d in daily:
        day_views[d["weekday"]].append(d["views"])

    day_avg = {}
    for day, views_list in day_views.items():
        day_avg[DAYS_ES[day]] = round(sum(views_list) / len(views_list))

    return {
        "avg_views_by_day": day_avg,
        "best_viewing_day": max(day_avg, key=day_avg.get) if day_avg else "?",
    }


# ════════════════════════════════════════════════════════════════════════
# SCHEDULE RECOMMENDATION
# ════════════════════════════════════════════════════════════════════════

def generate_schedule(hour_analysis: dict, day_analysis: dict,
                      content_analysis: dict, audience_tz: dict) -> dict:
    """Generate optimal publishing schedule."""
    print("  📅 Generando horario óptimo...")

    best_hours = hour_analysis.get("best_hours", [])
    best_days = day_analysis.get("best_days", [])

    # Build recommendation
    schedule = {
        "timezone": audience_tz.get("primary_tz", "America/Denver"),
        "slots": [],
    }

    # Create publishing slots
    if best_days and best_hours:
        for day in best_days[:3]:
            for hour in best_hours[:2]:
                schedule["slots"].append({
                    "day": day["day_name"],
                    "day_number": day["day"],
                    "hour": hour["hour"],
                    "hour_label": hour["hour_label"],
                    "expected_vpd": round((day["avg_views_per_day"] + hour["avg_views_per_day"]) / 2, 2),
                })

    # Sort by expected performance
    schedule["slots"].sort(key=lambda x: x["expected_vpd"], reverse=True)

    # Content mix recommendation
    content_rec = content_analysis.get("recommendation", "shorts")
    schedule["content_mix"] = {
        "recommendation": content_rec,
        "suggested_ratio": "3 shorts : 1 long" if content_rec == "shorts" else "2 longs : 2 shorts",
        "weekly_target": 4,
    }

    return schedule


# ════════════════════════════════════════════════════════════════════════
# REPORT
# ════════════════════════════════════════════════════════════════════════

def build_report(videos: list, hour_analysis: dict, day_analysis: dict,
                 content_analysis: dict, audience_tz: dict, schedule: dict) -> str:
    """Build human-readable report."""
    lines = []
    lines.append("📅 SCHEDULE OPTIMIZER — BitTrader")
    lines.append("=" * 50)
    lines.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')} MST")
    lines.append(f"Videos analizados: {len(videos)}")

    # Best hours
    lines.append("\n🕐 MEJORES HORAS DE PUBLICACIÓN (UTC)")
    for h in hour_analysis.get("best_hours", [])[:5]:
        bar = "█" * min(20, int(h["avg_views_per_day"]))
        lines.append(f"  {h['hour_label']} | {h['avg_views']:>6,} avg views | {h['avg_views_per_day']:>6.1f} vpd | {bar}")

    # Best days
    lines.append("\n📅 MEJORES DÍAS")
    for d in day_analysis.get("best_days", [])[:3]:
        lines.append(f"  {d['day_name']:>10} | {d['avg_views']:>6,} avg views | {d['avg_views_per_day']:>6.1f} vpd | {d['videos_published']} videos")

    # Shorts vs Longs
    lines.append("\n📊 SHORTS vs LONGS")
    s = content_analysis.get("shorts", {})
    l = content_analysis.get("longs", {})
    lines.append(f"  Shorts: {s.get('count',0)} videos | {s.get('avg_views',0):,} avg views | {s.get('avg_views_per_day',0):.1f} vpd")
    lines.append(f"  Longs:  {l.get('count',0)} videos | {l.get('avg_views',0):,} avg views | {l.get('avg_views_per_day',0):.1f} vpd")
    lines.append(f"  👉 Recomendación: Priorizar {content_analysis.get('recommendation','?')}")

    # Audience
    tz_data = audience_tz
    if tz_data.get("top_countries"):
        lines.append("\n🌍 AUDIENCIA POR PAÍS")
        for c in tz_data["top_countries"][:5]:
            lines.append(f"  {c['country']}: {c['views']:,} views | {c['watch_minutes']:,.0f} min")
        lines.append(f"  Zona horaria dominante: {tz_data.get('primary_tz','?')}")

    # Schedule
    lines.append("\n📅 HORARIO ÓPTIMO RECOMENDADO")
    lines.append(f"  Zona horaria: {schedule.get('timezone','?')}")
    for slot in schedule.get("slots", [])[:6]:
        lines.append(f"  {slot['day']:>10} {slot['hour_label']} → {slot['expected_vpd']:.1f} vpd esperado")

    mix = schedule.get("content_mix", {})
    lines.append(f"\n  📝 Mix semanal: {mix.get('suggested_ratio','?')}")
    lines.append(f"  🎯 Videos por semana: {mix.get('weekly_target',4)}")

    lines.append("\n" + "=" * 50)
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def run_schedule_optimizer() -> dict:
    """Run full analysis and generate optimal schedule."""
    print("\n📅 BitTrader Schedule Optimizer v1.0")

    yt = get_youtube_client()
    analytics = get_youtube_analytics_client()

    # Collect data
    videos = collect_video_data(yt, max_videos=200)
    analytics_data = collect_analytics_data(analytics)

    # Analyze
    hour_analysis = analyze_best_hours(videos)
    day_analysis = analyze_best_days(videos)
    content_analysis = analyze_shorts_vs_longs(videos)
    audience_tz = analyze_audience_timezone(analytics_data)
    daily_patterns = analyze_daily_patterns(analytics_data)

    # Generate schedule
    schedule = generate_schedule(hour_analysis, day_analysis, content_analysis, audience_tz)

    # Build report
    report = build_report(videos, hour_analysis, day_analysis,
                         content_analysis, audience_tz, schedule)
    print(report)

    # Save results
    result = {
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "videos_analyzed": len(videos),
        "best_hours": hour_analysis,
        "best_days": day_analysis,
        "content_analysis": content_analysis,
        "audience_timezone": audience_tz,
        "daily_patterns": daily_patterns,
        "schedule": schedule,
    }

    SCHEDULE_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n  💾 Guardado: {SCHEDULE_FILE}")

    return result


if __name__ == "__main__":
    result = run_schedule_optimizer()
