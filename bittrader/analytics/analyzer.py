#!/usr/bin/env python3
"""
BitTrader Pattern Analyzer
Analiza datos recolectados e identifica patrones ganadores
"""
import json, glob, re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import Counter

DATA_DIR = Path("/home/enderj/.openclaw/workspace/bittrader/analytics/data")
INSIGHTS_FILE = Path("/home/enderj/.openclaw/workspace/bittrader/analytics/insights_latest.json")

def load_latest_data():
    """Cargar el reporte más reciente"""
    files = sorted(DATA_DIR.glob("*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())

def load_previous_data(days_back=2):
    """Cargar reporte de hace N días para comparar"""
    files = sorted(DATA_DIR.glob("*.json"))
    if len(files) < 2:
        return None
    target = datetime.now(timezone.utc) - timedelta(days=days_back)
    for f in reversed(files[:-1]):
        ts = f.stem  # YYYY-MM-DD_HH
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d_%H").replace(tzinfo=timezone.utc)
            if dt <= target:
                return json.loads(f.read_text())
        except:
            continue
    return json.loads(files[0].read_text()) if files else None

def classify_videos(videos):
    """Clasificar videos en winners/average/losers"""
    if not videos:
        return {"winners": [], "average": [], "losers": []}
    
    # Sort by views
    sorted_vids = sorted(videos, key=lambda v: v["views"], reverse=True)
    n = len(sorted_vids)
    top20 = max(1, n // 5)
    bottom20 = max(1, n // 5)
    
    return {
        "winners": sorted_vids[:top20],
        "average": sorted_vids[top20:-bottom20] if n > 2 else [],
        "losers": sorted_vids[-bottom20:]
    }

def extract_title_patterns(videos):
    """Extraer patrones de títulos que funcionan"""
    patterns = {
        "pregunta": {"regex": r"^¿|^Qué|^Cómo|^Por qué|^Cuál", "views": [], "count": 0},
        "numero": {"regex": r"\d+", "views": [], "count": 0},
        "emoji_inicio": {"regex": r"^[^\w]", "views": [], "count": 0},
        "mayusculas": {"regex": r"[A-ZÁÉÍÓÚ]{3,}", "views": [], "count": 0},
        "negativo": {"regex": r"error|pierd|miedo|nunca|NO ", "views": [], "count": 0},
        "secreto": {"regex": r"secreto|nadie|oculto|verdad", "views": [], "count": 0},
        "lista": {"regex": r"^\d+ |Las \d+|Los \d+", "views": [], "count": 0},
    }
    
    for v in videos:
        for key, p in patterns.items():
            if re.search(p["regex"], v["title"], re.IGNORECASE):
                p["views"].append(v["views"])
                p["count"] += 1
    
    results = {}
    for key, p in patterns.items():
        if p["views"]:
            results[key] = {
                "avg_views": sum(p["views"]) / len(p["views"]),
                "count": p["count"],
                "total_views": sum(p["views"])
            }
    
    return dict(sorted(results.items(), key=lambda x: x[1]["avg_views"], reverse=True))

def analyze_timing(videos):
    """Analizar qué día/hora funciona mejor"""
    day_views = {}
    hour_views = {}
    
    for v in videos:
        try:
            dt = datetime.fromisoformat(v["published"].replace("Z", "+00:00"))
            day = dt.strftime("%A")
            hour = dt.hour
            
            if day not in day_views:
                day_views[day] = []
            day_views[day].append(v["views"])
            
            if hour not in hour_views:
                hour_views[hour] = []
            hour_views[hour].append(v["views"])
        except:
            continue
    
    best_day = max(day_views.items(), key=lambda x: sum(x[1])/len(x[1]), default=("Unknown", [0]))
    best_hour = max(hour_views.items(), key=lambda x: sum(x[1])/len(x[1]), default=(12, [0]))
    
    return {
        "best_day": best_day[0],
        "best_day_avg": sum(best_day[1]) / len(best_day[1]) if best_day[1] else 0,
        "best_hour_utc": best_hour[0],
        "best_hour_avg": sum(best_hour[1]) / len(best_hour[1]) if best_hour[1] else 0,
        "by_day": {k: {"avg": sum(v)/len(v), "count": len(v)} for k,v in day_views.items()},
        "by_hour": {str(k): {"avg": sum(v)/len(v), "count": len(v)} for k,v in sorted(hour_views.items())}
    }

def analyze_duration(videos):
    """Analizar duración óptima"""
    duration_buckets = {
        "15-25s": {"range": (15, 25), "views": []},
        "25-35s": {"range": (25, 35), "views": []},
        "35-45s": {"range": (35, 45), "views": []},
        "45-60s": {"range": (45, 60), "views": []},
        "1-3min": {"range": (60, 180), "views": []},
        "3-5min": {"range": (180, 300), "views": []},
        "5-10min": {"range": (300, 600), "views": []},
        "10min+": {"range": (600, 99999), "views": []},
    }
    
    for v in videos:
        dur = v.get("duration_sec", 0)
        for key, bucket in duration_buckets.items():
            if bucket["range"][0] <= dur < bucket["range"][1]:
                bucket["views"].append(v["views"])
                break
    
    return {k: {"avg_views": sum(b["views"])/len(b["views"]) if b["views"] else 0, 
                "count": len(b["views"])}
            for k, b in duration_buckets.items() if b["views"]}

def analyze_topics(videos):
    """Detectar temas y su rendimiento"""
    topic_keywords = {
        "bitcoin": r"bitcoin|btc|halving",
        "ethereum": r"ethereum|eth",
        "solana": r"solana|sol",
        "trading": r"trading|trader|operar",
        "stoploss": r"stop loss|stop-loss",
        "dca": r"dca|dollar cost",
        "miedo_emociones": r"miedo|emociones|pánico|codicia",
        "bot_automatico": r"bot|robot|automát|algorit",
        "educativo": r"qué es|cómo|aprende|explica",
        "motivacional": r"éxito|millón|rico|secreto",
        "controversial": r"error|pierd|mentira|verdad|nunca",
    }
    
    results = {}
    for topic, regex in topic_keywords.items():
        matching = [v for v in videos if re.search(regex, v["title"], re.IGNORECASE)]
        if matching:
            results[topic] = {
                "avg_views": sum(v["views"] for v in matching) / len(matching),
                "count": len(matching),
                "total_views": sum(v["views"] for v in matching),
                "best_video": max(matching, key=lambda v: v["views"])["title"][:40]
            }
    
    return dict(sorted(results.items(), key=lambda x: x[1]["avg_views"], reverse=True))

def detect_velocity(current, previous):
    """Detectar videos que están creciendo rápido"""
    if not previous:
        return []
    
    prev_map = {v["id"]: v["views"] for v in previous.get("youtube", {}).get("videos", [])}
    growing = []
    
    for v in current.get("youtube", {}).get("videos", []):
        prev_views = prev_map.get(v["id"], 0)
        delta = v["views"] - prev_views
        if delta > 10:  # Más de 10 views nuevas
            growing.append({
                "title": v["title"],
                "delta": delta,
                "total": v["views"],
                "type": v["type"]
            })
    
    return sorted(growing, key=lambda x: x["delta"], reverse=True)[:10]

def main():
    print("🧠 BitTrader Analyzer\n")
    
    current = load_latest_data()
    if not current:
        print("❌ No hay datos para analizar")
        return
    
    previous = load_previous_data()
    videos = current["youtube"]["videos"]
    shorts = [v for v in videos if v["type"] == "short"]
    longs = [v for v in videos if v["type"] == "long"]
    
    print(f"📊 Analizando {len(videos)} videos ({len(shorts)} shorts, {len(longs)} largos)\n")
    
    # 1. Classify
    short_class = classify_videos(shorts)
    long_class = classify_videos(longs)
    
    print("🟢 Top Shorts:")
    for v in short_class["winners"][:5]:
        print(f"   {v['views']:>5} views | {v['title'][:45]}")
    
    print("\n🟢 Top Videos Largos:")
    for v in long_class["winners"][:5]:
        print(f"   {v['views']:>5} views | {v['title'][:45]}")
    
    # 2. Title patterns
    title_patterns = extract_title_patterns(videos)
    print("\n📝 Patrones de Título:")
    for k, v in list(title_patterns.items())[:5]:
        print(f"   {k}: {v['avg_views']:.0f} avg views ({v['count']} videos)")
    
    # 3. Timing
    timing = analyze_timing(videos)
    print(f"\n⏰ Mejor día: {timing['best_day']} ({timing['best_day_avg']:.0f} avg)")
    print(f"   Mejor hora: {timing['best_hour_utc']}:00 UTC ({timing['best_hour_avg']:.0f} avg)")
    
    # 4. Duration
    duration = analyze_duration(videos)
    print("\n⏱️ Duración óptima:")
    for k, v in duration.items():
        print(f"   {k}: {v['avg_views']:.0f} avg views ({v['count']} videos)")
    
    # 5. Topics
    topics = analyze_topics(videos)
    print("\n🏷️ Temas que funcionan:")
    for k, v in list(topics.items())[:5]:
        print(f"   {k}: {v['avg_views']:.0f} avg ({v['count']} videos)")
    
    # 6. Velocity
    velocity = detect_velocity(current, previous)
    if velocity:
        print("\n🚀 Videos creciendo rápido:")
        for v in velocity[:5]:
            print(f"   +{v['delta']} views | {v['title'][:40]}")
    
    # 7. Trending crypto
    trending = current.get("crypto_trending", [])
    market = current.get("market_overview", {})
    
    # Build insights
    insights = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "channel": current["youtube"]["channel"],
        "shorts_analysis": {
            "total": len(shorts),
            "winners": [{"title": v["title"], "views": v["views"]} for v in short_class["winners"]],
            "losers": [{"title": v["title"], "views": v["views"]} for v in short_class["losers"]],
        },
        "longs_analysis": {
            "total": len(longs),
            "winners": [{"title": v["title"], "views": v["views"]} for v in long_class["winners"]],
            "losers": [{"title": v["title"], "views": v["views"]} for v in long_class["losers"]],
        },
        "title_patterns": title_patterns,
        "timing": timing,
        "duration": duration,
        "topics": topics,
        "velocity": velocity,
        "crypto_trending": trending,
        "market": market,
        "recommendations": {
            "best_title_pattern": list(title_patterns.keys())[0] if title_patterns else "unknown",
            "best_topic": list(topics.keys())[0] if topics else "unknown",
            "best_duration_shorts": max(
                {k: v for k, v in duration.items() if "s" in k}.items(),
                key=lambda x: x[1]["avg_views"], default=("30-40s", {})
            )[0] if duration else "30-40s",
            "avoid_topics": [k for k, v in topics.items() if v["avg_views"] < 10][:3],
        }
    }
    
    INSIGHTS_FILE.write_text(json.dumps(insights, indent=2, ensure_ascii=False))
    print(f"\n✅ Insights guardados: {INSIGHTS_FILE}")
    
    return insights

if __name__ == "__main__":
    main()
