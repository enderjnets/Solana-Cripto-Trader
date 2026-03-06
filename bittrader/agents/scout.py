#!/usr/bin/env python3
"""
🔭 BitTrader Scout — Agente Investigador
Recolecta stats YouTube, trending crypto, y detecta oportunidades de contenido.
Ejecutar: python3 agents/scout.py
"""
import json
import os
import re
import sys
import time
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import Counter

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE   = Path("/home/enderj/.openclaw/workspace")
BITTRADER   = WORKSPACE / "bittrader"
DATA_DIR    = BITTRADER / "agents/data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# YouTube env
sys.path.insert(0, str(WORKSPACE / "youtube_env/lib/python3.13/site-packages"))

# ── Config ─────────────────────────────────────────────────────────────────
# Claude Sonnet 4.6 (local API) - PRIMARY
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")
CLAUDE_BASE_URL = "http://127.0.0.1:8443/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-6"

# GLM-4.7 Coding Plan - FALLBACK #1 (más rápido que GLM-5)
ZAI_CODING_KEY = "863f222d3de340df8b6ff7e1e36ab216.DFOH1veovvWzaoFT"
ZAI_CODING_BASE_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
ZAI_CODING_MODEL = "glm-4.7"  # Cambiado de glm-5 a glm-4.7 (2.0s vs 2.6s)

# MiniMax M2.5 - FALLBACK #2 (Anthropic API compatible)
MINIMAX_KEY_FILE = BITTRADER / "keys/minimax.json"
MINIMAX_KEY = json.loads(MINIMAX_KEY_FILE.read_text())["minimax_coding_key"] if MINIMAX_KEY_FILE.exists() else ""
MINIMAX_URL = "https://api.minimax.io/anthropic/v1/messages"
MINIMAX_MODEL = "MiniMax-M2.5"

YT_CREDS    = WORKSPACE / "memory/youtube_credentials.json"

COINGECKO_BASE = "https://api.coingecko.com/api/v3"


# ══════════════════════════════════════════════════════════════════════════
# YOUTUBE
# ══════════════════════════════════════════════════════════════════════════

def get_youtube_client():
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
        # Persist updated token
        d["access_token"] = creds.token
        YT_CREDS.write_text(json.dumps(d, indent=2))
    return build("youtube", "v3", credentials=creds)


def parse_iso_duration(d: str) -> int:
    m = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', d or "")
    if m:
        return int(m.group(1) or 0)*3600 + int(m.group(2) or 0)*60 + int(m.group(3) or 0)
    return 0


def collect_youtube_stats(yt) -> dict:
    print("  📊 Recolectando estadísticas de YouTube...")

    # Channel info
    ch_resp = yt.channels().list(mine=True, part="statistics,contentDetails,id").execute()
    ch = ch_resp["items"][0]
    channel_stats = {
        "subscribers":   int(ch["statistics"].get("subscriberCount", 0)),
        "total_views":   int(ch["statistics"].get("viewCount", 0)),
        "total_videos":  int(ch["statistics"].get("videoCount", 0)),
    }
    playlist_id = ch["contentDetails"]["relatedPlaylists"]["uploads"]

    # Fetch all videos
    all_videos = []
    next_page = None
    while True:
        pl = yt.playlistItems().list(
            playlistId=playlist_id, part="snippet,contentDetails",
            maxResults=50, pageToken=next_page
        ).execute()

        video_ids = [item["contentDetails"]["videoId"] for item in pl["items"]]

        # Batch-fetch details
        details = yt.videos().list(
            id=",".join(video_ids),
            part="statistics,contentDetails,snippet"
        ).execute()

        for item in details["items"]:
            stats = item["statistics"]
            dur   = parse_iso_duration(item["contentDetails"]["duration"])
            pub   = item["snippet"]["publishedAt"]
            pub_dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))

            all_videos.append({
                "id":           item["id"],
                "title":        item["snippet"]["title"],
                "published_at": pub,
                "day_of_week":  pub_dt.strftime("%A"),
                "hour_utc":     pub_dt.hour,
                "duration_s":   dur,
                "type":         "short" if dur <= 62 else "long",
                "views":        int(stats.get("viewCount", 0)),
                "likes":        int(stats.get("likeCount", 0)),
                "comments":     int(stats.get("commentCount", 0)),
                "vph":          0,  # views per hour — calculated below
            })

        next_page = pl.get("nextPageToken")
        if not next_page:
            break

    # Calculate views-per-hour for each video
    now = datetime.now(timezone.utc)
    for v in all_videos:
        pub_dt = datetime.fromisoformat(v["published_at"].replace("Z", "+00:00"))
        age_h  = max(1, (now - pub_dt).total_seconds() / 3600)
        v["vph"] = round(v["views"] / age_h, 2)
        v["age_hours"] = round(age_h, 1)

    shorts = [v for v in all_videos if v["type"] == "short"]
    longs  = [v for v in all_videos if v["type"] == "long"]

    print(f"  ✅ {len(all_videos)} videos: {len(shorts)} shorts + {len(longs)} largos")
    return {
        "channel":  channel_stats,
        "shorts":   shorts,
        "longs":    longs,
        "all":      all_videos,
    }


# ══════════════════════════════════════════════════════════════════════════
# COINGECKO
# ══════════════════════════════════════════════════════════════════════════

def fetch_coingecko(endpoint: str, params: dict = None) -> dict | list:
    url = f"{COINGECKO_BASE}/{endpoint}"
    headers = {"Accept": "application/json"}
    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()


def collect_crypto_trending() -> dict:
    print("  📈 Obteniendo tendencias crypto de CoinGecko...")
    result = {}

    # Global market
    try:
        global_data = fetch_coingecko("global")
        m = global_data.get("data", {})
        result["global"] = {
            "total_market_cap_usd":   m.get("total_market_cap", {}).get("usd", 0),
            "total_volume_usd":       m.get("total_volume", {}).get("usd", 0),
            "btc_dominance":          round(m.get("market_cap_percentage", {}).get("btc", 0), 2),
            "eth_dominance":          round(m.get("market_cap_percentage", {}).get("eth", 0), 2),
            "market_cap_change_24h":  round(m.get("market_cap_change_percentage_24h_usd", 0), 2),
        }
        time.sleep(1)
    except Exception as e:
        print(f"    ⚠️ global falló: {e}")
        result["global"] = {}

    # Trending coins
    try:
        trending = fetch_coingecko("search/trending")
        result["trending_coins"] = [
            {
                "id":     c["item"]["id"],
                "name":   c["item"]["name"],
                "symbol": c["item"]["symbol"].upper(),
                "rank":   c["item"].get("market_cap_rank"),
            }
            for c in trending.get("coins", [])[:10]
        ]
        time.sleep(1)
    except Exception as e:
        print(f"    ⚠️ trending falló: {e}")
        result["trending_coins"] = []

    # Top movers (top 50 by market cap, sorted by 24h change)
    try:
        coins = fetch_coingecko("coins/markets", {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 50,
            "page": 1,
            "price_change_percentage": "24h",
        })
        sorted_coins = sorted(coins, key=lambda x: abs(x.get("price_change_percentage_24h") or 0), reverse=True)
        result["top_movers"] = [
            {
                "id":           c["id"],
                "name":         c["name"],
                "symbol":       c["symbol"].upper(),
                "price_usd":    c["current_price"],
                "change_24h":   round(c.get("price_change_percentage_24h") or 0, 2),
                "market_cap":   c.get("market_cap", 0),
                "volume_24h":   c.get("total_volume", 0),
            }
            for c in sorted_coins[:10]
        ]

        # BTC specifically
        btc = next((c for c in coins if c["id"] == "bitcoin"), None)
        if btc:
            result["bitcoin"] = {
                "price_usd":  btc["current_price"],
                "change_24h": round(btc.get("price_change_percentage_24h") or 0, 2),
                "change_7d":  round(btc.get("price_change_percentage_7d_in_currency") or 0, 2),
                "market_cap": btc.get("market_cap", 0),
            }
        time.sleep(1)
    except Exception as e:
        print(f"    ⚠️ top movers falló: {e}")
        result["top_movers"] = []

    # Narratives from trending
    try:
        nfts = fetch_coingecko("search/trending")
        result["trending_nfts"] = [
            n.get("name", "?") for n in nfts.get("nfts", [])[:5]
        ]
        result["trending_categories"] = [
            c.get("name", "?") for c in nfts.get("categories", [])[:5]
        ]
        time.sleep(1)
    except Exception as e:
        result["trending_nfts"] = []
        result["trending_categories"] = []

    print(f"  ✅ Crypto: BTC={result.get('bitcoin', {}).get('price_usd','?')} | {len(result.get('top_movers',[]))} movers")
    return result


# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def classify_videos(videos: list) -> dict:
    if not videos:
        return {"winners": [], "average": [], "losers": []}
    ranked = sorted(videos, key=lambda v: v["views"], reverse=True)
    n       = len(ranked)
    top20   = max(1, n // 5)
    bot20   = max(1, n // 5)
    return {
        "winners": ranked[:top20],
        "average": ranked[top20: n - bot20] if n > 2 else [],
        "losers":  ranked[-bot20:],
    }


def extract_title_patterns(videos: list) -> dict:
    patterns = {
        "pregunta":      r"^[¿?]|^qu[eé]|^c[oó]mo|^por\s+qu[eé]|^cu[aá]l",
        "numero":        r"\b\d+\b",
        "emoji_inicio":  r"^[^\w\s]",
        "mayusculas":    r"[A-ZÁÉÍÓÚÑ]{3,}",
        "negativo":      r"error|pierd|miedo|nunca|no |fail|crash|caída",
        "secreto":       r"secreto|nadie|oculto|verdad|revelado",
        "lista":         r"^\d+\s+|\blos\s+\d+|\blas\s+\d+",
        "urgencia":      r"ahora|hoy|ya|urgente|alerta|rompió|pump",
    }
    data = {k: {"views": [], "count": 0} for k in patterns}
    for v in videos:
        t = v["title"].lower()
        for k, pat in patterns.items():
            if re.search(pat, t, re.IGNORECASE):
                data[k]["views"].append(v["views"])
                data[k]["count"] += 1

    result = {}
    for k, d in data.items():
        if d["views"]:
            result[k] = {
                "count":     d["count"],
                "avg_views": round(sum(d["views"]) / len(d["views"]), 0),
                "total_views": sum(d["views"]),
            }
    return dict(sorted(result.items(), key=lambda x: x[1]["avg_views"], reverse=True))


def analyze_publish_times(videos: list) -> dict:
    days  = Counter()
    hours = Counter()
    for v in videos:
        pub = datetime.fromisoformat(v["published_at"].replace("Z", "+00:00"))
        days[pub.strftime("%A")]  += 1
        hours[pub.hour]           += 1
    return {
        "by_day":  dict(days.most_common()),
        "by_hour": {str(h): c for h, c in sorted(hours.items())},
        "best_day":  days.most_common(1)[0][0] if days else "unknown",
        "best_hour": hours.most_common(1)[0][0] if hours else 12,
    }


def detect_velocity(current_videos: list, previous_file: Path) -> list:
    """Detectar videos que están acelerando (despegando)."""
    if not previous_file.exists():
        return []
    prev_data = json.loads(previous_file.read_text())
    prev_map  = {v["id"]: v for v in prev_data.get("all_videos", [])}

    rising = []
    for v in current_videos:
        prev = prev_map.get(v["id"])
        if not prev:
            continue
        delta = v["views"] - prev["views"]
        if delta > 500 and delta > prev["views"] * 0.1:
            rising.append({**v, "view_delta": delta, "delta_pct": round(delta/max(1,prev["views"])*100,1)})

    return sorted(rising, key=lambda x: x["view_delta"], reverse=True)[:5]


def detect_saturated_topics(videos: list) -> list:
    """Detectar keywords que aparecen en múltiples losers."""
    losers = [v for v in videos if v.get("category") == "loser"]
    word_counts = Counter()
    for v in losers:
        words = re.findall(r'\b\w{4,}\b', v["title"].lower())
        word_counts.update(words)
    # Keywords with 2+ losers are potentially saturated
    return [w for w, c in word_counts.most_common(10) if c >= 2]


def detect_urgent_alert(crypto: dict) -> dict | None:
    """Genera alerta urgente si hay evento significativo."""
    btc = crypto.get("bitcoin", {})
    change = btc.get("change_24h", 0)
    movers = crypto.get("top_movers", [])

    alerts = []
    if abs(change) >= 10:
        direction = "subió" if change > 0 else "cayó"
        alerts.append(f"🚨 ALERTA: BTC {direction} {abs(change):.1f}% en 24h — precio: ${btc.get('price_usd','?'):,}")

    for coin in movers[:3]:
        c = coin.get("change_24h", 0)
        if abs(c) >= 20:
            direction = "pump" if c > 0 else "dump"
            alerts.append(f"⚡ {coin['symbol']} {direction} {abs(c):.1f}% en 24h")

    mkt_change = crypto.get("global", {}).get("market_cap_change_24h", 0)
    if abs(mkt_change) >= 8:
        direction = "subió" if mkt_change > 0 else "cayó"
        alerts.append(f"📊 Market cap total {direction} {abs(mkt_change):.1f}% — posible movimiento macro")

    if alerts:
        return {
            "urgent": True,
            "alerts": alerts,
            "suggested_content": "Crea un short de NOTICIAS URGENTES ahora mismo",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    return None


def extract_content_gaps(winners: list, all_videos: list, trending_coins: list) -> list:
    """Detecta temas trending que el canal no ha cubierto recientemente."""
    recent_titles = " ".join(v["title"].lower() for v in all_videos[-20:])
    gaps = []
    for coin in trending_coins:
        name   = coin["name"].lower()
        symbol = coin["symbol"].lower()
        if name not in recent_titles and symbol not in recent_titles:
            gaps.append(f"{coin['symbol']} ({coin['name']})")
    return gaps[:5]


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def run_scout(skip_youtube: bool = False, no_llm: bool = False) -> dict:
    print("\n🔭 BitTrader Scout iniciando...")
    timestamp = datetime.now(timezone.utc)
    date_str  = timestamp.strftime("%Y-%m-%d")

    report = {
        "generated_at": timestamp.isoformat(),
        "date":         date_str,
        "youtube":      {},
        "crypto":       {},
        "analysis":     {},
        "alert":        None,
    }

    # ── YouTube ─────────────────────────────────────────────────────────
    if not skip_youtube:
        try:
            yt   = get_youtube_client()
            yt_data = collect_youtube_stats(yt)

            shorts_classified = classify_videos(yt_data["shorts"])
            longs_classified  = classify_videos(yt_data["longs"])

            # Tag each video
            for cat, vids in shorts_classified.items():
                for v in vids:
                    v["category"] = cat
            for cat, vids in longs_classified.items():
                for v in vids:
                    v["category"] = cat

            all_videos = yt_data["shorts"] + yt_data["longs"]

            # Velocity detection
            prev_file  = DATA_DIR / "scout_latest.json"
            rising     = detect_velocity(all_videos, prev_file)

            report["youtube"] = {
                "channel":             yt_data["channel"],
                "all_videos":          all_videos,
                "shorts_classified":   shorts_classified,
                "longs_classified":    longs_classified,
                "rising_videos":       rising,
                "saturated_topics":    detect_saturated_topics(all_videos),
                "title_patterns":      extract_title_patterns(all_videos),
                "publish_times":       analyze_publish_times(all_videos),
                "winners_titles":      [v["title"] for v in shorts_classified["winners"] + longs_classified["winners"]],
                "losers_titles":       [v["title"] for v in shorts_classified["losers"] + longs_classified["losers"]],
            }
            print(f"  ✅ YouTube: {len(all_videos)} videos analizados")
        except Exception as e:
            print(f"  ⚠️ YouTube falló (¿quota?): {e}")
            report["youtube"] = {"error": str(e)}
    else:
        print("  ⏭️  YouTube omitido (--no-youtube)")
        report["youtube"] = {"skipped": True}

    # ── Crypto ───────────────────────────────────────────────────────────
    print("\n🪙 Recolectando datos crypto...")
    try:
        crypto = collect_crypto_trending()
        report["crypto"] = crypto
    except Exception as e:
        print(f"  ⚠️ Crypto falló: {e}")
        report["crypto"] = {"error": str(e)}

    # ── Analysis ─────────────────────────────────────────────────────────
    print("\n🧠 Analizando...")
    all_vids   = report["youtube"].get("all_videos", [])
    winners    = report["youtube"].get("shorts_classified", {}).get("winners", []) + \
                 report["youtube"].get("longs_classified", {}).get("winners", [])
    trending   = report["crypto"].get("trending_coins", [])

    report["analysis"] = {
        "content_gaps":    extract_content_gaps(winners, all_vids, trending),
        "top_winners":     sorted(winners, key=lambda v: v["views"], reverse=True)[:5],
        "best_publish_day":  report["youtube"].get("publish_times", {}).get("best_day", "Tuesday"),
        "best_publish_hour": report["youtube"].get("publish_times", {}).get("best_hour", 14),
        "recommended_topics": _build_recommended_topics(report["crypto"], use_llm=not no_llm),
        "saturated_avoid":   report["youtube"].get("saturated_topics", []),
    }

    # ── Urgent alert ─────────────────────────────────────────────────────
    alert = detect_urgent_alert(report["crypto"])
    if alert:
        report["alert"] = alert
        print(f"\n🚨 ALERTA URGENTE DETECTADA: {alert['alerts']}")

    # ── Save ─────────────────────────────────────────────────────────────
    out_file    = DATA_DIR / f"scout_report_{date_str}.json"
    latest_file = DATA_DIR / "scout_latest.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    latest_file.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n✅ Scout completado → {out_file.name}")
    if alert:
        print(f"🚨 {len(alert['alerts'])} alertas urgentes generadas")
    return report


# ════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
# LLM CALLS (Claude Sonnet 4.6 PRIMARY -> GLM-5 FALLBACK)
# ══════════════════════════════════════════════════════════════════════

def call_claude_sonnet(prompt: str, system_prompt: str = None) -> str:
    """Llamar a Claude Sonnet 4.6 (PRIMARY) para análisis de contenido"""
    import requests

    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }

    if CLAUDE_API_KEY:
        headers["x-api-key"] = CLAUDE_API_KEY

    messages = []
    if system_prompt:
        messages.append({"role": "user", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    data = {
        "model": CLAUDE_MODEL,
        "max_tokens": 2000,
        "messages": messages
    }

    try:
        response = requests.post(CLAUDE_BASE_URL, json=data, headers=headers, timeout=15)  # Timeout reducido
        response.raise_for_status()
        result = response.json()

        if "content" in result:
            for block in result["content"]:
                if block.get("type") == "text":
                    return block["text"]
            return result.get("content", [{}])[0].get("text", "")
        else:
            return str(result)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print(f"  ⚠️ Claude Sonnet rate limit (429) - usando fallback")
            return None
        print(f"  ⚠️ Claude Sonnet HTTP error: {e}")
        return None
    except Exception as e:
        print(f"  ⚠️ Claude Sonnet error: {e}")
        return None


def call_glm_4_7(prompt: str, system_prompt: str = None, max_retries: int = 2) -> str:
    """Llamar a GLM-4.7 con retry para manejar timeouts intermitentes"""
    import requests
    import time

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ZAI_CODING_KEY}"
    }

    # Construir el cuerpo del mensaje
    messages = []
    if system_prompt:
        messages.append({"role": "user", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    data = {
        "model": ZAI_CODING_MODEL,
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.7
    }

    for attempt in range(max_retries + 1):
        try:
            timeout = 10 + (attempt * 5)  # 10s, 15s, 20s
            response = requests.post(ZAI_CODING_BASE_URL, json=data, headers=headers, timeout=timeout)
            response.raise_for_status()
            result = response.json()

            # Parsear respuesta de GLM-4.7 (formato OpenAI-compatible)
            # Endpoint de coding devuelve reasoning_content, content puede estar vacío
            if "choices" in result and len(result["choices"]) > 0:
                msg = result["choices"][0]["message"]
                content = msg.get("content", "")
                reasoning = msg.get("reasoning_content", "")
                # Usar reasoning_content si content está vacío (endpoint de coding)
                if not content and reasoning:
                    return reasoning
                return content or reasoning or str(result)
            else:
                return str(result)

        except requests.exceptions.Timeout:
            print(f"  ⚠️ GLM-4.7 timeout (intento {attempt + 1}/{max_retries + 1})")
            if attempt < max_retries:
                time.sleep(1)  # Backoff corto
        except Exception as e:
            print(f"  ⚠️ GLM-4.7 error (intento {attempt + 1}/{max_retries + 1}): {type(e).__name__}")
            if attempt == max_retries:
                return None
            time.sleep(1)

    return None


def call_minimax(prompt: str, system_prompt: str = None) -> str:
    """Llamar a MiniMax M2.5 usando Anthropic API compatible"""
    if not MINIMAX_KEY:
        print(f"  ⚠️ MiniMax key no configurada")
        return None

    headers = {
        "x-api-key": MINIMAX_KEY,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    messages = []
    if system_prompt:
        messages.append({"role": "user", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MINIMAX_MODEL,
        "max_tokens": 2000,
        "messages": messages,
    }
    try:
        response = requests.post(MINIMAX_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        # Parsear respuesta Anthropic API compatible
        for block in data.get("content", []):
            if block.get("type") == "text":
                return block.get("text", "").strip()
        return None
    except Exception as e:
        print(f"  ⚠️ MiniMax error: {e}")
        return None


def call_llm(prompt: str, system_prompt: str = None, use_glm_fallback: bool = False) -> str:
    """Llamar al LLM con fallback: Claude Sonnet 4.6 PRIMARY -> MiniMax M2.5 FALLBACK"""
    # Intentar Claude Sonnet 4.6 primero
    result = call_claude_sonnet(prompt, system_prompt)
    if result:
        return result

    print("  ⚠️ Claude Sonnet falló")

    # Intentar MiniMax M2.5 como fallback (GLM-4.7 deshabilitado por inestabilidad)
    print("  🔄 Intentando MiniMax M2.5 fallback...")
    result = call_minimax(prompt, system_prompt)
    if result:
        return result

    print("  ❌ Todos los LLM fallaron")
    return None


def _analyze_with_llm(crypto: dict, youtube_data: dict) -> list:
    """Usar GLM-5 para generar temas recomendados inteligentes"""
    btc = crypto.get("bitcoin", {})
    trending = crypto.get("trending_coins", [])
    movers = crypto.get("top_movers", [])
    winners = youtube_data.get("winners_titles", [])
    losers = youtube_data.get("losers_titles", [])

    # Preparar contexto para Claude Sonnet
    context = f"""
BITTRADER CHANNEL CONTEXT:
- Canal: Trading en TODOS los mercados (crypto, futuros, fondeo) + IA para Trading
- Subs: ~3,000
- Videos recientes exitosos: {winners[:3] if winners else 'N/A'}
- Videos con bajo rendimiento: {losers[:3] if losers else 'N/A'}

CRYPTO DATA:
- BTC: ${btc.get('price_usd', 0):,.0f} ({btc.get('change_24h', 0):+.1f}%)
- Trending coins: {', '.join([c['symbol'] for c in trending[:5]])}
- Top movers: {', '.join([f"{c['symbol']} {c['change_24h']:+.1f}%" for c in movers[:3]])}

CATEGORÍAS DE CONTENIDO:
- Cripto: Bitcoin, altcoins, DeFi
- Futuros: NAS100, S&P500, commodities
- Fondeo: FTMO, Topstep, prop firms
- IA + Trading: Agentes, bots, automatización, Claude, LLMs en trading
- Educación: Estrategias, gestión de riesgo, psicología

INSTRUCCIONES:
Genera 8-10 ideas de video para BitTrader (canales de trading/finanzas).
Cada idea debe tener:
1. type: "noticia" | "educativo" | "controversial" | "tutorial" | "ia_trading"
2. priority: "alta" | "media" | "baja"
3. title: Título atractivo (max 60 chars)
4. format: "short" | "long" (short <60s, long >3min)

FORMATO DE RESPUESTA (JSON):
[
  {{"type": "...", "priority": "...", "title": "...", "format": "..."}},
  ...
]

Reglas:
- Usa datos reales del crypto arriba
- Evita temas ya cubiertos en videos exitosos
- Mix de short (rápido viral) y long (SEO/ancla)
- Prioriza temas que están trended ahora mismo
- INCLUYE temas de IA + Trading: agentes, Claude, automatización con LLMs
"""

    print("  🧠 Generando temas con Claude Sonnet 4.6 (PRIMARY)...")
    system_prompt = """Eres un experto en contenido de trading y finanzas para YouTube. Entiendes:
- Qué hace viral un video de trading
- Qué temas generan engagement y comentarios
- Cómo balancear contenido educativo vs noticias vs análisis
- Diferencias entre shorts (30-60s) y videos largos (3-8min)"""

    response = call_llm(context, system_prompt, use_glm_fallback=True)

    if not response:
        return None

    # Parsear respuesta JSON
    try:
        import json as json_lib
        # Limpiar respuesta (remover markdown si existe)
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        topics = json_lib.loads(response)

        # Validar formato
        validated = []
        for t in topics:
            if all(k in t for k in ["type", "priority", "title", "format"]):
                validated.append(t)
            else:
                print(f"  ⚠️ Tema inválido: {t}")

        print(f"  ✅ LLM generó {len(validated)} temas válidos")
        return validated[:10]
    except Exception as e:
        print(f"  ⚠️ Error parseando respuesta LLM: {e}")
        print(f"     Response: {response[:200]}...")
        return None


def _build_recommended_topics(crypto: dict, use_llm: bool = True) -> list:
    """Generar temas recomendados, opcionalmente usando LLM o solo fallback manual."""
    # Solo intentar LLM si use_llm=True
    if use_llm:
        try:
            # Recuperar datos del reporte
            import json
            latest_file = DATA_DIR / "scout_latest.json"
            if latest_file.exists():
                latest = json.loads(latest_file.read_text())
                youtube_data = latest.get("youtube", {})
                topics = _analyze_with_llm(crypto, youtube_data)
                if topics:
                    return topics
        except Exception as e:
            print(f"  ⚠️ LLM error, usando fallback manual: {e}")
    else:
        print("  📝 Usando solo fallback manual (LLM desactivado)")

    # Fallback: lógica manual
    topics = []
    btc = crypto.get("bitcoin", {})
    if btc:
        price  = btc.get("price_usd", 0)
        change = btc.get("change_24h", 0)
        if abs(change) >= 5:
            direction = "sube" if change > 0 else "cae"
            topics.append({
                "type": "noticia", "priority": "alta",
                "title": f"Bitcoin {direction} {abs(change):.1f}%: ¿qué está pasando?",
                "format": "short",
            })
        topics.append({
            "type": "educativo", "priority": "media",
            "title": f"¿Por qué Bitcoin vale ${price:,.0f}? La respuesta te sorprenderá",
            "format": "short",
        })

    for coin in crypto.get("trending_coins", [])[:3]:
        topics.append({
            "type": "noticia", "priority": "media",
            "title": f"{coin['symbol']} está en tendencia — ¿vale la pena comprar?",
            "format": "short",
        })

    for mover in crypto.get("top_movers", [])[:2]:
        c = mover.get("change_24h", 0)
        if abs(c) >= 10:
            topics.append({
                "type": "noticia", "priority": "alta",
                "title": f"{mover['symbol']} {'+' if c>0 else ''}{c:.1f}% en 24h — análisis rápido",
                "format": "short",
            })

    # Static evergreen topics (always valid)
    topics += [
        # Trading tradicional
        {"type": "educativo", "priority": "media",
         "title": "La regla del 1% que usan los traders profesionales",
         "format": "short"},
        {"type": "educativo", "priority": "media",
         "title": "¿Cómo funciona NAS100? El índice que más mueve Wall Street",
         "format": "long"},
        {"type": "controversial", "priority": "baja",
         "title": "¿FTMO es una estafa? Lo que nadie te dice de las prop firms",
         "format": "short"},
        {"type": "educativo", "priority": "baja",
         "title": "5 errores que cometen los traders principiantes (y cómo evitarlos)",
         "format": "long"},

        # IA + Trading (Nueva categoría prioritaria)
        {"type": "ia_trading", "priority": "alta",
         "title": "Claude AI creó un bot de trading en 10 minutos — ¿funciona?",
         "format": "short"},
        {"type": "ia_trading", "priority": "alta",
         "title": "Agentes de trading con IA: el futuro de la automatización",
         "format": "long"},
        {"type": "ia_trading", "priority": "media",
         "title": "Cómo configurar un agente de IA que monitorea noticias 24/7",
         "format": "short"},
        {"type": "ia_trading", "priority": "media",
         "title": "Comparativa: Bot tradicional vs Agente de IA con LLM",
         "format": "long"},
        {"type": "ia_trading", "priority": "alta",
         "title": "La IA que predice movimientos del mercado antes que nadie",
         "format": "short"},
        {"type": "ia_trading", "priority": "media",
         "title": "Cómo usar Claude/GPT para analizar patrones de trading",
         "format": "short"},
        {"type": "ia_trading", "priority": "baja",
         "title": "Top 5 herramientas de IA para traders en 2025",
         "format": "long"},
        {"type": "ia_trading", "priority": "media",
         "title": "Bots que aprenden: trading con reinforcement learning",
         "format": "long"},
    ]
    return topics[:12]


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BitTrader Scout — Agente Investigador")
    parser.add_argument("--no-youtube", action="store_true", help="Omitir recolección YouTube (ahorra quota)")
    parser.add_argument("--crypto-only", action="store_true", help="Solo datos crypto")
    parser.add_argument("--no-llm", action="store_true", help="Desactivar LLM, usar solo fallback manual")
    args = parser.parse_args()

    skip_yt = args.no_youtube or args.crypto_only

    report = run_scout(skip_youtube=skip_yt, no_llm=args.no_llm)

    print("\n── Resumen ──────────────────────────────")
    crypto = report.get("crypto", {})
    btc = crypto.get("bitcoin", {})
    if btc:
        print(f"  BTC: ${btc.get('price_usd',0):,} ({btc.get('change_24h',0):+.1f}%)")
    trending = crypto.get("trending_coins", [])
    if trending:
        print(f"  🔥 Trending: {', '.join(c['symbol'] for c in trending[:5])}")
    movers = crypto.get("top_movers", [])
    if movers:
        top = movers[0]
        print(f"  ⚡ Top mover: {top['symbol']} {top['change_24h']:+.1f}%")
    topics = report.get("analysis", {}).get("recommended_topics", [])
    print(f"  💡 {len(topics)} temas recomendados generados")
    gaps = report.get("analysis", {}).get("content_gaps", [])
    if gaps:
        print(f"  🎯 Gaps: {', '.join(gaps)}")
    print("─────────────────────────────────────────\n")
