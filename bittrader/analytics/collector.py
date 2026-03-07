#!/usr/bin/env python3
"""
BitTrader Analytics Collector
Recolecta stats de YouTube cada 12h + trending crypto
"""
import json, os, sys, re
from datetime import datetime, timezone
from pathlib import Path

# Setup paths
WORKSPACE = Path("/home/enderj/.openclaw/workspace")
DATA_DIR = WORKSPACE / "bittrader/analytics/data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# YouTube setup
sys.path.insert(0, str(WORKSPACE / "youtube_env/lib/python3.13/site-packages"))
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

def get_youtube():
    with open(WORKSPACE / "memory/youtube_credentials.json") as f:
        d = json.load(f)
    creds = Credentials(
        token=d["access_token"], refresh_token=d["refresh_token"],
        token_uri=d["token_uri"], client_id=d["client_id"],
        client_secret=d["client_secret"], scopes=d["scope"].split()
    )
    if creds.expired:
        creds.refresh(Request())
    return build('youtube', 'v3', credentials=creds)

def parse_duration(d):
    """ISO 8601 duration → seconds"""
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', d)
    if match:
        return int(match.group(1) or 0)*3600 + int(match.group(2) or 0)*60 + int(match.group(3) or 0)
    return 0

def collect_youtube_stats():
    """Recolectar stats de todos los videos"""
    yt = get_youtube()
    
    # Get channel
    channel = yt.channels().list(mine=True, part="statistics,id").execute()
    ch = channel["items"][0]
    channel_stats = {
        "subscribers": int(ch["statistics"]["subscriberCount"]),
        "total_views": int(ch["statistics"]["viewCount"]),
        "total_videos": int(ch["statistics"]["videoCount"])
    }
    
    # Get all videos
    uploads = yt.channels().list(id=ch["id"], part="contentDetails").execute()
    playlist_id = uploads["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    
    all_videos = []
    next_page = None
    
    while True:
        playlist = yt.playlistItems().list(
            playlistId=playlist_id, part="snippet,contentDetails",
            maxResults=50, pageToken=next_page
        ).execute()
        
        for item in playlist["items"]:
            vid = item["contentDetails"]["videoId"]
            title = item["snippet"]["title"]
            date = item["snippet"]["publishedAt"]
            
            # Get detailed stats
            details = yt.videos().list(id=vid, part="statistics,contentDetails").execute()
            if details["items"]:
                s = details["items"][0]["statistics"]
                dur = parse_duration(details["items"][0]["contentDetails"]["duration"])
                
                video_type = "short" if dur <= 60 else "long"
                
                all_videos.append({
                    "id": vid,
                    "title": title,
                    "published": date,
                    "type": video_type,
                    "duration_sec": dur,
                    "views": int(s.get("viewCount", 0)),
                    "likes": int(s.get("likeCount", 0)),
                    "comments": int(s.get("commentCount", 0)),
                })
        
        next_page = playlist.get("nextPageToken")
        if not next_page:
            break
    
    return {"channel": channel_stats, "videos": all_videos}

def collect_crypto_trending():
    """Trending crypto desde CoinGecko"""
    import requests
    try:
        resp = requests.get("https://api.coingecko.com/api/v3/search/trending", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            coins = [{"name": c["item"]["name"], "symbol": c["item"]["symbol"],
                      "rank": c["item"]["market_cap_rank"]}
                     for c in data.get("coins", [])[:10]]
            return coins
    except:
        pass
    return []

def collect_crypto_news():
    """Noticias crypto principales"""
    import requests
    try:
        # CoinGecko global data for market overview
        resp = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        if resp.status_code == 200:
            data = resp.json()["data"]
            return {
                "total_market_cap_usd": data["total_market_cap"]["usd"],
                "market_cap_change_24h": data["market_cap_change_percentage_24h_usd"],
                "btc_dominance": data["market_cap_percentage"]["btc"],
                "active_cryptos": data["active_cryptocurrencies"]
            }
    except:
        pass
    return {}

def main():
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%d_%H")
    
    print(f"📊 BitTrader Collector — {now.strftime('%Y-%m-%d %H:%M UTC')}")
    
    # 1. YouTube stats
    print("   📺 Recolectando YouTube stats...")
    yt_data = collect_youtube_stats()
    print(f"      ✅ {len(yt_data['videos'])} videos | {yt_data['channel']['subscribers']} subs")
    
    # Separar shorts vs largos
    shorts = [v for v in yt_data["videos"] if v["type"] == "short"]
    longs = [v for v in yt_data["videos"] if v["type"] == "long"]
    print(f"      📹 {len(shorts)} shorts | {len(longs)} videos largos")
    
    # 2. Crypto trending
    print("   🪙 Recolectando crypto trending...")
    trending = collect_crypto_trending()
    print(f"      ✅ {len(trending)} trending coins")
    
    # 3. Market overview
    print("   📈 Recolectando market overview...")
    market = collect_crypto_news()
    if market:
        change = market.get("market_cap_change_24h", 0)
        print(f"      ✅ Market {'🟢' if change > 0 else '🔴'} {change:+.1f}% 24h")
    
    # Save
    report = {
        "timestamp": now.isoformat(),
        "youtube": yt_data,
        "crypto_trending": trending,
        "market_overview": market
    }
    
    output = DATA_DIR / f"{timestamp}.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n✅ Guardado: {output}")
    
    return report

if __name__ == "__main__":
    main()
