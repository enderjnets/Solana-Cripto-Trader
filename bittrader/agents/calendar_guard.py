#!/usr/bin/env python3
"""
📅 BitTrader Calendar Guard
Verifica que haya al menos 1 video programado para publicar en las próximas 24-48 horas.
Si no hay, lanza el pipeline para producir y programar nuevos videos.
"""
import json, sys, requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

YT_CREDS = Path("/home/enderj/.openclaw/workspace/memory/youtube_credentials.json")

def get_token():
    d = json.loads(YT_CREDS.read_text())
    r = requests.post('https://oauth2.googleapis.com/token', data={
        'client_id': d['client_id'],
        'client_secret': d['client_secret'],
        'refresh_token': d['refresh_token'],
        'grant_type': 'refresh_token'
    })
    return r.json().get('access_token')

def check_scheduled_videos(token):
    """Retorna videos programados en las próximas 48h."""
    now = datetime.now(timezone.utc)
    window_end = now + timedelta(hours=48)
    
    # Buscar videos del canal
    r = requests.get('https://www.googleapis.com/youtube/v3/search',
        params={'part': 'snippet', 'forMine': True, 'type': 'video', 
                'maxResults': 20, 'order': 'date'},
        headers={'Authorization': f'Bearer {token}'})
    
    items = r.json().get('items', [])
    if not items:
        return []
    
    ids = [i['id']['videoId'] for i in items]
    r2 = requests.get('https://www.googleapis.com/youtube/v3/videos',
        params={'part': 'snippet,status', 'id': ','.join(ids)},
        headers={'Authorization': f'Bearer {token}'})
    
    upcoming = []
    for item in r2.json().get('items', []):
        pub_at = item['status'].get('publishAt','')
        privacy = item['status'].get('privacyStatus','')
        
        # Video ya público hoy
        published = item['snippet'].get('publishedAt','')
        if privacy == 'public' and published:
            pub_dt = datetime.fromisoformat(published.replace('Z','+00:00'))
            if pub_dt >= now - timedelta(hours=12):
                upcoming.append({'title': item['snippet']['title'], 'when': 'HOY (ya público)', 'id': item['id']})
                continue
        
        # Video programado próximamente
        if pub_at and privacy in ['private','unlisted']:
            try:
                sched_dt = datetime.fromisoformat(pub_at.replace('Z','+00:00'))
                if now <= sched_dt <= window_end:
                    upcoming.append({'title': item['snippet']['title'], 'when': sched_dt.isoformat(), 'id': item['id']})
            except:
                pass
    
    return upcoming

def main():
    token = get_token()
    upcoming = check_scheduled_videos(token)
    
    now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    
    if upcoming:
        print(f"✅ CALENDAR OK — {len(upcoming)} video(s) en las próximas 48h:")
        for v in upcoming:
            print(f"   • {v['when'][:16]} | {v['title'][:50]}")
        return 0
    else:
        print(f"⚠️ ALERTA — No hay videos programados para las próximas 48h ({now_str})")
        print("Acción requerida: Lanzar pipeline de producción")
        return 1

if __name__ == '__main__':
    sys.exit(main())
