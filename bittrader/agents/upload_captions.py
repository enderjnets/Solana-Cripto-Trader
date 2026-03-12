#!/usr/bin/env python3
"""
Upload SRT captions to YouTube videos.
"""
import json
from pathlib import Path
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload

# Auth
creds = Credentials.from_authorized_user_file('/home/enderj/.openclaw/workspace/memory/youtube_credentials.json')
youtube = build('youtube', 'v3', credentials=creds)

# Video IDs and caption files
videos = [
    {
        'id': '2VEzHeGgDG8',
        'title': 'Le di mis trades a Claude',
        'srt': 'output/rhino_v1/claude_trades/captions.srt'
    },
    {
        'id': 'Bb2H6nChYwQ',
        'title': 'De $0 a cuenta fondeada',
        'srt': 'output/rhino_v1/fondeada/captions.srt'
    },
    {
        'id': '2RzldSPc2Ck',
        'title': 'El bot que lee noticias',
        'srt': 'output/rhino_v1/bot_noticias/captions.srt'
    }
]

print("=" * 80)
print("SUBIENDO SUBTÍTULOS A VIDEOS LARGOS")
print("=" * 80)

for video in videos:
    video_id = video['id']
    title = video['title']
    srt_path = Path(video['srt'])
    
    print(f"\n📤 {title}")
    print(f"   Video ID: {video_id}")
    print(f"   SRT: {srt_path}")
    
    if not srt_path.exists():
        print(f"   ❌ Archivo no encontrado")
        continue
    
    try:
        # Check if captions already exist
        captions = youtube.captions().list(
            part='snippet',
            videoId=video_id
        ).execute()
        
        # Try to insert new caption
        print(f"   📤 Subiendo nuevos subtítulos...")
        
        # Upload caption
        media = MediaFileUpload(str(srt_path), mimetype='application/x-subrip', resumable=True)
        
        # Insert new caption
        result = youtube.captions().insert(
            part='snippet',
            body={
                'snippet': {
                    'videoId': video_id,
                    'language': 'es',
                    'name': 'Español',
                    'isDraft': False
                }
            },
            media_body=media
        ).execute()
        
        print(f"   ✅ Subtítulos subidos: {result['id']}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 80)
print("COMPLETADO")
