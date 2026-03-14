#!/usr/bin/env python3
"""Quick thumbnails for 2 videos"""
import sys
sys.path.insert(0, "/home/enderj/.openclaw/workspace/bittrader/agents")

from thumbnail_agent_huggingface import generate_thumbnail

videos_data = [{"script_id": "short_1773470150_254", "title": "PI se hunde -29% en 24h (24 caracteres - OK)", "description": "#crypto #trading #short_1773470150_254", "type": "short", "tags": ["crypto", "trading", "shorts"], "output_file": "/home/enderj/.openclaw/workspace/bittrader/agents/output/2026-03-14/short_1773470150_254/short_1773470150_254.mp4", "duration": 12.7, "size_mb": 0.3, "status": "ready", "thumbnail": "/home/enderj/.openclaw/workspace/bittrader/agents/output/2026-03-14/short_1773470150_254/short_1773470150_254_thumbnail.jpg", "scheduled_date": "2026-03-14T18:00:00+00:00"}, {"script_id": "short_1773470200_200", "title": "TRUMP Coin: \u00bfEstafa o MILLONES?", "description": "#crypto #trading #short_1773470200_200", "type": "short", "tags": ["crypto", "trading", "shorts"], "output_file": "/home/enderj/.openclaw/workspace/bittrader/agents/output/2026-03-14/short_1773470200_200/short_1773470200_200.mp4", "duration": 25.8, "size_mb": 1.0, "status": "ready", "thumbnail": "/home/enderj/.openclaw/workspace/bittrader/agents/output/2026-03-14/short_1773470200_200/short_1773470200_200_thumbnail.jpg", "scheduled_date": "2026-03-15T18:00:00+00:00"}]

for video in videos_data:
    output = video["output_file"]
    thumb_path = video["thumbnail"]
    title = video["title"]
    
    print(f"Generating thumbnail for: {title}")
    print(f"  Video: {output}")
    print(f"  Thumb: {thumb_path}")
    
    # Generate thumbnail using Hugging Face
    try:
        generate_thumbnail(str(Path(output)), str(Path(thumb_path)), title)
        print(f"  ✅ Success")
    except Exception as e:
        print(f"  ❌ Error: {e}")
