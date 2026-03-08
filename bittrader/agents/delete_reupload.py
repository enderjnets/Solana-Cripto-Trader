#!/usr/bin/env python3
"""Delete 8 old YouTube videos and re-upload v2 versions."""
import json
import time
from pathlib import Path
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload

WORKSPACE = Path("/home/enderj/.openclaw/workspace")
CREDS_FILE = WORKSPACE / "memory/youtube_credentials.json"
DATA_DIR = WORKSPACE / "bittrader/agents/data"

# YouTube IDs to delete (uploaded earlier today with bad audio)
OLD_VIDEO_IDS = [
    "ggPhndp_AJQ",  # AKT explota (SHORT)
    "R7KfhqXyBrs",  # PENGU sube (SHORT)
    "jDqKyWCwv7A",  # 90% traders (SHORT)
    "sJFogsruO4k",  # PI coin +13% (SHORT)
    "Ov87mjomP0I",  # Claude trades (LONG)
    "EmNLR9dF7Ak",  # ZEC cae 7% (SHORT)
    "yqWtj6nuzsE",  # $0 a fondeada (LONG)
    "zOiK12HoZss",  # Bot noticias (LONG)
]

def get_youtube_client():
    creds_data = json.loads(CREDS_FILE.read_text())
    creds = Credentials(
        token=creds_data.get("access_token") or creds_data.get("token"),
        refresh_token=creds_data["refresh_token"],
        token_uri=creds_data["token_uri"],
        client_id=creds_data["client_id"],
        client_secret=creds_data["client_secret"],
        scopes=creds_data.get("scopes") or creds_data.get("scope", "").split(),
    )
    return build("youtube", "v3", credentials=creds)

def delete_videos(youtube, video_ids):
    print("🗑️  Deleting old videos...")
    for vid_id in video_ids:
        try:
            youtube.videos().delete(id=vid_id).execute()
            print(f"  ✅ Deleted: {vid_id}")
        except Exception as e:
            print(f"  ⚠️ Could not delete {vid_id}: {e}")
        time.sleep(1)

def upload_video(youtube, video_path: Path, title: str, description: str,
                 tags: list, category_id: str = "22", privacy: str = "private") -> str:
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category_id,
            "defaultLanguage": "es",
            "defaultAudioLanguage": "es",
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        }
    }

    media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True, chunksize=5*1024*1024)
    req = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = req.next_chunk()
        if status:
            pct = int(status.resumable_progress / status.total_size * 100)
            print(f"    Upload: {pct}%", end="\r")

    return response["id"]

def set_thumbnail(youtube, video_id: str, thumb_path: Path):
    if not thumb_path.exists():
        return
    try:
        media = MediaFileUpload(str(thumb_path), mimetype="image/jpeg")
        youtube.thumbnails().set(videoId=video_id, media_body=media).execute()
        print(f"    🖼️ Thumbnail uploaded")
    except Exception as e:
        print(f"    ⚠️ Thumbnail failed: {e}")

def main():
    youtube = get_youtube_client()
    guiones = json.loads((DATA_DIR / "guiones_latest.json").read_text())
    scripts = guiones["scripts"]

    # Step 1: Delete old videos
    delete_videos(youtube, OLD_VIDEO_IDS)
    print()

    # Step 2: Upload new v2 videos
    print("📤 Uploading v2 videos...")
    new_ids = {}

    for i, script in enumerate(scripts, 1):
        title = script["title"]
        vtype = script.get("type", "short")
        out_file = Path(script.get("output_file", ""))
        script_dir = out_file.parent
        v2_path = script_dir / out_file.name.replace("_final.mp4", "_v2_final.mp4")
        thumb_path = script_dir / "thumbnail.jpg"

        if not v2_path.exists():
            print(f"  [{i}] ❌ v2 not found: {v2_path.name}")
            continue

        tags = script.get("tags", []) + ["BitTrader", "trading", "crypto", "IA"]
        desc = f"{title}\n\n#BitTrader #Trading #Crypto #IA #Bot"
        if vtype == "short":
            desc += "\n\n#Shorts"

        print(f"  [{i}/{len(scripts)}] {title[:45]}")
        try:
            vid_id = upload_video(youtube, v2_path, title, desc, tags, privacy="private")
            print(f"    ✅ Uploaded: https://youtube.com/watch?v={vid_id}")
            set_thumbnail(youtube, vid_id, thumb_path)
            new_ids[title] = vid_id
            time.sleep(2)
        except Exception as e:
            print(f"    ❌ Upload failed: {e}")

    # Save new IDs
    (DATA_DIR / "youtube_new_ids.json").write_text(json.dumps(new_ids, indent=2))
    print(f"\n✅ Done. {len(new_ids)}/8 uploaded")
    print(f"📁 New IDs saved to youtube_new_ids.json")
    print("\nNew video IDs:")
    for title, vid_id in new_ids.items():
        print(f"  {vid_id} — {title[:45]}")

if __name__ == "__main__":
    main()
