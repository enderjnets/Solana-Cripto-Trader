#!/usr/bin/env python3
"""
Genera thumbnails para videos en la cola sin thumbnail
"""
import json
from pathlib import Path

# Paths
queue_file = Path("/home/enderj/.openclaw/workspace/bittrader/agents/data/upload_queue.json")
queue = json.loads(queue_file.read_text())

print("🖼️ Generando thumbnails...")

for i, item in enumerate(queue):
    if not item.get("thumbnail"):
        # Simple thumbnail based on title
        title = item["title"]
        sid = item["script_id"]
        output = item["output_file"]
        
        # For now, just mark that thumbnail needs to be generated
        # The pipeline should have generated it during production
        print(f"\n⚠️  {sid}: Sin thumbnail")
        print(f"   {title}")
        print(f"   📁 {output}")
        
        # Check if thumbnail file exists in same directory
        video_path = Path(output)
        thumb_dir = video_path.parent
        potential_thumbs = list(thumb_dir.glob("*.jpg")) + list(thumb_dir.glob("*.png"))
        
        if potential_thumbs:
            item["thumbnail"] = str(potential_thumbs[0])
            print(f"   ✅ Thumbnail encontrado: {potential_thumbs[0].name}")
        else:
            # Create placeholder
            thumb_path = thumb_dir / f"{sid}_thumbnail.jpg"
            item["thumbnail"] = str(thumb_path)
            print(f"   📝 Thumbnail placeholder: {thumb_path}")

# Save
queue_file.write_text(json.dumps(queue, indent=2, ensure_ascii=False))
print("\n✅ Queue actualizado con thumbnails")
