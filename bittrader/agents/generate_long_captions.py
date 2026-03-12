#!/usr/bin/env python3
"""
Generate SRT captions for long videos using script text.
"""
import json
import re
from pathlib import Path

def extract_script_from_tags(tags):
    """Extract script text from malformed tags in production_latest.json"""
    script_parts = []
    in_script = False
    
    for tag in tags:
        if "GUIÓN COMPLETO:" in tag:
            in_script = True
            continue
        if in_script and tag.startswith("VIDEO_PROMPT_"):
            break
        if in_script:
            # Clean up the tag
            clean = tag.replace("\n", " ").strip()
            # Remove section markers
            clean = re.sub(r'(HOOK|PROBLEMA|EXPLICACION|EJEMPLOS|CTA)\s*\([^)]*\):?\s*', '', clean)
            if clean and not clean.startswith("#"):
                script_parts.append(clean)
    
    return " ".join(script_parts)

def text_to_srt(text, duration_sec, output_path):
    """Convert text to SRT format with estimated timing."""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Calculate time per sentence (distribute evenly)
    time_per_sentence = duration_sec / len(sentences)
    
    srt_lines = []
    current_time = 0.0
    
    for i, sentence in enumerate(sentences, 1):
        if not sentence.strip():
            continue
            
        # Start and end times
        start_time = current_time
        end_time = current_time + time_per_sentence
        
        # Format timestamps
        start_h = int(start_time // 3600)
        start_m = int((start_time % 3600) // 60)
        start_s = int(start_time % 60)
        start_ms = int((start_time - int(start_time)) * 1000)
        
        end_h = int(end_time // 3600)
        end_m = int((end_time % 3600) // 60)
        end_s = int(end_time % 60)
        end_ms = int((end_time - int(end_time)) * 1000)
        
        start_str = f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms:03d}"
        end_str = f"{end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms:03d}"
        
        # Add to SRT
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_str} --> {end_str}")
        srt_lines.append(sentence.strip())
        srt_lines.append("")
        
        current_time = end_time
    
    # Write SRT file
    output_path.write_text("\n".join(srt_lines), encoding='utf-8')
    return output_path

# Main
production_file = Path("data/production_latest.json")
with open(production_file) as f:
    data = json.load(f)

long_videos = [v for v in data['videos'] if v['type'] == 'long']

print("=" * 80)
print("GENERANDO SUBTÍTULOS PARA VIDEOS LARGOS")
print("=" * 80)

for video in long_videos:
    title = video['title']
    script_id = video['script_id']
    duration = video['duration']
    tags = video.get('tags', [])
    
    print(f"\n📝 {title}")
    print(f"   Duración: {duration}s")
    
    # Extract script
    script_text = extract_script_from_tags(tags)
    print(f"   Guión extraído: {len(script_text)} caracteres")
    
    if not script_text:
        print("   ❌ No se encontró guión")
        continue
    
    # Generate SRT - use output_file path to find folder
    output_file = video.get('output_file', '')
    folder_name = Path(output_file).parent.name
    output_dir = Path(f"output/rhino_v1/{folder_name}")
    
    if not output_dir.exists():
        print(f"   ❌ Directorio no existe: {output_dir}")
        continue
    
    output_path = output_dir / "captions.srt"
    text_to_srt(script_text, duration, output_path)
    
    print(f"   ✅ Subtítulos generados: {output_path}")

print("\n" + "=" * 80)
print("COMPLETADO")
