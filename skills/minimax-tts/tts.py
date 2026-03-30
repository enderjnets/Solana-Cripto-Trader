#!/usr/bin/env python3
"""MiniMax TTS - Text to Speech using MiniMax Speech 2.0 API"""

import requests
import argparse
import sys
import json
import re
from pathlib import Path

API_KEY = "sk-api-dPHO8UbieX0zTe92NRVWSQyN6FfCJQnY0qpbu6RcO12MPhGixFPd--c5pQrkSFNfSft5d6hpvz4w59SZR9GeXyfPN_7pHQPfdqu4_9vFhalNnz0at8r3UV4"
ENDPOINT = "https://api.minimax.io/v1/t2a_v2"

def generate_speech(text: str, output_path: str = "/tmp/minimax_tts.mp3", 
                    voice: str = "male_1", model: str = "speech-02-hd",
                    speed: float = 1.0) -> str:
    """
    Generate speech using MiniMax TTS API.
    
    Args:
        text: Text to synthesize
        output_path: Where to save the MP3 file
        voice: Voice ID (male_1, male_2, female_1, female_2)
        model: Model (speech-02-hd, speech-02-turbo)
        speed: Speech speed (0.5 to 2.0)
    
    Returns:
        Path to the generated audio file
    """
    payload = {
        "model": model,
        "text": text,
        "stream": True,
        "voice_setting": {
            "voice_id": voice,
            "speed": speed,
            "volume": 1.0,
            "pitch": 0
        }
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"Generating speech: {text[:50]}...", file=sys.stderr)
    print(f"Voice: {voice}, Model: {model}", file=sys.stderr)
    
    response = requests.post(ENDPOINT, json=payload, headers=headers, timeout=30, stream=True)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}", file=sys.stderr)
        return None
    
    # Collect all audio chunks from streaming response
    audio_chunks = []
    for line in response.iter_lines():
        if line and line.startswith(b'data: '):
            try:
                chunk = json.loads(line[6:])
                audio_hex = chunk['data']['audio']
                audio_bytes = bytes.fromhex(audio_hex)
                audio_chunks.append(audio_bytes)
            except (json.JSONDecodeError, KeyError) as e:
                continue
    
    if not audio_chunks:
        print("Error: No audio data received", file=sys.stderr)
        return None
    
    # Combine all chunks and save
    combined = b''.join(audio_chunks)
    Path(output_path).write_bytes(combined)
    print(f"Saved {len(combined)} bytes to {output_path}", file=sys.stderr)
    return output_path

def main():
    parser = argparse.ArgumentParser(description="MiniMax TTS")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("output", nargs="?", default="/tmp/minimax_tts.mp3", help="Output file path")
    parser.add_argument("--voice", "-v", default="male_1", 
                       choices=["male_1", "male_2", "female_1", "female_2"],
                       help="Voice ID")
    parser.add_argument("--model", "-m", default="speech-02-hd",
                       choices=["speech-02-hd", "speech-02-turbo"],
                       help="Model")
    parser.add_argument("--speed", "-s", type=float, default=1.0,
                       help="Speech speed (0.5-2.0)")
    
    args = parser.parse_args()
    result = generate_speech(args.text, args.output, args.voice, args.model, args.speed)
    
    if result:
        print(result)
        return 0
    return 1

if __name__ == "__main__":
    sys.exit(main() or 0)
