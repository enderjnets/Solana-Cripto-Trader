#!/usr/bin/env python3
"""MiniMax TTS - Text to Speech using MiniMax Speech API

API Docs: https://platform.minimax.io/docs/api-reference/speech-t2a-intro

Models ( newest = best ):
  - speech-2.8-hd  : Latest HD model, best quality
  - speech-2.8-turbo: Latest Turbo model, fast
  - speech-2.6-hd  : HD model with prosody
  - speech-2.6-turbo: Turbo model, 40 languages
  - speech-02-hd   : Superior rhythm & stability
  - speech-02-turbo: Enhanced multilingual

Voices available (confirmed working):
  - male_1   : Masculino claro  ⭐ DEFAULT
  - male_2   : Masculino alternativo
  - female_1 : Femenino claro
  - female_2 : Femenino alternativo

Output formats: mp3 (default), pcm, flac, wav (non-streaming only)

API: https://api.minimax.io/v1/t2a_v2
"""

import requests
import argparse
import sys
import json
import re
from pathlib import Path

API_KEY = "sk-api-dPHO8UbieX0zTe92NRVWSQyN6FfCJQnY0qpbu6RcO12MPhGixFPd--c5pQrkSFNfSft5d6hpvz4w59SZR9GeXyfPN_7pHQPfdqu4_9vFhalNnz0at8r3UV4"
ENDPOINT = "https://api.minimax.io/v1/t2a_v2"

DEFAULT_MODEL = "speech-2.8-hd"  # Latest and best quality
DEFAULT_VOICE = "male_1"


def generate_speech(
    text: str,
    output_path: str = "/tmp/minimax_tts.mp3",
    voice: str = DEFAULT_VOICE,
    model: str = DEFAULT_MODEL,
    speed: float = 1.0,
    volume: float = 1.0,
    pitch: float = 0,
    audio_format: str = "mp3",
    stream: bool = True
) -> str | None:
    """
    Generate speech using MiniMax TTS API.

    Args:
        text: Text to synthesize (up to 10,000 chars)
        output_path: Where to save the audio file
        voice: Voice ID (male_1, male_2, female_1, female_2)
        model: Model (speech-2.8-hd, speech-2.8-turbo, speech-2.6-hd, speech-2.6-turbo, speech-02-hd, speech-02-turbo)
        speed: Speech speed (0.5 to 2.0, default 1.0)
        volume: Volume (0.0 to 2.0, default 1.0)
        pitch: Pitch adjustment (-500 to 500, default 0)
        audio_format: Output format (mp3, pcm, flac, wav)
        stream: Use streaming mode (recommended)

    Returns:
        Path to the generated audio file, or None on error
    """
    payload = {
        "model": model,
        "text": text,
        "stream": stream,
        "voice_setting": {
            "voice_id": voice,
            "speed": speed,
            "volume": volume,
            "pitch": pitch
        }
    }

    # Non-streaming needs format set explicitly
    if not stream:
        payload["audio_setting"] = {"audio_format": audio_format}

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    print(f"🎙️  Generating: {text[:60]}{'...' if len(text) > 60 else ''}", file=sys.stderr)
    print(f"   Model: {model} | Voice: {voice} | Speed: {speed}x | Format: {audio_format}", file=sys.stderr)

    try:
        response = requests.post(ENDPOINT, json=payload, headers=headers, timeout=60, stream=stream)

        if response.status_code != 200:
            # Try to parse error
            try:
                err = response.json()
                status_msg = err.get("base_resp", {}).get("status_msg", response.text)
            except:
                status_msg = response.text[:200]
            print(f"❌ Error {response.status_code}: {status_msg}", file=sys.stderr)
            return None

        if stream:
            # Collect all SSE data chunks
            audio_chunks = []
            for line in response.iter_lines():
                if line and line.startswith(b"data: "):
                    try:
                        chunk = json.loads(line[6:])
                        audio_hex = chunk["data"]["audio"]
                        audio_bytes = bytes.fromhex(audio_hex)
                        audio_chunks.append(audio_bytes)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue

            if not audio_chunks:
                print("❌ No audio data received", file=sys.stderr)
                return None

            combined = b"".join(audio_chunks)
        else:
            combined = response.content

        Path(output_path).write_bytes(combined)
        size_kb = len(combined) / 1024
        print(f"✅ Saved {size_kb:.1f}KB → {output_path}", file=sys.stderr)
        return output_path

    except Exception as e:
        print(f"❌ Exception: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="MiniMax TTS - Text to Speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Hola Ender" /tmp/audio.mp3
  %(prog)s --voice female_1 "Buenos días" /tmp/audio.mp3
  %(prog)s --model speech-02-turbo --speed 0.9 "Texto rápido" /tmp/audio.mp3
  %(prog)s --format wav "Hola" /tmp/audio.wav

Models:
  speech-2.8-hd   : Latest HD (default, best quality)
  speech-2.8-turbo: Latest Turbo (fast)
  speech-2.6-hd   : HD with excellent prosody
  speech-2.6-turbo: Turbo, 40 languages
  speech-02-hd    : Superior rhythm & stability
  speech-02-turbo : Enhanced multilingual

Voices:
  male_1, male_2, female_1, female_2
        """
    )
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("output", nargs="?", default="/tmp/minimax_tts.mp3", help="Output file path")
    parser.add_argument("--voice", "-v", default=DEFAULT_VOICE,
                       choices=["male_1", "male_2", "female_1", "female_2"],
                       help=f"Voice ID (default: {DEFAULT_VOICE})")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                       choices=["speech-2.8-hd", "speech-2.8-turbo", "speech-2.6-hd",
                               "speech-2.6-turbo", "speech-02-hd", "speech-02-turbo"],
                       help=f"Model (default: {DEFAULT_MODEL})")
    parser.add_argument("--speed", "-s", type=float, default=1.0,
                       help="Speech speed 0.5-2.0 (default: 1.0)")
    parser.add_argument("--volume", type=float, default=1.0,
                       help="Volume 0.0-2.0 (default: 1.0)")
    parser.add_argument("--pitch", type=int, default=0,
                       help="Pitch -500 to 500 (default: 0)")
    parser.add_argument("--format", "-f", default="mp3",
                       choices=["mp3", "pcm", "flac", "wav"],
                       help="Audio format (default: mp3)")

    args = parser.parse_args()

    # Use non-streaming for wav format
    stream = args.format != "wav"

    result = generate_speech(
        text=args.text,
        output_path=args.output,
        voice=args.voice,
        model=args.model,
        speed=args.speed,
        volume=args.volume,
        pitch=args.pitch,
        audio_format=args.format,
        stream=stream
    )

    if result:
        print(result)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
