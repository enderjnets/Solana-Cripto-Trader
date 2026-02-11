#!/usr/bin/env python3
"""
PersonaPlex Voice Integration for Eko Trading Bot
==============================================
Implements NVIDIA PersonaPlex for real-time voice AI.

Features:
- Full-duplex voice conversations (80ms latency)
- Personality control via voice + text prompts
- WebSocket communication for real-time audio
- Docker deployment ready

Based on NVIDIA PersonaPlex research.

Requirements:
- Docker with GPU support
- NVIDIA Container Toolkit
- Hugging Face account (for model access)
"""

import os
import sys
import json
import asyncio
import logging
import websockets
import base64
import struct
from datetime import datetime
from typing import Dict, Optional, Callable
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("personaplex")

# ============================================================================
# DOCKER CONFIGURATION
# ============================================================================
DOCKER_COMPOSE = """version: '3.8'

services:
  personaplex:
    image: nvcr.io/nvidia/personaplex:latest
    container_name: personaplex-eko
    ports:
      - "8998:8998"
    volumes:
      - ./persona:/app/persona
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - PERSONA_VOICE=/app/persona/voice.wav
      - PERSONA_TEXT=/app/persona/personality.txt
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8998/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

volumes:
  persona:
"""

DOCKERFILE = """FROM nvcr.io/nvidia/personaplex:latest

# Install dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create persona directory
RUN mkdir -p /app/persona

# Copy persona files
COPY ./persona/voice.wav /app/persona/voice.wav
COPY ./persona/personality.txt /app/persona/personality.txt

# Expose WebSocket port
EXPOSE 8998

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8998/health || exit 1

CMD ["python3", "-m", "personaplex", "--port", "8998"]
"""


# ============================================================================
# PERSONA FILES
# ============================================================================
PERSONALITY_TEXT = """# Eko Personality
You are Eko, an AI trading assistant specialized in Solana and cryptocurrency markets.

## Voice Characteristics
- Professional but friendly tone
- Clear and concise responses
- Enthusiastic about trading opportunities

## Trading Expertise
- Solana blockchain
- Jupiter DEX
- Trading strategies (RSI, SMA, MACD, Bollinger)
- Risk management

## Response Style
- Short for quick updates
- Detailed for explanations
- Always confirm before executing trades
"""

VOICE_NOTE = """# Voice Sample for PersonaPlex

To create a custom voice:
1. Record a 10-30 second sample in WAV format
2. 24kHz sampling rate recommended
3. Clear speech without background noise
4. Save as: persona/voice.wav

For default voice, leave this file empty.
"""


# ============================================================================
# PERSONAPLEX CLIENT
# ============================================================================
class PersonaPlexClient:
    """
    Client for NVIDIA PersonaPlex WebSocket API.
    
    Features:
    - Real-time voice conversations
    - Personality injection via prompts
    - Audio streaming support
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8998,
        persona_text: str = None,
        persona_voice: str = None
    ):
        self.host = host
        self.port = port
        self.persona_text = persona_text or PERSONALITY_TEXT
        self.persona_voice = persona_voice
        self.ws = None
        self.session_id = None
        self.connected = False
        
    async def connect(self) -> bool:
        """Connect to PersonaPlex WebSocket server"""
        try:
            url = f"ws://{self.host}:{self.port}/ws"
            self.ws = await websockets.connect(url)
            self.connected = True
            
            # Initialize session with personality
            await self._init_session()
            
            logger.info(f"✅ Connected to PersonaPlex at {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to PersonaPlex: {e}")
            return False
    
    async def _init_session(self):
        """Initialize session with personality"""
        if self.persona_text:
            message = {
                "type": "system",
                "content": {
                    "role": "system",
                    "content": self.persona_text
                }
            }
            await self.ws.send(json.dumps(message))
            
            # Set voice if provided
            if self.persona_voice:
                voice_msg = {
                    "type": "voice",
                    "content": self.persona_voice
                }
                await self.ws.send(json.dumps(voice_msg))
    
    async def send_text(self, text: str) -> str:
        """Send text and get audio response"""
        if not self.connected:
            raise RuntimeError("Not connected to PersonaPlex")
        
        # Send user message
        message = {
            "type": "user",
            "content": {
                "role": "user",
                "content": text
            }
        }
        await self.ws.send(json.dumps(message))
        
        # Wait for response
        response = await self.ws.recv()
        data = json.loads(response)
        
        if data.get("type") == "audio":
            # Return base64 decoded audio
            return data.get("content", "")
        
        return ""
    
    async def send_audio(self, audio_data: bytes) -> bytes:
        """Send audio and get audio response (full-duplex)"""
        if not self.connected:
            raise RuntimeError("Not connected to PersonaPlex")
        
        # Encode audio as base64
        audio_b64 = base64.b64encode(audio_data).decode()
        
        message = {
            "type": "audio",
            "content": audio_b64
        }
        await self.ws.send(json.dumps(message))
        
        # Wait for response
        response = await self.ws.recv()
        data = json.loads(response)
        
        if data.get("type") == "audio":
            return base64.b64decode(data.get("content", ""))
        
        return b""
    
    async def stream_audio(
        self,
        audio_chunk: bytes,
        callback: Callable[[bytes], None]
    ):
        """Stream audio chunks and receive responses"""
        if not self.connected:
            raise RuntimeError("Not connected to PersonaPlex")
        
        # Send audio chunk
        audio_b64 = base64.b64encode(audio_chunk).decode()
        message = {
            "type": "audio_chunk",
            "content": audio_b64
        }
        await self.ws.send(json.dumps(message))
        
        # Receive response
        response = await self.ws.recv()
        data = json.loads(response)
        
        if data.get("type") == "audio_chunk":
            audio_response = base64.b64decode(data.get("content", b""))
            callback(audio_response)
    
    async def close(self):
        """Close WebSocket connection"""
        if self.ws:
            await self.ws.close()
            self.connected = False
            logger.info("PersonaPlex connection closed")


# ============================================================================
# SPEECH-TO-TEXT (Whisper fallback)
# ============================================================================
class SpeechToText:
    """
    Speech-to-text using OpenAI Whisper.
    
    Can use:
    - OpenAI Whisper API (cloud)
    - Local Whisper (openai-whisper package)
    """
    
    def __init__(self, use_local: bool = True):
        self.use_local = use_local
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
    
    async def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text"""
        if self.use_local:
            return await self._transcribe_local(audio_path)
        else:
            return await self._transcribe_api(audio_path)
    
    async def _transcribe_local(self, audio_path: str) -> str:
        """Local transcription using whisper package"""
        try:
            import whisper
            
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            
            return result.get("text", "").strip()
            
        except Exception as e:
            logger.error(f"Local transcription failed: {e}")
            return ""
    
    async def _transcribe_api(self, audio_path: str) -> str:
        """OpenAI Whisper API transcription"""
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            
            with open(audio_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"API transcription failed: {e}")
            return ""


# ============================================================================
# TEXT-TO-SPEECH (ElevenLabs fallback)
# ============================================================================
class TextToSpeech:
    """
    Text-to-speech using ElevenLabs or local TTS.
    """
    
    def __init__(self, use_local: bool = True):
        self.use_local = use_local
        self.api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    
    async def synthesize(self, text: str, voice: str = "nova") -> bytes:
        """Convert text to speech audio"""
        if self.use_local:
            return await self._synthesize_local(text)
        else:
            return await self._synthesize_api(text, voice)
    
    async def _synthesize_local(self, text: str) -> bytes:
        """Local TTS using pyttsx3 or similar"""
        try:
            import pyttsx3
            
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            
            # Save to file then read
            temp_file = "/tmp/tts_output.wav"
            engine.save_to_file(text, temp_file)
            engine.runAndWait()
            
            with open(temp_file, "rb") as f:
                return f.read()
                
        except Exception as e:
            logger.error(f"Local TTS failed: {e}")
            return b""
    
    async def _synthesize_api(self, text: str, voice: str = "nova") -> bytes:
        """ElevenLabs API TTS"""
        try:
            import requests
            
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
            
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                return response.content
            else:
                logger.error(f"TTS API error: {response.text}")
                return b""
                
        except Exception as e:
            logger.error(f"API TTS failed: {e}")
            return b""


# ============================================================================
# VOICE TRADING ASSISTANT
# ============================================================================
class VoiceTradingAssistant:
    """
    Complete voice assistant for trading.
    
    Combines:
    - STT (Whisper)
    - PersonaPlex (conversation)
    - TTS (ElevenLabs/local)
    
    Usage:
        assistant = VoiceTradingAssistant()
        await assistant.start()
        
        # Send voice command
        response = await assistant.process_voice_command("Compra 2 SOL")
        
        # Or send audio file
        response = await assistant.process_audio_file("command.wav")
    """
    
    def __init__(self):
        self.personaplex = None
        self.stt = SpeechToText(use_local=True)
        self.tts = TextToSpeech(use_local=True)
        self.running = False
        
    async def start(
        self,
        host: str = "localhost",
        port: int = 8998
    ) -> bool:
        """Start the voice assistant"""
        # Connect to PersonaPlex
        self.personaplex = PersonaPlexClient(host, port)
        connected = await self.personaplex.connect()
        
        if connected:
            self.running = True
            logger.info("✅ Voice Trading Assistant started")
        
        return connected
    
    async def stop(self):
        """Stop the voice assistant"""
        if self.personaplex:
            await self.personaplex.close()
        self.running = False
        logger.info("Voice Trading Assistant stopped")
    
    async def process_text_command(self, text: str) -> Dict:
        """Process text command and get voice response"""
        if not self.running:
            return {"error": "Assistant not running"}
        
        try:
            # Get response from PersonaPlex
            audio_b64 = await self.personaplex.send_text(text)
            
            if audio_b64:
                audio_data = base64.b64decode(audio_b64)
                
                return {
                    "status": "success",
                    "text_response": text,
                    "audio": audio_b64,
                    "audio_size": len(audio_data)
                }
            else:
                return {"status": "error", "message": "No response"}
                
        except Exception as e:
            return {"error": str(e)}
    
    async def process_audio_command(self, audio_path: str) -> Dict:
        """Process audio command file"""
        if not self.running:
            return {"error": "Assistant not running"}
        
        try:
            # Transcribe audio to text
            text = await self.stt.transcribe(audio_path)
            
            if not text:
                return {"error": "Transcription failed"}
            
            # Get voice response
            response = await self.process_text_command(text)
            response["transcribed"] = text
            
            return response
            
        except Exception as e:
            return {"error": str(e)}
    
    async def send_voice_feedback(self, feedback: str):
        """Send quick voice feedback (backchannel)"""
        if not self.running:
            return
        
        # Quick acknowledgments
        acknowledgments = {
            "ack": "Entendido",
            "thinking": "Hmm, déjame ver...",
            "confirm": "Confirmado",
            "error": "Hubo un error"
        }
        
        text = acknowledgments.get(feedback, feedback)
        await self.process_text_command(text)


# ============================================================================
# TRADING COMMAND PARSER
# ============================================================================
TRADING_COMMANDS = {
    "compra": "BUY",
    "comprare": "BUY",
    "buy": "BUY",
    "vende": "SELL",
    "vender": "SELL",
    "sell": "SELL",
    "balance": "BALANCE",
    "saldo": "BALANCE",
    "status": "STATUS",
    "estado": "STATUS",
    "precio": "PRICE",
    "price": "PRICE",
    "stop": "STOP",
    "para": "STOP",
    "ayuda": "HELP",
    "help": "HELP"
}


def parse_trading_command(text: str) -> Dict:
    """
    Parse natural language trading command.
    
    Examples:
        "Compra 2 SOL" -> {"action": "BUY", "amount": 2, "token": "SOL"}
        "Cuál es mi balance" -> {"action": "BALANCE"}
        "Dame el precio de SOL" -> {"action": "PRICE", "token": "SOL"}
    """
    import re
    
    text_lower = text.lower()
    result = {
        "original": text,
        "action": None,
        "amount": None,
        "token": None,
        "confidence": 0.0
    }
    
    # Detect action
    for keyword, action in TRADING_COMMANDS.items():
        if keyword in text_lower:
            result["action"] = action
            break
    
    if not result["action"]:
        result["action"] = "UNKNOWN"
        return result
    
    # Detect amount
    amount_match = re.search(r'(\d+(?:\.\d+)?)', text)
    if amount_match:
        result["amount"] = float(amount_match.group(1))
    
    # Detect token
    tokens = ["SOL", "USDC", "USDT", "BTC", "ETH", "JUP", "BONK"]
    for token in tokens:
        if token in text.upper():
            result["token"] = token
            break
    
    # Calculate confidence
    confidence = 0.5  # Base
    if result["amount"]:
        confidence += 0.2
    if result["token"]:
        confidence += 0.2
    
    result["confidence"] = min(1.0, confidence)
    
    return result


# ============================================================================
# SETUP SCRIPTS
# ============================================================================
def create_docker_files():
    """Create Docker configuration files"""
    files = {
        "docker-compose.yml": DOCKER_COMPOSE,
        "Dockerfile": DOCKERFILE,
        "persona/personality.txt": PERSONALITY_TEXT,
        "persona/README.txt": VOICE_NOTE
    }
    
    for filename, content in files.items():
        path = Path(__file__).parent / filename
        path.parent.mkdir(exist_ok=True)
        
        with open(path, "w") as f:
            f.write(content)
        
        logger.info(f"Created: {filename}")
    
    return True


# ============================================================================
# MAIN DEMO
# ============================================================================
async def demo():
    """Demo voice trading assistant"""
    
    print("="*70)
    print("🎤 PERSONAPLEX VOICE TRADING DEMO")
    print("="*70)
    
    # Create Docker files
    print("\n📁 Creating Docker configuration...")
    create_docker_files()
    
    # Demo trading command parser
    print("\n🗣️ Trading Command Parser Demo:")
    
    commands = [
        "Compra 2 SOL",
        "Vende 0.5 SOL",
        "Cuál es mi balance",
        "Dame el precio de SOL",
        "Para todo"
    ]
    
    for cmd in commands:
        parsed = parse_trading_command(cmd)
        print(f"\n   Input: '{cmd}'")
        print(f"   → Action: {parsed['action']}")
        print(f"   → Amount: {parsed['amount']}")
        print(f"   → Token: {parsed['token']}")
        print(f"   → Confidence: {parsed['confidence']:.0%}")
    
    # Demo voice assistant
    print("\n" + "="*70)
    print("📝 Voice Assistant (PersonaPlex required)")
    print("="*70)
    print("""
To use PersonaPlex voice:

1. Create Docker configuration:
   python3 personaplex.py --setup

2. Start PersonaPlex server:
   docker-compose up -d

3. Configure Hugging Face token:
   export HF_TOKEN=your_token

4. Start voice assistant:
   python3 personaplex.py --start

Commands:
   --setup     Create Docker files
   --start     Start voice assistant
   --demo      Show command parser demo
   --help      Show this help
    """)
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PersonaPlex Voice Integration")
    parser.add_argument("--setup", action="store_true", help="Create Docker configuration")
    parser.add_argument("--start", action="store_true", help="Start voice assistant")
    parser.add_argument("--demo", action="store_true", help="Show demo")
    
    args = parser.parse_args()
    
    if args.setup or not any([args.setup, args.start, args.demo]):
        create_docker_files()
    
    if args.demo:
        asyncio.run(demo())
    
    if not any([args.setup, args.start, args.demo]):
        asyncio.run(demo())
