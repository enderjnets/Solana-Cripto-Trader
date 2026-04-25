# PersonaPlex Voice Integration Setup

## 🎤 NVIDIA PersonaPlex for Eko Trading Bot

This guide covers setting up NVIDIA PersonaPlex for real-time voice AI.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | RTX 3060 (12GB) | RTX 4090 (24GB) |
| VRAM | 16GB | 24GB |
| RAM | 16GB | 32GB |
| Docker | 20.x | Latest |
| NVIDIA Container Toolkit | Latest | Latest |
| HF Token | Free account | Pro (for faster downloads) |

---

## Setup Steps

### 1. Install Docker & NVIDIA Toolkit

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. Get Hugging Face Token

1. Create account: https://huggingface.co
2. Go to Settings → Access Tokens
3. Create new token (read permission)
4. Export token:
   ```bash
   export HF_TOKEN=your_token_here
   ```

### 3. Configure Persona Files

Edit `persona/personality.txt`:
```markdown
# Eko Personality
You are Eko, an AI trading assistant specialized in Solana...

## Voice Characteristics
- Professional but friendly
- Clear and concise
- Enthusiastic about trading
```

Optional: Add voice sample at `persona/voice.wav` (24kHz WAV file)

### 4. Start PersonaPlex

```bash
# Start the server
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f personaplex
```

### 5. Test Connection

```bash
# Check if server is running
curl http://localhost:8998/health

# Should return: {"status": "ok"}
```

---

## Docker Configuration

### docker-compose.yml

```yaml
version: '3.8'

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
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Usage

### Start Voice Assistant

```python
from personaplex import VoiceTradingAssistant

assistant = VoiceTradingAssistant()
await assistant.start(host="localhost", port=8998)

# Process voice command
result = await assistant.process_text_command("Compra 2 SOL")
print(result)
```

### Command Parser

```python
from personaplex import parse_trading_command

result = parse_trading_command("Vende 0.5 SOL")
# {
#     "action": "SELL",
#     "amount": 0.5,
#     "token": "SOL",
#     "confidence": 0.9
# }
```

### Supported Commands

| Command | Action |
|---------|---------|
| "Compra X SOL" | BUY |
| "Vende X SOL" | SELL |
| "Mi balance" | BALANCE |
| "Precio de SOL" | PRICE |
| "Estado del sistema" | STATUS |
| "Para todo" | STOP |
| "Ayuda" | HELP |

---

## Without GPU (CPU Offload)

If you don't have GPU:

```yaml
# docker-compose.yml
services:
  personaplex:
    image: nvcr.io/nvidia/personaplex:latest
    command: python3 -m personaplex --cpu-offload
    deploy:
      resources:
        reservations: {}
```

⚠️ **Warning:** CPU mode significantly increases latency (2-3 seconds instead of 80ms).

---

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA drivers
nvidia-smi

# Verify CUDA
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

### Container Won't Start

```bash
# Check logs
docker-compose logs personaplex

# Common issues:
# - HF_TOKEN not exported
# - GPU memory insufficient
# - Port 8998 already in use
```

### High Latency

1. Check GPU utilization: `nvidia-smi`
2. Reduce batch size if VRAM < 16GB
3. Consider CPU offload for testing

---

## API Reference

### PersonaPlexClient

```python
client = PersonaPlexClient(host="localhost", port=8998)
await client.connect()

# Send text, get audio response
audio = await client.send_text("Hola, soy Eko")

# Stream audio
async def handle_response(audio_chunk):
    play(audio_chunk)
await client.stream_audio(input_audio, handle_response)

await client.close()
```

### VoiceTradingAssistant

```python
assistant = VoiceTradingAssistant()
await assistant.start()

# Process voice command
result = await assistant.process_voice_command("Compra SOL")

# Or from audio file
result = await assistant.process_audio_command("command.wav")

await assistant.stop()
```

---

## Files

```
personaplex/
├── personaplex.py          # Main integration
├── docker-compose.yml     # Docker configuration
├── Dockerfile            # Container setup
└── persona/
    ├── personality.txt   # Voice personality
    └── voice.wav        # Optional voice sample
```

---

## References

- **PersonaPlex:** https://huggingface.co/nvidia/personaplex-7b-v1
- **Docker Installation:** https://docs.docker.com/engine/install/
- **NVIDIA Container Toolkit:** https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

---

*Eko - Autonomous AI Trading Agent with Voice*
