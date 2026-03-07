# BitTrader Agents 🤖📈

Automated YouTube content production system for trading channels powered by AI agents.

## Architecture

```
┌─────────────────────────────────────┐
│        ORCHESTRATOR (main)          │
│   Coordinates all agents            │
└──────────┬──────────────────────────┘
           │
    ┌──────┼──────────┬───────────┐
    ▼      ▼          ▼           ▼
┌───────┐┌────────┐┌──────────┐┌──────────┐
│ SCOUT ││CREATOR ││PRODUCER  ││PUBLISHER │
│       ││        ││          ││          │
│Market ││Scripts ││Videos    ││YouTube   │
│Research││& Plans ││& Audio   ││Upload    │
└───────┘└────────┘└──────────┘└──────────┘
```

## Agents

| Agent | Role | Frequency |
|-------|------|-----------|
| **Scout** | YouTube stats + crypto trending + news + velocity detection | Every 12h |
| **Creator** | Generate scripts with LLM based on data patterns | Wed + Sat + urgent |
| **Producer** | TTS + video clips + karaoke subs + assembly | After Creator |
| **Publisher** | Upload to YouTube with smart scheduling | After Producer |

## Stack

- **LLM**: MiniMax Text-01 (script generation)
- **TTS**: MiniMax speech-02-hd (presenter_male voice, Latin Spanish)
- **Video**: MiniMax Hailuo 2.3 (6s AI-generated clips)
- **Subtitles**: Whisper (word-level timestamps) → ASS karaoke format
- **Assembly**: ffmpeg (loop clips + audio + subs + logo)
- **Upload**: YouTube Data API v3
- **Market Data**: CoinGecko API (free tier)

## Usage

```bash
# Full pipeline: Scout → Creator → Producer → Publisher
python3 orchestrator.py --full

# Individual agents
python3 orchestrator.py --scout-only
python3 orchestrator.py --scout-only --no-youtube  # saves API quota
python3 orchestrator.py --create-only
python3 orchestrator.py --produce-only
python3 orchestrator.py --publish-only

# Process upload queue (retry failed uploads)
python3 orchestrator.py --process-queue

# Dry run (plan without LLM calls)
python3 orchestrator.py --full --dry-run
```

## How It Works

1. **Scout** collects YouTube analytics + crypto market data
2. **Scout** classifies videos as Winners/Average/Losers, detects trending topics
3. **Creator** reads Scout report, generates optimized scripts with MiniMax LLM
4. **Creator** avoids failed topics, duplicates winner patterns, includes wildcards
5. **Producer** generates voice (TTS), video clips, karaoke subtitles, assembles final video
6. **Producer** auto-shortens scripts if audio exceeds 58s, retries on failure
7. **Publisher** uploads to YouTube with data-driven scheduling, manages quota limits

## Setup

1. Copy `config.example.json` and fill in your API keys
2. Set MiniMax API key (for LLM, TTS, and video generation)
3. Configure YouTube OAuth2 credentials
4. Place your channel logo PNG
5. Run `python3 orchestrator.py --scout-only --no-youtube` to test

## Cost

| Component | Cost |
|-----------|------|
| Scout (YouTube API + CoinGecko) | Free |
| Creator (MiniMax LLM, 10 scripts) | ~$0.05 |
| Producer (10 clips + 10 voices) | ~$1.80 |
| Publisher (YouTube API) | Free |
| **Total per cycle (10 shorts)** | **~$1.85** |
| **Monthly (80 shorts)** | **~$15** |

## Topics Covered

- **Crypto**: Bitcoin, altcoins, DeFi, exchanges
- **Futures**: NAS100, S&P500, commodities
- **CFDs**: Contract for difference trading
- **Prop Firms**: FTMO, Topstep, funding challenges
- **Bots & Automation**: Trading systems, algorithmic trading
- **Education**: Technical analysis, risk management, psychology

## Video Specs

- **Shorts**: 9:16 vertical, <60s, karaoke subtitles, logo top-right
- **Long-form**: 16:9 horizontal, 3-8min, structured sections
- **Subtitles**: Yellow active word (karaoke \kf), white inactive, Arial 66pt
- **Logo**: Always top-right (overlay=W-w-30:30, scale=100:100)
- **Voice**: Latin American Spanish (neutral accent, NOT Spain Spanish)

## License

Private — © BitTrader 2026
