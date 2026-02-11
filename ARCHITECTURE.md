# Eko - Multi-Agent Trading System

## 🚀 Complete Implementation

This document summarizes all the research and implementations based on OpenClaw and NVIDIA PersonaPlex studies.

---

## 📚 Research Integrated

### 1. OpenClaw Multi-Agent Architecture
- **Source:** OpenClaw docs, GitHub, community research
- **Key Concepts:**
  - Mission Control orchestration
  - Brain and Muscles pattern
  - sessions_send / sessions_spawn APIs
  - AGENTS.md team structure

### 2. NVIDIA PersonaPlex Voice AI
- **Source:** Hugging Face, GitHub, research papers
- **Key Concepts:**
  - Full-duplex speech (80ms latency)
  - Personality control via voice + text prompts
  - Mimi codec (12.5Hz, 1.1kbps)
  - Docker deployment

### 3. MiniMax 2.1 Integration
- **Source:** MiniMax docs, API references
- **Key Specifications:**
  - 230B total parameters
  - 10B active (MoE)
  - 204,800 token context
  - Thinking mode for CoT

### 4. Agent Economy
- **Source:** Community research
- **Platforms:**
  - Moltbook (agent social network)
  - OpenWork (task marketplace)
  - Bankr (AI financial identity)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          EKO ECOSYSTEM                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  👤 USER                                                           │
│     │                                                              │
│     ├── 📱 Telegram (text)                                        │
│     ├── 🎤 Voice - PersonaPlex (future)                           │
│     └── 💰 Budget                                                 │
│                                                                      │
│     │                                                              │
│     ▼                                                              │
│  🤖 MULTI-AGENT ORCHESTRATOR (Brain)                              │
│     │                                                              │
│     ├── 📋 AGENTS.md (team structure)                             │
│     ├── 📡 sessions_send (messaging)                              │
│     ├── 🚀 sessions_spawn (sub-agents)                            │
│     └── 💾 Memory flush protocol                                   │
│                                                                      │
│     │                                                              │
│     ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    SPECIALIZED AGENTS (Muscles)               │  │
│  │                                                              │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │  │
│  │  │COORDINATOR│  │RESEARCHER│  │ DEVBOT   │  │ AUDITOR  │   │  │
│  │  │  Brain   │  │  Web     │  │  Code    │  │Security  │   │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │  │
│  │                                                              │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │  │
│  │  │UX_MANAGER│  │  TRADER  │  │   RISK    │              │  │
│  │  │ Dashboard│  │  Jupiter │  │ Assessment│              │  │
│  │  └──────────┘  └──────────┘  └──────────┘              │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│              ┌─────────────────────────────┐                      │
│              │    MiniMax 2.1 (MoE)        │                      │
│              │  230B params, 10B active    │                      │
│              │  Thinking mode enabled      │                      │
│              └─────────────────────────────┘                      │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    TRADING LAYER                             │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐   │  │
│  │  │ Solana/Jupiter │  │ Genetic       │  │ Backtesting   │   │  │
│  │  │ DEX Integration│  │ Strategy Miner│  │ Engine        │   │  │
│  │  └────────────────┘  └────────────────┘  └────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Files Implemented

### Core Architecture
| File | Purpose |
|------|---------|
| `mission_control.py` | Basic orchestrator |
| `mission_control_v2.py` | Enhanced with MiniMax 2.1 specs |
| `agents/AGENTS.md` | Team structure documentation |
| `agents/multi_agent_orchestrator.py` | Full multi-agent system |
| `agents/openclaw_config.json` | OpenClaw config template |

### Trading Layer
| File | Purpose |
|------|---------|
| `strategies/__init__.py` | RSI, SMA, MACD, Bollinger strategies |
| `strategies/genetic_miner.py` | Genetic algorithm optimizer |
| `workers/jupiter_worker.py` | Real-time price monitoring |
| `data/historical_data.py` | Real SOL price data |

### Integrations
| File | Purpose |
|------|---------|
| `telegram_bot.py` | Telegram bot commands |
| `coordinator.py` | Distributed work coordinator |
| `crypto_worker.py` | Worker client |
| `AGENT_ECONOMY.md` | Agent economy research |

---

## 🎯 Key Features

### Multi-Agent Communication
```python
# Send message between agents
await orchestrator.messaging.send_message(
    from_agent="coordinator",
    to_agent="researcher",
    message="Analyze SOL trends",
    priority="high"
)

# Spawn sub-agent for parallel work
await orchestrator.spawner.spawn(
    parent_agent="coordinator",
    task="Implement RSI strategy",
    model="minimax/MiniMax-M2.1-lightning"
)
```

### MiniMax 2.1 Integration
```python
minimax = MiniMaxClient(config=MiniMaxConfig(
    model="minimax/MiniMax-M2.1",
    thinking=True,  # Required for CoT
    max_output=128000,
    context_window=204800
))

thinking = await minimax.think(
    "Analyze this trading task...",
    context={"market": "SOL"}
)
```

### Trading Workflow
```
┌─────────────────────────────────────────┐
│ 1. Researcher: Market analysis          │
│    → Signals identified                 │
│                                         │
│ 2. Trader: Execute strategy             │
│    → Position opened                   │
│                                         │
│ 3. Risk: Validate limits                │
│    → Approved                          │
│                                         │
│ 4. Auditor: Review compliance            │
│    → Clean                            │
│                                         │
│ 5. UX Manager: Update dashboard          │
│    → Visualized                        │
└─────────────────────────────────────────┘
```

---

## 🔐 Security Framework

### Configured Protections
| Mechanism | Status |
|-----------|--------|
| Manual approval for trades | ✅ |
| Docker sandboxing | ✅ |
| Secret redaction | ✅ |
| Loopback binding | ✅ |
| Memory flush protocol | ✅ |

### Trade Limits
- **Max position:** 20% of portfolio
- **Daily loss limit:** 5%
- **Approval required:** Above 10%

---

## 📊 Specifications

### MiniMax 2.1
| Parameter | Value |
|-----------|-------|
| Total parameters | 230B |
| Active parameters | 10B (MoE) |
| Context window | 204,800 tokens |
| Max output | 128,000 tokens |
| Thinking mode | Enabled |

### PersonaPlex (Future)
| Parameter | Value |
|-----------|-------|
| Latency | 80ms |
| Sampling | 24kHz |
| Codec | Mimi (1.1kbps) |
| Deployment | Docker |

---

## 🚀 Quick Start

```bash
# Start multi-agent orchestrator
python3 agents/multi_agent_orchestrator.py

# Run Mission Control v2
python3 mission_control_v2.py

# Start trading dashboard
streamlit run dashboard/solana_dashboard.py

# Connect workers to coordinator
COORDINATOR_URL="http://localhost:5001" python3 crypto_worker.py
```

---

## 📚 References

| Topic | Source |
|-------|--------|
| OpenClaw | https://docs.openclaw.ai |
| PersonaPlex | https://huggingface.co/nvidia/personaplex-7b-v1 |
| MiniMax | https://platform.minimax.io/docs |
| Security - CrowdStrike | https://www.crowdstrike.com |
| Security - JFrog | https://jfrog.com/blog |

---

*Implementation based on comprehensive research of OpenClaw, PersonaPlex, and multi-agent systems.*
*Eko - Autonomous AI Trading Agent*
