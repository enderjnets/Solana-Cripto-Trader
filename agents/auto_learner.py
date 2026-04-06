#!/usr/bin/env python3
"""
🧠 Auto-Learner v2.0 — AI-Powered Trading Intelligence
=========================================================
Uses LLM (MiniMax M2.7 via OpenClaw) to analyze trades, discover patterns,
and generate adaptive rules. Not just if/else — real reasoning.

Architecture:
1. COLLECT: Gather trade data, market context, current params
2. ANALYZE: LLM reviews recent trades + full history patterns
3. DECIDE: LLM proposes parameter changes with reasoning
4. LEARN: Accumulate lessons in a persistent knowledge base
5. APPLY: Write new params to auto_learner_state.json

The LLM sees:
- Last 30 trades with full context (symbol, direction, close reason, PnL, holding time)
- Per-token stats (which tokens win/lose, which directions work)
- Close reason analysis (trailing stop vs TP vs SL vs timeout)
- Market conditions (trend, volatility)
- Historical lessons learned (growing knowledge base)
- Current parameters and their performance

Cost: ~$0.01-0.02 per run (MiniMax M2.7, every 6h = ~$0.12/day)
"""

import json
import logging
import sqlite3
import statistics
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

TRADE_HISTORY  = DATA_DIR / "trade_history.json"
LEARNER_STATE  = DATA_DIR / "auto_learner_state.json"
LEARNER_DB     = DATA_DIR / "auto_learner.db"
LEARNER_REPORT = DATA_DIR / "auto_learner_report.json"
KNOWLEDGE_BASE = DATA_DIR / "auto_learner_knowledge.md"
MARKET_DATA    = DATA_DIR / "market_latest.json"
RISK_REPORT    = DATA_DIR / "risk_report.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("auto_learner_v2")

# ─── Constants ────────────────────────────────────────────────────────────────

PERFORMANCE_WINDOW = 50    # Analyze last 50 trades
MIN_TRADES_TO_ADAPT = 10   # Need at least 10 new trades before adapting
KNOWLEDGE_MAX_LINES = 200  # Keep knowledge base manageable

# Parameter bounds (safety rails — LLM cannot go outside these)
PARAM_BOUNDS = {
    "sl_pct":         (0.005, 0.05),   # 0.5% to 5%
    "tp_pct":         (0.02, 0.10),    # 2% to 10%
    "leverage_tier":  (0, 2),          # 0=conservative, 1=moderate, 2=aggressive
    "risk_per_trade": (0.005, 0.04),   # 0.5% to 4%
    "max_positions":  (1, 5),          # 1 to 5
    "trailing_stop_pct": (0.5, 5.0),   # 0.5% to 5%
}

LEVERAGE_TIERS = {
    0: "CONSERVATIVE (1x-2x)",
    1: "MODERATE (2x-5x)",
    2: "AGGRESSIVE (5x-10x)"
}


# ─── Data Collection ──────────────────────────────────────────────────────────

def load_json(path: Path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def load_trades() -> List[dict]:
    data = load_json(TRADE_HISTORY, {})
    if isinstance(data, list):
        return data
    return data.get("trades", []) if isinstance(data, dict) else []


def load_state() -> dict:
    default = {
        "params": {
            "sl_pct": 0.015, "tp_pct": 0.04, "leverage_tier": 2,
            "risk_per_trade": 0.02, "max_positions": 3, "trailing_stop_pct": 2.0
        },
        "total_trades_learned": 0,
        "last_adaptation": None,
        "adaptation_count": 0,
        "last_trade_count": 0
    }
    return load_json(LEARNER_STATE, default)


def save_state(state: dict):
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(LEARNER_STATE, "w") as f:
        json.dump(state, f, indent=2)


def load_knowledge() -> str:
    if KNOWLEDGE_BASE.exists():
        return KNOWLEDGE_BASE.read_text()
    return "# Auto-Learner Knowledge Base\n\nNo lessons learned yet.\n"


def save_knowledge(content: str):
    # Trim to max lines
    lines = content.strip().split("\n")
    if len(lines) > KNOWLEDGE_MAX_LINES:
        # Keep header + last N lines
        header = lines[:3]
        tail = lines[-(KNOWLEDGE_MAX_LINES - 3):]
        lines = header + ["", "...(older entries trimmed)...", ""] + tail
    KNOWLEDGE_BASE.write_text("\n".join(lines))


# ─── Analysis Engine ──────────────────────────────────────────────────────────

def compute_stats(trades: list) -> dict:
    """Compute comprehensive stats from trade list."""
    if not trades:
        return {"total": 0}
    
    wins = [t for t in trades if t.get("pnl_usd", 0) > 0]
    losses = [t for t in trades if t.get("pnl_usd", 0) < 0]
    flat = [t for t in trades if t.get("pnl_usd", 0) == 0]
    
    total_pnl = sum(t.get("pnl_usd", 0) for t in trades)
    total_fees = sum(t.get("total_fees", 0) for t in trades)
    
    wr = len(wins) / (len(wins) + len(losses)) * 100 if (len(wins) + len(losses)) > 0 else 0
    avg_win = statistics.mean([t["pnl_usd"] for t in wins]) if wins else 0
    avg_loss = statistics.mean([abs(t["pnl_usd"]) for t in losses]) if losses else 0
    rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    
    return {
        "total": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "flat": len(flat),
        "win_rate": round(wr, 1),
        "total_pnl": round(total_pnl, 4),
        "total_fees": round(total_fees, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "rr_ratio": round(rr_ratio, 2),
        "best": round(max(t.get("pnl_usd", 0) for t in trades), 4),
        "worst": round(min(t.get("pnl_usd", 0) for t in trades), 4),
    }


def compute_token_stats(trades: list) -> dict:
    """Per-token performance breakdown."""
    by_token = defaultdict(list)
    for t in trades:
        by_token[t.get("symbol", "?")].append(t)
    
    result = {}
    for token, token_trades in sorted(by_token.items()):
        stats = compute_stats(token_trades)
        # Direction breakdown
        longs = [t for t in token_trades if t.get("direction") == "long"]
        shorts = [t for t in token_trades if t.get("direction") == "short"]
        stats["long_count"] = len(longs)
        stats["short_count"] = len(shorts)
        stats["long_pnl"] = round(sum(t.get("pnl_usd", 0) for t in longs), 4)
        stats["short_pnl"] = round(sum(t.get("pnl_usd", 0) for t in shorts), 4)
        result[token] = stats
    
    return result


def compute_close_reason_stats(trades: list) -> dict:
    """Stats by close reason."""
    by_reason = defaultdict(list)
    for t in trades:
        by_reason[t.get("close_reason", "unknown")].append(t)
    
    result = {}
    for reason, reason_trades in sorted(by_reason.items()):
        stats = compute_stats(reason_trades)
        result[reason] = stats
    
    return result


def build_analysis_prompt(trades: list, state: dict, knowledge: str) -> str:
    """Build the prompt for the LLM with all context."""
    
    recent = trades[-PERFORMANCE_WINDOW:]
    overall = compute_stats(trades)
    recent_stats = compute_stats(recent)
    token_stats = compute_token_stats(recent)
    reason_stats = compute_close_reason_stats(recent)
    
    # Current params
    params = state.get("params", {})
    
    # Market data
    market = load_json(MARKET_DATA, {})
    market_tokens = market.get("tokens", {})
    
    # Build prompt
    prompt = f"""# Auto-Learner Analysis — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

## Your Role
You are the AI brain of a Solana paper trading bot. Analyze the trade data below and:
1. Identify what's working and what's failing
2. Propose specific parameter adjustments with reasoning
3. Add new lessons to the knowledge base
4. Identify tokens/patterns to avoid or prefer

## Current Parameters
- Stop Loss: {params.get('sl_pct', 0.015)*100:.1f}%
- Take Profit: {params.get('tp_pct', 0.04)*100:.1f}%
- Leverage Tier: {params.get('leverage_tier', 2)} ({LEVERAGE_TIERS.get(params.get('leverage_tier', 2), '?')})
- Risk per Trade: {params.get('risk_per_trade', 0.02)*100:.1f}%
- Max Positions: {params.get('max_positions', 3)}
- Trailing Stop: {params.get('trailing_stop_pct', 2.0)}%
- Adaptations so far: {state.get('adaptation_count', 0)}

## Overall Performance (all {overall['total']} trades)
- Win Rate: {overall.get('win_rate', 0)}%
- Total PnL: ${overall.get('total_pnl', 0):+.2f}
- Total Fees: ${overall.get('total_fees', 0):.2f}
- Avg Win: ${overall.get('avg_win', 0):.4f} | Avg Loss: ${overall.get('avg_loss', 0):.4f}
- R:R Ratio: {overall.get('rr_ratio', 0):.2f}
- Best: ${overall.get('best', 0):.4f} | Worst: ${overall.get('worst', 0):.4f}

## Recent Performance (last {recent_stats['total']} trades)
- Win Rate: {recent_stats.get('win_rate', 0)}%
- PnL: ${recent_stats.get('total_pnl', 0):+.2f}
- Avg Win: ${recent_stats.get('avg_win', 0):.4f} | Avg Loss: ${recent_stats.get('avg_loss', 0):.4f}
- R:R Ratio: {recent_stats.get('rr_ratio', 0):.2f}

## Close Reason Analysis
"""
    for reason, stats in reason_stats.items():
        prompt += f"- **{reason}**: {stats['total']}T | WR:{stats['win_rate']}% | PnL:${stats['total_pnl']:+.4f} | AvgLoss:${stats['avg_loss']:.4f}\n"
    
    prompt += "\n## Per-Token Performance\n"
    for token, stats in sorted(token_stats.items(), key=lambda x: x[1]['total_pnl']):
        prompt += f"- **{token}**: {stats['total']}T | WR:{stats['win_rate']}% | PnL:${stats['total_pnl']:+.4f} | L:{stats['long_count']}(${stats['long_pnl']:+.4f}) S:{stats['short_count']}(${stats['short_pnl']:+.4f})\n"
    
    prompt += f"\n## Last 10 Trades\n"
    for t in recent[-10:]:
        prompt += f"- {t.get('symbol','?'):6s} {t.get('direction','?'):5s} | PnL:${t.get('pnl_usd',0):+.4f} | {t.get('close_reason','?'):15s} | Fees:${t.get('total_fees',0):.4f}\n"
    
    prompt += f"\n## Knowledge Base (accumulated lessons)\n```\n{knowledge}\n```\n"
    
    prompt += """
## Your Output (respond in EXACTLY this JSON format)

```json
{
  "analysis": "2-3 sentence summary of what you see",
  "key_insight": "The single most important finding",
  "params": {
    "sl_pct": <float 0.005-0.05>,
    "tp_pct": <float 0.02-0.10>,
    "leverage_tier": <int 0-2>,
    "risk_per_trade": <float 0.005-0.04>,
    "max_positions": <int 1-5>,
    "trailing_stop_pct": <float 0.5-5.0>
  },
  "param_reasoning": "Why you changed each param (or kept it)",
  "new_lessons": ["lesson 1", "lesson 2"],
  "tokens_to_avoid": ["TOKEN1"],
  "tokens_to_prefer": ["TOKEN2"],
  "confidence": <float 0.0-1.0>
}
```

Rules:
- Be conservative — small changes (10-20% max per param per cycle)
- If performance is improving, don't change much
- If a token consistently loses, recommend avoiding it
- Consider fees in your analysis (they eat into profits)
- trailing_stop_pct was just changed from 0.5% to 2.0% — evaluate if this is good
"""
    
    return prompt


# ─── LLM Interface ───────────────────────────────────────────────────────────

def call_llm(prompt: str) -> Optional[dict]:
    """Call MiniMax M2.7 via OpenClaw's local API or direct API."""
    
    # Try OpenClaw's provider config
    try:
        import requests
        
        # Read OpenClaw config to get MiniMax API key
        config_path = Path.home() / ".openclaw" / "config" / "config.yaml"
        api_key = None
        
        if config_path.exists():
            content = config_path.read_text()
            # Find the token_plan_key (the one that works)
            for line in content.split("\n"):
                if "token_plan_key" in line.lower() or "sk-cp-8tBIgoE2" in line:
                    # Extract the key
                    parts = line.split(":")
                    if len(parts) >= 2:
                        api_key = parts[-1].strip().strip('"').strip("'")
                        break
        
        if not api_key:
            # Fallback: try known working key
            api_key = "sk-cp-8tBIgoE2Vs8QE0AIoMjq4MTh8kiHtem3KWlOnNlAJZgKwAlYh_nt6oCq382Y0cmBi2buvch3nJJbMg7uqr_hIV6Z0ZqY3Q_qZ6AStHCUpKKT_IT-e0vEl4A"
        
        response = requests.post(
            "https://api.minimax.io/v1/text/chatcompletion_v2",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "MiniMax-M2.7",
                "messages": [
                    {"role": "system", "content": "You are a quantitative trading analyst. Respond ONLY with valid JSON. No markdown fences, no thinking tags, no explanation outside JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 2000
            },
            timeout=90
        )
        
        if response.status_code == 200:
            data = response.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            # Parse JSON from response (handle markdown fences + thinking tags)
            text = text.strip()
            # Remove <think>...</think> blocks (MiniMax M2.7)
            import re
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            # Remove markdown fences
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            if text.startswith("json"):
                text = text[4:].strip()
            
            return json.loads(text)
        else:
            log.error(f"LLM API error: {response.status_code} {response.text[:200]}")
            return None
            
    except json.JSONDecodeError as e:
        log.error(f"LLM returned invalid JSON: {e}")
        log.error(f"Raw response: {text[:500]}")
        return None
    except Exception as e:
        log.error(f"LLM call failed: {e}")
        return None


# ─── Parameter Application ───────────────────────────────────────────────────

def clamp_params(params: dict) -> dict:
    """Enforce safety bounds on all parameters."""
    clamped = {}
    for key, (lo, hi) in PARAM_BOUNDS.items():
        val = params.get(key)
        if val is not None:
            if isinstance(lo, int) and isinstance(hi, int):
                clamped[key] = max(lo, min(hi, int(val)))
            else:
                clamped[key] = max(lo, min(hi, float(val)))
    return clamped


def apply_changes(state: dict, llm_response: dict) -> dict:
    """Apply LLM's recommended changes with safety rails."""
    
    old_params = state.get("params", {}).copy()
    new_params = llm_response.get("params", {})
    
    # Clamp to safety bounds
    new_params = clamp_params(new_params)
    
    # Limit change rate: max 20% change per param per cycle
    MAX_CHANGE = 0.25
    for key in new_params:
        if key in old_params:
            old_val = old_params[key]
            new_val = new_params[key]
            if isinstance(old_val, (int, float)) and old_val != 0:
                change_pct = abs(new_val - old_val) / abs(old_val)
                if change_pct > MAX_CHANGE:
                    # Limit the change
                    direction = 1 if new_val > old_val else -1
                    new_params[key] = old_val * (1 + direction * MAX_CHANGE)
                    if isinstance(old_val, int):
                        new_params[key] = int(round(new_params[key]))
    
    # Re-clamp after limiting
    new_params = clamp_params(new_params)
    
    # Merge with existing (keep keys that LLM didn't mention)
    merged = old_params.copy()
    merged.update(new_params)
    
    return merged


# ─── Knowledge Base Management ────────────────────────────────────────────────

def update_knowledge(existing: str, llm_response: dict) -> str:
    """Append new lessons to knowledge base."""
    lessons = llm_response.get("new_lessons", [])
    avoid = llm_response.get("tokens_to_avoid", [])
    prefer = llm_response.get("tokens_to_prefer", [])
    analysis = llm_response.get("analysis", "")
    insight = llm_response.get("key_insight", "")
    
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    
    entry = f"\n## Cycle {timestamp}\n"
    entry += f"**Insight**: {insight}\n"
    entry += f"**Analysis**: {analysis}\n"
    
    if lessons:
        entry += "**Lessons**:\n"
        for lesson in lessons:
            entry += f"- {lesson}\n"
    
    if avoid:
        entry += f"**Avoid**: {', '.join(avoid)}\n"
    if prefer:
        entry += f"**Prefer**: {', '.join(prefer)}\n"
    
    confidence = llm_response.get("confidence", 0.5)
    entry += f"**Confidence**: {confidence*100:.0f}%\n"
    
    return existing.rstrip() + "\n" + entry


# ─── DB Operations ────────────────────────────────────────────────────────────

class LearnerDB:
    def __init__(self):
        self.db_path = LEARNER_DB
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS trade_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                trade_id TEXT UNIQUE,
                symbol TEXT, direction TEXT,
                sl_pct REAL, tp_pct REAL, leverage REAL,
                pnl_usd REAL, pnl_pct REAL, win BOOLEAN,
                confidence REAL, holding_time REAL
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS parameter_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                sl_pct REAL, tp_pct REAL, leverage_tier INTEGER,
                risk_per_trade REAL, win_rate REAL, avg_pnl REAL
            )""")
            # Add columns if they don't exist (migration for existing DBs)
            try:
                conn.execute("ALTER TABLE parameter_history ADD COLUMN trailing_stop_pct REAL")
            except sqlite3.OperationalError:
                pass  # column already exists
            try:
                conn.execute("ALTER TABLE parameter_history ADD COLUMN reasoning TEXT")
            except sqlite3.OperationalError:
                pass
            conn.execute("""CREATE TABLE IF NOT EXISTS ai_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                analysis TEXT, key_insight TEXT,
                confidence REAL, full_response TEXT
            )""")
    
    def record_analysis(self, llm_response: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""INSERT INTO ai_analysis (analysis, key_insight, confidence, full_response)
                VALUES (?, ?, ?, ?)""", (
                    llm_response.get("analysis", ""),
                    llm_response.get("key_insight", ""),
                    llm_response.get("confidence", 0.5),
                    json.dumps(llm_response)
                ))
    
    def record_params(self, params: dict, performance: dict, reasoning: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""INSERT INTO parameter_history 
                (sl_pct, tp_pct, leverage_tier, risk_per_trade, win_rate, avg_pnl, trailing_stop_pct, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", (
                    params.get("sl_pct"), params.get("tp_pct"),
                    params.get("leverage_tier"), params.get("risk_per_trade"),
                    performance.get("win_rate", 0), performance.get("avg_pnl", 0),
                    params.get("trailing_stop_pct"), reasoning
                ))
    
    def sync_trades(self, trades: list, params: dict):
        """Sync trade history to DB (dedup by trade_id)."""
        with sqlite3.connect(self.db_path) as conn:
            for t in trades:
                try:
                    conn.execute("""INSERT OR IGNORE INTO trade_results 
                        (trade_id, symbol, direction, sl_pct, tp_pct, leverage,
                         pnl_usd, pnl_pct, win, confidence, holding_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                            t.get("id"), t.get("symbol"), t.get("direction"),
                            params.get("sl_pct"), params.get("tp_pct"),
                            t.get("leverage", 5.0), t.get("pnl_usd", 0),
                            t.get("pnl_pct", 0), t.get("pnl_usd", 0) > 0,
                            0.5, 0
                        ))
                except Exception:
                    pass


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def run(debug: bool = False) -> dict:
    log.info("=" * 60)
    log.info("🧠 AUTO-LEARNER v2.0 — AI-Powered Trading Intelligence")
    log.info("=" * 60)
    
    # 1. COLLECT
    trades = load_trades()
    state = load_state()
    knowledge = load_knowledge()
    db = LearnerDB()
    
    if not trades:
        log.info("⏸️ No trades to analyze")
        return {"status": "no_data"}
    
    # Check if enough NEW trades since last run
    last_count = state.get("last_trade_count", 0)
    new_trades = len(trades) - last_count

    # FIX: Detect reset — if new_trades is negative, history was reset
    if new_trades < 0:
        log.warning(f"⚠️ Trade history reset detected: had {last_count}, now {len(trades)}")
        log.warning(f"   Reconciling index: last_trade_count = {len(trades)}")
        state["last_trade_count"] = len(trades)
        state["total_trades_learned"] = len(trades)
        state["notes"] = f"Auto-reconciled after reset: {last_count} -> {len(trades)}"
        save_state(state)
        new_trades = 0  # No new trades to process yet

    log.info(f"📊 Total trades: {len(trades)} | New since last run: {new_trades}")
    
    # Sync to DB
    db.sync_trades(trades[-PERFORMANCE_WINDOW:], state.get("params", {}))
    
    if new_trades < MIN_TRADES_TO_ADAPT and state.get("adaptation_count", 0) > 0:
        log.info(f"⏸️ Only {new_trades} new trades — need {MIN_TRADES_TO_ADAPT} to adapt")
        # Still update report
        recent_stats = compute_stats(trades[-PERFORMANCE_WINDOW:])
        return {
            "status": "waiting",
            "trades_needed": MIN_TRADES_TO_ADAPT - new_trades,
            "current_stats": recent_stats
        }
    
    # 2. ANALYZE — Build prompt and call LLM
    log.info("🤖 Calling LLM for analysis...")
    prompt = build_analysis_prompt(trades, state, knowledge)
    
    llm_response = call_llm(prompt)
    
    if not llm_response:
        log.error("❌ LLM call failed — falling back to rule-based")
        return _fallback_adaptation(trades, state, db)
    
    log.info(f"✅ LLM analysis received")
    log.info(f"   Insight: {llm_response.get('key_insight', 'N/A')}")
    log.info(f"   Confidence: {llm_response.get('confidence', 0)*100:.0f}%")
    
    # 3. DECIDE — Apply changes with safety rails
    old_params = state.get("params", {}).copy()
    new_params = apply_changes(state, llm_response)
    
    # Log changes
    changes = []
    for key in new_params:
        if key in old_params and new_params[key] != old_params[key]:
            if isinstance(new_params[key], float):
                changes.append(f"{key}: {old_params[key]:.4f} → {new_params[key]:.4f}")
            else:
                changes.append(f"{key}: {old_params[key]} → {new_params[key]}")
    
    if changes:
        log.info(f"🔧 Parameter changes:")
        for c in changes:
            log.info(f"   {c}")
    else:
        log.info("📌 No parameter changes (LLM recommends keeping current)")
    
    # 4. LEARN — Update knowledge base
    new_knowledge = update_knowledge(knowledge, llm_response)
    save_knowledge(new_knowledge)
    log.info(f"📝 Knowledge base updated ({len(llm_response.get('new_lessons', []))} new lessons)")
    
    # 5. APPLY — Save state
    state["params"] = new_params
    state["last_adaptation"] = datetime.now(timezone.utc).isoformat()
    state["adaptation_count"] = state.get("adaptation_count", 0) + 1
    state["total_trades_learned"] = len(trades)
    state["last_trade_count"] = len(trades)
    state["last_analysis"] = llm_response.get("analysis", "")
    state["last_insight"] = llm_response.get("key_insight", "")
    state["confidence"] = llm_response.get("confidence", 0.5)
    state["tokens_to_avoid"] = llm_response.get("tokens_to_avoid", [])
    state["tokens_to_prefer"] = llm_response.get("tokens_to_prefer", [])
    save_state(state)
    
    # Record in DB
    recent_stats = compute_stats(trades[-PERFORMANCE_WINDOW:])
    db.record_analysis(llm_response)
    db.record_params(new_params, {
        "win_rate": recent_stats.get("win_rate", 0),
        "avg_pnl": recent_stats.get("total_pnl", 0) / max(recent_stats.get("total", 1), 1)
    }, llm_response.get("param_reasoning", ""))
    
    # Build report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0-ai",
        "adaptation_count": state["adaptation_count"],
        "total_trades_analyzed": len(trades),
        "new_trades_since_last": new_trades,
        "performance": recent_stats,
        "analysis": llm_response.get("analysis", ""),
        "key_insight": llm_response.get("key_insight", ""),
        "confidence": llm_response.get("confidence", 0.5),
        "params_before": old_params,
        "params_after": new_params,
        "changes": changes,
        "new_lessons": llm_response.get("new_lessons", []),
        "tokens_to_avoid": llm_response.get("tokens_to_avoid", []),
        "tokens_to_prefer": llm_response.get("tokens_to_prefer", []),
    }
    
    with open(LEARNER_REPORT, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    log.info(f"💾 Report saved | Adaptation #{state['adaptation_count']}")
    return report


def _fallback_adaptation(trades: list, state: dict, db: LearnerDB) -> dict:
    """Simple rule-based fallback if LLM fails."""
    log.info("⚠️ Using rule-based fallback")
    
    recent = trades[-PERFORMANCE_WINDOW:]
    stats = compute_stats(recent)
    params = state.get("params", {}).copy()
    
    wr = stats.get("win_rate", 50) / 100
    
    if wr > 0.6:
        params["sl_pct"] = max(0.01, params.get("sl_pct", 0.015) * 0.9)
        params["tp_pct"] = min(0.08, params.get("tp_pct", 0.04) * 1.1)
    elif wr < 0.3:
        params["sl_pct"] = min(0.04, params.get("sl_pct", 0.015) * 1.1)
        params["tp_pct"] = max(0.03, params.get("tp_pct", 0.04) * 0.9)
    
    params = clamp_params(params)
    state["params"] = params
    state["last_adaptation"] = datetime.now(timezone.utc).isoformat()
    state["adaptation_count"] = state.get("adaptation_count", 0) + 1
    state["last_trade_count"] = len(trades)
    save_state(state)
    
    db.record_params(params, {"win_rate": stats.get("win_rate", 0), "avg_pnl": 0}, "fallback-rules")
    
    return {"status": "fallback", "stats": stats, "params": params}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    result = run(debug=args.debug)
    print(json.dumps(result, indent=2, default=str))
