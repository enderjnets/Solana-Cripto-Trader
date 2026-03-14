#!/usr/bin/env python3
"""
Solana Trading Agents — LLM Configuration
Claude Sonnet 4.6 (PRIMARY via OpenClaw Gateway)
MiniMax M2.5 (FALLBACK via Anthropic-compatible API)
Updated: 2026-03-14 (Auditoría - Ingeniero)
"""
import json
import time
import requests
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/enderj/.openclaw/workspace")
BITTRADER = WORKSPACE / "bittrader"
KEYS_DIR  = BITTRADER / "keys"

# ── Circuit Breaker ────────────────────────────────────────────────────────
LLM_HEALTH_FILE      = Path(__file__).parent / "data" / "llm_health.json"
CB_FAILURE_THRESHOLD = 3
CB_COOLDOWN_CALLS    = 5


def _load_health() -> dict:
    try:
        if LLM_HEALTH_FILE.exists():
            return json.loads(LLM_HEALTH_FILE.read_text())
    except Exception:
        pass
    return {"consecutive_failures": 0, "cooldown_remaining": 0}


def _save_health(state: dict):
    try:
        LLM_HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        LLM_HEALTH_FILE.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


def _cb_is_open() -> bool:
    state = _load_health()
    if state["cooldown_remaining"] > 0:
        state["cooldown_remaining"] -= 1
        _save_health(state)
        print(f"    ⚡ Circuit breaker OPEN — cooldown: {state['cooldown_remaining'] + 1}")
        return True
    return False


def _cb_record_success():
    _save_health({"consecutive_failures": 0, "cooldown_remaining": 0})


def _cb_record_failure():
    state = _load_health()
    state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
    if state["consecutive_failures"] >= CB_FAILURE_THRESHOLD:
        state["cooldown_remaining"] = CB_COOLDOWN_CALLS
        print(f"    🚨 Circuit breaker ABIERTO — {CB_COOLDOWN_CALLS} llamadas de cooldown")
    _save_health(state)


# ══════════════════════════════════════════════════════════════════════
# PRIMARY: Claude Sonnet 4.6 via OpenClaw Gateway
# ══════════════════════════════════════════════════════════════════════

CLAUDE_BASE_URL = "http://127.0.0.1:8443/v1/messages"
CLAUDE_MODEL    = "claude-sonnet-4-6"


def call_claude_sonnet(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """Claude Sonnet 4.6 via OpenClaw Gateway — PRIMARY"""
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    messages = [{"role": "user", "content": prompt}]
    data = {"model": CLAUDE_MODEL, "max_tokens": max_tokens, "messages": messages}
    if system:
        data["system"] = system
    try:
        r = requests.post(CLAUDE_BASE_URL, headers=headers, json=data, timeout=60)
        r.raise_for_status()
        result = r.json()
        if "content" in result:
            texts = [c["text"] for c in result["content"] if c.get("type") == "text"]
            if texts:
                return "\n".join(texts)
        return None
    except Exception as e:
        print(f"    ⚠️ Claude Sonnet error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# FALLBACK: MiniMax M2.5
# ══════════════════════════════════════════════════════════════════════

MINIMAX_KEY   = json.loads((KEYS_DIR / "minimax.json").read_text())["minimax_api_key"]
MINIMAX_URL   = "https://api.minimax.io/anthropic/v1/messages"
MINIMAX_MODEL = "MiniMax-M2.5"


def call_minimax(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """MiniMax M2.5 — FALLBACK"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINIMAX_KEY}",
        "anthropic-version": "2023-06-01"
    }
    data = {
        "model": MINIMAX_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    if system:
        data["system"] = system
    try:
        r = requests.post(MINIMAX_URL, headers=headers, json=data, timeout=60)
        r.raise_for_status()
        result = r.json()
        if "content" in result:
            texts = [c["text"] for c in result["content"] if c.get("type") == "text"]
            if texts:
                return "\n".join(texts)
        return None
    except Exception as e:
        print(f"    ⚠️ MiniMax error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# MAIN: call_llm with fallback chain
# ══════════════════════════════════════════════════════════════════════

def call_llm(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """
    Fallback chain:
    1. Claude Sonnet 4.6 (PRIMARY - OpenClaw Gateway)
    2. MiniMax M2.5 (FALLBACK)
    """
    if _cb_is_open():
        return None

    # Try Claude first
    result = call_claude_sonnet(prompt, system, max_tokens)
    if result:
        _cb_record_success()
        return result

    print("    ⚠️ Claude Sonnet falló — intentando MiniMax fallback...")
    result = call_minimax(prompt, system, max_tokens)
    if result:
        _cb_record_success()
        return result

    _cb_record_failure()
    print("    ❌ Todos los LLMs fallaron")
    return None


if __name__ == "__main__":
    print("🧪 Test: Claude Sonnet 4.6...")
    r = call_llm("Responde solo: OK", "Eres un bot de trading.")
    print(f"{'✅' if r else '❌'} Claude: {r or 'falló'}")
