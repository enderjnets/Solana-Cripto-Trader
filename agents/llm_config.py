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
        # Reset consecutive failures while in cooldown so they don't accumulate
        state["consecutive_failures"] = 0
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
# PRIMARY: Claude Sonnet 4.6 via Direct Anthropic API
# Fix 2026-03-23: Gateway port 8443 was wrong (18789 is dashboard, not API)
#                 Now using api.anthropic.com directly with key from config
# ══════════════════════════════════════════════════════════════════════

CLAUDE_BASE_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL    = "claude-sonnet-4-6"


def _get_claude_key() -> tuple:
    """Load Anthropic API key + extra headers from openclaw config."""
    try:
        import json as _json
        cfg_path = Path.home() / ".openclaw" / "openclaw.json"
        if cfg_path.exists():
            cfg = _json.loads(cfg_path.read_text())
            # Structure: models.providers.claude
            provider = cfg.get("models", {}).get("providers", {}).get("claude", {})
            key = provider.get("apiKey", "")
            headers_extra = provider.get("headers", {})
            return key, headers_extra
    except Exception:
        pass
    return "", {}


def call_claude_sonnet(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """Claude Sonnet 4.6 via Direct Anthropic API — PRIMARY"""
    api_key, extra_headers = _get_claude_key()
    if not api_key:
        print("    ⚠️ Claude Sonnet: No API key found in openclaw config")
        return None
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        **extra_headers
    }
    messages = [{"role": "user", "content": prompt}]
    data = {"model": CLAUDE_MODEL, "max_tokens": max_tokens, "messages": messages}
    if system:
        data["system"] = system
    try:
        r = requests.post(CLAUDE_BASE_URL, headers=headers, json=data, timeout=60)
        r.raise_for_status()
        result = r.json()
        content = result.get("content")
        if content and isinstance(content, list):
            texts = [c["text"] for c in content if c.get("type") == "text"]
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
MINIMAX_URL   = "https://api.minimax.io/v1/text/chatcompletion_v2"
MINIMAX_MODEL = "MiniMax-M2.5"

# ── OpenRouter config ──────────────────────────────────────────────────────
OPENROUTER_URL   = "https://openrouter.ai/api/v1/chat/completions"
NEMOTRON_MODEL   = "nvidia/nemotron-3-super-120b-a12b:free"


def _get_openrouter_key() -> str:
    try:
        import json as _json
        cfg_path = Path.home() / ".openclaw" / "openclaw.json"
        if cfg_path.exists():
            cfg = _json.loads(cfg_path.read_text())
            return cfg.get("models", {}).get("providers", {}).get("openrouter", {}).get("apiKey", "")
    except Exception:
        pass
    return ""


def call_nemotron(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """NVIDIA Nemotron Super 120B via OpenRouter (FREE) — PRIMARY"""
    api_key = _get_openrouter_key()
    if not api_key:
        print("    ⚠️ Nemotron: No OpenRouter API key found")
        return None
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": NEMOTRON_MODEL,
        "messages": messages,
        "max_tokens": max(max_tokens, 500)  # Nemotron needs min ~500 tokens
    }
    try:
        r = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=60)
        r.raise_for_status()
        result = r.json()
        content = result["choices"][0]["message"].get("content")
        if content:
            return content
        return None
    except Exception as e:
        print(f"    ⚠️ Nemotron error: {e}")
        return None


def call_minimax(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """MiniMax M2.5 — PRIMARY via standard chat completions API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINIMAX_KEY}",
    }
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    # Always use at least 4000 tokens — MiniMax M2.5 uses <think> blocks
    # that can consume 500-2000 tokens before the actual response
    effective_max = max(max_tokens, 4000)
    data = {
        "model": MINIMAX_MODEL,
        "messages": messages,
        "max_tokens": effective_max,
        "temperature": 0.2,
    }
    try:
        r = requests.post(MINIMAX_URL, headers=headers, json=data, timeout=60)
        r.raise_for_status()
        result = r.json()
        # Standard OpenAI-compatible response format
        choices = result.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            text = msg.get("content", "")
            # Strip <think>...</think> blocks if present
            if "<think>" in text and "</think>" in text:
                text = text.split("</think>", 1)[-1].strip()
            # If think block consumed all tokens, try to extract JSON from it
            if not text and "<think>" in msg.get("content", ""):
                think_content = msg["content"].split("<think>")[1].split("</think>")[0]
                # Try to find JSON in the think block
                import re
                json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', think_content)
                if json_match:
                    text = json_match.group(0)
            if text:
                return text
        return None
    except Exception as e:
        print(f"    ⚠️ MiniMax error: {type(e).__name__}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"    ⚠️ MiniMax response status: {e.response.status_code}")
            print(f"    ⚠️ MiniMax response body: {e.response.text[:200]}")
        return None


# ══════════════════════════════════════════════════════════════════════
# MAIN: call_llm with fallback chain
# ══════════════════════════════════════════════════════════════════════

def call_llm(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """
    LLM chain: MiniMax M2.5 (PRIMARY - reliable).
    Claude Sonnet fallback DISABLED — API key is invalid (401).
    Updated: 2026-03-30 — Removed broken Claude fallback to stop 324 consecutive failures.
    """
    if _cb_is_open():
        return None

    result = call_minimax(prompt, system, max_tokens)
    if result:
        _cb_record_success()
        return result

    _cb_record_failure()
    print("    ❌ MiniMax failed — circuit breaker activated")
    return None


if __name__ == "__main__":
    print("🧪 Test: Claude Sonnet 4.6...")
    r = call_llm("Responde solo: OK", "Eres un bot de trading.")
    print(f"{'✅' if r else '❌'} Claude: {r or 'falló'}")
