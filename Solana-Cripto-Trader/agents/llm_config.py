#!/usr/bin/env python3
"""
Solana Trading Agents — LLM Configuration
MiniMax M2.5 (Anthropic-compatible API)
"""
import json
import time
import requests
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/enderj/.openclaw/workspace")
BITTRADER = WORKSPACE / "bittrader"
KEYS_DIR = BITTRADER / "keys"

# ── Circuit Breaker ────────────────────────────────────────────────────────
LLM_HEALTH_FILE = Path(__file__).parent / "data" / "llm_health.json"
CB_FAILURE_THRESHOLD = 3   # Failures before opening circuit
CB_COOLDOWN_CALLS    = 5   # Number of calls to skip during cooldown


def _load_health() -> dict:
    """Load circuit breaker state from disk."""
    try:
        if LLM_HEALTH_FILE.exists():
            return json.loads(LLM_HEALTH_FILE.read_text())
    except Exception:
        pass
    return {"consecutive_failures": 0, "cooldown_remaining": 0}


def _save_health(state: dict):
    """Persist circuit breaker state."""
    try:
        LLM_HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        LLM_HEALTH_FILE.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


def _cb_is_open() -> bool:
    """Returns True if circuit is open (should block call)."""
    state = _load_health()
    if state["cooldown_remaining"] > 0:
        # Decrement cooldown counter and save
        state["cooldown_remaining"] -= 1
        _save_health(state)
        print(f"    ⚡ Circuit breaker OPEN — cooldown restante: {state['cooldown_remaining'] + 1} llamadas")
        return True
    return False


def _cb_record_success():
    """Reset circuit breaker on success."""
    _save_health({"consecutive_failures": 0, "cooldown_remaining": 0})


def _cb_record_failure():
    """Record a failure; open circuit if threshold reached."""
    state = _load_health()
    state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
    if state["consecutive_failures"] >= CB_FAILURE_THRESHOLD:
        state["cooldown_remaining"] = CB_COOLDOWN_CALLS
        print(f"    🚨 Circuit breaker ABIERTO tras {state['consecutive_failures']} fallos — enfriando {CB_COOLDOWN_CALLS} llamadas")
    _save_health(state)

# ══════════════════════════════════════════════════════════════════════
# MINIMAX M2.5 (Anthropic-compatible API)
# ══════════════════════════════════════════════════════════════════════

MINIMAX_KEY = json.loads((KEYS_DIR / "minimax.json").read_text())["minimax_api_key"]
MINIMAX_URL = "https://api.minimax.io/anthropic/v1/messages"
MINIMAX_MODEL = "MiniMax-M2.5"

# ══════════════════════════════════════════════════════════════════════
# LLM CALL FUNCTION
# ══════════════════════════════════════════════════════════════════════

def call_llm(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """
    Llama a MiniMax M2.5 vía Anthropic-compatible API.
    Incluye circuit breaker: si hay >= 3 fallos consecutivos, se omiten
    las próximas 5 llamadas (cooldown). Se resetea al primer éxito.
    """
    # ── Circuit Breaker Check ─────────────────────────────────────────────
    if _cb_is_open():
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINIMAX_KEY}",
        "anthropic-version": "2023-06-01"
    }

    # Build request (Anthropic format: "system" is a top-level field, not a user message)
    data = {
        "model": MINIMAX_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,  # Más determinista para outputs JSON de trading
    }
    if system:
        data["system"] = system

    try:
        response = requests.post(MINIMAX_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()

        # Parsear respuesta Anthropic-compatible
        # MiniMax devuelve content con múltiples objetos (thinking, text)
        if "content" in result and len(result["content"]) > 0:
            # Filtrar solo objetos tipo "text"
            text_parts = [c["text"] for c in result["content"] if c.get("type") == "text"]
            if text_parts:
                _cb_record_success()  # ✅ Éxito — resetear circuit breaker
                return "\n".join(text_parts)

        print(f"    ⚠️ Respuesta inesperada: {json.dumps(result, indent=2)[:300]}")
        _cb_record_failure()
        return None

    except requests.exceptions.Timeout:
        print(f"    ⚠️ Timeout (60s) en API MiniMax")
        _cb_record_failure()
        return None
    except Exception as e:
        print(f"    ⚠️ MiniMax error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"       Status: {e.response.status_code}")
            print(f"       Body: {e.response.text[:300]}")
        _cb_record_failure()
        return None


# ══════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧪 Test: MiniMax M2.5 Anthropic API...")
    result = call_llm("Hola, preséntate en una frase.", "Eres un asistente de trading.")
    if result:
        print(f"✅ MiniMax: {result}")
    else:
        print("❌ MiniMax falló")
