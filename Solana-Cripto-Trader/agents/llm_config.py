#!/usr/bin/env python3
"""
Solana Trading Agents — LLM Configuration
MiniMax M2.5 (Anthropic-compatible API)
"""
import json
import requests
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/enderj/.openclaw/workspace")
BITTRADER = WORKSPACE / "bittrader"
KEYS_DIR = BITTRADER / "keys"

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
    """
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
                return "\n".join(text_parts)

        print(f"    ⚠️ Respuesta inesperada: {json.dumps(result, indent=2)[:300]}")
        return None

    except requests.exceptions.Timeout:
        print(f"    ⚠️ Timeout (60s) en API MiniMax")
        return None
    except Exception as e:
        print(f"    ⚠️ MiniMax error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"       Status: {e.response.status_code}")
            print(f"       Body: {e.response.text[:300]}")
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
