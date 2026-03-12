#!/usr/bin/env python3
"""
BitTrader Agents — Shared LLM Configuration
Z.ai GLM-4.7 Coding Plan (PRIMARY) + MiniMax fallback
(Cambiado de GLM-5 a GLM-4.7 el 12 mar 2026 por API rate limit)
"""
import json
import os
import requests
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/enderj/.openclaw/workspace")
BITTRADER = WORKSPACE / "bittrader"
KEYS_DIR = BITTRADER / "keys"

# ══════════════════════════════════════════════════════════════════════
# ZAI GLM-5 CODING PLAN (PRIMARY)
# ══════════════════════════════════════════════════════════════════════

ZAI_CODING_KEY = "863f222d3de340df8b6ff7e1e36ab216.DFOH1veovvWzaoFT"
ZAI_CODING_BASE_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
ZAI_CODING_MODEL = "glm-4.7"  # Cambiado a GLM-4.7 por API rate limit (12 mar 2026)

# ══════════════════════════════════════════════════════════════════════
# MINIMAX M2.5 (FALLBACK)
# ══════════════════════════════════════════════════════════════════════

MINIMAX_KEY = json.loads((KEYS_DIR / "minimax.json").read_text())["minimax_api_key"]
MINIMAX_URL = "https://api.minimax.io/v1/text/chatcompletion_v2"
MINIMAX_MODEL = "MiniMax-Text-01"

# ══════════════════════════════════════════════════════════════════════
# LLM CALL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def call_glm_5_coding(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """
    Llama a GLM-5 via Z.ai Coding Plan (PRIMARY).
    Endpoint: https://api.z.ai/api/coding/paas/v4/chat/completions
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ZAI_CODING_KEY}"
    }

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    data = {
        "model": ZAI_CODING_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    try:
        response = requests.post(ZAI_CODING_BASE_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()

        # Parsear respuesta
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0]["message"]

            # Try content first, then reasoning_content (GLM-4.7 feature)
            content = message.get("content", "").strip()
            if not content:
                content = message.get("reasoning_content", "").strip()

            return content if content else str(result)
        else:
            return str(result)
    except Exception as e:
        print(f"    ⚠️ GLM-4.7 Coding error: {e}")
        return None


def call_minimax_llm(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """Llama a MiniMax M2.5 (FALLBACK)."""
    headers = {
        "Authorization": f"Bearer {MINIMAX_KEY}",
        "Content-Type": "application/json",
    }

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MINIMAX_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.85,
        "top_p": 0.95,
    }

    try:
        r = requests.post(MINIMAX_URL, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()

        # Check for API errors (e.g., insufficient balance)
        if "choices" not in data or data["choices"] is None:
            # Check base_resp for error details
            if "base_resp" in data and data["base_resp"].get("status_msg"):
                print(f"    ⚠️ MiniMax error: {data['base_resp']['status_msg']}")
            else:
                print(f"    ⚠️ MiniMax error: No choices in response")
            return None

        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"    ⚠️ MiniMax error: {e}")
        return None


def call_llm(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """
    Llama al LLM con fallback:
    1. GLM-5 Coding (PRIMARY)
    2. MiniMax M2.5 (FALLBACK)
    """
    # Intentar GLM-5 Coding primero
    result = call_glm_5_coding(prompt, system, max_tokens)
    if result:
        return result

    print("    ⚠️ GLM-5 Coding falló, intentando MiniMax fallback...")
    result = call_minimax_llm(prompt, system, max_tokens)
    if result:
        return result

    print("    ❌ Todos los LLM fallaron")
    return None


# ══════════════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test
    print("🧪 Test: GLM-5 Coding Plan...")
    result = call_glm_5_coding("Hola, preséntate en una frase.", "Eres un asistente útil.")
    if result:
        print(f"✅ GLM-5: {result[:100]}")
    else:
        print("❌ GLM-5 falló")
