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
# CLAUDE SONNET 4.6 (PRIMARY for MrBeast Agent)
# ══════════════════════════════════════════════════════════════════════

CLAUDE_BASE_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_API_KEY = "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"
CLAUDE_MODEL = "claude-sonnet-4-6"

def call_claude_sonnet(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """
    Llama a Claude Sonnet 4.6 vía API directa de Anthropic (PRIMARY).
    Endpoint: https://api.anthropic.com/v1/messages
    """
    import requests
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01"
    }
    
    messages = []
    if system:
        messages.append({"role": "user", "content": f"{system}\n\n{prompt}"})
    else:
        messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "messages": messages
    }
    
    try:
        response = requests.post(CLAUDE_BASE_URL, json=data, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if "content" in result:
            for block in result["content"]:
                if block.get("type") == "text":
                    return block["text"]
            return result.get("content", [{}])[0].get("text", "")
        else:
            return str(result)
    except Exception as e:
        print(f"    ⚠️ Claude Sonnet error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# ZAI GLM-4.7 CODING PLAN (PRIMARY)
# ══════════════════════════════════════════════════════════════════════

ZAI_CODING_KEY = "863f222d3de340df8b6ff7e1e36ab216.DFOH1veovvWzaoFT"
ZAI_CODING_BASE_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
ZAI_CODING_MODEL = "glm-4.7"  # Cambiado a GLM-4.7 por API rate limit (12 mar 2026)

# ══════════════════════════════════════════════════════════════════════
# MINIMAX M2.5 CODING PLAN (FALLBACK - Anthropic-compatible API)
# ══════════════════════════════════════════════════════════════════════

# Usar Coding Key (sk-cp-*) con endpoint Anthropic-compatible
MINIMAX_KEY = json.loads((KEYS_DIR / "minimax.json").read_text())["minimax_coding_key"]
MINIMAX_TOKEN_PLAN_KEY = json.loads((KEYS_DIR / "minimax.json").read_text())["minimax_token_plan_key"]
MINIMAX_URL = "https://api.minimax.io/anthropic/v1/messages"
MINIMAX_MODEL = "MiniMax-M2.5"
MINIMAX_MODEL_27 = "MiniMax-M2.7"

# ══════════════════════════════════════════════════════════════════════
# LLM CALL FUNCTIONS
# ════════════════════════════════════════════════════════════════════════

def call_glm_5_coding(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """
    Llama a GLM-4.7 via Z.ai Coding Plan (PRIMARY).
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
    """Llama a MiniMax M2.5 via Anthropic-compatible API (FALLBACK 2)."""
    headers = {
        "Authorization": f"Bearer {MINIMAX_KEY}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    messages = []
    if system:
        messages.append({"role": "user", "content": f"{system}\n\n{prompt}"})
    else:
        messages.append({"role": "user", "content": prompt})

    data = {
        "model": MINIMAX_MODEL,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    try:
        r = requests.post(MINIMAX_URL, headers=headers, json=data, timeout=60)
        r.raise_for_status()
        result = r.json()

        # Check for API errors
        if result.get("type") == "error":
            print(f"    ⚠️ MiniMax M2.5 error: {result.get('error', {}).get('message', 'Unknown')}")
            return None

        # Parse response
        if "content" in result and len(result["content"]) > 0:
            return result["content"][0].get("text", "").strip()
        else:
            print(f"    ⚠️ MiniMax M2.5: No content in response")
            return None

    except Exception as e:
        print(f"    ⚠️ MiniMax M2.5 error: {e}")
        return None


def call_minimax_m2_7(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """Llama a MiniMax M2.7 via Token Plan API (FALLBACK 1)."""
    headers = {
        "Authorization": f"Bearer {MINIMAX_TOKEN_PLAN_KEY}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    messages = []
    if system:
        messages.append({"role": "user", "content": f"{system}\n\n{prompt}"})
    else:
        messages.append({"role": "user", "content": prompt})

    data = {
        "model": MINIMAX_MODEL_27,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    try:
        r = requests.post(MINIMAX_URL, headers=headers, json=data, timeout=60)
        r.raise_for_status()
        result = r.json()

        if result.get("type") == "error":
            print(f"    ⚠️ MiniMax M2.7 error: {result.get('error', {}).get('message', 'Unknown')}")
            return None

        if "content" in result and len(result["content"]) > 0:
            # MiniMax M2.7 may return "thinking" blocks — extract "text" blocks first
            for block in result["content"]:
                if block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text:
                        return text
            # If only thinking blocks, extract thinking content as fallback
            for block in result["content"]:
                if block.get("type") == "thinking":
                    text = block.get("thinking", "").strip()
                    if text:
                        return text
            print(f"    ⚠️ MiniMax M2.7: No usable content in response")
            return None
        else:
            print(f"    ⚠️ MiniMax M2.7: No content in response")
            return None

    except Exception as e:
        print(f"    ⚠️ MiniMax M2.7 error: {e}")
        return None


def call_llm(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """
    Llama al LLM con fallback:
    1. GLM-4.7 Coding (PRIMARY — fast, reliable, free)
    2. MiniMax M2.7 Token Plan (FALLBACK 1)
    3. MiniMax M2.5 Coding Plan (FALLBACK 2)
    4. Claude Sonnet 4.6 (FALLBACK 3 — OAuth key, may fail)
    
    NOTE: Claude moved to last position because the OAuth key (sk-ant-oat01)
    requires special auth that doesn't work with direct API calls.
    GLM-4.7 is the most reliable option for standalone scripts.
    """
    # Intentar GLM-4.7 primero (más confiable para scripts)
    result = call_glm_5_coding(prompt, system, max_tokens)
    if result:
        return result

    print("    ⚠️ GLM-4.7 falló, intentando MiniMax M2.7 Token Plan...")
    result = call_minimax_m2_7(prompt, system, max_tokens)
    if result:
        return result

    print("    ⚠️ MiniMax M2.7 falló, intentando MiniMax M2.5...")
    result = call_minimax_llm(prompt, system, max_tokens)
    if result:
        return result

    print("    ⚠️ MiniMax M2.5 falló, intentando Claude Sonnet (puede fallar con OAuth key)...")
    result = call_claude_sonnet(prompt, system, max_tokens)
    if result:
        return result

    print("    ❌ Todos los LLM fallaron")
    return None


# ══════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test
    print("🧪 Test: GLM-4.7 Coding Plan...")
    result = call_glm_5_coding("Hola, preséntate en una frase.", "Eres un asistente útil.")
    if result:
        print(f"✅ GLM-4.7: {result[:100]}")
    else:
        print("❌ GLM-4.7 falló")

    print("\n🧪 Test: MiniMax Coding Plan fallback...")
    result = call_minimax_llm("Hola, responde solo con la palabra EXITO", max_tokens=10)
    if result:
        print(f"✅ MiniMax: {result}")
    else:
        print("❌ MiniMax falló")
