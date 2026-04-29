#!/usr/bin/env python3
"""
BitTrader Agents — Shared LLM Configuration
Kimi Code (PRIMARY via subscription) + Claude Haiku + MiniMax + GLM-4.7 fallback
Updated 24 Apr 2026: Switched to Kimi Code Platform Anthropic-compatible endpoint.
  Endpoint: https://api.kimi.com/coding/v1/messages
  Model:    kimi-for-coding
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
# KIMI CODE (PRIMARY — subscription plan, 24 Apr 2026)
# ══════════════════════════════════════════════════════════════════════

KIMI_API_KEY = os.environ.get(
    "KIMI_API_KEY",
    "sk-kimi-UGpyqyvYvo5xqL5Oh9blHOAsfvSMSepDuN9X4qC6gZHx4qr162qctXhv94l8ueFe"
)
KIMI_BASE_URL = "https://api.kimi.com/coding/v1/messages"
KIMI_MODEL = "kimi-for-coding"

def call_kimi(prompt: str, system: str = "", model: str = None, max_tokens: int = 2000, temperature: float = 0.7) -> str:
    """
    Llama a Kimi Code via API Anthropic-compatible de suscripcion (PRIMARY).
    Endpoint: https://api.kimi.com/coding/v1/messages
    Modelo:   kimi-for-coding (siempre)
    """
    headers = {
        "Content-Type": "application/json",
        "x-api-key": KIMI_API_KEY,
        "anthropic-version": "2023-06-01",
    }

    messages = []
    if system:
        messages.append({"role": "user", "content": f"{system}\n\n{prompt}"})
    else:
        messages.append({"role": "user", "content": prompt})

    data = {
        "model": KIMI_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(KIMI_BASE_URL, headers=headers, json=data, timeout=120)
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
        print(f"    ⚠️ Kimi Code error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# CLAUDE SONNET 4.6 (FALLBACK 1)
# ══════════════════════════════════════════════════════════════════════

CLAUDE_BASE_URL = "https://api.anthropic.com/v1/messages"
def _get_claude_oauth_token():
    """Reads the current Claude OAuth token from credentials file (refreshed every 5min)."""
    try:
        import json as _json
        creds = _json.load(open("/home/enderj/.claude/.credentials.json"))
        return creds["claudeAiOauth"]["accessToken"]
    except Exception:
        return None

CLAUDE_API_KEY = _get_claude_oauth_token()
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

def call_claude_sonnet(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """Llama a Claude Haiku 4.5 via API directa de Anthropic (FALLBACK 1)."""
    import requests
    _api_key = _get_claude_oauth_token() or CLAUDE_API_KEY
    headers = {
        "Content-Type": "application/json",
        "x-api-key": _api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "oauth-2025-04-20"
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
        print(f"    ⚠️ Claude Haiku error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# ZAI GLM-4.7 CODING PLAN (FALLBACK 4)
# ══════════════════════════════════════════════════════════════════════

ZAI_CODING_KEY = "863f222d3de340df8b6ff7e1e36ab216.DFOH1veovvWzaoFT"
ZAI_CODING_BASE_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
ZAI_CODING_MODEL = "glm-4.7"

# ══════════════════════════════════════════════════════════════════════
# MINIMAX M2.5 CODING PLAN (FALLBACK 3)
# ══════════════════════════════════════════════════════════════════════

MINIMAX_KEY = json.loads((KEYS_DIR / "minimax.json").read_text())["minimax_coding_key"]
MINIMAX_TOKEN_PLAN_KEY = json.loads((KEYS_DIR / "minimax.json").read_text())["minimax_token_plan_key"]
MINIMAX_URL = "https://api.minimax.io/anthropic/v1/messages"
MINIMAX_MODEL = "MiniMax-M2.5"
MINIMAX_MODEL_27 = "MiniMax-M2.7"

# ══════════════════════════════════════════════════════════════════════
# LLM CALL FUNCTIONS
# ════════════════════════════════════════════════════════════════════════

def call_glm_5_coding(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
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
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0]["message"]
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
        if result.get("type") == "error":
            err_msg = result.get("error", {}).get("message", "Unknown")
            print(f"    ⚠️ MiniMax M2.5 error: {err_msg}")
            return None
        if "content" in result and len(result["content"]) > 0:
            return result["content"][0].get("text", "").strip()
        else:
            print("    ⚠️ MiniMax M2.5: No content in response")
            return None
    except Exception as e:
        print(f"    ⚠️ MiniMax M2.5 error: {e}")
        return None


def call_minimax_m2_7(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
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
            err_msg = result.get("error", {}).get("message", "Unknown")
            print(f"    ⚠️ MiniMax M2.7 error: {err_msg}")
            return None
        if "content" in result and len(result["content"]) > 0:
            for block in result["content"]:
                if block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text:
                        return text
            for block in result["content"]:
                if block.get("type") == "thinking":
                    text = block.get("thinking", "").strip()
                    if text:
                        return text
            print("    ⚠️ MiniMax M2.7: No usable content in response")
            return None
        else:
            print("    ⚠️ MiniMax M2.7: No content in response")
            return None
    except Exception as e:
        print(f"    ⚠️ MiniMax M2.7 error: {e}")
        return None


def call_llm(prompt: str, system: str = "", max_tokens: int = 2000, model: str = None) -> str:
    """
    Llama al LLM con fallback (actualizado 24 Apr 2026):
    1. Kimi Code (PRIMARY — subscription plan)
    2. Claude Haiku 4.5 (FALLBACK 1)
    3. MiniMax M2.7 Token Plan (FALLBACK 2)
    4. MiniMax M2.5 Coding Plan (FALLBACK 3)
    5. GLM-4.7 Coding (LAST RESORT)
    """
    result = call_kimi(prompt, system, max_tokens=max_tokens)
    if result:
        return result

    print("    ⚠️ Kimi Code fallo, intentando Claude Haiku...")
    result = call_claude_sonnet(prompt, system, max_tokens)
    if result:
        return result

    print("    ⚠️ Claude Haiku fallo, intentando MiniMax M2.7...")
    result = call_minimax_m2_7(prompt, system, max_tokens)
    if result:
        return result

    print("    ⚠️ MiniMax M2.7 fallo, intentando MiniMax M2.5...")
    result = call_minimax_llm(prompt, system, max_tokens)
    if result:
        return result

    print("    ⚠️ MiniMax M2.5 fallo, intentando GLM-4.7...")
    result = call_glm_5_coding(prompt, system, max_tokens)
    if result:
        return result

    print("    ❌ Todos los LLM fallaron")
    return None


# ══════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧪 Test: Kimi Code PRIMARY...")
    result = call_kimi("Hola, preséntate en una frase.", "Eres un asistente util.")
    if result:
        print(f"✅ Kimi Code: {result[:100]}")
    else:
        print("❌ Kimi Code fallo")

    print("\n🧪 Test: Claude fallback...")
    result = call_claude_sonnet("Hola, responde solo con la palabra EXITO", max_tokens=10)
    if result:
        print(f"✅ Claude: {result}")
    else:
        print("❌ Claude fallo")
