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
import os as _os_llm
WORKSPACE = Path(_os_llm.environ.get("SOLANA_WORKSPACE", "/home/enderj/.openclaw/workspace"))
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

def _get_minimax_key() -> str:
    # 1. Env var — portable (cualquier instancia nueva)
    if _os_llm.environ.get("MINIMAX_API_KEY"):
        return _os_llm.environ["MINIMAX_API_KEY"]
    # 2. Archivo legacy — backward compat para ROG
    try:
        return json.loads((KEYS_DIR / "minimax.json").read_text())["minimax_api_key"]
    except Exception:
        return ""

MINIMAX_KEY   = _get_minimax_key()
MINIMAX_URL   = "https://api.minimax.io/v1/text/chatcompletion_v2"
MINIMAX_MODEL = "MiniMax-M2.7"  # Upgraded from M2.5 - better reasoning for trading decisions


# ══════════════════════════════════════════════════════════════════════
# 4to FALLBACK: Qwen 2.5 14B local via Ollama
# Sin internet, sin API key, sin coste -- siempre disponible en ROG
# ══════════════════════════════════════════════════════════════════════
QWEN_URL   = _os_llm.environ.get("QWEN_URL",   "http://localhost:11434/api/chat")
QWEN_MODEL = _os_llm.environ.get("QWEN_MODEL", "qwen2.5:14b")


def call_qwen_local(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """Qwen 2.5 14B via Ollama -- 4to fallback local. Sin internet, sin coste."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        r = requests.post(QWEN_URL, json={
            "model": QWEN_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.2}
        }, timeout=90)
        r.raise_for_status()
        text = r.json().get("message", {}).get("content", "").strip()
        if "<think>" in text and "</think>" in text:
            text = text.split("</think>", 1)[-1].strip()
        return text if text else None
    except Exception as e:
        print(f"    \u26a0\ufe0f Qwen local error: {e}")
        return None


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

# ── GPT-5.4 via OpenClaw Proxy (PRIMARY for deep reasoning) ──────────────
# AUDIT FIX: env-overridable so the proxy can be moved without code edits
import os as _os_gpt5
_DEFAULT_GPT5_TOKEN = "eyJhbGciOiJSUzI1NiIsImtpZCI6IjE5MzQ0ZTY1LWJiYzktNDRkMS1hOWQwLWY5NTdiMDc5YmQwZSIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS92MSJdLCJjbGllbnRfaWQiOiJhcHBfRU1vYW1FRVo3M2YwQ2tYYVhwN2hyYW5uIiwiZXhwIjoxNzc2NDc0MzE2LCJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiMDIxZDhmMTMtMjM1Yy00ZDY4LWJkN2MtODhlNmY4MmVlZmNmIiwiY2hhdGdwdF9hY2NvdW50X3VzZXJfaWQiOiJ1c2VyLW11U2F5ZWhqeG1zTzZNR1lRZDhld2dLaF9fMDIxZDhmMTMtMjM1Yy00ZDY4LWJkN2MtODhlNmY4MmVlZmNmIiwiY2hhdGdwdF9jb21wdXRlX3Jlc2lkZW5jeSI6Im5vX2NvbnN0cmFpbnQiLCJjaGF0Z3B0X3BsYW5fdHlwZSI6InBsdXMiLCJjaGF0Z3B0X3VzZXJfaWQiOiJ1c2VyLW11U2F5ZWhqeG1zTzZNR1lRZDhld2dLaCIsImxvY2FsaG9zdCI6dHJ1ZSwidXNlcl9pZCI6InVzZXItbXVTYXllaGp4bXNPNk1HWVFkOGV3Z0toIn0sImh0dHBzOi8vYXBpLm9wZW5haS5jb20vcHJvZmlsZSI6eyJlbWFpbCI6ImVuZGVyam5ldHNAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWV9LCJpYXQiOjE3NzU2MTAzMTYsImlzcyI6Imh0dHBzOi8vYXV0aC5vcGVuYWkuY29tIiwianRpIjoiYWYyZTcyNDAtYzk4MC00NjgwLTg3OTAtOTJhYjkwMGE4YTg5IiwibmJmIjoxNzc1NjEwMzE2LCJwd2RfYXV0aF90aW1lIjoxNzc1NjEwMzE1MzI2LCJzY3AiOlsib3BlbmlkIiwicHJvZmlsZSIsImVtYWlsIiwib2ZmbGluZV9hY2Nlc3MiLCJhcGkuY29ubmVjdG9ycy5yZWFkIiwiYXBpLmNvbm5lY3RvcnMuaW52b2tlIl0sInNlc3Npb25faWQiOiJhdXRoc2Vzc19xcHJCRUZ6b1BmR3AxUUt1Y0U4cW55WXUiLCJzbCI6dHJ1ZSwic3ViIjoiZ29vZ2xlLW9hdXRoMnwxMDM5NzkwMTQ0Njk4ODM5MDczMTMifQ.GvUdegjur8dIjmsIPIPShzmSHiZgi26GjYIHXxFozQ9KVrmwfQBeBEVND0dMEhrZAHGPiihCcaqpanQoa-zoBhtWoWqSfNJgm7a6wMIyVgbQ6nchSjMOOVCds9qoAyahYBsyKjKkM2dMcCCCHGjGRSZrjuBwei22amGssdsF2Fx9hhko2txtVk3UvDboUUA42hOR5zELiWqoGCpWmAHyC08nryET0HpKIaOj2bRaVTK_-qBhmXKlEkkG90XyGHuraD49N4xpm3_7fgvbWn-fAJFgqyR_GVurl7YKK2tobUYfAg4CSlpLhuGEPdM_WEnFibrx0tJn7EbLUa1ILi8apAKLoQF7-oeXLf63logh-dFjqCwoUumaPWQc6ZoMg47gTxvhhtewOWiB6h_sJQh8GufdHylc2DYBVSeeu_YB7vAu3wp-pvl1jUosPNnMaJcYdKYIVJp75MH3TiMHL1alKjiGQwRyTCnafLRKQj__kf0rxUn06zknE-v5uomxflaRolkTzhZlVeDr9krgqBh4srsnR212S9XMzipqk-p1KUYSbjkb5sVuFJNFE42gsKP_sgKdR0d6dvlOqQzB9aXnkGXLLSkjZlHbbNid9OJDerFhi4_X9KLvd8MCg2-UzTcM_mP3qNJkYeSNbzqiw7v4sp7NzxPPMWMqMG3MaEjEEDY"

GPT5_URL = _os_gpt5.environ.get("GPT5_URL", "http://127.0.0.1:18793/v1/chat/completions")
GPT5_TOKEN = _os_gpt5.environ.get("GPT5_TOKEN", _DEFAULT_GPT5_TOKEN)
GPT5_MODEL = _os_gpt5.environ.get("GPT5_MODEL", "gpt-5.4")

def call_gpt5(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """Call GPT-5.4 via OpenClaw proxy. Returns None if unavailable."""
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        r = requests.post(GPT5_URL, headers={
            "Authorization": f"Bearer {GPT5_TOKEN}",
            "Content-Type": "application/json"
        }, json={
            "model": GPT5_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }, timeout=30)
        if r.ok:
            text = r.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if text:
                # Codex wraps the full conversation in its response (header + user + assistant).
                # Strip everything before the last "assistant\n" section to get only the reply.
                if "\nassistant\n" in text:
                    text = text.split("\nassistant\n")[-1].strip()
                    if text:
                        return text
                    return None  # empty assistant section
                # No assistant section = Codex header/error (usage limit etc)
                errs = ("usage limit", "upgrade to pro", "hit your usage")
                if any(kw in text.lower() for kw in errs) or text.startswith("OpenAI Codex"):
                    print("    ⚠️ GPT-5.4: Codex limit detected -- fallback")
                    return None
                return text
    except Exception as e:
        print(f"GPT-5.4 error: {e}")
    return None


def call_llm(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """
    LLM chain: GPT-5.4 → MiniMax M2.7 → Claude Sonnet 4.6 (tertiary).
    GPT-5.4 via OpenClaw proxy (localhost:18793).
    MiniMax M2.7 as reliable 2nd fallback, Claude Sonnet 4.6 as last resort.
    """
    if _cb_is_open():
        return None

    # Try GPT-5.4 first (best reasoning for trading decisions)
    result = call_gpt5(prompt, system, max_tokens)
    if result:
        _cb_record_success()
        print("    🧠 LLM: GPT-5.4 (primary)")
        return result

    # Fallback to MiniMax M2.7 (always available)
    result = call_minimax(prompt, system, max_tokens)
    if result:
        _cb_record_success()
        print("    🧠 LLM: MiniMax M2.7 (fallback)")
        return result

    # Last resort: Claude Sonnet 4.6 via direct Anthropic API
    result = call_claude_sonnet(prompt, system, max_tokens)
    if result:
        _cb_record_success()
        print("    🧠 LLM: Claude Sonnet 4.6 (tertiary)")
        return result

    # 4to fallback: Qwen 2.5 14B local via Ollama
    result = call_qwen_local(prompt, system, max_tokens)
    if result:
        _cb_record_success()
        print("    \U0001f9e0 LLM: Qwen 2.5 14B local (4to fallback)")
        return result

    _cb_record_failure()
    print("    \u274c GPT-5.4 + MiniMax + Claude + Qwen all failed")
    return None


if __name__ == "__main__":
    print("🧪 Test: Claude Sonnet 4.6...")
    r = call_llm("Responde solo: OK", "Eres un bot de trading.")
    print(f"{'✅' if r else '❌'} Claude: {r or 'falló'}")
