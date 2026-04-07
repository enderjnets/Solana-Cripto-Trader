"""
Ollama Client — Shared module for Gemma 4 local inference.
Used by both Solana Cripto Trader and BitTrader YouTube projects.
All calls include retry logic for Gemma 4 cold-start empty responses.
Falls back gracefully if Ollama is unavailable.
"""
import json
import logging
import re
import time
import base64
import requests
from pathlib import Path

log = logging.getLogger("ollama")

OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma4:e2b"
TIMEOUT = 60
MAX_RETRIES = 5

_cache = {"available": None, "ts": 0}


def is_ollama_available() -> bool:
    now = time.time()
    if _cache["available"] is not None and now - _cache["ts"] < 60:
        return _cache["available"]
    try:
        r = requests.get(f"{OLLAMA_URL}/api/version", timeout=3)
        ok = r.ok
    except Exception:
        ok = False
    _cache.update({"available": ok, "ts": now})
    return ok


def _chat(messages: list, max_tokens: int = 500) -> str | None:
    """Internal: send chat request with retries for cold-start empty responses."""
    if not is_ollama_available():
        return None
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": "1h",
        "think": False,
        "options": {"num_predict": max_tokens, "temperature": 0.7},
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=TIMEOUT)
            if r.ok:
                text = r.json().get("message", {}).get("content", "").strip()
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
                if text:
                    return text
                time.sleep(1)  # Model cold, retry
            else:
                return None
        except requests.Timeout:
            log.warning("Ollama timeout")
            return None
        except Exception as e:
            log.debug(f"Ollama error: {e}")
            return None
    return None


def query_gemma4(prompt: str, system: str = "", max_tokens: int = 500) -> str | None:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return _chat(messages, max_tokens)


def query_gemma4_vision(image_path: str, prompt: str, system: str = "") -> str | None:
    if not is_ollama_available():
        return None
    p = Path(image_path)
    if not p.exists():
        return None
    img_b64 = base64.b64encode(p.read_bytes()).decode()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt, "images": [img_b64]})
    return _chat(messages, 500)


def validate_trading_signal(signal: dict, market_context: str) -> dict:
    prompt = f"""Evaluate this trading signal:
Signal: {signal.get('direction','?').upper()} {signal.get('symbol','?')}
Strategy: {signal.get('strategy','?')} | Confidence: {signal.get('confidence',0):.0%}
Market: {market_context}

AGREE or DISAGREE? JSON only: {{"agree": true/false, "reasoning": "1 sentence", "confidence_adj": 0.10 or -0.15}}"""

    result = _chat([
        {"role": "system", "content": "Crypto analyst. JSON only."},
        {"role": "user", "content": prompt}
    ], 150)

    if not result:
        return {"agree": True, "reasoning": "Gemma4 unavailable", "confidence_adj": 0.0}
    try:
        m = re.search(r'\{[^}]+\}', result)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return {"agree": True, "reasoning": "Parse error", "confidence_adj": 0.0}


def evaluate_thumbnail(image_path: str) -> dict:
    result = query_gemma4_vision(image_path,
        'Rate this YouTube thumbnail 1-10 for click potential. JSON: {"score": 7, "ctr_potential": "high", "feedback": "reason"}',
        "YouTube thumbnail expert. JSON only.")
    if not result:
        return {"score": 5, "ctr_potential": "unknown", "feedback": "Vision unavailable"}
    try:
        m = re.search(r'\{[^}]+\}', result)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return {"score": 5, "ctr_potential": "unknown", "feedback": result[:200] if result else "Error"}


def review_script(script_text: str, topic: str = "") -> dict:
    result = _chat([
        {"role": "system", "content": "YouTube script editor. JSON only."},
        {"role": "user", "content": f'Review script (topic: {topic}). Score 1-10. JSON: {{"score": 7, "issues": ["x"], "suggestions": ["y"]}}\n\nScript:\n{script_text[:2000]}'}
    ], 300)
    if not result:
        return {"score": 5, "issues": ["Unavailable"], "suggestions": []}
    try:
        m = re.search(r'\{.*\}', result, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return {"score": 5, "issues": ["Parse error"], "suggestions": []}
