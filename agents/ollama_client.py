"""
Ollama Client — Shared module for Gemma 4 local inference.
Used by both Solana Cripto Trader and BitTrader YouTube projects.
Provides text, vision, and audio capabilities via local Ollama server.
Falls back to MiniMax API if Ollama is unavailable.
"""
import json
import logging
import base64
import requests
from pathlib import Path

log = logging.getLogger("ollama")

# ── Config ──────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma4:e4b"
TIMEOUT_GENERATE = 60  # seconds for generation
TIMEOUT_HEALTH = 3     # seconds for health check

# Cache availability check for 60 seconds
_available_cache = {"value": None, "ts": 0}


def is_ollama_available() -> bool:
    """Check if Ollama server is running and model is loaded."""
    import time
    now = time.time()
    if _available_cache["value"] is not None and now - _available_cache["ts"] < 60:
        return _available_cache["value"]
    try:
        r = requests.get(f"{OLLAMA_URL}/api/version", timeout=TIMEOUT_HEALTH)
        available = r.ok
    except Exception:
        available = False
    _available_cache["value"] = available
    _available_cache["ts"] = now
    return available


def query_gemma4(prompt: str, system: str = "", max_tokens: int = 500) -> str | None:
    """
    Query Gemma 4 via Ollama. Returns response text or None if unavailable.

    Args:
        prompt: User prompt
        system: System prompt (optional)
        max_tokens: Max tokens in response
    Returns:
        Response text or None
    """
    if not is_ollama_available():
        return None
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                },
            },
            timeout=TIMEOUT_GENERATE,
        )
        if r.ok:
            data = r.json()
            text = data.get("message", {}).get("content", "").strip()
            # Strip thinking blocks if present
            import re
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            if text.startswith("Thinking"):
                # Remove "Thinking..." prefix lines
                lines = text.split("\n")
                text = "\n".join(l for l in lines if not l.startswith("Thinking"))
            return text if text else None
        else:
            log.warning(f"Ollama error: {r.status_code}")
            return None
    except requests.Timeout:
        log.warning("Ollama timeout")
        return None
    except Exception as e:
        log.debug(f"Ollama error: {e}")
        return None


def query_gemma4_vision(image_path: str, prompt: str, system: str = "") -> str | None:
    """
    Query Gemma 4 with an image (vision capability).

    Args:
        image_path: Path to image file (PNG, JPG)
        prompt: What to analyze about the image
        system: System prompt (optional)
    Returns:
        Response text or None
    """
    if not is_ollama_available():
        return None
    try:
        img_path = Path(image_path)
        if not img_path.exists():
            log.warning(f"Image not found: {image_path}")
            return None

        img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": prompt,
            "images": [img_b64],
        })

        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "options": {"num_predict": 500, "temperature": 0.5},
            },
            timeout=TIMEOUT_GENERATE * 2,  # Vision takes longer
        )
        if r.ok:
            text = r.json().get("message", {}).get("content", "").strip()
            import re
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            return text if text else None
        return None
    except Exception as e:
        log.debug(f"Ollama vision error: {e}")
        return None


def validate_trading_signal(signal: dict, market_context: str) -> dict:
    """
    Use Gemma 4 as second opinion on a trading signal.

    Args:
        signal: Trading signal dict with symbol, direction, confidence, strategy
        market_context: Brief market context string
    Returns:
        {"agree": bool, "reasoning": str, "confidence_adj": float}
    """
    prompt = f"""You are a crypto trading analyst. Evaluate this trading signal:

Signal: {signal.get('direction', '?').upper()} {signal.get('symbol', '?')}
Strategy: {signal.get('strategy', '?')}
Confidence: {signal.get('confidence', 0):.0%}
Entry: ${signal.get('entry_price', 0):.8f}

Market context: {market_context}

Do you AGREE or DISAGREE with this trade? Answer in this exact JSON format:
{{"agree": true/false, "reasoning": "1 sentence why", "confidence_adj": 0.10 or -0.15}}

If you agree, confidence_adj should be +0.10. If you disagree, -0.15.
Respond ONLY with JSON, no other text."""

    result = query_gemma4(prompt, system="You are a concise crypto analyst. Respond only in JSON.", max_tokens=150)
    if not result:
        return {"agree": True, "reasoning": "Ollama unavailable", "confidence_adj": 0.0}

    try:
        # Extract JSON from response
        import re
        json_match = re.search(r'\{[^}]+\}', result)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass

    return {"agree": True, "reasoning": "Could not parse response", "confidence_adj": 0.0}


def evaluate_thumbnail(image_path: str) -> dict:
    """
    Use Gemma 4 vision to evaluate a YouTube thumbnail.

    Args:
        image_path: Path to thumbnail image
    Returns:
        {"score": 1-10, "ctr_potential": "low/medium/high", "feedback": str}
    """
    prompt = """Evaluate this YouTube thumbnail for a crypto trading channel.
Rate it on a scale of 1-10 for click-through potential.

Consider: visual impact, text readability, emotional hook, color contrast, curiosity gap.

Respond in this exact JSON format:
{"score": 7, "ctr_potential": "high", "feedback": "Strong contrast and clear text. Could improve emotional hook."}

Respond ONLY with JSON."""

    result = query_gemma4_vision(
        image_path, prompt,
        system="You are a YouTube thumbnail expert. Rate thumbnails for CTR potential. Respond only in JSON."
    )
    if not result:
        return {"score": 5, "ctr_potential": "unknown", "feedback": "Vision unavailable"}

    try:
        import re
        json_match = re.search(r'\{[^}]+\}', result)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass

    return {"score": 5, "ctr_potential": "unknown", "feedback": result[:200] if result else "Parse error"}


def review_script(script_text: str, topic: str = "") -> dict:
    """
    Use Gemma 4 to review a video script quality.

    Args:
        script_text: The script to review
        topic: Video topic for context
    Returns:
        {"score": 1-10, "issues": [str], "suggestions": [str]}
    """
    prompt = f"""Review this YouTube video script about crypto trading.
Topic: {topic or 'crypto'}

Script:
{script_text[:2000]}

Rate quality 1-10. List issues and suggestions.
Respond in JSON: {{"score": 7, "issues": ["issue1"], "suggestions": ["suggestion1"]}}
Respond ONLY with JSON."""

    result = query_gemma4(prompt, system="You are a YouTube content editor. Rate scripts for engagement.", max_tokens=300)
    if not result:
        return {"score": 5, "issues": ["Ollama unavailable"], "suggestions": []}

    try:
        import re
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass

    return {"score": 5, "issues": ["Parse error"], "suggestions": [result[:200] if result else ""]}
