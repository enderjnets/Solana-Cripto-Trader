"""
ai_strategy.py — AI-Generated Trading Signals via LLM
Generates signals using MiniMax when market conditions are extreme.
"""
import json
import logging
import os
import time
from pathlib import Path
from datetime import datetime, timezone

log = logging.getLogger("ai_strategy")

DATA_DIR = Path(__file__).parent / "data"

MINIMAX_API_KEY = os.environ.get(
    "MINIMAX_API_KEY",
    "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJPcGVuQ2xhdXYiLCJhbGciOiJIUzUxMiIsImlhdCI6MTc0MDYyMTYwMH0."
    "L macronChxpRUTLCgS2IA3Z2R5aZ2g2xqnqQpVBDlxRaV1Rjg"
)


def generate_signals_with_llm(market: dict, research: dict, portfolio: dict) -> dict:
    """
    Generate trading signals using MiniMax LLM.
    Called only in extreme conditions (F&G < 25 or > 75).
    Returns {"signals": [...]} or list.
    """
    try:
        fear_greed = market.get("fear_greed", {})
        fg_val = fear_greed.get("value", 50) if isinstance(fear_greed, dict) else 50
        
        tokens = market.get("tokens", {})
        top_tokens = list(tokens.items())[:10]
        
        # Build token summary
        token_summary = "\n".join([
            f"- {sym}: ${d.get('price', 0):.4f} | "
            f"5m: {d.get('price_5min_change_pct', 0):.2f}% | "
            f"24h: {d.get('price_24h_change_pct', 0):.2f}%"
            for sym, d in top_tokens
        ])
        
        # Build portfolio summary
        positions = portfolio.get("positions", [])
        pos_summary = "\n".join([
            f"- {p.get('symbol')} {p.get('direction')}: entry=${p.get('entry_price', 0):.4f}"
            for p in positions
        ]) if positions else "No open positions"
        
        capital = portfolio.get("capital_usd", 0)
        
        # System prompt
        system_prompt = (
            "You are a Solana trading expert. Analyze the market data and generate "
            "1-3 trading signals. Return ONLY a JSON object like: "
            '{"signals": [{"symbol": "SOL", "direction": "long", "confidence": 0.75, '
            '"reasoning": "brief reason"}]}'
        )
        
        user_prompt = (
            f"Fear & Greed Index: {fg_val}/100\n"
            f"Market condition: {'EXTREME FEAR' if fg_val < 25 else 'EXTREME GREED' if fg_val > 75 else 'NEUTRAL'}\n"
            f"\nToken Data:\n{token_summary}\n"
            f"\nCurrent Portfolio (${capital:.2f}):\n{pos_summary}\n"
            f"\nGenerate 1-3 signals for tokens with strongest momentum. "
            f"Use CONTRARIAN logic: in fear, look for longs on oversold. "
            f"In greed, look for shorts on overbought. "
            f"Confidence must be >= 0.70 to include a signal."
        )
        
        # Call MiniMax API
        import urllib.request
        import urllib.error
        
        url = "https://api.minimax.chat/v1/text/chatcompletion_pro"
        headers = {
            "Authorization": f"Bearer {MINIMAX_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "MiniMax-Text-01",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 800,
            "temperature": 0.3
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
        
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Parse JSON from response
        # Find JSON block
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = content[json_start:json_end]
            signals_data = json.loads(json_str)
        else:
            signals_data = {"signals": []}
        
        log.info(f"AI generated {len(signals_data.get('signals', []))} signals")
        return signals_data
    
    except Exception as e:
        log.error(f"ai_strategy error: {e}")
        return {"signals": []}
