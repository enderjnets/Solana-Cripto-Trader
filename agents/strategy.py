"""
strategy.py — Technical Strategy Signals
Wraps strategy_agent and adds technical analysis signal generation.
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

log = logging.getLogger("strategy")

DATA_DIR = Path(__file__).parent / "data"


def run(debug=False) -> dict:
    """
    Generate technical trading signals from market data.
    Returns dict with total_signals and signals list.
    """
    try:
        market_file = DATA_DIR / "market_latest.json"
        if not market_file.exists():
            log.warning("market_latest.json not found — no signals")
            return {"total_signals": 0, "signals": []}
        
        market_data = json.loads(market_file.read_text())
        tokens = market_data.get("tokens", {})
        
        signals = []
        for symbol, data in tokens.items():
            # Skip if no momentum
            if not data.get("momentum"):
                continue
            
            price = data.get("price", 0)
            change_5m = data.get("price_5min_change_pct", 0)
            change_24h = data.get("price_24h_change_pct", 0)
            trend = data.get("price_1h_trend", "unknown")
            
            # Generate signal based on momentum
            if change_5m > 0.5 and trend in ("up", "unknown"):
                direction = "long"
                confidence = min(0.95, 0.60 + abs(change_5m) / 10)
            elif change_5m < -0.5 and trend in ("down", "unknown"):
                direction = "short"
                confidence = min(0.95, 0.60 + abs(change_5m) / 10)
            else:
                continue
            
            if confidence >= 0.70:
                signals.append({
                    "symbol": symbol,
                    "direction": direction,
                    "confidence": confidence,
                    "entry_price": price,
                    "reasoning": f"Technical: {change_5m:.2f}% 5m, {change_24h:.2f}% 24h",
                    "source": "technical",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        
        # Save to strategy_signals.json
        out_file = DATA_DIR / "strategy_signals.json"
        out_file.write_text(json.dumps({
            "signals": signals,
            "total_signals": len(signals),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }, indent=2))
        
        log.info(f"Generated {len(signals)} technical signals")
        return {"total_signals": len(signals), "signals": signals}
    
    except Exception as e:
        log.error(f"Error generating signals: {e}")
        return {"total_signals": 0, "signals": []}
