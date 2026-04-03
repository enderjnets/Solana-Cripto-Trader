"""
daily_target.py — Daily P&L Target Tracking
Evaluates if daily profit target has been hit and closes positions.
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

log = logging.getLogger("daily_target")

DATA_DIR = Path(__file__).parent / "data"

# Daily target: 2% profit
DAILY_TARGET_PCT = 2.0


def run() -> dict:
    """Called each cycle. Returns daily target status."""
    try:
        state_file = DATA_DIR / "daily_target_state.json"
        state = {}
        if state_file.exists():
            state = json.loads(state_file.read_text())
        
        # Check if it's a new day
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        last_date = state.get("date", "")
        
        if last_date != today:
            # New day — reset
            state = {
                "date": today,
                "start_capital": 0,
                "target_hit": False,
                "trades_closed": 0
            }
            state_file.write_text(json.dumps(state, indent=2))
        
        return {"ok": True, "state": state}
    
    except Exception as e:
        log.error(f"daily_target.run error: {e}")
        return {"ok": False}


def evaluate_daily_target(portfolio: dict, signals: dict) -> dict:
    """
    Evaluate if daily target has been hit.
    Returns: {
        daily_pnl_pct: float,
        target_pct: float,
        should_close_all: bool,
        close_reason: str,
        market_conditions: dict
    }
    """
    try:
        state_file = DATA_DIR / "daily_target_state.json"
        state = {}
        if state_file.exists():
            state = json.loads(state_file.read_text())
        
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        last_date = state.get("date", "")
        
        # Initialize for new day
        if last_date != today:
            state = {
                "date": today,
                "start_capital": portfolio.get("capital_usd", 1000),
                "target_hit": False,
                "trades_closed": 0
            }
        
        start_capital = state.get("start_capital", portfolio.get("initial_capital", 1000))
        current_capital = portfolio.get("capital_usd", start_capital)
        
        daily_pnl_pct = ((current_capital - start_capital) / start_capital * 100) if start_capital > 0 else 0
        
        # Calculate market conditions
        avg_rsi = 50
        market_conditions = {
            "avg_rsi": avg_rsi,
            "target_hit": state.get("target_hit", False)
        }
        
        # Should close all if target is hit
        should_close = False
        close_reason = ""
        
        if daily_pnl_pct >= DAILY_TARGET_PCT and not state.get("target_hit"):
            should_close = True
            close_reason = f"Daily target {DAILY_TARGET_PCT}% hit ({daily_pnl_pct:.2f}%)"
            state["target_hit"] = True
            log.info(f"🎯 Daily target HIT: {daily_pnl_pct:.2f}% >= {DAILY_TARGET_PCT}%")
        elif daily_pnl_pct <= -3.0:
            # Stop loss for the day
            should_close = True
            close_reason = f"Daily stop loss triggered ({daily_pnl_pct:.2f}%)"
            log.info(f"🛡️ Daily stop loss: {daily_pnl_pct:.2f}%")
        else:
            log.info(f"🎯 Daily P&L: {daily_pnl_pct:.2f}% (target: {DAILY_TARGET_PCT}%)")
        
        # Save state
        state_file.write_text(json.dumps(state, indent=2))
        
        return {
            "daily_pnl_pct": round(daily_pnl_pct, 2),
            "target_pct": DAILY_TARGET_PCT,
            "should_close_all": should_close,
            "close_reason": close_reason,
            "market_conditions": market_conditions
        }
    
    except Exception as e:
        log.error(f"evaluate_daily_target error: {e}")
        return {
            "daily_pnl_pct": 0,
            "target_pct": DAILY_TARGET_PCT,
            "should_close_all": False,
            "close_reason": "",
            "market_conditions": {"avg_rsi": 50}
        }
