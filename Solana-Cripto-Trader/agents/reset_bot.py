#!/usr/bin/env python3
"""
🔄 Reset Bot - Resetea todo el estado del bot a valores iniciales
Uso: python3 reset_bot.py [--capital 500]
"""

import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

def reset_all(capital: float = 500.0):
    """Resetea todos los archivos de estado del bot."""
    now = datetime.now(timezone.utc).isoformat()
    
    results = []
    
    # 1. Portfolio
    portfolio = {
        "capital_usd": capital,
        "initial_capital": capital,
        "positions": [],
        "status": "ACTIVE",
        "mode": "paper_drift",
        "created_at": now,
        "last_updated": now,
        "total_trades": 0,
        "wins": 0,
        "losses": 0
    }
    (DATA_DIR / "portfolio.json").write_text(json.dumps(portfolio, indent=2))
    results.append("✅ portfolio.json")
    
    # 2. Trade History
    (DATA_DIR / "trade_history.json").write_text("[]")
    results.append("✅ trade_history.json")
    
    # 3. Daily Target State
    daily_state = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "starting_capital": capital,
        "target_reached": False,
        "closed_at": None,
        "enabled": True,
        "current_pnl_pct": 0.0,
        "target_pct": 0.05
    }
    (DATA_DIR / "daily_target_state.json").write_text(json.dumps(daily_state, indent=2))
    results.append("✅ daily_target_state.json")
    
    # 4. Daily Report
    daily_report = {
        "timestamp": now,
        "capital_usd": capital,
        "equity": capital,
        "pnl_total_usd": 0.0,
        "pnl_total_pct": 0.0,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "drawdown_pct": 0.0,
        "positions_open": 0
    }
    (DATA_DIR / "daily_report.json").write_text(json.dumps(daily_report, indent=2))
    results.append("✅ daily_report.json")
    
    # 5. Compound State
    compound = {
        "enabled": True,
        "last_compound": None,
        "compound_count": 0,
        "total_compounded": 0.0
    }
    (DATA_DIR / "compound_state.json").write_text(json.dumps(compound, indent=2))
    results.append("✅ compound_state.json")
    
    # 6. Auto Learner State
    learner = {
        "params": {
            "sl_pct": 0.025,
            "tp_pct": 0.05,
            "leverage_tier": 2,
            "risk_per_trade": 0.02,
            "max_positions": 5
        },
        "total_trades_learned": 0,
        "last_adaptation": None,
        "adaptation_count": 0,
        "last_updated": now,
        "notes": f"Fresh reset - capital ${capital}"
    }
    (DATA_DIR / "auto_learner_state.json").write_text(json.dumps(learner, indent=2))
    results.append("✅ auto_learner_state.json")
    
    # 7. Alerts State
    alerts = {
        "last_daily_report": "",
        "alerted_trades": [],
        "reset_at": now
    }
    (DATA_DIR / "alerts_state.json").write_text(json.dumps(alerts, indent=2))
    results.append("✅ alerts_state.json")
    
    return {
        "success": True,
        "capital": capital,
        "files_reset": results,
        "timestamp": now
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset bot state")
    parser.add_argument("--capital", type=float, default=500.0, help="Initial capital (default: 500)")
    args = parser.parse_args()
    
    print(f"\n🔄 Reseteando bot con capital ${args.capital}...\n")
    result = reset_all(args.capital)
    
    for f in result["files_reset"]:
        print(f"   {f}")
    
    print(f"\n🚀 Bot reseteado exitosamente!")
    print(f"   Capital: ${result['capital']}")
    print(f"   Timestamp: {result['timestamp']}")
