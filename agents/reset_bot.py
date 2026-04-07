#!/usr/bin/env python3
"""
🔄 Reset Bot - Resetea todo el estado del bot a valores iniciales
Uso: python3 reset_bot.py [--capital 1000]
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Add parent directory to path for imports
sys.path.insert(0, str(BASE_DIR))

def _load_json(filepath, default=None):
    """Load JSON file, return default if doesn't exist or error."""
    try:
        if filepath.exists():
            return json.loads(filepath.read_text())
    except (json.JSONDecodeError, OSError):
        pass
    return default if default is not None else {}


def _get_trade_stats(trade_history):
    """Calculate trade statistics from trade history."""
    if not trade_history or not isinstance(trade_history, list):
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "flat": 0,
            "win_rate": 0.0,
            "total_pnl_usd": 0.0,
            "best_trade_usd": 0.0,
            "worst_trade_usd": 0.0
        }

    total = len(trade_history)
    wins = sum(1 for t in trade_history if t.get("pnl_usd", 0) > 0)
    losses = sum(1 for t in trade_history if t.get("pnl_usd", 0) < 0)
    flat = sum(1 for t in trade_history if t.get("pnl_usd", 0) == 0)
    win_rate = (wins / total * 100) if total > 0 else 0.0

    pnls = [t.get("pnl_usd", 0) for t in trade_history]
    total_pnl = sum(pnls)
    best_trade = max(pnls) if pnls else 0.0
    worst_trade = min(pnls) if pnls else 0.0

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "flat": flat,
        "win_rate": round(win_rate, 1),
        "total_pnl_usd": round(total_pnl, 2),
        "best_trade_usd": round(best_trade, 4),
        "worst_trade_usd": round(worst_trade, 4)
    }


def reset_all(capital: float = 1000.0, reason: str = "Manual reset"):
    """Resetea todos los archivos de estado del bot."""
    now = datetime.now(timezone.utc).isoformat()

    results = []

    # 0. Load existing state BEFORE reset for logging
    portfolio = _load_json(DATA_DIR / "portfolio.json", {"capital_usd": capital})
    old_capital = portfolio.get("capital_usd", capital)

    trade_history = _load_json(DATA_DIR / "trade_history.json", [])
    trade_stats = _get_trade_stats(trade_history)

    # Load previous reset info for period_start calculation
    reset_history = _load_json(DATA_DIR / "reset_history.json", [])
    period_start = None
    if reset_history and isinstance(reset_history, list) and len(reset_history) > 0:
        last_reset = reset_history[-1]
        period_start = last_reset.get("reset_date")

    # 1. Portfolio
    new_portfolio = {
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
    (DATA_DIR / "portfolio.json").write_text(json.dumps(new_portfolio, indent=2))
    results.append("✅ portfolio.json")

    # 2. Trade History - save to backup BEFORE clearing
    if trade_history:
        backup_name = f"trade_history_pre_reset_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        (DATA_DIR / backup_name).write_text(json.dumps(trade_history, indent=2))
        results.append(f"✅ trade_history.json backup ({backup_name})")
    (DATA_DIR / "trade_history.json").write_text("[]")
    results.append("✅ trade_history.json (cleared)")
    
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
    
    # 6. Auto Learner State — PRESERVAR conocimiento aprendido
    learner_file = DATA_DIR / "auto_learner_state.json"
    if learner_file.exists():
        # Mantener parámetros aprendidos, solo actualizar timestamp
        existing_learner = json.loads(learner_file.read_text())
        existing_learner["last_updated"] = now
        existing_learner["notes"] = f"Reset capital ${capital} - conocimiento preservado"
        learner_file.write_text(json.dumps(existing_learner, indent=2))
        results.append("✅ auto_learner_state.json (PRESERVADO)")
    else:
        # Solo crear nuevo si no existe
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
            "notes": f"Fresh start - capital ${capital}"
        }
        learner_file.write_text(json.dumps(learner, indent=2))
        results.append("✅ auto_learner_state.json (nuevo)")
    
    # 7. Alerts State
    alerts = {
        "last_daily_report": "",
        "alerted_trades": [],
        "reset_at": now
    }
    (DATA_DIR / "alerts_state.json").write_text(json.dumps(alerts, indent=2))
    results.append("✅ alerts_state.json")

    # 8. Add to reset_history.json with COMPLETE data
    # Use ACTUAL portfolio capital (not calculated from trades, which may be incomplete)
    actual_capital = portfolio.get("capital_usd", old_capital)
    actual_pnl = actual_capital - old_capital
    # If trade history has data, use its PnL; otherwise use portfolio-based PnL
    pnl_usd = trade_stats["total_pnl_usd"] if trade_stats["total_trades"] > 0 else round(actual_pnl, 2)
    reset_entry = {
        "reset_date": now,
        "period_start": period_start,
        "period_end": now,
        "initial_capital": old_capital,
        "final_capital": round(actual_capital, 2),
        "return_pct": round((actual_pnl / old_capital * 100) if old_capital > 0 else 0, 2),
        "total_pnl_usd": pnl_usd,
        "total_trades": trade_stats["total_trades"],
        "wins": trade_stats["wins"],
        "losses": trade_stats["losses"],
        "flat": trade_stats["flat"],
        "win_rate": trade_stats["win_rate"],
        "best_trade_usd": trade_stats["best_trade_usd"],
        "worst_trade_usd": trade_stats["worst_trade_usd"],
        "new_capital": capital,
        "reason": reason
    }

    reset_history = _load_json(DATA_DIR / "reset_history.json", [])
    if not isinstance(reset_history, list):
        reset_history = []
    reset_history.append(reset_entry)
    (DATA_DIR / "reset_history.json").write_text(json.dumps(reset_history, indent=2))
    results.append("✅ reset_history.json (updated)")

    # 9. Call Paperclip to log the reset (if available)
    paperclip_issue_id = None
    try:
        from paperclip_client import on_reset
        paperclip_issue_id = on_reset(old_capital, capital, reason)
        if paperclip_issue_id:
            results.append(f"✅ Paperclip reset issue: {paperclip_issue_id}")
        else:
            results.append("⚠️ Paperclip reset issue not created (service unavailable)")
    except Exception as e:
        results.append(f"⚠️ Paperclip not available: {e}")

    return {
        "success": True,
        "capital": capital,
        "old_capital": old_capital,
        "trade_stats": trade_stats,
        "files_reset": results,
        "timestamp": now,
        "paperclip_issue_id": paperclip_issue_id
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset bot state")
    parser.add_argument("--capital", type=float, default=1000.0, help="Initial capital (default: 1000)")
    parser.add_argument("--reason", type=str, default="Manual reset", help="Reason for reset")
    args = parser.parse_args()

    print(f"\n🔄 Reseteando bot con capital ${args.capital}...")
    print(f"📝 Razón: {args.reason}\n")

    result = reset_all(args.capital, args.reason)

    # Show trade statistics before reset
    stats = result.get("trade_stats", {})
    print("📊 Estadísticas del período anterior:")
    print(f"   Trades: {stats.get('total_trades', 0)}")
    print(f"   Wins/Losses: {stats.get('wins', 0)}/{stats.get('losses', 0)}")
    print(f"   Win Rate: {stats.get('win_rate', 0)}%")
    print(f"   PnL Total: ${stats.get('total_pnl_usd', 0):.2f}")
    print(f"   Mejor Trade: ${stats.get('best_trade_usd', 0):.4f}")
    print(f"   Peor Trade: ${stats.get('worst_trade_usd', 0):.4f}")
    print()

    print("📁 Archivos procesados:")
    for f in result["files_reset"]:
        print(f"   {f}")

    print(f"\n🚀 Bot reseteado exitosamente!")
    print(f"   Capital Anterior: ${result.get('old_capital', 0):.2f}")
    print(f"   Capital Nuevo: ${result['capital']}")
    print(f"   Timestamp: {result['timestamp']}")
    if result.get("paperclip_issue_id"):
        print(f"   Paperclip Issue: {result['paperclip_issue_id']}")
