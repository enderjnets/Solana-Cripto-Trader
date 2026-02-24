#!/usr/bin/env python3
"""
Daily Trading Summary Generator
Generates and sends a daily trading summary via Telegram
"""
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import Config
from paper_trading_engine import PaperTradingEngine
from notifications import get_notifier
import sqlite3


def get_daily_stats(date: str = None) -> dict:
    """Get daily trading statistics from database"""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    db_path = PROJECT_ROOT / "data" / "trades.db"
    if not db_path.exists():
        return {
            "date": date,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0
        }

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get trades for the day
    cursor.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
               SUM(pnl) as total_pnl,
               AVG(pnl) as avg_pnl
        FROM trades
        WHERE DATE(entry_time) = ?
        AND status = 'closed'
    """, (date,))

    row = cursor.fetchone()
    conn.close()

    if row[0] == 0:
        return {
            "date": date,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0
        }

    total, wins, losses, total_pnl, avg_pnl = row
    win_rate = (wins / total * 100) if total > 0 else 0.0

    return {
        "date": date,
        "total_trades": total,
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl if total_pnl else 0.0,
        "avg_pnl": avg_pnl if avg_pnl else 0.0
    }


def generate_summary() -> dict:
    """Generate complete daily summary"""
    # Get paper trading engine state
    engine = PaperTradingEngine()
    stats = engine.state.stats

    # Get daily stats from database
    today = datetime.now().strftime("%Y-%m-%d")
    daily_stats = get_daily_stats(today)

    # Get open positions
    open_trades = engine.get_open_trades()

    # Calculate portfolio P&L
    portfolio_pnl = 0.0
    for trade in open_trades:
        if trade.get("pnl"):
            portfolio_pnl += trade["pnl"]

    summary = {
        "date": today,
        "time": datetime.now().strftime("%H:%M:%S"),

        # Balance
        "balance": engine.state.balance_usd,
        "initial_balance": engine.state.initial_balance,
        "total_pnl_percent": engine._get_pnl_pct(),

        # Daily stats
        "daily_trades": daily_stats["total_trades"],
        "daily_wins": daily_stats["winning_trades"],
        "daily_losses": daily_stats["losing_trades"],
        "daily_win_rate": daily_stats["win_rate"],
        "daily_pnl": daily_stats["total_pnl"],

        # Overall stats
        "total_trades": stats.get("total_trades", 0),
        "total_wins": stats.get("winning_trades", 0),
        "total_losses": stats.get("losing_trades", 0),
        "total_win_rate": stats.get("win_rate", 0.0),
        "total_pnl": stats.get("total_pnl", 0.0),

        # Open positions
        "open_positions": len(open_trades),
        "portfolio_pnl": portfolio_pnl,

        # Streaks
        "current_streak": stats.get("current_streak", 0),
        "best_streak": stats.get("best_streak", 0),
        "worst_streak": stats.get("worst_streak", 0),

        # Fees & Liquidations
        "total_fees": stats.get("total_fees", 0.0),
        "liquidations": stats.get("liquidations", 0)
    }

    return summary


def send_daily_summary():
    """Generate and send daily summary to Telegram"""
    try:
        summary = generate_summary()
        notifier = get_notifier()

        # Build formatted message
        pnl_emoji = "📈" if summary["daily_pnl"] > 0 else "📉"
        win_emoji = "✅" if summary["daily_win_rate"] >= 50 else "⚠️"

        message = f"""
📊 **RESUMEN DIARIO DE TRADING**

📅 Fecha: {summary['date']}
⏰ Hora: {summary['time']}

💰 **Balance**
   • Actual: ${summary['balance']:,.2f}
   • Inicial: ${summary['initial_balance']:,.2f}
   • Total P&L: {summary['total_pnl_percent']:+.2f}%

📈 **Hoy**
{pnl_emoji} • P&L: ${summary['daily_pnl']:,.2f}
📊 • Trades: {summary['daily_trades']} ({summary['daily_wins']}W / {summary['daily_losses']}L)
{win_emoji} • Win rate: {summary['daily_win_rate']:.1f}%

📊 **Global**
💵 • Total trades: {summary['total_trades']}
✅ • Win rate: {summary['total_win_rate']:.1f}%
💎 • Total P&L: ${summary['total_pnl']:,.2f}

🔓 **Posiciones Abiertas**
📊 • Total: {summary['open_positions']}
💰 • P&L: ${summary['portfolio_pnl']:,.2f}

🔥 **Rachas**
📈 • Actual: {summary['current_streak']:+d}
🏆 • Mejor: {summary['best_streak']:+d}
⚠️ • Peor: -{summary['worst_streak']:+d}

💸 **Costos**
💰 • Fees totales: ${summary['total_fees']:,.2f}
⚠️ • Liquidaciones: {summary['liquidations']}
"""

        # Send via notifier
        notifier._send(message.strip(), priority="normal")

        print("✅ Daily summary sent successfully")
        print(f"   Balance: ${summary['balance']:,.2f}")
        print(f"   Daily P&L: ${summary['daily_pnl']:,.2f}")
        print(f"   Win rate: {summary['daily_win_rate']:.1f}%")

        return summary

    except Exception as e:
        print(f"❌ Failed to send daily summary: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate and send daily trading summary")
    parser.add_argument("--print", action="store_true", help="Print summary without sending")
    parser.add_argument("--date", type=str, help="Date for stats (YYYY-MM-DD)")

    args = parser.parse_args()

    if args.print:
        summary = generate_summary()
        print(json.dumps(summary, indent=2))
    elif args.date:
        stats = get_daily_stats(args.date)
        print(json.dumps(stats, indent=2))
    else:
        send_daily_summary()
