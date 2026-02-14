#!/usr/bin/env python3
"""
Paper Trading Dashboard
====================
Real-time dashboard for paper trading performance.

Usage:
    python3 dashboard/paper_dashboard.py     # Start dashboard
    streamlit run dashboard/paper_dashboard.py  # Alternative
"""

import json
import time
from pathlib import Path
from datetime import datetime

# Paper trading state file
PAPER_STATE_FILE = Path(__file__).parent.parent / "paper_trading_state.json"


def load_state():
    """Load paper trading state."""
    if PAPER_STATE_FILE.exists():
        return json.loads(PAPER_STATE_FILE.read_text())
    return {
        "enabled": False,
        "balance_usd": 500.0,
        "initial_balance": 500.0,
        "trades": [],
        "stats": {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0
        }
    }


def get_ansi_console():
    """Generate ANSI console output (for terminal dashboard)."""
    state = load_state()

    # Calculate P&L
    pnl = state["balance_usd"] - state["initial_balance"]
    pnl_pct = (pnl / state["initial_balance"]) * 100 if state["initial_balance"] > 0 else 0

    # Runtime
    if state.get("start_time"):
        start = datetime.fromisoformat(state["start_time"])
        runtime = datetime.now() - start
        runtime_str = str(runtime).split(".")[0]
    else:
        runtime_str = "N/A"

    # Clear screen (ANSI escape)
    print("\033[2J\033[H", end="")

    # Header
    print("╔" + "═" * 78 + "╗")
    print("║" + " 📊 PAPER TRADING DASHBOARD ".center(78) + "║")
    print("╠" + "═" * 78 + "╣")

    # Status line
    status = "🟢 RUNNING" if state.get("enabled", False) else "🔴 STOPPED"
    print(f"║  Status: {status:<72} ║")
    print(f"║  Runtime: {runtime_str:<70} ║")
    print("╠" + "═" * 78 + "╣")

    # Balance section
    print(f"║  💰 BALANCE".center(78) + "║")
    print(f"║  ├─ Initial:    ${state['initial_balance']:>10,.2f}".ljust(78) + "║")
    print(f"║  ├─ Current:    ${state['balance_usd']:>10,.2f}".ljust(78) + "║")
    print(f"║  └─ P&L:        ${pnl:>10,.2f} ({pnl_pct:>+.2f}%)".ljust(78) + "║")
    print("╠" + "═" * 78 + "╣")

    # Stats section
    stats = state.get("stats", {})
    print(f"║  📈 STATISTICS".center(78) + "║")
    print(f"║  ├─ Total Trades:    {stats.get('total_trades', 0):>10}".ljust(78) + "║")
    print(f"║  ├─ Winning:        {stats.get('winning_trades', 0):>10}".ljust(78) + "║")
    print(f"║  ├─ Losing:         {stats.get('losing_trades', 0):>10}".ljust(78) + "║")
    print(f"║  └─ Win Rate:       {stats.get('win_rate', 0):>10.1f}%".ljust(78) + "║")
    print("╠" + "═" * 78 + "╣")

    # Recent signals
    signals = state.get("signals", [])[-5:]
    print(f"║  📡 RECENT SIGNALS".center(78) + "║")
    if signals:
        for s in reversed(signals):
            time_str = s.get("time", "")[-8:]
            symbol = s.get("symbol", "")
            direction = s.get("direction", "").upper()
            price = s.get("price", 0)
            emoji = "🟢" if direction == "LONG" else "🔴" if direction == "SHORT" else "📊"
            print(f"║  {emoji} {time_str} | {symbol:>4} {direction:>5} @ ${price:>7.2f}".ljust(78) + "║")
    else:
        print(f"║  No signals yet".ljust(78) + "║")
    print("╠" + "═" * 78 + "╣")

    # Open positions
    trades = [t for t in state.get("trades", []) if t.get("status") == "open"]
    print(f"║  📋 OPEN POSITIONS ({len(trades)})".center(78) + "║")
    if trades:
        for t in trades:
            entry_time = t.get("entry_time", "")[-8:]
            symbol = t.get("symbol", "")
            direction = t.get("direction", "").upper()
            entry = t.get("entry_price", 0)
            pnl = t.get("pnl", 0)
            pnl_str = f"${pnl:+.2f}"
            emoji = "🟢" if pnl >= 0 else "🔴"
            print(f"║  {emoji} {entry_time} | {symbol:>4} {direction:>5} @ ${entry:>7.2f} | P&L: {pnl_str:>10}".ljust(78) + "║")
    else:
        print(f"║  No open positions".ljust(78) + "║")
    print("╚" + "═" * 78 + "╝")

    # Footer
    print()
    print("  Press Ctrl+C to exit")
    print()


def main():
    """Main loop."""
    print("\n📊 Paper Trading Dashboard")
    print("=" * 40)
    print("Showing real-time paper trading status...")
    print("Press Ctrl+C to exit\n")

    try:
        while True:
            get_ansi_console()
            time.sleep(2)  # Refresh every 2 seconds
    except KeyboardInterrupt:
        print("\n👋 Dashboard closed")


if __name__ == "__main__":
    main()
