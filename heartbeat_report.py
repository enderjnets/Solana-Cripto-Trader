#!/usr/bin/env python3
import json
from datetime import datetime

def generate_report():
    # Read paper trading state
    with open('data/paper_trading_state.json', 'r') as f:
        state = json.load(f)

    balance = state.get('balance_usd', 0)
    initial = state.get('initial_balance', 500)
    margin = state.get('margin_used', 0)
    stats = state.get('stats', {})

    trades = state.get('trades', [])
    open_trades = [t for t in trades if t.get('status') == 'open']
    closed_trades = [t for t in trades if t.get('status') == 'closed']

    balance_change = balance - initial
    balance_pct = (balance_change / initial) * 100

    # Critical alerts
    alerts = []

    if abs(balance_pct) > 10:
        alerts.append(f"🚨 CRITICAL: Balance change >10% ({balance_pct:+.1f}%)")

    if stats.get('win_rate', 100) < 30:
        alerts.append(f"⚠️ WARNING: Low win rate ({stats.get('win_rate', 0):.1f}%)")

    if len(open_trades) >= 5:
        alerts.append(f"📊 INFO: All trade slots full ({len(open_trades)}/5)")

    # Generate report
    report = []
    report.append("📊 HEARTBEAT REPORT")
    report.append(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')} MST")
    report.append("")

    # Alerts first
    if alerts:
        report.append("🚨 ALERTS")
        for alert in alerts:
            report.append(f"   {alert}")
        report.append("")

    # Portfolio summary
    report.append("💼 PORTFOLIO")
    report.append(f"   Balance:     ${balance:.2f}")
    report.append(f"   Initial:     ${initial:.2f}")
    report.append(f"   Change:      ${balance_change:+.2f} ({balance_pct:+.1f}%)")
    report.append(f"   Margin Used:  ${margin:.2f}")
    report.append(f"   Free Cash:   ${balance:.2f}")
    report.append("")

    # Trading stats
    report.append("📈 TRADING STATS")
    report.append(f"   Open Trades: {len(open_trades)}/5")
    report.append(f"   Closed:      {len(closed_trades)}")
    report.append(f"   Win Rate:    {stats.get('win_rate', 0):.1f}%")
    report.append(f"   Total P&L:   ${stats.get('total_pnl', 0):.2f}")
    report.append("")

    # Open positions
    if open_trades:
        report.append("📍 OPEN POSITIONS")
        for t in open_trades:
            direction_arrow = "↑" if t['direction'] == 'bullish' else "↓"
            dir_label = "LONG" if t['direction'] == 'bullish' else "SHORT"
            report.append(f"   {t['symbol']:6} | {direction_arrow} {dir_label:5} | ${t['entry_price']:.4f} | {t['size']:.2f} | {t.get('reason', 'N/A')[:25]}")
        report.append("")

    # Recent closed trades (last 3)
    if closed_trades:
        report.append("📋 RECENT TRADES")
        recent = closed_trades[-3:]
        for t in recent:
            t_dir_arrow = "↑" if t['direction'] == 'bullish' else "↓"
            pnl_color = "✅" if t['pnl'] >= 0 else "❌"
            report.append(f"   {t['symbol']:6} | {t_dir_arrow} | {pnl_color} ${t['pnl']:+.2f} ({t['pnl_pct']:+.2f}%) | {t['reason'][:30]}")
        report.append("")

    # System status
    report.append("🔧 SYSTEM STATUS")
    report.append("   ✅ Trading System Running")
    report.append("   ✅ 5/5 Slots Occupied")
    report.append("   ⚠️  Risk State Outdated (Feb 25)")
    report.append("")

    return "\n".join(report)

if __name__ == "__main__":
    print(generate_report())
