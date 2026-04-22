#!/usr/bin/env python3
"""Basic backtesting framework for Solana Cripto Trader signals."""
import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent / "data"


def load_trade_history(mode_filter: str | None = None, days: int | None = None) -> list:
    """Load closed trades from trade_history.json.

    v2.12.24: handles both formats (plain list OR dict with 'trades' key).
    Optional filters: mode ('live'/'paper'/'paper_drift'), days (last N days).
    """
    from datetime import datetime, timezone, timedelta
    f = DATA_DIR / "trade_history.json"
    if not f.exists():
        return []
    data = json.loads(f.read_text())
    if isinstance(data, list):
        trades = data
    elif isinstance(data, dict):
        trades = data.get("trades", [])
    else:
        return []

    if mode_filter:
        trades = [t for t in trades if t.get("mode") == mode_filter]

    if days:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        filtered = []
        for t in trades:
            ts_str = t.get("close_time") or t.get("open_time")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                if ts >= cutoff:
                    filtered.append(t)
            except (ValueError, TypeError):
                pass
        trades = filtered

    return trades


def analyze_performance(trades):
    """Calculate key metrics from trade history."""
    if not trades:
        return {"error": "No trades to analyze", "total_trades": 0}

    wins = [t for t in trades if t.get("pnl_usd", 0) > 0]
    losses = [t for t in trades if t.get("pnl_usd", 0) <= 0]
    total_pnl = sum(t.get("pnl_usd", 0) for t in trades)
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = sum(t.get("pnl_usd", 0) for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.get("pnl_usd", 0) for t in losses) / len(losses) if losses else 0

    gross_wins = sum(t.get("pnl_usd", 0) for t in wins)
    gross_losses = sum(t.get("pnl_usd", 0) for t in losses)
    profit_factor = abs(gross_wins / gross_losses) if gross_losses != 0 else float("inf")

    # Payoff ratio
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # Max drawdown (sequential)
    equity_curve = []
    running = 0
    for t in trades:
        running += t.get("pnl_usd", 0)
        equity_curve.append(running)

    peak = 0
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd

    # By token
    by_token = {}
    for t in trades:
        tk = t.get("symbol", "?")
        if tk not in by_token:
            by_token[tk] = {"trades": 0, "wins": 0, "pnl": 0.0}
        by_token[tk]["trades"] += 1
        by_token[tk]["pnl"] = round(by_token[tk]["pnl"] + t.get("pnl_usd", 0), 2)
        if t.get("pnl_usd", 0) > 0:
            by_token[tk]["wins"] += 1

    # By signal source
    by_source = {}
    for t in trades:
        src = t.get("signal_source", "unknown")
        if src not in by_source:
            by_source[src] = {"trades": 0, "wins": 0, "pnl": 0.0}
        by_source[src]["trades"] += 1
        by_source[src]["pnl"] = round(by_source[src]["pnl"] + t.get("pnl_usd", 0), 2)
        if t.get("pnl_usd", 0) > 0:
            by_source[src]["wins"] += 1

    # By direction
    by_direction = {}
    for t in trades:
        d = t.get("direction", "long")
        if d not in by_direction:
            by_direction[d] = {"trades": 0, "wins": 0, "pnl": 0.0}
        by_direction[d]["trades"] += 1
        by_direction[d]["pnl"] = round(by_direction[d]["pnl"] + t.get("pnl_usd", 0), 2)
        if t.get("pnl_usd", 0) > 0:
            by_direction[d]["wins"] += 1

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(win_rate, 1),
        "total_pnl_usd": round(total_pnl, 4),
        "avg_win_usd": round(avg_win, 4),
        "avg_loss_usd": round(avg_loss, 4),
        "profit_factor": round(profit_factor, 2),
        "payoff_ratio": round(payoff, 2),
        "max_drawdown_usd": round(max_dd, 4),
        "by_token": by_token,
        "by_source": by_source,
        "by_direction": by_direction,
    }


def compare_strategies(trades):
    """Compare AI vs Technical signal performance."""
    ai = [t for t in trades if t.get("signal_source") == "ai_strategy"]
    tech = [t for t in trades if t.get("signal_source") in ("technical", None, "unknown")]
    return {
        "ai_strategy": analyze_performance(ai),
        "technical": analyze_performance(tech),
    }


def run(mode_filter: str | None = None, days: int | None = None):
    """Run backtest analysis and save report.

    v2.12.24: supports --mode live/paper/paper_drift filter + --days N window.
    """
    trades = load_trade_history(mode_filter=mode_filter, days=days)
    filter_desc = []
    if mode_filter: filter_desc.append(f"mode={mode_filter}")
    if days: filter_desc.append(f"last {days}d")
    filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""
    print(f"📊 Loaded {len(trades)} trades{filter_str}")

    if not trades:
        print("No trades to analyze.")
        return

    perf = analyze_performance(trades)
    print(f"\n=== Overall Performance ===")
    print(f"  Trades: {perf['total_trades']} ({perf['wins']}W / {perf['losses']}L)")
    print(f"  Win Rate: {perf['win_rate_pct']}%")
    print(f"  Total P&L: ${perf['total_pnl_usd']}")
    print(f"  Profit Factor: {perf['profit_factor']}")
    print(f"  Payoff Ratio: {perf['payoff_ratio']}")
    print(f"  Max Drawdown: ${perf['max_drawdown_usd']}")
    print(f"  Avg Win: ${perf['avg_win_usd']} | Avg Loss: ${perf['avg_loss_usd']}")

    comp = compare_strategies(trades)
    print(f"\n=== Strategy Comparison ===")
    for name, stats in comp.items():
        if stats.get("total_trades", 0) > 0:
            print(f"  {name}: {stats['total_trades']} trades, "
                  f"WR {stats.get('win_rate_pct', 0)}%, "
                  f"P&L ${stats.get('total_pnl_usd', 0)}, "
                  f"PF {stats.get('profit_factor', 0)}")

    print(f"\n=== By Token ===")
    for token, stats in perf["by_token"].items():
        wr = round(stats["wins"] / stats["trades"] * 100, 1) if stats["trades"] > 0 else 0
        print(f"  {token}: {stats['trades']} trades, WR {wr}%, P&L ${stats['pnl']}")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "performance": perf,
        "comparison": comp,
    }
    report_file = DATA_DIR / "backtest_report.json"
    report_file.write_text(json.dumps(report, indent=2))
    print(f"\n✅ Report saved to {report_file}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Backtester analysis on trade_history.json")
    ap.add_argument("--mode", choices=["live", "paper", "paper_drift", None], default=None,
                    help="Filter trades by mode (default: all)")
    ap.add_argument("--days", type=int, default=None, help="Last N days (default: all-time)")
    args = ap.parse_args()
    run(mode_filter=args.mode, days=args.days)
