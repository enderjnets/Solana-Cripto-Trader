"""
reporter.py — System Reporting Module
Generates daily reports and cycle summaries.
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

log = logging.getLogger("reporter")

DATA_DIR = Path(__file__).parent / "data"


def run(daily: bool = False) -> dict:
    """
    Generate a report. If daily=True, generates full daily report.
    Otherwise, generates a lightweight cycle report.
    """
    try:
        portfolio_file = DATA_DIR / "portfolio.json"
        history_file = DATA_DIR / "trade_history.json"
        
        portfolio = {}
        if portfolio_file.exists():
            portfolio = json.loads(portfolio_file.read_text())
        
        history = []
        if history_file.exists():
            history = json.loads(history_file.read_text())
            if isinstance(history, dict):
                history = history.get("trades", [])
        
        trades = [t for t in history if t.get("status") == "closed"]
        
        # Calculate stats
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        win_trades = [t for t in trades if t.get("pnl", 0) > 0]
        loss_trades = [t for t in trades if t.get("pnl", 0) <= 0]
        win_rate = len(win_trades) / len(trades) * 100 if trades else 0
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "daily": daily,
            "capital": portfolio.get("capital_usd", 0),
            "initial_capital": portfolio.get("initial_capital", 1000),
            "total_trades": len(trades),
            "winning_trades": len(win_trades),
            "losing_trades": len(loss_trades),
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "open_positions": len(portfolio.get("positions", [])),
            "drawdown": portfolio.get("drawdown_pct", 0),
        }
        
        # Save daily report
        report_file = DATA_DIR / "daily_report.json"
        report_file.write_text(json.dumps(report, indent=2))
        
        log.info(f"Report generated: {len(trades)} trades, ${total_pnl:.2f} P&L")
        return {"ok": True, "report": report}
    
    except Exception as e:
        log.error(f"Reporter error: {e}")
        return {"ok": False}
