#!/usr/bin/env python3
"""tools/weekly_report.py — compile paper vs live performance metrics.

Phase 1 of v2.12.17 roadmap — weekly validation cadence.

Usage:
    python3 tools/weekly_report.py                      # default: last 7 days, table format
    python3 tools/weekly_report.py --days 14            # last 14 days
    python3 tools/weekly_report.py --format gate-check  # Phase 1 → Phase 2 gate criteria
    python3 tools/weekly_report.py --format json        # machine-readable

v2.12.19 — Phase 1 observability.
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

PAPER_DIR = Path("/home/enderj/.openclaw/workspace/Solana-Cripto-Trader/agents/data")
LIVE_DIR = Path("/home/enderj/.openclaw/workspace/Solana-Cripto-Trader-Live/agents/data")


def _load_history(data_dir: Path) -> list[dict]:
    p = data_dir / "trade_history.json"
    if not p.exists():
        return []
    raw = json.loads(p.read_text())
    trades = raw if isinstance(raw, list) else raw.get("trades", [])
    return [t for t in trades if isinstance(t, dict)]


def _parse_ts(s: Any) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _filter_window(trades: list[dict], days: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out = []
    for t in trades:
        ts = _parse_ts(t.get("close_time") or t.get("open_time"))
        if ts and ts >= cutoff:
            out.append(t)
    return out


# v2.12.32.2: exclude operational ops (orphan reconciles, manual closes) from metric calc.
# These are NOT trade decisions of the bot, they're recovery actions that should not skew
# gate-check stats. Real strategy trades close with reasons: SL, TP, TRAILING_SL, TIME_EXIT,
# EMERGENCY_CLOSE, BREAKEVEN. Operational closes start with "ORPHAN_RECONCILE" or
# "reconcile_orphan" or "MANUAL_CLOSE".
_OP_REASON_PREFIXES = ("ORPHAN_RECONCILE", "reconcile_orphan", "MANUAL_CLOSE", "orphan_needs_manual_reconcile")


def _is_strategy_trade(t: dict) -> bool:
    reason = str(t.get("close_reason", "") or "")
    return not any(reason.startswith(p) for p in _OP_REASON_PREFIXES)


def _filter_strategy_only(trades: list[dict]) -> list[dict]:
    return [t for t in trades if _is_strategy_trade(t)]


def compute_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {
            "total": 0, "wins": 0, "losses": 0, "flats": 0,
            "win_rate_pct": 0.0, "net_pnl_usd": 0.0,
            "avg_win_usd": 0.0, "avg_loss_usd": 0.0,
            "best_usd": 0.0, "worst_usd": 0.0,
            "profit_factor": 0.0, "expectancy_usd": 0.0,
        }

    wins = [t for t in trades if float(t.get("pnl_usd", 0) or 0) > 0]
    losses = [t for t in trades if float(t.get("pnl_usd", 0) or 0) < 0]
    flats = [t for t in trades if float(t.get("pnl_usd", 0) or 0) == 0]
    net = sum(float(t.get("pnl_usd", 0) or 0) for t in trades)

    gross_win = sum(float(t["pnl_usd"]) for t in wins)
    gross_loss = abs(sum(float(t["pnl_usd"]) for t in losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0

    return {
        "total": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "flats": len(flats),
        "win_rate_pct": (len(wins) / max(1, len(wins) + len(losses))) * 100,
        "net_pnl_usd": net,
        "avg_win_usd": gross_win / len(wins) if wins else 0.0,
        "avg_loss_usd": -gross_loss / len(losses) if losses else 0.0,
        "best_usd": max((float(t["pnl_usd"]) for t in trades), default=0.0),
        "worst_usd": min((float(t["pnl_usd"]) for t in trades), default=0.0),
        "profit_factor": profit_factor if profit_factor != float("inf") else 999.0,
        "expectancy_usd": net / len(trades) if trades else 0.0,
    }


def _load_portfolio(data_dir: Path) -> dict:
    p = data_dir / "portfolio.json"
    return json.loads(p.read_text()) if p.exists() else {}


def render_table(paper_m: dict, live_m: dict, paper_p: dict, live_p: dict, days: int) -> str:
    rows = [
        ("Capital actual (USD)",
         f"${paper_p.get('capital_usd', 0):.2f}",
         f"${live_p.get('capital_usd', 0):.4f}"),
        ("Capital inicial",
         f"${paper_p.get('initial_capital', 0):.2f}",
         f"${live_p.get('initial_capital', 0):.4f}"),
        ("Total trades (all-time)",
         str(paper_p.get('total_trades', 0)),
         str(live_p.get('total_trades', 0))),
        ("—" * 20, "—" * 14, "—" * 14),
        (f"Trades últimos {days} días", str(paper_m["total"]), str(live_m["total"])),
        ("  Wins", str(paper_m["wins"]), str(live_m["wins"])),
        ("  Losses", str(paper_m["losses"]), str(live_m["losses"])),
        ("  Flats", str(paper_m["flats"]), str(live_m["flats"])),
        ("Win rate (excl flats)",
         f"{paper_m['win_rate_pct']:.1f}%",
         f"{live_m['win_rate_pct']:.1f}%"),
        ("Net P&L (USD)",
         f"${paper_m['net_pnl_usd']:+.4f}",
         f"${live_m['net_pnl_usd']:+.4f}"),
        ("Avg win (USD)",
         f"${paper_m['avg_win_usd']:+.4f}",
         f"${live_m['avg_win_usd']:+.4f}"),
        ("Avg loss (USD)",
         f"${paper_m['avg_loss_usd']:+.4f}",
         f"${live_m['avg_loss_usd']:+.4f}"),
        ("Best (USD)",
         f"${paper_m['best_usd']:+.4f}",
         f"${live_m['best_usd']:+.4f}"),
        ("Worst (USD)",
         f"${paper_m['worst_usd']:+.4f}",
         f"${live_m['worst_usd']:+.4f}"),
        ("Profit factor",
         f"{paper_m['profit_factor']:.2f}",
         f"{live_m['profit_factor']:.2f}"),
        ("Expectancy/trade (USD)",
         f"${paper_m['expectancy_usd']:+.4f}",
         f"${live_m['expectancy_usd']:+.4f}"),
    ]
    w_metric = max(len(r[0]) for r in rows) + 2
    w_paper = max(len(r[1]) for r in rows) + 2
    w_live = max(len(r[2]) for r in rows) + 2

    lines = []
    lines.append(f"{'=' * (w_metric + w_paper + w_live + 6)}")
    lines.append(f"  WEEKLY REPORT — {days} days — generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"{'=' * (w_metric + w_paper + w_live + 6)}")
    header = f"  {'Metric':<{w_metric}}  {'Paper':>{w_paper}}  {'Live':>{w_live}}"
    lines.append(header)
    lines.append(f"  {'-' * w_metric}  {'-' * w_paper}  {'-' * w_live}")
    for metric, paper_v, live_v in rows:
        lines.append(f"  {metric:<{w_metric}}  {paper_v:>{w_paper}}  {live_v:>{w_live}}")
    lines.append(f"{'=' * (w_metric + w_paper + w_live + 6)}")
    return "\n".join(lines)


def render_gate_check(paper_m: dict, live_m: dict, paper_p: dict, live_p: dict, days: int) -> str:
    """Phase 1 → Phase 2 gate check (roadmap criteria)."""
    lines = []
    lines.append(f"=== PHASE 1 → PHASE 2 GATE CHECK (window: {days} days) ===\n")

    gates = [
        ("Live trades >= 50 (post-fix sample)",
         live_m["total"] >= 50,
         f"{live_m['total']} trades"),
        ("Paper win rate >= 70%",
         paper_m["win_rate_pct"] >= 70,
         f"{paper_m['win_rate_pct']:.1f}%"),
        ("Live NET pnl/trade positive",
         live_m["expectancy_usd"] > 0,
         f"${live_m['expectancy_usd']:+.4f}/trade"),
        ("Live profit factor > 1.0",
         live_m["profit_factor"] > 1.0,
         f"{live_m['profit_factor']:.2f}"),
    ]

    all_pass = True
    for name, ok, detail in gates:
        status = "✅" if ok else "❌"
        all_pass = all_pass and ok
        lines.append(f"  {status} {name}: {detail}")

    lines.append("")
    if all_pass:
        lines.append("✅ ALL GATES PASSED — ready to move to Phase 2 (Telegram channel launch).")
    else:
        lines.append("❌ Gates NOT met — continue Phase 1 observation. Re-check in 3-5 days.")

    lines.append("")
    lines.append("Note: kill-switch + orphan-crisis count NOT available from trade history alone —")
    lines.append("check /tmp/solana_live_killswitch history + watchdog log for 'KILL SWITCH ACTIVO'.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=7, help="Lookback window (default 7)")
    ap.add_argument("--format", default="table", choices=["table", "json", "gate-check"])
    args = ap.parse_args()

    paper_all = _load_history(PAPER_DIR)
    live_all = _load_history(LIVE_DIR)
    paper_w = _filter_window(paper_all, args.days)
    live_w = _filter_window(live_all, args.days)

    # v2.12.32.2: exclude operational ops (orphan reconciles, manual closes) from metrics.
    # Reconciles are recovery actions, not strategy decisions — counting them skews PF/WR.
    paper_strategy = _filter_strategy_only(paper_w)
    live_strategy = _filter_strategy_only(live_w)

    paper_m = compute_metrics(paper_strategy)
    live_m = compute_metrics(live_strategy)
    paper_p = _load_portfolio(PAPER_DIR)
    live_p = _load_portfolio(LIVE_DIR)

    if args.format == "json":
        print(json.dumps({
            "window_days": args.days,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "paper": {"metrics": paper_m, "portfolio": paper_p},
            "live": {"metrics": live_m, "portfolio": live_p},
        }, indent=2, default=str))
    elif args.format == "gate-check":
        print(render_gate_check(paper_m, live_m, paper_p, live_p, args.days))
    else:
        print(render_table(paper_m, live_m, paper_p, live_p, args.days))
    return 0


if __name__ == "__main__":
    sys.exit(main())
