#!/usr/bin/env python3
"""tools/auto_learner_soft_reset.py — soft reset counters post-capital-scale.

v2.12.22: Preserves learned params (SL/TP/leverage/risk_per_trade — all scale-invariant %)
and token preferences. Resets trade counters so auto-learner re-evaluates from the new
$100 capital baseline.

Use case: after v2.12.20 capital scale $8 → $100, previous 19 trades were learned at
a different size regime. Params are % so still valid, but adaptation counters should
restart so new trades accumulate against the new baseline.

Usage:
    python3 tools/auto_learner_soft_reset.py --dry-run     # preview
    python3 tools/auto_learner_soft_reset.py               # apply (auto-backup)
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BOT = Path("/home/enderj/.openclaw/workspace/Solana-Cripto-Trader-Live")
STATE_FILE = BOT / "agents/data/auto_learner_state.json"
HISTORY_FILE = BOT / "agents/data/trade_history.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Preview only, don't write")
    ap.add_argument("--reason", default="Capital scale v2.12.20 $8 → $100",
                    help="Reason string for notes field")
    args = ap.parse_args()

    if not STATE_FILE.exists():
        print(f"❌ state file missing: {STATE_FILE}")
        return 1

    state = json.loads(STATE_FILE.read_text())

    # Snapshot BEFORE
    print("=== BEFORE ===")
    print(f"  total_trades_learned:  {state.get('total_trades_learned', 0)}")
    print(f"  last_trade_count:      {state.get('last_trade_count', 0)}")
    print(f"  adaptation_count:      {state.get('adaptation_count', 0)}")
    print(f"  last_adaptation:       {state.get('last_adaptation', '-')}")
    print(f"  params preserved:      {list(state.get('params', {}).keys())}")
    print(f"  tokens_to_avoid:       {state.get('tokens_to_avoid', [])}")
    print(f"  tokens_to_prefer:      {state.get('tokens_to_prefer', [])}")

    # Compute current trade count (baseline freeze point)
    current_trade_count = 0
    if HISTORY_FILE.exists():
        raw = json.loads(HISTORY_FILE.read_text())
        trades = raw if isinstance(raw, list) else raw.get("trades", [])
        current_trade_count = len(trades)

    # Apply changes in memory
    now_iso = datetime.now(timezone.utc).isoformat()
    state["total_trades_learned"] = 0
    state["last_trade_count"] = current_trade_count  # freeze as baseline
    state["adaptation_count"] = 0
    state["last_adaptation"] = now_iso
    state["last_updated"] = now_iso
    state["notes"] = f"Soft reset v2.12.22: {args.reason}. Counters=0, params+tokens preserved. Baseline trade_count={current_trade_count}."

    print()
    print("=== AFTER (preview) ===")
    print(f"  total_trades_learned:  0 (reset)")
    print(f"  last_trade_count:      {current_trade_count} (frozen as new baseline)")
    print(f"  adaptation_count:      0 (reset)")
    print(f"  last_adaptation:       {now_iso}")
    print(f"  notes:                 {state['notes'][:100]}...")
    print(f"  params:                UNCHANGED (scale-invariant %)")
    print(f"  tokens_to_avoid:       UNCHANGED")
    print(f"  tokens_to_prefer:      UNCHANGED")

    if args.dry_run:
        print("\n✓ DRY RUN — no files modified")
        return 0

    # Backup before write
    epoch = int(time.time())
    bak = STATE_FILE.with_suffix(f".json.bak_{epoch}")
    shutil.copy2(STATE_FILE, bak)
    print(f"\n✓ backup: {bak.name}")

    STATE_FILE.write_text(json.dumps(state, indent=2))
    print(f"✓ written: {STATE_FILE.name}")
    print("\nAuto-learner will re-learn from trades accumulated post-scale.")
    print("First adaptation in ~5 new trades (MIN_TRADES_TO_ADAPT).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
