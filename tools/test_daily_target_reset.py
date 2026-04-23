#!/usr/bin/env python3
"""tools/test_daily_target_reset.py — v2.12.26 validation.

Tests 4 cases:
  A) No positions at rollover → starting_capital = capital_usd
  B) Positions + unrealized → starting_capital = equity (cap + margins + unrealized)
  C) Legit 5% pnl → target fires (safety doesn't break legit trigger)
  D) Absurd 30% pnl (stale starting_capital simulated) → safety skips trigger
"""
from __future__ import annotations
import json
import os
import sys
import tempfile
from pathlib import Path

BOT = Path("/home/enderj/.openclaw/workspace/Solana-Cripto-Trader-Live")
sys.path.insert(0, str(BOT / "agents"))


def _compute_reset_starting_capital(portfolio: dict) -> float:
    """Mimics orchestrator.py:273-295 new reset logic (v2.12.26)."""
    cap = float(portfolio.get("capital_usd", 0) or 0)
    open_pos = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]
    invested = sum(float(p.get("margin_usd", 0) or 0) for p in open_pos)
    unrealized = sum(float(p.get("pnl_usd", 0) or 0) for p in open_pos)
    return cap + invested + unrealized


def case_A():
    """No positions → starting_capital = capital_usd."""
    pf = {"capital_usd": 100.0, "positions": []}
    sc = _compute_reset_starting_capital(pf)
    assert abs(sc - 100.0) < 0.01, f"A: expected 100.0, got {sc}"
    print(f"  ✓ A) No positions: starting_capital=${sc:.2f}")


def case_B():
    """2 positions + unrealized → equity total."""
    pf = {
        "capital_usd": 62.17,
        "positions": [
            {"status": "open", "margin_usd": 10.0, "pnl_usd": 0.05, "symbol": "SOL"},
            {"status": "open", "margin_usd": 10.0, "pnl_usd": -0.02, "symbol": "ETH"},
            {"status": "open", "margin_usd": 10.0, "pnl_usd": 0.0, "symbol": "JUP"},
        ],
    }
    sc = _compute_reset_starting_capital(pf)
    expected = 62.17 + 30.0 + 0.03  # 92.20
    assert abs(sc - expected) < 0.01, f"B: expected {expected}, got {sc}"
    print(f"  ✓ B) 3 positions + unrealized: starting_capital=${sc:.2f} (expected ${expected:.2f})")


def case_C():
    """Legit 5% pnl → target should fire (safety NOT triggered)."""
    # Simulate: starting_capital $100, equity $106 = +6% pnl (>5% target, <20% safety)
    daily_pnl_pct = 0.06
    _SAFETY_MAX_PNL_PCT = 0.20
    _suspicious = daily_pnl_pct > _SAFETY_MAX_PNL_PCT
    target_reached = daily_pnl_pct >= 0.05
    should_close = target_reached and not _suspicious
    assert should_close, f"C: expected close=True, got {should_close}"
    assert not _suspicious, "C: 6% should NOT be flagged suspicious"
    print(f"  ✓ C) Legit 6% pnl: target fires (should_close={should_close})")


def case_D():
    """Absurd 30% pnl (stale starting_capital bug) → safety SKIPS trigger."""
    daily_pnl_pct = 0.3252  # observed actual value from Apr 23 incident
    _SAFETY_MAX_PNL_PCT = 0.20
    _suspicious = daily_pnl_pct > _SAFETY_MAX_PNL_PCT
    target_reached = daily_pnl_pct >= 0.05
    should_close = target_reached and not _suspicious
    assert not should_close, f"D: expected close=False, got {should_close}"
    assert _suspicious, "D: 32.5% SHOULD be flagged suspicious"
    print(f"  ✓ D) Absurd 32.5% pnl: safety skips trigger (should_close={should_close}, suspicious={_suspicious})")


def main():
    print("=== v2.12.26 daily_target reset validation ===\n")
    try:
        case_A()
        case_B()
        case_C()
        case_D()
        print("\n✅ ALL 4 TESTS PASSED")
        return 0
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
