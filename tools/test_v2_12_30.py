#!/usr/bin/env python3
"""v2.12.30 test suite — drawdown wallet + emergency guards + fee tracking.

Consolidated single file per plan. Runs offline with mocks; no live Solana RPC
dependency. Safe to run in any environment with the workspace on disk.

Usage:
    python3 tools/test_v2_12_30.py
    # Expected: "ALL TESTS PASSED (N cases)"
"""
from __future__ import annotations
import sys
import os
import json
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timezone, timedelta

BASE = Path("/home/enderj/.openclaw/workspace/Solana-Cripto-Trader-Live")
sys.path.insert(0, str(BASE / "agents"))

# Silence noisy loggers during tests
import logging
logging.basicConfig(level=logging.ERROR)

# ─────────────────────────────────────────────────────────────────────────────
# SUITE A — calculate_drawdown uses wallet-total with fallback
# ─────────────────────────────────────────────────────────────────────────────

def test_A1_drawdown_uses_wallet_primary():
    """Wallet equity available → drawdown computed from wallet_total."""
    import risk_manager
    pf = {"initial_capital": 100.0, "capital_usd": 50.0, "positions": []}
    # wallet $105 (above initial) → drawdown should be 0
    with patch("wallet_equity.fetch_wallet_equity", return_value={"wallet_total": 105.0}):
        dd = risk_manager.calculate_drawdown(pf)
    assert dd == 0.0, f"expected 0 drawdown when wallet > initial, got {dd}"
    print("  ✓ A1: wallet $105 > initial $100 → DD 0.0")


def test_A2_drawdown_positive_when_wallet_below():
    """Wallet $80 < initial $100 → DD 20%."""
    import risk_manager
    pf = {"initial_capital": 100.0, "capital_usd": 30.0, "positions": []}
    with patch("wallet_equity.fetch_wallet_equity", return_value={"wallet_total": 80.0}):
        dd = risk_manager.calculate_drawdown(pf)
    assert abs(dd - 0.20) < 0.001, f"expected 0.20, got {dd}"
    print(f"  ✓ A2: wallet $80 < initial $100 → DD {dd*100:.1f}%")


def test_A3_fallback_on_wallet_equity_none():
    """wallet_equity returns None → falls back to bot equity calc."""
    import risk_manager
    # bot equity = 70 + 15 + (-2) = 83 → DD = (100-83)/100 = 0.17
    pf = {
        "initial_capital": 100.0,
        "capital_usd": 70.0,
        "positions": [
            {"status": "open", "margin_usd": 15.0, "pnl_usd": -2.0},
        ],
    }
    with patch("wallet_equity.fetch_wallet_equity", return_value=None):
        dd = risk_manager.calculate_drawdown(pf)
    assert abs(dd - 0.17) < 0.001, f"expected 0.17 fallback, got {dd}"
    print(f"  ✓ A3: wallet_equity=None → fallback bot equity DD {dd*100:.1f}%")


def test_A4_zero_initial_safe():
    import risk_manager
    pf = {"initial_capital": 0, "capital_usd": 50, "positions": []}
    dd = risk_manager.calculate_drawdown(pf)
    assert dd == 0.0
    print("  ✓ A4: initial_capital=0 → DD 0 (no div-by-zero)")


def test_A5_wallet_equity_exception_falls_back():
    """Exception in wallet_equity → falls back without raising."""
    import risk_manager
    pf = {
        "initial_capital": 100.0,
        "capital_usd": 80.0,
        "positions": [],
    }
    with patch("wallet_equity.fetch_wallet_equity", side_effect=Exception("RPC down")):
        dd = risk_manager.calculate_drawdown(pf)
    assert abs(dd - 0.20) < 0.001
    print(f"  ✓ A5: wallet_equity raised → caught + fallback DD {dd*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# SUITE B — evaluate_emergency_close Cause #4 guards
# ─────────────────────────────────────────────────────────────────────────────

def _pos(symbol, age_min, pnl_pct, direction="long", pnl_usd=None):
    now = datetime.now(timezone.utc)
    opened = (now - timedelta(minutes=age_min)).isoformat()
    return {
        "symbol": symbol,
        "status": "open",
        "direction": direction,
        "open_time": opened,
        "pnl_pct": pnl_pct,
        "pnl_usd": pnl_usd if pnl_usd is not None else (pnl_pct * 10),
        "margin_usd": 10.0,
    }


def test_B1_cause4_skipped_when_position_too_young():
    """DD 20% + position 2 min old → skip (age guard)."""
    import risk_manager
    pf = {
        "initial_capital": 100.0,
        "capital_usd": 50.0,
        "positions": [_pos("SOL", age_min=2, pnl_pct=-0.02)],
    }
    with patch("wallet_equity.fetch_wallet_equity", return_value={"wallet_total": 75.0}):
        result = risk_manager.evaluate_emergency_close(pf, {"trend": "NEUTRAL", "confidence": 0}, {})
    assert not result["emergency_close"], f"should skip young position, got {result}"
    print("  ✓ B1: age 2min < 5min → skip Cause #4")


def test_B2_cause4_skipped_when_loss_too_small():
    """Position aged 10 min but only -0.3% loss → skip (loss guard)."""
    import risk_manager
    pf = {
        "initial_capital": 100.0,
        "capital_usd": 50.0,
        "positions": [_pos("SOL", age_min=10, pnl_pct=-0.003)],
    }
    with patch("wallet_equity.fetch_wallet_equity", return_value={"wallet_total": 75.0}):
        result = risk_manager.evaluate_emergency_close(pf, {"trend": "NEUTRAL", "confidence": 0}, {})
    assert not result["emergency_close"], f"should skip micro-loss, got {result}"
    print("  ✓ B2: loss 0.3% < 1% → skip Cause #4")


def test_B3_cause4_fires_when_both_guards_satisfied():
    """Age 10 min + loss 2% + DD 20% → trigger."""
    import risk_manager
    pf = {
        "initial_capital": 100.0,
        "capital_usd": 50.0,
        "positions": [_pos("SOL", age_min=10, pnl_pct=-0.02)],
    }
    with patch("wallet_equity.fetch_wallet_equity", return_value={"wallet_total": 75.0}):
        result = risk_manager.evaluate_emergency_close(pf, {"trend": "NEUTRAL", "confidence": 0}, {})
    assert result["emergency_close"], f"should trigger with both guards OK, got {result}"
    assert "SOL" in result["symbols"]
    print(f"  ✓ B3: age 10min + loss 2% + DD 25% → trigger")


def test_B4_cause1_unchanged_all_long_bearish():
    """Regression: Cause #1 (all LONG + BEARISH 80%) still works."""
    import risk_manager
    pf = {
        "initial_capital": 100.0,
        "capital_usd": 50.0,
        "positions": [_pos("SOL", age_min=2, pnl_pct=0.0)],  # age irrelevant for Cause #1
    }
    with patch("wallet_equity.fetch_wallet_equity", return_value={"wallet_total": 99.0}):
        result = risk_manager.evaluate_emergency_close(
            pf, {"trend": "BEARISH", "confidence": 0.85}, {}
        )
    assert result["emergency_close"], f"Cause #1 should still fire, got {result}"
    print("  ✓ B4: Cause #1 (BEARISH + all LONG 85%) unchanged")


def test_B5_cause3_fg_extreme_unchanged():
    """Regression: Cause #3 (F&G extreme) still works."""
    import risk_manager
    pf = {
        "initial_capital": 100.0,
        "capital_usd": 50.0,
        "positions": [_pos("SOL", age_min=2, pnl_pct=-0.02)],
    }
    with patch("wallet_equity.fetch_wallet_equity", return_value={"wallet_total": 99.0}):
        result = risk_manager.evaluate_emergency_close(
            pf, {"trend": "NEUTRAL", "confidence": 0},
            {"fear_greed": {"value": 5}}  # extreme panic
        )
    assert result["emergency_close"], f"Cause #3 should fire, got {result}"
    print("  ✓ B5: Cause #3 (F&G<10 + all LONG negative) unchanged")


# ─────────────────────────────────────────────────────────────────────────────
# SUITE C — fee_exit tracking
# ─────────────────────────────────────────────────────────────────────────────

def test_C1_fee_exit_computed_from_paper_model():
    """Verify paper model: fee = notional × (TAKER_FEE + slippage)."""
    import executor as ex
    # SOL tier 0.001 slippage, TAKER_FEE 0.001 → fee = notional * 0.002
    notional = 10.0
    expected = notional * (ex.TAKER_FEE + ex.get_slippage("SOL"))
    # Manually compute to match the real_close_position logic
    assert expected > 0, "fee should be positive"
    assert expected == 0.02, f"SOL fee expected 0.02 got {expected}"  # 10 * (0.001 + 0.001)
    print(f"  ✓ C1: SOL fee model = $10 × 0.2% = ${expected:.4f}")


def test_C2_fee_exit_higher_for_illiquid():
    """JUP has higher slippage tier (0.002) → fee > SOL (0.001)."""
    import executor as ex
    notional = 10.0
    fee_sol = notional * (ex.TAKER_FEE + ex.get_slippage("SOL"))
    fee_jup = notional * (ex.TAKER_FEE + ex.get_slippage("JUP"))
    assert fee_jup > fee_sol, f"JUP fee {fee_jup} should be > SOL {fee_sol}"
    print(f"  ✓ C2: JUP fee ${fee_jup:.4f} > SOL fee ${fee_sol:.4f} (per-symbol slippage)")


def test_C3_source_contains_new_return_fields():
    """Sanity: real_close_position source includes fee_exit in return dict."""
    src = (BASE / "agents/executor.py").read_text()
    # Look for the new return block structure
    assert '"fee_exit": fee_exit,' in src, "return dict missing fee_exit"
    assert "_cr.get(\"fee_exit\", 0)" in src or '_cr.get("fee_exit", 0)' in src, \
        "close_positions_emergency not consuming _cr.fee_exit"
    print("  ✓ C3: real_close_position + close_positions_emergency wired up")


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [
    test_A1_drawdown_uses_wallet_primary,
    test_A2_drawdown_positive_when_wallet_below,
    test_A3_fallback_on_wallet_equity_none,
    test_A4_zero_initial_safe,
    test_A5_wallet_equity_exception_falls_back,
    test_B1_cause4_skipped_when_position_too_young,
    test_B2_cause4_skipped_when_loss_too_small,
    test_B3_cause4_fires_when_both_guards_satisfied,
    test_B4_cause1_unchanged_all_long_bearish,
    test_B5_cause3_fg_extreme_unchanged,
    test_C1_fee_exit_computed_from_paper_model,
    test_C2_fee_exit_higher_for_illiquid,
    test_C3_source_contains_new_return_fields,
]


def main():
    print(f"=== v2.12.30 test suite — {len(TESTS)} cases ===\n")
    print("SUITE A — calculate_drawdown wallet-total")
    for t in TESTS[:5]:
        t()
    print("\nSUITE B — evaluate_emergency_close Cause #4 guards")
    for t in TESTS[5:10]:
        t()
    print("\nSUITE C — fee_exit tracking")
    for t in TESTS[10:]:
        t()
    print(f"\n✓ ALL TESTS PASSED ({len(TESTS)} cases)")


if __name__ == "__main__":
    main()
