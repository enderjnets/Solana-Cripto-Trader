#!/usr/bin/env python3
"""tools/test_reconcile_asymmetric.py — unit tests for v2.12.23 asymmetric reconcile.

Tests the 4 critical cases:
  A) SOL excess (false-positive): wallet > position → severity=info, no kill switch
  B) JUP orphan (real): wallet=0 → severity=critical, kill switch
  C) ETH partial close: wallet < position → severity=critical, kill switch
  D) ETH exact match: wallet ≈ position → severity=info

v2.12.23 fix validation.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

BOT = Path("/home/enderj/.openclaw/workspace/Solana-Cripto-Trader-Live")
sys.path.insert(0, str(BOT / "agents"))

# Required env vars before import
os.environ["LIVE_TRADING_ENABLED"] = "true"
os.environ["HOT_WALLET_ADDRESS"] = "EEmtkySNz1SLNZBMBu6EsuqkEhttEKjejsEXdEFT2fMH"
os.environ.setdefault("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")


def run_case(name: str, symbol: str, position_tokens: float, wallet_balance: float,
             expect_severity: str, expect_kill: bool) -> bool:
    """Run one reconcile case with mocked RPC + portfolio."""
    import importlib
    import reconcile
    importlib.reload(reconcile)  # fresh state per test

    # Mock portfolio with 1 open live position
    mock_portfolio = {
        "positions": [
            {
                "id": f"{symbol}_live_test",
                "symbol": symbol,
                "mode": "live",
                "status": "open",
                "tokens": position_tokens,
            }
        ]
    }

    portfolio_path = Path("/tmp") / f"test_portfolio_{symbol}.json"
    import json
    portfolio_path.write_text(json.dumps(mock_portfolio))

    # Mock RPC + executor.MINT_MAP
    mock_rpc = MagicMock()
    if symbol == "SOL":
        # For SOL, reconcile calls get_balance_sol for native
        mock_rpc.get_balance_sol.return_value = wallet_balance
    else:
        mock_rpc.get_token_balance.return_value = wallet_balance

    mock_solana_rpc = MagicMock()
    mock_solana_rpc.get_rpc.return_value = mock_rpc

    mock_executor = MagicMock()
    mock_executor.MINT_MAP = {
        "SOL": "So11111111111111111111111111111111111111112",
        "JUP": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
        "ETH": "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",
    }

    # Mock safety.activate_kill_switch to track calls
    mock_safety = MagicMock()
    killed = {"called": False}
    def _fake_activate(reason: str) -> None:
        killed["called"] = True
    mock_safety.activate_kill_switch.side_effect = _fake_activate

    # Patch the imports inside reconcile module
    with patch.dict(sys.modules, {
        "solana_rpc": mock_solana_rpc,
        "safety": mock_safety,
        "executor": mock_executor,
    }):
        result = reconcile.check_reconciliation(
            portfolio_path=portfolio_path,
            trigger_kill_switch=True,
        )

    # Check results
    actual_severity = "info"
    actual_is_shortage = None
    for d in result.discrepancies:
        if d.severity in ("warning", "critical"):
            actual_severity = d.severity
            actual_is_shortage = d.is_shortage

    # SOL special: wallet is native, tool subtracts FUEL_RESERVE=0.01
    effective_wallet = max(0, wallet_balance - 0.01) if symbol == "SOL" else wallet_balance
    is_shortage_expected = effective_wallet < position_tokens

    passed = (actual_severity == expect_severity) and (killed["called"] == expect_kill)
    status = "✓" if passed else "✗"
    print(f"  {status} {name}: severity={actual_severity} (expected {expect_severity}), "
          f"kill_switch={killed['called']} (expected {expect_kill})")
    if result.discrepancies:
        for d in result.discrepancies:
            print(f"      → {d.symbol} {d.severity} shortage={d.is_shortage} diff={d.diff_pct:.2f}%")
    return passed


def main():
    print("=== v2.12.23 asymmetric reconcile unit tests ===\n")

    all_pass = True

    # Case A: SOL excess false-positive (the actual bug)
    # Position 0.115 SOL, wallet 0.163 SOL (user fuel 0.048 + position 0.115)
    # effective = 0.163 - 0.01 = 0.153 > 0.115 → excess
    all_pass &= run_case(
        "A) SOL excess (v2.12.22 33% false-positive)",
        symbol="SOL", position_tokens=0.115, wallet_balance=0.163,
        expect_severity="info", expect_kill=False,
    )

    # Case B: JUP orphan (real, wallet empty)
    # Position 11.68 JUP, wallet 0 → 100% shortage
    all_pass &= run_case(
        "B) JUP real orphan (100% shortage)",
        symbol="JUP", position_tokens=11.68, wallet_balance=0.0,
        expect_severity="critical", expect_kill=True,
    )

    # Case C: ETH partial close (50% shortage)
    all_pass &= run_case(
        "C) ETH 50% shortage",
        symbol="ETH", position_tokens=0.001, wallet_balance=0.0005,
        expect_severity="critical", expect_kill=True,
    )

    # Case D: ETH exact match (no action)
    all_pass &= run_case(
        "D) ETH exact match",
        symbol="ETH", position_tokens=0.001, wallet_balance=0.001,
        expect_severity="info", expect_kill=False,
    )

    # Case E (bonus): small excess below info threshold (benign)
    all_pass &= run_case(
        "E) JUP tiny excess 3% (benign)",
        symbol="JUP", position_tokens=10.0, wallet_balance=10.3,
        expect_severity="info", expect_kill=False,
    )

    # Case F (bonus): warning-level shortage (1% shortage, above 0.5% tolerance)
    all_pass &= run_case(
        "F) JUP 1% shortage (warning, no kill)",
        symbol="JUP", position_tokens=10.0, wallet_balance=9.9,
        expect_severity="warning", expect_kill=False,
    )

    print()
    if all_pass:
        print("✅ ALL TESTS PASSED")
        return 0
    print("❌ SOME TESTS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
