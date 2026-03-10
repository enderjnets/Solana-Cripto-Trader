#!/usr/bin/env python3
"""Tests for risk_manager.py position sizing and limits."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.risk_manager import calculate_position_size


def test_basic_position_size():
    """Position size should be reasonable for given capital."""
    result = calculate_position_size(capital=500, price=140.0, sl_pct=0.025, leverage=3)
    assert result["margin_usd"] > 0, "Margin should be positive"
    assert result["margin_usd"] <= 500 * 0.30, f"Margin ${result['margin_usd']} should not exceed 30% of capital"
    assert result["notional_usd"] > result["margin_usd"], "Notional should be > margin with leverage"
    assert result["tokens"] > 0, "Should calculate tokens"
    print(f"✅ test_basic_position_size: margin=${result['margin_usd']}, notional=${result['notional_usd']}, tokens={result['tokens']:.4f}")


def test_leverage_capped():
    """Leverage should be capped at MAX_LEVERAGE."""
    result = calculate_position_size(capital=500, price=140.0, leverage=100)
    # Should be capped at 10x (MAX_LEVERAGE)
    effective_lev = result["notional_usd"] / result["margin_usd"] if result["margin_usd"] > 0 else 0
    assert effective_lev <= 10.1, f"Leverage should be capped, got {effective_lev}x"
    print(f"✅ test_leverage_capped: effective leverage={effective_lev:.1f}x")


def test_zero_price():
    """Zero price should not crash."""
    result = calculate_position_size(capital=500, price=0, leverage=3)
    assert result["tokens"] == 0, "Zero price should give zero tokens"
    print("✅ test_zero_price: handles gracefully")


def test_small_capital():
    """Small capital should still produce valid sizing."""
    result = calculate_position_size(capital=10, price=140.0, leverage=3)
    assert result["margin_usd"] > 0, "Even small capital should produce margin"
    assert result["margin_usd"] <= 10, "Margin should not exceed capital"
    print(f"✅ test_small_capital: margin=${result['margin_usd']}")


if __name__ == "__main__":
    test_basic_position_size()
    test_leverage_capped()
    test_zero_price()
    test_small_capital()
    print("\n✅ All risk_manager tests passed!")
