#!/usr/bin/env python3
"""Fix E verification — emergency close record must contain full audit fields."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents"))

import executor as ex

REQUIRED_FIELDS = [
    "id", "symbol", "direction", "entry_price", "exit_price", "size_usd",
    "pnl_usd", "pnl_pct", "open_time", "close_time", "close_reason", "strategy",
    # New in Fix E:
    "fee_entry", "fee_exit", "margin_usd", "notional_value", "leverage", "current_price",
]

def make_fake_pos(symbol="BONK", direction="short", margin_usd=50.0, notional_value=350.0,
                  leverage=7, entry_price=1e-5, fee_entry=0.525):
    return {
        "id": f"{symbol}_TEST",
        "symbol": symbol,
        "direction": direction,
        "strategy": "test_strategy",
        "entry_price": entry_price,
        "current_price": entry_price,
        "margin_usd": margin_usd,
        "notional_value": notional_value,
        "leverage": leverage,
        "size_usd": notional_value,
        "tokens": notional_value / entry_price,
        "sl_price": entry_price * 1.05,
        "tp_price": entry_price * 0.95,
        "liquidation_price": entry_price * 1.12,
        "margin_maintenance": 0.0,
        "fee_entry": fee_entry,
        "fee_exit": 0.0,
        "funding_accumulated": 0.0,
        "pnl_usd": 0.0,
        "pnl_pct": 0.0,
        "status": "open",
        "open_time": "2026-04-18T00:00:00+00:00",
        "close_time": None,
        "mode": "paper",
        "confidence": 0.8,
        "last_funding_time": "2026-04-18T00:00:00+00:00",
    }

def test_emergency_record_has_all_fields():
    pos = make_fake_pos()
    portfolio = {"capital_usd": 500.0, "positions": [pos]}
    market = {"tokens": {"BONK": {"price": pos["entry_price"] * 1.02}}}  # 2% adverse for short
    history = []

    ex.close_positions_emergency(portfolio, ["BONK"], market, history, reason="TEST_EMERGENCY")

    assert len(history) == 1, f"expected 1 record, got {len(history)}"
    rec = history[0]
    missing = [f for f in REQUIRED_FIELDS if f not in rec]
    assert not missing, f"missing fields in emergency record: {missing}"

    # Sanity values
    assert rec["margin_usd"] == 50.0, f"margin_usd={rec['margin_usd']}"
    assert rec["notional_value"] == 350.0
    assert rec["leverage"] == 7
    assert rec["fee_entry"] == 0.525
    assert rec["fee_exit"] > 0, "fee_exit should be non-zero"
    assert rec["close_reason"] == "TEST_EMERGENCY"
    assert rec["current_price"] > 0
    print(f"✓ Record has all {len(REQUIRED_FIELDS)} required fields")
    print(f"  margin_usd={rec['margin_usd']}, notional={rec['notional_value']}, lev={rec['leverage']}")
    print(f"  fee_entry={rec['fee_entry']}, fee_exit={rec['fee_exit']:.4f}")
    print(f"  pnl_usd={rec['pnl_usd']}, reason={rec['close_reason']}")

if __name__ == "__main__":
    test_emergency_record_has_all_fields()
    print("\n✅ Fix E verification passed")
