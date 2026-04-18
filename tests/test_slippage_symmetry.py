#!/usr/bin/env python3
"""Fix D verification — emergency close fee_exit must include slippage like normal close."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents"))

import executor as ex

def make_pos(symbol, entry=1.0, notional=500.0, leverage=5, direction="long"):
    return {
        "id": f"{symbol}_SLIPTEST",
        "symbol": symbol,
        "direction": direction,
        "strategy": "test",
        "entry_price": entry,
        "current_price": entry,
        "margin_usd": notional / leverage,
        "notional_value": notional,
        "leverage": leverage,
        "size_usd": notional,
        "tokens": notional / entry,
        "sl_price": entry * 0.95,
        "tp_price": entry * 1.05,
        "liquidation_price": entry * 0.85,
        "margin_maintenance": 0.0,
        "fee_entry": notional * (ex.TAKER_FEE + ex.get_slippage(symbol)),
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

def test_bonk_meme_emergency_fee_includes_slippage():
    # Meme token: slippage = 0.005
    sym = "BONK"
    notional = 500.0
    expected_slippage = 0.005
    expected_fee = notional * (ex.TAKER_FEE + expected_slippage)

    pos = make_pos(sym, notional=notional)
    portfolio = {"capital_usd": 1000.0, "positions": [pos]}
    market = {"tokens": {sym: {"price": pos["entry_price"]}}}
    history = []

    ex.close_positions_emergency(portfolio, [sym], market, history, reason="TEST_SLIP")

    rec = history[0]
    actual_fee = rec["fee_exit"]
    # Allow $0.01 rounding
    assert abs(actual_fee - expected_fee) < 0.01, (
        f"emergency fee_exit={actual_fee:.4f} should equal {expected_fee:.4f} "
        f"(TAKER {ex.TAKER_FEE} + slippage {expected_slippage})"
    )
    print(f"✓ BONK emergency fee_exit=${actual_fee:.4f} includes slippage (${expected_fee:.4f} expected)")

def test_btc_emergency_fee_includes_slippage():
    # BTC tier: slippage = 0.001
    sym = "BTC"
    notional = 500.0
    expected_slippage = 0.001
    expected_fee = notional * (ex.TAKER_FEE + expected_slippage)

    pos = make_pos(sym, entry=70000.0, notional=notional)
    portfolio = {"capital_usd": 1000.0, "positions": [pos]}
    market = {"tokens": {sym: {"price": pos["entry_price"]}}}
    history = []

    ex.close_positions_emergency(portfolio, [sym], market, history, reason="TEST_SLIP")

    rec = history[0]
    actual_fee = rec["fee_exit"]
    assert abs(actual_fee - expected_fee) < 0.01, (
        f"BTC emergency fee_exit={actual_fee:.4f} should equal {expected_fee:.4f}"
    )
    print(f"✓ BTC emergency fee_exit=${actual_fee:.4f} includes slippage (${expected_fee:.4f} expected)")

def test_parity_with_normal_close_fee():
    # Normal close formula: notional * (TAKER + slippage)
    # Emergency should match exactly now.
    for sym in ["BTC", "ETH", "SOL", "JUP", "BONK", "FARTCOIN"]:
        notional = 300.0
        expected = notional * (ex.TAKER_FEE + ex.get_slippage(sym))

        pos = make_pos(sym, entry=1.0, notional=notional)
        portfolio = {"capital_usd": 1000.0, "positions": [pos]}
        market = {"tokens": {sym: {"price": 1.0}}}
        history = []
        ex.close_positions_emergency(portfolio, [sym], market, history, reason="TEST")
        actual = history[0]["fee_exit"]
        assert abs(actual - expected) < 0.01, (
            f"{sym}: emergency fee_exit={actual:.4f} != normal formula {expected:.4f}"
        )
    print(f"✓ Emergency fee_exit matches normal-close formula for all 6 symbols")

if __name__ == "__main__":
    test_bonk_meme_emergency_fee_includes_slippage()
    test_btc_emergency_fee_includes_slippage()
    test_parity_with_normal_close_fee()
    print("\n✅ Fix D verification passed")
