#!/usr/bin/env python3
"""Fix A verification — when capital is clamped by max(0, margin+pnl), the
history record's pnl_usd must match the actual capital delta (-margin)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents"))

import executor as ex

def make_pos(symbol="PENGU", entry=1.0, notional=500.0, leverage=10, direction="long"):
    # high leverage → small margin → easy to blow past margin
    margin = notional / leverage
    fe = notional * (ex.TAKER_FEE + ex.get_slippage(symbol))
    return {
        "id": f"{symbol}_CLAMPTEST",
        "symbol": symbol,
        "direction": direction,
        "strategy": "test",
        "entry_price": entry,
        "current_price": entry,
        "margin_usd": margin,
        "notional_value": notional,
        "leverage": leverage,
        "size_usd": notional,
        "tokens": notional / entry,
        "sl_price": entry * 0.95,
        "tp_price": entry * 1.05,
        "liquidation_price": entry * 0.90,
        "margin_maintenance": 0.0,
        "fee_entry": fe,
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

def test_clamp_pnl_matches_capital_delta():
    """Force a loss bigger than margin; recorded pnl_usd should equal -margin."""
    sym = "PENGU"
    notional = 500.0
    leverage = 10
    margin = notional / leverage  # $50

    pos = make_pos(sym, entry=1.0, notional=notional, leverage=leverage)
    # Simulate open: capital -= margin (like executor.py:930)
    portfolio = {"capital_usd": 1000.0 - margin, "positions": [pos]}
    # Move price -15% adverse for long (notional loss = $75, much > margin $50)
    market = {"tokens": {sym: {"price": 0.85}}}
    history = []
    cap_at_open = 1000.0  # before -= margin
    cap_after_open = portfolio["capital_usd"]

    ex.close_positions_emergency(portfolio, [sym], market, history, reason="CLAMP_TEST")

    cap_after_close = portfolio["capital_usd"]
    # Total capital delta from the lifetime of this trade
    total_delta = cap_after_close - cap_at_open
    rec = history[0]
    rec_pnl = rec["pnl_usd"]

    # With margin floor: returned=0; total cap_delta = -margin (open charged margin, close returned 0)
    # Fix A: recorded pnl_usd should equal -margin, not the unclamped (more negative) value.
    assert abs(total_delta - (-margin)) < 0.01, (
        f"total capital_delta={total_delta:.4f} should be -margin={-margin:.4f}"
    )
    assert abs(rec_pnl - (-margin)) < 0.01, (
        f"Fix A: recorded pnl_usd={rec_pnl:.4f} should equal -margin={-margin:.4f}, "
        f"not the unclamped computed value"
    )
    print(f"✓ Loss > margin: capital_delta=${total_delta:+.2f}, recorded pnl=${rec_pnl:+.2f} (both match -margin)")

def test_no_clamp_leaves_pnl_untouched():
    """Normal loss within margin: recorded pnl == computed pnl (no clamp)."""
    sym = "SOL"
    notional = 500.0
    leverage = 5
    pos = make_pos(sym, entry=100.0, notional=notional, leverage=leverage)

    margin = notional / leverage
    portfolio = {"capital_usd": 1000.0 - margin, "positions": [pos]}
    market = {"tokens": {sym: {"price": 99.0}}}  # 1% adverse = $5 loss on $500 notional
    history = []
    cap_before_close = portfolio["capital_usd"]
    ex.close_positions_emergency(portfolio, [sym], market, history, reason="NOCLAMP")
    cap_after_close = portfolio["capital_usd"]
    # close-side delta = returned amount (margin + pnl_usd if > 0)
    close_delta = cap_after_close - cap_before_close
    rec = history[0]
    # Total lifetime delta = -margin (at open) + returned (at close) = pnl_usd (when no clamp)
    total_delta = close_delta - margin
    # No clamp expected; recorded pnl should equal total lifetime capital change
    assert abs(total_delta - rec["pnl_usd"]) < 0.01, (
        f"total_delta={total_delta:.4f} vs rec_pnl={rec['pnl_usd']:.4f}"
    )
    print(f"✓ Normal close (no clamp): total_delta=${total_delta:+.2f} == recorded=${rec['pnl_usd']:+.2f}")

if __name__ == "__main__":
    test_clamp_pnl_matches_capital_delta()
    test_no_clamp_leaves_pnl_untouched()
    print("\n✅ Fix A verification passed")
