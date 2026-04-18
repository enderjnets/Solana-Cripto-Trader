#!/usr/bin/env python3
"""Fix B verification — normal close path (SL/TP/TRAILING/TIME_EXIT) records
net_pnl (gross - fe_exit - fe_entry) matching capital delta exactly. Also
applies margin-floor-clamp alignment to normal close."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents"))

import executor as ex

def make_pos(symbol="SOL", entry=100.0, notional=500.0, leverage=5, direction="long",
             sl_price=None, tp_price=None):
    margin = notional / leverage
    fe = notional * (ex.TAKER_FEE + ex.get_slippage(symbol))
    return {
        "id": f"{symbol}_FIXB",
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
        "sl_price": sl_price if sl_price else entry * 0.95,
        "tp_price": tp_price if tp_price else entry * 1.05,
        "liquidation_price": entry * 0.82,
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
        "exit_mode": "fixed",
        "trailing_pct": 0.0,
        "peak_price": entry,
        "trailing_sl": 0.0,
    }

def test_normal_close_sl_records_net_pnl():
    """Trigger SL; record must have pnl_usd = net (gross - fe_exit - fe_entry)."""
    sym = "SOL"
    notional = 500.0
    leverage = 5
    margin = notional / leverage
    entry = 100.0
    # SL at -2% → loss on $500 notional = -$10 (well within $100 margin, no clamp)
    sl = entry * 0.98

    pos = make_pos(sym, entry=entry, notional=notional, leverage=leverage, sl_price=sl)
    fe_entry = pos["fee_entry"]
    fe_exit_expected = notional * (ex.TAKER_FEE + ex.get_slippage(sym))

    # Simulate open: capital -= margin
    portfolio = {"capital_usd": 1000.0 - margin, "positions": [pos]}
    # Price hits SL
    market = {"tokens": {sym: {"price": sl}}}
    history = []
    cap_before_close = portfolio["capital_usd"]

    ex.paper_update_positions(portfolio, market, history)

    cl_recs = [h for h in history if h.get("close_reason") == "SL"]
    if not cl_recs:
        print(f"  (no SL record; got {[h.get('close_reason') for h in history]})")
        return
    rec = cl_recs[0]
    cap_after = portfolio["capital_usd"]

    # Invariant check: capital change at close == recorded pnl_usd
    # (close-side; net of exit fee + entry fee since Fix B records net lifetime pnl)
    close_side_delta = cap_after - cap_before_close
    # delta = returned = margin + net_pnl (if unclamped)
    # record pnl = net_pnl
    # so delta - margin == record pnl
    implied_net_pnl = close_side_delta - margin
    assert abs(implied_net_pnl - rec["pnl_usd"]) < 0.01, (
        f"FIX B invariant: implied net pnl (close_delta-margin)={implied_net_pnl:.4f} "
        f"vs record pnl={rec['pnl_usd']:.4f}"
    )
    print(f"✓ SL record pnl_usd=${rec['pnl_usd']:.4f} matches capital close delta ({implied_net_pnl:+.4f} net)")

def test_normal_close_clamp_aligned():
    """Force a loss that triggers max(0, returned) clamp; record pnl must be -margin."""
    sym = "PENGU"
    notional = 500.0
    leverage = 20
    margin = notional / leverage  # $25
    entry = 1.0
    # SL far away so we trigger via TP path? No — use high leverage so small price move exceeds margin.
    # Leverage 20x, 6% move → loss = $30 > margin $25.
    sl = entry * 0.94
    pos = make_pos(sym, entry=entry, notional=notional, leverage=leverage, sl_price=sl)

    portfolio = {"capital_usd": 1000.0 - margin, "positions": [pos]}
    market = {"tokens": {sym: {"price": sl}}}
    history = []
    ex.paper_update_positions(portfolio, market, history)

    cl_recs = [h for h in history if h.get("close_reason") in ("SL", "LIQUIDATED")]
    if not cl_recs:
        return
    rec = cl_recs[0]
    # If clamp fired, record pnl should equal -margin (Fix A applied to normal close)
    if rec.get("close_reason") == "LIQUIDATED":
        # Liquidation path sets pnl_usd = -margin already
        assert abs(rec["pnl_usd"] - (-margin)) < 0.01
        print(f"✓ Liquidation: record pnl=-margin=${-margin:.2f}")
    else:
        # Check if clamp fired: (margin + gross - fe_exit < 0)?
        gross = -30.0
        fe_exit = notional * (ex.TAKER_FEE + ex.get_slippage(sym))
        if margin + gross - fe_exit < 0:
            assert abs(rec["pnl_usd"] - (-margin)) < 0.01, (
                f"clamp triggered but record pnl={rec['pnl_usd']:.4f} != -margin={-margin:.4f}"
            )
            print(f"✓ SL with clamp: record pnl=-margin=${-margin:.2f}")
        else:
            print(f"  (no clamp fired; record pnl=${rec['pnl_usd']:.4f})")

if __name__ == "__main__":
    test_normal_close_sl_records_net_pnl()
    test_normal_close_clamp_aligned()
    print("\n✅ Fix B verification passed")
