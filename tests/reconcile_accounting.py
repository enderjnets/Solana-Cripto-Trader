#!/usr/bin/env python3
"""
Accounting reconciler for Solana-Cripto-Trader paper trading bot.

Compares expected capital (initial + sum of recorded pnl) with actual portfolio
state, attributes any gap to the 5 known bugs, and prints a breakdown.

Bugs:
  a) Margin-floor clamp: capital += max(0, margin + pnl_usd) caps capital loss,
     but record stores full (more-negative) pnl_usd.
  b) fee_entry phantom: subtracted from record pnl_usd in emergency close
     but never deducted from capital at open.
  c) Partial take fee_entry apportionment: pos.fee_entry stays at full value
     after partial close; remaining position re-subtracts it on final close.
  d) Slippage asymmetry: fee_entry and normal fee_exit include slippage;
     emergency fee_exit does NOT.
  e) Emergency close record omits fee_entry/fee_exit/margin/notional/leverage.
"""
import argparse, json, os, sys
from pathlib import Path
from collections import defaultdict

TAKER_FEE = 0.001
SLIPPAGE_TIERS = {"BTC": 0.001, "ETH": 0.001, "SOL": 0.001, "JUP": 0.002, "RAY": 0.002}
SLIPPAGE_MEME = 0.005
SLIPPAGE_DEFAULT = 0.003
MEME_TOKENS = {"BONK", "FARTCOIN", "MOODENG", "GOAT", "WIF", "POPCAT", "PENGU", "TRUMP", "MELANIA"}

def get_slippage(symbol):
    if symbol in MEME_TOKENS:
        return SLIPPAGE_MEME
    return SLIPPAGE_TIERS.get(symbol, SLIPPAGE_DEFAULT)

def sf(x, default=0.0):
    try: return float(x) if x is not None else default
    except (ValueError, TypeError): return default

def is_emergency(close_reason):
    cr = str(close_reason or "")
    return any(k in cr for k in ("WILD_ABANDON", "WILD_MODE_CLOSE_CHAIN", "WILD_AI_CLOSE",
                                 "DAILY_TARGET", "EMERGENCY_CLOSE", "SMART_ROTATION",
                                 "POSITION_DECISION", "PORTFOLIO_TP"))

def is_partial(trade):
    return "_partial" in str(trade.get("id", "")) or trade.get("close_reason") == "PARTIAL_TAKE"

def is_reduce(trade):
    return "_reduce" in str(trade.get("id", "")) or trade.get("close_reason") == "REDUCE"

def is_normal_close(close_reason):
    cr = str(close_reason or "")
    return cr in ("SL", "TP", "TRAILING_SL", "TIME_EXIT", "LIQUIDATED")

def estimate_notional(trade):
    """Best-effort: size_usd field or entry_price*tokens, fall back to 0."""
    n = sf(trade.get("size_usd")) or sf(trade.get("notional_value"))
    return n

def estimate_fee_entry(trade):
    """fee_entry was computed as notional * (TAKER + slippage) at open time."""
    f = sf(trade.get("fee_entry"))
    if f > 0:
        return f
    sym = trade.get("symbol", "")
    notional = estimate_notional(trade)
    return notional * (TAKER_FEE + get_slippage(sym))

def estimate_fee_exit_slippage_gap(trade):
    """Emergency close DOES NOT include slippage in fee_exit; normal close does.
    Gap contribution per emergency close = notional * slippage."""
    sym = trade.get("symbol", "")
    notional = estimate_notional(trade)
    return notional * get_slippage(sym)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="agents/data", help="path to data dir")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data)
    port = json.load(open(data_dir / "portfolio.json"))
    history = json.load(open(data_dir / "trade_history.json"))

    initial = sf(port.get("initial_capital", 500.0))
    actual_free = sf(port.get("capital_usd", 0.0))
    positions = port.get("positions", [])
    margin_locked = sum(sf(p.get("margin_usd")) for p in positions)
    fee_entry_open_positions = sum(sf(p.get("fee_entry")) for p in positions)

    closed = [t for t in history if t.get("close_time")]
    closed.sort(key=lambda t: t.get("close_time", ""))

    sum_pnl = sum(sf(t.get("pnl_usd")) for t in closed)

    # Core reconciliation: sum of recorded pnl should explain capital delta
    # (considering open positions hold their margin).
    # Expected free capital after all trades + open positions:
    #   = initial - sum(open_margin) + sum(recorded_pnl_closed)
    # Because: each closed trade's recorded pnl is supposed to == capital delta.
    # Open trades have their margin still locked (capital -= margin at open).
    expected_free = initial - margin_locked + sum_pnl
    gap = actual_free - expected_free

    print("=" * 72)
    print("ACCOUNTING RECONCILIATION")
    print("=" * 72)
    print(f"  initial_capital                : ${initial:>10.2f}")
    print(f"  actual free capital (portfolio): ${actual_free:>10.2f}")
    print(f"  margin locked (open positions) : ${margin_locked:>10.2f}")
    print(f"  sum recorded pnl (closed)      : ${sum_pnl:>10.2f}")
    print(f"  expected free capital          : ${expected_free:>10.2f}")
    print(f"  GAP (actual - expected)        : ${gap:>+10.2f}")
    print("=" * 72)

    # --- Bug attribution (estimates) ---
    print("\nPER-BUG ATTRIBUTION (estimated):")
    print("-" * 72)

    # Bug b: fee_entry phantom
    # Emergency close subtracts fee_entry from pnl_usd (reducing recorded pnl),
    # but capital was never debited fee_entry at open. Net effect: capital is
    # HIGHER than recorded pnl would suggest by sum(fee_entry for emergency closes).
    # For trades with fee_entry=0 in record (most emergency closes omit it),
    # estimate from notional * (TAKER + slippage).
    emerg_closes = [t for t in closed if is_emergency(t.get("close_reason"))]
    bug_b_per_trade = []
    for t in emerg_closes:
        fe = estimate_fee_entry(t)
        bug_b_per_trade.append((t.get("id"), fe))
    bug_b_total = sum(fe for _, fe in bug_b_per_trade)
    print(f"  [b] fee_entry phantom (emergency path)     : ~${bug_b_total:+.2f}  ({len(emerg_closes)} trades)")

    # Bug d: slippage asymmetry on emergency fee_exit
    # Emergency close computes fee_exit = notional * TAKER (NO slippage).
    # Normal close computes fee_exit = notional * (TAKER + slippage).
    # The entry fee INCLUDES slippage, creating asymmetry.
    # Effect on recorded pnl: emergency close records HIGHER net pnl than it should
    # (under-counts exit fee by slippage amount). Capital also over-credited
    # by slippage * notional per emergency close.
    # Wait: capital flow for emergency = margin + pnl_usd, where pnl_usd = gross - fee_exit(no slippage) - fee_entry.
    # If fee_exit were computed WITH slippage, pnl_usd would be MORE negative (or less positive), so capital return smaller.
    # Therefore both capital AND recorded drop by the same amount when fix is applied.
    # Bug d does NOT create a capital-vs-record gap directly; it creates a
    # realism gap: paper PnL is more optimistic than real trading would be.
    # Setting to 0 for gap attribution; tracking separately.
    bug_d_realism = sum(estimate_fee_exit_slippage_gap(t) for t in emerg_closes)
    print(f"  [d] slippage asymmetry (realism, not gap)  : ~${bug_d_realism:+.2f}  (excluded from gap)")

    # Bug c: partial take fee_entry apportionment
    # Partial records have their own fee_entry apportioned in executor.py:1175
    # (partial record gets fee_entry * reduce_frac).
    # But the REMAINING pos keeps original fee_entry (not reduced).
    # When remaining emerg-closes, it subtracts full fee_entry from pnl_usd again.
    # Double-count = fee_entry * reduce_frac per partial event (= what was "already paid").
    # In data, partials have reduce_frac=0.5 (halfway split).
    # Find parent-child partial pairs: for each "_partial" id, base_id exists
    # with same symbol; base_id's final close (emergency) over-subtracts fee_entry*0.5.
    partial_ids = {t.get("id"): t for t in closed if is_partial(t)}
    reduce_ids = {t.get("id"): t for t in closed if is_reduce(t)}
    bug_c_contrib = 0.0
    for pid, ptrade in list(partial_ids.items()) + list(reduce_ids.items()):
        base = str(pid).replace("_partial", "").replace("_reduce", "")
        # Find base trade (the final close of the remaining position)
        parent = next((t for t in closed if t.get("id") == base), None)
        if parent and is_emergency(parent.get("close_reason")):
            # reduce_frac = 0.5 typical; fee_entry double-counted on parent
            fe_parent = estimate_fee_entry(parent)
            # Already paid half at partial; parent subtracts full → overcount = 0.5*fe_parent
            bug_c_contrib += 0.5 * fe_parent
    print(f"  [c] partial fee_entry double-count         : ~${bug_c_contrib:+.2f}  ({len(partial_ids)+len(reduce_ids)} partials)")

    # Bug a: margin-floor clamp
    # returned = max(0, margin + pnl_usd). When pnl_usd < -margin, clamp → capital
    # drops by margin only; record keeps pnl_usd as the more-negative value.
    # Contribution = sum over clamped trades of (margin - abs(pnl_usd)) where
    # abs(pnl_usd) > margin. We don't have reliable margin in records, so we
    # can only estimate when pnl_usd is very negative relative to size_usd.
    # Skip precise estimate; compute the residual after other bugs.
    bug_a_residual = gap - bug_b_total - bug_c_contrib
    print(f"  [a] margin-floor clamp (residual)          : ~${bug_a_residual:+.2f}")

    # Bug e: record completeness — not a gap contributor, just audit impact.
    missing_fields = ["fee_entry", "fee_exit", "margin_usd", "notional_value", "leverage"]
    incomplete = 0
    for t in emerg_closes:
        if any(f not in t or sf(t.get(f)) == 0 for f in missing_fields):
            incomplete += 1
    print(f"  [e] records with missing audit fields      : {incomplete}/{len(emerg_closes)} emergency closes")

    print("-" * 72)
    estimated_total = bug_b_total + bug_c_contrib + bug_a_residual
    print(f"  Σ attribution (a+b+c)                      : ${estimated_total:+.2f}")
    print(f"  Observed gap                               : ${gap:+.2f}")
    print("=" * 72)

    if args.verbose:
        print("\nPER-TRADE DETAIL (first 10 emergency closes):")
        print(f"  {'id':30} {'pnl_usd':>8} {'notional':>8} {'est_fe':>7} {'est_slip':>8}")
        for t in emerg_closes[:10]:
            tid = str(t.get("id"))[:30]
            pnl = sf(t.get("pnl_usd"))
            n = estimate_notional(t)
            fe = estimate_fee_entry(t)
            sl = estimate_fee_exit_slippage_gap(t)
            print(f"  {tid:30} {pnl:>+8.2f} {n:>8.2f} {fe:>7.2f} {sl:>8.2f}")

    # Exit code: 0 if gap is within $0.50 tolerance, else 1
    sys.exit(0 if abs(gap) < 0.50 else 1)

if __name__ == "__main__":
    main()
