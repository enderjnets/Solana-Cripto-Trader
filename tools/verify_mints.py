#!/usr/bin/env python3
"""tools/verify_mints.py — pre-deploy check: validate each symbol in MINT_MAP has a Jupiter route.

Usage:
    python3 tools/verify_mints.py [--size-usd 2] [--max-impact-pct 0.5]

Exits 0 if all mints OK, 1 if any fails or impact exceeds max.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

BOT = Path(__file__).resolve().parent.parent
for l in (BOT / '.env').read_text().splitlines():
    l = l.strip()
    if l and not l.startswith('#') and '=' in l:
        k, v = l.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())
sys.path.insert(0, str(BOT / 'agents'))

import solana_rpc, wallet, jupiter_swap, executor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--size-usd', type=float, default=2.0, help='USD amount to quote')
    ap.add_argument('--max-impact-pct', type=float, default=0.5, help='Max acceptable price impact')
    args = ap.parse_args()

    w = wallet.load_wallet()
    rpc = solana_rpc.get_rpc()
    swap = jupiter_swap.JupiterSwap(wallet=w, rpc=rpc)

    USDC = solana_rpc.MINT_USDC
    usdc_lamports = int(args.size_usd * 1_000_000)

    errors = []
    print(f"{'Symbol':10s} {'Mint (first 12)':15s} {'Route':>8s} {'Impact':>10s} {'Steps':>6s} {'Reverse':>10s}")
    print('-' * 75)
    for sym, mint in executor.MINT_MAP.items():
        if sym in ('USDC',):
            continue
        decimals = executor.DECIMALS_MAP.get(sym, 9)
        q = swap.get_quote(USDC, mint, usdc_lamports, slippage_bps=100)
        if not q:
            print(f"{sym:10s} {mint[:12]:15s} {'NO_ROUTE':>8s}")
            errors.append(f"{sym}: no USDC→{sym} route")
            continue
        out = int(q.get('outAmount', 0)) / (10 ** decimals)
        impact = float(q.get('priceImpactPct', 0)) * 100
        steps = len(q.get('routePlan', []))
        if abs(impact) > args.max_impact_pct:
            errors.append(f"{sym}: impact {impact:.4f}% > max {args.max_impact_pct}%")
        # Reverse: sell 'out' back to USDC
        out_lamports = int(out * (10 ** decimals))
        qr = swap.get_quote(mint, USDC, out_lamports, slippage_bps=100) if out_lamports > 0 else None
        rev_ok = 'OK' if qr else 'FAIL'
        if not qr and sym != 'SOL':
            errors.append(f"{sym}: reverse {sym}→USDC no route")
        print(f"{sym:10s} {mint[:12]:15s} {'OK':>8s} {impact:>9.4f}% {steps:>6d} {rev_ok:>10s}")

    if errors:
        print('\n❌ FAIL:')
        for e in errors:
            print(f'  - {e}')
        return 1
    print(f'\n✅ All mints in MINT_MAP have Jupiter routes with impact <= {args.max_impact_pct}%')
    return 0


if __name__ == '__main__':
    sys.exit(main())
