#!/usr/bin/env python3
"""tools/sweep_token.py — sweep any token in MINT_MAP → USDC via Jupiter.

v2.12.10: generaliza emergency_sol_to_usdc.py para cualquier símbolo.

Usage:
    python3 tools/sweep_token.py --symbol JUP --amount 0.002  [--dry-run]
    python3 tools/sweep_token.py --symbol ETH --amount 0.00086
"""
from __future__ import annotations
import argparse, os, sys
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
    ap.add_argument('--symbol', required=True, help='Token symbol (SOL/JUP/ETH/etc)')
    ap.add_argument('--amount', type=float, required=True, help='Token amount to swap → USDC')
    ap.add_argument('--dry-run', action='store_true', help='Quote only, no broadcast')
    ap.add_argument('--slippage-bps', type=int, default=100)
    ap.add_argument('--priority', default='veryHigh', choices=['low','medium','high','veryHigh'])
    args = ap.parse_args()

    sym = args.symbol.upper()
    mint = executor.MINT_MAP.get(sym)
    if not mint:
        print(f'ERROR: {sym} not in MINT_MAP (available: {sorted(executor.MINT_MAP)})'); return 1
    decimals = executor.DECIMALS_MAP.get(sym, 9)

    w = wallet.load_wallet()
    if not w.is_live:
        print('ERROR: wallet is not live'); return 1
    rpc = solana_rpc.get_rpc()

    # Check on-chain balance
    if sym == 'SOL':
        wallet_balance = rpc.get_balance_sol(w.pubkey)
    else:
        wallet_balance = rpc.get_token_balance(w.pubkey, mint)
    usdc_before = rpc.get_token_balance(w.pubkey, solana_rpc.MINT_USDC)

    print(f'BEFORE: {sym}={wallet_balance:.6f}  USDC=${usdc_before:.4f}')
    print(f'Swap: {args.amount} {sym} → USDC (slippage {args.slippage_bps}bps, priority {args.priority})')

    if args.amount > wallet_balance:
        print(f'ERROR: requested {args.amount} {sym} but only {wallet_balance:.6f} on-chain'); return 1

    if sym == 'SOL':
        remaining = wallet_balance - args.amount
        if remaining < 0.01:
            print(f'ERROR: would leave only {remaining:.6f} SOL (<0.01 fuel reserve)'); return 1

    lamports = int(args.amount * (10 ** decimals))
    swap = jupiter_swap.JupiterSwap(wallet=w, rpc=rpc)
    result = swap.execute_swap(
        input_mint=mint,
        output_mint=solana_rpc.MINT_USDC,
        amount_lamports=lamports,
        slippage_bps=args.slippage_bps,
        priority_fee_level=args.priority,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(f'DRY RUN would receive: ${result.out_amount/1_000_000:.4f} USDC')
        print(f'  impact: {result.price_impact_pct:.4f}% | steps: {result.route_plan_steps}')
        return 0

    if not result.success or not result.confirmed:
        print(f'SWAP FAILED: {result.error}'); return 1

    usdc_received = result.out_amount / 1_000_000
    print(f'\n✅ SWAP OK: {args.amount} {sym} → ${usdc_received:.4f} USDC')
    print(f'  tx: {result.signature}')
    print(f'  impact: {result.price_impact_pct:.4f}%')

    # Post-check
    import time; time.sleep(8)
    if sym == 'SOL':
        wallet_after = rpc.get_balance_sol(w.pubkey)
    else:
        wallet_after = rpc.get_token_balance(w.pubkey, mint)
    usdc_after = rpc.get_token_balance(w.pubkey, solana_rpc.MINT_USDC)
    print(f'\nAFTER:  {sym}={wallet_after:.6f}  USDC=${usdc_after:.4f}')
    print(f'Delta:  {sym}={wallet_after-wallet_balance:+.6f}  USDC={usdc_after-usdc_before:+.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
