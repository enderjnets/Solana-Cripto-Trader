#!/usr/bin/env python3
"""One-shot recovery: swap 0.023 SOL → USDC para restaurar baseline tras el incident v2.12.1.

Antes del fix, el bot abrió position SOL long pero no ejecutó el swap reverso.
Quedaron 0.023 SOL huérfanos en wallet. Este script los convierte de vuelta a USDC.

Usage: python3 tools/emergency_sol_to_usdc.py [--amount 0.023] [--dry-run]
"""
import argparse
import os
import sys
from pathlib import Path

BOT_ROOT = Path('/home/enderj/.openclaw/workspace/Solana-Cripto-Trader-Live')
env_file = BOT_ROOT / '.env'
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())
sys.path.insert(0, str(BOT_ROOT / 'agents'))

import solana_rpc, wallet, jupiter_swap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--amount', type=float, default=0.023, help='SOL amount to swap back to USDC')
    ap.add_argument('--dry-run', action='store_true', help='Quote only, no broadcast')
    args = ap.parse_args()

    w = wallet.load_wallet()
    if not w.is_live:
        print('ERROR: wallet is not live'); sys.exit(1)

    rpc = solana_rpc.get_rpc()
    sol_before = rpc.get_balance_sol(w.pubkey)
    usdc_before = rpc.get_token_balance(w.pubkey, solana_rpc.MINT_USDC)
    print(f'BEFORE: SOL={sol_before:.6f}  USDC={usdc_before:.4f}')
    print(f'Swap amount: {args.amount} SOL')

    if args.amount >= sol_before:
        print(f'ERROR: requested {args.amount} SOL but only {sol_before:.6f} available'); sys.exit(1)

    # Safety: leave at least 0.01 SOL as fuel post-swap
    remaining = sol_before - args.amount
    if remaining < 0.01:
        print(f'ERROR: would leave only {remaining:.6f} SOL (need >=0.01 for fees)'); sys.exit(1)

    lamports = int(args.amount * 1_000_000_000)
    swap = jupiter_swap.JupiterSwap(wallet=w, rpc=rpc)
    max_slip = int(os.environ.get('MAX_SLIPPAGE_BPS', 100))
    print(f'Executing swap: {args.amount} SOL → USDC (slippage max {max_slip}bps, dry_run={args.dry_run})')
    result = swap.execute_swap(
        input_mint=solana_rpc.MINT_SOL,
        output_mint=solana_rpc.MINT_USDC,
        amount_lamports=lamports,
        slippage_bps=max_slip,
        priority_fee_level='medium',
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(f'DRY RUN would receive: {result.out_amount / 1_000_000:.4f} USDC')
        print(f'  impact: {result.price_impact_pct:.4f}%')
        print(f'  route steps: {result.route_plan_steps}')
        return

    if not result.success or not result.confirmed:
        print(f'SWAP FAILED: {result.error}'); sys.exit(1)

    usdc_received = result.out_amount / 1_000_000
    print(f'\n✅ SWAP OK: {args.amount} SOL → ${usdc_received:.4f} USDC')
    print(f'  tx: {result.signature}')
    print(f'  impact: {result.price_impact_pct:.4f}%')

    sol_after = rpc.get_balance_sol(w.pubkey)
    usdc_after = rpc.get_token_balance(w.pubkey, solana_rpc.MINT_USDC)
    print(f'\nAFTER:  SOL={sol_after:.6f}  USDC={usdc_after:.4f}')
    print(f'Delta:  SOL={sol_after-sol_before:+.6f}  USDC={usdc_after-usdc_before:+.4f}')

if __name__ == '__main__':
    main()
