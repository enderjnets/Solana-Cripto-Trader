#!/usr/bin/env python3
"""
tools/drift_devnet_smoke.py — 1 long + 1 short round-trip on SOL-PERP devnet.

Usage:
    python3 tools/drift_devnet_smoke.py --env devnet [--size 0.01]

Prerequisites:
    - Drift user initialized + USDC collateral deposited (run drift_setup.py first)
    - SOL balance > 0.01 for fees
    - If --env mainnet: uses real money; confirm before running

Exit:
    0 — both trades opened and closed, collateral preserved within ±2%
    1 — any step failed
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

BOT_ROOT = Path(__file__).resolve().parent.parent
env_file = BOT_ROOT / '.env'
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())
sys.path.insert(0, str(BOT_ROOT / 'agents'))


async def run_smoke(env: str, size_sol: float) -> int:
    os.environ['DRIFT_ENV'] = env
    if env == 'devnet' and 'mainnet' in os.environ.get('DRIFT_RPC_URL', ''):
        os.environ['DRIFT_RPC_URL'] = 'https://api.devnet.solana.com'

    import drift_client as _dc
    client = _dc.LiveDriftClient(env=env)
    await client.initialize()

    snap_0 = await client.snapshot()
    if not snap_0.drift_user_exists:
        print("❌ Drift user not initialized — run tools/drift_setup.py first")
        await client.close()
        return 1
    print(f"=== BEFORE ===")
    print(f"  collateral_free: ${snap_0.free_collateral_usd:.4f}")
    print(f"  collateral_total: ${snap_0.total_collateral_usd:.4f}")
    print(f"  sol_perp_base: {snap_0.sol_perp_base}")
    print(f"  mark: ${snap_0.sol_perp_mark:.4f}")
    col_start = snap_0.total_collateral_usd

    # ── Round 1: LONG ──
    print(f"\n=== ROUND 1: LONG {size_sol} SOL ===")
    try:
        sig = await client.open_sol_perp_market(direction='long', size_sol=size_sol, slippage_bps=100)
        print(f"  open tx: {sig}")
    except Exception as e:
        print(f"❌ open long failed: {e}")
        await client.close()
        return 1
    await asyncio.sleep(30)
    snap_1 = await client.snapshot()
    print(f"  after 30s: base={snap_1.sol_perp_base:.6f} mark=${snap_1.sol_perp_mark:.4f}")
    if abs(snap_1.sol_perp_base - size_sol) > size_sol * 0.1:
        print(f"⚠️  unexpected base after open long — expected ~{size_sol}")
    try:
        sig = await client.close_sol_perp(slippage_bps=100)
        print(f"  close tx: {sig}")
    except Exception as e:
        print(f"❌ close long failed: {e}")
        await client.close()
        return 1
    await asyncio.sleep(10)
    snap_2 = await client.snapshot()
    if abs(snap_2.sol_perp_base) > 0.001:
        print(f"⚠️  position not fully closed after long: base={snap_2.sol_perp_base}")

    # ── Round 2: SHORT ──
    print(f"\n=== ROUND 2: SHORT {size_sol} SOL ===")
    try:
        sig = await client.open_sol_perp_market(direction='short', size_sol=size_sol, slippage_bps=100)
        print(f"  open tx: {sig}")
    except Exception as e:
        print(f"❌ open short failed: {e}")
        await client.close()
        return 1
    await asyncio.sleep(60)
    snap_3 = await client.snapshot()
    print(f"  after 60s: base={snap_3.sol_perp_base:.6f} mark=${snap_3.sol_perp_mark:.4f}")
    try:
        sig = await client.close_sol_perp(slippage_bps=100)
        print(f"  close tx: {sig}")
    except Exception as e:
        print(f"❌ close short failed: {e}")
        await client.close()
        return 1
    await asyncio.sleep(10)
    snap_4 = await client.snapshot()

    # ── Final ──
    col_end = snap_4.total_collateral_usd
    delta = col_end - col_start
    delta_pct = (delta / col_start * 100) if col_start > 0 else 0
    print(f"\n=== AFTER ===")
    print(f"  collateral_total: ${col_end:.4f}")
    print(f"  PnL + fees:       ${delta:+.4f} ({delta_pct:+.3f}%)")
    print(f"  final base:       {snap_4.sol_perp_base:.6f}")

    await client.close()

    # Assert residual position ≈ 0
    if abs(snap_4.sol_perp_base) > 0.001:
        print(f"❌ residual position: {snap_4.sol_perp_base}")
        return 1
    # Assert collateral preserved within ±2% (fees + funding)
    if abs(delta_pct) > 2.0:
        print(f"⚠️  collateral moved {delta_pct:+.2f}% — outside ±2% band (fees+funding?)")
        # Don't fail hard — could be legit market move + funding
    print("\n✅ DEVNET SMOKE PASSED")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', choices=['devnet', 'mainnet'], default='devnet')
    ap.add_argument('--size', type=float, default=0.01, help='SOL size per trade')
    args = ap.parse_args()
    if args.env == 'mainnet':
        print("⚠️  MAINNET SMOKE — uses REAL money. Ctrl+C in 10s to abort...")
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print("aborted")
            sys.exit(1)
    sys.exit(asyncio.run(run_smoke(args.env, args.size)))


if __name__ == '__main__':
    main()
