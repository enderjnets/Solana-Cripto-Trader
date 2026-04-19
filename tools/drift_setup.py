#!/usr/bin/env python3
"""
tools/drift_setup.py — Initialize Drift subaccount + deposit USDC collateral.

Usage:
    python3 tools/drift_setup.py --env devnet --deposit 0.5     # devnet smoke
    python3 tools/drift_setup.py --env mainnet --deposit 3.0    # mainnet real
    python3 tools/drift_setup.py --status                        # read-only snapshot

What it does:
    1. Verifies driftpy is installed + wallet exists
    2. Initializes LiveDriftClient (connects to RPC, subscribes)
    3. If Drift user PDA not initialized → creates it (costs ~0.015 SOL)
    4. Ensures USDC ATA exists (costs ~0.002 SOL if missing)
    5. Deposits --deposit USDC into spot market 0 (collateral vault)
    6. Writes agents/data/drift_state.json with current state
    7. Reports final snapshot

Safety:
    - Caps total deposit at DRIFT_MAX_COLLATERAL_USD (default $3)
    - Won't deposit if would push collateral_total above cap
    - --status mode does no on-chain writes
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Load bot .env
BOT_ROOT = Path(__file__).resolve().parent.parent
env_file = BOT_ROOT / '.env'
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(BOT_ROOT / 'agents'))

STATE_FILE = BOT_ROOT / 'agents' / 'data' / 'drift_state.json'


async def do_setup(env: str, deposit_usd: float, status_only: bool) -> int:
    os.environ['DRIFT_ENV'] = env
    # Override DRIFT_RPC_URL for devnet if current is mainnet
    if env == 'devnet' and 'mainnet' in os.environ.get('DRIFT_RPC_URL', ''):
        os.environ['DRIFT_RPC_URL'] = 'https://api.devnet.solana.com'

    import drift_client as _dc
    client = _dc.LiveDriftClient(env=env)

    print(f"[drift_setup] env={env} rpc={client.rpc_url}")
    print(f"[drift_setup] wallet_path={client.wallet_path}")
    if not client.wallet_path.exists():
        print(f"❌ wallet file missing: {client.wallet_path}")
        return 1

    await client.initialize()
    print(f"✅ connected, pubkey={client.pubkey}")

    sol_balance = await client.get_native_sol_balance()
    print(f"   native SOL: {sol_balance:.6f}")

    user_exists = await client.drift_user_exists()
    print(f"   drift user exists: {user_exists}")

    if status_only:
        snap = await client.snapshot()
        print("\n=== SNAPSHOT ===")
        for k, v in snap.__dict__.items():
            print(f"  {k}: {v}")
        await client.close()
        return 0

    # Step 1: initialize Drift user if needed
    if not user_exists:
        if sol_balance < 0.02:
            print(f"❌ need ≥0.02 SOL to initialize Drift user (have {sol_balance:.6f})")
            await client.close()
            return 1
        print("→ initializing Drift user (costs ~0.015 SOL)...")
        sig = await client.initialize_user()
        print(f"   initialize_user tx: {sig or 'already-exists'}")

    # Step 2: deposit collateral
    if deposit_usd > 0:
        max_col = float(os.environ.get('DRIFT_MAX_COLLATERAL_USD', 3.0))
        snap_pre = await client.snapshot()
        if snap_pre.total_collateral_usd + deposit_usd > max_col:
            print(f"❌ would exceed DRIFT_MAX_COLLATERAL_USD (${max_col}) — "
                  f"current ${snap_pre.total_collateral_usd:.2f} + deposit ${deposit_usd} > cap")
            await client.close()
            return 1
        print(f"→ depositing ${deposit_usd} USDC into Drift spot market 0...")
        try:
            sig = await client.deposit_usdc(deposit_usd)
            print(f"   deposit tx: {sig}")
        except Exception as e:
            print(f"❌ deposit failed: {e}")
            await client.close()
            return 1

    # Step 3: final snapshot + state file
    snap = await client.snapshot()
    state = {
        'env': env,
        'subaccount_initialized': snap.drift_user_exists,
        'subaccount_id': 0,
        'pubkey': snap.pubkey,
        'total_collateral_usd': snap.total_collateral_usd,
        'free_collateral_usd': snap.free_collateral_usd,
        'current_leverage': snap.leverage,
        'last_setup_at': datetime.now(timezone.utc).isoformat(),
    }
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))
    print(f"\n✅ state written to {STATE_FILE}")
    print(f"   total collateral: ${snap.total_collateral_usd:.2f}")
    print(f"   free collateral:  ${snap.free_collateral_usd:.2f}")
    print(f"   leverage now:     {snap.leverage:.2f}x")
    print(f"   SOL-PERP base:    {snap.sol_perp_base:.4f} (0 = no position)")

    await client.close()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', choices=['devnet', 'mainnet'], default='devnet')
    ap.add_argument('--deposit', type=float, default=0.0,
                    help='USDC amount to deposit as collateral (0 = skip deposit, just init user)')
    ap.add_argument('--status', action='store_true',
                    help='Read-only: show current drift account snapshot')
    args = ap.parse_args()
    sys.exit(asyncio.run(do_setup(args.env, args.deposit, args.status)))


if __name__ == '__main__':
    main()
