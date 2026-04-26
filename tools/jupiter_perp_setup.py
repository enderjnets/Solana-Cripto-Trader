#!/usr/bin/env python3
"""tools/jupiter_perp_setup.py — Initialize Jupiter Perps account + verify state.

Usage:
    python3 tools/jupiter_perp_setup.py --status         # read-only snapshot
    python3 tools/jupiter_perp_setup.py --verify-program # check on-chain program ID

Phase 1 (this version): READ-ONLY. Snapshot wallet + check program account.
Phase 2: deposit USDC collateral via addLiquidity2 (TODO).

Safety:
    - --status mode does ZERO on-chain writes
    - Prints expected leverage cap, market whitelist from .env
    - Requires JUP_PERP_ENABLED=true to enable any future write ops
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

STATE_FILE = BOT_ROOT / 'agents' / 'data' / 'jupiter_perp_state.json'


async def do_status() -> int:
    """Read-only snapshot of Jupiter Perps account."""
    import jupiter_perp_client as _jpc
    client = _jpc.JupiterPerpClient()

    print(f"[jup_perp_setup] env=mainnet rpc={client.rpc_url}")
    print(f"[jup_perp_setup] wallet_path={client.wallet_path}")
    print(f"[jup_perp_setup] program_id={_jpc.JUPITER_PERPS_PROGRAM_ID}")

    if not client.wallet_path.exists():
        print(f"❌ wallet file missing: {client.wallet_path}")
        return 1

    try:
        await client.initialize()
    except Exception as e:
        print(f"❌ initialize failed: {e}")
        return 1

    print(f"✅ connected, pubkey={client.pubkey}")

    sol_balance = await client.get_native_sol_balance()
    print(f"   native SOL: {sol_balance:.6f}")

    perp_exists = await client.perp_account_exists("SOL")
    print(f"   SOL perp account exists: {perp_exists}")

    snap = await client.snapshot()
    print("\n=== SNAPSHOT (Phase 1 stub) ===")
    print(f"  pubkey: {snap.pubkey}")
    print(f"  sol_balance: {snap.sol_balance:.6f}")
    print(f"  free_collateral_usd: ${snap.free_collateral_usd:.2f}")
    print(f"  total_collateral_usd: ${snap.total_collateral_usd:.2f}")
    print(f"  sol_perp_base: {snap.sol_perp_base:+.4f}")

    print("\n=== CONFIG (.env) ===")
    print(f"  JUP_PERP_ENABLED: {os.environ.get('JUP_PERP_ENABLED', 'unset')}")
    print(f"  JUP_PERP_MAX_LEVERAGE: {os.environ.get('JUP_PERP_MAX_LEVERAGE', 'unset')}")
    print(f"  JUP_PERP_MAX_COLLATERAL_USD: {os.environ.get('JUP_PERP_MAX_COLLATERAL_USD', 'unset')}")
    print(f"  JUP_PERP_MARKET_WHITELIST: {os.environ.get('JUP_PERP_MARKET_WHITELIST', 'unset')}")

    # Persist state file for monitoring
    state = {
        'env': 'mainnet',
        'pubkey': snap.pubkey,
        'sol_balance': snap.sol_balance,
        'perp_account_exists': snap.perp_account_exists,
        'last_status_at': datetime.now(timezone.utc).isoformat(),
        'phase': 1,
        'note': 'Phase 1 = read-only. Phase 2 (write ops) pending IDL parse + ix building.',
    }
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))
    print(f"\n✅ state written to {STATE_FILE}")

    await client.close()
    return 0


async def do_verify_program() -> int:
    """Verify Jupiter Perps program ID is reachable on-chain."""
    import jupiter_perp_client as _jpc
    from solana.rpc.async_api import AsyncClient

    rpc = os.environ.get("JUP_PERP_RPC_URL", "https://api.mainnet-beta.solana.com")
    print(f"checking program {_jpc.JUPITER_PERPS_PROGRAM_ID} on {rpc}...")
    async with AsyncClient(rpc) as client:
        info = await client.get_account_info(_jpc.JUPITER_PERPS_PROGRAM_ID)
        if info.value is None:
            print(f"❌ program not found")
            return 1
        print(f"✅ program found, owner={info.value.owner}, executable={info.value.executable}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--status', action='store_true', help='Read-only snapshot')
    ap.add_argument('--verify-program', action='store_true', help='Verify program ID exists on-chain')
    args = ap.parse_args()

    if args.verify_program:
        sys.exit(asyncio.run(do_verify_program()))
    elif args.status:
        sys.exit(asyncio.run(do_status()))
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
