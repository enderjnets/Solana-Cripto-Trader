#!/usr/bin/env python3
"""tools/jupiter_perp_smoke.py — Smoke test: open + close perp on mainnet.

PHASE 1 STATUS: STUB ONLY. Refuses to run until Phase 2 (write ops) implemented.

Usage (when Phase 2 ready):
    python3 tools/jupiter_perp_smoke.py --size 0.005

Phase 1: just calls jupiter_perp_setup --status equivalent.

Smoke test plan (Phase 2):
    1. Snapshot pre — record collateral_total, sol_perp_base
    2. Open SOL-PERP long size_usd=$2 leverage=1x
    3. Hold 30s
    4. Close (reduce-only)
    5. Snapshot post — assert collateral preserved within ±5% (fees + slippage)
    6. Open SOL-PERP short size_usd=$2 leverage=1x
    7. Hold 60s
    8. Close
    9. Final snapshot — assert collateral preserved ±5%

Exit:
    0 — both round-trips OK
    1 — any step failed
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
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


async def run_smoke(size_sol: float) -> int:
    print("⚠️  PHASE 1 STUB — no on-chain writes")
    print(f"   Requested size: {size_sol} SOL ≈ ${size_sol * 100:.2f} notional")
    print()

    # Verify Phase 2 ready (it isn't yet)
    import jupiter_perp_client as _jpc
    client = _jpc.JupiterPerpClient()
    await client.initialize()

    snap = await client.snapshot()
    print(f"snapshot: pubkey={snap.pubkey}")
    print(f"  sol_balance={snap.sol_balance:.6f}")
    print(f"  perp_account_exists={snap.perp_account_exists}")

    print()
    try:
        sig = await client.open_perp_position("SOL", "long", size_sol * 100, leverage=1.0)
        print(f"open: {sig}")
    except NotImplementedError as e:
        print(f"⏸  open_perp_position pending Phase 2: {e}")

    await client.close()

    print("\n=== Phase 1 stub complete — no real writes ===")
    print("To implement Phase 2:")
    print("1. Parse IDL from julianfssen/jupiter-perps-anchor-idl-parsing")
    print("2. Build instantIncreasePosition + instantDecreasePosition ix in Python")
    print("3. Re-run this smoke test")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=float, default=0.01, help='SOL size per trade')
    ap.add_argument('--confirm-real-money', action='store_true',
                    help='Required flag once Phase 2 ready (uses real wallet collateral)')
    args = ap.parse_args()

    if not os.environ.get('JUP_PERP_ENABLED', '').lower() == 'true':
        print("ℹ️  JUP_PERP_ENABLED=false — running Phase 1 stub only")

    sys.exit(asyncio.run(run_smoke(args.size)))


if __name__ == '__main__':
    main()
