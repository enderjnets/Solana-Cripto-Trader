"""Initialize Drift user account + deposit USDC collateral (devnet).

Idempotent: if the user already exists, skips init. Deposit runs every call
(so re-running would double-deposit — guarded by a prompt unless --yes).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.live_drift_client import LiveDriftClient

DEFAULT_DEPOSIT_USD = 500.0


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amount", type=float, default=DEFAULT_DEPOSIT_USD,
                        help=f"USDC to deposit (default: {DEFAULT_DEPOSIT_USD})")
    parser.add_argument("--yes", action="store_true", help="Skip deposit confirmation prompt")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = LiveDriftClient()

    try:
        await client.initialize()
        print(f"\nwallet          : {client.pubkey}")
        print(f"USDC ATA        : {client.usdc_ata}")

        existed = await client.drift_user_exists()
        if existed:
            print("drift user      : already initialized (skipping init)")
        else:
            print("drift user      : not found — initializing…")
            sig = await client.initialize_user()
            print(f"init tx         : {sig}")

        if not args.yes:
            resp = input(f"\nDeposit {args.amount} USDC into Drift? [y/N] ").strip().lower()
            if resp != "y":
                print("aborted.")
                return 1

        print(f"\ndepositing {args.amount} USDC…")
        sig = await client.deposit_usdc(args.amount)
        print(f"deposit tx      : {sig}")

        snap = await client.snapshot()
        print()
        print(f"free collateral : ${snap.free_collateral_usd:,.4f}")
        print(f"total collateral: ${snap.total_collateral_usd:,.4f}")
        return 0
    finally:
        await client.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
