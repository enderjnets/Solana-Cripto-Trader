"""Phase 1 smoke test: connect to Drift on devnet and print account state.

No writes. Safe to run repeatedly.
Run with:   .venv-drift/bin/python scripts/live_connection_test.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.live_drift_client import LiveDriftClient


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = LiveDriftClient()
    try:
        await client.initialize()
        snap = await client.snapshot()
    finally:
        await client.close()

    print()
    print(f"network           : {client.env}")
    print(f"rpc               : {client.rpc_url}")
    print(f"pubkey            : {snap.pubkey}")
    print(f"native SOL        : {snap.sol_balance:.6f} SOL")
    print(f"SOL-PERP mark     : ${snap.sol_perp_mark:,.4f}")
    print(f"SOL-PERP funding  : {snap.sol_perp_funding_hourly * 100:+.4f}%/hr")
    print(f"drift user exists : {snap.drift_user_exists}")
    if snap.drift_user_exists:
        print(f"free collateral   : ${snap.free_collateral_usd:,.4f}")
        print(f"total collateral  : ${snap.total_collateral_usd:,.4f}")
        print(f"leverage          : {snap.leverage:.2f}x")
        print(f"SOL-PERP position : {snap.sol_perp_base:+.4f} SOL")
    else:
        print("(no Drift user account yet — fund wallet and initialize in Phase 3)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
