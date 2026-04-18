"""Phase 3.3 smoke test — round-trip SOL-PERP trade on devnet.

Opens a small market position, waits, then closes it. Prints snapshots at
each stage so you can see fills, fees, funding, and realized PnL.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.live_drift_client import LiveDriftClient


def _fmt_snapshot(label: str, snap) -> str:
    return (
        f"--- {label} ---\n"
        f"  free collateral  : ${snap.free_collateral_usd:,.4f}\n"
        f"  total collateral : ${snap.total_collateral_usd:,.4f}\n"
        f"  leverage         : {snap.leverage:.3f}x\n"
        f"  SOL-PERP base    : {snap.sol_perp_base:+.4f} SOL\n"
        f"  SOL-PERP mark    : ${snap.sol_perp_mark:,.4f}\n"
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=["long", "short"], default="long")
    parser.add_argument("--size", type=float, default=0.1, help="SOL position size")
    parser.add_argument("--hold", type=float, default=10.0, help="seconds to hold")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = LiveDriftClient()

    try:
        await client.initialize()

        before = await client.snapshot()
        print("\n" + _fmt_snapshot("BEFORE", before))
        if before.free_collateral_usd <= 0:
            print("ERROR: no free collateral — run live_init_and_deposit.py first")
            return 1

        print(f"opening {args.direction} {args.size} SOL via market order…")
        t0 = time.time()
        open_sig = await client.open_sol_perp_market(args.direction, args.size)
        print(f"  open tx      : {open_sig}")
        print(f"  open elapsed : {time.time() - t0:.2f}s")

        # small settle window for the account subscriber to pick up the fill
        await asyncio.sleep(3)

        after_open = await client.snapshot()
        print("\n" + _fmt_snapshot("AFTER OPEN", after_open))

        print(f"holding for {args.hold}s…")
        await asyncio.sleep(args.hold)

        held = await client.snapshot()
        print("\n" + _fmt_snapshot("AFTER HOLD", held))

        print("closing position…")
        t0 = time.time()
        close_sig = await client.close_sol_perp()
        print(f"  close tx     : {close_sig}")
        print(f"  close elapsed: {time.time() - t0:.2f}s")

        await asyncio.sleep(3)
        after_close = await client.snapshot()
        print("\n" + _fmt_snapshot("AFTER CLOSE", after_close))

        pnl = after_close.total_collateral_usd - before.total_collateral_usd
        print(f"\nrealized PnL (collateral delta): ${pnl:+.4f}")
        return 0
    finally:
        await client.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
