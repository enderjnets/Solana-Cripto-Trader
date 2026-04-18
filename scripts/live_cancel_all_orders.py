"""Emergency cleanup: cancel all open orders on the Drift user account."""
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
        assert client._drift_client is not None
        sig = await client._drift_client.cancel_orders()
        print(f"cancel_orders tx: {sig}")
        return 0
    finally:
        await client.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
