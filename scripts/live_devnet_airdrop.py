"""Request a devnet SOL airdrop to the Drift wallet.

Devnet faucets are rate-limited and flaky — if the RPC faucet refuses, fall back
to https://faucet.solana.com with the pubkey printed below.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solders.keypair import Keypair

WALLET_PATH = Path.home() / ".config" / "solana-drift-bot" / "id.json"
RPC = "https://api.devnet.solana.com"
AMOUNT_SOL = 1.0  # devnet per-request cap is typically 1–2 SOL; 1 succeeds more often


async def main() -> int:
    secret = json.loads(WALLET_PATH.read_text())
    kp = Keypair.from_bytes(bytes(secret))
    print(f"pubkey: {kp.pubkey()}")

    async with AsyncClient(RPC) as client:
        before = (await client.get_balance(kp.pubkey())).value / 1e9
        print(f"balance before: {before:.6f} SOL")

        print(f"requesting airdrop of {AMOUNT_SOL} SOL...")
        resp = await client.request_airdrop(kp.pubkey(), int(AMOUNT_SOL * 1e9))
        sig = resp.value
        print(f"tx signature  : {sig}")

        print("waiting for confirmation...")
        await client.confirm_transaction(sig, commitment=Confirmed)

        after = (await client.get_balance(kp.pubkey())).value / 1e9
        print(f"balance after : {after:.6f} SOL")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except Exception as e:
        print(f"airdrop failed: {e}", file=sys.stderr)
        print("fall back to https://faucet.solana.com", file=sys.stderr)
        sys.exit(1)
