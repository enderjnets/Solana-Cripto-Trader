#!/usr/bin/env python3
"""tools/drift_devnet_mint_usdc.py — Mint devnet USDC via Drift token_faucet program.

v2.12.21 port from origin/drift-integration commit 741ab31. Changes:
- Uses DRIFT_WALLET_PATH from .env (not hardcoded solana-drift-bot path)
- Reuses _load_keypair logic (handles Jupiter custom wallet dict format)
- Idempotent: creates ATA if missing, then mints

Usage:
    python3 tools/drift_devnet_mint_usdc.py                # mints 1000 USDC devnet
    python3 tools/drift_devnet_mint_usdc.py --amount 500   # mints 500 USDC devnet
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pathlib
import sys
from pathlib import Path

from anchorpy import Idl, Program, Provider, Wallet
from anchorpy.program.context import Context
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import Transaction
from spl.token.async_client import AsyncToken
from spl.token.constants import TOKEN_PROGRAM_ID
from spl.token.instructions import (
    create_associated_token_account,
    get_associated_token_address,
)

import driftpy

BOT_ROOT = Path(__file__).resolve().parent.parent
for line in (BOT_ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

DEVNET_RPC = "https://api.devnet.solana.com"
USDC_MINT = Pubkey.from_string("8zGuJQqwhZafTah7Uc7Z4tXRnguqkn5KLFAP8oV6PHe2")
FAUCET_PROGRAM_ID = Pubkey.from_string("V4v1mQiAdLz4qwckEb45WqHYceYizoib39cDBHSWfaB")


def _load_keypair(path: Path) -> Keypair:
    """Same loader pattern as agents/drift_client.py — supports both formats."""
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return Keypair.from_bytes(bytes(data))
    if isinstance(data, dict) and "private_key" in data:
        pk = data["private_key"]
        if isinstance(pk, list):
            return Keypair.from_bytes(bytes(pk))
        if isinstance(pk, str):
            s = pk.strip()
            if s.startswith("["):
                return Keypair.from_bytes(bytes(json.loads(s)))
            import base58
            return Keypair.from_bytes(base58.b58decode(s))
    raise ValueError(f"unsupported wallet format in {path}")


async def main(amount_usdc: int) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("mint-usdc")

    wallet_path = Path(os.environ.get(
        "DRIFT_WALLET_PATH",
        str(Path.home() / ".config" / "solana-jupiter-bot" / "wallet.json"),
    ))
    if not wallet_path.exists():
        log.error(f"wallet not found: {wallet_path}")
        return 1

    kp = _load_keypair(wallet_path)
    log.info(f"wallet pubkey: {kp.pubkey()}")
    log.info(f"minting to  : {amount_usdc} USDC devnet")

    connection = AsyncClient(DEVNET_RPC, commitment=Confirmed)
    wallet = Wallet(kp)
    provider = Provider(connection, wallet)

    try:
        ata = get_associated_token_address(kp.pubkey(), USDC_MINT)
        log.info(f"USDC ATA    : {ata}")

        ata_info = await connection.get_account_info(ata)
        if ata_info.value is None:
            log.info("ATA missing — creating…")
            ix = create_associated_token_account(
                payer=kp.pubkey(), owner=kp.pubkey(), mint=USDC_MINT
            )
            blockhash = (await connection.get_latest_blockhash()).value.blockhash
            tx = Transaction.new_signed_with_payer([ix], kp.pubkey(), [kp], blockhash)
            sig_resp = await connection.send_raw_transaction(bytes(tx))
            await connection.confirm_transaction(sig_resp.value, commitment=Confirmed)
            log.info(f"ATA created : {sig_resp.value}")
        else:
            log.info("ATA already exists")

        idl_path = pathlib.Path(driftpy.__file__).parent / "idl" / "token_faucet.json"
        if not idl_path.exists():
            log.error(f"token_faucet.json not found in driftpy package: {idl_path}")
            return 1
        idl = Idl.from_json(idl_path.read_text())
        program = Program(idl, FAUCET_PROGRAM_ID, provider)

        faucet_config, _ = Pubkey.find_program_address(
            [b"faucet_config", bytes(USDC_MINT)], FAUCET_PROGRAM_ID
        )
        mint_authority, _ = Pubkey.find_program_address(
            [b"mint_authority", bytes(USDC_MINT)], FAUCET_PROGRAM_ID
        )

        amount_base = amount_usdc * 10**6
        log.info(f"calling mint_to_user({amount_base})…")
        sig = await program.rpc["mint_to_user"](
            amount_base,
            ctx=Context(accounts={
                "faucet_config": faucet_config,
                "mint_account": USDC_MINT,
                "user_token_account": ata,
                "mint_authority": mint_authority,
                "token_program": TOKEN_PROGRAM_ID,
            }),
        )
        log.info(f"mint tx     : {sig}")
        await connection.confirm_transaction(sig, commitment=Confirmed)

        token = AsyncToken(connection, USDC_MINT, TOKEN_PROGRAM_ID, kp)
        bal = await token.get_balance(ata)
        ui_amount = bal.value.ui_amount
        log.info(f"USDC balance: {ui_amount}")
        print(f"\n✓ minted — ATA {ata} now holds {ui_amount} USDC devnet")
        return 0
    finally:
        await connection.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--amount", type=int, default=1000, help="USDC units to mint (default 1000)")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(args.amount)))
