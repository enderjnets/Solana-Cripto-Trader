"""Mint devnet USDC to the Drift wallet via Drift's token_faucet program.

Idempotent: creates the USDC ATA if missing, then calls mint_to_user(amount).
Safe to re-run — each call mints more USDC (within faucet limits).
"""
from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import sys
from pathlib import Path

from anchorpy import Idl, Program, Provider, Wallet
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

WALLET_PATH = Path.home() / ".config" / "solana-drift-bot" / "id.json"
RPC = "https://api.devnet.solana.com"

# Verified on devnet (see investigation notes).
USDC_MINT = Pubkey.from_string("8zGuJQqwhZafTah7Uc7Z4tXRnguqkn5KLFAP8oV6PHe2")
FAUCET_PROGRAM_ID = Pubkey.from_string("V4v1mQiAdLz4qwckEb45WqHYceYizoib39cDBHSWfaB")

AMOUNT_USDC = 1000  # 6 decimals — 1000 USDC


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("mint-usdc")

    secret = json.loads(WALLET_PATH.read_text())
    kp = Keypair.from_bytes(bytes(secret))
    log.info(f"wallet pubkey: {kp.pubkey()}")

    connection = AsyncClient(RPC, commitment=Confirmed)
    wallet = Wallet(kp)
    provider = Provider(connection, wallet)

    try:
        ata = get_associated_token_address(kp.pubkey(), USDC_MINT)
        log.info(f"USDC ATA     : {ata}")

        # Create ATA if missing.
        ata_info = await connection.get_account_info(ata)
        if ata_info.value is None:
            log.info("ATA missing — creating…")
            ix = create_associated_token_account(
                payer=kp.pubkey(), owner=kp.pubkey(), mint=USDC_MINT
            )
            tx = Transaction.new_signed_with_payer([ix], kp.pubkey(), [kp],
                                                   (await connection.get_latest_blockhash()).value.blockhash)
            sig = (await connection.send_raw_transaction(bytes(tx))).value
            await connection.confirm_transaction(sig, commitment=Confirmed)
            log.info(f"ATA created  : {sig}")
        else:
            log.info("ATA already exists")

        # Load bundled faucet IDL.
        idl_path = pathlib.Path(driftpy.__file__).parent / "idl" / "token_faucet.json"
        idl = Idl.from_json(idl_path.read_text())
        program = Program(idl, FAUCET_PROGRAM_ID, provider)

        faucet_config, _ = Pubkey.find_program_address(
            [b"faucet_config", bytes(USDC_MINT)], FAUCET_PROGRAM_ID
        )
        mint_authority, _ = Pubkey.find_program_address(
            [b"mint_authority", bytes(USDC_MINT)], FAUCET_PROGRAM_ID
        )

        amount_base = AMOUNT_USDC * 10**6
        log.info(f"minting {AMOUNT_USDC} USDC ({amount_base} base units)…")
        sig = await program.rpc["mint_to_user"](
            amount_base,
            ctx=_build_ctx(
                accounts={
                    "faucet_config": faucet_config,
                    "mint_account": USDC_MINT,
                    "user_token_account": ata,
                    "mint_authority": mint_authority,
                    "token_program": TOKEN_PROGRAM_ID,
                }
            ),
        )
        log.info(f"mint tx      : {sig}")
        await connection.confirm_transaction(sig, commitment=Confirmed)

        # Read back.
        token = AsyncToken(connection, USDC_MINT, TOKEN_PROGRAM_ID, kp)
        bal = await token.get_balance(ata)
        ui_amount = bal.value.ui_amount
        log.info(f"USDC balance : {ui_amount}")
        print(f"\n✓ minted successfully — ATA {ata} now holds {ui_amount} USDC devnet")
        return 0
    finally:
        await connection.close()


def _build_ctx(accounts: dict):
    from anchorpy.program.context import Context
    return Context(accounts=accounts)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
