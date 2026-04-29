"""Transaction assembly for Jupiter Perpetuals position requests.

Composes priority fee + compute budget + main ix into a versioned transaction.
Default mode is dry_run=True: builds and serializes the tx but does NOT broadcast.

Real broadcasting is gated behind explicit `dry_run=False` and `confirm_real_money=True`.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Optional

from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solders.hash import Hash
from solders.instruction import Instruction
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction


# Defaults — tunable via env vars
DEFAULT_PRIORITY_FEE_MICRO_LAMPORTS = int(
    os.environ.get("JUP_PERP_PRIORITY_FEE_MICROLAMPORTS", "50000")
)
DEFAULT_COMPUTE_UNIT_LIMIT = int(
    os.environ.get("JUP_PERP_COMPUTE_UNIT_LIMIT", "400000")
)


@dataclass
class TxAssemblyResult:
    """Output of tx assembly. Contains both the unsigned and signed tx."""

    versioned_tx: VersionedTransaction  # signed
    serialized_b64: str
    expected_signature: str  # base58 of first signature
    recent_blockhash: str


def assemble_position_request_tx(
    *,
    main_ix: Instruction,
    payer: Keypair,
    recent_blockhash: Hash,
    priority_fee_micro_lamports: int = DEFAULT_PRIORITY_FEE_MICRO_LAMPORTS,
    compute_unit_limit: int = DEFAULT_COMPUTE_UNIT_LIMIT,
    extra_ixs: Optional[list] = None,
) -> TxAssemblyResult:
    """Assemble a v0 transaction with compute budget + main ix, signed by payer.

    Does NOT broadcast. Caller is responsible for sending via RPC.
    """
    ixs = []
    if compute_unit_limit > 0:
        ixs.append(set_compute_unit_limit(compute_unit_limit))
    if priority_fee_micro_lamports > 0:
        ixs.append(set_compute_unit_price(priority_fee_micro_lamports))
    if extra_ixs:
        ixs.extend(extra_ixs)
    ixs.append(main_ix)

    msg = MessageV0.try_compile(
        payer=payer.pubkey(),
        instructions=ixs,
        address_lookup_table_accounts=[],
        recent_blockhash=recent_blockhash,
    )
    tx = VersionedTransaction(msg, [payer])

    raw = bytes(tx)
    return TxAssemblyResult(
        versioned_tx=tx,
        serialized_b64=base64.b64encode(raw).decode("ascii"),
        expected_signature=str(tx.signatures[0]),
        recent_blockhash=str(recent_blockhash),
    )


def estimate_total_fee(
    priority_fee_micro_lamports: int = DEFAULT_PRIORITY_FEE_MICRO_LAMPORTS,
    compute_unit_limit: int = DEFAULT_COMPUTE_UNIT_LIMIT,
    base_fee_lamports: int = 5000,
) -> dict:
    """Calculate expected fee in lamports + USD estimate.

    Solana fees: base (5000 lamports/sig) + priority (CU * micro_lamports / 1_000_000).
    """
    priority_lamports = (compute_unit_limit * priority_fee_micro_lamports) // 1_000_000
    total_lamports = base_fee_lamports + priority_lamports
    sol_cost = total_lamports / 1_000_000_000
    return {
        "base_lamports": base_fee_lamports,
        "priority_lamports": priority_lamports,
        "total_lamports": total_lamports,
        "sol_cost": sol_cost,
        "usd_estimate_at_sol_100": sol_cost * 100,
    }


# ---------- ATA pre-creation helper (the keeper sometimes needs ATAs to exist) ----------


def needs_ata_creation_ix(ata_address: Pubkey) -> bool:
    """Stub: returns True if ATA might not exist yet.

    Real implementation should query getAccountInfo. For now we assume YES on first
    request (worst case: idempotent ATA creation).
    """
    return True


def build_create_ata_idempotent_ix(
    payer: Pubkey,
    owner: Pubkey,
    mint: Pubkey,
) -> Instruction:
    """Idempotent ATA creation ix (Associated Token Program).

    Discriminator: instruction type byte 1 (Create idempotent).
    Accounts: [payer(SW), ata(W), owner, mint, system_program, token_program]
    """
    from agents import jupiter_perp_pdas as pdas

    ata = pdas.derive_associated_token_address(owner, mint)
    from solders.instruction import AccountMeta

    accounts = [
        AccountMeta(payer, is_signer=True, is_writable=True),
        AccountMeta(ata, is_signer=False, is_writable=True),
        AccountMeta(owner, is_signer=False, is_writable=False),
        AccountMeta(mint, is_signer=False, is_writable=False),
        AccountMeta(pdas.SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),
        AccountMeta(pdas.TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),
    ]
    return Instruction(
        program_id=pdas.ASSOCIATED_TOKEN_PROGRAM_ID,
        accounts=accounts,
        data=bytes([1]),  # 1 = CreateIdempotent
    )
