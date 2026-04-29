"""Borsh instruction builders for Jupiter Perpetuals.

Pure functions — no RPC, no signing. Produces solders.instruction.Instruction
objects ready to compose into a transaction.

Workflow:
- Client calls build_create_increase_position_market_request_ix(...)
- Sends tx with this ix
- Jupiter keeper detects PositionRequest PDA, executes increasePosition4 (~500ms-1s)
- Client polls PositionRequest.executed flag

References:
- IDL: agents/jupiter_perps_idl.json
- Specs: docs/JUPITER_PERPS_PHASE2_SPEC.md
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from solders.instruction import AccountMeta, Instruction
from solders.pubkey import Pubkey

from . import jupiter_perp_pdas as pdas


# ---------- Borsh encoding helpers (minimal subset) ----------


def _u64(value: int) -> bytes:
    if value < 0 or value >= (1 << 64):
        raise ValueError("u64 out of range: {0}".format(value))
    return value.to_bytes(8, "little")


def _option_u64(value):
    if value is None:
        return bytes([0])
    return bytes([1]) + _u64(value)


def _option_bool(value):
    if value is None:
        return bytes([0])
    return bytes([1]) + (bytes([1]) if value else bytes([0]))


# ---------- Increase position ----------


@dataclass
class IncreasePositionRequestArgs:
    """Args for create_increase_position_market_request.

    sizeUsdDelta: position size to add, in USD lamports (1 USD = 1_000_000).
    collateralTokenDelta: collateral input in raw token lamports (USDC: 1 USDC = 1_000_000).
    side: pdas.SIDE_LONG (1) or pdas.SIDE_SHORT (2).
    priceSlippage: max acceptable price (USD lamports). For longs > current, for shorts < current.
        e.g. SOL @ $100 with 1% slippage long => priceSlippage = 101_000_000.
    jupiter_minimum_out: optional u64 (only used when input mint != collateral mint, requires swap).
        Pass None for direct USDC->USDC.
    counter: unique u64 nonce (e.g. millis-since-epoch). Must be unique per request from same owner.
    """

    size_usd_delta: int
    collateral_token_delta: int
    side: int
    price_slippage: int
    jupiter_minimum_out: Optional[int]
    counter: int

    def encode(self) -> bytes:
        if self.side not in (pdas.SIDE_LONG, pdas.SIDE_SHORT):
            raise ValueError("side must be 1 or 2, got {0}".format(self.side))
        return (
            _u64(self.size_usd_delta)
            + _u64(self.collateral_token_delta)
            + bytes([self.side])
            + _u64(self.price_slippage)
            + _option_u64(self.jupiter_minimum_out)
            + _u64(self.counter)
        )


def build_create_increase_position_market_request_ix(
    *,
    owner: Pubkey,
    target_mint: Pubkey,
    side: int,
    args: IncreasePositionRequestArgs,
    input_mint: Pubkey = pdas.USDC_MINT,
    collateral_mint: Pubkey = pdas.USDC_MINT,
    referral: Optional[Pubkey] = None,
) -> Instruction:
    """Build the create_increase_position_market_request ix.

    target_mint: which custody you're trading (SOL/BTC/ETH for those PERP markets).
    input_mint: what token the user is providing (usually USDC).
    collateral_mint: what token backs the position (usually USDC for stables-collateral mode).

    For SOL-PERP long with USDC collateral: target_mint=SOL_MINT, input/collateral=USDC_MINT.
    """
    if args.side != side:
        raise ValueError("args.side must match side parameter")

    # Derive PDAs
    perpetuals_pda, _ = pdas.derive_perpetuals_pda()
    custody_pda, _ = pdas.derive_custody_pda(target_mint)
    collateral_custody_pda, _ = pdas.derive_custody_pda(collateral_mint)
    position_pda, _ = pdas.derive_position_pda(
        owner, custody_pda, collateral_custody_pda, side
    )
    position_request_pda, _ = pdas.derive_position_request_pda(
        position_pda, args.counter, request_change=1  # 1 = Increase
    )
    event_authority_pda, _ = pdas.derive_event_authority_pda()

    # ATAs
    funding_account = pdas.derive_associated_token_address(owner, input_mint)
    position_request_ata = pdas.derive_associated_token_address(
        position_request_pda, input_mint
    )

    # Optional referral handled below
    referral_account = referral if referral is not None else pdas.JUPITER_PERPS_PROGRAM_ID

    accounts = [
        AccountMeta(owner, is_signer=True, is_writable=True),                       # 0
        AccountMeta(funding_account, is_signer=False, is_writable=True),            # 1
        AccountMeta(perpetuals_pda, is_signer=False, is_writable=False),            # 2
        AccountMeta(pdas.JLP_POOL, is_signer=False, is_writable=False),             # 3
        AccountMeta(position_pda, is_signer=False, is_writable=True),               # 4
        AccountMeta(position_request_pda, is_signer=False, is_writable=True),       # 5
        AccountMeta(position_request_ata, is_signer=False, is_writable=True),       # 6
        AccountMeta(custody_pda, is_signer=False, is_writable=False),               # 7
        AccountMeta(collateral_custody_pda, is_signer=False, is_writable=False),    # 8
        AccountMeta(input_mint, is_signer=False, is_writable=False),                # 9
        AccountMeta(referral_account, is_signer=False, is_writable=False),          # 10
        AccountMeta(pdas.TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),     # 11
        AccountMeta(pdas.ASSOCIATED_TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),  # 12
        AccountMeta(pdas.SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),    # 13
        AccountMeta(event_authority_pda, is_signer=False, is_writable=False),       # 14
        AccountMeta(pdas.JUPITER_PERPS_PROGRAM_ID, is_signer=False, is_writable=False),  # 15
    ]

    data = pdas.DISC_CREATE_INCREASE_POSITION + args.encode()

    return Instruction(
        program_id=pdas.JUPITER_PERPS_PROGRAM_ID,
        accounts=accounts,
        data=data,
    )


# ---------- Decrease position ----------


@dataclass
class DecreasePositionRequestArgs:
    """Args for createDecreasePositionMarketRequest.

    collateralUsdDelta: collateral to withdraw in USD lamports. 0 if entire_position=True.
    sizeUsdDelta: size to close in USD lamports. 0 if entire_position=True.
    priceSlippage: max acceptable price.
    jupiter_minimum_out: only when output mint != collateral.
    entire_position: True to fully close (overrides size/collateral deltas).
    counter: unique nonce.
    """

    collateral_usd_delta: int
    size_usd_delta: int
    price_slippage: int
    jupiter_minimum_out: Optional[int]
    entire_position: Optional[bool]
    counter: int

    def encode(self) -> bytes:
        return (
            _u64(self.collateral_usd_delta)
            + _u64(self.size_usd_delta)
            + _u64(self.price_slippage)
            + _option_u64(self.jupiter_minimum_out)
            + _option_bool(self.entire_position)
            + _u64(self.counter)
        )


def build_create_decrease_position_market_request_ix(
    *,
    owner: Pubkey,
    target_mint: Pubkey,
    side: int,
    args: DecreasePositionRequestArgs,
    desired_mint: Pubkey = pdas.USDC_MINT,
    collateral_mint: Pubkey = pdas.USDC_MINT,
    referral: Optional[Pubkey] = None,
) -> Instruction:
    """Build the createDecreasePositionMarketRequest ix.

    desired_mint: which token user wants to receive on close (usually USDC).
    """
    perpetuals_pda, _ = pdas.derive_perpetuals_pda()
    custody_pda, _ = pdas.derive_custody_pda(target_mint)
    collateral_custody_pda, _ = pdas.derive_custody_pda(collateral_mint)
    position_pda, _ = pdas.derive_position_pda(
        owner, custody_pda, collateral_custody_pda, side
    )
    position_request_pda, _ = pdas.derive_position_request_pda(
        position_pda, args.counter, request_change=2  # 2 = Decrease
    )
    event_authority_pda, _ = pdas.derive_event_authority_pda()

    receiving_account = pdas.derive_associated_token_address(owner, desired_mint)
    position_request_ata = pdas.derive_associated_token_address(
        position_request_pda, desired_mint
    )

    referral_account = referral if referral is not None else pdas.JUPITER_PERPS_PROGRAM_ID

    accounts = [
        AccountMeta(owner, is_signer=True, is_writable=True),                       # 0
        AccountMeta(receiving_account, is_signer=False, is_writable=True),          # 1
        AccountMeta(perpetuals_pda, is_signer=False, is_writable=False),            # 2
        AccountMeta(pdas.JLP_POOL, is_signer=False, is_writable=False),             # 3
        AccountMeta(position_pda, is_signer=False, is_writable=False),              # 4
        AccountMeta(position_request_pda, is_signer=False, is_writable=True),       # 5
        AccountMeta(position_request_ata, is_signer=False, is_writable=True),       # 6
        AccountMeta(custody_pda, is_signer=False, is_writable=False),               # 7
        AccountMeta(collateral_custody_pda, is_signer=False, is_writable=False),    # 8
        AccountMeta(desired_mint, is_signer=False, is_writable=False),              # 9
        AccountMeta(referral_account, is_signer=False, is_writable=False),          # 10
        AccountMeta(pdas.TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),     # 11
        AccountMeta(pdas.ASSOCIATED_TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),  # 12
        AccountMeta(pdas.SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),    # 13
        AccountMeta(event_authority_pda, is_signer=False, is_writable=False),       # 14
        AccountMeta(pdas.JUPITER_PERPS_PROGRAM_ID, is_signer=False, is_writable=False),  # 15
    ]

    data = pdas.DISC_CREATE_DECREASE_POSITION + args.encode()

    return Instruction(
        program_id=pdas.JUPITER_PERPS_PROGRAM_ID,
        accounts=accounts,
        data=data,
    )


# ---------- Convenience ----------


def millis_counter() -> int:
    """Standard counter: ms since epoch. Unique enough for a single owner."""
    import time
    return int(time.time() * 1000)
