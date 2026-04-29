"""PDA derivations for Jupiter Perpetuals on-chain instructions.

Pure functions — no RPC, no signing. Offline-testable.

Validated 2026-04-26: SOL custody derivation matches known Jupiter address
7xS2gz2bTp3fwCC7knJvUWTEU9Tycczu6VhJYKgi1wdz.

References:
- IDL: agents/jupiter_perps_idl.json
- Docs: docs/JUPITER_PERPS_PHASE2_SPEC.md
- Community: julianfssen/jupiter-perps-anchor-idl-parsing
"""
from __future__ import annotations

import hashlib
from typing import Tuple

from solders.pubkey import Pubkey

# Program IDs
JUPITER_PERPS_PROGRAM_ID = Pubkey.from_string("PERPHjGBqRHArX4DySjwM6UJHiR3sWAatqfdBS2qQJu")
SYSTEM_PROGRAM_ID = Pubkey.from_string("11111111111111111111111111111111")
TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ASSOCIATED_TOKEN_PROGRAM_ID = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")

# Token mints (mainnet)
USDC_MINT = Pubkey.from_string("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
SOL_MINT = Pubkey.from_string("So11111111111111111111111111111111111111112")
BTC_MINT = Pubkey.from_string("3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh")
ETH_MINT = Pubkey.from_string("7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs")

# Jupiter Perpetuals JLP pool — singleton, hardcoded (NOT a PDA derivation).
# Verified on-chain 2026-04-26: owner=PERPHjGBqRH..., dataSize=2668.
JLP_POOL = Pubkey.from_string("5BUwFW4nRbftYTDMbgxykoFWqWHPzahFSNAaaaJtVKsq")

# Side enum bytes (Borsh — also used as PDA seed)
SIDE_NONE = 0
SIDE_LONG = 1
SIDE_SHORT = 2


def derive_perpetuals_pda() -> Tuple[Pubkey, int]:
    """Singleton PDA: seeds=["perpetuals"]."""
    return Pubkey.find_program_address([b"perpetuals"], JUPITER_PERPS_PROGRAM_ID)


def derive_custody_pda(mint: Pubkey, pool: Pubkey = JLP_POOL) -> Tuple[Pubkey, int]:
    """Custody PDA: seeds=["custody", pool, mint].

    Validated against known SOL custody (7xS2gz2bTp3fwCC7knJvUWTEU9Tycczu6VhJYKgi1wdz).
    """
    return Pubkey.find_program_address(
        [b"custody", bytes(pool), bytes(mint)],
        JUPITER_PERPS_PROGRAM_ID,
    )


def derive_position_pda(
    owner: Pubkey,
    custody: Pubkey,
    collateral_custody: Pubkey,
    side: int,
    pool: Pubkey = JLP_POOL,
) -> Tuple[Pubkey, int]:
    """Position PDA: seeds=["position", owner, pool, custody, collateral_custody, side_byte].

    Note: full anchor seed includes collateralCustody for non-stable (long/short) markets.
    For longs: custody=target_mint_custody, collateralCustody=USDC_custody.
    For shorts: custody=target_mint_custody, collateralCustody=USDC_custody.
    """
    if side not in (SIDE_LONG, SIDE_SHORT):
        raise ValueError(f"side must be 1 (Long) or 2 (Short), got {side}")
    return Pubkey.find_program_address(
        [
            b"position",
            bytes(owner),
            bytes(pool),
            bytes(custody),
            bytes(collateral_custody),
            bytes([side]),
        ],
        JUPITER_PERPS_PROGRAM_ID,
    )


def derive_position_request_pda(
    position: Pubkey,
    counter: int,
    request_change: int = 1,  # 1=Increase, 2=Decrease (RequestChange enum)
) -> Tuple[Pubkey, int]:
    """PositionRequest PDA: seeds=["position_request", position, counter_le_8, request_change_byte]."""
    if counter < 0 or counter >= (1 << 64):
        raise ValueError(f"counter out of u64 range: {counter}")
    if request_change not in (1, 2):
        raise ValueError(f"request_change must be 1 (Increase) or 2 (Decrease), got {request_change}")
    return Pubkey.find_program_address(
        [
            b"position_request",
            bytes(position),
            counter.to_bytes(8, "little"),
            bytes([request_change]),
        ],
        JUPITER_PERPS_PROGRAM_ID,
    )


def derive_event_authority_pda() -> Tuple[Pubkey, int]:
    """Event authority PDA: seeds=["__event_authority"]."""
    return Pubkey.find_program_address(
        [b"__event_authority"],
        JUPITER_PERPS_PROGRAM_ID,
    )


def derive_associated_token_address(owner: Pubkey, mint: Pubkey) -> Pubkey:
    """Standard ATA derivation. seeds=[owner, token_program, mint]."""
    address, _bump = Pubkey.find_program_address(
        [bytes(owner), bytes(TOKEN_PROGRAM_ID), bytes(mint)],
        ASSOCIATED_TOKEN_PROGRAM_ID,
    )
    return address


def anchor_discriminator(method: str) -> bytes:
    """Anchor 8-byte discriminator: sha256("global:<method>")[:8]."""
    return hashlib.sha256(f"global:{method}".encode("utf-8")).digest()[:8]


# Pre-computed discriminators for hot-path methods
DISC_CREATE_INCREASE_POSITION = anchor_discriminator("createIncreasePositionMarketRequest")
DISC_CREATE_DECREASE_POSITION = anchor_discriminator("createDecreasePositionMarketRequest")


def get_target_mint(market: str) -> Pubkey:
    """Map market symbol to target custody mint for SOL-PERP / BTC-PERP / ETH-PERP."""
    m = market.upper().replace("-PERP", "")
    if m == "SOL":
        return SOL_MINT
    if m == "BTC":
        return BTC_MINT
    if m == "ETH":
        return ETH_MINT
    raise ValueError(f"unsupported market: {market}")


if __name__ == "__main__":
    # Smoke test offline — no RPC needed
    print("=== Jupiter Perps PDAs offline smoke test ===")
    print(f"Program ID: {JUPITER_PERPS_PROGRAM_ID}")
    print(f"JLP pool (hardcoded): {JLP_POOL}")
    print()

    perp, b = derive_perpetuals_pda()
    print(f"perpetuals PDA: {perp} (bump {b})")

    sol_custody, b = derive_custody_pda(SOL_MINT)
    expected = "7xS2gz2bTp3fwCC7knJvUWTEU9Tycczu6VhJYKgi1wdz"
    match = " ✓ MATCH" if str(sol_custody) == expected else " ✗ MISMATCH"
    print(f"SOL custody: {sol_custody} (bump {b}){match}")

    usdc_custody, b = derive_custody_pda(USDC_MINT)
    print(f"USDC custody: {usdc_custody} (bump {b})")

    btc_custody, b = derive_custody_pda(BTC_MINT)
    print(f"BTC custody: {btc_custody} (bump {b})")

    eth_custody, b = derive_custody_pda(ETH_MINT)
    print(f"ETH custody: {eth_custody} (bump {b})")

    # dummy owner = our wallet
    owner = Pubkey.from_string("EEmtkyiCBnTtzhpKjJoY7VshDh4ZK5o4Rcdi3PJ55kbV")
    pos, b = derive_position_pda(owner, sol_custody, usdc_custody, SIDE_LONG)
    print(f"position SOL-LONG: {pos} (bump {b})")

    pr, b = derive_position_request_pda(pos, 1714119600000, request_change=1)
    print(f"positionRequest (counter=ms): {pr} (bump {b})")

    ev, b = derive_event_authority_pda()
    print(f"eventAuthority: {ev} (bump {b})")

    print()
    print(f"discriminator createIncrease: {DISC_CREATE_INCREASE_POSITION.hex()}")
    print(f"discriminator createDecrease: {DISC_CREATE_DECREASE_POSITION.hex()}")

    # ATA derivation
    ata = derive_associated_token_address(owner, USDC_MINT)
    print(f"owner USDC ATA: {ata}")
