"""Offline regression tests for jupiter_perp_ix builders.

Run: python3 -m pytest tests/test_jupiter_perp_ix.py
Or:  python3 tests/test_jupiter_perp_ix.py
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solders.pubkey import Pubkey

from agents import jupiter_perp_pdas as pdas
from agents.jupiter_perp_ix import (
    DecreasePositionRequestArgs,
    IncreasePositionRequestArgs,
    build_create_decrease_position_market_request_ix,
    build_create_increase_position_market_request_ix,
)

# Known reference: Jupiter SOL custody (publicly verifiable)
EXPECTED_SOL_CUSTODY = "7xS2gz2bTp3fwCC7knJvUWTEU9Tycczu6VhJYKgi1wdz"
# Discriminators for the two methods (sha256("global:<name>")[:8])
EXPECTED_DISC_INCREASE = bytes.fromhex("b7c661a92301e139")
EXPECTED_DISC_DECREASE = bytes.fromhex("93ee4c5b3056a7fd")

OWNER = Pubkey.from_string("EEmtkyiCBnTtzhpKjJoY7VshDh4ZK5o4Rcdi3PJ55kbV")
COUNTER = 1714119600000


def test_sol_custody_derivation():
    sol_custody, _ = pdas.derive_custody_pda(pdas.SOL_MINT)
    assert str(sol_custody) == EXPECTED_SOL_CUSTODY


def test_discriminators():
    assert pdas.DISC_CREATE_INCREASE_POSITION == EXPECTED_DISC_INCREASE
    assert pdas.DISC_CREATE_DECREASE_POSITION == EXPECTED_DISC_DECREASE


def test_increase_ix_layout():
    args = IncreasePositionRequestArgs(
        size_usd_delta=2_000_000,
        collateral_token_delta=2_000_000,
        side=pdas.SIDE_LONG,
        price_slippage=101_000_000,
        jupiter_minimum_out=None,
        counter=COUNTER,
    )
    ix = build_create_increase_position_market_request_ix(
        owner=OWNER,
        target_mint=pdas.SOL_MINT,
        side=pdas.SIDE_LONG,
        args=args,
    )

    assert ix.program_id == pdas.JUPITER_PERPS_PROGRAM_ID
    assert len(ix.accounts) == 16
    data = bytes(ix.data)
    assert data[:8] == EXPECTED_DISC_INCREASE
    assert len(data) == 8 + 34  # disc + (8+8+1+8+1+8) for None option

    # Account flags
    assert ix.accounts[0].is_signer and ix.accounts[0].is_writable  # owner
    assert not ix.accounts[2].is_writable  # perpetuals read-only
    assert ix.accounts[3].pubkey == pdas.JLP_POOL  # pool
    assert ix.accounts[4].is_writable  # position writable
    assert ix.accounts[5].is_writable  # positionRequest writable
    assert str(ix.accounts[7].pubkey) == EXPECTED_SOL_CUSTODY  # SOL custody
    assert ix.accounts[15].pubkey == pdas.JUPITER_PERPS_PROGRAM_ID  # program self-ref

    # Args round-trip
    arg_bytes = data[8:]
    assert struct.unpack("<Q", arg_bytes[0:8])[0] == 2_000_000
    assert struct.unpack("<Q", arg_bytes[8:16])[0] == 2_000_000
    assert arg_bytes[16] == pdas.SIDE_LONG
    assert struct.unpack("<Q", arg_bytes[17:25])[0] == 101_000_000
    assert arg_bytes[25] == 0  # None option
    assert struct.unpack("<Q", arg_bytes[26:34])[0] == COUNTER


def test_increase_ix_with_jupiter_min_out():
    args = IncreasePositionRequestArgs(
        size_usd_delta=10_000_000,
        collateral_token_delta=10_000_000,
        side=pdas.SIDE_SHORT,
        price_slippage=99_000_000,
        jupiter_minimum_out=9_900_000,
        counter=COUNTER,
    )
    ix = build_create_increase_position_market_request_ix(
        owner=OWNER,
        target_mint=pdas.SOL_MINT,
        side=pdas.SIDE_SHORT,
        args=args,
    )
    data = bytes(ix.data)
    # Now option byte = 1, then u64 follows
    assert len(data) == 8 + 42  # 34 + 8 extra for Some(u64)
    arg_bytes = data[8:]
    assert arg_bytes[16] == pdas.SIDE_SHORT
    assert arg_bytes[25] == 1  # Some option
    assert struct.unpack("<Q", arg_bytes[26:34])[0] == 9_900_000
    assert struct.unpack("<Q", arg_bytes[34:42])[0] == COUNTER


def test_decrease_ix_entire_position():
    args = DecreasePositionRequestArgs(
        collateral_usd_delta=0,
        size_usd_delta=0,
        price_slippage=99_000_000,
        jupiter_minimum_out=None,
        entire_position=True,
        counter=COUNTER + 1,
    )
    ix = build_create_decrease_position_market_request_ix(
        owner=OWNER,
        target_mint=pdas.SOL_MINT,
        side=pdas.SIDE_LONG,
        args=args,
    )
    data = bytes(ix.data)
    assert data[:8] == EXPECTED_DISC_DECREASE
    assert len(ix.accounts) == 16
    # Position is read-only on decrease (vs writable on increase)
    assert not ix.accounts[4].is_writable

    arg_bytes = data[8:]
    assert struct.unpack("<Q", arg_bytes[0:8])[0] == 0  # collateral_usd_delta
    assert struct.unpack("<Q", arg_bytes[8:16])[0] == 0  # size_usd_delta
    assert struct.unpack("<Q", arg_bytes[16:24])[0] == 99_000_000
    assert arg_bytes[24] == 0  # jup_min_out None
    assert arg_bytes[25] == 1  # entire_position Some
    assert arg_bytes[26] == 1  # entire_position True
    assert struct.unpack("<Q", arg_bytes[27:35])[0] == COUNTER + 1


def test_invalid_side_rejected():
    args = IncreasePositionRequestArgs(
        size_usd_delta=1, collateral_token_delta=1,
        side=99,  # invalid
        price_slippage=1, jupiter_minimum_out=None, counter=1,
    )
    try:
        args.encode()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_market_to_mint_mapping():
    assert pdas.get_target_mint("SOL") == pdas.SOL_MINT
    assert pdas.get_target_mint("SOL-PERP") == pdas.SOL_MINT
    assert pdas.get_target_mint("BTC") == pdas.BTC_MINT
    assert pdas.get_target_mint("ETH") == pdas.ETH_MINT
    try:
        pdas.get_target_mint("DOGE")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unsupported market")


if __name__ == "__main__":
    fns = [
        test_sol_custody_derivation,
        test_discriminators,
        test_increase_ix_layout,
        test_increase_ix_with_jupiter_min_out,
        test_decrease_ix_entire_position,
        test_invalid_side_rejected,
        test_market_to_mint_mapping,
    ]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print()
    print(f"{len(fns)-failed}/{len(fns)} passed")
    sys.exit(0 if failed == 0 else 1)
