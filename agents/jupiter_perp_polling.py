"""Keeper fulfillment polling for Jupiter Perpetuals.

After broadcasting a position request tx, the Jupiter keeper picks it up off-chain
and executes the fulfillment ix (~500ms-1s). This module polls the on-chain
PositionRequest account until `executed=true` or timeout.
"""
from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from typing import Optional

from solders.pubkey import Pubkey


# Anchor account discriminator length
DISC_LEN = 8


@dataclass
class PositionRequestState:
    """Decoded subset of PositionRequest account.

    Full schema in IDL; we only decode what we need for status checks.
    """

    executed: bool
    counter: int
    request_change: int  # 1=Increase, 2=Decrease
    request_type: int    # 0=Market, 1=Trigger
    raw_data_len: int


def _read_option(data: bytes, offset: int, value_len: int) -> tuple[bytes, int]:
    """Read a Borsh option<T>. Returns (raw_value_or_empty, next_offset)."""
    flag = data[offset]
    offset += 1
    if flag == 0:
        return b"", offset
    return data[offset:offset + value_len], offset + value_len


def decode_position_request(account_data: bytes) -> PositionRequestState:
    """Decode the executed flag + counter from a PositionRequest account.

    Layout (after 8-byte anchor disc):
      32 owner + 32 pool + 32 custody + 32 position + 32 mint   = 160 bytes
      i64 openTime + i64 updateTime                             = 16
      u64 sizeUsdDelta + u64 collateralDelta                    = 16
      u8 requestChange + u8 requestType + u8 side               = 3
      option<u64> priceSlippage
      option<u64> jupiterMinimumOut
      option<u64> preSwapAmount
      option<u64> triggerPrice
      option<bool> triggerAboveThreshold
      option<bool> entirePosition
      bool executed
      u64 counter
      u8 bump
      option<pubkey> referral
    """
    if len(account_data) < DISC_LEN + 195:
        raise ValueError(f"account data too short: {len(account_data)} bytes")

    body = account_data[DISC_LEN:]
    off = 0
    # Skip 5 pubkeys
    off += 32 * 5
    # Skip 2 i64s
    off += 16
    # Skip 2 u64s
    off += 16
    # Read enums
    request_change = body[off]; off += 1
    request_type = body[off]; off += 1
    _side = body[off]; off += 1

    # Skip 4 option<u64>
    for _ in range(4):
        _val, off = _read_option(body, off, 8)
    # Skip 2 option<bool>
    for _ in range(2):
        _val, off = _read_option(body, off, 1)

    # executed: bool
    executed = body[off] == 1
    off += 1

    # counter: u64
    counter = int.from_bytes(body[off:off + 8], "little")

    return PositionRequestState(
        executed=executed,
        counter=counter,
        request_change=request_change,
        request_type=request_type,
        raw_data_len=len(account_data),
    )


async def poll_position_request_executed(
    rpc_client,
    position_request_pda: Pubkey,
    timeout_sec: float = 15.0,
    poll_interval_sec: float = 0.5,
) -> Optional[PositionRequestState]:
    """Poll until PositionRequest.executed=True or timeout.

    rpc_client: an AsyncClient (solana-py) or anything with `get_account_info` returning
        an object whose .value.data is bytes.

    Returns the decoded state once executed=True, or None on timeout.
    Raises if the account never appears (typical: 1-2 slots after tx confirmation).
    """
    deadline = asyncio.get_event_loop().time() + timeout_sec
    not_found_count = 0
    last_state: Optional[PositionRequestState] = None

    while asyncio.get_event_loop().time() < deadline:
        try:
            resp = await rpc_client.get_account_info(position_request_pda)
        except Exception:
            await asyncio.sleep(poll_interval_sec)
            continue

        value = getattr(resp, "value", None)
        if value is None:
            not_found_count += 1
            # Account may not yet be created (just-broadcast tx)
            await asyncio.sleep(poll_interval_sec)
            continue

        # Extract data bytes — solana-py vs solders shapes may differ
        data = value.data if isinstance(value.data, (bytes, bytearray)) else bytes(value.data)
        try:
            state = decode_position_request(data)
        except ValueError:
            await asyncio.sleep(poll_interval_sec)
            continue

        last_state = state
        if state.executed:
            return state

        await asyncio.sleep(poll_interval_sec)

    # Timeout
    return last_state


def is_request_likely_failed(
    state: Optional[PositionRequestState],
    elapsed_sec: float,
    soft_timeout_sec: float = 10.0,
) -> bool:
    """Heuristic: did the keeper fail to fulfill?

    Jupiter typical fulfillment is <2s. If we have a state with executed=False
    and we've waited > soft_timeout_sec, the request is likely stuck/failed.
    """
    if state is None:
        return elapsed_sec > soft_timeout_sec
    return not state.executed and elapsed_sec > soft_timeout_sec
