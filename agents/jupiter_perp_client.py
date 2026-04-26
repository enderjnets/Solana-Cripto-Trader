"""Jupiter Perpetuals trading client (mainnet).

Async wrapper around Jupiter's on-chain Perpetuals program for the rest of the
bot. Mirrors agents/drift_client.py API so we can swap implementations.

Phase 1 scope: connect + read-only (snapshot, mark price, funding rate).
Phase 2: instantIncreasePosition / instantDecreasePosition for write ops.

External dependencies (TODO research):
- Program ID: PERPHjGBqRHArX4DySjwM6UJHiR3sWAatqfdBS2qQJu (TENTATIVE - verify)
- IDL: not officially published, parse from on-chain via anchorpy or use
  community repo https://github.com/julianfssen/jupiter-perps-anchor-idl-parsing
- No official Python SDK as of 2026-04. Build manual ix using anchorpy + IDL.

Environment:
- JUP_PERP_ENABLED:      "true" to enable (default false)
- JUP_PERP_RPC_URL:      Helius mainnet endpoint
- JUP_PERP_WALLET_PATH:  ~/.config/solana-jupiter-bot/wallet.json (shared with spot)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from solders.pubkey import Pubkey

log = logging.getLogger("jupiter_perp")

# ── On-chain constants (TODO verify) ─────────────────────────────────────────
# Source: search results 2026-04-25, needs on-chain verification
JUPITER_PERPS_PROGRAM_ID = Pubkey.from_string("PERPHjGBqRHArX4DySjwM6UJHiR3sWAatqfdBS2qQJu")

# JLP pool PDA (TODO derive from program ID + seed "pool")
JLP_POOL_PUBKEY: Optional[Pubkey] = None  # populated in initialize()

# Custody PDAs per asset (SOL, BTC, ETH, USDC, USDT)
# TODO: derive from program ID + seed "custody" + mint pubkey
CUSTODY_PUBKEYS: dict[str, Optional[Pubkey]] = {
    "SOL": None,
    "BTC": None,
    "ETH": None,
}

USDC_MAINNET_MINT = Pubkey.from_string("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")

_DEFAULT_RPC = "https://api.mainnet-beta.solana.com"
_DEFAULT_WALLET = Path.home() / ".config" / "solana-jupiter-bot" / "wallet.json"


def _load_keypair(path: Path) -> Keypair:
    """Load Solana keypair from JSON file (array of 64 bytes)."""
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "private_key" in raw:
        secret_bytes = bytes(raw["private_key"]) if isinstance(raw["private_key"], list) else bytes(json.loads(raw["private_key"]))
    else:
        secret_bytes = bytes(raw)
    return Keypair.from_bytes(secret_bytes)


@dataclass
class AccountSnapshot:
    """Mirror of drift_client.AccountSnapshot for API compatibility."""
    pubkey: str
    sol_balance: float
    free_collateral_usd: float       # USDC available for new positions
    total_collateral_usd: float
    leverage: float
    sol_perp_base: float             # signed: positive=long, negative=short
    sol_perp_mark: float
    sol_perp_funding_hourly: float
    perp_account_exists: bool        # has user opened Jupiter perp position before?


class JupiterPerpClient:
    """Thin async wrapper around Jupiter Perpetuals program for SOL-PERP.

    USAGE:
        client = JupiterPerpClient()
        await client.initialize()
        snap = await client.snapshot()
        # read-only OK
        await client.open_perp_position("SOL", "long", size_usd=2.0, leverage=1.5)  # TODO Phase 2
        await client.close()
    """

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        wallet_path: Optional[Path] = None,
    ):
        self.rpc_url = rpc_url or os.environ.get("JUP_PERP_RPC_URL", _DEFAULT_RPC)
        self.wallet_path = Path(wallet_path or os.environ.get("JUP_PERP_WALLET_PATH", str(_DEFAULT_WALLET)))
        self._connection: Optional[AsyncClient] = None
        self._keypair: Optional[Keypair] = None
        self._initialized = False

    @property
    def pubkey(self) -> Pubkey:
        assert self._keypair is not None
        return self._keypair.pubkey()

    async def initialize(self) -> None:
        """Connect RPC + load wallet + derive PDAs.

        TODO: derive JLP_POOL_PUBKEY and CUSTODY_PUBKEYS via Anchor PDA derivation
        (program_id + seed). Reference julianfssen/jupiter-perps-anchor-idl-parsing
        examples/get-pool-aum.ts for derivation logic.
        """
        if not self.wallet_path.exists():
            raise FileNotFoundError(f"wallet not found: {self.wallet_path}")
        self._keypair = _load_keypair(self.wallet_path)
        self._connection = AsyncClient(self.rpc_url)
        # TODO: verify program account exists on-chain
        info = await self._connection.get_account_info(JUPITER_PERPS_PROGRAM_ID)
        if info.value is None:
            raise RuntimeError(f"Jupiter Perps program not found on this RPC: {self.rpc_url}")
        self._initialized = True
        log.info(f"JupiterPerpClient connected: pubkey={self.pubkey}")

    async def close(self) -> None:
        if self._connection:
            await self._connection.close()

    async def get_native_sol_balance(self) -> float:
        assert self._connection and self._keypair
        resp = await self._connection.get_balance(self._keypair.pubkey())
        return resp.value / 1e9

    async def perp_account_exists(self, market: str = "SOL") -> bool:
        """Check if user has any position account on Jupiter Perps for this market.

        TODO: derive position PDA (program_id + seed "position" + owner + custody)
        and check if account exists on-chain.
        """
        # TODO Phase 2
        return False

    async def get_mark_price(self, market: str = "SOL") -> float:
        """Get current mark price from Jupiter pool oracle.

        TODO: read custody account, parse oracle price from price feed (Pyth).
        """
        # TODO Phase 2 — for now return placeholder
        raise NotImplementedError("get_mark_price requires custody PDA derivation + Pyth read")

    async def get_funding_rate(self, market: str = "SOL") -> float:
        """Get hourly funding rate (signed: positive = longs pay shorts).

        TODO: parse from custody account.funding_rate fields.
        """
        raise NotImplementedError("get_funding_rate requires custody account parse")

    async def snapshot(self) -> AccountSnapshot:
        """Read-only snapshot of perp account state.

        TODO: read position accounts for all markets in whitelist, aggregate.
        """
        assert self._initialized
        sol_balance = await self.get_native_sol_balance()
        # TODO: full implementation requires PDA derivation + position account parse
        return AccountSnapshot(
            pubkey=str(self.pubkey),
            sol_balance=sol_balance,
            free_collateral_usd=0.0,
            total_collateral_usd=0.0,
            leverage=0.0,
            sol_perp_base=0.0,
            sol_perp_mark=0.0,
            sol_perp_funding_hourly=0.0,
            perp_account_exists=False,
        )

    # ── Write operations (Phase 2) ────────────────────────────────────────────

    async def open_perp_position(
        self,
        market: str,
        direction: str,
        size_usd: float,
        leverage: float = 1.0,
        slippage_bps: int = 100,
    ) -> str:
        """Open a perp position with given size + leverage.

        TODO Phase 2:
        1. Derive position PDA
        2. Build instantIncreasePosition ix:
           - accounts: program_id, owner, position_pda, pool, custody (collateral),
             custody (target), pyth_oracle, custody_token_account, owner_token_account
           - args: { collateral_token, side: long|short, price, size_usd_delta,
             collateral_usd_delta, slippage }
        3. Sign + broadcast
        4. Return tx signature

        Reference: julianfssen/jupiter-perps-anchor-idl-parsing examples
        """
        raise NotImplementedError(
            "open_perp_position pending Phase 2 — needs IDL parse + ix building. "
            "Reference https://github.com/julianfssen/jupiter-perps-anchor-idl-parsing"
        )

    async def close_perp_position(
        self,
        market: str,
        slippage_bps: int = 100,
    ) -> str:
        """Close a perp position (reduce-only).

        TODO Phase 2: build instantDecreasePosition ix with size_delta=full position size.
        """
        raise NotImplementedError("close_perp_position pending Phase 2")

    async def deposit_usdc(self, amount_usd: float) -> str:
        """Deposit USDC as collateral via addLiquidity2 (becomes JLP) — NOT what we want.

        Note: Jupiter Perps doesn't have a separate "deposit collateral" step like Drift.
        Collateral is included in instantIncreasePosition call directly.
        This method is a no-op kept for API compatibility with drift_client.
        """
        log.info("deposit_usdc: Jupiter Perps doesn't require pre-deposit (collateral is bundled in open ix)")
        return ""
