"""Jupiter Perpetuals trading client (mainnet).

Async wrapper around Jupiter's on-chain Perpetuals program for the rest of the
bot. Mirrors agents/drift_client.py API so we can swap implementations.

Phase 2 wired 2026-04-26:
- open_perp_position: builds + (optionally) broadcasts createIncreasePositionMarketRequest
- close_perp_position: builds + (optionally) broadcasts createDecreasePositionMarketRequest
- Default mode is dry_run=True (returns the unsigned tx, NO broadcast)
- Real broadcasting requires dry_run=False AND confirm_real_money=True

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

from agents import jupiter_perp_pdas as pdas
from agents.jupiter_perp_ix import (
    DecreasePositionRequestArgs,
    IncreasePositionRequestArgs,
    build_create_decrease_position_market_request_ix,
    build_create_increase_position_market_request_ix,
    millis_counter,
)
from agents.jupiter_perp_polling import (
    PositionRequestState,
    decode_position_request,
    poll_position_request_executed,
)
from agents.jupiter_perp_tx import assemble_position_request_tx, estimate_total_fee

log = logging.getLogger("jupiter_perp")

JUPITER_PERPS_PROGRAM_ID = pdas.JUPITER_PERPS_PROGRAM_ID
USDC_MAINNET_MINT = pdas.USDC_MINT

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
    free_collateral_usd: float
    total_collateral_usd: float
    leverage: float
    sol_perp_base: float
    sol_perp_mark: float
    sol_perp_funding_hourly: float
    perp_account_exists: bool


@dataclass
class PerpOpenResult:
    """Returned by open_perp_position. Either dry-run preview or real broadcast result."""
    dry_run: bool
    market: str
    direction: str
    size_usd: float
    counter: int
    position_pda: str
    position_request_pda: str
    fee_estimate_usd: float
    serialized_tx_b64: str
    expected_signature: str
    # Only set when dry_run=False and broadcast succeeded:
    broadcast_signature: Optional[str] = None
    request_executed: bool = False


class JupiterPerpClient:
    """Thin async wrapper around Jupiter Perpetuals program for SOL-PERP."""

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
        if not self.wallet_path.exists():
            raise FileNotFoundError(f"wallet not found: {self.wallet_path}")
        self._keypair = _load_keypair(self.wallet_path)
        self._connection = AsyncClient(self.rpc_url)
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

    async def perp_account_exists(self, market: str = "SOL", side: int = pdas.SIDE_LONG) -> bool:
        """Check whether the user has a Position PDA on-chain for this market+side."""
        assert self._connection and self._keypair
        target_mint = pdas.get_target_mint(market)
        custody, _ = pdas.derive_custody_pda(target_mint)
        collateral_custody, _ = pdas.derive_custody_pda(pdas.USDC_MINT)
        pos_pda, _ = pdas.derive_position_pda(self.pubkey, custody, collateral_custody, side)
        info = await self._connection.get_account_info(pos_pda)
        return info.value is not None

    async def get_mark_price(self, market: str = "SOL") -> float:
        raise NotImplementedError("get_mark_price requires Pyth oracle read (TODO)")

    async def get_funding_rate(self, market: str = "SOL") -> float:
        raise NotImplementedError("get_funding_rate requires custody account parse (TODO)")

    async def snapshot(self) -> AccountSnapshot:
        assert self._initialized
        sol_balance = await self.get_native_sol_balance()
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

    # ── Write operations (Phase 2 wired 2026-04-26) ──────────────────────────

    async def open_perp_position(
        self,
        market: str,
        direction: str,
        size_usd: float,
        leverage: float = 1.0,
        slippage_bps: int = 100,
        mark_price: Optional[float] = None,
        dry_run: bool = True,
        confirm_real_money: bool = False,
    ) -> PerpOpenResult:
        """Build (and optionally broadcast) a createIncreasePositionMarketRequest tx.

        SAFETY: dry_run=True is default. Real broadcast requires both
        dry_run=False AND confirm_real_money=True.

        size_usd: notional position size in USD (scales with leverage).
        leverage: 1x..MAX. collateral_usd = size_usd / leverage.
        mark_price: required for priceSlippage calculation. Caller must fetch.
        slippage_bps: e.g. 100 = 1%.
        """
        assert self._initialized

        if dry_run is False and not confirm_real_money:
            raise PermissionError(
                "Refusing to broadcast without confirm_real_money=True. "
                "Set both dry_run=False AND confirm_real_money=True."
            )

        if mark_price is None or mark_price <= 0:
            raise ValueError("mark_price required (USD per base unit)")
        if size_usd <= 0:
            raise ValueError("size_usd must be > 0")
        if leverage < 1.0:
            raise ValueError("leverage must be >= 1.0")

        side = pdas.SIDE_LONG if direction.lower() == "long" else pdas.SIDE_SHORT
        target_mint = pdas.get_target_mint(market)
        collateral_usd = size_usd / leverage

        # Convert to USD lamports (×10^6 — both for size and USDC token)
        size_usd_delta = int(size_usd * 1_000_000)
        collateral_token_delta = int(collateral_usd * 1_000_000)

        # priceSlippage as USD lamports (per IDL — verify via smoke test)
        slip_factor = slippage_bps / 10_000
        if side == pdas.SIDE_LONG:
            price_slippage_usd = mark_price * (1 + slip_factor)
        else:
            price_slippage_usd = mark_price * (1 - slip_factor)
        price_slippage = int(price_slippage_usd * 1_000_000)

        counter = millis_counter()

        args = IncreasePositionRequestArgs(
            size_usd_delta=size_usd_delta,
            collateral_token_delta=collateral_token_delta,
            side=side,
            price_slippage=price_slippage,
            jupiter_minimum_out=None,
            counter=counter,
        )
        ix = build_create_increase_position_market_request_ix(
            owner=self.pubkey,
            target_mint=target_mint,
            side=side,
            args=args,
        )

        # Identify request PDA for caller logging + polling
        custody, _ = pdas.derive_custody_pda(target_mint)
        coll_custody, _ = pdas.derive_custody_pda(pdas.USDC_MINT)
        pos_pda, _ = pdas.derive_position_pda(self.pubkey, custody, coll_custody, side)
        req_pda, _ = pdas.derive_position_request_pda(pos_pda, counter, request_change=1)

        # Get recent blockhash (works for both dry_run and broadcast)
        bh_resp = await self._connection.get_latest_blockhash()
        recent_bh = bh_resp.value.blockhash

        assembled = assemble_position_request_tx(
            main_ix=ix,
            payer=self._keypair,
            recent_blockhash=recent_bh,
        )
        fee_info = estimate_total_fee()

        result = PerpOpenResult(
            dry_run=dry_run,
            market=market,
            direction=direction,
            size_usd=size_usd,
            counter=counter,
            position_pda=str(pos_pda),
            position_request_pda=str(req_pda),
            fee_estimate_usd=fee_info["sol_cost"] * mark_price if market.upper() == "SOL" else fee_info["sol_cost"] * 100,
            serialized_tx_b64=assembled.serialized_b64,
            expected_signature=assembled.expected_signature,
        )

        if dry_run:
            log.info(
                f"DRY RUN open_perp_position: market={market} dir={direction} size_usd={size_usd} "
                f"leverage={leverage} req_pda={req_pda} expected_sig={assembled.expected_signature}"
            )
            return result

        # ── Real broadcast path (only with confirm_real_money=True) ──
        log.warning(
            f"BROADCASTING open_perp_position: market={market} dir={direction} "
            f"size_usd={size_usd} req_pda={req_pda}"
        )
        send_resp = await self._connection.send_raw_transaction(
            bytes(assembled.versioned_tx),
        )
        result.broadcast_signature = str(send_resp.value)
        log.info(f"broadcast OK: {result.broadcast_signature}")

        # Poll keeper fulfillment
        state = await poll_position_request_executed(
            self._connection, req_pda, timeout_sec=15.0
        )
        result.request_executed = bool(state and state.executed)
        if not result.request_executed:
            log.warning(f"keeper did NOT fulfill within 15s for req_pda={req_pda}")
        return result

    async def close_perp_position(
        self,
        market: str = "SOL",
        side: int = pdas.SIDE_LONG,
        mark_price: Optional[float] = None,
        slippage_bps: int = 100,
        dry_run: bool = True,
        confirm_real_money: bool = False,
    ) -> PerpOpenResult:
        """Build (and optionally broadcast) a createDecreasePositionMarketRequest with entirePosition=True."""
        assert self._initialized

        if dry_run is False and not confirm_real_money:
            raise PermissionError(
                "Refusing to broadcast without confirm_real_money=True."
            )
        if mark_price is None or mark_price <= 0:
            raise ValueError("mark_price required")

        target_mint = pdas.get_target_mint(market)

        slip_factor = slippage_bps / 10_000
        if side == pdas.SIDE_LONG:
            # closing a long => sell at mark*(1-slip)
            price_slippage_usd = mark_price * (1 - slip_factor)
        else:
            price_slippage_usd = mark_price * (1 + slip_factor)
        price_slippage = int(price_slippage_usd * 1_000_000)

        counter = millis_counter()
        args = DecreasePositionRequestArgs(
            collateral_usd_delta=0,
            size_usd_delta=0,
            price_slippage=price_slippage,
            jupiter_minimum_out=None,
            entire_position=True,
            counter=counter,
        )
        ix = build_create_decrease_position_market_request_ix(
            owner=self.pubkey,
            target_mint=target_mint,
            side=side,
            args=args,
        )

        custody, _ = pdas.derive_custody_pda(target_mint)
        coll_custody, _ = pdas.derive_custody_pda(pdas.USDC_MINT)
        pos_pda, _ = pdas.derive_position_pda(self.pubkey, custody, coll_custody, side)
        req_pda, _ = pdas.derive_position_request_pda(pos_pda, counter, request_change=2)

        bh_resp = await self._connection.get_latest_blockhash()
        recent_bh = bh_resp.value.blockhash
        assembled = assemble_position_request_tx(
            main_ix=ix,
            payer=self._keypair,
            recent_blockhash=recent_bh,
        )
        fee_info = estimate_total_fee()

        result = PerpOpenResult(
            dry_run=dry_run,
            market=market,
            direction="close",
            size_usd=0.0,
            counter=counter,
            position_pda=str(pos_pda),
            position_request_pda=str(req_pda),
            fee_estimate_usd=fee_info["sol_cost"] * mark_price if market.upper() == "SOL" else fee_info["sol_cost"] * 100,
            serialized_tx_b64=assembled.serialized_b64,
            expected_signature=assembled.expected_signature,
        )

        if dry_run:
            log.info(f"DRY RUN close_perp_position: req_pda={req_pda}")
            return result

        log.warning(f"BROADCASTING close_perp_position: req_pda={req_pda}")
        send_resp = await self._connection.send_raw_transaction(
            bytes(assembled.versioned_tx),
        )
        result.broadcast_signature = str(send_resp.value)

        state = await poll_position_request_executed(
            self._connection, req_pda, timeout_sec=15.0
        )
        result.request_executed = bool(state and state.executed)
        return result

    async def deposit_usdc(self, amount_usd: float) -> str:
        """No-op kept for API compatibility — Jupiter bundles collateral into open ix."""
        log.info("deposit_usdc: no-op (Jupiter bundles collateral in open ix)")
        return ""
