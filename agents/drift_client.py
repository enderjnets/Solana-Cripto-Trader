"""Drift Protocol live trading client (devnet / mainnet).

Wraps driftpy so the rest of the bot can treat Drift like the paper simulator.
Phase 1 scope: connect + read-only (balance, mark price, funding, position).
Write ops land in Phase 3.

Environment:
- DRIFT_ENV:        "devnet" (default) or "mainnet"
- DRIFT_RPC_URL:    optional override (else public RPC for the env)
- DRIFT_WALLET_PATH: path to keypair JSON (default ~/.config/solana-drift-bot/id.json)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from anchorpy import Wallet
from driftpy.account_subscription_config import AccountSubscriptionConfig
from driftpy.constants.numeric_constants import (
    BASE_PRECISION,
    FUNDING_RATE_PRECISION,
    PRICE_PRECISION,
    QUOTE_PRECISION,
)
from driftpy.addresses import get_user_account_public_key
from driftpy.constants.config import get_markets_and_oracles
from driftpy.drift_client import DriftClient
from driftpy.drift_user import DriftUser
from driftpy.types import (
    MarketType,
    OrderParams,
    OrderType,
    PositionDirection,
    PostOnlyParams,
)
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from spl.token.instructions import get_associated_token_address

USDC_DEVNET_MINT = Pubkey.from_string("8zGuJQqwhZafTah7Uc7Z4tXRnguqkn5KLFAP8oV6PHe2")
USDC_MAINNET_MINT = Pubkey.from_string("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")

log = logging.getLogger("drift_client")

SOL_PERP_MARKET_INDEX = 0
USDC_SPOT_MARKET_INDEX = 0

_DEFAULT_RPC = {
    "devnet": "https://api.devnet.solana.com",
    "mainnet": "https://api.mainnet-beta.solana.com",
}
_DEFAULT_WALLET = Path.home() / ".config" / "solana-jupiter-bot" / "wallet.json"


def _load_keypair(path: Path) -> Keypair:
    """Load Solana keypair from file. Supports two formats:
    1. Standard Solana CLI: JSON array of 64 ints, e.g. [12, 34, ..., 99]
    2. Jupiter-bot custom: dict {address, private_key, network} where
       private_key can be a list of ints or a JSON-array-as-string "[12,34,...]"
       or a base58-encoded string.
    """
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return Keypair.from_bytes(bytes(data))
    if isinstance(data, dict) and "private_key" in data:
        pk = data["private_key"]
        if isinstance(pk, list):
            return Keypair.from_bytes(bytes(pk))
        if isinstance(pk, str):
            pk_str = pk.strip()
            if pk_str.startswith("["):
                return Keypair.from_bytes(bytes(json.loads(pk_str)))
            try:
                import base58
                return Keypair.from_bytes(base58.b58decode(pk_str))
            except Exception as e:
                raise ValueError(f"private_key string not JSON array or base58: {e}")
    raise ValueError(f"unsupported wallet format in {path}")


@dataclass
class AccountSnapshot:
    pubkey: str
    sol_balance: float             # native SOL on the wallet
    free_collateral_usd: float     # USDC free collateral inside Drift
    total_collateral_usd: float
    leverage: float
    sol_perp_base: float           # signed: positive=long, negative=short
    sol_perp_mark: float           # oracle price in USD
    sol_perp_funding_hourly: float # fraction (0.0001 = 0.01%/hr), signed
    drift_user_exists: bool


class LiveDriftClient:
    """Thin async wrapper around driftpy.DriftClient for SOL-PERP.

    Call `await client.initialize()` before use and `await client.close()` when done.
    """

    def __init__(
        self,
        env: str | None = None,
        rpc_url: str | None = None,
        wallet_path: Path | None = None,
    ):
        self.env = env or os.environ.get("DRIFT_ENV", "devnet")
        if self.env not in ("devnet", "mainnet"):
            raise ValueError(f"DRIFT_ENV must be devnet|mainnet, got {self.env!r}")
        self.rpc_url = rpc_url or os.environ.get("DRIFT_RPC_URL", _DEFAULT_RPC[self.env])
        self.wallet_path = Path(
            wallet_path
            or os.environ.get("DRIFT_WALLET_PATH", str(_DEFAULT_WALLET))
        )

        self._keypair: Keypair | None = None
        self._wallet: Wallet | None = None
        self._connection: AsyncClient | None = None
        self._drift_client: DriftClient | None = None
        self._drift_user: DriftUser | None = None

    @property
    def pubkey(self) -> Pubkey:
        assert self._keypair is not None, "call initialize() first"
        return self._keypair.pubkey()

    async def initialize(self) -> None:
        self._keypair = _load_keypair(self.wallet_path)
        self._wallet = Wallet(self._keypair)
        self._connection = AsyncClient(self.rpc_url)

        spot_oracles, perp_oracles, spot_indexes = get_markets_and_oracles(
            env=self.env, perp_markets=[SOL_PERP_MARKET_INDEX]
        )
        self._drift_client = DriftClient(
            self._connection,
            self._wallet,
            self.env,
            perp_market_indexes=[SOL_PERP_MARKET_INDEX],
            spot_market_indexes=spot_indexes,
            oracle_infos=spot_oracles + perp_oracles,
            account_subscription=AccountSubscriptionConfig("cached"),
        )
        await self._drift_client.subscribe()
        self._user_pk = get_user_account_public_key(
            self._drift_client.program_id, self.pubkey, 0
        )
        # If user account exists on-chain, subscribe so balance reads work.
        # If not, we'll subscribe lazily after initialize_user().
        info = await self._connection.get_account_info(self._user_pk)
        if info.value is not None:
            await self._drift_client.add_user(0)
            self._drift_user = self._drift_client.get_user(0)
        else:
            self._drift_user = None
        log.info(f"LiveDriftClient connected: env={self.env} pubkey={self.pubkey}")

    async def close(self) -> None:
        if self._drift_client is not None:
            await self._drift_client.unsubscribe()
        if self._connection is not None:
            await self._connection.close()

    @property
    def usdc_mint(self) -> Pubkey:
        return USDC_DEVNET_MINT if self.env == "devnet" else USDC_MAINNET_MINT

    @property
    def usdc_ata(self) -> Pubkey:
        return get_associated_token_address(self.pubkey, self.usdc_mint)

    async def get_native_sol_balance(self) -> float:
        """Native SOL balance on the wallet (not inside Drift)."""
        assert self._connection is not None
        resp = await self._connection.get_balance(self.pubkey)
        return resp.value / 1_000_000_000  # lamports → SOL

    async def drift_user_exists(self) -> bool:
        """Check if the Drift user PDA account is initialized on-chain."""
        assert self._connection is not None
        info = await self._connection.get_account_info(self._user_pk)
        return info.value is not None

    async def initialize_user(self) -> str:
        """Create the Drift user + user_stats accounts (sub_account 0). Idempotent."""
        assert self._drift_client is not None
        if await self.drift_user_exists():
            log.info("Drift user already initialized")
            if self._drift_user is None:
                await self._drift_client.add_user(0)
                self._drift_user = self._drift_client.get_user(0)
            return ""
        sig = await self._drift_client.initialize_user(sub_account_id=0, name="live-bot")
        log.info(f"initialize_user tx: {sig}")
        await self._drift_client.add_user(0)
        self._drift_user = self._drift_client.get_user(0)
        return str(sig)

    def _market_auction_prices(
        self, direction: str, oracle_price_raw: int, slippage_bps: int = 100
    ) -> tuple[int, int]:
        """Return (start, end) auction prices in PRICE_PRECISION for a market order.

        LONG: auction climbs from oracle to oracle*(1+slip) — worst fill at top.
        SHORT: auction drops from oracle to oracle*(1-slip) — worst fill at bottom.
        Default slippage 1% — generous enough to fill on devnet vAMM.
        """
        if direction == "long":
            start = oracle_price_raw
            end = oracle_price_raw + (oracle_price_raw * slippage_bps) // 10_000
        else:
            start = oracle_price_raw
            end = oracle_price_raw - (oracle_price_raw * slippage_bps) // 10_000
        return start, end

    async def open_sol_perp_market(
        self,
        direction: str,
        size_sol: float,
        market_index: int = SOL_PERP_MARKET_INDEX,
        slippage_bps: int = 100,
    ) -> str:
        """Open a SOL-PERP position with a market order. Returns tx signature.

        direction    : "long" or "short"
        size_sol     : position size in SOL (0.1 = 0.1 SOL ≈ $8 notional at $80)
        slippage_bps : max slippage on auction end price (100 = 1%)
        """
        assert self._drift_client is not None
        oracle = self._drift_client.get_oracle_price_data_for_perp_market(market_index)
        start, end = self._market_auction_prices(direction, oracle.price, slippage_bps)
        dir_enum = (
            PositionDirection.Long() if direction == "long" else PositionDirection.Short()
        )
        params = OrderParams(
            order_type=OrderType.Market(),
            base_asset_amount=int(size_sol * BASE_PRECISION),
            market_index=market_index,
            direction=dir_enum,
            market_type=MarketType.Perp(),
            post_only=PostOnlyParams.NONE(),
            auction_duration=10,
            auction_start_price=start,
            auction_end_price=end,
        )
        result = await self._drift_client.place_and_take_perp_order(params)
        sig = str(getattr(result, "tx_sig", result))
        log.info(f"open {direction} {size_sol} SOL @ oracle ${oracle.price/PRICE_PRECISION:.2f} "
                 f"auction=[{start/PRICE_PRECISION:.2f}..{end/PRICE_PRECISION:.2f}] tx: {sig}")
        return sig

    async def open_sol_perp_limit(
        self,
        direction: str,
        size_sol: float,
        limit_price: float,
        timeout_seconds: int = 60,
        market_index: int = SOL_PERP_MARKET_INDEX,
    ) -> tuple[str, bool, float]:
        """Place a SOL-PERP limit order and wait up to timeout_seconds for a fill.

        On timeout the order auto-expires via max_ts; we also send an explicit
        cancel as belt-and-suspenders so the user account is clean.

        Returns (tx_sig, filled, filled_size_sol). filled_size_sol is 0 if the
        order timed out with no partial fill.
        """
        assert self._drift_client is not None and self._drift_user is not None
        dir_enum = (
            PositionDirection.Long() if direction == "long" else PositionDirection.Short()
        )
        price_int = int(limit_price * PRICE_PRECISION)
        max_ts = int(time.time() + timeout_seconds)
        size_base = int(size_sol * BASE_PRECISION)
        params = OrderParams(
            order_type=OrderType.Limit(),
            base_asset_amount=size_base,
            market_index=market_index,
            direction=dir_enum,
            market_type=MarketType.Perp(),
            price=price_int,
            post_only=PostOnlyParams.NONE(),
            max_ts=max_ts,
        )
        result = await self._drift_client.place_perp_order(params)
        sig = str(getattr(result, "tx_sig", result))
        log.info(
            f"limit {direction} {size_sol} SOL @ ${limit_price:.4f} "
            f"(ttl {timeout_seconds}s) tx: {sig}"
        )

        initial_base = 0
        pos = self._drift_user.get_perp_position(market_index)
        if pos is not None:
            initial_base = pos.base_asset_amount

        deadline = time.time() + timeout_seconds + 3
        poll_interval = 2.0
        filled_base = 0
        while time.time() < deadline:
            await asyncio.sleep(poll_interval)
            pos = self._drift_user.get_perp_position(market_index)
            current_base = pos.base_asset_amount if pos is not None else 0
            delta = current_base - initial_base
            # For a LONG fill delta > 0; for a SHORT fill delta < 0. Use abs.
            if (direction == "long" and delta > 0) or (direction == "short" and delta < 0):
                filled_base = abs(delta)
                if filled_base >= size_base:
                    break

        filled = filled_base > 0
        if not filled:
            log.info("limit order timed out — cancelling explicitly")
            try:
                await self._drift_client.cancel_orders(
                    market_type=MarketType.Perp(), market_index=market_index
                )
            except Exception as e:
                log.debug(f"cancel after timeout failed (may already be expired): {e}")

        return sig, filled, filled_base / BASE_PRECISION

    async def close_sol_perp(
        self,
        market_index: int = SOL_PERP_MARKET_INDEX,
        slippage_bps: int = 100,
    ) -> str | None:
        """Close existing SOL-PERP position with a reduce-only market order.

        Returns tx signature, or None if there was no open position.
        """
        assert self._drift_client is not None and self._drift_user is not None
        pos = self._drift_user.get_perp_position(market_index)
        if pos is None or pos.base_asset_amount == 0:
            log.info("no open position to close")
            return None

        # Opposite direction of current position.
        is_long = pos.base_asset_amount > 0
        close_direction = "short" if is_long else "long"
        dir_enum = (
            PositionDirection.Short() if is_long else PositionDirection.Long()
        )
        oracle = self._drift_client.get_oracle_price_data_for_perp_market(market_index)
        start, end = self._market_auction_prices(close_direction, oracle.price, slippage_bps)
        params = OrderParams(
            order_type=OrderType.Market(),
            base_asset_amount=abs(pos.base_asset_amount),
            market_index=market_index,
            direction=dir_enum,
            market_type=MarketType.Perp(),
            post_only=PostOnlyParams.NONE(),
            reduce_only=True,
            auction_duration=10,
            auction_start_price=start,
            auction_end_price=end,
        )
        result = await self._drift_client.place_and_take_perp_order(params)
        sig = str(getattr(result, "tx_sig", result))
        log.info(f"close tx: {sig}")
        return sig

    async def deposit_usdc(self, amount_usd: float) -> str:
        """Deposit USDC from our ATA into Drift spot market 0.

        amount_usd is a float in USD units (e.g. 500.0 = 500 USDC).
        """
        assert self._drift_client is not None
        base_units = int(amount_usd * QUOTE_PRECISION)
        result = await self._drift_client.deposit(
            amount=base_units,
            spot_market_index=USDC_SPOT_MARKET_INDEX,
            user_token_account=self.usdc_ata,
            sub_account_id=0,
        )
        sig = str(result.tx_sig)
        log.info(f"deposit {amount_usd} USDC tx: {sig}")
        return sig

    async def get_mark_price(self, market_index: int = SOL_PERP_MARKET_INDEX) -> float:
        assert self._drift_client is not None
        oracle = self._drift_client.get_oracle_price_data_for_perp_market(market_index)
        return oracle.price / PRICE_PRECISION

    async def get_hourly_funding_rate(self, market_index: int = SOL_PERP_MARKET_INDEX) -> float:
        """Return last hourly funding rate as signed fraction (0.0001 = 0.01%/hr).

        Positive → longs pay shorts. Negative → shorts pay longs.
        Formula: last_funding_rate / FUNDING_RATE_PRECISION / oracle_price,
        which yields the per-funding-period rate relative to notional.
        """
        assert self._drift_client is not None
        market = self._drift_client.get_perp_market_account(market_index)
        oracle = self._drift_client.get_oracle_price_data_for_perp_market(market_index)
        oracle_price = oracle.price / PRICE_PRECISION
        if oracle_price <= 0:
            return 0.0
        raw = market.amm.last_funding_rate / FUNDING_RATE_PRECISION
        return raw / oracle_price

    async def snapshot(self) -> AccountSnapshot:
        """One-shot account state (native SOL, collateral, position, mark)."""
        assert self._drift_user is not None
        sol = await self.get_native_sol_balance()
        mark = await self.get_mark_price()
        funding = await self.get_hourly_funding_rate()

        exists = await self.drift_user_exists()
        free_coll = total_coll = leverage = 0.0
        sol_base = 0.0
        if exists and self._drift_user is not None:
            try:
                free_coll = self._drift_user.get_free_collateral() / QUOTE_PRECISION
                total_coll = self._drift_user.get_total_collateral() / QUOTE_PRECISION
                leverage = self._drift_user.get_leverage() / 10_000
                pos = self._drift_user.get_perp_position(SOL_PERP_MARKET_INDEX)
                if pos is not None:
                    sol_base = pos.base_asset_amount / BASE_PRECISION
            except Exception as e:
                log.debug(f"drift_user read failed: {e}")

        return AccountSnapshot(
            pubkey=str(self.pubkey),
            sol_balance=sol,
            free_collateral_usd=free_coll,
            total_collateral_usd=total_coll,
            leverage=leverage,
            sol_perp_base=sol_base,
            sol_perp_mark=mark,
            sol_perp_funding_hourly=funding,
            drift_user_exists=exists,
        )
