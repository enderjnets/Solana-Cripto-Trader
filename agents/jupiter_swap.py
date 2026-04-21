"""
jupiter_swap.py — Jupiter v6 Swap API wrapper (Sprint 2 Fase 2)

Full swap flow: quote → build → sign → broadcast → confirm.

Defense-in-depth:
- Pre-quote: TRADE_WHITELIST_STRICT validated upstream in executor.py
- Post-quote: priceImpactPct validated against MAX_SLIPPAGE_BPS via safety module
- On-chain: slippageBps passed to Jupiter — the swap REVERTS if exceeded
- Retries: transient failures (network, RPC) backoff up to 3 times; quote errors (no route) fail fast

Usage:
    from agents.jupiter_swap import JupiterSwap
    from agents.solana_rpc import get_rpc
    from agents.wallet import load_wallet

    w = load_wallet()
    r = get_rpc()
    swap = JupiterSwap(wallet=w, rpc=r)

    result = swap.execute_swap(
        input_mint=MINT_USDC,
        output_mint=MINT_SOL,
        amount_lamports=1_000_000,   # 1 USDC (USDC has 6 decimals)
        slippage_bps=100,
    )
    # result.signature, result.confirmed, result.out_amount, result.price_impact_pct
"""
from __future__ import annotations

import os
import json
import base64
import logging
import time
from dataclasses import dataclass
from typing import Optional, Any

import requests

log = logging.getLogger('jupiter_swap')

DEFAULT_QUOTE_URL = os.environ.get("JUPITER_QUOTE_URL", "https://lite-api.jup.ag/swap/v1/quote")
DEFAULT_SWAP_URL  = os.environ.get("JUPITER_SWAP_URL",  "https://lite-api.jup.ag/swap/v1/swap")
DEFAULT_HTTP_TIMEOUT = 20
DEFAULT_RETRIES = 3


@dataclass
class SwapResult:
    """Result of a swap execution. All fields may be None on failure."""
    signature: Optional[str] = None
    confirmed: bool = False
    in_amount: int = 0              # input amount in lamports/smallest-unit
    out_amount: int = 0             # output amount received
    price_impact_pct: float = 0.0   # percentage (e.g., 0.15 = 0.15%)
    route_plan_steps: int = 0       # number of hops in the route
    error: Optional[str] = None     # human-readable error if failed
    raw_quote: Optional[dict] = None

    @property
    def success(self) -> bool:
        return self.confirmed and self.signature is not None and self.error is None


class JupiterSwap:
    """Client for Jupiter v6 Swap API + signing via provided wallet."""

    def __init__(self, wallet, rpc,
                 quote_url: Optional[str] = None,
                 swap_url: Optional[str] = None):
        self.wallet = wallet
        self.rpc = rpc
        self.quote_url = quote_url or DEFAULT_QUOTE_URL
        self.swap_url = swap_url or DEFAULT_SWAP_URL

    # ── Step 1: GET quote ──
    def get_quote(self, input_mint: str, output_mint: str,
                   amount_lamports: int, slippage_bps: int = 100,
                   only_direct_routes: bool = False) -> Optional[dict]:
        """Get a swap quote from Jupiter. Returns None if no route."""
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount_lamports),
            "slippageBps": str(slippage_bps),
            "onlyDirectRoutes": str(only_direct_routes).lower(),
        }
        last_err: Optional[Exception] = None
        for attempt in range(DEFAULT_RETRIES):
            try:
                r = requests.get(self.quote_url, params=params, timeout=DEFAULT_HTTP_TIMEOUT)
                if r.status_code in (400, 404):
                    log.info(f"Jupiter: no route for {input_mint[:8]}→{output_mint[:8]} amount={amount_lamports}")
                    return None
                r.raise_for_status()
                data = r.json()
                data["_fetched_at"] = time.time()  # v2.12.10: stale detection
                return data
            except Exception as e:
                last_err = e
                if attempt < DEFAULT_RETRIES - 1:
                    time.sleep(min(2 ** attempt, 4))
        log.warning(f"Jupiter get_quote failed after {DEFAULT_RETRIES} retries: {last_err}")
        return None

    # ── Step 2: POST /swap to build signed transaction skeleton ──
    def build_swap_transaction(self, quote: dict, priority_fee_microlamports: Optional[int] = None,
                                 wrap_unwrap_sol: bool = True,
                                 priority_level: str = "high") -> Optional[str]:  # v2.12.10 passthrough
        """POST /swap with the quote + our pubkey → returns base64-encoded unsigned tx.
        Returns None on failure."""
        if self.wallet.pubkey in (None, "", "MOCK_NO_PUBKEY"):
            raise ValueError("Wallet has no pubkey — cannot build swap tx")

        body: dict[str, Any] = {
            "quoteResponse": quote,
            "userPublicKey": self.wallet.pubkey,
            "wrapAndUnwrapSol": wrap_unwrap_sol,
            "dynamicComputeUnitLimit": True,
        }
        if priority_fee_microlamports is not None:
            # Jupiter expects prioritizationFeeLamports in LAMPORTS (not microlamports)
            # Convert: total_fee_lamports = (microlamports_per_CU * CU_used) / 1e6
            # Without knowing CU, approximate with a cap (Jupiter will cap internally)
            # v2.12.10: use caller-provided priority_level (was hardcoded "high")
            body["prioritizationFeeLamports"] = {
                "priorityLevelWithMaxLamports": {
                    "maxLamports": 1_000_000,   # 0.001 SOL cap
                    "priorityLevel": priority_level,
                }
            }

        try:
            r = requests.post(self.swap_url, json=body, timeout=DEFAULT_HTTP_TIMEOUT)
            r.raise_for_status()
            d = r.json()
            return d.get("swapTransaction")
        except Exception as e:
            log.warning(f"Jupiter build_swap_transaction failed: {e}")
            return None

    # ── Step 3: sign base64 tx using our wallet ──
    def sign_transaction(self, swap_tx_b64: str) -> Optional[str]:
        """Decode base64 → deserialize VersionedTransaction → sign → reserialize base64.
        Returns None on failure."""
        try:
            from solders.transaction import VersionedTransaction
            from solders.keypair import Keypair

            raw = base64.b64decode(swap_tx_b64)
            tx = VersionedTransaction.from_bytes(raw)

            # Jupiter returns an unsigned VersionedTransaction. We sign using our keypair.
            # The wallet must be LiveWallet with _keypair attribute.
            if not getattr(self.wallet, "is_live", False):
                raise RuntimeError("Cannot sign with MockWallet (LIVE_TRADING_ENABLED must be true)")
            if not hasattr(self.wallet, "_keypair"):
                raise RuntimeError("LiveWallet missing _keypair attribute")

            # Sign: replace the dummy signatures with real ones
            signed_tx = VersionedTransaction(tx.message, [self.wallet._keypair])
            return base64.b64encode(bytes(signed_tx)).decode("ascii")
        except Exception as e:
            log.error(f"sign_transaction failed: {e}")
            return None

    # ── Step 4+5: Broadcast + Confirm (via solana_rpc) ──
    def broadcast_and_confirm(self, signed_tx_b64: str, timeout_sec: int = 30) -> tuple[Optional[str], bool]:
        """v2.12.10 broadcast retry: 3 attempts with backoff. Slippage errors (6024) skip retry."""
        last_exc = None
        last_err_str = ""
        for attempt in range(3):
            try:
                sig = self.rpc.send_transaction(signed_tx_b64, skip_preflight=False,
                                                preflight_commitment="confirmed")
                confirmed = self.rpc.confirm_transaction(sig, timeout_sec=timeout_sec)
                return sig, confirmed
            except Exception as e:
                last_exc = e
                last_err_str = str(e)
                # Slippage / output errors need a fresh quote, not a retry of same signed tx
                if ("Custom(6024)" in last_err_str or "0x1788" in last_err_str
                        or "slippage" in last_err_str.lower()
                        or "insufficient" in last_err_str.lower()):
                    log.warning(f"broadcast slippage/output error — no retry (needs fresh quote): {last_err_str[:200]}")
                    break
                if attempt < 2:
                    backoff = min(2 ** attempt, 4)
                    log.info(f"broadcast retry {attempt+1}/3 after {backoff}s: {last_err_str[:120]}")
                    time.sleep(backoff)
                    continue
        # Propagate detailed error so caller can decide on fresh-quote retry
        self._last_broadcast_error = last_err_str
        log.error(f"broadcast_and_confirm failed after retries: {last_exc}")
        return None, False

    # ── High-level flow: execute_swap ──
    def execute_swap(self, input_mint: str, output_mint: str,
                      amount_lamports: int,
                      slippage_bps: int = 100,
                      priority_fee_level: str = 'medium',
                      dry_run: bool = False) -> SwapResult:
        """Full swap: quote → validate slippage via safety → build → sign → broadcast → confirm.

        If dry_run=True: stops after quote + slippage check, returns result with success=False
        but populated fields. Useful for testing without spending gas.
        """
        result = SwapResult()

        # Step 1: Quote
        quote = self.get_quote(input_mint, output_mint, amount_lamports, slippage_bps)
        if not quote:
            result.error = "no_route"
            return result

        try:
            result.in_amount = int(quote.get("inAmount", 0))
            result.out_amount = int(quote.get("outAmount", 0))
            result.price_impact_pct = float(quote.get("priceImpactPct", 0))
            result.route_plan_steps = len(quote.get("routePlan", []))
            result.raw_quote = quote
        except Exception as e:
            result.error = f"quote_parse_error: {e}"
            return result

        # Step 2: Safety slippage check (defense-in-depth beyond on-chain revert)
        try:
            import safety
            max_slippage_bps = safety.MAX_SLIPPAGE_BPS
        except Exception:
            max_slippage_bps = slippage_bps

        if max_slippage_bps > 0:
            actual_bps = int(abs(result.price_impact_pct) * 100)  # 0.15% → 15 bps
            if actual_bps > max_slippage_bps:
                result.error = f"price_impact_exceeds_max_slippage:{actual_bps}bps>{max_slippage_bps}bps"
                log.warning(f"Slippage rejected: {actual_bps}bps > {max_slippage_bps}bps")
                return result

        if dry_run:
            log.info(f"DRY RUN: quote OK (out={result.out_amount}, impact={result.price_impact_pct:.4f}%)")
            return result

        # v2.12.10: stale quote detection — refresh if >3s old
        quote_age = time.time() - float(quote.get("_fetched_at", time.time()))
        if quote_age > 3.0:
            log.info(f"    quote age {quote_age:.1f}s > 3s — refreshing before build")
            quote = self.get_quote(input_mint, output_mint, amount_lamports, slippage_bps)
            if not quote:
                result.error = "no_route_on_refresh"
                return result
            result.raw_quote = quote
            result.in_amount = int(quote.get("inAmount", 0))
            result.out_amount = int(quote.get("outAmount", 0))

        # Step 3: Priority fee (microlamports/CU via Helius — informational)
        try:
            priority_fee = self.rpc.get_priority_fee_estimate(level=priority_fee_level)
        except Exception:
            priority_fee = None

        # v2.12.10: map caller priority_fee_level to Jupiter priorityLevel
        # valid Jupiter values: "min", "low", "medium", "high", "veryHigh"
        _jup_level = priority_fee_level if priority_fee_level in ("min","low","medium","high","veryHigh") else "high"

        # Step 4: Build swap tx
        swap_tx_b64 = self.build_swap_transaction(quote, priority_fee_microlamports=priority_fee,
                                                   priority_level=_jup_level)
        if not swap_tx_b64:
            result.error = "build_swap_tx_failed"
            return result

        # Step 5: Sign (requires LiveWallet — raises on MockWallet)
        signed_tx_b64 = self.sign_transaction(swap_tx_b64)
        if not signed_tx_b64:
            result.error = "sign_tx_failed"
            return result

        # Step 6: Broadcast + confirm (broadcast_and_confirm has internal retry for transient)
        sig, confirmed = self.broadcast_and_confirm(signed_tx_b64)
        result.signature = sig
        result.confirmed = confirmed
        if not confirmed:
            result.error = "tx_not_confirmed" if sig else "broadcast_failed"

        # v2.12.10 A2 / v2.12.17: fresh-quote retry on slippage error (6024) with escalating ladder
        _broadcast_err = getattr(self, "_last_broadcast_error", "") or ""
        if (not confirmed and not sig
                and ("Custom(6024)" in _broadcast_err or "0x1788" in _broadcast_err
                     or "slippage" in _broadcast_err.lower())):
            # v2.12.17: escalating retry ladder (caps at 2000bps = 20% safety)
            # ETH Wormhole thin liquidity → needs bigger range than SOL/JUP
            retry_bps_ladder = [
                max(500, min(slippage_bps * 2, 2000)),
                min(max(slippage_bps * 8, 1500), 2000),
            ]
            # Dedup + skip values not above initial
            seen = set()
            retry_bps_ladder = [b for b in retry_bps_ladder
                                if b > slippage_bps and not (b in seen or seen.add(b))]

            for retry_bps in retry_bps_ladder:
                log.warning(f"🔄 fresh-quote retry ({retry_bps}bps, {priority_fee_level})")
                fresh_quote = self.get_quote(input_mint, output_mint, amount_lamports, slippage_bps=retry_bps)
                if not fresh_quote:
                    continue
                result.raw_quote = fresh_quote
                result.in_amount = int(fresh_quote.get("inAmount", 0))
                result.out_amount = int(fresh_quote.get("outAmount", 0))
                result.price_impact_pct = float(fresh_quote.get("priceImpactPct", 0))
                swap_tx_b64_2 = self.build_swap_transaction(fresh_quote,
                                                             priority_fee_microlamports=priority_fee,
                                                             priority_level=_jup_level)
                if not swap_tx_b64_2:
                    continue
                signed_tx_b64_2 = self.sign_transaction(swap_tx_b64_2)
                if not signed_tx_b64_2:
                    continue
                sig2, confirmed2 = self.broadcast_and_confirm(signed_tx_b64_2)
                if confirmed2:
                    log.info(f"    ✅ fresh-quote retry succeeded at {retry_bps}bps (sig {str(sig2)[:16]}...)")
                    result.signature = sig2
                    result.confirmed = True
                    result.error = None
                    break
                else:
                    log.warning(f"    ❌ retry at {retry_bps}bps failed: {getattr(self,'_last_broadcast_error','')[:150]}")
            else:
                log.error(f"    ❌ all {len(retry_bps_ladder)} fresh-quote retries exhausted")

        return result
