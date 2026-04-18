"""
solana_rpc.py — Solana RPC client wrapper for live trading (Sprint 2 Fase 1)

Thin abstraction over Solana JSON-RPC with:
- Retry + exponential backoff on transient errors
- Helius-specific getPriorityFeeEstimate (fallback to baseline if unsupported)
- Singleton accessor via get_rpc()

Reads SOLANA_RPC_URL from env. Defaults to mainnet-beta public RPC
if not set (with lower rate limits — not recommended for production).

Usage:
    from agents.solana_rpc import get_rpc
    rpc = get_rpc()
    assert rpc.get_health() == "ok"
    balance = rpc.get_balance_sol("YOUR_PUBKEY")
    fee = rpc.get_priority_fee_estimate(level="medium")  # microlamports
"""
from __future__ import annotations

import os
import time
import logging
from typing import Optional

import requests

log = logging.getLogger('solana_rpc')

DEFAULT_RPC_URL     = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
DEFAULT_TIMEOUT_SEC = 30
DEFAULT_RETRIES     = 3
LAMPORTS_PER_SOL    = 1_000_000_000

# Helius priority fee levels (microlamports per compute unit)
FEE_LEVELS = {'min', 'low', 'medium', 'high', 'veryHigh', 'unsafeMax'}
FEE_FALLBACK_MEDIUM = 26532   # baseline medium value observed 2026-04-18

# Well-known mint addresses
MINT_SOL  = "So11111111111111111111111111111111111111112"  # wSOL
MINT_USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

# SPL Token Program ID
SPL_TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"


class SolanaRPC:
    """JSON-RPC client with retry + Helius priority-fee support."""

    def __init__(self, rpc_url: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT_SEC):
        self.rpc_url = rpc_url or DEFAULT_RPC_URL
        self.timeout = timeout
        if not self.rpc_url:
            raise ValueError("SOLANA_RPC_URL is empty (set in env or pass rpc_url)")

    # ── Low-level RPC call with retry ──
    def _rpc_call(self, method: str, params: Optional[list] = None,
                  retries: int = DEFAULT_RETRIES) -> dict:
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []}
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                r = requests.post(self.rpc_url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                d = r.json()
                if 'error' in d:
                    raise RuntimeError(f"{method} RPC error: {d['error']}")
                return d.get('result')
            except Exception as e:
                last_err = e
                if attempt < retries - 1:
                    sleep_sec = min(2 ** attempt, 8)
                    log.debug(f"{method} attempt {attempt+1}/{retries} failed ({e}); retry in {sleep_sec}s")
                    time.sleep(sleep_sec)
        raise RuntimeError(f"{method} failed after {retries} retries: {last_err}")

    # ── Standard Solana RPC ──
    def get_health(self) -> str:
        r = self._rpc_call("getHealth")
        return r if r == "ok" else "unhealthy"

    def get_version(self) -> dict:
        return self._rpc_call("getVersion")

    def get_balance(self, pubkey: str) -> int:
        """Returns SOL balance in lamports for given pubkey string."""
        r = self._rpc_call("getBalance", [pubkey])
        return r.get("value", 0) if isinstance(r, dict) else int(r or 0)

    def get_balance_sol(self, pubkey: str) -> float:
        return self.get_balance(pubkey) / LAMPORTS_PER_SOL

    def get_slot(self) -> int:
        return int(self._rpc_call("getSlot") or 0)

    def get_token_accounts(self, owner_pubkey: str, mint: Optional[str] = None) -> list:
        """List SPL token accounts for owner. Optional filter by mint."""
        filter_obj = {"mint": mint} if mint else {"programId": SPL_TOKEN_PROGRAM_ID}
        r = self._rpc_call(
            "getTokenAccountsByOwner",
            [owner_pubkey, filter_obj, {"encoding": "jsonParsed"}],
        )
        return r.get("value", []) if isinstance(r, dict) else []

    def get_token_balance(self, owner_pubkey: str, mint: str) -> float:
        """Returns token balance (uiAmount) for owner+mint combo. 0 if no account."""
        accounts = self.get_token_accounts(owner_pubkey, mint=mint)
        for acc in accounts:
            try:
                info = acc.get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
                amt = info.get("tokenAmount", {})
                return float(amt.get("uiAmount") or 0)
            except Exception:
                continue
        return 0.0

    # ── Helius-specific ──
    def get_priority_fee_estimate(self, account_keys: Optional[list] = None,
                                    level: str = 'medium') -> int:
        """Returns priority fee in microlamports/CU for given level.
        Fallback to FEE_FALLBACK_MEDIUM if RPC doesn't support this method."""
        if level not in FEE_LEVELS:
            level = 'medium'
        try:
            r = self._rpc_call(
                "getPriorityFeeEstimate",
                [{
                    "accountKeys": account_keys or [MINT_SOL],
                    "options": {"includeAllPriorityFeeLevels": True},
                }],
                retries=1,  # don't retry this (Helius-specific)
            )
            levels = (r or {}).get('priorityFeeLevels', {})
            return int(levels.get(level, FEE_FALLBACK_MEDIUM))
        except Exception as e:
            log.debug(f"priority fee estimate fallback ({e})")
            return FEE_FALLBACK_MEDIUM

    # ── Transaction submission (Fase 2 extends this) ──
    def send_transaction(self, signed_tx_base64: str, skip_preflight: bool = False,
                          preflight_commitment: str = "confirmed") -> str:
        """Broadcasts a base64-encoded signed transaction. Returns signature string."""
        params = [signed_tx_base64, {
            "encoding": "base64",
            "skipPreflight": skip_preflight,
            "preflightCommitment": preflight_commitment,
        }]
        r = self._rpc_call("sendTransaction", params)
        if not r:
            raise RuntimeError("sendTransaction returned empty result")
        return str(r)

    def confirm_transaction(self, signature: str, timeout_sec: int = 30) -> bool:
        """Polls getSignatureStatuses until confirmed or timeout. Raises on tx failure."""
        start = time.time()
        while time.time() - start < timeout_sec:
            r = self._rpc_call("getSignatureStatuses", [[signature], {"searchTransactionHistory": True}])
            values = r.get("value", []) if isinstance(r, dict) else []
            if values and values[0]:
                status = values[0]
                if status.get("err") is not None:
                    raise RuntimeError(f"Transaction {signature} failed: {status['err']}")
                cs = status.get("confirmationStatus", "")
                if cs in ("confirmed", "finalized"):
                    return True
            time.sleep(2)
        return False


# ── Singleton accessor ──
_rpc_singleton: Optional[SolanaRPC] = None


def get_rpc() -> SolanaRPC:
    """Returns shared SolanaRPC instance (lazy-init)."""
    global _rpc_singleton
    if _rpc_singleton is None:
        _rpc_singleton = SolanaRPC()
    return _rpc_singleton


def reset_rpc() -> None:
    """Reset the singleton (useful for tests when SOLANA_RPC_URL changes)."""
    global _rpc_singleton
    _rpc_singleton = None
