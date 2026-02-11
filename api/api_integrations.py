"""
Jupiter API Integration for Solana Trading Bot
=============================================
Based on official Jupiter documentation: https://dev.jup.ag/

APIs Available:
1. Ultra Swap - Swap execution (quotes + execute)
2. Tokens V2 - Token discovery
3. Price V3 - Pricing (no API key required)

Note: API keys are OPTIONAL for higher rate limits via portal.jup.ag
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Jupiter API Endpoints
JUPITER_BASE_URL = "https://lite-api.jup.ag"
JUPITER_PRICE_URL = f"{JUPITER_BASE_URL}/price/v3"
JUPITER_TOKENS_URL = f"{JUPITER_BASE_URL}/tokens/v2"
JUPITER_ULTRA_URL = f"{JUPITER_BASE_URL}/ultra/v1"

# Common Tokens
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenuNYW"


@dataclass
class TokenInfo:
    """Token metadata from Jupiter"""
    mint: str
    name: str
    symbol: str
    decimals: int
    usd_price: float
    logo_uri: Optional[str] = None


@dataclass
class QuoteRequest:
    """Swap quote request"""
    input_mint: str
    output_mint: str
    amount: int  # In lamports/smallest unit
    slippage_bps: int = 50  # 0.5% default


@dataclass  
class QuoteResponse:
    """Swap quote response"""
    input_mint: str
    output_mint: str
    in_amount: int
    out_amount: int
    other_amount_threshold: int
    swap_mode: str  # "ExactIn" or "ExactOut"
    price_impact_pct: float
    route_plan: List[Dict]


@dataclass
class SwapRequest:
    """Swap execution request"""
    quote_response: Dict  # Full quote response
    user_public_key: str
    wrap_and_unwrap_sol: bool = True
    prioritization_fee_lamports: int = 0
    as_legacy_transaction: bool = False


class JupiterClient:
    """
    Jupiter DEX Client for Solana
    
    Features:
    - Token discovery
    - Price lookup
    - Swap quotes
    - Swap execution
    
    No API key required for basic usage.
    Get API key from portal.jup.ag for higher rate limits.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize Jupiter client
        
        Args:
            api_key: Optional API key for higher rate limits
        """
        self.api_key = api_key
        self.session = None
        self._tokens_cache = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            headers = {}
            if self.api_key:
                headers["x-api-key"] = self.api_key
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session
    
    async def close(self):
        """Close the HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    # ==================== PRICE API ====================
    
    async def get_price(self, token_mints: List[str]) -> Dict[str, Dict]:
        """
        Get USD prices for tokens
        
        GET https://lite-api.jup.ag/price/v3?ids={mint1},{mint2},...
        
        Args:
            token_mints: List of token mint addresses
            
        Returns:
            Dict mapping mint addresses to price info
        """
        session = await self._get_session()
        ids = ",".join(token_mints)
        url = f"{JUPITER_PRICE_URL}?ids={ids}"
        
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Price API error: {response.status}")
            return await response.json()
    
    async def get_token_price(self, mint: str) -> float:
        """Get price for a single token in USD"""
        data = await self.get_price([mint])
        return float(data.get(mint, {}).get("usdPrice", 0))
    
    # ==================== TOKENS API ====================
    
    async def get_tokens(self) -> List[TokenInfo]:
        """
        Get all supported tokens
        
        GET https://lite-api.jup.ag/tokens/v2/all
        
        Returns:
            List of TokenInfo objects
        """
        session = await self._get_session()
        url = f"{JUPITER_TOKENS_URL}/all"
        
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Tokens API error: {response.status}")
            data = await response.json()
            
            return [
                TokenInfo(
                    mint=t.get("address", ""),
                    name=t.get("name", ""),
                    symbol=t.get("symbol", ""),
                    decimals=t.get("decimals", 0),
                    usd_price=float(t.get("usdPrice", 0)),
                    logo_uri=t.get("logoURI")
                )
                for t in data
            ]
    
    async def search_tokens(self, query: str) -> List[TokenInfo]:
        """
        Search tokens by symbol, name, or mint
        
        GET https://lite-api.jup.ag/tokens/v2/search?query={query}
        
        Args:
            query: Search term
            
        Returns:
            List of matching tokens
        """
        session = await self._get_session()
        url = f"{JUPITER_TOKENS_URL}/search?query={query}"
        
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Search API error: {response.status}")
            data = await response.json()
            
            return [
                TokenInfo(
                    mint=t.get("address", ""),
                    name=t.get("name", ""),
                    symbol=t.get("symbol", ""),
                    decimals=t.get("decimals", 0),
                    usd_price=float(t.get("usdPrice", 0)),
                    logo_uri=t.get("logoURI")
                )
                for t in data
            ]
    
    # ==================== ULTRA SWAP API ====================
    
    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 50,
        swap_mode: str = "ExactIn"
    ) -> QuoteResponse:
        """
        Get swap quote
        
        GET https://lite-api.jup.ag/ultra/v1/quote
        
        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            amount: Amount in smallest unit (lamports for SOL)
            slippage_bps: Slippage in basis points (50 = 0.5%)
            swap_mode: "ExactIn" or "ExactOut"
            
        Returns:
            QuoteResponse with swap details
        """
        session = await self._get_session()
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": amount,
            "slippageBps": slippage_bps,
            "swapMode": swap_mode
        }
        
        url = f"{JUPITER_ULTRA_URL}/quote?" + "&".join(f"{k}={v}" for k, v in params.items())
        
        async with session.get(url) as response:
            if response.status != 200:
                error = await response.text()
                raise Exception(f"Quote API error {response.status}: {error}")
            data = await response.json()
            
            return QuoteResponse(
                input_mint=data.get("inputMint", ""),
                output_mint=data.get("outputMint", ""),
                in_amount=int(data.get("inAmount", 0)),
                out_amount=int(data.get("outAmount", 0)),
                other_amount_threshold=int(data.get("otherAmountThreshold", 0)),
                swap_mode=data.get("swapMode", "ExactIn"),
                price_impact_pct=float(data.get("priceImpactPct", 0)),
                route_plan=data.get("routePlan", [])
            )
    
    async def get_swap_instructions(self, quote_response: Dict, user_public_key: str) -> Dict:
        """
        Get swap transaction instructions
        
        POST https://lite-api.jup.ag/ultra/v1/swap-instructions
        
        Args:
            quote_response: Full quote response from get_quote()
            user_public_key: User's wallet public key
            
        Returns:
            Transaction instructions ready for signing
        """
        session = await self._get_session()
        url = f"{JUPITER_ULTRA_URL}/swap-instructions"
        
        payload = {
            "quoteResponse": quote_response,
            "userPublicKey": user_public_key,
            "wrapAndUnwrapSol": True,
            "prioritizationFeeLamports": 0,
            "asLegacyTransaction": False
        }
        
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error = await response.text()
                raise Exception(f"Swap instructions error {response.status}: {error}")
            return await response.json()
    
    async def get_serialized_transaction(
        self,
        quote_response: Dict,
        user_public_key: str
    ) -> str:
        """
        Get serialized transaction (simpler than instructions)
        
        POST https://lite-api.jup.ag/ultra/v1/swap
        
        Args:
            quote_response: Full quote response from get_quote()
            user_public_key: User's wallet public key
            
        Returns:
            Serialized transaction as base64 string
        """
        session = await self._get_session()
        url = f"{JUPITER_ULTRA_URL}/swap"
        
        payload = {
            "quoteResponse": quote_response,
            "userPublicKey": user_public_key,
            "wrapAndUnwrapSol": True,
            "prioritizationFeeLamports": 0,
            "asLegacyTransaction": False
        }
        
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error = await response.text()
                raise Exception(f"Swap error {response.status}: {error}")
            data = await response.json()
            return data.get("serializedTransaction", "")
    
    # ==================== HELPER METHODS ====================
    
    def sol_to_lamports(self, sol_amount: float) -> int:
        """Convert SOL to lamports"""
        return int(sol_amount * 1e9)
    
    def lamports_to_sol(self, lamports: int) -> float:
        """Convert lamports to SOL"""
        return lamports / 1e9
    
    def usdc_to_micro(self, usdc_amount: float) -> int:
        """Convert USDC to micro-USDC (6 decimals)"""
        return int(usdc_amount * 1e6)
    
    def micro_to_usdc(self, micro: int) -> float:
        """Convert micro-USDC to USDC"""
        return micro / 1e6
    
    async def get_sol_to_usdc_quote(self, sol_amount: float) -> QuoteResponse:
        """
        Get quote for SOL to USDC swap
        
        Args:
            sol_amount: Amount in SOL
            
        Returns:
            QuoteResponse
        """
        lamports = self.sol_to_lamports(sol_amount)
        return await self.get_quote(SOL_MINT, USDC_MINT, lamports)
    
    async def get_usdc_to_sol_quote(self, usdc_amount: float) -> QuoteResponse:
        """
        Get quote for USDC to SOL swap
        
        Args:
            usdc_amount: Amount in USDC
            
        Returns:
            QuoteResponse
        """
        micro = self.usdc_to_micro(usdc_amount)
        return await self.get_quote(USDC_MINT, SOL_MINT, micro)
    
    async def get_output_amount(
        self,
        from_token: str,
        to_token: str,
        amount: float
    ) -> float:
        """
        Get output amount in human-readable format
        
        Args:
            from_token: Source token (SOL, USDC, or mint)
            to_token: Destination token
            amount: Amount in source token
            
        Returns:
            Amount in destination token
        """
        # Convert to mints
        from_mint = self._get_mint(from_token)
        to_mint = self._get_mint(to_token)
        
        # Get quote
        if from_mint == SOL_MINT:
            in_amount = self.sol_to_lamports(amount)
        elif from_mint == USDC_MINT:
            in_amount = self.usdc_to_micro(amount)
        else:
            # For other tokens, assume 9 decimals
            in_amount = int(amount * 1e9)
        
        quote = await self.get_quote(from_mint, to_mint, in_amount)
        
        # Convert output to human-readable
        if to_mint == SOL_MINT:
            return self.lamports_to_sol(quote.out_amount)
        elif to_mint == USDC_MINT:
            return self.micro_to_usdc(quote.out_amount)
        else:
            return quote.out_amount / 1e9
    
    def _get_mint(self, token: str) -> str:
        """Convert token name to mint address"""
        tokens = {
            "SOL": SOL_MINT,
            "WSOL": SOL_MINT,
            "USDC": USDC_MINT,
            "USDT": USDT_MINT
        }
        return tokens.get(token.upper(), token)


# ==================== CONVENIENCE FUNCTIONS ====================

async def get_sol_price() -> float:
    """Get current SOL price in USD"""
    client = JupiterClient()
    try:
        return await client.get_token_price(SOL_MINT)
    finally:
        await client.close()


async def get_usdc_price() -> float:
    """Get current USDC price in USD (should be $1.00)"""
    client = JupiterClient()
    try:
        return await client.get_token_price(USDC_MINT)
    finally:
        await client.close()


async def check_portfolio(wallet_address: str) -> Dict[str, Any]:
    """
    Get complete portfolio from Jupiter
    
    Args:
        wallet_address: Solana wallet address
        
    Returns:
        Dict with SOL balance and token holdings
    """
    client = JupiterClient()
    try:
        # Get SOL price
        sol_price = await client.get_token_price(SOL_MINT)
        
        # Get Jupiter holdings
        session = await client._get_session()
        url = f"{JUPITER_ULTRA_URL}/holdings/{wallet_address}"
        async with session.get(url) as response:
            if response.status != 200:
                holdings = []
            else:
                holdings = await response.json()
        
        # Get token info for each holding
        portfolio = {
            "wallet": wallet_address,
            "sol_balance": 0,  # Will be filled by Solana RPC
            "sol_price": sol_price,
            "tokens": [],
            "total_usd": 0
        }
        
        for holding in holdings:
            mint = holding.get("mint", "")
            amount = float(holding.get("amount", 0))
            
            if mint == SOL_MINT:
                portfolio["sol_balance"] = amount
                portfolio["total_usd"] += amount * sol_price
            else:
                price = await client.get_token_price(mint)
                value = amount * price
                portfolio["tokens"].append({
                    "mint": mint,
                    "amount": amount,
                    "usd_value": value
                })
                portfolio["total_usd"] += value
        
        return portfolio
    finally:
        await client.close()


# ==================== MAIN DEMO ====================

async def demo():
    """Demo Jupiter API integration"""
    
    print("="*60)
    print("🚀 JUPITER API DEMO")
    print("="*60)
    
    client = JupiterClient()
    
    try:
        # 1. Get SOL Price
        print("\n📊 SOL Price:")
        price = await client.get_token_price(SOL_MINT)
        print(f"   ${price:.2f}")
        
        # 2. Search Tokens
        print("\n🔍 Search for JUP:")
        tokens = await client.search_tokens("JUP")
        for t in tokens[:3]:
            print(f"   {t.symbol}: {t.name}")
        
        # 3. Get Quote (SOL → USDC)
        print("\n💱 Quote: 1 SOL → USDC")
        quote = await client.get_sol_to_usdc_quote(1.0)
        out_usdc = quote.out_amount / 1e6
        print(f"   Output: {out_usdc:.2f} USDC")
        print(f"   Price Impact: {quote.price_impact_pct:.4f}%")
        print(f"   Route: {len(quote.route_plan)} hops")
        
        # 4. Get Full Quote
        print("\n📋 Full Quote Details:")
        full_quote = await client.get_quote(SOL_MINT, USDC_MINT, 1_000_000_000)  # 1 SOL
        print(f"   In: {full_quote.in_amount} lamports")
        print(f"   Out: {full_quote.out_amount} micro-USDC")
        print(f"   Min Output: {full_quote.other_amount_threshold}")
        print(f"   Mode: {full_quote.swap_mode}")
        
    finally:
        await client.close()
    
    print("\n" + "="*60)
    print("✅ Demo complete!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(demo())
