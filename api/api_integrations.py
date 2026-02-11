"""
Jupiter API Integration for Solana Trading Bot
=============================================
Based on: https://dev.jup.ag/

API STATUS (Feb 2026):
- Price V3: ✅ FREE (no key required)
- Tokens V2: ✅ FREE (no key required)  
- Swap V6: ⚠️ REQUIRES API KEY (portal.jup.ag)

Get free API key: https://portal.jup.ag
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Optional
from dataclasses import dataclass

# Jupiter API Endpoints
JUPITER_BASE_URL = "https://lite-api.jup.ag"
JUPITER_V6_URL = "https://api.jup.ag/swap/v6"
JUPITER_PRICE_URL = f"{JUPITER_BASE_URL}/price/v3"
JUPITER_TOKENS_URL = f"{JUPITER_BASE_URL}/tokens/v2"

# Common Tokens
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenuNYW"


@dataclass
class QuoteResponse:
    """Swap quote response"""
    input_mint: str
    output_mint: str
    in_amount: int
    out_amount: int
    other_amount_threshold: int
    price_impact_pct: float
    route_plan: List[Dict]


class JupiterClient:
    """
    Jupiter DEX Client
    
    Docs: https://dev.jup.ag/
    API Key: https://portal.jup.ag (optional, for higher rate limits)
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            headers = {}
            if self.api_key:
                headers["x-api-key"] = self.api_key
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session
    
    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
    
    # ==================== PRICE API (FREE) ====================
    
    async def get_price(self, token_mints: List[str]) -> Dict[str, Dict]:
        """Get USD prices - FREE, no key required"""
        session = await self._get_session()
        ids = ",".join(token_mints)
        url = f"{JUPITER_PRICE_URL}?ids={ids}"
        
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Price API error: {resp.status}")
            return await resp.json()
    
    async def get_token_price(self, mint: str) -> float:
        """Get single token price in USD"""
        data = await self.get_price([mint])
        return float(data.get(mint, {}).get("usdPrice", 0))
    
    # ==================== TOKENS API (FREE) ====================
    
    async def get_tokens(self) -> List[Dict]:
        """Get all supported tokens - FREE"""
        session = await self._get_session()
        url = f"{JUPITER_TOKENS_URL}/all"
        
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Tokens API error: {resp.status}")
            return await resp.json()
    
    async def search_tokens(self, query: str) -> List[Dict]:
        """Search tokens by name/symbol/mint"""
        session = await self._get_session()
        url = f"{JUPITER_TOKENS_URL}/search?query={query}"
        
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Search API error: {resp.status}")
            return await resp.json()
    
    # ==================== SWAP API (REQUIRES KEY) ====================
    
    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 50
    ) -> QuoteResponse:
        """
        Get swap quote
        
        ⚠️ REQUIRES API KEY from portal.jup.ag
        
        Returns:
            QuoteResponse with swap details
        """
        session = await self._get_session()
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": amount,
            "slippageBps": slippage_bps
        }
        url = f"{JUPITER_V6_URL}/quote?" + "&".join(f"{k}={v}" for k, v in params.items())
        
        async with session.get(url) as resp:
            if resp.status == 401:
                raise Exception(
                    "❌ Swap API requires API key!\n"
                    "   Get free key: https://portal.jup.ag\n"
                    "   Then add to .env: JUPITER_API_KEY=your_key"
                )
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Quote API error {resp.status}: {text[:100]}")
            
            data = await resp.json()
            return QuoteResponse(
                input_mint=data.get("inputMint", ""),
                output_mint=data.get("outputMint", ""),
                in_amount=int(data.get("inAmount", 0)),
                out_amount=int(data.get("outAmount", 0)),
                other_amount_threshold=int(data.get("otherAmountThreshold", 0)),
                price_impact_pct=float(data.get("priceImpactPct", 0)),
                route_plan=data.get("routePlan", [])
            )
    
    async def get_swap_transaction(
        self,
        quote_response: Dict,
        user_public_key: str
    ) -> str:
        """
        Get serialized swap transaction
        
        ⚠️ REQUIRES API KEY
        
        Returns:
            Base64 serialized transaction (sign with wallet)
        """
        session = await self._get_session()
        url = f"{JUPITER_V6_URL}/swap"
        payload = {
            "quoteResponse": quote_response,
            "userPublicKey": user_public_key,
            "wrapAndUnwrapSol": True
        }
        
        async with session.post(url, json=payload) as resp:
            if resp.status == 401:
                raise Exception(
                    "❌ Swap API requires API key!\n"
                    "   Get free key: https://portal.jup.ag"
                )
            if resp.status != 200:
                raise Exception(f"Swap error {resp.status}")
            data = await resp.json()
            return data.get("swapTransaction", "")
    
    # ==================== HELPERS ====================
    
    def sol_to_lamports(self, sol: float) -> int:
        return int(sol * 1e9)
    
    def lamports_to_sol(self, lamports: int) -> float:
        return lamports / 1e9
    
    def usdc_to_micro(self, usdc: float) -> int:
        return int(usdc * 1e6)
    
    def micro_to_usdc(self, micro: int) -> float:
        return micro / 1e6
    
    async def get_sol_to_usdc_quote(self, sol_amount: float, slippage: int = 50) -> QuoteResponse:
        """Quote SOL → USDC"""
        lamports = self.sol_to_lamports(sol_amount)
        return await self.get_quote(SOL_MINT, USDC_MINT, lamports, slippage)
    
    async def get_usdc_to_sol_quote(self, usdc_amount: float, slippage: int = 50) -> QuoteResponse:
        """Quote USDC → SOL"""
        micro = self.usdc_to_micro(usdc_amount)
        return await self.get_quote(USDC_MINT, SOL_MINT, micro, slippage)


# ==================== CONVENIENCE FUNCTIONS ====================

async def get_sol_price() -> float:
    """Get SOL price in USD"""
    client = JupiterClient()
    try:
        return await client.get_token_price(SOL_MINT)
    finally:
        await client.close()


# ==================== MAIN DEMO ====================

async def demo():
    print("="*60)
    print("🚀 JUPITER API DEMO")
    print("="*60)
    
    client = JupiterClient()
    
    try:
        # Price (FREE)
        print("\n📊 Price API (FREE):")
        price = await client.get_token_price(SOL_MINT)
        print(f"   SOL: ${price:.2f}")
        
        # Tokens (FREE)
        print("\n🔍 Tokens API (FREE):")
        tokens = await client.search_tokens("JUP")
        print(f"   Found {len(tokens)} tokens matching 'JUP'")
        
        # Quote (REQUIRES KEY)
        print("\n💱 Quote API (REQUIRES API KEY):")
        print("   ⚠️ Need API key from https://portal.jup.ag")
        print("   Add to .env: JUPITER_API_KEY=your_key")
        
        # Try anyway to show error
        try:
            quote = await client.get_sol_to_usdc_quote(1.0)
            out = client.micro_to_usdc(quote.out_amount)
            print(f"   1 SOL → {out:.2f} USDC")
        except Exception as e:
            print(f"   {str(e)[:80]}...")
        
    finally:
        await client.close()
    
    print("\n" + "="*60)
    print("📚 Docs: https://dev.jup.ag/")
    print("🔑 Get API Key: https://portal.jup.ag")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(demo())
