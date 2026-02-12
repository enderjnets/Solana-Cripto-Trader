"""
Jupiter API Integration for Solana Trading Bot
=============================================
Based on: https://dev.jup.ag/

APIs:
- Ultra API v1: https://lite-api.jup.ag/ultra/v1 (NO requiere API key)
- Price V3: https://lite-api.jup.ag/price/v3 (FREE)
- Tokens V2: https://lite-api.jup.ag/tokens/v2 (FREE)

Ultra API usa la wallet como "taker" - sin API key requerida!
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Optional
from dataclasses import dataclass

# Jupiter API Endpoints
JUPITER_BASE = "https://lite-api.jup.ag"
ULTRA_V1 = f"{JUPITER_BASE}/ultra/v1"
PRICE_V3 = f"{JUPITER_BASE}/price/v3"
TOKENS_V2 = f"{JUPITER_BASE}/tokens/v2"

# Common Tokens
SOL = "So11111111111111111111111111111111111111112"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"


@dataclass
class OrderResponse:
    """Order/quote from Ultra API"""
    swap_type: str
    in_amount: str
    out_amount: str
    other_amount_threshold: str
    swap_mode: str
    price_impact_pct: str
    route_plan: List[Dict]
    fee_mint: str
    fee_bps: int
    transaction: Optional[str]  # Base64 when taker provided
    request_id: str
    input_mint: str
    output_mint: str
    in_usd_value: float
    out_usd_value: float


@dataclass
class ExecutionResponse:
    """Swap execution result"""
    signature: Optional[str]
    status: str  # "Success", "Failed", etc.


class JupiterClient:
    """
    Jupiter DEX Client for Solana
    
    Docs: https://dev.jup.ag/
    
    Features:
    - Token discovery (Tokens V2 API)
    - Price lookup (Price V3 API)
    - Get quotes (Ultra V1 API - FREE)
    - Execute swaps (Ultra V1 API - FREE)
    
    No API key required - uses wallet address as taker!
    """
    
    def __init__(self):
        self.session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
    
    # ==================== PRICE API (FREE) ====================
    
    async def get_price(self, mints: List[str]) -> Dict[str, Dict]:
        """Get USD prices - FREE, no key"""
        session = await self._get_session()
        url = f"{PRICE_V3}?ids={','.join(mints)}"
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Price API error: {resp.status}")
            return await resp.json()
    
    async def get_token_price(self, mint: str) -> float:
        """Get single token price"""
        data = await self.get_price([mint])
        return float(data.get(mint, {}).get("usdPrice", 0))
    
    # ==================== TOKENS API (FREE) ====================
    
    async def get_all_tokens(self) -> List[Dict]:
        """Get all supported tokens"""
        session = await self._get_session()
        url = f"{TOKENS_V2}/all"
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Tokens API error: {resp.status}")
            return await resp.json()
    
    async def search_tokens(self, query: str) -> List[Dict]:
        """Search tokens by name/symbol/mint"""
        session = await self._get_session()
        url = f"{TOKENS_V2}/search?query={query}"
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Search error: {resp.status}")
            return await resp.json()
    
    async def get_verified_tokens(self) -> List[Dict]:
        """Get all verified tokens"""
        session = await self._get_session()
        url = f"{TOKENS_V2}/tag?query=verified"
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Verified tokens error: {resp.status}")
            return await resp.json()
    
    async def get_trending_tokens(self, interval: str = "1h") -> List[Dict]:
        """Get trending tokens"""
        session = await self._get_session()
        url = f"{TOKENS_V2}/toptrending/{interval}"
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Trending error: {resp.status}")
            return await resp.json()
    
    # ==================== ULTRA API (FREE - NO KEY!) ====================
    
    async def get_order(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        taker: str = None,
        slippage_bps: int = 0
    ) -> OrderResponse:
        """
        Get swap order/quote
        
        GET https://lite-api.jup.ag/ultra/v1/order
        
        Args:
            input_mint: Input token mint
            output_mint: Output token mint
            amount: Amount in smallest units (lamports for SOL)
            taker: Wallet address (optional - without it, no transaction)
            slippage_bps: Slippage in basis points
            
        Returns:
            OrderResponse with quote details
        """
        session = await self._get_session()
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount)
        }
        if taker:
            params["taker"] = taker
        if slippage_bps > 0:
            params["slippageBps"] = str(slippage_bps)
        
        url = f"{ULTRA_V1}/order?" + "&".join(f"{k}={v}" for k, v in params.items())
        
        async with session.get(url) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Order error {resp.status}: {text[:100]}")
            
            data = await resp.json()
            
            return OrderResponse(
                swap_type=data.get("swapType", ""),
                in_amount=data.get("inAmount", "0"),
                out_amount=data.get("outAmount", "0"),
                other_amount_threshold=data.get("otherAmountThreshold", "0"),
                swap_mode=data.get("swapMode", ""),
                price_impact_pct=data.get("priceImpactPct", "0"),
                route_plan=data.get("routePlan", []),
                fee_mint=data.get("feeMint", ""),
                fee_bps=data.get("feeBps", 0),
                transaction=data.get("transaction"),
                request_id=data.get("requestId", ""),
                input_mint=data.get("inputMint", ""),
                output_mint=data.get("outputMint", ""),
                in_usd_value=float(data.get("inUsdValue", 0)),
                out_usd_value=float(data.get("outUsdValue", 0))
            )
    
    async def execute_swap(
        self,
        signed_transaction: str,
        request_id: str
    ) -> ExecutionResponse:
        """
        Execute a signed swap transaction
        
        POST https://lite-api.jup.ag/ultra/v1/execute
        
        Args:
            signed_transaction: Base64 signed transaction
            request_id: Request ID from order response
            
        Returns:
            ExecutionResponse with signature and status
        """
        session = await self._get_session()
        url = f"{ULTRA_V1}/execute"
        payload = {
            "signedTransaction": signed_transaction,
            "requestId": request_id
        }
        
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Execute error {resp.status}: {text[:100]}")
            
            data = await resp.json()
            
            return ExecutionResponse(
                signature=data.get("signature"),
                status=data.get("status", "Unknown")
            )
    
    async def get_holdings(self, wallet: str) -> List[Dict]:
        """Get wallet token holdings"""
        session = await self._get_session()
        url = f"{ULTRA_V1}/holdings/{wallet}"
        async with session.get(url) as resp:
            if resp.status != 200:
                return []
            return await resp.json()
    
    async def get_token_warnings(self, mints: List[str]) -> Dict:
        """Get security warnings for tokens"""
        session = await self._get_session()
        url = f"{ULTRA_V1}/shield?mints={','.join(mints)}"
        async with session.get(url) as resp:
            if resp.status != 200:
                return {}
            return await resp.json()
    
    # ==================== HELPER METHODS ====================
    
    def sol_to_lamports(self, sol: float) -> int:
        """SOL → lamports (9 decimals)"""
        return int(sol * 1e9)
    
    def lamports_to_sol(self, lamports: int) -> float:
        """lamports → SOL"""
        return lamports / 1e9
    
    def usdc_to_micro(self, usdc: float) -> int:
        """USDC → micro-USDC (6 decimals)"""
        return int(usdc * 1e6)
    
    def micro_to_usdc(self, micro: int) -> float:
        """micro-USDC → USDC"""
        return micro / 1e6
    
    async def get_quote(
        self,
        from_token: str,
        to_token: str,
        amount: float,
        taker: str = None
    ) -> OrderResponse:
        """
        Get quote with human-readable amounts
        
        Args:
            from_token: Source token (SOL, USDC, or mint)
            to_token: Dest token
            amount: Amount in source token
            taker: Wallet address (optional)
        """
        from_mint = self._get_mint(from_token)
        to_mint = self._get_mint(to_token)
        
        # Convert to smallest unit
        if from_mint == SOL:
            in_amount = self.sol_to_lamports(amount)
        elif from_mint == USDC:
            in_amount = self.usdc_to_micro(amount)
        else:
            in_amount = int(amount * 1e9)
        
        return await self.get_order(from_mint, to_mint, in_amount, taker)
    
    async def quote_sol_to_usdc(self, sol_amount: float, taker: str = None) -> OrderResponse:
        """Quote SOL → USDC"""
        lamports = self.sol_to_lamports(sol_amount)
        return await self.get_order(SOL, USDC, lamports, taker)
    
    async def quote_usdc_to_sol(self, usdc_amount: float, taker: str = None) -> OrderResponse:
        """Quote USDC → SOL"""
        micro = self.usdc_to_micro(usdc_amount)
        return await self.get_order(USDC, SOL, micro, taker)
    
    def _get_mint(self, token: str) -> str:
        """Token name → mint address"""
        tokens = {
            "SOL": SOL, "WSOL": SOL,
            "USDC": USDC,
            "USDT": USDT
        }
        return tokens.get(token.upper(), token)


# ==================== CONVENIENCE FUNCTIONS ====================

async def get_sol_price() -> float:
    """Get SOL price in USD"""
    client = JupiterClient()
    try:
        return await client.get_token_price(SOL)
    finally:
        await client.close()


async def get_portfolio(wallet: str) -> Dict:
    """Get wallet portfolio"""
    client = JupiterClient()
    try:
        sol_price = await client.get_token_price(SOL)
        holdings = await client.get_holdings(wallet)
        
        portfolio = {
            "wallet": wallet,
            "sol_price": sol_price,
            "holdings": holdings,
            "total_usd": 0
        }
        
        for h in holdings:
            mint = h.get("mint", "")
            amount = float(h.get("amount", 0))
            
            if mint == SOL:
                value = client.lamports_to_sol(amount) * sol_price
                portfolio["sol_amount"] = client.lamports_to_sol(amount)
            else:
                price = await client.get_token_price(mint)
                value = amount * price
                portfolio["holdings"].append({
                    "mint": mint,
                    "amount": amount,
                    "price": price,
                    "value": value
                })
            
            portfolio["total_usd"] += value
        
        return portfolio
    finally:
        await client.close()


# ==================== MAIN DEMO ====================

async def demo():
    print("="*60)
    print("🚀 JUPITER ULTRA API DEMO")
    print("="*60)
    
    client = JupiterClient()
    wallet = "65YqSYGwR6UNCUmeaKt1V1HV99Ky1tii2bgg6jwJSGN3"  # Devnet wallet
    
    try:
        # 1. Get SOL Price (FREE)
        print("\n📊 Price API (FREE):")
        price = await client.get_token_price(SOL)
        print(f"   SOL: ${price:.2f}")
        
        # 2. Search Tokens (FREE)
        print("\n🔍 Search 'JUP':")
        tokens = await client.search_tokens("JUP")
        for t in tokens[:3]:
            print(f"   {t.get('symbol')}: {t.get('name')}")
        
        # 3. Get Quote (FREE - no wallet needed)
        print("\n💱 Quote: 1 SOL → USDC")
        order = await client.quote_sol_to_usdc(1.0)
        out_usdc = client.micro_to_usdc(int(order.out_amount))
        print(f"   Output: {out_usdc:.2f} USDC")
        print(f"   Impact: {order.price_impact_pct}%")
        print(f"   Route: {len(order.route_plan)} hops")
        print(f"   Request ID: {order.request_id[:16]}...")
        
        # 4. Quote with wallet (includes transaction)
        print(f"\n🔐 Quote with wallet: {wallet[:12]}...")
        order_with_tx = await client.quote_sol_to_usdc(1.0, wallet)
        has_tx = order_with_tx.transaction is not None
        print(f"   Has transaction: {'✅' if has_tx else '❌'}")
        
        # 5. Holdings
        print("\n💰 Wallet Holdings:")
        holdings = await client.get_holdings(wallet)
        print(f"   Tokens: {len(holdings)}")
        
        # 6. Trending
        print("\n📈 Trending Tokens (1h):")
        trending = await client.get_trending_tokens("1h")
        for t in trending[:3]:
            print(f"   {t.get('symbol')}: {t.get('name')}")
        
    finally:
        await client.close()
    
    print("\n" + "="*60)
    print("✅ Demo complete!")
    print("="*60)
    print("\n📚 Docs: https://dev.jup.ag/")
    print("🔑 No API key required!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(demo())
