"""
Jupiter API Integration for Solana Trading Bot
=============================================
Based on: https://dev.jup.ag/

APIs:
- Ultra API v1: https://lite-api.jup.ag/ultra/v1 (NO requiere API key)
- Price V3: https://lite-api.jup.ag/price/v3 (FREE)
- Tokens V2: https://lite-api.jup.ag/tokens/v2 (FREE)

Ultra API usa la wallet como "taker" - sin API key requerida!

Features:
- Exponential backoff with jitter for retries
- Circuit breaker pattern for API resilience
- Rate limiting awareness
- Comprehensive error handling
"""

import os
import json
import asyncio
import aiohttp
import time
import random
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("jupiter_api")

# Jupiter API Endpoints
JUPITER_BASE = "https://lite-api.jup.ag"
ULTRA_V1 = f"{JUPITER_BASE}/ultra/v1"
PRICE_V3 = f"{JUPITER_BASE}/price/v3"
TOKENS_V2 = f"{JUPITER_BASE}/tokens/v2"

# Common Tokens
SOL = "So11111111111111111111111111111111111111112"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"


# ============================================================================
# API CONFIGURATION & RETRY LOGIC
# ============================================================================

@dataclass
class RetryConfig:
    """Configuration for exponential backoff retry logic"""
    max_retries: int = 5           # Maximum number of retries
    base_delay: float = 1.0        # Initial delay in seconds
    max_delay: float = 60.0       # Maximum delay cap
    exponential_base: float = 2.0 # Base for exponential growth
    jitter: bool = True            # Add random jitter to prevent thundering herd
    jitter_factor: float = 0.3     # Jitter intensity (0-1)
    timeout: float = 30.0          # Request timeout in seconds
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a specific attempt number"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            jitter_range = delay * self.jitter_factor
            delay = delay + random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)  # Ensure non-negative


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern"""
    failure_threshold: int = 5     # Number of failures before opening
    success_threshold: int = 3     # Successes needed to close after half-open
    timeout_seconds: float = 30.0   # Time in open state before half-open
    monitoring_window: int = 60    # Window for tracking failures (seconds)


class CircuitState:
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing if API recovered


class CircuitBreaker:
    """
    Circuit Breaker implementation for API resilience.
    
    Prevents cascading failures by stopping requests to a failing service.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is down, requests are blocked immediately
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._last_state_change = datetime.now()
        self._lock = asyncio.Lock()
    
    async def __aenter__(self):
        await self._before_request()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            await self._on_success()
        else:
            await self._on_failure(exc_val)
        return False  # Don't suppress exceptions
    
    async def _before_request(self):
        """Check if request should be allowed"""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                time_since_open = (datetime.now() - self._last_state_change).total_seconds()
                if time_since_open >= self.config.timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(f"Circuit {self.name}: OPEN → HALF_OPEN")
                else:
                    raise CircuitOpenError(
                        f"Circuit {self.name} is OPEN. Requests blocked for "
                        f"{self.config.timeout_seconds - time_since_open:.1f}s more"
                    )
    
    async def _on_success(self):
        """Handle successful request"""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._failure_count = 0
                    self.state = CircuitState.CLOSED
                    self._last_state_change = datetime.now()
                    logger.info(f"Circuit {self.name}: HALF_OPEN → CLOSED")
            else:
                self._failure_count = max(0, self._failure_count - 1)
    
    async def _on_failure(self, exception: Exception):
        """Handle failed request"""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self._last_state_change = datetime.now()
                logger.warning(f"Circuit {self.name}: HALF_OPEN → OPEN (failure)")
            elif self._failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self._last_state_change = datetime.now()
                logger.warning(
                    f"Circuit {self.name}: CLOSED → OPEN "
                    f"({self._failure_count} failures)"
                )
    
    def get_state_info(self) -> Dict:
        """Get circuit breaker status"""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": (
                self._last_failure_time.isoformat() 
                if self._last_failure_time else None
            ),
            "seconds_until_retry": max(
                0,
                self.config.timeout_seconds - 
                (datetime.now() - self._last_state_change).total_seconds()
            ) if self.state == CircuitState.OPEN else 0
        }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


class APIError(Exception):
    """Base exception for API errors"""
    def __init__(self, message: str, status_code: int = None, response: str = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response = response
    
    def __str__(self):
        if self.status_code:
            return f"{self.message} (HTTP {self.status_code})"
        return self.message


class RateLimitError(APIError):
    """Raised when rate limit is exceeded"""
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class APIConnectionError(APIError):
    """Raised when connection fails"""
    pass


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
    - Exponential backoff with jitter for retries
    - Circuit breaker pattern for API resilience
    
    No API key required - uses wallet address as taker!
    """
    
    def __init__(self, retry_config: RetryConfig = None, circuit_config: CircuitBreakerConfig = None):
        self.session = None
        self.retry_config = retry_config or RetryConfig()
        self.circuit_config = circuit_config or CircuitBreakerConfig()
        self._price_circuit = CircuitBreaker("jupiter_price", self.circuit_config)
        self._order_circuit = CircuitBreaker("jupiter_order", self.circuit_config)
        self._tokens_circuit = CircuitBreaker("jupiter_tokens", self.circuit_config)
        self._price_cache: Dict[str, tuple] = {}
        self._cache_ttl = 30  # seconds
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.retry_config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """
        Make an HTTP request with exponential backoff retry logic.
        
        Args:
            method: HTTP method (get, post, etc.)
            url: Request URL
            **kwargs: Additional arguments for aiohttp
            
        Returns:
            Response object
            
        Raises:
            APIConnectionError: If all retries fail
            CircuitOpenError: If circuit breaker is open
        """
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                session = await self._get_session()
                
                # Make request
                if method.lower() == "get":
                    async with session.get(url, **kwargs) as resp:
                        await self._handle_response(resp)
                        return resp
                elif method.lower() == "post":
                    async with session.post(url, **kwargs) as resp:
                        await self._handle_response(resp)
                        return resp
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                    
            except asyncio.TimeoutError as e:
                last_exception = APIConnectionError(
                    f"Request timeout after {self.retry_config.timeout}s",
                    response=str(e)
                )
                logger.warning(f"Request timeout (attempt {attempt + 1}): {e}")
                
            except aiohttp.ClientError as e:
                last_exception = APIConnectionError(
                    f"Connection error: {e}",
                    response=str(e)
                )
                logger.warning(f"Connection error (attempt {attempt + 1}): {e}")
            
            # Check if we should retry
            if attempt < self.retry_config.max_retries:
                delay = self.retry_config.get_delay(attempt)
                logger.debug(f"Retrying in {delay:.2f}s (attempt {attempt + 2})")
                await asyncio.sleep(delay)
            else:
                # Max retries exceeded
                logger.error(f"All {self.retry_config.max_retries + 1} attempts failed")
                raise last_exception
        
        raise APIConnectionError("Max retries exceeded")
    
    async def _handle_response(self, resp: aiohttp.ClientResponse) -> None:
        """
        Handle HTTP response, raising appropriate errors.
        
        Raises:
            RateLimitError: For 429 responses
            APIError: For other error responses
        """
        if resp.status == 200:
            return
        
        text = await resp.text()
        
        if resp.status == 429:
            # Rate limited - try to get Retry-After header
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    retry_after = int(retry_after)
                except ValueError:
                    retry_after = 60
            
            raise RateLimitError(
                message=f"Rate limit exceeded",
                retry_after=retry_after
            )
        
        # Other errors
        raise APIError(
            message=f"API request failed",
            status_code=resp.status,
            response=text[:500]
        )
    
    # ==================== PRICE API (FREE) ====================
    
    async def get_price(self, mints: List[str]) -> Dict[str, Dict]:
        """Get USD prices - FREE, no key
        
        Uses circuit breaker and retry logic for resilience.
        """
        # Check cache first
        now = time.time()
        cache_key = tuple(sorted(mints))
        if cache_key in self._price_cache:
            cached_time, cached_data = self._price_cache[cache_key]
            if now - cached_time < self._cache_ttl:
                logger.debug(f"Price cache hit for {mints}")
                return cached_data
        
        try:
            async with self._price_circuit:
                url = f"{PRICE_V3}?ids={','.join(mints)}"
                resp = await self._request_with_retry("get", url)
                data = await resp.json()
                
                # Update cache
                self._price_cache[cache_key] = (now, data)
                logger.debug(f"Price cached for {mints}")
                
                return data
        except CircuitOpenError:
            # Try to return stale cache if available
            if cache_key in self._price_cache:
                logger.warning(f"Returning stale cache for {mints}")
                return self._price_cache[cache_key][1]
            raise
    
    async def get_token_price(self, mint: str) -> float:
        """Get single token price"""
        data = await self.get_price([mint])
        return float(data.get(mint, {}).get("usdPrice", 0))
    
    # ==================== TOKENS API (FREE) ====================
    
    async def get_all_tokens(self) -> List[Dict]:
        """Get all supported tokens"""
        try:
            async with self._tokens_circuit:
                url = f"{TOKENS_V2}/all"
                resp = await self._request_with_retry("get", url)
                return await resp.json()
        except CircuitOpenError:
            logger.error("Circuit open for tokens API - cannot fetch tokens")
            return []
    
    async def search_tokens(self, query: str) -> List[Dict]:
        """Search tokens by name/symbol/mint"""
        try:
            async with self._tokens_circuit:
                url = f"{TOKENS_V2}/search?query={query}"
                resp = await self._request_with_retry("get", url)
                return await resp.json()
        except CircuitOpenError:
            logger.error(f"Circuit open for search query: {query}")
            return []
    
    async def get_verified_tokens(self) -> List[Dict]:
        """Get all verified tokens"""
        try:
            async with self._tokens_circuit:
                url = f"{TOKENS_V2}/tag?query=verified"
                resp = await self._request_with_retry("get", url)
                return await resp.json()
        except CircuitOpenError:
            logger.error("Circuit open for verified tokens")
            return []
    
    async def get_trending_tokens(self, interval: str = "1h") -> List[Dict]:
        """Get trending tokens"""
        try:
            async with self._tokens_circuit:
                url = f"{TOKENS_V2}/toptrending/{interval}"
                resp = await self._request_with_retry("get", url)
                return await resp.json()
        except CircuitOpenError:
            logger.error(f"Circuit open for trending tokens (interval={interval})")
            return []
    
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
            
        Raises:
            CircuitOpenError: If circuit is open
            RateLimitError: If rate limited
            APIError: For other errors
        """
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
        
        try:
            async with self._order_circuit:
                resp = await self._request_with_retry("get", url)
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
        except CircuitOpenError:
            raise
        except RateLimitError:
            raise
        except APIError:
            raise
        except Exception as e:
            raise APIError(f"Failed to get order: {e}", response=str(e))
    
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
            
        Raises:
            CircuitOpenError: If circuit is open
            APIError: For other errors
        """
        url = f"{ULTRA_V1}/execute"
        payload = {
            "signedTransaction": signed_transaction,
            "requestId": request_id
        }
        
        try:
            async with self._order_circuit:
                resp = await self._request_with_retry("post", url, json=payload)
                data = await resp.json()
                
                return ExecutionResponse(
                    signature=data.get("signature"),
                    status=data.get("status", "Unknown")
                )
        except CircuitOpenError:
            raise
        except APIError:
            raise
        except Exception as e:
            raise APIError(f"Failed to execute swap: {e}", response=str(e))
    
    async def get_holdings(self, wallet: str) -> List[Dict]:
        """Get wallet token holdings"""
        try:
            async with self._order_circuit:
                url = f"{ULTRA_V1}/holdings/{wallet}"
                resp = await self._request_with_retry("get", url)
                return await resp.json()
        except CircuitOpenError:
            logger.warning(f"Circuit open for holdings: {wallet[:10]}...")
            return []
        except APIError:
            logger.warning(f"Failed to get holdings for {wallet[:10]}...")
            return []
    
    async def get_token_warnings(self, mints: List[str]) -> Dict:
        """Get security warnings for tokens"""
        try:
            async with self._tokens_circuit:
                url = f"{ULTRA_V1}/shield?mints={','.join(mints)}"
                resp = await self._request_with_retry("get", url)
                return await resp.json()
        except CircuitOpenError:
            logger.warning(f"Circuit open for token warnings")
            return {}
        except APIError:
            logger.warning(f"Failed to get token warnings")
            return {}
    
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
    
    def get_circuit_status(self) -> Dict:
        """Get status of all circuit breakers"""
        return {
            "price": self._price_circuit.get_state_info(),
            "order": self._order_circuit.get_state_info(),
            "tokens": self._tokens_circuit.get_state_info()
        }
    
    def reset_circuits(self):
        """Reset all circuit breakers to closed state"""
        self._price_circuit.state = CircuitState.CLOSED
        self._price_circuit._failure_count = 0
        self._order_circuit.state = CircuitState.CLOSED
        self._order_circuit._failure_count = 0
        self._tokens_circuit.state = CircuitState.CLOSED
        self._tokens_circuit._failure_count = 0
        logger.info("All circuit breakers reset to CLOSED state")
    
    def clear_cache(self):
        """Clear price cache"""
        self._price_cache.clear()
        logger.info("Price cache cleared")


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
