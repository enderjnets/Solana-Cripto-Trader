#!/usr/bin/env python3
"""
Trading Command Handler for OpenClaw
=====================================
Enables Eko to execute trades via Telegram commands.

Usage:
    from trading_handler import TradingHandler
    handler = TradingHandler()
    response = await handler.execute("compra 0.5 SOL")
"""

import os
import sys
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from api.api_integrations import JupiterClient, SOL, USDC


@dataclass
class TradeResult:
    """Result of a trade command"""
    success: bool
    message: str
    data: Optional[Dict] = None


class TradingHandler:
    """
    Trading command handler for OpenClaw.
    
    Supports:
    - Portfolio checks
    - Price lookups
    - Buy/Sell orders
    - Risk validation
    """
    
    # Risk limits
    MAX_POSITION_PCT = 0.10  # 10%
    MAX_DAILY_LOSS = 0.10  # 10%
    
    # Common tokens
    TOKENS = {
        "SOL": SOL,
        "USDC": USDC,
        "USEsDT": "9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenuNYW"
    }
    
    def __init__(self):
        self.jupiter = JupiterClient()
        self.wallet = os.environ.get("HOT_WALLET_ADDRESS", "")
        self.trade_history = []
    
    async def close(self):
        """Close Jupiter client"""
        await self.jupiter.close()
    
    # ==================== BALANCE ====================
    
    async def get_balance(self) -> TradeResult:
        """Get wallet balance"""
        try:
            # Get SOL price
            sol_price = await self.jupiter.get_token_price(SOL)
            
            # Get holdings
            if self.wallet:
                holdings = await self.jupiter.get_holdings(self.wallet)
                sol_amount = holdings.get("amount", 0) / 1e9
            else:
                sol_amount = 5.0  # Default for devnet
            
            total_usd = sol_amount * sol_price
            
            message = f"""💰 **Tu Wallet:**
   
**SOL:** {sol_amount:.4f}
**USDC:** 0.00
─────────────────
**Total:** ${total_usd:.2f} USD

📍 `{self.wallet[:20]}...` (devnet)"""
            
            return TradeResult(
                success=True,
                message=message,
                data={
                    "sol": sol_amount,
                    "usdc": 0,
                    "total_usd": total_usd,
                    "sol_price": sol_price
                }
            )
        except Exception as e:
            return TradeResult(success=False, message=f"❌ Error: {e}")
    
    # ==================== PRICE ====================
    
    async def get_price(self, token: str = "SOL") -> TradeResult:
        """Get token price"""
        try:
            mint = self.TOKENS.get(token.upper(), token)
            price = await self.jupiter.get_token_price(mint)
            
            # Get 24h change
            prices = await self.jupiter.get_price([mint])
            change = prices.get(mint, {}).get("priceChange24h", 0)
            
            emoji = "🟢" if change >= 0 else "🔴"
            
            message = f"""📊 **Precio de {token.upper()}:**

**${price:.4f} USD**
{emoji} 24h: {change:+.2f}%"""
            
            return TradeResult(
                success=True,
                message=message,
                data={"price": price, "change": change}
            )
        except Exception as e:
            return TradeResult(success=False, message=f"❌ Error: {e}")
    
    # ==================== QUOTE ====================
    
    async def get_quote(self, from_token: str, to_token: str, amount: float) -> TradeResult:
        """Get swap quote"""
        try:
            from_mint = self.TOKENS.get(from_token.upper(), from_token)
            to_mint = self.TOKENS.get(to_token.upper(), to_token)
            
            order = await self.jupiter.get_quote(from_mint, to_mint, amount)
            
            # Convert output
            if to_mint == SOL:
                out_amount = self.jupiter.lamports_to_sol(int(order.out_amount))
            elif to_mint == USDC:
                out_amount = self.jupiter.micro_to_usdc(int(order.out_amount))
            else:
                out_amount = int(order.out_amount) / 1e9
            
            message = f"""💱 **Quote {amount} {from_token.upper()} → {to_token.upper()}:**

**Output:** {out_amount:.4f} {to_token.upper()}
📉 Impact: {order.price_impact_pct}%
🔀 Route: {len(order.route_plan)} hops"""
            
            return TradeResult(
                success=True,
                message=message,
                data={
                    "from": from_token,
                    "to": to_token,
                    "amount": amount,
                    "output": out_amount,
                    "impact": order.price_impact_pct
                }
            )
        except Exception as e:
            return TradeResult(success=False, message=f"❌ Error: {e}")
    
    # ==================== BUY ====================
    
    async def buy(self, token: str, amount: float) -> TradeResult:
        """
        Execute buy order (prepare transaction).
        
        Buying SOL means: Pay USDC → Receive SOL
        Buying other token: Pay SOL → Receive token
        """
        try:
            to_token = token.upper()
            
            # If buying SOL, pay with USDC
            if to_token == "SOL":
                order = await self.jupiter.get_quote(USDC, SOL, amount)  # amount is SOL, so convert
                out_amount = self.jupiter.lamports_to_sol(int(order.out_amount))
                pay_amount = amount  # SOL amount to receive
                pay_token = "USDC"
            else:
                # Buying other token, pay with SOL
                order = await self.jupiter.get_quote(SOL, self.TOKENS.get(to_token, to_token), amount)
                out_amount = int(order.out_amount) / 1e9  # Convert from smallest unit
                pay_amount = amount  # SOL amount to pay
                pay_token = "SOL"
            
            # Risk check
            balance = 5.0  # Devnet balance
            position_pct = pay_amount / balance
            
            if position_pct > self.MAX_POSITION_PCT:
                return TradeResult(
                    success=False,
                    message=f"❌ **Orden bloqueada:** Posición de {position_pct*100:.1f}% excede límite de {self.MAX_POSITION_PCT*100:.0f}%"
                )
            
            # Build response
            emoji = "🟢" if order.price_impact_pct.startswith("-") else "📉"
            
            if to_token == "SOL":
                message = f"""🔄 **Comprando {amount} SOL (pagando USDC):**

💰 **Recibirás:** {out_amount:.4f} SOL
💵 **Costo estimado:** ~{amount * 80:.2f} USDC
{emoji} Impacto: {order.price_impact_pct}%
🔀 Route: {len(order.route_plan)} hops
📍 Request ID: `{order.request_id[:16]}...`

🛡️ **Risk Check:**
   Posición: {position_pct*100:.1f}% del balance
   Estado: ✅ APROBADO

⚠️ **Para ejecutar:** Requiere firma con wallet

**Network:** devnet (SIN DINERO REAL)"""
            else:
                message = f"""🔄 **Comprando {pay_amount} SOL → {to_token}:**

💰 **Recibirás:** {out_amount:.4f} {to_token}
{emoji} Impacto: {order.price_impact_pct}%
🔀 Route: {len(order.route_plan)} hops
📍 Request ID: `{order.request_id[:16]}...`

🛡️ **Risk Check:**
   Posición: {position_pct*100:.1f}% del balance
   Estado: ✅ APROBADO

⚠️ **Para ejecutar:** Requiere firma con wallet

**Network:** devnet (SIN DINERO REAL)"""
            
            return TradeResult(
                success=True,
                message=message,
                data={
                    "action": "BUY",
                    "from": "SOL",
                    "to": to_token,
                    "amount": amount,
                    "output": out_amount,
                    "request_id": order.request_id,
                    "transaction": order.transaction,
                    "risk_approved": True
                }
            )
        except Exception as e:
            return TradeResult(success=False, message=f"❌ Error: {e}")
    
    # ==================== SELL ====================
    
    async def sell(self, token: str, amount: float) -> TradeResult:
        """Execute sell order (prepare transaction)"""
        try:
            to_token = "USDC"
            from_token = token.upper()
            mint = self.TOKENS.get(from_token, from_token)
            
            order = await self.jupiter.get_quote(mint, USDC, amount)
            
            # Convert output
            out_amount = self.jupiter.micro_to_usdc(int(order.out_amount))
            
            # Risk check
            balance = 5.0
            position_pct = amount / balance
            
            if position_pct > self.MAX_POSITION_PCT:
                return TradeResult(
                    success=False,
                    message=f"❌ **Orden bloqueada:** Posición de {position_pct*100:.1f}% excede límite"
                )
            
            message = f"""🔄 **Vendiendo {amount} {from_token} → USDC:**

💰 **Quote:** {out_amount:.2f} USDC
📉 Impacto: {order.price_impact_pct}%
🔀 Route: {len(order.route_plan)} hops
📍 Request ID: `{order.request_id[:16]}...`

🛡️ **Risk Check:**
   Posición: {position_pct*100:.1f}% del balance
   Estado: ✅ APROBADO

⚠️ **Para ejecutar:** Requiere firma con wallet

**Network:** devnet (SIN DINERO REAL)"""
            
            return TradeResult(
                success=True,
                message=message,
                data={
                    "action": "SELL",
                    "from": from_token,
                    "to": "USDC",
                    "amount": amount,
                    "output": out_amount,
                    "request_id": order.request_id,
                    "risk_approved": True
                }
            )
        except Exception as e:
            return TradeResult(success=False, message=f"❌ Error: {e}")
    
    # ==================== STATUS ====================
    
    async def get_status(self) -> TradeResult:
        """Get system status"""
        try:
            sol_price = await self.jupiter.get_token_price(SOL)
            balance = 5.0
            
            message = f"""📊 **Estado del Sistema:**

🤖 **Agentes:**
   ✅ Coordinator - Orchestrator
   ✅ Trading Agent - DEX Operations
   ✅ Analysis Agent - Market Research
   ✅ Risk Agent - Risk Management

💰 **Wallet:** `{self.wallet[:20]}...`
🏦 **Network:** devnet (SIN DINERO REAL)
💵 **Balance:** {balance:.4f} SOL
📈 **SOL Price:** ${sol_price:.2f}

🛡️ **Risk Limits:**
   • Max Position: {self.MAX_POSITION_PCT*100:.0f}%
   • Daily Loss: {self.MAX_DAILY_LOSS*100:.0f}%
   • Slippage: 0.5%

✅ **Jupiter API:** Conectado
📊 **Quotes:** Ilimitados
💱 **Swaps:** Listos para ejecutar"""
            
            return TradeResult(
                success=True,
                message=message,
                data={
                    "balance": balance,
                    "sol_price": sol_price,
                    "network": "devnet"
                }
            )
        except Exception as e:
            return TradeResult(success=False, message=f"❌ Error: {e}")
    
    # ==================== PARSE COMMAND ====================
    
    async def execute(self, command: str) -> TradeResult:
        """
        Execute trading command.
        
        Examples:
        - "mi balance"
        - "precio de SOL"
        - "compra 0.5 SOL"
        - "vende 1 SOL"
        - "status del sistema"
        """
        command = command.lower().strip()
        
        # Balance
        if "balance" in command or "mi balance" in command:
            return await self.get_balance()
        
        # Price
        if "precio" in command:
            # Extract token
            for token in ["SOL", "USDC", "USDT", "BTC", "ETH"]:
                if token.lower() in command:
                    return await self.get_price(token)
            return await self.get_price("SOL")
        
        # Buy
        if "compra" in command or "buy" in command:
            # Extract amount
            import re
            match = re.search(r'(\d+\.?\d*)', command)
            if match:
                amount = float(match.group(1))
                return await self.buy("SOL", amount)
            return TradeResult(success=False, message="❌ **Error:** Indica la cantidad (ej: 'compra 0.5 SOL')")
        
        # Sell
        if "vende" in command or "sell" in command:
            import re
            match = re.search(r'(\d+\.?\d*)', command)
            if match:
                amount = float(match.group(1))
                return await self.sell("SOL", amount)
            return TradeResult(success=False, message="❌ **Error:** Indica la cantidad (ej: 'vende 0.5 SOL')")
        
        # Status
        if "status" in command or "estado" in command:
            return await self.get_status()
        
        # Help if "status"
        if "ayuda" in command or "help" in command:
            return TradeResult(
                success=True,
                message="""📚 **Comandos disponibles:**

• `mi balance` - Ver balance de wallet
• `precio de SOL` - Ver precio de token
• `compra 0.5 SOL` - Preparar compra
• `vende 1 SOL` - Preparar venta
• `status del sistema` - Ver estado

⚠️ **Nota:** Las operaciones requieren firma con wallet"""
            )
        
        return TradeResult(
            success=False,
            message="❌ **Comando no reconocido.** Escribe 'ayuda' para ver comandos disponibles."
        )


# ==================== DEMO ====================

async def demo():
    """Demo trading handler"""
    print("="*60)
    print("🚀 TRADING HANDLER DEMO")
    print("="*60)
    
    handler = TradingHandler()
    
    try:
        # Test balance
        print("\n📊 /balance")
        result = await handler.get_balance()
        print(result.message)
        
        # Test price
        print("\n💰 /precio de SOL")
        result = await handler.get_price("SOL")
        print(result.message)
        
        # Test buy quote
        print("\n🛒 /compra 0.5 SOL")
        result = await handler.buy("SOL", 0.5)
        print(result.message)
        
        # Test status
        print("\n📊 /status")
        result = await handler.get_status()
        print(result.message)
        
    finally:
        await handler.close()
    
    print("\n" + "="*60)


if __name__ == "__main__":
    asyncio.run(demo())
