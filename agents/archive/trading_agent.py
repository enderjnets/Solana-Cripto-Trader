#!/usr/bin/env python3
"""
Trading Agent with Real API Integration
======================================
Specialized agent for Solana/Jupiter trading operations.

Uses:
- Solana RPC for blockchain operations
- Jupiter DEX for token swaps
- Helius for enhanced data (optional)

Based on OpenClaw Brain and Muscles pattern.
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.api_integrations import TradingAPIClient, APIConfig

logger = logging.getLogger("trading_agent")


# ============================================================================
# TRADING AGENT
# ============================================================================
class TradingAgentWithAPI:
    """
    Trading agent with real API integration.
    
    Capabilities:
    - Portfolio management
    - Swap execution
    - Order management
    - Risk checks before trades
    """
    
    def __init__(self, config = None):
        # Default config
        self.config = {
            "name": "trading_agent",
            "role": "Solana/Jupiter Trading",
            "capabilities": ["portfolio", "swap", "orders", "risk_check"]
        }
        
        self.name = self.config["name"]
        self.role = self.config["role"]
        self.state = {}
        self.memory = []
        self.task_history = []
        
        # Initialize API client
        self.api_config = APIConfig.from_env()
        self.client = TradingAPIClient(self.api_config)
        
        # Trading limits
        self.max_trade_pct = 0.10  # 10% of portfolio
        self.daily_loss_limit = 0.10  # 10% daily loss
        
        # Default tokens
        self.SOL = "So11111111111111111111111111111111111111112"
        self.USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        self.USDT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYW"
        
        logger.info(f"🚀 TradingAgent initialized for {self.api_config.network}")
    
    async def think(self, task: str, context: Dict = None) -> Dict:
        """Analyze trading task"""
        return await self.minimax.think(
            f"Analyze this trading task: {task}",
            context
        )
    
    async def execute(self, action: str, **kwargs) -> Dict:
        """Execute trading action"""
        actions = {
            "portfolio": self._get_portfolio,
            "swap": self._do_swap,
            "quote": self._get_quote,
            "balance": self._check_balance,
            "prepare_swap": self._prepare_swap,
            "validate_trade": self._validate_trade,
            "risk_check": self._risk_check
        }
        
        if action in actions:
            return await actions[action](**kwargs)
        
        return {"error": f"Unknown action: {action}"}
    
    async def _get_portfolio(self, wallet: str = None) -> Dict:
        """Get complete portfolio"""
        address = wallet or self.api_config.wallet_address
        
        portfolio = await self.client.get_portfolio(address)
        
        return {
            "status": "success",
            "wallet": address,
            "network": self.api_config.network,
            "sol_balance": portfolio.get("sol", {}).get("sol", 0),
            "holdings": portfolio.get("holdings", []),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _check_balance(self, wallet: str = None, token: str = "SOL") -> Dict:
        """Check specific token balance"""
        address = wallet or self.api_config.wallet_address
        
        if token.upper() == "SOL":
            result = await self.client.solana.get_balance(address)
        else:
            balances = await self.client.solana.get_token_balances(address)
            token_balances = balances.get("balances", {})
            result = token_balances.get(token, {"ui_amount": 0})
        
        return {
            "status": "success",
            "wallet": address,
            "token": token,
            "balance": result.get("sol", result.get("ui_amount", 0)),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _get_quote(self, from_token: str, to_token: str, amount: float) -> Dict:
        """Get swap quote"""
        # Map token names to addresses
        from_addr = self._get_token_address(from_token)
        to_addr = self._get_token_address(to_token)
        
        if not from_addr or not to_addr:
            return {"error": f"Unknown token: {from_token} or {to_token}"}
        
        quote = await self.client.get_quote(from_addr, to_addr, amount)
        
        return {
            "status": "success",
            "from": from_token,
            "to": to_token,
            "amount": amount,
            "output_amount": quote.get("output_amount", 0),
            "price": quote.get("output_price", 0),
            "route": quote.get("route", "direct"),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _prepare_swap(self, from_token: str, to_token: str, amount: float) -> Dict:
        """Prepare swap transaction"""
        from_addr = self._get_token_address(from_token)
        to_addr = self._get_token_address(to_token)
        
        if not from_addr or not to_addr:
            return {"error": f"Unknown token: {from_token} or {to_token}"}
        
        # First validate trade
        validation = await self._validate_trade(
            from_token=from_token,
            to_token=to_token,
            amount=amount
        )
        
        if not validation.get("approved", False):
            return {
                "error": "Trade validation failed",
                "reason": validation.get("reason", "Unknown")
            }
        
        # Prepare swap
        swap_data = await self.client.prepare_swap(
            from_token=from_token,
            to_token=to_token,
            amount=amount
        )
        
        return {
            "status": "prepared",
            "from": from_token,
            "to": to_token,
            "amount": amount,
            "wallet": self.api_config.wallet_address,
            "network": self.api_config.network,
            "instruction": swap_data.get("instruction", {}),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _do_swap(self, from_token: str, to_token: str, amount: float) -> Dict:
        """
        Execute swap.
        
        Note: This prepares the transaction. Actual signing
        requires wallet integration.
        """
        # Prepare swap first
        prepared = await self._prepare_swap(from_token, to_token, amount)
        
        if prepared.get("status") != "prepared":
            return prepared
        
        # Execute (this would normally sign and send)
        result = await self.client.execute_swap(
            from_token=from_token,
            to_token=to_token,
            amount=amount
        )
        
        return {
            "status": "pending",
            "tx": result.get("data", {}).get("quote", {}),
            "network": self.api_config.network,
            "message": "Transaction prepared. Requires wallet signing.",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _validate_trade(
        self,
        from_token: str,
        to_token: str,
        amount: float,
        wallet: str = None
    ) -> Dict:
        """Validate trade against risk limits"""
        address = wallet or self.api_config.wallet_address
        
        # Get current portfolio
        portfolio = await self._get_portfolio(address)
        sol_balance = portfolio.get("sol_balance", 0)
        
        # Check if amount is reasonable
        trade_value = amount * 0.05  # Assume SOL ~$50 for rough estimate
        
        if amount > sol_balance * self.max_trade_pct:
            return {
                "approved": False,
                "reason": f"Trade amount ({amount} SOL) exceeds {self.max_trade_pct*100}% limit ({sol_balance * self.max_trade_pct:.4f} SOL)"
            }
        
        # Check minimum trade
        if amount < 0.01:
            return {
                "approved": False,
                "reason": "Amount below minimum (0.01 SOL)"
            }
        
        return {
            "approved": True,
            "trade_amount": amount,
            "portfolio_pct": (amount / sol_balance) * 100 if sol_balance > 0 else 0,
            "wallet": address,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _risk_check(self, portfolio: Dict = None) -> Dict:
        """Perform risk check on portfolio"""
        address = self.api_config.wallet_address
        
        if portfolio is None:
            portfolio = await self._get_portfolio(address)
        
        # Simplified risk check
        sol_balance = portfolio.get("sol_balance", 0)
        
        return {
            "status": "success",
            "wallet": address,
            "sol_balance": sol_balance,
            "risk_level": "low" if sol_balance > 0.1 else "medium",
            "recommendations": [],
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_token_address(self, token: str) -> Optional[str]:
        """Map token name to address"""
        tokens = {
            "SOL": self.SOL,
            "USDC": self.USDC,
            "USDT": self.USDT,
            "WSOL": self.SOL
        }
        return tokens.get(token.upper())
    
    async def run_task(self, task: str, context: Dict = None) -> Dict:
        """Run complete trading task with thinking"""
        # Phase 1: Think
        thinking = await self.think(task, context)
        
        # Phase 2: Extract action
        action_type = context.get("action", "portfolio") if context else "portfolio"
        
        # Phase 3: Execute
        action_result = await self.execute(
            action_type,
            **context.get("payload", {}) if context else {}
        )
        
        # Phase 4: Return result
        return {
            "agent": self.name,
            "task": task,
            "thinking": thinking,
            "result": action_result,
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# MULTI-AGENT INTEGRATION
# ============================================================================
class TradingAgentMultiAgent:
    """
    Trading agent ready for multi-agent orchestration.
    
    Implements sessions_send pattern for:
    - Receiving tasks from coordinator
    - Sending results to risk agent
    - Communication with other agents
    """
    
    def __init__(self, orchestrator_url: str = None):
        self.trading_agent = TradingAgentWithAPI()
        self.orchestrator_url = orchestrator_url
        self.task_history = []
    
    async def receive_task(self, task: Dict) -> Dict:
        """
        Receive task from orchestrator.
        
        Task structure:
        {
            "type": "swap", "portfolio", "quote"
            "payload": {...},
            "from_agent": "coordinator"
        }
        """
        task_id = f"trade_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        try:
            # Execute trading action
            result = await self.trading_agent.execute(
                task.get("type", "portfolio"),
                **task.get("payload", {})
            )
            
            response = {
                "task_id": task_id,
                "status": "completed",
                "agent": "trading_agent",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
            self.task_history.append(response)
            
            return response
            
        except Exception as e:
            return {
                "task_id": task_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# ============================================================================
# MAIN DEMO
# ============================================================================
async def main():
    """Demo trading agent with real APIs"""
    
    print("="*70)
    print("🚀 TRADING AGENT - Real API Integration")
    print("="*70)
    
    # Load API config
    config = APIConfig.from_env()
    print(f"\n📡 Network: {config.network}")
    print(f"   Wallet: {config.wallet_address[:20]}...")
    print(f"   RPC: {config.rpc_url[:40]}...")
    
    # Initialize agent
    agent = TradingAgentWithAPI()
    
    # Demo 1: Get portfolio
    print("\n📊 1. Getting Portfolio...")
    portfolio = await agent._get_portfolio()
    print(f"   Status: {portfolio.get('status')}")
    print(f"   SOL: {portfolio.get('sol_balance', 0):.4f}")
    
    # Demo 2: Get quote
    print("\n💱 2. Getting Quote: SOL → USDC (1 SOL)")
    quote = await agent._get_quote("SOL", "USDC", 1.0)
    print(f"   Status: {quote.get('status')}")
    if "output_amount" in quote:
        print(f"   Output: {quote['output_amount']:.2f} USDC")
    
    # Demo 3: Validate trade
    print("\n✅ 3. Validating Trade: 0.5 SOL → USDC")
    validation = await agent._validate_trade("SOL", "USDC", 0.5)
    print(f"   Approved: {validation.get('approved')}")
    print(f"   Trade %: {validation.get('portfolio_pct', 0):.2f}%")
    
    # Demo 4: Prepare swap
    print("\n🔄 4. Preparing Swap: 0.1 SOL → USDC")
    swap = await agent._prepare_swap("SOL", "USDC", 0.1)
    print(f"   Status: {swap.get('status')}")
    if swap.get("status") == "prepared":
        print(f"   Wallet: {swap.get('wallet', 'N/A')[:20]}...")
    
    # Demo 5: Risk check
    print("\n🛡️ 5. Risk Check")
    risk = await agent._risk_check()
    print(f"   Status: {risk.get('status')}")
    print(f"   Risk Level: {risk.get('risk_level')}")
    
    # Show task history
    print("\n📝 Task History:")
    print(f"   Tasks run: {len(agent.task_history)}")
    
    print("\n" + "="*70)
    print("✅ Trading Agent Demo Complete")
    print("="*70)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
