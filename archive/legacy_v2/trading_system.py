#!/usr/bin/env python3
"""
Complete Multi-Agent Trading System
=================================
Integrates Mission Control, Multi-Agents, and Real Trading APIs.

Architecture:
    Mission Control (Orchestrator)
        ├── Coordinator Agent
        ├── Trading Agent (Jupiter/Solana)
        ├── Analysis Agent (Research)
        ├── Risk Agent (Validation)
        └── Communication Agent (Telegram)

Based on OpenClaw Brain and Muscles pattern.
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.api_integrations import SolanaClient, JupiterClient, APIConfig
from agents.multi_agent_orchestrator import (
    MultiAgentOrchestrator, 
    AgentRegistry,
    InterAgentMessaging
)

logger = logging.getLogger("trading_system")

# ============================================================================
# TRADING AGENT (Specialized)
# ============================================================================
class TradingAgent:
    """
    Specialized trading agent with real API integration.
    
    Capabilities:
    - Portfolio management
    - Swap execution (Jupiter)
    - Order management
    - Position tracking
    """
    
    def __init__(self, config: APIConfig = None):
        self.config = config or APIConfig.from_env()
        self.client = SolanaClient(self.config)
        self.jupiter = JupiterClient(self.config)
        
        # Trading state
        self.positions: Dict[str, Dict] = {}
        self.order_history: List[Dict] = []
        
        # Default tokens
        self.SOL = "So11111111111111111111111111111111111111112"
        self.USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        
        logger.info(f"🚀 TradingAgent initialized for {self.config.network}")
    
    async def get_portfolio(self, wallet: str = None) -> Dict:
        """Get complete portfolio"""
        address = wallet or self.config.wallet_address
        
        sol = await self.client.get_balance(address)
        tokens = await self.client.get_token_balances(address)
        
        return {
            "wallet": address,
            "network": self.config.network,
            "sol": sol,
            "tokens": tokens.get("balances", {}),
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_quote(
        self, 
        from_token: str, 
        to_token: str, 
        amount: float
    ) -> Dict:
        """Get swap quote from Jupiter"""
        # Map token names to addresses
        from_addr = self._get_token_address(from_token)
        to_addr = self._get_token_address(to_token)
        
        if not from_addr or not to_addr:
            return {"error": f"Unknown token: {from_token} or {to_token}"}
        
        quote = await self.jupiter.get_quote(from_addr, to_addr, amount)
        
        return {
            "status": "success",
            "from": from_token,
            "to": to_token,
            "amount": amount,
            "output_amount": quote.get("output_amount", 0),
            "route": quote.get("route", "direct"),
            "timestamp": datetime.now().isoformat()
        }
    
    async def prepare_swap(
        self,
        from_token: str,
        to_token: str,
        amount: float,
        wallet: str = None
    ) -> Dict:
        """Prepare swap transaction"""
        address = wallet or self.config.wallet_address
        
        from_addr = self._get_token_address(from_token)
        to_addr = self._get_token_address(to_token)
        
        # Get quote
        quote = await self.jupiter.get_swap_quote(from_addr, to_addr, amount * 1e9)
        
        if quote.get("status") != "success":
            return {"error": "Failed to get quote"}
        
        # Get instruction
        instruction = await self.jupiter.get_swap_instruction(
            from_addr, to_addr, amount, address
        )
        
        return {
            "status": "prepared",
            "from": from_token,
            "to": to_token,
            "amount": amount,
            "wallet": address,
            "quote": quote.get("data", {}),
            "instruction": instruction,
            "timestamp": datetime.now().isoformat()
        }
    
    async def execute_swap(self, from_token: str, to_token: str, amount: float) -> Dict:
        """
        Execute swap.
        
        Note: This prepares the transaction. Signing requires wallet integration.
        """
        prepared = await self.prepare_swap(from_token, to_token, amount)
        
        if prepared.get("status") != "prepared":
            return prepared
        
        # Record order
        order = {
            "order_id": f"order_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "from_token": from_token,
            "to_token": to_token,
            "amount": amount,
            "status": "pending_signature",
            "wallet": prepared.get("wallet"),
            "timestamp": datetime.now().isoformat()
        }
        self.order_history.append(order)
        
        return {
            "status": "pending_signature",
            "order_id": order["order_id"],
            "message": "Transaction prepared. Requires wallet signing.",
            "network": self.config.network,
            "data": prepared.get("instruction", {})
        }
    
    async def check_order(self, order_id: str) -> Dict:
        """Check order status"""
        for order in self.order_history:
            if order["order_id"] == order_id:
                return order
        return {"error": "Order not found"}
    
    def _get_token_address(self, token: str) -> Optional[str]:
        """Map token name to address"""
        tokens = {
            "SOL": self.SOL,
            "USDC": self.USDC,
            "WSOL": self.SOL
        }
        return tokens.get(token.upper())


# ============================================================================
# ANALYSIS AGENT (Specialized)
# ============================================================================
class AnalysisAgent:
    """
    Market analysis agent.
    
    Capabilities:
    - Price analysis
    - Trend detection
    - Opportunity scanning
    """
    
    def __init__(self):
        self.market_data = {}
        
    async def analyze_price(self, token: str = "SOL") -> Dict:
        """Get current price analysis"""
        jupiter = JupiterClient()
        quote = await jupiter.get_quote(
            "So11111111111111111111111111111111111111112",
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            1.0
        )
        
        return {
            "token": token,
            "price": quote.get("output_amount", 0),
            "direction": "unknown",
            "confidence": 0.5,
            "timestamp": datetime.now().isoformat()
        }
    
    async def scan_opportunities(self) -> Dict:
        """Scan for trading opportunities"""
        return {
            "opportunities": [],
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# RISK AGENT (Specialized)
# ============================================================================
class RiskAgent:
    """
    Risk management agent.
    
    Capabilities:
    - Position limits
    - Daily loss limits
    - Trade validation
    """
    
    def __init__(self):
        self.max_position_pct = 0.10  # 10%
        self.max_daily_loss = 0.10   # 10%
        self.daily_pnl = 0.0
        
    async def validate_trade(
        self,
        trade: Dict,
        portfolio: Dict
    ) -> Dict:
        """Validate trade against risk limits"""
        amount = trade.get("amount", 0)
        sol_balance = portfolio.get("sol", {}).get("sol", 0)
        
        # Check position size
        trade_pct = amount / sol_balance if sol_balance > 0 else 1.0
        
        if trade_pct > self.max_position_pct:
            return {
                "approved": False,
                "reason": f"Position size {trade_pct*100:.1f}% exceeds {self.max_position_pct*100}% limit"
            }
        
        # Check daily loss
        if self.daily_pnl < -self.max_daily_loss:
            return {
                "approved": False,
                "reason": f"Daily loss {self.daily_pnl*100:.1f}% exceeds {self.max_daily_loss*100}% limit"
            }
        
        return {
            "approved": True,
            "trade_pct": trade_pct,
            "remaining_daily_loss": self.max_daily_loss - abs(self.daily_pnl),
            "timestamp": datetime.now().isoformat()
        }
    
    async def check_portfolio_risk(self, portfolio: Dict) -> Dict:
        """Assess overall portfolio risk"""
        sol_balance = portfolio.get("sol", {}).get("sol", 0)
        
        return {
            "risk_level": "low" if sol_balance > 0.1 else "medium",
            "sol_balance": sol_balance,
            "recommendations": [],
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# TRADING SYSTEM (Complete Integration)
# ============================================================================
class TradingSystem:
    """
    Complete multi-agent trading system.
    
    Integrates:
    - Mission Control orchestrator
    - Specialized agents (Trading, Analysis, Risk)
    - Real APIs (Solana, Jupiter)
    """
    
    def __init__(self):
        self.config = APIConfig.from_env()
        self.orchestrator = MultiAgentOrchestrator()
        
        # Initialize agents
        self.trading = TradingAgent(self.config)
        self.analysis = AnalysisAgent()
        self.risk = RiskAgent()
        
        # Communication
        self.messaging = InterAgentMessaging(self.orchestrator.registry)
        
        logger.info("🚀 TradingSystem initialized")
    
    async def execute_trade_workflow(self, command: Dict) -> Dict:
        """
        Execute complete trading workflow through agents.
        
        Workflow:
        1. Receive command
        2. Analysis (check price)
        3. Risk (validate)
        4. Trading (execute)
        5. Communication (notify)
        """
        workflow_id = f"wf_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        result = {
            "workflow_id": workflow_id,
            "command": command,
            "steps": [],
            "status": "processing",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Step 1: Get current portfolio
            portfolio = await self.trading.get_portfolio()
            result["steps"].append({
                "step": "portfolio",
                "status": "success",
                "data": portfolio
            })
            
            # Step 2: Get quote
            quote = await self.trading.get_quote(
                command.get("from", "SOL"),
                command.get("to", "USDC"),
                command.get("amount", 0)
            )
            result["steps"].append({
                "step": "quote",
                "status": "success",
                "data": quote
            })
            
            # Step 3: Risk validation
            risk = await self.risk.validate_trade(command, portfolio)
            result["steps"].append({
                "step": "risk",
                "status": "success" if risk.get("approved") else "blocked",
                "data": risk
            })
            
            if not risk.get("approved"):
                result["status"] = "blocked"
                result["reason"] = risk.get("reason")
                return result
            
            # Step 4: Execute trade
            if command.get("dry_run", False):
                # Just show quote
                result["steps"].append({
                    "step": "execute",
                    "status": "dry_run",
                    "data": quote
                })
            else:
                # Prepare swap (requires signing)
                execution = await self.trading.execute_swap(
                    command.get("from", "SOL"),
                    command.get("to", "USDC"),
                    command.get("amount", 0)
                )
                result["steps"].append({
                    "step": "execute",
                    "status": execution.get("status"),
                    "data": execution
                })
            
            result["status"] = "completed"
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        
        return result
    
    async def get_status(self) -> Dict:
        """Get complete system status"""
        portfolio = await self.trading.get_portfolio()
        analysis = await self.analysis.analyze_price()
        risk = await self.risk.check_portfolio_risk(portfolio)
        
        return {
            "system": "trading",
            "network": self.config.network,
            "wallet": portfolio.get("wallet", "")[:20] + "...",
            "portfolio": portfolio,
            "market": analysis,
            "risk": risk,
            "orders": len(self.trading.order_history),
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# MAIN DEMO
# ============================================================================
async def demo():
    """Demo complete trading system"""
    
    print("="*70)
    print("🚀 MULTI-AGENT TRADING SYSTEM DEMO")
    print("="*70)
    
    # Initialize system
    system = TradingSystem()
    
    # Show agents
    print("\n📋 Agents:")
    agents = system.orchestrator.list_agents()
    for agent in agents:
        print(f"   ✅ {agent['name']} ({agent['role']})")
    
    # Get status
    print("\n📊 System Status:")
    status = await system.get_status()
    print(f"   Network: {status['network']}")
    print(f"   Wallet: {status['wallet']}")
    print(f"   SOL: {status['portfolio'].get('sol', {}).get('sol', 0):.4f}")
    print(f"   Price: ${status['market'].get('price', 0):.2f}")
    print(f"   Risk: {status['risk'].get('risk_level', 'unknown')}")
    
    # Demo trade workflow
    print("\n🔄 Demo Trade Workflow:")
    print("-"*50)
    
    # Dry run trade
    trade_command = {
        "type": "swap",
        "from": "SOL",
        "to": "USDC",
        "amount": 0.5,
        "dry_run": True
    }
    
    result = await system.execute_trade_workflow(trade_command)
    
    print(f"\n   Workflow: {result['workflow_id']}")
    print(f"   Status: {result['status']}")
    
    for step in result["steps"]:
        print(f"\n   📌 {step['step'].upper()}:")
        print(f"      Status: {step['status']}")
        if step["step"] == "quote":
            print(f"      Output: {step['data'].get('output_amount', 0):.2f} USDC")
    
    # Check orders
    print(f"\n   Orders: {len(system.trading.order_history)}")
    
    print("\n" + "="*70)
    print("✅ TRADING SYSTEM DEMO COMPLETE")
    print("="*70)


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())
