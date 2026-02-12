#!/usr/bin/env python3
"""
Agent Runner - Continuous Trading Agent System
===============================================
Launches all agents in a continuous loop:
- Market analysis every cycle
- Strategy generation and evaluation
- Risk-checked trade execution (devnet)
- Logging and learning

Usage:
    python3 agent_runner.py              # Dry-run mode (default)
    python3 agent_runner.py --live       # Execute real swaps on devnet
    python3 agent_runner.py --interval 60  # Check every 60 seconds
"""

import os
import sys
import json
import asyncio
import logging
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.api_integrations import JupiterClient, SOL, USDC, USDT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "agent_runner.log")
    ]
)
logger = logging.getLogger("agent_runner")

# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

class AnalysisAgent:
    """Analyzes market conditions and generates signals."""

    def __init__(self, jupiter: JupiterClient):
        self.jupiter = jupiter
        self.price_history: List[Dict] = []
        self.max_history = 100

    async def analyze(self) -> Dict:
        """Run full market analysis."""
        sol_price = await self.jupiter.get_token_price(SOL)
        trending = await self.jupiter.get_trending_tokens("1h")

        self.price_history.append({
            "time": datetime.now().isoformat(),
            "price": sol_price
        })
        if len(self.price_history) > self.max_history:
            self.price_history = self.price_history[-self.max_history:]

        signal = self._generate_signal(sol_price)
        trend_names = [t.get("symbol", "?") for t in trending[:5]] if trending else []

        return {
            "sol_price": sol_price,
            "signal": signal,
            "confidence": self._confidence(),
            "trending": trend_names,
            "history_len": len(self.price_history),
            "timestamp": datetime.now().isoformat()
        }

    def _generate_signal(self, current_price: float) -> str:
        if len(self.price_history) < 3:
            return "WAIT"
        prices = [p["price"] for p in self.price_history[-10:]]
        avg = sum(prices) / len(prices)
        if current_price < avg * 0.98:
            return "BUY"
        elif current_price > avg * 1.02:
            return "SELL"
        return "HOLD"

    def _confidence(self) -> float:
        if len(self.price_history) < 5:
            return 0.3
        elif len(self.price_history) < 20:
            return 0.5
        return 0.7


class RiskAgent:
    """Validates trades against risk limits."""

    def __init__(self, max_position_pct: float = 0.10, daily_loss_pct: float = 0.10):
        self.max_position_pct = max_position_pct
        self.daily_loss_pct = daily_loss_pct
        self.daily_pnl = 0.0
        self.last_reset = datetime.now().date()
        self.trades_today = 0
        self.max_trades_day = 20

    def validate(self, trade: Dict, portfolio_sol: float) -> Dict:
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.last_reset = today

        amount = trade.get("amount", 0)
        max_allowed = portfolio_sol * self.max_position_pct

        if amount > max_allowed:
            return {"approved": False, "reason": f"Amount {amount} exceeds max {max_allowed:.4f} SOL"}
        if amount < 0.001:
            return {"approved": False, "reason": "Amount too small (min 0.001 SOL)"}
        if self.trades_today >= self.max_trades_day:
            return {"approved": False, "reason": f"Max {self.max_trades_day} trades/day reached"}

        self.trades_today += 1
        return {"approved": True, "reason": "Within limits", "max_allowed": max_allowed}


class TradingAgent:
    """Executes trades on Jupiter DEX."""

    def __init__(self, jupiter: JupiterClient, wallet_address: str):
        self.jupiter = jupiter
        self.wallet = wallet_address
        self.order_history: List[Dict] = []

    async def get_quote(self, from_token: str, to_token: str, amount_lamports: int) -> Dict:
        try:
            order = await self.jupiter.get_order(
                input_mint=from_token,
                output_mint=to_token,
                amount=amount_lamports,
                taker=self.wallet
            )
            return {
                "status": "ok",
                "in_amount": int(order.in_amount) / 1e9 if from_token == SOL else int(order.in_amount) / 1e6,
                "out_amount": int(order.out_amount) / 1e6 if to_token == USDC else int(order.out_amount) / 1e9,
                "in_usd": order.in_usd_value,
                "out_usd": order.out_usd_value,
                "impact": order.price_impact_pct,
                "request_id": order.request_id,
                "has_tx": order.transaction is not None
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def log_trade(self, trade: Dict):
        trade["timestamp"] = datetime.now().isoformat()
        self.order_history.append(trade)
        logger.info(f"Trade logged: {trade}")


class AccumulationReader:
    """Reads accumulation targets set by agent_brain's AccumulationAgent."""

    def __init__(self):
        self.target_file = PROJECT_ROOT / "data" / "accumulation_target.json"
        self.current: Dict = {"SOL": 0.60, "BTC": 0.40, "recommendation": "SOL"}
        self._last_read = None

    def get_target(self) -> Dict:
        """Read latest accumulation target (cached for 60s)."""
        now = datetime.now()
        if self._last_read and (now - self._last_read).total_seconds() < 60:
            return self.current

        if self.target_file.exists():
            try:
                data = json.loads(self.target_file.read_text())
                self.current = data
                self._last_read = now
                logger.info(f"[Accumulation] Target: SOL={data.get('SOL', 0.5):.0%} "
                           f"BTC={data.get('BTC', 0.5):.0%} "
                           f"Rec={data.get('recommendation', '?')}")
            except Exception as e:
                logger.warning(f"[Accumulation] Read error: {e}")
        return self.current


class StrategyAgent:
    """Generates and evaluates trading strategies using brain-deployed rules."""

    def __init__(self):
        self.strategies: List[Dict] = []
        self.active_strategy: Optional[Dict] = None
        self._last_deploy_check = None
        self._price_window: List[float] = []  # Rolling price window for indicators
        self._max_window = 200
        self._init_strategies()

    def _load_deployed_strategies(self):
        """Load strategies deployed by agent_brain.py."""
        deploy_file = PROJECT_ROOT / "data" / "active_strategies.json"
        if not deploy_file.exists():
            return
        try:
            data = json.loads(deploy_file.read_text())
            brain_strats = data.get("strategies", [])
            if brain_strats:
                # Merge: keep brain strategies + defaults, brain first
                existing_names = {s["name"] for s in self.strategies}
                for bs in brain_strats:
                    if bs["name"] not in existing_names:
                        self.strategies.insert(0, bs)
                self.active_strategy = self.strategies[0]
                logger.info(f"Loaded {len(brain_strats)} brain-deployed strategies")
        except Exception as e:
            logger.warning(f"Error loading deployed strategies: {e}")

    def _init_strategies(self):
        self.strategies = [
            {
                "name": "Mean Reversion",
                "description": "Buy when price drops >2% below average, sell when >2% above",
                "buy_threshold": -0.02,
                "sell_threshold": 0.02,
                "lookback": 10,
                "score": 0,
                "trades": 0,
                "wins": 0
            },
            {
                "name": "Momentum",
                "description": "Buy on 3 consecutive up candles, sell on 3 down",
                "consecutive_up": 3,
                "consecutive_down": 3,
                "score": 0,
                "trades": 0,
                "wins": 0
            },
            {
                "name": "SOL Accumulator",
                "description": "DCA into SOL when price is below 24h average",
                "dca_amount": 0.05,
                "buy_below_avg": True,
                "score": 0,
                "trades": 0,
                "wins": 0
            }
        ]
        self.active_strategy = self.strategies[0]

    def _compute_rsi(self, prices: List[float], period: int) -> float:
        """Compute RSI from price list."""
        if len(prices) < period + 1:
            return 50.0
        deltas = [prices[i] - prices[i - 1] for i in range(-period, 0)]
        gains = [d for d in deltas if d > 0]
        losses = [-d for d in deltas if d < 0]
        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period if losses else 1e-10
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        return 100.0 - (100.0 / (1.0 + rs))

    def _compute_sma(self, prices: List[float], period: int) -> float:
        """Compute SMA from price list."""
        if len(prices) < period:
            return prices[-1] if prices else 0
        return sum(prices[-period:]) / period

    def _compute_ema(self, prices: List[float], period: int) -> float:
        """Compute EMA from price list."""
        if len(prices) < period:
            return prices[-1] if prices else 0
        k = 2.0 / (period + 1)
        ema = prices[-period]
        for p in prices[-period + 1:]:
            ema = p * k + ema * (1 - k)
        return ema

    def _evaluate_brain_rules(self, rules: List[Dict], price: float) -> str:
        """Evaluate brain strategy rules against current price data.

        Returns BUY/SELL/HOLD signal based on rule evaluation.
        """
        if not rules or len(self._price_window) < 20:
            return "HOLD"

        all_pass = True
        for rule in rules:
            ind = rule.get("indicator", "RSI")
            period = rule.get("period", 14)
            operator = rule.get("operator", ">")
            threshold = rule.get("threshold", 50)

            # Compute indicator value
            if ind == "RSI":
                value = self._compute_rsi(self._price_window, period)
            elif ind == "SMA":
                value = self._compute_sma(self._price_window, period)
            elif ind == "EMA":
                value = self._compute_ema(self._price_window, period)
            else:
                continue

            # Evaluate rule
            if operator == ">":
                if not (value > threshold):
                    all_pass = False
                    break
            elif operator == "<":
                if not (value < threshold):
                    all_pass = False
                    break

        return "BUY" if all_pass else "HOLD"

    def evaluate(self, analysis: Dict) -> Optional[Dict]:
        """Evaluate current strategy against market data."""
        # Check for brain-deployed strategies every 5 minutes
        now = datetime.now()
        if self._last_deploy_check is None or (now - self._last_deploy_check).total_seconds() > 300:
            self._load_deployed_strategies()
            self._last_deploy_check = now

        if not self.active_strategy:
            return None

        price = analysis.get("sol_price", 0)
        confidence = analysis.get("confidence", 0)

        # Track price for indicator computation
        if price > 0:
            self._price_window.append(price)
            if len(self._price_window) > self._max_window:
                self._price_window = self._price_window[-self._max_window:]

        if confidence < 0.4:
            return {"action": "WAIT", "reason": "Low confidence", "strategy": self.active_strategy["name"]}

        # Determine signal: use brain rules if available, else fallback to mean-reversion
        brain_rules = self.active_strategy.get("brain_rules", [])
        if brain_rules:
            signal = self._evaluate_brain_rules(brain_rules, price)
        else:
            signal = analysis.get("signal", "WAIT")

        # Also check for SELL signal using mean-reversion when in brain mode
        if signal == "HOLD" and brain_rules:
            # Use sell_threshold from active strategy for profit-taking
            sell_threshold = self.active_strategy.get("sell_threshold", 0.02)
            if len(self._price_window) >= 10:
                avg = sum(self._price_window[-10:]) / 10
                if price > avg * (1 + sell_threshold):
                    signal = "SELL"

        if signal == "BUY":
            # Use accumulation target to decide WHAT to buy
            accum = analysis.get("accumulation", {})
            rec = accum.get("recommendation", "SOL")
            sol_pct = accum.get("SOL", 0.6)
            btc_pct = accum.get("BTC", 0.4)

            if rec == "BTC" and btc_pct >= 0.5:
                return {
                    "action": "BUY",
                    "token_from": "SOL",
                    "token_to": "cbBTC",
                    "token_from_mint": SOL,
                    "token_to_mint": "cbbtcf3aa214zXHbiAZQwf4122FBYbraNdFqgw4iMij",
                    "amount_sol": 0.05,
                    "reason": f"Brain:{self.active_strategy['name']} @ ${price:.2f} | BTC ({btc_pct:.0%})",
                    "strategy": self.active_strategy["name"],
                    "accumulation": rec,
                }
            else:
                return {
                    "action": "BUY",
                    "token_from": "USDC",
                    "token_to": "SOL",
                    "token_from_mint": USDC,
                    "token_to_mint": SOL,
                    "amount_sol": 0.05,
                    "reason": f"Brain:{self.active_strategy['name']} @ ${price:.2f} | SOL ({sol_pct:.0%})",
                    "strategy": self.active_strategy["name"],
                    "accumulation": rec,
                }
        elif signal == "SELL":
            return {
                "action": "SELL",
                "token_from": "SOL",
                "token_to": "USDC",
                "token_from_mint": SOL,
                "token_to_mint": USDC,
                "amount_sol": 0.05,
                "reason": f"Signal: {signal} @ ${price:.2f}",
                "strategy": self.active_strategy["name"]
            }

        return {"action": "HOLD", "reason": f"No signal @ ${price:.2f}", "strategy": self.active_strategy["name"]}

    def get_summary(self) -> Dict:
        return {
            "active": self.active_strategy["name"] if self.active_strategy else "None",
            "total_strategies": len(self.strategies),
            "strategies": [
                {"name": s["name"], "score": s["score"], "trades": s["trades"], "wins": s["wins"]}
                for s in self.strategies
            ]
        }


# ============================================================================
# COORDINATOR - Main Loop
# ============================================================================

class AgentCoordinator:
    """Orchestrates all agents in a continuous loop."""

    def __init__(self, wallet_address: str, live: bool = False, interval: int = 120):
        self.wallet = wallet_address
        self.live = live
        self.interval = interval
        self.running = False
        self.cycle = 0
        self.start_time = None

        self.jupiter = JupiterClient()
        self.analyst = AnalysisAgent(self.jupiter)
        self.risk = RiskAgent()
        self.trader = TradingAgent(self.jupiter, wallet_address)
        self.strategist = StrategyAgent()
        self.accumulation = AccumulationReader()
        self.sol_client = None  # Lazy init async Solana RPC client

        self.activity_log: List[Dict] = []

    def _log(self, agent: str, action: str, detail: str):
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "cycle": self.cycle,
            "agent": agent,
            "action": action,
            "detail": detail
        }
        self.activity_log.append(entry)
        if len(self.activity_log) > 500:
            self.activity_log = self.activity_log[-500:]
        logger.info(f"[{agent}] {action}: {detail}")

    async def run_cycle(self) -> Dict:
        """Run one complete trading cycle."""
        self.cycle += 1
        cycle_result = {"cycle": self.cycle, "time": datetime.now().isoformat(), "actions": []}

        # 1. Analysis
        self._log("Analysis", "Starting", "Market scan")
        analysis = await self.analyst.analyze()
        self._log("Analysis", "Complete",
                  f"SOL=${analysis['sol_price']:.2f} Signal={analysis['signal']} "
                  f"Conf={analysis['confidence']:.0%} Trending={analysis['trending'][:3]}")
        cycle_result["analysis"] = analysis

        # 1.5 Read accumulation target from brain
        accum = self.accumulation.get_target()
        analysis["accumulation"] = accum
        cycle_result["accumulation"] = {
            "SOL": accum.get("SOL", 0.5),
            "BTC": accum.get("BTC", 0.5),
            "recommendation": accum.get("recommendation", "SOL"),
        }

        # 2. Strategy evaluation
        self._log("Strategy", "Evaluating", self.strategist.active_strategy["name"])
        decision = self.strategist.evaluate(analysis)
        action = decision.get("action", "WAIT") if decision else "WAIT"
        reason = decision.get("reason", "") if decision else ""
        self._log("Strategy", "Decision", f"{action}: {reason}")
        cycle_result["decision"] = decision

        # 3. If action needed, validate with risk
        if action in ("BUY", "SELL"):
            amount = decision.get("amount_sol", 0.05)

            # Get current balance (async, persistent client)
            from solana.rpc.async_api import AsyncClient
            from solders.pubkey import Pubkey
            if self.sol_client is None:
                self.sol_client = AsyncClient("https://api.devnet.solana.com")
            balance_resp = await self.sol_client.get_balance(Pubkey.from_string(self.wallet))
            sol_balance = balance_resp.value / 1e9

            self._log("Risk", "Validating", f"{action} {amount} SOL (balance: {sol_balance:.4f})")
            risk_check = self.risk.validate({"amount": amount}, sol_balance)

            if risk_check["approved"]:
                self._log("Risk", "Approved", risk_check["reason"])

                # 4. Get quote - use mints from decision (supports BTC accumulation)
                from_mint = decision.get("token_from_mint", SOL if action == "SELL" else USDC)
                to_mint = decision.get("token_to_mint", USDC if action == "SELL" else SOL)

                if action == "SELL":
                    lamports = int(amount * 1e9)
                    quote = await self.trader.get_quote(from_mint, to_mint, lamports)
                elif decision.get("token_from") == "SOL":
                    # BTC accumulation: SOL -> cbBTC
                    lamports = int(amount * 1e9)
                    quote = await self.trader.get_quote(from_mint, to_mint, lamports)
                else:
                    # Default: USDC -> SOL
                    usdc_amount = int(amount * analysis["sol_price"] * 1e6)
                    quote = await self.trader.get_quote(from_mint, to_mint, usdc_amount)

                if quote["status"] == "ok":
                    self._log("Trading", "Quote",
                              f"In: {quote['in_amount']:.4f} -> Out: {quote['out_amount']:.4f} "
                              f"Impact: {quote['impact']}%")

                    if self.live and quote.get("has_tx"):
                        self._log("Trading", "EXECUTING", "Live trade on devnet!")
                        # NOTE: actual signing requires wallet private key
                        # For now log as dry run even in live mode
                        self.trader.log_trade({
                            "action": action,
                            "amount": amount,
                            "quote": quote,
                            "mode": "devnet",
                            "executed": False
                        })
                        self._log("Trading", "Logged", "Trade recorded (signing not implemented yet)")
                    else:
                        self.trader.log_trade({
                            "action": action,
                            "amount": amount,
                            "quote": quote,
                            "mode": "dry_run"
                        })
                        self._log("Trading", "DryRun", f"Would {action} {amount} SOL")
                else:
                    self._log("Trading", "QuoteFailed", quote.get("error", "Unknown"))
            else:
                self._log("Risk", "Rejected", risk_check["reason"])
        else:
            self._log("Coordinator", "NoAction", f"{action}: {reason}")

        cycle_result["trades_today"] = self.risk.trades_today
        cycle_result["order_history"] = len(self.trader.order_history)
        return cycle_result

    async def run(self):
        """Main continuous loop."""
        self.running = True
        self.start_time = datetime.now()

        print("=" * 70)
        print("  MULTI-AGENT TRADING SYSTEM - ACTIVE")
        print("=" * 70)
        print(f"  Wallet:   {self.wallet}")
        print(f"  Network:  devnet")
        print(f"  Mode:     {'LIVE' if self.live else 'DRY RUN'}")
        print(f"  Interval: {self.interval}s")
        print(f"  Started:  {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("  Agents:")
        print("    Analysis      - Market scanning & signals")
        print("    Accumulation  - BTC vs SOL target (from brain)")
        print("    Strategy      - Strategy evaluation")
        print("    Risk          - Trade validation")
        print("    Trading       - Quote & execution")
        print("=" * 70)

        while self.running:
            try:
                result = await self.run_cycle()
                print(f"\n--- Cycle {result['cycle']} complete "
                      f"| SOL=${result['analysis']['sol_price']:.2f} "
                      f"| Signal={result['analysis']['signal']} "
                      f"| Action={result['decision']['action'] if result.get('decision') else 'WAIT'} "
                      f"| Trades={result['trades_today']} ---")

            except Exception as e:
                logger.error(f"Cycle error: {e}")
                print(f"\n  ERROR in cycle {self.cycle}: {e}")

            # Write state file for dashboard
            self._write_state()

            # Wait for next cycle
            for i in range(self.interval):
                if not self.running:
                    break
                await asyncio.sleep(1)

        await self.jupiter.close()
        if self.sol_client:
            await self.sol_client.close()
        print("\nAgents stopped.")

    def _write_state(self):
        """Write current state to JSON for dashboard to read."""
        state = {
            "running": self.running,
            "cycle": self.cycle,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_update": datetime.now().isoformat(),
            "mode": "live" if self.live else "dry_run",
            "wallet": self.wallet,
            "strategies": self.strategist.get_summary(),
            "risk": {
                "trades_today": self.risk.trades_today,
                "daily_pnl": self.risk.daily_pnl
            },
            "accumulation": self.accumulation.current,
            "order_history": self.trader.order_history[-10:],
            "recent_activity": self.activity_log[-20:],
            "price_history": self.analyst.price_history[-30:]
        }
        state_file = PROJECT_ROOT / "agent_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def stop(self):
        self.running = False


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent Trading Runner")
    parser.add_argument("--live", action="store_true", help="Enable live execution (devnet)")
    parser.add_argument("--interval", type=int, default=120, help="Seconds between cycles (default: 120)")
    parser.add_argument("--wallet", type=str, default=None, help="Wallet address override")
    args = parser.parse_args()

    # Load wallet
    wallet = args.wallet
    if not wallet:
        wallet_file = Path.home() / ".config" / "solana-jupiter-bot" / "wallet.json"
        if wallet_file.exists():
            data = json.loads(wallet_file.read_text())
            if isinstance(data, dict) and "address" in data:
                wallet = data["address"]
            else:
                from solders.keypair import Keypair
                kp = Keypair.from_bytes(bytes(data))
                wallet = str(kp.pubkey())
            print(f"Wallet loaded: {wallet}")
        else:
            print("ERROR: No wallet found. Run: python3 tools/solana_wallet.py --generate")
            sys.exit(1)

    coordinator = AgentCoordinator(
        wallet_address=wallet,
        live=args.live,
        interval=args.interval
    )

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\nStopping agents...")
        coordinator.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    asyncio.run(coordinator.run())


if __name__ == "__main__":
    main()
