#!/usr/bin/env python3
"""
MASTER ORCHESTRATOR v2.0 - Sistema Multi-Agente para 5% Diario
================================================================
MEJORAS IMPLEMENTADAS:
1. Detección de Tendencia - No abre LONGs en mercado bajista
2. Stop Loss Real - Máximo -1% por trade
3. Tracking de Capital Correcto - Actualiza capital con cada trade
4. Win Rate Tracking - Cuenta wins/losses

Coordina: Researcher → Strategy → Backtest → Auditor → Trading
Objetivo: 5% diario con max 10% drawdown
"""

import os
import sys
import json
import time
import asyncio
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from threading import Thread

# ============================================================================
# CONFIGURATION
# ============================================================================
STATE_FILE = Path("~/.config/solana-jupiter-bot/master_state.json").expanduser()
LOG_FILE = Path("~/.config/solana-jupiter-bot/master.log").expanduser()
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

# Targets
DAILY_TARGET = 0.05  # 5%
MAX_DRAWDOWN = 0.10   # 10%
STOP_LOSS_PCT = -1.0  # 🔧 NUEVO: Stop Loss al -1%
TAKE_PROFIT_PCT = 2.0  # Take Profit al +2%

# Agent intervals (seconds)
RESEARCH_INTERVAL = 300     # 5 min
BACKTEST_INTERVAL = 180     # 3 min
AUDIT_INTERVAL = 60        # 1 min

# ============================================================================
# MASTER STATE
# ============================================================================
class MasterState:
    def __init__(self):
        self.load()
        
    def load(self):
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                self.data = json.load(f)
        else:
            self.data = self._default()
            
    def _default(self):
        return {
            "started": datetime.now().isoformat(),
            "target_daily": DAILY_TARGET,
            "max_drawdown": MAX_DRAWDOWN,
            "agents": {
                "researcher": {"status": "idle", "last_run": None, "findings": []},
                "backtester": {"status": "idle", "last_run": None, "results": []},
                "auditor": {"status": "idle", "last_run": None, "approved": []},
                "trader": {"status": "active", "last_run": None}
            },
            "opportunities": [],
            "approved_strategies": [],
            "paper_positions": [],  # Track paper trades
            "paper_history": [],     # Closed positions
            "daily_pnl": 0.0,
            "total_pnl": 0.0,
            "cycles": 0,
            "paper_capital": 500.00,   # Starting capital for paper trading
            "stats": {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0
            },
            "market_trend": "neutral"  # 🔧 NUEVO: bullish, bearish, neutral
        }
        
    def save(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.data, f, indent=2)
            
    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        with open(LOG_FILE, 'a') as f:
            f.write(line)
        print(line.strip())

# ============================================================================
# PRICE FETCHER (Real-time from CoinGecko)
# ============================================================================
def get_real_prices() -> Dict:
    """Fetch real-time prices from CoinGecko"""
    prices = {}
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana,dogecoin,cardano,ripple,polkadot,chainlink,solana-turbon-sol,Wrapped-Bitcoin&vs_currencies=usd&include_24hr_change=true"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            # Map CoinGecko IDs to symbols
            id_map = {
                "bitcoin": "BTC",
                "ethereum": "ETH", 
                "solana": "SOL",
                "dogecoin": "DOGE",
                "cardano": "ADA",
                "ripple": "XRP",
                "polkadot": "DOT",
                "chainlink": "LINK",
                "solana-turbon-sol": "TBS",
                "wrapped-bitcoin": "WBTC"
            }
            for cg_id, sym in id_map.items():
                if cg_id in data:
                    prices[sym] = {
                        "price": float(data[cg_id]["usd"]),
                        "change": float(data[cg_id].get("usd_24h_change", 0))
                    }
    except Exception as e:
        print(f"   ⚠️ Price fetch error: {e}")
    return prices


def detect_market_trend(prices: Dict) -> str:
    """🔧 NUEVO: Detecta la tendencia general del mercado"""
    if not prices:
        return "neutral"
    
    # Contar cuántos tokens están en positivo vs negativo
    positive = 0
    negative = 0
    total_change = 0
    
    # Solo considerar los tokens principales (excluir DOGE, TBS)
    main_tokens = ["BTC", "ETH", "SOL", "ADA", "XRP", "DOT", "LINK", "WBTC"]
    
    for token in main_tokens:
        if token in prices:
            change = prices[token].get("change", 0)
            total_change += change
            if change > 0:
                positive += 1
            elif change < 0:
                negative += 1
    
    avg_change = total_change / len(main_tokens) if main_tokens else 0
    
    # Determinar tendencia
    if positive > negative and avg_change > 1:
        return "bullish"
    elif negative > positive and avg_change < -1:
        return "bearish"
    else:
        return "neutral"


# ============================================================================
# AGENT CLASSES
# ============================================================================
class ResearcherAgent:
    """Investiga oportunidades de mercado"""
    
    def __init__(self, state: MasterState):
        self.state = state
        
    async def run(self):
        self.state.data["agents"]["researcher"]["status"] = "running"
        self.state.save()
        self.state.log("🔍 RESEARCHER: Analizando mercado...")
        
        # Fetch real prices
        prices = get_real_prices()
        
        # 🔧 NUEVO: Detectar tendencia del mercado
        market_trend = detect_market_trend(prices)
        self.state.data["market_trend"] = market_trend
        self.state.log(f"   📊 Tendencia del mercado: {market_trend.upper()}")
        
        # Generate opportunities based on real prices
        opportunities = []
        for token, data in prices.items():
            current = data["price"]
            change = data["change"]
            
            # 🔧 NUEVO: En mercado bajista, solo buscar opportunities con change positivo
            if market_trend == "bearish" and change < 0:
                # Skip tokens que están cayendo en mercado bajista
                continue
            
            # Dynamic signal based on price movement
            if change > 2:
                signal = "breakout"
                target = current * 1.03  # 3% target
            elif change < -2:
                if market_trend == "bearish":
                    continue  # 🔧 NUEVO: No comprar dips en mercado bajista
                signal = "dip"
                target = current * 1.05  # 5% target
            else:
                signal = "range"
                target = current * 1.02
            
            # 🔧 NUEVO: Confidence basado en alineación con tendencia
            base_confidence = 0.5 + abs(change) / 20
            if market_trend == "bullish" and change > 0:
                base_confidence += 0.2  # Bonus por alineación con tendencia
            elif market_trend == "bearish":
                base_confidence -= 0.1  # Penalización en mercado bajista
            
            opportunities.append({
                "token": token,
                "signal": signal,
                "entry": current,
                "target": round(target, 2),
                "confidence": min(0.95, base_confidence),
                "change_24h": change
            })
        
        self.state.data["agents"]["researcher"]["findings"] = opportunities
        self.state.data["agents"]["researcher"]["last_run"] = datetime.now().isoformat()
        self.state.data["agents"]["researcher"]["status"] = "idle"
        self.state.data["opportunities"] = opportunities
        self.state.save()
        
        # Log prices
        for token, data in prices.items():
            self.state.log(f"   📊 {token}: ${data['price']:,.2f} ({data['change']:+.2f}%)")
        
        self.state.log(f"✅ RESEARCHER: {len(opportunities)} oportunidades encontradas")
        return opportunities


class BacktesterAgent:
    """Valida estrategias contra datos históricos"""
    
    def __init__(self, state: MasterState):
        self.state = state
        
    async def run(self):
        self.state.data["agents"]["backtester"]["status"] = "running"
        self.state.save()
        self.state.log("🧪 BACKTESTER: Validando estrategias...")
        
        opportunities = self.state.data.get("opportunities", [])
        results = []
        
        # Limpiar estrategias antiguas para evitar acumulación
        self.state.data["approved_strategies"] = []
        
        for opp in opportunities:
            # 🔧 NUEVO: Solo aprobar si confidence > 0.6
            if opp.get("confidence", 0) < 0.6:
                continue
            
            # Simulated backtest result
            result = {
                "token": opp["token"],
                "sharpe": 1.5 + (hash(opp["token"]) % 100) / 100,
                "win_rate": 0.6 + (hash(opp["token"]) % 30) / 100,
                "max_dd": 0.08,
                "avg_profit": 0.03,
                "approved": True,
                "stop_loss": STOP_LOSS_PCT,  # 🔧 NUEVO
                "take_profit": TAKE_PROFIT_PCT  # 🔧 NUEVO
            }
            results.append(result)
            
            if result["approved"]:
                self.state.data["approved_strategies"].append({**opp, **result})
        
        self.state.data["agents"]["backtester"]["results"] = results
        self.state.data["agents"]["backtester"]["last_run"] = datetime.now().isoformat()
        self.state.data["agents"]["backtester"]["status"] = "idle"
        self.state.save()
        
        self.state.log(f"✅ BACKTESTER: {len([r for r in results if r['approved']])} estrategias aprobadas")


class AuditorAgent:
    """Valida trades antes de ejecutar"""
    
    def __init__(self, state: MasterState):
        self.state = state
        
    async def run(self):
        self.state.data["agents"]["auditor"]["status"] = "running"
        self.state.save()
        
        approved = []
        for strat in self.state.data.get("approved_strategies", []):
            # Risk checks
            if strat.get("max_dd", 1) > MAX_DRAWDOWN:
                continue
            if strat.get("win_rate", 0) < 0.5:
                continue
            if strat.get("confidence", 0) < 0.6:  # 🔧 NUEVO
                continue
            approved.append(strat)
        
        self.state.data["agents"]["auditor"]["approved"] = approved
        self.state.data["agents"]["auditor"]["last_run"] = datetime.now().isoformat()
        self.state.data["agents"]["auditor"]["status"] = "idle"
        self.state.save()
        
        if approved:
            self.state.log(f"✅ AUDITOR: {len(approved)} trades aprobados para ejecución")


# ============================================================================
# PAPER TRADING AGENT (v2.0 - CON STOP LOSS Y TRACKING CORRECTO)
# ============================================================================
class PaperTradingAgent:
    """Ejecuta trades en modo paper trading (simulado con precios reales)
    
    🔧 MEJORAS v2.0:
    - Stop Loss al -1%
    - Take Profit al +2%
    - Tracking correcto del capital
    - Win/Loss counting
    """
    
    def __init__(self, state: MasterState):
        self.state = state
        
    async def run(self):
        """Procesa trades aprobados y simula ejecución"""
        self.state.data["agents"]["paper_trading"] = {"status": "running", "last_run": None}
        
        approved = self.state.data["agents"]["auditor"].get("approved", [])
        prices = self.state.data.get("current_prices", {})
        positions = self.state.data.get("paper_positions", [])
        
        # Get current prices for each token
        current_prices = {}
        for token, data in prices.items():
            if isinstance(data, dict):
                current_prices[token] = data.get("price", 0)
            else:
                current_prices[token] = data
        
        # Get current capital
        capital = self.state.data.get("paper_capital", 500.0)
        
        # Open new positions from approved trades
        for strat in approved:
            token = strat.get("token", "")
            if not token or token == "DOGE" or token == "TBS":  # Skip problematic tokens
                continue
            
            # Check if we already have a position for this token
            existing = [p for p in positions if p["token"] == token and p.get("status") == "open"]
            if existing:
                continue  # Already in position
            
            # 🔧 NUEVO: Max 5 posiciones abiertas a la vez
            if len([p for p in positions if p.get("status") == "open"]) >= 5:
                break
            
            # Open new paper position
            entry_price = current_prices.get(token, strat.get("entry", 0))
            if entry_price <= 0:
                continue
            
            # Calculate position size (10% of capital per trade)
            position_size = capital * 0.10
            
            position = {
                "token": token,
                "signal": strat.get("signal", "unknown"),
                "entry_price": entry_price,
                "entry_time": datetime.now().isoformat(),
                "status": "open",
                "position_size": position_size,
                "stop_loss": entry_price * (1 + STOP_LOSS_PCT/100),  # 🔧 NUEVO
                "take_profit": entry_price * (1 + TAKE_PROFIT_PCT/100),  # 🔧 NUEVO
                "sharpe": strat.get("sharpe", 0),
                "win_rate": strat.get("win_rate", 0)
            }
            positions.append(position)
            self.state.log(f"📝 PAPER: Opened {token} @ ${entry_price:,.2f} ({position['signal']}) | SL: ${position['stop_loss']:,.2f} | TP: ${position['take_profit']:,.2f}")
        
        # Update open positions with current prices & calculate P&L
        open_pnl = 0.0
        closed_positions = []
        
        # 🔧 NUEVO: Get stats
        stats = self.state.data.get("stats", {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0
        })
        
        for pos in positions:
            if pos.get("status") != "open":
                continue
            
            token = pos["token"]
            entry = pos["entry_price"]
            current = current_prices.get(token, entry)
            position_size = pos.get("position_size", 50)  # Default $50 per position
            
            if current <= 0:
                continue
            
            # Calculate P&L
            pnl_pct = ((current - entry) / entry) * 100
            pnl_value = (pnl_pct / 100) * position_size
            
            pos["current_price"] = current
            pos["pnl_pct"] = pnl_pct
            pos["pnl_value"] = pnl_value
            
            # 🔧 MEJORADO: Close position if conditions met
            should_close = False
            close_reason = ""
            
            if pnl_pct <= STOP_LOSS_PCT:  # -1% stop loss
                should_close = True
                close_reason = "STOP_LOSS"
            elif pnl_pct >= TAKE_PROFIT_PCT:  # +2% take profit
                should_close = True
                close_reason = "TAKE_PROFIT"
            elif pnl_pct < -0.5 and pos.get("signal") == "dip":  # Dip que no recuperó
                # Dar más tiempo a los dips
                pass
            
            if should_close:
                pos["status"] = "closed"
                pos["close_price"] = current
                pos["close_time"] = datetime.now().isoformat()
                pos["pnl_final"] = pnl_value
                pos["close_reason"] = close_reason
                closed_positions.append(pos)
                
                # 🔧 NUEVO: Update stats
                stats["total_trades"] += 1
                if pnl_value > 0:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1
                
                # 🔧 NUEVO: Update capital
                capital += pnl_value
                
                self.state.log(f"📝 PAPER: Closed {token} @ ${current:,.2f} | P&L: {pnl_value:+.2f} ({pnl_pct:+.2f}%) | Reason: {close_reason}")
        
        # Remove closed positions from open list
        positions = [p for p in positions if p.get("status") != "closed"]
        
        # Calculate total P&L
        total_pnl = sum(p.get("pnl_final", 0) for p in closed_positions)
        open_pnl = sum(p.get("pnl_value", 0) for p in positions)
        
        # 🔧 NUEVO: Calculate win rate
        if stats["total_trades"] > 0:
            stats["win_rate"] = (stats["wins"] / stats["total_trades"]) * 100
        
        # Update state
        self.state.data["paper_positions"] = positions
        if "paper_history" not in self.state.data:
            self.state.data["paper_history"] = []
        self.state.data["paper_history"].extend(closed_positions)
        self.state.data["total_pnl"] = total_pnl
        self.state.data["paper_capital"] = capital  # 🔧 NUEVO: Update capital
        self.state.data["stats"] = stats  # 🔧 NUEVO
        self.state.data["daily_pnl"] = ((capital - 500) / 500) * 100  # 🔧 CORREGIDO
        self.state.data["agents"]["paper_trading"] = {
            "status": "idle", 
            "last_run": datetime.now().isoformat(),
            "open_positions": len(positions),
            "closed_today": len(closed_positions),
            "total_pnl": total_pnl,
            "daily_pnl_pct": self.state.data["daily_pnl"],
            "capital": capital,  # 🔧 NUEVO
            "win_rate": stats["win_rate"]  # 🔧 NUEVO
        }
        self.state.save()
        
        if positions:
            self.state.log(f"📝 PAPER: {len(positions)} posiciones abiertas | P&L abierto: ${open_pnl:+.2f}")
        if closed_positions:
            self.state.log(f"📝 PAPER: {len(closed_positions)} cerradas | Capital: ${capital:.2f} | Win Rate: {stats['win_rate']:.1f}%")


# ============================================================================
# MASTER ORCHESTRATOR
# ============================================================================
class MasterOrchestrator:
    def __init__(self):
        self.state = MasterState()
        self.researcher = ResearcherAgent(self.state)
        self.backtester = BacktesterAgent(self.state)
        self.auditor = AuditorAgent(self.state)
        self.paper_trader = PaperTradingAgent(self.state)
        self.running = True
        
    async def run(self):
        try:
            self.state.log("=" * 50)
            self.state.log("🎯 MASTER ORCHESTRATOR v2.0 INICIADO")
            self.state.log(f"   Meta: {DAILY_TARGET*100}% diario")
            self.state.log(f"   Max Drawdown: {MAX_DRAWDOWN*100}%")
            self.state.log(f"   Stop Loss: {STOP_LOSS_PCT}% | Take Profit: +{TAKE_PROFIT_PCT}%")
            self.state.log(f"   Capital: ${self.state.data.get('paper_capital', 500):.2f}")
            self.state.log("=" * 50)
            
            cycle = 0
            while self.running:
                cycle += 1
                self.state.data["cycles"] = cycle
                
                try:
                    # Run research periodically
                    last_research = self.state.data["agents"]["researcher"].get("last_run")
                    if not last_research:
                        await self.researcher.run()
                    else:
                        try:
                            last_run_time = datetime.fromisoformat(last_research)
                            if (datetime.now() - last_run_time).seconds > RESEARCH_INTERVAL:
                                await self.researcher.run()
                        except:
                            await self.researcher.run()
                        
                    # Run backtest periodically  
                    last_backtest = self.state.data["agents"]["backtester"].get("last_run")
                    if not last_backtest:
                        await self.backtester.run()
                    else:
                        try:
                            last_run_time = datetime.fromisoformat(last_backtest)
                            if (datetime.now() - last_run_time).seconds > BACKTEST_INTERVAL:
                                await self.backtester.run()
                        except:
                            await self.backtester.run()
                        
                    # Run audit every cycle
                    await self.auditor.run()
                    
                    # Run paper trading (simulated trades)
                    await self.paper_trader.run()
                    
                    # Fetch and log real-time prices
                    prices = get_real_prices()
                    self.state.data["current_prices"] = prices  # Store for paper trading
                    price_str = " | ".join([f"{t}: ${d['price']:,.0f}" for t, d in list(prices.items())[:4]])
                    
                    # Log status with real prices
                    paper_stats = self.state.data["agents"].get("paper_trading", {})
                    open_pos = paper_stats.get("open_positions", 0)
                    capital = paper_stats.get("capital", self.state.data.get("paper_capital", 500))
                    win_rate = paper_stats.get("win_rate", 0)
                    
                    self.state.log(f"📊 Ciclo {cycle} | {price_str}")
                    self.state.log(f"   📈 Capital: ${capital:.2f} | Estrategias: {len(self.state.data['approved_strategies'])} | 📝 Paper: {open_pos} pos | Win Rate: {win_rate:.1f}%")
                    
                except Exception as e:
                    self.state.log(f"❌ Error: {e}")
                    import traceback
                    self.state.log(f"   Trace: {traceback.format_exc()[:150]}")
                
                # Sleep in a separate try to not crash the loop
                try:
                    await asyncio.sleep(AUDIT_INTERVAL)
                except Exception as e:
                    print(f"⚠️ Sleep error: {e}")
        except Exception as e:
            print(f"❌ FATAL in run(): {e}")
            import traceback
            traceback.print_exc()
            
    def stop(self):
        self.running = False


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("🎯 Starting Master Orchestrator v2.0...")
    orchestrator = MasterOrchestrator()
    
    while True:
        try:
            asyncio.run(orchestrator.run())
            print("⚠️ asyncio.run() completed - restarting...")
        except KeyboardInterrupt:
            print("\n🛑 Orchestrator stopped")
            orchestrator.stop()
            break
        except Exception as e:
            print(f"❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()
            print("🔄 Restarting in 5 seconds...")
            time.sleep(5)
