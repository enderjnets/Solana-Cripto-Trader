#!/usr/bin/env python3
"""
MASTER ORCHESTRATOR - Sistema Multi-Agente para 5% Diario
=========================================================
Coordina: Researcher → Strategy → Backtest → Auditor → Trading

Objetivo: 5% diario con max 10% drawdown
"""

import os
import sys
import json
import time
import asyncio
import subprocess
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
            "daily_pnl": 0.0,
            "total_pnl": 0.0,
            "cycles": 0
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
        
        # Simulate research (replace with real logic)
        opportunities = [
            {"token": "BTC", "signal": "dip", "entry": 408, "target": 420, "confidence": 0.8},
            {"token": "SOL", "signal": "breakout", "entry": 80, "target": 85, "confidence": 0.7}
        ]
        
        self.state.data["agents"]["researcher"]["findings"] = opportunities
        self.state.data["agents"]["researcher"]["last_run"] = datetime.now().isoformat()
        self.state.data["agents"]["researcher"]["status"] = "idle"
        self.state.data["opportunities"] = opportunities
        self.state.save()
        
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
        
        for opp in opportunities:
            # Simulated backtest result
            result = {
                "token": opp["token"],
                "sharpe": 1.5 + (hash(opp["token"]) % 100) / 100,
                "win_rate": 0.6 + (hash(opp["token"]) % 30) / 100,
                "max_dd": 0.08,
                "avg_profit": 0.03,
                "approved": True
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
            approved.append(strat)
        
        self.state.data["agents"]["auditor"]["approved"] = approved
        self.state.data["agents"]["auditor"]["last_run"] = datetime.now().isoformat()
        self.state.data["agents"]["auditor"]["status"] = "idle"
        self.state.save()
        
        if approved:
            self.state.log(f"✅ AUDITOR: {len(approved)} trades aprobados para ejecución")


# ============================================================================
# MASTER ORCHESTRATOR
# ============================================================================
class MasterOrchestrator:
    def __init__(self):
        self.state = MasterState()
        self.researcher = ResearcherAgent(self.state)
        self.backtester = BacktesterAgent(self.state)
        self.auditor = AuditorAgent(self.state)
        self.running = True
        
    async def run(self):
        self.state.log("=" * 50)
        self.state.log("🎯 MASTER ORCHESTRATOR INICIADO")
        self.state.log(f"   Meta: {DAILY_TARGET*100}% diario")
        self.state.log(f"   Max Drawdown: {MAX_DRAWDOWN*100}%")
        self.state.log("=" * 50)
        
        cycle = 0
        while self.running:
            cycle += 1
            self.state.data["cycles"] = cycle
            
            try:
                # Run research periodically
                last_research = self.state.data["agents"]["researcher"].get("last_run")
                if not last_research or \
                   (datetime.now() - datetime.fromisoformat(last_research)).seconds > RESEARCH_INTERVAL:
                    await self.researcher.run()
                    
                # Run backtest periodically  
                last_backtest = self.state.data["agents"]["backtester"].get("last_run")
                if not last_backtest or \
                   (datetime.now() - datetime.fromisoformat(last_backtest)).seconds > BACKTEST_INTERVAL:
                    await self.backtester.run()
                    
                # Run audit every cycle
                await self.auditor.run()
                
                # Log status
                self.state.log(f"📊 Ciclo {cycle} | "
                             f"Daily: {self.state.data['daily_pnl']*100:.2f}% | "
                             f"Estrategias: {len(self.state.data['approved_strategies'])}")
                
            except Exception as e:
                self.state.log(f"❌ Error: {e}")
                
            await asyncio.sleep(AUDIT_INTERVAL)
            
    def stop(self):
        self.running = False


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("🎯 Starting Master Orchestrator...")
    orchestrator = MasterOrchestrator()
    
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        print("\n🛑 Orchestrator stopped")
        orchestrator.stop()
