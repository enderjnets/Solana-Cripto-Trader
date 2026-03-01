#!/usr/bin/env python3
"""
Paper Trading Engine
===================
Simulates trades without executing them on blockchain.

Usage:
    python3 paper_trading_engine.py --start     # Start paper trading
    python3 paper_trading_engine.py --status    # Check status
    python3 paper_trading_engine.py --reset     # Reset balance
"""

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logger = logging.getLogger("paper_trading_engine")

class DateTimeEncoder(json.JSONEncoder):
    """Handle datetime serialization."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


from typing import Dict, List, Optional
from dataclasses import dataclass, field

# Storage
PAPER_STATE_FILE = Path(__file__).parent / "data" / "paper_trading_state.json"


@dataclass
class PaperTrade:
    """Simulated trade."""
    id: str
    entry_time: datetime
    entry_price: float
    direction: str  # long, short
    size: float
    symbol: str
    status: str = "open"  # open, closed
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""


@dataclass
class PaperTradingState:
    """Paper trading state."""
    enabled: bool = False
    start_time: Optional[datetime] = None
    balance_usd: float = 500.0  # Start with $500
    initial_balance: float = 500.0
    leverage: int = 1  # Default 1x (like spot), supports up to 20x like Drift
    margin_used: float = 0.0  # Total margin locked
    liquidation_threshold: float = 0.005  # 0.5% maintenance margin (like Drift)
    maker_fee: float = 0.0002  # 0.02% maker fee (like Drift)
    taker_fee: float = 0.0005  # 0.05% taker fee (like Drift)
    trades: List[Dict] = field(default_factory=list)
    stats: Dict = field(default_factory=lambda: {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "total_pnl": 0.0,
        "total_fees": 0.0,
        "liquidations": 0,
        "win_rate": 0.0,
        "max_drawdown": 0.0,
        "current_streak": 0,
        "best_streak": 0,
        "worst_streak": 0,
    })
    signals: List[Dict] = field(default_factory=list)


class PaperTradingEngine:
    """Simulates trades without real execution."""

    def __init__(self):
        self.state = self._load_state()

    def _load_state(self) -> PaperTradingState:
        """Load state from file."""
        if PAPER_STATE_FILE.exists():
            data = json.loads(PAPER_STATE_FILE.read_text())
            # Convert datetime strings back
            if data.get("start_time"):
                data["start_time"] = datetime.fromisoformat(data["start_time"])
            for trade in data.get("trades", []):
                if trade.get("entry_time"):
                    trade["entry_time"] = datetime.fromisoformat(trade["entry_time"])
                if trade.get("exit_time"):
                    trade["exit_time"] = datetime.fromisoformat(trade["exit_time"])
            # FIXED: Don't change initial_balance - it should stay fixed
            # ALWAYS recalculate margin_used from open trades (fix for corrupted values)
            open_trades = [t for t in data.get("trades", []) if t.get("status") == "open"]
            data["margin_used"] = sum(t.get("margin", t.get("size", 0)) for t in open_trades)
            return PaperTradingState(**data)
        return PaperTradingState()

    def _save_state(self):
        """Save state to file."""
        # Build data dict manually to avoid datetime serialization issues
        data = {
            "enabled": self.state.enabled,
            "start_time": self.state.start_time.isoformat() if self.state.start_time else None,
            "balance_usd": self.state.balance_usd,
            "initial_balance": self.state.initial_balance,
            "leverage": self.state.leverage,
            "margin_used": self.state.margin_used,  # CRITICAL: Save margin_used
            "trades": [],
            "stats": self.state.stats.copy(),
            "signals": []
        }
        
        # Convert trades manually
        for trade in self.state.trades:
            trade_data = {
                "id": trade["id"],
                "entry_time": trade["entry_time"].isoformat() if hasattr(trade["entry_time"], 'isoformat') else str(trade["entry_time"]),
                "entry_price": trade["entry_price"],
                "direction": trade["direction"],
                "size": trade["size"],
                "symbol": trade["symbol"],
                "status": trade["status"],
                "exit_time": trade["exit_time"].isoformat() if trade["exit_time"] and hasattr(trade["exit_time"], 'isoformat') else (str(trade["exit_time"]) if trade["exit_time"] else None),
                "exit_price": trade["exit_price"],
                "pnl": trade["pnl"],
                "pnl_pct": trade["pnl_pct"],
                "reason": trade["reason"],
                "confidence": trade.get("confidence", 0.0)  # Persist ML confidence
            }
            data["trades"].append(trade_data)
        
        # Convert signals manually
        for sig in self.state.signals[-100:]:
            sig_data = {
                "time": sig["time"].isoformat() if hasattr(sig["time"], 'isoformat') else str(sig["time"]),
                "symbol": sig["symbol"],
                "direction": sig["direction"],
                "price": sig["price"],
                "size": sig["size"],
                "reason": sig["reason"],
                "trade_id": sig["trade_id"]
            }
            data["signals"].append(sig_data)
        
        # Save with custom encoder
        PAPER_STATE_FILE.write_text(json.dumps(data, indent=2, default=str))

    def start(self):
        """Start paper trading."""
        if not self.state.enabled:
            self.state.enabled = True
            self.state.start_time = datetime.now()
            self._save_state()
            print(f"\n✅ PAPER TRADING STARTED")
            print(f"   Start Time: {self.state.start_time}")
            print(f"   Initial Balance: ${self.state.initial_balance:,.2f}")
            print(f"   Current Balance: ${self.state.balance_usd:,.2f}")
        else:
            print(f"\n⚠️ Paper trading already running")
            self.status()

    def stop(self):
        """Stop paper trading."""
        self.state.enabled = False
        self._save_state()
        print(f"\n🛑 PAPER TRADING STOPPED")

    def reset(self):
        """Reset paper trading state."""
        self.state = PaperTradingState()
        self._save_state()
        print(f"\n🔄 PAPER TRADING RESET")
        print(f"   Balance: ${self.state.balance_usd:,.2f}")

    def status(self):
        """Show current status."""
        print(f"\n📊 PAPER TRADING STATUS")
        print(f"{'='*50}")
        print(f"   Status: {'🟢 RUNNING' if self.state.enabled else '🔴 STOPPED'}")
        if self.state.start_time:
            elapsed = datetime.now() - self.state.start_time
            print(f"   Runtime: {elapsed}")
        print(f"   Balance: ${self.state.balance_usd:,.2f}")
        print(f"   P&L: ${self.state.stats['total_pnl']:,.2f} ({self._get_pnl_pct():.2f}%)")
        print(f"{'='*50}")
        print(f"   Total Trades: {self.state.stats['total_trades']}")
        print(f"   Win Rate: {self.state.stats['win_rate']:.1f}%")
        print(f"   Streak: {self.state.stats['current_streak']} (+/-)")
        print(f"{'='*50}")

    def _get_pnl_pct(self) -> float:
        """Calculate P&L percentage."""
        if self.state.initial_balance == 0:
            return 0
        return (self.state.balance_usd - self.state.initial_balance) / self.state.initial_balance * 100

    def set_leverage(self, leverage: int) -> bool:
        """Set leverage (1-20x like Drift)."""
        if leverage < 1 or leverage > 20:
            return False
        self.state.leverage = leverage
        self._save_state()
        return True
    
    def get_leverage_info(self) -> Dict:
        """Get current leverage info (Drift-like)."""
        return {
            "leverage": self.state.leverage,
            "margin_used": self.state.margin_used,
            "available_leverage": [1, 2, 5, 10, 20],
            "effective_exposure": self.state.margin_used * self.state.leverage if self.state.margin_used else 0,
            "liquidation_threshold": self.state.liquidation_threshold,
            "maker_fee": self.state.maker_fee,
            "taker_fee": self.state.taker_fee,
            "total_fees": self.state.stats.get("total_fees", 0),
            "liquidations": self.state.stats.get("liquidations", 0)
        }

    def execute_signal(self, signal: Dict) -> Optional[PaperTrade]:
        """
        Execute a trading signal (paper) with leverage support.
        
        Like Drift Protocol: up to 20x leverage.

        Args:
            signal: Dict with 'symbol', 'direction', 'price', 'size', 'reason', 'leverage'

        Returns:
            PaperTrade if executed, None otherwise
        """
        if not self.state.enabled:
            return None

        # SAFETY CHECK: Stop trading if balance is below minimum threshold
        MIN_BALANCE_THRESHOLD = 50.0  # $50 minimum to continue trading
        if self.state.balance_usd < MIN_BALANCE_THRESHOLD:
            logger.warning(f"⚠️ TRADING STOPPED: Balance ${self.state.balance_usd:.2f} below threshold ${MIN_BALANCE_THRESHOLD}")
            self.state.enabled = False
            self._save_state()
            return None

        # SAFETY CHECK: Stop if drawdown is too severe (>80%)
        drawdown_pct = (self.state.initial_balance - self.state.balance_usd) / self.state.initial_balance * 100
        if drawdown_pct > 80:
            logger.warning(f"⚠️ TRADING STOPPED: Drawdown {drawdown_pct:.1f}% exceeds 80% limit")
            self.state.enabled = False
            self._save_state()
            return None

        symbol = signal.get("symbol", "SOL")
        direction = signal.get("direction", "long")
        price = signal.get("price", 86.0)
        size_usd = signal.get("size", self.state.balance_usd * 0.1)  # 10% of balance
        leverage = signal.get("leverage", self.state.leverage)  # Use configured leverage
        reason = signal.get("reason", "Signal")
        confidence = signal.get("confidence", 0.0)  # ML confidence (0-100)

        # SAFETY CHECK: Limit position size to max 15% of current balance
        MAX_POSITION_PCT = 0.15  # 15% max position
        max_position_usd = self.state.balance_usd * MAX_POSITION_PCT
        if size_usd > max_position_usd:
            size_usd = max_position_usd

        # Apply leverage: position size = size_usd * leverage
        leveraged_size = size_usd * leverage
        
        # Margin required = leveraged_size / leverage = size_usd
        # But we limit by available margin
        margin_required = size_usd  # Base margin
        
        # SAFETY CHECK: Ensure enough balance for margin + fees
        entry_fee = leveraged_size * self.state.taker_fee
        total_required = margin_required + entry_fee
        available_balance = self.state.balance_usd - self.state.margin_used
        
        if total_required > available_balance:
            logger.warning(f"⚠️ INSUFFICIENT FUNDS: Need ${total_required:.2f}, have ${available_balance:.2f}")
            return None  # Not enough margin + fees

        # Count open trades
        open_trades = [t for t in self.state.trades if t["status"] == "open"]
        max_concurrent = 5  # Maximum concurrent trades
        
        if len(open_trades) >= max_concurrent:
            return None  # Max concurrent trades reached

        # DEDUCIR MARGIN del balance (no el tamaño completo)
        self.state.margin_used += margin_required
        self.state.balance_usd -= margin_required  # ← BUG FIX: Deduct margin from balance

        # Deduct entry fee (taker fee like Drift)
        self.state.balance_usd -= entry_fee
        
        # Create trade - use consistent "trade_" prefix
        import uuid
        trade = PaperTrade(
            id=f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}",
            entry_time=datetime.now(),
            entry_price=price,
            direction=direction,
            size=size_usd,
            symbol=symbol
        )

        # Log signal
        self.state.signals.append({
            "time": trade.entry_time.isoformat(),
            "symbol": symbol,
            "direction": direction,
            "price": price,
            "size": size_usd,
            "reason": reason,
            "trade_id": trade.id
        })

        # Add trade to list
        self.state.trades.append({
            "id": trade.id,
            "entry_time": trade.entry_time.isoformat(),
            "entry_price": trade.entry_price,
            "direction": trade.direction,
            "size": leveraged_size,  # Notional size with leverage
            "margin": margin_required,  # Actual margin used
            "leverage": leverage,
            "entry_fee": entry_fee,  # Fee paid on entry
            "symbol": trade.symbol,
            "status": trade.status,
            "exit_time": None,
            "exit_price": None,
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "reason": reason,
            "confidence": confidence  # ML confidence stored with trade
        })

        self._save_state()
        return trade

    def close_trade(self, trade_id: str, exit_price: float, reason: str = "Close", leverage: int = 1):
        """Close an open trade with leverage."""
        # Find trade
        for trade_data in self.state.trades:
            if trade_data["id"] == trade_id and trade_data["status"] == "open":
                # Calculate P&L
                entry_price = trade_data["entry_price"]
                direction = trade_data["direction"]
                size = trade_data["size"]  # This is the NOTIONAL size (with leverage applied)
                margin = trade_data.get("margin", size / leverage)  # Original margin
                entry_fee = trade_data.get("entry_fee", 0)  # Fee paid on entry

                # Calculate P&L with leverage effect
                # P&L = notional * price_change%
                if direction in ["bullish", "long"]:
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:  # bearish or short
                    pnl_pct = (entry_price - exit_price) / entry_price

                # Apply leverage to P&L
                pnl = size * pnl_pct
                
                # Deduct exit fee (taker fee like Drift)
                exit_fee = size * self.state.taker_fee
                pnl -= exit_fee
                
                # MAX LOSS LIMIT: Cannot lose more than the margin invested
                max_loss = -margin  # Maximum possible loss is the margin
                if pnl < max_loss:
                    pnl = max_loss  # Cap the loss at the margin
                    reason = "MAX_LOSS_LIMIT"
                
                # Check for liquidation (like Drift)
                # Liquidation happens when position loses > (1 - maintenance_margin) * margin
                maintenance_margin = self.state.liquidation_threshold * 100  # 50% of margin
                loss_threshold = margin * (1 - maintenance_margin/100)
                
                if pnl < -loss_threshold:
                    # Liquidated!
                    pnl = -margin  # Lose entire margin
                    reason = "LIQUIDATION"
                    self.state.stats["liquidations"] = self.state.stats.get("liquidations", 0) + 1
                    logger.warning(f"⚠️ LIQUIDATION: {trade_id} lost ${margin:.2f}")

                # BALANCE VALIDATION: Ensure balance never goes negative
                new_balance = self.state.balance_usd + margin + pnl
                if new_balance < 0:
                    # Cap the loss to keep balance at 0
                    pnl = -self.state.balance_usd - margin
                    new_balance = 0
                    reason = "BALANCE_PROTECTION"
                    logger.error(f"❌ BALANCE PROTECTION: {trade_id} capped loss to keep balance at $0")

                # Update trade
                trade_data["status"] = "closed"
                trade_data["exit_time"] = datetime.now().isoformat()
                trade_data["exit_price"] = exit_price
                trade_data["pnl"] = pnl
                trade_data["pnl_pct"] = pnl_pct * leverage * 100  # Show leverage-adjusted %
                trade_data["leverage"] = leverage
                trade_data["exit_fee"] = exit_fee
                trade_data["reason"] = reason

                # Update balance: return margin + PnL (minus fees)
                self.state.balance_usd = new_balance  # Safe balance update
                self.state.margin_used -= margin
                self.state.stats["total_trades"] += 1
                self.state.stats["total_pnl"] += pnl
                self.state.stats["total_fees"] = self.state.stats.get("total_fees", 0) + entry_fee + exit_fee

                if pnl > 0:
                    self.state.stats["winning_trades"] += 1
                    self.state.stats["current_streak"] += 1
                    if self.state.stats["current_streak"] > self.state.stats["best_streak"]:
                        self.state.stats["best_streak"] = self.state.stats["current_streak"]
                else:
                    self.state.stats["losing_trades"] += 1
                    self.state.stats["current_streak"] = 0
                    if abs(self.state.stats["current_streak"]) > self.state.stats["worst_streak"]:
                        self.state.stats["worst_streak"] = abs(self.state.stats["current_streak"])

                # Update win rate
                if self.state.stats["total_trades"] > 0:
                    self.state.stats["win_rate"] = (
                        self.state.stats["winning_trades"] /
                        self.state.stats["total_trades"] * 100
                    )

                self._save_state()

                print(f"\n📝 PAPER TRADE CLOSED")
                print(f"   ID: {trade_id}")
                print(f"   P&L: ${pnl:.2f} ({pnl_pct*100:.2f}%)")
                print(f"   New Balance: ${self.state.balance_usd:,.2f}")

                return trade_data

        return None

    def get_open_trades(self) -> List[Dict]:
        """Get list of open trades."""
        return [t for t in self.state.trades if t["status"] == "open"]

    def get_recent_trades(self, count: int = 10) -> List[Dict]:
        """Get recent closed trades."""
        closed = [t for t in self.state.trades if t["status"] == "closed"]
        return sorted(closed, key=lambda x: x.get("exit_time", ""), reverse=True)[:count]


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Engine")
    parser.add_argument("--start", action="store_true", help="Start paper trading")
    parser.add_argument("--stop", action="store_true", help="Stop paper trading")
    parser.add_argument("--reset", action="store_true", help="Reset paper trading state")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--trades", action="store_true", help="Show recent trades")

    args = parser.parse_args()

    engine = PaperTradingEngine()

    if args.start:
        engine.start()
    elif args.stop:
        engine.stop()
    elif args.reset:
        engine.reset()
    elif args.status:
        engine.status()
    elif args.trades:
        print(f"\n📋 OPEN TRADES: {len(engine.get_open_trades())}")
        for trade in engine.get_open_trades():
            print(f"   {trade['symbol']} {trade['direction']} @ ${trade['entry_price']:.2f}")
        print(f"\n📋 RECENT CLOSED: {len(engine.get_recent_trades())}")
        for trade in engine.get_recent_trades():
            print(f"   {trade['symbol']} {trade['direction']} {trade['pnl_pct']:.2f}%")
    else:
        engine.status()


if __name__ == "__main__":
    main()
