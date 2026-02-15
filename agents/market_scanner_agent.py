#!/usr/bin/env python3
"""
Market Scanner Agent
====================
Scans markets for trading opportunities.

Features:
- Volume anomaly detection
- Trend identification
- Pattern recognition
- Token screening
- Opportunity ranking

Usage:
    agent = MarketScanner()
    opportunities = agent.scan_market()
    opportunities = agent.scan_solana_tokens()
"""

import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.api_integrations import JupiterClient

logger = logging.getLogger("market_scanner")

# ============================================================================
# SCANNER CONFIG
# ============================================================================
@dataclass
class ScannerConfig:
    """Market scanner configuration"""
    min_volume_multiplier: float = 2.0  # 2x average volume
    min_trend_strength: float = 0.6  # 60% trend strength
    min_liquidity: float = 10000  # $10k minimum liquidity
    scan_interval_seconds: int = 60
    top_tokens_count: int = 20


@dataclass
class Opportunity:
    """Trading opportunity"""
    symbol: str
    name: str
    price: float
    change_24h: float
    volume_24h: float
    liquidity: float
    trend: str  # bullish, bearish, neutral
    volume_anomaly: float  # 2.0 = 2x normal
    score: float  # 0-100 opportunity score
    reasons: List[str]
    timestamp: str


# ============================================================================
# MARKET SCANNER AGENT
# ============================================================================
class MarketScannerAgent:
    """
    Market scanning agent for finding trading opportunities.
    
    Scans Solana tokens for:
    - Volume anomalies
    - Trend strength
    - Liquidity levels
    - Price patterns
    """
    
    def __init__(self, config: Optional[ScannerConfig] = None):
        self.config = config or ScannerConfig()
        self.jupiter = JupiterClient()
        self.state_file = PROJECT_ROOT / "data" / "scanner_state.json"
        self.opportunities = []
        self._load_state()
    
    def _load_state(self):
        """Load scanner state from file"""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.opportunities = data.get("opportunities", [])
            except:
                self.opportunities = []
        else:
            self.opportunities = []
    
    def _save_state(self):
        """Save scanner state to file"""
        data = {
            "opportunities": self.opportunities,
            "last_scan": datetime.now().isoformat()
        }
        self.state_file.parent.mkdir(exist_ok=True)
        self.state_file.write_text(json.dumps(data, indent=2))
    
    # =========================================================================
    # SCANNING FUNCTIONS
    # =========================================================================
    def scan_market(self) -> List[Opportunity]:
        """
        Scan entire market for opportunities.
        
        Returns:
            List of ranked trading opportunities
        """
        logger.info("🔍 Scanning market for opportunities...")
        
        opportunities = []
        
        # Get top tokens by volume
        tokens = self._get_top_tokens()
        
        for token in tokens:
            try:
                opp = self._analyze_token(token)
                if opp:
                    opportunities.append(opp)
            except Exception as e:
                logger.debug(f"Error analyzing {token}: {e}")
        
        # Sort by score
        opportunities.sort(key=lambda x: x.score, reverse=True)
        
        # Store top opportunities
        self.opportunities = opportunities[:10]
        self._save_state()
        
        logger.info(f"✅ Found {len(opportunities)} opportunities")
        
        return opportunities
    
    def scan_solana_tokens(self) -> List[Opportunity]:
        """Alias for scan_market()"""
        return self.scan_market()
    
    def _get_top_tokens(self) -> List[Dict]:
        """Get list of tokens to scan"""
        # Default SOL tokens to scan
        default_tokens = [
            {"symbol": "SOL", "name": "Solana"},
            {"symbol": "BTC", "name": "Bitcoin"},
            {"symbol": "ETH", "name": "Ethereum"},
            {"symbol": "USDC", "name": "USD Coin"},
            {"symbol": "USDT", "name": "Tether"},
        ]
        
        # Try to get more from Jupiter
        try:
            tokens = self.jupiter.get_tokens()
            if tokens and len(tokens) > 0:
                return tokens[:self.config.top_tokens_count]
        except:
            pass
        
        return default_tokens
    
    def _analyze_token(self, token: Dict) -> Optional[Opportunity]:
        """
        Analyze a single token for opportunities.
        
        Args:
            token: Dict with 'symbol', 'name'
            
        Returns:
            Opportunity if found, None otherwise
        """
        symbol = token.get("symbol", "UNKNOWN")
        name = token.get("name", symbol)
        
        try:
            # Get quote data
            quote = self.jupiter.get_quote(symbol, "USDC")
            
            if not quote:
                return None
            
            price = quote.get("price", 0)
            change_24h = quote.get("change_24h", 0)
            volume_24h = quote.get("volume_24h", 0)
            liquidity = quote.get("liquidity", 0)
            
            # Skip if low liquidity
            if liquidity < self.config.min_liquidity:
                return None
            
            # Analyze
            reasons = []
            score = 50  # Base score
            
            # Volume anomaly
            volume_multiplier = quote.get("volume_multiplier", 1.0)
            if volume_multiplier >= self.config.min_volume_multiplier:
                reasons.append(f"High volume: {volume_multiplier:.1f}x normal")
                score += 20
            
            # Trend strength
            trend = quote.get("trend", "neutral")
            trend_strength = quote.get("trend_strength", 0.5)
            
            if trend == "bullish" and trend_strength >= self.config.min_trend_strength:
                reasons.append(f"Bullish trend: {trend_strength:.0%} strength")
                score += 15
            elif trend == "bearish" and trend_strength >= self.config.min_trend_strength:
                reasons.append(f"Bearish trend: {trend_strength:.0%} strength")
                score += 10
            
            # Price change
            if change_24h > 5:
                reasons.append(f"Strong rally: +{change_24h:.1f}% 24h")
                score += 10
            elif change_24h < -5:
                reasons.append(f"Drop detected: {change_24h:.1f}% 24h")
                score += 5
            
            # Cap score at 100
            score = min(score, 100)
            
            # Create opportunity
            return Opportunity(
                symbol=symbol,
                name=name,
                price=price,
                change_24h=change_24h,
                volume_24h=volume_24h,
                liquidity=liquidity,
                trend=trend,
                volume_anomaly=volume_multiplier,
                score=score,
                reasons=reasons,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.debug(f"Error analyzing {symbol}: {e}")
            return None
    
    # =========================================================================
    # PATTERN DETECTION
    # =========================================================================
    def detect_patterns(self, symbol: str) -> List[Dict]:
        """
        Detect chart patterns for a symbol.
        
        Args:
            symbol: Token symbol
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        try:
            # Get price data
            data = self._get_price_data(symbol)
            
            if not data or len(data) < 10:
                return patterns
            
            # Detect patterns
            patterns.extend(self._detect_squeeze(data))
            patterns.extend(self._detect_breakout(data))
            patterns.extend(self._detect_reversal(data))
            
        except Exception as e:
            logger.debug(f"Pattern detection error for {symbol}: {e}")
        
        return patterns
    
    def _get_price_data(self, symbol: str) -> List[Dict]:
        """Get price data for pattern detection"""
        try:
            # This would integrate with price feed
            return []
        except:
            return []
    
    def _detect_squeeze(self, data: List[Dict]) -> List[Dict]:
        """Detect Bollinger Band squeeze"""
        patterns = []
        # Pattern detection logic
        return patterns
    
    def _detect_breakout(self, data: List[Dict]) -> List[Dict]:
        """Detect price breakout patterns"""
        patterns = []
        # Pattern detection logic
        return patterns
    
    def _detect_reversal(self, data: List[Dict]) -> List[Dict]:
        """Detect price reversal patterns"""
        patterns = []
        # Pattern detection logic
        return patterns
    
    # =========================================================================
    # OPPORTUNITY MANAGEMENT
    # =========================================================================
    def get_top_opportunities(self, limit: int = 5) -> List[Opportunity]:
        """Get top ranked opportunities"""
        return self.opportunities[:limit]
    
    def get_opportunity_by_symbol(self, symbol: str) -> Optional[Opportunity]:
        """Find opportunity by symbol"""
        for opp in self.opportunities:
            if opp.symbol == symbol:
                return opp
        return None
    
    def clear_opportunities(self):
        """Clear stored opportunities"""
        self.opportunities = []
        self._save_state()
    
    # =========================================================================
    # STATUS
    # =========================================================================
    def status(self) -> Dict:
        """Get agent status"""
        return {
            "agent": "market_scanner",
            "status": "active",
            "config": {
                "min_volume_multiplier": self.config.min_volume_multiplier,
                "min_trend_strength": self.config.min_trend_strength,
                "min_liquidity": self.config.min_liquidity,
                "scan_interval": self.config.scan_interval_seconds
            },
            "opportunities_count": len(self.opportunities),
            "last_scan": self.state_file.exists() and datetime.fromisoformat(
                json.loads(self.state_file.read_text()).get("last_scan", "")
            ) or None
        }


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Market Scanner Agent CLI")
    parser.add_argument("--status", action="store_true", help="Show agent status")
    parser.add_argument("--scan", action="store_true", help="Run market scan")
    parser.add_argument("--top", type=int, default=5, help="Show top N opportunities")
    parser.add_argument("--clear", action="store_true", help="Clear opportunities")
    
    args = parser.parse_args()
    
    agent = MarketScannerAgent()
    
    if args.status:
        print(json.dumps(agent.status(), indent=2))
    elif args.scan:
        opportunities = agent.scan_market()
        print(f"\n📊 Found {len(opportunities)} opportunities:")
        for i, opp in enumerate(opportunities[:args.top], 1):
            print(f"\n{i}. {opp.symbol} ({opp.name})")
            print(f"   Price: ${opp.price:.4f}")
            print(f"   24h Change: {opp.change_24h:+.1f}%")
            print(f"   Trend: {opp.trend}")
            print(f"   Score: {opp.score}/100")
            print(f"   Reasons: {', '.join(opp.reasons)}")
    elif args.clear:
        agent.clear_opportunities()
        print("✅ Opportunities cleared")
    else:
        print("Market Scanner Agent - Market Opportunity Detection")
        print("Usage:")
        print("  python3 market_scanner_agent.py --status   # Show status")
        print("  python3 market_scanner_agent.py --scan     # Run market scan")
        print("  python3 market_scanner_agent.py --top 10   # Show top 10 opportunities")
        print("  python3 market_scanner_agent.py --clear    # Clear opportunities")
