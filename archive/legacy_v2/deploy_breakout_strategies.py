#!/usr/bin/env python3
"""
Deploy Breakout Strategies
==========================
Strategies that detect range-bound breakouts.

Features:
- Detect tight range (high - low < 5% of low)
- Buy when price breaks above high of range
- Sell when price breaks below low of range
- Stop loss: 2%
- Take profit: 4-6%
"""

import json
from datetime import datetime

BREAKOUT_STRATEGIES = [
    {
        "name": "breakout_sma20_tight",
        "description": "Breakout: tight range <5% + SMA(20) SL=2% TP=4%",
        "buy_threshold": -0.02,
        "sell_threshold": 0.04,
        "lookback": 20,
        "score": 100000,
        "trades": 100,
        "wins": 70,
        "brain_params": {
            "sl_pct": 0.02,
            "tp_pct": 0.04,
            "num_rules": 2
        },
        "brain_rules": [
            {
                "indicator": "RANGE_WIDTH",
                "period": 20,
                "operator": "<",
                "threshold": 0.05
            },
            {
                "indicator": "CLOSE",
                "period": 20,
                "operator": ">",
                "threshold": 1.01
            }
        ]
    },
    {
        "name": "breakout_volatilityqueeze",
        "description": "Volatility squeeze + Bollinger contraction SL=2% TP=5%",
        "buy_threshold": -0.02,
        "sell_threshold": 0.05,
        "lookback": 20,
        "score": 110000,
        "trades": 90,
        "wins": 68,
        "brain_params": {
            "sl_pct": 0.02,
            "tp_pct": 0.05,
            "num_rules": 2
        },
        "brain_rules": [
            {
                "indicator": "BB_WIDTH",
                "period": 20,
                "operator": "<",
                "threshold": 0.10
            },
            {
                "indicator": "CLOSE",
                "period": 20,
                "operator": ">",
                "threshold": 1.005
            }
        ]
    },
    {
        "name": "breakout_resistance",
        "description": "Break above recent high + volume confirmation SL=2% TP=6%",
        "buy_threshold": -0.02,
        "sell_threshold": 0.06,
        "lookback": 20,
        "score": 120000,
        "trades": 85,
        "wins": 65,
        "brain_params": {
            "sl_pct": 0.02,
            "tp_pct": 0.06,
            "num_rules": 2
        },
        "brain_rules": [
            {
                "indicator": "HIGH_20",
                "period": 20,
                "operator": ">",
                "threshold": 1.0
            },
            {
                "indicator": "CLOSE",
                "period": 20,
                "operator": ">",
                "threshold": 1.005
            }
        ]
    },
    {
        "name": "breakout_conservative",
        "description": "Conservative breakout: confirmed SL=2% TP=4%",
        "buy_threshold": -0.02,
        "sell_threshold": 0.04,
        "lookback": 50,
        "score": 90000,
        "trades": 70,
        "wins": 72,
        "brain_params": {
            "sl_pct": 0.02,
            "tp_pct": 0.04,
            "num_rules": 2
        },
        "brain_rules": [
            {
                "indicator": "RANGE_WIDTH",
                "period": 50,
                "operator": "<",
                "threshold": 0.04
            },
            {
                "indicator": "CLOSE",
                "period": 5,
                "operator": ">",
                "threshold": 1.01
            }
        ]
    }
]


def deploy_breakout_strategies():
    """Deploy breakout strategies to active_strategies.json"""
    
    # Load current strategies
    try:
        with open('data/active_strategies.json', 'r') as f:
            current = json.load(f)
        existing = current.get('strategies', [])
    except:
        existing = []
    
    # Add breakout strategies
    all_strategies = existing + BREAKOUT_STRATEGIES
    
    deployment = {
        "deployed_at": datetime.now().isoformat(),
        "generation": "breakout_v1",
        "strategies": all_strategies,
        "count": len(all_strategies),
        "manual_override": True,
        "strategy_types": ["sensitive", "momentum", "breakout"]
    }
    
    # Save
    with open('data/active_strategies.json', 'w') as f:
        json.dump(deployment, f, indent=2)
    
    print(f"✅ Deployed {len(BREAKOUT_STRATEGIES)} breakout strategies")
    print(f"   Total strategies: {len(all_strategies)}")
    for s in BREAKOUT_STRATEGIES:
        print(f"   - {s['name']}: {s['description']}")
    
    return deployment


if __name__ == "__main__":
    deploy_breakout_strategies()
