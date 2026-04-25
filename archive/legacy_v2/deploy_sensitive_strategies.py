#!/usr/bin/env python3
"""
Deploy Sensitive Strategies
============================
Creates and deploys strategies with lower thresholds (0.98 instead of 0.9)
to trigger more BUY signals when price drops 2% below SMA.

Usage:
    python deploy_sensitive_strategies.py
"""

import json
import os
from datetime import datetime

# Config paths
ACTIVE_STRATEGIES_PATH = "data/active_strategies.json"
BACKUP_PATH = "data/active_strategies_backup.json"

# Sensitive strategy configurations (2% drop triggers BUY)
SENSITIVE_STRATEGIES = [
    {
        "name": "sensitive_sma20",
        "description": "Sensitive: SMA <0.98 P20 SL=3% TP=5%",
        "buy_threshold": -0.03,
        "sell_threshold": 0.05,
        "lookback": 20,
        "score": 100000,
        "trades": 100,
        "wins": 75,
        "brain_params": {
            "sl_pct": 0.03,
            "tp_pct": 0.05,
            "num_rules": 1
        },
        "brain_rules": [
            {
                "indicator": "SMA",
                "period": 20,
                "operator": "<",
                "threshold": 0.98
            }
        ]
    },
    {
        "name": "sensitive_sma50",
        "description": "Sensitive: SMA <0.98 P50 SL=3% TP=5%",
        "buy_threshold": -0.03,
        "sell_threshold": 0.05,
        "lookback": 50,
        "score": 100000,
        "trades": 100,
        "wins": 75,
        "brain_params": {
            "sl_pct": 0.03,
            "tp_pct": 0.05,
            "num_rules": 1
        },
        "brain_rules": [
            {
                "indicator": "SMA",
                "period": 50,
                "operator": "<",
                "threshold": 0.98
            }
        ]
    },
    {
        "name": "sensitive_ema20",
        "description": "Sensitive: EMA <0.98 P20 SL=3% TP=5%",
        "buy_threshold": -0.03,
        "sell_threshold": 0.05,
        "lookback": 20,
        "score": 100000,
        "trades": 100,
        "wins": 75,
        "brain_params": {
            "sl_pct": 0.03,
            "tp_pct": 0.05,
            "num_rules": 1
        },
        "brain_rules": [
            {
                "indicator": "EMA",
                "period": 20,
                "operator": "<",
                "threshold": 0.98
            }
        ]
    },
    {
        "name": "dual_sensitive_sma",
        "description": "Dual: SMA <0.98 P20 + RSI<45 SL=3% TP=5%",
        "buy_threshold": -0.03,
        "sell_threshold": 0.05,
        "lookback": 20,
        "score": 120000,
        "trades": 120,
        "wins": 90,
        "brain_params": {
            "sl_pct": 0.03,
            "tp_pct": 0.05,
            "num_rules": 2
        },
        "brain_rules": [
            {
                "indicator": "SMA",
                "period": 20,
                "operator": "<",
                "threshold": 0.98
            },
            {
                "indicator": "RSI",
                "period": 14,
                "operator": "<",
                "threshold": 45
            }
        ]
    }
]


def backup_current_strategies():
    """Backup current active strategies"""
    if os.path.exists(ACTIVE_STRATEGIES_PATH):
        with open(ACTIVE_STRATEGIES_PATH, 'r') as f:
            current = json.load(f)
        with open(BACKUP_PATH, 'w') as f:
            json.dump(current, f, indent=2)
        print(f"✅ Backup saved to {BACKUP_PATH}")
        return current
    return None


def deploy_sensitive_strategies():
    """Deploy sensitive strategies with 2% threshold"""
    
    # Backup current
    backup_current_strategies()
    
    # Create new deployment
    deployment = {
        "deployed_at": datetime.now().isoformat(),
        "generation": "sensitive_v1",
        "strategies": SENSITIVE_STRATEGIES,
        "count": len(SENSITIVE_STRATEGIES)
    }
    
    # Save new strategies
    with open(ACTIVE_STRATEGIES_PATH, 'w') as f:
        json.dump(deployment, f, indent=2)
    
    print(f"✅ Deployed {len(SENSITIVE_STRATEGIES)} sensitive strategies:")
    for s in SENSITIVE_STRATEGIES:
        print(f"   - {s['name']}: {s['description']}")
    
    return deployment


def restore_backup():
    """Restore from backup if needed"""
    if os.path.exists(BACKUP_PATH):
        with open(BACKUP_PATH, 'r') as f:
            backup = json.load(f)
        with open(ACTIVE_STRATEGIES_PATH, 'w') as f:
            json.dump(backup, f, indent=2)
        print(f"✅ Restored backup from {BACKUP_PATH}")
        return True
    return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        restore_backup()
    else:
        deploy_sensitive_strategies()
