# TRADING PURO STRATEGY v1.0
# Objective: Grow $500 through active trading

INITIAL_CAPITAL = 500  # USD
RISK_PER_TRADE = 0.05  # 5% per trade ($25 max)
STOP_LOSS = -0.10      # -10% per trade
TAKE_PROFIT = 0.20     # +20% per trade
DAILY_RISK_LIMIT = 0.15  # Max 15% daily loss
MONTHLY_GOAL = 0.50    # 50% monthly growth target

# Reinvestment Rules
REINVEST_RATE = 0.70    # 70% of profits reinvested
RESERVE_RATE = 0.30     # 30% accumulated as USDT reserve

# Trading Pairs Priority
PRIORITY_PAIRS = {
    "SOL-USDC":  {"weight": 0.30, "risk": "low"},
    "cbBTC-USDC": {"weight": 0.25, "risk": "low"},
    "JUP-SOL":   {"weight": 0.15, "risk": "medium"},
    "RAY-SOL":   {"weight": 0.10, "risk": "medium"},
    "BONK-USDC": {"weight": 0.05, "risk": "high"},
    "WIF-SOL":   {"weight": 0.05, "risk": "high"},
}

# USDT Reserve Target (for opportunities)
USDT_TARGET = 0.30  # Keep 30% in USDT for dips
USDT_BUY_TRIGGER = -0.15  # Buy when market dips > 15%
