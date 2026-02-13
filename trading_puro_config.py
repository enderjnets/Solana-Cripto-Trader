# TRADING PURO STRATEGY v1.0
# Objective: Grow $500 through active trading

# ============================================
# 🎯 PRIMARY OBJECTIVE: +5% DAILY (171%/month compounded)
# 🎯 SECONDARY OBJECTIVE: DOUBLE ACCOUNT EVERY MONTH (+100%/month)
# ============================================

INITIAL_CAPITAL = 500  # USD
DAILY_TARGET = 0.05     # +5% daily target
MONTHLY_TARGET = 1.00  # +100% monthly target (double)

# Risk Parameters
RISK_PER_TRADE = 0.05  # 5% per trade ($25 max)
STOP_LOSS = 0.08        # -8% per trade
TAKE_PROFIT = 0.10     # +10% per trade (2:1 ratio)
DAILY_LOSS_LIMIT = -0.10  # Max 10% daily loss

# Trading Frequency
MIN_TRADES_PER_DAY = 5  # Minimum to hit +5% daily
MAX_TRADES_PER_DAY = 10
MIN_DAILY_PNL = 0.02    # 2% minimum daily PnL
WARNING_DAILY_PNL = 0.02  # Alert if below 2%

# Reinvestment Rules
REINVEST_RATE = 0.70    # 70% of profits reinvested
RESERVE_RATE = 0.30     # 30% accumulated as USDT reserve

# Trading Pairs Priority
PRIORITY_PAIRS = {
    "SOL-USDC":  {"weight": 0.30, "risk": "low"},
    "cbBTC-USDC": {"weight": 0.25, "risk": "low"},
    "JUP-SOL":   {"weight": 0.15, "risk": "medium"},
    "RAY-SOL":   {"weight": 0.10, "risk": "medium"},
    "BONK-USDC": {"weight": 0.10, "risk": "high"},
    "WIF-SOL":   {"weight": 0.10, "risk": "high"},
}

# USDT Reserve Target
USDT_TARGET = 0.30      # Keep 30% in USDT for dips
USDT_BUY_TRIGGER = -0.15  # Buy when market dips > 15%
