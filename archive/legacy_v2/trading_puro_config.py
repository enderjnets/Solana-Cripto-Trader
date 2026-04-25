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

# ============================================
# 🎯 AUTO REBALANCING CONFIG
# ============================================
REBALANCE_ENABLED = True
REBALANCE_THRESHOLD = 0.05   # 5% drift from target triggers rebalance
REBALANCE_CONFIDENCE_AUTO = 0.80  # 80%+ = execute immediately
REBALANCE_CONFIDENCE_ALERT = 0.60  # 60-80% = alert user first

# Target Allocation (STUDY RESULT: 50/30/20 is optimal for +5% daily)
TARGET_ALLOCATION = {
    "SOL": 0.50,   # 50% - Más oportunidades de trading
    "BTC": 0.30,   # 30% - Estabilidad  
    "USDT": 0.20   # 20% - Reserve y dips
}

# Adaptive Allocation Rules
ADAPTIVE_ALLOCATION = {
    "BULL_MARKET": {"SOL": 0.55, "BTC": 0.25, "USDT": 0.20},
    "BEAR_MARKET": {"SOL": 0.30, "BTC": 0.50, "USDT": 0.20},
    "LATERAL": {"SOL": 0.50, "BTC": 0.30, "USDT": 0.20},  # DEFAULT
    "HIGH_VOL": {"SOL": 0.60, "BTC": 0.20, "USDT": 0.20}
}
