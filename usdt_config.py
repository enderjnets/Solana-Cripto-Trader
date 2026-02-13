# USDT Configuration for Solana Trading Bot
# Strategy: Accumulate BTC + SOL, hold USDT for dips

USDT_TOKEN_ADDRESS = "Es9vMFrzaCERmkhfr9WMq8i5icD4Qwpq6xS5VUUSbmE1"  # USDT on Solana
SOL_TOKEN_ADDRESS = "So11111111111111111111111111111111111111111112"

PORTFOLIO_ALLOCATION = {
    "BTC": {"target": 0.40, "min": 0.30, "max": 0.60},  # 40% target
    "SOL": {"target": 0.40, "min": 0.30, "max": 0.60},  # 40% target  
    "USDT": {"target": 0.20, "min": 0.10, "max": 0.40},  # 20% reserve
}

TRADING_RULES = {
    "rebalance_threshold": 0.05,  # Rebalance when allocation drifts 5%
    "usdt_buy_trigger": -0.10,     # Buy more USDT when portfolio down 10%
    "dip_buy_threshold": -0.15,    # Buy BTC/SOL when market down 15%
    "max_slippage": 0.02,          # 2% max slippage
    "min_trade_size": 0.001,        # Min 0.001 SOL equivalent
}

USDT_STRATEGY = {
    "name": "Stable Reserve",
    "description": "Hold 20% in USDT for market opportunities",
    "buy_on_dip": True,
    "sell_high_target": 0.15,       # Take profits at +15%
    "stop_loss": -0.10,            # Stop loss at -10%
}
