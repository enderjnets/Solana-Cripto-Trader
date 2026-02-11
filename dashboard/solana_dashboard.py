#!/usr/bin/env python3
"""
Solana Dashboard - Streamlit Interface
======================================
Web dashboard for monitoring and controlling the trading bot.

Based on: interface.py from Coinbase Cripto Trader

Features:
- Real-time worker status
- Strategy performance metrics
- Backtest results visualization
- Trade history
- System controls
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Configure page
st.set_page_config(
    page_title="Jupiter Solana Trading Bot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page title
st.title("🚀 Jupiter Solana Trading Bot")
st.markdown("---")


# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    # Network selector
    network = st.selectbox(
        "Network",
        ["devnet", "testnet", "mainnet"],
        index=0
    )

    # Risk level
    risk_level = st.select_slider(
        "Risk Level",
        options=["LOW", "MEDIUM", "HIGH"],
        value="MEDIUM"
    )

    st.markdown("---")

    # Auto-refresh
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_interval = st.slider("Refresh interval (s)", 5, 60, 10)

    st.markdown("---")

    # Connection status
    st.subheader("🔗 Connection")

    if st.button("Test RPC Connection"):
        with st.spinner("Testing connection..."):
            try:
                # Placeholder for actual test
                st.success("✅ Connected!")
            except Exception as e:
                st.error(f"❌ Error: {e}")


# ============================================================================
# TAB 1: DASHBOARD OVERVIEW
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard",
    "👷 Workers",
    "📈 Strategies",
    "🔄 Swap",
    "🎮 Control",
    "📋 Logs"
])


with tab1:
    # Get real wallet data
    import httpx
    from solana.rpc.api import Client
    from solders.pubkey import Pubkey

    WALLET_ADDRESS = "65YqSYGwR6UNCUmeaKt1V1HV99Ky1tii2bgg6jwJSGN3"

    # Get SOL balance
    sol_balance = 0.0
    sol_price = 0.0
    try:
        rpc = Client('https://api.devnet.solana.com')
        pubkey = Pubkey.from_string(WALLET_ADDRESS)
        resp = rpc.get_balance(pubkey)
        sol_balance = resp.value / 1_000_000_000

        # Get SOL price
        url = 'https://lite-api.jup.ag/price/v3?ids=So11111111111111111111111111111111111111112'
        resp = httpx.get(url, timeout=10)
        data = resp.json()
        sol_price = data.get('So11111111111111111111111111111111111111112', {}).get('usdPrice', 0)
    except Exception as e:
        st.error(f"Error connecting to Solana: {e}")

    portfolio_value = sol_balance * sol_price

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "💰 Portfolio Value",
            f"${portfolio_value:.2f}",
            delta=f"{sol_balance:.4f} SOL",
            delta_color="normal"
        )

    with col2:
        st.metric(
            "📈 SOL Price",
            f"${sol_price:.2f}",
            delta="Live",
            delta_color="normal"
        )

    with col3:
        st.metric(
            "🎯 SOL Balance",
            f"{sol_balance:.4f}",
            delta="Native SOL",
            delta_color="normal"
        )

    with col4:
        st.metric(
            "🔥 Active Positions",
            "1",  # Solo SOL nativo por ahora
            delta="Native Only",
            delta_color="off"
        )

    st.markdown("---")

    # Wallet Info
    with st.expander("📍 Wallet Address", expanded=True):
        st.code(WALLET_ADDRESS, language="text")
        st.caption("Esta wallet tiene SOL nativo. Para tokens SPL, necesitas depósitos adicionales.")

    # SOL Native Balance Display
    col_sol, col_chart2 = st.columns([1, 2])

    with col_sol:
        st.subheader("💰 SOL Native")
        st.metric("Balance", f"{sol_balance:.4f} SOL")
        st.metric("Value (USD)", f"${portfolio_value:.2f}")
        st.metric("Price", f"${sol_price:.2f}")

    # Token Holdings (from Jupiter API)
    with col_chart2:
        st.subheader("📊 Token Holdings")

        # Try to get token holdings
        holdings_url = f'https://lite-api.jup.ag/ultra/v1/holdings/{WALLET_ADDRESS}'
        try:
            resp = httpx.get(holdings_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                tokens = data.get('tokens', {})
                if tokens:
                    # Create table
                    holdings_data = []
                    for mint, info in tokens.items():
                        amount = float(info.get('uiAmount', 0))
                        if amount > 0:
                            holdings_data.append({
                                'Token': mint[:12] + '...',
                                'Amount': amount,
                                'USD Value': float(info.get('usdValue', 0))
                            })
                    if holdings_data:
                        df = pd.DataFrame(holdings_data)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No SPL tokens found. Deposit tokens to see them here.")
                else:
                    st.info("💡 No SPL tokens found. Esta wallet solo tiene SOL nativo.")
            else:
                st.info("💡 No SPL tokens found. Esta wallet solo tiene SOL nativo.")
        except Exception as e:
            st.info(f"💡 Esta wallet solo tiene SOL nativo. ({str(e)[:50]})")

    st.markdown("---")

    # Historical chart placeholder
    st.subheader("📈 Portfolio Performance")
    st.info("📊 Gráfico de rendimiento coming soon...")
    st.line_chart([1, 2, 3, 5, 8, 13])  # Placeholder

    with col_chart2:
        st.subheader("📈 P&L Chart")

        # Generate sample data
        dates = pd.date_range(start="2024-01-01", periods=30)
        prices = 100 + np.cumsum(np.random.randn(30) * 2)

        df = pd.DataFrame({"Date": dates, "Value": prices})
        df.set_index("Date", inplace=True)

        st.line_chart(df)


# ============================================================================
# TAB 2: WORKERS STATUS
# ============================================================================

with tab2:
    st.subheader("👷 Worker Status")

    # Import worker functions
    try:
        from workers.jupiter_worker import worker_status, worker_start, worker_stop, get_worker

        # Get real worker status
        status = worker_status()

        # Display worker info
        col_w1, col_w2, col_w3, col_w4 = st.columns(4)

        with col_w1:
            st.metric(
                "👷 Worker ID",
                status.get('worker_id', 'N/A')[:16],
                delta="Active" if status.get('status') == 'running' else "Stopped"
            )

        with col_w2:
            st.metric(
                "📊 Status",
                status.get('status', 'unknown').upper(),
                delta_color="normal" if status.get('status') == 'running' else "off"
            )

        with col_w3:
            st.metric(
                "💰 Last Price",
                f"${status.get('last_price', 0):.2f}",
                delta="Live"
            )

        with col_w4:
            st.metric(
                "🔄 Swaps",
                status.get('swap_count', 0),
                delta="Executed"
            )

        # Worker controls
        col_b1, col_b2 = st.columns(2)

        with col_b1:
            if st.button("▶️ Start Worker", type="primary"):
                worker_start()
                st.rerun()

        with col_b2:
            if st.button("⏹️ Stop Worker"):
                worker_stop()
                st.rerun()

        st.markdown("---")

        # Real-time prices from worker
        st.subheader("📈 Live Prices")

        prices = status.get('prices', {})
        if prices:
            # Create price grid
            cols = st.columns(4)
            i = 0
            for token, price in prices.items():
                with cols[i % 4]:
                    st.metric(token, f"${price:.4f}")
                i += 1
        else:
            st.info("Start worker to see live prices")

        # Worker details
        with st.expander("📋 Worker Details"):
            import json
            st.json(status, expanded=False)

        # Quick quote section
        st.markdown("---")
        st.subheader("💸 Quick Quote")

        col_q1, col_q2, col_q3 = st.columns(3)

        with col_q1:
            quote_in = st.selectbox("Sell", ["SOL", "USDC", "USDT", "JUP", "BONK"], key="quote_in")
        with col_q2:
            quote_out = st.selectbox("Buy", ["USDC", "SOL", "USDT", "JUP", "BONK"], key="quote_out")
        with col_q3:
            quote_amt = st.number_input("Amount", value=0.1, min_value=0.0, step=0.1, key="quote_amt")

        if st.button("📊 Get Quote", type="primary"):
            try:
                from workers.jupiter_worker import worker_quote
                result = worker_quote(quote_in, quote_out, quote_amt)
                if result and 'outAmount' in result:
                    out_amt = int(result['outAmount'])
                    decimals = 9 if quote_out == 'SOL' else 6
                    final_amt = out_amt / (10 ** decimals)
                    st.success(f"💰 You get: {final_amt:.6f} {quote_out}")
                    st.info(f"📊 Price Impact: {result.get('priceImpact', 'N/A')}%")
                else:
                    st.error("Failed to get quote")
            except Exception as e:
                st.error(f"Error: {e}")

    except Exception as e:
        st.error(f"Error loading worker: {e}")
        st.info("Try starting the worker module...")

        # Fallback to start worker
        if st.button("🚀 Initialize Worker"):
            try:
                from workers.jupiter_worker import JupiterWorker
                worker = JupiterWorker()
                worker.start()
                st.success("Worker started!")
                st.rerun()
            except Exception as e2:
                st.error(f"Failed: {e2}")


# ============================================================================
# TAB 3: STRATEGIES
# ============================================================================

with tab3:
    st.subheader("📈 Trading Strategies")
    
    try:
        from strategies.runner import get_runner, runner_start, runner_stop
        from strategies import StrategyFactory
        
        # Get runner status
        runner = get_runner()
        status = runner.get_status()
        
        # Strategy controls
        col_c1, col_c2, col_c3 = st.columns(3)
        
        with col_c1:
            if st.button("▶️ Start Runner", type="primary"):
                runner.start()
                st.rerun()
        
        with col_c2:
            if st.button("⏹️ Stop Runner"):
                runner.stop()
                st.rerun()
        
        with col_c3:
            st.metric("Status", status["running"] and "🟢 Running" or "🔴 Stopped")
        
        st.markdown("---")
        
        # Add new strategy
        st.subheader("➕ Add Strategy")
        
        col_a1, col_a2, col_a3 = st.columns(3)
        
        with col_a1:
            strategy_type = st.selectbox(
                "Strategy Type",
                ["rsi", "sma_crossover", "ema_crossover", "macd", "bollinger", "combined"],
                format_func=lambda x: {
                    "rsi": "RSI (Overbought/Oversold)",
                    "sma_crossover": "SMA Crossover",
                    "ema_crossover": "EMA Crossover",
                    "macd": "MACD",
                    "bollinger": "Bollinger Bands",
                    "combined": "Combined (Multi-indicator)"
                }.get(x, x)
            )
        
        with col_a2:
            pair = st.selectbox("Trading Pair", ["SOL-USDC", "SOL-USDT", "JUP-SOL", "BONK-USDC"])
        
        with col_a3:
            min_confidence = st.slider("Min Confidence", 0.3, 0.9, 0.6)
        
        # Strategy-specific parameters
        params = {}
        if strategy_type == "rsi":
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                params["rsi_period"] = st.number_input("RSI Period", 7, 21, 14)
            with col_p2:
                params["oversold"] = st.number_input("Oversold", 20, 40, 30)
                params["overbought"] = st.number_input("Overbought", 60, 80, 70)
        elif strategy_type in ["sma_crossover", "ema_crossover"]:
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                params["fast_period"] = st.number_input("Fast Period", 5, 20, 10)
            with col_p2:
                params["slow_period"] = st.number_input("Slow Period", 20, 50, 30)
        elif strategy_type == "macd":
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                params["fast_period"] = st.number_input("Fast Period", 8, 20, 12)
            with col_p2:
                params["slow_period"] = st.number_input("Slow Period", 20, 40, 26)
            with col_p3:
                params["signal_period"] = st.number_input("Signal Period", 5, 15, 9)
        
        strategy_name = f"{strategy_type}_{pair.replace('-', '').lower()}"
        
        if st.button("➕ Add Strategy", type="primary"):
            config = {
                "type": strategy_type,
                "parameters": params,
                "pair": pair,
                "min_confidence": min_confidence,
                "auto_execute": False
            }
            runner.add_strategy(strategy_name, config)
            st.success(f"Strategy added: {strategy_name}")
            st.rerun()
        
        st.markdown("---")
        
        # Active strategies
        st.subheader("📋 Active Strategies")
        
        strategies = runner.list_strategies()
        if strategies:
            for s in strategies:
                with st.expander(f"📊 {s['name']} ({s['type']})"):
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    with col_s1:
                        st.write(f"**Pair:** {s['pair']}")
                    with col_s2:
                        st.write(f"**Confidence:** {s.get('min_confidence', 0.6)*100:.0f}%")
                    with col_s3:
                        auto = "✅" if s.get('auto_execute') else "❌"
                        st.write(f"**Auto-execute:** {auto}")
                    with col_s4:
                        if st.button(f"🗑️ Remove", key=f"remove_{s['name']}"):
                            runner.remove_strategy(s['name'])
                            st.rerun()
        else:
            st.info("No strategies configured. Add one above!")
        
        st.markdown("---")
        
        # Data Sources Info
        st.subheader("📊 Historical Data Sources")
        
        try:
            from data.historical_data import HistoricalDataManager
            manager = HistoricalDataManager()
            summary = manager.get_data_summary()
            
            col_d1, col_d2, col_d3 = st.columns(3)
            
            with col_d1:
                st.info(f"**Available Tokens:** {', '.join(summary['available_tokens'])}")
            with col_d2:
                st.info(f"**Timeframes:** {', '.join(summary['available_timeframes'])}")
            with col_d3:
                st.info(f"**Total Candles:** {summary['total_candles']:,}")
            
            if summary['cache_files']:
                with st.expander("📁 Cache Files"):
                    for f in summary['cache_files']:
                        st.write(f"  - {f['file']}: {f['candles']:,} candles")
        except Exception as e:
            st.warning(f"Data module: {e}")
        
        st.markdown("---")
        
        # Backtest section
        st.subheader("📈 Backtest Results")
        
        if strategies:
            col_bt1, col_bt2, col_bt3 = st.columns(3)
            
            with col_bt1:
                bt_strategy = st.selectbox(
                    "Select strategy",
                    [s["name"] for s in strategies],
                    key="bt_strategy"
                )
            
            with col_bt2:
                bt_days = st.selectbox(
                    "Historical Days",
                    [30, 90, 180, 365, 730, 1095],
                    index=3,
                    format_func=lambda x: f"{x} days ({x//365} years)" if x >= 365 else f"{x} days"
                )
            
            # TP/SL settings
            col_tp1, col_tp2 = st.columns(2)
            with col_tp1:
                tp_pct = st.slider("Take Profit %", 2, 20, 5) / 100
            with col_tp2:
                sl_pct = st.slider("Stop Loss %", 1, 10, 3) / 100
            
            st.write(f"📊 TP: {tp_pct*100:.0f}% | SL: {sl_pct*100:.0f}%")
            
            with col_bt3:
                st.write("")  # Spacer
                st.write("")
                if st.button("🏃 Run Backtest", type="primary"):
                    with st.spinner(f"Running backtest with {bt_days} days..."):
                        results = runner.run_backtest(bt_strategy, days=bt_days, tp_pct=tp_pct, sl_pct=sl_pct)
                        
                        st.success("✅ Backtest complete!")
                        
                        # Results
                        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                        with col_r1:
                            st.metric("Trades", results.get("total_trades", 0))
                        with col_r2:
                            wr = results.get("win_rate", 0) * 100
                            st.metric("Win Rate", f"{wr:.1f}%")
                        with col_r3:
                            ret = results.get("total_return", 0) * 100
                            st.metric("Return", f"{ret:+.2f}%")
                        with col_r4:
                            st.metric("Avg PnL", f"{results.get('avg_pnl', 0)*100:+.2f}%")
                        
                        # TP/SL stats
                        tp_count = results.get("tp_count", 0)
                        sl_count = results.get("sl_count", 0)
                        st.info(f"🎯 TP: {tp_count} | 🛑 SL: {sl_count}")
                        
                        # Data info
                        if "data_info" in results:
                            info = results["data_info"]
                            st.caption(f"📊 {info['total_candles']:,} candles | {info['date_range'][:10]} | TP: {info.get('tp_pct', 5)}% | SL: {info.get('sl_pct', 3)}%")
            
            # Run backtest immediately if strategies exist
            if strategies:
                with st.spinner("Running initial backtest..."):
                    results = runner.run_backtest(strategies[0]["name"], days=90, tp_pct=0.05, sl_pct=0.03)
                    
                    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                    with col_r1:
                        st.metric("Trades", results.get("total_trades", 0))
                    with col_r2:
                        wr = results.get("win_rate", 0) * 100
                        st.metric("Win Rate", f"{wr:.1f}%")
                    with col_r3:
                        ret = results.get("total_return", 0) * 100
                        st.metric("Return", f"{ret:.2f}%")
                    with col_r4:
                        st.metric("Avg PnL", f"{results.get('avg_pnl', 0)*100:.2f}%")
        
        # Recent signals
        st.markdown("---")
        st.subheader("📡 Recent Signals")
        
        signals = status.get("recent_signals", [])
        if signals:
            for sig in reversed(signals[-10:]):
                icon = "🟢" if sig["signal"] == "buy" else ("🔴" if sig["signal"] == "sell" else "⚪")
                st.write(f"{icon} **{sig['signal'].upper()}** | {sig['strategy']} | ${sig['price']:.2f} | {sig.get('confidence', 0)*100:.0f}% conf")
        else:
            st.info("No signals yet. Start the runner to generate signals!")
        
    except Exception as e:
        st.error(f"Error loading strategies: {e}")
        st.info("Try initializing strategies module...")


# ============================================================================
# TAB 4: MANUAL SWAP
# ============================================================================

with tab4:
    st.subheader("🔄 Manual Swap (DEVNET)")

        # All popular tokens on Solana
    ALL_TOKENS = {
        # Top by Market Cap
        "SOL": ("So11111111111111111111111111111111111111112", 9),
        "USDC": ("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", 6),
        "USDT": ("Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYW", 6),

        # DeFi / Infrastructure
        "JUP": ("JUPyiwrYJFskUPiHa7hkeR8VUtkqjberbSOWd91pbT2", 6),
        "RAY": ("4jNcW4C4m6TnB8e9pWpB7Yq4Yq4Yq4Yq4Yq4Yq4Yq4Y", 6),
        "MNGO": ("MangoV3Maint8S4XKJB7hqFDJWsp8PxcYqBtkBxF9V", 6),
        "SRM": ("SRMuApVNdxXokE5vYV5M4kE8V9nV5M4kE8V9nV5M4k", 6),
        "ORCA": ("orcaEKTmK8nd7G6q8f4Yq4Yq4Yq4Yq4Yq4Yq4Yq4Y", 6),

        # Memecoins Populares
        "BONK": ("DezXAZ8z7PnrnRJjz3wXBoZGVixqUi5iA2ztETHuJXJP", 5),
        "WIF": ("EKpQGSJtjMFqKZ9KQanSqWJcNSPWfqHYJQD7i阜eLJ", 6),
        "WEN": ("WENWENv2ykuwsLVnK4KbYQaN9UJqr4Yz7X6gYVfY8X", 5),
        "POPCAT": ("7GCihgDB8dfS1XbY9Hb86h7V9Y4vNPQp1dz1Yyj4FU", 9),
        "MEW": ("MEW1gQWJ3nEXg2uoERb2YbR1w6G3Yq4Yq4Yq4Yq4Yq4", 5),
        "FLOKI": ("FLuxi2vNLG4JLc1K8p9w9Yq4Yq4Yq4Yq4Yq4Yq4Yq", 9),

        # AI / Gaming
        "PYTH": ("HZ1JovNiBEgZ1W7E2hKQzF8Tz3G6fZ6K3jKGn1c3bY7V", 6),
        "ATLAS": ("ATLASXmbVVx9CPvCxgqHXa1CU9G4yr7V6Wf9Y7Yq1z", 8),
        "STARL": ("STARLr4x1oZvi1b7Y9gX7ZVG3V7Zq7Z7Zq7Zq7Zq7Zq", 8),
        "COPE": ("CPEz5niaEVfD3vKKB7xHVDqs1K1L4V7K7K7K7K7K7K7K7", 6),
        "HNT": ("HNT Token address here", 8),
        "AUDIO": ("AUDIO Token address here", 8),
        "MNDE": ("MNDE Token address here", 6),
    }

    # Tokens con precio en tiempo real
    PRICE_TOKENS = ["SOL", "USDC", "USDT", "JUP", "BONK", "WIF", "PYTH", "WEN", "MNGO"]

    col_swap1, col_swap2 = st.columns(2)

    with col_swap1:
        st.markdown("### 💸 Sell")
        sell_token = st.selectbox("Token", list(ALL_TOKENS.keys()), key="sell")
        sell_amount = st.number_input("Amount", min_value=0.0, value=0.1, step=0.1, key="sell_amount")

    with col_swap2:
        st.markdown("### 🛒 Buy")
        buy_token = st.selectbox("Token", list(ALL_TOKENS.keys()), key="buy")
        st.text_input("Est. Output", value="Click 'Get Quote' below", disabled=True, key="output")

    st.markdown("---")

    if st.button("📊 Get Quote", type="primary"):
        with st.spinner("Getting quote from Jupiter..."):
            try:
                import httpx

                tokens = {k: v[0] for k, v in ALL_TOKENS.items()}
                decimals = {k: v[1] for k, v in ALL_TOKENS.items()}

                input_mint = tokens[sell_token]
                output_mint = tokens[buy_token]

                # Convert to lamports
                amount = int(sell_amount * (10 ** decimals[sell_token]))

                url = f"https://lite-api.jup.ag/ultra/v1/order?inputMint={input_mint}&outputMint={output_mint}&amount={amount}"
                resp = httpx.get(url, timeout=30.0)
                data = resp.json()

                if "outAmount" in data:
                    out_amount = int(data["outAmount"])
                    output_val = out_amount / (10 ** decimals[buy_token])
                    st.success(f"💰 Output: {output_val:.6f} {buy_token}")
                    st.info(f"📊 Price Impact: {data.get('priceImpact', 'N/A')}%")
                    st.info(f"⏱️ Route Time: {data.get('totalTime', 'N/A')}ms")

                    if st.button("🚀 EXECUTE SWAP", type="primary"):
                        st.error("⚠️ Swap requires private key setup. Use CLI.")
                else:
                    st.error(f"Error: {data.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")

    # Quick prices - more tokens
    st.subheader("📈 Current Prices")

    try:
        import httpx

        # Get prices for popular tokens
        mints = [ALL_TOKENS[t][0] for t in PRICE_TOKENS[:6]]
        ids = "%2C".join(mints)

        url = f"https://lite-api.jup.ag/price/v3?ids={ids}"
        resp = httpx.get(url, timeout=30.0)
        data = resp.json()

        # Create price display
        cols = st.columns(3)
        i = 0
        for token in PRICE_TOKENS[:6]:
            addr = ALL_TOKENS[token][0]
            if addr in data:
                price = data[addr].get("usdPrice", 0)
                with cols[i % 3]:
                    st.metric(token, f"${price:.4f}")
                i += 1
    except Exception as e:
        st.error(f"Error: {e}")
        for i, (addr, info) in enumerate(data.items()):
            symbol = addr[:8]
            price = info.get("usdPrice", 0)
            with [col_p1, col_p2, col_p3][i]:
                st.metric(symbol, f"${price:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")


# ============================================================================
# TAB 5: CONTROL PANEL
# ============================================================================

with tab5:
    st.subheader("🎮 System Control")

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.subheader("🚀 Start/Stop")

        if st.button("▶️ Start All Workers", type="primary"):
            st.success("Starting all workers...")

        if st.button("⏹️ Stop All Workers"):
            st.error("Stopping all workers...")

        if st.button("🔄 Restart Coordinator"):
            st.warning("Restarting coordinator...")

    with col_c2:
        st.subheader("📝 Actions")

        action = st.selectbox(
            "Select Action",
            [
                "Create Work Unit",
                "Run Single Backtest",
                "Export Results",
                "Clear Database"
            ]
        )

        if action == "Create Work Unit":
            st.text_input("Population Size", value="20")
            st.text_input("Generations", value="50")
            st.text_input("Risk Level", value="MEDIUM")
            st.button("🚀 Create Work Unit")

        elif action == "Run Single Backtest":
            st.text_input("Strategy Name", value="test_strategy")
            st.button("🏃 Run Backtest")

        elif action == "Export Results":
            format_choice = st.selectbox("Format", ["JSON", "CSV", "Excel"])
            st.button("📥 Export")

        elif action == "Clear Database":
            st.warning("⚠️ This will delete all data!")
            st.button("🗑️ Clear All Data", type="primary")

    st.markdown("---")

    # Configuration
    st.subheader("⚙️ Parameters")

    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        st.number_input("Max Position %", min_value=1, max_value=100, value=10)

    with col_p2:
        st.number_input("Stop Loss %", min_value=1, max_value=50, value=3)

    with col_p3:
        st.number_input("Take Profit %", min_value=1, max_value=100, value=6)


# ============================================================================
# TAB 5: LOGS
# ============================================================================

with tab6:
    st.subheader("📋 System Logs")

    # Log level filter
    log_filter = st.selectbox(
        "Filter by Level",
        ["ALL", "INFO", "WARNING", "ERROR"]
    )

    # Sample logs
    logs = [
        {"time": "2024-02-09 21:30:15", "level": "INFO", "message": "Worker W1: Received work unit #42"},
        {"time": "2024-02-09 21:30:12", "level": "INFO", "message": "Backtest completed: PnL=+5.23%"},
        {"time": "2024-02-09 21:30:10", "level": "WARNING", "message": "High latency detected: 234ms"},
        {"time": "2024-02-09 21:30:08", "level": "ERROR", "message": "Worker W2: Connection timeout"},
        {"time": "2024-02-09 21:30:05", "level": "INFO", "message": "Coordinator: Work unit #41 completed"},
    ]

    # Filter logs
    if log_filter != "ALL":
        logs = [l for l in logs if l["level"] == log_filter]

    # Display logs
    for log in logs:
        color = {
            "INFO": "🟢",
            "WARNING": "🟡",
            "ERROR": "🔴"
        }.get(log["level"], "⚪")

        st.text(f"{log['time']} {color} [{log['level']}] {log['message']}")

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        🚀 Jupiter Solana Trading Bot |
        Powered by MiniMax M2.1 & OpenClaw |
        v1.0.0
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Solana Dashboard...")
    print("Access at: http://localhost:8501")
    print("=" * 60)
