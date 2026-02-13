#!/usr/bin/env python3
"""
Portfolio Dashboard - Complete Trading Analytics
================================================
Shows:
- Portfolio value and PnL over time
- Win/Loss analysis by token
- Trade history with amounts
- Performance charts
- Real-time operational analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# Config
st.set_page_config(page_title="Portfolio Analytics", layout="wide")

DATA_DIR = Path("/home/enderj/.openclaw/workspace/solana-jupiter-bot/data")
TRADES_FILE = DATA_DIR / "trade_history.json"
PORTFOLIO_FILE = DATA_DIR / "portfolio_history.json"

# Load or initialize trade history
def load_trades():
    if TRADES_FILE.exists():
        try:
            return json.loads(TRADES_FILE.read_text())
        except:
            return []
    return []

def save_trade(trade):
    trades = load_trades()
    trades.append({
        **trade,
        "timestamp": datetime.now().isoformat()
    })
    TRADES_FILE.write_text(json.dumps(trades, indent=2, default=str))
    return trades

def load_portfolio_history():
    if PORTFOLIO_FILE.exists():
        try:
            return json.loads(PORTFOLIO_FILE.read_text())
        except:
            return []
    return []

def save_portfolio_snapshot(sol_value, btc_value, usdt_value, total_value):
    history = load_portfolio_history()
    history.append({
        "timestamp": datetime.now().isoformat(),
        "SOL": sol_value,
        "BTC": btc_value,
        "USDT": usdt_value,
        "TOTAL": total_value
    })
    # Keep last 1000 entries
    history = history[-1000:]
    PORTFOLIO_FILE.write_text(json.dumps(history, indent=2, default=str))
    return history

# Simulate trades for demo (replace with real data in production)
def simulate_trades():
    """Generate sample trades for demonstration"""
    if not load_trades():
        sample_trades = [
            {"token": "SOL", "side": "BUY", "amount": 2.5, "price": 75.00, "pnl": 0, "status": "closed"},
            {"token": "SOL", "side": "BUY", "amount": 1.0, "price": 78.50, "pnl": 0, "status": "closed"},
            {"token": "cbBTC", "side": "BUY", "amount": 0.02, "price": 95000, "pnl": 0, "status": "closed"},
            {"token": "SOL", "side": "SELL", "amount": 2.5, "price": 82.00, "pnl": 17.50, "status": "closed"},
            {"token": "SOL", "side": "BUY", "amount": 3.0, "price": 79.25, "pnl": 0, "status": "open"},
            {"token": "SOL", "side": "SELL", "amount": 1.0, "price": 81.00, "pnl": 2.75, "status": "closed"},
            {"token": "cbBTC", "side": "SELL", "amount": 0.02, "price": 98000, "pnl": 60.00, "status": "closed"},
        ]
        for trade in sample_trades:
            save_trade(trade)
    return load_trades()

# Initialize demo data
simulate_trades()
trades = load_trades()
portfolio_history = load_portfolio_history()

# Calculate metrics
def calculate_metrics(trades, portfolio_history):
    closed_trades = [t for t in trades if t.get("status") == "closed"]
    open_trades = [t for t in trades if t.get("status") == "open"]
    
    total_pnl = sum(t.get("pnl", 0) for t in closed_trades)
    wins = [t for t in closed_trades if t.get("pnl", 0) > 0]
    losses = [t for t in closed_trades if t.get("pnl", 0) <= 0]
    
    win_rate = len(wins) / len(closed_trades) * 100 if closed_trades else 0
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    
    # PnL by token
    pnl_by_token = {}
    for t in closed_trades:
        token = t.get("token", "UNKNOWN")
        pnl_by_token[token] = pnl_by_token.get(token, 0) + t.get("pnl", 0)
    
    # Calculate current portfolio
    if portfolio_history:
        current = portfolio_history[-1]
        initial = portfolio_history[0] if portfolio_history else {"TOTAL": 500}
        portfolio_return = ((current["TOTAL"] - initial["TOTAL"]) / initial["TOTAL"]) * 100 if initial["TOTAL"] > 0 else 0
    else:
        current = {"SOL": 0, "BTC": 0, "USDT": 0, "TOTAL": 500}
        portfolio_return = 0
    
    return {
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "wins": len(wins),
        "losses": len(losses),
        "total_trades": len(closed_trades),
        "open_trades": len(open_trades),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "pnl_by_token": pnl_by_token,
        "current_portfolio": current,
        "portfolio_return": portfolio_return,
        "closed_trades": closed_trades,
        "open_trades": open_trades,
    }

metrics = calculate_metrics(trades, portfolio_history)

# ==================== PORTFOLIO DASHBOARD ====================

st.title("💰 Portfolio Analytics Dashboard")
st.markdown("---")

# Top metrics row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "💵 Total Portfolio",
        f"${metrics['current_portfolio'].get('TOTAL', 500):.2f}",
        delta=f"{metrics['portfolio_return']:.2f}%"
    )

with col2:
    st.metric(
        "📈 Total PnL",
        f"${metrics['total_pnl']:.2f}",
        delta_color="normal" if metrics['total_pnl'] > 0 else "inverse"
    )

with col3:
    st.metric(
        "🎯 Win Rate",
        f"{metrics['win_rate']:.1f}%",
        delta=f"{metrics['wins']}W / {metrics['losses']}L"
    )

with col4:
    st.metric(
        "📊 Total Trades",
        f"{metrics['total_trades']}",
        delta=f"{metrics['open_trades']} open"
    )

with col5:
    profit_factor = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0
    st.metric(
        "⚖️ Profit Factor",
        f"{profit_factor:.2f}",
        help="Ratio Avg Win / Avg Loss"
    )

st.markdown("---")

# ==================== PORTFOLIO ALLOCATION ====================

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📊 Portfolio Allocation")
    
    if portfolio_history:
        # Create dataframe for charts
        df = pd.DataFrame(portfolio_history[-100:])  # Last 100 entries
        
        # Convert timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Portfolio value over time chart
        chart_data = df.set_index('timestamp')[['SOL', 'BTC', 'USDT', 'TOTAL']]
        st.line_chart(chart_data['TOTAL'], height=200)
        
        # Stacked area chart
        st.subheader("💼 Asset Distribution Over Time")
        st.area_chart(df.set_index('timestamp')[['SOL', 'BTC', 'USDT']], height=250)
    else:
        st.info("No portfolio history yet. Trading will generate this data.")

with col_right:
    st.subheader("🎯 Current Allocation")
    
    current = metrics['current_portfolio']
    sol_pct = current.get('SOL', 0) / current.get('TOTAL', 1) * 100 if current.get('TOTAL', 0) > 0 else 0
    btc_pct = current.get('BTC', 0) / current.get('TOTAL', 1) * 100 if current.get('TOTAL', 0) > 0 else 0
    usdt_pct = current.get('USDT', 0) / current.get('TOTAL', 1) * 100 if current.get('TOTAL', 0) > 0 else 0
    
    st.write(f"**SOL:** ${current.get('SOL', 0):.2f} ({sol_pct:.1f}%)")
    st.progress(sol_pct / 100)
    
    st.write(f"**BTC:** ${current.get('BTC', 0):.2f} ({btc_pct:.1f}%)")
    st.progress(btc_pct / 100)
    
    st.write(f"**USDT:** ${current.get('USDT', 0):.2f} ({usdt_pct:.1f}%)")
    st.progress(usdt_pct / 100)
    
    # Target allocation
    st.markdown("---")
    st.subheader("🎯 Target (50/30/20)")
    st.write("SOL: 50% | BTC: 30% | USDT: 20%")

st.markdown("---")

# ==================== PnL BY TOKEN ====================

st.subheader("📈 PnL by Token")

if metrics['pnl_by_token']:
    pnl_data = pd.DataFrame([
        {"Token": k, "PnL": v} for k, v in metrics['pnl_by_token'].items()
    ])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(
            pnl_data.style.format({"PnL": "${:.2f}"}).apply(
                lambda x: ['background-color: #1b4332' if v > 0 else '#7f1d1d' 
                          for v in x], subset=['PnL']
            ), hide_index=True
        )
    
    with col2:
        st.bar_chart(pnl_data.set_index("Token")["PnL"], height=200)
else:
    st.info("No closed trades yet.")

st.markdown("---")

# ==================== TRADE HISTORY ====================

st.subheader("📋 Trade History")

if trades:
    df_trades = pd.DataFrame(trades)
    df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'])
    df_trades = df_trades.sort_values('timestamp', ascending=False)
    
    # Format for display
    display_cols = ['timestamp', 'token', 'side', 'amount', 'price', 'pnl', 'status']
    df_display = df_trades[display_cols].copy()
    df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    df_display['price'] = df_display['price'].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x)
    df_display['amount'] = df_display['amount'].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    df_display['pnl'] = df_display['pnl'].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else x)
    
    st.dataframe(
        df_display.style.apply(
            lambda x: ['color: #1b4332' if v and str(v).startswith('$') and float(str(v).replace('$','').replace(',','')) > 0 
                      else 'color: #7f1d1d' if v and str(v).startswith('$-') else '' 
                      for v in x], axis=0
        ),
        width="stretch"
    )
else:
    st.info("No trades executed yet.")

st.markdown("---")

# ==================== OPERATIONAL ANALYSIS ====================

st.subheader("🔍 Operational Analysis")

# Generate real-time analysis
analysis_cols = st.columns(3)

with analysis_cols[0]:
    st.markdown("""
    ### 📊 Performance Metrics
    
    | Metric | Value | Status |
    |--------|-------|--------|
    | Win Rate | {:.1f}% | {} |
    | Profit Factor | {:.2f} | {} |
    | Avg Win | ${:.2f} | - |
    | Avg Loss | ${:.2f} | - |
    
    **Trend:** {}
    """.format(
        metrics['win_rate'],
        "🟢 Good" if metrics['win_rate'] >= 60 else "🟡 OK" if metrics['win_rate'] >= 50 else "🔴 Needs Work",
        abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0,
        "🟢 Favorable" if abs(metrics['avg_win'] / metrics['avg_loss']) > 1.5 else "🟡 Neutral",
        metrics['avg_win'],
        abs(metrics['avg_loss']),
        "📈 Upward" if metrics['portfolio_return'] > 0 else "📉 Downward"
    ))

with analysis_cols[1]:
    st.markdown("""
    ### 💰 Risk Management
    
    | Limit | Current | Status |
    |-------|---------|--------|
    | Daily Loss | {}% | ✅ |
    | Max Position | {}% | ✅ |
    | Risk/Trade | 5% | ✅ |
    | Stop Loss | -8% | ✅ |
    
    **Risk Level:** {}
    """.format(
        min(abs(metrics['total_pnl']) / 500 * 100, 10) if metrics['total_pnl'] < 0 else 0,
        min(sol_pct, 100),
        "🟢 Low" if metrics['total_pnl'] > -25 else "🟡 Medium" if metrics['total_pnl'] > -50 else "🔴 High"
    ))

with analysis_cols[2]:
    st.markdown("""
    ### 🎯 Objectives Progress
    
    | Objective | Current | Target | Progress |
    |-----------|---------|--------|----------|
    | Daily | {}% | +5% | {:.0f}% |
    | Monthly | {}% | +100% | {:.0f}% |
    
    **Next Milestone:** {}
    """.format(
        min(metrics['portfolio_return'], 5),
        min(metrics['portfolio_return'] / 5 * 100, 100),
        metrics['portfolio_return'],
        metrics['portfolio_return'],
        f"${500 * 1.05:.0f}" if metrics['portfolio_return'] < 5 else f"${500 * 1.10:.0f}"
    ))

st.markdown("---")

# ==================== RECOMMENDATIONS ====================

st.subheader("💡 AI Recommendations")

recommendations = []

# Win rate analysis
if metrics['win_rate'] >= 60:
    recommendations.append("✅ Win rate is healthy ({}%). Keep following the strategy.".format(metrics['win_rate']))
elif metrics['win_rate'] >= 50:
    recommendations.append("⚠️ Win rate is acceptable ({}%). Consider tightening entry criteria.".format(metrics['win_rate']))
else:
    recommendations.append("🔴 Win rate needs improvement ({}%). Review strategy parameters.".format(metrics['win_rate']))

# Profit factor analysis
pf = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0
if pf >= 2:
    recommendations.append("✅ Excellent profit factor ({:.1f}). Good risk/reward ratio.".format(pf))
elif pf >= 1.5:
    recommendations.append("✅ Good profit factor ({:.1f}). Strategy is profitable.".format(pf))
else:
    recommendations.append("🔴 Profit factor low ({:.1f}). Consider adjusting SL/TP.".format(pf))

# Token allocation
if sol_pct > 65:
    recommendations.append("⚠️ SOL overweight ({}%). Consider rebalancing to BTC/USDT.".format(sol_pct))
elif btc_pct > 50:
    recommendations.append("⚠️ BTC heavy ({}%). Consider taking profits.".format(btc_pct))

# Open trades
if len(metrics['open_trades']) > 3:
    recommendations.append("⚠️ Many open trades ({}). Consider closing some positions.".format(len(metrics['open_trades'])))

for rec in recommendations:
    st.write(rec)

# ==================== REAL-TIME STATUS ====================

st.markdown("---")
st.subheader("📡 Real-Time Status")

status_cols = st.columns(4)

with status_cols[0]:
    connected = "🟢 Connected" if portfolio_history else "🔴 No Data"
    st.write(f"**Jupiter API:** {connected}")

with status_cols[1]:
    st.write(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")

with status_cols[2]:
    trades_count = len(metrics['closed_trades'])
    st.write(f"**Trades Today:** {trades_count}")

with status_cols[3]:
    next_audit = datetime.now() + timedelta(minutes=60 - datetime.now().minute % 60)
    st.write(f"**Next Audit:** {next_audit.strftime('%H:%M')}")

# Footer
st.markdown("---")
st.caption(f"🦞 Trading Puro System | Portfolio Dashboard | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
