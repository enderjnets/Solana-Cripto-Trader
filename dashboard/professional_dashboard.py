#!/usr/bin/env python3
"""
Professional Trading Dashboard
============================
Professional-grade dashboard with detailed analytics and interactive charts.

Features:
- Daily Progress Tracking
- P&L Charts
- Win Rate Analytics
- Trade History
- Performance Metrics
- Capital Evolution
- Risk Analysis

Usage:
    streamlit run dashboard/professional_dashboard.py --server.port 8501
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import json
import os

# Configure page
st.set_page_config(
    page_title="Eko Trading System - Professional Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
STATE_FILE = PROJECT_ROOT / "unified_brain_state.json"
DB_FILE = PROJECT_ROOT / "db" / "unified_trading.db"

# Custom CSS for professional look
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #3D3D3D;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #FFFFFF;
    }
    .metric-label {
        font-size: 14px;
        color: #888888;
    }
    .metric-positive {
        color: #00CC96;
    }
    .metric-negative {
        color: #EF553B;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        color: #FFFFFF;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid #3D3D3D;
    }
    .trade-card {
        background: #2D2D2D;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        background: #2D2D2D;
        border-radius: 10px 10px 0 0;
    }
</style>
""", unsafe_allow_html=True)


class DashboardData:
    """Fetch and process data for dashboard."""

    @staticmethod
    def get_state() -> dict:
        """Get brain state."""
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
        return {}

    @staticmethod
    def get_trades() -> list:
        """Get all trades."""
        if DB_FILE.exists():
            data = json.loads(DB_FILE.read_text())
            return data.get("trades", [])
        return []

    @staticmethod
    def get_daily_progress() -> pd.DataFrame:
        """Calculate daily progress metrics."""
        trades = DashboardData.get_trades()
        if not trades:
            return pd.DataFrame()

        # Group by date
        df = pd.DataFrame(trades)
        df["date"] = pd.to_datetime(df.get("timestamp", datetime.now())).dt.date
        daily = df.groupby("date").agg({
            "id": "count",
            "pnl_pct": "sum" if "pnl_pct" in df.columns else "count",
            "symbol": lambda x: list(x)
        }).reset_index()
        daily.columns = ["date", "trades", "pnl_pct", "symbols"]

        return daily

    @staticmethod
    def get_performance_metrics() -> dict:
        """Calculate comprehensive performance metrics."""
        trades = DashboardData.get_trades()
        if not trades:
            return {}

        closed = [t for t in trades if t.get("status") == "closed"]
        open_pos = [t for t in trades if t.get("status") == "open"]

        wins = sum(1 for t in closed if t.get("pnl_pct", 0) > 0)
        losses = len(closed) - wins

        total_pnl = sum(t.get("pnl", 0) for t in closed)
        avg_win = sum(t.get("pnl", 0) for t in closed if t.get("pnl", 0) > 0) / wins if wins > 0 else 0
        avg_loss = sum(abs(t.get("pnl", 0)) for t in closed if t.get("pnl", 0) <= 0) / losses if losses > 0 else 0

        return {
            "total_trades": len(trades),
            "closed_trades": len(closed),
            "open_positions": len(open_pos),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(closed) * 100 if closed else 0,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(avg_win / avg_loss) if avg_loss > 0 else 0,
            "best_trade": max((t.get("pnl_pct", 0) for t in closed), default=0),
            "worst_trade": min((t.get("pnl_pct", 0) for t in closed), default=0)
        }


def main():
    """Main dashboard application."""

    # Header
    st.title("🧠 Eko Trading System")
    st.markdown("### Professional Trading Dashboard")
    st.markdown("---")

    # Fetch data
    state = DashboardData.get_state()
    trades = DashboardData.get_trades()
    perf = DashboardData.get_performance_metrics()
    daily = DashboardData.get_daily_progress()

    # Sidebar - Quick Stats
    with st.sidebar:
        st.title("📊 Quick Stats")
        st.markdown("---")

        # Current session stats
        cycles = state.get("stats", {}).get("cycles", 0)
        trades_today = state.get("stats", {}).get("trades_today", 0)
        daily_pnl = state.get("stats", {}).get("daily_pnl_pct", 0)

        st.metric("Cycles Completed", cycles)
        st.metric("Trades Today", trades_today)
        st.metric("Daily P&L", f"{daily_pnl:+.2f}%",
                  delta=f"{daily_pnl:+.2f}%",
                  delta_color="normal")

        st.markdown("---")

        # System modules
        st.subheader("⚙️ System Status")
        modules = state.get("modules", {})
        for module, enabled in modules.items():
            emoji = "✅" if enabled else "❌"
            st.text(f"{emoji} {module.replace('_', ' ').title()}")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview",
        "💰 P&L Analysis",
        "📊 Performance",
        "📋 Trade History",
        "🎯 Risk Analysis"
    ])

    # Tab 1: Overview
    with tab1:
        st.markdown('<div class="section-header">📈 System Overview</div>', unsafe_allow_html=True)

        # Top metrics row
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{perf.get('total_trades', 0)}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            win_rate = perf.get('win_rate', 0)
            color = "metric-positive" if win_rate >= 50 else "metric-negative"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value {color}">{win_rate:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            total_pnl = perf.get('total_pnl', 0)
            color = "metric-positive" if total_pnl >= 0 else "metric-negative"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value {color}">${total_pnl:+.2f}</div>
                <div class="metric-label">Total P&L</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            best = perf.get('best_trade', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value metric-positive">+{best:.2f}%</div>
                <div class="metric-label">Best Trade</div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            worst = perf.get('worst_trade', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value metric-negative">{worst:.2f}%</div>
                <div class="metric-label">Worst Trade</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Charts row
        col_left, col_right = st.columns([2, 1])

        with col_left:
            # P&L over time chart
            if trades:
                df = pd.DataFrame(trades[-50:])
                if "pnl_pct" in df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(len(df))),
                        y=df["pnl_pct"].cumsum(),
                        mode="lines+markers",
                        line=dict(color="#00CC96", width=2),
                        marker=dict(size=6, color=df["pnl_pct"].apply(
                            lambda x: "#00CC96" if x >= 0 else "#EF553B"
                        )),
                        name="Cumulative P&L"
                    ))
                    fig.update_layout(
                        title="Cumulative P&L Over Time",
                        paper_bgcolor="#1E1E1E",
                        plot_bgcolor="#2D2D2D",
                        font=dict(color="white"),
                        xaxis_title="Trade Number",
                        yaxis_title="P&L (%)",
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with col_right:
            # Win/Loss pie chart
            if perf.get('wins', 0) > 0 or perf.get('losses', 0) > 0:
                fig = go.Figure(data=[go.Pie(
                    labels=["Wins", "Losses"],
                    values=[perf.get('wins', 0), perf.get('losses', 0)],
                    hole=0.6,
                    marker=dict(colors=["#00CC96", "#EF553B"])
                )])
                fig.update_layout(
                    title="Win/Loss Distribution",
                    paper_bgcolor="#1E1E1E",
                    font=dict(color="white"),
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)

    # Tab 2: P&L Analysis
    with tab2:
        st.markdown('<div class="section-header">💰 P&L Analysis</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # P&L by symbol
            if trades:
                df = pd.DataFrame(trades)
                if "symbol" in df.columns and "pnl" in df.columns:
                    pnl_by_symbol = df.groupby("symbol")["pnl"].sum().sort_values(ascending=False)

                    fig = go.Figure(go.Bar(
                        x=pnl_by_symbol.index,
                        y=pnl_by_symbol.values,
                        marker_color=["#00CC96" if v >= 0 else "#EF553B" for v in pnl_by_symbol.values]
                    ))
                    fig.update_layout(
                        title="P&L by Symbol",
                        paper_bgcolor="#1E1E1E",
                        plot_bgcolor="#2D2D2D",
                        font=dict(color="white"),
                        xaxis_title="Symbol",
                        yaxis_title="P&L ($)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Daily P&L bar chart
            if not daily.empty:
                colors = ["#00CC96" if v >= 0 else "#EF553B" for v in daily["pnl_pct"]]
                fig = go.Figure(go.Bar(
                    x=[str(d) for d in daily["date"]],
                    y=daily["pnl_pct"],
                    marker_color=colors
                ))
                fig.update_layout(
                    title="Daily P&L",
                    paper_bgcolor="#1E1E1E",
                    plot_bgcolor="#2D2D2D",
                    font=dict(color="white"),
                    xaxis_title="Date",
                    yaxis_title="P&L (%)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

        # Detailed P&L stats
        st.markdown("### 📊 P&L Statistics")
        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            st.metric("Total P&L", f"${perf.get('total_pnl', 0):+.2f}")
        with col_b:
            st.metric("Average Win", f"${perf.get('avg_win', 0):+.2f}")
        with col_c:
            st.metric("Average Loss", f"${perf.get('avg_loss', 0):+.2f}")
        with col_d:
            pf = perf.get('profit_factor', 0)
            st.metric("Profit Factor", f"{pf:.2f}x")

    # Tab 3: Performance
    with tab3:
        st.markdown('<div class="section-header">📊 Performance Analytics</div>', unsafe_allow_html=True)

        # Performance metrics grid
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Win Rate", f"{perf.get('win_rate', 0):.1f}%",
                     delta="Good" if perf.get('win_rate', 0) >= 50 else "Needs Work")
        with col2:
            st.metric("Total Trades", perf.get('total_trades', 0))
        with col3:
            st.metric("Closed Trades", perf.get('closed_trades', 0))
        with col4:
            st.metric("Open Positions", perf.get('open_positions', 0))

        st.markdown("---")

        # Performance over time
        if trades:
            df = pd.DataFrame(trades[-100:])
            if "pnl_pct" in df.columns:
                # Rolling win rate
                df["rolling_wr"] = df["pnl_pct"].apply(lambda x: 1 if x > 0 else 0).rolling(10).mean() * 100

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  subplot_titles=("Trade P&L", "Rolling Win Rate (10 trades)"))

                # P&L line
                fig.add_trace(go.Scatter(
                    x=list(range(len(df))),
                    y=df["pnl_pct"].cumsum(),
                    mode="lines",
                    line=dict(color="#00CC96", width=2),
                    name="Cumulative P&L"
                ), row=1, col=1)

                # Win rate
                fig.add_trace(go.Scatter(
                    x=list(range(len(df))),
                    y=df["rolling_wr"],
                    mode="lines",
                    line=dict(color="#636EFA", width=2),
                    name="Win Rate"
                ), row=2, col=1)

                fig.update_layout(
                    title="Performance Over Time",
                    paper_bgcolor="#1E1E1E",
                    plot_bgcolor="#2D2D2D",
                    font=dict(color="white"),
                    height=500,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

    # Tab 4: Trade History
    with tab4:
        st.markdown('<div class="section-header">📋 Trade History</div>', unsafe_allow_html=True)

        if trades:
            df = pd.DataFrame(trades[-50:])
            display_df = df[["symbol", "direction", "entry_price", "pnl_pct", "status", "timestamp", "strategy"]].copy()
            display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%H:%M:%S")
            display_df["pnl_pct"] = display_df["pnl_pct"].apply(lambda x: f"{x:+.2f}%")

            # Color coding
            def color_pnl(val):
                if "+" in str(val):
                    return "color: #00CC96"
                elif "-" in str(val):
                    return "color: #EF553B"
                return ""

            st.dataframe(
                display_df.style.applymap(color_pnl, subset=["pnl_pct"]),
                use_container_width=True
            )

            # Export option
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Export Trade History",
                csv,
                "trade_history.csv",
                "text/csv"
            )
        else:
            st.info("No trades recorded yet")

    # Tab 5: Risk Analysis
    with tab5:
        st.markdown('<div class="section-header">🎯 Risk Analysis</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⚠️ Risk Metrics")

            risk_metrics = {
                "Max Daily Loss Limit": "10%",
                "Current Exposure": f"{min(perf.get('open_positions', 0) * 2, 100)}%",
                "Stop Loss per Trade": "5%",
                "Take Profit per Trade": "10%",
                "Position Size": "$20 (4% of capital)",
                "Max Positions": "15"
            }

            for metric, value in risk_metrics.items():
                st.text(f"• {metric}: {value}")

        with col2:
            st.subheader("📉 Risk Visualization")

            if perf.get('total_trades', 0) > 0:
                # Risk/reward ratio
                fig = go.Figure()

                fig.add_trace(go.Bar(
                    name="Wins",
                    x=["Average"],
                    y=[perf.get('avg_win', 0)],
                    marker_color="#00CC96"
                ))

                fig.add_trace(go.Bar(
                    name="Losses",
                    x=["Average"],
                    y=[perf.get('avg_loss', 0)],
                    marker_color="#EF553B"
                ))

                fig.update_layout(
                    title="Average Win vs Loss",
                    paper_bgcolor="#1E1E1E",
                    plot_bgcolor="#2D2D2D",
                    font=dict(color="white"),
                    barmode="group",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

        # Risk warnings
        st.markdown("### 🚨 Risk Alerts")
        alerts = []

        if perf.get('win_rate', 0) < 40:
            alerts.append("⚠️ Win rate below 40% - Review strategy")
        if perf.get('profit_factor', 0) < 1:
            alerts.append("⚠️ Profit factor below 1 - Losing strategy")
        if perf.get('worst_trade', 0) < -10:
            alerts.append("⚠️ Large losses detected - Check stop losses")

        if alerts:
            for alert in alerts:
                st.warning(alert)
        else:
            st.success("✅ No risk alerts - All metrics within limits")

    # Footer
    st.markdown("---")
    timestamp = state.get("timestamp", "")
    st.caption(f"Last updated: {timestamp} | Eko Trading System v3.0 Professional Dashboard")
