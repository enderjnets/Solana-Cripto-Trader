#!/usr/bin/env python3
"""
Unified Trading Dashboard
========================
Real-time monitoring dashboard for the trading system.

Features:
- Live P&L tracking
- Open positions monitor
- Trade history
- Performance metrics
- Alert feed

Usage:
    python3 dashboard/unified_dashboard.py --port 8501
    streamlit run dashboard/unified_dashboard.py
"""

import json
import asyncio
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
STATE_FILE = PROJECT_ROOT / "unified_brain_state.json"
DB_FILE = PROJECT_ROOT / "db" / "unified_trading.db"


class DashboardData:
    """Fetches data for dashboard."""

    @staticmethod
    def get_state() -> Dict:
        """Get brain state."""
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
        return {}

    @staticmethod
    def get_trades() -> List[Dict]:
        """Get trade history."""
        if DB_FILE.exists():
            data = json.loads(DB_FILE.read_text())
            return data.get("trades", [])
        return []

    @staticmethod
    def get_performance() -> Dict:
        """Calculate performance metrics."""
        trades = DashboardData.get_trades()
        closed = [t for t in trades if t.get("status") == "closed"]

        if not closed:
            return {"win_rate": 0, "total_pnl": 0, "trades": 0}

        wins = sum(1 for t in closed if t.get("pnl", 0) > 0)
        total_pnl = sum(t.get("pnl", 0) for t in closed)

        return {
            "total_trades": len(closed),
            "winning_trades": wins,
            "losing_trades": len(closed) - wins,
            "win_rate": wins / len(closed) * 100 if closed else 0,
            "total_pnl": total_pnl
        }


def main():
    """Main dashboard."""
    st.set_page_config(
        page_title="Unified Trading Dashboard",
        page_icon="📈",
        layout="wide"
    )

    st.title("🧠 Unified Trading Dashboard")
    st.markdown("---")

    # Auto-refresh
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 60, 10)
    st_autorefresh = st.autorefresh(refresh_rate * 1000)

    # Get data
    state = DashboardData.get_state()
    trades = DashboardData.get_trades()
    perf = DashboardData.get_performance()

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        daily_pnl = state.get("stats", {}).get("daily_pnl_pct", 0)
        color = "green" if daily_pnl >= 0 else "red"
        st.markdown(f"""
        <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: {color}; margin: 0;">{daily_pnl:+.2f}%</h2>
            <p style="color: #888; margin: 0;">Daily P&L</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        trades_today = state.get("stats", {}).get("trades_today", 0)
        st.markdown(f"""
        <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: white; margin: 0;">{trades_today}</h2>
            <p style="color: #888; margin: 0;">Trades Today</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        win_rate = perf.get("win_rate", 0)
        color = "green" if win_rate >= 50 else "orange"
        st.markdown(f"""
        <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: {color}; margin: 0;">{win_rate:.1f}%</h2>
            <p style="color: #888; margin: 0;">Win Rate</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        total_pnl = perf.get("total_pnl", 0)
        color = "green" if total_pnl >= 0 else "red"
        st.markdown(f"""
        <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: {color}; margin: 0;">${total_pnl:+.2f}</h2>
            <p style="color: #888; margin: 0;">Total P&L</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Main content
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Open positions
        st.subheader("📋 Open Positions")
        open_trades = [t for t in trades if t.get("status") == "open"]

        if open_trades:
            for trade in open_trades[-10:]:
                symbol = trade.get("symbol", "")
                pnl = trade.get("pnl_pct", 0)
                color = "green" if pnl >= 0 else "red"
                st.markdown(f"""
                <div style="background-color: #2D2D2D; padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <span style="color: white;">{symbol}</span>
                    <span style="float: right; color: {color};">{pnl:+.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No open positions")

    with col_right:
        # System status
        st.subheader("⚙️ System Status")
        modules = state.get("modules", {})
        for module, enabled in modules.items():
            emoji = "✅" if enabled else "❌"
            st.text(f"{emoji} {module.replace('_', ' ').title()}")

        # Performance chart
        st.subheader("📊 Performance")
        if len(trades) >= 2:
            df = pd.DataFrame(trades[-20:])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(df))),
                y=df.get("pnl_pct", [0]*len(df)),
                mode="lines+markers",
                line=dict(color="#00CC96", width=2),
                marker=dict(size=8)
            ))
            fig.update_layout(
                paper_bgcolor="#1E1E1E",
                plot_bgcolor="#1E1E1E",
                font=dict(color="white"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=200
            )
            st.plotly_chart(fig, use_container_width=True)

    # Trade history
    st.markdown("---")
    st.subheader("📜 Trade History")

    if trades:
        df = pd.DataFrame(trades[-20:])
        st.dataframe(
            df[["symbol", "direction", "entry_price", "pnl_pct", "status", "timestamp"]],
            use_container_width=True
        )
    else:
        st.info("No trades yet")

    # Footer
    st.markdown("---")
    timestamp = state.get("timestamp", "")
    st.caption(f"Last updated: {timestamp}")


if __name__ == "__main__":
    main()
