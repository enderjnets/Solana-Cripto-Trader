#!/usr/bin/env python3
"""
Multi-Agent Dashboard - LIVE Agent Interaction Visualizer
=========================================================
Reads real-time data from agent_state.json and brain_state.json
produced by agent_runner.py and agent_brain.py.

Auto-refreshes every 5 seconds.
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# DATA LOADING
# ============================================================================

def load_json(path: Path) -> Dict:
    """Safely load a JSON file."""
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def load_runner_state() -> Dict:
    return load_json(PROJECT_ROOT / "agent_state.json")


def load_brain_state() -> Dict:
    return load_json(PROJECT_ROOT / "brain_state.json")


def load_active_strategies() -> Dict:
    return load_json(PROJECT_ROOT / "data" / "active_strategies.json")


def load_knowledge() -> Dict:
    return load_json(PROJECT_ROOT / "data" / "knowledge" / "strategy_performance.json")


def load_watchlist() -> Dict:
    return load_json(PROJECT_ROOT / "data" / "token_watchlist.json")


def load_optimized() -> Dict:
    return load_json(PROJECT_ROOT / "data" / "optimized_strategies.json")


# ============================================================================
# DASHBOARD
# ============================================================================

def create_dashboard():
    import streamlit as st
    import pandas as pd
    import numpy as np

    st.set_page_config(
        page_title="Eko - Multi-Agent Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .agent-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 12px;
        margin: 8px 0;
        border-left: 4px solid #00d4ff;
        color: #e0e0e0;
    }
    .agent-active { border-left-color: #00ff88; }
    .agent-error { border-left-color: #ff6b6b; }
    .agent-idle { border-left-color: #ffd700; }
    .feed-entry {
        background: #0f0f23;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        border-left: 3px solid #00d4ff;
        color: #d0d0d0;
        font-size: 0.9em;
    }
    .feed-entry strong { color: #00d4ff; }
    .feed-entry .action { color: #00ff88; }
    .metric-card {
        background: #16213e;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Load all state
    runner = load_runner_state()
    brain = load_brain_state()
    strategies = load_active_strategies()
    knowledge = load_knowledge()
    watchlist = load_watchlist()
    optimized = load_optimized()

    runner_running = runner.get("running", False)
    brain_running = brain.get("running", False)

    # ======================== SIDEBAR ========================
    with st.sidebar:
        st.header("🤖 System Status")

        # Runner status
        r_color = "#00ff88" if runner_running else "#ff6b6b"
        r_status = "RUNNING" if runner_running else "STOPPED"
        st.markdown(f"**Agent Runner**: <span style='color:{r_color}'>{r_status}</span>", unsafe_allow_html=True)
        if runner_running:
            st.caption(f"Cycle: {runner.get('cycle', 0)} | Mode: {runner.get('mode', '?')}")

        # Brain status
        b_color = "#00ff88" if brain_running else "#ff6b6b"
        b_status = "RUNNING" if brain_running else "STOPPED"
        st.markdown(f"**Agent Brain**: <span style='color:{b_color}'>{b_status}</span>", unsafe_allow_html=True)
        if brain_running:
            opt = brain.get("optimizer", {})
            st.caption(f"Cycle: {brain.get('cycle', 0)} | Gen: {opt.get('generation', 0)}")

        st.markdown("---")

        # Agent cards
        st.subheader("Agents")

        # Runner agents
        runner_agents = [
            ("🎯", "Coordinator", "active" if runner_running else "idle"),
            ("📊", "Analysis", "active" if runner_running else "idle"),
            ("🛡️", "Risk", "active" if runner_running else "idle"),
            ("💰", "Trading", "active" if runner_running else "idle"),
            ("📈", "Strategy", "active" if runner_running else "idle"),
        ]

        # Brain agents
        last_cycle = brain.get("last_cycle", {}).get("agents", {})
        brain_agents = [
            ("🔍", "TokenScout", last_cycle.get("scout", {}).get("status", "idle")),
            ("📥", "DataCollector", last_cycle.get("data_collector", {}).get("status", "idle")),
            ("🧪", "Backtester", last_cycle.get("backtester", {}).get("status", "idle")),
            ("🧬", "Optimizer", last_cycle.get("optimizer", {}).get("status", "idle")),
            ("🧠", "Learner", last_cycle.get("learner", {}).get("status", "idle")),
            ("🚀", "Deployer", last_cycle.get("deployer", {}).get("status", "idle")),
        ]

        for icon, name, status in runner_agents + brain_agents:
            css = "agent-active" if status == "ok" or status == "active" else "agent-error" if status == "error" else "agent-idle"
            color = "#00ff88" if status in ("ok", "active") else "#ff6b6b" if status == "error" else "#ffd700"
            st.markdown(f"""<div class="agent-card {css}">
                {icon} <strong>{name}</strong>
                <span style="color:{color};float:right">● {status.upper()}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Quick metrics
        price_hist = runner.get("price_history", [])
        current_price = price_hist[-1].get("price", 0) if price_hist else 0
        st.metric("SOL Price", f"${current_price:.2f}" if current_price else "N/A")
        st.metric("Wallet", runner.get("wallet", "N/A")[:12] + "...")

        last_update = runner.get("last_update", brain.get("last_update", ""))
        if last_update:
            try:
                dt = datetime.fromisoformat(last_update)
                ago = (datetime.now() - dt).seconds
                st.caption(f"Updated {ago}s ago")
            except Exception:
                pass

    # ======================== MAIN TABS ========================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📡 Activity Feed",
        "🧬 Brain & Optimizer",
        "📈 Trading & Strategies",
        "🧠 Knowledge Base",
        "🔍 Token Scout",
    ])

    # ======================== TAB 1: ACTIVITY FEED ========================
    with tab1:
        st.header("📡 Live Activity Feed")

        # Combine activities from runner and brain
        activities = runner.get("recent_activity", [])
        feed_items = []

        for a in activities:
            feed_items.append({
                "time": a.get("time", ""),
                "cycle": a.get("cycle", 0),
                "agent": a.get("agent", "?"),
                "action": a.get("action", ""),
                "detail": a.get("detail", ""),
                "source": "runner",
            })

        # Brain activities from last cycle
        brain_cycle = brain.get("last_cycle", {})
        brain_agents_data = brain_cycle.get("agents", {})
        brain_time = brain_cycle.get("time", "")
        for agent_name, info in brain_agents_data.items():
            status = info.get("status", "?")
            detail_parts = []
            for k, v in info.items():
                if k != "status":
                    detail_parts.append(f"{k}={v}")
            feed_items.append({
                "time": brain_time[:19] if brain_time else "",
                "cycle": brain.get("cycle", 0),
                "agent": agent_name.replace("_", " ").title(),
                "action": status.upper(),
                "detail": ", ".join(detail_parts),
                "source": "brain",
            })

        # Sort by time (newest first)
        feed_items.sort(key=lambda x: x.get("time", ""), reverse=True)

        if not feed_items:
            st.info("No activity yet. Waiting for agents to produce data...")
        else:
            st.caption(f"Showing {len(feed_items)} entries")
            for item in feed_items[:30]:
                source_icon = "🔄" if item["source"] == "runner" else "🧠"
                action_color = "#00ff88" if item["action"] in ("OK", "Complete", "Approved", "EXECUTING") else "#ffd700" if item["action"] in ("Starting", "Evaluating", "Validating") else "#ff6b6b" if "error" in item["action"].lower() or "Rejected" in item["action"] else "#00d4ff"
                st.markdown(f"""<div class="feed-entry">
                    {source_icon} <strong>[{item['time']}]</strong>
                    Cycle {item['cycle']} |
                    <strong>{item['agent']}</strong> →
                    <span class="action" style="color:{action_color}">{item['action']}</span>:
                    {item['detail']}
                </div>""", unsafe_allow_html=True)

    # ======================== TAB 2: BRAIN & OPTIMIZER ========================
    with tab2:
        st.header("🧬 Brain & Genetic Optimizer")

        opt = brain.get("optimizer", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Generation", opt.get("generation", 0))
        col2.metric("Population", opt.get("population_size", 0))
        best = opt.get("best_ever")
        col3.metric("Best PnL", f"{best:.4f}" if best else "N/A")
        col4.metric("Best Strategy", opt.get("best_strategy", "N/A"))

        st.markdown("---")

        # Optimized strategies
        st.subheader("🏆 Top Optimized Strategies")
        opt_strats = optimized.get("strategies", [])
        if opt_strats:
            rows = []
            for s in opt_strats:
                bt = s.get("backtest", {})
                rows.append({
                    "Name": s.get("name", "?"),
                    "PnL": f"{bt.get('pnl', s.get('fitness', 0)):.4f}",
                    "Win Rate": f"{bt.get('win_rate', 0):.0%}",
                    "Trades": bt.get("trades", 0),
                    "SL%": f"{s.get('params', {}).get('sl_pct', 0)*100:.1f}%",
                    "TP%": f"{s.get('params', {}).get('tp_pct', 0)*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("Optimizer hasn't produced strategies yet...")

        # Deployed strategies
        st.subheader("🚀 Deployed to Live Trading")
        deployed = strategies.get("strategies", [])
        if deployed:
            for s in deployed:
                st.success(f"**{s.get('name', '?')}**: {s.get('description', '')}")
        else:
            st.warning("No strategies deployed yet")

        # Deployment history
        deploy_hist = brain.get("deployer", {}).get("history", [])
        if deploy_hist:
            st.subheader("📋 Deployment History")
            for h in reversed(deploy_hist[-5:]):
                st.write(f"- {h.get('time', '?')[:19]}: Deployed {h.get('count', 0)} strategies: {', '.join(h.get('names', []))}")

    # ======================== TAB 3: TRADING & STRATEGIES ========================
    with tab3:
        st.header("📈 Trading & Strategy Performance")

        # Runner strategy info
        runner_strats = runner.get("strategies", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("Active Strategy", runner_strats.get("active", "N/A"))
        col2.metric("Total Strategies", runner_strats.get("total_strategies", 0))
        col3.metric("Trades Today", runner.get("risk", {}).get("trades_today", 0))

        st.markdown("---")

        # Strategy details
        st.subheader("📊 All Strategies")
        strat_list = runner_strats.get("strategies", [])
        if strat_list:
            rows = []
            for s in strat_list:
                rows.append({
                    "Name": s.get("name", "?"),
                    "Score": s.get("score", 0),
                    "Trades": s.get("trades", 0),
                    "Wins": s.get("wins", 0),
                    "Win Rate": f"{s['wins']/s['trades']:.0%}" if s.get("trades", 0) > 0 else "N/A",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Price history chart
        st.subheader("📉 SOL Price History")
        price_hist = runner.get("price_history", [])
        if price_hist and len(price_hist) > 1:
            df = pd.DataFrame(price_hist)
            df["time"] = pd.to_datetime(df["time"])
            st.line_chart(df.set_index("time")["price"])
        elif price_hist:
            st.info(f"Collecting price data... ({len(price_hist)} points so far)")
        else:
            st.info("No price data yet. Runner needs a few cycles to collect data.")

        # Order history
        st.subheader("📋 Recent Orders")
        orders = runner.get("order_history", [])
        if orders:
            for o in reversed(orders[-10:]):
                mode = o.get("mode", "dry_run")
                icon = "🟢" if mode == "live" else "🔵"
                st.write(f"{icon} **{o.get('action', '?')}** {o.get('amount', 0)} SOL | "
                        f"Mode: {mode} | {o.get('timestamp', '')[:19]}")
        else:
            st.info("No trades executed yet. Agents are building confidence...")

    # ======================== TAB 4: KNOWLEDGE BASE ========================
    with tab4:
        st.header("🧠 Knowledge Base")

        learner = brain.get("learner", {})
        col1, col2 = st.columns(2)
        col1.metric("Strategies Tracked", learner.get("strategies_tracked", 0))
        col2.metric("Lessons Learned", learner.get("lessons_count", 0))

        st.markdown("---")

        # Strategy performance
        st.subheader("📊 Strategy Performance History")
        kn_strategies = knowledge.get("strategies", {})
        if kn_strategies:
            rows = []
            for name, stats in kn_strategies.items():
                rows.append({
                    "Strategy": name,
                    "Tests": stats.get("total_tests", 0),
                    "Avg PnL": f"{stats.get('avg_pnl', 0):.4f}",
                    "Best PnL": f"{stats.get('best_pnl', 0):.4f}",
                    "Avg Win Rate": f"{stats.get('avg_win_rate', 0):.0%}",
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            # Bar chart of avg PnL
            chart_data = pd.DataFrame({
                "Strategy": list(kn_strategies.keys()),
                "Avg PnL": [s.get("avg_pnl", 0) for s in kn_strategies.values()],
            })
            st.bar_chart(chart_data.set_index("Strategy"))
        else:
            st.info("Knowledge base is empty. Brain needs a few cycles to gather data...")

        # Lessons
        st.subheader("📝 Lessons Learned")
        lessons = learner.get("lessons", [])
        if lessons:
            for lesson in lessons:
                st.write(f"- {lesson}")
        else:
            st.info("No lessons yet. The system learns as it accumulates more data...")

        # Market patterns
        patterns = knowledge.get("market_patterns", [])
        if patterns:
            st.subheader("📈 Optimizer Progress")
            df = pd.DataFrame(patterns)
            if "best_pnl" in df.columns and len(df) > 1:
                st.line_chart(df.set_index("generation")["best_pnl"])

    # ======================== TAB 5: TOKEN SCOUT ========================
    with tab5:
        st.header("🔍 Token Scout")

        wl = watchlist.get("tokens", [])
        st.caption(f"Last updated: {watchlist.get('updated', 'N/A')}")

        if wl:
            st.subheader(f"📋 Watchlist ({len(wl)} tokens)")
            rows = []
            for t in wl:
                rows.append({
                    "Symbol": t.get("symbol", "?"),
                    "Name": t.get("name", "?"),
                    "Price": f"${t.get('price', 0):.4f}" if t.get("price") else "N/A",
                    "Source": t.get("source", "?"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("Token Scout hasn't run yet...")

        # Scout info from brain
        scout = brain.get("scout", {})
        if scout.get("watchlist"):
            st.subheader("🎯 Active Watchlist")
            for sym in scout["watchlist"]:
                st.write(f"- {sym}")

    # ======================== FOOTER & AUTO-REFRESH ========================
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🤖 Eko - Multi-Agent Trading System")
    with col2:
        now = datetime.now().strftime("%H:%M:%S")
        st.caption(f"Dashboard refreshed: {now}")
    with col3:
        st.caption("Auto-refresh: 5s")

    # Auto-refresh every 5 seconds
    time.sleep(5)
    st.rerun()


# ============================================================================
# MAIN
# ============================================================================

_is_streamlit = False
try:
    import streamlit.runtime.scriptrunner
    _is_streamlit = True
except ImportError:
    try:
        from streamlit import runtime
        _is_streamlit = hasattr(runtime, 'exists') and runtime.exists()
    except Exception:
        pass

if _is_streamlit or (os.environ.get("STREAMLIT_SERVER_PORT") or "streamlit" in sys.argv[0] if sys.argv else False):
    create_dashboard()
elif __name__ == "__main__":
    create_dashboard()
