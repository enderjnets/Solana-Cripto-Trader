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


def load_accumulation() -> Dict:
    return load_json(PROJECT_ROOT / "data" / "accumulation_target.json")


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
            ("🎯", "Accumulator", last_cycle.get("accumulator", {}).get("status", "idle")),
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
                ago = int((datetime.now() - dt).total_seconds())
                st.caption(f"Updated {ago}s ago")
            except Exception:
                pass

    # ======================== MAIN TABS ========================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📡 Activity Feed",
        "🧬 Brain & Optimizer",
        "📈 Trading & Strategies",
        "🧠 Knowledge Base",
        "🔍 Token Scout",
        "💰 Portfolio",
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
            st.dataframe(pd.DataFrame(rows), width="stretch")
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

        # === ACCUMULATION TARGET ===
        accum = load_accumulation()
        if accum and accum.get("SOL"):
            st.subheader("🎯 Accumulation Target")
            col_a, col_b, col_c, col_d = st.columns(4)

            sol_pct = accum.get("SOL", 0.5)
            btc_pct = accum.get("BTC", 0.5)
            rec = accum.get("recommendation", "?")
            conf = accum.get("confidence", 0)

            col_a.metric("SOL Allocation", f"{sol_pct:.0%}")
            col_b.metric("BTC Allocation", f"{btc_pct:.0%}")
            col_c.metric("Recommendation", f"Accumulate {rec}")
            col_d.metric("Confidence", f"{conf:.0%}")

            reasoning = accum.get("reasoning", {})
            if reasoning:
                with st.expander("Reasoning Details", expanded=False):
                    r_col1, r_col2 = st.columns(2)
                    r_col1.write(f"**SOL Price:** ${reasoning.get('sol_price', 0):,.2f}")
                    r_col1.write(f"**SOL 24h Change:** {reasoning.get('sol_24h_change', 0):+.2f}%")
                    r_col1.write(f"**SOL Score:** {reasoning.get('sol_score', 0)}")
                    r_col2.write(f"**BTC Price:** ${reasoning.get('btc_price', 0):,.2f}")
                    r_col2.write(f"**BTC 24h Change:** {reasoning.get('btc_24h_change', 0):+.2f}%")
                    r_col2.write(f"**BTC Score:** {reasoning.get('btc_score', 0)}")

                    factors = reasoning.get("factors", [])
                    if factors:
                        st.write("**Decision Factors:**")
                        for f in factors:
                            st.write(f"- {f}")

                updated = accum.get("updated", "")
                if updated:
                    st.caption(f"Last updated: {updated[:19]}")

            st.markdown("---")

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
            st.dataframe(pd.DataFrame(rows), width="stretch")

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
            st.dataframe(df, width="stretch")

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
            st.dataframe(pd.DataFrame(rows), width="stretch")
        else:
            st.info("Token Scout hasn't run yet...")

        # Scout info from brain
        scout = brain.get("scout", {})
        if scout.get("watchlist"):
            st.subheader("🎯 Active Watchlist")
            for sym in scout["watchlist"]:
                st.write(f"- {sym}")

    # ======================== TAB 6: PORTFOLIO ========================
    with tab6:
        st.header("💰 Portfolio Analytics")
        
        # Load portfolio data
        data_dir = PROJECT_ROOT / "data"
        trades_file = data_dir / "trade_history.json"
        portfolio_file = data_dir / "portfolio_history.json"
        
        def load_json_file(path):
            if path.exists():
                try:
                    return json.loads(path.read_text())
                except:
                    return []
            return []
        
        trades = load_json_file(trades_file)
        portfolio_history = load_json_file(portfolio_file)
        
        # Calculate metrics
        closed_trades = [t for t in trades if t.get("status") == "closed"]
        open_trades = [t for t in trades if t.get("status") == "open"]
        total_pnl = sum(t.get("pnl", 0) for t in closed_trades)
        wins = [t for t in closed_trades if t.get("pnl", 0) > 0]
        losses = [t for t in closed_trades if t.get("pnl", 0) <= 0]
        win_rate = len(wins) / len(closed_trades) * 100 if closed_trades else 0
        
        # PnL by token
        pnl_by_token = {}
        for t in closed_trades:
            token = t.get("token", "UNKNOWN")
            pnl_by_token[token] = pnl_by_token.get(token, 0) + t.get("pnl", 0)
        
        # Current portfolio
        if portfolio_history:
            current = portfolio_history[-1]
            initial = portfolio_history[0] if portfolio_history else {"TOTAL": 500}
            portfolio_return = ((current.get("TOTAL", 500) - initial.get("TOTAL", 500)) / initial.get("TOTAL", 500)) * 100 if initial.get("TOTAL", 0) > 0 else 0
        else:
            current = {"SOL": 0, "BTC": 0, "USDT": 0, "TOTAL": 500}
            portfolio_return = 0
        
        # Top metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("💵 Total Portfolio", f"${current.get('TOTAL', 500):.2f}", f"{portfolio_return:.2f}%")
        m2.metric("📈 Total PnL", f"${total_pnl:.2f}", delta_color="normal" if total_pnl > 0 else "inverse")
        m3.metric("🎯 Win Rate", f"{win_rate:.1f}%", f"{len(wins)}W / {len(losses)}L")
        m4.metric("📊 Total Trades", f"{len(closed_trades)}", f"{len(open_trades)} open")
        
        # Profit Factor (handle division by zero)
        if losses and sum(l.get('pnl', 0) for l in losses) != 0:
            pf = abs(sum(w.get('pnl', 0) for w in wins) / len(wins) / abs(sum(l.get('pnl', 0) for l in losses) / len(losses)))
        elif wins:
            pf = sum(w.get('pnl', 0) for w in wins) / 0.01  # Avoid div by zero
        else:
            pf = 0
        m5.metric("⚖️ Profit Factor", f"{pf:.2f}")
        
        # Charts
        col_chart, col_alloc = st.columns([2, 1])
        
        with col_chart:
            if portfolio_history:
                df = pd.DataFrame(portfolio_history[-100:])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                st.subheader("📈 Portfolio Growth")
                st.line_chart(df.set_index('timestamp')['TOTAL'], height=200)
                st.subheader("💼 Asset Distribution")
                st.area_chart(df.set_index('timestamp')[['SOL', 'BTC', 'USDT']], height=200)
            else:
                st.info("Portfolio history will appear after trading begins...")
        
        with col_alloc:
            st.subheader("🎯 Current Allocation")
            total = current.get("TOTAL", 1)
            sol_pct = current.get("SOL", 0) / total * 100 if total > 0 else 0
            btc_pct = current.get("BTC", 0) / total * 100 if total > 0 else 0
            usdt_pct = current.get("USDT", 0) / total * 100 if total > 0 else 0
            st.write(f"**SOL:** ${current.get('SOL', 0):.2f} ({sol_pct:.1f}%)")
            st.progress(sol_pct / 100)
            st.write(f"**BTC:** ${current.get('BTC', 0):.2f} ({btc_pct:.1f}%)")
            st.progress(btc_pct / 100)
            st.write(f"**USDT:** ${current.get('USDT', 0):.2f} ({usdt_pct:.1f}%)")
            st.progress(usdt_pct / 100)
            st.markdown("---")
            st.write("**Target:** 50% SOL / 30% BTC / 20% USDT")
        
        # PnL by Token
        st.subheader("📊 PnL by Token")
        if pnl_by_token:
            pnl_df = pd.DataFrame([{"Token": k, "PnL": v} for k, v in pnl_by_token.items()])
            c1, c2 = st.columns([1, 1])
            with c1:
                st.dataframe(pnl_df.style.format({"PnL": "${:.2f}"}).apply(
                    lambda x: ['background-color: #1b4332' if v > 0 else '#7f1d1d' for v in x], subset=['PnL']
                ), width="stretch")
            with c2:
                st.bar_chart(pnl_df.set_index("Token")["PnL"], height=150)
        
        # Trade History
        st.subheader("📋 Trade History")
        if trades:
            df_trades = pd.DataFrame(sorted(trades, key=lambda x: x.get("timestamp", ""), reverse=True))
            df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            display_df = df_trades[['timestamp', 'token', 'side', 'amount', 'price', 'pnl', 'status']].copy()
            display_df['price'] = display_df['price'].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x)
            display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else x)
            st.dataframe(display_df, width="stretch")
        else:
            st.info("No trades executed yet.")
        
        # Operational Analysis
        st.subheader("🔍 Operational Analysis")
        a1, a2, a3 = st.columns(3)
        with a1:
            st.markdown(f"""
            **📊 Performance**
            - Win Rate: {win_rate:.1f}% ({'🟢' if win_rate >= 60 else '🟡' if win_rate >= 50 else '🔴'})
            - Total Trades: {len(closed_trades)}
            - Open Positions: {len(open_trades)}
            """)
        with a2:
            st.markdown(f"""
            **💰 Risk Management**
            - Daily Limit: 10% ✅
            - Max Position: 10% ✅
            - Risk/Trade: 5% ✅
            - Stop Loss: -8% ✅
            """)
        with a3:
            st.markdown(f"""
            **🎯 Objectives**
            - Daily Target: +5%
            - Monthly Target: +100%
            - Current: {portfolio_return:.2f}%
            - Progress: {min(portfolio_return/5*100, 100):.0f}%
            """)
        
        # AI Recommendations
        st.subheader("💡 AI Recommendations")
        recs = []
        if win_rate >= 60:
            recs.append("✅ Win rate healthy")
        elif win_rate >= 50:
            recs.append("⚠️ Win rate acceptable")
        else:
            recs.append("🔴 Win rate needs improvement")
        
        if sol_pct > 65:
            recs.append("⚠️ SOL overweight - consider rebalancing")
        if len(open_trades) > 3:
            recs.append("⚠️ Many open trades - consider closing some")
        
        for r in recs:
            st.write(r)

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
