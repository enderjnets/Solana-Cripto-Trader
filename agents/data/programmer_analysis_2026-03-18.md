# Programmer Agent Analysis: Solana Bot Win Rate 0.3%
**Date:** 2026-03-18
**Author:** Programmer Agent (Eko Team)
**Status:** CRITICAL BUGS IDENTIFIED

---

## Executive Summary

The 0.3% win rate (2W/581L) is **NOT real**. The bot has only made **3 real trades** in its lifetime. The other **580 trades are "ghost trades"** — positions opened and immediately closed within milliseconds by an emergency close loop, each counting as a loss with $0 P&L.

**Real performance:** 2 wins, 1 loss = 66.7% win rate, +$20.24 realized P&L.

---

## 1. Analysis of the 2 Winning Trades

| # | Symbol | Direction | Strategy | Entry | Exit | P&L USD | P&L % | Duration | Close Reason |
|---|--------|-----------|----------|-------|------|---------|-------|----------|--------------|
| 1 | GOAT | LONG | trend_momentum | $0.01870 | $0.01942 | +$12.38 | +9.91% | ~4 days | RISK_AGENT_DECISION (TP near, R/R 0.18x) |
| 2 | SOL | LONG | macd_cross | $86.48 | $90.31 | +$14.57 | +11.66% | ~4 days | RISK_AGENT_DECISION (TP near, R/R 0.08x) |

**What worked:**
- Both were LONG positions opened during an uptrend
- Both used 3x leverage with $125 margin ($375 notional)
- Both were closed by the Risk Agent's intelligent decision system (close when near TP with bad remaining R/R)
- Combined P&L: **+$26.95**

**The 1 real loss:**
- BTC SHORT via macd_cross: -$6.71 (manual close near SL on Mar 14 audit)

**Net real P&L:** +$26.95 - $6.71 = **+$20.24** (matches the equity gain)

---

## 2. Root Cause: The Emergency Close Loop (580 Ghost Trades)

### What's happening

Every 60 seconds, the orchestrator runs this cycle:
```
Step 2: Risk Manager → generates risk_report.json (may set emergency_close)
Step 3: Strategy    → generates signals (almost always has signals)
Step 4: Executor    → closes emergency positions, then OPENS new ones
Step 4c: Daily Target → immediately closes the newly opened positions
Step 4d: Position Decisions → another chance to close them
```

**The loop:**
1. Executor opens 3 positions (Step 4)
2. Daily Target or Position Decisions immediately closes them (Steps 4c/4d) using `close_positions_emergency()`
3. Positions had 0ms of price movement → `entry_price == exit_price` → `pnl_usd = 0`
4. `pnl_usd = 0` is counted as a **LOSS** (because `0 > 0` is `False`)
5. Next cycle: same thing repeats

**Evidence:** 216 batches of emergency closes, averaging 2.7 trades per batch, running every ~60-120 seconds from Mar 16 22:51 through Mar 18 21:30. All 580 have `entry_price == exit_price` and `pnl_usd = 0.0`.

---

## 3. Top 3 Critical Bugs

### BUG 1: Open-Then-Immediately-Close Loop (CRITICAL)
**File:** `orchestrator.py:94-99` and `orchestrator.py:162-175`
**Problem:** The orchestrator opens positions in Step 4 (Executor), but Steps 4c (Daily Target) and 4d (Position Decisions) can immediately close them in the **same cycle**, before any price movement occurs.
**Impact:** 580 phantom trades, all counted as losses.

### BUG 2: $0 P&L Counted as Loss (HIGH)
**File:** `executor.py:218-219`
**Problem:**
```python
if pnl_usd > 0:
    portfolio["wins"] += 1
else:
    portfolio["losses"] += 1  # $0 PnL = LOSS
```
A trade with exactly $0 P&L (entry == exit) is counted as a loss. These should either be excluded from win/loss stats or counted as breakeven.
**Impact:** Inflates loss count by 580.

### BUG 3: No Cooldown After Emergency Close (HIGH)
**File:** `executor.py:578-726` (the `run()` function)
**Problem:** After closing positions via emergency, the executor immediately opens new ones in the same cycle. The emergency condition persists in risk_report until market conditions change, so next cycle the new positions get emergency-closed too.
**Impact:** Creates infinite open/close loop every 60 seconds.

---

## 4. Proposed Fixes

### Fix 1: Prevent opening positions in the same cycle as emergency close

**File:** `executor.py`, in the `run()` function (~line 598-601):

```python
# CURRENT CODE (buggy):
emergency_closed = close_positions_emergency(portfolio, symbols_to_close, market, history)
if emergency_closed:
    log.error(f"🚨 {len(emergency_closed)} posiciones cerradas por emergencia")
# ... later opens new positions anyway

# FIXED CODE:
emergency_closed = close_positions_emergency(portfolio, symbols_to_close, market, history)
if emergency_closed:
    log.error(f"🚨 {len(emergency_closed)} posiciones cerradas por emergencia")
    log.warning("⛔ Skipping new positions this cycle (cooldown after emergency)")
    save_portfolio(portfolio)
    save_history(history)
    return {
        "status": "EMERGENCY_COOLDOWN",
        "capital": portfolio["capital_usd"],
        "opened": 0,
        "closed": len(emergency_closed),
    }
```

### Fix 2: Don't count $0 PnL trades as losses

**File:** `executor.py`, in `close_positions_emergency()` (~line 217-219):

```python
# CURRENT CODE (buggy):
if pnl_usd > 0:
    portfolio["wins"] = portfolio.get("wins", 0) + 1
else:
    portfolio["losses"] = portfolio.get("losses", 0) + 1

# FIXED CODE:
if pnl_usd > 0.001:  # Meaningful win (> $0.001)
    portfolio["wins"] = portfolio.get("wins", 0) + 1
elif pnl_usd < -0.001:  # Meaningful loss
    portfolio["losses"] = portfolio.get("losses", 0) + 1
# else: breakeven — don't count in stats
```

Same fix needed in `paper_update_positions()` (~line 531-537):
```python
# CURRENT:
net_pnl = pnl_usd - fee_exit
is_win = net_pnl > 0

# FIXED:
net_pnl = pnl_usd - fee_exit
if abs(net_pnl) < 0.001:
    pass  # Breakeven, don't count
elif net_pnl > 0:
    portfolio["wins"] = portfolio.get("wins", 0) + 1
else:
    portfolio["losses"] = portfolio.get("losses", 0) + 1
```

### Fix 3: Prevent orchestrator from closing newly opened positions

**File:** `orchestrator.py`, add a guard in Steps 4c and 4d:

```python
# After Step 4 (executor), track what was just opened:
newly_opened_symbols = [p["symbol"] for p in result.get("opened_positions", [])] if isinstance(result, dict) else []

# In Step 4c (Daily Target), skip newly opened:
if target_result["should_close_all"]:
    open_symbols = [
        p["symbol"] for p in portfolio_data.get("positions", [])
        if p.get("status") == "open" and p["symbol"] not in newly_opened_symbols
    ]

# Same guard in Step 4d (Position Decisions)
```

### Fix 4 (Bonus): Reset the corrupted stats

```python
# One-time cleanup script to fix portfolio stats
import json
portfolio = json.load(open("agents/data/portfolio.json"))
history = json.load(open("agents/data/trade_history.json"))

# Recalculate from actual trade history
real_trades = [t for t in history if abs(t.get("pnl_usd", 0)) > 0.001]
portfolio["total_trades"] = len(real_trades)
portfolio["wins"] = len([t for t in real_trades if t["pnl_usd"] > 0])
portfolio["losses"] = len([t for t in real_trades if t["pnl_usd"] < 0])
json.dump(portfolio, open("agents/data/portfolio.json", "w"), indent=2)
print(f"Fixed: {portfolio['total_trades']} trades, {portfolio['wins']}W/{portfolio['losses']}L")
```

---

## 5. Impact Estimation

| Metric | Current (Broken) | After Fix |
|--------|-----------------|-----------|
| Total Trades | 583 | 3 (real) |
| Wins | 2 | 2 |
| Losses | 581 | 1 |
| Win Rate | 0.3% | **66.7%** |
| Equity | $520.24 | $520.24 (unchanged) |
| Ghost trades/day | ~290 | **0** |

**The bot is actually profitable.** The strategy (trend_momentum + macd_cross with 3x leverage) works well. The problem is purely in the execution loop creating phantom trades.

After fixing:
- Win rate jumps from 0.3% to 66.7% (reflecting reality)
- No more ghost trades burning CPU and inflating history
- Risk management decisions (Daily Target, Position Decisions) will only evaluate positions that have had time to develop

---

## 6. Priority Order

1. **Fix 1 (Emergency cooldown)** — stops the bleeding immediately
2. **Fix 2 ($0 PnL counting)** — prevents future stat corruption
3. **Fix 3 (Orchestrator guard)** — defense in depth
4. **Fix 4 (Stats reset)** — clean up historical data

Estimated time to implement all fixes: ~30 minutes of coding.

---

*Analysis completed by Programmer Agent, 2026-03-18*
