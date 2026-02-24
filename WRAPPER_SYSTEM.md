# 🛡️ Trading Bot Wrapper System

**Created:** 2026-02-24
**Purpose:** Auto-restart, notifications, and crash prevention for the trading bot

## 📋 Overview

The wrapper system provides three layers of protection:

1. **Error Handling in Code** - Catches exceptions and continues running
2. **Auto-Restart Wrapper** - Restarts bot if it crashes
3. **Telegram Notifications** - Alerts you when something happens

---

## 🚀 Usage

### Start the Wrapper (Recommended)

```bash
cd /home/enderj/.openclaw/workspace/solana-jupiter-bot
./trading_wrapper.sh
```

### Start the Bot Directly (Without Wrapper)

```bash
cd /home/enderj/.openclaw/workspace/solana-jupiter-bot
python3 unified_trading_system.py --continuous
```

---

## 🎯 Features Implemented

### 1. Error Handling in Code

**File:** `unified_trading_system.py`

**Features:**
- ✅ Try/catch blocks around trading cycles
- ✅ Graceful shutdown on SIGINT (Ctrl+C) and SIGTERM
- ✅ Maximum consecutive errors (10) before stopping
- ✅ Detailed error logging
- ✅ Wait 30s before retry after error
- ✅ Notifications on critical errors (3+ consecutive)

**Code Example:**
```python
while self.running:
    try:
        self.run_cycle()
        cycle_error_count = 0  # Reset on success
    except Exception as e:
        cycle_error_count += 1
        logger.error(f"❌ Error in trading cycle ({cycle_error_count}/10): {e}")

        if cycle_error_count >= 3:
            self.notifier.system_error(str(e), f"Cycle {cycle_error_count} errors")

        if cycle_error_count >= 10:
            logger.error("❌ Too many consecutive errors, stopping...")
            self.running = False
            break
```

---

### 2. Auto-Restart Wrapper

**File:** `trading_wrapper.sh`

**Features:**
- ✅ Auto-restart bot if it crashes
- ✅ Restart rate limiting (5 restarts per 5 minutes max)
- ✅ Prevents infinite crash loops
- ✅ Graceful shutdown handling (SIGINT/SIGTERM)
- ✅ Detailed logging to `/tmp/trading_wrapper.log`
- ✅ Notifications when bot crashes or restarts
- ✅ Proper cleanup of PID files and processes

**Configuration:**
```bash
MAX_RESTARTS=5          # Max restarts in time window
RESTART_WINDOW=300      # Time window in seconds (5 minutes)
```

**How it works:**
1. Starts the bot with `--continuous` mode
2. Waits for the bot to exit
3. If it crashes, checks restart rate limit
4. If under limit, waits 10s and restarts
5. If over limit, stops wrapper and sends alert

---

### 3. Telegram Notifications

**File:** `notifications.py`

**New Notification Types:**

| Notification | Priority | Trigger |
|-------------|----------|----------|
| 🚀 System Started | Normal | Bot starts |
| 🛑 System Stopped | **HIGH** | Bot stops (any reason) |
| ❌ System Error | **HIGH** | 3+ consecutive cycle errors |
| ⚠️ Crash Alert | **HIGH** | Bot crashes and wrapper restarts |

**Example Notifications:**

```
🚀 SISTEMA DE TRADING INICIADO
📊 Modo: DAY TRADING
⏰ 2026-02-24 06:32:40

---

⚠️ TRADING BOT ALERT: Too many crashes!
Stopped to prevent infinite loop.

---

⚠️ Trading bot crashed (exit code: 1)
Restart #3...
```

---

## 📊 Monitoring

### Check Wrapper Status

```bash
# See wrapper logs
tail -f /tmp/trading_wrapper.log

# Check if wrapper is running
ps aux | grep trading_wrapper

# Check wrapper output (if any)
tail -f /tmp/trading_wrapper_output.log
```

### Check Bot Status

```bash
# See bot logs
tail -f /tmp/trading_bot.log

# Check if bot is running
ps aux | grep unified_trading_system

# Check paper trading status
cd /home/enderj/.openclaw/workspace/solana-jupiter-bot
python3 unified_trading_system.py --paper-status
```

### Check Notifications

```bash
# See recent notifications
tail -f /home/enderj/.openclaw/workspace/solana-jupiter-bot/data/notifications.log
```

---

## 🛑 Stopping the Bot Safely

### Method 1: Graceful Shutdown (Recommended)

```bash
# Send SIGINT to wrapper (graceful)
kill -SIGINT $(cat /tmp/trading_wrapper.pid)

# Or Ctrl+C if running in terminal
```

### Method 2: Direct Kill

```bash
# Kill wrapper (will stop bot gracefully)
pkill -f trading_wrapper

# Force kill if needed
pkill -9 -f trading_wrapper
```

### Method 3: Stop Bot Only

```bash
# Let wrapper know to stop
python3 unified_trading_system.py --stop

# Or kill the bot process only
pkill -f "unified_trading_system.py --continuous"
```

---

## 🔧 Troubleshooting

### Bot Keeps Crashing

1. **Check error logs:**
   ```bash
   tail -100 /tmp/trading_bot.log | grep ERROR
   ```

2. **Check restart rate:**
   ```bash
   tail -20 /tmp/trading_wrapper.log | grep "restart\|crash"
   ```

3. **Disable wrapper temporarily:**
   ```bash
   pkill -f trading_wrapper
   python3 unified_trading_system.py --continuous
   ```

### Wrapper Won't Start

1. **Check permissions:**
   ```bash
   chmod +x trading_wrapper.sh
   ```

2. **Check for syntax errors:**
   ```bash
   bash -n trading_wrapper.sh
   ```

3. **Run with debug:**
   ```bash
   bash -x trading_wrapper.sh
   ```

### Not Getting Notifications

1. **Check notification log:**
   ```bash
   tail -50 data/notifications.log
   ```

2. **Verify notification system:**
   ```bash
   python3 -c "from notifications import get_notifier; n = get_notifier(); n.system_started('TEST')"
   tail data/notifications.log
   ```

---

## 📈 Architecture

```
┌─────────────────────────────────────────────────────────┐
│              TRADING WRAPPER (bash)                     │
│  • Auto-restart on crash                               │
│  • Rate limiting (5/5min)                              │
│  • Graceful shutdown                                   │
│  • Crash notifications                                 │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│          UNIFIED TRADING SYSTEM (Python)                │
│  • Signal handling (SIGINT/SIGTERM)                    │
│  • Error catching in cycles                           │
│  • Max consecutive errors (10)                        │
│  • System start/stop notifications                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              NOTIFICATION SYSTEM                        │
│  • Logs to data/notifications.log                      │
│  • OpenClaw reads and sends via Telegram              │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 Logs Summary

| Log File | Purpose | Location |
|----------|---------|----------|
| `trading_wrapper.log` | Wrapper activity | `/tmp/trading_wrapper.log` |
| `trading_bot.log` | Bot main output | `/tmp/trading_bot.log` |
| `trading_wrapper_output.log` | Wrapper stdout/stderr | `/tmp/trading_wrapper_output.log` |
| `notifications.log` | Telegram notifications | `data/notifications.log` |

---

## ✅ Testing

### Test Error Handling

```bash
# Start wrapper
./trading_wrapper.sh

# In another terminal, force a crash
kill -9 $(pgrep -f unified_trading_system)

# Watch wrapper restart it
tail -f /tmp/trading_wrapper.log
```

### Test Notifications

```bash
# Test start notification
python3 -c "from notifications import get_notifier; get_notifier().system_started('TEST')"

# Test error notification
python3 -c "from notifications import get_notifier; get_notifier().system_error('Test error', 'Test context')"

# View notifications
tail data/notifications.log
```

---

## 🎓 Best Practices

1. **Always use the wrapper** - It provides auto-restart and crash protection
2. **Monitor notifications** - They alert you to issues early
3. **Check logs periodically** - Look for patterns in errors
4. **Don't ignore consecutive errors** - They indicate a real problem
5. **Use graceful shutdown** - Prefer SIGINT over SIGKILL
6. **Keep wrapper running** - Start it with systemd or tmux for persistence

---

## 🔮 Future Enhancements

- [ ] Systemd service for auto-start on boot
- [ ] Health check endpoint (HTTP)
- [ ] Metrics dashboard (Prometheus/Grafana)
- [ ] Auto-recovery strategies (config changes on repeated errors)
- [ ] Email notifications (fallback if Telegram fails)

---

**Author:** Eko (EkoBit)
**Date:** 2026-02-24
**Version:** 1.0
