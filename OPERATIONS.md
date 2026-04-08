# Solana Jupiter Bot - Operations Guide

## Supported Start Path

**The ONLY supported way to manage the bot is via systemd:**

```bash
# Start
sudo systemctl start solana-jupiter-bot.service

# Stop
sudo systemctl stop solana-jupiter-bot.service

# View status
sudo systemctl status solana-jupiter-bot.service

# View logs
sudo journalctl -u solana-jupiter-bot.service -f
```

**Do NOT run `run_watchdog.sh` manually.** If the bot is already running under systemd, manual invocation will exit with a message directing you to use systemctl.

---

## Recovery Commands

### Watchdog Ownership Conflict (restart loop)

**Symptom:** `systemctl status` shows restart counter growing rapidly (9212+).

**Cause:** A manual watchdog process is holding `/tmp/solana_modular_orchestrator.lock` while the systemd service also tries to start.

**Fix:**
```bash
# 1. Find the conflicting watchdog process
ps aux | grep run_watchdog

# 2. Kill it (replace PID with actual)
sudo kill <PID>

# 3. Verify service is stable
sudo systemctl status solana-jupiter-bot.service
sleep 10
sudo systemctl status solana-jupiter-bot.service  # confirm no new restarts

# 4. Run validation
bash /home/enderj/.openclaw/workspace/Solana-Cripto-Trader/validate_watchdog_ownership.sh
```

### Service in Failed State (StartLimitHit)

**Symptom:** `systemctl status` shows `StartLimitBurst` was hit.

**Fix:**
```bash
# Reset the failure state
sudo systemctl reset-failed solana-jupiter-bot.service

# Start the service
sudo systemctl start solana-jupiter-bot.service
```

### Service Won't Start (lock file stale)

**Symptom:** Lock file exists but no watchdog process is running.

**Fix:**
```bash
# Remove stale lock
sudo rm /tmp/solana_modular_orchestrator.lock
sudo rm /tmp/solana_watchdog_restart_marker

# Restart service
sudo systemctl restart solana-jupiter-bot.service
```

### Full Reboot Recovery

After a server reboot, simply:
```bash
sudo systemctl enable solana-jupiter-bot.service
sudo systemctl start solana-jupiter-bot.service
```

The service is configured to start automatically on boot (`WantedBy=multi-user.target`).

---

## Validation

### Run Ownership Validation

```bash
bash /home/enderj/.openclaw/workspace/Solana-Cripto-Trader/validate_watchdog_ownership.sh
```

Exit code 0 = OK (single watchdog owner). Exit code 1 = conflict detected.

### Check Restart Counter

```bash
systemctl show solana-jupiter-bot.service -p NRestarts --value
```

If > 5 within a short period, a watchdog conflict exists.

### View Service Logs

```bash
# Recent logs
sudo journalctl -u solana-jupiter-bot.service -n 50

# Follow in real-time
sudo journalctl -u solana-jupiter-bot.service -f

# Service file logs
tail -f /home/enderj/.config/solana-jupiter-bot/jupiter.service.log
```

---

## Architecture

```
systemd (solana-jupiter-bot.service)
└── run_watchdog.sh (watchdog supervisor)
    ├── chat_agent.py (background - Telegram/chat interface)
    └── agents/orchestrator.py (trading orchestrator)
```

Lock file: `/tmp/solana_modular_orchestrator.lock`

The watchdog uses this lock to ensure exactly one instance runs. When systemd starts the service, it becomes the sole watchdog supervisor. Any manual invocation of `run_watchdog.sh` while the service is running will exit cleanly (not create a conflict) but will be redirected to systemctl guidance.

---

## Alert Suppression

The fix for the restart alert storm (SOLAA-20):
1. `run_watchdog.sh` now exits with code 1 (not 0) when another watchdog holds the lock
2. Systemd service uses `Restart=on-failure` with `StartLimitBurst=10`
3. Restart loop detection in watchdog exits after 60s of continuous restart attempts
4. Validation script can detect and report conflicts before they cause alert storms
