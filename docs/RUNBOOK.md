# Solana Live Bot — Runbook

**Last updated**: 2026-04-22 · **Applies to**: v2.12.22-live+

Operational playbook for incident response on the Solana live trading bot running at PC ROG (`10.0.0.240`), workspace `/home/enderj/.openclaw/workspace/Solana-Cripto-Trader-Live`.

---

## Quick Status Check

| Resource | URL / Command |
|---|---|
| Dashboard | http://100.88.47.99:8082 |
| Health JSON | http://100.88.47.99:8082/api/health |
| Paperclip SOLAAA issues | http://100.88.47.99:3100/SOLAAA/dashboard |
| GitHub (branch `live`) | https://github.com/enderjnets/Solana-Cripto-Trader |
| SSH to ROG | `ssh enderj@10.0.0.240` |
| Wallet on Solscan | https://solscan.io/account/EEmtkySNz1SLNZBMBu6EsuqkEhttEKjejsEXdEFT2fMH |

### Single-command health probe
```bash
ssh enderj@10.0.0.240 'curl -sS http://localhost:8082/api/health | python3 -m json.tool'
```

---

## Auto-monitoring (already active — you don't have to watch manually)

- **Watchdog** (bash process) restarts orch if it dies
- **Cron `*/5 * * * * tools/health_check.py --alert`** detects unhealthy state, creates Paperclip SOLAAA issue (urgent, dedup 15min)
- **`/api/health`** endpoint exposes real-time state
- **Dashboard badge** polls health every 30s (green=healthy, red=alert)

If cron alert fires → issue appears in SOLAAA with runbook note. Just follow that note (or this doc).

---

## Incidents by type

### 1. Kill switch active

**Symptom**: `/api/health.kill_switch.active=true` OR watchdog log shows `🛑 KILL SWITCH ACTIVO` repeating

**Common causes**:
- `reconcile_critical_N_discrepancies` — portfolio vs wallet mismatch
- `daily_loss_exceeded_X` — max daily loss tripped
- `emergency_close: N consecutive fails` — close_failures >= 3
- manual toggle via dashboard button

**Fix**:
```bash
# Inspect reason
cat /tmp/solana_live_killswitch

# If reconcile discrepancy → run reconcile
cd ~/.openclaw/workspace/Solana-Cripto-Trader-Live
python3 tools/reconcile_all_orphans.py --dry-run    # audit
python3 tools/reconcile_all_orphans.py              # live (after audit ok)
# tool auto-clears /tmp/solana_live_killswitch post-reconcile

# If daily_loss / max fails → manually clear + investigate root cause
rm /tmp/solana_live_killswitch
# then restart orch (see section 5)
```

### 2. Orphan positions (portfolio says N open, wallet disagrees)

**Symptom**: `reconcile_critical_N_discrepancies` in log; positions marked open but wallet has different balances

**Fix** (uses `tools/reconcile_all_orphans.py` — multi-orphan safe, with backups):
```bash
cd ~/.openclaw/workspace/Solana-Cripto-Trader-Live
python3 tools/reconcile_all_orphans.py --dry-run    # preview what would change
python3 tools/reconcile_all_orphans.py              # apply (auto-backup to .bak_<epoch>)
```

The tool:
- Compares each open position against wallet balance + recent closing txs
- Reconciles orphans (matches close tx → pnl → marks closed in portfolio)
- Skips positions that are NOT orphan (preserves)
- Clears kill switch, stale HB, lock, close_failures
- Resets `wild_mode_state` + `daily_target_state` baselines

### 3. Close failing — Jupiter error 6024 (slippage exceeded)

**Symptom**: log shows `broadcast_failed` with `Custom(6024)` / `0x1788`; `LIVE EMERGENCY CLOSE FAILED <SYMBOL>`

**Fix**: check per-symbol slippage config; bridged/thin-liquidity tokens need higher bps.
```bash
# Check current per-symbol config
grep SLIPPAGE_BPS .env

# Bump if needed (ej. ETH Wormhole needs >= 800bps)
sed -i 's/SLIPPAGE_BPS_ETH=.*/SLIPPAGE_BPS_ETH=1200/' .env

# Restart watchdog to pick up new env (see section 5)
```

The retry ladder (v2.12.17) auto-escalates from base → 2x → 8x (capped at 2000bps).

### 4. Heartbeat stale (> 5 min)

**Symptom**: `/api/health.heartbeat.age_s > 300` OR health_check reports HB stale

**Fix**:
```bash
# 1. Check if orch is actually alive
for p in $(pgrep -f orchestrator.py); do
  CWD=$(readlink /proc/$p/cwd 2>/dev/null)
  [ "$CWD" = "/home/enderj/.openclaw/workspace/Solana-Cripto-Trader-Live" ] && echo "alive: $p"
done

# 2. If alive but HB stale → orch is stuck in a cycle (probably LLM call hung)
kill -TERM <orch_pid>
# Watchdog respawns in 10-30s

# 3. If orch not found → see section 5 (watchdog also dead)
```

### 5. Watchdog + orch both dead

**Fix** — relaunch watchdog with `.env` sourced (important: env vars like SLIPPAGE_BPS_ETH, PAPERCLIP_API_KEY must be in watchdog's env so orch inherits):
```bash
cd ~/.openclaw/workspace/Solana-Cripto-Trader-Live
(set -a; source .env; set +a; nohup bash run_watchdog.sh > /tmp/solana_live_watchdog.log 2>&1 & disown)
sleep 15
# Verify
for p in $(pgrep -f orchestrator.py); do
  CWD=$(readlink /proc/$p/cwd 2>/dev/null)
  [ "$CWD" = "/home/enderj/.openclaw/workspace/Solana-Cripto-Trader-Live" ] && echo "orch PID: $p"
done
```

### 6. Dust on-chain (JUP/ETH residue after trades)

**Symptom**: wallet has small amounts of tokens that aren't in any open position

**Fix** — use `tools/sweep_token.py`:
```bash
cd ~/.openclaw/workspace/Solana-Cripto-Trader-Live
# Example: sweep 0.107 JUP to USDC with 5% slippage tolerance
python3 tools/sweep_token.py --symbol JUP --amount 0.107032 --slippage-bps 500 --priority veryHigh

# For small SOL excess (keep 0.022 fuel baseline):
python3 tools/sweep_token.py --symbol SOL --amount 0.017 --slippage-bps 100 --priority veryHigh
```

### 7. Capital change / scale

**Use case**: user deposits USDC to wallet; need to sync portfolio + env

**Pattern** (see commit v2.12.20-live for reference `scale_to_100.py`):
```bash
# 1. Confirm wallet balance
curl -sS http://localhost:8082/api/health | python3 -m json.tool

# 2. Kill watchdog + orch
kill -TERM <watchdog_pid> <orch_pid>

# 3. Update portfolio.json (capital_usd + initial_capital → new baseline)
# + daily_target_state (reset)
# + wild_mode_state (reset)
# + .env LIVE_MAX_POSITION_USD (proportional)

# 4. Relaunch watchdog (section 5)
```

---

## Rollback procedure

If a deploy introduces a regression:
```bash
cd ~/.openclaw/workspace/Solana-Cripto-Trader-Live
git log --oneline -5                    # find bad commit
git revert <hash> --no-edit             # creates revert commit
git push origin live

# Restart watchdog
kill -TERM <watchdog_pid> <orch_pid>
# relaunch per section 5
```

Portfolio/history restore from auto-backups:
```bash
ls agents/data/portfolio.json.bak_*      # find timestamp
cp agents/data/portfolio.json.bak_<epoch> agents/data/portfolio.json
cp agents/data/trade_history.json.bak_<epoch> agents/data/trade_history.json
# relaunch watchdog
```

---

## Useful one-liners

```bash
# Current state
curl -sS http://localhost:8082/api/health | python3 -m json.tool

# Weekly report (paper vs live)
python3 tools/weekly_report.py --days 7 --format table

# Gate check (Phase 1 → Phase 2)
python3 tools/weekly_report.py --days 14 --format gate-check

# Manual health check (verbose)
python3 tools/health_check.py --verbose

# Force-close all open positions (emergency)
python3 tools/emergency_close.py --force

# Recent errors from watchdog log
grep -aE "ERROR|EMERGENCY|KILL SWITCH" /tmp/solana_live_watchdog.log | tail -20
```

---

## Infrastructure reference

- **Live bot workspace**: `/home/enderj/.openclaw/workspace/Solana-Cripto-Trader-Live` (git branch `live`)
- **Paper bot workspace**: `/home/enderj/.openclaw/workspace/Solana-Cripto-Trader` (git branch `master`)
- **Drift branch**: `origin/drift-integration` (not merged to live; ver v2.12.21 commit para status)
- **Wallet file**: `/home/enderj/.config/solana-jupiter-bot/wallet.json` (Jupiter custom dict format)
- **Wallet pubkey**: `EEmtkySNz1SLNZBMBu6EsuqkEhttEKjejsEXdEFT2fMH`
- **State files**: all in `agents/data/`
- **Log file**: `/tmp/solana_live_watchdog.log` (truncated to 20MB by watchdog at 100MB threshold)
- **Flag files**: `/tmp/solana_live_{heartbeat,killswitch,close_failures,orchestrator.lock}`

## Paperclip company map (live vs paper)

| Prefix | Name | UUID | Use |
|---|---|---|---|
| SOLAAA | Solana Cripto Trading Live | `2b0a3d7d-4929-42ef-9602-86324c0a202a` | **Live issues (current)** |
| SOLAA | Solana Cripto Trader | `782b926b-4fb7-424a-a881-f368b0f79e3c` | Paper (legacy SOLAA-115/116 migrated) |
| BIT | BitTrader YouTube | `027cbdd7-2e13-4e95-a608-10661bf49b2b` | BitTrader project |
| SOL, SOLA | (archived) | — | legacy |

Set `PAPERCLIP_API_KEY=pcp_board_...` (board key) for cross-company access. Agent keys are scoped to one company.

---

## Contact points

- Source of truth: commits on `origin/live` at https://github.com/enderjnets/Solana-Cripto-Trader
- Active alerts: SOLAAA dashboard
- Weekly review: run `tools/weekly_report.py --format gate-check`

## Versioning

Follow `feedback_solana_commit_trio.md`: every feature/fix = code + `VERSION` bump in `dashboard/app.py` + changelog entry + Paperclip issue (SOLAAA `status=done`) in the same commit flow.
