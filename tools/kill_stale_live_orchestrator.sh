#!/bin/bash
# Anti-hang guard para Solana-Cripto-Trader-Live.
# Si el heartbeat está stale >STALE_SEC y hay un proceso orchestrator vivo,
# lo mata con SIGKILL para que run_watchdog_live.sh lo resucite (5s ciclo).
# Invocado desde cron cada 2 min.
# Log: /tmp/solana_live_killguard.log
set -u
STALE_SEC="${STALE_SEC:-600}"   # 10 min
HB=/tmp/solana_live_heartbeat
PATTERN='Solana-Cripto-Trader-Live/agents/orchestrator'
LOG=/tmp/solana_live_killguard.log
ts=$(date +'%Y-%m-%d %H:%M:%S')
[ -f "$HB" ] || { echo "[$ts] heartbeat file missing — skip (watchdog handles cold start)" >> "$LOG"; exit 0; }
hb_ts=$(stat -c '%Y' "$HB" 2>/dev/null)
now=$(date +%s)
age=$(( now - hb_ts ))
if [ "$age" -lt "$STALE_SEC" ]; then
  exit 0   # healthy, quiet
fi
pids=$(pgrep -f "$PATTERN" 2>/dev/null | tr '\n' ' ')
if [ -z "$pids" ]; then
  echo "[$ts] heartbeat stale ${age}s but no orchestrator proc — watchdog will spawn fresh" >> "$LOG"
  exit 0
fi
echo "[$ts] KILL-GUARD: heartbeat stale ${age}s (>${STALE_SEC}) — SIGKILL orchestrator pids: ${pids}" >> "$LOG"
kill -9 $pids 2>>"$LOG"
echo "[$ts] kill sent. Watchdog (5s retry) will respawn." >> "$LOG"
