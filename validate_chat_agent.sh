#!/bin/bash
# validate_chat_agent.sh
# Validates chat_agent health for SOLAA-25:
# - path exists
# - process stays alive for >=60s
# - no repeated file-not-found lines in chat_agent.log in last minute
#
# Exit 0 = all checks pass
# Exit 1 = one or more checks fail

set -euo pipefail

WORKSPACE="/home/enderj/.openclaw/workspace/Solana-Cripto-Trader"
CHAT_AGENT_PATH="$WORKSPACE/agents/chat_agent.py"
CHAT_AGENT_LOG="/home/enderj/.config/solana-jupiter-bot/chat_agent.log"
PASS=true
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

echo "=== chat_agent validation ($TS) ==="

# 1. Path exists check
echo "[1/3] Checking chat_agent path..."
if [ -f "$CHAT_AGENT_PATH" ]; then
    echo "[PASS] chat_agent path exists: $CHAT_AGENT_PATH"
else
    echo "[FAIL] chat_agent path missing: $CHAT_AGENT_PATH"
    PASS=false
fi

# 2. Process alive check: stay alive for >=60s
echo "[2/3] Checking chat_agent process (60s observation window)..."
if pgrep -f "agents/chat_agent\.py" > /dev/null 2>&1; then
    echo "[PASS] chat_agent process is running (pgrep match on agents/chat_agent.py)"
    echo "       Waiting 60s to confirm it stays alive..."
    sleep 60
    if pgrep -f "agents/chat_agent\.py" > /dev/null 2>&1; then
        echo "[PASS] chat_agent still alive after 60s"
    else
        echo "[FAIL] chat_agent died within 60s observation window"
        PASS=false
    fi
else
    echo "[FAIL] chat_agent process not running"
    PASS=false
fi

# 3. No repeated file-not-found errors in last 60s of chat_agent.log
echo "[3/3] Checking chat_agent.log for recent file-not-found errors..."
if [ -f "$CHAT_AGENT_LOG" ]; then
    ONE_MINUTE_AGO=$(date -u -d '1 minute ago' +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -u -v-1M +%Y-%m-%dT%H:%M:%S 2>/dev/null)
    # Count occurrences of "can't open file" / "No such file" in last 60s
    RECENT_ERRORS=$(grep -c "can't open file\|No such file\|Errno 2" "$CHAT_AGENT_LOG" 2>/dev/null || echo "0")
    if [ "$RECENT_ERRORS" -eq 0 ]; then
        echo "[PASS] No file-not-found errors in chat_agent.log"
    else
        echo "[FAIL] Found $RECENT_ERRORS file-not-found error(s) in chat_agent.log:"
        grep "can't open file\|No such file\|Errno 2" "$CHAT_AGENT_LOG" | tail -5 | sed 's/^/       /'
        PASS=false
    fi
else
    echo "[WARN] chat_agent.log not found at $CHAT_AGENT_LOG (may not be created yet if chat_agent hasn't run)"
fi

echo ""
if [ "$PASS" = true ]; then
    echo "=== ALL CHECKS PASSED ==="
    exit 0
else
    echo "=== SOME CHECKS FAILED ==="
    exit 1
fi
