#!/bin/bash
# check_heartbeat.sh - Monitor Unified Trading System v4 heartbeat
# ================================================================
# This script checks the heartbeat file to ensure the system is alive.
# Can be used for automated recovery.

HEARTBEAT_FILE="/home/enderj/.openclaw/workspace/solana-jupiter-bot/data/heartbeat.txt"
MAX_AGE=300  # 5 minutes = 300 seconds
PROJECT_DIR="/home/enderj/.openclaw/workspace/solana-jupiter-bot"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if heartbeat file exists
if [ ! -f "$HEARTBEAT_FILE" ]; then
    echo -e "${RED}❌ CRITICAL: No heartbeat file found!${NC}"
    echo "   The system may have crashed or never started."
    echo ""
    echo "   To restart:"
    echo "   cd $PROJECT_DIR"
    echo "   nohup python3 unified_trading_system.py --continuous > unified_trading_system_output.log 2>&1 &"
    exit 1
fi

# Extract timestamp from heartbeat file
TIMESTAMP=$(cat "$HEARTBEAT_FILE" | jq -r '.timestamp' 2>/dev/null)

if [ "$TIMESTAMP" == "null" ] || [ -z "$TIMESTAMP" ]; then
    echo -e "${RED}❌ ERROR: Invalid heartbeat file!${NC}"
    exit 1
fi

# Calculate age
if command -v date &> /dev/null; then
    # Linux with GNU date
    CURRENT_EPOCH=$(date +%s)
    HEARTBEAT_EPOCH=$(date -d "$TIMESTAMP" +%s 2>/dev/null || echo 0)
    AGE=$((CURRENT_EPOCH - HEARTBEAT_EPOCH))
else
    echo -e "${YELLOW}⚠️  WARNING: Cannot calculate heartbeat age (date command not available)${NC}"
    AGE=0
fi

# Extract other info
PID=$(cat "$HEARTBEAT_FILE" | jq -r '.pid' 2>/dev/null)
MEMORY_MB=$(cat "$HEARTBEAT_FILE" | jq -r '.memory_mb' 2>/dev/null)
STATUS=$(cat "$HEARTBEAT_FILE" | jq -r '.status' 2>/dev/null)

# Check age
if [ "$AGE" -gt "$MAX_AGE" ]; then
    echo -e "${RED}❌ CRITICAL: System not responding!${NC}"
    echo "   Heartbeat is ${AGE}s old (max: ${MAX_AGE}s)"
    echo "   Last heartbeat: $TIMESTAMP"
    echo "   PID: $PID"
    echo ""
    echo "   Possible causes:"
    echo "   - Process hung during cycle"
    echo "   - Process crashed"
    echo "   - Memory leak (current: ${MEMORY_MB} MB)"
    echo ""
    echo "   To restart:"
    echo "   cd $PROJECT_DIR"
    echo "   kill -9 $PID"
    echo "   nohup python3 unified_trading_system.py --continuous > unified_trading_system_output.log 2>&1 &"
    exit 1
fi

# Check if process is actually running
if ! ps -p "$PID" > /dev/null 2>&1; then
    echo -e "${RED}❌ CRITICAL: Process not running!${NC}"
    echo "   Heartbeat exists but PID $PID is not found"
    echo "   This may indicate a hard crash"
    echo ""
    echo "   To restart:"
    echo "   cd $PROJECT_DIR"
    echo "   nohup python3 unified_trading_system.py --continuous > unified_trading_system_output.log 2>&1 &"
    exit 1
fi

# All checks passed
echo -e "${GREEN}✅ SYSTEM HEALTHY${NC}"
echo "   Status: $STATUS"
echo "   PID: $PID"
echo "   Heartbeat age: ${AGE}s"
echo "   Memory: ${MEMORY_MB} MB"
echo "   Last heartbeat: $TIMESTAMP"

# Memory warning
if [ "$MEMORY_MB" -gt 500 ]; then
    echo -e "${YELLOW}⚠️  WARNING: High memory usage (${MEMORY_MB} MB)${NC}"
fi

exit 0
