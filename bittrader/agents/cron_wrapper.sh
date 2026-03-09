#!/bin/bash
# cron_wrapper.sh — Ejecuta un script y guarda el output para que Qwen lo resuma
# Uso: bash cron_wrapper.sh <script> [args...]
# Output se guarda en /tmp/cron_last_output.txt

SCRIPT="$1"
shift

cd /home/enderj/.openclaw/workspace/bittrader/agents

OUTPUT=$(python3 "$SCRIPT" "$@" 2>&1)
EXIT_CODE=$?

echo "$OUTPUT" > /tmp/cron_last_output.txt
echo "EXIT_CODE=$EXIT_CODE" >> /tmp/cron_last_output.txt

# Imprimir para que el sub-agente lo vea directamente en el prompt
echo "$OUTPUT"
echo ""
echo "Script exit code: $EXIT_CODE"
