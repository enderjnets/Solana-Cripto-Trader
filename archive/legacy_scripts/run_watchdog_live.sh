#!/bin/bash
# ─── Watchdog wrapper para Solana-Cripto-Trader-Live ──────────────────────────
# Exporta env vars para aislar locks/pattern de la instancia paper, luego
# delega al run_watchdog.sh compartido.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Cargar .env del proyecto (provee ORCH_LOCK_FILE, WATCHDOG_PREFIX, etc.)
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Fallback defensivo (por si .env no define — no queremos colisionar con paper)
export ORCH_LOCK_FILE="${ORCH_LOCK_FILE:-/tmp/solana_live_orchestrator.lock}"
export WATCHDOG_PREFIX="${WATCHDOG_PREFIX:-solana_live}"
export ORCH_PGREP_PATTERN="${ORCH_PGREP_PATTERN:-Solana-Cripto-Trader-Live/agents/orchestrator}"

# Delegar al watchdog principal (que ahora respeta estas env vars)
exec "$SCRIPT_DIR/run_watchdog.sh" "$@"
