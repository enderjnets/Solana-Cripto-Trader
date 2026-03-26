#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# email_monitor.sh — Email Monitor Eko (bash puro, $0 tokens)
# Revisa enderjnets@gmail.com y blackvoltmobility@gmail.com
# Notifica correos importantes via Telegram direct curl.
# Cron: */30 * * * * /bin/bash /home/enderj/.openclaw/workspace/scripts/email_monitor.sh >> /home/enderj/.openclaw/workspace/memory/email_monitor.log 2>&1
# ══════════════════════════════════════════════════════════════════════════════

GOG="/home/enderj/.local/bin/gog"
STATE_FILE="/home/enderj/.openclaw/workspace/memory/email_monitor_state.json"
BOT_TOKEN="8704216511:AAECcULQZMNR0HE5pGUS4TuwhI_N4mg6P7g"
CHAT_ID="771213858"
MAX_EMAILS=30
TMP_JSON="/tmp/email_monitor_$$.json"

# ── Logging ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Init state ────────────────────────────────────────────────────────────────
if [[ ! -f "$STATE_FILE" ]]; then
    echo '{"seen_ids":[]}' > "$STATE_FILE"
    log "📁 State file creado"
fi

# ── Enviar Telegram ───────────────────────────────────────────────────────────
send_telegram() {
    local text="$1"
    local payload
    payload=$(python3 -c "
import json, sys
print(json.dumps({'chat_id': '$CHAT_ID', 'text': sys.argv[1], 'parse_mode': 'HTML'}))
" "$text")
    curl -s -X POST \
        "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
        -H "Content-Type: application/json" \
        -d "$payload" \
        --max-time 15 \
        -o /tmp/tg_response_$$.json
    local tg_ok
    tg_ok=$(python3 -c "import json; d=json.load(open('/tmp/tg_response_$$.json')); print('ok' if d.get('ok') else 'fail')" 2>/dev/null)
    rm -f /tmp/tg_response_$$.json
    [[ "$tg_ok" == "ok" ]]
}

# ── Procesar cuenta ───────────────────────────────────────────────────────────
process_account() {
    local account="$1"
    local label="$2"   # "enderjnets" o "blackvolt"
    local extra_args=()
    [[ "$account" != "enderjnets@gmail.com" ]] && extra_args=("-a" "$account")

    log "📬 Buscando en $account..."

    # Buscar en inbox, sin spam/trash
    "$GOG" gmail search "in:inbox -in:spam -in:trash" \
        --max "$MAX_EMAILS" --json "${extra_args[@]}" \
        > "$TMP_JSON" 2>/tmp/gog_err_$$.txt

    if [[ $? -ne 0 ]]; then
        log "⚠️  gog falló para $account: $(cat /tmp/gog_err_$$.txt)"
        rm -f /tmp/gog_err_$$.txt
        return 1
    fi
    rm -f /tmp/gog_err_$$.txt

    # Procesar con python3 — lee del archivo, NO de stdin
    python3 /home/enderj/.openclaw/workspace/scripts/email_monitor_filter.py \
        "$TMP_JSON" "$STATE_FILE" "$label"
}

# ── Main ──────────────────────────────────────────────────────────────────────
log "═══════════════════════════════════════════"
log "🚀 Email Monitor iniciando..."

ALERTS_FILE="/tmp/email_alerts_$$.txt"
> "$ALERTS_FILE"

# Procesar ambas cuentas — cada una agrega líneas a ALERTS_FILE
process_account "enderjnets@gmail.com"   "enderjnets"   >> "$ALERTS_FILE"
process_account "blackvoltmobility@gmail.com" "blackvolt" >> "$ALERTS_FILE"

# Leer alertas generadas
ALERTS=$(grep "^NOTIFY:" "$ALERTS_FILE" | sed 's/^NOTIFY://')
ALERT_COUNT=$(echo "$ALERTS" | grep -c . || true)

if [[ -n "$ALERTS" && "$ALERT_COUNT" -gt 0 ]]; then
    log "📤 Enviando $ALERT_COUNT alerta(s) a Telegram..."

    # Construir mensaje único
    HEADER="📬 <b>Correos nuevos importantes</b>"
    BODY=$(echo "$ALERTS" | head -5)   # máx 5 en un mensaje
    FULL_MSG="${HEADER}

${BODY}"

    if send_telegram "$FULL_MSG"; then
        log "✅ Telegram OK"
    else
        log "⚠️  Telegram falló"
    fi
else
    log "✅ Sin correos nuevos importantes — silencio"
fi

rm -f "$ALERTS_FILE" "$TMP_JSON"
log "🏁 Email Monitor terminado"
log "═══════════════════════════════════════════"
