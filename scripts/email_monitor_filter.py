#!/usr/bin/env python3
"""
email_monitor_filter.py — Filtro de correos para email_monitor.sh
Uso: python3 email_monitor_filter.py <json_file> <state_file> <account_label>
Salida: líneas "NOTIFY:<mensaje>" para correos importantes
"""

import json
import sys
import re
from datetime import datetime
from pathlib import Path
from fnmatch import fnmatch

# ── Args ──────────────────────────────────────────────────────────────────────
if len(sys.argv) < 4:
    print(f"Uso: {sys.argv[0]} <json_file> <state_file> <label>", file=sys.stderr)
    sys.exit(1)

JSON_FILE   = Path(sys.argv[1])
STATE_FILE  = Path(sys.argv[2])
LABEL       = sys.argv[3]   # "enderjnets" o "blackvolt"

# ── Config ────────────────────────────────────────────────────────────────────

# Remitentes VIP — siempre notificar (case-insensitive, substring match)
VIP_SENDERS = [
    "americainsurance@outlook.com",
    "kswin@eisgroups.com",
    "crcontadoresmexico-usa@hotmail.com",
    "cristinaocando@gmail.com",
    "yonayonalife@gmail.com",
    "support@u-innovations.com",
    "margie240478@gmail.com",
    "alpari.com",
    "topstep.com",
    "ftmo.com",
    "kp.org",
    "kaiser",
    "uber.com",
    "lyft.com",
    "progressive.com",
    "mneal@decisionnext.com",
]

# Keywords urgentes en subject
URGENT_KEYWORDS = [
    "invoice", "payment", "pago", "quote", "cotizacion", "cotización",
    "insurance", "seguro", "tax", "impuesto", "urgent", "urgente",
    "importante", "important", "factura", "retiro", "withdrawal",
    "challenge", "ftmo", "topstep", "settlement", "deposit",
    "wire", "transferencia", "overdue", "past due", "vencido",
]

# Labels que descartan automáticamente (salvo VIP)
SKIP_LABELS = {
    "CATEGORY_PROMOTIONS",
    "CATEGORY_SOCIAL",
    "SPAM",
    "TRASH",
    "CATEGORY_UPDATES",
}

# Senders junk — ignorar siempre (fnmatch patterns)
JUNK_PATTERNS = [
    "noreply@*", "no-reply@*", "donotreply@*",
    "newsletter@*", "notifications@*", "mailer@*",
    "*@marketing.*", "*@email.*", "*@mail.*",
    "*groupon*", "*tacobell*", "*innosupps*",
    "*smartbrief*", "*ed.team*", "*coinbase bytes*",
    "*googleplay-noreply*", "*amazon*marketing*",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", file=sys.stderr)

def extract_addr(from_field: str) -> str:
    m = re.search(r'<([^>]+)>', from_field)
    return (m.group(1) if m else from_field).strip().lower()

def sender_name(from_field: str) -> str:
    m = re.match(r'^(.+?)\s*<', from_field)
    if m:
        return m.group(1).strip().strip('"')
    return extract_addr(from_field)

def is_vip(from_field: str) -> bool:
    f = from_field.lower()
    return any(v.lower() in f for v in VIP_SENDERS)

def is_junk(from_field: str) -> bool:
    addr = extract_addr(from_field)
    # Si es VIP, nunca junk
    if is_vip(from_field):
        return False
    for pattern in JUNK_PATTERNS:
        if fnmatch(addr, pattern):
            return True
    return False

def has_urgent_keyword(subject: str) -> bool:
    s = subject.lower()
    return any(k in s for k in URGENT_KEYWORDS)

def classify(thread: dict) -> tuple[str, str]:
    """
    Retorna (prioridad, razón): ("HIGH"/"MEDIUM"/"LOW", texto)
    """
    from_f  = thread.get("from", "")
    subject = thread.get("subject", "")
    labels  = set(thread.get("labels", []))
    count   = thread.get("messageCount", 1)

    # Junk → siempre LOW
    if is_junk(from_f):
        return "LOW", ""

    # Labels de skip → LOW salvo VIP
    if labels & SKIP_LABELS:
        if not is_vip(from_f):
            return "LOW", ""

    # VIP → HIGH siempre
    if is_vip(from_f):
        return "HIGH", "⭐ VIP"

    # Keyword urgente → HIGH
    if has_urgent_keyword(subject):
        return "HIGH", "🔥 Urgente"

    # Hilo con reply de persona real → MEDIUM
    if count > 1 and "UNREAD" in labels:
        return "MEDIUM", "💬 Respuesta"

    # IMPORTANT sin promos → MEDIUM
    if "IMPORTANT" in labels and not (labels & SKIP_LABELS):
        return "MEDIUM", "📌 Importante"

    # UNREAD en INBOX sin nada malo → MEDIUM (persona real escribió)
    if "UNREAD" in labels and "INBOX" in labels and not (labels & SKIP_LABELS):
        return "MEDIUM", "📧 Nuevo"

    return "LOW", ""

# ── Load data ─────────────────────────────────────────────────────────────────

# Cargar threads del JSON (escrito por gog)
try:
    raw = json.loads(JSON_FILE.read_text())
    threads = raw.get("threads", raw) if isinstance(raw, dict) else raw
    if not isinstance(threads, list):
        threads = []
except Exception as e:
    log(f"⚠️  Error leyendo {JSON_FILE}: {e}")
    sys.exit(0)

log(f"  → {len(threads)} threads en {LABEL}")

# Cargar state
try:
    state = json.loads(STATE_FILE.read_text())
except Exception:
    state = {}

seen_ids: set = set(state.get("seen_ids", []))

# ── Clasificar ────────────────────────────────────────────────────────────────

alerts   = []   # (prioridad, mensaje formateado)
new_seen = []   # todos los IDs procesados (importantes o no)

acct_emoji = "🟦" if "blackvolt" in LABEL else "📨"

for thread in threads:
    tid     = thread.get("id", "")
    subject = thread.get("subject", "(sin asunto)")
    from_f  = thread.get("from", "?")
    date    = thread.get("date", "")

    if not tid:
        continue

    if tid in seen_ids:
        continue  # ya notificado antes

    priority, reason = classify(thread)
    new_seen.append(tid)

    if priority == "LOW":
        log(f"  ⬜ LOW: {subject[:60]}")
        continue

    name = sender_name(from_f)
    msg = (
        f"{acct_emoji} <b>{reason}</b>\n"
        f"👤 {name}\n"
        f"📋 {subject}\n"
        f"🕐 {date}"
    )
    alerts.append((priority, msg))
    log(f"  {'🔴' if priority == 'HIGH' else '🟡'} {priority} [{reason}]: {subject[:60]}")

# ── Emitir alertas ────────────────────────────────────────────────────────────
# Ordenar HIGH primero
alerts.sort(key=lambda x: 0 if x[0] == "HIGH" else 1)

for _, msg in alerts:
    print(f"NOTIFY:{msg}")

log(f"  → {len(alerts)} alertas emitidas, {len(new_seen)} IDs nuevos")

# ── Actualizar state ──────────────────────────────────────────────────────────
all_seen = list(seen_ids) + new_seen
# Mantener últimos 500
all_seen = list(dict.fromkeys(all_seen))[-500:]

state["seen_ids"] = all_seen
state["last_run"] = datetime.now().isoformat()

try:
    STATE_FILE.write_text(json.dumps(state, indent=2))
except Exception as e:
    log(f"⚠️  Error guardando state: {e}")
