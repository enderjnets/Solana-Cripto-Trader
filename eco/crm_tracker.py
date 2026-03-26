#!/usr/bin/env python3
"""
crm_tracker.py — Eco AI Automation LLC
=======================================
CRM simple para tracking de prospectos del AI Voice Agent service.

Lee y actualiza /eco/crm_prospects.md con estados de cada prospecto.

ESTADOS:
    PENDIENTE → CONTACTADO → RESPONDIÓ → DEMO → PROPUESTA → CERRADO | DESCARTADO

USO:
    python3 crm_tracker.py                           # Ver resumen del CRM
    python3 crm_tracker.py --list                    # Lista todos los prospectos
    python3 crm_tracker.py --update 1 CONTACTADO     # Actualiza estado por ID
    python3 crm_tracker.py --update 1 CONTACTADO --note "Llamé, no contestó"
    python3 crm_tracker.py --update 1 CONTACTADO --phone "+13035551234"
    python3 crm_tracker.py --stats                   # Estadísticas del funnel
    python3 crm_tracker.py --ready                   # Prospectos listos para llamar
    python3 crm_tracker.py --export                  # Exporta a JSON
    python3 crm_tracker.py --add-phone 1 +13035551234  # Agrega teléfono a prospecto
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from collections import Counter

# ─── PATHS ───────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/enderj/.openclaw/workspace/eco")
CRM_FILE = WORKSPACE / "crm_prospects.md"
CRM_JSON = WORKSPACE / "crm_data.json"

# ─── ESTADOS VÁLIDOS ─────────────────────────────────────────────────────────
VALID_STATES = [
    "PENDIENTE",
    "CONTACTADO",
    "RESPONDIÓ",
    "DEMO",
    "PROPUESTA",
    "CERRADO",
    "DESCARTADO",
]

STATE_EMOJI = {
    "PENDIENTE": "⚪",
    "CONTACTADO": "📞",
    "RESPONDIÓ": "💬",
    "DEMO": "🎯",
    "PROPUESTA": "📄",
    "CERRADO": "✅",
    "DESCARTADO": "❌",
}

# ─── PARSER DE crm_prospects.md ──────────────────────────────────────────────

def parse_crm_markdown(filepath: Path) -> list:
    """Parsea el archivo CRM markdown y devuelve lista de prospectos."""
    if not filepath.exists():
        print(f"❌ Archivo no encontrado: {filepath}")
        return []

    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    prospects = []
    # Busca filas de tabla: | # | Negocio | Contacto | Teléfono | Email | Canal | Estado | Fecha | Notas |
    table_pattern = re.compile(
        r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|",
        re.MULTILINE,
    )

    for m in table_pattern.finditer(content):
        try:
            pid = int(m.group(1))
        except ValueError:
            continue

        raw_state = m.group(7).strip().upper()
        state = raw_state if raw_state in VALID_STATES else "PENDIENTE"

        phone_raw = m.group(4).strip()
        phone = phone_raw if phone_raw and phone_raw != "-" else None

        email_raw = m.group(5).strip()
        email = email_raw if email_raw and email_raw != "-" else None

        contact_raw = m.group(3).strip()
        contact = contact_raw if contact_raw and contact_raw != "-" else None

        prospects.append({
            "id": pid,
            "business": m.group(2).strip().strip("**"),
            "contact": contact,
            "phone": phone,
            "email": email,
            "channel": m.group(6).strip(),
            "state": state,
            "date_contacted": m.group(8).strip() if m.group(8).strip() != "-" else None,
            "notes": m.group(9).strip() if m.group(9).strip() != "-" else "",
        })

    return sorted(prospects, key=lambda x: x["id"])


def write_crm_markdown(prospects: list, filepath: Path) -> None:
    """Reescribe el archivo CRM markdown con los datos actualizados."""
    with open(filepath, encoding="utf-8") as f:
        original = f.read()

    # Reconstruir la sección de tabla
    header = "| # | Negocio | Contacto | Teléfono | Email | Canal | Estado | Fecha Contacto | Notas |"
    separator = "|---|---------|----------|----------|-------|-------|--------|----------------|-------|"

    rows = []
    for p in sorted(prospects, key=lambda x: x["id"]):
        rows.append(
            f"| {p['id']} | {p['business']} | {p.get('contact') or '-'} | "
            f"{p.get('phone') or '-'} | {p.get('email') or '-'} | "
            f"{p.get('channel', 'SMS')} | {p['state']} | "
            f"{p.get('date_contacted') or '-'} | {p.get('notes') or '-'} |"
        )

    new_table = "\n".join([header, separator] + rows)

    # Reemplazar la tabla existente
    table_pattern = re.compile(
        r"\| # \| Negocio \| Contacto.*?\n(\|[-|]+\|\n)(\|.*?\n)*",
        re.DOTALL,
    )

    new_content = table_pattern.sub(new_table + "\n", original)

    # Si no se encontró la tabla, agregar al final
    if new_content == original:
        new_content = original + "\n" + new_table + "\n"

    # Actualizar fecha
    new_content = re.sub(
        r"\*\*Actualizado:\*\* \d{4}-\d{2}-\d{2}",
        f"**Actualizado:** {datetime.now().strftime('%Y-%m-%d')}",
        new_content,
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)


def save_json_cache(prospects: list) -> None:
    """Guarda una copia JSON del CRM para uso programático."""
    data = {
        "updated_at": datetime.now().isoformat(),
        "total": len(prospects),
        "prospects": prospects,
    }
    with open(CRM_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─── DISPLAY FUNCTIONS ────────────────────────────────────────────────────────

def show_summary(prospects: list) -> None:
    """Muestra resumen ejecutivo del CRM."""
    state_counts = Counter(p["state"] for p in prospects)
    with_phone = sum(1 for p in prospects if p.get("phone"))
    with_email = sum(1 for p in prospects if p.get("email"))

    print(f"\n{'─'*50}")
    print(f"  📊 CRM — Eco AI Automation LLC")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'─'*50}")
    print(f"\n  FUNNEL DE VENTAS ({len(prospects)} prospectos total):\n")

    for state in VALID_STATES:
        count = state_counts.get(state, 0)
        bar = "█" * count + "░" * (max(0, 10 - count))
        pct = f"{(count/len(prospects)*100):.0f}%" if prospects else "0%"
        emoji = STATE_EMOJI.get(state, "⚪")
        print(f"  {emoji} {state:<12} {bar} {count:2d} ({pct})")

    print(f"\n  CONTACTO:")
    print(f"  📞 Con teléfono: {with_phone}/{len(prospects)}")
    print(f"  📧 Con email:    {with_email}/{len(prospects)}")

    # MRR estimado
    closed = state_counts.get("CERRADO", 0)
    mrr = closed * 400
    print(f"\n  💰 MRR ACTUAL: ${mrr:,} ({closed} clientes)")
    print(f"  🎯 META: $2,000 (5 clientes)")
    gap = max(0, 5 - closed)
    print(f"  📈 Faltan {gap} clientes para meta")
    print(f"{'─'*50}\n")


def show_list(prospects: list, filter_state: str = None) -> None:
    """Lista prospectos, opcionalmente filtrados por estado."""
    filtered = prospects
    if filter_state:
        filtered = [p for p in prospects if p["state"] == filter_state.upper()]

    if not filtered:
        print(f"\n  No hay prospectos con estado: {filter_state}")
        return

    print(f"\n  {'#':<3} {'NEGOCIO':<30} {'ESTADO':<12} {'TELÉFONO':<16} {'FECHA':<12} NOTAS")
    print(f"  {'─'*3} {'─'*30} {'─'*12} {'─'*16} {'─'*12} {'─'*20}")

    for p in filtered:
        emoji = STATE_EMOJI.get(p["state"], "⚪")
        phone = p.get("phone") or "—"
        date = p.get("date_contacted") or "—"
        notes = (p.get("notes") or "—")[:25]
        print(f"  {p['id']:<3} {p['business']:<30} {emoji}{p['state']:<11} {phone:<16} {date:<12} {notes}")

    print(f"\n  Total: {len(filtered)}")


def show_ready_to_call(prospects: list) -> None:
    """Muestra prospectos listos para llamar (PENDIENTE + tienen teléfono)."""
    ready = [
        p for p in prospects
        if p["state"] == "PENDIENTE" and p.get("phone")
    ]
    no_phone = [
        p for p in prospects
        if p["state"] == "PENDIENTE" and not p.get("phone")
    ]

    print(f"\n  📞 LISTOS PARA LLAMAR ({len(ready)}):\n")
    if ready:
        for p in ready:
            print(f"  [{p['id']:2d}] {p['business']:<35} {p['phone']}")
            print(f"       Industria: {p.get('channel','?')} | Ciudad: (ver sales plan)")
    else:
        print("  Ninguno — agrega teléfonos con --add-phone <id> <phone>")

    if no_phone:
        print(f"\n  ❌ PENDIENTES SIN TELÉFONO ({len(no_phone)}) — necesitan investigación:")
        for p in no_phone:
            print(f"  [{p['id']:2d}] {p['business']}")

    print(f"\n  COMANDO PARA LLAMAR:")
    print(f"  python3 outreach_caller.py --mode prospect --dry-run")


def show_stats(prospects: list) -> None:
    """Muestra estadísticas de conversión del funnel."""
    total = len(prospects)
    if total == 0:
        print("No hay datos.")
        return

    contacted = sum(1 for p in prospects if p["state"] != "PENDIENTE")
    responded = sum(1 for p in prospects if p["state"] in ["RESPONDIÓ", "DEMO", "PROPUESTA", "CERRADO"])
    demo = sum(1 for p in prospects if p["state"] in ["DEMO", "PROPUESTA", "CERRADO"])
    proposal = sum(1 for p in prospects if p["state"] in ["PROPUESTA", "CERRADO"])
    closed = sum(1 for p in prospects if p["state"] == "CERRADO")

    print(f"\n  📈 ESTADÍSTICAS DE CONVERSIÓN:\n")
    print(f"  Total prospectos:     {total}")
    print(f"  Contactados:          {contacted} ({contacted/total*100:.0f}%)")

    if contacted > 0:
        print(f"  Respondieron:         {responded} ({responded/contacted*100:.0f}% de contactados)")
    if responded > 0:
        print(f"  Llegaron a demo:      {demo} ({demo/responded*100:.0f}% de respondidos)")
    if demo > 0:
        print(f"  Recibieron propuesta: {proposal} ({proposal/demo*100:.0f}% de demos)")
    if proposal > 0:
        print(f"  Cerraron deal:        {closed} ({closed/proposal*100:.0f}% de propuestas)")

    mrr = closed * 400
    setup = closed * 500
    print(f"\n  💰 INGRESOS:")
    print(f"  Setup recibido:  ${setup:,}")
    print(f"  MRR actual:      ${mrr:,}/mes")
    print(f"  MRR meta:        $2,000/mes")
    print(f"  Progreso:        {mrr/2000*100:.0f}%")


# ─── UPDATE FUNCTIONS ─────────────────────────────────────────────────────────

def update_prospect(
    prospects: list,
    prospect_id: int,
    new_state: str,
    note: str = None,
    phone: str = None,
    email: str = None,
    contact: str = None,
) -> list:
    """Actualiza el estado de un prospecto."""
    new_state = new_state.upper()
    if new_state not in VALID_STATES:
        print(f"❌ Estado inválido: {new_state}")
        print(f"   Válidos: {', '.join(VALID_STATES)}")
        sys.exit(1)

    updated = False
    for p in prospects:
        if p["id"] == prospect_id:
            old_state = p["state"]
            p["state"] = new_state
            p["date_contacted"] = datetime.now().strftime("%Y-%m-%d")

            if note:
                existing = p.get("notes") or ""
                timestamp = datetime.now().strftime("%m/%d")
                p["notes"] = f"{existing} [{timestamp}] {note}".strip()

            if phone:
                p["phone"] = phone
            if email:
                p["email"] = email
            if contact:
                p["contact"] = contact

            print(f"\n✅ Actualizado: [{prospect_id}] {p['business']}")
            print(f"   Estado: {old_state} → {new_state} {STATE_EMOJI.get(new_state,'')}")
            if note:
                print(f"   Nota: {note}")
            if phone:
                print(f"   Teléfono: {phone}")
            updated = True
            break

    if not updated:
        print(f"❌ No se encontró prospecto con ID: {prospect_id}")
        print(f"   IDs disponibles: {[p['id'] for p in prospects]}")

    return prospects


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CRM tracker para Eco AI Automation prospecting"
    )

    parser.add_argument(
        "--list", "-l",
        nargs="?",
        const="ALL",
        metavar="ESTADO",
        help="Lista prospectos. Opcional: filtrar por estado (ej: --list CONTACTADO)",
    )
    parser.add_argument(
        "--update", "-u",
        nargs=2,
        metavar=("ID", "ESTADO"),
        help="Actualiza estado: --update 1 CONTACTADO",
    )
    parser.add_argument("--note", help="Nota a agregar al prospecto")
    parser.add_argument("--phone", help="Teléfono a agregar/actualizar")
    parser.add_argument("--email", help="Email a agregar/actualizar")
    parser.add_argument("--contact", help="Nombre del contacto")
    parser.add_argument(
        "--add-phone",
        nargs=2,
        metavar=("ID", "PHONE"),
        help="Agrega teléfono a un prospecto: --add-phone 1 +13035551234",
    )
    parser.add_argument("--stats", action="store_true", help="Estadísticas del funnel")
    parser.add_argument("--ready", action="store_true", help="Prospectos listos para llamar")
    parser.add_argument("--export", action="store_true", help="Exporta CRM a JSON")
    parser.add_argument("--file", default=str(CRM_FILE), help="Ruta del archivo CRM")

    args = parser.parse_args()

    crm_path = Path(args.file)
    prospects = parse_crm_markdown(crm_path)

    if not prospects:
        print(f"⚠️  No se pudieron parsear prospectos de: {crm_path}")
        print("   Verifica que el archivo existe y tiene el formato correcto.")
        sys.exit(1)

    # Comandos de solo lectura
    if args.stats:
        show_stats(prospects)
        return

    if args.ready:
        show_ready_to_call(prospects)
        return

    if args.export:
        save_json_cache(prospects)
        print(f"✅ CRM exportado a: {CRM_JSON}")
        return

    if args.list:
        state_filter = None if args.list == "ALL" else args.list
        show_list(prospects, state_filter)
        return

    # Agregar teléfono
    if args.add_phone:
        pid, phone = args.add_phone
        prospects = update_prospect(
            prospects, int(pid), 
            prospects[int(pid)-1]["state"],  # mantener estado actual
            phone=phone,
        )
        write_crm_markdown(prospects, crm_path)
        save_json_cache(prospects)
        return

    # Actualizar estado
    if args.update:
        pid, new_state = args.update
        prospects = update_prospect(
            prospects,
            int(pid),
            new_state,
            note=args.note,
            phone=args.phone,
            email=args.email,
            contact=args.contact,
        )
        write_crm_markdown(prospects, crm_path)
        save_json_cache(prospects)
        show_summary(prospects)
        return

    # Default: mostrar resumen
    show_summary(prospects)
    print("  Comandos útiles:")
    print("    python3 crm_tracker.py --list")
    print("    python3 crm_tracker.py --ready")
    print("    python3 crm_tracker.py --update 1 CONTACTADO --note 'Llamé hoy'")
    print("    python3 crm_tracker.py --add-phone 1 +13035551234")
    print("    python3 crm_tracker.py --stats")


if __name__ == "__main__":
    main()
