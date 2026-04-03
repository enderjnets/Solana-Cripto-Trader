#!/usr/bin/env python3
"""
🔄 Reset Bot - Resetea todo el estado del bot a valores iniciales
Uso: python3 reset_bot.py [--capital 1000] [--force]

⚠️  IMPORTANTE: Este script hace BACKUP de trade_history antes de resetear.
    Los backups se guardan en data/backups/ con timestamp.
"""

import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
BACKUP_DIR = DATA_DIR / "auto_backups"

def create_pre_reset_backup() -> str:
    """
    Hace backup de TODO el estado actual ANTES de cualquier modificación.
    Incluye: portfolio.json, trade_history.json, daily_target_state.json
    Retorna la ruta del backup.
    """
    BACKUP_DIR.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    backup_subdir = BACKUP_DIR / f"pre_reset_{timestamp}"
    backup_subdir.mkdir(exist_ok=True)

    files_to_backup = [
        "portfolio.json",
        "trade_history.json",
        "daily_target_state.json",
        "auto_learner_state.json",
        "compound_state.json",
    ]

    backed_up = []
    for fname in files_to_backup:
        src = DATA_DIR / fname
        if src.exists():
            shutil.copy2(src, backup_subdir / fname)
            backed_up.append(fname)

    # Guardar metadata del backup
    backup_meta = {
        "timestamp": now.isoformat(),
        "capital_before_reset": None,
        "files_backed_up": backed_up,
    }
    try:
        port = json.loads((DATA_DIR / "portfolio.json").read_text())
        backup_meta["capital_before_reset"] = port.get("capital_usd")
    except Exception:
        pass

    (backup_subdir / "backup_meta.json").write_text(json.dumps(backup_meta, indent=2))
    backed_up.append("backup_meta.json")

    print(f"   💾 Backup creado: {backup_subdir.name}/ ({len(backed_up)} archivos)")
    return str(backup_subdir)


def get_current_capital() -> float:
    """Obtiene el capital actual del portfolio.json si existe."""
    try:
        port_file = DATA_DIR / "portfolio.json"
        if port_file.exists():
            port = json.loads(port_file.read_text())
            return port.get("capital_usd", 1000.0)
    except Exception:
        pass
    return 1000.0


def reset_all(capital: float = None, force: bool = False):
    """
    Resetea todos los archivos de estado del bot.

    Args:
        capital: Capital inicial. Si es None, usa el capital actual del portfolio.
                 Esto evita perder el capital real tras un crash.
        force:   Si True, permite reset aunque ya hay 0 trades (para 'fresh start' manual).
    """
    now = datetime.now(timezone.utc).isoformat()

    # Si no se especifica capital, usar el actual (preserva capital real tras crash)
    if capital is None:
        capital = get_current_capital()
        print(f"   ℹ️  Capital no especificado — usando capital actual: ${capital:.2f}")

    # Hacer backup ANTES de cualquier modificación
    backup_path = create_pre_reset_backup()
    results = [f"💾 Backup pre-reset: {backup_path}"]

    # ── 1. Portfolio ──────────────────────────────────────────────────
    portfolio = {
        "capital_usd": capital,
        "initial_capital": capital,  # ← Usa el capital REAL, no hardcoded
        "positions": [],
        "status": "ACTIVE",
        "mode": "paper_drift",
        "created_at": now,
        "last_updated": now,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "drawdown_pct": 0.0,
    }
    (DATA_DIR / "portfolio.json").write_text(json.dumps(portfolio, indent=2))
    results.append("✅ portfolio.json (capital preservado)")

    # ── 2. Trade History — SIEMPRE backup antes de limpiar ──────────
    # El backup ya se hizo arriba, solo documentar
    history_file = DATA_DIR / "trade_history.json"
    if history_file.exists():
        try:
            existing = json.loads(history_file.read_text())
            results.append(f"✅ trade_history.json RESETEADO (tenía {len(existing)} trades — backup en {backup_path})")
        except Exception:
            results.append("✅ trade_history.json RESETEADO (backup en pre-reset)")
    else:
        results.append("✅ trade_history.json (ya estaba vacío)")
    history_file.write_text("[]")

    # ── 3. Daily Target State ────────────────────────────────────────
    daily_state = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "starting_capital": capital,
        "target_reached": False,
        "closed_at": None,
        "enabled": True,
        "current_pnl_pct": 0.0,
        "target_pct": 0.05,
    }
    (DATA_DIR / "daily_target_state.json").write_text(json.dumps(daily_state, indent=2))
    results.append("✅ daily_target_state.json")

    # ── 4. Daily Report ───────────────────────────────────────────────
    daily_report = {
        "timestamp": now,
        "capital_usd": capital,
        "equity": capital,
        "pnl_total_usd": 0.0,
        "pnl_total_pct": 0.0,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "drawdown_pct": 0.0,
        "positions_open": 0,
    }
    (DATA_DIR / "daily_report.json").write_text(json.dumps(daily_report, indent=2))
    results.append("✅ daily_report.json")

    # ── 5. Compound State ─────────────────────────────────────────────
    compound = {
        "enabled": True,
        "last_compound": None,
        "compound_count": 0,
        "total_compounded": 0.0,
    }
    (DATA_DIR / "compound_state.json").write_text(json.dumps(compound, indent=2))
    results.append("✅ compound_state.json")

    # ── 6. Auto Learner State — PRESERVAR conocimiento ───────────────
    learner_file = DATA_DIR / "auto_learner_state.json"
    if learner_file.exists():
        try:
            existing_learner = json.loads(learner_file.read_text())
            existing_learner["last_updated"] = now
            existing_learner["notes"] = f"Reset capital ${capital:.2f} - conocimiento preservado"
            learner_file.write_text(json.dumps(existing_learner, indent=2))
            results.append("✅ auto_learner_state.json (conocimiento preservado)")
        except Exception:
            results.append("⚠️  auto_learner_state.json (error leyendo, preservado)")
    else:
        learner = {
            "params": {
                "sl_pct": 0.025,
                "tp_pct": 0.05,
                "leverage_tier": 2,
                "risk_per_trade": 0.02,
                "max_positions": 5,
            },
            "total_trades_learned": 0,
            "last_adaptation": None,
            "adaptation_count": 0,
            "last_updated": now,
            "notes": f"Fresh start - capital ${capital:.2f}",
        }
        learner_file.write_text(json.dumps(learner, indent=2))
        results.append("✅ auto_learner_state.json (nuevo)")

    # ── 7. Alerts State ───────────────────────────────────────────────
    alerts = {
        "last_daily_report": "",
        "alerted_trades": [],
        "reset_at": now,
    }
    (DATA_DIR / "alerts_state.json").write_text(json.dumps(alerts, indent=2))
    results.append("✅ alerts_state.json")

    # ── 8. Reset History ─────────────────────────────────────────────
    reset_history_file = DATA_DIR / "reset_history.json"
    try:
        reset_history = json.loads(reset_history_file.read_text()) if reset_history_file.exists() else []
    except Exception:
        reset_history = []

    reset_history.append({
        "reset_date": now,
        "period_start": None,
        "period_end": None,
        "initial_capital": capital,
        "final_capital": capital,
        "return_pct": 0.0,
        "total_pnl_usd": 0,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0,
        "best_trade_usd": 0,
        "worst_trade_usd": 0,
        "new_capital": capital,
        "reason": "manual_reset",
    })

    # Mantener solo los últimos 50 resets
    if len(reset_history) > 50:
        reset_history = reset_history[-50:]

    reset_history_file.write_text(json.dumps(reset_history, indent=2))
    results.append(f"✅ reset_history.json ({len(reset_history)} resets históricos)")

    return {
        "success": True,
        "capital": capital,
        "backup_path": backup_path,
        "files_reset": results,
        "timestamp": now,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset bot state")
    parser.add_argument("--capital", type=float, default=None,
                        help="Capital inicial. Si se omite, usa el capital actual del portfolio.")
    parser.add_argument("--force", action="store_true",
                        help="Fuerza reset incluso si ya hay 0 trades.")
    args = parser.parse_args()

    print(f"\n🔄 Reseteando bot...")
    if args.capital:
        print(f"   Capital especificado: ${args.capital:.2f}")
    else:
        current = get_current_capital()
        print(f"   Capital actual (será preservado): ${current:.2f}")
    print()

    result = reset_all(capital=args.capital, force=args.force)

    for f in result["files_reset"]:
        print(f"   {f}")

    print(f"\n✅ Bot reseteado exitosamente!")
    print(f"   Capital: ${result['capital']:.2f}")
    print(f"   Backup pre-reset: {result['backup_path']}")
    print(f"   Timestamp: {result['timestamp']}")
