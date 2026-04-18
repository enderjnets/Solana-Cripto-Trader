#!/usr/bin/env python3
"""
emergency_close.py — Standalone emergency close watchdog (Sprint 2 Fase 5)

Corre INDEPENDIENTE del orchestrator. Detecta si el bot está vivo via
heartbeat file. Si orchestrator murió o se congeló, cierra todas las
posiciones live abiertas convirtiéndolas a USDC (salida limpia).

Uso:
    # One-shot (chequea una vez, ejecuta si stale):
    python3 tools/emergency_close.py --check

    # Forzar close ahora (ignora heartbeat, cierra todo):
    python3 tools/emergency_close.py --force

    # Solo status (no cierra nada):
    python3 tools/emergency_close.py --status

Setup recomendado (cron o systemd timer, cada 60s):
    * * * * * /usr/bin/python3 /path/to/tools/emergency_close.py --check

Safety:
- LIVE_TRADING_ENABLED=false → no-op total
- Sin wallet → no-op (no puede cerrar)
- Usa el mismo safety.activate_kill_switch() al disparar
- Después de close_all, el kill switch queda activo → orchestrator no
  reabrirá posiciones aunque vuelva
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

# Path hackery para poder importar agents sin instalación
AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"
sys.path.insert(0, str(AGENTS_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger('emergency_close')

DEFAULT_HEARTBEAT_FILE = "/tmp/solana_live_heartbeat"
DEFAULT_TIMEOUT_SEC = 300   # 5 min — si heartbeat older than this, trigger


def _load_env():
    """Load .env from parent dir if exists."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())


def heartbeat_status() -> dict:
    """Returns status dict: {'exists', 'age_sec', 'stale', 'path'}."""
    path = os.environ.get("HEARTBEAT_FILE", DEFAULT_HEARTBEAT_FILE)
    timeout = int(os.environ.get("HEARTBEAT_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC))
    p = Path(path)
    status = {"exists": p.exists(), "path": path, "timeout_sec": timeout}
    if status["exists"]:
        try:
            age = time.time() - p.stat().st_mtime
            status["age_sec"] = age
            status["stale"] = age > timeout
        except Exception:
            status["age_sec"] = None
            status["stale"] = True
    else:
        status["age_sec"] = None
        status["stale"] = True   # no heartbeat = as bad as stale
    return status


def _get_live_positions() -> list:
    """Read portfolio.json and return open live positions."""
    pf = AGENTS_DIR / "data" / "portfolio.json"
    if not pf.exists():
        return []
    try:
        data = json.loads(pf.read_text())
        return [
            p for p in data.get('positions', [])
            if p.get('status') == 'open' and p.get('mode') == 'live'
        ]
    except Exception as e:
        log.error(f"portfolio read error: {e}")
        return []


def close_all_live_positions(dry_run: bool = False) -> dict:
    """
    Cierra todas las live positions abiertas convirtiéndolas a USDC via Jupiter.
    Retorna dict con resumen: {closed, failed, positions: [...]}.
    """
    result = {'closed': 0, 'failed': 0, 'positions': [], 'dry_run': dry_run}

    if os.environ.get('LIVE_TRADING_ENABLED', 'false').lower() != 'true':
        log.info("LIVE_TRADING_ENABLED=false — no-op (nothing to close)")
        return result

    try:
        import solana_rpc
        import wallet as wallet_mod
        import jupiter_swap
        import executor   # for MINT_MAP, DECIMALS_MAP
    except Exception as e:
        log.error(f"deps import failed: {e}")
        result['error'] = str(e)
        return result

    positions = _get_live_positions()
    if not positions:
        log.info("No live positions to close")
        return result

    log.warning(f"🚨 EMERGENCY CLOSE initiated — {len(positions)} live positions")
    if dry_run:
        log.info("(DRY RUN — no actual swaps)")

    try:
        w = wallet_mod.load_wallet()
    except Exception as e:
        log.error(f"wallet load failed: {e}")
        result['error'] = f"wallet_load_failed: {e}"
        return result

    if not getattr(w, 'is_live', False):
        log.error("wallet is MockWallet — cannot close live positions")
        result['error'] = "mockwallet_cannot_close"
        return result

    rpc = solana_rpc.get_rpc()
    swap = jupiter_swap.JupiterSwap(wallet=w, rpc=rpc)

    for pos in positions:
        sym = pos.get('symbol', '?')
        tokens = float(pos.get('tokens', 0) or 0)
        mint = executor.MINT_MAP.get(sym)
        decimals = executor.DECIMALS_MAP.get(sym, 9)

        entry = {
            'symbol': sym,
            'tokens': tokens,
            'status': 'pending',
        }

        if not mint or tokens <= 0:
            entry['status'] = 'skipped_no_mint_or_zero_tokens'
            result['positions'].append(entry)
            continue

        if dry_run:
            entry['status'] = 'dry_run_would_close'
            entry['tx_signature'] = None
            result['positions'].append(entry)
            continue

        # token → USDC swap
        amount_lamports = int(tokens * (10 ** decimals))
        max_slippage = int(os.environ.get('MAX_SLIPPAGE_BPS', 200))   # wider on emergency

        log.warning(f"Emergency close: {sym} {tokens:.6f} → USDC (slippage {max_slippage}bps)")
        try:
            res = swap.execute_swap(
                input_mint=mint,
                output_mint=solana_rpc.MINT_USDC,
                amount_lamports=amount_lamports,
                slippage_bps=max_slippage,
                priority_fee_level='high',   # priorizar salida
                dry_run=False,
            )
            if res.success:
                entry['status'] = 'closed'
                entry['tx_signature'] = res.signature
                entry['usdc_received'] = res.out_amount / 1_000_000
                result['closed'] += 1
                log.info(f"✅ {sym} cerrado: tx={res.signature[:16]}... USDC={entry['usdc_received']:.2f}")
            else:
                entry['status'] = f'failed: {res.error}'
                result['failed'] += 1
                log.error(f"❌ {sym} close failed: {res.error}")
        except Exception as e:
            entry['status'] = f'exception: {e}'
            result['failed'] += 1
            log.error(f"❌ {sym} close exception: {e}")

        result['positions'].append(entry)

    # Después de intentar cerrar → activar kill switch (bot no reabre)
    try:
        import safety
        safety.activate_kill_switch(f"emergency_close: closed={result['closed']} failed={result['failed']}")
        log.warning("Kill switch activado post-emergency close")
    except Exception as e:
        log.error(f"kill switch activation failed: {e}")

    return result


def main():
    _load_env()
    parser = argparse.ArgumentParser(description="Emergency close watchdog for live trading")
    parser.add_argument('--check', action='store_true', help='Check heartbeat, close if stale')
    parser.add_argument('--force', action='store_true', help='Force close all live positions (ignores heartbeat)')
    parser.add_argument('--status', action='store_true', help='Show heartbeat status only (no action)')
    parser.add_argument('--dry-run', action='store_true', help='Simulate close without actual swaps')
    args = parser.parse_args()

    if not any([args.check, args.force, args.status]):
        parser.print_help()
        return 1

    status = heartbeat_status()
    log.info(f"Heartbeat: exists={status['exists']} age={status.get('age_sec')}s stale={status['stale']} "
             f"(timeout={status['timeout_sec']}s, file={status['path']})")

    if args.status:
        positions = _get_live_positions()
        log.info(f"Open live positions: {len(positions)}")
        for p in positions[:10]:
            log.info(f"  - {p.get('symbol')}: tokens={p.get('tokens')} @ entry={p.get('entry_price')}")
        return 0

    if args.force or (args.check and status['stale']):
        if args.check and status['stale']:
            log.warning("🚨 HEARTBEAT STALE — triggering emergency close")
        elif args.force:
            log.warning("🚨 FORCE close requested")
        result = close_all_live_positions(dry_run=args.dry_run)
        log.info(f"Emergency close result: closed={result['closed']} failed={result['failed']}")
        return 0 if result['failed'] == 0 else 1

    if args.check and not status['stale']:
        log.info("Heartbeat OK — no action needed")
        return 0

    return 0


if __name__ == '__main__':
    sys.exit(main())
