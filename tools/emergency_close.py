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
        # v2.12.6 pre-flight: skip orphan (position claims more tokens than wallet holds)
        try:
            _wallet_sol = rpc.get_balance_sol(w.pubkey) if sym == "SOL" else None
            _FEE_RESERVE = 0.005
            if _wallet_sol is not None and tokens > (_wallet_sol - _FEE_RESERVE):
                log.error(f"🛑 ORPHAN {sym}: wallet has {_wallet_sol:.6f} SOL but position claims {tokens:.6f} — skipping broadcast")
                entry['status'] = 'orphan_insufficient_onchain_balance'
                entry['wallet_sol'] = _wallet_sol
                entry['expected_tokens'] = tokens
                result['positions'].append(entry)
                continue  # orphan handled: NOT counted as broadcast failure
        except Exception as _pf_err:
            log.warning(f"pre-flight check {sym} failed (non-fatal): {_pf_err}")
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
                # v2.12.13 persist portfolio: update portfolio.json inmediatamente
                # para que orchestrator next cycle no vea position stale → orphan → kill switch
                try:
                    from pathlib import Path as _P
                    import json as _j
                    from datetime import datetime as _dt, timezone as _tz
                    _pf_path = _P(BOT_ROOT) / "agents/data/portfolio.json"
                    _pf = _j.loads(_pf_path.read_text())
                    _hist_path = _P(BOT_ROOT) / "agents/data/trade_history.json"
                    _hist_raw = _j.loads(_hist_path.read_text()) if _hist_path.exists() else []
                    _history = _hist_raw['trades'] if isinstance(_hist_raw, dict) else _hist_raw
                    _wrap = isinstance(_hist_raw, dict)
                    _margin = float(pos.get('margin_usd', 2.0))
                    _pnl_usd = float(entry['usdc_received']) - _margin
                    _tokens = float(pos.get('tokens', 0))
                    _exit_price = entry['usdc_received'] / _tokens if _tokens else 0
                    _new_pos_list = []
                    for _p in _pf.get('positions', []):
                        if _p.get('id') == pos.get('id'):
                            _p['status'] = 'closed'
                            _p['close_time'] = _dt.now(_tz.utc).isoformat()
                            _p['close_reason'] = 'emergency_close_v2.12.13'
                            _p['close_price'] = round(_exit_price, 8)
                            _p['pnl_usd'] = round(_pnl_usd, 4)
                            _p['pnl_pct'] = round(_pnl_usd / _margin * 100, 4) if _margin else 0
                            _p['fee_exit'] = 0
                            _p['tx_signature_close'] = res.signature
                            _history.append(dict(_p))
                        else:
                            _new_pos_list.append(_p)
                    _pf['positions'] = _new_pos_list
                    _pf['capital_usd'] = round(_pf.get('capital_usd', 0) + entry['usdc_received'], 4)
                    _pf['total_trades'] = _pf.get('total_trades', 0) + 1
                    if _pnl_usd > 0:
                        _pf['wins'] = _pf.get('wins', 0) + 1
                    else:
                        _pf['losses'] = _pf.get('losses', 0) + 1
                    _pf_path.write_text(_j.dumps(_pf, indent=2))
                    if _wrap:
                        _hist_raw['trades'] = _history
                        _hist_raw['last_updated'] = _dt.now(_tz.utc).isoformat()
                        _hist_path.write_text(_j.dumps(_hist_raw, indent=2))
                    else:
                        _hist_path.write_text(_j.dumps(_history, indent=2))
                    log.info(f"    💾 persisted portfolio: capital_usd=${_pf['capital_usd']} pnl={_pnl_usd:+.4f}")
                except Exception as _ps_err:
                    log.error(f"    ⚠️ portfolio persist failed (manual reconcile needed): {_ps_err}")
            else:
                entry['status'] = f'failed: {res.error}'
                result['failed'] += 1
                log.error(f"❌ {sym} close failed: {res.error}")
        except Exception as e:
            entry['status'] = f'exception: {e}'
            result['failed'] += 1
            log.error(f"❌ {sym} close exception: {e}")

        result['positions'].append(entry)

    # v2.12.3-live: kill switch tolerance — require 3 consecutive failures before blocking bot.
    # Transient RPC issues shouldn't permanently block. Orphan reconcile is not counted as fail.
    MAX_CONSECUTIVE_FAILS = 3
    close_failures_path = Path('/tmp/solana_live_close_failures')
    try:
        import safety
        # Count orphans vs real fails
        orphan_count = sum(1 for p in result.get('positions', []) if 'orphan' in str(p.get('status', '')))
        real_fails = result['failed'] - orphan_count
        if result['closed'] > 0 or result['failed'] == 0:
            # Success path — reset counter
            if close_failures_path.exists():
                close_failures_path.unlink()
        elif real_fails > 0:
            # Transient fail — increment counter
            prior = 0
            try:
                if close_failures_path.exists():
                    prior = int(close_failures_path.read_text().strip() or 0)
            except Exception:
                prior = 0
            new_count = prior + 1
            close_failures_path.write_text(str(new_count))
            log.warning(f"emergency_close consecutive fails: {new_count}/{MAX_CONSECUTIVE_FAILS}")
            if new_count >= MAX_CONSECUTIVE_FAILS:
                safety.activate_kill_switch(f"emergency_close: {new_count} consecutive fails — closed={result['closed']} failed={result['failed']}")
                log.warning(f"Kill switch activado tras {new_count} fallos consecutivos")
                close_failures_path.unlink()  # reset
            else:
                log.info(f"Kill switch NOT activated — tolerating transient fail ({new_count}/{MAX_CONSECUTIVE_FAILS})")
        elif orphan_count > 0:
            log.warning(f"emergency_close: {orphan_count} orphan(s) detected — NOT counting as fail, NOT activating kill switch")
    except Exception as e:
        log.error(f"tolerance/kill-switch logic failed: {e}")

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
