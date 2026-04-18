"""
reconcile.py — On-chain balance reconciliation (Sprint 2 Fase 4)

Verifica que portfolio.json esté sincronizado con los balances reales
on-chain de la wallet. Si hay divergencia crítica → activa kill switch +
loguea alert.

Solo opera cuando LIVE_TRADING_ENABLED=true. En paper mode,
portfolio.json es ground truth (no hay on-chain que reconciliar).

Usage:
    from agents.reconcile import check_reconciliation
    result = check_reconciliation()
    # result.ok: bool
    # result.discrepancies: list of dicts
    # result.kill_switch_triggered: bool

Integration: el orchestrator llama reconcile.check() cada N ciclos.

Filosofía:
- No reconcilia automáticamente (no sobreescribe nada)
- Detecta + reporta + activa kill switch
- Humano decide cómo resolver (stop bot, close all manually, etc.)
"""
from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger('reconcile')

# Tolerance: 0.5% por defecto (maneja dust, rounding, fees pequeñas)
DEFAULT_TOLERANCE_PCT = 0.5


@dataclass
class Discrepancy:
    symbol: str
    expected: float        # tokens según portfolio.json
    actual: float          # tokens según on-chain
    diff_pct: float        # abs((actual-expected)/expected) * 100
    severity: str          # 'info' | 'warning' | 'critical'


@dataclass
class ReconcileResult:
    ok: bool = True
    discrepancies: list = field(default_factory=list)
    kill_switch_triggered: bool = False
    checked_positions: int = 0
    on_chain_sol_balance: float = 0.0
    error: Optional[str] = None


def _get_tolerance_pct() -> float:
    try:
        return float(os.environ.get('RECONCILE_TOLERANCE_PCT', DEFAULT_TOLERANCE_PCT))
    except Exception:
        return DEFAULT_TOLERANCE_PCT


def _live_enabled() -> bool:
    return os.environ.get('LIVE_TRADING_ENABLED', 'false').lower() == 'true'


def check_reconciliation(portfolio_path: Optional[Path] = None,
                          trigger_kill_switch: bool = True) -> ReconcileResult:
    """
    Compara posiciones live en portfolio.json contra balances on-chain.

    Args:
      portfolio_path: default DATA_DIR/portfolio.json
      trigger_kill_switch: si True, activa kill switch automático en critical

    Returns ReconcileResult. Si hay critical discrepancy → kill_switch_triggered=True.
    """
    result = ReconcileResult()

    if not _live_enabled():
        log.debug("reconcile: LIVE_TRADING_ENABLED=false — skipping (paper mode)")
        return result

    # Lazy imports (evita romper si deps no están en paper)
    try:
        import solana_rpc
        import safety
    except Exception as e:
        result.error = f"dep_import_failed: {e}"
        return result

    # Resolver path del portfolio
    if portfolio_path is None:
        try:
            from pathlib import Path as _P
            portfolio_path = _P(__file__).parent / "data" / "portfolio.json"
        except Exception as e:
            result.error = f"portfolio_path_error: {e}"
            return result

    if not portfolio_path.exists():
        result.error = "portfolio.json not found"
        return result

    try:
        portfolio = json.loads(portfolio_path.read_text())
    except Exception as e:
        result.error = f"portfolio_read_error: {e}"
        return result

    # Wallet pubkey
    wallet_pubkey = os.environ.get('HOT_WALLET_ADDRESS', '').strip()
    if not wallet_pubkey:
        result.error = "HOT_WALLET_ADDRESS not set"
        return result

    rpc = solana_rpc.get_rpc()

    # Record SOL balance for context
    try:
        result.on_chain_sol_balance = rpc.get_balance_sol(wallet_pubkey)
    except Exception as e:
        log.warning(f"reconcile: get_balance failed: {e}")

    # Solo chequear posiciones con mode="live" (las paper son separadas y no tienen on-chain counterpart)
    live_positions = [
        p for p in portfolio.get('positions', [])
        if p.get('status') == 'open' and p.get('mode') == 'live'
    ]
    result.checked_positions = len(live_positions)

    if not live_positions:
        log.debug("reconcile: no live positions to check")
        return result

    # Lazy import de MINT_MAP de executor
    try:
        import executor
        mint_map = getattr(executor, 'MINT_MAP', {})
    except Exception:
        mint_map = {}

    tolerance_pct = _get_tolerance_pct()

    for pos in live_positions:
        sym = pos.get('symbol', '?')
        expected_tokens = float(pos.get('tokens', 0) or 0)

        mint = mint_map.get(sym)
        if not mint:
            log.warning(f"reconcile: {sym} no MINT_MAP entry — skip")
            continue

        try:
            actual_tokens = rpc.get_token_balance(wallet_pubkey, mint)
        except Exception as e:
            log.warning(f"reconcile: {sym} get_token_balance failed: {e}")
            continue

        # Comparar
        if expected_tokens <= 0:
            continue   # posición malformada en portfolio, skip
        diff_pct = abs(actual_tokens - expected_tokens) / expected_tokens * 100

        severity = (
            'critical' if diff_pct > tolerance_pct * 10  # >5% = crítico
            else 'warning' if diff_pct > tolerance_pct   # >0.5% = warning
            else 'info'
        )

        if severity != 'info':
            d = Discrepancy(
                symbol=sym,
                expected=expected_tokens,
                actual=actual_tokens,
                diff_pct=diff_pct,
                severity=severity,
            )
            result.discrepancies.append(d)
            log_fn = log.error if severity == 'critical' else log.warning
            log_fn(
                f"reconcile [{severity}]: {sym} expected={expected_tokens:.6f} "
                f"actual={actual_tokens:.6f} diff={diff_pct:.2f}%"
            )

    # Cualquier crítico → kill switch
    criticals = [d for d in result.discrepancies if d.severity == 'critical']
    if criticals:
        result.ok = False
        if trigger_kill_switch:
            reason = (
                f"reconcile_critical_{len(criticals)}_discrepancies: "
                + ', '.join(f"{d.symbol}({d.diff_pct:.1f}%)" for d in criticals[:3])
            )
            try:
                safety.activate_kill_switch(reason)
                result.kill_switch_triggered = True
                log.error(f"🛑 KILL SWITCH ACTIVATED — {reason}")
            except Exception as e:
                log.error(f"reconcile: failed to activate kill switch: {e}")

    # Warnings-only también hacen ok=False pero sin kill switch
    if result.discrepancies and result.ok:
        result.ok = False

    return result


def summary(result: ReconcileResult) -> str:
    """Human-readable one-line summary para logging."""
    if result.error:
        return f"reconcile ERROR: {result.error}"
    if not result.discrepancies:
        return f"reconcile OK: {result.checked_positions} positions aligned (SOL balance: {result.on_chain_sol_balance:.4f})"
    parts = [f"reconcile: {len(result.discrepancies)} discrepancies"]
    if result.kill_switch_triggered:
        parts.append("KILL_SWITCH_ACTIVATED")
    return " ".join(parts) + " — " + ", ".join(
        f"{d.symbol}[{d.severity}]:{d.diff_pct:.1f}%" for d in result.discrepancies[:5]
    )
