from safe_io import atomic_write_json
import os
#!/usr/bin/env python3
"""
🎯 Solana Trading Bot - Orchestrator Modular
Ejecuta el ciclo completo con Smart Rotation + Daily Target
"""
import sys
import json
import time
try:
    import paperclip_client as pc
    _PAPERCLIP = True
except ImportError:
    _PAPERCLIP = False
from datetime import datetime, timezone
from pathlib import Path
try:
    import performance_tracker as pt
    _PERF_TRACKER = True
except ImportError:
    _PERF_TRACKER = False
try:
    import candle_aggregator as ca
    _CANDLE_AGG = True
except ImportError:
    _CANDLE_AGG = False

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
sys.path.insert(0, str(BASE_DIR))

# ── Helpers para trade_history.json (puede ser dict o list) ──────────────────
# ── 30m Candle Close Detection ───────────────────────────────────────────────
_last_candle_bucket: int = -1   # floor(unix_ts / 1800)
_last_strategy_ts: float    = 0.0   # unix ts of last AI strategy generation
_last_strategy_fg: int      = -1    # fear_greed value at last generation
_last_strategy_prices: dict = {}    # {symbol: price} at last generation
_STRATEGY_CACHE_SEC: int    = int(os.environ.get('STRATEGY_CACHE_SEC', '900'))

def _is_new_candle_30m() -> bool:
    """
    Retorna True si acaba de cerrar una vela de 30 minutos.
    La vela cierra en :00 y :30 de cada hora (UTC).
    Solo retorna True UNA VEZ por cierre de vela.
    """
    global _last_candle_bucket
    import time as _time
    now_ts = int(_time.time())
    bucket = now_ts // 1800
    if bucket != _last_candle_bucket:
        _last_candle_bucket = bucket
        return True
    return False


def _strategy_needs_refresh(current_fg: int, current_prices: dict, force: bool = False) -> bool:
    """
    Returns True if AI Strategy should regenerate signals.
    Refresh if: forced (new 30m candle), cache expired (>15min),
    F&G changed >=5 pts, or any token moved >=1% since last generation.
    Updates module-level tracking vars when returning True.
    """
    global _last_strategy_ts, _last_strategy_fg, _last_strategy_prices
    import time as _t
    now = _t.time()
    needs = False
    if force or (now - _last_strategy_ts) >= _STRATEGY_CACHE_SEC:
        needs = True
    elif abs(current_fg - _last_strategy_fg) >= 5:
        needs = True
    else:
        for sym, price in current_prices.items():
            last_p = _last_strategy_prices.get(sym, price)
            if last_p and price and abs((price - last_p) / last_p * 100) >= 1.0:
                needs = True
                break
    if needs:
        _last_strategy_ts = now
        _last_strategy_fg = current_fg
        _last_strategy_prices = dict(current_prices)
    return needs


def _load_trade_history() -> list:
    """Carga trade_history.json y retorna SIEMPRE una lista de trades."""
    f = DATA_DIR / "trade_history.json"
    if not f.exists():
        return []
    data = json.loads(f.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("trades", [])
    return []

def _save_trade_history(trades: list):
    """Guarda trade_history.json en formato dict con timestamp."""
    f = DATA_DIR / "trade_history.json"
    data = {"trades": trades, "last_updated": datetime.now(timezone.utc).isoformat()}
    f.write_text(json.dumps(data, indent=2))

import logging

LOG_FILE = Path.home() / ".config" / "solana-jupiter-bot" / "modular.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Rotación de logs (si > 50MB) ─────────────────────────────────────────────
def _rotate_log_if_needed():
    try:
        if LOG_FILE.exists() and LOG_FILE.stat().st_size >= 50 * 1024 * 1024:
            old = LOG_FILE.with_suffix(".log.old")
            LOG_FILE.rename(old)
    except Exception:
        pass

_rotate_log_if_needed()

# ── Logging unificado: formato [YYYY-MM-DD HH:MM:SS] [NIVEL] mensaje ─────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),            # stdout → capturado por watchdog en LOG_FILE
    ],
)
log = logging.getLogger("orchestrator")

# ─── Circuit Breaker ─────────────────────────────────────────────────────────
# Detecta loops de emergency closes y detiene el bot automáticamente.
CIRCUIT_BREAKER_FILE = DATA_DIR / "circuit_breaker_state.json"
CIRCUIT_BREAKER_LIMIT = 10  # Máximo de emergency closes POR CICLO antes de detener el bot


def _circuit_breaker_check_and_record(n_emergency_closes: int) -> bool:
    """
    Registra el número de emergency closes del ciclo actual.
    Si supera CIRCUIT_BREAKER_LIMIT → escribe STOP_TRADING, pausa el bot y retorna True.
    Retorna True si el circuit breaker se activó (bot debe detenerse).
    """
    if n_emergency_closes < CIRCUIT_BREAKER_LIMIT:
        return False

    log.error("=" * 60)
    log.error(f"🔴 CIRCUIT BREAKER ACTIVADO: {n_emergency_closes} emergency closes en un ciclo")
    log.error(f"   Límite: {CIRCUIT_BREAKER_LIMIT} — BOT DETENIDO AUTOMÁTICAMENTE")
    log.error("=" * 60)

    # Crear STOP_TRADING para que executor no abra más posiciones
    stop_file = DATA_DIR / "STOP_TRADING"
    stop_file.write_text(
        f"Circuit breaker activado: {n_emergency_closes} emergency closes en un ciclo.\n"
        f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n"
        f"Elimina este archivo para reanudar el bot.\n"
    )

    # Registrar estado del circuit breaker
    cb_state = {
        "triggered": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "emergency_closes_in_cycle": n_emergency_closes,
        "limit": CIRCUIT_BREAKER_LIMIT,
        "action": "STOP_TRADING file creado, portfolio pausado",
    }
    CIRCUIT_BREAKER_FILE.write_text(json.dumps(cb_state, indent=2))

    # Pausar el portfolio
    portfolio_file = DATA_DIR / "portfolio.json"
    try:
        if portfolio_file.exists():
            portfolio_data = json.loads(portfolio_file.read_text())
            portfolio_data["status"] = "PAUSED"
            portfolio_data["last_updated"] = datetime.now(timezone.utc).isoformat()
            portfolio_file.write_text(json.dumps(portfolio_data, indent=2))
            log.error("   ✅ Portfolio marcado como PAUSED")
    except Exception as e:
        log.error(f"   ⚠️ No se pudo pausar portfolio: {e}")

    return True


def import_agents():
    """Importa módulos de agentes."""
    import market_data
    import risk_manager as rm
    import strategy
    import executor
    import reporter
    try:
        import daily_target
    except ImportError:
        daily_target = None
        log.warning("⚠️  daily_target.py no encontrado")
    return market_data, rm, strategy, executor, reporter, daily_target


_cycle_global_count = 0

def run_cycle(safe=True, debug=False):
    """Ejecuta un ciclo completo del sistema."""
    global _cycle_global_count
    _cycle_global_count += 1
    run_cycle._cycle_count = _cycle_global_count
    cycle_start = time.time()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.info(f"📁 [run_cycle] DATA_DIR={DATA_DIR} | PID={os.getpid()}")

    # v2.9.0-live Sprint 1: Safety gates antes de cualquier actividad
    try:
        import safety as _safety
        _ks_active, _ks_reason = _safety.is_kill_switch_active()
        if _ks_active:
            log.error(f"🛑 KILL SWITCH ACTIVO: {_ks_reason} — ciclo omitido")
            return {"status": "KILL_SWITCH_ACTIVE", "reason": _ks_reason}
        # Chequear limite diario de perdida
        _pf_file = DATA_DIR / "portfolio.json"
        _pf_data = None
        if _pf_file.exists():
            try: _pf_data = json.loads(_pf_file.read_text())
            except Exception: _pf_data = None
        _hit, _todays_pnl = _safety.check_daily_loss(_pf_data)
        if _hit:
            log.error(f"🛑 MAX_DAILY_LOSS_USD alcanzado (${_todays_pnl:.2f}) — activando kill switch")
            _safety.activate_kill_switch(f"daily_loss_exceeded_{_todays_pnl:.2f}")
            return {"status": "DAILY_LOSS_LIMIT_HIT", "todays_pnl": _todays_pnl}
    except Exception as _se:
        log.warning(f"safety check error (non-fatal, continuando): {_se}")
    
    # v2.10.0-live Sprint 2 Fase 5: heartbeat para watchdog externo (emergency_close.py)
    try:
        _hb_path = os.environ.get("HEARTBEAT_FILE", "/tmp/solana_live_heartbeat")
        with open(_hb_path, "w") as _hf:
            _hf.write(str(time.time()))
    except Exception as _hb_err:
        log.debug(f"heartbeat write error (non-fatal): {_hb_err}")

    # v2.10.0-live Sprint 2 Fase 4: on-chain reconciliation cada 10 ciclos (solo live)
    if _cycle_global_count % 10 == 0 and os.environ.get("LIVE_TRADING_ENABLED", "false").lower() == "true":
        try:
            import reconcile
            _rec = reconcile.check_reconciliation(trigger_kill_switch=True)
            log.info(f"🔄 reconcile: {reconcile.summary(_rec)}")
        except Exception as _rec_err:
            log.warning(f"reconcile error (non-fatal): {_rec_err}")

    log.info("=" * 60)
    log.info(f"🔄 CICLO INICIADO — {now}")
    mode_label = "📄 PAPER" if safe else "🔴 LIVE"
    log.info(f"   Modo: {mode_label}")
    log.info("=" * 60)

    # v2.9.0: auto-reset daily_target_state si la fecha cambió (evita queda stale)
    try:
        _dt_state_file = DATA_DIR / "daily_target_state.json"
        if _dt_state_file.exists():
            _dts = json.loads(_dt_state_file.read_text())
            _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if _dts.get("date") != _today:
                _pf_file = DATA_DIR / "portfolio.json"
                _cap = 0.0
                if _pf_file.exists():
                    try:
                        _cap = float(json.loads(_pf_file.read_text()).get("capital_usd", 0))
                    except Exception:
                        _cap = 0.0
                _prev_date = _dts.get("date")
                _dts.update({
                    "date": _today,
                    "starting_capital": _cap,
                    "current_pnl_pct": 0.0,
                    "target_reached": False,
                })
                _dt_state_file.write_text(json.dumps(_dts, indent=2))
                log.info(f"🗓️  daily_target_state reseteado: {_prev_date} → {_today} (capital: ${_cap:.2f})")
    except Exception as _e_dt:
        log.warning(f"daily_target reset error (non-fatal): {_e_dt}")
    
    results = {}
    _cycle_emergency_closes = 0  # Circuit breaker: contador de emergency closes este ciclo
    
    # Importar agentes
    try:
        md, rm, st, ex, rp, dt = import_agents()
    except ImportError as e:
        log.error(f"❌ Error importando agentes: {e}")
        return {"status": "IMPORT_ERROR"}
    
    # Paso 1: Market Data
    log.info("━" * 40)
    log.info("🌐 [1/6] Market Data")
    try:
        result = md.run(debug=debug)
        results["market_data"] = {"ok": True, "prices": result.get("prices_ok", 0)}
        log.info(f"   → {result.get('prices_ok', 0)} precios obtenidos")
    except Exception as e:
        log.error(f"   ❌ Error: {e}")
        results["market_data"] = {"ok": False, "error": str(e)}
        return results
    
    # Paso 2: Risk Manager
    log.info("━" * 40)
    log.info("🛡️  [2/6] Risk Manager")
    try:
        result = rm.run(debug=debug)
        results["risk_manager"] = {"ok": True}
        log.info(f"   → Status: {result.get('portfolio_status', 'ACTIVE')}")
    except Exception as e:
        log.warning(f"   ⚠️ Error: {e}")
        results["risk_manager"] = {"ok": False}
    
    # Detectar modo ANTES de llamar strategy — determina qué estrategias activar
    try:
        import martingale_engine as _wild_me_early
        _wild_active = _wild_me_early.is_active()
    except Exception:
        _wild_active = False
    log.info(f"   🎯 Modo activo: {'COMBO + Wild Mode' if _wild_active else 'Pure Strategy'}")

    # Paso 3: Strategy (Technical + AI)
    log.info("━" * 40)
    log.info("🧠 [3/6] Strategy")
    try:
        result = st.run(debug=debug, wild_mode=_wild_active)
        n_signals = result.get("total_signals", 0)
        results["strategy"] = {"ok": True, "signals": n_signals}
        log.info(f"   → {n_signals} señal(es) técnicas")
    except Exception as e:
        log.warning(f"   ⚠️ Error: {e}")
        results["strategy"] = {"ok": False}
    
    # Paso 3b: AI Strategy — genera señales LLM en condiciones extremas
    try:
        market_file = DATA_DIR / "market_latest.json"
        portfolio_file_3b = DATA_DIR / "portfolio.json"
        if market_file.exists():
            mkt = json.loads(market_file.read_text())
            fg = mkt.get("fear_greed", {})
            fg_val = fg.get("value", 50) if isinstance(fg, dict) else 50
            
            # Solo invocar AI strategy si hubo cambio relevante (F&G, precio, vela, o >15min)
            _curr_prices_s = {s: float(v.get('price', 0)) for s, v in mkt.get('tokens', {}).items() if v.get('price')}
            _force_s = _is_new_candle_30m()
            if _strategy_needs_refresh(fg_val, _curr_prices_s, force=_force_s):
                import ai_strategy
                port_3b = json.loads(portfolio_file_3b.read_text()) if portfolio_file_3b.exists() else {}
                research_file = DATA_DIR / "research_latest.json"
                res_3b = json.loads(research_file.read_text()) if research_file.exists() else {}
                
                ai_result = ai_strategy.generate_signals_with_llm(mkt, res_3b, port_3b)
                # Extract signals list from result (can be dict with "signals" key or list)
                if isinstance(ai_result, dict):
                    ai_signals = ai_result.get("signals", [])
                elif isinstance(ai_result, list):
                    ai_signals = ai_result
                else:
                    ai_signals = []
                
                # Pre-save filter: reject template/placeholder signals
                ai_signals = [
                    s for s in ai_signals
                    if s.get("confidence", 0) >= 0.55
                    and s.get("symbol", "") not in ("", "TOKEN")
                    and "|" not in s.get("direction", "")
                ]
                if ai_signals and len(ai_signals) > 0:
                    # Guardar en strategy_llm.json para que el executor las use
                    llm_out = DATA_DIR / "strategy_llm.json"
                    llm_out.write_text(json.dumps({
                        "signals": ai_signals,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "fear_greed": fg_val,
                        "source": "ai_strategy_extreme"
                    }, indent=2))
                    log.info(f"   🤖 AI Strategy: {len(ai_signals)} señales LLM (F&G={fg_val})")
                else:
                    log.info(f"   🤖 AI Strategy: 0 señales (F&G={fg_val})")
    except Exception as e:
        log.warning(f"   ⚠️ AI Strategy error: {e}")
    
    # Paso 3c: UPDATE POSITIONS — actualizar precios y P&L ANTES de decisiones
    log.info("━" * 40)
    log.info("🔄 [3c/6] Position Updater")
    try:
        portfolio_file = DATA_DIR / "portfolio.json"
        market_file = DATA_DIR / "market_latest.json"
        if portfolio_file.exists() and market_file.exists():
            portfolio_data = json.loads(portfolio_file.read_text())
            market_data_upd = json.loads(market_file.read_text())
            history = _load_trade_history()
            # Actualizar TODAS las posiciones abiertas con precios actuales
            # M5: Removed - executor.run() already calls paper_update_positions
            closed_updated = []
            if closed_updated:
                log.info(f"   🔄 {len(closed_updated)} posición(es) cerradas por SL/TP/Liq")
                for cp in closed_updated:
                    log.info(f"      💀 {cp.get('symbol')} | P&L: ${cp.get('pnl_usd',0):.2f} | Razón: {cp.get('close_reason')}")
                _save_trade_history(history)
            else:
                log.info(f"   ✅ Posiciones actualizadas — sin cierres automáticos")
            ex.save_portfolio(portfolio_data)
            # Log P&L actual de cada posición
            for p in portfolio_data.get("positions", []):
                if p.get("status") == "open":
                    log.info(f"      📊 {p.get('symbol')} {p.get('direction')} | Entry: \${p.get('entry_price',0):.6f} | Current: \${p.get('current_price',0):.6f} | P&L: \${p.get('pnl_usd',0):.2f} ({p.get('pnl_pct',0):.2f}%)")
        results["position_updater"] = {"ok": True}
    except Exception as e:
        log.warning(f"   ⚠️ Position Updater error: {e}")
        results["position_updater"] = {"ok": False, "error": str(e)}

    # Paso 4: Executor
    log.info("━" * 40)
    log.info("⚡ [4/6] Executor")

    # Definir portfolio_file aquí para que esté disponible en todos los pasos siguientes
    portfolio_file = DATA_DIR / "portfolio.json"

    # Fix 3: Snapshot de posiciones ANTES del executor para detectar recién abiertas
    pre_exec_symbols = set()
    try:
        pre_portfolio = json.loads(portfolio_file.read_text()) if portfolio_file.exists() else {}
        pre_exec_symbols = {p["symbol"] for p in pre_portfolio.get("positions", []) if p.get("status") == "open"}
    except Exception:
        pass

    try:
        result = ex.run(safe=safe, debug=debug)
        results["executor"] = {"ok": True}
    except Exception as e:
        log.warning(f"   ⚠️ Error: {e}")
        results["executor"] = {"ok": False}

    # Fix 3: Detectar símbolos recién abiertos en este ciclo
    just_opened_symbols = set()
    try:
        post_portfolio = json.loads(portfolio_file.read_text()) if portfolio_file.exists() else {}
        post_exec_symbols = {p["symbol"] for p in post_portfolio.get("positions", []) if p.get("status") == "open"}
        just_opened_symbols = post_exec_symbols - pre_exec_symbols
        if just_opened_symbols:
            log.info(f"   🛡️ Recién abiertas (protegidas de cierre este ciclo): {just_opened_symbols}")
    except Exception:
        pass

    # Paso 4b: Smart Rotation
    log.info("━" * 40)
    log.info("🔄 [4b/6] Smart Rotation")
    
    portfolio_data = {}
    if portfolio_file.exists():
        portfolio_data = json.loads(portfolio_file.read_text())
    
    stale_positions = []
    if portfolio_data:
        try:
            # OPTIMIZADO 2026-03-31 (orden de Ender): max_hours 96→24, improvement_hours 36→12
            # Rotar posiciones más rápido para iterar y aprender
            stale_positions = rm.check_stale_losing_positions(
                portfolio_data, max_hours=24, improvement_hours=12
            )
        except Exception as e:
            log.warning(f"   ⚠️ Error: {e}")
        
        if stale_positions:
            log.info(f"   → {len(stale_positions)} posición(es) a cerrar:")
            for pos in stale_positions:
                log.info(f"     - {pos['symbol']}: {pos['reason']}")
            # FIX #2: Excluir posiciones recién abiertas del Smart Rotation (cooldown same-cycle)
            symbols_to_close = [p["symbol"] for p in stale_positions if p["symbol"] not in just_opened_symbols]
            skipped_fresh = [p["symbol"] for p in stale_positions if p["symbol"] in just_opened_symbols]
            if skipped_fresh:
                log.info(f"   🛡️ Smart Rotation cooldown — skipped recién abiertas: {skipped_fresh}")
            if symbols_to_close:
                try:
                    market_data = json.loads((DATA_DIR / "market_latest.json").read_text()) if (DATA_DIR / "market_latest.json").exists() else {}
                    history = _load_trade_history()
                    closed = ex.close_positions_emergency(portfolio_data, symbols_to_close, market_data, history, reason="SMART_ROTATION")
                    for pos in closed:
                        ex.record_emergency_cooldown(pos.get("symbol", "")) if pos.get("symbol") else None
                    ex.save_portfolio(portfolio_data)
                    _save_trade_history(history)
                    log.info(f"   ✅ Cerradas {len(closed)} posición(es) por Smart Rotation")
                except Exception as e:
                    log.warning(f"   ⚠️ Error cerrando posiciones: {e}")
            results["smart_rotation"] = {"ok": True, "closed": len(symbols_to_close)}
        else:
            log.info("   → ✅ Sin posiciones a rotar")
            results["smart_rotation"] = {"ok": True, "closed": 0}
    else:
        log.info("   → ✅ Sin datos de portfolio")
        results["smart_rotation"] = {"ok": False}
    
    # Paso 4c: Daily Target
    log.info("━" * 40)
    log.info("🎯 [4c/6] Daily Target")
    
    if not dt:
        log.warning("   ⚠️ Módulo daily_target no disponible")
        results["daily_target"] = {"ok": False}
    else:
        signals_data = {}
        signals_file = DATA_DIR / "signals_latest.json"
        if signals_file.exists() and signals_file.stat().st_size > 0:
            try:
                signals_data = json.loads(signals_file.read_text())
            except (json.JSONDecodeError, Exception) as e:
                log.warning(f"   ⚠️ signals_latest.json corrupt or empty: {e}")
                signals_data = {}
        
        try:
            target_result = dt.evaluate_daily_target(portfolio_data, signals_data)
            
            log.info(f"   → P&L: {target_result['daily_pnl_pct']:.2f}%")
            log.info(f"   → Target: {target_result['target_pct']:.1f}%")
            log.info(f"   → RSI: {target_result['market_conditions']['avg_rsi']:.1f}")
            
            if target_result["should_close_all"]:
                log.info(f"   🚨 CERRAR TODO: {target_result['close_reason']}")
                # Ejecutar cierre real de todas las posiciones
                try:
                    # Fix 3: Excluir posiciones recién abiertas del cierre por Daily Target
                    open_symbols = [p["symbol"] for p in portfolio_data.get("positions", [])
                                    if p.get("status") == "open" and p["symbol"] not in just_opened_symbols]
                    if open_symbols:
                        market_data = json.loads((DATA_DIR / "market_latest.json").read_text()) if (DATA_DIR / "market_latest.json").exists() else {}
                        history = _load_trade_history()
                        # FIX: Usar reason específica "DAILY_TARGET" en lugar de "EMERGENCY_CLOSE"
                        close_reason_label = f"DAILY_TARGET: {target_result['close_reason'][:60]}"
                        closed = ex.close_positions_emergency(portfolio_data, open_symbols, market_data, history, reason=close_reason_label)
                        for pos in closed:
                            ex.record_emergency_cooldown(pos.get("symbol", "")) if pos.get("symbol") else None
                        _cycle_emergency_closes += len(closed)
                        # REMOVED: lock file was blocking all trading for the rest of the day
                        # The bot should keep operating as long as conditions are valid
                        # Positions close when their individual TP/SL is hit
                        # target_hit_file = DATA_DIR / "DAILY_TARGET_HIT"
                        # target_hit_file.write_text(...)
                        log.info(f"   ✅ Cerradas {len(closed)} posición(es) por Daily Target — continuando...")
                        ex.save_portfolio(portfolio_data)
                        _save_trade_history(history)
                        log.info(f"   ✅ Cerradas {len(closed)} posición(es) por Daily Target")
                    elif just_opened_symbols:
                        log.info(f"   🛡️ Skipped cierre — todas las posiciones son recién abiertas")
                except Exception as e:
                    log.warning(f"   ⚠️ Error cerrando posiciones: {e}")
                results["daily_target"] = {
                    "ok": True,
                    "should_close": True,
                    "reason": target_result["close_reason"]
                }
            else:
                log.info("   ✅ Mantener posiciones")
                results["daily_target"] = {"ok": True, "should_close": False}
        except Exception as e:
            log.warning(f"   ⚠️ Error: {e}")
            results["daily_target"] = {"ok": False}
    
    # Paso 4d: Position Decisions (LLM + Quant)
    log.info("━" * 40)
    log.info("🧠 [4d/6] Position Decisions — LLM + Quant")
    try:
        market_data_for_dec = {}
        market_file = DATA_DIR / "market_latest.json"
        if market_file.exists():
            market_data_for_dec = json.loads(market_file.read_text())

        research_data = {}
        research_file = DATA_DIR / "research_latest.json"
        if research_file.exists():
            research_data = json.loads(research_file.read_text())

        # FIX 4 (audit): Staleness check — never feed stale data to LLM-based position decisions
        _stale_dec = False
        try:
            if market_data_for_dec.get("_stale"):
                _stale_dec = True
            else:
                _ts_dec = market_data_for_dec.get("timestamp", "")
                if _ts_dec:
                    _age_sec = (datetime.now(timezone.utc) - datetime.fromisoformat(str(_ts_dec).replace("Z", "+00:00"))).total_seconds()
                    if _age_sec > 300:
                        _stale_dec = True
                        log.warning(f"   ⚠️ Position decisions: market data is {_age_sec:.0f}s old (>300s), skipping LLM eval")
        except Exception as _e_st:
            log.debug(f"staleness check error: {_e_st}")

        if portfolio_data and portfolio_data.get("positions") and not _stale_dec:
            decisions = rm.evaluate_position_decision(portfolio_data, market_data_for_dec, research_data)
            # Fix 3: Excluir posiciones recién abiertas de recomendaciones de cierre
            # FIX 2026-04-05: Umbrales MUY estrictos — solo cerrar si confidence >= 0.75 AND hours >= 20min
            close_recs  = [d for d in decisions if d["action"] == "CLOSE"
                           and d["confidence"] >= 0.75
                           and d.get("hours_open", 0) >= 0.33  # 20 min minimum
                           and d["quant_score"] >= 60
                           and d["symbol"] not in just_opened_symbols]
            reduce_recs = [d for d in decisions if d["action"] == "REDUCE" and d["confidence"] >= 0.80]

            if close_recs:
                log.info(f"   🔴 CERRAR ({len(close_recs)}): {', '.join(d['symbol'] for d in close_recs)}")
                # Ejecutar cierre real
                try:
                    close_symbols = [d["symbol"] for d in close_recs]
                    market_data = json.loads((DATA_DIR / "market_latest.json").read_text()) if (DATA_DIR / "market_latest.json").exists() else {}
                    history = _load_trade_history()
                    # FIX: Usar reason específica para Position Decision
                    closed = ex.close_positions_emergency(portfolio_data, close_symbols, market_data, history, reason="POSITION_DECISION")
                    for pos in closed:
                        ex.record_emergency_cooldown(pos.get("symbol", "")) if pos.get("symbol") else None
                    ex.save_portfolio(portfolio_data)
                    _save_trade_history(history)
                    log.info(f"   ✅ Cerradas {len(closed)} posición(es) por Position Decision")
                except Exception as e:
                    log.warning(f"   ⚠️ Error cerrando posiciones: {e}")
            if reduce_recs:
                log.info(f"   🟡 REDUCIR ({len(reduce_recs)}): {', '.join(d['symbol'] for d in reduce_recs)}")
                # FIX 1.7: Ejecutar REDUCE — cerrar 50% de la posicion y mover SL a breakeven
                try:
                    for rec in reduce_recs:
                        symbol = rec["symbol"]
                        for pos in portfolio_data.get("positions", []):
                            if pos.get("symbol") == symbol and pos.get("status") == "open":
                                original_notional = pos.get("notional_value", 0)
                                original_margin = pos.get("margin_usd", 0)
                                if original_notional <= 0 or original_margin <= 0:
                                    continue
                                # Reducir 50%
                                reduce_frac = 0.5
                                reduced_notional = original_notional * reduce_frac
                                reduced_margin = original_margin * reduce_frac
                                # FIX C (2026-04-18): include slippage in REDUCE exit fee
                                fee_exit = reduced_notional * (ex.TAKER_FEE + ex.get_slippage(symbol))
                                partial_pnl = pos.get("pnl_usd", 0) * reduce_frac - fee_exit
                                returned = max(0, reduced_margin + partial_pnl)
                                portfolio_data["capital_usd"] = round(portfolio_data["capital_usd"] + returned, 2)
                                # Actualizar posicion restante
                                pos["notional_value"] = round(original_notional - reduced_notional, 2)
                                pos["margin_usd"] = round(original_margin - reduced_margin, 2)
                                pos["size_usd"] = pos["notional_value"]
                                # FIX C (2026-04-18): apportion fee_entry on remaining
                                pos["fee_entry"] = round(pos.get("fee_entry", 0) * (1 - reduce_frac), 4)
                                if pos.get("tokens", 0) > 0:
                                    pos["tokens"] = round(pos["tokens"] * (1 - reduce_frac), 8)
                                # Mover SL a breakeven
                                pos["sl_price"] = pos["entry_price"]
                                pos["partial_taken"] = True
                                log.info(f"   ✂️ REDUCED {symbol} 50%: returned ${returned:.2f}, SL->breakeven")

                                # Record REDUCE as trade in history (fix accounting gap)
                                history.append({
                                    "id": f"{pos.get('id','')}_reduce",
                                    "symbol": symbol,
                                    "direction": pos.get("direction", ""),
                                    "strategy": pos.get("strategy", "unknown"),
                                    "entry_price": pos.get("entry_price", 0),
                                    "close_price": pos.get("current_price", 0),
                                    "margin_usd": round(reduced_margin, 2),
                                    "pnl_usd": round(partial_pnl, 4),
                                    "pnl_pct": round((partial_pnl / reduced_margin * 100) if reduced_margin > 0 else 0, 4),
                                    "open_time": pos.get("open_time", ""),
                                    "close_time": datetime.now(timezone.utc).isoformat(),
                                    "close_reason": "REDUCE",
                                    "status": "closed",
                                })
                                portfolio_data["total_trades"] = portfolio_data.get("total_trades", 0) + 1
                                if partial_pnl > 0:
                                    portfolio_data["wins"] = portfolio_data.get("wins", 0) + 1
                                else:
                                    portfolio_data["losses"] = portfolio_data.get("losses", 0) + 1
                    ex.save_portfolio(portfolio_data)
                except Exception as e:
                    log.warning(f"   ⚠️ Error ejecutando REDUCE: {e}")
            # Handle TIGHTEN recommendations
            tighten_recs = [d for d in decisions if d.get("action") == "TIGHTEN"]
            if tighten_recs:
                try:
                    for rec in tighten_recs:
                        symbol = rec["symbol"]
                        new_trail = rec.get("trailing_pct", 0.005)
                        for pos in portfolio_data.get("positions", []):
                            if pos.get("symbol") == symbol and pos.get("status") == "open":
                                old_trail = pos.get("trailing_pct", 0.015)
                                if new_trail < old_trail:
                                    pos["trailing_pct"] = new_trail
                                    pos["exit_mode"] = "trailing"
                                    log.info(f"   \U0001f3af TIGHTEN {symbol}: trail {old_trail*100:.1f}% -> {new_trail*100:.1f}%")
                    ex.save_portfolio(portfolio_data)
                except Exception as e:
                    log.warning(f"   \u26a0\ufe0f Error TIGHTEN: {e}")

            if not close_recs and not reduce_recs:
                log.info(f"   🟢 MANTENER todas las posiciones")

            results["position_decisions"] = {
                "ok": True,
                "evaluated": len(decisions),
                "close_signals": len(close_recs),
                "reduce_signals": len(reduce_recs),
            }
        else:
            log.info("   ℹ️  Sin posiciones abiertas")
            results["position_decisions"] = {"ok": True, "evaluated": 0}
    except Exception as e:
        log.warning(f"   ⚠️ Error en position decisions: {e}")
        results["position_decisions"] = {"ok": False}

    # ── Portfolio Take Profit — DESHABILITADO (2026-04-05) ─────────────────
    # RAZÓN: El Portfolio TP cerraba TODAS las posiciones cuando el PnL combinado
    # llegaba a $2-5, cortando wins al 0.4-0.8% mientras losses comian el SL completo
    # (12%). Esto resultaba en R:R real de 1:0.12 en vez del 1:2.5 configurado.
    # Ahora cada posición corre hasta su propio SL/TP individual.
    # El trailing stop + partial profit taking protegen las ganancias.
    try:
        portfolio_file = DATA_DIR / "portfolio.json"
        if portfolio_file.exists():
            portfolio_data = json.loads(portfolio_file.read_text())
            open_positions = [p for p in portfolio_data.get("positions", []) if p.get("status") == "open"]
            if open_positions:
                total_pnl = sum(p.get("pnl_usd", 0) for p in open_positions)
                # Solo loguear el PnL combinado para visibilidad
                log.info(f"   📊 Portfolio PnL combinado: ${total_pnl:.2f} ({len(open_positions)} posiciones)")

                if False:  # DESHABILITADO — cada posicion maneja su propio SL/TP
                    log.info(f"   🎯 PORTFOLIO TP HIT: P&L ${total_pnl:.2f} >= ${portfolio_tp:.2f}")
                    log.info(f"   → Cerrando todas las posiciones para asegurar ganancias")
                    try:
                        market_data = json.loads((DATA_DIR / "market_latest.json").read_text()) if (DATA_DIR / "market_latest.json").exists() else {}
                        history = _load_trade_history()
                        closed = ex.close_positions_emergency(portfolio_data, symbols, market_data, history, reason="PORTFOLIO_TP")
                        for pos in closed:
                            ex.record_emergency_cooldown(pos.get("symbol", "")) if pos.get("symbol") else None
                        ex.save_portfolio(portfolio_data)
                        _save_trade_history(history)
                        _cycle_emergency_closes += len(closed)
                        log.info(f"   ✅ Cerradas {len(closed)} posición(es) por Portfolio TP — P&L capturado: ${total_pnl:.2f}")
                    except Exception as e:
                        log.warning(f"   ⚠️ Error en Portfolio TP: {e}")

                # C4: Portfolio TP disabled (was crashing on undefined portfolio_min)
                elif False:  # disabled
                    # AI evalúa si las posiciones tienen chance de llegar a $2
                    # Usamos la misma lógica de _quant_score para decidir
                    try:
                        from risk_manager import _quant_score
                        research_data = json.loads((DATA_DIR / "research_latest.json").read_text()) if (DATA_DIR / "research_latest.json").exists() else {}
                        market_for_dec = json.loads((DATA_DIR / "market_latest.json").read_text()) if (DATA_DIR / "market_latest.json").exists() else {}

                        total_score = 0
                        reachable_count = 0
                        for pos in open_positions:
                            qs = _quant_score(pos, market_for_dec, research_data)
                            score = qs.get("score", 0)
                            total_score += score
                            # Score > 0 = signal to close (take profit), < 0 = hold
                            if score < 0:
                                reachable_count += 1

                        avg_score = total_score / len(open_positions) if open_positions else 0
                        # Si el score promedio es >= 0, la AI recomienda cerrar
                        # (promedio >= 0 significa que en conjunto dice HOLD/CERRAR)
                        will_reach = reachable_count >= len(open_positions) // 2 + 1  # mayoría dice "cerrar"

                        if not will_reach and avg_score >= 0:
                            log.info(f"   🎯 PORTFOLIO MIN: P&L ${total_pnl:.2f} >= ${portfolio_min:.2f} pero AI duda que llegue a ${portfolio_tp:.2f}")
                            log.info(f"   → AI score: {avg_score:.0f} → Cerrando para asegurar ${total_pnl:.2f}")
                            try:
                                market_data = json.loads((DATA_DIR / "market_latest.json").read_text()) if (DATA_DIR / "market_latest.json").exists() else {}
                                history = _load_trade_history()
                                closed = ex.close_positions_emergency(portfolio_data, symbols, market_data, history, reason="PORTFOLIO_TP_MIN")
                                for pos in closed:
                                    ex.record_emergency_cooldown(pos.get("symbol", "")) if pos.get("symbol") else None
                                ex.save_portfolio(portfolio_data)
                                _save_trade_history(history)
                                _cycle_emergency_closes += len(closed)
                                log.info(f"   ✅ Cerradas {len(closed)} posición(es) por Portfolio Min — P&L capturado: ${total_pnl:.2f}")
                            except Exception as e:
                                log.warning(f"   ⚠️ Error en Portfolio Min: {e}")
                        else:
                            log.info(f"   💪 PORTFOLIO: P&L ${total_pnl:.2f} >= ${portfolio_min:.2f}, AI ve potencial para llegar a ${portfolio_tp:.2f} → Dejando correr")
                    except ImportError:
                        log.warning(f"   ⚠️  No se pudo importar _quant_score para Portfolio Min")
                    except Exception as e:
                        log.warning(f"   ⚠️  Error en Portfolio Min AI check: {e}")
    except Exception as e:
        log.warning(f"   ⚠️  Error en Portfolio Take Profit check: {e}")

    # ── Circuit Breaker Check ─────────────────────────────────────────────────
    if _cycle_emergency_closes >= CIRCUIT_BREAKER_LIMIT:
        if _circuit_breaker_check_and_record(_cycle_emergency_closes):
            log.error("🛑 CIRCUIT BREAKER: Bot detenido. Revisar causa y eliminar STOP_TRADING para reanudar.")
            results["circuit_breaker"] = {"triggered": True, "closes": _cycle_emergency_closes}
            return results

    # Paso 4.5: WILD MODE (martingale_engine) — opcional, controlado por dashboard
    try:
        import martingale_engine as wild_me
        if wild_me.is_active():
            log.info("━" * 40)
            log.info("🔥 [4.5] Wild Mode (Martingale Engine)")
            # Reload portfolio (may have changed during this cycle)
            from executor import load_portfolio as _load_p, load_history as _load_h, save_portfolio as _save_p, save_history as _save_h, load_market as _load_m
            _wild_port = _load_p()
            _wild_market = _load_m()
            _wild_history = _load_h()
            try:
                _wild_mkt = _wild_market or {}
                _fg_raw = _wild_mkt.get("fear_greed", {})
                _fg_val = _fg_raw.get("value", 50) if isinstance(_fg_raw, dict) else int(_fg_raw or 50)
            except Exception:
                _fg_val = 50
            wild_result = wild_me.run_cycle(_wild_port, _wild_market, _wild_history, _fg_val)
            # Persist any changes the engine made
            if wild_result.get("opened") or wild_result.get("closed") or wild_result.get("abandoned") or wild_result.get("target_hit"):
                _save_p(_wild_port)
                _save_h(_wild_history)
            if wild_result.get("abandoned"):
                log.warning(f"   🔥 WILD MODE ABANDONED: {wild_result.get('reason')}")
            elif wild_result.get("target_hit"):
                log.info(f"   🎯 WILD MODE TARGET HIT: closed {wild_result.get('closed')} positions ({wild_result.get('reason')})")
            elif wild_result.get("opened"):
                log.info(f"   🔥 WILD MODE: opened {wild_result.get('opened')} hedge levels, closed {wild_result.get('closed')}")
            else:
                log.info(f"   🔥 WILD MODE: HOLD (no changes)")
            results["wild_mode"] = wild_result
    except Exception as e:
        log.warning(f"   ⚠️ Wild Mode error (non-fatal): {e}")
        results["wild_mode"] = {"ok": False, "error": str(e)}

    # Paso 5: Auto-Learner — analiza trades y adapta parámetros
    log.info("━" * 40)
    log.info("🧠 [5/6] Auto-Learner")
    try:
        cycle_count_for_learner = getattr(run_cycle, '_cycle_count', 0)
        learner_interval = 10  # Cada 10 ciclos (~10 min)
        if cycle_count_for_learner % learner_interval == 0 or cycle_count_for_learner == 1:
            import auto_learner
            result_learner = auto_learner.run(debug=debug)
            # Recargar params en risk_manager después de learner
            try:
                from risk_manager import reload_auto_learner_params
                reload_auto_learner_params()
                log.info(f"   🧠 Params recargados desde auto_learner")
            except Exception as lr_e:
                log.warning(f"   ⚠️ reload params error: {lr_e}")
            results["auto_learner"] = {"ok": True}
        else:
            log.info(f"   ⏭️  Auto-Learner skip (ciclo {cycle_count_for_learner % learner_interval}/{learner_interval})")
            results["auto_learner"] = {"ok": True, "skipped": True}
    except Exception as e:
        log.warning(f"   ⚠️ Auto-Learner error: {e}")
        results["auto_learner"] = {"ok": False, "error": str(e)}

    # Paso 6: Reporter
    log.info("━" * 40)
    log.info("📊 [6/6] Reporter")
    try:
        result = rp.run(daily=False)
        results["reporter"] = {"ok": True}
    except Exception as e:
        log.warning(f"   ⚠️ Error: {e}")
        results["reporter"] = {"ok": False}
    
    # Calcular health score
    ok_count = sum(1 for r in results.values() if r.get("ok", False))
    health_score = min(10, ok_count * 2)
    
    log.info("=" * 60)
    log.info("📊 CICLO COMPLETADO")
    elapsed = time.time() - cycle_start
    log.info(f"   ⏱  Tiempo: {elapsed:.1f}s | Health: {health_score}/10")
    
    # Guardar reporte
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "health_score": health_score
    }
    report_file = DATA_DIR / "report_latest.json"
    report_file.write_text(json.dumps(report, indent=2))

    # ── HEALTH LINE ───────────────────────────────────────────────────────────
    # Extraer equity y posiciones del portfolio para la línea de salud
    try:
        pf = json.loads(portfolio_file.read_text()) if portfolio_file.exists() else {}
        _cash = float(pf.get("capital_usd", 0))
        _open_pos = [p for p in pf.get("positions", []) if p.get("status") == "open"]
        _margins = sum(float(p.get("margin_usd", 0)) for p in _open_pos)
        _unreal = sum(float(p.get("pnl_usd", 0)) for p in _open_pos)
        equity = _cash + _margins + _unreal
        n_positions = len([p for p in pf.get("positions", []) if p.get("status") == "open"])
    except Exception:
        equity = 0.0
        n_positions = 0
    exit_code = 0 if health_score >= 4 else 1
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    health_line = (
        f"[{ts}] [HEALTH] Ciclo completado — exit_code={exit_code} "
        f"— equity=${equity:.2f} — posiciones={n_positions}"
    )
    # Escribir directamente para garantizar el tag [HEALTH] literal
    print(health_line, flush=True)
    try:
        with open(LOG_FILE, "a") as _lf:
            _lf.write(health_line + "\n")
    except Exception:
        pass

    # Rotar log al finalizar si supera 50MB (doble check desde Python)
    _rotate_log_if_needed()

    return results


def run_token_scanner():
    """Ejecuta el scanner de tokens para encontrar nuevas oportunidades."""
    try:
        import token_scanner
        log.info("━" * 40)
        log.info("🔍 [SCANNER] Buscando nuevas oportunidades...")
        result = token_scanner.scan(debug=False)
        if result.get("tokens_added"):
            log.info(f"   ➕ Nuevos tokens: {result['tokens_added']}")
        log.info(f"   📊 Total oportunidades: {result.get('opportunities_found', 0)}")
        return result
    except Exception as e:
        log.warning(f"   ⚠️ Scanner error: {e}")
        return {}


# FIX 3.4: PID lock to prevent dual instances
# Use absolute path so ALL entry points (agents/orchestrator.py, root orchestrator.py, etc.)
# share the same lock regardless of working directory.
import tempfile
# Lock file override via ORCH_LOCK_FILE env var (permite correr paper + live en paralelo)
LOCK_FILE = Path(os.environ.get("ORCH_LOCK_FILE", str(Path(tempfile.gettempdir()) / "solana_jupiter_orchestrator.lock")))

# AUDIT FIX: Persistent cycle counter so scanner runs even across restarts
CYCLE_COUNT_FILE = DATA_DIR / "cycle_counter.txt"

def _load_cycle_count() -> int:
    """Load persistent cycle counter from file. Returns 0 if missing/corrupt."""
    try:
        if CYCLE_COUNT_FILE.exists():
            return int(CYCLE_COUNT_FILE.read_text().strip() or "0")
    except Exception:
        pass
    return 0

def _save_cycle_count(n: int) -> None:
    """Persist cycle counter to file (atomic via temp+rename)."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _tmp = CYCLE_COUNT_FILE.with_suffix(".tmp")
        _tmp.write_text(str(n))
        _tmp.rename(CYCLE_COUNT_FILE)
    except Exception as _e_save:
        log.warning(f"cycle_count save error: {_e_save}")


def _acquire_lock():
    """Acquire PID lock. Exit if another instance is running."""
    log.info(f"📁 DATA_DIR={DATA_DIR} | PID={os.getpid()} | LOCK_FILE={LOCK_FILE}")
    # v2.9.0-live: validar configuracion al arrancar
    try:
        import safety as _safety
        _startup_errors = _safety.validate_startup()
        if _startup_errors:
            for _e in _startup_errors:
                log.error(f"❌ STARTUP VALIDATION: {_e}")
            log.error("❌ Abortando arranque. Corregir errores y reiniciar.")
            sys.exit(1)
    except SystemExit:
        raise
    except Exception as _ve:
        log.warning(f"validate_startup error (non-fatal): {_ve}")
    if LOCK_FILE.exists():
        try:
            old_pid = int(LOCK_FILE.read_text().strip())
            os.kill(old_pid, 0)  # Check if process alive
            log.error(f"Another orchestrator is running (PID {old_pid}). Exiting.")
            sys.exit(1)
        except (ProcessLookupError, ValueError, PermissionError):
            pass  # Stale lock
    LOCK_FILE.write_text(str(os.getpid()))
    import atexit
    atexit.register(lambda: LOCK_FILE.unlink(missing_ok=True))
    log.info(f"PID lock acquired: {os.getpid()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--live", action="store_true", help="Run in continuous loop (alias for default loop mode)")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between cycles in loop mode (default: 60)")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--scan-interval", type=int, default=10, help="Run token scanner every N cycles (default: 10)")
    args = parser.parse_args()

    if args.once:
        _acquire_lock()  # Prevent dual instances even in single-cycle mode
        log.info(f"📁 DATA_DIR={DATA_DIR} | PID={os.getpid()} | MODE=once")
        # Detectar cierre de vela 30m
        _candle_close = _is_new_candle_30m()
        if _candle_close:
            log.info("🕯️ [CANDLE CLOSE] Vela 30m cerrada — ejecutando ciclo en punto óptimo")
        run_cycle(debug=args.debug)
        # Performance snapshot cada 10 ciclos
        if _PERF_TRACKER and cycle_count % 10 == 0:
            try:
                pt.save_performance_snapshot()
                if cycle_count % 50 == 0:
                    pt.print_dashboard()
                # Alerta Telegram si estrategia degradada (cada ~60 ciclos = 1h)
                if cycle_count % 60 == 0:
                    pt.check_and_alert(cooldown_hours=6.0)
            except Exception as _pt_err:
                log.debug(f"perf_tracker error: {_pt_err}")
    else:
        # --live or bare invocation: run continuous loop
        _acquire_lock()  # Prevent dual orchestrator instances
        interval = args.interval if args.interval else 60
        scan_interval = args.scan_interval if args.scan_interval else 10
        cycle_count = _load_cycle_count()
        log.info(f"🔢 Cycle counter loaded from disk: {cycle_count}")

        log.info(f"🔄 Modo continuo — intervalo: {interval}s")
        log.info(f"🔍 Token Scanner cada {scan_interval} ciclos (~{scan_interval}min)")
        log.info(f"📁 DATA_DIR={DATA_DIR} | PID={os.getpid()}")

        # Kickstart: run scanner once at startup if last scan is stale (>30 min)
        try:
            _scanner_report = DATA_DIR / "scanner_report.json"
            _should_kickstart = True
            if _scanner_report.exists():
                import json as _json_kick
                _sr = _json_kick.loads(_scanner_report.read_text())
                _ts = _sr.get("timestamp", "")
                if _ts:
                    _age_min = (datetime.now(timezone.utc) - datetime.fromisoformat(str(_ts).replace("Z", "+00:00"))).total_seconds() / 60
                    if _age_min < 30:
                        _should_kickstart = False
                        log.info(f"🔍 Scanner: last scan {_age_min:.0f}min ago, skipping kickstart")
            if _should_kickstart:
                log.info(f"🔍 Scanner kickstart on startup (stale or never run)")
                run_token_scanner()
        except Exception as _e_kick:
            log.warning(f"scanner kickstart check error: {_e_kick}")

        while True:
            try:
                cycle_count += 1
                if cycle_count > 1_000_000:
                    cycle_count = 0
                _save_cycle_count(cycle_count)
                
                # Ejecutar scanner cada N ciclos
                if cycle_count % scan_interval == 0:
                    run_token_scanner()
                
                run_cycle(debug=args.debug)
                # FIX 2.1: Polling rapido (15s) con posiciones abiertas, normal (60s) sin
                try:
                    pf_path = DATA_DIR / "portfolio.json"
                    has_open = False
                    if pf_path.exists():
                        import json as _json
                        _pf = _json.loads(pf_path.read_text())
                        has_open = any(p.get("status") == "open" for p in _pf.get("positions", []))
                    actual_interval = 10 if has_open else interval  # E6: 10s polling (was 15s)
                    time.sleep(actual_interval)
                except Exception:
                    time.sleep(interval)
            except KeyboardInterrupt:
                log.info("🛑 Detenido por usuario")
                break
