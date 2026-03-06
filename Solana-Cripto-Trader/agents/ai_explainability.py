#!/usr/bin/env python3
"""
💬 AI Explainability Agent - Solana Trading Bot
Explica decisiones de trading en lenguaje natural

Función:
- Explicar por qué se abrieron/cerraron posiciones
- Generar resúmenes del portafolio
- Enviar alertas de riesgo con voz MiniMax TTS

Input:
- Señales del Strategy Agent
- P&L actual
- Posiciones abiertas

Output:
- Alertas por Telegram
- Archivo de explicaciones
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

# ─── Configuración ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
STRATEGY_FILE = DATA_DIR / "strategy_llm.json"
EXPLANATION_FILE = DATA_DIR / "latest_explanation.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("ai_explainability")

# ─── Carga de Datos ───────────────────────────────────────────────────────────

def load_portfolio() -> dict:
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE, 'r') as f:
            return json.load(f)
    return {}

def load_strategy() -> dict:
    if STRATEGY_FILE.exists():
        with open(STRATEGY_FILE, 'r') as f:
            return json.load(f)
    return {"signals": [], "summary": {}}

# ─── Generación de Explicaciones ───────────────────────────────────────────────────

def generate_portfolio_summary(portfolio: dict) -> str:
    """Genera un resumen del portafolio en lenguaje natural."""
    
    capital = portfolio.get("capital_usd", 0)
    initial = portfolio.get("initial_capital", 500)
    positions = portfolio.get("positions", [])
    
    open_positions = [p for p in positions if p.get("status") == "open"]
    closed_positions = [p for p in positions if p.get("status") == "closed"]
    
    # Calcular P&L total
    total_pnl = sum(p.get("pnl_usd", 0) for p in closed_positions)
    
    # Calcular win rate
    total_trades = portfolio.get("total_trades", 0)
    wins = portfolio.get("wins", 0)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    # Generar texto
    summary = f"""📊 RESUMEN DEL PORTAFOLIO

💰 Capital: ${capital:.2f} (Inicial: ${initial:.2f})
📈 P&L Total: ${total_pnl:+.2f}
💼 Posiciones abiertas: {len(open_positions)}
📝 Trades cerrados: {len(closed_positions)}
🎯 Win Rate: {win_rate:.1f}% ({wins}W/{total_trades}L)
"""
    
    if open_positions:
        summary += "\n💼 POSICIONES ABIERTAS:\n"
        for pos in open_positions:
            symbol = pos["symbol"]
            direction = pos["direction"].upper()
            entry = pos["entry_price"]
            current = pos["current_price"]
            pnl_pct = pos["pnl_pct"]
            pnl_val = pos["pnl_usd"]
            strategy = pos["strategy"]
            
            emoji = "📈" if pnl_val > 0 else "📉"
            summary += f"\n{emoji} {symbol} ({direction}) - {strategy}\n"
            summary += f"   Entry: ${entry:.6f} → Current: ${current:.6f}\n"
            summary += f"   P&L: {pnl_pct:+.2f}% (${pnl_val:+.4f})\n"
    
    return summary

def generate_signals_summary(strategy: dict) -> str:
    """Genera un resumen de las señales generadas."""
    
    signals = strategy.get("signals", [])
    summary_data = strategy.get("summary", {})
    
    if not signals:
        return "🚫 No se generaron señales en este ciclo"
    
    text = f"🎯 SEÑALES GENERADAS ({len(signals)})\n"
    
    for i, signal in enumerate(signals, 1):
        symbol = signal.get("symbol", "N/A")
        direction = signal.get("direction", "none").upper()
        entry = signal.get("entry_price", 0)
        sl = signal.get("sl_price", 0)
        tp = signal.get("tp_price", 0)
        confidence = signal.get("confidence", 0)
        strategy_type = signal.get("strategy", "N/A")
        reasoning = signal.get("reasoning", "")
        
        text += f"\n{i}. {symbol} ({direction}) - {strategy_type}\n"
        text += f"   Entry: ${entry:.6f} | SL: ${sl:.6f} | TP: ${tp:.6f}\n"
        text += f"   Confianza: {confidence:.2f}\n"
        
        if reasoning:
            text += f"   📝 {reasoning}\n"
    
    return text

def generate_risk_alert(portfolio: dict) -> str:
    """Genera alertas de riesgo si hay problemas."""
    
    alerts = []
    
    capital = portfolio.get("capital_usd", 0)
    initial = portfolio.get("initial_capital", 500)
    
    # Alerta de drawdown
    if capital < initial * 0.9:  # >10% drawdown
        drawdown = (initial - capital) / initial * 100
        alerts.append(f"⚠️ DRAWDOWN CRÍTICO: {drawdown:.1f}%")
    elif capital < initial * 0.95:  # >5% drawdown
        drawdown = (initial - capital) / initial * 100
        alerts.append(f"⚠️ Drawdown alto: {drawdown:.1f}%")
    
    # Alerta de muchas posiciones
    positions = portfolio.get("positions", [])
    open_positions = [p for p in positions if p.get("status") == "open"]
    
    if len(open_positions) >= 5:
        alerts.append("⚠️ 5 posiciones abiertas - máximo alcanzado")
    
    # Alerta de P&L negativo
    closed_positions = [p for p in positions if p.get("status") == "closed"]
    recent_pnl = sum(p.get("pnl_usd", 0) for p in closed_positions[-5:])
    
    if recent_pnl < -10:
        alerts.append(f"⚠️ P&L reciente negativo: ${recent_pnl:.2f}")
    
    if not alerts:
        return None
    
    return "\n".join(alerts)

# ─── Entry Point ───────────────────────────────────────────────────────────────

def run(debug: bool = False) -> dict:
    log.info("=" * 50)
    log.info("💬 AI EXPLAINABILITY - Generación de Alertas")
    log.info("=" * 50)
    
    # Cargar datos
    portfolio = load_portfolio()
    strategy = load_strategy()
    
    # Generar explicaciones
    portfolio_summary = generate_portfolio_summary(portfolio)
    signals_summary = generate_signals_summary(strategy)
    risk_alert = generate_risk_alert(portfolio)
    
    # Mostrar en consola
    print("\n" + portfolio_summary)
    print("\n" + signals_summary)
    
    if risk_alert:
        print(f"\n⚠️ ALERTAS DE RIESGO:\n{risk_alert}")
    else:
        print("\n✅ Sin alertas de riesgo")
    
    # Crear explicación completa
    explanation = {
        "portfolio_summary": portfolio_summary,
        "signals_summary": signals_summary,
        "risk_alert": risk_alert,
        "generated_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Guardar
    with open(EXPLANATION_FILE, 'w') as f:
        json.dump(explanation, f, indent=2)
    
    log.info(f"💾 Explicación guardada en: {EXPLANATION_FILE}")
    log.info("=" * 50)
    
    return explanation

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Output detallado")
    args = parser.parse_args()
    
    run(debug=args.debug)
