"""
DREAM ENGINE — Solana Bot
Background analysis while the bot is not actively operating.
Identifies patterns, generates insights, proposes strategy improvements.
"""
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Optional

log = logging.getLogger("dream_engine")

DATA_DIR = Path(__file__).parent / "data"
HISTORY_FILE = DATA_DIR / "trade_history.json"
INSIGHTS_FILE = DATA_DIR / "dream_insights.json"
REPORT_FILE = DATA_DIR / "dream_report.md"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
MARKET_FILE = DATA_DIR / "market_latest.json"


def run(debug: bool = False) -> dict:
    """
    Ejecuta el Dream Engine + Auto-Learner (LLM-powered insights).
    Diseñado para correr via cron o entre ciclos principales.
    """
    log.info("💤 DREAM ENGINE — Iniciando analisis de fondo...")
    
    insights = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "insights": [],
        "recommendations": [],
        "anomalies": [],
        "ai_insight": None,
        "summary": ""
    }
    
    # 1. Load trade history
    history = _load_history()
    market = _load_market()
    portfolio = _load_portfolio()
    anomalies = []
    
    # 2. Quantitative Analysis (always runs)
    if len(history) >= 3:
        trade_insights = _analyze_trades(history)
        insights["insights"].extend(trade_insights["insights"])
        post_mortem = _analyze_open_positions(portfolio, market)
        insights["insights"].extend(post_mortem["insights"])
        patterns = _find_patterns(history, market)
        insights["insights"].extend(patterns["insights"])
        insights["recommendations"].extend(patterns["recommendations"])
        anomalies = _detect_anomalies(history, portfolio)
        insights["anomalies"] = anomalies
    else:
        log.info(f"   Sin suficientes trades para analisis cuantitativo ({len(history)} trades)")
    
    # 3. Auto-Learner (LLM-powered insights)
    try:
        import auto_learner
        log.info("   🤖 Invocando Auto-Learner (LLM)...")
        al_result = auto_learner.run(debug=debug)
        
        if al_result.get("status") == "no_data":
            log.info("   ⏸️ Auto-Learner: sin datos suficientes")
        elif al_result.get("status") == "waiting":
            log.info(f"   ⏸️ Auto-Learner: esperando más trades ({al_result.get('trades_needed', '?')} necesarios)")
        elif al_result.get("status") == "ok" or al_result.get("status") == "applied":
            key_insight = al_result.get("key_insight", "")
            confidence = al_result.get("confidence", 0)
            analysis = al_result.get("analysis", "")
            tokens_avoid = al_result.get("tokens_to_avoid", [])
            tokens_prefer = al_result.get("tokens_to_prefer", [])
            changes = al_result.get("changes", [])
            
            if key_insight:
                insights["ai_insight"] = {
                    "key_insight": key_insight,
                    "confidence": confidence,
                    "analysis": analysis,
                    "tokens_to_avoid": tokens_avoid,
                    "tokens_to_prefer": tokens_prefer,
                    "changes": changes,
                    "new_lessons": al_result.get("new_lessons", []),
                }
                log.info(f"   🤖 AI Insight: {key_insight}")
                log.info(f"   🤖 Confidence: {confidence*100:.0f}%")
                
                # Convert to recommendation
                if confidence >= 0.7:
                    insights["recommendations"].insert(0, {
                        "type": "ai_insight",
                        "finding": key_insight,
                        "action": al_result.get("analysis", "")[:200],
                        "confidence": confidence,
                        "auto_apply": al_result.get("status") == "applied",
                        "is_ai": True,
                    })
                
                # Add tokens to avoid/prefer as insights
                if tokens_avoid:
                    insights["insights"].append({
                        "type": "token_ai",
                        "confidence": 0.85,
                        "finding": f"Tokens a evitar según IA: {', '.join(tokens_avoid)}",
                    })
                if tokens_prefer:
                    insights["insights"].append({
                        "type": "token_ai",
                        "confidence": 0.85,
                        "finding": f"Tokens preferidos según IA: {', '.join(tokens_prefer)}",
                    })
        else:
            log.info(f"   Auto-Learner status: {al_result.get('status')}")
    except Exception as e:
        log.warning(f"   ⚠️ Auto-Learner error: {e}")
    
    # 4. Generate markdown report
    report = _generate_report(insights, history, portfolio)
    REPORT_FILE.write_text(report)
    
    # 5. Auto-apply high-confidence recommendations
    applied = _auto_apply_recommendations(insights["recommendations"])
    
    insights["summary"] = (
        f"{len(history)} trades | "
        f"{len(insights['insights'])} insights | "
        f"{len(anomalies) if isinstance(anomalies, list) else 0} anomalias | "
        f"{applied} aplicados | "
        f"{'🤖 AI' if insights.get('ai_insight') else 'sin IA'}"
    )
    
    _save_insights(insights)
    log.info(f"   💤 Dream Engine completado: {insights['summary']}")
    
    return insights


# ─── Trade Analysis ───────────────────────────────────────────────

def _load_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text())
        return data.get("trades", []) if isinstance(data, dict) else data
    except Exception:
        return []


def _load_market() -> dict:
    if not MARKET_FILE.exists():
        return {}
    try:
        return json.loads(MARKET_FILE.read_text())
    except Exception:
        return {}


def _load_portfolio() -> dict:
    if not PORTFOLIO_FILE.exists():
        return {}
    try:
        return json.loads(PORTFOLIO_FILE.read_text())
    except Exception:
        return {}


def _analyze_trades(history: list) -> dict:
    """Analiza patrones en historial de trades."""
    insights = []
    
    if not history:
        return {"insights": insights}
    
    winners = [t for t in history if t.get("pnl_usd", 0) > 0]
    losers = [t for t in history if t.get("pnl_usd", 0) <= 0]
    
    total = len(history)
    win_rate = len(winners) / total if total > 0 else 0
    avg_win = sum(t["pnl_usd"] for t in winners) / len(winners) if winners else 0
    avg_loss = sum(t["pnl_usd"] for t in losers) / len(losers) if losers else 0
    
    # Win rate por dirección
    longs = [t for t in history if t.get("direction") == "long"]
    shorts = [t for t in history if t.get("direction") == "short"]
    long_wr = len([t for t in longs if t.get("pnl_usd", 0) > 0]) / len(longs) if longs else 0
    short_wr = len([t for t in shorts if t.get("pnl_usd", 0) > 0]) / len(shorts) if shorts else 0
    
    insights.append({
        "type": "win_rate",
        "confidence": 0.9,
        "finding": f"Win rate global: {win_rate*100:.0f}% ({len(winners)}W/{len(losers)}L)",
        "detail": f"Avg win: ${avg_win:.2f} | Avg loss: ${avg_loss:.2f} | Avg R:R: {abs(avg_win/avg_loss):.1f}x" if losers and avg_loss != 0 else "",
    })
    
    if longs:
        insights.append({
            "type": "direction_analysis",
            "confidence": 0.8,
            "finding": f"Longs: {long_wr*100:.0f}% WR ({len(longs)} trades)",
        })
    
    if shorts:
        insights.append({
            "type": "direction_analysis",
            "confidence": 0.8,
            "finding": f"Shorts: {short_wr*100:.0f}% WR ({len(shorts)} trades)",
        })
    
    # Promedio de R:R en winners
    if winners:
        rr_list = []
        for t in winners:
            entry = t.get("entry_price", 0)
            tp = t.get("tp_price", 0)
            sl = t.get("sl_price", 0)
            if entry > 0 and sl != entry:
                rr = abs(tp - entry) / abs(sl - entry) if abs(sl - entry) > 0 else 0
                rr_list.append(rr)
        if rr_list:
            avg_rr = sum(rr_list) / len(rr_list)
            insights.append({
                "type": "risk_reward",
                "confidence": 0.85,
                "finding": f"R:R promedio en winners: 1:{avg_rr:.1f}x (target: 1:2.0)",
                "auto_apply": False
            })
    
    return {"insights": insights}


# ─── Open Position Post-Mortem ───────────────────────────────────

def _analyze_open_positions(portfolio: dict, market: dict) -> dict:
    """Analiza posiciones abiertas y da contexto."""
    insights = []
    positions = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]
    
    if not positions:
        return {"insights": insights}
    
    total_pnl = sum(p.get("pnl_usd", 0) for p in positions)
    fg_data = market.get("fear_greed", {})
    fg = fg_data.get("value", 50) if isinstance(fg_data, dict) else 50
    
    for pos in positions:
        sym = pos["symbol"]
        pnl = pos.get("pnl_usd", 0)
        hours_open = pos.get("hours_open", 0)
        direction = pos.get("direction", "unknown")
        rr_remaining = pos.get("rr_remaining", 0)
        
        # Si está en pérdida después de >1h
        if pnl < -0.50 and hours_open > 1:
            insights.append({
                "type": "position_concern",
                "confidence": 0.75,
                "symbol": sym,
                "finding": f"{sym} en perdida ${pnl:.2f} por {hours_open:.1f}h — R:R restante {rr_remaining:.1f}x",
                "action": "Monitorear de cerca. Considerar cierre si F&G cambia de sentimiento."
            })
        
        # Si está en ganancia sólida
        if pnl > 1.0:
            insights.append({
                "type": "position_winning",
                "confidence": 0.9,
                "symbol": sym,
                "finding": f"{sym} en ganancia ${pnl:.2f} — mantener hasta TP o señal de cierre."
            })
    
    if total_pnl > 0:
        insights.append({
            "type": "portfolio_status",
            "confidence": 0.95,
            "finding": f"Portfolio en ganancia ${total_pnl:.2f} con {len(positions)} posiciones",
        })
    else:
        insights.append({
            "type": "portfolio_status",
            "confidence": 0.95,
            "finding": f"Portfolio en perdida ${total_pnl:.2f} con {len(positions)} posiciones | F&G={fg}",
        })
    
    return {"insights": insights}


# ─── Pattern Recognition ─────────────────────────────────────────

def _find_patterns(history: list, market: dict) -> dict:
    """Busca patrones en los trades."""
    insights = []
    recommendations = []
    
    if len(history) < 5:
        return {"insights": insights, "recommendations": recommendations}
    
    # Patrón: Fear & Greed vs resultados
    fg_data = market.get("fear_greed", {})
    fg = fg_data.get("value", 50) if isinstance(fg_data, dict) else 50
    
    # Analizar trades por F&G aproximado (usando trend como proxy)
    short_wins = 0; short_total = 0
    long_wins = 0; long_total = 0
    
    for t in history:
        d = t.get("direction", "unknown")
        if d == "short":
            short_total += 1
            if t.get("pnl_usd", 0) > 0:
                short_wins += 1
        elif d == "long":
            long_total += 1
            if t.get("pnl_usd", 0) > 0:
                long_wins += 1
    
    if short_total >= 3:
        short_wr = short_wins / short_total
        if short_wr > 0.6:
            recommendations.append({
                "type": "direction_bias",
                "confidence": short_wr,
                "finding": f"Shorts tienen {short_wr*100:.0f}% WR — el mercado favorece cortos",
                "action": "Priorizar shorts en siguientes señales",
                "auto_apply": short_wr > 0.85
            })
        elif short_wr < 0.3:
            recommendations.append({
                "type": "direction_bias",
                "confidence": 1 - short_wr,
                "finding": f"Shorts tienen solo {short_wr*100:.0f}% WR — considerar reducir shorts",
                "action": "Revisar strategy para shorts",
                "auto_apply": False
            })
    
    # F&G actual
    if fg <= 20:
        insights.append({
            "type": "sentiment",
            "confidence": 0.95,
            "finding": f"F&G = {fg} (Extreme Fear) — shorts tienen ventaja histórica",
        })
    elif fg >= 75:
        insights.append({
            "type": "sentiment",
            "confidence": 0.95,
            "finding": f"F&G = {fg} (Greed) — longs tienen ventaja histórica",
        })
    
    return {"insights": insights, "recommendations": recommendations}


# ─── Anomaly Detection ───────────────────────────────────────────

def _detect_anomalies(history: list, portfolio: dict) -> list:
    """Detecta anomalías en los datos."""
    anomalies = []
    
    if not history:
        return anomalies
    
    # Promedio de PnL
    pnls = [t.get("pnl_usd", 0) for t in history]
    if pnls:
        avg = sum(pnls) / len(pnls)
        # Trades extreme outlier (>3x avg)
        for t in history:
            pnl = t.get("pnl_usd", 0)
            if abs(pnl) > abs(avg) * 3 and abs(pnl) > 5:
                anomalies.append({
                    "type": "outlier",
                    "symbol": t.get("symbol", "unknown"),
                    "pnl": pnl,
                    "finding": f"PnL extremo: ${pnl:.2f} en {t.get('symbol','?')} (avg ${avg:.2f})"
                })
    
    # Posiciones abiertas con PnL muy negativo
    positions = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]
    for pos in positions:
        pnl = pos.get("pnl_usd", 0)
        if pnl < -2.0:  # >$2 pérdida
            anomalies.append({
                "type": "large_loss",
                "symbol": pos.get("symbol"),
                "pnl": pnl,
                "finding": f"Posicion {pos.get('symbol')} en perdida ${pnl:.2f} — riesgo de liquidation"
            })
    
    return anomalies


# ─── Auto-Apply Recommendations ─────────────────────────────────

def _auto_apply_recommendations(recommendations: list) -> int:
    """Aplica automaticamente recommendations con confidence > 0.85."""
    applied = 0
    
    for rec in recommendations:
        if rec.get("auto_apply") and rec.get("confidence", 0) >= 0.85:
            log.info(f"   🚀 Auto-aplicando: {rec.get('finding', rec)}")
            applied += 1
            # TODO: integrate with strategy.py or executor.py
            # For now just log — manual integration needed per recommendation type
    
    return applied


# ─── Report Generation ───────────────────────────────────────────

def _generate_report(insights: dict, history: list, portfolio: dict) -> str:
    """Genera reporte markdown legible."""
    lines = [
        "# 🌙 Dream Engine Report",
        f"Generado: {insights['timestamp']}",
        "",
    ]
    
    # Summary stats
    winners = [t for t in history if t.get("pnl_usd", 0) > 0]
    losers = [t for t in history if t.get("pnl_usd", 0) <= 0]
    total = len(history)
    wr = len(winners) / total * 100 if total > 0 else 0
    
    lines.append(f"**Trades analizados**: {total} ({len(winners)}W / {len(losers)}L)")
    lines.append(f"**Win Rate**: {wr:.0f}%")
    lines.append("")
    
    # AI Insight
    if insights.get("ai_insight"):
        ai = insights["ai_insight"]
        lines.append("## 🤖 AI Insight (Auto-Learner)")
        lines.append(f"**[{ai['confidence']*100:.0f}% confidence]**")
        lines.append(f"{ai['key_insight']}")
        if ai.get("tokens_to_avoid"):
            lines.append(f"- 🔴 Tokens a evitar: {', '.join(ai['tokens_to_avoid'])}")
        if ai.get("tokens_to_prefer"):
            lines.append(f"- 🟢 Tokens preferidos: {', '.join(ai['tokens_to_prefer'])}")
        if ai.get("changes"):
            lines.append(f"- ⚙️ Cambios aplicados: {', '.join(str(c) for c in ai['changes'])}")
        lines.append("")
    
    # Insights
    if insights["insights"]:
        lines.append("## 💡 Insights")
        for ins in insights["insights"]:
            conf_emoji = "🔥" if ins.get("confidence", 0) >= 0.9 else "💡" if ins.get("confidence", 0) >= 0.75 else "💭"
            lines.append(f"- {conf_emoji} [{ins.get('confidence', 0)*100:.0f}%] {ins.get('finding', '')}")
        lines.append("")
    
    # Recommendations
    if insights["recommendations"]:
        lines.append("## 🎯 Recomendaciones")
        for rec in insights["recommendations"]:
            auto = "🤖 AUTO" if rec.get("auto_apply") else "📋 MANUAL"
            if rec.get("is_ai"):
                lines.append(f"- 🤖 **[AI]** {rec.get('finding', '')}")
            else:
                lines.append(f"- [{auto}] {rec.get('finding', '')}")
            lines.append(f"  → {rec.get('action', '')}")
        lines.append("")
    
    # Anomalies
    if insights["anomalies"]:
        lines.append("## ⚠️ Anomalías Detectadas")
        for anom in insights["anomalies"]:
            lines.append(f"- 🔴 {anom.get('finding', str(anom))}")
        lines.append("")
    
    return "\n".join(lines)


# ─── Save Outputs ────────────────────────────────────────────────

def _save_insights(insights: dict):
    INSIGHTS_FILE.write_text(json.dumps(insights, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    result = run()
    print(f"\n💤 Dream Engine: {result['summary']}")
