# v1.0-rentable — Primera versión rentable confirmada
**Fecha:** 2026-04-13  
**Capital inicial:** $500  
**Equity en el momento del tag:** $524.58  
**Retorno 24h:** +4.92%  
**Win Rate:** 81.25% (13W / 3L / 16 trades)  
**Profit Factor:** 4.25x  
**Sharpe:** 7.35  
**Max Drawdown:** 1.44%  

## Contexto de mercado
- Fear & Greed Index: 12-21 (miedo extremo)
- Modo activo: Wild Mode (scalping 30m con martingala)
- 2 posiciones abiertas: WIF +1.04%, POPCAT -0.90%

## Commits incluidos (desde reset Apr 13)
- de5ea3d fix: tarjeta P&L Realizado muestra realized_pnl
- ffdc5d3 feat: barra de resumen P&L reactiva en historial
- e55ae68 fix: WILD_* close reason descriptions en dashboard
- dd27484 feat(SOL-8): dual position sizing Wild/Pure
- 00c6380 fix: logica contrarian F&G invertida correctamente
- 970e6d2 fix: fallback Codex -> MiniMax en limites

## Por que es la mejor version
Primera version que logra rentabilidad real y sostenida en 24h.
Las anteriores tenian bugs criticos (969 EMERGENCY_CLOSEs, logic invertida en F&G,
sizing demasiado agresivo). Esta version tiene:
- Wild Mode con scalping 30m calibrado
- Dual sizing (Wild 1.5%/10x, Pure 2%/5x)
- MART_CORE_TOKENS guard (sin martingala en meme coins)
- Dashboard funcional con equity curve correcta

## Como restaurar
```bash
git checkout v1.0-rentable   # checkout del tag
# O para restaurar datos de runtime:
cp snapshots/v1.0-rentable-2026-04-13/* agents/data/
```
