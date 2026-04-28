# 🧠 Solana Cripto Trader — Estado del Proyecto (Memoria viva)

> Este archivo se actualiza después de cada cambio, fix, mejora o release.
> Última actualización: 2026-04-28 v2.13.3-live

---

## 📌 Versión Actual

**Bot:** `v2.13.3-live`  
**Dashboard:** `v2.13.3-live`  
**Branch:** `live`  
**Wallet:** `EEmtkySNz1SLNZBMBu6EsuqkEhttEKjejsEXdEFT2fMH`

---

## 🏦 Capital & Estado de Cuenta

- **On-chain equity:** ~$103.74 (USDC $90.45 + SOL $12.86 + JUP $0.43)
- **Initial capital (baseline):** $100.00
- **Return:** +3.74%
- **Capital tier:** `micro` ($50-$149)
  - Risk per trade: 10% → $15.00 (Jupiter $10 min collateral @ 1.5x)
  - Leverage: 1.5x
  - Max positions: 1

---

## ✅ Cambios Recientes (v2.13.3-live)

### Capital-Tier Auto-Sizing Engine
- `_get_capital_tier()` auto-detecta tier según `wallet_total` on-chain.
- `_get_risk_for_capital()` aplica floor de Jupiter Perps ($10 × leverage).
- `paper_open_position()` usa `fetch_wallet_equity()` como fuente primaria de capital.
- `MAX_POSITIONS` es tier-aware (cruzado con auto-learner).
- `jupiter_perp_adapter.py` eliminó el override silencioso de $10; ahora es guardia estricta.
- Dashboard expone `capital_tier` + `max_positions_tier` y muestra badge color-coded.

---

## 📋 Changelog Rápido

| Versión | Fecha | Tema principal |
|---------|-------|----------------|
| v2.13.3-live | 2026-04-28 | Capital-Tier Auto-Sizing (risk/leverage/positions dinámicos) |
| v2.13.2-live | 2026-04-28 | Expand perps universe: SOL, BTC, ETH + close test position |
| v2.13.1-live | 2026-04-28 | Jupiter Perps safety: on-chain SL/TP + real liq_price + RPC resilience |
| v2.13.0-alpha | 2026-04-25 | Jupiter Perps skeleton (CLI wrapper + adapter) |

---

## 🔧 Configuración Activa (env relevantes)

```
LIVE_TRADING_ENABLED=true
JUP_PERP_ENABLED=true
JUP_PERP_MARKET_WHITELIST=SOL,BTC,ETH
JUP_PERP_DEFAULT_LEVERAGE=2.0
JUP_PERP_MAX_LEVERAGE=3.0
TRADE_WHITELIST=SOL,JUP,ETH,BTC
```

---

## ⚠️ Problemas Conocidos / Watch

1. **RPC rate limiting (residual):** Public RPC aún retorna 429s. Retry logic en `wallet_equity.py` lo maneja, pero private RPC (Helius/QuickNode) recomendado para estabilidad a largo plazo.
2. **Jupiter CLI dry-run limitation:** CLI rechaza `--dry-run` con "Insufficient funds" si wallet no tiene balance real. No es bug del bot, es comportamiento del CLI.
3. **Soft-reset accounting gap:** `initial_capital=100.00` pero wallet on-chain es $103.74. Gap de $3.74 por diseño (baseline manual). Por debajo del umbral de banner ($25).

---

## 🎯 Próximos Pasos / Backlog

- [ ] Evaluar private RPC (Helius/QuickNode) para eliminar 429s
- [ ] Wild Mode + Perps: integrar martingale chains con capital-tier sizing
- [ ] BTC/ETH perps smoke test (primera posición real en BTC o ETH)
- [ ] Auto-learner adaptation: verificar si params se ajustan correctamente bajo tier micro
- [ ] Considerar compound engine integration con tier-aware sizing

---

## 📁 Estructura Clave de Archivos

| Archivo | Propósito |
|---------|-----------|
| `agents/executor.py` | Abre/cierra posiciones, sizing, risk management |
| `agents/jupiter_perp_adapter.py` | Wrapper sync para Jupiter Perps CLI |
| `agents/jupiter_perp_cli_wrapper.py` | Async wrapper @jup-ag/cli |
| `agents/wallet_equity.py` | On-chain equity fetcher con retry |
| `dashboard/app.py` | Flask dashboard + API |
| `safety.py` | Kill switch, daily loss, whitelist |

---

*Mantenido por el agente de desarrollo. Actualizar tras cada release.*
