# PROJECT_STATE.md — Solana Cripto Trader

> Estado actual del proyecto. Actualizado tras cada cambio significativo.

## Última actualización
- **Fecha**: 2026-04-28
- **Versión**: v2.13.4-live
- **Commit**: `e12b458` (remote) / `7adecb6` (local)

## Estado del pipeline

| Componente | Estado |
|------------|--------|
| Orchestrator | ✅ Activo (ciclos cada ~2 min) |
| Risk Manager | ✅ ACTIVE — drawdown 0.0% |
| Strategy Agent | ✅ Generando señales, `blocked=False` |
| Executor | ✅ Modo LIVE, 0 posiciones abiertas |
| Dashboard | ✅ Puerto 8082, v2.13.4-live |
| Jupiter Perps | ✅ Habilitado (SOL, BTC, ETH) |

## Posiciones actuales

**0 posiciones abiertas** (todas cerradas).

Último cierre:
- Asset: SOL
- Side: Long
- Size: $29.91
- PnL: +$0.015
- Tx: `2G2RbTNXSnHAjo3vUCUXDffZ8WVZtBDX5gFFhUxRriNqEnEjDjtTk37D16shS4mjmv9M7xkJtCa2SeoWHgGMPPvT`

## Wallet

- **Address**: `EEmtkySNz1SLNZBMBu6EsuqkEhttEKjejsEXdEFT2fMH`
- **Balance SOL**: ~0.14 SOL
- **Balance USDC**: ~90.40
- **Balance JUP**: ~2.28
- **Equity**: ~$102.49
- **Capital tier**: MICRO ($50-149)
- **Max positions**: 1

## Infraestructura

| Entorno | Ubicación |
|---------|-----------|
| Bot live | Servidor remoto `100.88.47.99` (Tailscale) |
| Dashboard | `http://100.88.47.99:8082` |
| Wallet local | `~/.config/solana-jupiter-bot/wallet.json` |
| Wallet remota | `~/wallet.json` en servidor |

## Issues activos

| Issue | Estado |
|-------|--------|
| Accidental SOL position | ✅ Cerrada |
| Stale LLM signals | ✅ Mitigado (drawdown fix) |
| Wallet equity cache | ✅ Limpio |

## Paperclip

- **Issue**: `SOLAA-117` — v2.13.4-live: Close accidental SOL position + reset stats + sync wallet
- **Status**: Done
