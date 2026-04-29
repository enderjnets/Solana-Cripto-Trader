# PROJECT_STATE.md — Solana Cripto Trader

> Estado actual del proyecto. Actualizado tras cada cambio significativo.

## Última actualización
- **Fecha**: 2026-04-28
- **Versión**: v2.13.4-live
- **Commit**: (pending)

## Estado del pipeline

| Componente | Estado |
|------------|--------|
| Orchestrator | ✅ Activo (PID 39915, heartbeat <60s) |
| Risk Manager | ✅ ACTIVE — drawdown 0.0% |
| Strategy Agent | ✅ Generando señales,  |
| Executor | ✅ Modo LIVE, 1 posición abierta (BTC long) |
| Dashboard | ✅ Puerto 8082, v2.13.4-live |
| Jupiter Perps | ✅ Habilitado (SOL, BTC, ETH) |
| Health | ✅ HEALTHY |

## Posiciones actuales

**1 posición abierta:**
- **BTC LONG**: Size $167.10, Collateral $83.55, Leverage 2.0x
- Entry: $76,429.37, Mark: ~$76,989
- PnL: ~+$0.67 (+0.80%)
- SL: $73,946.32 | TP: $80,807.32 | Liq: $38,421.90
- Position pubkey: `HfB4rkCYrPaCvhNWJU237YXcU25eE5D9ZZkPTD7oKw6Z`

## Wallet

- **Address**: `EEmtkySNz1SLNZBMBu6EsuqkEhttEKjejsEXdEFT2fMH`
- **Capital real**: $99.71 (initial $100 + PnL -$0.29)
- **Capital libre**: ~$16.18 (wallet)
- **Total equity**: ~$183.29 (libre + perps)
- **Capital tier**: MICRO ($50-149)
- **Max positions**: 1
- **SOL**: ~0.1082

## Fixes aplicados en esta sesión

1. **Cierre posición SOL accidental** — Tx: `2G2RbTN...GMPPvT`
2. **Reset estadísticas** — portfolio, cache, risk, strategy
3. **Sync wallet** — local macOS + servidor remoto
4. **Fix heartbeat** — orchestrator correcto desde `Solana-Cripto-Trader-Live/`
5. **Fix wallet_equity.py** — incluye `jupiter_perps_value`
6. **Fix dashboard** — usa `total_equity` para tier, capital real como principal
7. **Fix accounting_gap** — resta collateral de perps
8. **Fix return_pct** — basado en PnL real, no equity delta
9. **Fix BTC open_time** — corregido de 2026-04-28T22:01:18Z → 2026-04-29T01:41:18+00:00 (coincide con ID timestamp)

## Paperclip

- **Issue**: `SOLAA-117` — v2.13.4-live: Close accidental SOL position + reset stats + sync wallet
- **Status**: Done
- **Issue**: `79ffe661-60bf-41b5-a4c3-f38412ac4baf` — Fix BTC open_time incorrect after manual registration
- **Status**: Done

## Infraestructura

| Entorno | Ubicación |
|---------|-----------|
| Bot live | Servidor remoto `100.88.47.99` (Tailscale) |
| Dashboard | `http://100.88.47.99:8082` |
| Wallet local | `~/.config/solana-jupiter-bot/wallet.json` |
| Wallet remota | `~/wallet.json` en servidor |
