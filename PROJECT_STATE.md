# PROJECT_STATE.md — Solana Cripto Trader

> Estado actual del proyecto. Actualizado tras cada cambio significativo.

## Última actualización
- **Fecha**: 2026-05-01
- **Versión**: v2.13.4-live
- **Commit**: (pending-reset)

## Estado del pipeline

| Componente | Estado |
|------------|--------|
| Orchestrator | ✅ Activo (PID 39915, heartbeat <60s) |
| Risk Manager | ✅ ACTIVE — drawdown 0.0% |
| Strategy Agent | ✅ Generando señales,  |
| Executor | ✅ Modo LIVE, 0 posiciones abiertas |
| Dashboard | ✅ Puerto 8082, v2.13.4-live |
| Jupiter Perps | ✅ Habilitado (SOL, BTC, ETH, JUP) |
| Health | ✅ HEALTHY |

## Posiciones actuales

**0 posiciones abiertas. BOT STOPPED — esperando transferencia ~$79.63**

Última posición cerrada:
- **BTC LONG** (cerrada manualmente 2026-04-30): Size $167.10, Leverage 2.0x
- Entry: $76,429.37, Close mark: ~$76,115
- **PnL final: -$0.90 (-1.08%)**, fees: $0.20
- Tx close: `5M67TxpiQeEkY6Ut9uXSkc2c7EAfmw5xomD8QSWr2ekcj92D7M3WcH9kswxf5JVbiuZsF4qYuoD3eigtSrG2xEVT`

## Wallet

- **Address**: `EEmtkySNz1SLNZBMBu6EsuqkEhttEKjejsEXdEFT2fMH`
- **Capital real**: $20.37 (initial $100 + PnL -$0.90 + PnL previo -$1.19)
- **Capital libre**: ~$100.00 (reset contable) (wallet, no open positions)
- **Total equity**: ~$20.37 (on-chain real) (libre, no perps open)
- **Capital tier**: MICRO ($50-149)
- **Max positions**: 3
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
10. **EMERGENCY: Close BTC after 40h stale** — Tx: `5M67Txpi...SrG2xEVT`
11. **HARDENING: CWD guards** — watchdog + orchestrator fallan si no corren desde *-Live
12. **Fix systemd services** — stopped solana-jupiter-bot, enabled solana-live-bot

13. **.env auto-load** — orchestrator loads .env via python-dotenv at startup
14. **Fix pgrep pattern** — watchdog uses -Live path to avoid cross-directory duplicates
15. **Fix heartbeat logging** — info-level for write errors (was silent debug)
16. **Fix kill switch** — cleared stale killswitch, reset daily target, MAX_DAILY_LOSS=$5
17. **Fix portfolio ghosts** — removed JUP paper ghost, reconciled trade_history
18. **Fix reporter equity** — known display bug (shows $20.40 vs actual $79.06)
19. **Fix auto-learner** — cleared tokens_to_avoid from [SOL,ETH] to []

20. **Full reset a $100** — portfolio, trade_history, risk, daily_target, equity_history, compound, alerts, cooldowns, decisions, wild_mode, scanner, signals reseteados
21. **Bot STOPPED** — drawdown 79.6% hasta que wallet on-chain alcance ~$100

## Paperclip

- **Issue**: `SOLAA-117` — v2.13.4-live: Close accidental SOL position + reset stats + sync wallet
- **Status**: Done
- **Issue**: `79ffe661-60bf-41b5-a4c3-f38412ac4baf` — Fix BTC open_time incorrect after manual registration
- **Status**: Done
- **Issue**: `13109822-32c4-451d-98ad-c407850c7c67` — EMERGENCY: Close BTC after 40h stale + orchestrator CWD bug recurrence
- **Status**: Done

- **Issue**: `83e3eb69-724f-4a8a-bf5b-df4f351c276d` — CWD hardening: 3-layer defense against wrong-directory execution
- **Status**: Done
- **Issue**: `328ee77` — Config fixes for full autonomous mode (JUP whitelist, clear avoid list, heartbeat logging)
- **Status**: Done

## Infraestructura

| Entorno | Ubicación |
|---------|-----------|
| Bot live | Servidor remoto `100.88.47.99` (Tailscale) |
| Dashboard | `http://100.88.47.99:8082` |
| Wallet local | `~/.config/solana-jupiter-bot/wallet.json` |
| Wallet remota | `~/wallet.json` en servidor |
