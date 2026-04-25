# Drift Devnet Smoke Test — Estado 2026-04-25

## TL;DR (UPDATED 2026-04-25 14:00 UTC)
**Root cause confirmado**: Drift Labs upgradó el programa Drift en **devnet** sin lanzar driftpy compatible. driftpy 0.8.89 (último released) **funciona en mainnet pero falla en devnet** con `PerpMarketNotFound 6078`. Bloqueo es **externo**, no resoluble del lado client sin upgrade de driftpy upstream.

**Ramificación práctica**: smoke test devnet **bloqueado indefinidamente** hasta release de driftpy 0.8.90+. Pero driftpy 0.8.89 + Drift mainnet **es plenamente funcional** — podemos saltar devnet y validar directamente en mainnet con monto pequeño ($2-5 test).

## Lo que está listo (95% wired en `live` branch)
- `agents/drift_client.py` (17.7KB) — full async wrapper de driftpy con `open_sol_perp_market`, `open_sol_perp_limit`, `close_sol_perp`, `deposit_usdc`, `initialize_user`, `snapshot`.
- `agents/drift_adapter.py` (10.6KB) — sync wrapper con 8 safety gates.
- `tools/drift_setup.py` + `drift_devnet_smoke.py` + `drift_devnet_mint_usdc.py`.
- `requirements.txt` ya tiene `driftpy>=0.8.89`.
- `.env` ya tiene todos los DRIFT_* vars.
- `.venv-drift/` creado en LIVE workspace.

## Diagnóstico técnico (2026-04-25 sesión nocturna + matutina)

### Read path: ✅ funciona end-to-end (devnet + mainnet)
- `drift_setup.py --status` pasa
- snapshot + oracle + funding rate todo OK

### Write path en devnet: ❌ falla en TODO ix de perp order
- `place_and_take_perp_order` → `PerpMarketNotFound 6078`
- `place_perp_order` (limit) → mismo `PerpMarketNotFound 6078`
- Confirma que NO es bug de un método específico — es bug de **account discrimination en el programa Drift devnet**

### Pruebas hechas
- ✅ SOL-PERP PDA exists devnet (`8UJg...`, owner=Drift program, data_len=1216)
- ✅ Drift program devnet executable, accounts enabled
- ✅ `get_perp_market_account(0)` deserializa OK con `market_index=0`, `status=Active`, `name=SOL-PERP`
- ❌ `pm.insurance_claim` no tiene campo esperado por driftpy → schema diff vs versión instalada
- ✅ `add_perp_market_to_remaining_account_maps(0, True, ...)` retorna pubkey correcto
- ✅ `get_remaining_accounts(writable_perp_market_indexes=[0])` retorna 4 accounts (oracle×2, spot×1, perp×1) en orden correcto
- ❌ Drift program rechaza el ix con "Could not find perp market 0 at programs/drift/src/instructions/optional_accounts.rs:52"
- ❌ Mismo error con `place_perp_order` (limit, no fill) — descarta hipótesis de PlaceAndTake-specific
- ✅ **Mainnet** SOL-PERP: driftpy 0.8.89 deserializa con `insurance_claim` válido (last_revenue_withdraw_ts, max_revenue_withdraw_per_period, etc.) — confirma compatibilidad mainnet

### Investigación externa
- GitHub drift-rs issue #38 + PR #39: confirma "race condition" donde market accounts no se incluyen en TX, fix en drift-rs (Rust SDK) pero NO portado a driftpy aún
- driftpy 0.8.89 = último release publicado (master HEAD del repo aún en 0.8.89)
- Devnet program = bleeding edge, mainnet program = estable/older

## Plan alternativo recomendado: skip devnet, mainnet $2-5 test

**Justificación**: devnet bloqueo es externo y no tiene ETA. Mainnet es nuestro target real y funciona con driftpy 0.8.89.

### Risk profile mainnet $2-5 test
- 🟡 Capital REAL pero MÍNIMO ($2-5 collateral en Drift, NO se mueven los $90 USDC del bot spot)
- 🟢 Wallet `EEmtky...` ya tiene $90 USDC + 0.039 SOL fuel + Drift user inicializado
- 🟢 1x leverage solamente (no apalancamiento, equivale a spot direccional)
- 🟢 Smoke test = open + close en <2 min, fees ~$0.01
- 🟢 8 safety gates de `drift_adapter.py` activos
- 🟢 DRIFT_MAX_COLLATERAL_USD=3 + DRIFT_MAX_LEVERAGE=2 ya en .env

### Pasos (cuando user autorice)
1. Deposit $2 USDC adicional al subaccount Drift mainnet (de wallet existente, no toca capital del spot bot)
2. Run smoke test mainnet: `tools/drift_devnet_smoke.py --env mainnet --size 0.005` (0.005 SOL ≈ $0.40 notional con 1x leverage)
3. Verificar: open SOL-PERP long, hold 30s, close, collateral preserved ±2%
4. Repeat short
5. Si pasa → retirar collateral hasta gate-check spot pase (≥30 trades, PF≥1.5)
6. Cuando habilitemos: DRIFT_ENABLED=true en branch separado primero, gate-check, merge a live

### Si user prefiere esperar
- Monitorear `https://github.com/drift-labs/driftpy/releases` para driftpy 0.8.90
- ETA desconocido (Drift Labs no publica roadmap fix)
- Mientras tanto: spot bot v2.12.32.x corre intacto, validación 7 días sigue su curso

## Estado del bot live (intacto)
- v2.12.32.1 corriendo en mainnet spot
- Drift code 100% gated detrás de `DRIFT_ENABLED=false` — cero impacto del trabajo de Drift en operación spot
- 3 trades simultaneos activos (JUP+SOL+ETH) post-cleanup auto-learner

## Recomendación
**No habilitar DRIFT_ENABLED=true automáticamente.** Esperar autorización explícita del user para mainnet $2-5 test.

Mientras user no llega, pivotar a:
1. Monitor del bot spot (3 trades activos = oportunidad de validar v2.12.32.1)
2. Cleanup pendientes (.env duplicates, weekly_report.py si no existe)
3. Reboot ROG cuando todos los trades cierren (TIME_EXIT >24h)
