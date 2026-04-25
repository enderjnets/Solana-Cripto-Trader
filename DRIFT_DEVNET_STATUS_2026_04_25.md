# Drift Devnet Smoke Test — Estado 2026-04-25

## TL;DR (UPDATED 2026-04-25 14:30 UTC — TERCERA ITERACIÓN)
**Drift integration BLOQUEADA en ambos networks (devnet + mainnet)**. Driftpy 0.8.89 (último released hace 2+ meses) tiene IDL/discriminadores desactualizados respecto al programa Drift on-chain en ambos environments:
- **Devnet**: `PerpMarketNotFound 6078` en account discrimination
- **Mainnet**: `InstructionFallbackNotFound 101` (Anchor error) en `initialize_user` — discriminator de la ix no reconocido

**Lo que sí funciona**: read-only paths (snapshot, oracle price, funding rate) — esos usan deserialización tolerante.

**Lo que NO funciona**: cualquier write op (initialize_user, deposit_usdc, place_perp_order, etc.) en MAINNET o DEVNET.

**Bloqueo es 100% externo**: depende de release de driftpy 0.8.90+ por Drift Labs (último release 2026-02-18, 2+ meses sin update). Sin ETA público.

**Cero costo financiero**: Tests fallaron en preflight (simulación), no se sometieron txs. SOL fees mínimos por simulación (~$0.001). Wallet intacta.

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

## ❌ Mainnet $2-5 test ATTEMPTED — también falló (2026-04-25 14:30 UTC)

User autorizó proceder con mainnet. Ejecuté `tools/drift_setup.py --env mainnet --deposit 2.0`:
- Step 1 (initialize_user): falló con `InstructionFallbackNotFound 101` en simulación
- Tx NO sometida (preflight fail)
- Cero costo: $0.00 perdidos (solo CPU de simulación local)
- Wallet intacta: $90.20 USDC + 0.038 SOL + 3 spot positions OK

Esto confirma que driftpy 0.8.89 está **stale en ambos networks**, no solo devnet. La hipótesis previa de "mainnet OK" era falsa — el read path funcionaba pero el write path nunca fue testeado en mainnet hasta hoy.

## Plan alternativo recomendado original (DESCARTADO ahora):

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
- v2.12.32.3 corriendo en mainnet spot
- Drift code 100% gated detrás de `DRIFT_ENABLED=false` — cero impacto del trabajo de Drift en operación spot
- 3 trades simultaneos activos (JUP+SOL+ETH) post-cleanup auto-learner

## Recomendación FINAL
**Drift integration está bloqueada hasta nuevo aviso por upstream**. No hay forma de desbloquear desde nuestro lado.

### Acciones a hacer cuando driftpy 0.8.90+ release:
1. `pip install --upgrade driftpy` en `.venv-drift`
2. Re-test devnet first (segundo, gratis): `tools/drift_devnet_smoke.py --env devnet --size 0.01`
3. Si devnet pasa → mainnet $2-5 test con autorización fresca
4. Si mainnet $2-5 pasa → activar DRIFT_ENABLED en branch separado

### Mientras tanto:
1. **Spot validation continúa**: 3 trades simultaneos llenan sample para gate-check (~6-9 trades/día esperado, gate=50)
2. **Monitorear** `https://github.com/drift-labs/driftpy/releases` para nuevas versiones
3. **Considerar TypeScript SDK**: drift-labs/protocol-v2 (TS) está más activamente mantenido que driftpy. Si Drift es prioridad, podríamos prototipar bridge Python→TS via subprocess. Alta complejidad pero viable.

### Costo total de la sesión 2026-04-25
- Tiempo invertido: ~3 horas (audit + investigación + 2 attempts mainnet)
- Costo financiero: **$0.00** (todos los failures fueron en preflight/simulación)
- Aprendizaje: mapa completo del estado Drift, plan claro para cuando upstream esté listo
