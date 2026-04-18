# Solana Cripto Trader — LIVE Onboarding Guide (Sprint 2)

**Versión:** v2.10.0-live (branch `live`)
**Última actualización:** 2026-04-18

Este documento describe cómo pasar del paper (currently running) al live trading real.
Complementa `MAINNET_GUIDE.md` con los pasos específicos del Sprint 2.

---

## Estado actual del live

| Componente | Estado | Archivo |
|------------|--------|---------|
| `agents/safety.py` | ✅ done (Sprint 1) | 7 capas de defensa |
| `agents/solana_rpc.py` | ✅ done (Fase 1) | Helius RPC wrapper |
| `agents/wallet.py` | ✅ done (Fase 1) | MockWallet + LiveWallet |
| `agents/jupiter_swap.py` | ✅ done (Fase 2) | Swap wrapper v1 API |
| `agents/executor.real_open_position()` | ✅ done (Fase 3) | Real swap execution |
| `agents/reconcile.py` | ✅ done (Fase 4) | On-chain balance check |
| `tools/emergency_close.py` | ✅ done (Fase 5) | Heartbeat watchdog |
| `orchestrator` heartbeat + reconcile hooks | ✅ integrated | Each cycle |
| **Testing devnet + mainnet $50** | ⚠️ **pending** (Fase 6) | Requiere acción humana |

---

## Arquitectura de defensa — 9 capas

Cada trade live tiene que pasar las siguientes validaciones secuenciales:

1. **TRADE_WHITELIST_STRICT** (hardcoded) — solo majors {SOL}
2. **safety.is_whitelisted** (env TRADE_WHITELIST) — restricción adicional opcional
3. **MIN_SL_DISTANCE_PCT=2.0** — rechaza SL <2% del entry
4. **MAX_NOTIONAL_PCT_EQUITY=0.50** — cap notional 50% del equity
5. **LIVE_TRADING_ENABLED=true** — master switch (default false)
6. **direction=long** — Jupiter no soporta shorts
7. **symbol en MINT_MAP** — mint resolvible (actualmente solo SOL)
8. **wallet.is_live** — MockWallet refusa firmar
9. **size_usd >= $1** — Jupiter mínimo

Más salvaguardas operacionales:
- **`safety.check_daily_loss`** — pausa cuando diario supera `MAX_DAILY_LOSS_USD`
- **`safety.is_kill_switch_active`** — file-based pausa manual `/tmp/solana_live_killswitch`
- **`safety.validate_startup`** — boot gate que exige config completa si LIVE=true
- **`reconcile.check_reconciliation`** — cada 10 ciclos verifica balances on-chain
- **`tools/emergency_close.py --check`** (cron cada 60s) — si heartbeat stale, cierra todo

---

## Checklist de tu parte antes de activar LIVE_TRADING_ENABLED=true

### 1. Generar wallet dedicada

```bash
cd ~/.openclaw/workspace/Solana-Cripto-Trader-Live
python3 tools/solana_wallet.py --generate
# Output: Address: XXX... | Private Key: YYY...
```

⚠️ **Guarda la private key en un lugar seguro** (password manager). NUNCA la commites.

### 2. Pegar credentials en .env (gitignored)

Edita `~/.openclaw/workspace/Solana-Cripto-Trader-Live/.env`:

```
LIVE_TRADING_ENABLED=false    # mantener false hasta Paso 4
HOT_WALLET_ADDRESS=<tu-pubkey-aquí>
HOT_WALLET_PRIVATE_KEY=<tu-privkey-base58-aquí>
SOLANA_RPC_URL=https://mainnet.helius-rpc.com/?api-key=<tu-helius-key>
```

### 3. Validar setup (sin dinero real todavía)

```bash
cd ~/.openclaw/workspace/Solana-Cripto-Trader-Live

# RPC reachable?
python3 -c "from agents.solana_rpc import get_rpc; print('health:', get_rpc().get_health())"

# Wallet load correctly? (aún con LIVE=false debería cargar a MockWallet)
python3 -c "from agents.wallet import load_wallet; w=load_wallet(); print('wallet:', w, 'is_live:', w.is_live)"

# safety validation pasa?
python3 -c "from agents.safety import validate_startup; print('errors:', validate_startup())"
```

Todo debe responder sin errores.

### 4. Devnet testing (GRATIS, 24h)

```bash
# Cambiar RPC a devnet temporalmente
export SOLANA_RPC_URL="https://api.devnet.solana.com"

# Obtener SOL gratis del faucet
solana airdrop 5 <tu-pubkey> --url devnet
# o via web: https://faucet.solana.com

# Activar live en devnet
export LIVE_TRADING_ENABLED=true
export TRADE_WHITELIST=SOL
export MAX_DAILY_LOSS_USD=5

# Correr el bot 24h
python3 dashboard/app.py 8082 &   # dashboard
python3 agents/orchestrator.py    # orchestrator
```

**Validaciones a observar:**
- [ ] Al menos 10 swaps exitosos (verificar en dashboard)
- [ ] Reconciliación on-chain matches portfolio.json
- [ ] SL/TP disparan correctamente cuando precio se mueve
- [ ] Emergency close funciona: kill orchestrator → esperar 5min → verificar que emergency_close.py cerró posiciones
- [ ] Ningún error en logs

### 5. Mainnet shadow ($50, 48h)

Solo después de 24h de devnet exitoso:

```bash
# Fondear wallet mainnet con $50 USDC + 0.05 SOL (fees)
# Vía exchange → retirar a tu pubkey

# Config conservadora
export LIVE_TRADING_ENABLED=true
export SOLANA_RPC_URL=https://mainnet.helius-rpc.com/?api-key=XXX
export MAX_DAILY_LOSS_USD=5
export MAX_SLIPPAGE_BPS=100
export TRADE_WHITELIST=SOL
export LIVE_MAX_POSITION_USD=20
export LIVE_MAX_POSITIONS=1

# Arrancar bot
python3 agents/orchestrator.py

# Setup cron para emergency_close cada 60s:
crontab -e
# Añadir:
# * * * * * cd /ruta/al/clone && /usr/bin/python3 tools/emergency_close.py --check >> /tmp/emergency_close.log 2>&1
```

**48h mainnet shadow — criterios éxito:**
- [ ] P&L del live dentro de ±10% del paper (slippage real cerca del simulado)
- [ ] 0 incidents de reconcile crítico
- [ ] 0 trades con price_impact >1%
- [ ] Claude/MiniMax respondiendo sin cascadas de falla

### 6. Mainnet limited ($100-200, 7 días)

Después de 48h shadow exitoso, ampliar gradualmente:

```bash
export TRADE_WHITELIST="SOL"   # mantener solo SOL hasta agregar más MINT_MAP entries
export MAX_DAILY_LOSS_USD=10
export LIVE_MAX_POSITION_USD=50
export LIVE_MAX_POSITIONS=2
```

Monitorear 7 días. Si P&L neto positivo (o neutral) + sin incidents → considerar full production.

---

## Emergency procedures

### 🚨 Pausar bot YA

```bash
# Opción 1: kill switch file (orchestrator next cycle detecta y skip)
touch /tmp/solana_live_killswitch

# Opción 2: activar via API (requiere dashboard vivo)
curl -X POST http://localhost:8082/api/safety/kill-switch \
  -H "Content-Type: application/json" \
  -d '{"action":"activate","reason":"manual pause"}'

# Opción 3: matar orchestrator process (emergency_close cron lo detectará)
kill $(cat /tmp/solana_live_orchestrator.lock)
```

### 🚨 Cerrar TODAS las posiciones YA

```bash
cd ~/.openclaw/workspace/Solana-Cripto-Trader-Live
python3 tools/emergency_close.py --force --dry-run   # primero dry-run
python3 tools/emergency_close.py --force             # real
```

### 🚨 Desactivar live completo

Editar `.env`:
```
LIVE_TRADING_ENABLED=false
```
Reiniciar orchestrator. El bot vuelve a paper-like behavior (real_open_position retorna None → no ejecuta).

---

## Troubleshooting

### Claude 401 Unauthorized

Tu account Max OAuth token venció. Refresh:
```bash
claude login
```

### Jupiter "no route found"

El token que intentas comprar no tiene liquidez suficiente o no existe en Solana.
Verifica que esté en `MINT_MAP` de `agents/executor.py`.

### Reconcile critical discrepancy

Algo divergió entre portfolio.json y on-chain. Kill switch se activa automático.
1. Revisa logs: `tail -f /home/enderj/.config/solana-jupiter-bot/modular.log | grep reconcile`
2. Verifica manualmente balances: `python3 -c "from agents.solana_rpc import get_rpc; r=get_rpc(); print(r.get_balance_sol('tu-pubkey'))"`
3. Si discrepancia es real → cierra manualmente via `emergency_close.py --force`

### Rate limits

Helius free tier: 500k req/mes. Con cache v2.6.0 del bot, debería estar bien para 1 instancia.
Si te pegas: upgrade a $49/mes o crear nueva cuenta free.

---

## Contactos de emergencia

- **Helius dashboard:** https://dev.helius.xyz
- **Jupiter status:** https://stats.jup.ag
- **Solana status:** https://status.solana.com
- **Tu wallet explorer:** https://solscan.io/account/YOUR_PUBKEY

---

## Checklist final antes de "full production"

- [ ] 24h devnet exitoso (Paso 4)
- [ ] 48h mainnet shadow $50 (Paso 5)
- [ ] 7 días mainnet limited $100-200 (Paso 6)
- [ ] `emergency_close.py` corriendo en cron cada 60s
- [ ] Heartbeat visible cada ciclo en dashboard
- [ ] Paperclip SOLAAA tiene último issue "Full Production Ready" en done
- [ ] User ha practicado manualmente: kill switch, emergency close, resume

**Solo entonces considerar ampliar capital.**
