#!/bin/bash
# Wrapper: ejecuta run_watchdog_safe.sh
# Este archivo ahora solo redirige al watchdog seguro
cd /home/enderj/.openclaw/workspace/Solana-Cripto-Trader
exec bash run_watchdog_safe.sh
