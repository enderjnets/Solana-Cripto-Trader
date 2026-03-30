#!/bin/bash
# Daily backup script for Solana Trading Bot
# Add to crontab: 0 3 * * * /path/to/daily_backup.sh

cd /home/enderj/.openclaw/workspace/Solana-Cripto-Trader
python3 agents/backup_manager.py backup

# Keep logs
echo "[$(date)] Daily backup completed" >> agents/data/backup_log.txt