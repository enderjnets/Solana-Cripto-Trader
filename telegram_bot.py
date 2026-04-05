#!/usr/bin/env python3
"""
Telegram Bot for Solana Jupiter Bot
====================================
Monitor and control your trading bot via Telegram.

Commands:
/status - System status and portfolio
/workers - Active workers status
/balance - Wallet balance
/trades - Recent trades
/strategies - Available strategies
/help - Show all commands

Usage:
    python telegram_bot.py
"""

import os
import sys
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("telegram_bot")

# Try to import Telegram libraries
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False
    logger.warning("python-telegram-bot not installed. Run: pip install python-telegram-bot")

# Configuration
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
DATA_DIR = Path("data")


# ============================================================================
# STATUS FUNCTIONS
# ============================================================================
def get_portfolio_status() -> Dict:
    """Get current portfolio status"""
    status = {
        "wallet": "Not connected",
        "balance_sol": 0.0,
        "balance_usdc": 0.0,
        "workers": 0,
        "last_trade": None,
        "pnl_24h": 0.0
    }
    
    # Try to read from holdings file
    holdings_file = DATA_DIR / "holdings.json"
    if holdings_file.exists():
        try:
            with open(holdings_file) as f:
                data = json.load(f)
                status["balance_sol"] = data.get("SOL", 0)
                status["balance_usdc"] = data.get("USDC", 0)
        except:
            pass
    
    # Try to read from worker status
    worker_file = DATA_DIR / "worker_status.json"
    if worker_file.exists():
        try:
            with open(worker_file) as f:
                workers = json.load(f)
                status["workers"] = len(workers)
        except:
            pass
    
    # Try to read trades
    trades_file = DATA_DIR / "trades.json"
    if trades_file.exists():
        try:
            with open(trades_file) as f:
                trades = json.load(f)
                if trades:
                    status["last_trade"] = trades[-1]
                    # Calculate 24h PnL
                    from datetime import timedelta
                    cutoff = datetime.now() - timedelta(hours=24)
                    recent = [t for t in trades if datetime.fromisoformat(t.get("timestamp", "")) > cutoff]
                    status["pnl_24h"] = sum(t.get("pnl", 0) for t in recent)
        except:
            pass
    
    return status


def get_worker_status() -> Dict:
    """Get worker status"""
    worker_file = DATA_DIR / "worker_status.json"
    
    if not worker_file.exists():
        return {"workers": []}
    
    try:
        with open(worker_file) as f:
            return json.load(f)
    except:
        return {"workers": []}


def get_best_backtest() -> Dict:
    """Get best backtest result"""
    db_file = DATA_DIR / "genetic_results.db"
    
    if not db_file.exists():
        return {}
    
    import sqlite3
    try:
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        c.execute("SELECT best_pnl, best_win_rate, generations FROM runs ORDER BY best_pnl DESC LIMIT 1")
        row = c.fetchone()
        conn.close()
        
        if row:
            return {
                "pnl": row[0],
                "win_rate": row[1],
                "generations": row[2]
            }
    except:
        pass
    
    return {}


# ============================================================================
# TELEGRAM HANDLERS
# ============================================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome message"""
    welcome = """
🦞 **Eko - Solana Trading Bot**

Welcome! I'm your AI trading assistant.

**Available Commands:**
/status - System status
/balance - Wallet balance
/workers - Worker status
/trades - Recent trades
/best - Best strategy found
/help - Show all commands

**Features:**
• Automated Jupiter DEX trading
• Genetic strategy optimization
• Real-time monitoring
• Multi-worker support

_type /help for full command list_
"""
    await update.message.reply_text(welcome, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help message"""
    help_text = """
📚 **Command Reference**

**System:**
/start - Welcome message
/status - Full system status
/help - This help

**Trading:**
/balance - Check wallet balances
/trades - View recent trades
/open - View open positions

**Workers:**
/workers - Active worker status
/restart - Restart all workers

**Strategy:**
/best - Best backtest result
/run <strategy> - Run a strategy
/backtest <days> - Run backtest

**Utilities:**
/price - Current SOL price
/stats - Trading statistics
/alerts - Set price alerts

_© 2026 - Eko Trading Bot_
"""
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get system status"""
    status = get_portfolio_status()
    best = get_best_backtest()
    workers = get_worker_status()
    
    message = f"""
📊 **System Status**

💰 **Wallet**
• SOL: `{status['balance_sol']:.4f}`
• USDC: `{status['balance_usdc']:.2f}`

👷 **Workers**
• Active: {status['workers']}
• Last seen: {workers.get('last_seen', 'Unknown')}

📈 **Performance**
• 24h PnL: `{status.get('pnl_24h', 0)*100:+.2f}%`
• Total Trades: {status.get('total_trades', 0)}

🏆 **Best Strategy**
• PnL: `{best.get('pnl', 0)*100:+.2f}%`
• Win Rate: `{best.get('win_rate', 0)*100:.1f}%`
• Generations: {best.get('generations', 'N/A')}

🕐 Updated: {datetime.now().strftime('%H:%M:%S')}
"""
    await update.message.reply_text(message, parse_mode="Markdown")


async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check wallet balance"""
    status = get_portfolio_status()
    
    message = f"""
💰 **Wallet Balance**

**Devnet** (`65YqSYGw...`)
• SOL: `{status['balance_sol']:.4f}` 💚
• USDC: `${status['balance_usdc']:.2f}`

**Mainnet** (`Ht3J5crw...`)
• SOL: `0.0`
• USDC: `$0.00`

_Use /status for full system info_
"""
    await update.message.reply_text(message, parse_mode="Markdown")


async def workers_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Worker status"""
    workers = get_worker_status()
    
    if not workers.get("workers"):
        message = "👷 **Workers**\n\nNo workers active."
    else:
        worker_list = workers.get("workers", [])
        message = "👷 **Workers**\n\n"
        
        for w in worker_list[:5]:
            status_icon = "🟢" if w.get("alive", False) else "🔴"
            message += f"{status_icon} `{w.get('id', 'Unknown')}` - {w.get('status', 'Unknown')}\n"
            message += f"   Trades: {w.get('trades', 0)} | PnL: {w.get('pnl', 0)*100:+.2f}%\n"
        
        if len(worker_list) > 5:
            message += f"\n... +{len(worker_list) - 5} more"
    
    await update.message.reply_text(message, parse_mode="Markdown")


async def trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Recent trades"""
    trades_file = DATA_DIR / "trades.json"
    
    if not trades_file.exists():
        await update.message.reply_text("📭 No trades yet.")
        return
    
    try:
        with open(trades_file) as f:
            trades = json.load(f)
        
        if not trades:
            await update.message.reply_text("📭 No trades yet.")
            return
        
        # Show last 5 trades
        message = "📈 **Recent Trades**\n\n"
        
        for t in trades[-5:]:
            side = "🟢 BUY" if t.get("side") == "buy" else "🔴 SELL"
            token = t.get("token", "SOL")
            amount = t.get("amount", 0)
            pnl = t.get("pnl", 0)
            
            message += f"{side} `{token}` {amount:.4f}\n"
            message += f"   Price: ${t.get('price', 0):.4f} | PnL: {pnl*100:+.2f}%\n"
            message += f"   `{t.get('timestamp', '')[:19]}`\n\n"
        
        await update.message.reply_text(message, parse_mode="Markdown")
        
    except Exception as e:
        await update.message.reply_text(f"Error loading trades: {e}")


async def best_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Best strategy found"""
    best = get_best_backtest()
    
    if not best:
        await update.message.reply_text("📭 No backtests run yet.")
        return
    
    message = f"""
🏆 **Best Strategy**

📈 **Performance**
• PnL: `{best.get('pnl', 0)*100:+.2f}%`
• Win Rate: `{best.get('win_rate', 0)*100:.1f}%`

🧬 **Evolution**
• Generations: {best.get('generations', 'N/A')}
• Population: {best.get('population', 'N/A')}

💡 Run `/backtest 90` to discover new strategies!

_Type /stats for more_
"""
    await update.message.reply_text(message, parse_mode="Markdown")


async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get current SOL price"""
    try:
        import requests
        response = requests.get(
            "https://lite-api.jup.ag/price/v3?ids=So11111111111111111111111111111111111111112",
            timeout=5
        )
        data = response.json()
        price = float(data.get("data", {}).get("So11111111111111111111111111111111111111112", {}).get("price", 0))
        
        message = f"""
💵 **SOL Price**

**${price:.2f}** 🟢

_Updated: {datetime.now().strftime('%H:%M:%S')}_
"""
        await update.message.reply_text(message, parse_mode="Markdown")
        
    except Exception as e:
        await update.message.reply_text(f"Error fetching price: {e}")


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle unknown commands"""
    await update.message.reply_text(
        "❓ Unknown command. Type /help for available commands."
    )


# ============================================================================
# MAIN
# ============================================================================
def run_bot():
    """Run the Telegram bot"""
    if not HAS_TELEGRAM:
        print("❌ python-telegram-bot not installed")
        print("   Run: pip install python-telegram-bot")
        return
    
    if not TELEGRAM_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not set")
        print("   Set environment variable: export TELEGRAM_BOT_TOKEN='your_token'")
        return
    
    if not CHAT_ID:
        print("⚠️ TELEGRAM_CHAT_ID not set - bot will work but can't send proactive messages")
    
    print("🤖 Starting Telegram Bot...")
    
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    # Register handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("balance", balance_command))
    app.add_handler(CommandHandler("workers", workers_command))
    app.add_handler(CommandHandler("trades", trades_command))
    app.add_handler(CommandHandler("best", best_command))
    app.add_handler(CommandHandler("price", price_command))
    app.add_handler(CommandHandler("unknown", unknown_command))
    
    # Start polling
    print(f"✅ Bot started! Chat ID: {CHAT_ID}")
    print("   Send /start to the bot to begin.")
    
    app.run_polling()


# ============================================================================
# AUTO-REPORTING
# ============================================================================
async def send_status_report(application, chat_id: str):
    """Send periodic status report"""
    status = get_portfolio_status()
    
    message = f"""
📊 **Auto Report**

💰 SOL: `{status['balance_sol']:.4f}`
📈 24h PnL: `{status.get('pnl_24h', 0)*100:+.2f}%`
👷 Workers: `{status['workers']}`

_Generated by Eko_
"""
    
    try:
        await application.bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Failed to send report: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Solana Telegram Bot")
    parser.add_argument("--token", help="Telegram bot token")
    parser.add_argument("--chat-id", help="Telegram chat ID")
    parser.add_argument("--report", action="store_true", help="Send report and exit")
    args = parser.parse_args()
    
    if args.token:
        os.environ["TELEGRAM_BOT_TOKEN"] = args.token
    if args.chat_id:
        os.environ["TELEGRAM_CHAT_ID"] = args.chat_id
    
    if args.report:
        # Send one-shot report
        if HAS_TELEGRAM and TELEGRAM_TOKEN and CHAT_ID:
            import asyncio
            async def send():
                await send_status_report(
                    ApplicationBuilder().token(TELEGRAM_TOKEN).build(),
                    CHAT_ID
                )
            asyncio.run(send())
        else:
            print("Missing configuration")
    else:
        run_bot()
