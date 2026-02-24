#!/usr/bin/env python3
"""
OpenClaw Message Bridge
Sends messages to Telegram via OpenClaw's messaging system
"""
import os
import logging
import subprocess
from typing import Optional

logger = logging.getLogger("openclaw_tools")


def send_telegram_message(chat_id: str, message: str, priority: str = "normal") -> bool:
    """
    Send a message to Telegram via OpenClaw message system

    Args:
        chat_id: Telegram chat ID
        message: Message content
        priority: Message priority (normal, high, critical)

    Returns:
        True if sent successfully, False otherwise
    """
    try:
        # Check if OpenClaw is available
        result = subprocess.run(
            ["openclaw", "message", "send"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            # OpenClaw is available, use message tool
            # We'll use the message action via openclaw CLI
            import json

            # Build the command
            cmd = [
                "openclaw", "message", "send",
                "--channel", "telegram",
                "--target", chat_id,
                "--message", message
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                logger.info(f"✅ Message sent via OpenClaw: {message[:50]}...")
                return True
            else:
                logger.warning(f"OpenClaw message failed: {result.stderr}")
                return False
        else:
            logger.warning("OpenClaw not available, using fallback")
            return False

    except FileNotFoundError:
        logger.debug("OpenClaw CLI not found")
        return False
    except Exception as e:
        logger.error(f"Failed to send via OpenClaw: {e}")
        return False


def send_message_to_chat(chat_id: str, message: str) -> bool:
    """
    Simplified function to send message to a Telegram chat

    This is used by the notification system
    """
    return send_telegram_message(chat_id, message, priority="normal")
