#!/usr/bin/env python3
"""
Simple notification sender for OpenClaw
Reads pending notifications and returns them as text
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
NOTIFICATION_LOG = PROJECT_ROOT / "data" / "notifications.log"


def get_pending_notifications():
    """Read and return pending notifications"""
    if not NOTIFICATION_LOG.exists():
        return None

    try:
        with open(NOTIFICATION_LOG, "r", encoding="utf-8") as f:
            content = f.read()

        if content.strip():
            return content
        return None
    except Exception:
        return None


def clear_notifications():
    """Clear the notification log"""
    if NOTIFICATION_LOG.exists():
        with open(NOTIFICATION_LOG, "w") as f:
            f.write("")


if __name__ == "__main__":
    # Output pending notifications to stdout
    notifications = get_pending_notifications()

    if notifications:
        print(notifications, end="")
        # Clear after output
        clear_notifications()
        sys.exit(0)
    else:
        sys.exit(0)
