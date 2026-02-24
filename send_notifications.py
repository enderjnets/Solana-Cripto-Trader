#!/usr/bin/env python3
"""
Send Pending Notifications
Reads the notification log and sends pending notifications via OpenClaw message system
"""
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

NOTIFICATION_LOG = PROJECT_ROOT / "data" / "notifications.log"
SENT_LOG = PROJECT_ROOT / "data" / "notifications_sent.log"


def send_telegram_via_openclaw(message: str) -> bool:
    """Send message to Telegram using OpenClaw's message tool"""
    try:
        # Use message tool via the tool system (if available)
        # For now, write to a file that the main session can read
        pending_file = PROJECT_ROOT / "data" / "pending_telegram.txt"
        pending_file.parent.mkdir(exist_ok=True)

        with open(pending_file, "a", encoding="utf-8") as f:
            f.write(f"{message}\n---END---\n")

        print(f"✅ Queued message: {message[:50]}...")
        return True

    except Exception as e:
        print(f"❌ Failed to queue message: {e}")
        return False


def process_notifications():
    """Read and send pending notifications"""
    if not NOTIFICATION_LOG.exists():
        print("No notification log found")
        return 0

    # Read all notifications
    with open(NOTIFICATION_LOG, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        print("No pending notifications")
        return 0

    sent_count = 0
    current_message = ""
    current_priority = "normal"

    for line in lines:
        if line.startswith("[") and "]" in line:
            # New notification
            # Send previous message if exists
            if current_message:
                if send_telegram_via_openclaw(current_message):
                    sent_count += 1
                # Log as sent
                with open(SENT_LOG, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now()}] {current_priority.upper()}: {current_message}\n")

            # Parse new notification
            try:
                parts = line.split("]")
                timestamp = parts[0][1:].strip()
                priority = parts[1][1:].strip() if len(parts) > 1 else "normal"
                current_priority = priority
                current_message = line[len(parts[0]) + len(parts[1]) + 3:].strip() if len(parts) > 1 else ""
            except Exception as e:
                print(f"Failed to parse line: {line}")
                current_message = ""
        else:
            # Continuation of current message
            current_message += line

    # Send last message
    if current_message:
        if send_telegram_via_openclaw(current_message):
            sent_count += 1

    # Clear notification log after sending
    if sent_count > 0:
        with open(NOTIFICATION_LOG, "w") as f:
            f.write("")
        print(f"✅ Sent {sent_count} notifications, log cleared")

    return sent_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Send pending notifications")
    parser.add_argument("--test", action="store_true", help="Test sending without clearing log")
    parser.add_argument("--read", action="store_true", help="Just read and display pending notifications")

    args = parser.parse_args()

    if args.read:
        if NOTIFICATION_LOG.exists():
            with open(NOTIFICATION_LOG, "r") as f:
                print(f.read())
        else:
            print("No pending notifications")
    elif args.test:
        # Just display what would be sent
        if NOTIFICATION_LOG.exists():
            with open(NOTIFICATION_LOG, "r") as f:
                print(f.read())
        print(f"\nWould send to Telegram")
    else:
        count = process_notifications()
        print(f"Processed {count} notifications")
