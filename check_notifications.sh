#!/bin/bash
# Check and send pending notifications to Telegram
# This script is called by OpenClaw heartbeat

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if there are pending notifications
NOTIFICATIONS_FILE="data/notifications.log"

if [ ! -f "$NOTIFICATIONS_FILE" ] || [ ! -s "$NOTIFICATIONS_FILE" ]; then
    exit 0
fi

# Read and send notifications
while IFS= read -r line; do
    # Parse the notification
    if [[ $line == \[*\]\ \[*\]* ]]; then
        # Extract priority and message
        PRIORITY=$(echo "$line" | sed -E 's/^\[.*\] \[(.*)\].*/\1/')
        MESSAGE=$(echo "$line" | sed -E 's/^\[.*\] \[.*\] //')

        # Send via OpenClaw message system
        # (This will be handled by the OpenClaw session)
        echo "$MESSAGE"
    fi
done < "$NOTIFICATIONS_FILE"

# Clear the log after sending
> "$NOTIFICATIONS_FILE"

echo "Sent pending notifications"
