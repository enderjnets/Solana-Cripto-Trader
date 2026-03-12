#!/usr/bin/env python3
"""
🔄 YouTube OAuth Auto-Refresh System
Detecta cuando el token está por expirar y lo renueva automáticamente
"""
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

CREDENTIALS_FILE = Path("/home/enderj/.openclaw/workspace/memory/youtube_credentials.json")
TOKEN_EXPIRY_FILE = Path("/home/enderj/.openclaw/workspace/memory/youtube_token_expiry.json")

def check_token_status() -> dict:
    """Check if YouTube token needs refresh"""
    
    if not CREDENTIALS_FILE.exists():
        return {
            "status": "MISSING",
            "message": "YouTube credentials not found",
            "action_required": "Run regenerate_youtube_oauth_manual.py"
        }
    
    try:
        creds = Credentials.from_authorized_user_file(str(CREDENTIALS_FILE))
        
        if not creds.valid:
            if creds.expired and creds.refresh_token:
                return {
                    "status": "EXPIRED",
                    "message": "Token expired, attempting auto-refresh",
                    "action": "refresh"
                }
            else:
                return {
                    "status": "INVALID",
                    "message": "Token invalid and cannot be refreshed",
                    "action_required": "Run regenerate_youtube_oauth_manual.py"
                }
        
        # Check if token expires soon (within 7 days)
        if creds.expiry:
            time_until_expiry = creds.expiry - datetime.now(timezone.utc)
            
            if time_until_expiry < timedelta(days=7):
                return {
                    "status": "EXPIRING_SOON",
                    "message": f"Token expires in {time_until_expiry.days} days",
                    "action": "refresh",
                    "expiry": creds.expiry.isoformat()
                }
        
        return {
            "status": "VALID",
            "message": "Token is valid",
            "expiry": creds.expiry.isoformat() if creds.expiry else "Unknown"
        }
        
    except Exception as e:
        return {
            "status": "ERROR",
            "message": f"Error checking token: {str(e)}",
            "action_required": "Run regenerate_youtube_oauth_manual.py"
        }


def auto_refresh_token() -> dict:
    """Automatically refresh the token if possible"""
    
    status = check_token_status()
    
    if status["status"] == "VALID":
        return {
            "success": True,
            "message": "Token is already valid",
            "status": status
        }
    
    if status["status"] not in ["EXPIRED", "EXPIRING_SOON"]:
        return {
            "success": False,
            "message": "Cannot auto-refresh token",
            "status": status
        }
    
    try:
        creds = Credentials.from_authorized_user_file(str(CREDENTIALS_FILE))
        
        # Refresh the token
        creds.refresh(Request())
        
        # Save the refreshed credentials
        creds_data = {
            'token': creds.token,
            'refresh_token': creds.refresh_token,
            'token_uri': creds.token_uri,
            'client_id': creds.client_id,
            'client_secret': creds.client_secret,
            'scopes': creds.scopes,
            'token_type': 'Bearer'
        }
        
        CREDENTIALS_FILE.write_text(json.dumps(creds_data, indent=2))
        
        return {
            "success": True,
            "message": "Token refreshed successfully",
            "new_expiry": creds.expiry.isoformat() if creds.expiry else "Unknown"
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to refresh token: {str(e)}",
            "action_required": "Run regenerate_youtube_oauth_manual.py"
        }


def monitor_oauth_health():
    """Monitor YouTube OAuth health and auto-refresh if needed"""
    
    print("=" * 80)
    print("🔄 YOUTUBE OAUTH AUTO-REFRESH MONITOR")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check status
    status = check_token_status()
    
    print(f"Status: {status['status']}")
    print(f"Message: {status['message']}")
    
    if 'expiry' in status:
        print(f"Expires: {status['expiry']}")
    
    print()
    
    # Auto-refresh if needed
    if status.get('action') == 'refresh':
        print("🔄 Attempting auto-refresh...")
        result = auto_refresh_token()
        
        if result['success']:
            print(f"✅ {result['message']}")
            if 'new_expiry' in result:
                print(f"New expiry: {result['new_expiry']}")
        else:
            print(f"❌ {result['message']}")
            if 'action_required' in result:
                print(f"Action required: {result['action_required']}")
    elif 'action_required' in status:
        print(f"⚠️ Action required: {status['action_required']}")
    else:
        print("✅ No action needed")
    
    print()
    print("=" * 80)
    
    return status


if __name__ == "__main__":
    monitor_oauth_health()
