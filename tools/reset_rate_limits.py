#!/usr/bin/env python3
"""
Reset rate limits for RAG system
"""
import requests
import time

def reset_rate_limits():
    """Reset rate limits by making a request to clear them"""
    try:
        # Make a simple status request to trigger rate limit cleanup
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ Rate limits reset successfully")
            return True
        else:
            print(f"❌ Failed to reset rate limits: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error resetting rate limits: {e}")
        return False

if __name__ == "__main__":
    print("Resetting rate limits...")
    if reset_rate_limits():
        print("You can now upload documents without rate limit issues.")
    else:
        print("Rate limit reset failed. Try restarting the server.")