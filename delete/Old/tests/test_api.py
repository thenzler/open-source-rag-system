#!/usr/bin/env python3
"""
Test script to check if the API server is working
"""
import requests
import time

def test_api():
    base_url = "http://127.0.0.1:8001"
    
    print("Testing API server...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"Root endpoint: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Root endpoint failed: {e}")
    
    # Test documents endpoint
    try:
        response = requests.get(f"{base_url}/api/v1/documents", timeout=5)
        print(f"Documents endpoint: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Documents endpoint failed: {e}")

if __name__ == "__main__":
    test_api()