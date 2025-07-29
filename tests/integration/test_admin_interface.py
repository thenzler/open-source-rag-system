#!/usr/bin/env python3
"""
Test script for admin interface functionality
"""
import requests
import json

def test_admin_interface():
    """Test the admin interface endpoints"""
    base_url = "http://localhost:8000"
    
    print("[TESTING] Admin Interface")
    print("=" * 50)
    
    try:
        # Test 1: Check if admin dashboard loads
        print("\n1. Testing admin dashboard...")
        response = requests.get(f"{base_url}/admin/", timeout=10)
        if response.status_code == 200:
            print("[SUCCESS] Admin dashboard loads successfully")
        else:
            print(f"[ERROR] Admin dashboard failed: {response.status_code}")
        
        # Test 2: Get available models
        print("\n2. Testing models API...")
        response = requests.get(f"{base_url}/admin/models", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            print(f"[SUCCESS] Models API working - Found {models_data['total_count']} models")
            print(f"   Current model: {models_data['current_model']}")
            for model_key, model_info in models_data['models'].items():
                status = model_info.get('status', 'unknown')
                print(f"   - {model_key}: {status}")
        else:
            print(f"[ERROR] Models API failed: {response.status_code}")
        
        # Test 3: Get system stats
        print("\n3. Testing system stats...")
        response = requests.get(f"{base_url}/admin/system/stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print(f"[SUCCESS] System stats working")
            print(f"   Status: {stats.get('system_status')}")
            print(f"   Available models: {len(stats.get('available_models', []))}")
        else:
            print(f"[ERROR] System stats failed: {response.status_code}")
        
        # Test 4: Check main UI has settings button
        print("\n4. Testing main UI settings button...")
        response = requests.get(f"{base_url}/ui", timeout=10)
        if response.status_code == 200 and 'Settings' in response.text:
            print("[SUCCESS] Settings button found in main UI")
        else:
            print("[ERROR] Settings button not found in main UI")
        
        print(f"\n[COMPLETED] Admin interface testing completed!")
        print(f"[INFO] Access admin at: {base_url}/admin")
        print(f"[INFO] Main UI at: {base_url}/ui")
        
    except requests.exceptions.ConnectionError:
        print("[ERROR] Server not running on localhost:8000")
        print("Start server with: python -m uvicorn core.main:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")

if __name__ == "__main__":
    test_admin_interface()