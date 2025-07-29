#!/usr/bin/env python3
"""
Quick test of qwen2.5:7b model performance
"""
import requests
import time
import json

def test_qwen_performance():
    """Test the new qwen2.5 model performance"""
    print("[TESTING] qwen2.5:7b Model Performance")
    print("=" * 50)
    
    # API endpoint
    api_url = "http://localhost:8000/api/v1/query"
    
    # Test query in German
    test_query = "Welche Regeln gelten für Gewerbeabfall?"
    
    print(f"Testing query: {test_query}")
    print("Starting performance test...")
    
    # Performance test
    print(f"\n--- Performance Test: qwen2.5:7b model ---")
    start_time = time.time()
    
    try:
        response = requests.post(api_url, 
                               json={"query": test_query},
                               timeout=120)  # 2 minute timeout
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('answer', '')
            sources = result.get('sources', [])
            confidence = result.get('confidence', 0)
            
            print(f"[SUCCESS] Response time: {response_time:.2f} seconds")
            print(f"   Answer length: {len(answer)} characters")
            print(f"   Sources found: {len(sources)}")
            print(f"   Confidence: {confidence:.3f}")
            
            # Performance comparison
            if response_time < 30:
                print("[EXCELLENT] Under 30 seconds (target achieved!)")
            elif response_time < 45:
                print("[GOOD] Under 45 seconds (good performance)")
            elif response_time < 60:
                print("[OK] Under 60 seconds (acceptable)")
            else:
                print("[WARNING] SLOW: Over 60 seconds (needs optimization)")
            
            # Show first part of answer
            print(f"\n[ANSWER PREVIEW]:")
            print(f"   {answer[:200]}...")
            
            return True
            
        else:
            print(f"[ERROR] API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("[ERROR] Request timed out - model may be too slow")
        return False
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False

def check_system_status():
    """Check if system is ready"""
    try:
        status_response = requests.get("http://localhost:8000/api/v1/status", timeout=5)
        if status_response.status_code == 200:
            print("[OK] System is ready")
            return True
        else:
            print(f"[WARNING] System status: {status_response.status_code}")
            return False
    except Exception:
        print("ERROR: System not ready - make sure server is running on port 8000")
        return False

if __name__ == "__main__":
    print("Checking system status...")
    
    if check_system_status():
        print("\nRunning performance test with qwen2.5:7b...")
        success = test_qwen_performance()
        
        if success:
            print(f"\n[COMPLETED] qwen2.5:7b model test completed successfully!")
            print(f"[INFO] Expected improvement over arlesheim-german: 50-70% faster")
        else:
            print(f"\n[ERROR] Test failed - check server logs")
    else:
        print("\n[ERROR] Cannot run test - server not ready")
        print("Start server with: python -m uvicorn core.main:app --host 0.0.0.0 --port 8000")