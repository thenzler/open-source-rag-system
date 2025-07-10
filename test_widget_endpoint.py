#!/usr/bin/env python3
"""
Test script to verify the widget endpoint works
"""
import requests
import json

def test_widget_endpoint():
    """Test the optimized endpoint that the widget uses"""
    
    # API endpoint
    url = "http://localhost:8001/api/v1/query/optimized"
    
    # Test data
    test_data = {
        "query": "Was gehört in die Bio Tonne?",
        "context_limit": 3,
        "max_tokens": 200
    }
    
    print("Testing widget endpoint...")
    print(f"URL: {url}")
    print(f"Data: {json.dumps(test_data, indent=2)}")
    print("-" * 50)
    
    try:
        # Make request
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=test_data,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.ok:
            result = response.json()
            print("\n✅ SUCCESS!")
            print(f"Response: {result.get('response', 'No response field')}")
            print(f"Query: {result.get('query', 'No query field')}")
            print(f"Context count: {len(result.get('context', []))}")
            print(f"Processing time: {result.get('processing_time', 'N/A')}s")
            
            if result.get('context'):
                print("\nSources:")
                for i, source in enumerate(result['context'][:2]):
                    print(f"  {i+1}. {source.get('source_document', 'Unknown')}")
        else:
            print(f"\n❌ ERROR!")
            print(f"Response text: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR: Cannot connect to server")
        print("Make sure the API server is running: python simple_api.py")
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT ERROR: Request took too long")
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")

if __name__ == "__main__":
    test_widget_endpoint()