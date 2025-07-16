#!/usr/bin/env python3
"""
Debug server LLM issue
"""

import sys
import os
import requests
import json

def test_server_llm():
    print("=== SERVER LLM DEBUG ===")
    
    # Test LLM status first
    try:
        response = requests.get("http://localhost:8001/api/v1/llm/status")
        if response.status_code == 200:
            data = response.json()
            print(f"LLM Status: {data}")
        else:
            print(f"Failed to get LLM status: {response.status_code}")
            return
    except Exception as e:
        print(f"Error getting LLM status: {e}")
        return
    
    # Test simple query first
    try:
        print("\nTesting simple vector query...")
        response = requests.post(
            "http://localhost:8001/api/v1/query",
            headers={"Content-Type": "application/json"},
            json={"query": "test", "use_llm": False}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Vector query results: {len(data.get('results', []))} results")
        else:
            print(f"Vector query failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error in vector query: {e}")
    
    # Test enhanced query with explicit LLM=True
    try:
        print("\nTesting enhanced query with LLM=True...")
        response = requests.post(
            "http://localhost:8001/api/v1/query/enhanced",
            headers={"Content-Type": "application/json"},
            json={"query": "test", "use_llm": True}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Enhanced query method: {data.get('method')}")
            print(f"Enhanced query answer length: {len(data.get('answer', ''))}")
            print(f"Enhanced query sources: {len(data.get('sources', []))}")
            
            # Show first 200 chars of answer
            answer = data.get('answer', '')
            if answer:
                print(f"Answer preview: {answer[:200]}...")
        else:
            print(f"Enhanced query failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error in enhanced query: {e}")

if __name__ == "__main__":
    test_server_llm()