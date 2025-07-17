#!/usr/bin/env python3
"""
Simple test for the RAG system
"""
import requests
import json
import time
from pathlib import Path

API_BASE = "http://localhost:8001"

def test_api_health():
    """Test API health endpoint"""
    print("Testing API health...")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            print("+ API health check passed")
            return True
        else:
            print(f"X API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"X API health check failed: {e}")
        return False

def test_document_upload():
    """Test document upload"""
    print("Testing document upload...")
    
    # Create a test document
    test_content = """
    This is a test document for the RAG system.
    It contains information about artificial intelligence and machine learning.
    
    Machine learning is a subset of artificial intelligence that focuses on 
    algorithms that can learn from data and make predictions or decisions.
    
    Natural language processing (NLP) is another important area of AI that 
    deals with understanding and generating human language.
    
    Vector databases are used to store and retrieve high-dimensional vectors 
    efficiently, which is crucial for semantic search applications.
    """
    
    test_file = Path("test_document.txt")
    test_file.write_text(test_content)
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file.name, f, 'text/plain')}
            response = requests.post(f"{API_BASE}/api/v1/documents", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"+ Document uploaded successfully: {result['filename']}")
            print(f"  Status: {result['status']}")
            print(f"  Size: {result['size']} bytes")
            test_file.unlink()  # Clean up
            return True
        else:
            print(f"X Document upload failed: {response.status_code}")
            print(f"  Error: {response.text}")
            test_file.unlink()  # Clean up
            return False
    except Exception as e:
        print(f"X Document upload failed: {e}")
        if test_file.exists():
            test_file.unlink()  # Clean up
        return False

def test_document_query():
    """Test document querying"""
    print("Testing document query...")
    
    # Wait a moment for document processing
    time.sleep(2)
    
    test_queries = [
        "What is machine learning?",
        "Tell me about natural language processing",
        "What are vector databases used for?"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        try:
            response = requests.post(
                f"{API_BASE}/api/v1/query",
                json={"query": query, "top_k": 3},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"+ Query successful: {result['total_results']} results")
                for i, res in enumerate(result['results']):
                    print(f"  Result {i+1}: Score {res['score']:.3f}")
                    print(f"    Content: {res['content'][:100]}...")
            else:
                print(f"X Query failed: {response.status_code}")
                print(f"  Error: {response.text}")
                return False
        except Exception as e:
            print(f"X Query failed: {e}")
            return False
    
    return True

def test_documents_list():
    """Test documents listing"""
    print("Testing documents list...")
    try:
        response = requests.get(f"{API_BASE}/api/v1/documents")
        if response.status_code == 200:
            result = response.json()
            print(f"+ Documents list retrieved: {result['total']} documents")
            return True
        else:
            print(f"X Documents list failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"X Documents list failed: {e}")
        return False

def main():
    print("Simple RAG System Test")
    print("=" * 30)
    
    # Test API health
    if not test_api_health():
        print("\nX API is not running. Please start the API first with: python simple_api.py")
        return
    
    # Test document upload
    if not test_document_upload():
        print("\nX Document upload test failed")
        return
    
    # Test document query
    if not test_document_query():
        print("\nX Document query test failed")
        return
    
    # Test documents list
    if not test_documents_list():
        print("\nX Documents list test failed")
        return
    
    print("\n+ All tests passed! The RAG system is working correctly.")
    print("\nYou can now:")
    print("1. Open http://localhost:8001/simple_frontend.html in your browser")
    print("2. Upload documents and ask questions")
    print("3. View API documentation at http://localhost:8001/docs")

if __name__ == "__main__":
    main()