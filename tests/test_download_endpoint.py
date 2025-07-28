#!/usr/bin/env python3
"""
Test the download endpoint functionality
"""
import requests
import json
from pathlib import Path

def test_download_endpoint():
    """Test the download endpoint"""
    base_url = "http://localhost:8000"
    
    # Check if server is running
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code != 200:
            print("❌ Server is not running")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running - start it with: python simple_api.py")
        return False
    
    # Get list of documents
    try:
        response = requests.get(f"{base_url}/api/v1/documents")
        if response.status_code != 200:
            print("❌ Could not get documents list")
            return False
        
        documents = response.json()
        if not documents:
            print("❌ No documents found. Please upload some documents first.")
            return False
        
        # Try to download the first document
        first_doc = documents[0]
        doc_id = first_doc.get('id')
        filename = first_doc.get('filename', 'unknown')
        
        print(f"📄 Attempting to download document {doc_id}: {filename}")
        
        # Test download endpoint
        download_response = requests.get(f"{base_url}/api/v1/documents/{doc_id}/download")
        
        if download_response.status_code == 200:
            print("✅ Download endpoint works!")
            print(f"   Content-Type: {download_response.headers.get('Content-Type')}")
            print(f"   Content-Length: {download_response.headers.get('Content-Length')}")
            print(f"   Content-Disposition: {download_response.headers.get('Content-Disposition')}")
            return True
        elif download_response.status_code == 404:
            print("❌ Document file not found on disk")
            return False
        elif download_response.status_code == 403:
            print("❌ Access denied")
            return False
        else:
            print(f"❌ Download failed with status {download_response.status_code}")
            print(f"   Error: {download_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing download endpoint: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing download endpoint...")
    success = test_download_endpoint()
    if success:
        print("\n✅ Download endpoint test passed!")
    else:
        print("\n❌ Download endpoint test failed!")