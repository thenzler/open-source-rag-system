#!/usr/bin/env python3
"""
Test script to debug file upload issues
"""
import requests
import os
import time
from pathlib import Path

def test_upload(file_path: str, server_url: str = "http://localhost:8001"):
    """Test file upload and provide detailed error information"""
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
        
    file_size = os.path.getsize(file_path)
    print(f"📁 File: {file_path}")
    print(f"📏 Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    print(f"📝 Extension: {Path(file_path).suffix}")
    
    try:
        # Test server health first
        health_response = requests.get(f"{server_url}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"❌ Server health check failed: {health_response.status_code}")
            return False
        print("✅ Server is healthy")
        
        # Upload file
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
            print(f"⬆️  Uploading file...")
            
            response = requests.post(
                f"{server_url}/api/v1/documents", 
                files=files,
                timeout=30
            )
            
            if response.status_code == 200:
                print("✅ Upload successful!")
                result = response.json()
                print(f"📄 Document ID: {result.get('id')}")
                print(f"📝 Filename: {result.get('filename')}")
                print(f"📊 Chunks: {result.get('chunks_created')}")
                return True
            else:
                print(f"❌ Upload failed: {response.status_code}")
                print(f"📛 Error: {response.text}")
                return False
                
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure it's running at http://localhost:8001")
        return False
    except requests.exceptions.Timeout:
        print("❌ Upload timed out")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 File Upload Test")
    print("=" * 50)
    
    # Test with common file types
    test_files = [
        "C:\\Users\\THE\\Desktop\\test.pdf",
        "C:\\Users\\THE\\Desktop\\test.docx",
        "C:\\Users\\THE\\Desktop\\test.txt",
        "Transferarbeit.docx"  # The specific file mentioned
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n🔍 Testing: {test_file}")
            test_upload(test_file)
            print("-" * 50)
        else:
            print(f"⏭️  Skipping {test_file} (not found)")
    
    print("\n📝 Manual test:")
    print("1. Make sure server is running: python quick_start.py")
    print("2. Try uploading through web interface: http://localhost:8001")
    print("3. Check server logs for detailed error messages")

if __name__ == "__main__":
    main()