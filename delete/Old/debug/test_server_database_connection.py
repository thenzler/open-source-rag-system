#!/usr/bin/env python3
"""
Test what happens when server tries to load documents from database
"""

import sys
import os
sys.path.append("C:/Users/THE/open-source-rag-system")

def test_server_database_integration():
    print("=" * 60)
    print("TESTING SERVER-DATABASE INTEGRATION")
    print("=" * 60)
    
    # Test 1: Direct database access (like our diagnostic)
    print("\n1. DIRECT DATABASE ACCESS:")
    try:
        from services.persistent_storage import PersistentStorage
        storage = PersistentStorage("C:/Users/THE/open-source-rag-system/rag_database.db")
        docs = storage.get_all_documents(limit=10)
        print(f"   Direct access: {len(docs)} documents found")
        for i, doc in enumerate(docs[:3]):
            print(f"   - {doc.get('filename', 'unknown')} (ID: {doc.get('id')})")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test 2: Server initialization path
    print("\n2. SERVER INITIALIZATION PATH:")
    try:
        # Mimic server initialization
        from services.persistent_storage import get_persistent_storage, init_persistent_storage
        persistent_storage = init_persistent_storage("rag_database.db")
        
        if persistent_storage:
            print("   Persistent storage initialized successfully")
            storage_stats = persistent_storage.get_stats()
            print(f"   Stats: {storage_stats}")
            
            # Test get_all_documents like the server does
            stored_docs = persistent_storage.get_all_documents(limit=100)
            print(f"   Server path: {len(stored_docs)} documents retrieved")
            
            # Test the exact formatting logic from the server
            formatted_docs = []
            for i, doc in enumerate(stored_docs):
                try:
                    formatted_doc = {
                        "id": doc.get("id", f"doc_{i}"),
                        "filename": doc.get("filename", "unknown"),
                        "original_filename": doc.get("original_filename", doc.get("filename", "unknown")),
                        "file_type": doc.get("file_type", "unknown"),
                        "file_size": doc.get("file_size", 0),
                        "size": doc.get("file_size", 0),
                        "content_type": doc.get("content_type", "unknown"),
                        "status": doc.get("status", "processed"),
                        "upload_date": str(doc.get("upload_timestamp", "unknown")),
                        "chunks_count": doc.get("chunk_count", 0)
                    }
                    formatted_docs.append(formatted_doc)
                except Exception as doc_error:
                    print(f"   ERROR formatting document {i}: {doc_error}")
                    continue
            
            print(f"   Successfully formatted: {len(formatted_docs)} documents")
            
            # Show first few formatted documents
            for i, doc in enumerate(formatted_docs[:3]):
                print(f"   - {doc['filename']} (ID: {doc['id']}, Size: {doc['file_size']})")
            
        else:
            print("   ERROR: persistent_storage is None")
    except Exception as e:
        print(f"   ERROR in server path: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Check working directory
    print("\n3. WORKING DIRECTORY CHECK:")
    current_dir = os.getcwd()
    print(f"   Current working directory: {current_dir}")
    
    db_path_relative = "rag_database.db"
    db_path_absolute = "C:/Users/THE/open-source-rag-system/rag_database.db"
    
    print(f"   Relative path exists: {os.path.exists(db_path_relative)}")
    print(f"   Absolute path exists: {os.path.exists(db_path_absolute)}")
    
    if os.path.exists(db_path_relative):
        print(f"   Relative DB size: {os.path.getsize(db_path_relative)} bytes")
    if os.path.exists(db_path_absolute):
        print(f"   Absolute DB size: {os.path.getsize(db_path_absolute)} bytes")

if __name__ == "__main__":
    test_server_database_integration()