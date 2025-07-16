#!/usr/bin/env python3
"""
Database diagnostic script to identify SQLite issues
"""

import sqlite3
import os
import sys
from pathlib import Path

def check_database_status():
    db_path = "C:/Users/THE/open-source-rag-system/rag_database.db"
    
    print("=" * 50)
    print("DATABASE DIAGNOSTIC REPORT")
    print("=" * 50)
    
    # Check if database file exists
    if not os.path.exists(db_path):
        print("ERROR: Database file does not exist!")
        return False
    
    print(f"OK: Database file exists: {db_path}")
    print(f"File size: {os.path.getsize(db_path)} bytes")
    
    # Check WAL files
    shm_path = db_path + "-shm"
    wal_path = db_path + "-wal"
    
    if os.path.exists(shm_path):
        print(f"WAL SHM file exists: {os.path.getsize(shm_path)} bytes")
    if os.path.exists(wal_path):
        print(f"WAL file exists: {os.path.getsize(wal_path)} bytes")
    
    # Try to connect and query
    try:
        print("\nTesting database connection...")
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check database integrity
        print("Checking database integrity...")
        cursor.execute("PRAGMA integrity_check")
        integrity = cursor.fetchone()[0]
        print(f"   Integrity: {integrity}")
        
        # Check if tables exist
        print("Checking tables...")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"   Tables: {tables}")
        
        # Count documents
        if 'documents' in tables:
            print("Counting documents...")
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            print(f"   Documents count: {doc_count}")
            
            # Try to fetch a few documents
            print("Testing document retrieval...")
            cursor.execute("SELECT id, filename, upload_date FROM documents LIMIT 5")
            docs = cursor.fetchall()
            for doc in docs:
                print(f"   - ID: {doc[0]}, File: {doc[1]}, Date: {doc[2]}")
        
        conn.close()
        print("OK: Database connection test PASSED")
        return True
        
    except sqlite3.OperationalError as e:
        print(f"ERROR: SQLite Error: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Connection Error: {e}")
        return False

def test_persistent_storage():
    print("\n" + "=" * 50)
    print("PERSISTENT STORAGE TEST")
    print("=" * 50)
    
    try:
        # Add the services path
        sys.path.append("C:/Users/THE/open-source-rag-system")
        from services.persistent_storage import PersistentStorage
        
        print("Creating PersistentStorage instance...")
        storage = PersistentStorage("C:/Users/THE/open-source-rag-system/rag_database.db")
        
        print("Getting storage stats...")
        stats = storage.get_stats()
        print(f"   Stats: {stats}")
        
        print("Testing get_all_documents with limit=5...")
        docs = storage.get_all_documents(limit=5)
        print(f"   Retrieved {len(docs)} documents")
        
        for i, doc in enumerate(docs):
            print(f"   Document {i+1}: {doc.get('filename', 'unknown')}")
        
        print("OK: Persistent storage test PASSED")
        return True
        
    except Exception as e:
        print(f"ERROR: Persistent storage error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting database diagnostics...\n")
    
    db_ok = check_database_status()
    storage_ok = test_persistent_storage()
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Database File: {'OK' if db_ok else 'FAILED'}")
    print(f"Persistent Storage: {'OK' if storage_ok else 'FAILED'}")
    
    if db_ok and storage_ok:
        print("\nAll tests passed! Database should be working.")
    else:
        print("\nIssues detected. Database may need repair or recreation.")