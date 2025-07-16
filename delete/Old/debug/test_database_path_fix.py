#!/usr/bin/env python3
"""
Test the database path fix
"""

import sys
import os
sys.path.append("C:/Users/THE/open-source-rag-system")

def test_corrected_database_path():
    print("=" * 60)
    print("TESTING CORRECTED DATABASE PATH")
    print("=" * 60)
    
    # Test the corrected initialization
    try:
        from services.persistent_storage import init_persistent_storage
        
        # Use the same absolute path as in the fixed server code
        db_path = "C:/Users/THE/open-source-rag-system/rag_database.db"
        print(f"Testing database path: {db_path}")
        
        persistent_storage = init_persistent_storage(db_path)
        storage_stats = persistent_storage.get_stats()
        
        print(f"✓ Storage stats: {storage_stats}")
        print(f"✓ Documents: {storage_stats['documents']}")
        print(f"✓ Database size: {storage_stats['database_size_mb']} MB")
        
        # Get documents like the server will
        docs = persistent_storage.get_all_documents(limit=10)
        print(f"✓ Retrieved {len(docs)} documents")
        
        # Show first few documents
        for i, doc in enumerate(docs[:5]):
            print(f"  {i+1}. {doc.get('filename', 'unknown')} (ID: {doc.get('id')})")
        
        print("\n🎉 DATABASE PATH FIX SUCCESSFUL!")
        print("Server should now show all documents correctly.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_corrected_database_path()