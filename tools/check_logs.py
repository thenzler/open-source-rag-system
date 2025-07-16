#!/usr/bin/env python3
"""Check what the server logs are showing"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=== Checking Current Storage State ===")

# Check memory-safe storage directly
try:
    from services.memory_safe_storage import get_memory_safe_storage
    
    storage = get_memory_safe_storage()
    stats = storage.get_stats()
    
    print(f"Memory-safe storage stats:")
    print(f"  Documents: {stats['documents']}")
    print(f"  Chunks: {stats['chunks']}")
    print(f"  Embeddings: {stats['embeddings']}")
    print(f"  Storage mode: {stats['storage_mode']}")
    
    if stats['documents'] > 0:
        print("\n=== Documents in storage ===")
        docs = storage.get_all_documents()
        for doc in docs:
            print(f"  - ID: {doc['id']}, Name: {doc['filename']}, Chunks: {doc['chunk_count']}")
    else:
        print("\n[PROBLEM] No documents found in memory-safe storage!")
        print("This means uploads are not reaching memory-safe storage.")
        
except Exception as e:
    print(f"[ERROR] Failed to check memory-safe storage: {e}")
    import traceback
    traceback.print_exc()

# Check if there are any files in the upload directory
print("\n=== Checking Upload Directory ===")
try:
    upload_dir = "storage/uploads"
    if os.path.exists(upload_dir):
        files = os.listdir(upload_dir)
        print(f"Files in {upload_dir}: {len(files)} files")
        for file in files[:5]:  # Show first 5 files
            file_path = os.path.join(upload_dir, file)
            size = os.path.getsize(file_path)
            print(f"  - {file} ({size} bytes)")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more files")
    else:
        print(f"Upload directory {upload_dir} does not exist")
except Exception as e:
    print(f"[ERROR] Failed to check upload directory: {e}")

print("\n=== Diagnosis ===")
print("If you see:")
print("- Files in upload directory BUT 0 documents in memory-safe storage")
print("  -> Upload process is failing to store in memory-safe storage")
print("- No files in upload directory")
print("  -> Upload process is failing completely")
print("- Documents in memory-safe storage but frontend shows 0")
print("  -> Status endpoint problem (should be fixed now)")

print("\nNext step: Check the server logs when you upload a file to see the DEBUG messages")