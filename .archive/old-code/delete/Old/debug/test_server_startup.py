#!/usr/bin/env python3
"""
Test server startup components individually to find the hanging issue
"""

import sys
import time

def test_import(module_name, description):
    """Test importing a module and time how long it takes"""
    print(f"Testing {description}...")
    start_time = time.time()
    try:
        if module_name == "fastapi_app":
            # Import FastAPI components
            from fastapi import FastAPI
            from fastapi.middleware.cors import CORSMiddleware
            print(f"  ✓ FastAPI imported successfully in {time.time() - start_time:.2f}s")
        elif module_name == "persistent_storage":
            sys.path.append("C:/Users/THE/open-source-rag-system")
            from services.persistent_storage import get_persistent_storage, init_persistent_storage
            storage = init_persistent_storage("rag_database.db")
            print(f"  ✓ Persistent storage initialized in {time.time() - start_time:.2f}s")
        elif module_name == "memory_safe_storage":
            from services.memory_safe_storage import get_memory_safe_storage
            storage = get_memory_safe_storage()
            print(f"  ✓ Memory-safe storage initialized in {time.time() - start_time:.2f}s")
        elif module_name == "llm_manager":
            from services.llm_manager import LLMManager
            manager = LLMManager()
            print(f"  ✓ LLM Manager initialized in {time.time() - start_time:.2f}s")
        elif module_name == "sentence_transformers":
            from sentence_transformers import SentenceTransformer
            print(f"  ✓ SentenceTransformers imported in {time.time() - start_time:.2f}s")
        else:
            exec(f"import {module_name}")
            print(f"  ✓ {module_name} imported successfully in {time.time() - start_time:.2f}s")
        
        return True
    except Exception as e:
        print(f"  ✗ Error importing {module_name}: {e}")
        return False

def main():
    print("Testing server startup components...\n")
    
    # Test basic imports first
    components = [
        ("fastapi_app", "FastAPI framework"),
        ("persistent_storage", "Persistent Storage"),
        ("memory_safe_storage", "Memory-safe Storage"),
        ("sentence_transformers", "SentenceTransformers"),
        ("llm_manager", "LLM Manager"),
    ]
    
    for module, desc in components:
        success = test_import(module, desc)
        if not success:
            print(f"\n❌ Found issue with {desc}!")
            return
        print()
    
    print("✅ All components imported successfully!")
    print("The hanging issue might be in the FastAPI app initialization or server startup.")

if __name__ == "__main__":
    main()