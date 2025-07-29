#!/usr/bin/env python3
"""
Simple Repository Organization Script - ASCII only
"""
import os
import shutil
from pathlib import Path

def main():
    print("Repository Organization Script")
    print("=" * 50)
    
    # Get base directory
    base_dir = Path(__file__).parent
    print(f"Working directory: {base_dir}")
    
    # Phase 1: Archive legacy Simple API files (root and core level)
    legacy_files = [
        "simple_api.py",
        "start_simple_rag.py", 
        "simple_requirements.txt",
        "core/simple_api.py",
        "core/simple_frontend.html",
        "core/start_simple_rag.py"
    ]
    
    print("\nPhase 1: Archiving Legacy Simple API Files")
    print("-" * 50)
    
    # Create archive directory
    archive_dir = base_dir / ".archive" / "legacy-simple-api"
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in legacy_files:
        source = base_dir / file_path
        if source.exists():
            # Create a safe filename for archive (replace / with _)
            safe_name = file_path.replace("/", "_").replace("\\", "_")
            dest = archive_dir / safe_name
            try:
                shutil.move(str(source), str(dest))
                print(f"[OK] Moved: {file_path} -> {safe_name}")
            except Exception as e:
                print(f"[ERROR] Failed to move {file_path}: {e}")
        else:
            print(f"[SKIP] Not found: {file_path}")
    
    # Phase 2: Archive cleanup documentation
    cleanup_docs = [
        "CLEANUP_PLAN.md",
        "CLEANUP_SUMMARY.md", 
        "CODE_FIXES_SUMMARY.md",
        "ZERO_HALLUCINATION_PLAN.md"
    ]
    
    print("\nPhase 2: Archiving Cleanup Documentation")
    print("-" * 50)
    
    cleanup_dir = base_dir / ".archive" / "cleanup-docs"
    cleanup_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in cleanup_docs:
        source = base_dir / file_path
        if source.exists():
            dest = cleanup_dir / file_path
            try:
                shutil.move(str(source), str(dest))
                print(f"[OK] Moved: {file_path}")
            except Exception as e:
                print(f"[ERROR] Failed to move {file_path}: {e}")
        else:
            print(f"[SKIP] Not found: {file_path}")
    
    print("\nRepository Organization Complete!")
    print("Files moved to .archive/ directories")
    print("Test system with: python run_core.py")

if __name__ == "__main__":
    main()