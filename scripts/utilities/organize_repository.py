#!/usr/bin/env python3
"""
Repository Organization Script
Safely moves legacy files to archive directories
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
    
    # Create archive directories if they don't exist
    archive_dirs = [
        ".archive/legacy-simple-api",
        ".archive/cleanup-docs", 
        ".archive/old-services",
        ".archive/old-config",
        ".archive/old-storage"
    ]
    
    for dir_path in archive_dirs:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created: {dir_path}")
    
    # Phase 1: Archive legacy Simple API files
    legacy_simple_api_files = [
        # Root level legacy files
        "simple_api.py",
        "simple_frontend.html", 
        "start_simple_rag.py",
        "simple_requirements.txt",
        "ollama_client.py",
        # Inside core/ but legacy
        "core/simple_api.py",
        "core/simple_frontend.html",
        "core/start_simple_rag.py"
    ]
    
    print("\nPhase 1: Archiving Legacy Simple API Files")
    print("-" * 50)
    
    for file_path in legacy_simple_api_files:
        source = base_dir / file_path
        if source.exists():
            # Preserve directory structure in archive
            relative_path = Path(file_path)
            if relative_path.parts[0] == "core":
                # Remove 'core' prefix for core files
                dest_name = "_".join(relative_path.parts[1:])
            else:
                dest_name = relative_path.name
            
            dest = base_dir / ".archive/legacy-simple-api" / dest_name
            
            try:
                shutil.move(str(source), str(dest))
                print(f"[OK] Moved: {file_path} -> .archive/legacy-simple-api/{dest_name}")
            except Exception as e:
                print(f"[ERROR] Failed to move {file_path}: {e}")
        else:
            print(f"[SKIP] Not found: {file_path}")
    
    # Phase 2: Archive cleanup documentation
    cleanup_docs = [
        "CLEANUP_PLAN.md",
        "CLEANUP_SUMMARY.md", 
        "CODE_FIXES_SUMMARY.md",
        "ZERO_HALLUCINATION_PLAN.md",
        "cleanup_script.py"
    ]
    
    print("\nPhase 2: Archiving Cleanup Documentation")
    print("-" * 50)
    
    for file_path in cleanup_docs:
        source = base_dir / file_path
        if source.exists():
            dest = base_dir / ".archive/cleanup-docs" / file_path
            try:
                shutil.move(str(source), str(dest))
                print(f"[OK] Moved: {file_path} -> .archive/cleanup-docs/")
            except Exception as e:
                print(f"[ERROR] Failed to move {file_path}: {e}")
        else:
            print(f"[SKIP] Not found: {file_path}")
    
    # Phase 3: Archive old services
    old_services = [
        "services/",  # Entire old services directory
        "api/simple_rag_api.py",
        "api/confidence_endpoints.py"
    ]
    
    print("\n🔧 Phase 3: Archiving Old Services")
    print("-" * 50)
    
    for item_path in old_services:
        source = base_dir / item_path
        if source.exists():
            if source.is_dir():
                dest = base_dir / ".archive/old-services" / source.name
                try:
                    shutil.move(str(source), str(dest))
                    print(f"✅ Moved directory: {item_path} → .archive/old-services/")
                except Exception as e:
                    print(f"❌ Failed to move {item_path}: {e}")
            else:
                dest = base_dir / ".archive/old-services" / source.name
                try:
                    shutil.move(str(source), str(dest))
                    print(f"✅ Moved: {item_path} → .archive/old-services/")
                except Exception as e:
                    print(f"❌ Failed to move {item_path}: {e}")
        else:
            print(f"⚠️ Not found: {item_path}")
    
    # Phase 4: Archive old config
    old_config = [
        "config/config.py",
        "config/database_config.py", 
        "config/confidence_config.yaml"
    ]
    
    print("\n⚙️ Phase 4: Archiving Old Configuration")
    print("-" * 50)
    
    for file_path in old_config:
        source = base_dir / file_path
        if source.exists():
            dest = base_dir / ".archive/old-config" / source.name
            try:
                shutil.move(str(source), str(dest))
                print(f"✅ Moved: {file_path} → .archive/old-config/")
            except Exception as e:
                print(f"❌ Failed to move {file_path}: {e}")
        else:
            print(f"⚠️ Not found: {file_path}")
    
    # Phase 5: Archive old storage
    old_storage = [
        "storage/",  # Root level storage directory (old)
        "database/"  # Old database directory
    ]
    
    print("\n💾 Phase 5: Archiving Old Storage")
    print("-" * 50)
    
    for item_path in old_storage:
        source = base_dir / item_path
        if source.exists():
            dest = base_dir / ".archive/old-storage" / source.name
            try:
                if source.is_dir():
                    shutil.move(str(source), str(dest))
                    print(f"✅ Moved directory: {item_path} → .archive/old-storage/")
                else:
                    shutil.move(str(source), str(dest))
                    print(f"✅ Moved: {item_path} → .archive/old-storage/")
            except Exception as e:
                print(f"❌ Failed to move {item_path}: {e}")
        else:
            print(f"⚠️ Not found: {item_path}")
    
    print("\n✨ Repository Organization Complete!")
    print("=" * 50)
    print("📁 Files have been safely moved to .archive/ directories")
    print("🔍 You can check the archive directories to verify all files were moved correctly")
    print("🧪 Test the system with: python run_core.py")
    print("⚠️ If anything breaks, all files can be restored from .archive/")

if __name__ == "__main__":
    main()