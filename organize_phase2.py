#!/usr/bin/env python3
"""
Repository Organization Phase 2 - Move old services and config
"""
import os
import shutil
from pathlib import Path

def main():
    print("Repository Organization Phase 2")
    print("=" * 50)
    
    base_dir = Path(__file__).parent
    print(f"Working directory: {base_dir}")
    
    # Phase 1: Archive old services directory and API files
    print("\nPhase 1: Archiving Old Services")
    print("-" * 50)
    
    old_services_dir = base_dir / ".archive" / "old-services"
    old_services_dir.mkdir(parents=True, exist_ok=True)
    
    # Move entire services/ directory if it exists
    services_dir = base_dir / "services"
    if services_dir.exists():
        dest = old_services_dir / "services"
        try:
            shutil.move(str(services_dir), str(dest))
            print(f"[OK] Moved services/ directory")
        except Exception as e:
            print(f"[ERROR] Failed to move services/: {e}")
    else:
        print("[SKIP] services/ directory not found")
    
    # Move API files
    api_files = [
        "api/simple_rag_api.py",
        "api/confidence_endpoints.py"
    ]
    
    for file_path in api_files:
        source = base_dir / file_path
        if source.exists():
            safe_name = file_path.replace("/", "_")
            dest = old_services_dir / safe_name
            try:
                shutil.move(str(source), str(dest))
                print(f"[OK] Moved: {file_path}")
            except Exception as e:
                print(f"[ERROR] Failed to move {file_path}: {e}")
        else:
            print(f"[SKIP] Not found: {file_path}")
    
    # Phase 2: Archive old configuration
    print("\nPhase 2: Archiving Old Configuration")
    print("-" * 50)
    
    old_config_dir = base_dir / ".archive" / "old-config"
    old_config_dir.mkdir(parents=True, exist_ok=True)
    
    old_config_files = [
        "config/config.py",
        "config/database_config.py",
        "config/confidence_config.yaml"
    ]
    
    for file_path in old_config_files:
        source = base_dir / file_path
        if source.exists():
            filename = Path(file_path).name
            dest = old_config_dir / filename
            try:
                shutil.move(str(source), str(dest))
                print(f"[OK] Moved: {file_path}")
            except Exception as e:
                print(f"[ERROR] Failed to move {file_path}: {e}")
        else:
            print(f"[SKIP] Not found: {file_path}")
    
    # Phase 3: Archive old storage directory
    print("\nPhase 3: Archiving Old Storage")
    print("-" * 50)
    
    old_storage_dir = base_dir / ".archive" / "old-storage"  
    old_storage_dir.mkdir(parents=True, exist_ok=True)
    
    # Move root-level storage/ directory (old system)
    storage_dir = base_dir / "storage"
    if storage_dir.exists():
        dest = old_storage_dir / "storage"
        try:
            shutil.move(str(storage_dir), str(dest))
            print(f"[OK] Moved storage/ directory")
        except Exception as e:
            print(f"[ERROR] Failed to move storage/: {e}")
    else:
        print("[SKIP] storage/ directory not found")
    
    # Move database/ directory if it exists
    database_dir = base_dir / "database"
    if database_dir.exists():
        dest = old_storage_dir / "database"
        try:
            shutil.move(str(database_dir), str(dest))
            print(f"[OK] Moved database/ directory")
        except Exception as e:
            print(f"[ERROR] Failed to move database/: {e}")
    else:
        print("[SKIP] database/ directory not found")
    
    print("\nPhase 2 Organization Complete!")
    print("All old services and config archived")

if __name__ == "__main__":
    main()