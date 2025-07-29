#!/usr/bin/env python3
"""
Simple Repository Organization Script
Focuses on the most critical organization tasks
"""

import os
import shutil
from pathlib import Path

def create_dirs(dirs):
    """Create directories"""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"Created: {d}")

def move_file(src, dst):
    """Move file if exists"""
    if Path(src).exists():
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dst)
        print(f"Moved: {src} -> {dst}")
    return Path(src).exists()

def main():
    print("Starting simple repository organization...\n")
    
    # 1. Create essential directories
    print("1. Creating directory structure...")
    create_dirs([
        # Scripts organization
        "scripts/setup",
        "scripts/debug", 
        "scripts/deployment",
        "scripts/maintenance",
        
        # Documentation organization
        "docs/guides",
        "docs/admin",
        "docs/architecture", 
        "docs/legacy",
        
        # Test organization
        "tests/unit",
        "tests/integration",
        "tests/fixtures",
        
        # Tools organization
        "tools/training",
        "tools/utilities",
        "tools/legacy_municipal",
        
        # Deployment
        "deployment/configs",
        "deployment/requirements",
        
        # Examples
        "examples/demos",
    ])
    
    # 2. Move Python test files
    print("\n2. Organizing test files...")
    test_files = [
        "test_admin_fixed.py",
        "test_admin_import.py", 
        "test_admin_interface.py",
        "test_cleanup_api.py",
        "test_cleaned_rag.py",
        "test_confidence_system.py",
        "test_configurable_confidence.py",
        "test_fix.py",
        "test_fix_simple.py",
        "test_fixes_directly.py",
        "test_llm_integration.py",
        "test_ollama_direct.py",
        "test_ollama_integration.py",
        "test_performance.py",
        "test_processing.py",
        "test_qwen_performance.py",
        "test_rag_fixes.py",
        "test_simple_rag.py",
        "test_single_endpoint.py",
        "test_zero_hallucination.py",
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            # Determine test type
            if "performance" in test_file:
                move_file(test_file, f"tests/performance/{test_file}")
            elif any(x in test_file for x in ["admin", "ollama", "llm", "rag"]):
                move_file(test_file, f"tests/integration/{test_file}")
            else:
                move_file(test_file, f"tests/unit/{test_file}")
    
    # Move test fixtures
    move_file("test_document.txt", "tests/fixtures/test_document.txt")
    move_file("test_document2.txt", "tests/fixtures/test_document2.txt")
    move_file("test_query.json", "tests/fixtures/test_query.json")
    
    # 3. Move debug scripts
    print("\n3. Organizing debug scripts...")
    debug_scripts = [
        "debug_imports.py",
        "debug_llm_issue.py",
        "debug_rag_failure.py",
        "debug_search.py",
        "debug_search_flow.py",
        "debug_search_similar_text.py",
        "debug_vector_search.py",
        "debug_vector_similarity.py",
    ]
    
    for script in debug_scripts:
        move_file(script, f"scripts/debug/{script}")
    
    # 4. Move setup and deployment scripts
    print("\n4. Organizing setup/deployment scripts...")
    setup_scripts = [
        "setup_rag_system.py",
        "quick_start.py",
        "start_simple_rag.py",
        "start_admin_system.py",
        "init_vector_index.py",
        "run_core.py",
    ]
    
    for script in setup_scripts:
        move_file(script, f"scripts/setup/{script}")
        
    move_file("start_server.bat", "scripts/deployment/start_server.bat")
    
    # 5. Move maintenance scripts
    print("\n5. Organizing maintenance scripts...")
    maintenance_scripts = [
        "cleanup_script.py",
        "clean_vector_index.py",
        "fix_document_processing.py",
        "fix_unicode.py",
        "organize_phase2.py",
        "simple_organize.py",
        "switch_model.py",
    ]
    
    for script in maintenance_scripts:
        move_file(script, f"scripts/maintenance/{script}")
    
    # 6. Move documentation
    print("\n6. Organizing documentation...")
    
    # Admin docs
    admin_docs = [
        "ADMIN_FIXED_READY.md",
        "ADMIN_SYSTEM_COMPLETE.md",
        "ADMIN_SYSTEM_GUIDE.md",
    ]
    
    for doc in admin_docs:
        move_file(doc, f"docs/admin/{doc}")
    
    # Guide docs
    guide_docs = [
        "SIMPLE_RAG_README.md",
        "NORMAL_STARTUP_GUIDE.md",
        "FRONTEND_GUIDE.md",
        "DOWNLOAD_ENDPOINT_README.md",
        "SERVER_DEPLOYMENT_GUIDE.md",
    ]
    
    for doc in guide_docs:
        move_file(doc, f"docs/guides/{doc}")
    
    # Architecture docs
    arch_docs = [
        "OBJECTIVE_BEST_SOLUTION.md",
        "PROJECT_STRUCTURE.md",
    ]
    
    for doc in arch_docs:
        move_file(doc, f"docs/architecture/{doc}")
        
    # Legacy docs
    move_file("LEGACY_FILES.md", "docs/legacy/LEGACY_FILES.md")
    
    # 7. Move training tools
    print("\n7. Organizing tools...")
    training_tools = [
        "train_arlesheim_model.py",
        "train_gaming_pc.py", 
        "train_rtx3070.py",
        "train_simple_rtx3070.py",
    ]
    
    for tool in training_tools:
        move_file(tool, f"tools/training/{tool}")
        
    # Move municipal tools to legacy
    if Path("tools/municipal").exists():
        shutil.move("tools/municipal", "tools/legacy_municipal")
        print("Moved: tools/municipal -> tools/legacy_municipal")
        
    move_file("tools/municipal_web_scraper.py", "tools/legacy_municipal/municipal_web_scraper.py")
    move_file("tools/municipal_model_trainer.py", "tools/legacy_municipal/municipal_model_trainer.py")
    
    # 8. Move requirements files
    print("\n8. Organizing deployment files...")
    move_file("simple_requirements.txt", "deployment/requirements/requirements.txt")
    move_file("rtx3070_requirements.txt", "deployment/requirements/requirements-rtx3070.txt")
    
    # 9. Move example files
    print("\n9. Organizing examples...")
    move_file("demo_rag_vs_training.py", "examples/demos/demo_rag_vs_training.py")
    
    # Move example-website if exists
    if Path("example-website").exists():
        shutil.move("example-website", "examples/website")
        print("Moved: example-website -> examples/website")
        
    # 10. Create simple entry point
    print("\n10. Creating main entry point...")
    if Path("simple_api.py").exists():
        print("Found simple_api.py - keeping as main entry point")
    else:
        # Create a simple wrapper
        wrapper_content = '''#!/usr/bin/env python3
"""
Main entry point for the RAG system
"""

import sys
import os

# Add src to path if it exists
if os.path.exists("src"):
    sys.path.insert(0, "src")

# Try to import from different locations
try:
    from core.main import main
except ImportError:
    try:
        from src.main import main
    except ImportError:
        print("Error: Could not find main application module")
        print("Please ensure the application is properly installed")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        with open("simple_api.py", "w") as f:
            f.write(wrapper_content)
        print("Created: simple_api.py (main entry point)")
    
    # 11. Create organized .gitignore
    print("\n11. Updating .gitignore...")
    gitignore_additions = '''
# Organized structure ignores
data/storage/
data/cache/
data/logs/
data/databases/
*.db
*.db-shm
*.db-wal

# Test artifacts
tests/fixtures/temp/
.coverage
htmlcov/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Excel temp files
~$*.xlsx

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db
'''
    
    with open(".gitignore", "a") as f:
        f.write(gitignore_additions)
    print("Updated: .gitignore")
    
    # 12. Summary
    print("\n" + "="*50)
    print("ORGANIZATION COMPLETE!")
    print("="*50)
    print("\nKey changes:")
    print("- Test files organized in tests/ directory")
    print("- Scripts organized in scripts/ directory") 
    print("- Documentation organized in docs/ directory")
    print("- Tools organized in tools/ directory")
    print("- Municipal files moved to legacy directories")
    print("\nMain entry point: simple_api.py")
    print("\nNext steps:")
    print("1. Review the changes")
    print("2. Update any broken imports")
    print("3. Run tests to ensure everything works")
    print("4. Commit the organized structure")

if __name__ == "__main__":
    main()