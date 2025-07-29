#!/usr/bin/env python3
"""
Repository Reorganization Script
This script reorganizes the repository structure to follow best practices
"""

import os
import shutil
from pathlib import Path
import json

class RepositoryReorganizer:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.root = Path.cwd()
        self.moves = []
        self.creates = []
        self.deletes = []
        
    def log(self, message):
        """Log actions"""
        print(f"{'[DRY RUN] ' if self.dry_run else ''}[FILE] {message}")
        
    def create_directory(self, path):
        """Create a directory"""
        full_path = self.root / path
        self.creates.append(str(path))
        if not self.dry_run:
            full_path.mkdir(parents=True, exist_ok=True)
        self.log(f"Create directory: {path}")
        
    def move_file(self, src, dst):
        """Move a file or directory"""
        src_path = self.root / src
        dst_path = self.root / dst
        
        if src_path.exists():
            self.moves.append((str(src), str(dst)))
            if not self.dry_run:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_path), str(dst_path))
            self.log(f"Move: {src} -> {dst}")
        else:
            self.log(f"Skip (not found): {src}")
            
    def delete_file(self, path):
        """Delete a file or directory"""
        full_path = self.root / path
        if full_path.exists():
            self.deletes.append(str(path))
            if not self.dry_run:
                if full_path.is_dir():
                    shutil.rmtree(full_path)
                else:
                    full_path.unlink()
            self.log(f"Delete: {path}")
            
    def reorganize(self):
        """Main reorganization logic"""
        print("[START] Starting repository reorganization...")
        
        # 1. Create new directory structure
        self._create_directory_structure()
        
        # 2. Move core application files
        self._move_source_code()
        
        # 3. Organize tests
        self._organize_tests()
        
        # 4. Organize scripts
        self._organize_scripts()
        
        # 5. Organize documentation
        self._organize_documentation()
        
        # 6. Organize tools
        self._organize_tools()
        
        # 7. Clean up root directory
        self._cleanup_root()
        
        # 8. Create new configuration files
        self._create_config_files()
        
        # Save reorganization log
        self._save_log()
        
        print(f"\n[DONE] Reorganization complete!")
        print(f"[STATS] Summary: {len(self.creates)} directories created, {len(self.moves)} moves, {len(self.deletes)} deletions")
        
    def _create_directory_structure(self):
        """Create the new directory structure"""
        print("\n[DIR] Creating new directory structure...")
        
        directories = [
            # Source code
            "src/api/routers",
            "src/api/middleware",
            "src/core/models",
            "src/core/repositories",
            "src/core/services",
            "src/config",
            "src/di",
            "src/utils",
            
            # Tests
            "tests/unit",
            "tests/integration",
            "tests/performance",
            "tests/fixtures/test_documents",
            
            # Scripts
            "scripts/setup",
            "scripts/maintenance",
            "scripts/debug",
            "scripts/deployment",
            
            # Tools
            "tools/benchmarks",
            "tools/training",
            "tools/municipal",
            "tools/project_management",
            
            # Documentation
            "docs/setup",
            "docs/development",
            "docs/api",
            "docs/architecture",
            "docs/operations",
            "docs/business/strategy",
            "docs/business/requirements",
            "docs/business/swiss-market",
            
            # Deployment
            "deployment/docker",
            "deployment/nginx",
            "deployment/monitoring",
            "deployment/requirements",
            
            # Examples
            "examples/widget",
            "examples/website",
            "examples/demo_scripts",
            
            # Static files
            "static/templates",
            "static/assets",
            
            # Data directories (will be gitignored)
            "data/storage/uploads",
            "data/storage/processed",
            "data/cache",
            "data/logs",
            "data/databases",
        ]
        
        for directory in directories:
            self.create_directory(directory)
            
    def _move_source_code(self):
        """Move source code to proper locations"""
        print("\n[CODE] Moving source code...")
        
        # Move core modules
        if (self.root / "core").exists():
            # Move routers
            for router in ["admin.py", "documents.py", "query.py", "system.py", "llm.py", "document_manager.py"]:
                self.move_file(f"core/routers/{router}", f"src/api/routers/{router}")
            self.move_file("core/routers/__init__.py", "src/api/routers/__init__.py")
            
            # Move services
            for service in Path("core/services").glob("*.py"):
                self.move_file(f"core/services/{service.name}", f"src/core/services/{service.name}")
                
            # Move repositories
            for repo in Path("core/repositories").glob("*.py"):
                self.move_file(f"core/repositories/{repo.name}", f"src/core/repositories/{repo.name}")
                
            # Move DI
            self.move_file("core/di/container.py", "src/di/container.py")
            self.move_file("core/di/services.py", "src/di/services.py")
            self.move_file("core/di/__init__.py", "src/di/__init__.py")
            
            # Move other core files
            self.move_file("core/ollama_client.py", "src/core/ollama_client.py")
            self.move_file("core/startup_checks.py", "src/utils/startup_checks.py")
            self.move_file("core/main.py", "src/main.py")
            
        # Move config files
        self.move_file("config/llm_config.yaml", "src/config/llm_config.yaml")
        self.move_file("config/production_config.yaml", "src/config/production_config.yaml")
        
        # Move templates
        if (self.root / "core/templates").exists():
            for template in Path("core/templates").glob("*.html"):
                self.move_file(f"core/templates/{template.name}", f"static/templates/{template.name}")
                
    def _organize_tests(self):
        """Organize test files"""
        print("\n[TEST] Organizing tests...")
        
        # Unit tests
        unit_tests = [
            "test_confidence_system.py",
            "test_configurable_confidence.py",
            "test_fixes_directly.py",
            "test_processing.py",
            "test_single_endpoint.py",
            "test_zero_hallucination.py",
        ]
        
        for test in unit_tests:
            if (self.root / test).exists():
                self.move_file(test, f"tests/unit/{test}")
                
        # Integration tests
        integration_tests = [
            "test_simple_rag.py",
            "test_ollama_integration.py",
            "test_admin_interface.py",
            "test_admin_fixed.py",
            "test_admin_import.py",
            "test_cleanup_api.py",
            "test_cleaned_rag.py",
            "test_llm_integration.py",
            "test_ollama_direct.py",
            "test_rag_fixes.py",
        ]
        
        for test in integration_tests:
            if (self.root / test).exists():
                self.move_file(test, f"tests/integration/{test}")
                
        # Performance tests
        performance_tests = [
            "test_performance.py",
            "test_qwen_performance.py",
        ]
        
        for test in performance_tests:
            if (self.root / test).exists():
                self.move_file(test, f"tests/performance/{test}")
                
        # Move test files and fixtures
        self.move_file("test_document.txt", "tests/fixtures/test_document.txt")
        self.move_file("test_document2.txt", "tests/fixtures/test_document2.txt")
        self.move_file("test_query.json", "tests/fixtures/test_query.json")
        
        # Move existing test directory contents
        if (self.root / "tests").exists():
            for test_file in Path("tests").glob("test_*.py"):
                if test_file.is_file():
                    self.move_file(f"tests/{test_file.name}", f"tests/integration/{test_file.name}")
                    
    def _organize_scripts(self):
        """Organize scripts"""
        print("\n[SCRIPT] Organizing scripts...")
        
        # Setup scripts
        setup_scripts = [
            "setup_rag_system.py",
            "quick_start.py",
            "start_simple_rag.py",
            "start_admin_system.py",
            "init_vector_index.py",
        ]
        
        for script in setup_scripts:
            if (self.root / script).exists():
                self.move_file(script, f"scripts/setup/{script}")
                
        # Maintenance scripts
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
            if (self.root / script).exists():
                self.move_file(script, f"scripts/maintenance/{script}")
                
        # Debug scripts
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
            if (self.root / script).exists():
                self.move_file(script, f"scripts/debug/{script}")
                
        # Deployment scripts
        self.move_file("start_server.bat", "scripts/deployment/start_server.bat")
        self.move_file("run_core.py", "scripts/deployment/run_core.py")
        
    def _organize_documentation(self):
        """Organize documentation"""
        print("\n[DOCS] Organizing documentation...")
        
        # Setup documentation
        setup_docs = [
            "SIMPLE_RAG_README.md",
            "NORMAL_STARTUP_GUIDE.md",
            "QUICKSTART.md",
            "SERVER_DEPLOYMENT_GUIDE.md",
        ]
        
        for doc in setup_docs:
            if (self.root / doc).exists():
                self.move_file(doc, f"docs/setup/{doc}")
                
        # Development documentation
        dev_docs = [
            "CLAUDE.md",
            "TESTING.md",
            "FRONTEND_GUIDE.md",
            "DEVELOPMENT_NOTES.md",
        ]
        
        for doc in dev_docs:
            if (self.root / doc).exists():
                self.move_file(doc, f"docs/development/{doc}")
                
        # Architecture documentation
        arch_docs = [
            "OBJECTIVE_BEST_SOLUTION.md",
            "PROJECT_STRUCTURE.md",
            "ADMIN_SYSTEM_GUIDE.md",
            "ADMIN_SYSTEM_COMPLETE.md",
            "ADMIN_FIXED_READY.md",
        ]
        
        for doc in arch_docs:
            if (self.root / doc).exists():
                self.move_file(doc, f"docs/architecture/{doc}")
                
        # Operations documentation
        ops_docs = [
            "DOWNLOAD_ENDPOINT_README.md",
            "LEGACY_FILES.md",
        ]
        
        for doc in ops_docs:
            if (self.root / doc).exists():
                self.move_file(doc, f"docs/operations/{doc}")
                
        # Move existing docs subdirectories
        if (self.root / "docs").exists():
            # API docs
            self.move_file("docs/API_DOCUMENTATION.md", "docs/api/API_DOCUMENTATION.md")
            self.move_file("docs/WIDGET_INTEGRATION_GUIDE.md", "docs/api/WIDGET_INTEGRATION_GUIDE.md")
            
            # Architecture docs
            self.move_file("docs/ARCHITECTURE.md", "docs/architecture/ARCHITECTURE.md")
            self.move_file("docs/TECHNOLOGY_STACK.md", "docs/architecture/TECHNOLOGY_STACK.md")
            
            # Other docs
            self.move_file("docs/DOMAIN_CONFIGURATION_GUIDE.md", "docs/operations/DOMAIN_CONFIGURATION_GUIDE.md")
            
    def _organize_tools(self):
        """Organize tools"""
        print("\n[TOOLS] Organizing tools...")
        
        # Training tools
        training_tools = [
            "train_arlesheim_model.py",
            "train_gaming_pc.py",
            "train_rtx3070.py",
            "train_simple_rtx3070.py",
        ]
        
        for tool in training_tools:
            if (self.root / tool).exists():
                self.move_file(tool, f"tools/training/{tool}")
                
        # Move tools subdirectories
        if (self.root / "tools").exists():
            # Municipal tools
            if (self.root / "tools/municipal").exists():
                for file in Path("tools/municipal").glob("*.py"):
                    self.move_file(f"tools/municipal/{file.name}", f"tools/municipal/{file.name}")
                    
            # Training tools
            if (self.root / "tools/training").exists():
                for file in Path("tools/training").glob("*.py"):
                    self.move_file(f"tools/training/{file.name}", f"tools/training/{file.name}")
                    
            # Other tools
            self.move_file("tools/municipal_web_scraper.py", "tools/municipal/municipal_web_scraper.py")
            self.move_file("tools/municipal_model_trainer.py", "tools/municipal/municipal_model_trainer.py")
            
    def _cleanup_root(self):
        """Clean up root directory"""
        print("\n[CLEAN] Cleaning up root directory...")
        
        # Move requirements files
        self.move_file("simple_requirements.txt", "deployment/requirements/requirements.txt")
        self.move_file("rtx3070_requirements.txt", "deployment/requirements/requirements-rtx3070.txt")
        
        # Move example and demo files
        self.move_file("demo_rag_vs_training.py", "examples/demo_scripts/demo_rag_vs_training.py")
        
        # Move static files
        if (self.root / "static").exists():
            for file in Path("static").glob("*"):
                if file.is_file():
                    self.move_file(f"static/{file.name}", f"static/assets/{file.name}")
                    
        # Delete temporary files
        temp_files = [
            "test_fix.py",
            "test_fix_simple.py",
            "simple_config.py",
            "reorganize_repository.py",  # This script itself
        ]
        
        for temp_file in temp_files:
            if (self.root / temp_file).exists():
                self.delete_file(temp_file)
                
    def _create_config_files(self):
        """Create new configuration files"""
        print("\n[CONFIG] Creating configuration files...")
        
        # Create pyproject.toml
        pyproject_content = '''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "open-source-rag"
version = "2.0.0"
description = "Production-ready RAG system with comprehensive admin interface"
readme = "README.md"
requires-python = ">=3.8"
license = {file = "LICENSE"}
authors = [
    {name = "RAG System Contributors"},
]
dependencies = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.22.0",
    "sqlalchemy>=2.0.0",
    "pydantic>=2.0.0",
    "sentence-transformers>=2.2.0",
    "pypdf2>=3.0.0",
    "python-docx>=0.8.11",
    "pandas>=2.0.0",
    "httpx>=0.24.0",
    "python-multipart>=0.0.6",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
]

[project.scripts]
rag-server = "src.main:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
'''
        
        if not self.dry_run:
            with open(self.root / "pyproject.toml", "w") as f:
                f.write(pyproject_content)
        self.log("Create: pyproject.toml")
        
        # Create setup.py for compatibility
        setup_content = '''from setuptools import setup

setup()
'''
        
        if not self.dry_run:
            with open(self.root / "setup.py", "w") as f:
                f.write(setup_content)
        self.log("Create: setup.py")
        
        # Create __init__.py files
        init_files = [
            "src/__init__.py",
            "src/api/__init__.py",
            "src/core/__init__.py",
            "src/config/__init__.py",
            "src/utils/__init__.py",
            "tests/__init__.py",
            "tests/unit/__init__.py",
            "tests/integration/__init__.py",
            "tests/performance/__init__.py",
        ]
        
        for init_file in init_files:
            if not self.dry_run:
                Path(self.root / init_file).parent.mkdir(parents=True, exist_ok=True)
                Path(self.root / init_file).touch()
            self.log(f"Create: {init_file}")
            
    def _save_log(self):
        """Save reorganization log"""
        log_data = {
            "timestamp": str(Path.cwd()),
            "dry_run": self.dry_run,
            "creates": self.creates,
            "moves": self.moves,
            "deletes": self.deletes,
        }
        
        log_file = "reorganization_log.json"
        if not self.dry_run:
            with open(log_file, "w") as f:
                json.dump(log_data, f, indent=2)
        self.log(f"Save log: {log_file}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reorganize repository structure")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the reorganization (default is dry run)"
    )
    
    args = parser.parse_args()
    
    reorganizer = RepositoryReorganizer(dry_run=not args.execute)
    reorganizer.reorganize()
    
    if not args.execute:
        print("\n[WARNING] This was a DRY RUN. To execute the reorganization, run:")
        print("    python reorganize_repository.py --execute")


if __name__ == "__main__":
    main()