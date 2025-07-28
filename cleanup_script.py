#!/usr/bin/env python3
"""
🧹 Automated Cleanup Script for RAG System
Reorganizes the project structure for better maintainability
"""

import os
import shutil
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ProjectCleaner:
    def __init__(self, project_root="."):
        self.root = Path(project_root).resolve()
        self.dry_run = False
        
    def create_directories(self):
        """Create new directory structure"""
        directories = [
            "core",
            "docs/technical",
            "docs/business", 
            "docs/api",
            "tests",
            "tools/training",
            "tools/municipal",
            "tools/deployment",
            "config",
            "data/storage",
            "data/training_data",
            "data/logs",
            "deployment/docker",
            "deployment/scripts",
            "deployment/requirements",
            ".archive/old-code",
            ".archive/overengineered-services"
        ]
        
        for dir_path in directories:
            full_path = self.root / dir_path
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created: {dir_path}")
                
    def move_file(self, source, destination):
        """Move a file with logging"""
        src_path = self.root / source
        dst_path = self.root / destination
        
        if src_path.exists():
            if self.dry_run:
                logger.info(f"🔄 Would move: {source} → {destination}")
            else:
                # Create destination directory if needed
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_path), str(dst_path))
                logger.info(f"✅ Moved: {source} → {destination}")
        else:
            logger.debug(f"⚠️  File not found: {source}")
            
    def cleanup_core_files(self):
        """Move core MVP files to core/ directory"""
        logger.info("\n📦 Organizing Core Files...")
        
        core_files = [
            ("simple_api.py", "core/simple_api.py"),
            ("ollama_client.py", "core/ollama_client.py"),
            ("simple_frontend.html", "core/simple_frontend.html"),
            ("start_simple_rag.py", "core/start_simple_rag.py"),
            ("startup_checks.py", "core/startup_checks.py"),
        ]
        
        for src, dst in core_files:
            self.move_file(src, dst)
            
    def cleanup_tests(self):
        """Consolidate all test files"""
        logger.info("\n🧪 Organizing Test Files...")
        
        test_files = [
            ("test_download_endpoint.py", "tests/test_download_endpoint.py"),
            ("test_simple_rag.py", "tests/test_simple_rag.py"),
            ("test_ollama_integration.py", "tests/test_ollama_integration.py"),
        ]
        
        for src, dst in test_files:
            self.move_file(src, dst)
            
    def cleanup_training_scripts(self):
        """Move training scripts to tools/training/"""
        logger.info("\n🤖 Organizing Training Scripts...")
        
        training_files = [
            ("train_arlesheim_model.py", "tools/training/train_arlesheim_model.py"),
            ("train_gaming_pc.py", "tools/training/train_gaming_pc.py"),
            ("train_rtx3070.py", "tools/training/train_rtx3070.py"),
            ("train_simple_rtx3070.py", "tools/training/train_simple_rtx3070.py"),
            ("train_german_auto.py", "tools/training/train_german_auto.py"),
            ("create_german_training_data.py", "tools/training/create_german_training_data.py"),
            ("fine_tune_arlesheim.py", "tools/training/fine_tune_arlesheim.py"),
        ]
        
        for src, dst in training_files:
            self.move_file(src, dst)
            
    def cleanup_municipal_files(self):
        """Move municipal files to tools/municipal/"""
        logger.info("\n🏛️  Organizing Municipal Files...")
        
        municipal_files = [
            ("municipal_setup.py", "tools/municipal/municipal_setup.py"),
            ("demo_municipal_rag.py", "tools/municipal/demo_municipal_rag.py"),
            ("services/municipal_api_integration.py", "tools/municipal/municipal_api_integration.py"),
            ("services/municipal_rag.py", "tools/municipal/municipal_rag.py"),
        ]
        
        for src, dst in municipal_files:
            self.move_file(src, dst)
            
    def cleanup_documentation(self):
        """Reorganize documentation"""
        logger.info("\n📚 Organizing Documentation...")
        
        # Move business/strategy docs
        if (self.root / "strategy").exists():
            for file in (self.root / "strategy").glob("*"):
                if file.is_file():
                    self.move_file(f"strategy/{file.name}", f"docs/business/{file.name}")
                    
        # Move enterprise requirements
        if (self.root / "enterprise-requirements").exists():
            self.move_file("enterprise-requirements", "docs/business/enterprise-requirements")
            
        # Move technical docs
        doc_files = [
            ("SIMPLE_RAG_README.md", "docs/technical/SIMPLE_RAG_README.md"),
            ("TESTING.md", "docs/technical/TESTING.md"),
            ("CLAUDE.md", "docs/CLAUDE.md"),  # Keep in docs root
        ]
        
        for src, dst in doc_files:
            self.move_file(src, dst)
            
    def cleanup_requirements(self):
        """Consolidate requirements files"""
        logger.info("\n📋 Organizing Requirements Files...")
        
        req_files = [
            ("simple_requirements.txt", "deployment/requirements/simple_requirements.txt"),
            ("rtx3070_requirements.txt", "deployment/requirements/rtx3070_requirements.txt"),
            ("advanced_requirements.txt", "deployment/requirements/advanced_requirements.txt"),
        ]
        
        for src, dst in req_files:
            self.move_file(src, dst)
            
        # Keep main requirements.txt in root
        simple_req = self.root / "deployment/requirements/simple_requirements.txt"
        if simple_req.exists() and not (self.root / "requirements.txt").exists():
            shutil.copy(str(simple_req), str(self.root / "requirements.txt"))
            logger.info("✅ Created main requirements.txt from simple_requirements.txt")
            
    def archive_old_code(self):
        """Archive obsolete code"""
        logger.info("\n🗄️  Archiving Old Code...")
        
        # Archive delete folder
        if (self.root / "delete").exists():
            self.move_file("delete", ".archive/old-code/delete")
            
        # Archive overengineered services
        services_to_archive = [
            "services/api-gateway",
            "services/document-processor", 
            "services/vector-engine",
            "services/auth-service",
        ]
        
        for service in services_to_archive:
            if (self.root / service).exists():
                service_name = Path(service).name
                self.move_file(service, f".archive/overengineered-services/{service_name}")
                
    def cleanup_data_files(self):
        """Move data files to data/ directory"""
        logger.info("\n💾 Organizing Data Files...")
        
        # Move storage
        if (self.root / "storage").exists():
            self.move_file("storage", "data/storage")
            
        # Move training data
        if (self.root / "training_data").exists():
            self.move_file("training_data", "data/training_data")
            
        # Move logs
        if (self.root / "logs").exists():
            self.move_file("logs", "data/logs")
            
        # Move database files
        for db_file in self.root.glob("*.db*"):
            self.move_file(db_file.name, f"data/{db_file.name}")
            
    def remove_node_modules(self):
        """Remove node_modules directories"""
        logger.info("\n🗑️  Removing node_modules...")
        
        node_modules_path = self.root / "services/web-interface/node_modules"
        if node_modules_path.exists():
            if self.dry_run:
                logger.info(f"🔄 Would remove: {node_modules_path} (43,000+ files)")
            else:
                shutil.rmtree(node_modules_path)
                logger.info(f"✅ Removed: {node_modules_path} (43,000+ files)")
                
    def update_gitignore(self):
        """Update .gitignore with new paths"""
        logger.info("\n📝 Updating .gitignore...")
        
        gitignore_additions = [
            "\n# Archived code",
            ".archive/",
            "\n# Data files", 
            "data/logs/",
            "data/*.db",
            "data/*.db-journal",
            "data/*.db-wal",
            "\n# Python cache",
            "__pycache__/",
            "*.pyc",
            "\n# Environment",
            ".env",
            "venv/",
            ".venv/",
        ]
        
        gitignore_path = self.root / ".gitignore"
        
        if self.dry_run:
            logger.info("🔄 Would update .gitignore")
        else:
            with open(gitignore_path, "a") as f:
                for line in gitignore_additions:
                    f.write(f"{line}\n")
            logger.info("✅ Updated .gitignore")
            
    def create_new_readme(self):
        """Create updated README with new structure"""
        logger.info("\n📄 Creating Structure README...")
        
        readme_content = """# 📁 Project Structure

After cleanup and reorganization:

```
open-source-rag-system/
├── 📦 core/                    # Core MVP Application  
├── 📚 docs/                    # All Documentation
├── 🧪 tests/                   # All Test Files
├── 🛠️ tools/                   # Utility Scripts
├── ⚙️ config/                  # Configuration Files
├── 📊 data/                    # Data Storage (gitignored)
├── 🚀 deployment/              # Deployment Files
├── 🗄️ .archive/                # Archived Code (gitignored)
└── requirements.txt            # Main requirements
```

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `python core/start_simple_rag.py`
3. Access web interface: `http://localhost:8000`

See `docs/README.md` for full documentation.
"""
        
        structure_path = self.root / "PROJECT_STRUCTURE.md"
        
        if self.dry_run:
            logger.info("🔄 Would create PROJECT_STRUCTURE.md")
        else:
            with open(structure_path, "w") as f:
                f.write(readme_content)
            logger.info("✅ Created PROJECT_STRUCTURE.md")
            
    def run_cleanup(self, dry_run=False):
        """Run the complete cleanup process"""
        self.dry_run = dry_run
        
        logger.info(f"\n{'🏃 DRY RUN MODE' if dry_run else '🚀 EXECUTING'} CLEANUP...")
        logger.info(f"Project root: {self.root}\n")
        
        # Execute cleanup steps
        self.create_directories()
        self.cleanup_core_files()
        self.cleanup_tests()
        self.cleanup_training_scripts()
        self.cleanup_municipal_files()
        self.cleanup_documentation()
        self.cleanup_requirements()
        self.archive_old_code()
        self.cleanup_data_files()
        self.remove_node_modules()
        self.update_gitignore()
        self.create_new_readme()
        
        logger.info(f"\n✨ {'DRY RUN' if dry_run else 'CLEANUP'} COMPLETE!")
        
        if dry_run:
            logger.info("\n💡 To execute the cleanup, run: python cleanup_script.py --execute")
            
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up and restructure the RAG project")
    parser.add_argument("--execute", action="store_true", help="Execute the cleanup (default is dry run)")
    parser.add_argument("--root", default=".", help="Project root directory")
    
    args = parser.parse_args()
    
    cleaner = ProjectCleaner(args.root)
    cleaner.run_cleanup(dry_run=not args.execute)
    
if __name__ == "__main__":
    main()