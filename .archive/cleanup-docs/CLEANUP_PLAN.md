# 🧹 Folder Cleanup & Restructuring Plan

## Current Issues
- **43,496 total files** with 43,441 in services/ (mostly node_modules)
- Multiple redundant test files and training scripts
- Obsolete code mixed with active code
- Documentation scattered across multiple locations
- MVP files mixed with overengineered microservices

## Proposed New Structure

```
open-source-rag-system/
├── 📦 core/                      # Core MVP Application
│   ├── simple_api.py            # Main API server
│   ├── ollama_client.py         # LLM integration
│   ├── simple_frontend.html     # Web interface
│   ├── start_simple_rag.py      # Startup script
│   └── startup_checks.py        # System checks
│
├── 📚 docs/                      # All Documentation
│   ├── README.md                # Main project README
│   ├── CLAUDE.md               # AI assistant instructions
│   ├── technical/              # Technical docs
│   ├── business/               # Business/strategy docs
│   └── api/                    # API documentation
│
├── 🧪 tests/                     # All Test Files
│   ├── test_simple_rag.py
│   ├── test_ollama_integration.py
│   └── test_api.py
│
├── 🛠️ tools/                     # Utility Scripts
│   ├── training/               # Training scripts
│   ├── municipal/              # Municipal tools
│   └── deployment/             # Deployment utilities
│
├── ⚙️ config/                    # Configuration Files
│   ├── llm_config.yaml
│   ├── database_config.yaml
│   └── .env.example
│
├── 📊 data/                      # Data Storage
│   ├── storage/                # Document storage
│   ├── training_data/          # Training datasets
│   └── logs/                   # Application logs
│
├── 🚀 deployment/                # Deployment Files
│   ├── docker/                 # Docker configs
│   ├── scripts/                # Deployment scripts
│   └── requirements/           # Dependencies
│
├── 🗄️ .archive/                  # Archived/Old Code
│   └── [old files moved here]
│
├── .gitignore
├── LICENSE
└── requirements.txt            # Main requirements file
```

## Cleanup Actions

### 1. **Archive Obsolete Code**
```bash
# Move delete/ folder to .archive/
mv delete/ .archive/

# Archive overengineered services
mkdir -p .archive/overengineered-services
mv services/api-gateway .archive/overengineered-services/
mv services/document-processor .archive/overengineered-services/
mv services/vector-engine .archive/overengineered-services/
mv services/auth-service .archive/overengineered-services/
```

### 2. **Consolidate Core Files**
```bash
# Create core directory for MVP files
mkdir -p core
mv simple_api.py core/
mv ollama_client.py core/
mv simple_frontend.html core/
mv start_simple_rag.py core/
mv startup_checks.py core/
```

### 3. **Organize Tests**
```bash
# Move all test files to tests/
mv test_download_endpoint.py tests/
# Remove duplicate test files in delete/Old/tests/
```

### 4. **Consolidate Training Scripts**
```bash
# Move training scripts to tools/training/
mkdir -p tools/training
mv train_*.py tools/training/
mv create_german_training_data.py tools/training/
mv fine_tune_arlesheim.py tools/training/
```

### 5. **Organize Municipal Files**
```bash
# Create municipal directory
mkdir -p tools/municipal
mv municipal_*.py tools/municipal/
mv demo_municipal_rag.py tools/municipal/
# Move services/municipal_* to tools/municipal/
```

### 6. **Clean Documentation**
```bash
# Move business docs
mkdir -p docs/business
mv strategy/* docs/business/
mv enterprise-requirements docs/business/

# Move technical docs
mkdir -p docs/technical
mv docs/TESTING.md docs/technical/
mv TESTING.md .archive/  # Remove duplicate
```

### 7. **Consolidate Requirements**
```bash
# Create single requirements directory
mkdir -p deployment/requirements
mv *requirements*.txt deployment/requirements/
# Keep main requirements.txt in root
cp deployment/requirements/simple_requirements.txt requirements.txt
```

### 8. **Data Organization**
```bash
# Create proper data structure
mkdir -p data
mv storage/ data/
mv training_data/ data/
mv logs/ data/
mv *.db* data/  # Move database files
```

## Files to Delete

### Definitely Delete:
- Everything in `delete/Old/` after archiving
- Duplicate test files
- Old requirements files after consolidation
- `node_modules` in services/web-interface (43,000+ files!)

### Consider Deleting:
- Complex microservices in `services/` (not needed for MVP)
- `monitoring/` setup (overengineered for current stage)
- `nginx/` configuration (not needed yet)

## Quick Cleanup Commands

```bash
# 1. Remove node_modules (saves 43,000+ files)
rm -rf services/web-interface/node_modules

# 2. Archive old code
mkdir -p .archive
mv delete/ .archive/old-code/

# 3. Create new structure
mkdir -p core docs/technical docs/business tests tools/training tools/municipal data deployment/requirements

# 4. Move files to new locations (run cleanup script)
# See detailed commands above

# 5. Update .gitignore
echo ".archive/" >> .gitignore
echo "data/logs/" >> .gitignore
echo "data/*.db*" >> .gitignore
```

## Post-Cleanup Tasks

1. **Update import paths** in Python files after moving
2. **Update README.md** with new structure
3. **Test that everything still works**
4. **Commit with clear message**: "Restructure: Organize codebase for clarity and maintainability"

## Expected Results

- **From**: 43,496 files → **To**: ~200 files (after removing node_modules and archiving)
- Clear separation between MVP code and future features
- Easy to find any file based on its purpose
- Better for new team members to understand the project
- Cleaner git history going forward

Would you like me to create a script to automate this cleanup process?