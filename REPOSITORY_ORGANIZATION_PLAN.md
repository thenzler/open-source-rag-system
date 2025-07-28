# Repository Organization Plan

## 🎯 Current Status
Repository has grown to ~400+ files with mixed production code, legacy systems, documentation, and development tools all in the root directory.

## 📊 Analysis Summary

**Total Files Analyzed:** ~400+
- **Production Critical:** ~30 files (core system)
- **Legacy/Outdated:** ~50 files (can be removed/archived)
- **Documentation:** ~80 files (needs organization)
- **Development Tools:** ~100 files (can be moved to dev/ folder)
- **Testing:** ~40 files (already mostly organized)
- **Data/Storage:** ~100+ files (database and documents)

---

## 🚀 PRODUCTION CRITICAL FILES
*These files are essential for the current working system*

### Core System (KEEP IN ROOT)
```
core/                           # Modular FastAPI application
├── main.py                    # FastAPI entry point
├── ollama_client.py           # LLM integration
├── routers/                   # API endpoints
├── services/simple_rag_service.py  # Core RAG logic
├── repositories/              # Data access layer
├── di/                        # Dependency injection
└── models/                    # Data models

run_core.py                    # Production startup script
requirements.txt               # Python dependencies
config/llm_config.yaml         # LLM configuration
static/index.html              # Main web interface
```

### Storage & Data (KEEP AS IS)
```
data/
├── rag_database.db*           # Main SQLite database
├── audit.db*                  # Audit logging
└── storage/                   # Document storage
    ├── uploads/               # User uploads
    └── processed/             # Processed documents
```

---

## 🗑️ LEGACY FILES TO REMOVE/ARCHIVE

### Outdated Simple API System (ARCHIVE)
```
❌ simple_api.py               # Replaced by core/main.py
❌ simple_frontend.html        # Replaced by static/index.html
❌ start_simple_rag.py        # Replaced by run_core.py
❌ simple_requirements.txt    # Replaced by requirements.txt
❌ ollama_client.py           # Moved to core/ollama_client.py
```

### Legacy Services & Config (ARCHIVE)
```
❌ api/simple_rag_api.py      # Old API implementation
❌ api/confidence_endpoints.py # Old confidence system
❌ services/                  # Old service implementations
❌ config/config.py           # Old configuration
❌ config/database_config.py  # Old database config
❌ config/confidence_config.yaml # Unused confidence system
❌ storage/                   # Old storage (replaced by data/storage/)
```

### Cleanup Documentation (REMOVE)
```
❌ cleanup_script.py
❌ CLEANUP_PLAN.md
❌ CLEANUP_SUMMARY.md
❌ CODE_FIXES_SUMMARY.md
❌ ZERO_HALLUCINATION_PLAN.md
```

---

## 📁 REORGANIZATION PLAN

### 1. Create Archive Directory
```
.archive/
├── legacy-simple-api/         # Old monolithic system
├── old-services/              # Legacy service implementations
├── old-config/                # Outdated configuration
├── cleanup-docs/              # Temporary cleanup documentation
└── old-storage/               # Legacy storage system
```

### 2. Create Development Directory
```
dev/
├── tools/                     # Development utilities
├── training/                  # Model training scripts
├── debug/                     # Debug utilities
├── monitoring/                # Grafana/Prometheus
└── deployment/                # Docker, nginx configs
```

### 3. Organize Documentation
```
docs/
├── user/                      # End-user documentation
│   ├── README.md
│   ├── SIMPLE_RAG_README.md
│   └── QUICKSTART.md
├── technical/                 # Technical documentation
├── business/                  # Business strategy docs
└── api/                       # API documentation
```

### 4. Clean Root Directory
**After reorganization, root should contain only:**
```
📁 core/                       # Main application
📁 data/                       # Databases and storage
📁 docs/                       # Organized documentation
📁 tests/                      # Test suite
📁 dev/                        # Development tools
📁 static/                     # Frontend assets
📁 widget/                     # Embeddable widget
📁 example-website/            # Example integration
📁 .archive/                   # Legacy files

📄 run_core.py                 # Production startup
📄 requirements.txt            # Dependencies
📄 CLAUDE.md                   # AI assistant instructions
📄 README.md                   # Main documentation
📄 LICENSE                     # License file
📄 pytest.ini                  # Test configuration
```

---

## 🛠️ IMPLEMENTATION STEPS

### Phase 1: Archive Legacy Files (SAFE)
1. Create `.archive/legacy-simple-api/`
2. Move all legacy simple API files
3. Create `.archive/cleanup-docs/`
4. Move all CLEANUP_*.md files

### Phase 2: Reorganize Development Tools
1. Create `dev/tools/` 
2. Move `tools/` directory contents
3. Move `deployment/`, `monitoring/`, `nginx/`
4. Move debug scripts (`debug_*.py`)

### Phase 3: Clean Documentation
1. Reorganize `docs/` structure
2. Move user docs to `docs/user/`
3. Keep `CLAUDE.md` and `README.md` in root

### Phase 4: Verify System Works
1. Test production system with `python run_core.py`
2. Run test suite: `python -m pytest tests/`
3. Verify all endpoints work

---

## 🎯 EXPECTED RESULTS

### Before Cleanup:
- **Root directory:** 50+ files
- **Total files:** 400+ files
- **Confusing structure** with mixed legacy/production code

### After Cleanup:
- **Root directory:** ~15 essential files
- **Archived files:** ~50 legacy files safely stored
- **Organized structure** with clear separation of concerns
- **Faster navigation** and easier maintenance

---

## ⚠️ SAFETY NOTES

1. **Test thoroughly** after each phase
2. **Keep archives** until system is verified working
3. **Database files** are never moved (critical data)
4. **CLAUDE.md stays in root** (AI assistant needs it)
5. **Core directory** is never modified (production system)

---

## 🚀 NEXT STEPS

1. **Review this plan** - Confirm approach is correct
2. **Execute Phase 1** - Archive legacy files (safest first)
3. **Test system** - Ensure production system still works
4. **Continue phases** - Gradually reorganize remaining files
5. **Update documentation** - Reflect new structure

This plan will transform the repository from a cluttered development workspace into a clean, production-ready codebase while preserving all critical functionality and maintaining complete system safety.