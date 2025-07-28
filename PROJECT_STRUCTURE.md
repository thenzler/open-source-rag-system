# 📁 Project Structure (After Cleanup)

**Clean, organized RAG system ready for Swiss market launch**

## 🎯 Overview
This project structure has been optimized for a **lean 2-4 person team** with clear separation between MVP core, business documentation, and archived code.

## 📂 Directory Structure

```
open-source-rag-system/
├── 📦 core/                    # Core MVP Application (Ready to Ship)
│   ├── simple_api.py          # Main FastAPI server
│   ├── ollama_client.py       # LLM integration
│   ├── simple_frontend.html   # Web interface
│   ├── start_simple_rag.py    # Startup script
│   └── startup_checks.py      # System health checks
│
├── 📚 docs/                    # All Documentation
│   ├── README.md              # Main project documentation
│   ├── CLAUDE.md              # AI assistant instructions
│   ├── business/              # Business & Strategy Documents
│   │   ├── requirements/      # Project management & requirements
│   │   ├── strategy/          # Business strategy & roadmaps
│   │   ├── swiss-market/      # Swiss market analysis & plans
│   │   └── training/          # Model training guides
│   ├── technical/             # Technical Documentation
│   └── api/                   # API Documentation
│
├── 🧪 tests/                   # All Test Files
│   ├── test_simple_rag.py     # Core system tests
│   ├── test_ollama_integration.py # LLM integration tests
│   ├── test_api.py            # API endpoint tests
│   └── test_download_endpoint.py # File download tests
│
├── 🛠️ tools/                   # Utility Scripts & Tools
│   ├── training/              # Model Training Scripts
│   │   ├── train_arlesheim_model.py
│   │   ├── train_rtx3070.py
│   │   ├── create_german_training_data.py
│   │   └── fine_tune_arlesheim.py
│   ├── municipal/             # Municipal-Specific Tools
│   │   ├── municipal_setup.py
│   │   ├── demo_municipal_rag.py
│   │   └── municipal_rag.py
│   └── deployment/            # Deployment Utilities
│
├── ⚙️ services/               # Core Services (MVP Only)
│   ├── async_processor.py    # Background processing
│   ├── auth.py               # Authentication
│   ├── document_manager.py   # Document handling
│   ├── llm_manager.py        # LLM orchestration
│   ├── vector_search.py      # Search functionality
│   └── smart_answer.py       # Answer generation
│
├── 📊 data/                   # Data Storage (gitignored)
│   ├── storage/              # Document storage
│   ├── training_data/        # Training datasets
│   └── logs/                 # Application logs
│
├── 🚀 deployment/             # Deployment Configuration
│   ├── requirements/         # Dependency specifications
│   │   ├── simple_requirements.txt
│   │   ├── rtx3070_requirements.txt
│   │   └── test_requirements.txt
│   ├── docker/               # Docker configurations
│   └── scripts/              # Deployment scripts
│
├── ⚙️ config/                 # Configuration Files
│   ├── llm_config.yaml       # LLM settings
│   └── database_config.py    # Database configuration
│
├── 🗄️ .archive/               # Archived Code (gitignored)
│   ├── old-code/             # Legacy code from delete/ folder
│   └── overengineered-services/ # Complex microservices
│       ├── api-gateway/      # Overengineered API gateway
│       ├── document-processor/ # Standalone doc processor
│       ├── vector-engine/    # Dedicated vector service
│       └── web-interface/    # Complex React interface
│
├── requirements.txt          # Main requirements file
├── .gitignore               # Git ignore rules
├── LICENSE                  # Project license
└── README.md               # Project overview
```

## 🎯 Key Improvements

### ✅ **File Count Reduction**
- **Before**: 43,496 files (mostly node_modules)
- **After**: ~200 active files (99.5% reduction)

### ✅ **Clear Organization**
- **MVP Focus**: Core files in `core/` directory
- **Business Docs**: Strategy and planning in `docs/business/`
- **Clean Separation**: Archive vs. active code
- **Easy Navigation**: Logical file grouping

### ✅ **Team-Friendly Structure**
- **Developers**: Focus on `core/` and `services/`
- **Business Team**: Focus on `docs/business/`
- **Everyone**: Clear understanding of what's active vs. archived

## 🚀 Quick Start

### For Development:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the system
python core/start_simple_rag.py

# 3. Access web interface
http://localhost:8000
```

### For Training Models:
```bash
# GPU training (RTX 3070)
pip install -r deployment/requirements/rtx3070_requirements.txt
python tools/training/train_rtx3070.py
```

### For Testing:
```bash
# Install test dependencies
pip install -r deployment/requirements/test_requirements.txt

# Run tests
pytest tests/
```

## 📋 Swiss Market Launch Ready

This structure is optimized for the **6-week Swiss market launch plan**:

- **Week 1-2**: Core development in `core/` directory
- **Week 3-4**: Business execution using `docs/business/` plans
- **Week 5-6**: Deployment using `deployment/` configurations

## 🔒 Archived Components

The `.archive/` directory contains:
- **Overengineered microservices** (not needed for MVP)
- **Complex React interface** (simple HTML serves MVP better)
- **Legacy debug code** (preserved for reference)

These can be reintroduced later when the business scales beyond MVP.

## 📞 Contact & Support

- **Documentation**: See `docs/README.md`
- **Business Plans**: See `docs/business/`
- **Technical Issues**: Check `docs/technical/`
- **Swiss Market**: See `docs/business/swiss-market/`

---

**Status**: ✅ Ready for Swiss market launch
**Team**: Optimized for 2-4 person team
**Focus**: MVP-first approach with business scaling plans