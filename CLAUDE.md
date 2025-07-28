# CLAUDE.md - RAG System Project Settings

## 🎯 Project Overview

**Open Source RAG System** - A complete locally-hosted Retrieval-Augmented Generation system with Ollama LLM integration. This is an MVP-focused project prioritizing reliability, ease of use, and production readiness over advanced features.

**Core Mission**: Create a bulletproof RAG system that works perfectly for the core use case: "Upload documents → Ask questions → Get intelligent answers" with zero crashes and maximum reliability.

## 🏗️ Project Structure

### **Production Architecture**
```
C:\Users\THE\open-source-rag-system\
├── core/                      # Modular FastAPI application
│   ├── main.py               # FastAPI entry point
│   ├── ollama_client.py      # Ollama LLM client with robust error handling
│   ├── routers/              # API endpoints
│   │   ├── query.py          # Single AI-only query endpoint
│   │   ├── documents.py      # Document management
│   │   ├── system.py         # System status
│   │   └── llm.py           # LLM management
│   ├── services/             # Business logic
│   │   ├── simple_rag_service.py  # Core RAG processing
│   │   ├── document_service.py    # Document processing
│   │   └── query_service.py       # Query processing
│   ├── repositories/         # Data access layer
│   └── di/                  # Dependency injection
├── run_core.py              # Production startup script
├── requirements.txt         # Python dependencies
├── static/index.html        # Web interface (AI answers only)
├── data/                    # SQLite databases and document storage
│   ├── rag_database.db      # Main database
│   ├── audit.db            # Audit logging
│   └── storage/            # Document storage (uploads/, processed/)
├── config/llm_config.yaml  # LLM configuration
├── tests/                   # Comprehensive test suite
├── .archive/               # Legacy files (safely archived)
└── CLAUDE.md              # This file (AI assistant instructions)
```

## 🧠 AI Assistant Guidelines

### **Core Principles When Working on This Project**

1. **Reliability First**: Zero crashes under normal use - every error must be handled gracefully
2. **MVP Focus**: Perfect core functionality before adding complexity
3. **User Experience**: 5-minute setup time for non-technical users
4. **Graceful Degradation**: System works even when components fail (e.g., Ollama unavailable)
5. **Security Minded**: Validate inputs, sanitize data, prevent injection attacks

### **Development Standards**

#### **Error Handling Philosophy**
- **Never crash**: Every exception must be caught and handled gracefully
- **Fail informatively**: Clear error messages that help users understand and fix issues
- **Cleanup on failure**: Remove partial files, reset state on errors
- **Retry with backoff**: Temporary failures should be retried intelligently
- **Fallback mechanisms**: Always provide alternative when primary method fails

#### **Code Quality Requirements**
- Input validation for all user data
- Rate limiting on all endpoints
- Comprehensive logging with proper levels
- File size and type validation
- Request sanitization (XSS, injection prevention)
- Proper resource cleanup (files, connections)

#### **Testing Standards**
- All critical paths must have error handling tests
- Test with malformed inputs, oversized files, network failures
- Verify fallback mechanisms work correctly
- Test concurrent usage scenarios
- Memory usage should remain reasonable under load

### **Technology Stack**

#### **Backend Architecture**
- **Framework**: Modular FastAPI with dependency injection (`core/main.py`)
- **API Design**: Single AI-only endpoint (`/api/v1/query`) with zero-hallucination protection
- **LLM Integration**: Ollama client with retry logic and health checks (`core/ollama_client.py`)
- **RAG Engine**: SimpleRAGService with 4 environment variables for configuration
- **Vector Search**: FAISS with sentence transformers for high-performance similarity search
- **Document Processing**: PyPDF2, python-docx, pandas with robust error handling
- **Storage**: SQLite databases (`data/rag_database.db`, `data/audit.db`) with file system storage
- **Architecture**: Clean Architecture with repositories, services, and dependency injection

#### **Frontend**
- **Interface**: Single-page HTML with vanilla JavaScript (`static/index.html`)
- **Features**: **AI answers only** - single mode with professional zero-hallucination responses
- **UX**: Source citations, confidence indicators, document download links, system status
- **Design**: Clean, professional interface optimized for production use

#### **Security & Reliability**
- **Input Validation**: Filename sanitization, content type checking
- **Rate Limiting**: Per-endpoint limits to prevent abuse
- **File Security**: Size limits, extension validation, malware prevention
- **Error Recovery**: Graceful degradation, automatic retries, cleanup

#### **Monitoring & Error Tracking**
- **Sentry MCP**: AI-powered error monitoring and root cause analysis
  - Install: `claude mcp add --transport http sentry https://mcp.sentry.dev/mcp`
  - OAuth authentication with Sentry organization
  - Real-time error tracking and performance monitoring
  - Automated root cause analysis with Seer AI agent
  - Full trace context for debugging (errors, logs, spans)
  - Integration with existing development workflows

## 📋 Current Development Status

### **✅ Production-Ready Features**
- ✅ **Modular Architecture**: Clean FastAPI application with dependency injection
- ✅ **Single AI-Only Endpoint**: `/api/v1/query` with zero-hallucination protection
- ✅ **Zero-Hallucination System**: Professional RAG with source citations and confidence scoring
- ✅ **SimpleRAGService**: Clean, maintainable RAG engine with 4 environment variables
- ✅ **Fine-tuned arlesheim-german model** for municipality-specific responses
- ✅ **Document download links** with secure `/api/v1/documents/{id}/download` endpoint
- ✅ **Production Frontend**: Single-mode interface with AI answers only
- ✅ **SQLite Storage**: Persistent databases with FAISS vector search
- ✅ **Repository Organization**: Clean codebase with legacy files safely archived
- ✅ **Comprehensive Testing**: Test suite with production validation
- ✅ **Error Handling**: Graceful failure handling with proper cleanup
- ✅ **Security Features**: Input validation, rate limiting, XSS prevention

### **🔄 Performance Optimization**
- ⏳ **Response Time**: AI generation speed optimization for local machines
- ⏳ **Caching**: Query result caching for repeated questions
- ⏳ **Memory Management**: Optimize memory usage for large document sets

### **📋 Future Enhancements** (Post-MVP)
1. **Authentication**: Optional user authentication system
2. **Advanced Monitoring**: Detailed performance metrics and alerts
3. **Multi-Model Support**: Additional LLM model options
4. **API Extensions**: Additional endpoints for specific use cases
5. **Deployment Tools**: Docker containers and cloud deployment scripts

## 🎯 Success Criteria for MVP

### **Reliability (Critical)**
- ✅ Zero crashes under normal usage
- ✅ Graceful handling of all error conditions
- ✅ Proper resource cleanup on failures
- ✅ Fallback mechanisms when services unavailable

### **Usability (Critical)**
- 🔄 60-second setup time for new users
- ✅ Clear error messages and recovery instructions
- ✅ Works smoothly with 1-1000 documents
- 🔄 Helpful onboarding and example queries

### **Security (Important)**
- ✅ Input validation and sanitization
- ✅ File upload security (size, type, content validation)
- ✅ Rate limiting to prevent abuse
- 🔄 Optional authentication for production use

### **Performance (Important)**
- ✅ Response times <10s for all operations
- 🔄 Memory usage stays reasonable with large document sets
- 🔄 Caching for repeated queries
- ✅ Handles concurrent users properly

## 🔧 Common Development Tasks

### **Running the Production System**
```bash
# Start the production RAG system
python run_core.py

# Test the system
python test_simple_rag.py
python test_ollama_integration.py
python -m pytest tests/

# Check system status
curl http://localhost:8000/api/v1/status
curl http://localhost:8000/api/v1/health
```

### **Key Configuration Settings**
```python
# SimpleRAGService Environment Variables (Production)
RAG_SIMILARITY_THRESHOLD=0.3    # Similarity threshold for document matching
RAG_MAX_RESULTS=5               # Maximum number of source documents
RAG_REQUIRE_SOURCES=true        # Require source citations (zero-hallucination)
RAG_MAX_QUERY_LENGTH=500        # Maximum query length for validation

# LLM Configuration (config/llm_config.yaml)
default_model: arlesheim-german  # Fine-tuned model for municipality
timeout: 300                     # LLM timeout in seconds
max_retries: 3                  # Retry attempts for failed requests

# Database Configuration (automatic)
database_path: data/rag_database.db    # Main SQLite database
audit_database: data/audit.db          # Audit logging database
storage_path: data/storage/             # Document storage directory
```

### **Error Handling Patterns**
```python
# Always use this pattern for operations that might fail
try:
    # Main operation
    result = risky_operation()
    
    # Validate result
    if not result:
        raise ValueError("Operation failed")
        
    return result
    
except SpecificException as e:
    # Log specific error
    logger.error(f"Specific error: {e}")
    # Cleanup if needed
    cleanup_resources()
    # Return graceful fallback or raise HTTPException
    
except Exception as e:
    # Log unexpected error
    logger.error(f"Unexpected error: {e}")
    # Cleanup if needed
    cleanup_resources()
    # Always provide user-friendly error message
    raise HTTPException(status_code=500, detail="Operation failed, please try again")
```

## 🚨 Critical Guidelines

### **When Adding New Features**
1. **Error Handling First**: Plan failure modes before implementing
2. **Input Validation**: Sanitize and validate all inputs
3. **Resource Management**: Ensure proper cleanup on success and failure
4. **Testing**: Test failure scenarios, not just happy path
5. **Documentation**: Update error messages and troubleshooting docs

### **When Debugging Issues**
1. **Check Logs**: Review error logs for patterns
2. **Verify Dependencies**: Ensure Ollama, models, files are available
3. **Test Isolation**: Isolate issues to specific components
4. **Reproduce**: Create minimal reproduction case
5. **Document**: Add prevention for similar issues

### **MVP Focus Areas**
- **Fix before feature**: Resolve reliability issues before adding features
- **User experience**: Prioritize clear error messages and recovery
- **Documentation**: Focus on setup and troubleshooting guides
- **Testing**: Validate edge cases and error conditions
- **Performance**: Ensure system scales to 100+ documents

## 🏛️ Fine-tuned Arlesheim Model

### **Model Overview**
The system now uses a **fine-tuned arlesheim-german model** optimized for municipality-specific responses in German.

### **Model Features**
- **Professional German responses** (no fake Swiss German)
- **Document download links** instead of full text dumps
- **Municipality-specific knowledge** for Arlesheim
- **Consistent formatting** with download references

### **Model Management**
```bash
# Rebuild the model after changes
cd training_data/arlesheim_german
ollama create arlesheim-german:latest -f Modelfile_final

# Test the model directly
ollama run arlesheim-german:latest "Frage über Dokumente"

# Check if model is available
ollama list | grep arlesheim
```

### **Model Configuration**
- **Base Model**: mistral:latest
- **Temperature**: 0.2 (consistent responses)
- **Context Length**: 32768 tokens
- **System Prompt**: Located in `Modelfile_final`

### **Response Format**
The model now returns:
1. **Brief content summary** (200 chars max)
2. **Download URL**: `/api/v1/documents/{document_id}/download`
3. **Professional German** without fake dialect

## 📞 Quick Reference

### **File Locations (Production)**
- **Main API**: `core/main.py` - FastAPI application entry point
- **Production Startup**: `run_core.py` - Production server startup script
- **LLM Client**: `core/ollama_client.py` - Ollama integration with retry logic
- **RAG Engine**: `core/services/simple_rag_service.py` - Core RAG processing logic
- **Query Router**: `core/routers/query.py` - Single `/api/v1/query` endpoint
- **Frontend**: `static/index.html` - Production web interface (AI answers only)
- **Tests**: `tests/` directory and `test_*.py` files in root
- **Storage**: `data/storage/uploads/` and `data/storage/processed/`
- **Databases**: `data/rag_database.db` and `data/audit.db`
- **Configuration**: `config/llm_config.yaml` - LLM model settings
- **Legacy Files**: `.archive/` directory - safely archived old code

### **Key API Endpoints**
- **Main Query**: `POST /api/v1/query` - Single AI-only endpoint with zero-hallucination
- **Document Upload**: `POST /api/v1/documents` - Upload and process documents
- **Document Download**: `GET /api/v1/documents/{id}/download` - Secure document download
- **System Status**: `GET /api/v1/status` - System health and configuration
- **Health Check**: `GET /api/v1/health` - Simple health check endpoint

### **Key Services & Functions**
- **RAG Processing**: `SimpleRAGService.answer_query()` - Main RAG processing logic
- **Document Management**: `DocumentService` in `core/services/document_service.py`
- **LLM Integration**: `OllamaClient.generate_answer()` - LLM response generation
- **Vector Search**: FAISS-based similarity search in repositories
- **Error Handling**: Comprehensive error handling throughout all services

### **Common Issues & Solutions**
- **System won't start**: Check `python run_core.py` - verify all dependencies installed
- **Ollama not available**: Check if Ollama service running, model downloaded
- **arlesheim-german model not found**: Rebuild with `ollama create arlesheim-german:latest -f data/training_data/arlesheim_german/Modelfile_final`
- **File upload fails**: Check file size limits, storage permissions, database connectivity
- **No search results**: Ensure documents uploaded, processed, and indexed in FAISS
- **Slow AI responses**: Performance optimization needed - consider model quantization or caching
- **Download links not working**: Verify document exists in database and `/api/v1/documents/{id}/download` endpoint
- **Database errors**: Check SQLite database files in `data/` directory are accessible
- **Legacy code confusion**: Old code is safely archived in `.archive/` - use production code in `core/`

---

**Remember**: This is now a production-ready system with modular architecture. The core mission remains: reliable RAG system with zero-hallucination protection and professional AI answers.

**Last Updated**: January 2025 
**Version**: 2.0-Production
**Focus**: Production-ready modular architecture with single AI-only endpoint and organized codebase