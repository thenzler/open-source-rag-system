# CLAUDE.md - RAG System Project Settings

## 🎯 Project Overview

**Open Source RAG System** - A complete locally-hosted Retrieval-Augmented Generation system with Ollama LLM integration. This is an MVP-focused project prioritizing reliability, ease of use, and production readiness over advanced features.

**Core Mission**: Create a bulletproof RAG system that works perfectly for the core use case: "Upload documents → Ask questions → Get intelligent answers" with zero crashes and maximum reliability.

## 🏗️ Project Structure

### **Primary Directory**
```
C:\Users\THE\open-source-rag-system\
├── simple_api.py              # Main FastAPI server with LLM integration
├── ollama_client.py           # Ollama LLM client with robust error handling
├── simple_requirements.txt    # Python dependencies
├── simple_frontend.html       # Web interface with AI/vector search modes
├── start_simple_rag.py       # Startup script
├── test_simple_rag.py        # Basic tests
├── test_ollama_integration.py # Comprehensive LLM tests
├── storage/                  # Document storage (uploads/, processed/)
├── SIMPLE_RAG_README.md     # User documentation
└── CLAUDE.md                # This file (AI assistant instructions)
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

#### **Backend**
- **Framework**: FastAPI with comprehensive error handling
- **LLM Integration**: Ollama client with retry logic and health checks
- **Vector Search**: Sentence Transformers with cosine similarity
- **Document Processing**: PyPDF2, python-docx, pandas
- **Storage**: Local filesystem with proper validation

#### **Frontend**
- **Interface**: Single-page HTML with vanilla JavaScript
- **Features**: AI/vector search modes, document management, system status
- **UX**: Progress indicators, helpful error messages, clear feedback

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

### **✅ Completed Features**
- Core RAG functionality with document upload and querying
- Ollama LLM integration with fallback to vector search
- Comprehensive file upload validation and size limits
- Request validation and sanitization (XSS/injection prevention)
- Rate limiting on all endpoints
- Graceful error handling with cleanup
- Web interface with AI/vector search modes
- Robust Ollama connection handling with retries

### **🔄 In Progress**
- Startup dependency checks
- One-click setup script creation
- Advanced security measures
- Performance optimizations

### **📋 Remaining MVP Tasks**
1. **Startup Checks**: Verify all dependencies on startup
2. **Setup Script**: One-click installation and configuration
3. **Security Hardening**: Optional authentication, CORS configuration
4. **Performance**: Caching, memory optimization
5. **Documentation**: Troubleshooting guides, setup instructions
6. **Testing**: Comprehensive validation under various conditions

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

### **Running the System**
```bash
# Start the RAG system
python simple_api.py

# Or use startup script
python start_simple_rag.py

# Test the system
python test_simple_rag.py
python test_ollama_integration.py
```

### **Key Configuration Settings**
```python
# In simple_api.py
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_TOTAL_DOCUMENTS = 1000
USE_LLM_DEFAULT = True
MAX_CONTEXT_LENGTH = 4000

# Rate limiting
RATE_LIMITS = {
    "upload": 10,  # per minute
    "query": 60,   # per minute  
    "status": 20   # per minute
}
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

## 📞 Quick Reference

### **File Locations**
- Main API: `simple_api.py`
- LLM Client: `ollama_client.py`
- Frontend: `simple_frontend.html`
- Tests: `test_*.py`
- Storage: `storage/uploads/` and `storage/processed/`

### **Key Functions**
- Document upload: `upload_document()` in simple_api.py
- Query processing: `query_documents_enhanced()` in simple_api.py
- LLM integration: `generate_answer()` in ollama_client.py
- Error handling: Various validation functions throughout

### **Common Issues & Solutions**
- **Ollama not available**: Check if service running, model downloaded
- **File upload fails**: Verify file size, type, permissions
- **No search results**: Ensure documents uploaded and processed
- **Memory issues**: Check document count, restart if needed

---

**Remember**: This is an MVP focused on reliability. Every error is a lost user. Make it work perfectly for the core use case before adding complexity.

**Last Updated**: December 2024
**Version**: 1.0-MVP
**Focus**: Production readiness and zero-crash reliability