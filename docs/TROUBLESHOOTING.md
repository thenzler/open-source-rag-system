# 🆘 Troubleshooting Guide

Complete troubleshooting guide for the Open Source RAG System.

## 🔍 Quick Diagnostics

### System Health Check

Run this command to check system status:

```bash
# Check API status
curl http://localhost:8001/api/status

# Test optimized endpoint
python test_widget_endpoint.py

# Check widget functionality
python test_widget_server.py
```

### Common Quick Fixes

1. **Restart the API server**: `python simple_api.py`
2. **Clear browser cache**: Ctrl+F5 or Cmd+Shift+R
3. **Check Ollama status**: `ollama list`
4. **Verify documents uploaded**: Check `/api/v1/documents`

## 🐛 Common Issues

### 1. Server Won't Start

#### Symptoms
```bash
$ python simple_api.py
Error: Address already in use
```

#### Causes & Solutions

**Port 8001 already in use:**
```bash
# Find process using port 8001
netstat -ano | findstr :8001  # Windows
lsof -i :8001                 # macOS/Linux

# Kill the process
taskkill /PID <pid> /F        # Windows
kill -9 <pid>                # macOS/Linux

# Or use different port
python simple_api.py --port 8002
```

**Missing dependencies:**
```bash
# Install missing packages
pip install -r simple_requirements.txt

# Install specific packages
pip install fastapi uvicorn sentence-transformers
```

**Python version issues:**
```bash
# Check Python version (requires 3.8+)
python --version

# Use specific Python version
python3.9 simple_api.py
```

### 2. Ollama Connection Issues

#### Symptoms
```
ERROR:ollama_client:Request timeout on attempt 1
WARNING:simple_api:LLM generation failed, falling back to vector search
```

#### Diagnosis
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Check available models
ollama list

# Test model directly
ollama run mistral:latest "Hello"
```

#### Solutions

**Ollama not installed:**
```bash
# Install Ollama
# Visit: https://ollama.com/download
# Windows: Download installer
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh
```

**Ollama not running:**
```bash
# Start Ollama service
ollama serve

# Or start as background service (Linux/macOS)
sudo systemctl start ollama
```

**No models installed:**
```bash
# Install fast models
ollama pull phi3-mini
ollama pull llama3.2:1b
ollama pull llama3.2:3b

# Verify installation
ollama list
```

**Model too slow:**
```bash
# Switch to faster model
ollama pull phi3-mini

# Configure in simple_api.py
PREFERRED_MODEL = "phi3-mini:latest"
```

### 3. Document Upload Issues

#### Symptoms
```
Error uploading document.pdf: Unsupported file type
```

#### Supported File Types
- **PDF**: `.pdf`
- **Word**: `.docx` (not `.doc`)
- **Text**: `.txt`
- **Markdown**: `.md`

#### Solutions

**Unsupported file format:**
```python
# Convert to supported format
# DOC → DOCX: Open in Word, Save As DOCX
# RTF → TXT: Open in text editor, Save As TXT
# HTML → MD: Use pandoc conversion
```

**File too large:**
```python
# Check file size limit (default: 50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Reduce file size:
# - Compress PDF
# - Remove images from DOCX
# - Split large documents
```

**Corrupted file:**
```bash
# Test file integrity
file document.pdf              # Linux/macOS
Get-ItemProperty document.pdf  # Windows PowerShell

# Try opening in native application
# PDF: Adobe Reader, browser
# DOCX: Microsoft Word
```

**Processing failed:**
```python
# Check logs for specific error
tail -f logs/rag_system.log

# Common issues:
# - Encrypted/password-protected PDFs
# - Corrupted document structure
# - Non-text content (images only)
```

### 4. No Search Results

#### Symptoms
```
No results found for your query
No relevant information found
```

#### Diagnosis
```bash
# Check if documents are uploaded
curl http://localhost:8001/api/v1/documents

# Check document processing status
curl http://localhost:8001/api/status
```

#### Solutions

**No documents uploaded:**
```bash
# Upload test document
curl -X POST "http://localhost:8001/api/v1/documents" \
  -F "file=@test_document.pdf"
```

**Documents not processed:**
```python
# Check processing status in logs
# Look for: "Document processed successfully"
# Or: "Error processing document"

# Common processing issues:
# - Empty document
# - Scanned PDF (no text)
# - Encrypted document
```

**Query too specific:**
```python
# Try broader queries
# Instead of: "What is the exact temperature coefficient?"
# Try: "temperature" or "coefficient"

# Check similarity threshold
MIN_SIMILARITY = 0.3  # Lower = more results
```

**Embedding model issue:**
```python
# Check embedding model status
if embedding_model is None:
    print("Embedding model not loaded")
    
# Reinstall sentence-transformers
pip uninstall sentence-transformers
pip install sentence-transformers
```

### 5. Widget Not Working

#### Symptoms
- Widget doesn't appear
- "Sorry, I encountered an error" messages
- CORS errors in browser console

#### Diagnosis
```javascript
// Open browser console (F12)
// Check for errors:
console.log('Widget loaded:', !!window.ragWidget);

// Test API directly
fetch('http://localhost:8001/api/status')
  .then(r => r.json())
  .then(console.log)
  .catch(console.error);
```

#### Solutions

**Widget not appearing:**
```html
<!-- Check script path -->
<script src="widget-loader.js"></script>  <!-- ✓ Correct -->
<script src="/widget-loader.js"></script> <!-- ✗ May fail -->

<!-- Check for JavaScript errors -->
<script>
console.log('Script loaded');
</script>
```

**API connection failed:**
```html
<!-- Verify API URL -->
data-api-url="http://localhost:8001"  <!-- ✓ Correct port -->
data-api-url="http://localhost:8000"  <!-- ✗ Wrong port -->

<!-- Test with curl -->
curl http://localhost:8001/api/status
```

**CORS errors:**
```bash
# Use test server to avoid CORS
python test_widget_server.py
# Then visit: http://localhost:3000/widget/

# Or configure CORS headers
# (Already configured in simple_api.py)
```

**Widget styling broken:**
```css
/* Fix z-index conflicts */
.rag-widget-container {
    z-index: 999999 !important;
}

/* Fix positioning */
.rag-chat-window {
    position: fixed !important;
}
```

### 6. Slow Performance

#### Symptoms
- Responses take >30 seconds
- UI freezes during queries
- High CPU/memory usage

#### Diagnosis
```python
# Check performance metrics
curl http://localhost:8001/api/v1/performance

# Monitor system resources
top    # Linux/macOS
taskmgr # Windows

# Check cache hit rates
curl http://localhost:8001/api/status | grep cache
```

#### Solutions

**Switch to faster model:**
```bash
# Install fast models
ollama pull phi3-mini
ollama pull llama3.2:1b

# Verify in API
curl http://localhost:8001/api/status | grep model
```

**Optimize configuration:**
```python
# In simple_api.py
TIMEOUT = 10           # Reduced timeout
CONTEXT_LIMIT = 3      # Fewer chunks
MAX_TOKENS = 200       # Shorter responses
CACHE_SIZE = 1000      # Enable caching
```

**Reduce document size:**
```python
# Check document statistics
curl http://localhost:8001/api/status

# If total_chunks > 10,000:
# - Remove unnecessary documents
# - Split large documents
# - Increase chunk size
```

**Memory optimization:**
```python
# Restart server periodically
# Clear caches
# Use memory profiling tools
```

## 🔧 Debugging Tools

### 1. API Testing

#### Direct API Test
```bash
# Test status endpoint
curl http://localhost:8001/api/status

# Test document list
curl http://localhost:8001/api/v1/documents

# Test optimized query
curl -X POST "http://localhost:8001/api/v1/query/optimized" \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "context_limit": 3}'
```

#### Python Test Script
```python
#!/usr/bin/env python3
import requests
import json

def test_rag_api():
    """Test RAG API endpoints"""
    
    base_url = "http://localhost:8001"
    
    # Test status
    try:
        response = requests.get(f"{base_url}/api/status")
        print(f"Status: {response.status_code}")
        print(f"Data: {response.json()}")
    except Exception as e:
        print(f"Status test failed: {e}")
    
    # Test query
    try:
        response = requests.post(
            f"{base_url}/api/v1/query/optimized",
            json={"query": "test", "context_limit": 3}
        )
        print(f"Query: {response.status_code}")
        if response.ok:
            data = response.json()
            print(f"Response: {data.get('response', 'No response')}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Query test failed: {e}")

if __name__ == "__main__":
    test_rag_api()
```

### 2. Widget Testing

#### Standalone Widget Test
```html
<!-- Save as test_widget.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Widget Test</title>
</head>
<body>
    <h1>Widget Test Page</h1>
    
    <!-- Widget integration -->
    <script src="widget-loader.js" 
            data-api-key="test-key"
            data-api-url="http://localhost:8001"
            data-debug="true">
    </script>
    
    <!-- Debug script -->
    <script>
    setTimeout(() => {
        console.log('Widget status:', !!window.ragWidget);
        console.log('Widget config:', window.ragWidgetConfig);
    }, 2000);
    </script>
</body>
</html>
```

#### Browser Console Tests
```javascript
// Test widget API
window.ragWidget.sendMessage("test");

// Check widget state
console.log('Widget open:', window.ragWidget.isOpen());

// Test API endpoint
fetch('http://localhost:8001/api/v1/query/optimized', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query: 'test'})
})
.then(r => r.json())
.then(console.log)
.catch(console.error);
```

### 3. Log Analysis

#### Enable Debug Logging
```python
# In simple_api.py
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export RAG_LOG_LEVEL=DEBUG
```

#### Common Log Patterns
```bash
# Search for errors
grep "ERROR" logs/rag_system.log

# Check processing times
grep "Processing time" logs/rag_system.log

# Monitor cache performance
grep "cache" logs/rag_system.log

# Track API requests
grep "POST\|GET" logs/rag_system.log
```

### 4. Performance Profiling

#### Memory Profiling
```python
import tracemalloc
import psutil

def profile_memory():
    """Profile memory usage"""
    
    # Start tracing
    tracemalloc.start()
    
    # Run your code here
    query_documents("test query")
    
    # Get memory stats
    current, peak = tracemalloc.get_traced_memory()
    process = psutil.Process()
    memory_info = process.memory_info()
    
    print(f"Current memory: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
    print(f"RSS: {memory_info.rss / 1024 / 1024:.1f} MB")
    
    tracemalloc.stop()
```

#### CPU Profiling
```python
import cProfile
import pstats

def profile_cpu():
    """Profile CPU usage"""
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run your code
    query_documents("test query")
    
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

## 🔍 Error Code Reference

### HTTP Status Codes

| Code | Meaning | Common Causes | Solutions |
|------|---------|---------------|-----------|
| 200 | Success | - | Request completed successfully |
| 400 | Bad Request | Invalid JSON, missing fields | Check request format |
| 404 | Not Found | Wrong endpoint, document not found | Verify URL path |
| 422 | Validation Error | Invalid field values | Check Pydantic model |
| 429 | Rate Limited | Too many requests | Wait and retry |
| 500 | Server Error | Python exception, system error | Check logs |

### RAG-Specific Errors

| Error Message | Cause | Solution |
|---------------|-------|----------|
| "Embedding model not available" | SentenceTransformers not installed | `pip install sentence-transformers` |
| "Ollama not available" | Ollama not running | `ollama serve` |
| "No documents uploaded" | No documents in system | Upload documents via frontend |
| "Query cannot be empty" | Empty query string | Provide non-empty query |
| "File too large" | File > 50MB | Reduce file size |
| "Unsupported file type" | Wrong file extension | Use PDF, DOCX, TXT, MD |
| "Processing failed" | Document processing error | Check file integrity |

### Widget Error Messages

| Error Message | Cause | Solution |
|---------------|-------|----------|
| "Cannot connect to server" | API server down | Start `python simple_api.py` |
| "API endpoint not found" | Wrong URL | Check `data-api-url` |
| "Cross-Origin Request Blocked" | CORS issue | Use test server |
| "Invalid response format" | API returned wrong format | Check API endpoint |
| "Network timeout" | Slow connection | Increase timeout |

## 🛠️ System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Network**: Internet for model downloads

### Recommended Requirements
- **Python**: 3.9+
- **RAM**: 8GB+
- **CPU**: 4+ cores
- **Storage**: 10GB+ SSD
- **GPU**: Optional (for faster inference)

### Dependency Versions
```bash
# Core dependencies
fastapi>=0.104.0
uvicorn>=0.24.0
sentence-transformers>=2.2.0
numpy>=1.21.0
scikit-learn>=1.0.0

# Optional dependencies
PyPDF2>=3.0.0        # PDF processing
python-docx>=0.8.11  # DOCX processing
pandas>=1.3.0        # CSV processing
```

## 📞 Getting Help

### 1. Self-Diagnosis

Before seeking help, try these steps:

1. **Check system status**: `curl http://localhost:8001/api/status`
2. **Review logs**: Look for ERROR messages
3. **Test basic functionality**: Upload document, run query
4. **Verify configuration**: Check ports, file paths
5. **Restart services**: API server, Ollama

### 2. Information to Collect

When reporting issues, include:

#### System Information
```bash
# Operating system
uname -a                    # Linux/macOS
systeminfo                 # Windows

# Python version
python --version

# Installed packages
pip list | grep -E "(fastapi|sentence|ollama)"
```

#### Error Details
```bash
# Full error message
# Stack trace if available
# Steps to reproduce
# Expected vs actual behavior
```

#### Configuration
```bash
# API server logs
# Browser console errors (F12)
# Network requests (DevTools → Network)
# Configuration files
```

### 3. Common Support Channels

- **Documentation**: Check README and docs/ folder
- **GitHub Issues**: Create detailed issue report
- **Stack Overflow**: Tag with `rag-system`
- **Discord/Slack**: Real-time community help

### 4. Emergency Recovery

#### Complete System Reset
```bash
# Stop all services
pkill -f "simple_api.py"
pkill -f "ollama"

# Clear temporary data
rm -rf storage/processed/*
rm -rf logs/*
rm -rf __pycache__/*

# Restart services
python simple_api.py
```

#### Factory Reset
```bash
# Backup important data
cp -r storage/uploads backup_uploads/

# Remove all generated data
rm -rf storage/
rm -rf logs/
rm -rf models/

# Reinstall dependencies
pip uninstall -r simple_requirements.txt -y
pip install -r simple_requirements.txt

# Restart system
python simple_api.py
```

## 📋 Maintenance Checklist

### Daily Maintenance
- [ ] Check API server status
- [ ] Monitor log files for errors
- [ ] Verify Ollama models available
- [ ] Test widget functionality

### Weekly Maintenance
- [ ] Review performance metrics
- [ ] Clean up old log files
- [ ] Update Ollama models
- [ ] Check disk space usage

### Monthly Maintenance
- [ ] Update dependencies
- [ ] Review and optimize cache settings
- [ ] Performance tuning
- [ ] Backup configuration and data

### Quarterly Maintenance
- [ ] System security review
- [ ] Performance benchmarking
- [ ] Documentation updates
- [ ] Feature planning

---

## 📝 Summary

This troubleshooting guide covers the most common issues you'll encounter with the RAG system. Remember:

1. **Start with basics**: Check server status, logs, configuration
2. **Use diagnostic tools**: Built-in tests, browser console, API tests
3. **Follow systematic approach**: Isolate the problem, test solutions
4. **Document solutions**: Keep notes for future reference

Most issues can be resolved quickly with the right diagnostic approach. When in doubt, check the logs! 🔍

---

*Need more help? Check our [GitHub Issues](https://github.com/your-repo/issues) or [Documentation](../README.md)*