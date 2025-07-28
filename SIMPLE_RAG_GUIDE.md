# Simple Professional RAG System

Clean, maintainable RAG system with **AI answers only**.

## 🎯 Design Principles

- **Simple**: 4 environment variables, not 200 YAML lines
- **Predictable**: Clear threshold-based logic anyone can understand  
- **Maintainable**: One mode, one flow, easy debugging
- **Professional**: Standard deployment practices

## 🚀 Quick Start

### 1. Configuration (Environment Variables)
```bash
# Core settings - adjust for your domain
export RAG_SIMILARITY_THRESHOLD=0.3    # Lower = more results, higher = stricter
export RAG_MAX_RESULTS=5               # Number of source documents  
export RAG_REQUIRE_SOURCES=true        # Include source citations
export RAG_MAX_QUERY_LENGTH=500        # Input validation
```

### 2. Start the Service
```bash
python -m uvicorn simple_api:app --host 0.0.0.0 --port 8000
```

### 3. Use the API
```bash
# Ask a question
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "How does this work?"}'

# Check status  
curl "http://localhost:8000/api/v1/rag/status"

# Health check
curl "http://localhost:8000/api/v1/rag/health"
```

## 📊 How It Works

### Simple Flow
```
Query → Search Documents → Filter by Threshold → Generate AI Answer → Return with Sources
```

### Core Logic
```python
def should_answer(similarity_score: float, threshold: float) -> bool:
    return similarity_score >= threshold

# That's it. No complex rules, no domain-specific logic.
```

## 🔧 Configuration Guide

### For Different Domains

**Legal Documents (High Precision)**
```bash
export RAG_SIMILARITY_THRESHOLD=0.5
export RAG_MAX_RESULTS=3
```

**General Knowledge (Broad Coverage)**  
```bash
export RAG_SIMILARITY_THRESHOLD=0.2
export RAG_MAX_RESULTS=10
```

**Technical Documentation (Balanced)**
```bash
export RAG_SIMILARITY_THRESHOLD=0.3
export RAG_MAX_RESULTS=5
```

## 📝 API Reference

### POST /api/v1/rag/query
Ask a question and get an AI answer with sources.

**Request:**
```json
{
  "query": "Your question here"
}
```

**Response:**
```json
{
  "answer": "AI-generated answer with [Quelle 1] citations",
  "sources": [
    {
      "id": 1,
      "document_id": 123,
      "similarity": 0.85,
      "download_url": "/api/v1/documents/123/download"
    }
  ],
  "confidence": 0.85,
  "timestamp": "2024-01-01T12:00:00Z",
  "query": "Your question here"
}
```

### GET /api/v1/rag/status
Get current configuration and service status.

**Response:**
```json
{
  "service": "Simple RAG Service",
  "mode": "AI answers only", 
  "config": {
    "similarity_threshold": 0.3,
    "max_results": 5,
    "require_sources": true,
    "max_query_length": 500
  },
  "healthy": true
}
```

## 🐛 Debugging

### No Results?
1. **Check threshold**: Lower `RAG_SIMILARITY_THRESHOLD`
2. **Check documents**: Ensure documents are uploaded and indexed
3. **Check query**: Try different wording

### Too Many Irrelevant Results?
1. **Raise threshold**: Increase `RAG_SIMILARITY_THRESHOLD` 
2. **Reduce results**: Lower `RAG_MAX_RESULTS`

### Service Issues?
1. **Check health**: `GET /api/v1/rag/health`
2. **Check status**: `GET /api/v1/rag/status` 
3. **Check logs**: Standard application logs

## 🏗️ Architecture 

### Components
- **SimpleRAGService**: Core business logic (150 lines)
- **RAGConfig**: Environment variable configuration (20 lines)
- **Simple API**: REST endpoints (50 lines)

### Dependencies
- Vector search repository
- LLM client (Ollama)
- Optional audit logging

### Total Complexity
- **~220 lines of code** vs 2000+ in complex systems
- **4 environment variables** vs dozens of config files
- **1 mode** vs multiple search strategies
- **Clear logic** vs complex decision trees

## 🎯 Production Benefits

### Deployment
- ✅ Standard 12-factor app configuration
- ✅ Docker-friendly environment variables  
- ✅ Health checks and status endpoints
- ✅ Structured logging

### Maintenance  
- ✅ Easy to understand and debug
- ✅ Predictable behavior
- ✅ Simple performance tuning
- ✅ Clear error messages

### Scaling
- ✅ Stateless service design
- ✅ Configurable performance parameters
- ✅ Standard monitoring patterns
- ✅ Easy horizontal scaling

## 💡 Best Practices

### Tuning Guidelines
1. **Start with threshold 0.3** for most domains
2. **Adjust based on precision/recall needs**
3. **Monitor query success rates**
4. **Use A/B testing for threshold optimization**

### Monitoring
- Track query success rates
- Monitor response times  
- Log similarity scores
- Alert on error rates

### Security
- Input validation (query length, content)
- Rate limiting (implement at API gateway)
- Source attribution (prevent hallucination)
- Audit logging (optional but recommended)

---

**This is what a professional RAG system looks like: Simple, predictable, maintainable.** 🎉