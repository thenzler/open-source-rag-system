# API Documentation

## Overview

The Open Source RAG System provides a comprehensive RESTful API built with FastAPI, offering automatic OpenAPI documentation, type validation, and high performance. All endpoints return structured JSON responses with consistent error handling.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

The API uses JWT-based authentication for protected endpoints.

### Authentication Flow

```bash
# 1. Login to get access token
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "password"}'

# Response
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}

# 2. Use token in subsequent requests
curl -X GET "http://localhost:8000/api/v1/documents" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

## Core Endpoints

### Document Management

#### Upload Document
Upload a new document for processing.

```http
POST /api/v1/documents
```

**Request**:
```bash
curl -X POST "http://localhost:8000/api/v1/documents" \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "metadata={\"category\": \"research\", \"tags\": [\"AI\", \"ML\"]}"
```

**Parameters**:
- `file` (required): Document file (PDF, DOCX, XLSX, XML, TXT)
- `metadata` (optional): JSON string with additional metadata

**Response**:
```json
{
  "document_id": "123e4567-e89b-12d3-a456-426614174000",
  "filename": "document.pdf",
  "status": "processing",
  "upload_timestamp": "2025-07-04T09:30:00Z",
  "file_size": 2048576,
  "mime_type": "application/pdf",
  "metadata": {
    "category": "research",
    "tags": ["AI", "ML"]
  }
}
```

#### Get Document Status
Check the processing status of a document.

```http
GET /api/v1/documents/{document_id}/status
```

**Response**:
```json
{
  "document_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "progress": 100,
  "processed_chunks": 45,
  "processing_started": "2025-07-04T09:30:05Z",
  "processing_completed": "2025-07-04T09:31:23Z",
  "errors": []
}
```

**Status Values**:
- `pending`: Queued for processing
- `processing`: Currently being processed
- `completed`: Successfully processed
- `failed`: Processing failed
- `cancelled`: Processing was cancelled

#### List Documents
Get a paginated list of all documents.

```http
GET /api/v1/documents
```

**Query Parameters**:
- `skip` (optional): Number of documents to skip (default: 0)
- `limit` (optional): Maximum number of documents to return (default: 50, max: 100)
- `status` (optional): Filter by status
- `category` (optional): Filter by metadata category
- `search` (optional): Search in filename and metadata

**Response**:
```json
{
  "documents": [
    {
      "document_id": "123e4567-e89b-12d3-a456-426614174000",
      "filename": "document.pdf",
      "status": "completed",
      "upload_timestamp": "2025-07-04T09:30:00Z",
      "file_size": 2048576,
      "mime_type": "application/pdf",
      "chunk_count": 45,
      "metadata": {
        "category": "research",
        "tags": ["AI", "ML"]
      }
    }
  ],
  "total": 156,
  "skip": 0,
  "limit": 50
}
```

#### Get Document Details
Retrieve detailed information about a specific document.

```http
GET /api/v1/documents/{document_id}
```

**Response**:
```json
{
  "document_id": "123e4567-e89b-12d3-a456-426614174000",
  "filename": "document.pdf",
  "status": "completed",
  "upload_timestamp": "2025-07-04T09:30:00Z",
  "processing_completed": "2025-07-04T09:31:23Z",
  "file_size": 2048576,
  "mime_type": "application/pdf",
  "checksum": "sha256:abc123...",
  "chunk_count": 45,
  "metadata": {
    "category": "research",
    "tags": ["AI", "ML"],
    "author": "John Doe",
    "pages": 12
  },
  "processing_stats": {
    "extraction_time_ms": 5430,
    "chunking_time_ms": 1200,
    "embedding_time_ms": 12800
  }
}
```

#### Delete Document
Remove a document and all associated data.

```http
DELETE /api/v1/documents/{document_id}
```

**Response**:
```json
{
  "message": "Document deleted successfully",
  "document_id": "123e4567-e89b-12d3-a456-426614174000",
  "deleted_chunks": 45,
  "deleted_vectors": 45
}
```

### Query & Retrieval

#### Query Documents
Perform a semantic search across all documents.

```http
POST /api/v1/query
```

**Request**:
```json
{
  "query": "What are the main advantages of transformer models?",
  "top_k": 5,
  "min_score": 0.7,
  "filters": {
    "category": ["research", "technical"],
    "tags": ["AI", "ML"],
    "document_ids": ["123e4567-e89b-12d3-a456-426614174000"]
  },
  "include_metadata": true,
  "max_char_length": 1000
}
```

**Parameters**:
- `query` (required): The search query
- `top_k` (optional): Number of results to return (default: 5, max: 50)
- `min_score` (optional): Minimum similarity score (default: 0.0)
- `filters` (optional): Metadata filters to apply
- `include_metadata` (optional): Include document metadata in response
- `max_char_length` (optional): Maximum character length for snippets

**Response**:
```json
{
  "query": "What are the main advantages of transformer models?",
  "response": "Transformer models offer several key advantages: 1) Parallel processing capabilities that significantly speed up training, 2) Superior handling of long-range dependencies through self-attention mechanisms, and 3) Better performance on various NLP tasks compared to traditional RNN architectures.",
  "sources": [
    {
      "document_id": "123e4567-e89b-12d3-a456-426614174000",
      "filename": "transformer_architecture.pdf",
      "chunk_id": "chunk_12",
      "relevance_score": 0.91,
      "page_number": 3,
      "text_snippet": "The self-attention mechanism in transformers allows for parallel processing of sequences, unlike RNNs which process sequentially. This parallelization leads to significant speedups during training...",
      "start_char": 245,
      "end_char": 589,
      "metadata": {
        "category": "research",
        "tags": ["AI", "transformers"],
        "section": "Architecture Overview"
      }
    }
  ],
  "total_sources": 5,
  "confidence_score": 0.87,
  "processing_time_ms": 245
}
```

#### Advanced Query with Re-ranking
Perform a query with advanced re-ranking and filtering.

```http
POST /api/v1/query/advanced
```

**Request**:
```json
{
  "query": "machine learning optimization techniques",
  "retrieval_strategy": "hybrid",
  "rerank": true,
  "rerank_model": "cross-encoder",
  "expand_query": true,
  "semantic_threshold": 0.75,
  "keyword_boost": 0.3,
  "top_k": 10,
  "final_k": 5
}
```

**Response** (same structure as basic query with additional metrics):
```json
{
  "query": "machine learning optimization techniques",
  "expanded_query": "machine learning optimization techniques gradient descent adam optimizer learning rate",
  "response": "...",
  "sources": [...],
  "reranking_applied": true,
  "retrieval_metrics": {
    "initial_candidates": 50,
    "post_semantic_filter": 23,
    "post_reranking": 5,
    "semantic_search_time_ms": 89,
    "reranking_time_ms": 156
  }
}
```

#### Similar Documents
Find documents similar to a given document.

```http
GET /api/v1/documents/{document_id}/similar
```

**Query Parameters**:
- `top_k` (optional): Number of similar documents (default: 5)
- `min_score` (optional): Minimum similarity threshold

**Response**:
```json
{
  "source_document": {
    "document_id": "123e4567-e89b-12d3-a456-426614174000",
    "filename": "transformer_architecture.pdf"
  },
  "similar_documents": [
    {
      "document_id": "456e7890-e89b-12d3-a456-426614174001",
      "filename": "attention_mechanisms.pdf",
      "similarity_score": 0.89,
      "common_topics": ["transformers", "attention", "neural networks"],
      "metadata": {
        "category": "research",
        "tags": ["AI", "attention"]
      }
    }
  ]
}
```

### Analytics & Statistics

#### System Statistics
Get overall system statistics.

```http
GET /api/v1/analytics/stats
```

**Response**:
```json
{
  "documents": {
    "total": 1247,
    "processed": 1198,
    "processing": 12,
    "failed": 37,
    "total_size_bytes": 15728640000
  },
  "chunks": {
    "total": 54321,
    "average_per_document": 45.2,
    "average_length_chars": 512
  },
  "queries": {
    "total_today": 342,
    "total_this_week": 2105,
    "average_response_time_ms": 187,
    "most_common_topics": ["AI", "machine learning", "optimization"]
  },
  "storage": {
    "documents_size_bytes": 15728640000,
    "vector_index_size_bytes": 2145728000,
    "database_size_bytes": 524288000
  }
}
```

#### Query Analytics
Get analytics for queries over a time period.

```http
GET /api/v1/analytics/queries
```

**Query Parameters**:
- `start_date`: Start date (ISO format)
- `end_date`: End date (ISO format)
- `granularity`: Time granularity (hour, day, week)

**Response**:
```json
{
  "period": {
    "start": "2025-07-01T00:00:00Z",
    "end": "2025-07-04T00:00:00Z",
    "granularity": "day"
  },
  "metrics": [
    {
      "date": "2025-07-01",
      "query_count": 456,
      "average_response_time_ms": 198,
      "unique_users": 23,
      "top_queries": [
        "machine learning algorithms",
        "neural network architectures",
        "optimization techniques"
      ]
    }
  ],
  "summary": {
    "total_queries": 1834,
    "average_daily_queries": 458.5,
    "peak_hour": "14:00-15:00",
    "most_active_day": "2025-07-03"
  }
}
```

### System Management

#### Health Check
Check system health and service status.

```http
GET /api/v1/health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-07-04T09:45:00Z",
  "services": {
    "database": {
      "status": "healthy",
      "response_time_ms": 12,
      "connection_pool": {
        "active": 5,
        "idle": 15,
        "total": 20
      }
    },
    "vector_database": {
      "status": "healthy",
      "response_time_ms": 8,
      "collections": 3,
      "total_vectors": 54321
    },
    "llm_service": {
      "status": "healthy",
      "model": "llama3.1:8b",
      "response_time_ms": 234
    },
    "storage": {
      "status": "healthy",
      "available_space_gb": 512.7,
      "used_space_gb": 187.3
    }
  }
}
```

#### Configuration
Get or update system configuration.

```http
GET /api/v1/config
```

**Response**:
```json
{
  "embedding_model": "sentence-transformers/all-mpnet-base-v2",
  "llm_model": "llama3.1:8b",
  "chunk_size": 512,
  "chunk_overlap": 50,
  "similarity_threshold": 0.7,
  "max_query_length": 1000,
  "rate_limits": {
    "queries_per_minute": 60,
    "uploads_per_hour": 100
  },
  "features": {
    "query_expansion": true,
    "reranking": true,
    "caching": true
  }
}
```

## Error Handling

### HTTP Status Codes

- `200`: Success
- `201`: Created
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `413`: Payload Too Large
- `422`: Validation Error
- `429`: Rate Limit Exceeded
- `500`: Internal Server Error
- `503`: Service Unavailable

### Error Response Format

```json
{
  "error": {
    "code": "DOCUMENT_NOT_FOUND",
    "message": "Document with ID 123e4567-e89b-12d3-a456-426614174000 not found",
    "details": {
      "document_id": "123e4567-e89b-12d3-a456-426614174000",
      "timestamp": "2025-07-04T09:45:00Z"
    }
  }
}
```

### Common Error Codes

- `INVALID_FILE_FORMAT`: Unsupported file type
- `FILE_TOO_LARGE`: File exceeds size limit
- `DOCUMENT_NOT_FOUND`: Document doesn't exist
- `PROCESSING_FAILED`: Document processing error
- `INVALID_QUERY`: Query validation failed
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INSUFFICIENT_STORAGE`: Not enough disk space
- `SERVICE_UNAVAILABLE`: Dependent service offline

## Rate Limiting

The API implements rate limiting to ensure fair usage:

### Default Limits
- **Authentication**: 5 requests per minute per IP
- **Queries**: 60 requests per minute per user
- **Document uploads**: 100 requests per hour per user
- **Bulk operations**: 10 requests per minute per user

### Rate Limit Headers
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1625654400
X-RateLimit-Type: query
```

## SDK Examples

### Python SDK
```python
import requests
from typing import List, Dict, Any

class RAGClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def upload_document(self, file_path: str, metadata: Dict = None) -> Dict:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'metadata': json.dumps(metadata)} if metadata else {}
            response = requests.post(
                f"{self.base_url}/documents",
                files=files,
                data=data,
                headers=self.headers
            )
        return response.json()
    
    def query(self, query: str, top_k: int = 5, filters: Dict = None) -> Dict:
        data = {
            "query": query,
            "top_k": top_k,
            "filters": filters or {}
        }
        response = requests.post(
            f"{self.base_url}/query",
            json=data,
            headers=self.headers
        )
        return response.json()

# Usage
client = RAGClient("http://localhost:8000/api/v1", "your-api-key")

# Upload document
result = client.upload_document("document.pdf", {"category": "research"})
print(f"Document uploaded: {result['document_id']}")

# Query
response = client.query("What is machine learning?", top_k=3)
print(f"Answer: {response['response']}")
print(f"Sources: {len(response['sources'])}")
```

### JavaScript/Node.js SDK
```javascript
class RAGClient {
    constructor(baseUrl, apiKey) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        };
    }
    
    async uploadDocument(file, metadata = {}) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('metadata', JSON.stringify(metadata));
        
        const response = await fetch(`${this.baseUrl}/documents`, {
            method: 'POST',
            body: formData,
            headers: { 'Authorization': this.headers.Authorization }
        });
        return response.json();
    }
    
    async query(query, options = {}) {
        const data = {
            query,
            top_k: options.topK || 5,
            filters: options.filters || {}
        };
        
        const response = await fetch(`${this.baseUrl}/query`, {
            method: 'POST',
            body: JSON.stringify(data),
            headers: this.headers
        });
        return response.json();
    }
}

// Usage
const client = new RAGClient('http://localhost:8000/api/v1', 'your-api-key');

// Query documents
const response = await client.query('What is artificial intelligence?', {
    topK: 3,
    filters: { category: ['research'] }
});
console.log(response.response);
```

## OpenAPI/Swagger Documentation

The API provides interactive documentation at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

This allows for easy testing and integration with API client generators.
