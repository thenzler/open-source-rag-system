# Domain-Specific RAG Configuration Guide

This guide explains how to configure the RAG system for specific domains using the admin interface and configurable filtering.

## Overview

The RAG system provides domain-agnostic document management with configurable filtering for any use case:

- **Configurable keyword filtering** via admin interface
- **Content analysis and categorization** for any domain
- **Quality assessment** to identify problematic documents
- **Flexible document management** with cleanup tools
- **Multi-database support** (SQLite, PostgreSQL, MySQL)

## Quick Start for Any Domain

### 1. Upload Documents

```bash
# Start the system
python simple_api.py

# Upload documents via web interface
# Visit: http://localhost:8001
# Or upload via API:
curl -X POST "http://localhost:8001/api/v1/documents" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

### 2. Configure Domain-Specific Keywords

Access the admin interface at `http://localhost:8001/admin/documents/management`

```yaml
# Example for Healthcare Domain
target_keywords:
  - "patient", "medical", "treatment", "diagnosis", "healthcare"
problematic_keywords:
  - "training instructions", "guidelines for following"
exclude_keywords:
  - "programming", "software", "unrelated content"
```

### 3. Query the System

```bash
# Query documents
curl -X POST "http://localhost:8001/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the treatment options?"}'

# Get system health
curl "http://localhost:8001/health"

# List documents
curl "http://localhost:8001/api/v1/documents"
```

## System Architecture

### Components

1. **Document Management** (`core/routers/admin.py`)
   - Admin interface for document analysis and management
   - Configurable keyword filtering for any domain
   - Content quality assessment and cleanup tools

2. **RAG Service** (`simple_api.py`)
   - Zero-hallucination document-based responses
   - Vector search with sentence transformers
   - Confidence scoring and source citations

3. **Admin Interface** (`core/templates/`)
   - Model management and switching
   - Database configuration (SQLite, PostgreSQL, MySQL)
   - Document filtering and analysis tools

### Data Flow

```
Document Upload → Processing → Vector Indexing → Admin Analysis → Filtering → Query Processing → AI Response
```

## Configuration

### Domain-Specific Filtering

The system supports configurable filtering for any domain through the admin interface:

### Example Domain Configurations

**Healthcare Domain:**
```yaml
target_keywords:
  - "patient", "medical", "treatment", "diagnosis", "healthcare", "clinical"
problematic_keywords:
  - "training instructions", "zero-hallucination", "guidelines for following"
exclude_keywords:
  - "software", "programming", "unrelated"
```

**Legal Domain:**
```yaml
target_keywords:
  - "contract", "legal", "law", "regulation", "compliance", "statute"
problematic_keywords:
  - "training data", "instruction manual"
exclude_keywords:
  - "entertainment", "sports", "cooking"
```

**Education Domain:**
```yaml
target_keywords:
  - "curriculum", "student", "course", "learning", "education", "academic"
problematic_keywords:
  - "system prompts", "ai training"
exclude_keywords:
  - "commercial", "marketing", "sales"
```

## Content Analysis

The system automatically analyzes document content and provides:

### Quality Assessment
- **Clean Documents**: Well-formatted, relevant content with target keywords
- **Problematic Documents**: Training instructions, corrupted files, or off-topic content  
- **Unknown Documents**: Content that doesn't match configured keywords

### Confidence Scoring
Documents receive confidence scores based on:
- **Keyword Match**: Presence of target keywords (domain-specific)
- **Content Length**: Adequate content for meaningful analysis
- **File Quality**: Proper encoding and formatting

### Categorization
The admin interface allows you to:
- **Analyze All Documents**: Get comprehensive content analysis
- **Filter by Quality**: View clean vs problematic documents
- **Preview Content**: See document content before making decisions
- **Bulk Operations**: Clean multiple documents with filtering criteria

## Document Management Features

### Individual Document Operations
- **View**: Preview document content and metadata
- **Edit**: Update document metadata and classifications
- **Delete**: Remove documents from the system
- **Download**: Access original document files

### Bulk Management
- **Content Analysis**: Automatic categorization of all documents
- **Cleanup Tools**: Remove problematic documents in bulk
- **Dry Run**: Preview changes before applying them
- **Filter Configuration**: Save and update domain-specific keywords

## Model Configuration

The system supports multiple Ollama models through the admin interface:

### Available Models
Access model management at `http://localhost:8001/admin`

- **Switch Models**: Easy switching between different Ollama models
- **Model Status**: Check availability and installation status
- **Configuration**: Adjust temperature, context length, and other parameters
- **Health Monitoring**: Real-time model availability and performance metrics

### Model Examples
```yaml
# config/llm_config.yaml
models:
  mistral:
    name: "mistral:latest"
    description: "High-quality general purpose model"
    context_length: 32768
    temperature: 0.2
  
  llama3.2:
    name: "llama3.2:1b"
    description: "Fast, lightweight model"
    context_length: 4096
    temperature: 0.1
```

## API Endpoints

### Document Query
```
POST /api/v1/query
```
Query the RAG system with your uploaded documents.

**Request:**
```json
{
  "query": "What are the key features of this product?"
}
```

**Response:**
```json
{
  "answer": "Based on the uploaded documents, the key features include...",
  "sources": [
    {
      "id": 1,
      "document_id": 123,
      "similarity": 0.95,
      "download_url": "/api/v1/documents/123/download"
    }
  ],
  "confidence": 0.95,
  "processing_time": 1.2
}
```

### Document Management
```
GET /api/v1/documents          # List all documents
POST /api/v1/documents         # Upload new document
GET /api/v1/documents/{id}     # Get document details
DELETE /api/v1/documents/{id}  # Delete document
GET /api/v1/documents/{id}/download  # Download document
```

### Admin Endpoints
```
GET /admin                     # Admin dashboard
GET /admin/documents/management # Document management interface
POST /admin/documents/filter-config  # Update filtering configuration
POST /admin/documents/cleanup  # Clean problematic documents
```

### System Health
```
GET /health                    # System health check
GET /api/v1/status            # Detailed system status
```

## Business Use Cases

### 1. Knowledge Management
Deploy as an internal knowledge base for employee questions about company policies and procedures.

### 2. Customer Support
Integrate intelligent document search into customer support systems for instant answers.

### 3. Research Platform
Create a research platform for academic papers, legal documents, or technical specifications.

### 4. Compliance Assistant
Build a compliance system that helps users find relevant regulations and guidelines.

### 5. Training System
Develop training materials search for educational institutions or corporate training programs.

## Performance Optimization

### Vector Search
The system uses efficient similarity search with sentence transformers:
- Fast cosine similarity computation
- Optimized embedding generation
- Configurable similarity thresholds

### Caching
- Document embeddings are cached in the database
- Query results can be cached for repeated questions
- Model responses are optimized for speed

### Batch Processing
- Documents are processed efficiently in batches
- Embeddings are computed using optimized transformers
- Database operations are batched for performance

## Security Considerations

### Rate Limiting
- API endpoints have built-in rate limiting to prevent abuse
- Configurable limits per endpoint and user

### Data Privacy
- All processing happens locally - no external API calls required
- Document content stays on your infrastructure
- Optional database encryption support

### Content Validation
- All file uploads are validated for type and size
- Input sanitization prevents injection attacks
- Configurable security policies via admin interface

## Monitoring and Maintenance

### Health Checks
```bash
curl "http://localhost:8001/health"
```

### System Statistics
Access the admin dashboard at `http://localhost:8001/admin` for:
- Document count and status
- Model availability and performance
- Database connection status
- System resource usage

### Data Management
Use the admin interface for:
- Document analysis and cleanup
- Model switching and configuration
- Database backup and restore
- Filter configuration updates

## Troubleshooting

### Common Issues

1. **No Search Results**
   - Check if documents are uploaded and processed
   - Verify Ollama models are installed and available
   - Use admin interface to analyze document quality

2. **Poor Answer Quality**
   - Configure domain-specific keywords via admin interface
   - Clean problematic documents using admin tools
   - Switch to a higher-quality model via admin dashboard

3. **System Performance Issues**
   - Check database configuration (consider PostgreSQL for large datasets)
   - Monitor system resources via admin dashboard
   - Optimize document filtering to reduce noise

### Debug Mode
```bash
# Check system status
curl "http://localhost:8001/health"

# View admin dashboard for detailed diagnostics
# Visit: http://localhost:8001/admin

# Check document analysis
# Visit: http://localhost:8001/admin/documents/management
```

## Future Enhancements

### Planned Features
- Advanced caching and performance optimization
- Multi-language query support
- Enhanced content categorization with AI
- Integration with external databases and APIs
- Advanced analytics and reporting

### Business Applications
This system is perfect for:
- Enterprise knowledge management
- Customer support automation
- Research and compliance systems
- Educational platforms
- Legal and healthcare documentation

The MIT license ensures open-source availability with maximum flexibility for commercial use.

## Support

For questions or issues:
- Check the troubleshooting section above
- Use the admin interface for system diagnostics
- Test with simple queries first
- Verify all dependencies are installed via admin dashboard

---

**Note**: This system is domain-agnostic and can be configured for any use case through the admin interface and configurable keyword filtering.