# Open Source RAG AI System

🚀 **Complete locally hosted Retrieval-Augmented Generation system with document processing, vector search, and API access**

## Overview

This project provides a comprehensive, open-source RAG (Retrieval-Augmented Generation) AI system designed to run entirely locally while providing enterprise-grade capabilities for document processing, information retrieval, and API access.

### Key Features

- **Multi-format Document Processing**: Support for PDF, Word (.docx), Excel (.xlsx), XML, and text files
- **Advanced Vector Search**: High-performance semantic search with source attribution
- **Reliable Source Tracking**: Every response includes document ID and location references
- **RESTful API**: Clean, well-documented API for easy integration
- **Local Deployment**: Runs entirely on your infrastructure - no external dependencies
- **Scalable Architecture**: Designed to handle enterprise workloads
- **Open Source**: MIT licensed with full transparency

### Core Principles

1. **Source Verifiability**: All responses must be traceable to source documents
2. **Data Privacy**: Complete local processing - no external API calls
3. **Accuracy**: Only return information that exists in the knowledge base
4. **Performance**: Sub-second response times for most queries
5. **Scalability**: Support for millions of documents and concurrent users

## Quick Start

```bash
# Clone the repository
git clone https://github.com/thenzler/open-source-rag-system.git
cd open-source-rag-system

# Start with Docker Compose
docker-compose up -d

# Upload your first document
curl -X POST "http://localhost:8000/api/documents" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@example.pdf"

# Query the system
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "top_k": 5}'
```

## Architecture

The system consists of several microservices:

- **Document Processor**: Extracts text and metadata from various file formats
- **Vector Engine**: Handles embedding generation and similarity search
- **API Gateway**: Provides RESTful endpoints and authentication
- **Database Layer**: PostgreSQL for metadata, Qdrant for vectors
- **Web Interface**: Optional React-based admin panel

## Documentation

- [📋 Architecture Details](./docs/ARCHITECTURE.md)
- [🔧 Technology Stack](./docs/TECHNOLOGY_STACK.md)
- [🚀 Deployment Guide](./docs/DEPLOYMENT.md)
- [📡 API Documentation](./docs/API.md)
- [🧪 Testing Strategy](./docs/TESTING.md)
- [🔒 Security Guide](./docs/SECURITY.md)

## Project Structure

```
open-source-rag-system/
├── services/
│   ├── api-gateway/          # FastAPI main service
│   ├── document-processor/   # Document parsing and processing
│   ├── vector-engine/        # Embedding and search service
│   └── web-interface/        # React admin panel
├── infrastructure/
│   ├── docker/              # Docker configurations
│   ├── kubernetes/          # K8s deployment files
│   └── monitoring/          # Observability setup
├── docs/                    # Comprehensive documentation
├── tests/                   # Test suites
└── scripts/                 # Utility scripts
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](./CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](./LICENSE) for details.

## Support

- 📖 [Documentation](./docs/)
- 🐛 [Issue Tracker](https://github.com/thenzler/open-source-rag-system/issues)
- 💬 [Discussions](https://github.com/thenzler/open-source-rag-system/discussions)
