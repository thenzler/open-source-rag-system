# Open Source RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with local LLM integration using Ollama. This system allows you to upload documents, ask questions, and get intelligent answers based on your document collection.

![RAG System Demo](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Version](https://img.shields.io/badge/Version-2.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Admin Interface](#admin-interface)
- [Documentation](#documentation)
- [Architecture](#architecture)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Functionality
- **Document Upload and Processing**: Support for PDF, DOCX, TXT, and CSV files
- **Intelligent Question Answering**: Uses Ollama LLM for context-aware responses
- **Vector Search**: Fast similarity search using sentence transformers
- **Hybrid Search**: Combines vector similarity with keyword matching
- **Real-time Processing**: Asynchronous document processing for better performance
- **Smart Answer Engine**: Advanced context extraction and answer generation
- **Zero-Hallucination Design**: Only provides answers based on uploaded documents

### Admin Interface & Management
- **Comprehensive Admin Dashboard**: Model switching, system monitoring, and configuration
- **Document Management**: Content analysis, filtering, and cleanup tools
- **Configurable Filtering**: Domain-agnostic keyword-based document filtering
- **Database Configuration**: Support for SQLite, PostgreSQL, and MySQL
- **Single Document Management**: View, edit, and delete individual documents
- **System Health Monitoring**: Real-time status and performance metrics

### Technical Features
- **RESTful API**: Built with FastAPI for high performance
- **Graceful Degradation**: Falls back to vector search when LLM unavailable
- **Comprehensive Error Handling**: Robust error recovery and user feedback
- **Rate Limiting**: Prevents abuse and ensures fair usage
- **Input Validation**: Security-focused validation of all inputs
- **Document Management**: Full CRUD operations for documents
- **Persistent Storage**: SQLite database with optional PostgreSQL/MySQL support
- **Multi-Model Support**: Easy switching between different Ollama models
- **Performance Optimizations**: Caching, optimized search, and streaming responses

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, Linux, or macOS
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space
- **CPU**: 4 cores recommended

### Software Dependencies
- **Ollama**: For local LLM inference
- **Python Libraries**: See `simple_requirements.txt`

## Quick Start

### Prerequisites

- **Python 3.8+**
- **Ollama** (for AI generation) - [Download here](https://ollama.com/download)
- **Git** (for cloning)

### 1. Installation

```bash
# Clone repository
git clone https://github.com/thenzler/open-source-rag-system.git
cd open-source-rag-system

# Install dependencies
pip install -r simple_requirements.txt

# Install Ollama models (optional but recommended)
ollama pull phi3-mini    # Fast, lightweight model
ollama pull llama3.2:1b  # Ultra-fast model
ollama pull mistral      # High-quality general model
```

### 2. Start the System

```bash
# Start API server
python simple_api.py
# Server runs on http://localhost:8001

# Open web interface
# Visit: http://localhost:8001
```

### 3. Upload Documents

1. Open http://localhost:8001 in your browser
2. Click "Choose Files" and select PDF/DOCX/TXT files
3. Wait for processing to complete
4. Start asking questions!

### 4. Access Admin Interface

```bash
# Visit the admin interface at:
# http://localhost:8001/admin

# Features available:
# - Model switching and configuration
# - Document management and analysis  
# - System monitoring and health checks
# - Database configuration
```

## Admin Interface

The system includes a comprehensive admin interface for managing your RAG system:

### Document Management
- **Content Analysis**: Automatically categorize and analyze document quality
- **Configurable Filtering**: Set up domain-specific keywords for document classification
- **Cleanup Tools**: Remove problematic or off-topic documents
- **Individual Management**: View, edit, and delete specific documents

### Model Management
```bash
# Access admin interface at: http://localhost:8001/admin

# Available features:
# - Switch between different Ollama models
# - Monitor model availability and status
# - Download configuration backups
# - View system health and performance metrics
```

### Database Configuration
- **Multiple Database Support**: SQLite (default), PostgreSQL, MySQL
- **Connection Testing**: Verify database connectivity before saving
- **Migration Tools**: Easy switching between database types
- **Backup and Restore**: Configuration download and restore capabilities

### Use Cases
This RAG system is perfect for:
- **Knowledge Management**: Company documentation and policies
- **Customer Support**: FAQ and help documentation
- **Research**: Academic papers and research materials
- **Legal**: Contract and document analysis
- **Healthcare**: Medical documentation and guidelines
- **Education**: Course materials and educational content

## 📚 Documentation

### Quick Navigation
- **[Setup Guide](SIMPLE_RAG_README.md)** - Quick setup and usage guide
- **[API Reference](docs/API_DOCUMENTATION.md)** - Complete API documentation
- **[Domain Configuration](docs/DOMAIN_CONFIGURATION_GUIDE.md)** - Configure for specific domains
- **[Testing Guide](TESTING.md)** - Testing framework and guidelines
- **[CLAUDE.md](CLAUDE.md)** - AI assistant project guidelines

### Documentation Structure
```
📁 docs/           - Technical documentation
📁 core/           - Main application code
📁 tests/          - Test suite and examples
📁 config/         - Configuration files
📁 static/         - Web interface assets
```

### Key Documents
- **[Architecture Overview](docs/ARCHITECTURE.md)** - System design and components
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Production deployment
- **[Security Guidelines](docs/SECURITY.md)** - Security best practices
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

## Core Principles

1. **Source Verifiability**: All responses must be traceable to source documents
2. **Data Privacy**: Complete local processing - no external API calls required
3. **Zero Hallucination**: Only return information that exists in the knowledge base
4. **Performance**: Sub-second response times for most queries
5. **Scalability**: Support for thousands of documents and concurrent users
6. **Reliability**: Graceful degradation when services are unavailable
7. **Security**: Input validation and rate limiting built-in

## API Usage

### **Test the System via API**
```bash
# Upload a document
curl -X POST "http://localhost:8001/api/v1/documents" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@example.pdf"

# Query documents
curl -X POST "http://localhost:8001/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?"}'

# Get system health
curl "http://localhost:8001/health"

# List all documents
curl "http://localhost:8001/api/v1/documents"
```

## 🎯 Performance Features

Version 2.0.0 includes major performance and usability improvements:

- **⚡ Smart Caching**: Faster repeated queries
- **🔍 Optimized Search**: Improved vector similarity search  
- **📊 Admin Dashboard**: Complete system management interface
- **🗂️ Document Management**: Content analysis and cleanup tools
- **🔄 Model Switching**: Easy switching between Ollama models
- **🗃️ Database Options**: SQLite, PostgreSQL, and MySQL support

## Architecture

The system uses a modern, modular architecture:

- **FastAPI Backend**: High-performance API with comprehensive admin interface
- **Document Processor**: Extracts text and metadata from various file formats
- **Vector Engine**: Handles embedding generation and similarity search using sentence transformers
- **Admin Interface**: Complete management dashboard for models, documents, and system configuration
- **Database Layer**: Flexible storage with SQLite default and PostgreSQL/MySQL support
- **Web Interface**: Modern, responsive UI for document upload and querying

## Project Structure

```
open-source-rag-system/
├── core/                    # Main application code
│   ├── routers/            # FastAPI route handlers
│   ├── services/           # Business logic services
│   ├── repositories/       # Data access layer
│   └── templates/          # HTML templates for admin interface
├── static/                 # Web interface assets
├── config/                 # Configuration files
├── storage/               # Document storage
├── docs/                  # Technical documentation
├── tests/                 # Test suites
└── simple_api.py          # Main application entry point
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](./CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](./LICENSE) for details.

## Support

- 📖 [Documentation](./docs/)
- 🐛 [Issue Tracker](https://github.com/thenzler/open-source-rag-system/issues)
- 💬 [Discussions](https://github.com/thenzler/open-source-rag-system/discussions)
