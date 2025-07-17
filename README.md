# Open Source RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with local LLM integration using Ollama. This system allows you to upload documents, ask questions, and get intelligent answers based on your document collection.

![RAG System Demo](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Version](https://img.shields.io/badge/Version-1.3.0-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Municipal AI Features](#municipal-ai-features)
- [Documentation](#documentation)
- [Business Strategy](#business-strategy)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Functionality
- **Document Upload and Processing**: Support for PDF, DOCX, TXT, and CSV files
- **Intelligent Question Answering**: Uses Ollama LLM for context-aware responses
- **Vector Search**: Fast similarity search using FAISS
- **Hybrid Search**: Combines vector similarity with keyword matching
- **Real-time Processing**: Asynchronous document processing for better performance
- **Smart Answer Engine**: Advanced context extraction and answer generation

### Technical Features
- **RESTful API**: Built with FastAPI for high performance
- **Graceful Degradation**: Falls back to vector search when LLM unavailable
- **Comprehensive Error Handling**: Robust error recovery and user feedback
- **Rate Limiting**: Prevents abuse and ensures fair usage
- **Input Validation**: Security-focused validation of all inputs
- **Document Management**: Full CRUD operations for documents
- **Persistent Storage**: SQLite database for document metadata
- **Embeddable Widget**: Easy integration into any website
- **Performance Optimizations**: Caching, optimized search, and streaming responses

### Municipal AI Features 🏛️
- **Municipal-Specific Training**: Specialized for Swiss government services
- **150+ Use Cases**: Comprehensive coverage of citizen inquiries
- **Multilingual Support**: German, French, and Italian
- **Automated Web Scraping**: Extract municipal data automatically
- **Importance Scoring**: Prioritize official documents and services
- **Custom Model Training**: Train LLMs on municipal-specific data

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
git clone https://github.com/your-username/open-source-rag-system.git
cd open-source-rag-system

# Install dependencies
pip install -r simple_requirements.txt

# Install Ollama models (optional but recommended)
ollama pull phi3-mini    # Fast, lightweight model
ollama pull llama3.2:1b  # Ultra-fast model
```

### 2. Start the System

```bash
# Start API server
python simple_api.py
# Server runs on http://localhost:8001

# Open frontend
# Double-click simple_frontend.html
# Or visit: http://localhost:8001 (if served by API)
```

### 3. Upload Documents

1. Open the frontend in your browser
2. Click "Choose Files" and select PDF/DOCX/TXT files
3. Wait for processing to complete
4. Start asking questions!

### 4. Test the System

```bash
# Test API directly
python test_widget_endpoint.py

# Test widget integration
python test_widget_server.py
# Then visit: http://localhost:3000/widget/
```

## 🏛️ Municipal AI Features

This system includes specialized features for Swiss municipalities and government services:

### Quick Municipal Setup
```bash
# Set up municipal RAG for Arlesheim
python municipal_setup.py arlesheim --scrape --max-pages 50

# Test municipal queries
python municipal_setup.py arlesheim --query "Was sind die Öffnungszeiten der Gemeindeverwaltung?"

# Train custom municipal model
python train_arlesheim_model.py
```

### Supported Municipalities
- **Arlesheim** (BL) - Complete implementation with trained model
- **Basel, Bern, Zürich, Geneva, Lausanne** - Configuration ready
- **Custom municipalities** - Easy to add with municipal_setup.py

### Use Cases Covered
Our system handles **150+ municipal use cases** including:
- Citizen services and document requests
- Building permits and property services  
- Business licensing and commercial permits
- Transportation and parking information
- Parks, recreation, and environmental services
- Emergency preparedness and safety

See [strategy/MUNICIPAL_USE_CASES.md](strategy/MUNICIPAL_USE_CASES.md) for the complete list.

## 📚 Documentation

### Quick Navigation
- **[Get Started](docs/QUICKSTART.md)** - Quick setup guide
- **[API Reference](docs/API_DOCUMENTATION.md)** - Complete API documentation
- **[Municipal Guide](docs/MUNICIPAL_RAG_GUIDE.md)** - Municipal-specific features
- **[Testing Guide](TESTING.md)** - Testing framework and guidelines

### Documentation Structure
```
📁 docs/           - Technical documentation
📁 strategy/       - Business strategy and planning
📁 tests/         - Test suite and examples
📁 tools/         - Municipal tools and utilities
```

### Key Documents
- **[Architecture Overview](docs/ARCHITECTURE.md)** - System design and components
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Production deployment
- **[Security Guidelines](docs/SECURITY.md)** - Security best practices
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

## 💼 Business Strategy

### Market Opportunity
**Target**: Swiss municipalities seeking AI-powered citizen services
- **Market Size**: 300+ municipalities with digital transformation budgets
- **Revenue Potential**: CHF 660,000 ARR by Year 3
- **Business Model**: SaaS subscription (CHF 500-2,000/month)

### Key Strategic Documents
- **[Business Strategy](strategy/BUSINESS_STRATEGY.md)** - Complete business plan and market analysis
- **[Technical Roadmap](strategy/TECHNICAL_ROADMAP.md)** - Development roadmap and architecture evolution
- **[Municipal Use Cases](strategy/MUNICIPAL_USE_CASES.md)** - 150 comprehensive use cases

### Go-to-Market Strategy
1. **Perfect Arlesheim demo** (proven working system)
2. **Add 3-5 municipalities** (demonstrate scalability)
3. **Direct municipal outreach** (target IT departments)
4. **Build partner network** (municipal software vendors)

### Core Principles

1. **Source Verifiability**: All responses must be traceable to source documents
2. **Data Privacy**: Complete local processing - no external API calls
3. **Accuracy**: Only return information that exists in the knowledge base
4. **Performance**: Sub-second response times for most queries
5. **Scalability**: Support for millions of documents and concurrent users

## 🚀 Quick Start

### **Option 1: One-Click Setup (Recommended)**
```bash
# Clone the repository
git clone https://github.com/thenzler/open-source-rag-system.git
cd open-source-rag-system

# One-click setup and launch
python setup_rag_system.py

# Or use the quick start
python quick_start.py
```

### **Option 2: Windows Users**
```cmd
# Double-click one of these batch files:
start_server.bat
# or
start_rag.bat
```

### **Option 3: Manual Setup**
```bash
# Install dependencies
pip install -r simple_requirements.txt

# Start Ollama (for LLM support)
ollama serve
ollama pull mistral

# Start the system
python simple_api.py
```

### **Test the System**
```bash
# Upload a document
curl -X POST "http://localhost:8001/api/v1/documents" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@example.pdf"

# Query with streaming response
curl -X POST "http://localhost:8001/api/v1/query-stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "use_llm": true}'

# Regular query
curl -X POST "http://localhost:8001/api/v1/query-enhanced" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "use_llm": true}'
```

## 🎯 Performance Optimizations

Version 1.3.0 introduces major performance improvements:

- **⚡ Smart Caching**: 2-10x faster repeated queries
- **🔍 Optimized Search**: 3-5x faster vector similarity search  
- **📡 Streaming Responses**: Immediate user feedback
- **🚀 Batch Processing**: Efficient memory usage
- **🎯 Early Termination**: Stop at high-similarity matches

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
