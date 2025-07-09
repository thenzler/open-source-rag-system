# Development Guide

## 🚀 Getting Started

This guide will help you set up and run the Open Source RAG System for development.

### Prerequisites

- Docker and Docker Compose
- Git
- At least 8GB RAM
- 10GB free disk space

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/thenzler/open-source-rag-system.git
   cd open-source-rag-system
   ```

2. **Set up environment**
   ```bash
   make setup-env
   ```

3. **Edit the .env file**
   ```bash
   nano .env
   ```
   
   Key settings to configure:
   ```env
   # Database
   POSTGRES_PASSWORD=your_secure_password
   
   # Security
   SECRET_KEY=your-secret-key-here
   JWT_SECRET_KEY=your-jwt-secret-key
   
   # LLM Configuration
   LLM_MODEL_NAME=llama3.1:8b
   EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
   
   # File Upload Settings
   MAX_FILE_SIZE_MB=100
   CHUNK_SIZE=512
   CHUNK_OVERLAP=50
   ```

4. **Start the development environment**
   ```bash
   make dev
   ```

5. **Initialize the database**
   ```bash
   make db-init
   ```

6. **Access the services**
   - Web Interface: http://localhost:3000
   - API Documentation: http://localhost:8000/docs
   - API Gateway: http://localhost:8000
   - Grafana Dashboard: http://localhost:3001 (admin/admin)
   - Prometheus: http://localhost:9090

## 🏗️ Architecture Overview

### Services

1. **API Gateway** (Port 8000)
   - FastAPI-based REST API
   - Document upload/management
   - Query processing
   - Authentication

2. **Document Processor** (Port 8001)
   - Multi-format document parsing
   - Text extraction and chunking
   - Background processing with Celery

3. **Vector Engine** (Port 8002)
   - Embedding generation
   - Vector similarity search
   - Qdrant integration

4. **Web Interface** (Port 3000)
   - React frontend
   - Document management UI
   - Search interface

5. **Infrastructure**
   - PostgreSQL: Document metadata
   - Qdrant: Vector database
   - Redis: Caching and queues
   - Ollama: Local LLM service

## 🔧 Development Commands

### Environment Management
```bash
# Start development environment
make dev

# Build and start (fresh build)
make dev-build

# Stop all services
make stop

# View logs
make logs

# View specific service logs
make logs-api
make logs-processor
make logs-vector
```

### Database Management
```bash
# Initialize database
make db-init

# Run migrations
make db-migrate

# Reset database (WARNING: destroys data)
make db-reset

# Backup database
make db-backup

# Restore database
make db-restore BACKUP_FILE=backup.sql
```

### Testing
```bash
# Run all tests
make test

# Run unit tests
make test-unit

# Run integration tests
make test-integration

# Run with coverage
make test-coverage
```

### Code Quality
```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Security scan
make security-scan
```

## 🧪 Testing the System

### 1. Upload a Document
```bash
# Via API
curl -X POST "http://localhost:8000/api/v1/documents" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@example.pdf"

# Via Web Interface
# Navigate to http://localhost:3000 and use the upload form
```

### 2. Query Documents
```bash
# Via API
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "top_k": 5}'

# Via Web Interface
# Use the search box on the main page
```

### 3. Check System Health
```bash
# API Gateway health
curl http://localhost:8000/health

# Document Processor health
curl http://localhost:8001/health

# Vector Engine health
curl http://localhost:8002/health
```

## 📁 Project Structure

```
open-source-rag-system/
├── services/
│   ├── api-gateway/          # FastAPI main service
│   │   ├── app/
│   │   │   ├── core/         # Configuration, database, security
│   │   │   ├── models/       # SQLAlchemy models
│   │   │   ├── schemas/      # Pydantic schemas
│   │   │   ├── services/     # Business logic
│   │   │   └── main.py       # FastAPI app
│   │   └── requirements.txt
│   ├── document-processor/   # Document processing service
│   │   ├── app/
│   │   │   └── main.py       # Document processor
│   │   └── requirements.txt
│   ├── vector-engine/        # Vector search service
│   │   ├── app/
│   │   │   └── main.py       # Vector engine
│   │   └── requirements.txt
│   └── web-interface/        # React frontend
│       ├── src/
│       │   ├── App.js
│       │   └── App.css
│       └── package.json
├── scripts/
│   └── init.sql              # Database initialization
├── docker-compose.yml        # Docker services
├── Makefile                  # Development commands
└── .env.example              # Environment template
```

## 🔄 Development Workflow

### 1. Making Changes

1. **API Gateway Changes**
   - Edit files in `services/api-gateway/app/`
   - Changes are hot-reloaded in development

2. **Document Processor Changes**
   - Edit files in `services/document-processor/app/`
   - Restart service: `docker-compose restart document-processor`

3. **Vector Engine Changes**
   - Edit files in `services/vector-engine/app/`
   - Restart service: `docker-compose restart vector-engine`

4. **Web Interface Changes**
   - Edit files in `services/web-interface/src/`
   - Changes are hot-reloaded in development

### 2. Adding New Features

1. **Create a new branch**
   ```bash
   git checkout -b feature/new-feature
   ```

2. **Make your changes**
   - Follow the existing code structure
   - Add tests for new functionality
   - Update documentation

3. **Test your changes**
   ```bash
   make test
   make lint
   ```

4. **Create a pull request**

### 3. Database Schema Changes

1. **Create migration**
   ```bash
   # In api-gateway container
   alembic revision --autogenerate -m "Add new table"
   ```

2. **Apply migration**
   ```bash
   make db-migrate
   ```

## 🐛 Debugging

### Common Issues

1. **Services not starting**
   - Check logs: `make logs`
   - Verify .env configuration
   - Check port conflicts

2. **Database connection errors**
   - Verify PostgreSQL is running
   - Check database credentials
   - Run `make db-init`

3. **Vector search not working**
   - Check Qdrant is running
   - Verify embedding model is loaded
   - Check vector engine logs

4. **Document processing fails**
   - Check supported file formats
   - Verify file size limits
   - Check celery worker logs

### Debugging Tools

1. **Container shells**
   ```bash
   make shell-api      # API Gateway shell
   make shell-db       # PostgreSQL shell
   make shell-redis    # Redis CLI
   ```

2. **Service logs**
   ```bash
   make logs-api
   make logs-processor
   make logs-vector
   ```

3. **Database queries**
   ```bash
   make shell-db
   ragdb=# SELECT * FROM documents;
   ```

## 📊 Monitoring

### Metrics and Dashboards

1. **Grafana Dashboard**
   - URL: http://localhost:3001
   - Login: admin/admin
   - Pre-configured dashboards for system metrics

2. **Prometheus Metrics**
   - URL: http://localhost:9090
   - Raw metrics endpoint: http://localhost:8000/metrics

### Health Checks

All services expose health check endpoints:
- API Gateway: `/health`
- Document Processor: `/health`
- Vector Engine: `/health`

## 🚀 Production Deployment

### 1. Production Build
```bash
make prod-build
```

### 2. Production Configuration
- Use strong passwords
- Configure SSL/TLS
- Set up proper logging
- Configure monitoring

### 3. Scaling
- Scale document processors: `docker-compose up -d --scale document-processor=3`
- Scale vector engines: `docker-compose up -d --scale vector-engine=2`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

### Code Style

- Python: Follow PEP 8
- JavaScript: Follow Airbnb style guide
- Use meaningful variable names
- Add docstrings to functions
- Keep functions small and focused

### Testing

- Write unit tests for new features
- Add integration tests for API endpoints
- Test edge cases and error conditions
- Maintain test coverage above 80%

## 📝 Logging

Logs are stored in the `logs/` directory:
- `api-gateway.log`: API Gateway logs
- `document-processor.log`: Document processing logs
- `vector-engine.log`: Vector engine logs

Configure log levels in the `.env` file:
```env
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

## 🔒 Security

- Change default passwords
- Use HTTPS in production
- Implement rate limiting
- Validate all inputs
- Keep dependencies updated

## 📖 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://reactjs.org/)
- [Qdrant Documentation](https://qdrant.tech/)
- [Sentence Transformers](https://www.sbert.net/)

---

For questions or issues, please check the [GitHub Issues](https://github.com/thenzler/open-source-rag-system/issues) or start a [Discussion](https://github.com/thenzler/open-source-rag-system/discussions).
