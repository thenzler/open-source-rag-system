# Archived Services Documentation

This document lists the services that were moved to the archive during the MVP simplification process.

## Archived Services

### Authentication & Security
- `auth.py` - JWT authentication system with role-based access control
  - **Reason for archival**: Authentication not required for MVP, adds complexity
  - **Features**: JWT tokens, user management, role-based permissions

### Async Processing
- `async_processor.py` - Background document processing with job queues
  - **Reason for archival**: Synchronous processing sufficient for MVP
  - **Features**: Async job queues, progress tracking, worker management

### Advanced Storage Solutions
- `vector_store_db.py` - PostgreSQL + pgvector database storage
  - **Reason for archival**: Database complexity beyond MVP scope
  - **Features**: Persistent storage, SQL queries, performance optimization

- `hybrid_storage.py` - Automatic switching between database and memory storage
  - **Reason for archival**: Overengineered fallback system
  - **Features**: Automatic backend switching, capacity monitoring

- `memory_safe_storage.py` - Memory-safe storage with capacity limits
  - **Reason for archival**: Redundant with core simple storage
  - **Features**: Memory monitoring, capacity warnings

- `persistent_storage.py` - SQLite-based persistent storage
  - **Reason for archival**: MVP uses simple file-based storage
  - **Features**: SQLite database, persistent embeddings

### Advanced Search
- `vector_search.py` - FAISS-optimized vector search
  - **Reason for archival**: Simple cosine similarity sufficient for MVP
  - **Features**: FAISS indexing, optimized search algorithms

### LLM Management
- `llm_manager.py` - Dynamic model switching and configuration
  - **Reason for archival**: Single model sufficient for MVP
  - **Features**: Multi-model support, dynamic switching

## Services Kept for MVP

### Core Services (Kept)
- `validation.py` - Input validation and sanitization
  - **Reason kept**: Security is critical for MVP
  - **Features**: XSS prevention, input validation

- `document_manager.py` - Document lifecycle management
  - **Reason kept**: Core functionality for document handling
  - **Features**: Document metadata, lifecycle tracking

## Recovery Process

If any archived service is needed in the future:

1. Copy the service from `.archive/` back to `services/`
2. Update `services/__init__.py` to include the imports
3. Add any required dependencies to requirements
4. Test integration with current codebase
5. Update documentation

## MVP Philosophy

These services were archived to maintain focus on:
- **Reliability**: Zero crashes under normal usage
- **Simplicity**: Easy setup and maintenance
- **Core functionality**: Document upload and querying only

The core mission is: "Upload documents → Ask questions → Get intelligent answers" with maximum reliability.