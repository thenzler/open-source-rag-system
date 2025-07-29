"""
PostgreSQL Repository Implementation
Handles database operations with PostgreSQL backend
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import asyncpg

from .base import RepositoryResult
from .interfaces import IDocumentRepository
from .models import Document, User

logger = logging.getLogger(__name__)


class PostgreSQLRepository(IDocumentRepository):
    """PostgreSQL implementation of document repository"""

    def __init__(self, connection_string: str, pool_size: int = 10):
        """Initialize PostgreSQL repository"""
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False

    async def initialize(self):
        """Initialize database connection pool and tables"""
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=self.pool_size,
                command_timeout=60,
            )

            # Create tables
            await self._create_tables()

            self._initialized = True
            logger.info(
                f"PostgreSQL repository initialized with pool size {self.pool_size}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL repository: {e}")
            raise

    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")

    async def _create_tables(self):
        """Create database tables if they don't exist"""
        async with self.pool.acquire() as conn:
            # Create tenants table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tenants (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    slug VARCHAR(100) UNIQUE NOT NULL,
                    domain VARCHAR(255) UNIQUE,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    settings JSONB DEFAULT '{}',
                    limits JSONB DEFAULT '{}'
                )
            """
            )
            # Create users table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    tenant_id INTEGER REFERENCES tenants(id) DEFAULT 1,
                    username VARCHAR(100) NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role VARCHAR(50) DEFAULT 'user',
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    metadata JSONB DEFAULT '{}',
                    UNIQUE(tenant_id, username)
                )
            """
            )

            # Create documents table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    tenant_id INTEGER REFERENCES tenants(id) DEFAULT 1,
                    filename VARCHAR(500) NOT NULL,
                    original_filename VARCHAR(500) NOT NULL,
                    file_path TEXT NOT NULL,
                    content_type VARCHAR(100),
                    file_size BIGINT DEFAULT 0,
                    status VARCHAR(50) DEFAULT 'uploading',
                    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_timestamp TIMESTAMP,
                    completion_timestamp TIMESTAMP,
                    uploader VARCHAR(255),
                    description TEXT,
                    tags TEXT[],
                    metadata JSONB DEFAULT '{}',
                    text_content TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    embedding_count INTEGER DEFAULT 0
                )
            """
            )

            # Create chunks table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    tenant_id INTEGER REFERENCES tenants(id) DEFAULT 1,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    text_content TEXT NOT NULL,
                    character_count INTEGER DEFAULT 0,
                    word_count INTEGER DEFAULT 0,
                    start_char INTEGER DEFAULT 0,
                    end_char INTEGER DEFAULT 0,
                    quality_score FLOAT DEFAULT 0.0,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create embeddings table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    tenant_id INTEGER REFERENCES tenants(id) DEFAULT 1,
                    chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    embedding_data BYTEA NOT NULL,
                    embedding_model VARCHAR(255) NOT NULL,
                    dimensions INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create query_logs table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_logs (
                    id SERIAL PRIMARY KEY,
                    tenant_id INTEGER REFERENCES tenants(id) DEFAULT 1,
                    query_text TEXT NOT NULL,
                    user_id INTEGER REFERENCES users(id),
                    result_count INTEGER DEFAULT 0,
                    processing_time FLOAT DEFAULT 0.0,
                    method VARCHAR(100) DEFAULT 'vector_search',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB DEFAULT '{}'
                )
            """
            )

            # Create indexes for performance
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_tenant_id ON documents(tenant_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_upload_timestamp ON documents(upload_timestamp)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_tenant_id ON chunks(tenant_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_embeddings_document_id ON embeddings(document_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_embeddings_tenant_id ON embeddings(tenant_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_users_tenant_id ON users(tenant_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_query_logs_tenant_id ON query_logs(tenant_id)"
            )

            # Insert default tenant if not exists
            await conn.execute(
                """
                INSERT INTO tenants (id, name, slug, is_active, settings, limits)
                VALUES (1, 'Default Organization', 'default', TRUE, '{}', '{"max_documents": 1000, "max_storage_mb": 1024}')
                ON CONFLICT (id) DO NOTHING
            """
            )

            logger.info("PostgreSQL tables created/verified successfully")

    async def create(self, document: Document) -> Document:
        """Create a new document"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO documents (
                    tenant_id, filename, original_filename, file_path, content_type,
                    file_size, status, upload_timestamp, uploader, description,
                    tags, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                RETURNING *
            """,
                document.tenant_id,
                document.filename,
                document.original_filename,
                document.file_path,
                document.content_type,
                document.file_size,
                document.status.value if document.status else "uploading",
                document.upload_timestamp or datetime.now(),
                document.uploader,
                document.description,
                document.tags or [],
                json.dumps(document.metadata or {}),
            )

            return self._row_to_document(row)

    async def get_by_id(self, document_id: int) -> Optional[Document]:
        """Get document by ID"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM documents WHERE id = $1", document_id
            )

            if not row:
                return None

            return self._row_to_document(row)

    async def list_all(
        self, tenant_id: Optional[int] = None, limit: int = 100, offset: int = 0
    ) -> RepositoryResult:
        """List all documents with optional tenant filtering"""
        async with self.pool.acquire() as conn:
            if tenant_id:
                # Get total count
                total_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM documents WHERE tenant_id = $1", tenant_id
                )

                # Get documents
                rows = await conn.fetch(
                    """
                    SELECT * FROM documents
                    WHERE tenant_id = $1
                    ORDER BY upload_timestamp DESC
                    LIMIT $2 OFFSET $3
                """,
                    tenant_id,
                    limit,
                    offset,
                )
            else:
                # Get total count
                total_count = await conn.fetchval("SELECT COUNT(*) FROM documents")

                # Get documents
                rows = await conn.fetch(
                    """
                    SELECT * FROM documents
                    ORDER BY upload_timestamp DESC
                    LIMIT $1 OFFSET $2
                """,
                    limit,
                    offset,
                )

            documents = [self._row_to_document(row) for row in rows]

            return RepositoryResult(
                items=documents, total_count=total_count, limit=limit, offset=offset
            )

    async def update(
        self, document_id: int, updates: Dict[str, Any]
    ) -> Optional[Document]:
        """Update document"""
        # Build dynamic update query
        set_clauses = []
        params = []
        param_count = 1

        for key, value in updates.items():
            if key in [
                "filename",
                "description",
                "status",
                "text_content",
                "chunk_count",
                "embedding_count",
            ]:
                set_clauses.append(f"{key} = ${param_count}")
                params.append(value)
                param_count += 1
            elif key == "metadata":
                set_clauses.append(f"metadata = ${param_count}")
                params.append(json.dumps(value))
                param_count += 1
            elif key == "tags":
                set_clauses.append(f"tags = ${param_count}")
                params.append(value)
                param_count += 1

        if not set_clauses:
            return await self.get_by_id(document_id)

        # Add timestamps
        if "status" in updates:
            if updates["status"] == "processing":
                set_clauses.append(f"processing_timestamp = ${param_count}")
                params.append(datetime.now())
                param_count += 1
            elif updates["status"] == "completed":
                set_clauses.append(f"completion_timestamp = ${param_count}")
                params.append(datetime.now())
                param_count += 1

        params.append(document_id)

        async with self.pool.acquire() as conn:
            # Safe SQL construction: set_clauses built from validated fields only
            query = f"""
                UPDATE documents
                SET {', '.join(set_clauses)}
                WHERE id = ${param_count}
                RETURNING *
            """  # nosec B608

            row = await conn.fetchrow(query, *params)

            if not row:
                return None

            return self._row_to_document(row)

    async def delete(self, document_id: int) -> bool:
        """Delete document"""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM documents WHERE id = $1", document_id
            )

            return result != "DELETE 0"

    async def find_by_hash(self, file_hash: str) -> Optional[Document]:
        """Find document by file hash"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM documents
                WHERE metadata->>'file_hash' = $1
            """,
                file_hash,
            )

            if not row:
                return None

            return self._row_to_document(row)

    async def update_status(self, document_id: int, status: str) -> bool:
        """Update document status"""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE documents
                SET status = $1,
                    processing_timestamp = CASE WHEN $1 = 'processing' THEN CURRENT_TIMESTAMP ELSE processing_timestamp END,
                    completion_timestamp = CASE WHEN $1 = 'completed' THEN CURRENT_TIMESTAMP ELSE completion_timestamp END
                WHERE id = $2
            """,
                status,
                document_id,
            )

            return result != "UPDATE 0"

    def _row_to_document(self, row) -> Document:
        """Convert database row to Document object"""
        from .models import DocumentStatus

        # Parse status
        status = DocumentStatus.UPLOADING
        if row["status"]:
            try:
                status = DocumentStatus(row["status"])
            except ValueError:
                status = DocumentStatus.UPLOADING

        # Parse metadata
        metadata = {}
        if row["metadata"]:
            try:
                metadata = (
                    json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"]
                )
            except (json.JSONDecodeError, TypeError):
                metadata = {}

        return Document(
            id=row["id"],
            tenant_id=row.get("tenant_id", 1),
            filename=row["filename"],
            original_filename=row["original_filename"],
            file_path=row["file_path"],
            content_type=row["content_type"],
            file_size=row["file_size"],
            status=status,
            upload_timestamp=row["upload_timestamp"],
            processing_timestamp=row["processing_timestamp"],
            completion_timestamp=row["completion_timestamp"],
            uploader=row["uploader"],
            description=row["description"],
            tags=row.get("tags", []) or [],
            metadata=metadata,
            text_content=row.get("text_content"),
            chunk_count=row.get("chunk_count", 0),
            embedding_count=row.get("embedding_count", 0),
        )


class PostgreSQLUserRepository:
    """PostgreSQL implementation for user management"""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def create_user(self, user: User) -> User:
        """Create a new user"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO users (
                    tenant_id, username, email, password_hash, role,
                    is_active, created_at, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING *
            """,
                user.tenant_id,
                user.username,
                user.email,
                user.password_hash,
                user.role,
                user.is_active,
                user.created_at or datetime.now(),
                json.dumps(user.metadata or {}),
            )

            return self._row_to_user(row)

    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)

            if not row:
                return None

            return self._row_to_user(row)

    async def get_user_by_email(
        self, email: str, tenant_id: Optional[int] = None
    ) -> Optional[User]:
        """Get user by email"""
        async with self.pool.acquire() as conn:
            if tenant_id:
                row = await conn.fetchrow(
                    "SELECT * FROM users WHERE email = $1 AND tenant_id = $2",
                    email,
                    tenant_id,
                )
            else:
                row = await conn.fetchrow("SELECT * FROM users WHERE email = $1", email)

            if not row:
                return None

            return self._row_to_user(row)

    async def list_users(
        self, tenant_id: Optional[int] = None, limit: int = 100, offset: int = 0
    ) -> List[User]:
        """List users"""
        async with self.pool.acquire() as conn:
            if tenant_id:
                rows = await conn.fetch(
                    """
                    SELECT * FROM users
                    WHERE tenant_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2 OFFSET $3
                """,
                    tenant_id,
                    limit,
                    offset,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM users
                    ORDER BY created_at DESC
                    LIMIT $1 OFFSET $2
                """,
                    limit,
                    offset,
                )

            return [self._row_to_user(row) for row in rows]

    def _row_to_user(self, row) -> User:
        """Convert database row to User object"""
        metadata = {}
        if row["metadata"]:
            try:
                metadata = (
                    json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"]
                )
            except (json.JSONDecodeError, TypeError):
                metadata = {}

        return User(
            id=row["id"],
            tenant_id=row.get("tenant_id", 1),
            username=row["username"],
            email=row["email"],
            password_hash=row["password_hash"],
            role=row["role"],
            is_active=row["is_active"],
            created_at=row["created_at"],
            last_login=row["last_login"],
            metadata=metadata,
        )
