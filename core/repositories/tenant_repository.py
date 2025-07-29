"""
Tenant Repository for Multi-Tenancy Support
Handles tenant management and data isolation
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseRepository
from .models import Tenant

logger = logging.getLogger(__name__)


class TenantRepository(BaseRepository[Tenant, int]):
    """Repository for tenant management"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_tenant_tables()

    def _ensure_tenant_tables(self):
        """Create tenant tables if they don't exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create tenants table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tenants (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        slug TEXT UNIQUE NOT NULL,
                        domain TEXT UNIQUE,
                        is_active BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        settings TEXT DEFAULT '{}',
                        limits TEXT DEFAULT '{}'
                    )
                """
                )

                # Create default tenant if it doesn't exist
                cursor = conn.execute("SELECT COUNT(*) FROM tenants WHERE id = 1")
                if cursor.fetchone()[0] == 0:
                    conn.execute(
                        """
                        INSERT INTO tenants (id, name, slug, is_active, settings, limits)
                        VALUES (1, 'Default Organization', 'default', 1, '{}', '{"max_documents": 1000, "max_storage_mb": 1024}')
                    """
                    )

                # Add tenant_id columns to existing tables if they don't exist
                self._add_tenant_columns(conn)

                conn.commit()
                logger.info("Tenant tables initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize tenant tables: {e}")
            raise

    def _add_tenant_columns(self, conn: sqlite3.Connection):
        """Add tenant_id columns to existing tables"""
        tables_to_update = [
            ("documents", "tenant_id INTEGER DEFAULT 1 NOT NULL"),
            ("chunks", "tenant_id INTEGER DEFAULT 1 NOT NULL"),
            ("embeddings", "tenant_id INTEGER DEFAULT 1 NOT NULL"),
            ("users", "tenant_id INTEGER DEFAULT 1 NOT NULL"),
            ("query_logs", "tenant_id INTEGER DEFAULT 1 NOT NULL"),
        ]

        for table_name, column_def in tables_to_update:
            try:
                # Check if table exists
                cursor = conn.execute(
                    """
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name=?
                """,
                    (table_name,),
                )

                if cursor.fetchone():
                    # Check if column exists
                    cursor = conn.execute(f"PRAGMA table_info({table_name})")
                    columns = [row[1] for row in cursor.fetchall()]

                    if "tenant_id" not in columns:
                        conn.execute(
                            f"ALTER TABLE {table_name} ADD COLUMN {column_def}"
                        )
                        logger.info(f"Added tenant_id column to {table_name}")

                        # Create index for performance
                        conn.execute(
                            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_tenant_id ON {table_name}(tenant_id)"
                        )

            except Exception as e:
                logger.warning(f"Could not add tenant_id to {table_name}: {e}")

    async def create_tenant(self, tenant: Tenant) -> Tenant:
        """Create a new tenant"""
        try:
            import json

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO tenants (name, slug, domain, is_active, settings, limits)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        tenant.name,
                        tenant.slug,
                        tenant.domain,
                        tenant.is_active,
                        json.dumps(tenant.settings),
                        json.dumps(tenant.limits),
                    ),
                )

                tenant.id = cursor.lastrowid
                tenant.created_at = datetime.now()

                logger.info(f"Created tenant: {tenant.name} (ID: {tenant.id})")
                return tenant

        except sqlite3.IntegrityError as e:
            if "slug" in str(e):
                raise ValueError(f"Tenant slug '{tenant.slug}' already exists")
            elif "domain" in str(e):
                raise ValueError(f"Domain '{tenant.domain}' already exists")
            else:
                raise ValueError(f"Tenant creation failed: {e}")
        except Exception as e:
            logger.error(f"Failed to create tenant: {e}")
            raise

    async def get_by_id(self, tenant_id: int) -> Optional[Tenant]:
        """Get tenant by ID"""
        try:
            import json

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT id, name, slug, domain, is_active, created_at, settings, limits
                    FROM tenants WHERE id = ?
                """,
                    (tenant_id,),
                )

                row = cursor.fetchone()
                if not row:
                    return None

                return Tenant(
                    id=row[0],
                    name=row[1],
                    slug=row[2],
                    domain=row[3],
                    is_active=bool(row[4]),
                    created_at=datetime.fromisoformat(row[5]) if row[5] else None,
                    settings=json.loads(row[6] or "{}"),
                    limits=json.loads(row[7] or "{}"),
                )

        except Exception as e:
            logger.error(f"Failed to get tenant {tenant_id}: {e}")
            return None

    async def get_by_slug(self, slug: str) -> Optional[Tenant]:
        """Get tenant by slug"""
        try:
            import json

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT id, name, slug, domain, is_active, created_at, settings, limits
                    FROM tenants WHERE slug = ? AND is_active = 1
                """,
                    (slug,),
                )

                row = cursor.fetchone()
                if not row:
                    return None

                return Tenant(
                    id=row[0],
                    name=row[1],
                    slug=row[2],
                    domain=row[3],
                    is_active=bool(row[4]),
                    created_at=datetime.fromisoformat(row[5]) if row[5] else None,
                    settings=json.loads(row[6] or "{}"),
                    limits=json.loads(row[7] or "{}"),
                )

        except Exception as e:
            logger.error(f"Failed to get tenant by slug {slug}: {e}")
            return None

    async def get_by_domain(self, domain: str) -> Optional[Tenant]:
        """Get tenant by custom domain"""
        try:
            import json

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT id, name, slug, domain, is_active, created_at, settings, limits
                    FROM tenants WHERE domain = ? AND is_active = 1
                """,
                    (domain,),
                )

                row = cursor.fetchone()
                if not row:
                    return None

                return Tenant(
                    id=row[0],
                    name=row[1],
                    slug=row[2],
                    domain=row[3],
                    is_active=bool(row[4]),
                    created_at=datetime.fromisoformat(row[5]) if row[5] else None,
                    settings=json.loads(row[6] or "{}"),
                    limits=json.loads(row[7] or "{}"),
                )

        except Exception as e:
            logger.error(f"Failed to get tenant by domain {domain}: {e}")
            return None

    async def _list_all_tenants(self) -> List[Tenant]:
        """List all active tenants"""
        try:
            import json

            tenants = []
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT id, name, slug, domain, is_active, created_at, settings, limits
                    FROM tenants WHERE is_active = 1
                    ORDER BY created_at DESC
                """
                )

                for row in cursor.fetchall():
                    tenant = Tenant(
                        id=row[0],
                        name=row[1],
                        slug=row[2],
                        domain=row[3],
                        is_active=bool(row[4]),
                        created_at=datetime.fromisoformat(row[5]) if row[5] else None,
                        settings=json.loads(row[6] or "{}"),
                        limits=json.loads(row[7] or "{}"),
                    )
                    tenants.append(tenant)

            return tenants

        except Exception as e:
            logger.error(f"Failed to list tenants: {e}")
            return []

    async def update_tenant(
        self, tenant_id: int, updates: Dict[str, Any]
    ) -> Optional[Tenant]:
        """Update tenant information"""
        try:
            import json

            # Build update query dynamically
            set_clauses = []
            params = []

            for key, value in updates.items():
                if key in ["name", "slug", "domain", "is_active"]:
                    set_clauses.append(f"{key} = ?")
                    params.append(value)
                elif key in ["settings", "limits"]:
                    set_clauses.append(f"{key} = ?")
                    params.append(json.dumps(value))

            if not set_clauses:
                raise ValueError("No valid update fields provided")

            params.append(tenant_id)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    f"""
                    UPDATE tenants 
                    SET {', '.join(set_clauses)}
                    WHERE id = ?
                """,
                    params,
                )

                if conn.total_changes == 0:
                    return None

            # Return updated tenant
            return await self.get_by_id(tenant_id)

        except Exception as e:
            logger.error(f"Failed to update tenant {tenant_id}: {e}")
            raise

    async def delete_tenant(self, tenant_id: int) -> bool:
        """Soft delete tenant (mark as inactive)"""
        try:
            if tenant_id == 1:
                raise ValueError("Cannot delete default tenant")

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE tenants SET is_active = 0 WHERE id = ?
                """,
                    (tenant_id,),
                )

                return conn.total_changes > 0

        except Exception as e:
            logger.error(f"Failed to delete tenant {tenant_id}: {e}")
            return False

    async def get_tenant_stats(self, tenant_id: int) -> Dict[str, Any]:
        """Get usage statistics for a tenant"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Document count
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) FROM documents WHERE tenant_id = ?
                """,
                    (tenant_id,),
                )
                doc_count = cursor.fetchone()[0] or 0

                # Total storage size
                cursor = conn.execute(
                    """
                    SELECT SUM(file_size) FROM documents WHERE tenant_id = ?
                """,
                    (tenant_id,),
                )
                storage_size = cursor.fetchone()[0] or 0

                # User count
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) FROM users WHERE tenant_id = ? AND is_active = 1
                """,
                    (tenant_id,),
                )
                user_count = cursor.fetchone()[0] or 0

                # Query count (last 30 days)
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) FROM query_logs 
                    WHERE tenant_id = ? AND timestamp > datetime('now', '-30 days')
                """,
                    (tenant_id,),
                )
                query_count = cursor.fetchone()[0] or 0

                return {
                    "document_count": doc_count,
                    "storage_size_bytes": storage_size,
                    "user_count": user_count,
                    "query_count_30d": query_count,
                    "storage_size_mb": round(storage_size / (1024 * 1024), 2),
                }

        except Exception as e:
            logger.error(f"Failed to get tenant stats for {tenant_id}: {e}")
            return {}

    # Abstract method implementations
    async def create(self, entity: Tenant) -> Tenant:
        """Create a new tenant (implements BaseRepository.create)"""
        return await self.create_tenant(entity)

    async def update(self, entity_id: int, updates: Dict[str, Any]) -> Optional[Tenant]:
        """Update tenant (implements BaseRepository.update)"""
        return await self.update_tenant(entity_id, updates)

    async def delete(self, entity_id: int) -> bool:
        """Delete tenant (implements BaseRepository.delete)"""
        return await self.delete_tenant(entity_id)

    async def list_all(self, options: Optional[Any] = None) -> Any:
        """List all tenants (implements BaseRepository.list_all)"""
        from .base import QueryResult
        tenants = await self._list_all_tenants()
        return QueryResult(items=tenants, total_count=len(tenants))

    async def exists(self, entity_id: int) -> bool:
        """Check if tenant exists"""
        tenant = await self.get_by_id(entity_id)
        return tenant is not None

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count tenants"""
        tenants = await self._list_all_tenants()
        return len(tenants)
