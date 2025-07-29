"""
Backup and Recovery Service
Handles automated backups of databases and documents
"""

import asyncio
import gzip
import json
import logging
import os
import shutil
import sqlite3
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BackupService:
    """Handles backup and recovery operations"""

    def __init__(
        self,
        backup_dir: str = "backups",
        retention_days: int = 30,
        compress_backups: bool = True,
    ):
        """Initialize backup service"""
        self.backup_dir = Path(backup_dir)
        self.retention_days = retention_days
        self.compress_backups = compress_backups

        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Backup service initialized: {self.backup_dir} (retention: {retention_days} days)"
        )

    async def create_full_backup(self) -> Dict[str, Any]:
        """Create a full system backup"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"rag_backup_{timestamp}"
        backup_path = self.backup_dir / backup_name

        try:
            backup_path.mkdir(parents=True, exist_ok=True)

            backup_info = {
                "timestamp": timestamp,
                "backup_name": backup_name,
                "backup_path": str(backup_path),
                "components": {},
                "success": False,
            }

            # Backup databases
            db_backup = await self._backup_databases(backup_path)
            backup_info["components"]["databases"] = db_backup

            # Backup documents
            docs_backup = await self._backup_documents(backup_path)
            backup_info["components"]["documents"] = docs_backup

            # Backup configuration
            config_backup = await self._backup_configuration(backup_path)
            backup_info["components"]["configuration"] = config_backup

            # Backup vector indices
            vector_backup = await self._backup_vector_indices(backup_path)
            backup_info["components"]["vector_indices"] = vector_backup

            # Create backup manifest
            await self._create_backup_manifest(backup_path, backup_info)

            # Compress backup if enabled
            if self.compress_backups:
                compressed_path = await self._compress_backup(backup_path)
                backup_info["compressed_path"] = str(compressed_path)
                backup_info["compressed"] = True

            backup_info["success"] = True
            logger.info(f"Full backup completed: {backup_name}")

            return backup_info

        except Exception as e:
            logger.error(f"Full backup failed: {e}")
            backup_info["success"] = False
            backup_info["error"] = str(e)
            return backup_info

    async def _backup_databases(self, backup_path: Path) -> Dict[str, Any]:
        """Backup database files"""
        db_backup_dir = backup_path / "databases"
        db_backup_dir.mkdir(parents=True, exist_ok=True)

        backed_up = []
        errors = []

        # Common database files to backup
        db_files = ["data/rag_database.db", "data/audit.db", "test.db", "test_audit.db"]

        for db_file in db_files:
            db_path = Path(db_file)
            if db_path.exists():
                try:
                    # For SQLite, create a consistent backup
                    if db_file.endswith(".db"):
                        backup_file = db_backup_dir / db_path.name
                        await self._backup_sqlite_db(db_path, backup_file)
                        backed_up.append(str(db_path))
                    else:
                        # Simple file copy for other files
                        shutil.copy2(db_path, db_backup_dir)
                        backed_up.append(str(db_path))

                except Exception as e:
                    logger.error(f"Failed to backup database {db_file}: {e}")
                    errors.append({"file": db_file, "error": str(e)})

        return {"backed_up": backed_up, "errors": errors, "count": len(backed_up)}

    async def _backup_sqlite_db(self, source_db: Path, backup_file: Path):
        """Create a consistent SQLite backup"""
        try:
            # Use SQLite's backup API for consistency
            source_conn = sqlite3.connect(str(source_db))
            backup_conn = sqlite3.connect(str(backup_file))

            source_conn.backup(backup_conn)

            source_conn.close()
            backup_conn.close()

            logger.debug(f"SQLite backup completed: {source_db} -> {backup_file}")

        except Exception as e:
            logger.error(f"SQLite backup failed: {e}")
            # Fallback to file copy
            shutil.copy2(source_db, backup_file)

    async def _backup_documents(self, backup_path: Path) -> Dict[str, Any]:
        """Backup document files"""
        docs_backup_dir = backup_path / "documents"
        docs_backup_dir.mkdir(parents=True, exist_ok=True)

        backed_up = []
        errors = []
        total_size = 0

        # Document directories to backup
        doc_dirs = [
            "data/storage/uploads",
            "data/storage/processed",
            "data/storage/documents",
        ]

        for doc_dir in doc_dirs:
            doc_path = Path(doc_dir)
            if doc_path.exists() and doc_path.is_dir():
                try:
                    target_dir = docs_backup_dir / doc_path.name

                    # Copy directory tree
                    if doc_path.exists():
                        shutil.copytree(doc_path, target_dir, dirs_exist_ok=True)

                        # Calculate size
                        dir_size = sum(
                            f.stat().st_size
                            for f in target_dir.rglob("*")
                            if f.is_file()
                        )
                        total_size += dir_size

                        backed_up.append(
                            {
                                "directory": str(doc_path),
                                "size_bytes": dir_size,
                                "file_count": len(list(target_dir.rglob("*"))),
                            }
                        )

                except Exception as e:
                    logger.error(f"Failed to backup documents from {doc_dir}: {e}")
                    errors.append({"directory": doc_dir, "error": str(e)})

        return {
            "backed_up": backed_up,
            "errors": errors,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }

    async def _backup_configuration(self, backup_path: Path) -> Dict[str, Any]:
        """Backup configuration files"""
        config_backup_dir = backup_path / "configuration"
        config_backup_dir.mkdir(parents=True, exist_ok=True)

        backed_up = []
        errors = []

        # Configuration files to backup
        config_files = [
            "config/llm_config.yaml",
            "config/production_config.yaml",
            ".env",
            "CLAUDE.md",
            "requirements.txt",
            "deployment/docker-compose.yml",
        ]

        for config_file in config_files:
            config_path = Path(config_file)
            if config_path.exists():
                try:
                    target_path = config_backup_dir / config_path.name
                    shutil.copy2(config_path, target_path)
                    backed_up.append(str(config_path))
                except Exception as e:
                    logger.error(f"Failed to backup config {config_file}: {e}")
                    errors.append({"file": config_file, "error": str(e)})

        return {"backed_up": backed_up, "errors": errors, "count": len(backed_up)}

    async def _backup_vector_indices(self, backup_path: Path) -> Dict[str, Any]:
        """Backup vector search indices"""
        vector_backup_dir = backup_path / "vector_indices"
        vector_backup_dir.mkdir(parents=True, exist_ok=True)

        backed_up = []
        errors = []

        # Vector index files to backup
        vector_files = [
            "data/faiss_index.bin",
            "data/vector_cache",
            "data/embeddings_cache",
        ]

        for vector_file in vector_files:
            vector_path = Path(vector_file)
            if vector_path.exists():
                try:
                    if vector_path.is_file():
                        shutil.copy2(vector_path, vector_backup_dir)
                    elif vector_path.is_dir():
                        target_dir = vector_backup_dir / vector_path.name
                        shutil.copytree(vector_path, target_dir, dirs_exist_ok=True)

                    backed_up.append(str(vector_path))
                except Exception as e:
                    logger.error(f"Failed to backup vector index {vector_file}: {e}")
                    errors.append({"file": vector_file, "error": str(e)})

        return {"backed_up": backed_up, "errors": errors, "count": len(backed_up)}

    async def _create_backup_manifest(
        self, backup_path: Path, backup_info: Dict[str, Any]
    ):
        """Create backup manifest with metadata"""
        manifest = {
            "backup_info": backup_info,
            "system_info": {
                "python_version": os.sys.version,
                "platform": os.name,
                "backup_service_version": "1.0.0",
            },
            "restore_instructions": {
                "databases": "Copy .db files to data/ directory",
                "documents": "Copy document directories to data/storage/",
                "configuration": "Copy config files to their original locations",
                "vector_indices": "Copy vector index files to data/ directory",
            },
        }

        manifest_file = backup_path / "backup_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        logger.debug(f"Backup manifest created: {manifest_file}")

    async def _compress_backup(self, backup_path: Path) -> Path:
        """Compress backup directory"""
        try:
            # Create tar.gz archive
            archive_path = backup_path.with_suffix(".tar.gz")

            # Use shutil.make_archive for cross-platform compatibility
            shutil.make_archive(
                str(backup_path),
                "gztar",
                root_dir=backup_path.parent,
                base_dir=backup_path.name,
            )

            # Remove original directory
            shutil.rmtree(backup_path)

            logger.info(f"Backup compressed: {archive_path}")
            return archive_path

        except Exception as e:
            logger.error(f"Backup compression failed: {e}")
            return backup_path

    async def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = []

        try:
            for item in self.backup_dir.iterdir():
                if item.is_dir() and item.name.startswith("rag_backup_"):
                    # Uncompressed backup
                    manifest_file = item / "backup_manifest.json"
                    if manifest_file.exists():
                        with open(manifest_file, "r") as f:
                            manifest = json.load(f)

                        backups.append(
                            {
                                "name": item.name,
                                "path": str(item),
                                "timestamp": manifest.get("backup_info", {}).get(
                                    "timestamp"
                                ),
                                "compressed": False,
                                "size_bytes": self._get_directory_size(item),
                                "manifest": manifest,
                            }
                        )

                elif (
                    item.is_file()
                    and item.name.startswith("rag_backup_")
                    and item.suffix == ".gz"
                ):
                    # Compressed backup
                    backups.append(
                        {
                            "name": item.stem,
                            "path": str(item),
                            "timestamp": (
                                item.stem.split("_")[-1] if "_" in item.stem else None
                            ),
                            "compressed": True,
                            "size_bytes": item.stat().st_size,
                            "manifest": None,
                        }
                    )

            # Sort by timestamp (newest first)
            backups.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        except Exception as e:
            logger.error(f"Failed to list backups: {e}")

        return backups

    def _get_directory_size(self, directory: Path) -> int:
        """Calculate directory size recursively"""
        try:
            return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
        except Exception:
            return 0

    async def restore_backup(
        self, backup_name: str, components: List[str] = None
    ) -> Dict[str, Any]:
        """Restore from backup"""
        if components is None:
            components = ["databases", "documents", "configuration", "vector_indices"]

        backup_path = None
        compressed = False

        # Find backup
        for item in self.backup_dir.iterdir():
            if item.name == backup_name or item.stem == backup_name:
                backup_path = item
                compressed = item.suffix == ".gz"
                break

        if not backup_path:
            return {"success": False, "error": f"Backup {backup_name} not found"}

        try:
            restore_info = {
                "backup_name": backup_name,
                "components_restored": [],
                "errors": [],
                "success": False,
            }

            # Extract if compressed
            if compressed:
                extract_path = backup_path.parent / backup_path.stem
                shutil.unpack_archive(backup_path, extract_path)
                backup_path = extract_path

            # Read manifest
            manifest_file = backup_path / "backup_manifest.json"
            if manifest_file.exists():
                with open(manifest_file, "r") as f:
                    manifest = json.load(f)
            else:
                logger.warning(
                    "No backup manifest found, proceeding with best effort restore"
                )
                manifest = {}

            # Restore components
            for component in components:
                try:
                    if component == "databases":
                        await self._restore_databases(backup_path)
                    elif component == "documents":
                        await self._restore_documents(backup_path)
                    elif component == "configuration":
                        await self._restore_configuration(backup_path)
                    elif component == "vector_indices":
                        await self._restore_vector_indices(backup_path)

                    restore_info["components_restored"].append(component)

                except Exception as e:
                    logger.error(f"Failed to restore {component}: {e}")
                    restore_info["errors"].append(
                        {"component": component, "error": str(e)}
                    )

            # Cleanup extracted files if it was compressed
            if compressed and backup_path.exists():
                shutil.rmtree(backup_path)

            restore_info["success"] = len(restore_info["errors"]) == 0
            logger.info(
                f"Restore completed: {backup_name} (success: {restore_info['success']})"
            )

            return restore_info

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return {"success": False, "error": str(e)}

    async def _restore_databases(self, backup_path: Path):
        """Restore database files"""
        db_backup_dir = backup_path / "databases"
        if not db_backup_dir.exists():
            return

        for db_file in db_backup_dir.iterdir():
            if db_file.is_file():
                target_path = Path("data") / db_file.name
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(db_file, target_path)
                logger.info(f"Restored database: {target_path}")

    async def _restore_documents(self, backup_path: Path):
        """Restore document files"""
        docs_backup_dir = backup_path / "documents"
        if not docs_backup_dir.exists():
            return

        for doc_dir in docs_backup_dir.iterdir():
            if doc_dir.is_dir():
                target_path = Path("data/storage") / doc_dir.name
                target_path.parent.mkdir(parents=True, exist_ok=True)

                if target_path.exists():
                    shutil.rmtree(target_path)

                shutil.copytree(doc_dir, target_path)
                logger.info(f"Restored documents: {target_path}")

    async def _restore_configuration(self, backup_path: Path):
        """Restore configuration files"""
        config_backup_dir = backup_path / "configuration"
        if not config_backup_dir.exists():
            return

        for config_file in config_backup_dir.iterdir():
            if config_file.is_file():
                # Map files to their original locations
                if config_file.name.endswith(".yaml"):
                    target_path = Path("config") / config_file.name
                elif config_file.name == ".env":
                    target_path = Path(".env")
                elif config_file.name == "docker-compose.yml":
                    target_path = Path("deployment") / config_file.name
                else:
                    target_path = Path(config_file.name)

                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(config_file, target_path)
                logger.info(f"Restored config: {target_path}")

    async def _restore_vector_indices(self, backup_path: Path):
        """Restore vector index files"""
        vector_backup_dir = backup_path / "vector_indices"
        if not vector_backup_dir.exists():
            return

        for vector_item in vector_backup_dir.iterdir():
            target_path = Path("data") / vector_item.name
            target_path.parent.mkdir(parents=True, exist_ok=True)

            if vector_item.is_file():
                shutil.copy2(vector_item, target_path)
            elif vector_item.is_dir():
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.copytree(vector_item, target_path)

            logger.info(f"Restored vector index: {target_path}")

    async def cleanup_old_backups(self) -> Dict[str, Any]:
        """Remove old backups based on retention policy"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        deleted = []
        errors = []

        try:
            for item in self.backup_dir.iterdir():
                if item.name.startswith("rag_backup_"):
                    # Extract timestamp from filename
                    timestamp_str = item.name.split("_")[-1].replace(".tar.gz", "")
                    try:
                        backup_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                        if backup_date < cutoff_date:
                            if item.is_dir():
                                shutil.rmtree(item)
                            else:
                                item.unlink()

                            deleted.append(
                                {
                                    "name": item.name,
                                    "date": backup_date.isoformat(),
                                    "age_days": (datetime.now() - backup_date).days,
                                }
                            )

                    except ValueError as e:
                        logger.warning(
                            f"Could not parse backup date from {item.name}: {e}"
                        )
                        errors.append({"file": item.name, "error": str(e)})

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            errors.append({"error": str(e)})

        logger.info(
            f"Backup cleanup completed: {len(deleted)} deleted, {len(errors)} errors"
        )

        return {
            "deleted": deleted,
            "errors": errors,
            "retention_days": self.retention_days,
        }

    async def get_backup_stats(self) -> Dict[str, Any]:
        """Get backup service statistics"""
        backups = await self.list_backups()

        total_size = sum(backup.get("size_bytes", 0) for backup in backups)

        return {
            "backup_directory": str(self.backup_dir),
            "total_backups": len(backups),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "retention_days": self.retention_days,
            "compression_enabled": self.compress_backups,
            "latest_backup": backups[0] if backups else None,
            "oldest_backup": backups[-1] if backups else None,
        }
