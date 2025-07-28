#!/usr/bin/env python3
"""
Document Management Service
Provides comprehensive CRUD operations for document management
"""

import os
import json
import shutil
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import hashlib
import tempfile
from enum import Enum

logger = logging.getLogger(__name__)

class DocumentStatus(Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"
    DELETED = "deleted"

@dataclass
class DocumentMetadata:
    """Document metadata model"""
    id: int
    filename: str
    original_filename: str
    file_type: str
    content_type: str
    file_size: int
    file_hash: str
    status: DocumentStatus
    upload_date: datetime
    last_modified: datetime
    chunks_count: int
    processing_time: float
    tags: List[str]
    description: Optional[str] = None
    uploader: Optional[str] = None
    file_path: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "file_type": self.file_type,
            "content_type": self.content_type,
            "file_size": self.file_size,
            "file_hash": self.file_hash,
            "status": self.status.value,
            "upload_date": self.upload_date.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "chunks_count": self.chunks_count,
            "processing_time": self.processing_time,
            "tags": self.tags,
            "description": self.description,
            "uploader": self.uploader,
            "file_path": self.file_path,
            "error_message": self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentMetadata':
        """Create from dictionary"""
        return cls(
            id=data["id"],
            filename=data["filename"],
            original_filename=data["original_filename"],
            file_type=data["file_type"],
            content_type=data["content_type"],
            file_size=data["file_size"],
            file_hash=data["file_hash"],
            status=DocumentStatus(data["status"]),
            upload_date=datetime.fromisoformat(data["upload_date"]),
            last_modified=datetime.fromisoformat(data["last_modified"]),
            chunks_count=data["chunks_count"],
            processing_time=data["processing_time"],
            tags=data["tags"],
            description=data.get("description"),
            uploader=data.get("uploader"),
            file_path=data.get("file_path"),
            error_message=data.get("error_message")
        )

@dataclass
class DocumentChunk:
    """Document chunk model"""
    chunk_id: int
    document_id: int
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "created_at": self.created_at.isoformat()
        }

class DocumentManager:
    """
    Comprehensive document management system with CRUD operations
    Handles document storage, metadata management, and chunk tracking
    """
    
    def __init__(self, storage_dir: str = "./storage", 
                 metadata_file: str = "documents_metadata.json"):
        self.storage_dir = Path(storage_dir)
        self.documents_dir = self.storage_dir / "documents"
        self.metadata_file = self.storage_dir / metadata_file
        
        # Create directories
        self.storage_dir.mkdir(exist_ok=True)
        self.documents_dir.mkdir(exist_ok=True)
        
        # In-memory storage (in production, use database)
        self.documents: Dict[int, DocumentMetadata] = {}
        self.chunks: Dict[int, List[DocumentChunk]] = {}
        self.next_doc_id = 1
        self.next_chunk_id = 1
        
        # Load existing metadata
        self._load_metadata()
        
        logger.info(f"Document manager initialized with storage: {self.storage_dir}")
    
    def _load_metadata(self):
        """Load document metadata from file"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Load documents
                    for doc_data in data.get("documents", []):
                        doc = DocumentMetadata.from_dict(doc_data)
                        self.documents[doc.id] = doc
                        self.next_doc_id = max(self.next_doc_id, doc.id + 1)
                    
                    # Load chunks
                    for chunk_data in data.get("chunks", []):
                        chunk = DocumentChunk(
                            chunk_id=chunk_data["chunk_id"],
                            document_id=chunk_data["document_id"],
                            text=chunk_data["text"],
                            chunk_index=chunk_data["chunk_index"],
                            start_char=chunk_data["start_char"],
                            end_char=chunk_data["end_char"],
                            created_at=datetime.fromisoformat(chunk_data["created_at"])
                        )
                        
                        if chunk.document_id not in self.chunks:
                            self.chunks[chunk.document_id] = []
                        self.chunks[chunk.document_id].append(chunk)
                        self.next_chunk_id = max(self.next_chunk_id, chunk.chunk_id + 1)
                    
                    logger.info(f"Loaded {len(self.documents)} documents and {sum(len(chunks) for chunks in self.chunks.values())} chunks")
        
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
    
    def _save_metadata(self):
        """Save document metadata to file"""
        try:
            data = {
                "documents": [doc.to_dict() for doc in self.documents.values()],
                "chunks": []
            }
            
            # Flatten chunks
            for doc_chunks in self.chunks.values():
                data["chunks"].extend([chunk.to_dict() for chunk in doc_chunks])
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Could not save metadata: {e}")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate file hash: {e}")
            return ""
    
    def create_document(self, filename: str, original_filename: str, 
                       file_type: str, content_type: str, file_size: int,
                       file_path: str, uploader: Optional[str] = None,
                       description: Optional[str] = None,
                       tags: Optional[List[str]] = None) -> DocumentMetadata:
        """Create a new document entry"""
        
        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)
        
        # Check for duplicates based on hash
        for doc in self.documents.values():
            if doc.file_hash == file_hash and doc.status != DocumentStatus.DELETED:
                logger.warning(f"Duplicate document detected: {original_filename}")
                raise ValueError(f"Document with identical content already exists (ID: {doc.id})")
        
        # Create document metadata
        doc_id = self.next_doc_id
        self.next_doc_id += 1
        
        # Move file to managed storage
        storage_path = self.documents_dir / f"{doc_id}_{filename}"
        try:
            shutil.copy2(file_path, storage_path)
        except Exception as e:
            logger.error(f"Failed to move file to storage: {e}")
            raise ValueError(f"Failed to store document: {e}")
        
        document = DocumentMetadata(
            id=doc_id,
            filename=filename,
            original_filename=original_filename,
            file_type=file_type,
            content_type=content_type,
            file_size=file_size,
            file_hash=file_hash,
            status=DocumentStatus.UPLOADING,
            upload_date=datetime.now(),
            last_modified=datetime.now(),
            chunks_count=0,
            processing_time=0.0,
            tags=tags or [],
            description=description,
            uploader=uploader,
            file_path=str(storage_path)
        )
        
        self.documents[doc_id] = document
        self.chunks[doc_id] = []
        self._save_metadata()
        
        logger.info(f"Created document: {original_filename} (ID: {doc_id})")
        return document
    
    def get_document(self, doc_id: int) -> Optional[DocumentMetadata]:
        """Get document by ID"""
        return self.documents.get(doc_id)
    
    def get_document_by_filename(self, filename: str) -> Optional[DocumentMetadata]:
        """Get document by filename"""
        for doc in self.documents.values():
            if doc.filename == filename or doc.original_filename == filename:
                return doc
        return None
    
    def list_documents(self, status: Optional[DocumentStatus] = None,
                      tags: Optional[List[str]] = None,
                      uploader: Optional[str] = None,
                      limit: int = 100,
                      offset: int = 0) -> Tuple[List[DocumentMetadata], int]:
        """List documents with filtering options"""
        
        # Filter documents
        filtered_docs = []
        for doc in self.documents.values():
            # Status filter
            if status and doc.status != status:
                continue
            
            # Tags filter
            if tags and not any(tag in doc.tags for tag in tags):
                continue
            
            # Uploader filter
            if uploader and doc.uploader != uploader:
                continue
            
            filtered_docs.append(doc)
        
        # Sort by upload date (newest first)
        filtered_docs.sort(key=lambda x: x.upload_date, reverse=True)
        
        # Pagination
        total = len(filtered_docs)
        paginated_docs = filtered_docs[offset:offset + limit]
        
        return paginated_docs, total
    
    def update_document(self, doc_id: int, **updates) -> Optional[DocumentMetadata]:
        """Update document metadata"""
        document = self.documents.get(doc_id)
        if not document:
            return None
        
        # Update allowed fields
        allowed_updates = ['description', 'tags', 'status', 'error_message', 'chunks_count', 'processing_time']
        updated = False
        
        for field, value in updates.items():
            if field in allowed_updates and hasattr(document, field):
                setattr(document, field, value)
                updated = True
        
        if updated:
            document.last_modified = datetime.now()
            self._save_metadata()
            logger.info(f"Updated document {doc_id}: {list(updates.keys())}")
        
        return document
    
    def delete_document(self, doc_id: int, permanent: bool = False) -> bool:
        """Delete document (soft delete by default)"""
        document = self.documents.get(doc_id)
        if not document:
            return False
        
        try:
            if permanent:
                # Delete file
                if document.file_path and os.path.exists(document.file_path):
                    os.remove(document.file_path)
                
                # Remove from memory
                del self.documents[doc_id]
                if doc_id in self.chunks:
                    del self.chunks[doc_id]
                
                logger.info(f"Permanently deleted document {doc_id}")
            else:
                # Soft delete
                document.status = DocumentStatus.DELETED
                document.last_modified = datetime.now()
                logger.info(f"Soft deleted document {doc_id}")
            
            self._save_metadata()
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def restore_document(self, doc_id: int) -> bool:
        """Restore soft-deleted document"""
        document = self.documents.get(doc_id)
        if not document or document.status != DocumentStatus.DELETED:
            return False
        
        document.status = DocumentStatus.COMPLETED
        document.last_modified = datetime.now()
        self._save_metadata()
        
        logger.info(f"Restored document {doc_id}")
        return True
    
    def add_chunks(self, doc_id: int, text_chunks: List[str]) -> List[DocumentChunk]:
        """Add text chunks for a document"""
        if doc_id not in self.documents:
            raise ValueError(f"Document {doc_id} not found")
        
        document = self.documents[doc_id]
        chunks = []
        
        current_char = 0
        for i, text in enumerate(text_chunks):
            chunk = DocumentChunk(
                chunk_id=self.next_chunk_id,
                document_id=doc_id,
                text=text,
                chunk_index=i,
                start_char=current_char,
                end_char=current_char + len(text),
                created_at=datetime.now()
            )
            
            chunks.append(chunk)
            self.next_chunk_id += 1
            current_char += len(text)
        
        # Update document chunks
        if doc_id not in self.chunks:
            self.chunks[doc_id] = []
        
        self.chunks[doc_id].extend(chunks)
        
        # Update document metadata
        document.chunks_count = len(self.chunks[doc_id])
        document.last_modified = datetime.now()
        
        self._save_metadata()
        
        logger.info(f"Added {len(chunks)} chunks to document {doc_id}")
        return chunks
    
    def get_document_chunks(self, doc_id: int) -> List[DocumentChunk]:
        """Get all chunks for a document"""
        return self.chunks.get(doc_id, [])
    
    def search_documents(self, query: str, search_content: bool = False) -> List[DocumentMetadata]:
        """Search documents by filename, description, or content"""
        query_lower = query.lower()
        results = []
        
        for doc in self.documents.values():
            if doc.status == DocumentStatus.DELETED:
                continue
            
            # Search in metadata
            if (query_lower in doc.filename.lower() or 
                query_lower in doc.original_filename.lower() or
                (doc.description and query_lower in doc.description.lower()) or
                any(query_lower in tag.lower() for tag in doc.tags)):
                results.append(doc)
                continue
            
            # Search in content if requested
            if search_content and doc.id in self.chunks:
                for chunk in self.chunks[doc.id]:
                    if query_lower in chunk.text.lower():
                        results.append(doc)
                        break
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get document management statistics"""
        total_docs = len(self.documents)
        total_chunks = sum(len(chunks) for chunks in self.chunks.values())
        total_size = sum(doc.file_size for doc in self.documents.values() 
                        if doc.status != DocumentStatus.DELETED)
        
        status_counts = {}
        for status in DocumentStatus:
            status_counts[status.value] = sum(1 for doc in self.documents.values() 
                                             if doc.status == status)
        
        # File type distribution
        file_types = {}
        for doc in self.documents.values():
            if doc.status != DocumentStatus.DELETED:
                file_types[doc.file_type] = file_types.get(doc.file_type, 0) + 1
        
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "total_size_bytes": total_size,
            "status_distribution": status_counts,
            "file_type_distribution": file_types,
            "storage_path": str(self.storage_dir),
            "documents_in_storage": len([f for f in self.documents_dir.iterdir() if f.is_file()])
        }
    
    def cleanup_orphaned_files(self) -> int:
        """Clean up orphaned files in storage directory"""
        if not self.documents_dir.exists():
            return 0
        
        # Get all valid file paths
        valid_paths = set()
        for doc in self.documents.values():
            if doc.file_path and doc.status != DocumentStatus.DELETED:
                valid_paths.add(Path(doc.file_path).name)
        
        # Find orphaned files
        orphaned_count = 0
        for file_path in self.documents_dir.iterdir():
            if file_path.is_file() and file_path.name not in valid_paths:
                try:
                    file_path.unlink()
                    orphaned_count += 1
                    logger.info(f"Removed orphaned file: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Could not remove orphaned file {file_path.name}: {e}")
        
        return orphaned_count
    
    def export_metadata(self, export_path: str) -> bool:
        """Export document metadata to JSON file"""
        try:
            export_data = {
                "export_date": datetime.now().isoformat(),
                "statistics": self.get_statistics(),
                "documents": [doc.to_dict() for doc in self.documents.values()],
                "chunks_summary": {
                    doc_id: len(chunks) for doc_id, chunks in self.chunks.items()
                }
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported metadata to: {export_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to export metadata: {e}")
            return False

# Global document manager instance
document_manager: Optional[DocumentManager] = None

def get_document_manager() -> DocumentManager:
    """Get global document manager instance"""
    global document_manager
    if document_manager is None:
        document_manager = DocumentManager()
    return document_manager

def init_document_manager(storage_dir: str = "./storage") -> DocumentManager:
    """Initialize document manager with custom storage directory"""
    global document_manager
    document_manager = DocumentManager(storage_dir)
    return document_manager