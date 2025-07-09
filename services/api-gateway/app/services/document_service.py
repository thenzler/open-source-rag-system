"""
Document service implementation with all required methods.
"""

import asyncio
import logging
import uuid
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from fastapi import UploadFile
import aiofiles
import hashlib
import os
from datetime import datetime

from app.models.documents import Document, DocumentChunk
from app.core.config import get_settings
from app.core.exceptions import DocumentNotFoundError, ProcessingError, ValidationError

logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentService:
    """Document service for handling document operations."""
    
    def __init__(self):
        self.initialized = False
        self.upload_dir = settings.upload_directory
        
    async def initialize(self):
        """Initialize document service."""
        logger.info("Initializing Document Service")
        
        # Create upload directory if it doesn't exist
        os.makedirs(self.upload_dir, exist_ok=True)
        
        # Initialize any required connections or services
        # This is where you'd initialize vector database connections, etc.
        
        self.initialized = True
        logger.info("Document Service initialized successfully")
        
    async def health_check(self) -> bool:
        """Check service health."""
        try:
            # Check if service is initialized
            if not self.initialized:
                return False
                
            # Check if upload directory exists and is writable
            if not os.path.exists(self.upload_dir):
                return False
                
            # Add more health checks as needed
            return True
            
        except Exception as e:
            logger.error(f"Document service health check failed: {e}")
            return False
    
    async def upload_document(
        self,
        file: UploadFile,
        metadata: Optional[str] = None,
        user_id: str = "anonymous",
        db: AsyncSession = None
    ) -> Document:
        """Upload a new document."""
        try:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            file_extension = os.path.splitext(file.filename)[1]
            filename = f"{file_id}{file_extension}"
            file_path = os.path.join(self.upload_dir, filename)
            
            # Calculate file hash
            file_content = await file.read()
            file_hash = hashlib.sha256(file_content).hexdigest()
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_content)
            
            # Create document record
            document = Document(
                id=uuid.uuid4(),
                filename=filename,
                original_filename=file.filename,
                content_type=file.content_type,
                file_size=len(file_content),
                file_path=file_path,
                file_hash=file_hash,
                user_id=user_id,
                status="pending",
                metadata={"uploaded_metadata": metadata} if metadata else {},
                created_at=datetime.utcnow()
            )
            
            if db:
                db.add(document)
                await db.commit()
                await db.refresh(document)
            
            return document
            
        except Exception as e:
            logger.error(f"Document upload failed: {e}")
            raise ProcessingError(f"Failed to upload document: {e}")
    
    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 50,
        status: Optional[str] = None,
        category: Optional[str] = None,
        search: Optional[str] = None,
        user_id: str = "anonymous",
        db: AsyncSession = None
    ) -> Tuple[List[Document], int]:
        """List documents with filters."""
        try:
            if not db:
                return [], 0
                
            query = select(Document).where(Document.user_id == user_id)
            
            if status:
                query = query.where(Document.status == status)
            if category:
                query = query.where(Document.category == category)
            if search:
                query = query.where(Document.title.ilike(f"%{search}%"))
            
            # Get total count
            count_query = select(func.count(Document.id)).where(Document.user_id == user_id)
            if status:
                count_query = count_query.where(Document.status == status)
            if category:
                count_query = count_query.where(Document.category == category)
            if search:
                count_query = count_query.where(Document.title.ilike(f"%{search}%"))
            
            total_result = await db.execute(count_query)
            total = total_result.scalar()
            
            # Get documents
            query = query.offset(skip).limit(limit).order_by(Document.created_at.desc())
            result = await db.execute(query)
            documents = result.scalars().all()
            
            return documents, total
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise ProcessingError(f"Failed to list documents: {e}")
    
    async def get_document(
        self,
        document_id: str,
        user_id: str = "anonymous",
        db: AsyncSession = None
    ) -> Optional[Document]:
        """Get a specific document."""
        try:
            if not db:
                return None
                
            query = select(Document).where(
                and_(
                    Document.id == document_id,
                    Document.user_id == user_id
                )
            )
            result = await db.execute(query)
            document = result.scalar_one_or_none()
            
            if not document:
                raise DocumentNotFoundError(f"Document {document_id} not found")
                
            return document
            
        except DocumentNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get document: {e}")
            raise ProcessingError(f"Failed to get document: {e}")
    
    async def get_processing_status(
        self,
        document_id: str,
        user_id: str = "anonymous",
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """Get document processing status."""
        try:
            document = await self.get_document(document_id, user_id, db)
            
            return {
                "document_id": str(document.id),
                "status": document.status,
                "progress": document.processing_progress,
                "error_message": document.error_message,
                "created_at": document.created_at,
                "processed_at": document.processed_at
            }
            
        except Exception as e:
            logger.error(f"Failed to get processing status: {e}")
            raise ProcessingError(f"Failed to get processing status: {e}")
    
    async def delete_document(
        self,
        document_id: str,
        user_id: str = "anonymous",
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """Delete a document and all associated data."""
        try:
            document = await self.get_document(document_id, user_id, db)
            
            if not document:
                raise DocumentNotFoundError(f"Document {document_id} not found")
            
            # Count chunks before deletion
            chunk_count_query = select(func.count(DocumentChunk.id)).where(
                DocumentChunk.document_id == document_id
            )
            chunk_result = await db.execute(chunk_count_query)
            chunk_count = chunk_result.scalar()
            
            # Delete file if it exists
            if document.file_path and os.path.exists(document.file_path):
                try:
                    os.remove(document.file_path)
                except Exception as e:
                    logger.warning(f"Failed to remove file {document.file_path}: {e}")
            
            # Delete database records (chunks will be deleted by cascade)
            await db.delete(document)
            await db.commit()
            
            return {
                "deleted_chunks": chunk_count,
                "deleted_vectors": chunk_count,  # Assuming 1:1 mapping
                "file_deleted": True
            }
            
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            raise ProcessingError(f"Failed to delete document: {e}")
    
    async def process_document_async(
        self,
        document_id: str,
        db: AsyncSession = None
    ):
        """Process document asynchronously (background task)."""
        try:
            if not db:
                return
                
            # Get document
            query = select(Document).where(Document.id == document_id)
            result = await db.execute(query)
            document = result.scalar_one_or_none()
            
            if not document:
                return
            
            # Update status to processing
            document.status = "processing"
            document.processing_progress = 0.1
            await db.commit()
            
            # Simulate processing steps
            await asyncio.sleep(1)
            document.processing_progress = 0.5
            await db.commit()
            
            # Mark as completed
            document.status = "completed"
            document.processing_progress = 1.0
            document.processed_at = datetime.utcnow()
            await db.commit()
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            if db and document:
                document.status = "failed"
                document.error_message = str(e)
                await db.commit()
