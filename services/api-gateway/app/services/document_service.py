"""
Document Service - Handles document management and processing coordination
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import aiofiles
import aiofiles.os
from fastapi import UploadFile, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload

from app.models.documents import Document, DocumentChunk
from app.core.config import get_settings
from app.core.exceptions import DocumentNotFoundError, ProcessingError, ValidationError
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

class DocumentService:
    """Service for managing documents and coordinating processing."""
    
    def __init__(self):
        self.upload_directory = Path(settings.upload_directory)
        self.supported_formats = {
            'pdf': ['pdf'],
            'word': ['docx', 'doc'],
            'excel': ['xlsx', 'xls'],
            'text': ['txt', 'md', 'rtf'],
            'html': ['html', 'htm'],
            'xml': ['xml'],
            'csv': ['csv']
        }
        self.max_file_size = settings.max_file_size_mb * 1024 * 1024  # Convert MB to bytes
        self.document_processor_url = settings.document_processor_url
        self.vector_engine_url = settings.vector_engine_url
        
    async def initialize(self):
        """Initialize the document service."""
        # Create upload directory if it doesn't exist
        self.upload_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Document service initialized with upload directory: {self.upload_directory}")
    
    def _get_file_type(self, filename: str) -> str:
        """Determine file type from filename."""
        extension = filename.lower().split('.')[-1]
        for file_type, extensions in self.supported_formats.items():
            if extension in extensions:
                return file_type
        return 'unknown'
    
    def _validate_file(self, file: UploadFile) -> Dict[str, Any]:
        """Validate uploaded file."""
        if not file.filename:
            raise ValidationError("No filename provided")
        
        # Check file type
        file_type = self._get_file_type(file.filename)
        if file_type == 'unknown':
            raise ValidationError(f"Unsupported file type: {file.filename}")
        
        # Check file size
        if file.size and file.size > self.max_file_size:
            raise ValidationError(f"File too large: {file.size} bytes (max: {self.max_file_size})")
        
        return {
            'file_type': file_type,
            'size': file.size or 0,
            'mime_type': file.content_type
        }
    
    async def upload_document(
        self, 
        file: UploadFile, 
        metadata: Optional[str] = None,
        user_id: str = "default",
        db: AsyncSession = None
    ) -> Document:
        """Upload and store a document."""
        try:
            # Validate file
            file_info = self._validate_file(file)
            
            # Generate unique filename
            file_id = str(uuid.uuid4())
            file_extension = file.filename.split('.')[-1].lower()
            stored_filename = f"{file_id}.{file_extension}"
            file_path = self.upload_directory / stored_filename
            
            # Save file to disk
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Create document record
            document = Document(
                id=file_id,
                filename=stored_filename,
                original_filename=file.filename,
                file_path=str(file_path),
                file_size=len(content),
                file_type=file_info['file_type'],
                mime_type=file_info['mime_type'],
                status='pending',
                user_id=user_id,
                metadata={'original_metadata': metadata} if metadata else {}
            )
            
            # Save to database
            db.add(document)
            await db.commit()
            await db.refresh(document)
            
            logger.info(f"Document uploaded successfully: {document.id}")
            return document
            
        except Exception as e:
            logger.error(f"Error uploading document: {e}")
            # Clean up file if it was created
            if 'file_path' in locals() and file_path.exists():
                await aiofiles.os.remove(file_path)
            raise ProcessingError(f"Failed to upload document: {str(e)}")
    
    async def process_document_async(self, document_id: str, db: AsyncSession):
        """Process document asynchronously."""
        try:
            # Get document
            document = await self.get_document(document_id, db=db)
            if not document:
                raise DocumentNotFoundError(f"Document not found: {document_id}")
            
            # Update status to processing
            document.status = 'processing'
            await db.commit()
            
            # Call document processor service
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.document_processor_url}/process",
                    json={
                        "document_id": document_id,
                        "file_path": document.file_path,
                        "file_type": document.file_type,
                        "metadata": document.metadata
                    },
                    timeout=300.0  # 5 minutes timeout
                )
                
                if response.status_code != 200:
                    raise ProcessingError(f"Document processor failed: {response.text}")
                
                result = response.json()
                logger.info(f"Document processing queued: {result}")
            
            logger.info(f"Document processing started: {document_id}")
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            # Update status to failed
            if 'document' in locals():
                document.status = 'failed'
                await db.commit()
            raise
    
    async def get_document(
        self, 
        document_id: str, 
        user_id: str = None,
        db: AsyncSession = None
    ) -> Optional[Document]:
        """Get a document by ID."""
        try:
            query = select(Document).where(Document.id == document_id)
            
            if user_id:
                query = query.where(Document.user_id == user_id)
            
            result = await db.execute(query)
            document = result.scalar_one_or_none()
            
            if not document:
                raise DocumentNotFoundError(f"Document not found: {document_id}")
            
            return document
            
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {e}")
            raise
    
    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 50,
        status: Optional[str] = None,
        category: Optional[str] = None,
        search: Optional[str] = None,
        user_id: str = None,
        db: AsyncSession = None
    ) -> Tuple[List[Document], int]:
        """List documents with pagination and filtering."""
        try:
            # Base query
            query = select(Document)
            count_query = select(func.count(Document.id))
            
            # Apply filters
            conditions = []
            
            if user_id:
                conditions.append(Document.user_id == user_id)
            
            if status:
                conditions.append(Document.status == status)
            
            if category:
                conditions.append(Document.file_type == category)
            
            if search:
                search_term = f"%{search}%"
                conditions.append(
                    or_(
                        Document.original_filename.ilike(search_term),
                        Document.filename.ilike(search_term)
                    )
                )
            
            if conditions:
                query = query.where(and_(*conditions))
                count_query = count_query.where(and_(*conditions))
            
            # Apply pagination
            query = query.offset(skip).limit(limit).order_by(Document.upload_date.desc())
            
            # Execute queries
            result = await db.execute(query)
            documents = result.scalars().all()
            
            count_result = await db.execute(count_query)
            total_count = count_result.scalar()
            
            return documents, total_count
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            raise
    
    async def delete_document(
        self,
        document_id: str,
        user_id: str = None,
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """Delete a document and all associated data."""
        try:
            # Get document
            document = await self.get_document(document_id, user_id, db)
            if not document:
                raise DocumentNotFoundError(f"Document not found: {document_id}")
            
            # Delete document chunks
            chunks_query = select(DocumentChunk).where(DocumentChunk.document_id == document_id)
            chunks_result = await db.execute(chunks_query)
            chunks = chunks_result.scalars().all()
            
            deleted_chunks = len(chunks)
            
            # Delete chunks from database
            for chunk in chunks:
                await db.delete(chunk)
            
            # Delete vectors from vector database
            deleted_vectors = 0
            try:
                async with httpx.AsyncClient() as client:
                    # Delete from vector database
                    response = await client.delete(
                        f"{self.vector_engine_url}/vectors/document/{document_id}",
                        timeout=30.0
                    )
                    if response.status_code == 200:
                        result = response.json()
                        deleted_vectors = result.get('deleted_count', 0)
            except Exception as e:
                logger.warning(f"Error deleting vectors: {e}")
            
            # Delete file from filesystem
            try:
                if os.path.exists(document.file_path):
                    await aiofiles.os.remove(document.file_path)
            except Exception as e:
                logger.warning(f"Error deleting file: {e}")
            
            # Delete document from database
            await db.delete(document)
            await db.commit()
            
            logger.info(f"Document deleted successfully: {document_id}")
            
            return {
                'deleted_chunks': deleted_chunks,
                'deleted_vectors': deleted_vectors
            }
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            raise
    
    async def update_document_status(
        self,
        document_id: str,
        status: str,
        chunks_count: int = 0,
        processed_date: Optional[datetime] = None,
        db: AsyncSession = None
    ):
        """Update document processing status."""
        try:
            document = await self.get_document(document_id, db=db)
            if not document:
                raise DocumentNotFoundError(f"Document not found: {document_id}")
            
            document.status = status
            document.chunks_count = chunks_count
            
            if processed_date:
                document.processed_date = processed_date
            elif status == 'completed':
                document.processed_date = datetime.utcnow()
            
            await db.commit()
            logger.info(f"Document status updated: {document_id} -> {status}")
            
        except Exception as e:
            logger.error(f"Error updating document status: {e}")
            raise
    
    async def get_document_chunks(
        self,
        document_id: str,
        skip: int = 0,
        limit: int = 50,
        db: AsyncSession = None
    ) -> List[DocumentChunk]:
        """Get chunks for a document."""
        try:
            query = select(DocumentChunk).where(
                DocumentChunk.document_id == document_id
            ).offset(skip).limit(limit).order_by(DocumentChunk.chunk_index)
            
            result = await db.execute(query)
            chunks = result.scalars().all()
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting document chunks: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if the document service is healthy."""
        try:
            # Check if upload directory exists and is writable
            if not self.upload_directory.exists():
                return False
            
            # Check if document processor is available
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.document_processor_url}/health", timeout=5.0)
                    if response.status_code != 200:
                        return False
            except:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Document service health check failed: {e}")
            return False
    
    async def get_processing_stats(self, db: AsyncSession = None) -> Dict[str, Any]:
        """Get document processing statistics."""
        try:
            # Total documents
            total_query = select(func.count(Document.id))
            total_result = await db.execute(total_query)
            total_documents = total_result.scalar()
            
            # Documents by status
            status_query = select(Document.status, func.count(Document.id)).group_by(Document.status)
            status_result = await db.execute(status_query)
            status_counts = {status: count for status, count in status_result.fetchall()}
            
            # Documents by type
            type_query = select(Document.file_type, func.count(Document.id)).group_by(Document.file_type)
            type_result = await db.execute(type_query)
            type_counts = {file_type: count for file_type, count in type_result.fetchall()}
            
            # Total file size
            size_query = select(func.sum(Document.file_size))
            size_result = await db.execute(size_query)
            total_size = size_result.scalar() or 0
            
            # Total chunks
            chunks_query = select(func.sum(Document.chunks_count))
            chunks_result = await db.execute(chunks_query)
            total_chunks = chunks_result.scalar() or 0
            
            return {
                'total_documents': total_documents,
                'status_counts': status_counts,
                'type_counts': type_counts,
                'total_file_size': total_size,
                'total_chunks': total_chunks
            }
            
        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            return {}
