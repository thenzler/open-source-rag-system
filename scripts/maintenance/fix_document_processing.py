#!/usr/bin/env python3
"""
Document Processing Fix Script
Process existing documents from data/storage/ into the query database
"""
import os
import sys
import sqlite3
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """Process existing documents into the query database"""
    try:
        logger.info("Starting document processing fix...")
        
        # Import after path setup
        from core.repositories.factory import RepositoryFactory
        from core.services.document_service import DocumentProcessingService
        from core.repositories.models import Document, DocumentStatus
        
        # Initialize repositories
        rag_repo = RepositoryFactory.create_production_repository()
        await rag_repo.initialize()
        
        doc_repo = rag_repo.documents
        vector_repo = rag_repo.vector_search
        audit_repo = rag_repo.audit
        doc_service = DocumentProcessingService(doc_repo, vector_repo, audit_repo)
        
        # Find existing documents in data/storage/
        data_storage_dir = Path("data/storage")
        if not data_storage_dir.exists():
            logger.error(f"Directory {data_storage_dir} does not exist")
            return
        
        # Find all uploaded files
        upload_dirs = [
            data_storage_dir / "uploads",
            data_storage_dir / "processed"
        ]
        
        processed_count = 0
        
        for upload_dir in upload_dirs:
            if not upload_dir.exists():
                logger.info(f"Directory {upload_dir} does not exist, skipping")
                continue
                
            logger.info(f"Processing files in {upload_dir}")
            
            for file_path in upload_dir.glob("*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.txt', '.md', '.csv']:
                    try:
                        logger.info(f"Processing file: {file_path.name}")
                        
                        # Check if already processed
                        with open(file_path, 'rb') as f:
                            content = f.read()
                        
                        # Generate hash to check for duplicates
                        import hashlib
                        file_hash = hashlib.sha256(content).hexdigest()
                        
                        # Check if document already exists in database
                        existing = await doc_repo.find_by_hash(file_hash)
                        if existing:
                            logger.info(f"Document {file_path.name} already exists in database (ID: {existing.id})")
                            continue
                        
                        # Determine content type
                        import mimetypes
                        content_type = mimetypes.guess_type(str(file_path))[0] or 'application/octet-stream'
                        
                        # Create document record
                        document = Document(
                            filename=file_path.name,
                            original_filename=file_path.name,
                            file_path=str(file_path),
                            file_size=len(content),
                            content_type=content_type,
                            uploader='system_migration',
                            upload_timestamp=datetime.now(),
                            status=DocumentStatus.UPLOADING,
                            metadata={'file_hash': file_hash, 'migrated': True}
                        )
                        
                        # Store in repository
                        document = await doc_repo.create(document)
                        logger.info(f"Created document record ID: {document.id}")
                        
                        # Process document immediately (not async for this script)
                        await process_document_sync(doc_service, document.id, file_path)
                        
                        processed_count += 1
                        logger.info(f"Successfully processed: {file_path.name}")
                        
                    except Exception as e:
                        logger.error(f"Failed to process {file_path.name}: {e}")
                        continue
        
        logger.info(f"Document processing fix completed. Processed {processed_count} documents.")
        
        # Verify by running a test query
        await test_query(vector_repo)
        
    except Exception as e:
        logger.error(f"Document processing fix failed: {e}")
        raise

async def process_document_sync(doc_service, document_id: int, file_path: Path):
    """Process document synchronously for migration script"""
    try:
        # Update status to processing
        await doc_service.doc_repo.update_status(document_id, 'processing')
        
        # Extract text from document
        text_content = await doc_service._extract_text(file_path)
        if not text_content:
            raise ValueError("No text content extracted from document")
        
        logger.info(f"Extracted {len(text_content)} characters from document {document_id}")
        
        # Update document with text content
        await doc_service.doc_repo.update(document_id, {'text_content': text_content})
        
        # Create chunks
        chunks = doc_service._create_chunks(text_content, chunk_size=500, overlap=50)
        logger.info(f"Created {len(chunks)} chunks for document {document_id}")
        
        if not chunks:
            raise ValueError("No chunks created from document")
        
        # Store chunks in database
        chunk_ids = await doc_service._store_chunks(document_id, chunks)
        logger.info(f"Stored {len(chunk_ids)} chunks for document {document_id}")
        
        # Generate embeddings for chunks
        embeddings = await doc_service._generate_embeddings(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings for document {document_id}")
        
        # Store embeddings in database
        embedding_records = await doc_service._store_embeddings(document_id, chunk_ids, embeddings)
        logger.info(f"Stored {len(embedding_records)} embedding records for document {document_id}")
        
        # Update vector index
        await doc_service.vector_repo.add_to_index(embedding_records)
        logger.info(f"Added {len(embedding_records)} embeddings to vector index for document {document_id}")
        
        # Update document status and counts
        await doc_service.doc_repo.update(document_id, {
            'status': 'completed',
            'chunk_count': len(chunks),
            'embedding_count': len(embeddings),
            'completion_timestamp': datetime.now()
        })
        
        logger.info(f"Document {document_id} processed successfully with {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Document processing failed for {document_id}: {e}")
        await doc_service.doc_repo.update_status(document_id, 'failed')
        raise

async def test_query(vector_repo):
    """Test that documents are now queryable"""
    try:
        logger.info("Testing query functionality...")
        
        # Test query
        test_query = "Welche Regeln gelten für Gewerbeabfall?"
        result = await vector_repo.search_similar_text(test_query, limit=5, threshold=0.1)
        
        logger.info(f"Test query returned {len(result.items)} results")
        for i, item in enumerate(result.items[:3]):  # Show first 3 results
            similarity = item.metadata.get('similarity_score', 0.0)
            content_preview = (item.text_content or '')[:100] + "..." if len(item.text_content or '') > 100 else item.text_content
            logger.info(f"Result {i+1}: Document {item.document_id}, Similarity: {similarity:.3f}, Content: {content_preview}")
        
        if len(result.items) > 0:
            logger.info("✅ Query test successful - documents are now searchable!")
        else:
            logger.warning("❌ Query test failed - no results found")
            
    except Exception as e:
        logger.error(f"Query test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())