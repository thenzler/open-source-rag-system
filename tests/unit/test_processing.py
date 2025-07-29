#!/usr/bin/env python3
"""
Test document processing directly
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_processing():
    try:
        from core.services.document_service import DocumentProcessingService
        from core.di.services import ServiceConfiguration, get_container
        from core.repositories.interfaces import IDocumentRepository, IVectorSearchRepository
        from core.repositories.audit_repository import SwissAuditRepository
        
        # Configure DI
        ServiceConfiguration.configure_all()
        container = get_container()
        
        # Get services
        doc_repo = container.get(IDocumentRepository)
        vector_repo = container.get(IVectorSearchRepository)
        audit_repo = container.get(SwissAuditRepository)
        
        # Create document service
        doc_service = DocumentProcessingService(doc_repo, vector_repo, audit_repo)
        
        # Get document 40
        doc = await doc_repo.get_by_id(40)
        if not doc:
            print("Document 40 not found")
            return
        
        print(f"Processing document: {doc.filename}")
        print(f"File path: {doc.file_path}")
        
        # Check if file exists
        file_path = Path(doc.file_path)
        if not file_path.exists():
            print(f"File not found at: {file_path}")
            # Try relative path
            file_path = Path('storage/uploads') / doc.filename
            if file_path.exists():
                print(f"Found file at: {file_path}")
            else:
                # Try with timestamp prefix
                import os
                upload_dir = Path('storage/uploads')
                for f in upload_dir.glob(f"*_{doc.filename}"):
                    file_path = f
                    print(f"Found file at: {file_path}")
                    break
        
        # Test text extraction
        print("\nTesting text extraction...")
        text = await doc_service._extract_text(file_path)
        print(f"Extracted text length: {len(text)}")
        print(f"Text preview: {text[:200]}...")
        
        # Test chunking
        print("\nTesting chunking...")
        chunks = doc_service._create_chunks(text)
        print(f"Created {len(chunks)} chunks")
        
        # Test embedding generation
        print("\nTesting embedding generation...")
        embeddings = await doc_service._generate_embeddings(chunks[:2])  # Just test first 2
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
        
        # Actually process the document
        print("\nRunning full processing...")
        await doc_service._process_document_async(40, file_path)
        print("Processing complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_processing())