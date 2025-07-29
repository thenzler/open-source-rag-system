"""
Document management router
Handles all document-related API endpoints
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..di.services import (get_document_repository,
                           get_document_service, get_validation_service)
from ..models.api_models import DocumentResponse, DocumentUpdate
from ..repositories.interfaces import IDocumentRepository
from ..services import DocumentProcessingService, ValidationService
from ..utils.security import decode_document_id, encode_document_id

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
logger = logging.getLogger(__name__)


@router.post("", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    doc_service: DocumentProcessingService = Depends(get_document_service),
):
    """Upload a document for processing"""
    try:
        # Read file content
        content = await file.read()

        # Use document service for processing
        response = await doc_service.process_upload(
            filename=file.filename,
            content=content,
            content_type=file.content_type or "application/octet-stream",
        )

        # Replace real ID with obfuscated ID
        if hasattr(response, "id") and response.id:
            response.obfuscated_id = encode_document_id(response.id)

        return response

    except ValueError as e:
        logger.warning(f"Document upload validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_documents(
    doc_repo: IDocumentRepository = Depends(get_document_repository),
):
    """List all uploaded documents"""
    try:
        # Use the injected repository
        result = await doc_repo.list_all()
        return {
            "documents": [doc.to_dict() for doc in result.items],
            "total": result.total_count,
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}")
async def get_document_details(
    document_id: str,  # Changed to string for obfuscated IDs
    doc_service: DocumentProcessingService = Depends(get_document_service),
    validation_service: ValidationService = Depends(get_validation_service),
):
    """Get detailed information about a specific document"""
    try:
        # Decode obfuscated document ID
        real_doc_id = decode_document_id(document_id)
        if real_doc_id is None:
            raise HTTPException(status_code=400, detail="Invalid document ID format")

        # Validate document ID
        is_valid, message = validation_service.validate_document_id(real_doc_id)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)

        # Get document details using service
        details = await doc_service.get_document_details(real_doc_id)

        # Replace real ID with obfuscated ID in response
        if isinstance(details, dict) and "document" in details:
            details["document"]["id"] = document_id  # Return the obfuscated ID

        return details

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting document details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{document_id}")
async def update_document(
    document_id: int,
    updates: DocumentUpdate,
    doc_service: DocumentProcessingService = Depends(get_document_service),
    validation_service: ValidationService = Depends(get_validation_service),
):
    """Update document metadata"""
    try:
        # Validate document ID
        is_valid, message = validation_service.validate_document_id(document_id)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)

        # Update document using service
        result = await doc_service.update_document(document_id, updates)
        return result

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    doc_service: DocumentProcessingService = Depends(get_document_service),
    validation_service: ValidationService = Depends(get_validation_service),
):
    """Delete a document"""
    try:
        # Validate document ID
        is_valid, message = validation_service.validate_document_id(document_id)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)

        # Delete document using service
        result = await doc_service.delete_document(document_id)
        return result

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}/download")
async def download_document(
    document_id: int,
    doc_service: DocumentProcessingService = Depends(get_document_service),
    validation_service: ValidationService = Depends(get_validation_service),
):
    """Download a specific document file"""
    try:
        # Validate document ID
        is_valid, message = validation_service.validate_document_id(document_id)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)

        # Get file path from service
        file_path = await doc_service.get_download_path(document_id)

        # Return file response
        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type="application/octet-stream",
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error downloading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: int,
    page: int = 1,
    page_size: int = 20,
    search_query: Optional[str] = None,
    # query_service will be injected manually,
    validation_service: ValidationService = Depends(get_validation_service),
):
    """Get chunks for a specific document with optional search"""
    try:
        # Validate inputs
        is_valid, message = validation_service.validate_document_id(document_id)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)

        is_valid, message = validation_service.validate_pagination(page, page_size)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)

        if search_query:
            is_valid, message = validation_service.validate_query(search_query)
            if not is_valid:
                raise HTTPException(status_code=400, detail=message)

        # Get chunks using query service
        from ..di.services import get_query_service

        query_service = get_query_service()
        result = await query_service.get_document_chunks(
            document_id=document_id,
            page=page,
            page_size=page_size,
            search_query=search_query,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e))
