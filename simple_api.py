#!/usr/bin/env python3
"""
Simple FastAPI server for document uploads and basic RAG functionality
"""
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Dict
import logging
import json
import hashlib
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Document processing imports
try:
    import PyPDF2
    from docx import Document as DocxDocument
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    PDF_SUPPORT = True
except ImportError as e:
    print(f"Some dependencies not installed: {e}")
    PDF_SUPPORT = False

# Ollama integration
try:
    from ollama_client import get_ollama_client
    OLLAMA_SUPPORT = True
except ImportError as e:
    print(f"Ollama client not available: {e}")
    OLLAMA_SUPPORT = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Simple RAG API",
    description="Basic API for document upload and processing with embeddings",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8001", "http://127.0.0.1:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = Path("./storage/uploads")
PROCESSED_DIR = Path("./storage/processed")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# In-memory storage
documents = []
document_chunks = []
document_embeddings = []
document_id_counter = 1

# Initialize embedding model
embedding_model = None
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Embedding model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    embedding_model = None

# Initialize Ollama client
ollama_client = None
if OLLAMA_SUPPORT:
    try:
        ollama_client = get_ollama_client()
        if ollama_client.is_available():
            logger.info(f"Ollama client initialized successfully with model: {ollama_client.model}")
        else:
            logger.warning("Ollama client initialized but not available")
    except Exception as e:
        logger.error(f"Failed to initialize Ollama client: {e}")
        ollama_client = None

# Configuration
USE_LLM_DEFAULT = True  # Try to use LLM by default
MAX_CONTEXT_LENGTH = 4000  # Maximum context length for LLM

class DocumentResponse(BaseModel):
    id: int
    filename: str
    size: int
    content_type: str
    status: str

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    use_llm: Optional[bool] = None  # None = use default, True/False = override

class QueryResponse(BaseModel):
    query: str
    results: List[dict]
    total_results: int

class LLMQueryResponse(BaseModel):
    query: str
    answer: str
    method: str  # "llm_generated" or "vector_search"
    sources: List[dict]
    total_sources: int
    processing_time: Optional[float] = None

# Document processing functions
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    try:
        if not PDF_SUPPORT:
            return "PDF processing not available - install PyPDF2"
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return f"Error processing PDF: {str(e)}"

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file."""
    try:
        doc = DocxDocument(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        return f"Error processing DOCX: {str(e)}"

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error extracting text from TXT: {e}")
        return f"Error processing TXT: {str(e)}"

def extract_text_from_csv(file_path: str) -> str:
    """Extract text from CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df.to_string(index=False)
    except Exception as e:
        logger.error(f"Error extracting text from CSV: {e}")
        return f"Error processing CSV: {str(e)}"

def extract_text_from_file(file_path: str, content_type: str) -> str:
    """Extract text from various file types."""
    if content_type == "application/pdf":
        return extract_text_from_pdf(file_path)
    elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(file_path)
    elif content_type == "text/plain":
        return extract_text_from_txt(file_path)
    elif content_type == "text/csv":
        return extract_text_from_csv(file_path)
    else:
        return f"Unsupported file type: {content_type}"

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into chunks with overlap."""
    if not text:
        return []
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Create embeddings for text chunks."""
    if not embedding_model:
        return []
    
    try:
        embeddings = embedding_model.encode(texts)
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        return []

def find_similar_chunks(query: str, top_k: int = 5) -> List[dict]:
    """Find similar chunks using cosine similarity."""
    if not embedding_model or not document_embeddings:
        return []
    
    try:
        query_embedding = embedding_model.encode([query])
        
        similarities = []
        for i, doc_embedding in enumerate(document_embeddings):
            if doc_embedding:
                similarity = cosine_similarity(query_embedding, [doc_embedding])[0][0]
                similarities.append({
                    'chunk_id': i,
                    'similarity': float(similarity)
                })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    except Exception as e:
        logger.error(f"Error finding similar chunks: {e}")
        return []

def prepare_context_for_llm(similar_chunks: List[dict], max_length: int = MAX_CONTEXT_LENGTH) -> str:
    """
    Prepare context string for LLM from similar chunks
    
    Args:
        similar_chunks: List of similar chunks with metadata
        max_length: Maximum character length for context
    
    Returns:
        str: Formatted context string
    """
    if not similar_chunks:
        return ""
    
    context_parts = []
    current_length = 0
    
    for chunk_data in similar_chunks:
        if chunk_data['chunk_id'] < len(document_chunks):
            chunk_info = document_chunks[chunk_data['chunk_id']]
            
            # Format: [Document: filename] content
            chunk_text = f"[Document: {chunk_info['filename']}]\n{chunk_info['text']}\n"
            
            # Check if adding this chunk would exceed max length
            if current_length + len(chunk_text) > max_length:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
    
    return "\n".join(context_parts)

def generate_llm_answer(query: str, context: str) -> Optional[str]:
    """
    Generate answer using Ollama LLM
    
    Args:
        query: User's question
        context: Prepared context from documents
    
    Returns:
        Optional[str]: Generated answer or None if failed
    """
    if not ollama_client or not ollama_client.is_available():
        logger.warning("Ollama not available for answer generation")
        return None
    
    try:
        answer = ollama_client.generate_answer(query, context)
        return answer
    except Exception as e:
        logger.error(f"Error generating LLM answer: {e}")
        return None

def format_sources_for_response(similar_chunks: List[dict]) -> List[dict]:
    """
    Format source information for API response
    
    Args:
        similar_chunks: List of similar chunks with metadata
    
    Returns:
        List[dict]: Formatted source information
    """
    sources = []
    for chunk_data in similar_chunks:
        if chunk_data['chunk_id'] < len(document_chunks):
            chunk_info = document_chunks[chunk_data['chunk_id']]
            sources.append({
                "document_id": chunk_info['document_id'],
                "source_document": chunk_info['filename'],
                "content": chunk_info['text'],
                "similarity_score": chunk_data['similarity'],
                "chunk_id": chunk_data['chunk_id']
            })
    
    return sources

@app.get("/")
async def root():
    return {"message": "Simple RAG API is running", "frontend": "/simple_frontend.html", "docs": "/docs"}

@app.get("/simple_frontend.html", response_class=HTMLResponse)
async def get_frontend():
    """Serve the simple frontend HTML"""
    try:
        with open("simple_frontend.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1><p>simple_frontend.html file not found</p>", status_code=404)

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "simple-rag-api"}

@app.get("/api/v1/status")
async def get_system_status():
    """Get comprehensive system status including LLM availability"""
    status = {
        "service": "simple-rag-api",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "vector_search": embedding_model is not None,
            "llm_generation": False,
            "document_processing": PDF_SUPPORT
        },
        "statistics": {
            "documents_uploaded": len(documents),
            "total_chunks": len(document_chunks),
            "embeddings_created": len(document_embeddings)
        }
    }
    
    # Check Ollama status
    if OLLAMA_SUPPORT and ollama_client:
        try:
            ollama_health = ollama_client.health_check()
            status["features"]["llm_generation"] = ollama_health["available"]
            status["ollama"] = ollama_health
        except Exception as e:
            status["ollama"] = {
                "available": False,
                "error": str(e)
            }
    else:
        status["ollama"] = {
            "available": False,
            "error": "Ollama support not enabled"
        }
    
    return status

@app.get("/api/v1/analytics/stats")
async def get_stats():
    """Get system statistics"""
    total_size = sum(doc["file_size"] for doc in documents)
    return {
        "total_documents": len(documents),
        "total_queries": 0,
        "avg_response_time": 0.5,
        "storage_used": total_size
    }

@app.post("/api/v1/documents", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document for processing"""
    global document_id_counter
    
    try:
        # Validate file type
        allowed_types = {
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "text/csv",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        }
        
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file.content_type} not supported"
            )
        
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{document_id_counter}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract text from document
        logger.info(f"Processing document: {file.filename}")
        extracted_text = extract_text_from_file(str(file_path), file.content_type)
        
        # Create text chunks
        chunks = chunk_text(extracted_text)
        logger.info(f"Created {len(chunks)} chunks from {file.filename}")
        
        # Create embeddings for chunks
        embeddings = create_embeddings(chunks)
        logger.info(f"Created {len(embeddings)} embeddings for {file.filename}")
        
        # Store chunks and embeddings
        doc_chunk_start = len(document_chunks)
        for i, chunk in enumerate(chunks):
            document_chunks.append({
                'document_id': document_id_counter,
                'chunk_id': doc_chunk_start + i,
                'text': chunk,
                'filename': file.filename
            })
        
        # Store embeddings
        document_embeddings.extend(embeddings)
        
        # Create document record
        document = {
            "id": document_id_counter,
            "filename": file.filename,
            "original_filename": file.filename,
            "file_type": file.content_type,
            "file_size": len(content),
            "size": len(content),
            "content_type": file.content_type,
            "status": "processed",
            "upload_date": datetime.now().isoformat(),
            "chunks_count": len(chunks),
            "file_path": str(file_path),
            "text_preview": extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text
        }
        
        documents.append(document)
        logger.info(f"Document processed: {file.filename} (ID: {document_id_counter}, Chunks: {len(chunks)})")
        
        response = DocumentResponse(
            id=document_id_counter,
            filename=file.filename,
            size=len(content),
            content_type=file.content_type,
            status="processed"
        )
        
        document_id_counter += 1
        return response
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.get("/api/v1/documents")
async def list_documents():
    """List all uploaded documents"""
    return {
        "documents": [
            {
                "id": doc["id"],
                "filename": doc["filename"],
                "original_filename": doc["original_filename"],
                "file_type": doc["file_type"],
                "file_size": doc["file_size"],
                "size": doc["size"],
                "content_type": doc["content_type"],
                "status": doc["status"],
                "upload_date": doc["upload_date"],
                "chunks_count": doc["chunks_count"]
            }
            for doc in documents
        ],
        "total": len(documents)
    }

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using vector similarity search (legacy endpoint)"""
    try:
        if not embedding_model:
            raise HTTPException(
                status_code=500,
                detail="Embedding model not available. Please install sentence-transformers."
            )
        
        if not document_chunks:
            return QueryResponse(
                query=request.query,
                results=[],
                total_results=0
            )
        
        # Find similar chunks using vector search
        similar_chunks = find_similar_chunks(request.query, request.top_k)
        
        # Format results
        results = []
        for chunk_data in similar_chunks:
            chunk_info = document_chunks[chunk_data['chunk_id']]
            results.append({
                "document_id": chunk_info['document_id'],
                "source_document": chunk_info['filename'],
                "content": chunk_info['text'],
                "score": chunk_data['similarity'],
                "metadata": {
                    "chunk_id": chunk_data['chunk_id'],
                    "similarity_score": chunk_data['similarity']
                }
            })
        
        logger.info(f"Query: '{request.query}' returned {len(results)} results")
        
        return QueryResponse(
            query=request.query,
            results=results,
            total_results=len(results)
        )
        
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")

@app.post("/api/v1/query/enhanced", response_model=LLMQueryResponse)
async def query_documents_enhanced(request: QueryRequest):
    """Enhanced query with LLM answer generation and fallback to vector search"""
    start_time = time.time()
    
    try:
        if not embedding_model:
            raise HTTPException(
                status_code=500,
                detail="Embedding model not available. Please install sentence-transformers."
            )
        
        if not document_chunks:
            return LLMQueryResponse(
                query=request.query,
                answer="No documents have been uploaded yet. Please upload some documents first.",
                method="no_documents",
                sources=[],
                total_sources=0,
                processing_time=time.time() - start_time
            )
        
        # Find similar chunks using vector search
        similar_chunks = find_similar_chunks(request.query, request.top_k)
        
        if not similar_chunks:
            return LLMQueryResponse(
                query=request.query,
                answer="I couldn't find any relevant information in the uploaded documents for your question.",
                method="no_results",
                sources=[],
                total_sources=0,
                processing_time=time.time() - start_time
            )
        
        # Determine whether to use LLM
        use_llm = request.use_llm
        if use_llm is None:
            use_llm = USE_LLM_DEFAULT
        
        sources = format_sources_for_response(similar_chunks)
        
        # Try LLM generation if requested and available
        if use_llm and ollama_client and ollama_client.is_available():
            try:
                # Prepare context for LLM
                context = prepare_context_for_llm(similar_chunks)
                
                # Generate answer
                llm_answer = generate_llm_answer(request.query, context)
                
                if llm_answer:
                    logger.info(f"LLM query successful: '{request.query}'")
                    return LLMQueryResponse(
                        query=request.query,
                        answer=llm_answer,
                        method="llm_generated",
                        sources=sources,
                        total_sources=len(sources),
                        processing_time=time.time() - start_time
                    )
                else:
                    logger.warning("LLM generation failed, falling back to vector search")
            
            except Exception as e:
                logger.error(f"LLM generation error: {e}, falling back to vector search")
        
        # Fallback to vector search results
        if sources:
            # Create a summary from the top chunks
            top_chunks = [source["content"] for source in sources[:3]]
            fallback_answer = f"Based on the uploaded documents, here are the most relevant excerpts:\n\n"
            
            for i, chunk in enumerate(top_chunks, 1):
                source_doc = sources[i-1]["source_document"]
                fallback_answer += f"{i}. From '{source_doc}':\n{chunk}\n\n"
            
            method = "vector_search_fallback" if use_llm else "vector_search"
            
            return LLMQueryResponse(
                query=request.query,
                answer=fallback_answer,
                method=method,
                sources=sources,
                total_sources=len(sources),
                processing_time=time.time() - start_time
            )
        
        # No results found
        return LLMQueryResponse(
            query=request.query,
            answer="I couldn't find relevant information to answer your question.",
            method="no_results",
            sources=[],
            total_sources=0,
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Error in enhanced query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.delete("/api/v1/documents/{document_id}")
async def delete_document(document_id: int):
    """Delete a document"""
    global documents
    
    document = next((doc for doc in documents if doc["id"] == document_id), None)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Remove file if it exists
        if os.path.exists(document["file_path"]):
            os.remove(document["file_path"])
        
        # Remove from documents list
        documents = [doc for doc in documents if doc["id"] != document_id]
        
        logger.info(f"Document deleted: {document['filename']} (ID: {document_id})")
        return {"message": "Document deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "simple_api:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        log_level="info"
    )