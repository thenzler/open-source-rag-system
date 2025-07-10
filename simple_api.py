#!/usr/bin/env python3
"""
Simple FastAPI server for document uploads and basic RAG functionality
"""
import os
import tempfile
import time
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import json
import hashlib
from datetime import datetime
from functools import lru_cache
import pickle

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
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
    allow_origins=["*"],  # Allow all origins for widget integration
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
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

# Performance Caching System
class FastCache:
    """Fast in-memory cache for embeddings and search results"""
    def __init__(self, max_size=1000):
        self.query_cache = {}
        self.embedding_cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def get_query_cache(self, query_hash: str):
        """Get cached query result"""
        if query_hash in self.query_cache:
            self.access_times[query_hash] = time.time()
            return self.query_cache[query_hash]
        return None
    
    def set_query_cache(self, query_hash: str, result):
        """Set cached query result"""
        self._cleanup_if_needed()
        self.query_cache[query_hash] = result
        self.access_times[query_hash] = time.time()
    
    def get_embedding_cache(self, text_hash: str):
        """Get cached embedding"""
        if text_hash in self.embedding_cache:
            self.access_times[text_hash] = time.time()
            return self.embedding_cache[text_hash]
        return None
    
    def set_embedding_cache(self, text_hash: str, embedding):
        """Set cached embedding"""
        self._cleanup_if_needed()
        self.embedding_cache[text_hash] = embedding
        self.access_times[text_hash] = time.time()
    
    def _cleanup_if_needed(self):
        """Remove oldest entries if cache is full"""
        total_items = len(self.query_cache) + len(self.embedding_cache)
        if total_items >= self.max_size:
            # Remove 20% of oldest entries
            items_to_remove = int(total_items * 0.2)
            all_items = [(k, v) for k, v in self.access_times.items()]
            all_items.sort(key=lambda x: x[1])  # Sort by access time
            
            for key, _ in all_items[:items_to_remove]:
                if key in self.query_cache:
                    del self.query_cache[key]
                if key in self.embedding_cache:
                    del self.embedding_cache[key]
                del self.access_times[key]
    
    def clear(self):
        """Clear all caches"""
        self.query_cache.clear()
        self.embedding_cache.clear()
        self.access_times.clear()

# Initialize performance cache
fast_cache = FastCache(max_size=1000)

# File upload limits
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB maximum file size
MAX_TOTAL_DOCUMENTS = 1000  # Maximum number of documents
MAX_CONCURRENT_UPLOADS = 5  # Maximum concurrent uploads

# Allowed file types and extensions
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
}

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".xlsx"}

# Security settings
ALLOWED_FILENAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_")
MAX_FILENAME_LENGTH = 255

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

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[dict]] = []
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    context_limit: Optional[int] = 5

class ChatResponse(BaseModel):
    response: str
    query: Optional[str] = None
    context: Optional[List[dict]] = []
    confidence: Optional[float] = 0.0
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
    """Create embeddings for text chunks with caching."""
    if not embedding_model:
        return []
    
    try:
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cached_embedding = fast_cache.get_embedding_cache(text_hash)
            
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Process uncached texts in batch
        if uncached_texts:
            new_embeddings = embedding_model.encode(uncached_texts)
            
            # Cache new embeddings and fill placeholders
            for j, embedding in enumerate(new_embeddings):
                idx = uncached_indices[j]
                text_hash = hashlib.md5(texts[idx].encode()).hexdigest()
                embedding_list = embedding.tolist()
                fast_cache.set_embedding_cache(text_hash, embedding_list)
                embeddings[idx] = embedding_list
        
        return embeddings
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        return []

def find_similar_chunks(query: str, top_k: int = 5) -> List[dict]:
    """Find similar chunks using cosine similarity with caching."""
    if not embedding_model or not document_embeddings:
        return []
    
    try:
        # Check cache first
        query_hash = hashlib.md5(f"{query}_{top_k}".encode()).hexdigest()
        cached_result = fast_cache.get_query_cache(query_hash)
        if cached_result is not None:
            return cached_result
        
        # Get or create query embedding
        query_text_hash = hashlib.md5(query.encode()).hexdigest()
        query_embedding = fast_cache.get_embedding_cache(query_text_hash)
        
        if query_embedding is None:
            query_embedding = embedding_model.encode([query])[0].tolist()
            fast_cache.set_embedding_cache(query_text_hash, query_embedding)
        
        # Use numpy for faster batch similarity computation
        query_embedding_np = np.array([query_embedding])
        valid_embeddings = []
        valid_indices = []
        
        # Filter out None embeddings and track indices
        for i, emb in enumerate(document_embeddings):
            if emb:
                valid_embeddings.append(emb)
                valid_indices.append(i)
        
        if valid_embeddings:
            # Batch computation is much faster
            doc_embeddings_np = np.array(valid_embeddings)
            similarities_np = cosine_similarity(query_embedding_np, doc_embeddings_np)[0]
            
            # Create result list with proper indices
            similarities = []
            for idx, similarity in enumerate(similarities_np):
                similarities.append({
                    'chunk_id': valid_indices[idx],
                    'similarity': float(similarity)
                })
        else:
            similarities = []
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Early termination if we have very high similarity matches
        if similarities and similarities[0]['similarity'] > 0.95:
            # If we have a nearly perfect match, we can return fewer results
            result = similarities[:min(top_k, 3)]
        else:
            result = similarities[:top_k]
        
        # Cache the result
        fast_cache.set_query_cache(query_hash, result)
        return result
    
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

# File validation functions
def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal and other security issues
    
    Args:
        filename: Original filename
    
    Returns:
        str: Sanitized filename
    """
    if not filename:
        return "untitled"
    
    # Remove path components
    filename = os.path.basename(filename)
    
    # Remove or replace dangerous characters
    sanitized = "".join(c for c in filename if c in ALLOWED_FILENAME_CHARS)
    
    # Ensure filename is not too long
    if len(sanitized) > MAX_FILENAME_LENGTH:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:MAX_FILENAME_LENGTH - len(ext)] + ext
    
    # Ensure filename is not empty
    if not sanitized:
        sanitized = "untitled"
    
    return sanitized

def validate_file_upload(file: UploadFile) -> Dict[str, Any]:
    """
    Validate file upload for security and size constraints
    
    Args:
        file: FastAPI UploadFile object
    
    Returns:
        Dict with validation results
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check if file exists
    if not file or not file.filename:
        validation_result["valid"] = False
        validation_result["errors"].append("No file provided")
        return validation_result
    
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        validation_result["valid"] = False
        validation_result["errors"].append(f"File type {file_ext} not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}")
    
    # Check content type (more lenient - log warning but don't fail)
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning(f"Unknown content type: {file.content_type} for file: {file.filename}")
        validation_result["warnings"].append(f"Unknown content type: {file.content_type}, but proceeding based on file extension")
    
    # Check filename length
    if len(file.filename) > MAX_FILENAME_LENGTH:
        validation_result["warnings"].append(f"Filename too long, will be truncated to {MAX_FILENAME_LENGTH} characters")
    
    # Check total document limit
    if len(documents) >= MAX_TOTAL_DOCUMENTS:
        validation_result["valid"] = False
        validation_result["errors"].append(f"Maximum number of documents ({MAX_TOTAL_DOCUMENTS}) reached")
    
    return validation_result

def validate_file_content(file_path: str, file_size: int) -> Dict[str, Any]:
    """
    Validate file content after upload
    
    Args:
        file_path: Path to uploaded file
        file_size: File size in bytes
    
    Returns:
        Dict with validation results
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check file size
    if file_size > MAX_FILE_SIZE:
        validation_result["valid"] = False
        validation_result["errors"].append(f"File size ({file_size:,} bytes) exceeds maximum allowed size ({MAX_FILE_SIZE:,} bytes)")
    
    # Check if file is empty
    if file_size == 0:
        validation_result["valid"] = False
        validation_result["errors"].append("File is empty")
    
    # Check if file actually exists (only if file_path is provided)
    if file_path and not os.path.exists(file_path):
        validation_result["valid"] = False
        validation_result["errors"].append("File was not saved properly")
    
    # Additional content validation can be added here
    # For example, checking if PDF is corrupted, etc.
    
    return validation_result

# Request validation and sanitization functions
def sanitize_query_string(query: str) -> str:
    """
    Sanitize query string to prevent injection attacks
    
    Args:
        query: Raw query string
    
    Returns:
        str: Sanitized query string
    """
    if not query:
        return ""
    
    # Remove null bytes and control characters
    query = query.replace('\x00', '').replace('\r', '').replace('\n', ' ')
    
    # Remove potentially dangerous HTML/script tags
    query = re.sub(r'<[^>]*>', '', query)
    
    # Remove excessive whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    
    # Limit length to prevent DoS
    if len(query) > 2000:
        query = query[:2000]
    
    return query

def validate_query_request(request: QueryRequest) -> Dict[str, Any]:
    """
    Validate and sanitize query request
    
    Args:
        request: Query request object
    
    Returns:
        Dict with validation results and sanitized request
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "sanitized_request": None
    }
    
    # Validate query string
    if not request.query or not request.query.strip():
        validation_result["valid"] = False
        validation_result["errors"].append("Query string cannot be empty")
        return validation_result
    
    # Sanitize query
    sanitized_query = sanitize_query_string(request.query)
    
    if not sanitized_query:
        validation_result["valid"] = False
        validation_result["errors"].append("Query string is invalid after sanitization")
        return validation_result
    
    # Warn if query was significantly modified
    if len(sanitized_query) < len(request.query.strip()) * 0.8:
        validation_result["warnings"].append("Query was significantly modified during sanitization")
    
    # Validate top_k parameter
    if request.top_k is not None:
        if not isinstance(request.top_k, int) or request.top_k < 1:
            validation_result["valid"] = False
            validation_result["errors"].append("top_k must be a positive integer")
            return validation_result
        
        if request.top_k > 20:
            validation_result["warnings"].append("top_k is very high, limiting to 20")
            request.top_k = 20
    
    # Validate use_llm parameter
    if request.use_llm is not None and not isinstance(request.use_llm, bool):
        validation_result["valid"] = False
        validation_result["errors"].append("use_llm must be a boolean")
        return validation_result
    
    # Create sanitized request
    sanitized_request = QueryRequest(
        query=sanitized_query,
        top_k=request.top_k,
        use_llm=request.use_llm
    )
    
    validation_result["sanitized_request"] = sanitized_request
    return validation_result

def validate_document_id(document_id: str) -> bool:
    """
    Validate document ID to prevent injection attacks
    
    Args:
        document_id: Document ID to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not document_id:
        return False
    
    # Allow only digits
    if not re.match(r'^\d+$', document_id):
        return False
    
    # Check reasonable range
    try:
        doc_id_int = int(document_id)
        if doc_id_int < 1 or doc_id_int > 1000000:
            return False
    except ValueError:
        return False
    
    return True

def rate_limit_check(request_type: str, client_id: str = "default") -> bool:
    """
    Simple rate limiting check
    
    Args:
        request_type: Type of request (upload, query, etc.)
        client_id: Client identifier
    
    Returns:
        bool: True if request is allowed, False if rate limited
    """
    # Simple in-memory rate limiting
    if not hasattr(rate_limit_check, 'requests'):
        rate_limit_check.requests = {}
    
    current_time = time.time()
    key = f"{client_id}:{request_type}"
    
    # Clean old entries
    if key in rate_limit_check.requests:
        rate_limit_check.requests[key] = [
            timestamp for timestamp in rate_limit_check.requests[key]
            if current_time - timestamp < 60  # 1 minute window
        ]
    else:
        rate_limit_check.requests[key] = []
    
    # Check limits
    limits = {
        "upload": 50,  # 50 uploads per minute
        "query": 100,  # 100 queries per minute
        "status": 50   # 50 status checks per minute
    }
    
    limit = limits.get(request_type, 30)
    
    if len(rate_limit_check.requests[key]) >= limit:
        return False
    
    # Add current request
    rate_limit_check.requests[key].append(current_time)
    return True

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

@app.post("/api/v1/reset-rate-limits")
async def reset_rate_limits():
    """Reset rate limits for all clients"""
    if hasattr(rate_limit_check, 'requests'):
        rate_limit_check.requests.clear()
        return {"message": "Rate limits reset successfully"}
    return {"message": "No rate limits to reset"}

@app.post("/api/v1/clear-cache")
async def clear_cache():
    """Clear all performance caches"""
    fast_cache.clear()
    return {
        "message": "Performance caches cleared successfully",
        "cache_stats": {
            "query_cache_size": len(fast_cache.query_cache),
            "embedding_cache_size": len(fast_cache.embedding_cache)
        }
    }

@app.get("/api/v1/status")
async def get_system_status():
    """Get comprehensive system status including LLM availability"""
    # Rate limiting check
    if not rate_limit_check("status"):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
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
    """Upload a document for processing with comprehensive validation"""
    global document_id_counter
    
    try:
        # Rate limiting check
        if not rate_limit_check("upload"):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        # Validate file upload
        logger.info(f"Validating file upload: {file.filename}, content_type: {file.content_type}")
        validation_result = validate_file_upload(file)
        
        if not validation_result["valid"]:
            error_message = "Upload validation failed: " + "; ".join(validation_result["errors"])
            logger.error(f"Upload validation failed for {file.filename}: {error_message}")
            raise HTTPException(status_code=400, detail=error_message)
        
        # Log warnings if any
        if validation_result["warnings"]:
            logger.warning(f"Upload warnings for {file.filename}: {'; '.join(validation_result['warnings'])}")
        
        # Sanitize filename
        sanitized_filename = sanitize_filename(file.filename)
        logger.info(f"Processing document: {file.filename} -> {sanitized_filename}")
        
        # Read file content
        content = await file.read()
        
        # Validate file content
        logger.info(f"Validating content for {file.filename}, size: {len(content)} bytes")
        content_validation = validate_file_content("", len(content))
        
        if not content_validation["valid"]:
            error_message = "Content validation failed: " + "; ".join(content_validation["errors"])
            logger.error(f"Content validation failed for {file.filename}: {error_message}")
            raise HTTPException(status_code=400, detail=error_message)
        
        # Save uploaded file with sanitized name
        file_path = UPLOAD_DIR / f"{document_id_counter}_{sanitized_filename}"
        
        # Ensure upload directory exists
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            
            # Final validation on saved file
            saved_file_validation = validate_file_content(str(file_path), len(content))
            
            if not saved_file_validation["valid"]:
                # Clean up failed file
                if os.path.exists(file_path):
                    os.remove(file_path)
                error_message = "File save validation failed: " + "; ".join(saved_file_validation["errors"])
                logger.error(f"File save validation failed for {file.filename}: {error_message}")
                raise HTTPException(status_code=500, detail=error_message)
            
        except Exception as e:
            # Clean up on file write error
            if os.path.exists(file_path):
                os.remove(file_path)
            logger.error(f"Error saving file {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
        
        # Extract text from document
        try:
            logger.info(f"Extracting text from: {sanitized_filename}")
            extracted_text = extract_text_from_file(str(file_path), file.content_type)
            
            # Check if text extraction was successful
            if not extracted_text or extracted_text.strip() == "":
                logger.warning(f"No text extracted from {sanitized_filename}")
                raise HTTPException(status_code=400, detail="No text content could be extracted from the file")
            
            # Check for extraction errors
            if extracted_text.startswith("Error processing"):
                logger.error(f"Text extraction error for {sanitized_filename}: {extracted_text}")
                raise HTTPException(status_code=400, detail=f"Text extraction failed: {extracted_text}")
            
        except HTTPException:
            # Clean up on text extraction error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise
        except Exception as e:
            # Clean up on unexpected error
            if os.path.exists(file_path):
                os.remove(file_path)
            logger.error(f"Unexpected error extracting text from {sanitized_filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")
        
        # Create text chunks
        try:
            chunks = chunk_text(extracted_text)
            logger.info(f"Created {len(chunks)} chunks from {sanitized_filename}")
            
            if not chunks:
                logger.warning(f"No chunks created from {sanitized_filename}")
                raise HTTPException(status_code=400, detail="Document could not be processed into searchable chunks")
            
        except Exception as e:
            # Clean up on chunking error
            if os.path.exists(file_path):
                os.remove(file_path)
            logger.error(f"Error chunking text from {sanitized_filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
        
        # Create embeddings for chunks
        try:
            embeddings = create_embeddings(chunks)
            logger.info(f"Created {len(embeddings)} embeddings for {sanitized_filename}")
            
            if not embeddings:
                logger.warning(f"No embeddings created for {sanitized_filename}")
                raise HTTPException(status_code=500, detail="Could not create embeddings for document")
            
        except Exception as e:
            # Clean up on embedding error
            if os.path.exists(file_path):
                os.remove(file_path)
            logger.error(f"Error creating embeddings for {sanitized_filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error creating embeddings: {str(e)}")
        
        # Store chunks and embeddings
        doc_chunk_start = len(document_chunks)
        for i, chunk in enumerate(chunks):
            document_chunks.append({
                'document_id': document_id_counter,
                'chunk_id': doc_chunk_start + i,
                'text': chunk,
                'filename': sanitized_filename
            })
        
        # Store embeddings
        document_embeddings.extend(embeddings)
        
        # Create document record
        document = {
            "id": document_id_counter,
            "filename": sanitized_filename,
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
        logger.info(f"Document processed successfully: {sanitized_filename} (ID: {document_id_counter}, Chunks: {len(chunks)}, Size: {len(content):,} bytes)")
        
        response = DocumentResponse(
            id=document_id_counter,
            filename=sanitized_filename,
            size=len(content),
            content_type=file.content_type,
            status="processed"
        )
        
        document_id_counter += 1
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Handle any other unexpected errors
        logger.error(f"Unexpected error uploading document {file.filename if file else 'unknown'}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error uploading document: {str(e)}")

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
        # Rate limiting check
        if not rate_limit_check("query"):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Validate and sanitize request
        validation_result = validate_query_request(request)
        
        if not validation_result["valid"]:
            error_message = "Request validation failed: " + "; ".join(validation_result["errors"])
            logger.warning(f"Query validation failed: {error_message}")
            raise HTTPException(status_code=400, detail=error_message)
        
        # Log warnings if any
        if validation_result["warnings"]:
            logger.warning(f"Query validation warnings: {'; '.join(validation_result['warnings'])}")
        
        # Use sanitized request
        sanitized_request = validation_result["sanitized_request"]
        
        if not embedding_model:
            raise HTTPException(
                status_code=500,
                detail="Embedding model not available. Please install sentence-transformers."
            )
        
        if not document_chunks:
            return QueryResponse(
                query=sanitized_request.query,
                results=[],
                total_results=0
            )
        
        # Find similar chunks using vector search
        similar_chunks = find_similar_chunks(sanitized_request.query, sanitized_request.top_k)
        
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
        
        logger.info(f"Query: '{sanitized_request.query}' returned {len(results)} results")
        
        return QueryResponse(
            query=sanitized_request.query,
            results=results,
            total_results=len(results)
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")

@app.post("/api/v1/query/enhanced", response_model=LLMQueryResponse)
async def query_documents_enhanced(request: QueryRequest):
    """Enhanced query with LLM answer generation and fallback to vector search"""
    start_time = time.time()
    
    try:
        # Rate limiting check
        if not rate_limit_check("query"):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Validate and sanitize request
        validation_result = validate_query_request(request)
        
        if not validation_result["valid"]:
            error_message = "Request validation failed: " + "; ".join(validation_result["errors"])
            logger.warning(f"Enhanced query validation failed: {error_message}")
            raise HTTPException(status_code=400, detail=error_message)
        
        # Log warnings if any
        if validation_result["warnings"]:
            logger.warning(f"Enhanced query validation warnings: {'; '.join(validation_result['warnings'])}")
        
        # Use sanitized request
        sanitized_request = validation_result["sanitized_request"]
        
        if not embedding_model:
            raise HTTPException(
                status_code=500,
                detail="Embedding model not available. Please install sentence-transformers."
            )
        
        if not document_chunks:
            return LLMQueryResponse(
                query=sanitized_request.query,
                answer="No documents have been uploaded yet. Please upload some documents first.",
                method="no_documents",
                sources=[],
                total_sources=0,
                processing_time=time.time() - start_time
            )
        
        # Find similar chunks using vector search
        similar_chunks = find_similar_chunks(sanitized_request.query, sanitized_request.top_k)
        
        if not similar_chunks:
            return LLMQueryResponse(
                query=sanitized_request.query,
                answer="I couldn't find any relevant information in the uploaded documents for your question.",
                method="no_results",
                sources=[],
                total_sources=0,
                processing_time=time.time() - start_time
            )
        
        # Determine whether to use LLM
        use_llm = sanitized_request.use_llm
        if use_llm is None:
            use_llm = USE_LLM_DEFAULT
        
        sources = format_sources_for_response(similar_chunks)
        
        # Try LLM generation if requested and available
        if use_llm and ollama_client:
            try:
                # Check if Ollama is available (this may trigger a fresh check)
                if not ollama_client.is_available():
                    logger.warning("Ollama not available, falling back to vector search")
                    method = "vector_search_fallback"
                else:
                    # Prepare context for LLM
                    context = prepare_context_for_llm(similar_chunks)
                    
                    if not context:
                        logger.warning("No context available for LLM generation")
                        method = "vector_search_fallback"
                    else:
                        # Generate answer
                        llm_answer = generate_llm_answer(sanitized_request.query, context)
                        
                        if llm_answer:
                            logger.info(f"LLM query successful: '{sanitized_request.query}'")
                            return LLMQueryResponse(
                                query=sanitized_request.query,
                                answer=llm_answer,
                                method="llm_generated",
                                sources=sources,
                                total_sources=len(sources),
                                processing_time=time.time() - start_time
                            )
                        else:
                            logger.warning("LLM generation failed, falling back to vector search")
                            method = "vector_search_fallback"
            
            except Exception as e:
                logger.error(f"LLM generation error: {e}, falling back to vector search")
                method = "vector_search_fallback"
        
        # Fallback to vector search results
        if sources:
            # Create a summary from the top chunks
            top_chunks = [source["content"] for source in sources[:3]]
            fallback_answer = f"Based on the uploaded documents, here are the most relevant excerpts:\n\n"
            
            for i, chunk in enumerate(top_chunks, 1):
                source_doc = sources[i-1]["source_document"]
                fallback_answer += f"{i}. From '{source_doc}':\n{chunk}\n\n"
            
            # Determine the method based on whether LLM was attempted
            if use_llm and ollama_client:
                method = "vector_search_fallback"
            else:
                method = "vector_search"
            
            return LLMQueryResponse(
                query=sanitized_request.query,
                answer=fallback_answer,
                method=method,
                sources=sources,
                total_sources=len(sources),
                processing_time=time.time() - start_time
            )
        
        # No results found
        return LLMQueryResponse(
            query=sanitized_request.query,
            answer="I couldn't find relevant information to answer your question.",
            method="no_results",
            sources=[],
            total_sources=0,
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Error in enhanced query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/api/v1/query-stream")
async def query_documents_stream(request: QueryRequest):
    """Enhanced query with streaming LLM response"""
    def generate_stream():
        try:
            # Rate limiting check
            if not rate_limit_check("query"):
                yield f"data: {json.dumps({'error': 'Rate limit exceeded'})}\n\n"
                return
            
            # Validate request
            validation_result = validate_query_request(request)
            if not validation_result["valid"]:
                yield f"data: {json.dumps({'error': 'Invalid request'})}\n\n"
                return
            
            sanitized_request = validation_result["sanitized_request"]
            
            # Check if we have documents
            if not document_chunks:
                yield f"data: {json.dumps({'error': 'No documents uploaded'})}\n\n"
                return
            
            # Find similar chunks
            similar_chunks = find_similar_chunks(sanitized_request.query, sanitized_request.top_k)
            
            if not similar_chunks:
                yield f"data: {json.dumps({'error': 'No relevant information found'})}\n\n"
                return
            
            # Prepare context
            context = prepare_context_for_llm(similar_chunks)
            
            # Send initial metadata
            sources = format_sources_for_response(similar_chunks)
            metadata = {
                "type": "metadata",
                "sources": sources,
                "total_sources": len(sources),
                "query": sanitized_request.query
            }
            yield f"data: {json.dumps(metadata)}\n\n"
            
            # Check if LLM is available
            if ollama_client and ollama_client.is_available():
                # Stream LLM response
                yield f"data: {json.dumps({'type': 'answer_start'})}\n\n"
                
                answer_chunks = []
                for chunk in ollama_client.generate_answer_stream(sanitized_request.query, context):
                    if chunk:
                        answer_chunks.append(chunk)
                        yield f"data: {json.dumps({'type': 'answer_chunk', 'content': chunk})}\n\n"
                
                yield f"data: {json.dumps({'type': 'answer_end', 'method': 'llm'})}\n\n"
            else:
                # Fallback to vector search
                fallback_answer = f"Based on the uploaded documents, here are the most relevant excerpts:\n\n"
                for i, source in enumerate(sources[:3], 1):
                    fallback_answer += f"{i}. From '{source['source_document']}':\n{source['content']}\n\n"
                
                yield f"data: {json.dumps({'type': 'answer_complete', 'content': fallback_answer, 'method': 'vector_search'})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )

@app.delete("/api/v1/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document with validation"""
    global documents
    
    # Validate document ID
    if not validate_document_id(document_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid document ID format"
        )
    
    # Rate limiting check
    if not rate_limit_check("delete"):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    document_id_int = int(document_id)
    document = next((doc for doc in documents if doc["id"] == document_id_int), None)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Remove file if it exists
        if os.path.exists(document["file_path"]):
            os.remove(document["file_path"])
        
        # Remove from documents list
        documents = [doc for doc in documents if doc["id"] != document_id_int]
        
        # Also remove associated chunks and embeddings
        # Note: This is a simplified approach - in a production system,
        # you'd want more efficient chunk/embedding management
        global document_chunks, document_embeddings
        original_chunk_count = len(document_chunks)
        
        # Remove chunks for this document
        document_chunks = [chunk for chunk in document_chunks if chunk["document_id"] != document_id_int]
        
        # Remove corresponding embeddings (assuming same order)
        # This is a simplified approach - a real system would need proper indexing
        chunks_removed = original_chunk_count - len(document_chunks)
        if chunks_removed > 0:
            # Remove the same number of embeddings from the end
            # This assumes chunks and embeddings are in the same order
            document_embeddings = document_embeddings[:-chunks_removed] if chunks_removed < len(document_embeddings) else []
        
        logger.info(f"Document deleted: {document['filename']} (ID: {document_id_int}), removed {chunks_removed} chunks")
        return {"message": "Document deleted successfully", "chunks_removed": chunks_removed}
        
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint for the widget - compatible with widget expectations"""
    start_time = time.time()
    
    try:
        # Rate limiting check
        if not rate_limit_check("query"):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Validate query
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        # Sanitize query
        sanitized_query = sanitize_query_string(request.query)
        if not sanitized_query:
            raise HTTPException(
                status_code=400,
                detail="Invalid query after sanitization"
            )
        
        # Check if we have any documents
        if not document_chunks:
            return ChatResponse(
                response="I don't have any documents to search through yet. Please upload some documents first so I can help answer your questions!",
                context=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
        
        # Find similar chunks
        similar_chunks = find_similar_chunks(sanitized_query, request.context_limit or 5)
        
        if not similar_chunks:
            return ChatResponse(
                response="I couldn't find any relevant information in the uploaded documents to answer your question. Could you try rephrasing your question or asking about a different topic?",
                context=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
        
        # Calculate average confidence
        avg_confidence = sum(chunk['similarity'] for chunk in similar_chunks) / len(similar_chunks)
        
        # Prepare context for response
        context_sources = format_sources_for_response(similar_chunks)
        
        # Try to use LLM if available
        if ollama_client and ollama_client.is_available():
            try:
                # Prepare context for LLM
                context_text = prepare_context_for_llm(similar_chunks)
                
                if context_text:
                    # Generate LLM response
                    llm_response = generate_llm_answer(sanitized_query, context_text)
                    
                    if llm_response:
                        return ChatResponse(
                            response=llm_response,
                            context=context_sources,
                            confidence=avg_confidence,
                            processing_time=time.time() - start_time
                        )
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
        
        # Fallback to vector search response with better formatting
        if context_sources:
            # Create a concise, direct answer from the most relevant source
            top_source = context_sources[0]
            content = top_source['content']
            
            # Extract key sentences that likely contain the answer
            sentences = content.split('.')
            key_sentences = []
            query_words = set(sanitized_query.lower().split())
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:  # Skip very short sentences
                    sentence_words = set(sentence.lower().split())
                    if query_words.intersection(sentence_words):
                        key_sentences.append(sentence)
                        if len(key_sentences) >= 2:  # Limit to 2 key sentences
                            break
            
            # Create formatted response
            if key_sentences:
                response_text = '. '.join(key_sentences) + '.'
            else:
                # Fallback to first 150 characters if no key sentences found
                response_text = content[:150] + '...' if len(content) > 150 else content
            
            # Add source information
            response_text += f"\n\nQuelle: {top_source['source_document']}"
            
            # Add additional sources if available
            if len(context_sources) > 1:
                other_sources = [s['source_document'] for s in context_sources[1:3]]
                response_text += f"\nWeitere relevante Quellen: {', '.join(other_sources)}"
            
            return ChatResponse(
                response=response_text,
                context=context_sources,
                confidence=avg_confidence,
                processing_time=time.time() - start_time
            )
        
        # No relevant content found
        return ChatResponse(
            response="I couldn't find specific information about that topic in the uploaded documents. Could you try asking about something else or upload more relevant documents?",
            context=[],
            confidence=0.0,
            processing_time=time.time() - start_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.post("/api/v1/query/optimized", response_model=ChatResponse)
async def query_documents_optimized(request: ChatRequest):
    """Optimized query with faster timeouts and better fallback formatting"""
    start_time = time.time()
    
    try:
        # Rate limiting check
        if not rate_limit_check("query"):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Validate query
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        # Sanitize query
        sanitized_query = sanitize_query_string(request.query)
        if not sanitized_query:
            raise HTTPException(
                status_code=400,
                detail="Invalid query after sanitization"
            )
        
        # Check if we have any documents
        if not document_chunks:
            return ChatResponse(
                response="Bitte laden Sie zuerst Dokumente hoch, damit ich Ihnen helfen kann!",
                query=sanitized_query,
                context=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
        
        # Find similar chunks
        similar_chunks = find_similar_chunks(sanitized_query, request.context_limit or 3)
        
        if not similar_chunks:
            return ChatResponse(
                response="Ich konnte keine relevanten Informationen in den Dokumenten finden. Versuchen Sie eine andere Fragestellung.",
                query=sanitized_query,
                context=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
        
        # Calculate average confidence
        avg_confidence = sum(chunk['similarity'] for chunk in similar_chunks) / len(similar_chunks)
        
        # Prepare context for response
        context_sources = format_sources_for_response(similar_chunks)
        
        # Try LLM with reduced timeout (5 seconds) - skip if Ollama unavailable
        if ollama_client and ollama_client.is_available():
            try:
                context_text = prepare_context_for_llm(similar_chunks[:2])  # Use fewer chunks
                
                if context_text:
                    # Create a more concise prompt
                    prompt = f"""Beantworte die Frage kurz und direkt basierend auf dem bereitgestellten Kontext.
                    
Kontext: {context_text[:1000]}

Frage: {sanitized_query}

Antwort (max 2-3 Sätze):"""
                    
                    # Quick LLM generation with timeout
                    import asyncio
                    import concurrent.futures
                    
                    def quick_llm_call():
                        return ollama_client.generate_answer(prompt, max_tokens=150)
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(quick_llm_call)
                        try:
                            llm_response = future.result(timeout=5)  # 5 second timeout
                            if llm_response:
                                return ChatResponse(
                                    response=llm_response + f"\n\nQuelle: {context_sources[0]['source_document']}",
                                    query=sanitized_query,
                                    context=context_sources,
                                    confidence=avg_confidence,
                                    processing_time=time.time() - start_time
                                )
                        except concurrent.futures.TimeoutError:
                            logger.warning("LLM timeout after 5 seconds, falling back to vector search")
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
        
        # Optimized vector search fallback
        if context_sources:
            top_source = context_sources[0]
            content = top_source['content']
            
            # Smart content extraction - find the most relevant sentences
            sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 15]
            query_words = set(sanitized_query.lower().split())
            
            # Score sentences by relevance
            scored_sentences = []
            for sentence in sentences[:5]:  # Only check first 5 sentences
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words.intersection(sentence_words))
                if overlap > 0:
                    scored_sentences.append((sentence, overlap))
            
            # Create response from best sentences
            if scored_sentences:
                scored_sentences.sort(key=lambda x: x[1], reverse=True)
                response_text = scored_sentences[0][0]
                if len(scored_sentences) > 1:
                    response_text += '. ' + scored_sentences[1][0]
                response_text += '.'
            else:
                # Use first 120 characters if no relevant sentences
                response_text = content[:120] + '...' if len(content) > 120 else content
            
            # Add source
            response_text += f"\n\nQuelle: {top_source['source_document']}"
            
            return ChatResponse(
                response=response_text,
                query=sanitized_query,
                context=context_sources,
                confidence=avg_confidence,
                processing_time=time.time() - start_time
            )
        
        # No relevant content found
        return ChatResponse(
            response="Keine relevanten Informationen gefunden. Versuchen Sie eine andere Frage.",
            query=sanitized_query,
            context=[],
            confidence=0.0,
            processing_time=time.time() - start_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in optimized query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/api/status")
async def get_api_status():
    """Status endpoint for widget health checks"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "uptime": time.time(),
        "features": {
            "vector_search": embedding_model is not None,
            "llm_generation": ollama_client is not None and ollama_client.is_available() if ollama_client else False,
            "document_processing": True
        },
        "statistics": {
            "documents_uploaded": len(documents),
            "total_chunks": len(document_chunks)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "simple_api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )