#!/usr/bin/env python3
"""
🤖 Project SUSI - Smart Universal Search Intelligence
Advanced FastAPI server for intelligent document processing and AI-powered search
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

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn

# Configure logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    logger.error(f"Some dependencies not installed: {e}")
    PDF_SUPPORT = False

# FAISS vector search import
try:
    from services.vector_search import OptimizedVectorStore, FAISSVectorSearch
    FAISS_AVAILABLE = True
    logger.info("OK: FAISS vector search loaded successfully!")
except ImportError as e:
    logger.warning(f"FAISS not available, using fallback search: {e}")
    logger.info("   Install FAISS for 10-100x faster search: pip install faiss-cpu")
    FAISS_AVAILABLE = False
    OptimizedVectorStore = None

# Async processing import
try:
    from services.async_processor import AsyncDocumentProcessor, ProcessingStatus, ProcessingJob, get_async_processor
    ASYNC_PROCESSING_AVAILABLE = True
    logger.info("OK: Async document processing loaded successfully!")
except ImportError as e:
    logger.warning(f"Async processing not available: {e}")
    ASYNC_PROCESSING_AVAILABLE = False
    AsyncDocumentProcessor = None

# Authentication import
try:
    from services.auth import AuthManager, User, UserRole, get_auth_manager
    AUTH_AVAILABLE = True
    logger.info("OK: Authentication system loaded successfully!")
except ImportError as e:
    logger.warning(f"Authentication not available: {e}")
    AUTH_AVAILABLE = False
    AuthManager = None

# Input validation import
try:
    from services.validation import InputValidator, ValidationResult, ValidationError, get_input_validator
    VALIDATION_AVAILABLE = True
    logger.info("OK: Input validation system loaded successfully!")
except ImportError as e:
    logger.warning(f"Input validation not available: {e}")
    VALIDATION_AVAILABLE = False
    InputValidator = None

# Document manager import
try:
    from services.document_manager import DocumentManager, DocumentMetadata, DocumentStatus, get_document_manager
    DOCUMENT_MANAGER_AVAILABLE = True
    logger.info("OK: Document management system loaded successfully!")
except ImportError as e:
    logger.warning(f"Document manager not available: {e}")
    DOCUMENT_MANAGER_AVAILABLE = False
    DocumentManager = None

# Smart answer engine import
try:
    from services.smart_answer import SmartAnswerEngine, get_smart_answer_engine, AnswerType
    SMART_ANSWER_AVAILABLE = True
    logger.info("OK: Smart answer engine loaded successfully!")
except ImportError as e:
    logger.warning(f"Smart answer engine not available: {e}")
    SMART_ANSWER_AVAILABLE = False
    SmartAnswerEngine = None

# Improved chunking import
try:
    from services.improved_chunking import get_improved_chunker, chunk_text_improved
    IMPROVED_CHUNKING_AVAILABLE = True
    logger.info("OK: Improved chunking system loaded successfully!")
except ImportError as e:
    logger.warning(f"Improved chunking not available: {e}")
    IMPROVED_CHUNKING_AVAILABLE = False
    chunk_text_improved = None

# Query expansion import
try:
    from services.query_expansion import get_query_expander
    QUERY_EXPANSION_AVAILABLE = True
    logger.info("OK: Query expansion loaded successfully!")
except ImportError as e:
    logger.warning(f"Query expansion not available: {e}")
    QUERY_EXPANSION_AVAILABLE = False
    get_query_expander = None

# Reranking import
try:
    from services.reranking import get_reranker
    RERANKING_AVAILABLE = True
    logger.info("OK: Chunk reranking loaded successfully!")
except ImportError as e:
    logger.warning(f"Chunk reranking not available: {e}")
    RERANKING_AVAILABLE = False
    get_reranker = None

# Hybrid search import
try:
    from services.hybrid_search import get_hybrid_search
    HYBRID_SEARCH_AVAILABLE = True
    logger.info("OK: Hybrid search loaded successfully!")
except ImportError as e:
    logger.warning(f"Hybrid search not available: {e}")
    HYBRID_SEARCH_AVAILABLE = False
    get_hybrid_search = None

# Ollama integration
try:
    from ollama_client import get_ollama_client
    OLLAMA_SUPPORT = True
except ImportError as e:
    logger.warning(f"Ollama client not available: {e}")
    OLLAMA_SUPPORT = False

# Configuration management
try:
    from config.config import config
    CONFIG_AVAILABLE = True
    logger.info("OK: Configuration system loaded successfully!")
except ImportError as e:
    logger.warning(f"Configuration system not available: {e} - using defaults")
    CONFIG_AVAILABLE = False
    config = None

# LLM Manager import
try:
    from services.llm_manager import get_llm_manager
    LLM_MANAGER_AVAILABLE = True
    logger.info("OK: LLM Manager loaded successfully!")
except ImportError as e:
    logger.warning(f"LLM Manager not available: {e}")
    LLM_MANAGER_AVAILABLE = False
    get_llm_manager = None

# Logger already configured at the top

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events"""
    # Startup
    logger.info("Starting RAG API server...")
    
    # Initialize async processor with callbacks
    if ASYNC_PROCESSING_AVAILABLE and async_processor:
        try:
            # Set callback functions for async processing
            async_processor.set_callbacks(
                extract_text_from_file,
                chunk_text,
                create_embeddings,
                lambda filename, content_type, file_size, text, chunks, embeddings: store_document_async(
                    filename, content_type, file_size, text, chunks, embeddings
                )
            )
            await async_processor.start_workers()
            logger.info("[OK] Async processor workers started")
        except Exception as e:
            logger.error(f"Failed to start async processor: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG API server...")
    if ASYNC_PROCESSING_AVAILABLE and async_processor:
        try:
            await async_processor.stop_workers()
            logger.info("[OK] Async processor workers stopped")
        except Exception as e:
            logger.error(f"Error stopping async processor: {e}")

app = FastAPI(
    title="🤖 Project SUSI - Smart Universal Search Intelligence",
    description="Advanced AI-powered document processing and intelligent search system",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware - secure configuration
# For development, allow localhost origins. For production, specify exact domains.
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8001",
]

# Allow environment variable override for production
if os.getenv("RAG_ALLOWED_ORIGINS"):
    ALLOWED_ORIGINS = os.getenv("RAG_ALLOWED_ORIGINS").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Specific origins only for security
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],  # Specific headers only
)

# Create directories
if CONFIG_AVAILABLE and config:
    UPLOAD_DIR = config.UPLOAD_DIR
    PROCESSED_DIR = config.PROCESSED_DIR
else:
    # Fallback to defaults if config not available
    UPLOAD_DIR = Path("./data/storage/uploads")
    PROCESSED_DIR = Path("./data/storage/processed")
    
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Storage initialization - try persistent storage first, fallback to memory-safe
persistent_storage = None
memory_safe_storage = None

# Try persistent storage first (preferred)
try:
    from services.persistent_storage import get_persistent_storage, init_persistent_storage
    
    # Use relative path or environment variable for database location
    db_path = os.getenv("RAG_DATABASE_PATH", "./rag_database.db")
    
    # Ensure the database path is relative to the current working directory
    if not os.path.isabs(db_path):
        db_path = os.path.join(os.getcwd(), db_path)
    
    persistent_storage = init_persistent_storage(db_path)
    storage_stats = persistent_storage.get_stats()
    logger.info(f"[OK] Persistent storage initialized - {storage_stats['documents']} documents, {storage_stats['database_size_mb']} MB")
    logger.info(f"[DB] Using database at: {db_path}")
except ImportError as e:
    logger.warning(f"Persistent storage not available: {e} - falling back to memory-safe storage")
except Exception as e:
    logger.error(f"Error initializing persistent storage: {e}")
    persistent_storage = None

# Fallback to memory-safe storage if persistent storage failed
if not persistent_storage:
    try:
        from services.memory_safe_storage import get_memory_safe_storage
        memory_safe_storage = get_memory_safe_storage()
        storage_stats = memory_safe_storage.get_stats()
        logger.info(f"[OK] Memory-safe storage initialized - capacity: {storage_stats['capacity_documents']} documents")
    except ImportError as e:
        logger.warning(f"Memory-safe storage not available: {e} - using legacy in-memory storage")

# Legacy in-memory storage (for backward compatibility)
documents = []
document_chunks = []
document_embeddings = []
document_id_counter = 1

# Initialize embedding model
embedding_model = None
vector_store = None
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("[OK] Embedding model loaded successfully")
    
    # Initialize FAISS vector store if available
    if FAISS_AVAILABLE and embedding_model:
        vector_store = OptimizedVectorStore(embedding_model)
        logger.info("[OK] FAISS vector store initialized!")
    else:
        logger.info("[WARN] Using fallback vector search (install FAISS for 10-100x speedup)")
        
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    embedding_model = None
    vector_store = None

# Initialize Ollama client
ollama_client = None
if OLLAMA_SUPPORT:
    try:
        ollama_client = get_ollama_client()
        if ollama_client.is_available():
            logger.info(f"Ollama client initialized successfully with model: {ollama_client.model}")
        else:
            logger.warning("Ollama client initialized but not available - will retry on demand")
    except Exception as e:
        logger.error(f"Failed to initialize Ollama client: {e}")
        logger.info("Will attempt to reconnect when needed")
        ollama_client = None

# Initialize async processor
async_processor = None
if ASYNC_PROCESSING_AVAILABLE:
    try:
        async_processor = get_async_processor()
        logger.info("[OK] Async document processor initialized!")
    except Exception as e:
        logger.error(f"Failed to initialize async processor: {e}")
        async_processor = None

# Initialize authentication
auth_manager = None
security = HTTPBearer(auto_error=False)
if AUTH_AVAILABLE:
    try:
        auth_manager = get_auth_manager()
        logger.info("[OK] Authentication manager initialized!")
        logger.info("Default admin user: username='admin', password='admin123' (change in production!)")
    except Exception as e:
        logger.error(f"Failed to initialize auth manager: {e}")
        auth_manager = None

# Initialize input validator
input_validator = None
if VALIDATION_AVAILABLE:
    try:
        input_validator = get_input_validator()
        logger.info("[OK] Input validator initialized!")
    except Exception as e:
        logger.error(f"Failed to initialize input validator: {e}")
        input_validator = None

# Initialize document manager
doc_manager = None
if DOCUMENT_MANAGER_AVAILABLE:
    try:
        doc_manager = get_document_manager()
        logger.info("[OK] Document manager initialized!")
    except Exception as e:
        logger.error(f"Failed to initialize document manager: {e}")
        doc_manager = None

# Initialize smart answer engine
smart_answer_engine = None
if SMART_ANSWER_AVAILABLE:
    try:
        smart_answer_engine = get_smart_answer_engine()
        logger.info("[OK] Smart answer engine initialized!")
    except Exception as e:
        logger.error(f"Failed to initialize smart answer engine: {e}")
        smart_answer_engine = None

# Initialize improved chunker
improved_chunker = None
if IMPROVED_CHUNKING_AVAILABLE:
    try:
        improved_chunker = get_improved_chunker()
        logger.info("[OK] Improved chunker initialized!")
    except Exception as e:
        logger.error(f"Failed to initialize improved chunker: {e}")
        improved_chunker = None

# Initialize query expander
query_expander = None
if QUERY_EXPANSION_AVAILABLE:
    try:
        query_expander = get_query_expander()
        logger.info("[OK] Query expander initialized!")
    except Exception as e:
        logger.error(f"Failed to initialize query expander: {e}")
        query_expander = None

# Configuration
USE_LLM_DEFAULT = True  # Try to use LLM by default
MAX_CONTEXT_LENGTH = 4000  # Maximum context length for LLM

# Performance Caching System
class FastCache:
    """Fast in-memory cache for embeddings, search results, and LLM responses"""
    def __init__(self, max_size=1000):
        self.query_cache = {}
        self.embedding_cache = {}
        self.response_cache = {}  # Cache for full LLM responses
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
    
    def get_response_cache(self, query_hash: str):
        """Get cached LLM response"""
        if query_hash in self.response_cache:
            cached_data = self.response_cache[query_hash]
            # Check if cache is still valid (30 minutes for responses)
            if time.time() - cached_data['timestamp'] < 1800:
                self.access_times[query_hash] = time.time()
                return cached_data['response']
            else:
                # Remove expired cache
                del self.response_cache[query_hash]
                if query_hash in self.access_times:
                    del self.access_times[query_hash]
        return None
    
    def set_response_cache(self, query_hash: str, response):
        """Set cached LLM response"""
        self._cleanup_if_needed()
        self.response_cache[query_hash] = {
            'response': response,
            'timestamp': time.time()
        }
        self.access_times[query_hash] = time.time()
    
    def _cleanup_if_needed(self):
        """Remove oldest entries if cache is full"""
        total_items = len(self.query_cache) + len(self.embedding_cache) + len(self.response_cache)
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
                if key in self.response_cache:
                    del self.response_cache[key]
                if key in self.access_times:
                    del self.access_times[key]
    
    def clear_all(self):
        """Clear all cached data"""
        self.query_cache.clear()
        self.embedding_cache.clear()
        self.response_cache.clear()
        self.access_times.clear()
    
    def clear(self):
        """Clear all caches"""
        self.query_cache.clear()
        self.embedding_cache.clear()
        self.response_cache.clear()
        self.access_times.clear()

# Initialize performance cache
fast_cache = FastCache(max_size=1000)

# File upload limits
MAX_FILE_SIZE = config.MAX_FILE_SIZE if CONFIG_AVAILABLE and config else 50 * 1024 * 1024  # 50MB maximum file size
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
    message: Optional[str] = None

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
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    context_limit: Optional[int] = 5

class ChatResponse(BaseModel):
    response: str
    query: Optional[str] = None
    context: Optional[List[dict]] = []
    confidence: Optional[float] = 0.0
    processing_time: Optional[float] = None

# Authentication models
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user: dict

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    role: Optional[str] = "user"

class UserResponse(BaseModel):
    user_id: str
    username: str
    email: str
    role: str
    created_at: str
    last_login: Optional[str] = None
    is_active: bool

# Document management models
class DocumentUpdate(BaseModel):
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    status: Optional[str] = None

class DocumentSearchResponse(BaseModel):
    query: str
    search_content: bool
    results: List[Dict[str, Any]]
    total_found: int
    returned: int

# Authentication dependencies
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Any]:
    """Get current authenticated user"""
    if not AUTH_AVAILABLE or not auth_manager:
        return None
    
    if not credentials:
        return None
    
    token = credentials.credentials
    user = auth_manager.verify_token(token)
    return user

async def require_auth(current_user: Any = Depends(get_current_user)) -> Any:
    """Require authentication"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user

async def require_admin(current_user: Any = Depends(require_auth)) -> Any:
    """Require admin role"""
    if not AUTH_AVAILABLE or not auth_manager:
        raise HTTPException(status_code=503, detail="Authentication not available")
    
    if not auth_manager.check_permission(current_user, UserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required"
        )
    return current_user

async def optional_auth(current_user: Any = Depends(get_current_user)) -> Optional[Any]:
    """Optional authentication (doesn't raise error if not authenticated)"""
    return current_user

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
    """
    Split text into chunks with overlap.
    
    NOTE: This is the legacy chunking function. For better results, use the improved chunker.
    """
    if not text:
        return []
    
    # Use improved chunker if available
    if IMPROVED_CHUNKING_AVAILABLE and improved_chunker:
        try:
            chunk_objects = chunk_text_improved(text, "document")
            # Extract just the text for backward compatibility
            return [chunk_obj['text'] for chunk_obj in chunk_objects]
        except Exception as e:
            logger.warning(f"Improved chunking failed, falling back to legacy: {e}")
    
    # Legacy chunking fallback
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
                # Convert cached embedding back to numpy array
                embeddings.append(np.array(cached_embedding))
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
                embeddings[idx] = embedding  # Store original numpy array, not the list!
        
        return embeddings
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        return []

def has_documents() -> bool:
    """Check if we have any documents in any storage system"""
    # Check persistent storage first
    if persistent_storage:
        try:
            stats = persistent_storage.get_stats()
            return stats.get('documents', 0) > 0
        except:
            pass
    
    # Check memory-safe storage
    if memory_safe_storage:
        try:
            stats = memory_safe_storage.get_stats()
            return stats.get('documents', 0) > 0
        except:
            pass
    
    # Check legacy storage
    return len(document_chunks) > 0

def find_similar_chunks(query: str, top_k: int = 5) -> List[dict]:
    """Find similar chunks using persistent storage with query expansion for better matches."""
    # Try persistent storage first
    if persistent_storage and embedding_model:
        try:
            logger.info(f"DEBUG: find_similar_chunks called with query: '{query}', top_k: {top_k} (using persistent storage)")
            
            # Generate query embedding
            query_embedding = embedding_model.encode([query])[0]
            
            # Search using persistent storage
            initial_results = persistent_storage.search(
                query_embedding=query_embedding,
                top_k=top_k * 3,  # Get more results for reranking
                min_similarity=0.15  # Lower threshold for initial search
            )
            
            if initial_results:
                logger.info(f"DEBUG: Persistent storage returned {len(initial_results)} initial results")
                
                # Apply reranking if available
                if RERANKING_AVAILABLE:
                    try:
                        reranker = get_reranker()
                        reranked_results = reranker.rerank_chunks(query, initial_results, top_k)
                        logger.info(f"DEBUG: Reranked to {len(reranked_results)} results")
                        return reranked_results
                    except Exception as e:
                        logger.warning(f"Reranking failed: {e}")
                
                # Apply hybrid search if available
                if HYBRID_SEARCH_AVAILABLE:
                    try:
                        hybrid_search = get_hybrid_search()
                        # Index documents for keyword search
                        hybrid_search.index_documents(initial_results)
                        # Get hybrid results
                        hybrid_results = hybrid_search.hybrid_search(query, initial_results, top_k)
                        logger.info(f"DEBUG: Hybrid search returned {len(hybrid_results)} results")
                        return hybrid_results
                    except Exception as e:
                        logger.warning(f"Hybrid search failed: {e}")
                
                return initial_results[:top_k]
            else:
                logger.info("DEBUG: No results from persistent storage")
                
        except Exception as e:
            logger.error(f"Error in persistent storage search: {e}")
    
    # Try memory-safe storage as fallback
    if memory_safe_storage and embedding_model:
        try:
            logger.info(f"DEBUG: find_similar_chunks called with query: '{query}', top_k: {top_k}")
            
            # 1. Query Expansion für bessere Treffer
            all_results = []
            queries_to_try = [query]  # Original query
            
            if QUERY_EXPANSION_AVAILABLE:
                try:
                    query_expander = get_query_expander()
                    expanded_queries = query_expander.expand_query(query, max_expansions=2)
                    queries_to_try = expanded_queries
                    logger.info(f"DEBUG: Expanded query '{query}' to {len(queries_to_try)} variants: {queries_to_try}")
                except Exception as e:
                    logger.warning(f"Query expansion failed: {e}")
            
            # 2. Suche mit allen Query-Varianten
            seen_chunks = set()  # Duplikate vermeiden
            
            for i, search_query in enumerate(queries_to_try):
                # Generate query embedding
                query_embedding = embedding_model.encode([search_query])[0]
                
                # Search using memory-safe storage
                results = memory_safe_storage.search(
                    query_embedding=query_embedding,
                    top_k=top_k * 2,  # Mehr Ergebnisse für bessere Auswahl
                    min_similarity=0.0  # No threshold here
                )
                
                # Gewichtung: Original query hat Faktor 1.0, Expansions haben 0.9, 0.8, etc.
                weight = 1.0 - (i * 0.1)
                
                for result in results:
                    chunk_key = result.get('chunk_id', result.get('document_id', 0))
                    if chunk_key not in seen_chunks:
                        seen_chunks.add(chunk_key)
                        
                        # Gewichtete Similarity
                        weighted_similarity = result['similarity'] * weight
                        
                        all_results.append({
                            'chunk_id': chunk_key,
                            'similarity': weighted_similarity,
                            'original_similarity': result['similarity'],
                            'query_variant': search_query,
                            'weight': weight,
                            'text': result.get('text', ''),
                            'filename': result.get('filename', 'Unknown'),
                            'document_id': result.get('document_id', 0)
                        })
            
            # 3. Sortiere nach gewichteter Similarity
            all_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # 4. Top-K auswählen
            final_results = all_results[:top_k]
            
            logger.info(f"DEBUG: Combined search with {len(queries_to_try)} query variants returned {len(final_results)} unique results")
            for i, result in enumerate(final_results[:3]):
                logger.info(f"DEBUG: Result {i}: similarity={result['similarity']:.4f} (original: {result['original_similarity']:.4f}, query: '{result['query_variant']}')")
            
            # 5. Format für Rückgabe - preserve original search results with full data
            similar_chunks = []
            for result in final_results:
                # Memory-safe storage results already contain all needed data
                similar_chunks.append({
                    'chunk_id': result['chunk_id'],
                    'similarity': result['similarity'],
                    'text': result.get('text', ''),
                    'filename': result.get('filename', 'Unknown'),
                    'document_id': result.get('document_id', 0)
                })
            
            logger.info(f"DEBUG: Returning {len(similar_chunks)} chunks to caller")
            return similar_chunks
        except Exception as e:
            logger.error(f"Memory-safe storage search failed: {e}")
            # Fall through to legacy search
    
    # Legacy search
    if not embedding_model or (not document_embeddings and (not vector_store or not FAISS_AVAILABLE)):
        logger.warning("No embedding model or documents available")
        return []
    
    try:
        # Check cache first
        query_hash = hashlib.md5(f"{query}_{top_k}".encode()).hexdigest()
        cached_result = fast_cache.get_query_cache(query_hash)
        if cached_result is not None:
            return cached_result
        
        # Use FAISS if available and has data
        if FAISS_AVAILABLE and vector_store and vector_store.faiss_search.index_size > 0:
            logger.debug("Using FAISS vector search")
            
            # Perform FAISS search
            results = vector_store.similarity_search(query, k=top_k)
            
            # Format results for compatibility
            similarities = []
            for i, (text, score, metadata) in enumerate(results):
                # Find the chunk index in document_chunks for compatibility
                chunk_id = -1
                for j, chunk in enumerate(document_chunks):
                    if chunk.get("text", "").strip() == text.strip():
                        chunk_id = j
                        break
                
                if chunk_id >= 0:
                    similarities.append({
                        'chunk_id': chunk_id,
                        'similarity': float(score)
                    })
            
            # Cache and return
            fast_cache.set_query_cache(query_hash, similarities)
            return similarities
        
        # Check if we need to migrate existing documents to FAISS
        elif FAISS_AVAILABLE and vector_store and document_embeddings and len(document_embeddings) > 0:
            logger.info(f"Migrating {len(document_embeddings)} existing embeddings to FAISS")
            
            # Prepare data for FAISS
            chunk_texts = [chunk["text"] for chunk in document_chunks]
            metadatas = []
            
            for i, chunk in enumerate(document_chunks):
                metadatas.append({
                    "document_id": chunk.get("document_id", 0),
                    "chunk_index": i,
                    "filename": documents[chunk["document_id"]]["filename"] if chunk.get("document_id", 0) < len(documents) else "Unknown"
                })
            
            # Build FAISS index
            embeddings_array = np.array(document_embeddings)
            chunk_ids = list(range(len(document_chunks)))
            vector_store.faiss_search.build_index(
                embeddings_array,
                chunk_ids,
                chunk_texts,
                metadatas
            )
            logger.info("[OK] Migration to FAISS complete! Future searches will be much faster.")
            
            # Now perform the search with FAISS
            results = vector_store.similarity_search(query, k=top_k)
            similarities = []
            for text, score, metadata in results:
                chunk_id = metadata.get("chunk_index", -1)
                if chunk_id >= 0:
                    similarities.append({
                        'chunk_id': chunk_id,
                        'similarity': float(score)
                    })
            
            fast_cache.set_query_cache(query_hash, similarities)
            return similarities
        
        else:
            # Fallback to original cosine similarity search
            logger.debug("Using fallback cosine similarity search")
            
            if not document_embeddings:
                return []
            
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
                result = similarities[:min(top_k, 3)]
            else:
                result = similarities[:top_k]
            
            # Cache the result
            fast_cache.set_query_cache(query_hash, result)
            return result
            
    except Exception as e:
        logger.error(f"Error in find_similar_chunks: {str(e)}")
        return []

def prepare_context_for_llm(similar_chunks: List[dict], max_length: int = 3000) -> str:
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
    
    # Check if we're using hybrid storage results (has 'text' field) or legacy format
    if similar_chunks and 'text' in similar_chunks[0]:
        # Hybrid storage format - text is directly available
        for chunk_data in similar_chunks:
            filename = chunk_data.get('filename', 'Unknown')
            text = chunk_data.get('text', '')
            
            # Format: [Document: filename] content
            chunk_text = f"[Document: {filename}]\n{text}\n"
            
            # Check if adding this chunk would exceed max length
            if current_length + len(chunk_text) > max_length:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
    else:
        # Legacy format - need to look up in document_chunks
        for chunk_data in similar_chunks:
            chunk_id = chunk_data['chunk_id']
            if chunk_id < len(document_chunks):
                chunk_info = document_chunks[chunk_id]
                
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
    logger.info(f"generate_llm_answer called with query: '{query[:50]}...', context length: {len(context)}")
    
    if not ollama_client:
        logger.warning("Ollama client is None")
        return None
        
    logger.info(f"Ollama client exists, checking availability...")
    if not ollama_client.is_available():
        logger.warning("Ollama not available for answer generation")
        return None
    
    logger.info(f"Ollama available, calling generate_answer...")
    try:
        answer = ollama_client.generate_answer(query, context)
        logger.info(f"Ollama generate_answer returned: {type(answer)}, length: {len(answer) if answer else 0}")
        return answer
    except Exception as e:
        logger.error(f"Error generating LLM answer: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def format_sources_for_response(similar_chunks: List[dict]) -> List[dict]:
    """
    Format source information for API response with download links instead of full content
    
    Args:
        similar_chunks: List of similar chunks with metadata
    
    Returns:
        List[dict]: Formatted source information with download links
    """
    sources = []
    
    logger.info(f"DEBUG: format_sources_for_response called with {len(similar_chunks)} chunks")
    if similar_chunks:
        logger.info(f"DEBUG: First chunk keys: {list(similar_chunks[0].keys())}")
        logger.info(f"DEBUG: Has 'text' in first chunk: {'text' in similar_chunks[0]}")
        logger.info(f"DEBUG: Legacy document_chunks length: {len(document_chunks)}")
    
    # Check if we're using hybrid storage results
    if similar_chunks and 'text' in similar_chunks[0]:
        # Hybrid storage format - already has all the data
        logger.info("DEBUG: Using hybrid storage format")
        for chunk_data in similar_chunks:
            filename = chunk_data.get('filename', 'Unknown')
            document_id = chunk_data.get('document_id', 0)
            download_url = f"/api/v1/documents/{document_id}/download" if filename != 'Unknown' else None
            sources.append({
                "document_id": document_id,
                "source_document": filename,
                "download_url": download_url,
                "content_preview": chunk_data.get('text', '')[:200] + "..." if len(chunk_data.get('text', '')) > 200 else chunk_data.get('text', ''),
                "similarity_score": chunk_data.get('similarity', 0.0),
                "chunk_id": chunk_data.get('chunk_id', 0)
            })
    else:
        # Legacy format - need to look up in document_chunks
        logger.info("DEBUG: Using legacy format")
        for chunk_data in similar_chunks:
            chunk_id = chunk_data['chunk_id']
            logger.info(f"DEBUG: Looking for chunk_id {chunk_id} in {len(document_chunks)} legacy chunks")
            if chunk_id < len(document_chunks):
                chunk_info = document_chunks[chunk_id]
                filename = chunk_info['filename']
                document_id = chunk_info['document_id']
                download_url = f"/api/v1/documents/{document_id}/download" if filename != 'Unknown' else None
                sources.append({
                    "document_id": document_id,
                    "source_document": filename,
                    "download_url": download_url,
                    "content_preview": chunk_info['text'][:200] + "..." if len(chunk_info['text']) > 200 else chunk_info['text'],
                    "similarity_score": chunk_data['similarity'],
                    "chunk_id": chunk_id
                })
            else:
                logger.error(f"DEBUG: Chunk ID {chunk_id} not found in legacy storage!")
    
    logger.info(f"DEBUG: Formatted {len(sources)} sources")
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
    if not filename or not filename.strip():
        return "untitled"
    
    # Remove path components and normalize
    filename = os.path.basename(filename.strip())
    
    # Check for OS-reserved names (Windows)
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5',
        'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4',
        'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    
    name_part = os.path.splitext(filename)[0].upper()
    if name_part in reserved_names:
        filename = f"file_{filename}"
    
    # Remove or replace dangerous characters
    sanitized = "".join(c for c in filename if c in ALLOWED_FILENAME_CHARS)
    
    # Remove leading dots and ensure it doesn't start with a dot
    sanitized = sanitized.lstrip('.')
    if not sanitized or sanitized[0] == '.':
        sanitized = "file_" + sanitized
    
    # Ensure filename is not too long
    if len(sanitized) > MAX_FILENAME_LENGTH:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:MAX_FILENAME_LENGTH - len(ext)] + ext
    
    # Ensure filename is not empty and has valid extension
    if not sanitized or sanitized.isspace():
        sanitized = "untitled"
    
    # Validate extension
    if '.' in sanitized:
        ext = sanitized.lower().split('.')[-1]
        if f'.{ext}' not in {'.pdf', '.docx', '.txt', '.csv'}:
            sanitized = os.path.splitext(sanitized)[0] + '.txt'
    else:
        sanitized += '.txt'
    
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
    Enhanced validation and sanitization for query requests
    Uses the comprehensive validation system when available
    
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
    
    # Use comprehensive validation system if available
    if VALIDATION_AVAILABLE and input_validator:
        try:
            # Validate search request with comprehensive validation
            search_validation = input_validator.validate_search_request(
                query=request.query,
                limit=request.top_k,
                offset=None
            )
            
            # Check query validation
            query_validation = search_validation['query']
            if not query_validation.is_valid:
                validation_result["valid"] = False
                validation_result["errors"].extend(query_validation.errors)
                return validation_result
            
            sanitized_query = query_validation.sanitized_value
            
            # Add warnings from query validation
            if query_validation.warnings:
                validation_result["warnings"].extend(query_validation.warnings)
            
            # Validate top_k if provided
            validated_top_k = request.top_k
            if request.top_k is not None and 'limit' in search_validation:
                limit_validation = search_validation['limit']
                if not limit_validation.is_valid:
                    validation_result["valid"] = False
                    validation_result["errors"].extend(limit_validation.errors)
                    return validation_result
                
                validated_top_k = limit_validation.sanitized_value
                if limit_validation.warnings:
                    validation_result["warnings"].extend(limit_validation.warnings)
            
        except Exception as e:
            logger.warning(f"Comprehensive validation failed, falling back to basic validation: {e}")
            # Fall back to basic validation
            return _basic_validate_query_request(request)
    else:
        # Fall back to basic validation
        return _basic_validate_query_request(request)
    
    # Validate use_llm parameter
    if request.use_llm is not None and not isinstance(request.use_llm, bool):
        validation_result["valid"] = False
        validation_result["errors"].append("use_llm must be a boolean")
        return validation_result
    
    # Create sanitized request
    sanitized_request = QueryRequest(
        query=sanitized_query,
        top_k=validated_top_k,
        use_llm=request.use_llm
    )
    
    validation_result["sanitized_request"] = sanitized_request
    return validation_result

def _basic_validate_query_request(request: QueryRequest) -> Dict[str, Any]:
    """
    Basic validation for when comprehensive validation is not available
    This is the original validation logic
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
            "document_processing": PDF_SUPPORT,
            "faiss_search": FAISS_AVAILABLE,
            "async_processing": ASYNC_PROCESSING_AVAILABLE,
            "authentication": AUTH_AVAILABLE,
            "input_validation": VALIDATION_AVAILABLE,
            "document_management": DOCUMENT_MANAGER_AVAILABLE,
            "smart_answers": SMART_ANSWER_AVAILABLE,
            "improved_chunking": IMPROVED_CHUNKING_AVAILABLE
        },
        "statistics": {
            "documents_uploaded": len(documents),
            "total_chunks": len(document_chunks),
            "embeddings_created": len(document_embeddings)
        }
    }
    
    # Add storage information - prioritize persistent storage
    if persistent_storage:
        try:
            storage_stats = persistent_storage.get_stats()
            status["storage"] = storage_stats
            status["features"]["persistent_storage"] = True
            status["features"]["memory_safe_storage"] = False
            
            # Update statistics with persistent storage data
            status["statistics"]["documents_uploaded"] = storage_stats.get('documents', 0)
            status["statistics"]["total_chunks"] = storage_stats.get('chunks', 0)
            status["statistics"]["embeddings_created"] = storage_stats.get('embeddings', 0)
            
        except Exception as e:
            logger.error(f"Error getting persistent storage stats: {e}")
            status["features"]["persistent_storage"] = False
    
    # Add memory-safe storage information if persistent storage not available
    elif memory_safe_storage:
        try:
            storage_stats = memory_safe_storage.get_stats()
            status["storage"] = storage_stats
            status["features"]["memory_safe_storage"] = True
            status["features"]["persistent_storage"] = False
            
            # Update statistics with memory-safe storage data
            status["statistics"]["documents_uploaded"] = storage_stats.get('documents', len(documents))
            status["statistics"]["total_chunks"] = storage_stats.get('chunks', len(document_chunks))
            status["statistics"]["embeddings_created"] = storage_stats.get('embeddings', len(document_embeddings))
            
            # Add storage warning if near capacity
            if storage_stats.get('is_near_limit', False):
                status["storage_warning"] = f"Storage capacity warning: {storage_stats['usage_percentage_docs']}% documents, {storage_stats['usage_percentage_chunks']}% chunks used"
                
        except Exception as e:
            status["storage"] = {"error": f"Could not retrieve storage stats: {e}"}
    else:
        status["features"]["memory_safe_storage"] = False
    
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

# Authentication endpoints
@app.post("/api/auth/login", response_model=LoginResponse)
async def login(login_data: LoginRequest):
    """Authenticate user and return JWT tokens"""
    if not AUTH_AVAILABLE or not auth_manager:
        raise HTTPException(status_code=503, detail="Authentication not available")
    
    tokens = auth_manager.login(login_data.username, login_data.password)
    if not tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    return LoginResponse(**tokens)

@app.post("/api/auth/logout")
async def logout(current_user: Any = Depends(require_auth)):
    """Logout user (revoke current token)"""
    # Note: In a real implementation, we'd get the token from the request
    # For now, just return success
    return {"message": "Logged out successfully"}

@app.post("/api/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate, admin_user: Any = Depends(require_admin)):
    """Register a new user (admin only)"""
    if not AUTH_AVAILABLE or not auth_manager:
        raise HTTPException(status_code=503, detail="Authentication not available")
    
    try:
        role = UserRole(user_data.role) if user_data.role else UserRole.USER
        new_user = auth_manager.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            role=role
        )
        
        return UserResponse(
            user_id=new_user.user_id,
            username=new_user.username,
            email=new_user.email,
            role=new_user.role.value,
            created_at=new_user.created_at.isoformat(),
            last_login=new_user.last_login.isoformat() if new_user.last_login else None,
            is_active=new_user.is_active
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: Any = Depends(require_auth)):
    """Get current user information"""
    return UserResponse(
        user_id=current_user.user_id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role.value,
        created_at=current_user.created_at.isoformat(),
        last_login=current_user.last_login.isoformat() if current_user.last_login else None,
        is_active=current_user.is_active
    )

@app.get("/api/auth/users", response_model=List[UserResponse])
async def list_users(admin_user: Any = Depends(require_admin)):
    """List all users (admin only)"""
    if not AUTH_AVAILABLE or not auth_manager:
        raise HTTPException(status_code=503, detail="Authentication not available")
    
    users = auth_manager.user_store.list_users()
    return [
        UserResponse(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            role=user.role.value,
            created_at=user.created_at.isoformat(),
            last_login=user.last_login.isoformat() if user.last_login else None,
            is_active=user.is_active
        )
        for user in users
    ]

def check_for_duplicate_document(sanitized_filename: str, original_filename: str) -> Dict[str, Any]:
    """
    Check if a document with the same filename already exists in any storage system.
    Returns information about duplicate status and existing document if found.
    """
    try:
        # Check persistent storage first
        if persistent_storage:
            try:
                # Get all documents and check for filename matches
                stored_docs = persistent_storage.get_all_documents(limit=1000)
                for doc in stored_docs:
                    doc_filename = doc.get("filename", "")
                    doc_original = doc.get("original_filename", "")
                    
                    # Check both sanitized and original filename matches
                    if (doc_filename == sanitized_filename or 
                        doc_original == original_filename or
                        doc_filename == original_filename):
                        
                        return {
                            "is_duplicate": True,
                            "existing_id": doc.get("id"),
                            "storage_type": "persistent",
                            "document_info": {
                                "id": doc.get("id"),
                                "filename": doc_filename,
                                "original_filename": doc_original,
                                "upload_date": str(doc.get("upload_timestamp", "unknown")),
                                "chunks_count": doc.get("chunk_count", 0)
                            }
                        }
            except Exception as e:
                logger.warning(f"Error checking duplicates in persistent storage: {e}")
        
        # Check memory-safe storage if persistent storage not available or failed
        if memory_safe_storage:
            try:
                stored_docs = memory_safe_storage.get_all_documents(limit=1000)
                for doc in stored_docs:
                    doc_filename = doc.get("filename", "")
                    doc_original = doc.get("metadata", {}).get("original_filename", "")
                    
                    if (doc_filename == sanitized_filename or 
                        doc_original == original_filename or
                        doc_filename == original_filename):
                        
                        return {
                            "is_duplicate": True,
                            "existing_id": doc.get("id"),
                            "storage_type": "memory_safe",
                            "document_info": {
                                "id": doc.get("id"),
                                "filename": doc_filename,
                                "original_filename": doc_original,
                                "upload_date": str(doc.get("upload_timestamp", "unknown")),
                                "chunks_count": doc.get("chunk_count", 0)
                            }
                        }
            except Exception as e:
                logger.warning(f"Error checking duplicates in memory-safe storage: {e}")
        
        # Check legacy storage
        for doc in documents:
            doc_filename = doc.get("filename", "")
            doc_original = doc.get("original_filename", "")
            
            if (doc_filename == sanitized_filename or 
                doc_original == original_filename or
                doc_filename == original_filename):
                
                return {
                    "is_duplicate": True,
                    "existing_id": doc.get("id"),
                    "storage_type": "legacy",
                    "document_info": {
                        "id": doc.get("id"),
                        "filename": doc_filename,
                        "original_filename": doc_original,
                        "upload_date": doc.get("upload_date", "unknown"),
                        "chunks_count": doc.get("chunks_count", 0)
                    }
                }
        
        # No duplicate found
        return {
            "is_duplicate": False,
            "existing_id": None,
            "storage_type": None,
            "document_info": None
        }
        
    except Exception as e:
        logger.error(f"Error in duplicate checking: {e}")
        # Return safe default - assume not duplicate to allow upload
        return {
            "is_duplicate": False,
            "existing_id": None,
            "storage_type": None,
            "document_info": None
        }

def get_fast_max_tokens() -> int:
    """
    Get appropriate max_tokens for fast queries based on model configuration
    
    Returns:
        int: Token limit for fast queries
    """
    try:
        if LLM_MANAGER_AVAILABLE:
            llm_manager = get_llm_manager()
            model_config = llm_manager.get_model_config()
            configured_max = model_config.get("max_tokens", 2048)
            
            # For fast queries, use the configured max tokens
            # Allow models to use their full capability for better responses
            return configured_max
        else:
            # Fallback to higher default for better responses
            return 2048
    except Exception as e:
        logger.warning(f"Error getting fast max tokens: {e}")
        return 2048

def check_and_reinitialize_llm() -> Dict[str, Any]:
    """
    Check LLM availability and attempt to reinitialize if needed.
    Returns status information about LLM connectivity.
    """
    global ollama_client
    
    try:
        # First check if current client is working
        if ollama_client:
            try:
                if ollama_client.is_available():
                    return {
                        "status": "ok",
                        "available": True,
                        "model": getattr(ollama_client, 'model', 'unknown'),
                        "message": "LLM is working correctly"
                    }
            except Exception as e:
                logger.warning(f"Current LLM client failed health check: {e}")
        
        # Try to reinitialize if not working or not available
        if OLLAMA_SUPPORT:
            try:
                logger.info("Attempting to reinitialize LLM client...")
                
                # Load current model from config
                import yaml
                try:
                    config_path = config.get_llm_config_path() if CONFIG_AVAILABLE and config else Path("./config/llm_config.yaml")
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    default_model_key = config.get("default_model", "tinyllama")
                    models = config.get("models", {})
                    
                    if default_model_key in models:
                        model_name = models[default_model_key]["name"]
                        logger.info(f"Using model from config: {model_name}")
                    else:
                        model_name = "tinyllama:latest"
                        logger.warning(f"Default model key '{default_model_key}' not found, using fallback")
                        
                except Exception as config_error:
                    logger.error(f"Error loading config: {config_error}")
                    model_name = "tinyllama:latest"
                
                # Create new client instance with config model
                from ollama_client import OllamaClient
                new_client = OllamaClient(
                    base_url="http://localhost:11434",
                    model=model_name,
                    timeout=300
                )
                
                if new_client and new_client.is_available():
                    ollama_client = new_client
                    logger.info(f"LLM client successfully reinitialized with model: {ollama_client.model}")
                    return {
                        "status": "reinitialized",
                        "available": True,
                        "model": ollama_client.model,
                        "message": f"LLM client reinitialized successfully with model: {ollama_client.model}"
                    }
                else:
                    return {
                        "status": "failed",
                        "available": False,
                        "model": model_name,
                        "message": f"LLM client could not be initialized with {model_name} - Ollama may not be running or model not available"
                    }
            except Exception as e:
                logger.error(f"Failed to reinitialize LLM client: {e}")
                return {
                    "status": "error",
                    "available": False,
                    "model": None,
                    "message": f"LLM reinitialization failed: {str(e)}"
                }
        else:
            return {
                "status": "not_supported",
                "available": False,
                "model": None,
                "message": "Ollama support not available"
            }
    
    except Exception as e:
        logger.error(f"Error checking LLM status: {e}")
        return {
            "status": "error", 
            "available": False,
            "model": None,
            "message": f"Error checking LLM: {str(e)}"
        }

@app.get("/api/v1/llm/status")
async def get_llm_status():
    """Get LLM status and attempt reconnection if needed"""
    return check_and_reinitialize_llm()

@app.post("/api/v1/llm/reconnect")
async def reconnect_llm():
    """Force LLM reconnection with current config"""
    global ollama_client
    ollama_client = None  # Force reinitialization
    return check_and_reinitialize_llm()

@app.post("/api/v1/llm/reload-config")
async def reload_llm_config():
    """Reload LLM configuration and reinitialize client"""
    global ollama_client
    
    try:
        # Force reinitialization
        ollama_client = None
        
        # Load current config
        import yaml
        config_path = "C:/Users/THE/open-source-rag-system/config/llm_config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        default_model_key = config.get("default_model", "tinyllama")
        models = config.get("models", {})
        
        if default_model_key in models:
            model_name = models[default_model_key]["name"]
            logger.info(f"Reloading config with model: {model_name}")
            
            # Create new client with the config model
            from ollama_client import OllamaClient
            ollama_client = OllamaClient(
                base_url="http://localhost:11434",
                model=model_name,
                timeout=300
            )
            
            if ollama_client.is_available():
                return {
                    "success": True,
                    "message": f"Configuration reloaded with model: {model_name}",
                    "model": model_name,
                    "model_key": default_model_key,
                    "available": True
                }
            else:
                return {
                    "success": False,
                    "message": f"Configuration reloaded but model {model_name} not available",
                    "model": model_name,
                    "model_key": default_model_key,
                    "available": False,
                    "suggestion": f"Try: ollama pull {model_name}"
                }
        else:
            return {
                "success": False,
                "message": f"Default model key '{default_model_key}' not found in config",
                "model_key": default_model_key,
                "available": False
            }
            
    except Exception as e:
        logger.error(f"Error reloading LLM config: {e}")
        return {
            "success": False,
            "message": f"Error reloading configuration: {str(e)}",
            "error": str(e)
        }

@app.post("/api/v1/llm/preload")
async def preload_llm():
    """Preload the current LLM model to reduce response time"""
    global ollama_client
    
    if not ollama_client:
        llm_status = check_and_reinitialize_llm()
        if not llm_status.get("available", False):
            return {
                "success": False,
                "message": "LLM not available for preloading",
                "error": llm_status.get("message", "Unknown error")
            }
    
    try:
        start_time = time.time()
        success = ollama_client.preload_model()
        preload_time = time.time() - start_time
        
        if success:
            return {
                "success": True,
                "message": f"Model {ollama_client.model} preloaded successfully",
                "model": ollama_client.model,
                "preload_time": round(preload_time, 2)
            }
        else:
            return {
                "success": False,
                "message": f"Failed to preload model {ollama_client.model}",
                "model": ollama_client.model,
                "preload_time": round(preload_time, 2)
            }
            
    except Exception as e:
        logger.error(f"Error preloading LLM: {e}")
        return {
            "success": False,
            "message": f"Error preloading LLM: {str(e)}",
            "error": str(e)
        }

@app.get("/api/v1/llm/models")
async def get_available_models():
    """Get available LLM models and current selection"""
    try:
        # Try to load LLM config
        import yaml
        config_path = "C:/Users/THE/open-source-rag-system/config/llm_config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return {
            "models": config.get("models", {}),
            "current_model": config.get("default_model", "unknown"),
            "available": True
        }
    except Exception as e:
        logger.error(f"Error loading LLM models: {e}")
        return {
            "models": {
                "mistral": {
                    "name": "mistral:latest",
                    "description": "Currently available model"
                }
            },
            "current_model": "mistral",
            "available": False,
            "error": str(e)
        }

@app.post("/api/v1/llm/switch")
async def switch_llm_model(request: dict):
    """Switch the active LLM model with proper reinitialization"""
    try:
        model_key = request.get("model")
        if not model_key:
            raise HTTPException(status_code=400, detail="Model key required")
        
        # Load current config
        import yaml
        config_path = "C:/Users/THE/open-source-rag-system/config/llm_config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check if model exists
        if model_key not in config.get("models", {}):
            raise HTTPException(status_code=400, detail=f"Model '{model_key}' not found in configuration")
        
        model_info = config["models"][model_key]
        model_name = model_info["name"]
        
        # Update default model
        config["default_model"] = model_key
        
        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Force LLM client reinitialization with new model
        global ollama_client
        try:
            logger.info(f"Switching from current model to: {model_name}")
            
            # Clear existing client
            ollama_client = None
            
            # Import fresh client
            from ollama_client import OllamaClient
            logger.info(f"Reinitializing Ollama client with model: {model_name}")
            
            # Create new client instance with the specific model
            ollama_client = OllamaClient(
                base_url="http://localhost:11434",
                model=model_name,
                timeout=300
            )
            
            # Test the new model availability
            if ollama_client.is_available():
                logger.info(f"Successfully switched to model: {model_name}")
                return {
                    "success": True,
                    "message": f"Successfully switched to {model_name}",
                    "new_model": model_name,
                    "model_key": model_key,
                    "llm_available": True,
                    "description": model_info.get("description", ""),
                    "max_tokens": model_info.get("max_tokens", 2048),
                    "context_length": model_info.get("context_length", 4096)
                }
            else:
                # Model switch failed, try to pull the model
                logger.warning(f"Model {model_name} not available, attempting to pull...")
                
                # Try to pull model
                pull_success = ollama_client.pull_model(model_name)
                if pull_success:
                    # Test again after pulling
                    if ollama_client.is_available():
                        logger.info(f"Successfully pulled and switched to model: {model_name}")
                        return {
                            "success": True,
                            "message": f"Successfully pulled and switched to {model_name}",
                            "new_model": model_name,
                            "model_key": model_key,
                            "llm_available": True,
                            "description": model_info.get("description", ""),
                            "pulled": True
                        }
                    else:
                        logger.error(f"Model {model_name} pulled but still not available")
                        return {
                            "success": False,
                            "message": f"Model {model_name} was pulled but is not responding",
                            "new_model": model_name,
                            "model_key": model_key,
                            "llm_available": False,
                            "error": "Model pulled but not responding",
                            "suggestion": f"Check if Ollama is running and try again"
                        }
                else:
                    logger.error(f"Failed to pull model {model_name}")
                    return {
                        "success": False,
                        "message": f"Failed to pull model {model_name}",
                        "new_model": model_name,
                        "model_key": model_key,
                        "llm_available": False,
                        "error": "Model pull failed",
                        "suggestion": f"Try manually: ollama pull {model_name}"
                    }
        
        except Exception as init_error:
            logger.error(f"Error reinitializing Ollama client: {init_error}")
            return {
                "success": False,
                "message": f"Failed to initialize with {model_name}: {str(init_error)}",
                "new_model": model_name,
                "model_key": model_key,
                "llm_available": False,
                "error": str(init_error)
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching LLM model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to switch model: {str(e)}")

@app.post("/api/v1/documents", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...), current_user: Optional[Any] = Depends(optional_auth)):
    """Upload a document for processing with comprehensive validation"""
    global document_id_counter, memory_safe_storage
    
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
        
        # Check for duplicate documents BEFORE processing
        duplicate_check = check_for_duplicate_document(sanitized_filename, file.filename)
        if duplicate_check["is_duplicate"]:
            logger.info(f"Duplicate document detected: {file.filename}")
            return {
                "message": "Document already exists in the database",
                "document_id": duplicate_check["existing_id"],
                "filename": sanitized_filename,
                "original_filename": file.filename,
                "status": "duplicate",
                "existing_document": duplicate_check["document_info"],
                "action": "skipped_duplicate"
            }
        
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
        
        # Try memory-safe storage first
        doc_id_assigned = document_id_counter
        
        # DEBUG: Check memory-safe storage status
        # Try persistent storage first
        if persistent_storage:
            logger.info(f"Using persistent storage for document upload")
        elif memory_safe_storage:
            logger.info(f"Using memory-safe storage for document upload")
        else:
            logger.warning(f"Using legacy in-memory storage for document upload")
        
        if persistent_storage:
            try:
                logger.info(f"Attempting to add document to persistent storage: {sanitized_filename}")
                logger.info(f"Chunks created: {len(chunks)}, Embeddings created: {len(embeddings) if embeddings else 0}")
                
                # Add document with chunks and embeddings to persistent storage
                metadata = {
                    "original_filename": file.filename,
                    "file_type": file.content_type,
                    "file_size": len(content),
                    "upload_date": datetime.now().isoformat(),
                    "file_path": str(file_path),
                    "chunk_count": len(chunks)
                }
                
                doc_id = persistent_storage.add_document(
                    filename=sanitized_filename,
                    chunks=chunks,
                    embeddings=embeddings,
                    metadata=metadata
                )
                
                logger.info(f"[OK] Document {sanitized_filename} stored in persistent storage with ID: {doc_id}")
                
                # Move to processed directory
                processed_path = PROCESSED_DIR / sanitized_filename
                file_path.rename(processed_path)
                
                # Clear cache since we have new data
                fast_cache.clear_all()
                
                return {
                    "message": "Document uploaded and processed successfully",
                    "document_id": doc_id,
                    "filename": sanitized_filename,
                    "chunks_created": len(chunks),
                    "embeddings_created": len(embeddings) if embeddings else 0,
                    "storage_type": "persistent"
                }
                
            except Exception as e:
                logger.error(f"Failed to store in persistent storage: {e}")
                logger.info("Falling back to memory-safe storage")
        
        if memory_safe_storage and not persistent_storage:
            try:
                logger.info(f"Attempting to add document to memory-safe storage: {sanitized_filename}")
                logger.info(f"Memory-safe storage stats before: {memory_safe_storage.get_stats()}")
                logger.info(f"Chunks created: {len(chunks)}, Embeddings created: {len(embeddings) if embeddings else 0}")
                logger.info(f"Embeddings type: {type(embeddings)}, First embedding type: {type(embeddings[0]) if embeddings and len(embeddings) > 0 else 'N/A'}")
                
                # Add document with chunks and embeddings to memory-safe storage
                metadata = {
                    "original_filename": file.filename,
                    "file_type": file.content_type,
                    "file_size": len(content),
                    "upload_date": datetime.now().isoformat(),
                    "file_path": str(file_path),
                    "text_preview": extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text
                }
                
                doc_id_assigned = memory_safe_storage.add_document(
                    filename=sanitized_filename,
                    chunks=chunks,
                    embeddings=embeddings,
                    metadata=metadata
                )
                
                logger.info(f"Document added to memory-safe storage with ID: {doc_id_assigned}")
                logger.info(f"Memory-safe storage stats after: {memory_safe_storage.get_stats()}")
                
                # Check storage capacity
                storage_stats = memory_safe_storage.get_stats()
                if storage_stats.get('is_near_limit', False):
                    logger.warning(f"Storage capacity warning: {storage_stats['usage_percentage_docs']}% documents, {storage_stats['usage_percentage_chunks']}% chunks")
                    
            except MemoryError as e:
                logger.error(f"Memory limit reached: {e}")
                raise HTTPException(
                    status_code=507,
                    detail=str(e)
                )
            except Exception as e:
                logger.error(f"Memory-safe storage failed, falling back to legacy: {e}")
                import traceback
                logger.error(f"Exception details: {traceback.format_exc()}")
                memory_safe_storage = None  # Fall back to legacy
        else:
            logger.info("No persistent or memory-safe storage available, using legacy storage")
        
        # Legacy storage (fallback or if memory-safe not available)
        if not memory_safe_storage:
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
            
            # Add to FAISS vector store if available
            if FAISS_AVAILABLE and vector_store and chunks and embeddings:
                logger.info(f"Adding {len(chunks)} chunks to FAISS vector store")
                
                chunk_texts = chunks
                metadatas = [{
                    "document_id": document_id_counter,
                    "chunk_index": doc_chunk_start + i,
                    "filename": sanitized_filename
                } for i in range(len(chunks))]
                
                try:
                    vector_store.add_documents(chunk_texts, metadatas)
                    logger.info("[OK] Added chunks to FAISS vector store")
                except Exception as e:
                    logger.warning(f"Failed to add chunks to FAISS: {e}")
        
        # Create document record
        document = {
            "id": doc_id_assigned,
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
        
        # Only append to legacy documents if not using memory-safe storage
        if not memory_safe_storage:
            documents.append(document)
        logger.info(f"Document processed successfully: {sanitized_filename} (ID: {doc_id_assigned}, Chunks: {len(chunks)}, Size: {len(content):,} bytes)")
        
        response = DocumentResponse(
            id=doc_id_assigned,
            filename=sanitized_filename,
            size=len(content),
            content_type=file.content_type,
            status="processed"
        )
        
        # Only increment counter if using legacy storage
        if not memory_safe_storage:
            document_id_counter += 1
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Handle any other unexpected errors
        logger.error(f"Unexpected error uploading document {file.filename if file else 'unknown'}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error uploading document: {str(e)}")

def store_document_async(filename: str, content_type: str, file_size: int, 
                        extracted_text: str, chunks: List[str], embeddings: List) -> int:
    """
    Helper function to store document data (used by async processor)
    Returns the document ID
    """
    global document_id_counter
    
    # Try persistent storage first
    if persistent_storage:
        try:
            metadata = {
                'original_filename': filename,
                'file_type': content_type,
                'file_size': file_size,
                'content_type': content_type,
                'upload_date': datetime.now().isoformat(),
                'text_preview': extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text
            }
            
            doc_id = persistent_storage.add_document(
                filename=filename,
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata
            )
            logger.info(f"[OK] Document stored in persistent storage with ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to store in persistent storage: {e}")
            logger.info("Falling back to memory-safe storage")
    
    # Try memory-safe storage as fallback
    if memory_safe_storage:
        try:
            metadata = {
                'original_filename': filename,
                'file_type': content_type,
                'file_size': file_size,
                'content_type': content_type,
                'upload_date': datetime.now().isoformat(),
                'text_preview': extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text
            }
            
            doc_id = memory_safe_storage.add_document(
                filename=filename,
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata
            )
            logger.info(f"[OK] Document stored in memory-safe storage with ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to store in memory-safe storage: {e}")
            logger.info("Falling back to legacy storage")
    
    # Legacy storage as last resort
    doc_chunk_start = len(document_chunks)
    for i, chunk in enumerate(chunks):
        document_chunks.append({
            'document_id': document_id_counter,
            'chunk_id': doc_chunk_start + i,
            'text': chunk,
            'filename': filename
        })
    
    # Store embeddings
    document_embeddings.extend(embeddings)
    
    # Add to FAISS vector store if available
    if FAISS_AVAILABLE and vector_store and chunks and embeddings:
        try:
            chunk_texts = chunks
            metadatas = [{
                "document_id": document_id_counter,
                "chunk_index": doc_chunk_start + i,
                "filename": filename
            } for i in range(len(chunks))]
            
            vector_store.add_documents(chunk_texts, metadatas)
            logger.info("[OK] Added chunks to FAISS vector store (async)")
        except Exception as e:
            logger.warning(f"Failed to add chunks to FAISS (async): {e}")
    
    # Create document record
    document = {
        "id": document_id_counter,
        "filename": filename,
        "original_filename": filename,
        "file_type": content_type,
        "file_size": file_size,
        "size": file_size,
        "content_type": content_type,
        "status": "processed",
        "upload_date": datetime.now().isoformat(),
        "chunks_count": len(chunks),
        "text_preview": extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text
    }
    
    documents.append(document)
    current_id = document_id_counter
    document_id_counter += 1
    
    return current_id

# Async Document Upload Endpoints

@app.post("/api/v1/documents/async")
async def upload_document_async_endpoint(file: UploadFile = File(...), current_user: Optional[Any] = Depends(optional_auth)):
    """
    Upload a document for async processing (non-blocking)
    Returns immediately with job_id for tracking progress
    """
    if not ASYNC_PROCESSING_AVAILABLE or not async_processor:
        raise HTTPException(
            status_code=503, 
            detail="Async processing not available. Use /api/v1/documents for synchronous upload."
        )
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Read file content
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Save file temporarily for validation
        temp_dir = tempfile.gettempdir()
        temp_file_path = Path(temp_dir) / f"validation_{int(time.time())}_{file.filename}"
        
        with open(temp_file_path, 'wb') as f:
            f.write(content)
        
        # Comprehensive file validation
        if VALIDATION_AVAILABLE and input_validator:
            validation_results = input_validator.validate_document_upload(
                filename=file.filename,
                file_path=str(temp_file_path),
                content_type=file.content_type
            )
            
            # Check validation results
            file_validation = validation_results['file']
            if not file_validation.is_valid:
                # Clean up temp file
                try:
                    temp_file_path.unlink()
                except:
                    pass
                raise HTTPException(
                    status_code=400,
                    detail=f"File validation failed: {'; '.join(file_validation.errors)}"
                )
            
            # Use sanitized filename
            sanitized_filename = file_validation.sanitized_value
            
            # Log warnings if any
            if file_validation.warnings:
                logger.warning(f"File upload warnings: {'; '.join(file_validation.warnings)}")
        else:
            # Fallback validation
            if len(content) > MAX_FILE_SIZE:
                try:
                    temp_file_path.unlink()
                except:
                    pass
                raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {MAX_FILE_SIZE} bytes")
            
            if file.content_type not in ALLOWED_CONTENT_TYPES:
                try:
                    temp_file_path.unlink()
                except:
                    pass
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
            
            # Sanitize filename
            sanitized_filename = sanitize_filename(file.filename)
        
        # Save file temporarily
        temp_dir = tempfile.gettempdir()
        file_path = Path(temp_dir) / f"async_{int(time.time())}_{sanitized_filename}"
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Queue for async processing
        job_id = await async_processor.queue_document(
            filename=sanitized_filename,
            file_path=str(file_path),
            content_type=file.content_type,
            file_size=len(content)
        )
        
        return {
            "job_id": job_id,
            "filename": sanitized_filename,
            "status": "queued",
            "message": "Document queued for processing",
            "status_url": f"/api/v1/documents/async/{job_id}/status",
            "estimated_time": "30-120 seconds"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error queuing async upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error queuing document: {str(e)}")

@app.get("/api/v1/documents/async/{job_id}/status")
async def get_async_job_status(job_id: str):
    """Get the status of an async processing job"""
    if not ASYNC_PROCESSING_AVAILABLE or not async_processor:
        raise HTTPException(status_code=503, detail="Async processing not available")
    
    # Validate job_id
    if VALIDATION_AVAILABLE and input_validator:
        # Validate job_id as UUID string
        try:
            # Import SanitizationLevel properly
            from services.validation import SanitizationLevel
            sanitization_level = SanitizationLevel.STRICT
        except ImportError:
            sanitization_level = None
        
        job_id_validation = input_validator.text_validator.validate_text(
            job_id, 
            max_length=100,
            sanitization_level=sanitization_level
        )
        
        if not job_id_validation.is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid job ID: {'; '.join(job_id_validation.errors)}"
            )
        
        sanitized_job_id = job_id_validation.sanitized_value
    else:
        # Basic validation
        if not job_id or len(job_id.strip()) == 0:
            raise HTTPException(status_code=400, detail="Job ID cannot be empty")
        if len(job_id) > 100:
            raise HTTPException(status_code=400, detail="Job ID too long")
        sanitized_job_id = job_id.strip()
    
    job = await async_processor.get_job_status(sanitized_job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response = {
        "job_id": job.job_id,
        "filename": job.filename,
        "status": job.status.value,
        "progress": job.progress,
        "created_at": job.created_at.isoformat(),
        "file_size": job.file_size
    }
    
    if job.started_at:
        response["started_at"] = job.started_at.isoformat()
    
    if job.completed_at:
        response["completed_at"] = job.completed_at.isoformat()
        response["processing_time"] = job.processing_time
    
    if job.status == ProcessingStatus.COMPLETED:
        response["document_id"] = job.document_id
        response["chunks_created"] = job.chunks_created
        response["message"] = "Document processed successfully"
    
    if job.status == ProcessingStatus.FAILED:
        response["error"] = job.error_message
    
    return response

@app.get("/api/v1/documents/async/jobs")
async def list_async_jobs(limit: int = 50):
    """List all async processing jobs"""
    if not ASYNC_PROCESSING_AVAILABLE or not async_processor:
        raise HTTPException(status_code=503, detail="Async processing not available")
    
    # Validate limit parameter
    if VALIDATION_AVAILABLE and input_validator:
        limit_validation = input_validator.numeric_validator.validate_integer(
            limit, min_value=1, max_value=1000
        )
        
        if not limit_validation.is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid limit parameter: {'; '.join(limit_validation.errors)}"
            )
        
        validated_limit = limit_validation.sanitized_value
    else:
        # Basic validation
        if limit < 1:
            raise HTTPException(status_code=400, detail="Limit must be at least 1")
        if limit > 1000:
            raise HTTPException(status_code=400, detail="Limit cannot exceed 1000")
        validated_limit = limit
    
    jobs = await async_processor.get_all_jobs(limit=validated_limit)
    
    return {
        "jobs": [
            {
                "job_id": job.job_id,
                "filename": job.filename,
                "status": job.status.value,
                "progress": job.progress,
                "created_at": job.created_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "processing_time": job.processing_time,
                "document_id": job.document_id,
                "chunks_created": job.chunks_created
            }
            for job in jobs
        ],
        "total": len(jobs)
    }

@app.get("/api/v1/processing/stats")
async def get_processing_stats():
    """Get async processing statistics"""
    if not ASYNC_PROCESSING_AVAILABLE or not async_processor:
        return {
            "async_processing": "disabled",
            "message": "Async processing not available"
        }
    
    stats = async_processor.get_queue_stats()
    return {
        "async_processing": "enabled",
        "queue": {
            "current_size": stats["queue_size"],
            "max_size": stats["max_queue_size"],
            "available_slots": stats["max_queue_size"] - stats["queue_size"]
        },
        "workers": {
            "active": stats["active_workers"],
            "max_workers": stats["max_workers"],
            "running": stats["workers_running"]
        },
        "jobs": {
            "total": stats["total_jobs"],
            "by_status": stats["job_counts"]
        }
    }

@app.get("/api/v1/documents-debug")
async def debug_documents():
    """Debug endpoint to test if API routing works"""
    return {
        "status": "debug_working",
        "message": "API routing is functional",
        "timestamp": str(datetime.now())
    }

@app.get("/api/v1/documents")
async def list_documents():
    """List all uploaded documents"""
    
    # Get documents from persistent storage first
    if persistent_storage:
        try:
            logger.info("Attempting to get documents from persistent storage...")
            stored_docs = persistent_storage.get_all_documents(limit=100)
            logger.info(f"Retrieved {len(stored_docs)} documents from persistent storage")
            
            formatted_docs = []
            
            for i, doc in enumerate(stored_docs):
                try:
                    # Convert persistent storage format to expected API format
                    formatted_doc = {
                        "id": doc.get("id", f"doc_{i}"),
                        "filename": doc.get("filename", "unknown"),
                        "original_filename": doc.get("original_filename", doc.get("filename", "unknown")),
                        "file_type": doc.get("file_type", "unknown"),
                        "file_size": doc.get("file_size", 0),
                        "size": doc.get("file_size", 0),
                        "content_type": doc.get("content_type", "unknown"),
                        "status": doc.get("status", "processed"),
                        "upload_date": str(doc.get("upload_timestamp", "unknown")),
                        "chunks_count": doc.get("chunk_count", 0)
                    }
                    formatted_docs.append(formatted_doc)
                except Exception as doc_error:
                    logger.warning(f"Error formatting document {i}: {doc_error}")
                    continue
            
            logger.info(f"Successfully formatted {len(formatted_docs)} documents")
            return {
                "documents": formatted_docs,
                "total": len(formatted_docs)
            }
        except Exception as e:
            logger.error(f"Failed to get documents from persistent storage: {e}")
            # Fall through to memory-safe storage
    
    # Get documents from memory-safe storage as fallback
    if memory_safe_storage:
        try:
            logger.info("Attempting to get documents from memory-safe storage...")
            stored_docs = memory_safe_storage.get_all_documents(limit=100)
            formatted_docs = []
            
            for doc in stored_docs:
                # Convert memory-safe storage format to expected API format
                formatted_doc = {
                    "id": doc["id"],
                    "filename": doc["filename"],
                    "original_filename": doc.get("metadata", {}).get("original_filename", doc["filename"]),
                    "file_type": doc.get("metadata", {}).get("file_type", "unknown"),
                    "file_size": doc.get("metadata", {}).get("file_size", 0),
                    "size": doc.get("metadata", {}).get("file_size", 0),
                    "content_type": doc.get("metadata", {}).get("file_type", "unknown"),
                    "status": "processed",
                    "upload_date": doc.get("metadata", {}).get("upload_date", str(doc["upload_timestamp"])),
                    "chunks_count": doc["chunk_count"]
                }
                formatted_docs.append(formatted_doc)
            
            return {
                "documents": formatted_docs,
                "total": len(formatted_docs)
            }
        except Exception as e:
            logger.error(f"Failed to get documents from memory-safe storage: {e}")
            # Fall through to legacy
    
    # Legacy documents list - simplified to avoid hanging
    try:
        logger.info("Using legacy documents list as final fallback")
        legacy_docs = []
        for i, doc in enumerate(documents[:50]):  # Limit to first 50 to avoid hanging
            try:
                formatted_doc = {
                    "id": doc.get("id", f"legacy_{i}"),
                    "filename": doc.get("filename", "unknown"),
                    "original_filename": doc.get("original_filename", doc.get("filename", "unknown")),
                    "file_type": doc.get("file_type", "unknown"),
                    "file_size": doc.get("file_size", 0),
                    "size": doc.get("size", 0),
                    "content_type": doc.get("content_type", "unknown"),
                    "status": doc.get("status", "processed"),
                    "upload_date": doc.get("upload_date", "unknown"),
                    "chunks_count": doc.get("chunks_count", 0)
                }
                legacy_docs.append(formatted_doc)
            except Exception as legacy_error:
                logger.warning(f"Error formatting legacy document {i}: {legacy_error}")
                continue
        
        return {
            "documents": legacy_docs,
            "total": len(legacy_docs),
            "message": "Using legacy storage - some features may be limited"
        }
    except Exception as e:
        logger.error(f"Failed to process legacy documents: {e}")
        # Ultimate fallback - return empty list with error info
        return {
            "documents": [],
            "total": 0,
            "error": "All storage systems failed",
            "message": "No documents could be retrieved. Please try uploading new documents."
        }

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using vector similarity search with relevance filtering"""
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
        
        if not has_documents():
            return QueryResponse(
                query=sanitized_request.query,
                results=[],
                total_results=0
            )
        
        # Find similar chunks using vector search
        similar_chunks = find_similar_chunks(sanitized_request.query, sanitized_request.top_k)
        
        # RELEVANCE FILTERING: Only return chunks above minimum threshold
        MIN_RELEVANCE_THRESHOLD = 0.20  # 20% minimum threshold (sehr niedrig für bessere UX)
        relevant_chunks = [chunk for chunk in similar_chunks if chunk['similarity'] >= MIN_RELEVANCE_THRESHOLD]
        
        # If no relevant chunks found, return empty results with explanation
        if not relevant_chunks:
            max_score = max([chunk['similarity'] for chunk in similar_chunks]) if similar_chunks else 0.0
            return QueryResponse(
                query=sanitized_request.query,
                results=[],
                total_results=0,
                message=f"No relevant information found. Highest similarity: {max_score:.3f} (below threshold: {MIN_RELEVANCE_THRESHOLD}). The documents don't contain information about this topic."
            )
        
        # Format results - use memory-safe storage if available
        results = []
        for chunk_data in relevant_chunks:
            try:
                # Try to get chunk data from the search results first
                if 'text' in chunk_data and 'filename' in chunk_data:
                    # Data already in chunk_data from memory-safe storage
                    results.append({
                        "document_id": chunk_data.get('document_id', 0),
                        "source_document": chunk_data.get('filename', 'Unknown'),
                        "content": chunk_data['text'],
                        "score": chunk_data['similarity'],
                        "metadata": {
                            "chunk_id": chunk_data['chunk_id'],
                            "similarity_score": chunk_data['similarity']
                        }
                    })
                else:
                    # Fallback: get from legacy storage
                    if chunk_data['chunk_id'] < len(document_chunks):
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
                    else:
                        logger.error(f"Chunk ID {chunk_data['chunk_id']} not found in document_chunks")
            except Exception as e:
                logger.error(f"Error formatting result for chunk {chunk_data.get('chunk_id', 'unknown')}: {e}")
        
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
        
        if not has_documents():
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
        
        logger.info(f"LLM usage decision: use_llm={use_llm}, request.use_llm={sanitized_request.use_llm}, USE_LLM_DEFAULT={USE_LLM_DEFAULT}")
        
        # Remove timeout constraints - let LLM take time it needs
        use_llm = True
        logger.info(f"Using LLM with no timeout constraints")
        
        sources = format_sources_for_response(similar_chunks)
        
        # Try LLM generation if requested and available
        if use_llm and sources:
            # Check if LLM is available, attempt reconnection if needed
            if not ollama_client:
                logger.info("LLM client not initialized, attempting to connect...")
                llm_status = check_and_reinitialize_llm()
                if not llm_status["available"]:
                    logger.warning(f"LLM reconnection failed: {llm_status['message']}")
                    method = "vector_search_fallback"
                else:
                    logger.info(f"LLM reconnected successfully: {llm_status['message']}")
            
            if ollama_client:
                try:
                    # Check if Ollama is available (this may trigger a fresh check)
                    if not ollama_client.is_available():
                        logger.warning("Ollama not available, attempting reconnection...")
                        llm_status = check_and_reinitialize_llm()
                        if not llm_status["available"]:
                            logger.warning("Ollama reconnection failed, falling back to vector search")
                            method = "vector_search_fallback"
                        else:
                            logger.info("Ollama reconnected successfully")
                    
                    if ollama_client and ollama_client.is_available():
                        # Prepare context for LLM
                        context = prepare_context_for_llm(similar_chunks)
                        
                        # Prepare context - use full content for LLM processing
                        context_parts = []
                        for i, source in enumerate(sources[:5]):  # Use top 5 sources for better context
                            # Get original content from similar_chunks for LLM context
                            if i < len(similar_chunks):
                                if 'text' in similar_chunks[i]:
                                    content = similar_chunks[i]['text']
                                else:
                                    chunk_id = similar_chunks[i]['chunk_id']
                                    if chunk_id < len(document_chunks):
                                        content = document_chunks[chunk_id]['text']
                                    else:
                                        content = source.get('content_preview', '')
                            else:
                                content = source.get('content_preview', '')
                            
                            filename = source.get('source_document', 'Unknown')
                            if content:
                                context_parts.append(f"[Document: {filename}]\n{content[:2000]}\n")
                        context = "\n".join(context_parts)
                        
                        if not context:
                            logger.warning("No context available for LLM generation")
                            method = "vector_search_fallback"
                        else:
                            logger.info(f"Context prepared for LLM, length: {len(context)}, sources: {len(context_parts)}")
                            # Generate answer with no artificial timeout constraints
                            logger.info(f"Calling LLM with query: '{sanitized_request.query[:50]}...'")
                            logger.info(f"DEBUG: About to call ollama_client.generate_answer")
                            logger.info(f"DEBUG: ollama_client type: {type(ollama_client)}")
                            logger.info(f"DEBUG: ollama_client.is_available(): {ollama_client.is_available()}")
                            
                            # Get model configuration for proper token limits
                            if LLM_MANAGER_AVAILABLE:
                                llm_manager = get_llm_manager()
                                model_config = llm_manager.get_model_config()
                                max_tokens = model_config.get("max_tokens", 2048)
                                temperature = model_config.get("temperature", 0.7)
                            else:
                                # Fallback to higher limits if LLM manager not available
                                max_tokens = 2048
                                temperature = 0.7
                            
                            llm_answer = ollama_client.generate_answer(
                                sanitized_request.query, 
                                context, 
                                max_tokens=max_tokens,
                                temperature=temperature
                            )
                            
                            logger.info(f"DEBUG: LLM response received")
                            logger.info(f"DEBUG: llm_answer type: {type(llm_answer)}")
                            logger.info(f"DEBUG: llm_answer is None: {llm_answer is None}")
                            if llm_answer:
                                logger.info(f"DEBUG: llm_answer length: {len(llm_answer)}")
                                logger.info(f"DEBUG: llm_answer stripped: '{llm_answer.strip()[:100]}...'")
                            
                            if llm_answer and llm_answer.strip():
                                logger.info(f"SUCCESS: LLM generated answer successfully: {len(llm_answer)} chars")
                                return LLMQueryResponse(
                                    query=sanitized_request.query,
                                    answer=llm_answer.strip(),
                                    method="llm_generated",
                                    sources=sources,
                                    total_sources=len(sources),
                                    processing_time=time.time() - start_time
                                )
                            else:
                                logger.error(f"FAILURE: LLM returned empty/None response - llm_answer={llm_answer}")
                                method = "vector_search_fallback"
                    else:
                        method = "vector_search_fallback"
                
                except Exception as e:
                    logger.error(f"LLM generation error: {e}, falling back to vector search")
                    method = "vector_search_fallback"
        
        # Fallback to vector search results
        if sources:
            # Create a summary with download links instead of full content
            fallback_answer = f"Based on the uploaded documents, here are the most relevant sources:\n\n"
            
            for i, source in enumerate(sources[:3], 1):
                source_doc = source["source_document"]
                download_url = source.get("download_url", "")
                content_preview = source.get("content_preview", "")
                
                fallback_answer += f"{i}. From '{source_doc}':\n"
                fallback_answer += f"   Preview: {content_preview}\n"
                if download_url:
                    fallback_answer += f"   Download: {download_url}\n"
                fallback_answer += "\n"
            
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
            if not has_documents():
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
                
                # Get model configuration for proper token limits
                if LLM_MANAGER_AVAILABLE:
                    llm_manager = get_llm_manager()
                    model_config = llm_manager.get_model_config()
                    max_tokens = model_config.get("max_tokens", 2048)
                    temperature = model_config.get("temperature", 0.7)
                else:
                    max_tokens = 2048
                    temperature = 0.7
                
                answer_chunks = []
                for chunk in ollama_client.generate_answer_stream(sanitized_request.query, context, max_tokens=max_tokens, temperature=temperature):
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
    """Delete a document with validation - supports all storage types"""
    global documents, document_chunks, document_embeddings
    
    # Validate document ID format
    if not document_id:
        raise HTTPException(
            status_code=400,
            detail="Document ID is required"
        )
    
    # Rate limiting check
    if not rate_limit_check("delete"):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    deleted_info = {
        "deleted": False,
        "storage_type": None,
        "chunks_removed": 0,
        "filename": None
    }
    
    try:
        # Try persistent storage first
        if persistent_storage:
            try:
                logger.info(f"Attempting to delete document {document_id} from persistent storage")
                
                # Convert document_id to integer for persistent storage
                try:
                    doc_id_int = int(document_id)
                except ValueError:
                    logger.warning(f"Document ID {document_id} is not a valid integer for persistent storage")
                    # Skip persistent storage if ID is not numeric
                    pass
                else:
                    # Get document info before deletion
                    doc_info = persistent_storage.get_document_by_id(doc_id_int)
                    if doc_info:
                        deleted_info["filename"] = doc_info.get("filename", "unknown")
                        
                        # Delete from persistent storage
                        success = persistent_storage.delete_document(doc_id_int)
                        if success:
                            deleted_info["deleted"] = True
                            deleted_info["storage_type"] = "persistent"
                            deleted_info["chunks_removed"] = doc_info.get("chunk_count", 0)
                            
                            # Clear cache since we deleted data
                            fast_cache.clear_all()
                            
                            logger.info(f"Document {document_id} deleted from persistent storage")
                            return {
                                "message": f"Document '{deleted_info['filename']}' deleted successfully",
                                "storage_type": "persistent",
                                "chunks_removed": deleted_info["chunks_removed"]
                            }
            except Exception as e:
                logger.error(f"Error deleting from persistent storage: {e}")
                # Continue to try other storage types
        
        # Try memory-safe storage
        if memory_safe_storage and not deleted_info["deleted"]:
            try:
                logger.info(f"Attempting to delete document {document_id} from memory-safe storage")
                
                # Get document info
                doc_info = memory_safe_storage.get_document(document_id)
                if doc_info:
                    deleted_info["filename"] = doc_info.get("filename", "unknown")
                    
                    # Delete from memory-safe storage
                    success = memory_safe_storage.delete_document(document_id)
                    if success:
                        deleted_info["deleted"] = True
                        deleted_info["storage_type"] = "memory_safe"
                        deleted_info["chunks_removed"] = doc_info.get("chunk_count", 0)
                        
                        # Clear cache since we deleted data
                        fast_cache.clear_all()
                        
                        logger.info(f"Document {document_id} deleted from memory-safe storage")
                        return {
                            "message": f"Document '{deleted_info['filename']}' deleted successfully",
                            "storage_type": "memory_safe",
                            "chunks_removed": deleted_info["chunks_removed"]
                        }
            except Exception as e:
                logger.error(f"Error deleting from memory-safe storage: {e}")
        
        # Try legacy storage as last resort
        if not deleted_info["deleted"]:
            # Handle numeric document IDs for legacy storage
            try:
                document_id_int = int(document_id)
                document = next((doc for doc in documents if doc["id"] == document_id_int), None)
                
                if document:
                    deleted_info["filename"] = document.get("filename", "unknown")
                    
                    # Remove file if it exists
                    if "file_path" in document and os.path.exists(document["file_path"]):
                        try:
                            os.remove(document["file_path"])
                        except Exception as e:
                            logger.warning(f"Failed to remove file: {e}")
                    
                    # Remove from documents list
                    documents = [doc for doc in documents if doc["id"] != document_id_int]
                    
                    # Remove associated chunks and embeddings
                    original_chunk_count = len(document_chunks)
                    document_chunks = [chunk for chunk in document_chunks if chunk.get("document_id") != document_id_int]
                    chunks_removed = original_chunk_count - len(document_chunks)
                    
                    # Remove corresponding embeddings
                    if chunks_removed > 0 and len(document_embeddings) >= chunks_removed:
                        document_embeddings = document_embeddings[:-chunks_removed]
                    
                    deleted_info["deleted"] = True
                    deleted_info["storage_type"] = "legacy"
                    deleted_info["chunks_removed"] = chunks_removed
                    
                    # Clear cache since we deleted data
                    fast_cache.clear_all()
                    
                    logger.info(f"Document deleted from legacy storage: {deleted_info['filename']} (ID: {document_id_int})")
                    return {
                        "message": f"Document '{deleted_info['filename']}' deleted successfully",
                        "storage_type": "legacy",
                        "chunks_removed": chunks_removed
                    }
            except ValueError:
                # Non-numeric ID, not found in legacy storage
                pass
        
        # Document not found in any storage
        if not deleted_info["deleted"]:
            logger.warning(f"Document {document_id} not found in any storage")
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID '{document_id}' not found in any storage system"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )

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
        if not has_documents():
            return ChatResponse(
                response="Bitte laden Sie zuerst Dokumente hoch, damit ich Ihnen helfen kann!",
                query=sanitized_query,
                context=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
        
        # Find similar chunks
        similar_chunks = find_similar_chunks(sanitized_query, request.context_limit or 5)
        
        # RELEVANCE FILTERING: Only use chunks above minimum threshold
        MIN_RELEVANCE_THRESHOLD = 0.4  # 40% minimum threshold
        relevant_chunks = [chunk for chunk in similar_chunks if chunk['similarity'] >= MIN_RELEVANCE_THRESHOLD]
        
        if not relevant_chunks:
            max_score = max([chunk['similarity'] for chunk in similar_chunks]) if similar_chunks else 0.0
            return ChatResponse(
                response=f"Ich konnte keine relevanten Informationen in den hochgeladenen Dokumenten finden, um Ihre Frage zu beantworten. Die höchste Ähnlichkeit war {max_score:.3f}, was unter dem Grenzwert von {MIN_RELEVANCE_THRESHOLD} liegt. Die Dokumente enthalten keine Informationen zu diesem Thema.",
                context=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
        
        # Calculate average confidence using only relevant chunks
        avg_confidence = sum(chunk['similarity'] for chunk in relevant_chunks) / len(relevant_chunks)
        
        # Prepare context for response using only relevant chunks
        context_sources = format_sources_for_response(relevant_chunks)
        
        # Try to use LLM if available
        if ollama_client and ollama_client.is_available():
            try:
                # Prepare context for LLM using only relevant chunks
                context_text = prepare_context_for_llm(relevant_chunks)
                
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

@app.post("/api/v1/query/fast", response_model=ChatResponse)
async def query_documents_fast(request: ChatRequest):
    """Fast query with aggressive timeouts and intelligent fallback"""
    start_time = time.time()
    
    try:
        # Rate limiting check
        if not rate_limit_check("query"):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Validate and sanitize request
        sanitized_query = sanitize_query_string(request.query)
        if not sanitized_query:
            raise HTTPException(
                status_code=400,
                detail="Invalid query after sanitization"
            )
        
        # Check if we have documents
        if not has_documents():
            return ChatResponse(
                response="Please upload documents first so I can help you!",
                query=sanitized_query,
                context=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
        
        # Find similar chunks
        similar_chunks = find_similar_chunks(sanitized_query, 5)
        
        if not similar_chunks:
            return ChatResponse(
                response="I couldn't find relevant information in the uploaded documents for your question.",
                query=sanitized_query,
                context=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
        
        # Format context sources
        context_sources = format_sources_for_response(similar_chunks)
        
        # Try LLM with aggressive timeout (10 seconds max)
        if ollama_client and ollama_client.is_available():
            try:
                # Use a shorter timeout for fast mode
                import asyncio
                from concurrent.futures import ThreadPoolExecutor
                
                # Use context manager to ensure proper resource cleanup
                with ThreadPoolExecutor(max_workers=1) as executor:
                    context_text = prepare_context_for_llm(similar_chunks)
                    
                    # Run LLM generation in thread with timeout
                    loop = asyncio.get_event_loop()
                    future = loop.run_in_executor(
                        executor,
                        lambda: ollama_client.generate_answer(
                            sanitized_query,
                            context_text,
                            max_tokens=get_fast_max_tokens(),  # Use config-based tokens with fast fallback
                            temperature=0.3   # Lower temp for faster generation
                        )
                    )
                    
                    try:
                        llm_response = await asyncio.wait_for(future, timeout=10.0)  # 10 second timeout
                        
                        if llm_response and llm_response.strip():
                            avg_confidence = sum(chunk['similarity'] for chunk in similar_chunks) / len(similar_chunks)
                            return ChatResponse(
                                response=llm_response,
                                query=sanitized_query,
                                context=context_sources,
                                confidence=avg_confidence,
                                processing_time=time.time() - start_time,
                                method="llm_fast"
                            )
                            
                    except asyncio.TimeoutError:
                        logger.warning("LLM generation timed out in fast mode, falling back to vector search")
                    
            except Exception as e:
                logger.error(f"LLM generation error in fast mode: {e}")
        
        # Fallback to enhanced vector search response
        if context_sources:
            top_source = context_sources[0]
            content = top_source['content']
            
            # Extract key sentences
            sentences = content.split('.')
            key_sentences = []
            query_words = set(sanitized_query.lower().split())
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:
                    sentence_words = set(sentence.lower().split())
                    if query_words.intersection(sentence_words):
                        key_sentences.append(sentence)
                        if len(key_sentences) >= 2:
                            break
            
            if key_sentences:
                response_text = '. '.join(key_sentences) + '.'
            else:
                response_text = content[:200] + '...' if len(content) > 200 else content
            
            response_text += f"\n\nSource: {top_source['source_document']}"
            
            if len(context_sources) > 1:
                other_sources = [s['source_document'] for s in context_sources[1:3]]
                response_text += f"\nAdditional sources: {', '.join(other_sources)}"
            
            avg_confidence = sum(chunk['similarity'] for chunk in similar_chunks) / len(similar_chunks)
            return ChatResponse(
                response=response_text,
                query=sanitized_query,
                context=context_sources,
                confidence=avg_confidence,
                processing_time=time.time() - start_time,
                method="vector_fast"
            )
        
        # No results found
        return ChatResponse(
            response="I couldn't find specific information about that topic in the uploaded documents.",
            query=sanitized_query,
            context=[],
            confidence=0.0,
            processing_time=time.time() - start_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in fast query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/api/v1/query/optimized", response_model=ChatResponse)
async def query_documents_optimized(request: ChatRequest):
    """Optimized query with caching, faster timeouts and better fallback formatting"""
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
        
        # Check response cache first
        query_hash = hashlib.md5(sanitized_query.encode()).hexdigest()
        cached_response = fast_cache.get_response_cache(query_hash)
        if cached_response:
            logger.info(f"Serving cached response for query: {sanitized_query[:50]}...")
            # Mark as cached and return immediately
            cached_response = cached_response.copy()  # Don't modify original
            cached_response.processing_time = time.time() - start_time
            return ChatResponse(
                response=cached_response.response,
                query=cached_response.query,
                context=cached_response.context,
                confidence=cached_response.confidence,
                processing_time=time.time() - start_time,
                method="cached"
            )
        
        # Check if we have any documents
        if not has_documents():
            return ChatResponse(
                response="Bitte laden Sie zuerst Dokumente hoch, damit ich Ihnen helfen kann!",
                query=sanitized_query,
                context=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
        
        # Find similar chunks
        similar_chunks = find_similar_chunks(sanitized_query, request.context_limit or 3)
        
        # RELEVANCE FILTERING: Only use chunks above 20% threshold  
        MIN_RELEVANCE_THRESHOLD = 0.20
        relevant_chunks = [chunk for chunk in similar_chunks if chunk['similarity'] >= MIN_RELEVANCE_THRESHOLD]
        
        if not relevant_chunks:
            max_score = max([chunk['similarity'] for chunk in similar_chunks]) if similar_chunks else 0.0
            return ChatResponse(
                response=f"Ich konnte keine relevanten Informationen in den Dokumenten finden. Die höchste Ähnlichkeit war {max_score:.3f}, was unter dem Grenzwert von {MIN_RELEVANCE_THRESHOLD} liegt. Die Dokumente enthalten keine Informationen zu diesem Thema.",
                query=sanitized_query,
                context=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
        
        # Calculate average confidence using only relevant chunks
        avg_confidence = sum(chunk['similarity'] for chunk in relevant_chunks) / len(relevant_chunks)
        
        # Prepare context for response using only relevant chunks
        context_sources = format_sources_for_response(relevant_chunks)
        
        # Try LLM with reasonable timeout (45 seconds) - skip if Ollama unavailable
        if ollama_client and ollama_client.is_available():
            try:
                context_text = prepare_context_for_llm(relevant_chunks[:2])  # Use fewer chunks for speed
                
                if context_text:
                    # Create a practical, user-friendly prompt with filtering
                    prompt = f"""Du bist ein KI-Assistent, spezialisiert auf die Beantwortung von Fragen primär basierend auf einem bereitgestellten Dokumentenkontext, aber mit der Fähigkeit, auf Allgemeinwissen zurückzugreifen, falls nötig.

**Deine Anweisungen:**

1. **Strikte Kontextpräferenz:** Lies die unten stehende Benutzeranfrage und versuche **zuerst und vorrangig**, sie **nur** mit Informationen zu beantworten, die direkt aus den bereitgestellten Kontextdokumenten stammen.

2. **Kein externes Wissen (im Kontextfall):** Solange du die Antwort im Kontext findest, beziehe kein externes Wissen, keine Annahmen oder Informationen aus früheren Interaktionen mit ein. Erfinde keine Informationen, die nicht im Kontext stehen.

3. **Prüfung auf Kontext-Antwort:** Prüfe sorgfältig, ob der Kontext die Informationen zur vollständigen Beantwortung der Benutzeranfrage enthält.

4. **Szenario 1: Antwort im Kontext gefunden:**
   * Wenn die Antwort im Kontext gefunden wird, formuliere eine klare, präzise und umfassende Antwort basierend **nur** auf dem Kontext.
   * Konzentriere dich auf praktische, für Bürger relevante Informationen: Was? Wo? Wann? Wie?
   * Verwende einfache, klare Sprache und strukturiere gut mit Aufzählungen.
   * Fahre dann **zwingend** mit Schritt 6 (Detaillierte Quellenangabe) fort.

5. **Szenario 2: Antwort NICHT im Kontext gefunden:**
   * Wenn die Antwort **nicht** oder nur unvollständig in den Kontextdokumenten gefunden werden kann:
   * **Gib zuerst explizit und deutlich an:** "Ich konnte die Antwort auf diese Frage nicht in den bereitgestellten Dokumenten finden. Die folgende Antwort basiert auf meinem Allgemeinwissen."
   * Beantworte die Benutzeranfrage **danach** bestmöglich unter Verwendung deines allgemeinen Wissens.
   * **Wichtig: In diesem Fall entfällt Schritt 6 (Quellenangabe).**

6. **Detaillierte Quellenangabe (Nur für Antworten aus Kontext!):** Füge **nur dann**, wenn deine Antwort vollständig auf dem bereitgestellten Kontext basiert, **nach** deiner formulierten Antwort einen Abschnitt mit dem Titel "**Quellen:**" hinzu.

**Kontextdokumente:**
{context_text[:3000]}

**Benutzeranfrage:**
{sanitized_query}

**Antwort:**"""
                    
                    # Quick LLM generation with timeout
                    import asyncio
                    import concurrent.futures
                    
                    def quick_llm_call():
                        return ollama_client.generate_answer(sanitized_query, context_text, max_tokens=get_fast_max_tokens())  # Use config-based tokens
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(quick_llm_call)
                        try:
                            llm_response = future.result(timeout=300)  # 5 minute timeout for laptop performance
                            if llm_response:
                                response_obj = ChatResponse(
                                    response=llm_response + f"\n\nQuelle: {context_sources[0]['source_document']}",
                                    query=sanitized_query,
                                    context=context_sources,
                                    confidence=avg_confidence,
                                    processing_time=time.time() - start_time,
                                    method="llm_optimized"
                                )
                                # Cache the response
                                fast_cache.set_response_cache(query_hash, response_obj)
                                return response_obj
                        except concurrent.futures.TimeoutError:
                            logger.warning("LLM timeout after 5 minutes, falling back to vector search")
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
            
            response_obj = ChatResponse(
                response=response_text,
                query=sanitized_query,
                context=context_sources,
                confidence=avg_confidence,
                processing_time=time.time() - start_time,
                method="vector_optimized"
            )
            # Cache the response
            fast_cache.set_response_cache(query_hash, response_obj)
            return response_obj
        
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

@app.get("/api/v1/vector-stats")
async def get_vector_stats():
    """Get vector store statistics and performance information"""
    if FAISS_AVAILABLE and vector_store:
        stats = vector_store.get_stats()
        stats["type"] = "FAISS (Optimized)"
        stats["performance"] = "10-100x faster than cosine similarity"
        stats["description"] = "Using FAISS for ultra-fast vector similarity search"
        stats["index_type"] = stats.get("index_type", "Auto-selected")
        return stats
    else:
        return {
            "type": "Cosine Similarity (Basic)",
            "status": "fallback",
            "total_vectors": len(document_embeddings),
            "performance": "Baseline performance",
            "description": "Consider installing FAISS for massive performance improvements",
            "installation": "pip install faiss-cpu",
            "expected_speedup": "10-100x faster with FAISS"
        }

# ============================================================================
# Document Management CRUD Endpoints
# ============================================================================

@app.get("/api/v1/documents/advanced", response_model=Dict)
async def list_documents_advanced(
    status: Optional[str] = None,
    tags: Optional[str] = None,  # Comma-separated
    uploader: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """Advanced document listing with filtering and search"""
    if not DOCUMENT_MANAGER_AVAILABLE or not doc_manager:
        # Fallback to simple listing
        return {
            "documents": [
                {
                    "id": doc["id"],
                    "filename": doc["filename"],
                    "file_type": doc["file_type"],
                    "file_size": doc["file_size"],
                    "status": doc["status"],
                    "upload_date": doc["upload_date"],
                    "chunks_count": doc["chunks_count"]
                }
                for doc in documents[offset:offset + limit]
            ],
            "total": len(documents),
            "message": "Basic listing (document manager not available)"
        }
    
    try:
        # Validate parameters
        if VALIDATION_AVAILABLE and input_validator:
            limit_validation = input_validator.numeric_validator.validate_integer(
                limit, min_value=1, max_value=1000
            )
            offset_validation = input_validator.numeric_validator.validate_integer(
                offset, min_value=0
            )
            
            if not limit_validation.is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid limit: {'; '.join(limit_validation.errors)}")
            if not offset_validation.is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid offset: {'; '.join(offset_validation.errors)}")
            
            limit = limit_validation.sanitized_value
            offset = offset_validation.sanitized_value
        
        # Parse filters
        status_filter = DocumentStatus(status) if status else None
        tags_filter = [tag.strip() for tag in tags.split(',')] if tags else None
        
        # Search if query provided
        if search:
            docs = doc_manager.search_documents(search, search_content=True)
            total = len(docs)
            docs = docs[offset:offset + limit]
        else:
            docs, total = doc_manager.list_documents(
                status=status_filter,
                tags=tags_filter,
                uploader=uploader,
                limit=limit,
                offset=offset
            )
        
        return {
            "documents": [doc.to_dict() for doc in docs],
            "total": total,
            "offset": offset,
            "limit": limit,
            "filters": {
                "status": status,
                "tags": tags_filter,
                "uploader": uploader,
                "search": search
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in advanced document listing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/documents/{document_id}/details")
async def get_document_details(document_id: int):
    """Get detailed information about a specific document"""
    if not DOCUMENT_MANAGER_AVAILABLE or not doc_manager:
        # Fallback to simple document lookup
        doc = next((d for d in documents if d["id"] == document_id), None)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc
    
    try:
        # Validate document_id
        if VALIDATION_AVAILABLE and input_validator:
            id_validation = input_validator.numeric_validator.validate_integer(
                document_id, min_value=1
            )
            if not id_validation.is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid document ID: {'; '.join(id_validation.errors)}")
            document_id = id_validation.sanitized_value
        
        document = doc_manager.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get chunks info
        chunks = doc_manager.get_document_chunks(document_id)
        
        # Prepare detailed response
        details = document.to_dict()
        details["chunks"] = [
            {
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "character_range": f"{chunk.start_char}-{chunk.end_char}",
                "length": len(chunk.text)
            }
            for chunk in chunks
        ]
        details["total_chunks"] = len(chunks)
        details["total_characters"] = sum(len(chunk.text) for chunk in chunks)
        
        return details
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document details: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.put("/api/v1/documents/{document_id}")
async def update_document(document_id: int, updates: Dict[str, Any]):
    """Update document metadata"""
    if not DOCUMENT_MANAGER_AVAILABLE or not doc_manager:
        raise HTTPException(status_code=503, detail="Document management not available")
    
    try:
        # Validate document_id
        if VALIDATION_AVAILABLE and input_validator:
            id_validation = input_validator.numeric_validator.validate_integer(
                document_id, min_value=1
            )
            if not id_validation.is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid document ID: {'; '.join(id_validation.errors)}")
            document_id = id_validation.sanitized_value
        
        # Validate update fields
        allowed_fields = {"description", "tags", "status"}
        invalid_fields = set(updates.keys()) - allowed_fields
        if invalid_fields:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid fields: {', '.join(invalid_fields)}. Allowed: {', '.join(allowed_fields)}"
            )
        
        # Validate individual fields
        if "tags" in updates and not isinstance(updates["tags"], list):
            raise HTTPException(status_code=400, detail="Tags must be a list of strings")
        
        if "status" in updates:
            try:
                updates["status"] = DocumentStatus(updates["status"])
            except ValueError:
                valid_statuses = [s.value for s in DocumentStatus]
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid status. Valid values: {', '.join(valid_statuses)}"
                )
        
        # Update document
        updated_doc = doc_manager.update_document(document_id, **updates)
        if not updated_doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "message": "Document updated successfully",
            "document": updated_doc.to_dict(),
            "updated_fields": list(updates.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/documents/{document_id}/restore")
async def restore_document(document_id: int):
    """Restore a soft-deleted document"""
    if not DOCUMENT_MANAGER_AVAILABLE or not doc_manager:
        raise HTTPException(status_code=503, detail="Document management not available")
    
    try:
        # Validate document_id
        if VALIDATION_AVAILABLE and input_validator:
            id_validation = input_validator.numeric_validator.validate_integer(
                document_id, min_value=1
            )
            if not id_validation.is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid document ID: {'; '.join(id_validation.errors)}")
            document_id = id_validation.sanitized_value
        
        success = doc_manager.restore_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found or not deleted")
        
        return {"message": f"Document {document_id} restored successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restoring document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/documents/{document_id}/chunks")
async def get_document_chunks(
    document_id: int,
    limit: int = 50,
    offset: int = 0
):
    """Get chunks for a specific document"""
    if not DOCUMENT_MANAGER_AVAILABLE or not doc_manager:
        # Fallback to simple chunk lookup
        doc_chunks = [chunk for chunk in document_chunks if chunk["document_id"] == document_id]
        return {
            "chunks": doc_chunks[offset:offset + limit],
            "total": len(doc_chunks),
            "document_id": document_id
        }
    
    try:
        # Validate parameters
        if VALIDATION_AVAILABLE and input_validator:
            id_validation = input_validator.numeric_validator.validate_integer(
                document_id, min_value=1
            )
            limit_validation = input_validator.numeric_validator.validate_integer(
                limit, min_value=1, max_value=1000
            )
            offset_validation = input_validator.numeric_validator.validate_integer(
                offset, min_value=0
            )
            
            if not id_validation.is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid document ID: {'; '.join(id_validation.errors)}")
            if not limit_validation.is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid limit: {'; '.join(limit_validation.errors)}")
            if not offset_validation.is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid offset: {'; '.join(offset_validation.errors)}")
            
            document_id = id_validation.sanitized_value
            limit = limit_validation.sanitized_value
            offset = offset_validation.sanitized_value
        
        # Check document exists
        document = doc_manager.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get chunks
        all_chunks = doc_manager.get_document_chunks(document_id)
        total = len(all_chunks)
        chunks = all_chunks[offset:offset + limit]
        
        return {
            "chunks": [chunk.to_dict() for chunk in chunks],
            "total": total,
            "offset": offset,
            "limit": limit,
            "document": {
                "id": document.id,
                "filename": document.filename,
                "status": document.status.value
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document chunks: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/documents/statistics")
async def get_document_statistics():
    """Get comprehensive document management statistics"""
    if not DOCUMENT_MANAGER_AVAILABLE or not doc_manager:
        # Fallback to basic stats
        total_size = sum(doc["file_size"] for doc in documents)
        return {
            "total_documents": len(documents),
            "total_chunks": len(document_chunks),
            "total_size_bytes": total_size,
            "message": "Basic statistics (document manager not available)"
        }
    
    try:
        stats = doc_manager.get_statistics()
        
        # Add additional API-level statistics
        stats["api_features"] = {
            "faiss_search": FAISS_AVAILABLE,
            "async_processing": ASYNC_PROCESSING_AVAILABLE,
            "authentication": AUTH_AVAILABLE,
            "input_validation": VALIDATION_AVAILABLE
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting document statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/documents/cleanup")
async def cleanup_orphaned_files():
    """Clean up orphaned files in storage"""
    if not DOCUMENT_MANAGER_AVAILABLE or not doc_manager:
        raise HTTPException(status_code=503, detail="Document management not available")
    
    try:
        cleaned_count = doc_manager.cleanup_orphaned_files()
        return {
            "message": f"Cleanup completed",
            "orphaned_files_removed": cleaned_count
        }
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/documents/search")
async def search_documents_endpoint(
    q: str,
    content: bool = False,
    limit: int = 20
):
    """Search documents by metadata or content"""
    if not DOCUMENT_MANAGER_AVAILABLE or not doc_manager:
        raise HTTPException(status_code=503, detail="Document management not available")
    
    try:
        # Validate query
        if VALIDATION_AVAILABLE and input_validator:
            query_validation = input_validator.text_validator.validate_query(q)
            if not query_validation.is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid search query: {'; '.join(query_validation.errors)}")
            q = query_validation.sanitized_value
        
        # Validate limit
        if VALIDATION_AVAILABLE and input_validator:
            limit_validation = input_validator.numeric_validator.validate_integer(
                limit, min_value=1, max_value=100
            )
            if not limit_validation.is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid limit: {'; '.join(limit_validation.errors)}")
            limit = limit_validation.sanitized_value
        
        # Perform search
        results = doc_manager.search_documents(q, search_content=content)
        
        # Limit results
        limited_results = results[:limit]
        
        return {
            "query": q,
            "search_content": content,
            "results": [doc.to_dict() for doc in limited_results],
            "total_found": len(results),
            "returned": len(limited_results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/documents/{document_id}/download")
async def download_document(document_id: int, current_user: Optional[Dict] = Depends(get_current_user) if AUTH_AVAILABLE else None):
    """Download a specific document file"""
    try:
        # Validate document_id
        if VALIDATION_AVAILABLE and input_validator:
            id_validation = input_validator.numeric_validator.validate_integer(
                document_id, min_value=1
            )
            if not id_validation.is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid document ID: {'; '.join(id_validation.errors)}")
            document_id = id_validation.sanitized_value
        
        # Find the document
        doc = None
        if DOCUMENT_MANAGER_AVAILABLE and doc_manager:
            doc = doc_manager.get_document(document_id)
        else:
            # Fallback to simple document lookup
            doc = next((d for d in documents if d.get("id") == document_id), None)
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get document info
        if hasattr(doc, 'to_dict'):
            doc_dict = doc.to_dict()
        else:
            doc_dict = doc
        
        # Check if user has permission to download (if auth is available)
        if AUTH_AVAILABLE and current_user:
            # Check if user is the uploader or has admin role
            if (doc_dict.get('uploader') != current_user.get('username') and 
                current_user.get('role') != 'admin'):
                raise HTTPException(status_code=403, detail="You don't have permission to download this document")
        
        # Determine file path
        filename = doc_dict.get('filename', f"document_{document_id}")
        
        # Try to find the file in uploads directory first
        uploads_path = UPLOAD_DIR / filename
        processed_path = PROCESSED_DIR / filename
        
        file_path = None
        if uploads_path.exists():
            file_path = uploads_path
        elif processed_path.exists():
            file_path = processed_path
        else:
            # Try to find file with numeric prefix (common in the system)
            for directory in [UPLOAD_DIR, PROCESSED_DIR]:
                for file in Path(directory).glob(f"*{filename}"):
                    if file.is_file():
                        file_path = file
                        break
                if file_path:
                    break
        
        if not file_path or not file_path.exists():
            raise HTTPException(status_code=404, detail="Document file not found on disk")
        
        # Security check - ensure file is within allowed directories
        try:
            file_path.resolve().relative_to(UPLOAD_DIR.resolve())
        except ValueError:
            try:
                file_path.resolve().relative_to(PROCESSED_DIR.resolve())
            except ValueError:
                logger.warning(f"Attempt to access file outside allowed directories: {file_path}")
                raise HTTPException(status_code=403, detail="Access denied")
        
        # Get file stats
        file_size = file_path.stat().st_size
        
        # Determine content type
        content_type = "application/octet-stream"
        if file_path.suffix.lower() == '.pdf':
            content_type = "application/pdf"
        elif file_path.suffix.lower() in ['.doc', '.docx']:
            content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif file_path.suffix.lower() == '.txt':
            content_type = "text/plain"
        elif file_path.suffix.lower() in ['.csv']:
            content_type = "text/csv"
        elif file_path.suffix.lower() in ['.json']:
            content_type = "application/json"
        
        # Create response headers
        headers = {
            "Content-Disposition": f"attachment; filename=\"{filename}\"",
            "Content-Length": str(file_size),
            "Content-Type": content_type
        }
        
        # Return file response
        return FileResponse(
            path=str(file_path),
            filename=filename,
            headers=headers,
            media_type=content_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ============================================================================
# Smart Answer Endpoints
# ============================================================================

class SmartQueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_llm_fallback: bool = True
    strict_mode: bool = False  # If True, only return document-based answers

class SmartQueryResponse(BaseModel):
    query: str
    answer: str
    answer_type: str
    confidence: str
    confidence_score: float
    sources: List[Dict[str, Any]]
    reasoning: str
    chunk_count: int
    is_document_based: bool
    is_llm_generated: bool
    processing_time: float

@app.post("/api/v1/query/smart", response_model=SmartQueryResponse)
async def smart_query_documents(request: SmartQueryRequest):
    """
    Smart query with relevance scoring and document-first approach
    Returns only relevant answers or clearly marked LLM responses
    """
    if not SMART_ANSWER_AVAILABLE or not smart_answer_engine:
        # Fall back to regular query
        raise HTTPException(
            status_code=503, 
            detail="Smart answer engine not available. Use /api/v1/query instead."
        )
    
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
            logger.warning(f"Smart query validation failed: {error_message}")
            raise HTTPException(status_code=400, detail=error_message)
        
        # Use sanitized request
        sanitized_request = validation_result["sanitized_request"]
        
        if not embedding_model:
            raise HTTPException(
                status_code=500,
                detail="Embedding model not available. Please install sentence-transformers."
            )
        
        if not has_documents():
            # No documents uploaded
            if request.strict_mode:
                return SmartQueryResponse(
                    query=sanitized_request.query,
                    answer="❌ **No Documents Available**\n\nI cannot answer your question because no documents have been uploaded. Please upload documents first.",
                    answer_type="no_answer",
                    confidence="insufficient",
                    confidence_score=0.0,
                    sources=[],
                    reasoning="No documents available in the system",
                    chunk_count=0,
                    is_document_based=False,
                    is_llm_generated=False,
                    processing_time=time.time() - start_time
                )
            else:
                # Could use LLM for general questions, but mark clearly
                return SmartQueryResponse(
                    query=sanitized_request.query,
                    answer="❌ **No Documents Available**\n\nI cannot provide a document-based answer because no documents have been uploaded. Please upload relevant documents to get answers based on your content.",
                    answer_type="no_answer",
                    confidence="insufficient",
                    confidence_score=0.0,
                    sources=[],
                    reasoning="No documents available in the system",
                    chunk_count=0,
                    is_document_based=False,
                    is_llm_generated=False,
                    processing_time=time.time() - start_time
                )
        
        # Find similar chunks using existing search
        similar_chunks = find_similar_chunks(sanitized_request.query, sanitized_request.top_k)
        
        # Use smart answer engine to generate intelligent response
        smart_result = smart_answer_engine.generate_smart_answer(
            query=sanitized_request.query,
            similar_chunks=similar_chunks,
            llm_client=ollama_client if OLLAMA_SUPPORT else None,
            use_llm_fallback=request.use_llm_fallback and not request.strict_mode
        )
        
        processing_time = time.time() - start_time
        
        logger.info(f"Smart query: '{sanitized_request.query}' -> {smart_result.answer_type.value} "
                   f"(confidence: {smart_result.confidence_score:.3f}, time: {processing_time:.2f}s)")
        
        # Convert to response model
        response_dict = smart_result.to_dict()
        response_dict["query"] = sanitized_request.query
        response_dict["processing_time"] = processing_time
        
        return SmartQueryResponse(**response_dict)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in smart query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing smart query: {str(e)}")

@app.post("/api/v1/query/llm-only", response_model=LLMQueryResponse)
async def query_llm_only(request: QueryRequest):
    """
    Query the LLM directly without any document context or vector search.
    This mode uses the fine-tuned arlesheim-german model to respond based purely on its training data.
    """
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
            logger.warning(f"LLM-only query validation failed: {error_message}")
            raise HTTPException(status_code=400, detail=error_message)
        
        # Use sanitized request
        sanitized_request = validation_result["sanitized_request"]
        
        # Check if LLM is available
        if not ollama_client:
            raise HTTPException(
                status_code=503,
                detail="LLM not available. Please ensure Ollama is running."
            )
        
        if not ollama_client.is_available():
            # Attempt reconnection
            llm_status = check_and_reinitialize_llm()
            if not llm_status["available"]:
                raise HTTPException(
                    status_code=503,
                    detail=f"LLM not available: {llm_status['message']}"
                )
        
        # Get model configuration for proper parameters
        if LLM_MANAGER_AVAILABLE:
            llm_manager = get_llm_manager()
            model_config = llm_manager.get_model_config()
            max_tokens = model_config.get("max_tokens", 2048)
            temperature = model_config.get("temperature", 0.7)
        else:
            max_tokens = 2048
            temperature = 0.7
        
        # Generate answer using LLM without any context
        logger.info(f"Generating LLM-only response for query: '{sanitized_request.query[:50]}...'")
        
        # Use the chat completion method for direct LLM interaction
        messages = [
            {"role": "user", "content": sanitized_request.query}
        ]
        
        llm_answer = ollama_client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        if llm_answer and llm_answer.strip():
            logger.info(f"LLM-only response generated successfully: {len(llm_answer)} chars")
            return LLMQueryResponse(
                query=sanitized_request.query,
                answer=llm_answer.strip(),
                method="llm_only",
                sources=[],  # No sources for LLM-only mode
                total_sources=0,
                processing_time=time.time() - start_time
            )
        else:
            logger.error("LLM returned empty response")
            raise HTTPException(
                status_code=503,
                detail="LLM generated an empty response. Please try again."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in LLM-only query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing LLM-only query: {str(e)}")

@app.get("/api/v1/chunking/analysis")
async def analyze_chunking_quality():
    """Analyze current chunking strategy and suggest improvements"""
    if not SMART_ANSWER_AVAILABLE or not smart_answer_engine:
        raise HTTPException(status_code=503, detail="Smart answer engine not available")
    
    try:
        # Analyze current documents
        analysis = smart_answer_engine.suggest_better_chunking(documents)
        
        # Add additional metrics
        if document_chunks:
            chunk_lengths = [len(chunk.get('text', '')) for chunk in document_chunks]
            analysis.update({
                "current_chunks": {
                    "total_chunks": len(document_chunks),
                    "avg_length": sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0,
                    "min_length": min(chunk_lengths) if chunk_lengths else 0,
                    "max_length": max(chunk_lengths) if chunk_lengths else 0,
                    "length_distribution": {
                        "very_short": sum(1 for l in chunk_lengths if l < 200),
                        "short": sum(1 for l in chunk_lengths if 200 <= l < 500),
                        "medium": sum(1 for l in chunk_lengths if 500 <= l < 1000),
                        "long": sum(1 for l in chunk_lengths if 1000 <= l < 1500),
                        "very_long": sum(1 for l in chunk_lengths if l >= 1500)
                    }
                }
            })
        
        # Add chunking recommendations
        analysis["recommendations"] = [
            "Use the improved chunking system for better context preservation",
            "Target chunk size: 800-1200 characters for optimal relevance",
            "Include overlap between chunks for better context continuity",
            "Preserve sentence and paragraph boundaries when possible"
        ]
        
        if IMPROVED_CHUNKING_AVAILABLE:
            analysis["improved_chunking_available"] = True
            analysis["usage_note"] = "Improved chunking is automatically used for new document uploads"
        else:
            analysis["improved_chunking_available"] = False
            analysis["setup_note"] = "Install improved chunking service for better results"
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing chunking: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/status")
async def get_api_status():
    """Status endpoint for widget health checks"""
    return {
        "status": "healthy",
        "version": "1.3.0",
        "uptime": time.time(),
        "features": {
            "vector_search": embedding_model is not None,
            "faiss_search": FAISS_AVAILABLE and vector_store is not None,
            "async_processing": ASYNC_PROCESSING_AVAILABLE and async_processor is not None,
            "authentication": AUTH_AVAILABLE and auth_manager is not None,
            "llm_generation": ollama_client is not None and ollama_client.is_available() if ollama_client else False,
            "document_processing": True,
            "document_management": DOCUMENT_MANAGER_AVAILABLE and doc_manager is not None,
            "smart_answers": SMART_ANSWER_AVAILABLE and smart_answer_engine is not None,
            "improved_chunking": IMPROVED_CHUNKING_AVAILABLE and improved_chunker is not None
        },
        "statistics": {
            "documents_uploaded": len(documents),
            "total_chunks": len(document_chunks)
        }
    }

@app.get("/api/v1/status")
async def get_api_v1_status():
    """Enhanced status endpoint that checks memory-safe storage first"""
    global memory_safe_storage
    
    # DEBUG: Log memory-safe storage status and ID
    logger.info(f"DEBUG: /api/v1/status called")
    logger.info(f"DEBUG: memory_safe_storage type: {type(memory_safe_storage)}, None={memory_safe_storage is None}")
    if memory_safe_storage:
        logger.info(f"DEBUG: memory_safe_storage ID: {id(memory_safe_storage)}")
    
    # Get statistics from memory-safe storage if available
    statistics = {
        "documents_uploaded": 0,
        "total_chunks": 0,
        "embeddings_created": 0
    }
    
    if memory_safe_storage:
        try:
            storage_stats = memory_safe_storage.get_stats()
            logger.info(f"DEBUG: Memory-safe storage stats from global instance: {storage_stats}")
            statistics = {
                "documents_uploaded": storage_stats.get('documents', 0),
                "total_chunks": storage_stats.get('chunks', 0),
                "embeddings_created": storage_stats.get('embeddings', 0)
            }
            
            # Additional debug: check if we can get documents list
            try:
                docs = memory_safe_storage.get_all_documents(limit=5)
                logger.info(f"DEBUG: Found {len(docs)} documents in memory-safe storage")
                for doc in docs[:3]:
                    logger.info(f"DEBUG: Document: ID={doc['id']}, filename={doc['filename']}")
            except Exception as e:
                logger.error(f"DEBUG: Failed to get documents list: {e}")
                
        except Exception as e:
            logger.error(f"Failed to get memory-safe storage stats: {e}")
            # Fall back to legacy statistics
            statistics = {
                "documents_uploaded": len(documents),
                "total_chunks": len(document_chunks),
                "embeddings_created": len(document_embeddings)
            }
    else:
        logger.warning("Memory-safe storage not available, using legacy statistics")
        statistics = {
            "documents_uploaded": len(documents),
            "total_chunks": len(document_chunks),
            "embeddings_created": len(document_embeddings)
        }
    
    # Check Ollama status
    ollama_status = None
    if ollama_client:
        try:
            if ollama_client.is_available():
                ollama_status = {
                    "available": True,
                    "model": ollama_client.model,
                    "models": ollama_client.get_available_models() if hasattr(ollama_client, 'get_available_models') else []
                }
            else:
                ollama_status = {
                    "available": False,
                    "error": "Service not reachable"
                }
        except Exception as e:
            ollama_status = {
                "available": False,
                "error": str(e)
            }
    
    return {
        "status": "healthy",
        "version": "1.3.0",
        "uptime": time.time(),
        "features": {
            "vector_search": embedding_model is not None,
            "faiss_search": FAISS_AVAILABLE and vector_store is not None,
            "async_processing": ASYNC_PROCESSING_AVAILABLE and async_processor is not None,
            "authentication": AUTH_AVAILABLE and auth_manager is not None,
            "llm_generation": ollama_client is not None and ollama_client.is_available() if ollama_client else False,
            "document_processing": True,
            "document_management": DOCUMENT_MANAGER_AVAILABLE and doc_manager is not None,
            "smart_answers": SMART_ANSWER_AVAILABLE and smart_answer_engine is not None,
            "improved_chunking": IMPROVED_CHUNKING_AVAILABLE and improved_chunker is not None
        },
        "statistics": statistics,
        "ollama": ollama_status,
        "storage_mode": "memory_safe" if memory_safe_storage else "legacy"
    }

# ========================
# LLM Management Endpoints  
# ========================

@app.get("/api/v1/llm/models")
async def list_llm_models():
    """List all available LLM models and configurations"""
    if not LLM_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="LLM Manager not available")
    
    llm_manager = get_llm_manager()
    models = llm_manager.list_available_models()
    current_model_name = llm_manager.get_current_model()
    
    # Find the key for the current model
    current_model_key = None
    for key, model_info in models.items():
        if model_info.get('name') == current_model_name:
            current_model_key = key
            break
    
    return {
        "models": models,
        "current_model": current_model_key or "mistral",
        "available": True
    }

@app.post("/api/v1/llm/switch/{model_key}")
async def switch_llm_model(model_key: str):
    """Switch to a different LLM model"""
    if not LLM_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="LLM Manager not available")
    
    if not OLLAMA_SUPPORT:
        raise HTTPException(status_code=503, detail="Ollama not available")
    
    llm_manager = get_llm_manager()
    
    # Check if model exists in config
    models = llm_manager.list_available_models()
    if model_key not in models:
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_key}' not found. Available: {list(models.keys())}"
        )
    
    # Switch model in LLM manager
    if llm_manager.set_model(model_key):
        # Update ollama client
        global ollama_client
        if ollama_client:
            success = ollama_client.switch_model(model_key)
            if success:
                new_model = llm_manager.get_current_model()
                
                # Test if model is available in Ollama
                if ollama_client.is_available():
                    return {
                        "success": True,
                        "message": f"Switched to model: {new_model}",
                        "model": new_model,
                        "available": True
                    }
                else:
                    return {
                        "success": True,
                        "message": f"Switched to model: {new_model}", 
                        "model": new_model,
                        "available": False,
                        "warning": "Model not available in Ollama. Please pull it first."
                    }
            else:
                raise HTTPException(status_code=500, detail="Failed to switch model in Ollama client")
        else:
            raise HTTPException(status_code=500, detail="Ollama client not initialized")
    else:
        raise HTTPException(status_code=500, detail="Failed to switch model in LLM manager")

@app.get("/api/v1/llm/status")
async def get_llm_status():
    """Get current LLM status and configuration"""
    if not LLM_MANAGER_AVAILABLE:
        return {
            "llm_manager": False,
            "ollama": OLLAMA_SUPPORT,
            "current_model": None
        }
    
    llm_manager = get_llm_manager()
    current_model = llm_manager.get_current_model()
    model_config = llm_manager.get_model_config()
    
    status = {
        "llm_manager": True,
        "ollama": OLLAMA_SUPPORT,
        "current_model": current_model,
        "model_config": model_config
    }
    
    # Check Ollama availability
    if OLLAMA_SUPPORT and ollama_client:
        ollama_available = ollama_client.is_available()
        status["ollama_available"] = ollama_available
        
        if ollama_available:
            status["ollama_models"] = ollama_client.list_models()
    
    return status

@app.post("/api/v1/llm/pull/{model_name}")
async def pull_ollama_model(model_name: str):
    """Pull/download a model in Ollama"""
    if not OLLAMA_SUPPORT:
        raise HTTPException(status_code=503, detail="Ollama not available")
    
    if not ollama_client:
        raise HTTPException(status_code=500, detail="Ollama client not initialized")
    
    try:
        success = ollama_client.pull_model(model_name)
        if success:
            return {
                "success": True,
                "message": f"Successfully pulled model: {model_name}",
                "model": model_name
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to pull model: {model_name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error pulling model: {str(e)}")

@app.post("/api/v1/llm/test")
async def test_llm_generation():
    """Test LLM generation directly"""
    try:
        test_query = "What is the waste collection schedule?"
        test_context = "The waste collection happens on Mondays and Thursdays."
        
        result = generate_llm_answer(test_query, test_context)
        
        return {
            "success": result is not None,
            "result": result,
            "result_type": type(result).__name__,
            "result_length": len(result) if result else 0,
            "ollama_client_exists": ollama_client is not None,
            "ollama_available": ollama_client.is_available() if ollama_client else False
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "ollama_client_exists": ollama_client is not None,
            "ollama_available": ollama_client.is_available() if ollama_client else False
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