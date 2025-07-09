"""
Vector Engine Service - Handles embedding generation and similarity search
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
import uuid
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct, 
    Filter, 
    FieldCondition, 
    MatchValue,
    SearchParams
)
from qdrant_client.http.models import CollectionStatus
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
DEVICE = os.getenv("DEVICE", "cpu")  # or "cuda"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "768"))  # Default for all-mpnet-base-v2

# Pydantic models
class EmbeddingRequest(BaseModel):
    texts: List[str]
    model: Optional[str] = None

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimensions: int

class DocumentVector(BaseModel):
    id: str
    vector: List[float]
    metadata: Dict[str, Any]

class StoreVectorRequest(BaseModel):
    vectors: List[DocumentVector]
    collection_name: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    query_vector: Optional[List[float]] = None
    top_k: int = Field(default=5, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = None
    collection_name: Optional[str] = None

class SearchResult(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]
    content: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total_results: int
    search_time: float

class CollectionInfo(BaseModel):
    name: str
    vectors_count: int
    indexed_vectors_count: int
    points_count: int
    segments_count: int
    status: str

# FastAPI app
app = FastAPI(
    title="Vector Engine Service",
    description="Handles embedding generation and similarity search",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
embedding_model = None
qdrant_client = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global embedding_model, qdrant_client
    
    logger.info("Initializing Vector Engine Service...")
    
    # Initialize Qdrant client
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL)
        logger.info(f"Connected to Qdrant at {QDRANT_URL}")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        qdrant_client = None
    
    # Initialize embedding model
    try:
        device = DEVICE if torch.cuda.is_available() and DEVICE == "cuda" else "cpu"
        embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        logger.info(f"Loaded embedding model: {EMBEDDING_MODEL} on {device}")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        embedding_model = None
    
    # Create default collection if it doesn't exist
    await create_collection_if_not_exists(COLLECTION_NAME)
    
    logger.info("Vector Engine Service initialized successfully")

async def create_collection_if_not_exists(collection_name: str):
    """Create collection if it doesn't exist"""
    if not qdrant_client:
        logger.warning("Qdrant client not available")
        return
    
    try:
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name not in collection_names:
            logger.info(f"Creating collection: {collection_name}")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {collection_name}")
        else:
            logger.info(f"Collection {collection_name} already exists")
    except Exception as e:
        logger.error(f"Error creating collection: {e}")

class VectorEngine:
    """Vector engine utilities"""
    
    @staticmethod
    def generate_embeddings(texts: List[str], model: Optional[str] = None) -> Tuple[List[List[float]], str]:
        """Generate embeddings for texts"""
        if not embedding_model:
            raise RuntimeError("Embedding model not available")
        
        try:
            # Use global model or specified model
            model_to_use = embedding_model
            if model and model != EMBEDDING_MODEL:
                model_to_use = SentenceTransformer(model)
            
            # Generate embeddings
            embeddings = model_to_use.encode(texts, convert_to_tensor=False)
            
            # Convert to list format
            embeddings_list = [embedding.tolist() for embedding in embeddings]
            
            return embeddings_list, model or EMBEDDING_MODEL
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @staticmethod
    def store_vectors(vectors: List[DocumentVector], collection_name: str = COLLECTION_NAME) -> bool:
        """Store vectors in Qdrant"""
        if not qdrant_client:
            raise RuntimeError("Qdrant client not available")
        
        try:
            points = []
            for vector in vectors:
                points.append(PointStruct(
                    id=vector.id,
                    vector=vector.vector,
                    payload=vector.metadata
                ))
            
            # Batch upload
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"Stored {len(vectors)} vectors in collection {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error storing vectors: {e}")
            raise
    
    @staticmethod
    def search_vectors(
        query_vector: List[float],
        top_k: int = 5,
        min_score: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: str = COLLECTION_NAME
    ) -> List[SearchResult]:
        """Search for similar vectors"""
        if not qdrant_client:
            raise RuntimeError("Qdrant client not available")
        
        try:
            # Build filter if provided
            filter_obj = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    ))
                if conditions:
                    filter_obj = Filter(must=conditions)
            
            # Perform search
            search_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=min_score,
                query_filter=filter_obj
            )
            
            # Convert to our format
            results = []
            for result in search_results:
                results.append(SearchResult(
                    id=str(result.id),
                    score=result.score,
                    metadata=result.payload,
                    content=result.payload.get('content', None)
                ))
            
            return results
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            raise
    
    @staticmethod
    def get_collection_info(collection_name: str = COLLECTION_NAME) -> CollectionInfo:
        """Get collection information"""
        if not qdrant_client:
            raise RuntimeError("Qdrant client not available")
        
        try:
            info = qdrant_client.get_collection(collection_name)
            return CollectionInfo(
                name=collection_name,
                vectors_count=info.vectors_count,
                indexed_vectors_count=info.indexed_vectors_count,
                points_count=info.points_count,
                segments_count=info.segments_count,
                status=info.status.value
            )
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise

# API endpoints
@app.post("/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings for texts"""
    try:
        embeddings, model_used = VectorEngine.generate_embeddings(
            request.texts, 
            request.model
        )
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=model_used,
            dimensions=len(embeddings[0]) if embeddings else 0
        )
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vectors/store")
async def store_vectors(request: StoreVectorRequest):
    """Store vectors in the vector database"""
    try:
        collection_name = request.collection_name or COLLECTION_NAME
        
        # Ensure collection exists
        await create_collection_if_not_exists(collection_name)
        
        # Store vectors
        success = VectorEngine.store_vectors(request.vectors, collection_name)
        
        return {
            "success": success,
            "stored_count": len(request.vectors),
            "collection_name": collection_name
        }
    except Exception as e:
        logger.error(f"Error storing vectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vectors/search", response_model=SearchResponse)
async def search_vectors(request: SearchRequest):
    """Search for similar vectors"""
    try:
        start_time = datetime.now()
        
        # Generate query vector if not provided
        query_vector = request.query_vector
        if not query_vector:
            embeddings, _ = VectorEngine.generate_embeddings([request.query])
            query_vector = embeddings[0]
        
        collection_name = request.collection_name or COLLECTION_NAME
        
        # Perform search
        results = VectorEngine.search_vectors(
            query_vector=query_vector,
            top_k=request.top_k,
            min_score=request.min_score,
            filters=request.filters,
            collection_name=collection_name
        )
        
        search_time = (datetime.now() - start_time).total_seconds()
        
        return SearchResponse(
            results=results,
            query=request.query,
            total_results=len(results),
            search_time=search_time
        )
    except Exception as e:
        logger.error(f"Error searching vectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections/{collection_name}/info", response_model=CollectionInfo)
async def get_collection_info(collection_name: str):
    """Get collection information"""
    try:
        info = VectorEngine.get_collection_info(collection_name)
        return info
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def list_collections():
    """List all collections"""
    try:
        if not qdrant_client:
            raise RuntimeError("Qdrant client not available")
        
        collections = qdrant_client.get_collections()
        return {
            "collections": [
                {
                    "name": col.name,
                    "status": col.status.value if hasattr(col, 'status') else "unknown"
                }
                for col in collections.collections
            ]
        }
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections/{collection_name}/create")
async def create_collection(collection_name: str, vector_size: int = VECTOR_SIZE):
    """Create a new collection"""
    try:
        if not qdrant_client:
            raise RuntimeError("Qdrant client not available")
        
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        
        return {
            "success": True,
            "collection_name": collection_name,
            "vector_size": vector_size
        }
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a collection"""
    try:
        if not qdrant_client:
            raise RuntimeError("Qdrant client not available")
        
        qdrant_client.delete_collection(collection_name)
        
        return {
            "success": True,
            "collection_name": collection_name,
            "message": "Collection deleted successfully"
        }
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Qdrant connection
        qdrant_healthy = False
        if qdrant_client:
            try:
                collections = qdrant_client.get_collections()
                qdrant_healthy = True
            except:
                pass
        
        # Check embedding model
        model_healthy = embedding_model is not None
        
        return {
            "status": "healthy" if qdrant_healthy and model_healthy else "unhealthy",
            "qdrant_connected": qdrant_healthy,
            "embedding_model_loaded": model_healthy,
            "embedding_model": EMBEDDING_MODEL,
            "device": DEVICE,
            "vector_size": VECTOR_SIZE,
            "collection_name": COLLECTION_NAME
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True
    )
