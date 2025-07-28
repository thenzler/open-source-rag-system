"""
Vector service client for Document Processor.
Communicates with the Vector Engine service.
"""
import logging
from typing import List, Dict, Any, Optional
import aiohttp
import asyncio

from app.core.config import get_settings
from app.core.exceptions import VectorServiceError, EmbeddingError

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorService:
    """Client for Vector Engine service."""
    
    def __init__(self):
        self.base_url = "http://vector-engine:8002"
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = aiohttp.ClientTimeout(total=30)
    
    async def initialize(self):
        """Initialize the vector service client."""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
    
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def add_documents(
        self, 
        texts: List[str], 
        metadatas: List[Dict[str, Any]], 
        document_id: str
    ) -> List[str]:
        """
        Add documents to the vector database.
        
        Args:
            texts: List of text chunks
            metadatas: List of metadata for each chunk
            document_id: ID of the parent document
            
        Returns:
            List of vector IDs
        """
        try:
            await self.initialize()
            
            # First, generate embeddings
            embeddings_response = await self._generate_embeddings(texts)
            embeddings = embeddings_response["embeddings"]
            
            # Prepare vectors for storage
            vectors = []
            vector_ids = []
            
            for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
                vector_id = f"{document_id}_{i}"
                vector_ids.append(vector_id)
                
                # Add text preview to metadata
                metadata["text"] = text[:1000]  # First 1000 chars
                metadata["document_id"] = document_id
                
                vectors.append({
                    "id": vector_id,
                    "vector": embedding,
                    "metadata": metadata
                })
            
            # Store vectors
            store_request = {
                "vectors": vectors,
                "collection_name": settings.qdrant_collection_name
            }
            
            async with self.session.post(
                f"{self.base_url}/vectors/store",
                json=store_request
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise VectorServiceError(f"Failed to store vectors: {error_text}")
                
                result = await response.json()
                logger.info(f"Stored {result['stored_count']} vectors for document {document_id}")
            
            return vector_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector service: {e}")
            raise VectorServiceError(f"Vector storage failed: {e}")
    
    async def _generate_embeddings(self, texts: List[str]) -> Dict[str, Any]:
        """Generate embeddings for texts."""
        try:
            request_data = {
                "texts": texts,
                "model": settings.embedding_model
            }
            
            async with self.session.post(
                f"{self.base_url}/embeddings",
                json=request_data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise EmbeddingError(f"Failed to generate embeddings: {error_text}")
                
                return await response.json()
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise EmbeddingError(f"Embedding generation failed: {e}")
    
    async def search_similar(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filters: Additional filters
            document_ids: Limit search to specific documents
            
        Returns:
            List of search results
        """
        try:
            await self.initialize()
            
            search_request = {
                "query": query,
                "top_k": top_k,
                "min_score": score_threshold,
                "collection_name": settings.qdrant_collection_name
            }
            
            if filters or document_ids:
                search_filters = filters or {}
                if document_ids:
                    search_filters["document_id"] = document_ids
                search_request["filters"] = search_filters
            
            async with self.session.post(
                f"{self.base_url}/vectors/search",
                json=search_request
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise VectorServiceError(f"Search failed: {error_text}")
                
                result = await response.json()
                return result["results"]
                
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            raise VectorServiceError(f"Vector search failed: {e}")
    
    async def delete_document(self, document_id: str) -> int:
        """
        Delete all vectors for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Number of vectors deleted
        """
        try:
            await self.initialize()
            
            # For now, we'll need to implement a delete endpoint in the vector service
            # This is a placeholder that would need to be implemented
            logger.warning(f"Delete operation for document {document_id} not implemented")
            return 0
            
        except Exception as e:
            logger.error(f"Error deleting document vectors: {e}")
            raise VectorServiceError(f"Vector deletion failed: {e}")
    
    async def health_check(self) -> bool:
        """Check if vector service is healthy."""
        try:
            await self.initialize()
            
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("status") == "healthy"
                return False
                
        except Exception as e:
            logger.error(f"Vector service health check failed: {e}")
            return False