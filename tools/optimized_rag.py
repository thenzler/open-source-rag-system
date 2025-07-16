#!/usr/bin/env python3
"""
Optimized RAG System with Fast Timeouts and Concise Responses
Implements intelligent fallback, response caching, and support for fast models
"""
import os
import time
import json
import hashlib
import logging
import asyncio
import concurrent.futures
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class ModelConfig(Enum):
    """Fast model configurations with expected performance characteristics"""
    PHI4 = ("phi-4:latest", 5, 0.5)  # model_name, timeout_seconds, temperature
    SMOLLM2 = ("smollm2:latest", 5, 0.5)
    DEEPSEEK_R1 = ("deepseek-r1:latest", 8, 0.3)
    QWEN2_5 = ("qwen2.5:0.5b", 3, 0.5)
    PHI3_MINI = ("phi3:mini", 5, 0.5)
    LLAMA3_2_1B = ("llama3.2:1b", 5, 0.5)
    FALLBACK = ("llama3.2:1b", 10, 0.5)  # Fallback with slightly longer timeout

@dataclass
class ResponseConfig:
    """Configuration for response generation"""
    max_response_length: int = 300  # Maximum characters in response
    max_context_length: int = 2000  # Maximum context to send to LLM
    initial_timeout: int = 10  # Initial LLM timeout in seconds
    fast_timeout: int = 5  # Fast timeout for quick models
    max_sources: int = 3  # Maximum number of sources to include
    concise_mode: bool = True  # Whether to enforce concise responses

class ResponseCache:
    """High-performance response cache with TTL and size limits"""
    def __init__(self, max_size: int = 500, ttl_minutes: int = 30):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if valid"""
        if key not in self.cache:
            return None
        
        # Check TTL
        if datetime.now() - self.access_times[key] > self.ttl:
            del self.cache[key]
            del self.access_times[key]
            return None
        
        # Update access time
        self.access_times[key] = datetime.now()
        return self.cache[key]
    
    def set(self, key: str, value: Dict[str, Any]):
        """Cache a response"""
        # Clean up if needed
        if len(self.cache) >= self.max_size:
            # Remove 20% oldest entries
            items_to_remove = int(self.max_size * 0.2)
            sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
            for k, _ in sorted_items[:items_to_remove]:
                del self.cache[k]
                del self.access_times[k]
        
        self.cache[key] = value
        self.access_times[key] = datetime.now()
    
    def clear(self):
        """Clear all cached responses"""
        self.cache.clear()
        self.access_times.clear()

class OptimizedRAG:
    """Optimized RAG system with fast timeouts and concise responses"""
    
    def __init__(self, 
                 ollama_base_url: str = "http://localhost:11434",
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 config: Optional[ResponseConfig] = None):
        """
        Initialize the optimized RAG system
        
        Args:
            ollama_base_url: Base URL for Ollama API
            embedding_model_name: Name of the sentence transformer model
            config: Response configuration
        """
        self.ollama_base_url = ollama_base_url.rstrip('/')
        self.config = config or ResponseConfig()
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            logger.info(f"Loaded embedding model: {embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
        
        # Initialize response cache
        self.response_cache = ResponseCache()
        
        # Available models (will be populated on first use)
        self.available_models = []
        self.preferred_model = None
        
        # Thread pool for parallel operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        logger.info("Optimized RAG system initialized")
    
    def _detect_available_models(self) -> List[str]:
        """Detect available Ollama models"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available = [model['name'] for model in models]
                logger.info(f"Available Ollama models: {available}")
                return available
        except Exception as e:
            logger.warning(f"Failed to detect models: {e}")
        return []
    
    def _select_best_model(self) -> Optional[Tuple[str, int, float]]:
        """Select the best available fast model"""
        if not self.available_models:
            self.available_models = self._detect_available_models()
        
        # Try models in preference order
        for model_config in ModelConfig:
            model_name, timeout, temp = model_config.value
            if any(model_name in available for available in self.available_models):
                logger.info(f"Selected model: {model_name} with {timeout}s timeout")
                return model_name, timeout, temp
        
        # If no preferred model found, use first available
        if self.available_models:
            model = self.available_models[0]
            logger.info(f"Using first available model: {model}")
            return model, self.config.initial_timeout, 0.5
        
        return None
    
    def _create_concise_prompt(self, query: str, context: str) -> str:
        """Create a prompt that encourages concise, direct answers"""
        return f"""You are a helpful assistant that provides CONCISE, DIRECT answers based ONLY on the provided context.

CRITICAL RULES:
1. Answer in 1-3 sentences maximum
2. Use ONLY information from the context below
3. If the answer isn't in the context, say "Information not found in documents"
4. Be direct - no introductions or explanations
5. Focus on the most relevant facts

Context:
{context[:self.config.max_context_length]}

Question: {query}

Concise Answer:"""
    
    def _generate_llm_response(self, query: str, context: str, model_info: Tuple[str, int, float]) -> Optional[str]:
        """Generate LLM response with timeout"""
        model_name, timeout, temperature = model_info
        
        prompt = self._create_concise_prompt(query, context)
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "options": {
                "num_predict": self.config.max_response_length,
                "temperature": temperature,
                "top_p": 0.9,
                "stop": ["Question:", "\n\nContext:", "\n\nHuman:"],
                "num_ctx": 2048  # Smaller context window for faster processing
            },
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                # Enforce length limit
                if len(answer) > self.config.max_response_length:
                    answer = answer[:self.config.max_response_length].rsplit(' ', 1)[0] + "..."
                
                return answer
            else:
                logger.warning(f"LLM generation failed with status {response.status_code}")
                return None
                
        except requests.Timeout:
            logger.info(f"LLM timeout after {timeout}s - falling back to vector search")
            return None
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return None
    
    def _format_vector_search_answer(self, chunks: List[Dict[str, Any]], query: str) -> str:
        """Format vector search results into a concise answer"""
        if not chunks:
            return "No relevant information found in the documents."
        
        # Take top chunks based on similarity
        top_chunks = chunks[:self.config.max_sources]
        
        # Try to extract the most relevant sentences
        relevant_parts = []
        query_terms = set(query.lower().split())
        
        for chunk in top_chunks:
            text = chunk.get('text', '')
            sentences = text.split('. ')
            
            # Find sentences with query terms
            relevant_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(term in sentence_lower for term in query_terms):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                # Take the most relevant sentence
                relevant_parts.append(relevant_sentences[0])
            elif len(text) > 100:
                # If no query terms found, take first 100 chars
                relevant_parts.append(text[:100].strip() + "...")
        
        if relevant_parts:
            # Combine relevant parts into concise answer
            answer = " ".join(relevant_parts[:2])  # Max 2 sentences
            if len(answer) > self.config.max_response_length:
                answer = answer[:self.config.max_response_length].rsplit(' ', 1)[0] + "..."
            return answer
        else:
            # Fallback to first chunk excerpt
            first_chunk = chunks[0].get('text', '')[:150]
            return f"Related information: {first_chunk}..."
    
    def _format_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format source citations"""
        sources = []
        for i, chunk in enumerate(chunks[:self.config.max_sources]):
            sources.append({
                "document": chunk.get('filename', 'Unknown'),
                "excerpt": chunk.get('text', '')[:100] + "...",
                "relevance": f"{chunk.get('similarity', 0):.2f}"
            })
        return sources
    
    async def query_async(self, 
                         query: str, 
                         document_chunks: List[Dict[str, Any]], 
                         document_embeddings: List[List[float]],
                         use_llm: bool = True) -> Dict[str, Any]:
        """
        Async query with fast timeout and intelligent fallback
        
        Args:
            query: User's question
            document_chunks: List of document chunks
            document_embeddings: List of chunk embeddings
            use_llm: Whether to try LLM generation
        
        Returns:
            Dict with answer, sources, and metadata
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = hashlib.md5(f"{query}_{len(document_chunks)}_{use_llm}".encode()).hexdigest()
        cached_response = self.response_cache.get(cache_key)
        if cached_response:
            cached_response['cached'] = True
            cached_response['processing_time'] = time.time() - start_time
            return cached_response
        
        # Find similar chunks using vector search
        similar_chunks = await self._find_similar_chunks_async(query, document_chunks, document_embeddings)
        
        if not similar_chunks:
            response = {
                "answer": "No relevant information found in the documents.",
                "sources": [],
                "method": "no_results",
                "processing_time": time.time() - start_time,
                "cached": False
            }
            return response
        
        # Prepare context
        context = self._prepare_context(similar_chunks)
        sources = self._format_sources(similar_chunks)
        
        answer = None
        method = "vector_search"
        
        # Try LLM generation if requested
        if use_llm and self.preferred_model is None:
            self.preferred_model = self._select_best_model()
        
        if use_llm and self.preferred_model:
            # Try LLM with fast timeout
            answer = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._generate_llm_response,
                query,
                context,
                self.preferred_model
            )
            
            if answer:
                method = "llm_generated"
        
        # Fallback to vector search if LLM failed or not requested
        if not answer:
            answer = self._format_vector_search_answer(similar_chunks, query)
            method = "vector_search" if not use_llm else "vector_search_fallback"
        
        response = {
            "answer": answer,
            "sources": sources,
            "method": method,
            "model": self.preferred_model[0] if self.preferred_model else None,
            "processing_time": time.time() - start_time,
            "cached": False
        }
        
        # Cache the response
        self.response_cache.set(cache_key, response.copy())
        
        return response
    
    def query(self, 
              query: str, 
              document_chunks: List[Dict[str, Any]], 
              document_embeddings: List[List[float]],
              use_llm: bool = True) -> Dict[str, Any]:
        """
        Synchronous wrapper for query_async
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.query_async(query, document_chunks, document_embeddings, use_llm)
            )
        finally:
            loop.close()
    
    async def _find_similar_chunks_async(self, 
                                       query: str, 
                                       document_chunks: List[Dict[str, Any]], 
                                       document_embeddings: List[List[float]],
                                       top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar chunks using vector similarity (async)"""
        if not self.embedding_model or not document_embeddings:
            return []
        
        # Generate query embedding
        query_embedding = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.embedding_model.encode,
            [query]
        )
        query_embedding = query_embedding[0]
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], document_embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Filter by minimum similarity threshold
        min_similarity = 0.3
        results = []
        
        for idx in top_indices:
            if similarities[idx] >= min_similarity and idx < len(document_chunks):
                chunk = document_chunks[idx].copy()
                chunk['similarity'] = float(similarities[idx])
                results.append(chunk)
        
        return results
    
    def _prepare_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Prepare context for LLM"""
        context_parts = []
        current_length = 0
        
        for chunk in chunks:
            text = chunk.get('text', '')
            filename = chunk.get('filename', 'Unknown')
            
            # Format chunk
            chunk_text = f"[{filename}]: {text}"
            
            if current_length + len(chunk_text) > self.config.max_context_length:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "\n\n".join(context_parts)
    
    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        logger.info("Response cache cleared")
    
    def set_preferred_model(self, model_name: str, timeout: int = 5, temperature: float = 0.5):
        """Manually set preferred model"""
        self.preferred_model = (model_name, timeout, temperature)
        logger.info(f"Set preferred model: {model_name}")
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# FastAPI Integration
def create_optimized_endpoints(app, rag_system: OptimizedRAG, document_chunks: List, document_embeddings: List):
    """Create optimized FastAPI endpoints"""
    
    from fastapi import HTTPException
    from pydantic import BaseModel
    
    class OptimizedQueryRequest(BaseModel):
        query: str
        use_llm: Optional[bool] = True
        max_response_length: Optional[int] = 300
    
    class OptimizedQueryResponse(BaseModel):
        answer: str
        sources: List[Dict[str, Any]]
        method: str
        model: Optional[str]
        processing_time: float
        cached: bool
    
    @app.post("/api/v1/query/optimized", response_model=OptimizedQueryResponse)
    async def optimized_query(request: OptimizedQueryRequest):
        """Optimized query endpoint with fast timeouts and concise responses"""
        if not document_chunks:
            raise HTTPException(
                status_code=400,
                detail="No documents uploaded. Please upload documents first."
            )
        
        # Update config if custom max length provided
        if request.max_response_length:
            rag_system.config.max_response_length = request.max_response_length
        
        try:
            result = await rag_system.query_async(
                query=request.query,
                document_chunks=document_chunks,
                document_embeddings=document_embeddings,
                use_llm=request.use_llm
            )
            
            return OptimizedQueryResponse(**result)
            
        except Exception as e:
            logger.error(f"Optimized query error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Query processing error: {str(e)}"
            )
    
    @app.post("/api/v1/cache/clear")
    async def clear_cache():
        """Clear the response cache"""
        rag_system.clear_cache()
        return {"message": "Cache cleared successfully"}
    
    @app.post("/api/v1/model/set")
    async def set_model(model_name: str, timeout: int = 5):
        """Set preferred model"""
        rag_system.set_preferred_model(model_name, timeout)
        return {"message": f"Model set to {model_name} with {timeout}s timeout"}
    
    @app.get("/api/v1/model/list")
    async def list_models():
        """List available models"""
        models = rag_system._detect_available_models()
        return {"models": models, "preferred": rag_system.preferred_model}


# Example usage and integration
if __name__ == "__main__":
    # Example of using the optimized RAG system
    rag = OptimizedRAG()
    
    # Example document chunks and embeddings
    example_chunks = [
        {"text": "Python is a high-level programming language.", "filename": "python_intro.txt"},
        {"text": "Machine learning is a subset of artificial intelligence.", "filename": "ml_basics.txt"}
    ]
    
    # Generate example embeddings
    if rag.embedding_model:
        texts = [chunk["text"] for chunk in example_chunks]
        example_embeddings = rag.embedding_model.encode(texts).tolist()
        
        # Test query
        result = rag.query(
            query="What is Python?",
            document_chunks=example_chunks,
            document_embeddings=example_embeddings,
            use_llm=True
        )
        
        print(f"Answer: {result['answer']}")
        print(f"Method: {result['method']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Sources: {result['sources']}")