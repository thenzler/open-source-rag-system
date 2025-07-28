#!/usr/bin/env python3
"""
Hybrid Search Service
Combines semantic search with keyword search for better retrieval
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

class HybridSearch:
    """
    Combines multiple search strategies:
    1. Semantic search (embeddings)
    2. Keyword search (BM25)
    3. Fuzzy matching
    """
    
    def __init__(self):
        self.bm25_index = None
        self.documents = []
        self.tokenized_docs = []
        logger.info("Hybrid search initialized")
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Index documents for keyword search
        
        Args:
            documents: List of document chunks with 'text' field
        """
        self.documents = documents
        
        # Tokenize documents for BM25
        self.tokenized_docs = [
            self._tokenize(doc.get('text', '')) 
            for doc in documents
        ]
        
        # Create BM25 index
        if self.tokenized_docs:
            self.bm25_index = BM25Okapi(self.tokenized_docs)
            logger.info(f"Indexed {len(documents)} documents for BM25 search")
    
    def hybrid_search(self, query: str, 
                     semantic_results: List[Dict[str, Any]],
                     top_k: int = 10,
                     semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword results
        
        Args:
            query: Search query
            semantic_results: Results from semantic search
            top_k: Number of results to return
            semantic_weight: Weight for semantic search (0-1)
            
        Returns:
            Combined and re-ranked results
        """
        
        if not self.bm25_index:
            logger.warning("BM25 index not initialized, returning semantic results only")
            return semantic_results[:top_k]
        
        # Get keyword search results
        keyword_results = self._keyword_search(query, top_k * 2)
        
        # Combine results
        combined_results = self._combine_results(
            semantic_results, 
            keyword_results,
            semantic_weight
        )
        
        # Add additional scoring factors
        final_results = self._apply_boosting(query, combined_results)
        
        # Sort by final score
        final_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return final_results[:top_k]
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep German umlauts
        text = re.sub(r'[^\w\säöüß-]', ' ', text)
        
        # Split into tokens
        tokens = text.split()
        
        # Remove stopwords (basic German stopwords)
        stopwords = {
            'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einer',
            'eines', 'einem', 'einen', 'und', 'oder', 'aber', 'als', 'am',
            'an', 'auf', 'aus', 'bei', 'bis', 'durch', 'für', 'gegen', 'in',
            'mit', 'nach', 'seit', 'über', 'um', 'von', 'vor', 'zu', 'zur'
        }
        
        tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
        
        return tokens
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search
        """
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                result = self.documents[idx].copy()
                result['bm25_score'] = float(scores[idx])
                result['bm25_rank'] = len(results) + 1
                results.append(result)
        
        return results
    
    def _combine_results(self, semantic_results: List[Dict[str, Any]],
                        keyword_results: List[Dict[str, Any]],
                        semantic_weight: float) -> List[Dict[str, Any]]:
        """
        Combine semantic and keyword search results
        """
        keyword_weight = 1.0 - semantic_weight
        
        # Create mapping of chunk_id to results
        combined = defaultdict(dict)
        
        # Add semantic results
        for i, result in enumerate(semantic_results):
            chunk_id = result.get('chunk_id', id(result))
            combined[chunk_id].update(result)
            combined[chunk_id]['semantic_rank'] = i + 1
            combined[chunk_id]['semantic_score'] = result.get('similarity', 0)
        
        # Add keyword results
        for result in keyword_results:
            chunk_id = result.get('chunk_id', id(result))
            if chunk_id not in combined:
                combined[chunk_id].update(result)
            combined[chunk_id]['bm25_score'] = result.get('bm25_score', 0)
            combined[chunk_id]['bm25_rank'] = result.get('bm25_rank', 999)
        
        # Calculate hybrid scores
        results = []
        for chunk_id, data in combined.items():
            # Normalize scores
            semantic_score = data.get('semantic_score', 0)
            bm25_score = data.get('bm25_score', 0)
            
            # Normalize BM25 score (typically 0-10 range)
            normalized_bm25 = min(bm25_score / 10.0, 1.0) if bm25_score > 0 else 0
            
            # Calculate hybrid score
            hybrid_score = (semantic_score * semantic_weight + 
                          normalized_bm25 * keyword_weight)
            
            # Boost if appears in both results
            if 'semantic_score' in data and 'bm25_score' in data:
                hybrid_score *= 1.2  # 20% boost
            
            data['hybrid_score'] = hybrid_score
            results.append(data)
        
        return results
    
    def _apply_boosting(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply additional boosting factors
        """
        query_lower = query.lower()
        
        for result in results:
            text = result.get('text', '').lower()
            boost = 1.0
            
            # Exact phrase match boost
            if query_lower in text:
                boost *= 1.5
            
            # Title/header boost
            if result.get('chunk_index', 999) < 3:  # Early chunks often contain overview
                boost *= 1.1
            
            # Question answering boost
            if '?' in query:
                # Boost chunks that look like answers
                if any(indicator in text for indicator in [
                    'antwort', 'lösung', 'bedeutet', 'ist', 'sind',
                    'müssen sie', 'können sie', 'sollten sie'
                ]):
                    boost *= 1.2
            
            # Apply boost
            result['hybrid_score'] = result.get('hybrid_score', 0) * boost
        
        return results

# Global instance
_hybrid_search = None

def get_hybrid_search() -> HybridSearch:
    """Get or create hybrid search instance"""
    global _hybrid_search
    if _hybrid_search is None:
        _hybrid_search = HybridSearch()
    return _hybrid_search