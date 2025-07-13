#!/usr/bin/env python3
"""
Re-ranking Service for Better Context Selection
Improves answer quality by intelligently selecting the most relevant chunks
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, util
import re

logger = logging.getLogger(__name__)

class ChunkReranker:
    """
    Re-ranks chunks based on multiple factors:
    - Semantic similarity
    - Keyword overlap
    - Context coherence
    - Information density
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Reranker initialized with model: {model_name}")
    
    def rerank_chunks(self, query: str, chunks: List[Dict[str, Any]], 
                      top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Re-rank chunks for better relevance
        
        Args:
            query: User query
            chunks: List of chunk dictionaries with 'text' and 'similarity'
            top_k: Number of top chunks to return
            
        Returns:
            Re-ranked list of chunks
        """
        
        if not chunks:
            return []
        
        # Calculate multiple scoring factors
        scored_chunks = []
        
        for chunk in chunks:
            score = self._calculate_chunk_score(query, chunk)
            chunk_copy = chunk.copy()
            chunk_copy['rerank_score'] = score
            scored_chunks.append(chunk_copy)
        
        # Sort by rerank score
        scored_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        logger.info(f"Re-ranked {len(chunks)} chunks, returning top {top_k}")
        return scored_chunks[:top_k]
    
    def _calculate_chunk_score(self, query: str, chunk: Dict[str, Any]) -> float:
        """
        Calculate comprehensive relevance score for a chunk
        """
        
        text = chunk.get('text', '')
        base_similarity = chunk.get('similarity', 0.0)
        
        # 1. Base similarity (40% weight)
        score = base_similarity * 0.4
        
        # 2. Keyword overlap (20% weight)
        keyword_score = self._calculate_keyword_overlap(query, text)
        score += keyword_score * 0.2
        
        # 3. Information density (20% weight)
        density_score = self._calculate_information_density(text)
        score += density_score * 0.2
        
        # 4. Answer likelihood (20% weight)
        answer_score = self._calculate_answer_likelihood(query, text)
        score += answer_score * 0.2
        
        return score
    
    def _calculate_keyword_overlap(self, query: str, text: str) -> float:
        """
        Calculate keyword overlap between query and text
        """
        # Extract meaningful words (3+ characters)
        query_words = set(w.lower() for w in re.findall(r'\b\w{3,}\b', query))
        text_words = set(w.lower() for w in re.findall(r'\b\w{3,}\b', text))
        
        if not query_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(query_words & text_words)
        union = len(query_words | text_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_information_density(self, text: str) -> float:
        """
        Calculate information density of text
        Higher scores for text with:
        - Numbers/dates
        - Named entities
        - Action words
        - Specific information
        """
        
        score = 0.0
        
        # Check for numbers and dates
        numbers = re.findall(r'\d+', text)
        score += min(len(numbers) * 0.1, 0.3)
        
        # Check for structured information (lists, steps)
        if re.search(r'^\s*[-•*\d]+\.?\s+', text, re.MULTILINE):
            score += 0.2
        
        # Check for specific indicators
        indicators = ['müssen', 'sollen', 'können', 'wo', 'wann', 'wie', 
                     'öffnungszeiten', 'adresse', 'telefon', 'email', 'kontakt']
        
        text_lower = text.lower()
        for indicator in indicators:
            if indicator in text_lower:
                score += 0.1
        
        # Check for concrete vs abstract language
        concrete_words = ['ort', 'zeit', 'datum', 'uhr', 'euro', 'prozent', 
                         'meter', 'kilogramm', 'liter', 'stück']
        
        for word in concrete_words:
            if word in text_lower:
                score += 0.05
        
        return min(score, 1.0)
    
    def _calculate_answer_likelihood(self, query: str, text: str) -> float:
        """
        Calculate likelihood that text contains answer to query
        """
        
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Question type detection
        score = 0.0
        
        # Who/What/Where/When/Why/How questions
        if 'wer' in query_lower and any(word in text_lower for word in ['person', 'menschen', 'mitarbeiter', 'team']):
            score += 0.3
        
        if 'was' in query_lower and len(text.split()) > 20:  # Explanatory text
            score += 0.2
        
        if 'wo' in query_lower and any(word in text_lower for word in ['ort', 'adresse', 'standort', 'gebäude']):
            score += 0.4
        
        if 'wann' in query_lower and any(word in text_lower for word in ['zeit', 'uhr', 'öffnung', 'datum', 'tag']):
            score += 0.4
        
        if 'wie' in query_lower and any(word in text_lower for word in ['schritt', 'anleitung', 'müssen', 'können']):
            score += 0.3
        
        # Check for direct answer patterns
        if 'ist' in query_lower or 'sind' in query_lower:
            # Definition-style answers
            if ':' in text or '=' in text or 'bedeutet' in text_lower:
                score += 0.2
        
        return min(score, 1.0)
    
    def create_coherent_context(self, chunks: List[Dict[str, Any]], 
                              max_length: int = 3000) -> str:
        """
        Create coherent context from chunks
        """
        
        if not chunks:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '').strip()
            
            if not text:
                continue
            
            # Add chunk with metadata
            chunk_intro = f"\n[Dokument {i+1}, Relevanz: {chunk.get('rerank_score', chunk.get('similarity', 0)):.2f}]\n"
            
            if current_length + len(chunk_intro) + len(text) > max_length:
                # Truncate if necessary
                remaining = max_length - current_length - len(chunk_intro)
                if remaining > 100:  # Only include if we have reasonable space
                    text = text[:remaining] + "..."
                else:
                    break
            
            context_parts.append(chunk_intro + text)
            current_length += len(chunk_intro) + len(text)
        
        return "\n".join(context_parts)

# Global reranker instance
_reranker = None

def get_reranker() -> ChunkReranker:
    """Get or create reranker instance"""
    global _reranker
    if _reranker is None:
        _reranker = ChunkReranker()
    return _reranker