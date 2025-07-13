#!/usr/bin/env python3
"""
Smart Answer Generation Service
Provides intelligent document-based responses with relevance scoring
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class AnswerType(Enum):
    DOCUMENT_BASED = "document_based"      # Answer found in documents
    LLM_GENERATED = "llm_generated"        # LLM generated answer (marked as such)
    NO_ANSWER = "no_answer"                # No relevant information found

class ConfidenceLevel(Enum):
    HIGH = "high"           # Very relevant (>0.8)
    MEDIUM = "medium"       # Moderately relevant (0.6-0.8)
    LOW = "low"             # Weakly relevant (0.4-0.6)
    INSUFFICIENT = "insufficient"  # Below threshold (<0.4)

@dataclass
class SmartAnswerResult:
    """Result of smart answer generation"""
    answer: str
    answer_type: AnswerType
    confidence: ConfidenceLevel
    confidence_score: float
    sources: List[Dict[str, Any]]
    reasoning: str
    chunk_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "answer_type": self.answer_type.value,
            "confidence": self.confidence.value,
            "confidence_score": round(self.confidence_score, 3),
            "sources": self.sources,
            "reasoning": self.reasoning,
            "chunk_count": self.chunk_count,
            "is_document_based": self.answer_type == AnswerType.DOCUMENT_BASED,
            "is_llm_generated": self.answer_type == AnswerType.LLM_GENERATED
        }

class SmartAnswerEngine:
    """
    Intelligent answer generation with relevance scoring and document-first approach
    """
    
    def __init__(self):
        # Relevance thresholds - balanced settings
        self.HIGH_CONFIDENCE_THRESHOLD = 0.8
        self.MEDIUM_CONFIDENCE_THRESHOLD = 0.6
        self.MIN_RELEVANCE_THRESHOLD = 0.4  # 40% minimum threshold
        
        # Answer generation settings
        self.MAX_SOURCES_FOR_ANSWER = 5
        self.MIN_CHUNK_LENGTH_FOR_QUALITY = 100
        
        logger.info("Smart Answer Engine initialized")
    
    def generate_smart_answer(self, query: str, similar_chunks: List[Dict[str, Any]], 
                            llm_client=None, use_llm_fallback: bool = True) -> SmartAnswerResult:
        """
        Generate intelligent answer based on document relevance and LLM capabilities
        
        Args:
            query: User's question
            similar_chunks: List of similar document chunks with scores
            llm_client: Optional LLM client for fallback
            use_llm_fallback: Whether to use LLM when documents are insufficient
        """
        
        # Step 1: Analyze document relevance
        relevance_analysis = self._analyze_relevance(query, similar_chunks)
        
        # Step 2: Decide answer strategy based on relevance
        if relevance_analysis["max_confidence"] >= self.MIN_RELEVANCE_THRESHOLD:
            # Documents contain relevant information
            return self._generate_document_based_answer(query, similar_chunks, relevance_analysis)
        
        elif use_llm_fallback and llm_client and llm_client.is_available():
            # Documents insufficient, but LLM available
            return self._generate_llm_fallback_answer(query, similar_chunks, llm_client, relevance_analysis)
        
        else:
            # No relevant documents and no LLM fallback
            return self._generate_no_answer_response(query, relevance_analysis)
    
    def _analyze_relevance(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the relevance of document chunks to the query"""
        
        if not chunks:
            return {
                "max_confidence": 0.0,
                "avg_confidence": 0.0,
                "relevant_chunks": 0,
                "total_chunks": 0,
                "confidence_distribution": {"high": 0, "medium": 0, "low": 0, "insufficient": 0}
            }
        
        # Extract confidence scores
        confidences = [chunk.get('similarity', 0.0) for chunk in chunks]
        max_confidence = max(confidences) if confidences else 0.0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Count relevant chunks by confidence level
        high_conf = sum(1 for conf in confidences if conf >= self.HIGH_CONFIDENCE_THRESHOLD)
        medium_conf = sum(1 for conf in confidences if self.MEDIUM_CONFIDENCE_THRESHOLD <= conf < self.HIGH_CONFIDENCE_THRESHOLD)
        low_conf = sum(1 for conf in confidences if self.MIN_RELEVANCE_THRESHOLD <= conf < self.MEDIUM_CONFIDENCE_THRESHOLD)
        insufficient = len(confidences) - high_conf - medium_conf - low_conf
        
        relevant_chunks = high_conf + medium_conf + low_conf
        
        return {
            "max_confidence": max_confidence,
            "avg_confidence": avg_confidence,
            "relevant_chunks": relevant_chunks,
            "total_chunks": len(chunks),
            "confidence_distribution": {
                "high": high_conf,
                "medium": medium_conf, 
                "low": low_conf,
                "insufficient": insufficient
            }
        }
    
    def _generate_document_based_answer(self, query: str, chunks: List[Dict[str, Any]], 
                                       analysis: Dict[str, Any]) -> SmartAnswerResult:
        """Generate answer based on relevant document chunks"""
        
        # Filter chunks by relevance threshold
        relevant_chunks = [
            chunk for chunk in chunks 
            if chunk.get('similarity', 0.0) >= self.MIN_RELEVANCE_THRESHOLD
        ][:self.MAX_SOURCES_FOR_ANSWER]
        
        # Determine confidence level
        max_score = analysis["max_confidence"]
        if max_score >= self.HIGH_CONFIDENCE_THRESHOLD:
            confidence = ConfidenceLevel.HIGH
        elif max_score >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW
        
        # Generate document-based answer
        answer_parts = []
        sources = []
        
        for i, chunk in enumerate(relevant_chunks, 1):
            content = chunk.get('content', '')
            source_doc = chunk.get('source_document', 'Unknown')
            score = chunk.get('similarity', 0.0)
            
            # Add numbered source
            answer_parts.append(f"{i}. {content}")
            
            sources.append({
                "source_document": source_doc,
                "content": content,
                "similarity_score": round(score, 3),
                "relevance": self._get_relevance_label(score)
            })
        
        # Create comprehensive answer
        if confidence == ConfidenceLevel.HIGH:
            answer_intro = "Based on the documents, here's what I found:"
        elif confidence == ConfidenceLevel.MEDIUM:
            answer_intro = "The documents contain some relevant information:"
        else:
            answer_intro = "I found limited relevant information in the documents:"
        
        full_answer = f"{answer_intro}\n\n" + "\n\n".join(answer_parts)
        
        # Add source information
        source_docs = list(set(source['source_document'] for source in sources))
        full_answer += f"\n\nSources: {', '.join(source_docs)}"
        
        reasoning = (f"Found {analysis['relevant_chunks']} relevant chunks out of {analysis['total_chunks']} "
                    f"with maximum confidence {max_score:.3f}. Using document-based answer.")
        
        return SmartAnswerResult(
            answer=full_answer,
            answer_type=AnswerType.DOCUMENT_BASED,
            confidence=confidence,
            confidence_score=max_score,
            sources=sources,
            reasoning=reasoning,
            chunk_count=len(relevant_chunks)
        )
    
    def _generate_llm_fallback_answer(self, query: str, chunks: List[Dict[str, Any]], 
                                    llm_client, analysis: Dict[str, Any]) -> SmartAnswerResult:
        """Generate LLM-based answer when documents are insufficient"""
        
        try:
            # Prepare context from best available chunks (even if low relevance)
            context_chunks = chunks[:3] if chunks else []
            context = "\n".join([chunk.get('content', '') for chunk in context_chunks])
            
            # Generate LLM response with clear instructions
            llm_prompt = f"""
Question: {query}

Available document context (may be limited or not directly relevant):
{context if context else "No relevant document context available."}

Instructions: 
1. If the document context answers the question, use it
2. If not, provide a helpful general answer but CLEARLY state this is not from the documents
3. Be honest about the limitations of the available information

Answer:"""
            
            llm_response = llm_client.generate_answer(llm_prompt, max_tokens=300)
            
            # Format the LLM response with clear marking
            marked_answer = f"""⚠️ **LLM-Generated Response** (Not from documents)

{llm_response}

---
*Note: This answer was generated by the AI model because the uploaded documents don't contain sufficient information to answer your question. For document-based answers, please ensure your question relates to the uploaded content.*"""
            
            # Create sources from best chunks (if any)
            sources = []
            if context_chunks:
                for chunk in context_chunks[:2]:
                    sources.append({
                        "source_document": chunk.get('source_document', 'Unknown'),
                        "content": chunk.get('content', '')[:200] + "...",
                        "similarity_score": round(chunk.get('similarity', 0.0), 3),
                        "relevance": "insufficient_for_answer"
                    })
            
            reasoning = (f"Documents insufficient (max confidence: {analysis['max_confidence']:.3f}). "
                        f"Used LLM fallback with available context.")
            
            return SmartAnswerResult(
                answer=marked_answer,
                answer_type=AnswerType.LLM_GENERATED,
                confidence=ConfidenceLevel.INSUFFICIENT,
                confidence_score=analysis['max_confidence'],
                sources=sources,
                reasoning=reasoning,
                chunk_count=len(context_chunks)
            )
            
        except Exception as e:
            logger.error(f"LLM fallback failed: {e}")
            return self._generate_no_answer_response(query, analysis)
    
    def _generate_no_answer_response(self, query: str, analysis: Dict[str, Any]) -> SmartAnswerResult:
        """Generate response when no relevant information is available"""
        
        if analysis['total_chunks'] == 0:
            answer = """❌ **No Information Available**

I don't have any documents to search through. Please upload some documents first, then ask your question again.

To get better results:
1. Upload documents related to your question
2. Make sure the documents contain the information you're looking for
3. Try rephrasing your question"""
        
        else:
            answer = f"""❌ **No Relevant Information Found**

I searched through {analysis['total_chunks']} document chunks but couldn't find information relevant to your question: "{query}"

The documents I searched don't appear to contain information about this topic.

To get better results:
1. Try rephrasing your question with different keywords
2. Upload additional documents that might contain the answer
3. Check if your question relates to the content of the uploaded documents

**Highest similarity found:** {analysis['max_confidence']:.3f} (below threshold of {self.MIN_RELEVANCE_THRESHOLD}). The documents don't contain information about this topic."""
        
        reasoning = f"No relevant information found. Max confidence {analysis['max_confidence']:.3f} below threshold {self.MIN_RELEVANCE_THRESHOLD}."
        
        return SmartAnswerResult(
            answer=answer,
            answer_type=AnswerType.NO_ANSWER,
            confidence=ConfidenceLevel.INSUFFICIENT,
            confidence_score=analysis['max_confidence'],
            sources=[],
            reasoning=reasoning,
            chunk_count=0
        )
    
    def _get_relevance_label(self, score: float) -> str:
        """Get human-readable relevance label for a similarity score"""
        if score >= self.HIGH_CONFIDENCE_THRESHOLD:
            return "highly_relevant"
        elif score >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            return "moderately_relevant"
        elif score >= self.MIN_RELEVANCE_THRESHOLD:
            return "somewhat_relevant"
        else:
            return "low_relevance"
    
    def suggest_better_chunking(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze documents and suggest better chunking strategies"""
        
        if not documents:
            return {"suggestion": "No documents to analyze"}
        
        total_docs = len(documents)
        total_chunks = sum(doc.get('chunks_count', 0) for doc in documents)
        avg_chunks_per_doc = total_chunks / total_docs if total_docs > 0 else 0
        
        suggestions = []
        
        # Analyze chunk density
        if avg_chunks_per_doc < 5:
            suggestions.append("Consider using smaller chunk sizes - very few chunks per document")
        elif avg_chunks_per_doc > 50:
            suggestions.append("Consider using larger chunk sizes - too many small chunks may dilute relevance")
        
        # Check document sizes
        large_docs = sum(1 for doc in documents if doc.get('file_size', 0) > 1024*1024)  # >1MB
        if large_docs > 0:
            suggestions.append(f"{large_docs} large documents detected - ensure chunking is appropriate for document size")
        
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "avg_chunks_per_document": round(avg_chunks_per_doc, 1),
            "suggestions": suggestions,
            "recommended_chunk_size": "800-1200 characters for balanced context and relevance"
        }

# Global smart answer engine instance
smart_answer_engine: Optional[SmartAnswerEngine] = None

def get_smart_answer_engine() -> SmartAnswerEngine:
    """Get global smart answer engine instance"""
    global smart_answer_engine
    if smart_answer_engine is None:
        smart_answer_engine = SmartAnswerEngine()
    return smart_answer_engine