#!/usr/bin/env python3
"""
Improved Text Chunking Service
Creates larger, more meaningful chunks with better context preservation
"""

import re
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ChunkingConfig:
    """Configuration for text chunking"""
    target_chunk_size: int = 1200      # Target characters per chunk (optimal for context)
    max_chunk_size: int = 1500         # Maximum characters per chunk  
    min_chunk_size: int = 400          # Minimum characters per chunk
    overlap_size: int = 200            # Character overlap between chunks
    preserve_sentences: bool = True     # Try to keep sentences intact
    preserve_paragraphs: bool = True    # Try to keep paragraphs together
    separate_practical_content: bool = False  # Disabled - let LLM handle this

class ImprovedChunker:
    """
    Advanced text chunking with context preservation and overlap
    """
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_breaks = re.compile(r'\n\s*\n')
        
        # Document structure markers
        self.section_headers = re.compile(r'^#{1,6}\s+.*$|^[A-Z][^.!?]*:$', re.MULTILINE)
        self.list_items = re.compile(r'^\s*[-*•]\s+|^\s*\d+\.\s+', re.MULTILINE)
        
        # Universal content classification patterns (language/domain agnostic)
        self.structural_patterns = {
            'lists': re.compile(r'^\s*[-*•]\s+|^\s*\d+\.\s+', re.MULTILINE),
            'headers': re.compile(r'^[A-Z][^.!?]*:$|^#{1,6}\s+', re.MULTILINE),  
            'instructions': re.compile(r'\b(how to|step|first|then|next|finally|follow|do|don\'t)\b', re.IGNORECASE),
            'actions': re.compile(r'\b(should|must|can|may|will|need to|have to|able to)\b', re.IGNORECASE),
            'contact_info': re.compile(r'\b(phone|tel|email|address|contact|hours|open)\b', re.IGNORECASE),
            'numbers_data': re.compile(r'\d+[.,]\d+\s*(%|percent|million|billion|kg|tons?|meters?)', re.IGNORECASE),
            'research_terms': re.compile(r'\b(study|research|analysis|according to|results show|data indicates)\b', re.IGNORECASE),
            'citations': re.compile(r'\([^)]*\d{4}[^)]*\)|et al\.|ibid\.|op\. cit\.', re.IGNORECASE)
        }
        
        logger.info(f"Improved chunker initialized with target size: {self.config.target_chunk_size}")
    
    def chunk_text(self, text: str, document_name: str = "unknown") -> List[Dict[str, Any]]:
        """
        Create improved chunks with better context preservation
        
        Args:
            text: Text to chunk
            document_name: Name of the source document
            
        Returns:
            List of chunk dictionaries with metadata
        """
        
        if not text or len(text.strip()) < self.config.min_chunk_size:
            return []
        
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        # First try paragraph-based chunking
        if self.config.preserve_paragraphs:
            chunks = self._chunk_by_paragraphs(cleaned_text)
        else:
            chunks = self._chunk_by_sentences(cleaned_text)
        
        # If chunks are too large or small, re-chunk
        chunks = self._optimize_chunk_sizes(chunks)
        
        # Add overlaps for better context
        chunks = self._add_overlaps(chunks)
        
        # Classify content types if enabled
        if self.config.separate_practical_content:
            chunks = self._classify_and_prioritize_chunks(chunks)
        
        # Create final chunk objects with metadata
        return self._create_chunk_objects(chunks, document_name)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better chunking"""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Add space after sentence endings
        
        # Normalize quotes and dashes
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r'[''`]', "'", text)
        text = re.sub(r'[—–]', '-', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text.strip()
    
    def _chunk_by_paragraphs(self, text: str) -> List[str]:
        """Chunk text by paragraphs, combining small ones"""
        
        paragraphs = self.paragraph_breaks.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed max size, finalize current chunk
            if (len(current_chunk) + len(paragraph) > self.config.max_chunk_size and 
                len(current_chunk) >= self.config.min_chunk_size):
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            
            # If current chunk is very large, split it by sentences
            if len(current_chunk) > self.config.max_chunk_size:
                sentence_chunks = self._chunk_by_sentences(current_chunk)
                if len(sentence_chunks) > 1:
                    chunks.extend(sentence_chunks[:-1])
                    current_chunk = sentence_chunks[-1]
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Chunk text by sentences when paragraph chunking isn't sufficient"""
        
        sentences = self.sentence_endings.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed target size
            if (len(current_chunk) + len(sentence) > self.config.target_chunk_size and 
                len(current_chunk) >= self.config.min_chunk_size):
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _optimize_chunk_sizes(self, chunks: List[str]) -> List[str]:
        """Optimize chunk sizes by merging small chunks and splitting large ones"""
        
        optimized = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # If chunk is too small, try to merge with next chunk
            if (len(current_chunk) < self.config.min_chunk_size and 
                i + 1 < len(chunks) and 
                len(current_chunk) + len(chunks[i + 1]) <= self.config.max_chunk_size):
                current_chunk += "\n\n" + chunks[i + 1]
                i += 2  # Skip next chunk as it's been merged
            else:
                i += 1
            
            # If chunk is still too large, split it more aggressively
            if len(current_chunk) > self.config.max_chunk_size:
                split_chunks = self._force_split_chunk(current_chunk)
                optimized.extend(split_chunks)
            else:
                optimized.append(current_chunk)
        
        return optimized
    
    def _force_split_chunk(self, chunk: str) -> List[str]:
        """Force split a chunk that's too large"""
        
        # Try to split at sentence boundaries first
        sentences = self.sentence_endings.split(chunk)
        if len(sentences) > 1:
            return self._chunk_by_sentences(chunk)
        
        # If no sentence boundaries, split at word boundaries
        words = chunk.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            if len(current_chunk) + len(word) + 1 > self.config.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                if current_chunk:
                    current_chunk += " " + word
                else:
                    current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _add_overlaps(self, chunks: List[str]) -> List[str]:
        """Add overlapping content between chunks for better context"""
        
        if len(chunks) <= 1 or self.config.overlap_size <= 0:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            enhanced_chunk = chunk
            
            # Add overlap from previous chunk (beginning)
            if i > 0:
                prev_chunk = chunks[i - 1]
                if len(prev_chunk) > self.config.overlap_size:
                    # Take last part of previous chunk
                    overlap_start = len(prev_chunk) - self.config.overlap_size
                    # Try to start at word boundary
                    space_pos = prev_chunk.find(' ', overlap_start)
                    if space_pos != -1 and space_pos < len(prev_chunk) - 50:
                        overlap_start = space_pos + 1
                    
                    overlap = prev_chunk[overlap_start:]
                    enhanced_chunk = f"...{overlap} {chunk}"
            
            # Add overlap from next chunk (end)
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                if len(next_chunk) > self.config.overlap_size:
                    # Take first part of next chunk
                    overlap_end = self.config.overlap_size
                    # Try to end at word boundary
                    space_pos = next_chunk.rfind(' ', 0, overlap_end)
                    if space_pos != -1 and space_pos > 50:
                        overlap_end = space_pos
                    
                    overlap = next_chunk[:overlap_end]
                    enhanced_chunk = f"{enhanced_chunk} {overlap}..."
            
            overlapped_chunks.append(enhanced_chunk)
        
        return overlapped_chunks
    
    def _create_chunk_objects(self, chunks: List[str], document_name: str) -> List[Dict[str, Any]]:
        """Create chunk objects with metadata"""
        
        chunk_objects = []
        total_chars = sum(len(chunk) for chunk in chunks)
        
        for i, chunk in enumerate(chunks):
            # Calculate chunk quality score
            quality_score = self._calculate_chunk_quality(chunk)
            
            chunk_obj = {
                "text": chunk,
                "chunk_index": i,
                "character_count": len(chunk),
                "word_count": len(chunk.split()),
                "quality_score": quality_score,
                "document_name": document_name,
                "relative_position": i / len(chunks) if len(chunks) > 1 else 0.0,
                "size_category": self._get_size_category(len(chunk)),
                "has_overlap": "..." in chunk,
                "structure_hints": self._detect_structure_hints(chunk)
            }
            
            chunk_objects.append(chunk_obj)
        
        return chunk_objects
    
    def _calculate_chunk_quality(self, chunk: str) -> float:
        """Calculate a quality score for a chunk (0.0 to 1.0)"""
        
        score = 0.0
        
        # Size score (prefer chunks near target size)
        size_ratio = len(chunk) / self.config.target_chunk_size
        if 0.8 <= size_ratio <= 1.2:
            score += 0.3
        elif 0.5 <= size_ratio <= 1.5:
            score += 0.2
        elif 0.3 <= size_ratio <= 2.0:
            score += 0.1
        
        # Sentence completeness (prefer complete sentences)
        if chunk.strip().endswith(('.', '!', '?')):
            score += 0.2
        
        # Paragraph structure (prefer complete paragraphs)
        if '\n\n' in chunk or len(chunk) > self.config.target_chunk_size * 0.8:
            score += 0.1
        
        # Content richness (prefer chunks with varied punctuation)
        punctuation_variety = len(set(chunk) & set('.,;:!?()[]"')) / 10
        score += min(punctuation_variety, 0.2)
        
        # Avoid very short chunks
        if len(chunk) < self.config.min_chunk_size:
            score *= 0.5
        
        # Avoid chunks with too much repetition
        words = chunk.lower().split()
        if len(words) > 10:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)
            if repetition_ratio < 0.5:
                score *= 0.7
        
        return min(score, 1.0)
    
    def _get_size_category(self, char_count: int) -> str:
        """Categorize chunk by size"""
        if char_count < self.config.min_chunk_size:
            return "small"
        elif char_count > self.config.max_chunk_size:
            return "large"
        elif self.config.target_chunk_size * 0.8 <= char_count <= self.config.target_chunk_size * 1.2:
            return "optimal"
        else:
            return "medium"
    
    def _detect_structure_hints(self, chunk: str) -> List[str]:
        """Detect structural elements in the chunk"""
        hints = []
        
        if self.section_headers.search(chunk):
            hints.append("has_headers")
        
        if self.list_items.search(chunk):
            hints.append("has_lists")
        
        if chunk.count('\n\n') >= 2:
            hints.append("multi_paragraph")
        
        if any(keyword in chunk.lower() for keyword in ['table', 'figure', 'chart', 'graph']):
            hints.append("contains_references")
        
        if len(re.findall(r'\d+', chunk)) >= 3:
            hints.append("number_heavy")
        
        if chunk.count('"') >= 2 or chunk.count("'") >= 2:
            hints.append("has_quotes")
        
        return hints
    
    def _classify_and_prioritize_chunks(self, chunks: List[str]) -> List[str]:
        """
        Classify chunks by content type and prioritize actionable/practical content
        Universal approach that works across domains and languages
        """
        classified_chunks = []
        
        for chunk in chunks:
            # Calculate content type scores
            practicality_score = self._calculate_practicality_score(chunk)
            
            # Add metadata prefix for better retrieval
            if practicality_score > 0.6:
                # High practical value - add priority marker
                classified_chunks.append(f"[PRACTICAL] {chunk}")
            elif practicality_score < 0.3:
                # Low practical value - mark as background/technical
                classified_chunks.append(f"[BACKGROUND] {chunk}")
            else:
                # Medium practical value - neutral
                classified_chunks.append(chunk)
        
        return classified_chunks
    
    def _calculate_practicality_score(self, text: str) -> float:
        """
        Calculate how practical/actionable a piece of text is
        Uses universal linguistic patterns, not domain-specific keywords
        """
        score = 0.0
        text_lower = text.lower()
        
        # Positive indicators (increase practicality score)
        
        # 1. Action/instruction patterns (universal)
        if self.structural_patterns['instructions'].search(text):
            score += 0.3
        if self.structural_patterns['actions'].search(text):
            score += 0.2
            
        # 2. Lists and structured content (usually practical)
        if self.structural_patterns['lists'].search(text):
            score += 0.25
            
        # 3. Contact information (always practical)
        if self.structural_patterns['contact_info'].search(text):
            score += 0.4
            
        # 4. Questions and answers (often practical)
        question_count = text.count('?')
        if question_count > 0:
            score += min(0.2, question_count * 0.1)
            
        # 5. Imperative mood indicators (commands/instructions)
        imperative_patterns = [
            r'\b\w+\s+(the|your|a|an)\s+\w+',  # "wash the container"
            r'^\s*\w+\s+\w+\s*[.!]',          # "Call now." 
            r'\bremember\b|\bnote\b|\bimportant\b'
        ]
        for pattern in imperative_patterns:
            if re.search(pattern, text_lower):
                score += 0.15
                
        # Negative indicators (decrease practicality score)
        
        # 1. Research/academic language
        if self.structural_patterns['research_terms'].search(text):
            score -= 0.3
            
        # 2. Heavy statistical/numerical content
        if self.structural_patterns['numbers_data'].search(text):
            score -= 0.2
            
        # 3. Citations and references
        if self.structural_patterns['citations'].search(text):
            score -= 0.25
            
        # 4. Very long sentences (often academic/technical)
        sentences = text.split('. ')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        if avg_sentence_length > 25:  # Very long sentences
            score -= 0.2
            
        # 5. Passive voice (often in academic/bureaucratic text)
        passive_indicators = ['was made', 'were taken', 'is determined', 'are shown']
        for indicator in passive_indicators:
            if indicator in text_lower:
                score -= 0.1
        
        # Normalize score to 0-1 range
        return max(0.0, min(1.0, score + 0.5))  # Base score of 0.5, then adjust

# Global improved chunker instance
improved_chunker: ImprovedChunker = ImprovedChunker()

def get_improved_chunker() -> ImprovedChunker:
    """Get global improved chunker instance"""
    return improved_chunker

def chunk_text_improved(text: str, document_name: str = "unknown") -> List[Dict[str, Any]]:
    """Convenience function for improved text chunking"""
    return improved_chunker.chunk_text(text, document_name)