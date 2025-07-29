#!/usr/bin/env python3
"""
Municipal RAG System
Specialized RAG implementation for Swiss municipalities with weighted importance
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MunicipalChunk:
    """Enhanced chunk with municipal-specific metadata"""
    content: str
    embedding: np.ndarray
    source_url: str
    title: str
    category: str
    importance_score: float
    language: str
    document_type: str
    chunk_id: str
    last_updated: str

class MunicipalRAG:
    """RAG system optimized for municipal information"""
    
    def __init__(self, municipality_name: str, embedding_model, ollama_client):
        self.municipality_name = municipality_name
        self.embedding_model = embedding_model
        self.ollama_client = ollama_client
        self.municipal_chunks: List[MunicipalChunk] = []
        self.db_path = f"municipal_rag_{municipality_name.lower()}.db"
        self._init_database()
        
        # Municipal-specific prompt templates
        self.municipal_prompts = {
            'services': """Du bist ein Experte für die Dienstleistungen der Gemeinde {municipality}.
Beantworte Fragen über Gemeindeverwaltung, Formulare, Öffnungszeiten und Dienstleistungen.

Kontext aus der Gemeinde {municipality}:
{context}

Frage: {query}

Antwort basierend auf offiziellen Informationen der Gemeinde {municipality}:""",
            
            'general': """Du bist ein Assistent für die Gemeinde {municipality} in der Schweiz.
Du hilfst Bürgern mit Fragen zur Gemeindeverwaltung, Dienstleistungen und lokalen Informationen.

Offizielle Informationen der Gemeinde {municipality}:
{context}

Frage: {query}

Antwort (basierend auf offiziellen Quellen der Gemeinde {municipality}):""",
            
            'events': """Du bist ein Experte für Veranstaltungen und Events in der Gemeinde {municipality}.
Beantworte Fragen über lokale Veranstaltungen, Termine und kulturelle Aktivitäten.

Aktuelle Informationen aus {municipality}:
{context}

Frage: {query}

Antwort mit Details zu Veranstaltungen in {municipality}:""",
            
            'politics': """Du bist ein Experte für das politische System der Gemeinde {municipality}.
Beantworte Fragen über Gemeinderat, Abstimmungen, Wahlen und politische Prozesse.

Politische Informationen aus {municipality}:
{context}

Frage: {query}

Sachliche Antwort basierend auf offiziellen Informationen der Gemeinde {municipality}:"""
        }
    
    def _init_database(self):
        """Initialize SQLite database for municipal data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS municipal_chunks (
                chunk_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                source_url TEXT NOT NULL,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                importance_score REAL NOT NULL,
                language TEXT NOT NULL,
                document_type TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_category ON municipal_chunks(category);
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_importance ON municipal_chunks(importance_score);
        ''')
        
        conn.commit()
        conn.close()
    
    def load_municipal_data(self, data_path: str):
        """Load municipal data from scraper output"""
        data_file = Path(data_path) / self.municipality_name / "documents.json"
        
        if not data_file.exists():
            logger.error(f"Municipal data file not found: {data_file}")
            return
        
        with open(data_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        logger.info(f"Loading {len(documents)} documents for {self.municipality_name}")
        
        for doc in documents:
            # Chunk the document
            chunks = self._chunk_document(doc)
            
            # Create embeddings and store
            for chunk in chunks:
                self._add_chunk_to_database(chunk)
        
        logger.info(f"Loaded {len(self.municipal_chunks)} chunks for {self.municipality_name}")
    
    def _chunk_document(self, document: Dict[str, Any]) -> List[MunicipalChunk]:
        """Chunk a document with municipal-specific logic"""
        content = document['content']
        
        # Smart chunking based on document type
        if document['document_type'] == 'webpage':
            # Split by sections for web pages
            chunks = self._split_by_sections(content)
        else:
            # Standard chunking for other documents
            chunks = self._split_by_sentences(content)
        
        municipal_chunks = []
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < 100:  # Skip very short chunks
                continue
            
            # Create embedding
            embedding = self.embedding_model.encode(chunk_text)
            
            # Create municipal chunk
            chunk = MunicipalChunk(
                content=chunk_text,
                embedding=embedding,
                source_url=document['url'],
                title=document['title'],
                category=document['category'],
                importance_score=document['importance_score'],
                language=document['language'],
                document_type=document['document_type'],
                chunk_id=f"{document['url']}_{i}",
                last_updated=document['last_updated']
            )
            
            municipal_chunks.append(chunk)
        
        return municipal_chunks
    
    def _split_by_sections(self, content: str) -> List[str]:
        """Split content by sections (for web pages)"""
        # Split by common section indicators
        section_markers = [
            '\n\n\n',  # Triple newline
            '\n\n',    # Double newline
            '. ',      # Sentence end
        ]
        
        chunks = [content]
        for marker in section_markers:
            new_chunks = []
            for chunk in chunks:
                if len(chunk) > 1000:  # Only split large chunks
                    new_chunks.extend(chunk.split(marker))
                else:
                    new_chunks.append(chunk)
            chunks = new_chunks
        
        return [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]
    
    def _split_by_sentences(self, content: str) -> List[str]:
        """Split content by sentences with overlap"""
        sentences = content.split('. ')
        chunks = []
        
        chunk_size = 3  # 3 sentences per chunk
        overlap = 1     # 1 sentence overlap
        
        for i in range(0, len(sentences), chunk_size - overlap):
            chunk = '. '.join(sentences[i:i + chunk_size])
            if len(chunk.strip()) > 50:
                chunks.append(chunk)
        
        return chunks
    
    def _add_chunk_to_database(self, chunk: MunicipalChunk):
        """Add chunk to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize embedding
        embedding_bytes = chunk.embedding.tobytes()
        
        cursor.execute('''
            INSERT OR REPLACE INTO municipal_chunks 
            (chunk_id, content, embedding, source_url, title, category, 
             importance_score, language, document_type, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            chunk.chunk_id,
            chunk.content,
            embedding_bytes,
            chunk.source_url,
            chunk.title,
            chunk.category,
            chunk.importance_score,
            chunk.language,
            chunk.document_type,
            chunk.last_updated
        ))
        
        conn.commit()
        conn.close()
        
        # Also keep in memory for fast access
        self.municipal_chunks.append(chunk)
    
    def municipal_search(self, query: str, top_k: int = 5, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search with municipal-specific weighting"""
        # Create query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Load chunks from database if not in memory
        if not self.municipal_chunks:
            self._load_chunks_from_database()
        
        # Filter by category if specified
        relevant_chunks = self.municipal_chunks
        if category_filter:
            relevant_chunks = [chunk for chunk in relevant_chunks if chunk.category == category_filter]
        
        # Calculate weighted similarity scores
        scored_chunks = []
        for chunk in relevant_chunks:
            # Basic cosine similarity
            similarity = np.dot(query_embedding, chunk.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding)
            )
            
            # Apply importance weighting
            weighted_score = similarity * (0.7 + 0.3 * chunk.importance_score)
            
            # Boost for exact matches in content
            if any(word.lower() in chunk.content.lower() for word in query.split()):
                weighted_score *= 1.1
            
            scored_chunks.append({
                'chunk': chunk,
                'similarity': similarity,
                'weighted_score': weighted_score
            })
        
        # Sort by weighted score
        scored_chunks.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        # Return top results
        results = []
        for scored_chunk in scored_chunks[:top_k]:
            chunk = scored_chunk['chunk']
            results.append({
                'content': chunk.content,
                'source_url': chunk.source_url,
                'title': chunk.title,
                'category': chunk.category,
                'importance_score': chunk.importance_score,
                'similarity': scored_chunk['similarity'],
                'weighted_score': scored_chunk['weighted_score']
            })
        
        return results
    
    def _load_chunks_from_database(self):
        """Load chunks from database into memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM municipal_chunks')
        rows = cursor.fetchall()
        
        for row in rows:
            # Deserialize embedding
            embedding = np.frombuffer(row[2], dtype=np.float32)
            
            chunk = MunicipalChunk(
                content=row[1],
                embedding=embedding,
                source_url=row[3],
                title=row[4],
                category=row[5],
                importance_score=row[6],
                language=row[7],
                document_type=row[8],
                chunk_id=row[0],
                last_updated=row[9]
            )
            
            self.municipal_chunks.append(chunk)
        
        conn.close()
    
    def generate_municipal_answer(self, query: str, category_hint: Optional[str] = None) -> Dict[str, Any]:
        """Generate answer using municipal-specific context"""
        # Search for relevant chunks
        results = self.municipal_search(query, top_k=3, category_filter=category_hint)
        
        if not results:
            return {
                'answer': f"Entschuldigung, ich konnte keine Informationen zu Ihrer Frage in den offiziellen Dokumenten der Gemeinde {self.municipality_name} finden.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Determine category for prompt selection
        primary_category = results[0]['category'] if results else 'general'
        
        # Create context from results
        context_parts = []
        for result in results:
            context_parts.append(f"[{result['title']}] {result['content']}")
        
        context = '\n\n'.join(context_parts)
        
        # Select appropriate prompt template
        prompt_template = self.municipal_prompts.get(primary_category, self.municipal_prompts['general'])
        
        # Generate prompt
        prompt = prompt_template.format(
            municipality=self.municipality_name,
            context=context,
            query=query
        )
        
        # Generate answer using Ollama
        try:
            answer = self.ollama_client.generate_answer(
                query=query,
                context=context,
                max_tokens=2048,
                temperature=0.3  # Lower temperature for factual municipal information
            )
            
            if not answer:
                answer = f"Basierend auf den verfügbaren Informationen der Gemeinde {self.municipality_name}:\n\n{context[:500]}..."
            
            # Calculate confidence based on relevance scores
            avg_confidence = sum(r['weighted_score'] for r in results) / len(results)
            
            return {
                'answer': answer,
                'sources': [{'title': r['title'], 'url': r['source_url'], 'category': r['category']} for r in results],
                'confidence': min(avg_confidence, 1.0),
                'municipality': self.municipality_name
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': f"Entschuldigung, es gab einen Fehler bei der Antwortgenerierung. Hier sind die relevanten Informationen der Gemeinde {self.municipality_name}:\n\n{context[:500]}...",
                'sources': [{'title': r['title'], 'url': r['source_url'], 'category': r['category']} for r in results],
                'confidence': 0.5,
                'municipality': self.municipality_name
            }
    
    def get_municipal_categories(self) -> Dict[str, int]:
        """Get available categories and their document counts"""
        if not self.municipal_chunks:
            self._load_chunks_from_database()
        
        categories = {}
        for chunk in self.municipal_chunks:
            categories[chunk.category] = categories.get(chunk.category, 0) + 1
        
        return categories
    
    def get_municipal_stats(self) -> Dict[str, Any]:
        """Get statistics about municipal knowledge base"""
        if not self.municipal_chunks:
            self._load_chunks_from_database()
        
        total_chunks = len(self.municipal_chunks)
        categories = self.get_municipal_categories()
        
        # Calculate average importance score
        avg_importance = sum(chunk.importance_score for chunk in self.municipal_chunks) / total_chunks if total_chunks > 0 else 0
        
        # High importance chunks
        high_importance = len([chunk for chunk in self.municipal_chunks if chunk.importance_score > 0.7])
        
        return {
            'municipality': self.municipality_name,
            'total_chunks': total_chunks,
            'categories': categories,
            'average_importance': avg_importance,
            'high_importance_chunks': high_importance,
            'languages': list(set(chunk.language for chunk in self.municipal_chunks))
        }

# Example usage function
def create_arlesheim_rag(embedding_model, ollama_client):
    """Create RAG system for Arlesheim municipality"""
    municipal_rag = MunicipalRAG("Arlesheim", embedding_model, ollama_client)
    municipal_rag.load_municipal_data("data/municipal_data")
    return municipal_rag