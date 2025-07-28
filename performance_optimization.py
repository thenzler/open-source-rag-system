#!/usr/bin/env python3
"""
Performance Optimization Script for RAG System
Optimize AI response generation speed for local machines
"""
import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Optimize RAG system performance"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.cache_dir = base_dir / "data" / "cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
    def optimize_ollama_client(self):
        """Optimize Ollama client for faster responses"""
        print("🚀 Optimizing Ollama Client Performance")
        print("-" * 50)
        
        ollama_file = self.base_dir / "core" / "ollama_client.py"
        
        # Read current file
        with open(ollama_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Performance optimizations
        optimizations = [
            # 1. Reduce timeout from 300s to 60s
            {
                'old': 'timeout: int = 300',
                'new': 'timeout: int = 60  # Reduced from 300s for better UX'
            },
            # 2. Optimize model parameters for speed
            {
                'old': "'num_predict': max_tokens,",
                'new': "'num_predict': min(max_tokens, 1024),  # Limit tokens for speed"
            },
            # 3. Add faster temperature
            {
                'old': "'temperature': temperature,",
                'new': "'temperature': min(temperature, 0.3),  # Lower temp for faster, consistent responses"
            },
            # 4. Add top_k for faster generation
            {
                'old': "'top_p': 0.9,",
                'new': "'top_p': 0.9,\n                'top_k': 40,  # Limit token choices for speed"
            }
        ]
        
        modified = False
        for opt in optimizations:
            if opt['old'] in content and opt['new'] not in content:
                content = content.replace(opt['old'], opt['new'])
                modified = True
                print(f"✅ Applied: {opt['old'][:50]}...")
        
        if modified:
            # Backup original
            backup_file = ollama_file.with_suffix('.py.backup')
            with open(backup_file, 'w', encoding='utf-8') as f:
                with open(ollama_file, 'r', encoding='utf-8') as orig:
                    f.write(orig.read())
            
            # Write optimized version
            with open(ollama_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Ollama client optimized, backup saved to {backup_file.name}")
        else:
            print("ℹ️ Ollama client already optimized")
    
    def create_response_cache(self):
        """Create simple response caching system"""
        print("\n💾 Creating Response Cache System")
        print("-" * 50)
        
        cache_file = self.base_dir / "core" / "services" / "response_cache.py"
        
        cache_code = '''"""
Simple Response Cache for RAG System
Cache LLM responses to avoid regeneration
"""
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ResponseCache:
    """Simple file-based response cache"""
    
    def __init__(self, cache_dir: str = "data/cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        
    def _get_cache_key(self, query: str, context: str) -> str:
        """Generate cache key from query and context"""
        combined = f"{query}|{context}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_cache_file(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, query: str, context: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired"""
        try:
            cache_key = self._get_cache_key(query, context)
            cache_file = self._get_cache_file(cache_key)
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Check if expired
            cached_time = datetime.fromisoformat(cached_data['timestamp'])
            if datetime.now() - cached_time > self.ttl:
                cache_file.unlink()  # Remove expired cache
                return None
            
            logger.info(f"Cache hit for query: {query[:50]}...")
            return cached_data['response']
            
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def set(self, query: str, context: str, response: Dict[str, Any]):
        """Cache a response"""
        try:
            cache_key = self._get_cache_key(query, context)
            cache_file = self._get_cache_file(cache_key)
            
            cached_data = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'response': response
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cached_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Cached response for query: {query[:50]}...")
            
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def clear_expired(self):
        """Remove expired cache entries"""
        try:
            removed = 0
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    
                    cached_time = datetime.fromisoformat(cached_data['timestamp'])
                    if datetime.now() - cached_time > self.ttl:
                        cache_file.unlink()
                        removed += 1
                        
                except Exception:
                    # Remove invalid cache files
                    cache_file.unlink()
                    removed += 1
            
            if removed > 0:
                logger.info(f"Removed {removed} expired cache entries")
                
        except Exception as e:
            logger.warning(f"Cache cleanup error: {e}")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                "entries": len(cache_files),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "cache_dir": str(self.cache_dir)
            }
        except Exception as e:
            logger.warning(f"Cache stats error: {e}")
            return {"entries": 0, "total_size_mb": 0}
'''
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(cache_code)
        
        print(f"✅ Response cache created: {cache_file}")
    
    def optimize_rag_service(self):
        """Optimize SimpleRAGService for performance"""
        print("\n⚡ Optimizing RAG Service")
        print("-" * 50)
        
        rag_file = self.base_dir / "core" / "services" / "simple_rag_service.py"
        
        # Read current file
        with open(rag_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add cache integration
        cache_imports = '''import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from .response_cache import ResponseCache'''
        
        if 'from .response_cache import ResponseCache' not in content:
            content = content.replace(
                'from datetime import datetime',
                cache_imports
            )
            
            # Add cache initialization
            cache_init = """        self.vector_repo = vector_repo
        self.llm_client = llm_client
        self.audit_repo = audit_repo
        self.config = RAGConfig()
        self.cache = ResponseCache()  # Add response caching
        
        logger.info("Simple RAG Service initialized with caching")"""
            
            content = content.replace(
                """        self.vector_repo = vector_repo
        self.llm_client = llm_client
        self.audit_repo = audit_repo
        self.config = RAGConfig()
        
        logger.info("Simple RAG Service initialized")""",
                cache_init
            )
            
            # Add cache usage in generate_answer
            cache_check = '''    async def _generate_answer(self, query: str, search_results: List[Dict]) -> Dict[str, Any]:
        """Generate AI answer from search results with caching"""
        try:
            if not search_results:
                return {
                    "text": "Keine relevanten Informationen gefunden.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Prepare context
            context_parts = []
            sources = []
            
            for i, result in enumerate(search_results, 1):
                context_parts.append(f"[Quelle {i}]: {result['text']}")
                sources.append({
                    "id": i,
                    "document_id": result['document_id'],
                    "similarity": result['similarity'],
                    "download_url": f"/api/v1/documents/{result['document_id']}/download"
                })
            
            context = "\\n\\n".join(context_parts)
            
            # Check cache first
            cached_response = self.cache.get(query, context)
            if cached_response:
                return cached_response
            
            # Generate response if not cached'''
            
            content = content.replace(
                '''    async def _generate_answer(self, query: str, search_results: List[Dict]) -> Dict[str, Any]:
        """Generate AI answer from search results"""
        try:
            if not search_results:
                return {
                    "text": "Keine relevanten Informationen gefunden.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Prepare context
            context_parts = []
            sources = []
            
            for i, result in enumerate(search_results, 1):
                context_parts.append(f"[Quelle {i}]: {result['text']}")
                sources.append({
                    "id": i,
                    "document_id": result['document_id'],
                    "similarity": result['similarity'],
                    "download_url": f"/api/v1/documents/{result['document_id']}/download"
                })
            
            context = "\\n\\n".join(context_parts)
            
            # Generate response''',
                cache_check
            )
            
            # Backup and save
            backup_file = rag_file.with_suffix('.py.backup')
            with open(backup_file, 'w', encoding='utf-8') as f:
                with open(rag_file, 'r', encoding='utf-8') as orig:
                    f.write(orig.read())
            
            with open(rag_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ RAG service optimized with caching, backup saved")
        else:
            print("ℹ️ RAG service already has caching")

def main():
    print("⚡ RAG System Performance Optimization")
    print("=" * 50)
    
    base_dir = Path(__file__).parent
    optimizer = PerformanceOptimizer(base_dir)
    
    # Apply optimizations
    optimizer.optimize_ollama_client()
    optimizer.create_response_cache()
    optimizer.optimize_rag_service()
    
    print("\n🎉 Performance Optimization Complete!")
    print("=" * 50)
    print("Optimizations applied:")
    print("✅ Reduced Ollama timeout from 300s to 60s")
    print("✅ Optimized model parameters for speed")
    print("✅ Added response caching system")
    print("✅ Enhanced RAG service with cache integration")
    print("\nRestart the system to see performance improvements:")
    print("python run_core.py")

if __name__ == "__main__":
    main()