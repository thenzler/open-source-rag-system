"""
Simple Response Cache for RAG System
Cache LLM responses to avoid regeneration for similar queries
"""
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ResponseCache:
    """Simple file-based response cache for LLM responses"""
    
    def __init__(self, cache_dir: str = "data/cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        
    def _get_cache_key(self, query: str, context: str) -> str:
        """Generate cache key from query and context"""
        # Create a deterministic hash from query + context
        combined = f"{query}|{context[:1000]}"  # Limit context for performance
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
            
            logger.info(f"✅ Cache hit for query: {query[:50]}...")
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
                'context_hash': hashlib.md5(context.encode()).hexdigest(),
                'response': response
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cached_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Cached response for query: {query[:50]}...")
            
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
                logger.info(f"🧹 Removed {removed} expired cache entries")
                
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
    
    def clear_all(self):
        """Clear all cache entries"""
        try:
            removed = 0
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
                removed += 1
            
            logger.info(f"🗑️ Cleared {removed} cache entries")
            
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")