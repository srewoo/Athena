"""
Redis Cache Manager for Athena
Implements caching layer for prompt evaluations to reduce API costs and improve response times
"""

import redis
import json
import hashlib
import os
from typing import Optional, Dict, Any
import logging
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheManager:
    """Manages Redis caching for prompt evaluations"""
    
    def __init__(self, redis_url: Optional[str] = None, enabled: bool = True):
        """
        Initialize Redis cache manager
        
        Args:
            redis_url: Redis connection URL (default: redis://localhost:6379)
            enabled: Enable/disable caching (useful for testing)
        """
        self.enabled = enabled
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        
        if self.enabled:
            try:
                self.client = redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # Test connection
                self.client.ping()
                logger.info(f"✅ Redis cache connected: {self.redis_url}")
            except redis.ConnectionError as e:
                logger.warning(f"⚠️  Redis connection failed: {e}. Caching disabled.")
                self.enabled = False
                self.client = None
            except Exception as e:
                logger.warning(f"⚠️  Redis initialization error: {e}. Caching disabled.")
                self.enabled = False
                self.client = None
        else:
            self.client = None
            logger.info("ℹ️  Redis caching disabled")
    
    def _generate_cache_key(self, prompt_text: str, evaluation_mode: str, provider: str, model: str) -> str:
        """
        Generate unique cache key for an evaluation request
        
        Args:
            prompt_text: The prompt to evaluate
            evaluation_mode: Mode of evaluation (quick, standard, deep, etc.)
            provider: LLM provider (openai, claude, gemini)
            model: Model name
            
        Returns:
            MD5 hash as cache key
        """
        cache_string = f"{prompt_text}|{evaluation_mode}|{provider}|{model}"
        return f"eval:{hashlib.md5(cache_string.encode()).hexdigest()}"
    
    def get_evaluation(
        self,
        prompt_text: str,
        evaluation_mode: str,
        provider: str,
        model: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached evaluation result
        
        Returns:
            Cached evaluation data or None if not found
        """
        if not self.enabled or not self.client:
            return None
        
        try:
            cache_key = self._generate_cache_key(prompt_text, evaluation_mode, provider, model)
            cached_data = self.client.get(cache_key)
            
            if cached_data:
                logger.info(f"✅ Cache HIT: {cache_key[:16]}...")
                return json.loads(cached_data)
            else:
                logger.info(f"❌ Cache MISS: {cache_key[:16]}...")
                return None
                
        except Exception as e:
            logger.error(f"Cache GET error: {e}")
            return None
    
    def set_evaluation(
        self,
        prompt_text: str,
        evaluation_mode: str,
        provider: str,
        model: str,
        evaluation_data: Dict[str, Any],
        ttl_hours: int = 24
    ) -> bool:
        """
        Cache an evaluation result
        
        Args:
            ttl_hours: Time-to-live in hours (default: 24)
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.enabled or not self.client:
            return False
        
        try:
            cache_key = self._generate_cache_key(prompt_text, evaluation_mode, provider, model)
            ttl = timedelta(hours=ttl_hours)
            
            self.client.setex(
                cache_key,
                ttl,
                json.dumps(evaluation_data)
            )
            
            logger.info(f"✅ Cached evaluation: {cache_key[:16]}... (TTL: {ttl_hours}h)")
            return True
            
        except Exception as e:
            logger.error(f"Cache SET error: {e}")
            return False
    
    def invalidate_evaluation(self, prompt_text: str, evaluation_mode: str, provider: str, model: str) -> bool:
        """
        Invalidate (delete) a cached evaluation
        
        Returns:
            True if deleted successfully, False otherwise
        """
        if not self.enabled or not self.client:
            return False
        
        try:
            cache_key = self._generate_cache_key(prompt_text, evaluation_mode, provider, model)
            deleted = self.client.delete(cache_key)
            
            if deleted:
                logger.info(f"✅ Invalidated cache: {cache_key[:16]}...")
            
            return bool(deleted)
            
        except Exception as e:
            logger.error(f"Cache DELETE error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        if not self.enabled or not self.client:
            return {"enabled": False}
        
        try:
            info = self.client.info("stats")
            return {
                "enabled": True,
                "connected": True,
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                )
            }
        except Exception as e:
            logger.error(f"Cache STATS error: {e}")
            return {"enabled": True, "connected": False, "error": str(e)}
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate percentage"""
        total = hits + misses
        if total == 0:
            return 0.0
        return round((hits / total) * 100, 2)
    
    def clear_all(self) -> bool:
        """
        Clear all cached evaluations (use with caution!)
        
        Returns:
            True if cleared successfully
        """
        if not self.enabled or not self.client:
            return False
        
        try:
            # Only delete keys with 'eval:' prefix
            keys = self.client.keys("eval:*")
            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"✅ Cleared {deleted} cached evaluations")
                return True
            return True
            
        except Exception as e:
            logger.error(f"Cache CLEAR error: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Redis connection
        
        Returns:
            Health status dictionary
        """
        if not self.enabled:
            return {"status": "disabled", "healthy": True}
        
        try:
            if self.client:
                self.client.ping()
                return {"status": "connected", "healthy": True}
            else:
                return {"status": "not_initialized", "healthy": False}
        except Exception as e:
            return {"status": "error", "healthy": False, "error": str(e)}


# Global cache instance
cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """
    Get or create global cache manager instance
    
    Returns:
        CacheManager instance
    """
    global cache_manager
    if cache_manager is None:
        cache_manager = CacheManager()
    return cache_manager


def init_cache(redis_url: Optional[str] = None, enabled: bool = True) -> CacheManager:
    """
    Initialize cache manager with custom settings
    
    Args:
        redis_url: Custom Redis URL
        enabled: Enable/disable caching
        
    Returns:
        CacheManager instance
    """
    global cache_manager
    cache_manager = CacheManager(redis_url=redis_url, enabled=enabled)
    return cache_manager
