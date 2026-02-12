"""
Vector Database Service for RAG-Enhanced Eval Generation

This service manages:
- Embedding generation using sentence-transformers
- Storage and retrieval of eval prompts with ChromaDB
- Similarity search for finding relevant past evals
- Feedback loop integration
"""

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional, Any
import json
import uuid
from datetime import datetime, timezone
import logging
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

logger = logging.getLogger(__name__)

# Global thread pool for CPU-intensive embedding operations
# Using 2 workers to balance CPU usage with concurrency
_embedding_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedding")

class EvalVectorService:
    """Service for vector-based eval prompt storage and retrieval"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the vector service with embedding model and ChromaDB

        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        logger.info("Initializing EvalVectorService...")

        # Initialize embedding model (all-MiniLM-L6-v2: 384 dimensions, 80MB)
        logger.info("Loading sentence-transformers model: all-MiniLM-L6-v2")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully")

        # Initialize ChromaDB client
        logger.info(f"Initializing ChromaDB at: {persist_directory}")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection for eval prompts
        self.collection = self.client.get_or_create_collection(
            name="eval_prompts",
            metadata={
                "hnsw:space": "cosine",  # Use cosine similarity
                "description": "Storage for generated evaluation prompts with metadata"
            }
        )

        logger.info(f"ChromaDB collection initialized. Current count: {self.collection.count()}")

    def _create_embedding_sync(self, eval_context: Dict[str, Any]) -> List[float]:
        """
        Create embedding from eval context (synchronous version for thread pool)

        Combines multiple signals:
        - Dimension being evaluated
        - Use case description
        - Domain context (industry, products)
        - Quality score (for weighting during retrieval)

        Args:
            eval_context: Dictionary containing eval metadata

        Returns:
            List of floats representing the embedding (384 dimensions)
        """
        # Build rich context text for embedding
        context_parts = [
            f"Dimension: {eval_context.get('dimension', 'Unknown')}"
        ]

        if 'use_case' in eval_context:
            context_parts.append(f"Use Case: {eval_context['use_case']}")

        if 'system_prompt' in eval_context:
            # Include first 200 chars of system prompt for context
            context_parts.append(f"System Prompt: {eval_context['system_prompt'][:200]}")

        domain_context = eval_context.get('domain_context', {})
        if domain_context:
            if 'industry' in domain_context:
                industries = domain_context['industry']
                if isinstance(industries, list) and industries:
                    context_parts.append(f"Industry: {', '.join(industries[:3])}")

            if 'products' in domain_context:
                products = domain_context['products']
                if isinstance(products, list) and products:
                    context_parts.append(f"Products: {', '.join(products[:5])}")

        if 'quality_score' in eval_context:
            context_parts.append(f"Quality: {eval_context['quality_score']}/10")

        context_text = "\n".join(context_parts)

        # Generate embedding (CPU-intensive operation)
        embedding = self.model.encode(context_text, convert_to_tensor=False)

        return embedding.tolist()

    async def create_embedding(self, eval_context: Dict[str, Any]) -> List[float]:
        """
        Create embedding from eval context (async wrapper)

        Runs the CPU-intensive embedding generation in a thread pool to avoid
        blocking the async event loop.

        Args:
            eval_context: Dictionary containing eval metadata

        Returns:
            List of floats representing the embedding (384 dimensions)
        """
        loop = asyncio.get_event_loop()
        # Run CPU-intensive embedding in thread pool
        return await loop.run_in_executor(
            _embedding_executor,
            self._create_embedding_sync,
            eval_context
        )

    async def store_eval(
        self,
        eval_prompt: str,
        dimension: str,
        system_prompt: str,
        domain_context: Dict[str, Any],
        quality_score: float,
        meta_feedback: str,
        use_case: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Store eval prompt with embedding in vector database

        Args:
            eval_prompt: The generated evaluation prompt
            dimension: Evaluation dimension (e.g., "Discovery Quality")
            system_prompt: The system prompt being evaluated
            domain_context: Domain-specific context (products, industry, etc.)
            quality_score: Meta-evaluation quality score (0-10)
            meta_feedback: Feedback from meta-evaluation
            use_case: Optional use case description
            project_id: Optional project ID for tracking
            session_id: Optional session ID for tracking

        Returns:
            ID of stored eval entry
        """
        try:
            eval_id = f"eval_{uuid.uuid4()}"

            # Create embedding
            eval_context = {
                'dimension': dimension,
                'use_case': use_case or '',
                'system_prompt': system_prompt,
                'domain_context': domain_context,
                'quality_score': quality_score
            }
            embedding = await self.create_embedding(eval_context)

            # Prepare metadata
            metadata = {
                'eval_id': eval_id,
                'dimension': dimension,
                'quality_score': float(quality_score),
                'meta_feedback': meta_feedback[:500],  # Truncate to avoid size limits
                'use_case': use_case or '',
                'project_id': project_id or '',
                'session_id': session_id or '',
                'created_at': datetime.now(timezone.utc).isoformat(),
                'used_count': 0,
                'domain_context_json': json.dumps(domain_context)[:1000]  # Truncate
            }

            # Store in ChromaDB
            self.collection.add(
                embeddings=[embedding],
                documents=[eval_prompt],
                metadatas=[metadata],
                ids=[eval_id]
            )

            logger.info(f"Stored eval {eval_id} for dimension '{dimension}' with quality {quality_score}")
            return eval_id

        except Exception as e:
            logger.error(f"Error storing eval: {str(e)}", exc_info=True)
            raise

    async def search_similar_evals(
        self,
        dimension: str,
        system_prompt: str,
        domain_context: Dict[str, Any],
        use_case: Optional[str] = None,
        top_k: int = 5,
        min_quality: float = 8.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar high-quality eval prompts

        Args:
            dimension: Evaluation dimension to match
            system_prompt: System prompt for context
            domain_context: Domain context for similarity
            use_case: Optional use case description
            top_k: Number of results to return
            min_quality: Minimum quality score threshold

        Returns:
            List of similar eval prompts with metadata and similarity scores
        """
        try:
            # Check if collection has any data
            if self.collection.count() == 0:
                logger.info("No stored evals yet, returning empty list")
                return []

            # Create query embedding
            query_context = {
                'dimension': dimension,
                'use_case': use_case or '',
                'system_prompt': system_prompt,
                'domain_context': domain_context,
                'quality_score': min_quality  # Bias towards high quality
            }
            query_embedding = await self.create_embedding(query_context)

            # Search with filters (ChromaDB requires $and for multiple conditions)
            where_filter = {
                "$and": [
                    {"dimension": dimension},
                    {"quality_score": {"$gte": min_quality}}
                ]
            }

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            similar_evals = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, meta, dist) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    similarity = 1.0 - dist  # Convert distance to similarity

                    similar_evals.append({
                        'eval_id': meta.get('eval_id', ''),
                        'eval_prompt': doc,
                        'quality_score': meta.get('quality_score', 0),
                        'meta_feedback': meta.get('meta_feedback', ''),
                        'similarity': float(similarity),
                        'use_case': meta.get('use_case', ''),
                        'created_at': meta.get('created_at', ''),
                        'used_count': meta.get('used_count', 0)
                    })

            logger.info(f"Found {len(similar_evals)} similar evals for dimension '{dimension}'")
            return similar_evals

        except Exception as e:
            logger.error(f"Error searching similar evals: {str(e)}", exc_info=True)
            return []  # Return empty list on error, don't fail generation

    async def update_usage_count(self, eval_id: str):
        """
        Increment usage count for an eval prompt

        Args:
            eval_id: ID of the eval to update
        """
        try:
            # Get existing entry
            result = self.collection.get(
                ids=[eval_id],
                include=["metadatas"]
            )

            if result['metadatas'] and result['metadatas'][0]:
                metadata = result['metadatas'][0]
                metadata['used_count'] = metadata.get('used_count', 0) + 1

                # Update in database
                self.collection.update(
                    ids=[eval_id],
                    metadatas=[metadata]
                )

                logger.info(f"Updated usage count for eval {eval_id}")

        except Exception as e:
            logger.error(f"Error updating usage count: {str(e)}", exc_info=True)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored evals

        Returns:
            Dictionary with stats (total count, dimensions, avg quality, etc.)
        """
        try:
            total_count = self.collection.count()

            if total_count == 0:
                return {
                    'total_evals': 0,
                    'dimensions': [],
                    'avg_quality': 0,
                    'high_quality_count': 0,
                    'high_quality_percentage': 0
                }

            # Get all metadata
            results = self.collection.get(
                limit=1000,  # Limit for performance
                include=["metadatas"]
            )

            dimensions = set()
            quality_scores = []
            high_quality_count = 0

            for meta in results['metadatas']:
                if 'dimension' in meta:
                    dimensions.add(meta['dimension'])
                if 'quality_score' in meta:
                    score = meta['quality_score']
                    quality_scores.append(score)
                    if score >= 8.0:
                        high_quality_count += 1

            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

            return {
                'total_evals': total_count,
                'dimensions': list(dimensions),
                'avg_quality': round(avg_quality, 2),
                'high_quality_count': high_quality_count,
                'high_quality_percentage': round((high_quality_count / total_count * 100), 1) if total_count > 0 else 0
            }

        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}", exc_info=True)
            return {'error': str(e)}

# Global instance (singleton pattern)
_vector_service: Optional[EvalVectorService] = None

def get_vector_service() -> EvalVectorService:
    """Get or create global vector service instance"""
    global _vector_service
    if _vector_service is None:
        persist_dir = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
        _vector_service = EvalVectorService(persist_directory=persist_dir)
    return _vector_service
