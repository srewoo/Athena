"""
Dimension Pattern Service

Stores and retrieves expert dimension design patterns from ChromaDB.
Enables dimension generation to learn from proven patterns extracted
from production systems (PAM, BI, etc.).
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Any
import json
import uuid
import logging
from datetime import datetime, timezone
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Global thread pool for CPU-intensive embedding operations
_pattern_embedding_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pattern_embedding")


class DimensionPatternService:
    """
    Manages dimension design pattern storage and retrieval using vector search.

    Stores expert knowledge about dimension design (architecture patterns,
    design principles, scoring patterns, failure modes) and retrieves
    relevant patterns during dimension generation.
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the dimension pattern service.

        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        logger.info("Initializing DimensionPatternService...")

        # Initialize embedding model (shared with other services)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model loaded")

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection for dimension patterns
        self.collection = self.client.get_or_create_collection(
            name="dimension_design_patterns",
            metadata={
                "hnsw:space": "cosine",
                "description": "Expert dimension design patterns and intent"
            }
        )

        logger.info(f"DimensionPatternService initialized. Collection size: {self.collection.count()}")

    def _create_embedding_sync(self, text: str) -> List[float]:
        """Create embedding for text (sync version for thread pool)."""
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

    async def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _pattern_embedding_executor,
            self._create_embedding_sync,
            text
        )

    async def store_pattern(
        self,
        category: str,
        pattern_data: Dict[str, Any],
        pattern_id: Optional[str] = None
    ) -> str:
        """
        Store a dimension design pattern as a searchable chunk.

        Args:
            category: Pattern category (architecture, principle, scoring, failure_mode, etc.)
            pattern_data: Pattern information (name, description, examples, etc.)
            pattern_id: Optional custom ID, auto-generated if not provided

        Returns:
            Pattern ID
        """
        if not pattern_id:
            pattern_id = f"{category}_{uuid.uuid4()}"

        # Build searchable text from pattern data
        text_parts = [f"Category: {category}"]

        for key, value in pattern_data.items():
            if isinstance(value, list):
                text_parts.append(f"{key}: {', '.join(str(v) for v in value)}")
            elif isinstance(value, dict):
                text_parts.append(f"{key}: {json.dumps(value)}")
            else:
                text_parts.append(f"{key}: {value}")

        searchable_text = "\n".join(text_parts)

        # Create embedding
        embedding = await self.create_embedding(searchable_text)

        # Store in ChromaDB
        self.collection.add(
            ids=[pattern_id],
            embeddings=[embedding],
            documents=[searchable_text],
            metadatas=[{
                "category": category,
                "pattern_data": json.dumps(pattern_data)[:5000],  # Limit size
                "created_at": datetime.now(timezone.utc).isoformat()
            }]
        )

        logger.info(f"[Pattern Service] Stored pattern: {pattern_id} (category: {category})")
        return pattern_id

    async def retrieve_relevant_patterns(
        self,
        system_prompt: str,
        prompt_characteristics: List[str],
        top_k: int = 5
    ) -> str:
        """
        Retrieve relevant dimension design patterns based on prompt.

        Args:
            system_prompt: The system prompt being analyzed
            prompt_characteristics: List of characteristics (e.g., ["structured_output", "recommendation"])
            top_k: Number of patterns to retrieve

        Returns:
            Formatted string to inject into dimension generation prompt
        """
        try:
            # Build query combining prompt excerpt and characteristics
            query_parts = [
                f"System prompt characteristics: {', '.join(prompt_characteristics)}",
                f"Prompt excerpt: {system_prompt[:500]}"
            ]
            query_text = "\n".join(query_parts)

            # Create query embedding
            query_embedding = await self.create_embedding(query_text)

            # Search for similar patterns
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count())
            )

            if not results['documents'] or not results['documents'][0]:
                logger.info("[Pattern Service] No patterns found")
                return ""

            # Format patterns for injection
            formatted_patterns = []
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                category = metadata.get('category', 'unknown')
                pattern_data = json.loads(metadata.get('pattern_data', '{}'))

                # Extract key fields
                pattern_name = pattern_data.get('pattern_name') or pattern_data.get('principle') or pattern_data.get('pattern') or category
                description = pattern_data.get('description', '')
                when_to_use = pattern_data.get('when_to_use', '')
                benefit = pattern_data.get('benefit', '')
                example = pattern_data.get('example', '')

                pattern_text = f"Pattern {i+1}: {pattern_name}\n"
                pattern_text += f"  Description: {description}\n"
                if when_to_use:
                    pattern_text += f"  When to use: {when_to_use}\n"
                if benefit:
                    pattern_text += f"  Benefit: {benefit}\n"
                if example:
                    pattern_text += f"  Example: {example}\n"

                formatted_patterns.append(pattern_text)

            guidance = "\n".join(formatted_patterns)
            logger.info(f"[Pattern Service] Retrieved {len(formatted_patterns)} relevant patterns")
            return guidance

        except Exception as e:
            logger.error(f"[Pattern Service] Error retrieving patterns: {e}")
            return ""

    async def analyze_prompt_characteristics(
        self,
        system_prompt: str
    ) -> List[str]:
        """
        Analyze system prompt to extract characteristics.

        Returns:
            List of characteristics (e.g., ["structured_output", "recommendation", "has_constraints"])
        """
        characteristics = []
        prompt_lower = system_prompt.lower()

        # Check for structured output
        if any(keyword in prompt_lower for keyword in ['json', 'schema', 'format', 'structure', 'output format']):
            characteristics.append('structured_output')

        # Check for recommendation/classification
        if any(keyword in prompt_lower for keyword in ['recommend', 'classify', 'categorize', 'framework', 'map to']):
            characteristics.append('recommendation_system')

        # Check for diagnostic/analysis
        if any(keyword in prompt_lower for keyword in ['diagnos', 'analyz', 'assess', 'evaluat', 'identify']):
            characteristics.append('diagnostic_analysis')

        # Check for hard constraints
        if any(keyword in prompt_lower for keyword in ['must', 'required', 'constraint', 'validation', 'exactly']):
            characteristics.append('has_constraints')

        # Check for style/template requirements
        if any(keyword in prompt_lower for keyword in ['style', 'template', 'format', 'voice', 'tone']):
            characteristics.append('style_requirements')

        # Check for evidence/grounding
        if any(keyword in prompt_lower for keyword in ['evidence', 'citation', 'reference', 'ground', 'support']):
            characteristics.append('evidence_based')

        # Check for scoring/rating
        if any(keyword in prompt_lower for keyword in ['score', 'rating', 'grade', 'rank']):
            characteristics.append('scoring_system')

        # Check for multi-step/complex logic
        if any(keyword in prompt_lower for keyword in ['step 1', 'first', 'then', 'next', 'finally', 'algorithm']):
            characteristics.append('multi_step_logic')

        logger.info(f"[Pattern Service] Detected characteristics: {characteristics}")
        return characteristics if characteristics else ['general']

    async def get_pattern_by_category(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve all patterns of a specific category.

        Args:
            category: Pattern category to filter by
            limit: Maximum number of patterns to return

        Returns:
            List of pattern dictionaries
        """
        try:
            results = self.collection.get(
                where={"category": category},
                limit=limit
            )

            patterns = []
            if results['metadatas']:
                for metadata in results['metadatas']:
                    pattern_data = json.loads(metadata.get('pattern_data', '{}'))
                    pattern_data['category'] = metadata.get('category')
                    pattern_data['created_at'] = metadata.get('created_at')
                    patterns.append(pattern_data)

            return patterns

        except Exception as e:
            logger.error(f"[Pattern Service] Error getting patterns by category: {e}")
            return []

    def clear_patterns(self):
        """Clear all patterns from the collection (use with caution)."""
        try:
            self.collection.delete(where={})
            logger.info("[Pattern Service] All patterns cleared")
        except Exception as e:
            logger.error(f"[Pattern Service] Error clearing patterns: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored patterns."""
        try:
            total_count = self.collection.count()

            # Get all metadatas to count by category
            results = self.collection.get()
            categories = {}
            if results['metadatas']:
                for metadata in results['metadatas']:
                    cat = metadata.get('category', 'unknown')
                    categories[cat] = categories.get(cat, 0) + 1

            return {
                "total_patterns": total_count,
                "patterns_by_category": categories
            }
        except Exception as e:
            logger.error(f"[Pattern Service] Error getting stats: {e}")
            return {"total_patterns": 0, "patterns_by_category": {}}


# Global service instance
_dimension_pattern_service = None


def get_dimension_pattern_service() -> DimensionPatternService:
    """Get or create the global DimensionPatternService instance."""
    global _dimension_pattern_service
    if _dimension_pattern_service is None:
        _dimension_pattern_service = DimensionPatternService()
    return _dimension_pattern_service
