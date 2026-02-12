"""
Domain Context RAG Service

Stores domain context in ChromaDB and retrieves relevant chunks
based on semantic similarity to dimension + system prompt.

This replaces the inefficient "dump all context" approach with
smart retrieval of only relevant context pieces.
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
import os

logger = logging.getLogger(__name__)

# Global thread pool for CPU-intensive embedding operations
_domain_embedding_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="domain_embedding")


class DomainContextService:
    """
    Manages domain context storage and retrieval using vector search.

    Instead of injecting all domain context into every eval generation,
    this service uses semantic search to retrieve only the most relevant
    context pieces for a given dimension and system prompt.
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the domain context service.

        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        logger.info("Initializing DomainContextService...")

        # Initialize embedding model (shared with eval vector service)
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

        # Get or create collection for domain context
        self.collection = self.client.get_or_create_collection(
            name="domain_context",
            metadata={
                "hnsw:space": "cosine",
                "description": "Domain-specific context chunks for selective injection"
            }
        )

        logger.info(f"DomainContextService initialized. Collection size: {self.collection.count()}")

    def _create_embedding_sync(self, text: str) -> List[float]:
        """Create embedding for text (sync version for thread pool)."""
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

    async def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _domain_embedding_executor,
            self._create_embedding_sync,
            text
        )

    async def store_domain_context(
        self,
        domain_context: Dict[str, Any],
        session_id: str,
        project_id: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Store domain context in ChromaDB as searchable chunks.

        Breaks down domain context into semantic chunks that can be
        retrieved independently based on relevance.

        Args:
            domain_context: Full domain context dictionary
            session_id: User session ID
            project_id: Optional project ID

        Returns:
            Dictionary with chunk counts per category
        """
        try:
            logger.info(f"[Domain RAG] Storing domain context for session {session_id}")

            chunks = []
            chunk_counts = {}

            # 1. PRODUCTS/SERVICES - Store each product separately
            products = domain_context.get('products', [])
            if products:
                for product in products[:10]:  # Limit to top 10
                    chunk_id = f"product_{session_id}_{uuid.uuid4()}"
                    chunks.append({
                        'id': chunk_id,
                        'text': f"Product/Service: {product}",
                        'category': 'product',
                        'metadata': {
                            'session_id': session_id,
                            'project_id': project_id or '',
                            'category': 'product',
                            'item': product,
                            'created_at': datetime.now(timezone.utc).isoformat()
                        }
                    })
                chunk_counts['products'] = len(chunks)

            # 2. PROSPECTS/CUSTOMERS - Store each prospect separately
            prospects = domain_context.get('prospects', [])
            if prospects:
                for prospect in prospects[:10]:
                    chunk_id = f"prospect_{session_id}_{uuid.uuid4()}"
                    chunks.append({
                        'id': chunk_id,
                        'text': f"Key Prospect/Customer: {prospect}",
                        'category': 'prospect',
                        'metadata': {
                            'session_id': session_id,
                            'project_id': project_id or '',
                            'category': 'prospect',
                            'item': prospect,
                            'created_at': datetime.now(timezone.utc).isoformat()
                        }
                    })
                chunk_counts['prospects'] = len(chunks) - chunk_counts.get('products', 0)

            # 3. DOMAIN TERMINOLOGY - Store each term separately
            terminology = domain_context.get('domain_terminology', [])
            if terminology:
                for term in terminology[:20]:
                    chunk_id = f"term_{session_id}_{uuid.uuid4()}"
                    chunks.append({
                        'id': chunk_id,
                        'text': f"Domain Term: {term}",
                        'category': 'terminology',
                        'metadata': {
                            'session_id': session_id,
                            'project_id': project_id or '',
                            'category': 'terminology',
                            'item': term,
                            'created_at': datetime.now(timezone.utc).isoformat()
                        }
                    })
                chunk_counts['terminology'] = len(chunks) - sum(chunk_counts.values())

            # 4. QUALITY PRINCIPLES - Store each principle separately
            quality_principles = domain_context.get('quality_principles', [])
            if quality_principles:
                for i, principle in enumerate(quality_principles[:15]):
                    chunk_id = f"quality_{session_id}_{uuid.uuid4()}"
                    chunks.append({
                        'id': chunk_id,
                        'text': f"Quality Principle: {principle}",
                        'category': 'quality_principle',
                        'metadata': {
                            'session_id': session_id,
                            'project_id': project_id or '',
                            'category': 'quality_principle',
                            'priority': i,
                            'item': principle,
                            'created_at': datetime.now(timezone.utc).isoformat()
                        }
                    })
                chunk_counts['quality_principles'] = len(chunks) - sum(chunk_counts.values())

            # 5. ANTI-PATTERNS - Store each anti-pattern separately
            anti_patterns = domain_context.get('anti_patterns', [])
            if anti_patterns:
                for pattern in anti_patterns[:15]:
                    chunk_id = f"antipattern_{session_id}_{uuid.uuid4()}"
                    chunks.append({
                        'id': chunk_id,
                        'text': f"Anti-Pattern to Avoid: {pattern}",
                        'category': 'anti_pattern',
                        'metadata': {
                            'session_id': session_id,
                            'project_id': project_id or '',
                            'category': 'anti_pattern',
                            'item': pattern,
                            'created_at': datetime.now(timezone.utc).isoformat()
                        }
                    })
                chunk_counts['anti_patterns'] = len(chunks) - sum(chunk_counts.values())

            # 6. FAILURE MODES - Store each failure mode separately
            failure_modes = domain_context.get('failure_modes', [])
            if failure_modes:
                for fm in failure_modes[:10]:
                    chunk_id = f"failure_{session_id}_{uuid.uuid4()}"
                    text = f"Failure Mode: {fm.get('id', 'unknown')} - {fm.get('description', '')} (Severity: {fm.get('severity', 'medium')})"
                    chunks.append({
                        'id': chunk_id,
                        'text': text,
                        'category': 'failure_mode',
                        'metadata': {
                            'session_id': session_id,
                            'project_id': project_id or '',
                            'category': 'failure_mode',
                            'severity': fm.get('severity', 'medium'),
                            'item': json.dumps(fm)[:500],
                            'created_at': datetime.now(timezone.utc).isoformat()
                        }
                    })
                chunk_counts['failure_modes'] = len(chunks) - sum(chunk_counts.values())

            # 7. FEW-SHOT EXAMPLES - Store each example separately
            few_shot_examples = domain_context.get('few_shot_examples', [])
            if few_shot_examples:
                for i, example in enumerate(few_shot_examples[:5]):
                    chunk_id = f"example_{session_id}_{uuid.uuid4()}"
                    text = f"Few-Shot Example for {example.get('dimension', 'unknown')}: Input: {str(example.get('input', {}))[:100]}... Ideal Score: {example.get('ideal_evaluation', {}).get('score', 'N/A')}"
                    chunks.append({
                        'id': chunk_id,
                        'text': text,
                        'category': 'few_shot_example',
                        'metadata': {
                            'session_id': session_id,
                            'project_id': project_id or '',
                            'category': 'few_shot_example',
                            'dimension': example.get('dimension', ''),
                            'item': json.dumps(example)[:1000],
                            'created_at': datetime.now(timezone.utc).isoformat()
                        }
                    })
                chunk_counts['few_shot_examples'] = len(chunks) - sum(chunk_counts.values())

            # 8. ONE-SHOT TEMPLATE - Store as single chunk
            one_shot_template = domain_context.get('one_shot_template', {})
            if one_shot_template:
                chunk_id = f"template_{session_id}_{uuid.uuid4()}"
                chunks.append({
                    'id': chunk_id,
                    'text': f"One-Shot Evaluation Template: {json.dumps(one_shot_template)[:500]}",
                    'category': 'one_shot_template',
                    'metadata': {
                        'session_id': session_id,
                        'project_id': project_id or '',
                        'category': 'one_shot_template',
                        'item': json.dumps(one_shot_template)[:1000],
                        'created_at': datetime.now(timezone.utc).isoformat()
                    }
                })
                chunk_counts['one_shot_template'] = 1

            # Create embeddings and store in ChromaDB
            if chunks:
                logger.info(f"[Domain RAG] Creating embeddings for {len(chunks)} chunks...")

                # Create embeddings in parallel
                embedding_tasks = [self.create_embedding(chunk['text']) for chunk in chunks]
                embeddings = await asyncio.gather(*embedding_tasks)

                # Store in ChromaDB
                self.collection.add(
                    embeddings=embeddings,
                    documents=[chunk['text'] for chunk in chunks],
                    metadatas=[chunk['metadata'] for chunk in chunks],
                    ids=[chunk['id'] for chunk in chunks]
                )

                logger.info(f"[Domain RAG] Stored {len(chunks)} context chunks")
                logger.info(f"[Domain RAG] Breakdown: {chunk_counts}")

            return chunk_counts

        except Exception as e:
            logger.error(f"[Domain RAG] Error storing domain context: {e}", exc_info=True)
            raise

    async def retrieve_relevant_context(
        self,
        dimension: str,
        system_prompt: str,
        session_id: str,
        top_k: int = 5
    ) -> str:
        """
        Retrieve most relevant domain context for this eval generation.

        Uses semantic search to find context chunks most relevant to
        the dimension and system prompt.

        Args:
            dimension: Evaluation dimension (e.g., "schema", "accuracy")
            system_prompt: System prompt being evaluated
            session_id: User session ID
            top_k: Number of context chunks to retrieve (default: 5)

        Returns:
            Formatted string with relevant context (much smaller than full context)
        """
        try:
            # Check if we have any domain context for this session
            if self.collection.count() == 0:
                logger.info("[Domain RAG] No domain context stored yet")
                return ""

            # Build query combining dimension and system prompt
            query_text = f"Evaluation dimension: {dimension}\nSystem prompt context: {system_prompt[:300]}"

            # Create query embedding
            query_embedding = await self.create_embedding(query_text)

            # Search for relevant chunks - try user's session first
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                where={"session_id": session_id},  # Filter by session
                include=["documents", "metadatas", "distances"]
            )

            # If no results for user's session, fallback to default context
            if not results['documents'] or not results['documents'][0]:
                logger.info(f"[Domain RAG] No context for session {session_id}, trying default...")
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, self.collection.count()),
                    where={"session_id": "default"},  # Fallback to default
                    include=["documents", "metadatas", "distances"]
                )

            # Format relevant context
            if not results['documents'] or not results['documents'][0]:
                logger.info("[Domain RAG] No relevant context found")
                return ""

            relevant_chunks = []
            for i, (doc, meta, dist) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                similarity = 1.0 - dist
                category = meta.get('category', 'unknown')

                logger.info(f"[Domain RAG] Chunk {i+1}: {category} (similarity: {similarity:.3f})")

                # Only include if similarity is high enough
                if similarity > 0.3:  # Threshold for relevance
                    relevant_chunks.append(doc)

            if not relevant_chunks:
                logger.info("[Domain RAG] No chunks met similarity threshold")
                return ""

            # Format as compact context section
            context_section = f"""
RELEVANT DOMAIN CONTEXT ({len(relevant_chunks)} items):
{chr(10).join([f"• {chunk}" for chunk in relevant_chunks])}
"""

            logger.info(f"[Domain RAG] Retrieved {len(relevant_chunks)} relevant chunks (~{len(context_section)} chars)")

            return context_section

        except Exception as e:
            logger.error(f"[Domain RAG] Error retrieving context: {e}", exc_info=True)
            # Don't fail eval generation if context retrieval fails
            return ""

    async def delete_session_context(self, session_id: str) -> int:
        """
        Delete all domain context for a session.

        Args:
            session_id: Session ID to delete context for

        Returns:
            Number of chunks deleted
        """
        try:
            # Get all IDs for this session
            results = self.collection.get(
                where={"session_id": session_id},
                include=[]
            )

            if results['ids']:
                self.collection.delete(ids=results['ids'])
                count = len(results['ids'])
                logger.info(f"[Domain RAG] Deleted {count} chunks for session {session_id}")
                return count

            return 0

        except Exception as e:
            logger.error(f"[Domain RAG] Error deleting context: {e}")
            return 0

    def get_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about stored domain context.

        Args:
            session_id: Optional session ID to filter by

        Returns:
            Dictionary with statistics
        """
        try:
            if session_id:
                results = self.collection.get(
                    where={"session_id": session_id},
                    include=["metadatas"]
                )
            else:
                results = self.collection.get(
                    limit=1000,
                    include=["metadatas"]
                )

            if not results['metadatas']:
                return {
                    "total_chunks": 0,
                    "categories": {},
                    "sessions": []
                }

            # Count by category
            categories = {}
            sessions = set()

            for meta in results['metadatas']:
                category = meta.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
                sessions.add(meta.get('session_id', 'unknown'))

            return {
                "total_chunks": len(results['metadatas']),
                "categories": categories,
                "sessions": list(sessions),
                "session_count": len(sessions)
            }

        except Exception as e:
            logger.error(f"[Domain RAG] Error getting stats: {e}")
            return {"error": str(e)}


# Global instance (singleton pattern)
_domain_context_service: Optional[DomainContextService] = None


def get_domain_context_service() -> DomainContextService:
    """Get or create global domain context service instance."""
    global _domain_context_service
    if _domain_context_service is None:
        persist_dir = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
        _domain_context_service = DomainContextService(persist_directory=persist_dir)
    return _domain_context_service
