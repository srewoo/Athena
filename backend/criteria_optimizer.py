"""
Criteria Optimization Suite
Inspired by EvalLM's criteria management approach

Provides three meta-operations on evaluation dimensions:
1. MERGE: Identify and consolidate redundant dimensions
2. SPLIT: Break overly broad dimensions into atomic components
3. REFINE: Clarify vague or unclear dimension descriptions
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
import numpy as np
from llm_client import LlmClient

logger = logging.getLogger(__name__)


class CriteriaOptimizer:
    """
    Optimizes evaluation dimensions for clarity, coverage, and mutual exclusivity.
    """

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("CriteriaOptimizer initialized")

    def detect_overlaps(
        self,
        dimensions: List[Dict[str, str]],
        similarity_threshold: float = 0.70
    ) -> List[Dict[str, Any]]:
        """
        Detect overlapping/redundant dimensions using semantic similarity.

        Args:
            dimensions: List of dicts with 'name' and 'description'
            similarity_threshold: Cosine similarity threshold for overlap (default 0.70)

        Returns:
            List of overlap pairs with similarity scores
        """
        if len(dimensions) < 2:
            return []

        # Create embeddings for dimension descriptions
        descriptions = [f"{d['name']}: {d.get('description', '')}" for d in dimensions]
        embeddings = self.model.encode(descriptions, convert_to_tensor=False)

        overlaps = []
        for i in range(len(dimensions)):
            for j in range(i + 1, len(dimensions)):
                # Calculate cosine similarity
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )

                if sim >= similarity_threshold:
                    overlaps.append({
                        "dimension1": dimensions[i]['name'],
                        "dimension2": dimensions[j]['name'],
                        "similarity": float(sim),
                        "recommendation": "Consider merging - these dimensions test similar concepts"
                    })

        logger.info(f"[Criteria Optimizer] Detected {len(overlaps)} overlapping dimension pairs")
        return overlaps

    async def detect_overlaps_async(
        self,
        dimensions: List[Dict[str, str]],
        similarity_threshold: float = 0.70
    ) -> List[Dict[str, Any]]:
        """Async wrapper for detect_overlaps using thread pool to avoid blocking event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.detect_overlaps(dimensions, similarity_threshold)
        )

    async def merge_dimensions(
        self,
        dimensions: List[Dict[str, str]],
        overlap_pairs: List[Dict[str, Any]],
        provider: str,
        model: str,
        api_key: str
    ) -> List[Dict[str, Any]]:
        """
        Suggest merged versions of overlapping dimensions.

        Args:
            dimensions: List of dimension dicts
            overlap_pairs: Detected overlaps from detect_overlaps()
            provider: LLM provider
            model: Model name
            api_key: API key

        Returns:
            List of merge suggestions with original_dimensions and merged_dimension
        """
        if not overlap_pairs:
            return []

        merge_suggestions = []

        for overlap in overlap_pairs:
            dim1_name = overlap['dimension1']
            dim2_name = overlap['dimension2']

            # Find full dimension objects
            dim1 = next((d for d in dimensions if d['name'] == dim1_name), None)
            dim2 = next((d for d in dimensions if d['name'] == dim2_name), None)

            if not dim1 or not dim2:
                continue

            merge_prompt = f"""You are a helpful assistant that reviews evaluation criteria for redundancy and suggests consolidations.

Two evaluation dimensions have been identified as overlapping (similarity: {overlap['similarity']:.2f}):

**Dimension 1: {dim1['name']}**
Description: {dim1.get('description', 'N/A')}

**Dimension 2: {dim2['name']}**
Description: {dim2.get('description', 'N/A')}

Your task: Create a single, consolidated dimension that captures both concepts without overlap.

Output format (JSON):
{{
  "merged_name": "concise_merged_dimension_name",
  "merged_description": "Clear description covering both original dimensions",
  "rationale": "Brief explanation of why this merge makes sense",
  "original_dimensions": ["{dim1['name']}", "{dim2['name']}"]
}}

Provide ONLY valid JSON, no markdown.
"""

            try:
                client = LlmClient(provider=provider, model=model, api_key=api_key)
                response = await client.send_message(merge_prompt)

                # Parse JSON response
                import json
                merged = json.loads(response)
                merge_suggestions.append(merged)

            except Exception as e:
                logger.error(f"[Criteria Optimizer] Merge generation failed: {e}")
                continue

        logger.info(f"[Criteria Optimizer] Generated {len(merge_suggestions)} merge suggestions")
        return merge_suggestions

    async def split_dimension(
        self,
        dimension: Dict[str, str],
        provider: str,
        model: str,
        api_key: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a dimension is overly broad and suggest atomic splits.

        Args:
            dimension: Dimension dict with name and description
            provider: LLM provider
            model: Model name
            api_key: API key

        Returns:
            Split suggestion with sub-dimensions, or None if no split needed
        """
        split_prompt = f"""You are a helpful assistant that identifies overly broad evaluation criteria and breaks them into specific, mutually exclusive components.

**Dimension: {dimension['name']}**
Description: {dimension.get('description', 'N/A')}

Your task: Determine if this dimension tests multiple distinct concepts that should be evaluated separately.

Analysis criteria:
- Does it mix multiple evaluation aspects? (e.g., "accuracy AND completeness")
- Could it be split into 2-3 atomic sub-dimensions?
- Would splitting improve clarity and specificity?

If split is recommended, output (JSON):
{{
  "should_split": true,
  "rationale": "Brief explanation of why split is needed",
  "sub_dimensions": [
    {{
      "name": "specific_sub_dimension_1",
      "description": "Clear, focused description of first aspect"
    }},
    {{
      "name": "specific_sub_dimension_2",
      "description": "Clear, focused description of second aspect"
    }}
  ],
  "original_dimension": "{dimension['name']}"
}}

If NO split needed, output:
{{
  "should_split": false,
  "rationale": "This dimension is already focused and atomic"
}}

Provide ONLY valid JSON, no markdown.
"""

        try:
            client = LlmClient(provider=provider, model=model, api_key=api_key)
            response = await client.send_message(split_prompt)

            import json
            result = json.loads(response)

            if result.get('should_split'):
                logger.info(f"[Criteria Optimizer] Split recommended for '{dimension['name']}'")
                return result
            else:
                logger.info(f"[Criteria Optimizer] No split needed for '{dimension['name']}'")
                return None

        except Exception as e:
            logger.error(f"[Criteria Optimizer] Split analysis failed: {e}")
            return None

    async def refine_dimension(
        self,
        dimension: Dict[str, str],
        provider: str,
        model: str,
        api_key: str
    ) -> Dict[str, Any]:
        """
        Refine vague or unclear dimension descriptions.

        Args:
            dimension: Dimension dict with name and description
            provider: LLM provider
            model: Model name
            api_key: API key

        Returns:
            Refined dimension with improved description
        """
        refine_prompt = f"""You are a helpful assistant that clarifies vague or confusing evaluation criteria.

**Dimension: {dimension['name']}**
Current Description: {dimension.get('description', 'N/A')}

Your task: Improve this dimension description to be:
- Specific (not generic)
- Actionable (evaluator knows what to check)
- Clear (no ambiguity)
- Focused (tests one thing)

If the current description is vague, unclear, or generic, provide a refined version.

Output format (JSON):
{{
  "needs_refinement": true/false,
  "original_description": "{dimension.get('description', '')}",
  "refined_description": "Improved, specific description" OR null if no refinement needed,
  "rationale": "Brief explanation of what was improved",
  "dimension_name": "{dimension['name']}"
}}

Provide ONLY valid JSON, no markdown.
"""

        try:
            client = LlmClient(provider=provider, model=model, api_key=api_key)
            response = await client.send_message(refine_prompt)

            import json
            result = json.loads(response)

            if result.get('needs_refinement'):
                logger.info(f"[Criteria Optimizer] Refinement suggested for '{dimension['name']}'")
            else:
                logger.info(f"[Criteria Optimizer] '{dimension['name']}' is already clear")

            return result

        except Exception as e:
            logger.error(f"[Criteria Optimizer] Refinement failed: {e}")
            return {
                "needs_refinement": False,
                "error": str(e)
            }

    async def optimize_suite(
        self,
        dimensions: List[Dict[str, str]],
        provider: str,
        model: str,
        api_key: str,
        operations: List[str] = ["merge", "split", "refine"]
    ) -> Dict[str, Any]:
        """
        Run full optimization suite on dimension set.

        Args:
            dimensions: List of dimension dicts
            provider: LLM provider
            model: Model name
            api_key: API key
            operations: Which operations to run (default all)

        Returns:
            Comprehensive optimization report
        """
        report = {
            "original_count": len(dimensions),
            "overlaps": [],
            "merge_suggestions": [],
            "split_suggestions": [],
            "refinement_suggestions": [],
            "optimized_count": len(dimensions)
        }

        # 1. Detect overlaps
        if "merge" in operations:
            overlaps = self.detect_overlaps(dimensions)
            report["overlaps"] = overlaps

            if overlaps:
                merge_suggestions = await self.merge_dimensions(
                    dimensions, overlaps, provider, model, api_key
                )
                report["merge_suggestions"] = merge_suggestions

        # 2. Check for splits
        if "split" in operations:
            for dim in dimensions:
                split_result = await self.split_dimension(dim, provider, model, api_key)
                if split_result and split_result.get('should_split'):
                    report["split_suggestions"].append(split_result)

        # 3. Refine descriptions
        if "refine" in operations:
            for dim in dimensions:
                refine_result = await self.refine_dimension(dim, provider, model, api_key)
                if refine_result.get('needs_refinement'):
                    report["refinement_suggestions"].append(refine_result)

        # Calculate optimized count
        dims_to_remove = len(report["merge_suggestions"])  # Each merge removes 1 dim
        dims_to_add = sum(len(s.get('sub_dimensions', [])) - 1 for s in report["split_suggestions"])
        report["optimized_count"] = len(dimensions) - dims_to_remove + dims_to_add

        logger.info(f"[Criteria Optimizer] Suite optimization complete: {len(dimensions)} → {report['optimized_count']} dimensions")
        return report


# Singleton instance
_criteria_optimizer = None


def get_criteria_optimizer() -> CriteriaOptimizer:
    """Get or create the global CriteriaOptimizer instance."""
    global _criteria_optimizer
    if _criteria_optimizer is None:
        _criteria_optimizer = CriteriaOptimizer()
    return _criteria_optimizer
