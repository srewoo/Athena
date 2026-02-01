# DEPRECATED - SHIM FOR BACKWARD COMPATIBILITY
"""
This module is deprecated and kept only for backward compatibility
with multi_eval_api.py (which is also deprecated).

Use unified_eval_generator.py and unified_eval_api.py instead.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FinalVerdict:
    """Minimal verdict class for compatibility"""
    verdict: str
    score: float
    reason: str
    individual_evaluations: list
    total_latency_ms: int
    total_tokens_used: int
    auto_fail_triggered: bool = False


class MultiEvalIntegration:
    """
    DEPRECATED: Use unified_eval_generator.UnifiedEvalGenerator instead.

    This is a compatibility shim that redirects to the unified system.
    """

    def __init__(self, llm_client: Any):
        self.llm_client = llm_client
        logger.warning(
            "MultiEvalIntegration is deprecated. "
            "Use unified_eval_generator.UnifiedEvalGenerator instead."
        )

    async def evaluate_with_multi_system(
        self,
        system_prompt: str,
        requirements: str,
        use_case: str,
        input_data: str,
        output: str,
        provider: str = "anthropic",
        api_key: str = "",
        structured_requirements: Optional[Dict[str, Any]] = None
    ) -> FinalVerdict:
        """
        DEPRECATED: Redirects to unified system.
        """
        logger.warning("evaluate_with_multi_system is deprecated. Use unified_eval_api instead.")

        # Return a basic verdict for compatibility
        # The unified API should be used instead
        return FinalVerdict(
            verdict="DEPRECATED",
            score=0.0,
            reason="This endpoint is deprecated. Use /api/projects/{id}/eval/generate instead.",
            individual_evaluations=[],
            total_latency_ms=0,
            total_tokens_used=0
        )

    async def compare_approaches(self, **kwargs):
        """DEPRECATED: Use unified API instead."""
        logger.warning("compare_approaches is deprecated.")
        return None


def convert_verdict_to_legacy_format(verdict: FinalVerdict) -> Dict[str, Any]:
    """
    DEPRECATED: Convert verdict to legacy format for backward compatibility.
    """
    return {
        'verdict': verdict.verdict,
        'score': verdict.score,
        'reason': verdict.reason,
        'dimension_scores': {},
        'individual_evaluations': verdict.individual_evaluations,
        'performance': {
            'total_latency_ms': verdict.total_latency_ms,
            'total_tokens_used': verdict.total_tokens_used
        }
    }


async def quick_evaluate(*args, **kwargs):
    """DEPRECATED: Use unified_eval_api instead."""
    logger.warning("quick_evaluate is deprecated. Use unified API instead.")
    return FinalVerdict(
        verdict="DEPRECATED",
        score=0.0,
        reason="Use /api/projects/{id}/eval/generate instead",
        individual_evaluations=[],
        total_latency_ms=0,
        total_tokens_used=0
    )
