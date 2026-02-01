"""
Multi-Evaluator API Endpoints

New endpoints that support the multi-evaluator system alongside
the existing monolithic evaluator.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

from llm_client_v2 import EnhancedLLMClient
from multi_eval_integration import (
    MultiEvalIntegration,
    quick_evaluate,
    convert_verdict_to_legacy_format
)
from evaluator_factory import create_multi_evaluator_system
from multi_eval_with_meta import generate_multi_eval_with_meta
import project_storage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/projects", tags=["multi-evaluator"])


# ============================================================================
# Request/Response Models
# ============================================================================

class MultiEvalRequest(BaseModel):
    """Request to evaluate using multi-evaluator system"""
    input_data: str
    output: str
    use_legacy_format: bool = False  # Return in old format for compatibility


class MultiEvalResponse(BaseModel):
    """Response from multi-evaluator"""
    verdict: str
    score: float
    reason: str
    dimension_scores: Dict[str, float]
    individual_evaluations: List[Dict[str, Any]]
    performance: Dict[str, Any]
    evaluator_type: str = "multi_evaluator"


class ComparisonRequest(BaseModel):
    """Request to compare both evaluator types"""
    input_data: str
    output: str


class ComparisonResponse(BaseModel):
    """Comparison between monolithic and multi-evaluator"""
    multi_evaluator: MultiEvalResponse
    monolithic: Optional[Dict[str, Any]] = None
    comparison: Dict[str, Any]


class AutoDetectResponse(BaseModel):
    """Response showing auto-detected dimensions"""
    dimensions: List[Dict[str, Any]]
    total_evaluators: int
    tier1_count: int
    tier2_count: int
    estimated_cost: float
    estimated_latency_ms: int


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/{project_id}/multi-eval/evaluate", response_model=MultiEvalResponse)
async def evaluate_with_multi_system(project_id: str, request: MultiEvalRequest):
    """
    Evaluate an output using the multi-evaluator system.

    This automatically detects which dimensions to evaluate based on
    the system prompt and requirements.

    Benefits:
    - 70% cheaper than monolithic
    - 3x faster (parallel execution)
    - Per-dimension attribution
    - Focused, shorter prompts
    """

    # Load project
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get system prompt
    if not project.system_prompt_versions or len(project.system_prompt_versions) == 0:
        raise HTTPException(status_code=400, detail="No system prompt found")

    system_prompt = project.system_prompt_versions[-1]["prompt_text"]

    # Get requirements
    requirements = project.requirements
    if isinstance(requirements, dict):
        requirements = requirements.get("use_case", "")

    # Get API settings
    from server import get_settings
    settings = get_settings()
    provider = settings.get("provider", "anthropic")
    api_key = settings.get("api_key", "")

    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="API key not configured. Please set up your API key in settings."
        )

    # Create LLM client
    llm_client = EnhancedLLMClient(
        provider=provider,
        api_key=api_key
    )

    # Create integration
    integration = MultiEvalIntegration(llm_client)

    try:
        # Run multi-evaluator
        result = await integration.evaluate_with_multi_system(
            system_prompt=system_prompt,
            requirements=requirements,
            use_case=project.use_case,
            input_data=request.input_data,
            output=request.output,
            provider=provider,
            api_key=api_key,
            structured_requirements=project.structured_requirements
        )

        # Convert to response format
        if request.use_legacy_format:
            # Return in old format for backward compatibility
            legacy = convert_verdict_to_legacy_format(result)
            return legacy

        # Return new format with full details
        return MultiEvalResponse(
            verdict=result.verdict,
            score=result.score,
            reason=result.reason,
            dimension_scores={
                eval_result.dimension_name: eval_result.score
                for eval_result in result.individual_evaluations
            },
            individual_evaluations=[
                {
                    "dimension_name": e.dimension_name,
                    "score": e.score,
                    "passes": e.passes,
                    "reason": e.reason,
                    "weight": e.weight,
                    "is_critical": e.is_critical,
                    "evidence": e.evidence,
                    "model_used": e.model_used,
                    "latency_ms": e.latency_ms
                }
                for e in result.individual_evaluations
            ],
            performance={
                "total_latency_ms": result.total_latency_ms,
                "total_tokens_used": result.total_tokens_used,
                "auto_fail_triggered": result.auto_fail_triggered
            }
        )

    except Exception as e:
        logger.error(f"Multi-evaluator error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.get("/{project_id}/multi-eval/detect-dimensions", response_model=AutoDetectResponse)
async def detect_evaluation_dimensions(project_id: str):
    """
    Preview which evaluation dimensions will be used without running evaluation.

    This shows what the auto-detection system will evaluate based on
    your system prompt and requirements.
    """

    # Load project
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get system prompt
    if not project.system_prompt_versions or len(project.system_prompt_versions) == 0:
        raise HTTPException(status_code=400, detail="No system prompt found")

    system_prompt = project.system_prompt_versions[-1]["prompt_text"]

    # Get requirements
    requirements = project.requirements
    if isinstance(requirements, dict):
        requirements = requirements.get("use_case", "")

    # Detect dimensions
    from dimension_detector import DimensionDetector
    detector = DimensionDetector()

    detected = detector.detect_dimensions(
        system_prompt=system_prompt,
        requirements=requirements,
        use_case=project.use_case,
        structured_requirements=project.structured_requirements
    )

    # Count tiers
    tier1_count = sum(1 for d in detected if "Validator" in d.evaluator_type or "Checker" in d.evaluator_type)
    tier2_count = len(detected) - tier1_count

    # Estimate cost and latency (rough estimates)
    estimated_tokens = tier1_count * 800 + tier2_count * 2000
    estimated_cost = (estimated_tokens / 1_000_000) * 3.0  # Rough average
    estimated_latency_ms = max(
        tier1_count * 1500,  # Tier 1 parallel
        tier2_count * 3000   # Tier 2 parallel
    ) if tier1_count > 0 and tier2_count > 0 else (tier1_count + tier2_count) * 2000

    return AutoDetectResponse(
        dimensions=[
            {
                "name": d.name,
                "evaluator_type": d.evaluator_type,
                "weight": d.weight,
                "is_critical": d.is_critical,
                "min_pass_score": d.min_pass_score,
                "detection_reason": d.detection_reason
            }
            for d in detected
        ],
        total_evaluators=len(detected),
        tier1_count=tier1_count,
        tier2_count=tier2_count,
        estimated_cost=estimated_cost,
        estimated_latency_ms=estimated_latency_ms
    )


@router.post("/{project_id}/multi-eval/compare", response_model=ComparisonResponse)
async def compare_evaluator_approaches(project_id: str, request: ComparisonRequest):
    """
    Compare multi-evaluator vs monolithic evaluator side-by-side.

    Useful for validating that the multi-evaluator produces similar
    verdicts while being faster and cheaper.
    """

    # Load project
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get system prompt
    if not project.system_prompt_versions or len(project.system_prompt_versions) == 0:
        raise HTTPException(status_code=400, detail="No system prompt found")

    system_prompt = project.system_prompt_versions[-1]["prompt_text"]

    # Get requirements
    requirements = project.requirements
    if isinstance(requirements, dict):
        requirements = requirements.get("use_case", "")

    # Get API settings
    from server import get_settings
    settings = get_settings()
    provider = settings.get("provider", "anthropic")
    api_key = settings.get("api_key", "")

    if not api_key:
        raise HTTPException(status_code=400, detail="API key not configured")

    # Create LLM client
    llm_client = EnhancedLLMClient(provider=provider, api_key=api_key)
    integration = MultiEvalIntegration(llm_client)

    try:
        # Run comparison
        comparison = await integration.compare_approaches(
            system_prompt=system_prompt,
            requirements=requirements,
            use_case=project.use_case,
            input_data=request.input_data,
            output=request.output,
            provider=provider,
            api_key=api_key,
            structured_requirements=project.structured_requirements
        )

        # Format response
        multi_response = MultiEvalResponse(
            verdict=comparison.multi_eval_result.verdict,
            score=comparison.multi_eval_result.score,
            reason=comparison.multi_eval_result.reason,
            dimension_scores={
                e.dimension_name: e.score
                for e in comparison.multi_eval_result.individual_evaluations
            },
            individual_evaluations=[
                {
                    "dimension_name": e.dimension_name,
                    "score": e.score,
                    "passes": e.passes,
                    "reason": e.reason,
                    "weight": e.weight,
                    "is_critical": e.is_critical
                }
                for e in comparison.multi_eval_result.individual_evaluations
            ],
            performance={
                "total_latency_ms": comparison.multi_eval_latency_ms,
                "total_tokens_used": comparison.multi_eval_result.total_tokens_used
            }
        )

        return ComparisonResponse(
            multi_evaluator=multi_response,
            monolithic=comparison.monolithic_result,
            comparison={
                "cost_savings_pct": comparison.cost_savings_pct,
                "latency_improvement_pct": comparison.latency_improvement_pct,
                "verdict_agreement": comparison.verdict_agreement,
                "multi_cost": comparison.multi_eval_cost,
                "monolithic_cost": comparison.monolithic_cost
            }
        )

    except Exception as e:
        logger.error(f"Comparison error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.post("/{project_id}/multi-eval/batch", response_model=List[MultiEvalResponse])
async def batch_evaluate(
    project_id: str,
    requests: List[MultiEvalRequest]
):
    """
    Batch evaluate multiple input/output pairs.

    Efficient for running evaluations on multiple test cases.
    """

    results = []

    for req in requests:
        try:
            result = await evaluate_with_multi_system(project_id, req)
            results.append(result)
        except Exception as e:
            logger.error(f"Batch eval error: {e}")
            # Return error result
            results.append(MultiEvalResponse(
                verdict="ERROR",
                score=0.0,
                reason=f"Evaluation failed: {str(e)}",
                dimension_scores={},
                individual_evaluations=[],
                performance={"error": str(e)}
            ))

    return results
