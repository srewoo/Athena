"""
Meta-Evaluation API Endpoints

Endpoints for running meta-evaluation on eval prompts
to ensure they are high-quality before deployment.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

from llm_client_v2 import EnhancedLLMClient
from meta_evaluator import (
    MetaEvaluator,
    iterative_meta_eval_refinement,
    MetaEvalResult
)
import project_storage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/projects", tags=["meta-evaluation"])


# ============================================================================
# Request/Response Models
# ============================================================================

class MetaEvalRequest(BaseModel):
    """Request to run meta-evaluation"""
    eval_prompt: Optional[str] = None  # If not provided, use latest from project
    quality_threshold: float = 7.5


class MetaEvalResponse(BaseModel):
    """Response from meta-evaluation"""
    overall_quality_score: float
    passes_quality_gate: bool
    executive_summary: str

    audit_breakdown: Dict[str, Any]
    logic_gaps: List[Dict[str, str]]
    refinement_roadmap: List[str]
    suggested_improvements: Dict[str, Any]


class RefineEvalPromptRequest(BaseModel):
    """Request to refine eval prompt using meta-eval feedback"""
    max_iterations: int = 3
    quality_threshold: float = 7.5
    auto_apply: bool = False  # Automatically save improved version


class RefineEvalPromptResponse(BaseModel):
    """Response from refinement"""
    original_quality_score: float
    final_quality_score: float
    iterations_used: int
    improved_eval_prompt: str
    improvement_history: List[Dict[str, Any]]
    auto_applied: bool


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/{project_id}/eval-prompt/meta-evaluate", response_model=MetaEvalResponse)
async def meta_evaluate_eval_prompt(project_id: str, request: MetaEvalRequest):
    """
    Run meta-evaluation on an eval prompt to check quality.

    This audits the eval prompt against 5 criteria:
    1. Effectiveness & Relevance
    2. Structural Clarity
    3. Bias & Logic
    4. Metric Conflation
    5. Granularity

    Returns detailed feedback and suggestions for improvement.
    """

    # Load project
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get system prompt
    if not project.system_prompt_versions or len(project.system_prompt_versions) == 0:
        raise HTTPException(status_code=400, detail="No system prompt found")

    system_prompt = project.system_prompt_versions[-1]["prompt_text"]

    # Get eval prompt to evaluate
    if request.eval_prompt:
        eval_prompt = request.eval_prompt
    elif project.eval_prompt:
        eval_prompt = project.eval_prompt
    else:
        raise HTTPException(
            status_code=400,
            detail="No eval prompt found. Generate one first or provide in request."
        )

    # Get API settings
    from server import get_settings
    settings = get_settings()
    provider = settings.get("provider", "anthropic")
    api_key = settings.get("api_key", "")

    if not api_key:
        raise HTTPException(status_code=400, detail="API key not configured")

    # Create LLM client
    llm_client = EnhancedLLMClient(provider=provider, api_key=api_key)

    # Create meta-evaluator
    meta_evaluator = MetaEvaluator(
        llm_client=llm_client,
        quality_threshold=request.quality_threshold
    )

    try:
        # Run meta-evaluation
        result = await meta_evaluator.evaluate_eval_prompt(
            system_prompt=system_prompt,
            eval_prompt=eval_prompt
        )

        return MetaEvalResponse(
            overall_quality_score=result.overall_quality_score,
            passes_quality_gate=result.passes_quality_gate,
            executive_summary=result.executive_summary,
            audit_breakdown={
                "effectiveness_relevance": {
                    "score": result.effectiveness_score,
                    "analysis": result.effectiveness_analysis
                },
                "structural_clarity": {
                    "score": result.structural_clarity_score,
                    "analysis": result.structural_clarity_analysis
                },
                "bias_logic": {
                    "score": result.bias_score,
                    "analysis": result.bias_analysis
                },
                "metric_conflation": {
                    "score": result.metric_conflation_score,
                    "analysis": result.metric_conflation_analysis
                },
                "granularity": {
                    "score": result.granularity_score,
                    "analysis": result.granularity_analysis
                }
            },
            logic_gaps=result.logic_gaps,
            refinement_roadmap=result.refinement_roadmap,
            suggested_improvements=result.suggested_improvements
        )

    except Exception as e:
        logger.error(f"Meta-evaluation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Meta-evaluation failed: {str(e)}")


@router.post("/{project_id}/eval-prompt/refine-with-meta-eval", response_model=RefineEvalPromptResponse)
async def refine_eval_prompt_with_meta_eval(
    project_id: str,
    request: RefineEvalPromptRequest
):
    """
    Automatically refine eval prompt using meta-evaluation feedback.

    This runs iterative refinement:
    1. Meta-evaluate current eval prompt
    2. If quality < threshold, apply improvements
    3. Meta-evaluate again
    4. Repeat until quality threshold reached or max iterations

    If auto_apply=true, the improved eval prompt is saved to the project.
    """

    # Load project
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get system prompt
    if not project.system_prompt_versions or len(project.system_prompt_versions) == 0:
        raise HTTPException(status_code=400, detail="No system prompt found")

    system_prompt = project.system_prompt_versions[-1]["prompt_text"]

    # Get eval prompt
    if not project.eval_prompt:
        raise HTTPException(
            status_code=400,
            detail="No eval prompt found. Generate one first."
        )

    initial_eval_prompt = project.eval_prompt

    # Get API settings
    from server import get_settings
    settings = get_settings()
    provider = settings.get("provider", "anthropic")
    api_key = settings.get("api_key", "")

    if not api_key:
        raise HTTPException(status_code=400, detail="API key not configured")

    # Create LLM client
    llm_client = EnhancedLLMClient(provider=provider, api_key=api_key)

    try:
        logger.info(f"Starting iterative refinement for project {project_id}")

        # Run iterative refinement
        improved_prompt, history = await iterative_meta_eval_refinement(
            system_prompt=system_prompt,
            initial_eval_prompt=initial_eval_prompt,
            llm_client=llm_client,
            max_iterations=request.max_iterations,
            quality_threshold=request.quality_threshold
        )

        # Extract scores
        original_score = history[0].overall_quality_score if history else 0
        final_score = history[-1].overall_quality_score if history else 0

        # Auto-apply if requested
        auto_applied = False
        if request.auto_apply and improved_prompt != initial_eval_prompt:
            project.eval_prompt = improved_prompt
            from datetime import datetime
            project.updated_at = datetime.now()
            project_storage.save_project(project)
            auto_applied = True
            logger.info(f"Auto-applied improved eval prompt to project {project_id}")

        # Build response
        improvement_history = [
            {
                "iteration": i + 1,
                "quality_score": h.overall_quality_score,
                "passes_gate": h.passes_quality_gate,
                "summary": h.executive_summary,
                "logic_gaps_count": len(h.logic_gaps),
                "suggestions_count": len(h.refinement_roadmap)
            }
            for i, h in enumerate(history)
        ]

        return RefineEvalPromptResponse(
            original_quality_score=original_score,
            final_quality_score=final_score,
            iterations_used=len(history),
            improved_eval_prompt=improved_prompt,
            improvement_history=improvement_history,
            auto_applied=auto_applied
        )

    except Exception as e:
        logger.error(f"Refinement error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Refinement failed: {str(e)}")


@router.get("/{project_id}/eval-prompt/quality-check")
async def quick_quality_check(project_id: str):
    """
    Quick quality check of current eval prompt.

    Returns a simple pass/fail and quality score without full analysis.
    Useful for dashboards.
    """

    # Load project
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.eval_prompt:
        return {
            "has_eval_prompt": False,
            "quality_score": 0,
            "passes_gate": False,
            "message": "No eval prompt found"
        }

    # Get system prompt
    if not project.system_prompt_versions or len(project.system_prompt_versions) == 0:
        raise HTTPException(status_code=400, detail="No system prompt found")

    system_prompt = project.system_prompt_versions[-1]["prompt_text"]

    # Get API settings
    from server import get_settings
    settings = get_settings()
    provider = settings.get("provider", "anthropic")
    api_key = settings.get("api_key", "")

    if not api_key:
        raise HTTPException(status_code=400, detail="API key not configured")

    # Create LLM client
    llm_client = EnhancedLLMClient(provider=provider, api_key=api_key)
    meta_evaluator = MetaEvaluator(llm_client=llm_client, quality_threshold=7.5)

    try:
        result = await meta_evaluator.evaluate_eval_prompt(
            system_prompt=system_prompt,
            eval_prompt=project.eval_prompt
        )

        return {
            "has_eval_prompt": True,
            "quality_score": result.overall_quality_score,
            "passes_gate": result.passes_quality_gate,
            "message": result.executive_summary[:200],
            "needs_improvement": not result.passes_quality_gate
        }

    except Exception as e:
        logger.error(f"Quality check error: {e}")
        return {
            "has_eval_prompt": True,
            "quality_score": 0,
            "passes_gate": False,
            "message": f"Quality check failed: {str(e)}"
        }
