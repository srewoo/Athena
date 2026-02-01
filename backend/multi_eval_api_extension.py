"""
Extension to multi_eval_api.py for separate eval prompts with meta-evaluation

This adds endpoints to generate separate eval prompts for each dimension
and meta-evaluate each one.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging

from llm_client_v2 import EnhancedLLMClient
from multi_eval_with_meta import generate_multi_eval_with_meta
import project_storage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/projects", tags=["multi-evaluator-separate"])


class SeparateEvalsResponse(BaseModel):
    """Response containing separate eval prompts for each dimension"""
    total_evaluators: int
    evaluators: List[Dict[str, Any]]
    extraction_metadata: Dict[str, Any]  # NEW: Shows how requirements were extracted


@router.get("/{project_id}/multi-eval/separate-prompts", response_model=SeparateEvalsResponse)
async def get_separate_eval_prompts(project_id: str):
    """
    Generate separate eval prompts for each dimension and meta-evaluate each.

    Returns a list of evaluators, each with:
    - dimension_name: The dimension being evaluated
    - eval_prompt: The dedicated eval prompt for this dimension
    - meta_quality_score: Quality score from meta-evaluation
    - meta_passes_gate: Whether it passes quality threshold
    - meta_executive_summary: Summary from meta-eval
    - meta_audit_scores: Breakdown of 5-point audit
    - was_refined: Whether it was auto-refined

    This allows the UI to display each eval prompt in a separate box.
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

    try:
        # Generate separate eval prompts with meta-evaluation
        # Using intelligent LLM-based extraction for better accuracy
        result = await generate_multi_eval_with_meta(
            system_prompt=system_prompt,
            requirements=requirements,
            use_case=project.use_case,
            llm_client=llm_client,
            structured_requirements=project.structured_requirements,
            meta_quality_threshold=7.5,
            use_intelligent_extraction=True  # Use intelligent extraction
        )

        # Format response
        evaluators_data = [
            {
                "dimension_name": e.dimension_name,
                "evaluator_type": e.evaluator_type,
                "weight": e.weight,
                "is_critical": e.is_critical,
                "eval_prompt": e.eval_prompt,
                "meta_evaluation": {
                    "quality_score": e.meta_quality_score,
                    "passes_gate": e.meta_passes_gate,
                    "executive_summary": e.meta_executive_summary,
                    "audit_scores": e.meta_audit_scores,
                    "logic_gaps": e.meta_logic_gaps,
                    "refinement_roadmap": e.meta_refinement_roadmap
                },
                "was_refined": e.was_refined,
                "original_prompt": e.original_prompt if e.was_refined else None
            }
            for e in result.evaluators
        ]

        return SeparateEvalsResponse(
            total_evaluators=len(evaluators_data),
            evaluators=evaluators_data,
            extraction_metadata=result.extraction_metadata  # Include extraction metadata
        )

    except Exception as e:
        logger.error(f"Error generating separate eval prompts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate eval prompts: {str(e)}")


@router.post("/{project_id}/multi-eval/test-extraction")
async def test_intelligent_extraction(project_id: str):
    """
    Test endpoint to verify intelligent extraction is working.

    Returns the extracted requirements without generating eval prompts.
    Use this to verify:
    - LLM settings are correct
    - Extraction is using configured API key
    - Domain and risk level are correctly detected
    - Requirements are properly extracted
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

    try:
        # Test intelligent extraction
        from intelligent_requirements_extractor import extract_requirements_intelligently
        from dataclasses import asdict

        logger.info("Testing intelligent extraction...")

        extracted = await extract_requirements_intelligently(
            system_prompt=system_prompt,
            requirements=requirements,
            use_case=project.use_case,
            llm_client=llm_client,
            structured_requirements=project.structured_requirements
        )

        # Return full extraction result
        return {
            "success": True,
            "llm_settings": {
                "provider": provider,
                "api_key_configured": bool(api_key),
                "api_key_preview": f"{api_key[:8]}..." if api_key else None
            },
            "extracted_requirements": {
                "use_case": extracted.use_case,
                "primary_function": extracted.primary_function,
                "domain": extracted.domain,
                "risk_level": extracted.risk_level,
                "key_requirements": extracted.key_requirements,
                "must_do": extracted.must_do,
                "must_not_do": extracted.must_not_do,
                "output_format": extracted.output_format,
                "tone": extracted.tone,
                "style_requirements": extracted.style_requirements,
                "quality_priorities": extracted.quality_priorities,
                "critical_dimensions": extracted.critical_dimensions,
                "important_dimensions": extracted.important_dimensions,
                "target_audience": extracted.target_audience,
                "example_scenarios": extracted.example_scenarios,
                "confidence_score": extracted.confidence_score,
                "extraction_notes": extracted.extraction_notes
            }
        }

    except Exception as e:
        logger.error(f"Error testing extraction: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "llm_settings": {
                "provider": provider,
                "api_key_configured": bool(api_key)
            }
        }
