"""
Unified Eval API

This module consolidates ALL eval-related API endpoints:
- Eval prompt generation (single and multi)
- Meta-evaluation
- Testing and validation
- Intelligent extraction

Replaces:
- project_api.py eval endpoints
- multi_eval_api.py
- multi_eval_api_extension.py
- meta_eval_api.py

This is the SINGLE SOURCE OF TRUTH for eval APIs.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging

from llm_client_v2 import EnhancedLLMClient
from unified_eval_generator import (
    UnifiedEvalGenerator,
    EvalGenerationConfig,
    EvalMode,
    generate_eval_unified
)
import project_storage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/projects", tags=["unified-eval"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class GenerateEvalRequest(BaseModel):
    """Request to generate eval prompt(s)"""
    mode: Optional[str] = "auto"  # "single", "multi", or "auto"
    use_meta_evaluation: bool = True
    use_intelligent_extraction: bool = True
    meta_quality_threshold: float = 8.5
    auto_refine: bool = True
    requirements: Optional[str] = None  # Override project requirements with custom aspect


class EvalPromptResponse(BaseModel):
    """Response containing generated eval prompt(s)"""
    mode: str
    eval_prompt: Optional[str]  # For single mode
    evaluators: Optional[List[Dict[str, Any]]]  # For multi mode
    extraction_metadata: Dict[str, Any]
    meta_evaluation: Optional[Dict[str, Any]]  # For single mode
    generation_metadata: Dict[str, Any]


class TestExtractionResponse(BaseModel):
    """Response from testing intelligent extraction"""
    success: bool
    extracted_requirements: Dict[str, Any]
    llm_settings: Dict[str, Any]


class MetaEvaluateRequest(BaseModel):
    """Request to meta-evaluate an eval prompt"""
    eval_prompt: str


class MetaEvaluateResponse(BaseModel):
    """Response from meta-evaluation"""
    quality_score: float
    passes_gate: bool
    executive_summary: str
    audit_scores: Dict[str, float]
    logic_gaps: List[Dict[str, str]]
    refinement_roadmap: List[str]


class SuggestAspectsResponse(BaseModel):
    """Response with suggested evaluation aspects"""
    suggested_aspects: List[str]
    domain: str
    risk_level: str
    reasoning: str


class BatchEvalRequest(BaseModel):
    """Request to generate multiple eval prompts in parallel"""
    aspects: List[str]  # List of aspects to evaluate
    max_retries: int = 3  # Retry attempts per aspect


class BatchEvalResponse(BaseModel):
    """Response with multiple eval prompts"""
    total_aspects: int
    successful: int
    failed: int
    results: List[Dict[str, Any]]
    total_duration_ms: int
    average_duration_ms: int


# ============================================================================
# MAIN ENDPOINTS
# ============================================================================

@router.post("/{project_id}/eval/generate", response_model=EvalPromptResponse)
async def generate_eval_prompt(project_id: str, request: GenerateEvalRequest):
    """
    Generate eval prompt(s) with meta-evaluation.

    This is the MAIN ENDPOINT for all eval generation.

    Features:
    - Auto-detects whether to use single or multi mode
    - Always includes meta-evaluation (unless disabled)
    - Uses intelligent requirements extraction
    - Auto-refines low-quality prompts
    - Returns both single eval or multiple evaluators

    Modes:
    - "auto": Auto-detect based on complexity (default)
    - "single": Generate one monolithic eval prompt
    - "multi": Generate multiple separate evaluators
    """

    # Load project
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get system prompt
    if not project.system_prompt_versions or len(project.system_prompt_versions) == 0:
        raise HTTPException(status_code=400, detail="No system prompt found")

    system_prompt = project.system_prompt_versions[-1]["prompt_text"]

    # Get requirements (use custom requirements if provided, otherwise use project requirements)
    if hasattr(request, 'requirements') and request.requirements:
        requirements = request.requirements
    else:
        requirements = project.requirements
        if isinstance(requirements, dict):
            requirements = requirements.get("use_case", "")

    # Get API settings
    from shared_settings import get_settings
    settings = get_settings()
    provider = settings.get("provider", "anthropic")
    api_key = settings.get("api_key", "")

    if not api_key:
        raise HTTPException(status_code=400, detail="API key not configured")

    # Create LLM client
    llm_client = EnhancedLLMClient()

    # Parse mode
    try:
        mode = EvalMode(request.mode.lower())
    except ValueError:
        mode = EvalMode.AUTO

    # Configure generation
    config = EvalGenerationConfig(
        mode=mode,
        use_intelligent_extraction=request.use_intelligent_extraction,
        use_meta_evaluation=request.use_meta_evaluation,
        meta_quality_threshold=request.meta_quality_threshold,
        auto_refine_on_low_quality=request.auto_refine
    )

    generator = UnifiedEvalGenerator(llm_client, config)

    try:
        # Generate eval
        result = await generator.generate_eval(
            system_prompt=system_prompt,
            requirements=requirements,
            use_case=project.use_case,
            structured_requirements=project.structured_requirements
        )

        # Format response based on mode
        if result.mode == EvalMode.MULTI:
            return EvalPromptResponse(
                mode="multi",
                eval_prompt=None,
                evaluators=result.multi_eval.evaluators,
                extraction_metadata=result.multi_eval.extraction_metadata,
                meta_evaluation=None,
                generation_metadata=result.generation_metadata
            )
        else:
            meta_eval_dict = None
            if result.single_eval.meta_evaluation:
                meta = result.single_eval.meta_evaluation
                meta_eval_dict = {
                    "quality_score": meta.quality_score,
                    "passes_gate": meta.passes_gate,
                    "executive_summary": meta.executive_summary,
                    "audit_scores": meta.audit_scores,
                    "logic_gaps": meta.logic_gaps,
                    "refinement_roadmap": meta.refinement_roadmap
                }

            return EvalPromptResponse(
                mode="single",
                eval_prompt=result.single_eval.prompt,
                evaluators=None,
                extraction_metadata=result.generation_metadata,
                meta_evaluation=meta_eval_dict,
                generation_metadata={
                    "was_refined": result.single_eval.was_refined,
                    "validation_passed": result.single_eval.validation_passed,
                    "validation_issues": result.single_eval.validation_issues
                }
            )

    except Exception as e:
        logger.error(f"Error generating eval prompt: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate eval: {str(e)}")


# ============================================================================
# TESTING & UTILITIES
# ============================================================================

@router.post("/{project_id}/eval/test-extraction", response_model=TestExtractionResponse)
async def test_intelligent_extraction(project_id: str):
    """
    Test intelligent requirements extraction.

    Returns extracted requirements without generating full eval prompts.
    Use this to verify:
    - LLM settings are correct
    - Domain and risk level are detected
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

    # Get requirements (use custom requirements if provided, otherwise use project requirements)
    if hasattr(request, 'requirements') and request.requirements:
        requirements = request.requirements
    else:
        requirements = project.requirements
        if isinstance(requirements, dict):
            requirements = requirements.get("use_case", "")

    # Get API settings
    from shared_settings import get_settings
    settings = get_settings()
    provider = settings.get("provider", "anthropic")
    api_key = settings.get("api_key", "")

    if not api_key:
        raise HTTPException(status_code=400, detail="API key not configured")

    # Create LLM client
    llm_client = EnhancedLLMClient()

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
        return TestExtractionResponse(
            success=True,
            llm_settings={
                "provider": provider,
                "api_key_configured": bool(api_key),
                "api_key_preview": f"{api_key[:8]}..." if api_key else None
            },
            extracted_requirements={
                "use_case": extracted.use_case,
                "primary_function": extracted.primary_function,
                "domain": extracted.domain,
                "risk_level": extracted.risk_level,
                "key_requirements": extracted.key_requirements,
                "must_do": extracted.must_do,
                "must_not_do": extracted.must_not_do,
                "output_format": extracted.output_format,
                "tone": extracted.tone,
                "quality_priorities": extracted.quality_priorities,
                "critical_dimensions": extracted.critical_dimensions,
                "confidence_score": extracted.confidence_score,
                "extraction_notes": extracted.extraction_notes
            }
        )

    except Exception as e:
        logger.error(f"Error testing extraction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.post("/{project_id}/eval/meta-evaluate", response_model=MetaEvaluateResponse)
async def meta_evaluate_prompt(project_id: str, request: MetaEvaluateRequest):
    """
    Meta-evaluate an eval prompt.

    Returns quality assessment and improvement suggestions.
    """

    # Load project
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get system prompt
    if not project.system_prompt_versions or len(project.system_prompt_versions) == 0:
        raise HTTPException(status_code=400, detail="No system prompt found")

    system_prompt = project.system_prompt_versions[-1]["prompt_text"]

    # Get API settings
    from shared_settings import get_settings
    settings = get_settings()
    api_key = settings.get("api_key", "")

    if not api_key:
        raise HTTPException(status_code=400, detail="API key not configured")

    # Create LLM client
    llm_client = EnhancedLLMClient()

    try:
        # Meta-evaluate
        from meta_evaluator import MetaEvaluator

        meta_evaluator = MetaEvaluator(llm_client, quality_threshold=7.5)

        meta_result = await meta_evaluator.evaluate_eval_prompt(
            system_prompt=system_prompt,
            eval_prompt=request.eval_prompt
        )

        return MetaEvaluateResponse(
            quality_score=meta_result.overall_quality_score,
            passes_gate=meta_result.passes_quality_gate,
            executive_summary=meta_result.executive_summary,
            audit_scores={
                "effectiveness": meta_result.effectiveness_score,
                "structural_clarity": meta_result.structural_clarity_score,
                "bias": meta_result.bias_score,
                "metric_conflation": meta_result.metric_conflation_score,
                "granularity": meta_result.granularity_score
            },
            logic_gaps=meta_result.logic_gaps,
            refinement_roadmap=meta_result.refinement_roadmap
        )

    except Exception as e:
        logger.error(f"Error in meta-evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Meta-evaluation failed: {str(e)}")


@router.post("/{project_id}/eval/generate-batch", response_model=BatchEvalResponse)
async def generate_batch_evals(project_id: str, request: BatchEvalRequest):
    """
    Generate multiple eval prompts in PARALLEL with retry logic.

    This endpoint:
    - Processes multiple aspects in parallel (not sequential)
    - Each aspect gets its own token budget (no token limit issues)
    - Automatic retry on failure (configurable attempts)
    - Returns all results together
    - Fast: 6 aspects in ~same time as 1 aspect

    Perfect for multi-aspect evaluation generation!
    """

    # Load project
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get system prompt
    if not project.system_prompt_versions or len(project.system_prompt_versions) == 0:
        raise HTTPException(status_code=400, detail="No system prompt found")

    system_prompt = project.system_prompt_versions[-1]["prompt_text"]

    # Get API settings
    from shared_settings import get_settings
    settings = get_settings()
    provider = settings.get("provider", "anthropic")
    api_key = settings.get("api_key", "")
    model_name = settings.get("model_name", "claude-3-5-sonnet-20241022")

    if not api_key:
        raise HTTPException(status_code=400, detail="API key not configured")

    if not request.aspects or len(request.aspects) == 0:
        raise HTTPException(status_code=400, detail="No aspects provided")

    # Create LLM client
    llm_client = EnhancedLLMClient()

    try:
        # Generate all evals in parallel
        from parallel_eval_generator import generate_multiple_evals_parallel

        results = await generate_multiple_evals_parallel(
            aspects=request.aspects,
            system_prompt=system_prompt,
            use_case=project.use_case,
            llm_client=llm_client,
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            max_retries=request.max_retries
        )

        # Format results (include enhanced fields)
        formatted_results = []
        for result in results:
            formatted_results.append({
                "aspect": result.aspect,
                "success": result.success,
                "eval_prompt": result.eval_prompt,
                "meta_quality": result.meta_quality,
                "passes_gate": result.passes_gate,
                "was_refined": result.was_refined,
                "executive_summary": result.executive_summary,
                "audit_scores": result.audit_scores or {},
                "logic_gaps": result.logic_gaps or [],
                "refinement_roadmap": result.refinement_roadmap or [],
                "error": result.error,
                "attempts": result.attempts,
                "duration_ms": result.duration_ms,
                # Enhanced fields (best practices implementation)
                "quality_analysis": result.quality_analysis or {},
                "calibration_examples": result.calibration_examples or [],
                "rubric_levels": result.rubric_levels or [],
                "evaluation_purpose": result.evaluation_purpose,
                "ai_system_context": result.ai_system_context or {}
            })

        # Calculate stats
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_duration = sum(r.duration_ms for r in results)
        avg_duration = total_duration // len(results) if results else 0

        return BatchEvalResponse(
            total_aspects=len(results),
            successful=successful,
            failed=failed,
            results=formatted_results,
            total_duration_ms=total_duration,
            average_duration_ms=avg_duration
        )

    except Exception as e:
        logger.error(f"Error in batch eval generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch eval generation failed: {str(e)}")


@router.post("/{project_id}/eval/generate-simple")
async def generate_simple_eval(project_id: str, request: GenerateEvalRequest):
    """
    Generate eval prompt using SIMPLE method (no meta-eval, no auto-refine).

    This is a lightweight alternative that:
    - Makes ONE LLM call per aspect
    - No meta-evaluation (faster, avoids compatibility issues)
    - No auto-refinement (avoids token limits)
    - Returns immediately

    Use this for multi-aspect generation until full system is fixed.
    """

    # Load project
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get system prompt
    if not project.system_prompt_versions or len(project.system_prompt_versions) == 0:
        raise HTTPException(status_code=400, detail="No system prompt found")

    system_prompt = project.system_prompt_versions[-1]["prompt_text"]

    # Get requirements (use custom requirements if provided, otherwise use project requirements)
    if hasattr(request, 'requirements') and request.requirements:
        aspect = request.requirements
    else:
        requirements = project.requirements
        if isinstance(requirements, dict):
            aspect = requirements.get("use_case", "General quality")
        else:
            aspect = requirements or "General quality"

    # Get API settings
    from shared_settings import get_settings
    settings = get_settings()
    provider = settings.get("provider", "anthropic")
    api_key = settings.get("api_key", "")
    model = settings.get("model_name", "claude-3-5-sonnet-20241022")

    if not api_key:
        raise HTTPException(status_code=400, detail="API key not configured")

    # Create LLM client
    llm_client = EnhancedLLMClient()

    try:
        # Use simple generator
        from simple_eval_generator import generate_simple_eval_prompt

        result = await generate_simple_eval_prompt(
            llm_client=llm_client,
            system_prompt=system_prompt,
            aspect=aspect,
            use_case=project.use_case,
            provider=provider,
            api_key=api_key,
            model=model
        )

        # Return in unified format
        return EvalPromptResponse(
            mode="single",
            eval_prompt=result["eval_prompt"],
            evaluators=None,
            extraction_metadata={
                "method": "simple_generation",
                "aspect": result["aspect"]
            },
            meta_evaluation={
                "quality_score": result["quality_score"],
                "passes_gate": result["passes_gate"],
                "executive_summary": result["reasoning"],
                "audit_scores": {},
                "logic_gaps": [],
                "refinement_roadmap": []
            },
            generation_metadata={
                "was_refined": result["was_refined"],
                "validation_passed": True,
                "validation_issues": [],
                "method": "simple_single_call"
            }
        )

    except Exception as e:
        logger.error(f"Error in simple eval generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Simple eval generation failed: {str(e)}")


@router.get("/{project_id}/eval/suggest-aspects", response_model=SuggestAspectsResponse)
async def suggest_evaluation_aspects(project_id: str):
    """
    Suggest evaluation aspects based on system prompt analysis.

    Analyzes the system prompt to suggest relevant evaluation dimensions
    that should be tested. Helps users get started quickly with appropriate
    evaluation criteria for their specific use case.
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

    try:
        # Use intelligent extraction to understand the domain and context
        from intelligent_requirements_extractor import extract_requirements_intelligently
        from shared_settings import get_settings

        settings = get_settings()
        api_key = settings.get("api_key", "")

        if not api_key:
            # Fallback to basic suggestions if no API key
            return _get_generic_suggestions()

        llm_client = EnhancedLLMClient()

        # Use LLM to suggest aspects directly based on system prompt
        suggestions_result = await _suggest_aspects_with_llm(
            llm_client=llm_client,
            system_prompt=system_prompt,
            use_case=project.use_case,
            requirements=requirements,
            api_key=api_key
        )

        return SuggestAspectsResponse(
            suggested_aspects=suggestions_result["aspects"],
            domain=suggestions_result.get("domain", "General"),
            risk_level=suggestions_result.get("risk_level", "medium"),
            reasoning=suggestions_result.get("reasoning", "LLM-generated suggestions based on system prompt analysis")
        )

    except Exception as e:
        logger.error(f"Error suggesting aspects: {e}", exc_info=True)
        # Return generic suggestions on error
        return _get_generic_suggestions()


def _generate_aspect_suggestions(
    domain: str,
    risk_level: str,
    use_case: str,
    primary_function: str,
    critical_dimensions: List[str],
    must_do: List[str],
    must_not_do: List[str],
    output_format: str
) -> List[str]:
    """Generate domain-specific evaluation aspect suggestions"""

    suggestions = []

    # Domain-specific suggestions
    domain_lower = domain.lower()

    if "medical" in domain_lower or "healthcare" in domain_lower or "health" in domain_lower:
        suggestions = [
            "Medical accuracy and evidence-based information",
            "Appropriate medical disclaimers and limitations",
            "HIPAA compliance and privacy protection",
            "Contraindication and safety warnings",
            "Appropriate urgency and emergency escalation",
            "Clear communication for patient understanding"
        ]
    elif "legal" in domain_lower or "law" in domain_lower:
        suggestions = [
            "Legal accuracy and jurisdiction awareness",
            "Appropriate legal disclaimers",
            "Ethical compliance and conflict of interest",
            "Citation of relevant laws and precedents",
            "Clear distinction between legal advice and information"
        ]
    elif "financial" in domain_lower or "banking" in domain_lower or "investment" in domain_lower:
        suggestions = [
            "Financial accuracy and calculation correctness",
            "Regulatory compliance (SEC, FINRA)",
            "Risk disclosures and disclaimers",
            "Conflict of interest transparency",
            "Clear communication of financial concepts"
        ]
    elif "customer service" in domain_lower or "support" in domain_lower:
        suggestions = [
            "Empathy and emotional intelligence",
            "Policy and product knowledge accuracy",
            "Appropriate escalation decisions",
            "Professional and courteous tone",
            "Clear and actionable guidance",
            "Efficient issue resolution"
        ]
    elif "code" in domain_lower or "programming" in domain_lower or "software" in domain_lower:
        suggestions = [
            "Code correctness and functionality",
            "Security vulnerability detection",
            "Performance and efficiency",
            "Best practices adherence",
            "Clear code explanations",
            "Appropriate error handling"
        ]
    elif "education" in domain_lower or "tutoring" in domain_lower or "teaching" in domain_lower:
        suggestions = [
            "Pedagogical effectiveness",
            "Accuracy of educational content",
            "Age-appropriate explanations",
            "Encouragement and positive feedback",
            "Clear learning objectives",
            "Appropriate difficulty progression"
        ]
    elif "creative" in domain_lower or "writing" in domain_lower or "content" in domain_lower:
        suggestions = [
            "Creative quality and originality",
            "Tone and style appropriateness",
            "Grammar and clarity",
            "Engagement and readability",
            "Brand voice consistency",
            "Target audience appropriateness"
        ]
    else:
        # Generic suggestions based on risk level
        if risk_level == "high":
            suggestions = [
                "Accuracy and factual correctness",
                "Safety and harm prevention",
                "Appropriate disclaimers and limitations",
                "Regulatory compliance",
                "Clear risk communication",
                "Proper escalation and oversight"
            ]
        elif risk_level == "medium":
            suggestions = [
                "Accuracy and correctness",
                "Professional tone and appropriateness",
                "Clear and understandable responses",
                "Following instructions precisely",
                "Handling edge cases appropriately"
            ]
        else:  # low risk
            suggestions = [
                "Response quality and relevance",
                "Clarity and conciseness",
                "Tone and professionalism",
                "Instruction following",
                "Completeness of response"
            ]

    # Add output format validation if specified
    if output_format and output_format.lower() not in ["text", "natural language", "free-form"]:
        if "json" in output_format.lower():
            suggestions.append("JSON structure and schema compliance")
        elif "xml" in output_format.lower():
            suggestions.append("XML structure and schema compliance")
        elif "markdown" in output_format.lower():
            suggestions.append("Markdown formatting correctness")
        elif "csv" in output_format.lower():
            suggestions.append("CSV format and data integrity")

    # Add critical dimensions from extraction
    for dim in critical_dimensions[:3]:  # Add top 3 critical dimensions
        if dim not in suggestions:
            suggestions.append(dim)

    # Limit to 8 suggestions to avoid overwhelming
    return suggestions[:8]


async def _suggest_aspects_with_llm(
    llm_client: EnhancedLLMClient,
    system_prompt: str,
    use_case: str,
    requirements: str,
    api_key: str
) -> Dict[str, Any]:
    """Use LLM to suggest evaluation aspects grounded in the system prompt"""

    suggestion_prompt = f"""Analyze this AI system prompt and suggest 6-8 specific evaluation aspects that should be tested.

SYSTEM PROMPT TO ANALYZE:
{system_prompt}

USE CASE: {use_case or "Not specified"}
REQUIREMENTS: {requirements or "Not specified"}

Your task: Suggest 6-8 specific, actionable evaluation aspects that are:
1. Directly relevant to THIS specific system prompt
2. Concrete and measurable
3. Covering different quality dimensions
4. Prioritized by importance for this use case

Focus on what THIS SPECIFIC AI needs to be evaluated on, not generic criteria.

Respond in JSON format:
{{
  "domain": "the domain/industry (e.g., 'Healthcare', 'Customer Service', 'Code Assistant')",
  "risk_level": "low/medium/high",
  "aspects": [
    "First specific aspect to evaluate",
    "Second specific aspect to evaluate",
    ...
  ],
  "reasoning": "Brief explanation of why these aspects are critical for THIS system"
}}

Focus on aspects like: accuracy, safety, tone, formatting, domain-specific requirements, compliance, etc. - but TAILORED to this specific system prompt."""

    try:
        # Call LLM
        response = await llm_client.generate(
            prompt=suggestion_prompt,
            provider="anthropic",
            api_key=api_key,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.3
        )

        # Parse JSON response
        import json
        import re

        # Extract JSON from response (in case there's extra text)
        json_match = re.search(r'\{.*\}', response.strip(), re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            # Fallback if JSON parsing fails
            return {
                "domain": "General",
                "risk_level": "medium",
                "aspects": [
                    "Accuracy and factual correctness",
                    "Tone and professionalism",
                    "Clarity and conciseness",
                    "Response completeness",
                    "Following instructions precisely",
                    "Handling edge cases appropriately"
                ],
                "reasoning": "Generic suggestions (LLM response parsing failed)"
            }

    except Exception as e:
        logger.error(f"Error in LLM-based aspect suggestion: {e}")
        # Fallback to generic suggestions
        return {
            "domain": "General",
            "risk_level": "medium",
            "aspects": [
                "Accuracy and factual correctness",
                "Tone and professionalism",
                "Clarity and conciseness",
                "Response completeness",
                "Following instructions precisely",
                "Handling edge cases appropriately"
            ],
            "reasoning": "Generic suggestions (LLM call failed)"
        }


def _get_generic_suggestions() -> SuggestAspectsResponse:
    """Fallback generic suggestions when extraction fails"""
    return SuggestAspectsResponse(
        suggested_aspects=[
            "Accuracy and factual correctness",
            "Tone and professionalism",
            "Clarity and conciseness",
            "Response completeness",
            "Following instructions precisely"
        ],
        domain="General",
        risk_level="unknown",
        reasoning="Generic suggestions provided (system prompt analysis unavailable)"
    )


# ============================================================================
# BACKWARD COMPATIBILITY ENDPOINTS
# ============================================================================

@router.get("/{project_id}/multi-eval/separate-prompts")
async def get_separate_eval_prompts_compat(project_id: str):
    """
    DEPRECATED: Use POST /eval/generate with mode="multi" instead.

    This endpoint is kept for backward compatibility.
    """

    logger.warning("DEPRECATED endpoint called: /multi-eval/separate-prompts. Use /eval/generate instead.")

    # Call the new unified endpoint
    request = GenerateEvalRequest(mode="multi")
    response = await generate_eval_prompt(project_id, request)

    # Format as old response format
    return {
        "total_evaluators": len(response.evaluators) if response.evaluators else 0,
        "evaluators": response.evaluators or [],
        "extraction_metadata": response.extraction_metadata
    }


@router.post("/{project_id}/multi-eval/test-extraction")
async def test_extraction_compat(project_id: str):
    """
    DEPRECATED: Use POST /eval/test-extraction instead.

    This endpoint is kept for backward compatibility.
    """

    logger.warning("DEPRECATED endpoint called: /multi-eval/test-extraction. Use /eval/test-extraction instead.")

    return await test_intelligent_extraction(project_id)
