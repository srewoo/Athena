"""
Clean FastAPI server for Athena - 5-Step Prompt Testing Workflow

Enhanced with:
- SQLite database instead of file storage
- Improved LLM client with retry logic
- Best-in-class evaluator prompt generation
- Structured logging
- Rate limiting
- Proper error handling
"""
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Load .env from root directory (parent of backend/)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

from fastapi import FastAPI, HTTPException, Body, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import json
import re
import uuid
from datetime import datetime

from models import (
    ProjectInput,
    PromptOptimizationResult,
    EvaluationPromptResult,
    TestDataResult,
    TestExecutionResult,
    FinalReport
)

# Use new enhanced components
from llm_client_v2 import get_llm_client, parse_json_response, EnhancedLLMClient
from shared_settings import settings_store, get_settings as get_llm_settings_shared, update_settings
import project_api
import database as db
from prompt_analyzer import analyze_prompt, analysis_to_dict, PromptType
from prompt_analyzer_v2 import analyze_prompt_hybrid, enhanced_analysis_to_dict, analyze_prompt_quick
from agentic_rewrite import agentic_rewrite, result_to_dict as agentic_result_to_dict, get_thinking_model_for_provider
from agentic_eval import agentic_eval_generation, result_to_dict as agentic_eval_result_to_dict
from eval_generator_v2 import generate_best_eval_prompt
from eval_generator_v3 import generate_gold_standard_eval_prompt
from eval_best_practices import apply_best_practices_check
from smart_test_generator import detect_input_type, build_input_generation_prompt, get_scenario_variations, InputType
from security import check_rate_limit, validate_api_key_format, mask_api_key, generate_request_id
from logging_config import (
    setup_logging, get_logger, set_request_id, get_request_id,
    log_performance, metrics
)
import quality_api
from eval_quality_system import get_quality_manager, build_compact_eval_prompt

# Setup structured logging
log_level = os.getenv("LOG_LEVEL", "INFO")
json_logs = os.getenv("JSON_LOGS", "false").lower() == "true"
setup_logging(level=log_level, json_format=json_logs)

logger = get_logger(__name__)

# Cleanup configuration
CLEANUP_INTERVAL_HOURS = 24  # Run cleanup every 24 hours
CLEANUP_AGE_DAYS = 30  # Delete projects older than 30 days


async def cleanup_scheduler():
    """Background task to periodically clean up old projects"""
    while True:
        try:
            # Wait for the specified interval
            await asyncio.sleep(CLEANUP_INTERVAL_HOURS * 3600)

            # Run cleanup using database
            logger.info("Running scheduled project cleanup...")
            result = db.cleanup_old_projects(days=CLEANUP_AGE_DAYS)
            logger.info(f"Cleanup completed: {result['deleted_count']} projects deleted")

        except asyncio.CancelledError:
            logger.info("Cleanup scheduler stopped")
            break
        except Exception as e:
            logger.error(f"Error in cleanup scheduler: {e}")
            # Continue running even if there's an error
            await asyncio.sleep(3600)  # Wait an hour before retrying


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - start/stop background tasks"""
    # Startup: Initialize database and start the cleanup scheduler
    logger.info("Initializing database...")
    db.init_database()

    cleanup_task = asyncio.create_task(cleanup_scheduler())
    logger.info("Started project cleanup scheduler (runs every 24 hours, deletes projects older than 30 days)")

    yield

    # Shutdown: Cancel the cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("Cleanup scheduler stopped")


app = FastAPI(title="Athena - Prompt Testing Application", lifespan=lifespan)

# CORS middleware - load origins from env
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
)


# Request middleware for logging and rate limiting
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Add request ID, logging, and rate limiting"""
    # Generate and set request ID
    request_id = request.headers.get("X-Request-ID") or generate_request_id()
    set_request_id(request_id)

    # Get client IP for rate limiting
    client_ip = request.client.host if request.client else "unknown"

    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")

    # Check rate limit for LLM endpoints
    if request.url.path.startswith("/api/step"):
        allowed, remaining, reset = check_rate_limit(client_ip, "llm_call")
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded", "reset_seconds": reset},
                headers={"X-RateLimit-Reset": str(reset)}
            )

    start_time = datetime.now()
    try:
        response = await call_next(request)
        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms}ms"

        # Log response
        logger.info(f"Response: {response.status_code} in {duration_ms}ms")

        # Record metrics
        metrics.increment(f"requests.{request.method}")
        metrics.record_timing("response_time", duration_ms)

        return response

    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise


# Initialize LLM client
llm_client = get_llm_client()

# Include project management router
app.include_router(project_api.router)

# Include quality API router for eval quality features
app.include_router(quality_api.router)


# NOTE: The following endpoints were removed as they are not used by the UI:
#   - GET / (root endpoint)
#   - GET /api/admin/storage-stats
#   - GET /api/admin/metrics
#   - POST /api/admin/cleanup
#   - POST /api/step1/validate


# ============================================================================
# STEP 2: PROMPT OPTIMIZATION
# ============================================================================
# NOTE: The following endpoints were removed as duplicates:
#   - /api/step2/analyze -> use /api/projects/{id}/analyze instead
#   - /api/step2/analyze-enhanced -> use /api/projects/{id}/analyze instead
#   - /api/step2/optimize -> use /api/projects/{id}/rewrite instead
#   - /api/step2/refine -> use /api/projects/{id}/rewrite instead
# Only /api/step2/agentic-rewrite is kept as it's used by the frontend


def build_dna_preservation_instructions(analysis) -> str:
    """Build instructions for preserving prompt DNA"""
    instructions = ["## CRITICAL: DNA PRESERVATION RULES"]

    dna = analysis.dna

    if dna.template_variables:
        instructions.append(f"\n### Template Variables (MUST PRESERVE EXACTLY)")
        for var in dna.template_variables:
            instructions.append(f"- `{{{{{var}}}}}` - DO NOT modify, rename, or remove")

    if dna.output_format:
        instructions.append(f"\n### Output Format (MUST PRESERVE)")
        instructions.append(f"- Format type: {dna.output_format}")
        if dna.output_schema:
            instructions.append(f"- Schema structure must remain compatible")

    if dna.scoring_scale:
        instructions.append(f"\n### Scoring Scale (MUST PRESERVE)")
        scale = dna.scoring_scale
        if scale.get('type') == 'numeric':
            instructions.append(f"- Range: {scale.get('min', 0)}-{scale.get('max', 10)} (DO NOT change)")
        elif scale.get('type') == 'categorical':
            instructions.append(f"- Categories: {', '.join(scale.get('categories', []))}")

    if dna.key_terminology:
        instructions.append(f"\n### Key Terminology (PRESERVE)")
        for term in dna.key_terminology[:10]:
            instructions.append(f"- \"{term}\"")

    if dna.sections:
        instructions.append(f"\n### Existing Sections (PRESERVE STRUCTURE)")
        for section in dna.sections[:10]:
            instructions.append(f"- {section}")

    return "\n".join(instructions)


def build_type_specific_guidance(prompt_type: PromptType) -> str:
    """Build optimization guidance specific to the prompt type"""
    guidance = {
        PromptType.ANALYTICAL: """
## TYPE-SPECIFIC GUIDANCE (Analytical/Scoring Prompt)
- Ensure scoring rubric is explicit and detailed
- Each score level should have clear criteria
- Include examples of what constitutes each score
- Add instructions for handling ambiguous cases
- Ensure evidence-based reasoning is required""",

        PromptType.STRUCTURED_OUTPUT: """
## TYPE-SPECIFIC GUIDANCE (Structured Output Prompt)
- Output schema must be explicit and complete
- Include field-by-field specifications
- Add validation rules for each field
- Specify data types clearly
- Include an example of valid output""",

        PromptType.CONVERSATIONAL: """
## TYPE-SPECIFIC GUIDANCE (Conversational Prompt)
- Define tone and personality clearly
- Include examples of good responses
- Add safety and boundary guidelines
- Specify how to handle edge cases
- Define what topics are in/out of scope""",

        PromptType.CREATIVE: """
## TYPE-SPECIFIC GUIDANCE (Creative Prompt)
- Define style and tone requirements
- Include constraints (length, format, etc.)
- Add examples of desired output quality
- Specify any themes or elements to include/avoid""",

        PromptType.EXTRACTION: """
## TYPE-SPECIFIC GUIDANCE (Extraction Prompt)
- Define exactly what to extract
- Specify output format for extracted data
- Add rules for handling missing data
- Include examples of input/output pairs""",

        PromptType.INSTRUCTIONAL: """
## TYPE-SPECIFIC GUIDANCE (Instructional Prompt)
- Ensure step-by-step clarity
- Add numbered steps where appropriate
- Include prerequisites or assumptions
- Add tips for common mistakes""",

        PromptType.HYBRID: """
## TYPE-SPECIFIC GUIDANCE (Hybrid Prompt)
- Balance all identified prompt types
- Ensure clear transitions between modes
- Maintain consistency across sections"""
    }

    return guidance.get(prompt_type, guidance[PromptType.HYBRID])


def validate_dna_preservation(original: str, optimized: str, analysis) -> List[str]:
    """Validate that DNA elements were preserved in the optimized prompt"""
    issues = []
    dna = analysis.dna

    # Check template variables
    for var in dna.template_variables:
        patterns = [f"{{{{{var}}}}}", f"{{{var}}}", f"<<{var}>>"]
        found = any(pattern in optimized for pattern in patterns)
        if not found:
            issues.append(f"Template variable '{var}' may be missing")

    # Check scoring scale preservation (if numeric)
    if dna.scoring_scale and dna.scoring_scale.get('type') == 'numeric':
        min_val = str(dna.scoring_scale.get('min', 0))
        max_val = str(dna.scoring_scale.get('max', 10))
        scale_pattern = f"{min_val}.*{max_val}|{min_val}-{max_val}"
        if not re.search(scale_pattern, optimized):
            # Check if the scale is mentioned in any form
            if min_val not in optimized or max_val not in optimized:
                issues.append(f"Scoring scale {min_val}-{max_val} may be missing")

    return issues


def build_eval_dimensions_text(analysis) -> str:
    """Build text describing recommended evaluation dimensions based on prompt type"""
    dimensions = analysis.suggested_eval_dimensions
    if not dimensions:
        return "Use standard evaluation dimensions: Format Compliance, Accuracy, Completeness, Clarity"

    lines = []
    for i, dim in enumerate(dimensions, 1):
        lines.append(f"{i}. **{dim['name']}**")
        lines.append(f"   - Description: {dim['description']}")
        lines.append(f"   - Check: {dim['check']}")

    return "\n".join(lines)


def build_test_categories_text(analysis) -> str:
    """Build text describing recommended test categories based on prompt type"""
    categories = analysis.suggested_test_categories
    if not categories:
        return "Use standard distribution: 50% positive, 20% edge cases, 15% negative, 15% adversarial"

    lines = []
    for cat in categories:
        lines.append(f"### {cat['name'].upper()} ({cat['percentage']}%)")
        lines.append(f"Description: {cat['description']}")
        if 'examples' in cat:
            lines.append("Examples:")
            for ex in cat['examples'][:3]:
                lines.append(f"  - {ex}")
        lines.append("")

    return "\n".join(lines)


@app.post("/api/step2/agentic-rewrite")
async def agentic_rewrite_optimization(project: ProjectInput, current_result: dict, use_thinking_model: bool = True):
    """
    Step 2 Agentic Rewrite: Multi-step, self-correcting prompt optimization.

    This is an enhanced version of AI Rewrite that uses:
    1. Deep analysis with thinking model (o3-mini, etc.)
    2. Structured improvement planning
    3. Careful execution with DNA preservation
    4. Validation and iteration loop

    NOTE: This endpoint is used for ad-hoc prompt optimization with provided API keys.
    For project-based rewriting, use POST /api/projects/{id}/rewrite which uses stored
    project context and saved settings.

    Parameters:
    - project: Project configuration including API keys
    - current_result: Current optimization result with optimized_prompt
    - use_thinking_model: Whether to use a thinking model for analysis (default: True)
    """
    current_prompt = current_result.get('optimized_prompt', '')
    if not current_prompt:
        raise HTTPException(status_code=400, detail="optimized_prompt is required in current_result")

    # Extract analysis context from current_result (from Re-Analyze feedback)
    analysis_context = current_result.get('analysis_context', None)

    # Determine thinking model to use
    thinking_model = None
    if use_thinking_model:
        thinking_model = get_thinking_model_for_provider(project.provider)
        logger.info(f"Using thinking model: {thinking_model} for deep analysis")

    if analysis_context:
        logger.info(f"Incorporating analysis context: {len(analysis_context.get('suggestions', []))} suggestions, {len(analysis_context.get('missing_requirements', []))} missing requirements")

    try:
        result = await agentic_rewrite(
            prompt=current_prompt,
            use_case=project.use_case,
            requirements=project.requirements,
            llm_client=llm_client,
            provider=project.provider,
            api_key=project.api_key,
            model_name=project.model_name,
            thinking_model=thinking_model,
            max_iterations=2,
            user_analysis_context=analysis_context
        )

        # Log key details for debugging
        prompt_changed = result.original_prompt != result.final_prompt
        logger.info(f"Agentic rewrite result: no_change={result.no_change}, prompt_changed={prompt_changed}, "
                    f"original_len={len(result.original_prompt)}, final_len={len(result.final_prompt)}, "
                    f"score: {result.original_score:.1f} -> {result.final_score:.1f}")

        # Convert to the expected response format
        return {
            "optimized_prompt": result.final_prompt,
            "score": result.final_score,
            "analysis": result.reason,
            "improvements": [
                f"{imp.get('area', 'Unknown')}: {imp.get('target', 'Improved')}"
                for imp in (result.improvement_plan.get("improve", []) if result.improvement_plan else [])
            ],
            "suggestions": result.improvement_plan.get("add", []) if result.improvement_plan else [],
            "no_change": result.no_change,
            "agentic_details": {
                "iterations": result.iterations,
                "steps_taken": result.steps_taken,
                "original_score": result.original_score,
                "final_score": result.final_score,
                "quality_delta": result.final_score - result.original_score,
                "validation": result.validation,
                "prompt_changed": prompt_changed
            }
        }

    except Exception as e:
        logger.error(f"Agentic rewrite failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agentic rewrite failed: {str(e)}")


# ============================================================================
# STEP 3: EVALUATION PROMPT GENERATION
# ============================================================================

@app.post("/api/step3/agentic-generate-eval")
async def agentic_generate_evaluation_prompt(
    project: ProjectInput,
    optimized_prompt: str = Body(...),
    use_thinking_model: bool = Body(True),
    use_gold_standard: bool = Body(True)
):
    """
    Step 3 Agentic Eval Generation: Gold-standard evaluation prompt generation.

    NOTE: This endpoint is used for ad-hoc eval prompt generation with provided API keys.
    For project-based generation, use POST /api/projects/{id}/eval-prompt/generate which
    uses stored project context and saved settings.

    Uses the 20-point best practices framework:
    1. Explicit Role Separation - Evaluator doesn't re-do task
    2. Clear Success Definition - System-specific, not generic
    3. Auto-Fail Conditions - Non-negotiable failures
    4. Evidence-Based Judging - Observable evidence only
    5. Strict Scope Enforcement - What NOT to judge
    6. Dimensioned Scoring - Multiple weighted dimensions
    7. Clear Rubrics per Dimension - 5/3/1 examples
    8. Grounding Verification - Claims tied to inputs
    9. Anti-Hallucination Checks - Explicit patterns
    10. Format & Schema Enforcement - First-class failures
    11. Severity Alignment Checks - Score-narrative consistency
    12. False Positive/Negative Awareness - Bias warnings
    13. Verdict Thresholds - Clear PASS/NEEDS_REVIEW/FAIL
    14. Reasoning Without Re-Solving - Outcome-focused justification
    15. Consistency Clause - Similar outputs = similar scores
    16. Known Failure Mode Section - System-specific pitfalls
    17. Evaluator Self-Check - Mandatory bias check
    18. Downstream Consumer Awareness - Who uses this
    19. Minimal but Sufficient Output - Concise explanations
    20. Calibration Examples - Score calibration

    Parameters:
    - project: Project configuration including API keys
    - optimized_prompt: The system prompt to generate eval for
    - use_thinking_model: Whether to use a thinking model for analysis (default: True)
    - use_gold_standard: Use v3 gold-standard generator (default: True)
    """
    if not optimized_prompt:
        raise HTTPException(status_code=400, detail="optimized_prompt is required")

    # Validate API key
    is_valid, error_msg = validate_api_key_format(project.api_key, project.provider)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    # Determine thinking model to use
    thinking_model = None
    if use_thinking_model:
        thinking_model = get_thinking_model_for_provider(project.provider)
        logger.info(f"Using thinking model: {thinking_model} for eval generation")

    try:
        # Use gold-standard v3 generator by default
        if use_gold_standard:
            result = await generate_gold_standard_eval_prompt(
                system_prompt=optimized_prompt,
                use_case=project.use_case,
                requirements=project.requirements,
                llm_client=llm_client,
                provider=project.provider,
                api_key=project.api_key,
                model_name=project.model_name,
                thinking_model=thinking_model,
                max_iterations=2
            )
        else:
            # Fallback to v2
            result = await generate_best_eval_prompt(
                system_prompt=optimized_prompt,
                use_case=project.use_case,
                requirements=project.requirements,
                llm_client=llm_client,
                provider=project.provider,
                api_key=project.api_key,
                model_name=project.model_name,
                thinking_model=thinking_model,
                max_iterations=2
            )

        # Record metrics
        metrics.increment("eval_prompts_generated")

        # Run Anthropic best practices check for consistency with project-based flow
        best_practices_report = apply_best_practices_check(result.eval_prompt)

        # Enhance rationale with best practices score
        enhanced_rationale = f"{result.rationale} Anthropic best practices score: {best_practices_report['score']}/100."

        return {
            "eval_prompt": result.eval_prompt,
            "eval_criteria": result.eval_criteria,
            "rationale": enhanced_rationale,
            "agentic_details": {
                "failure_modes": result.failure_modes,
                "eval_dimensions": result.dimensions,
                "calibration_examples": result.calibration_examples,
                "self_test": result.self_test_results,
                "metadata": result.metadata,
                "best_practices": best_practices_report
            }
        }

    except Exception as e:
        logger.error(f"Agentic eval generation failed: {str(e)}")
        metrics.increment("eval_prompts_failed")
        raise HTTPException(status_code=500, detail=f"Agentic eval generation failed: {str(e)}")


# NOTE: /api/step3/generate-eval-v2 was removed - use /api/step3/agentic-generate-eval instead


# NOTE: /api/step3/refine was removed - use /api/projects/{id}/eval-prompt/refine instead


# ============================================================================
# STEP 4: TEST DATA GENERATION
# ============================================================================

# NOTE: /api/step4/generate-testdata was removed - use /api/projects/{id}/test-cases/generate instead


# NOTE: /api/step4/refine was removed - use /api/projects/{id}/test-cases/refine instead


# NOTE: /api/step4/smart-generate-testdata was removed - use /api/projects/{id}/test-cases/generate instead


# ============================================================================
# STEP 5: TEST EXECUTION & REPORTING
# ============================================================================

# NOTE: /api/step5/execute-tests was removed - use /api/projects/{id}/test-cases/execute instead


# NOTE: POST /api/generate-report was removed as it is not used by the UI


# ============================================================================
# REWRITE API - Used by EvaluationDetail.js
# ============================================================================

@app.post("/api/rewrite")
async def rewrite_prompt(data: dict):
    """
    Rewrite/improve a prompt based on evaluation feedback.
    Used by EvaluationDetail.js to improve prompts after evaluation.
    """
    prompt_text = data.get("prompt_text", "")
    suggestions = data.get("suggestions", [])
    evaluation_id = data.get("evaluation_id", "")

    if not prompt_text:
        raise HTTPException(status_code=400, detail="prompt_text is required")

    # Get LLM settings
    settings = get_llm_settings()
    provider = settings["provider"]
    api_key = settings["api_key"]
    model_name = settings["model_name"]

    if not api_key:
        raise HTTPException(status_code=400, detail="API key not configured. Please configure settings first.")

    # Build improvement context from suggestions
    improvements_context = ""
    if suggestions:
        improvements_context = "\n\nSuggested improvements to incorporate:\n"
        for i, sug in enumerate(suggestions[:5], 1):
            if isinstance(sug, dict):
                improvements_context += f"{i}. [{sug.get('priority', 'Medium')}] {sug.get('suggestion', sug.get('text', str(sug)))}\n"
            else:
                improvements_context += f"{i}. {sug}\n"

    system_prompt = """You are an expert prompt engineer. Your task is to improve the given prompt while preserving its core intent and any template variables.

RULES:
1. Preserve ALL template variables exactly as written (e.g., {{variable_name}}, {variable}, <<var>>)
2. Improve clarity, structure, and specificity
3. Add appropriate delimiters and sections
4. Ensure the output format is clearly specified
5. Add edge case handling if missing
6. Maintain the original task/goal

Return ONLY the improved prompt text, no explanations or meta-commentary."""

    user_message = f"""Improve this prompt:

{prompt_text}
{improvements_context}

Return the improved prompt:"""

    result = await llm_client.chat(
        system_prompt=system_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.7,
        max_tokens=8000
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    rewritten_prompt = result.get("output", "").strip()

    # Clean up any markdown code blocks if present
    if rewritten_prompt.startswith("```"):
        lines = rewritten_prompt.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        rewritten_prompt = "\n".join(lines)

    return {
        "original_prompt": prompt_text,
        "rewritten_prompt": rewritten_prompt,
        "evaluation_id": evaluation_id,
        "improvements_applied": len(suggestions) if suggestions else 0,
        "provider": provider,
        "model": model_name or "default"
    }


# ============================================================================
# EVALUATIONS API (for Dashboard, History, Compare pages)
# ============================================================================

# In-memory evaluations storage
evaluations_store = {}


@app.post("/api/evaluate")
async def evaluate_prompt(data: dict):
    """
    Evaluate a prompt using LLM - used by Dashboard.js.

    NOTE: This endpoint evaluates a prompt against general quality criteria.
    For project-based analysis that considers project context, requirements, and
    provides improvement suggestions, use POST /api/projects/{id}/analyze.
    """
    prompt_text = data.get("prompt_text", "")
    evaluation_mode = data.get("evaluation_mode", "quick")

    if not prompt_text:
        raise HTTPException(status_code=400, detail="prompt_text is required")

    # Get LLM settings
    settings = get_llm_settings()
    provider = settings["provider"]
    api_key = settings["api_key"]
    model_name = settings["model_name"]

    if not api_key:
        raise HTTPException(status_code=400, detail="API key not configured. Please configure settings first.")

    word_count = len(prompt_text.split())

    # Use LLM for actual evaluation
    system_prompt = """You are a prompt engineering expert. Evaluate the given prompt and return a JSON object with:

{
  "categories": {
    "clarity": {"score": 1-5, "feedback": "specific feedback"},
    "specificity": {"score": 1-5, "feedback": "specific feedback"},
    "context": {"score": 1-5, "feedback": "specific feedback"},
    "examples": {"score": 1-5, "feedback": "specific feedback"},
    "constraints": {"score": 1-5, "feedback": "specific feedback"}
  },
  "suggestions": [
    {"priority": "High/Medium/Low", "suggestion": "specific improvement"}
  ],
  "overall_assessment": "brief summary"
}

Score each category 1-5:
- clarity: Is the prompt clear and unambiguous?
- specificity: Does it provide enough detail?
- context: Does it set appropriate context/role?
- examples: Does it include helpful examples?
- constraints: Does it define boundaries/constraints?

Return ONLY valid JSON, no other text."""

    result = await llm_client.chat(
        system_prompt=system_prompt,
        user_message=f"Evaluate this prompt:\n\n{prompt_text}",
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.3,
        max_tokens=8000
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    # Parse LLM response
    try:
        output = result.get("output", "{}")
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0]
        elif "```" in output:
            output = output.split("```")[1].split("```")[0]
        eval_data = json.loads(output.strip())

        categories = eval_data.get("categories", {})
        suggestions = eval_data.get("suggestions", [])
        overall = eval_data.get("overall_assessment", "")
    except:
        # Fallback to basic analysis if parsing fails
        categories = {
            "clarity": {"score": 3, "feedback": "Unable to parse detailed feedback"},
            "specificity": {"score": 3, "feedback": "Unable to parse detailed feedback"},
            "context": {"score": 3, "feedback": "Unable to parse detailed feedback"},
            "examples": {"score": 3, "feedback": "Unable to parse detailed feedback"},
            "constraints": {"score": 3, "feedback": "Unable to parse detailed feedback"}
        }
        suggestions = [{"priority": "Medium", "suggestion": "Review prompt for improvements"}]
        overall = result.get("output", "Evaluation completed")

    # Calculate total score (each category max 5, scaled to 250 max)
    total_score = sum(c.get("score", 3) * 10 for c in categories.values())
    max_score = 250

    # Create evaluation record with frontend-expected field names
    eval_id = str(uuid.uuid4())
    evaluation = {
        "id": eval_id,
        "prompt_text": prompt_text,
        "evaluation_mode": evaluation_mode,
        "total_score": round(total_score, 1),
        "max_score": max_score,
        "categories": categories,
        "refinement_suggestions": suggestions,  # Frontend expects this field name
        "overall_assessment": overall,
        "llm_provider": provider,
        "created_at": datetime.now().isoformat(),
        "word_count": word_count
    }

    # Store evaluation
    evaluations_store[eval_id] = evaluation

    return evaluation


@app.get("/api/evaluations")
async def list_evaluations(limit: int = 100):
    """List all evaluations - used by History.js and Compare.js"""
    evals = list(evaluations_store.values())
    evals.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return evals[:limit]


@app.get("/api/evaluations/{eval_id}")
async def get_evaluation(eval_id: str):
    """Get a specific evaluation - used by EvaluationDetail.js"""
    if eval_id not in evaluations_store:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return evaluations_store[eval_id]


@app.delete("/api/evaluations/{eval_id}")
async def delete_evaluation(eval_id: str):
    """Delete an evaluation - used by History.js"""
    if eval_id not in evaluations_store:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    del evaluations_store[eval_id]
    return {"message": "Evaluation deleted successfully"}


@app.post("/api/compare")
async def compare_evaluations(data: dict):
    """Compare multiple evaluations - used by Compare.js"""
    evaluation_ids = data.get("evaluation_ids", [])

    if len(evaluation_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 evaluations required for comparison")

    comparisons = []
    for eval_id in evaluation_ids:
        if eval_id in evaluations_store:
            ev = evaluations_store[eval_id]
            comparisons.append({
                "id": eval_id,
                "prompt_text": ev.get("prompt_text", "")[:100] + "...",
                "total_score": ev.get("total_score", 0),
                "max_score": ev.get("max_score", 250),
                "categories": ev.get("categories", {}),
                "created_at": ev.get("created_at")
            })

    # Find best performing
    best = max(comparisons, key=lambda x: x["total_score"]) if comparisons else None

    return {
        "comparisons": comparisons,
        "best_evaluation_id": best["id"] if best else None,
        "summary": f"Compared {len(comparisons)} evaluations"
    }


# ============================================================================
# PLAYGROUND API
# ============================================================================

@app.post("/api/playground")
async def playground_test(data: dict):
    """Quick test a prompt with input - used by Playground.js"""
    prompt_text = data.get("prompt_text", "")
    test_input = data.get("test_input", "")

    if not prompt_text:
        raise HTTPException(status_code=400, detail="prompt_text is required")

    # Get LLM settings
    settings = get_llm_settings()
    provider = settings["provider"]
    api_key = settings["api_key"]
    model_name = settings["model_name"]

    if not api_key:
        raise HTTPException(status_code=400, detail="API key not configured")

    # Call LLM
    user_message = test_input if test_input else "Hello, please demonstrate your capabilities."

    result = await llm_client.chat(
        system_prompt=prompt_text,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.7,
        max_tokens=8000
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return {
        "response": result.get("output", ""),  # Frontend expects 'response' not 'output'
        "output": result.get("output", ""),    # Keep for backward compatibility
        "provider": provider,
        "model": model_name or "default",
        "latency_ms": result.get("latency_ms", 0),
        "tokens_used": result.get("tokens_used", 0)
    }


# ============================================================================
# A/B TESTING API
# ============================================================================

@app.post("/api/ab-test")
async def ab_test(data: dict):
    """
    A/B test two prompts with quality evaluation - used by ABTesting.js

    NOTE: This is the canonical A/B testing endpoint. It compares two prompts directly
    with provided test inputs. Project-based A/B testing endpoints have been removed.

    Enhanced to include:
    - Quality scoring for each output
    - Statistical significance testing
    - Detailed comparison metrics
    """
    prompt_a = data.get("prompt_a", "")
    prompt_b = data.get("prompt_b", "")
    test_inputs = data.get("test_inputs", ["Test input 1", "Test input 2", "Test input 3"])
    eval_prompt = data.get("eval_prompt", "")  # Optional eval prompt for quality scoring
    use_case = data.get("use_case", "General task")

    if not prompt_a or not prompt_b:
        raise HTTPException(status_code=400, detail="Both prompt_a and prompt_b are required")

    # Get LLM settings
    settings = get_llm_settings()
    provider = settings["provider"]
    api_key = settings["api_key"]
    model_name = settings["model_name"]

    if not api_key:
        raise HTTPException(status_code=400, detail="API key not configured")

    results_a = []
    results_b = []

    # Default eval prompt if not provided
    if not eval_prompt:
        eval_prompt = f"""Evaluate this output for the task: {use_case}

Rate the output on a scale of 1-5 where:
- 5: Excellent - Fully addresses the request with high quality
- 4: Good - Addresses the request well with minor issues
- 3: Acceptable - Addresses the request but has notable weaknesses
- 2: Poor - Partially addresses the request with significant issues
- 1: Unacceptable - Does not address the request or has major problems

Return JSON:
{{"score": <1-5>, "reasoning": "<brief explanation>"}}"""

    for test_input in test_inputs[:10]:  # Allow up to 10 tests
        # Test prompt A
        result_a = await llm_client.chat(
            system_prompt=prompt_a,
            user_message=test_input,
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            temperature=0.7,
            max_tokens=8000
        )

        output_a = result_a.get("output", "") if not result_a.get("error") else ""
        score_a = 0
        reasoning_a = ""

        # Evaluate output A
        if output_a and not result_a.get("error"):
            eval_result_a = await llm_client.chat(
                system_prompt=eval_prompt,
                user_message=f"Input: {test_input}\n\nOutput to evaluate:\n{output_a}",
                provider=provider,
                api_key=api_key,
                model_name=model_name,
                temperature=0.1,
                max_tokens=500
            )
            if not eval_result_a.get("error"):
                parsed = parse_json_response(eval_result_a.get("output", ""), "object")
                if parsed:
                    score_a = parsed.get("score", 0)
                    reasoning_a = parsed.get("reasoning", "")

        results_a.append({
            "input": test_input,
            "output": output_a or "Error",
            "latency_ms": result_a.get("latency_ms", 0),
            "tokens_used": result_a.get("tokens_used", 0),
            "quality_score": score_a,
            "quality_reasoning": reasoning_a,
            "error": result_a.get("error")
        })

        # Test prompt B
        result_b = await llm_client.chat(
            system_prompt=prompt_b,
            user_message=test_input,
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            temperature=0.7,
            max_tokens=8000
        )

        output_b = result_b.get("output", "") if not result_b.get("error") else ""
        score_b = 0
        reasoning_b = ""

        # Evaluate output B
        if output_b and not result_b.get("error"):
            eval_result_b = await llm_client.chat(
                system_prompt=eval_prompt,
                user_message=f"Input: {test_input}\n\nOutput to evaluate:\n{output_b}",
                provider=provider,
                api_key=api_key,
                model_name=model_name,
                temperature=0.1,
                max_tokens=500
            )
            if not eval_result_b.get("error"):
                parsed = parse_json_response(eval_result_b.get("output", ""), "object")
                if parsed:
                    score_b = parsed.get("score", 0)
                    reasoning_b = parsed.get("reasoning", "")

        results_b.append({
            "input": test_input,
            "output": output_b or "Error",
            "latency_ms": result_b.get("latency_ms", 0),
            "tokens_used": result_b.get("tokens_used", 0),
            "quality_score": score_b,
            "quality_reasoning": reasoning_b,
            "error": result_b.get("error")
        })

    # Calculate metrics
    valid_a = [r for r in results_a if r["quality_score"] > 0]
    valid_b = [r for r in results_b if r["quality_score"] > 0]

    avg_latency_a = sum(r["latency_ms"] for r in results_a) / len(results_a) if results_a else 0
    avg_latency_b = sum(r["latency_ms"] for r in results_b) / len(results_b) if results_b else 0

    avg_quality_a = sum(r["quality_score"] for r in valid_a) / len(valid_a) if valid_a else 0
    avg_quality_b = sum(r["quality_score"] for r in valid_b) / len(valid_b) if valid_b else 0

    avg_tokens_a = sum(r["tokens_used"] for r in results_a) / len(results_a) if results_a else 0
    avg_tokens_b = sum(r["tokens_used"] for r in results_b) / len(results_b) if results_b else 0

    # Calculate statistical significance (simple t-test approximation)
    def calculate_stats(scores):
        if len(scores) < 2:
            return {"mean": 0, "std": 0, "n": 0}
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / (len(scores) - 1) if len(scores) > 1 else 0
        return {"mean": mean, "std": variance ** 0.5, "n": len(scores)}

    stats_a = calculate_stats([r["quality_score"] for r in valid_a])
    stats_b = calculate_stats([r["quality_score"] for r in valid_b])

    # Determine winner based on quality (primary) and latency (secondary)
    quality_diff = avg_quality_a - avg_quality_b
    if abs(quality_diff) >= 0.5:
        winner = "A" if quality_diff > 0 else "B"
        winner_reason = f"Higher quality score ({max(avg_quality_a, avg_quality_b):.2f} vs {min(avg_quality_a, avg_quality_b):.2f})"
    elif abs(avg_latency_a - avg_latency_b) > 100:
        winner = "A" if avg_latency_a < avg_latency_b else "B"
        winner_reason = f"Lower latency ({min(avg_latency_a, avg_latency_b):.0f}ms vs {max(avg_latency_a, avg_latency_b):.0f}ms)"
    else:
        winner = "TIE"
        winner_reason = "No significant difference in quality or latency"

    # Calculate effect size (Cohen's d)
    pooled_std = ((stats_a["std"]**2 + stats_b["std"]**2) / 2) ** 0.5 if stats_a["std"] or stats_b["std"] else 1
    effect_size = quality_diff / pooled_std if pooled_std > 0 else 0

    return {
        "prompt_a_results": results_a,
        "prompt_b_results": results_b,
        "summary": {
            "tests_run": len(test_inputs[:10]),
            "prompt_a": {
                "avg_latency_ms": round(avg_latency_a, 2),
                "avg_quality_score": round(avg_quality_a, 2),
                "avg_tokens": round(avg_tokens_a, 0),
                "valid_tests": len(valid_a),
                "stats": stats_a
            },
            "prompt_b": {
                "avg_latency_ms": round(avg_latency_b, 2),
                "avg_quality_score": round(avg_quality_b, 2),
                "avg_tokens": round(avg_tokens_b, 0),
                "valid_tests": len(valid_b),
                "stats": stats_b
            },
            "comparison": {
                "winner": winner,
                "winner_reason": winner_reason,
                "quality_difference": round(quality_diff, 2),
                "latency_difference_ms": round(avg_latency_a - avg_latency_b, 2),
                "effect_size": round(effect_size, 3),
                "is_significant": abs(effect_size) >= 0.5  # Medium effect size
            }
        }
    }


# ============================================================================
# EXPORT API
# ============================================================================

@app.get("/api/export/json/{eval_id}")
async def export_json(eval_id: str):
    """Export evaluation as JSON - used by EvaluationDetail.js"""
    if eval_id not in evaluations_store:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    evaluation = evaluations_store[eval_id]
    return {
        "format": "json",
        "data": evaluation,
        "exported_at": datetime.now().isoformat()
    }


@app.get("/api/export/pdf/{eval_id}")
async def export_pdf(eval_id: str):
    """Export evaluation as PDF data - used by Dashboard.js, History.js"""
    if eval_id not in evaluations_store:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    evaluation = evaluations_store[eval_id]
    # Return structured data that frontend can render as PDF
    return {
        "format": "pdf",
        "title": f"Prompt Evaluation Report - {eval_id[:8]}",
        "generated_at": datetime.now().isoformat(),
        "content": {
            "prompt": evaluation.get("prompt_text", ""),
            "total_score": evaluation.get("total_score", 0),
            "max_score": evaluation.get("max_score", 250),
            "categories": evaluation.get("categories", {}),
            "suggestions": evaluation.get("suggestions", [])
        }
    }


# ============================================================================
# PROMPT TOOLS API (Contradiction, Metaprompt, Delimiter)
# ============================================================================

@app.post("/api/detect-contradictions")
async def detect_contradictions(data: dict):
    """Detect contradictions in a prompt - used by ContradictionDetector.js"""
    prompt_text = data.get("prompt_text", "")

    if not prompt_text:
        raise HTTPException(status_code=400, detail="prompt_text is required")

    # Get LLM settings
    settings = get_llm_settings()
    provider = settings["provider"]
    api_key = settings["api_key"]
    model_name = settings["model_name"]

    if not api_key:
        # Return template analysis if no API key
        return {
            "contradictions": [],
            "analysis": "Configure API key for AI-powered contradiction detection",
            "severity": "unknown"
        }

    # Use LLM to detect contradictions
    system_prompt = """Analyze the following prompt for logical contradictions, conflicting instructions, or ambiguous statements that could confuse an AI.

Return a JSON object with:
- contradictions: array of {statement1, statement2, explanation}
- severity: "none", "low", "medium", "high"
- suggestions: array of strings with fixes

Return ONLY valid JSON."""

    result = await llm_client.chat(
        system_prompt=system_prompt,
        user_message=f"Analyze this prompt:\n\n{prompt_text}",
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.3,
        max_tokens=8000
    )

    if result.get("error"):
        return {
            "contradictions": [],
            "analysis": f"Analysis failed: {result['error']}",
            "severity": "unknown"
        }

    # Parse response
    try:
        output = result.get("output", "{}")
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0]
        elif "```" in output:
            output = output.split("```")[1].split("```")[0]
        analysis = json.loads(output.strip())
        return analysis
    except:
        return {
            "contradictions": [],
            "analysis": result.get("output", "No contradictions detected"),
            "severity": "none"
        }


@app.post("/api/generate-metaprompt")
async def generate_metaprompt(data: dict):
    """Generate a metaprompt - used by MetapromptGenerator.js"""
    prompt_text = data.get("prompt_text", "")
    desired_behavior = data.get("desired_behavior", "")
    target_model = data.get("target_model", "general")

    if not prompt_text:
        raise HTTPException(status_code=400, detail="prompt_text is required")

    # Get LLM settings
    settings = get_llm_settings()
    provider = settings["provider"]
    api_key = settings["api_key"]
    model_name = settings["model_name"]

    if not api_key:
        return {
            "metaprompt": "Configure API key for AI-powered metaprompt generation",
            "explanation": "API key required"
        }

    system_prompt = f"""You are a metaprompt engineer. Create a sophisticated metaprompt that will help an AI model follow the given instructions more effectively.

Target model: {target_model}
Desired behavior: {desired_behavior or 'Follow instructions precisely'}

The metaprompt should:
1. Set clear context and role
2. Define expected behavior patterns
3. Include self-correction mechanisms
4. Specify output format requirements

Return the improved prompt directly."""

    result = await llm_client.chat(
        system_prompt=system_prompt,
        user_message=f"Original prompt:\n{prompt_text}",
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.7,
        max_tokens=8000
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return {
        "metaprompt": result.get("output", ""),
        "explanation": f"Enhanced for {target_model} with focus on {desired_behavior or 'instruction following'}"
    }


@app.post("/api/analyze-delimiters")
async def analyze_delimiters(data: dict):
    """Analyze delimiter usage in a prompt - used by DelimiterAnalyzer.js"""
    prompt_text = data.get("prompt_text", "")

    if not prompt_text:
        raise HTTPException(status_code=400, detail="prompt_text is required")

    # Analyze delimiters
    delimiters = {
        "xml_tags": len(re.findall(r'<[^>]+>', prompt_text)),
        "markdown_headers": len(re.findall(r'^#{1,6}\s', prompt_text, re.MULTILINE)),
        "triple_quotes": prompt_text.count('"""') + prompt_text.count("'''"),
        "brackets": prompt_text.count('[') + prompt_text.count(']'),
        "curly_braces": prompt_text.count('{') + prompt_text.count('}'),
        "template_vars": len(re.findall(r'\{\{[^}]+\}\}', prompt_text))
    }

    total_delimiters = sum(delimiters.values())

    # Generate recommendations
    recommendations = []
    if delimiters["xml_tags"] == 0 and len(prompt_text) > 200:
        recommendations.append("Consider using XML tags to structure long prompts")
    if delimiters["markdown_headers"] == 0 and len(prompt_text) > 150:
        recommendations.append("Add markdown headers to improve readability")
    if delimiters["template_vars"] > 0:
        recommendations.append(f"Found {delimiters['template_vars']} template variables - ensure they're documented")

    return {
        "delimiter_counts": delimiters,
        "total_delimiters": total_delimiters,
        "structure_score": min(100, total_delimiters * 10 + 20),
        "recommendations": recommendations,
        "analysis": "Well-structured" if total_delimiters > 5 else "Consider adding more structure"
    }


# ============================================================================
# SETTINGS MANAGEMENT
# ============================================================================

# Simple in-memory settings storage
# Standardized field names to match frontend:
#   - llm_provider: "openai" | "claude" | "gemini"
#   - api_key: API key string
#   - model_name: Model identifier
# Note: shared_settings is imported at the top of the file


def get_llm_settings():
    """Helper to get LLM settings with consistent field names"""
    return get_llm_settings_shared()


@app.get("/api/settings")
async def get_settings():
    """Get LLM settings - returns fields frontend expects"""
    if not settings_store:
        return {}
    # Return in frontend format
    return {
        "llm_provider": settings_store.get("llm_provider", "openai"),
        "api_key": settings_store.get("api_key", ""),
        "model_name": settings_store.get("model_name", "")
    }


@app.post("/api/settings")
async def save_settings(settings: dict):
    """Save LLM settings - accepts frontend format"""
    api_key = settings.get("api_key", "")
    logger.info(f"Saving settings: provider={settings.get('llm_provider')}, model={settings.get('model_name')}, api_key={'[SET]' if api_key else '[EMPTY]'}")

    updated = update_settings(
        llm_provider=settings.get("llm_provider", "openai"),
        api_key=api_key,
        model_name=settings.get("model_name", "")
    )

    logger.info(f"Settings saved successfully. Provider: {updated['llm_provider']}, Model: {updated['model_name']}")
    return {"message": "Settings saved successfully", "settings": {
        "llm_provider": updated["llm_provider"],
        "api_key": "[REDACTED]" if updated["api_key"] else "",
        "model_name": updated["model_name"]
    }}


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
