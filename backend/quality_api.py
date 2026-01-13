"""
Quality API - Endpoints for eval quality management

This module provides API endpoints for:
1. Cost tracking
2. Ground truth management
3. Reliability verification
4. Statistical analysis
5. Adversarial testing
6. Drift detection
7. Feedback integration
8. Comprehensive validation
"""

from fastapi import APIRouter, HTTPException, Body
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import json

from eval_quality_system import (
    get_quality_manager,
    EvalQualityManager,
    GroundTruthExample,
    build_compact_eval_prompt
)
import project_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["quality"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class GroundTruthExampleRequest(BaseModel):
    input_text: str
    output_text: str
    expected_score: float = Field(ge=1, le=5)
    expected_verdict: str = Field(pattern="^(PASS|FAIL|NEEDS_REVIEW)$")
    score_tolerance: float = Field(default=0.5, ge=0, le=2)
    category: str = "general"
    reasoning: str = ""


class HumanFeedbackRequest(BaseModel):
    test_case_id: str
    llm_score: float
    llm_verdict: str
    llm_reasoning: str
    human_score: float = Field(ge=1, le=5)
    human_verdict: str
    human_feedback: str


class ReliabilityTestRequest(BaseModel):
    input_text: str
    output_text: str
    num_runs: int = Field(default=3, ge=2, le=10)


class CompareScoresRequest(BaseModel):
    scores_a: List[float]
    scores_b: List[float]
    labels: Optional[Dict[str, str]] = None  # e.g., {"a": "Version 1", "b": "Version 2"}


class MetricsSnapshotRequest(BaseModel):
    eval_prompt_version: int
    metrics: Dict[str, float]  # avg_score, pass_rate, std_deviation
    sample_size: int


class CompactEvalRequest(BaseModel):
    system_purpose: str
    dimensions: List[Dict[str, Any]]
    auto_fails: List[Dict[str, Any]]
    input_var: str = "input"
    output_var: str = "output"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def run_eval_with_project(
    project_id: str,
    eval_prompt: str,
    input_text: str,
    output_text: str
):
    """
    Run evaluation using project's configured provider.
    Returns (score, verdict, raw_output)
    """
    from llm_client_v2 import get_llm_client

    project = project_storage.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get provider config from project
    provider = project.get("provider", "openai")
    api_key = project.get("api_key", "")
    model_name = project.get("model_name", "gpt-4o-mini")

    if not api_key:
        raise HTTPException(status_code=400, detail="Project API key not configured")

    # Substitute variables in eval prompt
    filled_prompt = eval_prompt
    filled_prompt = filled_prompt.replace("{input}", input_text)
    filled_prompt = filled_prompt.replace("{output}", output_text)
    # Also handle other common variable names
    filled_prompt = filled_prompt.replace("{INPUT}", input_text)
    filled_prompt = filled_prompt.replace("{OUTPUT}", output_text)

    llm_client = get_llm_client()
    result = await llm_client.chat(
        system_prompt="You are an evaluation assistant. Output only valid JSON.",
        user_message=filled_prompt,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.0,
        max_tokens=1000
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail=f"LLM error: {result['error']}")

    raw_output = result.get("output", "")

    # Track cost
    qm = get_quality_manager()
    if result.get("usage"):
        qm.track_cost(
            operation="eval_run",
            provider=provider,
            model=model_name,
            prompt_tokens=result["usage"].get("prompt_tokens", 0),
            completion_tokens=result["usage"].get("completion_tokens", 0),
            project_id=project_id
        )

    # Parse score and verdict from output
    score = 3.0
    verdict = "NEEDS_REVIEW"

    try:
        # Try to extract JSON
        import re
        json_match = re.search(r'\{[\s\S]*\}', raw_output)
        if json_match:
            parsed = json.loads(json_match.group())
            score = float(parsed.get("weighted_score", parsed.get("score", 3.0)))
            verdict = parsed.get("verdict", "NEEDS_REVIEW")
    except Exception as e:
        logger.warning(f"Failed to parse eval output: {e}")

    return score, verdict, raw_output


# =============================================================================
# COST TRACKING ENDPOINTS
# =============================================================================

@router.get("/{project_id}/quality/costs")
async def get_project_costs(project_id: str):
    """Get cost summary for a project"""
    qm = get_quality_manager()
    return {
        "project_id": project_id,
        "costs": qm.get_cost_summary(project_id),
        "session_costs": qm.get_cost_summary()
    }


@router.get("/quality/costs/session")
async def get_session_costs():
    """Get cost summary for current session"""
    qm = get_quality_manager()
    return qm.get_cost_summary()


# =============================================================================
# GROUND TRUTH ENDPOINTS
# =============================================================================

@router.post("/{project_id}/quality/ground-truth")
async def add_ground_truth_example(
    project_id: str,
    example: GroundTruthExampleRequest
):
    """Add a ground truth example for validation"""
    qm = get_quality_manager()

    import hashlib
    example_id = hashlib.md5(
        f"{project_id}_{example.input_text[:50]}_{datetime.utcnow().isoformat()}".encode()
    ).hexdigest()[:12]

    gt_example = GroundTruthExample(
        id=example_id,
        input_text=example.input_text,
        output_text=example.output_text,
        expected_score=example.expected_score,
        expected_verdict=example.expected_verdict,
        score_tolerance=example.score_tolerance,
        category=example.category,
        reasoning=example.reasoning
    )

    qm.ground_truth.add_example(project_id, gt_example)

    return {
        "success": True,
        "example_id": example_id,
        "total_examples": len(qm.ground_truth.get_examples(project_id))
    }


@router.get("/{project_id}/quality/ground-truth")
async def get_ground_truth_examples(project_id: str):
    """Get all ground truth examples for a project"""
    qm = get_quality_manager()
    examples = qm.ground_truth.get_examples(project_id)

    return {
        "project_id": project_id,
        "count": len(examples),
        "examples": [
            {
                "id": e.id,
                "input_text": e.input_text[:200] + "..." if len(e.input_text) > 200 else e.input_text,
                "output_text": e.output_text[:200] + "..." if len(e.output_text) > 200 else e.output_text,
                "expected_score": e.expected_score,
                "expected_verdict": e.expected_verdict,
                "category": e.category
            }
            for e in examples
        ]
    }


@router.post("/{project_id}/quality/ground-truth/validate")
async def validate_against_ground_truth(project_id: str):
    """Validate current eval prompt against ground truth examples"""
    project = project_storage.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    eval_prompt = project.get("eval_prompt")
    if not eval_prompt:
        raise HTTPException(status_code=400, detail="No eval prompt configured")

    qm = get_quality_manager()

    async def eval_func(ep, inp, outp):
        score, verdict, _ = await run_eval_with_project(project_id, ep, inp, outp)
        return score, verdict

    result = await qm.validate_against_ground_truth(project_id, eval_prompt, eval_func)

    return {
        "project_id": project_id,
        "validation": {
            "passed": result.passed,
            "total_examples": result.total_examples,
            "correct_scores": result.correct_scores,
            "correct_verdicts": result.correct_verdicts,
            "accuracy_percentage": result.accuracy_percentage,
            "verdict_accuracy_percentage": result.verdict_accuracy_percentage,
            "avg_score_deviation": result.avg_score_deviation,
            "max_score_deviation": result.max_score_deviation,
            "failed_examples": result.failed_examples
        }
    }


# =============================================================================
# RELIABILITY ENDPOINTS
# =============================================================================

@router.post("/{project_id}/quality/reliability")
async def test_reliability(
    project_id: str,
    request: ReliabilityTestRequest
):
    """Test eval prompt reliability with multiple runs"""
    project = project_storage.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    eval_prompt = project.get("eval_prompt")
    if not eval_prompt:
        raise HTTPException(status_code=400, detail="No eval prompt configured")

    qm = get_quality_manager()

    async def eval_func(ep, inp, outp):
        score, verdict, _ = await run_eval_with_project(project_id, ep, inp, outp)
        return score, verdict

    result = await qm.verify_reliability(
        eval_prompt,
        request.input_text,
        request.output_text,
        eval_func,
        num_runs=request.num_runs
    )

    return {
        "project_id": project_id,
        "reliability": {
            "is_reliable": result.is_reliable,
            "num_runs": result.num_runs,
            "mean_score": result.mean_score,
            "std_deviation": result.std_deviation,
            "score_variance": result.score_variance,
            "verdict_consistency": result.verdict_consistency,
            "confidence_interval_95": result.confidence_interval_95,
            "scores": result.scores,
            "verdicts": result.verdicts,
            "details": result.details
        }
    }


# =============================================================================
# STATISTICAL ANALYSIS ENDPOINTS
# =============================================================================

@router.post("/{project_id}/quality/statistics")
async def analyze_scores(
    project_id: str,
    scores: List[float] = Body(..., embed=True)
):
    """Get statistical analysis of scores"""
    if len(scores) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 scores")

    qm = get_quality_manager()
    result = qm.analyze_scores_statistically(scores)

    return {
        "project_id": project_id,
        "analysis": {
            "sample_size": result.sample_size,
            "mean": result.mean,
            "median": result.median,
            "std_deviation": result.std_deviation,
            "variance": result.variance,
            "confidence_interval_95": result.confidence_interval_95,
            "confidence_interval_99": result.confidence_interval_99,
            "skewness": result.skewness,
            "is_normal_distribution": result.is_normal_distribution,
            "percentiles": result.percentiles
        }
    }


@router.post("/{project_id}/quality/compare")
async def compare_scores(
    project_id: str,
    request: CompareScoresRequest
):
    """Compare two sets of scores for statistical significance"""
    qm = get_quality_manager()
    result = qm.compare_score_sets(request.scores_a, request.scores_b)

    return {
        "project_id": project_id,
        "comparison": result,
        "labels": request.labels
    }


# =============================================================================
# ADVERSARIAL TESTING ENDPOINTS
# =============================================================================

@router.post("/{project_id}/quality/adversarial")
async def run_adversarial_tests(project_id: str):
    """Run adversarial test suite against eval prompt"""
    project = project_storage.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    eval_prompt = project.get("eval_prompt")
    if not eval_prompt:
        raise HTTPException(status_code=400, detail="No eval prompt configured")

    qm = get_quality_manager()

    async def eval_func(ep, inp, outp):
        score, verdict, raw = await run_eval_with_project(project_id, ep, inp, outp)
        return score, verdict, raw

    result = await qm.run_adversarial_tests(
        eval_prompt,
        eval_func,
        normal_input="Standard test input for adversarial testing"
    )

    return {
        "project_id": project_id,
        "adversarial_tests": result
    }


@router.get("/{project_id}/quality/adversarial/tests")
async def get_adversarial_test_suite(project_id: str):
    """Get list of available adversarial tests"""
    qm = get_quality_manager()
    tests = qm.adversarial.get_test_suite()

    return {
        "tests": [
            {
                "id": t.id,
                "name": t.name,
                "type": t.test_type.value,
                "severity": t.severity,
                "description": t.description
            }
            for t in tests
        ]
    }


# =============================================================================
# DRIFT DETECTION ENDPOINTS
# =============================================================================

@router.post("/{project_id}/quality/drift/snapshot")
async def record_drift_snapshot(
    project_id: str,
    request: MetricsSnapshotRequest
):
    """Record a metrics snapshot for drift detection"""
    qm = get_quality_manager()

    snapshot = qm.record_metrics_snapshot(
        project_id,
        request.eval_prompt_version,
        request.metrics,
        request.sample_size
    )

    return {
        "success": True,
        "snapshot": {
            "timestamp": snapshot.timestamp,
            "metrics": snapshot.metrics,
            "sample_size": snapshot.sample_size
        }
    }


@router.get("/{project_id}/quality/drift")
async def check_drift(project_id: str, baseline_days: int = 7):
    """Check for performance drift"""
    qm = get_quality_manager()
    result = qm.check_drift(project_id)

    if not result:
        return {
            "project_id": project_id,
            "drift": None,
            "message": "Insufficient data for drift analysis"
        }

    return {
        "project_id": project_id,
        "drift": {
            "detected": result.drift_detected,
            "severity": result.drift_severity,
            "metrics_changed": result.metrics_changed,
            "recommendations": result.recommendations,
            "baseline_timestamp": result.baseline_snapshot.timestamp,
            "current_timestamp": result.current_snapshot.timestamp
        }
    }


# =============================================================================
# FEEDBACK INTEGRATION ENDPOINTS
# =============================================================================

@router.post("/{project_id}/quality/feedback")
async def record_human_feedback(
    project_id: str,
    request: HumanFeedbackRequest
):
    """Record human feedback for an evaluation"""
    project = project_storage.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    qm = get_quality_manager()

    # Get eval prompt version
    eval_versions = project.get("eval_prompt_versions", [])
    eval_version = len(eval_versions)

    entry_id = qm.record_human_feedback(
        project_id=project_id,
        eval_prompt_version=eval_version,
        test_case_id=request.test_case_id,
        llm_score=request.llm_score,
        llm_verdict=request.llm_verdict,
        llm_reasoning=request.llm_reasoning,
        human_score=request.human_score,
        human_verdict=request.human_verdict,
        human_feedback=request.human_feedback
    )

    return {
        "success": True,
        "feedback_id": entry_id
    }


@router.get("/{project_id}/quality/feedback/improvements")
async def get_feedback_improvements(project_id: str):
    """Get improvement suggestions based on accumulated feedback"""
    qm = get_quality_manager()
    improvements = qm.get_feedback_improvements(project_id)

    return {
        "project_id": project_id,
        "improvements": improvements
    }


@router.post("/{project_id}/quality/feedback/adapt")
async def adapt_eval_from_feedback(project_id: str):
    """Adapt eval prompt based on feedback patterns"""
    project = project_storage.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    eval_prompt = project.get("eval_prompt")
    if not eval_prompt:
        raise HTTPException(status_code=400, detail="No eval prompt configured")

    # Get current dimensions and auto-fails from project
    # These would typically be stored with the project
    dimensions = project.get("eval_dimensions", [])
    auto_fails = project.get("eval_auto_fails", [])

    qm = get_quality_manager()
    adapted_prompt, adapted_dims, adapted_fails = qm.adapt_eval_prompt(
        project_id, eval_prompt, dimensions, auto_fails
    )

    # Update project with adapted eval prompt
    project_storage.update_project(project_id, {
        "eval_prompt": adapted_prompt,
        "eval_dimensions": adapted_dims,
        "eval_auto_fails": adapted_fails,
        "eval_adapted_from_feedback": True,
        "eval_adapted_at": datetime.utcnow().isoformat()
    })

    return {
        "success": True,
        "adapted": True,
        "changes": {
            "dimensions_adjusted": sum(1 for d in adapted_dims if d.get("weight_adjusted")),
            "auto_fails_added": len(adapted_fails) - len(auto_fails)
        }
    }


# =============================================================================
# COMPREHENSIVE VALIDATION ENDPOINT
# =============================================================================

@router.post("/{project_id}/quality/validate")
async def comprehensive_validation(project_id: str):
    """
    Run comprehensive validation suite:
    - Ground truth validation
    - Reliability verification
    - Adversarial testing
    - Drift check
    """
    project = project_storage.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    eval_prompt = project.get("eval_prompt")
    if not eval_prompt:
        raise HTTPException(status_code=400, detail="No eval prompt configured")

    qm = get_quality_manager()

    # Get sample input/output from test cases if available
    test_cases = project.get("test_cases", [])
    if test_cases:
        sample_input = test_cases[0].get("input", "Sample test input")
        sample_output = "This is a sample response to evaluate."
    else:
        sample_input = "Sample test input"
        sample_output = "Sample response"

    async def eval_func(ep, inp, outp):
        score, verdict, raw = await run_eval_with_project(project_id, ep, inp, outp)
        return score, verdict

    async def eval_func_with_raw(ep, inp, outp):
        return await run_eval_with_project(project_id, ep, inp, outp)

    results = await qm.comprehensive_validation(
        project_id,
        eval_prompt,
        eval_func,
        sample_input,
        sample_output
    )

    return results


# =============================================================================
# COMPACT EVAL PROMPT ENDPOINT
# =============================================================================

@router.post("/{project_id}/quality/compact-eval")
async def generate_compact_eval(
    project_id: str,
    request: CompactEvalRequest
):
    """Generate a compact, efficient eval prompt"""
    compact_prompt = build_compact_eval_prompt(
        system_purpose=request.system_purpose,
        dimensions=request.dimensions,
        auto_fails=request.auto_fails,
        input_var=request.input_var,
        output_var=request.output_var
    )

    # Estimate token count (rough: ~4 chars per token)
    estimated_tokens = len(compact_prompt) // 4

    return {
        "compact_eval_prompt": compact_prompt,
        "estimated_tokens": estimated_tokens,
        "character_count": len(compact_prompt)
    }


# =============================================================================
# VALIDATED PIPELINE ENDPOINTS
# =============================================================================

class ValidatedGenerationRequest(BaseModel):
    """Request for validated eval generation"""
    sample_input: Optional[str] = None
    sample_output: Optional[str] = None
    max_cost_usd: float = Field(default=0.50, ge=0.01, le=5.0)
    require_adversarial_pass: bool = True
    skip_live_validation: bool = False


class SemanticCheckRequest(BaseModel):
    """Request for semantic analysis"""
    prompt_text: str


@router.post("/{project_id}/quality/generate-validated")
async def generate_validated_eval_prompt(
    project_id: str,
    request: ValidatedGenerationRequest = None
):
    """
    Generate eval prompt with full validation pipeline.

    This is the RECOMMENDED way to generate eval prompts as it:
    1. Runs semantic analysis for contradictions
    2. Applies feedback learning adaptations
    3. Validates against ground truth (if available)
    4. Tests reliability with multiple runs
    5. Runs adversarial security tests
    6. Enforces statistical confidence
    7. Tracks costs with budget limits
    8. Saves version with validation metadata

    Returns eval prompt only if ALL blocking gates pass.
    """
    from validated_eval_pipeline import (
        ValidatedEvalPipeline,
        EvalPromptVersionManager,
        CostBudget
    )
    from shared_settings import get_settings

    if request is None:
        request = ValidatedGenerationRequest()

    project = project_storage.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get system prompt
    system_prompt_versions = project.get("system_prompt_versions", [])
    if not system_prompt_versions:
        raise HTTPException(status_code=400, detail="No system prompt found")

    current_prompt = system_prompt_versions[-1].get("prompt_text", "")

    # Get settings
    settings = get_settings()
    provider = settings.get("provider", "openai")
    api_key = settings.get("api_key", "")
    model_name = settings.get("model_name", "gpt-4o-mini")

    if not api_key:
        raise HTTPException(status_code=400, detail="API key not configured")

    # Create cost budget
    cost_budget = CostBudget(
        max_cost_per_validation=request.max_cost_usd,
        hard_limit=True
    )

    # Create pipeline
    pipeline = ValidatedEvalPipeline(
        project_id=project_id,
        cost_budget=cost_budget,
        require_adversarial_pass=request.require_adversarial_pass
    )

    # Create eval function if we have sample data
    run_eval_func = None
    if not request.skip_live_validation and request.sample_input and request.sample_output:
        async def run_eval_func(ep, inp, outp):
            score, verdict, _ = await run_eval_with_project(project_id, ep, inp, outp)
            return score, verdict

    # Generate and validate
    eval_prompt, validation_result = await pipeline.generate_and_validate(
        system_prompt=current_prompt,
        use_case=project.get("use_case", ""),
        requirements=project.get("key_requirements", []),
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        run_eval_func=run_eval_func,
        sample_input=request.sample_input,
        sample_output=request.sample_output
    )

    # Save version
    version_manager = EvalPromptVersionManager(project_id)
    version_info = version_manager.save_version(
        eval_prompt=eval_prompt,
        validation_result=validation_result,
        dimensions=[],
        auto_fails=[],
        changes_made="Generated via validated pipeline"
    )

    return {
        "eval_prompt": eval_prompt if validation_result.can_deploy else None,
        "validation": {
            "status": validation_result.overall_status.value,
            "score": validation_result.overall_score,
            "can_deploy": validation_result.can_deploy,
            "blocking_failures": validation_result.blocking_failures,
            "warnings": validation_result.warnings,
            "cost_usd": validation_result.cost_incurred,
            "gates": [
                {
                    "name": g.gate_name,
                    "status": g.status.value,
                    "score": g.score,
                    "blocking": g.blocking,
                    "recommendation": g.recommendation
                }
                for g in validation_result.gates
            ]
        },
        "version": version_info.get("version"),
        "is_deployed": version_info.get("is_deployed", False)
    }


@router.post("/{project_id}/quality/semantic-check")
async def check_semantic_issues(
    project_id: str,
    request: SemanticCheckRequest
):
    """
    Check a prompt for semantic contradictions and ambiguities.

    This can be used on:
    - System prompts before generating eval prompts
    - Eval prompts before deployment
    - Any prompt text for quality analysis
    """
    from validated_eval_pipeline import SemanticContradictionDetector

    detector = SemanticContradictionDetector()
    analysis = detector.analyze(request.prompt_text)

    return {
        "project_id": project_id,
        "analysis": analysis,
        "has_blocking_issues": analysis["has_critical_issues"],
        "severity_level": "critical" if analysis["has_critical_issues"] else (
            "warning" if analysis["severity_score"] > 30 else "ok"
        )
    }


@router.get("/{project_id}/quality/versions")
async def list_eval_versions(project_id: str):
    """List all eval prompt versions with validation status"""
    from validated_eval_pipeline import EvalPromptVersionManager

    version_manager = EvalPromptVersionManager(project_id)
    versions = version_manager.list_versions()

    return {
        "project_id": project_id,
        "versions": versions,
        "total": len(versions)
    }


@router.put("/{project_id}/quality/versions/{version_number}/rollback")
async def rollback_eval_version(project_id: str, version_number: int):
    """Rollback to a specific eval prompt version"""
    from validated_eval_pipeline import EvalPromptVersionManager

    version_manager = EvalPromptVersionManager(project_id)

    try:
        result = version_manager.rollback_to_version(version_number)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{project_id}/quality/deployed")
async def get_deployed_eval_version(project_id: str):
    """Get the currently deployed eval prompt version"""
    from validated_eval_pipeline import EvalPromptVersionManager

    version_manager = EvalPromptVersionManager(project_id)
    deployed = version_manager.get_deployed_version()

    if not deployed:
        return {
            "project_id": project_id,
            "deployed": None,
            "message": "No eval prompt deployed"
        }

    return {
        "project_id": project_id,
        "deployed": {
            "version": deployed.get("version"),
            "created_at": deployed.get("created_at"),
            "validation_status": deployed.get("validation", {}).get("status"),
            "validation_score": deployed.get("validation", {}).get("score"),
            "eval_prompt_preview": deployed.get("eval_prompt_text", "")[:500] + "..."
        }
    }


@router.post("/{project_id}/quality/pipeline-status")
async def get_pipeline_readiness(project_id: str):
    """
    Check if project is ready for validated eval generation.

    Returns status of all prerequisites:
    - System prompt exists
    - API key configured
    - Ground truth examples (optional but recommended)
    - Previous feedback (optional)
    """
    from shared_settings import get_settings

    project = project_storage.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    settings = get_settings()
    qm = get_quality_manager()

    # Check prerequisites
    has_system_prompt = bool(project.get("system_prompt_versions"))
    has_api_key = bool(settings.get("api_key"))

    ground_truth_examples = qm.ground_truth.get_examples(project_id)
    has_ground_truth = len(ground_truth_examples) >= 3

    feedback_ready = qm.feedback_enabled

    # Calculate readiness score
    readiness_score = 0
    if has_system_prompt:
        readiness_score += 40
    if has_api_key:
        readiness_score += 30
    if has_ground_truth:
        readiness_score += 20
    if feedback_ready:
        readiness_score += 10

    return {
        "project_id": project_id,
        "ready_for_generation": has_system_prompt and has_api_key,
        "ready_for_full_validation": has_system_prompt and has_api_key and has_ground_truth,
        "readiness_score": readiness_score,
        "prerequisites": {
            "system_prompt": {
                "status": "ready" if has_system_prompt else "missing",
                "required": True
            },
            "api_key": {
                "status": "ready" if has_api_key else "missing",
                "required": True
            },
            "ground_truth_examples": {
                "status": "ready" if has_ground_truth else "insufficient",
                "count": len(ground_truth_examples),
                "required_count": 3,
                "required": False
            },
            "feedback_learning": {
                "status": "enabled" if feedback_ready else "disabled",
                "required": False
            }
        },
        "recommendations": [
            r for r in [
                None if has_system_prompt else "Add a system prompt first",
                None if has_api_key else "Configure API key in settings",
                None if has_ground_truth else f"Add {3 - len(ground_truth_examples)} more ground truth examples for full validation",
            ] if r
        ]
    }
