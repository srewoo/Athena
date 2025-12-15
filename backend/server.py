"""
Clean FastAPI server for Athena - 5-Step Prompt Testing Workflow
"""
import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Load .env from root directory (parent of backend/)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List
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
from llm_client import get_llm_client
from shared_settings import settings_store, get_settings as get_llm_settings_shared, update_settings
import project_api
import project_storage
from prompt_analyzer import analyze_prompt, analysis_to_dict, PromptType
from agentic_rewrite import agentic_rewrite, result_to_dict as agentic_result_to_dict, get_thinking_model_for_provider
from agentic_eval import agentic_eval_generation, result_to_dict as agentic_eval_result_to_dict
from smart_test_generator import detect_input_type, build_input_generation_prompt, get_scenario_variations, InputType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cleanup configuration
CLEANUP_INTERVAL_HOURS = 24  # Run cleanup every 24 hours
CLEANUP_AGE_DAYS = 30  # Delete projects older than 30 days


async def cleanup_scheduler():
    """Background task to periodically clean up old projects"""
    while True:
        try:
            # Wait for the specified interval
            await asyncio.sleep(CLEANUP_INTERVAL_HOURS * 3600)

            # Run cleanup
            logger.info("Running scheduled project cleanup...")
            result = project_storage.cleanup_old_projects(days=CLEANUP_AGE_DAYS)
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
    # Startup: Start the cleanup scheduler
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
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM client
llm_client = get_llm_client()

# Include project management router
app.include_router(project_api.router)


@app.get("/")
async def root():
    return {"message": "Prompt Testing Application API"}


# ============================================================================
# ADMIN/MAINTENANCE ENDPOINTS
# ============================================================================

@app.get("/api/admin/storage-stats")
async def get_storage_stats():
    """Get storage statistics for project files"""
    return project_storage.get_storage_stats()


@app.post("/api/admin/cleanup")
async def trigger_cleanup(days: int = 30):
    """Manually trigger cleanup of old projects"""
    result = project_storage.cleanup_old_projects(days=days)
    logger.info(f"Manual cleanup triggered: {result['deleted_count']} projects deleted")
    return result


# ============================================================================
# STEP 1: VALIDATION
# ============================================================================

@app.post("/api/step1/validate")
async def validate_project(project: ProjectInput):
    """Validate project input"""
    if not all([project.project_name, project.use_case, project.requirements, project.initial_prompt]):
        raise HTTPException(status_code=400, detail="All fields are required")
    return {"message": "Validation successful"}


# ============================================================================
# STEP 2: PROMPT OPTIMIZATION
# ============================================================================

@app.post("/api/step2/analyze")
async def analyze_prompt_endpoint(prompt_text: str):
    """Analyze a prompt to extract DNA, detect type, and assess quality"""
    if not prompt_text:
        raise HTTPException(status_code=400, detail="prompt_text is required")

    analysis = analyze_prompt(prompt_text)
    return analysis_to_dict(analysis)


@app.post("/api/step2/optimize", response_model=PromptOptimizationResult)
async def optimize_prompt(project: ProjectInput):
    """Step 2: Optimize the initial prompt with DNA-aware optimization"""

    # First, analyze the prompt to understand its structure
    analysis = analyze_prompt(project.initial_prompt)
    analysis_dict = analysis_to_dict(analysis)

    # Build DNA preservation instructions
    dna_instructions = build_dna_preservation_instructions(analysis)

    # Build type-specific optimization guidance
    type_guidance = build_type_specific_guidance(analysis.prompt_type)

    # Check if prompt is already high quality
    if analysis.quality_score >= 8.5 and not analysis.improvement_needed:
        # Return the original with high score - no changes needed
        return PromptOptimizationResult(
            optimized_prompt=project.initial_prompt,
            score=analysis.quality_score,
            analysis=f"This prompt is already production-ready (score: {analysis.quality_score}/10). Strengths: {', '.join(analysis.strengths[:3])}. No optimization needed.",
            improvements=[],
            suggestions=analysis.improvement_areas if analysis.improvement_areas else ["Consider adding examples for edge cases"]
        )

    system_prompt = f"""You are an expert prompt engineer. Your task is to transform an initial prompt into a highly effective, production-ready system prompt.

## PROMPT ANALYSIS RESULTS
- **Detected Type:** {analysis.prompt_type.value}
- **Current Quality Score:** {analysis.quality_score}/10
- **Improvement Needed:** {analysis.improvement_needed}

## AREAS TO IMPROVE
{chr(10).join(f'- {area}' for area in analysis.improvement_areas) if analysis.improvement_areas else '- Minor refinements only'}

## CURRENT STRENGTHS (PRESERVE THESE)
{chr(10).join(f'- {strength}' for strength in analysis.strengths) if analysis.strengths else '- None identified'}

{dna_instructions}

{type_guidance}

## OPTIMIZATION RULES
1. **PRESERVE DNA**: Keep all template variables, output format, scoring scales, and key terminology EXACTLY as found
2. **FIX WEAKNESSES**: Address the specific improvement areas identified above
3. **MAINTAIN STRENGTHS**: Do not change what's already working well
4. **ADD STRUCTURE**: If missing, add clear sections (Role, Task, Format, Constraints)
5. **DO NOT OVER-ENGINEER**: Only add what's genuinely needed

## QUALITY SCORING (1-10)
- **9-10**: Exceptional - comprehensive, clear structure, examples, edge cases handled
- **7-8**: Strong - well-structured, clear instructions, most requirements covered
- **5-6**: Good - clear improvements, addresses main requirements
- **3-4**: Adequate - some improvements but missing key elements
- **1-2**: Poor - minimal improvement or missing critical requirements

Return a JSON object with this structure:
{{
    "optimized_prompt": "The complete optimized system prompt",
    "score": 8.5,
    "analysis": "Detailed analysis of optimizations made",
    "improvements": ["Improvement 1", "Improvement 2", "..."],
    "suggestions": ["Additional suggestion 1", "Additional suggestion 2", "..."]
}}"""

    user_message = f"""Please optimize this prompt based on the analysis above.

**Use Case:** {project.use_case}

**Requirements:**
{project.requirements}

**Initial Prompt:**
```
{project.initial_prompt}
```

Focus on fixing the identified weaknesses while preserving the DNA elements and strengths."""

    result = await llm_client.chat(
        system_prompt=system_prompt,
        user_message=user_message,
        provider=project.provider,
        api_key=project.api_key,
        model_name=project.model_name,
        temperature=0.5,  # Lower temperature for more consistent optimization
        max_tokens=8000
    )

    if result["error"]:
        raise HTTPException(status_code=500, detail=f"LLM Error: {result['error']}")

    try:
        json_match = re.search(r'\{[\s\S]*\}', result["output"])
        if json_match:
            data = json.loads(json_match.group())

            # Validate DNA preservation
            optimized = data.get("optimized_prompt", "")
            validation_issues = validate_dna_preservation(project.initial_prompt, optimized, analysis)

            if validation_issues:
                logger.warning(f"DNA preservation issues: {validation_issues}")
                # Add warning to analysis
                data["analysis"] = data.get("analysis", "") + f"\n\nWARNING: Some DNA elements may have been modified: {', '.join(validation_issues)}"

            # Include analysis metadata in response
            data["prompt_analysis"] = analysis_dict

            return PromptOptimizationResult(**data)
        else:
            raise ValueError("No JSON found in response")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse optimization result: {str(e)}")


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


@app.post("/api/step2/refine", response_model=PromptOptimizationResult)
async def refine_optimization(project: ProjectInput, current_result: dict, feedback: str):
    """Step 2 Refinement: Refine optimized prompt based on user feedback"""
    system_prompt = """You are an expert prompt engineer. Refine an already optimized prompt based on user feedback.

You will receive the CURRENT optimized prompt and USER FEEDBACK requesting specific changes.

Your job is to incorporate the user's feedback while maintaining the quality of the prompt.

Return a JSON object with this structure:
{
    "optimized_prompt": "The refined system prompt incorporating user feedback",
    "score": 8.5,
    "analysis": "Detailed analysis of what changed based on user feedback",
    "improvements": ["Change 1 based on feedback", "Change 2 based on feedback", "..."],
    "suggestions": ["Additional suggestion 1", "Additional suggestion 2", "..."]
}"""

    user_message = f"""Please refine the optimized prompt based on user feedback.

**Use Case:** {project.use_case}

**Requirements:** {project.requirements}

**CURRENT Optimized Prompt:**
```
{current_result.get('optimized_prompt', '')}
```

**Current Score:** {current_result.get('score', 0)} / 10

**USER FEEDBACK:**
{feedback}

Please incorporate this feedback and provide an improved version."""

    result = await llm_client.chat(
        system_prompt=system_prompt,
        user_message=user_message,
        provider=project.provider,
        api_key=project.api_key,
        model_name=project.model_name,
        temperature=0.7,
        max_tokens=8000
    )

    if result["error"]:
        raise HTTPException(status_code=500, detail=f"LLM Error: {result['error']}")

    try:
        json_match = re.search(r'\{[\s\S]*\}', result["output"])
        if json_match:
            data = json.loads(json_match.group())
            return PromptOptimizationResult(**data)
        else:
            raise ValueError("No JSON found in response")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse refinement result: {str(e)}")


@app.post("/api/step2/agentic-rewrite")
async def agentic_rewrite_optimization(project: ProjectInput, current_result: dict, use_thinking_model: bool = True):
    """
    Step 2 Agentic Rewrite: Multi-step, self-correcting prompt optimization.

    This is an enhanced version of AI Rewrite that uses:
    1. Deep analysis with thinking model (o3-mini, etc.)
    2. Structured improvement planning
    3. Careful execution with DNA preservation
    4. Validation and iteration loop

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
                "validation": result.validation
            }
        }

    except Exception as e:
        logger.error(f"Agentic rewrite failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agentic rewrite failed: {str(e)}")


# ============================================================================
# STEP 3: EVALUATION PROMPT GENERATION
# ============================================================================

@app.post("/api/step3/agentic-generate-eval")
async def agentic_generate_evaluation_prompt(project: ProjectInput, optimized_prompt: str = Body(...), use_thinking_model: bool = Body(True)):
    """
    Step 3 Agentic Eval Generation: Multi-step, self-validating evaluation prompt generation.

    This is an enhanced version of eval generation that uses:
    1. Deep analysis with thinking model to understand what could go wrong
    2. Systematic failure mode identification
    3. Eval dimensions designed to catch each failure mode
    4. Self-test validation to ensure coverage
    5. Iterative refinement if gaps found

    Parameters:
    - project: Project configuration including API keys
    - optimized_prompt: The system prompt to generate eval for
    - use_thinking_model: Whether to use a thinking model for analysis (default: True)
    """
    if not optimized_prompt:
        raise HTTPException(status_code=400, detail="optimized_prompt is required")

    # Determine thinking model to use
    thinking_model = None
    if use_thinking_model:
        thinking_model = get_thinking_model_for_provider(project.provider)
        logger.info(f"Using thinking model: {thinking_model} for eval generation")

    try:
        result = await agentic_eval_generation(
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

        return {
            "eval_prompt": result.eval_prompt,
            "eval_criteria": result.eval_criteria,
            "rationale": result.rationale,
            "agentic_details": {
                "failure_modes": result.failure_modes,
                "eval_dimensions": result.eval_dimensions,
                "self_test": result.self_test,
                "iterations": result.iterations,
                "steps_taken": result.steps_taken
            }
        }

    except Exception as e:
        logger.error(f"Agentic eval generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agentic eval generation failed: {str(e)}")


@app.post("/api/step3/refine", response_model=EvaluationPromptResult)
async def refine_eval_prompt(project: ProjectInput, optimized_prompt: str, current_result: dict, feedback: str):
    """Step 3 Refinement: Refine evaluation prompt based on user feedback"""
    system_prompt = """You are an expert in LLM evaluation. Refine an evaluation prompt based on user feedback while maintaining the 5-section structure.

Return JSON:
{
    "eval_prompt": "Refined evaluation prompt",
    "eval_criteria": ["Criterion 1", "..."],
    "rationale": "Explanation of changes based on feedback"
}"""

    user_message = f"""Refine the evaluation prompt based on feedback.

**Use Case:** {project.use_case}
**Requirements:** {project.requirements}

**System Prompt Being Evaluated:**
```
{optimized_prompt}
```

**CURRENT Evaluation Prompt:**
```
{current_result.get("eval_prompt", "")}
```

**USER FEEDBACK:**
{feedback}

Please incorporate this feedback and provide an improved evaluation prompt."""

    result = await llm_client.chat(
        system_prompt=system_prompt,
        user_message=user_message,
        provider=project.provider,
        api_key=project.api_key,
        model_name=project.model_name,
        temperature=0.5,
        max_tokens=8000
    )

    if result["error"]:
        raise HTTPException(status_code=500, detail=f"LLM Error: {result['error']}")

    try:
        json_match = re.search(r'\{[\s\S]*\}', result["output"])
        if json_match:
            data = json.loads(json_match.group())
            return EvaluationPromptResult(**data)
        else:
            raise ValueError("No JSON found in response")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse refinement: {str(e)}")


# ============================================================================
# STEP 4: TEST DATA GENERATION
# ============================================================================

@app.post("/api/step4/generate-testdata", response_model=TestDataResult)
async def generate_test_data(project: ProjectInput, optimized_prompt: str, num_cases: int = 10):
    """Step 4: Generate test data using smart, type-aware distribution"""

    # Analyze the optimized prompt to get type-specific test categories
    analysis = analyze_prompt(optimized_prompt)
    analysis_dict = analysis_to_dict(analysis)

    # Build dynamic test categories based on prompt type
    test_categories_text = build_test_categories_text(analysis)

    # Calculate distribution based on analysis
    categories = analysis.suggested_test_categories
    distribution = {}
    for cat in categories:
        count = max(1, int(num_cases * cat['percentage'] / 100))
        distribution[cat['name']] = count

    # Adjust to match total
    total = sum(distribution.values())
    if total != num_cases:
        # Add/remove from largest category
        largest = max(distribution, key=distribution.get)
        distribution[largest] += (num_cases - total)

    system_prompt = f"""You are an expert test data generator. Generate REALISTIC INPUT DATA (user queries/requests) tailored to test a specific type of system prompt.

## PROMPT ANALYSIS
- **Prompt Type:** {analysis.prompt_type.value}
- **Output Format:** {analysis.dna.output_format or 'unspecified'}
- **Template Variables:** {', '.join(analysis.dna.template_variables) if analysis.dna.template_variables else 'None'}
- **Key Terminology:** {', '.join(analysis.dna.key_terminology[:5]) if analysis.dna.key_terminology else 'None'}

## TEST CASE CATEGORIES (use these exact categories)
{test_categories_text}

## DISTRIBUTION FOR THIS RUN
{chr(10).join(f'- {cat}: {count} cases' for cat, count in distribution.items())}

## CRITICAL RULES
1. Generate inputs that match the prompt's expected input format
2. If template variables exist ({', '.join(analysis.dna.template_variables) if analysis.dna.template_variables else 'none'}), generate values for those variables
3. Make adversarial cases realistic - actual prompt injection attempts, not obvious markers
4. Edge cases should test real boundary conditions relevant to THIS prompt type
5. Each test case must be unique and test something different

## OUTPUT FORMAT
Return a JSON object:
{{
    "test_cases": [
        {{
            "input": "The actual input text/query (or JSON if the prompt expects structured input)",
            "category": "positive|edge_case|negative|adversarial",
            "test_focus": "What specific aspect this test case validates",
            "expected_behavior": "How the system should handle this input"
        }}
    ]
}}"""

    user_message = f"""Generate {num_cases} test input cases for this {analysis.prompt_type.value} system.

**Use Case:** {project.use_case}

**Requirements:** {project.requirements}

**System Prompt:**
```
{optimized_prompt}
```

**Distribution to generate:**
{chr(10).join(f'- {cat}: {count} cases' for cat, count in distribution.items())}

Generate {num_cases} diverse, realistic INPUT cases that properly test a {analysis.prompt_type.value} prompt.
Ensure each test case is designed to validate a specific aspect of the system's behavior."""

    result = await llm_client.chat(
        system_prompt=system_prompt,
        user_message=user_message,
        provider=project.provider,
        api_key=project.api_key,
        model_name=project.model_name,
        temperature=0.8,
        max_tokens=8000
    )

    if result["error"]:
        raise HTTPException(status_code=500, detail=f"LLM Error: {result['error']}")

    try:
        json_match = re.search(r'\{[\s\S]*\}', result["output"])
        if json_match:
            data = json.loads(json_match.group())
            test_cases = data.get("test_cases", [])
            categories = {}
            for tc in test_cases:
                cat = tc.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1

            return TestDataResult(
                test_cases=test_cases,
                count=len(test_cases),
                categories=categories
            )
        else:
            raise ValueError("No JSON found in response")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse test data result: {str(e)}")


@app.post("/api/step4/refine", response_model=TestDataResult)
async def refine_test_data(project: ProjectInput, optimized_prompt: str, current_result: dict, feedback: str, num_cases: int = 10):
    """Step 4 Refinement: Refine test data based on user feedback"""
    system_prompt = """You are an expert test data generator. Refine test cases based on user feedback while maintaining diversity and proper distribution.

Return JSON:
{
    "test_cases": [{
        "input": "...",
        "category": "positive|edge_case|negative|adversarial",
        "test_focus": "...",
        "expected_behavior": "..."
    }]
}"""

    user_message = f"""Refine the test cases based on feedback.

**Use Case:** {project.use_case}
**Requirements:** {project.requirements}

**System Prompt:**
```
{optimized_prompt}
```

**USER FEEDBACK:**
{feedback}

Generate {num_cases} improved test cases incorporating this feedback."""

    result = await llm_client.chat(
        system_prompt=system_prompt,
        user_message=user_message,
        provider=project.provider,
        api_key=project.api_key,
        model_name=project.model_name,
        temperature=0.8,
        max_tokens=8000
    )

    if result["error"]:
        raise HTTPException(status_code=500, detail=f"LLM Error: {result['error']}")

    try:
        json_match = re.search(r'\{[\s\S]*\}', result["output"])
        if json_match:
            data = json.loads(json_match.group())
            test_cases = data.get("test_cases", [])
            categories = {}
            for tc in test_cases:
                cat = tc.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
            return TestDataResult(
                test_cases=test_cases,
                count=len(test_cases),
                categories=categories
            )
        else:
            raise ValueError("No JSON found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse refinement: {str(e)}")


@app.post("/api/step4/smart-generate-testdata", response_model=TestDataResult)
async def smart_generate_test_data(project: ProjectInput, optimized_prompt: str = Body(...), num_cases: int = Body(10)):
    """
    Step 4 Smart Test Data Generation: Context-aware test input generation.

    This enhanced version:
    1. Detects the expected input type (call transcript, email, code, etc.)
    2. Generates realistic, domain-appropriate test inputs
    3. Creates full, complete inputs (not placeholders)
    4. Varies scenarios based on the input type
    """

    # Analyze the prompt to understand its structure
    analysis = analyze_prompt(optimized_prompt)
    analysis_dict = analysis_to_dict(analysis)

    # Detect the expected input format
    input_spec = detect_input_type(optimized_prompt, analysis.dna.template_variables)
    logger.info(f"Detected input type: {input_spec.input_type.value}, template variable: {input_spec.template_variable}")

    # Get scenario variations for this input type
    scenarios = get_scenario_variations(input_spec, num_cases)

    # Build the smart generation prompt
    system_prompt = build_input_generation_prompt(input_spec, optimized_prompt, project.use_case, project.requirements)

    # Build scenario descriptions for the user message
    scenario_list = "\n".join([
        f"{i+1}. [{s['category'].upper()}] {s['scenario']}: {s['description']}"
        for i, s in enumerate(scenarios)
    ])

    user_message = f"""Generate {num_cases} COMPLETE, REALISTIC test inputs for this system prompt.

**Input Type Detected:** {input_spec.input_type.value.replace('_', ' ').title()}
**Template Variable:** {{{{{input_spec.template_variable}}}}}
**Domain Context:** {input_spec.domain_context or 'General'}

**System Prompt Being Tested:**
```
{optimized_prompt}
```

**Scenarios to Generate (one test case per scenario):**
{scenario_list}

## CRITICAL REQUIREMENTS:
1. Generate FULL, COMPLETE inputs - NOT summaries or placeholders
2. For {input_spec.input_type.value.replace('_', ' ')}, include ALL necessary details
3. Each input should be {input_spec.expected_length} length (200-800 words for transcripts/documents)
4. Make inputs REALISTIC with specific names, dates, numbers, and details
5. Vary the content significantly between test cases

Return JSON with complete test cases."""

    result = await llm_client.chat(
        system_prompt=system_prompt,
        user_message=user_message,
        provider=project.provider,
        api_key=project.api_key,
        model_name=project.model_name,
        temperature=0.8,
        max_tokens=16000  # More tokens for full transcripts/documents
    )

    if result["error"]:
        raise HTTPException(status_code=500, detail=f"LLM Error: {result['error']}")

    try:
        json_match = re.search(r'\{[\s\S]*\}', result["output"])
        if json_match:
            data = json.loads(json_match.group())
            test_cases = data.get("test_cases", [])
            categories = {}
            for tc in test_cases:
                cat = tc.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1

            return TestDataResult(
                test_cases=test_cases,
                count=len(test_cases),
                categories=categories,
                metadata={
                    "input_type": input_spec.input_type.value,
                    "template_variable": input_spec.template_variable,
                    "domain_context": input_spec.domain_context,
                    "expected_length": input_spec.expected_length
                }
            )
        else:
            raise ValueError("No JSON found in response")
    except Exception as e:
        logger.error(f"Failed to parse smart test data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to parse test data result: {str(e)}")


# ============================================================================
# STEP 5: TEST EXECUTION & REPORTING
# ============================================================================

@app.post("/api/step5/execute-tests", response_model=List[TestExecutionResult])
async def execute_tests(project: ProjectInput, optimized_prompt: str, eval_prompt: str, test_cases: List[dict]):
    """Step 5: Execute tests and evaluate results"""
    results = []

    for test_case in test_cases:
        # Execute the optimized prompt with test input
        prompt_result = await llm_client.chat(
            system_prompt=optimized_prompt,
            user_message=test_case["input"],
            provider=project.provider,
            api_key=project.api_key,
            model_name=project.model_name,
            temperature=0.7,
            max_tokens=8000
        )

        if prompt_result["error"]:
            continue

        # Evaluate the output
        eval_message = eval_prompt.replace("{{input}}", test_case["input"]).replace("{{output}}", prompt_result["output"])

        eval_result = await llm_client.chat(
            system_prompt="You are an evaluation assistant. Assess the provided output according to the evaluation criteria.",
            user_message=eval_message,
            provider=project.provider,
            api_key=project.api_key,
            model_name=project.model_name,
            temperature=0.3,
            max_tokens=8000
        )

        if eval_result["error"]:
            continue

        # Parse evaluation score
        try:
            json_match = re.search(r'\{[\s\S]*\}', eval_result["output"])
            if json_match:
                eval_data = json.loads(json_match.group())
                score = float(eval_data.get("score", 0))
                reasoning = eval_data.get("reasoning", "")
            else:
                score = 0
                reasoning = "Failed to parse evaluation"
        except:
            score = 0
            reasoning = "Failed to parse evaluation"

        results.append(TestExecutionResult(
            test_case=test_case,
            prompt_output=prompt_result["output"],
            eval_score=score,
            eval_feedback=reasoning,
            passed=score >= 3.5,
            latency_ms=prompt_result.get("latency_ms", 0),
            tokens_used=prompt_result.get("tokens_used", 0)
        ))

    return results


@app.post("/api/generate-report", response_model=FinalReport)
async def generate_report(project_name: str, optimized_prompt: str, optimization_score: float, test_results: List[dict]):
    """Generate final test report with statistics"""
    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results if r.get("passed", False))
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

    avg_score = sum(r.get("eval_score", 0) for r in test_results) / total_tests if total_tests > 0 else 0

    # Category breakdown
    category_stats = {}
    for result in test_results:
        category = result.get("test_case", {}).get("category", "unknown")
        if category not in category_stats:
            category_stats[category] = {"total": 0, "passed": 0}
        category_stats[category]["total"] += 1
        if result.get("passed", False):
            category_stats[category]["passed"] += 1

    category_breakdown = {
        cat: {
            "pass_rate": (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0,
            "total_tests": stats["total"]
        }
        for cat, stats in category_stats.items()
    }

    total_tokens = sum(r.get("tokens_used", 0) for r in test_results)
    avg_latency = sum(r.get("latency_ms", 0) for r in test_results) / total_tests if total_tests > 0 else 0

    return FinalReport(
        project_name=project_name,
        optimization_score=optimization_score,
        pass_rate=pass_rate,
        avg_score=avg_score,
        total_tests=total_tests,
        passed_tests=passed_tests,
        category_breakdown=category_breakdown,
        total_tokens=total_tokens,
        avg_latency_ms=avg_latency
    )


# ============================================================================
# EVALUATIONS API (for Dashboard, History, Compare pages)
# ============================================================================

# In-memory evaluations storage
evaluations_store = {}


@app.post("/api/evaluate")
async def evaluate_prompt(data: dict):
    """Evaluate a prompt using LLM - used by Dashboard.js"""
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
    """A/B test two prompts - used by ABTesting.js"""
    prompt_a = data.get("prompt_a", "")
    prompt_b = data.get("prompt_b", "")
    test_inputs = data.get("test_inputs", ["Test input 1", "Test input 2", "Test input 3"])

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

    for test_input in test_inputs[:5]:  # Limit to 5 tests
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
        results_a.append({
            "input": test_input,
            "output": result_a.get("output", "Error") if not result_a.get("error") else "Error",
            "latency_ms": result_a.get("latency_ms", 0)
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
        results_b.append({
            "input": test_input,
            "output": result_b.get("output", "Error") if not result_b.get("error") else "Error",
            "latency_ms": result_b.get("latency_ms", 0)
        })

    # Calculate metrics
    avg_latency_a = sum(r["latency_ms"] for r in results_a) / len(results_a) if results_a else 0
    avg_latency_b = sum(r["latency_ms"] for r in results_b) / len(results_b) if results_b else 0

    return {
        "prompt_a_results": results_a,
        "prompt_b_results": results_b,
        "summary": {
            "prompt_a_avg_latency": round(avg_latency_a, 2),
            "prompt_b_avg_latency": round(avg_latency_b, 2),
            "tests_run": len(test_inputs[:5]),
            "winner": "A" if avg_latency_a < avg_latency_b else "B"
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
    updated = update_settings(
        llm_provider=settings.get("llm_provider", "openai"),
        api_key=settings.get("api_key", ""),
        model_name=settings.get("model_name", "")
    )
    return {"message": "Settings saved successfully", "settings": {
        "llm_provider": updated["llm_provider"],
        "api_key": updated["api_key"],
        "model_name": updated["model_name"]
    }}


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
