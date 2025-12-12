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

from fastapi import FastAPI, HTTPException
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


@app.post("/api/rewrite")
async def rewrite_prompt(data: dict):
    """AI-powered prompt rewriting based on focus areas"""
    prompt_text = data.get("prompt_text", "")
    focus_areas = data.get("focus_areas", [])
    use_case = data.get("use_case", "")
    key_requirements = data.get("key_requirements", [])

    if not prompt_text:
        raise HTTPException(status_code=400, detail="prompt_text is required")

    # Create improvement instructions
    focus_list = "\n".join(f"- {area}" for area in focus_areas)
    requirements_list = "\n".join(f"- {req}" for req in key_requirements) if key_requirements else "Not specified"

    system_prompt = f"""You are an expert prompt engineer. Your task is to improve the given prompt by addressing the specific focus areas.

Use Case: {use_case}
Key Requirements:
{requirements_list}

Focus Areas to Address:
{focus_list}

Rewrite the prompt to:
1. Address all focus areas mentioned above
2. Maintain the core intent of the original prompt
3. Add missing elements (context, examples, constraints, format, etc.)
4. Make it more specific and actionable
5. Keep it clear and well-structured

Return the improved prompt directly, without explanations."""

    # Get LLM settings from settings store (standardized field names)
    provider = settings_store.get("llm_provider", "openai")
    api_key = settings_store.get("api_key", "")
    model_name = settings_store.get("model_name")

    if not api_key:
        raise HTTPException(status_code=400, detail="API key not configured. Please configure settings first.")

    # Call LLM to rewrite prompt
    user_message = f"Original prompt:\n{prompt_text}\n\nImprove this prompt by addressing the focus areas listed above."
    response = await llm_client.chat(
        system_prompt=system_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.7,
        max_tokens=8000
    )

    if response.get("error"):
        raise HTTPException(status_code=500, detail=response["error"])

    rewritten_prompt = response.get("output", "").strip()

    # Extract what was changed
    changes_made = focus_areas[:3]  # First 3 focus areas as changes

    return {
        "rewritten_prompt": rewritten_prompt,
        "changes_made": changes_made
    }


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

@app.post("/api/step2/optimize", response_model=PromptOptimizationResult)
async def optimize_prompt(project: ProjectInput):
    """Step 2: Optimize the initial prompt"""
    system_prompt = """You are an expert prompt engineer. Your task is to transform an initial prompt into a highly effective, production-ready system prompt.

## CRITICAL PRIORITY ORDER
1. **Requirements Alignment** - Ensure ALL requirements are addressed
2. **Template Variable Preservation** - NEVER remove or modify {{variable}} placeholders
3. **Clarity & Structure** - Make instructions crystal clear and well-organized
4. **Best Practices** - Apply proven prompt engineering techniques

## TEMPLATE VARIABLE RULES
- **PRESERVE ALL** `{{variable}}` placeholders EXACTLY as they appear
- **DO NOT** rename, remove, or modify variable syntax

## SUBSTANTIAL IMPROVEMENT GUIDELINES
- Add new sections (e.g., Role Definition, Output Format, Error Handling)
- Provide concrete examples where helpful
- Remove redundancy and ambiguity
- Add structure with clear headers or XML tags
- Specify output format explicitly

## QUALITY SCORING (1-10)
Rate the optimized prompt on this scale:
- **9-10**: Exceptional - comprehensive, clear structure, examples, edge cases handled
- **7-8**: Strong - well-structured, clear instructions, most requirements covered
- **5-6**: Good - clear improvements, addresses main requirements
- **3-4**: Adequate - some improvements but missing key elements
- **1-2**: Poor - minimal improvement or missing critical requirements

Return a JSON object with this structure:
{
    "optimized_prompt": "The complete optimized system prompt",
    "score": 8.5,
    "analysis": "Detailed analysis of optimizations made",
    "improvements": ["Improvement 1", "Improvement 2", "..."],
    "suggestions": ["Additional suggestion 1", "Additional suggestion 2", "..."]
}"""

    user_message = f"""Please optimize this prompt.

**Use Case:** {project.use_case}

**Requirements:**
{project.requirements}

**Initial Prompt:**
```
{project.initial_prompt}
```

Transform this into a high-quality system prompt that addresses all requirements and follows best practices."""

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
        raise HTTPException(status_code=500, detail=f"Failed to parse optimization result: {str(e)}")


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


@app.post("/api/step2/ai-rewrite", response_model=PromptOptimizationResult)
async def ai_rewrite_optimization(project: ProjectInput, current_result: dict):
    """Step 2 AI Rewrite: Automatically refine the optimized prompt without user feedback"""
    system_prompt = """You are an expert prompt engineer. Your task is to critically review and further improve an already optimized prompt.

You will receive a CURRENT optimized prompt and its score. Your job is to:
1. Identify any remaining weaknesses or areas for improvement
2. Enhance clarity, structure, and effectiveness
3. Add more specific examples or edge case handling if needed
4. Improve the prompt to achieve a higher quality score

Be critical and look for subtle improvements that weren't addressed in the initial optimization.

Return a JSON object with this structure:
{
    "optimized_prompt": "The further improved system prompt",
    "score": 9.0,
    "analysis": "Detailed analysis of what was improved in this iteration",
    "improvements": ["Improvement 1", "Improvement 2", "..."],
    "suggestions": ["Additional suggestion 1", "Additional suggestion 2", "..."]
}"""

    user_message = f"""Please review and further improve this optimized prompt.

**Use Case:** {project.use_case}

**Requirements:** {project.requirements}

**CURRENT Optimized Prompt:**
```
{current_result.get('optimized_prompt', '')}
```

**Current Score:** {current_result.get('score', 0)} / 10

**CURRENT Analysis:**
{current_result.get('analysis', '')}

Critically review this prompt and identify any areas that can be further improved. Focus on:
- Clarity and specificity
- Edge case handling
- Examples and demonstrations
- Output format specifications
- Error handling instructions

Provide an enhanced version that addresses these areas."""

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
        raise HTTPException(status_code=500, detail=f"Failed to parse AI rewrite result: {str(e)}")


# ============================================================================
# STEP 3: EVALUATION PROMPT GENERATION
# ============================================================================

@app.post("/api/step3/generate-eval", response_model=EvaluationPromptResult)
async def generate_evaluation_prompt(project: ProjectInput, optimized_prompt: str):
    """Step 3: Generate evaluation prompt using 5-section structure"""
    system_prompt = """You are an expert in LLM evaluation. Your task is to create a comprehensive evaluation prompt following the 5-SECTION STRUCTURE:

### **I. Evaluator's Role & Goal**
- Define the evaluator's primary role
- State the specific goal: rigorously evaluate AI responses against core operational principles

### **II. Core Expectations**
Define what a high-quality output must do based on requirements.

### **III. Detailed 1-5 Rating Scale**
**Rating 1 (Very Poor):** Hallucination, complete irrelevance, broken format, major requirement violations
**Rating 2 (Poor):** Incorrect logic, partially ungrounded responses, significant requirement gaps
**Rating 3 (Acceptable):** Functionally correct but with flaws, minor logical gaps
**Rating 4 (Good):** High quality with minor room for improvement
**Rating 5 (Excellent):** Flawless execution, all requirements perfectly addressed

### **IV. Evaluation Task**
Provide step-by-step instructions for the evaluator.

### **V. Output Format**
```json
{
    "score": 3,
    "reasoning": "Detailed explanation of why this score was assigned"
}
```

**IMPORTANT:** The evaluation prompt must use placeholders {{input}} and {{output}}.

Return a JSON object:
{
    "eval_prompt": "Complete evaluation prompt with all 5 sections",
    "eval_criteria": ["Criterion 1", "Criterion 2", "..."],
    "rationale": "Brief explanation of the evaluation approach"
}"""

    user_message = f"""Create an evaluation prompt for this use case.

**Use Case:** {project.use_case}

**Requirements:** {project.requirements}

**System Prompt to Evaluate:**
```
{optimized_prompt}
```

Create a comprehensive evaluation prompt that will assess whether outputs from this system prompt meet all requirements and quality standards."""

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
        raise HTTPException(status_code=500, detail=f"Failed to parse evaluation prompt result: {str(e)}")


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


@app.post("/api/step3/ai-rewrite", response_model=EvaluationPromptResult)
async def ai_rewrite_eval_prompt(project: ProjectInput, optimized_prompt: str, current_result: dict):
    """Step 3 AI Rewrite: Automatically refine evaluation prompt without user feedback"""
    system_prompt = """You are an expert in LLM evaluation. Critically review and further improve an evaluation prompt.

Your job is to:
1. Identify any missing evaluation criteria
2. Make the rating scale more precise and actionable
3. Add specific examples for each rating level if missing
4. Improve clarity and reduce ambiguity
5. Ensure comprehensive coverage of edge cases

Return JSON:
{
    "eval_prompt": "The further improved evaluation prompt",
    "eval_criteria": ["Criterion 1", "..."],
    "rationale": "Explanation of improvements made"
}"""

    user_message = f"""Please review and further improve this evaluation prompt.

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

Critically review and enhance this evaluation prompt."""

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
        raise HTTPException(status_code=500, detail=f"Failed to parse AI rewrite: {str(e)}")


# ============================================================================
# STEP 4: TEST DATA GENERATION
# ============================================================================

@app.post("/api/step4/generate-testdata", response_model=TestDataResult)
async def generate_test_data(project: ProjectInput, optimized_prompt: str, num_cases: int = 10):
    """Step 4: Generate test data using distribution-based approach"""
    system_prompt = """You are an expert test data generator. Generate REALISTIC INPUT DATA (user queries/requests).

## Test Case Distribution
- **60% Positive** (typical, valid use cases)
- **20% Edge Cases** (boundary conditions, unusual but valid)
- **10% Negative** (invalid, out-of-scope)
- **10% Adversarial** (prompt injection, jailbreak attempts)

## OUTPUT FORMAT
Return a JSON object:
{
    "test_cases": [
        {
            "input": "The actual input text/query",
            "category": "positive|edge_case|negative|adversarial",
            "test_focus": "What this test case is designed to test",
            "expected_behavior": "How the system should handle this input"
        }
    ]
}"""

    user_message = f"""Generate {num_cases} test input cases for this system.

**Use Case:** {project.use_case}

**Requirements:** {project.requirements}

**System Prompt:**
```
{optimized_prompt}
```

Generate {num_cases} diverse, realistic INPUT cases following the distribution guidelines."""

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
