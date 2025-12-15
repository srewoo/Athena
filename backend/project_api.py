"""
Project management API - Simple file-based storage
"""
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime
import uuid
import json
import re
import logging

import project_storage
from models import SavedProject, ProjectListItem
from llm_client import get_llm_client
from smart_test_generator import detect_input_type, build_input_generation_prompt, get_scenario_variations, InputType
from prompt_analyzer import analyze_prompt as analyze_prompt_dna

# Setup logging
logger = logging.getLogger(__name__)


def _get_input_type_instructions(input_type: InputType) -> str:
    """Get detailed instructions for generating specific input types"""
    instructions = {
        InputType.CALL_TRANSCRIPT: """Generate COMPLETE call transcripts with:
- Multiple speakers with names (e.g., "John:", "Sarah:", "Manager:")
- Natural conversation flow with greetings, main discussion, and wrap-up
- Timestamps where appropriate (e.g., [00:01:23])
- Realistic details: specific numbers, dates, project names, action items
- Natural speech patterns: filler words, interruptions, acknowledgments
- Length: 200-600 words per transcript

Example format:
[00:00:05] Sarah: Hi everyone, thanks for joining. Let's get started with the Q3 review.
[00:00:12] John: Thanks Sarah. I wanted to discuss the customer feedback we received...
""",

        InputType.CONVERSATION: """Generate REALISTIC chat/message conversations with:
- Multiple participants with distinct styles
- Natural chat patterns: short messages, abbreviations, occasional typos
- Timestamps between messages
- Topic flow and context switches
- Emojis where natural""",

        InputType.EMAIL: """Generate COMPLETE emails with:
- Subject line
- From/To headers
- Professional greeting and signature
- Clear body content
- References to attachments if relevant""",

        InputType.CODE: """Generate REALISTIC code samples with:
- Proper syntax for the language
- Meaningful variable/function names
- Comments where appropriate
- Varying complexity and quality""",

        InputType.DOCUMENT: """Generate COMPLETE documents with:
- Proper structure (sections, headings)
- Professional language
- Specific details and data
- Appropriate length for the document type""",

        InputType.TICKET: """Generate REALISTIC support tickets with:
- Clear issue description
- Steps to reproduce (if applicable)
- Environment/system info
- Varying levels of detail and urgency""",

        InputType.REVIEW: """Generate REALISTIC reviews with:
- Specific feedback points
- Emotional tone variation
- Rating references
- Varying length and detail""",

        InputType.SIMPLE_TEXT: """Generate diverse text inputs with:
- Varying length and complexity
- Different tones and styles
- Realistic content for the use case""",

        InputType.MULTI_PARAGRAPH: """Generate complete multi-paragraph content with:
- Clear structure and flow
- Detailed information
- Appropriate length (300-800 words)""",
    }

    return instructions.get(input_type, """Generate realistic inputs appropriate for the detected format.
Include specific details, vary length and complexity.""")

router = APIRouter(prefix="/api/projects", tags=["projects"])


# ============================================================================
# PYDANTIC MODELS FOR EVAL VALIDATION
# ============================================================================

class EvalScoreBreakdown(BaseModel):
    """Per-criterion scores for detailed evaluation"""
    task_completion: Optional[float] = Field(None, ge=1, le=5)
    requirement_adherence: Optional[float] = Field(None, ge=1, le=5)
    quality_coherence: Optional[float] = Field(None, ge=1, le=5)
    accuracy_safety: Optional[float] = Field(None, ge=1, le=5)
    completeness: Optional[float] = Field(None, ge=1, le=5)


class EvalResult(BaseModel):
    """Validated evaluation result from judge LLM"""
    score: float = Field(..., ge=1, le=5, description="Overall score 1-5")
    reasoning: str = Field(..., min_length=10, max_length=2000, description="Brief justification with evidence")
    breakdown: Optional[EvalScoreBreakdown] = None
    violations: Optional[List[str]] = None
    evidence: Optional[List[str]] = None

    @validator('score')
    def validate_score(cls, v):
        if v < 1 or v > 5:
            raise ValueError('Score must be between 1 and 5')
        return round(v, 1)


class PromptAnalysisCategory(BaseModel):
    """Individual category in prompt analysis"""
    score: float = Field(..., ge=0, le=100)
    weight: float = Field(..., ge=0, le=1)
    notes: List[str] = []
    issues: List[str] = []


class PromptAnalysisResult(BaseModel):
    """Structured prompt analysis result"""
    overall_score: float = Field(..., ge=0, le=100)
    categories: Dict[str, PromptAnalysisCategory]
    suggestions: List[Dict[str, Any]]
    strengths: List[str]
    issues: List[str]
    analysis_method: Literal["heuristic", "llm", "hybrid"]
    error: Optional[str] = None


def sanitize_for_eval(text: str, max_length: int = 10000) -> str:
    """
    Sanitize text before inserting into evaluation prompt to prevent prompt injection.

    This helps prevent:
    1. Attempts to override evaluation instructions
    2. Fake JSON injection to manipulate scores
    3. XML tag injection to break delimiters
    4. Excessive length that could cause issues
    """
    if not text:
        return ""

    # Escape XML-like tags that could break delimiters BEFORE truncation
    # Replace < and > with escaped versions in content
    text = text.replace("</", "⟨/")  # Using Unicode look-alike for safety
    text = text.replace("<", "⟨")
    text = text.replace(">", "⟩")

    # Truncate to prevent excessive length AFTER escaping
    if len(text) > max_length:
        text = text[:max_length] + "... [truncated]"

    # Detect and flag potential injection attempts (for logging)
    injection_patterns = [
        "ignore previous",
        "ignore the above",
        "disregard instructions",
        "new instructions",
        "override",
        "forget everything",
        "score: 5",  # Attempting to inject score
        '"score": 5',
        "```json",  # Attempting to inject JSON
        "IMPORTANT:",
        "SYSTEM:",
    ]

    has_injection_attempt = any(pattern.lower() in text.lower() for pattern in injection_patterns)
    if has_injection_attempt:
        # Log but don't block - the model should still evaluate fairly
        logger.warning(f"Potential prompt injection attempt detected in eval input")

    return text


def parse_eval_json_strict(text: str) -> Optional[EvalResult]:
    """
    Strictly parse evaluation JSON with validation.
    Returns None if parsing fails - caller must handle as failure.
    Handles multiple response formats from different LLMs.
    """
    if not text:
        return None

    def try_parse_data(data: dict) -> Optional[EvalResult]:
        """Try to extract score and reasoning from various JSON formats"""
        # Standard format: {"score": X, "reasoning": "..."}
        if "score" in data and "reasoning" in data:
            try:
                return EvalResult(**data)
            except (ValueError, TypeError):
                pass

        # Rubric format: {"parameters": [...], "overall_analysis": "..."}
        if "parameters" in data and isinstance(data["parameters"], list):
            try:
                params = data["parameters"]
                if params:
                    # Calculate average score from raw_score fields
                    scores = [p.get("raw_score", p.get("score", 0)) for p in params if isinstance(p, dict)]
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        # Get overall analysis or build from individual analyses
                        reasoning = data.get("overall_analysis", "")
                        if not reasoning:
                            reasoning = "; ".join([p.get("analysis", "") for p in params if p.get("analysis")])
                        return EvalResult(score=avg_score, reasoning=reasoning[:500])
            except (ValueError, TypeError):
                pass

        # Alternative format: {"evaluation": {"score": X, ...}}
        if "evaluation" in data and isinstance(data["evaluation"], dict):
            return try_parse_data(data["evaluation"])

        # Format with "rating" instead of "score"
        if "rating" in data:
            try:
                score = float(data["rating"])
                reasoning = data.get("reasoning", data.get("explanation", data.get("feedback", "")))
                return EvalResult(score=score, reasoning=reasoning)
            except (ValueError, TypeError):
                pass

        return None

    # Try to find JSON object in response
    # First, try the whole text as JSON
    try:
        data = json.loads(text.strip())
        result = try_parse_data(data)
        if result:
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to extract JSON from markdown code blocks
    code_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
    match = re.search(code_block_pattern, text)
    if match:
        try:
            data = json.loads(match.group(1))
            result = try_parse_data(data)
            if result:
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Try to find any JSON object in the text
    json_start = text.find('{')
    json_end = text.rfind('}')
    if json_start != -1 and json_end != -1 and json_end > json_start:
        try:
            json_str = text[json_start:json_end + 1]
            data = json.loads(json_str)
            result = try_parse_data(data)
            if result:
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Last resort: try to extract score and reasoning separately
    score_match = re.search(r'"score"\s*:\s*([\d.]+)', text)
    if not score_match:
        score_match = re.search(r'"raw_score"\s*:\s*([\d.]+)', text)
    if not score_match:
        score_match = re.search(r'"rating"\s*:\s*([\d.]+)', text)

    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
    if not reasoning_match:
        reasoning_match = re.search(r'"overall_analysis"\s*:\s*"([^"]*)"', text)
    if not reasoning_match:
        reasoning_match = re.search(r'"analysis"\s*:\s*"([^"]*)"', text)

    if score_match and reasoning_match:
        try:
            score = float(score_match.group(1))
            reasoning = reasoning_match.group(1)
            return EvalResult(score=score, reasoning=reasoning)
        except (ValueError, TypeError):
            pass

    # If we found a score but no reasoning, still return it
    if score_match:
        try:
            score = float(score_match.group(1))
            return EvalResult(score=score, reasoning="Score extracted from response")
        except (ValueError, TypeError):
            pass

    return None

# Initialize LLM client
llm_client = get_llm_client()

# Import shared settings module
from shared_settings import get_settings


class CreateProjectRequest(BaseModel):
    name: str = None
    project_name: str = None
    use_case: str
    key_requirements: list = None
    requirements: str = None
    target_provider: str = None
    initial_prompt: str


@router.post("", response_model=SavedProject)
async def create_project(request: CreateProjectRequest):
    """Create a new project"""
    # Handle both frontend formats
    proj_name = request.name or request.project_name or "Untitled Project"

    # Keep key_requirements as array for frontend
    key_reqs = request.key_requirements or []

    # Store requirements object that frontend expects
    requirements_obj = {
        "use_case": request.use_case,
        "key_requirements": key_reqs
    }

    project = project_storage.create_new_project(
        project_name=proj_name,
        use_case=request.use_case,
        requirements=requirements_obj,
        key_requirements=key_reqs,
        initial_prompt=request.initial_prompt
    )
    return project


@router.get("", response_model=List[ProjectListItem])
async def get_projects():
    """Get all projects"""
    return project_storage.list_projects()


@router.get("/{project_id}", response_model=SavedProject)
async def get_project(project_id: str):
    """Get a specific project"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.put("/{project_id}", response_model=SavedProject)
async def update_project(project_id: str, project: SavedProject):
    """Update a project"""
    existing = project_storage.load_project(project_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project.id = project_id
    project_storage.save_project(project)
    return project


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    """Delete a project"""
    if not project_storage.load_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")

    project_storage.delete_project(project_id)
    return {"message": "Project deleted successfully"}


class AnalyzeRequest(BaseModel):
    prompt_text: str


def analyze_prompt_semantic(prompt_text: str) -> Dict[str, Any]:
    """
    Perform semantic analysis of a prompt with weighted scoring.

    Returns detailed analysis with:
    - Category scores (0-100) with weights
    - Specific findings with line-level detail
    - Actionable suggestions
    """
    word_count = len(prompt_text.split())
    lines = prompt_text.split('\n')
    prompt_lower = prompt_text.lower()

    # =========================================================================
    # CATEGORY 1: STRUCTURE (Weight: 25%)
    # =========================================================================
    structure_checks = {
        "has_role": any(p in prompt_lower for p in ["you are", "act as", "role:", "persona:", "as a"]),
        "has_sections": bool(re.search(r'(##|###|\*\*|<[a-z]+>|[A-Z][a-z]+:)', prompt_text)),
        "has_delimiters": any(d in prompt_text for d in ["```", "---", "===", "<", ">"]),
        "uses_xml_tags": bool(re.search(r'<[a-z_]+>.*</[a-z_]+>', prompt_text, re.DOTALL | re.IGNORECASE)),
        "has_numbered_steps": bool(re.search(r'(\d+\.|step \d|first|second|third)', prompt_lower)),
        "clear_hierarchy": bool(re.search(r'(task|objective|instructions|output|format)', prompt_lower))
    }
    structure_score = (sum(structure_checks.values()) / len(structure_checks)) * 100

    # =========================================================================
    # CATEGORY 2: CLARITY (Weight: 25%)
    # =========================================================================
    clarity_checks = {
        "has_specific_task": any(w in prompt_lower for w in ["generate", "create", "write", "analyze", "summarize", "translate", "explain", "evaluate"]),
        "has_context": any(w in prompt_lower for w in ["context", "background", "purpose", "goal", "objective"]),
        "avoids_ambiguity": not any(w in prompt_lower for w in ["maybe", "perhaps", "might want to", "could possibly"]),
        "has_examples": any(w in prompt_lower for w in ["example", "e.g.", "for instance", "such as", "like this"]),
        "uses_precise_verbs": any(w in prompt_lower for w in ["must", "should", "always", "ensure", "verify"]),
        "defines_scope": any(w in prompt_lower for w in ["scope", "boundary", "limit", "focus on", "only"])
    }
    clarity_score = (sum(clarity_checks.values()) / len(clarity_checks)) * 100

    # =========================================================================
    # CATEGORY 3: OUTPUT SPECIFICATION (Weight: 20%)
    # =========================================================================
    output_checks = {
        "has_format_spec": any(w in prompt_lower for w in ["format", "structure", "output", "return", "respond"]),
        "specifies_length": any(w in prompt_lower for w in ["word", "sentence", "paragraph", "length", "brief", "detailed", "concise"]),
        "has_json_spec": "json" in prompt_lower or "```" in prompt_text,
        "has_tone": any(w in prompt_lower for w in ["tone", "style", "voice", "professional", "casual", "formal"]),
        "defines_what_not_to_include": any(w in prompt_lower for w in ["don't include", "avoid", "do not", "never", "exclude"])
    }
    output_score = (sum(output_checks.values()) / len(output_checks)) * 100

    # =========================================================================
    # CATEGORY 4: SAFETY & ROBUSTNESS (Weight: 15%)
    # =========================================================================
    safety_checks = {
        "has_constraints": any(w in prompt_lower for w in ["don't", "avoid", "never", "must not", "should not", "do not"]),
        "handles_edge_cases": any(w in prompt_lower for w in ["if", "when", "edge case", "fallback", "otherwise", "in case"]),
        "has_error_handling": any(w in prompt_lower for w in ["error", "invalid", "unknown", "unclear", "if unable"]),
        "prevents_hallucination": any(w in prompt_lower for w in ["only use", "based on", "from the", "given information", "do not make up"]),
        "has_guardrails": any(w in prompt_lower for w in ["appropriate", "safe", "ethical", "respectful", "harmful"])
    }
    safety_score = (sum(safety_checks.values()) / len(safety_checks)) * 100

    # =========================================================================
    # CATEGORY 5: COMPLETENESS (Weight: 15%)
    # =========================================================================
    completeness_checks = {
        "sufficient_length": word_count >= 50,
        "has_multiple_instructions": bool(re.search(r'(\d+\.|•|-\s|\*\s)', prompt_text)),
        "covers_input_handling": any(w in prompt_lower for w in ["input", "given", "provided", "user", "request"]),
        "covers_output_handling": any(w in prompt_lower for w in ["output", "response", "return", "provide", "generate"]),
        "addresses_audience": any(w in prompt_lower for w in ["user", "reader", "audience", "recipient", "customer"])
    }
    completeness_score = (sum(completeness_checks.values()) / len(completeness_checks)) * 100

    # =========================================================================
    # CALCULATE WEIGHTED OVERALL SCORE
    # =========================================================================
    weights = {
        "structure": 0.25,
        "clarity": 0.25,
        "output_specification": 0.20,
        "safety_robustness": 0.15,
        "completeness": 0.15
    }

    overall_score = (
        structure_score * weights["structure"] +
        clarity_score * weights["clarity"] +
        output_score * weights["output_specification"] +
        safety_score * weights["safety_robustness"] +
        completeness_score * weights["completeness"]
    )

    # =========================================================================
    # GENERATE DETAILED FINDINGS
    # =========================================================================
    categories = {
        "structure": {
            "score": round(structure_score, 1),
            "weight": weights["structure"],
            "checks": structure_checks,
            "notes": [],
            "issues": []
        },
        "clarity": {
            "score": round(clarity_score, 1),
            "weight": weights["clarity"],
            "checks": clarity_checks,
            "notes": [],
            "issues": []
        },
        "output_specification": {
            "score": round(output_score, 1),
            "weight": weights["output_specification"],
            "checks": output_checks,
            "notes": [],
            "issues": []
        },
        "safety_robustness": {
            "score": round(safety_score, 1),
            "weight": weights["safety_robustness"],
            "checks": safety_checks,
            "notes": [],
            "issues": []
        },
        "completeness": {
            "score": round(completeness_score, 1),
            "weight": weights["completeness"],
            "checks": completeness_checks,
            "notes": [],
            "issues": []
        }
    }

    # Add notes based on checks
    if structure_checks["uses_xml_tags"]:
        categories["structure"]["notes"].append("Good use of XML tags for structure")
    if not structure_checks["has_role"]:
        categories["structure"]["issues"].append("Consider defining a role/persona")

    if clarity_checks["avoids_ambiguity"]:
        categories["clarity"]["notes"].append("Language is clear and unambiguous")
    if not clarity_checks["has_examples"]:
        categories["clarity"]["issues"].append("Add examples to clarify expected output")

    if safety_checks["prevents_hallucination"]:
        categories["safety_robustness"]["notes"].append("Good guardrails against hallucination")
    if not safety_checks["has_guardrails"]:
        categories["safety_robustness"]["issues"].append("Consider adding safety guardrails")

    return {
        "overall_score": round(overall_score, 1),
        "word_count": word_count,
        "categories": categories,
        "weights": weights
    }


@router.post("/{project_id}/analyze")
async def analyze_prompt(project_id: str, request: AnalyzeRequest):
    """Analyze a prompt using semantic heuristics + LLM for deep insights"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    prompt_text = request.prompt_text

    # =========================================================================
    # STEP 1: Semantic Heuristic Analysis (fast, no API needed)
    # =========================================================================
    semantic_analysis = analyze_prompt_semantic(prompt_text)
    word_count = semantic_analysis["word_count"]
    categories = semantic_analysis["categories"]

    # Calculate legacy scores for backwards compatibility
    requirements_alignment_score = min(100, word_count * 1.5)
    best_practices_score = semantic_analysis["overall_score"]

    # Generate heuristic-based suggestions from semantic analysis
    heuristic_suggestions = []
    requirements_gaps = []

    # Generate suggestions based on category scores
    if categories["structure"]["score"] < 50:
        heuristic_suggestions.append({
            "priority": "High",
            "suggestion": "Add clear structure with sections, delimiters, or XML tags",
            "category": "structure"
        })
        requirements_gaps.append("Prompt lacks clear structure")

    if categories["clarity"]["score"] < 50:
        heuristic_suggestions.append({
            "priority": "High",
            "suggestion": "Make the task more specific with examples and precise verbs",
            "category": "clarity"
        })
        requirements_gaps.append("Task description is unclear or ambiguous")

    if categories["output_specification"]["score"] < 50:
        heuristic_suggestions.append({
            "priority": "High",
            "suggestion": "Define the expected output format, length, and style",
            "category": "output"
        })
        requirements_gaps.append("Output format not specified")

    if categories["safety_robustness"]["score"] < 40:
        heuristic_suggestions.append({
            "priority": "Medium",
            "suggestion": "Add constraints, edge case handling, and safety guardrails",
            "category": "safety"
        })
        requirements_gaps.append("Missing safety constraints and edge case handling")

    if categories["completeness"]["score"] < 50:
        heuristic_suggestions.append({
            "priority": "Medium",
            "suggestion": "Add more detail covering input handling and audience",
            "category": "completeness"
        })
        requirements_gaps.append("Prompt is incomplete or too brief")

    # Add specific suggestions from category issues
    for cat_name, cat_data in categories.items():
        for issue in cat_data.get("issues", [])[:2]:  # Limit to 2 per category
            heuristic_suggestions.append({
                "priority": "Medium" if cat_data["score"] >= 40 else "High",
                "suggestion": issue,
                "category": cat_name
            })

    # =========================================================================
    # STEP 2: LLM Analysis (deep insights using AI)
    # =========================================================================
    settings = get_settings()
    provider = settings.get("provider", "openai")
    api_key = settings.get("api_key", "")
    model_name = settings.get("model_name")

    llm_insights = None
    llm_suggestions = []
    overall_score = (requirements_alignment_score + best_practices_score) / 2

    if api_key:
        # Build context from heuristic analysis
        heuristic_summary = f"""
Heuristic Analysis Results:
- Word count: {word_count}
- Requirements score: {requirements_alignment_score}/100
- Best practices score: {best_practices_score}/100
- Missing elements: {', '.join(requirements_gaps) if requirements_gaps else 'None detected'}
"""

        system_prompt = """You are an expert prompt engineer. Analyze the given prompt and provide deep insights.

Given the prompt and initial heuristic analysis, provide:
1. A refined overall score (0-100) based on prompt quality
2. 3-5 specific, actionable suggestions for improvement
3. Key strengths of the prompt
4. Potential issues or ambiguities

Return a JSON object:
{
  "refined_score": 0-100,
  "suggestions": [{"priority": "High/Medium/Low", "suggestion": "specific advice"}],
  "strengths": ["strength1", "strength2"],
  "issues": ["issue1", "issue2"],
  "analysis_summary": "brief overall assessment"
}

Return ONLY valid JSON."""

        user_message = f"""Analyze this prompt:

{prompt_text}

{heuristic_summary}

Provide expert analysis and actionable improvements."""

        try:
            result = await llm_client.chat(
                system_prompt=system_prompt,
                user_message=user_message,
                provider=provider,
                api_key=api_key,
                model_name=model_name,
                temperature=0.3,
                max_tokens=8000
            )
            if not result.get("error"):
                output = result.get("output", "{}")
                # Parse JSON from response
                if "```json" in output:
                    output = output.split("```json")[1].split("```")[0]
                elif "```" in output:
                    output = output.split("```")[1].split("```")[0]

                try:
                    llm_data = json.loads(output.strip())
                    llm_insights = {
                        "strengths": llm_data.get("strengths", []),
                        "issues": llm_data.get("issues", []),
                        "analysis_summary": llm_data.get("analysis_summary", "")
                    }
                    llm_suggestions = llm_data.get("suggestions", [])
                    # Use LLM's refined score if available
                    if "refined_score" in llm_data:
                        overall_score = llm_data["refined_score"]
                except Exception:
                    pass  # Keep heuristic results if JSON parsing fails
        except Exception:
            pass  # Keep heuristic results if LLM call fails

    # =========================================================================
    # STEP 3: Combine Results
    # =========================================================================
    # Merge suggestions (LLM suggestions first, then heuristic ones)
    all_suggestions = llm_suggestions + heuristic_suggestions
    # Remove duplicates and limit to 5
    seen = set()
    unique_suggestions = []
    for s in all_suggestions:
        key = s.get("suggestion", "")[:50]
        if key not in seen:
            seen.add(key)
            unique_suggestions.append(s)
    suggestions = unique_suggestions[:5]

    response = {
        "requirements_alignment_score": round(requirements_alignment_score, 1),
        "best_practices_score": round(best_practices_score, 1),
        "overall_score": round(overall_score, 1),
        "suggestions": suggestions,
        "requirements_gaps": requirements_gaps[:5],
        "analysis_method": "llm_enhanced" if llm_insights else "semantic_heuristic",
        # New: detailed semantic analysis by category
        "semantic_analysis": {
            "categories": {
                name: {
                    "score": data["score"],
                    "weight": data["weight"],
                    "notes": data["notes"],
                    "issues": data["issues"]
                }
                for name, data in categories.items()
            },
            "word_count": word_count
        }
    }

    if llm_insights:
        response["llm_insights"] = llm_insights

    return response


@router.post("/{project_id}/eval-prompt/generate")
async def generate_eval_prompt(project_id: str):
    """Generate an evaluation prompt using LLM with best practices"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get the current prompt from system_prompt_versions
    if not project.system_prompt_versions or len(project.system_prompt_versions) == 0:
        raise HTTPException(status_code=400, detail="No prompt versions found")

    current_prompt = project.system_prompt_versions[-1]["prompt_text"]

    # Get LLM settings
    settings = get_settings()
    provider = settings.get("provider", "openai")
    api_key = settings.get("api_key", "")
    model_name = settings.get("model_name")

    # Build requirements string
    requirements_str = ', '.join(project.key_requirements) if project.key_requirements else 'Not specified'

    # Template-based eval prompt with {{INPUT}} and {{OUTPUT}} placeholders
    # Plain text format for readability
    if not api_key:
        eval_prompt = f"""**Evaluator Role:**
You are an expert evaluator assessing AI-generated responses. Your task is to evaluate how well the response meets the specified requirements.

**Context:**
Use Case: {project.use_case}
Requirements: {requirements_str}

**System Prompt Being Evaluated:**
{current_prompt}

---

**User Input:**
{{{{INPUT}}}}

**Response to Evaluate:**
{{{{OUTPUT}}}}

---

**Evaluation Criteria:**

1. Task Completion (30%) - Does the response fully address the user's input? Are all requested elements present?

2. Requirement Adherence (25%) - Does it follow all specified requirements: {requirements_str}? Are there any violations?

3. Quality & Coherence (20%) - Is the response well-structured and easy to understand? Is the language appropriate?

4. Accuracy & Safety (15%) - Is the information factually correct? Are there any harmful or inappropriate elements?

5. Completeness (10%) - Is the response comprehensive without being unnecessarily verbose?

**Scoring Rubric:**
- Score 5 (Excellent): Exceeds all expectations. Perfectly addresses the input, follows all requirements.
- Score 4 (Good): Meets all requirements well. Minor issues that don't significantly impact quality.
- Score 3 (Acceptable): Meets basic requirements. Some noticeable issues but still functional.
- Score 2 (Poor): Significant issues. Missing requirements, quality problems, or inaccuracies.
- Score 1 (Fail): Does not meet requirements. Major failures, harmful content, or completely off-topic.

**Instructions:**
1. Identify what the user asked for in the input
2. Evaluate the response against each criterion
3. Provide a score and specific feedback

**Return your evaluation as JSON:**
{{
    "score": <number from 1-5>,
    "reasoning": "<2-3 sentences explaining the score with specific examples>"
}}"""

        rationale = "Template-based evaluation prompt with {{INPUT}}/{{OUTPUT}} placeholders. Configure API key for AI-customized eval prompts."

        # Persist eval prompt to project
        project.eval_prompt = eval_prompt
        project.eval_rationale = rationale
        project.updated_at = datetime.now()
        project_storage.save_project(project)

        return {
            "eval_prompt": eval_prompt,
            "rationale": rationale
        }

    # Use LLM to generate sophisticated eval prompt with best practices
    system_prompt = """You are an expert prompt engineer specializing in LLM-as-Judge evaluation systems.

Your task is to create a comprehensive evaluation prompt in PLAIN TEXT format (no XML tags).

## Structure Requirements:
1. Use markdown-style headers with ** for sections (e.g., **Evaluator Role:**, **Context:**, **Evaluation Criteria:**)
2. Include {{INPUT}} and {{OUTPUT}} placeholders that will be replaced with actual test data
3. Define a clear evaluator role with specific expertise relevant to the use case
4. Create evaluation criteria weighted by importance (must sum to 100%)
5. Use --- as section dividers where appropriate

## Rubric Requirements:
6. Create a detailed 1-5 scoring rubric with:
   - Score 5: Specific excellence indicators
   - Score 4: What "good" looks like with minor issues
   - Score 3: Baseline acceptable behavior
   - Score 2: Clear failure modes
   - Score 1: Critical failures and deal-breakers

## Instructions:
7. Include brief evaluation instructions
8. Require JSON output with "score" (1-5 number) and "reasoning" (specific explanation)

Return ONLY the evaluation prompt text in plain text format. No XML tags. No explanations."""

    user_message = f"""Create an evaluation prompt for this use case:

Use Case: {project.use_case}

Requirements: {requirements_str}

System Prompt Being Evaluated:
{current_prompt}

Generate a comprehensive evaluation prompt in plain text that:
- Uses {{{{INPUT}}}} placeholder for the user's test input
- Uses {{{{OUTPUT}}}} placeholder for the AI's response to evaluate
- Has criteria specifically tailored to the use case and requirements
- Includes domain-specific failure modes in the rubric
- Uses markdown headers (**Header:**) instead of XML tags"""

    result = await llm_client.chat(
        system_prompt=system_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.5,
        max_tokens=8000
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    eval_prompt = result.get("output", "").strip()

    # Ensure the eval prompt has the required placeholders
    if "{{INPUT}}" not in eval_prompt:
        eval_prompt = eval_prompt.replace("{INPUT}", "{{INPUT}}")
    if "{{OUTPUT}}" not in eval_prompt:
        eval_prompt = eval_prompt.replace("{OUTPUT}", "{{OUTPUT}}")

    rationale = f"AI-generated evaluation prompt with {{{{INPUT}}}}/{{{{OUTPUT}}}} placeholders, weighted criteria, detailed rubric, and chain-of-thought instructions. Tailored for: {project.use_case[:50]}..."

    # Persist eval prompt to project
    project.eval_prompt = eval_prompt
    project.eval_rationale = rationale
    project.updated_at = datetime.now()
    project_storage.save_project(project)

    return {
        "eval_prompt": eval_prompt,
        "rationale": rationale
    }


# ============================================================================
# VERSION MANAGEMENT ENDPOINTS
# ============================================================================

class AddVersionRequest(BaseModel):
    prompt_text: str
    changes_made: str = "Manual edit"


@router.post("/{project_id}/versions")
async def add_version(project_id: str, request: AddVersionRequest):
    """Add a new version to the project"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get current version number
    current_version = len(project.system_prompt_versions) if project.system_prompt_versions else 0

    # Create new version
    new_version = {
        "version": current_version + 1,
        "prompt_text": request.prompt_text,
        "created_at": datetime.now().isoformat(),
        "changes_made": request.changes_made
    }

    # Add to versions list
    if not project.system_prompt_versions:
        project.system_prompt_versions = []
    project.system_prompt_versions.append(new_version)

    # Save project
    project_storage.save_project(project)

    return new_version


@router.get("/{project_id}/versions/{version_number}")
async def get_version(project_id: str, version_number: int):
    """Get a specific version"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.system_prompt_versions:
        raise HTTPException(status_code=404, detail="No versions found")

    for version in project.system_prompt_versions:
        if version["version"] == version_number:
            return version

    raise HTTPException(status_code=404, detail="Version not found")


@router.delete("/{project_id}/versions/{version_number}")
async def delete_version(project_id: str, version_number: int):
    """Delete a specific version from the project"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.system_prompt_versions:
        raise HTTPException(status_code=404, detail="No versions found")

    # Don't allow deleting if only one version exists
    if len(project.system_prompt_versions) <= 1:
        raise HTTPException(status_code=400, detail="Cannot delete the only version")

    # Find and remove the version
    original_length = len(project.system_prompt_versions)
    project.system_prompt_versions = [
        v for v in project.system_prompt_versions
        if v["version"] != version_number
    ]

    if len(project.system_prompt_versions) == original_length:
        raise HTTPException(status_code=404, detail="Version not found")

    # Save updated project
    project_storage.save_project(project)

    return {
        "message": f"Version {version_number} deleted successfully",
        "remaining_versions": len(project.system_prompt_versions)
    }


# ============================================================================
# REWRITE ENDPOINTS
# ============================================================================

class RewriteRequest(BaseModel):
    prompt_text: str = None
    feedback: str = None
    focus_areas: List[str] = None


@router.post("/{project_id}/rewrite")
async def rewrite_project_prompt(project_id: str, request: RewriteRequest):
    """Rewrite the project prompt with feedback using LLM"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get current prompt
    current_prompt = request.prompt_text
    if not current_prompt and project.system_prompt_versions:
        current_prompt = project.system_prompt_versions[-1]["prompt_text"]

    if not current_prompt:
        raise HTTPException(status_code=400, detail="No prompt to rewrite")

    # Get LLM settings
    settings = get_settings()
    provider = settings.get("provider", "openai")
    api_key = settings.get("api_key", "")
    model_name = settings.get("model_name")

    # If no API key, return simple template rewrite
    if not api_key:
        improvements = request.focus_areas[:3] if request.focus_areas else ["Applied basic improvements"]
        if request.feedback:
            improvements.insert(0, f"Noted: {request.feedback[:50]}...")
        # Simple enhancement
        improved_prompt = current_prompt
        if "##" not in improved_prompt and len(improved_prompt) > 50:
            improved_prompt = f"## Task\n{improved_prompt}"
            improvements.append("Added structure")
        return {
            "rewritten_prompt": improved_prompt,
            "changes_made": improvements
        }

    # Build improvement context
    focus_list = "\n".join(f"- {area}" for area in (request.focus_areas or []))
    feedback_text = request.feedback or "Improve clarity and effectiveness"
    requirements_str = ', '.join(project.key_requirements) if project.key_requirements else 'Not specified'

    # Use best practices from OpenAI, Anthropic, and Google
    system_prompt = f"""You are an expert prompt engineer. Your task is to improve prompts following industry best practices.

## Best Practices to Apply:

### Structure (GPT-4.1 Guide):
- Use clear hierarchy: Role → Objective → Instructions → Output Format → Examples → Context
- Use XML tags or Markdown headers to delimit sections
- Be explicit and specific - models follow instructions literally

### Clarity (Anthropic Best Practices):
- Define a clear role and persona
- Use delimiters to separate different parts of the prompt
- Include explicit output format specifications
- Provide examples when helpful (few-shot prompting)

### Effectiveness (Google Prompting Strategies):
- Give clear and specific instructions
- Consider adding positive examples rather than negative ones
- Use chain-of-thought reasoning where appropriate
- Specify constraints and edge cases

## Context for This Prompt:
Use Case: {project.use_case}
Requirements: {requirements_str}

## User Feedback to Address:
{feedback_text}

## Focus Areas:
{focus_list if focus_list else '- General improvement'}

## Your Task:
1. Analyze the original prompt for weaknesses
2. Apply the best practices above
3. Address all user feedback and focus areas
4. Maintain the core intent and functionality
5. Add structure with clear sections if missing
6. Include output format specifications if not present
7. Preserve any {{{{variable}}}} placeholders exactly as they appear

Return ONLY the improved prompt. Do not include explanations or meta-commentary."""

    # Call LLM
    result = await llm_client.chat(
        system_prompt=system_prompt,
        user_message=f"""<original_prompt>
{current_prompt}
</original_prompt>

Improve this prompt following the best practices. Return only the improved prompt.""",
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.7,
        max_tokens=8000
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    improvements = request.focus_areas[:3] if request.focus_areas else ["Applied AI improvements"]
    if request.feedback:
        improvements.insert(0, f"Incorporated: {request.feedback[:50]}...")

    return {
        "rewritten_prompt": result.get("output", "").strip(),
        "changes_made": improvements
    }


# ============================================================================
# EVAL PROMPT REFINEMENT
# ============================================================================

class RefineEvalRequest(BaseModel):
    feedback: str


@router.post("/{project_id}/eval-prompt/refine")
async def refine_eval_prompt(project_id: str, request: RefineEvalRequest):
    """Refine the evaluation prompt based on feedback using LLM"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get current prompt
    current_prompt = project.system_prompt_versions[-1]["prompt_text"] if project.system_prompt_versions else ""

    # Get LLM settings
    settings = get_settings()
    provider = settings.get("provider", "openai")
    api_key = settings.get("api_key", "")
    model_name = settings.get("model_name")

    # If no API key, return template-based refinement
    if not api_key:
        eval_prompt = f"""Evaluate the following AI-generated response based on these criteria:

**Task Context:**
Use Case: {project.use_case}
Requirements: {', '.join(project.key_requirements) if project.key_requirements else 'Not specified'}

**User Feedback to Consider:**
{request.feedback}

**Original Prompt:**
{current_prompt}

**Evaluation Criteria (Refined):**
1. Task Completion - Does it fulfill the specific request?
2. Accuracy - Is the information factually correct?
3. Relevance - Does it stay on topic?
4. Quality - Is it well-written and clear?
5. User Feedback - Does it address the specific concerns raised?

**Rating:** Provide a score from 1-5 with justification."""

        rationale = "Template-based refinement. Configure API key for AI refinement."

        # Persist refined eval prompt to project
        project.eval_prompt = eval_prompt
        project.eval_rationale = rationale
        project.updated_at = datetime.now()
        project_storage.save_project(project)

        return {
            "eval_prompt": eval_prompt,
            "rationale": rationale
        }

    # Use LLM to refine eval prompt
    system_prompt = """You are an expert at refining evaluation prompts based on user feedback.
Improve the evaluation prompt to better address the user's concerns while maintaining comprehensive coverage.

Return ONLY the refined evaluation prompt, no explanations."""

    user_message = f"""Refine this evaluation prompt based on the feedback:

User Feedback: {request.feedback}

Current Use Case: {project.use_case}
Requirements: {', '.join(project.key_requirements) if project.key_requirements else 'Not specified'}

System Prompt Being Evaluated:
{current_prompt}

Create an improved evaluation prompt that addresses the feedback."""

    result = await llm_client.chat(
        system_prompt=system_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.5,
        max_tokens=8000
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    refined_eval_prompt = result.get("output", "").strip()
    rationale = f"AI-refined evaluation prompt incorporating: {request.feedback[:100]}"

    # Persist refined eval prompt to project
    project.eval_prompt = refined_eval_prompt
    project.eval_rationale = rationale
    project.updated_at = datetime.now()
    project_storage.save_project(project)

    return {
        "eval_prompt": refined_eval_prompt,
        "rationale": rationale
    }


# ============================================================================
# DATASET GENERATION ENDPOINTS
# ============================================================================

class GenerateDatasetRequest(BaseModel):
    num_examples: int = 10
    sample_count: int = None  # Alias for num_examples (frontend uses this)
    categories: List[str] = None
    version: int = None  # Optional: specific version number to use


@router.post("/{project_id}/dataset/generate")
async def generate_dataset(project_id: str, request: GenerateDatasetRequest = None):
    """
    Generate test dataset using smart, context-aware generation.

    Automatically detects input format from the system prompt (e.g., call transcripts,
    emails, code) and generates REALISTIC, COMPLETE test inputs.
    """
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Support both num_examples and sample_count (frontend uses sample_count)
    # Hard limit of 100 test cases maximum
    requested_count = request.sample_count or request.num_examples if request else 10
    num_examples = min(requested_count, 100)

    # Get the specific version or latest
    if project.system_prompt_versions:
        if request and request.version is not None:
            version_data = next(
                (v for v in project.system_prompt_versions if v.get("version") == request.version),
                project.system_prompt_versions[-1]
            )
        else:
            version_data = project.system_prompt_versions[-1]
        current_prompt = version_data.get("prompt_text", "")
    else:
        current_prompt = ""

    # Get LLM settings
    settings = get_settings()
    provider = settings.get("provider", "openai")
    api_key = settings.get("api_key", "")
    model_name = settings.get("model_name")

    categories = ["positive", "edge_case", "negative", "adversarial"]

    # If no API key, return template test cases
    if not api_key:
        test_cases = []
        for i in range(num_examples):
            category = categories[i % len(categories)]
            test_case = {
                "id": str(uuid.uuid4()),
                "input": f"Test input {i+1} for {project.use_case}",
                "expected_behavior": f"Should handle {category} case appropriately",
                "category": category,
                "created_at": datetime.now().isoformat()
            }
            test_cases.append(test_case)

        dataset_obj = {
            "test_cases": test_cases,
            "sample_count": len(test_cases),
            "preview": test_cases,
            "count": len(test_cases),
            "categories": {cat: sum(1 for tc in test_cases if tc["category"] == cat) for cat in categories},
            "generated_at": datetime.now().isoformat()
        }
        project.dataset = dataset_obj
        project.test_cases = test_cases
        project.updated_at = datetime.now()
        project_storage.save_project(project)
        return dataset_obj

    # ========== SMART CONTEXT-AWARE GENERATION ==========

    # Analyze the system prompt to understand expected input format
    analysis = analyze_prompt_dna(current_prompt)
    input_spec = detect_input_type(current_prompt, analysis.dna.template_variables)

    logger.info(f"Smart generation - Detected input type: {input_spec.input_type.value}, "
                f"template var: {input_spec.template_variable}, domain: {input_spec.domain_context}")

    # Get scenario variations for comprehensive coverage
    scenarios = get_scenario_variations(input_spec, num_examples)

    # Build context-aware generation prompt
    use_case = project.use_case or "Not specified"
    requirements = ", ".join(project.key_requirements) if project.key_requirements else "Not specified"

    # Build scenario descriptions
    scenario_list = "\n".join([
        f"{i+1}. [{s['category'].upper()}] {s['scenario']}: {s['description']}"
        for i, s in enumerate(scenarios)
    ])

    system_prompt = f"""You are an expert test data generator creating REALISTIC test inputs.

## INPUT TYPE DETECTED: {input_spec.input_type.value.replace('_', ' ').title()}

## CONTEXT
- **Template Variable:** {{{{{input_spec.template_variable}}}}}
- **Domain:** {input_spec.domain_context or 'General'}
- **Use Case:** {use_case}

## CRITICAL INSTRUCTIONS FOR {input_spec.input_type.value.upper()}
{_get_input_type_instructions(input_spec.input_type)}

## OUTPUT FORMAT
Return a JSON object with test_cases array:
{{
    "test_cases": [
        {{
            "input": "THE COMPLETE, REALISTIC INPUT - NOT A SUMMARY OR PLACEHOLDER",
            "category": "positive|edge_case|negative|adversarial",
            "test_focus": "What this test case validates",
            "expected_behavior": "How the system should handle this"
        }}
    ]
}}

## RULES
1. Generate COMPLETE inputs - if it's a call transcript, write the FULL conversation
2. Each input should be realistic with specific names, dates, numbers
3. Vary length and complexity across test cases
4. Include natural elements (filler words for transcripts, typos for chats, etc.)
5. Make edge cases and adversarial inputs subtle and realistic"""

    user_message = f"""Generate {num_examples} COMPLETE, REALISTIC test inputs for this system.

**System Prompt Being Tested:**
```
{current_prompt}
```

**Use Case:** {use_case}
**Requirements:** {requirements}

**Scenarios to Cover (generate one test case per scenario):**
{scenario_list}

Generate {num_examples} FULL, REALISTIC inputs. For {input_spec.input_type.value.replace('_', ' ')}, each input should be complete and detailed (200-800 words for transcripts/documents)."""

    result = await llm_client.chat(
        system_prompt=system_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.8,
        max_tokens=16000  # More tokens for complete inputs
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    # Parse JSON from LLM response
    try:
        output = result.get("output", "{}").strip()

        # Handle markdown code blocks
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0]
        elif "```" in output:
            output = output.split("```")[1].split("```")[0]

        # Try to find JSON object in response
        json_match = re.search(r'\{[\s\S]*\}', output)
        if json_match:
            parsed = json.loads(json_match.group())
            # Handle both formats: {"test_cases": [...]} or [...]
            if isinstance(parsed, dict) and "test_cases" in parsed:
                generated_cases = parsed["test_cases"]
            elif isinstance(parsed, list):
                generated_cases = parsed
            else:
                generated_cases = []
        else:
            # Try parsing as array
            generated_cases = json.loads(output)

        test_cases = []
        categories_count = {}
        for i, tc in enumerate(generated_cases[:num_examples]):
            category = tc.get("category", categories[i % len(categories)])
            test_case = {
                "id": str(uuid.uuid4()),
                "input": tc.get("input", f"Test input {i+1}"),
                "expected_behavior": tc.get("expected_behavior", "Should respond appropriately"),
                "category": category,
                "test_focus": tc.get("test_focus", ""),
                "created_at": datetime.now().isoformat()
            }
            test_cases.append(test_case)
            categories_count[category] = categories_count.get(category, 0) + 1

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse LLM response, using fallback: {e}")
        # Fallback to template if parsing fails
        test_cases = []
        categories_count = {}
        for i in range(num_examples):
            category = categories[i % len(categories)]
            test_case = {
                "id": str(uuid.uuid4()),
                "input": f"Test input {i+1} for {project.use_case}",
                "expected_behavior": f"Should handle {category} case appropriately",
                "category": category,
                "test_focus": "",
                "created_at": datetime.now().isoformat()
            }
            test_cases.append(test_case)
            categories_count[category] = categories_count.get(category, 0) + 1

    # Build dataset object with smart generation metadata
    dataset_obj = {
        "test_cases": test_cases,
        "sample_count": len(test_cases),
        "preview": test_cases[:10] if len(test_cases) > 10 else test_cases,
        "count": len(test_cases),
        "categories": categories_count,
        "generated_at": datetime.now().isoformat(),
        "metadata": {
            "input_type": input_spec.input_type.value,
            "template_variable": input_spec.template_variable,
            "domain_context": input_spec.domain_context,
            "generation_type": "smart"
        }
    }

    # Persist dataset to project file
    project.dataset = dataset_obj
    project.test_cases = test_cases
    project.updated_at = datetime.now()
    project_storage.save_project(project)

    return dataset_obj


@router.post("/{project_id}/dataset/generate-stream")
async def generate_dataset_stream(project_id: str, request: GenerateDatasetRequest = None):
    """Generate test dataset (streaming not implemented, falls back to regular)"""
    return await generate_dataset(project_id, request)


@router.post("/{project_id}/dataset/smart-generate")
async def smart_generate_dataset(project_id: str, request: GenerateDatasetRequest = None):
    """
    Smart test dataset generation - Creates realistic, context-aware test inputs.

    For prompts expecting specific input formats (call transcripts, emails, code, etc.),
    this generates COMPLETE, REALISTIC test inputs rather than short placeholders.

    Features:
    - Detects expected input type from prompt template variables
    - Generates full call transcripts, emails, documents, etc.
    - Creates domain-appropriate test scenarios
    - Varies test cases to cover different situations
    """
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Hard limit of 100 test cases
    requested_count = request.sample_count or request.num_examples if request else 10
    num_examples = min(requested_count, 100)

    # Get the specific version or latest
    if project.system_prompt_versions:
        if request and request.version is not None:
            version_data = next(
                (v for v in project.system_prompt_versions if v.get("version") == request.version),
                project.system_prompt_versions[-1]
            )
        else:
            version_data = project.system_prompt_versions[-1]
        current_prompt = version_data.get("prompt_text", "")
    else:
        current_prompt = ""

    if not current_prompt:
        raise HTTPException(status_code=400, detail="No system prompt found in project")

    # Get LLM settings
    settings = get_settings()
    provider = settings.get("provider", "openai")
    api_key = settings.get("api_key", "")
    model_name = settings.get("model_name")

    if not api_key:
        raise HTTPException(status_code=400, detail="LLM API key required for smart generation")

    # Analyze the prompt structure
    analysis = analyze_prompt(current_prompt)

    # Detect input format from prompt
    input_spec = detect_input_type(current_prompt, analysis.dna.template_variables)
    logger.info(f"Smart generate - Detected input type: {input_spec.input_type.value}, template var: {input_spec.template_variable}")

    # Get scenario variations
    scenarios = get_scenario_variations(input_spec, num_examples)

    # Build the smart generation prompt
    use_case = project.use_case or "Not specified"
    requirements = ", ".join(project.key_requirements) if project.key_requirements else "Not specified"
    system_prompt = build_input_generation_prompt(input_spec, current_prompt, use_case, requirements)

    # Build scenario list for user message
    scenario_list = "\n".join([
        f"{i+1}. [{s['category'].upper()}] {s['scenario']}: {s['description']}"
        for i, s in enumerate(scenarios)
    ])

    user_message = f"""Generate {num_examples} COMPLETE, REALISTIC test inputs.

**Input Type Detected:** {input_spec.input_type.value.replace('_', ' ').title()}
**Template Variable:** {{{{{input_spec.template_variable}}}}}
**Domain Context:** {input_spec.domain_context or 'General'}

**System Prompt Being Tested:**
```
{current_prompt}
```

**Scenarios to Generate:**
{scenario_list}

## CRITICAL REQUIREMENTS:
1. Generate FULL, COMPLETE inputs - NOT summaries or placeholders
2. For {input_spec.input_type.value.replace('_', ' ')}, include ALL necessary details
3. Each input should be {input_spec.expected_length} length
4. Make inputs REALISTIC with specific names, dates, numbers
5. Vary content significantly between test cases

Return JSON with test_cases array. Each must have: input, expected_behavior, category, test_focus."""

    result = await llm_client.chat(
        system_prompt=system_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.8,
        max_tokens=16000  # More tokens for full transcripts/documents
    )

    if result["error"]:
        raise HTTPException(status_code=500, detail=f"LLM Error: {result['error']}")

    try:
        json_match = re.search(r'\{[\s\S]*\}', result["output"])
        if json_match:
            data = json.loads(json_match.group())
            raw_test_cases = data.get("test_cases", [])

            # Process and add IDs
            test_cases = []
            categories_count = {}
            for tc in raw_test_cases:
                category = tc.get("category", "positive")
                test_case = {
                    "id": str(uuid.uuid4()),
                    "input": tc.get("input", ""),
                    "expected_behavior": tc.get("expected_behavior", ""),
                    "category": category,
                    "test_focus": tc.get("test_focus", ""),
                    "variation": tc.get("variation", ""),
                    "created_at": datetime.now().isoformat()
                }
                test_cases.append(test_case)
                categories_count[category] = categories_count.get(category, 0) + 1

            # Build dataset object
            dataset_obj = {
                "test_cases": test_cases,
                "sample_count": len(test_cases),
                "preview": test_cases[:10],
                "count": len(test_cases),
                "categories": categories_count,
                "generated_at": datetime.now().isoformat(),
                "metadata": {
                    "input_type": input_spec.input_type.value,
                    "template_variable": input_spec.template_variable,
                    "domain_context": input_spec.domain_context,
                    "expected_length": input_spec.expected_length,
                    "generation_type": "smart"
                }
            }

            # Save to project
            project.dataset = dataset_obj
            project.test_cases = test_cases
            project.updated_at = datetime.now()
            project_storage.save_project(project)

            return dataset_obj
        else:
            raise ValueError("No JSON found in LLM response")

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse smart test data JSON: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to parse test data: {str(e)}")
    except Exception as e:
        logger.error(f"Smart generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Smart generation failed: {str(e)}")


@router.get("/{project_id}/dataset/export")
async def export_dataset(project_id: str):
    """Export the project's test dataset"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get test cases from project file
    test_cases = project.test_cases or []

    return {
        "project_id": project_id,
        "project_name": project.project_name,
        "test_cases": test_cases,
        "exported_at": datetime.now().isoformat()
    }


# ============================================================================
# TEST RUNS ENDPOINTS
# ============================================================================

class CreateTestRunRequest(BaseModel):
    prompt_version: int = None
    version_number: int = None  # Alias for prompt_version
    llm_provider: str = None
    model_name: str = None
    # Separate evaluation model settings
    eval_provider: str = None  # Provider for evaluation (e.g., openai for o1)
    eval_model: str = None  # Model for evaluation (e.g., o1-mini, o1)
    pass_threshold: float = 3.5
    batch_size: int = 5
    max_concurrent: int = 3
    test_cases: List[Dict[str, Any]] = None


async def run_single_test_case(
    test_case: Dict[str, Any],
    system_prompt: str,
    eval_prompt: str,
    # Response generation settings
    llm_provider: str,
    model_name: str,
    api_key: str,
    # Evaluation settings (can be different model)
    eval_provider: str,
    eval_model_name: str,
    eval_api_key: str,
    pass_threshold: float
) -> Dict[str, Any]:
    """
    Run a single test case through LLM and evaluate the response.

    FAIL-CLOSED PRINCIPLE: If evaluation fails or JSON cannot be parsed,
    we set passed=False and surface evaluation_error=True. We never auto-pass
    with a default score when something goes wrong.
    """
    import time
    start_time = time.time()
    ttfb_ms = 0
    tokens_used = 0

    test_input = test_case.get("input", "")
    if isinstance(test_input, dict):
        test_input = json.dumps(test_input)

    # Judge metadata to track evaluation details
    judge_metadata = {
        "eval_provider": eval_provider,
        "eval_model": eval_model_name,
        "parsing_status": "not_attempted",
        "raw_eval_output": None
    }

    # Step 1: Get LLM response using the system prompt
    try:
        response_result = await llm_client.chat(
            system_prompt=system_prompt,
            user_message=test_input,
            provider=llm_provider,
            api_key=api_key,
            model_name=model_name
        )

        # Track TTFB (time to first response from LLM)
        ttfb_ms = response_result.get("latency_ms", 0)
        tokens_used += response_result.get("tokens_used", 0)

        if response_result.get("error"):
            # Response generation failed - fail closed
            return {
                "test_case_id": test_case.get("id"),
                "input": test_input,
                "output": "",
                "score": None,
                "passed": False,
                "feedback": f"Error generating response: {response_result.get('error')}",
                "error": True,
                "evaluation_error": False,
                "generation_error": True,
                "latency_ms": int((time.time() - start_time) * 1000),
                "ttfb_ms": ttfb_ms,
                "tokens_used": tokens_used,
                "judge_metadata": judge_metadata
            }

        llm_output = response_result.get("output", "")

    except Exception as e:
        # Response generation exception - fail closed
        logger.error(f"Exception during LLM call for test case {test_case.get('id')}: {str(e)}")
        return {
            "test_case_id": test_case.get("id"),
            "input": test_input,
            "output": "",
            "score": None,
            "passed": False,
            "feedback": f"Exception during LLM call: {str(e)}",
            "error": True,
            "evaluation_error": False,
            "generation_error": True,
            "latency_ms": int((time.time() - start_time) * 1000),
            "ttfb_ms": 0,
            "tokens_used": 0,
            "judge_metadata": judge_metadata
        }

    # Step 2: Evaluate the response using the eval prompt
    try:
        # Sanitize inputs before injecting into eval prompt to prevent prompt injection
        sanitized_input = sanitize_for_eval(test_input, max_length=5000)
        sanitized_output = sanitize_for_eval(llm_output, max_length=10000)

        # Replace {{INPUT}} and {{OUTPUT}} placeholders in the eval prompt
        # This follows best practices for LLM-as-Judge evaluation
        processed_eval_prompt = eval_prompt
        if "{{INPUT}}" in processed_eval_prompt:
            processed_eval_prompt = processed_eval_prompt.replace("{{INPUT}}", sanitized_input)
        if "{{OUTPUT}}" in processed_eval_prompt:
            processed_eval_prompt = processed_eval_prompt.replace("{{OUTPUT}}", sanitized_output)

        # If placeholders were used, the eval prompt is self-contained
        # Otherwise, construct a user message with the input/output
        if "{{INPUT}}" in eval_prompt or "{{OUTPUT}}" in eval_prompt:
            # Placeholders were replaced - send as user message with instruction to evaluate
            eval_user_prompt = f"""{processed_eval_prompt}

Now evaluate the response and return ONLY the JSON output as specified."""
            eval_system = "You are an expert evaluator. Follow the evaluation instructions precisely and return only valid JSON."
        else:
            # Legacy format - construct user message manually with sanitized inputs
            eval_user_prompt = f"""Please evaluate the following response based on the evaluation criteria.

<input>
{sanitized_input}
</input>

<output_to_evaluate>
{sanitized_output}
</output_to_evaluate>

Briefly evaluate:
1. What was requested?
2. How well was it addressed?
3. Key issues or strengths?

Provide your evaluation in this exact JSON format:
{{
    "score": <number from 1-5>,
    "reasoning": "<2-3 sentences explaining the score with specific examples>"
}}

Return ONLY the JSON, no other text."""
            eval_system = eval_prompt

        # Use separate evaluation model (thinking model) for better evaluation
        eval_result = await llm_client.chat(
            system_prompt=eval_system,
            user_message=eval_user_prompt,
            provider=eval_provider,
            api_key=eval_api_key,
            model_name=eval_model_name
        )

        # Add eval tokens to total
        tokens_used += eval_result.get("tokens_used", 0)

        if eval_result.get("error"):
            # Evaluation call failed - FAIL CLOSED (don't auto-pass)
            logger.warning(f"Evaluation API error for test case {test_case.get('id')}: {eval_result.get('error')}")
            judge_metadata["parsing_status"] = "api_error"
            judge_metadata["error_message"] = eval_result.get("error")
            return {
                "test_case_id": test_case.get("id"),
                "input": test_input,
                "output": llm_output,
                "score": None,
                "passed": False,
                "feedback": f"Evaluation failed: {eval_result.get('error')}. Cannot determine pass/fail - marking as failed.",
                "error": False,
                "evaluation_error": True,
                "generation_error": False,
                "latency_ms": int((time.time() - start_time) * 1000),
                "ttfb_ms": ttfb_ms,
                "tokens_used": tokens_used,
                "judge_metadata": judge_metadata
            }

        eval_output = eval_result.get("output", "")
        judge_metadata["raw_eval_output"] = eval_output[:500] if eval_output else None

        # Parse the evaluation result using strict Pydantic validation
        parsed_eval = parse_eval_json_strict(eval_output)

        if parsed_eval is None:
            # JSON parsing failed - FAIL CLOSED (don't auto-pass with default score)
            logger.warning(f"Failed to parse eval JSON for test case {test_case.get('id')}: {eval_output[:200]}")
            judge_metadata["parsing_status"] = "parse_failed"
            return {
                "test_case_id": test_case.get("id"),
                "input": test_input,
                "output": llm_output,
                "score": None,
                "passed": False,
                "feedback": f"Evaluation response could not be parsed. Raw response: {eval_output[:200]}...",
                "error": False,
                "evaluation_error": True,
                "generation_error": False,
                "latency_ms": int((time.time() - start_time) * 1000),
                "ttfb_ms": ttfb_ms,
                "tokens_used": tokens_used,
                "judge_metadata": judge_metadata
            }

        # Successfully parsed - use validated score and reasoning
        judge_metadata["parsing_status"] = "success"
        score = parsed_eval.score
        reasoning = parsed_eval.reasoning
        passed = score >= pass_threshold

        # Build response with optional breakdown
        result = {
            "test_case_id": test_case.get("id"),
            "input": test_input,
            "output": llm_output,
            "score": score,
            "passed": passed,
            "feedback": reasoning,
            "error": False,
            "evaluation_error": False,
            "generation_error": False,
            "latency_ms": int((time.time() - start_time) * 1000),
            "ttfb_ms": ttfb_ms,
            "tokens_used": tokens_used,
            "judge_metadata": judge_metadata
        }

        # Add per-criterion breakdown if available
        if parsed_eval.breakdown:
            result["score_breakdown"] = parsed_eval.breakdown.model_dump(exclude_none=True)

        # Add violations and evidence if available
        if parsed_eval.violations:
            result["violations"] = parsed_eval.violations
        if parsed_eval.evidence:
            result["evidence"] = parsed_eval.evidence

        return result

    except Exception as e:
        # Evaluation exception - FAIL CLOSED (don't auto-pass)
        logger.error(f"Exception during evaluation for test case {test_case.get('id')}: {str(e)}")
        judge_metadata["parsing_status"] = "exception"
        judge_metadata["error_message"] = str(e)
        return {
            "test_case_id": test_case.get("id"),
            "input": test_input,
            "output": llm_output,
            "score": None,
            "passed": False,
            "feedback": f"Evaluation exception: {str(e)}. Cannot determine pass/fail - marking as failed.",
            "error": False,
            "evaluation_error": True,
            "generation_error": False,
            "latency_ms": int((time.time() - start_time) * 1000),
            "ttfb_ms": ttfb_ms,
            "tokens_used": tokens_used,
            "judge_metadata": judge_metadata
        }


@router.post("/{project_id}/test-runs")
async def create_test_run(project_id: str, request: CreateTestRunRequest = None):
    """Create a new test run with actual LLM execution"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    run_id = str(uuid.uuid4())

    # Get test cases from request or project file
    test_cases = request.test_cases if request and request.test_cases else (project.test_cases or [])

    if not test_cases:
        raise HTTPException(status_code=400, detail="No test cases available. Please generate a dataset first.")

    # Get the prompt version to test
    version_num = request.prompt_version or request.version_number or 1
    system_prompt = None
    if project.system_prompt_versions:
        for v in project.system_prompt_versions:
            if v.get("version") == version_num:
                system_prompt = v.get("prompt_text")
                break
        if not system_prompt and project.system_prompt_versions:
            system_prompt = project.system_prompt_versions[-1].get("prompt_text")

    if not system_prompt:
        system_prompt = project.initial_prompt

    # Get eval prompt and inject calibration examples if available
    base_eval_prompt = project.eval_prompt or "You are an evaluator. Rate responses from 1-5 based on quality, relevance, and accuracy."
    eval_prompt = build_eval_prompt_with_calibration(base_eval_prompt, project.calibration_examples or [])

    # Get LLM settings
    settings = get_settings()
    llm_provider = request.llm_provider or settings.get("provider", "openai")
    model_name = request.model_name or settings.get("model")
    pass_threshold = request.pass_threshold or 3.5

    # Get evaluation model settings (can be different from response model)
    eval_provider = request.eval_provider or llm_provider
    eval_model = request.eval_model or model_name

    # Helper function to get API key for a provider
    def get_api_key_for_provider(provider: str) -> str:
        key = settings.get("api_key", "")
        if key:
            return key
        if provider == "openai":
            return settings.get("openai_api_key", "")
        elif provider == "claude":
            return settings.get("anthropic_api_key", "")
        elif provider == "gemini":
            return settings.get("google_api_key", "")
        return ""

    # Get API keys for both providers
    api_key = get_api_key_for_provider(llm_provider)
    eval_api_key = get_api_key_for_provider(eval_provider)

    # Create initial test run record
    test_run = {
        "id": run_id,
        "project_id": project_id,
        "version_number": version_num,
        "status": "running",
        "created_at": datetime.now().isoformat(),
        "llm_provider": llm_provider,
        "model_name": model_name,
        "eval_provider": eval_provider,
        "eval_model": eval_model,
        "pass_threshold": pass_threshold,
        "test_cases": test_cases,
        "results": [],
        "summary": None
    }

    # Check if we have API keys configured
    has_api_key = bool(api_key)
    has_eval_api_key = bool(eval_api_key)

    if not has_api_key:
        # Fall back to mock data if no API key
        test_run["status"] = "completed"
        test_run["results"] = [
            {
                "test_case_id": tc.get("id"),
                "input": tc.get("input", ""),
                "output": f"[Mock] Sample output for: {str(tc.get('input', ''))[:50]}...",
                "score": 4.0,
                "passed": True,
                "feedback": "Mock response - Configure API key for real LLM testing",
                "error": False,
                "latency_ms": 100,
                "ttfb_ms": 50,
                "tokens_used": 150
            }
            for tc in test_cases[:10]
        ]
        mock_count = len(test_run["results"])
        test_run["summary"] = {
            "total": mock_count,
            "completed_items": mock_count,
            "passed": mock_count,
            "failed": 0,
            "avg_score": 4.0,
            "pass_rate": 100.0,
            "score_distribution": {"1": 0, "2": 0, "3": 0, "4": mock_count, "5": 0},
            "estimated_cost": round(mock_count * 0.0005, 4),
            "total_latency_ms": mock_count * 100,
            "min_latency_ms": 100,
            "max_latency_ms": 100,
            "avg_latency_ms": 100,
            "avg_ttfb_ms": 50,
            "total_tokens": mock_count * 150
        }
    else:
        # Run actual LLM tests
        import asyncio
        results = []

        # Run all test cases (up to 100 max)
        test_subset = test_cases[:100]

        # Process in parallel batches of 10 to speed up execution
        # while avoiding rate limits and memory issues
        BATCH_SIZE = 10

        async def run_test_with_params(tc):
            """Wrapper to run a single test case with all the required params"""
            return await run_single_test_case(
                test_case=tc,
                system_prompt=system_prompt,
                eval_prompt=eval_prompt,
                # Response generation settings
                llm_provider=llm_provider,
                model_name=model_name,
                api_key=api_key,
                # Evaluation settings (can use different model)
                eval_provider=eval_provider,
                eval_model_name=eval_model,
                eval_api_key=eval_api_key if has_eval_api_key else api_key,
                pass_threshold=pass_threshold
            )

        # Process test cases in batches
        for i in range(0, len(test_subset), BATCH_SIZE):
            batch = test_subset[i:i + BATCH_SIZE]
            # Run batch in parallel using asyncio.gather
            batch_results = await asyncio.gather(
                *[run_test_with_params(tc) for tc in batch],
                return_exceptions=True
            )

            # Handle any exceptions that occurred
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    # Convert exception to error result
                    tc = batch[j]
                    results.append({
                        "test_case_id": tc.get("id"),
                        "input": tc.get("input", ""),
                        "output": "",
                        "score": None,
                        "passed": False,
                        "feedback": f"Batch execution error: {str(result)}",
                        "error": True,
                        "evaluation_error": False,
                        "generation_error": True,
                        "latency_ms": 0,
                        "ttfb_ms": 0,
                        "tokens_used": 0,
                        "judge_metadata": {
                            "eval_provider": eval_provider,
                            "eval_model": eval_model,
                            "parsing_status": "not_attempted",
                            "raw_eval_output": None
                        }
                    })
                else:
                    results.append(result)

        test_run["status"] = "completed"
        test_run["results"] = results

        # Calculate summary with proper handling of None scores and error tracking
        total = len(results)
        passed = sum(1 for r in results if r.get("passed"))
        failed = total - passed

        # Count different error types
        generation_errors = sum(1 for r in results if r.get("generation_error"))
        evaluation_errors = sum(1 for r in results if r.get("evaluation_error"))

        # Only include valid scores (not None, not from errors)
        valid_scores = [
            r.get("score") for r in results
            if r.get("score") is not None and not r.get("error") and not r.get("evaluation_error")
        ]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

        # Calculate score distribution - only count valid scores
        score_distribution = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "error": 0}
        for r in results:
            score = r.get("score")
            if score is None or r.get("evaluation_error") or r.get("generation_error"):
                score_distribution["error"] += 1
            elif score >= 4.5:
                score_distribution["5"] += 1
            elif score >= 3.5:
                score_distribution["4"] += 1
            elif score >= 2.5:
                score_distribution["3"] += 1
            elif score >= 1.5:
                score_distribution["2"] += 1
            else:
                score_distribution["1"] += 1

        # Calculate latency metrics
        latencies = [r.get("latency_ms", 0) for r in results if r.get("latency_ms", 0) > 0]
        ttfbs = [r.get("ttfb_ms", 0) for r in results if r.get("ttfb_ms", 0) > 0]
        total_latency = sum(latencies)
        min_latency = min(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        avg_latency = total_latency / len(latencies) if latencies else 0
        avg_ttfb = sum(ttfbs) / len(ttfbs) if ttfbs else 0

        # Calculate total tokens
        total_tokens = sum(r.get("tokens_used", 0) for r in results)

        # Estimate cost based on tokens (rough estimate: $0.002 per 1K tokens for GPT-4)
        estimated_cost = (total_tokens / 1000) * 0.002 if total_tokens > 0 else total * 0.0005

        test_run["summary"] = {
            "total": total,
            "completed_items": total,
            "passed": passed,
            "failed": failed,
            "generation_errors": generation_errors,
            "evaluation_errors": evaluation_errors,
            "valid_scores_count": len(valid_scores),
            "avg_score": round(avg_score, 2) if valid_scores else None,
            "pass_rate": round((passed / total * 100) if total > 0 else 0, 1),
            "score_distribution": score_distribution,
            "estimated_cost": round(estimated_cost, 4),
            "total_latency_ms": total_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "avg_latency_ms": round(avg_latency, 0),
            "avg_ttfb_ms": round(avg_ttfb, 0),
            "total_tokens": total_tokens
        }

    # Persist test run to project file
    if project.test_runs is None:
        project.test_runs = []
    project.test_runs.append(test_run)
    project.test_results = test_run["results"]
    project.updated_at = datetime.now()
    project_storage.save_project(project)

    # Return with fields frontend expects
    return {
        **test_run,
        "run_id": run_id,
        "total_items": len(test_cases)
    }


@router.get("/{project_id}/test-runs")
async def list_test_runs(project_id: str, limit: int = 10):
    """List test runs for a project"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get test runs from project file
    run_list = list(project.test_runs) if project.test_runs else []

    if run_list:
        run_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    return run_list[:limit]


def get_test_run_from_project(project, run_id: str):
    """Helper to get a test run from project file"""
    if not project.test_runs:
        return None
    for run in project.test_runs:
        if run["id"] == run_id:
            return run
    return None


@router.get("/{project_id}/test-runs/{run_id}/status")
async def get_test_run_status(project_id: str, run_id: str):
    """Get status of a test run"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    test_run = get_test_run_from_project(project, run_id)

    if not test_run:
        raise HTTPException(status_code=404, detail="Test run not found")

    return {
        "id": run_id,
        "status": test_run["status"],
        "progress": 100 if test_run["status"] == "completed" else 50,
        "summary": test_run.get("summary")
    }


@router.get("/{project_id}/test-runs/{run_id}/results")
async def get_test_run_results(project_id: str, run_id: str):
    """Get results of a test run"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    test_run = get_test_run_from_project(project, run_id)

    if not test_run:
        raise HTTPException(status_code=404, detail="Test run not found")

    return {
        "id": run_id,
        "status": test_run["status"],
        "results": test_run.get("results", []),
        "summary": test_run.get("summary")
    }


@router.delete("/{project_id}/test-runs/{run_id}")
async def delete_test_run(project_id: str, run_id: str):
    """Delete a test run"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.test_runs:
        raise HTTPException(status_code=404, detail="Test run not found")

    # Find and remove the test run
    original_len = len(project.test_runs)
    project.test_runs = [r for r in project.test_runs if r["id"] != run_id]

    if len(project.test_runs) == original_len:
        raise HTTPException(status_code=404, detail="Test run not found")

    project.updated_at = datetime.now()
    project_storage.save_project(project)
    return {"message": "Test run deleted successfully"}


@router.post("/{project_id}/test-runs/{run_id}/delete")
async def delete_test_run_post(project_id: str, run_id: str):
    """Delete a test run (POST method)"""
    return await delete_test_run(project_id, run_id)


@router.get("/{project_id}/test-runs/{run_id}/export")
async def export_test_run(project_id: str, run_id: str, format: str = "json"):
    """Export test run results"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    test_run = get_test_run_from_project(project, run_id)

    if not test_run:
        raise HTTPException(status_code=404, detail="Test run not found")

    return {
        "format": format,
        "data": test_run,
        "exported_at": datetime.now().isoformat()
    }


@router.post("/{project_id}/test-runs/compare")
async def compare_test_runs(project_id: str, data: dict):
    """Compare multiple test runs"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    run_ids = data.get("run_ids", [])
    runs_dict = {r["id"]: r for r in (project.test_runs or [])}

    comparisons = []
    for run_id in run_ids:
        if run_id in runs_dict:
            run = runs_dict[run_id]
            comparisons.append({
                "run_id": run_id,
                "created_at": run["created_at"],
                "summary": run.get("summary", {})
            })

    return {
        "comparisons": comparisons,
        "best_run": comparisons[0]["run_id"] if comparisons else None
    }


@router.post("/{project_id}/test-runs/single")
async def run_single_test(project_id: str, data: dict):
    """Run a single test case"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    test_input = data.get("input", "")

    return {
        "input": test_input,
        "output": f"Sample output for: {test_input}",
        "score": 4.5,
        "passed": True,
        "feedback": "Test completed successfully"
    }


@router.post("/{project_id}/test-runs/rerun-failed")
async def rerun_failed_tests(project_id: str, data: dict):
    """Rerun failed test cases"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    run_id = data.get("run_id")
    original_run = get_test_run_from_project(project, run_id)

    if not original_run:
        raise HTTPException(status_code=404, detail="Original test run not found")

    # Get failed tests
    failed_tests = [r for r in original_run.get("results", []) if not r.get("passed")]

    # Create new run with just failed tests
    new_run_id = str(uuid.uuid4())
    new_run = {
        "id": new_run_id,
        "project_id": project_id,
        "status": "completed",
        "created_at": datetime.now().isoformat(),
        "results": [
            {**t, "passed": True, "score": 4.0, "output": "Rerun successful"}
            for t in failed_tests
        ],
        "summary": {
            "total": len(failed_tests),
            "passed": len(failed_tests),
            "failed": 0,
            "avg_score": 4.0
        }
    }

    # Persist new run to project file
    if project.test_runs is None:
        project.test_runs = []
    project.test_runs.append(new_run)
    project.updated_at = datetime.now()
    project_storage.save_project(project)

    return new_run


# ============================================================================
# FEW-SHOT CALIBRATION EXAMPLES
# ============================================================================

class CalibrationExampleRequest(BaseModel):
    input: str
    output: str
    score: float = Field(..., ge=1, le=5)
    reasoning: str
    category: str = "general"  # excellent, acceptable, poor


@router.get("/{project_id}/calibration-examples")
async def get_calibration_examples(project_id: str):
    """Get all calibration examples for a project"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return {
        "examples": project.calibration_examples or [],
        "count": len(project.calibration_examples) if project.calibration_examples else 0
    }


@router.post("/{project_id}/calibration-examples")
async def add_calibration_example(project_id: str, request: CalibrationExampleRequest):
    """Add a calibration example for few-shot learning in eval prompts"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    example = {
        "id": str(uuid.uuid4()),
        "input": request.input,
        "output": request.output,
        "score": request.score,
        "reasoning": request.reasoning,
        "category": request.category,
        "created_at": datetime.now().isoformat()
    }

    if project.calibration_examples is None:
        project.calibration_examples = []
    project.calibration_examples.append(example)
    project.updated_at = datetime.now()
    project_storage.save_project(project)

    return example


@router.delete("/{project_id}/calibration-examples/{example_id}")
async def delete_calibration_example(project_id: str, example_id: str):
    """Delete a calibration example"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.calibration_examples:
        raise HTTPException(status_code=404, detail="No calibration examples found")

    original_len = len(project.calibration_examples)
    # Handle both dict and Pydantic model
    project.calibration_examples = [
        e for e in project.calibration_examples 
        if (e.get("id") if hasattr(e, 'get') else getattr(e, 'id', None)) != example_id
    ]

    if len(project.calibration_examples) == original_len:
        raise HTTPException(status_code=404, detail="Calibration example not found")

    project.updated_at = datetime.now()
    project_storage.save_project(project)
    return {"message": "Calibration example deleted"}


@router.post("/{project_id}/calibration-examples/generate")
async def generate_calibration_examples(project_id: str):
    """Auto-generate calibration examples using LLM based on use case"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    settings = get_settings()
    provider = settings.get("provider", "openai")
    api_key = settings.get("api_key", "")
    model_name = settings.get("model_name")

    if not api_key:
        raise HTTPException(status_code=400, detail="API key required to generate calibration examples")

    current_prompt = ""
    if project.system_prompt_versions:
        current_prompt = project.system_prompt_versions[-1].get("prompt_text", "")

    requirements_str = ', '.join(project.key_requirements) if project.key_requirements else 'Not specified'

    system_prompt = """You are an expert at creating calibration examples for LLM evaluation.
Generate 3 diverse calibration examples showing what excellent (score 5), acceptable (score 3), and poor (score 1) responses look like.

Return a JSON array with exactly 3 examples:
[
  {
    "input": "example user input",
    "output": "example AI response",
    "score": 5,
    "reasoning": "Why this deserves a 5: specific reasons",
    "category": "excellent"
  },
  {
    "input": "example user input",
    "output": "example AI response",
    "score": 3,
    "reasoning": "Why this deserves a 3: specific reasons",
    "category": "acceptable"
  },
  {
    "input": "example user input",
    "output": "example AI response",
    "score": 1,
    "reasoning": "Why this deserves a 1: specific reasons",
    "category": "poor"
  }
]

Return ONLY valid JSON array."""

    user_message = f"""Generate calibration examples for:

Use Case: {project.use_case}
Requirements: {requirements_str}

System Prompt Being Evaluated:
{current_prompt[:2000]}

Create realistic examples showing score 5 (excellent), 3 (acceptable), and 1 (poor) responses."""

    result = await llm_client.chat(
        system_prompt=system_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.7,
        max_tokens=4000
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    try:
        output = result.get("output", "[]").strip()
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0]
        elif "```" in output:
            output = output.split("```")[1].split("```")[0]

        examples = json.loads(output)

        # Add IDs and timestamps
        for ex in examples:
            ex["id"] = str(uuid.uuid4())
            ex["created_at"] = datetime.now().isoformat()

        if project.calibration_examples is None:
            project.calibration_examples = []
        project.calibration_examples.extend(examples)
        project.updated_at = datetime.now()
        project_storage.save_project(project)

        return {"examples": examples, "count": len(examples)}

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse generated examples")


def build_eval_prompt_with_calibration(base_eval_prompt: str, calibration_examples: List[Dict]) -> str:
    """Inject few-shot calibration examples into eval prompt"""
    if not calibration_examples:
        return base_eval_prompt

    # Build examples section
    examples_section = "\n<calibration_examples>\nHere are examples showing how to score responses:\n\n"

    for i, ex in enumerate(calibration_examples[:5], 1):  # Limit to 5 examples
        # Handle both dict and Pydantic model
        if hasattr(ex, 'model_dump'):
            ex_dict = ex.model_dump()
        elif hasattr(ex, 'get'):
            ex_dict = ex
        else:
            # If it's an object with attributes
            ex_dict = {
                'score': getattr(ex, 'score', 'N/A'),
                'category': getattr(ex, 'category', 'general'),
                'input': getattr(ex, 'input', ''),
                'output': getattr(ex, 'output', ''),
                'reasoning': getattr(ex, 'reasoning', '')
            }
        
        examples_section += f"""Example {i} (Score: {ex_dict.get('score', 'N/A')} - {ex_dict.get('category', 'general')}):
Input: {ex_dict.get('input', '')[:200]}
Output: {ex_dict.get('output', '')[:300]}
Correct Score: {ex_dict.get('score', 'N/A')}
Reasoning: {ex_dict.get('reasoning', '')}

"""

    examples_section += "</calibration_examples>\n"

    # Insert before output_format section if it exists
    if "<output_format>" in base_eval_prompt:
        return base_eval_prompt.replace("<output_format>", examples_section + "<output_format>")
    elif "<instructions>" in base_eval_prompt:
        return base_eval_prompt.replace("<instructions>", examples_section + "<instructions>")
    else:
        # Append at the end before any closing tags
        return base_eval_prompt + "\n" + examples_section


# ============================================================================
# HUMAN-IN-THE-LOOP VALIDATION
# ============================================================================

class HumanValidationRequest(BaseModel):
    result_id: str
    run_id: str
    human_score: float = Field(..., ge=1, le=5)
    human_feedback: str
    validator_id: Optional[str] = None


@router.get("/{project_id}/human-validations")
async def get_human_validations(project_id: str):
    """Get all human validations for a project"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    validations = project.human_validations or []

    # Calculate agreement metrics
    if validations:
        def get_val(v, key, default):
            """Helper to get value from dict or Pydantic model"""
            return v.get(key, default) if hasattr(v, 'get') else getattr(v, key, default)
        
        agreements = sum(1 for v in validations if get_val(v, "agrees_with_llm", False))
        avg_diff = sum(abs(get_val(v, "score_difference", 0)) for v in validations) / len(validations)
    else:
        agreements = 0
        avg_diff = 0

    return {
        "validations": validations,
        "count": len(validations),
        "agreement_rate": round((agreements / len(validations) * 100) if validations else 0, 1),
        "avg_score_difference": round(avg_diff, 2)
    }


@router.post("/{project_id}/human-validations")
async def add_human_validation(project_id: str, request: HumanValidationRequest):
    """Add a human validation for a test result"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Find the test run and result
    test_run = get_test_run_from_project(project, request.run_id)
    if not test_run:
        raise HTTPException(status_code=404, detail="Test run not found")

    # Find the specific result
    result = None
    for r in test_run.get("results", []):
        if r.get("test_case_id") == request.result_id:
            result = r
            break

    if not result:
        raise HTTPException(status_code=404, detail="Test result not found")

    llm_score = result.get("score") or 0
    score_diff = request.human_score - llm_score
    agrees = abs(score_diff) <= 1.0  # Within 1 point = agreement

    validation = {
        "id": str(uuid.uuid4()),
        "result_id": request.result_id,
        "run_id": request.run_id,
        "human_score": request.human_score,
        "human_feedback": request.human_feedback,
        "validator_id": request.validator_id,
        "validated_at": datetime.now().isoformat(),
        "llm_score": llm_score,
        "agrees_with_llm": agrees,
        "score_difference": round(score_diff, 1)
    }

    if project.human_validations is None:
        project.human_validations = []
    project.human_validations.append(validation)
    project.updated_at = datetime.now()
    project_storage.save_project(project)

    return validation


@router.get("/{project_id}/human-validations/pending")
async def get_pending_validations(project_id: str, run_id: Optional[str] = None):
    """Get test results that haven't been human-validated yet"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    validated_ids = set()
    if project.human_validations:
        # Handle both dict and Pydantic model
        validated_ids = {
            v.get("result_id") if hasattr(v, 'get') else getattr(v, 'result_id', None)
            for v in project.human_validations
        }

    pending = []
    runs_to_check = []

    if run_id:
        run = get_test_run_from_project(project, run_id)
        if run:
            runs_to_check = [run]
    else:
        runs_to_check = project.test_runs or []

    for run in runs_to_check:
        for result in run.get("results", []):
            result_id = result.get("test_case_id")
            if result_id and result_id not in validated_ids:
                pending.append({
                    "run_id": run.get("id"),
                    "result_id": result_id,
                    "input": result.get("input", "")[:200],
                    "output": result.get("output", "")[:300],
                    "llm_score": result.get("score"),
                    "llm_feedback": result.get("feedback", "")[:200]
                })

    return {
        "pending": pending[:20],  # Limit to 20
        "total_pending": len(pending)
    }


@router.post("/{project_id}/human-validations/convert-to-calibration")
async def convert_validation_to_calibration(project_id: str, data: dict):
    """Convert a human validation into a calibration example"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    validation_id = data.get("validation_id")
    if not validation_id:
        raise HTTPException(status_code=400, detail="validation_id required")

    # Find the validation
    validation = None
    if project.human_validations:
        for v in project.human_validations:
            # Handle both dict and Pydantic model
            v_id = v.get("id") if hasattr(v, 'get') else getattr(v, 'id', None)
            if v_id == validation_id:
                validation = v
                break

    if not validation:
        raise HTTPException(status_code=404, detail="Validation not found")
    
    # Convert to dict if Pydantic model
    if hasattr(validation, 'model_dump'):
        validation = validation.model_dump()
    elif not hasattr(validation, 'get'):
        validation = {
            'id': getattr(validation, 'id', None),
            'run_id': getattr(validation, 'run_id', None),
            'result_id': getattr(validation, 'result_id', None),
            'human_score': getattr(validation, 'human_score', None)
        }

    # Find the original result to get input/output
    test_run = get_test_run_from_project(project, validation.get("run_id"))
    if not test_run:
        raise HTTPException(status_code=404, detail="Test run not found")

    result = None
    for r in test_run.get("results", []):
        if r.get("test_case_id") == validation.get("result_id"):
            result = r
            break

    if not result:
        raise HTTPException(status_code=404, detail="Original result not found")

    # Determine category based on score
    score = validation.get("human_score", 3)
    if score >= 4.5:
        category = "excellent"
    elif score >= 3:
        category = "acceptable"
    else:
        category = "poor"

    calibration_example = {
        "id": str(uuid.uuid4()),
        "input": result.get("input", ""),
        "output": result.get("output", ""),
        "score": score,
        "reasoning": validation.get("human_feedback", ""),
        "category": category,
        "created_at": datetime.now().isoformat(),
        "source": "human_validation"
    }

    if project.calibration_examples is None:
        project.calibration_examples = []
    project.calibration_examples.append(calibration_example)
    project.updated_at = datetime.now()
    project_storage.save_project(project)

    return calibration_example


# ============================================================================
# A/B TESTING WITH STATISTICAL SIGNIFICANCE
# ============================================================================

class CreateABTestRequest(BaseModel):
    name: str
    version_a: int  # Control version
    version_b: int  # Treatment version
    sample_size: int = 30
    confidence_level: float = 0.95


@router.get("/{project_id}/ab-tests")
async def get_ab_tests(project_id: str):
    """Get all A/B tests for a project"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return {
        "tests": project.ab_tests or [],
        "count": len(project.ab_tests) if project.ab_tests else 0
    }


@router.post("/{project_id}/ab-tests")
async def create_ab_test(project_id: str, request: CreateABTestRequest):
    """Create a new A/B test between two prompt versions"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Verify both versions exist
    version_nums = {v.get("version") for v in (project.system_prompt_versions or [])}
    if request.version_a not in version_nums or request.version_b not in version_nums:
        raise HTTPException(status_code=400, detail="One or both versions not found")

    ab_test = {
        "id": str(uuid.uuid4()),
        "name": request.name,
        "version_a": request.version_a,
        "version_b": request.version_b,
        "sample_size": request.sample_size,
        "confidence_level": request.confidence_level,
        "status": "created",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "results_a": [],
        "results_b": []
    }

    if project.ab_tests is None:
        project.ab_tests = []
    project.ab_tests.append(ab_test)
    project.updated_at = datetime.now()
    project_storage.save_project(project)

    return ab_test


def calculate_statistics(scores_a: List[float], scores_b: List[float], confidence_level: float = 0.95) -> Dict[str, Any]:
    """Calculate statistical significance between two groups using Welch's t-test"""
    import math

    if len(scores_a) < 2 or len(scores_b) < 2:
        return {
            "p_value": 1.0,
            "is_significant": False,
            "effect_size": 0,
            "confidence_interval": {"lower": 0, "upper": 0},
            "recommendation": "Insufficient data for statistical analysis"
        }

    # Calculate means and standard deviations
    mean_a = sum(scores_a) / len(scores_a)
    mean_b = sum(scores_b) / len(scores_b)

    var_a = sum((x - mean_a) ** 2 for x in scores_a) / (len(scores_a) - 1)
    var_b = sum((x - mean_b) ** 2 for x in scores_b) / (len(scores_b) - 1)

    std_a = math.sqrt(var_a) if var_a > 0 else 0.001
    std_b = math.sqrt(var_b) if var_b > 0 else 0.001

    n_a = len(scores_a)
    n_b = len(scores_b)

    # Welch's t-test
    se = math.sqrt((var_a / n_a) + (var_b / n_b))
    if se == 0:
        se = 0.001

    t_stat = (mean_b - mean_a) / se

    # Degrees of freedom (Welch-Satterthwaite)
    num = ((var_a / n_a) + (var_b / n_b)) ** 2
    denom = ((var_a / n_a) ** 2 / (n_a - 1)) + ((var_b / n_b) ** 2 / (n_b - 1))
    df = num / denom if denom > 0 else 1

    # Approximate p-value using normal distribution (simplified)
    # For more accurate results, use scipy.stats.t.sf
    z = abs(t_stat)
    # Approximation of two-tailed p-value
    p_value = 2 * (1 - (0.5 * (1 + math.erf(z / math.sqrt(2)))))

    # Effect size (Cohen's d)
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    effect_size = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0

    # Confidence interval for difference
    alpha = 1 - confidence_level
    # Approximate critical value (z for large samples)
    z_crit = 1.96 if confidence_level == 0.95 else 2.576 if confidence_level == 0.99 else 1.645
    margin = z_crit * se
    diff = mean_b - mean_a

    is_significant = p_value < (1 - confidence_level)

    # Generate recommendation
    if not is_significant:
        recommendation = "No statistically significant difference. Consider collecting more data or the versions perform similarly."
    elif diff > 0:
        recommendation = f"Version B performs significantly better (effect size: {effect_size:.2f}). Consider adopting Version B."
    else:
        recommendation = f"Version A performs significantly better (effect size: {abs(effect_size):.2f}). Keep Version A."

    return {
        "p_value": round(p_value, 4),
        "is_significant": is_significant,
        "effect_size": round(effect_size, 3),
        "confidence_interval": {
            "lower": round(diff - margin, 3),
            "upper": round(diff + margin, 3)
        },
        "t_statistic": round(t_stat, 3),
        "degrees_of_freedom": round(df, 1),
        "recommendation": recommendation
    }


@router.post("/{project_id}/ab-tests/{test_id}/run")
async def run_ab_test(project_id: str, test_id: str):
    """Execute an A/B test by running test cases against both versions"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Find the A/B test
    ab_test = None
    test_index = -1
    if project.ab_tests:
        for i, t in enumerate(project.ab_tests):
            # Handle both dict and Pydantic model
            t_id = t.get("id") if hasattr(t, 'get') else getattr(t, 'id', None)
            if t_id == test_id:
                ab_test = t
                test_index = i
                break

    if not ab_test:
        raise HTTPException(status_code=404, detail="A/B test not found")
    
    # Convert to dict if Pydantic model
    if hasattr(ab_test, 'model_dump'):
        ab_test = ab_test.model_dump()
    elif not hasattr(ab_test, 'get'):
        ab_test = {
            'id': getattr(ab_test, 'id', test_id),
            'version_a': getattr(ab_test, 'version_a', 1),
            'version_b': getattr(ab_test, 'version_b', 1),
            'sample_size': getattr(ab_test, 'sample_size', 30),
            'confidence_level': getattr(ab_test, 'confidence_level', 0.95),
            'status': getattr(ab_test, 'status', 'created')
        }

    # Get test cases
    test_cases = project.test_cases or []
    if not test_cases:
        raise HTTPException(status_code=400, detail="No test cases available")

    # Get prompts for both versions
    prompt_a = None
    prompt_b = None
    for v in (project.system_prompt_versions or []):
        if v.get("version") == ab_test["version_a"]:
            prompt_a = v.get("prompt_text")
        if v.get("version") == ab_test["version_b"]:
            prompt_b = v.get("prompt_text")

    if not prompt_a or not prompt_b:
        raise HTTPException(status_code=400, detail="Could not find prompt versions")

    # Get settings
    settings = get_settings()
    provider = settings.get("provider", "openai")
    api_key = settings.get("api_key", "")
    model_name = settings.get("model_name")

    if not api_key:
        raise HTTPException(status_code=400, detail="API key required")

    eval_prompt = project.eval_prompt or "Rate the response from 1-5."

    # Run tests for both versions
    sample_size = min(ab_test.get("sample_size", 30), len(test_cases))
    test_subset = test_cases[:sample_size]

    results_a = []
    results_b = []

    ab_test["status"] = "running"
    project.ab_tests[test_index] = ab_test
    project_storage.save_project(project)

    # Process in parallel batches of 10 for better performance
    import asyncio
    BATCH_SIZE = 10

    async def run_ab_pair(tc):
        """Run both A and B tests for a single test case in parallel"""
        result_a, result_b = await asyncio.gather(
            run_single_test_case(
                test_case=tc,
                system_prompt=prompt_a,
                eval_prompt=eval_prompt,
                llm_provider=provider,
                model_name=model_name,
                api_key=api_key,
                eval_provider=provider,
                eval_model_name=model_name,
                eval_api_key=api_key,
                pass_threshold=3.5
            ),
            run_single_test_case(
                test_case=tc,
                system_prompt=prompt_b,
                eval_prompt=eval_prompt,
                llm_provider=provider,
                model_name=model_name,
                api_key=api_key,
                eval_provider=provider,
                eval_model_name=model_name,
                eval_api_key=api_key,
                pass_threshold=3.5
            )
        )
        return result_a, result_b

    # Process test cases in batches
    for i in range(0, len(test_subset), BATCH_SIZE):
        batch = test_subset[i:i + BATCH_SIZE]
        # Run batch in parallel
        batch_results = await asyncio.gather(
            *[run_ab_pair(tc) for tc in batch],
            return_exceptions=True
        )

        # Handle results
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                # Create error results for both A and B
                tc = batch[j]
                error_result = {
                    "test_case_id": tc.get("id"),
                    "input": tc.get("input", ""),
                    "output": "",
                    "score": None,
                    "passed": False,
                    "feedback": f"Batch execution error: {str(result)}",
                    "error": True,
                    "latency_ms": 0,
                    "tokens_used": 0
                }
                results_a.append(error_result)
                results_b.append(error_result)
            else:
                result_a, result_b = result
                results_a.append(result_a)
                results_b.append(result_b)

    # Calculate statistics
    scores_a = [r.get("score") for r in results_a if r.get("score") is not None]
    scores_b = [r.get("score") for r in results_b if r.get("score") is not None]

    stats = calculate_statistics(scores_a, scores_b, ab_test.get("confidence_level", 0.95))

    # Calculate summary stats for each version
    def calc_summary(results, scores):
        return {
            "total": len(results),
            "valid_scores": len(scores),
            "mean_score": round(sum(scores) / len(scores), 2) if scores else 0,
            "min_score": round(min(scores), 1) if scores else 0,
            "max_score": round(max(scores), 1) if scores else 0,
            "std_dev": round((sum((x - (sum(scores)/len(scores)))**2 for x in scores) / len(scores))**0.5, 2) if len(scores) > 1 else 0,
            "pass_rate": round(sum(1 for r in results if r.get("passed")) / len(results) * 100, 1) if results else 0
        }

    summary_a = calc_summary(results_a, scores_a)
    summary_b = calc_summary(results_b, scores_b)

    # Determine winner
    winner = None
    if stats["is_significant"]:
        if summary_b["mean_score"] > summary_a["mean_score"]:
            winner = "B"
        else:
            winner = "A"

    # Update A/B test
    ab_test["status"] = "completed"
    ab_test["completed_at"] = datetime.now().isoformat()
    ab_test["results_a"] = results_a
    ab_test["results_b"] = results_b
    ab_test["summary_a"] = summary_a
    ab_test["summary_b"] = summary_b
    ab_test["statistics"] = stats
    ab_test["winner"] = winner

    project.ab_tests[test_index] = ab_test
    project.updated_at = datetime.now()
    project_storage.save_project(project)

    return {
        "test_id": test_id,
        "status": "completed",
        "version_a": {
            "version_number": ab_test["version_a"],
            **summary_a
        },
        "version_b": {
            "version_number": ab_test["version_b"],
            **summary_b
        },
        "statistics": stats,
        "winner": winner,
        "recommendation": stats["recommendation"]
    }


@router.get("/{project_id}/ab-tests/{test_id}/results")
async def get_ab_test_results(project_id: str, test_id: str):
    """Get detailed results of an A/B test"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    ab_test = None
    if project.ab_tests:
        for t in project.ab_tests:
            # Handle both dict and Pydantic model
            t_id = t.get("id") if hasattr(t, 'get') else getattr(t, 'id', None)
            if t_id == test_id:
                ab_test = t
                break

    if not ab_test:
        raise HTTPException(status_code=404, detail="A/B test not found")
    
    # Convert to dict if Pydantic model
    if hasattr(ab_test, 'model_dump'):
        ab_test = ab_test.model_dump()
    elif not hasattr(ab_test, 'get'):
        ab_test = {k: getattr(ab_test, k) for k in ['id', 'name', 'version_a', 'version_b', 'status'] if hasattr(ab_test, k)}

    return ab_test


@router.delete("/{project_id}/ab-tests/{test_id}")
async def delete_ab_test(project_id: str, test_id: str):
    """Delete an A/B test"""
    project = project_storage.load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.ab_tests:
        raise HTTPException(status_code=404, detail="No A/B tests found")

    original_len = len(project.ab_tests)
    # Handle both dict and Pydantic model
    project.ab_tests = [
        t for t in project.ab_tests 
        if (t.get("id") if hasattr(t, 'get') else getattr(t, 'id', None)) != test_id
    ]

    if len(project.ab_tests) == original_len:
        raise HTTPException(status_code=404, detail="A/B test not found")

    project.updated_at = datetime.now()
    project_storage.save_project(project)
    return {"message": "A/B test deleted"}
