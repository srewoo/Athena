"""
Agentic Rewrite Module for Athena
Implements multi-step, self-correcting prompt optimization with thinking model support.

Workflow:
1. Deep Analysis (thinking model) - Understand prompt structure, intent, and gaps
2. Plan Improvements - Create specific improvement plan
3. Execute Rewrite - Apply improvements while preserving DNA
4. Validate - Check for regressions and DNA preservation
5. Iterate - Fix issues if found (max 2 iterations)
"""
import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from prompt_analyzer import analyze_prompt, analysis_to_dict, PromptAnalysis, PromptDNA

logger = logging.getLogger(__name__)


class RewriteStep(Enum):
    """Steps in the agentic rewrite workflow"""
    ANALYSIS = "analysis"
    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    ITERATION = "iteration"


@dataclass
class ImprovementPlan:
    """Plan for improving a prompt"""
    preserve: List[str]  # Elements to keep unchanged
    improve: List[Dict[str, str]]  # Changes to make: {area, current, target}
    add: List[str]  # Elements to add
    remove: List[str]  # Elements to remove (rarely used)
    priority_order: List[str]  # Order of importance
    risk_areas: List[str]  # Areas where changes could cause regression


@dataclass
class ValidationResult:
    """Result of validating a rewritten prompt"""
    is_valid: bool
    dna_preserved: bool
    issues: List[str]
    warnings: List[str]
    quality_delta: float  # Change in quality score
    regression_detected: bool


@dataclass
class AgenticRewriteResult:
    """Complete result from agentic rewrite"""
    original_prompt: str
    final_prompt: str
    original_score: float
    final_score: float
    iterations: int
    steps_taken: List[Dict[str, Any]]
    improvement_plan: Optional[Dict[str, Any]]
    validation: Optional[Dict[str, Any]]
    no_change: bool
    reason: str


# Thinking model identifiers
THINKING_MODELS = {
    "openai": ["o1", "o1-mini", "o1-preview", "o3", "o3-mini", "gpt-5"],
    "claude": ["claude-sonnet-4-5", "claude-opus-4"],  # Latest Claude models
    "gemini": ["gemini-3", "gemini-2.0-flash-thinking"]
}


def is_thinking_model(model_name: str, provider: str) -> bool:
    """Check if a model is a thinking/reasoning model"""
    if not model_name:
        return False

    thinking_prefixes = THINKING_MODELS.get(provider, [])
    return any(model_name.startswith(prefix) for prefix in thinking_prefixes)


def get_thinking_model_for_provider(provider: str) -> Optional[str]:
    """Get recommended thinking model for a provider"""
    recommendations = {
        "openai": "o3",  # OpenAI's advanced reasoning model
        "claude": "claude-sonnet-4-5-20241022",  # Claude Sonnet 4.5
        "gemini": "gemini-3"  # Google Gemini 3
    }
    return recommendations.get(provider)


@dataclass
class ScoreWithConfidence:
    """LLM score with statistical confidence"""
    mean_score: float
    std_deviation: float
    individual_scores: List[float]
    confidence_level: str  # "high", "medium", "low"
    is_reliable: bool
    reasoning: str


async def llm_score_prompt(
    prompt: str,
    use_case: str,
    llm_client,
    provider: str,
    api_key: str,
    model_name: str,
    num_runs: int = 1
) -> float:
    """
    Use LLM to score a prompt's quality (1-10 scale).
    This provides more accurate scoring than heuristic analysis.
    
    Args:
        num_runs: Number of scoring runs (default 1 for speed, use 3 for reliable decisions)
    
    Returns:
        float: Mean score across runs
    """
    result = await llm_score_prompt_with_confidence(
        prompt=prompt,
        use_case=use_case,
        llm_client=llm_client,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        num_runs=num_runs
    )
    return result.mean_score


async def llm_score_prompt_with_confidence(
    prompt: str,
    use_case: str,
    llm_client,
    provider: str,
    api_key: str,
    model_name: str,
    num_runs: int = 3
) -> ScoreWithConfidence:
    """
    Use LLM to score a prompt's quality with statistical confidence.
    Runs multiple times to account for LLM variance.
    
    Args:
        num_runs: Number of scoring runs (default 3 for confidence, minimum 1)
    
    Returns:
        ScoreWithConfidence: Score with mean, stddev, and confidence level
    """
    import statistics
    
    system_prompt = """You are an expert prompt engineer. Score this system prompt on a scale of 1-10.

## Scoring Criteria:
- **Structure (25%)**: Clear sections, logical organization, headers/delimiters
- **Clarity (25%)**: Specific instructions, unambiguous language, explicit expectations
- **Completeness (25%)**: Role definition, output format, constraints, examples
- **Robustness (25%)**: Edge case handling, safety constraints, error handling

## Score Guidelines:
- 9-10: Production-ready, comprehensive, follows all best practices
- 7-8: Good quality, minor improvements possible
- 5-6: Functional but needs significant improvements
- 3-4: Weak, missing critical elements
- 1-2: Poor, fundamentally flawed

Return ONLY a JSON object: {"score": X.X, "reasoning": "brief explanation"}"""

    user_message = f"""Score this prompt for use case: {use_case}

PROMPT:
```
{prompt}
```

Return JSON with score (1-10) and brief reasoning."""

    scores = []
    reasoning = ""
    
    # Run scoring multiple times to get confidence interval
    actual_runs = max(1, min(5, num_runs))  # Cap at 5 runs
    
    for run in range(actual_runs):
        try:
            result = await llm_client.chat(
                system_prompt=system_prompt,
                user_message=user_message,
                provider=provider,
                api_key=api_key,
                model_name=model_name,
                temperature=0.3,  # Slightly higher for variance detection
                max_tokens=500
            )

            if not result or result.get("error"):
                error_msg = result.get("error") if result else "No response"
                logger.warning(f"LLM scoring run {run+1} failed: {error_msg}")
                continue

            json_match = re.search(r'\{[\s\S]*\}', result.get("output", ""))
            if json_match:
                data = json.loads(json_match.group())
                score = float(data.get("score", 0))
                score = min(10.0, max(1.0, score))
                scores.append(score)
                if not reasoning:
                    reasoning = data.get('reasoning', '')[:200]
                logger.debug(f"LLM scoring run {run+1}: {score}/10")
        except Exception as e:
            logger.warning(f"LLM scoring run {run+1} exception: {e}")
            continue
    
    # Calculate statistics
    if not scores:
        logger.warning("All LLM scoring runs failed, using fallback")
        return ScoreWithConfidence(
            mean_score=5.0,  # Conservative fallback (was 6.0, now more honest)
            std_deviation=2.0,  # High uncertainty
            individual_scores=[],
            confidence_level="low",
            is_reliable=False,
            reasoning="LLM scoring failed; using conservative fallback"
        )
    
    mean_score = statistics.mean(scores)
    
    if len(scores) >= 2:
        std_dev = statistics.stdev(scores)
    else:
        std_dev = 0.5  # Assume moderate variance for single run
    
    # Determine confidence level
    # Low stddev = high confidence, high stddev = low confidence
    if std_dev <= 0.3:
        confidence_level = "high"
        is_reliable = True
    elif std_dev <= 0.7:
        confidence_level = "medium"
        is_reliable = True
    else:
        confidence_level = "low"
        is_reliable = False
    
    logger.info(f"LLM scored prompt: {mean_score:.1f} ± {std_dev:.2f} ({confidence_level} confidence)")
    
    return ScoreWithConfidence(
        mean_score=round(mean_score, 1),
        std_deviation=round(std_dev, 2),
        individual_scores=scores,
        confidence_level=confidence_level,
        is_reliable=is_reliable,
        reasoning=reasoning
    )


async def deep_analysis(
    prompt: str,
    use_case: str,
    requirements: str,
    llm_client,
    provider: str,
    api_key: str,
    model_name: str,
    user_analysis_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Step 1: Deep analysis using thinking model.
    Analyzes the prompt's structure, intent, strengths, and weaknesses.
    Incorporates user-provided analysis context from Re-Analyze if available.
    """
    # First, get programmatic analysis
    programmatic_analysis = analyze_prompt(prompt)
    analysis_dict = analysis_to_dict(programmatic_analysis)

    # Build user feedback section if analysis context provided
    user_feedback_section = ""
    if user_analysis_context:
        suggestions = user_analysis_context.get("suggestions", [])
        missing_reqs = user_analysis_context.get("missing_requirements", [])
        best_practices = user_analysis_context.get("best_practices_gaps", [])

        if suggestions or missing_reqs or best_practices:
            user_feedback_section = "\n\n## USER'S PRIOR ANALYSIS FEEDBACK (incorporate these insights):\n"
            if suggestions:
                user_feedback_section += "\n**Suggestions from analysis:**\n"
                for sug in suggestions[:5]:
                    if isinstance(sug, dict):
                        user_feedback_section += f"- [{sug.get('priority', 'Medium')}] {sug.get('suggestion', sug.get('text', str(sug)))}\n"
                    else:
                        user_feedback_section += f"- {sug}\n"
            if missing_reqs:
                user_feedback_section += "\n**Missing requirements identified:**\n"
                for req in missing_reqs[:5]:
                    user_feedback_section += f"- {req}\n"
            if best_practices:
                user_feedback_section += "\n**Best practices gaps:**\n"
                for bp in best_practices[:5]:
                    user_feedback_section += f"- {bp}\n"

    # Use thinking model for deeper analysis
    system_prompt = """You are an expert prompt engineer performing deep analysis. Think step-by-step about this prompt.

Your task is to DEEPLY ANALYZE a system prompt and identify:
1. **Core Intent**: What is this prompt trying to achieve?
2. **Critical DNA Elements**: What MUST be preserved (variables, format, scale, terminology)?
3. **Structural Strengths**: What's working well?
4. **Specific Weaknesses**: What's genuinely broken or missing?
5. **Risk Assessment**: Where could "improvements" actually make things worse?

## IMPORTANT: Quality Bar
- Score 8.5+ with good structure = NO CHANGES NEEDED
- Only flag weaknesses that genuinely affect functionality
- Cosmetic issues are NOT weaknesses

Return your analysis as JSON:
{
    "core_intent": "What the prompt is designed to do",
    "critical_dna": {
        "template_variables": ["list of {{variables}} that MUST be preserved"],
        "output_format": "The exact output format required",
        "scoring_scale": "Any scoring scale that must be preserved",
        "key_terminology": ["Domain-specific terms to preserve"]
    },
    "strengths": ["List of what's working well"],
    "genuine_weaknesses": ["Only real functional issues, not cosmetic"],
    "risk_areas": ["Where changes could cause regression"],
    "quality_assessment": {
        "current_score": 0.0,
        "needs_improvement": true/false,
        "rationale": "Why it does/doesn't need changes"
    }
}"""

    user_message = f"""Analyze this prompt deeply:

**Use Case:** {use_case}

**Requirements:** {requirements}

**Prompt to Analyze:**
```
{prompt}
```

**Programmatic Analysis Results:**
- Detected Type: {analysis_dict['prompt_type']}
- Quality Score: {analysis_dict['quality_score']}/10
- Template Variables: {analysis_dict['dna']['template_variables']}
- Scoring Scale: {analysis_dict['dna']['scoring_scale']}
- Sections Found: {analysis_dict['dna']['sections']}
- Suggested Improvement Areas: {analysis_dict['improvement_areas']}
- Identified Strengths: {analysis_dict['strengths']}
{user_feedback_section}
Provide your deep analysis. Incorporate the user's prior analysis feedback if provided. Be conservative - only flag genuine issues."""

    result = await llm_client.chat(
        system_prompt=system_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.2,  # Low temperature for analysis
        max_tokens=4000
    )

    if not result or result.get("error"):
        error_msg = result.get("error") if result else "LLM returned no response"
        logger.error(f"Deep analysis failed: {error_msg}")
        # Fall back to programmatic analysis
        return {
            "programmatic": analysis_dict,
            "deep": None,
            "error": error_msg
        }

    # Parse LLM response
    try:
        json_match = re.search(r'\{[\s\S]*\}', result["output"])
        if json_match:
            deep_analysis = json.loads(json_match.group())
        else:
            deep_analysis = {"raw_response": result["output"]}
    except json.JSONDecodeError:
        deep_analysis = {"raw_response": result["output"]}

    return {
        "programmatic": analysis_dict,
        "deep": deep_analysis,
        "combined_score": analysis_dict["quality_score"]
    }


async def plan_improvements(
    prompt: str,
    analysis: Dict[str, Any],
    use_case: str,
    requirements: str,
    llm_client,
    provider: str,
    api_key: str,
    model_name: str
) -> ImprovementPlan:
    """
    Step 2: Create a specific improvement plan.
    Only plans changes that are genuinely needed.
    """
    deep = analysis.get("deep", {})
    programmatic = analysis.get("programmatic", {})

    # Check if improvements are needed
    quality_assessment = deep.get("quality_assessment", {})
    if not quality_assessment.get("needs_improvement", True) and programmatic.get("quality_score", 0) >= 8.5:
        # No improvements needed
        return ImprovementPlan(
            preserve=["ALL - prompt is production-ready"],
            improve=[],
            add=[],
            remove=[],
            priority_order=[],
            risk_areas=[]
        )

    system_prompt = """You are a very Experienced, Efficient and Crafty Prompt Developer who assists Developers, Product Managers alike in Creating Prompts For Their Usecase.

Your goal is to create a SPECIFIC improvement plan to refine an existing prompt based on the Requirements (PRD).

## YOUR PROCESS
1.  **Analyze**: Go through the PRD (Requirements) and Existing Prompt in detail.
2.  **Identify Contradictions**: Identify major contradictions within the prompt, PRD, or between them.
3.  **Clarify**: Since you cannot ask real-time questions here, you must IDENTIFY what needs clarification and document it in the 'risk_areas' section as potential ambiguities.
4.  **Plan**: State the changes you're seeking to make and tell WHY you make them.

## GUIDELINES
-   **Source of Truth**: The PRD (Requirements) is the source of truth.
-   **Minimal Changes**: Prefer to Not Change an Existing Prompt overly much if possible. ONLY Change if there is a contradiction or a clear way to make it MORE RELIABLE.
-   **No DNA Changes**: Never plan changes to DNA elements (variables, format, scale) unless they directly contradict the PRD.

Return a JSON improvement plan:
{
    "preserve": ["List of elements that MUST NOT change"],
    "improve": [
        {"area": "What to improve", "current": "Current state", "target": "Target state", "reason": "Why this change is needed"}
    ],
    "add": ["Elements to add (only if genuinely missing from PRD)"],
    "remove": ["Elements to remove (only if contradicting PRD)"],
    "priority_order": ["Order of importance"],
    "risk_areas": ["Ambiguities or contradictions found that need user attention"]
}"""

    user_message = f"""Create an improvement plan for this prompt:

**Use Case:** {use_case}
**Requirements:** {requirements}

**Prompt:**
```
{prompt}
```

**Analysis Results:**
- Quality Score: {programmatic.get('quality_score', 'unknown')}/10
- Genuine Weaknesses: {deep.get('genuine_weaknesses', programmatic.get('improvement_areas', []))}
- Risk Areas: {deep.get('risk_areas', [])}
- Critical DNA: {deep.get('critical_dna', programmatic.get('dna', {}))}

Create a minimal, focused improvement plan. If the prompt is already good (8+), plan should be minimal."""

    result = await llm_client.chat(
        system_prompt=system_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.3,
        max_tokens=3000
    )

    if result.get("error"):
        logger.error(f"Planning failed: {result['error']}")
        # Return minimal plan
        return ImprovementPlan(
            preserve=programmatic.get("dna", {}).get("template_variables", []),
            improve=[],
            add=[],
            remove=[],
            priority_order=[],
            risk_areas=[]
        )

    try:
        json_match = re.search(r'\{[\s\S]*\}', result["output"])
        if json_match:
            plan_data = json.loads(json_match.group())
            return ImprovementPlan(
                preserve=plan_data.get("preserve", []),
                improve=plan_data.get("improve", []),
                add=plan_data.get("add", []),
                remove=plan_data.get("remove", []),
                priority_order=plan_data.get("priority_order", []),
                risk_areas=plan_data.get("risk_areas", [])
            )
    except json.JSONDecodeError:
        pass

    return ImprovementPlan(
        preserve=[],
        improve=[],
        add=[],
        remove=[],
        priority_order=[],
        risk_areas=[]
    )


async def execute_rewrite(
    prompt: str,
    plan: ImprovementPlan,
    analysis: Dict[str, Any],
    use_case: str,
    requirements: str,
    llm_client,
    provider: str,
    api_key: str,
    model_name: str,
    retry_count: int = 0,
    user_analysis_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Step 3: Execute the rewrite using a TWO-PHASE approach for guaranteed changes.

    Phase 1: DECOMPOSE - Extract components from original prompt
    Phase 2: RECONSTRUCT - Build new prompt from components + improvements

    This approach guarantees structural changes because we're rebuilding, not editing.
    """
    logger.info(f"Executing rewrite (attempt {retry_count + 1})")

    programmatic = analysis.get("programmatic", {})
    dna = programmatic.get("dna", {})

    # Extract template variables to preserve
    template_vars = dna.get("template_variables", [])
    template_vars_str = ", ".join([f"{{{{{v}}}}}" for v in template_vars]) if template_vars else "None"

    # Build improvements list
    improvements_list = []
    for imp in plan.improve:
        improvements_list.append(f"- {imp.get('area', 'General')}: {imp.get('target', 'Improve')}")
    for add in plan.add:
        improvements_list.append(f"- ADD: {add}")

    # Add user feedback
    if user_analysis_context:
        for req in user_analysis_context.get("missing_requirements", [])[:3]:
            improvements_list.append(f"- FIX: {req}")
        for sug in user_analysis_context.get("suggestions", [])[:3]:
            if isinstance(sug, dict):
                improvements_list.append(f"- {sug.get('suggestion', str(sug))}")
            else:
                improvements_list.append(f"- {sug}")

    improvements_str = "\n".join(improvements_list) if improvements_list else "- Add clear structure\n- Improve clarity"

    # PHASE 1 & 2 COMBINED: Decompose and reconstruct in one call
    system_prompt = f"""You are a prompt engineering expert. Your task is to COMPLETELY REWRITE a prompt.

## YOUR PROCESS
1. First, mentally extract the CORE INTENT from the original prompt
2. Then, BUILD A NEW PROMPT from scratch using proper structure
3. The new prompt should achieve the same goal but be structured professionally

## REQUIRED OUTPUT STRUCTURE
You MUST output a prompt with ALL of these sections:

```
# Role
You are a [specific role] specialized in [domain].

## Context
[Why this task matters, background information]

## Task
[Clear, specific description of what to do]

## Input
The input will be provided in the following format:
<input>
[Describe expected input structure]
</input>

## Requirements
1. [First requirement]
2. [Second requirement]
3. [Continue as needed]

## Constraints
- [What NOT to do]
- [Limitations to respect]

## Output Format
<output>
[Exact format specification with field names]
</output>
```

## IMPROVEMENTS TO INCORPORATE
{improvements_str}

## TEMPLATE VARIABLES TO PRESERVE
{template_vars_str}
These must appear EXACTLY as written (with double curly braces) in your output.

## CRITICAL RULES
1. DO NOT copy sentences from the original - rephrase everything
2. DO NOT skip any section - include ALL sections listed above
3. DO NOT add explanations before or after the prompt
4. DO NOT wrap in markdown code blocks
5. The output should be 2-3x longer than the input due to added structure"""

    user_message = f"""ORIGINAL PROMPT TO REWRITE:
\"\"\"
{prompt}
\"\"\"

USE CASE: {use_case}
REQUIREMENTS: {requirements}

Now output the COMPLETE rewritten prompt with all sections. Start directly with "# Role":"""

    result = await llm_client.chat(
        system_prompt=system_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.7,  # Higher temperature for creativity
        max_tokens=8000
    )

    if not result or result.get("error"):
        error_msg = result.get("error") if result else "No response"
        logger.error(f"Rewrite failed: {error_msg}")
        return prompt

    rewritten = result.get("output", "").strip()

    # Clean up output
    rewritten = _clean_rewrite_output(rewritten)

    # Validate the rewrite has actual structure
    has_structure = any(marker in rewritten for marker in ["# Role", "## Task", "## Output", "## Requirements"])

    if not has_structure and retry_count < 2:
        logger.warning("Rewrite lacks required structure, retrying...")
        return await execute_rewrite(
            prompt=prompt,
            plan=plan,
            analysis=analysis,
            use_case=use_case,
            requirements=requirements,
            llm_client=llm_client,
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            retry_count=retry_count + 1,
            user_analysis_context=user_analysis_context
        )

    # Check similarity
    similarity = _calculate_similarity(
        ' '.join(prompt.lower().split()),
        ' '.join(rewritten.lower().split())
    )
    logger.info(f"Rewrite complete. Similarity: {similarity:.1%}, Length: {len(prompt)} -> {len(rewritten)}")

    # If still too similar, try once more with even higher temperature
    if similarity > 0.85 and retry_count < 2:
        logger.warning(f"Rewrite too similar ({similarity:.1%}), retrying...")
        return await execute_rewrite(
            prompt=prompt,
            plan=plan,
            analysis=analysis,
            use_case=use_case,
            requirements=requirements,
            llm_client=llm_client,
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            retry_count=retry_count + 1,
            user_analysis_context=user_analysis_context
        )

    if not rewritten:
        logger.warning("Empty rewrite result, returning original")
        return prompt

    logger.info(f"Rewrite complete. Original length: {len(prompt)}, New length: {len(rewritten)}")
    return rewritten


def _clean_rewrite_output(text: str) -> str:
    """Clean up LLM output from rewrite - remove code blocks, explanations, etc."""
    if not text:
        return text

    text = text.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```markdown or ```)
        if lines:
            lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Remove common preambles
    preambles = [
        "Here is the rewritten prompt:",
        "Here's the rewritten prompt:",
        "Here is your rewritten prompt:",
        "Below is the rewritten prompt:",
        "The rewritten prompt:",
        "Rewritten prompt:",
    ]
    for preamble in preambles:
        if text.lower().startswith(preamble.lower()):
            text = text[len(preamble):].strip()

    # If text starts with quotes, remove them
    if text.startswith('"""') and text.endswith('"""'):
        text = text[3:-3].strip()
    elif text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()

    return text


def _calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple similarity ratio between two texts."""
    if text1 == text2:
        return 1.0
    if not text1 or not text2:
        return 0.0

    # Use word-based Jaccard similarity
    words1 = set(text1.split())
    words2 = set(text2.split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


async def validate_rewrite(
    original: str,
    rewritten: str,
    analysis: Dict[str, Any],
    llm_client,
    provider: str,
    api_key: str,
    model_name: str
) -> ValidationResult:
    """
    Step 4: Validate the rewrite for quality and DNA preservation.
    """
    programmatic = analysis.get("programmatic", {})
    original_dna = programmatic.get("dna", {})
    original_score = programmatic.get("quality_score", 0)

    # Programmatic validation first
    issues = []
    warnings = []

    # Check template variables
    for var in original_dna.get("template_variables", []):
        patterns = [f"{{{{{var}}}}}", f"{{{var}}}", f"<<{var}>>"]
        found = any(pattern in rewritten for pattern in patterns)
        if not found:
            issues.append(f"Template variable '{var}' is missing")

    # Check scoring scale
    if original_dna.get("scoring_scale"):
        scale = original_dna["scoring_scale"]
        if scale.get("type") == "numeric":
            min_val = str(scale.get("min", 0))
            max_val = str(scale.get("max", 10))
            if min_val not in rewritten or max_val not in rewritten:
                issues.append(f"Scoring scale {min_val}-{max_val} may be missing")

    # Check for common regression patterns
    if rewritten.strip().startswith("```"):
        issues.append("Rewrite incorrectly wrapped in code blocks")
    if rewritten.strip().startswith("{") and original.strip()[0] != "{":
        issues.append("Rewrite incorrectly starts with JSON")
    if len(rewritten) < len(original) * 0.5:
        warnings.append("Rewrite is significantly shorter than original")
    if len(rewritten) > len(original) * 2:
        warnings.append("Rewrite is significantly longer than original")

    # Analyze the rewritten prompt
    rewritten_analysis = analyze_prompt(rewritten)
    rewritten_score = rewritten_analysis.quality_score
    quality_delta = rewritten_score - original_score

    # Regression detection
    regression_detected = quality_delta < -0.5 or len(issues) > 0

    # DNA preservation check
    dna_preserved = len([i for i in issues if "variable" in i.lower() or "scale" in i.lower()]) == 0

    return ValidationResult(
        is_valid=len(issues) == 0,
        dna_preserved=dna_preserved,
        issues=issues,
        warnings=warnings,
        quality_delta=quality_delta,
        regression_detected=regression_detected
    )


async def iterate_fix(
    original: str,
    current: str,
    validation: ValidationResult,
    analysis: Dict[str, Any],
    llm_client,
    provider: str,
    api_key: str,
    model_name: str
) -> str:
    """
    Step 5: Fix issues found in validation.
    """
    if not validation.issues:
        return current

    programmatic = analysis.get("programmatic", {})
    dna = programmatic.get("dna", {})

    system_prompt = f"""You are fixing a prompt rewrite that has issues.

## ISSUES TO FIX
{chr(10).join(f'- {issue}' for issue in validation.issues)}

## WARNINGS
{chr(10).join(f'- {warning}' for warning in validation.warnings)}

## ORIGINAL DNA TO RESTORE
- Template Variables: {dna.get('template_variables', [])}
- Scoring Scale: {dna.get('scoring_scale', 'None')}
- Output Format: {dna.get('output_format', 'None')}

## RULES
1. Fix ONLY the listed issues
2. Restore any missing DNA elements from the original
3. Do not make other changes
4. Return the fixed prompt only, no explanations"""

    user_message = f"""Fix these issues in the rewritten prompt:

**Original Prompt:**
```
{original}
```

**Current Rewrite (with issues):**
```
{current}
```

Fix the issues and return the corrected prompt."""

    result = await llm_client.chat(
        system_prompt=system_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.2,
        max_tokens=8000
    )

    if result.get("error"):
        logger.error(f"Iteration fix failed: {result['error']}")
        return original  # Fall back to original on failure

    fixed = result.get("output", "").strip()

    # Remove markdown wrapper if present
    if fixed.startswith("```") and fixed.endswith("```"):
        lines = fixed.split("\n")
        fixed = "\n".join(lines[1:-1])

    return fixed if fixed else original


async def agentic_rewrite(
    prompt: str,
    use_case: str,
    requirements: str,
    llm_client,
    provider: str,
    api_key: str,
    model_name: str,
    thinking_model: Optional[str] = None,
    max_iterations: int = 2,
    user_analysis_context: Optional[Dict[str, Any]] = None
) -> AgenticRewriteResult:
    """
    Main agentic rewrite function.
    Orchestrates the multi-step workflow with thinking model support.

    Args:
        user_analysis_context: Optional dict with analysis feedback from Re-Analyze:
            - suggestions: List of improvement suggestions
            - missing_requirements: List of missing requirements
            - best_practices_gaps: List of best practices not followed
    """
    steps_taken = []

    # Determine which model to use for analysis (prefer thinking model)
    analysis_model = thinking_model or model_name

    # Step 1: Deep Analysis (now includes user analysis context)
    logger.info(f"Step 1: Deep analysis with {analysis_model}")
    steps_taken.append({"step": "analysis", "model": analysis_model, "status": "started"})

    analysis = await deep_analysis(
        prompt=prompt,
        use_case=use_case,
        requirements=requirements,
        llm_client=llm_client,
        provider=provider,
        api_key=api_key,
        model_name=analysis_model,
        user_analysis_context=user_analysis_context
    )

    steps_taken[-1]["status"] = "completed"

    # Use LLM to score the original prompt for accurate comparison
    logger.info("Scoring original prompt with LLM for accurate baseline")
    original_score = await llm_score_prompt(
        prompt=prompt,
        use_case=use_case,
        llm_client=llm_client,
        provider=provider,
        api_key=api_key,
        model_name=model_name
    )

    steps_taken[-1]["result"] = {
        "quality_score": original_score,
        "prompt_type": analysis.get("programmatic", {}).get("prompt_type", "unknown")
    }

    # Check if prompt is already good enough
    deep_analysis_result = analysis.get("deep", {})
    quality_assessment = deep_analysis_result.get("quality_assessment", {})

    if original_score >= 8.5 and not quality_assessment.get("needs_improvement", True):
        logger.info("Prompt is already production-ready, no changes needed")
        return AgenticRewriteResult(
            original_prompt=prompt,
            final_prompt=prompt,
            original_score=original_score,
            final_score=original_score,
            iterations=0,
            steps_taken=steps_taken,
            improvement_plan=None,
            validation=None,
            no_change=True,
            reason=f"Prompt is already production-ready (score: {original_score}/10). No changes needed."
        )

    # Step 2: Plan Improvements
    logger.info(f"Step 2: Planning improvements with {analysis_model}")
    steps_taken.append({"step": "planning", "model": analysis_model, "status": "started"})

    plan = await plan_improvements(
        prompt=prompt,
        analysis=analysis,
        use_case=use_case,
        requirements=requirements,
        llm_client=llm_client,
        provider=provider,
        api_key=api_key,
        model_name=analysis_model
    )

    steps_taken[-1]["status"] = "completed"
    steps_taken[-1]["result"] = {
        "improvements_planned": len(plan.improve),
        "additions_planned": len(plan.add)
    }

    # If no improvements planned
    if not plan.improve and not plan.add:
        logger.info("No improvements needed based on analysis")
        return AgenticRewriteResult(
            original_prompt=prompt,
            final_prompt=prompt,
            original_score=original_score,
            final_score=original_score,
            iterations=0,
            steps_taken=steps_taken,
            improvement_plan={
                "preserve": plan.preserve,
                "improve": plan.improve,
                "add": plan.add
            },
            validation=None,
            no_change=True,
            reason="Analysis found no improvements needed."
        )

    # Step 3: Execute Rewrite
    logger.info(f"Step 3: Executing rewrite with {model_name}")
    steps_taken.append({"step": "execution", "model": model_name, "status": "started"})

    current_prompt = await execute_rewrite(
        prompt=prompt,
        plan=plan,
        analysis=analysis,
        use_case=use_case,
        requirements=requirements,
        llm_client=llm_client,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        user_analysis_context=user_analysis_context
    )

    steps_taken[-1]["status"] = "completed"

    # Iteration loop
    iteration = 0
    final_validation = None

    while iteration < max_iterations:
        iteration += 1

        # Step 4: Validate
        logger.info(f"Step 4: Validating (iteration {iteration})")
        steps_taken.append({"step": "validation", "iteration": iteration, "status": "started"})

        validation = await validate_rewrite(
            original=prompt,
            rewritten=current_prompt,
            analysis=analysis,
            llm_client=llm_client,
            provider=provider,
            api_key=api_key,
            model_name=model_name
        )

        steps_taken[-1]["status"] = "completed"
        steps_taken[-1]["result"] = {
            "is_valid": validation.is_valid,
            "issues": validation.issues,
            "quality_delta": validation.quality_delta
        }

        final_validation = validation

        # If valid, we're done
        if validation.is_valid and not validation.regression_detected:
            logger.info("Validation passed")
            break

        # If regression detected, try to fix
        if validation.regression_detected:
            logger.warning(f"Regression detected: {validation.issues}")

            # Step 5: Iterate fix
            steps_taken.append({"step": "iteration", "iteration": iteration, "status": "started"})

            current_prompt = await iterate_fix(
                original=prompt,
                current=current_prompt,
                validation=validation,
                analysis=analysis,
                llm_client=llm_client,
                provider=provider,
                api_key=api_key,
                model_name=model_name
            )

            steps_taken[-1]["status"] = "completed"

    # Final quality check using LLM scoring WITH CONFIDENCE (3 runs for reliable decision)
    # This is critical for rollback decisions - we need statistical confidence
    logger.info("Final quality check: Using LLM scoring with confidence (3 runs)")
    final_score_result = await llm_score_prompt_with_confidence(
        prompt=current_prompt,
        use_case=use_case,
        llm_client=llm_client,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        num_runs=3  # Multiple runs for statistical confidence
    )
    final_score = final_score_result.mean_score

    # Also get heuristic score for validation
    final_analysis = analyze_prompt(current_prompt)
    heuristic_score = final_analysis.quality_score
    
    # Log comprehensive scoring info
    logger.info(
        f"Scores - LLM: {final_score:.1f} ± {final_score_result.std_deviation:.2f} "
        f"({final_score_result.confidence_level} confidence), Heuristic: {heuristic_score}/10"
    )
    
    # Warn if LLM scoring has high variance
    if not final_score_result.is_reliable:
        logger.warning(
            f"LLM scoring variance is high (stddev={final_score_result.std_deviation:.2f}). "
            f"Using conservative approach for rollback decision."
        )
        # Use worst-case score when confidence is low
        final_score = min(final_score_result.individual_scores) if final_score_result.individual_scores else final_score

    # ========================================================================
    # ROLLBACK DECISION - STRICT THRESHOLD
    # ========================================================================
    # Threshold: -1.0 max (10% degradation on 10-point scale)
    # Previously was -2.0 which allowed 20% degradation - too lenient!
    # 
    # Rationale for -1.0:
    # - A prompt rated 8.0 should not drop below 7.0 after "improvement"
    # - If improvement causes >10% quality drop, it's not an improvement
    # - Critical DNA issues (missing variables, broken format) trigger immediate revert
    MAX_ALLOWED_SCORE_DROP = 1.0  # Strict: max 1 point (10%) drop allowed
    
    critical_issues = []
    if final_validation:
        critical_issues = [
            i for i in final_validation.issues 
            if any(kw in i.lower() for kw in ["variable", "code block", "json", "template", "schema"])
        ]

    score_drop = original_score - final_score
    should_revert = (
        score_drop > MAX_ALLOWED_SCORE_DROP or  # STRICT: max 1 point drop
        len(critical_issues) > 0  # Critical DNA issues
    )
    
    revert_reason = ""
    if score_drop > MAX_ALLOWED_SCORE_DROP:
        revert_reason = f"Score dropped by {score_drop:.1f} points (max allowed: {MAX_ALLOWED_SCORE_DROP})"
    elif critical_issues:
        revert_reason = f"Critical DNA issues: {', '.join(critical_issues[:3])}"

    if should_revert:
        logger.warning(f"Reverting: {revert_reason}")
        return AgenticRewriteResult(
            original_prompt=prompt,
            final_prompt=prompt,
            original_score=original_score,
            final_score=original_score,
            iterations=iteration,
            steps_taken=steps_taken,
            improvement_plan={
                "preserve": plan.preserve,
                "improve": plan.improve,
                "add": plan.add
            },
            validation={
                "is_valid": False,
                "issues": critical_issues if critical_issues else [f"Quality dropped {score_drop:.1f} points (threshold: {MAX_ALLOWED_SCORE_DROP})"],
                "quality_delta": -score_drop,
                "max_allowed_drop": MAX_ALLOWED_SCORE_DROP
            },
            no_change=True,
            reason=f"Rewrite reverted: {revert_reason}. Original prompt preserved."
        )

    improvement = final_score - original_score
    logger.info(f"Rewrite successful: {original_score:.1f} -> {final_score:.1f} (+{improvement:.1f})")

    return AgenticRewriteResult(
        original_prompt=prompt,
        final_prompt=current_prompt,
        original_score=original_score,
        final_score=final_score,
        iterations=iteration,
        steps_taken=steps_taken,
        improvement_plan={
            "preserve": plan.preserve,
            "improve": [{"area": i.get("area"), "current": i.get("current"), "target": i.get("target")} for i in plan.improve],
            "add": plan.add
        },
        validation={
            "is_valid": final_validation.is_valid if final_validation else True,
            "dna_preserved": final_validation.dna_preserved if final_validation else True,
            "issues": final_validation.issues if final_validation else [],
            "warnings": final_validation.warnings if final_validation else [],
            "quality_delta": improvement,
            "score_confidence": {
                "std_deviation": final_score_result.std_deviation,
                "confidence_level": final_score_result.confidence_level,
                "individual_scores": final_score_result.individual_scores
            }
        },
        no_change=False,
        reason=f"Successfully improved prompt from {original_score:.1f}/10 to {final_score:.1f}/10 (+{improvement:.1f}, {final_score_result.confidence_level} confidence)"
    )


def result_to_dict(result: AgenticRewriteResult) -> Dict[str, Any]:
    """Convert AgenticRewriteResult to dictionary for JSON serialization"""
    return {
        "original_prompt": result.original_prompt,
        "final_prompt": result.final_prompt,
        "original_score": result.original_score,
        "final_score": result.final_score,
        "iterations": result.iterations,
        "steps_taken": result.steps_taken,
        "improvement_plan": result.improvement_plan,
        "validation": result.validation,
        "no_change": result.no_change,
        "reason": result.reason
    }
