"""
Shared Eval Best Practices Module

This module encapsulates Anthropic's recommended best practices for AI agent evaluations.
Based on: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents

Key Principles:
1. Grade outcomes, not paths - Avoid checking rigid step sequences
2. Balance problem sets - Test both positive and negative cases
3. Clear success criteria - Unambiguous criteria where experts would agree
4. Avoid over-constraining - Don't penalize valid alternative solutions
5. Include domain-specific failure modes - Common failure patterns for the domain
6. Reference solutions - Known-working outputs to validate grading
7. Calibrate model-based graders - Regular calibration against human judgment
"""

from typing import Dict, List, Optional, Any
from enum import Enum


class EvalBestPractice(Enum):
    """Anthropic's recommended eval best practices"""
    OUTCOME_FOCUSED = "outcome_focused"
    BALANCED_CRITERIA = "balanced_criteria"
    CLEAR_SUCCESS = "clear_success"
    FLEXIBLE_GRADING = "flexible_grading"
    DOMAIN_FAILURES = "domain_failures"
    CALIBRATION = "calibration"


# System prompt for improving eval prompts based on best practices
ANTHROPIC_EVAL_IMPROVEMENT_SYSTEM_PROMPT = """You are an expert at improving LLM evaluation prompts based on Anthropic's best practices for AI agent evaluations.

## Core Principles (from Anthropic's Engineering Blog):

### 1. Grade Outcomes, Not Paths
- Avoid checking that agents followed very specific steps in rigid sequences
- Focus on whether the final result meets the requirements
- Allow for valid alternative approaches

### 2. Balance Problem Sets
- Ensure criteria test both positive cases (what should happen) AND negative cases (what shouldn't happen)
- One-sided evaluations lead to poor real-world performance
- Include edge cases and boundary conditions

### 3. Clear Success Criteria
- Write unambiguous criteria where two domain experts would independently reach the same verdict
- Avoid subjective language like "clearly", "obviously", "appropriately"
- Be specific about what constitutes pass/fail

### 4. Avoid Over-Constraining
- Don't penalize valid alternative solutions
- High-performing models may find unexpected but correct approaches
- Focus on correctness of output, not method

### 5. Domain-Specific Failure Modes
- Include common ways this type of system can fail
- Be explicit about what constitutes failure
- Cover both obvious and subtle failure patterns

### 6. Grader Calibration
- Model-based graders require calibration against human judgment
- Include calibration examples showing expected scores for different quality levels
- Avoid vague rubrics that produce inconsistent assessments

## When Improving an Eval Prompt:
1. Preserve effective elements that follow these principles
2. Strengthen criteria that are vague or subjective
3. Add negative test cases if only positive cases exist
4. Ensure the rubric has clear distinctions between score levels
5. Add domain-specific failure modes if missing
6. Remove over-constraining requirements that penalize valid alternatives
"""


def get_eval_improvement_user_prompt(
    existing_eval_prompt: str,
    system_prompt: str,
    use_case: str,
    requirements: str,
    input_var_name: str = "INPUT",
    output_var_name: str = "OUTPUT",
    detected_type: str = "general",
    eval_changes: Optional[List[str]] = None
) -> str:
    """
    Generate the user prompt for improving an existing eval prompt.

    Args:
        existing_eval_prompt: The current eval prompt to improve
        system_prompt: The system prompt being evaluated
        use_case: Description of the use case
        requirements: Key requirements for the system
        input_var_name: Variable name for input (e.g., "callTranscript")
        output_var_name: Variable name for output (default "OUTPUT")
        detected_type: Type of system detected (e.g., "analytical", "conversational")
        eval_changes: History of previous changes made

    Returns:
        User prompt for the LLM to improve the eval prompt
    """
    changes_context = ""
    if eval_changes:
        changes_context = f"\n\nPrevious Changes Made:\n" + "\n".join(f"- {change}" for change in eval_changes[-5:])

    return f"""IMPROVE this existing evaluation prompt based on Anthropic's best practices:

## Context
Use Case: {use_case}
Requirements: {requirements}
System Type: {detected_type}

## System Prompt Being Evaluated:
{system_prompt}

## Template Variables
- Input variable: {{{{{input_var_name}}}}}
- Output variable: {{{{{output_var_name}}}}}

---
## EXISTING EVALUATION PROMPT TO IMPROVE:
{existing_eval_prompt}
{changes_context}
---

## Improvement Instructions:
1. **Outcome Focus**: Ensure criteria judge results, not specific steps taken
2. **Balance**: Add negative test criteria if only positive ones exist
3. **Clarity**: Replace vague language with specific, measurable criteria
4. **Flexibility**: Remove requirements that penalize valid alternative approaches
5. **Failure Modes**: Add domain-specific ways this system type commonly fails
6. **Rubric**: Ensure clear distinctions between score levels (1-5)

## Requirements:
- Preserve the variable placeholders exactly: {{{{{input_var_name}}}}} and {{{{{output_var_name}}}}}
- Keep effective elements from the existing prompt
- Return ONLY the improved evaluation prompt text
- No XML tags or explanations"""


def get_fresh_eval_generation_prompt(
    system_prompt: str,
    use_case: str,
    requirements: str,
    input_var_name: str = "INPUT",
    output_var_name: str = "OUTPUT",
    detected_type: str = "general",
    domain_criteria: str = "",
    domain_rubric: str = ""
) -> str:
    """
    Generate the user prompt for creating a fresh eval prompt with best practices.

    Args:
        system_prompt: The system prompt being evaluated
        use_case: Description of the use case
        requirements: Key requirements for the system
        input_var_name: Variable name for input
        output_var_name: Variable name for output
        detected_type: Type of system detected
        domain_criteria: Domain-specific evaluation criteria
        domain_rubric: Domain-specific scoring rubric

    Returns:
        User prompt for the LLM to generate a fresh eval prompt
    """
    return f"""Create a comprehensive evaluation prompt following Anthropic's best practices:

## Context
Use Case: {use_case}
Requirements: {requirements}
System Type: {detected_type}

## System Prompt Being Evaluated:
{system_prompt}

## Template Variables
- Input variable: {{{{{input_var_name}}}}}
- Output variable: {{{{{output_var_name}}}}}

## Domain-Specific Criteria:
{domain_criteria if domain_criteria else "Apply general evaluation criteria appropriate for this system type."}

## Rubric Guidelines:
{domain_rubric if domain_rubric else "Create a 1-5 scoring rubric with clear distinctions between levels."}

## Best Practices to Follow:
1. **Outcome Focus**: Grade results, not the specific path taken to get there
2. **Balanced Criteria**: Include both positive (should do) and negative (shouldn't do) criteria
3. **Clear Success**: Use unambiguous, measurable criteria
4. **Flexible Grading**: Don't penalize valid alternative approaches
5. **Failure Modes**: Include domain-specific failure patterns
6. **Calibration-Ready**: Create clear rubric distinctions for consistent scoring

## Required Output Format:
- Use markdown-style headers with ** for sections
- Include {{{{{input_var_name}}}}} and {{{{{output_var_name}}}}} placeholders
- Define evaluation criteria with clear pass/fail conditions
- Create a detailed 1-5 scoring rubric
- Require JSON output with "score" (1-5) and "reasoning" fields

Return ONLY the evaluation prompt text. No XML tags or explanations."""


def get_anthropic_system_prompt_for_generation(
    system_description: str,
    input_var_name: str,
    output_var_name: str,
    template_vars: List[str],
    detected_type: str,
    domain_criteria: str = "",
    domain_rubric: str = ""
) -> str:
    """
    Generate the system prompt for creating eval prompts with Anthropic best practices.

    This is used for fresh generation of eval prompts.
    """
    return f"""You are an expert prompt engineer specializing in LLM-as-Judge evaluation systems.

Your task is to create a comprehensive evaluation prompt for a {system_description}.

## CRITICAL: Variable Naming
The system prompt being evaluated uses these template variables: {template_vars if template_vars else ['INPUT']}
You MUST use the EXACT same variable names in your evaluation prompt:
- Use {{{{{input_var_name}}}}} for the input (NOT {{{{INPUT}}}} unless that's the actual variable name)
- Use {{{{{output_var_name}}}}} for the AI response to evaluate

## Anthropic's Eval Best Practices (MUST FOLLOW):

### 1. Grade Outcomes, Not Paths
- Focus on whether the result meets requirements, not how it got there
- Avoid checking for specific step sequences
- Allow valid alternative approaches

### 2. Balance Problem Sets
- Include criteria for what the system SHOULD do
- Include criteria for what the system SHOULD NOT do
- Cover edge cases and boundary conditions

### 3. Clear Success Criteria
- Write unambiguous criteria (two experts would agree on the verdict)
- Avoid subjective language like "appropriate", "clearly", "obviously"
- Be specific and measurable

### 4. Avoid Over-Constraining
- Don't penalize valid alternative solutions
- Focus on correctness of output, not method used
- Allow for unexpected but correct approaches

### 5. Domain-Specific Failure Modes
- Include common ways this type of system fails
- Be explicit about what constitutes failure
- Cover subtle failure patterns, not just obvious ones

## Domain Requirements for {detected_type}:
{domain_criteria if domain_criteria else "Apply standard evaluation criteria."}

## Rubric Requirements:
{domain_rubric if domain_rubric else "Create a detailed 1-5 scoring rubric."}

## Structure Requirements:
1. Use markdown-style headers with ** for sections
2. Include {{{{{input_var_name}}}}} and {{{{{output_var_name}}}}} placeholders (EXACT variable names!)
3. Define a clear evaluator role with expertise in {detected_type} analysis
4. Create evaluation criteria weighted by importance (must sum to 100%)
5. Include both positive AND negative criteria
6. Require JSON output with "score" (1-5 number) and "reasoning" (specific explanation)

Return ONLY the evaluation prompt text in plain text format. No XML tags. No explanations."""


def get_anthropic_system_prompt_for_improvement(
    system_description: str,
    input_var_name: str,
    output_var_name: str,
    template_vars: List[str],
    detected_type: str,
    domain_criteria: str = "",
    domain_rubric: str = ""
) -> str:
    """
    Generate the system prompt for improving existing eval prompts with Anthropic best practices.

    This is used when regenerating/improving existing eval prompts.
    """
    return f"""You are an expert prompt engineer specializing in LLM-as-Judge evaluation systems.

Your task is to IMPROVE an existing evaluation prompt for a {system_description}.

## CRITICAL: Variable Naming
The system prompt being evaluated uses these template variables: {template_vars if template_vars else ['INPUT']}
You MUST use the EXACT same variable names in your evaluation prompt:
- Use {{{{{input_var_name}}}}} for the input (NOT {{{{INPUT}}}} unless that's the actual variable name)
- Use {{{{{output_var_name}}}}} for the AI response to evaluate

## Anthropic's Eval Best Practices (Apply These to Improve):

### 1. Grade Outcomes, Not Paths
- Remove any criteria that check for specific step sequences
- Focus on whether the result meets requirements, not how it got there
- Allow valid alternative approaches

### 2. Balance Problem Sets
- If only positive criteria exist, ADD negative criteria
- Ensure criteria test both "should do" and "shouldn't do"
- Add edge cases and boundary conditions if missing

### 3. Clear Success Criteria
- Replace vague language with specific, measurable criteria
- Avoid words like "appropriate", "clearly", "obviously"
- Write criteria where two experts would agree on the verdict

### 4. Avoid Over-Constraining
- Remove requirements that penalize valid alternative solutions
- Keep focus on correctness of output, not method
- Don't require specific formatting unless it's essential

### 5. Domain-Specific Failure Modes
- Add common failure patterns for {detected_type} if missing
- Be explicit about what constitutes failure
- Include both obvious and subtle failure patterns

## What to Preserve:
- Keep effective elements from the existing prompt
- Retain the variable placeholders exactly as they are
- Maintain any criteria that are already clear and balanced

## What to Improve:
- Strengthen weak or vague criteria
- Add missing negative test cases
- Improve the scoring rubric specificity
- Add domain-specific failure modes if missing

## Domain Requirements for {detected_type}:
{domain_criteria if domain_criteria else "Apply standard evaluation criteria."}

## Rubric Requirements:
{domain_rubric if domain_rubric else "Ensure clear distinctions between score levels."}

Return ONLY the improved evaluation prompt text in plain text format. No XML tags. No explanations."""


def apply_best_practices_check(eval_prompt: str) -> Dict[str, Any]:
    """
    Check an eval prompt against Anthropic's best practices.
    Returns a report of which practices are followed and which need improvement.

    This is a simple heuristic check - for thorough validation, use LLM-based validation.
    """
    checks = {
        "outcome_focused": {
            "passed": True,
            "issues": [],
            "description": "Grades outcomes, not paths"
        },
        "balanced_criteria": {
            "passed": False,
            "issues": [],
            "description": "Tests both positive and negative cases"
        },
        "clear_success": {
            "passed": True,
            "issues": [],
            "description": "Clear, unambiguous success criteria"
        },
        "flexible_grading": {
            "passed": True,
            "issues": [],
            "description": "Doesn't over-constrain valid alternatives"
        },
        "failure_modes": {
            "passed": False,
            "issues": [],
            "description": "Includes domain-specific failure modes"
        },
        "calibration_ready": {
            "passed": False,
            "issues": [],
            "description": "Has clear rubric for calibration"
        }
    }

    prompt_lower = eval_prompt.lower()

    # Check for balanced criteria (positive and negative)
    negative_indicators = ["should not", "shouldn't", "must not", "mustn't", "avoid", "do not", "don't", "never", "prohibited", "forbidden"]
    positive_indicators = ["should", "must", "ensure", "verify", "check that", "confirm"]

    has_negative = any(ind in prompt_lower for ind in negative_indicators)
    has_positive = any(ind in prompt_lower for ind in positive_indicators)

    if has_negative and has_positive:
        checks["balanced_criteria"]["passed"] = True
    else:
        if not has_negative:
            checks["balanced_criteria"]["issues"].append("Missing negative criteria (what NOT to do)")
        if not has_positive:
            checks["balanced_criteria"]["issues"].append("Missing positive criteria (what to do)")

    # Check for vague language (excluding legitimate uses like "What Good Looks Like")
    vague_terms = ["appropriate", "obviously", "clearly", "properly", "correctly", "bad", "nice", "reasonable"]
    # "good" is only vague if not part of "what good looks like" or similar domain terms
    found_vague = [term for term in vague_terms if term in prompt_lower]
    # Check for standalone "good" that's not part of WGLL
    if "good" in prompt_lower and "what good looks like" not in prompt_lower and "wgll" not in prompt_lower:
        found_vague.append("good")
    if found_vague:
        checks["clear_success"]["passed"] = False
        checks["clear_success"]["issues"].append(f"Contains vague terms: {', '.join(found_vague)}")

    # Check for failure modes
    failure_indicators = ["fail", "failure", "error", "invalid", "incorrect", "wrong", "missing", "hallucination", "fabricat"]
    if any(ind in prompt_lower for ind in failure_indicators):
        checks["failure_modes"]["passed"] = True
    else:
        checks["failure_modes"]["issues"].append("No explicit failure modes mentioned")

    # Check for rubric/scoring
    rubric_indicators = ["score 1", "score 2", "score 3", "score 4", "score 5", "1:", "2:", "3:", "4:", "5:", "rubric", "scoring"]
    if any(ind in prompt_lower for ind in rubric_indicators):
        checks["calibration_ready"]["passed"] = True
    else:
        checks["calibration_ready"]["issues"].append("No clear scoring rubric found")

    # Check for over-constraining (step-by-step requirements)
    step_indicators = ["step 1", "step 2", "first step", "second step", "then do", "after that", "following steps"]
    if any(ind in prompt_lower for ind in step_indicators):
        checks["outcome_focused"]["passed"] = False
        checks["outcome_focused"]["issues"].append("Contains step-by-step requirements that may over-constrain")

    # Calculate overall score
    passed_count = sum(1 for check in checks.values() if check["passed"])
    total_count = len(checks)

    return {
        "checks": checks,
        "passed_count": passed_count,
        "total_count": total_count,
        "score": round((passed_count / total_count) * 100),
        "summary": f"Passed {passed_count}/{total_count} best practice checks"
    }
