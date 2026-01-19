"""
Enhanced Prompt Analyzer V2 - Python + LLM Hybrid

Combines fast programmatic analysis with deep LLM understanding for:
1. Better quality assessment
2. Semantic understanding of intent
3. Domain-specific insights
4. Actionable improvement suggestions

The hybrid approach:
- Python: Fast DNA extraction, structure detection, basic scoring
- LLM: Deep analysis, semantic understanding, smart suggestions
"""
import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict

from prompt_analyzer import (
    analyze_prompt as programmatic_analyze,
    analysis_to_dict,
    PromptType,
    PromptDNA,
    PromptAnalysis
)
from llm_client_v2 import EnhancedLLMClient, parse_json_response

logger = logging.getLogger(__name__)


@dataclass
class EnhancedAnalysis:
    """Complete analysis combining programmatic and LLM insights"""
    # From programmatic analysis
    prompt_type: str
    prompt_types_detected: List[str]
    dna: Dict[str, Any]
    programmatic_score: float
    quality_breakdown: Dict[str, float]

    # From LLM analysis
    llm_score: float
    combined_score: float  # Weighted average

    # Understanding
    intent_summary: str
    target_audience: str
    expected_input_type: str
    expected_output_description: str

    # Quality insights
    strengths: List[str]
    weaknesses: List[str]
    improvement_areas: List[Dict[str, str]]  # {area, description, priority, suggestion}

    # Issues detected
    ambiguities: List[str]
    contradictions: List[str]
    missing_elements: List[str]

    # Suggestions
    suggestions: List[Dict[str, str]]  # {category, suggestion, impact}

    # Evaluation guidance
    suggested_eval_dimensions: List[Dict[str, str]]
    suggested_test_categories: List[Dict[str, Any]]

    # Metadata
    analysis_method: str = "hybrid"
    llm_enhanced: bool = True


async def analyze_prompt_hybrid(
    prompt_text: str,
    use_case: str = "",
    requirements: str = "",
    structured_requirements: Optional[Dict[str, Any]] = None,
    llm_client: EnhancedLLMClient = None,
    provider: str = "openai",
    api_key: str = "",
    model_name: str = "gpt-4o",
    quick_mode: bool = False
) -> EnhancedAnalysis:
    """
    Perform hybrid Python + LLM analysis of a prompt.

    Args:
        prompt_text: The system prompt to analyze
        use_case: Optional context about what the prompt is for
        requirements: Optional requirements the prompt should meet
        structured_requirements: Optional structured requirements dict
        llm_client: LLM client for enhanced analysis
        provider: LLM provider
        api_key: API key for LLM
        model_name: Model to use
        quick_mode: If True, skip LLM and return programmatic only

    Returns:
        EnhancedAnalysis with combined insights
    """
    # Step 1: Fast programmatic analysis (always runs)
    programmatic = programmatic_analyze(prompt_text)
    prog_dict = analysis_to_dict(programmatic)

    # If no LLM client or quick mode, return programmatic-only results
    if quick_mode or not llm_client or not api_key:
        return _convert_programmatic_to_enhanced(programmatic, prog_dict)

    # Step 2: LLM-enhanced analysis
    try:
        llm_analysis = await _get_llm_analysis(
            prompt_text=prompt_text,
            programmatic=prog_dict,
            use_case=use_case,
            requirements=requirements,
            structured_requirements=structured_requirements,
            llm_client=llm_client,
            provider=provider,
            api_key=api_key,
            model_name=model_name
        )

        # Step 3: Combine results
        return _combine_analyses(programmatic, prog_dict, llm_analysis)

    except Exception as e:
        logger.error(f"LLM analysis failed, falling back to programmatic: {e}")
        return _convert_programmatic_to_enhanced(programmatic, prog_dict)


async def _get_llm_analysis(
    prompt_text: str,
    programmatic: Dict[str, Any],
    use_case: str,
    requirements: str,
    structured_requirements: Optional[Dict[str, Any]],
    llm_client: EnhancedLLMClient,
    provider: str,
    api_key: str,
    model_name: str
) -> Dict[str, Any]:
    """Get LLM-enhanced analysis"""

    system_prompt = """You are an expert prompt engineer and QA analyst. Analyze this system prompt deeply to understand its intent, quality, and areas for improvement.

Your analysis should go BEYOND surface-level observations. Focus on:

1. **INTENT & PURPOSE**: What is this prompt truly trying to accomplish? Who is the target user?

2. **QUALITY ASSESSMENT**: Rate the prompt 1-10 based on:
   - Clarity: Is it unambiguous?
   - Completeness: Does it cover all necessary aspects?
   - Structure: Is it well-organized?
   - Actionability: Can an LLM follow it reliably?
   - Edge case handling: Does it address unusual inputs?

3. **CRITICAL ISSUES**: Identify any:
   - Ambiguities that could cause inconsistent outputs
   - Contradictions between different instructions
   - Missing elements that are implicitly expected
   - Potential failure modes

4. **IMPROVEMENT OPPORTUNITIES**: Specific, actionable suggestions with priority

Return your analysis as JSON:
{
    "intent_summary": "One sentence describing what this prompt does",
    "target_audience": "Who will use the outputs from this prompt",
    "expected_input_type": "What kind of input this prompt expects",
    "expected_output_description": "What the output should look like",

    "quality_score": 7.5,
    "quality_reasoning": "Brief explanation of the score",

    "strengths": [
        "Specific strength 1",
        "Specific strength 2"
    ],

    "weaknesses": [
        "Specific weakness 1",
        "Specific weakness 2"
    ],

    "ambiguities": [
        "Ambiguous instruction or requirement"
    ],

    "contradictions": [
        "Contradictory instructions found"
    ],

    "missing_elements": [
        "Element that should be present but isn't"
    ],

    "improvement_areas": [
        {
            "area": "Section/aspect to improve",
            "description": "What's wrong",
            "priority": "high|medium|low",
            "suggestion": "Specific fix"
        }
    ],

    "suggestions": [
        {
            "category": "structure|clarity|completeness|safety|examples",
            "suggestion": "Specific actionable suggestion",
            "impact": "high|medium|low"
        }
    ],

    "eval_dimensions": [
        {
            "name": "Dimension name",
            "description": "What to evaluate",
            "why_important": "Why this matters for this prompt"
        }
    ],

    "test_categories": [
        {
            "category": "positive|edge_case|negative|adversarial",
            "description": "What to test",
            "example_input": "Example input for this category"
        }
    ]
}"""

    # Build context from programmatic analysis
    prog_context = f"""
**Programmatic Analysis Results:**
- Detected Type: {programmatic.get('prompt_type', 'unknown')}
- Quality Score: {programmatic.get('quality_score', 0)}/10
- Output Format: {programmatic.get('dna', {}).get('output_format', 'not specified')}
- Template Variables: {programmatic.get('dna', {}).get('template_variables', [])}
- Sections Found: {programmatic.get('dna', {}).get('sections', [])}
- Role: {programmatic.get('dna', {}).get('role', 'not specified')}
- Constraints: {len(programmatic.get('dna', {}).get('constraints', []))} found
"""

    # Add structured requirements context if provided
    structured_req_context = ""
    if structured_requirements:
        sr_parts = []
        if structured_requirements.get('must_do'):
            sr_parts.append(f"**Must Do:** {', '.join(structured_requirements['must_do'])}")
        if structured_requirements.get('must_not_do'):
            sr_parts.append(f"**Must NOT Do:** {', '.join(structured_requirements['must_not_do'])}")
        if structured_requirements.get('tone'):
            sr_parts.append(f"**Required Tone:** {structured_requirements['tone']}")
        if structured_requirements.get('output_format'):
            sr_parts.append(f"**Required Output Format:** {structured_requirements['output_format']}")
        if structured_requirements.get('success_criteria'):
            sr_parts.append(f"**Success Criteria:** {', '.join(structured_requirements['success_criteria'])}")
        if structured_requirements.get('edge_cases'):
            sr_parts.append(f"**Edge Cases to Handle:** {', '.join(structured_requirements['edge_cases'])}")
        
        if sr_parts:
            structured_req_context = "\n\n**Structured Requirements:**\n" + "\n".join(sr_parts)

    user_message = f"""Analyze this system prompt:

```
{prompt_text}
```

{f"**Use Case:** {use_case}" if use_case else ""}
{f"**Requirements:** {requirements}" if requirements else ""}
{structured_req_context}

{prog_context}

Provide a deep analysis focusing on semantic understanding, not just surface patterns. Be specific and actionable in your suggestions.
{f"IMPORTANT: Check if the prompt addresses all structured requirements (must_do, must_not_do, tone, output_format, success_criteria, edge_cases)." if structured_requirements else ""}"""

    result = await llm_client.chat(
        system_prompt=system_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.2,  # Low for consistent analysis
        max_tokens=4000
    )

    if result.get("error"):
        raise Exception(f"LLM call failed: {result['error']}")

    # Parse response
    parsed = parse_json_response(result.get("output", ""), "object")

    if not parsed:
        raise Exception("Failed to parse LLM response as JSON")

    return parsed


def _convert_programmatic_to_enhanced(
    programmatic: PromptAnalysis,
    prog_dict: Dict[str, Any]
) -> EnhancedAnalysis:
    """Convert programmatic-only analysis to EnhancedAnalysis format"""
    return EnhancedAnalysis(
        prompt_type=prog_dict.get("prompt_type", "unknown"),
        prompt_types_detected=prog_dict.get("prompt_types_detected", []),
        dna=prog_dict.get("dna", {}),
        programmatic_score=prog_dict.get("quality_score", 5.0),
        quality_breakdown=prog_dict.get("quality_breakdown", {}),

        llm_score=0.0,
        combined_score=prog_dict.get("quality_score", 5.0),

        intent_summary="Analysis based on pattern matching only",
        target_audience="Unknown - LLM analysis not performed",
        expected_input_type="Unknown",
        expected_output_description=prog_dict.get("dna", {}).get("output_format", "Unknown"),

        strengths=prog_dict.get("strengths", []),
        weaknesses=[],
        improvement_areas=[
            {"area": area, "description": area, "priority": "medium", "suggestion": ""}
            for area in prog_dict.get("improvement_areas", [])
        ],

        ambiguities=[],
        contradictions=[],
        missing_elements=[],

        suggestions=[
            {"category": "general", "suggestion": area, "impact": "medium"}
            for area in prog_dict.get("improvement_areas", [])
        ],

        suggested_eval_dimensions=prog_dict.get("suggested_eval_dimensions", []),
        suggested_test_categories=prog_dict.get("suggested_test_categories", []),

        analysis_method="programmatic_only",
        llm_enhanced=False
    )


def _combine_analyses(
    programmatic: PromptAnalysis,
    prog_dict: Dict[str, Any],
    llm_analysis: Dict[str, Any]
) -> EnhancedAnalysis:
    """Combine programmatic and LLM analyses"""

    prog_score = prog_dict.get("quality_score", 5.0)
    llm_score = llm_analysis.get("quality_score", 5.0)

    # Weighted average: 30% programmatic, 70% LLM
    # LLM weighted higher because it understands semantics
    combined_score = (prog_score * 0.3) + (llm_score * 0.7)

    # Merge strengths (deduplicate)
    all_strengths = list(set(
        prog_dict.get("strengths", []) +
        llm_analysis.get("strengths", [])
    ))

    # Merge improvement areas
    improvement_areas = llm_analysis.get("improvement_areas", [])
    for area in prog_dict.get("improvement_areas", []):
        if not any(imp.get("area") == area for imp in improvement_areas):
            improvement_areas.append({
                "area": area,
                "description": area,
                "priority": "medium",
                "suggestion": ""
            })

    return EnhancedAnalysis(
        prompt_type=prog_dict.get("prompt_type", "unknown"),
        prompt_types_detected=prog_dict.get("prompt_types_detected", []),
        dna=prog_dict.get("dna", {}),
        programmatic_score=prog_score,
        quality_breakdown=prog_dict.get("quality_breakdown", {}),

        llm_score=llm_score,
        combined_score=round(combined_score, 1),

        intent_summary=llm_analysis.get("intent_summary", ""),
        target_audience=llm_analysis.get("target_audience", ""),
        expected_input_type=llm_analysis.get("expected_input_type", ""),
        expected_output_description=llm_analysis.get("expected_output_description", ""),

        strengths=all_strengths,
        weaknesses=llm_analysis.get("weaknesses", []),
        improvement_areas=improvement_areas,

        ambiguities=llm_analysis.get("ambiguities", []),
        contradictions=llm_analysis.get("contradictions", []),
        missing_elements=llm_analysis.get("missing_elements", []),

        suggestions=llm_analysis.get("suggestions", []),

        suggested_eval_dimensions=llm_analysis.get("eval_dimensions", prog_dict.get("suggested_eval_dimensions", [])),
        suggested_test_categories=llm_analysis.get("test_categories", prog_dict.get("suggested_test_categories", [])),

        analysis_method="hybrid",
        llm_enhanced=True
    )


def enhanced_analysis_to_dict(analysis: EnhancedAnalysis) -> Dict[str, Any]:
    """Convert EnhancedAnalysis to dictionary for JSON serialization"""
    return asdict(analysis)


# Convenience function for quick programmatic-only analysis
def analyze_prompt_quick(prompt_text: str) -> Dict[str, Any]:
    """
    Quick programmatic-only analysis (no LLM).
    Use this for instant feedback before full analysis.
    """
    programmatic = programmatic_analyze(prompt_text)
    return analysis_to_dict(programmatic)
