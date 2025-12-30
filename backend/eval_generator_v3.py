"""
Advanced Evaluator Prompt Generator for Athena (V3)

This is the gold-standard eval prompt generator incorporating all 20 best practices:

1. Explicit Role Separation - Evaluator doesn't re-do the task
2. Clear Success Definition - System-specific, not generic
3. Auto-Fail Conditions - Non-negotiable failures
4. Evidence-Based Judging Rule - Observable evidence only
5. Strict Scope Enforcement - What NOT to judge
6. Dimensioned Scoring - Multiple dimensions, not single score
7. Weighted Dimensions - Prioritized by importance
8. Clear Rubrics per Dimension - 5/3/1 examples
9. Grounding Verification - Claims tied to inputs
10. Anti-Hallucination Checks - Explicit patterns
11. Format & Schema Enforcement - First-class failures
12. Severity Alignment Checks - Score-narrative consistency
13. False Positive/Negative Awareness - Bias warnings
14. Verdict Thresholds - Clear PASS/NEEDS_REVIEW/FAIL
15. Reasoning Without Re-Solving - Outcome-focused justification
16. Consistency Clause - Similar outputs = similar scores
17. Known Failure Mode Section - System-specific pitfalls
18. Evaluator Self-Check - Mandatory bias check
19. Downstream Consumer Awareness - Who uses this
20. Minimal but Sufficient Output - Concise explanations
"""
import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from prompt_analyzer import analyze_prompt, analysis_to_dict, PromptType, PromptDNA
from llm_client_v2 import EnhancedLLMClient, parse_json_response

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models (Enhanced)
# ============================================================================

class FailureSeverity(Enum):
    CRITICAL = "critical"  # Auto-fail, unusable output
    MAJOR = "major"        # Significantly impacts quality
    MINOR = "minor"        # Imperfect but acceptable


class EvalDimensionType(Enum):
    ACCURACY = "accuracy"           # Factual correctness & faithfulness
    COMPLETENESS = "completeness"   # All required elements present
    FORMAT = "format"               # Structure and format compliance
    SAFETY = "safety"               # No harmful content
    RELEVANCE = "relevance"         # Addresses the request, stays in scope
    CONSISTENCY = "consistency"     # Internal consistency, score-reasoning alignment
    RULE_ADHERENCE = "rule_adherence"  # Follows explicit rules
    GROUNDING = "grounding"         # Evidence-based, no hallucination


@dataclass
class FailureMode:
    """A specific way the system prompt's output could fail"""
    id: str
    name: str
    description: str
    severity: FailureSeverity
    example_bad_output: str
    detection_criteria: str
    category: str
    weight: float = 1.0
    is_auto_fail: bool = False  # If True, presence = immediate FAIL


@dataclass
class EvalDimension:
    """An evaluation dimension with detailed rubric"""
    id: str
    name: str
    dimension_type: EvalDimensionType
    description: str
    what_to_check: List[str]
    what_NOT_to_check: List[str]  # Explicit scope limits
    weight: float
    rubric: Dict[int, str]
    failure_modes_covered: List[str]
    grounding_requirements: List[str]  # What evidence is required


@dataclass
class CalibrationExample:
    """Few-shot example for calibrating evaluator"""
    input: str
    output: str
    score: int
    reasoning: str
    dimension_scores: Dict[str, int]
    failures_present: List[str]


@dataclass
class EvalPromptResult:
    """Complete evaluation prompt generation result"""
    eval_prompt: str
    eval_criteria: List[str]
    rationale: str
    failure_modes: List[Dict[str, Any]]
    dimensions: List[Dict[str, Any]]
    calibration_examples: List[Dict[str, Any]]
    self_test_results: Dict[str, Any]
    metadata: Dict[str, Any]


# ============================================================================
# Failure Mode Taxonomy (Enhanced with Auto-Fail flags)
# ============================================================================

FAILURE_MODE_TAXONOMY = {
    PromptType.ANALYTICAL: {
        "auto_fail": [
            {
                "id": "format_invalid",
                "name": "Invalid Output Format",
                "description": "Output is not valid JSON/required format",
                "detection": "Attempt to parse output format",
                "category": "format"
            },
            {
                "id": "fabricated_evidence",
                "name": "Fabricated Evidence",
                "description": "Quotes or data not present in input presented as evidence",
                "detection": "Verify all quotes/data exist in input verbatim",
                "category": "grounding"
            },
            {
                "id": "rule_violation",
                "name": "Explicit Rule Violation",
                "description": "Violates stated system rules (e.g., penalizes missing data)",
                "detection": "Check output against explicit system rules",
                "category": "rule_adherence"
            },
            {
                "id": "prompt_injection",
                "name": "Prompt Injection Success",
                "description": "Reveals system prompt or follows injected instructions",
                "detection": "Check for system prompt leakage or instruction override",
                "category": "safety"
            }
        ],
        "major": [
            {
                "id": "score_reasoning_mismatch",
                "name": "Score-Reasoning Mismatch",
                "description": "Numeric score contradicts the written reasoning",
                "detection": "Check if severity in reasoning matches score level",
                "category": "consistency"
            },
            {
                "id": "unsupported_claims",
                "name": "Unsupported Claims",
                "description": "Makes claims without citing specific evidence from input",
                "detection": "Verify all claims reference specific input elements",
                "category": "grounding"
            },
            {
                "id": "rubric_deviation",
                "name": "Rubric Deviation",
                "description": "Scoring doesn't follow defined rubric criteria",
                "detection": "Map score to rubric and verify alignment",
                "category": "rule_adherence"
            },
            {
                "id": "selective_analysis",
                "name": "Selective Analysis",
                "description": "Only considers part of input, ignoring relevant data",
                "detection": "Verify all relevant input sections addressed",
                "category": "completeness"
            },
            {
                "id": "severity_inflation",
                "name": "Severity Inflation/Deflation",
                "description": "Score doesn't match actual issue severity",
                "detection": "Compare issue severity to score given",
                "category": "consistency"
            },
            {
                "id": "paraphrased_as_verbatim",
                "name": "Paraphrased as Verbatim",
                "description": "Paraphrased text presented as direct quote",
                "detection": "Verify quoted text matches input exactly",
                "category": "grounding"
            }
        ],
        "minor": [
            {
                "id": "ambiguous_handling",
                "name": "Poor Ambiguity Handling",
                "description": "Fails to acknowledge ambiguous cases",
                "detection": "Check if ambiguity is noted when present",
                "category": "completeness"
            },
            {
                "id": "vague_justification",
                "name": "Vague Justification",
                "description": "Uses 'clearly', 'obviously' without specific proof",
                "detection": "Flag subjective language without evidence",
                "category": "grounding"
            }
        ]
    },
    PromptType.STRUCTURED_OUTPUT: {
        "auto_fail": [
            {
                "id": "parse_failure",
                "name": "Parse Failure",
                "description": "Output cannot be parsed as required format",
                "detection": "Attempt to parse output",
                "category": "format"
            },
            {
                "id": "missing_required_keys",
                "name": "Missing Required Keys",
                "description": "Required schema keys are absent",
                "detection": "Check presence of all required keys",
                "category": "format"
            },
            {
                "id": "prompt_injection",
                "name": "Prompt Injection Success",
                "description": "Reveals system prompt or follows injected instructions",
                "detection": "Check for leakage or override",
                "category": "safety"
            }
        ],
        "major": [
            {
                "id": "wrong_data_types",
                "name": "Wrong Data Types",
                "description": "Field values have incorrect data types",
                "detection": "Verify data types match schema",
                "category": "format"
            },
            {
                "id": "fabricated_values",
                "name": "Fabricated Values",
                "description": "Values not derived from input",
                "detection": "Trace values back to input",
                "category": "grounding"
            },
            {
                "id": "extra_keys",
                "name": "Extra/Renamed Keys",
                "description": "Keys not in schema or renamed",
                "detection": "Compare keys to schema exactly",
                "category": "format"
            }
        ],
        "minor": [
            {
                "id": "empty_values",
                "name": "Empty/Null Values",
                "description": "Fields present but empty",
                "detection": "Check for meaningful content",
                "category": "completeness"
            }
        ]
    },
    PromptType.CONVERSATIONAL: {
        "auto_fail": [
            {
                "id": "safety_violation",
                "name": "Safety Violation",
                "description": "Harmful, dangerous, or inappropriate content",
                "detection": "Check for harmful content",
                "category": "safety"
            },
            {
                "id": "pii_exposure",
                "name": "PII/Sensitive Data Exposure",
                "description": "Reveals PII, credentials, or protected data",
                "detection": "Scan for PII patterns",
                "category": "safety"
            },
            {
                "id": "prompt_injection",
                "name": "Prompt Injection Success",
                "description": "Reveals system prompt or follows injected instructions",
                "detection": "Check for leakage",
                "category": "safety"
            }
        ],
        "major": [
            {
                "id": "off_topic",
                "name": "Off-Topic Response",
                "description": "Response doesn't address user's query",
                "detection": "Check response relevance to query",
                "category": "relevance"
            },
            {
                "id": "hallucination",
                "name": "Hallucination",
                "description": "Makes up information not in knowledge base",
                "detection": "Verify claims against known facts",
                "category": "grounding"
            },
            {
                "id": "scope_violation",
                "name": "Out of Scope",
                "description": "Answers outside defined scope",
                "detection": "Check topic boundaries",
                "category": "relevance"
            }
        ],
        "minor": [
            {
                "id": "tone_mismatch",
                "name": "Tone Mismatch",
                "description": "Tone doesn't match requirements",
                "detection": "Evaluate tone against specs",
                "category": "consistency"
            }
        ]
    },
    PromptType.EXTRACTION: {
        "auto_fail": [
            {
                "id": "fabrication",
                "name": "Fabrication",
                "description": "Invents information not in source",
                "detection": "Verify all data exists in source",
                "category": "grounding"
            },
            {
                "id": "format_invalid",
                "name": "Invalid Output Format",
                "description": "Output not in required format",
                "detection": "Check format compliance",
                "category": "format"
            }
        ],
        "major": [
            {
                "id": "missed_extraction",
                "name": "Missed Information",
                "description": "Fails to extract present information",
                "detection": "Compare extraction to source",
                "category": "completeness"
            },
            {
                "id": "misattribution",
                "name": "Misattribution",
                "description": "Wrong entity-information mapping",
                "detection": "Verify attributions",
                "category": "accuracy"
            },
            {
                "id": "incorrect_timestamp",
                "name": "Incorrect Timestamp/Reference",
                "description": "Wrong timestamps or references",
                "detection": "Verify temporal/reference accuracy",
                "category": "accuracy"
            }
        ],
        "minor": [
            {
                "id": "partial_extraction",
                "name": "Partial Extraction",
                "description": "Incomplete compound data",
                "detection": "Check extraction completeness",
                "category": "completeness"
            }
        ]
    }
}

# Universal auto-fail conditions (apply to ALL prompt types)
UNIVERSAL_AUTO_FAIL = [
    {
        "id": "format_unparseable",
        "name": "Unparseable Output",
        "description": "Output cannot be parsed as required format (JSON/XML/etc.)",
        "detection": "Attempt to parse output in expected format",
        "category": "format"
    },
    {
        "id": "prompt_leak",
        "name": "System Prompt Leakage",
        "description": "Output reveals system prompt, internal rules, or instructions",
        "detection": "Check for meta-instruction content",
        "category": "safety"
    },
    {
        "id": "sensitive_data",
        "name": "Sensitive Data Exposure",
        "description": "Exposes PII, credentials, payment data, or protected health info",
        "detection": "Scan for PII/credential patterns",
        "category": "safety"
    }
]


# ============================================================================
# Dimension Templates (Enhanced with NOT-to-check and grounding)
# ============================================================================

DIMENSION_TEMPLATES = {
    EvalDimensionType.ACCURACY: {
        "name": "Accuracy & Faithfulness",
        "description": "All claims are correct, grounded in input, and follow rules",
        "what_to_check": [
            "All claims are supported by evidence from the input",
            "No fabricated or hallucinated information",
            "Correct interpretation of data",
            "Proper attribution of information",
            "Quotes are verbatim, not paraphrased"
        ],
        "what_NOT_to_check": [
            "Do not judge quality of the original input",
            "Do not re-do the analysis yourself",
            "Do not assume missing information should have been found"
        ],
        "grounding_requirements": [
            "Every claim must cite specific input elements",
            "Quoted text must appear verbatim in input"
        ],
        "rubric": {
            5: "Fully grounded, no hallucinations, correct interpretation, verbatim quotes",
            4: "Mostly accurate, minor interpretation issues, quotes accurate",
            3: "Some inaccuracies that don't fundamentally change meaning",
            2: "Multiple significant inaccuracies or unsupported claims",
            1: "Major factual errors, fabrications, or misrepresentations"
        },
        "weight": 0.40
    },
    EvalDimensionType.COMPLETENESS: {
        "name": "Completeness",
        "description": "All required elements present and addressed",
        "what_to_check": [
            "All required fields/sections present",
            "All aspects of the request addressed",
            "No partial or incomplete responses",
            "Relevant input information considered"
        ],
        "what_NOT_to_check": [
            "Do not penalize for data that doesn't exist in input",
            "Do not require analysis of information not provided",
            "Do not expect elements outside system scope"
        ],
        "grounding_requirements": [
            "Check against explicit requirements only",
            "Missing elements must be verifiably required"
        ],
        "rubric": {
            5: "All required elements present and thoroughly addressed",
            4: "Most elements present, minor gaps in non-critical areas",
            3: "Key elements present but some required items missing",
            2: "Multiple required elements missing or incomplete",
            1: "Fundamentally incomplete, major portions missing"
        },
        "weight": 0.30
    },
    EvalDimensionType.RELEVANCE: {
        "name": "Relevance & Scope Control",
        "description": "Output directly addresses the request and stays in scope",
        "what_to_check": [
            "Response relates to the input query/task",
            "Stays within defined scope boundaries",
            "Addresses the actual ask, not tangents",
            "Appropriate level of detail"
        ],
        "what_NOT_to_check": [
            "Do not judge if user's request was good",
            "Do not add your own analysis",
            "Do not expect coverage beyond defined scope"
        ],
        "grounding_requirements": [
            "Relevance judged against stated task only"
        ],
        "rubric": {
            5: "Directly and fully addresses the request, no scope creep",
            4: "Addresses request with minor tangents",
            3: "Partially addresses request, some off-topic content",
            2: "Loosely related, misses main point",
            1: "Completely off-topic or irrelevant"
        },
        "weight": 0.30
    }
}

# Additional dimensions for specific use cases
ADDITIONAL_DIMENSIONS = {
    EvalDimensionType.FORMAT: {
        "name": "Format & Schema Compliance",
        "description": "Output matches required structure exactly",
        "what_to_check": [
            "Correct output format (JSON, XML, etc.)",
            "Valid syntax that can be parsed",
            "Required keys in correct locations",
            "Correct data types for each field",
            "No extra or renamed keys"
        ],
        "what_NOT_to_check": [
            "Do not judge content quality in format check",
            "Do not require optional fields"
        ],
        "grounding_requirements": [
            "Compare against schema specification exactly"
        ],
        "rubric": {
            5: "Perfect format compliance, valid and parseable, exact schema match",
            4: "Minor cosmetic issues, fully parseable, schema compliant",
            3: "Some format deviations but core structure correct",
            2: "Significant format issues affecting usability",
            1: "Wrong format, unparseable, or major schema violations"
        },
        "weight": 0.20
    },
    EvalDimensionType.CONSISTENCY: {
        "name": "Internal Consistency",
        "description": "Output is internally consistent, scores match reasoning",
        "what_to_check": [
            "No contradictory statements",
            "Scores/ratings match written reasoning",
            "Severity in narrative matches numeric severity",
            "Logical flow of information",
            "Consistent terminology"
        ],
        "what_NOT_to_check": [
            "Do not compare to external benchmarks",
            "Do not apply your own scoring opinion"
        ],
        "grounding_requirements": [
            "Inconsistencies must be observable in output",
            "Compare different parts of same output"
        ],
        "rubric": {
            5: "Fully consistent, scores perfectly match reasoning",
            4: "Minor inconsistencies that don't affect interpretation",
            3: "Some inconsistencies requiring interpretation",
            2: "Multiple contradictions or score-reasoning mismatches",
            1: "Major internal contradictions, scores contradict reasoning"
        },
        "weight": 0.15
    },
    EvalDimensionType.RULE_ADHERENCE: {
        "name": "Rule Adherence",
        "description": "Output follows all explicit system rules",
        "what_to_check": [
            "Follows all stated constraints",
            "Respects prohibited actions",
            "Uses required methodologies",
            "Adheres to specified boundaries"
        ],
        "what_NOT_to_check": [
            "Do not invent rules not stated",
            "Do not apply general best practices unless specified"
        ],
        "grounding_requirements": [
            "Rules must be explicitly stated in system prompt",
            "Cite specific rule for each violation"
        ],
        "rubric": {
            5: "Perfect rule adherence, all constraints satisfied",
            4: "Minor rule deviations with no impact",
            3: "Some rule violations but core compliance",
            2: "Multiple rule violations",
            1: "Major rule violations or systematic non-compliance"
        },
        "weight": 0.20
    }
}


# ============================================================================
# Core Generation Functions
# ============================================================================

async def analyze_system_prompt_for_eval(
    system_prompt: str,
    use_case: str,
    requirements: str,
    llm_client: EnhancedLLMClient,
    provider: str,
    api_key: str,
    model_name: str
) -> Dict[str, Any]:
    """Deep analysis focused on evaluation needs"""

    programmatic = analyze_prompt(system_prompt)
    analysis_dict = analysis_to_dict(programmatic)

    analysis_prompt = """You are a QA architect designing an evaluation framework. Analyze this system prompt to understand:

1. **WHAT SUCCESS LOOKS LIKE** (Be specific, not generic)
   - Exact content requirements
   - Format requirements
   - Quality thresholds
   - What "good" means in THIS system

2. **FAILURE RISKS** (What could go wrong)
   - Primary risks if system fails
   - Who is impacted
   - Severity of failures

3. **EXPLICIT RULES** (Non-negotiable requirements)
   - Stated constraints
   - Prohibited actions
   - Required behaviors

4. **GROUNDING REQUIREMENTS**
   - What evidence must be cited
   - What can't be assumed
   - Required verbatim elements

5. **DOWNSTREAM CONSUMERS**
   - Who uses this output
   - What decisions depend on it
   - Stakes of incorrect evaluation

Return JSON:
{
    "system_purpose": "One sentence: what this system does",
    "success_definition": {
        "content_requirements": ["Specific requirement 1", "..."],
        "format_requirements": ["Specific requirement 1", "..."],
        "quality_thresholds": ["Specific threshold 1", "..."]
    },
    "failure_risks": {
        "primary_risks": ["Risk 1", "..."],
        "impact": "Who is affected",
        "severity": "high|medium|low"
    },
    "explicit_rules": [
        {"rule": "...", "consequence_of_violation": "..."}
    ],
    "grounding_requirements": {
        "must_cite": ["What must be cited", "..."],
        "cannot_assume": ["What cannot be assumed", "..."],
        "verbatim_required": ["What must be verbatim", "..."]
    },
    "downstream_consumers": {
        "who": "Who uses this",
        "decisions": "What decisions depend on it",
        "stakes": "high|medium|low"
    },
    "known_failure_patterns": [
        {
            "pattern": "Subtle failure",
            "why_common": "Why it happens",
            "how_to_detect": "Detection method"
        }
    ]
}"""

    user_message = f"""Analyze for evaluation framework design:

**USE CASE:** {use_case}
**REQUIREMENTS:** {requirements}

**SYSTEM PROMPT:**
```
{system_prompt}
```

**PRELIMINARY ANALYSIS:**
- Prompt Type: {analysis_dict['prompt_type']}
- Output Format: {analysis_dict['dna']['output_format']}
- Template Variables: {analysis_dict['dna']['template_variables']}
- Scoring Scale: {analysis_dict['dna']['scoring_scale']}"""

    result = await llm_client.chat(
        system_prompt=analysis_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.2,
        max_tokens=4000
    )

    if result.get("error"):
        logger.error(f"Analysis failed: {result['error']}")
        return {"programmatic": analysis_dict, "deep": None, "prompt_type": programmatic.prompt_type}

    deep = parse_json_response(result["output"], "object")
    return {
        "programmatic": analysis_dict,
        "deep": deep or {"raw": result["output"]},
        "prompt_type": programmatic.prompt_type
    }


def build_auto_fail_conditions(
    prompt_type: PromptType,
    analysis: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Build comprehensive auto-fail conditions"""

    # Start with universal auto-fails
    auto_fails = UNIVERSAL_AUTO_FAIL.copy()

    # Add type-specific auto-fails
    type_failures = FAILURE_MODE_TAXONOMY.get(prompt_type, {})
    auto_fails.extend(type_failures.get("auto_fail", []))

    # Add rule-based auto-fails from analysis
    deep = analysis.get("deep", {})
    for rule in deep.get("explicit_rules", []):
        auto_fails.append({
            "id": f"rule_violation_{len(auto_fails)}",
            "name": f"Rule Violation: {rule.get('rule', 'Unknown')[:50]}",
            "description": rule.get("consequence_of_violation", "Violates explicit rule"),
            "detection": f"Check if output violates: {rule.get('rule', '')}",
            "category": "rule_adherence"
        })

    return auto_fails


def build_major_issues(
    prompt_type: PromptType,
    analysis: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Build major (non-auto-fail) issues to check"""

    type_failures = FAILURE_MODE_TAXONOMY.get(prompt_type, {})
    major = type_failures.get("major", []).copy()

    # Add known failure patterns from analysis
    deep = analysis.get("deep", {})
    for pattern in deep.get("known_failure_patterns", []):
        major.append({
            "id": f"pattern_{len(major)}",
            "name": pattern.get("pattern", "Unknown Pattern"),
            "description": pattern.get("why_common", ""),
            "detection": pattern.get("how_to_detect", ""),
            "category": "custom"
        })

    return major


def build_evaluation_dimensions(
    prompt_type: PromptType,
    analysis: Dict[str, Any],
    auto_fails: List[Dict[str, Any]],
    major_issues: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Build weighted evaluation dimensions with full rubrics"""

    # Base dimensions always included
    dimensions = [
        DIMENSION_TEMPLATES[EvalDimensionType.ACCURACY],
        DIMENSION_TEMPLATES[EvalDimensionType.COMPLETENESS],
        DIMENSION_TEMPLATES[EvalDimensionType.RELEVANCE]
    ]

    # Add type-specific dimensions
    if prompt_type in [PromptType.STRUCTURED_OUTPUT, PromptType.EXTRACTION]:
        dimensions.append(ADDITIONAL_DIMENSIONS[EvalDimensionType.FORMAT])

    if prompt_type == PromptType.ANALYTICAL:
        dimensions.append(ADDITIONAL_DIMENSIONS[EvalDimensionType.CONSISTENCY])

    # Add rule adherence if explicit rules exist
    deep = analysis.get("deep", {})
    if deep.get("explicit_rules"):
        dimensions.append(ADDITIONAL_DIMENSIONS[EvalDimensionType.RULE_ADHERENCE])

    # Normalize weights
    total_weight = sum(d["weight"] for d in dimensions)
    for d in dimensions:
        d["weight"] = round(d["weight"] / total_weight, 2)

    return dimensions


async def generate_calibration_examples_v3(
    system_prompt: str,
    dimensions: List[Dict[str, Any]],
    auto_fails: List[Dict[str, Any]],
    major_issues: List[Dict[str, Any]],
    llm_client: EnhancedLLMClient,
    provider: str,
    api_key: str,
    model_name: str
) -> List[Dict[str, Any]]:
    """Generate 4 calibration examples: 5, 4, 3, and 1-2 score"""

    calibration_prompt = """Generate 4 calibration examples for an LLM evaluator:

1. **SCORE 5 (Excellent)**: Perfect output meeting all requirements
2. **SCORE 4 (Good)**: Solid output with minor issues only
3. **SCORE 3 (Acceptable)**: Passable but has notable weaknesses
4. **SCORE 1-2 (Poor)**: Fails due to specific critical issues

For EACH example, provide:
- Realistic input
- Complete output (not truncated)
- Score with reasoning
- Dimension-by-dimension breakdown
- Specific failures present (for low scores)

Return JSON array:
[
    {
        "input": "...",
        "output": "...",
        "score": 5,
        "reasoning": "Evidence-based reasoning citing specific output elements",
        "dimension_scores": {"accuracy": 5, "completeness": 5, ...},
        "failures_present": [],
        "why_this_score": "Calibration note explaining why this deserves exactly this score"
    }
]

IMPORTANT: Make the score=1-2 example fail due to an AUTO-FAIL condition to show how those work."""

    user_message = f"""System prompt:
```
{system_prompt}
```

Dimensions: {json.dumps([{"name": d["name"], "weight": d["weight"]} for d in dimensions])}

Auto-fail conditions: {json.dumps([{"name": af["name"]} for af in auto_fails[:5]])}

Generate 4 calibration examples."""

    result = await llm_client.chat(
        system_prompt=calibration_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.5,
        max_tokens=8000
    )

    if result.get("error"):
        return []

    return parse_json_response(result["output"], "array") or []


def build_gold_standard_eval_prompt(
    system_prompt: str,
    use_case: str,
    analysis: Dict[str, Any],
    auto_fails: List[Dict[str, Any]],
    major_issues: List[Dict[str, Any]],
    dimensions: List[Dict[str, Any]],
    calibration_examples: List[Dict[str, Any]]
) -> str:
    """
    Build the gold-standard evaluation prompt incorporating ALL 20 best practices.
    """

    deep = analysis.get("deep", {})
    programmatic = analysis.get("programmatic", {})

    # Build dimension sections with full rubrics and NOT-to-check
    dimension_sections = []
    for i, dim in enumerate(dimensions, 1):
        rubric_lines = "\n".join([
            f"    - **{score}**: {desc}"
            for score, desc in sorted(dim["rubric"].items(), key=lambda x: -x[0])
        ])

        checks = "\n".join([f"    - {c}" for c in dim["what_to_check"][:5]])
        not_checks = "\n".join([f"    - {c}" for c in dim.get("what_NOT_to_check", [])[:3]])
        grounding = "\n".join([f"    - {g}" for g in dim.get("grounding_requirements", [])[:2]])

        section = f"""### {i}. {dim["name"]} (Weight: {dim["weight"]*100:.0f}%)

**Definition:** {dim["description"]}

**What to Check:**
{checks}

**What NOT to Check (Scope Limits):**
{not_checks}

**Grounding Requirements:**
{grounding}

**Scoring Rubric:**
{rubric_lines}"""
        dimension_sections.append(section)

    # Build calibration section
    calibration_section = ""
    if calibration_examples:
        cal_text = []
        for ex in calibration_examples[:3]:
            cal_text.append(f"""
**Example (Score {ex.get('score', '?')}/5):**
- Input: "{str(ex.get('input', ''))[:80]}..."
- Output: "{str(ex.get('output', ''))[:120]}..."
- Reasoning: {ex.get('reasoning', '')[:150]}
- Why this score: {ex.get('why_this_score', '')}""")

        calibration_section = f"""
## VIII. Calibration Examples

Use these to calibrate your scoring. Similar outputs should receive similar scores.
{''.join(cal_text)}
"""

    # Build the complete gold-standard eval prompt
    eval_prompt = f"""# GOLD-STANDARD EVALUATOR / LLM JUDGE
## For: {use_case}

---

## I. Evaluator Role & Non-Negotiable Stance

You are an **INDEPENDENT EVALUATOR** responsible for judging whether a system's output meets its stated requirements and rules.

**You are NOT allowed to:**
- Re-do the task
- Improve or "fix" the output
- Fill in missing analysis
- Assume intent or unstated reasoning
- Make charitable interpretations

**You ARE required to:**
- Judge ONLY what is explicitly present in the output
- Base ALL judgments on observable evidence
- Apply rubrics consistently across evaluations
- Cite specific output elements in your reasoning

Your evaluation must be **strict, evidence-based, and consistent**.

---

## II. System Context

**What the system does:**
{deep.get('system_purpose', use_case)}

**Primary failure risks:**
{chr(10).join(f"- {r}" for r in deep.get('failure_risks', {}).get('primary_risks', ['Output fails requirements'])[:3])}

**Downstream consumers:** {deep.get('downstream_consumers', {}).get('who', 'System users')}
**Stakes:** {deep.get('downstream_consumers', {}).get('stakes', 'medium')}

**Required output format:** {programmatic.get('dna', {}).get('output_format', 'As specified')}

---

## III. Inputs You Receive

For each evaluation, you are given:

- **{{{{input}}}}**: The exact prompt/data sent to the system
- **{{{{output}}}}**: The system's response that you must evaluate

**CRITICAL:** You must not assume any information outside these two artifacts.

---

## IV. Auto-Fail Conditions (Critical Failures)

If **ANY** of the following are present, the verdict is **FAIL**, regardless of other scores:

{chr(10).join(f"**{i+1}. {af['name']}**" + chr(10) + f"   - {af['description']}" + chr(10) + f"   - Detection: {af['detection']}" for i, af in enumerate(auto_fails[:6]))}

---

## V. Evaluation Dimensions & Weights

Score each dimension independently using the rubrics below.

{chr(10).join(dimension_sections)}

---

## VI. High-Risk Issues (Not Auto-Fail, But Score Impact)

These significantly lower scores if present:

{chr(10).join(f"- **{mi['name']}**: {mi['description']}" for mi in major_issues[:8])}

**Watch for evaluator bias:**
- Do NOT penalize absence of data (penalize only actual behavior)
- Do NOT ignore high-impact failures just because they don't repeat
- Do NOT use vague justifications ("clearly", "obviously") without proof

---

## VII. Evaluation Procedure (Follow Exactly)

1. **Read the input** to understand expectations
2. **Read the entire output** without scoring
3. **Check Auto-Fail conditions** first
4. **Score each dimension** using rubrics (be strict)
5. **Compute weighted average**
6. **Assign verdict** based on rules below
7. **Write concise reasoning** citing specific output elements
{calibration_section}

---

## IX. Verdict Criteria

| Verdict | Criteria |
|---------|----------|
| **PASS** | Score ≥ 4.0 AND no auto-fail issues |
| **NEEDS_REVIEW** | Score 3.0 – 3.99 AND no auto-fail issues |
| **FAIL** | Score < 3.0 OR any auto-fail condition present |

---

## X. Output Format (STRICT)

Return **ONLY** valid JSON:

```json
{{
  "score": <weighted average 1.0-5.0>,
  "verdict": "PASS | NEEDS_REVIEW | FAIL",
  "summary": "2-3 sentence assessment referencing observable issues.",
  "dimension_scores": {{
{chr(10).join(f'    "{dim["name"].lower().replace(" ", "_")}": {{"score": 1, "rationale": "Evidence-based."}},' for dim in dimensions[:-1])}
    "{dimensions[-1]["name"].lower().replace(" ", "_")}": {{"score": 1, "rationale": "Evidence-based."}}
  }},
  "critical_issues": [],
  "strengths": [],
  "weaknesses": [],
  "improvement_suggestions": []
}}
```

---

## XI. Evaluator Guardrails

**You MUST:**
- Judge what IS written, not what SHOULD have been written
- Cite specific elements of the output in your reasoning
- Apply rubrics consistently across evaluations
- Prefer false negatives over false positives (when in doubt, lower score)

**You MUST NOT:**
- Rewrite or complete the output
- Assume missing intent or reasoning
- Add new analysis not present in the output
- Use verbose justifications (concise = trustworthy)

---

## XII. Mandatory Self-Check

Before returning your evaluation, ask yourself:

> "Did I penalize this output for **missing data** — or for a **real violation observable in the output**?"

If it's the former, **revise your judgment**.

> "Does my score-reasoning match? Would another evaluator give the same score for the same output?"

If not, **calibrate against the examples**.

---

## XIII. Consistency Clause

Similar outputs should receive similar scores. If you've evaluated similar outputs before (within this session or based on calibration examples), ensure your scoring is consistent.

---

Now evaluate the output below:

**Input:**
{{{{input}}}}

**Output to Evaluate:**
{{{{output}}}}"""

    return eval_prompt


async def self_validate_eval_prompt(
    eval_prompt: str,
    dimensions: List[Dict[str, Any]],
    auto_fails: List[Dict[str, Any]],
    calibration_examples: List[Dict[str, Any]],
    llm_client: EnhancedLLMClient,
    provider: str,
    api_key: str,
    model_name: str
) -> Dict[str, Any]:
    """Validate the eval prompt against the 20-point checklist"""

    validation_prompt = """Audit this evaluation prompt against best practices:

1. Explicit Role Separation - Does it clearly state evaluator shouldn't re-do task?
2. Clear Success Definition - Is success defined specifically (not generically)?
3. Auto-Fail Conditions - Are non-negotiable failures listed?
4. Evidence-Based Judging - Is observable evidence required?
5. Strict Scope Enforcement - Is there a "what NOT to judge" section?
6. Dimensioned Scoring - Multiple dimensions, not single score?
7. Weighted Dimensions - Are dimensions weighted by importance?
8. Clear Rubrics - Does each dimension have 5/3/1 level descriptions?
9. Grounding Verification - Must claims tie to inputs?
10. Anti-Hallucination Checks - Are hallucination patterns called out?
11. Format Enforcement - Is format treated as first-class failure?
12. Severity Alignment - Is score-narrative consistency checked?
13. False Pos/Neg Awareness - Are bias warnings included?
14. Verdict Thresholds - Are PASS/NEEDS_REVIEW/FAIL clearly defined?
15. Reasoning Without Re-Solving - Is outcome-focused justification required?
16. Consistency Clause - Similar outputs = similar scores?
17. Known Failure Modes - Are system-specific pitfalls listed?
18. Evaluator Self-Check - Is bias self-check mandatory?
19. Downstream Awareness - Is consumer context provided?
20. Minimal Output - Is conciseness enforced?

Return JSON:
{
    "checklist_scores": {
        "1_role_separation": {"present": true/false, "score": 1-5},
        ...
    },
    "overall_score": 0-100,
    "missing_elements": ["Element 1", ...],
    "strengths": ["Strength 1", ...],
    "improvements_needed": ["Improvement 1", ...]
}"""

    result = await llm_client.chat(
        system_prompt=validation_prompt,
        user_message=f"Audit this eval prompt:\n\n{eval_prompt[:4000]}...",
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.2,
        max_tokens=2000
    )

    validation = {
        "dimensions_count": len(dimensions),
        "auto_fails_count": len(auto_fails),
        "calibration_examples_count": len(calibration_examples)
    }

    if not result.get("error"):
        parsed = parse_json_response(result["output"], "object")
        if parsed:
            validation.update(parsed)

    return validation


# ============================================================================
# Main Generation Function
# ============================================================================

async def generate_gold_standard_eval_prompt(
    system_prompt: str,
    use_case: str,
    requirements: str,
    llm_client: EnhancedLLMClient,
    provider: str,
    api_key: str,
    model_name: str,
    thinking_model: Optional[str] = None,
    max_iterations: int = 2
) -> EvalPromptResult:
    """
    Generate a gold-standard evaluation prompt incorporating all 20 best practices.

    This is the premium eval prompt generator that creates evaluators suitable for:
    - Release gating
    - Regression testing
    - QA automation
    - CI/CD pipelines
    """

    analysis_model = thinking_model or model_name
    steps = []

    # Step 1: Deep Analysis for Evaluation
    logger.info(f"Step 1: Analyzing system prompt for evaluation needs")
    analysis = await analyze_system_prompt_for_eval(
        system_prompt, use_case, requirements,
        llm_client, provider, api_key, analysis_model
    )
    steps.append({"step": "analysis", "status": "completed"})

    # Step 2: Build Auto-Fail Conditions
    prompt_type = analysis.get("prompt_type", PromptType.HYBRID)
    auto_fails = build_auto_fail_conditions(prompt_type, analysis)
    steps.append({"step": "auto_fails", "count": len(auto_fails)})

    # Step 3: Build Major Issues
    major_issues = build_major_issues(prompt_type, analysis)
    steps.append({"step": "major_issues", "count": len(major_issues)})

    # Step 4: Build Evaluation Dimensions
    dimensions = build_evaluation_dimensions(prompt_type, analysis, auto_fails, major_issues)
    steps.append({"step": "dimensions", "count": len(dimensions)})

    # Step 5: Generate Calibration Examples
    logger.info(f"Step 5: Generating calibration examples")
    calibration_examples = await generate_calibration_examples_v3(
        system_prompt, dimensions, auto_fails, major_issues,
        llm_client, provider, api_key, model_name
    )
    steps.append({"step": "calibration", "count": len(calibration_examples)})

    # Step 6: Build Gold-Standard Eval Prompt
    logger.info(f"Step 6: Building gold-standard eval prompt")
    eval_prompt = build_gold_standard_eval_prompt(
        system_prompt, use_case, analysis,
        auto_fails, major_issues, dimensions, calibration_examples
    )
    steps.append({"step": "build_prompt", "status": "completed"})

    # Step 7: Self-Validation
    logger.info(f"Step 7: Validating against 20-point checklist")
    validation = await self_validate_eval_prompt(
        eval_prompt, dimensions, auto_fails, calibration_examples,
        llm_client, provider, api_key, model_name
    )
    steps.append({
        "step": "validation",
        "overall_score": validation.get("overall_score", 80)
    })

    # Build rationale
    rationale = (
        f"Generated gold-standard eval prompt with {len(auto_fails)} auto-fail conditions, "
        f"{len(dimensions)} weighted dimensions, and {len(calibration_examples)} calibration examples. "
        f"Validation score: {validation.get('overall_score', 80)}/100. "
    )

    if validation.get("strengths"):
        rationale += f"Strengths: {', '.join(validation['strengths'][:2])}."

    return EvalPromptResult(
        eval_prompt=eval_prompt,
        eval_criteria=[d["name"] for d in dimensions],
        rationale=rationale,
        failure_modes=[
            {"id": af.get("id", ""), "name": af["name"], "description": af["description"],
             "severity": "critical", "category": af.get("category", ""), "is_auto_fail": True}
            for af in auto_fails
        ] + [
            {"id": mi.get("id", ""), "name": mi["name"], "description": mi["description"],
             "severity": "major", "category": mi.get("category", ""), "is_auto_fail": False}
            for mi in major_issues
        ],
        dimensions=[
            {
                "name": d["name"],
                "type": d.get("description", ""),
                "weight": d["weight"],
                "what_to_check": d["what_to_check"],
                "what_NOT_to_check": d.get("what_NOT_to_check", []),
                "grounding_requirements": d.get("grounding_requirements", []),
                "rubric": {str(k): v for k, v in d["rubric"].items()}
            }
            for d in dimensions
        ],
        calibration_examples=calibration_examples,
        self_test_results=validation,
        metadata={
            "steps_taken": steps,
            "prompt_type": str(prompt_type),
            "model_used": model_name,
            "analysis_model": analysis_model,
            "checklist_version": "20-point-v1"
        }
    )


# Alias for backward compatibility
generate_best_eval_prompt_v3 = generate_gold_standard_eval_prompt
