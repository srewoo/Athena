"""
Advanced Evaluator Prompt Generator for Athena (V3) - CANONICAL VERSION

This is the gold-standard eval prompt generator incorporating all 20 best practices.
This is the RECOMMENDED version - V1 (agentic_eval.py) and V2 (eval_generator_v2.py) 
are deprecated.

Best Practices Implemented:
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

Production Metrics:
- Tracks generation latency, LLM calls, and validation outcomes
- Records precision/recall estimates from self-validation
- Provides quality indicators for eval prompt reliability
"""
import re
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from prompt_analyzer import analyze_prompt, analysis_to_dict, PromptType, PromptDNA
from llm_client_v2 import EnhancedLLMClient, parse_json_response

logger = logging.getLogger(__name__)


# ============================================================================
# PRODUCTION METRICS TRACKING
# ============================================================================

@dataclass
class EvalGenerationMetrics:
    """
    Production metrics for eval prompt generation.
    Track these to monitor eval quality over time.
    """
    # Generation metadata
    project_id: Optional[str] = None
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_latency_ms: int = 0
    llm_calls_made: int = 0
    total_tokens_used: int = 0
    
    # Validation outcomes
    self_validation_passed: bool = False
    calibration_examples_count: int = 0
    dimensions_count: int = 0
    auto_fail_conditions_count: int = 0
    
    # Quality indicators
    estimated_precision: float = 0.0  # From self-validation
    estimated_recall: float = 0.0  # From coverage analysis
    best_practices_score: int = 0  # Out of 20
    
    # Issues detected
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "generated_at": self.generated_at,
            "latency_ms": self.generation_latency_ms,
            "llm_calls": self.llm_calls_made,
            "tokens_used": self.total_tokens_used,
            "validation": {
                "passed": self.self_validation_passed,
                "calibration_examples": self.calibration_examples_count,
                "dimensions": self.dimensions_count,
                "auto_fail_conditions": self.auto_fail_conditions_count
            },
            "quality": {
                "estimated_precision": self.estimated_precision,
                "estimated_recall": self.estimated_recall,
                "best_practices_score": self.best_practices_score,
                "best_practices_max": 20
            },
            "warnings": self.warnings
        }


# Global metrics storage (in-memory for now; can be persisted to DB)
_metrics_history: List[EvalGenerationMetrics] = []


def record_eval_metrics(metrics: EvalGenerationMetrics) -> None:
    """Record metrics for monitoring and analysis"""
    _metrics_history.append(metrics)
    # Keep only last 1000 entries in memory
    if len(_metrics_history) > 1000:
        _metrics_history.pop(0)
    logger.info(
        f"Eval generation metrics: latency={metrics.generation_latency_ms}ms, "
        f"llm_calls={metrics.llm_calls_made}, precision={metrics.estimated_precision:.2f}, "
        f"best_practices={metrics.best_practices_score}/20"
    )


def get_eval_metrics_summary() -> Dict[str, Any]:
    """Get summary statistics of recent eval generations"""
    if not _metrics_history:
        return {"count": 0, "message": "No metrics recorded yet"}
    
    recent = _metrics_history[-100:]  # Last 100
    return {
        "count": len(recent),
        "avg_latency_ms": sum(m.generation_latency_ms for m in recent) / len(recent),
        "avg_llm_calls": sum(m.llm_calls_made for m in recent) / len(recent),
        "validation_pass_rate": sum(1 for m in recent if m.self_validation_passed) / len(recent),
        "avg_precision": sum(m.estimated_precision for m in recent) / len(recent),
        "avg_best_practices_score": sum(m.best_practices_score for m in recent) / len(recent),
        "total_generations": len(_metrics_history)
    }

# Import quality enhancements for numeric thresholds
try:
    from eval_prompt_quality import (
        NUMERIC_RUBRIC_THRESHOLDS,
        get_numeric_rubric,
        TIEBREAKER_RULES_COMPACT,
        INTERMEDIATE_SCORE_GUIDANCE,
    )
    QUALITY_THRESHOLDS_AVAILABLE = True
except ImportError:
    QUALITY_THRESHOLDS_AVAILABLE = False
    NUMERIC_RUBRIC_THRESHOLDS = {}
    logger.warning("eval_prompt_quality not available - using basic rubrics")


def extract_template_variables(prompt_text: str) -> List[str]:
    """
    Extract template variables from the prompt.
    Supports both single brace {variable} and double brace {{variable}} formats.
    """
    # Match double braces {{variable}} first (more specific)
    double_brace_pattern = r'\{\{(\w+)\}\}'
    double_matches = re.findall(double_brace_pattern, prompt_text)

    # Match single braces {variable} - but exclude JSON-like patterns
    # This pattern matches {word} but not {"key" or {  or {{
    single_brace_pattern = r'(?<!\{)\{(\w+)\}(?!\})'
    single_matches = re.findall(single_brace_pattern, prompt_text)

    # Combine and deduplicate
    all_matches = list(set(double_matches + single_matches))
    return all_matches


def get_primary_input_variable(system_prompt: str) -> str:
    """
    Get the primary input variable name from the system prompt.
    Returns the extracted variable name or 'input' as fallback.
    """
    variables = extract_template_variables(system_prompt)
    if variables:
        # Prefer variables that look like primary inputs
        input_keywords = ['input', 'transcript', 'text', 'content', 'data', 'query', 'message', 'code', 'email', 'document', 'gaps', 'json']
        for var in variables:
            var_lower = var.lower()
            for keyword in input_keywords:
                if keyword in var_lower:
                    return var
        # If no keyword match, return first variable
        return variables[0]
    return "input"


def get_all_input_variables(system_prompt: str) -> List[str]:
    """
    Get all template variable names from the system prompt.
    Returns list of variable names for use in eval prompt.
    """
    variables = extract_template_variables(system_prompt)
    if not variables:
        return ["input"]
    return variables


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
    },
    PromptType.CREATIVE: {
        "auto_fail": [
            {
                "id": "safety_violation",
                "name": "Safety/Content Policy Violation",
                "description": "Harmful, offensive, or inappropriate content",
                "detection": "Check for policy violations",
                "category": "safety"
            },
            {
                "id": "plagiarism",
                "name": "Plagiarism/Copyright Violation",
                "description": "Directly copies copyrighted material",
                "detection": "Check for verbatim copied content",
                "category": "safety"
            }
        ],
        "major": [
            {
                "id": "constraint_violation",
                "name": "Constraint Violation",
                "description": "Ignores specified constraints (length, style, tone, format)",
                "detection": "Verify compliance with stated constraints",
                "category": "rule_adherence"
            },
            {
                "id": "off_brief",
                "name": "Off Brief",
                "description": "Output doesn't match the creative brief or requirements",
                "detection": "Compare output to requirements",
                "category": "relevance"
            },
            {
                "id": "inconsistent_voice",
                "name": "Inconsistent Voice/Tone",
                "description": "Voice or tone shifts unexpectedly",
                "detection": "Check consistency throughout output",
                "category": "consistency"
            },
            {
                "id": "factual_error",
                "name": "Factual Error in Creative Context",
                "description": "Gets real-world facts wrong when accuracy matters",
                "detection": "Verify factual claims if applicable",
                "category": "accuracy"
            }
        ],
        "minor": [
            {
                "id": "generic_output",
                "name": "Generic/Template-like Output",
                "description": "Output is bland, predictable, or lacks creativity",
                "detection": "Assess originality and engagement",
                "category": "quality"
            }
        ]
    },
    PromptType.INSTRUCTIONAL: {
        "auto_fail": [
            {
                "id": "dangerous_instruction",
                "name": "Dangerous/Harmful Instructions",
                "description": "Provides instructions that could cause harm",
                "detection": "Check for safety of instructions",
                "category": "safety"
            },
            {
                "id": "critical_step_missing",
                "name": "Critical Step Missing",
                "description": "Omits essential step that would cause failure",
                "detection": "Verify all critical steps present",
                "category": "completeness"
            }
        ],
        "major": [
            {
                "id": "incorrect_sequence",
                "name": "Incorrect Sequence",
                "description": "Steps in wrong order causing failure",
                "detection": "Verify logical step sequence",
                "category": "accuracy"
            },
            {
                "id": "unclear_instruction",
                "name": "Unclear Instructions",
                "description": "Steps are ambiguous or confusing",
                "detection": "Assess clarity of each step",
                "category": "clarity"
            },
            {
                "id": "wrong_prerequisites",
                "name": "Wrong/Missing Prerequisites",
                "description": "Incorrect or missing prerequisites",
                "detection": "Verify prerequisites are accurate",
                "category": "completeness"
            },
            {
                "id": "incorrect_values",
                "name": "Incorrect Values/Parameters",
                "description": "Wrong numbers, settings, or parameters",
                "detection": "Verify accuracy of all values",
                "category": "accuracy"
            }
        ],
        "minor": [
            {
                "id": "verbose_instructions",
                "name": "Overly Verbose",
                "description": "Instructions longer than necessary",
                "detection": "Assess conciseness",
                "category": "quality"
            }
        ]
    },
    PromptType.HYBRID: {
        "auto_fail": [
            {
                "id": "format_unparseable",
                "name": "Unparseable Output",
                "description": "Output cannot be parsed as required format",
                "detection": "Attempt to parse output",
                "category": "format"
            },
            {
                "id": "safety_violation",
                "name": "Safety Violation",
                "description": "Harmful or inappropriate content",
                "detection": "Check for safety violations",
                "category": "safety"
            }
        ],
        "major": [
            {
                "id": "requirement_miss",
                "name": "Requirement Miss",
                "description": "Fails to meet stated requirements",
                "detection": "Compare output to requirements",
                "category": "completeness"
            },
            {
                "id": "hallucination",
                "name": "Hallucination",
                "description": "Makes up information not in input",
                "detection": "Verify claims against input",
                "category": "grounding"
            }
        ],
        "minor": [
            {
                "id": "quality_issue",
                "name": "Quality Issue",
                "description": "Output quality below expectations",
                "detection": "Assess overall quality",
                "category": "quality"
            }
        ]
    }
}

# Universal auto-fail conditions (apply to ALL prompt types)
# Note: format_unparseable only applies to structured outputs, not plain text
UNIVERSAL_AUTO_FAIL_ALWAYS = [
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

# Format-related auto-fail - only applies to structured output formats (json, xml, csv)
STRUCTURED_FORMAT_AUTO_FAIL = {
    "id": "format_unparseable",
    "name": "Unparseable Output",
    "description": "Output cannot be parsed as required format (JSON/XML/etc.)",
    "detection": "Attempt to parse output in expected format",
    "category": "format"
}

# Output formats that require parsing/validation
STRUCTURED_OUTPUT_FORMATS = {"json", "xml", "csv"}


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

2. **MANDATORY OUTPUT ELEMENTS** (Extract EVERY required element from the system prompt)
   - Identifiers that MUST appear (e.g., Rep ID, Case ID, Customer ID)
   - Contextual framing that MUST appear (e.g., time period, date range, executive summary)
   - Structural elements that MUST appear (e.g., section headers, specific fields)
   - Grounding elements that MUST appear (e.g., "What Good Looks Like" references, baseline comparisons)

   BE EXHAUSTIVE. If the system prompt says "include X" or "must contain Y" or "always show Z", capture it.

3. **FAILURE RISKS** (What could go wrong)
   - Primary risks if system fails
   - Who is impacted
   - Severity of failures

4. **EXPLICIT RULES** (Non-negotiable requirements)
   - Stated constraints
   - Prohibited actions
   - Required behaviors

5. **GROUNDING REQUIREMENTS**
   - What evidence must be cited
   - What can't be assumed
   - Required verbatim elements

6. **DOWNSTREAM CONSUMERS**
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
    "mandatory_output_elements": {
        "identifiers": [
            {"element": "Rep ID", "description": "Sales rep identifier", "where_required": "header/summary"}
        ],
        "contextual_framing": [
            {"element": "Time period", "description": "Date range of analysis", "where_required": "header"}
        ],
        "structural_elements": [
            {"element": "Executive summary", "description": "High-level overview", "where_required": "beginning"}
        ],
        "grounding_elements": [
            {"element": "WGLL reference", "description": "What Good Looks Like baseline", "where_required": "comparisons"}
        ]
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
        return {"programmatic": analysis_dict, "deep": {}, "prompt_type": programmatic.prompt_type}

    deep = parse_json_response(result["output"], "object")
    return {
        "programmatic": analysis_dict,
        "deep": deep or {},
        "prompt_type": programmatic.prompt_type
    }


def build_auto_fail_conditions(
    prompt_type: PromptType,
    analysis: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Build comprehensive auto-fail conditions based on prompt type and output format"""

    # Start with universal auto-fails that apply to ALL outputs
    auto_fails = [af.copy() for af in UNIVERSAL_AUTO_FAIL_ALWAYS]

    # Get the detected output format from analysis
    # The analysis dict has structure: {"programmatic": {"dna": {"output_format": "..."}}, "deep": {...}}
    programmatic = analysis.get("programmatic", {})
    dna = programmatic.get("dna", {})
    output_format = dna.get("output_format", "plain")
    if output_format is None:
        output_format = "plain"

    # Only add format-parsing auto-fail for structured output formats
    if output_format.lower() in STRUCTURED_OUTPUT_FORMATS:
        auto_fails.append(STRUCTURED_FORMAT_AUTO_FAIL.copy())

    # Add type-specific auto-fails, but filter out format-related ones for plain text
    type_failures = FAILURE_MODE_TAXONOMY.get(prompt_type, {})
    for af in type_failures.get("auto_fail", []):
        # Skip format-related auto-fails for plain text outputs
        if output_format.lower() not in STRUCTURED_OUTPUT_FORMATS:
            if af.get("category") == "format" or af.get("id") in ["format_invalid", "parse_failure", "missing_required_keys"]:
                continue
        auto_fails.append(af.copy())

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

    # Add mandatory output element violations as major issues
    mandatory_elements = deep.get("mandatory_output_elements", {})
    all_elements = []
    for category, elements in mandatory_elements.items():
        if isinstance(elements, list):
            for elem in elements:
                if isinstance(elem, dict):
                    all_elements.append(elem.get("element", ""))
                elif isinstance(elem, str):
                    all_elements.append(elem)

    if all_elements:
        major.append({
            "id": "missing_mandatory_elements",
            "name": "Missing Mandatory Output Elements",
            "description": f"Output is missing required elements: {', '.join(all_elements[:5])}{'...' if len(all_elements) > 5 else ''}",
            "detection": "Check if all mandatory elements (identifiers, contextual framing, grounding elements) are present in output",
            "category": "completeness"
        })

    return major


def build_evaluation_dimensions_static(
    prompt_type: PromptType,
    analysis: Dict[str, Any],
    auto_fails: List[Dict[str, Any]],
    major_issues: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Build weighted evaluation dimensions with full rubrics (static fallback)"""

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

    # Normalize weights and add deterministic IDs
    total_weight = sum(d["weight"] for d in dimensions)
    for i, d in enumerate(dimensions):
        d["weight"] = round(d["weight"] / total_weight, 2)
        # Add deterministic ID for stable identification
        d["id"] = generate_deterministic_dimension_id(d.get("name", f"dim_{i}"), i)
        d["display_name"] = d.get("name", f"Dimension {i+1}")

    return dimensions


async def generate_dynamic_evaluation_dimensions(
    system_prompt: str,
    use_case: str,
    analysis: Dict[str, Any],
    auto_fails: List[Dict[str, Any]],
    major_issues: List[Dict[str, Any]],
    llm_client,
    provider: str,
    api_key: str,
    model_name: str,
    structured_requirements: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Generate domain-specific evaluation dimensions dynamically using LLM.

    This analyzes the system prompt to understand WHAT matters for this specific
    domain and generates appropriate dimension names, descriptions, and rubrics.
    
    Args:
        structured_requirements: Optional structured requirements to incorporate
                               into evaluation dimensions

    For example:
    - Call scoring system → "Diagnostic Signal Quality", "Severity Calibration"
    - Code review system → "Bug Detection Accuracy", "Code Quality Assessment"
    - Customer support → "Resolution Completeness", "Empathy & Tone"
    """

    deep = analysis.get("deep", {})
    programmatic = analysis.get("programmatic", {})

    dimension_prompt = """You are an expert evaluation system designer. Your task is to create
DOMAIN-SPECIFIC evaluation dimensions for judging outputs from a specific system.

DO NOT use generic dimension names like "Accuracy", "Completeness", "Relevance".
Instead, create dimensions that reflect WHAT ACTUALLY MATTERS for this specific system's outputs.

For each dimension, provide:
1. **name**: A domain-specific name that reflects what's being evaluated
   - BAD: "Accuracy" (too generic)
   - GOOD: "Evidence Grounding Quality" or "Diagnostic Signal Precision"
2. **description**: What this dimension evaluates in the context of THIS system
3. **what_to_check**: Specific things to look for (list of 3-5 items)
4. **what_NOT_to_check**: Scope limits to prevent evaluator overreach (list of 2-3 items)
5. **rubric**: Score criteria for 5 (excellent), 3 (acceptable), 1 (poor)
6. **weight**: Relative importance (0.0-1.0, should sum to ~1.0 across all dimensions)

Return a JSON array of 4-6 dimensions:
[
    {
        "name": "Domain-Specific Dimension Name",
        "description": "What this dimension evaluates for THIS system",
        "what_to_check": ["Specific check 1", "Specific check 2", "Specific check 3"],
        "what_NOT_to_check": ["Scope limit 1", "Scope limit 2"],
        "rubric": {
            "5": "What earns excellent - specific to this domain",
            "3": "What earns acceptable - specific to this domain",
            "1": "What earns poor - specific to this domain"
        },
        "weight": 0.25
    }
]

IMPORTANT:
- Dimension names should be UNIQUE to this system's domain
- Rubrics should describe domain-specific quality, not generic LLM quality
- Consider: What makes output from THIS system valuable vs useless?"""

    # Build context about what the system does
    core_function = deep.get("core_function", use_case)
    critical_factors = deep.get("critical_success_factors", [])
    failure_names = [f["name"] for f in auto_fails + major_issues]
    
    # Add structured requirements context
    structured_req_section = ""
    if structured_requirements:
        sr_parts = []
        if structured_requirements.get('must_do'):
            sr_parts.append(f"Must Do: {', '.join(structured_requirements['must_do'])}")
        if structured_requirements.get('must_not_do'):
            sr_parts.append(f"Must NOT Do: {', '.join(structured_requirements['must_not_do'])}")
        if structured_requirements.get('tone'):
            sr_parts.append(f"Required Tone: {structured_requirements['tone']}")
        if structured_requirements.get('success_criteria'):
            sr_parts.append(f"Success Criteria: {', '.join(structured_requirements['success_criteria'])}")
        
        if sr_parts:
            structured_req_section = f"\n\n**Structured Requirements:**\n" + "\n".join(sr_parts)

    user_message = f"""Design evaluation dimensions for this system:

**System Purpose:**
{core_function}

**Use Case:**
{use_case}

**Critical Success Factors (what MUST be right):**
{json.dumps(critical_factors, indent=2) if critical_factors else "Not specified - infer from system prompt"}

**Failure Modes to Catch:**
{json.dumps(failure_names[:10], indent=2)}
{structured_req_section}

**System Prompt:**
```
{system_prompt[:6000]}
```

Create 4-6 domain-specific evaluation dimensions that will effectively judge outputs from this system.
Focus on what makes outputs from THIS system valuable, not generic LLM quality metrics.
{f"IMPORTANT: Ensure dimensions cover all structured requirements (must_do, must_not_do, tone, success_criteria)." if structured_requirements else ""}"""

    result = await llm_client.chat(
        system_prompt=dimension_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.3,
        max_tokens=4000
    )

    if result.get("error"):
        logger.warning(f"Dynamic dimension generation failed: {result['error']}, using static fallback")
        return build_evaluation_dimensions_static(
            analysis.get("prompt_type", PromptType.HYBRID),
            analysis, auto_fails, major_issues
        )

    try:
        dimensions = parse_json_response(result["output"], "array")
        if dimensions and len(dimensions) >= 3:
            # Normalize weights and add deterministic IDs
            total_weight = sum(d.get("weight", 0.2) for d in dimensions)
            for i, d in enumerate(dimensions):
                d["weight"] = round(d.get("weight", 0.2) / total_weight, 2)
                # Convert rubric keys to integers for consistency
                if "rubric" in d:
                    d["rubric"] = {int(k): v for k, v in d["rubric"].items()}
                # Add deterministic ID for stable identification across regenerations
                d["id"] = generate_deterministic_dimension_id(d.get("name", f"dim_{i}"), i)
                # Keep display_name separate from id for UI purposes
                d["display_name"] = d.get("name", f"Dimension {i+1}")

            logger.info(f"Generated {len(dimensions)} domain-specific dimensions: {[d['name'] for d in dimensions]}")
            return dimensions
    except Exception as e:
        logger.warning(f"Failed to parse dynamic dimensions: {e}, using static fallback")

    return build_evaluation_dimensions_static(
        analysis.get("prompt_type", PromptType.HYBRID),
        analysis, auto_fails, major_issues
    )


# Alias for backward compatibility
def build_evaluation_dimensions(
    prompt_type: PromptType,
    analysis: Dict[str, Any],
    auto_fails: List[Dict[str, Any]],
    major_issues: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Synchronous fallback - use generate_dynamic_evaluation_dimensions for async"""
    return build_evaluation_dimensions_static(prompt_type, analysis, auto_fails, major_issues)


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
    """
    Generate 10 calibration examples covering full score range with intermediate scores.

    Includes:
    - Score 5.0: Perfect example
    - Score 4.5: Near-perfect with trivial issue
    - Score 4.0: Good with minor gaps
    - Score 3.5: Acceptable edge case
    - Score 3.0: Borderline acceptable
    - Score 2.5: Borderline unacceptable
    - Score 2.0: Poor quality
    - Score 1.5: Very poor with some redeeming quality
    - Score 1.0: Complete failure
    - Edge case: Auto-fail override (good scores but auto-fail triggered)
    """

    calibration_prompt = """Generate 10 calibration examples for an LLM evaluator covering the FULL score range:

1. **SCORE 5.0 (Perfect)**: Zero issues, exceeds expectations
2. **SCORE 4.5 (Near-perfect)**: One trivial issue that doesn't affect usability
3. **SCORE 4.0 (Good)**: Meets requirements with minor imperfections
4. **SCORE 3.5 (Acceptable)**: Meets core requirements but has noticeable gaps
5. **SCORE 3.0 (Borderline pass)**: Minimum acceptable quality
6. **SCORE 2.5 (Borderline fail)**: Close to acceptable but has blocking issues
7. **SCORE 2.0 (Poor)**: Significant problems across multiple dimensions
8. **SCORE 1.5 (Very poor)**: Mostly fails but has some correct elements
9. **SCORE 1.0 (Failure)**: Completely wrong or triggers auto-fail
10. **EDGE CASE (Auto-fail override)**: Good dimension scores BUT triggers an auto-fail condition

CRITICAL: Use INTERMEDIATE SCORES (4.5, 3.5, 2.5, 1.5) - DO NOT round to whole numbers!

For EACH example, provide:
- Realistic input scenario
- Complete output (not truncated)
- Score with evidence-based reasoning
- Dimension-by-dimension breakdown with percentages
- Specific failures present (for low scores)
- Whether it's an edge case

Return JSON array:
[
    {
        "input": "...",
        "output": "...",
        "score": 5.0,
        "reasoning": "Evidence-based reasoning citing specific output elements",
        "dimension_scores": {"accuracy": {"score": 5, "pct": 98}, "completeness": {"score": 5, "pct": 100}},
        "failures_present": [],
        "why_this_score": "Calibration note explaining why this deserves exactly this score",
        "is_edge_case": false,
        "edge_case_type": ""
    }
]

IMPORTANT:
- Include at least 4 intermediate scores (4.5, 3.5, 2.5, 1.5)
- The auto-fail edge case should show high dimension scores BUT still FAIL due to auto-fail
- Include "pct" (percentage) in dimension_scores to show numeric justification"""

    user_message = f"""System prompt:
```
{system_prompt[:2000]}
```

Dimensions: {json.dumps([{"name": d["name"], "weight": d["weight"]} for d in dimensions])}

Auto-fail conditions: {json.dumps([{"name": af["name"]} for af in auto_fails[:5]])}

Generate 10 calibration examples with intermediate scores (4.5, 3.5, 2.5, 1.5)."""

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
    Uses the actual template variable names from the system prompt (e.g., {{callTranscript}})
    instead of generic {{input}}/{{output}}.
    """

    deep = analysis.get("deep", {})
    programmatic = analysis.get("programmatic", {})

    # Extract ALL input variables from the system prompt
    all_input_vars = get_all_input_variables(system_prompt)
    primary_input_var = get_primary_input_variable(system_prompt)
    output_var_name = "output"  # Response is always 'output'

    logger.info(f"Found template variables: {all_input_vars} (primary: {primary_input_var})")

    # Build dimension sections with full rubrics, NUMERIC THRESHOLDS, and NOT-to-check
    dimension_sections = []
    for i, dim in enumerate(dimensions, 1):
        # Get numeric thresholds for this dimension
        dim_name = dim["name"]
        numeric_rubric = None
        if QUALITY_THRESHOLDS_AVAILABLE:
            numeric_rubric = get_numeric_rubric(dim_name)

        # Build rubric lines with NUMERIC THRESHOLDS injected
        rubric_lines_list = []
        for score, desc in sorted(dim["rubric"].items(), key=lambda x: -x[0]):
            if numeric_rubric and score in numeric_rubric:
                threshold = numeric_rubric[score]
                min_pct = threshold.get("min_pct", 0)
                # Inject numeric threshold into rubric description
                rubric_lines_list.append(f"    - **{score}** [≥{min_pct}%]: {desc}")
            else:
                rubric_lines_list.append(f"    - **{score}**: {desc}")

        rubric_lines = "\n".join(rubric_lines_list)

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

**Scoring Rubric (with numeric thresholds):**
{rubric_lines}"""
        dimension_sections.append(section)

    # Build mandatory output elements section
    mandatory_elements = deep.get("mandatory_output_elements", {})
    mandatory_section = ""

    all_mandatory = []
    for category, elements in mandatory_elements.items():
        if isinstance(elements, list):
            for elem in elements:
                if isinstance(elem, dict):
                    all_mandatory.append({
                        "category": category.replace("_", " ").title(),
                        "element": elem.get("element", ""),
                        "description": elem.get("description", ""),
                        "where": elem.get("where_required", "")
                    })
                elif isinstance(elem, str):
                    all_mandatory.append({
                        "category": category.replace("_", " ").title(),
                        "element": elem,
                        "description": "",
                        "where": ""
                    })

    if all_mandatory:
        mandatory_lines = []
        for item in all_mandatory:
            line = f"- **{item['element']}**"
            if item['description']:
                line += f": {item['description']}"
            if item['where']:
                line += f" *(required in: {item['where']})*"
            mandatory_lines.append(line)

        mandatory_section = f"""
## II.B Mandatory Output Elements (Content Checklist)

The system output **MUST** include the following elements. Missing ANY of these should significantly impact the score:

**Identifiers:**
{chr(10).join([l for l in mandatory_lines if any(x in l.lower() for x in ['id', 'identifier', 'name', 'reference'])] or ['- None specified'])}

**Contextual Framing:**
{chr(10).join([l for l in mandatory_lines if any(x in l.lower() for x in ['time', 'period', 'date', 'range', 'context', 'summary', 'executive', 'overview'])] or ['- None specified'])}

**Grounding Elements:**
{chr(10).join([l for l in mandatory_lines if any(x in l.lower() for x in ['good', 'wgll', 'baseline', 'benchmark', 'ground', 'reference', 'standard'])] or ['- None specified'])}

**Other Required Elements:**
{chr(10).join([l for l in mandatory_lines if not any(x in l.lower() for x in ['id', 'identifier', 'name', 'reference', 'time', 'period', 'date', 'range', 'context', 'summary', 'executive', 'overview', 'good', 'wgll', 'baseline', 'benchmark', 'ground', 'standard'])] or ['- None specified'])}

**Evaluation Rule:** For each missing mandatory element, deduct points from the relevant dimension score. If multiple mandatory elements are missing, this may constitute a major failure.

---
"""

    # Build calibration section with UP TO 8 EXAMPLES including intermediate scores
    calibration_section = ""
    if calibration_examples:
        cal_text = []
        # Include up to 8 examples to cover full score range
        for ex in calibration_examples[:8]:
            score = ex.get('score', '?')
            is_edge = ex.get('is_edge_case', False)
            edge_marker = " [EDGE CASE]" if is_edge else ""

            # Format dimension scores with percentages if available
            dim_scores = ex.get('dimension_scores', {})
            if isinstance(dim_scores, dict):
                dim_parts = []
                for dim_name, dim_data in list(dim_scores.items())[:3]:
                    if isinstance(dim_data, dict):
                        dim_parts.append(f"{dim_name}={dim_data.get('score', '?')} ({dim_data.get('pct', '?')}%)")
                    else:
                        dim_parts.append(f"{dim_name}={dim_data}")
                dim_str = ", ".join(dim_parts) if dim_parts else "N/A"
            else:
                dim_str = str(dim_scores)

            cal_text.append(f"""
**Example (Score {score}/5){edge_marker}:**
- Input: "{str(ex.get('input', ''))[:80]}..."
- Output: "{str(ex.get('output', ''))[:120]}..."
- Dimension Scores: {dim_str}
- Reasoning: {ex.get('reasoning', '')[:150]}
- Why this score: {ex.get('why_this_score', '')}""")

        # Add intermediate score guidance
        intermediate_guidance = ""
        if QUALITY_THRESHOLDS_AVAILABLE:
            intermediate_guidance = """

**IMPORTANT - Use Intermediate Scores:**
- **4.5**: Almost perfect, one trivial issue
- **3.5**: Acceptable but has notable gaps
- **2.5**: Borderline, human should review
- **1.5**: Very poor but not completely wrong

Do NOT round all scores to whole numbers. Use half-point increments when appropriate."""

        calibration_section = f"""
## VIII. Calibration Examples (Including Edge Cases & Intermediate Scores)

Use these to calibrate your scoring. Similar outputs should receive similar scores.
{''.join(cal_text)}
{intermediate_guidance}
"""

    # Build the inputs section with ALL template variables
    input_vars_list = []
    for var in all_input_vars:
        input_vars_list.append(f"- **{{{{{var}}}}}**: Input data for '{var}'")
    input_vars_section = chr(10).join(input_vars_list) if input_vars_list else f"- **{{{{input}}}}**: The input data"

    # Build the final evaluation block with ALL variables
    final_eval_inputs = []
    for var in all_input_vars:
        final_eval_inputs.append(f"**{var}:**\n{{{{{var}}}}}")
    final_eval_block = chr(10).join(final_eval_inputs) if final_eval_inputs else f"**Input:**\n{{{{input}}}}"

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
{mandatory_section}
## III. Inputs You Receive

For each evaluation, you are given:

{input_vars_section}
- **{{{{{output_var_name}}}}}**: The system's response that you must evaluate

**CRITICAL:** You must not assume any information outside these artifacts.

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

## IX. Verdict Criteria & Tiebreaker Rules

| Verdict | Criteria |
|---------|----------|
| **PASS** | Score ≥ 3.5 AND no auto-fail issues |
| **NEEDS_REVIEW** | Score 2.5 – 3.49 AND no auto-fail issues |
| **FAIL** | Score < 2.5 OR any auto-fail condition present |

**Tiebreaker Rules (when dimension scores conflict):**
1. **Auto-fail trumps all** → FAIL immediately, regardless of dimension scores
2. **Accuracy priority** → If accuracy differs >1pt from other dimensions, weight it 2x
3. **Minimum floor** → Lowest dimension can't pull score down >0.5 from weighted avg
4. **Round to 0.5** → Final score rounds to nearest 0.5 (3.7→3.5, 3.8→4.0)

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

{final_eval_block}

**Output to Evaluate ({output_var_name}):**
{{{{{output_var_name}}}}}"""

    return eval_prompt


def generate_deterministic_dimension_id(name: str, index: int) -> str:
    """
    Generate a stable, deterministic ID for a dimension based on its name.
    This ensures consistent IDs across regenerations for the same dimension type.
    """
    import hashlib
    # Normalize the name: lowercase, remove special chars, collapse whitespace
    normalized = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
    # Create a short hash suffix for uniqueness
    hash_suffix = hashlib.md5(name.encode()).hexdigest()[:6]
    return f"dim_{normalized}_{hash_suffix}"


def extract_output_schema(system_prompt: str) -> Optional[Dict[str, Any]]:
    """
    Extract the expected output schema from a system prompt.
    Returns a JSON schema if one can be detected, None otherwise.
    """
    # Look for JSON code blocks with schema-like content
    json_pattern = r'```json\s*(\{[\s\S]*?\})\s*```'
    matches = re.findall(json_pattern, system_prompt)

    for match in matches:
        try:
            parsed = json.loads(match)
            # Check if it looks like a schema or example output
            if isinstance(parsed, dict):
                # Extract keys and their types
                schema = {
                    "type": "object",
                    "required": [],
                    "properties": {}
                }
                for key, value in parsed.items():
                    schema["required"].append(key)
                    if isinstance(value, str):
                        schema["properties"][key] = {"type": "string"}
                    elif isinstance(value, (int, float)):
                        schema["properties"][key] = {"type": "number"}
                    elif isinstance(value, bool):
                        schema["properties"][key] = {"type": "boolean"}
                    elif isinstance(value, list):
                        schema["properties"][key] = {"type": "array"}
                    elif isinstance(value, dict):
                        schema["properties"][key] = {"type": "object"}
                return schema
        except json.JSONDecodeError:
            continue

    return None


def validate_output_against_schema(output: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate an output string against an expected schema.
    Returns validation results with specific errors if any.
    """
    result = {
        "valid": False,
        "parseable": False,
        "errors": [],
        "missing_keys": [],
        "type_mismatches": []
    }

    # Try to parse the output as JSON
    try:
        # Extract JSON from output (may have surrounding text)
        json_match = re.search(r'\{[\s\S]*\}', output)
        if not json_match:
            json_match = re.search(r'\[[\s\S]*\]', output)

        if not json_match:
            result["errors"].append("No JSON object or array found in output")
            return result

        parsed = json.loads(json_match.group())
        result["parseable"] = True

    except json.JSONDecodeError as e:
        result["errors"].append(f"JSON parse error: {str(e)}")
        return result

    # Validate against schema
    if schema.get("type") == "object" and isinstance(parsed, dict):
        # Check required keys
        for key in schema.get("required", []):
            if key not in parsed:
                result["missing_keys"].append(key)

        # Check property types
        for key, prop_schema in schema.get("properties", {}).items():
            if key in parsed:
                expected_type = prop_schema.get("type")
                actual_value = parsed[key]

                if expected_type == "string" and not isinstance(actual_value, str):
                    result["type_mismatches"].append(f"{key}: expected string, got {type(actual_value).__name__}")
                elif expected_type == "number" and not isinstance(actual_value, (int, float)):
                    result["type_mismatches"].append(f"{key}: expected number, got {type(actual_value).__name__}")
                elif expected_type == "boolean" and not isinstance(actual_value, bool):
                    result["type_mismatches"].append(f"{key}: expected boolean, got {type(actual_value).__name__}")
                elif expected_type == "array" and not isinstance(actual_value, list):
                    result["type_mismatches"].append(f"{key}: expected array, got {type(actual_value).__name__}")
                elif expected_type == "object" and not isinstance(actual_value, dict):
                    result["type_mismatches"].append(f"{key}: expected object, got {type(actual_value).__name__}")

    result["valid"] = (
        result["parseable"] and
        len(result["missing_keys"]) == 0 and
        len(result["type_mismatches"]) == 0
    )

    return result


async def run_eval_on_calibration_samples(
    eval_prompt: str,
    calibration_examples: List[Dict[str, Any]],
    llm_client: EnhancedLLMClient,
    provider: str,
    api_key: str,
    model_name: str,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Actually run the eval prompt on calibration examples to verify:
    1. JSON output is parseable
    2. Scores are consistent with expected calibration scores
    3. Required fields are present

    Returns detailed results for each sample.
    """
    results = {
        "samples_tested": 0,
        "json_parse_success": 0,
        "score_consistent": 0,
        "details": [],
        "issues": []
    }

    # Expected eval output schema
    eval_output_schema = {
        "type": "object",
        "required": ["score", "verdict"],
        "properties": {
            "score": {"type": "number"},
            "verdict": {"type": "string"},
            "summary": {"type": "string"},
            "dimension_scores": {"type": "object"}
        }
    }

    for i, example in enumerate(calibration_examples[:4]):  # Test up to 4 examples
        sample_result = {
            "example_index": i,
            "expected_score": example.get("score", 3.0),
            "actual_score": None,
            "json_valid": False,
            "score_diff": None,
            "errors": []
        }

        try:
            # Fill in the eval prompt with the example
            filled_eval = eval_prompt
            example_input = str(example.get("input", ""))[:1000]
            example_output = str(example.get("output", ""))[:2000]

            # Replace common placeholders
            for placeholder in ["{{input}}", "{input}", "{{INPUT}}"]:
                filled_eval = filled_eval.replace(placeholder, example_input)
            for placeholder in ["{{output}}", "{output}", "{{OUTPUT}}"]:
                filled_eval = filled_eval.replace(placeholder, example_output)

            # Run the eval
            eval_result = await llm_client.chat(
                system_prompt=filled_eval,
                user_message="Evaluate the output above and return your assessment as JSON.",
                provider=provider,
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
                max_tokens=2000
            )

            if eval_result.get("error"):
                sample_result["errors"].append(f"LLM error: {eval_result['error']}")
            else:
                eval_output = eval_result.get("output", "")

                # Validate JSON parsing
                validation = validate_output_against_schema(eval_output, eval_output_schema)

                if validation["parseable"]:
                    sample_result["json_valid"] = True
                    results["json_parse_success"] += 1

                    # Extract actual score
                    try:
                        json_match = re.search(r'\{[\s\S]*\}', eval_output)
                        if json_match:
                            parsed = json.loads(json_match.group())
                            actual_score = parsed.get("score")
                            if actual_score is not None:
                                sample_result["actual_score"] = float(actual_score)
                                expected = float(example.get("score", 3.0))
                                sample_result["score_diff"] = abs(actual_score - expected)

                                # Consider consistent if within 1 point
                                if sample_result["score_diff"] <= 1.0:
                                    results["score_consistent"] += 1
                                else:
                                    sample_result["errors"].append(
                                        f"Score inconsistent: expected ~{expected}, got {actual_score}"
                                    )
                    except (json.JSONDecodeError, ValueError) as e:
                        sample_result["errors"].append(f"Score extraction failed: {str(e)}")
                else:
                    sample_result["errors"].extend(validation["errors"])
                    if validation["missing_keys"]:
                        sample_result["errors"].append(f"Missing keys: {validation['missing_keys']}")

        except Exception as e:
            sample_result["errors"].append(f"Exception: {str(e)}")

        results["details"].append(sample_result)
        results["samples_tested"] += 1

    # Summarize issues
    if results["json_parse_success"] < results["samples_tested"]:
        results["issues"].append(
            f"JSON parsing failed for {results['samples_tested'] - results['json_parse_success']}/{results['samples_tested']} samples"
        )
    if results["score_consistent"] < results["samples_tested"]:
        results["issues"].append(
            f"Score inconsistency in {results['samples_tested'] - results['score_consistent']}/{results['samples_tested']} samples"
        )

    results["success_rate"] = (
        (results["json_parse_success"] + results["score_consistent"]) /
        (results["samples_tested"] * 2) * 100
        if results["samples_tested"] > 0 else 0
    )

    return results


async def check_judge_reliability(
    eval_prompt: str,
    test_sample: Dict[str, Any],
    llm_client: EnhancedLLMClient,
    provider: str,
    api_key: str,
    model_name: str,
    num_runs: int = 3
) -> Dict[str, Any]:
    """
    Check judge reliability by running the same eval multiple times with temp=0.
    High variance indicates an unreliable eval prompt.
    """
    scores = []
    verdicts = []

    sample_input = str(test_sample.get("input", ""))[:1000]
    sample_output = str(test_sample.get("output", ""))[:2000]

    filled_eval = eval_prompt
    for placeholder in ["{{input}}", "{input}", "{{INPUT}}"]:
        filled_eval = filled_eval.replace(placeholder, sample_input)
    for placeholder in ["{{output}}", "{output}", "{{OUTPUT}}"]:
        filled_eval = filled_eval.replace(placeholder, sample_output)

    for run in range(num_runs):
        try:
            result = await llm_client.chat(
                system_prompt=filled_eval,
                user_message="Evaluate the output above.",
                provider=provider,
                api_key=api_key,
                model_name=model_name,
                temperature=0.0,  # Deterministic
                max_tokens=2000
            )

            if not result.get("error"):
                json_match = re.search(r'\{[\s\S]*\}', result.get("output", ""))
                if json_match:
                    parsed = json.loads(json_match.group())
                    if "score" in parsed:
                        scores.append(float(parsed["score"]))
                    if "verdict" in parsed:
                        verdicts.append(parsed["verdict"])
        except Exception:
            pass

    # Calculate variance
    variance = 0.0
    if len(scores) >= 2:
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)

    # Check verdict consistency
    verdict_consistent = len(set(verdicts)) <= 1 if verdicts else False

    return {
        "num_runs": num_runs,
        "scores_collected": len(scores),
        "scores": scores,
        "score_variance": round(variance, 4),
        "score_range": round(max(scores) - min(scores), 2) if scores else 0,
        "verdicts": verdicts,
        "verdict_consistent": verdict_consistent,
        "reliable": variance < 0.25 and verdict_consistent,
        "issues": [] if (variance < 0.25 and verdict_consistent) else [
            f"High score variance: {variance:.2f}" if variance >= 0.25 else None,
            "Inconsistent verdicts across runs" if not verdict_consistent else None
        ]
    }


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
    """Validate the eval prompt against the 20-point checklist AND run live tests"""

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

    # Run live calibration tests if we have examples
    if calibration_examples:
        logger.info("Running live calibration tests on eval prompt...")
        calibration_test_results = await run_eval_on_calibration_samples(
            eval_prompt, calibration_examples,
            llm_client, provider, api_key, model_name
        )
        validation["live_calibration_tests"] = calibration_test_results

        # Check judge reliability on first calibration example
        if calibration_examples:
            logger.info("Checking judge reliability (variance check)...")
            reliability = await check_judge_reliability(
                eval_prompt, calibration_examples[0],
                llm_client, provider, api_key, model_name,
                num_runs=2  # Quick check with 2 runs
            )
            validation["judge_reliability"] = reliability

        # Adjust overall score based on live test results
        base_score = validation.get("overall_score", 80)
        if calibration_test_results.get("success_rate", 100) < 75:
            validation["overall_score"] = max(50, base_score - 20)
            validation["live_test_penalty"] = "Score reduced due to calibration test failures"
        if not reliability.get("reliable", True):
            validation["overall_score"] = max(50, validation.get("overall_score", base_score) - 10)
            validation["reliability_penalty"] = "Score reduced due to judge inconsistency"

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
    max_iterations: int = 2,
    project_id: Optional[str] = None,
    structured_requirements: Optional[Dict[str, Any]] = None,
    run_bias_detection: bool = False,
    test_cases_for_bias: Optional[List[Dict[str, str]]] = None
) -> EvalPromptResult:
    """
    Generate a gold-standard evaluation prompt incorporating all 20 best practices.

    This is the premium eval prompt generator that creates evaluators suitable for:
    - Release gating
    - Regression testing
    - QA automation
    - CI/CD pipelines
    
    Args:
        structured_requirements: Optional structured requirements with must_do, must_not_do, 
                                tone, output_format, success_criteria, edge_cases
    
    Production Metrics:
    - Tracks generation latency, LLM calls, validation outcomes
    - Records estimated precision/recall from self-validation
    - Logs quality indicators for monitoring
    """
    # Start metrics tracking
    start_time = time.time()
    metrics = EvalGenerationMetrics(project_id=project_id)

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
    
    # Enhance auto-fails with structured requirements must_not_do items
    if structured_requirements and structured_requirements.get('must_not_do'):
        for forbidden_item in structured_requirements['must_not_do']:
            auto_fails.append(f"Output violates requirement: Must NOT {forbidden_item}")
    
    steps.append({"step": "auto_fails", "count": len(auto_fails)})

    # Step 3: Build Major Issues
    major_issues = build_major_issues(prompt_type, analysis)
    
    # Add major issues from structured requirements
    if structured_requirements:
        if structured_requirements.get('must_do'):
            major_issues.append(f"Missing required behaviors: {', '.join(structured_requirements['must_do'][:3])}")
        if structured_requirements.get('edge_cases'):
            major_issues.append(f"Fails to handle edge cases: {', '.join(structured_requirements['edge_cases'][:2])}")
    
    steps.append({"step": "major_issues", "count": len(major_issues)})

    # Step 4: Generate Domain-Specific Evaluation Dimensions (Dynamic)
    logger.info(f"Step 4: Generating domain-specific evaluation dimensions")
    dimensions = await generate_dynamic_evaluation_dimensions(
        system_prompt, use_case, analysis, auto_fails, major_issues,
        llm_client, provider, api_key, model_name,
        structured_requirements=structured_requirements
    )
    steps.append({"step": "dimensions", "count": len(dimensions), "dynamic": True})

    # Step 5: Generate Calibration Examples
    logger.info(f"Step 5: Generating calibration examples")
    calibration_examples = await generate_calibration_examples_v3(
        system_prompt, dimensions, auto_fails, major_issues,
        llm_client, provider, api_key, model_name
    )
    steps.append({"step": "calibration", "count": len(calibration_examples)})

    # Step 6: Build Gold-Standard Eval Prompt with Iteration
    iteration = 0
    eval_prompt = None
    validation = None

    while iteration < max_iterations:
        iteration += 1
        logger.info(f"Step 6: Building gold-standard eval prompt (iteration {iteration}/{max_iterations})")

        eval_prompt = build_gold_standard_eval_prompt(
            system_prompt, use_case, analysis,
            auto_fails, major_issues, dimensions, calibration_examples
        )
        steps.append({"step": f"build_prompt_iter_{iteration}", "status": "completed"})

        # Step 7: Self-Validation (includes live calibration tests)
        logger.info(f"Step 7: Validating against 20-point checklist (iteration {iteration})")
        validation = await self_validate_eval_prompt(
            eval_prompt, dimensions, auto_fails, calibration_examples,
            llm_client, provider, api_key, model_name
        )

        # Check if calibration tests passed
        live_tests = validation.get("live_calibration_tests", {})
        success_rate = live_tests.get("success_rate", 100)
        reliability = validation.get("judge_reliability", {})
        is_reliable = reliability.get("reliable", True)

        if success_rate >= 75 and is_reliable:
            logger.info(f"Eval prompt passed validation (success_rate={success_rate}%, reliable={is_reliable})")
            steps.append({
                "step": f"validation_iter_{iteration}",
                "overall_score": validation.get("overall_score", 80),
                "passed": True
            })
            break
        elif iteration < max_iterations:
            logger.warning(f"Eval prompt validation issues (success_rate={success_rate}%, reliable={is_reliable}), iterating...")
            steps.append({
                "step": f"validation_iter_{iteration}",
                "overall_score": validation.get("overall_score", 80),
                "passed": False,
                "reason": live_tests.get("issues", []) + reliability.get("issues", [])
            })

            # Try to improve dimensions for next iteration
            # Regenerate dimensions with feedback from validation issues
            logger.info("Regenerating dimensions based on validation feedback...")
            dimensions = await generate_dynamic_evaluation_dimensions(
                system_prompt, use_case, analysis, auto_fails, major_issues,
                llm_client, provider, api_key, model_name
            )
        else:
            logger.warning(f"Max iterations reached, using best available eval prompt")
            steps.append({
                "step": f"validation_iter_{iteration}",
                "overall_score": validation.get("overall_score", 80),
                "passed": False,
                "reason": "Max iterations reached"
            })

    # Build rationale
    live_tests = validation.get("live_calibration_tests", {}) if validation else {}
    reliability = validation.get("judge_reliability", {}) if validation else {}

    rationale = (
        f"Generated gold-standard eval prompt with {len(auto_fails)} auto-fail conditions, "
        f"{len(dimensions)} weighted dimensions, and {len(calibration_examples)} calibration examples. "
        f"Validation score: {validation.get('overall_score', 80) if validation else 'N/A'}/100. "
        f"Iterations: {iteration}/{max_iterations}. "
    )

    if live_tests.get("success_rate"):
        rationale += f"Live calibration success: {live_tests['success_rate']:.0f}%. "

    if reliability.get("reliable") is not None:
        rationale += f"Judge reliable: {'Yes' if reliability['reliable'] else 'No'}. "

    if validation and validation.get("strengths"):
        rationale += f"Strengths: {', '.join(validation['strengths'][:2])}."

    # ========================================================================
    # RECORD PRODUCTION METRICS
    # ========================================================================
    metrics.generation_latency_ms = int((time.time() - start_time) * 1000)
    metrics.dimensions_count = len(dimensions)
    metrics.auto_fail_conditions_count = len(auto_fails)
    metrics.calibration_examples_count = len(calibration_examples)
    metrics.self_validation_passed = live_tests.get("success_rate", 0) >= 75 and reliability.get("reliable", False)
    
    # Estimate precision/recall from validation results
    if live_tests.get("success_rate") is not None:
        metrics.estimated_precision = live_tests["success_rate"] / 100.0
    if validation and validation.get("checklist_passed"):
        metrics.estimated_recall = sum(1 for v in validation["checklist_passed"].values() if v) / 20.0
    
    # Calculate best practices score
    if validation and validation.get("checklist_passed"):
        metrics.best_practices_score = sum(1 for v in validation["checklist_passed"].values() if v)
    else:
        # Estimate from structure
        bp_score = 0
        if auto_fails: bp_score += 2  # Auto-fail conditions
        if dimensions: bp_score += 3  # Dimensions with rubrics
        if calibration_examples: bp_score += 2  # Calibration examples
        if any(d.get("what_NOT_to_check") for d in dimensions): bp_score += 1  # Scope enforcement
        if any(d.get("grounding_requirements") for d in dimensions): bp_score += 1  # Grounding
        bp_score += min(5, len(dimensions))  # Up to 5 for dimension count
        metrics.best_practices_score = bp_score
    
    # Collect warnings
    if not reliability.get("reliable"):
        metrics.warnings.append("Judge reliability check failed")
    if live_tests.get("success_rate", 100) < 75:
        metrics.warnings.append(f"Low calibration success rate: {live_tests.get('success_rate', 0):.0f}%")
    if len(dimensions) < 3:
        metrics.warnings.append("Low dimension count (<3)")
    
    # Record metrics
    record_eval_metrics(metrics)
    
    # Optional: Run bias detection if requested
    bias_report = None
    if run_bias_detection and test_cases_for_bias and len(test_cases_for_bias) >= 3:
        try:
            logger.info("Running bias detection on generated eval prompt...")
            from eval_bias_detector import check_eval_bias
            
            # Create eval function for bias detection
            async def run_eval_for_bias(prompt, input_text, output_text):
                result = await llm_client.chat(
                    system_prompt=prompt.replace('{{input}}', input_text).replace('{{output}}', output_text),
                    user_message="Evaluate and return JSON.",
                    provider=provider,
                    api_key=api_key,
                    model_name=model_name,
                    temperature=0.0,
                    max_tokens=2000
                )
                
                # Parse score from result
                import json as json_lib
                try:
                    json_match = re.search(r'\{[\s\S]*\}', result.get('output', ''))
                    if json_match:
                        parsed = json_lib.loads(json_match.group())
                        return float(parsed.get('score', 3.0)), parsed.get('verdict', 'UNKNOWN')
                except:
                    pass
                return 3.0, 'UNKNOWN'
            
            bias_report_obj = await check_eval_bias(
                eval_prompt=eval_prompt,
                run_eval_func=run_eval_for_bias,
                test_cases=test_cases_for_bias,
                baseline_scores=None
            )
            
            from eval_bias_detector import bias_report_to_dict
            bias_report = bias_report_to_dict(bias_report_obj)
            
            logger.info(f"Bias detection complete: Score={bias_report_obj.overall_bias_score}, Biased={bias_report_obj.is_biased}")
            
        except Exception as e:
            logger.warning(f"Bias detection failed: {e}")
            bias_report = {"error": str(e), "overall_bias_score": 0, "is_biased": False}
    
    result = EvalPromptResult(
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
                "id": d.get("id", f"dim_{i}"),  # Deterministic ID for stable tracking
                "name": d["name"],
                "display_name": d.get("display_name", d["name"]),  # UI-friendly name
                "type": d.get("description", ""),
                "weight": d["weight"],
                "what_to_check": d["what_to_check"],
                "what_NOT_to_check": d.get("what_NOT_to_check", []),
                "grounding_requirements": d.get("grounding_requirements", []),
                "rubric": {str(k): v for k, v in d["rubric"].items()}
            }
            for i, d in enumerate(dimensions)
        ],
        calibration_examples=calibration_examples,
        self_test_results=validation,
        metadata={
            "steps_taken": steps,
            "prompt_type": str(prompt_type),
            "model_used": model_name,
            "analysis_model": analysis_model,
            "checklist_version": "20-point-v1",
            "iterations_used": iteration,
            "max_iterations": max_iterations,
            "live_calibration_success_rate": live_tests.get("success_rate"),
            "judge_reliable": reliability.get("reliable"),
            "generation_metrics": metrics.to_dict(),  # Include metrics in response
            "bias_report": bias_report  # Include bias detection results
        }
    )
    
    return result


# Alias for backward compatibility
generate_best_eval_prompt_v3 = generate_gold_standard_eval_prompt
