"""
Advanced Evaluator Prompt Generator for Athena (V2)

This is the core feature - generates best-in-class evaluation prompts based on system prompts.

Key improvements:
1. Deep semantic analysis of system prompt intent
2. Comprehensive failure mode taxonomy
3. Multi-dimensional evaluation with weighted scoring
4. Few-shot calibration examples generation
5. Self-validation with synthetic test cases
6. Iterative refinement loop
7. Domain-specific evaluation patterns
8. Evaluation consistency checks
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


def extract_template_variables(prompt_text: str) -> List[str]:
    """Extract template variables like {{callTranscript}} from the prompt"""
    pattern = r'\{\{(\w+)\}\}'
    matches = re.findall(pattern, prompt_text)
    return list(set(matches))  # Remove duplicates


def get_primary_input_variable(system_prompt: str) -> str:
    """
    Get the primary input variable name from the system prompt.
    Returns the extracted variable name or 'input' as fallback.
    """
    variables = extract_template_variables(system_prompt)
    if variables:
        # Prefer variables that look like inputs
        input_keywords = ['input', 'transcript', 'text', 'content', 'data', 'query', 'message', 'code', 'email', 'document']
        for var in variables:
            var_lower = var.lower()
            for keyword in input_keywords:
                if keyword in var_lower:
                    return var
        return variables[0]
    return "input"


# ============================================================================
# Data Models
# ============================================================================

class FailureSeverity(Enum):
    CRITICAL = "critical"  # Auto-fail, unusable output
    MAJOR = "major"        # Significantly impacts quality
    MINOR = "minor"        # Imperfect but acceptable


class EvalDimensionType(Enum):
    ACCURACY = "accuracy"           # Factual correctness
    COMPLETENESS = "completeness"   # All required elements present
    FORMAT = "format"               # Structure and format compliance
    SAFETY = "safety"               # No harmful content
    RELEVANCE = "relevance"         # Addresses the request
    CONSISTENCY = "consistency"     # Internal consistency
    STYLE = "style"                 # Tone, voice, style
    EFFICIENCY = "efficiency"       # Conciseness, no redundancy


@dataclass
class FailureMode:
    """A specific way the system prompt's output could fail"""
    id: str
    name: str
    description: str
    severity: FailureSeverity
    example_bad_output: str
    detection_criteria: str
    category: str  # Groups related failures
    weight: float = 1.0  # Importance in scoring


@dataclass
class EvalDimension:
    """An evaluation dimension with detailed rubric"""
    id: str
    name: str
    dimension_type: EvalDimensionType
    description: str
    what_to_check: List[str]
    weight: float  # 0-1, contribution to total score
    rubric: Dict[int, str]  # Score -> criteria (1-5)
    failure_modes_covered: List[str]  # IDs of failure modes
    examples: Dict[int, str]  # Score -> example output (optional)


@dataclass
class CalibrationExample:
    """Few-shot example for calibrating evaluator"""
    input: str
    output: str
    score: int  # 1-5
    reasoning: str
    dimension_scores: Dict[str, int]  # dimension_id -> score


@dataclass
class SelfTestCase:
    """Synthetic test case for validating eval criteria"""
    input: str
    output: str
    expected_score_range: Tuple[int, int]  # min, max expected
    expected_failures: List[str]  # failure mode IDs expected to trigger


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
# Failure Mode Taxonomy
# ============================================================================

# Comprehensive failure mode patterns by prompt type
FAILURE_MODE_TAXONOMY = {
    PromptType.ANALYTICAL: {
        "core_failures": [
            {
                "id": "score_reasoning_mismatch",
                "name": "Score-Reasoning Mismatch",
                "description": "Numeric score contradicts the written reasoning",
                "severity": "critical",
                "detection": "Check if reasoning supports the given score level",
                "category": "consistency"
            },
            {
                "id": "missing_evidence",
                "name": "Missing Evidence",
                "description": "Claims or scores given without supporting evidence from input",
                "severity": "critical",
                "detection": "Verify all claims trace back to input data",
                "category": "accuracy"
            },
            {
                "id": "rubric_deviation",
                "name": "Rubric Deviation",
                "description": "Scoring doesn't follow the defined rubric criteria",
                "severity": "major",
                "detection": "Check if score matches rubric level definitions",
                "category": "consistency"
            },
            {
                "id": "selective_analysis",
                "name": "Selective Analysis",
                "description": "Only considers part of the input, ignoring relevant data",
                "severity": "major",
                "detection": "Verify all relevant input sections are addressed",
                "category": "completeness"
            },
            {
                "id": "score_inflation",
                "name": "Score Inflation/Deflation",
                "description": "Systematically scores too high or too low",
                "severity": "major",
                "detection": "Compare reasoning severity to score given",
                "category": "accuracy"
            }
        ],
        "edge_failures": [
            {
                "id": "ambiguous_handling",
                "name": "Poor Ambiguity Handling",
                "description": "Fails to acknowledge or handle ambiguous cases",
                "severity": "minor",
                "detection": "Check if ambiguity is noted when present",
                "category": "completeness"
            }
        ]
    },
    PromptType.STRUCTURED_OUTPUT: {
        "core_failures": [
            {
                "id": "schema_violation",
                "name": "Schema Violation",
                "description": "Output doesn't match required JSON/XML schema",
                "severity": "critical",
                "detection": "Validate output against expected schema",
                "category": "format"
            },
            {
                "id": "missing_required_fields",
                "name": "Missing Required Fields",
                "description": "Required fields in schema are absent",
                "severity": "critical",
                "detection": "Check presence of all required fields",
                "category": "completeness"
            },
            {
                "id": "wrong_data_types",
                "name": "Wrong Data Types",
                "description": "Field values have incorrect data types",
                "severity": "major",
                "detection": "Verify data types match schema",
                "category": "format"
            },
            {
                "id": "parse_failure",
                "name": "Parse Failure",
                "description": "Output is malformed and cannot be parsed",
                "severity": "critical",
                "detection": "Attempt to parse output format",
                "category": "format"
            },
            {
                "id": "extra_content",
                "name": "Extra Content",
                "description": "Includes text outside the required format",
                "severity": "minor",
                "detection": "Check for extraneous text around structure",
                "category": "format"
            }
        ],
        "edge_failures": [
            {
                "id": "empty_values",
                "name": "Empty/Null Values",
                "description": "Fields present but with empty or null values",
                "severity": "minor",
                "detection": "Check for meaningful content in fields",
                "category": "completeness"
            }
        ]
    },
    PromptType.CONVERSATIONAL: {
        "core_failures": [
            {
                "id": "off_topic",
                "name": "Off-Topic Response",
                "description": "Response doesn't address the user's query",
                "severity": "critical",
                "detection": "Check if response relates to query",
                "category": "relevance"
            },
            {
                "id": "tone_mismatch",
                "name": "Tone Mismatch",
                "description": "Response tone doesn't match expected style",
                "severity": "major",
                "detection": "Evaluate tone against requirements",
                "category": "style"
            },
            {
                "id": "safety_violation",
                "name": "Safety Violation",
                "description": "Response contains harmful or inappropriate content",
                "severity": "critical",
                "detection": "Check for harmful, biased, or inappropriate content",
                "category": "safety"
            },
            {
                "id": "hallucination",
                "name": "Hallucination",
                "description": "Makes up information not in knowledge base",
                "severity": "major",
                "detection": "Verify claims against known facts",
                "category": "accuracy"
            },
            {
                "id": "scope_violation",
                "name": "Out of Scope",
                "description": "Answers questions outside defined scope",
                "severity": "major",
                "detection": "Check if topic is within defined boundaries",
                "category": "relevance"
            }
        ],
        "edge_failures": [
            {
                "id": "robotic_response",
                "name": "Robotic Response",
                "description": "Response feels mechanical, not natural",
                "severity": "minor",
                "detection": "Assess naturalness of language",
                "category": "style"
            }
        ]
    },
    PromptType.EXTRACTION: {
        "core_failures": [
            {
                "id": "missed_extraction",
                "name": "Missed Information",
                "description": "Fails to extract relevant information present in input",
                "severity": "critical",
                "detection": "Compare extracted data to source",
                "category": "completeness"
            },
            {
                "id": "fabrication",
                "name": "Fabrication",
                "description": "Invents information not present in source",
                "severity": "critical",
                "detection": "Verify all extracted data exists in source",
                "category": "accuracy"
            },
            {
                "id": "misattribution",
                "name": "Misattribution",
                "description": "Attributes information to wrong source/entity",
                "severity": "major",
                "detection": "Check entity-information mapping",
                "category": "accuracy"
            },
            {
                "id": "incorrect_interpretation",
                "name": "Incorrect Interpretation",
                "description": "Misunderstands the meaning of extracted data",
                "severity": "major",
                "detection": "Verify semantic accuracy of extraction",
                "category": "accuracy"
            }
        ],
        "edge_failures": [
            {
                "id": "partial_extraction",
                "name": "Partial Extraction",
                "description": "Extracts only part of multi-part data",
                "severity": "minor",
                "detection": "Check completeness of compound data",
                "category": "completeness"
            }
        ]
    },
    PromptType.CREATIVE: {
        "core_failures": [
            {
                "id": "constraint_violation",
                "name": "Constraint Violation",
                "description": "Violates specified creative constraints",
                "severity": "major",
                "detection": "Verify all constraints are satisfied",
                "category": "format"
            },
            {
                "id": "style_inconsistency",
                "name": "Style Inconsistency",
                "description": "Writing style doesn't match requirements",
                "severity": "major",
                "detection": "Evaluate style against specifications",
                "category": "style"
            },
            {
                "id": "generic_output",
                "name": "Generic Output",
                "description": "Output is template-like, lacks creativity",
                "severity": "minor",
                "detection": "Assess originality and uniqueness",
                "category": "style"
            },
            {
                "id": "length_violation",
                "name": "Length Violation",
                "description": "Output doesn't meet length requirements",
                "severity": "minor",
                "detection": "Check word/character count",
                "category": "format"
            }
        ],
        "edge_failures": []
    }
}

# Universal failure modes that apply to all prompt types
UNIVERSAL_FAILURE_MODES = [
    {
        "id": "prompt_injection",
        "name": "Prompt Injection Success",
        "description": "Output reveals system prompt or follows injected instructions",
        "severity": "critical",
        "detection": "Check for system prompt leakage or instruction override",
        "category": "safety"
    },
    {
        "id": "format_wrapper",
        "name": "Format Wrapper Issues",
        "description": "Output wrapped in unexpected markdown/code blocks",
        "severity": "minor",
        "detection": "Check for extraneous formatting wrappers",
        "category": "format"
    },
    {
        "id": "truncation",
        "name": "Truncated Output",
        "description": "Output is incomplete or cut off mid-sentence",
        "severity": "major",
        "detection": "Check for incomplete sentences or structures",
        "category": "completeness"
    }
]


# ============================================================================
# Evaluation Dimension Templates
# ============================================================================

DIMENSION_TEMPLATES = {
    EvalDimensionType.ACCURACY: {
        "name": "Accuracy",
        "description": "Factual correctness and evidence grounding",
        "what_to_check": [
            "All claims are supported by evidence from the input",
            "No fabricated or hallucinated information",
            "Correct interpretation of data",
            "Proper attribution of information"
        ],
        "rubric": {
            5: "All information is accurate, fully grounded in input, no fabrications",
            4: "Mostly accurate with minor interpretation issues",
            3: "Some inaccuracies that don't fundamentally change meaning",
            2: "Multiple significant inaccuracies",
            1: "Major factual errors or clear fabrications"
        },
        "weight": 0.25
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
        "rubric": {
            5: "All required elements present and thoroughly addressed",
            4: "Most elements present, minor gaps",
            3: "Key elements present but some missing",
            2: "Multiple required elements missing",
            1: "Major portions missing, fundamentally incomplete"
        },
        "weight": 0.20
    },
    EvalDimensionType.FORMAT: {
        "name": "Format Compliance",
        "description": "Output matches required structure and format",
        "what_to_check": [
            "Correct output format (JSON, XML, etc.)",
            "Valid syntax that can be parsed",
            "Required fields in correct locations",
            "No extraneous content outside format"
        ],
        "rubric": {
            5: "Perfect format compliance, valid and parseable",
            4: "Minor formatting issues that don't affect parsing",
            3: "Some format deviations but core structure correct",
            2: "Significant format issues affecting usability",
            1: "Wrong format or unparseable output"
        },
        "weight": 0.20
    },
    EvalDimensionType.RELEVANCE: {
        "name": "Relevance",
        "description": "Output directly addresses the request",
        "what_to_check": [
            "Response relates to the input query",
            "Stays within defined scope",
            "Addresses the actual ask, not tangential topics",
            "Appropriate level of detail for the request"
        ],
        "rubric": {
            5: "Directly and fully addresses the request",
            4: "Addresses the request with minor tangents",
            3: "Partially addresses request, some off-topic content",
            2: "Loosely related but misses main point",
            1: "Completely off-topic or irrelevant"
        },
        "weight": 0.20
    },
    EvalDimensionType.CONSISTENCY: {
        "name": "Internal Consistency",
        "description": "Output is internally consistent and coherent",
        "what_to_check": [
            "No contradictory statements",
            "Scores match reasoning (for analytical)",
            "Logical flow of information",
            "Consistent terminology and style"
        ],
        "rubric": {
            5: "Fully consistent, no contradictions",
            4: "Minor inconsistencies that don't affect meaning",
            3: "Some inconsistencies requiring interpretation",
            2: "Multiple contradictions affecting quality",
            1: "Major internal contradictions"
        },
        "weight": 0.15
    }
}


# ============================================================================
# Core Generation Functions
# ============================================================================

async def analyze_system_prompt_deeply(
    system_prompt: str,
    use_case: str,
    requirements: str,
    llm_client: EnhancedLLMClient,
    provider: str,
    api_key: str,
    model_name: str
) -> Dict[str, Any]:
    """
    Perform deep semantic analysis of the system prompt to understand:
    - What it's trying to achieve
    - What success looks like
    - What could go wrong
    - Domain-specific considerations
    """
    # Get programmatic analysis
    programmatic = analyze_prompt(system_prompt)
    analysis_dict = analysis_to_dict(programmatic)

    analysis_prompt = """You are an expert QA engineer and prompt analyst. Perform a DEEP analysis of this system prompt.

Your analysis must cover:

1. **CORE INTENT**: What is this prompt designed to accomplish? What is the primary user goal?

2. **SUCCESS CRITERIA**: What does a PERFECT output look like? Be specific about:
   - Content requirements
   - Format requirements
   - Quality standards
   - Edge case handling

3. **CRITICAL FAILURE POINTS**: What are the MOST LIKELY ways this prompt could fail?
   - List in order of likelihood and severity
   - Be specific about HOW each failure manifests

4. **IMPLICIT REQUIREMENTS**: What requirements are NOT explicitly stated but clearly expected?
   - Industry standards
   - Common sense expectations
   - Safety considerations

5. **DOMAIN CONTEXT**: What domain-specific knowledge is relevant?
   - Terminology
   - Best practices
   - Regulatory considerations

6. **INPUT VARIATIONS**: What types of inputs will this prompt receive?
   - Normal cases
   - Edge cases
   - Adversarial cases

Return a detailed JSON analysis:
{
    "core_intent": {
        "primary_goal": "...",
        "secondary_goals": ["..."],
        "target_user": "..."
    },
    "success_criteria": {
        "content_requirements": ["..."],
        "format_requirements": ["..."],
        "quality_standards": ["..."],
        "edge_case_handling": ["..."]
    },
    "critical_failure_points": [
        {
            "failure": "...",
            "likelihood": "high|medium|low",
            "severity": "critical|major|minor",
            "manifestation": "How it appears in output",
            "detection_hint": "How to detect it"
        }
    ],
    "implicit_requirements": ["..."],
    "domain_context": {
        "domain": "...",
        "terminology": ["..."],
        "best_practices": ["..."],
        "regulatory_considerations": ["..."]
    },
    "input_variations": {
        "normal_cases": ["..."],
        "edge_cases": ["..."],
        "adversarial_cases": ["..."]
    }
}"""

    user_message = f"""Analyze this system prompt deeply:

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
- Scoring Scale: {analysis_dict['dna']['scoring_scale']}
- Detected Sections: {analysis_dict['dna']['sections']}
- Role: {analysis_dict['dna']['role']}

Provide your deep analysis."""

    result = await llm_client.chat(
        system_prompt=analysis_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.2,
        max_tokens=6000
    )

    if result.get("error"):
        logger.error(f"Deep analysis failed: {result['error']}")
        return {
            "programmatic": analysis_dict,
            "deep": None,
            "error": result["error"]
        }

    deep_analysis = parse_json_response(result["output"], "object")

    return {
        "programmatic": analysis_dict,
        "deep": deep_analysis or {"raw": result["output"]},
        "prompt_type": programmatic.prompt_type
    }


async def generate_comprehensive_failure_modes(
    system_prompt: str,
    analysis: Dict[str, Any],
    llm_client: EnhancedLLMClient,
    provider: str,
    api_key: str,
    model_name: str
) -> List[FailureMode]:
    """
    Generate comprehensive list of failure modes by combining:
    1. Type-specific failure taxonomy
    2. Analysis-derived failure points
    3. LLM-generated domain-specific failures
    """
    prompt_type = analysis.get("prompt_type", PromptType.HYBRID)
    deep = analysis.get("deep", {})
    programmatic = analysis.get("programmatic", {})

    # Start with type-specific failures from taxonomy
    type_failures = FAILURE_MODE_TAXONOMY.get(prompt_type, {})
    core_failures = type_failures.get("core_failures", [])
    edge_failures = type_failures.get("edge_failures", [])

    # Add universal failures
    all_failures = UNIVERSAL_FAILURE_MODES + core_failures + edge_failures

    # Add analysis-derived failures
    if deep and "critical_failure_points" in deep:
        for fp in deep.get("critical_failure_points", []):
            failure_id = fp.get("failure", "").lower().replace(" ", "_")[:30]
            all_failures.append({
                "id": f"custom_{failure_id}",
                "name": fp.get("failure", "Unknown Failure"),
                "description": fp.get("manifestation", ""),
                "severity": fp.get("severity", "major"),
                "detection": fp.get("detection_hint", ""),
                "category": "custom"
            })

    # Ask LLM to identify any additional domain-specific failures
    failure_prompt = """You are a QA expert. Given the existing failure modes and the system prompt context, identify any ADDITIONAL failure modes that are specific to this domain/use case.

Focus on failures that are:
1. Specific to this particular use case
2. Not already covered by the existing failure modes
3. Likely to occur in practice

Return a JSON array of additional failure modes (return empty array [] if none needed):
[
    {
        "id": "unique_id",
        "name": "Failure Name",
        "description": "How this failure manifests",
        "severity": "critical|major|minor",
        "detection": "How to detect this failure",
        "category": "accuracy|completeness|format|safety|relevance|consistency|style"
    }
]"""

    user_message = f"""System prompt being evaluated:
```
{system_prompt}
```

Use case: {analysis.get('deep', {}).get('core_intent', {}).get('primary_goal', 'Unknown')}

Existing failure modes already identified:
{json.dumps([f['name'] for f in all_failures], indent=2)}

Domain context: {json.dumps(analysis.get('deep', {}).get('domain_context', {}), indent=2)}

Identify any additional domain-specific failure modes NOT covered above."""

    result = await llm_client.chat(
        system_prompt=failure_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.3,
        max_tokens=2000
    )

    if not result.get("error"):
        additional = parse_json_response(result["output"], "array")
        if additional:
            all_failures.extend(additional)

    # Convert to FailureMode objects
    failure_modes = []
    seen_ids = set()

    for f in all_failures:
        f_id = f.get("id", "").lower().replace(" ", "_")
        if f_id in seen_ids:
            continue
        seen_ids.add(f_id)

        try:
            severity = FailureSeverity(f.get("severity", "major").lower())
        except ValueError:
            severity = FailureSeverity.MAJOR

        failure_modes.append(FailureMode(
            id=f_id,
            name=f.get("name", "Unknown"),
            description=f.get("description", ""),
            severity=severity,
            example_bad_output=f.get("example_bad_output", ""),
            detection_criteria=f.get("detection", ""),
            category=f.get("category", "other"),
            weight=1.5 if severity == FailureSeverity.CRITICAL else (1.0 if severity == FailureSeverity.MAJOR else 0.5)
        ))

    return failure_modes


async def generate_eval_dimensions(
    system_prompt: str,
    failure_modes: List[FailureMode],
    analysis: Dict[str, Any],
    llm_client: EnhancedLLMClient,
    provider: str,
    api_key: str,
    model_name: str
) -> List[EvalDimension]:
    """
    Generate evaluation dimensions that comprehensively cover all failure modes
    with detailed rubrics tailored to the specific prompt
    """
    prompt_type = analysis.get("prompt_type", PromptType.HYBRID)
    deep = analysis.get("deep", {})

    # Group failure modes by category
    failures_by_category = {}
    for fm in failure_modes:
        if fm.category not in failures_by_category:
            failures_by_category[fm.category] = []
        failures_by_category[fm.category].append(fm)

    # Determine which dimension types are most relevant
    relevant_types = {EvalDimensionType.ACCURACY, EvalDimensionType.COMPLETENESS}

    if prompt_type in [PromptType.STRUCTURED_OUTPUT, PromptType.EXTRACTION]:
        relevant_types.add(EvalDimensionType.FORMAT)

    if prompt_type in [PromptType.CONVERSATIONAL, PromptType.CREATIVE]:
        relevant_types.add(EvalDimensionType.STYLE)
        relevant_types.add(EvalDimensionType.RELEVANCE)

    if prompt_type == PromptType.ANALYTICAL:
        relevant_types.add(EvalDimensionType.CONSISTENCY)

    # Always include relevance for understanding
    relevant_types.add(EvalDimensionType.RELEVANCE)

    # Create base dimensions from templates
    dimensions = []
    for dim_type in relevant_types:
        template = DIMENSION_TEMPLATES.get(dim_type)
        if not template:
            continue

        # Find failure modes this dimension should cover
        covered_modes = []
        for fm in failure_modes:
            if fm.category == dim_type.value or _dimension_covers_failure(dim_type, fm):
                covered_modes.append(fm.id)

        dimensions.append(EvalDimension(
            id=dim_type.value,
            name=template["name"],
            dimension_type=dim_type,
            description=template["description"],
            what_to_check=template["what_to_check"].copy(),
            weight=template["weight"],
            rubric=template["rubric"].copy(),
            failure_modes_covered=covered_modes,
            examples={}
        ))

    # Ask LLM to customize dimensions for this specific prompt
    customize_prompt = """You are customizing evaluation dimensions for a specific system prompt.

For each dimension, provide:
1. Customized "what_to_check" items specific to this prompt
2. Customized rubric descriptions that reference the specific requirements
3. Any additional dimension-specific considerations

Return JSON:
{
    "dimension_customizations": [
        {
            "dimension_id": "accuracy",
            "custom_checks": ["Specific check 1", "Specific check 2"],
            "custom_rubric": {
                "5": "Customized excellent criteria",
                "3": "Customized acceptable criteria",
                "1": "Customized poor criteria"
            },
            "weight_adjustment": 0.0  // -0.1 to +0.1 adjustment based on importance
        }
    ],
    "additional_dimension": null  // or a new dimension if needed
}"""

    user_message = f"""System prompt:
```
{system_prompt}
```

Success criteria: {json.dumps(deep.get('success_criteria', {}), indent=2)}

Current dimensions:
{json.dumps([{"id": d.id, "name": d.name, "description": d.description} for d in dimensions], indent=2)}

Failure modes to cover:
{json.dumps([{"id": fm.id, "name": fm.name, "severity": fm.severity.value} for fm in failure_modes], indent=2)}

Customize these dimensions for this specific prompt."""

    result = await llm_client.chat(
        system_prompt=customize_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.3,
        max_tokens=4000
    )

    if not result.get("error"):
        customizations = parse_json_response(result["output"], "object")
        if customizations:
            # Apply customizations
            for custom in customizations.get("dimension_customizations", []):
                dim_id = custom.get("dimension_id")
                for dim in dimensions:
                    if dim.id == dim_id:
                        if custom.get("custom_checks"):
                            dim.what_to_check.extend(custom["custom_checks"])
                        if custom.get("custom_rubric"):
                            for score_str, desc in custom["custom_rubric"].items():
                                try:
                                    score = int(score_str)
                                    dim.rubric[score] = desc
                                except ValueError:
                                    pass
                        if custom.get("weight_adjustment"):
                            dim.weight = max(0.05, min(0.4, dim.weight + custom["weight_adjustment"]))

    # Normalize weights to sum to 1
    total_weight = sum(d.weight for d in dimensions)
    if total_weight > 0:
        for d in dimensions:
            d.weight = d.weight / total_weight

    return dimensions


def _dimension_covers_failure(dim_type: EvalDimensionType, failure: FailureMode) -> bool:
    """Check if a dimension type would cover a failure mode"""
    coverage_map = {
        EvalDimensionType.ACCURACY: ["accuracy", "custom"],
        EvalDimensionType.COMPLETENESS: ["completeness"],
        EvalDimensionType.FORMAT: ["format"],
        EvalDimensionType.SAFETY: ["safety"],
        EvalDimensionType.RELEVANCE: ["relevance"],
        EvalDimensionType.CONSISTENCY: ["consistency"],
        EvalDimensionType.STYLE: ["style"],
    }
    return failure.category in coverage_map.get(dim_type, [])


async def generate_calibration_examples(
    system_prompt: str,
    dimensions: List[EvalDimension],
    failure_modes: List[FailureMode],
    llm_client: EnhancedLLMClient,
    provider: str,
    api_key: str,
    model_name: str
) -> List[CalibrationExample]:
    """
    Generate few-shot calibration examples showing:
    - What a 5-score output looks like
    - What a 3-score output looks like
    - What a 1-score output looks like with specific failures
    """
    calibration_prompt = """You are generating calibration examples for an LLM evaluator.

For this system prompt, generate 4 example input-output pairs with evaluations:

1. **EXCELLENT (Score 5)**: A perfect output that meets all requirements
2. **GOOD (Score 4)**: A solid output with minor issues
3. **ACCEPTABLE (Score 3)**: An output that passes but has notable weaknesses
4. **POOR (Score 1-2)**: An output that fails due to specific issues

For each example, provide:
- A realistic input that the system might receive
- The system's output (what the LLM would generate)
- The score (1-5)
- Detailed reasoning explaining the score
- Dimension-by-dimension breakdown

Return JSON array:
[
    {
        "input": "Example user input/query",
        "output": "Example system output",
        "score": 5,
        "reasoning": "Why this deserves a 5...",
        "dimension_scores": {
            "accuracy": 5,
            "completeness": 5,
            ...
        },
        "failures_present": []
    },
    {
        "input": "...",
        "output": "Output with issues...",
        "score": 2,
        "reasoning": "Why this scores low...",
        "dimension_scores": {...},
        "failures_present": ["failure_id_1", "failure_id_2"]
    }
]"""

    user_message = f"""System prompt:
```
{system_prompt}
```

Evaluation dimensions:
{json.dumps([{"id": d.id, "name": d.name, "weight": d.weight} for d in dimensions], indent=2)}

Failure modes to demonstrate in poor examples:
{json.dumps([{"id": fm.id, "name": fm.name, "severity": fm.severity.value} for fm in failure_modes[:10]], indent=2)}

Generate 4 realistic calibration examples."""

    result = await llm_client.chat(
        system_prompt=calibration_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.5,  # Slightly higher for creativity
        max_tokens=8000
    )

    examples = []
    if not result.get("error"):
        parsed = parse_json_response(result["output"], "array")
        if parsed:
            for ex in parsed:
                examples.append(CalibrationExample(
                    input=ex.get("input", ""),
                    output=ex.get("output", ""),
                    score=ex.get("score", 3),
                    reasoning=ex.get("reasoning", ""),
                    dimension_scores=ex.get("dimension_scores", {})
                ))

    return examples


async def self_validate_eval_criteria(
    eval_prompt: str,
    dimensions: List[EvalDimension],
    failure_modes: List[FailureMode],
    calibration_examples: List[CalibrationExample],
    llm_client: EnhancedLLMClient,
    provider: str,
    api_key: str,
    model_name: str
) -> Dict[str, Any]:
    """
    Self-validate the evaluation criteria by:
    1. Checking coverage of all failure modes
    2. Testing with calibration examples
    3. Checking for ambiguity in criteria
    """
    # Check coverage programmatically
    all_failure_ids = {fm.id for fm in failure_modes}
    covered_ids = set()
    for dim in dimensions:
        covered_ids.update(dim.failure_modes_covered)

    uncovered = all_failure_ids - covered_ids

    # Ask LLM to validate
    validation_prompt = """You are auditing evaluation criteria for quality and completeness.

Check:
1. **Coverage**: Do the dimensions cover all failure modes?
2. **Clarity**: Are the rubric descriptions unambiguous?
3. **Consistency**: Would two evaluators give the same score?
4. **Practicality**: Can these criteria be applied efficiently?

Return JSON:
{
    "coverage_score": 0.95,  // 0-1
    "clarity_score": 0.90,   // 0-1
    "consistency_score": 0.85, // 0-1
    "overall_confidence": 0.90, // 0-1
    "issues_found": [
        {
            "type": "coverage|clarity|consistency|practicality",
            "description": "Issue description",
            "severity": "high|medium|low",
            "suggestion": "How to fix"
        }
    ],
    "improvements_needed": ["Improvement 1", "..."],
    "strengths": ["Strength 1", "..."]
}"""

    user_message = f"""Evaluate these evaluation criteria:

**Eval Prompt:**
```
{eval_prompt[:3000]}...
```

**Dimensions:**
{json.dumps([{"id": d.id, "name": d.name, "weight": round(d.weight, 2)} for d in dimensions], indent=2)}

**Failure Modes (total: {len(failure_modes)}):**
{json.dumps([{"id": fm.id, "name": fm.name} for fm in failure_modes[:15]], indent=2)}

**Uncovered Failure Modes:** {list(uncovered) if uncovered else "None - full coverage"}

**Calibration Examples Available:** {len(calibration_examples)}

Validate these criteria."""

    result = await llm_client.chat(
        system_prompt=validation_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.2,
        max_tokens=2000
    )

    validation = {
        "coverage_score": 1.0 - (len(uncovered) / max(1, len(all_failure_ids))),
        "uncovered_failure_modes": list(uncovered),
        "dimensions_count": len(dimensions),
        "failure_modes_count": len(failure_modes),
        "calibration_examples_count": len(calibration_examples)
    }

    if not result.get("error"):
        parsed = parse_json_response(result["output"], "object")
        if parsed:
            validation.update(parsed)

    return validation


def build_eval_prompt(
    system_prompt: str,
    dimensions: List[EvalDimension],
    failure_modes: List[FailureMode],
    calibration_examples: List[CalibrationExample],
    analysis: Dict[str, Any],
    use_case: str
) -> str:
    """
    Build the final evaluation prompt with all components.
    Uses the actual template variable names from the system prompt (e.g., {{callTranscript}})
    instead of generic {{input}}/{{output}}.
    """
    deep = analysis.get("deep", {})
    programmatic = analysis.get("programmatic", {})

    # Extract the primary input variable from the system prompt
    input_var_name = get_primary_input_variable(system_prompt)
    output_var_name = "output"  # Response is always 'output'

    logger.info(f"Using input variable: {{{{{input_var_name}}}}} (extracted from system prompt)")

    # Group failure modes by severity
    critical_failures = [fm for fm in failure_modes if fm.severity == FailureSeverity.CRITICAL]
    major_failures = [fm for fm in failure_modes if fm.severity == FailureSeverity.MAJOR]

    # Build dimension sections with full rubrics
    dimension_sections = []
    for i, dim in enumerate(dimensions, 1):
        rubric_text = "\n".join([
            f"    - **{score}**: {desc}" for score, desc in sorted(dim.rubric.items(), reverse=True)
        ])

        checks_text = "\n".join([f"    - {check}" for check in dim.what_to_check[:5]])

        section = f"""### Dimension {i}: {dim.name} (Weight: {dim.weight*100:.0f}%)

**Description:** {dim.description}

**What to Check:**
{checks_text}

**Scoring Rubric:**
{rubric_text}"""
        dimension_sections.append(section)

    # Build calibration examples section
    calibration_section = ""
    if calibration_examples:
        examples_text = []
        for ex in calibration_examples[:3]:
            dim_scores = ", ".join([f"{k}: {v}" for k, v in ex.dimension_scores.items()])
            examples_text.append(f"""
**Example (Score {ex.score}/5):**
- Input: "{ex.input[:100]}..."
- Output snippet: "{ex.output[:150]}..."
- Reasoning: {ex.reasoning[:200]}
- Dimension scores: {dim_scores}""")

        calibration_section = f"""
## Calibration Examples

Use these examples to calibrate your scoring:
{''.join(examples_text)}
"""

    # Build the complete eval prompt
    eval_prompt = f"""# Evaluation Prompt for: {use_case}

## I. Evaluator Role & Objective

You are an EXPERT EVALUATOR assessing LLM outputs for quality and correctness.

**Your Goal:** Objectively evaluate whether the output meets requirements, using the detailed rubrics provided. You are NOT re-doing the task - you are JUDGING the quality of the output.

**Key Principles:**
- Base ALL judgments on observable evidence in the output
- Apply the rubrics consistently across all evaluations
- Note both strengths and weaknesses
- Be fair but rigorous

## II. Context: What This System Does

{deep.get('core_intent', {}).get('primary_goal', use_case)}

**Expected Output Format:** {programmatic.get('dna', {}).get('output_format', 'Not specified')}

## III. Information You Will Receive

For each evaluation, you receive:
- **{{{{{input_var_name}}}}}**: The original input/query sent to the system
- **{{{{{output_var_name}}}}}**: The system's response that you must evaluate

## IV. Critical Failure Modes (AUTO-FAIL)

The following issues result in an AUTOMATIC FAIL verdict regardless of other scores:

{chr(10).join(f"- **{fm.name}**: {fm.description}" for fm in critical_failures) if critical_failures else "- No critical failure modes defined - use dimension scores only"}

## V. Evaluation Dimensions

Evaluate the output on each dimension below. Each dimension contributes to the final score based on its weight.

{chr(10).join(dimension_sections)}

## VI. Major Issues to Watch For

These issues significantly impact quality (but are not auto-fail):

{chr(10).join(f"- **{fm.name}**: {fm.description}" for fm in major_failures[:8]) if major_failures else "- No major issues defined"}
{calibration_section}

## VII. Evaluation Process

Follow these steps EXACTLY:

1. **Read the input** to understand what was requested
2. **Read the output** completely before scoring
3. **Check for critical failures** first (auto-fail if found)
4. **Score each dimension** using the rubrics (1-5 scale)
5. **Calculate weighted average** for overall score
6. **Determine verdict** based on score and issues

## VIII. Output Format

Return your evaluation as valid JSON:

```json
{{
  "score": <weighted average 1.0-5.0>,
  "verdict": "<PASS|NEEDS_REVIEW|FAIL>",
  "reasoning": "<2-3 sentence overall assessment>",
  "dimension_scores": {{
{chr(10).join(f'    "{dim.id}": {{ "score": "<1-5>", "rationale": "<brief reason>" }},' for dim in dimensions[:-1])}
    "{dimensions[-1].id}": {{ "score": "<1-5>", "rationale": "<brief reason>" }}
  }},
  "critical_issues": ["<list any critical failures found>"],
  "strengths": ["<key strengths>"],
  "weaknesses": ["<key weaknesses>"],
  "improvement_suggestions": ["<specific improvements>"]
}}
```

## IX. Verdict Criteria

- **PASS**: Score >= 4.0 AND no critical issues
- **NEEDS_REVIEW**: Score >= 3.0 AND no critical issues (human review recommended)
- **FAIL**: Score < 3.0 OR any critical issue detected

## X. Evaluator Guidelines

1. **DO NOT** re-do the task or provide your own answer
2. **DO NOT** assume information not in the output
3. **DO** cite specific evidence from the output
4. **DO** be consistent - similar outputs should get similar scores
5. **DO** explain your reasoning clearly

---

Now evaluate the output below:

**Input ({input_var_name}):**
{{{{{input_var_name}}}}}

**Output to Evaluate ({output_var_name}):**
{{{{{output_var_name}}}}}"""

    return eval_prompt


# ============================================================================
# Main Generation Function
# ============================================================================

async def generate_best_eval_prompt(
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
    Main function to generate the best possible evaluation prompt.

    This is the core feature of Athena - creating evaluation prompts that:
    1. Comprehensively cover all failure modes
    2. Have clear, unambiguous rubrics
    3. Include calibration examples
    4. Are self-validated for quality
    """
    analysis_model = thinking_model or model_name
    steps_taken = []

    # Step 1: Deep Analysis
    logger.info(f"Step 1: Deep analysis with {analysis_model}")
    analysis = await analyze_system_prompt_deeply(
        system_prompt, use_case, requirements,
        llm_client, provider, api_key, analysis_model
    )
    steps_taken.append({"step": "analysis", "status": "completed"})

    # Step 2: Generate Failure Modes
    logger.info(f"Step 2: Generating failure modes")
    failure_modes = await generate_comprehensive_failure_modes(
        system_prompt, analysis,
        llm_client, provider, api_key, analysis_model
    )
    steps_taken.append({
        "step": "failure_modes",
        "status": "completed",
        "count": len(failure_modes)
    })

    # Step 3: Generate Evaluation Dimensions
    logger.info(f"Step 3: Generating eval dimensions")
    dimensions = await generate_eval_dimensions(
        system_prompt, failure_modes, analysis,
        llm_client, provider, api_key, model_name
    )
    steps_taken.append({
        "step": "dimensions",
        "status": "completed",
        "count": len(dimensions)
    })

    # Step 4: Generate Calibration Examples
    logger.info(f"Step 4: Generating calibration examples")
    calibration_examples = await generate_calibration_examples(
        system_prompt, dimensions, failure_modes,
        llm_client, provider, api_key, model_name
    )
    steps_taken.append({
        "step": "calibration",
        "status": "completed",
        "count": len(calibration_examples)
    })

    # Step 5: Build Initial Eval Prompt
    logger.info(f"Step 5: Building eval prompt")
    eval_prompt = build_eval_prompt(
        system_prompt, dimensions, failure_modes,
        calibration_examples, analysis, use_case
    )
    steps_taken.append({"step": "build_prompt", "status": "completed"})

    # Step 6: Self-Validation
    logger.info(f"Step 6: Self-validation")
    validation = await self_validate_eval_criteria(
        eval_prompt, dimensions, failure_modes, calibration_examples,
        llm_client, provider, api_key, model_name
    )
    steps_taken.append({
        "step": "validation",
        "status": "completed",
        "confidence": validation.get("overall_confidence", 0.8)
    })

    # Step 7: Iterate if needed
    iteration = 0
    while (
        iteration < max_iterations and
        validation.get("overall_confidence", 1.0) < 0.85 and
        validation.get("improvements_needed")
    ):
        iteration += 1
        logger.info(f"Step 7: Iteration {iteration} - improving based on validation")

        # Add dimensions for uncovered failure modes
        uncovered = validation.get("uncovered_failure_modes", [])
        if uncovered:
            uncovered_modes = [fm for fm in failure_modes if fm.id in uncovered]
            additional_dims = await generate_eval_dimensions(
                system_prompt, uncovered_modes, analysis,
                llm_client, provider, api_key, model_name
            )
            dimensions.extend(additional_dims)

        # Rebuild prompt
        eval_prompt = build_eval_prompt(
            system_prompt, dimensions, failure_modes,
            calibration_examples, analysis, use_case
        )

        # Re-validate
        validation = await self_validate_eval_criteria(
            eval_prompt, dimensions, failure_modes, calibration_examples,
            llm_client, provider, api_key, model_name
        )

        steps_taken.append({
            "step": f"iteration_{iteration}",
            "status": "completed",
            "confidence": validation.get("overall_confidence", 0.8)
        })

    # Build result
    rationale = (
        f"Generated {len(dimensions)} evaluation dimensions covering "
        f"{len(failure_modes)} failure modes. "
        f"Validation confidence: {validation.get('overall_confidence', 0.8):.0%}. "
    )

    if validation.get("strengths"):
        rationale += f"Strengths: {', '.join(validation['strengths'][:2])}."

    return EvalPromptResult(
        eval_prompt=eval_prompt,
        eval_criteria=[d.name for d in dimensions],
        rationale=rationale,
        failure_modes=[
            {
                "id": fm.id,
                "name": fm.name,
                "description": fm.description,
                "severity": fm.severity.value,
                "detection_criteria": fm.detection_criteria,
                "category": fm.category
            }
            for fm in failure_modes
        ],
        dimensions=[
            {
                "id": d.id,
                "name": d.name,
                "type": d.dimension_type.value,
                "description": d.description,
                "weight": round(d.weight, 3),
                "what_to_check": d.what_to_check,
                "rubric": {str(k): v for k, v in d.rubric.items()},
                "failure_modes_covered": d.failure_modes_covered
            }
            for d in dimensions
        ],
        calibration_examples=[
            {
                "input": ex.input,
                "output": ex.output,
                "score": ex.score,
                "reasoning": ex.reasoning,
                "dimension_scores": ex.dimension_scores
            }
            for ex in calibration_examples
        ],
        self_test_results=validation,
        metadata={
            "steps_taken": steps_taken,
            "iterations": iteration,
            "prompt_type": analysis.get("programmatic", {}).get("prompt_type", "unknown"),
            "model_used": model_name,
            "analysis_model": analysis_model
        }
    )
