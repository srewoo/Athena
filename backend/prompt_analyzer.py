"""
Prompt Analyzer Module for Athena
Phase 1: Prompt type detection, DNA extraction, Improvement threshold
Phase 2: Dynamic eval criteria, Smart test generation
"""
import re
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class PromptType(Enum):
    """Classification of prompt types"""
    STRUCTURED_OUTPUT = "structured_output"  # JSON, XML, specific format
    CONVERSATIONAL = "conversational"  # Chatbot, assistant
    ANALYTICAL = "analytical"  # Scoring, evaluation, classification
    CREATIVE = "creative"  # Writing, storytelling
    INSTRUCTIONAL = "instructional"  # How-to, step-by-step
    EXTRACTION = "extraction"  # Data extraction, parsing
    HYBRID = "hybrid"  # Combination of types


@dataclass
class PromptDNA:
    """Core elements that must be preserved in any rewrite"""
    # Template variables found in the prompt
    template_variables: List[str] = field(default_factory=list)

    # Output format specification
    output_format: Optional[str] = None  # "json", "xml", "markdown", "plain", etc.
    output_schema: Optional[Dict] = None  # Extracted JSON schema if applicable

    # Domain-specific terminology that should be preserved
    key_terminology: List[str] = field(default_factory=list)

    # Scoring/rating scales if present
    scoring_scale: Optional[Dict] = None  # e.g., {"min": 0, "max": 10, "type": "numeric"}

    # Required sections/headers in the prompt
    sections: List[str] = field(default_factory=list)

    # Constraints and rules
    constraints: List[str] = field(default_factory=list)

    # Role definition
    role: Optional[str] = None


@dataclass
class PromptAnalysis:
    """Complete analysis of a prompt"""
    prompt_type: PromptType
    prompt_types_detected: List[PromptType]  # Can have multiple
    dna: PromptDNA
    quality_score: float  # 1-10
    quality_breakdown: Dict[str, float]
    improvement_needed: bool
    improvement_areas: List[str]
    strengths: List[str]

    # For eval criteria generation
    suggested_eval_dimensions: List[Dict[str, str]]

    # For test data generation
    suggested_test_categories: List[Dict[str, Any]]


def extract_template_variables(text: str) -> List[str]:
    """Extract all {{variable}} and {variable} patterns"""
    # Match {{variable}}, {variable}, and <<variable>> patterns
    patterns = [
        r'\{\{([^}]+)\}\}',  # {{variable}}
        r'\{([^{}]+)\}',      # {variable} (but not JSON)
        r'<<([^>]+)>>',       # <<variable>>
    ]

    variables = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Filter out JSON-like content
            if not match.strip().startswith('"') and not ':' in match:
                var_name = match.strip()
                if var_name and var_name not in variables:
                    variables.append(var_name)

    # Also check for common placeholder patterns
    placeholder_patterns = [
        r'\[([A-Z_]+)\]',  # [PLACEHOLDER]
        r'<([a-z_]+)>',    # <placeholder>
    ]
    for pattern in placeholder_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if match not in variables and len(match) > 2:
                variables.append(match)

    return variables


def extract_output_format(text: str) -> tuple[Optional[str], Optional[Dict]]:
    """Detect the expected output format and schema"""
    text_lower = text.lower()

    # Check for JSON format
    if 'json' in text_lower or '```json' in text_lower:
        # Try to extract JSON schema
        json_pattern = r'```json\s*(\{[\s\S]*?\})\s*```'
        match = re.search(json_pattern, text)
        if match:
            try:
                schema = json.loads(match.group(1))
                return "json", schema
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in text
        json_obj_pattern = r'\{\s*"[^"]+"\s*:'
        if re.search(json_obj_pattern, text):
            return "json", None

        return "json", None

    # Check for XML format
    if '<xml' in text_lower or 'xml format' in text_lower or re.search(r'<[a-z]+>.*</[a-z]+>', text_lower):
        return "xml", None

    # Check for markdown
    if 'markdown' in text_lower or '```markdown' in text_lower:
        return "markdown", None

    # Check for specific structured formats
    if 'csv' in text_lower or 'comma-separated' in text_lower:
        return "csv", None

    if 'bullet' in text_lower or 'numbered list' in text_lower:
        return "list", None

    return "plain", None


def extract_scoring_scale(text: str) -> Optional[Dict]:
    """Extract scoring/rating scale if present"""
    # Common patterns for scoring scales
    patterns = [
        r'(\d+)\s*[-–to]+\s*(\d+)\s*scale',  # "0-10 scale", "1-5 scale"
        r'scale\s*(?:of\s*)?(\d+)\s*[-–to]+\s*(\d+)',  # "scale of 1-5"
        r'score\s*(?:of\s*)?(\d+)\s*[-–to]+\s*(\d+)',  # "score of 0-10"
        r'rating\s*(?:of\s*)?(\d+)\s*[-–to]+\s*(\d+)',  # "rating of 1-5"
        r'\((\d+)[-–](\d+)\)',  # (0-10), (1-5)
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            min_val = int(match.group(1))
            max_val = int(match.group(2))
            return {
                "min": min_val,
                "max": max_val,
                "type": "numeric",
                "range": max_val - min_val + 1
            }

    # Check for categorical scales
    if re.search(r'(poor|fair|good|excellent)', text.lower()):
        return {
            "type": "categorical",
            "categories": ["poor", "fair", "good", "excellent"]
        }

    if re.search(r'(pass|fail)', text.lower()):
        return {
            "type": "binary",
            "categories": ["pass", "fail"]
        }

    return None


def extract_key_terminology(text: str) -> List[str]:
    """Extract domain-specific terminology that should be preserved"""
    terminology = []

    # Look for quoted terms
    quoted = re.findall(r'"([^"]+)"', text)
    for term in quoted:
        if len(term.split()) <= 4 and term not in terminology:  # Max 4 words
            terminology.append(term)

    # Look for terms in bold/emphasis (markdown)
    bold = re.findall(r'\*\*([^*]+)\*\*', text)
    for term in bold:
        if len(term.split()) <= 4 and term not in terminology:
            terminology.append(term)

    # Look for capitalized multi-word terms (likely important)
    caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text)
    for term in caps:
        if term not in terminology:
            terminology.append(term)

    # Look for terms following "called", "named", "termed"
    named = re.findall(r'(?:called|named|termed|known as)\s+["\']?([^"\'.,\n]+)["\']?', text, re.IGNORECASE)
    for term in named:
        term = term.strip()
        if len(term.split()) <= 4 and term not in terminology:
            terminology.append(term)

    return terminology[:20]  # Limit to top 20


def extract_sections(text: str) -> List[str]:
    """Extract section headers from the prompt"""
    sections = []

    # Markdown headers
    md_headers = re.findall(r'^#{1,4}\s*\**([^#*\n]+)\**\s*$', text, re.MULTILINE)
    sections.extend([h.strip() for h in md_headers])

    # All caps headers
    caps_headers = re.findall(r'^([A-Z][A-Z\s]{3,}):?\s*$', text, re.MULTILINE)
    sections.extend([h.strip() for h in caps_headers])

    # Numbered sections
    numbered = re.findall(r'^\d+\.\s*\**([^*\n]+)\**\s*$', text, re.MULTILINE)
    sections.extend([h.strip() for h in numbered])

    # Roman numeral sections
    roman = re.findall(r'^[IVX]+\.\s*\**([^*\n]+)\**', text, re.MULTILINE)
    sections.extend([h.strip() for h in roman])

    return list(dict.fromkeys(sections))  # Remove duplicates, preserve order


def extract_constraints(text: str) -> List[str]:
    """Extract rules, constraints, and restrictions"""
    constraints = []

    # Look for constraint indicators
    constraint_patterns = [
        r'(?:must|should|always|never|do not|don\'t|cannot|can\'t|required to|ensure that)\s+([^.!?\n]+[.!?])',
        r'(?:important|critical|note|warning|caution):\s*([^.!?\n]+[.!?])',
        r'(?:rule|constraint|restriction|requirement):\s*([^.!?\n]+[.!?])',
    ]

    for pattern in constraint_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            constraint = match.strip()
            if len(constraint) > 10 and constraint not in constraints:
                constraints.append(constraint)

    return constraints[:15]  # Limit to top 15


def extract_role(text: str) -> Optional[str]:
    """Extract the role definition from the prompt"""
    # Common role patterns
    role_patterns = [
        r'you are (?:an? )?([^.!\n]+)',
        r'act as (?:an? )?([^.!\n]+)',
        r'your role is (?:to be )?(?:an? )?([^.!\n]+)',
        r'as (?:an? )?(expert|specialist|assistant|analyst|evaluator|generator)[^.!\n]*',
    ]

    for pattern in role_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            role = match.group(1).strip()
            if len(role) > 5:
                return role[:200]  # Limit length

    return None


def detect_prompt_type(text: str) -> tuple[PromptType, List[PromptType]]:
    """Detect the primary and secondary types of a prompt"""
    text_lower = text.lower()
    scores = {pt: 0 for pt in PromptType}

    # Structured Output indicators
    structured_indicators = [
        'json', 'xml', 'format', 'schema', 'structure', 'field',
        'return only', 'output format', 'response format', 'exactly'
    ]
    for indicator in structured_indicators:
        if indicator in text_lower:
            scores[PromptType.STRUCTURED_OUTPUT] += 2

    # Check for JSON/XML blocks
    if re.search(r'```(?:json|xml)', text_lower) or re.search(r'\{\s*"[^"]+"\s*:', text):
        scores[PromptType.STRUCTURED_OUTPUT] += 5

    # Analytical indicators
    analytical_indicators = [
        'score', 'rating', 'evaluate', 'assess', 'analyze', 'classify',
        'rubric', 'criteria', 'judgment', 'rate', 'rank', 'compare'
    ]
    for indicator in analytical_indicators:
        if indicator in text_lower:
            scores[PromptType.ANALYTICAL] += 2

    # Conversational indicators
    conversational_indicators = [
        'chat', 'conversation', 'respond to user', 'assistant', 'help',
        'friendly', 'natural', 'dialogue', 'talk', 'communicate'
    ]
    for indicator in conversational_indicators:
        if indicator in text_lower:
            scores[PromptType.CONVERSATIONAL] += 2

    # Creative indicators
    creative_indicators = [
        'write', 'create', 'story', 'creative', 'imagine', 'generate content',
        'compose', 'narrative', 'fiction', 'poetry', 'artistic'
    ]
    for indicator in creative_indicators:
        if indicator in text_lower:
            scores[PromptType.CREATIVE] += 2

    # Instructional indicators
    instructional_indicators = [
        'step-by-step', 'how to', 'guide', 'tutorial', 'instructions',
        'process', 'procedure', 'follow these', 'steps:'
    ]
    for indicator in instructional_indicators:
        if indicator in text_lower:
            scores[PromptType.INSTRUCTIONAL] += 2

    # Extraction indicators
    extraction_indicators = [
        'extract', 'parse', 'find', 'identify', 'locate', 'pull out',
        'get the', 'retrieve', 'summarize', 'key points'
    ]
    for indicator in extraction_indicators:
        if indicator in text_lower:
            scores[PromptType.EXTRACTION] += 2

    # Sort by score
    sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Determine primary type
    primary = sorted_types[0][0]
    primary_score = sorted_types[0][1]

    # If top score is very low, it's hybrid
    if primary_score < 3:
        primary = PromptType.HYBRID

    # Collect secondary types (score > 3)
    secondary = [pt for pt, score in sorted_types[1:4] if score >= 3]

    # If multiple high scores, might be hybrid
    if len([s for _, s in sorted_types[:3] if s >= 5]) > 1:
        all_types = [primary] + secondary
        return PromptType.HYBRID, all_types

    return primary, [primary] + secondary


def calculate_quality_score(text: str, dna: PromptDNA) -> tuple[float, Dict[str, float]]:
    """Calculate quality score and breakdown"""
    breakdown = {}

    # Structure score (0-10)
    structure_score = 5.0
    if len(dna.sections) >= 3:
        structure_score += 2
    if len(dna.sections) >= 5:
        structure_score += 1
    if dna.role:
        structure_score += 1
    if dna.output_format:
        structure_score += 1
    breakdown['structure'] = min(10, structure_score)

    # Clarity score (0-10)
    clarity_score = 5.0
    word_count = len(text.split())
    if 100 < word_count < 2000:
        clarity_score += 1
    if len(dna.constraints) > 0:
        clarity_score += 1
    if dna.scoring_scale:
        clarity_score += 1
    # Check for clear instructions
    if any(word in text.lower() for word in ['must', 'should', 'always', 'never']):
        clarity_score += 1
    if '1.' in text or 'step 1' in text.lower():
        clarity_score += 1
    breakdown['clarity'] = min(10, clarity_score)

    # Completeness score (0-10)
    completeness_score = 5.0
    if dna.role:
        completeness_score += 1
    if dna.output_format:
        completeness_score += 1
    if dna.output_schema:
        completeness_score += 1
    if len(dna.constraints) >= 3:
        completeness_score += 1
    if len(dna.template_variables) > 0:
        completeness_score += 1
    breakdown['completeness'] = min(10, completeness_score)

    # Output format score (0-10)
    format_score = 5.0
    if dna.output_format:
        format_score += 2
    if dna.output_schema:
        format_score += 2
    if 'example' in text.lower() or 'sample' in text.lower():
        format_score += 1
    breakdown['output_format'] = min(10, format_score)

    # Calculate overall
    weights = {
        'structure': 0.25,
        'clarity': 0.25,
        'completeness': 0.25,
        'output_format': 0.25
    }
    overall = sum(breakdown[k] * weights[k] for k in weights)

    return round(overall, 1), breakdown


def determine_improvement_areas(
    quality_breakdown: Dict[str, float],
    dna: PromptDNA,
    prompt_type: PromptType
) -> tuple[bool, List[str], List[str]]:
    """Determine if improvement is needed and what areas"""
    improvement_areas = []
    strengths = []

    # Check each dimension
    if quality_breakdown.get('structure', 0) < 7:
        improvement_areas.append("Add clearer section structure with headers")
    else:
        strengths.append("Well-organized section structure")

    if quality_breakdown.get('clarity', 0) < 7:
        improvement_areas.append("Make instructions more specific and actionable")
    else:
        strengths.append("Clear and specific instructions")

    if quality_breakdown.get('completeness', 0) < 7:
        improvement_areas.append("Add missing elements (role, constraints, examples)")
    else:
        strengths.append("Comprehensive coverage of requirements")

    if quality_breakdown.get('output_format', 0) < 7:
        improvement_areas.append("Specify output format more clearly")
    else:
        strengths.append("Well-defined output format")

    # Type-specific checks
    if prompt_type == PromptType.ANALYTICAL and not dna.scoring_scale:
        improvement_areas.append("Add explicit scoring rubric/scale")

    if prompt_type == PromptType.STRUCTURED_OUTPUT and not dna.output_schema:
        improvement_areas.append("Add example of expected output structure")

    if not dna.role:
        improvement_areas.append("Add clear role definition")

    if len(dna.constraints) < 2:
        improvement_areas.append("Add explicit constraints and rules")

    # Determine if improvement is needed
    overall_score = sum(quality_breakdown.values()) / len(quality_breakdown)
    improvement_needed = overall_score < 8.0 or len(improvement_areas) > 2

    return improvement_needed, improvement_areas, strengths


def generate_eval_dimensions(prompt_type: PromptType, dna: PromptDNA) -> List[Dict[str, str]]:
    """Generate evaluation dimensions specific to the prompt type"""
    dimensions = []

    # Universal dimensions
    dimensions.append({
        "name": "Format Compliance",
        "description": "Output matches the required format and structure",
        "check": f"Verify output is valid {dna.output_format or 'text'} with all required fields"
    })

    # Type-specific dimensions
    if prompt_type == PromptType.ANALYTICAL:
        dimensions.extend([
            {
                "name": "Rubric Fidelity",
                "description": "Scores align with the scoring rubric definitions",
                "check": "Verify each score matches the criteria for that level"
            },
            {
                "name": "Evidence Grounding",
                "description": "All claims are supported by evidence from input",
                "check": "No hallucinated facts or unsupported assertions"
            },
            {
                "name": "Score-Reasoning Consistency",
                "description": "Numeric scores match the written reasoning",
                "check": "High scores have positive reasoning, low scores identify issues"
            }
        ])

    elif prompt_type == PromptType.CONVERSATIONAL:
        dimensions.extend([
            {
                "name": "Response Relevance",
                "description": "Response directly addresses the user's query",
                "check": "Answer is on-topic and helpful"
            },
            {
                "name": "Tone Appropriateness",
                "description": "Tone matches the expected style (friendly, professional, etc.)",
                "check": "Language and style are appropriate for the context"
            },
            {
                "name": "Safety Compliance",
                "description": "Response follows safety guidelines",
                "check": "No harmful, biased, or inappropriate content"
            }
        ])

    elif prompt_type == PromptType.STRUCTURED_OUTPUT:
        dimensions.extend([
            {
                "name": "Schema Validity",
                "description": "Output matches the exact schema specification",
                "check": "All required fields present with correct data types"
            },
            {
                "name": "Data Accuracy",
                "description": "Extracted/generated data is correct",
                "check": "Values are accurate based on input"
            },
            {
                "name": "No Extra Content",
                "description": "No additional fields or commentary outside schema",
                "check": "Output contains only specified fields"
            }
        ])

    elif prompt_type == PromptType.CREATIVE:
        dimensions.extend([
            {
                "name": "Creativity & Originality",
                "description": "Content is creative and not generic",
                "check": "Unique elements, not template responses"
            },
            {
                "name": "Style Adherence",
                "description": "Writing style matches requirements",
                "check": "Tone, voice, and style are consistent"
            },
            {
                "name": "Constraint Satisfaction",
                "description": "All creative constraints are met",
                "check": "Length, format, and content requirements satisfied"
            }
        ])

    elif prompt_type == PromptType.EXTRACTION:
        dimensions.extend([
            {
                "name": "Extraction Accuracy",
                "description": "Correct information extracted from input",
                "check": "All relevant data points captured"
            },
            {
                "name": "Completeness",
                "description": "No relevant information missed",
                "check": "All required extractions present"
            },
            {
                "name": "No Fabrication",
                "description": "Only information from input, nothing invented",
                "check": "All extracted data traceable to source"
            }
        ])

    else:  # INSTRUCTIONAL or HYBRID
        dimensions.extend([
            {
                "name": "Accuracy",
                "description": "Information provided is correct",
                "check": "No factual errors or misleading information"
            },
            {
                "name": "Completeness",
                "description": "All required elements are present",
                "check": "Response addresses all aspects of the request"
            },
            {
                "name": "Clarity",
                "description": "Response is clear and understandable",
                "check": "Easy to follow, well-organized"
            }
        ])

    return dimensions


def generate_test_categories(prompt_type: PromptType, dna: PromptDNA) -> List[Dict[str, Any]]:
    """Generate test case categories specific to the prompt type"""
    categories = []

    # Universal categories with type-specific examples
    categories.append({
        "name": "positive",
        "percentage": 50,
        "description": "Typical, valid inputs that should produce good outputs",
        "examples": get_positive_examples(prompt_type)
    })

    categories.append({
        "name": "edge_case",
        "percentage": 20,
        "description": "Boundary conditions and unusual but valid inputs",
        "examples": get_edge_examples(prompt_type, dna)
    })

    categories.append({
        "name": "negative",
        "percentage": 15,
        "description": "Invalid or out-of-scope inputs",
        "examples": get_negative_examples(prompt_type)
    })

    categories.append({
        "name": "adversarial",
        "percentage": 15,
        "description": "Inputs designed to test robustness",
        "examples": get_adversarial_examples(prompt_type)
    })

    return categories


def get_positive_examples(prompt_type: PromptType) -> List[str]:
    """Get positive test case examples for a prompt type"""
    examples = {
        PromptType.ANALYTICAL: [
            "Complete data with clear indicators for scoring",
            "Standard case with typical values",
            "Good quality input requiring high scores"
        ],
        PromptType.CONVERSATIONAL: [
            "Clear, straightforward user question",
            "Polite request for information",
            "Follow-up question in context"
        ],
        PromptType.STRUCTURED_OUTPUT: [
            "Well-formed input with all required data",
            "Input matching expected schema",
            "Complete request with all parameters"
        ],
        PromptType.CREATIVE: [
            "Clear creative brief with specific requirements",
            "Standard format request",
            "Well-defined topic and constraints"
        ],
        PromptType.EXTRACTION: [
            "Text with clear, extractable information",
            "Well-structured source document",
            "Input with obvious data points"
        ],
        PromptType.INSTRUCTIONAL: [
            "Clear how-to question",
            "Specific task with defined scope",
            "Standard procedural request"
        ]
    }
    return examples.get(prompt_type, examples[PromptType.INSTRUCTIONAL])


def get_edge_examples(prompt_type: PromptType, dna: PromptDNA) -> List[str]:
    """Get edge case examples for a prompt type"""
    base_edges = [
        "Minimum viable input (very short)",
        "Maximum length input",
        "Input with special characters",
        "Input at boundary of requirements"
    ]

    type_edges = {
        PromptType.ANALYTICAL: [
            "Ambiguous case that could score multiple ways",
            "Borderline pass/fail scenario",
            "Mixed signals (some good, some bad indicators)"
        ],
        PromptType.CONVERSATIONAL: [
            "Ambiguous or unclear question",
            "Multi-part complex question",
            "Question requiring clarification"
        ],
        PromptType.STRUCTURED_OUTPUT: [
            "Input with optional fields missing",
            "Input with null/empty values",
            "Input with maximum/minimum values"
        ]
    }

    specific_edges = type_edges.get(prompt_type, [])

    # Add edges based on DNA
    if dna.scoring_scale:
        specific_edges.append(f"Case at exact boundary ({dna.scoring_scale.get('min', 0) + 1})")

    return base_edges[:2] + specific_edges[:2]


def get_negative_examples(prompt_type: PromptType) -> List[str]:
    """Get negative test case examples for a prompt type"""
    examples = {
        PromptType.ANALYTICAL: [
            "Missing required data fields",
            "Empty or null input",
            "Wrong data format"
        ],
        PromptType.CONVERSATIONAL: [
            "Off-topic request",
            "Request outside scope",
            "Gibberish input"
        ],
        PromptType.STRUCTURED_OUTPUT: [
            "Malformed input data",
            "Missing required fields",
            "Wrong data types"
        ]
    }
    return examples.get(prompt_type, ["Invalid input", "Out of scope request", "Malformed data"])


def get_adversarial_examples(prompt_type: PromptType) -> List[str]:
    """Get adversarial test case examples"""
    return [
        "Prompt injection attempt ('ignore previous instructions...')",
        "Request to reveal system prompt",
        "Input with hidden instructions",
        "Attempt to change output format"
    ]


def analyze_prompt(text: str) -> PromptAnalysis:
    """
    Main function to analyze a prompt and return complete analysis.
    This is the primary entry point for the prompt analyzer.
    """
    # Extract DNA components
    template_vars = extract_template_variables(text)
    output_format, output_schema = extract_output_format(text)
    scoring_scale = extract_scoring_scale(text)
    key_terms = extract_key_terminology(text)
    sections = extract_sections(text)
    constraints = extract_constraints(text)
    role = extract_role(text)

    dna = PromptDNA(
        template_variables=template_vars,
        output_format=output_format,
        output_schema=output_schema,
        key_terminology=key_terms,
        scoring_scale=scoring_scale,
        sections=sections,
        constraints=constraints,
        role=role
    )

    # Detect prompt type
    primary_type, all_types = detect_prompt_type(text)

    # Calculate quality
    quality_score, quality_breakdown = calculate_quality_score(text, dna)

    # Determine improvements
    improvement_needed, improvement_areas, strengths = determine_improvement_areas(
        quality_breakdown, dna, primary_type
    )

    # Generate eval dimensions
    eval_dimensions = generate_eval_dimensions(primary_type, dna)

    # Generate test categories
    test_categories = generate_test_categories(primary_type, dna)

    return PromptAnalysis(
        prompt_type=primary_type,
        prompt_types_detected=all_types,
        dna=dna,
        quality_score=quality_score,
        quality_breakdown=quality_breakdown,
        improvement_needed=improvement_needed,
        improvement_areas=improvement_areas,
        strengths=strengths,
        suggested_eval_dimensions=eval_dimensions,
        suggested_test_categories=test_categories
    )


def analysis_to_dict(analysis: PromptAnalysis) -> Dict[str, Any]:
    """Convert PromptAnalysis to dictionary for JSON serialization"""
    return {
        "prompt_type": analysis.prompt_type.value,
        "prompt_types_detected": [pt.value for pt in analysis.prompt_types_detected],
        "dna": {
            "template_variables": analysis.dna.template_variables,
            "output_format": analysis.dna.output_format,
            "output_schema": analysis.dna.output_schema,
            "key_terminology": analysis.dna.key_terminology,
            "scoring_scale": analysis.dna.scoring_scale,
            "sections": analysis.dna.sections,
            "constraints": analysis.dna.constraints,
            "role": analysis.dna.role
        },
        "quality_score": analysis.quality_score,
        "quality_breakdown": analysis.quality_breakdown,
        "improvement_needed": analysis.improvement_needed,
        "improvement_areas": analysis.improvement_areas,
        "strengths": analysis.strengths,
        "suggested_eval_dimensions": analysis.suggested_eval_dimensions,
        "suggested_test_categories": analysis.suggested_test_categories
    }
