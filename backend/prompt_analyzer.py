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
class PromptTypeLabel:
    """A single prompt type label with confidence score"""
    type: PromptType
    confidence: float  # 0.0 to 1.0
    indicators_found: List[str] = field(default_factory=list)
    indicator_count: int = 0

    def __post_init__(self):
        self.indicator_count = len(self.indicators_found)


@dataclass
class MultiLabelTypeResult:
    """Multi-label classification result for a prompt"""
    primary_type: PromptType
    primary_confidence: float
    all_labels: List[PromptTypeLabel]  # All types with confidence > threshold
    is_multi_type: bool  # True if multiple types have high confidence
    type_composition: Dict[str, float]  # Normalized percentages of each type

    def get_types_above_threshold(self, threshold: float = 0.3) -> List[PromptType]:
        """Get all types with confidence above threshold"""
        return [label.type for label in self.all_labels if label.confidence >= threshold]

    def get_secondary_types(self, threshold: float = 0.3) -> List[PromptType]:
        """Get secondary types (excluding primary)"""
        return [
            label.type for label in self.all_labels
            if label.confidence >= threshold and label.type != self.primary_type
        ]


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
    quality_score: float  # 0-10 (earned, not assumed)
    quality_breakdown: Dict[str, float]
    improvement_needed: bool
    improvement_areas: List[str]
    strengths: List[str]

    # For eval criteria generation
    suggested_eval_dimensions: List[Dict[str, str]]

    # For test data generation
    suggested_test_categories: List[Dict[str, Any]]

    # Multi-label typing (new)
    multi_label_result: Optional[MultiLabelTypeResult] = None
    
    # Confidence information for score reliability
    confidence_info: Optional[Dict[str, Any]] = None


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

    # Limit to prevent overwhelming output; 20 is sufficient for most prompts
    # based on analysis of typical prompt terminology density
    MAX_TERMINOLOGY = 20
    return terminology[:MAX_TERMINOLOGY]


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

    # Limit to most important constraints; 15 covers typical complex prompts
    # while avoiding noise from false positive pattern matches
    MAX_CONSTRAINTS = 15
    return constraints[:MAX_CONSTRAINTS]


def extract_role(text: str) -> Optional[str]:
    """Extract the role definition from the prompt"""
    # Common role patterns
    role_patterns = [
        r'you are (?:an? )?([^.!\n]+)',
        r'act as (?:an? )?([^.!\n]+)',
        r'your role is (?:to be )?(?:an? )?([^.!\n]+)',
        r'as (?:an? )?(expert|specialist|assistant|analyst|evaluator|generator)[^.!\n]*',
    ]

    # Minimum role length to filter out false positives like "a"
    MIN_ROLE_LENGTH = 5
    # Maximum role length to prevent extracting entire paragraphs
    MAX_ROLE_LENGTH = 200
    
    for pattern in role_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            role = match.group(1).strip()
            if len(role) > MIN_ROLE_LENGTH:
                return role[:MAX_ROLE_LENGTH]

    return None


def detect_prompt_type(text: str) -> tuple[PromptType, List[PromptType]]:
    """Detect the primary and secondary types of a prompt (legacy interface)"""
    result = detect_prompt_type_multi_label(text)
    secondary = [label.type for label in result.all_labels[1:4] if label.confidence >= 0.2]
    return result.primary_type, [result.primary_type] + secondary


def detect_prompt_type_multi_label(text: str, confidence_threshold: float = 0.15) -> MultiLabelTypeResult:
    """
    Detect prompt types using multi-label classification with confidence scores.

    This improved version:
    1. Returns confidence scores for each type (0-1)
    2. Tracks which indicators were found
    3. Handles multi-type prompts better
    4. Provides type composition percentages

    Args:
        text: The prompt text to analyze
        confidence_threshold: Minimum confidence to include a type (default 0.15)

    Returns:
        MultiLabelTypeResult with detailed classification
    """
    text_lower = text.lower()

    # Define indicators with weights for each type
    type_indicators = {
        PromptType.STRUCTURED_OUTPUT: {
            'high': ['```json', '```xml', 'output schema', 'json format', 'xml format'],
            'medium': ['json', 'xml', 'format', 'schema', 'structure', 'field'],
            'low': ['return only', 'output format', 'response format', 'exactly', 'strictly']
        },
        PromptType.ANALYTICAL: {
            'high': ['rubric', 'scoring criteria', 'evaluation criteria', 'grading scale'],
            'medium': ['score', 'rating', 'evaluate', 'assess', 'analyze', 'classify'],
            'low': ['criteria', 'judgment', 'rate', 'rank', 'compare', 'quality']
        },
        PromptType.CONVERSATIONAL: {
            'high': ['chatbot', 'conversation history', 'respond to user', 'user message'],
            'medium': ['chat', 'conversation', 'assistant', 'dialogue', 'user'],
            'low': ['help', 'friendly', 'natural', 'talk', 'communicate', 'respond']
        },
        PromptType.CREATIVE: {
            'high': ['creative writing', 'write a story', 'compose a', 'narrative voice'],
            'medium': ['write', 'create', 'story', 'creative', 'imagine', 'generate content'],
            'low': ['compose', 'narrative', 'fiction', 'poetry', 'artistic', 'original']
        },
        PromptType.INSTRUCTIONAL: {
            'high': ['step-by-step', 'step 1', 'follow these steps', 'instructions:'],
            'medium': ['how to', 'guide', 'tutorial', 'instructions', 'procedure'],
            'low': ['process', 'steps', 'method', 'approach', 'technique']
        },
        PromptType.EXTRACTION: {
            'high': ['extract the following', 'parse the', 'identify all', 'find all'],
            'medium': ['extract', 'parse', 'find', 'identify', 'locate', 'pull out'],
            'low': ['get the', 'retrieve', 'summarize', 'key points', 'main ideas']
        }
    }

    # Weights for indicator importance
    weights = {'high': 3.0, 'medium': 1.5, 'low': 0.5}

    # Calculate scores and track indicators
    type_scores: Dict[PromptType, float] = {}
    type_indicators_found: Dict[PromptType, List[str]] = {pt: [] for pt in PromptType if pt != PromptType.HYBRID}

    for prompt_type, indicators in type_indicators.items():
        score = 0.0
        found = []

        for weight_level, indicator_list in indicators.items():
            weight = weights[weight_level]
            for indicator in indicator_list:
                # Check for indicator (with word boundary awareness for short indicators)
                if len(indicator) <= 4:
                    # Short indicators need word boundaries
                    if re.search(rf'\b{re.escape(indicator)}\b', text_lower):
                        score += weight
                        found.append(f"{indicator} ({weight_level})")
                else:
                    if indicator in text_lower:
                        score += weight
                        found.append(f"{indicator} ({weight_level})")

        # Bonus for JSON/XML code blocks
        if prompt_type == PromptType.STRUCTURED_OUTPUT:
            if re.search(r'```(?:json|xml|yaml)', text_lower):
                score += 5.0
                found.append("code block (bonus)")
            if re.search(r'\{\s*"[^"]+"\s*:', text):
                score += 3.0
                found.append("JSON object (bonus)")

        # Bonus for numbered steps in instructional
        if prompt_type == PromptType.INSTRUCTIONAL:
            step_count = len(re.findall(r'^\s*\d+\.\s', text, re.MULTILINE))
            if step_count >= 3:
                score += 2.0 * min(step_count / 5, 1.0)
                found.append(f"{step_count} numbered steps (bonus)")

        type_scores[prompt_type] = score
        type_indicators_found[prompt_type] = found

    # Normalize scores to confidence (0-1)
    max_possible = 25.0  # Approximate max score
    total_score = sum(type_scores.values())

    labels = []
    for prompt_type, score in type_scores.items():
        # Confidence based on both absolute score and relative proportion
        absolute_conf = min(1.0, score / max_possible)
        relative_conf = score / total_score if total_score > 0 else 0

        # Blend absolute and relative confidence
        confidence = (absolute_conf * 0.6 + relative_conf * 0.4)

        if confidence >= confidence_threshold:
            labels.append(PromptTypeLabel(
                type=prompt_type,
                confidence=round(confidence, 3),
                indicators_found=type_indicators_found[prompt_type]
            ))

    # Sort by confidence
    labels.sort(key=lambda x: x.confidence, reverse=True)

    # Determine primary type
    if labels:
        primary = labels[0]
    else:
        primary = PromptTypeLabel(type=PromptType.HYBRID, confidence=0.5, indicators_found=[])
        labels = [primary]

    # Check if multi-type (multiple high-confidence types)
    high_conf_types = [l for l in labels if l.confidence >= 0.4]
    is_multi_type = len(high_conf_types) >= 2

    # If multi-type and primary isn't dramatically higher, consider it HYBRID
    if is_multi_type and len(labels) >= 2:
        if labels[0].confidence - labels[1].confidence < 0.15:
            # Close competition - might be hybrid
            primary = PromptTypeLabel(
                type=PromptType.HYBRID,
                confidence=labels[0].confidence,
                indicators_found=labels[0].indicators_found + labels[1].indicators_found[:3]
            )

    # Calculate type composition (how much of each type)
    type_composition = {}
    total_conf = sum(l.confidence for l in labels)
    for label in labels:
        type_composition[label.type.value] = round(label.confidence / total_conf, 3) if total_conf > 0 else 0

    return MultiLabelTypeResult(
        primary_type=primary.type,
        primary_confidence=primary.confidence,
        all_labels=labels,
        is_multi_type=is_multi_type,
        type_composition=type_composition
    )


def calculate_quality_score(text: str, dna: PromptDNA) -> tuple[float, Dict[str, float], Dict[str, Any]]:
    """
    Calculate quality score and breakdown using EARNED scoring (start at 0, not 5).
    
    Returns:
        tuple: (overall_score, breakdown_dict, confidence_info)
        
    Scoring Philosophy:
        - Scores are EARNED, not assumed (start at 0, add points for quality indicators)
        - Each dimension has clear criteria with documented thresholds
        - Confidence intervals provided based on how many indicators were found
    """
    breakdown = {}
    confidence_info = {"indicators_found": {}, "confidence_level": "low"}
    text_lower = text.lower()
    word_count = len(text.split())
    
    # ========================================================================
    # STRUCTURE SCORE (0-10) - Earn points for organizational quality
    # ========================================================================
    # Thresholds documented for reproducibility
    SECTION_THRESHOLD_GOOD = 3  # 3+ sections indicates good organization
    SECTION_THRESHOLD_EXCELLENT = 5  # 5+ sections indicates comprehensive structure
    
    structure_score = 0.0
    structure_indicators = []
    
    # Baseline: Prompt exists and has content (2 points)
    if word_count >= 10:  # At least 10 words
        structure_score += 2.0
        structure_indicators.append("has_content")
    
    # Sections indicate organization (up to 3 points)
    if len(dna.sections) >= SECTION_THRESHOLD_EXCELLENT:
        structure_score += 3.0
        structure_indicators.append(f"excellent_sections({len(dna.sections)})")
    elif len(dna.sections) >= SECTION_THRESHOLD_GOOD:
        structure_score += 2.0
        structure_indicators.append(f"good_sections({len(dna.sections)})")
    elif len(dna.sections) >= 1:
        structure_score += 1.0
        structure_indicators.append(f"some_sections({len(dna.sections)})")
    
    # Role definition shows intentional design (1.5 points)
    if dna.role:
        structure_score += 1.5
        structure_indicators.append("has_role")
    
    # Output format indicates structured thinking (1.5 points)
    if dna.output_format and dna.output_format != "plain":
        structure_score += 1.5
        structure_indicators.append(f"has_output_format({dna.output_format})")
    
    # Structural markers show formatting effort (up to 1.5 points)
    markdown_markers = ['##', '###', '**', '---', '===', '```']
    markers_found = sum(1 for m in markdown_markers if m in text)
    if markers_found >= 3:
        structure_score += 1.5
        structure_indicators.append(f"rich_formatting({markers_found})")
    elif markers_found >= 1:
        structure_score += 0.5
        structure_indicators.append(f"some_formatting({markers_found})")
    
    breakdown['structure'] = min(10.0, structure_score)
    confidence_info["indicators_found"]["structure"] = structure_indicators
    
    # ========================================================================
    # CLARITY SCORE (0-10) - Earn points for instruction quality
    # ========================================================================
    OPTIMAL_LENGTH_MIN = 50  # Too short = likely incomplete
    OPTIMAL_LENGTH_MAX = 3000  # Too long = hard to follow
    CONSTRAINT_THRESHOLD_GOOD = 2  # 2+ constraints shows attention to boundaries
    
    clarity_score = 0.0
    clarity_indicators = []
    
    # Length in optimal range (2 points)
    if OPTIMAL_LENGTH_MIN <= word_count <= OPTIMAL_LENGTH_MAX:
        clarity_score += 2.0
        clarity_indicators.append(f"optimal_length({word_count})")
    elif word_count > OPTIMAL_LENGTH_MAX:
        clarity_score += 1.0  # Partial credit for comprehensive but verbose
        clarity_indicators.append(f"verbose({word_count})")
    elif word_count >= 20:
        clarity_score += 0.5
        clarity_indicators.append(f"short({word_count})")
    
    # Explicit constraints show clear boundaries (up to 2 points)
    if len(dna.constraints) >= 5:
        clarity_score += 2.0
        clarity_indicators.append(f"many_constraints({len(dna.constraints)})")
    elif len(dna.constraints) >= CONSTRAINT_THRESHOLD_GOOD:
        clarity_score += 1.5
        clarity_indicators.append(f"good_constraints({len(dna.constraints)})")
    elif len(dna.constraints) >= 1:
        clarity_score += 0.5
        clarity_indicators.append(f"some_constraints({len(dna.constraints)})")
    
    # Scoring rubric for analytical prompts (1.5 points)
    if dna.scoring_scale:
        clarity_score += 1.5
        clarity_indicators.append("has_scoring_scale")
    
    # Directive language shows clear expectations (up to 2 points)
    directive_words = ['must', 'should', 'always', 'never', 'required', 'ensure']
    directives_found = sum(1 for w in directive_words if w in text_lower)
    if directives_found >= 3:
        clarity_score += 2.0
        clarity_indicators.append(f"strong_directives({directives_found})")
    elif directives_found >= 1:
        clarity_score += 1.0
        clarity_indicators.append(f"some_directives({directives_found})")
    
    # Step-by-step or numbered instructions (1.5 points)
    has_steps = bool(re.search(r'step\s*\d|^\s*\d+\.', text_lower, re.MULTILINE))
    if has_steps:
        clarity_score += 1.5
        clarity_indicators.append("has_steps")
    
    # Task definition (1 point)
    task_phrases = ['your task', 'you will', 'your goal', 'your job', 'you must']
    if any(phrase in text_lower for phrase in task_phrases):
        clarity_score += 1.0
        clarity_indicators.append("explicit_task")
    
    breakdown['clarity'] = min(10.0, clarity_score)
    confidence_info["indicators_found"]["clarity"] = clarity_indicators
    
    # ========================================================================
    # COMPLETENESS SCORE (0-10) - Earn points for comprehensive coverage
    # ========================================================================
    completeness_score = 0.0
    completeness_indicators = []
    
    # Role definition (2 points)
    if dna.role:
        completeness_score += 2.0
        completeness_indicators.append("has_role")
    
    # Output format specification (2 points)
    if dna.output_format and dna.output_format != "plain":
        completeness_score += 1.5
        completeness_indicators.append(f"output_format({dna.output_format})")
    if dna.output_schema:
        completeness_score += 1.5
        completeness_indicators.append("has_schema")
    
    # Constraints and rules (1.5 points)
    if len(dna.constraints) >= 3:
        completeness_score += 1.5
        completeness_indicators.append(f"sufficient_constraints({len(dna.constraints)})")
    elif len(dna.constraints) >= 1:
        completeness_score += 0.5
        completeness_indicators.append(f"minimal_constraints({len(dna.constraints)})")
    
    # Template variables show input awareness (1.5 points)
    if len(dna.template_variables) > 0:
        completeness_score += 1.5
        completeness_indicators.append(f"template_vars({len(dna.template_variables)})")
    
    # Input/output specification (1.5 points)
    io_words = ['input', 'output', 'return', 'respond', 'provide', 'generate']
    io_found = sum(1 for w in io_words if w in text_lower)
    if io_found >= 2:
        completeness_score += 1.5
        completeness_indicators.append(f"io_spec({io_found})")
    elif io_found >= 1:
        completeness_score += 0.5
        completeness_indicators.append(f"partial_io({io_found})")
    
    breakdown['completeness'] = min(10.0, completeness_score)
    confidence_info["indicators_found"]["completeness"] = completeness_indicators
    
    # ========================================================================
    # OUTPUT FORMAT SCORE (0-10) - Earn points for format specification
    # ========================================================================
    format_score = 0.0
    format_indicators = []
    
    # Explicit format declaration (3 points)
    if dna.output_format and dna.output_format != "plain":
        format_score += 3.0
        format_indicators.append(f"declared_format({dna.output_format})")
    
    # Schema/example provided (3 points)
    if dna.output_schema:
        format_score += 3.0
        format_indicators.append("has_schema")
    
    # Example output shown (2 points)
    example_patterns = ['example:', 'sample:', 'e.g.', 'for example', '```']
    examples_found = sum(1 for p in example_patterns if p in text_lower)
    if examples_found >= 2:
        format_score += 2.0
        format_indicators.append(f"has_examples({examples_found})")
    elif examples_found >= 1:
        format_score += 1.0
        format_indicators.append(f"partial_examples({examples_found})")
    
    # Format specification keywords (2 points)
    format_keywords = ['json', 'xml', 'markdown', 'yaml', 'csv', 'structured']
    format_kw_found = sum(1 for f in format_keywords if f in text_lower)
    if format_kw_found >= 1:
        format_score += min(2.0, format_kw_found * 0.5)
        format_indicators.append(f"format_keywords({format_kw_found})")
    
    breakdown['output_format'] = min(10.0, format_score)
    confidence_info["indicators_found"]["output_format"] = format_indicators
    
    # ========================================================================
    # CALCULATE OVERALL SCORE WITH CONFIDENCE
    # ========================================================================
    weights = {
        'structure': 0.25,
        'clarity': 0.25,
        'completeness': 0.25,
        'output_format': 0.25
    }
    overall = sum(breakdown[k] * weights[k] for k in weights)
    
    # Calculate confidence based on indicators found
    total_indicators = sum(len(v) for v in confidence_info["indicators_found"].values())
    if total_indicators >= 12:
        confidence_info["confidence_level"] = "high"
        confidence_info["confidence_note"] = "Many quality signals detected"
    elif total_indicators >= 6:
        confidence_info["confidence_level"] = "medium"
        confidence_info["confidence_note"] = "Moderate quality signals detected"
    else:
        confidence_info["confidence_level"] = "low"
        confidence_info["confidence_note"] = "Few quality signals detected; score may be less reliable"
    
    confidence_info["total_indicators"] = total_indicators

    return round(overall, 1), breakdown, confidence_info


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


def analyze_prompt(text: str, structured_requirements: Optional[Dict[str, Any]] = None) -> PromptAnalysis:
    """
    Main function to analyze a prompt and return complete analysis.
    This is the primary entry point for the prompt analyzer.
    
    Args:
        text: The prompt text to analyze
        structured_requirements: Optional structured requirements dict with keys:
            - must_do: List of required behaviors
            - must_not_do: List of forbidden behaviors  
            - tone: Expected tone
            - output_format: Expected output structure
            - constraints: Additional constraints
            - edge_cases: Known edge cases
            - success_criteria: Success metrics
    """
    # Extract DNA components
    template_vars = extract_template_variables(text)
    output_format, output_schema = extract_output_format(text)
    scoring_scale = extract_scoring_scale(text)
    key_terms = extract_key_terminology(text)
    sections = extract_sections(text)
    constraints = extract_constraints(text)
    role = extract_role(text)
    
    # Enhance constraints with structured requirements
    if structured_requirements:
        if structured_requirements.get('constraints'):
            constraints.extend(structured_requirements['constraints'])
        if structured_requirements.get('must_not_do'):
            constraints.extend([f"Must NOT: {item}" for item in structured_requirements['must_not_do']])

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

    # Detect prompt type using multi-label classification
    multi_label_result = detect_prompt_type_multi_label(text)
    primary_type = multi_label_result.primary_type
    all_types = [label.type for label in multi_label_result.all_labels]

    # Calculate quality with confidence information
    quality_score, quality_breakdown, confidence_info = calculate_quality_score(text, dna)

    # Determine improvements
    improvement_needed, improvement_areas, strengths = determine_improvement_areas(
        quality_breakdown, dna, primary_type
    )
    
    # Enhance improvement areas with structured requirements checks
    if structured_requirements:
        sr_improvements = []
        
        # Check must_do items
        if structured_requirements.get('must_do'):
            for item in structured_requirements['must_do']:
                if item.lower() not in text.lower():
                    sr_improvements.append(f"Missing required behavior: {item}")
        
        # Check tone alignment
        if structured_requirements.get('tone'):
            expected_tone = structured_requirements['tone']
            if expected_tone.lower() not in text.lower():
                sr_improvements.append(f"Consider specifying '{expected_tone}' tone explicitly")
        
        # Check output format
        if structured_requirements.get('output_format'):
            expected_format = structured_requirements['output_format']
            if not dna.output_format:
                sr_improvements.append(f"Specify output format: {expected_format}")
        
        # Add structured requirement improvements
        if sr_improvements:
            improvement_areas.extend(sr_improvements)
            improvement_needed = True

    # Generate eval dimensions (consider multiple types for multi-type prompts)
    eval_dimensions = generate_eval_dimensions(primary_type, dna)
    
    # Enhance eval dimensions with structured requirements
    if structured_requirements:
        # Add requirement compliance dimension
        if structured_requirements.get('must_do') or structured_requirements.get('must_not_do'):
            eval_dimensions.insert(0, {
                "name": "Requirement Compliance",
                "description": "Checks if output follows all specified must-do and must-not-do requirements",
                "check": "Verify all required behaviors present and forbidden behaviors absent",
                "weight": 1.5,
                "from_structured_requirements": True
            })
        
        # Add success criteria dimension
        if structured_requirements.get('success_criteria'):
            eval_dimensions.insert(0, {
                "name": "Success Criteria",
                "description": f"Evaluates: {', '.join(structured_requirements['success_criteria'][:3])}",
                "check": "Measure against defined success metrics",
                "weight": 1.5,
                "from_structured_requirements": True
            })

    # For multi-type prompts, add dimensions from secondary types
    if multi_label_result.is_multi_type:
        secondary_types = multi_label_result.get_secondary_types(threshold=0.35)
        for secondary_type in secondary_types[:2]:  # Max 2 secondary types
            secondary_dims = generate_eval_dimensions(secondary_type, dna)
            # Add unique dimensions not already present
            existing_names = {d["name"] for d in eval_dimensions}
            for dim in secondary_dims:
                if dim["name"] not in existing_names:
                    dim["from_secondary_type"] = secondary_type.value
                    eval_dimensions.append(dim)

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
        suggested_test_categories=test_categories,
        multi_label_result=multi_label_result,
        confidence_info=confidence_info
    )


def analysis_to_dict(analysis: PromptAnalysis) -> Dict[str, Any]:
    """Convert PromptAnalysis to dictionary for JSON serialization"""
    result = {
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

    # Add multi-label typing results if available
    if analysis.multi_label_result:
        mlr = analysis.multi_label_result
        result["multi_label_typing"] = {
            "primary_type": mlr.primary_type.value,
            "primary_confidence": mlr.primary_confidence,
            "is_multi_type": mlr.is_multi_type,
            "type_composition": mlr.type_composition,
            "all_labels": [
                {
                    "type": label.type.value,
                    "confidence": label.confidence,
                    "indicators_found": label.indicators_found,
                    "indicator_count": label.indicator_count
                }
                for label in mlr.all_labels
            ]
        }
    
    # Add confidence information for score reliability
    if analysis.confidence_info:
        result["confidence_info"] = {
            "confidence_level": analysis.confidence_info.get("confidence_level", "unknown"),
            "confidence_note": analysis.confidence_info.get("confidence_note", ""),
            "total_indicators": analysis.confidence_info.get("total_indicators", 0),
            "scoring_method": "earned",  # Document that we use earned scoring (start at 0)
            "score_interpretation": _get_score_interpretation(analysis.quality_score)
        }

    return result


def _get_score_interpretation(score: float) -> str:
    """Provide human-readable interpretation of quality score"""
    if score >= 8.5:
        return "Production-ready: Comprehensive, well-structured, follows best practices"
    elif score >= 7.0:
        return "Good quality: Solid foundation with minor improvements possible"
    elif score >= 5.0:
        return "Functional: Works but has significant room for improvement"
    elif score >= 3.0:
        return "Weak: Missing critical elements, needs substantial work"
    else:
        return "Poor: Fundamentally incomplete or flawed, requires major revision"


# =============================================================================
# SECTION: SEMANTIC VALIDATION - Detect contradictions, vague terms, issues
# =============================================================================

# Vague terms that should be made specific
# Note: Common rubric terms (acceptable, good, poor) are excluded as they're often
# intentionally used in scoring contexts with clear ordinal meaning
VAGUE_TERMS = [
    ("good quality", "Specify measurable quality thresholds (e.g., '≥90% accuracy')"),
    ("relevant information", "Define relevance criteria explicitly"),
    ("suitable for", "Specify what makes something 'suitable'"),
    ("properly formatted", "Define proper format with specific examples"),
    ("correctly implemented", "Add verification criteria for correctness"),
    ("reasonable time", "Quantify what is 'reasonable' (e.g., 'within 5 seconds')"),
    ("adequate coverage", "Specify minimum coverage requirements"),
    ("as needed", "Define triggers and conditions explicitly"),
    ("when necessary", "Specify the conditions that make it necessary"),
    ("if applicable", "List the cases where it applies"),
    ("etc.", "List all items explicitly instead of using 'etc.'"),
    ("and so on", "Enumerate all cases explicitly"),
    ("similar to", "Define similarity criteria"),
    ("optimal solution", "Specify optimization criteria and constraints"),
    ("efficient enough", "Define efficiency metrics"),
]

# Contradiction patterns to detect
# These are tuned to avoid false positives from normal rule lists
CONTRADICTION_PATTERNS = [
    # Always X and Never X about the SAME thing (not just nearby)
    # Requires same subject to be a real contradiction
    (r'\b(always|must always)\s+(\w+)\b[^.]*\b(never|must never|do not)\s+\2\b', "Direct contradiction: same action marked as 'always' and 'never'"),
    (r'\b(never|must never)\s+(\w+)\b[^.]*\b(always|must always)\s+\2\b', "Direct contradiction: same action marked as 'never' and 'always'"),
    # All X and No X about the same thing
    (r'\b(all|every)\s+(\w+)\b[^.]*\b(no|none of the)\s+\2\b', "Contradiction: 'all X' and 'no X'"),
    # Specific numbers that directly conflict
    (r'(?:must be|exactly|precisely)\s+(\d+)\b[^.]*(?:must be|exactly|precisely)\s+(?!\1)(\d+)\b', "Numerical contradiction: conflicting exact values"),
    # Include and exclude same thing
    (r'\b(include|add)\s+(\w+)\b[^.]*\b(exclude|remove|omit)\s+\2\b', "Contradiction: include and exclude same item"),
]

# Ambiguity patterns
AMBIGUITY_PATTERNS = [
    (r'\b(it|this|that|these|those)\b\s+(?:should|must|will|can)', "Ambiguous pronoun reference - clarify what 'it/this/that' refers to"),
    (r'\b(some|many|few|several)\b', "Vague quantifier - use specific numbers or percentages"),
    (r'\b(soon|quickly|fast|slow|long|short)\b', "Vague time/speed reference - use specific durations"),
    (r'\b(large|small|big|little|high|low)\b\s+(?:amount|number|value)', "Vague magnitude - use specific thresholds"),
]


@dataclass
class SemanticIssue:
    """A semantic issue found in the prompt"""
    severity: str  # "critical", "warning", "suggestion"
    category: str  # "contradiction", "vague_term", "ambiguity", "missing_spec"
    description: str
    location: str  # The text snippet where issue was found
    recommendation: str  # Specific fix recommendation


@dataclass
class SemanticValidationResult:
    """Result of semantic validation"""
    is_valid: bool  # True if no critical issues
    critical_issues: List[SemanticIssue]
    warnings: List[SemanticIssue]
    suggestions: List[SemanticIssue]
    overall_semantic_score: float  # 0-100
    actionable_fixes: List[Dict[str, str]]  # Priority-ordered fixes


def validate_prompt_semantics(prompt_text: str) -> SemanticValidationResult:
    """
    Perform semantic validation on a prompt.

    Checks for:
    1. Contradictions (always vs never, etc.)
    2. Vague terms that need specificity
    3. Ambiguous references
    4. Missing specifications

    Returns actionable recommendations to fix issues.
    """
    issues: List[SemanticIssue] = []
    prompt_lower = prompt_text.lower()

    # 1. Check for vague terms
    for vague_term, recommendation in VAGUE_TERMS:
        if vague_term in prompt_lower:
            # Find the context
            pattern = re.compile(rf'.{{0,50}}\b{re.escape(vague_term)}\b.{{0,50}}', re.IGNORECASE)
            matches = pattern.findall(prompt_text)
            for match in matches[:2]:  # Limit to 2 instances per term
                issues.append(SemanticIssue(
                    severity="warning",
                    category="vague_term",
                    description=f"Vague term '{vague_term}' found",
                    location=match.strip(),
                    recommendation=recommendation
                ))

    # 2. Check for contradictions
    for pattern, description in CONTRADICTION_PATTERNS:
        matches = re.findall(pattern, prompt_lower, re.IGNORECASE | re.DOTALL)
        if matches:
            # Find actual text
            context_match = re.search(pattern, prompt_text, re.IGNORECASE | re.DOTALL)
            if context_match:
                issues.append(SemanticIssue(
                    severity="critical",
                    category="contradiction",
                    description=description,
                    location=context_match.group(0)[:100],
                    recommendation="Resolve the contradiction by choosing one approach or adding conditional logic"
                ))

    # 3. Check for ambiguous references
    for pattern, description in AMBIGUITY_PATTERNS:
        matches = re.finditer(pattern, prompt_text, re.IGNORECASE)
        for match in list(matches)[:3]:  # Limit to 3 per pattern
            start = max(0, match.start() - 30)
            end = min(len(prompt_text), match.end() + 30)
            context = prompt_text[start:end]
            issues.append(SemanticIssue(
                severity="warning",
                category="ambiguity",
                description=description,
                location=context.strip(),
                recommendation="Replace with specific, measurable criteria"
            ))

    # 4. Check for missing critical specifications based on prompt type
    missing_specs = _check_missing_specifications(prompt_text)
    for spec in missing_specs:
        issues.append(SemanticIssue(
            severity=spec["severity"],
            category="missing_spec",
            description=spec["description"],
            location="[Not found in prompt]",
            recommendation=spec["recommendation"]
        ))

    # Categorize issues
    critical = [i for i in issues if i.severity == "critical"]
    warnings = [i for i in issues if i.severity == "warning"]
    suggestions = [i for i in issues if i.severity == "suggestion"]

    # Calculate semantic score
    # Start at 100, deduct for issues
    score = 100.0
    score -= len(critical) * 20  # Critical issues are severe
    score -= len(warnings) * 5   # Warnings are moderate
    score -= len(suggestions) * 2  # Suggestions are minor
    score = max(0, score)

    # Generate prioritized actionable fixes
    actionable_fixes = _generate_actionable_fixes(critical + warnings + suggestions)

    return SemanticValidationResult(
        is_valid=len(critical) == 0,
        critical_issues=critical,
        warnings=warnings,
        suggestions=suggestions,
        overall_semantic_score=score,
        actionable_fixes=actionable_fixes
    )


def _check_missing_specifications(prompt_text: str) -> List[Dict[str, str]]:
    """Check for missing critical specifications based on prompt content."""
    missing = []
    prompt_lower = prompt_text.lower()

    # Check if scoring is mentioned but no scale defined
    if any(word in prompt_lower for word in ["score", "rate", "rating", "evaluate"]):
        if not re.search(r'\d+\s*(?:to|-)\s*\d+|\d+\s*point|scale|1-5|1-10|0-100', prompt_lower):
            missing.append({
                "severity": "warning",
                "description": "Scoring mentioned but no scale defined",
                "recommendation": "Add explicit scoring scale (e.g., '1-5 where 5 is best')"
            })

    # Check if JSON output expected but no schema provided
    if "json" in prompt_lower:
        if not re.search(r'\{[^}]*"[^"]+"\s*:', prompt_text):
            missing.append({
                "severity": "warning",
                "description": "JSON output expected but no schema example provided",
                "recommendation": "Add a JSON schema example showing expected structure"
            })

    # Check for role definition
    if not any(phrase in prompt_lower for phrase in ["you are", "act as", "your role", "as a", "as an"]):
        missing.append({
            "severity": "suggestion",
            "description": "No explicit role definition found",
            "recommendation": "Add a clear role definition (e.g., 'You are a helpful assistant that...')"
        })

    # Check for output format specification
    if not any(phrase in prompt_lower for phrase in ["format", "output", "respond", "return", "provide"]):
        missing.append({
            "severity": "warning",
            "description": "No output format specification found",
            "recommendation": "Specify the expected output format (JSON, markdown, plain text, etc.)"
        })

    # Check for constraints
    if not any(phrase in prompt_lower for phrase in ["must", "should", "do not", "never", "always", "required"]):
        missing.append({
            "severity": "suggestion",
            "description": "No explicit constraints or requirements found",
            "recommendation": "Add constraints to define boundaries (e.g., 'Do not include personal opinions')"
        })

    return missing


def _generate_actionable_fixes(issues: List[SemanticIssue]) -> List[Dict[str, str]]:
    """Generate prioritized, actionable fixes from issues."""
    fixes = []

    # Group by category and prioritize
    critical_fixes = [i for i in issues if i.severity == "critical"]
    warning_fixes = [i for i in issues if i.severity == "warning"]
    suggestion_fixes = [i for i in issues if i.severity == "suggestion"]

    priority = 1

    # Critical fixes first
    for issue in critical_fixes[:5]:  # Top 5 critical
        fixes.append({
            "priority": priority,
            "severity": "CRITICAL",
            "issue": issue.description,
            "location": issue.location[:80] + "..." if len(issue.location) > 80 else issue.location,
            "fix": issue.recommendation,
            "impact": "High - Must fix before deployment"
        })
        priority += 1

    # Then warnings
    for issue in warning_fixes[:5]:  # Top 5 warnings
        fixes.append({
            "priority": priority,
            "severity": "WARNING",
            "issue": issue.description,
            "location": issue.location[:80] + "..." if len(issue.location) > 80 else issue.location,
            "fix": issue.recommendation,
            "impact": "Medium - Should fix for better reliability"
        })
        priority += 1

    # Then suggestions
    for issue in suggestion_fixes[:3]:  # Top 3 suggestions
        fixes.append({
            "priority": priority,
            "severity": "SUGGESTION",
            "issue": issue.description,
            "location": issue.location[:80] + "..." if len(issue.location) > 80 else issue.location,
            "fix": issue.recommendation,
            "impact": "Low - Nice to have"
        })
        priority += 1

    return fixes


def get_enhanced_analysis(prompt_text: str) -> Dict[str, Any]:
    """
    Get enhanced prompt analysis including semantic validation.

    This combines the standard analysis with semantic validation
    to provide a complete quality assessment.
    """
    # Run standard analysis
    standard_analysis = analyze_prompt(prompt_text)
    standard_dict = analysis_to_dict(standard_analysis)

    # Run semantic validation
    semantic_result = validate_prompt_semantics(prompt_text)

    # Combine results
    standard_dict["semantic_validation"] = {
        "is_valid": semantic_result.is_valid,
        "semantic_score": semantic_result.overall_semantic_score,
        "critical_issues_count": len(semantic_result.critical_issues),
        "warnings_count": len(semantic_result.warnings),
        "suggestions_count": len(semantic_result.suggestions),
        "critical_issues": [
            {
                "category": i.category,
                "description": i.description,
                "location": i.location[:100],
                "recommendation": i.recommendation
            }
            for i in semantic_result.critical_issues
        ],
        "warnings": [
            {
                "category": i.category,
                "description": i.description,
                "recommendation": i.recommendation
            }
            for i in semantic_result.warnings[:5]  # Limit to 5
        ],
        "actionable_fixes": semantic_result.actionable_fixes
    }

    # Compute combined quality score
    # Weight: 50% structure/clarity, 50% semantic (balanced approach)
    # Semantic validation is equally important as structural quality
    structural_score = standard_analysis.quality_score
    semantic_score = semantic_result.overall_semantic_score / 10  # Convert to 0-10

    # Bonus for having BOTH good structure AND good semantics
    if structural_score >= 6.0 and semantic_score >= 8.0:
        bonus = 0.5  # Bonus for well-rounded prompt
    else:
        bonus = 0.0

    combined_score = (structural_score * 0.5) + (semantic_score * 0.5) + bonus

    standard_dict["combined_quality_score"] = round(combined_score, 2)
    standard_dict["quality_assessment"] = {
        "structural_score": structural_score,
        "semantic_score": round(semantic_score, 2),
        "combined_score": round(combined_score, 2),
        "is_production_ready": combined_score >= 7.5 and semantic_result.is_valid,
        "recommendation": _get_quality_recommendation(combined_score, semantic_result.is_valid)
    }

    return standard_dict


def _get_quality_recommendation(score: float, is_semantically_valid: bool) -> str:
    """Get actionable recommendation based on quality assessment."""
    if not is_semantically_valid:
        return "BLOCK: Resolve critical semantic issues before deployment"
    elif score >= 8.5:
        return "DEPLOY: Prompt is production-ready"
    elif score >= 7.0:
        return "REVIEW: Minor improvements recommended before deployment"
    elif score >= 5.0:
        return "IMPROVE: Address key issues before deployment"
    else:
        return "REWRITE: Prompt needs significant revision"
