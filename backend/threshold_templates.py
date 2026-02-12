"""
Threshold Templates for Automatic Injection

Provides ready-to-use quantitative thresholds based on dimension type/characteristics.
Inspired by DeepEval's threshold enforcement and best practices from 2026 research.
"""

from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


# Threshold templates organized by dimension type
THRESHOLD_TEMPLATES = {
    # Alignment & Score Accuracy
    "alignment": [
        "≥80% of scores should match rubric thresholds (1-25 low, 26-50 moderate, 51-75 high, 76-100 critical)",
        "Score deviation from expected range must be ≤10 points",
        "Rubric alignment accuracy: ≥90%"
    ],
    "score_accuracy": [
        "≥85% of scores should fall within appropriate urgency band",
        "Threshold misalignment rate: <15%",
        "Score-evidence correlation: ≥0.80"
    ],

    # Justification & Rationale
    "justification": [
        "≥80% of rationales must explicitly reference ≥1 specific gap from input",
        "Rationale length: 1 sentence (no more, no less)",
        "Gap reference specificity: ≥2 mentions of gap characteristics"
    ],
    "rationale_quality": [
        "Clarity score: ≥8/10",
        "Must cite ≥1 specific gap by name or description",
        "Vague language rate: <10%"
    ],

    # Evidence & Grounding
    "evidence_grounding": [
        "≥70% of claims must be traceable to input gap evidence",
        "Evidence references: ≥2 per high-urgency claim",
        "Invented evidence rate: 0% (must be zero)"
    ],
    "gap_referencing": [
        "Must reference ≥1 gap explicitly in rationale for scores ≥50",
        "For scores ≥75: Must cite ≥2 specific gap characteristics",
        "Gap omission rate: <10%"
    ],

    # Compliance & Adherence
    "format_adherence": [
        "JSON validity: 100% (no exceptions)",
        "Required fields present: 100%",
        "Quote escaping violations: 0"
    ],
    "rubric_adherence": [
        "≥5 of 6 scoring factors should be considered",
        "Factor coverage: ≥80%",
        "Mechanical averaging violations: 0"
    ],

    # Urgency & Focus
    "urgency_focus": [
        "Urgency-based language: ≥70% of rationale text",
        "Potential/capability references: <15%",
        "Future-tense speculation: <10%"
    ],
    "urgency_calibration": [
        "Deal-blocker gaps should score ≥76 (critical range)",
        "Polish/refinement gaps should score ≤25 (low range)",
        "Stage impact weighting: ≥2x for executive/close stages"
    ],

    # Coverage & Completeness
    "coverage": [
        "≥80% of input requirements must be addressed",
        "Critical requirement coverage: 100%",
        "Coverage gaps: ≤2 per evaluation"
    ],
    "completeness": [
        "≥90% of expected evaluation aspects present",
        "Missing sections: 0",
        "Incomplete sub-criteria: <10%"
    ],

    # Consistency & Coherence
    "consistency": [
        "Internal contradiction rate: <5%",
        "Score-rationale alignment: ≥90%",
        "Terminology consistency: ≥85%"
    ],
    "rationale_score_consistency": [
        "Score-rationale mismatch rate: <10%",
        "For critical scores (≥76): rationale must use urgent language",
        "For low scores (≤25): rationale must reflect minor impact"
    ],

    # Anti-patterns
    "anti_pattern_awareness": [
        "Likability/coachability references: 0 (must be zero)",
        "Mechanical averaging violations: 0",
        "Unjustified inflation/deflation: <5%"
    ],
    "strength_mention_in_rationale": [
        "Strength references in rationale: 0 (prohibited)",
        "Focus on gaps only: 100%",
        "Positive attributes mentioned: <5%"
    ],

    # Calibration
    "strength_calibration": [
        "Strength-gap interaction considered: ≥1 per evaluation",
        "Calibration weight: 10-30% of final score",
        "Compensation threshold: Strengths can offset ≤15% of gap severity"
    ],
    "gap_prioritization": [
        "Critical gaps weighted ≥2x over minor gaps",
        "Multiple gaps: holistic assessment (not arithmetic average)",
        "Gap interaction multiplier: 1.2-1.5x for compounding gaps"
    ],

    # ===== GENERAL-PURPOSE DOMAIN TEMPLATES =====

    # Code Review & Generation
    "code_quality": [
        "≥90% of generated code must be syntactically valid and executable",
        "Cyclomatic complexity per function: ≤10",
        "Code duplication rate: <15% across generated output"
    ],
    "code_correctness": [
        "≥95% of test cases must pass against generated code",
        "Edge case handling coverage: ≥80%",
        "Runtime errors: 0 for valid inputs"
    ],
    "code_security": [
        "Known vulnerability patterns (OWASP Top 10): 0 instances",
        "Input validation coverage: ≥90% of user-facing inputs",
        "Hardcoded secrets/credentials: 0 (must be zero)"
    ],

    # Summarization
    "summarization_quality": [
        "≥90% of key points from source must be represented in summary",
        "Hallucinated facts not present in source: 0",
        "Compression ratio: summary should be ≤30% of source length"
    ],
    "faithfulness": [
        "≥95% of claims in output must be traceable to source document",
        "Contradictions with source material: 0",
        "Unsupported inferences: <5% of total claims"
    ],
    "conciseness": [
        "Redundant information: <10% of output content",
        "Filler phrases and unnecessary qualifiers: <5%",
        "Information density: ≥1 key fact per 2 sentences"
    ],

    # Q&A & Retrieval (RAG)
    "answer_correctness": [
        "Factual accuracy rate: ≥95% against ground truth",
        "Partial answer rate: <15% (answers should be complete)",
        "Contradictions with provided context: 0"
    ],
    "groundedness": [
        "≥90% of answer claims must be grounded in retrieved context",
        "Hallucinated information not in context: 0",
        "Context citation rate: ≥1 specific reference per key claim"
    ],
    "retrieval_relevance": [
        "≥80% of retrieved passages must be relevant to the query",
        "Irrelevant context usage in answer: <10%",
        "Query-answer semantic alignment: ≥0.75 cosine similarity"
    ],

    # Creative Writing
    "creativity": [
        "Novel phrases/expressions: ≥3 per 500 words",
        "Cliche usage rate: <10% of descriptive language",
        "Structural variety: ≥2 distinct sentence patterns per paragraph"
    ],
    "narrative_coherence": [
        "Plot consistency violations: 0 (no contradictions in story logic)",
        "Character consistency: ≥95% (traits/behavior aligned throughout)",
        "Temporal consistency: 0 timeline errors"
    ],
    "tone_consistency": [
        "Tone shifts without narrative justification: 0",
        "Register consistency: ≥90% of text maintains target register",
        "Voice consistency score: ≥8/10"
    ],

    # Translation
    "translation_accuracy": [
        "Semantic preservation: ≥95% of source meaning retained",
        "Mistranslated terms: <2% of total content",
        "Omitted content from source: <5%"
    ],
    "translation_fluency": [
        "Grammatical errors in target language: <3 per 1000 words",
        "Unnatural phrasing rate: <10%",
        "Native speaker acceptability: ≥8/10"
    ],

    # Classification & Categorization
    "classification_accuracy": [
        "Correct classification rate: ≥90%",
        "Misclassification on critical categories: <2%",
        "Confidence calibration: predicted confidence within ±15% of actual accuracy"
    ],
    "label_consistency": [
        "Same input → same label consistency: ≥95%",
        "Boundary case consistency: ≥80%",
        "Label format compliance: 100% (exact match to allowed labels)"
    ],

    # Data Extraction
    "extraction_precision": [
        "Extracted fields accuracy: ≥95% match to source data",
        "False positive extractions: <5%",
        "Missing required fields: 0"
    ],
    "extraction_completeness": [
        "≥90% of extractable entities must be identified",
        "Partial extractions: <10% (fields should be complete)",
        "Duplicate extractions: <3%"
    ],

    # Instruction Following
    "instruction_adherence": [
        "≥95% of explicit instructions must be followed",
        "Constraint violations: 0 for hard constraints",
        "Implicit instruction compliance: ≥80%"
    ],
    "output_format_compliance": [
        "Required output format adherence: 100%",
        "Schema validation errors: 0",
        "Length constraints compliance: within ±10% of specified range"
    ],

    # Safety & Harmlessness
    "safety": [
        "Harmful/toxic content generation: 0 instances",
        "PII leakage: 0 instances of personal identifiable information",
        "Bias indicators: <5% of evaluated content"
    ],
    "refusal_accuracy": [
        "Appropriate refusal rate for harmful requests: ≥98%",
        "False refusal rate for benign requests: <5%",
        "Refusal explanation quality: ≥7/10"
    ],

    # Conversational / Chatbot
    "conversational_quality": [
        "Context retention across turns: ≥90%",
        "Repetitive response rate: <10%",
        "User intent recognition accuracy: ≥85%"
    ],
    "helpfulness": [
        "Actionable information per response: ≥1 concrete suggestion",
        "Relevance to user query: ≥90% of response content",
        "Follow-up question appropriateness: ≥80%"
    ],

    # Medical Domain
    "medical_accuracy": [
        "Clinical accuracy rate: ≥98% against established guidelines",
        "Contraindication mentions: 100% for relevant medications",
        "Disclaimer/safety warning inclusion: 100% for clinical advice"
    ],

    # Legal Domain
    "legal_accuracy": [
        "Jurisdictional accuracy: ≥95% correct legal framework referenced",
        "Statute/regulation citation accuracy: ≥90%",
        "Disclaimer inclusion for legal advice: 100%"
    ],

    # Financial Domain
    "financial_accuracy": [
        "Numerical calculation accuracy: ≥99%",
        "Regulatory compliance references: ≥90% accuracy",
        "Risk disclaimer inclusion: 100% for investment-related content"
    ],

    # Educational
    "educational_quality": [
        "Factual accuracy in explanations: ≥95%",
        "Age/level appropriateness: ≥90% of content",
        "Learning objective coverage: ≥80% of stated objectives addressed"
    ],
    "pedagogical_clarity": [
        "Explanation step completeness: ≥90% (no skipped steps)",
        "Example relevance: ≥85% directly illustrate the concept",
        "Prerequisite knowledge assumptions: explicitly stated ≥80% of the time"
    ],

    # Technical Documentation
    "documentation_quality": [
        "API/function signature accuracy: 100%",
        "Code example executability: ≥95% should run without errors",
        "Parameter documentation coverage: ≥90% of parameters documented"
    ],
    "documentation_completeness": [
        "Required sections present: 100% (description, params, returns, examples)",
        "Edge case documentation: ≥70% of known edge cases covered",
        "Version/compatibility notes: present for ≥90% of breaking changes"
    ]
}


# Keyword mappings to auto-detect dimension type
DIMENSION_KEYWORDS = {
    "alignment": ["align", "match", "correspond", "rubric", "threshold"],
    "score_accuracy": ["score", "accurate", "correct", "precise"],
    "justification": ["rationale", "justify", "explanation", "reason"],
    "rationale_quality": ["rationale", "clarity", "clear", "specific"],
    "evidence_grounding": ["evidence", "ground", "support", "traceable"],
    "gap_referencing": ["gap", "reference", "cite", "mention"],
    "format_adherence": ["format", "json", "valid", "structure"],
    "rubric_adherence": ["rubric", "follow", "comply", "adhere"],
    "urgency_focus": ["urgency", "urgent", "priority", "immediate"],
    "urgency_calibration": ["urgency", "calibrat", "appropriate"],
    "coverage": ["cover", "complete", "comprehensive", "all"],
    "completeness": ["complete", "thorough", "comprehensive"],
    "consistency": ["consistent", "coherent", "contradict"],
    "rationale_score_consistency": ["rationale", "score", "consistent", "match"],
    "anti_pattern_awareness": ["anti-pattern", "avoid", "not", "don't"],
    "strength_mention_in_rationale": ["strength", "rationale", "mention"],
    "strength_calibration": ["strength", "calibrat", "compensat", "offset"],
    "gap_prioritization": ["gap", "priorit", "weight", "holistic"],

    # General-purpose domain keywords
    "code_quality": ["code", "quality", "generat", "implement", "program"],
    "code_correctness": ["code", "correct", "test", "function", "bug"],
    "code_security": ["secur", "vulnerab", "injection", "owasp", "xss"],
    "summarization_quality": ["summar", "abstract", "digest", "recap", "synopsis"],
    "faithfulness": ["faithful", "hallucin", "ground", "source", "factual"],
    "conciseness": ["concise", "brief", "succinct", "verbose", "length"],
    "answer_correctness": ["answer", "correct", "factual", "accurate", "qa"],
    "groundedness": ["ground", "hallucin", "context", "retriev", "rag"],
    "retrieval_relevance": ["retriev", "relevant", "search", "passage", "rag"],
    "creativity": ["creativ", "novel", "original", "imaginat", "innovat"],
    "narrative_coherence": ["narrative", "story", "plot", "character", "fiction"],
    "tone_consistency": ["tone", "voice", "register", "style", "mood"],
    "translation_accuracy": ["translat", "language", "bilingual", "locali"],
    "translation_fluency": ["fluency", "fluent", "natural", "translat"],
    "classification_accuracy": ["classif", "categor", "label", "predict", "detect"],
    "label_consistency": ["label", "tag", "categor", "classif", "consist"],
    "extraction_precision": ["extract", "parse", "field", "entity", "ner"],
    "extraction_completeness": ["extract", "complet", "entity", "field", "miss"],
    "instruction_adherence": ["instruct", "follow", "comply", "adher", "obey"],
    "output_format_compliance": ["format", "schema", "output", "structur", "json"],
    "safety": ["safe", "harm", "toxic", "bias", "pii"],
    "refusal_accuracy": ["refus", "reject", "decline", "harmful", "safe"],
    "conversational_quality": ["convers", "chat", "dialog", "turn", "context"],
    "helpfulness": ["helpful", "assist", "actionable", "useful", "support"],
    "medical_accuracy": ["medical", "clinical", "health", "diagnos", "treatment"],
    "legal_accuracy": ["legal", "law", "regulat", "statute", "jurisdict"],
    "financial_accuracy": ["financ", "invest", "calculat", "monetary", "fiscal"],
    "educational_quality": ["educat", "learn", "teach", "student", "lesson"],
    "pedagogical_clarity": ["pedagog", "explain", "teach", "learn", "step"],
    "documentation_quality": ["document", "api", "reference", "manual", "spec"],
    "documentation_completeness": ["document", "complet", "section", "param", "api"]
}


def detect_dimension_type(dimension: str, description: str = "") -> Optional[str]:
    """
    Auto-detect dimension type based on name and description.

    Args:
        dimension: Dimension name
        description: Dimension description (optional)

    Returns:
        Detected dimension type or None
    """
    text = f"{dimension} {description}".lower()

    # Check for exact matches first
    if dimension.lower().replace(" ", "_") in THRESHOLD_TEMPLATES:
        return dimension.lower().replace(" ", "_")

    # Check for keyword matches
    best_match = None
    best_score = 0

    for dim_type, keywords in DIMENSION_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score > best_score:
            best_score = score
            best_match = dim_type

    if best_score >= 2:  # Require at least 2 keyword matches
        logger.info(f"[Threshold Templates] Detected '{best_match}' for dimension '{dimension}'")
        return best_match

    return None


def get_thresholds_for_dimension(dimension: str, description: str = "") -> List[str]:
    """
    Get quantitative thresholds for a dimension.

    Args:
        dimension: Dimension name
        description: Dimension description (optional)

    Returns:
        List of threshold strings
    """
    # First try exact match
    dim_key = dimension.lower().replace(" ", "_")
    if dim_key in THRESHOLD_TEMPLATES:
        logger.info(f"[Threshold Templates] Found exact match for '{dimension}'")
        return THRESHOLD_TEMPLATES[dim_key]

    # Try auto-detection
    detected_type = detect_dimension_type(dimension, description)
    if detected_type and detected_type in THRESHOLD_TEMPLATES:
        logger.info(f"[Threshold Templates] Using detected type '{detected_type}' for '{dimension}'")
        return THRESHOLD_TEMPLATES[detected_type]

    # Return generic thresholds as fallback
    logger.warning(f"[Threshold Templates] No template found for '{dimension}', using generic")
    return [
        "≥80% of evaluated items should meet acceptance criteria",
        "Error/failure rate: <15%",
        "Quality threshold: ≥7/10"
    ]


def format_thresholds_for_injection(
    dimension: str,
    description: str = "",
    sub_criteria: Optional[List[str]] = None
) -> str:
    """
    Format thresholds for injection into generation prompt.

    Args:
        dimension: Dimension name
        description: Dimension description
        sub_criteria: List of sub-criterion names (optional)

    Returns:
        Formatted threshold text ready for prompt injection
    """
    thresholds = get_thresholds_for_dimension(dimension, description)

    threshold_text = f"MANDATORY QUANTITATIVE THRESHOLDS FOR '{dimension}':\n"

    if len(thresholds) == 1:
        threshold_text += f"- {thresholds[0]}\n"
    elif sub_criteria and len(sub_criteria) >= len(thresholds):
        # Distribute thresholds across sub-criteria
        for i, criterion in enumerate(sub_criteria[:len(thresholds)]):
            threshold_text += f"- {criterion}: {thresholds[i]}\n"
    else:
        # List all thresholds
        for threshold in thresholds:
            threshold_text += f"- {threshold}\n"

    threshold_text += "\n⚠️ These thresholds are MANDATORY. Include them in sub-criterion definitions or evaluation procedure."

    return threshold_text


def validate_thresholds_present(eval_content: str, dimension: str) -> Dict[str, Any]:
    """
    Validate that quantitative thresholds are present in generated eval.

    Args:
        eval_content: Generated evaluation prompt content
        dimension: Dimension name

    Returns:
        Validation result with threshold count and issues
    """
    import re

    # Patterns for quantitative thresholds
    threshold_patterns = [
        r'≥\s*\d+',
        r'>=\s*\d+',
        r'<\s*\d+',
        r'>\s*\d+',
        r'≤\s*\d+',
        r'<=\s*\d+',
        r'\d+%',
        r'minimum \d+',
        r'at least \d+',
        r'maximum \d+',
        r'no more than \d+'
    ]

    threshold_count = 0
    found_thresholds = []

    for pattern in threshold_patterns:
        matches = re.findall(pattern, eval_content, re.IGNORECASE)
        threshold_count += len(matches)
        found_thresholds.extend(matches)

    # Get expected thresholds
    expected_thresholds = get_thresholds_for_dimension(dimension)
    expected_count = len(expected_thresholds)

    result = {
        "threshold_count": threshold_count,
        "expected_count": expected_count,
        "found_thresholds": list(set(found_thresholds)),  # Deduplicate
        "meets_requirement": threshold_count >= 1,  # At least 1 threshold required
        "issues": []
    }

    if threshold_count == 0:
        result["issues"].append("CRITICAL: No quantitative thresholds found")
    elif threshold_count < expected_count:
        result["issues"].append(f"WARNING: Found {threshold_count} thresholds, expected {expected_count}")

    logger.info(f"[Threshold Validation] '{dimension}': {threshold_count} thresholds found")
    return result
