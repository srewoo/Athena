"""
Intelligent Dimension Detector

Automatically analyzes system prompts and requirements to determine
which evaluation dimensions are needed.

Now supports both:
1. Legacy regex-based detection
2. Intelligent detection using ExtractedRequirements from LLM analysis
"""

import logging
import re
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from intelligent_requirements_extractor import ExtractedRequirements

logger = logging.getLogger(__name__)


@dataclass
class DetectedDimension:
    """A dimension that was auto-detected from requirements"""
    name: str
    weight: float
    is_critical: bool
    min_pass_score: float
    detection_reason: str
    evaluator_type: str  # Class name of evaluator to use


class DimensionDetector:
    """
    Smart detector that analyzes system prompts and requirements
    to determine what needs to be evaluated
    """

    def __init__(self):
        self.detection_patterns = self._build_detection_patterns()

    def detect_dimensions(
        self,
        system_prompt: str,
        requirements: str,
        use_case: str,
        structured_requirements: Optional[Dict[str, Any]] = None
    ) -> List[DetectedDimension]:
        """
        Analyze inputs and detect which evaluation dimensions are needed

        Returns list of dimensions with suggested weights and criticality
        """
        dimensions = []

        # Combine all text for analysis
        combined_text = f"{system_prompt}\n{requirements}\n{use_case}".lower()

        # Check for structured output requirement
        schema_dim = self._detect_schema_requirement(
            system_prompt, requirements, structured_requirements
        )
        if schema_dim:
            dimensions.append(schema_dim)

        # Check for safety requirements
        safety_dim = self._detect_safety_requirement(combined_text)
        if safety_dim:
            dimensions.append(safety_dim)

        # Check for completeness requirements
        completeness_dim = self._detect_completeness_requirement(
            system_prompt, requirements, structured_requirements
        )
        if completeness_dim:
            dimensions.append(completeness_dim)

        # Check for accuracy requirements
        accuracy_dim = self._detect_accuracy_requirement(combined_text)
        if accuracy_dim:
            dimensions.append(accuracy_dim)

        # Check for actionability requirements
        actionability_dim = self._detect_actionability_requirement(combined_text)
        if actionability_dim:
            dimensions.append(actionability_dim)

        # Check for relevance requirements
        relevance_dim = self._detect_relevance_requirement(combined_text)
        if relevance_dim:
            dimensions.append(relevance_dim)

        # Check for tone requirements
        tone_dim = self._detect_tone_requirement(system_prompt, requirements)
        if tone_dim:
            dimensions.append(tone_dim)

        # Normalize weights
        dimensions = self._normalize_weights(dimensions)

        logger.info(f"Detected {len(dimensions)} evaluation dimensions: {[d.name for d in dimensions]}")

        return dimensions

    def detect_dimensions_from_extracted(
        self,
        extracted_requirements: 'ExtractedRequirements'
    ) -> List[DetectedDimension]:
        """
        Detect dimensions using intelligent LLM-extracted requirements.
        This is more accurate than regex-based detection.

        Args:
            extracted_requirements: Requirements extracted by IntelligentRequirementsExtractor

        Returns:
            List of DetectedDimension with smart weights and criticality
        """

        dimensions = []

        logger.info(f"Detecting dimensions from intelligent extraction (domain={extracted_requirements.domain}, risk={extracted_requirements.risk_level})")

        # 1. Schema/Format Compliance
        if extracted_requirements.output_format.get("type") in ["json", "structured", "xml", "yaml"]:
            dimensions.append(DetectedDimension(
                name="schema_compliance",
                weight=0.20,
                is_critical=True,
                min_pass_score=4.0,
                detection_reason=f"Structured output required: {extracted_requirements.output_format.get('type')}",
                evaluator_type="SchemaValidator"
            ))

        # 2. Safety (always included, weight varies by domain and risk)
        safety_weight = self._calculate_safety_weight(extracted_requirements)
        is_safety_critical = extracted_requirements.risk_level in ["high", "critical"] or \
                             "safety" in extracted_requirements.critical_dimensions

        dimensions.append(DetectedDimension(
            name="safety",
            weight=safety_weight,
            is_critical=is_safety_critical,
            min_pass_score=4.0 if is_safety_critical else 3.0,
            detection_reason=f"Domain: {extracted_requirements.domain}, Risk: {extracted_requirements.risk_level}",
            evaluator_type="DomainSafetyChecker" if extracted_requirements.domain in ["medical", "financial", "legal"] else "SafetyChecker"
        ))

        # 3. Completeness (if must_do items specified)
        if extracted_requirements.must_do:
            dimensions.append(DetectedDimension(
                name="completeness",
                weight=0.15,
                is_critical="completeness" in extracted_requirements.critical_dimensions,
                min_pass_score=3.0,
                detection_reason=f"{len(extracted_requirements.must_do)} required items detected",
                evaluator_type="CompletenessChecker"
            ))

        # 4. Accuracy (almost always needed, weight varies by domain)
        accuracy_weight = self._calculate_accuracy_weight(extracted_requirements)
        is_accuracy_critical = "accuracy" in extracted_requirements.critical_dimensions or \
                               extracted_requirements.domain in ["medical", "financial", "legal"]

        dimensions.append(DetectedDimension(
            name="accuracy",
            weight=accuracy_weight,
            is_critical=is_accuracy_critical,
            min_pass_score=4.0 if is_accuracy_critical else 3.0,
            detection_reason=f"Priority: {extracted_requirements.quality_priorities[0] if extracted_requirements.quality_priorities else 'default'}",
            evaluator_type="DomainAccuracyEvaluator" if extracted_requirements.domain in ["medical", "financial", "legal"] else "AccuracyEvaluator"
        ))

        # 5. Actionability (if in quality priorities or use case suggests it)
        if "actionability" in extracted_requirements.quality_priorities or \
           any(word in extracted_requirements.primary_function.lower() for word in ["help", "guide", "instruct", "assist"]):
            dimensions.append(DetectedDimension(
                name="actionability",
                weight=0.15,
                is_critical=False,
                min_pass_score=2.5,
                detection_reason="Actionability detected in quality priorities or function",
                evaluator_type="ActionabilityEvaluator"
            ))

        # 6. Relevance (almost always needed)
        dimensions.append(DetectedDimension(
            name="relevance",
            weight=0.15,
            is_critical=False,
            min_pass_score=2.5,
            detection_reason="Default relevance check",
            evaluator_type="RelevanceEvaluator"
        ))

        # 7. Tone (if specific tone requirements)
        if extracted_requirements.tone != "professional" or extracted_requirements.style_requirements:
            dimensions.append(DetectedDimension(
                name="tone",
                weight=0.10,
                is_critical=False,
                min_pass_score=2.0,
                detection_reason=f"Tone requirements: {extracted_requirements.tone}",
                evaluator_type="ToneEvaluator"
            ))

        # 8. Domain-specific evaluators
        domain_dims = self._add_domain_specific_dimensions(extracted_requirements)
        dimensions.extend(domain_dims)

        # Normalize weights
        dimensions = self._normalize_weights(dimensions)

        logger.info(
            f"Detected {len(dimensions)} dimensions using intelligent extraction: "
            f"{[d.name for d in dimensions]}"
        )

        return dimensions

    def _calculate_safety_weight(self, extracted: 'ExtractedRequirements') -> float:
        """Calculate appropriate safety weight based on domain and risk"""

        base_weight = 0.10

        # High-risk domains get higher safety weight
        if extracted.domain in ["medical", "legal", "financial"]:
            base_weight = 0.25
        elif extracted.risk_level == "critical":
            base_weight = 0.30
        elif extracted.risk_level == "high":
            base_weight = 0.20
        elif extracted.risk_level == "medium":
            base_weight = 0.15

        # Boost if safety is in critical dimensions
        if "safety" in extracted.critical_dimensions:
            base_weight += 0.10

        return min(base_weight, 0.40)  # Cap at 40%

    def _calculate_accuracy_weight(self, extracted: 'ExtractedRequirements') -> float:
        """Calculate appropriate accuracy weight based on priorities and domain"""

        base_weight = 0.30

        # Accuracy is first priority?
        if extracted.quality_priorities and extracted.quality_priorities[0] == "accuracy":
            base_weight = 0.35

        # High-precision domains
        if extracted.domain in ["medical", "financial", "legal", "technical"]:
            base_weight = max(base_weight, 0.35)

        # Critical dimension?
        if "accuracy" in extracted.critical_dimensions:
            base_weight += 0.10

        return min(base_weight, 0.50)  # Cap at 50%

    def _add_domain_specific_dimensions(
        self,
        extracted: 'ExtractedRequirements'
    ) -> List[DetectedDimension]:
        """Add domain-specific evaluation dimensions"""

        domain_dims = []

        # Medical domain
        if extracted.domain == "medical":
            # Add hallucination detector (critical for medical)
            domain_dims.append(DetectedDimension(
                name="hallucination_detection",
                weight=0.20,
                is_critical=True,
                min_pass_score=4.0,
                detection_reason="Medical domain requires hallucination detection",
                evaluator_type="HallucinationDetector"
            ))

        # Financial domain
        elif extracted.domain == "financial":
            # Add precision checker
            domain_dims.append(DetectedDimension(
                name="precision",
                weight=0.15,
                is_critical=True,
                min_pass_score=4.0,
                detection_reason="Financial domain requires numerical precision",
                evaluator_type="PrecisionEvaluator"
            ))

        # Customer service domain
        elif extracted.domain == "customer_service":
            # Add empathy evaluator
            domain_dims.append(DetectedDimension(
                name="empathy",
                weight=0.10,
                is_critical=False,
                min_pass_score=3.0,
                detection_reason="Customer service requires empathetic responses",
                evaluator_type="EmpathyEvaluator"
            ))

        return domain_dims

    def _detect_schema_requirement(
        self,
        system_prompt: str,
        requirements: str,
        structured_requirements: Optional[Dict[str, Any]]
    ) -> Optional[DetectedDimension]:
        """Detect if structured output format is required"""

        combined = f"{system_prompt}\n{requirements}".lower()

        # Check for explicit format mentions
        format_indicators = [
            r'\bjson\b',
            r'\bxml\b',
            r'\byaml\b',
            r'\bcsv\b',
            r'output format',
            r'structured output',
            r'schema',
            r'required fields',
            r'response format'
        ]

        matches = sum(1 for pattern in format_indicators if re.search(pattern, combined))

        # Check structured requirements
        has_output_format = False
        if structured_requirements and structured_requirements.get('output_format'):
            has_output_format = True
            matches += 2

        if matches >= 2 or has_output_format:
            return DetectedDimension(
                name="schema_compliance",
                weight=0.20,
                is_critical=True,
                min_pass_score=4.0,
                detection_reason=f"Detected {matches} format/schema indicators",
                evaluator_type="SchemaValidator"
            )

        return None

    def _detect_safety_requirement(self, text: str) -> Optional[DetectedDimension]:
        """Detect if safety checking is needed"""

        safety_indicators = [
            r'\bsafe\b',
            r'\bharm',
            r'\bprivacy\b',
            r'\bconfidential',
            r'\bsensitive',
            r'\bbias',
            r'\bappropriate\b',
            r'\bprofessional\b',
            r'no .*? information',
            r'must not',
            r'do not (?:include|share|reveal)'
        ]

        matches = sum(1 for pattern in safety_indicators if re.search(pattern, text))

        # Safety is always important, but criticality depends on context
        if matches >= 1:
            is_critical = matches >= 3  # High emphasis on safety

            return DetectedDimension(
                name="safety",
                weight=0.25 if is_critical else 0.15,
                is_critical=is_critical,
                min_pass_score=4.0,
                detection_reason=f"Detected {matches} safety indicators",
                evaluator_type="SafetyChecker"
            )

        # Default: always include basic safety check
        return DetectedDimension(
            name="safety",
            weight=0.10,
            is_critical=False,
            min_pass_score=3.0,
            detection_reason="Default safety check",
            evaluator_type="SafetyChecker"
        )

    def _detect_completeness_requirement(
        self,
        system_prompt: str,
        requirements: str,
        structured_requirements: Optional[Dict[str, Any]]
    ) -> Optional[DetectedDimension]:
        """Detect if completeness checking is needed"""

        combined = f"{system_prompt}\n{requirements}".lower()

        completeness_indicators = [
            r'\bmust include\b',
            r'\brequired\b',
            r'\ball of the following\b',
            r'\baddress all\b',
            r'\bcomplete\b',
            r'\bcomprehensive\b',
            r'should (?:include|contain|cover)',
            r'needs to (?:include|contain|address)'
        ]

        matches = sum(1 for pattern in completeness_indicators if re.search(pattern, combined))

        # Check for explicit must-do items
        has_must_do = False
        if structured_requirements and structured_requirements.get('must_do'):
            has_must_do = True
            matches += len(structured_requirements['must_do'])

        if matches >= 2 or has_must_do:
            return DetectedDimension(
                name="completeness",
                weight=0.15,
                is_critical=matches >= 4,
                min_pass_score=3.0,
                detection_reason=f"Detected {matches} completeness indicators",
                evaluator_type="CompletenessChecker"
            )

        return None

    def _detect_accuracy_requirement(self, text: str) -> DetectedDimension:
        """Detect accuracy requirements (almost always needed)"""

        accuracy_indicators = [
            r'\baccurate\b',
            r'\bcorrect\b',
            r'\bfactual\b',
            r'\bprecise\b',
            r'\bverif',
            r'\bground',
            r'based on',
            r'no hallucination',
            r'do not (?:invent|fabricate|make up)'
        ]

        matches = sum(1 for pattern in accuracy_indicators if re.search(pattern, text))

        # Accuracy is critical if heavily emphasized
        is_critical = matches >= 3

        return DetectedDimension(
            name="accuracy",
            weight=0.30,
            is_critical=is_critical,
            min_pass_score=3.0,
            detection_reason=f"Detected {matches} accuracy indicators (always important)",
            evaluator_type="AccuracyEvaluator"
        )

    def _detect_actionability_requirement(self, text: str) -> Optional[DetectedDimension]:
        """Detect if actionability is important"""

        actionability_indicators = [
            r'\bactionable\b',
            r'\bspecific\b',
            r'\bconcrete\b',
            r'\bpractical\b',
            r'\busable\b',
            r'\bstep[- ]by[- ]step\b',
            r'\binstructions?\b',
            r'\bguidance\b',
            r'how to',
            r'help (?:the user|users|them)'
        ]

        matches = sum(1 for pattern in actionability_indicators if re.search(pattern, text))

        if matches >= 2:
            return DetectedDimension(
                name="actionability",
                weight=0.15,
                is_critical=False,
                min_pass_score=2.5,
                detection_reason=f"Detected {matches} actionability indicators",
                evaluator_type="ActionabilityEvaluator"
            )

        return None

    def _detect_relevance_requirement(self, text: str) -> DetectedDimension:
        """Detect relevance requirements (almost always needed)"""

        relevance_indicators = [
            r'\brelevant\b',
            r'\bon[- ]topic\b',
            r'\bin[- ]scope\b',
            r'\bfocused\b',
            r'\baddress(?:es)? the (?:request|question|input)\b',
            r'stay (?:on topic|focused|relevant)',
            r'do not (?:go off|stray|drift)'
        ]

        matches = sum(1 for pattern in relevance_indicators if re.search(pattern, text))

        return DetectedDimension(
            name="relevance",
            weight=0.15,
            is_critical=False,
            min_pass_score=2.5,
            detection_reason=f"Detected {matches} relevance indicators (always important)",
            evaluator_type="RelevanceEvaluator"
        )

    def _detect_tone_requirement(
        self,
        system_prompt: str,
        requirements: str
    ) -> Optional[DetectedDimension]:
        """Detect if specific tone is required"""

        combined = f"{system_prompt}\n{requirements}"

        tone_patterns = {
            'professional': r'\bprofessional\b',
            'friendly': r'\b(?:friendly|warm|empathetic)\b',
            'concise': r'\b(?:concise|brief|short)\b',
            'formal': r'\b(?:formal|polite|courteous)\b',
            'casual': r'\b(?:casual|conversational)\b',
            'technical': r'\b(?:technical|precise)\b'
        }

        detected_tones = []
        for tone_name, pattern in tone_patterns.items():
            if re.search(pattern, combined.lower()):
                detected_tones.append(tone_name)

        if detected_tones:
            tone_description = ", ".join(detected_tones)
            return DetectedDimension(
                name="tone",
                weight=0.10,
                is_critical=False,
                min_pass_score=2.0,
                detection_reason=f"Detected tone requirements: {tone_description}",
                evaluator_type="ToneEvaluator"
            )

        return None

    def _normalize_weights(self, dimensions: List[DetectedDimension]) -> List[DetectedDimension]:
        """Normalize weights to sum to 1.0"""

        if not dimensions:
            return dimensions

        total_weight = sum(d.weight for d in dimensions)

        if total_weight > 0:
            for dim in dimensions:
                dim.weight = round(dim.weight / total_weight, 2)

        return dimensions

    def _build_detection_patterns(self) -> Dict[str, List[str]]:
        """Build regex patterns for dimension detection"""

        return {
            'schema': [
                r'\bjson\b', r'\bxml\b', r'\byaml\b',
                r'output format', r'structured output', r'schema'
            ],
            'safety': [
                r'\bsafe\b', r'\bharm', r'\bprivacy\b',
                r'\bconfidential', r'\bbias'
            ],
            'accuracy': [
                r'\baccurate\b', r'\bcorrect\b', r'\bfactual\b',
                r'no hallucination'
            ],
            'completeness': [
                r'\bmust include\b', r'\brequired\b',
                r'\bcomplete\b', r'\bcomprehensive\b'
            ],
            'actionability': [
                r'\bactionable\b', r'\bspecific\b',
                r'\bpractical\b', r'\busable\b'
            ],
            'relevance': [
                r'\brelevant\b', r'\bon[- ]topic\b',
                r'\bfocused\b'
            ],
            'tone': [
                r'\bprofessional\b', r'\bfriendly\b',
                r'\bconcise\b', r'\bformal\b'
            ]
        }


def extract_required_elements(
    system_prompt: str,
    requirements: str,
    structured_requirements: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Extract list of required elements from requirements"""

    required_elements = []

    # From structured requirements
    if structured_requirements:
        if structured_requirements.get('must_do'):
            required_elements.extend(structured_requirements['must_do'])

        if structured_requirements.get('output_format'):
            required_elements.append(f"Output format: {structured_requirements['output_format']}")

    # From text - look for "must include", "required", etc.
    combined = f"{system_prompt}\n{requirements}"

    # Pattern: "must include X", "required: X", etc.
    patterns = [
        r'must include:?\s*([^\n.]+)',
        r'required:?\s*([^\n.]+)',
        r'should include:?\s*([^\n.]+)',
        r'needs to include:?\s*([^\n.]+)'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, combined, re.IGNORECASE)
        required_elements.extend(matches)

    # Clean up
    required_elements = [elem.strip() for elem in required_elements if elem.strip()]

    return required_elements[:10]  # Limit to 10 most important


def extract_safety_requirements(
    system_prompt: str,
    requirements: str,
    structured_requirements: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Extract safety requirements from inputs"""

    safety_reqs = []

    # From structured requirements
    if structured_requirements and structured_requirements.get('must_not_do'):
        safety_reqs.extend(structured_requirements['must_not_do'])

    # From text - look for "must not", "do not", etc.
    combined = f"{system_prompt}\n{requirements}"

    patterns = [
        r'must not\s+([^\n.]+)',
        r'do not\s+([^\n.]+)',
        r'never\s+([^\n.]+)',
        r'should not\s+([^\n.]+)'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, combined, re.IGNORECASE)
        safety_reqs.extend(matches)

    # Default safety requirements
    if not safety_reqs:
        safety_reqs = [
            "No harmful or dangerous content",
            "No private/sensitive information",
            "No biased or discriminatory language"
        ]

    return safety_reqs[:8]


def extract_tone_requirements(
    system_prompt: str,
    requirements: str,
    structured_requirements: Optional[Dict[str, Any]] = None
) -> str:
    """Extract tone requirements into a description"""

    # From structured requirements
    if structured_requirements and structured_requirements.get('tone'):
        return structured_requirements['tone']

    # From text
    combined = f"{system_prompt}\n{requirements}".lower()

    tone_descriptors = []

    tone_keywords = {
        'professional': r'\bprofessional\b',
        'friendly': r'\b(?:friendly|warm)\b',
        'empathetic': r'\bempathetic\b',
        'concise': r'\b(?:concise|brief)\b',
        'formal': r'\bformal\b',
        'casual': r'\bcasual\b'
    }

    for tone, pattern in tone_keywords.items():
        if re.search(pattern, combined):
            tone_descriptors.append(tone)

    if tone_descriptors:
        return ", ".join(tone_descriptors)

    return "professional and helpful"


def extract_output_schema(
    system_prompt: str,
    requirements: str,
    structured_requirements: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Extract expected output schema if specified"""

    # From structured requirements
    if structured_requirements and structured_requirements.get('output_format'):
        format_str = structured_requirements['output_format']

        # Try to parse as JSON schema
        import json
        try:
            if '{' in format_str:
                return json.loads(format_str)
        except:
            pass

        # Return as simple schema
        return {
            "type": "object",
            "format_description": format_str
        }

    # Try to detect JSON schema in text
    combined = f"{system_prompt}\n{requirements}"

    # Look for JSON blocks
    json_blocks = re.findall(r'```json\s*([\s\S]*?)\s*```', combined, re.IGNORECASE)

    if json_blocks:
        import json
        try:
            return json.loads(json_blocks[0])
        except:
            pass

    return None
