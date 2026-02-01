"""
Intelligent Requirements Extractor

Uses LLM to deeply analyze system prompts and extract structured requirements,
use cases, constraints, and evaluation criteria.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from llm_client_v2 import EnhancedLLMClient

logger = logging.getLogger(__name__)


@dataclass
class ExtractedRequirements:
    """Structured requirements extracted from system prompt"""

    # Core understanding
    use_case: str
    primary_function: str
    domain: str  # e.g., "medical", "financial", "customer_service", "general"
    risk_level: str  # "low", "medium", "high", "critical"

    # Functional requirements
    key_requirements: List[str]
    must_do: List[str]
    must_not_do: List[str]

    # Output specifications
    output_format: Dict[str, Any]
    expected_structure: Optional[str]

    # Quality criteria
    tone: str
    style_requirements: List[str]
    quality_priorities: List[str]  # What matters most? accuracy, speed, completeness, etc.

    # Evaluation dimensions
    critical_dimensions: List[str]  # Dimensions that are make-or-break
    important_dimensions: List[str]  # Dimensions that matter but aren't critical

    # Context
    target_audience: str
    example_scenarios: List[str]

    # Extraction metadata
    confidence_score: float  # How confident is the extraction (0-1)
    extraction_notes: str


EXTRACTION_PROMPT = """You are an expert prompt analyst. Analyze the system prompt below and extract detailed, structured requirements.

**System Prompt:**
```
{system_prompt}
```

**Additional Requirements/Context:**
```
{requirements}
```

**Use Case:**
{use_case}

---

## Your Task

Extract comprehensive structured requirements from this system prompt. Think deeply about:

1. **What is this system supposed to do?** (use case, primary function)
2. **What domain is this?** (medical, financial, customer service, technical, creative, general)
3. **What's the risk level?** (Does poor performance have serious consequences?)
4. **What MUST it do?** (Critical requirements)
5. **What MUST it NOT do?** (Constraints, safety requirements)
6. **What format should outputs have?** (JSON, text, specific structure)
7. **What tone/style is expected?** (professional, friendly, technical, etc.)
8. **What quality dimensions are critical?** (accuracy, safety, relevance, etc.)

---

## Output Format

Return **ONLY** valid JSON matching this schema:

```json
{{
  "use_case": "One sentence: what is this system for?",
  "primary_function": "One sentence: what does it do?",
  "domain": "medical|financial|customer_service|legal|technical|creative|education|general",
  "risk_level": "low|medium|high|critical",

  "key_requirements": [
    "Requirement 1 (be specific)",
    "Requirement 2",
    "..."
  ],

  "must_do": [
    "Must do item 1",
    "Must do item 2",
    "..."
  ],

  "must_not_do": [
    "Must NOT do item 1",
    "Must NOT do item 2",
    "..."
  ],

  "output_format": {{
    "type": "json|text|structured|mixed",
    "description": "Description of expected format",
    "required_fields": ["field1", "field2"],
    "schema": {{}}  // If JSON schema is specified
  }},

  "expected_structure": "Description of how output should be structured (if applicable)",

  "tone": "professional|friendly|formal|casual|empathetic|technical|creative",

  "style_requirements": [
    "Concise",
    "No jargon",
    "..."
  ],

  "quality_priorities": [
    "accuracy",
    "safety",
    "relevance",
    "completeness",
    "actionability",
    "tone",
    "..."
  ],

  "critical_dimensions": [
    "Dimensions that are make-or-break (e.g., 'safety' for medical, 'accuracy' for financial)"
  ],

  "important_dimensions": [
    "Dimensions that matter but aren't critical"
  ],

  "target_audience": "Who will use this system's outputs?",

  "example_scenarios": [
    "Example use case 1",
    "Example use case 2"
  ],

  "confidence_score": 0.85,
  "extraction_notes": "Brief notes about extraction quality or uncertainties"
}}
```

---

## Important Guidelines

- **Be specific**: "Must provide accurate medical information" not just "be accurate"
- **Infer intelligently**: If prompt says "customer service bot", infer tone should be professional/friendly
- **Detect implicit requirements**: "medical diagnosis" → safety is CRITICAL even if not stated
- **Understand domain conventions**: Financial = precision matters, Creative = flexibility matters
- **Quality priorities order**: List in order of importance based on context
- **Critical vs important**: Critical = failure is unacceptable, Important = matters but not make-or-break

Return ONLY the JSON, no additional text.
"""


class IntelligentRequirementsExtractor:
    """
    Uses LLM to extract deep, structured understanding of requirements
    from system prompts
    """

    def __init__(self, llm_client: EnhancedLLMClient):
        self.llm_client = llm_client

    async def extract_requirements(
        self,
        system_prompt: str,
        requirements: str = "",
        use_case: str = "",
        structured_requirements: Optional[Dict[str, Any]] = None
    ) -> ExtractedRequirements:
        """
        Extract structured requirements using LLM analysis

        Args:
            system_prompt: The system prompt to analyze
            requirements: Additional requirements text
            use_case: Use case description
            structured_requirements: Any existing structured requirements

        Returns:
            ExtractedRequirements object with comprehensive understanding
        """

        logger.info("Extracting requirements using intelligent LLM analysis...")

        # Build extraction prompt
        prompt = EXTRACTION_PROMPT.format(
            system_prompt=system_prompt,
            requirements=requirements or "None provided",
            use_case=use_case or "Not specified"
        )

        try:
            # Call LLM for extraction
            response = await self.llm_client.generate(
                prompt=prompt,
                system_prompt="You are an expert prompt analyst. Extract detailed structured requirements as JSON.",
                temperature=0.3,  # Lower temperature for consistent extraction
                max_tokens=2000
            )

            # Parse JSON response
            extracted_json = self._extract_json_from_response(response)

            if not extracted_json:
                logger.warning("Failed to extract JSON, using fallback extraction")
                return self._fallback_extraction(system_prompt, requirements, use_case)

            # Convert to ExtractedRequirements object
            extracted = ExtractedRequirements(
                use_case=extracted_json.get("use_case", use_case or "Not specified"),
                primary_function=extracted_json.get("primary_function", "Not specified"),
                domain=extracted_json.get("domain", "general"),
                risk_level=extracted_json.get("risk_level", "medium"),
                key_requirements=extracted_json.get("key_requirements", []),
                must_do=extracted_json.get("must_do", []),
                must_not_do=extracted_json.get("must_not_do", []),
                output_format=extracted_json.get("output_format", {"type": "text"}),
                expected_structure=extracted_json.get("expected_structure"),
                tone=extracted_json.get("tone", "professional"),
                style_requirements=extracted_json.get("style_requirements", []),
                quality_priorities=extracted_json.get("quality_priorities", ["accuracy", "relevance"]),
                critical_dimensions=extracted_json.get("critical_dimensions", []),
                important_dimensions=extracted_json.get("important_dimensions", []),
                target_audience=extracted_json.get("target_audience", "General users"),
                example_scenarios=extracted_json.get("example_scenarios", []),
                confidence_score=extracted_json.get("confidence_score", 0.8),
                extraction_notes=extracted_json.get("extraction_notes", "")
            )

            # Merge with structured requirements if provided
            if structured_requirements:
                extracted = self._merge_with_structured_requirements(
                    extracted, structured_requirements
                )

            logger.info(
                f"Requirements extracted successfully: "
                f"domain={extracted.domain}, risk_level={extracted.risk_level}, "
                f"confidence={extracted.confidence_score:.2f}"
            )

            return extracted

        except Exception as e:
            logger.error(f"Error in intelligent extraction: {e}", exc_info=True)
            return self._fallback_extraction(system_prompt, requirements, use_case)

    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response"""

        try:
            # Try to parse directly
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block
        import re

        # Look for ```json ... ```
        json_block_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_block_match:
            try:
                return json.loads(json_block_match.group(1))
            except json.JSONDecodeError:
                pass

        # Look for { ... }
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("Could not extract JSON from LLM response")
        return None

    def _fallback_extraction(
        self,
        system_prompt: str,
        requirements: str,
        use_case: str
    ) -> ExtractedRequirements:
        """Fallback to basic extraction if LLM fails"""

        logger.warning("Using fallback extraction (LLM extraction failed)")

        # Basic extraction using simple heuristics
        combined = f"{system_prompt}\n{requirements}".lower()

        # Detect domain
        domain = "general"
        if any(word in combined for word in ["medical", "health", "diagnosis", "patient"]):
            domain = "medical"
        elif any(word in combined for word in ["financial", "banking", "trading", "investment"]):
            domain = "financial"
        elif any(word in combined for word in ["customer", "support", "service", "help desk"]):
            domain = "customer_service"
        elif any(word in combined for word in ["legal", "law", "contract", "compliance"]):
            domain = "legal"

        # Detect risk level
        risk_level = "medium"
        if domain in ["medical", "legal", "financial"]:
            risk_level = "high"

        return ExtractedRequirements(
            use_case=use_case or "General purpose assistant",
            primary_function="Process inputs and generate appropriate outputs",
            domain=domain,
            risk_level=risk_level,
            key_requirements=["Provide accurate responses", "Follow instructions"],
            must_do=["Be helpful", "Be accurate"],
            must_not_do=["Provide harmful information"],
            output_format={"type": "text"},
            expected_structure=None,
            tone="professional",
            style_requirements=["Clear", "Concise"],
            quality_priorities=["accuracy", "relevance", "safety"],
            critical_dimensions=["safety", "accuracy"],
            important_dimensions=["relevance", "completeness"],
            target_audience="General users",
            example_scenarios=[],
            confidence_score=0.5,
            extraction_notes="Fallback extraction used due to LLM extraction failure"
        )

    def _merge_with_structured_requirements(
        self,
        extracted: ExtractedRequirements,
        structured_requirements: Dict[str, Any]
    ) -> ExtractedRequirements:
        """Merge LLM-extracted requirements with provided structured requirements"""

        # Merge must_do
        if structured_requirements.get("must_do"):
            extracted.must_do.extend(structured_requirements["must_do"])
            extracted.must_do = list(set(extracted.must_do))  # Deduplicate

        # Merge must_not_do
        if structured_requirements.get("must_not_do"):
            extracted.must_not_do.extend(structured_requirements["must_not_do"])
            extracted.must_not_do = list(set(extracted.must_not_do))

        # Override output format if specified
        if structured_requirements.get("output_format"):
            extracted.output_format = {
                "type": "structured",
                "description": str(structured_requirements["output_format"])
            }

        # Override tone if specified
        if structured_requirements.get("tone"):
            extracted.tone = structured_requirements["tone"]

        return extracted


async def extract_requirements_intelligently(
    system_prompt: str,
    requirements: str,
    use_case: str,
    llm_client: EnhancedLLMClient,
    structured_requirements: Optional[Dict[str, Any]] = None
) -> ExtractedRequirements:
    """
    Convenience function for intelligent requirements extraction

    Usage:
        extracted = await extract_requirements_intelligently(
            system_prompt="You are a medical diagnosis assistant...",
            requirements="Must be accurate and safe...",
            use_case="Healthcare",
            llm_client=client
        )

        print(f"Domain: {extracted.domain}")
        print(f"Risk level: {extracted.risk_level}")
        print(f"Critical dimensions: {extracted.critical_dimensions}")
    """

    extractor = IntelligentRequirementsExtractor(llm_client)
    return await extractor.extract_requirements(
        system_prompt=system_prompt,
        requirements=requirements,
        use_case=use_case,
        structured_requirements=structured_requirements
    )
