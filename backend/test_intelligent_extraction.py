"""
Test script demonstrating the difference between regex-based and
intelligent LLM-based requirements extraction.

Run this to see the quality improvement.
"""

import asyncio
import sys
from typing import Dict, Any

# Mock LLM client for testing
class MockLLMClient:
    """Mock LLM client that returns predefined responses"""

    async def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """Return mock extraction response"""

        # Detect domain from prompt
        prompt_lower = prompt.lower()

        if "medical" in prompt_lower or "diagnosis" in prompt_lower:
            return """{
  "use_case": "Medical diagnosis assistant for healthcare providers",
  "primary_function": "Analyze patient symptoms and provide differential diagnosis with confidence levels",
  "domain": "medical",
  "risk_level": "critical",
  "key_requirements": [
    "Extreme accuracy (medical domain)",
    "Confidence levels for each diagnosis",
    "Differential diagnosis format",
    "No hallucinations or fabricated conditions",
    "Medical disclaimers required"
  ],
  "must_do": [
    "Include confidence levels for diagnoses",
    "Provide differential diagnosis format",
    "Base on actual medical conditions and current knowledge",
    "Include appropriate medical disclaimers"
  ],
  "must_not_do": [
    "Make up conditions or fabricate medical information",
    "Give definitive diagnosis without appropriate caveats",
    "Provide treatment advice without healthcare provider consultation disclaimer"
  ],
  "output_format": {
    "type": "structured",
    "description": "Differential diagnosis with confidence levels",
    "required_fields": ["symptoms_analyzed", "differential_diagnosis", "confidence_levels", "medical_disclaimer"]
  },
  "expected_structure": "Structured diagnosis with confidence scoring",
  "tone": "professional, cautious, clinical",
  "style_requirements": [
    "Clear medical terminology",
    "Appropriate caveats and disclaimers",
    "Evidence-based reasoning"
  ],
  "quality_priorities": [
    "accuracy",
    "safety",
    "completeness",
    "clarity"
  ],
  "critical_dimensions": [
    "accuracy",
    "safety",
    "hallucination_detection"
  ],
  "important_dimensions": [
    "completeness",
    "clarity",
    "tone"
  ],
  "target_audience": "Healthcare providers and medical professionals",
  "example_scenarios": [
    "Patient presents with fever and cough",
    "Patient reports chest pain and shortness of breath"
  ],
  "confidence_score": 0.92,
  "extraction_notes": "Medical domain detected with high confidence. Critical safety and accuracy requirements identified."
}"""

        elif "financial" in prompt_lower or "investment" in prompt_lower:
            return """{
  "use_case": "Financial advisory assistant",
  "primary_function": "Help users understand investment options with appropriate risk disclosures",
  "domain": "financial",
  "risk_level": "high",
  "key_requirements": [
    "Accurate financial information",
    "Numerical precision",
    "Risk disclosures required",
    "No guarantees of returns"
  ],
  "must_do": [
    "Include risk disclosures",
    "Provide accurate financial information",
    "Explain investment options clearly",
    "Use precise numerical values"
  ],
  "must_not_do": [
    "Guarantee financial returns",
    "Provide specific investment advice without disclaimers",
    "Misrepresent financial risks"
  ],
  "output_format": {
    "type": "structured",
    "description": "Financial analysis with risk disclosure"
  },
  "expected_structure": "Clear financial explanation with risk information",
  "tone": "professional, informative",
  "style_requirements": [
    "Precise numerical data",
    "Clear risk communication"
  ],
  "quality_priorities": [
    "accuracy",
    "precision",
    "safety",
    "clarity"
  ],
  "critical_dimensions": [
    "accuracy",
    "precision",
    "safety"
  ],
  "important_dimensions": [
    "completeness",
    "clarity"
  ],
  "target_audience": "Individual investors",
  "example_scenarios": [
    "User asks about investment options",
    "User wants to understand risk vs return"
  ],
  "confidence_score": 0.88,
  "extraction_notes": "Financial domain with high-risk implications. Precision and accuracy are critical."
}"""

        else:
            return """{
  "use_case": "General assistant",
  "primary_function": "Help users with general queries",
  "domain": "general",
  "risk_level": "medium",
  "key_requirements": ["Be helpful", "Be accurate"],
  "must_do": ["Provide accurate information"],
  "must_not_do": ["Provide harmful information"],
  "output_format": {"type": "text"},
  "expected_structure": null,
  "tone": "professional",
  "style_requirements": ["Clear", "Concise"],
  "quality_priorities": ["accuracy", "relevance"],
  "critical_dimensions": ["accuracy", "safety"],
  "important_dimensions": ["relevance"],
  "target_audience": "General users",
  "example_scenarios": [],
  "confidence_score": 0.75,
  "extraction_notes": "General domain with standard requirements"
}"""


async def test_extraction_comparison():
    """Compare regex-based vs intelligent extraction"""

    print("=" * 80)
    print("🧪 TESTING: Regex-Based vs Intelligent Extraction")
    print("=" * 80)
    print()

    # Test case: Medical diagnosis system
    system_prompt = """You are a medical diagnosis assistant. Analyze patient symptoms and provide
a differential diagnosis with confidence levels. Be extremely careful about
accuracy as this affects patient care. Never make up conditions."""

    requirements = "Must be accurate and safe. Provide clear explanations."
    use_case = "Healthcare"

    print("📝 SYSTEM PROMPT:")
    print("-" * 80)
    print(system_prompt)
    print()
    print(f"📋 REQUIREMENTS: {requirements}")
    print(f"🎯 USE CASE: {use_case}")
    print()

    # Import modules
    from dimension_detector import DimensionDetector
    try:
        from intelligent_requirements_extractor import IntelligentRequirementsExtractor
        intelligent_available = True
    except:
        intelligent_available = False
        print("⚠️ Intelligent extraction not available, testing regex only")
        print()

    # Test 1: Regex-based detection
    print("=" * 80)
    print("📊 REGEX-BASED DETECTION")
    print("=" * 80)

    detector = DimensionDetector()
    regex_dimensions = detector.detect_dimensions(
        system_prompt=system_prompt,
        requirements=requirements,
        use_case=use_case
    )

    print(f"\n✅ Detected {len(regex_dimensions)} dimensions:\n")
    for dim in regex_dimensions:
        critical_mark = "⚠️ CRITICAL" if dim.is_critical else "  "
        print(f"  {critical_mark} {dim.name:20s} | weight={dim.weight:.2f} | {dim.evaluator_type}")
        print(f"      → {dim.detection_reason}")

    # Test 2: Intelligent extraction
    if intelligent_available:
        print("\n" + "=" * 80)
        print("🧠 INTELLIGENT LLM-BASED EXTRACTION")
        print("=" * 80)

        llm_client = MockLLMClient()
        extractor = IntelligentRequirementsExtractor(llm_client)

        extracted = await extractor.extract_requirements(
            system_prompt=system_prompt,
            requirements=requirements,
            use_case=use_case
        )

        print(f"\n📦 Extracted Requirements:")
        print(f"   Domain: {extracted.domain}")
        print(f"   Risk Level: {extracted.risk_level}")
        print(f"   Confidence: {extracted.confidence_score:.2f}")
        print(f"\n   Critical Dimensions: {', '.join(extracted.critical_dimensions)}")
        print(f"\n   Must Do:")
        for item in extracted.must_do[:3]:
            print(f"      • {item}")
        print(f"\n   Must NOT Do:")
        for item in extracted.must_not_do[:3]:
            print(f"      • {item}")

        # Detect dimensions using intelligent extraction
        intelligent_dimensions = detector.detect_dimensions_from_extracted(extracted)

        print(f"\n✅ Detected {len(intelligent_dimensions)} dimensions:\n")
        for dim in intelligent_dimensions:
            critical_mark = "⚠️ CRITICAL" if dim.is_critical else "  "
            print(f"  {critical_mark} {dim.name:25s} | weight={dim.weight:.2f} | {dim.evaluator_type}")
            print(f"      → {dim.detection_reason}")

        # Comparison
        print("\n" + "=" * 80)
        print("📊 COMPARISON")
        print("=" * 80)

        print(f"\nNumber of Dimensions:")
        print(f"   Regex-based:     {len(regex_dimensions)}")
        print(f"   Intelligent:     {len(intelligent_dimensions)}")

        print(f"\nEvaluator Types:")
        regex_types = set(d.evaluator_type for d in regex_dimensions)
        intelligent_types = set(d.evaluator_type for d in intelligent_dimensions)

        print(f"   Regex-based:     {', '.join(sorted(regex_types))}")
        print(f"   Intelligent:     {', '.join(sorted(intelligent_types))}")

        new_evaluators = intelligent_types - regex_types
        if new_evaluators:
            print(f"\n   ✅ NEW in Intelligent: {', '.join(new_evaluators)}")

        print(f"\nCritical Dimensions:")
        regex_critical = [d.name for d in regex_dimensions if d.is_critical]
        intelligent_critical = [d.name for d in intelligent_dimensions if d.is_critical]

        print(f"   Regex-based:     {len(regex_critical)} ({', '.join(regex_critical)})")
        print(f"   Intelligent:     {len(intelligent_critical)} ({', '.join(intelligent_critical)})")

        # Quality assessment
        print("\n" + "=" * 80)
        print("🎯 QUALITY ASSESSMENT")
        print("=" * 80)

        improvements = []

        if "HallucinationDetector" in intelligent_types:
            improvements.append("✅ Added HallucinationDetector (critical for medical)")

        if "DomainSafetyChecker" in intelligent_types:
            improvements.append("✅ Added DomainSafetyChecker (medical-specific safety)")

        if "DomainAccuracyEvaluator" in intelligent_types:
            improvements.append("✅ Added DomainAccuracyEvaluator (medical expert evaluation)")

        if len(intelligent_critical) > len(regex_critical):
            improvements.append(f"✅ Better critical dimension detection ({len(intelligent_critical)} vs {len(regex_critical)})")

        if improvements:
            print("\nIntelligent extraction improvements:")
            for imp in improvements:
                print(f"   {imp}")
        else:
            print("\n   Similar results (may need real LLM for full benefits)")

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_extraction_comparison())
