"""
Load Expert Dimension Design Patterns into ChromaDB

Pre-loads dimension design patterns extracted from production systems
(PAM Roleplay Recommendation, BI RepDiagnostic) into ChromaDB for
use during dimension generation.

Run this script once to populate the dimension_design_patterns collection.
"""

import asyncio
import logging
from dimension_pattern_service import get_dimension_pattern_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def load_all_patterns():
    """Load all expert dimension design patterns."""
    service = get_dimension_pattern_service()

    logger.info("=" * 60)
    logger.info("Loading Expert Dimension Design Patterns")
    logger.info("=" * 60)

    # Clear existing patterns (optional - comment out if you want to keep existing)
    # service.clear_patterns()

    patterns_loaded = 0

    # ========== EVALUATION ARCHITECTURE PATTERNS ==========
    logger.info("\n[1/6] Loading Evaluation Architecture Patterns...")

    await service.store_pattern(
        category="eval_architecture",
        pattern_data={
            "pattern_name": "Layer 1 + Layer 2 Split",
            "description": "Separate deterministic checks (100% coverage) from LLM-as-judge (sampled). Math/logic verified programmatically, semantics evaluated by LLM.",
            "when_to_use": "When system has hard constraints (schema, formulas, logic) AND subjective quality aspects (style, voice, alignment)",
            "example_domains": ["structured_output", "recommendation_engines", "diagnostic_systems"],
            "example": "Layer 1: Schema validation, reference integrity, formula verification. Layer 2: Framework alignment, style enforcement, voice check."
        }
    )
    patterns_loaded += 1

    await service.store_pattern(
        category="eval_architecture",
        pattern_data={
            "pattern_name": "Tier 1 + Tier 2 Sampled",
            "description": "100% deterministic validation on structure, sampled LLM evaluation on quality dimensions",
            "when_to_use": "Subjective domains where 'correctness' is impossible, focus on usefulness",
            "example_domains": ["diagnostics", "coaching", "analysis", "creative_content"],
            "benefit": "Balances coverage (all outputs validated) with cost (only sample quality checks)"
        }
    )
    patterns_loaded += 1

    # ========== DIMENSION DESIGN PRINCIPLES ==========
    logger.info("[2/6] Loading Dimension Design Principles...")

    await service.store_pattern(
        category="design_principle",
        pattern_data={
            "principle": "Atomic Splits",
            "description": "Split complex dimensions into atomic, independent checks. E.g., 'Framework Fit' becomes: (1) Taxonomy Fit, (2) Evidence-Phase Alignment, (3) Classification Accuracy",
            "benefit": "Reduces ambiguity, enables precise failure attribution, easier to debug",
            "anti_pattern": "Single eval trying to check multiple unrelated aspects",
            "example": "Instead of 'framework_quality', use: taxonomy_alignment + evidence_phase_alignment + classification_accuracy"
        }
    )
    patterns_loaded += 1

    await service.store_pattern(
        category="design_principle",
        pattern_data={
            "principle": "Binary Constraints for Logic",
            "description": "Use PASS/FAIL for deterministic aspects, not gradient scales. Logic either works or doesn't.",
            "when_to_use": "Schema validation, reference integrity, formula correctness, template compliance",
            "anti_pattern": "Using 1-5 scale to evaluate if ID exists or formula is correct",
            "example": "schema_validity: PASS (all required fields present) or FAIL (missing fields)"
        }
    )
    patterns_loaded += 1

    await service.store_pattern(
        category="design_principle",
        pattern_data={
            "principle": "Four Core Questions",
            "description": "Structure dimensions around answering: (1) Is interpretation valid? (2) Is it evidenced? (3) Is it coherent? (4) Is it communicable?",
            "when_to_use": "Subjective evaluation domains where multiple quality aspects matter",
            "benefit": "Comprehensive coverage, clear separation of concerns",
            "example": "interpretation_quality, evidentiary_support, coherence, communicability"
        }
    )
    patterns_loaded += 1

    await service.store_pattern(
        category="design_principle",
        pattern_data={
            "principle": "Sub-Scores for Granularity",
            "description": "Break each dimension into 3-4 sub-criteria. Aggregate upward. Enables precise diagnosis of issues.",
            "example": "interpretation_quality -> grounding_alignment + signal_fidelity + specificity",
            "benefit": "Pinpoints exact failure modes, actionable feedback, drill-down analysis"
        }
    )
    patterns_loaded += 1

    await service.store_pattern(
        category="design_principle",
        pattern_data={
            "principle": "Acceptance Criteria over Rubrics",
            "description": "Define explicit 'what passes' criteria with examples, not vague rubric descriptions",
            "example": "PASS: Skill/Gap falls within 'Includes' and does NOT violate 'Excludes'. FAIL: Violates 'Excludes' rule.",
            "anti_pattern": "Vague rubric like '3 = moderately good fit'",
            "benefit": "Eliminates ambiguity, clear expectations, consistent scoring"
        }
    )
    patterns_loaded += 1

    await service.store_pattern(
        category="design_principle",
        pattern_data={
            "principle": "Separation of Concerns",
            "description": "Each eval tests ONE thing. Title style separate from content quality separate from schema validation.",
            "benefit": "Clear attribution, easier refinement, prevents eval overload",
            "anti_pattern": "Single eval checking 'overall quality' of multiple aspects",
            "example": "title_style_enforcement (checks template) + content_quality (checks substance) as separate evals"
        }
    )
    patterns_loaded += 1

    # ========== SCORING SYSTEM PATTERNS ==========
    logger.info("[3/6] Loading Scoring System Patterns...")

    await service.store_pattern(
        category="scoring_pattern",
        pattern_data={
            "pattern_name": "STRONG/ACCEPTABLE/WEAK/FAIL Bands",
            "description": "Qualitative bands instead of numeric scales. STRONG=1.0, ACCEPTABLE=0.75, WEAK=0.5, FAIL=0.0",
            "when_to_use": "Subjective quality evaluation where numeric precision is false precision",
            "benefit": "Clear semantic meaning, easier for judges to decide, aggregates well",
            "example": "STRONG: Fully meets criteria. ACCEPTABLE: Minor gaps. WEAK: Notable gaps. FAIL: Fundamental failures."
        }
    )
    patterns_loaded += 1

    await service.store_pattern(
        category="scoring_pattern",
        pattern_data={
            "pattern_name": "Binary PASS/FAIL",
            "description": "Simple binary for deterministic checks. No gradients.",
            "when_to_use": "Logic, math, schema validation, hard constraints",
            "benefit": "Eliminates ambiguity, clear failure signal",
            "example": "reference_integrity: PASS (all IDs exist) or FAIL (orphaned references)"
        }
    )
    patterns_loaded += 1

    await service.store_pattern(
        category="scoring_pattern",
        pattern_data={
            "pattern_name": "Aggregate with Sub-Scores",
            "description": "Dimension score aggregates from sub-criteria. Enables dimension-level and sub-dimension health tracking.",
            "example": "interpretation_quality (avg of 3 sub-scores) -> overall_score (avg of 4 dimensions)",
            "benefit": "Drill-down analysis, identifies specific weaknesses, enables targeted improvement"
        }
    )
    patterns_loaded += 1

    # ========== FAILURE MODE PATTERNS ==========
    logger.info("[4/6] Loading Failure Mode Patterns...")

    await service.store_pattern(
        category="failure_mode",
        pattern_data={
            "dimension": "interpretation_quality",
            "failure_modes": [
                "Wrong standards applied",
                "Missed patterns",
                "Generic findings",
                "Forced coherence"
            ],
            "detection": "Check if grounding principles were followed, if obvious patterns in signals were caught, if findings are specific vs generic",
            "example": "Diagnosis judges SDR call against executive value articulation criteria (wrong standard)"
        }
    )
    patterns_loaded += 1

    await service.store_pattern(
        category="failure_mode",
        pattern_data={
            "dimension": "evidentiary_support",
            "failure_modes": [
                "Claims without signal support",
                "Vague citations",
                "Broken traceability"
            ],
            "detection": "Verify each claim has cited evidence, snippets are specific, signalIds are valid",
            "example": "Claim: 'Rep consistently fails at discovery' but only one call example cited (insufficient evidence)"
        }
    )
    patterns_loaded += 1

    await service.store_pattern(
        category="failure_mode",
        pattern_data={
            "dimension": "coherence",
            "failure_modes": [
                "Contradictions",
                "Narrative doesn't match findings",
                "Score misalignment"
            ],
            "detection": "Cross-check strengths vs gaps for contradictions, verify narrative reflects findings, ensure scores align with severity",
            "example": "Strength: 'Excellent discovery skills' but Gap: 'Fails to ask discovery questions' (contradiction)"
        }
    )
    patterns_loaded += 1

    await service.store_pattern(
        category="failure_mode",
        pattern_data={
            "dimension": "communicability",
            "failure_modes": [
                "Jargon-heavy",
                "Not actionable",
                "Reads like algorithm output"
            ],
            "detection": "Check for domain jargon without explanation, vague recommendations, mechanical voice",
            "example": "Output: 'Suboptimal conversational vector alignment' instead of 'Talked too much, didn't listen'"
        }
    )
    patterns_loaded += 1

    # ========== SAMPLING STRATEGY PATTERNS ==========
    logger.info("[5/6] Loading Sampling Strategy Patterns...")

    await service.store_pattern(
        category="sampling_strategy",
        pattern_data={
            "pattern": "High Coverage for New/Changed Systems",
            "description": "100% sampling for: First 20-50 runs, new features, after prompt/grounding changes, edge cases",
            "rationale": "Validate changes work before reducing coverage",
            "steady_state": "5-20% random sampling once validated",
            "example": "New customer onboarding: 100% for first 20 diagnostics, then 10% steady state"
        }
    )
    patterns_loaded += 1

    await service.store_pattern(
        category="sampling_strategy",
        pattern_data={
            "pattern": "Alert Thresholds",
            "description": "Trigger investigation if: Any dimension < 60%, any FAIL, dimension degradation > 10% week-over-week",
            "use_case": "Production monitoring, regression detection",
            "example": "Evidentiary support drops from 85% to 70% → investigate prompt or data quality issues"
        }
    )
    patterns_loaded += 1

    # ========== INTENT KEYWORDS ==========
    logger.info("[6/6] Loading Intent Keywords...")

    await service.store_pattern(
        category="intent_keyword",
        pattern_data={
            "keywords": ["structured output", "JSON", "schema", "format"],
            "recommended_pattern": "Layer 1 + Layer 2 Split",
            "reasoning": "Structured outputs need schema validation (Layer 1) + semantic quality (Layer 2)",
            "recommended_dimensions": [
                "schema_validity (Layer 1, Binary)",
                "reference_integrity (Layer 1, Binary)",
                "content_quality (Layer 2, Gradient)"
            ]
        }
    )
    patterns_loaded += 1

    await service.store_pattern(
        category="intent_keyword",
        pattern_data={
            "keywords": ["recommendation", "classification", "framework", "categorization"],
            "recommended_pattern": "Atomic Splits",
            "recommended_dimensions": [
                "taxonomy_fit (Definition check)",
                "evidence_alignment (Context check)",
                "classification_accuracy (Correct category)",
                "scoring_integrity (Formula verification)"
            ],
            "reasoning": "Recommendation systems have multiple independent aspects that should be checked separately"
        }
    )
    patterns_loaded += 1

    await service.store_pattern(
        category="intent_keyword",
        pattern_data={
            "keywords": ["diagnostic", "analysis", "assessment", "evaluation"],
            "recommended_pattern": "Four Core Questions",
            "recommended_dimensions": [
                "interpretation_quality",
                "evidentiary_support",
                "coherence",
                "communicability"
            ],
            "scoring_pattern": "STRONG/ACCEPTABLE/WEAK/FAIL",
            "reasoning": "Diagnostic domains are subjective; focus on usefulness via four quality dimensions"
        }
    )
    patterns_loaded += 1

    await service.store_pattern(
        category="intent_keyword",
        pattern_data={
            "keywords": ["style", "template", "voice", "tone"],
            "dimension_type": "Binary Constraint",
            "example": "Title Style Enforcement (PASS/FAIL based on template compliance)",
            "reasoning": "Style requirements are usually deterministic templates or rules, not subjective gradients"
        }
    )
    patterns_loaded += 1

    # ========== DONE ==========
    logger.info("\n" + "=" * 60)
    logger.info(f"✅ Loaded {patterns_loaded} dimension design patterns")
    logger.info("=" * 60)

    # Print stats
    stats = service.get_stats()
    logger.info("\nPattern Statistics:")
    logger.info(f"  Total Patterns: {stats['total_patterns']}")
    logger.info("  By Category:")
    for category, count in sorted(stats['patterns_by_category'].items()):
        logger.info(f"    {category}: {count}")

    return patterns_loaded


async def test_pattern_retrieval():
    """Test pattern retrieval with sample prompts."""
    service = get_dimension_pattern_service()

    logger.info("\n" + "=" * 60)
    logger.info("Testing Pattern Retrieval")
    logger.info("=" * 60)

    # Test 1: Structured output prompt
    test_prompt_1 = """
    You are a recommendation engine. Generate JSON output with the following schema:
    {
      "recommendations": [{"id": string, "title": string, "score": number}]
    }
    Ensure all IDs are valid and scores follow the formula: base_score * priority_weight.
    """

    logger.info("\n--- Test 1: Structured Output + Recommendation ---")
    characteristics_1 = await service.analyze_prompt_characteristics(test_prompt_1)
    logger.info(f"Detected characteristics: {characteristics_1}")

    patterns_1 = await service.retrieve_relevant_patterns(
        system_prompt=test_prompt_1,
        prompt_characteristics=characteristics_1,
        top_k=3
    )
    logger.info("Retrieved patterns:")
    logger.info(patterns_1[:500] + "..." if len(patterns_1) > 500 else patterns_1)

    # Test 2: Diagnostic prompt
    test_prompt_2 = """
    Analyze the sales call transcript and provide a behavioral diagnostic.
    Identify strengths and gaps with specific evidence citations.
    Ensure findings are coherent and actionable for coaching.
    """

    logger.info("\n--- Test 2: Diagnostic/Analysis ---")
    characteristics_2 = await service.analyze_prompt_characteristics(test_prompt_2)
    logger.info(f"Detected characteristics: {characteristics_2}")

    patterns_2 = await service.retrieve_relevant_patterns(
        system_prompt=test_prompt_2,
        prompt_characteristics=characteristics_2,
        top_k=3
    )
    logger.info("Retrieved patterns:")
    logger.info(patterns_2[:500] + "..." if len(patterns_2) > 500 else patterns_2)


async def main():
    """Main function to load patterns and run tests."""
    try:
        # Load all patterns
        await load_all_patterns()

        # Run tests
        await test_pattern_retrieval()

        logger.info("\n✅ Dimension pattern service initialized successfully!")

    except Exception as e:
        logger.error(f"Error loading patterns: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
