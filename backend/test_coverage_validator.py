"""
Quick test script for coverage validation module.

Run this to verify the installation works correctly:
    python test_coverage_validator.py

Author: Athena Team
"""

import asyncio
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_imports():
    """Test 1: Verify all modules can be imported"""
    print("\n" + "="*60)
    print("TEST 1: Module Imports")
    print("="*60)

    try:
        from eval_coverage_validator import (
            CoverageValidator,
            SystemPromptAnalyzer,
            EvalCoverageAnalyzer,
            EvalPromptImprover,
            validate_eval_coverage,
            CoverageLevel
        )
        print("✓ eval_coverage_validator imports successful")

        from validated_eval_pipeline import (
            ValidatedEvalPipeline,
            COVERAGE_VALIDATION_AVAILABLE
        )
        print("✓ validated_eval_pipeline imports successful")
        print(f"  Coverage validation available: {COVERAGE_VALIDATION_AVAILABLE}")

        return True

    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


async def test_system_prompt_analysis():
    """Test 2: System prompt analysis (pattern-based, no LLM)"""
    print("\n" + "="*60)
    print("TEST 2: System Prompt Analysis (Pattern-Based)")
    print("="*60)

    try:
        from eval_coverage_validator import SystemPromptAnalyzer

        analyzer = SystemPromptAnalyzer()

        system_prompt = """
        You are a financial advisor assistant.

        You must:
        - Provide investment advice based on user risk profile
        - Calculate portfolio diversification scores
        - Explain financial terms in simple language

        You must never:
        - Provide specific stock recommendations
        - Guarantee returns or outcomes
        - Access user's actual financial accounts

        Quality requirements:
        - Answers must be accurate and evidence-based
        - Tone must be professional and trustworthy
        - Explanations must be clear and concise

        Output format: JSON with fields: advice, risk_assessment, disclaimer
        """

        result = analyzer._pattern_analyze(system_prompt)

        print(f"\nExtracted Analysis:")
        print(f"  Capabilities found: {len(result.key_capabilities)}")
        print(f"  Constraints found: {len(result.critical_constraints)}")
        print(f"  Quality criteria: {len(result.quality_criteria)}")
        print(f"  Output requirements: {len(result.output_requirements)}")

        if len(result.key_capabilities) > 0:
            print(f"\n  Sample capability: {result.key_capabilities[0][:80]}...")

        if len(result.critical_constraints) > 0:
            print(f"  Sample constraint: {result.critical_constraints[0][:80]}...")

        print("\n✓ System prompt analysis working")
        return True

    except Exception as e:
        print(f"\n✗ System prompt analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_coverage_analysis():
    """Test 3: Coverage analysis (heuristic, no LLM)"""
    print("\n" + "="*60)
    print("TEST 3: Coverage Analysis (Heuristic)")
    print("="*60)

    try:
        from eval_coverage_validator import (
            SystemPromptAnalyzer,
            EvalCoverageAnalyzer,
            SystemPromptAnalysis
        )

        # Create a simple system analysis
        system_analysis = SystemPromptAnalysis(
            primary_purpose="Provide financial advice",
            key_capabilities=[
                "Provide investment advice",
                "Calculate portfolio scores",
                "Explain financial terms"
            ],
            critical_constraints=[
                "Never provide specific stock recommendations",
                "Never guarantee returns"
            ],
            quality_criteria=[
                "Must be accurate",
                "Must be professional"
            ],
            input_expectations=["User risk profile"],
            output_requirements=["JSON format", "Include disclaimer"],
            edge_cases=[],
            domain_specific_terms=["portfolio", "diversification"]
        )

        # A basic eval prompt
        eval_prompt = """
        Evaluate the financial advice response.

        Dimensions:
        1. Accuracy: Is the advice accurate and evidence-based?
        2. Professionalism: Is the tone professional?
        3. Format: Is output valid JSON?

        Auto-fails:
        - Guarantees specific returns
        """

        analyzer = EvalCoverageAnalyzer()
        coverage_result = analyzer._heuristic_coverage_analysis(
            system_analysis=system_analysis,
            eval_prompt=eval_prompt,
            dimensions=[
                {"name": "Accuracy", "weight": 0.5},
                {"name": "Professionalism", "weight": 0.3},
                {"name": "Format", "weight": 0.2}
            ],
            auto_fails=[
                {"name": "guarantees_returns", "description": "Promises guaranteed returns"}
            ]
        )

        print(f"\nCoverage Result:")
        print(f"  Coverage: {coverage_result.overall_coverage_pct}%")
        print(f"  Level: {coverage_result.coverage_level.value}")
        print(f"  Passes threshold: {coverage_result.passes_threshold}")
        print(f"  Gaps found: {len(coverage_result.gaps)}")
        print(f"  Well covered: {len(coverage_result.well_covered_aspects)}")

        if coverage_result.gaps:
            print(f"\n  Sample gap:")
            gap = coverage_result.gaps[0]
            print(f"    Category: {gap.category}")
            print(f"    Severity: {gap.severity}")
            print(f"    Missing: {gap.missing_aspect[:60]}...")

        print("\n✓ Coverage analysis working")
        return True

    except Exception as e:
        print(f"\n✗ Coverage analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_pipeline_integration():
    """Test 4: Pipeline integration check"""
    print("\n" + "="*60)
    print("TEST 4: Pipeline Integration")
    print("="*60)

    try:
        from validated_eval_pipeline import (
            ValidatedEvalPipeline,
            CostBudget,
            COVERAGE_VALIDATION_AVAILABLE
        )

        if not COVERAGE_VALIDATION_AVAILABLE:
            print("⚠ Coverage validation not available in pipeline")
            print("  This is expected if llm_client_v2 or other dependencies are missing")
            return True

        # Just test initialization
        pipeline = ValidatedEvalPipeline(
            project_id="test_project",
            enable_coverage_validation=True,
            max_coverage_iterations=2,
            cost_budget=CostBudget(max_cost_per_validation=1.00)
        )

        print(f"\nPipeline Configuration:")
        print(f"  Project ID: {pipeline.project_id}")
        print(f"  Coverage enabled: {pipeline.enable_coverage_validation}")
        print(f"  Max iterations: {pipeline.max_coverage_iterations}")
        print(f"  Coverage validator: {pipeline.coverage_validator}")

        print("\n✓ Pipeline integration working")
        return True

    except Exception as e:
        print(f"\n✗ Pipeline integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_data_models():
    """Test 5: Data model instantiation"""
    print("\n" + "="*60)
    print("TEST 5: Data Models")
    print("="*60)

    try:
        from eval_coverage_validator import (
            CoverageGap,
            CoverageAnalysisResult,
            EvalImprovementSuggestion,
            CoverageLevel
        )

        # Test CoverageGap
        gap = CoverageGap(
            category="capability",
            missing_aspect="Test capability",
            severity="high",
            current_coverage="Not tested",
            recommended_addition="Add test dimension",
            example_test_case="Example case"
        )
        print(f"✓ CoverageGap: {gap.category} - {gap.severity}")

        # Test CoverageAnalysisResult
        result = CoverageAnalysisResult(
            overall_coverage_pct=75.5,
            coverage_level=CoverageLevel.GOOD,
            gaps=[gap],
            well_covered_aspects=["Aspect 1", "Aspect 2"],
            improvement_priority=["Fix gap 1"],
            passes_threshold=False
        )
        print(f"✓ CoverageAnalysisResult: {result.coverage_level.value}")

        # Test EvalImprovementSuggestion
        suggestion = EvalImprovementSuggestion(
            section_to_modify="dimensions",
            action="add",
            specific_change="Add new dimension",
            reason="Missing coverage",
            priority=1
        )
        print(f"✓ EvalImprovementSuggestion: {suggestion.action}")

        print("\n✓ All data models working")
        return True

    except Exception as e:
        print(f"\n✗ Data model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n")
    print("╔" + "═"*58 + "╗")
    print("║" + " "*15 + "COVERAGE VALIDATOR TESTS" + " "*19 + "║")
    print("╚" + "═"*58 + "╝")

    results = []

    # Run tests
    results.append(("Module Imports", await test_imports()))
    results.append(("System Prompt Analysis", await test_system_prompt_analysis()))
    results.append(("Coverage Analysis", await test_coverage_analysis()))
    results.append(("Pipeline Integration", await test_pipeline_integration()))
    results.append(("Data Models", await test_data_models()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Coverage validation is ready to use.")
        print("\nNext steps:")
        print("  1. Review COVERAGE_VALIDATION_README.md for usage guide")
        print("  2. Integrate into your eval generation workflow")
    else:
        print("\n⚠ Some tests failed. Check error messages above.")
        print("\nCommon issues:")
        print("  - Missing dependencies (llm_client_v2, eval_quality_system)")
        print("  - Import path issues")
        print("  - Check that all required modules are in backend/")

    print("\n" + "="*60 + "\n")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
