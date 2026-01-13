"""
Tests for eval_prompt_quality module.

Tests cover all 6 quality improvements:
1. Calibration examples (8-10 with edge cases)
2. Numeric rubric thresholds
3. Tiebreaker rules
4. Token optimization (30% reduction)
5. Consistency checking (±0.3 tolerance)
6. Intermediate score guidance
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_prompt_quality import (
    # Section 1: Numeric thresholds
    NUMERIC_RUBRIC_THRESHOLDS,
    get_numeric_rubric,

    # Section 2: Tiebreaker rules
    TIEBREAKER_RULES,
    TIEBREAKER_RULES_COMPACT,

    # Section 3: Calibration examples
    CalibrationExample,
    generate_calibration_examples,
    format_calibration_examples_for_prompt,

    # Section 4: Compact prompt builder
    build_production_eval_prompt,

    # Section 5: Consistency checker
    ConsistencyChecker,
    ConsistencyCheckResult,
    get_consistency_checker,

    # Section 6: Intermediate scores
    INTERMEDIATE_SCORE_GUIDANCE,

    # Section 7-9: Integration
    enhance_existing_rubric,
    calculate_weighted_score,
    determine_verdict,
    create_enhanced_eval_prompt,
    estimate_tokens,
    optimize_prompt_tokens,
)


class TestNumericRubricThresholds:
    """Test Section 1: Numeric thresholds in rubrics"""

    def test_all_dimensions_have_thresholds(self):
        """Verify all 6 core dimensions have numeric thresholds"""
        expected_dimensions = [
            "accuracy", "completeness", "format",
            "relevance", "clarity", "instruction_following"
        ]
        for dim in expected_dimensions:
            assert dim in NUMERIC_RUBRIC_THRESHOLDS, f"Missing dimension: {dim}"

    def test_each_dimension_has_5_scores(self):
        """Each dimension should have scores 1-5"""
        for dim, rubric in NUMERIC_RUBRIC_THRESHOLDS.items():
            for score in [1, 2, 3, 4, 5]:
                assert score in rubric, f"{dim} missing score {score}"

    def test_thresholds_have_percentages(self):
        """Each threshold should have min_pct defined"""
        for dim, rubric in NUMERIC_RUBRIC_THRESHOLDS.items():
            for score, threshold in rubric.items():
                assert "min_pct" in threshold, f"{dim} score {score} missing min_pct"
                assert "description" in threshold, f"{dim} score {score} missing description"

    def test_thresholds_are_ordered(self):
        """Higher scores should have higher thresholds"""
        for dim, rubric in NUMERIC_RUBRIC_THRESHOLDS.items():
            for score in [2, 3, 4, 5]:
                assert rubric[score]["min_pct"] >= rubric[score - 1]["min_pct"], \
                    f"{dim}: score {score} threshold should be >= score {score-1}"

    def test_get_numeric_rubric_known_dimension(self):
        """Test getting rubric for known dimension"""
        rubric = get_numeric_rubric("accuracy")
        assert 5 in rubric
        assert rubric[5]["min_pct"] == 95

    def test_get_numeric_rubric_unknown_dimension(self):
        """Unknown dimension should fall back to accuracy"""
        rubric = get_numeric_rubric("unknown_dimension")
        assert rubric == NUMERIC_RUBRIC_THRESHOLDS["accuracy"]

    def test_get_numeric_rubric_variations(self):
        """Test dimension name variations are mapped correctly"""
        # "accuracy_faithfulness" should map to "accuracy"
        rubric1 = get_numeric_rubric("accuracy_faithfulness")
        rubric2 = get_numeric_rubric("accuracy")
        assert rubric1 == rubric2

        # "format_compliance" should map to "format"
        rubric3 = get_numeric_rubric("format_compliance")
        rubric4 = get_numeric_rubric("format")
        assert rubric3 == rubric4


class TestTiebreakerRules:
    """Test Section 2: Tiebreaker rules"""

    def test_tiebreaker_rules_exist(self):
        """Verify tiebreaker rules are defined"""
        assert len(TIEBREAKER_RULES) > 100, "Full tiebreaker rules should be comprehensive"
        assert len(TIEBREAKER_RULES_COMPACT) > 50, "Compact rules should exist"

    def test_tiebreaker_rules_content(self):
        """Check key elements in tiebreaker rules"""
        assert "Auto-Fail" in TIEBREAKER_RULES
        assert "Accuracy" in TIEBREAKER_RULES
        assert "weighted" in TIEBREAKER_RULES.lower()
        assert "0.5" in TIEBREAKER_RULES  # Round to nearest 0.5

    def test_compact_rules_shorter(self):
        """Compact rules should be shorter than full rules"""
        assert len(TIEBREAKER_RULES_COMPACT) < len(TIEBREAKER_RULES)


class TestCalibrationExamples:
    """Test Section 3: Calibration examples"""

    def test_generate_calibration_examples_count(self):
        """Should generate 8-10 calibration examples"""
        dimensions = [{"name": "Accuracy", "weight": 0.5}]
        examples = generate_calibration_examples("Test purpose", dimensions)
        assert 8 <= len(examples) <= 12, f"Expected 8-12 examples, got {len(examples)}"

    def test_examples_cover_score_range(self):
        """Examples should cover full score range (1-5)"""
        dimensions = [{"name": "Quality", "weight": 1.0}]
        examples = generate_calibration_examples("Test", dimensions)

        scores = [ex.score for ex in examples]
        assert min(scores) == 1.0, "Should include score 1"
        assert max(scores) == 5.0, "Should include score 5"

    def test_examples_include_intermediate_scores(self):
        """Should include intermediate scores (2.5, 3.5, 4.5)"""
        dimensions = [{"name": "Quality", "weight": 1.0}]
        examples = generate_calibration_examples("Test", dimensions)

        scores = [ex.score for ex in examples]
        intermediate_scores = [1.5, 2.5, 3.5, 4.5]
        found_intermediates = [s for s in intermediate_scores if s in scores]
        assert len(found_intermediates) >= 2, f"Should have at least 2 intermediate scores, found {found_intermediates}"

    def test_examples_include_edge_cases(self):
        """Should include edge case examples"""
        dimensions = [{"name": "Quality", "weight": 1.0}]
        examples = generate_calibration_examples("Test", dimensions)

        edge_cases = [ex for ex in examples if ex.is_edge_case]
        assert len(edge_cases) >= 3, f"Should have at least 3 edge cases, got {len(edge_cases)}"

    def test_calibration_example_structure(self):
        """Each example should have required fields"""
        dimensions = [{"name": "Quality", "weight": 1.0}]
        examples = generate_calibration_examples("Test", dimensions)

        for ex in examples:
            assert ex.id, "Should have id"
            assert 1.0 <= ex.score <= 5.0, "Score should be 1-5"
            assert ex.verdict in ["PASS", "FAIL", "NEEDS_REVIEW"], f"Invalid verdict: {ex.verdict}"
            assert ex.input_summary, "Should have input_summary"
            assert ex.output_summary, "Should have output_summary"
            assert ex.reasoning, "Should have reasoning"

    def test_format_calibration_examples(self):
        """Test formatting examples for prompt inclusion"""
        dimensions = [{"name": "Accuracy", "weight": 1.0}]
        examples = generate_calibration_examples("Test", dimensions)

        formatted = format_calibration_examples_for_prompt(examples)
        assert "Calibration Examples" in formatted
        assert "Score 5" in formatted or "Score 5.0" in formatted
        assert "Score 1" in formatted or "Score 1.0" in formatted
        assert "[EDGE CASE]" in formatted


class TestProductionPromptBuilder:
    """Test Section 4: Compact eval prompt builder"""

    def test_build_production_eval_prompt_structure(self):
        """Verify prompt has all required sections"""
        dimensions = [
            {"name": "Accuracy", "weight": 0.5},
            {"name": "Completeness", "weight": 0.5}
        ]
        auto_fails = [{"name": "hallucination", "description": "Fabricated info"}]

        prompt = build_production_eval_prompt(
            system_purpose="Test assistant",
            dimensions=dimensions,
            auto_fails=auto_fails
        )

        assert "Evaluator for: Test assistant" in prompt
        assert "Auto-Fail" in prompt
        assert "Scoring Dimensions" in prompt
        assert "Verdict Determination" in prompt
        assert "Output Format" in prompt
        assert "json" in prompt.lower()

    def test_includes_numeric_thresholds(self):
        """Prompt should include numeric percentage thresholds"""
        dimensions = [{"name": "Accuracy", "weight": 1.0}]

        prompt = build_production_eval_prompt(
            system_purpose="Test",
            dimensions=dimensions,
            auto_fails=[]
        )

        # Should have percentage thresholds like "≥95%" or "≥90%"
        assert "≥" in prompt or ">=" in prompt, "Should have threshold indicators"

    def test_includes_calibration_examples(self):
        """Prompt should include calibration examples when enabled"""
        dimensions = [{"name": "Accuracy", "weight": 1.0}]

        prompt = build_production_eval_prompt(
            system_purpose="Test",
            dimensions=dimensions,
            auto_fails=[],
            include_calibration=True
        )

        assert "Calibration" in prompt
        assert "Score 5" in prompt or "Score 5.0" in prompt

    def test_compact_mode_uses_compact_tiebreaker(self):
        """Compact mode should use shorter tiebreaker rules"""
        dimensions = [{"name": "Accuracy", "weight": 1.0}]

        prompt_compact = build_production_eval_prompt(
            system_purpose="Test",
            dimensions=dimensions,
            auto_fails=[],
            compact_mode=True
        )

        prompt_full = build_production_eval_prompt(
            system_purpose="Test",
            dimensions=dimensions,
            auto_fails=[],
            compact_mode=False
        )

        assert len(prompt_compact) < len(prompt_full), "Compact mode should be shorter"


class TestConsistencyChecker:
    """Test Section 5: Consistency checking"""

    def test_consistency_checker_constants(self):
        """Verify consistency checker configuration"""
        checker = get_consistency_checker()
        assert checker.MAX_ALLOWED_DEVIATION == 0.3, "Max deviation should be 0.3"
        assert checker.MIN_RUNS >= 2, "Should require at least 2 runs"

    def test_check_scores_consistent_identical(self):
        """Identical scores should be consistent"""
        checker = ConsistencyChecker()
        is_consistent, max_dev = checker.check_scores_consistent([4.0, 4.0, 4.0])
        assert is_consistent is True
        assert max_dev == 0.0

    def test_check_scores_consistent_within_tolerance(self):
        """Scores within 0.3 tolerance should be consistent"""
        checker = ConsistencyChecker()
        # Mean = 4.0, max deviation = 0.2
        is_consistent, max_dev = checker.check_scores_consistent([3.8, 4.0, 4.2])
        assert is_consistent is True
        assert max_dev <= 0.3

    def test_check_scores_inconsistent(self):
        """Scores with deviation > 0.3 should be inconsistent"""
        checker = ConsistencyChecker()
        # Mean ≈ 3.0, max deviation = 1.0
        is_consistent, max_dev = checker.check_scores_consistent([2.0, 3.0, 4.0])
        assert is_consistent is False
        assert max_dev > 0.3

    def test_check_scores_edge_case_single(self):
        """Single score should be considered consistent"""
        checker = ConsistencyChecker()
        is_consistent, max_dev = checker.check_scores_consistent([4.0])
        assert is_consistent is True

    def test_get_global_checker(self):
        """Global checker should be singleton"""
        checker1 = get_consistency_checker()
        checker2 = get_consistency_checker()
        assert checker1 is checker2, "Should return same instance"


class TestIntermediateScoreGuidance:
    """Test Section 6: Intermediate score guidance"""

    def test_guidance_exists(self):
        """Intermediate score guidance should be defined"""
        assert len(INTERMEDIATE_SCORE_GUIDANCE) > 100

    def test_guidance_covers_all_scores(self):
        """Guidance should cover all score levels"""
        for score in ["5.0", "4.5", "4.0", "3.5", "3.0", "2.5", "2.0", "1.5", "1.0"]:
            assert score in INTERMEDIATE_SCORE_GUIDANCE, f"Missing guidance for {score}"


class TestIntegrationFunctions:
    """Test Section 7-9: Integration helpers"""

    def test_enhance_existing_rubric(self):
        """Test enhancing vague rubrics with numeric thresholds"""
        existing = {
            5: "Excellent quality",
            4: "Good quality",
            3: "Acceptable",
            2: "Poor",
            1: "Very poor"
        }

        enhanced = enhance_existing_rubric(existing, "accuracy")

        for score in [1, 2, 3, 4, 5]:
            assert "[≥" in enhanced[score], f"Score {score} should have threshold prefix"

    def test_calculate_weighted_score_basic(self):
        """Test basic weighted score calculation"""
        scores = {"Accuracy": 4.0, "Completeness": 4.0}
        weights = {"Accuracy": 0.5, "Completeness": 0.5}

        result = calculate_weighted_score(scores, weights, apply_tiebreaker=False)
        assert result == 4.0

    def test_calculate_weighted_score_rounding(self):
        """Score should round to nearest 0.5"""
        scores = {"Accuracy": 4.3, "Completeness": 4.3}
        weights = {"Accuracy": 0.5, "Completeness": 0.5}

        result = calculate_weighted_score(scores, weights)
        assert result == 4.5 or result == 4.0, "Should round to nearest 0.5"

    def test_calculate_weighted_score_accuracy_priority(self):
        """When accuracy differs significantly, it should be weighted higher"""
        scores = {"Accuracy": 2.0, "Completeness": 5.0, "Format": 5.0}
        weights = {"Accuracy": 0.4, "Completeness": 0.3, "Format": 0.3}

        # With accuracy priority, the low accuracy should pull score down
        result = calculate_weighted_score(scores, weights, apply_tiebreaker=True)
        assert result < 4.0, "Low accuracy should pull score down"

    def test_determine_verdict_pass(self):
        """Score >= 3.5 should be PASS"""
        assert determine_verdict(4.0) == "PASS"
        assert determine_verdict(3.5) == "PASS"
        assert determine_verdict(5.0) == "PASS"

    def test_determine_verdict_needs_review(self):
        """Score 2.5-3.49 should be NEEDS_REVIEW"""
        assert determine_verdict(3.0) == "NEEDS_REVIEW"
        assert determine_verdict(2.5) == "NEEDS_REVIEW"

    def test_determine_verdict_fail(self):
        """Score < 2.5 should be FAIL"""
        assert determine_verdict(2.0) == "FAIL"
        assert determine_verdict(1.0) == "FAIL"

    def test_determine_verdict_auto_fail(self):
        """Auto-fail triggered should always be FAIL"""
        assert determine_verdict(5.0, auto_fail_triggered="hallucination") == "FAIL"

    def test_create_enhanced_eval_prompt(self):
        """Test full enhanced prompt creation"""
        dimensions = [
            {"name": "Accuracy", "weight": 0.5},
            {"name": "Completeness", "weight": 0.5}
        ]
        auto_fails = [{"name": "hallucination", "description": "Fabricated info"}]

        prompt, metadata = create_enhanced_eval_prompt(
            system_purpose="Customer support",
            dimensions=dimensions,
            auto_fails=auto_fails
        )

        # Check prompt content
        assert len(prompt) > 1000, "Should be substantial prompt"
        assert "Customer support" in prompt

        # Check metadata
        assert metadata["calibration_examples"] == 8
        assert metadata["numeric_thresholds"] is True
        assert metadata["tiebreaker_rules"] is True
        assert metadata["intermediate_scores"] is True
        assert metadata["consistency_check_required"] is True
        assert metadata["max_deviation_allowed"] == 0.3
        assert metadata["token_count"] > 0


class TestTokenOptimization:
    """Test token optimization (30% reduction target)"""

    def test_estimate_tokens(self):
        """Token estimation should be roughly 4 chars per token"""
        text = "a" * 400  # 400 chars
        tokens = estimate_tokens(text)
        assert 90 <= tokens <= 110, f"Expected ~100 tokens, got {tokens}"

    def test_optimize_removes_extra_whitespace(self):
        """Optimization should remove excessive whitespace"""
        text = "Line 1\n\n\n\n\nLine 2"
        optimized = optimize_prompt_tokens(text)
        assert "\n\n\n" not in optimized

    def test_optimize_replaces_verbose_phrases(self):
        """Should replace verbose phrases with concise ones"""
        text = "Please note that this is important. It is important to understand."
        optimized = optimize_prompt_tokens(text)
        assert "Please note that" not in optimized

    def test_optimization_preserves_meaning(self):
        """Optimization should preserve core content"""
        text = "Score 5: Perfect accuracy with zero errors"
        optimized = optimize_prompt_tokens(text)
        assert "Score 5" in optimized
        assert "accuracy" in optimized or "errors" in optimized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
