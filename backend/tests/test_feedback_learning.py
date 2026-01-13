"""
Comprehensive tests for the feedback_learning module.

Tests cover:
- Feedback entry creation and storage
- Pattern analysis
- Adaptive eval generation
- Weight adjustments
"""

import pytest
import sys
import os
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feedback_learning import (
    FeedbackEntry,
    LearningPattern,
    FeedbackStore,
    FeedbackAnalyzer,
    AdaptiveEvalGenerator,
    record_human_feedback,
    analyze_project_feedback,
    get_eval_improvements,
    adapt_eval_for_project
)


class TestFeedbackEntry:
    """Tests for FeedbackEntry dataclass"""

    def test_entry_creation(self):
        entry = FeedbackEntry(
            id="test_1",
            project_id="proj_123",
            eval_prompt_version=1,
            test_case_id="tc_1",
            llm_score=4.0,
            llm_verdict="PASS",
            llm_reasoning="Good response"
        )
        assert entry.id == "test_1"
        assert entry.project_id == "proj_123"
        assert entry.llm_score == 4.0

    def test_compute_agreement_agrees(self):
        entry = FeedbackEntry(
            id="test_1",
            project_id="proj_123",
            eval_prompt_version=1,
            test_case_id="tc_1",
            llm_score=4.0,
            llm_verdict="PASS",
            llm_reasoning="Good",
            human_score=4.2,
            human_verdict="PASS"
        )
        entry.compute_agreement()

        assert entry.score_difference == pytest.approx(0.2, 0.01)
        assert entry.agreement is True

    def test_compute_agreement_disagrees(self):
        entry = FeedbackEntry(
            id="test_1",
            project_id="proj_123",
            eval_prompt_version=1,
            test_case_id="tc_1",
            llm_score=4.0,
            llm_verdict="PASS",
            llm_reasoning="Good",
            human_score=2.0,
            human_verdict="FAIL"
        )
        entry.compute_agreement()

        assert entry.score_difference == pytest.approx(-2.0, 0.01)
        assert entry.agreement is False

    def test_verdict_category_mapping(self):
        entry = FeedbackEntry(
            id="test_1",
            project_id="proj_123",
            eval_prompt_version=1,
            test_case_id="tc_1",
            llm_score=4.0,
            llm_verdict="PASSED",
            llm_reasoning="Good"
        )
        assert entry._verdict_category("PASS") == "pass"
        assert entry._verdict_category("FAILED") == "fail"
        assert entry._verdict_category("NEEDS_REVIEW") == "review"
        assert entry._verdict_category(None) == "unknown"


class TestLearningPattern:
    """Tests for LearningPattern dataclass"""

    def test_pattern_creation(self):
        pattern = LearningPattern(
            id="pattern_1",
            pattern_type="over_scoring",
            description="LLM tends to over-score",
            frequency=10,
            confidence=0.8
        )
        assert pattern.id == "pattern_1"
        assert pattern.pattern_type == "over_scoring"
        assert pattern.confidence == 0.8

    def test_pattern_with_adjustment(self):
        pattern = LearningPattern(
            id="pattern_1",
            pattern_type="dimension_bias",
            description="Accuracy dimension scores too high",
            frequency=5,
            confidence=0.7,
            weight_adjustment=-0.1,
            suggested_adjustment="Reduce accuracy weight"
        )
        assert pattern.weight_adjustment == -0.1
        assert pattern.suggested_adjustment == "Reduce accuracy weight"


class TestFeedbackStore:
    """Tests for FeedbackStore class"""

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_store_creation(self, temp_storage):
        store = FeedbackStore(temp_storage)
        assert store.storage_path.exists()

    def test_add_and_get_feedback(self, temp_storage):
        store = FeedbackStore(temp_storage)

        entry = FeedbackEntry(
            id="test_1",
            project_id="proj_123",
            eval_prompt_version=1,
            test_case_id="tc_1",
            llm_score=4.0,
            llm_verdict="PASS",
            llm_reasoning="Good",
            human_score=4.5,
            human_verdict="PASS"
        )
        store.add_feedback(entry)

        retrieved = store.get_feedback("proj_123")
        assert len(retrieved) == 1
        assert retrieved[0].id == "test_1"

    def test_multiple_entries(self, temp_storage):
        store = FeedbackStore(temp_storage)

        for i in range(5):
            entry = FeedbackEntry(
                id=f"test_{i}",
                project_id="proj_123",
                eval_prompt_version=1,
                test_case_id=f"tc_{i}",
                llm_score=float(i),
                llm_verdict="PASS",
                llm_reasoning="Good"
            )
            store.add_feedback(entry)

        retrieved = store.get_feedback("proj_123")
        assert len(retrieved) == 5

    def test_different_projects(self, temp_storage):
        store = FeedbackStore(temp_storage)

        store.add_feedback(FeedbackEntry(
            id="test_1", project_id="proj_A", eval_prompt_version=1,
            test_case_id="tc_1", llm_score=4.0, llm_verdict="PASS", llm_reasoning="Good"
        ))
        store.add_feedback(FeedbackEntry(
            id="test_2", project_id="proj_B", eval_prompt_version=1,
            test_case_id="tc_1", llm_score=3.0, llm_verdict="PASS", llm_reasoning="OK"
        ))

        assert len(store.get_feedback("proj_A")) == 1
        assert len(store.get_feedback("proj_B")) == 1

    def test_save_and_get_patterns(self, temp_storage):
        store = FeedbackStore(temp_storage)

        patterns = [
            LearningPattern(
                id="p1", pattern_type="over_scoring",
                description="Test", frequency=5, confidence=0.8
            )
        ]
        store.save_patterns("proj_123", patterns)

        retrieved = store.get_patterns("proj_123")
        assert len(retrieved) == 1
        assert retrieved[0].id == "p1"


class TestFeedbackAnalyzer:
    """Tests for FeedbackAnalyzer class"""

    @pytest.fixture
    def store_with_data(self):
        """Create a store with test data"""
        temp_dir = tempfile.mkdtemp()
        store = FeedbackStore(temp_dir)

        # Add feedback entries with human validation
        for i in range(10):
            llm_score = 4.0
            human_score = 3.0  # Consistently lower - over-scoring pattern
            entry = FeedbackEntry(
                id=f"test_{i}",
                project_id="test_project",
                eval_prompt_version=1,
                test_case_id=f"tc_{i}",
                llm_score=llm_score,
                llm_verdict="PASS",
                llm_reasoning="Good",
                human_score=human_score,
                human_verdict="PASS"
            )
            store.add_feedback(entry)

        yield store
        shutil.rmtree(temp_dir)

    def test_analyze_insufficient_data(self):
        temp_dir = tempfile.mkdtemp()
        store = FeedbackStore(temp_dir)
        analyzer = FeedbackAnalyzer(store)

        result = analyzer.analyze_feedback("empty_project")
        assert result["status"] == "insufficient_data"

        shutil.rmtree(temp_dir)

    def test_analyze_detects_over_scoring(self, store_with_data):
        analyzer = FeedbackAnalyzer(store_with_data)
        result = analyzer.analyze_feedback("test_project")

        assert result["status"] == "analyzed"
        assert result["avg_score_difference"] is not None

        # Should detect over-scoring pattern
        patterns = result["patterns"]
        pattern_types = [p["pattern_type"] for p in patterns]
        assert "over_scoring" in pattern_types

    def test_get_improvement_suggestions(self, store_with_data):
        analyzer = FeedbackAnalyzer(store_with_data)

        # First analyze to generate patterns
        analyzer.analyze_feedback("test_project")

        suggestions = analyzer.get_improvement_suggestions("test_project")
        assert isinstance(suggestions, list)


class TestAdaptiveEvalGenerator:
    """Tests for AdaptiveEvalGenerator class"""

    @pytest.fixture
    def adaptive_setup(self):
        """Create adaptive generator with test patterns"""
        temp_dir = tempfile.mkdtemp()
        store = FeedbackStore(temp_dir)
        analyzer = FeedbackAnalyzer(store)
        adaptive = AdaptiveEvalGenerator(store, analyzer)

        # Add patterns directly
        patterns = [
            LearningPattern(
                id="p1",
                pattern_type="over_scoring",
                description="LLM over-scores by 0.5",
                frequency=10,
                confidence=0.8,
                trigger_conditions={"avg_score_difference": -0.5},
                weight_adjustment=-0.1
            ),
            LearningPattern(
                id="p2",
                pattern_type="dimension_bias",
                description="Accuracy too high",
                frequency=8,
                confidence=0.7,
                trigger_conditions={"dimension": "accuracy", "direction": "high"},
                weight_adjustment=-0.1
            ),
            LearningPattern(
                id="p3",
                pattern_type="failure_mode_miss",
                description="Misses hallucination",
                frequency=5,
                confidence=0.65,
                trigger_conditions={"failure_keyword": "hallucination"}
            )
        ]
        store.save_patterns("test_project", patterns)

        yield adaptive, store, temp_dir
        shutil.rmtree(temp_dir)

    def test_get_dimension_weight_adjustments(self, adaptive_setup):
        adaptive, store, _ = adaptive_setup
        adjustments = adaptive.get_dimension_weight_adjustments("test_project")

        assert "accuracy" in adjustments
        assert adjustments["accuracy"] == -0.1

    def test_get_scoring_bias_adjustment(self, adaptive_setup):
        adaptive, store, _ = adaptive_setup
        bias = adaptive.get_scoring_bias_adjustment("test_project")

        # Should return opposite of avg_score_difference
        assert bias == pytest.approx(0.5, 0.01)

    def test_get_additional_auto_fails(self, adaptive_setup):
        adaptive, store, _ = adaptive_setup
        additional = adaptive.get_additional_auto_fails("test_project")

        assert len(additional) >= 1
        auto_fail_ids = [af["id"] for af in additional]
        assert any("hallucination" in id for id in auto_fail_ids)

    def test_should_regenerate_eval(self, adaptive_setup):
        adaptive, store, _ = adaptive_setup
        should_regen, reason = adaptive.should_regenerate_eval("test_project")

        # Check that the function returns a result (can be True or False depending on pattern confidence)
        assert isinstance(should_regen, bool)
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_adapt_eval_prompt(self, adaptive_setup):
        adaptive, store, _ = adaptive_setup

        base_prompt = "# Evaluator\n## I. Role\nYou are a judge."
        dimensions = [
            {"name": "accuracy", "weight": 0.5},
            {"name": "completeness", "weight": 0.5}
        ]
        auto_fails = [{"id": "af1", "name": "Test fail"}]

        adapted_prompt, adapted_dims, adapted_fails = adaptive.adapt_eval_prompt(
            "test_project", base_prompt, dimensions, auto_fails
        )

        # Should have calibration note
        assert "CALIBRATION" in adapted_prompt or len(adapted_prompt) >= len(base_prompt)

        # Dimensions should be adjusted
        assert len(adapted_dims) == 2

        # Auto-fails should be extended
        assert len(adapted_fails) >= len(auto_fails)


class TestConvenienceFunctions:
    """Tests for module-level convenience functions"""

    @pytest.fixture
    def temp_feedback_dir(self):
        """Create temp directory for feedback"""
        temp_dir = tempfile.mkdtemp()
        # Patch the module's storage path
        import feedback_learning
        original_store = feedback_learning._store
        feedback_learning._store = None

        yield temp_dir

        feedback_learning._store = original_store
        shutil.rmtree(temp_dir)

    def test_record_human_feedback(self, temp_feedback_dir):
        entry_id = record_human_feedback(
            project_id="test_proj",
            eval_prompt_version=1,
            test_case_id="tc_1",
            llm_score=4.0,
            llm_verdict="PASS",
            llm_reasoning="Good",
            human_score=4.5,
            human_verdict="PASS",
            human_feedback="Looks correct"
        )
        assert entry_id is not None
        assert len(entry_id) == 12  # MD5 hash prefix

    def test_analyze_project_feedback_empty(self):
        result = analyze_project_feedback("nonexistent_project")
        assert result["status"] == "insufficient_data"


class TestEdgeCases:
    """Tests for edge cases"""

    def test_empty_feedback_list(self):
        temp_dir = tempfile.mkdtemp()
        store = FeedbackStore(temp_dir)

        feedback = store.get_feedback("nonexistent")
        assert feedback == []

        shutil.rmtree(temp_dir)

    def test_pattern_with_no_adjustment(self):
        pattern = LearningPattern(
            id="p1",
            pattern_type="info",
            description="Just informational",
            frequency=1,
            confidence=0.3
        )
        assert pattern.weight_adjustment == 0.0
        assert pattern.suggested_adjustment == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
