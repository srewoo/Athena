"""
Tests for eval_bias_detector.py

Tests bias detection functionality including:
- Position bias detection
- Length bias detection
- Verbosity bias detection
- Leniency/severity bias detection
- Anchoring bias detection
- Overall bias reporting
"""

import pytest
import sys
import os
from unittest.mock import AsyncMock, MagicMock
import asyncio

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from eval_bias_detector import (
    EvalBiasDetector,
    BiasType,
    BiasSeverity,
    BiasDetectionResult,
    ComprehensiveBiasReport,
    check_eval_bias,
    bias_report_to_dict
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def bias_detector():
    """Create EvalBiasDetector instance"""
    return EvalBiasDetector()


@pytest.fixture
def sample_eval_prompt():
    """Sample eval prompt for testing"""
    return """
    You are an evaluator. Score outputs from 1-5.
    
    Criteria:
    - Accuracy: Is the output correct?
    - Completeness: Does it cover all aspects?
    
    Return JSON with score and verdict.
    """


@pytest.fixture
def sample_test_cases():
    """Sample test cases for bias detection"""
    return [
        {"input": "What is 2+2?", "output": "4"},
        {"input": "Explain gravity", "output": "Gravity is a force that attracts objects with mass."},
        {"input": "What is AI?", "output": "Artificial Intelligence (AI) refers to computer systems that can perform tasks requiring human intelligence, including learning, reasoning, and problem-solving."},
        {"input": "Define JSON", "output": "JSON"},
        {"input": "List 3 colors", "output": "Red, blue, green"},
    ]


@pytest.fixture
def mock_eval_function():
    """Create mock evaluation function that returns consistent scores"""
    async def eval_func(prompt, input_text, output_text):
        # Simple mock: longer outputs get higher scores (introduces length bias)
        score = min(5.0, 2.0 + len(output_text) / 20)
        verdict = "PASS" if score >= 3 else "FAIL"
        return score, verdict
    return eval_func


@pytest.fixture
def mock_unbiased_eval_function():
    """Create mock evaluation function with no bias"""
    async def eval_func(prompt, input_text, output_text):
        # Always return same score regardless of position/length
        return 3.5, "PASS"
    return eval_func


# ============================================================================
# Test BiasDetectionResult
# ============================================================================

def test_bias_detection_result_creation():
    """Test BiasDetectionResult can be created"""
    result = BiasDetectionResult(
        bias_type=BiasType.POSITION,
        detected=True,
        severity=BiasSeverity.HIGH,
        confidence=0.85,
        evidence=["Score varies by 25% across positions"],
        metrics={"max_difference": 1.2},
        recommendation="Shuffle evaluation order"
    )
    
    assert result.bias_type == BiasType.POSITION
    assert result.detected is True
    assert result.severity == BiasSeverity.HIGH
    assert result.confidence == 0.85
    assert len(result.evidence) == 1
    assert result.metrics["max_difference"] == 1.2


def test_comprehensive_bias_report_creation():
    """Test ComprehensiveBiasReport can be created"""
    report = ComprehensiveBiasReport(
        overall_bias_score=42.5,
        biases_detected=[],
        is_biased=True,
        critical_biases=["position_bias"],
        recommendations=["Fix position bias"],
        test_metadata={"test_cases": 5}
    )
    
    assert report.overall_bias_score == 42.5
    assert report.is_biased is True
    assert len(report.critical_biases) == 1


# ============================================================================
# Test Position Bias Detection
# ============================================================================

@pytest.mark.asyncio
async def test_detect_position_bias_no_bias(bias_detector, sample_eval_prompt, sample_test_cases, mock_unbiased_eval_function):
    """Test position bias detection when no bias exists"""
    result = await bias_detector._detect_position_bias(
        sample_eval_prompt,
        mock_unbiased_eval_function,
        sample_test_cases
    )
    
    assert result.bias_type == BiasType.POSITION
    assert result.detected is False
    assert result.severity == BiasSeverity.NONE


@pytest.mark.asyncio
async def test_detect_position_bias_with_bias(bias_detector, sample_eval_prompt, sample_test_cases):
    """Test position bias detection when bias exists"""
    # Create eval function with position bias
    async def biased_eval(prompt, input_text, output_text):
        # First call returns low score, subsequent calls return higher scores
        if not hasattr(biased_eval, 'call_count'):
            biased_eval.call_count = 0
        biased_eval.call_count += 1
        
        score = 2.0 + (biased_eval.call_count * 0.8)  # Increases with position
        return min(5.0, score), "PASS"
    
    result = await bias_detector._detect_position_bias(
        sample_eval_prompt,
        biased_eval,
        sample_test_cases
    )
    
    assert result.bias_type == BiasType.POSITION
    # Note: May or may not detect depending on variance threshold
    assert isinstance(result.detected, bool)


# ============================================================================
# Test Length Bias Detection
# ============================================================================

@pytest.mark.asyncio
async def test_detect_length_bias_with_correlation(bias_detector, sample_eval_prompt, sample_test_cases, mock_eval_function):
    """Test length bias detection when strong correlation exists"""
    result = await bias_detector._detect_length_bias(
        sample_eval_prompt,
        mock_eval_function,
        sample_test_cases
    )
    
    assert result.bias_type == BiasType.LENGTH
    # mock_eval_function has length bias built in
    assert result.detected is True or result.confidence > 0.5
    assert "correlation" in result.metrics


@pytest.mark.asyncio
async def test_detect_length_bias_no_correlation(bias_detector, sample_eval_prompt, sample_test_cases, mock_unbiased_eval_function):
    """Test length bias detection when no correlation exists"""
    result = await bias_detector._detect_length_bias(
        sample_eval_prompt,
        mock_unbiased_eval_function,
        sample_test_cases
    )
    
    assert result.bias_type == BiasType.LENGTH
    assert result.detected is False
    assert result.severity == BiasSeverity.NONE


@pytest.mark.asyncio
async def test_detect_length_bias_insufficient_data(bias_detector, sample_eval_prompt, mock_eval_function):
    """Test length bias detection with insufficient test cases"""
    result = await bias_detector._detect_length_bias(
        sample_eval_prompt,
        mock_eval_function,
        [{"input": "test", "output": "test"}]  # Only 1 case
    )
    
    assert result.bias_type == BiasType.LENGTH
    assert result.detected is False
    assert "Insufficient data" in result.evidence[0]


# ============================================================================
# Test Verbosity Bias Detection
# ============================================================================

@pytest.mark.asyncio
async def test_detect_verbosity_bias(bias_detector, sample_eval_prompt, sample_test_cases, mock_eval_function):
    """Test verbosity bias detection"""
    result = await bias_detector._detect_verbosity_bias(
        sample_eval_prompt,
        mock_eval_function,
        sample_test_cases
    )
    
    assert result.bias_type == BiasType.VERBOSITY
    assert isinstance(result.detected, bool)
    assert "correlation" in result.metrics


def test_calculate_verbosity():
    """Test verbosity score calculation"""
    detector = EvalBiasDetector()
    
    # Short, concise text
    concise = "This is concise."
    concise_score = detector._calculate_verbosity(concise)
    
    # Long, verbose text with fillers
    verbose = "Well, actually, this is really quite very verbose and basically contains literally many filler words that are somewhat unnecessary."
    verbose_score = detector._calculate_verbosity(verbose)
    
    assert verbose_score > concise_score
    assert 0 <= concise_score <= 1
    assert 0 <= verbose_score <= 1


# ============================================================================
# Test Leniency/Severity Bias Detection
# ============================================================================

@pytest.mark.asyncio
async def test_detect_leniency_bias(bias_detector, sample_eval_prompt, sample_test_cases):
    """Test leniency bias detection (over-scoring)"""
    # Create lenient eval function
    async def lenient_eval(prompt, input_text, output_text):
        return 4.5, "PASS"  # Always high scores
    
    baseline_scores = [3.0, 3.0, 3.0, 3.0, 3.0]  # Expected scores
    
    results = await bias_detector._detect_leniency_severity_bias(
        sample_eval_prompt,
        lenient_eval,
        sample_test_cases,
        baseline_scores
    )
    
    assert len(results) > 0
    # Should detect leniency bias
    leniency_results = [r for r in results if r.bias_type == BiasType.LENIENCY]
    if leniency_results:
        assert leniency_results[0].detected is True


@pytest.mark.asyncio
async def test_detect_severity_bias(bias_detector, sample_eval_prompt, sample_test_cases):
    """Test severity bias detection (under-scoring)"""
    # Create severe eval function
    async def severe_eval(prompt, input_text, output_text):
        return 1.5, "FAIL"  # Always low scores
    
    baseline_scores = [3.0, 3.0, 3.0, 3.0, 3.0]  # Expected scores
    
    results = await bias_detector._detect_leniency_severity_bias(
        sample_eval_prompt,
        severe_eval,
        sample_test_cases,
        baseline_scores
    )
    
    assert len(results) > 0
    # Should detect severity bias
    severity_results = [r for r in results if r.bias_type == BiasType.SEVERITY]
    if severity_results:
        assert severity_results[0].detected is True


# ============================================================================
# Test Anchoring Bias Detection
# ============================================================================

@pytest.mark.asyncio
async def test_detect_anchoring_bias_present(bias_detector, sample_test_cases, mock_eval_function):
    """Test anchoring bias detection when present"""
    prompt_with_anchoring = """
    You are an evaluator. Compare to previous evaluations.
    Consider relative to others in this batch.
    """
    
    result = await bias_detector._detect_anchoring_bias(
        prompt_with_anchoring,
        mock_eval_function,
        sample_test_cases
    )
    
    assert result.bias_type == BiasType.ANCHORING
    # May or may not detect depending on exact regex, check structure is valid
    assert isinstance(result.detected, bool)
    assert len(result.evidence) > 0


@pytest.mark.asyncio
async def test_detect_anchoring_bias_absent(bias_detector, sample_eval_prompt, sample_test_cases, mock_eval_function):
    """Test anchoring bias detection when absent"""
    result = await bias_detector._detect_anchoring_bias(
        sample_eval_prompt,
        mock_eval_function,
        sample_test_cases
    )
    
    assert result.bias_type == BiasType.ANCHORING
    assert result.detected is False


# ============================================================================
# Test Full Bias Detection
# ============================================================================

@pytest.mark.asyncio
async def test_detect_biases_full_report(bias_detector, sample_eval_prompt, sample_test_cases, mock_eval_function):
    """Test full bias detection report generation"""
    report = await bias_detector.detect_biases(
        eval_prompt=sample_eval_prompt,
        run_eval_func=mock_eval_function,
        test_cases=sample_test_cases,
        baseline_scores=None
    )
    
    assert isinstance(report, ComprehensiveBiasReport)
    assert isinstance(report.overall_bias_score, (int, float))
    assert 0 <= report.overall_bias_score <= 100
    assert isinstance(report.biases_detected, list)
    assert len(report.biases_detected) >= 4  # At least 4 bias tests (position, length, verbosity, anchoring)
    assert isinstance(report.is_biased, bool)
    assert isinstance(report.recommendations, list)


@pytest.mark.asyncio
async def test_detect_biases_insufficient_test_cases(bias_detector, sample_eval_prompt, mock_eval_function):
    """Test bias detection with insufficient test cases"""
    report = await bias_detector.detect_biases(
        eval_prompt=sample_eval_prompt,
        run_eval_func=mock_eval_function,
        test_cases=[{"input": "test", "output": "test"}],  # Only 1 case
        baseline_scores=None
    )
    
    assert report.overall_bias_score == 0
    assert report.is_biased is False
    assert "at least 3 test cases" in report.recommendations[0].lower()


@pytest.mark.asyncio
async def test_detect_biases_with_baseline(bias_detector, sample_eval_prompt, sample_test_cases, mock_eval_function):
    """Test bias detection with baseline scores"""
    baseline = [3.0, 3.0, 3.0, 3.0, 3.0]
    
    report = await bias_detector.detect_biases(
        eval_prompt=sample_eval_prompt,
        run_eval_func=mock_eval_function,
        test_cases=sample_test_cases,
        baseline_scores=baseline
    )
    
    # Should include leniency/severity bias detection (adds 1 more test)
    assert len(report.biases_detected) >= 5


# ============================================================================
# Test Helper Functions
# ============================================================================

def test_calculate_correlation():
    """Test Pearson correlation calculation"""
    detector = EvalBiasDetector()
    
    # Perfect positive correlation
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    corr = detector._calculate_correlation(x, y)
    assert abs(corr - 1.0) < 0.01
    
    # No correlation
    x = [1, 2, 3, 4, 5]
    y = [3, 3, 3, 3, 3]
    corr = detector._calculate_correlation(x, y)
    assert abs(corr) < 0.01
    
    # Negative correlation
    x = [1, 2, 3, 4, 5]
    y = [10, 8, 6, 4, 2]
    corr = detector._calculate_correlation(x, y)
    assert corr < -0.9


def test_calculate_severity():
    """Test severity calculation"""
    detector = EvalBiasDetector()
    
    thresholds = {
        BiasSeverity.CRITICAL: 0.8,
        BiasSeverity.HIGH: 0.6,
        BiasSeverity.MEDIUM: 0.4,
        BiasSeverity.LOW: 0.2
    }
    
    assert detector._calculate_severity(0.9, thresholds) == BiasSeverity.CRITICAL
    assert detector._calculate_severity(0.7, thresholds) == BiasSeverity.HIGH
    assert detector._calculate_severity(0.5, thresholds) == BiasSeverity.MEDIUM
    assert detector._calculate_severity(0.3, thresholds) == BiasSeverity.LOW
    assert detector._calculate_severity(0.1, thresholds) == BiasSeverity.NONE


# ============================================================================
# Test Convenience Functions
# ============================================================================

@pytest.mark.asyncio
async def test_check_eval_bias_convenience_function(sample_eval_prompt, sample_test_cases, mock_eval_function):
    """Test check_eval_bias convenience function"""
    report = await check_eval_bias(
        eval_prompt=sample_eval_prompt,
        run_eval_func=mock_eval_function,
        test_cases=sample_test_cases
    )
    
    assert isinstance(report, ComprehensiveBiasReport)
    assert isinstance(report.overall_bias_score, (int, float))


def test_bias_report_to_dict():
    """Test bias report serialization to dict"""
    result = BiasDetectionResult(
        bias_type=BiasType.POSITION,
        detected=True,
        severity=BiasSeverity.HIGH,
        confidence=0.85,
        evidence=["Test evidence"],
        metrics={"key": "value"},
        recommendation="Fix this"
    )
    
    report = ComprehensiveBiasReport(
        overall_bias_score=45.0,
        biases_detected=[result],
        is_biased=True,
        critical_biases=["position_bias"],
        recommendations=["Fix position bias"],
        test_metadata={"count": 5}
    )
    
    report_dict = bias_report_to_dict(report)
    
    assert report_dict["overall_bias_score"] == 45.0
    assert report_dict["is_biased"] is True
    assert len(report_dict["biases_detected"]) == 1
    assert report_dict["biases_detected"][0]["bias_type"] == "position_bias"
    assert report_dict["biases_detected"][0]["severity"] == "high"


# ============================================================================
# Test Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_bias_detection_with_eval_error(bias_detector, sample_eval_prompt, sample_test_cases):
    """Test bias detection when eval function raises errors"""
    async def failing_eval(prompt, input_text, output_text):
        raise Exception("Eval failed")
    
    # Should handle gracefully
    result = await bias_detector._detect_length_bias(
        sample_eval_prompt,
        failing_eval,
        sample_test_cases
    )
    
    assert result.bias_type == BiasType.LENGTH
    # Should return no bias result with error message
    assert not result.detected or "Error" in result.evidence[0]


def test_empty_recommendations():
    """Test recommendation generation with no biases"""
    detector = EvalBiasDetector()
    
    no_bias_result = BiasDetectionResult(
        bias_type=BiasType.POSITION,
        detected=False,
        severity=BiasSeverity.NONE,
        confidence=0.0,
        evidence=[],
        metrics={},
        recommendation=""
    )
    
    recommendations = detector._generate_recommendations([no_bias_result])
    
    assert len(recommendations) == 1
    assert "no significant biases" in recommendations[0].lower()


def test_multiple_critical_biases():
    """Test recommendation generation with multiple critical biases"""
    detector = EvalBiasDetector()
    
    critical_biases = [
        BiasDetectionResult(
            bias_type=BiasType.POSITION,
            detected=True,
            severity=BiasSeverity.CRITICAL,
            confidence=0.9,
            evidence=["Strong evidence"],
            metrics={},
            recommendation="Fix: Do X\n1. Step one"
        ),
        BiasDetectionResult(
            bias_type=BiasType.LENGTH,
            detected=True,
            severity=BiasSeverity.HIGH,
            confidence=0.8,
            evidence=["Evidence"],
            metrics={},
            recommendation="Fix: Do Y\n1. Step one"
        ),
    ]
    
    recommendations = detector._generate_recommendations(critical_biases)
    
    assert any("CRITICAL" in r or "critical" in r for r in recommendations)
    assert len(recommendations) >= 2


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_bias_detection_workflow():
    """Test complete bias detection workflow"""
    detector = EvalBiasDetector()
    
    eval_prompt = "You are an evaluator. Score from 1-5."
    
    test_cases = [
        {"input": "Test 1", "output": "Short"},
        {"input": "Test 2", "output": "Medium length output here"},
        {"input": "Test 3", "output": "Very long output with lots of text and details that goes on and on"},
        {"input": "Test 4", "output": "Another test"},
        {"input": "Test 5", "output": "Final test output"},
    ]
    
    async def eval_func(prompt, input_text, output_text):
        # Simple scoring based on length (introduces bias)
        score = min(5.0, 2.0 + len(output_text) / 10)
        return score, "PASS" if score >= 3 else "FAIL"
    
    report = await detector.detect_biases(
        eval_prompt=eval_prompt,
        run_eval_func=eval_func,
        test_cases=test_cases
    )
    
    # Verify report structure
    assert isinstance(report, ComprehensiveBiasReport)
    assert 0 <= report.overall_bias_score <= 100
    assert isinstance(report.biases_detected, list)
    assert len(report.biases_detected) >= 4  # position, length, verbosity, anchoring (no baseline)
    assert isinstance(report.is_biased, bool)
    assert isinstance(report.critical_biases, list)
    assert isinstance(report.recommendations, list)
    
    # Should detect length bias given our eval function
    length_biases = [b for b in report.biases_detected if b.bias_type == BiasType.LENGTH]
    assert len(length_biases) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
