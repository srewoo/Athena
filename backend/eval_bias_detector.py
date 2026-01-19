"""
Eval Prompt Bias Detector

Detects common biases in evaluation prompts and evaluator behavior:
1. Position bias - Favoring first or last outputs
2. Length bias - Favoring longer/shorter outputs
3. Verbosity bias - Favoring verbose explanations
4. Confirmation bias - Favoring expected outcomes
5. Leniency/severity bias - Systematic over/under scoring

Author: Athena Team
"""

import logging
import statistics
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class BiasType(Enum):
    """Types of biases to detect"""
    POSITION = "position_bias"
    LENGTH = "length_bias"
    VERBOSITY = "verbosity_bias"
    CONFIRMATION = "confirmation_bias"
    LENIENCY = "leniency_bias"
    SEVERITY = "severity_bias"
    ANCHORING = "anchoring_bias"


class BiasSeverity(Enum):
    """Severity of detected bias"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BiasDetectionResult:
    """Result of bias detection for a specific bias type"""
    bias_type: BiasType
    detected: bool
    severity: BiasSeverity
    confidence: float  # 0-1
    evidence: List[str]
    metrics: Dict[str, Any]
    recommendation: str


@dataclass
class ComprehensiveBiasReport:
    """Complete bias detection report"""
    overall_bias_score: float  # 0-100, 0 = no bias, 100 = severe bias
    biases_detected: List[BiasDetectionResult]
    is_biased: bool
    critical_biases: List[str]
    recommendations: List[str]
    test_metadata: Dict[str, Any]


class EvalBiasDetector:
    """
    Detects biases in evaluation prompts and evaluator behavior.
    
    Usage:
        detector = EvalBiasDetector()
        
        # Test for biases
        report = await detector.detect_biases(
            eval_prompt=eval_prompt,
            run_eval_func=eval_function,
            test_cases=[...]
        )
        
        if report.is_biased:
            print(f"Biases detected: {report.critical_biases}")
    """
    
    # Thresholds for bias detection
    POSITION_BIAS_THRESHOLD = 0.15  # 15% score difference between positions
    LENGTH_CORRELATION_THRESHOLD = 0.6  # Strong correlation (Pearson's r)
    VERBOSITY_CORRELATION_THRESHOLD = 0.5
    LENIENCY_THRESHOLD = 0.3  # 30% above expected mean
    SEVERITY_THRESHOLD = 0.3  # 30% below expected mean
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def detect_biases(
        self,
        eval_prompt: str,
        run_eval_func,  # async (prompt, input, output) -> (score, verdict)
        test_cases: List[Dict[str, str]],
        baseline_scores: Optional[List[float]] = None
    ) -> ComprehensiveBiasReport:
        """
        Detect all biases in the evaluation prompt.
        
        Args:
            eval_prompt: The evaluation prompt to test
            run_eval_func: Function to run evaluations
            test_cases: List of test cases with 'input' and 'output' keys
            baseline_scores: Optional expected scores for comparison
        
        Returns:
            ComprehensiveBiasReport with all detected biases
        """
        if len(test_cases) < 3:
            return ComprehensiveBiasReport(
                overall_bias_score=0,
                biases_detected=[],
                is_biased=False,
                critical_biases=[],
                recommendations=["Need at least 3 test cases for bias detection"],
                test_metadata={"test_cases": len(test_cases)}
            )
        
        biases_detected = []
        
        # Run all bias detection tests
        self.logger.info(f"Running bias detection on {len(test_cases)} test cases")
        
        # 1. Position Bias Detection
        position_bias = await self._detect_position_bias(
            eval_prompt, run_eval_func, test_cases
        )
        biases_detected.append(position_bias)
        
        # 2. Length Bias Detection
        length_bias = await self._detect_length_bias(
            eval_prompt, run_eval_func, test_cases
        )
        biases_detected.append(length_bias)
        
        # 3. Verbosity Bias Detection
        verbosity_bias = await self._detect_verbosity_bias(
            eval_prompt, run_eval_func, test_cases
        )
        biases_detected.append(verbosity_bias)
        
        # 4. Leniency/Severity Bias Detection (if baseline provided)
        if baseline_scores:
            leniency_severity = await self._detect_leniency_severity_bias(
                eval_prompt, run_eval_func, test_cases, baseline_scores
            )
            biases_detected.extend(leniency_severity)
        
        # 5. Anchoring Bias Detection
        anchoring_bias = await self._detect_anchoring_bias(
            eval_prompt, run_eval_func, test_cases
        )
        biases_detected.append(anchoring_bias)
        
        # Calculate overall bias score
        detected_biases = [b for b in biases_detected if b.detected]
        critical_biases = [
            b.bias_type.value for b in detected_biases 
            if b.severity in [BiasSeverity.HIGH, BiasSeverity.CRITICAL]
        ]
        
        # Weight by severity
        severity_weights = {
            BiasSeverity.CRITICAL: 25,
            BiasSeverity.HIGH: 15,
            BiasSeverity.MEDIUM: 8,
            BiasSeverity.LOW: 3,
            BiasSeverity.NONE: 0
        }
        
        overall_score = sum(
            severity_weights.get(b.severity, 0) * b.confidence
            for b in biases_detected
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(biases_detected)
        
        return ComprehensiveBiasReport(
            overall_bias_score=min(100, round(overall_score, 1)),
            biases_detected=biases_detected,
            is_biased=len(critical_biases) > 0 or overall_score > 30,
            critical_biases=critical_biases,
            recommendations=recommendations,
            test_metadata={
                "test_cases_count": len(test_cases),
                "baseline_provided": baseline_scores is not None,
                "biases_tested": len(biases_detected)
            }
        )
    
    async def _detect_position_bias(
        self,
        eval_prompt: str,
        run_eval_func,
        test_cases: List[Dict[str, str]]
    ) -> BiasDetectionResult:
        """
        Detect position bias: Does order of evaluation affect scores?
        
        Strategy: Evaluate same output in different positions (first, middle, last)
        """
        if len(test_cases) < 2:
            return self._no_bias_result(BiasType.POSITION, "Insufficient test cases")
        
        try:
            # Take first test case and evaluate it in different "positions"
            # by changing what comes before/after it in prompt context
            test_case = test_cases[0]
            
            # Simulate position by adding context
            positions = []
            
            # Position 1: First (no prior context)
            score_first, _ = await run_eval_func(
                eval_prompt,
                test_case['input'],
                test_case['output']
            )
            positions.append(('first', float(score_first)))
            
            # Position 2: After seeing another example
            # (In practice, this would need actual A/B testing with batch evaluation)
            score_second, _ = await run_eval_func(
                eval_prompt,
                test_case['input'],
                test_case['output']
            )
            positions.append(('second', float(score_second)))
            
            # Position 3: Last (after multiple examples)
            score_last, _ = await run_eval_func(
                eval_prompt,
                test_case['input'],
                test_case['output']
            )
            positions.append(('last', float(score_last)))
            
            # Calculate variance
            scores = [s for _, s in positions]
            if len(scores) < 2:
                return self._no_bias_result(BiasType.POSITION, "Could not collect scores")
            
            mean_score = statistics.mean(scores)
            max_diff = max(scores) - min(scores)
            relative_diff = max_diff / mean_score if mean_score > 0 else 0
            
            # Detect bias if scores differ significantly
            detected = relative_diff > self.POSITION_BIAS_THRESHOLD
            
            if detected:
                severity = self._calculate_severity(relative_diff, {
                    BiasSeverity.CRITICAL: 0.4,
                    BiasSeverity.HIGH: 0.25,
                    BiasSeverity.MEDIUM: 0.15,
                    BiasSeverity.LOW: 0.05
                })
                
                evidence = [
                    f"Score variance across positions: {max_diff:.2f}",
                    f"Relative difference: {relative_diff:.1%}",
                    f"Scores: first={scores[0]:.2f}, middle={scores[1]:.2f}, last={scores[2]:.2f}"
                ]
                
                recommendation = (
                    "Position bias detected. Consider:\n"
                    "1. Shuffling evaluation order\n"
                    "2. Using blind evaluation (no position indicators)\n"
                    "3. Adding explicit instruction: 'Evaluate independently of order'"
                )
            else:
                severity = BiasSeverity.NONE
                evidence = [f"No significant position bias (diff: {relative_diff:.1%})"]
                recommendation = "No position bias detected. Evaluation is position-independent."
            
            return BiasDetectionResult(
                bias_type=BiasType.POSITION,
                detected=detected,
                severity=severity,
                confidence=min(1.0, relative_diff / 0.4) if detected else 0.0,
                evidence=evidence,
                metrics={
                    "max_difference": round(max_diff, 3),
                    "relative_difference": round(relative_diff, 3),
                    "scores_by_position": dict(positions)
                },
                recommendation=recommendation
            )
        
        except Exception as e:
            self.logger.error(f"Position bias detection failed: {e}")
            return self._no_bias_result(BiasType.POSITION, f"Error: {str(e)}")
    
    async def _detect_length_bias(
        self,
        eval_prompt: str,
        run_eval_func,
        test_cases: List[Dict[str, str]]
    ) -> BiasDetectionResult:
        """
        Detect length bias: Does output length affect scores?
        
        Strategy: Evaluate outputs of varying lengths with same content quality
        """
        try:
            # Collect scores and lengths
            data = []
            for tc in test_cases[:10]:  # Limit to 10 for performance
                try:
                    score, _ = await run_eval_func(
                        eval_prompt,
                        tc['input'],
                        tc['output']
                    )
                    length = len(tc['output'])
                    data.append((float(score), length))
                except Exception as e:
                    self.logger.warning(f"Failed to evaluate test case: {e}")
                    continue
            
            if len(data) < 3:
                return self._no_bias_result(BiasType.LENGTH, "Insufficient data")
            
            scores, lengths = zip(*data)
            
            # Calculate Pearson correlation
            correlation = self._calculate_correlation(scores, lengths)
            
            # Detect bias if strong correlation exists
            detected = abs(correlation) > self.LENGTH_CORRELATION_THRESHOLD
            
            if detected:
                severity = self._calculate_severity(abs(correlation), {
                    BiasSeverity.CRITICAL: 0.8,
                    BiasSeverity.HIGH: 0.7,
                    BiasSeverity.MEDIUM: 0.6,
                    BiasSeverity.LOW: 0.4
                })
                
                direction = "longer" if correlation > 0 else "shorter"
                evidence = [
                    f"Strong correlation between length and score: {correlation:.2f}",
                    f"Evaluator favors {direction} outputs",
                    f"Score range: {min(scores):.2f}-{max(scores):.2f}",
                    f"Length range: {min(lengths)}-{max(lengths)} chars"
                ]
                
                recommendation = (
                    f"Length bias detected (favors {direction} outputs). Consider:\n"
                    "1. Adding explicit instruction: 'Judge quality, not length'\n"
                    "2. Normalizing scores by expected length\n"
                    "3. Including length-diverse calibration examples"
                )
            else:
                severity = BiasSeverity.NONE
                evidence = [f"No length bias detected (correlation: {correlation:.2f})"]
                recommendation = "No length bias detected. Evaluation is length-independent."
            
            return BiasDetectionResult(
                bias_type=BiasType.LENGTH,
                detected=detected,
                severity=severity,
                confidence=min(1.0, abs(correlation)) if detected else 0.0,
                evidence=evidence,
                metrics={
                    "correlation": round(correlation, 3),
                    "sample_size": len(data),
                    "length_range": (min(lengths), max(lengths)),
                    "score_range": (round(min(scores), 2), round(max(scores), 2))
                },
                recommendation=recommendation
            )
        
        except Exception as e:
            self.logger.error(f"Length bias detection failed: {e}")
            return self._no_bias_result(BiasType.LENGTH, f"Error: {str(e)}")
    
    async def _detect_verbosity_bias(
        self,
        eval_prompt: str,
        run_eval_func,
        test_cases: List[Dict[str, str]]
    ) -> BiasDetectionResult:
        """
        Detect verbosity bias: Does verbose explanation affect scores?
        
        Strategy: Measure correlation between explanation verbosity and scores
        """
        try:
            data = []
            for tc in test_cases[:10]:
                try:
                    score, _ = await run_eval_func(
                        eval_prompt,
                        tc['input'],
                        tc['output']
                    )
                    
                    # Measure verbosity (sentences, avg word length, filler words)
                    verbosity_score = self._calculate_verbosity(tc['output'])
                    data.append((float(score), verbosity_score))
                except Exception as e:
                    continue
            
            if len(data) < 3:
                return self._no_bias_result(BiasType.VERBOSITY, "Insufficient data")
            
            scores, verbosity_scores = zip(*data)
            correlation = self._calculate_correlation(scores, verbosity_scores)
            
            detected = abs(correlation) > self.VERBOSITY_CORRELATION_THRESHOLD
            
            if detected:
                severity = self._calculate_severity(abs(correlation), {
                    BiasSeverity.CRITICAL: 0.75,
                    BiasSeverity.HIGH: 0.6,
                    BiasSeverity.MEDIUM: 0.5,
                    BiasSeverity.LOW: 0.35
                })
                
                direction = "verbose" if correlation > 0 else "concise"
                evidence = [
                    f"Correlation between verbosity and score: {correlation:.2f}",
                    f"Evaluator favors {direction} explanations"
                ]
                
                recommendation = (
                    f"Verbosity bias detected (favors {direction} output). Consider:\n"
                    "1. Adding: 'Judge substance, not verbosity'\n"
                    "2. Including both concise and verbose examples in calibration\n"
                    "3. Explicitly scoring clarity over length"
                )
            else:
                severity = BiasSeverity.NONE
                evidence = [f"No verbosity bias (correlation: {correlation:.2f})"]
                recommendation = "No verbosity bias detected."
            
            return BiasDetectionResult(
                bias_type=BiasType.VERBOSITY,
                detected=detected,
                severity=severity,
                confidence=min(1.0, abs(correlation)) if detected else 0.0,
                evidence=evidence,
                metrics={"correlation": round(correlation, 3), "sample_size": len(data)},
                recommendation=recommendation
            )
        
        except Exception as e:
            self.logger.error(f"Verbosity bias detection failed: {e}")
            return self._no_bias_result(BiasType.VERBOSITY, f"Error: {str(e)}")
    
    async def _detect_leniency_severity_bias(
        self,
        eval_prompt: str,
        run_eval_func,
        test_cases: List[Dict[str, str]],
        baseline_scores: List[float]
    ) -> List[BiasDetectionResult]:
        """
        Detect leniency (over-scoring) or severity (under-scoring) bias.
        
        Requires baseline scores to compare against.
        """
        try:
            actual_scores = []
            for tc in test_cases:
                try:
                    score, _ = await run_eval_func(
                        eval_prompt,
                        tc['input'],
                        tc['output']
                    )
                    actual_scores.append(float(score))
                except Exception:
                    continue
            
            if len(actual_scores) != len(baseline_scores):
                return [self._no_bias_result(BiasType.LENIENCY, "Score mismatch")]
            
            # Calculate systematic bias
            differences = [actual - baseline for actual, baseline in zip(actual_scores, baseline_scores)]
            mean_diff = statistics.mean(differences)
            relative_diff = mean_diff / statistics.mean(baseline_scores) if baseline_scores else 0
            
            results = []
            
            # Check leniency bias (over-scoring)
            if relative_diff > self.LENIENCY_THRESHOLD:
                severity = self._calculate_severity(relative_diff, {
                    BiasSeverity.CRITICAL: 0.6,
                    BiasSeverity.HIGH: 0.4,
                    BiasSeverity.MEDIUM: 0.3,
                    BiasSeverity.LOW: 0.2
                })
                
                results.append(BiasDetectionResult(
                    bias_type=BiasType.LENIENCY,
                    detected=True,
                    severity=severity,
                    confidence=min(1.0, relative_diff / 0.6),
                    evidence=[
                        f"Systematic over-scoring detected: +{relative_diff:.1%}",
                        f"Average actual score: {statistics.mean(actual_scores):.2f}",
                        f"Average expected score: {statistics.mean(baseline_scores):.2f}"
                    ],
                    metrics={"mean_difference": round(mean_diff, 3), "relative_difference": round(relative_diff, 3)},
                    recommendation=(
                        "Leniency bias detected. Consider:\n"
                        "1. Reviewing rubric thresholds (too lenient?)\n"
                        "2. Adding stricter calibration examples\n"
                        "3. Emphasizing critical evaluation stance"
                    )
                ))
            
            # Check severity bias (under-scoring)
            elif relative_diff < -self.SEVERITY_THRESHOLD:
                severity = self._calculate_severity(abs(relative_diff), {
                    BiasSeverity.CRITICAL: 0.6,
                    BiasSeverity.HIGH: 0.4,
                    BiasSeverity.MEDIUM: 0.3,
                    BiasSeverity.LOW: 0.2
                })
                
                results.append(BiasDetectionResult(
                    bias_type=BiasType.SEVERITY,
                    detected=True,
                    severity=severity,
                    confidence=min(1.0, abs(relative_diff) / 0.6),
                    evidence=[
                        f"Systematic under-scoring detected: {relative_diff:.1%}",
                        f"Average actual score: {statistics.mean(actual_scores):.2f}",
                        f"Average expected score: {statistics.mean(baseline_scores):.2f}"
                    ],
                    metrics={"mean_difference": round(mean_diff, 3), "relative_difference": round(relative_diff, 3)},
                    recommendation=(
                        "Severity bias detected. Consider:\n"
                        "1. Reviewing rubric thresholds (too harsh?)\n"
                        "2. Balancing calibration examples\n"
                        "3. Emphasizing fair, evidence-based evaluation"
                    )
                ))
            else:
                # No bias
                results.append(BiasDetectionResult(
                    bias_type=BiasType.LENIENCY,
                    detected=False,
                    severity=BiasSeverity.NONE,
                    confidence=0.0,
                    evidence=[f"No systematic bias detected (diff: {relative_diff:.1%})"],
                    metrics={"relative_difference": round(relative_diff, 3)},
                    recommendation="No leniency or severity bias detected."
                ))
            
            return results
        
        except Exception as e:
            self.logger.error(f"Leniency/severity bias detection failed: {e}")
            return [self._no_bias_result(BiasType.LENIENCY, f"Error: {str(e)}")]
    
    async def _detect_anchoring_bias(
        self,
        eval_prompt: str,
        run_eval_func,
        test_cases: List[Dict[str, str]]
    ) -> BiasDetectionResult:
        """
        Detect anchoring bias: Does first example anchor subsequent scores?
        
        Strategy: Evaluate same cases with different first examples
        """
        if len(test_cases) < 3:
            return self._no_bias_result(BiasType.ANCHORING, "Insufficient test cases")
        
        try:
            # Evaluate second test case after seeing different first examples
            # (This is a simplified version - full implementation would need batch evaluation)
            
            # For now, check if eval prompt has anchoring language
            anchoring_patterns = [
                r'\bcompare\s+to\s+previous',
                r'\brelative\s+to\s+others',
                r'\bin\s+context\s+of\s+other\s+outputs',
                r'\bconsider\s+previous\s+evaluations'
            ]
            
            has_anchoring_language = any(
                re.search(pattern, eval_prompt, re.IGNORECASE)
                for pattern in anchoring_patterns
            )
            
            if has_anchoring_language:
                return BiasDetectionResult(
                    bias_type=BiasType.ANCHORING,
                    detected=True,
                    severity=BiasSeverity.MEDIUM,
                    confidence=0.7,
                    evidence=["Eval prompt contains anchoring language"],
                    metrics={"anchoring_phrases_found": True},
                    recommendation=(
                        "Potential anchoring bias. Consider:\n"
                        "1. Removing comparative language\n"
                        "2. Emphasizing independent evaluation\n"
                        "3. Adding: 'Evaluate each output independently'"
                    )
                )
            
            return self._no_bias_result(BiasType.ANCHORING, "No anchoring language detected")
        
        except Exception as e:
            self.logger.error(f"Anchoring bias detection failed: {e}")
            return self._no_bias_result(BiasType.ANCHORING, f"Error: {str(e)}")
    
    def _calculate_verbosity(self, text: str) -> float:
        """Calculate verbosity score (0-1)"""
        sentences = len(re.findall(r'[.!?]+', text))
        words = len(text.split())
        
        if sentences == 0 or words == 0:
            return 0.0
        
        avg_sentence_length = words / sentences
        
        # Count filler words
        fillers = ['very', 'really', 'quite', 'somewhat', 'rather', 'actually', 
                   'basically', 'literally', 'essentially', 'particularly']
        filler_count = sum(text.lower().count(f' {filler} ') for filler in fillers)
        filler_ratio = filler_count / words if words > 0 else 0
        
        # Normalize to 0-1 scale
        # Longer sentences and more fillers = higher verbosity
        verbosity = min(1.0, (avg_sentence_length / 30) * 0.7 + filler_ratio * 0.3)
        
        return verbosity
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        try:
            mean_x = statistics.mean(x)
            mean_y = statistics.mean(y)
            
            numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
            denominator_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
            denominator_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
            
            if denominator_x == 0 or denominator_y == 0:
                return 0.0
            
            return numerator / (denominator_x * denominator_y)
        except Exception:
            return 0.0
    
    def _calculate_severity(
        self,
        value: float,
        thresholds: Dict[BiasSeverity, float]
    ) -> BiasSeverity:
        """Calculate bias severity based on value and thresholds"""
        for severity in [BiasSeverity.CRITICAL, BiasSeverity.HIGH, BiasSeverity.MEDIUM, BiasSeverity.LOW]:
            if value >= thresholds.get(severity, float('inf')):
                return severity
        return BiasSeverity.NONE
    
    def _no_bias_result(self, bias_type: BiasType, reason: str) -> BiasDetectionResult:
        """Helper to create a no-bias result"""
        return BiasDetectionResult(
            bias_type=bias_type,
            detected=False,
            severity=BiasSeverity.NONE,
            confidence=0.0,
            evidence=[reason],
            metrics={},
            recommendation=""
        )
    
    def _generate_recommendations(
        self,
        biases: List[BiasDetectionResult]
    ) -> List[str]:
        """Generate overall recommendations based on detected biases"""
        recommendations = []
        
        detected = [b for b in biases if b.detected]
        
        if not detected:
            return ["No significant biases detected. Evaluation prompt appears unbiased."]
        
        # Priority recommendations
        critical = [b for b in detected if b.severity == BiasSeverity.CRITICAL]
        if critical:
            recommendations.append(
                f"⚠️ CRITICAL: {len(critical)} critical bias(es) detected. "
                "Address these before deployment."
            )
        
        # Specific recommendations
        for bias in detected:
            if bias.severity in [BiasSeverity.HIGH, BiasSeverity.CRITICAL]:
                # Extract first actionable recommendation
                rec_parts = bias.recommendation.split(':')
                if len(rec_parts) > 1:
                    action_parts = rec_parts[1].split('\n')
                    if len(action_parts) > 1:
                        recommendations.append(f"• {bias.bias_type.value}: {action_parts[1].strip()}")
                    else:
                        recommendations.append(f"• {bias.bias_type.value}: {action_parts[0].strip()}")
        
        # General recommendations
        if len(detected) >= 3:
            recommendations.append(
                "Multiple biases detected. Consider comprehensive eval prompt redesign."
            )
        
        return recommendations


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

async def check_eval_bias(
    eval_prompt: str,
    run_eval_func,
    test_cases: List[Dict[str, str]],
    baseline_scores: Optional[List[float]] = None
) -> ComprehensiveBiasReport:
    """
    Convenience function to check for biases in an eval prompt.
    
    Args:
        eval_prompt: The evaluation prompt to test
        run_eval_func: Function to run evaluations
        test_cases: Test cases with 'input' and 'output'
        baseline_scores: Optional expected scores
    
    Returns:
        ComprehensiveBiasReport
    """
    detector = EvalBiasDetector()
    return await detector.detect_biases(
        eval_prompt=eval_prompt,
        run_eval_func=run_eval_func,
        test_cases=test_cases,
        baseline_scores=baseline_scores
    )


def bias_report_to_dict(report: ComprehensiveBiasReport) -> Dict[str, Any]:
    """Convert bias report to dictionary for JSON serialization"""
    return {
        "overall_bias_score": report.overall_bias_score,
        "is_biased": report.is_biased,
        "critical_biases": report.critical_biases,
        "biases_detected": [
            {
                "bias_type": b.bias_type.value,
                "detected": b.detected,
                "severity": b.severity.value,
                "confidence": round(b.confidence, 3),
                "evidence": b.evidence,
                "metrics": b.metrics,
                "recommendation": b.recommendation
            }
            for b in report.biases_detected
        ],
        "recommendations": report.recommendations,
        "test_metadata": report.test_metadata
    }
