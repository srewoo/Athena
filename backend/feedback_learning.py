"""
Feedback Learning Module for Athena

This module implements a feedback-driven learning loop that:
1. Stores feedback from eval runs (human validations, score corrections)
2. Learns patterns from feedback to improve future eval prompt generation
3. Adjusts dimension weights based on historical accuracy
4. Suggests improvements based on common failure patterns

This is NOT fine-tuning (no model weights are changed), but rather:
- Prompt adaptation based on accumulated feedback
- Dynamic weight adjustment for dimensions
- Pattern-based suggestions for eval prompt improvement
"""

import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEntry:
    """A single feedback entry from an eval run"""
    id: str
    project_id: str
    eval_prompt_version: int
    test_case_id: str

    # The evaluation that was given
    llm_score: float
    llm_verdict: str
    llm_reasoning: str

    # Human feedback (if provided)
    human_score: Optional[float] = None
    human_verdict: Optional[str] = None
    human_feedback: Optional[str] = None

    # Computed fields
    score_difference: Optional[float] = None  # human - llm
    agreement: Optional[bool] = None  # Did human agree?

    # Dimension-level feedback
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    dimension_feedback: Dict[str, str] = field(default_factory=dict)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    feedback_type: str = "human_validation"  # human_validation, auto_correction, pattern_match

    def compute_agreement(self):
        """Compute agreement metrics after human feedback is provided"""
        if self.human_score is not None:
            self.score_difference = self.human_score - self.llm_score
            # Agreement if within 0.5 points and same verdict category
            self.agreement = (
                abs(self.score_difference) <= 0.5 and
                self._verdict_category(self.human_verdict) == self._verdict_category(self.llm_verdict)
            )

    def _verdict_category(self, verdict: Optional[str]) -> str:
        if not verdict:
            return "unknown"
        verdict = verdict.upper()
        if verdict in ["PASS", "PASSED"]:
            return "pass"
        elif verdict in ["FAIL", "FAILED"]:
            return "fail"
        else:
            return "review"


@dataclass
class LearningPattern:
    """A learned pattern from feedback"""
    id: str
    pattern_type: str  # "over_scoring", "under_scoring", "dimension_bias", "failure_mode_miss"
    description: str
    frequency: int  # How often this pattern appears
    confidence: float  # How confident we are in this pattern (0-1)

    # What triggers this pattern
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)

    # Suggested adjustment
    suggested_adjustment: str = ""
    weight_adjustment: float = 0.0  # e.g., -0.1 means reduce weight by 10%

    # Evidence
    example_cases: List[str] = field(default_factory=list)

    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class FeedbackStore:
    """Persistent storage for feedback data"""

    def __init__(self, storage_path: str = "feedback_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self._feedback_cache: Dict[str, List[FeedbackEntry]] = {}
        self._patterns_cache: Dict[str, List[LearningPattern]] = {}

    def _get_project_file(self, project_id: str) -> Path:
        return self.storage_path / f"feedback_{project_id}.json"

    def _get_patterns_file(self, project_id: str) -> Path:
        return self.storage_path / f"patterns_{project_id}.json"

    def add_feedback(self, entry: FeedbackEntry) -> None:
        """Add a feedback entry"""
        entry.compute_agreement()

        # Load existing
        entries = self.get_feedback(entry.project_id)
        entries.append(entry)

        # Save
        self._save_feedback(entry.project_id, entries)

        # Update cache
        self._feedback_cache[entry.project_id] = entries

        logger.info(f"Added feedback entry {entry.id} for project {entry.project_id}")

    def get_feedback(self, project_id: str) -> List[FeedbackEntry]:
        """Get all feedback for a project"""
        if project_id in self._feedback_cache:
            return self._feedback_cache[project_id]

        file_path = self._get_project_file(project_id)
        if not file_path.exists():
            return []

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                entries = [FeedbackEntry(**e) for e in data]
                self._feedback_cache[project_id] = entries
                return entries
        except Exception as e:
            logger.error(f"Error loading feedback: {e}")
            return []

    def _save_feedback(self, project_id: str, entries: List[FeedbackEntry]) -> None:
        """Save feedback entries"""
        file_path = self._get_project_file(project_id)
        with open(file_path, 'w') as f:
            json.dump([asdict(e) for e in entries], f, indent=2)

    def save_patterns(self, project_id: str, patterns: List[LearningPattern]) -> None:
        """Save learned patterns"""
        file_path = self._get_patterns_file(project_id)
        with open(file_path, 'w') as f:
            json.dump([asdict(p) for p in patterns], f, indent=2)
        self._patterns_cache[project_id] = patterns

    def get_patterns(self, project_id: str) -> List[LearningPattern]:
        """Get learned patterns for a project"""
        if project_id in self._patterns_cache:
            return self._patterns_cache[project_id]

        file_path = self._get_patterns_file(project_id)
        if not file_path.exists():
            return []

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                patterns = [LearningPattern(**p) for p in data]
                self._patterns_cache[project_id] = patterns
                return patterns
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            return []


class FeedbackAnalyzer:
    """Analyzes feedback to extract learning patterns"""

    def __init__(self, store: FeedbackStore):
        self.store = store

    def analyze_feedback(self, project_id: str, min_samples: int = 5) -> Dict[str, Any]:
        """Analyze all feedback for a project and extract patterns"""
        feedback = self.store.get_feedback(project_id)

        if len(feedback) < min_samples:
            return {
                "status": "insufficient_data",
                "samples": len(feedback),
                "min_required": min_samples,
                "patterns": []
            }

        # Filter to entries with human feedback
        validated = [f for f in feedback if f.human_score is not None]

        if len(validated) < min_samples:
            return {
                "status": "insufficient_validated_data",
                "total_samples": len(feedback),
                "validated_samples": len(validated),
                "min_required": min_samples,
                "patterns": []
            }

        patterns = []

        # Pattern 1: Systematic over/under scoring
        score_diffs = [f.score_difference for f in validated if f.score_difference is not None]
        if score_diffs:
            avg_diff = sum(score_diffs) / len(score_diffs)
            if abs(avg_diff) > 0.3:
                pattern_type = "over_scoring" if avg_diff < 0 else "under_scoring"
                patterns.append(LearningPattern(
                    id=f"bias_{pattern_type}_{project_id[:8]}",
                    pattern_type=pattern_type,
                    description=f"LLM judge systematically {'over' if avg_diff < 0 else 'under'}-scores by {abs(avg_diff):.2f} points on average",
                    frequency=len(score_diffs),
                    confidence=min(0.9, len(score_diffs) / 20),  # More samples = more confidence
                    trigger_conditions={"avg_score_difference": avg_diff},
                    suggested_adjustment=f"Adjust scoring threshold by {-avg_diff:.2f} points",
                    weight_adjustment=-avg_diff / 5,  # Subtle adjustment
                    example_cases=[f.id for f in validated[:3]]
                ))

        # Pattern 2: Dimension-specific biases
        dimension_diffs = defaultdict(list)
        for f in validated:
            if f.dimension_feedback:
                for dim, feedback_text in f.dimension_feedback.items():
                    if "too high" in feedback_text.lower():
                        dimension_diffs[dim].append(-1)
                    elif "too low" in feedback_text.lower():
                        dimension_diffs[dim].append(1)

        for dim, diffs in dimension_diffs.items():
            if len(diffs) >= 3:
                avg_dim_diff = sum(diffs) / len(diffs)
                if abs(avg_dim_diff) > 0.5:
                    patterns.append(LearningPattern(
                        id=f"dim_bias_{dim}_{project_id[:8]}",
                        pattern_type="dimension_bias",
                        description=f"Dimension '{dim}' consistently scores {'too high' if avg_dim_diff < 0 else 'too low'}",
                        frequency=len(diffs),
                        confidence=min(0.85, len(diffs) / 10),
                        trigger_conditions={"dimension": dim, "direction": "high" if avg_dim_diff < 0 else "low"},
                        suggested_adjustment=f"Adjust rubric for '{dim}' to be {'stricter' if avg_dim_diff < 0 else 'more lenient'}",
                        weight_adjustment=-avg_dim_diff * 0.1
                    ))

        # Pattern 3: Agreement rate analysis
        agreements = [f.agreement for f in validated if f.agreement is not None]
        if agreements:
            agreement_rate = sum(1 for a in agreements if a) / len(agreements)
            if agreement_rate < 0.7:
                patterns.append(LearningPattern(
                    id=f"low_agreement_{project_id[:8]}",
                    pattern_type="calibration_needed",
                    description=f"Low human-LLM agreement rate ({agreement_rate:.1%}). Eval prompt may need recalibration.",
                    frequency=len(agreements),
                    confidence=min(0.9, len(agreements) / 15),
                    trigger_conditions={"agreement_rate": agreement_rate},
                    suggested_adjustment="Consider regenerating eval prompt with more calibration examples"
                ))

        # Pattern 4: Common failure modes missed
        failure_keywords = defaultdict(int)
        for f in validated:
            if f.human_feedback and f.score_difference and f.score_difference < -1:
                # Human scored much lower - what did LLM miss?
                words = f.human_feedback.lower().split()
                for word in words:
                    if word in ['hallucination', 'fabricated', 'incorrect', 'wrong', 'missing', 'incomplete']:
                        failure_keywords[word] += 1

        for keyword, count in failure_keywords.items():
            if count >= 3:
                patterns.append(LearningPattern(
                    id=f"missed_{keyword}_{project_id[:8]}",
                    pattern_type="failure_mode_miss",
                    description=f"LLM judge frequently misses '{keyword}' issues ({count} occurrences)",
                    frequency=count,
                    confidence=min(0.8, count / 5),
                    trigger_conditions={"failure_keyword": keyword},
                    suggested_adjustment=f"Add explicit check for '{keyword}' in auto-fail conditions"
                ))

        # Save patterns
        self.store.save_patterns(project_id, patterns)

        return {
            "status": "analyzed",
            "total_samples": len(feedback),
            "validated_samples": len(validated),
            "agreement_rate": agreement_rate if agreements else None,
            "avg_score_difference": avg_diff if score_diffs else None,
            "patterns": [asdict(p) for p in patterns]
        }

    def get_improvement_suggestions(self, project_id: str) -> List[Dict[str, Any]]:
        """Get actionable improvement suggestions based on patterns"""
        patterns = self.store.get_patterns(project_id)

        suggestions = []
        for pattern in patterns:
            if pattern.confidence >= 0.6:  # Only high-confidence patterns
                suggestions.append({
                    "type": pattern.pattern_type,
                    "description": pattern.description,
                    "suggestion": pattern.suggested_adjustment,
                    "confidence": pattern.confidence,
                    "based_on_samples": pattern.frequency
                })

        # Sort by confidence
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)

        return suggestions


class AdaptiveEvalGenerator:
    """Uses feedback patterns to adapt eval prompt generation"""

    def __init__(self, store: FeedbackStore, analyzer: FeedbackAnalyzer):
        self.store = store
        self.analyzer = analyzer

    def get_dimension_weight_adjustments(self, project_id: str) -> Dict[str, float]:
        """Get weight adjustments for dimensions based on learned patterns"""
        patterns = self.store.get_patterns(project_id)

        adjustments = {}
        for pattern in patterns:
            if pattern.pattern_type == "dimension_bias" and pattern.confidence >= 0.6:
                dim = pattern.trigger_conditions.get("dimension", "")
                if dim:
                    adjustments[dim] = pattern.weight_adjustment

        return adjustments

    def get_scoring_bias_adjustment(self, project_id: str) -> float:
        """Get overall scoring bias adjustment"""
        patterns = self.store.get_patterns(project_id)

        for pattern in patterns:
            if pattern.pattern_type in ["over_scoring", "under_scoring"] and pattern.confidence >= 0.7:
                return -pattern.trigger_conditions.get("avg_score_difference", 0)

        return 0.0

    def get_additional_auto_fails(self, project_id: str) -> List[Dict[str, str]]:
        """Get additional auto-fail conditions based on missed failure modes"""
        patterns = self.store.get_patterns(project_id)

        additional = []
        for pattern in patterns:
            if pattern.pattern_type == "failure_mode_miss" and pattern.confidence >= 0.6:
                keyword = pattern.trigger_conditions.get("failure_keyword", "")
                if keyword:
                    additional.append({
                        "id": f"learned_{keyword}",
                        "name": f"Learned: {keyword.title()} Detection",
                        "description": f"Check for {keyword} issues (learned from feedback)",
                        "detection": f"Explicitly verify no {keyword} present",
                        "category": "learned"
                    })

        return additional

    def should_regenerate_eval(self, project_id: str) -> Tuple[bool, str]:
        """Determine if eval prompt should be regenerated based on patterns"""
        patterns = self.store.get_patterns(project_id)

        for pattern in patterns:
            if pattern.pattern_type == "calibration_needed" and pattern.confidence >= 0.8:
                return True, "Low agreement rate detected. Recommend regenerating eval prompt."

        # Check if too many significant patterns
        significant_patterns = [p for p in patterns if p.confidence >= 0.7]
        if len(significant_patterns) >= 3:
            return True, f"Multiple significant patterns detected ({len(significant_patterns)}). Consider regenerating."

        return False, "Eval prompt appears well-calibrated."

    def adapt_eval_prompt(
        self,
        project_id: str,
        base_eval_prompt: str,
        dimensions: List[Dict[str, Any]],
        auto_fails: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Adapt an eval prompt based on learned feedback patterns.

        Returns:
            - Adapted eval prompt text
            - Adjusted dimensions with new weights
            - Extended auto-fails list
        """
        # Get adjustments
        weight_adjustments = self.get_dimension_weight_adjustments(project_id)
        scoring_bias = self.get_scoring_bias_adjustment(project_id)
        additional_auto_fails = self.get_additional_auto_fails(project_id)

        # Adjust dimension weights
        adjusted_dimensions = []
        total_weight = 0
        for dim in dimensions:
            adjusted = dim.copy()
            dim_name = dim.get("name", "").lower().replace(" ", "_")

            # Apply adjustment if exists
            if dim_name in weight_adjustments:
                adjustment = weight_adjustments[dim_name]
                adjusted["weight"] = max(0.05, dim["weight"] + adjustment)
                adjusted["weight_adjusted"] = True
                adjusted["adjustment_reason"] = f"Adjusted by {adjustment:+.2f} based on feedback"

            adjusted_dimensions.append(adjusted)
            total_weight += adjusted["weight"]

        # Renormalize weights
        for dim in adjusted_dimensions:
            dim["weight"] = round(dim["weight"] / total_weight, 2)

        # Extend auto-fails
        extended_auto_fails = auto_fails + additional_auto_fails

        # Adjust eval prompt text if scoring bias detected
        adapted_prompt = base_eval_prompt
        if abs(scoring_bias) > 0.2:
            bias_note = f"""
---
## CALIBRATION NOTE (Learned from Feedback)

Historical analysis shows a scoring {'under' if scoring_bias > 0 else 'over'}-estimation tendency.
Adjust your scoring threshold by approximately {abs(scoring_bias):.1f} points in the {'positive' if scoring_bias > 0 else 'negative'} direction.

---
"""
            # Insert after the header section
            if "## I." in adapted_prompt:
                adapted_prompt = adapted_prompt.replace("## I.", bias_note + "## I.", 1)

        # Add learned failure mode checks
        if additional_auto_fails:
            failures_text = "\n".join([
                f"**{af['id']}. {af['name']}** (Learned from feedback)\n   - {af['description']}"
                for af in additional_auto_fails
            ])

            if "Auto-Fail Conditions" in adapted_prompt:
                adapted_prompt = adapted_prompt.replace(
                    "Auto-Fail Conditions",
                    f"Auto-Fail Conditions\n\n### Learned Failure Modes:\n{failures_text}\n\n### Standard"
                )

        return adapted_prompt, adjusted_dimensions, extended_auto_fails


# Convenience functions for API integration
_store: Optional[FeedbackStore] = None
_analyzer: Optional[FeedbackAnalyzer] = None
_adaptive: Optional[AdaptiveEvalGenerator] = None


def get_feedback_system(storage_path: str = "feedback_data") -> Tuple[FeedbackStore, FeedbackAnalyzer, AdaptiveEvalGenerator]:
    """Get or create the feedback system components"""
    global _store, _analyzer, _adaptive

    if _store is None:
        _store = FeedbackStore(storage_path)
        _analyzer = FeedbackAnalyzer(_store)
        _adaptive = AdaptiveEvalGenerator(_store, _analyzer)

    return _store, _analyzer, _adaptive


def record_human_feedback(
    project_id: str,
    eval_prompt_version: int,
    test_case_id: str,
    llm_score: float,
    llm_verdict: str,
    llm_reasoning: str,
    human_score: float,
    human_verdict: str,
    human_feedback: str,
    dimension_scores: Optional[Dict[str, float]] = None,
    dimension_feedback: Optional[Dict[str, str]] = None
) -> str:
    """Record human feedback for an eval result"""
    store, _, _ = get_feedback_system()

    entry_id = hashlib.md5(
        f"{project_id}_{test_case_id}_{datetime.utcnow().isoformat()}".encode()
    ).hexdigest()[:12]

    entry = FeedbackEntry(
        id=entry_id,
        project_id=project_id,
        eval_prompt_version=eval_prompt_version,
        test_case_id=test_case_id,
        llm_score=llm_score,
        llm_verdict=llm_verdict,
        llm_reasoning=llm_reasoning,
        human_score=human_score,
        human_verdict=human_verdict,
        human_feedback=human_feedback,
        dimension_scores=dimension_scores or {},
        dimension_feedback=dimension_feedback or {}
    )

    store.add_feedback(entry)
    return entry_id


def analyze_project_feedback(project_id: str) -> Dict[str, Any]:
    """Analyze feedback for a project and return patterns"""
    _, analyzer, _ = get_feedback_system()
    return analyzer.analyze_feedback(project_id)


def get_eval_improvements(project_id: str) -> List[Dict[str, Any]]:
    """Get improvement suggestions for a project's eval prompt"""
    _, analyzer, _ = get_feedback_system()
    return analyzer.get_improvement_suggestions(project_id)


def adapt_eval_for_project(
    project_id: str,
    base_eval_prompt: str,
    dimensions: List[Dict[str, Any]],
    auto_fails: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Adapt an eval prompt based on project feedback"""
    _, _, adaptive = get_feedback_system()
    return adaptive.adapt_eval_prompt(project_id, base_eval_prompt, dimensions, auto_fails)
