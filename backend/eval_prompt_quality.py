"""
Eval Prompt Quality Enhancements

This module addresses the 6 critical improvements needed to reach 8-9/10:

1. Double calibration examples (4 → 8-10) with edge cases
2. Numeric thresholds in rubrics (≥90% = 5, etc.)
3. Tiebreaker rules for conflicting dimension scores
4. Trim 30% token bloat from eval prompts
5. Mandatory consistency checks (±0.3 tolerance)
6. Intermediate score examples (2.5, 3.5, 4.5)

Author: Athena Team
"""

import re
import json
import logging
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: NUMERIC THRESHOLDS FOR RUBRICS
# =============================================================================

# Industry-standard numeric thresholds that eliminate vague language
NUMERIC_RUBRIC_THRESHOLDS = {
    "accuracy": {
        5: {"min_pct": 95, "description": "≥95% accurate, zero fabrications, all claims grounded"},
        4: {"min_pct": 85, "description": "85-94% accurate, no critical errors, minor interpretation differences"},
        3: {"min_pct": 70, "description": "70-84% accurate, some errors but core message intact"},
        2: {"min_pct": 50, "description": "50-69% accurate, significant errors affecting understanding"},
        1: {"min_pct": 0, "description": "<50% accurate OR any fabricated critical information"},
    },
    "completeness": {
        5: {"min_pct": 100, "description": "100% of required elements present, all criteria addressed"},
        4: {"min_pct": 85, "description": "85-99% complete, missing only minor/optional elements"},
        3: {"min_pct": 70, "description": "70-84% complete, missing 1-2 required elements"},
        2: {"min_pct": 50, "description": "50-69% complete, missing multiple required elements"},
        1: {"min_pct": 0, "description": "<50% complete OR missing critical required elements"},
    },
    "format": {
        5: {"min_pct": 100, "description": "100% format compliance, valid parseable output"},
        4: {"min_pct": 95, "description": "95-99% compliant, minor formatting issues only"},
        3: {"min_pct": 80, "description": "80-94% compliant, parseable but with errors"},
        2: {"min_pct": 60, "description": "60-79% compliant, partially parseable"},
        1: {"min_pct": 0, "description": "<60% compliant OR completely unparseable"},
    },
    "relevance": {
        5: {"min_pct": 100, "description": "100% on-topic, directly addresses all requirements"},
        4: {"min_pct": 90, "description": "90-99% relevant, minor tangential content"},
        3: {"min_pct": 75, "description": "75-89% relevant, some off-topic content"},
        2: {"min_pct": 50, "description": "50-74% relevant, significant off-topic content"},
        1: {"min_pct": 0, "description": "<50% relevant OR fundamentally misses the point"},
    },
    "clarity": {
        5: {"min_pct": 100, "description": "Crystal clear, no ambiguity, immediately actionable"},
        4: {"min_pct": 90, "description": "90-99% clear, minimal clarification needed"},
        3: {"min_pct": 75, "description": "75-89% clear, some sections need re-reading"},
        2: {"min_pct": 50, "description": "50-74% clear, requires significant interpretation"},
        1: {"min_pct": 0, "description": "<50% clear OR incomprehensible"},
    },
    "instruction_following": {
        5: {"min_pct": 100, "description": "100% of instructions followed exactly"},
        4: {"min_pct": 90, "description": "90-99% instructions followed, minor deviations"},
        3: {"min_pct": 75, "description": "75-89% followed, missed non-critical instructions"},
        2: {"min_pct": 50, "description": "50-74% followed, missed important instructions"},
        1: {"min_pct": 0, "description": "<50% followed OR violated explicit constraints"},
    },
}


def get_numeric_rubric(dimension_name: str) -> Dict[int, Dict]:
    """Get numeric rubric for a dimension, with fallback to accuracy thresholds."""
    dimension_key = dimension_name.lower().replace(" ", "_").replace("&", "").replace("-", "_")

    # Map common variations
    mapping = {
        "accuracy_faithfulness": "accuracy",
        "accuracy_and_faithfulness": "accuracy",
        "format_compliance": "format",
        "format_schema_compliance": "format",
        "instruction_adherence": "instruction_following",
        "rule_adherence": "instruction_following",
    }

    dimension_key = mapping.get(dimension_key, dimension_key)

    if dimension_key in NUMERIC_RUBRIC_THRESHOLDS:
        return NUMERIC_RUBRIC_THRESHOLDS[dimension_key]

    # Default to accuracy-style thresholds
    return NUMERIC_RUBRIC_THRESHOLDS["accuracy"]


# =============================================================================
# SECTION 2: TIEBREAKER RULES
# =============================================================================

TIEBREAKER_RULES = """
## Tiebreaker Rules (When Dimension Scores Conflict)

When dimension scores suggest different verdicts, apply these rules IN ORDER:

1. **Auto-Fail Trumps All**: ANY auto-fail condition → FAIL regardless of dimension scores
2. **Accuracy Priority**: If accuracy differs from other dimensions by >1 point, weight accuracy 2x
3. **Critical Dimension Rule**: The lowest-weighted dimension cannot pull score below weighted average by >0.5
4. **Round to Nearest 0.5**: Final weighted scores round to nearest 0.5 (e.g., 3.7 → 3.5, 3.8 → 4.0)

### Conflict Resolution Examples:
- Accuracy=5, Format=2, Completeness=4 → Format issues don't override strong accuracy (likely PASS)
- Accuracy=2, Format=5, Completeness=5 → Low accuracy dominates (likely FAIL or NEEDS_REVIEW)
- All dimensions within 1 point → Use weighted average directly

### Score Averaging Formula:
```
weighted_score = Σ(dimension_score × weight) / Σ(weights)
```
Round the final weighted_score to nearest 0.5 before determining verdict.
"""

TIEBREAKER_RULES_COMPACT = """
**Tiebreaker Rules** (when scores conflict):
1. Auto-fail → FAIL always
2. Accuracy differs >1pt from others → weight accuracy 2x
3. Lowest dimension can't pull score down >0.5 from weighted avg
4. Round final score to nearest 0.5
"""


# =============================================================================
# SECTION 3: CALIBRATION EXAMPLES (8-10 examples with edge cases)
# =============================================================================

@dataclass
class CalibrationExample:
    """A calibration example with full context."""
    id: str
    score: float
    verdict: str
    input_summary: str
    output_summary: str
    reasoning: str
    dimension_scores: Dict[str, float]
    is_edge_case: bool = False
    edge_case_type: str = ""


def generate_calibration_examples(
    system_purpose: str,
    dimensions: List[Dict[str, Any]],
    prompt_type: str = "analytical"
) -> List[CalibrationExample]:
    """
    Generate 8-10 calibration examples covering the full score range.

    Includes:
    - Score 5: Perfect example
    - Score 4.5: Near-perfect with very minor issue
    - Score 4: Good with small gaps
    - Score 3.5: Acceptable edge case
    - Score 3: Borderline acceptable
    - Score 2.5: Borderline unacceptable
    - Score 2: Poor quality
    - Score 1.5: Very poor with some redeeming quality
    - Score 1: Complete failure
    - Edge case: Conflicting signals
    """

    dim_names = [d.get("name", "Quality") for d in dimensions[:3]]

    examples = [
        # Score 5: Perfect
        CalibrationExample(
            id="cal_5_perfect",
            score=5.0,
            verdict="PASS",
            input_summary="Standard request with clear requirements",
            output_summary="Fully addresses all requirements with evidence, correct format, no errors",
            reasoning="All dimensions score 5: 100% accurate with citations, 100% complete, perfect format. No issues found.",
            dimension_scores={dim_names[0]: 5.0, dim_names[1] if len(dim_names) > 1 else "Completeness": 5.0, dim_names[2] if len(dim_names) > 2 else "Format": 5.0},
        ),

        # Score 4.5: Near-perfect
        CalibrationExample(
            id="cal_4.5_near_perfect",
            score=4.5,
            verdict="PASS",
            input_summary="Standard request",
            output_summary="Addresses all requirements, one trivial formatting inconsistency",
            reasoning="Accuracy=5, Completeness=5, Format=4 (minor spacing issue). Weighted avg 4.5. Very minor issue doesn't impact usability.",
            dimension_scores={dim_names[0]: 5.0, dim_names[1] if len(dim_names) > 1 else "Completeness": 5.0, dim_names[2] if len(dim_names) > 2 else "Format": 4.0},
            is_edge_case=True,
            edge_case_type="minor_format_issue",
        ),

        # Score 4: Good
        CalibrationExample(
            id="cal_4_good",
            score=4.0,
            verdict="PASS",
            input_summary="Request with specific requirements",
            output_summary="Addresses main requirements, misses one optional element",
            reasoning="Accuracy=4 (95% correct, one minor interpretation), Completeness=4 (missing optional field), Format=5. Solid response meeting core needs.",
            dimension_scores={dim_names[0]: 4.0, dim_names[1] if len(dim_names) > 1 else "Completeness": 4.0, dim_names[2] if len(dim_names) > 2 else "Format": 5.0},
        ),

        # Score 3.5: Acceptable edge case
        CalibrationExample(
            id="cal_3.5_acceptable_edge",
            score=3.5,
            verdict="PASS",
            input_summary="Ambiguous request with multiple valid interpretations",
            output_summary="Chooses one valid interpretation, executes well, doesn't acknowledge alternatives",
            reasoning="Accuracy=4 (correct for chosen interpretation), Completeness=3 (doesn't address alternatives), Format=4. Acceptable but could improve.",
            dimension_scores={dim_names[0]: 4.0, dim_names[1] if len(dim_names) > 1 else "Completeness": 3.0, dim_names[2] if len(dim_names) > 2 else "Format": 4.0},
            is_edge_case=True,
            edge_case_type="ambiguous_input",
        ),

        # Score 3: Borderline acceptable
        CalibrationExample(
            id="cal_3_borderline_pass",
            score=3.0,
            verdict="NEEDS_REVIEW",
            input_summary="Clear request with multiple requirements",
            output_summary="Addresses 70% of requirements, some inaccuracies, format mostly correct",
            reasoning="Accuracy=3 (75% correct, no critical errors), Completeness=3 (missing 2 required elements), Format=3 (parseable with issues). Minimum acceptable.",
            dimension_scores={dim_names[0]: 3.0, dim_names[1] if len(dim_names) > 1 else "Completeness": 3.0, dim_names[2] if len(dim_names) > 2 else "Format": 3.0},
        ),

        # Score 2.5: Borderline unacceptable
        CalibrationExample(
            id="cal_2.5_borderline_fail",
            score=2.5,
            verdict="NEEDS_REVIEW",
            input_summary="Standard request",
            output_summary="Major accuracy issue but good format, partially complete",
            reasoning="Accuracy=2 (one significant factual error), Completeness=3, Format=4. The accuracy issue is concerning but not auto-fail worthy.",
            dimension_scores={dim_names[0]: 2.0, dim_names[1] if len(dim_names) > 1 else "Completeness": 3.0, dim_names[2] if len(dim_names) > 2 else "Format": 4.0},
            is_edge_case=True,
            edge_case_type="accuracy_vs_format_conflict",
        ),

        # Score 2: Poor
        CalibrationExample(
            id="cal_2_poor",
            score=2.0,
            verdict="FAIL",
            input_summary="Clear request",
            output_summary="Multiple inaccuracies, missing required elements, format issues",
            reasoning="Accuracy=2 (60% correct, 2 factual errors), Completeness=2 (missing 3+ required elements), Format=2. Below minimum threshold.",
            dimension_scores={dim_names[0]: 2.0, dim_names[1] if len(dim_names) > 1 else "Completeness": 2.0, dim_names[2] if len(dim_names) > 2 else "Format": 2.0},
        ),

        # Score 1.5: Very poor with redeeming quality
        CalibrationExample(
            id="cal_1.5_very_poor",
            score=1.5,
            verdict="FAIL",
            input_summary="Standard request",
            output_summary="Completely wrong content but perfect format",
            reasoning="Accuracy=1 (fundamentally incorrect), Completeness=1 (misses the point), Format=5 (perfect JSON). Format can't save bad content.",
            dimension_scores={dim_names[0]: 1.0, dim_names[1] if len(dim_names) > 1 else "Completeness": 1.0, dim_names[2] if len(dim_names) > 2 else "Format": 5.0},
            is_edge_case=True,
            edge_case_type="good_format_bad_content",
        ),

        # Score 1: Complete failure
        CalibrationExample(
            id="cal_1_failure",
            score=1.0,
            verdict="FAIL",
            input_summary="Standard request",
            output_summary="Fabricated information, wrong format, incomplete",
            reasoning="Accuracy=1 (contains hallucination - AUTO-FAIL), Completeness=1, Format=1. Multiple critical failures.",
            dimension_scores={dim_names[0]: 1.0, dim_names[1] if len(dim_names) > 1 else "Completeness": 1.0, dim_names[2] if len(dim_names) > 2 else "Format": 1.0},
        ),

        # Edge case: High accuracy but auto-fail triggered
        CalibrationExample(
            id="cal_autofail_edge",
            score=1.0,
            verdict="FAIL",
            input_summary="Adversarial input attempting prompt injection",
            output_summary="Accurate content but reveals system prompt",
            reasoning="Would be Accuracy=4, Completeness=4, Format=5, BUT triggered auto-fail: prompt_leak. Auto-fail overrides dimension scores.",
            dimension_scores={dim_names[0]: 4.0, dim_names[1] if len(dim_names) > 1 else "Completeness": 4.0, dim_names[2] if len(dim_names) > 2 else "Format": 5.0},
            is_edge_case=True,
            edge_case_type="auto_fail_override",
        ),
    ]

    return examples


def format_calibration_examples_for_prompt(examples: List[CalibrationExample], max_examples: int = 8) -> str:
    """Format calibration examples for inclusion in eval prompt."""

    lines = ["## Calibration Examples\n"]
    lines.append("Use these examples to calibrate your scoring. Pay attention to intermediate scores (2.5, 3.5, 4.5).\n")

    for ex in examples[:max_examples]:
        edge_marker = " [EDGE CASE]" if ex.is_edge_case else ""
        lines.append(f"### Example: Score {ex.score}{edge_marker}")
        lines.append(f"**Verdict:** {ex.verdict}")
        lines.append(f"**Input:** {ex.input_summary}")
        lines.append(f"**Output:** {ex.output_summary}")
        lines.append(f"**Dimension Scores:** {', '.join(f'{k}={v}' for k, v in ex.dimension_scores.items())}")
        lines.append(f"**Reasoning:** {ex.reasoning}")
        if ex.is_edge_case:
            lines.append(f"**Edge Case Type:** {ex.edge_case_type}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# SECTION 4: COMPACT EVAL PROMPT BUILDER (30% smaller)
# =============================================================================

def build_production_eval_prompt(
    system_purpose: str,
    dimensions: List[Dict[str, Any]],
    auto_fails: List[Dict[str, Any]],
    input_var: str = "input",
    output_var: str = "output",
    include_calibration: bool = True,
    compact_mode: bool = True
) -> str:
    """
    Build a production-quality eval prompt.

    Improvements over previous version:
    - 30% fewer tokens through concise phrasing
    - Numeric thresholds in rubrics
    - Tiebreaker rules
    - 8-10 calibration examples
    - Intermediate score guidance
    """

    # Header (minimal)
    prompt_parts = [
        f"# Evaluator for: {system_purpose}\n",
        f"**Input:** {{{{{input_var}}}}}",
        f"**Response:** {{{{{output_var}}}}}\n",
    ]

    # Auto-fails (compact, numbered)
    if auto_fails:
        prompt_parts.append("## Auto-Fail Conditions (Score=1, FAIL immediately)")
        for i, af in enumerate(auto_fails[:5], 1):
            desc = af.get('description', '')[:60]
            prompt_parts.append(f"{i}. **{af['name']}**: {desc}")
        prompt_parts.append("")

    # Dimensions with NUMERIC rubrics
    prompt_parts.append("## Scoring Dimensions (use numeric thresholds)\n")

    total_weight = sum(d.get('weight', 0.2) for d in dimensions)

    for d in dimensions[:5]:
        weight_pct = int((d.get('weight', 0.2) / total_weight) * 100)
        dim_name = d.get('name', 'Quality')

        # Get numeric rubric
        rubric = get_numeric_rubric(dim_name)

        prompt_parts.append(f"### {dim_name} ({weight_pct}% weight)")
        prompt_parts.append("| Score | Threshold | Description |")
        prompt_parts.append("|-------|-----------|-------------|")
        for score in [5, 4, 3, 2, 1]:
            r = rubric.get(score, {"min_pct": 0, "description": "N/A"})
            prompt_parts.append(f"| {score} | ≥{r['min_pct']}% | {r['description'][:50]} |")
        prompt_parts.append("")

    # Tiebreaker rules (compact version)
    if compact_mode:
        prompt_parts.append(TIEBREAKER_RULES_COMPACT)
    else:
        prompt_parts.append(TIEBREAKER_RULES)

    # Calibration examples
    if include_calibration:
        examples = generate_calibration_examples(system_purpose, dimensions)
        prompt_parts.append(format_calibration_examples_for_prompt(examples, max_examples=8))

    # Verdict thresholds with intermediate scores
    prompt_parts.append("""
## Verdict Determination

**Calculate weighted_score** then apply thresholds:
- **PASS**: weighted_score ≥ 3.5 AND no auto-fails
- **NEEDS_REVIEW**: weighted_score 2.5-3.49 OR edge case requiring human judgment
- **FAIL**: weighted_score < 2.5 OR any auto-fail triggered

**Intermediate Scores (use these!):**
- 4.5 = Almost perfect, trivial issue only
- 3.5 = Acceptable but has notable gaps
- 2.5 = Borderline, human should review
- 1.5 = Very poor but not completely wrong
""")

    # Output format (concise)
    prompt_parts.append("""
## Output Format (JSON only)

```json
{
  "auto_fail_triggered": null | "failure_name",
  "dimensions": {
    "DimensionName": {"score": 1-5, "pct": 0-100, "reason": "brief"}
  },
  "weighted_score": 0.0,
  "verdict": "PASS|NEEDS_REVIEW|FAIL",
  "reasoning": "2-3 sentences with specific evidence"
}
```

**IMPORTANT**:
- Use intermediate scores (1.5, 2.5, 3.5, 4.5) when appropriate
- Include "pct" showing the percentage that justified your score
- Score same quality = same score (consistency)

Evaluate now. Output ONLY valid JSON.""")

    return "\n".join(prompt_parts)


# =============================================================================
# SECTION 5: MANDATORY CONSISTENCY CHECKER
# =============================================================================

@dataclass
class ConsistencyCheckResult:
    """Result of consistency check."""
    is_consistent: bool
    score_variance: float
    max_deviation: float
    scores: List[float]
    verdicts: List[str]
    recommendation: str


class ConsistencyChecker:
    """
    Mandatory consistency checker for eval prompts.

    Runs same evaluation multiple times and ensures variance ≤ 0.3.
    """

    MAX_ALLOWED_DEVIATION = 0.3
    MIN_RUNS = 3

    async def check_consistency(
        self,
        eval_prompt: str,
        test_input: str,
        test_output: str,
        run_eval_func,  # async (prompt, input, output) -> (score, verdict)
        num_runs: int = 3
    ) -> ConsistencyCheckResult:
        """
        Run evaluation num_runs times and check consistency.

        Returns:
            ConsistencyCheckResult with variance analysis
        """
        if num_runs < self.MIN_RUNS:
            num_runs = self.MIN_RUNS

        scores = []
        verdicts = []

        for _ in range(num_runs):
            try:
                score, verdict = await run_eval_func(eval_prompt, test_input, test_output)
                scores.append(float(score))
                verdicts.append(verdict.upper())
            except Exception as e:
                logger.error(f"Consistency check run failed: {e}")

        if len(scores) < 2:
            return ConsistencyCheckResult(
                is_consistent=False,
                score_variance=999,
                max_deviation=999,
                scores=scores,
                verdicts=verdicts,
                recommendation="Insufficient successful runs for consistency check"
            )

        # Calculate statistics
        mean_score = statistics.mean(scores)
        variance = statistics.variance(scores) if len(scores) > 1 else 0
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0
        max_dev = max(abs(s - mean_score) for s in scores)

        # Check verdict consistency
        verdict_set = set(verdicts)
        verdict_consistent = len(verdict_set) == 1

        # Determine if consistent
        is_consistent = (
            max_dev <= self.MAX_ALLOWED_DEVIATION and
            verdict_consistent
        )

        # Generate recommendation
        if is_consistent:
            recommendation = f"PASS: Max deviation {max_dev:.2f} ≤ {self.MAX_ALLOWED_DEVIATION}, consistent verdicts"
        elif max_dev > self.MAX_ALLOWED_DEVIATION:
            recommendation = f"FAIL: Max deviation {max_dev:.2f} > {self.MAX_ALLOWED_DEVIATION}. Rubrics may be ambiguous."
        else:
            recommendation = f"FAIL: Inconsistent verdicts ({verdict_set}). Verdict thresholds may be unclear."

        return ConsistencyCheckResult(
            is_consistent=is_consistent,
            score_variance=round(variance, 4),
            max_deviation=round(max_dev, 4),
            scores=scores,
            verdicts=verdicts,
            recommendation=recommendation
        )

    def check_scores_consistent(self, scores: List[float]) -> Tuple[bool, float]:
        """
        Quick check if a list of scores is consistent.

        Returns:
            (is_consistent, max_deviation)
        """
        if len(scores) < 2:
            return True, 0.0

        mean_score = statistics.mean(scores)
        max_dev = max(abs(s - mean_score) for s in scores)

        return max_dev <= self.MAX_ALLOWED_DEVIATION, max_dev


# =============================================================================
# SECTION 6: INTERMEDIATE SCORE GUIDANCE
# =============================================================================

INTERMEDIATE_SCORE_GUIDANCE = """
## Using Intermediate Scores (REQUIRED)

Do NOT round all scores to whole numbers. Use half-point increments:

| Score | When to Use |
|-------|-------------|
| **5.0** | Perfect - zero issues, exceeds expectations |
| **4.5** | Near-perfect - one trivial issue that doesn't affect usability |
| **4.0** | Good - meets all requirements with minor imperfections |
| **3.5** | Acceptable - meets core requirements but has noticeable gaps |
| **3.0** | Borderline pass - minimum acceptable quality |
| **2.5** | Borderline fail - close to acceptable but has blocking issues |
| **2.0** | Poor - significant problems across multiple dimensions |
| **1.5** | Very poor - mostly fails but has some correct elements |
| **1.0** | Failure - completely wrong or triggers auto-fail |

### Example Score Distinctions:
- Missing 1 optional field: 4.5 (not 4.0 or 5.0)
- 2 minor inaccuracies: 3.5 (not 3.0 or 4.0)
- Wrong format but correct content: 2.5 (not 2.0 or 3.0)
"""


# =============================================================================
# SECTION 7: INTEGRATION HELPERS
# =============================================================================

def enhance_existing_rubric(rubric: Dict[int, str], dimension_name: str) -> Dict[int, str]:
    """
    Enhance an existing rubric with numeric thresholds.

    Takes vague rubrics like "mostly accurate" and adds percentages.
    """
    numeric = get_numeric_rubric(dimension_name)
    enhanced = {}

    for score, existing_text in rubric.items():
        threshold = numeric.get(score, {})
        min_pct = threshold.get("min_pct", 0)

        # Prepend numeric threshold to existing text
        enhanced[score] = f"[≥{min_pct}%] {existing_text}"

    return enhanced


def calculate_weighted_score(
    dimension_scores: Dict[str, float],
    dimension_weights: Dict[str, float],
    apply_tiebreaker: bool = True
) -> float:
    """
    Calculate weighted score with tiebreaker rules applied.

    Args:
        dimension_scores: {"Accuracy": 4.5, "Completeness": 3.0, ...}
        dimension_weights: {"Accuracy": 0.4, "Completeness": 0.3, ...}
        apply_tiebreaker: Whether to apply tiebreaker rules

    Returns:
        Final weighted score, rounded to nearest 0.5
    """
    if not dimension_scores:
        return 0.0

    # Calculate base weighted average
    total_weight = sum(dimension_weights.get(d, 0.2) for d in dimension_scores)
    weighted_sum = sum(
        score * dimension_weights.get(dim, 0.2)
        for dim, score in dimension_scores.items()
    )

    base_score = weighted_sum / total_weight if total_weight > 0 else 0

    if apply_tiebreaker:
        # Tiebreaker Rule 2: Accuracy priority
        accuracy_score = None
        for dim, score in dimension_scores.items():
            if "accuracy" in dim.lower():
                accuracy_score = score
                break

        if accuracy_score is not None:
            other_scores = [s for d, s in dimension_scores.items() if "accuracy" not in d.lower()]
            if other_scores:
                avg_others = statistics.mean(other_scores)
                if abs(accuracy_score - avg_others) > 1.0:
                    # Weight accuracy 2x
                    accuracy_weight = dimension_weights.get("Accuracy", 0.4) * 2
                    adjusted_total = total_weight + dimension_weights.get("Accuracy", 0.4)
                    base_score = (
                        weighted_sum + accuracy_score * dimension_weights.get("Accuracy", 0.4)
                    ) / adjusted_total

    # Round to nearest 0.5
    rounded_score = round(base_score * 2) / 2

    return rounded_score


def determine_verdict(
    weighted_score: float,
    auto_fail_triggered: Optional[str] = None
) -> str:
    """Determine verdict based on score and auto-fail status."""
    if auto_fail_triggered:
        return "FAIL"

    if weighted_score >= 3.5:
        return "PASS"
    elif weighted_score >= 2.5:
        return "NEEDS_REVIEW"
    else:
        return "FAIL"


# =============================================================================
# SECTION 8: TOKEN COUNTING & OPTIMIZATION
# =============================================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count (rough: ~4 chars per token)."""
    return len(text) // 4


def optimize_prompt_tokens(prompt: str, target_reduction_pct: float = 30) -> str:
    """
    Optimize prompt to reduce token count.

    Removes:
    - Excessive whitespace
    - Redundant examples
    - Verbose phrasing
    """
    optimized = prompt

    # Remove multiple blank lines
    optimized = re.sub(r'\n{3,}', '\n\n', optimized)

    # Remove trailing whitespace
    optimized = re.sub(r' +\n', '\n', optimized)

    # Compact bullet points
    optimized = re.sub(r'\n- ', '\n• ', optimized)

    # Remove verbose phrases
    replacements = [
        ("Please note that", "Note:"),
        ("It is important to", ""),
        ("Make sure to", ""),
        ("You should", ""),
        ("In order to", "To"),
        ("as well as", "and"),
        ("in addition to", "plus"),
        ("with respect to", "for"),
    ]

    for old, new in replacements:
        optimized = optimized.replace(old, new)

    return optimized.strip()


# =============================================================================
# SECTION 9: MAIN INTEGRATION FUNCTION
# =============================================================================

def create_enhanced_eval_prompt(
    system_purpose: str,
    dimensions: List[Dict[str, Any]],
    auto_fails: List[Dict[str, Any]],
    input_var: str = "input",
    output_var: str = "output"
) -> Tuple[str, Dict[str, Any]]:
    """
    Create an enhanced eval prompt with all quality improvements.

    Returns:
        Tuple of (eval_prompt, metadata)

    Metadata includes:
        - token_count: Estimated tokens
        - calibration_examples: Number of examples included
        - numeric_thresholds: True
        - tiebreaker_rules: True
        - intermediate_scores: True
    """
    # Build the prompt
    prompt = build_production_eval_prompt(
        system_purpose=system_purpose,
        dimensions=dimensions,
        auto_fails=auto_fails,
        input_var=input_var,
        output_var=output_var,
        include_calibration=True,
        compact_mode=True
    )

    # Optimize tokens
    prompt = optimize_prompt_tokens(prompt)

    # Calculate metadata
    token_count = estimate_tokens(prompt)

    metadata = {
        "token_count": token_count,
        "calibration_examples": 8,
        "numeric_thresholds": True,
        "tiebreaker_rules": True,
        "intermediate_scores": True,
        "consistency_check_required": True,
        "max_deviation_allowed": ConsistencyChecker.MAX_ALLOWED_DEVIATION,
    }

    return prompt, metadata


# Global consistency checker instance
_consistency_checker: Optional[ConsistencyChecker] = None

def get_consistency_checker() -> ConsistencyChecker:
    """Get or create the global consistency checker."""
    global _consistency_checker
    if _consistency_checker is None:
        _consistency_checker = ConsistencyChecker()
    return _consistency_checker


# =============================================================================
# SECTION 10: OUTPUT VALIDATION - Verify features in generated prompt
# =============================================================================

@dataclass
class EvalPromptQualityReport:
    """Report on the quality of a generated eval prompt."""
    overall_score: float  # 0-100
    has_numeric_thresholds: bool
    has_calibration_examples: bool
    calibration_example_count: int
    has_intermediate_scores: bool
    has_tiebreaker_rules: bool
    has_consistency_guidance: bool
    has_auto_fail_conditions: bool
    has_rubrics: bool
    issues: List[str]
    recommendations: List[str]
    feature_scores: Dict[str, int]  # Each feature: 0-10


def validate_eval_prompt_quality(eval_prompt: str) -> EvalPromptQualityReport:
    """
    Validate that a generated eval prompt contains all quality features.

    This function checks for:
    1. Numeric thresholds in rubrics (≥90%, etc.)
    2. Calibration examples (ideally 8-10)
    3. Intermediate score guidance (2.5, 3.5, 4.5)
    4. Tiebreaker rules
    5. Consistency guidance
    6. Auto-fail conditions
    7. Rubrics for each dimension

    Returns a quality report with scores and recommendations.
    """
    issues = []
    recommendations = []
    feature_scores = {}

    # 1. Check for numeric thresholds
    has_numeric = any(t in eval_prompt for t in ["≥95%", "≥90%", "≥85%", "≥80%", "≥70%", "[≥"])
    feature_scores["numeric_thresholds"] = 10 if has_numeric else 0
    if not has_numeric:
        issues.append("Missing numeric thresholds in rubrics")
        recommendations.append("Add percentage thresholds like '≥95% = Score 5'")

    # 2. Check for calibration examples
    cal_patterns = [
        r"Score\s+5[./]5",
        r"Score\s+4[./]5",
        r"Score\s+3[./]5",
        r"Score\s+[12][./]5",
        r"Example.*Score.*\d",
        r"Calibration.*Example",
    ]
    cal_count = 0
    for pattern in cal_patterns:
        cal_count += len(re.findall(pattern, eval_prompt, re.IGNORECASE))
    cal_count = min(cal_count, 10)  # Cap at 10

    has_calibration = cal_count >= 3
    feature_scores["calibration_examples"] = min(10, cal_count)
    if cal_count < 8:
        if cal_count < 3:
            issues.append(f"Insufficient calibration examples ({cal_count} found, need 8-10)")
        else:
            recommendations.append(f"Consider adding more calibration examples ({cal_count} found, ideal is 8-10)")

    # 3. Check for intermediate scores
    intermediate_scores = ["4.5", "3.5", "2.5", "1.5"]
    found_intermediate = sum(1 for s in intermediate_scores if s in eval_prompt)
    has_intermediate = found_intermediate >= 2
    feature_scores["intermediate_scores"] = min(10, found_intermediate * 3)
    if not has_intermediate:
        issues.append("Missing intermediate score guidance (2.5, 3.5, 4.5)")
        recommendations.append("Add guidance for when to use half-point scores")

    # 4. Check for tiebreaker rules
    tiebreaker_indicators = [
        "tiebreaker",
        "accuracy priority",
        "dimension.*conflict",
        "round.*0.5",
        "auto-fail trumps",
    ]
    has_tiebreaker = any(re.search(p, eval_prompt, re.IGNORECASE) for p in tiebreaker_indicators)
    feature_scores["tiebreaker_rules"] = 10 if has_tiebreaker else 0
    if not has_tiebreaker:
        recommendations.append("Add tiebreaker rules for when dimension scores conflict")

    # 5. Check for consistency guidance
    consistency_indicators = ["±0.3", "consistency", "same output.*same score", "similar.*similar score"]
    has_consistency = any(re.search(p, eval_prompt, re.IGNORECASE) for p in consistency_indicators)
    feature_scores["consistency_guidance"] = 10 if has_consistency else 0
    if not has_consistency:
        recommendations.append("Add consistency requirements (e.g., ±0.3 tolerance)")

    # 6. Check for auto-fail conditions
    autofail_indicators = ["auto-fail", "auto_fail", "critical failure", "immediate fail", "non-negotiable"]
    has_autofail = any(re.search(p, eval_prompt, re.IGNORECASE) for p in autofail_indicators)
    feature_scores["auto_fail_conditions"] = 10 if has_autofail else 0
    if not has_autofail:
        issues.append("Missing auto-fail conditions")
        recommendations.append("Add explicit auto-fail conditions (hallucination, prompt leak, etc.)")

    # 7. Check for rubrics
    rubric_indicators = ["rubric", "score.*5", "score.*4", "score.*3", "scoring criteria"]
    has_rubrics = sum(1 for p in rubric_indicators if re.search(p, eval_prompt, re.IGNORECASE)) >= 2
    feature_scores["rubrics"] = 10 if has_rubrics else 0
    if not has_rubrics:
        issues.append("Missing or incomplete rubrics")
        recommendations.append("Add detailed rubrics for each dimension with score definitions")

    # Calculate overall score
    overall_score = sum(feature_scores.values()) / len(feature_scores) * 10

    return EvalPromptQualityReport(
        overall_score=round(overall_score, 1),
        has_numeric_thresholds=has_numeric,
        has_calibration_examples=has_calibration,
        calibration_example_count=cal_count,
        has_intermediate_scores=has_intermediate,
        has_tiebreaker_rules=has_tiebreaker,
        has_consistency_guidance=has_consistency,
        has_auto_fail_conditions=has_autofail,
        has_rubrics=has_rubrics,
        issues=issues,
        recommendations=recommendations,
        feature_scores=feature_scores
    )


def get_quality_summary(report: EvalPromptQualityReport) -> str:
    """Get a human-readable summary of the quality report."""
    lines = [
        f"Overall Quality Score: {report.overall_score}/100",
        "",
        "Features Present:",
    ]

    features = [
        ("Numeric Thresholds", report.has_numeric_thresholds),
        ("Calibration Examples", f"{report.calibration_example_count}/8-10"),
        ("Intermediate Scores", report.has_intermediate_scores),
        ("Tiebreaker Rules", report.has_tiebreaker_rules),
        ("Consistency Guidance", report.has_consistency_guidance),
        ("Auto-Fail Conditions", report.has_auto_fail_conditions),
        ("Rubrics", report.has_rubrics),
    ]

    for name, status in features:
        if isinstance(status, bool):
            mark = "✓" if status else "✗"
        else:
            mark = status
        lines.append(f"  {mark} {name}")

    if report.issues:
        lines.append("")
        lines.append("Issues:")
        for issue in report.issues:
            lines.append(f"  ⚠ {issue}")

    if report.recommendations:
        lines.append("")
        lines.append("Recommendations:")
        for rec in report.recommendations:
            lines.append(f"  → {rec}")

    return "\n".join(lines)
