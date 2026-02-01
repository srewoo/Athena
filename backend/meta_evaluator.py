"""
Meta-Evaluator System

Audits generated eval prompts to ensure they are high-quality, unbiased,
and properly aligned with the system prompt being evaluated.

This implements a "judge the judge" approach to ensure eval prompt quality.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from llm_client_v2 import EnhancedLLMClient

logger = logging.getLogger(__name__)


@dataclass
class MetaEvalResult:
    """Result from meta-evaluation of an eval prompt"""
    overall_quality_score: float  # 1-10
    passes_quality_gate: bool
    executive_summary: str

    # The 5-point audit
    effectiveness_score: float
    effectiveness_analysis: str

    structural_clarity_score: float
    structural_clarity_analysis: str

    bias_score: float
    bias_analysis: str

    metric_conflation_score: float
    metric_conflation_analysis: str

    granularity_score: float
    granularity_analysis: str

    # Issues found
    logic_gaps: List[Dict[str, str]]

    # Recommendations
    refinement_roadmap: List[str]

    # Auto-fix suggestions
    suggested_improvements: Dict[str, Any]


META_EVAL_PROMPT_TEMPLATE = """### ROLE
You are a Senior Prompt Engineer and LLM Evaluation Specialist. Your role is to perform a "Meta-Evaluation"—a high-level audit of the relationship between a generator prompt (the "Worker") and an evaluator prompt (the "Judge"). You ensure that the Judge is fair, objective, and aligns perfectly with the goals of the Worker.

### INPUT DATA
<model_prompt>
{model_prompt}
</model_prompt>

<evaluator_prompt>
{evaluator_prompt}
</evaluator_prompt>

### THE 5-POINT AUDIT FRAMEWORK
Evaluate the `<evaluator_prompt>` based on these specific criteria:

1. **Effectiveness & Relevance**: Is the evaluator actually measuring the specific task defined in the `<model_prompt>`? Does it align with the provided dataset or PRD?
2. **Structural Clarity**: Are the Role, Goal, and Scoring Rubrics defined properly?
3. **Bias & Logic**: Identify biases toward technicalities (e.g., penalizing JSON format over content quality).
   * *Note: Pure schema/JSON checks should be flagged as "Better handled via programmatic code."*
4. **Metric Conflation**: Are multiple variables (e.g., Tone, Accuracy, and Relevance) combined into a single score? Recommend separation if they conflict.
5. **Granularity**: Is the scoring scale appropriate (e.g., 1-5 Likert vs. Binary Pass/Fail)?

### OPERATING PRINCIPLES
* **Source of Truth**: The "Goal" defined in the `<model_prompt>` is the absolute source of truth.
* **CRITICAL CONSTRAINT**: DO NOT rewrite the `<evaluator_prompt>` directly. Do not change its technical style.
* **Technical Integrity**: Respect and preserve all Jinja templates, variable placeholders, and special technical characters.
* **Guidance vs. Replacement**: Provide logic gaps and suggestions for refinement rather than doing the work for the user.

### OUTPUT FORMAT
Your response must follow this structure and be valid JSON:

```json
{{
  "executive_summary": "A brief analysis of how well the Judge understands the Worker's task.",

  "audit_scores": {{
    "effectiveness_relevance": {{
      "score": <1-10>,
      "analysis": "Detailed analysis of effectiveness and relevance"
    }},
    "structural_clarity": {{
      "score": <1-10>,
      "analysis": "Analysis of structural clarity"
    }},
    "bias_logic": {{
      "score": <1-10>,
      "analysis": "Analysis of bias and logic issues"
    }},
    "metric_conflation": {{
      "score": <1-10>,
      "analysis": "Analysis of metric conflation"
    }},
    "granularity": {{
      "score": <1-10>,
      "analysis": "Analysis of scoring granularity"
    }}
  }},

  "logic_gaps": [
    {{
      "gap": "Description of what is missing or contradictory",
      "worker_evidence": "Raw string from model_prompt showing expectation",
      "judge_evidence": "Raw string from evaluator_prompt showing mismatch"
    }}
  ],

  "refinement_roadmap": [
    "Specific instruction 1 for improving the eval prompt",
    "Specific instruction 2 for improving the eval prompt",
    "Specific instruction 3 for improving the eval prompt"
  ],

  "suggested_improvements": {{
    "add_persona": "Suggested persona if missing",
    "add_tone_guidance": "Suggested tone guidance if missing",
    "add_success_criteria": ["Specific success criterion 1", "Specific success criterion 2"],
    "add_failure_criteria": ["Specific failure criterion 1", "Specific failure criterion 2"],
    "add_examples": ["Example 1", "Example 2"],
    "separate_metrics": ["Metric 1 to separate", "Metric 2 to separate"]
  }}
}}
```

CRITICAL: Return ONLY valid JSON. No additional text before or after.
"""


class MetaEvaluator:
    """
    Meta-evaluator that audits eval prompts for quality
    """

    def __init__(
        self,
        llm_client: EnhancedLLMClient,
        quality_threshold: float = 8.5
    ):
        self.llm_client = llm_client
        self.quality_threshold = quality_threshold

    async def evaluate_eval_prompt(
        self,
        system_prompt: str,
        eval_prompt: str,
        model: str = "claude-sonnet-4-5-20250929"
    ) -> MetaEvalResult:
        """
        Run meta-evaluation on an eval prompt

        Args:
            system_prompt: The original system prompt being evaluated
            eval_prompt: The eval prompt to audit
            model: LLM model to use for meta-evaluation

        Returns:
            MetaEvalResult with detailed analysis
        """

        logger.info("Running meta-evaluation on eval prompt...")

        # Build meta-eval prompt
        meta_prompt = META_EVAL_PROMPT_TEMPLATE.format(
            model_prompt=system_prompt[:5000],  # Truncate if too long
            evaluator_prompt=eval_prompt[:10000]
        )

        # Run meta-evaluation
        result = await self.llm_client.generate(
            prompt=meta_prompt,
            model=model,
            max_tokens=3000,
            temperature=0.3  # Lower temp for more consistent analysis
        )

        # Parse result
        output = result.get("output", "{}")

        # Extract JSON
        import json
        import re

        json_match = re.search(r'\{[\s\S]*\}', output)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except json.JSONDecodeError:
                logger.error("Failed to parse meta-eval JSON")
                return self._create_error_result("Failed to parse meta-evaluation response")
        else:
            logger.error("No JSON found in meta-eval response")
            return self._create_error_result("No valid JSON in meta-evaluation response")

        # Extract scores
        audit_scores = parsed.get("audit_scores", {})

        effectiveness = audit_scores.get("effectiveness_relevance", {})
        structural = audit_scores.get("structural_clarity", {})
        bias = audit_scores.get("bias_logic", {})
        conflation = audit_scores.get("metric_conflation", {})
        granularity = audit_scores.get("granularity", {})

        # Calculate overall quality score
        scores = [
            effectiveness.get("score", 5),
            structural.get("score", 5),
            bias.get("score", 5),
            conflation.get("score", 5),
            granularity.get("score", 5)
        ]
        overall_score = sum(scores) / len(scores)

        # Determine if passes quality gate
        passes = overall_score >= self.quality_threshold

        return MetaEvalResult(
            overall_quality_score=overall_score,
            passes_quality_gate=passes,
            executive_summary=parsed.get("executive_summary", ""),

            effectiveness_score=effectiveness.get("score", 5),
            effectiveness_analysis=effectiveness.get("analysis", ""),

            structural_clarity_score=structural.get("score", 5),
            structural_clarity_analysis=structural.get("analysis", ""),

            bias_score=bias.get("score", 5),
            bias_analysis=bias.get("analysis", ""),

            metric_conflation_score=conflation.get("score", 5),
            metric_conflation_analysis=conflation.get("analysis", ""),

            granularity_score=granularity.get("score", 5),
            granularity_analysis=granularity.get("analysis", ""),

            logic_gaps=parsed.get("logic_gaps", []),
            refinement_roadmap=parsed.get("refinement_roadmap", []),
            suggested_improvements=parsed.get("suggested_improvements", {})
        )

    def _create_error_result(self, error_message: str) -> MetaEvalResult:
        """Create error result when meta-eval fails"""
        return MetaEvalResult(
            overall_quality_score=0.0,
            passes_quality_gate=False,
            executive_summary=f"Meta-evaluation failed: {error_message}",
            effectiveness_score=0,
            effectiveness_analysis=error_message,
            structural_clarity_score=0,
            structural_clarity_analysis=error_message,
            bias_score=0,
            bias_analysis=error_message,
            metric_conflation_score=0,
            metric_conflation_analysis=error_message,
            granularity_score=0,
            granularity_analysis=error_message,
            logic_gaps=[],
            refinement_roadmap=[],
            suggested_improvements={}
        )


async def apply_meta_eval_improvements(
    eval_prompt: str,
    meta_result: MetaEvalResult,
    llm_client: EnhancedLLMClient
) -> str:
    """
    Apply suggested improvements from meta-evaluation to eval prompt

    This generates an improved version based on the refinement roadmap
    """

    if meta_result.passes_quality_gate:
        logger.info("Eval prompt already passes quality gate, no improvements needed")
        return eval_prompt

    logger.info(f"Applying improvements (quality score: {meta_result.overall_quality_score:.1f}/10)")

    improvement_prompt = f"""You are improving an LLM evaluation prompt based on expert feedback.

**Current Eval Prompt:**
{eval_prompt}

**Meta-Evaluation Feedback:**

Executive Summary: {meta_result.executive_summary}

**Issues Found:**
{chr(10).join(f"- {gap['gap']}" for gap in meta_result.logic_gaps[:5])}

**Refinement Roadmap:**
{chr(10).join(f"{i+1}. {item}" for i, item in enumerate(meta_result.refinement_roadmap[:5]))}

**Suggested Improvements:**
{chr(10).join(f"- {k}: {v}" for k, v in list(meta_result.suggested_improvements.items())[:5])}

**Task:**
Improve the eval prompt by addressing the issues in the refinement roadmap while:
1. Preserving all Jinja templates and variable placeholders exactly
2. Maintaining the technical structure
3. Adding missing elements (persona, criteria, examples)
4. Fixing bias and metric conflation issues
5. Keeping the same overall format

Return ONLY the improved eval prompt. No explanations.
"""

    result = await llm_client.generate(
        prompt=improvement_prompt,
        model="claude-sonnet-4-5-20250929",
        max_tokens=8000
    )

    improved_prompt = result.get("output", eval_prompt)

    # Basic validation - ensure key parts weren't lost
    if len(improved_prompt) < len(eval_prompt) * 0.5:
        logger.warning("Improved prompt too short, returning original")
        return eval_prompt

    return improved_prompt


async def iterative_meta_eval_refinement(
    system_prompt: str,
    initial_eval_prompt: str,
    llm_client: EnhancedLLMClient,
    max_iterations: int = 3,
    quality_threshold: float = 8.5
) -> tuple[str, List[MetaEvalResult]]:
    """
    Iteratively improve eval prompt using meta-evaluation feedback

    Returns:
        (final_eval_prompt, history_of_meta_evals)
    """

    meta_evaluator = MetaEvaluator(llm_client, quality_threshold=quality_threshold)

    current_prompt = initial_eval_prompt
    meta_eval_history = []

    for iteration in range(max_iterations):
        logger.info(f"Meta-eval iteration {iteration + 1}/{max_iterations}")

        # Run meta-evaluation
        meta_result = await meta_evaluator.evaluate_eval_prompt(
            system_prompt=system_prompt,
            eval_prompt=current_prompt
        )

        meta_eval_history.append(meta_result)

        logger.info(
            f"Quality score: {meta_result.overall_quality_score:.1f}/10 "
            f"(threshold: {quality_threshold})"
        )

        # Check if we've reached quality threshold
        if meta_result.passes_quality_gate:
            logger.info(f"✓ Quality gate passed in {iteration + 1} iterations")
            break

        # Apply improvements
        current_prompt = await apply_meta_eval_improvements(
            eval_prompt=current_prompt,
            meta_result=meta_result,
            llm_client=llm_client
        )

        logger.info(f"Applied improvements, prompt length: {len(current_prompt)}")

    final_result = meta_eval_history[-1]
    logger.info(
        f"Final quality score: {final_result.overall_quality_score:.1f}/10 "
        f"after {len(meta_eval_history)} iterations"
    )

    return current_prompt, meta_eval_history
