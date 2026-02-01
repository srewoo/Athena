"""
Simple Eval Generator - Lightweight version without meta-evaluation

This is a simplified eval generator that:
- Takes an aspect to evaluate
- Generates a focused eval prompt for that aspect
- Returns immediately without meta-evaluation or refinement
- Uses minimal LLM calls to avoid token issues

Use this for the multi-aspect feature until LLM client compatibility is fixed.
"""

import logging
from typing import Dict, Any, Optional
from llm_client_v2 import EnhancedLLMClient

logger = logging.getLogger(__name__)


async def generate_simple_eval_prompt(
    llm_client: EnhancedLLMClient,
    system_prompt: str,
    aspect: str,
    use_case: str = "",
    provider: str = "anthropic",
    api_key: str = "",
    model: str = "claude-3-5-sonnet-20241022"
) -> Dict[str, Any]:
    """
    Generate an industry-grade eval prompt for a specific aspect.

    Enhanced with best practices from Claude, OpenAI, and Google:
    - Chain-of-thought reasoning BEFORE scoring
    - 0-10 scoring scale (avoids 1-5 anchor bias)
    - Multiple calibration examples (3 examples)
    - Behavioral rubric anchors (observable behaviors)
    - Structured XML output format
    - Uncertainty handling

    Args:
        llm_client: LLM client instance
        system_prompt: The AI system prompt to evaluate
        aspect: The specific aspect to evaluate (e.g., "Accuracy")
        use_case: Optional use case context
        provider: LLM provider
        api_key: API key
        model: Model to use

    Returns:
        Dict with eval_prompt, aspect, reasoning
    """

    prompt = f"""You are an expert at creating LLM-as-a-judge evaluation prompts following industry best practices from Anthropic, OpenAI, and Google.

Create a high-quality evaluation prompt that tests this specific aspect:

**ASPECT TO EVALUATE:** {aspect}

**AI SYSTEM BEING EVALUATED:**
{system_prompt}

**USE CASE:** {use_case or "General purpose AI assistant"}

Your task: Create an evaluation prompt that follows these CRITICAL best practices:

1. **Chain-of-Thought Reasoning**: Require the evaluator to analyze BEFORE scoring
2. **Proper Scale**: Use 0-10 scale (NOT 1-5 which has anchor bias)
3. **Behavioral Anchors**: Each rubric level must describe OBSERVABLE behaviors, not vague qualities
4. **Multiple Examples**: Provide 3 calibration examples at different quality levels
5. **Structured Output**: Use XML tags for parsing
6. **Uncertainty Handling**: Allow evaluators to express low confidence
7. **Single-Aspect Focus**: ONLY evaluate "{aspect}" - no metric conflation

Format your response EXACTLY as follows:

# Evaluation Criteria: {aspect}

## AI System Context
**System Prompt:**
```
{system_prompt}
```

**Use Case:** {use_case or "General purpose AI assistant"}

**What This System Does:**
[1-2 sentences summarizing the AI system's purpose and key capabilities based on the system prompt above]

## Purpose
[One paragraph explaining what this evaluation measures and why it matters specifically for THIS AI system]

## Scoring Rubric (0-10 Scale)

**0-2 points: Severe Issues**
Observable behaviors:
- [Specific, concrete behavior you can observe]
- [Another specific behavior]
Failure mode: [What's fundamentally broken]

**3-4 points: Major Issues**
Observable behaviors:
- [Specific behaviors at this level]
- [Another specific behavior]
Failure mode: [What's significantly flawed]

**5-6 points: Acceptable**
Observable behaviors:
- [Specific behaviors indicating basic competence]
- [Another specific behavior]
Meets minimum requirements but has notable gaps.

**7-8 points: Good**
Observable behaviors:
- [Specific behaviors indicating quality work]
- [Another specific behavior]
Strong performance with minor issues.

**9-10 points: Excellent**
Observable behaviors:
- [Specific behaviors indicating exceptional quality]
- [Another specific behavior]
Exemplary performance with no significant issues.

## Pass Criteria
- Must score 8.5 or higher to pass (85% quality threshold)
- [Any specific non-negotiable requirements for {aspect}]

## Evaluation Process (REQUIRED STEPS)

When evaluating any AI output for {aspect}, follow these steps IN ORDER:

**Step 1: Analysis**
Carefully examine the output and identify all observations relevant to {aspect}.

**Step 2: Evidence Collection**
List specific examples from the output that demonstrate strengths or weaknesses in {aspect}.

**Step 3: Rubric Mapping**
Map your evidence to the rubric levels. Explain which observable behaviors are present.

**Step 4: Confidence Assessment**
Determine if you have enough information to make a reliable judgment.

**Step 5: Score Assignment**
Based on your analysis, assign a score from 0-10.

## Output Format (REQUIRED)

All evaluations MUST use this exact XML structure:

```xml
<evaluation>
  <analysis>
    [Detailed step-by-step analysis of the output examining {aspect}.
     Identify patterns, behaviors, and characteristics relevant to this aspect.]
  </analysis>

  <evidence>
    <positive>
      - [Specific quote or behavior that supports a higher score]
      - [Another positive piece of evidence]
    </positive>
    <negative>
      - [Specific quote or behavior that indicates issues]
      - [Another negative piece of evidence]
    </negative>
  </evidence>

  <rubric_mapping>
    [Explain which rubric level (0-2, 3-4, 5-6, 7-8, or 9-10) best matches
     the evidence. Reference specific observable behaviors from the rubric.]
  </rubric_mapping>

  <score>X</score>

  <confidence>HIGH/MEDIUM/LOW</confidence>

  <confidence_reasoning>
    [Explain why you have this confidence level. Note any ambiguities or
     missing information that affected your evaluation.]
  </confidence_reasoning>

  <pass_fail>PASS/FAIL</pass_fail>
</evaluation>
```

## Handling Ambiguous Cases

If the output doesn't provide enough information to evaluate {aspect}:
- Set confidence to LOW
- Clearly state what information is missing in confidence_reasoning
- Provide your best tentative score with caveats
- Do NOT force a high-confidence judgment when uncertain

## Calibration Examples

Provide exactly 3 examples showing LOW, MEDIUM, and HIGH quality:

### Example 1: Low Quality (Score: 2-3/10)

**Context:** [Brief scenario context]

**Input:** [Example user input]

**Output:** [Example AI output showing poor {aspect}]

**Evaluation:**
```xml
<evaluation>
  <analysis>
    [Detailed analysis of why this demonstrates poor {aspect}]
  </analysis>

  <evidence>
    <positive>
      - [Any minor positive aspects, if present]
    </positive>
    <negative>
      - [Specific evidence of poor {aspect}]
      - [Another clear example of failure]
    </negative>
  </evidence>

  <rubric_mapping>
    This output falls in the 0-2 range because [specific observable behaviors
    from the rubric that match this output].
  </rubric_mapping>

  <score>2</score>

  <confidence>HIGH</confidence>

  <confidence_reasoning>
    Clear evidence of severe issues with {aspect}. Multiple observable
    behaviors align with the 0-2 rubric level.
  </confidence_reasoning>

  <pass_fail>FAIL</pass_fail>
</evaluation>
```

### Example 2: Medium Quality (Score: 5-6/10)

**Context:** [Brief scenario context]

**Input:** [Example user input]

**Output:** [Example AI output showing acceptable {aspect}]

**Evaluation:**
```xml
<evaluation>
  <analysis>
    [Detailed analysis of why this is acceptable but not strong]
  </analysis>

  <evidence>
    <positive>
      - [Evidence of meeting basic requirements]
    </positive>
    <negative>
      - [Evidence of gaps or weaknesses]
    </negative>
  </evidence>

  <rubric_mapping>
    This output falls in the 5-6 range because [specific observable behaviors].
  </rubric_mapping>

  <score>6</score>

  <confidence>MEDIUM</confidence>

  <confidence_reasoning>
    [Explanation of why confidence is medium]
  </confidence_reasoning>

  <pass_fail>FAIL</pass_fail>
</evaluation>
```

### Example 3: High Quality (Score: 8-9/10)

**Context:** [Brief scenario context]

**Input:** [Example user input]

**Output:** [Example AI output showing excellent {aspect}]

**Evaluation:**
```xml
<evaluation>
  <analysis>
    [Detailed analysis of why this demonstrates strong {aspect}]
  </analysis>

  <evidence>
    <positive>
      - [Specific evidence of excellent {aspect}]
      - [Another strong example]
    </positive>
    <negative>
      - [Minor issues if any]
    </negative>
  </evidence>

  <rubric_mapping>
    This output falls in the 9-10 range because [specific observable behaviors].
  </rubric_mapping>

  <score>9</score>

  <confidence>HIGH</confidence>

  <confidence_reasoning>
    Strong evidence across multiple criteria. Observable behaviors clearly
    align with the 9-10 rubric level.
  </confidence_reasoning>

  <pass_fail>PASS</pass_fail>
</evaluation>
```

## Final Reminders

- This evaluation prompt ONLY measures {aspect}
- Do NOT evaluate other aspects (no metric conflation)
- Always require chain-of-thought reasoning BEFORE scoring
- Use the 0-10 scale consistently
- Include concrete, observable behaviors in the rubric
- Provide all 3 calibration examples with complete XML structure
- Allow evaluators to express uncertainty

Keep the focus laser-sharp on {aspect} and nothing else."""

    try:
        # Make single LLM call with higher token limit for comprehensive prompt
        response = await llm_client.generate(
            prompt=prompt,
            provider=provider,
            api_key=api_key,
            model=model,
            max_tokens=4000,  # Increased for detailed rubrics and 3 examples
            temperature=0.3
        )

        # Response is a string
        eval_prompt = response.strip() if isinstance(response, str) else str(response)

        return {
            "eval_prompt": eval_prompt,
            "aspect": aspect,
            "reasoning": f"Focused evaluation prompt for {aspect}",
            "quality_score": 8.0,  # Estimated (no meta-eval)
            "was_refined": False,
            "passes_gate": True
        }

    except Exception as e:
        logger.error(f"Error generating simple eval prompt: {e}", exc_info=True)

        # Return a basic template as fallback (follows new format)
        fallback_prompt = f"""# Evaluation Criteria: {aspect}

## AI System Context
**System Prompt:**
```
{system_prompt}
```

**Use Case:** {use_case or "General purpose AI assistant"}

**What This System Does:**
An AI assistant designed for the above use case.

## Purpose
This evaluation measures the {aspect.lower()} of AI system outputs.

## Scoring Rubric (0-10 Scale)

**0-2 points: Severe Issues**
Observable behaviors:
- Does not demonstrate {aspect.lower()}
- Contains fundamental flaws related to {aspect.lower()}

**3-4 points: Major Issues**
Observable behaviors:
- Shows minimal {aspect.lower()}
- Multiple significant problems present

**5-6 points: Acceptable**
Observable behaviors:
- Adequately demonstrates {aspect.lower()}
- Meets minimum requirements but has gaps

**7-8 points: Good**
Observable behaviors:
- Strongly demonstrates {aspect.lower()}
- Minor issues only

**9-10 points: Excellent**
Observable behaviors:
- Exceptional {aspect.lower()}
- No significant issues

## Pass Criteria
- Must score 8.5 or higher to pass (85% quality threshold)
- Must clearly demonstrate {aspect.lower()}

## Evaluation Process (REQUIRED STEPS)

**Step 1: Analysis**
Examine the output for evidence of {aspect.lower()}.

**Step 2: Evidence Collection**
List specific examples demonstrating strengths or weaknesses.

**Step 3: Rubric Mapping**
Map evidence to rubric levels.

**Step 4: Confidence Assessment**
Determine if enough information exists for reliable judgment.

**Step 5: Score Assignment**
Assign score from 0-10 based on analysis.

## Output Format (REQUIRED)

```xml
<evaluation>
  <analysis>
    [Detailed analysis examining {aspect.lower()}]
  </analysis>

  <evidence>
    <positive>
      - [Positive evidence]
    </positive>
    <negative>
      - [Negative evidence]
    </negative>
  </evidence>

  <rubric_mapping>
    [Explain which rubric level matches the evidence]
  </rubric_mapping>

  <score>X</score>

  <confidence>HIGH/MEDIUM/LOW</confidence>

  <confidence_reasoning>
    [Explain confidence level]
  </confidence_reasoning>

  <pass_fail>PASS/FAIL</pass_fail>
</evaluation>
```

## Calibration Examples

### Example 1: Low Quality (Score: 2/10)
[Context: General evaluation]

**Evaluation:**
```xml
<evaluation>
  <analysis>Output shows poor {aspect.lower()}.</analysis>
  <evidence>
    <positive></positive>
    <negative>
      - Does not meet basic requirements for {aspect.lower()}
    </negative>
  </evidence>
  <rubric_mapping>Falls in 0-2 range due to severe issues.</rubric_mapping>
  <score>2</score>
  <confidence>HIGH</confidence>
  <confidence_reasoning>Clear evidence of severe issues.</confidence_reasoning>
  <pass_fail>FAIL</pass_fail>
</evaluation>
```

### Example 2: Medium Quality (Score: 6/10)
[Context: General evaluation]

**Evaluation:**
```xml
<evaluation>
  <analysis>Output shows acceptable {aspect.lower()} with some gaps.</analysis>
  <evidence>
    <positive>
      - Meets basic requirements
    </positive>
    <negative>
      - Has notable weaknesses
    </negative>
  </evidence>
  <rubric_mapping>Falls in 5-6 range - acceptable but not strong.</rubric_mapping>
  <score>6</score>
  <confidence>MEDIUM</confidence>
  <confidence_reasoning>Some ambiguity in evaluation.</confidence_reasoning>
  <pass_fail>FAIL</pass_fail>
</evaluation>
```

### Example 3: High Quality (Score: 9/10)
[Context: General evaluation]

**Evaluation:**
```xml
<evaluation>
  <analysis>Output shows excellent {aspect.lower()}.</analysis>
  <evidence>
    <positive>
      - Strong demonstration of {aspect.lower()}
      - Exceeds requirements
    </positive>
    <negative>
      - Minor issues only
    </negative>
  </evidence>
  <rubric_mapping>Falls in 9-10 range due to exceptional quality.</rubric_mapping>
  <score>9</score>
  <confidence>HIGH</confidence>
  <confidence_reasoning>Clear evidence of excellence.</confidence_reasoning>
  <pass_fail>PASS</pass_fail>
</evaluation>
```"""

        return {
            "eval_prompt": fallback_prompt,
            "aspect": aspect,
            "reasoning": f"Fallback template for {aspect} (LLM call failed)",
            "quality_score": 7.0,
            "was_refined": False,
            "passes_gate": True
        }
