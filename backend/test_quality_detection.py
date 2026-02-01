"""
Test script to diagnose quality detection issues
"""

from eval_prompt_parser import analyze_eval_prompt_quality
import logging

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

# Sample eval prompt (what we expect the LLM to generate)
sample_prompt = """# Evaluation Criteria: Accuracy

## AI System Context
**System Prompt:**
```
Test system
```

## Purpose
This evaluation measures accuracy.

## Scoring Rubric (0-10 Scale)

**0-2 points: Severe Issues**
Observable behaviors:
- Contains factual errors
- Contradicts source material

**3-4 points: Major Issues**
Observable behaviors:
- Some inaccuracies present

**5-6 points: Acceptable**
Observable behaviors:
- Mostly accurate
- Minor errors

**7-8 points: Good**
Observable behaviors:
- Accurate information
- Well-sourced

**9-10 points: Excellent**
Observable behaviors:
- Completely accurate
- No errors

## Pass Criteria
- Must score 8.5 or higher to pass

## Evaluation Process (REQUIRED STEPS)

**Step 1: Analysis**
Examine the output for accuracy.

**Step 2: Evidence Collection**
List specific examples.

**Step 3: Rubric Mapping**
Map evidence to rubric.

**Step 4: Confidence Assessment**
Determine reliability.

**Step 5: Score Assignment**
Assign score 0-10.

## Output Format (REQUIRED)

```xml
<evaluation>
  <analysis>
    [Detailed analysis]
  </analysis>

  <evidence>
    <positive>
      - [Evidence]
    </positive>
    <negative>
      - [Evidence]
    </negative>
  </evidence>

  <rubric_mapping>
    [Mapping]
  </rubric_mapping>

  <score>X</score>

  <confidence>HIGH/MEDIUM/LOW</confidence>

  <confidence_reasoning>
    [Reasoning]
  </confidence_reasoning>

  <pass_fail>PASS/FAIL</pass_fail>
</evaluation>
```

## Handling Ambiguous Cases

If unclear:
- Set confidence to LOW
- Note missing information

## Calibration Examples

### Example 1: Low Quality (Score: 2/10)

<evaluation>
  <analysis>Contains multiple errors</analysis>
  <evidence>
    <positive></positive>
    <negative>
      - Factual error about dates
    </negative>
  </evidence>
  <rubric_mapping>Falls in 0-2 range</rubric_mapping>
  <score>2</score>
  <confidence>HIGH</confidence>
  <confidence_reasoning>Clear errors</confidence_reasoning>
  <pass_fail>FAIL</pass_fail>
</evaluation>

### Example 2: Medium Quality (Score: 6/10)

<evaluation>
  <analysis>Mostly accurate</analysis>
  <evidence>
    <positive>
      - Correct main points
    </positive>
    <negative>
      - Minor error in detail
    </negative>
  </evidence>
  <rubric_mapping>Falls in 5-6 range</rubric_mapping>
  <score>6</score>
  <confidence>MEDIUM</confidence>
  <confidence_reasoning>Some ambiguity</confidence_reasoning>
  <pass_fail>FAIL</pass_fail>
</evaluation>

### Example 3: High Quality (Score: 9/10)

<evaluation>
  <analysis>Excellent accuracy</analysis>
  <evidence>
    <positive>
      - All facts verified
      - No errors found
    </positive>
    <negative></negative>
  </evidence>
  <rubric_mapping>Falls in 9-10 range</rubric_mapping>
  <score>9</score>
  <confidence>HIGH</confidence>
  <confidence_reasoning>Strong evidence</confidence_reasoning>
  <pass_fail>PASS</pass_fail>
</evaluation>
"""

print("\n" + "="*60)
print("TESTING QUALITY DETECTION")
print("="*60 + "\n")

# Analyze the prompt
result = analyze_eval_prompt_quality(sample_prompt)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Chain-of-Thought: {result['has_chain_of_thought']}")
print(f"0-10 Scale: {result['has_0_10_scale']}")
print(f"Examples: {result['num_examples']}")
print(f"XML Format: {result['has_xml_format']}")
print(f"Uncertainty Handling: {result['has_uncertainty_handling']}")
print(f"Behavioral Anchors: {result['has_behavioral_anchors']}")
print(f"\n✨ QUALITY SCORE: {result['quality_score']}/10")
print("="*60)

# Expected score: 2.5 + 2.0 + 2.0 + 2.0 + 0.75 + 0.75 = 10.0
if result['quality_score'] >= 9.0:
    print("\n✅ PASS: Quality detection working correctly!")
else:
    print(f"\n⚠️ ISSUE: Expected 9.0+, got {result['quality_score']}")
    print("\nMissing features:")
    if not result['has_chain_of_thought']:
        print("  - Chain-of-thought")
    if not result['has_0_10_scale']:
        print("  - 0-10 scale")
    if result['num_examples'] < 3:
        print(f"  - Examples (got {result['num_examples']}, need 3)")
    if not result['has_xml_format']:
        print("  - XML format")
    if not result['has_uncertainty_handling']:
        print("  - Uncertainty handling")
    if not result['has_behavioral_anchors']:
        print("  - Behavioral anchors")
