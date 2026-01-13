"""
Example: Using Coverage Validation in the Validated Eval Pipeline

This example demonstrates how the new Coverage & Alignment Agent works
to ensure eval prompts comprehensively cover system prompt requirements.

Author: Athena Team
"""

import asyncio
import json
from validated_eval_pipeline import ValidatedEvalPipeline, CostBudget
from eval_coverage_validator import validate_eval_coverage


async def example_basic_coverage_check():
    """
    Example 1: Basic coverage check without auto-improvement.

    This just analyzes coverage and reports gaps.
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic Coverage Check")
    print("=" * 80)

    system_prompt = """
    You are a customer support AI assistant for an e-commerce platform.

    Your responsibilities:
    - Answer customer questions about orders, shipping, and returns
    - Help customers track their orders using order IDs
    - Process return requests following our 30-day return policy
    - Escalate complex issues to human agents

    Important constraints:
    - Never share customer data with unauthorized parties
    - Do not process refunds above $500 without manager approval
    - Always validate order IDs before providing information
    - Maintain a professional and empathetic tone

    Output format: Respond in clear, concise JSON format with fields:
    - "response": your answer to the customer
    - "action_taken": what action you performed (if any)
    - "escalation_needed": boolean indicating if human intervention is required
    """

    # A generated eval prompt that might have gaps
    eval_prompt = """
    Evaluate the customer support response for quality.

    Dimensions:
    1. Clarity (30%): Is the response clear and easy to understand?
    2. Helpfulness (40%): Does it address the customer's question?
    3. Professionalism (30%): Is the tone appropriate?

    Scoring:
    - 5: Excellent
    - 4: Good
    - 3: Acceptable
    - 2: Poor
    - 1: Unacceptable
    """

    # Run coverage check (no auto-improvement)
    result = await validate_eval_coverage(
        system_prompt=system_prompt,
        eval_prompt=eval_prompt,
        auto_improve=False
    )

    print("\nCoverage Analysis:")
    print(f"  Coverage: {result['coverage']['percentage']}%")
    print(f"  Level: {result['coverage']['level']}")
    print(f"  Passes Threshold: {result['coverage']['passes_threshold']}")
    print(f"  Gaps Found: {result['coverage']['gaps_found']}")

    print("\nTop Gaps:")
    for i, gap in enumerate(result['gaps'][:5], 1):
        print(f"\n  {i}. [{gap['severity'].upper()}] {gap['category']}")
        print(f"     Missing: {gap['missing'][:80]}")
        print(f"     Recommendation: {gap['recommendation'][:100]}")

    print("\n" + "=" * 80 + "\n")


async def example_auto_improvement():
    """
    Example 2: Coverage validation with automatic iterative improvement.

    This will analyze gaps and automatically improve the eval prompt.
    """
    print("=" * 80)
    print("EXAMPLE 2: Auto-Improvement with Iteration")
    print("=" * 80)

    system_prompt = """
    You are a code review assistant that analyzes Python code for best practices.

    You must check for:
    - PEP 8 style compliance
    - Security vulnerabilities (SQL injection, XSS, etc.)
    - Performance issues (inefficient loops, unnecessary copies)
    - Proper error handling with try-except blocks
    - Type hints for function parameters and returns
    - Docstrings for all public functions and classes

    Critical constraints:
    - Never suggest changes that would break functionality
    - Always provide specific line numbers for issues
    - Prioritize security issues over style issues

    Output format: JSON with fields:
    - "issues": list of issues found
    - "severity": critical/high/medium/low
    - "suggestions": specific improvements to make
    """

    # Initial eval prompt with gaps
    eval_prompt = """
    Evaluate the code review response.

    Check if the review:
    - Identifies issues correctly
    - Provides helpful suggestions
    - Uses appropriate tone

    Score 1-5 based on quality.
    """

    # This would need an actual LLM client in production
    # For demonstration, we'll show what it would do:

    print("\nInitial Eval Prompt:")
    print(eval_prompt[:200] + "...")

    # In production with LLM client:
    # result = await validate_eval_coverage(
    #     system_prompt=system_prompt,
    #     eval_prompt=eval_prompt,
    #     llm_client=your_llm_client,
    #     auto_improve=True,
    #     max_iterations=3
    # )

    print("\n[With LLM Client, this would:]")
    print("  1. Analyze system prompt → Extract 6 check requirements, 3 constraints")
    print("  2. Check coverage → Find gaps in security checking, output format validation")
    print("  3. Generate improvements → Add security dimension, format compliance checks")
    print("  4. Apply improvements → Regenerate eval prompt with additions")
    print("  5. Re-check coverage → Verify gaps are filled")
    print("  6. Iterate if needed → Up to 3 iterations")

    print("\nExpected Result:")
    print("  Original Coverage: ~35% (only checked tone and helpfulness)")
    print("  After Iteration 1: ~70% (added security and format checks)")
    print("  After Iteration 2: ~85% (added specific line number validation)")
    print("  Status: PASS (threshold met)")

    print("\n" + "=" * 80 + "\n")


async def example_integrated_pipeline():
    """
    Example 3: Coverage validation integrated into full pipeline.

    This shows how coverage validation works as Gate 3 in the complete
    validated eval pipeline.
    """
    print("=" * 80)
    print("EXAMPLE 3: Integrated Pipeline with Coverage Gate")
    print("=" * 80)

    system_prompt = """
    You are a medical symptom checker assistant.

    Capabilities:
    - Collect symptom information from users
    - Suggest possible conditions (educational purposes only)
    - Recommend when to seek medical attention

    Critical constraints:
    - NEVER provide definitive diagnoses
    - NEVER suggest specific medications
    - ALWAYS recommend consulting a healthcare provider
    - Do not ask for or store personal health information

    Output: Provide educational information with clear disclaimers.
    """

    # Initialize pipeline with coverage validation enabled
    pipeline = ValidatedEvalPipeline(
        project_id="medical_symptom_checker",
        enable_coverage_validation=True,  # ENABLE COVERAGE GATE
        max_coverage_iterations=3,
        cost_budget=CostBudget(max_cost_per_validation=1.00)
    )

    print("\nPipeline Configuration:")
    print(f"  Coverage Validation: Enabled")
    print(f"  Max Coverage Iterations: 3")
    print(f"  Coverage Threshold: 80%")

    print("\nValidation Gate Flow:")
    print("  Gate 1: Semantic Analysis ✓")
    print("  Gate 2: Feedback Learning ✓")
    print("  Gate 3: Coverage & Alignment ← NEW! (runs before other gates)")
    print("          ↓ Analyzes system prompt requirements")
    print("          ↓ Checks eval prompt coverage")
    print("          ↓ Identifies gaps")
    print("          ↓ Iteratively improves (up to 3x)")
    print("          ↓ Returns improved eval prompt")
    print("  Gate 4: Cost Budget ✓")
    print("  Gate 5-9: Ground Truth, Reliability, Adversarial, Statistical, Consistency ✓")

    print("\nWhat Coverage Gate Would Check:")
    print("  ✓ Does eval test 'collect symptom information' capability?")
    print("  ✓ Does eval enforce 'NEVER provide diagnoses' constraint?")
    print("  ✓ Does eval verify disclaimer is included?")
    print("  ✓ Does eval check for medication suggestions (auto-fail)?")
    print("  ✓ Does eval validate output format?")

    print("\nExpected Coverage Results:")
    print("  Iteration 1: 60% coverage → Add constraint checks")
    print("  Iteration 2: 82% coverage → Add disclaimer validation")
    print("  Status: PASS (threshold exceeded)")
    print("  Improved eval prompt used for subsequent gates")

    print("\n[In production, call:]")
    print("""
    eval_prompt, validation_result = await pipeline.generate_and_validate(
        system_prompt=system_prompt,
        use_case="Medical symptom education",
        requirements=["Educational only", "No diagnoses", "Clear disclaimers"],
        provider="openai",
        api_key="...",
        model_name="gpt-4o",
        sample_input="I have a headache and fever",
        sample_output="These symptoms could indicate...[with disclaimer]"
    )

    # Check if coverage gate passed
    coverage_gate = next(g for g in validation_result.gates if g.gate_name == "coverage_alignment")
    print(f"Coverage: {coverage_gate.details['coverage_percentage']}%")
    print(f"Status: {coverage_gate.status.value}")
    """)

    print("\n" + "=" * 80 + "\n")


async def example_coverage_report():
    """
    Example 4: Understanding the coverage analysis report.
    """
    print("=" * 80)
    print("EXAMPLE 4: Understanding Coverage Reports")
    print("=" * 80)

    print("""
Coverage Analysis Report Structure:
==================================

1. Overall Metrics:
   - coverage_percentage: 0-100% (80%+ passes)
   - coverage_level: insufficient/partial/good/comprehensive
   - passes_threshold: boolean
   - gaps_found: number of gaps

2. Gap Details (for each gap):
   - category: capability/constraint/quality_criteria/output_requirement
   - missing_aspect: what requirement is not being tested
   - severity: critical/high/medium/low
   - current_coverage: what's currently tested (if anything)
   - recommended_addition: specific suggestion to fix
   - example_test_case: example case that would catch this

3. Well-Covered Aspects:
   - List of requirements that ARE properly tested

4. Improvement Priority:
   - Ordered list of what to fix first (by severity)

5. Iteration History (if auto-improved):
   - iteration: 1, 2, 3
   - coverage_pct: coverage at each iteration
   - gaps_found: how many gaps remaining
   - improvements_generated: number of fixes applied
   - top_improvements: what was changed

Example Report:
--------------
{
  "coverage_percentage": 75.5,
  "coverage_level": "good",
  "passes_threshold": false,
  "gaps_found": 4,
  "gaps": [
    {
      "category": "constraint",
      "missing_aspect": "Never share customer data",
      "severity": "critical",
      "current_coverage": "Not tested as auto-fail",
      "recommended_addition": "Add auto-fail: data_leak",
      "example_test_case": "Response includes customer email/phone"
    },
    {
      "category": "capability",
      "missing_aspect": "Track orders using order IDs",
      "severity": "high",
      "current_coverage": "Not explicitly evaluated",
      "recommended_addition": "Add dimension: Order Tracking Accuracy",
      "example_test_case": "Given order ID, verifies tracking info"
    }
  ],
  "well_covered": [
    "Professional tone evaluation",
    "Response clarity checking",
    "JSON format validation"
  ],
  "iteration_history": [
    {"iteration": 1, "coverage_pct": 55.0, "gaps_found": 8},
    {"iteration": 2, "coverage_pct": 75.5, "gaps_found": 4}
  ]
}

Interpretation:
--------------
- Started at 55% coverage (insufficient)
- After 1 iteration: 75.5% coverage (good, but below 80% threshold)
- 4 gaps remain (1 critical, 1 high severity)
- Critical constraint "never share data" must be addressed before deployment
- Overall: Would fail validation gate, needs one more iteration
""")

    print("\n" + "=" * 80 + "\n")


async def main():
    """Run all examples"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "COVERAGE VALIDATION EXAMPLES" + " " * 30 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")

    await example_basic_coverage_check()
    await example_auto_improvement()
    await example_integrated_pipeline()
    await example_coverage_report()

    print("\n" + "=" * 80)
    print("KEY BENEFITS OF COVERAGE VALIDATION")
    print("=" * 80)
    print("""
1. Comprehensive Coverage:
   - Ensures eval prompt tests ALL capabilities mentioned in system prompt
   - Validates ALL constraints are enforced as auto-fails
   - Checks ALL quality criteria are evaluated

2. Iterative Improvement:
   - Automatically identifies gaps in coverage
   - Generates specific, actionable improvements
   - Iteratively refines until 80% threshold met (up to 3 iterations)

3. Early Detection:
   - Runs as Gate 3 (before expensive ground truth/reliability tests)
   - Catches missing coverage before deployment
   - Saves cost by fixing issues early

4. Detailed Reporting:
   - Shows exactly what's covered and what's missing
   - Prioritizes gaps by severity (critical → low)
   - Provides specific recommendations for each gap

5. Flexible Usage:
   - Can run standalone for analysis
   - Integrated into validated pipeline
   - Optional auto-improvement
   - Configurable threshold and iterations

6. Production Ready:
   - Handles errors gracefully (falls back to heuristics if LLM fails)
   - Tracks costs
   - Provides detailed audit trail
   - Non-blocking warnings for minor issues
    """)

    print("\n" + "=" * 80)
    print("USAGE IN YOUR CODE")
    print("=" * 80)
    print("""
# Option 1: Standalone coverage check
from eval_coverage_validator import validate_eval_coverage

result = await validate_eval_coverage(
    system_prompt=your_system_prompt,
    eval_prompt=your_eval_prompt,
    llm_client=your_llm_client,
    auto_improve=True,
    max_iterations=3
)

if result['final_coverage']['passes_threshold']:
    print("✓ Coverage validated!")
else:
    print(f"✗ Coverage: {result['final_coverage']['percentage']}%")
    print(f"Gaps: {result['gaps']}")

# Option 2: Integrated pipeline (automatically enabled)
from validated_eval_pipeline import ValidatedEvalPipeline

pipeline = ValidatedEvalPipeline(
    project_id="my_project",
    enable_coverage_validation=True,  # Default: True
    max_coverage_iterations=3
)

eval_prompt, validation = await pipeline.generate_and_validate(
    system_prompt=system_prompt,
    use_case="...",
    requirements=[...],
    provider="openai",
    api_key="...",
    model_name="gpt-4o"
)

# Check coverage gate result
coverage_gate = next(g for g in validation.gates if g.gate_name == "coverage_alignment")
print(f"Coverage: {coverage_gate.score}%")
print(f"Status: {coverage_gate.status.value}")

if validation.can_deploy:
    # Deploy eval_prompt (it's already been coverage-validated and improved!)
    pass
    """)

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
