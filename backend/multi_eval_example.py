"""
Example Usage of Multi-Evaluator System

This demonstrates how to use the new multi-evaluator architecture.
"""

import asyncio
import logging
from llm_client_v2 import EnhancedLLMClient
from multi_eval_integration import (
    MultiEvalIntegration,
    quick_evaluate,
    convert_verdict_to_legacy_format
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_1_basic_usage():
    """Example 1: Basic usage with auto-detection"""

    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage with Auto-Detection")
    print("="*70)

    # Initialize LLM client
    client = EnhancedLLMClient(
        provider="anthropic",
        api_key="your-api-key-here"  # Replace with actual key
    )

    # Define your system
    system_prompt = """You are a customer service assistant.
Respond professionally and helpfully to customer inquiries.
Provide accurate information and be concise."""

    requirements = """
- Must be professional and courteous
- Must provide accurate information
- Keep responses under 200 words
- Must address the customer's question directly
"""

    use_case = "Customer Service Automation"

    # Example evaluation
    input_data = "How do I reset my password?"

    output = """To reset your password:
1. Go to the login page
2. Click "Forgot Password"
3. Enter your email address
4. Check your email for a reset link
5. Follow the link and create a new password

The reset link expires in 24 hours. If you don't receive the email,
check your spam folder or contact support at support@example.com."""

    # Run evaluation - system auto-detects what to evaluate!
    integration = MultiEvalIntegration(client)

    result = await integration.evaluate_with_multi_system(
        system_prompt=system_prompt,
        requirements=requirements,
        use_case=use_case,
        input_data=input_data,
        output=output,
        provider="anthropic"
    )

    # Display results
    print(f"\nVERDICT: {result.verdict}")
    print(f"SCORE: {result.score:.2f}/5.0")
    print(f"REASON: {result.reason}")
    print(f"\nDIMENSION SCORES:")
    for eval_result in result.individual_evaluations:
        print(f"  - {eval_result.dimension_name}: {eval_result.score:.1f}/5.0")

    print(f"\nPERFORMANCE:")
    print(f"  - Latency: {result.total_latency_ms}ms")
    print(f"  - Tokens: {result.total_tokens_used}")


async def example_2_structured_requirements():
    """Example 2: Using structured requirements for better detection"""

    print("\n" + "="*70)
    print("EXAMPLE 2: Structured Requirements")
    print("="*70)

    client = EnhancedLLMClient(
        provider="anthropic",
        api_key="your-api-key-here"
    )

    system_prompt = "You are a JSON API that analyzes sentiment."

    # Structured requirements help with better auto-detection
    structured_requirements = {
        "must_do": [
            "Return valid JSON only",
            "Include sentiment field (positive/negative/neutral)",
            "Include confidence score (0-1)",
            "Provide reasoning for the sentiment"
        ],
        "must_not_do": [
            "Include any non-JSON text",
            "Hallucinate information not in the input"
        ],
        "output_format": '''{
            "sentiment": "positive|negative|neutral",
            "confidence": 0.95,
            "reasoning": "Brief explanation"
        }''',
        "tone": "technical and precise"
    }

    input_data = "I love this product! Best purchase ever."

    output = '''{
    "sentiment": "positive",
    "confidence": 0.98,
    "reasoning": "Strong positive indicators: 'love', 'best purchase ever'"
}'''

    integration = MultiEvalIntegration(client)

    result = await integration.evaluate_with_multi_system(
        system_prompt=system_prompt,
        requirements="Analyze sentiment accurately and return JSON",
        use_case="Sentiment Analysis API",
        input_data=input_data,
        output=output,
        provider="anthropic",
        structured_requirements=structured_requirements
    )

    # The system automatically detects:
    # - SchemaValidator (because of output_format)
    # - SafetyChecker (because of must_not_do)
    # - AccuracyEvaluator (always important)
    # - CompletenessChecker (because of must_do)
    # - ToneEvaluator (because of tone requirement)

    print(integration.format_verdict_for_display(result))


async def example_3_quick_evaluate():
    """Example 3: Quick evaluation helper"""

    print("\n" + "="*70)
    print("EXAMPLE 3: Quick Evaluate")
    print("="*70)

    client = EnhancedLLMClient(
        provider="anthropic",
        api_key="your-api-key-here"
    )

    # One-liner evaluation
    result = await quick_evaluate(
        llm_client=client,
        system_prompt="You are a helpful math tutor.",
        requirements="Explain concepts clearly and check accuracy",
        use_case="Math tutoring",
        input_data="What is 2 + 2?",
        output="2 + 2 equals 4. This is basic addition."
    )

    print(f"Quick result: {result.verdict} ({result.score:.1f}/5.0)")


async def example_4_comparison():
    """Example 4: Compare multi-evaluator vs monolithic"""

    print("\n" + "="*70)
    print("EXAMPLE 4: Comparison with Monolithic Evaluator")
    print("="*70)

    client = EnhancedLLMClient(
        provider="anthropic",
        api_key="your-api-key-here"
    )

    integration = MultiEvalIntegration(client)

    # You can compare both approaches
    # (requires implementing a monolithic eval function)

    comparison = await integration.compare_approaches(
        system_prompt="You are a helpful assistant.",
        requirements="Be accurate and helpful",
        use_case="General assistance",
        input_data="What's the capital of France?",
        output="The capital of France is Paris.",
        provider="anthropic",
        # monolithic_eval_func=your_monolithic_function  # Optional
    )

    print(f"\nMULTI-EVALUATOR:")
    print(f"  Verdict: {comparison.multi_eval_result.verdict}")
    print(f"  Cost: ${comparison.multi_eval_cost:.4f}")
    print(f"  Latency: {comparison.multi_eval_latency_ms}ms")

    if comparison.monolithic_result:
        print(f"\nMONOLITHIC:")
        print(f"  Verdict: {comparison.monolithic_result.get('verdict')}")
        print(f"  Cost: ${comparison.monolithic_cost:.4f}")
        print(f"  Latency: {comparison.monolithic_latency_ms}ms")

        print(f"\nIMPROVEMENTS:")
        print(f"  Cost savings: {comparison.cost_savings_pct:.1f}%")
        print(f"  Latency improvement: {comparison.latency_improvement_pct:.1f}%")
        print(f"  Verdict agreement: {comparison.verdict_agreement}")


async def example_5_legacy_format():
    """Example 5: Convert to legacy format for compatibility"""

    print("\n" + "="*70)
    print("EXAMPLE 5: Legacy Format Conversion")
    print("="*70)

    client = EnhancedLLMClient(
        provider="anthropic",
        api_key="your-api-key-here"
    )

    result = await quick_evaluate(
        llm_client=client,
        system_prompt="You are a code reviewer.",
        requirements="Check for bugs and best practices",
        use_case="Code review",
        input_data="def add(a, b): return a + b",
        output="This function looks good. It's simple and correct."
    )

    # Convert to legacy format for backward compatibility
    legacy_format = convert_verdict_to_legacy_format(result)

    print("Legacy format (compatible with old system):")
    import json
    print(json.dumps(legacy_format, indent=2))


def main():
    """Run all examples"""

    print("\n")
    print("*" * 70)
    print("MULTI-EVALUATOR SYSTEM EXAMPLES")
    print("*" * 70)

    # Note: Update with your actual API key before running
    print("\n⚠️  Remember to set your API key in the examples!")
    print("    Replace 'your-api-key-here' with your actual key\n")

    # Uncomment to run examples:
    # asyncio.run(example_1_basic_usage())
    # asyncio.run(example_2_structured_requirements())
    # asyncio.run(example_3_quick_evaluate())
    # asyncio.run(example_4_comparison())
    # asyncio.run(example_5_legacy_format())


if __name__ == "__main__":
    main()
