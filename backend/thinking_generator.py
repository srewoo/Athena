"""
Thinking Model Integration for Athena
Adds self-reflective, reasoning-enhanced generation for prompts, eval prompts, and datasets.

Uses thinking models (o1, o3, extended thinking) to:
1. Generate with reasoning
2. Self-critique and improve
3. Verify against requirements
4. Iterate until high quality
"""
from typing import Dict, Any, List, Optional
from llm_client import LLMClient, THINKING_MODELS, THINKING_TIMEOUT
from models import Requirements
import json


class ThinkingGenerator:
    """
    Enhanced generator that uses thinking models for self-reflective generation.
    """

    def __init__(self, client: LLMClient):
        self.client = client
        self.is_thinking_model = client.is_thinking_model()

    async def generate_prompt_with_thinking(
        self,
        requirements: Requirements,
        current_prompt: Optional[str] = None,
        feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate or refine a system prompt using thinking model.

        Args:
            requirements: User requirements for the prompt
            current_prompt: Existing prompt to refine (optional)
            feedback: User feedback on current prompt (optional)

        Returns:
            Dictionary with improved_prompt, reasoning, changes_made
        """
        if not self.is_thinking_model:
            raise ValueError("This method requires a thinking model (o1, o3, etc.)")

        # Build the generation task with explicit reasoning request
        task = self._build_prompt_generation_task(requirements, current_prompt, feedback)

        # Generate with thinking
        response = await self.client.generate(
            prompt=task,
            temperature=1.0,  # Thinking models work best at temperature 1
            max_tokens=16000  # Allow long reasoning chains
        )

        # Parse the response
        return self._parse_prompt_generation_response(response.content, requirements)

    async def generate_eval_prompt_with_thinking(
        self,
        system_prompt: str,
        requirements: Requirements,
        additional_scenarios: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate evaluation prompt using thinking model.

        The model will:
        1. Analyze the system prompt deeply
        2. Identify key behaviors to test
        3. Design comprehensive test scenarios
        4. Create calibration examples
        5. Self-verify the eval prompt quality

        Returns:
            Dictionary with eval_prompt, rationale, test_scenarios, reasoning
        """
        if not self.is_thinking_model:
            raise ValueError("This method requires a thinking model")

        task = self._build_eval_generation_task(system_prompt, requirements, additional_scenarios)

        response = await self.client.generate(
            prompt=task,
            temperature=1.0,
            max_tokens=16000
        )

        return self._parse_eval_generation_response(response.content)

    async def generate_dataset_with_thinking(
        self,
        system_prompt: str,
        requirements: Requirements,
        sample_count: int = 100,
        distribution: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Generate test dataset using thinking model.

        The model will:
        1. Understand what inputs the system should handle
        2. Reason about edge cases and adversarial scenarios
        3. Generate realistic, diverse test cases
        4. Verify coverage of all requirements
        5. Self-critique and improve diversity

        Returns:
            Dictionary with test_cases, metadata, reasoning
        """
        if not self.is_thinking_model:
            raise ValueError("This method requires a thinking model")

        task = self._build_dataset_generation_task(
            system_prompt, requirements, sample_count, distribution
        )

        response = await self.client.generate(
            prompt=task,
            temperature=1.0,
            max_tokens=32000  # Datasets can be large
        )

        return self._parse_dataset_generation_response(response.content)

    def _build_prompt_generation_task(
        self,
        requirements: Requirements,
        current_prompt: Optional[str],
        feedback: Optional[str]
    ) -> str:
        """Build task prompt for thinking model to generate/refine system prompt"""

        context = f"""You are an expert prompt engineer. Your task is to {'refine the current' if current_prompt else 'create a new'} system prompt.

## Requirements
**Use Case**: {requirements.use_case}
**Target Provider**: {requirements.target_provider}

**Key Requirements**:
{chr(10).join(f"- {req}" for req in requirements.key_requirements)}

**Expected Behavior**: {requirements.expected_behavior or 'Follow requirements'}

{f'''**Current Prompt**:
```
{current_prompt}
```
''' if current_prompt else ''}

{f'**User Feedback**: {feedback}' if feedback else ''}

## Your Task

Think step-by-step:

1. **Analyze Requirements**: Deeply understand what the system needs to do
2. **Identify Key Challenges**: What's hard about this task? What edge cases exist?
3. **Design Strategy**: What prompt structure would work best for {requirements.target_provider}?
4. **Draft Prompt**: Write a comprehensive system prompt
5. **Self-Critique**: Review your prompt - does it meet ALL requirements? Any gaps?
6. **Refine**: Improve based on your critique
7. **Verify**: Final check against all {len(requirements.key_requirements)} requirements

## Output Format

Return a JSON object:
```json
{{
  "improved_prompt": "The final system prompt...",
  "reasoning": "Your step-by-step thinking process...",
  "changes_made": ["Change 1", "Change 2", ...],
  "confidence": 0.95,
  "potential_improvements": ["Future enhancement 1", ...]
}}
```"""

        return context

    def _build_eval_generation_task(
        self,
        system_prompt: str,
        requirements: Requirements,
        additional_scenarios: Optional[List[str]]
    ) -> str:
        """Build task for generating eval prompt with thinking"""

        return f"""You are an expert in evaluation prompt design. Create a rigorous evaluation prompt to test this system prompt.

## System Prompt to Evaluate
```
{system_prompt}
```

## Use Case & Requirements
**Use Case**: {requirements.use_case}
{chr(10).join(f"- {req}" for req in requirements.key_requirements)}

{f"**Additional Scenarios**: {chr(10).join(f'- {s}' for s in additional_scenarios)}" if additional_scenarios else ""}

## Your Task

Think deeply about:

1. **What does this system actually do?** What inputs and outputs?
2. **What could go wrong?** Failure modes, edge cases, hallucination risks
3. **How to measure quality?** What makes a response "excellent" vs "poor"?
4. **Test scenario design**: What scenarios would truly stress-test this system?
5. **Calibration examples**: Create examples showing the rating scale
6. **Self-verification**: Is this eval prompt comprehensive and fair?

## Output Format

Return JSON:
```json
{{
  "eval_prompt": "Complete evaluation prompt with {{{{input}}}} and {{{{output}}}} placeholders...",
  "rationale": "Why this eval strategy works...",
  "test_scenarios": ["Scenario 1", "Scenario 2", ...],
  "calibration_examples": [
    {{"input": "...", "output": "...", "expected_score": 5, "reasoning": "..."}},
    ...
  ],
  "reasoning": "Your thinking process...",
  "coverage_analysis": "What this eval tests and what it might miss..."
}}
```"""

    def _build_dataset_generation_task(
        self,
        system_prompt: str,
        requirements: Requirements,
        sample_count: int,
        distribution: Optional[Dict[str, int]]
    ) -> str:
        """Build task for dataset generation with thinking"""

        dist = distribution or {
            "positive": int(sample_count * 0.4),
            "edge_case": int(sample_count * 0.3),
            "negative": int(sample_count * 0.2),
            "adversarial": int(sample_count * 0.1)
        }

        return f"""You are an expert test data generator. Create realistic test cases for this system.

## System Prompt
```
{system_prompt}
```

## Requirements
**Use Case**: {requirements.use_case}
{chr(10).join(f"- {req}" for req in requirements.key_requirements)}

## Distribution Needed
{chr(10).join(f"- {cat}: {count} cases" for cat, count in dist.items())}
**Total**: {sample_count} test cases

## Critical Thinking Required

1. **Understand the Input Type**: What INPUT does this system receive? (NOT what it outputs!)
2. **Realism**: Make inputs look like real-world data, not synthetic test descriptions
3. **Diversity**: Cover different subtypes, lengths, complexities
4. **Edge Cases**: Think of unusual but valid inputs
5. **Adversarial**: Design inputs that might trick the system
6. **Coverage**: Ensure all {len(requirements.key_requirements)} requirements are tested

## Output Format

Return JSON with test cases:
```json
{{
  "test_cases": [
    {{
      "input": "ACTUAL realistic input content (not a description!)",
      "category": "positive|edge_case|negative|adversarial",
      "test_focus": "Which requirement this tests",
      "difficulty": "easy|medium|hard"
    }},
    ...
  ],
  "reasoning": "Your thinking about test design...",
  "coverage_report": "How these cases cover all requirements...",
  "diversity_analysis": "Types of cases generated..."
}}
```

Generate exactly {sample_count} test cases with the specified distribution."""

    def _parse_prompt_generation_response(
        self,
        response: str,
        requirements: Requirements
    ) -> Dict[str, Any]:
        """Parse thinking model response for prompt generation"""
        try:
            from llm_client import parse_json_response
            data = parse_json_response(response)

            return {
                "improved_prompt": data.get("improved_prompt", ""),
                "changes_made": data.get("changes_made", []),
                "rationale": data.get("reasoning", ""),
                "confidence": data.get("confidence", 0.0),
                "potential_improvements": data.get("potential_improvements", []),
                "method": "thinking_model"
            }
        except Exception as e:
            # Fallback: extract prompt from response manually
            return {
                "improved_prompt": response,
                "changes_made": ["Generated with thinking model"],
                "rationale": "Thinking model generation",
                "confidence": 0.8,
                "potential_improvements": [],
                "method": "thinking_model_fallback"
            }

    def _parse_eval_generation_response(self, response: str) -> Dict[str, Any]:
        """Parse eval prompt generation response"""
        try:
            from llm_client import parse_json_response
            data = parse_json_response(response)

            return {
                "eval_prompt": data.get("eval_prompt", ""),
                "rationale": data.get("rationale", ""),
                "test_scenarios": data.get("test_scenarios", []),
                "calibration_examples": data.get("calibration_examples", []),
                "reasoning": data.get("reasoning", ""),
                "coverage_analysis": data.get("coverage_analysis", ""),
                "generation_method": "thinking_model"
            }
        except Exception as e:
            return {
                "eval_prompt": response,
                "rationale": "Generated with thinking model",
                "test_scenarios": [],
                "calibration_examples": [],
                "reasoning": "Thinking model reasoning",
                "coverage_analysis": "",
                "generation_method": "thinking_model_fallback"
            }

    def _parse_dataset_generation_response(self, response: str) -> Dict[str, Any]:
        """Parse dataset generation response"""
        try:
            from llm_client import parse_json_response
            data = parse_json_response(response)

            return {
                "test_cases": data.get("test_cases", []),
                "reasoning": data.get("reasoning", ""),
                "coverage_report": data.get("coverage_report", ""),
                "diversity_analysis": data.get("diversity_analysis", ""),
                "sample_count": len(data.get("test_cases", [])),
                "generation_method": "thinking_model"
            }
        except Exception as e:
            return {
                "test_cases": [],
                "reasoning": "Thinking model generation failed",
                "coverage_report": "",
                "diversity_analysis": "",
                "sample_count": 0,
                "generation_method": "thinking_model_error"
            }


def get_thinking_generator(
    provider: str,
    api_key: str,
    model_name: str
) -> ThinkingGenerator:
    """
    Factory function to create a thinking generator.

    Args:
        provider: "openai", "claude", or "gemini"
        api_key: API key
        model_name: Must be a thinking model (o1, o3, etc.)

    Returns:
        ThinkingGenerator instance

    Raises:
        ValueError: If model is not a thinking model
    """
    from llm_client import LLMClient, THINKING_TIMEOUT

    client = LLMClient(
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        timeout=THINKING_TIMEOUT
    )

    if not client.is_thinking_model():
        raise ValueError(
            f"Model '{model_name}' is not a thinking model. "
            f"Available thinking models: {THINKING_MODELS.get(client.provider, [])}"
        )

    return ThinkingGenerator(client)
