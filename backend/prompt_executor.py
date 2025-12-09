"""
Prompt Executor Service
Executes prompts against test datasets and evaluates outputs using eval prompts.
"""

import asyncio
import time
import re
import json
import random
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone
import openai
import anthropic
import google.generativeai as genai

from models import (
    ExecutionResult, TestRunSummary, TestRun, TestCase
)


# Retry configuration
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 30.0  # seconds

# Rate limit error patterns
RATE_LIMIT_ERRORS = [
    "rate_limit",
    "rate limit",
    "too many requests",
    "quota exceeded",
    "429",
    "overloaded",
    "capacity"
]


def is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception is a rate limit error."""
    error_str = str(error).lower()
    return any(pattern in error_str for pattern in RATE_LIMIT_ERRORS)


async def retry_with_backoff(
    func,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
    max_delay: float = MAX_DELAY
):
    """
    Retry an async function with exponential backoff for rate limit errors.
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_exception = e

            # Only retry on rate limit errors
            if not is_rate_limit_error(e):
                raise e

            if attempt == max_retries:
                raise e

            # Calculate delay with exponential backoff and jitter
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
            await asyncio.sleep(delay)

    raise last_exception


# Default models per provider
DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "claude": "claude-3-5-sonnet-20241022",
    "gemini": "gemini-2.0-flash-exp"
}

# Pricing per 1M tokens (input/output) - approximate
TOKEN_PRICING = {
    "openai": {"gpt-4o": {"input": 2.5, "output": 10.0}},
    "claude": {"claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0}},
    "gemini": {"gemini-2.0-flash-exp": {"input": 0.075, "output": 0.30}}
}


class PromptExecutor:
    """
    Service for executing prompts against test data and evaluating outputs.
    Supports batching, concurrent execution, and progress tracking.
    """

    def __init__(self):
        self.active_runs: Dict[str, TestRun] = {}
        self.cancelled_runs: set = set()  # Track cancelled run IDs

    async def execute_prompt(
        self,
        system_prompt: str,
        user_input: str,
        provider: str,
        api_key: str,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a system prompt with user input and return the response.
        Returns dict with: output, latency_ms, tokens_used, error
        """
        model = model_name or DEFAULT_MODELS.get(provider, "gpt-4o")
        start_time = time.time()

        try:
            if provider == "openai":
                return await self._execute_openai(system_prompt, user_input, api_key, model, start_time)
            elif provider == "claude":
                return await self._execute_claude(system_prompt, user_input, api_key, model, start_time)
            elif provider == "gemini":
                return await self._execute_gemini(system_prompt, user_input, api_key, model, start_time)
            else:
                return {
                    "output": "",
                    "latency_ms": 0,
                    "tokens_used": 0,
                    "error": f"Unsupported provider: {provider}"
                }
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "output": "",
                "latency_ms": latency_ms,
                "tokens_used": 0,
                "error": str(e)
            }

    async def _execute_openai(
        self, system_prompt: str, user_input: str, api_key: str, model: str, start_time: float
    ) -> Dict[str, Any]:
        client = openai.AsyncOpenAI(api_key=api_key)

        async def make_request():
            return await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7,
                max_tokens=2000
            )

        response = await retry_with_backoff(make_request)
        latency_ms = int((time.time() - start_time) * 1000)
        output = response.choices[0].message.content or ""
        tokens_used = response.usage.total_tokens if response.usage else 0
        return {
            "output": output,
            "latency_ms": latency_ms,
            "tokens_used": tokens_used,
            "error": None
        }

    async def _execute_claude(
        self, system_prompt: str, user_input: str, api_key: str, model: str, start_time: float
    ) -> Dict[str, Any]:
        client = anthropic.AsyncAnthropic(api_key=api_key)

        async def make_request():
            return await client.messages.create(
                model=model,
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_input}]
            )

        response = await retry_with_backoff(make_request)
        latency_ms = int((time.time() - start_time) * 1000)
        output = response.content[0].text if response.content else ""
        tokens_used = response.usage.input_tokens + response.usage.output_tokens if response.usage else 0
        return {
            "output": output,
            "latency_ms": latency_ms,
            "tokens_used": tokens_used,
            "error": None
        }

    async def _execute_gemini(
        self, system_prompt: str, user_input: str, api_key: str, model: str, start_time: float
    ) -> Dict[str, Any]:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_prompt
        )

        async def make_request():
            return await asyncio.to_thread(
                gemini_model.generate_content,
                user_input
            )

        response = await retry_with_backoff(make_request)
        latency_ms = int((time.time() - start_time) * 1000)
        output = response.text if response.text else ""
        # Gemini doesn't always provide token counts
        tokens_used = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            tokens_used = getattr(response.usage_metadata, 'total_token_count', 0)
        return {
            "output": output,
            "latency_ms": latency_ms,
            "tokens_used": tokens_used,
            "error": None
        }

    async def evaluate_output(
        self,
        eval_prompt: str,
        original_input: str,
        prompt_output: str,
        provider: str,
        api_key: str,
        model_name: Optional[str] = None,
        test_context: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Use the eval prompt to evaluate an output.
        Returns dict with: score (1-5), feedback, error

        Args:
            test_context: Optional dict with category, test_focus, difficulty
        """
        model = model_name or DEFAULT_MODELS.get(provider, "gpt-4o")

        # Build context section if provided
        context_section = ""
        if test_context:
            category = test_context.get("category", "unknown")
            test_focus = test_context.get("test_focus", "")
            difficulty = test_context.get("difficulty", "medium")

            context_section = f"""
## Test Case Context:
- **Category**: {category} {"(expected to succeed)" if category == "positive" else "(challenging scenario)" if category in ["edge_case", "adversarial"] else ""}
- **Test Focus**: {test_focus}
- **Difficulty**: {difficulty}

Consider this context when evaluating. For edge cases and adversarial inputs, appropriate handling (graceful failure, safety responses) should still score well.
"""

        # Construct the evaluation request
        eval_user_message = f"""Please evaluate the following LLM output based on the evaluation criteria.
{context_section}
## Original User Input:
{original_input}

## LLM Output to Evaluate:
{prompt_output}

## Your Task:
Based on the evaluation criteria provided in your system prompt, score this output on a scale of 1-5 and provide detailed feedback.

Respond in the following JSON format:
{{
    "score": <1-5>,
    "feedback": "<detailed explanation of the score>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"],
    "suggestions": ["<suggestion 1>", "<suggestion 2>"]
}}
"""

        try:
            if provider == "openai":
                result = await self._eval_openai(eval_prompt, eval_user_message, api_key, model)
            elif provider == "claude":
                result = await self._eval_claude(eval_prompt, eval_user_message, api_key, model)
            elif provider == "gemini":
                result = await self._eval_gemini(eval_prompt, eval_user_message, api_key, model)
            else:
                return {"score": 0, "feedback": f"Unsupported provider: {provider}", "error": True}

            return self._parse_eval_response(result)
        except Exception as e:
            return {"score": 0, "feedback": str(e), "error": str(e)}

    async def _eval_openai(
        self, eval_prompt: str, user_message: str, api_key: str, model: str
    ) -> str:
        client = openai.AsyncOpenAI(api_key=api_key)

        async def make_request():
            return await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": eval_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,  # Lower temperature for consistent evaluation
                max_tokens=1000
            )

        response = await retry_with_backoff(make_request)
        return response.choices[0].message.content or ""

    async def _eval_claude(
        self, eval_prompt: str, user_message: str, api_key: str, model: str
    ) -> str:
        client = anthropic.AsyncAnthropic(api_key=api_key)

        async def make_request():
            return await client.messages.create(
                model=model,
                max_tokens=1000,
                system=eval_prompt,
                messages=[{"role": "user", "content": user_message}]
            )

        response = await retry_with_backoff(make_request)
        return response.content[0].text if response.content else ""

    async def _eval_gemini(
        self, eval_prompt: str, user_message: str, api_key: str, model: str
    ) -> str:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=eval_prompt
        )

        async def make_request():
            return await asyncio.to_thread(
                gemini_model.generate_content,
                user_message
            )

        response = await retry_with_backoff(make_request)
        return response.text if response.text else ""

    def _parse_eval_response(self, response: str) -> Dict[str, Any]:
        """Parse the evaluation response to extract score and feedback."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                score = float(data.get("score", 3))
                # Clamp score to 1-5
                score = max(1, min(5, score))

                feedback_parts = []
                if data.get("feedback"):
                    feedback_parts.append(data["feedback"])
                if data.get("strengths"):
                    feedback_parts.append(f"Strengths: {', '.join(data['strengths'])}")
                if data.get("weaknesses"):
                    feedback_parts.append(f"Weaknesses: {', '.join(data['weaknesses'])}")
                if data.get("suggestions"):
                    feedback_parts.append(f"Suggestions: {', '.join(data['suggestions'])}")

                feedback = "\n".join(feedback_parts) if feedback_parts else "No detailed feedback provided."

                return {"score": score, "feedback": feedback, "error": None}
            else:
                # Fallback: try to extract score from text
                score_match = re.search(r'score[:\s]*(\d)', response, re.IGNORECASE)
                score = float(score_match.group(1)) if score_match else 3.0
                return {"score": score, "feedback": response, "error": None}
        except Exception as e:
            return {"score": 3.0, "feedback": f"Failed to parse evaluation: {response[:500]}", "error": str(e)}

    def _substitute_template_variables(self, system_prompt: str, test_input: Dict[str, Any]) -> str:
        """
        Substitute template variables in system prompt with test data values.
        Supports both {{variable}} and {variable} formats.
        """
        result = system_prompt

        for key, value in test_input.items():
            if key == "input":
                # Common variable names for user input
                for var_name in ["input", "user_input", "query", "user_query", "question", "message"]:
                    result = result.replace(f"{{{{{var_name}}}}}", str(value))
                    result = result.replace(f"{{{var_name}}}", str(value))
            else:
                # Replace exact variable name
                result = result.replace(f"{{{{{key}}}}}", str(value))
                result = result.replace(f"{{{key}}}", str(value))

        return result

    async def execute_single_test(
        self,
        system_prompt: str,
        test_input: Dict[str, Any],
        eval_prompt: Optional[str],
        provider: str,
        api_key: str,
        model_name: Optional[str] = None,
        pass_threshold: float = 3.5
    ) -> ExecutionResult:
        """
        Execute a single test case and optionally evaluate it.
        """
        # Extract user input from test data
        user_input = test_input.get("input", str(test_input))

        # Substitute template variables in system prompt
        processed_prompt = self._substitute_template_variables(system_prompt, test_input)

        # Execute the prompt
        exec_result = await self.execute_prompt(
            processed_prompt, user_input, provider, api_key, model_name
        )

        # Evaluate if eval prompt provided
        if eval_prompt and not exec_result.get("error"):
            # Build test context from test input data
            test_context = None
            if any(k in test_input for k in ["category", "test_focus", "difficulty"]):
                test_context = {
                    "category": test_input.get("category", "unknown"),
                    "test_focus": test_input.get("test_focus", ""),
                    "difficulty": test_input.get("difficulty", "medium")
                }

            eval_result = await self.evaluate_output(
                eval_prompt, user_input, exec_result["output"],
                provider, api_key, model_name,
                test_context=test_context
            )
            score = eval_result["score"]
            feedback = eval_result["feedback"]
        else:
            score = 0.0
            feedback = exec_result.get("error") or "No evaluation performed"

        return ExecutionResult(
            dataset_item_index=0,
            input_data=test_input,
            prompt_output=exec_result["output"],
            eval_score=score,
            eval_feedback=feedback,
            passed=score >= pass_threshold,
            latency_ms=exec_result["latency_ms"],
            tokens_used=exec_result.get("tokens_used"),
            error=exec_result.get("error")
        )

    async def execute_batch(
        self,
        system_prompt: str,
        test_items: List[Dict[str, Any]],
        eval_prompt: str,
        provider: str,
        api_key: str,
        model_name: Optional[str] = None,
        pass_threshold: float = 3.5,
        max_concurrent: int = 3,
        start_index: int = 0,
        progress_callback: Optional[Callable[[int, int, ExecutionResult], None]] = None
    ) -> List[ExecutionResult]:
        """
        Execute a batch of test items with controlled concurrency.
        """
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_item(index: int, item: Dict[str, Any]) -> ExecutionResult:
            async with semaphore:
                result = await self.execute_single_test(
                    system_prompt, item, eval_prompt,
                    provider, api_key, model_name, pass_threshold
                )
                result.dataset_item_index = start_index + index

                if progress_callback:
                    progress_callback(index + 1, len(test_items), result)

                return result

        # Create tasks for all items
        tasks = [
            process_item(i, item)
            for i, item in enumerate(test_items)
        ]

        # Execute with controlled concurrency
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ExecutionResult(
                    dataset_item_index=start_index + i,
                    input_data=test_items[i],
                    prompt_output="",
                    eval_score=0,
                    eval_feedback="",
                    passed=False,
                    latency_ms=0,
                    error=str(result)
                ))
            else:
                processed_results.append(result)

        return processed_results

    def calculate_summary(
        self,
        results: List[ExecutionResult],
        total_items: int,
        provider: str,
        model_name: Optional[str] = None
    ) -> TestRunSummary:
        """
        Calculate summary statistics from execution results.
        """
        completed = [r for r in results if r.error is None]
        errors = [r for r in results if r.error is not None]
        passed = [r for r in completed if r.passed]
        failed = [r for r in completed if not r.passed]

        scores = [r.eval_score for r in completed if r.eval_score > 0]
        latencies = [r.latency_ms for r in completed]
        tokens = [r.tokens_used or 0 for r in results]

        # Score distribution
        distribution = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        for score in scores:
            bucket = str(min(5, max(1, round(score))))
            distribution[bucket] = distribution.get(bucket, 0) + 1

        # Estimate cost
        total_tokens = sum(tokens)
        model = model_name or DEFAULT_MODELS.get(provider, "gpt-4o")
        pricing = TOKEN_PRICING.get(provider, {}).get(model, {"input": 2.5, "output": 10.0})
        # Rough estimate: 60% input, 40% output
        estimated_cost = (total_tokens * 0.6 * pricing["input"] + total_tokens * 0.4 * pricing["output"]) / 1_000_000

        return TestRunSummary(
            total_items=total_items,
            completed_items=len(completed),
            passed_items=len(passed),
            failed_items=len(failed),
            error_items=len(errors),
            pass_rate=round((len(passed) / len(completed) * 100) if completed else 0, 1),
            avg_score=round(sum(scores) / len(scores), 2) if scores else 0,
            min_score=min(scores) if scores else 0,
            max_score=max(scores) if scores else 0,
            score_distribution=distribution,
            avg_latency_ms=round(sum(latencies) / len(latencies), 0) if latencies else 0,
            total_tokens=total_tokens,
            estimated_cost=round(estimated_cost, 4)
        )

    async def run_full_test(
        self,
        test_run: TestRun,
        test_items: List[Dict[str, Any]],
        api_key: str,
        progress_callback: Optional[Callable[[TestRun], None]] = None
    ) -> TestRun:
        """
        Execute a full test run with batching and progress tracking.
        Supports cancellation between batches.
        """
        test_run.status = "running"
        test_run.started_at = datetime.now(timezone.utc)
        self.active_runs[test_run.id] = test_run

        # Remove from cancelled set if it was previously cancelled
        self.cancelled_runs.discard(test_run.id)

        if progress_callback:
            progress_callback(test_run)

        try:
            all_results = []
            batch_size = test_run.batch_size
            total_batches = (len(test_items) + batch_size - 1) // batch_size

            for batch_num in range(total_batches):
                # Check for cancellation before starting each batch
                if test_run.id in self.cancelled_runs:
                    test_run.status = "cancelled"
                    test_run.completed_at = datetime.now(timezone.utc)
                    break

                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(test_items))
                batch_items = test_items[start_idx:end_idx]

                # Execute batch
                batch_results = await self.execute_batch(
                    system_prompt=test_run.prompt_text,
                    test_items=batch_items,
                    eval_prompt=test_run.eval_prompt_text,
                    provider=test_run.llm_provider,
                    api_key=api_key,
                    model_name=test_run.model_name,
                    pass_threshold=test_run.pass_threshold,
                    max_concurrent=test_run.max_concurrent,
                    start_index=start_idx
                )

                # Check for cancellation after batch completes
                if test_run.id in self.cancelled_runs:
                    test_run.status = "cancelled"
                    test_run.completed_at = datetime.now(timezone.utc)
                    # Still save results from completed batches
                    all_results.extend(batch_results)
                    test_run.results = all_results
                    test_run.summary = self.calculate_summary(
                        all_results, len(test_items),
                        test_run.llm_provider, test_run.model_name
                    )
                    break

                all_results.extend(batch_results)
                test_run.results = all_results

                # Update partial summary
                test_run.summary = self.calculate_summary(
                    all_results, len(test_items),
                    test_run.llm_provider, test_run.model_name
                )

                if progress_callback:
                    progress_callback(test_run)

                # Small delay between batches to avoid rate limiting
                if batch_num < total_batches - 1:
                    await asyncio.sleep(0.5)

            # Only set to completed if not cancelled
            if test_run.status == "running":
                test_run.status = "completed"
                test_run.completed_at = datetime.now(timezone.utc)

        except Exception as e:
            test_run.status = "failed"
            test_run.error_message = str(e)
            test_run.completed_at = datetime.now(timezone.utc)

        if progress_callback:
            progress_callback(test_run)

        # Clean up
        if test_run.id in self.active_runs:
            del self.active_runs[test_run.id]
        self.cancelled_runs.discard(test_run.id)

        return test_run

    def get_active_run(self, run_id: str) -> Optional[TestRun]:
        """Get an active test run by ID."""
        return self.active_runs.get(run_id)

    def cancel_run(self, run_id: str) -> bool:
        """
        Cancel an active test run.
        The cancellation is checked between batches, so in-progress batches will complete.
        """
        if run_id in self.active_runs:
            # Add to cancelled set - will be checked during batch processing
            self.cancelled_runs.add(run_id)
            return True
        return False

    def is_cancelled(self, run_id: str) -> bool:
        """Check if a run has been cancelled."""
        return run_id in self.cancelled_runs


# Singleton instance
prompt_executor = PromptExecutor()
