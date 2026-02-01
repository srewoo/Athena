"""
Parallel Eval Generator - Generate multiple evals in parallel with retry logic

Architecture:
- Each aspect gets its own isolated async task
- Each task has its own token budget
- Tasks run in parallel (asyncio.gather)
- Automatic retry on failure (3 attempts with exponential backoff)
- All features preserved (meta-eval, auto-refine)
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class AspectEvalResult:
    """Result for a single aspect evaluation"""
    aspect: str
    success: bool
    eval_prompt: Optional[str] = None
    meta_quality: Optional[float] = None
    passes_gate: Optional[bool] = None
    was_refined: Optional[bool] = None
    executive_summary: Optional[str] = None
    audit_scores: Optional[Dict[str, float]] = None
    logic_gaps: Optional[List[Dict[str, str]]] = None
    refinement_roadmap: Optional[List[str]] = None
    error: Optional[str] = None
    attempts: int = 1
    duration_ms: int = 0
    # Enhanced quality metrics
    quality_analysis: Optional[Dict[str, Any]] = None
    calibration_examples: Optional[List[Dict[str, Any]]] = None
    rubric_levels: Optional[List[Dict[str, str]]] = None
    evaluation_purpose: Optional[str] = None
    ai_system_context: Optional[Dict[str, str]] = None


async def generate_single_aspect_with_retry(
    aspect: str,
    system_prompt: str,
    use_case: str,
    llm_client: Any,
    provider: str,
    api_key: str,
    model_name: str,
    max_retries: int = 3
) -> AspectEvalResult:
    """
    Generate eval for a single aspect with retry logic.

    This function is ISOLATED - it has its own:
    - Token budget
    - Error handling
    - Retry logic

    It won't affect other parallel tasks if it fails.
    """

    start_time = time.time()
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Generating eval for '{aspect}' (attempt {attempt}/{max_retries})")

            # Import here to avoid circular dependencies
            from simple_eval_generator import generate_simple_eval_prompt
            from eval_prompt_parser import (
                analyze_eval_prompt_quality,
                extract_all_examples_from_prompt,
                extract_rubric_levels,
                extract_evaluation_purpose,
                extract_ai_system_context
            )

            # Generate eval prompt (single LLM call, focused)
            result = await generate_simple_eval_prompt(
                llm_client=llm_client,
                system_prompt=system_prompt,
                aspect=aspect,
                use_case=use_case,
                provider=provider,
                api_key=api_key,
                model=model_name
            )

            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)

            # Analyze the generated eval prompt quality
            eval_prompt = result["eval_prompt"]
            quality_analysis = analyze_eval_prompt_quality(eval_prompt)
            calibration_examples = extract_all_examples_from_prompt(eval_prompt)
            rubric_levels = extract_rubric_levels(eval_prompt)
            evaluation_purpose = extract_evaluation_purpose(eval_prompt)
            ai_system_context = extract_ai_system_context(eval_prompt)

            # Success!
            return AspectEvalResult(
                aspect=aspect,
                success=True,
                eval_prompt=eval_prompt,
                meta_quality=quality_analysis.get("quality_score", 8.0),  # Use analyzed quality
                passes_gate=quality_analysis.get("quality_score", 8.0) >= 8.5,
                was_refined=result.get("was_refined", False),
                executive_summary=result.get("reasoning", ""),
                audit_scores={},  # Simple generator doesn't have audit scores
                logic_gaps=[],
                refinement_roadmap=[],
                attempts=attempt,
                duration_ms=duration_ms,
                # Enhanced fields
                quality_analysis=quality_analysis,
                calibration_examples=calibration_examples,
                rubric_levels=rubric_levels,
                evaluation_purpose=evaluation_purpose,
                ai_system_context=ai_system_context
            )

        except Exception as e:
            last_error = str(e)
            logger.error(f"Attempt {attempt} failed for '{aspect}': {e}")

            # Exponential backoff before retry
            if attempt < max_retries:
                wait_time = 2 ** attempt  # 2s, 4s, 8s
                logger.info(f"Retrying '{aspect}' in {wait_time}s...")
                await asyncio.sleep(wait_time)

    # All retries exhausted
    duration_ms = int((time.time() - start_time) * 1000)

    return AspectEvalResult(
        aspect=aspect,
        success=False,
        error=f"Failed after {max_retries} attempts: {last_error}",
        attempts=max_retries,
        duration_ms=duration_ms
    )


async def generate_multiple_evals_parallel(
    aspects: List[str],
    system_prompt: str,
    use_case: str,
    llm_client: Any,
    provider: str,
    api_key: str,
    model_name: str,
    max_retries: int = 3
) -> List[AspectEvalResult]:
    """
    Generate eval prompts for multiple aspects IN PARALLEL.

    Benefits:
    - Each aspect gets its own token budget (no token limit issues)
    - Parallel execution (faster than sequential)
    - Automatic retry on failure
    - Isolated tasks (one failure doesn't affect others)

    Args:
        aspects: List of aspects to evaluate (e.g., ["Accuracy", "Tone"])
        system_prompt: The AI system prompt
        use_case: Use case context
        llm_client: LLM client instance
        provider: LLM provider
        api_key: API key
        model_name: Model to use
        max_retries: Retry attempts per aspect

    Returns:
        List of AspectEvalResult (same order as input aspects)
    """

    logger.info(f"Starting parallel generation for {len(aspects)} aspects")

    # Create tasks for all aspects
    tasks = [
        generate_single_aspect_with_retry(
            aspect=aspect,
            system_prompt=system_prompt,
            use_case=use_case,
            llm_client=llm_client,
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            max_retries=max_retries
        )
        for aspect in aspects
    ]

    # Run all tasks in parallel and wait for ALL to complete
    # Using gather with return_exceptions=True so one failure doesn't cancel others
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions that weren't caught
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Task raised an exception that wasn't caught
            final_results.append(AspectEvalResult(
                aspect=aspects[i],
                success=False,
                error=f"Unexpected error: {str(result)}",
                attempts=max_retries
            ))
        else:
            final_results.append(result)

    # Log summary
    successful = sum(1 for r in final_results if r.success)
    failed = len(final_results) - successful
    total_duration = sum(r.duration_ms for r in final_results)
    avg_duration = total_duration / len(final_results) if final_results else 0

    logger.info(f"Parallel generation complete: {successful} successful, {failed} failed")
    logger.info(f"Total time: {total_duration}ms, Average: {avg_duration:.0f}ms per aspect")

    return final_results
