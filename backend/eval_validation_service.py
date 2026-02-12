"""
Eval Validation Service
Computes discrimination power and score consistency for eval prompts.

Discrimination Power: Measures whether an eval can distinguish easy from hard cases.
Score Consistency: Measures whether an eval produces stable scores across repeated runs.
"""

import logging
import asyncio
import json
import re
import statistics
from typing import Dict, Any, List
from llm_client import LlmClient

logger = logging.getLogger(__name__)


class EvalValidationService:
    """Validates eval prompt quality via discrimination and consistency metrics."""

    async def compute_discrimination_power(
        self,
        eval_prompt_content: str,
        system_prompt: str,
        dimension: str,
        provider: str,
        model: str,
        api_key: str,
        execution_provider: str,
        execution_model: str,
        execution_api_key: str,
    ) -> Dict[str, Any]:
        """
        Measure whether the eval can distinguish easy from hard cases.

        1. Generate 3 easy cases (should score 4-5) + 3 hard cases (should score 1-2)
        2. Get system outputs for all 6
        3. Run eval on all 6
        4. Score = avg(easy_scores) - avg(hard_scores)

        Rating: good (>=1.5), fair (>=0.5), poor (<0.5) on 0-5 scale
        """
        gen_client = LlmClient(
            provider=provider, model=model, api_key=api_key,
            system_message="You generate targeted test cases for LLM evaluation."
        )

        easy_prompt = f"""Generate 3 test inputs for a system with this prompt:

SYSTEM PROMPT:
{system_prompt[:2000]}

These inputs should be EASY cases where the system would perform PERFECTLY on the "{dimension}" dimension.
A well-functioning system should score HIGH (4-5 out of 5) on these.

Return ONLY a JSON array:
[{{"input": "the user input", "reason": "why this is easy for {dimension}"}}]"""

        hard_prompt = f"""Generate 3 test inputs for a system with this prompt:

SYSTEM PROMPT:
{system_prompt[:2000]}

These inputs should be CHALLENGING cases where the system would STRUGGLE on the "{dimension}" dimension.
Even a decent system should score LOW (1-2 out of 5) on these.

Think about:
- Inputs that expose common weaknesses in {dimension}
- Tricky scenarios where maintaining {dimension} is genuinely difficult
- Cases with conflicting requirements that make {dimension} hard

Return ONLY a JSON array:
[{{"input": "the user input", "reason": "why this is hard for {dimension}"}}]"""

        # Generate easy and hard cases in parallel
        easy_resp, hard_resp = await asyncio.gather(
            gen_client.send_message(easy_prompt),
            gen_client.send_message(hard_prompt)
        )

        easy_cases = self._parse_json_array(easy_resp)
        hard_cases = self._parse_json_array(hard_resp)

        if not easy_cases or not hard_cases:
            return {
                "discrimination_score": 0,
                "avg_easy_score": 0,
                "avg_hard_score": 0,
                "easy_details": [],
                "hard_details": [],
                "rating": "error",
                "error": "Could not generate test cases"
            }

        # Get system outputs for all cases in parallel
        system_client = LlmClient(
            provider=provider, model=model, api_key=api_key,
            system_message=system_prompt
        )

        all_inputs = [(c.get("input", ""), "easy") for c in easy_cases] + \
                     [(c.get("input", ""), "hard") for c in hard_cases]

        output_tasks = [system_client.send_message(inp) for inp, _ in all_inputs]
        outputs = await asyncio.gather(*output_tasks, return_exceptions=True)

        # Run eval on each (input, output) pair
        eval_client = LlmClient(
            provider=execution_provider, model=execution_model, api_key=execution_api_key,
            system_message=eval_prompt_content
        )

        eval_tasks = []
        valid_pairs = []
        for (inp, difficulty), output in zip(all_inputs, outputs):
            if isinstance(output, Exception):
                logger.warning(f"[Discrimination] Output generation failed: {output}")
                continue
            eval_input = f"Input: {inp}\n\nOutput: {output}"
            eval_tasks.append(eval_client.send_message(eval_input))
            valid_pairs.append((inp, difficulty, output))

        eval_responses = await asyncio.gather(*eval_tasks, return_exceptions=True)

        # Collect scores
        easy_details = []
        hard_details = []
        for (inp, difficulty, output), eval_resp in zip(valid_pairs, eval_responses):
            if isinstance(eval_resp, Exception):
                logger.warning(f"[Discrimination] Eval failed: {eval_resp}")
                continue
            score = self._extract_score(eval_resp)
            detail = {"input": inp[:200], "score": score, "output_preview": str(output)[:200]}
            if difficulty == "easy":
                easy_details.append(detail)
            else:
                hard_details.append(detail)

        easy_scores = [d["score"] for d in easy_details]
        hard_scores = [d["score"] for d in hard_details]

        avg_easy = statistics.mean(easy_scores) if easy_scores else 0
        avg_hard = statistics.mean(hard_scores) if hard_scores else 0
        discrimination_score = avg_easy - avg_hard

        rating = "good" if discrimination_score >= 1.5 else ("fair" if discrimination_score >= 0.5 else "poor")

        logger.info(f"[Discrimination] {dimension}: easy={avg_easy:.1f}, hard={avg_hard:.1f}, gap={discrimination_score:.1f} ({rating})")

        return {
            "discrimination_score": round(discrimination_score, 2),
            "avg_easy_score": round(avg_easy, 2),
            "avg_hard_score": round(avg_hard, 2),
            "easy_details": easy_details,
            "hard_details": hard_details,
            "rating": rating
        }

    async def compute_score_consistency(
        self,
        eval_prompt_content: str,
        system_prompt: str,
        dimension: str,
        provider: str,
        model: str,
        api_key: str,
        execution_provider: str,
        execution_model: str,
        execution_api_key: str,
        num_trials: int = 3,
    ) -> Dict[str, Any]:
        """
        Measure scoring consistency by running the same eval multiple times.

        1. Generate 3 diverse test cases (easy/medium/hard)
        2. Get system outputs
        3. Run eval num_trials times per case
        4. Compute std_dev per case
        5. Consistency = 1 - (avg_std_dev / max_score)

        Rating: good (>=0.8), fair (>=0.5), poor (<0.5)
        """
        max_score = 5.0

        gen_client = LlmClient(
            provider=provider, model=model, api_key=api_key,
            system_message="You generate test cases for LLM evaluation."
        )

        gen_prompt = f"""Generate 3 diverse test inputs for this system:

SYSTEM PROMPT:
{system_prompt[:2000]}

Include 1 easy, 1 medium, and 1 hard case for the "{dimension}" dimension.

Return ONLY a JSON array:
[{{"input": "the user input", "difficulty": "easy|medium|hard"}}]"""

        gen_resp = await gen_client.send_message(gen_prompt)
        test_cases = self._parse_json_array(gen_resp)

        if not test_cases:
            return {
                "consistency_score": 0,
                "avg_std_dev": 0,
                "case_details": [],
                "num_trials": num_trials,
                "rating": "error",
                "error": "Could not generate test cases"
            }

        # Get system outputs in parallel
        system_client = LlmClient(
            provider=provider, model=model, api_key=api_key,
            system_message=system_prompt
        )

        output_tasks = [system_client.send_message(tc.get("input", "")) for tc in test_cases]
        outputs = await asyncio.gather(*output_tasks, return_exceptions=True)

        # Run eval multiple times per case
        eval_client = LlmClient(
            provider=execution_provider, model=execution_model, api_key=execution_api_key,
            system_message=eval_prompt_content
        )

        case_results = []
        for tc, output in zip(test_cases, outputs):
            if isinstance(output, Exception):
                logger.warning(f"[Consistency] Output generation failed: {output}")
                continue

            eval_input = f"Input: {tc.get('input', '')}\n\nOutput: {output}"

            # Run eval num_trials times in parallel
            trial_tasks = [eval_client.send_message(eval_input) for _ in range(num_trials)]
            trial_results = await asyncio.gather(*trial_tasks, return_exceptions=True)

            scores = []
            for tr in trial_results:
                if not isinstance(tr, Exception):
                    score = self._extract_score(tr)
                    scores.append(score)

            if len(scores) >= 2:
                std_dev = statistics.stdev(scores)
                case_results.append({
                    "input": tc.get("input", "")[:150],
                    "difficulty": tc.get("difficulty", "unknown"),
                    "scores": scores,
                    "std_dev": round(std_dev, 2),
                    "mean": round(statistics.mean(scores), 2)
                })

        if case_results:
            avg_std_dev = statistics.mean([cr["std_dev"] for cr in case_results])
            consistency_score = max(0, 1.0 - (avg_std_dev / max_score))
        else:
            avg_std_dev = 0
            consistency_score = 0

        rating = "good" if consistency_score >= 0.8 else ("fair" if consistency_score >= 0.5 else "poor")

        logger.info(f"[Consistency] {dimension}: avg_std={avg_std_dev:.2f}, consistency={consistency_score:.2f} ({rating})")

        return {
            "consistency_score": round(consistency_score, 2),
            "avg_std_dev": round(avg_std_dev, 2),
            "case_details": case_results,
            "num_trials": num_trials,
            "rating": rating
        }

    def _parse_json_array(self, text: str) -> list:
        """Extract JSON array from LLM response."""
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("[Parse] Failed to parse JSON array from response")
        return []

    def _extract_score(self, eval_response: str) -> float:
        """Extract numeric score from eval response (mirrors server.py pattern)."""
        json_match = re.search(r'\{[\s\S]*\}', eval_response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                # Try common score field names
                for key in ['score', 'dimension_score', 'overall_score', 'total_score']:
                    if key in data:
                        return float(data[key])
                # Check nested: eval_result.score
                if 'eval_result' in data and isinstance(data['eval_result'], dict):
                    return float(data['eval_result'].get('score', 0))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        # Fallback: look for a number after "score"
        score_match = re.search(r'score["\s:]+(\d+(?:\.\d+)?)', eval_response, re.IGNORECASE)
        if score_match:
            return float(score_match.group(1))
        return 0.0


# Singleton
_eval_validation_service = None


def get_eval_validation_service() -> EvalValidationService:
    global _eval_validation_service
    if _eval_validation_service is None:
        _eval_validation_service = EvalValidationService()
    return _eval_validation_service
