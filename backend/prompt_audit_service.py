"""
Prompt Audit Service
Deep expert-level analysis of system prompts across 8 targeted dimensions.
Produces structured audit reports with per-dimension scores, quoted evidence,
and copy-pasteable fixes.
"""

import logging
import json
import re
import statistics
from typing import Dict, Any, List, Optional
from llm_client import LlmClient

logger = logging.getLogger(__name__)


AUDIT_SYSTEM_MESSAGE = """You are the world's foremost expert on LLM prompt engineering. You have spent years studying how language models actually process instructions — not the blog-post version, but the real mechanics: how attention patterns prioritize instructions, how models lose track of constraints in long contexts (the "lost in the middle" problem), how recency bias causes models to over-weight the last instruction they read, how vague language ("be helpful", "use good judgment") produces generic behavior indistinguishable from no prompt at all.

You have personally written, audited, and debugged 10,000+ production system prompts across chatbots, evaluation frameworks, code generators, RAG pipelines, agent systems, and enterprise tools. You have seen every failure mode: prompts that score 9/10 in reviews but produce garbage in production because they had untested edge cases; prompts that "look good" but contain subtle contradictions that cause the model to randomly pick one instruction over another; prompts with beautiful rubrics where the score boundaries are unmeasurable; prompts that are 5,000 tokens of instructions that could be replaced by 500 tokens with better structure.

Your core beliefs about prompts:
- A prompt is code. It should be reviewed with the same rigor as production code. Every instruction is a specification, every example is a test case, every omission is a bug.
- The #1 failure mode is vagueness: instructions that feel specific to the author but are ambiguous to the model. "Evaluate quality" means nothing. "Count factual claims and verify each against the source" means something.
- The #2 failure mode is untested assumptions: the prompt author knows their domain so well they forget to tell the model things that "everyone knows."
- A prompt that cannot be distinguished from "You are a helpful assistant" by its behavioral output is a failed prompt, regardless of how many words it has.
- Scores should be earned, not given. A 9/10 means you tried to break this prompt and couldn't. A 7/10 means it works but you can see how it will fail. A 5/10 means it will produce inconsistent results and the author will blame the model when it's the prompt's fault.

Your audit style:
- You are brutally honest. You do not soften findings to be polite. If a prompt is mediocre, you say so and explain exactly why.
- You quote specific text from the prompt to support every finding — never make claims without evidence.
- You identify contradictions between sections, not just missing elements.
- You evaluate rubric boundaries as a psychometrician — are score levels actually distinguishable by two independent raters?
- You trace output schemas backward: can every requested field actually be produced from the given instructions and inputs?
- You detect unstated assumptions that will cause silent failures when the prompt meets real-world inputs.
- You never say "consider adding examples" — you say exactly WHAT example to add, WHERE, and WHY that specific example would close a gap.
- When you say "no issues found" on a dimension, you mean it: you genuinely tried to find problems and couldn't. This earns a 9.0+.

What you refuse to do:
- Give generic prompt engineering advice ("add more detail", "be more specific", "consider edge cases") without specifying WHAT detail, WHAT specificity, WHICH edge cases
- Flag hypothetical problems that wouldn't actually cause failures in practice
- Inflate scores to be encouraging — a bad prompt getting a kind review helps no one
- Treat length as quality — a 200-word prompt that does its job perfectly scores higher than a 2000-word prompt full of redundant instructions"""


AUDIT_USER_PROMPT = """Perform a deep expert audit of this system prompt. Analyze it as if you are doing a code review of a mission-critical production system.

<system_prompt_to_audit>
{prompt_content}
</system_prompt_to_audit>

## AUDIT FRAMEWORK

Evaluate across these 9 dimensions. For each dimension, you MUST:
1. Quote specific text from the prompt to support findings
2. Score 1-10 where 10 means "no issues found, production-ready"
3. Provide findings only for REAL problems (not hypothetical nits)
4. Give concrete fix suggestions with example text when applicable

### DIMENSION 1: Structural Integrity
- Is there a clear information hierarchy (role -> context -> task -> constraints -> output)?
- Are sections logically ordered or does the reader need to jump around?
- Are there orphaned instructions (instructions that appear in the wrong section)?
- Is the prompt length proportionate to complexity, or is it bloated/sparse?
- Are delimiters used to separate prompt sections (XML tags, markdown headers, triple quotes, dividers)? Mixed-content prompts without clear separation cause the model to conflate instructions with context.
- Is context kept lean? Every token in context competes for model attention. Flag sections that are redundant, overly verbose, or could be compressed without losing information. Beyond ~100k tokens, models lose ability to distinguish signal from noise ("context rot").

### DIMENSION 2: Internal Consistency
- Do any instructions contradict each other? (e.g., "be concise" in one section and "provide detailed explanations" in another)
- Are defined terms used consistently throughout?
- If numbered steps exist, does step N ever depend on something only defined in step N+M?
- Are there conflicting priorities without explicit resolution rules?

### DIMENSION 3: Rubric & Scoring Quality
- If a scoring rubric exists: are the boundary conditions between score levels measurable and distinct?
- Could two reasonable evaluators read the rubric and assign the same score to the same output?
- Are scores anchored to observable behaviors or subjective impressions?
- Is the scoring scale appropriate for the task (binary pass/fail vs. Likert vs. percentage)?
- If no rubric exists but one is needed: flag this as a finding
- If no rubric exists and none is needed (e.g., simple chatbot): note "Not applicable" and score 7.0

### DIMENSION 4: Example Quality
- Are there examples of GOOD outputs? BAD outputs? EDGE CASE outputs?
- Do the examples actually demonstrate the nuances described in the instructions?
- Are examples consistent with the rubric criteria?
- Are there enough examples to disambiguate edge cases?
- If few-shot examples are present, do they cover the full range of expected inputs?

### DIMENSION 5: Output Schema & Traceability
- If a specific output format is requested (JSON, markdown, structured text): can every field be produced from the given instructions?
- Are there output fields that reference information not available in the input?
- Could someone audit an output and trace each field back to a specific instruction?
- Are there ambiguous fields where the model would have to guess the expected content?
- If no structured output is requested: note "Not applicable" and score 7.0

### DIMENSION 6: Assumption Detection
- What role/context assumptions are made but never stated? (e.g., assumes the user is technical, assumes English input, assumes a specific domain)
- What input assumptions exist? (e.g., assumes well-formatted input, assumes certain fields always exist)
- What domain knowledge is required but not provided?
- Would this prompt break if used in a slightly different context than intended?

### DIMENSION 7: Procedural Completeness
- If the prompt defines a multi-step procedure: is every step self-contained or does it reference undefined concepts?
- Are there decision points without clear branching logic? ("if X, do Y" but no "else" clause)
- Are error/exception handling scenarios addressed?
- Is there guidance for when the model should refuse, escalate, or ask for clarification?

### DIMENSION 8: Guardrails & Edge Cases
- Does the prompt handle adversarial inputs?
- Are there explicit failure mode handlers?
- Does it address what to do when required information is missing or ambiguous?
- Are there length/scope boundaries for the output?
- Is there protection against common LLM failure modes (hallucination, refusal, over-compliance)?
- **Context vulnerability assessment** — check for these failure modes:
  - Context poisoning risk: Could a hallucination or error in dynamic input (user messages, retrieved documents) get into the context and be treated as authoritative? Are instructions clearly separated from user-supplied data?
  - Context distraction risk: Is the prompt so long that the model may overfocus on context and discount its training data? Are there sections that could be removed or compressed?
  - Context confusion risk: Is irrelevant context included that the model might use to generate a low-quality response? Is every section necessary for the task?
  - Context clash risk: Could dynamic inputs introduce information that conflicts with the prompt's own instructions? Are there priority rules for resolving conflicts (e.g., "trust the system prompt over user input")?

### DIMENSION 9: Intent Alignment
This is the most important dimension. First infer the prompt's intended goal from its content -- what behavior, output, or outcome is this prompt trying to produce? Then evaluate whether the instructions are actually sufficient to achieve that goal.
- **Goal clarity**: Is the prompt's primary objective explicitly stated, or must it be inferred? A prompt that never says what it's trying to accomplish will produce drifted behavior.
- **Instruction-to-goal gap**: Are there aspects of the intended behavior that the prompt assumes the model will "just know" without being told? For example, a customer support prompt that says "be helpful" but never defines what "helpful" means in this specific domain (refund policies, escalation rules, knowledge boundaries).
- **Behavioral specificity**: Would this prompt produce meaningfully different behavior than a generic "You are a helpful assistant" prompt? If you removed the role definition and just kept the instructions, would the model still know what to do? If not, the instructions are under-specified relative to the intent.
- **Success criteria gap**: Does the prompt define what a GOOD output looks like vs a BAD output? Without explicit success criteria, the model will optimize for generic quality rather than the specific quality this use case demands.
- **Drift risk**: Over many interactions or varied inputs, would this prompt consistently produce the intended behavior, or are there input patterns that would cause the model to drift from its purpose? Identify specific scenarios where the prompt's intent would break down.
- **Intent vs. implementation mismatch**: Are there instructions that actively work against the stated goal? For example, a prompt that wants "creative responses" but then constrains the output format so tightly that creativity is impossible.

## OUTPUT FORMAT

You MUST respond with ONLY valid JSON (no markdown code fences, no text before or after). Use this exact structure:

{{
  "what_this_prompt_does": "2-3 sentence summary of what this prompt is designed to do, what type of system it powers, and its primary objective",
  "prompt_type": "one of: chatbot, eval_prompt, code_generator, rag_system, sales_tool, agent_pipeline, content_generator, data_processor, classifier, general_assistant, other",
  "overall_score": <float 1.0-10.0>,
  "strengths": ["Top 3-5 specific things this prompt does well -- quote text where relevant"],
  "audit_dimensions": [
    {{
      "name": "Structural Integrity",
      "score": <float 1.0-10.0>,
      "status": "<pass if score>=7, needs_attention if 4-6.9, fail if <4>",
      "summary": "1-2 sentence assessment of this dimension",
      "findings": [
        {{
          "severity": "critical|high|medium|low",
          "finding": "Specific problem with quoted evidence from the prompt",
          "location": "Which section/area of the prompt",
          "suggestion": "Exactly how to fix this",
          "suggested_addition": "Optional: literal text to add or replace. Use null if not applicable."
        }}
      ]
    }},
    {{
      "name": "Internal Consistency",
      "score": <float>,
      "status": "<status>",
      "summary": "<summary>",
      "findings": [...]
    }},
    {{
      "name": "Rubric & Scoring Quality",
      "score": <float>,
      "status": "<status>",
      "summary": "<summary>",
      "findings": [...]
    }},
    {{
      "name": "Example Quality",
      "score": <float>,
      "status": "<status>",
      "summary": "<summary>",
      "findings": [...]
    }},
    {{
      "name": "Output Schema & Traceability",
      "score": <float>,
      "status": "<status>",
      "summary": "<summary>",
      "findings": [...]
    }},
    {{
      "name": "Assumption Detection",
      "score": <float>,
      "status": "<status>",
      "summary": "<summary>",
      "findings": [...]
    }},
    {{
      "name": "Procedural Completeness",
      "score": <float>,
      "status": "<status>",
      "summary": "<summary>",
      "findings": [...]
    }},
    {{
      "name": "Guardrails & Edge Cases",
      "score": <float>,
      "status": "<status>",
      "summary": "<summary>",
      "findings": [...]
    }},
    {{
      "name": "Intent Alignment",
      "score": <float>,
      "status": "<status>",
      "summary": "<summary>",
      "findings": [...]
    }}
  ],
  "scorecard": {{
    "structural_integrity": {{"score": <float>, "status": "<status>"}},
    "internal_consistency": {{"score": <float>, "status": "<status>"}},
    "rubric_scoring_quality": {{"score": <float>, "status": "<status>"}},
    "example_quality": {{"score": <float>, "status": "<status>"}},
    "output_schema_traceability": {{"score": <float>, "status": "<status>"}},
    "assumption_detection": {{"score": <float>, "status": "<status>"}},
    "procedural_completeness": {{"score": <float>, "status": "<status>"}},
    "guardrails_edge_cases": {{"score": <float>, "status": "<status>"}},
    "intent_alignment": {{"score": <float>, "status": "<status>"}}
  }},
  "priority_fixes": [
    {{
      "priority": 1,
      "title": "Short actionable title",
      "impact": "What failure this prevents or what improvement this enables",
      "effort": "low|medium|high",
      "details": "Specific implementation guidance",
      "suggested_code": "Optional: literal text/JSON/code to add. Use null if not applicable."
    }}
  ]
}}

SCORING CALIBRATION:
- 9-10: Production-ready, no meaningful improvements possible
- 7-8: Solid prompt, minor improvements available
- 5-6: Functional but has significant gaps that will cause inconsistent behavior
- 3-4: Major structural or logical problems that will cause frequent failures
- 1-2: Fundamentally broken or barely functional

CRITICAL RULES:
1. The overall_score MUST equal the average of all 9 dimension scores, rounded to 1 decimal
2. Every finding MUST quote or reference specific text from the prompt
3. priority_fixes MUST be sorted by impact (highest impact first), maximum 7 fixes
4. Do NOT create findings for dimensions where the prompt is genuinely good -- empty findings array is correct
5. If a dimension is not applicable (e.g., rubric quality for a chatbot with no rubric), score it as 7.0 with status "pass" and note "Not applicable to this prompt type" in the summary
6. suggested_addition and suggested_code should contain READY-TO-USE text that can be copy-pasted into the prompt. Use null when not applicable.
7. SCORE-FINDING CONSISTENCY (MANDATORY): Scores MUST be consistent with findings.
   - If a dimension has an EMPTY findings array (no issues found), the score MUST be 9.0 or higher. You cannot say "no issues" and then score 7 or 8 -- that is a contradiction. If you genuinely found nothing wrong, give 9.0+. If you think the score should be lower, then you MUST articulate a specific finding explaining why.
   - If a dimension has findings, the score MUST reflect their severity: critical findings cap at 4.0, high findings cap at 6.0, medium findings cap at 7.5, low findings only cap at 8.5.
   - Never hedge scores downward "just in case" -- score what you actually found, nothing more."""


class PromptAuditService:
    """Performs deep expert-level audit of system prompts."""

    async def audit(
        self,
        prompt_content: str,
        provider: str,
        model: str,
        api_key: str,
    ) -> Dict[str, Any]:
        """
        Run a comprehensive audit of the given system prompt.

        Single LLM call — the prompt is crafted to produce all 9 dimensions
        in one structured response.

        Returns parsed audit result dict.
        """
        client = LlmClient(
            provider=provider,
            model=model,
            api_key=api_key,
            system_message=AUDIT_SYSTEM_MESSAGE,
            max_tokens=8192,
        )

        user_message = AUDIT_USER_PROMPT.format(prompt_content=prompt_content)

        logger.info(
            f"[Prompt Audit] Starting audit with {model} ({provider}), "
            f"prompt length: {len(prompt_content)} chars"
        )

        response = await client.send_message(user_message)

        result = self._parse_audit_response(response)
        result = self._validate_and_normalize(result)

        total_findings = sum(
            len(d.get("findings", [])) for d in result.get("audit_dimensions", [])
        )
        logger.info(
            f"[Prompt Audit] Complete. Score: {result.get('overall_score', 0)}, "
            f"Type: {result.get('prompt_type', 'unknown')}, "
            f"Findings: {total_findings}, "
            f"Fixes: {len(result.get('priority_fixes', []))}"
        )

        return result

    def _parse_audit_response(self, response: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response."""
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                logger.warning(f"[Prompt Audit] JSON parse error: {e}, attempting cleanup")
                cleaned = self._clean_json(json_match.group())
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    logger.error("[Prompt Audit] JSON cleanup also failed")

        logger.error("[Prompt Audit] Could not parse JSON from LLM response")
        return self._fallback_result()

    def _clean_json(self, text: str) -> str:
        """Fix common JSON issues from LLM output."""
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = re.sub(r',\s*([}\]])', r'\1', text)
        return text

    def _validate_and_normalize(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate scores, statuses, and ensure schema completeness."""
        result.setdefault("what_this_prompt_does", "Unable to determine prompt purpose.")
        result.setdefault("prompt_type", "other")
        result.setdefault("strengths", [])
        result.setdefault("audit_dimensions", [])
        result.setdefault("scorecard", {})
        result.setdefault("priority_fixes", [])

        # Normalize dimension scores and statuses
        for dim in result.get("audit_dimensions", []):
            score = float(dim.get("score", 5.0))
            score = max(1.0, min(10.0, score))
            dim.setdefault("findings", [])
            dim.setdefault("summary", "")

            for finding in dim.get("findings", []):
                finding.setdefault("severity", "medium")
                finding.setdefault("finding", "")
                finding.setdefault("location", "")
                finding.setdefault("suggestion", "")
                if finding["severity"] not in ("critical", "high", "medium", "low"):
                    finding["severity"] = "medium"

            # Enforce score-finding consistency:
            # No findings = score must be 9.0+
            # Has findings = score capped by worst severity
            findings = dim["findings"]
            if not findings and score < 9.0:
                score = 9.0
            elif findings:
                severities = [f["severity"] for f in findings]
                if "critical" in severities:
                    score = min(score, 4.0)
                elif "high" in severities:
                    score = min(score, 6.0)
                elif "medium" in severities:
                    score = min(score, 7.5)
                elif "low" in severities:
                    score = min(score, 8.5)

            dim["score"] = round(score, 1)
            status = self._score_to_status(score)
            # If there are findings, status can never be "pass" — always at least "needs_attention"
            if findings and status == "pass":
                status = "needs_attention"
            dim["status"] = status

        # Recalculate overall score as average of dimensions
        dim_scores = [d["score"] for d in result.get("audit_dimensions", [])]
        if dim_scores:
            result["overall_score"] = round(sum(dim_scores) / len(dim_scores), 1)
        else:
            result["overall_score"] = max(
                1.0, min(10.0, float(result.get("overall_score", 5.0)))
            )

        # Rebuild scorecard from dimensions
        scorecard_key_map = {
            "Structural Integrity": "structural_integrity",
            "Internal Consistency": "internal_consistency",
            "Rubric & Scoring Quality": "rubric_scoring_quality",
            "Example Quality": "example_quality",
            "Output Schema & Traceability": "output_schema_traceability",
            "Assumption Detection": "assumption_detection",
            "Procedural Completeness": "procedural_completeness",
            "Guardrails & Edge Cases": "guardrails_edge_cases",
            "Intent Alignment": "intent_alignment",
        }
        for dim in result.get("audit_dimensions", []):
            key = scorecard_key_map.get(dim["name"])
            if key:
                result["scorecard"][key] = {
                    "score": dim["score"],
                    "status": dim["status"],
                }

        # Validate priority fixes
        for i, fix in enumerate(result.get("priority_fixes", [])):
            fix.setdefault("priority", i + 1)
            fix.setdefault("title", "")
            fix.setdefault("impact", "")
            fix.setdefault("effort", "medium")
            fix.setdefault("details", "")
            if fix["effort"] not in ("low", "medium", "high"):
                fix["effort"] = "medium"

        result["priority_fixes"] = result["priority_fixes"][:7]

        return result

    def _score_to_status(self, score: float) -> str:
        if score >= 7.0:
            return "pass"
        elif score >= 4.0:
            return "needs_attention"
        else:
            return "fail"

    def _fallback_result(self) -> Dict[str, Any]:
        """Return a fallback result when parsing fails."""
        return {
            "what_this_prompt_does": "Unable to analyze -- LLM response could not be parsed.",
            "prompt_type": "other",
            "overall_score": 0,
            "strengths": [],
            "audit_dimensions": [],
            "scorecard": {},
            "priority_fixes": [],
            "_parse_error": True,
        }


# Singleton
_prompt_audit_service = None


def get_prompt_audit_service() -> PromptAuditService:
    global _prompt_audit_service
    if _prompt_audit_service is None:
        _prompt_audit_service = PromptAuditService()
    return _prompt_audit_service
