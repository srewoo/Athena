"""
Agentic Eval Prompt Generator for Athena
Implements multi-step, self-validating evaluation prompt generation with thinking model support.

Workflow:
1. Deep Analysis (thinking model) - Understand what the system prompt does and what could go wrong
2. Failure Mode Identification - Enumerate specific ways outputs could fail
3. Generate Eval Criteria - Create dimensions to catch each failure mode
4. Self-Test - Validate criteria would actually catch bad outputs
5. Refine - Add missing criteria if gaps found
"""
import re
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from prompt_analyzer import analyze_prompt, analysis_to_dict, PromptAnalysis

logger = logging.getLogger(__name__)


def extract_template_variables(prompt_text: str) -> List[str]:
    """Extract template variables like {{callTranscript}} from the prompt"""
    pattern = r'\{\{(\w+)\}\}'
    matches = re.findall(pattern, prompt_text)
    return list(set(matches))  # Remove duplicates


def get_primary_input_variable(system_prompt: str) -> str:
    """
    Get the primary input variable name from the system prompt.
    Returns the extracted variable name or 'input' as fallback.
    """
    variables = extract_template_variables(system_prompt)
    if variables:
        # Prefer variables that look like inputs
        input_keywords = ['input', 'transcript', 'text', 'content', 'data', 'query', 'message', 'code', 'email', 'document']
        for var in variables:
            var_lower = var.lower()
            for keyword in input_keywords:
                if keyword in var_lower:
                    return var
        return variables[0]
    return "input"


@dataclass
class FailureMode:
    """A specific way the system prompt's output could fail"""
    name: str
    description: str
    severity: str  # "critical", "major", "minor"
    example_bad_output: str
    detection_criteria: str


@dataclass
class EvalDimension:
    """An evaluation dimension with scoring rubric"""
    name: str
    description: str
    what_to_check: str
    score_5: str  # What earns a 5
    score_3: str  # What earns a 3
    score_1: str  # What earns a 1
    failure_modes_covered: List[str]


@dataclass
class SelfTestResult:
    """Result of self-testing the eval criteria"""
    covers_all_failure_modes: bool
    uncovered_modes: List[str]
    redundant_dimensions: List[str]
    suggestions: List[str]
    confidence_score: float  # 0-1


@dataclass
class AgenticEvalResult:
    """Complete result from agentic eval generation"""
    eval_prompt: str
    eval_criteria: List[str]
    rationale: str
    failure_modes: List[Dict[str, Any]]
    eval_dimensions: List[Dict[str, Any]]
    self_test: Dict[str, Any]
    iterations: int
    steps_taken: List[Dict[str, Any]]


async def deep_prompt_analysis(
    system_prompt: str,
    use_case: str,
    requirements: str,
    llm_client,
    provider: str,
    api_key: str,
    model_name: str
) -> Dict[str, Any]:
    """
    Step 1: Deep analysis of what the system prompt does and what could go wrong.
    Uses thinking model for thorough analysis.
    """
    # Get programmatic analysis first
    programmatic = analyze_prompt(system_prompt)
    analysis_dict = analysis_to_dict(programmatic)

    analysis_prompt = """You are an expert QA engineer analyzing a system prompt to understand what could go wrong.

Think deeply about:
1. **What does this prompt do?** - Core function, inputs, outputs
2. **What are the critical success factors?** - What MUST be right for the output to be useful
3. **What are the likely failure modes?** - How could outputs go wrong?
4. **What edge cases exist?** - Unusual inputs that could cause issues
5. **What safety concerns exist?** - Could outputs be harmful, biased, or inappropriate?

Return your analysis as JSON:
{
    "core_function": "What this prompt is designed to do",
    "expected_input": "What input the prompt expects",
    "expected_output": "What output format/content is expected",
    "critical_success_factors": [
        "Factor 1 - what must be right",
        "Factor 2 - what must be right"
    ],
    "likely_failure_modes": [
        {
            "name": "Short name for failure",
            "description": "How this failure manifests",
            "severity": "critical|major|minor",
            "example": "Example of what a bad output would look like"
        }
    ],
    "edge_cases": ["Edge case 1", "Edge case 2"],
    "safety_concerns": ["Concern 1", "Concern 2"]
}"""

    user_message = f"""Analyze this system prompt to understand what could go wrong:

**Use Case:** {use_case}

**Requirements:** {requirements}

**System Prompt:**
```
{system_prompt}
```

**Programmatic Analysis:**
- Prompt Type: {analysis_dict['prompt_type']}
- Output Format: {analysis_dict['dna']['output_format']}
- Template Variables: {analysis_dict['dna']['template_variables']}
- Scoring Scale: {analysis_dict['dna']['scoring_scale']}

Provide deep analysis of what could go wrong with outputs from this prompt."""

    result = await llm_client.chat(
        system_prompt=analysis_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.2,
        max_tokens=4000
    )

    if result.get("error"):
        logger.error(f"Deep analysis failed: {result['error']}")
        return {
            "programmatic": analysis_dict,
            "deep": None,
            "error": result["error"]
        }

    try:
        json_match = re.search(r'\{[\s\S]*\}', result["output"])
        if json_match:
            deep = json.loads(json_match.group())
        else:
            deep = {"raw_response": result["output"]}
    except json.JSONDecodeError:
        deep = {"raw_response": result["output"]}

    return {
        "programmatic": analysis_dict,
        "deep": deep
    }


async def identify_failure_modes(
    system_prompt: str,
    analysis: Dict[str, Any],
    llm_client,
    provider: str,
    api_key: str,
    model_name: str
) -> List[FailureMode]:
    """
    Step 2: Enumerate specific failure modes based on analysis.
    """
    deep = analysis.get("deep", {})
    programmatic = analysis.get("programmatic", {})

    # Start with failure modes from deep analysis
    likely_failures = deep.get("likely_failure_modes", [])

    # Add type-specific failure modes
    prompt_type = programmatic.get("prompt_type", "unknown")
    type_failures = get_type_specific_failures(prompt_type, programmatic.get("dna", {}))

    failure_prompt = """You are a QA expert identifying all possible ways an LLM output could fail.

For each failure mode, provide:
1. A clear name
2. Description of how it manifests
3. Severity (critical = unusable, major = significantly wrong, minor = imperfect but usable)
4. Example of what a bad output would look like
5. How to detect this failure

Return JSON array:
[
    {
        "name": "Failure name",
        "description": "How this failure manifests",
        "severity": "critical|major|minor",
        "example_bad_output": "Example of bad output",
        "detection_criteria": "How to detect this in evaluation"
    }
]"""

    user_message = f"""Identify ALL failure modes for outputs from this system prompt:

**System Prompt:**
```
{system_prompt}
```

**Already Identified Failures:**
{json.dumps(likely_failures, indent=2)}

**Type-Specific Concerns ({prompt_type}):**
{json.dumps(type_failures, indent=2)}

**Critical Success Factors:**
{json.dumps(deep.get('critical_success_factors', []), indent=2)}

Provide a COMPLETE list of failure modes, including any not yet identified.
Be thorough - missing a failure mode means the eval won't catch it."""

    result = await llm_client.chat(
        system_prompt=failure_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.3,
        max_tokens=4000
    )

    if result.get("error"):
        logger.error(f"Failure mode identification failed: {result['error']}")
        # Return basic failure modes from analysis
        return [
            FailureMode(
                name=f.get("name", "Unknown"),
                description=f.get("description", ""),
                severity=f.get("severity", "major"),
                example_bad_output=f.get("example", ""),
                detection_criteria=f.get("detection_criteria", "Check output quality")
            )
            for f in likely_failures[:5]
        ]

    try:
        json_match = re.search(r'\[[\s\S]*\]', result["output"])
        if json_match:
            failures_data = json.loads(json_match.group())
            return [
                FailureMode(
                    name=f.get("name", "Unknown"),
                    description=f.get("description", ""),
                    severity=f.get("severity", "major"),
                    example_bad_output=f.get("example_bad_output", ""),
                    detection_criteria=f.get("detection_criteria", "")
                )
                for f in failures_data
            ]
    except json.JSONDecodeError:
        pass

    return []


def get_type_specific_failures(prompt_type: str, dna: Dict) -> List[Dict]:
    """Get failure modes specific to the prompt type"""
    type_failures = {
        "analytical": [
            {"name": "Score-Reasoning Mismatch", "description": "Score doesn't match the reasoning provided"},
            {"name": "Missing Evidence", "description": "Claims made without supporting evidence"},
            {"name": "Rubric Deviation", "description": "Scoring doesn't follow the defined rubric"}
        ],
        "structured_output": [
            {"name": "Schema Violation", "description": "Output doesn't match required schema"},
            {"name": "Missing Fields", "description": "Required fields are absent"},
            {"name": "Wrong Data Types", "description": "Fields have incorrect data types"}
        ],
        "conversational": [
            {"name": "Off-Topic Response", "description": "Response doesn't address the query"},
            {"name": "Tone Mismatch", "description": "Tone doesn't match expected style"},
            {"name": "Safety Violation", "description": "Response contains harmful content"}
        ],
        "creative": [
            {"name": "Constraint Violation", "description": "Creative constraints not followed"},
            {"name": "Style Inconsistency", "description": "Writing style doesn't match requirements"},
            {"name": "Generic Output", "description": "Output is template-like, not creative"}
        ],
        "extraction": [
            {"name": "Missed Information", "description": "Relevant information not extracted"},
            {"name": "Fabrication", "description": "Information invented that wasn't in source"},
            {"name": "Misattribution", "description": "Information attributed to wrong source"}
        ]
    }

    return type_failures.get(prompt_type, [
        {"name": "Incorrect Output", "description": "Output doesn't meet requirements"},
        {"name": "Format Error", "description": "Output format is wrong"},
        {"name": "Incomplete Response", "description": "Response is missing required elements"}
    ])


async def generate_eval_dimensions(
    system_prompt: str,
    failure_modes: List[FailureMode],
    analysis: Dict[str, Any],
    llm_client,
    provider: str,
    api_key: str,
    model_name: str
) -> List[EvalDimension]:
    """
    Step 3: Generate evaluation dimensions that cover all failure modes.
    """
    failures_json = [
        {
            "name": f.name,
            "description": f.description,
            "severity": f.severity,
            "detection_criteria": f.detection_criteria
        }
        for f in failure_modes
    ]

    dimension_prompt = """You are designing evaluation criteria for an LLM evaluation system.

Create evaluation dimensions that will CATCH all the identified failure modes.

For each dimension:
1. Name - Clear, descriptive name
2. Description - What this dimension evaluates
3. What to check - Specific things to look for
4. Score 5 (Excellent) - What earns the highest score
5. Score 3 (Acceptable) - What earns a passing score
6. Score 1 (Poor) - What earns the lowest score
7. Which failure modes this dimension catches

Return JSON array:
[
    {
        "name": "Dimension Name",
        "description": "What this evaluates",
        "what_to_check": "Specific checks to perform",
        "score_5": "Criteria for excellent (5)",
        "score_3": "Criteria for acceptable (3)",
        "score_1": "Criteria for poor (1)",
        "failure_modes_covered": ["Failure 1", "Failure 2"]
    }
]

IMPORTANT: Every failure mode must be covered by at least one dimension."""

    user_message = f"""Create evaluation dimensions to catch these failure modes:

**Failure Modes to Cover:**
{json.dumps(failures_json, indent=2)}

**System Prompt Being Evaluated:**
```
{system_prompt}
```

Create 4-6 evaluation dimensions that together cover ALL failure modes.
Each dimension should have clear, measurable criteria."""

    result = await llm_client.chat(
        system_prompt=dimension_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.3,
        max_tokens=4000
    )

    if result.get("error"):
        logger.error(f"Dimension generation failed: {result['error']}")
        return []

    try:
        json_match = re.search(r'\[[\s\S]*\]', result["output"])
        if json_match:
            dimensions_data = json.loads(json_match.group())
            return [
                EvalDimension(
                    name=d.get("name", "Unknown"),
                    description=d.get("description", ""),
                    what_to_check=d.get("what_to_check", ""),
                    score_5=d.get("score_5", "Excellent quality"),
                    score_3=d.get("score_3", "Acceptable quality"),
                    score_1=d.get("score_1", "Poor quality"),
                    failure_modes_covered=d.get("failure_modes_covered", [])
                )
                for d in dimensions_data
            ]
    except json.JSONDecodeError:
        pass

    return []


async def self_test_criteria(
    failure_modes: List[FailureMode],
    eval_dimensions: List[EvalDimension],
    llm_client,
    provider: str,
    api_key: str,
    model_name: str
) -> SelfTestResult:
    """
    Step 4: Self-test to verify eval criteria would catch all failure modes.
    """
    # Check coverage programmatically first
    all_failure_names = {f.name for f in failure_modes}
    covered_failures = set()
    for dim in eval_dimensions:
        covered_failures.update(dim.failure_modes_covered)

    uncovered = all_failure_names - covered_failures

    # Check for redundancy
    coverage_count = {}
    for dim in eval_dimensions:
        for f in dim.failure_modes_covered:
            coverage_count[f] = coverage_count.get(f, 0) + 1

    redundant = [f for f, count in coverage_count.items() if count > 2]

    test_prompt = """You are a QA auditor reviewing evaluation criteria.

Analyze whether these eval dimensions would actually catch the failure modes in practice.

Consider:
1. Are there any failure modes that could slip through?
2. Are the scoring criteria clear enough to apply consistently?
3. Are there redundant dimensions that could be consolidated?
4. What's missing?

Return JSON:
{
    "covers_all_failure_modes": true/false,
    "uncovered_modes": ["Mode 1", "Mode 2"],
    "redundant_dimensions": ["Dim 1 and Dim 2 overlap significantly"],
    "suggestions": ["Suggestion 1", "Suggestion 2"],
    "confidence_score": 0.85
}"""

    failures_json = [{"name": f.name, "severity": f.severity} for f in failure_modes]
    dimensions_json = [
        {"name": d.name, "covers": d.failure_modes_covered}
        for d in eval_dimensions
    ]

    user_message = f"""Audit these evaluation criteria:

**Failure Modes to Catch:**
{json.dumps(failures_json, indent=2)}

**Eval Dimensions:**
{json.dumps(dimensions_json, indent=2)}

**Programmatic Check Results:**
- Uncovered failures: {list(uncovered)}
- Potentially redundant: {redundant}

Would these criteria reliably catch all failure modes?"""

    result = await llm_client.chat(
        system_prompt=test_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=0.2,
        max_tokens=2000
    )

    if result.get("error"):
        return SelfTestResult(
            covers_all_failure_modes=len(uncovered) == 0,
            uncovered_modes=list(uncovered),
            redundant_dimensions=redundant,
            suggestions=[],
            confidence_score=0.7 if len(uncovered) == 0 else 0.4
        )

    try:
        json_match = re.search(r'\{[\s\S]*\}', result["output"])
        if json_match:
            test_data = json.loads(json_match.group())
            return SelfTestResult(
                covers_all_failure_modes=test_data.get("covers_all_failure_modes", False),
                uncovered_modes=test_data.get("uncovered_modes", []),
                redundant_dimensions=test_data.get("redundant_dimensions", []),
                suggestions=test_data.get("suggestions", []),
                confidence_score=test_data.get("confidence_score", 0.5)
            )
    except json.JSONDecodeError:
        pass

    return SelfTestResult(
        covers_all_failure_modes=len(uncovered) == 0,
        uncovered_modes=list(uncovered),
        redundant_dimensions=redundant,
        suggestions=[],
        confidence_score=0.5
    )


async def build_eval_prompt(
    system_prompt: str,
    eval_dimensions: List[EvalDimension],
    failure_modes: List[FailureMode],
    analysis: Dict[str, Any],
    use_case: str
) -> str:
    """
    Step 5: Build the final evaluation prompt from the dimensions.
    Uses the actual template variable names from the system prompt (e.g., {{callTranscript}})
    instead of generic {{input}}/{{output}}.
    """
    deep = analysis.get("deep", {})

    # Extract the primary input variable from the system prompt
    input_var_name = get_primary_input_variable(system_prompt)
    output_var_name = "output"  # Response is always 'output'

    logger.info(f"Using input variable: {{{{{input_var_name}}}}} (extracted from system prompt)")

    # Build dimension sections
    dimension_sections = []
    for i, dim in enumerate(eval_dimensions, 1):
        section = f"""### {i}. {dim.name}
**Description:** {dim.description}
**What to Check:** {dim.what_to_check}

**Scoring Rubric:**
- **5 (Excellent):** {dim.score_5}
- **3 (Acceptable):** {dim.score_3}
- **1 (Poor):** {dim.score_1}"""
        dimension_sections.append(section)

    # Build failure modes reference
    critical_failures = [f for f in failure_modes if f.severity == "critical"]
    major_failures = [f for f in failure_modes if f.severity == "major"]

    eval_prompt = f"""# Evaluation Prompt

## I. Evaluator's Role & Goal

You are an expert evaluator assessing the quality of LLM outputs for the following use case:
**{use_case}**

Your goal is to evaluate the OUTPUT QUALITY - not to re-do the task yourself. Focus on:
- Whether the output meets the requirements
- Whether the output is accurate and well-formed
- Whether the output would be useful to the end user

## II. Information Provided for Evaluation

You will receive:
- **{{{{{input_var_name}}}}}**: The user's input/query that was sent to the system
- **{{{{{output_var_name}}}}}**: The system's response that you must evaluate

## III. Core Expectations

The output must meet these foundational standards:
1. **Accuracy**: Information must be correct and grounded in the input
2. **Completeness**: All required elements must be present
3. **Format Compliance**: Output must match expected format
4. **Safety**: No harmful, biased, or inappropriate content
5. **Relevance**: Output must address the actual request

## IV. Critical Failure Modes (Auto-Fail)

The following issues should result in a FAIL verdict:
{chr(10).join(f'- **{f.name}**: {f.description}' for f in critical_failures) if critical_failures else '- No critical failure modes defined'}

## V. Evaluation Dimensions

{chr(10).join(dimension_sections)}

## VI. Evaluation Task

Follow these steps:
1. Read the {{{{{input_var_name}}}}} carefully to understand what was requested
2. Read the {{{{{output_var_name}}}}} and assess it against each dimension
3. Check for any critical failure modes (auto-fail)
4. Assign a score (1-5) for each dimension with brief rationale
5. Calculate overall score and determine verdict

## VII. Output Format

Return your evaluation as JSON:
```json
{{
  "score": <overall 1-5 score>,
  "reasoning": "<2-3 sentence overall assessment>",
  "dimension_scores": {{
{chr(10).join(f'    "{dim.name}": {{ "score": <1-5>, "comment": "<brief rationale>" }},' for dim in eval_dimensions[:-1])}
    "{eval_dimensions[-1].name}": {{ "score": <1-5>, "comment": "<brief rationale>" }}
  }},
  "critical_issues": ["<any critical failures found>"],
  "verdict": "<PASS | NEEDS_REVIEW | FAIL>"
}}
```

## VIII. Decision Logic

- **PASS**: Overall score >= 4.0 AND no critical issues
- **NEEDS_REVIEW**: Overall score >= 3.0 AND no critical issues, but some dimensions scored low
- **FAIL**: Overall score < 3.0 OR any critical issue detected

## IX. Evaluator Rules

- DO NOT re-do the task or provide your own answer
- Base judgments ONLY on observable evidence in the output
- If something is ambiguous, note it but don't assume the worst
- Be consistent - similar outputs should receive similar scores"""

    return eval_prompt


async def agentic_eval_generation(
    system_prompt: str,
    use_case: str,
    requirements: str,
    llm_client,
    provider: str,
    api_key: str,
    model_name: str,
    thinking_model: Optional[str] = None,
    max_iterations: int = 2
) -> AgenticEvalResult:
    """
    Main agentic eval generation function.
    Orchestrates the multi-step workflow with thinking model support.
    """
    steps_taken = []
    analysis_model = thinking_model or model_name

    # Step 1: Deep Analysis
    logger.info(f"Step 1: Deep prompt analysis with {analysis_model}")
    steps_taken.append({"step": "analysis", "model": analysis_model, "status": "started"})

    analysis = await deep_prompt_analysis(
        system_prompt=system_prompt,
        use_case=use_case,
        requirements=requirements,
        llm_client=llm_client,
        provider=provider,
        api_key=api_key,
        model_name=analysis_model
    )

    steps_taken[-1]["status"] = "completed"
    steps_taken[-1]["result"] = {
        "core_function": analysis.get("deep", {}).get("core_function", "Unknown"),
        "failure_modes_found": len(analysis.get("deep", {}).get("likely_failure_modes", []))
    }

    # Step 2: Identify Failure Modes
    logger.info(f"Step 2: Identifying failure modes with {analysis_model}")
    steps_taken.append({"step": "failure_modes", "model": analysis_model, "status": "started"})

    failure_modes = await identify_failure_modes(
        system_prompt=system_prompt,
        analysis=analysis,
        llm_client=llm_client,
        provider=provider,
        api_key=api_key,
        model_name=analysis_model
    )

    steps_taken[-1]["status"] = "completed"
    steps_taken[-1]["result"] = {
        "total_modes": len(failure_modes),
        "critical": len([f for f in failure_modes if f.severity == "critical"]),
        "major": len([f for f in failure_modes if f.severity == "major"])
    }

    # Step 3: Generate Eval Dimensions
    logger.info(f"Step 3: Generating eval dimensions with {model_name}")
    steps_taken.append({"step": "dimensions", "model": model_name, "status": "started"})

    eval_dimensions = await generate_eval_dimensions(
        system_prompt=system_prompt,
        failure_modes=failure_modes,
        analysis=analysis,
        llm_client=llm_client,
        provider=provider,
        api_key=api_key,
        model_name=model_name
    )

    steps_taken[-1]["status"] = "completed"
    steps_taken[-1]["result"] = {
        "dimensions_created": len(eval_dimensions)
    }

    # Iteration loop for self-test and refinement
    iteration = 0
    final_self_test = None

    while iteration < max_iterations:
        iteration += 1

        # Step 4: Self-Test
        logger.info(f"Step 4: Self-testing criteria (iteration {iteration})")
        steps_taken.append({"step": "self_test", "iteration": iteration, "status": "started"})

        self_test = await self_test_criteria(
            failure_modes=failure_modes,
            eval_dimensions=eval_dimensions,
            llm_client=llm_client,
            provider=provider,
            api_key=api_key,
            model_name=model_name
        )

        steps_taken[-1]["status"] = "completed"
        steps_taken[-1]["result"] = {
            "covers_all": self_test.covers_all_failure_modes,
            "uncovered": self_test.uncovered_modes,
            "confidence": self_test.confidence_score
        }

        final_self_test = self_test

        # If good coverage and high confidence, we're done
        if self_test.covers_all_failure_modes and self_test.confidence_score >= 0.8:
            logger.info("Self-test passed with high confidence")
            break

        # If uncovered modes, add dimensions for them
        if self_test.uncovered_modes and iteration < max_iterations:
            logger.info(f"Adding dimensions for uncovered modes: {self_test.uncovered_modes}")
            steps_taken.append({"step": "refinement", "iteration": iteration, "status": "started"})

            # Find the failure modes that aren't covered
            uncovered_failure_modes = [f for f in failure_modes if f.name in self_test.uncovered_modes]

            if uncovered_failure_modes:
                additional_dimensions = await generate_eval_dimensions(
                    system_prompt=system_prompt,
                    failure_modes=uncovered_failure_modes,
                    analysis=analysis,
                    llm_client=llm_client,
                    provider=provider,
                    api_key=api_key,
                    model_name=model_name
                )
                eval_dimensions.extend(additional_dimensions)

            steps_taken[-1]["status"] = "completed"
            steps_taken[-1]["result"] = {"added_dimensions": len(additional_dimensions) if uncovered_failure_modes else 0}

    # Step 5: Build Final Eval Prompt
    logger.info("Step 5: Building final eval prompt")
    steps_taken.append({"step": "build_prompt", "status": "started"})

    eval_prompt = await build_eval_prompt(
        system_prompt=system_prompt,
        eval_dimensions=eval_dimensions,
        failure_modes=failure_modes,
        analysis=analysis,
        use_case=use_case
    )

    steps_taken[-1]["status"] = "completed"

    # Extract criteria names for response
    eval_criteria = [dim.name for dim in eval_dimensions]

    # Build rationale
    rationale = f"Generated {len(eval_dimensions)} evaluation dimensions covering {len(failure_modes)} identified failure modes. "
    if final_self_test:
        rationale += f"Self-test confidence: {final_self_test.confidence_score:.0%}. "
        if final_self_test.suggestions:
            rationale += f"Suggestions: {', '.join(final_self_test.suggestions[:2])}"

    return AgenticEvalResult(
        eval_prompt=eval_prompt,
        eval_criteria=eval_criteria,
        rationale=rationale,
        failure_modes=[
            {
                "name": f.name,
                "description": f.description,
                "severity": f.severity,
                "detection_criteria": f.detection_criteria
            }
            for f in failure_modes
        ],
        eval_dimensions=[
            {
                "name": d.name,
                "description": d.description,
                "what_to_check": d.what_to_check,
                "failure_modes_covered": d.failure_modes_covered
            }
            for d in eval_dimensions
        ],
        self_test={
            "covers_all_failure_modes": final_self_test.covers_all_failure_modes if final_self_test else False,
            "uncovered_modes": final_self_test.uncovered_modes if final_self_test else [],
            "confidence_score": final_self_test.confidence_score if final_self_test else 0,
            "suggestions": final_self_test.suggestions if final_self_test else []
        },
        iterations=iteration,
        steps_taken=steps_taken
    )


def result_to_dict(result: AgenticEvalResult) -> Dict[str, Any]:
    """Convert AgenticEvalResult to dictionary for JSON serialization"""
    return {
        "eval_prompt": result.eval_prompt,
        "eval_criteria": result.eval_criteria,
        "rationale": result.rationale,
        "failure_modes": result.failure_modes,
        "eval_dimensions": result.eval_dimensions,
        "self_test": result.self_test,
        "iterations": result.iterations,
        "steps_taken": result.steps_taken
    }
