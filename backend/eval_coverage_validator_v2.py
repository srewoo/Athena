"""
Eval Coverage Validator V2 - Enhanced with Thinking Models

IMPROVEMENTS OVER V1:
1. Better structured prompts with reasoning guidance
2. Support for thinking models (o1-mini, o1-preview, o3-mini)
3. Chain-of-thought prompting for better analysis
4. Examples and edge cases in prompts
5. Configurable model selection per step
6. More robust coverage detection

Author: Athena Team
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import base classes from V1
from eval_coverage_validator import (
    CoverageLevel,
    SystemPromptAnalysis,
    CoverageGap,
    CoverageAnalysisResult,
    EvalImprovementSuggestion
)

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for which models to use for each step"""

    # System prompt analysis: Extraction task, fast model is fine
    analysis_model: str = "gpt-4o-mini"
    analysis_temperature: float = 0.1

    # Coverage analysis: REASONING task - use thinking model!
    coverage_model: str = "o1-mini"  # or "o3-mini", "o1-preview"
    coverage_temperature: float = 1.0  # o1 models ignore temperature but set default

    # Improvement generation: Creative + reasoning task - thinking model helps
    improvement_model: str = "o1-mini"  # or "gpt-4o" for faster/cheaper
    improvement_temperature: float = 1.0

    # Application: Generation task, standard model is fine
    application_model: str = "gpt-4o"
    application_temperature: float = 0.3

    # Whether to use thinking models (o1/o3 series)
    use_thinking_models: bool = True

    def is_thinking_model(self, model_name: str) -> bool:
        """Check if a model is a thinking model (o1/o3 series)"""
        thinking_prefixes = ["o1-", "o3-", "o1", "o3"]
        return any(model_name.startswith(prefix) for prefix in thinking_prefixes)


# =============================================================================
# SECTION 2: ENHANCED PROMPTS
# =============================================================================

class EnhancedPrompts:
    """
    Enhanced prompts with better structure, reasoning guidance, and examples.
    """

    @staticmethod
    def get_analysis_prompt(system_prompt: str) -> str:
        """
        Enhanced system prompt analysis with structured extraction.

        This is an extraction task, so standard model is fine.
        """
        return f"""You are analyzing a system prompt to extract ALL testable requirements that an evaluation prompt must verify.

# SYSTEM PROMPT TO ANALYZE:
{system_prompt}

# YOUR TASK:
Extract comprehensive requirements in the following categories. Be thorough - missing a requirement means the eval won't test it.

## 1. KEY CAPABILITIES (What the system MUST be able to do)
Look for:
- Explicit action verbs: "must analyze", "should generate", "can process"
- Responsibilities: "your task is to...", "you are responsible for..."
- Functional requirements: "supports X", "handles Y"

Examples:
- "Process customer refund requests" → Capability
- "Translate text between languages" → Capability
- "Generate SQL queries from natural language" → Capability

## 2. CRITICAL CONSTRAINTS (What the system MUST NOT do / MUST avoid)
Look for:
- Prohibitions: "never", "do not", "must not", "avoid", "prohibited"
- Safety/security rules: "don't share PII", "don't execute code"
- Limitations: "only for X", "restricted to Y"

Examples:
- "Never share customer data" → Critical constraint (must be auto-fail)
- "Do not execute system commands" → Critical constraint
- "Only use approved data sources" → Critical constraint

## 3. QUALITY CRITERIA (Attributes the output must have)
Look for:
- Quality descriptors: "accurate", "clear", "concise", "professional"
- Standards: "follows PEP 8", "grammatically correct"
- Tone/style: "empathetic", "formal", "friendly"

Examples:
- "Must be accurate and factual" → Quality criterion
- "Tone should be professional" → Quality criterion
- "Responses must be concise" → Quality criterion

## 4. OUTPUT REQUIREMENTS (Format, structure, schema)
Look for:
- Format specifications: "JSON", "XML", "Markdown"
- Schema requirements: specific fields, structure
- Content requirements: "must include disclaimer", "cite sources"

Examples:
- "Output valid JSON with fields: name, age, email" → Output requirement
- "Include disclaimer at the end" → Output requirement
- "Must cite sources for claims" → Output requirement

## 5. INPUT EXPECTATIONS (What inputs to handle)
Look for:
- Input types: "customer questions", "code snippets", "images"
- Edge cases: "handle empty input", "process multiple languages"

## 6. EDGE CASES (Special scenarios or challenging cases)
Look for:
- "If X then Y" conditions
- "Handle edge case where..."
- Ambiguous or adversarial inputs mentioned

## 7. DOMAIN-SPECIFIC TERMS
Extract important terminology that evaluators need to understand.

# OUTPUT FORMAT (JSON only):
{{
  "primary_purpose": "One sentence summary of main purpose (max 200 chars)",
  "key_capabilities": ["capability 1", "capability 2", ...],
  "critical_constraints": ["constraint 1", "constraint 2", ...],
  "quality_criteria": ["criterion 1", "criterion 2", ...],
  "input_expectations": ["input type 1", "input type 2", ...],
  "output_requirements": ["requirement 1", "requirement 2", ...],
  "edge_cases": ["edge case 1", "edge case 2", ...],
  "domain_specific_terms": ["term 1", "term 2", ...]
}}

# IMPORTANT:
- Be comprehensive - extract ALL requirements
- Use direct quotes where possible
- Don't infer requirements not explicitly stated
- Constraints are CRITICAL - don't miss any "never"/"don't" statements

Output ONLY the JSON, no other text."""

    @staticmethod
    def get_coverage_analysis_prompt_thinking(
        system_analysis: SystemPromptAnalysis,
        eval_prompt: str,
        dimensions: List[Dict[str, Any]],
        auto_fails: List[Dict[str, Any]]
    ) -> str:
        """
        Enhanced coverage analysis prompt optimized for thinking models (o1/o3).

        Thinking models benefit from:
        - Clear problem statement
        - Structured reasoning steps
        - Examples of edge cases to consider
        - No explicit "think step by step" (they do this automatically)
        """

        return f"""You are a meta-evaluator analyzing whether an eval prompt comprehensively tests a system prompt's requirements.

# SYSTEM REQUIREMENTS (What MUST be tested):

Primary Purpose: {system_analysis.primary_purpose}

## Key Capabilities (MUST have dimensions testing these):
{chr(10).join(f"{i+1}. {cap}" for i, cap in enumerate(system_analysis.key_capabilities[:10]))}

## Critical Constraints (MUST have auto-fail conditions enforcing these):
{chr(10).join(f"{i+1}. {cons}" for i, cons in enumerate(system_analysis.critical_constraints[:10]))}

## Quality Criteria (MUST be evaluated in rubrics):
{chr(10).join(f"{i+1}. {crit}" for i, crit in enumerate(system_analysis.quality_criteria[:8]))}

## Output Requirements (MUST validate format/structure):
{chr(10).join(f"{i+1}. {req}" for i, req in enumerate(system_analysis.output_requirements[:8]))}

---

# CURRENT EVAL PROMPT:
{eval_prompt[:4000]}

# CURRENT DIMENSIONS:
{json.dumps([{{"name": d.get("name"), "description": d.get("description", "")[:100], "weight": d.get("weight")}} for d in dimensions[:15]], indent=2)}

# CURRENT AUTO-FAILS:
{json.dumps([{{"name": af.get("name"), "description": af.get("description", "")[:100]}} for af in auto_fails[:15]], indent=2)}

---

# YOUR TASK:

Analyze how comprehensively the eval prompt covers the system requirements. For each requirement, determine:

1. **Is it tested?** Does a dimension, rubric criterion, or auto-fail check this?
2. **How well?** Is the test explicit and specific, or vague and indirect?
3. **What's missing?** If not tested, what specific addition would cover it?

## Coverage Analysis Framework:

### For Capabilities:
- ✓ COVERED: Specific dimension evaluates this capability with clear rubric
- ~ PARTIAL: Mentioned in eval but not explicitly scored/tested
- ✗ MISSING: Not mentioned or tested at all

### For Constraints:
- ✓ COVERED: Auto-fail condition enforces this constraint
- ~ PARTIAL: Mentioned in dimensions but not auto-fail (should be auto-fail!)
- ✗ MISSING: Not enforced anywhere

### For Quality Criteria:
- ✓ COVERED: Rubric scores output on this criterion
- ~ PARTIAL: Implicitly related to a dimension but not explicit
- ✗ MISSING: No evaluation of this criterion

### For Output Requirements:
- ✓ COVERED: Format/structure validation exists (auto-fail or dimension)
- ~ PARTIAL: Format mentioned but validation not specific
- ✗ MISSING: No format checking

## Edge Cases to Consider:

1. **Semantic vs. Keyword Coverage**:
   - Don't just match keywords!
   - "Check if helpful" doesn't cover "Must provide actionable advice"
   - "Professional tone" might cover "Empathetic and supportive" if rubric is specific

2. **Constraint Severity**:
   - ALL constraints should be auto-fails (failing them = system failure)
   - Constraints in dimensions (not auto-fails) = PARTIAL coverage only

3. **Implicit Coverage**:
   - "Overall Quality" dimension doesn't explicitly cover specific quality criteria
   - Need specific rubrics for specific criteria

4. **Format Validation Depth**:
   - "Check JSON format" = PARTIAL if no schema validation
   - Need explicit field validation for full coverage

## Output Format (JSON only):

{{
  "coverage_percentage": 0-100,
  "reasoning": "Your detailed analysis of what's covered and what's missing",
  "gaps": [
    {{
      "category": "capability|constraint|quality_criteria|output_requirement",
      "missing_aspect": "exact requirement from system analysis",
      "severity": "critical|high|medium|low",
      "current_coverage": "description of any partial coverage",
      "recommended_addition": "specific dimension/auto-fail to add",
      "example_test_case": "concrete example that would catch this gap"
    }}
  ],
  "well_covered": ["list of requirements that ARE well tested"],
  "improvement_priority": ["ordered list of top 5 gaps to fix first"]
}}

## Severity Guidelines:
- **critical**: Constraint not enforced (safety/security risk)
- **high**: Key capability not tested OR output requirement missing
- **medium**: Quality criterion not evaluated
- **low**: Edge case or minor aspect not covered

Calculate coverage_percentage as: (well_covered_count / total_requirements) × 100

Output ONLY the JSON."""

    @staticmethod
    def get_coverage_analysis_prompt_standard(
        system_analysis: SystemPromptAnalysis,
        eval_prompt: str,
        dimensions: List[Dict[str, Any]],
        auto_fails: List[Dict[str, Any]]
    ) -> str:
        """
        Coverage analysis prompt for standard models (with explicit reasoning guidance).
        """

        prompt = EnhancedPrompts.get_coverage_analysis_prompt_thinking(
            system_analysis, eval_prompt, dimensions, auto_fails
        )

        # Add explicit reasoning instruction for standard models
        reasoning_addition = """

# REASONING PROCESS (think through each step):

Step 1: List all requirements from system analysis (capabilities + constraints + criteria + output reqs)
Step 2: For each requirement, check if eval prompt has a specific test for it
Step 3: Classify coverage as COVERED (✓), PARTIAL (~), or MISSING (✗)
Step 4: For MISSING or PARTIAL, determine severity and specific fix needed
Step 5: Calculate coverage percentage
Step 6: Prioritize gaps by severity and impact

Work through this systematically before generating the JSON output.

"""

        # Insert before "Output Format"
        prompt = prompt.replace("## Output Format (JSON only):",
                               reasoning_addition + "## Output Format (JSON only):")

        return prompt

    @staticmethod
    def get_improvement_generation_prompt_thinking(
        coverage_result: CoverageAnalysisResult,
        eval_prompt: str,
        dimensions: List[Dict[str, Any]],
        auto_fails: List[Dict[str, Any]]
    ) -> str:
        """
        Enhanced improvement generation for thinking models.

        This is a creative + reasoning task that benefits from thinking models.
        """

        gaps_json = json.dumps([
            {
                "category": g.category,
                "missing": g.missing_aspect,
                "severity": g.severity,
                "current_coverage": g.current_coverage,
                "recommended": g.recommended_addition
            }
            for g in coverage_result.gaps[:15]
        ], indent=2)

        return f"""You are an expert at improving evaluation prompts to maximize coverage of system requirements.

# CURRENT STATE:

## Coverage Metrics:
- Overall Coverage: {coverage_result.overall_coverage_pct}%
- Coverage Level: {coverage_result.coverage_level.value}
- Target Threshold: 80%
- Gap: {max(0, 80 - coverage_result.overall_coverage_pct)}% to reach threshold

## Identified Gaps:
{gaps_json}

## Current Eval Structure:
- Dimensions: {len(dimensions)}
- Auto-fails: {len(auto_fails)}
- Well-covered aspects: {len(coverage_result.well_covered_aspects)}

## Well-Covered (don't modify these):
{chr(10).join(f"- {aspect}" for aspect in coverage_result.well_covered_aspects[:10])}

---

# YOUR TASK:

Generate specific, actionable improvements to close the coverage gaps. Focus on high-impact changes that maximize coverage increase.

## Improvement Strategy:

### 1. Prioritization (address in this order):
   a. **Critical gaps** (severity=critical): Constraints not enforced → Add auto-fails
   b. **High gaps** (severity=high): Key capabilities not tested → Add dimensions
   c. **Medium gaps**: Quality criteria missing → Enhance rubrics
   d. **Low gaps**: Edge cases → Add to rubrics/examples

### 2. Specificity Requirements:
   - ✓ GOOD: "Add auto-fail 'pii_leak': Response contains email/phone/SSN patterns"
   - ✗ VAGUE: "Add privacy check"

   - ✓ GOOD: "Add dimension 'Order Tracking Accuracy' (20% weight): Score 5 if correctly retrieves order status from valid ID, Score 1 if fails to validate ID"
   - ✗ VAGUE: "Test order tracking"

### 3. Integration Guidelines:
   - New dimensions should have clear rubrics (1-5 scale with criteria)
   - Auto-fails should have detection patterns (what triggers the fail)
   - Weight adjustments if adding dimensions (must sum to 100%)
   - Preserve well-covered aspects (don't remove what works)

### 4. Coverage Impact:
   - Each improvement should close at least one gap
   - Prioritize improvements that close multiple related gaps
   - Aim for 80%+ coverage with minimal additions

## Edge Cases to Handle:

1. **Overlapping Coverage**:
   - If gap is "validate JSON schema" but you add "format compliance" dimension, ensure schema validation is explicit in rubric

2. **Constraint → Auto-Fail**:
   - ALL constraints MUST become auto-fails (not dimensions)
   - Dimension for constraint = still a gap!

3. **Capability → Dimension**:
   - Each key capability needs its own dimension OR clear rubric criteria in existing dimension
   - Generic "quality" dimension doesn't cover specific capabilities

## Output Format (JSON only):

{{
  "reasoning": "Your analysis of which improvements will have highest impact",
  "improvements": [
    {{
      "section_to_modify": "dimensions|auto_fails|rubrics|calibration_examples",
      "action": "add|modify|clarify",
      "specific_change": "Exact text/structure to add (be very specific!)",
      "reason": "Which gap(s) this addresses and why",
      "priority": 1-5 (1=highest, must fix; 5=nice to have),
      "estimated_coverage_gain": 5-20 (percentage points this will add)
    }}
  ],
  "expected_new_coverage": 85.0
}}

## Constraints:
- Maximum 8 improvements per iteration (focus on highest impact)
- Each improvement must close at least one gap
- Be specific enough that improvements can be directly applied
- Priority 1-2 improvements should get coverage to 80%+

Output ONLY the JSON."""

    @staticmethod
    def get_improvement_application_prompt(
        eval_prompt: str,
        improvements: List[EvalImprovementSuggestion],
        system_prompt: str
    ) -> str:
        """
        Enhanced prompt for applying improvements to eval prompt.

        This is a generation task, standard model works well.
        """

        improvements_text = "\n".join([
            f"""
## Improvement {i+1} [Priority {imp.priority}]:
**Section:** {imp.section_to_modify}
**Action:** {imp.action.upper()}
**Specific Change:**
{imp.specific_change}

**Reason:** {imp.reason}
"""
            for i, imp in enumerate(improvements[:10])
        ])

        return f"""You are refining an evaluation prompt to ensure comprehensive coverage of system requirements.

# CONTEXT:

## Original System Prompt (what the eval must test):
{system_prompt[:2500]}

## Current Eval Prompt (needs improvement):
{eval_prompt[:4000]}

---

# REQUIRED IMPROVEMENTS:

{improvements_text}

---

# YOUR TASK:

Apply ALL improvements to create an enhanced eval prompt. Requirements:

## 1. Integration (NOT replacement):
- Keep all well-functioning parts of current eval
- Add/modify only what's needed for improvements
- Maintain existing structure and flow
- Preserve good dimension definitions, rubrics, examples

## 2. Consistency:
- If adding dimensions, ensure rubrics are detailed (what distinguishes 5 vs 4 vs 3 etc)
- If adding auto-fails, include clear detection criteria
- Update dimension weights to sum to 100% if changed
- Use consistent terminology with system prompt

## 3. Clarity:
- Each dimension should have 1-5 scale with specific criteria
- Each auto-fail should explain what triggers it
- Use examples where helpful
- Avoid vague language ("good", "appropriate" without definition)

## 4. Format Preservation:
- Keep the same overall structure (dimensions, rubrics, auto-fails, etc)
- Maintain any existing calibration examples
- Preserve output format specifications
- Keep any consistency requirements

## 5. Completeness:
- Apply ALL improvements listed above (don't skip any)
- If improvement says "add dimension X with rubric Y", include the full rubric
- If adding auto-fail, include detection pattern
- Verify all critical constraints become auto-fails

## Example Transformations:

### Before (missing capability):
Dimensions:
1. Helpfulness (50%)
2. Clarity (50%)

### After (added improvement "test order tracking"):
Dimensions:
1. Order Tracking Accuracy (35%):
   - 5: Correctly retrieves and displays order status from valid ID
   - 4: Retrieves order but minor display issues
   - 3: Validates order ID but incomplete status info
   - 2: Attempts to check but fails validation
   - 1: Doesn't attempt order tracking
2. Helpfulness (35%)
3. Clarity (30%)

### Before (constraint not enforced):
Auto-fails:
- None

### After (added improvement "enforce no-PII constraint"):
Auto-fails:
- pii_leak: Response contains personal identifiable information (email, phone, SSN, address patterns)
- ...

---

# OUTPUT:

Return the COMPLETE improved eval prompt (not just the changes). The output should be a fully functional eval prompt ready to use.

Do NOT include explanations, markdown headers like "# Improved Eval Prompt", or meta-commentary. Output ONLY the eval prompt text itself."""


# =============================================================================
# SECTION 3: ENHANCED ANALYZER WITH MODEL SELECTION
# =============================================================================

class SystemPromptAnalyzerV2:
    """Enhanced system prompt analyzer with better prompts"""

    def __init__(self, model_config: Optional[ModelConfig] = None):
        self.model_config = model_config or ModelConfig()

    async def analyze(self, system_prompt: str, llm_client) -> SystemPromptAnalysis:
        """Analyze system prompt using enhanced prompts"""

        prompt = EnhancedPrompts.get_analysis_prompt(system_prompt)

        try:
            # Use configured analysis model
            response = await llm_client.generate(
                prompt=prompt,
                model=self.model_config.analysis_model,
                temperature=self.model_config.analysis_temperature,
                max_tokens=3000
            )

            # Extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return SystemPromptAnalysis(**data)
            else:
                logger.warning("LLM analysis did not return valid JSON")
                raise ValueError("No JSON in response")

        except Exception as e:
            logger.error(f"Enhanced analysis failed: {e}")
            raise


class EvalCoverageAnalyzerV2:
    """Enhanced coverage analyzer with thinking model support"""

    COVERAGE_THRESHOLD = 80.0

    def __init__(self, model_config: Optional[ModelConfig] = None):
        self.model_config = model_config or ModelConfig()

    async def analyze_coverage(
        self,
        system_analysis: SystemPromptAnalysis,
        eval_prompt: str,
        dimensions: List[Dict[str, Any]],
        auto_fails: List[Dict[str, Any]],
        llm_client
    ) -> CoverageAnalysisResult:
        """Analyze coverage using enhanced prompts and thinking models"""

        # Choose prompt based on model type
        model_to_use = self.model_config.coverage_model
        is_thinking = self.model_config.is_thinking_model(model_to_use)

        if is_thinking:
            prompt = EnhancedPrompts.get_coverage_analysis_prompt_thinking(
                system_analysis, eval_prompt, dimensions, auto_fails
            )
            logger.info(f"Using thinking model for coverage analysis: {model_to_use}")
        else:
            prompt = EnhancedPrompts.get_coverage_analysis_prompt_standard(
                system_analysis, eval_prompt, dimensions, auto_fails
            )
            logger.info(f"Using standard model for coverage analysis: {model_to_use}")

        try:
            response = await llm_client.generate(
                prompt=prompt,
                model=model_to_use,
                temperature=self.model_config.coverage_temperature,
                max_tokens=4000
            )

            # Extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))

                gaps = [CoverageGap(**g) for g in data.get("gaps", [])]
                coverage_pct = float(data.get("coverage_percentage", 0))

                if coverage_pct >= 90:
                    coverage_level = CoverageLevel.COMPREHENSIVE
                elif coverage_pct >= 75:
                    coverage_level = CoverageLevel.GOOD
                elif coverage_pct >= 50:
                    coverage_level = CoverageLevel.PARTIAL
                else:
                    coverage_level = CoverageLevel.INSUFFICIENT

                logger.info(f"Coverage analysis complete: {coverage_pct}% ({coverage_level.value})")

                return CoverageAnalysisResult(
                    overall_coverage_pct=coverage_pct,
                    coverage_level=coverage_level,
                    gaps=gaps,
                    well_covered_aspects=data.get("well_covered", []),
                    improvement_priority=data.get("improvement_priority", []),
                    passes_threshold=coverage_pct >= self.COVERAGE_THRESHOLD
                )
            else:
                logger.error("Coverage analysis did not return valid JSON")
                raise ValueError("No JSON in response")

        except Exception as e:
            logger.error(f"Enhanced coverage analysis failed: {e}")
            raise


class EvalPromptImproverV2:
    """Enhanced improver with thinking model support"""

    def __init__(self, model_config: Optional[ModelConfig] = None):
        self.model_config = model_config or ModelConfig()

    async def generate_improvements(
        self,
        coverage_result: CoverageAnalysisResult,
        eval_prompt: str,
        dimensions: List[Dict[str, Any]],
        auto_fails: List[Dict[str, Any]],
        llm_client
    ) -> List[EvalImprovementSuggestion]:
        """Generate improvements using enhanced prompts"""

        model_to_use = self.model_config.improvement_model
        is_thinking = self.model_config.is_thinking_model(model_to_use)

        prompt = EnhancedPrompts.get_improvement_generation_prompt_thinking(
            coverage_result, eval_prompt, dimensions, auto_fails
        )

        logger.info(f"Generating improvements with model: {model_to_use} (thinking={is_thinking})")

        try:
            response = await llm_client.generate(
                prompt=prompt,
                model=model_to_use,
                temperature=self.model_config.improvement_temperature,
                max_tokens=3000
            )

            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                improvements = [
                    EvalImprovementSuggestion(**imp)
                    for imp in data.get("improvements", [])
                ]
                improvements.sort(key=lambda x: x.priority)
                logger.info(f"Generated {len(improvements)} improvements")
                return improvements
            else:
                logger.error("Improvement generation did not return valid JSON")
                raise ValueError("No JSON in response")

        except Exception as e:
            logger.error(f"Enhancement improvement generation failed: {e}")
            raise

    async def apply_improvements(
        self,
        eval_prompt: str,
        improvements: List[EvalImprovementSuggestion],
        system_prompt: str,
        llm_client
    ) -> str:
        """Apply improvements using enhanced prompts"""

        model_to_use = self.model_config.application_model

        prompt = EnhancedPrompts.get_improvement_application_prompt(
            eval_prompt, improvements, system_prompt
        )

        logger.info(f"Applying improvements with model: {model_to_use}")

        try:
            improved_prompt = await llm_client.generate(
                prompt=prompt,
                model=model_to_use,
                temperature=self.model_config.application_temperature,
                max_tokens=6000
            )

            # Validation
            if len(improved_prompt) > 500 and "evaluate" in improved_prompt.lower():
                logger.info(f"Successfully applied improvements (new length: {len(improved_prompt)} chars)")
                return improved_prompt
            else:
                logger.warning("Improved prompt seems invalid, returning original")
                return eval_prompt

        except Exception as e:
            logger.error(f"Failed to apply improvements: {e}")
            return eval_prompt


# =============================================================================
# SECTION 4: ENHANCED COVERAGE VALIDATOR
# =============================================================================

class CoverageValidatorV2:
    """
    Enhanced coverage validator with thinking model support and better prompts.
    """

    MAX_ITERATIONS = 3
    MIN_COVERAGE_THRESHOLD = 80.0

    def __init__(self, llm_client, model_config: Optional[ModelConfig] = None):
        self.llm_client = llm_client
        self.model_config = model_config or ModelConfig()

        self.system_analyzer = SystemPromptAnalyzerV2(self.model_config)
        self.coverage_analyzer = EvalCoverageAnalyzerV2(self.model_config)
        self.improver = EvalPromptImproverV2(self.model_config)

        logger.info(f"CoverageValidatorV2 initialized with config:")
        logger.info(f"  Analysis: {self.model_config.analysis_model}")
        logger.info(f"  Coverage: {self.model_config.coverage_model} (thinking={self.model_config.is_thinking_model(self.model_config.coverage_model)})")
        logger.info(f"  Improvement: {self.model_config.improvement_model} (thinking={self.model_config.is_thinking_model(self.model_config.improvement_model)})")
        logger.info(f"  Application: {self.model_config.application_model}")

    async def validate_and_improve(
        self,
        system_prompt: str,
        eval_prompt: str,
        dimensions: List[Dict[str, Any]],
        auto_fails: List[Dict[str, Any]],
        max_iterations: Optional[int] = None
    ) -> Tuple[str, CoverageAnalysisResult, List[Dict[str, Any]]]:
        """
        Validate and improve eval prompt using enhanced V2 system.

        Returns same format as V1 for compatibility.
        """

        max_iter = max_iterations or self.MAX_ITERATIONS
        iteration_history = []

        logger.info("=" * 60)
        logger.info("COVERAGE VALIDATION V2 - WITH THINKING MODELS")
        logger.info("=" * 60)

        # Step 1: Analyze system prompt
        logger.info("\n[Step 1/4] Analyzing system prompt requirements...")
        system_analysis = await self.system_analyzer.analyze(system_prompt, self.llm_client)

        logger.info(f"  Extracted {len(system_analysis.key_capabilities)} capabilities")
        logger.info(f"  Extracted {len(system_analysis.critical_constraints)} constraints")
        logger.info(f"  Extracted {len(system_analysis.quality_criteria)} quality criteria")

        current_eval = eval_prompt
        current_dims = dimensions
        current_fails = auto_fails

        for iteration in range(max_iter):
            logger.info(f"\n[Iteration {iteration + 1}/{max_iter}] Running coverage analysis...")

            # Step 2: Analyze coverage
            coverage_result = await self.coverage_analyzer.analyze_coverage(
                system_analysis=system_analysis,
                eval_prompt=current_eval,
                dimensions=current_dims,
                auto_fails=current_fails,
                llm_client=self.llm_client
            )

            iteration_info = {
                "iteration": iteration + 1,
                "coverage_pct": coverage_result.overall_coverage_pct,
                "coverage_level": coverage_result.coverage_level.value,
                "gaps_found": len(coverage_result.gaps),
                "critical_gaps": sum(1 for g in coverage_result.gaps if g.severity == "critical"),
                "passes_threshold": coverage_result.passes_threshold
            }
            iteration_history.append(iteration_info)

            logger.info(f"  Coverage: {coverage_result.overall_coverage_pct}%")
            logger.info(f"  Level: {coverage_result.coverage_level.value}")
            logger.info(f"  Gaps: {len(coverage_result.gaps)} ({iteration_info['critical_gaps']} critical)")

            # Check if threshold met
            if coverage_result.passes_threshold:
                logger.info(f"✓ Coverage threshold met! ({coverage_result.overall_coverage_pct}% >= {self.MIN_COVERAGE_THRESHOLD}%)")
                return current_eval, coverage_result, iteration_history

            # Check if last iteration
            if iteration == max_iter - 1:
                logger.warning(f"Max iterations reached. Final coverage: {coverage_result.overall_coverage_pct}%")
                return current_eval, coverage_result, iteration_history

            # Step 3: Generate improvements
            logger.info(f"\n[Step 3/4] Generating improvements for {len(coverage_result.gaps)} gaps...")
            improvements = await self.improver.generate_improvements(
                coverage_result=coverage_result,
                eval_prompt=current_eval,
                dimensions=current_dims,
                auto_fails=current_fails,
                llm_client=self.llm_client
            )

            iteration_info["improvements_generated"] = len(improvements)
            iteration_info["top_improvements"] = [
                {
                    "section": imp.section_to_modify,
                    "action": imp.action,
                    "priority": imp.priority,
                    "change": imp.specific_change[:80] + "..."
                }
                for imp in improvements[:3]
            ]

            logger.info(f"  Generated {len(improvements)} improvements")
            for imp in improvements[:5]:
                logger.info(f"    [{imp.priority}] {imp.section_to_modify}: {imp.specific_change[:60]}...")

            # Step 4: Apply improvements
            logger.info(f"\n[Step 4/4] Applying improvements...")
            current_eval = await self.improver.apply_improvements(
                eval_prompt=current_eval,
                improvements=improvements,
                system_prompt=system_prompt,
                llm_client=self.llm_client
            )

            logger.info(f"  Improvements applied. New eval prompt length: {len(current_eval)} chars")

        # Final coverage check
        logger.info(f"\n[Final Check] Running final coverage analysis...")
        final_coverage = await self.coverage_analyzer.analyze_coverage(
            system_analysis=system_analysis,
            eval_prompt=current_eval,
            dimensions=current_dims,
            auto_fails=current_fails,
            llm_client=self.llm_client
        )

        logger.info(f"  Final coverage: {final_coverage.overall_coverage_pct}%")
        logger.info("=" * 60)

        return current_eval, final_coverage, iteration_history


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'ModelConfig',
    'CoverageValidatorV2',
    'SystemPromptAnalyzerV2',
    'EvalCoverageAnalyzerV2',
    'EvalPromptImproverV2',
    'EnhancedPrompts'
]
