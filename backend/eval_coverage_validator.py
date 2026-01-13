"""
Eval Coverage Validator - Meta-Evaluator Agent

This module implements a "meta-evaluator" that validates whether a generated eval prompt
comprehensively covers all aspects of the system prompt's purpose.

Flow:
1. Parse system prompt to extract critical requirements, capabilities, and constraints
2. Analyze generated eval prompt for coverage of those aspects
3. Generate detailed gap report with specific missing elements
4. Provide actionable improvement suggestions
5. Iteratively refine eval prompt until coverage threshold is met (max 3 iterations)

Author: Athena Team
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: DATA MODELS
# =============================================================================

class CoverageLevel(Enum):
    """Coverage quality levels"""
    COMPREHENSIVE = "comprehensive"  # 90-100%
    GOOD = "good"  # 75-89%
    PARTIAL = "partial"  # 50-74%
    INSUFFICIENT = "insufficient"  # <50%


@dataclass
class SystemPromptAnalysis:
    """Analysis of system prompt requirements"""
    primary_purpose: str
    key_capabilities: List[str]
    critical_constraints: List[str]
    quality_criteria: List[str]
    input_expectations: List[str]
    output_requirements: List[str]
    edge_cases: List[str]
    domain_specific_terms: List[str]


@dataclass
class CoverageGap:
    """A gap in eval prompt coverage"""
    category: str  # e.g., "capability", "constraint", "quality_criteria"
    missing_aspect: str
    severity: str  # critical, high, medium, low
    current_coverage: str  # What's currently tested (if anything)
    recommended_addition: str
    example_test_case: str


@dataclass
class CoverageAnalysisResult:
    """Result of coverage analysis"""
    overall_coverage_pct: float
    coverage_level: CoverageLevel
    gaps: List[CoverageGap]
    well_covered_aspects: List[str]
    improvement_priority: List[str]  # Ordered list of what to fix first
    passes_threshold: bool  # True if coverage >= 80%


@dataclass
class EvalImprovementSuggestion:
    """Specific suggestion for improving eval prompt"""
    section_to_modify: str  # e.g., "dimensions", "auto_fails", "rubrics"
    action: str  # e.g., "add", "modify", "clarify"
    specific_change: str
    reason: str
    priority: int  # 1 (highest) to 5 (lowest)


# =============================================================================
# SECTION 2: SYSTEM PROMPT ANALYZER
# =============================================================================

class SystemPromptAnalyzer:
    """
    Analyzes system prompt to extract what the eval prompt MUST test.

    Uses both pattern matching and LLM-based analysis.
    """

    # Patterns for extracting requirements
    CAPABILITY_PATTERNS = [
        r"(?:you (?:can|should|must|will)|capable of|able to|responsible for)\s+([^.!?\n]+)",
        r"(?:task is to|your role is to|you are tasked with)\s+([^.!?\n]+)",
        r"(?:should|must|need to|required to)\s+([^.!?\n]+)",
    ]

    CONSTRAINT_PATTERNS = [
        r"(?:do not|don't|never|avoid|must not|should not)\s+([^.!?\n]+)",
        r"(?:only|exclusively|strictly)\s+([^.!?\n]+)",
        r"(?:prohibited|forbidden|not allowed)\s+([^.!?\n]+)",
    ]

    OUTPUT_FORMAT_PATTERNS = [
        r"(?:output|respond|return|provide|format|structure).*?(?:json|xml|yaml|markdown|csv)",
        r"(?:json|xml|yaml|markdown|csv)\s+format",
        r"schema:?\s*{[^}]+}",
    ]

    async def analyze(self, system_prompt: str, llm_client=None) -> SystemPromptAnalysis:
        """
        Analyze system prompt to extract requirements.

        Args:
            system_prompt: The system prompt to analyze
            llm_client: Optional LLM client for deeper analysis

        Returns:
            SystemPromptAnalysis with extracted requirements
        """
        # Use LLM for comprehensive analysis if available
        if llm_client:
            return await self._llm_analyze(system_prompt, llm_client)

        # Fallback to pattern-based analysis
        return self._pattern_analyze(system_prompt)

    def _pattern_analyze(self, system_prompt: str) -> SystemPromptAnalysis:
        """Pattern-based analysis (fallback when no LLM available)"""

        capabilities = []
        for pattern in self.CAPABILITY_PATTERNS:
            matches = re.finditer(pattern, system_prompt, re.IGNORECASE)
            for match in matches:
                cap = match.group(1).strip()
                if len(cap) > 10 and len(cap) < 200:
                    capabilities.append(cap)

        constraints = []
        for pattern in self.CONSTRAINT_PATTERNS:
            matches = re.finditer(pattern, system_prompt, re.IGNORECASE)
            for match in matches:
                constraint = match.group(1).strip()
                if len(constraint) > 10 and len(constraint) < 200:
                    constraints.append(constraint)

        # Detect output format requirements
        output_reqs = []
        for pattern in self.OUTPUT_FORMAT_PATTERNS:
            matches = re.finditer(pattern, system_prompt, re.IGNORECASE)
            for match in matches:
                output_reqs.append(match.group(0))

        # Extract quality criteria (keywords)
        quality_keywords = ["accurate", "precise", "clear", "concise", "comprehensive",
                           "relevant", "factual", "grounded", "evidence", "citation"]
        quality_criteria = []
        for keyword in quality_keywords:
            if re.search(rf"\b{keyword}\b", system_prompt, re.IGNORECASE):
                quality_criteria.append(f"Must be {keyword}")

        # Primary purpose (first sentence or first significant statement)
        sentences = re.split(r'[.!?]\s+', system_prompt)
        primary_purpose = sentences[0] if sentences else "Not clearly defined"

        return SystemPromptAnalysis(
            primary_purpose=primary_purpose[:300],
            key_capabilities=list(set(capabilities))[:10],
            critical_constraints=list(set(constraints))[:10],
            quality_criteria=quality_criteria[:8],
            input_expectations=self._extract_input_expectations(system_prompt),
            output_requirements=list(set(output_reqs))[:8],
            edge_cases=self._extract_edge_cases(system_prompt),
            domain_specific_terms=self._extract_domain_terms(system_prompt)
        )

    async def _llm_analyze(self, system_prompt: str, llm_client) -> SystemPromptAnalysis:
        """LLM-based deep analysis for comprehensive extraction"""

        analysis_prompt = f"""Analyze this system prompt and extract the following information in JSON format:

System Prompt:
{system_prompt}

Extract:
1. **primary_purpose**: One sentence describing the main purpose (max 200 chars)
2. **key_capabilities**: List of 5-10 key things the system must be able to do
3. **critical_constraints**: List of 5-10 things the system must NOT do or must avoid
4. **quality_criteria**: List of quality attributes the output must have (e.g., "accurate", "formatted correctly")
5. **input_expectations**: What types of inputs should the system handle?
6. **output_requirements**: Specific requirements for the output (format, structure, content)
7. **edge_cases**: Potential edge cases or challenging scenarios mentioned
8. **domain_specific_terms**: Important domain-specific terms or concepts

Return ONLY valid JSON in this format:
{{
  "primary_purpose": "string",
  "key_capabilities": ["string", ...],
  "critical_constraints": ["string", ...],
  "quality_criteria": ["string", ...],
  "input_expectations": ["string", ...],
  "output_requirements": ["string", ...],
  "edge_cases": ["string", ...],
  "domain_specific_terms": ["string", ...]
}}"""

        try:
            response = await llm_client.generate(
                prompt=analysis_prompt,
                temperature=0.1,
                max_tokens=2000
            )

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return SystemPromptAnalysis(**data)
            else:
                logger.warning("LLM analysis did not return valid JSON, falling back to pattern analysis")
                return self._pattern_analyze(system_prompt)

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}, falling back to pattern analysis")
            return self._pattern_analyze(system_prompt)

    def _extract_input_expectations(self, system_prompt: str) -> List[str]:
        """Extract what kinds of inputs are expected"""
        expectations = []

        input_patterns = [
            r"given\s+([^,.\n]+)",
            r"input\s+(?:will be|is|contains)\s+([^,.\n]+)",
            r"(?:receive|get|accept)\s+([^,.\n]+)\s+as input",
        ]

        for pattern in input_patterns:
            matches = re.finditer(pattern, system_prompt, re.IGNORECASE)
            for match in matches:
                exp = match.group(1).strip()
                if len(exp) > 5 and len(exp) < 150:
                    expectations.append(exp)

        return list(set(expectations))[:5]

    def _extract_edge_cases(self, system_prompt: str) -> List[str]:
        """Extract mentioned edge cases"""
        edge_cases = []

        edge_keywords = ["edge case", "special case", "exception", "if.*then",
                        "when.*should", "handle.*carefully", "pay attention"]

        for keyword in edge_keywords:
            pattern = rf"({keyword}[^.!?\n]{{10,200}})"
            matches = re.finditer(pattern, system_prompt, re.IGNORECASE)
            for match in matches:
                edge_cases.append(match.group(1).strip())

        return edge_cases[:5]

    def _extract_domain_terms(self, system_prompt: str) -> List[str]:
        """Extract important domain-specific terms"""
        # Find capitalized terms and technical terms
        terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', system_prompt)  # CamelCase
        terms.extend(re.findall(r'\b[A-Z]{2,}\b', system_prompt))  # ACRONYMS

        # Technical terms (basic patterns)
        technical_patterns = [
            r'\bAPI\b', r'\bJSON\b', r'\bXML\b', r'\bSQL\b', r'\bHTTP\b',
            r'\bREST\b', r'\bGQL\b', r'\bJWT\b', r'\bOAuth\b'
        ]

        for pattern in technical_patterns:
            if re.search(pattern, system_prompt, re.IGNORECASE):
                terms.append(pattern.strip(r'\b').strip(r'\\'))

        return list(set(terms))[:10]


# =============================================================================
# SECTION 3: COVERAGE ANALYZER
# =============================================================================

class EvalCoverageAnalyzer:
    """
    Analyzes eval prompt to determine how well it covers system prompt requirements.
    """

    COVERAGE_THRESHOLD = 80.0  # 80% coverage required to pass

    async def analyze_coverage(
        self,
        system_analysis: SystemPromptAnalysis,
        eval_prompt: str,
        dimensions: List[Dict[str, Any]],
        auto_fails: List[Dict[str, Any]],
        llm_client=None
    ) -> CoverageAnalysisResult:
        """
        Analyze how well eval prompt covers system requirements.

        Returns:
            CoverageAnalysisResult with gaps and recommendations
        """

        if llm_client:
            return await self._llm_coverage_analysis(
                system_analysis, eval_prompt, dimensions, auto_fails, llm_client
            )

        # Fallback to heuristic analysis
        return self._heuristic_coverage_analysis(
            system_analysis, eval_prompt, dimensions, auto_fails
        )

    def _heuristic_coverage_analysis(
        self,
        system_analysis: SystemPromptAnalysis,
        eval_prompt: str,
        dimensions: List[Dict[str, Any]],
        auto_fails: List[Dict[str, Any]]
    ) -> CoverageAnalysisResult:
        """Heuristic-based coverage analysis"""

        gaps = []
        covered = []
        eval_lower = eval_prompt.lower()

        total_aspects = 0
        covered_count = 0

        # Check capability coverage
        for capability in system_analysis.key_capabilities:
            total_aspects += 1
            # Extract key terms from capability
            key_terms = [w for w in capability.lower().split() if len(w) > 4][:3]

            if any(term in eval_lower for term in key_terms if term):
                covered_count += 1
                covered.append(f"Capability: {capability[:60]}")
            else:
                gaps.append(CoverageGap(
                    category="capability",
                    missing_aspect=capability,
                    severity="high",
                    current_coverage="Not explicitly tested",
                    recommended_addition=f"Add dimension or rubric criterion to test: {capability[:80]}",
                    example_test_case=f"Test case where system must {capability[:100]}"
                ))

        # Check constraint coverage
        for constraint in system_analysis.critical_constraints:
            total_aspects += 1
            key_terms = [w for w in constraint.lower().split() if len(w) > 4][:3]

            # Check if constraint is in auto-fails or dimensions
            constraint_tested = any(term in eval_lower for term in key_terms if term)

            if constraint_tested:
                covered_count += 1
                covered.append(f"Constraint: {constraint[:60]}")
            else:
                gaps.append(CoverageGap(
                    category="constraint",
                    missing_aspect=constraint,
                    severity="critical",
                    current_coverage="Not tested as auto-fail or dimension",
                    recommended_addition=f"Add auto-fail condition for: {constraint[:80]}",
                    example_test_case=f"Violates constraint: {constraint[:100]}"
                ))

        # Check quality criteria coverage
        for criterion in system_analysis.quality_criteria:
            total_aspects += 1
            criterion_lower = criterion.lower()

            if criterion_lower in eval_lower:
                covered_count += 1
                covered.append(f"Quality: {criterion}")
            else:
                gaps.append(CoverageGap(
                    category="quality_criteria",
                    missing_aspect=criterion,
                    severity="medium",
                    current_coverage="Not explicitly evaluated",
                    recommended_addition=f"Add dimension to evaluate {criterion}",
                    example_test_case=f"Output lacks {criterion}"
                ))

        # Check output requirements coverage
        for req in system_analysis.output_requirements:
            total_aspects += 1
            if any(term in eval_lower for term in req.lower().split()[:3]):
                covered_count += 1
                covered.append(f"Output requirement: {req[:60]}")
            else:
                gaps.append(CoverageGap(
                    category="output_requirement",
                    missing_aspect=req,
                    severity="high",
                    current_coverage="Format not validated",
                    recommended_addition=f"Add validation for output requirement: {req[:80]}",
                    example_test_case=f"Output doesn't meet requirement: {req[:100]}"
                ))

        # Calculate coverage percentage
        coverage_pct = (covered_count / total_aspects * 100) if total_aspects > 0 else 0

        # Determine coverage level
        if coverage_pct >= 90:
            coverage_level = CoverageLevel.COMPREHENSIVE
        elif coverage_pct >= 75:
            coverage_level = CoverageLevel.GOOD
        elif coverage_pct >= 50:
            coverage_level = CoverageLevel.PARTIAL
        else:
            coverage_level = CoverageLevel.INSUFFICIENT

        # Sort gaps by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        gaps.sort(key=lambda g: severity_order.get(g.severity, 4))

        # Improvement priority (top 5 gaps)
        improvement_priority = [
            f"{g.category}: {g.missing_aspect[:80]}" for g in gaps[:5]
        ]

        return CoverageAnalysisResult(
            overall_coverage_pct=round(coverage_pct, 1),
            coverage_level=coverage_level,
            gaps=gaps,
            well_covered_aspects=covered,
            improvement_priority=improvement_priority,
            passes_threshold=coverage_pct >= self.COVERAGE_THRESHOLD
        )

    async def _llm_coverage_analysis(
        self,
        system_analysis: SystemPromptAnalysis,
        eval_prompt: str,
        dimensions: List[Dict[str, Any]],
        auto_fails: List[Dict[str, Any]],
        llm_client
    ) -> CoverageAnalysisResult:
        """LLM-based comprehensive coverage analysis"""

        analysis_prompt = f"""You are a meta-evaluator. Your job is to check if an eval prompt comprehensively tests all aspects of a system prompt.

**System Prompt Requirements:**
Primary Purpose: {system_analysis.primary_purpose}

Key Capabilities (must be tested):
{chr(10).join(f"- {cap}" for cap in system_analysis.key_capabilities[:8])}

Critical Constraints (must be enforced):
{chr(10).join(f"- {cons}" for cons in system_analysis.critical_constraints[:8])}

Quality Criteria (must be evaluated):
{chr(10).join(f"- {crit}" for crit in system_analysis.quality_criteria[:6])}

Output Requirements:
{chr(10).join(f"- {req}" for req in system_analysis.output_requirements[:6])}

**Current Eval Prompt:**
{eval_prompt[:3000]}

**Current Dimensions:**
{json.dumps([{"name": d.get("name"), "weight": d.get("weight")} for d in dimensions[:10]], indent=2)}

**Current Auto-Fails:**
{json.dumps([{"name": af.get("name")} for af in auto_fails[:10]], indent=2)}

**Task:** Analyze coverage and identify gaps. Return JSON:

{{
  "coverage_percentage": 0-100,
  "gaps": [
    {{
      "category": "capability|constraint|quality_criteria|output_requirement",
      "missing_aspect": "what's missing",
      "severity": "critical|high|medium|low",
      "current_coverage": "what's currently tested (if anything)",
      "recommended_addition": "specific suggestion to add",
      "example_test_case": "example case that would catch this"
    }}
  ],
  "well_covered": ["aspects that are well tested"],
  "improvement_priority": ["top 5 things to fix in priority order"]
}}

Return ONLY valid JSON."""

        try:
            response = await llm_client.generate(
                prompt=analysis_prompt,
                temperature=0.1,
                max_tokens=3000
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

                return CoverageAnalysisResult(
                    overall_coverage_pct=coverage_pct,
                    coverage_level=coverage_level,
                    gaps=gaps,
                    well_covered_aspects=data.get("well_covered", []),
                    improvement_priority=data.get("improvement_priority", []),
                    passes_threshold=coverage_pct >= self.COVERAGE_THRESHOLD
                )
            else:
                logger.warning("LLM coverage analysis did not return valid JSON")
                return self._heuristic_coverage_analysis(
                    system_analysis, eval_prompt, dimensions, auto_fails
                )

        except Exception as e:
            logger.error(f"LLM coverage analysis failed: {e}")
            return self._heuristic_coverage_analysis(
                system_analysis, eval_prompt, dimensions, auto_fails
            )


# =============================================================================
# SECTION 4: EVAL PROMPT IMPROVER
# =============================================================================

class EvalPromptImprover:
    """
    Generates specific improvement suggestions and can iteratively refine eval prompts.
    """

    async def generate_improvements(
        self,
        coverage_result: CoverageAnalysisResult,
        eval_prompt: str,
        dimensions: List[Dict[str, Any]],
        auto_fails: List[Dict[str, Any]],
        llm_client=None
    ) -> List[EvalImprovementSuggestion]:
        """
        Generate specific, actionable improvements based on coverage gaps.

        Returns:
            List of improvement suggestions, ordered by priority
        """

        if llm_client:
            return await self._llm_generate_improvements(
                coverage_result, eval_prompt, dimensions, auto_fails, llm_client
            )

        return self._heuristic_generate_improvements(coverage_result)

    def _heuristic_generate_improvements(
        self,
        coverage_result: CoverageAnalysisResult
    ) -> List[EvalImprovementSuggestion]:
        """Generate improvements using heuristics"""

        suggestions = []
        priority = 1

        for gap in coverage_result.gaps[:10]:  # Top 10 gaps
            if gap.category == "constraint" and gap.severity in ["critical", "high"]:
                suggestions.append(EvalImprovementSuggestion(
                    section_to_modify="auto_fails",
                    action="add",
                    specific_change=f"Add auto-fail: '{gap.missing_aspect[:100]}'",
                    reason=f"Critical constraint not enforced: {gap.missing_aspect[:80]}",
                    priority=priority
                ))
                priority += 1

            elif gap.category == "capability" and gap.severity in ["high", "critical"]:
                suggestions.append(EvalImprovementSuggestion(
                    section_to_modify="dimensions",
                    action="add",
                    specific_change=f"Add dimension to test: '{gap.missing_aspect[:100]}'",
                    reason=f"Key capability not evaluated: {gap.missing_aspect[:80]}",
                    priority=priority
                ))
                priority += 1

            elif gap.category == "quality_criteria":
                suggestions.append(EvalImprovementSuggestion(
                    section_to_modify="rubrics",
                    action="modify",
                    specific_change=f"Update rubrics to emphasize: {gap.missing_aspect}",
                    reason=f"Quality criterion not measured: {gap.missing_aspect}",
                    priority=priority
                ))
                priority += 1

            elif gap.category == "output_requirement":
                suggestions.append(EvalImprovementSuggestion(
                    section_to_modify="auto_fails or dimensions",
                    action="add",
                    specific_change=f"Add validation for: {gap.missing_aspect[:100]}",
                    reason=f"Output requirement not checked: {gap.missing_aspect[:80]}",
                    priority=priority
                ))
                priority += 1

        return suggestions

    async def _llm_generate_improvements(
        self,
        coverage_result: CoverageAnalysisResult,
        eval_prompt: str,
        dimensions: List[Dict[str, Any]],
        auto_fails: List[Dict[str, Any]],
        llm_client
    ) -> List[EvalImprovementSuggestion]:
        """LLM-based improvement generation"""

        gaps_json = json.dumps([
            {
                "category": g.category,
                "missing": g.missing_aspect,
                "severity": g.severity,
                "recommendation": g.recommended_addition
            }
            for g in coverage_result.gaps[:10]
        ], indent=2)

        improvement_prompt = f"""You are an expert at improving evaluation prompts.

**Coverage Analysis:**
- Overall Coverage: {coverage_result.overall_coverage_pct}%
- Coverage Level: {coverage_result.coverage_level.value}
- Threshold: {EvalCoverageAnalyzer.COVERAGE_THRESHOLD}%

**Identified Gaps:**
{gaps_json}

**Current Eval Prompt Structure:**
- Dimensions: {len(dimensions)}
- Auto-fails: {len(auto_fails)}

**Task:** Generate specific, actionable improvements to address these gaps. Return JSON:

{{
  "improvements": [
    {{
      "section_to_modify": "dimensions|auto_fails|rubrics|calibration_examples",
      "action": "add|modify|clarify",
      "specific_change": "exact text to add or modification to make",
      "reason": "why this addresses a gap",
      "priority": 1-5 (1 is highest)
    }}
  ]
}}

Focus on the highest-severity gaps first. Be specific and actionable.

Return ONLY valid JSON."""

        try:
            response = await llm_client.generate(
                prompt=improvement_prompt,
                temperature=0.2,
                max_tokens=2000
            )

            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                improvements = [
                    EvalImprovementSuggestion(**imp)
                    for imp in data.get("improvements", [])
                ]
                # Sort by priority
                improvements.sort(key=lambda x: x.priority)
                return improvements
            else:
                logger.warning("LLM improvement generation did not return valid JSON")
                return self._heuristic_generate_improvements(coverage_result)

        except Exception as e:
            logger.error(f"LLM improvement generation failed: {e}")
            return self._heuristic_generate_improvements(coverage_result)

    async def apply_improvements(
        self,
        eval_prompt: str,
        improvements: List[EvalImprovementSuggestion],
        system_prompt: str,
        llm_client
    ) -> str:
        """
        Apply improvements to eval prompt by regenerating with guidance.

        This uses the LLM to incorporate improvements naturally.
        """

        improvements_text = "\n".join([
            f"{i+1}. [{imp.section_to_modify}] {imp.action.upper()}: {imp.specific_change} "
            f"(Reason: {imp.reason})"
            for i, imp in enumerate(improvements[:8])  # Top 8 improvements
        ])

        refinement_prompt = f"""You are refining an evaluation prompt to ensure comprehensive coverage.

**Original System Prompt:**
{system_prompt[:2000]}

**Current Eval Prompt:**
{eval_prompt[:3000]}

**Required Improvements:**
{improvements_text}

**Task:** Update the eval prompt to incorporate these improvements. Ensure:
1. All critical gaps are addressed
2. Improvements are integrated naturally
3. Existing good content is preserved
4. The prompt remains coherent and well-structured

Return the COMPLETE improved eval prompt (not just the changes)."""

        try:
            improved_prompt = await llm_client.generate(
                prompt=refinement_prompt,
                temperature=0.3,
                max_tokens=4000
            )

            # Basic validation that we got a substantial eval prompt back
            if len(improved_prompt) > 500 and "evaluate" in improved_prompt.lower():
                logger.info(f"Successfully applied {len(improvements)} improvements to eval prompt")
                return improved_prompt
            else:
                logger.warning("Improved prompt seems invalid, returning original")
                return eval_prompt

        except Exception as e:
            logger.error(f"Failed to apply improvements: {e}")
            return eval_prompt


# =============================================================================
# SECTION 5: INTEGRATED COVERAGE VALIDATOR
# =============================================================================

class CoverageValidator:
    """
    Main class that orchestrates the entire coverage validation and improvement process.
    """

    MAX_ITERATIONS = 3
    MIN_COVERAGE_THRESHOLD = 80.0

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.system_analyzer = SystemPromptAnalyzer()
        self.coverage_analyzer = EvalCoverageAnalyzer()
        self.improver = EvalPromptImprover()

    async def validate_and_improve(
        self,
        system_prompt: str,
        eval_prompt: str,
        dimensions: List[Dict[str, Any]],
        auto_fails: List[Dict[str, Any]],
        max_iterations: Optional[int] = None
    ) -> Tuple[str, CoverageAnalysisResult, List[Dict[str, Any]]]:
        """
        Validate eval prompt coverage and iteratively improve until threshold is met.

        Args:
            system_prompt: The system prompt to evaluate against
            eval_prompt: The generated eval prompt to validate
            dimensions: Evaluation dimensions
            auto_fails: Auto-fail conditions
            max_iterations: Max refinement iterations (default: 3)

        Returns:
            Tuple of (improved_eval_prompt, final_coverage_result, iteration_history)
        """

        max_iter = max_iterations or self.MAX_ITERATIONS
        iteration_history = []

        # Step 1: Analyze system prompt requirements
        logger.info("Analyzing system prompt requirements...")
        system_analysis = await self.system_analyzer.analyze(system_prompt, self.llm_client)

        current_eval = eval_prompt
        current_dims = dimensions
        current_fails = auto_fails

        for iteration in range(max_iter):
            logger.info(f"Coverage validation iteration {iteration + 1}/{max_iter}")

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
                "passes_threshold": coverage_result.passes_threshold
            }
            iteration_history.append(iteration_info)

            logger.info(
                f"Iteration {iteration + 1}: Coverage = {coverage_result.overall_coverage_pct}%, "
                f"Level = {coverage_result.coverage_level.value}, "
                f"Gaps = {len(coverage_result.gaps)}"
            )

            # Check if we've met the threshold
            if coverage_result.passes_threshold:
                logger.info(f"Coverage threshold met ({coverage_result.overall_coverage_pct}% >= {self.MIN_COVERAGE_THRESHOLD}%)")
                return current_eval, coverage_result, iteration_history

            # Check if this is the last iteration
            if iteration == max_iter - 1:
                logger.warning(
                    f"Max iterations reached. Final coverage: {coverage_result.overall_coverage_pct}%"
                )
                return current_eval, coverage_result, iteration_history

            # Step 3: Generate improvements
            logger.info(f"Generating improvements for {len(coverage_result.gaps)} gaps...")
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
                    "change": imp.specific_change[:100]
                }
                for imp in improvements[:3]
            ]

            # Step 4: Apply improvements
            if improvements and self.llm_client:
                logger.info(f"Applying {len(improvements)} improvements...")
                current_eval = await self.improver.apply_improvements(
                    eval_prompt=current_eval,
                    improvements=improvements,
                    system_prompt=system_prompt,
                    llm_client=self.llm_client
                )
            else:
                logger.warning("No LLM client available, cannot apply improvements automatically")
                break

        # Return final state (may not have met threshold)
        final_coverage = await self.coverage_analyzer.analyze_coverage(
            system_analysis=system_analysis,
            eval_prompt=current_eval,
            dimensions=current_dims,
            auto_fails=current_fails,
            llm_client=self.llm_client
        )

        return current_eval, final_coverage, iteration_history


# =============================================================================
# SECTION 6: CONVENIENCE FUNCTIONS
# =============================================================================

async def validate_eval_coverage(
    system_prompt: str,
    eval_prompt: str,
    dimensions: List[Dict[str, Any]] = None,
    auto_fails: List[Dict[str, Any]] = None,
    llm_client=None,
    auto_improve: bool = True,
    max_iterations: int = 3
) -> Dict[str, Any]:
    """
    Convenience function to validate eval prompt coverage.

    Args:
        system_prompt: The system prompt
        eval_prompt: The eval prompt to validate
        dimensions: Evaluation dimensions
        auto_fails: Auto-fail conditions
        llm_client: LLM client for deep analysis
        auto_improve: If True, iteratively improve until threshold met
        max_iterations: Max improvement iterations

    Returns:
        Dict with validation results and improved prompt
    """

    validator = CoverageValidator(llm_client=llm_client)

    if auto_improve:
        improved_eval, coverage_result, history = await validator.validate_and_improve(
            system_prompt=system_prompt,
            eval_prompt=eval_prompt,
            dimensions=dimensions or [],
            auto_fails=auto_fails or [],
            max_iterations=max_iterations
        )

        return {
            "original_eval_prompt": eval_prompt,
            "improved_eval_prompt": improved_eval,
            "final_coverage": {
                "percentage": coverage_result.overall_coverage_pct,
                "level": coverage_result.coverage_level.value,
                "passes_threshold": coverage_result.passes_threshold,
                "gaps_remaining": len(coverage_result.gaps),
                "well_covered": coverage_result.well_covered_aspects
            },
            "gaps": [
                {
                    "category": g.category,
                    "missing": g.missing_aspect,
                    "severity": g.severity,
                    "recommendation": g.recommended_addition
                }
                for g in coverage_result.gaps
            ],
            "iteration_history": history,
            "was_improved": improved_eval != eval_prompt
        }
    else:
        # Just analyze, don't improve
        system_analysis = await validator.system_analyzer.analyze(system_prompt, llm_client)
        coverage_result = await validator.coverage_analyzer.analyze_coverage(
            system_analysis=system_analysis,
            eval_prompt=eval_prompt,
            dimensions=dimensions or [],
            auto_fails=auto_fails or [],
            llm_client=llm_client
        )

        return {
            "eval_prompt": eval_prompt,
            "coverage": {
                "percentage": coverage_result.overall_coverage_pct,
                "level": coverage_result.coverage_level.value,
                "passes_threshold": coverage_result.passes_threshold,
                "gaps_found": len(coverage_result.gaps)
            },
            "gaps": [
                {
                    "category": g.category,
                    "missing": g.missing_aspect,
                    "severity": g.severity,
                    "recommendation": g.recommended_addition
                }
                for g in coverage_result.gaps
            ]
        }


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'CoverageValidator',
    'SystemPromptAnalyzer',
    'EvalCoverageAnalyzer',
    'EvalPromptImprover',
    'CoverageAnalysisResult',
    'CoverageGap',
    'CoverageLevel',
    'validate_eval_coverage'
]
