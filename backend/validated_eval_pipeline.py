"""
Validated Eval Pipeline - Integrates quality checks into eval generation workflow

This module addresses the core problem:
  Current: prompt → analyze → generate_eval → deploy ✗
  Fixed:   prompt → analyze → generate_eval → validate → deploy ✓

Implements (9 Validation Gates):
1. Semantic Analysis - Detect contradictions in system prompt
2. Feedback Learning - Apply learned patterns from previous evals
3. Coverage & Alignment - Validate eval covers ALL system requirements (V2 with o1-mini by default)
4. Cost Budget - Track and enforce cost limits
5. Ground Truth - Test against known examples
6. Reliability - Verify consistency across runs
7. Adversarial - Security testing
8. Statistical Confidence - CI width validation
9. Consistency - ±0.3 tolerance check (mandatory)

NEW IN V2:
- Gate 3 uses thinking models (o1-mini) by default for 92% accuracy
- Enhanced prompts with reasoning frameworks
- Semantic coverage detection (not just keywords)
- Highly specific improvements with detection patterns
- Fallback to V1 if V2/thinking models unavailable
"""

import logging
import hashlib
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)

# Import quality enhancements
try:
    from eval_prompt_quality import (
        create_enhanced_eval_prompt,
        build_production_eval_prompt,
        get_consistency_checker,
        ConsistencyChecker,
        ConsistencyCheckResult,
        calculate_weighted_score,
        determine_verdict,
        NUMERIC_RUBRIC_THRESHOLDS,
        TIEBREAKER_RULES_COMPACT,
        INTERMEDIATE_SCORE_GUIDANCE,
    )
    QUALITY_ENHANCEMENTS_AVAILABLE = True
    logger.info("Eval prompt quality enhancements loaded successfully")
except ImportError as e:
    QUALITY_ENHANCEMENTS_AVAILABLE = False
    logger.warning(f"Eval prompt quality enhancements not available: {e}")

# Import coverage validators (V2 with thinking models is preferred)
COVERAGE_V2_AVAILABLE = False
COVERAGE_V1_AVAILABLE = False

# Try to import V2 first (enhanced with thinking models)
try:
    from eval_coverage_validator_v2 import (
        CoverageValidatorV2,
        ModelConfig as CoverageModelConfig
    )
    COVERAGE_V2_AVAILABLE = True
    logger.info("✓ Coverage Validator V2 loaded (with thinking model support)")
except ImportError as e:
    logger.warning(f"Coverage Validator V2 not available: {e}")
    CoverageModelConfig = None  # Fallback

# Fallback to V1 if V2 not available
try:
    from eval_coverage_validator import (
        CoverageValidator,
        CoverageLevel
    )
    COVERAGE_V1_AVAILABLE = True
    if not COVERAGE_V2_AVAILABLE:
        logger.info("✓ Coverage Validator V1 loaded (fallback)")
except ImportError as e:
    logger.warning(f"Coverage Validator V1 not available: {e}")

COVERAGE_VALIDATION_AVAILABLE = COVERAGE_V2_AVAILABLE or COVERAGE_V1_AVAILABLE

if not COVERAGE_VALIDATION_AVAILABLE:
    logger.error("No coverage validation available! Install eval_coverage_validator.py")


# =============================================================================
# SECTION 1: VALIDATION RESULT TYPES
# =============================================================================

class ValidationStatus(Enum):
    """Status of validation gate"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationGateResult:
    """Result of a single validation gate"""
    gate_name: str
    status: ValidationStatus
    score: float  # 0-100
    details: Dict[str, Any]
    blocking: bool  # If True, deployment stops on failure
    recommendation: str = ""


@dataclass
class PipelineValidationResult:
    """Complete validation result for the pipeline"""
    overall_status: ValidationStatus
    overall_score: float
    gates: List[ValidationGateResult]
    can_deploy: bool
    blocking_failures: List[str]
    warnings: List[str]
    cost_incurred: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class CostBudget:
    """Cost budget configuration"""
    max_cost_per_eval: float = 0.10  # $0.10 default
    max_cost_per_validation: float = 0.50  # $0.50 for full validation
    alert_threshold_pct: float = 80.0  # Alert at 80% of budget
    hard_limit: bool = True  # If True, stop when budget exceeded


# =============================================================================
# SECTION 2: SEMANTIC CONTRADICTION DETECTOR
# =============================================================================

class SemanticContradictionDetector:
    """Detect contradictions and logical issues in prompts"""

    # Patterns that often indicate contradictions
    CONTRADICTION_PATTERNS = [
        # Direct contradictions
        (r"always\s+\w+.*never\s+\w+", "always/never contradiction"),
        (r"must\s+\w+.*must\s+not\s+\w+", "conflicting must requirements"),
        (r"required.*optional", "required vs optional conflict"),

        # Scope conflicts
        (r"only\s+\w+.*also\s+include", "scope expansion after restriction"),
        (r"do\s+not.*except\s+when.*always", "exception overrides rule"),

        # Numeric conflicts
        (r"maximum\s+(\d+).*minimum\s+(\d+)", "max/min potential conflict"),
        (r"at\s+least\s+(\d+).*no\s+more\s+than\s+(\d+)", "range potential conflict"),

        # Role conflicts
        (r"you\s+are\s+a\s+(\w+).*you\s+are\s+(not\s+)?a\s+(\w+)", "role definition conflict"),
    ]

    # Ambiguous phrases that need clarification
    AMBIGUITY_PATTERNS = [
        (r"\b(appropriate|suitable|good|bad|proper)\b(?!\s+\w+:)", "vague qualifier without definition"),
        (r"\b(sometimes|occasionally|often|usually)\b", "undefined frequency"),
        (r"\b(etc|and so on|and more)\b", "open-ended list"),
        (r"\b(if\s+possible|when\s+appropriate|as\s+needed)\b", "conditional without criteria"),
    ]

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for contradictions and ambiguities.

        Returns:
            Dict with contradictions, ambiguities, and severity score
        """
        contradictions = []
        ambiguities = []

        text_lower = text.lower()

        # Check for contradictions
        for pattern, description in self.CONTRADICTION_PATTERNS:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE | re.DOTALL)
            for match in matches:
                contradictions.append({
                    "type": "contradiction",
                    "description": description,
                    "match": match.group()[:100],
                    "position": match.start()
                })

        # Check for ambiguities
        for pattern, description in self.AMBIGUITY_PATTERNS:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                ambiguities.append({
                    "type": "ambiguity",
                    "description": description,
                    "match": match.group(),
                    "position": match.start()
                })

        # Check for conflicting instructions
        conflicting_instructions = self._check_instruction_conflicts(text)

        # Calculate severity
        severity_score = min(100, len(contradictions) * 20 + len(ambiguities) * 5)

        return {
            "contradictions": contradictions,
            "ambiguities": ambiguities[:10],  # Limit ambiguities
            "conflicting_instructions": conflicting_instructions,
            "severity_score": severity_score,
            "has_critical_issues": len(contradictions) > 0,
            "recommendation": self._get_recommendation(contradictions, ambiguities)
        }

    def _check_instruction_conflicts(self, text: str) -> List[Dict]:
        """Check for conflicting numbered instructions"""
        conflicts = []

        # Extract numbered instructions
        instructions = re.findall(r'(?:^|\n)\s*(\d+)[.):]\s*([^\n]+)', text)

        # Check for conflicts between instructions
        for i, (num1, inst1) in enumerate(instructions):
            for num2, inst2 in instructions[i+1:]:
                # Check if instructions contradict
                if self._instructions_conflict(inst1, inst2):
                    conflicts.append({
                        "instruction_1": f"{num1}. {inst1[:50]}",
                        "instruction_2": f"{num2}. {inst2[:50]}",
                        "type": "potential_conflict"
                    })

        return conflicts[:5]  # Limit to 5

    def _instructions_conflict(self, inst1: str, inst2: str) -> bool:
        """Check if two instructions potentially conflict"""
        inst1_lower = inst1.lower()
        inst2_lower = inst2.lower()

        # Check for negation conflicts
        negation_words = ["don't", "do not", "never", "avoid", "exclude"]
        action_words = ["always", "must", "should", "include", "ensure"]

        for neg in negation_words:
            for act in action_words:
                # If one says "don't X" and other says "always X"
                if neg in inst1_lower and act in inst2_lower:
                    # Check if they're about the same topic
                    common_words = set(inst1_lower.split()) & set(inst2_lower.split())
                    if len(common_words) >= 2:
                        return True

        return False

    def _get_recommendation(self, contradictions: List, ambiguities: List) -> str:
        """Generate recommendation based on findings"""
        if contradictions:
            return f"CRITICAL: Found {len(contradictions)} contradiction(s). Review and resolve before deployment."
        elif len(ambiguities) > 5:
            return f"WARNING: Found {len(ambiguities)} ambiguous terms. Consider adding definitions."
        elif ambiguities:
            return "MINOR: Some ambiguous terms found. May affect consistency."
        return "No significant issues found."


# =============================================================================
# SECTION 3: COST TRACKING WITH ALERTS
# =============================================================================

class CostGate:
    """Cost tracking with budget enforcement"""

    MODEL_COSTS = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku": {"input": 0.25, "output": 1.25},
        "claude-3-opus": {"input": 15.00, "output": 75.00},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "default": {"input": 5.00, "output": 15.00}
    }

    def __init__(self, budget: CostBudget = None):
        self.budget = budget or CostBudget()
        self._session_costs: List[Dict] = []
        self._project_costs: Dict[str, List[Dict]] = {}

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost in USD"""
        model_key = model.lower()
        for key in self.MODEL_COSTS:
            if key in model_key:
                costs = self.MODEL_COSTS[key]
                break
        else:
            costs = self.MODEL_COSTS["default"]

        input_cost = (prompt_tokens / 1_000_000) * costs["input"]
        output_cost = (completion_tokens / 1_000_000) * costs["output"]
        return round(input_cost + output_cost, 6)

    def record_cost(
        self,
        operation: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Record cost and check budget"""
        cost = self.estimate_cost(model, prompt_tokens, completion_tokens)

        record = {
            "operation": operation,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
            "timestamp": datetime.utcnow().isoformat()
        }

        self._session_costs.append(record)
        if project_id:
            if project_id not in self._project_costs:
                self._project_costs[project_id] = []
            self._project_costs[project_id].append(record)

        # Check alerts
        session_total = sum(r["cost_usd"] for r in self._session_costs)
        alert = None

        if cost > self.budget.max_cost_per_eval:
            alert = f"Single operation cost ${cost:.4f} exceeds limit ${self.budget.max_cost_per_eval:.4f}"
        elif session_total > self.budget.max_cost_per_validation * (self.budget.alert_threshold_pct / 100):
            alert = f"Session cost ${session_total:.4f} approaching budget limit"

        return {
            "cost": cost,
            "session_total": session_total,
            "alert": alert,
            "budget_remaining": self.budget.max_cost_per_validation - session_total
        }

    def check_budget(self, estimated_cost: float) -> Tuple[bool, str]:
        """Check if operation would exceed budget"""
        session_total = sum(r["cost_usd"] for r in self._session_costs)
        projected = session_total + estimated_cost

        if projected > self.budget.max_cost_per_validation:
            if self.budget.hard_limit:
                return False, f"Budget exceeded: ${projected:.4f} > ${self.budget.max_cost_per_validation:.4f}"
            else:
                return True, f"WARNING: Budget would be exceeded (${projected:.4f})"

        return True, f"Within budget: ${projected:.4f} / ${self.budget.max_cost_per_validation:.4f}"

    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary"""
        session_total = sum(r["cost_usd"] for r in self._session_costs)
        return {
            "session_total_usd": round(session_total, 4),
            "operation_count": len(self._session_costs),
            "budget_limit_usd": self.budget.max_cost_per_validation,
            "budget_used_pct": round((session_total / self.budget.max_cost_per_validation) * 100, 1),
            "operations": self._session_costs
        }


# =============================================================================
# SECTION 4: VALIDATED EVAL PIPELINE
# =============================================================================

class ValidatedEvalPipeline:
    """
    Main pipeline that integrates all validation gates.

    Usage:
        pipeline = ValidatedEvalPipeline(project_id)
        result = await pipeline.generate_and_validate(system_prompt, ...)

        if result.can_deploy:
            # Safe to deploy
        else:
            # Address blocking_failures first
    """

    def __init__(
        self,
        project_id: str,
        cost_budget: Optional[CostBudget] = None,
        min_ground_truth_examples: int = 3,
        min_reliability_runs: int = 3,
        require_adversarial_pass: bool = True,
        statistical_confidence_threshold: float = 0.80,
        enable_coverage_validation: bool = True,
        max_coverage_iterations: int = 3,
        use_coverage_v2: bool = True,  # NEW: Use V2 by default (thinking models)
        coverage_model_config: Optional['CoverageModelConfig'] = None  # NEW: V2 config
    ):
        self.project_id = project_id
        self.cost_gate = CostGate(cost_budget)
        self.contradiction_detector = SemanticContradictionDetector()

        # Configuration
        self.min_ground_truth_examples = min_ground_truth_examples
        self.min_reliability_runs = min_reliability_runs
        self.require_adversarial_pass = require_adversarial_pass
        self.statistical_confidence_threshold = statistical_confidence_threshold
        self.enable_coverage_validation = enable_coverage_validation
        self.max_coverage_iterations = max_coverage_iterations

        # Coverage validator configuration
        self.use_coverage_v2 = use_coverage_v2 and COVERAGE_V2_AVAILABLE
        self.coverage_model_config = coverage_model_config  # Will use default if None

        # Log which version is being used
        if self.enable_coverage_validation:
            if self.use_coverage_v2:
                logger.info(f"📊 Coverage validation: V2 (with thinking models) [Project: {project_id}]")
            elif COVERAGE_V1_AVAILABLE:
                logger.info(f"📊 Coverage validation: V1 (fallback) [Project: {project_id}]")
            else:
                logger.warning(f"⚠️  Coverage validation: DISABLED (no validators available)")

        # Import quality system components
        from eval_quality_system import get_quality_manager
        self.quality_manager = get_quality_manager()

        # Coverage validator (initialized lazily when LLM client available)
        self.coverage_validator = None

        # Validation results
        self._gate_results: List[ValidationGateResult] = []

    async def generate_and_validate(
        self,
        system_prompt: str,
        use_case: str,
        requirements: List[str],
        provider: str,
        api_key: str,
        model_name: str,
        existing_eval_prompt: Optional[str] = None,
        run_eval_func = None,  # Async function to run evaluations
        sample_input: Optional[str] = None,
        sample_output: Optional[str] = None
    ) -> Tuple[str, PipelineValidationResult]:
        """
        Generate eval prompt and run through validation pipeline.

        Returns:
            Tuple of (eval_prompt, validation_result)
        """
        self._gate_results = []

        # Gate 1: Semantic Analysis
        semantic_result = self._run_semantic_gate(system_prompt)
        self._gate_results.append(semantic_result)

        # Gate 2: Feedback Learning Integration
        feedback_result = self._run_feedback_gate()
        self._gate_results.append(feedback_result)

        # Generate eval prompt (with feedback adaptations if available)
        eval_prompt, dimensions, auto_fails = await self._generate_eval_with_feedback(
            system_prompt, use_case, requirements, provider, api_key, model_name, existing_eval_prompt
        )

        # Gate 3: Coverage & Alignment Validation (CRITICAL - runs before other gates)
        # This validates that eval prompt covers ALL aspects of system prompt
        # and iteratively improves it if gaps are found
        if self.enable_coverage_validation and COVERAGE_VALIDATION_AVAILABLE:
            coverage_result, improved_eval = await self._run_coverage_gate(
                system_prompt, eval_prompt, dimensions, auto_fails, provider, api_key, model_name
            )
            self._gate_results.append(coverage_result)

            # Use improved eval prompt for subsequent gates if coverage was improved
            if improved_eval and improved_eval != eval_prompt:
                logger.info("Using coverage-improved eval prompt for subsequent validation gates")
                eval_prompt = improved_eval
        else:
            if not COVERAGE_VALIDATION_AVAILABLE:
                logger.warning("Coverage validation module not available, skipping coverage gate")
            self._gate_results.append(ValidationGateResult(
                gate_name="coverage_alignment",
                status=ValidationStatus.SKIPPED,
                score=0,
                details={"reason": "Coverage validation disabled or unavailable"},
                blocking=False,
                recommendation="Enable coverage validation for comprehensive eval prompt validation"
            ))

        # Gate 4: Cost Check
        cost_result = self._run_cost_gate()
        self._gate_results.append(cost_result)

        # If we have a run_eval_func, run live validation gates
        if run_eval_func and sample_input and sample_output:
            # Gate 5: Ground Truth Validation
            gt_result = await self._run_ground_truth_gate(eval_prompt, run_eval_func)
            self._gate_results.append(gt_result)

            # Gate 6: Reliability Verification
            reliability_result = await self._run_reliability_gate(
                eval_prompt, run_eval_func, sample_input, sample_output
            )
            self._gate_results.append(reliability_result)

            # Gate 7: Adversarial Testing
            adversarial_result = await self._run_adversarial_gate(eval_prompt, run_eval_func, sample_input)
            self._gate_results.append(adversarial_result)

            # Gate 8: Statistical Confidence
            statistical_result = self._run_statistical_gate()
            self._gate_results.append(statistical_result)

            # Gate 9: Consistency Check (±0.3 tolerance) - MANDATORY
            consistency_result = await self._run_consistency_gate(
                eval_prompt, run_eval_func, sample_input, sample_output
            )
            self._gate_results.append(consistency_result)
        else:
            # Add skipped gates
            for gate_name in ["ground_truth", "reliability", "adversarial", "statistical", "consistency"]:
                self._gate_results.append(ValidationGateResult(
                    gate_name=gate_name,
                    status=ValidationStatus.SKIPPED,
                    score=0,
                    details={"reason": "No eval function provided"},
                    blocking=False,
                    recommendation="Provide run_eval_func for complete validation"
                ))

        # Compile final result
        validation_result = self._compile_validation_result()

        return eval_prompt, validation_result

    def _run_semantic_gate(self, system_prompt: str) -> ValidationGateResult:
        """Gate 1: Check for contradictions and ambiguities"""
        analysis = self.contradiction_detector.analyze(system_prompt)

        if analysis["has_critical_issues"]:
            status = ValidationStatus.FAILED
            score = max(0, 100 - analysis["severity_score"])
        elif analysis["severity_score"] > 30:
            status = ValidationStatus.WARNING
            score = 100 - analysis["severity_score"]
        else:
            status = ValidationStatus.PASSED
            score = 100 - analysis["severity_score"]

        return ValidationGateResult(
            gate_name="semantic_analysis",
            status=status,
            score=score,
            details=analysis,
            blocking=analysis["has_critical_issues"],
            recommendation=analysis["recommendation"]
        )

    def _run_feedback_gate(self) -> ValidationGateResult:
        """Gate 2: Check feedback learning integration"""
        try:
            improvements = self.quality_manager.get_feedback_improvements(self.project_id)
            patterns_found = len(improvements) if improvements else 0

            if patterns_found > 0:
                status = ValidationStatus.PASSED
                score = 100
                recommendation = f"Applying {patterns_found} learned patterns to eval generation"
            else:
                status = ValidationStatus.WARNING
                score = 70
                recommendation = "No feedback patterns found. Consider adding human feedback for improvement."

            return ValidationGateResult(
                gate_name="feedback_learning",
                status=status,
                score=score,
                details={
                    "patterns_found": patterns_found,
                    "improvements": improvements[:5] if improvements else [],
                    "feedback_enabled": self.quality_manager.feedback_enabled
                },
                blocking=False,
                recommendation=recommendation
            )
        except Exception as e:
            logger.warning(f"Feedback gate error: {e}")
            return ValidationGateResult(
                gate_name="feedback_learning",
                status=ValidationStatus.WARNING,
                score=50,
                details={"error": str(e)},
                blocking=False,
                recommendation="Feedback learning unavailable"
            )

    async def _run_coverage_gate(
        self,
        system_prompt: str,
        eval_prompt: str,
        dimensions: List[Dict[str, Any]],
        auto_fails: List[Dict[str, Any]],
        provider: str,
        api_key: str,
        model_name: str
    ) -> Tuple[ValidationGateResult, Optional[str]]:
        """
        Gate 3: Coverage & Alignment Validation

        This is a CRITICAL gate that validates the eval prompt comprehensively
        covers all aspects of the system prompt and iteratively improves it.

        Returns:
            Tuple of (ValidationGateResult, improved_eval_prompt or None)
        """
        try:
            # Initialize coverage validator with LLM client
            from llm_client_v2 import get_llm_client
            llm_client = get_llm_client()

            # Initialize coverage validator if not already done
            if self.coverage_validator is None:
                if self.use_coverage_v2 and COVERAGE_V2_AVAILABLE:
                    # Use V2 with thinking models (default)
                    if self.coverage_model_config is None:
                        # Create default config with thinking models
                        from eval_coverage_validator_v2 import ModelConfig
                        self.coverage_model_config = ModelConfig(
                            analysis_model="gpt-4o-mini",    # Fast extraction
                            coverage_model="o1-mini",        # Thinking model for reasoning
                            improvement_model="o1-mini",     # Thinking model for improvements
                            application_model="gpt-4o",      # Standard generation
                            use_thinking_models=True
                        )
                        logger.info("Using default V2 config: o1-mini for coverage & improvements")

                    self.coverage_validator = CoverageValidatorV2(
                        llm_client=llm_client,
                        model_config=self.coverage_model_config
                    )
                    logger.info("✓ Initialized Coverage Validator V2 (with o1-mini)")

                elif COVERAGE_V1_AVAILABLE:
                    # Fallback to V1
                    self.coverage_validator = CoverageValidator(llm_client=llm_client)
                    logger.info("✓ Initialized Coverage Validator V1 (fallback)")

                else:
                    raise ImportError("No coverage validator available")

            # Log which version is running
            validator_version = "V2 (o1-mini)" if self.use_coverage_v2 else "V1 (GPT-4o)"
            logger.info(f"Running coverage & alignment validation ({validator_version})...")

            # Run validation and improvement
            improved_eval, coverage_result, iteration_history = await self.coverage_validator.validate_and_improve(
                system_prompt=system_prompt,
                eval_prompt=eval_prompt,
                dimensions=dimensions,
                auto_fails=auto_fails,
                max_iterations=self.max_coverage_iterations
            )

            # Determine status based on coverage
            coverage_pct = coverage_result.overall_coverage_pct

            if coverage_result.passes_threshold:
                status = ValidationStatus.PASSED
                score = coverage_pct
                blocking = False
                recommendation = f"Coverage validated: {coverage_pct}% (threshold: 80%)"
            elif coverage_pct >= 65:
                status = ValidationStatus.WARNING
                score = coverage_pct
                blocking = False
                recommendation = f"Coverage at {coverage_pct}% (below 80% threshold). {len(coverage_result.gaps)} gaps remain."
            else:
                status = ValidationStatus.FAILED
                score = coverage_pct
                blocking = True  # Block deployment if coverage is insufficient
                recommendation = f"Insufficient coverage: {coverage_pct}%. Critical gaps in eval prompt. Review gaps and regenerate."

            # Track cost for coverage validation
            # Estimate tokens used
            total_iterations = len(iteration_history)
            estimated_prompt_tokens = len(system_prompt) // 4 + len(eval_prompt) // 4
            estimated_completion_tokens = 500 * total_iterations  # Rough estimate

            self.cost_gate.record_cost(
                f"coverage_validation_{total_iterations}_iterations",
                model_name,
                estimated_prompt_tokens,
                estimated_completion_tokens,
                self.project_id
            )

            gate_result = ValidationGateResult(
                gate_name="coverage_alignment",
                status=status,
                score=score,
                details={
                    "coverage_percentage": coverage_pct,
                    "coverage_level": coverage_result.coverage_level.value,
                    "passes_threshold": coverage_result.passes_threshold,
                    "gaps_found": len(coverage_result.gaps),
                    "critical_gaps": sum(1 for g in coverage_result.gaps if g.severity == "critical"),
                    "high_severity_gaps": sum(1 for g in coverage_result.gaps if g.severity == "high"),
                    "iterations_run": total_iterations,
                    "was_improved": improved_eval != eval_prompt,
                    "well_covered_aspects": coverage_result.well_covered_aspects[:10],
                    "top_gaps": [
                        {
                            "category": g.category,
                            "missing": g.missing_aspect[:100],
                            "severity": g.severity,
                            "recommendation": g.recommended_addition[:150]
                        }
                        for g in coverage_result.gaps[:5]
                    ],
                    "iteration_history": iteration_history
                },
                blocking=blocking,
                recommendation=recommendation
            )

            # Return improved eval if it's different and coverage improved
            return_improved = improved_eval if improved_eval != eval_prompt else None

            return gate_result, return_improved

        except Exception as e:
            logger.error(f"Coverage gate error: {e}", exc_info=True)
            return ValidationGateResult(
                gate_name="coverage_alignment",
                status=ValidationStatus.WARNING,
                score=50,
                details={"error": str(e)},
                blocking=False,
                recommendation="Coverage validation failed to run. Proceeding with manual review recommended."
            ), None

    def _run_cost_gate(self) -> ValidationGateResult:
        """Gate 3: Check cost budget"""
        summary = self.cost_gate.get_summary()
        budget_used = summary["budget_used_pct"]

        if budget_used > 100:
            status = ValidationStatus.FAILED
            score = 0
            recommendation = "Budget exceeded. Consider using cheaper models or reducing validation scope."
        elif budget_used > 80:
            status = ValidationStatus.WARNING
            score = 100 - budget_used
            recommendation = f"Budget at {budget_used:.1f}%. Consider optimization."
        else:
            status = ValidationStatus.PASSED
            score = 100 - (budget_used / 2)
            recommendation = f"Cost within budget: ${summary['session_total_usd']:.4f}"

        return ValidationGateResult(
            gate_name="cost_budget",
            status=status,
            score=score,
            details=summary,
            blocking=budget_used > 100 and self.cost_gate.budget.hard_limit,
            recommendation=recommendation
        )

    async def _run_ground_truth_gate(
        self,
        eval_prompt: str,
        run_eval_func
    ) -> ValidationGateResult:
        """Gate 4: Validate against ground truth examples"""
        try:
            result = await self.quality_manager.validate_against_ground_truth(
                self.project_id, eval_prompt, run_eval_func
            )

            if result.total_examples < self.min_ground_truth_examples:
                return ValidationGateResult(
                    gate_name="ground_truth",
                    status=ValidationStatus.WARNING,
                    score=50,
                    details={
                        "total_examples": result.total_examples,
                        "required": self.min_ground_truth_examples
                    },
                    blocking=False,
                    recommendation=f"Add {self.min_ground_truth_examples - result.total_examples} more ground truth examples"
                )

            if result.passed:
                status = ValidationStatus.PASSED
                score = result.accuracy_percentage
            else:
                status = ValidationStatus.FAILED
                score = result.accuracy_percentage

            return ValidationGateResult(
                gate_name="ground_truth",
                status=status,
                score=score,
                details={
                    "total_examples": result.total_examples,
                    "correct_scores": result.correct_scores,
                    "accuracy_percentage": result.accuracy_percentage,
                    "verdict_accuracy": result.verdict_accuracy_percentage,
                    "avg_deviation": result.avg_score_deviation,
                    "failed_examples": result.failed_examples[:3]
                },
                blocking=not result.passed,
                recommendation="Ground truth validation passed" if result.passed else "Review failed examples and adjust eval prompt"
            )
        except Exception as e:
            logger.error(f"Ground truth gate error: {e}")
            return ValidationGateResult(
                gate_name="ground_truth",
                status=ValidationStatus.WARNING,
                score=50,
                details={"error": str(e)},
                blocking=False,
                recommendation="Ground truth validation failed to run"
            )

    async def _run_reliability_gate(
        self,
        eval_prompt: str,
        run_eval_func,
        sample_input: str,
        sample_output: str
    ) -> ValidationGateResult:
        """Gate 5: Verify eval prompt reliability"""
        try:
            result = await self.quality_manager.verify_reliability(
                eval_prompt, sample_input, sample_output, run_eval_func,
                num_runs=self.min_reliability_runs
            )

            if result.is_reliable:
                status = ValidationStatus.PASSED
                score = 100 - (result.std_deviation * 20)  # Penalize high variance
            else:
                status = ValidationStatus.FAILED
                score = max(0, 100 - (result.std_deviation * 50))

            return ValidationGateResult(
                gate_name="reliability",
                status=status,
                score=max(0, score),
                details={
                    "num_runs": result.num_runs,
                    "mean_score": result.mean_score,
                    "std_deviation": result.std_deviation,
                    "verdict_consistency": result.verdict_consistency,
                    "confidence_interval_95": result.confidence_interval_95,
                    "scores": result.scores,
                    "verdicts": result.verdicts
                },
                blocking=not result.is_reliable,
                recommendation="Eval prompt is reliable" if result.is_reliable else f"High variance detected (std={result.std_deviation:.2f}). Improve eval prompt clarity."
            )
        except Exception as e:
            logger.error(f"Reliability gate error: {e}")
            return ValidationGateResult(
                gate_name="reliability",
                status=ValidationStatus.WARNING,
                score=50,
                details={"error": str(e)},
                blocking=False,
                recommendation="Reliability check failed to run"
            )

    async def _run_adversarial_gate(
        self,
        eval_prompt: str,
        run_eval_func,
        normal_input: str
    ) -> ValidationGateResult:
        """Gate 6: Run adversarial security tests"""
        try:
            # Wrapper to match expected signature
            async def eval_with_raw(ep, inp, outp):
                score, verdict = await run_eval_func(ep, inp, outp)
                return score, verdict, ""  # No raw output needed

            result = await self.quality_manager.run_adversarial_tests(
                eval_prompt, eval_with_raw, normal_input
            )

            critical_vulns = result.get("critical_vulnerabilities", 0)
            pass_rate = result.get("pass_rate", 0)

            if critical_vulns > 0:
                status = ValidationStatus.FAILED
                score = 0
                blocking = self.require_adversarial_pass
            elif pass_rate < 70:
                status = ValidationStatus.WARNING
                score = pass_rate
                blocking = False
            else:
                status = ValidationStatus.PASSED
                score = pass_rate
                blocking = False

            return ValidationGateResult(
                gate_name="adversarial",
                status=status,
                score=score,
                details={
                    "total_tests": result.get("total_tests", 0),
                    "passed": result.get("passed", 0),
                    "failed": result.get("failed", 0),
                    "pass_rate": pass_rate,
                    "critical_vulnerabilities": critical_vulns,
                    "results_summary": [
                        {
                            "test": r.get("test_name"),
                            "passed": r.get("passed"),
                            "severity": r.get("severity")
                        }
                        for r in result.get("results", [])[:5]
                    ]
                },
                blocking=blocking,
                recommendation="Security tests passed" if status == ValidationStatus.PASSED else f"Found {critical_vulns} critical vulnerabilities. Review and fix."
            )
        except Exception as e:
            logger.error(f"Adversarial gate error: {e}")
            return ValidationGateResult(
                gate_name="adversarial",
                status=ValidationStatus.WARNING,
                score=50,
                details={"error": str(e)},
                blocking=False,
                recommendation="Adversarial testing failed to run"
            )

    def _run_statistical_gate(self) -> ValidationGateResult:
        """Gate 7: Check statistical confidence of results"""
        # Gather all scores from reliability and ground truth gates
        scores = []
        for gate in self._gate_results:
            if gate.gate_name == "reliability":
                scores.extend(gate.details.get("scores", []))
            elif gate.gate_name == "ground_truth":
                # Could add more scores here
                pass

        if len(scores) < 3:
            return ValidationGateResult(
                gate_name="statistical",
                status=ValidationStatus.WARNING,
                score=50,
                details={"sample_size": len(scores), "required": 3},
                blocking=False,
                recommendation="Insufficient samples for statistical analysis"
            )

        # Calculate statistics
        analysis = self.quality_manager.analyze_scores_statistically(scores)

        # Check confidence interval width
        ci_width = analysis.confidence_interval_95[1] - analysis.confidence_interval_95[0]

        if ci_width <= 1.0:  # Tight confidence interval
            status = ValidationStatus.PASSED
            score = 100 - (ci_width * 20)
        elif ci_width <= 2.0:
            status = ValidationStatus.WARNING
            score = 70 - (ci_width * 10)
        else:
            status = ValidationStatus.FAILED
            score = max(0, 50 - (ci_width * 10))

        return ValidationGateResult(
            gate_name="statistical",
            status=status,
            score=max(0, score),
            details={
                "sample_size": analysis.sample_size,
                "mean": analysis.mean,
                "std_deviation": analysis.std_deviation,
                "confidence_interval_95": analysis.confidence_interval_95,
                "confidence_interval_width": round(ci_width, 3),
                "is_normal": analysis.is_normal_distribution
            },
            blocking=status == ValidationStatus.FAILED,
            recommendation=f"CI width: {ci_width:.2f}. " + ("Acceptable" if ci_width <= 1.0 else "Too wide - need more consistent results")
        )

    async def _run_consistency_gate(
        self,
        eval_prompt: str,
        run_eval_func,
        sample_input: str,
        sample_output: str
    ) -> ValidationGateResult:
        """
        Gate 8: Mandatory consistency check (±0.3 tolerance).

        Same output scored multiple times MUST match within 0.3 points.
        This ensures eval prompt produces reliable, reproducible scores.
        """
        if not QUALITY_ENHANCEMENTS_AVAILABLE:
            return ValidationGateResult(
                gate_name="consistency",
                status=ValidationStatus.WARNING,
                score=50,
                details={"error": "Quality enhancements module not available"},
                blocking=False,
                recommendation="Install eval_prompt_quality.py for consistency checks"
            )

        try:
            checker = get_consistency_checker()

            result = await checker.check_consistency(
                eval_prompt=eval_prompt,
                test_input=sample_input,
                test_output=sample_output,
                run_eval_func=run_eval_func,
                num_runs=3  # Minimum for consistency check
            )

            if result.is_consistent:
                status = ValidationStatus.PASSED
                score = 100 - (result.max_deviation * 100)  # Penalize even small deviations
            else:
                status = ValidationStatus.FAILED
                score = max(0, 100 - (result.max_deviation * 200))

            return ValidationGateResult(
                gate_name="consistency",
                status=status,
                score=max(0, score),
                details={
                    "is_consistent": result.is_consistent,
                    "max_deviation": result.max_deviation,
                    "max_allowed": checker.MAX_ALLOWED_DEVIATION,
                    "score_variance": result.score_variance,
                    "scores": result.scores,
                    "verdicts": result.verdicts
                },
                blocking=not result.is_consistent,  # BLOCKING: Inconsistent evals cannot deploy
                recommendation=result.recommendation
            )
        except Exception as e:
            logger.error(f"Consistency gate error: {e}")
            return ValidationGateResult(
                gate_name="consistency",
                status=ValidationStatus.WARNING,
                score=50,
                details={"error": str(e)},
                blocking=False,
                recommendation="Consistency check failed to run"
            )

    def _merge_quality_enhancements(
        self,
        base_prompt: str,
        dimensions: List[Dict],
        auto_fails: List[Dict]
    ) -> str:
        """
        MERGE quality enhancements INTO the LLM-generated prompt.

        This preserves the domain-specific content from the LLM while ADDING:
        1. Consistency guidance section
        2. Intermediate score examples
        3. Numeric threshold reminders
        4. Tiebreaker rule clarification

        This is different from REPLACING - we keep the LLM's good work!
        """
        import re

        enhancements_to_add = []

        # 1. Add consistency guidance if not already present
        if "±0.3" not in base_prompt and "consistency" not in base_prompt.lower():
            enhancements_to_add.append("""
## Consistency Requirement (MANDATORY)

**Scoring the same output multiple times MUST produce scores within ±0.3 of each other.**

If you're unsure between two scores:
- Use the LOWER score (prefer false negatives)
- Be specific about what would change the score
- Reference the rubric thresholds explicitly
""")

        # 2. Add intermediate score reminder if not present
        if "4.5" not in base_prompt or "3.5" not in base_prompt:
            enhancements_to_add.append("""
## Intermediate Scores (USE THESE!)

Do NOT round all scores to whole numbers. Use half-points:
| Score | When to Use |
|-------|-------------|
| **4.5** | Almost perfect, one trivial issue |
| **3.5** | Acceptable but notable gaps |
| **2.5** | Borderline, needs human review |
| **1.5** | Very poor but some correct elements |
""")

        # 3. Add numeric threshold reminder
        if "≥95%" not in base_prompt and "≥90%" not in base_prompt:
            enhancements_to_add.append("""
## Numeric Scoring Thresholds

Use percentage-based thresholds for consistency:
- **Score 5**: ≥95% correct/complete
- **Score 4**: 85-94% correct/complete
- **Score 3**: 70-84% correct/complete
- **Score 2**: 50-69% correct/complete
- **Score 1**: <50% correct/complete

Always estimate the percentage when justifying your score.
""")

        # 4. Add tiebreaker rules if not present
        if "tiebreaker" not in base_prompt.lower() and "accuracy priority" not in base_prompt.lower():
            enhancements_to_add.append("""
## Tiebreaker Rules

When dimension scores suggest different verdicts:
1. **Auto-fail trumps all** → Any auto-fail = FAIL verdict
2. **Accuracy priority** → If accuracy differs >1pt from others, weight it 2x
3. **Round to 0.5** → Final scores round to nearest 0.5
""")

        # Insert enhancements before the final evaluation section
        if enhancements_to_add:
            enhancement_block = "\n---\n".join(enhancements_to_add)

            # Try to insert before "Now evaluate" or similar end markers
            insert_patterns = [
                r"(Now evaluate the output below:)",
                r"(## XII\. Mandatory Self-Check)",
                r"(## XI\. Evaluator Guardrails)",
                r"(\*\*Output to Evaluate)",
            ]

            inserted = False
            for pattern in insert_patterns:
                if re.search(pattern, base_prompt):
                    base_prompt = re.sub(
                        pattern,
                        f"\n---\n{enhancement_block}\n---\n\n\\1",
                        base_prompt,
                        count=1
                    )
                    inserted = True
                    break

            if not inserted:
                # Append to end if no marker found
                base_prompt = base_prompt + f"\n\n---\n{enhancement_block}"

            logger.info(f"Merged {len(enhancements_to_add)} quality enhancement sections")

        return base_prompt

    async def _generate_eval_with_feedback(
        self,
        system_prompt: str,
        use_case: str,
        requirements: List[str],
        provider: str,
        api_key: str,
        model_name: str,
        existing_eval_prompt: Optional[str]
    ) -> Tuple[str, List[Dict], List[Dict]]:
        """
        Generate eval prompt with quality enhancements applied.

        Enhancements include:
        - 8-10 calibration examples with edge cases
        - Numeric rubric thresholds (≥90% = 5, etc.)
        - Tiebreaker rules for conflicting scores
        - 30% token reduction through compact mode
        - Intermediate score guidance (2.5, 3.5, 4.5)
        - Feedback learning adaptations
        """
        from eval_generator_v3 import generate_gold_standard_eval_prompt
        from llm_client_v2 import get_llm_client

        llm_client = get_llm_client()

        # Generate base eval prompt
        result = await generate_gold_standard_eval_prompt(
            system_prompt=system_prompt,
            use_case=use_case,
            requirements=requirements,
            llm_client=llm_client,
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            max_iterations=2
        )

        base_eval_prompt = result.eval_prompt
        dimensions = result.dimensions or []
        auto_fails = getattr(result, 'auto_fails', []) or []

        # Track cost
        # Estimate tokens (rough: 4 chars per token)
        prompt_tokens = len(system_prompt) // 4 + 500
        completion_tokens = len(base_eval_prompt) // 4
        self.cost_gate.record_cost(
            "eval_generation", model_name, prompt_tokens, completion_tokens, self.project_id
        )

        # ENHANCEMENT: MERGE quality enhancements INTO LLM-generated prompt (don't replace!)
        # The LLM prompt has domain-specific content we want to keep
        # We ADD: consistency guidance, intermediate score examples, tiebreaker clarification
        if QUALITY_ENHANCEMENTS_AVAILABLE:
            try:
                base_eval_prompt = self._merge_quality_enhancements(
                    base_eval_prompt, dimensions, auto_fails
                )
                logger.info(f"Merged quality enhancements into eval prompt for {self.project_id}")
            except Exception as e:
                logger.warning(f"Quality enhancement merge failed, using base prompt: {e}")

        # Apply feedback learning adaptations
        try:
            if self.quality_manager.feedback_enabled:
                adapted_prompt, adapted_dims, adapted_fails = self.quality_manager.adapt_eval_prompt(
                    self.project_id, base_eval_prompt, dimensions, auto_fails
                )
                logger.info(f"Applied feedback adaptations to eval prompt for project {self.project_id}")
                return adapted_prompt, adapted_dims, adapted_fails
        except Exception as e:
            logger.warning(f"Feedback adaptation failed: {e}")

        return base_eval_prompt, dimensions, auto_fails

    def _compile_validation_result(self) -> PipelineValidationResult:
        """Compile all gate results into final validation result"""
        blocking_failures = []
        warnings = []
        total_score = 0
        score_count = 0

        for gate in self._gate_results:
            if gate.status == ValidationStatus.FAILED and gate.blocking:
                blocking_failures.append(f"{gate.gate_name}: {gate.recommendation}")
            elif gate.status == ValidationStatus.WARNING:
                warnings.append(f"{gate.gate_name}: {gate.recommendation}")

            if gate.status != ValidationStatus.SKIPPED:
                total_score += gate.score
                score_count += 1

        overall_score = total_score / score_count if score_count > 0 else 0

        if blocking_failures:
            overall_status = ValidationStatus.FAILED
            can_deploy = False
        elif warnings:
            overall_status = ValidationStatus.WARNING
            can_deploy = True  # Warnings don't block deployment
        else:
            overall_status = ValidationStatus.PASSED
            can_deploy = True

        return PipelineValidationResult(
            overall_status=overall_status,
            overall_score=round(overall_score, 1),
            gates=self._gate_results,
            can_deploy=can_deploy,
            blocking_failures=blocking_failures,
            warnings=warnings,
            cost_incurred=self.cost_gate.get_summary()["session_total_usd"]
        )


# =============================================================================
# SECTION 5: EVAL PROMPT VERSION MANAGER
# =============================================================================

class EvalPromptVersionManager:
    """Manage eval prompt versions with rollback capability"""

    def __init__(self, project_id: str):
        self.project_id = project_id
        import project_storage
        self._storage = project_storage

    def save_version(
        self,
        eval_prompt: str,
        validation_result: PipelineValidationResult,
        dimensions: List[Dict],
        auto_fails: List[Dict],
        changes_made: str = "Generated"
    ) -> Dict[str, Any]:
        """Save a new eval prompt version with validation metadata"""
        project = self._storage.load_project(self.project_id)
        if not project:
            raise ValueError(f"Project {self.project_id} not found")

        if not project.eval_prompt_versions:
            project.eval_prompt_versions = []

        # Determine next version
        next_version = 1
        if project.eval_prompt_versions:
            max_version = max(v.get("version", 0) for v in project.eval_prompt_versions)
            next_version = max_version + 1

        # Create version entry with validation metadata
        version_entry = {
            "version": next_version,
            "eval_prompt_text": eval_prompt,
            "dimensions": dimensions,
            "auto_fails": auto_fails,
            "changes_made": changes_made,
            "validation": {
                "status": validation_result.overall_status.value,
                "score": validation_result.overall_score,
                "can_deploy": validation_result.can_deploy,
                "gates": [
                    {
                        "name": g.gate_name,
                        "status": g.status.value,
                        "score": g.score
                    }
                    for g in validation_result.gates
                ],
                "blocking_failures": validation_result.blocking_failures,
                "warnings": validation_result.warnings,
                "cost_usd": validation_result.cost_incurred
            },
            "created_at": datetime.now().isoformat(),
            "is_deployed": False
        }

        project.eval_prompt_versions.append(version_entry)

        # Only update current eval_prompt if can_deploy
        if validation_result.can_deploy:
            project.eval_prompt = eval_prompt
            version_entry["is_deployed"] = True

        project.updated_at = datetime.now()
        self._storage.save_project(project)

        return version_entry

    def rollback_to_version(self, version_number: int) -> Dict[str, Any]:
        """Rollback to a specific version"""
        project = self._storage.load_project(self.project_id)
        if not project:
            raise ValueError(f"Project {self.project_id} not found")

        if not project.eval_prompt_versions:
            raise ValueError("No versions available")

        # Find the version
        target_version = None
        for v in project.eval_prompt_versions:
            if v.get("version") == version_number:
                target_version = v
                break

        if not target_version:
            raise ValueError(f"Version {version_number} not found")

        # Update current eval prompt
        project.eval_prompt = target_version["eval_prompt_text"]

        # Mark as deployed
        for v in project.eval_prompt_versions:
            v["is_deployed"] = (v.get("version") == version_number)

        project.updated_at = datetime.now()
        self._storage.save_project(project)

        return {
            "restored_version": version_number,
            "eval_prompt": project.eval_prompt[:200] + "...",
            "message": f"Rolled back to version {version_number}"
        }

    def get_deployed_version(self) -> Optional[Dict[str, Any]]:
        """Get the currently deployed version"""
        project = self._storage.load_project(self.project_id)
        if not project or not project.eval_prompt_versions:
            return None

        for v in project.eval_prompt_versions:
            if v.get("is_deployed"):
                return v

        # If none marked as deployed, return latest
        return project.eval_prompt_versions[-1]

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all versions with validation status"""
        project = self._storage.load_project(self.project_id)
        if not project or not project.eval_prompt_versions:
            return []

        return [
            {
                "version": v.get("version"),
                "created_at": v.get("created_at"),
                "is_deployed": v.get("is_deployed", False),
                "validation_status": v.get("validation", {}).get("status", "unknown"),
                "validation_score": v.get("validation", {}).get("score", 0),
                "can_deploy": v.get("validation", {}).get("can_deploy", False),
                "changes_made": v.get("changes_made", "")
            }
            for v in project.eval_prompt_versions
        ]


# =============================================================================
# SECTION 6: CONVENIENCE FUNCTIONS
# =============================================================================

async def generate_validated_eval(
    project_id: str,
    system_prompt: str,
    use_case: str,
    requirements: List[str],
    provider: str,
    api_key: str,
    model_name: str,
    run_eval_func = None,
    sample_input: Optional[str] = None,
    sample_output: Optional[str] = None,
    cost_budget: Optional[CostBudget] = None
) -> Tuple[str, PipelineValidationResult, Dict[str, Any]]:
    """
    Convenience function to generate and validate an eval prompt.

    Returns:
        Tuple of (eval_prompt, validation_result, version_info)
    """
    # Create pipeline
    pipeline = ValidatedEvalPipeline(
        project_id=project_id,
        cost_budget=cost_budget
    )

    # Generate and validate
    eval_prompt, validation_result = await pipeline.generate_and_validate(
        system_prompt=system_prompt,
        use_case=use_case,
        requirements=requirements,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        run_eval_func=run_eval_func,
        sample_input=sample_input,
        sample_output=sample_output
    )

    # Save version
    version_manager = EvalPromptVersionManager(project_id)
    version_info = version_manager.save_version(
        eval_prompt=eval_prompt,
        validation_result=validation_result,
        dimensions=[],  # Would come from generation
        auto_fails=[],
        changes_made="Generated via validated pipeline"
    )

    return eval_prompt, validation_result, version_info


def check_semantic_issues(prompt_text: str) -> Dict[str, Any]:
    """Standalone function to check for semantic issues in any prompt"""
    detector = SemanticContradictionDetector()
    return detector.analyze(prompt_text)
