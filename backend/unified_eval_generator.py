"""
Unified Eval Generation System

This module consolidates ALL eval generation logic into one place:
- Single eval prompt generation (from eval_generator_v3)
- Multi-evaluator generation (from multi_eval_with_meta)
- Intelligent requirements extraction (from intelligent_requirements_extractor)
- Meta-evaluation (from meta_evaluator) - ALWAYS INCLUDED
- Validation pipeline (from validated_eval_pipeline)

ALL eval generation should go through this module.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from llm_client_v2 import EnhancedLLMClient

logger = logging.getLogger(__name__)


class EvalMode(Enum):
    """Eval generation mode"""
    SINGLE = "single"  # Single monolithic eval prompt
    MULTI = "multi"  # Multiple separate evaluators
    AUTO = "auto"  # Auto-detect based on complexity


@dataclass
class EvalGenerationConfig:
    """Configuration for eval generation"""
    mode: EvalMode = EvalMode.AUTO
    use_intelligent_extraction: bool = True
    use_meta_evaluation: bool = True  # ALWAYS True by default
    meta_quality_threshold: float = 8.5
    auto_refine_on_low_quality: bool = True
    max_refinement_iterations: int = 3
    use_validation_pipeline: bool = True
    thinking_model: Optional[str] = None


@dataclass
class MetaEvaluationResult:
    """Result from meta-evaluation of an eval prompt"""
    quality_score: float
    passes_gate: bool
    executive_summary: str
    audit_scores: Dict[str, float]
    logic_gaps: List[Dict[str, str]]
    refinement_roadmap: List[str]


@dataclass
class EvalPromptResult:
    """Single eval prompt with meta-evaluation"""
    prompt: str
    meta_evaluation: Optional[MetaEvaluationResult]
    was_refined: bool
    original_prompt: Optional[str]
    validation_passed: bool
    validation_issues: List[str]


@dataclass
class MultiEvaluatorResult:
    """Multiple separate evaluators with meta-evaluation"""
    evaluators: List[Dict[str, Any]]
    extraction_metadata: Dict[str, Any]
    total_evaluators: int


@dataclass
class UnifiedEvalResult:
    """Unified result that can be either single or multi"""
    mode: EvalMode
    single_eval: Optional[EvalPromptResult]
    multi_eval: Optional[MultiEvaluatorResult]
    generation_metadata: Dict[str, Any]


class UnifiedEvalGenerator:
    """
    Unified eval generation system.

    This is the SINGLE SOURCE OF TRUTH for all eval generation.
    All other eval generation code should be deprecated.
    """

    def __init__(
        self,
        llm_client: EnhancedLLMClient,
        config: Optional[EvalGenerationConfig] = None
    ):
        self.llm_client = llm_client
        self.config = config or EvalGenerationConfig()

    async def generate_eval(
        self,
        system_prompt: str,
        requirements: str = "",
        use_case: str = "",
        structured_requirements: Optional[Dict[str, Any]] = None,
        mode: Optional[EvalMode] = None
    ) -> UnifiedEvalResult:
        """
        Generate eval prompt(s) with meta-evaluation.

        This is the MAIN ENTRY POINT for all eval generation.

        Args:
            system_prompt: The system prompt to evaluate
            requirements: Additional requirements
            use_case: Use case description
            structured_requirements: Optional structured requirements
            mode: Override the default mode

        Returns:
            UnifiedEvalResult with single or multi eval prompts
        """

        generation_mode = mode or self.config.mode

        # Auto-detect mode if needed
        if generation_mode == EvalMode.AUTO:
            generation_mode = self._auto_detect_mode(system_prompt, requirements)

        logger.info(f"Starting unified eval generation (mode={generation_mode.value})")

        # Step 1: Intelligent extraction (if enabled)
        extraction_metadata = {}
        extracted_requirements = None

        if self.config.use_intelligent_extraction:
            try:
                from intelligent_requirements_extractor import extract_requirements_intelligently

                logger.info("Using intelligent requirements extraction...")
                extracted_requirements = await extract_requirements_intelligently(
                    system_prompt=system_prompt,
                    requirements=requirements,
                    use_case=use_case,
                    llm_client=self.llm_client,
                    structured_requirements=structured_requirements
                )

                extraction_metadata = {
                    "method": "intelligent_llm_extraction",
                    "domain": extracted_requirements.domain,
                    "risk_level": extracted_requirements.risk_level,
                    "confidence_score": extracted_requirements.confidence_score,
                    "critical_dimensions": extracted_requirements.critical_dimensions
                }

                logger.info(
                    f"Extraction complete: domain={extracted_requirements.domain}, "
                    f"risk={extracted_requirements.risk_level}, "
                    f"confidence={extracted_requirements.confidence_score:.2f}"
                )

            except Exception as e:
                logger.warning(f"Intelligent extraction failed: {e}, continuing without it")
                extraction_metadata = {"method": "none", "error": str(e)}

        # Step 2: Generate based on mode
        if generation_mode == EvalMode.MULTI:
            result = await self._generate_multi_eval(
                system_prompt=system_prompt,
                requirements=requirements,
                use_case=use_case,
                structured_requirements=structured_requirements,
                extracted_requirements=extracted_requirements
            )

            return UnifiedEvalResult(
                mode=EvalMode.MULTI,
                single_eval=None,
                multi_eval=result,
                generation_metadata=extraction_metadata
            )

        else:  # SINGLE mode
            result = await self._generate_single_eval(
                system_prompt=system_prompt,
                requirements=requirements,
                use_case=use_case,
                structured_requirements=structured_requirements,
                extracted_requirements=extracted_requirements
            )

            return UnifiedEvalResult(
                mode=EvalMode.SINGLE,
                single_eval=result,
                multi_eval=None,
                generation_metadata=extraction_metadata
            )

    async def _generate_single_eval(
        self,
        system_prompt: str,
        requirements: str,
        use_case: str,
        structured_requirements: Optional[Dict[str, Any]],
        extracted_requirements: Any
    ) -> EvalPromptResult:
        """Generate single monolithic eval prompt with meta-evaluation"""

        logger.info("Generating single eval prompt...")

        # Use eval_generator_v3 for generation
        from eval_generator_v3 import generate_gold_standard_eval_prompt

        # Get API settings
        from shared_settings import get_settings
        settings = get_settings()
        provider = settings.get("provider", "anthropic")
        api_key = settings.get("api_key", "")
        model_name = settings.get("model_name", "claude-3-5-sonnet-20241022")

        # Generate eval prompt
        eval_prompt_result = await generate_gold_standard_eval_prompt(
            system_prompt=system_prompt,
            requirements=requirements or "",
            use_case=use_case or "",
            llm_client=self.llm_client,
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            thinking_model=self.config.thinking_model,
            structured_requirements=structured_requirements
        )

        eval_prompt = eval_prompt_result.eval_prompt
        original_prompt = eval_prompt

        # Meta-evaluate if enabled
        meta_result = None
        was_refined = False

        if self.config.use_meta_evaluation and eval_prompt:
            logger.info("Running meta-evaluation on single eval prompt...")
            meta_result = await self._meta_evaluate_prompt(
                system_prompt=system_prompt,
                eval_prompt=eval_prompt
            )

            # Auto-refine if quality is low
            if (self.config.auto_refine_on_low_quality and
                not meta_result.passes_gate and
                self.config.max_refinement_iterations > 0):

                logger.info(f"Quality {meta_result.quality_score:.1f}/10, refining...")

                refined_prompt = await self._refine_prompt(
                    eval_prompt=eval_prompt,
                    meta_result=meta_result,
                    max_iterations=self.config.max_refinement_iterations
                )

                # Re-evaluate refined prompt
                meta_result = await self._meta_evaluate_prompt(
                    system_prompt=system_prompt,
                    eval_prompt=refined_prompt
                )

                eval_prompt = refined_prompt
                was_refined = True

                logger.info(f"Refined to quality {meta_result.quality_score:.1f}/10")

        # Validate if enabled
        validation_passed = True
        validation_issues = []

        if self.config.use_validation_pipeline:
            validation_passed, validation_issues = await self._validate_prompt(eval_prompt)

        return EvalPromptResult(
            prompt=eval_prompt,
            meta_evaluation=meta_result,
            was_refined=was_refined,
            original_prompt=original_prompt if was_refined else None,
            validation_passed=validation_passed,
            validation_issues=validation_issues
        )

    async def _generate_multi_eval(
        self,
        system_prompt: str,
        requirements: str,
        use_case: str,
        structured_requirements: Optional[Dict[str, Any]],
        extracted_requirements: Any
    ) -> MultiEvaluatorResult:
        """Generate multiple separate evaluators with meta-evaluation"""

        logger.info("Generating multi-evaluator system...")

        # Use multi_eval_with_meta for generation
        from multi_eval_with_meta import generate_multi_eval_with_meta

        result = await generate_multi_eval_with_meta(
            system_prompt=system_prompt,
            requirements=requirements,
            use_case=use_case,
            llm_client=self.llm_client,
            structured_requirements=structured_requirements,
            meta_quality_threshold=self.config.meta_quality_threshold,
            use_intelligent_extraction=False  # Already extracted above
        )

        # Format evaluators
        evaluators_data = [
            {
                "dimension_name": e.dimension_name,
                "evaluator_type": e.evaluator_type,
                "weight": e.weight,
                "is_critical": e.is_critical,
                "eval_prompt": e.eval_prompt,
                "meta_evaluation": {
                    "quality_score": e.meta_quality_score,
                    "passes_gate": e.meta_passes_gate,
                    "executive_summary": e.meta_executive_summary,
                    "audit_scores": e.meta_audit_scores,
                    "logic_gaps": e.meta_logic_gaps,
                    "refinement_roadmap": e.meta_refinement_roadmap
                },
                "was_refined": e.was_refined
            }
            for e in result.evaluators
        ]

        return MultiEvaluatorResult(
            evaluators=evaluators_data,
            extraction_metadata=result.extraction_metadata,
            total_evaluators=len(evaluators_data)
        )

    async def _meta_evaluate_prompt(
        self,
        system_prompt: str,
        eval_prompt: str
    ) -> MetaEvaluationResult:
        """Meta-evaluate an eval prompt"""

        from meta_evaluator import MetaEvaluator

        meta_evaluator = MetaEvaluator(
            llm_client=self.llm_client,
            quality_threshold=self.config.meta_quality_threshold
        )

        meta_result = await meta_evaluator.evaluate_eval_prompt(
            system_prompt=system_prompt,
            eval_prompt=eval_prompt
        )

        return MetaEvaluationResult(
            quality_score=meta_result.overall_quality_score,
            passes_gate=meta_result.passes_quality_gate,
            executive_summary=meta_result.executive_summary,
            audit_scores={
                "effectiveness": meta_result.effectiveness_score,
                "structural_clarity": meta_result.structural_clarity_score,
                "bias": meta_result.bias_score,
                "metric_conflation": meta_result.metric_conflation_score,
                "granularity": meta_result.granularity_score
            },
            logic_gaps=meta_result.logic_gaps,
            refinement_roadmap=meta_result.refinement_roadmap
        )

    async def _refine_prompt(
        self,
        eval_prompt: str,
        meta_result: MetaEvaluationResult,
        max_iterations: int
    ) -> str:
        """Refine eval prompt based on meta-evaluation feedback"""

        from meta_evaluator import apply_meta_eval_improvements

        refined_prompt = await apply_meta_eval_improvements(
            eval_prompt=eval_prompt,
            meta_result=meta_result,  # Convert to original type
            llm_client=self.llm_client
        )

        return refined_prompt

    async def _validate_prompt(self, eval_prompt: str) -> Tuple[bool, List[str]]:
        """Validate eval prompt using validation pipeline"""

        try:
            from validated_eval_pipeline import ValidatedEvalPipeline

            pipeline = ValidatedEvalPipeline(self.llm_client)

            # Quick validation (not full pipeline)
            issues = []

            # Basic checks
            if len(eval_prompt) < 100:
                issues.append("Eval prompt too short")

            if "{{" not in eval_prompt and "{{" not in eval_prompt:
                issues.append("No template variables found")

            if "score" not in eval_prompt.lower():
                issues.append("No scoring mentioned")

            passes = len(issues) == 0

            return passes, issues

        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return True, []  # Don't block on validation failure

    def _auto_detect_mode(self, system_prompt: str, requirements: str) -> EvalMode:
        """Auto-detect whether to use single or multi mode"""

        combined = f"{system_prompt}\n{requirements}".lower()

        # Heuristics for multi-evaluator mode
        multi_indicators = [
            len(combined) > 2000,  # Long prompt
            combined.count("must") > 5,  # Many requirements
            any(word in combined for word in ["json", "schema", "format"]),  # Structured output
            any(word in combined for word in ["medical", "financial", "legal"]),  # High-risk domain
        ]

        if sum(multi_indicators) >= 2:
            logger.info("Auto-detected MULTI mode (complex requirements)")
            return EvalMode.MULTI
        else:
            logger.info("Auto-detected SINGLE mode (simple requirements)")
            return EvalMode.SINGLE


# Convenience function
async def generate_eval_unified(
    system_prompt: str,
    requirements: str,
    use_case: str,
    llm_client: EnhancedLLMClient,
    mode: EvalMode = EvalMode.AUTO,
    use_meta_evaluation: bool = True,
    structured_requirements: Optional[Dict[str, Any]] = None
) -> UnifiedEvalResult:
    """
    Convenience function for unified eval generation.

    This is the recommended way to generate eval prompts.
    """

    config = EvalGenerationConfig(
        mode=mode,
        use_meta_evaluation=use_meta_evaluation
    )

    generator = UnifiedEvalGenerator(llm_client, config)

    return await generator.generate_eval(
        system_prompt=system_prompt,
        requirements=requirements,
        use_case=use_case,
        structured_requirements=structured_requirements
    )
