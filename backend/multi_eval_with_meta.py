"""
Multi-Evaluator System with Meta-Evaluation for Each Dimension

Generates separate eval prompts for each dimension (Accuracy, Relevance, Safety, etc.)
and meta-evaluates each one individually.

Now supports intelligent LLM-based requirements extraction for better accuracy.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from dimension_detector import DimensionDetector
from evaluator_factory import EvaluatorFactory
from meta_evaluator import MetaEvaluator
from llm_client_v2 import EnhancedLLMClient

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorWithMeta:
    """Single evaluator with its prompt and meta-evaluation results"""
    dimension_name: str
    evaluator_type: str
    weight: float
    is_critical: bool

    # The eval prompt for this dimension
    eval_prompt: str

    # Meta-evaluation results
    meta_quality_score: float
    meta_passes_gate: bool
    meta_executive_summary: str
    meta_audit_scores: Dict[str, Any]
    meta_logic_gaps: List[Dict[str, str]]
    meta_refinement_roadmap: List[str]

    # Original vs refined
    original_prompt: str
    was_refined: bool


@dataclass
class MultiEvalResult:
    """Result containing evaluators and extraction metadata"""
    evaluators: List[EvaluatorWithMeta]
    extraction_metadata: Dict[str, Any]


async def generate_multi_eval_with_meta(
    system_prompt: str,
    requirements: str,
    use_case: str,
    llm_client: EnhancedLLMClient,
    structured_requirements: Optional[Dict[str, Any]] = None,
    meta_quality_threshold: float = 7.5,
    use_intelligent_extraction: bool = True
) -> MultiEvalResult:
    """
    Generate separate eval prompts for each dimension and meta-evaluate each one.

    Args:
        system_prompt: The system prompt to evaluate
        requirements: Additional requirements text
        use_case: Use case description
        llm_client: LLM client for generation and meta-evaluation
        structured_requirements: Optional structured requirements
        meta_quality_threshold: Quality threshold for auto-refinement (default: 7.5)
        use_intelligent_extraction: Use LLM for intelligent extraction (default: True)

    Returns:
        List of EvaluatorWithMeta - one for each detected dimension
    """

    logger.info(
        f"Generating multi-evaluator with meta-evaluation "
        f"(intelligent_extraction={use_intelligent_extraction})..."
    )

    factory = EvaluatorFactory()
    extraction_metadata = {}
    extracted_requirements = None

    # Step 1 & 2: Detect dimensions and create evaluators
    if use_intelligent_extraction:
        try:
            # Use intelligent LLM-based extraction
            from intelligent_requirements_extractor import extract_requirements_intelligently
            from dataclasses import asdict

            logger.info("Using intelligent LLM-based requirements extraction...")

            extracted_requirements = await extract_requirements_intelligently(
                system_prompt=system_prompt,
                requirements=requirements,
                use_case=use_case,
                llm_client=llm_client,
                structured_requirements=structured_requirements
            )

            logger.info(
                f"Intelligent extraction complete: domain={extracted_requirements.domain}, "
                f"risk_level={extracted_requirements.risk_level}, "
                f"confidence={extracted_requirements.confidence_score:.2f}"
            )

            # Store extraction metadata for response
            extraction_metadata = {
                "method": "intelligent_llm_extraction",
                "domain": extracted_requirements.domain,
                "risk_level": extracted_requirements.risk_level,
                "confidence_score": extracted_requirements.confidence_score,
                "use_case": extracted_requirements.use_case,
                "primary_function": extracted_requirements.primary_function,
                "must_do": extracted_requirements.must_do,
                "must_not_do": extracted_requirements.must_not_do,
                "quality_priorities": extracted_requirements.quality_priorities,
                "critical_dimensions": extracted_requirements.critical_dimensions,
                "important_dimensions": extracted_requirements.important_dimensions,
                "tone": extracted_requirements.tone,
                "extraction_notes": extracted_requirements.extraction_notes
            }

            # Create evaluators from intelligent extraction
            evaluators = factory.create_evaluators_from_extracted_requirements(
                extracted_requirements
            )

        except Exception as e:
            logger.warning(f"Intelligent extraction failed: {e}, falling back to regex detection")
            use_intelligent_extraction = False

    if not use_intelligent_extraction:
        # Fallback to regex-based detection
        logger.info("Using legacy regex-based detection...")

        extraction_metadata = {
            "method": "regex_pattern_matching",
            "note": "Using legacy regex-based detection"
        }

        evaluators = factory.create_evaluators_from_prompt(
            system_prompt=system_prompt,
            requirements=requirements,
            use_case=use_case,
            structured_requirements=structured_requirements
        )

    logger.info(f"Created {len(evaluators)} evaluators")

    # Step 3: For each evaluator, extract its prompt and meta-evaluate
    results = []
    meta_evaluator = MetaEvaluator(llm_client, quality_threshold=meta_quality_threshold)

    for evaluator in evaluators:
        logger.info(f"Processing {evaluator.dimension_name}...")

        # Build the eval prompt for this dimension
        context = {
            'system_prompt': system_prompt,
            'requirements': requirements,
            'use_case': use_case,
            'input': '{{input}}',  # Template variable
            'output': '{{output}}'  # Template variable
        }

        eval_prompt = evaluator.build_prompt(context)
        original_prompt = eval_prompt

        # Meta-evaluate this eval prompt
        try:
            meta_result = await meta_evaluator.evaluate_eval_prompt(
                system_prompt=system_prompt,
                eval_prompt=eval_prompt
            )

            # If it doesn't pass quality gate, try to refine
            was_refined = False
            if not meta_result.passes_quality_gate:
                logger.info(f"{evaluator.dimension_name}: Quality {meta_result.overall_quality_score:.1f}/10, refining...")

                from meta_evaluator import apply_meta_eval_improvements
                refined_prompt = await apply_meta_eval_improvements(
                    eval_prompt=eval_prompt,
                    meta_result=meta_result,
                    llm_client=llm_client
                )

                # Re-evaluate the refined prompt
                meta_result = await meta_evaluator.evaluate_eval_prompt(
                    system_prompt=system_prompt,
                    eval_prompt=refined_prompt
                )

                eval_prompt = refined_prompt
                was_refined = True
                logger.info(f"{evaluator.dimension_name}: Refined to {meta_result.overall_quality_score:.1f}/10")

            results.append(EvaluatorWithMeta(
                dimension_name=evaluator.dimension_name,
                evaluator_type=evaluator.__class__.__name__,
                weight=evaluator.weight,
                is_critical=evaluator.is_critical,
                eval_prompt=eval_prompt,
                meta_quality_score=meta_result.overall_quality_score,
                meta_passes_gate=meta_result.passes_quality_gate,
                meta_executive_summary=meta_result.executive_summary,
                meta_audit_scores={
                    "effectiveness": meta_result.effectiveness_score,
                    "structural_clarity": meta_result.structural_clarity_score,
                    "bias": meta_result.bias_score,
                    "metric_conflation": meta_result.metric_conflation_score,
                    "granularity": meta_result.granularity_score
                },
                meta_logic_gaps=meta_result.logic_gaps,
                meta_refinement_roadmap=meta_result.refinement_roadmap,
                original_prompt=original_prompt,
                was_refined=was_refined
            ))

        except Exception as e:
            logger.error(f"Error meta-evaluating {evaluator.dimension_name}: {e}")
            # Include it anyway with default meta scores
            results.append(EvaluatorWithMeta(
                dimension_name=evaluator.dimension_name,
                evaluator_type=evaluator.__class__.__name__,
                weight=evaluator.weight,
                is_critical=evaluator.is_critical,
                eval_prompt=eval_prompt,
                meta_quality_score=0.0,
                meta_passes_gate=False,
                meta_executive_summary=f"Meta-evaluation failed: {str(e)}",
                meta_audit_scores={},
                meta_logic_gaps=[],
                meta_refinement_roadmap=[],
                original_prompt=original_prompt,
                was_refined=False
            ))

    logger.info(f"Completed meta-evaluation for {len(results)} evaluators")

    # Return results with extraction metadata
    return MultiEvalResult(
        evaluators=results,
        extraction_metadata=extraction_metadata
    )
