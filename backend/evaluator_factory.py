"""
Evaluator Factory

Automatically generates evaluator instances based on detected dimensions.

Supports both:
1. Legacy regex-based detection
2. Intelligent LLM-based detection with ExtractedRequirements
"""

import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dimension_detector import (
    DimensionDetector,
    DetectedDimension,
    extract_required_elements,
    extract_safety_requirements,
    extract_tone_requirements,
    extract_output_schema
)

if TYPE_CHECKING:
    from intelligent_requirements_extractor import ExtractedRequirements
from specialized_evaluators import (
    SchemaValidator,
    SafetyChecker,
    CompletenessChecker,
    AccuracyEvaluator,
    ActionabilityEvaluator,
    RelevanceEvaluator,
    ToneEvaluator,
    DomainSafetyChecker,
    HallucinationDetector,
    DomainAccuracyEvaluator,
    PrecisionEvaluator,
    EmpathyEvaluator
)
from multi_evaluator_system import BaseEvaluator, MultiEvaluatorOrchestrator

logger = logging.getLogger(__name__)


class EvaluatorFactory:
    """
    Factory that creates appropriate evaluators based on
    automatically detected dimensions
    """

    def __init__(self):
        self.detector = DimensionDetector()

    def create_evaluators_from_prompt(
        self,
        system_prompt: str,
        requirements: str,
        use_case: str,
        structured_requirements: Optional[Dict[str, Any]] = None
    ) -> List[BaseEvaluator]:
        """
        Automatically detect dimensions and create appropriate evaluators

        Returns:
            List of specialized evaluators ready to use
        """

        # Step 1: Detect dimensions
        detected_dimensions = self.detector.detect_dimensions(
            system_prompt=system_prompt,
            requirements=requirements,
            use_case=use_case,
            structured_requirements=structured_requirements
        )

        logger.info(f"Auto-detected {len(detected_dimensions)} dimensions to evaluate")

        # Step 2: Create evaluators for each dimension
        evaluators = []

        for dim in detected_dimensions:
            evaluator = self._create_evaluator_for_dimension(
                dimension=dim,
                system_prompt=system_prompt,
                requirements=requirements,
                structured_requirements=structured_requirements
            )

            if evaluator:
                evaluators.append(evaluator)
                logger.info(
                    f"Created {dim.evaluator_type} (weight={dim.weight:.2f}, "
                    f"critical={dim.is_critical}): {dim.detection_reason}"
                )

        if not evaluators:
            logger.warning("No evaluators created! Using default set.")
            evaluators = self._create_default_evaluators()

        return evaluators

    def create_evaluators_from_extracted_requirements(
        self,
        extracted_requirements: 'ExtractedRequirements'
    ) -> List[BaseEvaluator]:
        """
        Create evaluators using intelligent LLM-extracted requirements.
        This is more accurate than regex-based detection.

        Args:
            extracted_requirements: Requirements extracted by IntelligentRequirementsExtractor

        Returns:
            List of specialized evaluators tailored to extracted requirements
        """

        logger.info(
            f"Creating evaluators from intelligent extraction: "
            f"domain={extracted_requirements.domain}, risk={extracted_requirements.risk_level}"
        )

        # Use intelligent dimension detection
        detected_dimensions = self.detector.detect_dimensions_from_extracted(
            extracted_requirements
        )

        # Create evaluators for each dimension
        evaluators = []

        for dim in detected_dimensions:
            evaluator = self._create_evaluator_from_extracted(
                dimension=dim,
                extracted_requirements=extracted_requirements
            )

            if evaluator:
                evaluators.append(evaluator)
                logger.info(
                    f"Created {dim.evaluator_type} (weight={dim.weight:.2f}, "
                    f"critical={dim.is_critical}): {dim.detection_reason}"
                )

        if not evaluators:
            logger.warning("No evaluators created! Using default set.")
            evaluators = self._create_default_evaluators()

        return evaluators

    def _create_evaluator_from_extracted(
        self,
        dimension: DetectedDimension,
        extracted_requirements: 'ExtractedRequirements'
    ) -> Optional[BaseEvaluator]:
        """Create evaluator instance using extracted requirements"""

        evaluator_type = dimension.evaluator_type

        try:
            if evaluator_type == "SchemaValidator":
                schema = extracted_requirements.output_format
                if schema and schema.get("type") in ["json", "structured", "xml", "yaml"]:
                    return SchemaValidator(
                        expected_schema=schema.get("schema", {}),
                        weight=dimension.weight
                    )
                return None

            elif evaluator_type == "DomainSafetyChecker":
                return DomainSafetyChecker(
                    domain=extracted_requirements.domain,
                    safety_requirements=extracted_requirements.must_not_do,
                    weight=dimension.weight
                )

            elif evaluator_type == "SafetyChecker":
                return SafetyChecker(
                    safety_requirements=extracted_requirements.must_not_do,
                    weight=dimension.weight
                )

            elif evaluator_type == "CompletenessChecker":
                if extracted_requirements.must_do:
                    return CompletenessChecker(
                        required_elements=extracted_requirements.must_do,
                        weight=dimension.weight
                    )
                return None

            elif evaluator_type == "DomainAccuracyEvaluator":
                return DomainAccuracyEvaluator(
                    domain=extracted_requirements.domain,
                    weight=dimension.weight
                )

            elif evaluator_type == "AccuracyEvaluator":
                return AccuracyEvaluator(weight=dimension.weight)

            elif evaluator_type == "HallucinationDetector":
                return HallucinationDetector(weight=dimension.weight)

            elif evaluator_type == "ActionabilityEvaluator":
                return ActionabilityEvaluator(weight=dimension.weight)

            elif evaluator_type == "RelevanceEvaluator":
                return RelevanceEvaluator(weight=dimension.weight)

            elif evaluator_type == "ToneEvaluator":
                return ToneEvaluator(
                    expected_tone=extracted_requirements.tone,
                    weight=dimension.weight
                )

            elif evaluator_type == "PrecisionEvaluator":
                return PrecisionEvaluator(weight=dimension.weight)

            elif evaluator_type == "EmpathyEvaluator":
                return EmpathyEvaluator(weight=dimension.weight)

            else:
                logger.warning(f"Unknown evaluator type: {evaluator_type}")
                return None

        except Exception as e:
            logger.error(f"Error creating {evaluator_type}: {e}")
            return None

    def _create_evaluator_for_dimension(
        self,
        dimension: DetectedDimension,
        system_prompt: str,
        requirements: str,
        structured_requirements: Optional[Dict[str, Any]]
    ) -> Optional[BaseEvaluator]:
        """Create specific evaluator instance for a dimension"""

        evaluator_type = dimension.evaluator_type

        try:
            if evaluator_type == "SchemaValidator":
                schema = extract_output_schema(
                    system_prompt, requirements, structured_requirements
                )
                if schema:
                    return SchemaValidator(
                        expected_schema=schema,
                        weight=dimension.weight
                    )
                else:
                    logger.warning("Schema dimension detected but no schema found")
                    return None

            elif evaluator_type == "SafetyChecker":
                safety_reqs = extract_safety_requirements(
                    system_prompt, requirements, structured_requirements
                )
                return SafetyChecker(
                    safety_requirements=safety_reqs,
                    weight=dimension.weight
                )

            elif evaluator_type == "CompletenessChecker":
                required_elements = extract_required_elements(
                    system_prompt, requirements, structured_requirements
                )
                if required_elements:
                    return CompletenessChecker(
                        required_elements=required_elements,
                        weight=dimension.weight
                    )
                else:
                    logger.warning("Completeness dimension detected but no required elements found")
                    return None

            elif evaluator_type == "AccuracyEvaluator":
                return AccuracyEvaluator(weight=dimension.weight)

            elif evaluator_type == "ActionabilityEvaluator":
                return ActionabilityEvaluator(weight=dimension.weight)

            elif evaluator_type == "RelevanceEvaluator":
                return RelevanceEvaluator(weight=dimension.weight)

            elif evaluator_type == "ToneEvaluator":
                tone_requirements = extract_tone_requirements(
                    system_prompt, requirements, structured_requirements
                )
                return ToneEvaluator(
                    expected_tone=tone_requirements,
                    weight=dimension.weight
                )

            # Domain-specific evaluators
            elif evaluator_type == "DomainSafetyChecker":
                # Detect domain from context
                domain = self._detect_domain(system_prompt, requirements)
                safety_reqs = extract_safety_requirements(
                    system_prompt, requirements, structured_requirements
                )
                return DomainSafetyChecker(
                    domain=domain,
                    safety_requirements=safety_reqs,
                    weight=dimension.weight
                )

            elif evaluator_type == "HallucinationDetector":
                return HallucinationDetector(weight=dimension.weight)

            elif evaluator_type == "DomainAccuracyEvaluator":
                domain = self._detect_domain(system_prompt, requirements)
                return DomainAccuracyEvaluator(
                    domain=domain,
                    weight=dimension.weight
                )

            elif evaluator_type == "PrecisionEvaluator":
                return PrecisionEvaluator(weight=dimension.weight)

            elif evaluator_type == "EmpathyEvaluator":
                return EmpathyEvaluator(weight=dimension.weight)

            else:
                logger.warning(f"Unknown evaluator type: {evaluator_type}")
                return None

        except Exception as e:
            logger.error(f"Error creating {evaluator_type}: {e}")
            return None

    def _detect_domain(self, system_prompt: str, requirements: str) -> str:
        """Detect domain from system prompt and requirements"""

        combined = f"{system_prompt}\n{requirements}".lower()

        # Check for domain keywords
        if any(word in combined for word in ["medical", "health", "diagnosis", "patient", "doctor"]):
            return "medical"
        elif any(word in combined for word in ["financial", "banking", "trading", "investment", "money"]):
            return "financial"
        elif any(word in combined for word in ["legal", "law", "contract", "compliance", "regulation"]):
            return "legal"
        elif any(word in combined for word in ["customer", "support", "service", "help desk"]):
            return "customer_service"
        elif any(word in combined for word in ["technical", "engineering", "code", "api", "software"]):
            return "technical"

        return "general"

    def _create_default_evaluators(self) -> List[BaseEvaluator]:
        """Create a sensible default set of evaluators"""

        logger.info("Creating default evaluator set")

        return [
            SafetyChecker(weight=0.15),
            AccuracyEvaluator(weight=0.35),
            RelevanceEvaluator(weight=0.25),
            ActionabilityEvaluator(weight=0.25)
        ]

    def create_orchestrator(
        self,
        system_prompt: str,
        requirements: str,
        use_case: str,
        structured_requirements: Optional[Dict[str, Any]] = None,
        aggregation_strategy: str = "weighted_average"
    ) -> MultiEvaluatorOrchestrator:
        """
        One-stop method: detect dimensions, create evaluators, and create orchestrator

        Returns:
            Orchestrator ready to evaluate outputs
        """

        evaluators = self.create_evaluators_from_prompt(
            system_prompt=system_prompt,
            requirements=requirements,
            use_case=use_case,
            structured_requirements=structured_requirements
        )

        orchestrator = MultiEvaluatorOrchestrator(
            evaluators=evaluators,
            aggregation_strategy=aggregation_strategy
        )

        logger.info(
            f"Created orchestrator with {len(evaluators)} evaluators: "
            f"{[e.dimension_name for e in evaluators]}"
        )

        return orchestrator


def create_multi_evaluator_system(
    system_prompt: str,
    requirements: str,
    use_case: str,
    structured_requirements: Optional[Dict[str, Any]] = None
) -> MultiEvaluatorOrchestrator:
    """
    Convenience function to create a complete multi-evaluator system

    Usage:
        orchestrator = create_multi_evaluator_system(
            system_prompt="You are a customer service bot...",
            requirements="Must be professional and helpful...",
            use_case="Customer support automation"
        )

        result = await orchestrator.evaluate(
            llm_client=client,
            input_data="How do I reset my password?",
            output="To reset your password...",
            context={"provider": "anthropic"}
        )
    """

    factory = EvaluatorFactory()
    return factory.create_orchestrator(
        system_prompt=system_prompt,
        requirements=requirements,
        use_case=use_case,
        structured_requirements=structured_requirements
    )
