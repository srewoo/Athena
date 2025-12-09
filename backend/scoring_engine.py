"""
Weighted Scoring Engine for Athena
Implements intelligent scoring with use-case profiles and weighted criteria
"""

from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CriteriaWeight(Enum):
    """Criteria importance levels"""
    CRITICAL = 3.0
    IMPORTANT = 2.0
    STANDARD = 1.0


class UseCaseProfile(Enum):
    """Predefined use case profiles"""
    CODE_GENERATION = "code_generation"
    CREATIVE_WRITING = "creative_writing"
    DATA_ANALYSIS = "data_analysis"
    CUSTOMER_SERVICE = "customer_service"
    EDUCATION = "education"
    RESEARCH = "research"
    GENERAL = "general"


# Weighted scoring framework
EVALUATION_FRAMEWORK = {
    "criteria_groups": {
        "critical": {
            "weight": 3.0,
            "criteria": [
                "Clarity & Specificity",
                "Explicit Task Definition",
                "Instruction Consistency",
                "Desired Output Format / Style",
                "Hallucination Minimization",
            ]
        },
        "important": {
            "weight": 2.0,
            "criteria": [
                "Context / Background Provided",
                "Avoiding Ambiguity or Contradictions",
                "Delimiter Strategy",
                "Reasoning Strategy Definition",
                "Tool Use Specification",
                "Error Handling",
                "Safe Failure Mode",
            ]
        },
        "standard": {
            "weight": 1.0,
            "criteria": [
                "Use of Role or Persona",
                "Step-by-Step Reasoning Encouraged",
                "Structured / Numbered Instructions",
                "Verbosity Specification",
                "Context Organization",
                "System Prompt Reminders",
                "Tool Calling Patterns",
                "Prompt Structure Quality",
                "Metaprompt Capability",
                "Long Context Optimization",
                "OpenAI Optimization",
                "Claude Optimization",
                "Gemini Optimization",
                "Multi-Provider Compatibility",
                "Reasoning Effort Control",
                "Brevity vs. Detail Balance",
                "Iteration / Refinement Potential",
                "Examples or Demonstrations",
                "Handling Uncertainty / Gaps",
                "Knowledge Boundary Awareness",
                "Audience Specification",
                "Style Emulation or Imitation",
                "Memory Anchoring (Multi-Turn)",
                "Meta-Cognition Triggers",
                "Agentic Workflow Design",
                "Execution Feedback Loops",
                "Planning-Induced Reasoning",
                "Parallel Tool Safety",
                "Document Formatting",
                "Context Size Optimization",
                "Information Hierarchy",
                "Retrieval Optimization",
                "Context Reliance Tuning",
                "Ethical Alignment",
                "Output Validation Hooks",
                "Limitations Disclosure",
                "Self-Repair Loops",
                "Model Fit / Scenario Appropriateness",
                "Feasibility within Model Constraints",
            ]
        }
    },
    
    "use_case_profiles": {
        UseCaseProfile.CODE_GENERATION.value: {
            "critical_boost": [
                "Explicit Task Definition",
                "Desired Output Format / Style",
                "Error Handling",
                "Output Validation Hooks",
            ],
            "boost_factors": {
                "Structured / Numbered Instructions": 1.5,
                "Examples or Demonstrations": 1.4,
                "Tool Use Specification": 1.3,
            }
        },
        UseCaseProfile.CREATIVE_WRITING.value: {
            "critical_boost": [
                "Use of Role or Persona",
                "Style Emulation or Imitation",
                "Context / Background Provided",
            ],
            "boost_factors": {
                "Verbosity Specification": 1.3,
                "Audience Specification": 1.4,
                "Examples or Demonstrations": 1.2,
            }
        },
        UseCaseProfile.DATA_ANALYSIS.value: {
            "critical_boost": [
                "Hallucination Minimization",
                "Output Validation Hooks",
                "Error Handling",
                "Reasoning Strategy Definition",
            ],
            "boost_factors": {
                "Tool Use Specification": 1.5,
                "Step-by-Step Reasoning Encouraged": 1.4,
                "Context Organization": 1.3,
            }
        },
        UseCaseProfile.CUSTOMER_SERVICE.value: {
            "critical_boost": [
                "Use of Role or Persona",
                "Safe Failure Mode",
                "Ethical Alignment",
                "Handling Uncertainty / Gaps",
            ],
            "boost_factors": {
                "Audience Specification": 1.4,
                "Memory Anchoring (Multi-Turn)": 1.3,
                "Style Emulation or Imitation": 1.2,
            }
        },
        UseCaseProfile.EDUCATION.value: {
            "critical_boost": [
                "Clarity & Specificity",
                "Step-by-Step Reasoning Encouraged",
                "Examples or Demonstrations",
                "Knowledge Boundary Awareness",
            ],
            "boost_factors": {
                "Structured / Numbered Instructions": 1.4,
                "Audience Specification": 1.3,
                "Iteration / Refinement Potential": 1.2,
            }
        },
        UseCaseProfile.RESEARCH.value: {
            "critical_boost": [
                "Hallucination Minimization",
                "Knowledge Boundary Awareness",
                "Context / Background Provided",
                "Limitations Disclosure",
            ],
            "boost_factors": {
                "Reasoning Strategy Definition": 1.4,
                "Retrieval Optimization": 1.3,
                "Information Hierarchy": 1.3,
            }
        },
        UseCaseProfile.GENERAL.value: {
            "critical_boost": [],
            "boost_factors": {}
        }
    }
}


class WeightedScoringEngine:
    """Calculates weighted scores based on use case and criteria importance"""
    
    def __init__(self):
        self.framework = EVALUATION_FRAMEWORK
    
    def get_criteria_weight(self, criterion_name: str) -> float:
        """
        Get the base weight for a criterion
        
        Args:
            criterion_name: Name of the criterion
            
        Returns:
            Weight multiplier (3.0, 2.0, or 1.0)
        """
        for group_name, group_data in self.framework["criteria_groups"].items():
            if criterion_name in group_data["criteria"]:
                return group_data["weight"]
        
        # Default to standard weight
        return 1.0
    
    def get_use_case_boost(
        self,
        criterion_name: str,
        use_case: UseCaseProfile
    ) -> float:
        """
        Get use-case specific boost factor for a criterion
        
        Args:
            criterion_name: Name of the criterion
            use_case: Use case profile
            
        Returns:
            Boost multiplier (default 1.0)
        """
        profile = self.framework["use_case_profiles"].get(use_case.value, {})
        
        # Check if this criterion gets a boost
        boost_factors = profile.get("boost_factors", {})
        if criterion_name in boost_factors:
            return boost_factors[criterion_name]
        
        # Check if this criterion is critical for this use case
        critical_boost = profile.get("critical_boost", [])
        if criterion_name in critical_boost:
            return 1.5  # Default critical boost
        
        return 1.0
    
    def calculate_weighted_score(
        self,
        criteria_scores: List[Dict[str, Any]],
        use_case: UseCaseProfile = UseCaseProfile.GENERAL
    ) -> Dict[str, Any]:
        """
        Calculate weighted total score
        
        Args:
            criteria_scores: List of criterion score dictionaries
            use_case: Use case profile for context-aware scoring
            
        Returns:
            Dictionary with weighted scores and analysis
        """
        weighted_sum = 0.0
        max_weighted_sum = 0.0
        
        criterion_weights = []
        
        for criterion_data in criteria_scores:
            criterion_name = criterion_data.get("criterion", "")
            raw_score = criterion_data.get("score", 0)
            
            # Get base weight
            base_weight = self.get_criteria_weight(criterion_name)
            
            # Get use-case boost
            use_case_boost = self.get_use_case_boost(criterion_name, use_case)
            
            # Calculate final weight
            final_weight = base_weight * use_case_boost
            
            # Calculate weighted score
            weighted_score = raw_score * final_weight
            max_weighted_score = 5 * final_weight  # Max score is 5
            
            weighted_sum += weighted_score
            max_weighted_sum += max_weighted_score
            
            criterion_weights.append({
                "criterion": criterion_name,
                "raw_score": raw_score,
                "base_weight": base_weight,
                "use_case_boost": use_case_boost,
                "final_weight": final_weight,
                "weighted_score": weighted_score,
                "max_weighted_score": max_weighted_score
            })
        
        # Calculate percentage
        percentage = (weighted_sum / max_weighted_sum * 100) if max_weighted_sum > 0 else 0
        
        # Identify top strengths and weaknesses
        sorted_by_score = sorted(
            criterion_weights,
            key=lambda x: x["weighted_score"],
            reverse=True
        )
        
        top_strengths = sorted_by_score[:5]
        top_weaknesses = sorted_by_score[-5:][::-1]  # Reverse to show worst first
        
        # Calculate category breakdowns
        category_scores = self._calculate_category_scores(criterion_weights)
        
        return {
            "weighted_total_score": round(weighted_sum, 2),
            "max_weighted_score": round(max_weighted_sum, 2),
            "percentage": round(percentage, 1),
            "use_case": use_case.value,
            "criterion_weights": criterion_weights,
            "top_strengths": top_strengths,
            "top_weaknesses": top_weaknesses,
            "category_breakdown": category_scores,
            "scoring_methodology": {
                "critical_weight": 3.0,
                "important_weight": 2.0,
                "standard_weight": 1.0,
                "use_case_boosts_applied": use_case != UseCaseProfile.GENERAL
            }
        }
    
    def _calculate_category_scores(
        self,
        criterion_weights: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate scores by category (critical, important, standard)"""
        categories = {
            "critical": {"score": 0, "max": 0, "count": 0},
            "important": {"score": 0, "max": 0, "count": 0},
            "standard": {"score": 0, "max": 0, "count": 0}
        }
        
        for cw in criterion_weights:
            base_weight = cw["base_weight"]
            
            # Determine category
            if base_weight == 3.0:
                category = "critical"
            elif base_weight == 2.0:
                category = "important"
            else:
                category = "standard"
            
            categories[category]["score"] += cw["weighted_score"]
            categories[category]["max"] += cw["max_weighted_score"]
            categories[category]["count"] += 1
        
        # Calculate percentages
        for category in categories.values():
            if category["max"] > 0:
                category["percentage"] = round(
                    (category["score"] / category["max"]) * 100, 1
                )
            else:
                category["percentage"] = 0
        
        return categories
    
    def get_recommendations(
        self,
        weighted_analysis: Dict[str, Any],
        min_acceptable_score: float = 70.0
    ) -> List[str]:
        """
        Generate improvement recommendations based on weighted analysis
        
        Args:
            weighted_analysis: Output from calculate_weighted_score
            min_acceptable_score: Minimum acceptable percentage score
            
        Returns:
            List of prioritized recommendations
        """
        recommendations = []
        
        percentage = weighted_analysis["percentage"]
        
        # Overall score recommendations
        if percentage < min_acceptable_score:
            recommendations.append(
                f"⚠️ Overall score ({percentage}%) is below acceptable threshold ({min_acceptable_score}%). "
                "Focus on critical and important criteria."
            )
        
        # Critical criteria recommendations
        critical_weaknesses = [
            w for w in weighted_analysis["top_weaknesses"]
            if w["base_weight"] == 3.0 and w["raw_score"] <= 2
        ]
        
        if critical_weaknesses:
            recommendations.append(
                f"🔴 Critical weaknesses found: {', '.join([w['criterion'] for w in critical_weaknesses])}. "
                "These should be addressed immediately."
            )
        
        # Important criteria recommendations
        important_weaknesses = [
            w for w in weighted_analysis["top_weaknesses"]
            if w["base_weight"] == 2.0 and w["raw_score"] <= 2
        ]
        
        if important_weaknesses:
            recommendations.append(
                f"🟡 Important areas needing improvement: {', '.join([w['criterion'] for w in important_weaknesses[:3]])}."
            )
        
        # Use-case specific recommendations
        use_case = weighted_analysis["use_case"]
        if use_case != UseCaseProfile.GENERAL.value:
            recommendations.append(
                f"📊 Optimizing for {use_case.replace('_', ' ').title()} use case. "
                "Boosted criteria are weighted higher."
            )
        
        return recommendations


def get_scoring_engine() -> WeightedScoringEngine:
    """Get singleton scoring engine instance"""
    return WeightedScoringEngine()
