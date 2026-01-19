"""
PRD Document Extractor

Automatically extracts structured requirements from Product Requirements Documents (PRDs)
using LLM analysis. Extracts must-do behaviors, forbidden behaviors, constraints,
success criteria, edge cases, and other key information.
"""
from typing import Dict, List, Optional, Any
from llm_client_v2 import EnhancedLLMClient, parse_json_response
import logging

logger = logging.getLogger(__name__)


async def extract_requirements_from_prd(
    prd_text: str,
    llm_client: EnhancedLLMClient,
    provider: str,
    api_key: str,
    model_name: str
) -> Dict[str, Any]:
    """
    Extract structured requirements from a PRD document using LLM analysis.
    
    Args:
        prd_text: The full PRD document text
        llm_client: LLM client for analysis
        provider: LLM provider
        api_key: API key
        model_name: Model to use
        
    Returns:
        Dict with extracted requirements:
        - must_do: List of required behaviors
        - must_not_do: List of forbidden behaviors
        - constraints: List of constraints
        - success_criteria: List of success metrics
        - edge_cases: List of edge cases
        - user_personas: List of target users
        - business_goals: Business objectives
        - summary: One-sentence summary
    """
    
    system_prompt = """You are an expert requirements analyst. Extract structured requirements from the provided PRD document.

Your task is to analyze the PRD and extract key information into a structured format.

Return a JSON object with:
{
    "summary": "One sentence describing what this system/feature does",
    "must_do": [
        "Required behavior 1",
        "Required behavior 2",
        "Required behavior 3"
    ],
    "must_not_do": [
        "Forbidden behavior 1",
        "Forbidden behavior 2"
    ],
    "constraints": [
        "Technical or business constraint 1",
        "Technical or business constraint 2"
    ],
    "success_criteria": [
        "Measurable success metric 1",
        "Measurable success metric 2"
    ],
    "edge_cases": [
        "Edge case scenario 1",
        "Edge case scenario 2"
    ],
    "user_personas": [
        "Target user type 1",
        "Target user type 2"
    ],
    "business_goals": [
        "Business objective 1",
        "Business objective 2"
    ]
}

IMPORTANT:
- Extract ACTUAL requirements from the PRD, don't make up generic ones
- Be specific and actionable
- If a section is not mentioned in the PRD, return an empty array []
- Focus on what's explicitly stated or strongly implied
- Keep each item concise (1-2 sentences max)"""

    user_message = f"""Analyze this PRD document and extract structured requirements:

```
{prd_text[:10000]}  
```

Extract all key requirements, constraints, success criteria, edge cases, and context from this PRD.
Return ONLY valid JSON matching the specified format."""

    try:
        result = await llm_client.chat(
            system_prompt=system_prompt,
            user_message=user_message,
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            temperature=0.2,  # Low temperature for consistent extraction
            max_tokens=3000
        )
        
        if result.get("error"):
            logger.error(f"PRD extraction LLM call failed: {result['error']}")
            return _get_empty_extraction()
        
        # Parse the JSON response
        extracted = parse_json_response(result.get("output", ""), "object")
        
        if not extracted:
            logger.warning("Failed to parse PRD extraction response")
            return _get_empty_extraction()
        
        # Ensure all required keys exist
        return {
            "summary": extracted.get("summary", ""),
            "must_do": extracted.get("must_do", []),
            "must_not_do": extracted.get("must_not_do", []),
            "constraints": extracted.get("constraints", []),
            "success_criteria": extracted.get("success_criteria", []),
            "edge_cases": extracted.get("edge_cases", []),
            "user_personas": extracted.get("user_personas", []),
            "business_goals": extracted.get("business_goals", []),
        }
        
    except Exception as e:
        logger.error(f"PRD extraction failed: {e}")
        return _get_empty_extraction()


def _get_empty_extraction() -> Dict[str, Any]:
    """Return empty extraction result"""
    return {
        "summary": "",
        "must_do": [],
        "must_not_do": [],
        "constraints": [],
        "success_criteria": [],
        "edge_cases": [],
        "user_personas": [],
        "business_goals": []
    }


def merge_extracted_with_manual(
    extracted: Dict[str, Any],
    manual: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge PRD-extracted requirements with manually specified ones.
    Manual specifications take precedence and are additive.
    
    Args:
        extracted: Requirements extracted from PRD
        manual: Manually specified requirements (from UI fields)
        
    Returns:
        Merged requirements dict
    """
    merged = extracted.copy()
    
    # Merge lists (manual items are added to extracted)
    for list_key in ["must_do", "must_not_do", "constraints", "success_criteria", "edge_cases"]:
        extracted_items = set(extracted.get(list_key, []))
        manual_items = manual.get(list_key, [])
        
        if manual_items:
            # Add manual items that aren't duplicates
            for item in manual_items:
                if item not in extracted_items:
                    merged[list_key].append(item)
    
    # Override single-value fields if manual value provided
    for single_key in ["tone", "output_format"]:
        if manual.get(single_key):
            merged[single_key] = manual[single_key]
    
    return merged
