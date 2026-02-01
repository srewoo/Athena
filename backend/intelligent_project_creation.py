"""
Intelligent Project Creation with Enhanced Extraction

Integrates intelligent requirements extraction into project creation
to auto-fill Use Case and Key Requirements fields.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from llm_client_v2 import EnhancedLLMClient
from intelligent_requirements_extractor import extract_requirements_intelligently

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["intelligent-project-creation"])


class ExtractFromPromptRequest(BaseModel):
    """Request to extract use case and requirements from system prompt"""
    system_prompt: str
    provider: Optional[str] = None
    api_key: Optional[str] = None


class ExtractFromPromptResponse(BaseModel):
    """Response with extracted use case and requirements"""
    success: bool
    use_case: str
    key_requirements: list[str]
    extraction_metadata: dict


@router.post("/extract-from-prompt", response_model=ExtractFromPromptResponse)
async def extract_from_prompt(request: ExtractFromPromptRequest):
    """
    Extract use case and key requirements from system prompt using intelligent LLM analysis.

    This endpoint is called by the frontend during project creation to auto-fill
    the Use Case and Key Requirements fields.

    **Enhanced with intelligent extraction:**
    - Detects domain (medical, financial, sales, education, etc.)
    - Identifies risk level
    - Extracts must_do/must_not_do items
    - Determines critical dimensions
    - Provides confidence score
    """

    if not request.system_prompt or len(request.system_prompt.strip()) < 20:
        raise HTTPException(
            status_code=400,
            detail="System prompt too short (minimum 20 characters)"
        )

    # Get API settings
    from shared_settings import get_settings
    settings = get_settings()

    provider = request.provider or settings.get("provider", "anthropic")
    api_key = request.api_key or settings.get("api_key", "")

    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="API key not configured. Please configure in settings."
        )

    try:
        # Create LLM client
        llm_client = EnhancedLLMClient(
            api_key=api_key,
            # provider is set via environment or settings
        )

        logger.info("Using intelligent extraction for project creation...")

        # Extract requirements intelligently
        extracted = await extract_requirements_intelligently(
            system_prompt=request.system_prompt,
            requirements="",  # User hasn't provided requirements yet
            use_case="",  # This is what we're trying to extract!
            llm_client=llm_client
        )

        logger.info(
            f"Extraction complete: domain={extracted.domain}, "
            f"risk={extracted.risk_level}, confidence={extracted.confidence_score:.2f}"
        )

        # Format use case
        use_case = extracted.use_case or extracted.primary_function

        # Format key requirements (combine must_do and important items)
        key_requirements = []

        # Add must_do items
        key_requirements.extend(extracted.must_do[:5])  # Top 5 must-do items

        # Add must_not_do as negative requirements
        for item in extracted.must_not_do[:3]:  # Top 3 must-not-do
            key_requirements.append(f"Must NOT: {item}")

        # If we have fewer than 5 requirements, add key_requirements
        if len(key_requirements) < 5 and extracted.key_requirements:
            remaining = 5 - len(key_requirements)
            key_requirements.extend(extracted.key_requirements[:remaining])

        # Return response with metadata
        return ExtractFromPromptResponse(
            success=True,
            use_case=use_case,
            key_requirements=key_requirements,
            extraction_metadata={
                "method": "intelligent_llm_extraction",
                "domain": extracted.domain,
                "risk_level": extracted.risk_level,
                "confidence_score": extracted.confidence_score,
                "tone": extracted.tone,
                "critical_dimensions": extracted.critical_dimensions,
                "quality_priorities": extracted.quality_priorities,
                "extraction_notes": extracted.extraction_notes
            }
        )

    except Exception as e:
        logger.error(f"Error in intelligent extraction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract requirements: {str(e)}"
        )
