"""
Maxim Evaluator Integration Service for Athena.

Handles manual synchronization of evaluation prompts to Maxim LLMOps platform as evaluators.
Creates evaluators on-demand.
"""

import httpx
import logging
from typing import Optional, Dict, Any
import random

logger = logging.getLogger(__name__)


class MaximService:
    """
    Service for manually synchronizing evaluation prompts with Maxim LLMOps platform as evaluators.

    Handles:
    - Creating AI evaluators in Maxim
    - Managing evaluator configuration
    """

    def __init__(
        self,
        maxim_api_key: str,
        maxim_workspace_id: str,
        maxim_repository_id: Optional[str] = None,
        base_url: str = "https://api.getmaxim.ai"
    ):
        """
        Initialize Maxim service.

        Args:
            maxim_api_key: Maxim API key
            maxim_workspace_id: Maxim workspace ID
            maxim_repository_id: Maxim repository ID
            base_url: Maxim API base URL (defaults to api.getmaxim.ai)
        """
        self.api_key = maxim_api_key
        self.workspace_id = maxim_workspace_id
        self.repository_id = maxim_repository_id
        self.base_url = base_url
        self.client = None

        # Cache of evaluator IDs by name
        self._evaluator_cache: Dict[str, str] = {}

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def initialize(self):
        """Initialize HTTP client"""
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Content-Type": "application/json",
                "x-maxim-api-key": self.api_key
            },
            timeout=30.0
        )
        logger.info(f"Maxim service initialized (base_url={self.base_url})")

    async def close(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()

    def _is_configured(self) -> bool:
        """Check if Maxim is properly configured"""
        return bool(self.api_key and self.workspace_id)

    def _sanitize_evaluator_name(self, name: str) -> str:
        """
        Sanitize evaluator name to be Maxim-compatible.
        Removes special characters and ensures uniqueness.
        """
        # Remove special characters, keep alphanumeric, spaces, hyphens
        sanitized = ''.join(c if c.isalnum() or c in ' -_' else '' for c in name)
        # Add random suffix for uniqueness
        suffix = random.randint(10000, 99999)
        return f"{sanitized}-{suffix}"

    async def create_evaluator(
        self,
        evaluator_name: str,
        instructions: str,
        description: str = "",
        grading_style: str = "Scale",
        model: str = "o3-mini",
        model_provider: str = "openai",
        scale_min: int = 1,
        scale_max: int = 5,
        pass_threshold: int = 80
    ) -> Dict[str, Any]:
        """
        Create an AI evaluator in Maxim.

        Args:
            evaluator_name: Name of the evaluator
            instructions: The evaluation prompt/instructions
            description: Optional description
            grading_style: Grading style (Scale, YesNo, etc.)
            model: Model name (e.g., "o3-mini", "gpt-4o")
            model_provider: LLM provider (openai, anthropic, etc.)
            scale_min: Minimum scale value (default: 1)
            scale_max: Maximum scale value (default: 5)
            pass_threshold: Percentage threshold for passing (default: 80%)

        Returns:
            Dict with success status and evaluator details
        """
        if not self._is_configured():
            return {
                "success": False,
                "error": "Maxim not configured (missing API key or workspace ID)"
            }

        try:
            # Sanitize the evaluator name
            safe_name = self._sanitize_evaluator_name(evaluator_name)

            # Build the evaluator payload based on the API spec
            config = {
                "gradingStyle": grading_style,
                "model": model,
                "provider": model_provider,
                "instructions": instructions,
                "passFailCriteria": {
                    "entryLevel": {
                        "name": "score",
                        "operator": ">=",
                        "value": scale_min if grading_style == "Scale" else "Yes"
                    },
                    "runLevel": {
                        "name": "queriesPassed",
                        "operator": ">=",
                        "value": pass_threshold
                    }
                }
            }

            # Add scale config only for Scale grading style
            if grading_style == "Scale":
                config["scale"] = {
                    "min": scale_min,
                    "max": scale_max
                }

            payload = {
                "workspaceId": self.workspace_id,
                "name": safe_name,
                "description": description,
                "type": "AI",
                "config": config,
                "tags": []
            }

            logger.info(f"Creating evaluator '{safe_name}' in Maxim workspace {self.workspace_id}")

            # Create evaluator using v1 API
            # Correct endpoint: /v1/evaluators on api.getmaxim.ai
            response = await self.client.post(
                "/v1/evaluators",
                json=payload
            )
            response.raise_for_status()

            result = response.json()

            # Log the full response for debugging
            logger.info(f"Maxim API response: {result}")

            # Try different possible response structures
            evaluator_id = (
                result.get("id") or
                result.get("data", {}).get("id") or
                result.get("evaluatorId")
            )

            if evaluator_id:
                # Cache the evaluator
                self._evaluator_cache[evaluator_name] = evaluator_id

                logger.info(f"Created evaluator '{safe_name}': {evaluator_id}")

                return {
                    "success": True,
                    "evaluator_id": evaluator_id,
                    "evaluator_name": safe_name,
                    "message": f"Successfully created evaluator '{safe_name}' in Maxim"
                }
            else:
                logger.error(f"No ID found in response. Full response: {result}")
                return {
                    "success": False,
                    "error": f"Failed to create evaluator (no ID returned). Response: {result}"
                }

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            logger.error(f"Error creating evaluator in Maxim: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error creating evaluator in Maxim: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }

    async def get_evaluators(self) -> Dict[str, Any]:
        """
        Get all evaluators in the workspace.

        Returns:
            Dict with success status and evaluators list
        """
        if not self._is_configured():
            return {
                "success": False,
                "error": "Maxim not configured"
            }

        try:
            response = await self.client.get(
                "/v1/evaluators",
                params={"workspaceId": self.workspace_id}
            )
            response.raise_for_status()

            evaluators = response.json()

            return {
                "success": True,
                "evaluators": evaluators
            }

        except Exception as e:
            logger.error(f"Error fetching evaluators: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


async def create_maxim_service(
    maxim_api_key: str,
    maxim_workspace_id: str,
    maxim_repository_id: Optional[str] = None
) -> MaximService:
    """
    Factory function to create and initialize MaximService.

    Args:
        maxim_api_key: Maxim API key
        maxim_workspace_id: Maxim workspace ID
        maxim_repository_id: Maxim repository ID

    Returns:
        Initialized MaximService instance
    """
    service = MaximService(
        maxim_api_key=maxim_api_key,
        maxim_workspace_id=maxim_workspace_id,
        maxim_repository_id=maxim_repository_id
    )
    await service.initialize()
    return service
