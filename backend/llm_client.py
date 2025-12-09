"""
Unified LLM Client - Provides a consistent interface for OpenAI, Claude, and Gemini APIs.
Eliminates code duplication across the codebase.
"""
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import json
import re
import openai
import anthropic
import google.generativeai as genai


class LLMProvider(str, Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider"""
    content: str
    provider: str
    model: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None


# Default models for each provider (latest as of 2025)
DEFAULT_MODELS = {
    LLMProvider.OPENAI: "gpt-5",  # Latest flagship, released 2025
    LLMProvider.CLAUDE: "claude-sonnet-4-5-20250929",  # Best coding model, released Sep 2025
    LLMProvider.GEMINI: "gemini-2.5-flash"  # Stable 2025 release, optimized for speed
}

# All available models per provider (2025 releases)
AVAILABLE_MODELS = {
    LLMProvider.OPENAI: [
        "gpt-5",  # Latest flagship (2025)
        "gpt-4.1",  # Released April 2025, ~1M token context
        "gpt-4o",  # Previous flagship
        "gpt-4o-mini",  # Cost-efficient variant
        # Reasoning models
        "o1", "o1-preview", "o1-mini",
        "o3", "o3-mini"
    ],
    LLMProvider.CLAUDE: [
        "claude-sonnet-4-5-20250929",  # Best coding model (Sep 2025)
        "claude-haiku-4-5-20251001",  # Fast, low-latency (Oct 2025)
        "claude-3-7-sonnet-20250219",  # Previous flagship
        "claude-3-5-sonnet-20241022",  # Older version
    ],
    LLMProvider.GEMINI: [
        "gemini-2.5-pro",  # Hybrid-reasoning, stable 2025
        "gemini-2.5-flash",  # Lighter/faster 2.5 variant
        "gemini-2.0-flash-exp",  # Experimental 2.0
    ]
}

# Thinking/Reasoning models (extended reasoning, adjustable thinking budgets)
THINKING_MODELS = {
    LLMProvider.OPENAI: [
        "o1", "o1-preview", "o1-mini",
        "o3", "o3-mini"
    ],
    LLMProvider.CLAUDE: [
        # Claude 4.5 series has improved reasoning built-in
        "claude-sonnet-4-5-20250929",  # Strong reasoning capabilities
    ],
    LLMProvider.GEMINI: [
        "gemini-2.5-pro",  # Hybrid-reasoning with adjustable thinking budgets
        "gemini-2.5-flash",  # Lighter variant with reasoning
    ]
}

# Default timeout in seconds (longer for thinking models)
DEFAULT_TIMEOUT = 120.0
THINKING_TIMEOUT = 300.0  # 5 minutes for reasoning models


class LLMClient:
    """
    Unified client for interacting with multiple LLM providers.

    Usage:
        client = LLMClient(provider="openai", api_key="sk-...")
        response = await client.generate("Hello, world!")
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        model_name: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT
    ):
        self.provider = LLMProvider(provider.lower())
        self.api_key = api_key
        self.model_name = model_name or DEFAULT_MODELS[self.provider]
        self.timeout = timeout

        # Initialize the appropriate client
        self._openai_client = None
        self._anthropic_client = None
        self._gemini_model = None

        self._init_client()

    def _init_client(self):
        """Initialize the provider-specific client"""
        if self.provider == LLMProvider.OPENAI:
            self._openai_client = openai.AsyncOpenAI(
                api_key=self.api_key,
                timeout=self.timeout
            )
        elif self.provider == LLMProvider.CLAUDE:
            self._anthropic_client = anthropic.AsyncAnthropic(
                api_key=self.api_key,
                timeout=self.timeout
            )
        elif self.provider == LLMProvider.GEMINI:
            genai.configure(api_key=self.api_key)
            self._gemini_model = genai.GenerativeModel(self.model_name)

    def is_thinking_model(self) -> bool:
        """Check if the current model is a thinking/reasoning model"""
        if self.provider in THINKING_MODELS:
            return self.model_name in THINKING_MODELS[self.provider]
        return False

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt/message
            system_prompt: Optional system prompt (for OpenAI/Claude)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            json_mode: If True, request JSON output (OpenAI only)

        Returns:
            LLMResponse with the generated content
        """
        if self.provider == LLMProvider.OPENAI:
            return await self._generate_openai(
                prompt, system_prompt, temperature, max_tokens, json_mode
            )
        elif self.provider == LLMProvider.CLAUDE:
            return await self._generate_claude(
                prompt, system_prompt, temperature, max_tokens
            )
        elif self.provider == LLMProvider.GEMINI:
            return await self._generate_gemini(
                prompt, system_prompt, temperature, max_tokens
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        json_mode: bool
    ) -> LLMResponse:
        """Generate using OpenAI API"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._openai_client.chat.completions.create(**kwargs)

        return LLMResponse(
            content=response.choices[0].message.content,
            provider=self.provider.value,
            model=self.model_name,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else None,
            raw_response=response
        )

    async def _generate_claude(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """Generate using Anthropic Claude API"""
        kwargs = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        # Claude doesn't support temperature=0 exactly, use 0.01 as minimum
        if temperature > 0:
            kwargs["temperature"] = temperature

        response = await self._anthropic_client.messages.create(**kwargs)

        return LLMResponse(
            content=response.content[0].text,
            provider=self.provider.value,
            model=self.model_name,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            } if response.usage else None,
            raw_response=response
        )

    async def _generate_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """Generate using Google Gemini API"""
        # Gemini combines system and user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )

        response = await self._gemini_model.generate_content_async(
            full_prompt,
            generation_config=generation_config
        )

        return LLMResponse(
            content=response.text,
            provider=self.provider.value,
            model=self.model_name,
            usage=None,  # Gemini doesn't provide detailed usage in the same way
            raw_response=response
        )


def parse_json_response(response_text: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM response, handling various edge cases:
    - Markdown code blocks (```json, ```, etc.)
    - Extra whitespace
    - Case variations (JSON vs json)

    Args:
        response_text: Raw response from LLM

    Returns:
        Parsed JSON as dictionary

    Raises:
        json.JSONDecodeError: If JSON parsing fails after all cleanup attempts
    """
    text = response_text.strip()

    # Try parsing directly first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Pattern to match code blocks with optional language specifier
    # Handles: ```json, ```JSON, ```, ```javascript, etc.
    code_block_pattern = r'^```(?:json|JSON|javascript|JS|)?\s*\n?(.*?)\n?```$'
    match = re.match(code_block_pattern, text, re.DOTALL | re.IGNORECASE)

    if match:
        text = match.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Handle case where code block markers are on separate lines
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or similar)
        if lines:
            lines = lines[1:]
        # Remove last line if it's just ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try to find JSON object or array in the text
    # Look for {...} or [...]
    json_patterns = [
        r'(\{[\s\S]*\})',  # Object
        r'(\[[\s\S]*\])'   # Array
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # Final attempt - raise error with helpful message
    raise json.JSONDecodeError(
        f"Could not parse JSON from response. First 200 chars: {text[:200]}",
        text,
        0
    )


def get_llm_client(
    provider: str,
    api_key: str,
    model_name: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT
) -> LLMClient:
    """
    Factory function to create an LLM client.

    Args:
        provider: "openai", "claude", or "gemini"
        api_key: API key for the provider
        model_name: Optional model name override
        timeout: Request timeout in seconds

    Returns:
        Configured LLMClient instance
    """
    return LLMClient(
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        timeout=timeout
    )
