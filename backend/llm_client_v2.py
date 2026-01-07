"""
Enhanced LLM Client for Athena
Features:
- Retry logic with exponential backoff
- Caching for repeated requests
- Streaming support
- Proper error handling
- Multi-provider support with unified interface
- Request/response logging
"""
import time
import asyncio
import hashlib
import json
import logging
from typing import Optional, Dict, Any, AsyncGenerator, Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from cachetools import TTLCache
import re

logger = logging.getLogger(__name__)

# Cache for LLM responses (1 hour TTL, max 1000 items)
_response_cache = TTLCache(maxsize=1000, ttl=3600)


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "claude"
    GOOGLE = "gemini"


class LLMError(Exception):
    """Base exception for LLM errors"""
    def __init__(self, message: str, provider: str, retryable: bool = False):
        self.message = message
        self.provider = provider
        self.retryable = retryable
        super().__init__(message)


class RateLimitError(LLMError):
    """Rate limit exceeded"""
    def __init__(self, message: str, provider: str, retry_after: int = 60):
        super().__init__(message, provider, retryable=True)
        self.retry_after = retry_after


class AuthenticationError(LLMError):
    """Invalid API key"""
    def __init__(self, message: str, provider: str):
        super().__init__(message, provider, retryable=False)


class ModelNotFoundError(LLMError):
    """Model not available"""
    def __init__(self, message: str, provider: str, model: str):
        super().__init__(message, provider, retryable=False)
        self.model = model


@dataclass
class LLMResponse:
    """Standardized LLM response"""
    output: str
    error: Optional[str] = None
    latency_ms: int = 0
    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    provider: str = ""
    cached: bool = False
    request_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output": self.output,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model": self.model,
            "provider": self.provider,
            "cached": self.cached,
            "request_id": self.request_id
        }


# Model configurations
# Models that require max_completion_tokens instead of max_tokens
# and may not support system messages or temperature
REASONING_MODELS = {
    "openai": ["o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini", "gpt-5", "gpt-4o", "gpt-4o-mini"],
    "claude": ["claude-sonnet-4-5", "claude-opus-4"],
    "gemini": ["gemini-3", "gemini-2.0-flash-thinking"]
}

DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "claude": "claude-3-5-sonnet-20241022",
    "gemini": "gemini-2.0-flash-exp"
}

# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0
MAX_RETRY_DELAY = 60.0
RETRY_MULTIPLIER = 2.0

# Retryable error patterns
RETRYABLE_ERRORS = [
    "rate limit",
    "rate_limit",
    "too many requests",
    "overloaded",
    "capacity",
    "timeout",
    "connection",
    "temporary",
    "503",
    "529",
]


def is_retryable_error(error: str) -> bool:
    """Check if an error is retryable"""
    error_lower = error.lower()
    return any(pattern in error_lower for pattern in RETRYABLE_ERRORS)


def calculate_retry_delay(attempt: int, base_delay: float = INITIAL_RETRY_DELAY) -> float:
    """Calculate delay with exponential backoff and jitter"""
    import random
    delay = min(base_delay * (RETRY_MULTIPLIER ** attempt), MAX_RETRY_DELAY)
    # Add jitter (±25%)
    jitter = delay * 0.25 * (random.random() * 2 - 1)
    return delay + jitter


def get_cache_key(
    system_prompt: str,
    user_message: str,
    provider: str,
    model_name: str,
    temperature: float
) -> str:
    """Generate a cache key for a request"""
    content = f"{provider}:{model_name}:{temperature}:{system_prompt}:{user_message}"
    return hashlib.sha256(content.encode()).hexdigest()


def is_reasoning_model(model_name: str, provider: str) -> bool:
    """Check if a model is a reasoning/thinking model"""
    if not model_name:
        return False
    prefixes = REASONING_MODELS.get(provider, [])
    return any(model_name.startswith(prefix) for prefix in prefixes)


class EnhancedLLMClient:
    """Enhanced LLM client with retry, caching, and error handling"""

    def __init__(self):
        self._clients = {}

    async def _get_openai_client(self, api_key: str):
        """Get or create OpenAI client"""
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=api_key)

    async def _get_anthropic_client(self, api_key: str):
        """Get or create Anthropic client"""
        from anthropic import AsyncAnthropic
        return AsyncAnthropic(api_key=api_key)

    async def _call_openai(
        self,
        client,
        model: str,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """Call OpenAI API"""
        is_reasoning = is_reasoning_model(model, "openai")

        try:
            if is_reasoning:
                # Reasoning models: no system message, no temperature
                combined_message = f"{system_prompt}\n\n{user_message}"
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": combined_message}],
                    max_completion_tokens=max_tokens
                )
            else:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            usage = response.usage
            return LLMResponse(
                output=response.choices[0].message.content,
                tokens_used=usage.total_tokens if usage else 0,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                model=model,
                provider="openai"
            )

        except Exception as e:
            error_msg = str(e)

            if "401" in error_msg or "invalid_api_key" in error_msg.lower():
                raise AuthenticationError(
                    "Invalid OpenAI API key",
                    "openai"
                )

            if "rate" in error_msg.lower() or "429" in error_msg:
                raise RateLimitError(
                    f"OpenAI rate limit exceeded: {error_msg}",
                    "openai"
                )

            if "model" in error_msg.lower() and "not found" in error_msg.lower():
                raise ModelNotFoundError(
                    f"Model not found: {model}",
                    "openai",
                    model
                )

            raise LLMError(
                error_msg,
                "openai",
                retryable=is_retryable_error(error_msg)
            )

    async def _call_anthropic(
        self,
        client,
        model: str,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """Call Anthropic API"""
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                temperature=max(0.01, temperature)  # Anthropic min is 0
            )

            usage = response.usage
            return LLMResponse(
                output=response.content[0].text,
                tokens_used=(usage.input_tokens + usage.output_tokens) if usage else 0,
                input_tokens=usage.input_tokens if usage else 0,
                output_tokens=usage.output_tokens if usage else 0,
                model=model,
                provider="claude"
            )

        except Exception as e:
            error_msg = str(e)

            if "401" in error_msg or "authentication" in error_msg.lower():
                raise AuthenticationError(
                    "Invalid Anthropic API key",
                    "claude"
                )

            if "rate" in error_msg.lower() or "429" in error_msg:
                raise RateLimitError(
                    f"Anthropic rate limit exceeded: {error_msg}",
                    "claude"
                )

            raise LLMError(
                error_msg,
                "claude",
                retryable=is_retryable_error(error_msg)
            )

    async def _call_gemini(
        self,
        api_key: str,
        model: str,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """Call Google Gemini API"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)

            genai_model = genai.GenerativeModel(model)
            full_prompt = f"{system_prompt}\n\n{user_message}"

            response = await genai_model.generate_content_async(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )

            return LLMResponse(
                output=response.text,
                model=model,
                provider="gemini"
            )

        except Exception as e:
            error_msg = str(e)

            if "api key" in error_msg.lower() or "401" in error_msg:
                raise AuthenticationError(
                    "Invalid Google API key",
                    "gemini"
                )

            raise LLMError(
                error_msg,
                "gemini",
                retryable=is_retryable_error(error_msg)
            )

    async def chat(
        self,
        system_prompt: str,
        user_message: str,
        provider: str,
        api_key: str,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8000,
        use_cache: bool = True,
        request_id: str = ""
    ) -> Dict[str, Any]:
        """
        Execute chat completion with retry logic and caching

        Returns dict for backwards compatibility with existing code
        """
        start_time = time.time()
        model = model_name or DEFAULT_MODELS.get(provider, "gpt-4o")

        # Check cache
        if use_cache and temperature == 0:
            cache_key = get_cache_key(
                system_prompt, user_message, provider, model, temperature
            )
            if cache_key in _response_cache:
                cached = _response_cache[cache_key]
                logger.info(f"Cache hit for request {request_id}")
                return {
                    **cached,
                    "cached": True,
                    "latency_ms": int((time.time() - start_time) * 1000)
                }

        # Retry loop
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                if provider == "openai":
                    client = await self._get_openai_client(api_key)
                    response = await self._call_openai(
                        client, model, system_prompt, user_message,
                        temperature, max_tokens
                    )

                elif provider == "claude":
                    client = await self._get_anthropic_client(api_key)
                    response = await self._call_anthropic(
                        client, model, system_prompt, user_message,
                        temperature, max_tokens
                    )

                elif provider == "gemini":
                    response = await self._call_gemini(
                        api_key, model, system_prompt, user_message,
                        temperature, max_tokens
                    )

                else:
                    return {
                        "output": "",
                        "error": f"Unsupported provider: {provider}",
                        "latency_ms": int((time.time() - start_time) * 1000),
                        "tokens_used": 0
                    }

                # Success
                response.latency_ms = int((time.time() - start_time) * 1000)
                response.request_id = request_id

                result = response.to_dict()

                # Cache successful responses with temperature 0
                if use_cache and temperature == 0:
                    _response_cache[cache_key] = result

                return result

            except AuthenticationError as e:
                # Don't retry auth errors
                return {
                    "output": "",
                    "error": e.message,
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "tokens_used": 0
                }

            except ModelNotFoundError as e:
                return {
                    "output": "",
                    "error": e.message,
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "tokens_used": 0
                }

            except RateLimitError as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    delay = max(e.retry_after, calculate_retry_delay(attempt))
                    logger.warning(
                        f"Rate limited, retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    await asyncio.sleep(delay)

            except LLMError as e:
                last_error = e
                if e.retryable and attempt < MAX_RETRIES - 1:
                    delay = calculate_retry_delay(attempt)
                    logger.warning(
                        f"Retryable error: {e.message}, retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    await asyncio.sleep(delay)
                elif not e.retryable:
                    break

            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1 and is_retryable_error(str(e)):
                    delay = calculate_retry_delay(attempt)
                    logger.warning(
                        f"Error: {e}, retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    await asyncio.sleep(delay)
                else:
                    break

        # All retries exhausted
        error_msg = str(last_error) if last_error else "Unknown error"
        logger.error(f"All retries exhausted: {error_msg}")

        return {
            "output": "",
            "error": error_msg,
            "latency_ms": int((time.time() - start_time) * 1000),
            "tokens_used": 0
        }

    async def chat_stream(
        self,
        system_prompt: str,
        user_message: str,
        provider: str,
        api_key: str,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8000,
        on_chunk: Callable[[str], None] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion

        Yields chunks of text as they arrive
        """
        model = model_name or DEFAULT_MODELS.get(provider, "gpt-4o")

        try:
            if provider == "openai":
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=api_key)

                is_reasoning = is_reasoning_model(model, "openai")

                if is_reasoning:
                    combined_message = f"{system_prompt}\n\n{user_message}"
                    stream = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": combined_message}],
                        max_completion_tokens=max_tokens,
                        stream=True
                    )
                else:
                    stream = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True
                    )

                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        if on_chunk:
                            on_chunk(text)
                        yield text

            elif provider == "claude":
                from anthropic import AsyncAnthropic
                client = AsyncAnthropic(api_key=api_key)

                async with client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                    temperature=max(0.01, temperature)
                ) as stream:
                    async for text in stream.text_stream:
                        if on_chunk:
                            on_chunk(text)
                        yield text

            else:
                # Gemini doesn't support streaming the same way
                result = await self.chat(
                    system_prompt, user_message, provider, api_key,
                    model_name, temperature, max_tokens, use_cache=False
                )
                if result.get("output"):
                    yield result["output"]

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"[ERROR: {str(e)}]"


# ============================================================================
# JSON Parsing Utilities
# ============================================================================

def parse_json_response(text: str, expected_type: str = "object") -> Any:
    """
    Safely parse JSON from LLM response with fallbacks

    Args:
        text: Raw LLM response text
        expected_type: "object" or "array"

    Returns:
        Parsed JSON or None
    """
    if not text:
        return None

    # Remove markdown code blocks
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Try direct parsing first
    try:
        result = json.loads(text)
        if expected_type == "object" and isinstance(result, dict):
            return result
        if expected_type == "array" and isinstance(result, list):
            return result
        return result
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from text
    if expected_type == "object":
        # Find outermost braces
        start = text.find("{")
        if start == -1:
            return None

        # Find matching closing brace
        depth = 0
        end = start
        for i, char in enumerate(text[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if depth != 0:
            return None

        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    elif expected_type == "array":
        # Find outermost brackets
        start = text.find("[")
        if start == -1:
            return None

        depth = 0
        end = start
        for i, char in enumerate(text[start:], start):
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if depth != 0:
            return None

        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return None


def validate_json_schema(data: Any, schema: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Simple JSON schema validation

    Args:
        data: Data to validate
        schema: Schema with required fields and types

    Returns:
        (is_valid, list of errors)
    """
    errors = []

    if not isinstance(data, dict):
        return False, ["Expected object"]

    required = schema.get("required", [])
    properties = schema.get("properties", {})

    for field in required:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    for field, prop_schema in properties.items():
        if field not in data:
            continue

        value = data[field]
        expected_type = prop_schema.get("type")

        if expected_type == "string" and not isinstance(value, str):
            errors.append(f"Field '{field}' should be string")
        elif expected_type == "number" and not isinstance(value, (int, float)):
            errors.append(f"Field '{field}' should be number")
        elif expected_type == "array" and not isinstance(value, list):
            errors.append(f"Field '{field}' should be array")
        elif expected_type == "object" and not isinstance(value, dict):
            errors.append(f"Field '{field}' should be object")
        elif expected_type == "boolean" and not isinstance(value, bool):
            errors.append(f"Field '{field}' should be boolean")

    return len(errors) == 0, errors


# Factory function for backwards compatibility
def get_llm_client() -> EnhancedLLMClient:
    """Get LLM client instance"""
    return EnhancedLLMClient()


# Clear cache function
def clear_cache():
    """Clear the response cache"""
    _response_cache.clear()
