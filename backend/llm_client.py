"""
Unified LLM Client for Multiple Providers
Supports OpenAI, Anthropic (Claude), and Google (Gemini)
"""

import logging
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import google.generativeai as genai


class LlmClient:
    """Unified LLM client supporting multiple providers"""

    # Supported models mapping
    SUPPORTED_MODELS = {
        "openai": ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o3", "o3-mini"],
        "anthropic": ["claude-sonnet-4.5", "claude-opus-4.5", "claude-sonnet-4", "claude-opus-4"],
        "google": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-thinking-exp", "gemini-1.5-pro"]
    }

    def __init__(self, provider: str, model: str, api_key: str, system_message: str = "", max_tokens: int = 4096):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.system_message = system_message
        self.max_tokens = max_tokens

    async def send_message(self, user_message: str) -> str:
        """Send message and get response from LLM"""
        if self.provider == "openai":
            return await self._send_openai(user_message)
        elif self.provider == "claude" or self.provider == "anthropic":
            return await self._send_claude(user_message)
        elif self.provider == "google" or self.provider == "gemini":
            return await self._send_google(user_message)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _send_openai(self, user_message: str) -> str:
        """Send message to OpenAI"""
        try:
            client = AsyncOpenAI(api_key=self.api_key)
            messages = []

            if self.system_message:
                messages.append({"role": "system", "content": self.system_message})

            messages.append({"role": "user", "content": user_message})

            response = await client.chat.completions.create(
                model=self.model,
                messages=messages
            )

            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            raise

    async def _send_claude(self, user_message: str) -> str:
        """Send message to Claude/Anthropic"""
        try:
            client = AsyncAnthropic(api_key=self.api_key)

            response = await client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_message if self.system_message else "You are a helpful assistant.",
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )

            return response.content[0].text
        except Exception as e:
            logging.error(f"Claude API error: {e}")
            raise

    async def _send_google(self, user_message: str) -> str:
        """Send message to Google Gemini"""
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=self.system_message if self.system_message else None
            )

            response = await model.generate_content_async(user_message)
            return response.text
        except Exception as e:
            logging.error(f"Google Gemini API error: {e}")
            raise
