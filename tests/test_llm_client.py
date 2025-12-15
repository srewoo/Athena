"""
Unit tests for LLM client (mocked)
"""
import pytest
import sys
import os
from unittest.mock import AsyncMock, patch, MagicMock

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from llm_client import LLMClient, get_llm_client


class TestLLMClient:
    """Tests for LLMClient class"""
    
    def test_get_llm_client_returns_instance(self):
        """Positive: Should return LLMClient instance"""
        client = get_llm_client()
        assert isinstance(client, LLMClient)
    
    @pytest.mark.asyncio
    async def test_chat_openai_success(self):
        """Positive: Should handle OpenAI provider successfully"""
        client = LLMClient()
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 100
        
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client_instance = AsyncMock()
            mock_openai.return_value = mock_client_instance
            mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            
            result = await client.chat(
                system_prompt="You are helpful",
                user_message="Hello",
                provider="openai",
                api_key="test-key",
                model_name="gpt-4o"
            )
            
            assert result["error"] is None
            assert result["output"] == "Test response"
            assert result["tokens_used"] == 100
    
    @pytest.mark.asyncio
    async def test_chat_openai_reasoning_model(self):
        """Positive: Should handle reasoning models (o1, o3)"""
        client = LLMClient()
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Reasoning response"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 150
        
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client_instance = AsyncMock()
            mock_openai.return_value = mock_client_instance
            mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            
            result = await client.chat(
                system_prompt="You are helpful",
                user_message="Hello",
                provider="openai",
                api_key="test-key",
                model_name="o1-mini"
            )
            
            # Verify it was called with combined message (not separate system/user)
            call_args = mock_client_instance.chat.completions.create.call_args
            assert len(call_args[1]["messages"]) == 1  # Single user message
            assert "max_completion_tokens" in call_args[1]
    
    @pytest.mark.asyncio
    async def test_chat_claude_success(self):
        """Positive: Should handle Claude provider successfully"""
        client = LLMClient()
        
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Claude response"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 30
        
        with patch('anthropic.AsyncAnthropic') as mock_anthropic:
            mock_client_instance = AsyncMock()
            mock_anthropic.return_value = mock_client_instance
            mock_client_instance.messages.create = AsyncMock(return_value=mock_response)
            
            result = await client.chat(
                system_prompt="You are helpful",
                user_message="Hello",
                provider="claude",
                api_key="test-key",
                model_name="claude-3-5-sonnet-20241022"
            )
            
            assert result["error"] is None
            assert result["output"] == "Claude response"
            assert result["tokens_used"] == 80
    
    @pytest.mark.asyncio
    async def test_chat_gemini_success(self):
        """Positive: Should handle Gemini provider successfully"""
        client = LLMClient()
        
        mock_response = MagicMock()
        mock_response.text = "Gemini response"
        
        with patch('google.generativeai.configure') as mock_configure, \
             patch('google.generativeai.GenerativeModel') as mock_model_class:
            mock_model_instance = AsyncMock()
            mock_model_class.return_value = mock_model_instance
            mock_model_instance.generate_content_async = AsyncMock(return_value=mock_response)
            
            result = await client.chat(
                system_prompt="You are helpful",
                user_message="Hello",
                provider="gemini",
                api_key="test-key",
                model_name="gemini-2.0-flash-exp"
            )
            
            assert result["error"] is None
            assert result["output"] == "Gemini response"
    
    @pytest.mark.asyncio
    async def test_chat_unsupported_provider(self):
        """Negative: Should return error for unsupported provider"""
        client = LLMClient()
        
        result = await client.chat(
            system_prompt="Test",
            user_message="Test",
            provider="unsupported",
            api_key="test-key"
        )
        
        assert result["error"] is not None
        assert "Unsupported provider" in result["error"]
        assert result["output"] == ""
    
    @pytest.mark.asyncio
    async def test_chat_openai_exception(self):
        """Negative: Should handle OpenAI API exceptions"""
        client = LLMClient()
        
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client_instance = AsyncMock()
            mock_openai.return_value = mock_client_instance
            mock_client_instance.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
            
            result = await client.chat(
                system_prompt="Test",
                user_message="Test",
                provider="openai",
                api_key="test-key"
            )
            
            assert result["error"] == "API Error"
            assert result["output"] == ""
    
    @pytest.mark.asyncio
    async def test_chat_default_model_names(self):
        """Positive: Should use default model names when not specified"""
        client = LLMClient()
        
        # Test OpenAI default
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client_instance = AsyncMock()
            mock_openai.return_value = mock_client_instance
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Response"
            mock_response.usage = None
            mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            
            await client.chat(
                system_prompt="Test",
                user_message="Test",
                provider="openai",
                api_key="test-key",
                model_name=None  # Should use default
            )
            
            # Verify default model was used
            call_args = mock_client_instance.chat.completions.create.call_args
            assert call_args[1]["model"] == "gpt-4o"
    
    @pytest.mark.asyncio
    async def test_chat_tracks_latency(self):
        """Positive: Should track latency in milliseconds"""
        client = LLMClient()
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage = None
        
        with patch('openai.AsyncOpenAI') as mock_openai, \
             patch('time.time', side_effect=[0, 0.5]):  # 500ms latency
            mock_client_instance = AsyncMock()
            mock_openai.return_value = mock_client_instance
            mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            
            result = await client.chat(
                system_prompt="Test",
                user_message="Test",
                provider="openai",
                api_key="test-key"
            )
            
            assert result["latency_ms"] >= 0
            assert isinstance(result["latency_ms"], int)
