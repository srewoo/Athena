"""
Comprehensive tests for run_single_test_case function
"""
import pytest
import sys
import os
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from project_api import run_single_test_case


class TestRunSingleTestCase:
    """Tests for run_single_test_case function"""
    
    @pytest.mark.asyncio
    async def test_run_single_test_case_success(self):
        """Positive: Successful test case execution"""
        test_case = {
            "id": "test-123",
            "input": "What is AI?"
        }
        
        mock_llm_response = {
            "output": "AI is artificial intelligence.",
            "error": None,
            "latency_ms": 500,
            "tokens_used": 100
        }
        
        mock_eval_response = {
            "output": '{"score": 4.5, "reasoning": "Good answer"}',
            "error": None,
            "tokens_used": 50
        }
        
        with patch('project_api.llm_client') as mock_client:
            mock_client.chat = AsyncMock(side_effect=[mock_llm_response, mock_eval_response])
            
            result = await run_single_test_case(
                test_case=test_case,
                system_prompt="You are helpful.",
                eval_prompt="Rate from 1-5: {{INPUT}} {{OUTPUT}}",
                llm_provider="openai",
                model_name="gpt-4o",
                api_key="test-key",
                eval_provider="openai",
                eval_model_name="gpt-4o",
                eval_api_key="test-key",
                pass_threshold=3.5
            )
            
            assert result["test_case_id"] == "test-123"
            assert result["score"] == 4.5
            assert result["passed"] is True
            assert result["error"] is False
    
    @pytest.mark.asyncio
    async def test_run_single_test_case_generation_error(self):
        """Negative: LLM generation fails"""
        test_case = {"id": "test-456", "input": "Test"}
        
        mock_error_response = {
            "output": "",
            "error": "API Error",
            "latency_ms": 0,
            "tokens_used": 0
        }
        
        with patch('project_api.llm_client') as mock_client:
            mock_client.chat = AsyncMock(return_value=mock_error_response)
            
            result = await run_single_test_case(
                test_case=test_case,
                system_prompt="Test",
                eval_prompt="Rate: {{INPUT}} {{OUTPUT}}",
                llm_provider="openai",
                model_name="gpt-4o",
                api_key="test-key",
                eval_provider="openai",
                eval_model_name="gpt-4o",
                eval_api_key="test-key",
                pass_threshold=3.5
            )
            
            assert result["error"] is True
            assert result["generation_error"] is True
            assert result["passed"] is False
            assert result["score"] is None
    
    @pytest.mark.asyncio
    async def test_run_single_test_case_evaluation_error(self):
        """Negative: Evaluation fails"""
        test_case = {"id": "test-789", "input": "Test"}
        
        mock_llm_response = {
            "output": "Response",
            "error": None,
            "latency_ms": 500,
            "tokens_used": 100
        }
        
        mock_eval_error = {
            "output": "",
            "error": "Eval API Error",
            "tokens_used": 0
        }
        
        with patch('project_api.llm_client') as mock_client:
            mock_client.chat = AsyncMock(side_effect=[mock_llm_response, mock_eval_error])
            
            result = await run_single_test_case(
                test_case=test_case,
                system_prompt="Test",
                eval_prompt="Rate: {{INPUT}} {{OUTPUT}}",
                llm_provider="openai",
                model_name="gpt-4o",
                api_key="test-key",
                eval_provider="openai",
                eval_model_name="gpt-4o",
                eval_api_key="test-key",
                pass_threshold=3.5
            )
            
            assert result["evaluation_error"] is True
            assert result["passed"] is False
            assert result["score"] is None
    
    @pytest.mark.asyncio
    async def test_run_single_test_case_parse_error(self):
        """Negative: JSON parsing fails"""
        test_case = {"id": "test-parse", "input": "Test"}
        
        mock_llm_response = {
            "output": "Response",
            "error": None,
            "latency_ms": 500,
            "tokens_used": 100
        }
        
        mock_eval_invalid = {
            "output": "This is not JSON",
            "error": None,
            "tokens_used": 50
        }
        
        with patch('project_api.llm_client') as mock_client:
            mock_client.chat = AsyncMock(side_effect=[mock_llm_response, mock_eval_invalid])
            
            result = await run_single_test_case(
                test_case=test_case,
                system_prompt="Test",
                eval_prompt="Rate: {{INPUT}} {{OUTPUT}}",
                llm_provider="openai",
                model_name="gpt-4o",
                api_key="test-key",
                eval_provider="openai",
                eval_model_name="gpt-4o",
                eval_api_key="test-key",
                pass_threshold=3.5
            )
            
            assert result["evaluation_error"] is True
            assert result["passed"] is False
            assert result["score"] is None
    
    @pytest.mark.asyncio
    async def test_run_single_test_case_dict_input(self):
        """Edge case: Input is a dict"""
        test_case = {
            "id": "test-dict",
            "input": {"question": "What is AI?", "context": "Technology"}
        }
        
        mock_llm_response = {
            "output": "Response",
            "error": None,
            "latency_ms": 500,
            "tokens_used": 100
        }
        
        mock_eval_response = {
            "output": '{"score": 4.0, "reasoning": "Good"}',
            "error": None,
            "tokens_used": 50
        }
        
        with patch('project_api.llm_client') as mock_client:
            mock_client.chat = AsyncMock(side_effect=[mock_llm_response, mock_eval_response])
            
            result = await run_single_test_case(
                test_case=test_case,
                system_prompt="Test",
                eval_prompt="Rate: {{INPUT}} {{OUTPUT}}",
                llm_provider="openai",
                model_name="gpt-4o",
                api_key="test-key",
                eval_provider="openai",
                eval_model_name="gpt-4o",
                eval_api_key="test-key",
                pass_threshold=3.5
            )
            
            # Input should be converted to JSON string
            assert isinstance(result["input"], str)
            assert "question" in result["input"]
    
    @pytest.mark.asyncio
    async def test_run_single_test_case_with_breakdown(self):
        """Positive: Result may include score breakdown"""
        test_case = {"id": "test-breakdown", "input": "Test"}
        
        mock_llm_response = {
            "output": "Response",
            "error": None,
            "latency_ms": 500,
            "tokens_used": 100
        }
        
        mock_eval_response = {
            "output": '{"score": 4.0, "reasoning": "Good", "breakdown": {"task_completion": 4.5, "requirement_adherence": 4.0}}',
            "error": None,
            "tokens_used": 50
        }
        
        with patch('project_api.llm_client') as mock_client:
            mock_client.chat = AsyncMock(side_effect=[mock_llm_response, mock_eval_response])
            
            result = await run_single_test_case(
                test_case=test_case,
                system_prompt="Test",
                eval_prompt="Rate: {{INPUT}} {{OUTPUT}}",
                llm_provider="openai",
                model_name="gpt-4o",
                api_key="test-key",
                eval_provider="openai",
                eval_model_name="gpt-4o",
                eval_api_key="test-key",
                pass_threshold=3.5
            )
            
            # Breakdown may or may not be included depending on parsing
            assert result["score"] == 4.0
            assert "judge_metadata" in result
    
    @pytest.mark.asyncio
    async def test_run_single_test_case_below_threshold(self):
        """Positive: Score below pass threshold"""
        test_case = {"id": "test-low", "input": "Test"}
        
        mock_llm_response = {
            "output": "Response",
            "error": None,
            "latency_ms": 500,
            "tokens_used": 100
        }
        
        mock_eval_response = {
            "output": '{"score": 2.0, "reasoning": "Poor answer"}',
            "error": None,
            "tokens_used": 50
        }
        
        with patch('project_api.llm_client') as mock_client:
            mock_client.chat = AsyncMock(side_effect=[mock_llm_response, mock_eval_response])
            
            result = await run_single_test_case(
                test_case=test_case,
                system_prompt="Test",
                eval_prompt="Rate: {{INPUT}} {{OUTPUT}}",
                llm_provider="openai",
                model_name="gpt-4o",
                api_key="test-key",
                eval_provider="openai",
                eval_model_name="gpt-4o",
                eval_api_key="test-key",
                pass_threshold=3.5
            )
            
            assert result["score"] == 2.0
            assert result["passed"] is False
    
    @pytest.mark.asyncio
    async def test_run_single_test_case_exception_handling(self):
        """Negative: Exception during execution"""
        test_case = {"id": "test-exception", "input": "Test"}
        
        with patch('project_api.llm_client') as mock_client:
            mock_client.chat = AsyncMock(side_effect=Exception("Unexpected error"))
            
            result = await run_single_test_case(
                test_case=test_case,
                system_prompt="Test",
                eval_prompt="Rate: {{INPUT}} {{OUTPUT}}",
                llm_provider="openai",
                model_name="gpt-4o",
                api_key="test-key",
                eval_provider="openai",
                eval_model_name="gpt-4o",
                eval_api_key="test-key",
                pass_threshold=3.5
            )
            
            assert result["error"] is True
            assert result["generation_error"] is True
            assert "Exception" in result["feedback"]
    
    @pytest.mark.asyncio
    async def test_run_single_test_case_legacy_eval_format(self):
        """Edge case: Legacy eval prompt format (no placeholders)"""
        test_case = {"id": "test-legacy", "input": "Test"}
        
        mock_llm_response = {
            "output": "Response",
            "error": None,
            "latency_ms": 500,
            "tokens_used": 100
        }
        
        mock_eval_response = {
            "output": '{"score": 3.5, "reasoning": "Average"}',
            "error": None,
            "tokens_used": 50
        }
        
        with patch('project_api.llm_client') as mock_client:
            mock_client.chat = AsyncMock(side_effect=[mock_llm_response, mock_eval_response])
            
            result = await run_single_test_case(
                test_case=test_case,
                system_prompt="Test",
                eval_prompt="Rate the response from 1-5.",  # No placeholders
                llm_provider="openai",
                model_name="gpt-4o",
                api_key="test-key",
                eval_provider="openai",
                eval_model_name="gpt-4o",
                eval_api_key="test-key",
                pass_threshold=3.5
            )
            
            assert result["score"] == 3.5
            assert result["passed"] is True
    
    @pytest.mark.asyncio
    async def test_run_single_test_case_tracks_metadata(self):
        """Positive: Tracks judge metadata"""
        test_case = {"id": "test-metadata", "input": "Test"}
        
        mock_llm_response = {
            "output": "Response",
            "error": None,
            "latency_ms": 500,
            "tokens_used": 100
        }
        
        mock_eval_response = {
            "output": '{"score": 4.0, "reasoning": "Good"}',
            "error": None,
            "tokens_used": 50
        }
        
        with patch('project_api.llm_client') as mock_client:
            mock_client.chat = AsyncMock(side_effect=[mock_llm_response, mock_eval_response])
            
            result = await run_single_test_case(
                test_case=test_case,
                system_prompt="Test",
                eval_prompt="Rate: {{INPUT}} {{OUTPUT}}",
                llm_provider="openai",
                model_name="gpt-4o",
                api_key="test-key",
                eval_provider="openai",
                eval_model_name="o1-mini",
                eval_api_key="test-key",
                pass_threshold=3.5
            )
            
            assert "judge_metadata" in result
            assert result["judge_metadata"]["eval_model"] == "o1-mini"
            assert result["judge_metadata"]["parsing_status"] == "success"
