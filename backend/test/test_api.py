"""
Backend API Tests for Athena
Tests critical API endpoints and business logic
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add parent directory to path to import server
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server import app, API

client = TestClient(app)


class TestHealthEndpoints:
    """Test basic health and info endpoints"""

    def test_root_endpoint(self):
        """Test root endpoint returns API info"""
        response = client.get("/api/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data

    def test_models_endpoint(self):
        """Test models endpoint returns supported providers"""
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert "openai" in data["providers"]
        assert "anthropic" in data["providers"]
        assert "google" in data["providers"]


class TestSettingsAPI:
    """Test settings management endpoints"""

    @patch('server.db')
    async def test_create_settings(self, mock_db):
        """Test creating new settings"""
        mock_db.settings.find_one = AsyncMock(return_value=None)
        mock_db.settings.insert_one = AsyncMock()

        response = client.post("/api/settings", json={
            "session_id": "test-session",
            "openai_key": "test-openai-key",
            "default_provider": "openai",
            "default_model": "gpt-4o-mini"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session"
        assert data["default_provider"] == "openai"

    @patch('server.db')
    async def test_update_settings(self, mock_db):
        """Test updating existing settings"""
        existing_settings = {
            "id": "setting-123",
            "session_id": "test-session",
            "openai_key": "old-key",
            "default_provider": "openai"
        }
        mock_db.settings.find_one = AsyncMock(return_value=existing_settings)
        mock_db.settings.replace_one = AsyncMock()

        response = client.post("/api/settings", json={
            "session_id": "test-session",
            "openai_key": "new-key",
            "default_provider": "anthropic"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["default_provider"] == "anthropic"


class TestProjectAPI:
    """Test project management endpoints"""

    @patch('server.db')
    async def test_create_project(self, mock_db):
        """Test creating a new project"""
        mock_db.projects.insert_one = AsyncMock()

        response = client.post("/api/projects", json={
            "name": "Test Project",
            "use_case": "Testing use case",
            "requirements": "Test requirements"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Project"
        assert "id" in data
        assert "created_at" in data

    @patch('server.db')
    async def test_get_all_projects(self, mock_db):
        """Test retrieving all projects"""
        mock_projects = [
            {"id": "1", "name": "Project 1", "created_at": "2024-01-01"},
            {"id": "2", "name": "Project 2", "created_at": "2024-01-02"}
        ]
        
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=mock_projects)
        mock_db.projects.find.return_value = mock_cursor

        response = client.get("/api/projects")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    @patch('server.db')
    async def test_delete_project(self, mock_db):
        """Test deleting a project and associated data"""
        mock_db.evaluation_prompts.find.return_value.to_list = AsyncMock(return_value=[])
        mock_db.projects.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))
        mock_db.prompt_versions.delete_many = AsyncMock()
        mock_db.evaluation_prompts.delete_many = AsyncMock()
        mock_db.test_cases.delete_many = AsyncMock()
        mock_db.test_results.delete_many = AsyncMock()

        response = client.delete("/api/projects/project-123")
        assert response.status_code == 200


class TestProviderAPIKeySelection:
    """Test that correct API keys are used for different providers"""

    @patch('server.LlmClient')
    @patch('server.db')
    async def test_openai_provider_uses_openai_key(self, mock_db, mock_llm):
        """Test OpenAI provider uses OpenAI API key"""
        mock_db.projects.insert_one = AsyncMock()
        mock_llm_instance = AsyncMock()
        mock_llm_instance.send_message = AsyncMock(return_value='{"use_case": "test", "requirements": "test"}')
        mock_llm.return_value = mock_llm_instance

        response = client.post("/api/extract-project-info", json={
            "prompt_content": "Test prompt",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-openai-key"
        })

        # Verify LlmClient was called with correct parameters
        mock_llm.assert_called_once()
        call_args = mock_llm.call_args
        assert call_args[1]['provider'] == 'openai'
        assert call_args[1]['api_key'] == 'test-openai-key'

    @patch('server.LlmClient')
    async def test_anthropic_provider_uses_claude_key(self, mock_llm):
        """Test Anthropic provider uses Claude API key"""
        mock_llm_instance = AsyncMock()
        mock_llm_instance.send_message = AsyncMock(return_value='{"use_case": "test", "requirements": "test"}')
        mock_llm.return_value = mock_llm_instance

        response = client.post("/api/extract-project-info", json={
            "prompt_content": "Test prompt",
            "provider": "anthropic",
            "model": "claude-sonnet-4.5",
            "api_key": "test-claude-key"
        })

        # Verify LlmClient was called with Anthropic key
        mock_llm.assert_called_once()
        call_args = mock_llm.call_args
        assert call_args[1]['provider'] == 'anthropic'
        assert call_args[1]['api_key'] == 'test-claude-key'

    @patch('server.LlmClient')
    async def test_google_provider_uses_gemini_key(self, mock_llm):
        """Test Google provider uses Gemini API key"""
        mock_llm_instance = AsyncMock()
        mock_llm_instance.send_message = AsyncMock(return_value='{"use_case": "test", "requirements": "test"}')
        mock_llm.return_value = mock_llm_instance

        response = client.post("/api/extract-project-info", json={
            "prompt_content": "Test prompt",
            "provider": "google",
            "model": "gemini-2.5-pro",
            "api_key": "test-gemini-key"
        })

        # Verify LlmClient was called with Google key
        mock_llm.assert_called_once()
        call_args = mock_llm.call_args
        assert call_args[1]['provider'] == 'google'
        assert call_args[1]['api_key'] == 'test-gemini-key'


class TestAnalysisEndpoint:
    """Test prompt analysis endpoint"""

    @patch('server.LlmClient')
    async def test_analyze_prompt_with_different_providers(self, mock_llm):
        """Test analysis works with all providers"""
        mock_llm_instance = AsyncMock()
        mock_llm_instance.send_message = AsyncMock(
            return_value='{"combined_score": 8.5, "suggestions": [], "issues": [], "strengths": []}'
        )
        mock_llm.return_value = mock_llm_instance

        providers = [
            ("openai", "test-openai-key"),
            ("anthropic", "test-claude-key"),
            ("google", "test-gemini-key")
        ]

        for provider, api_key in providers:
            response = client.post("/api/analyze", json={
                "prompt_content": "Test prompt",
                "provider": provider,
                "model": "test-model",
                "api_key": api_key
            })

            assert response.status_code == 200
            data = response.json()
            assert "combined_score" in data


class TestEvalPromptGeneration:
    """Test evaluation prompt generation with meta-evaluation"""

    @patch('server.LlmClient')
    @patch('server.db')
    async def test_eval_prompt_generation_with_quality_check(self, mock_db, mock_llm):
        """Test eval prompt generation includes quality check"""
        mock_db.evaluation_prompts.insert_one = AsyncMock()
        
        mock_llm_instance = AsyncMock()
        # First call for generation, second for meta-evaluation
        mock_llm_instance.send_message = AsyncMock(
            side_effect=[
                "Generated evaluation prompt content",
                "8.5"  # Quality score
            ]
        )
        mock_llm.return_value = mock_llm_instance

        response = client.post("/api/eval-prompts", json={
            "project_id": "project-123",
            "prompt_version_id": "version-123",
            "system_prompt": "Test system prompt",
            "dimension": "Test Dimension",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-key"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["dimension"] == "Test Dimension"
        assert "quality_score" in data

    @patch('server.db')
    async def test_delete_all_eval_prompts(self, mock_db):
        """Test deleting all eval prompts for a project"""
        mock_db.evaluation_prompts.find.return_value.to_list = AsyncMock(
            return_value=[{"id": "eval-1"}, {"id": "eval-2"}]
        )
        mock_db.evaluation_prompts.delete_many = AsyncMock(
            return_value=MagicMock(deleted_count=2)
        )
        mock_db.meta_evaluations.delete_many = AsyncMock()

        response = client.delete("/api/eval-prompts/project-123")
        assert response.status_code == 200
        data = response.json()
        assert data["deleted_count"] == 2


class TestTestCaseGeneration:
    """Test test case generation"""

    @patch('server.LlmClient')
    @patch('server.db')
    async def test_generate_test_cases(self, mock_db, mock_llm):
        """Test generating test cases with proper distribution"""
        mock_db.test_cases.insert_many = AsyncMock()
        
        mock_llm_instance = AsyncMock()
        mock_llm_instance.send_message = AsyncMock(
            return_value='[{"input_text": "Test", "expected_behavior": "Behavior", "case_type": "positive"}]'
        )
        mock_llm.return_value = mock_llm_instance

        response = client.post("/api/test-cases", json={
            "project_id": "project-123",
            "prompt_version_id": "version-123",
            "system_prompt": "Test prompt",
            "sample_count": 10,
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-key"
        })

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestTestExecution:
    """Test test execution endpoint"""

    @patch('server.LlmClient')
    @patch('server.db')
    async def test_execute_tests_with_correct_provider(self, mock_db, mock_llm):
        """Test that test execution uses correct provider and API key"""
        mock_test_cases = [
            {"id": "test-1", "input_text": "Test input"}
        ]
        
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=mock_test_cases)
        mock_db.test_cases.find.return_value = mock_cursor
        mock_db.test_results.insert_one = AsyncMock()
        
        mock_llm_instance = AsyncMock()
        mock_llm_instance.send_message = AsyncMock(
            side_effect=[
                "Test output",  # Generation
                '{"score": 4, "passed": true, "reasoning": "Good"}'  # Evaluation
            ]
        )
        mock_llm.return_value = mock_llm_instance

        response = client.post("/api/execute-tests", json={
            "project_id": "project-123",
            "prompt_version_id": "version-123",
            "eval_prompt_id": "eval-123",
            "system_prompt": "Test prompt",
            "eval_prompt_content": "Eval prompt",
            "test_case_ids": ["test-1"],
            "provider": "anthropic",
            "model": "claude-sonnet-4.5",
            "api_key": "test-claude-key"
        })

        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        assert data[0]["passed"] == True


class TestMetaEvaluation:
    """Test meta-evaluation endpoint"""

    @patch('server.LlmClient')
    @patch('server.db')
    async def test_meta_evaluation(self, mock_db, mock_llm):
        """Test comprehensive meta-evaluation"""
        mock_db.meta_evaluations.insert_one = AsyncMock()
        
        mock_llm_instance = AsyncMock()
        mock_llm_instance.send_message = AsyncMock(
            return_value="Executive Summary: Good evaluation prompt..."
        )
        mock_llm.return_value = mock_llm_instance

        response = client.post("/api/meta-evaluate", json={
            "system_prompt": "Test system prompt",
            "eval_prompt": "Test eval prompt",
            "eval_prompt_id": "eval-123",
            "dimension": "Test Dimension",
            "provider": "openai",
            "model": "gpt-4o",
            "api_key": "test-key"
        })

        assert response.status_code == 200
        data = response.json()
        assert "analysis" in data
        assert "dimension" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
