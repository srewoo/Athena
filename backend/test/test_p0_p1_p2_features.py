"""
P0/P1/P2 Features API Tests
Tests for all new evaluation suite improvement features
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add parent directory to path to import server
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server import app

client = TestClient(app)


class TestP0OverlapDetection:
    """Test P0.1: Overlap Detection"""

    @patch('server.db')
    @patch('server.SentenceTransformer')
    async def test_overlap_detection_with_no_overlaps(self, mock_model, mock_db):
        """Test overlap detection when no overlaps exist"""
        # Mock eval prompts
        mock_evals = [
            {
                "eval_prompt_id": "eval-1",
                "dimension": "Accuracy",
                "eval_prompt": "Test accuracy..."
            },
            {
                "eval_prompt_id": "eval-2",
                "dimension": "Tone",
                "eval_prompt": "Test tone..."
            }
        ]

        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=mock_evals)
        mock_db.evaluation_prompts.find.return_value = mock_cursor

        # Mock embeddings (very different)
        mock_model_instance = MagicMock()
        mock_model_instance.encode.side_effect = [
            [0.1, 0.2, 0.3],  # Accuracy embedding
            [0.8, 0.9, 0.7]   # Tone embedding
        ]
        mock_model.return_value = mock_model_instance

        response = client.post(
            "/api/analyze-overlaps",
            params={"project_id": "project-123", "similarity_threshold": 0.7}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_evals"] == 2
        assert data["overlap_count"] == 0
        assert len(data["overlaps"]) == 0

    @patch('server.db')
    @patch('server.SentenceTransformer')
    async def test_overlap_detection_with_overlaps(self, mock_model, mock_db):
        """Test overlap detection when overlaps exist"""
        # Mock eval prompts with similar dimensions
        mock_evals = [
            {
                "eval_prompt_id": "eval-1",
                "dimension": "Accuracy",
                "eval_prompt": "Check accuracy..."
            },
            {
                "eval_prompt_id": "eval-2",
                "dimension": "Correctness",
                "eval_prompt": "Check correctness..."
            }
        ]

        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=mock_evals)
        mock_db.evaluation_prompts.find.return_value = mock_cursor

        # Mock embeddings (very similar)
        mock_model_instance = MagicMock()
        mock_model_instance.encode.side_effect = [
            [0.9, 0.8, 0.85],  # Accuracy embedding
            [0.89, 0.81, 0.84] # Correctness embedding (very similar)
        ]
        mock_model.return_value = mock_model_instance

        response = client.post(
            "/api/analyze-overlaps",
            params={"project_id": "project-123", "similarity_threshold": 0.7}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_evals"] == 2
        assert data["overlap_count"] >= 1
        assert len(data["overlaps"]) >= 1

        # Check overlap structure
        overlap = data["overlaps"][0]
        assert "dimension_1" in overlap
        assert "dimension_2" in overlap
        assert "similarity" in overlap
        assert "recommendation" in overlap

    @patch('server.db')
    async def test_overlap_detection_with_insufficient_evals(self, mock_db):
        """Test overlap detection with less than 2 evals"""
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=[{"eval_prompt_id": "eval-1"}])
        mock_db.evaluation_prompts.find.return_value = mock_cursor

        response = client.post(
            "/api/analyze-overlaps",
            params={"project_id": "project-123"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["overlap_count"] == 0
        assert "Not enough evals" in data["summary"]


class TestP0CoverageAnalysis:
    """Test P0.2: Coverage Analysis"""

    @patch('server.db')
    @patch('server.SentenceTransformer')
    async def test_coverage_analysis_full_coverage(self, mock_model, mock_db):
        """Test coverage analysis with 100% coverage"""
        # Mock eval prompts
        mock_evals = [
            {"dimension": "Check accuracy", "eval_prompt": "Accuracy eval..."},
            {"dimension": "Validate tone", "eval_prompt": "Tone eval..."}
        ]

        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=mock_evals)
        mock_db.evaluation_prompts.find.return_value = mock_cursor

        # Mock embeddings (similar to requirements)
        mock_model_instance = MagicMock()
        mock_model_instance.encode.side_effect = [
            [0.5, 0.5],  # Requirement 1
            [0.6, 0.6],  # Requirement 2
            [0.51, 0.49], # Eval 1 (matches req 1)
            [0.59, 0.61]  # Eval 2 (matches req 2)
        ]
        mock_model.return_value = mock_model_instance

        response = client.post(
            "/api/analyze-coverage",
            json={
                "project_id": "project-123",
                "requirements": [
                    "Check accuracy of outputs",
                    "Validate tone appropriateness"
                ],
                "similarity_threshold": 0.6
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_requirements"] == 2
        assert data["coverage_percentage"] == 100.0
        assert len(data["uncovered_requirements"]) == 0

    @patch('server.db')
    @patch('server.SentenceTransformer')
    async def test_coverage_analysis_partial_coverage(self, mock_model, mock_db):
        """Test coverage analysis with gaps"""
        mock_evals = [
            {"dimension": "Check accuracy", "eval_prompt": "Accuracy eval..."}
        ]

        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=mock_evals)
        mock_db.evaluation_prompts.find.return_value = mock_cursor

        mock_model_instance = MagicMock()
        mock_model_instance.encode.side_effect = [
            [0.5, 0.5],  # Requirement 1
            [0.9, 0.9],  # Requirement 2 (very different)
            [0.51, 0.49] # Eval 1 (only matches req 1)
        ]
        mock_model.return_value = mock_model_instance

        response = client.post(
            "/api/analyze-coverage",
            json={
                "project_id": "project-123",
                "requirements": [
                    "Check accuracy",
                    "Verify formatting"
                ],
                "similarity_threshold": 0.6
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_requirements"] == 2
        assert data["coverage_percentage"] < 100.0
        assert len(data["uncovered_requirements"]) > 0
        assert len(data["suggestions"]) > 0


class TestP0SuiteMetaEvaluation:
    """Test P0.3: Suite-Level Meta-Evaluation"""

    @patch('server.db')
    async def test_suite_meta_evaluation_high_quality(self, mock_db):
        """Test suite meta-evaluation with high quality evals"""
        mock_evals = [
            {
                "dimension": "Accuracy",
                "quality_score": 9.0,
                "eval_prompt": "Eval 1 with consistent rubric..."
            },
            {
                "dimension": "Tone",
                "quality_score": 8.5,
                "eval_prompt": "Eval 2 with consistent rubric..."
            }
        ]

        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=mock_evals)
        mock_db.evaluation_prompts.find.return_value = mock_cursor

        response = client.post(
            "/api/meta-evaluate-suite",
            params={"project_id": "project-123"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "suite_score" in data
        assert "individual_avg" in data
        assert "consistency_score" in data
        assert "coherence_score" in data
        assert "completeness_score" in data
        assert isinstance(data["issues"], list)
        assert isinstance(data["recommendations"], list)

    @patch('server.db')
    async def test_suite_meta_evaluation_insufficient_evals(self, mock_db):
        """Test suite meta-evaluation with less than 2 evals"""
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_db.evaluation_prompts.find.return_value = mock_cursor

        response = client.post(
            "/api/meta-evaluate-suite",
            params={"project_id": "project-123"}
        )

        # Should still return a response, but with low completeness
        assert response.status_code == 200


class TestP1HumanFeedback:
    """Test P1.1: Human Feedback Integration"""

    @patch('server.db')
    async def test_store_eval_feedback(self, mock_db):
        """Test storing user feedback"""
        mock_db.eval_feedback.insert_one = AsyncMock()
        mock_db.evaluation_prompts.find_one = AsyncMock(return_value={
            "eval_prompt_id": "eval-123"
        })
        mock_db.evaluation_prompts.update_one = AsyncMock()

        response = client.post(
            "/api/eval-feedback",
            json={
                "eval_prompt_id": "eval-123",
                "rating": 5,
                "comment": "Excellent eval prompt!",
                "user_id": "user-456"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "average_rating" in data

    @patch('server.db')
    async def test_retrieve_eval_feedback(self, mock_db):
        """Test retrieving aggregated feedback"""
        mock_feedback = [
            {"eval_prompt_id": "eval-123", "rating": 5, "comment": "Great!"},
            {"eval_prompt_id": "eval-123", "rating": 4, "comment": "Good"}
        ]

        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=mock_feedback)
        mock_db.eval_feedback.find.return_value = mock_cursor

        response = client.get("/api/eval-feedback/eval-123")

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 2
        assert data["average_rating"] == 4.5
        assert len(data["feedback_items"]) == 2

    @patch('server.db')
    async def test_feedback_invalid_rating(self, mock_db):
        """Test feedback with invalid rating"""
        response = client.post(
            "/api/eval-feedback",
            json={
                "eval_prompt_id": "eval-123",
                "rating": 0,  # Invalid
                "comment": "Test"
            }
        )

        # Should validate rating is 1-5
        assert response.status_code in [400, 422]


class TestP1ExecutionFeedback:
    """Test P1.2: Execution Feedback Loop"""

    @patch('server.db')
    async def test_eval_performance_too_easy(self, mock_db):
        """Test detecting too-easy evals"""
        # Mock test results with 95% pass rate
        mock_results = [
            {"passed": True, "score": 5} for _ in range(19)
        ] + [{"passed": False, "score": 2}]

        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=mock_results)
        mock_db.test_results.find.return_value = mock_cursor

        response = client.get("/api/eval-performance/eval-123")

        assert response.status_code == 200
        data = response.json()
        assert data["total_executions"] == 20
        assert data["pass_rate"] == 0.95
        assert data["status"] == "too_easy"
        assert "90%" in data["recommendation"]

    @patch('server.db')
    async def test_eval_performance_too_hard(self, mock_db):
        """Test detecting too-hard/broken evals"""
        # Mock test results with 5% pass rate
        mock_results = [
            {"passed": False, "score": 1} for _ in range(19)
        ] + [{"passed": True, "score": 4}]

        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=mock_results)
        mock_db.test_results.find.return_value = mock_cursor

        response = client.get("/api/eval-performance/eval-123")

        assert response.status_code == 200
        data = response.json()
        assert data["pass_rate"] == 0.05
        assert data["status"] in ["too_hard", "broken"]

    @patch('server.db')
    async def test_eval_performance_working_well(self, mock_db):
        """Test eval with good performance"""
        # Mock test results with 60% pass rate
        mock_results = [
            {"passed": True, "score": 4} for _ in range(12)
        ] + [{"passed": False, "score": 2} for _ in range(8)]

        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=mock_results)
        mock_db.test_results.find.return_value = mock_cursor

        response = client.get("/api/eval-performance/eval-123")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "working_well"


class TestP1GoldenDataset:
    """Test P1.3: Golden Dataset Validation"""

    @patch('server.db')
    async def test_store_golden_dataset(self, mock_db):
        """Test storing golden dataset"""
        mock_db.golden_datasets.insert_one = AsyncMock()

        response = client.post(
            "/api/golden-dataset",
            json={
                "project_id": "project-123",
                "examples": [
                    {
                        "input_data": {"query": "What is AI?"},
                        "expected_output": "AI is artificial intelligence...",
                        "is_good_example": True,
                        "notes": "Comprehensive answer"
                    },
                    {
                        "input_data": {"query": "What is AI?"},
                        "expected_output": "idk",
                        "is_good_example": False,
                        "notes": "Poor quality"
                    }
                ]
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "dataset_id" in data
        assert data["example_count"] == 2

    @patch('server.LlmClient')
    @patch('server.db')
    async def test_validate_eval_against_golden(self, mock_db, mock_llm):
        """Test validating eval against golden dataset"""
        # Mock golden dataset
        mock_db.golden_datasets.find_one = AsyncMock(return_value={
            "dataset_id": "golden-123",
            "examples": [
                {
                    "input_data": {"query": "Test"},
                    "expected_output": "Good answer",
                    "is_good_example": True
                },
                {
                    "input_data": {"query": "Test"},
                    "expected_output": "Bad answer",
                    "is_good_example": False
                }
            ]
        })

        # Mock eval prompt
        mock_db.evaluation_prompts.find_one = AsyncMock(return_value={
            "eval_prompt_id": "eval-123",
            "eval_prompt": "Rate the answer quality..."
        })

        # Mock LLM responses (correctly identifying good/bad)
        mock_llm_instance = AsyncMock()
        mock_llm_instance.send_message = AsyncMock(
            side_effect=[
                '{"score": 5, "passed": true}',  # Good example
                '{"score": 2, "passed": false}'  # Bad example
            ]
        )
        mock_llm.return_value = mock_llm_instance

        response = client.post(
            "/api/validate-eval",
            params={
                "eval_prompt_id": "eval-123",
                "dataset_id": "golden-123",
                "api_key": "test-key"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "accuracy" in data
        assert "precision" in data
        assert "recall" in data
        assert "f1_score" in data
        assert data["accuracy"] == 1.0  # Correctly identified both


class TestP2SuiteOptimization:
    """Test P2.1: Suite Optimization"""

    @patch('server.db')
    @patch('server.SentenceTransformer')
    async def test_optimize_suite(self, mock_model, mock_db):
        """Test suite optimization removes redundant evals"""
        mock_evals = [
            {
                "eval_prompt_id": "eval-1",
                "dimension": "Accuracy",
                "quality_score": 9.0,
                "eval_prompt": "Check accuracy..."
            },
            {
                "eval_prompt_id": "eval-2",
                "dimension": "Correctness",
                "quality_score": 8.0,
                "eval_prompt": "Check correctness..."
            },
            {
                "eval_prompt_id": "eval-3",
                "dimension": "Tone",
                "quality_score": 8.5,
                "eval_prompt": "Check tone..."
            }
        ]

        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=mock_evals)
        mock_db.evaluation_prompts.find.return_value = mock_cursor

        # Mock embeddings (1 and 2 are similar, 3 is different)
        mock_model_instance = MagicMock()
        mock_model_instance.encode.side_effect = [
            [0.9, 0.8],   # Accuracy
            [0.89, 0.81], # Correctness (similar to Accuracy)
            [0.1, 0.2]    # Tone (different)
        ]
        mock_model.return_value = mock_model_instance

        response = client.post(
            "/api/optimize-suite",
            params={"project_id": "project-123", "quality_threshold": 8.0}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "original_count" in data
        assert "optimized_count" in data
        assert "removed_evals" in data


class TestP2DependencyManagement:
    """Test P2.2: Dependency Management"""

    @patch('server.db')
    async def test_analyze_dependencies(self, mock_db):
        """Test dependency analysis and topological sort"""
        mock_evals = [
            {
                "eval_prompt_id": "eval-1",
                "dimension": "Evidence Grounding",
                "eval_prompt": "Check evidence..."
            },
            {
                "eval_prompt_id": "eval-2",
                "dimension": "Scoring Accuracy",
                "eval_prompt": "Check scores based on evidence..."
            }
        ]

        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=mock_evals)
        mock_db.evaluation_prompts.find.return_value = mock_cursor

        response = client.get("/api/eval-dependencies/project-123")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "total_evals" in data
        assert "execution_order" in data
        assert "dependencies" in data
        assert "has_cycles" in data


class TestP2AdvancedPromptAnalysis:
    """Test P2.3: Advanced Prompt Analysis"""

    @patch('server.LlmClient')
    async def test_advanced_prompt_analysis(self, mock_llm):
        """Test multi-pass prompt analysis"""
        mock_llm_instance = AsyncMock()
        # Mock responses for each extraction pass
        mock_llm_instance.send_message = AsyncMock(
            side_effect=[
                '["Requirement 1", "Requirement 2"]',  # Explicit
                '["Implicit 1", "Implicit 2"]',         # Implicit
                '["Edge case 1"]',                      # Edge cases
                '["Constraint 1"]'                      # Constraints
            ]
        )
        mock_llm.return_value = mock_llm_instance

        response = client.post(
            "/api/advanced-prompt-analysis",
            json={
                "system_prompt": "Complex system prompt...",
                "api_key": "test-key",
                "provider": "openai",
                "model": "gpt-4o"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "explicit_requirements" in data
        assert "implicit_requirements" in data
        assert "edge_cases" in data
        assert "constraints" in data
        assert "complexity_score" in data
        assert "recommended_eval_count" in data


class TestEndpointErrorHandling:
    """Test error handling across all new endpoints"""

    def test_overlap_detection_missing_project(self):
        """Test overlap detection with missing project_id"""
        response = client.post("/api/analyze-overlaps")
        assert response.status_code == 422  # Validation error

    def test_coverage_analysis_invalid_payload(self):
        """Test coverage analysis with invalid payload"""
        response = client.post(
            "/api/analyze-coverage",
            json={"invalid": "data"}
        )
        assert response.status_code == 422

    def test_golden_dataset_empty_examples(self):
        """Test golden dataset with empty examples"""
        response = client.post(
            "/api/golden-dataset",
            json={
                "project_id": "project-123",
                "examples": []
            }
        )
        # Should accept but return 0 examples
        assert response.status_code in [200, 400, 422]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
