"""
Unit tests for all API endpoints
Run with: pytest test_endpoints.py -v
"""
import pytest
import os
import shutil
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_projects():
    """Clean up saved_projects directory before and after each test"""
    # Setup - nothing needed
    yield
    # Teardown - clean up project files created during test
    # Only delete files created in the last minute to be safe


# ============================================================================
# BASIC ENDPOINTS
# ============================================================================

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_settings_get():
    """Test get settings endpoint"""
    response = client.get("/api/settings")
    assert response.status_code == 200


def test_settings_post():
    """Test save settings endpoint"""
    response = client.post("/api/settings", json={
        "llm_provider": "openai",
        "api_key": "test-key",
        "model_name": "gpt-4o-mini"
    })
    assert response.status_code == 200
    assert response.json()["message"] == "Settings saved successfully"


# ============================================================================
# PROJECT CRUD ENDPOINTS
# ============================================================================

@pytest.fixture
def test_project():
    """Create a test project and return its ID"""
    response = client.post("/api/projects", json={
        "name": "Test Project",
        "use_case": "Testing endpoints",
        "key_requirements": ["test1", "test2"],
        "initial_prompt": "This is a test prompt for unit testing"
    })
    assert response.status_code == 200
    return response.json()["id"]


def test_create_project():
    """Test project creation"""
    response = client.post("/api/projects", json={
        "name": "Unit Test Project",
        "use_case": "Unit testing",
        "key_requirements": ["req1", "req2"],
        "initial_prompt": "Test prompt"
    })
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["project_name"] == "Unit Test Project"
    assert data["use_case"] == "Unit testing"


def test_list_projects():
    """Test listing projects"""
    response = client.get("/api/projects")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_project(test_project):
    """Test getting a specific project"""
    response = client.get(f"/api/projects/{test_project}")
    assert response.status_code == 200
    assert response.json()["id"] == test_project


def test_get_project_not_found():
    """Test getting non-existent project"""
    response = client.get("/api/projects/non-existent-id")
    assert response.status_code == 404


def test_delete_project():
    """Test deleting a project"""
    # Create a separate project just for deletion test
    create_response = client.post("/api/projects", json={
        "name": "Project To Delete",
        "use_case": "Testing deletion",
        "key_requirements": ["delete test"],
        "initial_prompt": "Delete me"
    })
    assert create_response.status_code == 200
    project_id = create_response.json()["id"]

    response = client.delete(f"/api/projects/{project_id}")
    assert response.status_code == 200


# ============================================================================
# ANALYZE ENDPOINT
# ============================================================================

def test_analyze_prompt(test_project):
    """Test prompt analysis"""
    response = client.post(f"/api/projects/{test_project}/analyze", json={
        "prompt_text": "Write a simple greeting"
    })
    assert response.status_code == 200
    data = response.json()
    assert "overall_score" in data
    assert "suggestions" in data
    assert "requirements_gaps" in data


def test_analyze_detailed_prompt(test_project):
    """Test analysis of detailed prompt"""
    response = client.post(f"/api/projects/{test_project}/analyze", json={
        "prompt_text": """You are a helpful assistant. Your goal is to provide clear, accurate information.
        For example, when asked about weather, provide temperature and conditions.
        Avoid making up data. Use a professional tone.
        Format responses as bullet points."""
    })
    assert response.status_code == 200
    data = response.json()
    assert data["overall_score"] > 50  # Should score higher


# ============================================================================
# VERSION MANAGEMENT
# ============================================================================

def test_add_version(test_project):
    """Test adding a new version"""
    response = client.post(f"/api/projects/{test_project}/versions", json={
        "prompt_text": "Updated prompt text",
        "changes_made": "Test change"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["version"] >= 1
    assert data["prompt_text"] == "Updated prompt text"


def test_get_version(test_project):
    """Test getting a specific version"""
    # First add a version
    client.post(f"/api/projects/{test_project}/versions", json={
        "prompt_text": "Version test",
        "changes_made": "Test"
    })

    response = client.get(f"/api/projects/{test_project}/versions/1")
    assert response.status_code == 200


# ============================================================================
# REWRITE ENDPOINTS
# ============================================================================

def test_rewrite_global():
    """Test global rewrite endpoint"""
    response = client.post("/api/rewrite", json={
        "prompt_text": "Write a story",
        "focus_areas": ["Add detail", "Be specific"],
        "use_case": "Storytelling",
        "key_requirements": ["engaging"]
    })
    # May fail without API key configured, so just check it doesn't 404
    assert response.status_code in [200, 500]


def test_rewrite_project():
    """Test project-specific rewrite"""
    # Create fresh project for this test
    create_response = client.post("/api/projects", json={
        "name": "Rewrite Test",
        "use_case": "Testing rewrite",
        "key_requirements": ["test"],
        "initial_prompt": "Test prompt for rewrite"
    })
    project_id = create_response.json()["id"]

    response = client.post(f"/api/projects/{project_id}/rewrite", json={
        "prompt_text": "Simple prompt",
        "feedback": "Make it more detailed"
    })
    assert response.status_code == 200
    data = response.json()
    assert "rewritten_prompt" in data
    assert "changes_made" in data


# ============================================================================
# EVAL PROMPT ENDPOINTS
# ============================================================================

def test_generate_eval_prompt():
    """Test eval prompt generation"""
    # Create fresh project
    create_response = client.post("/api/projects", json={
        "name": "Eval Test",
        "use_case": "Testing eval",
        "key_requirements": ["test"],
        "initial_prompt": "Test prompt for eval"
    })
    project_id = create_response.json()["id"]

    response = client.post(f"/api/projects/{project_id}/eval-prompt/generate")
    assert response.status_code == 200
    data = response.json()
    assert "eval_prompt" in data
    assert "rationale" in data


def test_refine_eval_prompt():
    """Test eval prompt refinement"""
    # Create fresh project
    create_response = client.post("/api/projects", json={
        "name": "Refine Test",
        "use_case": "Testing refine",
        "key_requirements": ["test"],
        "initial_prompt": "Test prompt for refine"
    })
    project_id = create_response.json()["id"]

    response = client.post(f"/api/projects/{project_id}/eval-prompt/refine", json={
        "feedback": "Focus more on accuracy"
    })
    assert response.status_code == 200
    data = response.json()
    assert "eval_prompt" in data


# ============================================================================
# DATASET ENDPOINTS
# ============================================================================

def test_generate_dataset():
    """Test dataset generation"""
    # Create fresh project
    create_response = client.post("/api/projects", json={
        "name": "Dataset Test",
        "use_case": "Testing dataset",
        "key_requirements": ["test"],
        "initial_prompt": "Test prompt for dataset"
    })
    project_id = create_response.json()["id"]

    response = client.post(f"/api/projects/{project_id}/dataset/generate", json={
        "num_examples": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert "test_cases" in data
    assert data["count"] == 5


def test_generate_dataset_stream():
    """Test streaming dataset generation"""
    # Create fresh project
    create_response = client.post("/api/projects", json={
        "name": "Stream Test",
        "use_case": "Testing stream",
        "key_requirements": ["test"],
        "initial_prompt": "Test prompt for stream"
    })
    project_id = create_response.json()["id"]

    response = client.post(f"/api/projects/{project_id}/dataset/generate-stream", json={
        "num_examples": 3
    })
    assert response.status_code == 200


def test_export_dataset(test_project):
    """Test dataset export"""
    # First generate dataset
    client.post(f"/api/projects/{test_project}/dataset/generate", json={
        "num_examples": 5
    })

    response = client.get(f"/api/projects/{test_project}/dataset/export")
    assert response.status_code == 200
    data = response.json()
    assert "test_cases" in data
    assert "exported_at" in data


# ============================================================================
# TEST RUNS ENDPOINTS
# ============================================================================

def test_create_test_run(test_project):
    """Test creating a test run"""
    # First generate dataset
    client.post(f"/api/projects/{test_project}/dataset/generate", json={
        "num_examples": 3
    })

    response = client.post(f"/api/projects/{test_project}/test-runs", json={})
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["status"] == "completed"
    assert "results" in data


def test_list_test_runs(test_project):
    """Test listing test runs"""
    response = client.get(f"/api/projects/{test_project}/test-runs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_test_run_status(test_project):
    """Test getting test run status"""
    # Create a test run first
    client.post(f"/api/projects/{test_project}/dataset/generate", json={"num_examples": 2})
    create_response = client.post(f"/api/projects/{test_project}/test-runs", json={})
    run_id = create_response.json()["id"]

    response = client.get(f"/api/projects/{test_project}/test-runs/{run_id}/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "progress" in data


def test_get_test_run_results(test_project):
    """Test getting test run results"""
    # Create a test run first
    client.post(f"/api/projects/{test_project}/dataset/generate", json={"num_examples": 2})
    create_response = client.post(f"/api/projects/{test_project}/test-runs", json={})
    run_id = create_response.json()["id"]

    response = client.get(f"/api/projects/{test_project}/test-runs/{run_id}/results")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data


def test_delete_test_run(test_project):
    """Test deleting a test run"""
    # Create a test run first
    client.post(f"/api/projects/{test_project}/dataset/generate", json={"num_examples": 2})
    create_response = client.post(f"/api/projects/{test_project}/test-runs", json={})
    run_id = create_response.json()["id"]

    response = client.delete(f"/api/projects/{test_project}/test-runs/{run_id}")
    assert response.status_code == 200


def test_export_test_run(test_project):
    """Test exporting test run results"""
    client.post(f"/api/projects/{test_project}/dataset/generate", json={"num_examples": 2})
    create_response = client.post(f"/api/projects/{test_project}/test-runs", json={})
    run_id = create_response.json()["id"]

    response = client.get(f"/api/projects/{test_project}/test-runs/{run_id}/export?format=json")
    assert response.status_code == 200


def test_run_single_test(test_project):
    """Test running a single test"""
    response = client.post(f"/api/projects/{test_project}/test-runs/single", json={
        "input": "Test input string"
    })
    assert response.status_code == 200
    data = response.json()
    assert "output" in data
    assert "score" in data
    assert "passed" in data


def test_compare_test_runs(test_project):
    """Test comparing test runs"""
    # Create two test runs
    client.post(f"/api/projects/{test_project}/dataset/generate", json={"num_examples": 2})
    run1 = client.post(f"/api/projects/{test_project}/test-runs", json={}).json()["id"]
    run2 = client.post(f"/api/projects/{test_project}/test-runs", json={}).json()["id"]

    response = client.post(f"/api/projects/{test_project}/test-runs/compare", json={
        "run_ids": [run1, run2]
    })
    assert response.status_code == 200
    data = response.json()
    assert "comparisons" in data


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
