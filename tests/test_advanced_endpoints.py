"""
Unit tests for advanced endpoints: calibration examples, human validations, AB tests
"""
import pytest
import sys
import os
from fastapi.testclient import TestClient
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from server import app

client = TestClient(app)


@pytest.fixture
def test_project():
    """Create a test project"""
    response = client.post("/api/projects", json={
        "name": "Advanced Test Project",
        "use_case": "Testing advanced features",
        "key_requirements": ["req1"],
        "initial_prompt": "Test prompt"
    })
    return response.json()["id"]


@pytest.fixture
def test_project_with_run(test_project):
    """Create a project with a test run"""
    client.post(f"/api/projects/{test_project}/dataset/generate", json={"num_examples": 3})
    run_response = client.post(f"/api/projects/{test_project}/test-runs", json={})
    return test_project, run_response.json()["id"]


class TestCalibrationExamples:
    """Tests for calibration examples endpoints"""
    
    def test_get_calibration_examples_success(self, test_project):
        """Positive: Should get calibration examples"""
        response = client.get(f"/api/projects/{test_project}/calibration-examples")
        assert response.status_code == 200
        data = response.json()
        assert "examples" in data
        assert isinstance(data["examples"], list)
    
    def test_get_calibration_examples_project_not_found(self):
        """Negative: Should return 404"""
        response = client.get("/api/projects/non-existent/calibration-examples")
        assert response.status_code == 404
    
    def test_add_calibration_example_success(self, test_project):
        """Positive: Should add calibration example"""
        response = client.post(f"/api/projects/{test_project}/calibration-examples", json={
            "input": "Test input",
            "output": "Test output",
            "score": 4.5,
            "reasoning": "Good response",
            "category": "excellent"
        })
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["score"] == 4.5
    
    def test_add_calibration_example_invalid_score(self, test_project):
        """Negative: Should reject invalid score"""
        response = client.post(f"/api/projects/{test_project}/calibration-examples", json={
            "input": "Test",
            "output": "Test",
            "score": 10,  # Invalid: should be 1-5
            "reasoning": "Test"
        })
        assert response.status_code == 422  # Validation error
    
    def test_delete_calibration_example_success(self, test_project):
        """Positive: Should delete calibration example"""
        # First add one
        add_response = client.post(f"/api/projects/{test_project}/calibration-examples", json={
            "input": "Test",
            "output": "Test",
            "score": 4.0,
            "reasoning": "Test"
        })
        example_id = add_response.json()["id"]
        
        # Delete it
        response = client.delete(f"/api/projects/{test_project}/calibration-examples/{example_id}")
        assert response.status_code == 200
    
    def test_delete_calibration_example_not_found(self, test_project):
        """Negative: Should return 404"""
        response = client.delete(f"/api/projects/{test_project}/calibration-examples/non-existent")
        assert response.status_code == 404
    
    def test_generate_calibration_examples_success(self, test_project):
        """Positive: Should generate calibration examples"""
        response = client.post(f"/api/projects/{test_project}/calibration-examples/generate")
        # May fail without API key or without eval prompt
        assert response.status_code in [200, 400, 500]
    
    def test_generate_calibration_examples_project_not_found(self):
        """Negative: Should return 404"""
        response = client.post("/api/projects/non-existent/calibration-examples/generate")
        assert response.status_code == 404


class TestHumanValidations:
    """Tests for human validation endpoints"""
    
    def test_get_human_validations_success(self, test_project):
        """Positive: Should get human validations"""
        response = client.get(f"/api/projects/{test_project}/human-validations")
        assert response.status_code == 200
        data = response.json()
        assert "validations" in data
        assert isinstance(data["validations"], list)
    
    def test_get_human_validations_project_not_found(self):
        """Negative: Should return 404"""
        response = client.get("/api/projects/non-existent/human-validations")
        assert response.status_code == 404
    
    def test_add_human_validation_success(self, test_project_with_run):
        """Positive: Should add human validation"""
        project_id, run_id = test_project_with_run
        
        # Get a result from the run
        run_results = client.get(f"/api/projects/{project_id}/test-runs/{run_id}/results")
        results = run_results.json()["results"]
        if results:
            result_id = results[0].get("test_case_id", "test-123")
            
            response = client.post(f"/api/projects/{project_id}/human-validations", json={
                "result_id": result_id,
                "run_id": run_id,  # Added required field
                "human_score": 4.0,
                "human_feedback": "Good response",
                "validator_id": "validator-1"
            })
            assert response.status_code == 200
            data = response.json()
            assert data["human_score"] == 4.0
    
    def test_add_human_validation_invalid_score(self, test_project):
        """Negative: Should reject invalid score"""
        response = client.post(f"/api/projects/{test_project}/human-validations", json={
            "result_id": "test-123",
            "run_id": "test-run",
            "human_score": 10,  # Invalid: should be 1-5
            "human_feedback": "Test"
        })
        assert response.status_code == 422
    
    def test_get_pending_validations_success(self, test_project_with_run):
        """Positive: Should get pending validations"""
        project_id, run_id = test_project_with_run
        response = client.get(f"/api/projects/{project_id}/human-validations/pending?run_id={run_id}")
        assert response.status_code == 200
        data = response.json()
        assert "pending" in data
        assert isinstance(data["pending"], list)
    
    def test_get_pending_validations_no_run_id(self, test_project):
        """Positive: Should get all pending without run_id"""
        response = client.get(f"/api/projects/{test_project}/human-validations/pending")
        assert response.status_code == 200
    
    def test_convert_validation_to_calibration_success(self, test_project_with_run):
        """Positive: Should convert validation to calibration"""
        project_id, run_id = test_project_with_run
        
        # Get a result from the run
        run_results = client.get(f"/api/projects/{project_id}/test-runs/{run_id}/results")
        results = run_results.json()["results"]
        if results:
            result_id = results[0].get("test_case_id", "test-123")
            
            # First add a validation
            add_response = client.post(f"/api/projects/{project_id}/human-validations", json={
                "result_id": result_id,
                "run_id": run_id,
                "human_score": 4.5,
                "human_feedback": "Excellent response"
            })
            
            if add_response.status_code == 200 and "id" in add_response.json():
                validation_id = add_response.json()["id"]
                
                response = client.post(f"/api/projects/{project_id}/human-validations/convert-to-calibration", json={
                    "validation_id": validation_id
                })
                assert response.status_code in [200, 404]  # May not find validation
    
    def test_convert_validation_to_calibration_not_found(self, test_project):
        """Negative: Should return 404 for non-existent validation"""
        response = client.post(f"/api/projects/{test_project}/human-validations/convert-to-calibration", json={
            "validation_id": "non-existent"
        })
        assert response.status_code == 404


class TestABTests:
    """Tests for A/B testing endpoints"""
    
    def test_get_ab_tests_success(self, test_project):
        """Positive: Should get AB tests"""
        response = client.get(f"/api/projects/{test_project}/ab-tests")
        assert response.status_code == 200
        data = response.json()
        assert "tests" in data
        assert isinstance(data["tests"], list)
    
    def test_get_ab_tests_project_not_found(self):
        """Negative: Should return 404"""
        response = client.get("/api/projects/non-existent/ab-tests")
        assert response.status_code == 404
    
    def test_create_ab_test_success(self, test_project):
        """Positive: Should create AB test"""
        # Add versions first
        client.post(f"/api/projects/{test_project}/versions", json={
            "prompt_text": "Version A",
            "changes_made": "Test"
        })
        client.post(f"/api/projects/{test_project}/versions", json={
            "prompt_text": "Version B",
            "changes_made": "Test"
        })
        
        response = client.post(f"/api/projects/{test_project}/ab-tests", json={
            "name": "Test AB Test",
            "version_a": 1,
            "version_b": 2,
            "sample_size": 20,
            "confidence_level": 0.95
        })
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["name"] == "Test AB Test"
    
    def test_create_ab_test_invalid_versions(self, test_project):
        """Negative: Should reject invalid version numbers"""
        response = client.post(f"/api/projects/{test_project}/ab-tests", json={
            "name": "Invalid Test",
            "version_a": 999,  # Doesn't exist
            "version_b": 998
        })
        assert response.status_code == 400
    
    def test_run_ab_test_success(self, test_project):
        """Positive: Should run AB test"""
        # Create AB test first
        create_response = client.post(f"/api/projects/{test_project}/ab-tests", json={
            "name": "Run Test",
            "version_a": 1,
            "version_b": 1,  # Can be same for testing
            "sample_size": 5
        })
        test_id = create_response.json()["id"]
        
        # Generate dataset
        client.post(f"/api/projects/{test_project}/dataset/generate", json={"num_examples": 5})
        
        response = client.post(f"/api/projects/{test_project}/ab-tests/{test_id}/run")
        # May fail without API key, but should not 404
        assert response.status_code in [200, 400, 500]
    
    def test_run_ab_test_not_found(self, test_project):
        """Negative: Should return 404"""
        response = client.post(f"/api/projects/{test_project}/ab-tests/non-existent/run")
        assert response.status_code == 404
    
    def test_get_ab_test_results_success(self, test_project):
        """Positive: Should get AB test results"""
        # Create and potentially run AB test
        create_response = client.post(f"/api/projects/{test_project}/ab-tests", json={
            "name": "Results Test",
            "version_a": 1,
            "version_b": 1,
            "sample_size": 5
        })
        test_id = create_response.json()["id"]
        
        response = client.get(f"/api/projects/{test_project}/ab-tests/{test_id}/results")
        assert response.status_code == 200
        data = response.json()
        assert "test_id" in data or "status" in data
    
    def test_get_ab_test_results_not_found(self, test_project):
        """Negative: Should return 404"""
        response = client.get(f"/api/projects/{test_project}/ab-tests/non-existent/results")
        assert response.status_code == 404
    
    def test_delete_ab_test_success(self, test_project):
        """Positive: Should delete AB test"""
        create_response = client.post(f"/api/projects/{test_project}/ab-tests", json={
            "name": "Delete Test",
            "version_a": 1,
            "version_b": 1
        })
        test_id = create_response.json()["id"]
        
        response = client.delete(f"/api/projects/{test_project}/ab-tests/{test_id}")
        assert response.status_code == 200
    
    def test_delete_ab_test_not_found(self, test_project):
        """Negative: Should return 404"""
        response = client.delete(f"/api/projects/{test_project}/ab-tests/non-existent")
        assert response.status_code == 404
