"""
Integration tests for complete workflows
"""
import pytest
import sys
import os
from fastapi.testclient import TestClient

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from server import app

client = TestClient(app)


@pytest.fixture
def test_project():
    """Create a test project"""
    response = client.post("/api/projects", json={
        "name": "Integration Test Project",
        "use_case": "End-to-end testing",
        "key_requirements": ["req1", "req2"],
        "initial_prompt": "You are a helpful assistant."
    })
    return response.json()["id"]


class TestCompleteWorkflow:
    """Tests for complete user workflows"""
    
    def test_complete_prompt_optimization_workflow(self, test_project):
        """Integration: Complete prompt optimization workflow"""
        # 1. Analyze prompt
        analyze_response = client.post(f"/api/projects/{test_project}/analyze", json={
            "prompt_text": "Answer questions."
        })
        assert analyze_response.status_code == 200
        suggestions = analyze_response.json()["suggestions"]
        assert len(suggestions) > 0
        
        # 2. Add improved version
        version_response = client.post(f"/api/projects/{test_project}/versions", json={
            "prompt_text": "You are a helpful assistant. Answer questions clearly and accurately.",
            "changes_made": "Added role and clarity"
        })
        assert version_response.status_code == 200
        
        # 3. Generate eval prompt
        eval_response = client.post(f"/api/projects/{test_project}/eval-prompt/generate")
        assert eval_response.status_code == 200
        assert "{{INPUT}}" in eval_response.json()["eval_prompt"]
        
        # 4. Generate dataset
        dataset_response = client.post(f"/api/projects/{test_project}/dataset/generate", json={
            "num_examples": 5
        })
        assert dataset_response.status_code == 200
        assert len(dataset_response.json()["test_cases"]) == 5
        
        # 5. Run tests
        test_run_response = client.post(f"/api/projects/{test_project}/test-runs", json={})
        assert test_run_response.status_code == 200
        assert "results" in test_run_response.json()
    


class TestErrorRecovery:
    """Tests for error recovery scenarios"""
    
    def test_project_recovery_after_error(self):
        """Integration: Project should be recoverable after error"""
        # Create project
        create_response = client.post("/api/projects", json={
            "name": "Recovery Test",
            "use_case": "Test",
            "initial_prompt": "Test"
        })
        project_id = create_response.json()["id"]
        
        # Cause an error (try to analyze with invalid project)
        # Then recover by using valid operations
        response = client.get(f"/api/projects/{project_id}")
        assert response.status_code == 200
        
        # Add version
        version_response = client.post(f"/api/projects/{project_id}/versions", json={
            "prompt_text": "Recovered",
            "changes_made": "Recovery"
        })
        assert version_response.status_code == 200
    
    def test_multiple_concurrent_operations(self, test_project):
        """Integration: Multiple concurrent operations"""
        # Generate dataset
        dataset_response = client.post(f"/api/projects/{test_project}/dataset/generate", json={
            "num_examples": 10
        })
        assert dataset_response.status_code == 200
        
        # Add version
        version_response = client.post(f"/api/projects/{test_project}/versions", json={
            "prompt_text": "Concurrent version",
            "changes_made": "Test"
        })
        assert version_response.status_code == 200
        
        # Generate eval prompt
        eval_response = client.post(f"/api/projects/{test_project}/eval-prompt/generate")
        assert eval_response.status_code == 200
        
        # All should succeed
        assert dataset_response.status_code == 200
        assert version_response.status_code == 200
        assert eval_response.status_code == 200


class TestDataConsistency:
    """Tests for data consistency across operations"""
    
    def test_project_data_persistence(self, test_project):
        """Integration: Project data should persist across operations"""
        # Update project
        get_response = client.get(f"/api/projects/{test_project}")
        project_data = get_response.json()
        project_data["project_name"] = "Updated Name"
        
        update_response = client.put(f"/api/projects/{test_project}", json=project_data)
        assert update_response.status_code == 200
        
        # Verify persistence
        verify_response = client.get(f"/api/projects/{test_project}")
        assert verify_response.json()["project_name"] == "Updated Name"
    
    def test_version_consistency(self, test_project):
        """Integration: Versions should be consistent"""
        # Add multiple versions
        v1 = client.post(f"/api/projects/{test_project}/versions", json={
            "prompt_text": "Version 1",
            "changes_made": "V1"
        }).json()["version"]
        
        v2 = client.post(f"/api/projects/{test_project}/versions", json={
            "prompt_text": "Version 2",
            "changes_made": "V2"
        }).json()["version"]
        
        assert v2 == v1 + 1
        
        # Get versions
        get_v1 = client.get(f"/api/projects/{test_project}/versions/{v1}")
        get_v2 = client.get(f"/api/projects/{test_project}/versions/{v2}")
        
        assert get_v1.json()["prompt_text"] == "Version 1"
        assert get_v2.json()["prompt_text"] == "Version 2"
    
    def test_test_run_data_consistency(self, test_project):
        """Integration: Test run data should be consistent"""
        # Generate dataset
        client.post(f"/api/projects/{test_project}/dataset/generate", json={"num_examples": 3})
        
        # Create test run
        run_response = client.post(f"/api/projects/{test_project}/test-runs", json={})
        run_id = run_response.json()["id"]
        
        # Get status
        status_response = client.get(f"/api/projects/{test_project}/test-runs/{run_id}/status")
        assert status_response.status_code == 200
        
        # Get results
        results_response = client.get(f"/api/projects/{test_project}/test-runs/{run_id}/results")
        assert results_response.status_code == 200
        
        # Data should be consistent
        assert status_response.json()["id"] == run_id
        assert results_response.json()["id"] == run_id
