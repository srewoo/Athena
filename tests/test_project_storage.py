"""
Unit tests for project_storage.py functions
"""
import pytest
import sys
import os
import json
import shutil
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from models import SavedProject
import project_storage


@pytest.fixture(autouse=True)
def cleanup_test_projects():
    """Clean up test projects before and after each test"""
    # Setup
    project_storage.ensure_projects_dir()
    yield
    # Teardown - clean up test files
    if os.path.exists(project_storage.PROJECTS_DIR):
        for filename in os.listdir(project_storage.PROJECTS_DIR):
            if filename.endswith('.json') and filename.startswith('test_'):
                filepath = os.path.join(project_storage.PROJECTS_DIR, filename)
                try:
                    os.remove(filepath)
                except:
                    pass


class TestProjectStorage:
    """Tests for project storage functions"""
    
    def test_ensure_projects_dir_creates_directory(self):
        """Positive: Should create directory if it doesn't exist"""
        if os.path.exists(project_storage.PROJECTS_DIR):
            shutil.rmtree(project_storage.PROJECTS_DIR)
        project_storage.ensure_projects_dir()
        assert os.path.exists(project_storage.PROJECTS_DIR)
    
    def test_generate_project_id_returns_uuid(self):
        """Positive: Should generate valid UUID string"""
        project_id = project_storage.generate_project_id()
        assert isinstance(project_id, str)
        assert len(project_id) == 36  # UUID format
    
    def test_generate_project_id_unique(self):
        """Positive: Should generate unique IDs"""
        id1 = project_storage.generate_project_id()
        id2 = project_storage.generate_project_id()
        assert id1 != id2
    
    def test_get_project_file_path(self):
        """Positive: Should return correct file path"""
        project_id = "test-123"
        path = project_storage.get_project_file_path(project_id)
        assert path.endswith("test-123.json")
        assert project_storage.PROJECTS_DIR in path
    
    def test_save_and_load_project(self):
        """Positive: Should save and load project correctly"""
        project = SavedProject(
            id="test_save_load",
            project_name="Test Project",
            use_case="Testing",
            requirements={"use_case": "Testing"},
            initial_prompt="Test prompt",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Save
        saved = project_storage.save_project(project)
        assert saved.id == project.id
        
        # Load
        loaded = project_storage.load_project("test_save_load")
        assert loaded is not None
        assert loaded.project_name == "Test Project"
        assert loaded.use_case == "Testing"
    
    def test_load_nonexistent_project(self):
        """Negative: Should return None for non-existent project"""
        result = project_storage.load_project("nonexistent-id-12345")
        assert result is None
    
    def test_list_projects_returns_list(self):
        """Positive: Should return list of projects"""
        # Create a test project
        project = SavedProject(
            id="test_list_1",
            project_name="List Test",
            use_case="Testing",
            requirements={"use_case": "Testing"},
            initial_prompt="Test",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        project_storage.save_project(project)
        
        projects = project_storage.list_projects()
        assert isinstance(projects, list)
        assert len(projects) > 0
    
    def test_list_projects_sorted_by_updated_at(self):
        """Positive: Should sort by updated_at descending"""
        # Create two projects with different timestamps
        project1 = SavedProject(
            id="test_sort_1",
            project_name="Older",
            use_case="Test",
            requirements={},
            initial_prompt="Test",
            created_at=datetime.now(),
            updated_at=datetime(2024, 1, 1)
        )
        project2 = SavedProject(
            id="test_sort_2",
            project_name="Newer",
            use_case="Test",
            requirements={},
            initial_prompt="Test",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        project_storage.save_project(project1)
        project_storage.save_project(project2)
        
        projects = project_storage.list_projects()
        # Newer should come first
        assert projects[0].updated_at >= projects[1].updated_at
    
    def test_delete_project(self):
        """Positive: Should delete project file"""
        project = SavedProject(
            id="test_delete",
            project_name="Delete Test",
            use_case="Test",
            requirements={},
            initial_prompt="Test",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        project_storage.save_project(project)
        
        # Delete
        result = project_storage.delete_project("test_delete")
        assert result is True
        
        # Verify deleted
        loaded = project_storage.load_project("test_delete")
        assert loaded is None
    
    def test_delete_nonexistent_project(self):
        """Negative: Should return False for non-existent project"""
        result = project_storage.delete_project("nonexistent-delete")
        assert result is False
    
    def test_create_new_project(self):
        """Positive: Should create new project with initial version"""
        project = project_storage.create_new_project(
            project_name="New Project",
            use_case="Testing",
            requirements={"use_case": "Testing"},
            key_requirements=["req1", "req2"],
            initial_prompt="Initial prompt"
        )
        
        assert project.id is not None
        assert project.project_name == "New Project"
        assert project.use_case == "Testing"
        assert project.system_prompt_versions is not None
        assert len(project.system_prompt_versions) == 1
        assert project.system_prompt_versions[0]["version"] == 1
    
    def test_create_new_project_without_key_requirements(self):
        """Positive: Should handle missing key_requirements"""
        project = project_storage.create_new_project(
            project_name="No Reqs",
            use_case="Test",
            requirements={},
            initial_prompt="Test"
        )
        assert project.key_requirements == []
    
    def test_save_project_updates_timestamp(self):
        """Positive: Should update updated_at timestamp"""
        project = SavedProject(
            id="test_timestamp",
            project_name="Test",
            use_case="Test",
            requirements={},
            initial_prompt="Test",
            created_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1)
        )
        
        import time
        time.sleep(0.1)  # Ensure timestamp difference
        saved = project_storage.save_project(project)
        
        assert saved.updated_at > project.created_at
