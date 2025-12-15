"""
Pytest configuration and shared fixtures
"""
import pytest
import sys
import os
import shutil
from fastapi.testclient import TestClient

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from server import app

# Set test environment
os.environ["TESTING"] = "true"

client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before all tests"""
    # Ensure test projects directory exists
    if not os.path.exists("saved_projects"):
        os.makedirs("saved_projects")
    yield
    # Cleanup after all tests
    # Note: Individual test files handle their own cleanup


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up test files before and after each test"""
    yield
    # Cleanup handled by individual test modules
