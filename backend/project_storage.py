"""
Simple file-based project storage
"""
import json
import os
from datetime import datetime
from typing import List, Optional
import uuid

from models import SavedProject, ProjectListItem


PROJECTS_DIR = "saved_projects"


def ensure_projects_dir():
    """Ensure projects directory exists"""
    if not os.path.exists(PROJECTS_DIR):
        os.makedirs(PROJECTS_DIR)


def generate_project_id() -> str:
    """Generate unique project ID"""
    return str(uuid.uuid4())


def get_project_file_path(project_id: str) -> str:
    """Get file path for a project"""
    return os.path.join(PROJECTS_DIR, f"{project_id}.json")


def save_project(project: SavedProject) -> SavedProject:
    """Save a project to disk"""
    ensure_projects_dir()

    # Update timestamp
    project.updated_at = datetime.now()

    # Save to file
    file_path = get_project_file_path(project.id)
    with open(file_path, 'w') as f:
        # Convert datetime to ISO format for JSON serialization
        data = project.model_dump()
        if 'created_at' in data and isinstance(data['created_at'], datetime):
            data['created_at'] = data['created_at'].isoformat()
        if 'updated_at' in data and isinstance(data['updated_at'], datetime):
            data['updated_at'] = data['updated_at'].isoformat()
        json.dump(data, f, indent=2, default=str)

    return project


def load_project(project_id: str) -> Optional[SavedProject]:
    """Load a project from disk"""
    file_path = get_project_file_path(project_id)

    if not os.path.exists(file_path):
        return None

    with open(file_path, 'r') as f:
        data = json.load(f)

    return SavedProject(**data)


def list_projects() -> List[ProjectListItem]:
    """List all saved projects"""
    ensure_projects_dir()

    projects = []
    for filename in os.listdir(PROJECTS_DIR):
        if filename.endswith('.json'):
            project_id = filename[:-5]  # Remove .json
            project = load_project(project_id)
            if project:
                projects.append(ProjectListItem(
                    id=project.id,
                    project_name=project.project_name,
                    use_case=project.use_case,
                    requirements=project.requirements,
                    system_prompt_versions=project.system_prompt_versions,
                    created_at=project.created_at,
                    updated_at=project.updated_at,
                    version=project.version,
                    has_results=project.test_results is not None and len(project.test_results) > 0
                ))

    # Sort by updated_at descending (most recent first)
    projects.sort(key=lambda x: x.updated_at, reverse=True)

    return projects


def delete_project(project_id: str) -> bool:
    """Delete a project"""
    file_path = get_project_file_path(project_id)

    if not os.path.exists(file_path):
        return False

    os.remove(file_path)
    return True


def create_new_project(
    project_name: str,
    use_case: str,
    requirements: any,
    key_requirements: list = None,
    initial_prompt: str = ""
) -> SavedProject:
    """Create a new project"""
    project_id = generate_project_id()
    now = datetime.now()

    # Create initial version
    initial_version = {
        "version": 1,
        "prompt_text": initial_prompt,
        "created_at": now.isoformat(),
        "changes_made": "Initial prompt"
    }

    project = SavedProject(
        id=project_id,
        project_name=project_name,
        use_case=use_case,
        requirements=requirements,
        key_requirements=key_requirements or [],
        initial_prompt=initial_prompt,
        system_prompt_versions=[initial_version],
        created_at=now,
        updated_at=now,
        version=1
    )

    return save_project(project)
