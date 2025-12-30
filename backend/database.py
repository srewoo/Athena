"""
SQLite Database Layer for Athena
Replaces file-based storage with proper database with:
- ACID transactions
- Concurrent access support
- Proper indexing
- Migration support
"""
import sqlite3
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
from threading import local
import os

logger = logging.getLogger(__name__)

# Thread-local storage for connections
_thread_local = local()

# Database file path
DB_PATH = os.environ.get("ATHENA_DB_PATH", "athena.db")


def get_connection() -> sqlite3.Connection:
    """Get a thread-local database connection"""
    if not hasattr(_thread_local, "connection") or _thread_local.connection is None:
        _thread_local.connection = sqlite3.connect(
            DB_PATH,
            check_same_thread=False,
            timeout=30.0
        )
        _thread_local.connection.row_factory = sqlite3.Row
        # Enable foreign keys
        _thread_local.connection.execute("PRAGMA foreign_keys = ON")
        # Enable WAL mode for better concurrent access
        _thread_local.connection.execute("PRAGMA journal_mode = WAL")
    return _thread_local.connection


@contextmanager
def get_db():
    """Context manager for database operations with automatic commit/rollback"""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise


def init_database():
    """Initialize database schema"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Projects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                project_name TEXT NOT NULL,
                use_case TEXT NOT NULL,
                requirements TEXT,
                key_requirements TEXT,
                initial_prompt TEXT,
                optimized_prompt TEXT,
                optimization_score REAL,
                eval_prompt TEXT,
                eval_rationale TEXT,
                dataset TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                version INTEGER DEFAULT 1
            )
        """)

        # Create index on updated_at for efficient sorting
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_projects_updated_at
            ON projects(updated_at DESC)
        """)

        # Prompt versions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                prompt_text TEXT NOT NULL,
                changes_made TEXT,
                score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                UNIQUE(project_id, version)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prompt_versions_project
            ON prompt_versions(project_id, version DESC)
        """)

        # Test runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_runs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                version_number INTEGER,
                status TEXT DEFAULT 'pending',
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_runs_project
            ON test_runs(project_id, created_at DESC)
        """)

        # Test results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                test_case TEXT NOT NULL,
                prompt_output TEXT,
                eval_score REAL,
                eval_feedback TEXT,
                passed INTEGER,
                latency_ms INTEGER,
                tokens_used INTEGER,
                dimension_scores TEXT,
                FOREIGN KEY (run_id) REFERENCES test_runs(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_results_run
            ON test_results(run_id)
        """)

        # Calibration examples table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibration_examples (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                input TEXT NOT NULL,
                output TEXT NOT NULL,
                score REAL NOT NULL,
                reasoning TEXT,
                category TEXT DEFAULT 'general',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            )
        """)

        # Human validations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS human_validations (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                run_id TEXT,
                result_id INTEGER,
                human_score REAL NOT NULL,
                human_feedback TEXT,
                validator_id TEXT,
                validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                agrees_with_llm INTEGER,
                score_difference REAL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                FOREIGN KEY (run_id) REFERENCES test_runs(id) ON DELETE SET NULL
            )
        """)

        # A/B tests table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_tests (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                name TEXT NOT NULL,
                version_a INTEGER NOT NULL,
                version_b INTEGER NOT NULL,
                sample_size INTEGER DEFAULT 30,
                confidence_level REAL DEFAULT 0.95,
                status TEXT DEFAULT 'running',
                results TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            )
        """)

        # Evaluations table (for quick evaluations from Dashboard)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id TEXT PRIMARY KEY,
                prompt_text TEXT NOT NULL,
                evaluation_mode TEXT,
                total_score REAL,
                max_score REAL DEFAULT 250,
                categories TEXT,
                suggestions TEXT,
                overall_assessment TEXT,
                llm_provider TEXT,
                word_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_evaluations_created
            ON evaluations(created_at DESC)
        """)

        # Settings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # API keys table (encrypted storage)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                encrypted_key TEXT NOT NULL,
                key_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP
            )
        """)

        logger.info("Database initialized successfully")


# ============================================================================
# Project Operations
# ============================================================================

def create_project(
    project_name: str,
    use_case: str,
    requirements: Any,
    key_requirements: List[str] = None,
    initial_prompt: str = ""
) -> Dict[str, Any]:
    """Create a new project"""
    project_id = str(uuid.uuid4())
    now = datetime.now()

    with get_db() as conn:
        cursor = conn.cursor()

        # Insert project
        cursor.execute("""
            INSERT INTO projects (
                id, project_name, use_case, requirements, key_requirements,
                initial_prompt, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            project_id,
            project_name,
            use_case,
            json.dumps(requirements) if isinstance(requirements, dict) else requirements,
            json.dumps(key_requirements) if key_requirements else None,
            initial_prompt,
            now.isoformat(),
            now.isoformat()
        ))

        # Create initial version
        if initial_prompt:
            cursor.execute("""
                INSERT INTO prompt_versions (
                    project_id, version, prompt_text, changes_made, created_at
                ) VALUES (?, 1, ?, 'Initial prompt', ?)
            """, (project_id, initial_prompt, now.isoformat()))

    return get_project(project_id)


def get_project(project_id: str) -> Optional[Dict[str, Any]]:
    """Get a project by ID with all related data"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Get project
        cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = cursor.fetchone()

        if not row:
            return None

        project = dict(row)

        # Parse JSON fields
        if project.get("requirements"):
            try:
                project["requirements"] = json.loads(project["requirements"])
            except (json.JSONDecodeError, TypeError):
                pass

        if project.get("key_requirements"):
            try:
                project["key_requirements"] = json.loads(project["key_requirements"])
            except (json.JSONDecodeError, TypeError):
                project["key_requirements"] = []

        if project.get("dataset"):
            try:
                project["dataset"] = json.loads(project["dataset"])
            except (json.JSONDecodeError, TypeError):
                pass

        # Get prompt versions
        cursor.execute("""
            SELECT version, prompt_text, changes_made, score, created_at
            FROM prompt_versions
            WHERE project_id = ?
            ORDER BY version ASC
        """, (project_id,))

        project["system_prompt_versions"] = [
            {
                "version": r["version"],
                "prompt_text": r["prompt_text"],
                "changes_made": r["changes_made"],
                "score": r["score"],
                "created_at": r["created_at"]
            }
            for r in cursor.fetchall()
        ]

        # Get test runs
        cursor.execute("""
            SELECT id, version_number, status, summary, created_at, completed_at
            FROM test_runs
            WHERE project_id = ?
            ORDER BY created_at DESC
        """, (project_id,))

        project["test_runs"] = [
            {
                "id": r["id"],
                "version_number": r["version_number"],
                "status": r["status"],
                "summary": json.loads(r["summary"]) if r["summary"] else None,
                "created_at": r["created_at"],
                "completed_at": r["completed_at"]
            }
            for r in cursor.fetchall()
        ]

        # Get calibration examples
        cursor.execute("""
            SELECT id, input, output, score, reasoning, category, created_at
            FROM calibration_examples
            WHERE project_id = ?
            ORDER BY created_at DESC
        """, (project_id,))

        project["calibration_examples"] = [dict(r) for r in cursor.fetchall()]

        # Get human validations
        cursor.execute("""
            SELECT * FROM human_validations
            WHERE project_id = ?
            ORDER BY validated_at DESC
        """, (project_id,))

        project["human_validations"] = [dict(r) for r in cursor.fetchall()]

        # Get A/B tests
        cursor.execute("""
            SELECT * FROM ab_tests
            WHERE project_id = ?
            ORDER BY created_at DESC
        """, (project_id,))

        project["ab_tests"] = [
            {
                **dict(r),
                "results": json.loads(r["results"]) if r["results"] else None
            }
            for r in cursor.fetchall()
        ]

        return project


def update_project(project_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update a project"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Build update query
        set_clauses = []
        values = []

        for key, value in updates.items():
            if key in ["id", "created_at"]:
                continue

            if key in ["requirements", "key_requirements", "dataset"] and isinstance(value, (dict, list)):
                value = json.dumps(value)

            set_clauses.append(f"{key} = ?")
            values.append(value)

        if not set_clauses:
            return get_project(project_id)

        # Always update timestamp
        set_clauses.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(project_id)

        cursor.execute(f"""
            UPDATE projects
            SET {', '.join(set_clauses)}
            WHERE id = ?
        """, values)

    return get_project(project_id)


def list_projects(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """List all projects"""
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT p.*,
                   (SELECT COUNT(*) FROM test_results tr
                    JOIN test_runs r ON tr.run_id = r.id
                    WHERE r.project_id = p.id) as result_count
            FROM projects p
            ORDER BY p.updated_at DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))

        projects = []
        for row in cursor.fetchall():
            project = dict(row)

            # Parse requirements
            if project.get("requirements"):
                try:
                    project["requirements"] = json.loads(project["requirements"])
                except (json.JSONDecodeError, TypeError):
                    pass

            # Get versions count
            cursor.execute("""
                SELECT version, prompt_text, changes_made, created_at
                FROM prompt_versions
                WHERE project_id = ?
                ORDER BY version ASC
            """, (project["id"],))

            project["system_prompt_versions"] = [dict(r) for r in cursor.fetchall()]
            project["has_results"] = project.get("result_count", 0) > 0

            projects.append(project)

        return projects


def delete_project(project_id: str) -> bool:
    """Delete a project and all related data"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        return cursor.rowcount > 0


def cleanup_old_projects(days: int = 30) -> Dict[str, Any]:
    """Delete projects older than specified days"""
    cutoff = datetime.now() - timedelta(days=days)

    with get_db() as conn:
        cursor = conn.cursor()

        # Get projects to delete
        cursor.execute("""
            SELECT id, project_name, updated_at FROM projects
            WHERE updated_at < ?
        """, (cutoff.isoformat(),))

        to_delete = [dict(r) for r in cursor.fetchall()]

        # Delete them
        cursor.execute("DELETE FROM projects WHERE updated_at < ?", (cutoff.isoformat(),))

        return {
            "deleted_count": len(to_delete),
            "deleted_projects": to_delete,
            "cutoff_date": cutoff.isoformat()
        }


# ============================================================================
# Prompt Version Operations
# ============================================================================

def add_prompt_version(
    project_id: str,
    prompt_text: str,
    changes_made: str = None,
    score: float = None
) -> Dict[str, Any]:
    """Add a new prompt version"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Get next version number
        cursor.execute("""
            SELECT COALESCE(MAX(version), 0) + 1 as next_version
            FROM prompt_versions WHERE project_id = ?
        """, (project_id,))

        next_version = cursor.fetchone()["next_version"]

        cursor.execute("""
            INSERT INTO prompt_versions (
                project_id, version, prompt_text, changes_made, score, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            project_id, next_version, prompt_text, changes_made, score,
            datetime.now().isoformat()
        ))

        # Update project version
        cursor.execute("""
            UPDATE projects SET version = ?, updated_at = ?
            WHERE id = ?
        """, (next_version, datetime.now().isoformat(), project_id))

        return {
            "version": next_version,
            "prompt_text": prompt_text,
            "changes_made": changes_made,
            "score": score
        }


# ============================================================================
# Test Run Operations
# ============================================================================

def create_test_run(project_id: str, version_number: int = None) -> Dict[str, Any]:
    """Create a new test run"""
    run_id = str(uuid.uuid4())

    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO test_runs (id, project_id, version_number, status, created_at)
            VALUES (?, ?, ?, 'pending', ?)
        """, (run_id, project_id, version_number, datetime.now().isoformat()))

        return {
            "id": run_id,
            "project_id": project_id,
            "version_number": version_number,
            "status": "pending"
        }


def update_test_run(run_id: str, status: str = None, summary: Dict = None) -> None:
    """Update a test run"""
    with get_db() as conn:
        cursor = conn.cursor()

        updates = []
        values = []

        if status:
            updates.append("status = ?")
            values.append(status)

            if status == "completed":
                updates.append("completed_at = ?")
                values.append(datetime.now().isoformat())

        if summary:
            updates.append("summary = ?")
            values.append(json.dumps(summary))

        if updates:
            values.append(run_id)
            cursor.execute(f"""
                UPDATE test_runs SET {', '.join(updates)} WHERE id = ?
            """, values)


def add_test_result(
    run_id: str,
    test_case: Dict,
    prompt_output: str,
    eval_score: float,
    eval_feedback: str,
    passed: bool,
    latency_ms: int,
    tokens_used: int,
    dimension_scores: Dict = None
) -> int:
    """Add a test result"""
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO test_results (
                run_id, test_case, prompt_output, eval_score, eval_feedback,
                passed, latency_ms, tokens_used, dimension_scores
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            json.dumps(test_case),
            prompt_output,
            eval_score,
            eval_feedback,
            1 if passed else 0,
            latency_ms,
            tokens_used,
            json.dumps(dimension_scores) if dimension_scores else None
        ))

        return cursor.lastrowid


def get_test_run_results(run_id: str) -> List[Dict[str, Any]]:
    """Get all results for a test run"""
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM test_results WHERE run_id = ?
        """, (run_id,))

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            result["test_case"] = json.loads(result["test_case"])
            result["passed"] = bool(result["passed"])
            if result.get("dimension_scores"):
                result["dimension_scores"] = json.loads(result["dimension_scores"])
            results.append(result)

        return results


# ============================================================================
# Evaluation Operations (for Dashboard)
# ============================================================================

def save_evaluation(evaluation: Dict[str, Any]) -> str:
    """Save an evaluation"""
    eval_id = evaluation.get("id") or str(uuid.uuid4())

    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO evaluations (
                id, prompt_text, evaluation_mode, total_score, max_score,
                categories, suggestions, overall_assessment, llm_provider,
                word_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            eval_id,
            evaluation.get("prompt_text", ""),
            evaluation.get("evaluation_mode"),
            evaluation.get("total_score"),
            evaluation.get("max_score", 250),
            json.dumps(evaluation.get("categories", {})),
            json.dumps(evaluation.get("refinement_suggestions", [])),
            evaluation.get("overall_assessment"),
            evaluation.get("llm_provider"),
            evaluation.get("word_count"),
            evaluation.get("created_at", datetime.now().isoformat())
        ))

    return eval_id


def get_evaluation(eval_id: str) -> Optional[Dict[str, Any]]:
    """Get an evaluation by ID"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM evaluations WHERE id = ?", (eval_id,))
        row = cursor.fetchone()

        if not row:
            return None

        evaluation = dict(row)
        evaluation["categories"] = json.loads(evaluation["categories"]) if evaluation["categories"] else {}
        evaluation["refinement_suggestions"] = json.loads(evaluation["suggestions"]) if evaluation["suggestions"] else []
        del evaluation["suggestions"]

        return evaluation


def list_evaluations(limit: int = 100) -> List[Dict[str, Any]]:
    """List all evaluations"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM evaluations ORDER BY created_at DESC LIMIT ?
        """, (limit,))

        evaluations = []
        for row in cursor.fetchall():
            evaluation = dict(row)
            evaluation["categories"] = json.loads(evaluation["categories"]) if evaluation["categories"] else {}
            evaluation["refinement_suggestions"] = json.loads(evaluation["suggestions"]) if evaluation["suggestions"] else []
            del evaluation["suggestions"]
            evaluations.append(evaluation)

        return evaluations


def delete_evaluation(eval_id: str) -> bool:
    """Delete an evaluation"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM evaluations WHERE id = ?", (eval_id,))
        return cursor.rowcount > 0


# ============================================================================
# Settings Operations
# ============================================================================

def get_setting(key: str) -> Optional[str]:
    """Get a setting value"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row["value"] if row else None


def set_setting(key: str, value: str) -> None:
    """Set a setting value"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO settings (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (key, value, datetime.now().isoformat()))


def get_all_settings() -> Dict[str, str]:
    """Get all settings"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM settings")
        return {row["key"]: row["value"] for row in cursor.fetchall()}


# ============================================================================
# Storage Stats
# ============================================================================

def get_storage_stats() -> Dict[str, Any]:
    """Get database storage statistics"""
    with get_db() as conn:
        cursor = conn.cursor()

        stats = {}

        # Project count
        cursor.execute("SELECT COUNT(*) as count FROM projects")
        stats["total_projects"] = cursor.fetchone()["count"]

        # Test run count
        cursor.execute("SELECT COUNT(*) as count FROM test_runs")
        stats["total_test_runs"] = cursor.fetchone()["count"]

        # Test result count
        cursor.execute("SELECT COUNT(*) as count FROM test_results")
        stats["total_test_results"] = cursor.fetchone()["count"]

        # Evaluation count
        cursor.execute("SELECT COUNT(*) as count FROM evaluations")
        stats["total_evaluations"] = cursor.fetchone()["count"]

        # Oldest and newest projects
        cursor.execute("""
            SELECT id, project_name, updated_at FROM projects
            ORDER BY updated_at ASC LIMIT 1
        """)
        oldest = cursor.fetchone()
        stats["oldest_project"] = dict(oldest) if oldest else None

        cursor.execute("""
            SELECT id, project_name, updated_at FROM projects
            ORDER BY updated_at DESC LIMIT 1
        """)
        newest = cursor.fetchone()
        stats["newest_project"] = dict(newest) if newest else None

        # Database file size
        if os.path.exists(DB_PATH):
            stats["total_size_bytes"] = os.path.getsize(DB_PATH)
            stats["total_size_mb"] = round(stats["total_size_bytes"] / (1024 * 1024), 2)
        else:
            stats["total_size_bytes"] = 0
            stats["total_size_mb"] = 0

        return stats


# Initialize database on module import
init_database()
