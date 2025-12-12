"""
Simple data models for the 5-step prompt testing workflow
"""
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime


# ============= Request Models =============

class ProjectInput(BaseModel):
    """User input for the prompt testing workflow"""
    provider: str  # "openai", "claude", "gemini"
    api_key: str
    model_name: Optional[str] = None
    project_name: str
    use_case: str
    requirements: str
    initial_prompt: str


# ============= Step 2: Optimization =============

class PromptOptimizationResult(BaseModel):
    """Result from Step 2: Prompt Optimization"""
    optimized_prompt: str
    score: float  # Quality score 1-10
    analysis: str
    improvements: List[str]
    suggestions: List[str]


# ============= Step 3: Evaluation Prompt =============

class EvaluationPromptResult(BaseModel):
    """Result from Step 3: Evaluation Prompt Generation"""
    eval_prompt: str
    eval_criteria: List[str]
    rationale: str


# ============= Step 4: Test Data =============

class TestDataResult(BaseModel):
    """Result from Step 4: Test Data Generation"""
    test_cases: List[Dict[str, Any]]
    count: int
    categories: Dict[str, int]


# ============= Step 5: Test Execution =============

class TestExecutionResult(BaseModel):
    """Result from executing a single test case"""
    test_case: Dict[str, Any]
    prompt_output: str
    eval_score: float
    eval_feedback: str
    passed: bool
    latency_ms: int
    tokens_used: int


class FinalReport(BaseModel):
    """Final report summarizing all test results"""
    project_name: str
    optimization_score: float
    pass_rate: float
    avg_score: float
    total_tests: int
    passed_tests: int
    category_breakdown: Dict[str, Any]
    total_tokens: int
    avg_latency_ms: float


# ============= Project Management =============

class SavedProject(BaseModel):
    """Saved project with all workflow data"""
    id: str
    project_name: str
    use_case: str
    requirements: Any  # Can be string or object with use_case
    key_requirements: Optional[List[str]] = None
    initial_prompt: str
    optimized_prompt: Optional[str] = None
    optimization_score: Optional[float] = None
    eval_prompt: Optional[str] = None
    test_cases: Optional[List[Dict[str, Any]]] = None
    test_results: Optional[List[TestExecutionResult]] = None
    final_report: Optional[FinalReport] = None
    system_prompt_versions: Optional[List[Dict[str, Any]]] = None
    created_at: datetime
    updated_at: datetime
    version: int = 1


class ProjectListItem(BaseModel):
    """Minimal project info for listing"""
    id: str
    project_name: str
    use_case: str
    requirements: Any  # Include for frontend display
    system_prompt_versions: Optional[List[Dict[str, Any]]] = None
    created_at: datetime
    updated_at: datetime
    version: int
    has_results: bool
