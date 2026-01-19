"""
Simple data models for the 5-step prompt testing workflow
"""
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Optional, Any
from datetime import datetime


# ============= Request Models =============

class StructuredRequirements(BaseModel):
    """Structured requirements for better prompt generation"""
    prd_document: Optional[str] = None  # Full PRD text for AI extraction
    must_do: Optional[List[str]] = None  # List of required behaviors (auto-extracted from PRD or manual)
    must_not_do: Optional[List[str]] = None  # List of forbidden behaviors (auto-extracted from PRD or manual)
    tone: Optional[str] = None  # e.g., "professional", "empathetic", "concise"
    output_format: Optional[str] = None  # Expected output structure
    constraints: Optional[List[str]] = None  # Additional constraints
    edge_cases: Optional[List[str]] = None  # Known edge cases to handle
    success_criteria: Optional[List[str]] = None  # What defines success
    prd_extracted: Optional[bool] = None  # Flag indicating if PRD was auto-extracted


class ProjectInput(BaseModel):
    """User input for the prompt testing workflow"""
    model_config = ConfigDict(protected_namespaces=())

    provider: str  # "openai", "claude", "gemini"
    api_key: str
    model_name: Optional[str] = None
    project_name: str
    use_case: str
    requirements: str
    structured_requirements: Optional[StructuredRequirements] = None  # New structured format
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
    metadata: Optional[Dict[str, Any]] = None  # For smart generation info (input_type, domain, etc.)


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

class CalibrationExample(BaseModel):
    """Few-shot calibration example for eval prompt"""
    id: Optional[str] = None
    input: str
    output: str
    score: float  # 1-5
    reasoning: str
    category: str = "general"  # e.g., "excellent", "acceptable", "poor"
    created_at: Optional[str] = None


class HumanValidation(BaseModel):
    """Human validation record for a test result"""
    id: Optional[str] = None
    run_id: Optional[str] = None
    result_id: str
    human_score: float  # 1-5
    human_feedback: str
    validator_id: Optional[str] = None
    validated_at: Optional[datetime] = None
    agrees_with_llm: Optional[bool] = None
    score_difference: Optional[float] = None


class ABTestConfig(BaseModel):
    """Configuration for A/B testing between prompt versions"""
    id: str
    name: str
    version_a: int  # Prompt version number (control)
    version_b: int  # Prompt version number (treatment)
    sample_size: int = 30  # Minimum samples per variant
    confidence_level: float = 0.95  # Statistical confidence level
    status: str = "running"  # running, completed, stopped
    created_at: datetime
    completed_at: Optional[datetime] = None


class ABTestResult(BaseModel):
    """Results from an A/B test"""
    test_id: str
    version_a_stats: Dict[str, Any]
    version_b_stats: Dict[str, Any]
    winner: Optional[str] = None  # "A", "B", or None if inconclusive
    p_value: float
    is_significant: bool
    confidence_interval: Dict[str, float]
    effect_size: float
    recommendation: str


class SavedProject(BaseModel):
    """Saved project with all workflow data"""
    id: str
    project_name: str
    use_case: str
    requirements: Any  # Can be string or object with use_case
    structured_requirements: Optional[StructuredRequirements] = None  # New structured format
    key_requirements: Optional[List[str]] = None
    initial_prompt: str
    project_type: Optional[str] = None  # "eval" for imported eval prompts
    optimized_prompt: Optional[str] = None
    optimization_score: Optional[float] = None
    eval_prompt: Optional[str] = None
    eval_rationale: Optional[str] = None
    eval_prompt_versions: Optional[List[Dict[str, Any]]] = None  # Eval prompt version history
    eval_metadata: Optional[Dict[str, Any]] = None  # Metadata for eval prompt (calibration examples, bias reports)
    calibration_examples: Optional[List[CalibrationExample]] = None  # Few-shot examples
    human_validations: Optional[List[HumanValidation]] = None  # Human validation records
    ab_tests: Optional[List[ABTestConfig]] = None  # A/B test configurations
    dataset: Optional[Dict[str, Any]] = None  # Generated test dataset
    test_cases: Optional[List[Dict[str, Any]]] = None
    test_results: Optional[List[Dict[str, Any]]] = None  # Flexible format for test results
    test_runs: Optional[List[Dict[str, Any]]] = None  # All test run history
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
    project_type: Optional[str] = None  # "eval" for imported eval prompts
    created_at: datetime
    updated_at: datetime
    version: int
    has_results: bool
