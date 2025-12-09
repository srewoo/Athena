"""
Data models for Athena - System Prompt Optimization & Evaluation Tool
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import uuid


# ============= Project Models =============

class Requirements(BaseModel):
    """User requirements for the system prompt"""
    use_case: str
    key_requirements: List[str]
    constraints: Optional[Dict[str, Any]] = None
    expected_behavior: Optional[str] = None
    target_provider: str  # "openai", "claude", "gemini", "multi"


class SystemPromptVersion(BaseModel):
    """A version of the system prompt with evaluation results"""
    version: int
    prompt_text: str
    evaluation: Optional[Dict[str, Any]] = None  # Evaluation results
    changes_from_previous: List[str] = []
    user_feedback: Optional[str] = None
    is_final: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvalPrompt(BaseModel):
    """Evaluation prompt to test the system prompt"""
    prompt_text: str
    version: int = 1
    rationale: str
    test_scenarios: List[str] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TestCase(BaseModel):
    """Individual test case (CSV row)"""
    input: str  # User message/query to test
    category: str  # "positive", "negative", "edge_case", "adversarial"
    test_focus: str  # Which requirement/behavior to test
    difficulty: str  # "easy", "medium", "hard"


class Dataset(BaseModel):
    """Test dataset for evaluation"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    csv_content: str  # Raw CSV string
    test_cases: List[TestCase] = []  # Parsed test cases
    sample_count: int = 100  # Default: 100 samples
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Project(BaseModel):
    """Main project containing the entire workflow"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    requirements: Requirements
    system_prompt_versions: List[SystemPromptVersion] = []
    eval_prompt: Optional[EvalPrompt] = None
    dataset: Optional[Dataset] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ============= Request/Response Models =============

class ProjectCreate(BaseModel):
    """Request to create a new project"""
    name: str
    use_case: str
    key_requirements: List[str]
    constraints: Optional[Dict[str, Any]] = None
    expected_behavior: Optional[str] = None
    target_provider: str
    initial_prompt: str


class ProjectUpdate(BaseModel):
    """Request to update project metadata"""
    name: Optional[str] = None
    requirements: Optional[Requirements] = None


class AnalyzeRequest(BaseModel):
    """Request to analyze a prompt against requirements"""
    prompt_text: str


class AnalyzeResponse(BaseModel):
    """Analysis results"""
    requirements_alignment_score: float  # 0-100
    requirements_gaps: List[str]
    best_practices_score: float  # 0-100
    best_practices_violations: List[Dict[str, str]]
    suggestions: List[Dict[str, str]]  # Prioritized suggestions
    overall_score: float


class RewriteRequest(BaseModel):
    """Request to rewrite a prompt"""
    current_prompt: str
    focus_areas: Optional[List[str]] = None  # Specific areas to focus on


class RewriteResponse(BaseModel):
    """Rewritten prompt"""
    improved_prompt: str
    changes_made: List[str]
    rationale: str


class AddVersionRequest(BaseModel):
    """Request to add a new prompt version"""
    prompt_text: str
    user_feedback: Optional[str] = None
    is_final: bool = False


class GenerateEvalPromptRequest(BaseModel):
    """Request to generate evaluation prompt"""
    include_scenarios: Optional[List[str]] = None


class GenerateEvalPromptResponse(BaseModel):
    """Generated evaluation prompt"""
    eval_prompt: str
    rationale: str
    test_scenarios: List[str]


class GenerateDatasetRequest(BaseModel):
    """Request to generate test dataset"""
    sample_count: int = 100
    distribution: Optional[Dict[str, int]] = None  # Custom distribution


class GenerateDatasetResponse(BaseModel):
    """Generated dataset"""
    dataset_id: str
    csv_content: str
    sample_count: int
    preview: List[TestCase]  # First 10 for preview


class ExportFormat(BaseModel):
    """Export format options"""
    format: str = "csv"  # Currently only CSV supported


class RefineEvalPromptRequest(BaseModel):
    """Request to refine eval prompt with user feedback"""
    current_eval_prompt: str
    user_feedback: str


class RefineEvalPromptResponse(BaseModel):
    """Refined evaluation prompt response"""
    refined_prompt: str
    changes_made: List[str]
    rationale: str


# ============= Iterative Refinement Models =============

class IterativeRewriteRequest(BaseModel):
    """Request for iterative prompt refinement with user feedback"""
    current_prompt: str
    user_feedback: str
    focus_areas: Optional[List[str]] = None
    iteration: int = 1  # Track which iteration this is


class IterativeRewriteResponse(BaseModel):
    """Response from iterative refinement"""
    improved_prompt: str
    changes_made: List[str]
    rationale: str
    iteration: int
    improvement_score: float  # How much improvement from previous
    suggestions_for_next: List[str]  # Suggestions for further improvement


class RefinementSession(BaseModel):
    """Track a refinement session with multiple iterations"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    initial_prompt: str
    iterations: List[Dict[str, Any]] = []  # History of all iterations
    current_prompt: str
    total_improvement: float = 0.0
    status: str = "active"  # "active", "completed", "abandoned"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ============= A/B Comparison Models =============

class VersionComparison(BaseModel):
    """Side-by-side comparison of two prompt versions"""
    version_a: int
    version_b: int
    prompt_a: str
    prompt_b: str
    scores_a: Dict[str, float]  # requirements_alignment, best_practices, overall
    scores_b: Dict[str, float]
    differences: List[Dict[str, str]]  # What changed between versions
    recommendation: str  # Which version is better and why
    detailed_analysis: Dict[str, Any]  # Detailed breakdown


class ABCompareRequest(BaseModel):
    """Request for A/B comparison"""
    version_a: int
    version_b: int


class ABCompareResponse(BaseModel):
    """A/B comparison results"""
    comparison: VersionComparison
    winner: str  # "version_a", "version_b", or "tie"
    confidence: float  # How confident the recommendation is (0-100)
    key_differences: List[str]


# ============= Few-Shot Examples Models =============

class EvalExample(BaseModel):
    """A calibration example for eval prompts"""
    input: str
    output: str
    expected_score: int  # 1-5
    reasoning: str
    category: str  # "excellent", "good", "adequate", "poor", "very_poor"


class GenerateEvalPromptWithExamplesRequest(BaseModel):
    """Request to generate eval prompt with few-shot examples"""
    include_scenarios: Optional[List[str]] = None
    include_few_shot_examples: bool = True
    num_examples: int = 3  # Number of examples per rating level


class GenerateEvalPromptWithExamplesResponse(BaseModel):
    """Generated evaluation prompt with calibration examples"""
    eval_prompt: str
    rationale: str
    test_scenarios: List[str]
    calibration_examples: List[EvalExample]
    generation_method: str


# ============= Test Run Models =============

class ExecutionResult(BaseModel):
    """Result of executing prompt on a single test input and evaluating it"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset_item_index: int  # Index in the dataset
    input_data: Dict[str, Any]  # The test input (from dataset)
    prompt_output: str  # LLM response when running the prompt
    eval_score: float  # 1-5 score from evaluation
    eval_feedback: str  # Detailed feedback from evaluation
    passed: bool  # Whether score meets threshold
    latency_ms: int  # Time to execute prompt
    tokens_used: Optional[int] = None  # Token count if available
    error: Optional[str] = None  # Error message if execution failed
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TestRunSummary(BaseModel):
    """Aggregated summary of a test run"""
    total_items: int
    completed_items: int
    passed_items: int
    failed_items: int
    error_items: int
    pass_rate: float  # Percentage
    avg_score: float
    min_score: float
    max_score: float
    score_distribution: Dict[str, int]  # {"1": 5, "2": 10, ...}
    avg_latency_ms: float
    total_tokens: int
    estimated_cost: float


class TestRun(BaseModel):
    """A test run executing dataset against a prompt version"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    prompt_version: int  # Which prompt version to test
    prompt_text: str  # Snapshot of prompt at run time
    eval_prompt_text: str  # Snapshot of eval prompt at run time
    llm_provider: str  # Provider for execution
    model_name: Optional[str] = None
    status: str = "pending"  # "pending", "running", "completed", "failed", "cancelled"
    pass_threshold: float = 3.5  # Score threshold for pass/fail
    batch_size: int = 5
    max_concurrent: int = 3
    results: List[ExecutionResult] = []
    summary: Optional[TestRunSummary] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TestRunRequest(BaseModel):
    """Request to start a test run"""
    prompt_version: int  # Which prompt version to test
    dataset_item_indices: Optional[List[int]] = None  # Specific items, or None for all
    llm_provider: str
    model_name: Optional[str] = None
    pass_threshold: float = 3.5
    batch_size: int = 5
    max_concurrent: int = 3


class TestRunStatusResponse(BaseModel):
    """Response for test run status check"""
    run_id: str
    status: str
    progress: float  # 0-100 percentage
    completed_items: int
    total_items: int
    current_batch: int
    total_batches: int
    partial_summary: Optional[TestRunSummary] = None
    recent_results: List[ExecutionResult] = []  # Last 5 results


class TestRunResultsResponse(BaseModel):
    """Full results of a test run"""
    test_run: TestRun
    summary: TestRunSummary
    results: List[ExecutionResult]


class SingleTestRequest(BaseModel):
    """Request to run a single test (for debugging)"""
    prompt_text: str
    test_input: Dict[str, Any]
    llm_provider: str
    model_name: Optional[str] = None
    eval_prompt_text: Optional[str] = None  # If not provided, skip evaluation


class SingleTestResponse(BaseModel):
    """Response from single test execution"""
    input_data: Dict[str, Any]
    prompt_output: str
    eval_score: Optional[float] = None
    eval_feedback: Optional[str] = None
    latency_ms: int
    tokens_used: Optional[int] = None


class TestRunComparisonRequest(BaseModel):
    """Request to compare two test runs"""
    run_id_a: str
    run_id_b: str


class TestRunComparisonResult(BaseModel):
    """Comparison of two test runs"""
    run_a: Dict[str, Any]  # Summary of run A
    run_b: Dict[str, Any]  # Summary of run B
    pass_rate_delta: float  # B - A
    avg_score_delta: float  # B - A
    improved_items: int  # Items that scored higher in B
    regressed_items: int  # Items that scored lower in B
    unchanged_items: int  # Items with same score
    item_comparisons: List[Dict[str, Any]]  # Per-item comparison


class RerunFailedRequest(BaseModel):
    """Request to re-run failed items from a test run"""
    source_run_id: str
    llm_provider: Optional[str] = None  # Override provider
    model_name: Optional[str] = None  # Override model
