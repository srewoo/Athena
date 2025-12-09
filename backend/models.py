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
