"""
API Endpoints for Calibration Example Management and Bias Detection

Provides endpoints to:
- Get calibration examples from eval prompts
- Update calibration examples
- Run bias detection on eval prompts
- Get bias detection reports

Author: Athena Team
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import logging
import json
import re

from eval_bias_detector import (
    EvalBiasDetector,
    ComprehensiveBiasReport,
    bias_report_to_dict
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/calibration", tags=["calibration"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class CalibrationExample(BaseModel):
    """A single calibration example"""
    id: Optional[int] = None
    input: str
    output: str
    score: float = Field(ge=1, le=5)
    reasoning: str
    dimension_scores: Dict[str, Any] = {}
    is_edge_case: bool = False
    edge_case_type: str = ""


class UpdateCalibrationRequest(BaseModel):
    """Request to update calibration examples"""
    project_id: str
    calibration_examples: List[CalibrationExample]


class BiasDetectionRequest(BaseModel):
    """Request to run bias detection"""
    project_id: str
    eval_prompt: str
    test_cases: List[Dict[str, str]]  # [{"input": "...", "output": "..."}]
    baseline_scores: Optional[List[float]] = None


class BiasDetectionResponse(BaseModel):
    """Response from bias detection"""
    overall_bias_score: float
    is_biased: bool
    critical_biases: List[str]
    biases_detected: List[Dict[str, Any]]
    recommendations: List[str]
    test_metadata: Dict[str, Any]


# =============================================================================
# CALIBRATION EXAMPLE ENDPOINTS
# =============================================================================

@router.get("/examples/{project_id}")
async def get_calibration_examples(project_id: str):
    """
    Get calibration examples for a project's eval prompt.
    
    Extracts calibration examples from the eval prompt text.
    """
    try:
        import project_storage
        
        project = project_storage.load_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        eval_prompt = project.eval_prompt
        if not eval_prompt:
            return {
                "project_id": project_id,
                "calibration_examples": [],
                "message": "No eval prompt found"
            }
        
        # Extract calibration examples from eval prompt
        examples = extract_calibration_examples_from_prompt(eval_prompt)
        
        # Get from metadata if available
        if hasattr(project, 'eval_metadata') and project.eval_metadata:
            metadata_examples = project.eval_metadata.get('calibration_examples', [])
            if metadata_examples:
                examples = metadata_examples
        
        return {
            "project_id": project_id,
            "calibration_examples": examples,
            "count": len(examples)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get calibration examples: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get examples: {str(e)}")


@router.post("/examples/update")
async def update_calibration_examples(request: UpdateCalibrationRequest):
    """
    Update calibration examples for a project.
    
    Updates both the eval prompt text and metadata.
    """
    try:
        import project_storage
        
        project = project_storage.load_project(request.project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Store in metadata
        if not hasattr(project, 'eval_metadata') or not project.eval_metadata:
            project.eval_metadata = {}
        
        examples_data = [ex.dict() for ex in request.calibration_examples]
        project.eval_metadata['calibration_examples'] = examples_data
        
        # Update eval prompt with new examples
        if project.eval_prompt:
            updated_prompt = update_calibration_in_prompt(
                project.eval_prompt,
                examples_data
            )
            project.eval_prompt = updated_prompt
        
        # Save project
        project_storage.save_project(project)
        
        logger.info(f"Updated {len(examples_data)} calibration examples for project {request.project_id}")
        
        return {
            "success": True,
            "project_id": request.project_id,
            "examples_updated": len(examples_data),
            "message": "Calibration examples updated successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update calibration examples: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update examples: {str(e)}")


# =============================================================================
# BIAS DETECTION ENDPOINTS
# =============================================================================

@router.post("/bias/detect")
async def detect_bias(request: BiasDetectionRequest):
    """
    Run bias detection on an eval prompt.
    
    Tests for position bias, length bias, verbosity bias, and more.
    """
    try:
        from llm_client_v2 import get_llm_client
        from shared_settings import get_settings
        
        if len(request.test_cases) < 3:
            raise HTTPException(
                status_code=400,
                detail="Need at least 3 test cases for bias detection"
            )
        
        # Get LLM client
        llm_client = get_llm_client()
        settings = get_settings()
        
        # Create eval function
        async def run_eval(prompt, input_text, output_text):
            """Run evaluation and return score, verdict"""
            # Fill in prompt variables
            filled_prompt = prompt.replace('{{input}}', input_text)
            filled_prompt = filled_prompt.replace('{input}', input_text)
            filled_prompt = filled_prompt.replace('{{output}}', output_text)
            filled_prompt = filled_prompt.replace('{output}', output_text)
            
            result = await llm_client.chat(
                system_prompt=filled_prompt,
                user_message="Evaluate the output above and return your assessment as JSON.",
                provider=settings.get('provider', 'openai'),
                api_key=settings.get('api_key', ''),
                model_name=settings.get('model_name', 'gpt-4o-mini'),
                temperature=0.0,
                max_tokens=2000
            )
            
            if result.get('error'):
                raise Exception(result['error'])
            
            # Parse score from output
            output = result.get('output', '')
            
            # Try to parse JSON
            try:
                json_match = re.search(r'\{[\s\S]*\}', output)
                if json_match:
                    parsed = json.loads(json_match.group())
                    score = parsed.get('score', 3.0)
                    verdict = parsed.get('verdict', 'NEEDS_REVIEW')
                    return float(score), verdict
            except:
                pass
            
            # Fallback: try to find score in text
            score_match = re.search(r'"score":\s*([0-9.]+)', output)
            if score_match:
                return float(score_match.group(1)), 'UNKNOWN'
            
            return 3.0, 'UNKNOWN'
        
        # Run bias detection
        detector = EvalBiasDetector()
        report = await detector.detect_biases(
            eval_prompt=request.eval_prompt,
            run_eval_func=run_eval,
            test_cases=request.test_cases,
            baseline_scores=request.baseline_scores
        )
        
        # Convert to dict
        report_dict = bias_report_to_dict(report)
        
        logger.info(
            f"Bias detection for project {request.project_id}: "
            f"Score={report.overall_bias_score}, Biased={report.is_biased}"
        )
        
        return BiasDetectionResponse(**report_dict)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bias detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Bias detection failed: {str(e)}")


@router.get("/bias/report/{project_id}")
async def get_bias_report(project_id: str):
    """
    Get the latest bias detection report for a project.
    
    Returns cached report if available.
    """
    try:
        import project_storage
        
        project = project_storage.load_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Get from metadata
        if hasattr(project, 'eval_metadata') and project.eval_metadata:
            report = project.eval_metadata.get('bias_report')
            if report:
                return {
                    "project_id": project_id,
                    "report": report,
                    "cached": True
                }
        
        return {
            "project_id": project_id,
            "report": None,
            "message": "No bias report available. Run bias detection first."
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get bias report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get report: {str(e)}")


@router.post("/bias/save")
async def save_bias_report(project_id: str, report: Dict[str, Any]):
    """Save bias detection report to project metadata"""
    try:
        import project_storage
        
        project = project_storage.load_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if not hasattr(project, 'eval_metadata') or not project.eval_metadata:
            project.eval_metadata = {}
        
        project.eval_metadata['bias_report'] = report
        project_storage.save_project(project)
        
        return {"success": True, "message": "Bias report saved"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save bias report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_calibration_examples_from_prompt(eval_prompt: str) -> List[Dict[str, Any]]:
    """
    Extract calibration examples from eval prompt text.
    
    Looks for patterns like:
    - Example (Score 5/5):
    - Input: "..."
    - Output: "..."
    - Reasoning: "..."
    """
    examples = []
    
    # Pattern to match example blocks
    example_pattern = r'\*\*Example.*?Score\s+([0-9.]+).*?\*\*.*?[:-]\s*Input:?\s*["\']?(.*?)["\']?(?:\n|$).*?[:-]\s*Output:?\s*["\']?(.*?)["\']?(?:\n|$).*?[:-]\s*Reasoning:?\s*["\']?(.*?)["\']?(?:\n|$)'
    
    matches = re.finditer(example_pattern, eval_prompt, re.IGNORECASE | re.DOTALL)
    
    for i, match in enumerate(matches):
        try:
            score = float(match.group(1))
            input_text = match.group(2).strip()[:200]
            output_text = match.group(3).strip()[:300]
            reasoning = match.group(4).strip()[:300]
            
            # Check if edge case
            is_edge_case = 'edge case' in reasoning.lower() or '[EDGE CASE]' in match.group(0)
            
            examples.append({
                "id": i,
                "input": input_text,
                "output": output_text,
                "score": score,
                "reasoning": reasoning,
                "dimension_scores": {},
                "is_edge_case": is_edge_case,
                "edge_case_type": ""
            })
        except Exception as e:
            logger.warning(f"Failed to parse example {i}: {e}")
            continue
    
    return examples


def update_calibration_in_prompt(
    eval_prompt: str,
    new_examples: List[Dict[str, Any]]
) -> str:
    """
    Update calibration examples in eval prompt text.
    
    Replaces the calibration section with new examples.
    """
    # Find calibration section
    calibration_start = re.search(
        r'##\s*(?:VIII\.?|8\.?)\s*Calibration\s+Examples',
        eval_prompt,
        re.IGNORECASE
    )
    
    if not calibration_start:
        # Append calibration section
        logger.info("No calibration section found, appending new section")
        return eval_prompt + "\n\n" + format_calibration_section(new_examples)
    
    # Find next section or end
    calibration_end = re.search(
        r'\n##\s*(?:IX|9|X|10)',
        eval_prompt[calibration_start.end():],
        re.IGNORECASE
    )
    
    if calibration_end:
        # Replace section
        start_pos = calibration_start.start()
        end_pos = calibration_start.end() + calibration_end.start()
        
        new_prompt = (
            eval_prompt[:start_pos] +
            format_calibration_section(new_examples) +
            "\n" +
            eval_prompt[end_pos:]
        )
    else:
        # Replace to end of file
        start_pos = calibration_start.start()
        new_prompt = (
            eval_prompt[:start_pos] +
            format_calibration_section(new_examples)
        )
    
    return new_prompt


def format_calibration_section(examples: List[Dict[str, Any]]) -> str:
    """Format calibration examples into eval prompt text"""
    if not examples:
        return "## VIII. Calibration Examples\n\nNo calibration examples provided.\n"
    
    lines = [
        "## VIII. Calibration Examples (Including Edge Cases & Intermediate Scores)",
        "",
        "Use these to calibrate your scoring. Similar outputs should receive similar scores.",
        ""
    ]
    
    for ex in examples[:8]:  # Limit to 8 examples
        edge_marker = " [EDGE CASE]" if ex.get('is_edge_case') else ""
        
        lines.extend([
            f"**Example (Score {ex['score']}/5){edge_marker}:**",
            f"- Input: \"{ex['input'][:80]}...\"",
            f"- Output: \"{ex['output'][:120]}...\"",
            f"- Reasoning: {ex['reasoning'][:150]}",
            ""
        ])
    
    lines.extend([
        "**IMPORTANT - Use Intermediate Scores:**",
        "- **4.5**: Almost perfect, one trivial issue",
        "- **3.5**: Acceptable but notable gaps",
        "- **2.5**: Borderline, needs human review",
        "- **1.5**: Very poor but not completely wrong",
        "",
        "Do NOT round all scores to whole numbers. Use half-point increments when appropriate.",
        ""
    ])
    
    return "\n".join(lines)
