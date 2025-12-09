"""
API endpoints for Athena project workflow
"""
from fastapi import APIRouter, HTTPException
from typing import List
from datetime import datetime, timezone
from bson import ObjectId
import json

from models import (
    Project, ProjectCreate, ProjectUpdate, Requirements, SystemPromptVersion,
    AnalyzeRequest, AnalyzeResponse, RewriteRequest, RewriteResponse,
    AddVersionRequest, GenerateEvalPromptRequest, GenerateEvalPromptResponse,
    GenerateDatasetRequest, GenerateDatasetResponse, EvalPrompt, Dataset, TestCase,
    RefineEvalPromptRequest, RefineEvalPromptResponse,
    IterativeRewriteRequest, IterativeRewriteResponse, RefinementSession,
    ABCompareRequest, ABCompareResponse, VersionComparison,
    GenerateEvalPromptWithExamplesRequest, GenerateEvalPromptWithExamplesResponse, EvalExample,
    TestRunRequest, TestRunStatusResponse, TestRunResultsResponse, TestRun, ExecutionResult,
    TestRunSummary, SingleTestRequest, SingleTestResponse,
    TestRunComparisonRequest, TestRunComparisonResult, RerunFailedRequest
)
from best_practices_engine import get_best_practices_engine
from requirements_analyzer import get_requirements_analyzer
from prompt_rewriter import get_prompt_rewriter
from eval_generator import get_eval_prompt_generator
from dataset_generator import get_dataset_generator
from prompt_executor import prompt_executor
import asyncio
import csv
import io


# Create router
router = APIRouter(prefix="/api/projects", tags=["projects"])

# MongoDB client (will be injected)
db = None


def init_db(database):
    """Initialize database connection"""
    global db
    db = database


def get_object_id(project_id: str):
    """Convert project_id string to ObjectId for MongoDB queries"""
    try:
        return ObjectId(project_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid project ID format")


# ============= Project CRUD =============

@router.post("", response_model=Project)
async def create_project(project_data: ProjectCreate):
    """Create a new project"""
    # Create requirements object
    requirements = Requirements(
        use_case=project_data.use_case,
        key_requirements=project_data.key_requirements,
        constraints=project_data.constraints,
        expected_behavior=project_data.expected_behavior,
        target_provider=project_data.target_provider
    )

    # Create initial system prompt version
    initial_version = SystemPromptVersion(
        version=1,
        prompt_text=project_data.initial_prompt,
        changes_from_previous=[],
        is_final=False
    )

    # Create project
    project = Project(
        name=project_data.name,
        requirements=requirements,
        system_prompt_versions=[initial_version]
    )

    # Save to database - exclude the UUID id field so MongoDB creates _id
    project_data_dict = project.dict(by_alias=True)
    del project_data_dict["id"]  # Remove UUID, let MongoDB create _id
    result = await db.projects.insert_one(project_data_dict)

    # Return created project with MongoDB _id as the id
    project_data_dict["id"] = str(result.inserted_id)

    return project_data_dict


@router.get("", response_model=List[Project])
async def list_projects(skip: int = 0, limit: int = 10):
    """List all projects"""
    cursor = db.projects.find().skip(skip).limit(limit).sort("created_at", -1)
    projects = await cursor.to_list(length=limit)

    # Convert ObjectId to string
    for p in projects:
        if "_id" in p:
            p["id"] = str(p["_id"])
            del p["_id"]

    return projects


@router.get("/{project_id}", response_model=Project)
async def get_project(project_id: str):
    """Get a specific project"""
    project = await db.projects.find_one({"_id": get_object_id(project_id)})

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Convert ObjectId to string for response
    project["id"] = str(project["_id"])
    del project["_id"]

    return project


@router.put("/{project_id}", response_model=Project)
async def update_project(project_id: str, updates: ProjectUpdate):
    """Update project metadata"""
    update_data = {k: v for k, v in updates.dict(exclude_unset=True).items() if v is not None}
    update_data["updated_at"] = datetime.now(timezone.utc)

    result = await db.projects.update_one(
        {"_id": get_object_id(project_id)},
        {"$set": update_data}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Project not found")

    return await get_project(project_id)


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    """Delete a project"""
    result = await db.projects.delete_one({"_id": get_object_id(project_id)})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Project not found")

    return {"message": "Project deleted successfully"}


# ============= Prompt Analysis =============

@router.post("/{project_id}/analyze", response_model=AnalyzeResponse)
async def analyze_prompt(project_id: str, request: AnalyzeRequest):
    """Analyze a prompt against requirements and best practices using LLM semantic analysis"""
    # Get project
    project = await db.projects.find_one({"_id": get_object_id(project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Parse project data
    requirements = Requirements(**project["requirements"])

    # Get LLM settings from database for semantic analysis
    settings_doc = await db.settings.find_one()  # Get first settings doc (no _id filter)
    llm_provider = settings_doc.get("llm_provider", "openai") if settings_doc else "openai"
    api_key = settings_doc.get("api_key") if settings_doc else None
    model_name = settings_doc.get("model_name") if settings_doc else None

    # Debug logging for LLM settings
    print(f"[ANALYZE] Settings doc found: {settings_doc is not None}")
    print(f"[ANALYZE] Provider: {llm_provider}")
    print(f"[ANALYZE] API Key present: {bool(api_key)}")
    if not api_key:
        print("[ANALYZE] WARNING: No API key found - using keyword-based fallback (fast but less accurate)")

    # Get analyzers
    req_analyzer = get_requirements_analyzer()
    bp_engine = get_best_practices_engine()

    # Analyze requirements alignment using LLM semantic analysis
    req_analysis = await req_analyzer.analyze_alignment(
        request.prompt_text,
        requirements.use_case,
        requirements.key_requirements,
        requirements.expected_behavior,
        requirements.constraints,
        provider=llm_provider,
        api_key=api_key,
        model_name=model_name
    )

    # Analyze best practices
    bp_analysis = bp_engine.evaluate_prompt(
        request.prompt_text,
        requirements.target_provider
    )

    # Combine analyses
    all_violations = bp_analysis["violations"]
    all_suggestions = req_analysis["suggestions"] + [
        {"type": "best_practice", "priority": v["importance"], "suggestion": f"{v['name']}: {v['description']}"}
        for v in bp_analysis["violations"]
    ]

    # Sanitize suggestions - ensure all values are strings (not None or int)
    sanitized_suggestions = []
    for s in all_suggestions:
        sanitized = {}
        for key, value in s.items():
            if value is None:
                sanitized[key] = ""
            elif isinstance(value, int):
                sanitized[key] = str(value)
            else:
                sanitized[key] = str(value)
        sanitized_suggestions.append(sanitized)

    # Sort suggestions by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    sanitized_suggestions.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 2))

    # Calculate overall score (weighted average)
    overall_score = (
        req_analysis["alignment_score"] * 0.6 +
        bp_analysis["score"] * 0.4
    )

    return AnalyzeResponse(
        requirements_alignment_score=req_analysis["alignment_score"],
        requirements_gaps=req_analysis["gaps"],
        best_practices_score=bp_analysis["score"],
        best_practices_violations=all_violations,
        suggestions=sanitized_suggestions,
        overall_score=round(overall_score, 2)
    )


# ============= Prompt Rewriting =============

@router.post("/{project_id}/rewrite", response_model=RewriteResponse)
async def rewrite_prompt(project_id: str, request: RewriteRequest):
    """Rewrite a prompt using AI with LLM semantic analysis"""
    # Get project
    project = await db.projects.find_one({"_id": get_object_id(project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    requirements = Requirements(**project["requirements"])

    # Get LLM settings from database for semantic analysis
    settings_doc = await db.settings.find_one()
    llm_provider = settings_doc.get("llm_provider", "openai") if settings_doc else "openai"
    api_key = settings_doc.get("api_key") if settings_doc else None
    model_name = settings_doc.get("model_name") if settings_doc else None

    # Get analyzers
    req_analyzer = get_requirements_analyzer()
    bp_engine = get_best_practices_engine()

    # Analyze current prompt using LLM semantic analysis
    req_analysis = await req_analyzer.analyze_alignment(
        request.current_prompt,
        requirements.use_case,
        requirements.key_requirements,
        requirements.expected_behavior,
        requirements.constraints,
        provider=llm_provider,
        api_key=api_key,
        model_name=model_name
    )

    bp_analysis = bp_engine.evaluate_prompt(
        request.current_prompt,
        requirements.target_provider
    )

    # Rewrite prompt
    rewriter = get_prompt_rewriter()
    result = await rewriter.rewrite_prompt(
        request.current_prompt,
        requirements,
        req_analysis["gaps"],
        bp_analysis["violations"],
        request.focus_areas,
        llm_provider,
        api_key,
        model_name
    )

    return RewriteResponse(**result)


# ============= Version Management =============

@router.post("/{project_id}/versions", response_model=SystemPromptVersion)
async def add_version(project_id: str, request: AddVersionRequest):
    """Add a new prompt version with LLM semantic analysis"""
    # Get project
    project = await db.projects.find_one({"_id": get_object_id(project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get current version number
    versions = project.get("system_prompt_versions", [])
    next_version = len(versions) + 1

    # Get LLM settings from database for semantic analysis
    settings_doc = await db.settings.find_one()
    llm_provider = settings_doc.get("llm_provider", "openai") if settings_doc else "openai"
    api_key = settings_doc.get("api_key") if settings_doc else None
    model_name = settings_doc.get("model_name") if settings_doc else None

    # Analyze the new version
    requirements = Requirements(**project["requirements"])
    req_analyzer = get_requirements_analyzer()
    bp_engine = get_best_practices_engine()

    req_analysis = await req_analyzer.analyze_alignment(
        request.prompt_text,
        requirements.use_case,
        requirements.key_requirements,
        requirements.expected_behavior,
        requirements.constraints,
        provider=llm_provider,
        api_key=api_key,
        model_name=model_name
    )

    bp_analysis = bp_engine.evaluate_prompt(
        request.prompt_text,
        requirements.target_provider
    )

    # Combine evaluation results
    evaluation = {
        "requirements_alignment": req_analysis["alignment_score"],
        "best_practices_score": bp_analysis["score"],
        "gaps": req_analysis["gaps"],
        "violations": bp_analysis["violations"]
    }

    # Determine changes from previous version
    changes = []
    if versions:
        prev_prompt = versions[-1]["prompt_text"]
        if len(request.prompt_text) > len(prev_prompt):
            changes.append("Expanded prompt with additional content")
        prev_eval = versions[-1].get("evaluation") or {}
        if req_analysis["alignment_score"] > prev_eval.get("requirements_alignment", 0):
            changes.append("Improved requirements alignment")

    # Create new version
    new_version = SystemPromptVersion(
        version=next_version,
        prompt_text=request.prompt_text,
        evaluation=evaluation,
        changes_from_previous=changes,
        user_feedback=request.user_feedback,
        is_final=request.is_final
    )

    # Update project
    await db.projects.update_one(
        {"_id": get_object_id(project_id)},
        {
            "$push": {"system_prompt_versions": new_version.dict()},
            "$set": {"updated_at": datetime.now(timezone.utc)}
        }
    )

    return new_version


@router.put("/{project_id}/versions/{version}/finalize")
async def finalize_version(project_id: str, version: int):
    """Mark a version as final"""
    result = await db.projects.update_one(
        {"_id": get_object_id(project_id), "system_prompt_versions.version": version},
        {
            "$set": {
                "system_prompt_versions.$.is_final": True,
                "updated_at": datetime.now(timezone.utc)
            }
        }
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Project or version not found")

    return {"message": f"Version {version} marked as final"}


@router.delete("/{project_id}/versions/{version}")
async def delete_version(project_id: str, version: int):
    """Delete a prompt version"""
    # Get project to check if version exists and if it's the only version
    project = await db.projects.find_one({"_id": get_object_id(project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    versions = project.get("system_prompt_versions", [])

    # Don't allow deleting the only version
    if len(versions) <= 1:
        raise HTTPException(status_code=400, detail="Cannot delete the only version")

    # Find the version to delete
    version_exists = any(v.get("version") == version for v in versions)
    if not version_exists:
        raise HTTPException(status_code=404, detail="Version not found")

    # Remove the version
    result = await db.projects.update_one(
        {"_id": get_object_id(project_id)},
        {
            "$pull": {"system_prompt_versions": {"version": version}},
            "$set": {"updated_at": datetime.now(timezone.utc)}
        }
    )

    if result.modified_count == 0:
        raise HTTPException(status_code=500, detail="Failed to delete version")

    return {"message": f"Version {version} deleted successfully"}


# ============= Eval Prompt Generation =============

@router.post("/{project_id}/eval-prompt/generate", response_model=GenerateEvalPromptResponse)
async def generate_eval_prompt(project_id: str, request: GenerateEvalPromptRequest):
    """Generate an evaluation prompt using LLM"""
    # Get project
    project = await db.projects.find_one({"_id": get_object_id(project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get final system prompt (or latest)
    versions = project.get("system_prompt_versions", [])
    if not versions:
        raise HTTPException(status_code=400, detail="No system prompt versions found")

    # Find final version or use latest
    final_version = None
    for v in reversed(versions):
        if v.get("is_final"):
            final_version = v
            break

    if not final_version:
        final_version = versions[-1]

    # Get LLM settings from database
    settings_doc = await db.settings.find_one()
    llm_provider = settings_doc.get("llm_provider", "openai") if settings_doc else "openai"
    api_key = settings_doc.get("api_key") if settings_doc else None
    model_name = settings_doc.get("model_name") if settings_doc else None

    # Generate eval prompt using LLM
    requirements = Requirements(**project["requirements"])
    generator = get_eval_prompt_generator()

    result = await generator.generate_eval_prompt(
        final_version["prompt_text"],
        requirements,
        request.include_scenarios,
        provider=llm_provider,
        api_key=api_key,
        model_name=model_name
    )

    # Save eval prompt to project
    eval_prompt = EvalPrompt(
        prompt_text=result["eval_prompt"],
        rationale=result["rationale"],
        test_scenarios=result["test_scenarios"]
    )

    await db.projects.update_one(
        {"_id": get_object_id(project_id)},
        {
            "$set": {
                "eval_prompt": eval_prompt.dict(),
                "updated_at": datetime.now(timezone.utc)
            }
        }
    )

    return GenerateEvalPromptResponse(**result)


@router.put("/{project_id}/eval-prompt")
async def update_eval_prompt(project_id: str, eval_prompt: EvalPrompt):
    """Update the evaluation prompt"""
    result = await db.projects.update_one(
        {"_id": get_object_id(project_id)},
        {
            "$set": {
                "eval_prompt": eval_prompt.dict(),
                "updated_at": datetime.now(timezone.utc)
            }
        }
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Project not found")

    return {"message": "Eval prompt updated successfully"}


@router.post("/{project_id}/eval-prompt/refine", response_model=RefineEvalPromptResponse)
async def refine_eval_prompt(project_id: str, request: RefineEvalPromptRequest):
    """Refine the evaluation prompt based on user feedback using LLM"""
    # Get project to verify it exists
    project = await db.projects.find_one({"_id": get_object_id(project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get LLM settings from database
    settings = await db.settings.find_one()
    if not settings:
        raise HTTPException(status_code=400, detail="Please configure LLM settings first")

    # Get the eval generator
    generator = get_eval_prompt_generator()

    # Refine the prompt with user feedback
    result = await generator.refine_eval_prompt(
        request.current_eval_prompt,
        request.user_feedback,
        provider=settings.get("llm_provider", "openai"),
        api_key=settings.get("api_key"),
        model_name=settings.get("model_name")
    )

    # If refinement was successful, update the project's eval prompt
    if result.get("refined_prompt"):
        current_eval = project.get("eval_prompt", {})
        current_version = current_eval.get("version", 1) if current_eval else 1

        updated_eval_prompt = EvalPrompt(
            prompt_text=result["refined_prompt"],
            version=current_version + 1,
            rationale=result.get("rationale", "Refined based on user feedback"),
            test_scenarios=current_eval.get("test_scenarios", []) if current_eval else []
        )

        await db.projects.update_one(
            {"_id": get_object_id(project_id)},
            {
                "$set": {
                    "eval_prompt": updated_eval_prompt.dict(),
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )

    return RefineEvalPromptResponse(**result)


# ============= Dataset Generation =============

@router.post("/{project_id}/dataset/generate", response_model=GenerateDatasetResponse)
async def generate_dataset(project_id: str, request: GenerateDatasetRequest):
    """Generate a test dataset"""
    # Get project
    project = await db.projects.find_one({"_id": get_object_id(project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get final system prompt
    versions = project.get("system_prompt_versions", [])
    if not versions:
        raise HTTPException(status_code=400, detail="No system prompt versions found")

    final_version = None
    for v in reversed(versions):
        if v.get("is_final"):
            final_version = v
            break

    if not final_version:
        final_version = versions[-1]

    # Get eval prompt if exists
    eval_prompt_text = project.get("eval_prompt", {}).get("prompt_text", "")

    # Generate dataset
    requirements = Requirements(**project["requirements"])
    generator = get_dataset_generator()

    # Get LLM settings from database
    settings_doc = await db.settings.find_one()

    result = generator.generate_dataset(
        final_version["prompt_text"],
        requirements,
        eval_prompt_text,
        request.sample_count,
        request.distribution,
        provider=settings_doc.get("llm_provider", "openai") if settings_doc else "openai",
        api_key=settings_doc.get("api_key") if settings_doc else None,
        model_name=settings_doc.get("model_name") if settings_doc else None,
        use_llm=True
    )

    # Create dataset object
    dataset = Dataset(
        project_id=project_id,
        csv_content=result["csv_content"],
        test_cases=result["test_cases"],
        sample_count=result["sample_count"],
        metadata=result["metadata"]
    )

    # Save to project
    await db.projects.update_one(
        {"_id": get_object_id(project_id)},
        {
            "$set": {
                "dataset": dataset.dict(),
                "updated_at": datetime.now(timezone.utc)
            }
        }
    )

    # Return preview (first 10 cases)
    return GenerateDatasetResponse(
        dataset_id=dataset.id,
        csv_content=result["csv_content"],
        sample_count=result["sample_count"],
        preview=result["test_cases"][:10]
    )


@router.post("/{project_id}/dataset/generate-stream")
async def generate_dataset_stream(project_id: str, request: GenerateDatasetRequest):
    """
    Generate dataset with Server-Sent Events for progress updates and heartbeat.
    Prevents timeout by sending heartbeat every 25 seconds.
    """
    from fastapi.responses import StreamingResponse
    import asyncio

    async def event_generator():
        """Generate SSE events with progress and heartbeat"""
        try:
            # Get project
            project = await db.projects.find_one({"_id": get_object_id(project_id)})
            if not project:
                yield f"data: {json.dumps({'error': 'Project not found'})}\n\n"
                return

            # Get final system prompt
            versions = project.get("system_prompt_versions", [])
            if not versions:
                yield f"data: {json.dumps({'error': 'No system prompt versions found'})}\n\n"
                return

            final_version = None
            for v in reversed(versions):
                if v.get("is_final"):
                    final_version = v
                    break
            if not final_version:
                final_version = versions[-1]

            eval_prompt = project.get("eval_prompt")
            eval_prompt_text = eval_prompt.get("prompt_text") if eval_prompt else None

            requirements = Requirements(**project["requirements"])
            generator = get_dataset_generator()

            # Get LLM settings
            settings_doc = await db.settings.find_one()
            provider = settings_doc.get("llm_provider", "openai") if settings_doc else "openai"
            api_key = settings_doc.get("api_key") if settings_doc else None
            model_name = settings_doc.get("model_name") if settings_doc else None

            # Send initial status
            yield f"data: {json.dumps({'status': 'starting', 'message': 'Starting dataset generation...'})}\n\n"

            # Create heartbeat task
            heartbeat_interval = 25  # seconds
            last_heartbeat = asyncio.get_event_loop().time()

            # Progress tracking
            progress_data = {"status": "generating", "progress": 0, "batch": 0, "total_batches": 0}

            def progress_callback(data):
                nonlocal progress_data
                progress_data = data

            # Run generation in background with progress updates
            generation_task = asyncio.create_task(
                generator.generate_dataset_async(
                    final_version["prompt_text"],
                    requirements,
                    eval_prompt_text,
                    request.sample_count,
                    request.distribution,
                    provider=provider,
                    api_key=api_key,
                    model_name=model_name,
                    progress_callback=progress_callback
                )
            )

            # Poll for progress and send heartbeats
            while not generation_task.done():
                current_time = asyncio.get_event_loop().time()

                # Send progress update
                yield f"data: {json.dumps(progress_data)}\n\n"

                # Send heartbeat if needed
                if current_time - last_heartbeat >= heartbeat_interval:
                    yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': current_time})}\n\n"
                    last_heartbeat = current_time

                await asyncio.sleep(2)  # Check every 2 seconds

            # Get result
            result = await generation_task

            # Create and save dataset
            dataset = Dataset(
                project_id=project_id,
                csv_content=result["csv_content"],
                test_cases=result["test_cases"],
                sample_count=result["sample_count"],
                metadata=result["metadata"]
            )

            await db.projects.update_one(
                {"_id": get_object_id(project_id)},
                {
                    "$set": {
                        "dataset": dataset.dict(),
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )

            # Send completion with data (including csv_content for test run)
            yield f"data: {json.dumps({'status': 'completed', 'sample_count': result['sample_count'], 'csv_content': result['csv_content'], 'preview': [tc.dict() for tc in result['test_cases'][:10]], 'dataset_id': dataset.id})}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable buffering for nginx
        }
    )


@router.get("/{project_id}/dataset")
async def get_dataset(project_id: str):
    """Get the dataset from database"""
    project = await db.projects.find_one({"_id": get_object_id(project_id)})

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    dataset = project.get("dataset")
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return dataset


@router.get("/{project_id}/dataset/export")
async def export_dataset(project_id: str):
    """Export dataset as CSV file"""
    from fastapi.responses import Response

    project = await db.projects.find_one({"_id": get_object_id(project_id)})

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    dataset = project.get("dataset")
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    csv_content = dataset.get("csv_content", "")

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=dataset_{project_id}.csv"}
    )


# ============= Iterative Refinement =============

@router.post("/{project_id}/iterative-rewrite", response_model=IterativeRewriteResponse)
async def iterative_rewrite(project_id: str, request: IterativeRewriteRequest):
    """
    Perform iterative refinement on a prompt based on user feedback.
    Tracks iteration history and provides improvement metrics.
    """
    # Get project
    project = await db.projects.find_one({"_id": get_object_id(project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    requirements = Requirements(**project["requirements"])

    # Get LLM settings
    settings_doc = await db.settings.find_one()
    llm_provider = settings_doc.get("llm_provider", "openai") if settings_doc else "openai"
    api_key = settings_doc.get("api_key") if settings_doc else None
    model_name = settings_doc.get("model_name") if settings_doc else None

    # Get or create refinement session
    session = project.get("refinement_session")
    previous_iterations = []
    if session:
        previous_iterations = session.get("iterations", [])

    # Perform iterative rewrite
    rewriter = get_prompt_rewriter()
    result = await rewriter.iterative_rewrite(
        current_prompt=request.current_prompt,
        user_feedback=request.user_feedback,
        requirements=requirements,
        previous_iterations=previous_iterations,
        iteration=request.iteration,
        provider=llm_provider,
        api_key=api_key,
        model_name=model_name
    )

    # Store iteration in session
    iteration_record = {
        "iteration": request.iteration,
        "feedback": request.user_feedback,
        "changes_made": result.get("changes_made", []),
        "improvement_score": result.get("improvement_score", 0),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    await db.projects.update_one(
        {"_id": get_object_id(project_id)},
        {
            "$push": {"refinement_session.iterations": iteration_record},
            "$set": {
                "refinement_session.current_prompt": result.get("improved_prompt", ""),
                "refinement_session.project_id": project_id,
                "refinement_session.status": "active",
                "updated_at": datetime.now(timezone.utc)
            },
            "$inc": {"refinement_session.total_improvement": result.get("improvement_score", 0)}
        }
    )

    return IterativeRewriteResponse(
        improved_prompt=result.get("improved_prompt", ""),
        changes_made=result.get("changes_made", []),
        rationale=result.get("rationale", ""),
        iteration=request.iteration,
        improvement_score=result.get("improvement_score", 0),
        suggestions_for_next=result.get("suggestions_for_next", [])
    )


@router.get("/{project_id}/refinement-session")
async def get_refinement_session(project_id: str):
    """Get the current refinement session with all iterations"""
    project = await db.projects.find_one({"_id": get_object_id(project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    session = project.get("refinement_session")
    if not session:
        return {"message": "No refinement session found", "session": None}

    return {"session": session}


@router.delete("/{project_id}/refinement-session")
async def clear_refinement_session(project_id: str):
    """Clear the refinement session to start fresh"""
    result = await db.projects.update_one(
        {"_id": get_object_id(project_id)},
        {"$unset": {"refinement_session": ""}}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Project not found")

    return {"message": "Refinement session cleared"}


# ============= A/B Comparison =============

@router.post("/{project_id}/compare", response_model=ABCompareResponse)
async def compare_versions(project_id: str, request: ABCompareRequest):
    """
    Compare two prompt versions side-by-side with detailed analysis.
    Returns scores, key differences, and a recommendation.
    """
    # Get project
    project = await db.projects.find_one({"_id": get_object_id(project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    versions = project.get("system_prompt_versions", [])

    # Find the two versions
    version_a = None
    version_b = None
    for v in versions:
        if v.get("version") == request.version_a:
            version_a = v
        if v.get("version") == request.version_b:
            version_b = v

    if not version_a:
        raise HTTPException(status_code=404, detail=f"Version {request.version_a} not found")
    if not version_b:
        raise HTTPException(status_code=404, detail=f"Version {request.version_b} not found")

    requirements = Requirements(**project["requirements"])

    # Get LLM settings
    settings_doc = await db.settings.find_one()
    llm_provider = settings_doc.get("llm_provider", "openai") if settings_doc else "openai"
    api_key = settings_doc.get("api_key") if settings_doc else None
    model_name = settings_doc.get("model_name") if settings_doc else None

    # Perform comparison
    rewriter = get_prompt_rewriter()
    result = await rewriter.compare_versions(
        prompt_a=version_a["prompt_text"],
        prompt_b=version_b["prompt_text"],
        requirements=requirements,
        provider=llm_provider,
        api_key=api_key,
        model_name=model_name
    )

    # Build comparison object
    comparison = VersionComparison(
        version_a=request.version_a,
        version_b=request.version_b,
        prompt_a=version_a["prompt_text"],
        prompt_b=version_b["prompt_text"],
        scores_a=result.get("scores_a", {}),
        scores_b=result.get("scores_b", {}),
        differences=[{"description": d} for d in result.get("key_differences", [])],
        recommendation=result.get("detailed_analysis", ""),
        detailed_analysis=result
    )

    return ABCompareResponse(
        comparison=comparison,
        winner=result.get("winner", "tie"),
        confidence=result.get("confidence", 50),
        key_differences=result.get("key_differences", [])
    )


# ============= Eval Prompt with Examples =============

@router.post("/{project_id}/eval-prompt/generate-with-examples", response_model=GenerateEvalPromptWithExamplesResponse)
async def generate_eval_prompt_with_examples(project_id: str, request: GenerateEvalPromptWithExamplesRequest):
    """
    Generate an evaluation prompt WITH few-shot calibration examples.
    Includes gold-standard examples for each rating level (1-5).
    """
    # Get project
    project = await db.projects.find_one({"_id": get_object_id(project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get final system prompt
    versions = project.get("system_prompt_versions", [])
    if not versions:
        raise HTTPException(status_code=400, detail="No system prompt versions found")

    final_version = None
    for v in reversed(versions):
        if v.get("is_final"):
            final_version = v
            break
    if not final_version:
        final_version = versions[-1]

    # Get LLM settings
    settings_doc = await db.settings.find_one()
    llm_provider = settings_doc.get("llm_provider", "openai") if settings_doc else "openai"
    api_key = settings_doc.get("api_key") if settings_doc else None
    model_name = settings_doc.get("model_name") if settings_doc else None

    # Generate eval prompt with examples
    requirements = Requirements(**project["requirements"])
    generator = get_eval_prompt_generator()

    result = await generator.generate_eval_prompt_with_examples(
        system_prompt=final_version["prompt_text"],
        requirements=requirements,
        additional_scenarios=request.include_scenarios,
        provider=llm_provider,
        api_key=api_key,
        model_name=model_name,
        num_examples=request.num_examples
    )

    # Convert calibration examples to EvalExample objects
    calibration_examples = [
        EvalExample(
            input=ex["input"],
            output=ex["output"],
            expected_score=ex["expected_score"],
            reasoning=ex["reasoning"],
            category=ex["category"]
        )
        for ex in result.get("calibration_examples", [])
    ]

    # Save to project
    eval_prompt = EvalPrompt(
        prompt_text=result["eval_prompt"],
        rationale=result["rationale"],
        test_scenarios=result["test_scenarios"]
    )

    await db.projects.update_one(
        {"_id": get_object_id(project_id)},
        {
            "$set": {
                "eval_prompt": eval_prompt.dict(),
                "eval_prompt_calibration_examples": [ex.dict() for ex in calibration_examples],
                "updated_at": datetime.now(timezone.utc)
            }
        }
    )

    return GenerateEvalPromptWithExamplesResponse(
        eval_prompt=result["eval_prompt"],
        rationale=result["rationale"],
        test_scenarios=result["test_scenarios"],
        calibration_examples=calibration_examples,
        generation_method=result.get("generation_method", "llm_with_examples")
    )


# ============= Test Run Endpoints =============

# In-memory storage for active test runs (for progress tracking)
active_test_runs: dict = {}


@router.post("/{project_id}/test-runs", response_model=dict)
async def start_test_run(project_id: str, request: TestRunRequest):
    """
    Start a test run to execute the prompt against the dataset and evaluate outputs.
    Returns run_id which can be used to poll for status.
    """
    # Get project
    project = await db.projects.find_one({"_id": get_object_id(project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get the prompt version
    versions = project.get("system_prompt_versions", [])
    prompt_version = None
    for v in versions:
        if v.get("version") == request.prompt_version:
            prompt_version = v
            break

    if not prompt_version:
        raise HTTPException(status_code=404, detail=f"Prompt version {request.prompt_version} not found")

    # Get eval prompt
    eval_prompt = project.get("eval_prompt")
    if not eval_prompt:
        raise HTTPException(status_code=400, detail="No eval prompt found. Please generate an eval prompt first.")

    # Get dataset
    dataset = project.get("dataset")
    if not dataset:
        raise HTTPException(status_code=400, detail="No dataset found. Please generate a dataset first.")

    # Parse CSV to get test items
    csv_content = dataset.get("csv_content", "")
    test_items = parse_csv_to_test_items(csv_content)
    if not test_items:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    # Filter items if specific indices provided
    if request.dataset_item_indices:
        test_items = [test_items[i] for i in request.dataset_item_indices if i < len(test_items)]

    # Get API key from settings
    settings_doc = await db.settings.find_one()
    api_key = settings_doc.get("api_key") if settings_doc else None
    if not api_key:
        raise HTTPException(status_code=400, detail="API key not configured in settings")

    # Create test run
    test_run = TestRun(
        project_id=project_id,
        prompt_version=request.prompt_version,
        prompt_text=prompt_version["prompt_text"],
        eval_prompt_text=eval_prompt["prompt_text"],
        llm_provider=request.llm_provider,
        model_name=request.model_name,
        pass_threshold=request.pass_threshold,
        batch_size=request.batch_size,
        max_concurrent=request.max_concurrent
    )

    # Store in active runs
    active_test_runs[test_run.id] = {
        "test_run": test_run,
        "test_items": test_items,
        "total_items": len(test_items)
    }

    # Save initial test run to database
    test_run_dict = test_run.dict()
    test_run_dict["_id"] = test_run.id
    await db.test_runs.insert_one(test_run_dict)

    # Start execution in background
    asyncio.create_task(
        execute_test_run_background(test_run.id, test_run, test_items, api_key)
    )

    return {
        "run_id": test_run.id,
        "status": "started",
        "total_items": len(test_items),
        "estimated_batches": (len(test_items) + request.batch_size - 1) // request.batch_size
    }


def parse_csv_to_test_items(csv_content: str) -> List[dict]:
    """Parse CSV content into a list of test item dictionaries."""
    test_items = []
    try:
        reader = csv.DictReader(io.StringIO(csv_content))
        for row in reader:
            test_items.append(dict(row))
    except Exception:
        pass
    return test_items


async def execute_test_run_background(run_id: str, test_run: TestRun, test_items: List[dict], api_key: str):
    """Background task to execute the test run."""
    try:
        # Progress callback to update database
        async def update_progress(updated_run: TestRun):
            # Update in-memory store
            if run_id in active_test_runs:
                active_test_runs[run_id]["test_run"] = updated_run

            # Update database
            await db.test_runs.update_one(
                {"_id": run_id},
                {"$set": {
                    "status": updated_run.status,
                    "results": [r.dict() for r in updated_run.results],
                    "summary": updated_run.summary.dict() if updated_run.summary else None,
                    "started_at": updated_run.started_at,
                    "completed_at": updated_run.completed_at,
                    "error_message": updated_run.error_message
                }}
            )

        # Sync callback wrapper
        def progress_callback(updated_run: TestRun):
            asyncio.create_task(update_progress(updated_run))

        # Execute the test run
        result = await prompt_executor.run_full_test(
            test_run=test_run,
            test_items=test_items,
            api_key=api_key,
            progress_callback=progress_callback
        )

        # Final update
        await update_progress(result)

    except Exception as e:
        # Update with error
        test_run.status = "failed"
        test_run.error_message = str(e)
        await db.test_runs.update_one(
            {"_id": run_id},
            {"$set": {
                "status": "failed",
                "error_message": str(e)
            }}
        )
    finally:
        # Clean up active runs
        if run_id in active_test_runs:
            del active_test_runs[run_id]


@router.get("/{project_id}/test-runs/{run_id}/status", response_model=TestRunStatusResponse)
async def get_test_run_status(project_id: str, run_id: str):
    """
    Get the current status and progress of a test run.
    Poll this endpoint during test execution.
    """
    # Check in-memory first (for active runs)
    if run_id in active_test_runs:
        data = active_test_runs[run_id]
        test_run = data["test_run"]
        total_items = data["total_items"]

        completed = len(test_run.results)
        progress = (completed / total_items * 100) if total_items > 0 else 0
        current_batch = (completed // test_run.batch_size) + 1
        total_batches = (total_items + test_run.batch_size - 1) // test_run.batch_size

        return TestRunStatusResponse(
            run_id=run_id,
            status=test_run.status,
            progress=round(progress, 1),
            completed_items=completed,
            total_items=total_items,
            current_batch=current_batch,
            total_batches=total_batches,
            partial_summary=test_run.summary,
            recent_results=test_run.results[-5:] if test_run.results else []
        )

    # Fall back to database
    test_run_doc = await db.test_runs.find_one({"_id": run_id})
    if not test_run_doc:
        raise HTTPException(status_code=404, detail="Test run not found")

    results = test_run_doc.get("results", [])
    total_items = len(results)  # This is approximate for completed runs

    # If we have the dataset, get actual total
    project = await db.projects.find_one({"_id": get_object_id(project_id)})
    if project and project.get("dataset"):
        csv_content = project["dataset"].get("csv_content", "")
        test_items = parse_csv_to_test_items(csv_content)
        total_items = len(test_items)

    completed = len(results)
    batch_size = test_run_doc.get("batch_size", 5)
    progress = (completed / total_items * 100) if total_items > 0 else 100
    current_batch = (completed // batch_size) + 1
    total_batches = (total_items + batch_size - 1) // batch_size

    summary = None
    if test_run_doc.get("summary"):
        summary = TestRunSummary(**test_run_doc["summary"])

    recent_results = []
    if results:
        recent_results = [ExecutionResult(**r) for r in results[-5:]]

    return TestRunStatusResponse(
        run_id=run_id,
        status=test_run_doc.get("status", "unknown"),
        progress=round(progress, 1),
        completed_items=completed,
        total_items=total_items,
        current_batch=current_batch,
        total_batches=total_batches,
        partial_summary=summary,
        recent_results=recent_results
    )


@router.get("/{project_id}/test-runs/{run_id}/results", response_model=TestRunResultsResponse)
async def get_test_run_results(project_id: str, run_id: str):
    """
    Get the full results of a completed test run.
    """
    test_run_doc = await db.test_runs.find_one({"_id": run_id})
    if not test_run_doc:
        raise HTTPException(status_code=404, detail="Test run not found")

    # Convert to models
    results = [ExecutionResult(**r) for r in test_run_doc.get("results", [])]
    summary = TestRunSummary(**test_run_doc["summary"]) if test_run_doc.get("summary") else TestRunSummary(
        total_items=0, completed_items=0, passed_items=0, failed_items=0, error_items=0,
        pass_rate=0, avg_score=0, min_score=0, max_score=0, score_distribution={},
        avg_latency_ms=0, total_tokens=0, estimated_cost=0
    )

    test_run = TestRun(
        id=run_id,
        project_id=project_id,
        prompt_version=test_run_doc.get("prompt_version", 1),
        prompt_text=test_run_doc.get("prompt_text", ""),
        eval_prompt_text=test_run_doc.get("eval_prompt_text", ""),
        llm_provider=test_run_doc.get("llm_provider", "openai"),
        model_name=test_run_doc.get("model_name"),
        status=test_run_doc.get("status", "unknown"),
        pass_threshold=test_run_doc.get("pass_threshold", 3.5),
        results=results,
        summary=summary
    )

    return TestRunResultsResponse(
        test_run=test_run,
        summary=summary,
        results=results
    )


@router.get("/{project_id}/test-runs", response_model=List[dict])
async def list_test_runs(project_id: str, limit: int = 10):
    """
    List all test runs for a project.
    """
    cursor = db.test_runs.find({"project_id": project_id}).sort("created_at", -1).limit(limit)
    runs = await cursor.to_list(length=limit)

    return [
        {
            "run_id": run["_id"],
            "prompt_version": run.get("prompt_version"),
            "status": run.get("status"),
            "llm_provider": run.get("llm_provider"),
            "created_at": run.get("created_at"),
            "completed_at": run.get("completed_at"),
            "summary": run.get("summary")
        }
        for run in runs
    ]


@router.post("/{project_id}/test-runs/single", response_model=SingleTestResponse)
async def run_single_test(project_id: str, request: SingleTestRequest):
    """
    Run a single test for debugging purposes.
    Executes the prompt with one input and optionally evaluates it.
    """
    # Get API key from settings
    settings_doc = await db.settings.find_one()
    api_key = settings_doc.get("api_key") if settings_doc else None
    if not api_key:
        raise HTTPException(status_code=400, detail="API key not configured in settings")

    # Execute single test
    result = await prompt_executor.execute_single_test(
        system_prompt=request.prompt_text,
        test_input=request.test_input,
        eval_prompt=request.eval_prompt_text,
        provider=request.llm_provider,
        api_key=api_key,
        model_name=request.model_name
    )

    return SingleTestResponse(
        input_data=result.input_data,
        prompt_output=result.prompt_output,
        eval_score=result.eval_score if result.eval_score > 0 else None,
        eval_feedback=result.eval_feedback if result.eval_score > 0 else None,
        latency_ms=result.latency_ms,
        tokens_used=result.tokens_used
    )


@router.delete("/{project_id}/test-runs/{run_id}")
async def cancel_test_run(project_id: str, run_id: str):
    """
    Cancel an active test run.
    """
    if run_id in active_test_runs:
        prompt_executor.cancel_run(run_id)
        return {"status": "cancelled", "run_id": run_id}

    # Check if exists in database
    test_run_doc = await db.test_runs.find_one({"_id": run_id})
    if not test_run_doc:
        raise HTTPException(status_code=404, detail="Test run not found")

    if test_run_doc.get("status") == "running":
        await db.test_runs.update_one(
            {"_id": run_id},
            {"$set": {"status": "cancelled"}}
        )
        return {"status": "cancelled", "run_id": run_id}

    return {"status": test_run_doc.get("status"), "run_id": run_id, "message": "Run already completed or cancelled"}


@router.delete("/{project_id}/test-runs/{run_id}/delete")
async def delete_test_run(project_id: str, run_id: str):
    """
    Permanently delete a test run and its results.
    """
    # Cancel if still running
    if run_id in active_test_runs:
        prompt_executor.cancel_run(run_id)
        del active_test_runs[run_id]

    # Delete from database
    result = await db.test_runs.delete_one({"_id": run_id, "project_id": project_id})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Test run not found")

    return {"status": "deleted", "run_id": run_id, "message": "Test run permanently deleted"}


@router.post("/{project_id}/test-runs/compare", response_model=TestRunComparisonResult)
async def compare_test_runs(project_id: str, request: TestRunComparisonRequest):
    """
    Compare two test runs to see improvements/regressions.
    Useful for A/B testing prompt versions.
    """
    # Fetch both runs
    run_a = await db.test_runs.find_one({"_id": request.run_id_a, "project_id": project_id})
    run_b = await db.test_runs.find_one({"_id": request.run_id_b, "project_id": project_id})

    if not run_a:
        raise HTTPException(status_code=404, detail=f"Test run {request.run_id_a} not found")
    if not run_b:
        raise HTTPException(status_code=404, detail=f"Test run {request.run_id_b} not found")

    # Get summaries
    summary_a = run_a.get("summary", {})
    summary_b = run_b.get("summary", {})

    # Calculate deltas
    pass_rate_a = summary_a.get("pass_rate", 0)
    pass_rate_b = summary_b.get("pass_rate", 0)
    avg_score_a = summary_a.get("avg_score", 0)
    avg_score_b = summary_b.get("avg_score", 0)

    # Compare individual items by index
    results_a = {r.get("dataset_item_index"): r for r in run_a.get("results", [])}
    results_b = {r.get("dataset_item_index"): r for r in run_b.get("results", [])}

    improved = 0
    regressed = 0
    unchanged = 0
    item_comparisons = []

    all_indices = set(results_a.keys()) | set(results_b.keys())
    for idx in sorted(all_indices):
        item_a = results_a.get(idx)
        item_b = results_b.get(idx)

        score_a = item_a.get("eval_score", 0) if item_a else None
        score_b = item_b.get("eval_score", 0) if item_b else None

        if score_a is not None and score_b is not None:
            delta = score_b - score_a
            if delta > 0.5:
                improved += 1
                status = "improved"
            elif delta < -0.5:
                regressed += 1
                status = "regressed"
            else:
                unchanged += 1
                status = "unchanged"
        else:
            status = "missing"
            delta = 0

        item_comparisons.append({
            "index": idx,
            "input": (item_a or item_b or {}).get("input_data", {}),
            "score_a": score_a,
            "score_b": score_b,
            "delta": round(delta, 2) if score_a and score_b else None,
            "status": status,
            "passed_a": item_a.get("passed") if item_a else None,
            "passed_b": item_b.get("passed") if item_b else None
        })

    return TestRunComparisonResult(
        run_a={
            "run_id": request.run_id_a,
            "prompt_version": run_a.get("prompt_version"),
            "pass_rate": pass_rate_a,
            "avg_score": avg_score_a,
            "total_items": summary_a.get("completed_items", 0)
        },
        run_b={
            "run_id": request.run_id_b,
            "prompt_version": run_b.get("prompt_version"),
            "pass_rate": pass_rate_b,
            "avg_score": avg_score_b,
            "total_items": summary_b.get("completed_items", 0)
        },
        pass_rate_delta=round(pass_rate_b - pass_rate_a, 1),
        avg_score_delta=round(avg_score_b - avg_score_a, 2),
        improved_items=improved,
        regressed_items=regressed,
        unchanged_items=unchanged,
        item_comparisons=item_comparisons
    )


@router.get("/{project_id}/test-runs/{run_id}/export")
async def export_test_run(project_id: str, run_id: str, format: str = "csv"):
    """
    Export test run results as CSV or JSON.
    """
    from fastapi.responses import Response

    test_run_doc = await db.test_runs.find_one({"_id": run_id, "project_id": project_id})
    if not test_run_doc:
        raise HTTPException(status_code=404, detail="Test run not found")

    results = test_run_doc.get("results", [])
    summary = test_run_doc.get("summary", {})

    if format == "json":
        export_data = {
            "run_id": run_id,
            "project_id": project_id,
            "prompt_version": test_run_doc.get("prompt_version"),
            "status": test_run_doc.get("status"),
            "summary": summary,
            "results": results
        }
        return Response(
            content=json.dumps(export_data, indent=2, default=str),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=test_run_{run_id}.json"}
        )
    else:
        # CSV format
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "Index", "Input", "Output", "Score", "Passed", "Feedback", "Latency (ms)", "Error"
        ])

        # Data rows
        for result in results:
            input_data = result.get("input_data", {})
            input_str = input_data.get("input", json.dumps(input_data))
            writer.writerow([
                result.get("dataset_item_index", ""),
                input_str[:500],  # Truncate for CSV
                result.get("prompt_output", "")[:500],
                result.get("eval_score", ""),
                result.get("passed", ""),
                result.get("eval_feedback", "")[:500],
                result.get("latency_ms", ""),
                result.get("error", "")
            ])

        # Add summary at the bottom
        writer.writerow([])
        writer.writerow(["=== SUMMARY ==="])
        writer.writerow(["Pass Rate", f"{summary.get('pass_rate', 0)}%"])
        writer.writerow(["Avg Score", summary.get('avg_score', 0)])
        writer.writerow(["Total Items", summary.get('completed_items', 0)])
        writer.writerow(["Estimated Cost", f"${summary.get('estimated_cost', 0):.4f}"])

        csv_content = output.getvalue()
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=test_run_{run_id}.csv"}
        )


@router.post("/{project_id}/test-runs/rerun-failed", response_model=dict)
async def rerun_failed_items(project_id: str, request: RerunFailedRequest):
    """
    Re-run only the failed/errored items from a previous test run.
    Creates a new test run with just those items.
    """
    # Get the source run
    source_run = await db.test_runs.find_one({"_id": request.source_run_id, "project_id": project_id})
    if not source_run:
        raise HTTPException(status_code=404, detail="Source test run not found")

    # Find failed/errored items
    failed_indices = []
    for result in source_run.get("results", []):
        if not result.get("passed") or result.get("error"):
            failed_indices.append(result.get("dataset_item_index"))

    if not failed_indices:
        return {"message": "No failed items to re-run", "failed_count": 0}

    # Get project for dataset
    project = await db.projects.find_one({"_id": get_object_id(project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    dataset = project.get("dataset")
    if not dataset:
        raise HTTPException(status_code=400, detail="No dataset found")

    # Parse dataset
    csv_content = dataset.get("csv_content", "")
    all_items = parse_csv_to_test_items(csv_content)

    # Filter to only failed items
    test_items = [all_items[i] for i in failed_indices if i < len(all_items)]

    if not test_items:
        return {"message": "Could not find failed items in dataset", "failed_count": 0}

    # Get settings
    settings_doc = await db.settings.find_one()
    api_key = settings_doc.get("api_key") if settings_doc else None
    if not api_key:
        raise HTTPException(status_code=400, detail="API key not configured")

    llm_provider = request.llm_provider or source_run.get("llm_provider", "openai")
    model_name = request.model_name or source_run.get("model_name")

    # Create new test run
    test_run = TestRun(
        project_id=project_id,
        prompt_version=source_run.get("prompt_version"),
        prompt_text=source_run.get("prompt_text"),
        eval_prompt_text=source_run.get("eval_prompt_text"),
        llm_provider=llm_provider,
        model_name=model_name,
        pass_threshold=source_run.get("pass_threshold", 3.5),
        batch_size=source_run.get("batch_size", 5),
        max_concurrent=source_run.get("max_concurrent", 3)
    )

    # Store in active runs with proper indices
    active_test_runs[test_run.id] = {
        "test_run": test_run,
        "test_items": test_items,
        "total_items": len(test_items),
        "original_indices": failed_indices
    }

    # Save to database
    test_run_dict = test_run.dict()
    test_run_dict["_id"] = test_run.id
    test_run_dict["rerun_of"] = request.source_run_id
    test_run_dict["rerun_indices"] = failed_indices
    await db.test_runs.insert_one(test_run_dict)

    # Start execution
    asyncio.create_task(
        execute_test_run_background(test_run.id, test_run, test_items, api_key)
    )

    return {
        "run_id": test_run.id,
        "status": "started",
        "failed_count": len(failed_indices),
        "rerunning_items": failed_indices[:10]  # Show first 10
    }
