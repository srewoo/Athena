#!/usr/bin/env python3
"""
Test script to debug eval generation endpoint
"""

import asyncio
import sys
from unified_eval_api import generate_eval_prompt, GenerateEvalRequest

async def test_eval_generation():
    """Test the eval generation endpoint"""

    # Use the project ID from the error
    project_id = "a3863dc7-280b-433a-955f-05fea51fe348"

    # Create request
    request = GenerateEvalRequest(
        mode="single",
        requirements="Accuracy and factual correctness",
        use_intelligent_extraction=True,
        use_meta_evaluation=True,
        auto_refine=True,
        meta_quality_threshold=7.5
    )

    print(f"Testing eval generation for project: {project_id}")
    print(f"Request: {request.model_dump()}")
    print("=" * 60)

    try:
        result = await generate_eval_prompt(project_id, request)
        print("✅ SUCCESS!")
        print(f"Mode: {result.mode}")
        print(f"Eval prompt length: {len(result.eval_prompt) if result.eval_prompt else 0}")
        print(f"Meta quality: {result.meta_evaluation.get('quality_score') if result.meta_evaluation else 'N/A'}")
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_eval_generation())
    sys.exit(0 if success else 1)
