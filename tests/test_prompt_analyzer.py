"""
Tests for prompt_analyzer.py functions
"""
import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

try:
    from prompt_analyzer import analyze_prompt
    HAS_PROMPT_ANALYZER = True
except ImportError:
    HAS_PROMPT_ANALYZER = False


@pytest.mark.skipif(not HAS_PROMPT_ANALYZER, reason="prompt_analyzer module not available")
class TestPromptAnalyzer:
    """Tests for prompt_analyzer functions"""
    
    def test_analyze_prompt_basic(self):
        """Positive: Basic prompt analysis"""
        prompt = "You are a helpful assistant."
        result = analyze_prompt(prompt)
        # analyze_prompt may return a PromptAnalysis object, not a dict
        assert result is not None
        assert hasattr(result, 'quality_score') or isinstance(result, dict)
    
    def test_analyze_prompt_detailed(self):
        """Positive: Detailed prompt analysis"""
        prompt = """You are an expert assistant.
        
## Task
Answer questions accurately.

## Instructions
1. Provide clear answers
2. Cite sources
3. Use professional tone"""
        result = analyze_prompt(prompt)
        # analyze_prompt may return a PromptAnalysis object, not a dict
        assert result is not None
        assert hasattr(result, 'quality_score') or isinstance(result, dict)
