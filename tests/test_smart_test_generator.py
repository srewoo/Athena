"""
Tests for smart_test_generator.py functions
"""
import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from smart_test_generator import (
    detect_input_type,
    build_input_generation_prompt,
    get_scenario_variations,
    InputType,
    InputFormatSpec
)


class TestDetectInputType:
    """Tests for detect_input_type function"""
    
    def test_detect_call_transcript(self):
        """Positive: Detect call transcript input type"""
        prompt = "Analyze this call transcript and extract key points."
        spec = detect_input_type(prompt, ["callTranscript"])
        assert spec.input_type == InputType.CALL_TRANSCRIPT
    
    def test_detect_email(self):
        """Positive: Detect email input type"""
        prompt = "Process the email and summarize the main points."
        spec = detect_input_type(prompt, ["emailContent"])
        assert spec.input_type == InputType.EMAIL
    
    def test_detect_code(self):
        """Positive: Detect code input type"""
        prompt = "Review this code snippet and find bugs."
        spec = detect_input_type(prompt, ["codeSnippet"])
        assert spec.input_type == InputType.CODE
    
    def test_detect_conversation(self):
        """Positive: Detect conversation input type"""
        prompt = "Analyze this chat log and extract sentiment."
        spec = detect_input_type(prompt, ["chatLog"])
        assert spec.input_type == InputType.CONVERSATION
    
    def test_detect_document(self):
        """Positive: Detect document input type"""
        prompt = "Summarize this document."
        spec = detect_input_type(prompt, ["document"])
        assert spec.input_type == InputType.DOCUMENT
    
    def test_detect_ticket(self):
        """Positive: Detect ticket input type"""
        prompt = "Process this support ticket and categorize it."
        spec = detect_input_type(prompt, ["ticket"])
        assert spec.input_type == InputType.TICKET
    
    def test_detect_review(self):
        """Positive: Detect review input type"""
        prompt = "Analyze this product review."
        spec = detect_input_type(prompt, ["review"])
        assert spec.input_type == InputType.REVIEW
    
    def test_detect_simple_text_default(self):
        """Positive: Default to simple text when unclear"""
        prompt = "Answer this question."
        spec = detect_input_type(prompt, ["input"])
        assert spec.input_type == InputType.SIMPLE_TEXT
    
    def test_detect_with_empty_template_variables(self):
        """Edge case: Empty template variables"""
        prompt = "Process the input."
        spec = detect_input_type(prompt, [])
        assert isinstance(spec, InputFormatSpec)
    
    def test_detect_with_multiple_keywords(self):
        """Positive: Multiple keywords should score higher"""
        prompt = "Analyze this call transcript and extract information from the conversation."
        spec = detect_input_type(prompt, ["callTranscript"])
        assert spec.input_type == InputType.CALL_TRANSCRIPT


class TestBuildInputGenerationPrompt:
    """Tests for build_input_generation_prompt function"""
    
    def test_build_prompt_for_call_transcript(self):
        """Positive: Build prompt for call transcript"""
        spec = InputFormatSpec(
            input_type=InputType.CALL_TRANSCRIPT,
            template_variable="callTranscript"
        )
        prompt = build_input_generation_prompt(spec, "Analyze call transcript", "customer_support", "Extract key points")
        assert "transcript" in prompt.lower() or "call" in prompt.lower()
    
    def test_build_prompt_for_email(self):
        """Positive: Build prompt for email"""
        spec = InputFormatSpec(
            input_type=InputType.EMAIL,
            template_variable="emailContent"
        )
        prompt = build_input_generation_prompt(spec, "Process email", "email_processing", "Summarize")
        assert "email" in prompt.lower()
    
    def test_build_prompt_for_code(self):
        """Positive: Build prompt for code"""
        spec = InputFormatSpec(
            input_type=InputType.CODE,
            template_variable="codeSnippet"
        )
        prompt = build_input_generation_prompt(spec, "Review code", "code_review", "Find bugs")
        assert "code" in prompt.lower()
    
    def test_build_prompt_with_domain_context(self):
        """Positive: Include domain context"""
        spec = InputFormatSpec(
            input_type=InputType.SIMPLE_TEXT,
            template_variable="input",
            domain_context="healthcare"
        )
        prompt = build_input_generation_prompt(spec, "Process healthcare data", "healthcare", "Analyze")
        assert "healthcare" in prompt.lower()
    
    def test_build_prompt_with_format_hints(self):
        """Positive: Include format hints"""
        spec = InputFormatSpec(
            input_type=InputType.STRUCTURED_DATA,
            template_variable="data",
            format_hints=["JSON", "nested objects"]
        )
        prompt = build_input_generation_prompt(spec, "Process data", "data_processing", "Parse")
        assert isinstance(prompt, str) and len(prompt) > 0


class TestGetScenarioVariations:
    """Tests for get_scenario_variations function"""
    
    def test_get_scenario_variations_basic(self):
        """Positive: Get scenario variations with correct signature"""
        spec = InputFormatSpec(
            input_type=InputType.SIMPLE_TEXT,
            template_variable="input"
        )
        variations = get_scenario_variations(spec, 5)
        assert isinstance(variations, list)
    
    def test_get_scenario_variations_call_transcript(self):
        """Positive: Get variations for call transcript"""
        spec = InputFormatSpec(
            input_type=InputType.CALL_TRANSCRIPT,
            template_variable="transcript"
        )
        variations = get_scenario_variations(spec, 3)
        assert isinstance(variations, list)
    
    def test_get_scenario_variations_email(self):
        """Positive: Get variations for email"""
        spec = InputFormatSpec(
            input_type=InputType.EMAIL,
            template_variable="email"
        )
        variations = get_scenario_variations(spec, 4)
        assert isinstance(variations, list)
    
    def test_get_scenario_variations_zero_cases(self):
        """Edge case: Request zero variations"""
        spec = InputFormatSpec(
            input_type=InputType.SIMPLE_TEXT,
            template_variable="input"
        )
        variations = get_scenario_variations(spec, 0)
        assert isinstance(variations, list)
    
    def test_get_scenario_variations_large_number(self):
        """Edge case: Request many variations"""
        spec = InputFormatSpec(
            input_type=InputType.SIMPLE_TEXT,
            template_variable="input"
        )
        variations = get_scenario_variations(spec, 100)
        assert isinstance(variations, list)
