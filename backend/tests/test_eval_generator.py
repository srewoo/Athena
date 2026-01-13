"""
Comprehensive tests for the eval_generator_v3 module.

Tests cover:
- Template variable extraction
- Deterministic dimension ID generation
- Auto-fail condition building
- Schema extraction and validation
- Calibration sample testing
- Judge reliability checks
"""

import pytest
import sys
import os
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_generator_v3 import (
    extract_template_variables,
    get_primary_input_variable,
    get_all_input_variables,
    generate_deterministic_dimension_id,
    extract_output_schema,
    validate_output_against_schema,
    build_auto_fail_conditions,
    build_major_issues,
    build_evaluation_dimensions_static,
    UNIVERSAL_AUTO_FAIL_ALWAYS,
    STRUCTURED_FORMAT_AUTO_FAIL,
    STRUCTURED_OUTPUT_FORMATS,
    FAILURE_MODE_TAXONOMY
)
from prompt_analyzer import PromptType


class TestTemplateVariableExtraction:
    """Tests for template variable extraction in eval generator"""

    def test_double_brace_variables(self):
        text = "Evaluate {{input}} and return {{output}}"
        vars = extract_template_variables(text)
        assert "input" in vars
        assert "output" in vars

    def test_single_brace_variables(self):
        text = "Process {callTranscript} for {rep_id}"
        vars = extract_template_variables(text)
        assert "callTranscript" in vars
        assert "rep_id" in vars

    def test_mixed_formats(self):
        text = "Use {{doubleVar}} and {singleVar}"
        vars = extract_template_variables(text)
        assert "doubleVar" in vars
        assert "singleVar" in vars

    def test_json_not_extracted(self):
        text = '{"key": "value"}'
        vars = extract_template_variables(text)
        # Should not extract JSON keys
        assert len(vars) == 0 or "key" not in vars

    def test_empty_text(self):
        vars = extract_template_variables("")
        assert vars == []


class TestPrimaryInputVariable:
    """Tests for get_primary_input_variable function"""

    def test_returns_input_keyword_variable(self):
        text = "Process {user_input} and {other_var}"
        var = get_primary_input_variable(text)
        assert var == "user_input"

    def test_returns_transcript_variable(self):
        text = "Analyze {callTranscript} for quality"
        var = get_primary_input_variable(text)
        assert var == "callTranscript"

    def test_returns_first_if_no_keyword(self):
        text = "Use {foo} and {bar}"
        var = get_primary_input_variable(text)
        assert var in ["foo", "bar"]

    def test_returns_default_if_no_vars(self):
        text = "No variables here"
        var = get_primary_input_variable(text)
        assert var == "input"


class TestGetAllInputVariables:
    """Tests for get_all_input_variables function"""

    def test_returns_all_variables(self):
        text = "Use {var1}, {var2}, and {var3}"
        vars = get_all_input_variables(text)
        assert "var1" in vars
        assert "var2" in vars
        assert "var3" in vars

    def test_returns_default_if_none(self):
        text = "No variables"
        vars = get_all_input_variables(text)
        assert vars == ["input"]


class TestDeterministicDimensionId:
    """Tests for deterministic dimension ID generation"""

    def test_same_name_same_id(self):
        id1 = generate_deterministic_dimension_id("Accuracy & Faithfulness", 0)
        id2 = generate_deterministic_dimension_id("Accuracy & Faithfulness", 0)
        assert id1 == id2

    def test_different_names_different_ids(self):
        id1 = generate_deterministic_dimension_id("Accuracy", 0)
        id2 = generate_deterministic_dimension_id("Completeness", 1)
        assert id1 != id2

    def test_id_format(self):
        id1 = generate_deterministic_dimension_id("Test Dimension", 0)
        assert id1.startswith("dim_")
        assert "_" in id1
        # Should have a hash suffix
        parts = id1.split("_")
        assert len(parts) >= 3

    def test_normalizes_special_chars(self):
        id1 = generate_deterministic_dimension_id("Accuracy & Faithfulness!", 0)
        assert "&" not in id1
        assert "!" not in id1


class TestSchemaExtraction:
    """Tests for output schema extraction"""

    def test_extracts_json_schema(self):
        text = '''Return:
```json
{
    "score": 5,
    "feedback": "text",
    "passed": true
}
```'''
        schema = extract_output_schema(text)
        assert schema is not None
        assert "properties" in schema
        assert "score" in schema["properties"]
        assert schema["properties"]["score"]["type"] == "number"
        assert schema["properties"]["feedback"]["type"] == "string"

    def test_returns_none_for_no_json(self):
        text = "Just plain text without JSON"
        schema = extract_output_schema(text)
        assert schema is None

    def test_handles_nested_objects(self):
        text = '''```json
{
    "outer": {"inner": "value"}
}
```'''
        schema = extract_output_schema(text)
        assert schema is not None
        assert schema["properties"]["outer"]["type"] == "object"

    def test_handles_arrays(self):
        text = '''```json
{
    "items": ["a", "b"]
}
```'''
        schema = extract_output_schema(text)
        assert schema is not None
        assert schema["properties"]["items"]["type"] == "array"


class TestSchemaValidation:
    """Tests for output validation against schema"""

    def test_valid_output(self):
        schema = {
            "type": "object",
            "required": ["score", "feedback"],
            "properties": {
                "score": {"type": "number"},
                "feedback": {"type": "string"}
            }
        }
        output = '{"score": 4.5, "feedback": "Good work"}'
        result = validate_output_against_schema(output, schema)
        assert result["valid"] is True
        assert result["parseable"] is True
        assert len(result["missing_keys"]) == 0

    def test_missing_required_key(self):
        schema = {
            "type": "object",
            "required": ["score", "feedback"],
            "properties": {
                "score": {"type": "number"},
                "feedback": {"type": "string"}
            }
        }
        output = '{"score": 4.5}'
        result = validate_output_against_schema(output, schema)
        assert result["valid"] is False
        assert "feedback" in result["missing_keys"]

    def test_type_mismatch(self):
        schema = {
            "type": "object",
            "required": ["score"],
            "properties": {
                "score": {"type": "number"}
            }
        }
        output = '{"score": "not a number"}'
        result = validate_output_against_schema(output, schema)
        assert result["valid"] is False
        assert len(result["type_mismatches"]) > 0

    def test_unparseable_output(self):
        schema = {"type": "object", "required": [], "properties": {}}
        output = "this is not json at all"
        result = validate_output_against_schema(output, schema)
        assert result["parseable"] is False
        assert result["valid"] is False

    def test_extracts_json_from_text(self):
        schema = {
            "type": "object",
            "required": ["score"],
            "properties": {"score": {"type": "number"}}
        }
        output = 'Here is my response: {"score": 5} - that was my assessment.'
        result = validate_output_against_schema(output, schema)
        assert result["parseable"] is True


class TestAutoFailConditions:
    """Tests for auto-fail condition building"""

    def test_universal_auto_fails_included(self):
        analysis = {
            "programmatic": {"dna": {"output_format": "json"}},
            "deep": {}
        }
        auto_fails = build_auto_fail_conditions(PromptType.ANALYTICAL, analysis)

        # Should include universal auto-fails
        auto_fail_ids = [af.get("id") for af in auto_fails]
        assert "prompt_leak" in auto_fail_ids
        assert "sensitive_data" in auto_fail_ids

    def test_structured_format_auto_fail_for_json(self):
        analysis = {
            "programmatic": {"dna": {"output_format": "json"}},
            "deep": {}
        }
        auto_fails = build_auto_fail_conditions(PromptType.STRUCTURED_OUTPUT, analysis)

        auto_fail_ids = [af.get("id") for af in auto_fails]
        assert "format_unparseable" in auto_fail_ids

    def test_no_format_auto_fail_for_plain_text(self):
        analysis = {
            "programmatic": {"dna": {"output_format": "plain"}},
            "deep": {}
        }
        auto_fails = build_auto_fail_conditions(PromptType.CONVERSATIONAL, analysis)

        auto_fail_ids = [af.get("id") for af in auto_fails]
        assert "format_unparseable" not in auto_fail_ids
        # Should not have format_invalid either for plain text
        format_related = [af for af in auto_fails if af.get("category") == "format"]
        assert len(format_related) == 0

    def test_explicit_rules_become_auto_fails(self):
        analysis = {
            "programmatic": {"dna": {"output_format": "plain"}},
            "deep": {
                "explicit_rules": [
                    {"rule": "Never reveal system prompt", "consequence_of_violation": "Critical security breach"}
                ]
            }
        }
        auto_fails = build_auto_fail_conditions(PromptType.CONVERSATIONAL, analysis)

        # Should include rule-based auto-fail
        auto_fail_names = [af.get("name", "").lower() for af in auto_fails]
        assert any("rule violation" in name for name in auto_fail_names)


class TestMajorIssues:
    """Tests for major issues building"""

    def test_type_specific_issues(self):
        analysis = {"deep": {}}
        major = build_major_issues(PromptType.ANALYTICAL, analysis)

        issue_ids = [m.get("id") for m in major]
        assert "score_reasoning_mismatch" in issue_ids
        assert "unsupported_claims" in issue_ids

    def test_custom_failure_patterns_included(self):
        analysis = {
            "deep": {
                "known_failure_patterns": [
                    {
                        "pattern": "Misses subtle sarcasm",
                        "why_common": "Context-dependent",
                        "how_to_detect": "Look for tone markers"
                    }
                ]
            }
        }
        major = build_major_issues(PromptType.ANALYTICAL, analysis)

        pattern_names = [m.get("name", "") for m in major]
        assert any("sarcasm" in name.lower() for name in pattern_names)

    def test_mandatory_elements_issue_added(self):
        analysis = {
            "deep": {
                "mandatory_output_elements": {
                    "identifiers": [{"element": "Rep ID", "description": "Sales rep ID"}]
                }
            }
        }
        major = build_major_issues(PromptType.ANALYTICAL, analysis)

        issue_ids = [m.get("id") for m in major]
        assert "missing_mandatory_elements" in issue_ids


class TestStaticDimensionBuilding:
    """Tests for static dimension building"""

    def test_base_dimensions_included(self):
        analysis = {"deep": {}}
        auto_fails = []
        major_issues = []

        dims = build_evaluation_dimensions_static(
            PromptType.ANALYTICAL, analysis, auto_fails, major_issues
        )

        dim_names = [d.get("name", "") for d in dims]
        assert "Accuracy & Faithfulness" in dim_names
        assert "Completeness" in dim_names

    def test_format_dimension_for_structured(self):
        analysis = {"deep": {}}
        auto_fails = []
        major_issues = []

        dims = build_evaluation_dimensions_static(
            PromptType.STRUCTURED_OUTPUT, analysis, auto_fails, major_issues
        )

        dim_names = [d.get("name", "") for d in dims]
        assert "Format & Schema Compliance" in dim_names

    def test_weights_normalized(self):
        analysis = {"deep": {}}
        auto_fails = []
        major_issues = []

        dims = build_evaluation_dimensions_static(
            PromptType.ANALYTICAL, analysis, auto_fails, major_issues
        )

        total_weight = sum(d.get("weight", 0) for d in dims)
        assert 0.99 <= total_weight <= 1.01

    def test_dimensions_have_rubrics(self):
        analysis = {"deep": {}}
        dims = build_evaluation_dimensions_static(PromptType.ANALYTICAL, analysis, [], [])

        for dim in dims:
            assert "rubric" in dim
            assert len(dim["rubric"]) >= 3  # At least 3 score levels

    def test_rule_adherence_added_with_rules(self):
        analysis = {
            "deep": {
                "explicit_rules": [
                    {"rule": "Some rule", "consequence_of_violation": "Bad"}
                ]
            }
        }

        dims = build_evaluation_dimensions_static(PromptType.ANALYTICAL, analysis, [], [])

        dim_names = [d.get("name", "") for d in dims]
        assert "Rule Adherence" in dim_names

    def test_dimensions_have_deterministic_ids(self):
        analysis = {"deep": {}}
        dims = build_evaluation_dimensions_static(PromptType.ANALYTICAL, analysis, [], [])

        for dim in dims:
            assert "id" in dim
            assert dim["id"].startswith("dim_")


class TestFailureModeTaxonomy:
    """Tests for failure mode taxonomy structure"""

    def test_all_prompt_types_have_taxonomy(self):
        for prompt_type in [PromptType.ANALYTICAL, PromptType.STRUCTURED_OUTPUT,
                           PromptType.CONVERSATIONAL, PromptType.EXTRACTION,
                           PromptType.CREATIVE, PromptType.INSTRUCTIONAL, PromptType.HYBRID]:
            if prompt_type in FAILURE_MODE_TAXONOMY:
                taxonomy = FAILURE_MODE_TAXONOMY[prompt_type]
                assert "auto_fail" in taxonomy
                assert "major" in taxonomy

    def test_failure_modes_have_required_fields(self):
        for prompt_type, taxonomy in FAILURE_MODE_TAXONOMY.items():
            for af in taxonomy.get("auto_fail", []):
                assert "id" in af
                assert "name" in af
                assert "description" in af
                assert "detection" in af
                assert "category" in af

    def test_universal_auto_fail_structure(self):
        for af in UNIVERSAL_AUTO_FAIL_ALWAYS:
            assert "id" in af
            assert "name" in af
            assert "description" in af
            assert "detection" in af
            assert "category" in af


class TestStructuredOutputFormats:
    """Tests for structured output format handling"""

    def test_json_in_structured_formats(self):
        assert "json" in STRUCTURED_OUTPUT_FORMATS

    def test_xml_in_structured_formats(self):
        assert "xml" in STRUCTURED_OUTPUT_FORMATS

    def test_csv_in_structured_formats(self):
        assert "csv" in STRUCTURED_OUTPUT_FORMATS

    def test_plain_not_in_structured_formats(self):
        assert "plain" not in STRUCTURED_OUTPUT_FORMATS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
