"""
Eval Prompt Parser - Extract structured data from enhanced eval prompts

This module parses the XML-formatted evaluation examples from the enhanced
eval prompts to extract:
- Analysis text
- Evidence (positive and negative)
- Rubric mapping
- Scores
- Confidence levels
- Pass/fail status
"""

import re
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


def extract_xml_tag_content(text: str, tag: str) -> Optional[str]:
    """
    Extract content between XML tags.

    Args:
        text: The text to search
        tag: The tag name (e.g., "analysis", "score")

    Returns:
        Content between tags or None if not found
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_list_items(text: str) -> List[str]:
    """
    Extract bullet point items from text.

    Args:
        text: Text containing bullet points (-, *, or •)

    Returns:
        List of items
    """
    if not text:
        return []

    # Match lines starting with -, *, or •
    items = re.findall(r'^\s*[-*•]\s*(.+)$', text, re.MULTILINE)
    return [item.strip() for item in items if item.strip()]


def parse_evaluation_example(eval_xml: str) -> Dict[str, Any]:
    """
    Parse a single evaluation example from XML format.

    Args:
        eval_xml: XML string containing an evaluation

    Returns:
        Dict with parsed fields
    """
    result = {
        "analysis": None,
        "evidence_positive": [],
        "evidence_negative": [],
        "rubric_mapping": None,
        "score": None,
        "confidence": None,
        "confidence_reasoning": None,
        "pass_fail": None
    }

    try:
        # Extract main fields
        result["analysis"] = extract_xml_tag_content(eval_xml, "analysis")
        result["rubric_mapping"] = extract_xml_tag_content(eval_xml, "rubric_mapping")
        result["confidence"] = extract_xml_tag_content(eval_xml, "confidence")
        result["confidence_reasoning"] = extract_xml_tag_content(eval_xml, "confidence_reasoning")
        result["pass_fail"] = extract_xml_tag_content(eval_xml, "pass_fail")

        # Extract score (convert to int)
        score_text = extract_xml_tag_content(eval_xml, "score")
        if score_text:
            # Extract first number found
            score_match = re.search(r'\d+', score_text)
            if score_match:
                result["score"] = int(score_match.group())

        # Extract evidence (positive and negative)
        evidence_block = extract_xml_tag_content(eval_xml, "evidence")
        if evidence_block:
            # Extract positive evidence
            positive_block = extract_xml_tag_content(evidence_block, "positive")
            if positive_block:
                result["evidence_positive"] = extract_list_items(positive_block)

            # Extract negative evidence
            negative_block = extract_xml_tag_content(evidence_block, "negative")
            if negative_block:
                result["evidence_negative"] = extract_list_items(negative_block)

    except Exception as e:
        logger.error(f"Error parsing evaluation XML: {e}", exc_info=True)

    return result


def extract_all_examples_from_prompt(eval_prompt: str) -> List[Dict[str, Any]]:
    """
    Extract all calibration examples from an eval prompt.

    Args:
        eval_prompt: The full evaluation prompt text

    Returns:
        List of parsed evaluation examples
    """
    examples = []

    try:
        # Method 1: Try to find XML blocks with ```xml wrapper
        eval_blocks = re.findall(
            r'```xml\s*<evaluation>.*?</evaluation>\s*```',
            eval_prompt,
            re.DOTALL | re.IGNORECASE
        )

        if eval_blocks:
            logger.debug(f"Found {len(eval_blocks)} XML blocks with ```xml wrapper")
            for block in eval_blocks:
                # Remove the ```xml and ``` markers
                xml_content = re.sub(r'```xml\s*|```\s*', '', block, flags=re.IGNORECASE)
                parsed = parse_evaluation_example(xml_content)
                examples.append(parsed)
        else:
            # Method 2: Find XML blocks without wrapper (raw <evaluation> tags)
            eval_blocks = re.findall(
                r'<evaluation>.*?</evaluation>',
                eval_prompt,
                re.DOTALL | re.IGNORECASE
            )
            logger.debug(f"Found {len(eval_blocks)} XML blocks without wrapper")

            for block in eval_blocks:
                parsed = parse_evaluation_example(block)
                examples.append(parsed)

        logger.info(f"Extracted {len(examples)} evaluation examples from prompt (expected 3)")

    except Exception as e:
        logger.error(f"Error extracting examples from prompt: {e}", exc_info=True)

    return examples


def extract_rubric_levels(eval_prompt: str) -> List[Dict[str, str]]:
    """
    Extract rubric level descriptions from an eval prompt.

    Args:
        eval_prompt: The full evaluation prompt text

    Returns:
        List of dicts with "level" and "description"
    """
    rubric_levels = []

    try:
        # Find the scoring rubric section
        rubric_match = re.search(
            r'## Scoring Rubric \(0-10 Scale\)(.*?)(?=##|\Z)',
            eval_prompt,
            re.DOTALL | re.IGNORECASE
        )

        if rubric_match:
            rubric_text = rubric_match.group(1)

            # Extract each level (e.g., "**0-2 points: Severe Issues**")
            level_pattern = r'\*\*(\d+-\d+ points?:[^*]+)\*\*(.*?)(?=\*\*\d+-\d+ points?:|\Z)'
            level_matches = re.findall(level_pattern, rubric_text, re.DOTALL)

            for level_header, level_content in level_matches:
                rubric_levels.append({
                    "level": level_header.strip(),
                    "description": level_content.strip()
                })

        logger.info(f"Extracted {len(rubric_levels)} rubric levels")

    except Exception as e:
        logger.error(f"Error extracting rubric levels: {e}", exc_info=True)

    return rubric_levels


def extract_evaluation_purpose(eval_prompt: str) -> Optional[str]:
    """
    Extract the purpose section from an eval prompt.

    Args:
        eval_prompt: The full evaluation prompt text

    Returns:
        Purpose text or None
    """
    try:
        purpose_match = re.search(
            r'## Purpose\s*\n(.*?)(?=##|\Z)',
            eval_prompt,
            re.DOTALL | re.IGNORECASE
        )

        if purpose_match:
            return purpose_match.group(1).strip()

    except Exception as e:
        logger.error(f"Error extracting purpose: {e}", exc_info=True)

    return None


def extract_ai_system_context(eval_prompt: str) -> Optional[Dict[str, str]]:
    """
    Extract the AI System Context section from an eval prompt.

    Args:
        eval_prompt: The full evaluation prompt text

    Returns:
        Dict with system_prompt, use_case, system_summary or None
    """
    try:
        # Find the AI System Context section
        context_match = re.search(
            r'## AI System Context\s*\n(.*?)(?=##)',
            eval_prompt,
            re.DOTALL | re.IGNORECASE
        )

        if not context_match:
            return None

        context_text = context_match.group(1)

        result = {
            "system_prompt": None,
            "use_case": None,
            "system_summary": None
        }

        # Extract system prompt (between ``` markers)
        system_prompt_match = re.search(
            r'```\s*\n(.*?)\n\s*```',
            context_text,
            re.DOTALL
        )
        if system_prompt_match:
            result["system_prompt"] = system_prompt_match.group(1).strip()

        # Extract use case
        use_case_match = re.search(
            r'\*\*Use Case:\*\*\s*(.+?)(?=\n|$)',
            context_text,
            re.IGNORECASE
        )
        if use_case_match:
            result["use_case"] = use_case_match.group(1).strip()

        # Extract system summary
        summary_match = re.search(
            r'\*\*What This System Does:\*\*\s*\n(.+?)(?=##|\Z)',
            context_text,
            re.DOTALL | re.IGNORECASE
        )
        if summary_match:
            result["system_summary"] = summary_match.group(1).strip()

        return result

    except Exception as e:
        logger.error(f"Error extracting AI system context: {e}", exc_info=True)

    return None


def analyze_eval_prompt_quality(eval_prompt: str) -> Dict[str, Any]:
    """
    Analyze the quality of a generated eval prompt.

    Checks for:
    - Presence of chain-of-thought structure
    - 0-10 scoring scale
    - Multiple examples (should be 3)
    - XML output format
    - Uncertainty handling

    Args:
        eval_prompt: The full evaluation prompt text

    Returns:
        Dict with quality metrics
    """
    analysis = {
        "has_chain_of_thought": False,
        "has_0_10_scale": False,
        "num_examples": 0,
        "has_xml_format": False,
        "has_uncertainty_handling": False,
        "has_behavioral_anchors": False,
        "quality_score": 0.0
    }

    try:
        # Check for chain-of-thought (more flexible)
        cot_indicators = ["step 1", "step 2", "analysis", "evidence", "reasoning"]
        cot_count = sum(1 for indicator in cot_indicators if indicator in eval_prompt.lower())
        if cot_count >= 3:  # At least 3 indicators
            analysis["has_chain_of_thought"] = True
            logger.debug(f"Chain-of-thought: YES ({cot_count} indicators)")
        else:
            logger.debug(f"Chain-of-thought: NO (only {cot_count} indicators)")

        # Check for 0-10 scale (more flexible)
        scale_indicators = ["0-10", "0 to 10", "scale: 0-10", "out of 10", "/10"]
        has_scale = any(indicator in eval_prompt.lower() for indicator in scale_indicators)
        if has_scale:
            analysis["has_0_10_scale"] = True
            logger.debug("0-10 scale: YES")
        else:
            logger.debug("0-10 scale: NO")

        # Count examples (already flexible)
        examples = extract_all_examples_from_prompt(eval_prompt)
        analysis["num_examples"] = len(examples)
        logger.debug(f"Examples found: {len(examples)}")

        # Check for XML format (more flexible)
        xml_tags = ["<evaluation>", "<analysis>", "<evidence>", "<score>"]
        xml_count = sum(1 for tag in xml_tags if tag in eval_prompt.lower())
        if xml_count >= 2:  # At least 2 XML tags
            analysis["has_xml_format"] = True
            logger.debug(f"XML format: YES ({xml_count} tags)")
        else:
            logger.debug(f"XML format: NO (only {xml_count} tags)")

        # Check for uncertainty handling (more flexible)
        uncertainty_indicators = ["confidence", "ambiguous", "uncertain", "unclear"]
        uncertainty_count = sum(1 for ind in uncertainty_indicators if ind in eval_prompt.lower())
        if uncertainty_count >= 2:
            analysis["has_uncertainty_handling"] = True
            logger.debug(f"Uncertainty handling: YES ({uncertainty_count} indicators)")
        else:
            logger.debug(f"Uncertainty handling: NO (only {uncertainty_count} indicators)")

        # Check for behavioral anchors (more flexible)
        behavioral_indicators = ["observable", "specific", "behavior", "concrete", "measurable"]
        behavioral_count = sum(1 for ind in behavioral_indicators if ind in eval_prompt.lower())
        if behavioral_count >= 3:
            analysis["has_behavioral_anchors"] = True
            logger.debug(f"Behavioral anchors: YES ({behavioral_count} indicators)")
        else:
            logger.debug(f"Behavioral anchors: NO (only {behavioral_count} indicators)")

        # Calculate quality score (0-10)
        score = 0.0
        if analysis["has_chain_of_thought"]:
            score += 2.5
            logger.debug("  +2.5 for chain-of-thought")
        if analysis["has_0_10_scale"]:
            score += 2.0
            logger.debug("  +2.0 for 0-10 scale")
        if analysis["num_examples"] >= 3:
            score += 2.0
            logger.debug(f"  +2.0 for {analysis['num_examples']} examples")
        elif analysis["num_examples"] > 0:
            score += 1.0
            logger.debug(f"  +1.0 for {analysis['num_examples']} examples")
        if analysis["has_xml_format"]:
            score += 2.0
            logger.debug("  +2.0 for XML format")
        if analysis["has_uncertainty_handling"]:
            score += 0.75
            logger.debug("  +0.75 for uncertainty handling")
        if analysis["has_behavioral_anchors"]:
            score += 0.75
            logger.debug("  +0.75 for behavioral anchors")

        analysis["quality_score"] = round(score, 1)
        logger.info(f"Final quality score: {analysis['quality_score']}/10")

    except Exception as e:
        logger.error(f"Error analyzing eval prompt quality: {e}", exc_info=True)

    return analysis
