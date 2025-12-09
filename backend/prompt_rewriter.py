"""
AI-powered prompt rewriter - uses LLMs to improve prompts based on requirements and feedback

Uses official best practices from:
- OpenAI GPT-5/4.1: https://cookbook.openai.com/examples/gpt-5/gpt-5_prompting_guide
- Anthropic Claude 4: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-4-best-practices
- Google Gemini: https://ai.google.dev/gemini-api/docs/prompting-strategies
"""
from typing import List, Dict, Any, Optional
from models import Requirements
from official_best_practices import get_provider_guidance
import openai
import anthropic
import google.generativeai as genai


# Default models for each provider
DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "claude": "claude-3-5-sonnet-20241022",
    "gemini": "gemini-2.0-flash-exp"
}


class PromptRewriter:
    """Uses LLMs to rewrite and improve prompts"""

    def __init__(self):
        # No initialization needed - clients are created per-request with provided API keys
        pass

    async def rewrite_prompt(
        self,
        current_prompt: str,
        requirements: Requirements,
        gaps: List[str],
        violations: List[Dict[str, str]],
        focus_areas: Optional[List[str]] = None,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Rewrite a prompt to improve alignment with requirements and best practices

        Args:
            current_prompt: The current system prompt
            requirements: User requirements
            gaps: List of requirement gaps
            violations: List of best practice violations
            focus_areas: Optional specific areas to focus on
            provider: LLM provider to use for rewriting
            api_key: API key for the provider (from database settings)
            model_name: Optional specific model to use

        Returns:
            Dictionary with improved_prompt, changes_made, and rationale
        """
        # Build the rewrite instruction
        rewrite_instruction = self._build_rewrite_instruction(
            current_prompt,
            requirements,
            gaps,
            violations,
            focus_areas
        )

        # Use the specified provider to rewrite (only if API key is provided)
        if provider == "openai" and api_key:
            result = await self._rewrite_with_openai(rewrite_instruction, api_key, model_name)
        elif provider == "claude" and api_key:
            result = await self._rewrite_with_claude(rewrite_instruction, api_key, model_name)
        elif provider == "gemini" and api_key:
            result = await self._rewrite_with_gemini(rewrite_instruction, api_key, model_name)
        else:
            # Fallback to rule-based rewrite
            result = self._rule_based_rewrite(current_prompt, gaps, violations)

        return result

    def _build_rewrite_instruction(
        self,
        current_prompt: str,
        requirements: Requirements,
        gaps: List[str],
        violations: List[Dict[str, str]],
        focus_areas: Optional[List[str]] = None
    ) -> str:
        """Build the instruction for the LLM to rewrite the prompt"""

        # Get official provider-specific guidance
        provider_guidance = get_provider_guidance(requirements.target_provider)

        instruction = f"""You are an expert prompt engineer specializing in optimizing system prompts for production LLM applications.

## Current System Prompt
```
{current_prompt}
```

## Use Case
{requirements.use_case}

## Key Requirements
{self._format_list(requirements.key_requirements)}

## Identified Gaps (Missing Requirements)
{self._format_list(gaps) if gaps else "None - all requirements covered"}

## Best Practice Violations
{self._format_violations(violations) if violations else "None - follows best practices"}

## Target Provider
{requirements.target_provider}

{self._format_focus_areas(focus_areas)}

## OFFICIAL PROVIDER BEST PRACTICES

The following guidance is extracted from official provider documentation. Apply these practices when rewriting:

{provider_guidance}

## REWRITE INSTRUCTIONS

### 1. Fix Formatting Issues
- Fix any spelling, grammar, or punctuation errors
- Correct inconsistent formatting (headers, bullets, spacing)
- Remove duplicate or redundant content
- Fix broken markdown/XML syntax
- Ensure proper indentation and structure
- Standardize naming conventions and terminology

### 2. Address Content Gaps
- Rewrite the system prompt to address ALL identified gaps
- Fix ALL best practice violations
- Maintain the original intent and core functionality

### 3. Apply Official Provider Best Practices
Follow the provider-specific guidance above for {requirements.target_provider}. Key points:
- Use the recommended structure and formatting for this provider
- Apply the delimiter and section organization patterns
- Include examples if the provider guidance recommends them
- Follow instruction placement recommendations

### 4. Improve Clarity & Structure
Based on official documentation, organize the prompt with:
- Clear role/persona definition at the start
- Structured sections with appropriate delimiters
- Unambiguous, actionable instructions
- Examples where beneficial (especially for Gemini)
- Output format specification
- Constraints and boundaries

### 5. Quality Checklist
Before finalizing, verify:
- [ ] All {len(gaps)} requirement gaps addressed
- [ ] All best practice violations fixed
- [ ] Original intent preserved
- [ ] Provider-optimized for {requirements.target_provider}
- [ ] Clear, unambiguous instructions
- [ ] Appropriate length (not overly verbose)
- [ ] Follows official provider documentation patterns

## OUTPUT FORMAT

Provide your response in this EXACT format:

### Improved Prompt
```
[Your complete improved system prompt here - ready to use as-is]
```

### Changes Made
1. [Specific change #1 - what you changed and why]
2. [Specific change #2 - what you changed and why]
... (list ALL changes)

### Rationale
[2-3 sentences explaining how these changes improve alignment with requirements and official best practices]
"""
        return instruction

    async def _rewrite_with_openai(self, instruction: str, api_key: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Use OpenAI to rewrite the prompt"""
        try:
            client = openai.AsyncOpenAI(api_key=api_key)
            model = model_name or DEFAULT_MODELS["openai"]

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert prompt engineer specializing in optimizing system prompts."},
                    {"role": "user", "content": instruction}
                ],
                temperature=0.7,
                max_tokens=4000
            )

            content = response.choices[0].message.content
            return self._parse_rewrite_response(content)

        except Exception as e:
            return {
                "improved_prompt": "",
                "changes_made": [],
                "rationale": f"Error using OpenAI: {str(e)}"
            }

    async def _rewrite_with_claude(self, instruction: str, api_key: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Use Claude to rewrite the prompt"""
        try:
            client = anthropic.AsyncAnthropic(api_key=api_key)
            model = model_name or DEFAULT_MODELS["claude"]

            response = await client.messages.create(
                model=model,
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": instruction}
                ]
            )

            content = response.content[0].text
            return self._parse_rewrite_response(content)

        except Exception as e:
            return {
                "improved_prompt": "",
                "changes_made": [],
                "rationale": f"Error using Claude: {str(e)}"
            }

    async def _rewrite_with_gemini(self, instruction: str, api_key: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Use Gemini to rewrite the prompt"""
        try:
            genai.configure(api_key=api_key)
            model_id = model_name or DEFAULT_MODELS["gemini"]
            model = genai.GenerativeModel(model_id)

            response = await model.generate_content_async(instruction)

            content = response.text
            return self._parse_rewrite_response(content)

        except Exception as e:
            return {
                "improved_prompt": "",
                "changes_made": [],
                "rationale": f"Error using Gemini: {str(e)}"
            }

    def _parse_rewrite_response(self, content: str) -> Dict[str, Any]:
        """Parse the LLM response to extract improved prompt, changes, and rationale"""
        # Simple parsing logic
        improved_prompt = ""
        changes_made = []
        rationale = ""

        # Split by sections
        sections = content.split("###")

        for section in sections:
            section = section.strip()
            if section.startswith("Improved Prompt") or section.startswith(" Improved Prompt"):
                # Extract prompt (everything after the header until next section)
                lines = section.split("\n", 1)
                if len(lines) > 1:
                    improved_prompt = lines[1].strip()
                    # Remove markdown code blocks if present
                    improved_prompt = improved_prompt.replace("```", "").strip()

            elif section.startswith("Changes Made") or section.startswith(" Changes Made"):
                lines = section.split("\n")[1:]
                changes_made = [line.strip("- ").strip() for line in lines if line.strip().startswith("-")]

            elif section.startswith("Rationale") or section.startswith(" Rationale"):
                lines = section.split("\n", 1)
                if len(lines) > 1:
                    rationale = lines[1].strip()

        return {
            "improved_prompt": improved_prompt,
            "changes_made": changes_made,
            "rationale": rationale
        }

    def _rule_based_rewrite(
        self,
        current_prompt: str,
        gaps: List[str],
        violations: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Fallback rule-based rewrite if no LLM is available"""
        changes_made = []
        improved_prompt = current_prompt

        # Add missing requirements
        if gaps:
            additions = "\n\n## Additional Requirements\n"
            for gap in gaps:
                additions += f"- {gap}\n"
            improved_prompt += additions
            changes_made.append(f"Added {len(gaps)} missing requirements")

        # Add notes about best practice violations
        if violations:
            violations_note = "\n\n## Best Practices to Follow\n"
            for v in violations[:3]:  # Top 3
                violations_note += f"- {v['description']}\n"
            improved_prompt += violations_note
            changes_made.append(f"Added notes about {len(violations)} best practice violations")

        return {
            "improved_prompt": improved_prompt,
            "changes_made": changes_made,
            "rationale": "Rule-based improvements applied (no LLM available)"
        }

    def _format_list(self, items: List[str]) -> str:
        """Format a list as markdown"""
        return "\n".join([f"- {item}" for item in items])

    def _format_violations(self, violations: List[Dict[str, str]]) -> str:
        """Format violations list"""
        return "\n".join([f"- **{v['name']}**: {v['description']}" for v in violations])

    def _format_focus_areas(self, focus_areas: Optional[List[str]]) -> str:
        """Format focus areas section"""
        if not focus_areas:
            return ""
        areas_list = self._format_list(focus_areas)
        return f"## Focus Areas\n{areas_list}"


    async def iterative_rewrite(
        self,
        current_prompt: str,
        user_feedback: str,
        requirements: "Requirements",
        previous_iterations: List[Dict[str, Any]] = None,
        iteration: int = 1,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform iterative refinement based on user feedback

        Args:
            current_prompt: The current version of the prompt
            user_feedback: User's feedback on what to improve
            requirements: User requirements
            previous_iterations: History of previous iterations
            iteration: Current iteration number
            provider: LLM provider to use
            api_key: API key for the provider (from database settings)
            model_name: Optional specific model to use

        Returns:
            Dictionary with improved_prompt, changes_made, rationale, improvement_score, suggestions_for_next
        """
        # Build context from previous iterations
        iteration_context = ""
        if previous_iterations:
            iteration_context = "\n\n## Previous Iteration History\n"
            for prev in previous_iterations[-3:]:  # Last 3 iterations for context
                iteration_context += f"""
### Iteration {prev.get('iteration', '?')}
- **Feedback**: {prev.get('feedback', 'N/A')}
- **Changes Made**: {', '.join(prev.get('changes_made', []))}
- **Improvement Score**: {prev.get('improvement_score', 'N/A')}
"""

        # Get provider-specific guidance
        provider_guidance = get_provider_guidance(requirements.target_provider)

        instruction = f"""You are an expert prompt engineer performing ITERATIVE REFINEMENT on a system prompt.

## Current Iteration: {iteration}

## Current System Prompt
```
{current_prompt}
```

## User Feedback for This Iteration
{user_feedback}

## Use Case
{requirements.use_case}

## Key Requirements
{self._format_list(requirements.key_requirements)}

## Target Provider
{requirements.target_provider}
{iteration_context}

## OFFICIAL PROVIDER BEST PRACTICES
{provider_guidance}

## ITERATIVE REFINEMENT INSTRUCTIONS

Based on the user's feedback, make TARGETED improvements to the prompt:

1. **Address Specific Feedback**: Focus on what the user asked for
2. **Preserve Working Elements**: Don't change parts that are already good
3. **Incremental Improvement**: Make meaningful but controlled changes
4. **Measure Progress**: Compare to previous iteration if available

## OUTPUT FORMAT

Provide your response in this EXACT format:

### Improved Prompt
```
[Your improved system prompt here - complete and ready to use]
```

### Changes Made
1. [Specific change #1]
2. [Specific change #2]
...

### Rationale
[2-3 sentences explaining how these changes address the user's feedback]

### Improvement Score
[A number from 0-100 indicating how much improvement was made this iteration. 0 = no improvement, 100 = major improvement]

### Suggestions for Next Iteration
- [Suggestion 1 for further improvement]
- [Suggestion 2 for further improvement]
- [Suggestion 3 for further improvement]
"""

        # Use the specified provider (only if API key is provided)
        if provider == "openai" and api_key:
            result = await self._iterative_rewrite_openai(instruction, api_key, model_name)
        elif provider == "claude" and api_key:
            result = await self._iterative_rewrite_claude(instruction, api_key, model_name)
        elif provider == "gemini" and api_key:
            result = await self._iterative_rewrite_gemini(instruction, api_key, model_name)
        else:
            result = self._basic_iterative_rewrite(current_prompt, user_feedback)

        result["iteration"] = iteration
        return result

    async def _iterative_rewrite_openai(self, instruction: str, api_key: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Use OpenAI for iterative rewrite"""
        try:
            client = openai.AsyncOpenAI(api_key=api_key)
            model = model_name or DEFAULT_MODELS["openai"]

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert prompt engineer specializing in iterative refinement."},
                    {"role": "user", "content": instruction}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            content = response.choices[0].message.content
            return self._parse_iterative_response(content)
        except Exception as e:
            return {
                "improved_prompt": "",
                "changes_made": [],
                "rationale": f"Error: {str(e)}",
                "improvement_score": 0,
                "suggestions_for_next": []
            }

    async def _iterative_rewrite_claude(self, instruction: str, api_key: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Use Claude for iterative rewrite"""
        try:
            client = anthropic.AsyncAnthropic(api_key=api_key)
            model = model_name or DEFAULT_MODELS["claude"]

            response = await client.messages.create(
                model=model,
                max_tokens=4000,
                messages=[{"role": "user", "content": instruction}]
            )
            content = response.content[0].text
            return self._parse_iterative_response(content)
        except Exception as e:
            return {
                "improved_prompt": "",
                "changes_made": [],
                "rationale": f"Error: {str(e)}",
                "improvement_score": 0,
                "suggestions_for_next": []
            }

    async def _iterative_rewrite_gemini(self, instruction: str, api_key: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Use Gemini for iterative rewrite"""
        try:
            genai.configure(api_key=api_key)
            model_id = model_name or DEFAULT_MODELS["gemini"]
            model = genai.GenerativeModel(model_id)

            response = await model.generate_content_async(instruction)
            content = response.text
            return self._parse_iterative_response(content)
        except Exception as e:
            return {
                "improved_prompt": "",
                "changes_made": [],
                "rationale": f"Error: {str(e)}",
                "improvement_score": 0,
                "suggestions_for_next": []
            }

    def _parse_iterative_response(self, content: str) -> Dict[str, Any]:
        """Parse the iterative rewrite response"""
        improved_prompt = ""
        changes_made = []
        rationale = ""
        improvement_score = 50.0
        suggestions_for_next = []

        sections = content.split("###")

        for section in sections:
            section = section.strip()

            if section.startswith("Improved Prompt") or section.startswith(" Improved Prompt"):
                lines = section.split("\n", 1)
                if len(lines) > 1:
                    improved_prompt = lines[1].strip()
                    improved_prompt = improved_prompt.replace("```", "").strip()

            elif section.startswith("Changes Made") or section.startswith(" Changes Made"):
                lines = section.split("\n")[1:]
                for line in lines:
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith("-")):
                        change = line.lstrip("0123456789.-) ").strip()
                        if change:
                            changes_made.append(change)

            elif section.startswith("Rationale") or section.startswith(" Rationale"):
                lines = section.split("\n", 1)
                if len(lines) > 1:
                    rationale = lines[1].strip()

            elif section.startswith("Improvement Score") or section.startswith(" Improvement Score"):
                lines = section.split("\n", 1)
                if len(lines) > 1:
                    score_text = lines[1].strip()
                    # Extract number from text
                    import re
                    numbers = re.findall(r'\d+', score_text)
                    if numbers:
                        improvement_score = float(numbers[0])

            elif section.startswith("Suggestions for Next") or section.startswith(" Suggestions for Next"):
                lines = section.split("\n")[1:]
                suggestions_for_next = [
                    line.strip("- ").strip()
                    for line in lines
                    if line.strip().startswith("-") and line.strip("- ").strip()
                ]

        return {
            "improved_prompt": improved_prompt,
            "changes_made": changes_made,
            "rationale": rationale,
            "improvement_score": improvement_score,
            "suggestions_for_next": suggestions_for_next
        }

    def _basic_iterative_rewrite(self, current_prompt: str, user_feedback: str) -> Dict[str, Any]:
        """Basic fallback for iterative rewrite"""
        improved_prompt = current_prompt + f"\n\n## User Feedback Integration\n{user_feedback}"
        return {
            "improved_prompt": improved_prompt,
            "changes_made": ["Added user feedback as inline notes"],
            "rationale": "Basic refinement applied (no LLM available)",
            "improvement_score": 10.0,
            "suggestions_for_next": ["Configure LLM settings for AI-powered refinement"]
        }

    async def compare_versions(
        self,
        prompt_a: str,
        prompt_b: str,
        requirements: "Requirements",
        provider: str = "openai",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare two prompt versions side-by-side using LLM analysis

        Args:
            prompt_a: First prompt version
            prompt_b: Second prompt version
            requirements: User requirements
            provider: LLM provider to use
            api_key: API key for the provider (from database settings)
            model_name: Optional specific model to use

        Returns:
            Dictionary with detailed comparison analysis
        """
        provider_guidance = get_provider_guidance(requirements.target_provider)

        instruction = f"""You are an expert prompt engineer performing A/B COMPARISON of two system prompts.

## Use Case
{requirements.use_case}

## Key Requirements
{self._format_list(requirements.key_requirements)}

## Target Provider
{requirements.target_provider}

## PROMPT VERSION A
```
{prompt_a}
```

## PROMPT VERSION B
```
{prompt_b}
```

## OFFICIAL BEST PRACTICES
{provider_guidance}

## COMPARISON TASK

Perform a detailed side-by-side comparison of these two prompts. Analyze:

1. **Requirements Coverage**: Which prompt better addresses the requirements?
2. **Best Practices Compliance**: Which follows provider guidelines better?
3. **Clarity & Structure**: Which is clearer and better organized?
4. **Completeness**: Which is more comprehensive?
5. **Conciseness**: Which achieves goals with less verbosity?

## OUTPUT FORMAT

Provide your response in this EXACT format:

### Scores
Version A:
- Requirements Alignment: [0-100]
- Best Practices: [0-100]
- Clarity: [0-100]
- Overall: [0-100]

Version B:
- Requirements Alignment: [0-100]
- Best Practices: [0-100]
- Clarity: [0-100]
- Overall: [0-100]

### Key Differences
1. [Specific difference #1 and which version handles it better]
2. [Specific difference #2 and which version handles it better]
3. [Specific difference #3 and which version handles it better]

### Recommendation
Winner: [Version A / Version B / Tie]
Confidence: [0-100]

### Detailed Analysis
[3-5 sentences explaining the recommendation with specific examples from both prompts]
"""

        if provider == "openai" and api_key:
            result = await self._compare_with_openai(instruction, api_key, model_name)
        elif provider == "claude" and api_key:
            result = await self._compare_with_claude(instruction, api_key, model_name)
        elif provider == "gemini" and api_key:
            result = await self._compare_with_gemini(instruction, api_key, model_name)
        else:
            result = self._basic_compare(prompt_a, prompt_b)

        return result

    async def _compare_with_openai(self, instruction: str, api_key: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Use OpenAI for comparison"""
        try:
            client = openai.AsyncOpenAI(api_key=api_key)
            model = model_name or DEFAULT_MODELS["openai"]

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert at comparing and evaluating system prompts."},
                    {"role": "user", "content": instruction}
                ],
                temperature=0.5,
                max_tokens=3000
            )
            content = response.choices[0].message.content
            return self._parse_comparison_response(content)
        except Exception as e:
            return {"error": str(e)}

    async def _compare_with_claude(self, instruction: str, api_key: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Use Claude for comparison"""
        try:
            client = anthropic.AsyncAnthropic(api_key=api_key)
            model = model_name or DEFAULT_MODELS["claude"]

            response = await client.messages.create(
                model=model,
                max_tokens=3000,
                messages=[{"role": "user", "content": instruction}]
            )
            content = response.content[0].text
            return self._parse_comparison_response(content)
        except Exception as e:
            return {"error": str(e)}

    async def _compare_with_gemini(self, instruction: str, api_key: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Use Gemini for comparison"""
        try:
            genai.configure(api_key=api_key)
            model_id = model_name or DEFAULT_MODELS["gemini"]
            model = genai.GenerativeModel(model_id)

            response = await model.generate_content_async(instruction)
            content = response.text
            return self._parse_comparison_response(content)
        except Exception as e:
            return {"error": str(e)}

    def _parse_comparison_response(self, content: str) -> Dict[str, Any]:
        """Parse the comparison response"""
        import re

        scores_a = {"requirements_alignment": 70, "best_practices": 70, "clarity": 70, "overall": 70}
        scores_b = {"requirements_alignment": 70, "best_practices": 70, "clarity": 70, "overall": 70}
        key_differences = []
        winner = "tie"
        confidence = 50.0
        detailed_analysis = ""

        sections = content.split("###")

        for section in sections:
            section = section.strip()

            if section.startswith("Scores") or section.startswith(" Scores"):
                lines = section.split("\n")
                current_version = None
                for line in lines:
                    line = line.strip()
                    if "Version A" in line:
                        current_version = "a"
                    elif "Version B" in line:
                        current_version = "b"
                    elif "Requirements" in line and current_version:
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            if current_version == "a":
                                scores_a["requirements_alignment"] = float(numbers[0])
                            else:
                                scores_b["requirements_alignment"] = float(numbers[0])
                    elif "Best Practices" in line and current_version:
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            if current_version == "a":
                                scores_a["best_practices"] = float(numbers[0])
                            else:
                                scores_b["best_practices"] = float(numbers[0])
                    elif "Clarity" in line and current_version:
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            if current_version == "a":
                                scores_a["clarity"] = float(numbers[0])
                            else:
                                scores_b["clarity"] = float(numbers[0])
                    elif "Overall" in line and current_version:
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            if current_version == "a":
                                scores_a["overall"] = float(numbers[0])
                            else:
                                scores_b["overall"] = float(numbers[0])

            elif section.startswith("Key Differences") or section.startswith(" Key Differences"):
                lines = section.split("\n")[1:]
                for line in lines:
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith("-")):
                        diff = line.lstrip("0123456789.-) ").strip()
                        if diff:
                            key_differences.append(diff)

            elif section.startswith("Recommendation") or section.startswith(" Recommendation"):
                lines = section.split("\n")
                for line in lines:
                    line_lower = line.lower()
                    if "winner" in line_lower:
                        if "version a" in line_lower:
                            winner = "version_a"
                        elif "version b" in line_lower:
                            winner = "version_b"
                        else:
                            winner = "tie"
                    elif "confidence" in line_lower:
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            confidence = float(numbers[0])

            elif section.startswith("Detailed Analysis") or section.startswith(" Detailed Analysis"):
                lines = section.split("\n", 1)
                if len(lines) > 1:
                    detailed_analysis = lines[1].strip()

        return {
            "scores_a": scores_a,
            "scores_b": scores_b,
            "key_differences": key_differences,
            "winner": winner,
            "confidence": confidence,
            "detailed_analysis": detailed_analysis
        }

    def _basic_compare(self, prompt_a: str, prompt_b: str) -> Dict[str, Any]:
        """Basic comparison without LLM"""
        len_a = len(prompt_a)
        len_b = len(prompt_b)

        return {
            "scores_a": {"requirements_alignment": 50, "best_practices": 50, "clarity": 50, "overall": 50},
            "scores_b": {"requirements_alignment": 50, "best_practices": 50, "clarity": 50, "overall": 50},
            "key_differences": [
                f"Length difference: A={len_a} chars, B={len_b} chars",
                "Detailed comparison requires LLM configuration"
            ],
            "winner": "tie",
            "confidence": 0,
            "detailed_analysis": "Basic comparison only. Configure LLM settings for detailed analysis."
        }


# Global instance
_prompt_rewriter = None

def get_prompt_rewriter() -> PromptRewriter:
    """Get the global prompt rewriter instance"""
    global _prompt_rewriter
    if _prompt_rewriter is None:
        _prompt_rewriter = PromptRewriter()
    return _prompt_rewriter
