"""
Evaluation prompt generator - creates evaluation prompts to test system prompts
Focuses on contextual relevance, groundedness, and answer relevance

Uses official best practices from (2025 releases):
- OpenAI GPT-5 & GPT-4.1: https://cookbook.openai.com/examples/gpt-5/gpt-5_prompting_guide
- Anthropic Claude Sonnet 4.5: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-4-best-practices
- Google Gemini 2.5: https://ai.google.dev/gemini-api/docs/prompting-strategies
"""
from typing import List, Dict, Any, Optional
import os
from models import Requirements
from official_best_practices import get_provider_guidance, get_full_best_practices_reference
import openai
import anthropic
import google.generativeai as genai


class EvalPromptGenerator:
    """Generates evaluation prompts to test system prompts using LLM"""

    async def generate_eval_prompt(
        self,
        system_prompt: str,
        requirements: Requirements,
        additional_scenarios: List[str] = None,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an evaluation prompt based on the system prompt and requirements using LLM

        Args:
            system_prompt: The finalized system prompt to test
            requirements: User requirements
            additional_scenarios: Optional additional test scenarios
            provider: LLM provider to use
            api_key: API key for the provider
            model_name: Optional specific model to use

        Returns:
            Dictionary with eval_prompt, rationale, and test_scenarios

        Raises:
            ValueError: If no API key is provided (LLM is required)
        """
        # LLM-based generation is required - no template fallback
        if not api_key:
            raise ValueError("API key is required. Please configure LLM settings first (click the gear icon).")

        return await self._generate_with_llm(
            system_prompt=system_prompt,
            requirements=requirements,
            additional_scenarios=additional_scenarios,
            provider=provider,
            api_key=api_key,
            model_name=model_name
        )

    async def _generate_with_llm(
        self,
        system_prompt: str,
        requirements: Requirements,
        additional_scenarios: Optional[List[str]],
        provider: str,
        api_key: str,
        model_name: Optional[str]
    ) -> Dict[str, Any]:
        """Generate evaluation prompt using LLM"""

        # Detect system type for appropriate evaluation criteria
        system_type = self._detect_system_type(system_prompt, requirements.use_case)

        # Build the generation prompt
        generation_prompt = self._build_llm_generation_prompt(
            system_prompt=system_prompt,
            requirements=requirements,
            additional_scenarios=additional_scenarios
        )

        # Always use LLM - no fallback to templates
        if provider == "openai":
            result = await self._generate_eval_with_openai(generation_prompt, api_key, model_name)
        elif provider == "claude":
            result = await self._generate_eval_with_claude(generation_prompt, api_key, model_name)
        elif provider == "gemini":
            result = await self._generate_eval_with_gemini(generation_prompt, api_key, model_name)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. Supported providers: openai, claude, gemini")

        return self._parse_eval_generation_response(result, requirements, system_type)

    def _extract_template_variables(self, system_prompt: str) -> List[str]:
        """
        Extract template variables from the system prompt.
        Template variables are in the format {{variable_name}} or {variable_name}
        """
        import re
        # Match both {{var}} and {var} patterns, but not JSON-like structures
        double_brace = re.findall(r'\{\{(\w+)\}\}', system_prompt)
        # For single braces, be more careful to avoid JSON
        single_brace = re.findall(r'(?<!\{)\{(\w+)\}(?!\})', system_prompt)

        # Combine and deduplicate, preserving order
        all_vars = []
        seen = set()
        for var in double_brace + single_brace:
            if var not in seen and var.lower() not in ['score', 'json', 'object']:
                all_vars.append(var)
                seen.add(var)

        return all_vars

    def _get_template_variables_guidance(self, variables: List[str], system_prompt: str) -> str:
        """Generate guidance for using template variables in the eval prompt"""
        if not variables:
            return ""

        var_descriptions = []
        for var in variables:
            # Try to infer what the variable is for based on context
            var_lower = var.lower()
            if var_lower in ['input', 'user_input', 'query', 'user_query', 'question']:
                desc = "The user's input/query to the system"
            elif var_lower in ['output', 'response', 'result', 'answer']:
                desc = "The AI's generated response"
            elif var_lower in ['context', 'document', 'documents', 'knowledge', 'reference']:
                desc = "Context or reference documents provided"
            elif var_lower in ['custom_instructions', 'instructions', 'criteria', 'rules']:
                desc = "Custom instructions or criteria from the user"
            elif var_lower in ['history', 'conversation', 'chat_history']:
                desc = "Previous conversation history"
            else:
                desc = "Dynamic content injected at runtime"
            var_descriptions.append(f"  - `{{{{{var}}}}}`: {desc}")

        return f"""
## TEMPLATE VARIABLES DETECTED IN SYSTEM PROMPT

The system prompt contains the following template variables that get replaced with dynamic content at runtime:
{chr(10).join(var_descriptions)}

**CRITICAL**: Your evaluation prompt MUST:
1. Use these EXACT variable names (with double braces {{{{}}}}) in the evaluation inputs section
2. Reference these variables when defining what to evaluate
3. The eval prompt should test how well the AI handles different values of these variables

For example, if the system has `{{{{custom_instructions}}}}`, your eval should check if the AI follows those custom instructions.
"""

    def _detect_system_type(self, system_prompt: str, use_case: str) -> str:
        """
        Detect if the system is interactive (chatbot/copilot) or non-interactive (generator/extractor)

        Returns: 'interactive' or 'non_interactive'
        """
        combined_text = f"{system_prompt} {use_case}".lower()

        # Keywords indicating interactive/conversational systems
        interactive_keywords = [
            'chat', 'chatbot', 'conversation', 'conversational', 'assistant', 'copilot',
            'dialogue', 'dialog', 'multi-turn', 'follow-up', 'user interaction',
            'help desk', 'support agent', 'customer service', 'virtual assistant',
            'ai assistant', 'personal assistant', 'talk to', 'speak with',
            'interactive', 'respond to user', 'answer questions', 'q&a',
            'helpdesk', 'live chat', 'messaging', 'real-time'
        ]

        # Keywords indicating non-interactive/batch processing systems
        non_interactive_keywords = [
            'generate', 'generator', 'summarize', 'summarizer', 'summary',
            'extract', 'extraction', 'extractor', 'transform', 'convert',
            'analyze text', 'process document', 'parse', 'classify', 'classification',
            'translate', 'translation', 'rewrite', 'paraphrase',
            'story', 'content creation', 'article', 'report generation',
            'action items', 'key points', 'bullet points', 'tldr',
            'sentiment analysis', 'entity extraction', 'ner', 'tagging',
            'single input', 'batch', 'one-shot', 'stateless'
        ]

        interactive_score = sum(1 for kw in interactive_keywords if kw in combined_text)
        non_interactive_score = sum(1 for kw in non_interactive_keywords if kw in combined_text)

        # Default to non-interactive if unclear (simpler evaluation)
        if interactive_score > non_interactive_score:
            return 'interactive'
        return 'non_interactive'

    def _get_system_type_guidance(self, system_type: str) -> str:
        """Get evaluation guidance specific to the system type"""

        if system_type == 'interactive':
            return """
## SYSTEM TYPE: INTERACTIVE (Chatbot/Copilot/Assistant)

This is an **interactive conversational system** where users engage in multi-turn dialogues.

### Special Evaluation Considerations for Interactive Systems:

1. **Conversation Context Handling**
   - Does the response acknowledge and build upon previous turns?
   - Is context from earlier messages properly retained?
   - Are references to prior statements handled correctly?

2. **Persona Consistency**
   - Does the assistant maintain a consistent personality across turns?
   - Is the tone appropriate for the conversation stage?
   - Does it remember user preferences mentioned earlier?

3. **Turn-by-Turn Coherence**
   - Is each response appropriately scoped for the current turn?
   - Does it avoid repeating information already provided?
   - Are follow-up questions handled naturally?

4. **Conversation Flow**
   - Does the response move the conversation forward productively?
   - Are clarifying questions asked when appropriate?
   - Is the conversation kept on track without being rigid?

5. **Context Switching**
   - How well does it handle topic changes within conversation?
   - Can it return to previous topics when referenced?

### Input Format for Evaluation
The `{input}` will contain conversation history in this format:
```
[Previous turns if any]
User: [current user message]
```

### Additional Rating Criteria for Interactive Systems:
- **Rating 5**: Perfect context retention, natural conversation flow, consistent persona
- **Rating 4**: Good context handling with minor gaps
- **Rating 3**: Basic context awareness but some inconsistencies
- **Rating 2**: Poor context retention, persona drift
- **Rating 1**: Ignores conversation history, incoherent responses
"""
        else:
            return """
## SYSTEM TYPE: NON-INTERACTIVE (Generator/Extractor/Transformer)

This is a **non-interactive processing system** that takes a single input and produces a single output.

### Special Evaluation Considerations for Non-Interactive Systems:

1. **Output Completeness**
   - Does the output fully address all aspects of the input?
   - Is all required information extracted/generated?
   - Are there any missing elements?

2. **Format Compliance**
   - Does the output match the expected format exactly?
   - Are structural requirements (JSON, bullet points, etc.) met?
   - Is the output properly formatted and parseable?

3. **Accuracy & Faithfulness**
   - Is the output grounded in the input content?
   - Are there any hallucinations or fabricated information?
   - Does it accurately represent the source material?

4. **Transformation Quality**
   - How well does the output serve its intended purpose?
   - Is the transformation (summary, extraction, generation) effective?
   - Does it add appropriate value beyond the raw input?

5. **Standalone Validity**
   - Can the output be used independently?
   - Is it self-contained and complete?
   - Does it make sense without the original input context?

### Input Format for Evaluation
The `{input}` will contain the source content to be processed:
```
[Document/text/data to be processed]
```

### Additional Rating Criteria for Non-Interactive Systems:
- **Rating 5**: Perfect format, complete extraction, accurate transformation
- **Rating 4**: Minor format issues or small omissions
- **Rating 3**: Correct but incomplete or poorly formatted
- **Rating 2**: Significant accuracy issues or missing key elements
- **Rating 1**: Wrong format, hallucinations, or fundamentally incorrect output
"""

    def _build_llm_generation_prompt(
        self,
        system_prompt: str,
        requirements: Requirements,
        additional_scenarios: Optional[List[str]]
    ) -> str:
        """Build the prompt for LLM to generate evaluation prompt"""

        requirements_list = "\n".join([f"- {req}" for req in requirements.key_requirements])
        scenarios_text = ""
        if additional_scenarios:
            scenarios_text = f"\n\n## Additional Test Scenarios to Cover\n" + "\n".join([f"- {s}" for s in additional_scenarios])

        # Get provider-specific guidance for the eval prompt structure
        provider_guidance = get_provider_guidance(requirements.target_provider)

        # Build requirement-specific evaluation criteria
        requirement_criteria = self._build_requirement_criteria_for_llm(requirements)

        # Detect system type and get specific guidance
        system_type = self._detect_system_type(system_prompt, requirements.use_case)
        system_type_guidance = self._get_system_type_guidance(system_type)

        # Extract template variables from the system prompt
        template_variables = self._extract_template_variables(system_prompt)
        template_variables_guidance = self._get_template_variables_guidance(template_variables, system_prompt)

        return f"""You are an expert in LLM evaluation and quality assurance. Your task is to create a comprehensive evaluation prompt that will be used to assess responses generated by an AI assistant.

## SYSTEM PROMPT TO EVALUATE
<system_prompt>
{system_prompt}
</system_prompt>

## USE CASE
{requirements.use_case}

## KEY REQUIREMENTS TO EVALUATE
{requirements_list}

## EXPECTED BEHAVIOR
{requirements.expected_behavior or "Follow the system prompt instructions accurately"}

## TARGET PROVIDER
{requirements.target_provider}
{scenarios_text}

{system_type_guidance}
{template_variables_guidance}
## OFFICIAL PROVIDER BEST PRACTICES

When creating the evaluation prompt, follow these official guidelines for {requirements.target_provider}:

{provider_guidance}

---

## EXEMPLARY EVAL PROMPT STRUCTURE

Create an evaluation prompt following this EXACT 5-section structure:

### **I. Evaluator's Role & Goal**
- Define the evaluator's primary role (e.g., "meticulous quality assurance specialist")
- State the specific goal: rigorously evaluate AI responses against core operational principles
- Be specific to the use case: {requirements.use_case}

### **II. Core Expectations (Reference for Evaluation)**
Define what a high-quality output must do. For this use case, include:
{requirement_criteria}

### **III. Evaluation Criteria (Rating Scale)**

Create a **detailed 1-5 rating scale** with SPECIFIC failure modes for each level:

**Rating 1 (Very Poor):** Critical failures including:
- Hallucination: Response contains significant factual information not grounded in provided context
- Complete irrelevance: Response doesn't address the user's query
- Broken format: Output doesn't match expected structure
- Major requirement violations

**Rating 2 (Poor):** Major violations including:
- Incorrect logic or reasoning
- Partially ungrounded responses
- Significant requirement gaps
- Missing conditional handling

**Rating 3 (Adequate):** Functionally correct but with notable flaws:
- Technically correct but unhelpful
- Irrelevant or weak supplementary content
- Minor formatting or clarity issues
- Missed optimization opportunities

**Rating 4 (Good):** Correct with minor room for improvement:
- Slightly verbose or could be clearer
- All core requirements met
- Minor polish needed

**Rating 5 (Excellent):** Flawless execution:
- Perfectly addresses all requirements
- Grounded, relevant, and well-structured
- Optimal handling of all cases

### **IV. Evaluation Task**
Provide step-by-step instructions:
1. Analyze the #INPUT (user query, context, etc.)
2. Examine the #OUTPUT (AI assistant's response)
3. Compare against Core Expectations and Rating Scale
4. Assign a single numerical rating (1-5)
5. Write specific rationale with examples from the output

### **V. Output Format**
Specify exact JSON structure:
```json
{{
  "score": <1-5 rating>,
  "reasoning": "Specific explanation with examples from the output"
}}
```

---

## REQUIREMENTS FOR THE EVAL PROMPT

The evaluation prompt you create MUST:

1. **Use the correct placeholders for evaluation inputs**:
   - `{{{{input}}}}`: The full input provided to the AI (user query + context)
   - `{{{{output}}}}`: The AI assistant's response to evaluate
   - If the system prompt has OTHER template variables (like `{{{{custom_instructions}}}}`), include them in the eval inputs section as they may affect how the output should be judged

2. **Follow the 5-section structure** exactly as shown above

3. **Include use-case-specific criteria** based on these {len(requirements.key_requirements)} requirements:
{requirements_list}

4. **Define specific failure modes** for each rating level (1-5)

5. **Output structured JSON** with score and reasoning

6. **Be self-contained** - Include all context needed to evaluate without external references

7. **Be specific, not generic** - Tailor all criteria to the actual system prompt and use case

---

## OUTPUT FORMAT

Return your response in this EXACT format:

### EVALUATION PROMPT
```
[Your complete evaluation prompt here - ready to use as-is, with all 5 sections]
```

### RATIONALE
[2-3 sentences explaining your evaluation approach]

### TEST SCENARIOS
- [Scenario 1 this prompt tests]
- [Scenario 2 this prompt tests]
- [Scenario 3 this prompt tests]
"""

    def _build_requirement_criteria_for_llm(self, requirements: Requirements) -> str:
        """Build requirement-specific criteria for the LLM to incorporate"""
        criteria = []

        for i, req in enumerate(requirements.key_requirements, 1):
            criteria.append(f"- **Requirement {i}**: {req}")

        if requirements.expected_behavior:
            criteria.append(f"- **Expected Behavior**: {requirements.expected_behavior}")

        if requirements.constraints:
            for key, value in requirements.constraints.items():
                criteria.append(f"- **Constraint ({key})**: {value}")

        return "\n".join(criteria)

    async def _generate_eval_with_openai(
        self,
        generation_prompt: str,
        api_key: str,
        model_name: Optional[str]
    ) -> str:
        """Generate eval prompt using OpenAI with fallback to gpt-4o"""
        import logging
        client = openai.AsyncOpenAI(api_key=api_key)

        async def try_model(model: str) -> str:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in creating evaluation prompts for LLM systems. Create detailed, specific evaluation criteria."
                    },
                    {"role": "user", "content": generation_prompt}
                ],
                temperature=0.7,
                max_tokens=8000
            )
            return response.choices[0].message.content

        # Try with specified model first
        primary_model = model_name or "gpt-4o"
        try:
            return await try_model(primary_model)
        except Exception as e:
            if primary_model != "gpt-4o":
                logging.warning(f"Model {primary_model} failed: {e}. Falling back to gpt-4o")
                return await try_model("gpt-4o")
            raise

    async def _generate_eval_with_claude(
        self,
        generation_prompt: str,
        api_key: str,
        model_name: Optional[str]
    ) -> str:
        """Generate eval prompt using Claude"""
        client = anthropic.AsyncAnthropic(api_key=api_key)

        response = await client.messages.create(
            model=model_name or "claude-3-5-sonnet-20241022",
            max_tokens=8000,
            messages=[
                {"role": "user", "content": generation_prompt}
            ]
        )

        return response.content[0].text

    async def _generate_eval_with_gemini(
        self,
        generation_prompt: str,
        api_key: str,
        model_name: Optional[str]
    ) -> str:
        """Generate eval prompt using Gemini"""
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name or 'gemini-2.5-pro')

        response = await model.generate_content_async(generation_prompt)

        return response.text

    def _parse_eval_generation_response(
        self,
        llm_response: str,
        requirements: Requirements,
        system_type: str = "non_interactive"
    ) -> Dict[str, Any]:
        """Parse the LLM response to extract eval prompt, rationale, and scenarios"""
        eval_prompt = ""
        rationale = ""
        test_scenarios = []

        # Split by sections
        sections = llm_response.split("###")

        for section in sections:
            section = section.strip()

            if section.upper().startswith("EVALUATION PROMPT") or section.upper().startswith(" EVALUATION PROMPT"):
                lines = section.split("\n", 1)
                if len(lines) > 1:
                    eval_prompt = lines[1].strip()
                    # Remove markdown code blocks if present
                    if eval_prompt.startswith("```"):
                        parts = eval_prompt.split("```")
                        if len(parts) >= 2:
                            eval_prompt = parts[1].strip()
                            if eval_prompt.startswith("markdown") or eval_prompt.startswith("\n"):
                                eval_prompt = eval_prompt.split("\n", 1)[-1].strip()
                    if eval_prompt.endswith("```"):
                        eval_prompt = eval_prompt[:-3].strip()

            elif section.upper().startswith("RATIONALE") or section.upper().startswith(" RATIONALE"):
                lines = section.split("\n", 1)
                if len(lines) > 1:
                    rationale = lines[1].strip()

            elif section.upper().startswith("TEST SCENARIOS") or section.upper().startswith(" TEST SCENARIOS"):
                lines = section.split("\n")[1:]
                test_scenarios = [
                    line.strip("- ").strip()
                    for line in lines
                    if line.strip().startswith("-") and line.strip("- ").strip()
                ]

        # Format system type for display
        system_type_display = "Interactive (Chatbot/Copilot)" if system_type == "interactive" else "Non-Interactive (Generator/Processor)"

        # If parsing failed, use template fallback
        if not eval_prompt:
            return self._generate_template_based(
                system_prompt="",  # Not available here
                requirements=requirements,
                additional_scenarios=None
            )

        # Prepend system type info to rationale
        system_type_note = f"**Detected System Type:** {system_type_display}\n\n"
        full_rationale = system_type_note + (rationale or "LLM-generated evaluation prompt tailored to the specific use case and requirements.")

        return {
            "eval_prompt": eval_prompt,
            "rationale": full_rationale,
            "test_scenarios": test_scenarios or self._identify_test_scenarios(requirements, None),
            "generation_method": "llm",
            "system_type": system_type,
            "system_type_display": system_type_display
        }

    def _generate_template_based(
        self,
        system_prompt: str,
        requirements: Requirements,
        additional_scenarios: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Fallback template-based generation"""
        # Extract key aspects to test
        test_scenarios = self._identify_test_scenarios(requirements, additional_scenarios)

        # Generate the evaluation prompt
        eval_prompt = self._construct_eval_prompt(
            system_prompt,
            requirements,
            test_scenarios
        )

        # Generate rationale
        rationale = self._generate_rationale(requirements, test_scenarios)

        return {
            "eval_prompt": eval_prompt,
            "rationale": rationale,
            "test_scenarios": test_scenarios,
            "generation_method": "template"
        }

    def _identify_test_scenarios(
        self,
        requirements: Requirements,
        additional_scenarios: List[str] = None
    ) -> List[str]:
        """Identify test scenarios based on requirements"""
        scenarios = []

        # Core evaluation dimensions
        scenarios.append("Contextual relevance - Does the response align with the user's intent and context?")
        scenarios.append("Groundedness - Is the response factually accurate and based on verifiable information?")
        scenarios.append("Answer relevance - Does the response directly address the user's question?")

        # Add requirement-specific scenarios
        for req in requirements.key_requirements[:5]:
            scenarios.append(f"Requirement adherence: {req}")

        # Add additional scenarios
        if additional_scenarios:
            scenarios.extend(additional_scenarios)

        return scenarios

    def _construct_eval_prompt(
        self,
        system_prompt: str,
        requirements: Requirements,
        test_scenarios: List[str]
    ) -> str:
        """Construct the evaluation prompt following the exemplary 5-section structure"""

        # Build requirement-specific expectations
        core_expectations = self._build_core_expectations(requirements)

        eval_prompt = f"""### **I. Evaluator's Role & Goal**

Your primary role is to act as a meticulous quality assurance specialist. Your goal is to rigorously evaluate the response generated by an AI assistant to determine if it strictly adheres to its core operational principles and the specified requirements.

**Use Case Being Evaluated:** {requirements.use_case}

**System Prompt Being Tested:**
```
{system_prompt}
```

### **II. Core Expectations (Reference for Evaluation)**

A high-quality output must strictly follow these rules. The AI assistant is expected to:

{core_expectations}

### **III. Evaluation Criteria (How the Evaluator Assesses #OUTPUT)**

You will be provided with the following information:
*   **#INPUT - {{{{input}}}}:** This contains the full input provided to the AI assistant, including the user's query and any context.
*   **#OUTPUT - {{{{output}}}}:** This is the response generated by the AI assistant.

Carefully assess the `#OUTPUT` against the `#INPUT` based on the **Core Expectations**. Use the following rating scale to score the output's quality and adherence to instructions.

**Rating Scale Definitions (1-5):**

*   **Rating 1 (Very Poor):** The output has critical failures. This includes:
    *   **Hallucination:** The response contains significant factual information not present in the provided context or not grounded in reality.
    *   **Complete Irrelevance:** The response does not address the user's query at all.
    *   **Broken Format:** The output doesn't match expected structure or is malformed.
    *   **Major Requirement Violations:** Core requirements are completely ignored or violated.

*   **Rating 2 (Poor):** The output has major violations. This includes:
    *   **Incorrect Logic:** The response contains flawed reasoning or contradictions.
    *   **Partially Ungrounded:** The response contains minor pieces of information not supported by context.
    *   **Significant Requirement Gaps:** Multiple requirements are not addressed or poorly implemented.
    *   **Missing Conditional Handling:** Edge cases or error conditions are not handled appropriately.

*   **Rating 3 (Adequate):** The output is functionally correct but has notable flaws. This includes:
    *   **Technically Correct but Unhelpful:** The answer addresses the query but fails to synthesize information effectively.
    *   **Weak Supplementary Content:** Follow-up suggestions or additional content is irrelevant or poorly chosen.
    *   **Minor Formatting Issues:** The response is poorly structured or hard to read.
    *   **Missed Optimization:** Could be more concise, clearer, or better organized.

*   **Rating 4 (Good):** The output is correct and follows all major rules but has minor room for improvement. This could include:
    *   The response is slightly verbose or could be clearer.
    *   All core requirements are met with minor polish needed.
    *   The response is helpful but not exceptional.

*   **Rating 5 (Excellent):** The output is flawless. It perfectly adheres to all instructions:
    *   All requirements are fully met with excellence.
    *   The response is grounded, relevant, and well-structured.
    *   Optimal handling of the query with clear, actionable content.
    *   Professional tone and appropriate detail level.

### **IV. Evaluation Task**

1.  Analyze the `#INPUT` (the user's query and any provided context).
2.  Examine the `#OUTPUT` (the AI assistant's response).
3.  Compare the `#OUTPUT` against the **Core Expectations** and the **Rating Scale Definitions**.
4.  Assign a single numerical rating from 1 to 5.
5.  Write a brief but specific rationale for your rating, explaining *why* the output earned that score. Cite specific examples from the `#OUTPUT` to support your reasoning.

### **V. Output Format**

Provide your evaluation in the following JSON format:

```json
{{
  "score": <Your 1-5 rating>,
  "reasoning": "Your brief explanation here. Be specific and provide examples from the output to justify your rating."
}}
```

---

**EVALUATION INPUTS:**
- #INPUT: {{{{input}}}}
- #OUTPUT: {{{{output}}}}
"""
        return eval_prompt

    def _build_core_expectations(self, requirements: Requirements) -> str:
        """Build core expectations section based on requirements"""
        expectations = []

        # Add each requirement as an expectation
        for i, req in enumerate(requirements.key_requirements, 1):
            expectations.append(f"*   **Requirement {i}:** {req}")

        # Add expected behavior if specified
        if requirements.expected_behavior:
            expectations.append(f"*   **Expected Behavior:** {requirements.expected_behavior}")

        # Add constraints if specified
        if requirements.constraints:
            for key, value in requirements.constraints.items():
                expectations.append(f"*   **Constraint ({key}):** {value}")

        # Add standard quality expectations
        expectations.extend([
            "*   **Be Relevant:** The response must directly address the user's query.",
            "*   **Be Grounded:** All factual claims must be accurate and verifiable. The assistant must not hallucinate.",
            "*   **Be Clear:** The response must be well-structured, easy to understand, and appropriately detailed.",
            "*   **Handle Edge Cases:** If the query cannot be fully answered, the assistant must clearly state limitations."
        ])

        return "\n".join(expectations)

    async def generate_eval_prompt_with_examples(
        self,
        system_prompt: str,
        requirements: Requirements,
        additional_scenarios: List[str] = None,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        num_examples: int = 3
    ) -> Dict[str, Any]:
        """
        Generate an evaluation prompt WITH few-shot calibration examples

        Args:
            system_prompt: The finalized system prompt to test
            requirements: User requirements
            additional_scenarios: Optional additional test scenarios
            provider: LLM provider to use
            api_key: API key for the provider
            model_name: Optional specific model to use
            num_examples: Number of examples per rating category

        Returns:
            Dictionary with eval_prompt, rationale, test_scenarios, and calibration_examples
        """
        # First generate the base eval prompt
        base_result = await self.generate_eval_prompt(
            system_prompt=system_prompt,
            requirements=requirements,
            additional_scenarios=additional_scenarios,
            provider=provider,
            api_key=api_key,
            model_name=model_name
        )

        # Now generate calibration examples
        calibration_examples = await self._generate_calibration_examples(
            system_prompt=system_prompt,
            requirements=requirements,
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            num_examples=num_examples
        )

        # Build eval prompt with few-shot examples
        eval_prompt_with_examples = self._build_eval_prompt_with_examples(
            base_eval_prompt=base_result["eval_prompt"],
            calibration_examples=calibration_examples
        )

        return {
            "eval_prompt": eval_prompt_with_examples,
            "rationale": base_result["rationale"] + " Includes calibration examples for score alignment.",
            "test_scenarios": base_result["test_scenarios"],
            "calibration_examples": calibration_examples,
            "generation_method": base_result.get("generation_method", "llm") + "_with_examples"
        }

    async def _generate_calibration_examples(
        self,
        system_prompt: str,
        requirements: Requirements,
        provider: str,
        api_key: Optional[str],
        model_name: Optional[str],
        num_examples: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate gold-standard calibration examples for each rating level"""

        if not api_key:
            return self._get_default_calibration_examples(requirements)

        requirements_list = "\n".join([f"- {req}" for req in requirements.key_requirements])

        instruction = f"""You are an expert at creating calibration examples for LLM evaluation.

## System Prompt Being Evaluated
```
{system_prompt}
```

## Use Case
{requirements.use_case}

## Key Requirements
{requirements_list}

## TASK

Generate {num_examples} calibration examples for EACH of the 5 rating levels (1-5).
These examples should serve as "gold standard" references for evaluators to calibrate their scoring.

For each example, provide:
1. A realistic user INPUT
2. An AI OUTPUT that matches the rating level
3. The expected SCORE (1-5)
4. A brief REASONING explaining why this output deserves this score

## RATING LEVEL DEFINITIONS

- **Rating 5 (Excellent)**: Flawless execution, all requirements met perfectly
- **Rating 4 (Good)**: Correct with minor room for improvement
- **Rating 3 (Adequate)**: Functionally correct but with notable flaws
- **Rating 2 (Poor)**: Major violations or significant gaps
- **Rating 1 (Very Poor)**: Critical failures, hallucinations, or complete irrelevance

## OUTPUT FORMAT

Return a JSON array with exactly {num_examples * 5} examples:

```json
[
  {{
    "input": "Example user query",
    "output": "Example AI response",
    "expected_score": 5,
    "reasoning": "This response is excellent because...",
    "category": "excellent"
  }},
  {{
    "input": "Another example query",
    "output": "A mediocre response...",
    "expected_score": 3,
    "reasoning": "This response is adequate because...",
    "category": "adequate"
  }}
]
```

Ensure examples are:
- Realistic for the use case
- Clearly differentiated between rating levels
- Helpful for calibrating evaluator expectations
"""

        try:
            if provider == "openai":
                examples = await self._generate_examples_openai(instruction, api_key, model_name)
            elif provider == "claude":
                examples = await self._generate_examples_claude(instruction, api_key, model_name)
            elif provider == "gemini":
                examples = await self._generate_examples_gemini(instruction, api_key, model_name)
            else:
                examples = self._get_default_calibration_examples(requirements)

            return examples

        except Exception as e:
            print(f"Failed to generate calibration examples: {e}")
            return self._get_default_calibration_examples(requirements)

    async def _generate_examples_openai(self, instruction: str, api_key: str, model_name: Optional[str]) -> List[Dict]:
        """Generate examples using OpenAI with fallback to gpt-4o"""
        import json
        import logging
        client = openai.AsyncOpenAI(api_key=api_key)

        async def try_model(model: str) -> str:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You generate calibration examples for LLM evaluation. Always respond with valid JSON."},
                    {"role": "user", "content": instruction}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            return response.choices[0].message.content

        primary_model = model_name or "gpt-4o"
        try:
            content = await try_model(primary_model)
        except Exception as e:
            if primary_model != "gpt-4o":
                logging.warning(f"Model {primary_model} failed: {e}. Falling back to gpt-4o")
                content = await try_model("gpt-4o")
            else:
                raise

        return self._parse_examples_json(content)

    async def _generate_examples_claude(self, instruction: str, api_key: str, model_name: Optional[str]) -> List[Dict]:
        """Generate examples using Claude"""
        client = anthropic.AsyncAnthropic(api_key=api_key)

        response = await client.messages.create(
            model=model_name or "claude-3-5-sonnet-20241022",
            max_tokens=4000,
            messages=[{"role": "user", "content": instruction}]
        )

        content = response.content[0].text
        return self._parse_examples_json(content)

    async def _generate_examples_gemini(self, instruction: str, api_key: str, model_name: Optional[str]) -> List[Dict]:
        """Generate examples using Gemini"""
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name or 'gemini-2.5-pro')

        response = await model.generate_content_async(instruction)
        content = response.text
        return self._parse_examples_json(content)

    def _parse_examples_json(self, content: str) -> List[Dict]:
        """Parse JSON examples from LLM response"""
        import json
        import re

        # Try to extract JSON array from response
        json_match = re.search(r'\[[\s\S]*\]', content)
        if json_match:
            try:
                examples = json.loads(json_match.group())
                # Validate structure
                validated = []
                for ex in examples:
                    if all(k in ex for k in ["input", "output", "expected_score", "reasoning"]):
                        validated.append({
                            "input": ex["input"],
                            "output": ex["output"],
                            "expected_score": int(ex["expected_score"]),
                            "reasoning": ex["reasoning"],
                            "category": ex.get("category", self._score_to_category(ex["expected_score"]))
                        })
                return validated
            except json.JSONDecodeError:
                pass

        return []

    def _score_to_category(self, score: int) -> str:
        """Convert score to category name"""
        categories = {5: "excellent", 4: "good", 3: "adequate", 2: "poor", 1: "very_poor"}
        return categories.get(score, "unknown")

    def _get_default_calibration_examples(self, requirements: Requirements) -> List[Dict[str, Any]]:
        """Fallback calibration examples when LLM is not available"""
        use_case = requirements.use_case

        return [
            # Rating 5 - Excellent
            {
                "input": f"Help me with {use_case}",
                "output": f"I'd be happy to help you with {use_case}. Based on your requirements, here's a comprehensive solution that addresses all your needs: [detailed, accurate, well-structured response that fully addresses the query]",
                "expected_score": 5,
                "reasoning": "The response is comprehensive, directly addresses the query, follows all guidelines, and provides actionable value. No hallucinations or irrelevant content.",
                "category": "excellent"
            },
            # Rating 4 - Good
            {
                "input": f"Can you assist with {use_case}?",
                "output": f"Yes, I can help with {use_case}. Here's what you need to know: [mostly complete response with minor verbosity]",
                "expected_score": 4,
                "reasoning": "The response correctly addresses the query and follows guidelines, but could be more concise or slightly better organized.",
                "category": "good"
            },
            # Rating 3 - Adequate
            {
                "input": f"Tell me about {use_case}",
                "output": f"Here's some information about {use_case}: [basic response that addresses the query but misses some nuances or could be better structured]",
                "expected_score": 3,
                "reasoning": "The response is functionally correct but lacks depth, has formatting issues, or doesn't fully leverage the available context.",
                "category": "adequate"
            },
            # Rating 2 - Poor
            {
                "input": f"I need help with {use_case}",
                "output": f"Sure, {use_case} is interesting. [response that partially addresses the query but has significant gaps or minor factual issues]",
                "expected_score": 2,
                "reasoning": "The response has major issues: either doesn't fully address the query, contains minor inaccuracies, or violates some requirements.",
                "category": "poor"
            },
            # Rating 1 - Very Poor
            {
                "input": f"Explain {use_case} to me",
                "output": "I'm not sure about that. [irrelevant or completely wrong response with hallucinated facts]",
                "expected_score": 1,
                "reasoning": "Critical failure: the response contains hallucinations, is completely irrelevant, or fundamentally misunderstands the query.",
                "category": "very_poor"
            }
        ]

    def _build_eval_prompt_with_examples(
        self,
        base_eval_prompt: str,
        calibration_examples: List[Dict[str, Any]]
    ) -> str:
        """Build the final eval prompt with few-shot calibration examples"""

        # Group examples by category
        examples_by_category = {}
        for ex in calibration_examples:
            category = ex.get("category", "unknown")
            if category not in examples_by_category:
                examples_by_category[category] = []
            examples_by_category[category].append(ex)

        # Build examples section
        examples_section = """
### **Calibration Examples (Gold Standard References)**

Use these examples to calibrate your scoring. Each example shows what a response at that rating level looks like.

"""
        category_order = ["excellent", "good", "adequate", "poor", "very_poor"]
        score_map = {"excellent": 5, "good": 4, "adequate": 3, "poor": 2, "very_poor": 1}

        for category in category_order:
            if category in examples_by_category:
                score = score_map.get(category, 3)
                examples_section += f"""
#### Rating {score} ({category.replace('_', ' ').title()}) Example

"""
                for i, ex in enumerate(examples_by_category[category][:2], 1):  # Max 2 per category
                    examples_section += f"""**Example {i}:**
- **Input:** "{ex['input']}"
- **Output:** "{ex['output'][:200]}{'...' if len(ex['output']) > 200 else ''}"
- **Expected Score:** {ex['expected_score']}
- **Reasoning:** {ex['reasoning']}

"""

        # Insert examples section before the Output Format section
        if "### **V. Output Format**" in base_eval_prompt:
            parts = base_eval_prompt.split("### **V. Output Format**")
            return parts[0] + examples_section + "### **V. Output Format**" + parts[1]
        elif "**EVALUATION INPUTS:**" in base_eval_prompt:
            parts = base_eval_prompt.split("**EVALUATION INPUTS:**")
            return parts[0] + examples_section + "\n**EVALUATION INPUTS:**" + parts[1]
        else:
            # Append at the end before any final sections
            return base_eval_prompt + "\n" + examples_section

    def _build_requirement_rubrics(self, requirements: Requirements) -> str:
        """Build detailed rubrics for each requirement"""
        rubrics = []

        for i, req in enumerate(requirements.key_requirements, 1):
            rubrics.append(f"""
**Requirement {i}: {req}**
- **Score 5**: Fully implements this requirement with excellence
- **Score 4**: Strongly implements with minor gaps
- **Score 3**: Adequately addresses with notable limitations
- **Score 2**: Partially implements with significant gaps
- **Score 1**: Fails to implement or violates this requirement
""")

        if requirements.expected_behavior:
            rubrics.append(f"""
**Expected Behavior: {requirements.expected_behavior}**
- **Score 5**: Perfectly exhibits expected behavior
- **Score 4**: Strongly demonstrates expected behavior with minor deviations
- **Score 3**: Generally follows expected behavior with some inconsistencies
- **Score 2**: Partially demonstrates expected behavior with major gaps
- **Score 1**: Fails to demonstrate expected behavior
""")

        if requirements.constraints:
            constraint_text = ", ".join([f"{k}: {v}" for k, v in requirements.constraints.items()])
            rubrics.append(f"""
**Constraints: {constraint_text}**
- **Score 5**: Fully compliant with all constraints
- **Score 4**: Mostly compliant with minor violations
- **Score 3**: Generally compliant with some violations
- **Score 2**: Multiple constraint violations
- **Score 1**: Significant constraint non-compliance
""")

        return "\n".join(rubrics)

    def _generate_rationale(self, requirements: Requirements, test_scenarios: List[str]) -> str:
        """Generate rationale for the evaluation approach"""
        rationale = f"""This evaluation prompt is designed to comprehensively assess response quality across five critical dimensions:

1. **Contextual Relevance**: Ensures responses understand and align with user intent
2. **Groundedness**: Validates factual accuracy and prevents hallucinations
3. **Answer Relevance**: Confirms direct and complete addressing of user queries
4. **Requirement Adherence**: Tests compliance with all {len(requirements.key_requirements)} specified requirements
5. **Response Quality**: Evaluates overall clarity, structure, and effectiveness

The evaluation uses a standardized 1-5 scoring rubric per dimension to provide quantitative assessment,
along with qualitative feedback to identify specific issues. This approach follows industry best practices
for LLM response evaluation and is designed to be generic enough for use with any LLM provider.

The assessment focuses on observable, evidence-based evaluation rather than subjective preferences,
ensuring consistent and reliable quality measurement across all test cases.
"""
        return rationale

    async def refine_eval_prompt(
        self,
        current_eval_prompt: str,
        user_feedback: str,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Refine the evaluation prompt based on user feedback using LLM

        Args:
            current_eval_prompt: The current evaluation prompt
            user_feedback: User's feedback on what to change
            provider: LLM provider to use
            api_key: API key for the provider
            model_name: Optional specific model to use

        Returns:
            Dictionary with refined_prompt, changes_made, and rationale
        """
        # Build the refinement instruction
        refinement_instruction = f"""You are an expert prompt engineer specializing in evaluation prompts for LLM systems.

## Current Evaluation Prompt
```
{current_eval_prompt}
```

## User Feedback
{user_feedback}

## Instructions
Based on the user's feedback, refine the evaluation prompt above. Make targeted changes that address the feedback while maintaining the overall structure and effectiveness of the evaluation prompt.

Key considerations:
1. Preserve the 5-dimension evaluation framework (Contextual Relevance, Groundedness, Answer Relevance, Requirement Adherence, Response Quality)
2. Keep the JSON output format intact
3. Maintain the scoring rubrics (1-5 scale)
4. Only modify sections directly related to the feedback
5. Ensure changes enhance evaluation quality

## Output Format
Provide your response in the following format:

### Refined Evaluation Prompt
[Your refined evaluation prompt here - include the COMPLETE prompt, not just changes]

### Changes Made
- [List each specific change you made]

### Rationale
[Brief explanation of why these changes address the user's feedback and improve the evaluation]
"""

        # Use LLM to refine
        if provider == "openai" and api_key:
            result = await self._refine_with_openai(refinement_instruction, api_key, model_name)
        elif provider == "claude" and api_key:
            result = await self._refine_with_claude(refinement_instruction, api_key, model_name)
        elif provider == "gemini" and api_key:
            result = await self._refine_with_gemini(refinement_instruction, api_key, model_name)
        else:
            # Fallback to basic text modification
            result = self._basic_refine(current_eval_prompt, user_feedback)

        return result

    async def _refine_with_openai(self, instruction: str, api_key: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Use OpenAI to refine the eval prompt with fallback to gpt-4o"""
        import logging

        async def try_model(model: str, client) -> str:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert prompt engineer specializing in creating evaluation prompts for LLM systems. When refining prompts, return the complete refined prompt."},
                    {"role": "user", "content": instruction}
                ],
                temperature=0.7,
                max_tokens=8000
            )
            return response.choices[0].message.content

        try:
            client = openai.AsyncOpenAI(api_key=api_key)
            primary_model = model_name or "gpt-4o"

            try:
                content = await try_model(primary_model, client)
                logging.info(f"[EVAL REFINE] Successfully got response from {primary_model}, length: {len(content)}")
            except Exception as e:
                if primary_model != "gpt-4o":
                    logging.warning(f"[EVAL REFINE] Model {primary_model} failed: {e}. Falling back to gpt-4o")
                    content = await try_model("gpt-4o", client)
                    logging.info(f"[EVAL REFINE] Successfully got response from gpt-4o, length: {len(content)}")
                else:
                    raise

            result = self._parse_refine_response(content)
            logging.info(f"[EVAL REFINE] Parsed result - refined_prompt length: {len(result.get('refined_prompt', ''))}, changes: {len(result.get('changes_made', []))}")
            return result
        except Exception as e:
            logging.error(f"[EVAL REFINE] Error: {str(e)}")
            return {
                "refined_prompt": "",
                "changes_made": [],
                "rationale": f"Error using OpenAI: {str(e)}"
            }

    async def _refine_with_claude(self, instruction: str, api_key: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Use Claude to refine the eval prompt"""
        try:
            client = anthropic.AsyncAnthropic(api_key=api_key)
            response = await client.messages.create(
                model=model_name or "claude-3-5-sonnet-20241022",
                max_tokens=8000,
                messages=[
                    {"role": "user", "content": instruction}
                ]
            )
            content = response.content[0].text
            return self._parse_refine_response(content)
        except Exception as e:
            return {
                "refined_prompt": "",
                "changes_made": [],
                "rationale": f"Error using Claude: {str(e)}"
            }

    async def _refine_with_gemini(self, instruction: str, api_key: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Use Gemini to refine the eval prompt"""
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name or 'gemini-1.5-pro')
            response = await model.generate_content_async(instruction)
            content = response.text
            return self._parse_refine_response(content)
        except Exception as e:
            return {
                "refined_prompt": "",
                "changes_made": [],
                "rationale": f"Error using Gemini: {str(e)}"
            }

    def _parse_refine_response(self, content: str) -> Dict[str, Any]:
        """Parse the LLM response to extract refined prompt, changes, and rationale"""
        import logging
        refined_prompt = ""
        changes_made = []
        rationale = ""

        logging.info(f"[PARSE REFINE] Content length: {len(content)}")
        logging.info(f"[PARSE REFINE] Content preview (first 500 chars): {content[:500]}")

        # Split by sections (handle both ### and ** formatting)
        sections = content.split("###")

        for section in sections:
            section = section.strip()
            section_upper = section.upper()

            # Look for refined prompt section with various formats
            if (section_upper.startswith("REFINED EVALUATION PROMPT") or
                section_upper.startswith(" REFINED EVALUATION PROMPT") or
                section_upper.startswith("**REFINED EVALUATION PROMPT")):
                lines = section.split("\n", 1)
                if len(lines) > 1:
                    refined_prompt = lines[1].strip()
                    # Remove markdown code blocks if present
                    if refined_prompt.startswith("```"):
                        parts = refined_prompt.split("```")
                        if len(parts) >= 2:
                            refined_prompt = parts[1].strip()
                            # Remove language identifier if present
                            if refined_prompt.startswith(('markdown', 'text', '\n')):
                                refined_prompt = refined_prompt.split('\n', 1)[-1].strip() if '\n' in refined_prompt else refined_prompt
                    if refined_prompt.endswith("```"):
                        refined_prompt = refined_prompt[:-3].strip()

            elif (section_upper.startswith("CHANGES MADE") or
                  section_upper.startswith(" CHANGES MADE") or
                  section_upper.startswith("**CHANGES MADE")):
                lines = section.split("\n")[1:]
                changes_made = [line.strip("- *").strip() for line in lines if line.strip().startswith("-") or line.strip().startswith("*")]

            elif (section_upper.startswith("RATIONALE") or
                  section_upper.startswith(" RATIONALE") or
                  section_upper.startswith("**RATIONALE")):
                lines = section.split("\n", 1)
                if len(lines) > 1:
                    rationale = lines[1].strip()

        # If standard parsing failed, try to extract the largest code block as the refined prompt
        if not refined_prompt and "```" in content:
            import re
            code_blocks = re.findall(r'```(?:markdown|text)?\n?(.*?)```', content, re.DOTALL)
            if code_blocks:
                # Find the longest code block (likely the full prompt)
                refined_prompt = max(code_blocks, key=len).strip()

        # If still no refined prompt, check if content itself might be the prompt
        if not refined_prompt and len(content) > 100:
            # If the response doesn't follow our format, use the entire content minus any meta-text
            # Look for common prompt markers
            if "**I. Evaluator" in content or "### **I." in content or "Evaluator's Role" in content:
                # Try to extract just the prompt portion, removing any meta sections
                prompt_start = content
                # Remove any "Changes Made" or "Rationale" sections at the end
                for marker in ["### Changes Made", "### Rationale", "**Changes Made", "**Rationale"]:
                    if marker in prompt_start:
                        prompt_start = prompt_start.split(marker)[0].strip()
                refined_prompt = prompt_start
                if not rationale:
                    rationale = "Refined based on user feedback"
                if not changes_made:
                    changes_made = ["Incorporated user feedback into evaluation prompt"]

        # Last resort: if we got changes_made but no prompt, the LLM may have returned the prompt
        # without proper section markers - try to find it between code blocks
        if not refined_prompt and "```" in content:
            import re
            # Look for content between triple backticks
            all_blocks = re.findall(r'```(?:\w*\n)?(.*?)```', content, re.DOTALL)
            logging.info(f"[PARSE REFINE] Found {len(all_blocks)} code blocks")
            for i, block in enumerate(all_blocks):
                logging.info(f"[PARSE REFINE] Block {i} length: {len(block.strip())}, preview: {block.strip()[:100]}")
                if len(block.strip()) > 200 and ("Evaluator" in block or "Rating" in block or "INPUT" in block or "output" in block.lower()):
                    refined_prompt = block.strip()
                    logging.info(f"[PARSE REFINE] Using code block {i} as refined prompt")
                    break

        logging.info(f"[PARSE REFINE] Final result - refined_prompt: {len(refined_prompt)} chars, changes_made: {len(changes_made)}, rationale: {len(rationale)} chars")

        return {
            "refined_prompt": refined_prompt,
            "changes_made": changes_made,
            "rationale": rationale
        }

    def _basic_refine(self, current_prompt: str, feedback: str) -> Dict[str, Any]:
        """Basic fallback refinement without LLM"""
        # Add user feedback as a note in the prompt
        refined = current_prompt + f"\n\n## User Refinement Notes\n{feedback}"
        return {
            "refined_prompt": refined,
            "changes_made": ["Added user feedback as refinement notes"],
            "rationale": "Basic refinement applied (no LLM available). Configure LLM settings for AI-powered refinement."
        }


# Global instance
_eval_prompt_generator = None

def get_eval_prompt_generator() -> EvalPromptGenerator:
    """Get the global eval prompt generator instance"""
    global _eval_prompt_generator
    if _eval_prompt_generator is None:
        _eval_prompt_generator = EvalPromptGenerator()
    return _eval_prompt_generator
