"""
Smart Test Data Generator for Athena
Generates realistic, domain-appropriate test inputs based on prompt analysis.
Handles complex input types like call transcripts, emails, documents, etc.
"""
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class InputType(Enum):
    """Classification of expected input types"""
    SIMPLE_TEXT = "simple_text"          # Single line or paragraph
    MULTI_PARAGRAPH = "multi_paragraph"  # Article, essay, document
    CONVERSATION = "conversation"        # Chat logs, dialogues
    CALL_TRANSCRIPT = "call_transcript"  # Phone/video call transcripts
    EMAIL = "email"                      # Email content
    CODE = "code"                        # Source code
    STRUCTURED_DATA = "structured_data"  # JSON, XML, CSV
    LIST_ITEMS = "list_items"            # Bullet points, numbered lists
    FORM_DATA = "form_data"              # Key-value pairs, form submissions
    TICKET = "ticket"                    # Support tickets, bug reports
    REVIEW = "review"                    # Product/service reviews
    DOCUMENT = "document"                # Formal documents, contracts
    MEDICAL_RECORD = "medical_record"    # Healthcare data
    FINANCIAL_DATA = "financial_data"    # Financial statements, transactions


@dataclass
class InputFormatSpec:
    """Specification for generating test inputs"""
    input_type: InputType
    template_variable: str
    format_hints: List[str] = field(default_factory=list)
    domain_context: Optional[str] = None
    expected_length: str = "medium"  # short, medium, long
    structure_template: Optional[str] = None
    required_elements: List[str] = field(default_factory=list)


# Keywords that indicate specific input types
INPUT_TYPE_INDICATORS = {
    InputType.CALL_TRANSCRIPT: [
        'call transcript', 'transcript', 'phone call', 'video call', 'meeting',
        'conversation recording', 'call recording', 'phone conversation',
        'meeting notes', 'call summary', 'discussion', 'callTranscript'
    ],
    InputType.CONVERSATION: [
        'chat', 'dialogue', 'conversation', 'messages', 'chat log',
        'chat history', 'messaging', 'thread', 'discussion thread'
    ],
    InputType.EMAIL: [
        'email', 'mail', 'e-mail', 'message', 'correspondence',
        'inbox', 'emailContent', 'emailBody'
    ],
    InputType.CODE: [
        'code', 'source code', 'script', 'program', 'function',
        'snippet', 'codeSnippet', 'sourceCode', 'programming'
    ],
    InputType.DOCUMENT: [
        'document', 'article', 'report', 'paper', 'contract',
        'agreement', 'policy', 'documentation', 'spec', 'specification'
    ],
    InputType.TICKET: [
        'ticket', 'issue', 'bug report', 'support request', 'incident',
        'problem report', 'helpdesk', 'support ticket'
    ],
    InputType.REVIEW: [
        'review', 'feedback', 'rating', 'testimonial', 'comment',
        'customer feedback', 'product review', 'user review'
    ],
    InputType.MEDICAL_RECORD: [
        'patient', 'medical', 'clinical', 'health record', 'diagnosis',
        'prescription', 'symptoms', 'medical history', 'patient record'
    ],
    InputType.FINANCIAL_DATA: [
        'financial', 'transaction', 'invoice', 'statement', 'balance',
        'accounting', 'ledger', 'payment', 'expense', 'revenue'
    ],
    InputType.STRUCTURED_DATA: [
        'json', 'xml', 'csv', 'data', 'payload', 'object', 'record'
    ],
}


def detect_input_type(prompt_text: str, template_variables: List[str]) -> InputFormatSpec:
    """
    Analyze the prompt to determine what type of input is expected.
    Uses template variable names and prompt context.
    """
    text_lower = prompt_text.lower()

    # Check template variables for hints
    primary_variable = template_variables[0] if template_variables else None
    var_name_lower = primary_variable.lower() if primary_variable else ""

    # Score each input type based on keyword matches
    type_scores: Dict[InputType, int] = {t: 0 for t in InputType}

    for input_type, keywords in INPUT_TYPE_INDICATORS.items():
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Check in template variable name (high weight)
            if primary_variable and keyword_lower in var_name_lower:
                type_scores[input_type] += 5
            # Check in prompt text
            if keyword_lower in text_lower:
                type_scores[input_type] += 2

    # Find the highest scoring type
    detected_type = max(type_scores, key=type_scores.get)

    # If no strong signal, default to simple text or multi-paragraph based on prompt length
    if type_scores[detected_type] < 2:
        if len(prompt_text) > 500 or 'summarize' in text_lower or 'analyze' in text_lower:
            detected_type = InputType.MULTI_PARAGRAPH
        else:
            detected_type = InputType.SIMPLE_TEXT

    # Extract domain context
    domain_context = extract_domain_context(prompt_text)

    # Extract required elements from output format
    required_elements = extract_required_elements(prompt_text)

    # Determine expected length
    expected_length = detect_expected_length(prompt_text)

    return InputFormatSpec(
        input_type=detected_type,
        template_variable=primary_variable or "input",
        domain_context=domain_context,
        expected_length=expected_length,
        required_elements=required_elements,
        format_hints=extract_format_hints(prompt_text, detected_type)
    )


def extract_domain_context(prompt_text: str) -> Optional[str]:
    """Extract the domain/industry context from the prompt"""
    text_lower = prompt_text.lower()

    domain_keywords = {
        'sales': ['sales', 'deal', 'prospect', 'pipeline', 'quota', 'revenue', 'customer acquisition'],
        'customer_support': ['support', 'help desk', 'customer service', 'complaint', 'resolution', 'ticket'],
        'healthcare': ['patient', 'medical', 'health', 'clinical', 'diagnosis', 'treatment', 'hospital'],
        'finance': ['financial', 'accounting', 'investment', 'banking', 'loan', 'credit', 'transaction'],
        'legal': ['legal', 'contract', 'agreement', 'compliance', 'litigation', 'attorney', 'law'],
        'hr': ['employee', 'hiring', 'recruitment', 'performance', 'hr', 'human resources', 'interview'],
        'technology': ['software', 'engineering', 'technical', 'development', 'code', 'system', 'api'],
        'marketing': ['marketing', 'campaign', 'brand', 'advertising', 'content', 'social media'],
        'education': ['education', 'learning', 'student', 'teacher', 'course', 'training', 'curriculum'],
        'retail': ['retail', 'store', 'product', 'inventory', 'order', 'shipping', 'e-commerce'],
    }

    for domain, keywords in domain_keywords.items():
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches >= 2:
            return domain

    return None


def extract_required_elements(prompt_text: str) -> List[str]:
    """Extract elements the output must contain (useful for generating test inputs that cover all aspects)"""
    elements = []

    # Look for numbered list items in output format
    numbered = re.findall(r'^\s*\d+\.\s*([^\n(]+)', prompt_text, re.MULTILINE)
    elements.extend([n.strip() for n in numbered if len(n.strip()) > 3])

    # Look for bullet points
    bullets = re.findall(r'^\s*[-*]\s*([^\n(]+)', prompt_text, re.MULTILINE)
    elements.extend([b.strip() for b in bullets if len(b.strip()) > 3])

    return elements[:10]  # Limit to 10 elements


def detect_expected_length(prompt_text: str) -> str:
    """Detect expected input length based on prompt context"""
    text_lower = prompt_text.lower()

    # Long indicators
    if any(kw in text_lower for kw in ['detailed', 'comprehensive', 'full transcript', 'complete']):
        return "long"

    # Short indicators
    if any(kw in text_lower for kw in ['brief', 'short', 'quick', 'concise input']):
        return "short"

    return "medium"


def extract_format_hints(prompt_text: str, input_type: InputType) -> List[str]:
    """Extract specific formatting hints for the input type"""
    hints = []
    text_lower = prompt_text.lower()

    if input_type == InputType.CALL_TRANSCRIPT:
        # Look for speaker format hints
        if 'speaker' in text_lower or 'participant' in text_lower:
            hints.append("Include speaker labels")
        if 'timestamp' in text_lower or 'time' in text_lower:
            hints.append("Include timestamps")
        if 'action item' in text_lower:
            hints.append("Include discussion of action items")
        if 'decision' in text_lower:
            hints.append("Include decision points")
        if 'risk' in text_lower or 'concern' in text_lower or 'objection' in text_lower:
            hints.append("Include concerns or objections")

    elif input_type == InputType.EMAIL:
        if 'subject' in text_lower:
            hints.append("Include subject line")
        if 'attachment' in text_lower:
            hints.append("Reference attachments")

    elif input_type == InputType.CODE:
        # Detect language
        for lang in ['python', 'javascript', 'java', 'typescript', 'go', 'rust', 'c++', 'c#']:
            if lang in text_lower:
                hints.append(f"Language: {lang}")
                break

    return hints


def build_input_generation_prompt(spec: InputFormatSpec, prompt_text: str, use_case: str, requirements: str) -> str:
    """Build a specialized prompt for generating test inputs based on the detected format"""

    type_specific_instructions = get_type_specific_instructions(spec)

    prompt = f"""You are generating REALISTIC test inputs for a prompt testing system.

## INPUT TYPE DETECTED: {spec.input_type.value.replace('_', ' ').title()}

## CONTEXT
- **Template Variable:** {{{{{spec.template_variable}}}}}
- **Domain:** {spec.domain_context or 'General'}
- **Expected Length:** {spec.expected_length}
- **Use Case:** {use_case}
- **Requirements:** {requirements}

## TYPE-SPECIFIC INSTRUCTIONS
{type_specific_instructions}

## FORMAT HINTS FROM PROMPT
{chr(10).join(f'- {h}' for h in spec.format_hints) if spec.format_hints else '- No specific format hints detected'}

## REQUIRED OUTPUT ELEMENTS (ensure test inputs would generate these)
{chr(10).join(f'- {e}' for e in spec.required_elements) if spec.required_elements else '- General coverage'}

## CRITICAL RULES FOR GENERATING INPUTS
1. Generate COMPLETE, REALISTIC inputs - not placeholders or summaries
2. Each input should be different enough to test various aspects
3. Include both typical and edge cases
4. Make adversarial inputs subtle and realistic
5. Vary the complexity, length, and content of inputs
6. For {spec.input_type.value}: {get_realism_tips(spec.input_type)}

## OUTPUT FORMAT
Return a JSON object with test_cases array. Each test case should have:
- "input": The FULL, COMPLETE test input (not abbreviated)
- "category": "positive|edge_case|negative|adversarial"
- "test_focus": What aspect this tests
- "expected_behavior": Expected system behavior
- "variation": Brief description of what makes this case unique

For {spec.input_type.value.replace('_', ' ')} inputs, make them REALISTIC and COMPLETE."""

    return prompt


def get_type_specific_instructions(spec: InputFormatSpec) -> str:
    """Get detailed instructions for generating specific input types"""

    instructions = {
        InputType.CALL_TRANSCRIPT: """
Generate REALISTIC call transcripts with:
- Multiple speakers (use names like "John:", "Sarah:", "Manager:", etc.)
- Natural conversation flow with interruptions, acknowledgments, filler words
- Realistic timestamps if appropriate (e.g., [00:00:15], [00:02:30])
- Mix of topics: greetings, main discussion, questions, action items, wrap-up
- Include realistic details: numbers, dates, names, specific topics

TRANSCRIPT VARIATIONS TO INCLUDE:
1. **Clear/Structured Call**: Organized meeting with clear agenda and outcomes
2. **Chaotic Call**: Multiple speakers talking over each other, topic changes
3. **Short Check-in**: Brief call with minimal content
4. **Long Detailed Call**: Extensive discussion with many points
5. **Sales Call**: Pitch, objections, negotiation
6. **Support Call**: Problem description, troubleshooting, resolution
7. **Internal Meeting**: Team discussions, planning, status updates
8. **Call with Issues**: Poor audio references, speaker cut-offs
9. **One-sided Call**: One speaker dominates
10. **Technical Call**: Jargon-heavy, specific terminology

Make each transcript 200-800 words depending on the scenario.""",

        InputType.CONVERSATION: """
Generate REALISTIC chat conversations with:
- Multiple participants with distinct communication styles
- Typical chat patterns: short messages, typos, abbreviations, emojis
- Natural flow including questions, responses, topic changes
- Timestamps between messages

CONVERSATION VARIATIONS:
1. Professional chat (Slack/Teams style)
2. Customer support chat
3. Informal team discussion
4. Technical troubleshooting
5. Quick coordination messages""",

        InputType.EMAIL: """
Generate REALISTIC emails with:
- Subject line
- From/To headers
- Professional greeting and signature
- Proper email formatting
- Attachments references where relevant

EMAIL VARIATIONS:
1. Formal business communication
2. Internal team update
3. Customer inquiry/complaint
4. Sales outreach
5. Meeting follow-up
6. Brief acknowledgment
7. Detailed report/proposal""",

        InputType.DOCUMENT: """
Generate REALISTIC documents with:
- Proper structure (headings, sections, paragraphs)
- Professional language appropriate to the domain
- Specific details, facts, and figures
- Consistent formatting

DOCUMENT VARIATIONS:
1. Technical specification
2. Business proposal
3. Policy document
4. Report/Analysis
5. Contract/Agreement excerpt""",

        InputType.CODE: """
Generate REALISTIC code samples with:
- Proper syntax for the detected language
- Comments where appropriate
- Mix of quality (clean code, messy code, bugs)
- Realistic variable names and function names

CODE VARIATIONS:
1. Clean, well-documented code
2. Code with obvious bugs
3. Code with subtle issues
4. Legacy/outdated patterns
5. Complex nested logic""",

        InputType.TICKET: """
Generate REALISTIC support tickets with:
- Clear issue description
- Steps to reproduce (if applicable)
- Environment/system information
- Priority/severity indicators
- User frustration levels (polite to angry)

TICKET VARIATIONS:
1. Clear, well-written ticket
2. Vague/incomplete description
3. Urgent/critical issue
4. Feature request disguised as bug
5. User error vs actual bug""",

        InputType.REVIEW: """
Generate REALISTIC reviews with:
- Rating (if applicable)
- Specific feedback points
- Emotional tone variation
- Length variation

REVIEW VARIATIONS:
1. Detailed positive review
2. Constructive criticism
3. Angry complaint
4. Mixed feelings
5. Brief/unhelpful review""",
    }

    return instructions.get(spec.input_type, """
Generate realistic inputs that match the expected format.
Vary the length, complexity, and content of each input.
Include both typical and edge cases.""")


def get_realism_tips(input_type: InputType) -> str:
    """Get tips for making inputs more realistic"""
    tips = {
        InputType.CALL_TRANSCRIPT: "Include natural speech patterns, filler words (um, uh, you know), interruptions, and real names/dates",
        InputType.CONVERSATION: "Include typos, abbreviations, emoji, and natural message timing",
        InputType.EMAIL: "Include proper email headers, signatures, and typical email conventions",
        InputType.CODE: "Include realistic variable names, comments, and code patterns",
        InputType.DOCUMENT: "Include proper formatting, section headers, and document structure",
        InputType.TICKET: "Include system info, steps to reproduce, and user tone",
        InputType.REVIEW: "Include specific details, emotional language, and rating references",
    }
    return tips.get(input_type, "Include realistic details and natural language patterns")


def get_category_distribution(input_type: InputType, num_cases: int) -> Dict[str, int]:
    """Get the recommended category distribution for a specific input type"""

    # Base distribution - adjust based on input type
    if input_type in [InputType.CALL_TRANSCRIPT, InputType.CONVERSATION, InputType.DOCUMENT]:
        # More positive cases needed for complex inputs
        distribution = {
            "positive": 0.50,  # Standard cases
            "edge_case": 0.25,  # Unusual but valid
            "negative": 0.15,  # Invalid/error cases
            "adversarial": 0.10  # Tricky inputs
        }
    else:
        # Standard distribution
        distribution = {
            "positive": 0.40,
            "edge_case": 0.25,
            "negative": 0.20,
            "adversarial": 0.15
        }

    # Convert to counts
    result = {}
    remaining = num_cases
    for category, percentage in distribution.items():
        count = max(1, int(num_cases * percentage))
        result[category] = count
        remaining -= count

    # Add any remaining to positive
    if remaining > 0:
        result["positive"] += remaining
    elif remaining < 0:
        result["positive"] = max(1, result["positive"] + remaining)

    return result


def get_scenario_variations(spec: InputFormatSpec, num_cases: int) -> List[Dict[str, Any]]:
    """Generate specific scenario variations for the input type"""

    scenarios = {
        InputType.CALL_TRANSCRIPT: [
            {"scenario": "sales_call", "description": "Sales pitch with objections and negotiation", "category": "positive"},
            {"scenario": "support_call", "description": "Customer support with problem resolution", "category": "positive"},
            {"scenario": "team_meeting", "description": "Internal team status update meeting", "category": "positive"},
            {"scenario": "executive_call", "description": "Executive briefing with strategic decisions", "category": "positive"},
            {"scenario": "onboarding_call", "description": "New client/employee onboarding", "category": "positive"},
            {"scenario": "technical_call", "description": "Technical discussion with jargon", "category": "edge_case"},
            {"scenario": "multilingual_call", "description": "Call with occasional non-English phrases", "category": "edge_case"},
            {"scenario": "chaotic_call", "description": "Disorganized call with interruptions", "category": "edge_case"},
            {"scenario": "brief_call", "description": "Very short check-in call", "category": "edge_case"},
            {"scenario": "audio_issues", "description": "Call with noted audio problems", "category": "edge_case"},
            {"scenario": "no_decisions", "description": "Call with discussion but no clear decisions", "category": "negative"},
            {"scenario": "one_sided", "description": "Call dominated by one speaker", "category": "negative"},
            {"scenario": "incomplete", "description": "Transcript that cuts off mid-conversation", "category": "negative"},
            {"scenario": "irrelevant", "description": "Off-topic casual conversation", "category": "negative"},
            {"scenario": "injection_attempt", "description": "Speaker mentions 'ignore previous instructions'", "category": "adversarial"},
            {"scenario": "conflicting_info", "description": "Speakers give contradictory information", "category": "adversarial"},
        ],
        InputType.EMAIL: [
            {"scenario": "formal_request", "description": "Formal business request", "category": "positive"},
            {"scenario": "status_update", "description": "Project status update", "category": "positive"},
            {"scenario": "complaint", "description": "Customer complaint email", "category": "positive"},
            {"scenario": "thread", "description": "Email with forwarded chain", "category": "edge_case"},
            {"scenario": "brief", "description": "One-line acknowledgment", "category": "edge_case"},
            {"scenario": "empty_body", "description": "Subject only, empty body", "category": "negative"},
            {"scenario": "spam_like", "description": "Email that looks like spam", "category": "adversarial"},
        ],
        InputType.CODE: [
            {"scenario": "clean_code", "description": "Well-structured, documented code", "category": "positive"},
            {"scenario": "complex_logic", "description": "Complex but correct implementation", "category": "positive"},
            {"scenario": "minor_bugs", "description": "Code with subtle bugs", "category": "edge_case"},
            {"scenario": "syntax_errors", "description": "Code with syntax errors", "category": "negative"},
            {"scenario": "malicious", "description": "Code with potential security issues", "category": "adversarial"},
        ],
    }

    type_scenarios = scenarios.get(spec.input_type, [
        {"scenario": "standard", "description": "Typical input", "category": "positive"},
        {"scenario": "complex", "description": "Complex valid input", "category": "positive"},
        {"scenario": "minimal", "description": "Minimal valid input", "category": "edge_case"},
        {"scenario": "invalid", "description": "Invalid input format", "category": "negative"},
        {"scenario": "tricky", "description": "Edge case that might confuse", "category": "adversarial"},
    ])

    # Select scenarios up to num_cases
    selected = []
    category_counts = get_category_distribution(spec.input_type, num_cases)

    for category, count in category_counts.items():
        category_scenarios = [s for s in type_scenarios if s["category"] == category]
        for i in range(count):
            if category_scenarios:
                selected.append(category_scenarios[i % len(category_scenarios)])
            else:
                # Fallback
                selected.append({"scenario": f"{category}_{i}", "description": f"Test case for {category}", "category": category})

    return selected[:num_cases]
