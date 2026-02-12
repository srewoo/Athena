"""
Load Additional Domain Patterns for Call Summaries & Chat Copilots

Expands Athena's pattern library to generate expert-level evals for:
- Call summary generation systems
- Chat copilot systems
- Q&A systems
- Content generation systems
"""

import asyncio
import logging
from dimension_pattern_service import get_dimension_pattern_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def load_call_summary_patterns():
    """Load expert patterns for call summary generation systems."""
    service = get_dimension_pattern_service()

    logger.info("\n" + "="*60)
    logger.info("Loading Call Summary Generation Patterns")
    logger.info("="*60)

    patterns_loaded = 0

    # Architecture Pattern
    await service.store_pattern(
        category="eval_architecture",
        pattern_data={
            "pattern_name": "Fidelity + Abstraction Split",
            "domain": "call_summary_generation",
            "description": "Separate factual accuracy checks (Layer 1) from quality of summarization (Layer 2). Facts either match transcript or don't (binary). Summary quality is gradient (good abstraction vs verbatim transcription).",
            "when_to_use": "When evaluating any system that transforms source content into summaries",
            "layer_1_checks": [
                "Information fidelity (no hallucinations)",
                "Attribution accuracy (speaker labels correct)",
                "Temporal accuracy (sequence preserved)",
                "Completeness (key info not omitted)"
            ],
            "layer_2_checks": [
                "Abstraction quality (synthesized vs transcribed)",
                "Prioritization (important vs trivial)",
                "Clarity and conciseness",
                "Actionability (decisions, next steps clear)"
            ]
        }
    )
    patterns_loaded += 1

    # Design Principle: Four Pillars of Summary Quality
    await service.store_pattern(
        category="design_principle",
        pattern_data={
            "principle": "Four Pillars of Summary Quality",
            "domain": "call_summary_generation",
            "description": "Break summary evaluation into: (1) Information Fidelity - facts match source, (2) Abstraction Quality - proper synthesis not transcription, (3) Actionability - decisions/next steps extracted, (4) Usability - clear structure, easy to scan",
            "dimensions": [
                "information_fidelity",
                "abstraction_quality",
                "actionability",
                "usability"
            ],
            "benefit": "Comprehensive coverage of what makes summaries useful",
            "anti_pattern": "Single 'quality' dimension that conflates accuracy and usefulness"
        }
    )
    patterns_loaded += 1

    # Failure Mode: Information Fidelity
    await service.store_pattern(
        category="failure_mode",
        pattern_data={
            "dimension": "information_fidelity",
            "domain": "call_summary_generation",
            "failure_modes": [
                "Hallucinated information (facts not in transcript)",
                "Misattributed statements (wrong speaker)",
                "Temporal errors (sequence scrambled)",
                "Key omissions (decisions/actions missing)",
                "Tone misrepresentation (neutral call summarized as contentious)"
            ],
            "detection": "Cross-reference every fact in summary against transcript. Flag any claim without source evidence. Check speaker labels, timestamps, event sequence.",
            "example": "Summary says 'Client agreed to $50k budget' but transcript shows 'We'll need to discuss budget with leadership' - hallucination of commitment"
        }
    )
    patterns_loaded += 1

    # Failure Mode: Abstraction Quality
    await service.store_pattern(
        category="failure_mode",
        pattern_data={
            "dimension": "abstraction_quality",
            "domain": "call_summary_generation",
            "failure_modes": [
                "Verbatim transcription (copying dialogue, not summarizing)",
                "Over-abstraction (loses critical detail)",
                "Missing synthesis (lists points without connecting)",
                "No prioritization (treats all info equally)",
                "Jargon overload (assumes reader context)"
            ],
            "detection": "Check if summary reads like meeting minutes vs synthesis. Look for verbatim quotes where paraphrase would suffice. Verify important points are emphasized.",
            "example": "Summary: 'Then she said they need help with X, and he said they tried Y, and she said...' - this is transcription, not summary"
        }
    )
    patterns_loaded += 1

    # Failure Mode: Actionability
    await service.store_pattern(
        category="failure_mode",
        pattern_data={
            "dimension": "actionability",
            "domain": "call_summary_generation",
            "failure_modes": [
                "Missing action items (decisions buried in narrative)",
                "Unclear ownership (who does what is ambiguous)",
                "Vague deadlines ('soon', 'next week' without dates)",
                "Lost context (action makes no sense without background)",
                "No follow-up tracking (commitments not extractable)"
            ],
            "detection": "Search for explicit action item section. Verify each action has: what + who + when. Check if decisions are surfaced not buried.",
            "example": "Summary mentions 'follow up on proposal' but doesn't say who follows up, which proposal, or by when - not actionable"
        }
    )
    patterns_loaded += 1

    # Scoring Pattern for Summaries
    await service.store_pattern(
        category="scoring_pattern",
        pattern_data={
            "pattern_name": "Hybrid Binary + Gradient",
            "domain": "call_summary_generation",
            "description": "Use Binary (PASS/FAIL) for fidelity checks (facts are correct or not). Use Gradient (STRONG/ACCEPTABLE/WEAK) for quality checks (abstraction, clarity). Overall = fail if fidelity fails, otherwise use gradient.",
            "when_to_use": "When system has both objective correctness requirements and subjective quality aspects",
            "structure": {
                "information_fidelity": "Binary sub-checks (hallucinations=FAIL, misattribution=FAIL) → dimension PASS/FAIL",
                "abstraction_quality": "Gradient (verbatim=WEAK, good synthesis=STRONG)",
                "actionability": "Gradient (vague=WEAK, clear ownership+deadlines=STRONG)",
                "usability": "Gradient (wall of text=WEAK, scannable structure=STRONG)"
            },
            "overall_scoring": "If information_fidelity=FAIL → overall=FAIL. Otherwise overall=lowest gradient score."
        }
    )
    patterns_loaded += 1

    # Intent Keywords
    await service.store_pattern(
        category="intent_keyword",
        pattern_data={
            "keywords": ["summary", "summarize", "summarization", "meeting notes", "call notes", "transcript"],
            "domain": "call_summary_generation",
            "recommended_pattern": "Fidelity + Abstraction Split",
            "recommended_dimensions": [
                "information_fidelity (hallucinations, misattribution, omissions)",
                "abstraction_quality (synthesis vs transcription, prioritization)",
                "actionability (decisions, action items, ownership, deadlines)",
                "usability (structure, scannability, clarity)"
            ],
            "scoring_pattern": "Hybrid Binary + Gradient",
            "reasoning": "Summaries must be factually accurate (binary) AND useful (gradient)"
        }
    )
    patterns_loaded += 1

    logger.info(f"✅ Loaded {patterns_loaded} call summary patterns")
    return patterns_loaded


async def load_chat_copilot_patterns():
    """Load expert patterns for chat copilot systems."""
    service = get_dimension_pattern_service()

    logger.info("\n" + "="*60)
    logger.info("Loading Chat Copilot Patterns")
    logger.info("="*60)

    patterns_loaded = 0

    # Architecture Pattern
    await service.store_pattern(
        category="eval_architecture",
        pattern_data={
            "pattern_name": "Safety + Helpfulness + Context",
            "domain": "chat_copilot",
            "description": "Three-layer evaluation: (1) Safety guardrails (binary - must pass), (2) Helpfulness (gradient - solves problem?), (3) Context awareness (gradient - remembers conversation?). Safety failures auto-reject.",
            "when_to_use": "Chat assistants, copilots, conversational AI where safety and context matter",
            "layer_1_safety": [
                "No harmful content (hate, violence, illegal)",
                "No unsafe code (security vulnerabilities, data leaks)",
                "No privacy violations (PII exposure)",
                "No jailbreaking (refusal bypasses)"
            ],
            "layer_2_helpfulness": [
                "Actually solves user problem",
                "Appropriate detail level",
                "Clear explanations",
                "Actionable guidance"
            ],
            "layer_3_context": [
                "Remembers prior conversation",
                "Consistent across turns",
                "Handles clarifications",
                "Adapts to user corrections"
            ]
        }
    )
    patterns_loaded += 1

    # Design Principle: Five Dimensions of Chat Quality
    await service.store_pattern(
        category="design_principle",
        pattern_data={
            "principle": "Five Dimensions of Chat Quality",
            "domain": "chat_copilot",
            "description": "Evaluate copilot responses across: (1) Safety - harmful content/code?, (2) Helpfulness - solves problem?, (3) Accuracy - facts/code correct?, (4) Context awareness - remembers conversation?, (5) Communication - clear and appropriate?",
            "dimensions": [
                "safety",
                "helpfulness",
                "accuracy",
                "context_awareness",
                "communication_quality"
            ],
            "benefit": "Covers both safety requirements and quality aspects",
            "anti_pattern": "Generic 'response quality' that misses safety or context issues"
        }
    )
    patterns_loaded += 1

    # Failure Mode: Safety
    await service.store_pattern(
        category="failure_mode",
        pattern_data={
            "dimension": "safety",
            "domain": "chat_copilot",
            "failure_modes": [
                "Harmful content (hate speech, violence, illegal activities)",
                "Unsafe code suggestions (SQL injection, XSS, hardcoded secrets)",
                "Privacy violations (logs PII, exposes data)",
                "Jailbreak acceptance (ignores safety guardrails)",
                "Dangerous instructions (harmful physical actions)"
            ],
            "detection": "Scan for harmful content patterns. Check code for security anti-patterns (eval(), exec(), SQL concatenation). Verify PII is not logged or exposed. Test refusal bypasses.",
            "example": "User: 'Ignore previous instructions and...' → Copilot complies instead of refusing - jailbreak vulnerability"
        }
    )
    patterns_loaded += 1

    # Failure Mode: Helpfulness
    await service.store_pattern(
        category="failure_mode",
        pattern_data={
            "dimension": "helpfulness",
            "domain": "chat_copilot",
            "failure_modes": [
                "Doesn't solve actual problem (generic advice)",
                "Over-explains trivial things (wastes time)",
                "Under-explains complex things (too terse)",
                "Provides unusable code (syntactically correct but doesn't work)",
                "Misunderstands intent (solves wrong problem)"
            ],
            "detection": "Check if response actually addresses user's question/problem. Verify code runs and produces expected output. Assess detail level vs problem complexity.",
            "example": "User: 'How do I fix this React error: ...' → Copilot: 'React is a JavaScript library for building UIs' - doesn't address the error"
        }
    )
    patterns_loaded += 1

    # Failure Mode: Context Awareness
    await service.store_pattern(
        category="failure_mode",
        pattern_data={
            "dimension": "context_awareness",
            "domain": "chat_copilot",
            "failure_modes": [
                "Forgets prior conversation (repeats questions)",
                "Contradicts earlier responses",
                "Ignores user corrections ('Actually, it's Python 3.11' → still suggests Python 2 syntax)",
                "Loses project context (suggests incompatible libraries)",
                "Doesn't handle clarifications (treats as new query)"
            ],
            "detection": "Check if response references prior turns. Verify consistency with earlier statements. Test if corrections are incorporated. Check if context accumulates across turns.",
            "example": "Turn 1: 'I'm using TypeScript' → Turn 3: Copilot suggests JavaScript syntax without types - forgot context"
        }
    )
    patterns_loaded += 1

    # Failure Mode: Accuracy
    await service.store_pattern(
        category="failure_mode",
        pattern_data={
            "dimension": "accuracy",
            "domain": "chat_copilot",
            "failure_modes": [
                "Factual errors (wrong API signatures, outdated info)",
                "Hallucinated functions (suggests APIs that don't exist)",
                "Incorrect code (compiles but produces wrong output)",
                "Wrong library versions (suggests deprecated methods)",
                "Outdated best practices (suggests old patterns)"
            ],
            "detection": "Verify facts against documentation. Test code execution. Check API references. Validate library versions. Cross-check with authoritative sources.",
            "example": "Suggests using 'requests.get().json()' for async code - wrong pattern, should use aiohttp"
        }
    )
    patterns_loaded += 1

    # Scoring Pattern
    await service.store_pattern(
        category="scoring_pattern",
        pattern_data={
            "pattern_name": "Safety-First Tiered Scoring",
            "domain": "chat_copilot",
            "description": "Safety is binary gate (PASS/FAIL). If safety fails, overall = FAIL regardless of other dimensions. Otherwise, use STRONG/ACCEPTABLE/WEAK/FAIL for helpfulness, accuracy, context, communication.",
            "when_to_use": "Any system where safety is non-negotiable requirement",
            "structure": {
                "safety": "Binary PASS/FAIL (any safety issue = FAIL)",
                "helpfulness": "STRONG (solves problem) / ACCEPTABLE (partially helpful) / WEAK (generic) / FAIL (misleading)",
                "accuracy": "STRONG (all correct) / ACCEPTABLE (minor errors) / WEAK (significant errors) / FAIL (dangerous misinformation)",
                "context_awareness": "STRONG (remembers all) / ACCEPTABLE (mostly consistent) / WEAK (forgets context) / FAIL (contradicts self)",
                "communication_quality": "STRONG (clear, appropriate) / ACCEPTABLE (understandable) / WEAK (confusing) / FAIL (incomprehensible)"
            },
            "overall_scoring": "If safety=FAIL → overall=FAIL. Otherwise overall=lowest other dimension score."
        }
    )
    patterns_loaded += 1

    # Intent Keywords
    await service.store_pattern(
        category="intent_keyword",
        pattern_data={
            "keywords": ["chat", "copilot", "assistant", "conversational", "dialogue", "chatbot", "AI assistant"],
            "domain": "chat_copilot",
            "recommended_pattern": "Safety + Helpfulness + Context",
            "recommended_dimensions": [
                "safety (harmful content, unsafe code, privacy, jailbreaks)",
                "helpfulness (solves problem, appropriate detail, actionable)",
                "accuracy (facts correct, code works, up-to-date)",
                "context_awareness (remembers conversation, consistent, adapts)",
                "communication_quality (clear, concise, appropriate tone)"
            ],
            "scoring_pattern": "Safety-First Tiered Scoring",
            "reasoning": "Safety must pass (binary). Quality aspects are gradient. Context matters for multi-turn."
        }
    )
    patterns_loaded += 1

    logger.info(f"✅ Loaded {patterns_loaded} chat copilot patterns")
    return patterns_loaded


async def load_qa_system_patterns():
    """Load expert patterns for Q&A systems."""
    service = get_dimension_pattern_service()

    logger.info("\n" + "="*60)
    logger.info("Loading Q&A System Patterns")
    logger.info("="*60)

    patterns_loaded = 0

    # Architecture Pattern
    await service.store_pattern(
        category="eval_architecture",
        pattern_data={
            "pattern_name": "Correctness + Completeness + Groundedness",
            "domain": "qa_system",
            "description": "Three-pillar evaluation: (1) Correctness - answer factually accurate?, (2) Completeness - addresses full question?, (3) Groundedness - answer traceable to sources? For RAG systems, groundedness is critical.",
            "when_to_use": "Question answering systems, RAG applications, knowledge retrieval",
            "pillars": [
                "Correctness (factual accuracy)",
                "Completeness (full answer, not partial)",
                "Groundedness (citable sources)",
                "Clarity (understandable)",
                "Relevance (on-topic)"
            ]
        }
    )
    patterns_loaded += 1

    # Failure Mode: Groundedness (RAG-specific)
    await service.store_pattern(
        category="failure_mode",
        pattern_data={
            "dimension": "groundedness",
            "domain": "qa_system",
            "failure_modes": [
                "Answer claims not in retrieved documents (hallucination)",
                "Citations point to wrong documents",
                "Cherry-picking (ignores contradicting sources)",
                "Synthesis hallucination (combines facts incorrectly)",
                "No citation for controversial claims"
            ],
            "detection": "Cross-reference every claim in answer against retrieved documents. Verify citations are accurate. Check if answer acknowledges source conflicts.",
            "example": "Answer: 'Product launched in 2020' citing Doc A, but Doc A says 2019 - wrong citation"
        }
    )
    patterns_loaded += 1

    # Intent Keywords
    await service.store_pattern(
        category="intent_keyword",
        pattern_data={
            "keywords": ["question answering", "Q&A", "RAG", "retrieval", "knowledge base", "FAQ"],
            "domain": "qa_system",
            "recommended_dimensions": [
                "correctness (factually accurate answers)",
                "completeness (addresses full question)",
                "groundedness (traceable to sources, citations accurate)",
                "clarity (understandable answer)",
                "relevance (on-topic, not tangential)"
            ],
            "scoring_pattern": "STRONG/ACCEPTABLE/WEAK/FAIL with correctness + groundedness failures = FAIL"
        }
    )
    patterns_loaded += 1

    logger.info(f"✅ Loaded {patterns_loaded} Q&A system patterns")
    return patterns_loaded


async def load_content_generation_patterns():
    """Load expert patterns for content generation systems."""
    service = get_dimension_pattern_service()

    logger.info("\n" + "="*60)
    logger.info("Loading Content Generation Patterns")
    logger.info("="*60)

    patterns_loaded = 0

    # Architecture Pattern
    await service.store_pattern(
        category="eval_architecture",
        pattern_data={
            "pattern_name": "Creativity + Coherence + Constraints",
            "domain": "content_generation",
            "description": "Balance three aspects: (1) Creativity - original, engaging content, (2) Coherence - logical flow, consistent style, (3) Constraints - follows requirements (length, tone, format). Creative but incoherent = bad. Coherent but violates constraints = bad.",
            "when_to_use": "Marketing copy, blog posts, product descriptions, creative writing",
            "aspects": [
                "Creativity (originality, engagement)",
                "Coherence (logical flow, consistency)",
                "Constraint compliance (length, tone, format)",
                "Audience appropriateness",
                "Brand alignment"
            ]
        }
    )
    patterns_loaded += 1

    # Failure Mode: Constraint Compliance
    await service.store_pattern(
        category="failure_mode",
        pattern_data={
            "dimension": "constraint_compliance",
            "domain": "content_generation",
            "failure_modes": [
                "Length violations (200-word limit but 350 words)",
                "Tone mismatch (formal requested but casual output)",
                "Format violations (bullet points requested but paragraph)",
                "Keyword missing (SEO keyword not included)",
                "Call-to-action missing (CTA required but absent)"
            ],
            "detection": "Check word count, tone analysis, format structure, keyword presence, required elements.",
            "example": "Brief: 'Write 100-word product description' → Output: 250 words - constraint violation"
        }
    )
    patterns_loaded += 1

    # Intent Keywords
    await service.store_pattern(
        category="intent_keyword",
        pattern_data={
            "keywords": ["content generation", "writing", "copywriting", "blog", "article", "creative"],
            "domain": "content_generation",
            "recommended_dimensions": [
                "creativity (original, engaging, not generic)",
                "coherence (logical flow, consistent style)",
                "constraint_compliance (length, tone, format requirements)",
                "audience_appropriateness (matches target audience)",
                "brand_alignment (matches brand voice)"
            ],
            "scoring_pattern": "STRONG/ACCEPTABLE/WEAK/FAIL with constraint violations = FAIL"
        }
    )
    patterns_loaded += 1

    logger.info(f"✅ Loaded {patterns_loaded} content generation patterns")
    return patterns_loaded


async def main():
    """Load all additional domain patterns."""
    try:
        logger.info("\n" + "="*70)
        logger.info("LOADING ADDITIONAL DOMAIN PATTERNS")
        logger.info("Expanding Athena to cover more system types")
        logger.info("="*70)

        total_patterns = 0

        # Load patterns for each domain
        total_patterns += await load_call_summary_patterns()
        total_patterns += await load_chat_copilot_patterns()
        total_patterns += await load_qa_system_patterns()
        total_patterns += await load_content_generation_patterns()

        logger.info("\n" + "="*70)
        logger.info(f"✅ COMPLETE! Loaded {total_patterns} additional patterns")
        logger.info("="*70)

        # Get updated stats
        service = get_dimension_pattern_service()
        stats = service.get_stats()

        logger.info("\n📊 Updated Pattern Statistics:")
        logger.info(f"  Total Patterns: {stats['total_patterns']}")
        logger.info("  By Category:")
        for category, count in sorted(stats['patterns_by_category'].items()):
            logger.info(f"    {category}: {count}")

        logger.info("\n🎯 Athena now has expert-level patterns for:")
        logger.info("  ✅ Diagnostics & Analysis (BI)")
        logger.info("  ✅ Recommendations (PAM)")
        logger.info("  ✅ Structured Output")
        logger.info("  ✅ Call Summary Generation (NEW)")
        logger.info("  ✅ Chat Copilots (NEW)")
        logger.info("  ✅ Q&A Systems (NEW)")
        logger.info("  ✅ Content Generation (NEW)")

        logger.info("\n🚀 Ready to generate 10/10 evals for all these domains!")

    except Exception as e:
        logger.error(f"Error loading patterns: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
