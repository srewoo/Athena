"""
Shared settings store - allows settings to be shared between server.py and project_api.py
Settings are persisted to a JSON file so they survive server restarts.
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

# Settings file paths
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), ".athena_settings.json")
DOMAIN_CONTEXT_FILE = os.path.join(os.path.dirname(__file__), ".athena_domain_context.json")

# Global settings store (cached in memory)
settings_store = {}


def _load_from_file():
    """Load settings from file if it exists"""
    global settings_store
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                data = json.load(f)
                settings_store.update(data)
                logger.info(f"Loaded settings from {SETTINGS_FILE}")
                return True
        except Exception as e:
            logger.warning(f"Failed to load settings file: {e}")
    return False


def _save_to_file():
    """Save current settings to file"""
    try:
        # Don't save if no API key (nothing meaningful to persist)
        if not settings_store.get("api_key"):
            return False

        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings_store, f, indent=2)
        logger.info(f"Settings saved to {SETTINGS_FILE}")
        return True
    except Exception as e:
        logger.warning(f"Failed to save settings file: {e}")
        return False


# Load settings on module import
_load_from_file()


def get_settings():
    """Get current settings with file and environment variable fallback"""
    # Try to load from file if store is empty
    if not settings_store:
        _load_from_file()

    # 1. Determine Provider
    provider = settings_store.get("llm_provider") or os.getenv("LLM_PROVIDER", "openai")

    # 2. Determine API Key (fallback to env vars based on provider)
    api_key = settings_store.get("api_key")
    if not api_key:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
        elif provider == "google" or provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    # 3. Determine Model (fallback to env vars)
    model_name = settings_store.get("model_name")
    if not model_name:
        if provider == "openai":
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        elif provider == "anthropic":
            model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
        elif provider == "google":
            model_name = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")

    return {
        "provider": provider,
        "api_key": api_key or "",
        "model_name": model_name
    }


def update_settings(llm_provider: str, api_key: str, model_name: str):
    """Update settings and persist to file"""
    settings_store.clear()
    settings_store["llm_provider"] = llm_provider
    settings_store["api_key"] = api_key
    settings_store["model_name"] = model_name

    # Persist to file
    _save_to_file()

    return settings_store.copy()


# =============================================================================
# DOMAIN CONTEXT MANAGEMENT
# =============================================================================

# Default domain context template
DEFAULT_DOMAIN_CONTEXT = {
    "company": {
        "name": "",
        "industry": "",
        "description": ""
    },
    "prospects": [],
    "people": [],
    "products": [],
    "sample_queries": [],
    "industry": [],
    "domain_terminology": {},
    "customer_types": [],
    "common_scenarios": [],
    "sample_data": {}
}

# Global domain context store (cached in memory)
domain_context_store = {}


def _load_domain_context_from_file():
    """Load domain context from file if it exists"""
    global domain_context_store
    if os.path.exists(DOMAIN_CONTEXT_FILE):
        try:
            with open(DOMAIN_CONTEXT_FILE, 'r') as f:
                data = json.load(f)
                domain_context_store.clear()
                domain_context_store.update(data)
                logger.info(f"Loaded domain context from {DOMAIN_CONTEXT_FILE}")
                return True
        except Exception as e:
            logger.warning(f"Failed to load domain context file: {e}")
    return False


def _save_domain_context_to_file():
    """Save current domain context to file"""
    try:
        with open(DOMAIN_CONTEXT_FILE, 'w') as f:
            json.dump(domain_context_store, f, indent=2)
        logger.info(f"Domain context saved to {DOMAIN_CONTEXT_FILE}")
        return True
    except Exception as e:
        logger.warning(f"Failed to save domain context file: {e}")
        return False


# Load domain context on module import
_load_domain_context_from_file()


def get_domain_context():
    """Get current domain context"""
    if not domain_context_store:
        _load_domain_context_from_file()

    if not domain_context_store:
        return DEFAULT_DOMAIN_CONTEXT.copy()

    return domain_context_store.copy()


def update_domain_context(context: dict):
    """Update domain context and persist to file"""
    domain_context_store.clear()
    domain_context_store.update(context)
    _save_domain_context_to_file()
    return domain_context_store.copy()


def get_domain_context_for_prompt():
    """
    Get domain context formatted for injection into LLM prompts.
    Returns a formatted string that can be directly used in prompts.
    """
    context = get_domain_context()

    if not context or all(not v for v in context.values() if not isinstance(v, dict)):
        return None  # No context configured

    parts = ["## DOMAIN CONTEXT (Use this information to generate realistic, contextual test data)"]

    # Company info
    if context.get("company") and any(context["company"].values()):
        company = context["company"]
        if company.get("name"):
            parts.append(f"\n**Company:** {company['name']}")
        if company.get("industry"):
            parts.append(f"**Industry:** {company['industry']}")
        if company.get("description"):
            parts.append(f"**Description:** {company['description']}")

    # Prospects/Clients
    if context.get("prospects"):
        parts.append(f"\n**Prospects/Clients to reference:** {', '.join(context['prospects'][:15])}")

    # People names
    if context.get("people"):
        parts.append(f"\n**People names to use:** {', '.join(context['people'][:10])}")

    # Products
    if context.get("products"):
        parts.append(f"\n**Products/Services:** {', '.join(context['products'][:10])}")

    # Sample queries (for inspiration)
    if context.get("sample_queries"):
        parts.append(f"\n**Example realistic queries:**")
        for q in context["sample_queries"][:5]:
            parts.append(f"  - {q}")

    # Industries
    if context.get("industry"):
        industries = context["industry"]
        if isinstance(industries, list):
            parts.append(f"\n**Industries:** {', '.join(industries[:10])}")

    # Domain terminology
    if context.get("domain_terminology"):
        terms = context["domain_terminology"]
        if isinstance(terms, dict):
            parts.append(f"\n**Domain Terminology:**")
            for term, definition in list(terms.items())[:10]:
                parts.append(f"  - {term}: {definition}")
        elif isinstance(terms, list):
            parts.append(f"\n**Domain Terminology:** {', '.join(terms[:10])}")

    # Customer types
    if context.get("customer_types"):
        parts.append(f"\n**Customer Types:** {', '.join(context['customer_types'][:10])}")

    # Common scenarios
    if context.get("common_scenarios"):
        parts.append(f"\n**Common Scenarios:**")
        for scenario in context["common_scenarios"][:5]:
            parts.append(f"  - {scenario}")

    # Sample data
    if context.get("sample_data") and isinstance(context["sample_data"], dict):
        parts.append(f"\n**Sample Data Values:**")
        for key, values in list(context["sample_data"].items())[:5]:
            if isinstance(values, list):
                parts.append(f"  - {key}: {', '.join(str(v) for v in values[:5])}")
            else:
                parts.append(f"  - {key}: {values}")

    parts.append("\n**IMPORTANT:** Use the above context to make test cases realistic and relevant. Use actual names, products, and scenarios from this context instead of generic placeholders.")

    return "\n".join(parts)
