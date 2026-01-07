"""
Shared settings store - allows settings to be shared between server.py and project_api.py
Settings are persisted to a JSON file so they survive server restarts.
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

# Settings file path
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), ".athena_settings.json")

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
