"""
Shared settings store - allows settings to be shared between server.py and project_api.py
"""

import os

# Global settings store
settings_store = {}

def get_settings():
    """Get current settings with environment variable fallback"""
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
            model_name = os.getenv("OPENAI_MODEL")
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
    """Update settings"""
    settings_store.clear()
    settings_store["llm_provider"] = llm_provider
    settings_store["api_key"] = api_key
    settings_store["model_name"] = model_name
    return settings_store.copy()
