"""
Shared settings store - allows settings to be shared between server.py and project_api.py
"""

# Global settings store
settings_store = {}

def get_settings():
    """Get current settings"""
    return {
        "provider": settings_store.get("llm_provider", "openai"),
        "api_key": settings_store.get("api_key", ""),
        "model_name": settings_store.get("model_name")
    }

def update_settings(llm_provider: str, api_key: str, model_name: str):
    """Update settings"""
    settings_store.clear()
    settings_store["llm_provider"] = llm_provider
    settings_store["api_key"] = api_key
    settings_store["model_name"] = model_name
    return settings_store.copy()
