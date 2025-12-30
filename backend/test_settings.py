import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from shared_settings import get_settings, update_settings

def test_settings_fallback():
    print("Testing settings fallback...")
    
    # ensure store is empty
    update_settings("", "", "")
    
    # Set env var
    os.environ["OPENAI_API_KEY"] = "sk-test-key-from-env"
    os.environ["LLM_PROVIDER"] = "openai"
    
    settings = get_settings()
    
    if settings["api_key"] == "sk-test-key-from-env":
        print("SUCCESS: API keys loaded from environment.")
    else:
        print(f"FAIL: Expected 'sk-test-key-from-env', got '{settings['api_key']}'")

if __name__ == "__main__":
    test_settings_fallback()
