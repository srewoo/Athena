"""
Security utilities for Athena
- API key encryption/decryption
- Hashing for validation
- Rate limiting
"""
import os
import base64
import hashlib
import secrets
import logging
from typing import Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from threading import Lock
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

# Get or generate encryption key
# In production, this should be set via environment variable
ENCRYPTION_KEY = os.environ.get("ATHENA_ENCRYPTION_KEY")
_fernet = None


def _get_fernet() -> Fernet:
    """Get or create Fernet instance for encryption"""
    global _fernet, ENCRYPTION_KEY

    if _fernet is not None:
        return _fernet

    if not ENCRYPTION_KEY:
        # Generate a key if not set (for development)
        # In production, always use environment variable
        key_file = ".encryption_key"
        if os.path.exists(key_file):
            with open(key_file, "rb") as f:
                ENCRYPTION_KEY = f.read().decode()
        else:
            ENCRYPTION_KEY = Fernet.generate_key().decode()
            with open(key_file, "wb") as f:
                f.write(ENCRYPTION_KEY.encode())
            logger.warning(
                "Generated new encryption key. Set ATHENA_ENCRYPTION_KEY "
                "environment variable in production!"
            )

    _fernet = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)
    return _fernet


def encrypt_api_key(api_key: str) -> Tuple[str, str]:
    """
    Encrypt an API key and return (encrypted_key, key_hash)
    The hash is used for quick validation without decryption
    """
    if not api_key:
        return "", ""

    fernet = _get_fernet()
    encrypted = fernet.encrypt(api_key.encode())

    # Create a hash for quick validation (first 8 + last 4 chars pattern)
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]

    return base64.b64encode(encrypted).decode(), key_hash


def decrypt_api_key(encrypted_key: str) -> Optional[str]:
    """Decrypt an API key"""
    if not encrypted_key:
        return None

    try:
        fernet = _get_fernet()
        decrypted = fernet.decrypt(base64.b64decode(encrypted_key))
        return decrypted.decode()
    except Exception as e:
        logger.error(f"Failed to decrypt API key: {e}")
        return None


def hash_api_key(api_key: str) -> str:
    """Create a hash of an API key for comparison"""
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def validate_api_key_format(api_key: str, provider: str) -> Tuple[bool, str]:
    """
    Validate API key format for a given provider
    Returns (is_valid, error_message)
    """
    if not api_key:
        return False, "API key is required"

    api_key = api_key.strip()

    if provider == "openai":
        if not api_key.startswith("sk-"):
            return False, "OpenAI API keys should start with 'sk-'"
        if len(api_key) < 40:
            return False, "OpenAI API key appears too short"

    elif provider == "claude" or provider == "anthropic":
        if not api_key.startswith("sk-ant-"):
            return False, "Anthropic API keys should start with 'sk-ant-'"
        if len(api_key) < 50:
            return False, "Anthropic API key appears too short"

    elif provider == "gemini" or provider == "google":
        if len(api_key) < 30:
            return False, "Google API key appears too short"

    return True, ""


def mask_api_key(api_key: str) -> str:
    """Mask an API key for display (show first 8 and last 4 chars)"""
    if not api_key or len(api_key) < 12:
        return "***"
    return f"{api_key[:8]}...{api_key[-4:]}"


# ============================================================================
# Rate Limiting
# ============================================================================

class RateLimiter:
    """
    Simple in-memory rate limiter
    For production, use Redis-based rate limiting
    """

    def __init__(self):
        self.requests = defaultdict(list)
        self.lock = Lock()

    def is_allowed(
        self,
        key: str,
        max_requests: int = 60,
        window_seconds: int = 60
    ) -> Tuple[bool, int]:
        """
        Check if a request is allowed under rate limits
        Returns (is_allowed, requests_remaining)
        """
        now = datetime.now()
        window_start = now - timedelta(seconds=window_seconds)

        with self.lock:
            # Clean old requests
            self.requests[key] = [
                ts for ts in self.requests[key]
                if ts > window_start
            ]

            current_count = len(self.requests[key])

            if current_count >= max_requests:
                return False, 0

            # Record this request
            self.requests[key].append(now)

            return True, max_requests - current_count - 1

    def get_reset_time(self, key: str, window_seconds: int = 60) -> int:
        """Get seconds until rate limit resets"""
        with self.lock:
            if key not in self.requests or not self.requests[key]:
                return 0

            oldest = min(self.requests[key])
            reset_time = oldest + timedelta(seconds=window_seconds)
            remaining = (reset_time - datetime.now()).total_seconds()

            return max(0, int(remaining))


# Global rate limiter instance
rate_limiter = RateLimiter()


# Rate limit configurations per endpoint type
RATE_LIMITS = {
    "llm_call": {"max_requests": 30, "window_seconds": 60},  # 30 LLM calls per minute
    "project_create": {"max_requests": 10, "window_seconds": 60},  # 10 projects per minute
    "test_run": {"max_requests": 5, "window_seconds": 60},  # 5 test runs per minute
    "general": {"max_requests": 100, "window_seconds": 60},  # 100 requests per minute
}


def check_rate_limit(
    client_id: str,
    endpoint_type: str = "general"
) -> Tuple[bool, int, int]:
    """
    Check rate limit for a client/endpoint combination
    Returns (is_allowed, requests_remaining, reset_seconds)
    """
    limits = RATE_LIMITS.get(endpoint_type, RATE_LIMITS["general"])
    key = f"{client_id}:{endpoint_type}"

    allowed, remaining = rate_limiter.is_allowed(
        key,
        max_requests=limits["max_requests"],
        window_seconds=limits["window_seconds"]
    )

    reset_time = rate_limiter.get_reset_time(key, limits["window_seconds"])

    return allowed, remaining, reset_time


# ============================================================================
# Request Validation
# ============================================================================

def generate_request_id() -> str:
    """Generate a unique request ID for tracing"""
    return f"req_{secrets.token_hex(8)}"


def sanitize_input(text: str, max_length: int = 100000) -> str:
    """
    Sanitize user input to prevent issues
    - Limit length
    - Remove null bytes
    - Normalize whitespace
    """
    if not text:
        return ""

    # Remove null bytes
    text = text.replace("\x00", "")

    # Limit length
    if len(text) > max_length:
        text = text[:max_length]

    return text


def validate_project_name(name: str) -> Tuple[bool, str]:
    """Validate project name"""
    if not name:
        return False, "Project name is required"

    name = name.strip()

    if len(name) < 2:
        return False, "Project name must be at least 2 characters"

    if len(name) > 200:
        return False, "Project name must be less than 200 characters"

    return True, ""
