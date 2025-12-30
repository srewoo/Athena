"""
Structured Logging Configuration for Athena
Features:
- JSON formatted logs for production
- Request correlation IDs
- Performance metrics logging
- Error tracking with context
"""
import logging
import json
import sys
import time
import uuid
from typing import Optional, Dict, Any
from datetime import datetime
from functools import wraps
from contextvars import ContextVar

# Context variable for request ID
request_id_var: ContextVar[str] = ContextVar('request_id', default='')


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add request ID if available
        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id

        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class DevFormatter(logging.Formatter):
    """Human-readable formatter for development"""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        request_id = request_id_var.get()
        req_str = f"[{request_id[:8]}] " if request_id else ""

        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        message = f"{color}{timestamp}{reset} {req_str}{color}{record.levelname:8}{reset} {record.name}: {record.getMessage()}"

        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"

        return message


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None
):
    """
    Configure logging for the application

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON formatting (for production)
        log_file: Optional file to write logs to
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    if json_format:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(DevFormatter())

    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(StructuredFormatter())  # Always JSON for files
        root_logger.addHandler(file_handler)

    # Set log levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name"""
    return logging.getLogger(name)


def generate_request_id() -> str:
    """Generate a unique request ID"""
    return str(uuid.uuid4())


def set_request_id(request_id: str):
    """Set the request ID for the current context"""
    request_id_var.set(request_id)


def get_request_id() -> str:
    """Get the request ID for the current context"""
    return request_id_var.get()


class LogContext:
    """Context manager for adding extra data to logs"""

    def __init__(self, **kwargs):
        self.data = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def log_with_context(logger: logging.Logger, level: str, message: str, **extra):
    """Log a message with extra context data"""
    record = logger.makeRecord(
        logger.name,
        getattr(logging, level.upper()),
        "(unknown)",
        0,
        message,
        (),
        None
    )
    record.extra_data = extra
    logger.handle(record)


# Performance logging decorator
def log_performance(logger: logging.Logger, operation: str):
    """Decorator to log function performance"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            request_id = get_request_id()

            try:
                result = await func(*args, **kwargs)
                duration_ms = int((time.time() - start_time) * 1000)

                log_with_context(
                    logger, "INFO",
                    f"{operation} completed",
                    operation=operation,
                    duration_ms=duration_ms,
                    status="success",
                    request_id=request_id
                )

                return result

            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)

                log_with_context(
                    logger, "ERROR",
                    f"{operation} failed: {str(e)}",
                    operation=operation,
                    duration_ms=duration_ms,
                    status="error",
                    error=str(e),
                    request_id=request_id
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            request_id = get_request_id()

            try:
                result = func(*args, **kwargs)
                duration_ms = int((time.time() - start_time) * 1000)

                log_with_context(
                    logger, "INFO",
                    f"{operation} completed",
                    operation=operation,
                    duration_ms=duration_ms,
                    status="success",
                    request_id=request_id
                )

                return result

            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)

                log_with_context(
                    logger, "ERROR",
                    f"{operation} failed: {str(e)}",
                    operation=operation,
                    duration_ms=duration_ms,
                    status="error",
                    error=str(e),
                    request_id=request_id
                )
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Metrics tracking
class MetricsCollector:
    """Simple in-memory metrics collector"""

    def __init__(self):
        self.counters: Dict[str, int] = {}
        self.timings: Dict[str, list] = {}

    def increment(self, metric: str, value: int = 1):
        """Increment a counter"""
        self.counters[metric] = self.counters.get(metric, 0) + value

    def record_timing(self, metric: str, duration_ms: int):
        """Record a timing measurement"""
        if metric not in self.timings:
            self.timings[metric] = []
        self.timings[metric].append(duration_ms)

        # Keep only last 1000 measurements
        if len(self.timings[metric]) > 1000:
            self.timings[metric] = self.timings[metric][-1000:]

    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics stats"""
        stats = {
            "counters": self.counters.copy(),
            "timings": {}
        }

        for metric, values in self.timings.items():
            if values:
                stats["timings"][metric] = {
                    "count": len(values),
                    "avg_ms": sum(values) / len(values),
                    "min_ms": min(values),
                    "max_ms": max(values),
                    "p50_ms": sorted(values)[len(values) // 2],
                    "p95_ms": sorted(values)[int(len(values) * 0.95)] if len(values) >= 20 else None
                }

        return stats

    def reset(self):
        """Reset all metrics"""
        self.counters = {}
        self.timings = {}


# Global metrics instance
metrics = MetricsCollector()
