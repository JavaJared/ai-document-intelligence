"""
Structured logging configuration for production-grade observability.

Provides consistent logging across the application with JSON formatting
for easy parsing and monitoring in cloud environments.
"""

import sys
import logging
from typing import Any, Dict
import structlog
from pythonjsonlogger import jsonlogger


def setup_logging(log_level: str = "INFO", json_format: bool = True) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Whether to output logs in JSON format
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )
    
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


class Logger:
    """
    Application logger with contextual information.
    
    Provides structured logging with automatic context injection
    for tracing requests and operations.
    """
    
    def __init__(self, name: str):
        """
        Initialize logger with a name.
        
        Args:
            name: Logger name (typically module name)
        """
        self._logger = structlog.get_logger(name)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with context."""
        self._logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with context."""
        self._logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with context."""
        self._logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message with context."""
        self._logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message with context."""
        self._logger.critical(message, **kwargs)
    
    def bind(self, **kwargs: Any) -> "Logger":
        """
        Create a new logger with bound context.
        
        Args:
            **kwargs: Context to bind to logger
            
        Returns:
            New logger instance with bound context
        """
        new_logger = Logger.__new__(Logger)
        new_logger._logger = self._logger.bind(**kwargs)
        return new_logger


def get_logger(name: str) -> Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return Logger(name)


# Initialize logging on module import
setup_logging()
