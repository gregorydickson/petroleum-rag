"""Logging utilities for the petroleum RAG benchmark system.

This module provides structured logging setup with proper formatting,
log levels, and optional file output.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from config import settings


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured log output.

    Formats log messages with timestamp, level, module, and message.
    Uses colors for console output (if supported).
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True) -> None:
        """Initialize the formatter.

        Args:
            use_colors: Whether to use ANSI colors in output
        """
        super().__init__()
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record.

        Args:
            record: Log record to format

        Returns:
            Formatted log string
        """
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()

        # Get log level with optional color
        level = record.levelname
        if self.use_colors and sys.stdout.isatty():
            level = f"{self.COLORS.get(record.levelname, '')}{level}{self.RESET}"

        # Build module path
        module = f"{record.name}"
        if record.funcName and record.funcName != "<module>":
            module = f"{module}.{record.funcName}"

        # Format message
        message = record.getMessage()

        # Add exception info if present
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"

        return f"[{timestamp}] {level:8} | {module:30} | {message}"


def setup_logging(
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = None,
    log_file: str | Path | None = None,
    use_colors: bool = True,
) -> logging.Logger:
    """Set up logging configuration for the application.

    Args:
        log_level: Logging level (default from settings)
        log_file: Optional log file path (default from settings)
        use_colors: Whether to use colors in console output

    Returns:
        Configured root logger
    """
    # Get log level from settings if not provided
    if log_level is None:
        log_level = settings.log_level

    # Get log file from settings if not provided
    if log_file is None and settings.log_file:
        log_file = settings.log_file

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(StructuredFormatter(use_colors=use_colors))
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(StructuredFormatter(use_colors=False))
        root_logger.addHandler(file_handler)

    # Log startup
    root_logger.info(f"Logging initialized at {log_level} level")
    if log_file:
        root_logger.info(f"Logging to file: {log_file}")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Default logger for this module
logger = get_logger(__name__)
