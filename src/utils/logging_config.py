"""
Logging Configuration Module

Establishes a robust and structured logging system for the utils module and
the overall project, handling different log levels, output formats, and
optional advanced features like log rotation and JSON structured logging.
"""

import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


from .config import LoggingConfig


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    date_format: Optional[str] = None,
    file_path: Optional[str] = None,
    max_file_size: str = "10 MB",
    backup_count: int = 5,
    enable_rotation: bool = True,
    use_structured: bool = False,
    use_loguru: bool = True
) -> logging.Logger:
    """
    Setup logging configuration for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom log message format
        date_format: Custom date format for logs
        file_path: Optional path to log file
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
        enable_rotation: Enable log file rotation
        use_structured: Use structured JSON logging
        use_loguru: Use loguru if available (falls back to standard logging)

    Returns:
        Logger: Configured logger instance

    Raises:
        ValueError: If logging level is invalid
        OSError: If log file path is invalid
    """

    # Validate logging level
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if level.upper() not in valid_levels:
        raise ValueError(f"Invalid logging level: {level}. Must be one of {valid_levels}")

    # Try to use loguru if requested and available
    if use_loguru and LOGURU_AVAILABLE:
        return _setup_loguru_logging(
            level=level,
            format_string=format_string,
            date_format=date_format,
            file_path=file_path,
            max_file_size=max_file_size,
            backup_count=backup_count,
            enable_rotation=enable_rotation,
            use_structured=use_structured
        )

    # Fall back to standard logging
    return _setup_standard_logging(
        level=level,
        format_string=format_string,
        date_format=date_format,
        file_path=file_path,
        max_file_size=max_file_size,
        backup_count=backup_count,
        enable_rotation=enable_rotation,
        use_structured=use_structured
    )


def _setup_loguru_logging(
    level: str,
    format_string: Optional[str],
    date_format: Optional[str],
    file_path: Optional[str],
    max_file_size: str,
    backup_count: int,
    enable_rotation: bool,
    use_structured: bool
) -> logging.Logger:
    """Setup logging using loguru."""

    # Remove default loguru handler
    logger.remove()

    # Default format
    if format_string is None:
        if use_structured and STRUCTLOG_AVAILABLE:
            format_string = "{extra[timestamp]} | {level} | {name}:{function}:{line} | {message}"
        else:
            format_string = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    # Default date format
    if date_format is None:
        date_format = "YYYY-MM-DD HH:mm:ss"

    # Parse max file size
    try:
        if max_file_size.endswith("MB"):
            rotation_size = int(max_file_size.replace("MB", "").strip()) * 1024 * 1024
        elif max_file_size.endswith("GB"):
            rotation_size = int(max_file_size.replace("GB", "").strip()) * 1024 * 1024 * 1024
        else:
            rotation_size = int(max_file_size)
    except ValueError:
        rotation_size = 10 * 1024 * 1024  # Default 10 MB

    # Console handler
    logger.add(
        sys.stdout,
        format=format_string,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )

    # File handler
    if file_path:
        log_file_path = Path(file_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        if enable_rotation:
            logger.add(
                file_path,
                format=format_string,
                level=level,
                rotation=rotation_size,
                retention=backup_count,
                compression="zip",
                backtrace=True,
                diagnose=True
            )
        else:
            logger.add(
                file_path,
                format=format_string,
                level=level,
                backtrace=True,
                diagnose=True
            )

    return logger


def _setup_standard_logging(
    level: str,
    format_string: Optional[str],
    date_format: Optional[str],
    file_path: Optional[str],
    max_file_size: str,
    backup_count: int,
    enable_rotation: bool,
    use_structured: bool
) -> logging.Logger:
    """Setup logging using Python's standard logging module."""

    # Default format
    if format_string is None:
        if use_structured and STRUCTLOG_AVAILABLE:
            format_string = "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
        else:
            format_string = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # Default date format
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    if use_structured and STRUCTLOG_AVAILABLE:
        # Use structlog for structured JSON logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        logger = structlog.get_logger()
    else:
        # Use standard logging formatter
        formatter = logging.Formatter(format_string, datefmt=date_format)
        logger = root_logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter if not use_structured else None)
    root_logger.addHandler(console_handler)

    # File handler
    if file_path:
        log_file_path = Path(file_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        if enable_rotation:
            # Parse max file size
            try:
                if max_file_size.endswith("MB"):
                    max_bytes = int(max_file_size.replace("MB", "").strip()) * 1024 * 1024
                elif max_file_size.endswith("GB"):
                    max_bytes = int(max_file_size.replace("GB", "").strip()) * 1024 * 1024 * 1024
                else:
                    max_bytes = int(max_file_size)
            except ValueError:
                max_bytes = 10 * 1024 * 1024  # Default 10 MB

            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
        else:
            file_handler = logging.FileHandler(file_path, encoding='utf-8')

        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter if not use_structured else None)
        root_logger.addHandler(file_handler)

    return logger


def setup_logging_from_config(config: LoggingConfig, use_loguru: bool = True) -> logging.Logger:
    """
    Setup logging from a LoggingConfig object.

    Args:
        config: LoggingConfig instance
        use_loguru: Whether to use loguru if available

    Returns:
        Logger: Configured logger instance
    """
    return setup_logging(
        level=config.level,
        format_string=config.format if config.format != "%(asctime)s [%(levelname)s] %(name)s: %(message)s" else None,
        date_format=config.date_format if config.date_format != "%Y-%m-%d %H:%M:%S" else None,
        file_path=config.file_path,
        max_file_size=config.max_file_size,
        backup_count=config.backup_count,
        enable_rotation=config.enable_rotation,
        use_loguru=use_loguru
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name

    Returns:
        Logger: Logger instance
    """
    if LOGURU_AVAILABLE:
        return logger.bind(name=name)
    elif STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)


def log_system_info() -> None:
    """Log system information for debugging purposes."""
    import platform
    import sys

    logger = get_logger("system_info")

    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Python Executable: {sys.executable}")
    logger.info(f"Working Directory: {Path.cwd()}")

    # Log available logging libraries
    logger.info("=== Logging Libraries ===")
    logger.info(f"Loguru Available: {LOGURU_AVAILABLE}")
    logger.info(f"Structlog Available: {STRUCTLOG_AVAILABLE}")

    # Log installed packages
    try:
        import pkg_resources
        packages = ['pandas', 'numpy', 'pydantic', 'hmmlearn', 'scikit-learn']
        logger.info("=== Package Versions ===")
        for package in packages:
            try:
                version = pkg_resources.get_distribution(package).version
                logger.info(f"{package}: {version}")
            except pkg_resources.DistributionNotFound:
                logger.warning(f"{package}: Not installed")
    except ImportError:
        logger.warning("pkg_resources not available, cannot check package versions")


# Create a default logger instance
default_logger = None

def initialize_default_logging() -> logging.Logger:
    """Initialize default logging configuration."""
    global default_logger
    if default_logger is None:
        default_logger = setup_logging()
        log_system_info()
    return default_logger