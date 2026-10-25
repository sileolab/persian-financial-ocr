"""Structured and rotating logging system for Hesabyar."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

from hesabyar.core.config import get_settings

_logger_initialized = False
console = Console(force_terminal=True, legacy_windows=False)


def setup_logger(
    name: str = "hesabyar",
    log_file: Optional[str] = None,
    log_level: Optional[str] = None,
) -> logging.Logger:
    """Configure and return the root Hesabyar logger with Console and Rotating File handlers."""
    global _logger_initialized
    settings = get_settings()

    level_name = (log_level or settings.log_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if setup is called multiple times
    if not logger.handlers:
        # 1. Rich Console Handler
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            markup=True,
        )
        rich_handler.setLevel(level)
        logger.addHandler(rich_handler)

        # 2. Rotating File Handler
        target_file = log_file or str(settings.log_file_path)
        try:
            file_handler = RotatingFileHandler(
                filename=target_file,
                maxBytes=settings.log_max_bytes,
                backupCount=settings.log_backup_count,
                encoding="utf-8",
            )
            file_formatter = logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)-8s] [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not configure file logger: {e}[/yellow]")

    _logger_initialized = True
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Retrieve an initialized logger, optionally namespaced."""
    if not _logger_initialized:
        setup_logger()

    base_name = "hesabyar"
    if name:
        if name.startswith(base_name):
            return logging.getLogger(name)
        return logging.getLogger(f"{base_name}.{name}")
    return logging.getLogger(base_name)
