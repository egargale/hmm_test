"""Minimal logging utilities — stdlib only."""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a stdlib ``logging.Logger`` for *name*."""
    return logging.getLogger(name)


def suppress_stdout_logging() -> None:
    """Remove every logging handler that writes to stdout.

    Used in JSON mode so that only the JSON payload appears on stdout.
    """
    for handler in logging.root.handlers[:]:
        if getattr(handler, "stream", None) is sys.stdout:
            logging.root.removeHandler(handler)
