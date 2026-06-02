"""Tests for the slim logging_config module (Issue #60)."""

import logging
import sys

from hmm_futures_analysis.utils.logging_config import (
    get_logger,
    suppress_stdout_logging,
)


def test_get_logger_returns_stdlib_logger():
    """get_logger returns a stdlib logging.Logger with the requested name."""
    log = get_logger("test_module")
    assert isinstance(log, logging.Logger)
    assert log.name == "test_module"


def test_suppress_stdout_logging_removes_stdout_handlers():
    """suppress_stdout_logging removes handlers writing to stdout."""

    root = logging.getLogger()
    saved = root.handlers[:]
    try:
        root.handlers.clear()
        stdout_handler = logging.StreamHandler(sys.stdout)
        stderr_handler = logging.StreamHandler(sys.stderr)
        root.addHandler(stdout_handler)
        root.addHandler(stderr_handler)

        suppress_stdout_logging()

        streams = [
            h.stream for h in root.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert sys.stdout not in streams
        assert sys.stderr in streams
    finally:
        root.handlers = saved


def test_no_loguru_structlog_pydantic_imports():
    """Module must not import loguru, structlog, or pydantic."""
    import pathlib
    import re

    source = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "hmm_futures_analysis"
        / "utils"
        / "logging_config.py"
    ).read_text()
    for lib in ("loguru", "structlog", "pydantic"):
        assert not re.search(rf"\bimport\s+{lib}\b", source), (
            f"logging_config.py still imports {lib}"
        )
        assert not re.search(rf"\bfrom\s+{lib}\b", source), (
            f"logging_config.py still imports from {lib}"
        )


def test_module_under_50_lines():
    """logging_config.py must be 50 lines or fewer."""
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "hmm_futures_analysis"
        / "utils"
        / "logging_config.py"
    ).read_text()
    line_count = len(source.splitlines())
    assert line_count <= 50, f"logging_config.py is {line_count} lines (max 50)"


def test_cli_uses_shared_suppress_stdout_logging():
    """cli.py must import suppress_stdout_logging from utils, not define its own."""
    import pathlib
    import re

    source = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "hmm_futures_analysis"
        / "cli.py"
    ).read_text()
    # Must not define a local _suppress_stdout_logging
    assert not re.search(r"def _suppress_stdout_logging", source), (
        "cli.py still defines a local _suppress_stdout_logging"
    )
    # Must import from the utils module
    assert re.search(
        r"from \.utils(\.logging_config)? import.*suppress_stdout_logging", source
    ), "cli.py must import suppress_stdout_logging from utils"
