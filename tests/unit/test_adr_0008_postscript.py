"""Issue #68 — ADR-0008 postscript: document later dead-module removals.

Tests verify that ADR-0008 (docs/adr/0008-excise-dead-weight.md) has a
postscript listing subsequent deletions that were consistent with the
original ADR's "deletion test" criterion (zero callers in any engine path).
"""

from pathlib import Path

ADR = (
    Path(__file__).resolve().parent.parent.parent
    / "docs"
    / "adr"
    / "0008-excise-dead-weight.md"
)


def _adr() -> str:
    return ADR.read_text()


class TestPostscriptSection:
    """The ADR must end with a '## Postscript' section."""

    def test_postscript_section_exists(self):
        """ADR-0008 must have a '## Postscript' section at the end."""
        text = _adr()
        assert "## Postscript" in text, (
            "ADR-0008 should contain a '## Postscript' section"
        )


class TestPostscriptCsvDeletions:
    """CSV parser/detector/validation deletions from e0b772e."""

    def test_mentions_csv_parser_removed(self):
        """Postscript must mention csv_parser.py deletion with commit."""
        text = _adr()
        assert "csv_parser" in text, "Postscript should mention csv_parser.py removal"
        assert "e0b772e" in text, "Postscript should reference commit e0b772e"

    def test_mentions_csv_format_detector_removed(self):
        """Postscript must mention csv_format_detector.py deletion."""
        text = _adr()
        assert "csv_format_detector" in text, (
            "Postscript should mention csv_format_detector.py removal"
        )

    def test_mentions_data_validation_removed(self):
        """Postscript must mention data_validation.py deletion."""
        text = _adr()
        assert "data_validation" in text, (
            "Postscript should mention data_validation.py removal"
        )


class TestPostscriptDataclassRemovals:
    """Dataclass deletions from bbbb91c."""

    def test_mentions_dataclass_removals(self):
        """Postscript must mention dead dataclass removals from data_types.py."""
        text = _adr()
        assert "bbbb91c" in text, "Postscript should reference commit bbbb91c"
        assert "dataclasses" in text.lower() or "data_types" in text, (
            "Postscript should mention dataclass or data_types.py cleanup"
        )


class TestPostscriptFunctionRemovals:
    """Function deletions from 6cc0c79."""

    def test_mentions_function_removals(self):
        """Postscript must mention dead function removals with commit."""
        text = _adr()
        assert "6cc0c79" in text, "Postscript should reference commit 6cc0c79"
        assert "performance_metrics" in text, (
            "Postscript should mention performance_metrics.py cleanup"
        )
        assert "technical_indicators" in text, (
            "Postscript should mention technical_indicators.py cleanup"
        )


class TestPostscriptConsistency:
    """Postscript must note consistency with original ADR's deletion test."""

    def test_mentions_deletion_test(self):
        """Postscript must reference ADR-0008's deletion test criterion."""
        text = _adr()
        assert "deletion test" in text.lower() or "zero callers" in text.lower(), (
            "Postscript should reference the 'deletion test' or 'zero callers' criterion"
        )

    def test_says_consistent(self):
        """Postscript must state these removals are consistent with ADR-0008."""
        text = _adr()
        assert "consistent" in text.lower() or "consistent with" in text.lower(), (
            "Postscript should say removals are consistent with ADR-0008"
        )
