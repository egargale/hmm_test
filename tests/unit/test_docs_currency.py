"""Issue #64 — README.md and SKILL.md must reflect all five engines.

Tests verify that documentation accurately lists the five engines
(threshold, messina, hmm, robust_hmm, fshmm) and mentions duration
forecasting, with current usage examples and output contracts.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _readme() -> str:
    return (REPO_ROOT / "README.md").read_text()


def _skill() -> str:
    return (REPO_ROOT / "SKILL.md").read_text()


# ── Engine count ─────────────────────────────────────────────────────────


FIVE_ENGINES = ["threshold", "messina", "hmm", "robust_hmm", "fshmm"]


class TestEngineCount:
    """READ#1: first-paragraph engine count says five, not three."""

    def test_readme_says_five_engines(self):
        """README must say 'five' (or 'all') engines, not 'three'."""
        text = _readme()
        # The first paragraph or description must reference all five engines
        # Accept: "five independent engines" or similar
        assert "five" in text.lower()[:3000], (
            "README first ~3K chars should mention 'five' engines"
        )
        assert "three" not in text.lower()[:500], (
            "README should not say 'three' engines near the top"
        )

    def test_skill_says_five_engines(self):
        """SKILL.md must also say 'five' (or 'all') engines, not 'three'."""
        text = _skill()
        assert "five" in text.lower()[:3000], (
            "SKILL.md first ~3K chars should mention 'five' engines"
        )
        assert "three" not in text.lower()[:500], (
            "SKILL.md should not say 'three' engines near the top"
        )


class TestRobustHmmUsage:
    """READ#4: robust_hmm usage example with --robust-method."""

    def test_readme_has_robust_hmm_example(self):
        """README must include a robust_hmm usage example."""
        text = _readme()
        assert "robust_hmm" in text, "README should mention robust_hmm"
        assert "--robust-method" in text, (
            "README robust_hmm example should show --robust-method"
        )

    def test_skill_has_robust_hmm_example(self):
        """SKILL.md must include a robust_hmm usage example."""
        text = _skill()
        assert "robust_hmm" in text, "SKILL.md should mention robust_hmm"
        assert "--robust-method" in text, (
            "SKILL.md robust_hmm example should show --robust-method"
        )


class TestFshmmUsage:
    """READ#5: fshmm usage example with --saliency-threshold."""

    def test_readme_has_fshmm_example(self):
        """README must include an fshmm usage example."""
        text = _readme()
        assert "fshmm" in text, "README should mention fshmm"
        assert "--saliency-threshold" in text, (
            "README fshmm example should show --saliency-threshold"
        )

    def test_skill_has_fshmm_example(self):
        """SKILL.md must include an fshmm usage example."""
        text = _skill()
        assert "fshmm" in text, "SKILL.md should mention fshmm"
        assert "--saliency-threshold" in text, (
            "SKILL.md fshmm example should show --saliency-threshold"
        )


class TestDurationForecastUsage:
    """READ#6: duration forecast usage example."""

    def test_readme_has_duration_forecast_example(self):
        """README must include a --duration-forecast usage example."""
        text = _readme()
        assert "--duration-forecast" in text, (
            "README should have a --duration-forecast usage example"
        )

    def test_skill_has_duration_forecast_example(self):
        """SKILL.md must include a --duration-forecast usage example."""
        text = _skill()
        assert "--duration-forecast" in text, (
            "SKILL.md should have a --duration-forecast usage example"
        )


class TestOutputContract:
    """READ#7-9: output contract includes new fields or references SKILL.md."""

    def test_readme_contract_has_feature_saliency(self):
        """README output JSON should include feature_saliency in engine_info."""
        text = _readme()
        assert "feature_saliency" in text or "SKILL.md" in text, (
            "README output contract should include engine_info.feature_saliency "
            "or point to SKILL.md"
        )

    def test_readme_contract_has_selected_features(self):
        """README output JSON should include selected_features in engine_info."""
        text = _readme()
        assert "selected_features" in text or "SKILL.md" in text, (
            "README output contract should include engine_info.selected_features "
            "or point to SKILL.md"
        )

    def test_readme_contract_has_duration_forecast(self):
        """README output JSON should include duration_forecast block."""
        text = _readme()
        assert "duration_forecast" in text or "SKILL.md" in text, (
            "README output contract should include duration_forecast block "
            "or point to SKILL.md"
        )

    def test_skill_contract_has_feature_saliency(self):
        """SKILL.md output JSON should include feature_saliency in engine_info."""
        text = _skill()
        assert "feature_saliency" in text, (
            "SKILL.md output contract should include engine_info.feature_saliency"
        )

    def test_skill_contract_has_selected_features(self):
        """SKILL.md output JSON should include selected_features."""
        text = _skill()
        assert "selected_features" in text, (
            "SKILL.md output contract should include engine_info.selected_features"
        )

    def test_skill_contract_has_duration_forecast(self):
        """SKILL.md output JSON should include duration_forecast block."""
        text = _skill()
        assert "duration_forecast" in text, (
            "SKILL.md output contract should include duration_forecast"
        )


class TestDurationForecast:
    """READ#3: both docs mention duration forecasting."""

    def test_readme_mentions_duration_forecast(self):
        """README must mention duration forecasting."""
        text = _readme().lower()
        assert "duration" in text, "README should mention duration forecasting"

    def test_skill_mentions_duration_forecast(self):
        """SKILL.md must mention duration forecasting."""
        text = _skill().lower()
        assert "duration" in text, "SKILL.md should mention duration forecasting"


class TestAllFiveEnginesNamed:
    """READ#2: both docs list all five engines by name."""

    def test_readme_lists_all_five_engines(self):
        """README must mention all five engine names."""
        text = _readme().lower()
        missing = [e for e in FIVE_ENGINES if e not in text]
        assert not missing, f"README missing engine name(s): {missing}"

    def test_skill_lists_all_five_engines(self):
        """SKILL.md must mention all five engine names."""
        text = _skill().lower()
        missing = [e for e in FIVE_ENGINES if e not in text]
        assert not missing, f"SKILL.md missing engine name(s): {missing}"
