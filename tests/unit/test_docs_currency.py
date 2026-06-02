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


# ── Issue #67: configuration.md currency ────────────────────────────────


CONFIG_MD = REPO_ROOT / "references" / "configuration.md"


def _config() -> str:
    return CONFIG_MD.read_text()


EXPECTED_PARAMS = [
    "--engine",
    "--window",
    "--threshold",
    "--min-train",
    "--n-states",
    "--dwell-bars",
    "--hysteresis",
    "--duration-forecast",
    "--saliency-threshold",
    "--saliency-output",
    "--robust-method",
]


class TestConfigurationMdAllFiveEngines:
    """Issue #67: All five engines documented with CLI params."""

    def test_config_mentions_all_five_engines(self):
        """configuration.md must mention all five engine names."""
        text = _config().lower()
        missing = [e for e in FIVE_ENGINES if e not in text]
        assert not missing, f"configuration.md missing engine name(s): {missing}"

    def test_config_has_engine_dispatch_section(self):
        """configuration.md must document --engine flag."""
        text = _config()
        assert "--engine" in text, "configuration.md must document the --engine flag"


class TestConfigurationMdDefaultsWithEngine:
    """Issue #67: Recommended defaults include engine choice per asset class."""

    def test_defaults_table_has_engine_column(self):
        """Default recommendations table must include engine column."""
        text = _config()
        # Find the recommended defaults table
        rd_start = text.find("### Recommended Defaults by Asset Class")
        assert rd_start >= 0, "No Recommended Defaults section found"
        rd_section = text[rd_start : rd_start + 1500]
        assert "engine" in rd_section.lower(), (
            "Recommended defaults table should include an engine column"
        )

    def test_defaults_table_has_multiple_engines(self):
        """Defaults table should recommend different engines per asset class."""
        text = _config()
        rd_start = text.find("### Recommended Defaults by Asset Class")
        assert rd_start >= 0
        rd_section = text[rd_start : rd_start + 1500].lower()
        # Should mention at least two different engines
        engines_found = [e for e in FIVE_ENGINES if e in rd_section]
        assert len(engines_found) >= 2, (
            f"Defaults table should recommend at least 2 different engines, "
            f"found only: {engines_found}"
        )


class TestConfigurationMdGridSearchExamples:
    """Issue #67: Grid search covers both threshold and HMM tuning."""

    def test_grid_search_has_threshold_example(self):
        """Grid search section must include threshold window/threshold example."""
        text = _config()
        assert "window" in text and "threshold" in text, (
            "Grid search should include threshold tuning (window, threshold)"
        )

    def test_grid_search_has_hmm_example(self):
        """Grid search section must include HMM engine tuning example."""
        text = _config()
        # The grid search section should reference engine and n-states in tuning context
        # Find the grid search subsection
        gs_start = text.find("### Grid Search Pattern")
        assert gs_start >= 0, "No Grid Search Pattern section found"
        gs_section = text[gs_start : gs_start + 4000]
        assert "HMMGenericEngine" in gs_section or "--engine" in gs_section, (
            "Grid search should include HMM engine tuning"
        )
        assert "n_states" in gs_section, (
            "Grid search should include HMM state count tuning (n_states)"
        )

    def test_grid_search_mentions_dwell_or_hysteresis(self):
        """Grid search should mention dwell-bars or hysteresis tuning."""
        text = _config()
        assert "dwell" in text.lower() or "hysteresis" in text.lower(), (
            "Grid search should mention dwell-bars or hysteresis tuning"
        )


class TestConfigurationMdAllCLIParams:
    """Issue #67: All current CLI parameters documented."""

    def test_all_params_documented(self):
        """configuration.md must document every expected CLI parameter."""
        text = _config()
        missing = [p for p in EXPECTED_PARAMS if p not in text]
        assert not missing, f"configuration.md missing CLI parameter docs: {missing}"


class TestConfigurationMdNoStaleHmmFlags:
    """Issue #67: No stale --hmm/--no-hmm references."""

    def test_no_hmm_flag_in_section_header(self):
        """The `--hmm` / `--no-hmm` section header is removed."""
        text = _config()
        assert "--hmm" not in text, (
            "configuration.md should not contain stale '--hmm' references; "
            "use '--engine' instead"
        )

    def test_no_hmm_flag_in_descriptions(self):
        """The `--n-states` section no longer says 'Only relevant when --hmm'."""
        text = _config()
        assert "Only relevant when" not in text, (
            "configuration.md should not qualify --n-states as HMM-only"
        )
        assert "--no-hmm" not in text, "configuration.md should not contain '--no-hmm'"
