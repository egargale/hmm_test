"""
Visualization & Reporting Module

Implements comprehensive visualization and reporting capabilities for HMM futures analysis,
including state visualization, interactive performance dashboards, and detailed regime reports.
"""

from .chart_generator import plot_states
from .dashboard_builder import build_dashboard
from .report_generator import generate_regime_report

__all__ = [
    'plot_states',
    'build_dashboard',
    'generate_regime_report'
]

# Version information
__version__ = "1.0.0"
