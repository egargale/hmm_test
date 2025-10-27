"""
Bias Prevention Module

Implements lookahead bias detection and prevention mechanisms for ensuring
realistic backtesting results without information leakage from future data.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils import get_logger
from utils.data_types import BacktestResult

logger = get_logger(__name__)


@dataclass
class BiasDetectionResult:
    """Container for bias detection results."""
    has_lookahead_bias: bool
    timing_violations: List[Dict[str, Any]]
    data_leakage_detected: bool
    leakage_sources: List[str]
    position_shift_violations: List[Dict[str, Any]]
    overall_risk_score: float
    recommendations: List[str]


def validate_timing_consistency(
    states: np.ndarray,
    decisions: np.ndarray,
    timestamps: pd.DatetimeIndex,
    lag_periods: int = 1
) -> List[Dict[str, Any]]:
    """
    Validate timing consistency to ensure no lookahead bias.

    Args:
        states: HMM state sequence
        decisions: Trading decisions made based on states
        timestamps: Timestamps for each decision
        lag_periods: Expected lag periods between state and decision

    Returns:
        List of timing violations detected
    """
    violations = []

    if len(states) != len(decisions) or len(states) != len(timestamps):
        violations.append({
            'type': 'length_mismatch',
            'message': f"Length mismatch: states={len(states)}, decisions={len(decisions)}, timestamps={len(timestamps)}"
        })
        return violations

    logger.debug(f"Checking timing consistency for {len(states)} periods with lag={lag_periods}")

    # Check that decisions are based on past information only
    for i in range(lag_periods, len(states)):
        decision_state = decisions[i]
        available_state = states[i - lag_periods]

        if decision_state != available_state:
            violations.append({
                'type': 'timing_violation',
                'timestamp': timestamps[i],
                'index': i,
                'decision_state': decision_state,
                'available_state': available_state,
                'message': f"Decision at {timestamps[i]} used state {decision_state}, but state {available_state} should have been available"
            })

    # Check for initial periods where lag cannot be applied
    for i in range(min(lag_periods, len(states))):
        if decisions[i] != 0:  # Any non-zero decision before lag is possible
            violations.append({
                'type': 'early_decision',
                'timestamp': timestamps[i],
                'index': i,
                'decision': decisions[i],
                'message': f"Decision made at {timestamps[i]} before sufficient lag period"
            })

    logger.debug(f"Found {len(violations)} timing violations")
    return violations


def validate_feature_availability(
    features: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    feature_lookback_periods: Optional[Dict[str, int]] = None
) -> List[str]:
    """
    Validate that features used for decisions were actually available at decision time.

    Args:
        features: Feature DataFrame with datetime index
        timestamps: Decision timestamps
        feature_lookback_periods: Required lookback periods for each feature

    Returns:
        List of detected data leakage sources
    """
    leakage_sources = []

    if feature_lookback_periods is None:
        # Default lookback periods for common technical indicators
        feature_lookback_periods = {
            'sma_20': 20,
            'sma_50': 50,
            'ema_20': 20,
            'rsi_14': 14,
            'bollinger_upper': 20,
            'bollinger_lower': 20,
            'atr_14': 14,
            'macd': 26,  # Typically 26 for slow line
            'macd_signal': 9,
        }

    logger.debug(f"Validating feature availability for {len(features)} features")

    # Check each feature for NaN values that indicate insufficient history
    for feature_name in features.columns:
        feature_data = features[feature_name]
        required_periods = feature_lookback_periods.get(feature_name, 1)

        # Find timestamps where feature should be valid but is NaN
        for timestamp in timestamps[required_periods:]:
            if timestamp in feature_data.index and pd.isna(feature_data.loc[timestamp]):
                leakage_sources.append(
                    f"Feature '{feature_name}' contains NaN values at decision time {timestamp}, "
                    f"indicating insufficient lookback period (requires {required_periods} periods)"
                )

        # Check for forward-looking indicators that might use future data
        if 'future' in feature_name.lower() or 'ahead' in feature_name.lower():
            leakage_sources.append(
                f"Feature '{feature_name}' appears to be forward-looking based on its name"
            )

    logger.debug(f"Found {len(leakage_sources)} potential data leakage sources")
    return leakage_sources


def validate_position_shifting(
    positions: pd.Series,
    states: np.ndarray,
    state_map: Dict[int, int],
    timestamps: pd.DatetimeIndex
) -> List[Dict[str, Any]]:
    """
    Validate that position shifting is correctly applied to prevent lookahead bias.

    Args:
        positions: Position series from backtest
        states: Original state sequence
        state_map: State to position mapping
        timestamps: Timestamps for alignment

    Returns:
        List of position shift violations
    """
    violations = []

    if len(positions) != len(states) or len(positions) != len(timestamps):
        violations.append({
            'type': 'length_mismatch',
            'message': f"Length mismatch between positions ({len(positions)}) and states ({len(states)})"
        })
        return violations

    logger.debug("Validating position shifting consistency")

    # Calculate expected positions with proper lagging
    expected_positions = np.zeros_like(states)
    for i in range(1, len(states)):
        # Decision at time i should be based on state at time i-1
        prev_state = int(states[i-1])
        expected_positions[i] = state_map.get(prev_state, 0)

    # Compare actual vs expected positions
    position_mismatches = positions.values != expected_positions
    mismatch_count = np.sum(position_mismatches)

    if mismatch_count > 0:
        mismatch_indices = np.where(position_mismatches)[0]

        for idx in mismatch_indices[:10]:  # Limit to first 10 violations for brevity
            violations.append({
                'type': 'position_shift_violation',
                'timestamp': timestamps[idx],
                'index': idx,
                'actual_position': positions.iloc[idx],
                'expected_position': expected_positions[idx],
                'state': int(states[idx]),
                'prev_state': int(states[idx-1]) if idx > 0 else None,
                'message': f"Position shift violation at {timestamps[idx]}: actual={positions.iloc[idx]}, expected={expected_positions[idx]}"
            })

        if mismatch_count > 10:
            violations.append({
                'type': 'additional_violations',
                'count': mismatch_count - 10,
                'message': f"... and {mismatch_count - 10} additional position shift violations"
            })

    logger.debug(f"Found {len(violations)} position shift violations")
    return violations


def detect_lookahead_bias(
    states: np.ndarray,
    positions: pd.Series,
    timestamps: pd.DatetimeIndex,
    state_map: Dict[int, int],
    features: Optional[pd.DataFrame] = None,
    lag_periods: int = 1
) -> BiasDetectionResult:
    """
    Comprehensive lookahead bias detection analysis.

    Args:
        states: HMM state sequence
        positions: Position series from backtest
        timestamps: Timestamps for alignment
        state_map: State to position mapping
        features: Optional feature DataFrame for data leakage detection
        lag_periods: Expected lag periods between state and decision

    Returns:
        Comprehensive bias detection results
    """
    logger.info("Starting comprehensive lookahead bias detection")

    # Initialize results
    timing_violations = []
    leakage_sources = []
    position_shift_violations = []

    # Check timing consistency
    timing_violations = validate_timing_consistency(
        states, positions.values, timestamps, lag_periods
    )

    # Check for data leakage in features
    data_leakage_detected = False
    if features is not None:
        leakage_sources = validate_feature_availability(features, timestamps)
        data_leakage_detected = len(leakage_sources) > 0

    # Validate position shifting
    position_shift_violations = validate_position_shifting(
        positions, states, state_map, timestamps
    )

    # Calculate overall risk score
    total_violations = len(timing_violations) + len(leakage_sources) + len(position_shift_violations)
    max_possible_violations = len(states) * 3  # Rough estimate
    overall_risk_score = min(total_violations / max(max_possible_violations, 1), 1.0)

    # Determine if lookahead bias exists
    has_lookahead_bias = (
        len(timing_violations) > 0 or
        data_leakage_detected or
        len(position_shift_violations) > 0
    )

    # Generate recommendations
    recommendations = generate_bias_recommendations(
        timing_violations, leakage_sources, position_shift_violations
    )

    result = BiasDetectionResult(
        has_lookahead_bias=has_lookahead_bias,
        timing_violations=timing_violations,
        data_leakage_detected=data_leakage_detected,
        leakage_sources=leakage_sources,
        position_shift_violations=position_shift_violations,
        overall_risk_score=overall_risk_score,
        recommendations=recommendations
    )

    logger.info(f"Bias detection completed: risk_score={overall_risk_score:.3f}, "
                f"violations={total_violations}, has_bias={has_lookahead_bias}")

    return result


def generate_bias_recommendations(
    timing_violations: List[Dict[str, Any]],
    leakage_sources: List[str],
    position_shift_violations: List[Dict[str, Any]]
) -> List[str]:
    """
    Generate recommendations based on detected bias issues.

    Args:
        timing_violations: List of timing violations
        leakage_sources: List of data leakage sources
        position_shift_violations: List of position shift violations

    Returns:
        List of actionable recommendations
    """
    recommendations = []

    if timing_violations:
        recommendations.append(
            "⚠️ TIMING VIOLATIONS: Ensure decisions are based only on information available "
            "at decision time. Implement proper state lagging in the backtest engine."
        )

        early_decisions = [v for v in timing_violations if v['type'] == 'early_decision']
        if early_decisions:
            recommendations.append(
                f"📅 Remove {len(early_decisions)} early decisions made before sufficient lag period."
            )

    if leakage_sources:
        recommendations.append(
            "🔍 DATA LEAKAGE: Features may contain future information. Review feature calculation "
            "to ensure they only use historical data available at decision time."
        )

        # Specific recommendations for common leakage sources
        for source in leakage_sources[:5]:  # Limit to first 5 for brevity
            if 'NaN' in source:
                recommendations.append(f"📊 Fix NaN values in features: {source[:80]}...")
            elif 'future' in source.lower() or 'ahead' in source.lower():
                recommendations.append(f"🔮 Remove forward-looking feature: {source[:80]}...")

    if position_shift_violations:
        recommendations.append(
            "🔄 POSITION SHIFT: Ensure positions are properly lagged relative to states. "
            "Position at time t should be based on state at time t-1 or earlier."
        )

        mismatch_count = len([v for v in position_shift_violations if v['type'] == 'position_shift_violation'])
        if mismatch_count > 0:
            recommendations.append(
                f"📍 Fix {mismatch_count} position shift violations where actual and expected positions differ."
            )

    if not timing_violations and not leakage_sources and not position_shift_violations:
        recommendations.append(
            "✅ No lookahead bias detected. The backtest appears to have proper bias prevention measures."
        )

    # General recommendations
    recommendations.extend([
        "📈 Consider implementing automated bias detection in your backtesting pipeline.",
        "🔬 Use walk-forward analysis to validate strategy robustness over time.",
        "⏰ Document all lag assumptions and data availability constraints."
    ])

    return recommendations


def apply_bias_prevention(
    states: np.ndarray,
    positions: pd.Series,
    lag_periods: int = 1,
    fill_value: int = 0
) -> Tuple[np.ndarray, pd.Series]:
    """
    Apply bias prevention measures to state and position sequences.

    Args:
        states: Original state sequence
        positions: Original position series
        lag_periods: Number of periods to lag
        fill_value: Value to use for periods where lag cannot be applied

    Returns:
        Tuple of (lagged_states, lagged_positions)
    """
    logger.info(f"Applying bias prevention with {lag_periods}-period lag")

    # Apply lag to states
    lagged_states = np.full_like(states, fill_value, dtype=states.dtype)
    if lag_periods < len(states):
        lagged_states[lag_periods:] = states[:-lag_periods]

    # Apply lag to positions
    lagged_positions = positions.shift(lag_periods).fillna(fill_value)

    logger.debug(f"Applied lag: first {lag_periods} periods filled with {fill_value}")
    return lagged_states, lagged_positions


def create_bias_prevention_report(result: BiasDetectionResult) -> str:
    """
    Create a human-readable bias prevention report.

    Args:
        result: Bias detection results

    Returns:
        Formatted report string
    """
    report = []
    report.append("=== Lookahead Bias Detection Report ===")
    report.append(f"Overall Risk Score: {result.overall_risk_score:.3f}")
    report.append(f"Lookahead Bias Detected: {'Yes' if result.has_lookahead_bias else 'No'}")
    report.append(f"Data Leakage Detected: {'Yes' if result.data_leakage_detected else 'No'}")
    report.append("")

    # Timing violations
    report.append(f"Timing Violations: {len(result.timing_violations)}")
    if result.timing_violations:
        report.append("  Sample violations:")
        for violation in result.timing_violations[:3]:
            report.append(f"    - {violation.get('message', str(violation))}")

    # Data leakage
    report.append(f"Data Leakage Sources: {len(result.leakage_sources)}")
    if result.leakage_sources:
        report.append("  Sample leakage sources:")
        for source in result.leakage_sources[:3]:
            report.append(f"    - {source}")

    # Position shift violations
    report.append(f"Position Shift Violations: {len(result.position_shift_violations)}")
    if result.position_shift_violations:
        report.append("  Sample violations:")
        for violation in result.position_shift_violations[:3]:
            report.append(f"    - {violation.get('message', str(violation))}")

    # Recommendations
    report.append("")
    report.append("=== Recommendations ===")
    for i, recommendation in enumerate(result.recommendations, 1):
        report.append(f"{i}. {recommendation}")

    return "\n".join(report)


def validate_backtest_realism(
    result: BacktestResult,
    max_lookahead_risk: float = 0.1
) -> Dict[str, Any]:
    """
    Validate the overall realism of backtest results.

    Args:
        result: Backtest results to validate
        max_lookahead_risk: Maximum acceptable risk score for lookahead bias

    Returns:
        Dictionary with validation results
    """
    logger.info("Starting backtest realism validation")

    validation_results = {
        'is_realistic': True,
        'warnings': [],
        'errors': [],
        'risk_scores': {}
    }

    # Check for unrealistic returns
    returns = result.equity_curve.pct_change().dropna()
    if len(returns) > 0:
        daily_mean = returns.mean()
        daily_std = returns.std()

        # Flag unusually high daily returns
        if abs(daily_mean) > 0.05:  # > 5% daily return
            validation_results['warnings'].append(
                f"Unusually high daily return: {daily_mean:.2%} - consider checking for lookahead bias"
            )

        # Flag unusually low volatility
        if daily_std < 0.001:  # < 0.1% daily volatility
            validation_results['warnings'].append(
                f"Unusually low volatility: {daily_std:.2%} - may indicate insufficient market conditions"
            )

    # Check trade frequency
    if result.trades:
        trade_frequency = len(result.trades) / len(result.equity_curve)
        if trade_frequency > 0.5:  # Trading more than 50% of the time
            validation_results['warnings'].append(
                f"High trade frequency: {trade_frequency:.2%} - consider transaction costs impact"
            )

        # Check for consistent positive returns
        positive_trades = [t for t in result.trades if t.pnl and t.pnl > 0]
        if len(positive_trades) / len(result.trades) > 0.8:  # > 80% win rate
            validation_results['warnings'].append(
                f"High win rate: {len(positive_trades)/len(result.trades):.2%} - verify bias prevention"
            )

    # Check position persistence
    position_changes = (result.positions.diff() != 0).sum()
    if position_changes == 0:
        validation_results['warnings'].append("No position changes detected - static position strategy")

    validation_results['risk_scores']['trade_frequency'] = trade_frequency if result.trades else 0
    validation_results['risk_scores']['win_rate'] = len(positive_trades) / len(result.trades) if result.trades else 0

    # Overall assessment
    if validation_results['warnings']:
        validation_results['is_realistic'] = False

    logger.info(f"Realism validation completed: {'Realistic' if validation_results['is_realistic'] else 'Issues detected'}")
    return validation_results
