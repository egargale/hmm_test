"""
Pipeline metrics and monitoring system for the unified data pipeline.

This module provides comprehensive metrics collection, performance monitoring,
and reporting capabilities for the unified data pipeline.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class StageMetrics:
    """Metrics for individual pipeline stage execution."""
    stage_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    processing_time: float = 0.0
    success: bool = True
    input_rows: int = 0
    output_rows: int = 0
    input_columns: int = 0
    output_columns: int = 0
    memory_usage_mb: float = 0.0
    issues: List[str] = field(default_factory=list)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.issues:
            self.issues = []

    def finalize(self, success: bool, end_time: Optional[datetime] = None) -> None:
        """Finalize stage metrics."""
        self.success = success
        if end_time:
            self.end_time = end_time
            self.processing_time = (end_time - self.start_time).total_seconds()


@dataclass
class PipelineMetrics:
    """Comprehensive metrics for entire pipeline execution."""
    pipeline_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_processing_time: float = 0.0
    success: bool = True
    stage_metrics: List[StageMetrics] = field(default_factory=list)
    input_metadata: Dict[str, Any] = field(default_factory=dict)
    output_metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def add_stage_metrics(self, stage_metrics: StageMetrics) -> None:
        """Add stage metrics to pipeline metrics."""
        self.stage_metrics.append(stage_metrics)

    def finalize(self, success: bool, end_time: Optional[datetime] = None) -> None:
        """Finalize pipeline metrics."""
        self.success = success
        if end_time:
            self.end_time = end_time
            self.total_processing_time = (end_time - self.start_time).total_seconds()

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        successful_stages = sum(1 for sm in self.stage_metrics if sm.success)
        total_stages = len(self.stage_metrics)

        return {
            'pipeline_name': self.pipeline_name,
            'success': self.success,
            'total_processing_time': self.total_processing_time,
            'successful_stages': successful_stages,
            'total_stages': total_stages,
            'success_rate': successful_stages / total_stages if total_stages > 0 else 0,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'average_stage_time': sum(sm.processing_time for sm in self.stage_metrics) / len(self.stage_metrics) if self.stage_metrics else 0,
            'total_issues': sum(len(sm.issues) for sm in self.stage_metrics)
        }


class MetricsCollector:
    """Collects and manages pipeline metrics."""

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.metrics = PipelineMetrics(
            pipeline_name=pipeline_name,
            start_time=datetime.now()
        )
        self.current_stage: Optional[StageMetrics] = None
        self.logger = get_logger(f"{__name__}.MetricsCollector")

    def start_stage(self, stage_name: str, input_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Start collecting metrics for a stage."""
        self.current_stage = StageMetrics(
            stage_name=stage_name,
            start_time=datetime.now()
        )

        if input_metadata:
            self.current_stage.input_rows = input_metadata.get('rows', 0)
            self.current_stage.input_columns = input_metadata.get('columns', 0)

        self.logger.debug(f"Started metrics collection for stage: {stage_name}")

    def end_stage(self, success: bool = True, output_metadata: Optional[Dict[str, Any]] = None,
                  issues: Optional[List[str]] = None, custom_metrics: Optional[Dict[str, Any]] = None) -> None:
        """End metrics collection for current stage."""
        if self.current_stage is None:
            self.logger.warning("No stage currently being tracked")
            return

        self.current_stage.finalize(success)

        if output_metadata:
            self.current_stage.output_rows = output_metadata.get('rows', 0)
            self.current_stage.output_columns = output_metadata.get('columns', 0)
            self.current_stage.memory_usage_mb = output_metadata.get('memory_usage_mb', 0)

        if issues:
            self.current_stage.issues.extend(issues)

        if custom_metrics:
            self.current_stage.custom_metrics.update(custom_metrics)

        self.metrics.add_stage_metrics(self.current_stage)
        self.logger.debug(f"Ended metrics collection for stage: {self.current_stage.stage_name}")
        self.current_stage = None

    def finalize_pipeline(self, success: bool = True,
                         input_metadata: Optional[Dict[str, Any]] = None,
                         output_metadata: Optional[Dict[str, Any]] = None,
                         quality_metrics: Optional[Dict[str, Any]] = None) -> PipelineMetrics:
        """Finalize pipeline metrics collection."""
        self.metrics.finalize(success)

        if input_metadata:
            self.metrics.input_metadata.update(input_metadata)

        if output_metadata:
            self.metrics.output_metadata.update(output_metadata)

        if quality_metrics:
            self.metrics.quality_metrics.update(quality_metrics)

        # Calculate performance metrics
        self.metrics.performance_metrics = self._calculate_performance_metrics()

        # Calculate resource usage
        self.metrics.resource_usage = self._calculate_resource_usage()

        self.logger.info(f"Pipeline metrics finalized: {self.metrics.get_summary()}")
        return self.metrics

    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if not self.metrics.stage_metrics:
            return {}

        stage_times = [sm.processing_time for sm in self.metrics.stage_metrics]
        successful_stages = [sm for sm in self.metrics.stage_metrics if sm.success]

        return {
            'total_stage_time': sum(stage_times),
            'average_stage_time': sum(stage_times) / len(stage_times),
            'fastest_stage_time': min(stage_times),
            'slowest_stage_time': max(stage_times),
            'successful_stages': len(successful_stages),
            'stage_efficiency': len(successful_stages) / len(self.metrics.stage_metrics),
            'bottleneck_stage': max(self.metrics.stage_metrics, key=lambda sm: sm.processing_time).stage_name if self.metrics.stage_metrics else None
        }

    def _calculate_resource_usage(self) -> Dict[str, Any]:
        """Calculate resource usage metrics."""
        memory_usage = [sm.memory_usage_mb for sm in self.metrics.stage_metrics if sm.memory_usage_mb > 0]

        if not memory_usage:
            return {}

        return {
            'peak_memory_mb': max(memory_usage),
            'average_memory_mb': sum(memory_usage) / len(memory_usage),
            'total_memory_mb': sum(memory_usage),
            'memory_efficiency': max(memory_usage) / sum(memory_usage) if sum(memory_usage) > 0 else 0
        }

    def get_current_metrics(self) -> PipelineMetrics:
        """Get current metrics (including ongoing stage if any)."""
        return self.metrics

    def add_custom_metric(self, key: str, value: Any) -> None:
        """Add custom metric to current stage or pipeline."""
        if self.current_stage:
            self.current_stage.custom_metrics[key] = value
        else:
            self.metrics.custom_metrics[key] = value


class PerformanceProfiler:
    """Advanced performance profiling for pipeline optimization."""

    def __init__(self):
        self.profiles: Dict[str, List[Dict[str, Any]]] = {}
        self.logger = get_logger(f"{__name__}.PerformanceProfiler")

    def start_profiling(self, profile_name: str) -> Dict[str, Any]:
        """Start profiling session."""
        profile = {
            'name': profile_name,
            'start_time': time.time(),
            'start_memory': self._get_memory_usage(),
            'cpu_usage_start': self._get_cpu_usage()
        }

        if profile_name not in self.profiles:
            self.profiles[profile_name] = []

        self.profiles[profile_name].append(profile)
        return profile

    def end_profiling(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """End profiling session and return results."""
        end_time = time.time()
        end_memory = self._get_memory_usage()
        cpu_usage_end = self._get_cpu_usage()

        profile.update({
            'end_time': end_time,
            'end_memory': end_memory,
            'cpu_usage_end': cpu_usage_end,
            'duration': end_time - profile['start_time'],
            'memory_delta': end_memory - profile['start_memory'],
            'cpu_usage_avg': (profile['cpu_usage_start'] + cpu_usage_end) / 2
        })

        return profile

    def get_profile_summary(self, profile_name: str) -> Dict[str, Any]:
        """Get summary of profiling sessions."""
        if profile_name not in self.profiles or not self.profiles[profile_name]:
            return {}

        sessions = self.profiles[profile_name]

        return {
            'profile_name': profile_name,
            'total_sessions': len(sessions),
            'total_duration': sum(s['duration'] for s in sessions),
            'average_duration': sum(s['duration'] for s in sessions) / len(sessions),
            'fastest_session': min(sessions, key=lambda s: s['duration'])['duration'],
            'slowest_session': max(sessions, key=lambda s: s['duration'])['duration'],
            'total_memory_delta': sum(s['memory_delta'] for s in sessions),
            'average_cpu_usage': sum(s['cpu_usage_avg'] for s in sessions) / len(sessions)
        }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0


class MetricsReporter:
    """Generate reports from pipeline metrics."""

    def __init__(self):
        self.logger = get_logger(f"{__name__}.MetricsReporter")

    def generate_performance_report(self, metrics: PipelineMetrics) -> Dict[str, Any]:
        """Generate detailed performance report."""
        report = {
            'report_type': 'performance',
            'generated_at': datetime.now().isoformat(),
            'pipeline_summary': metrics.get_summary(),
            'stage_details': [],
            'performance_analysis': {},
            'recommendations': []
        }

        # Add stage details
        for stage_metric in metrics.stage_metrics:
            stage_detail = {
                'stage_name': stage_metric.stage_name,
                'processing_time': stage_metric.processing_time,
                'success': stage_metric.success,
                'input_shape': (stage_metric.input_rows, stage_metric.input_columns),
                'output_shape': (stage_metric.output_rows, stage_metric.output_columns),
                'memory_usage_mb': stage_metric.memory_usage_mb,
                'issues_count': len(stage_metric.issues),
                'custom_metrics': stage_metric.custom_metrics
            }
            report['stage_details'].append(stage_detail)

        # Performance analysis
        if metrics.stage_metrics:
            stage_times = [sm.processing_time for sm in metrics.stage_metrics]
            report['performance_analysis'] = {
                'total_processing_time': sum(stage_times),
                'average_stage_time': sum(stage_times) / len(stage_times),
                'processing_time_distribution': {
                    'min': min(stage_times),
                    'max': max(stage_times),
                    'median': sorted(stage_times)[len(stage_times) // 2],
                    'std_dev': self._calculate_std_dev(stage_times)
                },
                'bottleneck_stage': max(metrics.stage_metrics, key=lambda sm: sm.processing_time).stage_name,
                'efficiency_score': self._calculate_efficiency_score(metrics)
            }

        # Generate recommendations
        report['recommendations'] = self._generate_performance_recommendations(metrics)

        return report

    def generate_quality_report(self, metrics: PipelineMetrics) -> Dict[str, Any]:
        """Generate detailed quality report."""
        report = {
            'report_type': 'quality',
            'generated_at': datetime.now().isoformat(),
            'overall_quality_score': metrics.quality_metrics.get('overall_score', 0),
            'quality_breakdown': {},
            'issues_summary': {},
            'improvement_suggestions': []
        }

        # Quality breakdown
        if metrics.quality_metrics:
            report['quality_breakdown'] = {
                'completeness_score': metrics.quality_metrics.get('completeness_score', 0),
                'validation_score': metrics.quality_metrics.get('validation_score', 0),
                'feature_quality_score': metrics.quality_metrics.get('feature_quality_score', 0),
                'ohlcv_coverage': metrics.quality_metrics.get('ohlcv_coverage', 0)
            }

        # Issues summary
        total_issues = sum(len(sm.issues) for sm in metrics.stage_metrics)
        report['issues_summary'] = {
            'total_issues': total_issues,
            'stages_with_issues': len([sm for sm in metrics.stage_metrics if sm.issues]),
            'issue_categories': self._categorize_issues(metrics)
        }

        # Improvement suggestions
        report['improvement_suggestions'] = self._generate_quality_recommendations(metrics)

        return report

    def generate_summary_report(self, metrics: PipelineMetrics) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        return {
            'report_type': 'summary',
            'generated_at': datetime.now().isoformat(),
            'pipeline_info': {
                'name': metrics.pipeline_name,
                'success': metrics.success,
                'execution_time': metrics.total_processing_time,
                'stages_executed': len(metrics.stage_metrics),
                'stages_successful': sum(1 for sm in metrics.stage_metrics if sm.success)
            },
            'data_info': {
                'input_shape': (metrics.input_metadata.get('rows', 0), metrics.input_metadata.get('columns', 0)),
                'output_shape': (metrics.output_metadata.get('rows', 0), metrics.output_metadata.get('columns', 0)),
                'features_added': metrics.output_metadata.get('columns', 0) - metrics.input_metadata.get('columns', 0)
            },
            'performance_highlights': {
                'total_processing_time': metrics.total_processing_time,
                'average_stage_time': sum(sm.processing_time for sm in metrics.stage_metrics) / len(metrics.stage_metrics) if metrics.stage_metrics else 0,
                'peak_memory_mb': metrics.resource_usage.get('peak_memory_mb', 0)
            },
            'quality_highlights': {
                'overall_score': metrics.quality_metrics.get('overall_score', 0),
                'total_issues': sum(len(sm.issues) for sm in metrics.stage_metrics)
            }
        }

    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def _calculate_efficiency_score(self, metrics: PipelineMetrics) -> float:
        """Calculate overall efficiency score (0-1)."""
        if not metrics.stage_metrics:
            return 0.0

        # Factors: success rate, time efficiency, memory efficiency
        success_rate = sum(1 for sm in metrics.stage_metrics if sm.success) / len(metrics.stage_metrics)

        # Time efficiency (inverse of total time)
        time_efficiency = 1.0 / (1.0 + metrics.total_processing_time / 60)  # Normalize to minutes

        # Memory efficiency
        memory_usage = [sm.memory_usage_mb for sm in metrics.stage_metrics if sm.memory_usage_mb > 0]
        memory_efficiency = 1.0 / (1.0 + (max(memory_usage) / 1000)) if memory_usage else 1.0  # Normalize to GB

        # Weighted average
        return (success_rate * 0.5 + time_efficiency * 0.3 + memory_efficiency * 0.2)

    def _generate_performance_recommendations(self, metrics: PipelineMetrics) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []

        if not metrics.stage_metrics:
            return recommendations

        # Check for slow stages
        stage_times = [(sm.stage_name, sm.processing_time) for sm in metrics.stage_metrics]
        avg_time = sum(time for _, time in stage_times) / len(stage_times)

        for stage_name, time_taken in stage_times:
            if time_taken > avg_time * 2:
                recommendations.append(f"Consider optimizing '{stage_name}' stage ({time_taken:.2f}s vs avg {avg_time:.2f}s)")

        # Check memory usage
        memory_usage = [sm.memory_usage_mb for sm in metrics.stage_metrics if sm.memory_usage_mb > 0]
        if memory_usage and max(memory_usage) > 1000:  # > 1GB
            recommendations.append("Consider memory optimization for large datasets")

        # Check for failed stages
        failed_stages = [sm.stage_name for sm in metrics.stage_metrics if not sm.success]
        if failed_stages:
            recommendations.append(f"Address issues in failed stages: {', '.join(failed_stages)}")

        # Check for processing time
        if metrics.total_processing_time > 60:  # > 1 minute
            recommendations.append("Consider enabling parallel processing for large datasets")

        return recommendations

    def _generate_quality_recommendations(self, metrics: PipelineMetrics) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []

        # Quality score recommendations
        overall_score = metrics.quality_metrics.get('overall_score', 0)
        if overall_score < 0.7:
            recommendations.append("Overall data quality is below 70% - review validation issues")

        # Completeness recommendations
        completeness = metrics.quality_metrics.get('completeness_score', 1)
        if completeness < 0.9:
            recommendations.append("Consider handling missing data to improve completeness")

        # Feature quality recommendations
        feature_quality = metrics.quality_metrics.get('feature_quality_score', 1)
        if feature_quality < 0.8:
            recommendations.append("Consider feature selection to improve feature quality")

        # Issues recommendations
        total_issues = sum(len(sm.issues) for sm in metrics.stage_metrics)
        if total_issues > 10:
            recommendations.append("High number of issues detected - review data preprocessing")

        return recommendations

    def _categorize_issues(self, metrics: PipelineMetrics) -> Dict[str, int]:
        """Categorize issues by type."""
        categories = {
            'validation': 0,
            'processing': 0,
            'memory': 0,
            'other': 0
        }

        for stage_metric in metrics.stage_metrics:
            for issue in stage_metric.issues:
                issue_lower = issue.lower()
                if any(keyword in issue_lower for keyword in ['validation', 'invalid', 'missing']):
                    categories['validation'] += 1
                elif any(keyword in issue_lower for keyword in ['processing', 'failed', 'error']):
                    categories['processing'] += 1
                elif any(keyword in issue_lower for keyword in ['memory', 'ram']):
                    categories['memory'] += 1
                else:
                    categories['other'] += 1

        return categories
