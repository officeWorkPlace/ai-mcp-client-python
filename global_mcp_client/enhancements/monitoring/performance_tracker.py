"""
Performance Tracking System for Enhanced AI Capabilities

This module provides comprehensive performance monitoring and analysis
for all enhancement components including reasoning, context optimization,
and response quality.
"""

import json
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import threading

from global_mcp_client.core.logger import LoggerMixin


class MetricType(Enum):
    """Types of performance metrics"""
    TIMING = "timing"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    ACCURACY = "accuracy"
    USAGE = "usage"
    ERROR = "error"


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    component: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentPerformance:
    """Performance data for a specific component"""
    component_name: str
    total_operations: int
    success_rate: float
    average_execution_time: float
    average_quality_score: float
    error_rate: float
    efficiency_score: float
    recent_metrics: List[PerformanceMetric] = field(default_factory=list)


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    report_timestamp: float
    time_period: str
    overall_performance: Dict[str, float]
    component_performances: Dict[str, ComponentPerformance]
    trends: Dict[str, List[float]]
    recommendations: List[str]
    alerts: List[str]


class PerformanceAnalyzer:
    """Analyzes performance data and generates insights"""

    def __init__(self):
        self.trend_window = 50  # Number of data points for trend analysis

    def analyze_component_performance(self, metrics: List[PerformanceMetric]) -> ComponentPerformance:
        """Analyze performance for a specific component"""
        if not metrics:
            return ComponentPerformance(
                component_name="unknown",
                total_operations=0,
                success_rate=0.0,
                average_execution_time=0.0,
                average_quality_score=0.0,
                error_rate=0.0,
                efficiency_score=0.0
            )

        component_name = metrics[0].component
        total_operations = len(metrics)

        # Calculate timing metrics
        timing_metrics = [m for m in metrics if m.metric_type == MetricType.TIMING]
        avg_execution_time = statistics.mean([m.value for m in timing_metrics]) if timing_metrics else 0.0

        # Calculate quality metrics
        quality_metrics = [m for m in metrics if m.metric_type == MetricType.QUALITY]
        avg_quality_score = statistics.mean([m.value for m in quality_metrics]) if quality_metrics else 0.0

        # Calculate error rate
        error_metrics = [m for m in metrics if m.metric_type == MetricType.ERROR]
        error_rate = len(error_metrics) / max(total_operations, 1)

        # Calculate success rate (inverse of error rate)
        success_rate = 1.0 - error_rate

        # Calculate efficiency score (quality per unit time)
        if avg_execution_time > 0 and avg_quality_score > 0:
            efficiency_score = avg_quality_score / avg_execution_time
        else:
            efficiency_score = 0.0

        return ComponentPerformance(
            component_name=component_name,
            total_operations=total_operations,
            success_rate=success_rate,
            average_execution_time=avg_execution_time,
            average_quality_score=avg_quality_score,
            error_rate=error_rate,
            efficiency_score=efficiency_score,
            recent_metrics=metrics[-10:]  # Keep last 10 metrics
        )

    def calculate_trends(self, metrics: List[PerformanceMetric]) -> Dict[str, List[float]]:
        """Calculate performance trends over time"""
        trends = {}

        # Group metrics by type
        metric_groups = defaultdict(list)
        for metric in metrics[-self.trend_window:]:  # Last N metrics
            metric_groups[metric.metric_type].append(metric.value)

        # Calculate trend for each metric type
        for metric_type, values in metric_groups.items():
            if len(values) >= 3:  # Need at least 3 points for trend
                trends[metric_type.value] = values

        return trends

    def generate_recommendations(self, component_performances: Dict[str, ComponentPerformance]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []

        for component_name, performance in component_performances.items():
            # High error rate recommendation
            if performance.error_rate > 0.1:  # >10% error rate
                recommendations.append(f"{component_name}: High error rate ({performance.error_rate:.1%}) - investigate error causes")

            # Slow execution recommendation
            if performance.average_execution_time > 2.0:  # >2 seconds
                recommendations.append(f"{component_name}: Slow execution time ({performance.average_execution_time:.2f}s) - consider optimization")

            # Low quality recommendation
            if performance.average_quality_score < 6.0 and performance.average_quality_score > 0:  # <6/10 quality
                recommendations.append(f"{component_name}: Low quality scores ({performance.average_quality_score:.1f}/10) - review quality enhancement")

            # Low efficiency recommendation
            if performance.efficiency_score < 1.0 and performance.efficiency_score > 0:
                recommendations.append(f"{component_name}: Low efficiency score - balance quality vs speed")

        # Overall system recommendations
        avg_efficiency = statistics.mean([p.efficiency_score for p in component_performances.values() if p.efficiency_score > 0])
        if avg_efficiency < 2.0:
            recommendations.append("System: Overall efficiency is low - consider configuration tuning")

        return recommendations

    def generate_alerts(self, component_performances: Dict[str, ComponentPerformance]) -> List[str]:
        """Generate performance alerts for critical issues"""
        alerts = []

        for component_name, performance in component_performances.items():
            # Critical error rate
            if performance.error_rate > 0.5:  # >50% error rate
                alerts.append(f"CRITICAL: {component_name} has very high error rate ({performance.error_rate:.1%})")

            # System unresponsive
            if performance.average_execution_time > 10.0:  # >10 seconds
                alerts.append(f"WARNING: {component_name} is very slow ({performance.average_execution_time:.2f}s)")

            # No operations
            if performance.total_operations == 0:
                alerts.append(f"INFO: {component_name} has no recorded operations")

        return alerts


class PerformanceTracker(LoggerMixin):
    """
    Comprehensive performance tracking system for enhanced AI capabilities
    """

    def __init__(self, config):
        """
        Initialize the performance tracker

        Args:
            config: Configuration object with performance tracking settings
        """
        self.config = config
        self.analyzer = PerformanceAnalyzer()

        # Thread-safe metric storage
        self._lock = threading.Lock()
        self.metrics = deque(maxlen=config.performance_history_limit)
        self.component_metrics = defaultdict(lambda: deque(maxlen=1000))

        # Performance summaries
        self.session_start_time = time.time()
        self.last_report_time = time.time()

        self.logger.info("PerformanceTracker initialized", extra={
            "history_limit": config.performance_history_limit,
            "tracking_enabled": config.enable_performance_tracking
        })

    def track_operation(self, component: str, operation: str, execution_time: float,
                       quality_score: Optional[float] = None, success: bool = True,
                       context: Optional[Dict[str, Any]] = None) -> None:
        """
        Track a single operation performance

        Args:
            component: Component name (e.g., 'context_manager', 'cot_engine')
            operation: Operation name (e.g., 'optimize_context', 'enhance_query')
            execution_time: Time taken to execute in seconds
            quality_score: Quality score (0-10) if applicable
            success: Whether the operation succeeded
            context: Additional context information
        """
        if not self.config.enable_performance_tracking:
            return

        timestamp = time.time()
        context = context or {}

        with self._lock:
            # Track timing metric
            timing_metric = PerformanceMetric(
                name=f"{operation}_time",
                value=execution_time,
                metric_type=MetricType.TIMING,
                timestamp=timestamp,
                component=component,
                context=context
            )
            self.metrics.append(timing_metric)
            self.component_metrics[component].append(timing_metric)

            # Track quality metric if provided
            if quality_score is not None:
                quality_metric = PerformanceMetric(
                    name=f"{operation}_quality",
                    value=quality_score,
                    metric_type=MetricType.QUALITY,
                    timestamp=timestamp,
                    component=component,
                    context=context
                )
                self.metrics.append(quality_metric)
                self.component_metrics[component].append(quality_metric)

            # Track error if operation failed
            if not success:
                error_metric = PerformanceMetric(
                    name=f"{operation}_error",
                    value=1.0,
                    metric_type=MetricType.ERROR,
                    timestamp=timestamp,
                    component=component,
                    context=context
                )
                self.metrics.append(error_metric)
                self.component_metrics[component].append(error_metric)

        self.logger.debug("Operation tracked", extra={
            "component": component,
            "operation": operation,
            "execution_time": execution_time,
            "quality_score": quality_score,
            "success": success
        })

    def track_context_optimization(self, optimization_result, execution_time: float) -> None:
        """Track context optimization performance"""
        quality_score = None
        if hasattr(optimization_result, 'utilization_stats'):
            # Use compression ratio as a quality indicator
            compression_ratio = optimization_result.utilization_stats.get('compression_ratio', 0)
            quality_score = min(10.0, compression_ratio * 10)  # Convert to 0-10 scale

        context = {
            "message_count": len(optimization_result.optimized_messages) if hasattr(optimization_result, 'optimized_messages') else 0,
            "compression_ratio": optimization_result.utilization_stats.get('compression_ratio', 0) if hasattr(optimization_result, 'utilization_stats') else 0
        }

        self.track_operation(
            component="context_manager",
            operation="optimize_context",
            execution_time=execution_time,
            quality_score=quality_score,
            success=True,
            context=context
        )

    def track_reasoning_enhancement(self, reasoning_result, execution_time: float) -> None:
        """Track chain-of-thought reasoning performance"""
        quality_score = reasoning_result.confidence_score * 10 if hasattr(reasoning_result, 'confidence_score') else None

        context = {
            "reasoning_type": reasoning_result.reasoning_type.value if hasattr(reasoning_result, 'reasoning_type') else "unknown",
            "steps_count": len(reasoning_result.steps) if hasattr(reasoning_result, 'steps') else 0
        }

        self.track_operation(
            component="cot_engine",
            operation="enhance_reasoning",
            execution_time=execution_time,
            quality_score=quality_score,
            success=True,
            context=context
        )

    def track_quality_optimization(self, enhancement_result, execution_time: float) -> None:
        """Track response quality optimization performance"""
        quality_improvement = 0.0
        if hasattr(enhancement_result, 'after_assessment') and hasattr(enhancement_result, 'before_assessment'):
            quality_improvement = enhancement_result.after_assessment.overall_score - enhancement_result.before_assessment.overall_score

        quality_score = enhancement_result.after_assessment.overall_score if hasattr(enhancement_result, 'after_assessment') else None

        context = {
            "quality_improvement": quality_improvement,
            "improvements_made": len(enhancement_result.quality_improvements) if hasattr(enhancement_result, 'quality_improvements') else 0
        }

        self.track_operation(
            component="quality_optimizer",
            operation="optimize_response",
            execution_time=execution_time,
            quality_score=quality_score,
            success=True,
            context=context
        )

    def track_tool_usage(self, tool_name: str, execution_time: float, success: bool, result_size: int = 0) -> None:
        """Track tool usage performance"""
        # Use result size as a rough quality indicator
        quality_score = min(10.0, result_size / 100) if result_size > 0 else None

        context = {
            "result_size": result_size,
            "tool_name": tool_name
        }

        self.track_operation(
            component="tool_execution",
            operation=f"execute_{tool_name}",
            execution_time=execution_time,
            quality_score=quality_score,
            success=success,
            context=context
        )

    def generate_performance_report(self, time_period: str = "session") -> PerformanceReport:
        """
        Generate comprehensive performance report

        Args:
            time_period: Time period for the report ("session", "hour", "day")

        Returns:
            PerformanceReport with detailed analysis
        """
        with self._lock:
            # Filter metrics based on time period
            now = time.time()
            if time_period == "hour":
                cutoff_time = now - 3600
            elif time_period == "day":
                cutoff_time = now - 86400
            else:  # session
                cutoff_time = self.session_start_time

            recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]

            # Analyze performance by component
            component_performances = {}
            for component_name in self.component_metrics.keys():
                component_metrics = [m for m in recent_metrics if m.component == component_name]
                if component_metrics:
                    component_performances[component_name] = self.analyzer.analyze_component_performance(component_metrics)

            # Calculate overall performance metrics
            overall_performance = self._calculate_overall_performance(component_performances)

            # Calculate trends
            trends = self.analyzer.calculate_trends(recent_metrics)

            # Generate recommendations and alerts
            recommendations = self.analyzer.generate_recommendations(component_performances)
            alerts = self.analyzer.generate_alerts(component_performances)

            report = PerformanceReport(
                report_timestamp=now,
                time_period=time_period,
                overall_performance=overall_performance,
                component_performances=component_performances,
                trends=trends,
                recommendations=recommendations,
                alerts=alerts
            )

            self.last_report_time = now

            self.logger.info("Performance report generated", extra={
                "time_period": time_period,
                "components_analyzed": len(component_performances),
                "metrics_analyzed": len(recent_metrics),
                "alerts_count": len(alerts),
                "recommendations_count": len(recommendations)
            })

            return report

    def _calculate_overall_performance(self, component_performances: Dict[str, ComponentPerformance]) -> Dict[str, float]:
        """Calculate overall system performance metrics"""
        if not component_performances:
            return {}

        performances = list(component_performances.values())

        # Calculate weighted averages
        total_operations = sum(p.total_operations for p in performances)

        if total_operations == 0:
            return {}

        overall_performance = {
            "total_operations": total_operations,
            "overall_success_rate": sum(p.success_rate * p.total_operations for p in performances) / total_operations,
            "overall_avg_execution_time": sum(p.average_execution_time * p.total_operations for p in performances) / total_operations,
            "overall_error_rate": sum(p.error_rate * p.total_operations for p in performances) / total_operations,
            "session_duration": time.time() - self.session_start_time,
            "active_components": len(component_performances)
        }

        # Calculate overall quality score (only for components with quality metrics)
        quality_components = [p for p in performances if p.average_quality_score > 0]
        if quality_components:
            overall_performance["overall_avg_quality_score"] = sum(
                p.average_quality_score * p.total_operations for p in quality_components
            ) / sum(p.total_operations for p in quality_components)

        return overall_performance

    def get_component_summary(self, component_name: str) -> Optional[ComponentPerformance]:
        """Get performance summary for a specific component"""
        with self._lock:
            if component_name not in self.component_metrics:
                return None

            component_metrics = list(self.component_metrics[component_name])
            return self.analyzer.analyze_component_performance(component_metrics)

    def get_recent_metrics(self, component: Optional[str] = None, limit: int = 100) -> List[PerformanceMetric]:
        """Get recent metrics, optionally filtered by component"""
        with self._lock:
            if component:
                return list(self.component_metrics[component])[-limit:]
            else:
                return list(self.metrics)[-limit:]

    def export_performance_data(self, file_path: str) -> None:
        """Export performance data to JSON file"""
        with self._lock:
            data = {
                "session_start_time": self.session_start_time,
                "export_timestamp": time.time(),
                "metrics": [
                    {
                        "name": m.name,
                        "value": m.value,
                        "metric_type": m.metric_type.value,
                        "timestamp": m.timestamp,
                        "component": m.component,
                        "context": m.context
                    }
                    for m in self.metrics
                ]
            }

            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

            self.logger.info("Performance data exported", extra={
                "file_path": file_path,
                "metrics_count": len(data["metrics"])
            })

    def clear_metrics(self) -> None:
        """Clear all stored metrics"""
        with self._lock:
            self.metrics.clear()
            self.component_metrics.clear()
            self.session_start_time = time.time()

        self.logger.info("Performance metrics cleared")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a quick performance summary"""
        report = self.generate_performance_report("session")

        summary = {
            "session_duration": report.overall_performance.get("session_duration", 0),
            "total_operations": report.overall_performance.get("total_operations", 0),
            "success_rate": report.overall_performance.get("overall_success_rate", 0),
            "avg_execution_time": report.overall_performance.get("overall_avg_execution_time", 0),
            "active_components": report.overall_performance.get("active_components", 0),
            "alerts_count": len(report.alerts),
            "recommendations_count": len(report.recommendations)
        }

        if "overall_avg_quality_score" in report.overall_performance:
            summary["avg_quality_score"] = report.overall_performance["overall_avg_quality_score"]

        return summary