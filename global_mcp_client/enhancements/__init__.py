"""
Enhanced AI capabilities for Global MCP Client

This package provides advanced AI understanding and reasoning capabilities including:
- Intelligent context management
- Chain-of-thought reasoning
- Response quality optimization
- Metacognitive awareness
- Performance monitoring
"""

from .context.context_manager import IntelligentContextManager
from .reasoning.cot_engine import ChainOfThoughtEngine
from .quality.response_optimizer import ResponseQualityOptimizer
from .metacognition.meta_engine import MetacognitiveEngine
from .monitoring.performance_tracker import PerformanceTracker

__all__ = [
    "IntelligentContextManager",
    "ChainOfThoughtEngine",
    "ResponseQualityOptimizer",
    "MetacognitiveEngine",
    "PerformanceTracker"
]