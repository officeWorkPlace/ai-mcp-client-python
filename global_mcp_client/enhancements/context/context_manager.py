"""
Intelligent Context Manager for Enhanced AI Understanding

This module provides advanced context management capabilities that optimize
conversation history and tool results for maximum AI understanding.
"""

import json
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import structlog

from global_mcp_client.core.logger import LoggerMixin


@dataclass
class MessageRelevance:
    """Represents the relevance score of a message to current query"""
    message_id: str
    content: str
    timestamp: float
    relevance_score: float
    semantic_similarity: float
    temporal_relevance: float
    tool_usage_relevance: float
    concept_overlap: float


@dataclass
class ContextOptimization:
    """Result of context optimization"""
    optimized_messages: List[Dict[str, Any]]
    priority_tools: List[Dict[str, Any]]
    context_summary: str
    utilization_stats: Dict[str, Any]
    optimization_rationale: List[str]


@dataclass
class ConversationMemory:
    """Enhanced conversation memory with semantic understanding"""
    messages: deque = field(default_factory=lambda: deque(maxlen=100))
    tool_results: Dict[str, Any] = field(default_factory=dict)
    successful_patterns: List[Dict[str, Any]] = field(default_factory=list)
    failed_patterns: List[Dict[str, Any]] = field(default_factory=list)
    domain_context: Dict[str, Any] = field(default_factory=dict)


class IntelligentContextManager(LoggerMixin):
    """
    Advanced context management system that optimizes conversation history
    and tool results for maximum AI understanding and performance.
    """

    def __init__(self, config):
        """
        Initialize the intelligent context manager

        Args:
            config: Configuration object with context management settings
        """
        self.config = config
        self.memory = ConversationMemory()
        self.performance_metrics = defaultdict(list)

        # Context optimization settings
        self.max_context_messages = config.max_context_messages
        self.semantic_threshold = config.semantic_similarity_threshold
        self.optimization_level = config.context_optimization_level

        self.logger.info("IntelligentContextManager initialized", extra={
            "max_messages": self.max_context_messages,
            "semantic_threshold": self.semantic_threshold,
            "optimization_level": self.optimization_level
        })

    def add_message(self, role: str, content: str, tool_calls: Optional[List] = None,
                   tool_results: Optional[List] = None) -> None:
        """
        Add a message to conversation memory with enhanced metadata

        Args:
            role: Message role (user, assistant, tool)
            content: Message content
            tool_calls: Tool calls made in this message
            tool_results: Tool results from this message
        """
        message = {
            "id": f"msg_{int(time.time() * 1000)}",
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "tool_calls": tool_calls or [],
            "tool_results": tool_results or [],
            "concepts": self._extract_concepts(content),
            "complexity_score": self._calculate_message_complexity(content)
        }

        self.memory.messages.append(message)

        # Store tool results for future reference
        if tool_results:
            for result in tool_results:
                tool_name = result.get("tool_name", "unknown")
                self.memory.tool_results[f"{tool_name}_{message['id']}"] = result

        self.logger.debug("Message added to context", extra={
            "message_id": message["id"],
            "role": role,
            "concepts": len(message["concepts"]),
            "has_tools": bool(tool_calls or tool_results)
        })

    def optimize_context_for_query(self, query: str, available_tools: List[Dict]) -> ContextOptimization:
        """
        Optimize conversation context for a specific query

        Args:
            query: Current user query
            available_tools: List of available tools

        Returns:
            ContextOptimization with optimized context elements
        """
        start_time = time.time()

        # Extract query concepts and intent
        query_concepts = self._extract_concepts(query)
        query_intent = self._analyze_query_intent(query)

        # Score all messages for relevance
        message_scores = []
        for message in self.memory.messages:
            score = self._calculate_message_relevance(message, query, query_concepts)
            if score.relevance_score > 0.1:  # Only include relevant messages
                message_scores.append(score)

        # Sort by relevance and select top messages
        message_scores.sort(key=lambda x: x.relevance_score, reverse=True)
        optimized_messages = []

        # Always include the most recent message
        if self.memory.messages:
            latest_message = list(self.memory.messages)[-1]
            optimized_messages.append(latest_message)

        # Add most relevant historical messages
        seen_ids = {optimized_messages[0]["id"]} if optimized_messages else set()
        for score in message_scores[:self.max_context_messages-1]:
            if score.message_id not in seen_ids:
                # Find the actual message
                for msg in self.memory.messages:
                    if msg["id"] == score.message_id:
                        optimized_messages.append(msg)
                        seen_ids.add(score.message_id)
                        break

        # Prioritize tools based on query intent and past success
        priority_tools = self._prioritize_tools(available_tools, query_intent, query_concepts)

        # Generate context summary
        context_summary = self._generate_context_summary(optimized_messages, query_intent)

        # Calculate utilization statistics
        optimization_time = time.time() - start_time
        utilization_stats = {
            "original_message_count": len(self.memory.messages),
            "optimized_message_count": len(optimized_messages),
            "compression_ratio": len(optimized_messages) / max(len(self.memory.messages), 1),
            "optimization_time": optimization_time,
            "query_concepts": len(query_concepts),
            "avg_message_relevance": sum(s.relevance_score for s in message_scores) / max(len(message_scores), 1)
        }

        # Generate optimization rationale
        rationale = self._generate_optimization_rationale(message_scores, priority_tools, utilization_stats)

        optimization = ContextOptimization(
            optimized_messages=optimized_messages,
            priority_tools=priority_tools,
            context_summary=context_summary,
            utilization_stats=utilization_stats,
            optimization_rationale=rationale
        )

        # Track performance
        self.performance_metrics["optimization_time"].append(optimization_time)
        self.performance_metrics["compression_ratio"].append(utilization_stats["compression_ratio"])

        self.logger.info("Context optimized for query", extra=utilization_stats)

        return optimization

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text using pattern matching"""
        concepts = []

        # Database-related concepts
        db_patterns = [
            r'\b(table|schema|database|query|SQL|Oracle|MySQL|PostgreSQL)\b',
            r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b',
            r'\b(\w+_table|\w+_schema|\w+_db)\b',
            r'\b(analysis|dashboard|metrics|performance|trends)\b'
        ]

        for pattern in db_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.extend([match.lower() if isinstance(match, str) else match[0].lower() for match in matches])

        # Business concepts
        business_patterns = [
            r'\b(customer|client|user|employee|staff)\b',
            r'\b(revenue|profit|sales|transaction|payment)\b',
            r'\b(product|service|inventory|catalog)\b',
            r'\b(branch|office|location|department)\b'
        ]

        for pattern in business_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.extend([match.lower() for match in matches])

        return list(set(concepts))  # Remove duplicates

    def _calculate_message_complexity(self, content: str) -> float:
        """Calculate message complexity score"""
        if not content:
            return 0.0

        # Basic complexity factors
        word_count = len(content.split())
        sentence_count = len(re.split(r'[.!?]+', content))

        # Technical complexity indicators
        technical_terms = len(re.findall(r'\b(SQL|database|schema|table|query|analysis)\b', content, re.IGNORECASE))
        code_blocks = len(re.findall(r'```|\`\`\`', content))

        # Normalize to 0-10 scale
        complexity = min(10.0, (word_count / 50) + (technical_terms * 0.5) + (code_blocks * 2))

        return complexity

    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent and classification"""
        query_lower = query.lower()

        intent = {
            "primary_intent": "unknown",
            "confidence": 0.0,
            "secondary_intents": [],
            "requires_tools": False,
            "complexity_level": "medium"
        }

        # Intent classification patterns
        intent_patterns = {
            "data_analysis": [
                r'\b(analyze|analysis|examine|study|investigate)\b',
                r'\b(dashboard|metrics|performance|trends|statistics)\b',
                r'\b(summary|overview|report|insights)\b'
            ],
            "schema_exploration": [
                r'\b(schema|structure|tables|columns|database)\b',
                r'\b(explore|discover|understand|show|list)\b',
                r'\b(what.*tables|describe.*schema)\b'
            ],
            "data_retrieval": [
                r'\b(get|fetch|retrieve|find|search|query)\b',
                r'\b(data|records|information|details)\b',
                r'\b(select|show|display|return)\b'
            ],
            "problem_solving": [
                r'\b(how|why|solve|fix|troubleshoot|debug)\b',
                r'\b(problem|issue|error|challenge)\b',
                r'\b(optimize|improve|enhance)\b'
            ]
        }

        max_score = 0
        for intent_type, patterns in intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches

            if score > max_score:
                max_score = score
                intent["primary_intent"] = intent_type
                intent["confidence"] = min(1.0, score * 0.3)

        # Determine if tools are likely needed
        tool_indicators = [
            r'\b(database|table|schema|query|SQL)\b',
            r'\b(analyze|dashboard|metrics|data)\b',
            r'\b(show|display|get|find|search)\b'
        ]

        tool_score = sum(len(re.findall(pattern, query_lower)) for pattern in tool_indicators)
        intent["requires_tools"] = tool_score > 0

        # Complexity assessment
        if len(query.split()) > 20 or tool_score > 3:
            intent["complexity_level"] = "high"
        elif len(query.split()) < 5 or tool_score == 0:
            intent["complexity_level"] = "low"

        return intent

    def _calculate_message_relevance(self, message: Dict, query: str, query_concepts: List[str]) -> MessageRelevance:
        """Calculate comprehensive relevance score for a message"""

        # Semantic similarity (concept overlap)
        message_concepts = message.get("concepts", [])
        common_concepts = set(query_concepts) & set(message_concepts)
        semantic_similarity = len(common_concepts) / max(len(query_concepts), 1) if query_concepts else 0

        # Temporal relevance (more recent = higher score)
        current_time = time.time()
        message_age = current_time - message["timestamp"]
        temporal_relevance = max(0, 1 - (message_age / (24 * 3600)))  # Decay over 24 hours

        # Tool usage relevance
        tool_usage_relevance = 0
        if message.get("tool_calls") or message.get("tool_results"):
            # Higher relevance if message involved tool usage
            tool_usage_relevance = 0.5

            # Extra relevance if tools match query intent
            query_lower = query.lower()
            if any(tool_name in query_lower for tool_result in message.get("tool_results", [])
                   for tool_name in [tool_result.get("tool_name", "")]):
                tool_usage_relevance = 1.0

        # Content overlap (simple text matching)
        query_words = set(query.lower().split())
        message_words = set(message["content"].lower().split())
        word_overlap = len(query_words & message_words) / max(len(query_words), 1)

        # Combined relevance score (weighted average)
        weights = {
            "semantic": 0.4,
            "temporal": 0.2,
            "tool_usage": 0.3,
            "word_overlap": 0.1
        }

        relevance_score = (
            semantic_similarity * weights["semantic"] +
            temporal_relevance * weights["temporal"] +
            tool_usage_relevance * weights["tool_usage"] +
            word_overlap * weights["word_overlap"]
        )

        return MessageRelevance(
            message_id=message["id"],
            content=message["content"][:100] + "..." if len(message["content"]) > 100 else message["content"],
            timestamp=message["timestamp"],
            relevance_score=relevance_score,
            semantic_similarity=semantic_similarity,
            temporal_relevance=temporal_relevance,
            tool_usage_relevance=tool_usage_relevance,
            concept_overlap=len(common_concepts)
        )

    def _prioritize_tools(self, available_tools: List[Dict], query_intent: Dict, query_concepts: List[str]) -> List[Dict]:
        """Prioritize tools based on query intent and past success"""
        tool_scores = []

        for tool in available_tools:
            score = 0
            tool_name = tool.get("name", "").lower()
            tool_desc = tool.get("description", "").lower()

            # Score based on intent matching
            if query_intent["primary_intent"] == "data_analysis":
                if any(term in tool_name or term in tool_desc for term in ["query", "analyze", "dashboard", "metrics"]):
                    score += 3
            elif query_intent["primary_intent"] == "schema_exploration":
                if any(term in tool_name or term in tool_desc for term in ["schema", "table", "structure", "list"]):
                    score += 3
            elif query_intent["primary_intent"] == "data_retrieval":
                if any(term in tool_name or term in tool_desc for term in ["query", "get", "fetch", "records"]):
                    score += 3

            # Score based on concept matching
            concept_matches = sum(1 for concept in query_concepts if concept in tool_name or concept in tool_desc)
            score += concept_matches

            # Check historical success rate
            tool_success_rate = self._get_tool_success_rate(tool_name)
            score += tool_success_rate * 2

            tool_scores.append((tool, score))

        # Sort by score and return top tools
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, score in tool_scores[:20]]  # Top 20 tools

    def _get_tool_success_rate(self, tool_name: str) -> float:
        """Get historical success rate for a tool"""
        # Simple implementation - in a real system, this would track actual success rates
        default_rates = {
            "query_table_records": 0.9,
            "get_all_tables": 0.95,
            "analyze_table_structure": 0.9,
            "execute_complex_query": 0.8
        }
        return default_rates.get(tool_name, 0.7)

    def _generate_context_summary(self, messages: List[Dict], query_intent: Dict) -> str:
        """Generate a concise summary of the conversation context"""
        if not messages:
            return "No prior conversation context."

        # Analyze conversation flow
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
        tool_usage_count = sum(1 for msg in messages if msg.get("tool_calls") or msg.get("tool_results"))

        # Extract dominant concepts
        all_concepts = []
        for msg in messages:
            all_concepts.extend(msg.get("concepts", []))

        from collections import Counter
        top_concepts = [concept for concept, count in Counter(all_concepts).most_common(5)]

        summary = f"""Context Summary:
- Conversation length: {len(user_messages)} user queries, {len(assistant_messages)} responses
- Tool usage: {tool_usage_count} tool interactions
- Key topics: {', '.join(top_concepts[:3]) if top_concepts else 'General discussion'}
- Current intent: {query_intent['primary_intent']} (confidence: {query_intent['confidence']:.1f})
- Requires tools: {'Yes' if query_intent['requires_tools'] else 'No'}"""

        return summary

    def _generate_optimization_rationale(self, message_scores: List[MessageRelevance],
                                       priority_tools: List[Dict], stats: Dict) -> List[str]:
        """Generate rationale for optimization decisions"""
        rationale = []

        if stats["compression_ratio"] < 0.5:
            rationale.append(f"Compressed context by {(1-stats['compression_ratio'])*100:.1f}% to focus on most relevant messages")

        if message_scores:
            avg_relevance = sum(s.relevance_score for s in message_scores) / len(message_scores)
            if avg_relevance > 0.7:
                rationale.append("High relevance conversation history - retained most messages")
            elif avg_relevance < 0.3:
                rationale.append("Low relevance history - prioritized recent messages and tool results")

        if len(priority_tools) < 10:
            rationale.append(f"Identified {len(priority_tools)} highly relevant tools for this query")

        rationale.append(f"Optimization completed in {stats['optimization_time']:.3f}s")

        return rationale

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the context manager"""
        stats = {}

        for metric, values in self.performance_metrics.items():
            if values:
                stats[metric] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "recent": values[-10:]  # Last 10 values
                }

        stats["total_messages"] = len(self.memory.messages)
        stats["total_tool_results"] = len(self.memory.tool_results)

        return stats

    def clear_memory(self) -> None:
        """Clear conversation memory"""
        self.memory = ConversationMemory()
        self.logger.info("Conversation memory cleared")