"""
Chain-of-Thought Reasoning Engine

This module provides advanced reasoning capabilities that structure AI thinking
for better problem-solving and more reliable responses.
"""

import json
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from global_mcp_client.core.logger import LoggerMixin


class ReasoningType(Enum):
    """Types of reasoning approaches"""
    MATHEMATICAL = "mathematical"
    LOGICAL = "logical"
    ANALYTICAL = "analytical"
    DATABASE = "database"
    SCHEMA_DISCOVERY = "schema_discovery"
    PROBLEM_SOLVING = "problem_solving"
    MIXED = "mixed"


@dataclass
class ReasoningStep:
    """Individual step in reasoning process"""
    step_number: int
    step_type: str
    description: str
    input_data: Any
    reasoning: str
    output_data: Any
    confidence: float = 1.0
    verification_notes: Optional[str] = None


@dataclass
class ReasoningResult:
    """Complete reasoning result"""
    reasoning_type: ReasoningType
    steps: List[ReasoningStep]
    final_conclusion: str
    confidence_score: float
    enhanced_query: str
    verification_checks: List[str]
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReasoningStrategy(ABC):
    """Abstract base class for reasoning strategies"""

    @abstractmethod
    def get_reasoning_template(self) -> str:
        """Get the reasoning template for this strategy"""
        pass

    @abstractmethod
    def enhance_query(self, query: str, context: Dict[str, Any]) -> str:
        """Enhance the query with reasoning prompts"""
        pass

    @abstractmethod
    def extract_reasoning_steps(self, response: str) -> List[ReasoningStep]:
        """Extract reasoning steps from AI response"""
        pass


class DatabaseReasoningStrategy(ReasoningStrategy):
    """Reasoning strategy for database analysis and queries"""

    def get_reasoning_template(self) -> str:
        return """
DATABASE ANALYSIS REASONING PROTOCOL:

1. REQUIREMENT UNDERSTANDING:
   - What specific information is being requested?
   - What type of analysis or data is needed?
   - Are there any constraints or specific conditions?

2. SCHEMA EXPLORATION PLANNING:
   - What schemas and tables might contain the relevant data?
   - What relationships between tables should be considered?
   - What are the key columns and data types involved?

3. QUERY STRATEGY DESIGN:
   - What sequence of queries will efficiently gather the needed information?
   - How should the data be aggregated, filtered, or joined?
   - What potential issues or edge cases should be considered?

4. EXECUTION APPROACH:
   - What tools should be used for each step?
   - How should errors and unexpected results be handled?
   - What verification steps ensure data accuracy?

5. RESULT INTERPRETATION:
   - How should the results be analyzed and presented?
   - What insights can be derived from the data?
   - What follow-up questions or analyses might be valuable?
"""

    def enhance_query(self, query: str, context: Dict[str, Any]) -> str:
        enhanced = f"""
{query}

{self.get_reasoning_template()}

CONTEXT ANALYSIS:
- Available tools: {', '.join([tool.get('name', '') for tool in context.get('available_tools', [])])}
- Previous context: {context.get('context_summary', 'No prior context')}

Please work through this database analysis systematically, showing your reasoning for each step:
"""
        return enhanced

    def extract_reasoning_steps(self, response: str) -> List[ReasoningStep]:
        """Extract database reasoning steps from response"""
        steps = []

        # Look for numbered reasoning patterns
        step_patterns = [
            r'(\d+)\.\s*REQUIREMENT UNDERSTANDING:(.*?)(?=\d+\.\s*\w+:|$)',
            r'(\d+)\.\s*SCHEMA EXPLORATION:(.*?)(?=\d+\.\s*\w+:|$)',
            r'(\d+)\.\s*QUERY STRATEGY:(.*?)(?=\d+\.\s*\w+:|$)',
            r'(\d+)\.\s*EXECUTION:(.*?)(?=\d+\.\s*\w+:|$)',
            r'(\d+)\.\s*RESULT INTERPRETATION:(.*?)(?=\d+\.\s*\w+:|$)'
        ]

        step_types = ["understanding", "exploration", "strategy", "execution", "interpretation"]

        for i, pattern in enumerate(step_patterns):
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            for match in matches:
                step_num = int(match[0]) if match[0].isdigit() else i + 1
                reasoning_text = match[1].strip()

                steps.append(ReasoningStep(
                    step_number=step_num,
                    step_type=step_types[i] if i < len(step_types) else "general",
                    description=f"Database {step_types[i] if i < len(step_types) else 'reasoning'}",
                    input_data=query,
                    reasoning=reasoning_text,
                    output_data=None,
                    confidence=0.8
                ))

        return steps


class SchemaDiscoveryStrategy(ReasoningStrategy):
    """Reasoning strategy for schema discovery and understanding"""

    def get_reasoning_template(self) -> str:
        return """
SCHEMA DISCOVERY REASONING PROTOCOL:

1. DISCOVERY PLANNING:
   - What level of schema information is needed (high-level overview vs. detailed structure)?
   - Which schemas or databases should be explored?
   - What are the priorities for analysis?

2. SYSTEMATIC EXPLORATION:
   - Start with schema/database listing to understand scope
   - Identify key tables based on naming patterns and metadata
   - Analyze table structures and relationships systematically

3. PATTERN RECOGNITION:
   - What business domains do the table names suggest?
   - What data relationships can be inferred from column names and foreign keys?
   - What are the likely fact tables, dimension tables, and lookup tables?

4. BUSINESS LOGIC INFERENCE:
   - What business processes do these tables support?
   - What are the key business entities and their relationships?
   - What metrics and KPIs can be derived from this structure?

5. ANALYSIS RECOMMENDATIONS:
   - What are the most important tables for analysis?
   - What types of questions can this schema answer?
   - What additional exploration would be valuable?
"""

    def enhance_query(self, query: str, context: Dict[str, Any]) -> str:
        enhanced = f"""
{query}

{self.get_reasoning_template()}

DISCOVERY CONTEXT:
- Target schema: {context.get('schema_name', 'Unknown')}
- Analysis depth: {context.get('analysis_depth', 'Standard')}

Please approach this schema discovery systematically, documenting your reasoning:
"""
        return enhanced

    def extract_reasoning_steps(self, response: str) -> List[ReasoningStep]:
        """Extract schema discovery reasoning steps"""
        steps = []

        # Pattern matching for discovery steps
        discovery_patterns = [
            (r'DISCOVERY PLANNING:(.*?)(?=SYSTEMATIC EXPLORATION:|$)', "planning"),
            (r'SYSTEMATIC EXPLORATION:(.*?)(?=PATTERN RECOGNITION:|$)', "exploration"),
            (r'PATTERN RECOGNITION:(.*?)(?=BUSINESS LOGIC:|$)', "pattern_recognition"),
            (r'BUSINESS LOGIC INFERENCE:(.*?)(?=ANALYSIS RECOMMENDATIONS:|$)', "business_logic"),
            (r'ANALYSIS RECOMMENDATIONS:(.*?)$', "recommendations")
        ]

        for i, (pattern, step_type) in enumerate(discovery_patterns):
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            for match in matches:
                reasoning_text = match.strip() if isinstance(match, str) else str(match)

                steps.append(ReasoningStep(
                    step_number=i + 1,
                    step_type=step_type,
                    description=f"Schema {step_type.replace('_', ' ')}",
                    input_data=None,
                    reasoning=reasoning_text,
                    output_data=None,
                    confidence=0.85
                ))

        return steps


class AnalyticalReasoningStrategy(ReasoningStrategy):
    """Reasoning strategy for data analysis and insights"""

    def get_reasoning_template(self) -> str:
        return """
ANALYTICAL REASONING PROTOCOL:

1. PROBLEM DECOMPOSITION:
   - Break down the analytical question into component parts
   - Identify what data and calculations are needed
   - Determine the analytical approach and methodology

2. DATA ASSESSMENT:
   - What data sources are available and relevant?
   - What is the quality and completeness of the data?
   - What are the limitations and assumptions?

3. ANALYTICAL APPROACH:
   - What analytical techniques are most appropriate?
   - What metrics and KPIs should be calculated?
   - How should the data be segmented or grouped?

4. INSIGHT GENERATION:
   - What patterns and trends are evident in the data?
   - What are the key findings and their implications?
   - What hypotheses can be formed and tested?

5. VALIDATION AND VERIFICATION:
   - Do the results make business sense?
   - Are there any anomalies or outliers to investigate?
   - What additional analysis would strengthen the conclusions?
"""

    def enhance_query(self, query: str, context: Dict[str, Any]) -> str:
        enhanced = f"""
{query}

{self.get_reasoning_template()}

ANALYTICAL CONTEXT:
- Available data sources: {context.get('data_sources', 'To be determined')}
- Analysis type: {context.get('analysis_type', 'Exploratory')}
- Business context: {context.get('business_context', 'General analysis')}

Please approach this analysis systematically, showing your analytical reasoning:
"""
        return enhanced

    def extract_reasoning_steps(self, response: str) -> List[ReasoningStep]:
        """Extract analytical reasoning steps"""
        steps = []

        analytical_patterns = [
            (r'PROBLEM DECOMPOSITION:(.*?)(?=DATA ASSESSMENT:|$)', "decomposition"),
            (r'DATA ASSESSMENT:(.*?)(?=ANALYTICAL APPROACH:|$)', "assessment"),
            (r'ANALYTICAL APPROACH:(.*?)(?=INSIGHT GENERATION:|$)', "approach"),
            (r'INSIGHT GENERATION:(.*?)(?=VALIDATION:|$)', "insights"),
            (r'VALIDATION AND VERIFICATION:(.*?)$', "validation")
        ]

        for i, (pattern, step_type) in enumerate(analytical_patterns):
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            for match in matches:
                reasoning_text = match.strip() if isinstance(match, str) else str(match)

                steps.append(ReasoningStep(
                    step_number=i + 1,
                    step_type=step_type,
                    description=f"Analytical {step_type}",
                    input_data=None,
                    reasoning=reasoning_text,
                    output_data=None,
                    confidence=0.9
                ))

        return steps


class ProblemSolvingStrategy(ReasoningStrategy):
    """General problem-solving reasoning strategy"""

    def get_reasoning_template(self) -> str:
        return """
SYSTEMATIC PROBLEM-SOLVING PROTOCOL:

1. PROBLEM IDENTIFICATION:
   - What exactly is the problem or challenge?
   - What are the symptoms and underlying causes?
   - What are the constraints and requirements?

2. INFORMATION GATHERING:
   - What information is needed to solve this problem?
   - What tools and resources are available?
   - What expertise or domain knowledge is required?

3. SOLUTION DEVELOPMENT:
   - What are the possible approaches or solutions?
   - What are the pros and cons of each approach?
   - Which approach is most likely to succeed?

4. IMPLEMENTATION PLANNING:
   - What are the specific steps to implement the solution?
   - What tools and resources will be used?
   - How will progress and success be measured?

5. VERIFICATION AND REFINEMENT:
   - How can the solution be tested and validated?
   - What adjustments or improvements are needed?
   - What lessons learned can be applied to future problems?
"""

    def enhance_query(self, query: str, context: Dict[str, Any]) -> str:
        enhanced = f"""
{query}

{self.get_reasoning_template()}

PROBLEM CONTEXT:
- Domain: {context.get('domain', 'General')}
- Complexity: {context.get('complexity', 'Medium')}
- Available resources: {context.get('resources', 'Standard toolset')}

Please work through this problem systematically, documenting your reasoning process:
"""
        return enhanced

    def extract_reasoning_steps(self, response: str) -> List[ReasoningStep]:
        """Extract problem-solving reasoning steps"""
        steps = []

        problem_patterns = [
            (r'PROBLEM IDENTIFICATION:(.*?)(?=INFORMATION GATHERING:|$)', "identification"),
            (r'INFORMATION GATHERING:(.*?)(?=SOLUTION DEVELOPMENT:|$)', "information"),
            (r'SOLUTION DEVELOPMENT:(.*?)(?=IMPLEMENTATION:|$)', "solution"),
            (r'IMPLEMENTATION PLANNING:(.*?)(?=VERIFICATION:|$)', "implementation"),
            (r'VERIFICATION AND REFINEMENT:(.*?)$', "verification")
        ]

        for i, (pattern, step_type) in enumerate(problem_patterns):
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            for match in matches:
                reasoning_text = match.strip() if isinstance(match, str) else str(match)

                steps.append(ReasoningStep(
                    step_number=i + 1,
                    step_type=step_type,
                    description=f"Problem-solving {step_type}",
                    input_data=None,
                    reasoning=reasoning_text,
                    output_data=None,
                    confidence=0.85
                ))

        return steps


class ChainOfThoughtEngine(LoggerMixin):
    """
    Advanced Chain-of-Thought reasoning engine that enhances AI queries
    with structured thinking and reasoning processes.
    """

    def __init__(self, config):
        """
        Initialize the Chain-of-Thought engine

        Args:
            config: Configuration object with reasoning settings
        """
        self.config = config
        self.strategies = {
            ReasoningType.DATABASE: DatabaseReasoningStrategy(),
            ReasoningType.SCHEMA_DISCOVERY: SchemaDiscoveryStrategy(),
            ReasoningType.ANALYTICAL: AnalyticalReasoningStrategy(),
            ReasoningType.PROBLEM_SOLVING: ProblemSolvingStrategy()
        }

        self.reasoning_history = []
        self.performance_metrics = {
            "total_reasonings": 0,
            "avg_confidence": 0.0,
            "strategy_usage": {strategy.value: 0 for strategy in ReasoningType}
        }

        self.logger.info("ChainOfThoughtEngine initialized", extra={
            "available_strategies": list(self.strategies.keys()),
            "reasoning_enabled": config.enable_chain_of_thought
        })

    def enhance_query_with_reasoning(self, query: str, context: Dict[str, Any]) -> Tuple[str, ReasoningResult]:
        """
        Enhance a query with chain-of-thought reasoning

        Args:
            query: Original user query
            context: Context information including tools, history, etc.

        Returns:
            Tuple of (enhanced_query, reasoning_result)
        """
        if not self.config.enable_chain_of_thought:
            # Return original query if CoT is disabled
            return query, self._create_simple_reasoning_result(query)

        start_time = time.time()

        # Classify the query to select appropriate reasoning strategy
        reasoning_type = self._classify_query_reasoning_type(query, context)
        strategy = self.strategies.get(reasoning_type, self.strategies[ReasoningType.PROBLEM_SOLVING])

        # Enhance the query with reasoning prompts
        enhanced_query = strategy.enhance_query(query, context)

        # Create reasoning result metadata
        reasoning_result = ReasoningResult(
            reasoning_type=reasoning_type,
            steps=[],  # Will be populated when processing AI response
            final_conclusion="",
            confidence_score=0.8,
            enhanced_query=enhanced_query,
            verification_checks=[],
            execution_time=time.time() - start_time,
            metadata={
                "original_query": query,
                "strategy_used": reasoning_type.value,
                "context_size": len(str(context))
            }
        )

        # Update metrics
        self.performance_metrics["total_reasonings"] += 1
        self.performance_metrics["strategy_usage"][reasoning_type.value] += 1

        self.logger.info("Query enhanced with reasoning", extra={
            "reasoning_type": reasoning_type.value,
            "enhancement_time": reasoning_result.execution_time,
            "original_length": len(query),
            "enhanced_length": len(enhanced_query)
        })

        return enhanced_query, reasoning_result

    def extract_reasoning_from_response(self, response: str, reasoning_result: ReasoningResult) -> ReasoningResult:
        """
        Extract reasoning steps from AI response and update reasoning result

        Args:
            response: AI response text
            reasoning_result: Reasoning result to update

        Returns:
            Updated reasoning result with extracted steps
        """
        strategy = self.strategies.get(reasoning_result.reasoning_type,
                                     self.strategies[ReasoningType.PROBLEM_SOLVING])

        # Extract reasoning steps using the appropriate strategy
        reasoning_steps = strategy.extract_reasoning_steps(response)

        # Calculate overall confidence
        if reasoning_steps:
            avg_confidence = sum(step.confidence for step in reasoning_steps) / len(reasoning_steps)
        else:
            avg_confidence = 0.5

        # Generate verification checks
        verification_checks = self._generate_verification_checks(reasoning_steps, response)

        # Extract final conclusion
        final_conclusion = self._extract_final_conclusion(response)

        # Update reasoning result
        reasoning_result.steps = reasoning_steps
        reasoning_result.confidence_score = avg_confidence
        reasoning_result.final_conclusion = final_conclusion
        reasoning_result.verification_checks = verification_checks

        # Store in history
        self.reasoning_history.append(reasoning_result)

        # Update performance metrics
        self.performance_metrics["avg_confidence"] = (
            (self.performance_metrics["avg_confidence"] * (self.performance_metrics["total_reasonings"] - 1) +
             avg_confidence) / self.performance_metrics["total_reasonings"]
        )

        self.logger.debug("Reasoning extracted from response", extra={
            "steps_found": len(reasoning_steps),
            "confidence": avg_confidence,
            "verification_checks": len(verification_checks)
        })

        return reasoning_result

    def _classify_query_reasoning_type(self, query: str, context: Dict[str, Any]) -> ReasoningType:
        """Classify the query to determine appropriate reasoning type"""
        query_lower = query.lower()

        # Database-specific patterns
        if any(term in query_lower for term in ["database", "query", "sql", "table", "records", "data analysis"]):
            if any(term in query_lower for term in ["schema", "structure", "tables", "discover", "explore"]):
                return ReasoningType.SCHEMA_DISCOVERY
            else:
                return ReasoningType.DATABASE

        # Analytical patterns
        if any(term in query_lower for term in ["analyze", "analysis", "metrics", "dashboard", "trends", "insights"]):
            return ReasoningType.ANALYTICAL

        # Problem-solving patterns
        if any(term in query_lower for term in ["how", "why", "solve", "fix", "troubleshoot", "optimize", "improve"]):
            return ReasoningType.PROBLEM_SOLVING

        # Check context for additional clues
        if context.get("query_intent", {}).get("primary_intent") == "data_analysis":
            return ReasoningType.ANALYTICAL
        elif context.get("query_intent", {}).get("primary_intent") == "schema_exploration":
            return ReasoningType.SCHEMA_DISCOVERY

        # Default to problem-solving for general queries
        return ReasoningType.PROBLEM_SOLVING

    def _create_simple_reasoning_result(self, query: str) -> ReasoningResult:
        """Create a simple reasoning result when CoT is disabled"""
        return ReasoningResult(
            reasoning_type=ReasoningType.MIXED,
            steps=[],
            final_conclusion=query,
            confidence_score=1.0,
            enhanced_query=query,
            verification_checks=[],
            execution_time=0.0,
            metadata={"cot_disabled": True}
        )

    def _generate_verification_checks(self, steps: List[ReasoningStep], response: str) -> List[str]:
        """Generate verification checks for the reasoning process"""
        checks = []

        if steps:
            checks.append(f"Reasoning process completed with {len(steps)} documented steps")

            # Check for logical flow
            if len(steps) >= 3:
                checks.append("Multi-step reasoning approach used")

            # Check confidence levels
            low_confidence_steps = [step for step in steps if step.confidence < 0.7]
            if low_confidence_steps:
                checks.append(f"Warning: {len(low_confidence_steps)} steps have low confidence")
            else:
                checks.append("All reasoning steps show high confidence")

        # Check for specific reasoning patterns in response
        if re.search(r'\b(because|therefore|thus|hence|consequently)\b', response, re.IGNORECASE):
            checks.append("Causal reasoning patterns detected")

        if re.search(r'\b(first|second|third|then|next|finally)\b', response, re.IGNORECASE):
            checks.append("Sequential reasoning structure present")

        return checks

    def _extract_final_conclusion(self, response: str) -> str:
        """Extract the final conclusion from the AI response"""
        # Look for conclusion patterns
        conclusion_patterns = [
            r'(?:conclusion|summary|final result|in summary):\s*(.*?)(?:\n\n|\n$|$)',
            r'(?:therefore|thus|hence),?\s*(.*?)(?:\n\n|\n$|$)',
            r'(?:to answer your question|in conclusion),?\s*(.*?)(?:\n\n|\n$|$)'
        ]

        for pattern in conclusion_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                conclusion = match.group(1).strip()
                if conclusion and len(conclusion) > 10:  # Ensure it's substantial
                    return conclusion[:500]  # Limit length

        # If no specific conclusion pattern found, take the last paragraph
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        if paragraphs:
            return paragraphs[-1][:500]

        return "No clear conclusion extracted"

    def get_reasoning_performance(self) -> Dict[str, Any]:
        """Get performance metrics for the reasoning engine"""
        recent_reasonings = self.reasoning_history[-10:]  # Last 10 reasonings

        performance = {
            "total_reasonings": self.performance_metrics["total_reasonings"],
            "average_confidence": self.performance_metrics["avg_confidence"],
            "strategy_usage": self.performance_metrics["strategy_usage"].copy(),
            "recent_performance": {
                "count": len(recent_reasonings),
                "avg_steps": sum(len(r.steps) for r in recent_reasonings) / max(len(recent_reasonings), 1),
                "avg_confidence": sum(r.confidence_score for r in recent_reasonings) / max(len(recent_reasonings), 1),
                "avg_execution_time": sum(r.execution_time for r in recent_reasonings) / max(len(recent_reasonings), 1)
            }
        }

        return performance

    def clear_history(self) -> None:
        """Clear reasoning history"""
        self.reasoning_history.clear()
        self.logger.info("Reasoning history cleared")