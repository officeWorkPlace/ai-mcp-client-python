"""
Metacognitive Awareness Engine

This module provides self-reflection, confidence assessment, and uncertainty
quantification capabilities for enhanced AI understanding and reliability.
"""

import json
import time
import re
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from global_mcp_client.core.logger import LoggerMixin


class ConfidenceLevel(Enum):
    """Confidence levels for responses"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class UncertaintyType(Enum):
    """Types of uncertainty in responses"""
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    QUANTITATIVE = "quantitative"
    LINGUISTIC = "linguistic"


@dataclass
class UncertaintyFactor:
    """Represents a specific uncertainty in the response"""
    uncertainty_type: UncertaintyType
    description: str
    confidence_impact: float  # How much this reduces confidence (0-1)
    evidence: List[str] = field(default_factory=list)
    resolution_suggestions: List[str] = field(default_factory=list)


@dataclass
class SelfEvaluation:
    """Self-evaluation of response quality and accuracy"""
    response_completeness: float  # 0-10 scale
    factual_accuracy_confidence: float  # 0-10 scale
    logical_consistency: float  # 0-10 scale
    clarity_assessment: float  # 0-10 scale
    usefulness_score: float  # 0-10 scale
    potential_errors: List[str] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)
    assumptions_made: List[str] = field(default_factory=list)


@dataclass
class MetacognitiveAssessment:
    """Complete metacognitive assessment of a response"""
    overall_confidence: float  # 0-10 scale
    confidence_level: ConfidenceLevel
    uncertainty_factors: List[UncertaintyFactor]
    self_evaluation: SelfEvaluation
    confidence_reasoning: str
    knowledge_boundaries: List[str]
    verification_suggestions: List[str]
    alternative_perspectives: List[str]
    processing_notes: Dict[str, Any] = field(default_factory=dict)


class ConfidenceCalculator:
    """Calculates confidence scores based on various factors"""

    def __init__(self):
        self.confidence_factors = self._initialize_confidence_factors()

    def _initialize_confidence_factors(self) -> Dict[str, Dict[str, float]]:
        """Initialize factors that affect confidence assessment"""
        return {
            "response_indicators": {
                "hedging_language": -0.5,  # "might", "could", "possibly"
                "certainty_language": +0.8,  # "definitely", "certainly", "clearly"
                "uncertainty_markers": -0.7,  # "I'm not sure", "uncertain"
                "qualification_phrases": -0.3,  # "in most cases", "generally"
                "numerical_precision": +0.4,  # Specific numbers vs approximations
                "source_citations": +0.6,  # References to data sources
                "logical_connectors": +0.3,  # "therefore", "because", "thus"
                "contradictory_statements": -1.0,  # Internal contradictions
            },
            "structural_indicators": {
                "step_by_step_reasoning": +0.5,
                "multiple_examples": +0.4,
                "acknowledgment_of_limitations": +0.2,  # Paradoxically increases confidence
                "oversimplification": -0.4,
                "incomplete_explanation": -0.6,
                "clear_organization": +0.3,
            },
            "domain_indicators": {
                "technical_terminology": +0.2,
                "domain_specific_knowledge": +0.5,
                "cross_domain_connections": +0.4,
                "outdated_references": -0.8,
                "speculation_beyond_domain": -0.6,
            },
            "reasoning_indicators": {
                "causal_reasoning": +0.4,
                "analogical_reasoning": +0.3,
                "evidence_based_conclusions": +0.7,
                "unsupported_claims": -0.8,
                "circular_reasoning": -1.0,
                "false_dichotomies": -0.5,
            }
        }

    def calculate_confidence_score(self, response: str, reasoning_trace: Optional[Dict] = None,
                                 tool_results: Optional[List] = None) -> Tuple[float, Dict[str, float]]:
        """
        Calculate overall confidence score based on response analysis

        Args:
            response: The AI response text
            reasoning_trace: Optional reasoning information
            tool_results: Optional tool execution results

        Returns:
            Tuple of (confidence_score, factor_scores)
        """
        factor_scores = {}
        total_confidence_adjustment = 0.0

        # Analyze response text
        for factor_category, factors in self.confidence_factors.items():
            category_score = 0.0
            for factor_name, factor_weight in factors.items():
                factor_presence = self._detect_factor(response, factor_name)
                factor_score = factor_presence * factor_weight
                category_score += factor_score

            factor_scores[factor_category] = category_score
            total_confidence_adjustment += category_score

        # Analyze reasoning trace if available
        if reasoning_trace:
            reasoning_score = self._analyze_reasoning_confidence(reasoning_trace)
            factor_scores["reasoning_quality"] = reasoning_score
            total_confidence_adjustment += reasoning_score

        # Analyze tool results if available
        if tool_results:
            tool_score = self._analyze_tool_confidence(tool_results)
            factor_scores["tool_reliability"] = tool_score
            total_confidence_adjustment += tool_score

        # Base confidence starts at 5.0 (medium), adjusted by factors
        base_confidence = 5.0
        final_confidence = max(0.0, min(10.0, base_confidence + total_confidence_adjustment))

        return final_confidence, factor_scores

    def _detect_factor(self, response: str, factor_name: str) -> float:
        """Detect the presence and strength of a confidence factor in the response"""
        response_lower = response.lower()

        factor_patterns = {
            "hedging_language": [
                r'\b(might|could|possibly|perhaps|maybe|probably|likely|seems?|appears?)\b',
                r'\b(i think|i believe|in my opinion|it appears that)\b'
            ],
            "certainty_language": [
                r'\b(definitely|certainly|clearly|obviously|undoubtedly|without question)\b',
                r'\b(it is certain|it is clear|there is no doubt)\b'
            ],
            "uncertainty_markers": [
                r'\b(not sure|uncertain|unclear|unknown|unsure|ambiguous)\b',
                r'\b(i don\'t know|i\'m not certain|it\'s unclear)\b'
            ],
            "qualification_phrases": [
                r'\b(in most cases|generally|typically|usually|often|sometimes)\b',
                r'\b(under normal circumstances|in many situations)\b'
            ],
            "numerical_precision": [
                r'\b\d+\.\d+\b',  # Decimal numbers
                r'\b\d+%\b',      # Percentages
                r'\b\d{4}-\d{2}-\d{2}\b'  # Dates
            ],
            "source_citations": [
                r'\b(according to|based on|research shows|studies indicate)\b',
                r'\b(data suggests|evidence shows|statistics show)\b'
            ],
            "logical_connectors": [
                r'\b(therefore|thus|hence|consequently|as a result)\b',
                r'\b(because|since|given that|due to|owing to)\b'
            ],
            "contradictory_statements": [
                r'\b(however|but|although|despite|nevertheless|on the other hand)\b'
            ],
            "step_by_step_reasoning": [
                r'\b(first|second|third|next|then|finally|step \d+)\b',
                r'\b(to begin|initially|subsequently|in conclusion)\b'
            ],
            "multiple_examples": [
                r'\b(for example|for instance|such as|including|like)\b',
                r'\b(another example|similarly|likewise|in addition)\b'
            ],
            "acknowledgment_of_limitations": [
                r'\b(limitation|constraint|caveat|important to note)\b',
                r'\b(it should be noted|keep in mind|bear in mind)\b'
            ],
            "technical_terminology": [
                r'\b[A-Z]{2,}\b',  # Acronyms
                r'\b\w+(-\w+)+\b'  # Hyphenated technical terms
            ],
            "evidence_based_conclusions": [
                r'\b(evidence suggests|data shows|research indicates)\b',
                r'\b(based on evidence|supported by data|confirmed by)\b'
            ],
            "unsupported_claims": [
                r'\b(it is obvious|everyone knows|it is well known)\b',
                r'\b(without exception|always|never|all|none)\b'
            ]
        }

        if factor_name not in factor_patterns:
            return 0.0

        patterns = factor_patterns[factor_name]
        total_matches = 0

        for pattern in patterns:
            matches = re.findall(pattern, response_lower)
            total_matches += len(matches)

        # Normalize by response length (per 1000 characters)
        response_length = max(len(response), 1000)
        normalized_score = (total_matches * 1000) / response_length

        # Convert to 0-1 scale with diminishing returns
        return min(1.0, normalized_score)

    def _analyze_reasoning_confidence(self, reasoning_trace: Dict) -> float:
        """Analyze confidence based on reasoning quality"""
        if not reasoning_trace:
            return 0.0

        confidence_boost = 0.0

        # Check for structured reasoning
        if hasattr(reasoning_trace, 'steps') and reasoning_trace.steps:
            confidence_boost += 0.5

        # Check reasoning confidence score
        if hasattr(reasoning_trace, 'confidence_score'):
            confidence_boost += (reasoning_trace.confidence_score - 0.5) * 2.0

        # Check for verification
        if hasattr(reasoning_trace, 'verification_checks') and reasoning_trace.verification_checks:
            confidence_boost += 0.3

        return confidence_boost

    def _analyze_tool_confidence(self, tool_results: List) -> float:
        """Analyze confidence based on tool usage and results"""
        if not tool_results:
            return 0.0

        confidence_boost = 0.0

        # Tool usage generally increases confidence
        confidence_boost += min(1.0, len(tool_results) * 0.3)

        # Check for successful tool executions
        successful_tools = len([r for r in tool_results if r.get('success', True)])
        success_rate = successful_tools / max(len(tool_results), 1)
        confidence_boost += success_rate * 0.5

        return confidence_boost


class UncertaintyAnalyzer:
    """Analyzes uncertainty factors in responses"""

    def __init__(self):
        self.uncertainty_patterns = self._initialize_uncertainty_patterns()

    def _initialize_uncertainty_patterns(self) -> Dict[UncertaintyType, List[str]]:
        """Initialize patterns for detecting different types of uncertainty"""
        return {
            UncertaintyType.FACTUAL: [
                r'\b(may not be accurate|might be wrong|uncertain about|not confirmed)\b',
                r'\b(alleged|purported|claimed|reportedly)\b'
            ],
            UncertaintyType.PROCEDURAL: [
                r'\b(exact steps may vary|process might differ|procedure could change)\b',
                r'\b(depending on|varies by|may require different)\b'
            ],
            UncertaintyType.CAUSAL: [
                r'\b(may be caused by|could result from|might lead to)\b',
                r'\b(potential cause|possible reason|contributing factor)\b'
            ],
            UncertaintyType.TEMPORAL: [
                r'\b(timeframe may vary|duration uncertain|timing unclear)\b',
                r'\b(could take|might last|approximately|around)\b'
            ],
            UncertaintyType.QUANTITATIVE: [
                r'\b(approximately|roughly|about|around|nearly)\b',
                r'\b(estimate|ballpark|order of magnitude)\b'
            ],
            UncertaintyType.LINGUISTIC: [
                r'\b(difficult to explain|hard to describe|complex concept)\b',
                r'\b(in other words|that is to say|to put it simply)\b'
            ]
        }

    def identify_uncertainty_factors(self, response: str) -> List[UncertaintyFactor]:
        """Identify and categorize uncertainty factors in the response"""
        uncertainty_factors = []

        for uncertainty_type, patterns in self.uncertainty_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, response, re.IGNORECASE)
                for match in matches:
                    # Extract context around the match
                    start = max(0, match.start() - 50)
                    end = min(len(response), match.end() + 50)
                    context = response[start:end].strip()

                    factor = UncertaintyFactor(
                        uncertainty_type=uncertainty_type,
                        description=f"Uncertainty detected: '{match.group()}'",
                        confidence_impact=self._calculate_uncertainty_impact(uncertainty_type, context),
                        evidence=[context],
                        resolution_suggestions=self._generate_resolution_suggestions(uncertainty_type)
                    )
                    uncertainty_factors.append(factor)

        return uncertainty_factors

    def _calculate_uncertainty_impact(self, uncertainty_type: UncertaintyType, context: str) -> float:
        """Calculate how much this uncertainty factor reduces confidence"""
        base_impacts = {
            UncertaintyType.FACTUAL: 0.8,
            UncertaintyType.PROCEDURAL: 0.4,
            UncertaintyType.CAUSAL: 0.6,
            UncertaintyType.TEMPORAL: 0.3,
            UncertaintyType.QUANTITATIVE: 0.2,
            UncertaintyType.LINGUISTIC: 0.1
        }

        base_impact = base_impacts.get(uncertainty_type, 0.5)

        # Adjust based on context strength
        strong_indicators = ['very', 'extremely', 'highly', 'completely']
        weak_indicators = ['slightly', 'somewhat', 'partially', 'minor']

        context_lower = context.lower()
        if any(indicator in context_lower for indicator in strong_indicators):
            return min(1.0, base_impact * 1.5)
        elif any(indicator in context_lower for indicator in weak_indicators):
            return base_impact * 0.7

        return base_impact

    def _generate_resolution_suggestions(self, uncertainty_type: UncertaintyType) -> List[str]:
        """Generate suggestions for resolving uncertainty"""
        suggestions = {
            UncertaintyType.FACTUAL: [
                "Verify information with authoritative sources",
                "Cross-reference with multiple reliable sources",
                "Check for recent updates or corrections"
            ],
            UncertaintyType.PROCEDURAL: [
                "Consult official documentation or guidelines",
                "Seek expert advice for specific procedures",
                "Test or validate the process in a controlled environment"
            ],
            UncertaintyType.CAUSAL: [
                "Look for additional evidence of causation",
                "Consider alternative explanations",
                "Examine confounding variables"
            ],
            UncertaintyType.TEMPORAL: [
                "Seek more specific timing information",
                "Consider temporal context and variations",
                "Check for historical patterns or trends"
            ],
            UncertaintyType.QUANTITATIVE: [
                "Obtain more precise measurements or data",
                "Use statistical analysis for better estimates",
                "Consider confidence intervals or ranges"
            ],
            UncertaintyType.LINGUISTIC: [
                "Seek clearer definitions or explanations",
                "Use concrete examples or analogies",
                "Consult domain-specific terminology resources"
            ]
        }

        return suggestions.get(uncertainty_type, ["Seek additional information and verification"])


class SelfEvaluationEngine:
    """Performs self-evaluation of response quality"""

    def __init__(self):
        self.evaluation_criteria = self._initialize_evaluation_criteria()

    def _initialize_evaluation_criteria(self) -> Dict[str, Dict[str, Any]]:
        """Initialize criteria for self-evaluation"""
        return {
            "completeness": {
                "positive_indicators": ["comprehensive", "complete", "thorough", "detailed", "extensive"],
                "negative_indicators": ["incomplete", "partial", "brief", "superficial", "limited"],
                "weight": 0.25
            },
            "accuracy": {
                "positive_indicators": ["accurate", "correct", "precise", "exact", "verified"],
                "negative_indicators": ["inaccurate", "incorrect", "wrong", "mistaken", "false"],
                "weight": 0.30
            },
            "consistency": {
                "positive_indicators": ["consistent", "coherent", "logical", "aligned"],
                "negative_indicators": ["inconsistent", "contradictory", "conflicting", "illogical"],
                "weight": 0.20
            },
            "clarity": {
                "positive_indicators": ["clear", "understandable", "straightforward", "explicit"],
                "negative_indicators": ["confusing", "unclear", "ambiguous", "vague"],
                "weight": 0.15
            },
            "usefulness": {
                "positive_indicators": ["useful", "helpful", "practical", "actionable", "valuable"],
                "negative_indicators": ["useless", "unhelpful", "impractical", "irrelevant"],
                "weight": 0.10
            }
        }

    def evaluate_response(self, response: str, query: str) -> SelfEvaluation:
        """Perform comprehensive self-evaluation of the response"""

        # Evaluate each dimension
        completeness = self._evaluate_dimension(response, "completeness", query)
        accuracy = self._evaluate_dimension(response, "accuracy", query)
        consistency = self._evaluate_dimension(response, "consistency", query)
        clarity = self._evaluate_dimension(response, "clarity", query)
        usefulness = self._evaluate_dimension(response, "usefulness", query)

        # Identify potential issues
        potential_errors = self._identify_potential_errors(response)
        knowledge_gaps = self._identify_knowledge_gaps(response)
        assumptions = self._identify_assumptions(response)

        return SelfEvaluation(
            response_completeness=completeness,
            factual_accuracy_confidence=accuracy,
            logical_consistency=consistency,
            clarity_assessment=clarity,
            usefulness_score=usefulness,
            potential_errors=potential_errors,
            knowledge_gaps=knowledge_gaps,
            assumptions_made=assumptions
        )

    def _evaluate_dimension(self, response: str, dimension: str, query: str) -> float:
        """Evaluate a specific quality dimension"""
        criteria = self.evaluation_criteria.get(dimension, {})
        positive_indicators = criteria.get("positive_indicators", [])
        negative_indicators = criteria.get("negative_indicators", [])

        response_lower = response.lower()
        query_lower = query.lower()

        # Count positive and negative indicators
        positive_count = sum(response_lower.count(indicator) for indicator in positive_indicators)
        negative_count = sum(response_lower.count(indicator) for indicator in negative_indicators)

        # Base score calculation
        base_score = 6.0  # Start with above-average assumption

        # Adjust based on indicators
        positive_adjustment = min(2.0, positive_count * 0.5)
        negative_adjustment = min(2.0, negative_count * 0.7)

        # Dimension-specific evaluations
        if dimension == "completeness":
            base_score += self._assess_completeness(response, query)
        elif dimension == "accuracy":
            base_score += self._assess_accuracy(response)
        elif dimension == "consistency":
            base_score += self._assess_consistency(response)
        elif dimension == "clarity":
            base_score += self._assess_clarity(response)
        elif dimension == "usefulness":
            base_score += self._assess_usefulness(response, query)

        final_score = base_score + positive_adjustment - negative_adjustment
        return max(0.0, min(10.0, final_score))

    def _assess_completeness(self, response: str, query: str) -> float:
        """Assess response completeness"""
        # Simple heuristic: longer responses are generally more complete
        response_length = len(response)
        query_length = len(query)

        # Expect roughly 10-50 characters of response per character of query
        expected_length = query_length * 25
        length_ratio = response_length / max(expected_length, 100)

        if length_ratio < 0.5:
            return -1.0  # Too short
        elif length_ratio > 3.0:
            return -0.5  # Possibly too verbose
        else:
            return min(1.0, length_ratio)

    def _assess_accuracy(self, response: str) -> float:
        """Assess response accuracy indicators"""
        # Look for accuracy indicators
        accuracy_boost = 0.0

        # Citations and references
        if re.search(r'\b(according to|research shows|studies indicate|data suggests)\b', response, re.IGNORECASE):
            accuracy_boost += 0.5

        # Specific numbers and facts
        if re.search(r'\b\d+(\.\d+)?(%|kg|m|cm|years?|months?|days?)\b', response):
            accuracy_boost += 0.3

        # Hedging (actually positive for accuracy assessment)
        if re.search(r'\b(approximately|roughly|about|around)\b', response, re.IGNORECASE):
            accuracy_boost += 0.2

        return accuracy_boost

    def _assess_consistency(self, response: str) -> float:
        """Assess logical consistency"""
        # Look for consistency issues
        consistency_penalty = 0.0

        # Contradictory connectors might indicate inconsistency
        contradictions = re.findall(r'\b(however|but|although|despite|nevertheless)\b', response, re.IGNORECASE)
        if len(contradictions) > 3:  # Many contradictions might indicate inconsistency
            consistency_penalty += 0.5

        # Multiple conflicting statements
        conflicting_patterns = [
            (r'\byes\b.*?\bno\b', r'\bno\b.*?\byes\b'),
            (r'\btrue\b.*?\bfalse\b', r'\bfalse\b.*?\btrue\b'),
            (r'\bincorrect\b.*?\bcorrect\b', r'\bcorrect\b.*?\bincorrect\b')
        ]

        for pattern1, pattern2 in conflicting_patterns:
            if re.search(pattern1, response, re.IGNORECASE) or re.search(pattern2, response, re.IGNORECASE):
                consistency_penalty += 0.3

        return -consistency_penalty

    def _assess_clarity(self, response: str) -> float:
        """Assess response clarity"""
        clarity_boost = 0.0

        # Clear structure indicators
        if re.search(r'\b(first|second|third|finally)\b', response, re.IGNORECASE):
            clarity_boost += 0.3

        # Examples and explanations
        if re.search(r'\b(for example|for instance|such as)\b', response, re.IGNORECASE):
            clarity_boost += 0.2

        # Sentence length (shorter is clearer)
        sentences = re.split(r'[.!?]+', response)
        avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / max(len(sentences), 1)

        if avg_sentence_length < 20:
            clarity_boost += 0.3
        elif avg_sentence_length > 30:
            clarity_boost -= 0.3

        return clarity_boost

    def _assess_usefulness(self, response: str, query: str) -> float:
        """Assess response usefulness"""
        usefulness_boost = 0.0

        # Actionable advice
        if re.search(r'\b(should|can|recommend|suggest|try|consider)\b', response, re.IGNORECASE):
            usefulness_boost += 0.4

        # Direct answers to question
        question_words = re.findall(r'\b(what|how|why|when|where|who|which)\b', query, re.IGNORECASE)
        for word in question_words:
            if word.lower() in response.lower():
                usefulness_boost += 0.2

        return usefulness_boost

    def _identify_potential_errors(self, response: str) -> List[str]:
        """Identify potential errors in the response"""
        errors = []

        # Numerical inconsistencies
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
        if len(numbers) > 3:
            # Simple check for obviously wrong calculations
            try:
                nums = [float(n) for n in numbers[:3]]
                if len(nums) >= 3 and abs(nums[0] + nums[1] - nums[2]) > max(nums) * 0.1:
                    errors.append("Potential numerical inconsistency detected")
            except:
                pass

        # Contradictory statements
        if re.search(r'\balways\b.*?\bnever\b|\bnever\b.*?\balways\b', response, re.IGNORECASE):
            errors.append("Contradictory absolute statements detected")

        # Overgeneralization
        if re.search(r'\b(all|every|none|never|always)\b.*?\b(all|every|none|never|always)\b', response, re.IGNORECASE):
            errors.append("Potential overgeneralization detected")

        return errors

    def _identify_knowledge_gaps(self, response: str) -> List[str]:
        """Identify knowledge gaps indicated in the response"""
        gaps = []

        gap_indicators = [
            r'\bi don\'t know\b',
            r'\bunknown\b',
            r'\bunclear\b',
            r'\bnot sure\b',
            r'\buncertain\b',
            r'\blimited information\b',
            r'\boutside my knowledge\b'
        ]

        for pattern in gap_indicators:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(response), match.end() + 30)
                context = response[start:end].strip()
                gaps.append(f"Knowledge gap: {context}")

        return gaps

    def _identify_assumptions(self, response: str) -> List[str]:
        """Identify assumptions made in the response"""
        assumptions = []

        assumption_indicators = [
            r'\bassum\w+\b',
            r'\bpresum\w+\b',
            r'\blikely\b',
            r'\bprobably\b',
            r'\btypically\b',
            r'\bgenerally\b',
            r'\busually\b'
        ]

        for pattern in assumption_indicators:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 40)
                end = min(len(response), match.end() + 40)
                context = response[start:end].strip()
                assumptions.append(f"Assumption: {context}")

        return assumptions


class MetacognitiveEngine(LoggerMixin):
    """
    Main metacognitive awareness engine that provides self-reflection,
    confidence assessment, and uncertainty quantification capabilities.
    """

    def __init__(self, config):
        """
        Initialize the metacognitive engine

        Args:
            config: Configuration object with metacognitive settings
        """
        self.config = config
        self.confidence_calculator = ConfidenceCalculator()
        self.uncertainty_analyzer = UncertaintyAnalyzer()
        self.self_evaluator = SelfEvaluationEngine()

        self.assessment_history = []
        self.confidence_threshold = config.confidence_threshold

        self.logger.info("MetacognitiveEngine initialized", extra={
            "confidence_threshold": self.confidence_threshold,
            "metacognition_enabled": config.enable_metacognition
        })

    def assess_response(self, response: str, query: str,
                       reasoning_trace: Optional[Dict] = None,
                       tool_results: Optional[List] = None,
                       context: Optional[Dict[str, Any]] = None) -> MetacognitiveAssessment:
        """
        Perform comprehensive metacognitive assessment of a response

        Args:
            response: The AI response to assess
            query: Original user query
            reasoning_trace: Optional reasoning information
            tool_results: Optional tool execution results
            context: Optional additional context

        Returns:
            MetacognitiveAssessment with comprehensive analysis
        """
        if not self.config.enable_metacognition:
            # Return basic assessment if metacognition is disabled
            return self._create_basic_assessment(response)

        start_time = time.time()

        # Calculate confidence score
        confidence_score, confidence_factors = self.confidence_calculator.calculate_confidence_score(
            response, reasoning_trace, tool_results
        )

        # Determine confidence level
        confidence_level = self._determine_confidence_level(confidence_score)

        # Identify uncertainty factors
        uncertainty_factors = self.uncertainty_analyzer.identify_uncertainty_factors(response)

        # Perform self-evaluation
        self_evaluation = self.self_evaluator.evaluate_response(response, query)

        # Generate confidence reasoning
        confidence_reasoning = self._generate_confidence_reasoning(
            confidence_score, confidence_factors, uncertainty_factors
        )

        # Identify knowledge boundaries
        knowledge_boundaries = self._identify_knowledge_boundaries(response, context)

        # Generate verification suggestions
        verification_suggestions = self._generate_verification_suggestions(
            uncertainty_factors, self_evaluation
        )

        # Generate alternative perspectives
        alternative_perspectives = self._generate_alternative_perspectives(response, query)

        # Create assessment
        assessment = MetacognitiveAssessment(
            overall_confidence=confidence_score,
            confidence_level=confidence_level,
            uncertainty_factors=uncertainty_factors,
            self_evaluation=self_evaluation,
            confidence_reasoning=confidence_reasoning,
            knowledge_boundaries=knowledge_boundaries,
            verification_suggestions=verification_suggestions,
            alternative_perspectives=alternative_perspectives,
            processing_notes={
                "assessment_time": time.time() - start_time,
                "response_length": len(response),
                "query_length": len(query),
                "confidence_factors": confidence_factors
            }
        )

        # Store in history
        self.assessment_history.append(assessment)

        self.logger.info("Metacognitive assessment completed", extra={
            "confidence_score": confidence_score,
            "confidence_level": confidence_level.value,
            "uncertainty_count": len(uncertainty_factors),
            "assessment_time": assessment.processing_notes["assessment_time"]
        })

        return assessment

    def _create_basic_assessment(self, response: str) -> MetacognitiveAssessment:
        """Create basic assessment when metacognition is disabled"""
        return MetacognitiveAssessment(
            overall_confidence=7.0,  # Default medium-high confidence
            confidence_level=ConfidenceLevel.HIGH,
            uncertainty_factors=[],
            self_evaluation=SelfEvaluation(
                response_completeness=7.0,
                factual_accuracy_confidence=7.0,
                logical_consistency=7.0,
                clarity_assessment=7.0,
                usefulness_score=7.0
            ),
            confidence_reasoning="Metacognitive assessment disabled",
            knowledge_boundaries=[],
            verification_suggestions=[],
            alternative_perspectives=[],
            processing_notes={"metacognition_disabled": True}
        )

    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Map confidence score to confidence level"""
        if confidence_score < 2.0:
            return ConfidenceLevel.VERY_LOW
        elif confidence_score < 4.0:
            return ConfidenceLevel.LOW
        elif confidence_score < 6.0:
            return ConfidenceLevel.MEDIUM
        elif confidence_score < 8.0:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH

    def _generate_confidence_reasoning(self, confidence_score: float,
                                     confidence_factors: Dict[str, float],
                                     uncertainty_factors: List[UncertaintyFactor]) -> str:
        """Generate explanation for confidence assessment"""
        reasoning_parts = []

        # Overall confidence assessment
        confidence_level = self._determine_confidence_level(confidence_score)
        reasoning_parts.append(f"Overall confidence is {confidence_level.value.replace('_', ' ')} ({confidence_score:.1f}/10).")

        # Key contributing factors
        top_factors = sorted(confidence_factors.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

        for factor_name, factor_score in top_factors:
            if abs(factor_score) > 0.1:
                impact = "increases" if factor_score > 0 else "decreases"
                reasoning_parts.append(f"{factor_name.replace('_', ' ').title()} {impact} confidence by {abs(factor_score):.1f} points.")

        # Uncertainty impact
        if uncertainty_factors:
            total_uncertainty_impact = sum(f.confidence_impact for f in uncertainty_factors)
            reasoning_parts.append(f"Identified {len(uncertainty_factors)} uncertainty factors reducing confidence by {total_uncertainty_impact:.1f} points.")

        # Threshold comparison
        if confidence_score < self.confidence_threshold:
            reasoning_parts.append(f"Confidence is below the threshold of {self.confidence_threshold:.1f}, suggesting additional verification may be needed.")

        return " ".join(reasoning_parts)

    def _identify_knowledge_boundaries(self, response: str, context: Optional[Dict[str, Any]]) -> List[str]:
        """Identify the boundaries of knowledge demonstrated in the response"""
        boundaries = []

        # Domain boundaries
        if context and context.get("query_intent"):
            intent = context["query_intent"]
            if intent.get("primary_intent") == "data_analysis":
                boundaries.append("Analysis limited to provided data sources")
            elif intent.get("primary_intent") == "schema_exploration":
                boundaries.append("Knowledge limited to discoverable schema information")

        # Temporal boundaries
        if re.search(r'\b(recent|current|latest|up to date)\b', response, re.IGNORECASE):
            boundaries.append("Information currency may be limited by training data cutoff")

        # Scope boundaries
        if re.search(r'\b(general|typical|standard|common)\b', response, re.IGNORECASE):
            boundaries.append("Response focuses on general cases, specific situations may vary")

        # Technical boundaries
        if re.search(r'\b(technical|complex|advanced|specialized)\b', response, re.IGNORECASE):
            boundaries.append("Technical details may require domain expertise for full understanding")

        return boundaries

    def _generate_verification_suggestions(self, uncertainty_factors: List[UncertaintyFactor],
                                         self_evaluation: SelfEvaluation) -> List[str]:
        """Generate suggestions for verifying the response"""
        suggestions = []

        # Based on uncertainty factors
        for factor in uncertainty_factors:
            suggestions.extend(factor.resolution_suggestions)

        # Based on self-evaluation
        if self_evaluation.factual_accuracy_confidence < 7.0:
            suggestions.append("Cross-reference factual claims with authoritative sources")

        if self_evaluation.response_completeness < 7.0:
            suggestions.append("Consider seeking additional information to complete the analysis")

        if self_evaluation.knowledge_gaps:
            suggestions.append("Address identified knowledge gaps through additional research")

        if self_evaluation.potential_errors:
            suggestions.append("Double-check calculations and logical reasoning")

        # Remove duplicates and limit
        unique_suggestions = list(set(suggestions))
        return unique_suggestions[:5]  # Top 5 suggestions

    def _generate_alternative_perspectives(self, response: str, query: str) -> List[str]:
        """Generate alternative perspectives or approaches"""
        perspectives = []

        # Different analytical approaches
        if "analysis" in query.lower():
            perspectives.append("Consider alternative analytical frameworks or methodologies")

        # Different stakeholder viewpoints
        if any(word in query.lower() for word in ["business", "organization", "company"]):
            perspectives.append("Examine from different stakeholder perspectives (customers, employees, shareholders)")

        # Different time horizons
        if any(word in query.lower() for word in ["future", "planning", "strategy"]):
            perspectives.append("Consider both short-term and long-term implications")

        # Different scale considerations
        if any(word in query.lower() for word in ["system", "process", "implementation"]):
            perspectives.append("Evaluate at different scales (individual, team, organizational, industry)")

        # Conservative vs. aggressive approaches
        if any(word in query.lower() for word in ["recommend", "suggest", "should"]):
            perspectives.append("Compare conservative vs. aggressive implementation approaches")

        return perspectives[:3]  # Top 3 alternative perspectives

    def get_confidence_summary(self) -> Dict[str, Any]:
        """Get summary of confidence assessments over time"""
        if not self.assessment_history:
            return {"message": "No assessments performed yet"}

        recent_assessments = self.assessment_history[-20:]  # Last 20 assessments

        confidence_scores = [a.overall_confidence for a in recent_assessments]

        summary = {
            "total_assessments": len(self.assessment_history),
            "recent_average_confidence": statistics.mean(confidence_scores),
            "confidence_trend": "increasing" if len(confidence_scores) > 1 and confidence_scores[-1] > confidence_scores[0] else "stable",
            "low_confidence_count": len([s for s in confidence_scores if s < self.confidence_threshold]),
            "confidence_distribution": {
                level.value: len([a for a in recent_assessments if a.confidence_level == level])
                for level in ConfidenceLevel
            },
            "common_uncertainty_types": self._get_common_uncertainty_types(recent_assessments)
        }

        return summary

    def _get_common_uncertainty_types(self, assessments: List[MetacognitiveAssessment]) -> Dict[str, int]:
        """Get the most common types of uncertainty"""
        uncertainty_counts = defaultdict(int)

        for assessment in assessments:
            for factor in assessment.uncertainty_factors:
                uncertainty_counts[factor.uncertainty_type.value] += 1

        return dict(sorted(uncertainty_counts.items(), key=lambda x: x[1], reverse=True))

    def clear_history(self) -> None:
        """Clear assessment history"""
        self.assessment_history.clear()
        self.logger.info("Metacognitive assessment history cleared")